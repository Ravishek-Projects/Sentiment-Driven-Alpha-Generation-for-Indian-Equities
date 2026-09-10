# incremental_retrain.py
# Warm-start incremental retraining on the most recent N days of data.
# Called automatically by model_monitor.py when alerts fire,
# or manually from the command line.
#
# Key difference from Phase 3 full training
# ──────────────────────────────────────────
# Full training: random weight initialisation, 100 epochs.
# Incremental:   load existing checkpoint, fine-tune for 30 epochs with
#                10× smaller learning rate. Preserves learned representations
#                while adapting to recent regime changes.
#
# After retraining:
# - Old checkpoint archived as lstm_best_YYYYMMDD.pt
# - New checkpoint saved as lstm_best.pt (replaces the live model)
# - signal_generator.py automatically uses the new model on next run
#
# Usage
#   python incremental_retrain.py
#   python incremental_retrain.py --days 180     # use last 180 days
#   python incremental_retrain.py --epochs 50    # override epoch count
#   python incremental_retrain.py --dry-run      # check data but don't train

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config_phase5 import (
    DB_PATH, CHECKPOINTS_DIR, SCALERS_DIR,
    RETRAIN_LOOKBACK_DAYS, RETRAIN_EPOCHS,
    RETRAIN_PATIENCE, RETRAIN_LR_FACTOR,
)
from config import FEATURE_COLS, TARGET_COL, SEQ_LEN, LSTM_LR, TF_LR
from models import build_lstm, build_transformer
from trainer import load_checkpoint, save_checkpoint, get_device, MixedLoss, evaluate
from dataset import StockSequenceDataset, clean_features, chronological_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retrain")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_recent_features(days: int, db_path: Path) -> pd.DataFrame:
    """Load the last `days` days of labelled feature_matrix rows."""
    cutoff = (date.today() - __import__("datetime").timedelta(days=days)).isoformat()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(f"""
            SELECT ticker, date, {", ".join(FEATURE_COLS)}, {TARGET_COL}
            FROM   feature_matrix
            WHERE  {TARGET_COL} IS NOT NULL
              AND  date >= ?
            ORDER  BY date, ticker
        """, [cutoff]).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Archive helper
# ---------------------------------------------------------------------------

def _archive_checkpoint(name: str) -> Path | None:
    src = CHECKPOINTS_DIR / f"{name}_best.pt"
    if not src.exists():
        return None
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = CHECKPOINTS_DIR / f"{name}_best_{ts}.pt"
    shutil.copy2(src, dst)
    log.info(f"[retrain] Archived {src.name} → {dst.name}")
    return dst


# ---------------------------------------------------------------------------
# Training loop (simplified — no LR scheduler, short run)
# ---------------------------------------------------------------------------

def _retrain_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    epochs: int,
    patience: int,
    name: str,
    device: torch.device,
) -> float:
    """Fine-tune model and save new best checkpoint. Returns best val loss."""
    model.to(device)
    model.train()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    crit  = MixedLoss().to(device)

    best_val  = float("inf")
    no_improv = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for xb, yb, _, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb), yb).item() * len(xb)
        val_loss /= max(len(val_loader.dataset), 1)

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"  [{name}] Epoch {epoch:3d}/{epochs}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val  = val_loss
            no_improv = 0
            save_checkpoint(model, opt, epoch, val_loss, f"{name}_best")
        else:
            no_improv += 1
            if no_improv >= patience:
                log.info(f"  [{name}] Early stop at epoch {epoch}")
                break

    return best_val


# ---------------------------------------------------------------------------
# Main retrain function
# ---------------------------------------------------------------------------

def run_retrain(
    days:    int = RETRAIN_LOOKBACK_DAYS,
    epochs:  int = RETRAIN_EPOCHS,
    db_path: Path = DB_PATH,
    dry_run: bool = False,
) -> dict:
    """
    Incrementally retrain LSTM and Transformer on the last `days` of data.
    Archives old checkpoints and replaces with new ones.

    Returns a summary dict.
    """
    log.info(f"[retrain] Incremental retrain — last {days} days, {epochs} epochs")

    device = get_device()

    # ── Load data ────────────────────────────────────────────────────────
    df = _load_recent_features(days, db_path)
    if df.empty:
        log.error("[retrain] No data available. Aborting.")
        return {"error": "no data"}

    df = clean_features(df, FEATURE_COLS)
    log.info(f"[retrain] {len(df)} rows, {df['ticker'].nunique()} tickers")

    if len(df) < 200:
        log.warning(f"[retrain] Only {len(df)} rows — may not be enough for meaningful update.")

    if dry_run:
        log.info("[retrain] DRY RUN — not training.")
        return {"dry_run": True, "rows": len(df), "days": days}

    # ── Load scaler ──────────────────────────────────────────────────────
    scaler_path = SCALERS_DIR / "feature_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Scale features
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS].values)

    # Split
    # Split
    train_df, val_df, _ = chronological_split(df)

    # --- FIX: Add sequence warm-up buffer to Validation set ---
    train_dates = sorted(train_df["date"].unique())
    warmup_dates = train_dates[-SEQ_LEN:]
    warmup_df = train_df[train_df["date"].isin(warmup_dates)].copy()
    val_df = pd.concat([warmup_df, val_df]).sort_values(["ticker", "date"])
    # ----------------------------------------------------------

    # Build datasets (bypass the scaler-fitting logic)
    def _make_ds(split_df):
        ds         = StockSequenceDataset.__new__(StockSequenceDataset)
        ds.seq_len = SEQ_LEN
        ds.scaler  = None
        ds.samples = []
        ds._build  = StockSequenceDataset._build.__get__(ds)
        ds._build(split_df)
        return ds

    train_ds = _make_ds(train_df)
    val_ds   = _make_ds(val_df)

    if len(train_ds) < 50 or len(val_ds) < 10:
        log.error(f"[retrain] Too few samples: train={len(train_ds)}, val={len(val_ds)}")
        return {"error": "insufficient samples after split"}

    tr_loader  = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=64, shuffle=False)
    log.info(f"[retrain] train samples={len(train_ds)}  val samples={len(val_ds)}")

    results = {}

    # ── LSTM ─────────────────────────────────────────────────────────────
    log.info("[retrain] Fine-tuning LSTM ...")
    lstm = build_lstm().to(device)
    try:
        load_checkpoint(lstm, "lstm_best", device=device)
        log.info("[retrain] Loaded existing lstm_best checkpoint")
    except FileNotFoundError:
        log.warning("[retrain] No existing lstm_best — training from scratch")

    _archive_checkpoint("lstm")
    best_lstm = _retrain_model(
        lstm, tr_loader, val_loader,
        lr=LSTM_LR * RETRAIN_LR_FACTOR,
        epochs=epochs, patience=RETRAIN_PATIENCE,
        name="lstm", device=device,
    )
    results["lstm_val_loss"] = round(best_lstm, 8)

    # ── Transformer ───────────────────────────────────────────────────────
    log.info("[retrain] Fine-tuning Transformer ...")
    tf = build_transformer().to(device)
    try:
        load_checkpoint(tf, "transformer_best", device=device)
        log.info("[retrain] Loaded existing transformer_best checkpoint")
    except FileNotFoundError:
        log.warning("[retrain] No existing transformer_best — training from scratch")

    _archive_checkpoint("transformer")
    best_tf = _retrain_model(
        tf, tr_loader, val_loader,
        lr=TF_LR * RETRAIN_LR_FACTOR,
        epochs=epochs, patience=RETRAIN_PATIENCE,
        name="transformer", device=device,
    )
    results["transformer_val_loss"] = round(best_tf, 8)

    # ── Quick test evaluation ─────────────────────────────────────────────
    test_df  = df  # use all recent data as a quick proxy
    test_ds  = _make_ds(test_df)
    te_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    lstm.eval(); tf.eval()
    p_lstm, actuals, _, _ = evaluate(lstm, te_loader, device)
    p_tf,   _,       _, _ = evaluate(tf,   te_loader, device)
    p_ens = [0.4 * l + 0.6 * t for l, t in zip(p_lstm, p_tf)]

    a = np.array(actuals)
    results["hit_rate_ensemble"] = round(
        float(np.mean(np.sign(np.array(p_ens)) == np.sign(a))), 4
    )

    # Save retrain log
    now = datetime.now(timezone.utc)
    log_entry = {
        "timestamp": now.isoformat(),
        "days_used": days,
        "epochs":    epochs,
        "train_samples": len(train_ds),
        **results,
    }
    log_path = CHECKPOINTS_DIR / "retrain_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    log.info(f"[retrain] Complete — ensemble hit rate: {results.get('hit_rate_ensemble')}")
    log.info(f"[retrain] Log appended → {log_path}")
    return log_entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental model retraining")
    parser.add_argument("--db",      default=str(DB_PATH))
    parser.add_argument("--days",    type=int, default=RETRAIN_LOOKBACK_DAYS)
    parser.add_argument("--epochs",  type=int, default=RETRAIN_EPOCHS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_retrain(
        days    = args.days,
        epochs  = args.epochs,
        db_path = Path(args.db),
        dry_run = args.dry_run,
    )
