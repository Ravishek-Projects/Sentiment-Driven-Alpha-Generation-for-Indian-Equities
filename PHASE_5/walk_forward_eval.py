# walk_forward_eval.py
# Expanding-window walk-forward cross-validation for the ensemble model.
#
# Why walk-forward?
# ─────────────────
# A single train/test split is unreliable for financial time series —
# the chosen split date dominates the result. Walk-forward evaluation
# uses every possible split date and averages the out-of-sample metrics,
# giving a much more honest estimate of live performance.
#
# Method
# ──────
# Fold 1:  train on days [0 : 180],  test on days [180 : 201]
# Fold 2:  train on days [0 : 201],  test on days [201 : 222]
# Fold 3:  train on days [0 : 222],  test on days [222 : 243]
# ...
# (expanding window — we never shrink the training set)
#
# For each fold the LSTM and Transformer are fine-tuned from the previous
# checkpoint (warm start) for RETRAIN_EPOCHS, keeping the same scaler.
# The ensemble predictions on the test window are recorded.
#
# Usage (Windows Terminal, from phase5\ directory)
#   python walk_forward_eval.py
#   python walk_forward_eval.py --quick     # 5-fold subset for testing
#   python walk_forward_eval.py --no-retrain # use fixed trained model for all folds

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "PHASE_3"))

from config_phase5 import (
    DB_PATH, CHECKPOINTS_DIR, SCALERS_DIR, EVAL_DIR,
    WF_INITIAL_TRAIN_DAYS, WF_TEST_WINDOW_DAYS,
    WF_STEP_DAYS, WF_MIN_TEST_SAMPLES,
    RETRAIN_EPOCHS, RETRAIN_PATIENCE, RETRAIN_LR_FACTOR,
)
from config import FEATURE_COLS, TARGET_COL, SEQ_LEN, LSTM_LR, TF_LR
from models import build_lstm, build_transformer
from trainer import load_checkpoint, get_device, MixedLoss
from dataset import StockSequenceDataset, clean_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("walk_forward")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(f"""
            SELECT ticker, date, {", ".join(FEATURE_COLS)}, {TARGET_COL}
            FROM   feature_matrix
            WHERE  {TARGET_COL} IS NOT NULL
            ORDER  BY date, ticker
        """).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _metrics(preds, actuals) -> dict:
    p, a = np.array(preds), np.array(actuals)
    mask = ~(np.isnan(p) | np.isnan(a))
    p, a = p[mask], a[mask]
    if len(p) < 2:
        return {"mae": np.nan, "rmse": np.nan, "hit_rate": np.nan, "ic": np.nan}
    mae      = float(np.mean(np.abs(p - a)))
    rmse     = float(np.sqrt(np.mean((p - a) ** 2)))
    hit_rate = float(np.mean(np.sign(p) == np.sign(a)))
    ic       = float(np.corrcoef(p, a)[0, 1])
    return {"mae": mae, "rmse": rmse, "hit_rate": hit_rate, "ic": ic}


# ---------------------------------------------------------------------------
# Per-fold warm-start retraining
# ---------------------------------------------------------------------------

def _fine_tune(model: nn.Module, loader: DataLoader, epochs: int,
               lr: float, device: torch.device) -> None:
    """Fine-tune model for a small number of epochs on new data."""
    model.train()
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = MixedLoss().to(device)
    for _ in range(epochs):
        for xb, yb, _, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    db_path: Path = DB_PATH,
    no_retrain: bool = False,
    max_folds: int | None = None,
) -> pd.DataFrame:
    """
    Runs expanding-window walk-forward evaluation.

    Returns a DataFrame with one row per fold containing:
        fold, train_end, test_start, test_end,
        n_train_samples, n_test_samples,
        mae_lstm, rmse_lstm, hit_rate_lstm, ic_lstm,
        mae_tf,   rmse_tf,   hit_rate_tf,   ic_tf,
        mae_ens,  rmse_ens,  hit_rate_ens,  ic_ens
    """
    device = get_device()
    log.info(f"[wf] Walk-forward evaluation on {device}")

    # Load everything once
    df_full = _load_all(db_path)
    if df_full.empty:
        raise RuntimeError("feature_matrix is empty. Complete Phase 2 backfill first.")

    df_full = clean_features(df_full, FEATURE_COLS)

    # Load scaler
    scaler_path = SCALERS_DIR / "feature_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}. Run Phase 3 training first.")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Sort unique dates
    all_dates = sorted(df_full["date"].unique())
    n_dates   = len(all_dates)
    log.info(f"[wf] {n_dates} unique trading dates, {df_full['ticker'].nunique()} tickers")

    if n_dates < WF_INITIAL_TRAIN_DAYS + WF_TEST_WINDOW_DAYS:
        raise RuntimeError(
            f"Need at least {WF_INITIAL_TRAIN_DAYS + WF_TEST_WINDOW_DAYS} dates "
            f"but only have {n_dates}. Run longer Phase 2 backfill."
        )

    # Load base models
    lstm = build_lstm().to(device)
    tf   = build_transformer().to(device)
    try:
        load_checkpoint(lstm, "lstm_best",        device=device)
        load_checkpoint(tf,   "transformer_best", device=device)
    except FileNotFoundError as e:
        raise RuntimeError(f"Trained models not found: {e}. Run Phase 3 training first.")

    fold_results = []
    fold_num     = 0
    train_end_idx = WF_INITIAL_TRAIN_DAYS

    while train_end_idx + WF_TEST_WINDOW_DAYS <= n_dates:
        if max_folds and fold_num >= max_folds:
            break

        fold_num    += 1
        train_end_dt = all_dates[train_end_idx - 1]
        test_start_dt= all_dates[train_end_idx]
        test_end_idx = min(train_end_idx + WF_TEST_WINDOW_DAYS, n_dates) - 1
        test_end_dt  = all_dates[test_end_idx]

        # --- FIX: Include a warm-up period so the LSTM can form sequences ---
        warmup_start_dt = all_dates[max(0, train_end_idx - SEQ_LEN)]
        
        train_df = df_full[df_full["date"] <= train_end_dt].copy()
        test_df  = df_full[
            (df_full["date"] >= warmup_start_dt) & (df_full["date"] <= test_end_dt)
        ].copy()
        # --------------------------------------------------------------------

        if len(test_df) < WF_MIN_TEST_SAMPLES:
            train_end_idx += WF_STEP_DAYS
            continue

        log.info(
            f"[wf] Fold {fold_num:3d} | "
            f"train ≤ {train_end_dt.date()} ({len(train_df)} rows) | "
            f"test {test_start_dt.date()} → {test_end_dt.date()} ({len(test_df)} rows)"
        )

        # Scale test features with the FIXED scaler (fit on all training data once)
        train_scaled = train_df.copy()
        train_scaled[FEATURE_COLS] = scaler.transform(train_df[FEATURE_COLS].values)
        test_scaled  = test_df.copy()
        test_scaled[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS].values)

        # Build datasets
        train_ds = StockSequenceDataset.__new__(StockSequenceDataset)
        train_ds.seq_len = SEQ_LEN
        train_ds.scaler  = None       # already scaled
        train_ds.samples = []
        train_ds._build  = StockSequenceDataset._build.__get__(train_ds)
        train_ds._build(train_scaled)

        test_ds = StockSequenceDataset.__new__(StockSequenceDataset)
        test_ds.seq_len = SEQ_LEN
        test_ds.scaler  = None
        test_ds.samples = []
        test_ds._build  = StockSequenceDataset._build.__get__(test_ds)
        test_ds._build(test_scaled)

        if len(test_ds) < WF_MIN_TEST_SAMPLES:
            train_end_idx += WF_STEP_DAYS
            continue

        tr_loader   = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds,  batch_size=64, shuffle=False)

        # Fine-tune if requested
        if not no_retrain and fold_num > 1 and len(tr_loader) > 0:
            _fine_tune(lstm, tr_loader, RETRAIN_EPOCHS,
                       LSTM_LR * RETRAIN_LR_FACTOR, device)
            _fine_tune(tf,   tr_loader, RETRAIN_EPOCHS,
                       TF_LR  * RETRAIN_LR_FACTOR, device)

        # Evaluate on test fold
        lstm.eval(); tf.eval()
        p_lstm, p_tf, actuals = [], [], []
        with torch.no_grad():
            for xb, yb, _, _ in test_loader:
                xb = xb.to(device)
                p_lstm.extend(lstm(xb).cpu().squeeze(-1).tolist())
                p_tf.extend(  tf(xb).cpu().squeeze(-1).tolist())
                actuals.extend(yb.squeeze(-1).tolist())

        p_ens = [0.4 * l + 0.6 * t for l, t in zip(p_lstm, p_tf)]

        row = {
            "fold":             fold_num,
            "train_end":        str(train_end_dt.date()),
            "test_start":       str(test_start_dt.date()),
            "test_end":         str(test_end_dt.date()),
            "n_train_samples":  len(train_ds),
            "n_test_samples":   len(test_ds),
            **{f"{k}_lstm": v for k, v in _metrics(p_lstm, actuals).items()},
            **{f"{k}_tf":   v for k, v in _metrics(p_tf,   actuals).items()},
            **{f"{k}_ens":  v for k, v in _metrics(p_ens,  actuals).items()},
        }
        fold_results.append(row)
        train_end_idx += WF_STEP_DAYS

    results_df = pd.DataFrame(fold_results)

    # Save to CSV
    out_path = EVAL_DIR / "walk_forward_results.csv"
    results_df.to_csv(out_path, index=False)
    log.info(f"[wf] {len(results_df)} folds complete → {out_path}")

    # Print summary
    if not results_df.empty:
        log.info("\n── Walk-forward summary (mean across folds) ─────────────────")
        for col in ["hit_rate_lstm", "hit_rate_tf", "hit_rate_ens",
                    "ic_lstm", "ic_tf", "ic_ens",
                    "rmse_lstm", "rmse_tf", "rmse_ens"]:
            if col in results_df:
                val = results_df[col].dropna().mean()
                log.info(f"  {col:<22}: {val:.4f}")

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward cross-validation")
    parser.add_argument("--db",         default=str(DB_PATH))
    parser.add_argument("--no-retrain", action="store_true",
                        help="Use fixed model across all folds (faster)")
    parser.add_argument("--quick",      action="store_true",
                        help="Run only 5 folds (for testing)")
    args = parser.parse_args()

    results = run_walk_forward(
        db_path    = Path(args.db),
        no_retrain = args.no_retrain,
        max_folds  = 5 if args.quick else None,
    )
    print(f"\nResults saved to {EVAL_DIR / 'walk_forward_results.csv'}")
