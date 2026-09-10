# signal_generator.py
# Converts ensemble return predictions into actionable trading signals
# and writes them to DuckDB for the dashboard to read.
#
# Signal logic
# ─────────────
# 1. Predict next-day log return r̂ for each Nifty50 ticker
# 2. Normalise by recent realised volatility:  z = r̂ / vol_20d
# 3. Signal:  z > +threshold → LONG
#             z < -threshold → SHORT
#             else           → FLAT
# 4. Position size = min(|z| / 2, MAX_POSITION)
#    (larger z-score = larger conviction = larger position)
# 5. Top-N longs and top-N shorts selected per day
#
# DuckDB table: predictions
#   (ticker, date, pred_ret, signal, position_size, z_score,
#    pred_close, vol_20d, computed_at)
#
# Usage
#   python signal_generator.py                   # generate for today
#   python signal_generator.py --date 2024-03-18

from __future__ import annotations

import argparse
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DB_PATH, CHECKPOINTS_DIR, SCALERS_DIR, SCALER_NAME,
    FEATURE_COLS, SEQ_LEN,
    ENSEMBLE_W_LSTM, ENSEMBLE_W_TF,
    SIGNAL_Z_THRESHOLD, SIGNAL_TARGET_VOL,
    SIGNAL_MAX_POSITION, SIGNAL_TOP_N,
)
from models import build_lstm, build_transformer
from trainer import load_checkpoint, get_device


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _ensure_predictions_table(db_path: Path) -> None:
    """Create the predictions table if it doesn't exist."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                ticker          VARCHAR NOT NULL,
                date            DATE    NOT NULL,
                pred_ret        DOUBLE,
                signal          VARCHAR,
                position_size   DOUBLE,
                z_score         DOUBLE,
                pred_close      DOUBLE,
                vol_20d         DOUBLE,
                computed_at     TIMESTAMPTZ,
                PRIMARY KEY (ticker, date)
            )
        """)
    finally:
        con.close()


def _load_latest_sequences(target_date: str, db_path: Path) -> pd.DataFrame:
    """
    For each ticker, load the last SEQ_LEN rows of features up to target_date.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(f"""
            SELECT ticker, date, {", ".join(FEATURE_COLS)}
            FROM   feature_matrix
            WHERE  date <= ?
            ORDER  BY ticker, date
        """, [target_date]).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_close_and_vol(target_date: str, db_path: Path) -> pd.DataFrame:
    """Load today's close price and vol_20d for all tickers."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT o.ticker, o.close AS last_close,
                   t.vol_20d
            FROM   ohlcv o
            JOIN   technical_features t
                   ON o.ticker = t.ticker AND o.date = t.date
            WHERE  o.date = (
                SELECT MAX(date) FROM ohlcv WHERE date <= ?
            )
        """, [target_date]).df()
    finally:
        con.close()
    return df


def _write_predictions(df: pd.DataFrame, db_path: Path) -> int:
    """Bulk-insert / replace predictions for this date."""
    con = duckdb.connect(str(db_path))
    try:
        con.register("_batch_pred", df)
        result = con.execute("""
            INSERT OR REPLACE INTO predictions
            SELECT * FROM _batch_pred
        """).fetchone()
        return result[0] if result else 0
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_returns(
    target_date: str,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Run the ensemble model on the latest features for every ticker.

    Returns a DataFrame with columns:
        ticker, pred_ret_lstm, pred_ret_tf, pred_ret_ensemble
    """
    device = get_device()

    # Load models
    lstm = build_lstm().to(device)
    tf   = build_transformer().to(device)
    try:
        load_checkpoint(lstm, "lstm_best",        device=device)
        load_checkpoint(tf,   "transformer_best", device=device)
    except FileNotFoundError as e:
        raise RuntimeError(f"Trained models not found: {e}\nRun train_lstm.py and train_transformer.py first.")

    lstm.eval()
    tf.eval()

    # Load scaler
    scaler_path = SCALERS_DIR / SCALER_NAME
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load and prepare sequences
    raw_df = _load_latest_sequences(target_date, db_path)
    if raw_df.empty:
        raise RuntimeError(f"No features found in feature_matrix up to {target_date}.")

    # Clean NaN
    raw_df[FEATURE_COLS] = (
        raw_df.groupby("ticker")[FEATURE_COLS]
        .transform(lambda s: s.ffill().bfill().fillna(0.0))
    )

    records = []
    with torch.no_grad():
        for ticker, grp in raw_df.groupby("ticker"):
            grp = grp.sort_values("date").reset_index(drop=True)
            if len(grp) < SEQ_LEN:
                continue   # not enough history

            # Take the last SEQ_LEN rows
            window = grp.tail(SEQ_LEN)[FEATURE_COLS].values.astype(np.float32)
            window = scaler.transform(window)                    # (SEQ_LEN, N_FEATURES)
            x      = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

            r_lstm = float(lstm(x).cpu().squeeze())
            r_tf   = float(tf(x).cpu().squeeze())
            r_ens  = ENSEMBLE_W_LSTM * r_lstm + ENSEMBLE_W_TF * r_tf

            records.append({
                "ticker":            ticker,
                "pred_ret_lstm":     r_lstm,
                "pred_ret_tf":       r_tf,
                "pred_ret_ensemble": r_ens,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    target_date: str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Full pipeline: predict → z-score → signal → size → write to DB.
    Returns the signal DataFrame.
    """
    if target_date is None:
        target_date = str(date.today())

    _ensure_predictions_table(db_path)

    print(f"[signal] Generating signals for {target_date} ...")

    # Step 1 — model predictions
    pred_df = predict_returns(target_date, db_path)
    if pred_df.empty:
        print("[signal] No predictions generated — check feature_matrix coverage.")
        return pd.DataFrame()

    # Step 2 — load close price and volatility
    price_vol = _load_close_and_vol(target_date, db_path)
    pred_df   = pred_df.merge(price_vol, on="ticker", how="left")

    # Step 3 — z-score normalisation
    pred_df["vol_20d"] = pred_df["vol_20d"].fillna(0.15)   # fallback 15% vol
    pred_df["z_score"] = (
        pred_df["pred_ret_ensemble"] / pred_df["vol_20d"].clip(lower=0.005)
    )

    # Step 4 — signal classification
    def _classify(row):
        z = row["z_score"]
        if z > SIGNAL_Z_THRESHOLD:
            sig  = "LONG"
            size = min(abs(z) / 2, SIGNAL_MAX_POSITION)
        elif z < -SIGNAL_Z_THRESHOLD:
            sig  = "SHORT"
            size = min(abs(z) / 2, SIGNAL_MAX_POSITION)
        else:
            sig  = "FLAT"
            size = 0.0
        return sig, round(size, 4)

    pred_df[["signal", "position_size"]] = pred_df.apply(
        lambda r: pd.Series(_classify(r)), axis=1
    )

    # Step 5 — keep only top-N longs and top-N shorts
    longs  = pred_df[pred_df.signal == "LONG" ].nlargest(SIGNAL_TOP_N,  "z_score")
    shorts = pred_df[pred_df.signal == "SHORT"].nsmallest(SIGNAL_TOP_N, "z_score")
    flats  = pred_df[pred_df.signal == "FLAT" ]
    pred_df = pd.concat([longs, shorts, flats]).sort_values("ticker").reset_index(drop=True)

    # Step 6 — predicted close price
    pred_df["pred_close"] = pred_df["last_close"] * np.exp(pred_df["pred_ret_ensemble"])
    pred_df["pred_close"] = pred_df["pred_close"].round(2)

    # Step 7 — assemble final output table
    now = datetime.now(timezone.utc)
    out = pd.DataFrame({
        "ticker":        pred_df["ticker"],
        "date":          target_date,
        "pred_ret":      pred_df["pred_ret_ensemble"].round(8),
        "signal":        pred_df["signal"],
        "position_size": pred_df["position_size"],
        "z_score":       pred_df["z_score"].round(4),
        "pred_close":    pred_df["pred_close"],
        "vol_20d":       pred_df["vol_20d"].round(6),
        "computed_at":   now,
    })

    # Step 8 — write to DuckDB
    inserted = _write_predictions(out, db_path)
    print(f"[signal] Written {inserted} signal rows to DB.")

    # Print summary
    active = out[out.signal != "FLAT"]
    print(f"\n[signal] Active signals: {len(active)}")
    print(f"  LONG : {(out.signal=='LONG').sum()}")
    print(f"  SHORT: {(out.signal=='SHORT').sum()}")
    print(f"  FLAT : {(out.signal=='FLAT').sum()}")
    if not active.empty:
        print("\nTop signals:")
        print(
            active[["ticker","signal","pred_ret","z_score","position_size","pred_close"]]
            .sort_values("z_score", key=abs, ascending=False)
            .head(10)
            .to_string(index=False)
        )

    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate daily trading signals")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--db",   default=str(DB_PATH))
    # --- FIX: Add backfill arguments ---
    parser.add_argument("--backfill", action="store_true", help="Backfill historical signals")
    parser.add_argument("--days",     type=int, default=30, help="Days to backfill (default 30)")
    # -----------------------------------
    args = parser.parse_args()

    if args.backfill:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)
        
        print(f"Backfilling signals from {start_date} to {end_date}...")
        for i in range((end_date - start_date).days + 1):
            current_date = start_date + timedelta(days=i)
            generate_signals(target_date=str(current_date), db_path=Path(args.db))
    else:
        generate_signals(
            target_date=args.date,
            db_path=Path(args.db),
        )
