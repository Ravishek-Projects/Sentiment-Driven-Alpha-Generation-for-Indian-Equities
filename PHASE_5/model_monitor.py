# model_monitor.py
# Monitors model health on a rolling basis and triggers retraining
# alerts when performance degrades.
#
# Checks performed daily (called by scheduler_phase5.py)
# ──────────────────────────────────────────────────────
# 1. Rolling 30-day Sharpe — if < 0.5 → alert
# 2. Rolling 30-day hit rate — if < 48% → alert
# 3. Rolling 30-day IC  — if < 0.02 → alert
# 4. Sentiment drift (KL divergence) — if > 0.15 → alert
# 5. Prediction bias — is model consistently predicting wrong sign?
#
# Alert actions
# ─────────────
# - Write alert to monitor\alerts.jsonl (one JSON object per line)
# - Log warning to logs\monitor.log
# - If RETRAIN_ON_ALERT=True → trigger incremental retraining
#
# Usage
#   python model_monitor.py
#   python model_monitor.py --retrain-on-alert
#   python model_monitor.py --date 2024-03-28

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import rel_entr
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config_phase5 import (
    DB_PATH, MONITOR_DIR, LOGS_DIR,
    MONITOR_WINDOW_DAYS,
    SHARPE_RETRAIN_THRESH, HIT_RATE_RETRAIN_THRESH,
    IC_RETRAIN_THRESH, SENTIMENT_DRIFT_THRESH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOGS_DIR / "monitor.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("monitor")

ALERTS_FILE = MONITOR_DIR / "alerts.jsonl"
HEALTH_FILE = MONITOR_DIR / "health_history.csv"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_recent_snapshots(window: int, db_path: Path) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=window)).isoformat()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT snapshot_date, portfolio_value, daily_return,
                   daily_pnl, open_positions
            FROM   portfolio_snapshots
            WHERE  snapshot_date >= ?
            ORDER  BY snapshot_date
        """, [cutoff]).df()
    finally:
        con.close()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _load_recent_trades(window: int, db_path: Path) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=window)).isoformat()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, signal_date, direction, pred_ret, actual_ret, pnl_net
            FROM   trades
            WHERE  status      = 'closed'
              AND  signal_date >= ?
              AND  pnl_net     IS NOT NULL
            ORDER  BY signal_date
        """, [cutoff]).df()
    finally:
        con.close()
    return df


def _load_recent_sentiment(window: int, db_path: Path) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=window * 2)).isoformat()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT trading_date, sent_pos, sent_neg, sent_spread
            FROM   sentiment_scores
            WHERE  trading_date >= ?
            ORDER  BY trading_date
        """, [cutoff]).df()
    finally:
        con.close()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df


# ---------------------------------------------------------------------------
# Individual health checks
# ---------------------------------------------------------------------------

def check_rolling_sharpe(snap_df: pd.DataFrame) -> dict:
    r = snap_df["daily_return"].dropna().values
    if len(r) < 5:
        return {"status": "insufficient_data", "value": None, "threshold": SHARPE_RETRAIN_THRESH}
    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    sharpe  = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    status  = "ok" if sharpe >= SHARPE_RETRAIN_THRESH else "alert"
    return {
        "metric":    "rolling_sharpe",
        "value":     round(sharpe, 4),
        "threshold": SHARPE_RETRAIN_THRESH,
        "status":    status,
        "message":   f"Rolling {len(r)}-day Sharpe = {sharpe:.4f} (threshold {SHARPE_RETRAIN_THRESH})",
    }


def check_hit_rate(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty or len(trades_df) < 5:
        return {"status": "insufficient_data", "value": None, "threshold": HIT_RATE_RETRAIN_THRESH}
    hr = float((trades_df["pnl_net"] > 0).mean())
    status = "ok" if hr >= HIT_RATE_RETRAIN_THRESH else "alert"
    return {
        "metric":    "hit_rate",
        "value":     round(hr, 4),
        "threshold": HIT_RATE_RETRAIN_THRESH,
        "status":    status,
        "n_trades":  len(trades_df),
        "message":   f"Rolling hit rate = {hr*100:.1f}% over {len(trades_df)} trades (threshold {HIT_RATE_RETRAIN_THRESH*100:.0f}%)",
    }


def check_ic(trades_df: pd.DataFrame) -> dict:
    valid = trades_df.dropna(subset=["pred_ret", "actual_ret"])
    if len(valid) < 5:
        return {"status": "insufficient_data", "value": None, "threshold": IC_RETRAIN_THRESH}
    ic = float(np.corrcoef(valid["pred_ret"].values, valid["actual_ret"].values)[0, 1])
    status = "ok" if ic >= IC_RETRAIN_THRESH else "alert"
    return {
        "metric":    "information_coefficient",
        "value":     round(ic, 4),
        "threshold": IC_RETRAIN_THRESH,
        "status":    status,
        "message":   f"Rolling IC = {ic:.4f} (threshold {IC_RETRAIN_THRESH})",
    }


def check_prediction_bias(trades_df: pd.DataFrame) -> dict:
    """Is the model systematically predicting the wrong direction?"""
    valid = trades_df.dropna(subset=["pred_ret", "actual_ret"])
    if len(valid) < 5:
        return {"status": "insufficient_data", "bias": None}
    wrong_dir = float((np.sign(valid["pred_ret"]) != np.sign(valid["actual_ret"])).mean())
    # Bias means consistently predicting wrong direction (> 55% wrong)
    status = "alert" if wrong_dir > 0.55 else "ok"
    return {
        "metric":      "prediction_bias",
        "wrong_dir_rate": round(wrong_dir, 4),
        "status":      status,
        "message":     f"Wrong direction rate = {wrong_dir*100:.1f}% (>55% = systematic bias)",
    }


def check_sentiment_drift(sent_df: pd.DataFrame, window: int) -> dict:
    """
    KL divergence between the distribution of sent_spread in the first half
    vs the second half of the rolling window. Large divergence = drift.
    """
    if sent_df.empty or len(sent_df) < 20:
        return {"status": "insufficient_data", "kl_div": None}

    spread = sent_df["sent_spread"].dropna().values
    mid    = len(spread) // 2
    p, q   = spread[:mid], spread[mid:]

    # Bin into 20 bins for discrete KL divergence
    all_vals = np.concatenate([p, q])
    bins     = np.linspace(all_vals.min(), all_vals.max(), 21)
    p_hist   = np.histogram(p, bins=bins)[0] + 1e-9
    q_hist   = np.histogram(q, bins=bins)[0] + 1e-9
    p_hist  /= p_hist.sum()
    q_hist  /= q_hist.sum()

    kl_div = float(np.sum(rel_entr(p_hist, q_hist)))
    status = "alert" if kl_div > SENTIMENT_DRIFT_THRESH else "ok"
    return {
        "metric":    "sentiment_drift",
        "kl_div":    round(kl_div, 4),
        "threshold": SENTIMENT_DRIFT_THRESH,
        "status":    status,
        "message":   f"Sentiment KL divergence = {kl_div:.4f} (threshold {SENTIMENT_DRIFT_THRESH})",
    }


# ---------------------------------------------------------------------------
# Alert logger
# ---------------------------------------------------------------------------

def _write_alert(check_result: dict, check_date: date) -> None:
    alert = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "date":       str(check_date),
        **check_result,
    }
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")


def _append_health_row(checks: dict, check_date: date) -> None:
    row = {"date": str(check_date)}
    for name, result in checks.items():
        row[f"{name}_value"]  = result.get("value", result.get("kl_div"))
        row[f"{name}_status"] = result.get("status", "unknown")
    df = pd.DataFrame([row])
    if HEALTH_FILE.exists():
        df.to_csv(HEALTH_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HEALTH_FILE, index=False)


# ---------------------------------------------------------------------------
# Full daily health check
# ---------------------------------------------------------------------------

def run_health_check(
    check_date: date | None = None,
    db_path: Path = DB_PATH,
    retrain_on_alert: bool = False,
) -> dict:
    """
    Run all health checks and return a summary dict.
    """
    if check_date is None:
        check_date = date.today()

    log.info(f"[monitor] Health check for {check_date}")

    snap_df   = _load_recent_snapshots(MONITOR_WINDOW_DAYS, db_path)
    trades_df = _load_recent_trades(MONITOR_WINDOW_DAYS, db_path)
    sent_df   = _load_recent_sentiment(MONITOR_WINDOW_DAYS, db_path)

    checks = {
        "sharpe":    check_rolling_sharpe(snap_df),
        "hit_rate":  check_hit_rate(trades_df),
        "ic":        check_ic(trades_df),
        "bias":      check_prediction_bias(trades_df),
        "drift":     check_sentiment_drift(sent_df, MONITOR_WINDOW_DAYS),
    }

    alerts_fired = []
    for name, result in checks.items():
        status = result.get("status", "unknown")
        msg    = result.get("message", "")
        if status == "alert":
            log.warning(f"[monitor] ALERT — {name}: {msg}")
            _write_alert({**result, "check_name": name}, check_date)
            alerts_fired.append(name)
        elif status == "ok":
            log.info(f"[monitor] OK    — {name}: {msg}")
        else:
            log.info(f"[monitor] SKIP  — {name}: {status}")

    _append_health_row(checks, check_date)

    # Trigger retraining if requested and any alerts fired
    if retrain_on_alert and alerts_fired:
        log.warning(f"[monitor] Triggering incremental retraining due to: {alerts_fired}")
        try:
            from incremental_retrain import run_retrain
            run_retrain(db_path=db_path)
        except Exception as exc:
            log.error(f"[monitor] Retraining failed: {exc}")

    summary = {
        "date":          str(check_date),
        "checks":        checks,
        "alerts_fired":  alerts_fired,
        "all_ok":        len(alerts_fired) == 0,
    }

    log.info(
        f"[monitor] Done — {'all OK' if summary['all_ok'] else f'{len(alerts_fired)} alert(s)'}"
    )
    return summary


# ---------------------------------------------------------------------------
# Load alert history (used by dashboard)
# ---------------------------------------------------------------------------

def load_alerts(n: int = 50) -> list[dict]:
    """Load the last N alerts from alerts.jsonl."""
    if not ALERTS_FILE.exists():
        return []
    alerts = []
    with open(ALERTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                alerts.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return alerts[-n:]


def load_health_history() -> pd.DataFrame:
    """Load health_history.csv for dashboard charts."""
    if not HEALTH_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HEALTH_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily model health monitoring")
    parser.add_argument("--db",               default=str(DB_PATH))
    parser.add_argument("--date",             help="Check date YYYY-MM-DD (default: today)")
    parser.add_argument("--retrain-on-alert", action="store_true",
                        help="Trigger retraining if any alert fires")
    args = parser.parse_args()

    run_health_check(
        check_date        = date.fromisoformat(args.date) if args.date else None,
        db_path           = Path(args.db),
        retrain_on_alert  = args.retrain_on_alert,
    )
