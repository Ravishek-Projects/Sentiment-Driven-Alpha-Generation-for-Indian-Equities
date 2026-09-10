# technical_features.py
# Computes technical indicators from OHLCV data stored in DuckDB
# and writes results to `technical_features`.
#
# Indicators computed (all purely from past data — zero lookahead)
# ---------------------------------------------------------------
#   rsi_14       : 14-day Relative Strength Index
#   ma_20        : 20-day Simple Moving Average of close
#   ma_50        : 50-day Simple Moving Average of close
#   price_to_ma20: close / ma_20 - 1  (mean-reversion signal)
#   vol_20d      : 20-day realised volatility (annualised, from log returns)
#   rel_volume   : today's volume / 20-day average volume
#   bb_position  : Bollinger Band position = (close - lower) / (upper - lower)
#   lag_ret_1d   : log return t-1
#   lag_ret_2d   : log return t-2
#
# All computations use only rows where date <= target_date,
# so the function is safe to call intraday.
#
# Usage
#   python technical_features.py                  # compute for today
#   python technical_features.py --date 2024-03-15
#   python technical_features.py --backfill --days 252

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# --- FIX: Tell Python to look in the parent folder for imports ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
# -----------------------------------------------------------------

from db_writer import insert_technical_features, DEFAULT_DB
from nifty50_tickers import NIFTY50_TICKERS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Minimum history needed to compute all indicators reliably
MIN_HISTORY_DAYS = 60


# ---------------------------------------------------------------------------
# Load OHLCV from DuckDB
# ---------------------------------------------------------------------------

def _load_ohlcv(
    ticker: str,
    up_to_date: date,
    lookback_days: int,
    db_path: Path,
) -> pd.DataFrame:
    """
    Load OHLCV rows for `ticker` from (up_to_date - lookback_days) to up_to_date.
    Returns a DataFrame sorted by date ascending.
    """
    start = up_to_date - timedelta(days=lookback_days)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT date, open, high, low, close, volume, log_return_1d
            FROM   ohlcv
            WHERE  ticker = ?
              AND  date BETWEEN ? AND ?
            ORDER  BY date ASC
        """, [ticker, str(start), str(up_to_date)]).df()
    finally:
        con.close()
    return df


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI on the last `period+1` close prices."""
    if len(close) < period + 1:
        return float("nan")
    delta  = close.diff().dropna()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_l  = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if avg_l == 0:
        if avg_g == 0:
            return 50.0  # perfectly flat
        return 100.0     # only gains, no losses
    rs = avg_g / avg_l
    return round(100 - (100 / (1 + rs)), 4)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def _bb_position(close: pd.Series, period: int = 20, k: float = 2.0) -> float:
    """
    Position of latest close within the Bollinger Band channel.
    Returns 0.0 when price is at the lower band, 1.0 at the upper band.
    Returns NaN if bands are degenerate (std ≈ 0).
    """
    if len(close) < period:
        return float("nan")
    window = close.iloc[-period:]
    mid    = window.mean()
    std    = window.std(ddof=1)
    if std < 1e-10:
        return float("nan")
    upper  = mid + k * std
    lower  = mid - k * std
    pos    = (close.iloc[-1] - lower) / (upper - lower)
    return round(float(np.clip(pos, -0.5, 1.5)), 6)   # allow slight out-of-band


# ---------------------------------------------------------------------------
# Main compute function (per ticker per date)
# ---------------------------------------------------------------------------

def _compute_for_ticker(
    ticker: str,
    target_date: date,
    db_path: Path,
) -> dict | None:
    """
    Compute all technical indicators for `ticker` as of `target_date`.
    Returns None if there is insufficient history.
    """
    df = _load_ohlcv(ticker, target_date, lookback_days=120, db_path=db_path)

    if len(df) < MIN_HISTORY_DAYS:
        log.debug(f"[tech] {ticker} {target_date}: only {len(df)} rows — skipping")
        return None

    close  = df["close"]
    volume = df["volume"]
    log_r  = df["log_return_1d"]

    # Moving averages
    ma_20 = close.rolling(20).mean().iloc[-1]
    ma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else float("nan")
    price_today = close.iloc[-1]

    # Price to MA20
    price_to_ma20 = (price_today / ma_20 - 1) if (ma_20 and not np.isnan(ma_20)) else float("nan")

    # Realised volatility: std of log returns × sqrt(252) to annualise
    vol_20d = log_r.iloc[-20:].std(ddof=1) * np.sqrt(252)

    # Relative volume
    avg_vol_20 = volume.iloc[-21:-1].mean()   # exclude today
    rel_volume = (float(volume.iloc[-1]) / avg_vol_20) if avg_vol_20 > 0 else float("nan")

    # Lag returns (from log_return column stored in ohlcv)
    lag_ret_1d = float(log_r.iloc[-2]) if len(log_r) >= 2 else float("nan")
    lag_ret_2d = float(log_r.iloc[-3]) if len(log_r) >= 3 else float("nan")

    return {
        "ticker":        ticker,
        "date":          target_date,
        "rsi_14":        _rsi(close, 14),
        "ma_20":         round(float(ma_20), 4) if not np.isnan(ma_20) else float("nan"),
        "ma_50":         round(float(ma_50), 4) if not np.isnan(ma_50) else float("nan"),
        "price_to_ma20": round(float(price_to_ma20), 6) if not np.isnan(price_to_ma20) else float("nan"),
        "vol_20d":       round(float(vol_20d), 6),
        "rel_volume":    round(float(rel_volume), 4) if not np.isnan(rel_volume) else float("nan"),
        "bb_position":   _bb_position(close),
        "lag_ret_1d":    round(lag_ret_1d, 8) if not np.isnan(lag_ret_1d) else float("nan"),
        "lag_ret_2d":    round(lag_ret_2d, 8) if not np.isnan(lag_ret_2d) else float("nan"),
        "computed_at":   datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_technical_features(
    target_date: date,
    db_path: Path = DEFAULT_DB,
    force: bool = False,
) -> int:
    """
    Compute and store technical features for all Nifty50 tickers on `target_date`.

    Parameters
    ----------
    target_date : date to compute indicators for
    db_path     : path to alpha.duckdb
    force       : re-compute even if rows already exist

    Returns
    -------
    int : number of rows upserted
    """
    # Check which tickers already have features for this date
    if not force:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            already = {r[0] for r in con.execute("""
                SELECT ticker FROM technical_features WHERE date = ?
            """, [str(target_date)]).fetchall()}
        finally:
            con.close()
        tickers_to_compute = [t for t in NIFTY50_TICKERS if t not in already]
    else:
        tickers_to_compute = list(NIFTY50_TICKERS)

    if not tickers_to_compute:
        log.info(f"[tech] {target_date}: all tickers already computed.")
        return 0

    log.info(f"[tech] Computing indicators for {len(tickers_to_compute)} tickers on {target_date}")

    rows = []
    skipped = 0
    for ticker in tickers_to_compute:
        result = _compute_for_ticker(ticker, target_date, db_path)
        if result:
            rows.append(result)
        else:
            skipped += 1

    if skipped:
        log.warning(f"[tech] {skipped} tickers skipped (insufficient history)")

    if not rows:
        return 0

    df       = pd.DataFrame(rows)
    inserted = insert_technical_features(df, db_path)
    log.info(f"[tech] {target_date}: inserted {inserted} technical feature rows")
    return inserted


def compute_backfill(
    start_date: date,
    end_date: date,
    db_path: Path = DEFAULT_DB,
    force: bool = False,
) -> None:
    """Compute technical features for every date in [start_date, end_date]."""
    current = start_date
    total   = 0
    while current <= end_date:
        n      = compute_technical_features(current, db_path, force)
        total += n
        current += timedelta(days=1)
    log.info(f"[tech] Backfill complete: {total} total rows written.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Nifty50 technical features from OHLCV data"
    )
    parser.add_argument("--db",       default=str(DEFAULT_DB))
    parser.add_argument("--date",     help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--backfill", action="store_true",
                        help="Compute for a range of dates")
    parser.add_argument("--days",     type=int, default=30,
                        help="With --backfill: how many days back (default 30)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-compute even if rows exist")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.backfill:
        end   = date.today()
        start = end - timedelta(days=args.days)
        compute_backfill(start, end, db_path, args.force)
    else:
        target = date.fromisoformat(args.date) if args.date else date.today()
        compute_technical_features(target, db_path, args.force)
