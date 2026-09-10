# benchmarks.py
# Builds three benchmark daily return series from the ohlcv table
# and stores them in benchmark_returns for comparison.
#
# Benchmarks
# ──────────
#   nifty50_ew  : Nifty50 equal-weight buy-and-hold
#                 daily_ret = mean of all 50 tickers' log_return_1d
#
#   momentum    : Long top-decile (5 stocks) by 63-day momentum,
#                 short bottom-decile (5 stocks).
#                 Rebalanced monthly (first trading day of each month).
#                 Uses only data available on the signal date (no lookahead).
#
#   value       : Long bottom-decile by price/52-week-high ratio
#                 (cheap stocks trading far below their highs),
#                 short top-decile. Rebalanced monthly.
#
# Windows usage:
#   python benchmarks.py
#   python benchmarks.py --start 2022-01-01 --end 2024-12-31

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import duckdb
import numpy as np
import pandas as pd

DEFAULT_DB = Path(__file__).parent.parent / "db" / "alpha.duckdb"

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NIFTY50_TICKERS_IMPORT_PATH = Path(__file__).parent.parent / "phase1" / "nifty50_tickers.py"


def _load_nifty50_tickers() -> list[str]:
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("nifty50_tickers", NIFTY50_TICKERS_IMPORT_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NIFTY50_TICKERS


def _load_ohlcv(db_path: Path, start: str, end: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, date, close, log_return_1d, volume
            FROM   ohlcv
            WHERE  date BETWEEN ? AND ?
            ORDER  BY ticker, date
        """, [start, end]).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ── Benchmark 1: Nifty50 equal-weight ────────────────────────────────────

def build_nifty50_ew(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average daily log return across all 50 tickers.
    This is the simplest possible benchmark — no rebalancing cost.
    """
    ew = (
        df.groupby("date")["log_return_1d"]
        .mean()
        .reset_index()
        .rename(columns={"log_return_1d": "daily_ret"})
        .sort_values("date")
    )
    ew["cum_ret"] = ew["daily_ret"].cumsum()
    ew.insert(0, "strategy", "nifty50_ew")
    return ew[["strategy", "date", "daily_ret", "cum_ret"]]


# ── Benchmark 2: Momentum (12-1 month = 63 trading-day return) ───────────

def build_momentum(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Long top_n stocks by 63-day cumulative return.
    Short bottom_n stocks.
    Rebalanced on the first trading day of each calendar month.
    Return = equal-weight average of long leg minus short leg.
    """
    pivot = df.pivot(index="date", columns="ticker", values="log_return_1d").sort_index()
    dates = list(pivot.index)

    # Find first trading day of each month
    rebal_dates = set()
    prev_month  = None
    for d in dates:
        if d.month != prev_month:
            rebal_dates.add(d)
            prev_month = d.month

    current_longs:  list[str] = []
    current_shorts: list[str] = []
    rows = []

    for i, d in enumerate(dates):
        if d in rebal_dates and i >= 63:
            # 63-day momentum signal: sum of log returns over past 63 days
            window = pivot.iloc[max(0, i-63):i]
            cum63  = window.sum(axis=0).dropna()
            sorted_t = cum63.sort_values()
            current_shorts = list(sorted_t.head(top_n).index)
            current_longs  = list(sorted_t.tail(top_n).index)

        if not current_longs:
            rows.append({"date": d, "daily_ret": 0.0})
            continue

        row_data  = pivot.loc[d]
        long_ret  = row_data[current_longs].mean()
        short_ret = row_data[current_shorts].mean() if current_shorts else 0.0
        daily_ret = float(long_ret - short_ret)
        rows.append({"date": d, "daily_ret": daily_ret})

    result = pd.DataFrame(rows)
    result["cum_ret"]  = result["daily_ret"].cumsum()
    result["strategy"] = "momentum"
    return result[["strategy", "date", "daily_ret", "cum_ret"]]


# ── Benchmark 3: Value (price vs 52-week high) ────────────────────────────

def build_value(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Value proxy: price / 52-week high.
    Low ratio = 'cheap' (long); high ratio = 'expensive' (short).
    Rebalanced monthly. Return = long leg minus short leg.
    """
    pivot_close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    pivot_ret   = df.pivot(index="date", columns="ticker", values="log_return_1d").sort_index()
    dates = list(pivot_close.index)

    rebal_dates = set()
    prev_month  = None
    for d in dates:
        if d.month != prev_month:
            rebal_dates.add(d)
            prev_month = d.month

    current_longs:  list[str] = []
    current_shorts: list[str] = []
    rows = []

    for i, d in enumerate(dates):
        if d in rebal_dates and i >= 252:
            # 52-week high = max close over past 252 days
            window_252 = pivot_close.iloc[max(0, i-252):i]
            high_252   = window_252.max(axis=0)
            today_close = pivot_close.iloc[i]
            ratio = (today_close / high_252).dropna()
            sorted_r  = ratio.sort_values()
            current_longs  = list(sorted_r.head(top_n).index)   # cheapest
            current_shorts = list(sorted_r.tail(top_n).index)   # most expensive

        if not current_longs:
            rows.append({"date": d, "daily_ret": 0.0})
            continue

        row_data  = pivot_ret.loc[d]
        long_ret  = row_data[current_longs].mean()
        short_ret = row_data[current_shorts].mean() if current_shorts else 0.0
        rows.append({"date": d, "daily_ret": float(long_ret - short_ret)})

    result = pd.DataFrame(rows)
    result["cum_ret"]  = result["daily_ret"].cumsum()
    result["strategy"] = "value"
    return result[["strategy", "date", "daily_ret", "cum_ret"]]


# ── Write to DB ───────────────────────────────────────────────────────────

def _write_benchmark(df: pd.DataFrame, db_path: Path) -> int:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["daily_ret"] = df["daily_ret"].round(8)
    df["cum_ret"]   = df["cum_ret"].round(8)
    con = duckdb.connect(str(db_path))
    try:
        con.register("_bm", df)
        con.execute("""
            INSERT OR REPLACE INTO benchmark_returns
            SELECT * FROM _bm
        """)
        n = len(df)
    finally:
        con.close()
    return n


def build_all_benchmarks(
    db_path: Path = DEFAULT_DB,
    start:   str  = "2020-01-01",
    end:     str  | None = None,
) -> None:
    end = end or str(date.today())
    log.info(f"[benchmarks] Building {start} → {end}")

    df = _load_ohlcv(db_path, start, end)
    if df.empty:
        log.warning("[benchmarks] No OHLCV data found. Run ohlcv_downloader first.")
        return

    log.info("[benchmarks] Building nifty50_ew …")
    ew = build_nifty50_ew(df)
    n  = _write_benchmark(ew, db_path)
    log.info(f"[benchmarks] nifty50_ew: {n} rows written")

    log.info("[benchmarks] Building momentum …")
    mom = build_momentum(df)
    n   = _write_benchmark(mom, db_path)
    log.info(f"[benchmarks] momentum: {n} rows written")

    log.info("[benchmarks] Building value …")
    val = build_value(df)
    n   = _write_benchmark(val, db_path)
    log.info(f"[benchmarks] value: {n} rows written")

    log.info("[benchmarks] All benchmarks built and stored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build benchmark return series")
    parser.add_argument("--db",    default=str(DEFAULT_DB))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end",   default=None)
    args = parser.parse_args()
    build_all_benchmarks(Path(args.db), args.start, args.end)
