# ohlcv_downloader.py
# Downloads historical and daily OHLCV data for all Nifty50 stocks
# from yfinance and writes them into DuckDB.
#
# Two modes
# ---------
#   full    : Download last 2 years of history (run once on project setup)
#   topup   : Download last 5 trading days and upsert (run daily after market close)
#
# Usage
#   python ohlcv_downloader.py --mode full
#   python ohlcv_downloader.py --mode topup
#   python ohlcv_downloader.py --mode full --db /path/to/alpha.duckdb

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from nifty50_tickers import NIFTY50_TICKERS, NIFTY50_YF
from db_writer import insert_ohlcv, start_pipeline_run, finish_pipeline_run, DEFAULT_DB

import shutil
import os

# Delete the temporary folder where yfinance stores blocked cookies
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "yfinance")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("yfinance cache cleared!")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Strip the .NS suffix to get the plain NSE symbol
_YF_TO_NSE: dict[str, str] = {yf_sym: nse for nse, yf_sym in zip(NIFTY50_TICKERS, NIFTY50_YF)}


def _download(period: str, interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV for all Nifty50 stocks from yfinance.
    Returns a tidy DataFrame with columns:
        ticker, date, open, high, low, close, volume, log_return_1d
    """
    log.info(f"[ohlcv] Downloading {period} of data for {len(NIFTY50_YF)} tickers ...")
    raw = yf.download(
        tickers=NIFTY50_YF,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,       # prices adjusted for splits / dividends
        progress=False,
        threads=True,
    )

    if raw.empty:
        log.warning("[ohlcv] yfinance returned empty DataFrame.")
        return pd.DataFrame()

    records: list[dict] = []

    for yf_sym in NIFTY50_YF:
        nse_sym = _YF_TO_NSE[yf_sym]
        try:
            # Multi-ticker download → MultiIndex columns (field, ticker)
            if isinstance(raw.columns, pd.MultiIndex):
                ticker_df = raw[yf_sym].copy()
            else:
                # Single ticker fallback (shouldn't happen with a list input)
                ticker_df = raw.copy()

            ticker_df = ticker_df.dropna(subset=["Close"])
            ticker_df.index = pd.to_datetime(ticker_df.index).normalize()
            ticker_df = ticker_df.sort_index()

            # Log-return: log(close_t / close_{t-1})
            ticker_df["log_return_1d"] = np.log(
                ticker_df["Close"] / ticker_df["Close"].shift(1)
            )

            for dt, row in ticker_df.iterrows():
                records.append({
                    "ticker":        nse_sym,
                    "date":          dt.date(),
                    "open":          round(float(row.get("Open",  0) or 0), 4),
                    "high":          round(float(row.get("High",  0) or 0), 4),
                    "low":           round(float(row.get("Low",   0) or 0), 4),
                    "close":         round(float(row.get("Close", 0) or 0), 4),
                    "volume":        int(row.get("Volume", 0) or 0),
                    "log_return_1d": round(float(row["log_return_1d"]) if pd.notna(row["log_return_1d"]) else 0.0, 8),
                })

        except Exception as exc:
            log.warning(f"[ohlcv] {nse_sym} ({yf_sym}): skipped — {exc}")
            continue

    df = pd.DataFrame(records)
    log.info(f"[ohlcv] Assembled {len(df)} rows for {df['ticker'].nunique()} tickers")
    return df


def run_full(db_path: Path) -> None:
    run_id = start_pipeline_run("ohlcv_full", db_path)
    try:
        df = _download(period="5y")
        inserted = insert_ohlcv(df, db_path)
        finish_pipeline_run(run_id, articles_new=inserted, status="success", db_path=db_path)
        log.info(f"[ohlcv] Full download complete — {inserted} rows upserted.")
    except Exception as exc:
        finish_pipeline_run(run_id, status="error", error_msg=str(exc), db_path=db_path)
        raise


def run_topup(db_path: Path) -> None:
    run_id = start_pipeline_run("ohlcv_topup", db_path)
    try:
        df = _download(period="5d")  # last 5 days to cover long weekends
        inserted = insert_ohlcv(df, db_path)
        finish_pipeline_run(run_id, articles_new=inserted, status="success", db_path=db_path)
        log.info(f"[ohlcv] Top-up complete — {inserted} rows upserted.")
    except Exception as exc:
        finish_pipeline_run(run_id, status="error", error_msg=str(exc), db_path=db_path)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Nifty50 OHLCV into DuckDB")
    parser.add_argument("--mode", choices=["full", "topup"], default="topup",
                        help="full = 2yr history; topup = last 5 days")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to alpha.duckdb")
    args = parser.parse_args()

    db = Path(args.db)
    if args.mode == "full":
        run_full(db)
    else:
        run_topup(db)
