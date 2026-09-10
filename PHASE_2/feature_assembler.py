# feature_assembler.py
# Joins sentiment_scores + technical_features + ohlcv into the
# final `feature_matrix` table that the model reads at training/inference.
#
# Feature vector layout (19 dimensions)
# ──────────────────────────────────────
#  Sentiment block (5):
#    sent_pos, sent_neg, sent_neu, sent_spread, news_count (log1p-scaled)
#
#  Technical block (7):
#    rsi_14, price_to_ma20, vol_20d, rel_volume, bb_position,
#    lag_ret_1d, lag_ret_2d
#    (ma_20 and ma_50 stored for reference but NOT fed to model directly —
#     they are used to derive price_to_ma20)
#
#  Day-of-week one-hot (5):
#    dow_mon, dow_tue, dow_wed, dow_thu, dow_fri
#
#  Target (1, filled post-close):
#    target_ret_1d = log(close_{t+1} / close_t)
#
# Forward-fill rule for sentiment
# ─────────────────────────────────
# If a ticker has no news on a given trading day, the previous day's
# sentiment scores are forward-filled (up to 5 days).  Beyond 5 days
# of silence, neutral scores (1/3, 1/3, 1/3) are inserted.
#
# Scaling
# ─────────
# news_count is log1p-transformed before insertion.
# All other features are stored as raw floats; scaling to zero-mean /
# unit-variance happens inside the model's DataLoader (not here),
# so the database always holds interpretable values.
#
# Usage
#   python feature_assembler.py                   # assemble for today
#   python feature_assembler.py --date 2024-03-15
#   python feature_assembler.py --backfill --days 90
#   python feature_assembler.py --fill-targets     # backfill target_ret_1d

from __future__ import annotations

import argparse
import logging
import math
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

from db_writer import insert_feature_matrix, DEFAULT_DB
from nifty50_tickers import NIFTY50_TICKERS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NEUTRAL = 1 / 3
MAX_FFILL_DAYS = 5   # forward-fill sentiment gaps up to this many days


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_sentiment(dates: list[date], db_path: Path) -> pd.DataFrame:
    """Load sentiment_scores for the given dates (+ a lookback buffer for ffill)."""
    if not dates:
        return pd.DataFrame()
    start = min(dates) - timedelta(days=MAX_FFILL_DAYS + 7)
    end   = max(dates)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, trading_date AS date,
                   sent_pos, sent_neg, sent_neu, sent_spread, news_count
            FROM   sentiment_scores
            WHERE  trading_date BETWEEN ? AND ?
        """, [str(start), str(end)]).df()
    finally:
        con.close()
    return df


def _load_technical(dates: list[date], db_path: Path) -> pd.DataFrame:
    """Load technical_features for the given dates."""
    if not dates:
        return pd.DataFrame()
    start = min(dates)
    end   = max(dates)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, date,
                   rsi_14, ma_20, ma_50, price_to_ma20,
                   vol_20d, rel_volume, bb_position,
                   lag_ret_1d, lag_ret_2d
            FROM   technical_features
            WHERE  date BETWEEN ? AND ?
        """, [str(start), str(end)]).df()
    finally:
        con.close()
    return df


def _load_ohlcv_for_targets(db_path: Path) -> pd.DataFrame:
    """Load all OHLCV rows — used to backfill target_ret_1d."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, date, log_return_1d AS actual_ret
            FROM   ohlcv
            ORDER  BY ticker, date
        """).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Forward-fill sentiment
# ---------------------------------------------------------------------------

def _ffill_sentiment(
    sent_df: pd.DataFrame,
    tickers: list[str],
    target_dates: list[date],
) -> pd.DataFrame:
    """
    For every (ticker, date) in target_dates, ensure there is a sentiment row.
    Strategy:
      1. If the ticker had news on that exact date → use it.
      2. If not, look back up to MAX_FFILL_DAYS for the most recent score.
      3. If still not found, insert neutral scores.
    """
    sent_df = sent_df.copy()
    sent_df["date"] = pd.to_datetime(sent_df["date"]).dt.date

    # Build a quick lookup: (ticker, date) → row
    lookup: dict[tuple, dict] = {}
    for _, row in sent_df.iterrows():
        lookup[(row["ticker"], row["date"])] = row.to_dict()

    records = []
    for ticker in tickers:
        for td in target_dates:
            # Try exact match first, then walk back
            found = None
            for delta in range(MAX_FFILL_DAYS + 1):
                candidate = td - timedelta(days=delta)
                if (ticker, candidate) in lookup:
                    found = lookup[(ticker, candidate)]
                    break

            if found:
                records.append({
                    "ticker":     ticker,
                    "date":       td,
                    "sent_pos":   found["sent_pos"],
                    "sent_neg":   found["sent_neg"],
                    "sent_neu":   found["sent_neu"],
                    "sent_spread":found["sent_spread"],
                    "news_count": found["news_count"],
                })
            else:
                # Neutral fallback
                records.append({
                    "ticker":     ticker,
                    "date":       td,
                    "sent_pos":   NEUTRAL,
                    "sent_neg":   NEUTRAL,
                    "sent_neu":   NEUTRAL,
                    "sent_spread":0.0,
                    "news_count": 0,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Day-of-week encoding
# ---------------------------------------------------------------------------

def _dow_encode(d: date) -> dict:
    """Return a one-hot dict for Monday=0 … Friday=4."""
    dow = d.weekday()   # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    return {
        "dow_mon": 1.0 if dow == 0 else 0.0,
        "dow_tue": 1.0 if dow == 1 else 0.0,
        "dow_wed": 1.0 if dow == 2 else 0.0,
        "dow_thu": 1.0 if dow == 3 else 0.0,
        "dow_fri": 1.0 if dow == 4 else 0.0,
    }


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def assemble(
    target_dates: list[date],
    db_path: Path = DEFAULT_DB,
    force: bool = False,
) -> int:
    """
    Build and store feature_matrix rows for all Nifty50 tickers
    on each date in `target_dates`.

    Returns
    -------
    int : rows written
    """
    if not target_dates:
        return 0

    # Skip already-assembled rows unless forced
    if not force:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            already = {
                (r[0], r[1]) for r in con.execute("""
                    SELECT ticker, date FROM feature_matrix
                    WHERE  date IN ({})
                """.format(", ".join([f"'{d}'" for d in target_dates]))).fetchall()
            }
        finally:
            con.close()
    else:
        already = set()

    # Load raw data
    sent_raw  = _load_sentiment(target_dates, db_path)
    tech_raw  = _load_technical(target_dates, db_path)

    # Forward-fill sentiment to cover all (ticker, date) combos
    sent_filled = _ffill_sentiment(sent_raw, NIFTY50_TICKERS, target_dates)
    sent_filled["date"] = pd.to_datetime(sent_filled["date"]).dt.date

    tech_raw["date"] = pd.to_datetime(tech_raw["date"]).dt.date

    # Merge
    merged = sent_filled.merge(tech_raw, on=["ticker", "date"], how="left")

    now  = datetime.now(timezone.utc)
    rows = []

    for _, row in merged.iterrows():
        key = (row["ticker"], row["date"])
        if key in already:
            continue

        dow = _dow_encode(row["date"])

        # log1p-scale news_count so a spike in coverage doesn't dominate
        news_count_scaled = math.log1p(float(row.get("news_count") or 0))

        rows.append({
            "ticker":        row["ticker"],
            "date":          row["date"],
            # sentiment
            "sent_pos":      row.get("sent_pos",   NEUTRAL),
            "sent_neg":      row.get("sent_neg",   NEUTRAL),
            "sent_neu":      row.get("sent_neu",   NEUTRAL),
            "sent_spread":   row.get("sent_spread", 0.0),
            "news_count":    news_count_scaled,
            # technical
            "rsi_14":        row.get("rsi_14"),
            "ma_20":         row.get("ma_20"),
            "ma_50":         row.get("ma_50"),
            "price_to_ma20": row.get("price_to_ma20"),
            "vol_20d":       row.get("vol_20d"),
            "rel_volume":    row.get("rel_volume"),
            "bb_position":   row.get("bb_position"),
            "lag_ret_1d":    row.get("lag_ret_1d"),
            "lag_ret_2d":    row.get("lag_ret_2d"),
            # day-of-week
            **dow,
            # target — NULL until market closes
            "target_ret_1d": None,
            "computed_at":   now,
        })

    if not rows:
        log.info("[assembler] Nothing new to assemble.")
        return 0

    df       = pd.DataFrame(rows)
    inserted = insert_feature_matrix(df, db_path)
    log.info(f"[assembler] Feature matrix: {inserted} rows written.")
    return inserted


# ---------------------------------------------------------------------------
# Target backfill — fill target_ret_1d after market closes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Target backfill — fill target_ret_1d after market closes
# ---------------------------------------------------------------------------

def fill_targets(db_path: Path = DEFAULT_DB, overwrite: bool = False) -> int:
    """
    Fills target_ret_1d using Intraday (Open-to-Close) returns of T+1.
    target_ret_1d[T] = log( close[T+1] / open[T+1] )
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        ohlcv = con.execute("""
            SELECT ticker, date, open, close
            FROM   ohlcv
            ORDER  BY ticker, date
        """).df()
    finally:
        con.close()

    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date

    # Build lookup: (ticker, date) -> open and close
    open_lookup:  dict[tuple, float] = {}
    close_lookup: dict[tuple, float] = {}
    for _, r in ohlcv.iterrows():
        open_lookup[(r["ticker"], r["date"])] = r["open"]
        close_lookup[(r["ticker"], r["date"])] = r["close"]

    # Fetch feature_matrix rows
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        query = "SELECT ticker, date FROM feature_matrix"
        if not overwrite:
            query += " WHERE target_ret_1d IS NULL"
        unfilled = con.execute(query + " ORDER BY date").df()
    finally:
        con.close()

    if unfilled.empty:
        log.info("[assembler] No unfilled targets.")
        return 0

    unfilled["date"] = pd.to_datetime(unfilled["date"]).dt.date

    update_rows = []
    for _, row in unfilled.iterrows():
        ticker = row["ticker"]
        date_t = row["date"]

        # Find next trading day's open and close
        found_open = None
        found_close = None
        for delta in range(1, 8):
            candidate = date_t + timedelta(days=delta)
            if (ticker, candidate) in open_lookup and (ticker, candidate) in close_lookup:
                found_open = open_lookup[(ticker, candidate)]
                found_close = close_lookup[(ticker, candidate)]
                break

        if found_open is not None and found_close is not None and found_open > 0:
            # --- THE EXECUTION FIX: Intraday Return ---
            target_ret = np.log(found_close / found_open)
            update_rows.append({"ticker": ticker, "date": date_t, "target": target_ret})

    if not update_rows:
        log.info("[assembler] No fillable targets found.")
        return 0

    update_df = pd.DataFrame(update_rows)
    update_df["date"] = pd.to_datetime(update_df["date"]).dt.date

    con = duckdb.connect(str(db_path))
    try:
        con.register("_updates", update_df)
        con.execute("""
            UPDATE feature_matrix fm
            SET    target_ret_1d = u.target
            FROM   _updates u
            WHERE  fm.ticker = u.ticker
              AND  fm.date   = u.date
        """)
        count = len(update_rows)
    finally:
        con.close()

    log.info(f"[assembler] Filled {count} Intraday (Open-to-Close) targets.")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble the feature matrix from sentiment + technical data"
    )
    parser.add_argument("--db",          default=str(DEFAULT_DB))
    parser.add_argument("--date",        help="Single date: YYYY-MM-DD")
    parser.add_argument("--backfill",    action="store_true")
    parser.add_argument("--days",        type=int, default=30)
    parser.add_argument("--fill-targets",action="store_true",
                        help="Backfill NULL target_ret_1d values from OHLCV")
    
    # --- ADD THIS LINE ---
    parser.add_argument("--overwrite-targets", action="store_true", help="Force recalculate all targets")
    # ---------------------
    
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)

    # --- UPDATE THIS TO HANDLE OVERWRITE ---
    if args.fill_targets or args.overwrite_targets:
        fill_targets(db_path, overwrite=args.overwrite_targets)
    # ---------------------------------------

    elif args.backfill:
        end   = date.today()
        start = end - timedelta(days=args.days)
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        assemble(dates, db_path, args.force)
        fill_targets(db_path)

    else:
        target = date.fromisoformat(args.date) if args.date else date.today()
        assemble([target], db_path, args.force)
        fill_targets(db_path)
