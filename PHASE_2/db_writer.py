# db_writer.py  (Phase 1 + Phase 2 combined)
# -----------------------------------------------
# Single module responsible for ALL DuckDB writes.
#
# Design pattern
# --------------
# Each public function:
#   1. Opens a fresh DuckDB connection
#   2. Does one bulk INSERT (via register + INSERT INTO … SELECT FROM df)
#   3. Closes the connection immediately
#
# This "open → write → close" pattern keeps the database available
# to concurrent readers (Streamlit dashboard) at all times.

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).parent.parent / "db" / "alpha.duckdb"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))


# ===========================================================================
# PHASE 1 WRITERS  (unchanged from your updated version)
# ===========================================================================

def insert_raw_news(df: pd.DataFrame, db_path: Path = DEFAULT_DB) -> int:
    if df.empty:
        return 0
    required = ["article_id", "source", "title", "summary",
                "link", "published_at", "fetched_at", "trading_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"insert_raw_news: missing columns {missing}")
    df = df[required].copy()
    con = _connect(db_path)
    try:
        con.register("_batch_news", df)
        result = con.execute("""
            INSERT INTO raw_news
            SELECT b.* FROM _batch_news b
            WHERE  b.article_id NOT IN (SELECT article_id FROM raw_news)
        """).fetchone()
        inserted = result[0] if result else 0
        log.info(f"[db_writer] raw_news: inserted {inserted} / {len(df)} rows")
        return inserted
    finally:
        con.close()


def insert_tagged_articles(df: pd.DataFrame, db_path: Path = DEFAULT_DB) -> int:
    if df.empty:
        return 0
    required = ["tag_id", "article_id", "ticker", "match_keyword", "tagged_at"]
    df = df[required].copy()
    con = _connect(db_path)
    try:
        con.register("_batch_tags", df)
        result = con.execute("""
            INSERT INTO tagged_articles
            SELECT b.* FROM _batch_tags b
            WHERE  b.tag_id NOT IN (SELECT tag_id FROM tagged_articles)
        """).fetchone()
        inserted = result[0] if result else 0
        log.info(f"[db_writer] tagged_articles: inserted {inserted} / {len(df)} rows")
        return inserted
    finally:
        con.close()


def insert_ohlcv(df: pd.DataFrame, db_path: Path = DEFAULT_DB) -> int:
    if df.empty:
        return 0
    required = ["ticker", "date", "open", "high", "low", "close", "volume", "log_return_1d"]
    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    con = _connect(db_path)
    try:
        con.register("_batch_ohlcv", df)
        result = con.execute("""
            INSERT OR REPLACE INTO ohlcv
            SELECT * FROM _batch_ohlcv
        """).fetchone()
        inserted = result[0] if result else 0
        log.info(f"[db_writer] ohlcv: upserted {inserted} rows")
        return inserted
    finally:
        con.close()


def start_pipeline_run(run_type: str, db_path: Path = DEFAULT_DB) -> str:
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    con = _connect(db_path)
    try:
        con.execute("""
            INSERT INTO pipeline_runs (run_id, run_type, started_at, status)
            VALUES (?, ?, ?, 'running')
        """, [run_id, run_type, now])
    finally:
        con.close()
    return run_id


def finish_pipeline_run(
    run_id: str,
    articles_fetched: int = 0,
    articles_new: int = 0,
    articles_tagged: int = 0,
    status: str = "success",
    error_msg: str | None = None,
    db_path: Path = DEFAULT_DB,
) -> None:
    now = datetime.now(timezone.utc)
    con = _connect(db_path)
    try:
        con.execute("""
            UPDATE pipeline_runs
            SET finished_at      = ?,
                articles_fetched = ?,
                articles_new     = ?,
                articles_tagged  = ?,
                status           = ?,
                error_msg        = ?
            WHERE run_id = ?
        """, [now, articles_fetched, articles_new, articles_tagged,
              status, error_msg, run_id])
    finally:
        con.close()


# ===========================================================================
# PHASE 2 WRITERS
# ===========================================================================

def insert_sentiment_scores(df: pd.DataFrame, db_path: Path = DEFAULT_DB) -> int:
    """
    Bulk-insert FinBERT sentiment scores into `sentiment_scores`.
    One row per (ticker, trading_date). Upserts on conflict.

    Expected columns:
        ticker, trading_date, sent_pos, sent_neg, sent_neu,
        sent_spread, news_count, articles_used, computed_at
    """
    if df.empty:
        return 0
    required = ["ticker", "trading_date", "sent_pos", "sent_neg", "sent_neu",
                "sent_spread", "news_count", "articles_used", "computed_at"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"insert_sentiment_scores: missing columns {missing}")
    df = df[required].copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

    con = _connect(db_path)
    try:
        con.register("_batch_sent", df)
        result = con.execute("""
            INSERT OR REPLACE INTO sentiment_scores
            SELECT * FROM _batch_sent
        """).fetchone()
        inserted = result[0] if result else 0
        log.info(f"[db_writer] sentiment_scores: upserted {inserted} rows")
        return inserted
    finally:
        con.close()


def insert_technical_features(df: pd.DataFrame, db_path: Path = DEFAULT_DB) -> int:
    """
    Bulk-insert technical indicator features into `technical_features`.
    One row per (ticker, date). Upserts on conflict.

    Expected columns:
        ticker, date, rsi_14, ma_20, ma_50, price_to_ma20,
        vol_20d, rel_volume, bb_position, lag_ret_1d, lag_ret_2d, computed_at
    """
    if df.empty:
        return 0
    required = ["ticker", "date", "rsi_14", "ma_20", "ma_50", "price_to_ma20",
                "vol_20d", "rel_volume", "bb_position",
                "lag_ret_1d", "lag_ret_2d", "computed_at"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"insert_technical_features: missing columns {missing}")
    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    con = _connect(db_path)
    try:
        con.register("_batch_tech", df)
        result = con.execute("""
            INSERT OR REPLACE INTO technical_features
            SELECT * FROM _batch_tech
        """).fetchone()
        inserted = result[0] if result else 0
        log.info(f"[db_writer] technical_features: upserted {inserted} rows")
        return inserted
    finally:
        con.close()


def insert_feature_matrix(df: pd.DataFrame, db_path: Path = DEFAULT_DB) -> int:
    """
    Bulk-insert the final merged feature matrix into `feature_matrix`.
    One row per (ticker, date). Upserts on conflict.

    Expected columns:
        ticker, date,
        sent_pos, sent_neg, sent_neu, sent_spread, news_count,
        rsi_14, ma_20, ma_50, price_to_ma20, vol_20d,
        rel_volume, bb_position, lag_ret_1d, lag_ret_2d,
        dow_mon, dow_tue, dow_wed, dow_thu, dow_fri,
        target_ret_1d, computed_at
    """
    if df.empty:
        return 0
    required = [
        "ticker", "date",
        "sent_pos", "sent_neg", "sent_neu", "sent_spread", "news_count",
        "rsi_14", "ma_20", "ma_50", "price_to_ma20", "vol_20d",
        "rel_volume", "bb_position", "lag_ret_1d", "lag_ret_2d",
        "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri",
        "target_ret_1d", "computed_at",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"insert_feature_matrix: missing columns {missing}")
    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    con = _connect(db_path)
    try:
        con.register("_batch_feat", df)
        result = con.execute("""
            INSERT OR REPLACE INTO feature_matrix
            SELECT * FROM _batch_feat
        """).fetchone()
        inserted = result[0] if result else 0
        log.info(f"[db_writer] feature_matrix: upserted {inserted} rows")
        return inserted
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Shared read helpers
# ---------------------------------------------------------------------------

def count_rows(table: str, db_path: Path = DEFAULT_DB) -> int:
    con = _connect(db_path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def latest_article_date(db_path: Path = DEFAULT_DB) -> str:
    con = _connect(db_path)
    try:
        row = con.execute("SELECT MAX(fetched_at) FROM raw_news").fetchone()
        return str(row[0]) if row and row[0] else "none"
    finally:
        con.close()
