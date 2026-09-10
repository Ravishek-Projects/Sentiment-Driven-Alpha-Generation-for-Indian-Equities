# db_writer_historical.py
# Bulk-writes scraped historical articles to DuckDB.
# Opens a connection, inserts the batch, closes immediately.
# Idempotent: skips article_ids already in raw_news.
# Also runs ticker tagging inline so Phase 2 can score them immediately.

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

from scraper_config import DB_PATH, DB_BATCH_SIZE

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
_MARKET_CLOSE = 15 * 60 + 30   # 15:30 in minutes since midnight


def _trading_date(pub_utc: datetime) -> date:
    """Assign article to trading date following Phase 1 rules."""
    import pandas_market_calendars as mcal
    from datetime import timedelta

    IST_DELTA = timedelta(hours=5, minutes=30)
    pub_ist   = pub_utc + IST_DELTA
    pub_date  = pub_ist.date()
    mins_ist  = pub_ist.hour * 60 + pub_ist.minute

    cal = mcal.get_calendar("NSE")
    try:
        sched = cal.schedule(start_date=str(pub_date), end_date=str(pub_date))
        is_trading = len(sched) > 0
    except Exception:
        is_trading = pub_ist.weekday() < 5

    if is_trading and mins_ist <= _MARKET_CLOSE:
        return pub_date

    # Roll to next trading day
    next_d = pub_date + timedelta(days=1)
    for _ in range(10):
        try:
            sched = cal.schedule(start_date=str(next_d), end_date=str(next_d))
            if len(sched) > 0:
                return next_d
        except Exception:
            if next_d.weekday() < 5:
                return next_d
        next_d += timedelta(days=1)
    return next_d


def _assign_trading_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised trading date assignment (batch for speed)."""
    df = df.copy()
    df["trading_date"] = df["published_at"].apply(
        lambda x: _trading_date(x) if pd.notna(x) else date.today()
    )
    return df


def _tag_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Run flashtext ticker tagging on a batch of articles."""
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent ))
    from tagger import tag_articles
    return tag_articles(df)


def write_articles_batch(
    articles: list[dict],
    db_path: Path = DB_PATH,
    tag_inline: bool = True,
) -> dict:
    """
    Write a list of article dicts to DuckDB in batches of DB_BATCH_SIZE.

    Parameters
    ----------
    articles    : list of dicts from any scraper
    db_path     : path to alpha.duckdb
    tag_inline  : if True, also run ticker tagging and write tagged_articles

    Returns
    -------
    dict: {total_inserted, total_tagged, batches}
    """
    if not articles:
        return {"total_inserted": 0, "total_tagged": 0, "batches": 0}

    # Required columns for raw_news
    required = ["article_id","source","title","summary","link","published_at","fetched_at"]

    df = pd.DataFrame(articles)
    for col in required:
        if col not in df.columns:
            df[col] = None

    # --- FIX: Destroy any duplicates inside the batch before inserting ---
    df = df.drop_duplicates(subset=["article_id"])
    # -------------------------------------------------------------------

    # Assign trading dates
    log.info(f"[writer] Assigning trading dates for {len(df)} articles ...")
    df = _assign_trading_dates(df)

    total_inserted = 0
    total_tagged   = 0
    batches        = 0

    # Split into batches
    for start in range(0, len(df), DB_BATCH_SIZE):
        batch = df.iloc[start : start + DB_BATCH_SIZE].copy()
        batches += 1

        # Prepare raw_news batch
        news_df = batch[required + ["trading_date"]].copy()
        news_df["published_at"] = pd.to_datetime(news_df["published_at"], utc=True)
        news_df["fetched_at"]   = pd.to_datetime(news_df["fetched_at"],   utc=True)
        news_df["trading_date"] = pd.to_datetime(news_df["trading_date"]).dt.date

        # Bulk insert raw_news (skip existing)
        con = duckdb.connect(str(db_path))
        try:
            con.register("_batch_hist", news_df)
            result = con.execute("""
                INSERT INTO raw_news
                SELECT b.*
                FROM   _batch_hist b
                WHERE  b.article_id NOT IN (SELECT article_id FROM raw_news)
            """).fetchone()
            inserted = result[0] if result else 0
            total_inserted += inserted
        finally:
            con.close()

        log.info(f"[writer] Batch {batches}: inserted {inserted}/{len(batch)} articles")

        # Tag inline
        if tag_inline and inserted > 0:
            tags_df = _tag_articles(batch)
            if not tags_df.empty:
                import hashlib
                from datetime import datetime, timezone as tz

                now = datetime.now(tz.utc)
                tags_df["tagged_at"] = now

                con2 = duckdb.connect(str(db_path))
                try:
                    con2.register("_batch_tags", tags_df)
                    res2 = con2.execute("""
                        INSERT INTO tagged_articles
                        SELECT b.*
                        FROM   _batch_tags b
                        WHERE  b.tag_id NOT IN (SELECT tag_id FROM tagged_articles)
                    """).fetchone()
                    tagged = res2[0] if res2 else 0
                    total_tagged += tagged
                finally:
                    con2.close()

    log.info(
        f"[writer] Done — {total_inserted} articles inserted, "
        f"{total_tagged} tags created across {batches} batches"
    )
    return {"total_inserted": total_inserted, "total_tagged": total_tagged, "batches": batches}
