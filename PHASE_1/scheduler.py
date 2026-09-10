# scheduler.py
# The main long-running process for Phase 1.
#
# Schedule
# --------
#   Every 30 min  : fetch RSS feeds → deduplicate → tag tickers → bulk-insert DuckDB
#   Daily 16:15 IST : OHLCV top-up (after NSE close at 15:30 + 45 min buffer)
#   Daily 07:00 IST : OHLCV top-up (morning pre-open, catches any late T-1 data)
#
# The 30-minute batch interval balances freshness against:
#   - RSS feed update frequency (most update every 15–60 min)
#   - DuckDB write contention (short write window, long read availability)
#   - yfinance rate limits (not a concern at this volume)
#
# Usage
#   python scheduler.py                        # uses default DB path
#   python scheduler.py --db /path/to/alpha.duckdb
#   python scheduler.py --run-once             # single RSS batch, then exit (for testing)

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from rss_fetcher import fetch_and_deduplicate
from tagger import assign_trading_dates, tag_articles
from db_writer import (
    insert_raw_news,
    insert_tagged_articles,
    start_pipeline_run,
    finish_pipeline_run,
    count_rows,
    latest_article_date,
    DEFAULT_DB,
)
from ohlcv_downloader import run_topup
from rss_feeds import BATCH_INTERVAL_MIN

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "scheduler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("scheduler")


# ---------------------------------------------------------------------------
# Core batch job
# ---------------------------------------------------------------------------

def rss_batch_job(db_path: Path) -> None:
    """
    One full RSS batch cycle:
      fetch → deduplicate → assign trading dates → tag tickers → DB write
    """
    run_id = start_pipeline_run("rss_batch", db_path)
    log.info("=" * 60)
    log.info(f"RSS batch started  (run_id={run_id[:8]}…)")

    try:
        # Step 1 — fetch and deduplicate
        df = fetch_and_deduplicate()
        fetched = len(df)
        log.info(f"Step 1 complete: {fetched} unique articles")

        if df.empty:
            finish_pipeline_run(run_id, status="success", db_path=db_path)
            log.info("No articles this batch — done.")
            return

        # Step 2 — assign trading dates
        df = assign_trading_dates(df)
        log.info("Step 2 complete: trading dates assigned")

        # Step 3 — bulk-insert raw news (idempotent)
        new_articles = insert_raw_news(df, db_path)
        log.info(f"Step 3 complete: {new_articles} new articles inserted into raw_news")

        if new_articles == 0:
            # All articles already in DB — skip tagging
            finish_pipeline_run(
                run_id,
                articles_fetched=fetched,
                articles_new=0,
                articles_tagged=0,
                status="success",
                db_path=db_path,
            )
            log.info("All articles already existed — skipping tagging.")
            return

        # Step 4 — tag only genuinely new articles
        # Re-filter to rows that were actually inserted
        from db_writer import _connect
        con = _connect(db_path)
        existing_ids = set(
            row[0] for row in con.execute("SELECT article_id FROM raw_news").fetchall()
        )
        con.close()
        new_df = df[df["article_id"].isin(existing_ids)].copy()
        tags_df = tag_articles(new_df)
        tagged = insert_tagged_articles(tags_df, db_path)
        log.info(f"Step 4 complete: {tagged} ticker tags inserted")

        # Step 5 — audit log
        total_news = count_rows("raw_news", db_path)
        total_tags = count_rows("tagged_articles", db_path)
        log.info(f"DB totals → raw_news: {total_news}  tagged_articles: {total_tags}")

        finish_pipeline_run(
            run_id,
            articles_fetched=fetched,
            articles_new=new_articles,
            articles_tagged=tagged,
            status="success",
            db_path=db_path,
        )
        log.info(f"RSS batch complete  (run_id={run_id[:8]}…)")

    except Exception as exc:
        log.exception(f"RSS batch FAILED: {exc}")
        finish_pipeline_run(run_id, status="error", error_msg=str(exc), db_path=db_path)


def ohlcv_topup_job(db_path: Path) -> None:
    """Thin wrapper so APScheduler can call the OHLCV top-up."""
    log.info("OHLCV top-up started")
    try:
        run_topup(db_path)
        log.info("OHLCV top-up complete")
    except Exception as exc:
        log.exception(f"OHLCV top-up FAILED: {exc}")


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def build_scheduler(db_path: Path) -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # RSS batch every 30 minutes
    scheduler.add_job(
        rss_batch_job,
        trigger=IntervalTrigger(minutes=BATCH_INTERVAL_MIN),
        args=[db_path],
        id="rss_batch",
        name="RSS fetch + DuckDB insert",
        max_instances=1,          # never overlap
        misfire_grace_time=120,   # if missed by <2 min, still run
        coalesce=True,            # if multiple misfires, run once
    )

    # OHLCV top-up at 16:15 IST (NSE closes 15:30, buffer for data availability)
    scheduler.add_job(
        ohlcv_topup_job,
        trigger=CronTrigger(hour=16, minute=15, timezone="Asia/Kolkata"),
        args=[db_path],
        id="ohlcv_close",
        name="OHLCV post-close top-up",
        max_instances=1,
    )

    # OHLCV top-up at 07:00 IST (pre-open, catches late T-1 data adjustments)
    scheduler.add_job(
        ohlcv_topup_job,
        trigger=CronTrigger(hour=7, minute=0, timezone="Asia/Kolkata"),
        args=[db_path],
        id="ohlcv_preopen",
        name="OHLCV pre-open top-up",
        max_instances=1,
    )

    return scheduler


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 scheduler — RSS + OHLCV pipeline")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to alpha.duckdb")
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single RSS batch immediately then exit (useful for testing)",
    )
    args = parser.parse_args()
    db_path = Path(args.db)

    log.info(f"Phase 1 scheduler starting — DB: {db_path}")
    log.info(f"RSS batch interval: every {BATCH_INTERVAL_MIN} minutes")

    if args.run_once:
        log.info("--run-once flag set: running single batch then exiting")
        rss_batch_job(db_path)
        log.info("Single batch complete. Exiting.")
        return

    scheduler = build_scheduler(db_path)

    # Run one immediate RSS batch on startup so we don't wait 30 min for first data
    log.info("Running immediate startup RSS batch ...")
    rss_batch_job(db_path)

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(signum, frame):
        log.info("Shutdown signal received — stopping scheduler gracefully.")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Scheduler running. Press Ctrl+C to stop.")
    scheduler.start()


if __name__ == "__main__":
    main()
