# scheduler_phase2.py
# Extends the Phase 1 scheduler with Phase 2 jobs.
# Can be run standalone or alongside scheduler.py (Phase 1).
#
# Schedule
# ─────────────────────────────────────────────────────────────────
#  07:45 IST  Compute technical features for today
#             (after Phase 1 OHLCV top-up at 07:00)
#
#  08:00 IST  Run FinBERT sentiment on today's articles
#             (articles already in DB from overnight RSS batches)
#
#  08:15 IST  Assemble feature matrix for today
#             (feeds model inference at 08:30)
#
#  After every Phase 1 RSS batch (triggered by APScheduler):
#             Re-run sentiment scoring for today so the feature matrix
#             stays fresh as new headlines arrive through the day.
#             ↳ Implemented as a 30-min interval job, offset by 5 min
#               from the Phase 1 RSS batch.
#
#  16:30 IST  Fill target_ret_1d for yesterday's feature matrix rows
#             (after market close + OHLCV top-up at 16:15)
#
#  16:45 IST  Compute technical features for today (post-close values)
#
# Usage
#   python scheduler_phase2.py
#   python scheduler_phase2.py --db /path/to/alpha.duckdb
#   python scheduler_phase2.py --run-once   # one full cycle then exit

from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import date, timedelta
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from sentiment_engine import score_dates, fill_missing_tickers
from technical_features import compute_technical_features
from feature_assembler import assemble, fill_targets
from db_writer import DEFAULT_DB

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scheduler_phase2.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("scheduler_phase2")


# ---------------------------------------------------------------------------
# Job functions (each is independently safe to call multiple times)
# ---------------------------------------------------------------------------

def job_technical(db_path: Path) -> None:
    today = date.today()
    log.info(f"[job] Technical features — {today}")
    compute_technical_features(today, db_path)


def job_sentiment(db_path: Path) -> None:
    today     = date.today()
    yesterday = today - timedelta(days=1)
    log.info(f"[job] Sentiment scoring — {yesterday}, {today}")
    score_dates([yesterday, today], db_path)
    fill_missing_tickers(today, db_path)


def job_assemble(db_path: Path) -> None:
    today = date.today()
    log.info(f"[job] Feature assembly — {today}")
    assemble([today], db_path)


def job_fill_targets(db_path: Path) -> None:
    log.info("[job] Filling target_ret_1d values from OHLCV")
    fill_targets(db_path)


def run_full_cycle(db_path: Path) -> None:
    """Run all Phase 2 jobs in order — used on startup and for --run-once."""
    log.info("=" * 60)
    log.info("Phase 2 full cycle starting")
    today = date.today()
    yesterday = today - timedelta(days=1)

    log.info("Step 1/4 — Technical features")
    compute_technical_features(today, db_path)

    log.info("Step 2/4 — Sentiment scoring")
    score_dates([yesterday, today], db_path)
    fill_missing_tickers(today, db_path)

    log.info("Step 3/4 — Feature assembly")
    assemble([today], db_path)

    log.info("Step 4/4 — Fill targets (yesterday's rows)")
    fill_targets(db_path)

    log.info("Phase 2 full cycle complete")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def build_scheduler(db_path: Path) -> BlockingScheduler:
    sched = BlockingScheduler(timezone="Asia/Kolkata")

    # 07:45 — technical features (after OHLCV top-up at 07:00)
    sched.add_job(
        job_technical,
        trigger=CronTrigger(hour=7, minute=45, timezone="Asia/Kolkata"),
        args=[db_path], id="tech_preopen", name="Technical features pre-open",
        max_instances=1,
    )

    # 08:00 — sentiment
    sched.add_job(
        job_sentiment,
        trigger=CronTrigger(hour=8, minute=0, timezone="Asia/Kolkata"),
        args=[db_path], id="sentiment_preopen", name="Sentiment pre-open",
        max_instances=1,
    )

    # 08:15 — assemble
    sched.add_job(
        job_assemble,
        trigger=CronTrigger(hour=8, minute=15, timezone="Asia/Kolkata"),
        args=[db_path], id="assemble_preopen", name="Feature matrix assembly pre-open",
        max_instances=1,
    )

    # Every 30 min (offset 5 min from Phase 1 RSS batch at :00 and :30)
    # Re-score sentiment so fresh headlines are captured intraday
    sched.add_job(
        job_sentiment,
        trigger=IntervalTrigger(minutes=30, start_date="2000-01-01 00:05:00"),
        args=[db_path], id="sentiment_intraday", name="Sentiment intraday refresh",
        max_instances=1, coalesce=True, misfire_grace_time=120,
    )

    # Every 30 min — re-assemble after fresh sentiment
    sched.add_job(
        job_assemble,
        trigger=IntervalTrigger(minutes=30, start_date="2000-01-01 00:10:00"),
        args=[db_path], id="assemble_intraday", name="Feature matrix intraday refresh",
        max_instances=1, coalesce=True, misfire_grace_time=120,
    )

    # 16:30 — fill yesterday's targets (post-close)
    sched.add_job(
        job_fill_targets,
        trigger=CronTrigger(hour=16, minute=30, timezone="Asia/Kolkata"),
        args=[db_path], id="fill_targets", name="Fill target_ret_1d",
        max_instances=1,
    )

    # 16:45 — recompute technical features with today's final close
    sched.add_job(
        job_technical,
        trigger=CronTrigger(hour=16, minute=45, timezone="Asia/Kolkata"),
        args=[db_path], id="tech_postclose", name="Technical features post-close",
        max_instances=1,
    )

    return sched


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 feature engineering scheduler")
    parser.add_argument("--db",       default=str(DEFAULT_DB))
    parser.add_argument("--run-once", action="store_true",
                        help="Run one full cycle then exit (for testing)")
    args  = parser.parse_args()
    db_path = Path(args.db)

    log.info(f"Phase 2 scheduler starting — DB: {db_path}")

    if args.run_once:
        run_full_cycle(db_path)
        sys.exit(0)

    # Run immediately on startup
    run_full_cycle(db_path)

    sched = build_scheduler(db_path)

    def _shutdown(signum, frame):
        log.info("Shutdown signal — stopping Phase 2 scheduler.")
        sched.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Phase 2 scheduler running. Ctrl+C to stop.")
    sched.start()
