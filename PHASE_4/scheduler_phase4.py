# scheduler_phase4.py
# Schedules the daily P&L update after market close.
#
# Schedule (IST)
# ──────────────
#  16:30  Close yesterday's open trades, open today's new positions,
#         write portfolio snapshot.
#  08:45  Morning catch-up: if last night's data was late, retry.
#
# Usage (Windows Terminal, from phase4\ directory)
#   python scheduler_phase4.py
#   python scheduler_phase4.py --run-once

from __future__ import annotations

import argparse
import logging
import signal as _signal
import sys
from datetime import date
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# --- FIX: Tell Python where to find the other phase folders ---
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))             # Looks in main folder
sys.path.insert(0, str(_root / "PHASE_2")) # Looks in Phase 2
sys.path.insert(0, str(_root / "PHASE_3")) # Looks in Phase 3
# --------------------------------------------------------------
from daily_pnl_updater import run_daily_update, DEFAULT_DB
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs\\scheduler_phase4.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("scheduler_phase4")


def job_update(db_path: Path) -> None:
    try:
        result = run_daily_update(db_path=db_path)
        log.info(f"[job] Daily update complete: {result}")
    except Exception as exc:
        log.exception(f"[job] Daily update FAILED: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 P&L update scheduler")
    parser.add_argument("--db",       default=str(DEFAULT_DB))
    parser.add_argument("--run-once", action="store_true")
    args    = parser.parse_args()
    db_path = Path(args.db)

    # Create logs dir if missing
    Path("logs").mkdir(exist_ok=True)

    log.info(f"Phase 4 scheduler starting — DB: {db_path}")

    if args.run_once:
        job_update(db_path)
        sys.exit(0)

    # Run once on startup
    job_update(db_path)

    sched = BlockingScheduler(timezone="Asia/Kolkata")

    # Post-close: 16:30 IST (OHLCV top-up runs at 16:15)
    sched.add_job(
        job_update,
        trigger=CronTrigger(hour=16, minute=30, timezone="Asia/Kolkata"),
        args=[db_path], id="pnl_postclose", name="P&L update post-close",
        max_instances=1,
    )

    # Morning catch-up: 08:45 IST
    sched.add_job(
        job_update,
        trigger=CronTrigger(hour=8, minute=45, timezone="Asia/Kolkata"),
        args=[db_path], id="pnl_morning", name="P&L morning catch-up",
        max_instances=1,
    )

    def _stop(s, f):
        sched.shutdown(wait=False)
        sys.exit(0)
    _signal.signal(_signal.SIGINT,  _stop)
    _signal.signal(_signal.SIGTERM, _stop)

    log.info("Phase 4 scheduler running. Ctrl+C to stop.")
    sched.start()
