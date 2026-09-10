# scheduler_phase3.py
# Runs the daily model inference pipeline.
# Should be started AFTER scheduler.py (Phase 1) and scheduler_phase2.py (Phase 2).
#
# Schedule (IST)
# ──────────────
#  08:30  Generate signals — features are ready by 08:15 (Phase 2)
#  Every 30 min intraday — refresh signals as fresh sentiment arrives
#  16:50  Final signal refresh after post-close OHLCV + features
#
# Usage (Windows Terminal)
#   python scheduler_phase3.py
#   python scheduler_phase3.py --run-once

from __future__ import annotations

import argparse
import logging
import signal as _signal
import sys
from datetime import date
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from signal_generator import generate_signals
from config import DB_PATH
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs\\scheduler_phase3.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("scheduler_phase3")


def job_signals(db_path: Path) -> None:
    try:
        generate_signals(target_date=str(date.today()), db_path=db_path)
    except Exception as exc:
        log.exception(f"Signal generation failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",       default=str(DB_PATH))
    parser.add_argument("--run-once", action="store_true")
    args    = parser.parse_args()
    db_path = Path(args.db)

    log.info(f"Phase 3 scheduler starting — DB: {db_path}")

    if args.run_once:
        job_signals(db_path)
        sys.exit(0)

    # Run once immediately
    job_signals(db_path)

    sched = BlockingScheduler(timezone="Asia/Kolkata")

    sched.add_job(job_signals, CronTrigger(hour=8,  minute=30, timezone="Asia/Kolkata"),
                  args=[db_path], id="signals_open",  max_instances=1)
    sched.add_job(job_signals, IntervalTrigger(minutes=30),
                  args=[db_path], id="signals_intra", max_instances=1,
                  coalesce=True, misfire_grace_time=120)
    sched.add_job(job_signals, CronTrigger(hour=16, minute=50, timezone="Asia/Kolkata"),
                  args=[db_path], id="signals_close", max_instances=1)

    def _stop(s, f):
        sched.shutdown(wait=False)
        sys.exit(0)
    _signal.signal(_signal.SIGINT,  _stop)
    _signal.signal(_signal.SIGTERM, _stop)

    log.info("Phase 3 scheduler running. Ctrl+C to stop.")
    sched.start()
