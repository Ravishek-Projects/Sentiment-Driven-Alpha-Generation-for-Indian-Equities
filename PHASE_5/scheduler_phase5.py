# scheduler_phase5.py
# Schedules all Phase 5 jobs: daily monitoring, weekly walk-forward,
# monthly benchmark comparison and report generation.
#
# Schedule (IST)
# ──────────────
#  Daily  17:00  Model health check (after P&L update at 16:30)
#  Daily  17:15  Incremental retrain IF health check fires any alert
#  Weekly Sun 02:00  Walk-forward evaluation (full re-run)
#  Weekly Sun 04:00  Statistical significance tests
#  Weekly Sun 05:00  Benchmark comparison + report generation
#
# Usage (Windows Terminal, from phase5\ directory)
#   python scheduler_phase5.py
#   python scheduler_phase5.py --run-once      # run all jobs immediately

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

from config_phase5 import DB_PATH, EVAL_DIR, LOGS_DIR

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOGS_DIR / "scheduler_phase5.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("scheduler_phase5")


# ---------------------------------------------------------------------------
# Job wrappers
# ---------------------------------------------------------------------------

def job_monitor(db_path: Path, retrain_on_alert: bool = False) -> None:
    try:
        from model_monitor import run_health_check
        result = run_health_check(db_path=db_path, retrain_on_alert=retrain_on_alert)
        log.info(f"[job] Monitor done — alerts: {result.get('alerts_fired', [])}")
    except Exception as exc:
        log.exception(f"[job] Monitor FAILED: {exc}")


def job_walk_forward(db_path: Path) -> None:
    try:
        from walk_forward_eval import run_walk_forward
        results = run_walk_forward(db_path=db_path, no_retrain=False)
        log.info(f"[job] Walk-forward done — {len(results)} folds")
    except Exception as exc:
        log.exception(f"[job] Walk-forward FAILED: {exc}")


def job_stat_tests(db_path: Path) -> None:
    try:
        from statistical_tests import run_all_tests
        run_all_tests(EVAL_DIR / "walk_forward_results.csv")
        log.info("[job] Statistical tests done")
    except Exception as exc:
        log.exception(f"[job] Statistical tests FAILED: {exc}")


def job_benchmark(db_path: Path) -> None:
    try:
        from benchmark_comparison import run_comparison
        run_comparison(db_path)
        log.info("[job] Benchmark comparison done")
    except Exception as exc:
        log.exception(f"[job] Benchmark comparison FAILED: {exc}")


def job_report(db_path: Path) -> None:
    try:
        from report_generator import generate_report
        path = generate_report(db_path)
        log.info(f"[job] Report generated → {path}")
    except Exception as exc:
        log.exception(f"[job] Report generation FAILED: {exc}")


def run_all_once(db_path: Path, retrain_on_alert: bool = False) -> None:
    """Run every Phase 5 job once in order."""
    log.info("=" * 60)
    log.info("Phase 5 — running all jobs once")

    log.info("Step 1/5 — Model health check")
    job_monitor(db_path, retrain_on_alert)

    log.info("Step 2/5 — Walk-forward evaluation (--quick mode)")
    try:
        from walk_forward_eval import run_walk_forward
        run_walk_forward(db_path=db_path, no_retrain=True, max_folds=5)
    except Exception as exc:
        log.warning(f"Walk-forward skipped: {exc}")

    log.info("Step 3/5 — Statistical tests")
    job_stat_tests(db_path)

    log.info("Step 4/5 — Benchmark comparison")
    job_benchmark(db_path)

    log.info("Step 5/5 — Report generation")
    job_report(db_path)

    log.info("All Phase 5 jobs complete")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def build_scheduler(db_path: Path, retrain_on_alert: bool) -> BlockingScheduler:
    sched = BlockingScheduler(timezone="Asia/Kolkata")

    # Daily health check at 17:00 IST
    sched.add_job(
        job_monitor,
        trigger=CronTrigger(hour=17, minute=0, timezone="Asia/Kolkata"),
        args=[db_path, retrain_on_alert],
        id="monitor_daily", name="Daily health check",
        max_instances=1,
    )

    # Weekly walk-forward (Sunday 02:00 IST)
    sched.add_job(
        job_walk_forward,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0, timezone="Asia/Kolkata"),
        args=[db_path],
        id="walk_forward_weekly", name="Weekly walk-forward",
        max_instances=1,
    )

    # Weekly stat tests (Sunday 04:00 IST — after walk-forward finishes)
    sched.add_job(
        job_stat_tests,
        trigger=CronTrigger(day_of_week="sun", hour=4, minute=0, timezone="Asia/Kolkata"),
        args=[db_path],
        id="stat_tests_weekly", name="Weekly statistical tests",
        max_instances=1,
    )

    # Weekly benchmark comparison (Sunday 05:00)
    sched.add_job(
        job_benchmark,
        trigger=CronTrigger(day_of_week="sun", hour=5, minute=0, timezone="Asia/Kolkata"),
        args=[db_path],
        id="benchmark_weekly", name="Weekly benchmark comparison",
        max_instances=1,
    )

    # Weekly report (Sunday 05:30)
    sched.add_job(
        job_report,
        trigger=CronTrigger(day_of_week="sun", hour=5, minute=30, timezone="Asia/Kolkata"),
        args=[db_path],
        id="report_weekly", name="Weekly report generation",
        max_instances=1,
    )

    return sched


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 evaluation and monitoring scheduler")
    parser.add_argument("--db",               default=str(DB_PATH))
    parser.add_argument("--run-once",         action="store_true",
                        help="Run all jobs immediately then exit")
    parser.add_argument("--retrain-on-alert", action="store_true",
                        help="Trigger retraining when health check alerts fire")
    args    = parser.parse_args()
    db_path = Path(args.db)

    log.info(f"Phase 5 scheduler starting — DB: {db_path}")

    if args.run_once:
        run_all_once(db_path, args.retrain_on_alert)
        sys.exit(0)

    # Run monitor immediately on startup
    job_monitor(db_path, args.retrain_on_alert)

    sched = build_scheduler(db_path, args.retrain_on_alert)

    def _stop(s, f):
        sched.shutdown(wait=False)
        sys.exit(0)
    _signal.signal(_signal.SIGINT,  _stop)
    _signal.signal(_signal.SIGTERM, _stop)

    log.info("Phase 5 scheduler running. Ctrl+C to stop.")
    sched.start()
