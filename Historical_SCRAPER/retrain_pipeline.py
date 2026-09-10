# retrain_pipeline.py
# End-to-end pipeline that runs after historical scraping is complete:
#
#   1. Run Phase 2 sentiment scoring on all newly scraped articles
#   2. Run Phase 2 technical feature computation (extended date range)
#   3. Run Phase 2 feature assembly (joins sentiment + technical)
#   4. Fill target_ret_1d from OHLCV
#   5. Retrain LSTM from scratch on the full 2-year dataset
#   6. Retrain Transformer from scratch on the full 2-year dataset
#   7. Run ensemble evaluation and save results
#
# This is the "one-command" way to go from raw scraped data to new model weights.
#
# Usage (Windows Terminal, from historical_scraper\ directory)
#   python retrain_pipeline.py
#   python retrain_pipeline.py --skip-sentiment     # if already scored
#   python retrain_pipeline.py --skip-features      # if already computed
#   python retrain_pipeline.py --epochs 150         # more epochs for training
#   python retrain_pipeline.py --dry-run            # print plan, don't execute

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

PHASE2_DIR = Path(__file__).parent.parent / "PHASE_2"
PHASE3_DIR = Path(__file__).parent.parent / "PHASE_3"
DB_PATH    = Path(__file__).parent.parent  / "db" / "alpha.duckdb"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retrain_pipeline")


# ---------------------------------------------------------------------------
# Helper: run a Python script in the correct working directory
# ---------------------------------------------------------------------------

def run_script(script: Path, cwd: Path, extra_args: list[str] | None = None) -> bool:
    cmd = [sys.executable, str(script)] + (extra_args or [])
    log.info(f"  Running: {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    if result.returncode != 0:
        log.error(f"  Script failed with exit code {result.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Data coverage check
# ---------------------------------------------------------------------------

def check_data_coverage() -> dict:
    """Check how many articles and labelled feature rows we have."""
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        n_articles = con.execute("SELECT COUNT(*) FROM raw_news").fetchone()[0]
        n_tagged   = con.execute("SELECT COUNT(*) FROM tagged_articles").fetchone()[0]
        n_feat     = con.execute(
            "SELECT COUNT(*) FROM feature_matrix WHERE target_ret_1d IS NOT NULL"
        ).fetchone()[0]
        date_range = con.execute(
            "SELECT MIN(trading_date), MAX(trading_date) FROM raw_news"
        ).fetchone()
        tickers    = con.execute(
            "SELECT COUNT(DISTINCT ticker) FROM feature_matrix"
        ).fetchone()[0]
    finally:
        con.close()

    return {
        "raw_articles":  n_articles,
        "tagged":        n_tagged,
        "labelled_rows": n_feat,
        "date_min":      str(date_range[0]) if date_range[0] else "N/A",
        "date_max":      str(date_range[1]) if date_range[1] else "N/A",
        "tickers":       tickers,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    skip_sentiment:  bool = False,
    skip_features:   bool = False,
    skip_ohlcv:      bool = False,
    epochs:          int  = 100,
    dry_run:         bool = False,
) -> None:
    log.info("=" * 65)
    log.info("Retrain pipeline starting")
    log.info(f"  DB: {DB_PATH}")
    log.info(f"  dry_run={dry_run}  epochs={epochs}")
    log.info("=" * 65)

    # ── Pre-check ──────────────────────────────────────────────────────────
    coverage = check_data_coverage()
    log.info("\n── Current data coverage ───────────────────────────────────")
    for k, v in coverage.items():
        log.info(f"  {k:<20}: {v}")
    log.info("")

    if coverage["raw_articles"] < 1000:
        log.warning(
            "Only {n} articles in DB. Run run_scraper.py first to build history.".format(
                n=coverage["raw_articles"]
            )
        )
        if not dry_run:
            raise SystemExit(1)

    days_back = 800   # covers 2+ years

    # ── Step 1: OHLCV top-up ──────────────────────────────────────────────
    if not skip_ohlcv:
        log.info("\n── Step 1/6: OHLCV full download (2 years) ─────────────────")
        if not dry_run:
            ok = run_script(
                PHASE2_DIR.parent  / "ohlcv_downloader.py",
                PHASE2_DIR.parent ,
                ["--mode", "full"],
            )
            if not ok:
                log.error("OHLCV download failed — continuing anyway")
    else:
        log.info("\n── Step 1/6: OHLCV — SKIPPED ──────────────────────────────")

    # ── Step 2: Sentiment scoring (backfill) ──────────────────────────────
    if not skip_sentiment:
        log.info(f"\n── Step 2/6: FinBERT sentiment backfill ({days_back} days) ──")
        if not dry_run:
            ok = run_script(
                PHASE2_DIR / "sentiment_engine.py",
                PHASE2_DIR,
                ["--backfill", "--days", str(days_back), "--force"],
            )
            if not ok:
                log.error("Sentiment scoring failed — check GPU/CPU memory")
    else:
        log.info("\n── Step 2/6: Sentiment — SKIPPED ──────────────────────────")

    # ── Step 3: Technical features (backfill) ────────────────────────────
    if not skip_features:
        log.info(f"\n── Step 3/6: Technical features backfill ({days_back} days) ─")
        if not dry_run:
            ok = run_script(
                PHASE2_DIR / "technical_features.py",
                PHASE2_DIR,
                ["--backfill", "--days", str(days_back), "--force"],
            )
    else:
        log.info("\n── Step 3/6: Technical features — SKIPPED ─────────────────")

    # ── Step 4: Feature assembly ──────────────────────────────────────────
    log.info(f"\n── Step 4/6: Feature matrix assembly ({days_back} days) ────────")
    if not dry_run:
        run_script(
            PHASE2_DIR / "feature_assembler.py",
            PHASE2_DIR,
            ["--backfill", "--days", str(days_back), "--force"],
        )
        # Fill targets
        run_script(
            PHASE2_DIR / "feature_assembler.py",
            PHASE2_DIR,
            ["--fill-targets"],
        )

    # ── Post-assembly coverage check ─────────────────────────────────────
    coverage2 = check_data_coverage()
    log.info(f"\n── Feature matrix after assembly ───────────────────────────")
    log.info(f"  Labelled rows : {coverage2['labelled_rows']}")
    log.info(f"  Tickers       : {coverage2['tickers']}")
    log.info(f"  Date range    : {coverage2['date_min']}  →  {coverage2['date_max']}")

    if coverage2["labelled_rows"] < 500:
        log.error(
            f"Only {coverage2['labelled_rows']} labelled rows — need ≥500 for training. "
            "Check that OHLCV was downloaded and fill-targets ran successfully."
        )
        if not dry_run:
            raise SystemExit(1)

    log.info(f"\n  Ready to train on {coverage2['labelled_rows']:,} samples "
             f"({coverage2['tickers']} tickers)")

    # ── Step 5: Train LSTM ────────────────────────────────────────────────
    log.info(f"\n── Step 5/6: LSTM training ({epochs} max epochs) ───────────────")
    if not dry_run:
        run_script(
            PHASE3_DIR / "train_lstm.py",
            PHASE3_DIR,
            ["--epochs", str(epochs), "--db", str(DB_PATH)],
        )

    # ── Step 6: Train Transformer ─────────────────────────────────────────
    log.info(f"\n── Step 6/6: Transformer training ({epochs} max epochs) ────────")
    if not dry_run:
        run_script(
            PHASE3_DIR / "train_transformer.py",
            PHASE3_DIR,
            ["--epochs", str(epochs), "--db", str(DB_PATH)],
        )

    # ── Ensemble evaluation ───────────────────────────────────────────────
    log.info("\n── Ensemble evaluation ─────────────────────────────────────────")
    if not dry_run:
        run_script(
            PHASE3_DIR / "ensemble.py",
            PHASE3_DIR,
            ["--db", str(DB_PATH)],
        )

    log.info("\n" + "=" * 65)
    log.info("Retrain pipeline complete")
    log.info(f"  New model checkpoints in: {PHASE3_DIR / 'checkpoints'}")
    log.info("  Run signal_generator.py to use the new models for predictions.")
    log.info("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: scraped data → trained models"
    )
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip FinBERT sentiment scoring (use if already done)")
    parser.add_argument("--skip-features",  action="store_true",
                        help="Skip technical feature computation")
    parser.add_argument("--skip-ohlcv",     action="store_true",
                        help="Skip OHLCV download (use if already current)")
    parser.add_argument("--epochs",         type=int, default=100)
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print the plan without running anything")
    args = parser.parse_args()

    run_pipeline(
        skip_sentiment  = args.skip_sentiment,
        skip_features   = args.skip_features,
        skip_ohlcv      = args.skip_ohlcv,
        epochs          = args.epochs,
        dry_run         = args.dry_run,
    )
