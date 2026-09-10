# run_scraper.py  (v2)
# Main orchestrator for the 2-year historical news scraper.
#
# v2 changes vs v1
# ─────────────────
# • Google News RSS added as a primary source (works where firewalls block direct sites)
# • MoneyControl, Business Standard, Financial Express scrapers added
# • ET pagination fix: now consistently gets 200-400 articles/day
# • All previously scraped ET articles are preserved (idempotent writes)
# • Per-source per-week checkpointing (unchanged — safe to resume)
# • Wayback Machine fallback still available via --wayback-fill
#
# Execution order per week
# ─────────────────────────
# 1. ET sources   (fast, 200-400 articles/day, working well)
# 2. Google News  (broad, ~100-500 articles/week per window, any source)
# 3. MoneyControl (direct scrape, ~50-150 articles/week)
# 4. Business Standard (search API, ~30-80 articles/week)
# 5. Financial Express (RSS + search, ~20-60 articles/week)
# 6. [Optional] Wayback Machine gap fill
#
# Usage (Windows Terminal, from historical_scraper\ directory)
# ─────────────────────────────────────────────────────────────
#   python run_scraper.py                         # all sources, 2 years
#   python run_scraper.py --sources et_stocks et_markets google_news
#   python run_scraper.py --start 2023-06-01      # custom start
#   python run_scraper.py --full-articles         # fetch ET article bodies
#   python run_scraper.py --wayback-fill          # add Wayback Machine pass
#   python run_scraper.py --reset-checkpoint      # start fresh (preserves DB data)

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from datetime import date, timedelta
from pathlib import Path

from scraper_config import (
    SCRAPE_START_DATE, SCRAPE_END_DATE,
    CHECKPOINT, LOGS_DIR, DB_PATH,
    DEFAULT_SOURCES, SOURCE_REGISTRY,
    MIN_ARTICLES_PER_DAY,
)
from db_writer_historical import write_articles_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOGS_DIR / "scraper.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("run_scraper")

_STOP = False

def _sigint(s, f):
    global _STOP
    _STOP = True
    log.warning("Ctrl+C — stopping after current batch ...")

signal.signal(signal.SIGINT,  _sigint)
signal.signal(signal.SIGTERM, _sigint)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_cp() -> dict:
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT) as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": {}, "total_inserted": 0, "total_tagged": 0}


def _save_cp(cp: dict) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2, default=str)
    log.info(f"[checkpoint] Saved → {CHECKPOINT}")


def _week_key(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _iter_weeks(start: date, end: date):
    current = start
    while current <= end:
        week_end = min(current + timedelta(days=6), end)
        yield current, week_end
        current = week_end + timedelta(days=1)


# ---------------------------------------------------------------------------
# Source dispatcher
# ---------------------------------------------------------------------------

def _scrape_source_week(
    source_name:    str,
    week_start:     date,
    week_end:       date,
    full_articles:  bool = False,
) -> list[dict]:
    """
    Dispatch to the correct scraper module for a given source and week.
    Returns list of article dicts.
    """
    cfg    = SOURCE_REGISTRY.get(source_name, {})
    method = cfg.get("method", "")

    # ── ET sources ──────────────────────────────────────────────────────────
    if method == "et_archive":
        from et_scraper import scrape_date_range
        section = cfg.get("section", source_name)
        return scrape_date_range(
            week_start, week_end,
            section=section,
            article_scrape=full_articles,
        )

    # ── Google News RSS ─────────────────────────────────────────────────────
    elif method == "google_news":
        from google_news_scraper import scrape_week
        return scrape_week(week_start, week_end)

    # ── MoneyControl ────────────────────────────────────────────────────────
    elif method == "mc_archive":
        from multi_source_scraper import scrape_moneycontrol_week
        return scrape_moneycontrol_week(week_start, week_end)

    # ── Business Standard ───────────────────────────────────────────────────
    elif method == "bs_archive":
        from multi_source_scraper import scrape_business_standard_week
        return scrape_business_standard_week(week_start, week_end)

    # ── Financial Express ───────────────────────────────────────────────────
    elif method == "fe_archive":
        from multi_source_scraper import scrape_financial_express_week
        return scrape_financial_express_week(week_start, week_end)

    # ── Wayback Machine ─────────────────────────────────────────────────────
    elif method == "wayback":
        log.info(f"[scraper] Wayback source {source_name} — use --wayback-fill flag")
        return []

    else:
        log.warning(f"[scraper] Unknown method for source {source_name}: {method}")
        return []


# ---------------------------------------------------------------------------
# Wayback gap fill
# ---------------------------------------------------------------------------

def _run_wayback_fill(start: date, end: date, db_path: Path) -> None:
    from wayback_scraper import scrape_wayback, ET_PATTERNS, MINT_PATTERNS

    log.info(f"[wayback] Gap fill {start} → {end}")

    # ET gaps
    arts_et = scrape_wayback(ET_PATTERNS, start, end, "et", limit_per_pattern=150)
    if arts_et:
        write_articles_batch(arts_et, db_path)

    # Mint / general gaps
    arts_misc = scrape_wayback(MINT_PATTERNS, start, end, "mint", limit_per_pattern=100)
    if arts_misc:
        write_articles_batch(arts_misc, db_path)


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _db_article_count(db_path: Path) -> int:
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        n   = con.execute("SELECT COUNT(*) FROM raw_news").fetchone()[0]
        con.close()
        return n
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    sources:       list[str],
    start_date:    date,
    end_date:      date,
    full_articles: bool,
    wayback_fill:  bool,
    db_path:       Path,
) -> None:
    cp = _load_cp()

    n_initial = _db_article_count(db_path)

    log.info("=" * 65)
    log.info("Historical news scraper  (v2 — multi-source)")
    log.info(f"  Sources    : {sources}")
    log.info(f"  Date range : {start_date}  →  {end_date}")
    log.info(f"  Full articles: {full_articles}")
    log.info(f"  Wayback fill : {wayback_fill}")
    log.info(f"  DB articles before start: {n_initial:,}")
    log.info("=" * 65)

    total_inserted = cp.get("total_inserted", 0)
    total_tagged   = cp.get("total_tagged",   0)

    for week_start, week_end in _iter_weeks(start_date, end_date):
        if _STOP:
            break

        wk = _week_key(week_start)

        for source in sources:
            key = f"{source}|{wk}"

            if key in cp.get("completed", {}):
                prev = cp["completed"][key]
                log.info(
                    f"[skip] {source} {wk} — already done "
                    f"({prev.get('inserted',0)} articles)"
                )
                continue

            log.info(f"\n{'─'*55}")
            log.info(f"Source={source:<15}  Week={wk}  ({week_start}–{week_end})")

            try:
                articles = _scrape_source_week(source, week_start, week_end, full_articles)
            except Exception as exc:
                log.error(f"[scraper] {source} {wk} FAILED: {exc}")
                articles = []

            inserted = tagged = 0
            if articles:
                result    = write_articles_batch(articles, db_path)
                inserted  = result["total_inserted"]
                tagged    = result["total_tagged"]
                total_inserted += inserted
                total_tagged   += tagged
                log.info(
                    f"  ✓ {source} {wk}: "
                    f"{len(articles)} fetched → "
                    f"{inserted} new articles, {tagged} ticker tags"
                )
            else:
                log.warning(f"  ✗ {source} {wk}: 0 articles returned")

            cp.setdefault("completed", {})[key] = {
                "articles": len(articles),
                "inserted": inserted,
                "tagged":   tagged,
            }
            cp["total_inserted"] = total_inserted
            cp["total_tagged"]   = total_tagged
            _save_cp(cp)

            if _STOP:
                break

    # Wayback fill pass
    if wayback_fill and not _STOP:
        log.info("\n── Wayback Machine gap fill ──────────────────────────────")
        _run_wayback_fill(start_date, end_date, db_path)

    n_final = _db_article_count(db_path)

    log.info("\n" + "=" * 65)
    log.info("Scraping complete")
    log.info(f"  Articles in DB before : {n_initial:,}")
    log.info(f"  Articles in DB after  : {n_final:,}")
    log.info(f"  Net new articles      : {n_final - n_initial:,}")
    log.info(f"  Total ticker tags     : {total_tagged:,}")
    log.info(f"  Checkpoint file       : {CHECKPOINT}")
    log.info("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ALL = list(SOURCE_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Historical news scraper v2 — ET + Google News + MC + BS + FE"
    )
    parser.add_argument(
        "--sources", nargs="+", default=DEFAULT_SOURCES,
        help=(
            f"Sources to scrape. Available: {DEFAULT_SOURCES + ['wayback_et','wayback_misc']}. "
            f"Default: {DEFAULT_SOURCES}"
        ),
    )
    parser.add_argument(
        "--start", default=str(SCRAPE_START_DATE),
        help=f"Start date YYYY-MM-DD (default: {SCRAPE_START_DATE})",
    )
    parser.add_argument(
        "--end", default=str(SCRAPE_END_DATE),
        help=f"End date YYYY-MM-DD (default: {SCRAPE_END_DATE})",
    )
    parser.add_argument(
        "--full-articles", action="store_true",
        help="Scrape full article bodies for ET (much slower, richer summaries)",
    )
    parser.add_argument(
        "--wayback-fill", action="store_true",
        help="Run Wayback Machine pass after direct scraping",
    )
    parser.add_argument(
        "--reset-checkpoint", action="store_true",
        help="Clear checkpoint.json (previously scraped DB data is NOT deleted)",
    )
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    if args.reset_checkpoint and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        log.info("[checkpoint] Cleared — will re-scrape all weeks (DB data preserved).")

    run(
        sources       = args.sources,
        start_date    = date.fromisoformat(args.start),
        end_date      = date.fromisoformat(args.end),
        full_articles = args.full_articles,
        wayback_fill  = args.wayback_fill,
        db_path       = Path(args.db),
    )
