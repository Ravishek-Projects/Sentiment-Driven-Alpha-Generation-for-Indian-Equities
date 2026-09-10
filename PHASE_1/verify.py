# verify.py
# Run this after setup to confirm every component of Phase 1 is working.
# It does NOT touch production data — it uses a temp database.
#
# Usage:
#   python verify.py
#
# Expected output: all checks print [OK]. Any [FAIL] means something
# needs fixing before you start the scheduler.

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, timezone

PASS = "[OK]"
FAIL = "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator that wraps a check function and records pass/fail."""
    def decorator(fn):
        try:
            fn()
            results.append((name, True, ""))
            print(f"  {PASS}  {name}")
        except Exception as exc:
            tb = traceback.format_exc()
            results.append((name, False, str(exc)))
            print(f"  {FAIL}  {name}")
            print(f"         {exc}")
        return fn
    return decorator


print("\n── Phase 1 verification ──────────────────────────────────────")

# ── 1. Imports ────────────────────────────────────────────────────────────
print("\n[Imports]")

@check("feedparser")
def _(): import feedparser

@check("datasketch")
def _(): from datasketch import MinHash, MinHashLSH

@check("flashtext")
def _(): from flashtext import KeywordProcessor

@check("pandas_market_calendars")
def _(): import pandas_market_calendars as mcal; mcal.get_calendar("NSE")

@check("duckdb")
def _(): import duckdb

@check("yfinance")
def _(): import yfinance

@check("APScheduler")
def _(): from apscheduler.schedulers.blocking import BlockingScheduler

@check("pandas / numpy")
def _(): import pandas, numpy

# ── 2. Nifty50 ticker data ────────────────────────────────────────────────
print("\n[Nifty50 data]")

@check("NIFTY50_TICKERS has 50 entries")
def _():
    from nifty50_tickers import NIFTY50_TICKERS
    assert len(NIFTY50_TICKERS) == 50, f"got {len(NIFTY50_TICKERS)}"

@check("TICKER_ALIASES non-empty")
def _():
    from nifty50_tickers import TICKER_ALIASES
    assert len(TICKER_ALIASES) > 50

# ── 3. Database init ──────────────────────────────────────────────────────
print("\n[Database]")

_TMP_DB = Path(tempfile.mkdtemp()) / "verify_test.duckdb"

@check("db_init creates schema")
def _():
    from db_init import init_db
    init_db(_TMP_DB)

@check("all four tables exist")
def _():
    import duckdb
    con = duckdb.connect(str(_TMP_DB))
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    con.close()
    for t in ("raw_news", "tagged_articles", "ohlcv", "pipeline_runs"):
        assert t in tables, f"missing table: {t}"

# ── 4. RSS fetcher (single feed, live) ───────────────────────────────────
print("\n[RSS fetcher — live network call]")

@check("fetch ET Markets RSS (≥1 article)")
def _():
    import feedparser
    url = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    feed = feedparser.parse(url)
    assert len(feed.entries) >= 1, "feed returned 0 entries"

@check("fetch_and_deduplicate returns DataFrame")
def _():
    from rss_fetcher import fetch_and_deduplicate
    df = fetch_and_deduplicate()
    assert not df.empty, "empty DataFrame after dedup"
    assert "article_id" in df.columns

# ── 5. Tagger ─────────────────────────────────────────────────────────────
print("\n[Tagger]")

@check("flashtext tags RELIANCE correctly")
def _():
    import pandas as pd
    from datetime import timezone
    from tagger import assign_trading_dates, tag_articles
    sample = pd.DataFrame([{
        "article_id": "test001",
        "source":     "test",
        "title":      "Reliance Industries posts record quarterly profit",
        "summary":    "Mukesh Ambani-led RIL reports strong results",
        "link":       "https://example.com/test",
        "published_at": datetime(2024, 3, 18, 8, 0, tzinfo=timezone.utc),
        "fetched_at":   datetime(2024, 3, 18, 8, 5, tzinfo=timezone.utc),
    }])
    dated = assign_trading_dates(sample)
    assert "trading_date" in dated.columns
    tags = tag_articles(dated)
    assert not tags.empty, "no tags found"
    assert "RELIANCE" in tags["ticker"].values, f"tickers: {tags['ticker'].tolist()}"

@check("assign_trading_dates handles after-hours correctly")
def _():
    import pandas as pd
    from tagger import assign_trading_dates
    # 16:00 IST = 10:30 UTC (after 15:30 close) → rolls to next trading day
    after_close = datetime(2024, 3, 18, 10, 30, tzinfo=timezone.utc)
    sample = pd.DataFrame([{
        "article_id": "test002", "source": "test",
        "title": "test", "summary": "", "link": "",
        "published_at": after_close,
        "fetched_at":   after_close,
    }])
    result = assign_trading_dates(sample)
    pub_date = (after_close.replace(tzinfo=None) + __import__("datetime").timedelta(hours=5, minutes=30)).date()
    assert result["trading_date"].iloc[0] > pub_date, "after-hours article not rolled forward"

# ── 6. DB writer round-trip ───────────────────────────────────────────────
print("\n[DB writer round-trip]")

@check("insert_raw_news → count increases")
def _():
    import pandas as pd
    from datetime import date, timezone
    from db_writer import insert_raw_news, count_rows
    df = pd.DataFrame([{
        "article_id": "rnd001",
        "source": "test",
        "title": "Test article",
        "summary": "summary",
        "link": "https://example.com/1",
        "published_at": datetime.now(timezone.utc),
        "fetched_at":   datetime.now(timezone.utc),
        "trading_date": date.today(),
    }])
    inserted = insert_raw_news(df, _TMP_DB)
    assert inserted == 1
    # Idempotency: inserting same row again → 0 new rows
    inserted2 = insert_raw_news(df, _TMP_DB)
    assert inserted2 == 0, f"idempotency failed: {inserted2} rows inserted on second call"

@check("insert_tagged_articles works")
def _():
    import pandas as pd
    from db_writer import insert_tagged_articles, count_rows
    df = pd.DataFrame([{
        "tag_id":       "tg001",
        "article_id":   "rnd001",
        "ticker":       "RELIANCE",
        "match_keyword": "reliance",
        "tagged_at":    datetime.now(timezone.utc),
    }])
    inserted = insert_tagged_articles(df, _TMP_DB)
    assert inserted == 1

@check("pipeline_runs audit log")
def _():
    from db_writer import start_pipeline_run, finish_pipeline_run, count_rows
    rid = start_pipeline_run("test_run", _TMP_DB)
    finish_pipeline_run(rid, articles_fetched=5, articles_new=3,
                        articles_tagged=7, status="success", db_path=_TMP_DB)
    import duckdb
    con = duckdb.connect(str(_TMP_DB))
    row = con.execute(
        "SELECT status FROM pipeline_runs WHERE run_id = ?", [rid]
    ).fetchone()
    con.close()
    assert row[0] == "success"

# ── 7. OHLCV (yfinance — 2 tickers, 5 days) ──────────────────────────────
print("\n[OHLCV — live yfinance call, 2 tickers]")

@check("yfinance returns data for RELIANCE.NS")
def _():
    import yfinance as yf
    df = yf.download("RELIANCE.NS", period="5d", progress=False)
    assert not df.empty

# ── Summary ───────────────────────────────────────────────────────────────
print("\n── Summary ────────────────────────────────────────────────────")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  {passed} passed  /  {failed} failed  /  {len(results)} total\n")

if failed > 0:
    print("Failed checks:")
    for name, ok, msg in results:
        if not ok:
            print(f"  {FAIL}  {name}: {msg}")
    sys.exit(1)
else:
    print("All checks passed. Phase 1 is ready to run.")
    sys.exit(0)
