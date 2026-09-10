# google_news_scraper.py
# Scrapes Google News RSS feeds for historical financial news.
#
# Why Google News works when direct site scraping doesn't
# ─────────────────────────────────────────────────────────
# • Google News aggregates from ALL sources (MoneyControl, BS, FE, ET, Mint)
#   in one place — no per-site firewall issues.
# • The RSS endpoint (news.google.com/rss/search) is freely available,
#   no API key, no authentication, works with simple HTTP GET.
# • Supports the `after:YYYY-MM-DD before:YYYY-MM-DD` operators for
#   date-bounded historical queries.
# • Rate limit: ~100 queries/hour is safe. We use 1.5s between requests.
#
# Limitation: Google News RSS returns max ~100 results per query.
# We work around this by rotating multiple queries per date window
# and aggregating results.
#
# Additional: pygooglenews library (optional — falls back to raw RSS if not installed)

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser

from http_client import get
from scraper_config import GOOGLE_NEWS_BASE, GOOGLE_NEWS_QUERIES, DOMAIN_DELAYS

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
_GN_DELAY = DOMAIN_DELAYS.get("news.google.com", 1.5)


# ---------------------------------------------------------------------------
# Google News RSS query builder
# ---------------------------------------------------------------------------

def _gn_rss_url(query: str, after: date, before: date) -> str:
    """
    Build a Google News RSS URL with date operators.
    Google News supports: after:YYYY-MM-DD before:YYYY-MM-DD
    """
    date_filter = f"after:{after.isoformat()} before:{before.isoformat()}"
    full_query  = f"{query} {date_filter}"
    params = (
        f"?q={quote_plus(full_query)}"
        f"&hl=en-IN&gl=IN&ceid=IN:en"
    )
    return GOOGLE_NEWS_BASE + params


# ---------------------------------------------------------------------------
# RSS parser
# ---------------------------------------------------------------------------

def _parse_gn_entry(entry) -> dict | None:
    """Parse one feedparser entry from Google News RSS."""
    title = getattr(entry, "title", "").strip()
    link  = getattr(entry, "link",  "").strip()

    if not title or not link or len(title) < 8:
        return None

    # Google News link is a redirect — the real article URL is buried in it.
    # For our purposes (FinBERT sentiment scoring) the title is sufficient,
    # but we store the redirect URL as the link for deduplication.
    published_parsed = getattr(entry, "published_parsed", None)
    if published_parsed:
        try:
            published_at = datetime(*published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            published_at = datetime.now(timezone.utc)
    else:
        published_at = datetime.now(timezone.utc)

    # Summary: Google News often includes a snippet in the summary field
    raw_summary = getattr(entry, "summary", "")
    # Strip HTML tags from summary
    summary = re.sub(r"<[^>]+>", "", raw_summary).strip()[:1500]

    # Source publication name (e.g. "The Economic Times", "MoneyControl")
    source_tag = getattr(entry, "source", None)
    source_name = getattr(source_tag, "title", "google_news") if source_tag else "google_news"

    article_id = hashlib.sha1(f"gnews|{link}".encode()).hexdigest()

    return {
        "article_id":   article_id,
        "source":       f"gnews_{_slug(source_name)}",
        "title":        title[:500],
        "summary":      summary,
        "link":         link,
        "published_at": published_at,
        "fetched_at":   datetime.now(timezone.utc),
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:30]


# ---------------------------------------------------------------------------
# Per-query fetcher
# ---------------------------------------------------------------------------

def fetch_query(query: str, after: date, before: date) -> list[dict]:
    """
    Fetch Google News RSS results for one query and date window.
    Returns list of article dicts.
    """
    url  = _gn_rss_url(query, after, before)
    resp = get(url, extra_headers={"Accept": "application/rss+xml, application/xml, text/xml"})
    if resp is None:
        return []

    feed    = feedparser.parse(resp.text)
    entries = getattr(feed, "entries", [])
    articles = []

    for entry in entries:
        art = _parse_gn_entry(entry)
        if art:
            articles.append(art)

    log.debug(f"[gnews] '{query[:40]}' ({after}–{before}): {len(articles)} results")
    return articles


# ---------------------------------------------------------------------------
# Date-range scraper (rotates all queries)
# ---------------------------------------------------------------------------

def scrape_date_range(
    start_date: date,
    end_date:   date,
    window_days: int = 7,
) -> list[dict]:
    """
    Scrape Google News for all GOOGLE_NEWS_QUERIES across the date range.

    Parameters
    ----------
    start_date  : first date to cover
    end_date    : last date to cover
    window_days : days per query window (Google News returns ~100 results/query;
                  smaller windows give denser coverage for long ranges)

    Returns
    -------
    List of article dicts, deduplicated by article_id.
    """
    all_articles: dict[str, dict] = {}   # article_id → dict (dedup)
    current = start_date

    total_windows = ((end_date - start_date).days // window_days) + 1
    window_num    = 0

    while current <= end_date:
        window_end = min(current + timedelta(days=window_days - 1), end_date)
        window_num += 1

        log.info(
            f"[gnews] Window {window_num}/{total_windows}: "
            f"{current} → {window_end}"
        )

        for query in GOOGLE_NEWS_QUERIES:
            articles = fetch_query(query, current, window_end)
            for art in articles:
                all_articles[art["article_id"]] = art

            # Polite pause between queries
            time.sleep(_GN_DELAY)

        log.info(
            f"[gnews]   Window {window_num}: "
            f"{len(all_articles)} unique articles so far"
        )

        current = window_end + timedelta(days=1)

    result = list(all_articles.values())
    log.info(f"[gnews] Total: {len(result)} unique articles for range {start_date}–{end_date}")
    return result


# ---------------------------------------------------------------------------
# Per-day convenience wrapper (called by run_scraper.py per-week dispatch)
# ---------------------------------------------------------------------------

def scrape_week(week_start: date, week_end: date) -> list[dict]:
    """Wrapper for run_scraper.py compatibility."""
    return scrape_date_range(week_start, week_end, window_days=7)
