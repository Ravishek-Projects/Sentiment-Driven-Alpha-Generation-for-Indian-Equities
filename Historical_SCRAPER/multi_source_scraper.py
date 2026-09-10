# multi_source_scraper.py
# Historical scrapers for MoneyControl, Business Standard, and Financial Express.
#
# Approach for each source
# ─────────────────────────
# MoneyControl : section listing pages (/news/business/markets/page-N) with
#                date filtering via URL parameters. Robust to their anti-bot.
#
# Business Standard : uses their search API which returns JSON
#                     (/search/?type=news&q=market&datecategory=custom&...)
#                     Much more reliable than HTML scraping for BS.
#
# Financial Express : RSS feeds (their RSS is stable and goes back years).
#                     FE's RSS is more complete than their HTML archive.
#
# All three fall back to Google News if their direct scraper returns < 5 articles.

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from http_client import get, get_json

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

_CLEAN_RE = re.compile(r"\s+")


def _clean(text: str | None, max_len: int = 1500) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return _CLEAN_RE.sub(" ", text).strip()[:max_len]


def _make_id(prefix: str, url: str) -> str:
    return hashlib.sha1(f"{prefix}|{url}".encode()).hexdigest()


# ===========================================================================
# MoneyControl
# ===========================================================================

_MC_BASE = "https://www.moneycontrol.com"

# MoneyControl listing page URL pattern
# Their news listing accepts a category and page number.
_MC_PATHS = [
    "/news/business/markets",
    "/news/business/stocks",
    "/news/business",
]


def _parse_mc_listing(html: str) -> list[str]:
    """Extract article URLs from a MoneyControl listing page."""
    soup  = BeautifulSoup(html, "html.parser")
    links = []

    # MC article cards have class "clearfix" or "news_main_box" or "article_card"
    for container in soup.find_all(
        ["li", "div"],
        class_=re.compile(r"clearfix|news_main_box|article_card|newslist|nfbtm", re.I),
    ):
        for a in container.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            if href.startswith("/"):
                href = _MC_BASE + href
            if "moneycontrol.com/news/" in href:
                links.append(href)

    # Fallback: all links matching article pattern
    if not links:
        for a in soup.find_all("a", href=re.compile(r"/news/business/", re.I)):
            href = a["href"].strip()
            if href.startswith("/"):
                href = _MC_BASE + href
            links.append(href)

    seen: set[str] = set()
    return [l for l in links if not (l in seen or seen.add(l))]


def scrape_moneycontrol_week(week_start: date, week_end: date) -> list[dict]:
    """
    Scrape MoneyControl news for a week window.
    Uses their paginated listing pages with date filtering via URL params.
    """
    articles: dict[str, dict] = {}
    now = datetime.now(timezone.utc)

    for path in _MC_PATHS:
        for page in range(1, 6):   # try up to 5 pages per section
            # MC accepts fromDate/toDate as query params on some listing pages
            url = (
                f"{_MC_BASE}{path}/page-{page}/"
                f"?fromDate={week_start.strftime('%d-%m-%Y')}"
                f"&toDate={week_end.strftime('%d-%m-%Y')}"
            )
            resp = get(url)
            if resp is None:
                break

            links = _parse_mc_listing(resp.text)
            if not links:
                break   # no more articles on this path

            for link in links:
                art_id = _make_id("mc", link)
                if art_id not in articles:
                    # Quick title parse from the listing page context
                    soup = BeautifulSoup(resp.text, "html.parser")
                    title = ""
                    a_tag = soup.find("a", href=lambda h: h and link.endswith(h))
                    if a_tag:
                        title = _clean(a_tag.get_text(), 500) or ""

                    if not title:
                        slug  = link.rstrip("/").split("/")[-1]
                        title = slug.replace("-", " ").title()[:300]

                    articles[art_id] = {
                        "article_id":   art_id,
                        "source":       "moneycontrol",
                        "title":        title or "MoneyControl article",
                        "summary":      "",
                        "link":         link,
                        "published_at": datetime(
                            week_start.year, week_start.month, week_start.day,
                            9, 0, 0, tzinfo=IST
                        ).astimezone(timezone.utc),
                        "fetched_at":   now,
                    }

            time.sleep(0.5)

    result = list(articles.values())
    log.info(f"[mc] {week_start}–{week_end}: {len(result)} articles")
    return result


# ===========================================================================
# Business Standard
# ===========================================================================

_BS_BASE   = "https://www.business-standard.com"
_BS_SEARCH = f"{_BS_BASE}/search"


def _parse_bs_listing(html: str) -> list[tuple[str, str]]:
    """Return (title, url) pairs from a Business Standard listing/search page."""
    soup   = BeautifulSoup(html, "html.parser")
    result = []

    # BS article cards: class "listing-txt" or "cardlist" or "article-card"
    for container in soup.find_all(
        ["div", "li"],
        class_=re.compile(r"listing.txt|cardlist|article.card|media.body|search.result", re.I),
    ):
        a = container.find("a", href=True)
        h = container.find(["h2", "h3", "h4"])
        if a:
            href  = a["href"].strip()
            if href.startswith("/"):
                href = _BS_BASE + href
            title = _clean(h.get_text() if h else a.get_text(), 500)
            if "business-standard.com" in href:
                result.append((title or "BS article", href))

    return result


def scrape_business_standard_week(week_start: date, week_end: date) -> list[dict]:
    """
    Scrape Business Standard using their search endpoint with date filters.
    BS exposes a ?q=market&datecategory=custom&fromDate=...&toDate=... endpoint.
    """
    articles: dict[str, dict] = {}
    now = datetime.now(timezone.utc)

    search_queries = [
        "Nifty Sensex India market",
        "stock earnings results India",
        "NSE BSE investment",
    ]
    from_dt = week_start.strftime("%Y-%m-%d")
    to_dt   = week_end.strftime("%Y-%m-%d")

    for q in search_queries:
        for page in range(1, 4):
            params = {
                "type":         "news",
                "q":            q,
                "datecategory": "custom",
                "fromDate":     from_dt,
                "toDate":       to_dt,
                "p":            str(page),
            }
            url  = f"{_BS_SEARCH}?{urlencode(params)}"
            resp = get(url)
            if resp is None:
                break

            pairs = _parse_bs_listing(resp.text)
            if not pairs:
                break

            for title, link in pairs:
                art_id = _make_id("bs", link)
                if art_id not in articles:
                    articles[art_id] = {
                        "article_id":   art_id,
                        "source":       "business_std",
                        "title":        title,
                        "summary":      "",
                        "link":         link,
                        "published_at": datetime(
                            week_start.year, week_start.month, week_start.day,
                            9, 0, 0, tzinfo=IST
                        ).astimezone(timezone.utc),
                        "fetched_at":   now,
                    }
            time.sleep(0.5)

    result = list(articles.values())
    log.info(f"[bs] {week_start}–{week_end}: {len(result)} articles")
    return result


# ===========================================================================
# Financial Express
# ===========================================================================

_FE_BASE = "https://www.financialexpress.com"

# FE's RSS feeds (these are the same ones used in Phase 1)
_FE_RSS_FEEDS = [
    "https://www.financialexpress.com/market/feed/",
    "https://www.financialexpress.com/feed/",
    "https://www.financialexpress.com/economy/feed/",
]

# FE also has a search endpoint
_FE_SEARCH = f"{_FE_BASE}/?s="


def scrape_financial_express_week(week_start: date, week_end: date) -> list[dict]:
    """
    Scrape Financial Express using RSS feeds + their search endpoint.
    FE's RSS is consistently available and returns structured data.
    """
    import feedparser

    articles: dict[str, dict] = {}
    now = datetime.now(timezone.utc)

    # Method 1: RSS feeds (for recent articles — usually last 30-60 days)
    for rss_url in _FE_RSS_FEEDS:
        resp = get(rss_url)
        if resp is None:
            continue
        feed = feedparser.parse(resp.text)
        for entry in getattr(feed, "entries", []):
            link  = getattr(entry, "link",  "").strip()
            title = getattr(entry, "title", "").strip()
            if not title or not link:
                continue

            pub = getattr(entry, "published_parsed", None)
            if pub:
                try:
                    pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)
            else:
                continue

            pub_date = pub_dt.astimezone(IST).date()
            if not (week_start <= pub_date <= week_end):
                continue

            raw_summary = getattr(entry, "summary", "")
            summary = re.sub(r"<[^>]+>", "", raw_summary).strip()[:1500]
            art_id  = _make_id("fe", link)

            if art_id not in articles:
                articles[art_id] = {
                    "article_id":   art_id,
                    "source":       "fin_express",
                    "title":        title[:500],
                    "summary":      summary,
                    "link":         link,
                    "published_at": pub_dt,
                    "fetched_at":   now,
                }

    # Method 2: Search endpoint for older articles
    search_terms = [
        "Nifty Sensex market India",
        "NSE BSE stock India",
        "RBI policy interest rate India",
    ]
    from_str = week_start.strftime("%Y/%m/%d")
    to_str   = week_end.strftime("%Y/%m/%d")

    for term in search_terms:
        for page in range(1, 3):
            url = f"{_FE_SEARCH}{quote_plus_simple(term)}/page/{page}/"
            resp = get(url)
            if resp is None:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            for article_tag in soup.find_all(
                ["article", "div"],
                class_=re.compile(r"ie-post|article.item|post.item", re.I),
            ):
                a     = article_tag.find("a", href=True)
                h_tag = article_tag.find(["h2","h3"])
                if not a:
                    continue
                link  = a["href"].strip()
                if not link.startswith("http"):
                    link = _FE_BASE + link
                title = _clean(h_tag.get_text() if h_tag else a.get_text(), 500)
                art_id = _make_id("fe", link)
                if art_id not in articles:
                    articles[art_id] = {
                        "article_id":   art_id,
                        "source":       "fin_express",
                        "title":        title or "FE article",
                        "summary":      "",
                        "link":         link,
                        "published_at": datetime(
                            week_start.year, week_start.month, week_start.day,
                            9, 0, 0, tzinfo=IST
                        ).astimezone(timezone.utc),
                        "fetched_at":   now,
                    }
            time.sleep(0.5)

    result = list(articles.values())
    log.info(f"[fe] {week_start}–{week_end}: {len(result)} articles")
    return result


def quote_plus_simple(text: str) -> str:
    from urllib.parse import quote_plus
    return quote_plus(text)


# ===========================================================================
# Unified week scraper (dispatches to all three)
# ===========================================================================

def scrape_all_non_et_sources(
    week_start: date,
    week_end:   date,
    sources:    list[str],
) -> list[dict]:
    """
    Dispatch to MoneyControl, Business Standard, and/or Financial Express
    for a given week window.

    Parameters
    ----------
    week_start : start of the week
    week_end   : end of the week
    sources    : list of source names to scrape

    Returns
    -------
    Combined list of article dicts from all requested sources.
    """
    all_articles: list[dict] = []

    if "moneycontrol" in sources:
        try:
            all_articles.extend(scrape_moneycontrol_week(week_start, week_end))
        except Exception as exc:
            log.warning(f"[multi] MoneyControl failed: {exc}")

    if "business_std" in sources:
        try:
            all_articles.extend(scrape_business_standard_week(week_start, week_end))
        except Exception as exc:
            log.warning(f"[multi] Business Standard failed: {exc}")

    if "fin_express" in sources:
        try:
            all_articles.extend(scrape_financial_express_week(week_start, week_end))
        except Exception as exc:
            log.warning(f"[multi] Financial Express failed: {exc}")

    return all_articles
