# mint_scraper.py
# Scrapes Livemint market news archives using BeautifulSoup.
#
# Mint's archive structure
# ────────────────────────
# Mint has a sitemap-style archive accessible via:
#   https://www.livemint.com/market/{YYYY/MM/DD}
# and paginated search:
#   https://www.livemint.com/Search/Link/Keyword/Nifty/Page-{N}
#
# We use both approaches:
# 1. Date-based section pages  → /market/YYYY/MM/DD
# 2. Keyword search pagination → search for "nifty sensex market" by date
# 3. Sitemap XML               → https://www.livemint.com/sitemap-{YYYY}-{MM}.xml

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime, timedelta, timezone

from bs4 import BeautifulSoup

from http_client import get

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
_BASE = "https://www.livemint.com"

MINT_KEYWORDS = ["nifty", "sensex", "market", "stocks", "equity", "BSE", "NSE"]


# ---------------------------------------------------------------------------
# Sitemap-based link discovery (most reliable method)
# ---------------------------------------------------------------------------

def _sitemap_url(d: date) -> str:
    return f"{_BASE}/sitemap-{d.year}-{d.month:02d}.xml"


def _fetch_sitemap_links(d: date) -> list[str]:
    """
    Fetch Mint's monthly sitemap XML and filter links that fall on date `d`.
    Returns article URLs published on that date.
    """
    url  = _sitemap_url(d)
    resp = get(url, extra_headers={"Accept": "application/xml, text/xml"})
    if resp is None:
        return []

    soup  = BeautifulSoup(resp.text, "xml")
    links = []
    date_str = d.isoformat()   # "2024-03-15"

    for url_tag in soup.find_all("url"):
        loc      = url_tag.find("loc")
        lastmod  = url_tag.find("lastmod")
        changefreq = url_tag.find("changefreq")

        if loc is None:
            continue

        href = loc.get_text().strip()
        # Filter for market-related URLs
        if "/market" not in href and "/markets" not in href:
            continue

        # Date filter using lastmod
        if lastmod:
            if date_str not in lastmod.get_text():
                continue

        if href.startswith("/"):
            href = _BASE + href
        links.append(href)

    return links


# ---------------------------------------------------------------------------
# Date-section page scraping
# ---------------------------------------------------------------------------

def _section_url(d: date) -> str:
    return f"{_BASE}/market/{d.year}/{d.month:02d}/{d.day:02d}"


def _parse_listing_page(html: str) -> list[str]:
    """Parse a Mint listing/section page and return article URLs."""
    soup  = BeautifulSoup(html, "html.parser")
    links = []

    # Mint wraps articles in <div class="listingNew"> or <section class="storyListing">
    for container in soup.find_all(
        ["div","section","ul"],
        class_=re.compile(r"listingNew|storyListing|story-listing|headline", re.I)
    ):
        for a in container.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/"):
                href = _BASE + href
            if "livemint.com" in href and len(href.split("/")) >= 6:
                links.append(href)

    # Fallback: all h2/h3 links
    if not links:
        for heading in soup.find_all(["h2","h3"]):
            a = heading.find("a", href=True)
            if a:
                href = a["href"].strip()
                if href.startswith("/"):
                    href = _BASE + href
                if "livemint.com" in href:
                    links.append(href)

    seen = set()
    return [l for l in links if not (l in seen or seen.add(l))]


def fetch_mint_links(d: date) -> list[str]:
    """
    Fetch article links from Mint for a given date.
    Tries: sitemap → section page → fallback empty.
    """
    # Method 1: sitemap (most complete — covers whole month at once)
    links = _fetch_sitemap_links(d)
    if links:
        return links

    # Method 2: date-section page
    resp = get(_section_url(d))
    if resp:
        links = _parse_listing_page(resp.text)
        if links:
            return links

    return []


# ---------------------------------------------------------------------------
# Article scraper
# ---------------------------------------------------------------------------

def scrape_mint_article(url: str) -> dict | None:
    """Scrape a single Mint article page."""
    resp = get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title = None
    for sel in [
        soup.find("h1", class_=re.compile(r"headline|storyTitle|articleTitle", re.I)),
        soup.find("h1", itemprop="headline"),
        soup.find("h1"),
    ]:
        if sel and sel.get_text(strip=True):
            title = re.sub(r"\s+", " ", sel.get_text()).strip()
            break

    if not title or len(title) < 10:
        return None

    # Published time
    published_at = None
    for attr in [
        {"property": "article:published_time"},
        {"name": "publishdate"},
        {"itemprop": "datePublished"},
    ]:
        meta = soup.find("meta", attr)
        if meta and meta.get("content"):
            try:
                published_at = datetime.fromisoformat(
                    meta["content"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                break
            except ValueError:
                pass

    if published_at is None:
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            try:
                published_at = datetime.fromisoformat(
                    time_tag["datetime"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except ValueError:
                pass

    if published_at is None:
        return None

    # Summary
    summary = ""
    for body_class in [
        re.compile(r"storyContent|articleBody|mainContent|wordSection", re.I)
    ]:
        body = soup.find("div", class_=body_class)
        if body:
            for tag in body(["script", "style", "aside", "figure"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", body.get_text(" ", strip=True)).strip()
            if len(text) > 80:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                summary   = " ".join(sentences[:3])
                break

    article_id = hashlib.sha1(f"mint_scrape|{url}".encode()).hexdigest()
    return {
        "article_id":   article_id,
        "source":       "mint_markets",
        "title":        title[:500],
        "summary":      summary[:1500],
        "link":         url,
        "published_at": published_at,
        "fetched_at":   datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Batch scraper
# ---------------------------------------------------------------------------

def scrape_date_range(
    start_date: date,
    end_date:   date,
    article_scrape: bool = True,
) -> list[dict]:
    articles = []
    current  = start_date
    total    = (end_date - start_date).days + 1
    n        = 0

    while current <= end_date:
        n += 1
        log.info(f"[mint] Day {n}/{total}: {current}")
        links = fetch_mint_links(current)
        log.info(f"[mint]   → {len(links)} links")

        if article_scrape:
            for url in links:
                art = scrape_mint_article(url)
                if art:
                    articles.append(art)
        else:
            for url in links:
                slug  = url.rstrip("/").split("/")[-1].replace("-"," ")
                art_id = hashlib.sha1(f"mint|{url}".encode()).hexdigest()
                articles.append({
                    "article_id":   art_id,
                    "source":       "mint_markets",
                    "title":        slug.title()[:300],
                    "summary":      "",
                    "link":         url,
                    "published_at": datetime(current.year,current.month,current.day,
                                             9,0,0,tzinfo=IST).astimezone(timezone.utc),
                    "fetched_at":   datetime.now(timezone.utc),
                })

        current += timedelta(days=1)

    log.info(f"[mint] Scraped {len(articles)} articles over {n} days")
    return articles
