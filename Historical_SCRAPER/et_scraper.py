# et_scraper.py  (v2 — pagination fix + robust selectors)
# Fixes the inconsistent article counts (79 vs 500+) by paginating
# through ET's archive system instead of fetching a single page per day.

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from bs4 import BeautifulSoup

from http_client import get
from scraper_config import et_archive_ts, ET_MAX_ARCHIVE_PAGES

log = logging.getLogger(__name__)

IST   = timezone(timedelta(hours=5, minutes=30))
_BASE = "https://economictimes.indiatimes.com"


# ---------------------------------------------------------------------------
# URL builder (v2 — pagination strategy)
# ---------------------------------------------------------------------------

def _all_archive_urls(d: date, section: str) -> list[str]:
    """
    Return ordered list of archive URLs to try.
    Each URL is a different page or section variant.
    We stop trying when a page returns 0 new articles.
    """
    ts   = et_archive_ts(d)
    urls = []

    if section == "et_stocks":
        # Page 1
        urls.append(f"{_BASE}/markets/stocks/news/archivelist/starttime-{ts}.cms")
        # Pages 2..N
        for pg in range(2, ET_MAX_ARCHIVE_PAGES + 1):
            urls.append(
                f"{_BASE}/markets/stocks/news/archivelist/page-{pg}/starttime-{ts}.cms"
            )

    elif section == "et_markets":
        urls.append(f"{_BASE}/markets/archivelist/starttime-{ts}.cms")
        for pg in range(2, ET_MAX_ARCHIVE_PAGES + 1):
            urls.append(f"{_BASE}/markets/archivelist/page-{pg}/starttime-{ts}.cms")

    elif section == "et_topnews":
        urls.append(f"{_BASE}/news/archivelist/starttime-{ts}.cms")
        for pg in range(2, ET_MAX_ARCHIVE_PAGES + 1):
            urls.append(f"{_BASE}/news/archivelist/page-{pg}/starttime-{ts}.cms")

    # General archive as supplement
    urls.append(f"{_BASE}/archivelist/starttime-{ts}.cms")
    for pg in range(2, ET_MAX_ARCHIVE_PAGES + 1):
        urls.append(f"{_BASE}/archivelist/page-{pg}/starttime-{ts}.cms")

    return urls


# ---------------------------------------------------------------------------
# Link parser
# ---------------------------------------------------------------------------

def _parse_archive_page(html: str) -> list[str]:
    soup  = BeautifulSoup(html, "html.parser")
    links = []
    page_title = soup.title.string.lower() if soup.title else ""
    if "moment" in page_title or "robot" in page_title or "cloudflare" in page_title:
        log.error(f"[et] CLOUDFLARE JS WALL DETECTED! Title: {page_title}")
        return []
    # ET has used various class names over the years
    selectors = [
        ("ul",  re.compile(r"\bcontent\b",       re.I)),
        ("div", re.compile(r"\beachStory\b",      re.I)),
        ("div", re.compile(r"\barticleList\b",    re.I)),
        ("div", re.compile(r"\bstorySummary\b",   re.I)),
        ("li",  re.compile(r"\barticle\b",        re.I)),
        ("div", re.compile(r"\bnewsItem\b",       re.I)),
        ("div", re.compile(r"\bstoriesListing\b", re.I)),
    ]
    for tag, cls_re in selectors:
        for container in soup.find_all(tag, class_=cls_re):
            for a in container.find_all("a", href=True):
                href = a["href"].strip()
                if _is_et_article(href):
                    links.append(_abs(href))

    if not links:
        main = soup.find("div", id=re.compile(r"content|main|wrapper", re.I))
        if main:
            for a in main.find_all("a", href=True):
                href = a["href"].strip()
                if _is_et_article(href):
                    links.append(_abs(href))

    return _dedup(links)


def _abs(href: str) -> str:
    return href if href.startswith("http") else _BASE + href


def _is_et_article(href: str) -> bool:
    if not href:
        return False
    if href.startswith("/"):
        href = _BASE + href
    return (
        "economictimes.indiatimes.com" in href
        and href.endswith(".cms")
        and href.count("/") >= 5
        and "archivelist" not in href
        and "rssfeeds"    not in href
    )


def _dedup(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


# ---------------------------------------------------------------------------
# Multi-page fetch (core fix)
# ---------------------------------------------------------------------------

def fetch_archive_links(d: date, section: str = "et_stocks") -> list[str]:
    """
    Fetch article links from ET archive for a given date.
    Paginates through the archive until no new links are found.
    """
    ts = et_archive_ts(d)
    date_path = f"year-{d.year},month-{d.month},starttime-{ts}.cms"
    
    # Base URL for the specific section
    # Base URL for the specific section
    if section == "et_stocks":
        base_url = f"{_BASE}/markets/stocks/news/archivelist/{date_path}"
    elif section == "et_markets":
        base_url = f"{_BASE}/markets/archivelist/{date_path}"
    else:
        base_url = f"{_BASE}/archivelist/{date_path}"

    all_links = set()
    pages_scraped = 0
    # Loop through multiple pages to bypass the 80-article cap
    for page_num in range(1, 10): 
        # ET adds &page=2 to the URL for pagination
        url = f"{base_url}?page={page_num}" if page_num > 1 else base_url
        
        resp = get(url)
        if resp is None:
            break
            
        page_links = _parse_archive_page(resp.text)
        
        if not page_links:
            break
            
        # If the page returned links we already have, we hit the end of the archive
        new_links = set(page_links) - all_links
        if not new_links:
            break
            
        all_links.update(new_links)
        log.debug(f"[et] {d} {section} (Page {page_num}): found {len(new_links)} new links")
        
        # Be polite to the server when paginating
        pages_scraped += 1
        import time
        time.sleep(1)
    log.info(
        f"[et] {d} {section}: {len(all_links)} total links "
        f"({pages_scraped} archive pages tried)"
    )

    return list(all_links)


# ---------------------------------------------------------------------------
# Article body scraper
# ---------------------------------------------------------------------------

def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_sentences(text: str, n: int = 3) -> str:
    return " ".join(re.split(r"(?<=[.!?])\s+", text)[:n]).strip()


def scrape_article(url: str) -> Optional[dict]:
    resp = get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    title = None
    for sel in [
        soup.find("h1", class_=re.compile(r"artTitle|article.*title", re.I)),
        soup.find("h1", itemprop="headline"),
        soup.find("h1"),
    ]:
        if sel and sel.get_text(strip=True):
            title = _clean(sel.get_text())
            break
    if not title or len(title) < 10:
        return None

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
        tt = soup.find("time")
        if tt and tt.get("datetime"):
            try:
                published_at = datetime.fromisoformat(
                    tt["datetime"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except ValueError:
                pass
    if published_at is None:
        return None

    summary = ""
    body = soup.find("div", class_=re.compile(r"artText|article.*body", re.I))
    if body:
        for tag in body(["script", "style", "aside", "figure", "noscript"]):
            tag.decompose()
        text = _clean(body.get_text(" ", strip=True))
        if len(text) > 100:
            summary = _first_sentences(text, 3)

    return {
        "article_id":   hashlib.sha1(f"et_scrape|{url}".encode()).hexdigest(),
        "source":       "et_scrape",
        "title":        title[:500],
        "summary":      summary[:1500],
        "link":         url,
        "published_at": published_at,
        "fetched_at":   datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Date-range batch scraper
# ---------------------------------------------------------------------------

def scrape_date_range(
    start_date: date,
    end_date:   date,
    section:    str  = "et_stocks",
    article_scrape: bool = False,
) -> list[dict]:
    articles  = []
    current   = start_date
    total     = (end_date - start_date).days + 1
    n         = 0

    while current <= end_date:
        n += 1
        log.info(f"[et] Day {n}/{total}: {current}  section={section}")
        links = fetch_archive_links(current, section)

        if article_scrape:
            for i, url in enumerate(links):
                art = scrape_article(url)
                if art:
                    art["source"] = section
                    articles.append(art)
                if (i + 1) % 10 == 0:
                    log.info(f"[et]   {i+1}/{len(links)} articles scraped")
        else:
            for url in links:
                slug = url.split(".com")[-1].replace("/", " ").replace("-", " ").replace(".cms", "")
                title = slug.strip()[:300] or "ET article"
                articles.append({
                    "article_id": hashlib.sha1(f"{section}|{url}".encode()).hexdigest(),
                    "source":     section,
                    "title":      title,
                    "summary":    "",
                    "link":       url,
                    "published_at": datetime(
                        current.year, current.month, current.day,
                        9, 0, 0, tzinfo=IST
                    ).astimezone(timezone.utc),
                    "fetched_at": datetime.now(timezone.utc),
                })

        current += timedelta(days=1)

    log.info(f"[et] {section}: {len(articles)} total articles")
    return articles
