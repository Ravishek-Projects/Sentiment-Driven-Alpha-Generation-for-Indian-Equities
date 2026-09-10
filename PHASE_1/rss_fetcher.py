# rss_fetcher.py
# Fetches all RSS feeds in parallel, deduplicates headlines using
# MinHash LSH (Jaccard similarity), and returns a clean DataFrame
# ready for bulk insertion into DuckDB.
#
# This module is STATELESS — it never touches DuckDB directly.
# The caller (scheduler.py) is responsible for the DB write.

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import feedparser
import pandas as pd
from datasketch import MinHash, MinHashLSH

from rss_feeds import RSS_FEEDS, FETCH_TIMEOUT_SEC, FETCH_WORKERS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article_id(source: str, link: str) -> str:
    """Deterministic SHA-1 id so re-ingesting the same feed is idempotent."""
    raw = f"{source}|{link or ''}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _parse_dt(entry) -> Optional[datetime]:
    """Convert feedparser's time_struct to a timezone-aware UTC datetime."""
    try:
        t = entry.get("published_parsed") or entry.get("updated_parsed")
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def _shingle(text: str, k: int = 3) -> set[str]:
    """Character k-shingles for MinHash."""
    text = text.lower().strip()
    return {text[i : i + k] for i in range(len(text) - k + 1)} if len(text) >= k else {text}


def _minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in _shingle(text):
        m.update(shingle.encode())
    return m


# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------

def _fetch_one_feed(name: str, url: str) -> list[dict]:
    """Fetch a single RSS feed and return a list of raw article dicts."""
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "AlphaBot/1.0"})
        rows = []
        for entry in feed.entries:
            link    = entry.get("link", "")
            title   = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()[:1000]  # cap at 1000 chars
            pub_dt  = _parse_dt(entry)

            if not title:
                continue

            rows.append({
                "article_id":   _make_article_id(name, link),
                "source":       name,
                "title":        title,
                "summary":      summary,
                "link":         link,
                "published_at": pub_dt,
                "fetched_at":   datetime.now(timezone.utc),
            })

        log.info(f"[rss_fetcher] {name}: {len(rows)} entries fetched")
        return rows

    except Exception as exc:
        log.warning(f"[rss_fetcher] {name}: failed — {exc}")
        return []


def fetch_all_feeds() -> pd.DataFrame:
    """
    Fetch all configured feeds in parallel.
    Returns a DataFrame with columns:
        article_id, source, title, summary, link, published_at, fetched_at
    """
    all_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one_feed, name, url): name
            for name, url in RSS_FEEDS.items()
        }
        for future in as_completed(futures):
            all_rows.extend(future.result())

    if not all_rows:
        log.warning("[rss_fetcher] No articles fetched from any feed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    log.info(f"[rss_fetcher] Total raw articles before dedup: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Deduplication via MinHash LSH
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """
    Remove near-duplicate headlines using MinHash LSH.

    Two articles are considered duplicates when their title Jaccard
    similarity exceeds `threshold`.  The first occurrence (by published_at)
    is kept; later duplicates are dropped.

    Parameters
    ----------
    df : DataFrame returned by fetch_all_feeds()
    threshold : Jaccard similarity above which two headlines are duplicates

    Returns
    -------
    DataFrame with duplicates removed.
    """
    if df.empty:
        return df

    # Sort so we keep the earliest version of each story
    df = df.sort_values("published_at").reset_index(drop=True)

    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    keep_indices: list[int] = []

    for idx, row in df.iterrows():
        key = str(idx)
        m   = _minhash(row["title"])
        try:
            neighbours = lsh.query(m)
            if neighbours:
                # Duplicate of an already-kept article → skip
                continue
            lsh.insert(key, m)
            keep_indices.append(idx)
        except Exception:
            # Rare edge case: identical hash already in LSH
            keep_indices.append(idx)

    deduped = df.loc[keep_indices].reset_index(drop=True)
    removed = len(df) - len(deduped)
    log.info(f"[dedup] Removed {removed} duplicates → {len(deduped)} unique articles")
    return deduped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_and_deduplicate() -> pd.DataFrame:
    """
    End-to-end: fetch all feeds → deduplicate.
    Returns a clean DataFrame ready for trading-date assignment
    and ticker tagging.
    """
    raw = fetch_all_feeds()
    if raw.empty:
        return raw
    return deduplicate(raw)
