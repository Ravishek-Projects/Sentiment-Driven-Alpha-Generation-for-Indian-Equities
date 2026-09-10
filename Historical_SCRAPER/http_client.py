# http_client.py
# Robust HTTP client with:
#   - Exponential backoff retry
#   - Per-domain rate limiting (polite crawling)
#   - User-agent rotation
#   - Session reuse for connection pooling
#   - Response validation

from __future__ import annotations

import logging
import random
import time
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scraper_config import (
    REQUEST_TIMEOUT, RETRY_MAX, RETRY_BACKOFF,
    DOMAIN_DELAYS, USER_AGENTS,
)

log = logging.getLogger(__name__)

# Track last request time per domain for rate limiting
_LAST_REQUEST: dict[str, float] = {}


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=RETRY_MAX,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://",  adapter)
    session.mount("https://", adapter)
    return session


_SESSION = _make_session()


def _rate_limit(domain: str) -> None:
    """Sleep if we've requested this domain too recently."""
    delay = DOMAIN_DELAYS.get(domain, DOMAIN_DELAYS["default"])
    last  = _LAST_REQUEST.get(domain, 0.0)
    elapsed = time.time() - last
    if elapsed < delay:
        sleep_time = delay - elapsed + random.uniform(0.1, 0.5)
        time.sleep(sleep_time)
    _LAST_REQUEST[domain] = time.time()


def get(
    url: str,
    params: dict | None = None,
    extra_headers: dict | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> requests.Response | None:
    """
    Perform a GET request with rate limiting, user-agent rotation, and retry.
    Returns None on failure instead of raising.
    """
    domain = urlparse(url).netloc
    _rate_limit(domain)

    # --- FIX 1: Hyper-realistic Chrome Headers ---
    # We override your random USER_AGENTS with a hardcoded, modern Chrome signature
    # to bypass advanced bot detection (like Cloudflare or Akamai).
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Referer": "https://www.google.com/", # Pretend we clicked from Google
    }
    if extra_headers:
        headers.update(extra_headers)

    try:
        resp = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
        _LAST_REQUEST[domain] = time.time()

        if resp.status_code == 200:
            # --- FIX 2: The Soft Block Detector ---
            # Check the actual HTML text to ensure it's not a hidden CAPTCHA
            page_text = resp.text.lower()
            if "access denied" in page_text or "captcha" in page_text or "incapsula" in page_text:
                log.error(f"[http] SOFT BLOCK DETECTED (CAPTCHA): {url}")
                return None
            
            return resp
            
        elif resp.status_code == 404:
            log.debug(f"[http] 404: {url}")
        elif resp.status_code == 429:
            # Rate-limited — back off extra
            log.warning(f"[http] 429 rate-limited: {url}  sleeping 30s")
            time.sleep(30)
        elif resp.status_code in [401, 403]:
            log.warning(f"[http] {resp.status_code} FORBIDDEN (Hard Block): {url}")
            return None
        else:
            log.warning(f"[http] {resp.status_code}: {url}")
        return None

    except requests.exceptions.Timeout:
        log.warning(f"[http] Timeout: {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        log.warning(f"[http] Connection error: {url}  {e}")
        return None
    except Exception as e:
        log.error(f"[http] Unexpected error: {url}  {e}")
        return None


def get_json(url: str, params: dict | None = None) -> dict | list | None:
    """Fetch and parse JSON. Returns None on failure."""
    resp = get(url, params=params, extra_headers={"Accept": "application/json"})
    if resp is None:
        return None
    try:
        return resp.json()
    except Exception as e:
        log.warning(f"[http] JSON parse failed: {url}  {e}")
        return None
