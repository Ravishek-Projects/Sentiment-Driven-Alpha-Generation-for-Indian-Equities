# tagger.py
# Two responsibilities:
#   1. Assign each article a `trading_date` based on NSE session rules.
#   2. Tag each article with matching Nifty50 ticker symbols (flashtext).
#
# Both functions are PURE — no DB access, just DataFrame → DataFrame.

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import pandas_market_calendars as mcal
from flashtext import KeywordProcessor

from nifty50_tickers import TICKER_ALIASES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSE calendar (cached at module load — cheap, ~5 ms)
# ---------------------------------------------------------------------------
_NSE_CAL  = mcal.get_calendar("NSE")
_IST_OFFSET = timedelta(hours=5, minutes=30)
_MARKET_CLOSE_IST = time(15, 30)  # 3:30 PM IST

# Cache of trading days (rebuilt on first use per date range)
_TRADING_DAYS_CACHE: dict[tuple[str, str], list[date]] = {}


def _ist_now() -> datetime:
    return datetime.now(timezone.utc) + _IST_OFFSET


def _get_trading_days(start: str, end: str) -> list[date]:
    key = (start, end)
    if key not in _TRADING_DAYS_CACHE:
        schedule = _NSE_CAL.schedule(start_date=start, end_date=end)
        _TRADING_DAYS_CACHE[key] = [d.date() for d in schedule.index]
    return _TRADING_DAYS_CACHE[key]


def _next_trading_day(from_date: date) -> date:
    """Return the next NSE trading day after `from_date`."""
    end_str = (from_date + timedelta(days=14)).isoformat()
    days = _get_trading_days(from_date.isoformat(), end_str)
    for d in days:
        if d > from_date:
            return d
    # Fallback (very unlikely)
    return from_date + timedelta(days=1)


def _same_or_next_trading_day(from_date: date) -> date:
    """Return `from_date` if it's a trading day, else the next one."""
    end_str = (from_date + timedelta(days=14)).isoformat()
    days = _get_trading_days(from_date.isoformat(), end_str)
    for d in days:
        if d >= from_date:
            return d
    return from_date


# ---------------------------------------------------------------------------
# 1. Trading-date assignment
# ---------------------------------------------------------------------------

def assign_trading_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `trading_date` (DATE) column to `df`.

    Rules
    -----
    - published_at before 15:30 IST on a trading day  → that same day
    - published_at after  15:30 IST on a trading day  → next trading day
    - published_at on a weekend or NSE holiday         → next trading day
    """
    if df.empty:
        df["trading_date"] = pd.Series(dtype="object")
        return df

    trading_dates: list[date] = []

    for pub_utc in df["published_at"]:
        try:
            if pub_utc is None or pd.isna(pub_utc):
                pub_utc = datetime.now(timezone.utc)

            # Convert to IST
            pub_ist = pub_utc + _IST_OFFSET
            pub_date_ist = pub_ist.date()
            pub_time_ist = pub_ist.time()

            # Is this date a trading day?
            try:
                session_days = _get_trading_days(
                    pub_date_ist.isoformat(), pub_date_ist.isoformat()
                )
                is_trading_day = len(session_days) > 0
            except Exception:
                is_trading_day = False

            if is_trading_day and pub_time_ist <= _MARKET_CLOSE_IST:
                # Before or at close → belongs to today's session
                trading_dates.append(pub_date_ist)
            else:
                # After close or holiday/weekend → rolls to next session
                trading_dates.append(_next_trading_day(pub_date_ist))

        except Exception:
            # Graceful fallback: use today
            trading_dates.append(date.today())

    df = df.copy()
    df["trading_date"] = trading_dates
    return df


# ---------------------------------------------------------------------------
# 2. Ticker tagging (flashtext)
# ---------------------------------------------------------------------------

def _build_keyword_processor() -> KeywordProcessor:
    """Build a case-insensitive flashtext processor from TICKER_ALIASES."""
    kp = KeywordProcessor(case_sensitive=False)
    for alias, ticker in TICKER_ALIASES.items():
        kp.add_keyword(alias, ticker)
    return kp


_KP = _build_keyword_processor()  # built once at module load


def _make_tag_id(article_id: str, ticker: str) -> str:
    return hashlib.sha1(f"{article_id}|{ticker}".encode()).hexdigest()


def tag_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every article in `df`, find all Nifty50 tickers mentioned in
    title + summary.  Returns a new DataFrame with columns:

        tag_id, article_id, ticker, match_keyword, tagged_at

    One row per (article, ticker) match.
    Articles mentioning no Nifty50 company produce zero rows.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["tag_id", "article_id", "ticker", "match_keyword", "tagged_at"]
        )

    now = datetime.now(timezone.utc)
    rows: list[dict] = []

    for _, article in df.iterrows():
        text = f"{article['title']} {article.get('summary', '') or ''}"

        # flashtext returns list of (keyword_clean, ticker) tuples
        # We use extract_keywords with span_info=False → list of clean values
        found_tickers = _KP.extract_keywords(text)

        seen: set[str] = set()
        for ticker in found_tickers:
            if ticker in seen:
                continue
            seen.add(ticker)

            # Find which alias matched (for debugging)
            # Re-extract with span_info to get the actual matched string
            spans = _KP.extract_keywords(text, span_info=True)
            keyword = next(
                (kw for kw, val, *_ in spans if val == ticker), ticker
            )

            rows.append({
                "tag_id":        _make_tag_id(article["article_id"], ticker),
                "article_id":    article["article_id"],
                "ticker":        ticker,
                "match_keyword": keyword,
                "tagged_at":     now,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        log.info(
            f"[tagger] {len(df)} articles → {len(result)} ticker tags "
            f"({result['ticker'].nunique()} unique tickers)"
        )
    else:
        log.info("[tagger] No Nifty50 tickers found in this batch.")
    return result
