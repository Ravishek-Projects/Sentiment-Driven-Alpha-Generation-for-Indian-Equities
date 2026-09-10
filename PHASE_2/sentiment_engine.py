# sentiment_engine.py
# Runs FinBERT inference on news headlines stored in DuckDB and writes
# aggregated sentiment scores back to `sentiment_scores`.
#
# How it works
# ------------
# 1. Query DuckDB for articles in a given trading_date window that have
#    not yet been scored (or need re-scoring).
# 2. For each (ticker, trading_date) group, collect all headlines.
# 3. Batch-infer FinBERT on those headlines (GPU if available, else CPU).
# 4. Aggregate: mean of pos/neg/neu logits across all articles.
# 5. Bulk-write to `sentiment_scores` via db_writer.
#
# FinBERT model
# -------------
# ProsusAI/finbert — a BERT-base model fine-tuned on financial text.
# Output labels: "positive", "negative", "neutral".
# We use the softmax probabilities (not the raw label) so each article
# produces a (pos, neg, neu) triple that sums to 1.
#
# Usage
#   python sentiment_engine.py                    # score yesterday + today
#   python sentiment_engine.py --date 2024-03-15  # score a specific date
#   python sentiment_engine.py --backfill         # score all unscored dates
#   python sentiment_engine.py --backfill --days 90

from __future__ import annotations

import argparse
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import torch
from transformers import pipeline as hf_pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- FIX: Tell Python to look in the parent folder for imports ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
# -----------------------------------------------------------------

from db_writer import insert_sentiment_scores, DEFAULT_DB

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FINBERT_MODEL   = "ProsusAI/finbert"
BATCH_SIZE      = 32        # articles per FinBERT inference call
MAX_ARTICLES    = 32        # max articles per (ticker, date) — keeps latency bounded
MAX_TOKEN_LEN   = 512       # FinBERT's max context window
NEUTRAL_FILL    = 1 / 3     # fallback when a ticker has zero articles


# ---------------------------------------------------------------------------
# Model loader (singleton — loaded once per process)
# ---------------------------------------------------------------------------

_PIPELINE = None

def _get_pipeline():
    """
    Load FinBERT once and cache it for the process lifetime.
    Uses GPU (cuda) if available, else CPU.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (cuda:0)" if device == 0 else "CPU"
    log.info(f"[sentiment] Loading {FINBERT_MODEL} on {device_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

    _PIPELINE = hf_pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,           # return all 3 class scores per input
        truncation=True,
        max_length=MAX_TOKEN_LEN,
    )
    log.info("[sentiment] FinBERT loaded successfully.")
    return _PIPELINE


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _score_texts(texts: list[str]) -> list[dict]:
    """
    Run FinBERT on a list of texts.
    Returns a list of dicts: [{"positive": p, "negative": n, "neutral": u}, ...]
    """
    pipe = _get_pipeline()
    results = []

    # Process in batches to avoid OOM on large sets
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        outputs = pipe(chunk)           # list of list of {"label": ..., "score": ...}
        for item_scores in outputs:
            row = {d["label"].lower(): d["score"] for d in item_scores}
            results.append(row)

    return results


def _aggregate_scores(score_list: list[dict]) -> dict:
    """
    Average positive/negative/neutral scores across all articles
    for one (ticker, trading_date) group.
    """
    if not score_list:
        return {"sent_pos": NEUTRAL_FILL, "sent_neg": NEUTRAL_FILL,
                "sent_neu": NEUTRAL_FILL, "sent_spread": 0.0}

    pos = sum(s.get("positive", 0.0) for s in score_list) / len(score_list)
    neg = sum(s.get("negative", 0.0) for s in score_list) / len(score_list)
    neu = sum(s.get("neutral",  0.0) for s in score_list) / len(score_list)
    return {
        "sent_pos":    round(pos, 6),
        "sent_neg":    round(neg, 6),
        "sent_neu":    round(neu, 6),
        "sent_spread": round(pos - neg, 6),
    }


# ---------------------------------------------------------------------------
# Data loading from DuckDB
# ---------------------------------------------------------------------------

def _load_articles(
    trading_dates: list[date],
    db_path: Path,
) -> pd.DataFrame:
    """
    Fetch all tagged articles for the given trading_dates from DuckDB.
    Returns DataFrame with: ticker, trading_date, text
    where text = title + " " + summary (truncated).
    """
    if not trading_dates:
        return pd.DataFrame()

    # Build a parameterised IN clause
    placeholders = ", ".join(["?" for _ in trading_dates])
    date_strs = [str(d) for d in trading_dates]

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(f"""
            SELECT
                t.ticker,
                n.trading_date,
                n.title || ' ' || COALESCE(n.summary, '') AS text
            FROM   tagged_articles t
            JOIN   raw_news        n USING (article_id)
            WHERE  n.trading_date IN ({placeholders})
            ORDER  BY t.ticker, n.trading_date, n.published_at
        """, date_strs).df()
    finally:
        con.close()

    return df


def _already_scored_dates(
    trading_dates: list[date],
    db_path: Path,
) -> set[tuple[str, date]]:
    """
    Return the set of (ticker, trading_date) pairs already in sentiment_scores.
    Used to skip re-computation unless explicitly requested.
    """
    if not trading_dates:
        return set()
    placeholders = ", ".join(["?" for _ in trading_dates])
    date_strs = [str(d) for d in trading_dates]
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(f"""
            SELECT ticker, trading_date
            FROM   sentiment_scores
            WHERE  trading_date IN ({placeholders})
        """, date_strs).fetchall()
    finally:
        con.close()
    return {(r[0], r[1]) for r in rows}


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_dates(
    trading_dates: list[date],
    db_path: Path = DEFAULT_DB,
    force: bool = False,
) -> int:
    """
    Compute and store FinBERT sentiment for every (ticker, trading_date)
    pair in `trading_dates`.

    Parameters
    ----------
    trading_dates : list of dates to process
    db_path       : path to alpha.duckdb
    force         : if True, re-score even if already in sentiment_scores

    Returns
    -------
    int : number of (ticker, date) rows written
    """
    if not trading_dates:
        log.info("[sentiment] No dates to score.")
        return 0

    log.info(f"[sentiment] Scoring {len(trading_dates)} date(s): "
             f"{min(trading_dates)} → {max(trading_dates)}")

    # Load raw articles
    articles_df = _load_articles(trading_dates, db_path)

    if articles_df.empty:
        log.warning("[sentiment] No tagged articles found for these dates.")
        return 0

    # Filter out already-scored (ticker, date) pairs unless forced
    if not force:
        already = _already_scored_dates(trading_dates, db_path)
    else:
        already = set()

    now = datetime.now(timezone.utc)
    output_rows: list[dict] = []

    # Group by (ticker, trading_date)
    for (ticker, tdate), group in articles_df.groupby(["ticker", "trading_date"]):
        if (ticker, tdate) in already:
            continue

        texts = group["text"].tolist()
        total_articles = len(texts)

        # Truncate to MAX_ARTICLES — take most recent (list is ordered by published_at)
        used_texts = texts[-MAX_ARTICLES:] if len(texts) > MAX_ARTICLES else texts

        # Run FinBERT
        scores = _score_texts(used_texts)
        agg    = _aggregate_scores(scores)

        output_rows.append({
            "ticker":        ticker,
            "trading_date":  tdate,
            "sent_pos":      agg["sent_pos"],
            "sent_neg":      agg["sent_neg"],
            "sent_neu":      agg["sent_neu"],
            "sent_spread":   agg["sent_spread"],
            "news_count":    total_articles,
            "articles_used": len(used_texts),
            "computed_at":   now,
        })

    if not output_rows:
        log.info("[sentiment] All requested dates already scored.")
        return 0

    out_df   = pd.DataFrame(output_rows)
    inserted = insert_sentiment_scores(out_df, db_path)
    log.info(f"[sentiment] Wrote {inserted} sentiment rows to DB.")
    return inserted


def fill_missing_tickers(
    trading_date: date,
    db_path: Path = DEFAULT_DB,
) -> int:
    """
    For tickers that appear in technical_features but have NO articles
    on `trading_date`, insert a neutral sentiment row (1/3, 1/3, 1/3).
    This ensures the feature_matrix has no NULL sentiment columns.
    """
    from nifty50_tickers import NIFTY50_TICKERS

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        scored = {r[0] for r in con.execute("""
            SELECT ticker FROM sentiment_scores WHERE trading_date = ?
        """, [str(trading_date)]).fetchall()}
    finally:
        con.close()

    missing = [t for t in NIFTY50_TICKERS if t not in scored]
    if not missing:
        return 0

    now = datetime.now(timezone.utc)
    rows = [{
        "ticker":        t,
        "trading_date":  trading_date,
        "sent_pos":      NEUTRAL_FILL,
        "sent_neg":      NEUTRAL_FILL,
        "sent_neu":      NEUTRAL_FILL,
        "sent_spread":   0.0,
        "news_count":    0,
        "articles_used": 0,
        "computed_at":   now,
    } for t in missing]

    inserted = insert_sentiment_scores(pd.DataFrame(rows), db_path)
    log.info(f"[sentiment] Filled {inserted} neutral rows for tickers with no news.")
    return inserted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FinBERT sentiment scoring on RSS articles"
    )
    parser.add_argument("--db",       default=str(DEFAULT_DB))
    parser.add_argument("--date",     help="Score a single date: YYYY-MM-DD")
    parser.add_argument("--backfill", action="store_true",
                        help="Score all unscored dates in raw_news")
    parser.add_argument("--days",     type=int, default=7,
                        help="With --backfill: how many days back to look (default 7)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-score even if date already in sentiment_scores")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.date:
        target = date.fromisoformat(args.date)
        score_dates([target], db_path, force=args.force)
        fill_missing_tickers(target, db_path)

    elif args.backfill:
        end_d   = date.today()
        start_d = end_d - timedelta(days=args.days)
        dates   = [start_d + timedelta(days=i) for i in range((end_d - start_d).days + 1)]
        score_dates(dates, db_path, force=args.force)
        for d in dates:
            fill_missing_tickers(d, db_path)

    else:
        # Default: score yesterday and today
        today     = date.today()
        yesterday = today - timedelta(days=1)
        score_dates([yesterday, today], db_path, force=args.force)
        for d in [yesterday, today]:
            fill_missing_tickers(d, db_path)
