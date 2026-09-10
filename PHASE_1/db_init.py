# db_init.py
# Run once to create the DuckDB database and all tables.
# Safe to re-run — uses CREATE TABLE IF NOT EXISTS throughout.
#
# Usage:
#   python db_init.py
#   python db_init.py --db /path/to/custom.duckdb

import argparse
import duckdb
from pathlib import Path

DEFAULT_DB = Path(__file__).parent.parent / "db" / "alpha.duckdb"


DDL = """
-- ----------------------------------------------------------------
-- 1. RAW NEWS
--    One row per unique article ingested from any RSS feed.
--    article_id is a SHA-1 of (source + link) so re-ingesting
--    the same feed never creates duplicates.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_news (
    article_id      VARCHAR PRIMARY KEY,   -- SHA1(source||link)
    source          VARCHAR NOT NULL,      -- feed key, e.g. "et_markets"
    title           VARCHAR NOT NULL,
    summary         VARCHAR,
    link            VARCHAR,
    published_at    TIMESTAMPTZ,           -- original publish time (UTC)
    trading_date    DATE,                  -- NSE trading date this article belongs to
    fetched_at      TIMESTAMPTZ NOT NULL   -- when our pipeline ingested it
);

-- ----------------------------------------------------------------
-- 2. TAGGED ARTICLES
--    After ticker-tagging, one row per (article, ticker) pair.
--    An article mentioning three Nifty50 companies produces
--    three rows here.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tagged_articles (
    tag_id          VARCHAR PRIMARY KEY,   -- SHA1(article_id||ticker)
    article_id      VARCHAR NOT NULL,      -- FK → raw_news.article_id
    ticker          VARCHAR NOT NULL,      -- NSE symbol, e.g. "RELIANCE"
    match_keyword   VARCHAR,              -- which alias triggered the tag
    tagged_at       TIMESTAMPTZ NOT NULL
);

-- ----------------------------------------------------------------
-- 3. DAILY OHLCV
--    Adjusted close prices from yfinance for Nifty50 stocks.
--    Populated by ohlcv_downloader.py on startup and daily top-up.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker          VARCHAR NOT NULL,
    date            DATE    NOT NULL,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    volume          BIGINT,
    log_return_1d   DOUBLE,               -- log(close_t / close_{t-1})
    PRIMARY KEY (ticker, date)
);

-- ----------------------------------------------------------------
-- 4. PIPELINE RUNS
--    Audit log — one row per scheduled batch run.
--    Lets you see at a glance whether the 6 AM job succeeded.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          VARCHAR PRIMARY KEY,   -- UUID
    run_type        VARCHAR NOT NULL,      -- "rss_batch" | "ohlcv_topup"
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ,
    articles_fetched  INTEGER DEFAULT 0,
    articles_new      INTEGER DEFAULT 0,
    articles_tagged   INTEGER DEFAULT 0,
    status          VARCHAR DEFAULT 'running',  -- running | success | error
    error_msg       VARCHAR
);

-- ----------------------------------------------------------------
-- INDEXES  (DuckDB creates these as persistent ART indexes)
-- ----------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_raw_news_trading_date
    ON raw_news (trading_date);

CREATE INDEX IF NOT EXISTS idx_raw_news_fetched_at
    ON raw_news (fetched_at);

CREATE INDEX IF NOT EXISTS idx_tagged_ticker_date
    ON tagged_articles (ticker, tagged_at);

CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date
    ON ohlcv (ticker, date);
"""


def init_db(db_path: Path = DEFAULT_DB) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(DDL)
    con.close()
    print(f"[db_init] Database ready at {db_path}")
    print("[db_init] Tables: raw_news, tagged_articles, ohlcv, pipeline_runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise Phase 1 DuckDB schema")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to .duckdb file")
    args = parser.parse_args()
    init_db(Path(args.db))
