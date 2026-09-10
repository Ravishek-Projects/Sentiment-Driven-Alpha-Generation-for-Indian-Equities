# db_schema_phase2.py
# Adds Phase 2 tables to the existing alpha.duckdb created in Phase 1.
# Safe to re-run — every statement uses IF NOT EXISTS.
#
# Run AFTER Phase 1 db_init.py has already been executed.
#
# Usage:
#   python db_schema_phase2.py
#   python db_schema_phase2.py --db /path/to/alpha.duckdb

import argparse
import duckdb
from pathlib import Path

DEFAULT_DB = Path(__file__).parent.parent / "db" / "alpha.duckdb"

DDL_PHASE2 = """
-- ----------------------------------------------------------------
-- 5. SENTIMENT SCORES
--    One row per (ticker, trading_date).
--    Populated by sentiment_engine.py after each RSS batch.
--    sent_spread = sent_pos - sent_neg  (primary model input)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sentiment_scores (
    ticker          VARCHAR NOT NULL,
    trading_date    DATE    NOT NULL,
    sent_pos        DOUBLE  NOT NULL,   -- mean FinBERT positive score
    sent_neg        DOUBLE  NOT NULL,   -- mean FinBERT negative score
    sent_neu        DOUBLE  NOT NULL,   -- mean FinBERT neutral  score
    sent_spread     DOUBLE  NOT NULL,   -- sent_pos - sent_neg
    news_count      INTEGER NOT NULL,   -- total articles for this ticker+date
    articles_used   INTEGER NOT NULL,   -- articles after truncation (max 32)
    computed_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (ticker, trading_date)
);

-- ----------------------------------------------------------------
-- 6. TECHNICAL FEATURES
--    One row per (ticker, date).
--    Populated by technical_features.py daily after OHLCV top-up.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS technical_features (
    ticker          VARCHAR NOT NULL,
    date            DATE    NOT NULL,
    rsi_14          DOUBLE,             -- 14-day RSI  (0–100)
    ma_20           DOUBLE,             -- 20-day SMA
    ma_50           DOUBLE,             -- 50-day SMA
    price_to_ma20   DOUBLE,             -- close / ma_20 - 1  (mean reversion)
    vol_20d         DOUBLE,             -- 20-day realised vol (annualised)
    rel_volume      DOUBLE,             -- today_vol / avg_vol_20d
    bb_position     DOUBLE,             -- (close - lower_bb) / (upper_bb - lower_bb)
    lag_ret_1d      DOUBLE,             -- log return t-1
    lag_ret_2d      DOUBLE,             -- log return t-2
    computed_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- ----------------------------------------------------------------
-- 7. FEATURE MATRIX  (the model's training / inference table)
--    One row per (ticker, date) — joins sentiment + technical.
--    target_ret_1d = next day's log return (filled after market close).
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_matrix (
    ticker          VARCHAR NOT NULL,
    date            DATE    NOT NULL,

    -- sentiment block (5 features)
    sent_pos        DOUBLE,
    sent_neg        DOUBLE,
    sent_neu        DOUBLE,
    sent_spread     DOUBLE,
    news_count      DOUBLE,             -- log1p-scaled in assembler

    -- technical block (7 features)
    rsi_14          DOUBLE,
    ma_20           DOUBLE,
    ma_50           DOUBLE,
    price_to_ma20   DOUBLE,
    vol_20d         DOUBLE,
    rel_volume      DOUBLE,
    bb_position     DOUBLE,
    -- lag returns (2 features)
    lag_ret_1d      DOUBLE,
    lag_ret_2d      DOUBLE,

    -- day-of-week one-hot (5 features)  total = 19 features
    dow_mon         DOUBLE,
    dow_tue         DOUBLE,
    dow_wed         DOUBLE,
    dow_thu         DOUBLE,
    dow_fri         DOUBLE,

    -- target (filled after close, NULL during market hours)
    target_ret_1d   DOUBLE,

    computed_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- ----------------------------------------------------------------
-- INDEXES for Phase 2 tables
-- ----------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date
    ON sentiment_scores (ticker, trading_date);

CREATE INDEX IF NOT EXISTS idx_tech_ticker_date
    ON technical_features (ticker, date);

CREATE INDEX IF NOT EXISTS idx_feat_ticker_date
    ON feature_matrix (ticker, date);

CREATE INDEX IF NOT EXISTS idx_feat_date
    ON feature_matrix (date);
"""


def extend_schema(db_path: Path = DEFAULT_DB) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run 'python db_init.py' (Phase 1) first."
        )
    con = duckdb.connect(str(db_path))
    con.execute(DDL_PHASE2)
    con.close()
    print(f"[schema_phase2] Phase 2 tables added to {db_path}")
    print("[schema_phase2] New tables: sentiment_scores, technical_features, feature_matrix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Phase 2 tables to alpha.duckdb")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    extend_schema(Path(args.db))
