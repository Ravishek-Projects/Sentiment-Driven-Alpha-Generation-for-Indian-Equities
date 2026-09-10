# db_schema_phase4.py
# Adds Phase 4 tables to alpha.duckdb — the P&L accounting layer.
# Run ONCE, after Phases 1-3 schemas are in place.
#
# Usage (Windows Terminal, from phase4\ directory)
#   python db_schema_phase4.py
#   python db_schema_phase4.py --db ..\phase1\db\alpha.duckdb

import argparse
import duckdb
from pathlib import Path
import sys
from pathlib import Path

# --- FIX: Tell Python where to find the other phase folders ---
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))             # Looks in main folder
sys.path.insert(0, str(_root / "PHASE_2")) # Looks in Phase 2
sys.path.insert(0, str(_root / "PHASE_3")) # Looks in Phase 3
# --------------------------------------------------------------
DEFAULT_DB = Path(__file__).parent.parent / "db" / "alpha.duckdb"

DDL = """
-- ----------------------------------------------------------------
-- TRADES
-- One row per trade (entry T, exit T+1 by default).
-- status: 'open' while position is held, 'closed' after exit.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    trade_id            VARCHAR PRIMARY KEY,
    ticker              VARCHAR NOT NULL,
    signal_date         DATE    NOT NULL,
    entry_date          DATE    NOT NULL,
    exit_date           DATE,
    direction           VARCHAR NOT NULL,      -- LONG | SHORT
    entry_price         DOUBLE  NOT NULL,
    exit_price          DOUBLE,
    position_size       DOUBLE  NOT NULL,      -- fraction of portfolio
    capital_allocated   DOUBLE  NOT NULL,      -- INR value at entry
    pnl_gross           DOUBLE,
    pnl_net             DOUBLE,
    transaction_cost    DOUBLE,
    pred_ret            DOUBLE,
    actual_ret          DOUBLE,
    z_score             DOUBLE,
    status              VARCHAR DEFAULT 'open',
    created_at          TIMESTAMPTZ NOT NULL
);

-- ----------------------------------------------------------------
-- PORTFOLIO_SNAPSHOTS
-- End-of-day portfolio state — equity curve lives here.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_date       DATE PRIMARY KEY,
    portfolio_value     DOUBLE NOT NULL,
    cash                DOUBLE NOT NULL,
    daily_pnl           DOUBLE NOT NULL,
    daily_return        DOUBLE NOT NULL,
    cum_return          DOUBLE NOT NULL,
    open_positions      INTEGER NOT NULL,
    longs               INTEGER NOT NULL,
    shorts              INTEGER NOT NULL,
    daily_turnover      DOUBLE NOT NULL,
    transaction_costs   DOUBLE NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL
);

-- ----------------------------------------------------------------
-- BENCHMARK_SNAPSHOTS
-- Nifty50 equal-weight buy-and-hold for comparison.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_snapshots (
    snapshot_date       DATE PRIMARY KEY,
    portfolio_value     DOUBLE NOT NULL,
    daily_return        DOUBLE NOT NULL,
    cum_return          DOUBLE NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker_date ON trades (ticker, signal_date);
CREATE INDEX IF NOT EXISTS idx_trades_status      ON trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_exit        ON trades (exit_date);
CREATE INDEX IF NOT EXISTS idx_snap_date          ON portfolio_snapshots (snapshot_date);
"""


def extend_schema(db_path: Path = DEFAULT_DB) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}. Run Phase 1 db_init.py first.")
    con = duckdb.connect(str(db_path))
    con.execute(DDL)
    con.close()
    print(f"[schema_p4] Phase 4 tables ready in {db_path}")
    print("[schema_p4] Tables: trades, portfolio_snapshots, benchmark_snapshots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Phase 4 tables to alpha.duckdb")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    extend_schema(Path(args.db))
