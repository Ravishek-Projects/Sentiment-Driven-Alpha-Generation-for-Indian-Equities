# daily_pnl_updater.py
# Post-market-close job (runs at ~16:30 IST).
# Closes all open trades using today's actual close prices,
# and writes a portfolio snapshot for the day.
#
# This is the script that keeps the dashboard's P&L current.
# It is called by scheduler_phase4.py automatically,
# but can also be run manually after market close.
#
# Usage (Windows Terminal, from phase4\ directory)
#   python daily_pnl_updater.py
#   python daily_pnl_updater.py --date 2024-03-28   (re-run for a past date)
#   python daily_pnl_updater.py --dry-run           (print what would happen)

from __future__ import annotations
import sys
import argparse
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))             # Looks in main folder
sys.path.insert(0, str(_root / "PHASE_2")) # Looks in Phase 2
sys.path.insert(0, str(_root / "PHASE_3")) # Looks in Phase 3
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pnl_updater")

DEFAULT_DB      = Path(__file__).parent.parent / "db" / "alpha.duckdb"
INITIAL_CAPITAL = 1_000_000.0
COST_RATE       = 20 / 10_000   # 20 bps round-trip


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _load_open_trades(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT trade_id, ticker, entry_date, direction,
                   entry_price, capital_allocated, position_size,
                   pred_ret, z_score
            FROM   trades
            WHERE  status = 'open'
        """).df()
    finally:
        con.close()
    return df


def _load_close_price(ticker: str, target_date: date, db_path: Path) -> float | None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("""
            SELECT close FROM ohlcv
            WHERE ticker = ? AND date = ?
        """, [ticker, str(target_date)]).fetchone()
        return float(row[0]) if row else None
    finally:
        con.close()


def _load_last_snapshot(db_path: Path) -> dict | None:
    """Get the most recent portfolio snapshot."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("""
            SELECT portfolio_value, cash, snapshot_date
            FROM   portfolio_snapshots
            ORDER  BY snapshot_date DESC
            LIMIT  1
        """).fetchone()
        if row:
            return {"portfolio_value": row[0], "cash": row[1], "snapshot_date": row[2]}
        return None
    finally:
        con.close()


def _snapshot_exists(target_date: date, db_path: Path) -> bool:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("""
            SELECT COUNT(*) FROM portfolio_snapshots WHERE snapshot_date = ?
        """, [str(target_date)]).fetchone()
        return row[0] > 0
    finally:
        con.close()


def _load_new_signals(target_date: date, db_path: Path) -> pd.DataFrame:
    """Load today's signals to open new positions."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, signal, position_size, pred_ret, z_score
            FROM   predictions
            WHERE  date   = ?
              AND  signal != 'FLAT'
        """, [str(target_date)]).df()
    finally:
        con.close()
    return df


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _close_trades_in_db(updates: list[dict], db_path: Path) -> None:
    """Bulk-update trades table to mark positions as closed."""
    if not updates:
        return
    df = pd.DataFrame(updates)
    con = duckdb.connect(str(db_path))
    try:
        con.register("_trade_updates", df)
        con.execute("""
            UPDATE trades t
            SET  exit_date         = u.exit_date,
                 exit_price        = u.exit_price,
                 actual_ret        = u.actual_ret,
                 pnl_gross         = u.pnl_gross,
                 pnl_net           = u.pnl_net,
                 transaction_cost  = u.transaction_cost,
                 status            = 'closed'
            FROM _trade_updates u
            WHERE t.trade_id = u.trade_id
        """)
    finally:
        con.close()
    log.info(f"[updater] Closed {len(updates)} trade(s) in DB")


def _open_new_trades(trades: list[dict], db_path: Path) -> None:
    if not trades:
        return
    df = pd.DataFrame(trades)
    con = duckdb.connect(str(db_path))
    try:
        con.register("_new_trades", df)
        con.execute("""
            INSERT OR IGNORE INTO trades
            SELECT * FROM _new_trades
        """)
    finally:
        con.close()
    log.info(f"[updater] Opened {len(trades)} new trade(s)")


def _write_snapshot(snap: dict, db_path: Path) -> None:
    df = pd.DataFrame([snap])
    con = duckdb.connect(str(db_path))
    try:
        con.register("_snap", df)
        con.execute("INSERT OR REPLACE INTO portfolio_snapshots SELECT * FROM _snap")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Main update function
# ---------------------------------------------------------------------------

def run_daily_update(
    target_date: date | None = None,
    db_path: Path = DEFAULT_DB,
    dry_run: bool = False,
) -> dict:
    """
    1. Close all open trades at today's close price.
    2. Open new positions from today's signals.
    3. Write a portfolio snapshot.

    Returns a summary dict.
    """
    if target_date is None:
        target_date = date.today()

    log.info(f"[updater] Daily update for {target_date}  dry_run={dry_run}")

    if _snapshot_exists(target_date, db_path) and not dry_run:
        log.info(f"[updater] Snapshot for {target_date} already exists — skipping.")
        return {"skipped": True, "date": str(target_date)}

    # ── Load current state ───────────────────────────────────────────────
    open_trades  = _load_open_trades(db_path)
    last_snap    = _load_last_snapshot(db_path)
    new_signals  = _load_new_signals(target_date, db_path)

    portfolio_value = last_snap["portfolio_value"] if last_snap else INITIAL_CAPITAL
    cash            = last_snap["cash"]            if last_snap else INITIAL_CAPITAL

    now    = datetime.now(timezone.utc)
    day_pnl   = 0.0
    day_costs = 0.0
    day_turn  = 0.0
    closed_updates: list[dict] = []

    # ── Step 1: Close all open positions ────────────────────────────────
    for _, trade in open_trades.iterrows():
        exit_price = _load_close_price(trade["ticker"], target_date, db_path)
        if exit_price is None:
            log.warning(f"[updater] No price for {trade['ticker']} on {target_date} — keeping open")
            continue

        direction  = 1.0 if trade["direction"] == "LONG" else -1.0
        actual_ret = np.log(exit_price / trade["entry_price"]) if trade["entry_price"] > 0 else 0.0
        pnl_gross  = trade["capital_allocated"] * direction * actual_ret
        cost       = trade["capital_allocated"] * COST_RATE
        pnl_net    = pnl_gross - cost

        cash            += trade["capital_allocated"] + pnl_net
        portfolio_value += pnl_net
        day_pnl         += pnl_net
        day_costs       += cost
        day_turn        += trade["capital_allocated"]

        log.info(
            f"  CLOSE {trade['direction']:5s} {trade['ticker']:12s} "
            f"entry=₹{trade['entry_price']:.2f}  exit=₹{exit_price:.2f}  "
            f"P&L=₹{pnl_net:+.2f}  ret={actual_ret*100:+.3f}%"
        )

        closed_updates.append({
            "trade_id":       trade["trade_id"],
            "exit_date":      target_date,
            "exit_price":     round(exit_price, 4),
            "actual_ret":     round(actual_ret, 8),
            "pnl_gross":      round(pnl_gross, 4),
            "pnl_net":        round(pnl_net, 4),
            "transaction_cost": round(cost, 4),
        })

    # ── Step 2: Open new positions from today's signals ──────────────────
    new_trade_records: list[dict] = []
    open_longs  = 0
    open_shorts = 0

    for _, sig in new_signals.iterrows():
        entry_price = _load_close_price(sig["ticker"], target_date, db_path)
        if entry_price is None or entry_price <= 0:
            continue

        alloc = portfolio_value * sig["position_size"]
        alloc = min(alloc, cash)
        if alloc < 100:
            continue

        entry_cost  = alloc * (COST_RATE / 2)
        cash       -= alloc
        day_turn   += alloc
        day_costs  += entry_cost

        if sig["signal"] == "LONG":
            open_longs += 1
        else:
            open_shorts += 1

        log.info(
            f"  OPEN  {sig['signal']:5s} {sig['ticker']:12s} "
            f"@ ₹{entry_price:.2f}  alloc=₹{alloc:,.0f}  z={sig['z_score']:.2f}"
        )

        new_trade_records.append({
            "trade_id":          str(uuid.uuid4()),
            "ticker":            sig["ticker"],
            "signal_date":       target_date,
            "entry_date":        target_date,
            "exit_date":         None,
            "direction":         sig["signal"],
            "entry_price":       round(entry_price, 4),
            "exit_price":        None,
            "position_size":     round(sig["position_size"], 4),
            "capital_allocated": round(alloc, 4),
            "pnl_gross":         None,
            "pnl_net":           None,
            "transaction_cost":  None,
            "pred_ret":          round(float(sig["pred_ret"]), 8),
            "actual_ret":        None,
            "z_score":           round(float(sig["z_score"]), 4),
            "status":            "open",
            "created_at":        now,
        })

    # ── Step 3: Write snapshot ───────────────────────────────────────────
    prev_val   = last_snap["portfolio_value"] if last_snap else INITIAL_CAPITAL
    day_return = (portfolio_value / prev_val - 1) if prev_val > 0 else 0.0
    cum_return = (portfolio_value / INITIAL_CAPITAL) - 1
    total_open = len(new_trade_records)

    snapshot = {
        "snapshot_date":    target_date,
        "portfolio_value":  round(portfolio_value, 2),
        "cash":             round(cash, 2),
        "daily_pnl":        round(day_pnl, 2),
        "daily_return":     round(day_return, 8),
        "cum_return":       round(cum_return, 8),
        "open_positions":   total_open,
        "longs":            open_longs,
        "shorts":           open_shorts,
        "daily_turnover":   round(day_turn, 2),
        "transaction_costs":round(day_costs, 2),
        "created_at":       now,
    }

    if not dry_run:
        _close_trades_in_db(closed_updates, db_path)
        _open_new_trades(new_trade_records, db_path)
        _write_snapshot(snapshot, db_path)
    else:
        log.info("[updater] DRY RUN — nothing written to DB")

    log.info(
        f"[updater] Done — portfolio=₹{portfolio_value:,.2f}  "
        f"day_pnl=₹{day_pnl:+,.2f}  "
        f"closed={len(closed_updates)}  opened={len(new_trade_records)}"
    )

    return {
        "date":          str(target_date),
        "portfolio_value": round(portfolio_value, 2),
        "day_pnl":       round(day_pnl, 2),
        "closed":        len(closed_updates),
        "opened":        len(new_trade_records),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily P&L update (close trades + snapshot)")
    parser.add_argument("--date",    help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--db",      default=str(DEFAULT_DB))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_daily_update(
        target_date = date.fromisoformat(args.date) if args.date else None,
        db_path     = Path(args.db),
        dry_run     = args.dry_run,
    )
