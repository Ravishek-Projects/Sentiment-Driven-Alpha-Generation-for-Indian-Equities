# backtester.py
# Vectorised walk-forward backtester.
# Reads predictions from DuckDB, simulates daily trades, writes
# trades + portfolio_snapshots + benchmark_snapshots back to DuckDB.
#
# Execution model
# ───────────────
# Signal generated at T (after Phase 3 runs at 08:30).
# Entry  : close price on day T   (we "enter" at today's close).
# Exit   : close price on day T+1 (we hold exactly one trading day).
# Cost   : 20 bps round-trip (10 bps entry + 10 bps exit).
#
# Capital model
# ─────────────
# Start with INITIAL_CAPITAL (default ₹10,00,000 = 10 lakhs).
# Each day: allocate position_size × portfolio_value to each LONG/SHORT.
# Remaining cash earns 0% (conservative — can add savings rate later).
#
# Usage (Windows Terminal, from phase4\ directory)
#   python backtester.py                      # backtest all available predictions
#   python backtester.py --capital 500000     # start with ₹5 lakhs
#   python backtester.py --cost-bps 30        # 30 bps round-trip cost

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backtester")

DEFAULT_DB       = Path(__file__).parent.parent / "db" / "alpha.duckdb"
INITIAL_CAPITAL  = 1_000_000.0   # ₹10 lakhs
COST_BPS         = 20            # 20 basis points round-trip
COST_RATE        = COST_BPS / 10_000


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_predictions(db_path: Path) -> pd.DataFrame:
    """Load all predictions that are not yet simulated as trades."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT p.ticker, p.date AS signal_date, p.signal,
                   p.position_size, p.pred_ret, p.z_score
            FROM   predictions p
            WHERE  p.signal != 'FLAT'
            ORDER  BY p.date, p.ticker
        """).df()
    finally:
        con.close()
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
    return df


def _load_prices(db_path: Path) -> pd.DataFrame:
    """Load OHLCV close prices for all Nifty50 tickers."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, date, close, log_return_1d
            FROM   ohlcv
            ORDER  BY ticker, date
        """).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _load_existing_snapshot_dates(db_path: Path) -> set:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT snapshot_date FROM portfolio_snapshots"
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# DuckDB writers  (open → bulk insert → close)
# ---------------------------------------------------------------------------

def _write_trades(trades: list[dict], db_path: Path) -> int:
    if not trades:
        return 0
    df = pd.DataFrame(trades)
    con = duckdb.connect(str(db_path))
    try:
        con.register("_batch_trades", df)
        result = con.execute("""
            INSERT OR REPLACE INTO trades
            SELECT * FROM _batch_trades
        """).fetchone()
        return result[0] if result else 0
    finally:
        con.close()


def _write_snapshots(snapshots: list[dict], db_path: Path) -> int:
    if not snapshots:
        return 0
    df = pd.DataFrame(snapshots)
    con = duckdb.connect(str(db_path))
    try:
        con.register("_batch_snaps", df)
        result = con.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots
            SELECT * FROM _batch_snaps
        """).fetchone()
        return result[0] if result else 0
    finally:
        con.close()


def _write_benchmark(snapshots: list[dict], db_path: Path) -> int:
    if not snapshots:
        return 0
    df = pd.DataFrame(snapshots)
    con = duckdb.connect(str(db_path))
    try:
        con.register("_batch_bench", df)
        result = con.execute("""
            INSERT OR REPLACE INTO benchmark_snapshots
            SELECT * FROM _batch_bench
        """).fetchone()
        return result[0] if result else 0
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------

def build_benchmark(
    prices: pd.DataFrame,
    all_dates: list[date],
    initial_capital: float,
    db_path: Path,
) -> None:
    """
    Build and store Nifty50 equal-weight buy-and-hold benchmark.
    Each of the 50 stocks gets 1/50 of capital at the start.
    """
    existing = _load_existing_snapshot_dates(db_path)
    # Use set of benchmark dates already written
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        bench_existing = {
            r[0] for r in con.execute(
                "SELECT snapshot_date FROM benchmark_snapshots"
            ).fetchall()
        }
    finally:
        con.close()

    # Build equal-weight daily return as mean of all Nifty50 log returns
    nifty_daily = (
        prices.groupby("date")["log_return_1d"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    nifty_daily = nifty_daily[nifty_daily["date"].isin(all_dates)]

    value     = initial_capital
    prev_val  = initial_capital
    snapshots = []
    now = datetime.now(timezone.utc)

    for _, row in nifty_daily.iterrows():
        d = row["date"]
        if d in bench_existing:
            prev_val = value
            # Reconstruct value from last stored benchmark to keep continuity
            continue

        ret   = float(row["log_return_1d"] or 0.0)
        value = prev_val * np.exp(ret)

        snapshots.append({
            "snapshot_date":  d,
            "portfolio_value": round(value, 2),
            "daily_return":    round(ret, 8),
            "cum_return":      round((value / initial_capital) - 1, 8),
            "created_at":      now,
        })
        prev_val = value

    n = _write_benchmark(snapshots, db_path)
    log.info(f"[benchmark] Wrote {n} benchmark snapshot rows")


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    initial_capital: float = INITIAL_CAPITAL,
    cost_bps: int = COST_BPS,
    db_path: Path = DEFAULT_DB,
) -> dict:
    """
    Full walk-forward backtest over all available predictions.

    For every trading day T:
      1. Close any positions opened on T-1 (exit at T's open/close).
      2. Open new positions from today's signals (entry at T's close).
      3. Snapshot portfolio value.

    Returns a summary dict with Sharpe, hit rate, max drawdown, etc.
    """
    cost_rate = cost_bps / 10_000

    predictions = _load_predictions(db_path)
    prices      = _load_prices(db_path)

    if predictions.empty:
        log.error("No predictions found. Run Phase 3 signal_generator.py first.")
        return {}

    # Build a fast price lookup: (ticker, date) → close
    price_lookup: dict[tuple, float] = {
        (r["ticker"], r["date"]): r["close"]
        for _, r in prices.iterrows()
    }

    # Sorted unique signal dates
    all_dates = sorted(predictions["signal_date"].unique())

    existing_snaps = _load_existing_snapshot_dates(db_path)
    now            = datetime.now(timezone.utc)

    portfolio_value  = initial_capital
    cash             = initial_capital
    all_trades:       list[dict] = []
    snapshots:        list[dict] = []
    open_positions:   dict[str, dict] = {}   # trade_id → trade_dict
    daily_pnls:       list[float] = []

    for signal_date in all_dates:
        if signal_date in existing_snaps:
            # Reconstruct portfolio value from stored snapshots to stay consistent
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                row = con.execute("""
                    SELECT portfolio_value, cash
                    FROM   portfolio_snapshots
                    WHERE  snapshot_date = ?
                """, [str(signal_date)]).fetchone()
                if row:
                    portfolio_value = row[0]
                    cash            = row[1]
            finally:
                con.close()
            continue

        day_pnl      = 0.0
        day_costs    = 0.0
        day_turnover = 0.0
        closed_today = []

        # ── Step 1: Close all positions opened on the previous day ──────
        positions_to_close = list(open_positions.values())
        for pos in positions_to_close:
            exit_price = price_lookup.get((pos["ticker"], signal_date))
            if exit_price is None:
                # No price data — skip (keep position open another day)
                continue

            actual_ret = np.log(exit_price / pos["entry_price"]) if pos["entry_price"] > 0 else 0.0
            direction  = 1.0 if pos["direction"] == "LONG" else -1.0
            pnl_gross  = pos["capital_allocated"] * direction * actual_ret
            cost       = pos["capital_allocated"] * cost_rate
            pnl_net    = pnl_gross - cost

            # Return capital to cash
            cash            += pos["capital_allocated"] + pnl_net
            portfolio_value += pnl_net

            day_pnl      += pnl_net
            day_costs    += cost
            day_turnover += pos["capital_allocated"]

            pos.update({
                "exit_date":    signal_date,
                "exit_price":   exit_price,
                "actual_ret":   round(actual_ret, 8),
                "pnl_gross":    round(pnl_gross, 4),
                "pnl_net":      round(pnl_net, 4),
                "transaction_cost": round(cost, 4),
                "status":       "closed",
            })
            all_trades.append(pos)
            closed_today.append(pos["trade_id"])

        for tid in closed_today:
            open_positions.pop(tid, None)

        # ── Step 2: Open new positions from today's signals ─────────────
        day_signals = predictions[predictions["signal_date"] == signal_date]

        for _, sig in day_signals.iterrows():
            if sig["signal"] == "FLAT":
                continue

            entry_price = price_lookup.get((sig["ticker"], signal_date))
            if entry_price is None or entry_price <= 0:
                continue

            # Capital to allocate
            alloc = portfolio_value * sig["position_size"]
            if alloc > cash:
                alloc = cash    # can't allocate more than available cash
            if alloc < 100:     # minimum ₹100 position
                continue

            entry_cost   = alloc * (cost_rate / 2)   # half round-trip at entry
            cash        -= alloc
            day_turnover += alloc
            day_costs    += entry_cost

            trade_id = str(uuid.uuid4())
            trade = {
                "trade_id":          trade_id,
                "ticker":            sig["ticker"],
                "signal_date":       signal_date,
                "entry_date":        signal_date,
                "exit_date":         None,
                "direction":         sig["signal"],
                "entry_price":       round(entry_price, 4),
                "exit_price":        None,
                "position_size":     round(sig["position_size"], 4),
                "capital_allocated": round(alloc, 4),
                "pnl_gross":         None,
                "pnl_net":           None,
                "transaction_cost":  None,
                "pred_ret":          round(sig["pred_ret"], 8),
                "actual_ret":        None,
                "z_score":           round(sig["z_score"], 4),
                "status":            "open",
                "created_at":        now,
            }
            open_positions[trade_id] = trade

        # ── Step 3: Portfolio snapshot ───────────────────────────────────
        # Mark-to-market open positions
        mtm = 0.0
        for pos in open_positions.values():
            cur_price = price_lookup.get((pos["ticker"], signal_date), pos["entry_price"])
            direction = 1.0 if pos["direction"] == "LONG" else -1.0
            ret       = np.log(cur_price / pos["entry_price"]) if pos["entry_price"] > 0 else 0.0
            mtm      += pos["capital_allocated"] * direction * ret

        total_value = cash + sum(p["capital_allocated"] for p in open_positions.values()) + mtm
        prev_value  = portfolio_value if not daily_pnls else (portfolio_value - day_pnl)
        daily_return = (total_value / prev_value - 1) if prev_value > 0 else 0.0
        cum_return   = (total_value / initial_capital) - 1
        daily_pnls.append(day_pnl)
        portfolio_value = total_value

        longs  = sum(1 for p in open_positions.values() if p["direction"] == "LONG")
        shorts = sum(1 for p in open_positions.values() if p["direction"] == "SHORT")

        snapshots.append({
            "snapshot_date":    signal_date,
            "portfolio_value":  round(total_value, 2),
            "cash":             round(cash, 2),
            "daily_pnl":        round(day_pnl, 2),
            "daily_return":     round(daily_return, 8),
            "cum_return":       round(cum_return, 8),
            "open_positions":   len(open_positions),
            "longs":            longs,
            "shorts":           shorts,
            "daily_turnover":   round(day_turnover, 2),
            "transaction_costs":round(day_costs, 2),
            "created_at":       now,
        })

    # ── Write everything to DuckDB ───────────────────────────────────────
    n_trades = _write_trades(all_trades, db_path)
    n_snaps  = _write_snapshots(snapshots, db_path)
    log.info(f"[backtest] Wrote {n_trades} trades, {n_snaps} snapshots")

    # Build benchmark
    build_benchmark(prices, all_dates, initial_capital, db_path)

    # ── Compute summary metrics ──────────────────────────────────────────
    if not daily_pnls:
        return {"error": "No trades simulated"}

    returns = np.array([s["daily_return"] for s in snapshots])
    final_value  = snapshots[-1]["portfolio_value"] if snapshots else initial_capital
    total_return = (final_value / initial_capital) - 1
    ann_return   = float((1 + total_return) ** (252 / max(len(returns), 1)) - 1)
    ann_vol      = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe       = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cum_vals = np.array([s["portfolio_value"] for s in snapshots])
    rolling_max = np.maximum.accumulate(cum_vals)
    drawdowns   = (cum_vals - rolling_max) / rolling_max
    max_dd      = float(drawdowns.min())

    # Hit rate from closed trades
    closed = [t for t in all_trades if t["status"] == "closed" and t["pnl_net"] is not None]
    hit_rate = sum(1 for t in closed if t["pnl_net"] > 0) / max(len(closed), 1)

    # Average win / average loss
    wins   = [t["pnl_net"] for t in closed if t["pnl_net"] > 0]
    losses = [t["pnl_net"] for t in closed if t["pnl_net"] < 0]
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    summary = {
        "initial_capital":    initial_capital,
        "final_value":        round(final_value, 2),
        "total_return_pct":   round(total_return * 100, 2),
        "ann_return_pct":     round(ann_return * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio":       round(sharpe, 4),
        "max_drawdown_pct":   round(max_dd * 100, 2),
        "hit_rate_pct":       round(hit_rate * 100, 2),
        "win_loss_ratio":     round(win_loss_ratio, 2),
        "total_trades":       len(closed),
        "trading_days":       len(snapshots),
    }

    log.info("\n── Backtest Summary ────────────────────────────────────────")
    for k, v in summary.items():
        log.info(f"  {k:<25}: {v}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run walk-forward backtest")
    parser.add_argument("--db",       default=str(DEFAULT_DB))
    parser.add_argument("--capital",  type=float, default=INITIAL_CAPITAL,
                        help="Starting capital in INR (default: 1000000)")
    parser.add_argument("--cost-bps", type=int,   default=COST_BPS,
                        help="Round-trip transaction cost in bps (default: 20)")
    args = parser.parse_args()

    run_backtest(
        initial_capital=args.capital,
        cost_bps=args.cost_bps,
        db_path=Path(args.db),
    )
