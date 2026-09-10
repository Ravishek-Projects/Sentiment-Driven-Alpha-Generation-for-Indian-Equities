# report.py
# Generates a full performance comparison report:
#   Strategy vs Nifty50 vs Momentum vs Value
#   Prints to terminal and exports to a CSV file.
#
# Windows usage:
#   python report.py --run-id <run_id>
#   python report.py --run-id <run_id> --export results\my_report.csv
#   python report.py --list          (list all backtest run IDs)

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import duckdb
import numpy as np
import pandas as pd

DEFAULT_DB = Path(__file__).parent.parent / "phase1" / "db" / "alpha.duckdb"


# ── Helpers ───────────────────────────────────────────────────────────────

def _list_runs(db_path: Path) -> None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT run_id, run_name, strategy, start_date, end_date,
                   total_trades, sharpe_ratio, hit_rate_pct, total_net_pnl
            FROM   backtest_runs
            ORDER  BY created_at DESC
        """).df()
    finally:
        con.close()
    if df.empty:
        print("No backtest runs found. Run: python backtester.py --name <name>")
        return
    pd.set_option("display.max_colwidth", 20)
    print("\n" + df.to_string(index=False))


def _load_run(run_id: str, db_path: Path) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("""
            SELECT * FROM backtest_runs WHERE run_id = ?
        """, [run_id]).df()
    finally:
        con.close()
    if row.empty:
        raise ValueError(f"run_id {run_id} not found.")
    return row.iloc[0].to_dict()


def _bench_stats(strategy: str, start: str, end: str, db_path: Path) -> dict:
    """Compute annualised return, Sharpe, max drawdown for a benchmark."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT date, daily_ret FROM benchmark_returns
            WHERE  strategy = ? AND date BETWEEN ? AND ?
            ORDER  BY date
        """, [strategy, start, end]).df()
    finally:
        con.close()

    if df.empty:
        return {"strategy": strategy, "ann_ret": "N/A", "sharpe": "N/A",
                "max_dd": "N/A", "total_ret": "N/A"}

    rets  = df["daily_ret"].values
    cum   = np.cumsum(rets)
    n     = len(rets)

    # Annualised return from cumulative log return
    total_log_ret = cum[-1]
    ann_ret = (np.exp(total_log_ret * 252 / n) - 1) * 100

    # Sharpe
    rf = 0.065 / 252
    excess = rets - rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else float("nan")

    # Max drawdown
    wealth = np.exp(np.cumsum(rets))
    peak   = np.maximum.accumulate(wealth)
    dd     = (wealth - peak) / peak
    max_dd = float(dd.min()) * 100

    return {
        "strategy":   strategy,
        "ann_ret_pct": round(ann_ret, 2),
        "sharpe":     round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "total_ret_pct": round((np.exp(total_log_ret) - 1) * 100, 2),
    }


# ── Main report ───────────────────────────────────────────────────────────

def generate_report(run_id: str, db_path: Path = DEFAULT_DB, export: str | None = None) -> None:
    run   = _load_run(run_id, db_path)
    start = str(run["start_date"])
    end   = str(run["end_date"])

    # Benchmark stats over same period
    bench_nifty = _bench_stats("nifty50_ew", start, end, db_path)
    bench_mom   = _bench_stats("momentum",   start, end, db_path)
    bench_val   = _bench_stats("value",      start, end, db_path)

    # Load attribution if available
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        attr_df = con.execute("""
            SELECT factor, beta, t_stat, p_value
            FROM   factor_attribution WHERE run_id = ?
        """, [run_id]).df()
    finally:
        con.close()

    # ── Print ──
    W = 62
    print("\n" + "=" * W)
    print(f"  PHASE 4 BACKTEST REPORT")
    print(f"  Run: {run['run_name']}  [{run_id[:8]}]")
    print(f"  Strategy: {run['strategy'].upper()}  |  Period: {start} → {end}")
    print("=" * W)

    # Strategy metrics
    print(f"\n  {'Metric':<30} {'Strategy':>12} {'Nifty50 EW':>12} {'Momentum':>10} {'Value':>8}")
    print(f"  {'-'*(W-2)}")

    def fmt(val, suffix=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "    N/A"
        return f"{val:>11.2f}{suffix}"

    rows_data = [
        ("Annualised return (%)",   run.get("ann_return_pct"),
         bench_nifty.get("ann_ret_pct"), bench_mom.get("ann_ret_pct"),
         bench_val.get("ann_ret_pct")),
        ("Sharpe ratio",            run.get("sharpe_ratio"),
         bench_nifty.get("sharpe"), bench_mom.get("sharpe"),
         bench_val.get("sharpe")),
        ("Max drawdown (%)",        run.get("max_drawdown_pct"),
         bench_nifty.get("max_dd_pct"), bench_mom.get("max_dd_pct"),
         bench_val.get("max_dd_pct")),
        ("Calmar ratio",            run.get("calmar_ratio"), None, None, None),
        ("Hit rate (%)",            run.get("hit_rate_pct"), None, None, None),
        ("Win/loss ratio",          run.get("win_loss_ratio"), None, None, None),
        ("Total trades",            run.get("total_trades"), None, None, None),
        ("Net P&L (₹)",             run.get("total_net_pnl"), None, None, None),
    ]

    for label, strat_v, n50_v, mom_v, val_v in rows_data:
        strat_s = fmt(strat_v)
        n50_s   = fmt(n50_v)  if n50_v  is not None else "         —"
        mom_s   = fmt(mom_v)  if mom_v  is not None else "        —"
        val_s   = fmt(val_v)  if val_v  is not None else "      —"
        print(f"  {label:<30} {strat_s} {n50_s} {mom_s} {val_s}")

    # Factor attribution block
    if not attr_df.empty:
        print(f"\n  {'─'*(W-2)}")
        print(f"  FACTOR ATTRIBUTION  (OLS regression on daily returns)")
        print(f"  {'Factor':<14} {'Beta':>10} {'T-stat':>8} {'P-value':>10}  Sig")
        for _, r in attr_df.iterrows():
            sig = "***" if r["p_value"] < 0.01 else \
                  "**"  if r["p_value"] < 0.05 else \
                  "*"   if r["p_value"] < 0.10 else ""
            print(f"  {r['factor']:<14} {r['beta']:>10.6f} "
                  f"{r['t_stat']:>8.2f} {r['p_value']:>10.4f}  {sig}")
        alpha_row = attr_df[attr_df["factor"] == "alpha"]
        if not alpha_row.empty:
            ann_alpha = float(alpha_row.iloc[0]["beta"]) * 252 * 100
            p_alpha   = float(alpha_row.iloc[0]["p_value"])
            print(f"\n  Annualised alpha: {ann_alpha:.2f}%  "
                  f"({'significant' if p_alpha < 0.05 else 'not significant'} at 5%)")

    print(f"\n{'='*W}\n")

    # ── Export CSV ──
    if export:
        export_path = Path(export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        rows_csv = [
            ["metric", "strategy", "nifty50_ew", "momentum", "value"],
        ]
        for label, sv, nv, mv, vv in rows_data:
            rows_csv.append([label, sv, nv, mv, vv])
        if not attr_df.empty:
            rows_csv.append([])
            rows_csv.append(["factor", "beta", "t_stat", "p_value", ""])
            for _, r in attr_df.iterrows():
                rows_csv.append([r["factor"], r["beta"], r["t_stat"], r["p_value"], ""])
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows_csv)
        print(f"  Report exported → {export_path}")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 backtest report")
    parser.add_argument("--run-id", default=None, help="Backtest run_id to report on")
    parser.add_argument("--list",   action="store_true", help="List all backtest runs")
    parser.add_argument("--export", default=None, help="Export report to CSV path")
    parser.add_argument("--db",     default=str(DEFAULT_DB))
    args = parser.parse_args()

    if args.list:
        _list_runs(Path(args.db))
    elif args.run_id:
        generate_report(args.run_id, Path(args.db), args.export)
    else:
        parser.print_help()
