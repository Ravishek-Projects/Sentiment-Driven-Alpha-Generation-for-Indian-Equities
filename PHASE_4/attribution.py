# attribution.py
# Decomposes strategy returns into factor exposures via OLS regression.
#
# Model:  R_strategy(t) = alpha + beta_nifty * R_nifty(t)
#                       + beta_momentum * R_mom(t)
#                       + beta_value * R_val(t)
#                       + epsilon(t)
#
# A positive, statistically significant alpha means the strategy generates
# excess return beyond what can be explained by these three factors.
#
# Windows usage:
#   python attribution.py --run-id <run_id>
#   python attribution.py --run-id <run_id> --db C:\projects\phase1\db\alpha.duckdb

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_DB = Path(__file__).parent.parent / "phase1" / "db" / "alpha.duckdb"

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ── Data loading ──────────────────────────────────────────────────────────

def _load_strategy_daily_ret(run_id: str, db_path: Path) -> pd.DataFrame:
    """Daily net P&L as a fraction of initial_capital for the backtest run."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        meta = con.execute("""
            SELECT initial_capital FROM backtest_runs WHERE run_id = ?
        """, [run_id]).fetchone()
        if meta is None:
            raise ValueError(f"run_id {run_id[:8]} not found in backtest_runs")

        cap = float(meta[0])
        df  = con.execute("""
            SELECT signal_date AS date, SUM(net_pnl) AS daily_pnl
            FROM   backtest_trades
            WHERE  run_id = ?
            GROUP  BY signal_date
            ORDER  BY signal_date
        """, [run_id]).df()
    finally:
        con.close()

    df["date"]      = pd.to_datetime(df["date"]).dt.date
    df["daily_ret"] = df["daily_pnl"] / cap
    return df[["date", "daily_ret"]]


def _load_benchmark_daily_ret(strategy: str, db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT date, daily_ret FROM benchmark_returns
            WHERE  strategy = ? ORDER BY date
        """, [strategy]).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.rename(columns={"daily_ret": strategy})


# ── OLS regression ────────────────────────────────────────────────────────

def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple OLS: y = X @ beta.
    Returns (betas, t_stats, p_values).
    X must already include a constant column.
    """
    n, k  = X.shape
    beta  = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    s2    = resid @ resid / max(n - k, 1)
    XtXi  = np.linalg.pinv(X.T @ X)
    se    = np.sqrt(np.diag(XtXi) * s2)
    t_stat = beta / np.where(se > 0, se, np.nan)
    p_val  = 2 * stats.t.sf(np.abs(t_stat), df=max(n - k, 1))
    return beta, t_stat, p_val


def run_attribution(run_id: str, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """
    Regress backtest daily returns on nifty50_ew, momentum, value.
    Stores results in factor_attribution and returns a summary DataFrame.
    """
    strat = _load_strategy_daily_ret(run_id, db_path)
    nifty = _load_benchmark_daily_ret("nifty50_ew", db_path)
    mom   = _load_benchmark_daily_ret("momentum",   db_path)
    val   = _load_benchmark_daily_ret("value",      db_path)

    # Inner-join on date so all series are aligned
    merged = (
        strat
        .merge(nifty, on="date", how="inner")
        .merge(mom,   on="date", how="inner")
        .merge(val,   on="date", how="inner")
        .dropna()
    )

    if len(merged) < 10:
        log.warning(f"[attribution] Only {len(merged)} overlapping dates — "
                    "results may be unreliable. Build benchmarks for a longer period.")

    y = merged["daily_ret"].values
    X = np.column_stack([
        np.ones(len(merged)),           # intercept (alpha)
        merged["nifty50_ew"].values,
        merged["momentum"].values,
        merged["value"].values,
    ])

    betas, t_stats, p_vals = _ols(y, X)

    factor_names = ["alpha", "nifty50", "momentum", "value"]
    rows = []
    for name, b, t, p in zip(factor_names, betas, t_stats, p_vals):
        rows.append({
            "run_id":  run_id,
            "factor":  name,
            "beta":    round(float(b), 8),
            "t_stat":  round(float(t), 4),
            "p_value": round(float(p), 6),
        })

    result = pd.DataFrame(rows)

    # Write to DB
    con = duckdb.connect(str(db_path))
    try:
        con.register("_attr", result)
        con.execute("""
            INSERT OR REPLACE INTO factor_attribution
            SELECT * FROM _attr
        """)
    finally:
        con.close()

    # Print report
    print(f"\n{'='*60}")
    print(f"  FACTOR ATTRIBUTION  run_id={run_id[:8]}")
    print(f"  Observations: {len(merged)} trading days")
    print(f"{'='*60}")
    print(f"  {'Factor':<12} {'Beta':>10} {'T-stat':>8} {'P-value':>10}  Sig")
    print(f"  {'-'*52}")
    for _, row in result.iterrows():
        sig = "***" if row["p_value"] < 0.01 else \
              "**"  if row["p_value"] < 0.05 else \
              "*"   if row["p_value"] < 0.10 else ""
        print(f"  {row['factor']:<12} {row['beta']:>10.6f} "
              f"{row['t_stat']:>8.2f} {row['p_value']:>10.4f}  {sig}")
    print(f"{'='*60}\n")

    ann_alpha = float(betas[0]) * 252
    print(f"  Annualised alpha: {ann_alpha*100:.2f}%")
    if p_vals[0] < 0.05:
        print("  Alpha is statistically significant at 5% level.")
    else:
        print("  Alpha is NOT statistically significant at 5% level.")
    print()

    return result


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor attribution for a backtest run")
    parser.add_argument("--run-id", required=True, help="backtest_runs.run_id")
    parser.add_argument("--db",     default=str(DEFAULT_DB))
    args = parser.parse_args()

    run_attribution(args.run_id, Path(args.db))
