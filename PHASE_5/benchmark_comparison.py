# benchmark_comparison.py
# Compares the sentiment strategy against three standard benchmarks:
#   1. Nifty50 equal-weight buy-and-hold
#   2. Pure momentum factor (6-month return, skip 1 month)
#   3. Pure value factor (price / 52-week high, low = cheap)
#
# Also decomposes alpha using Fama-French-style factor regression:
#   strategy_return = α + β_market × market + β_mom × momentum + ε
#   The α intercept is the true sentiment-driven excess return.
#
# Usage
#   python benchmark_comparison.py
#   python benchmark_comparison.py --days 180

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config_phase5 import (
    DB_PATH, EVAL_DIR,
    MOMENTUM_LOOKBACK, MOMENTUM_SKIP,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_portfolio_returns(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT snapshot_date AS date, daily_return AS strategy_ret
            FROM   portfolio_snapshots
            ORDER  BY snapshot_date
        """).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_ohlcv(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, date, close, log_return_1d
            FROM   ohlcv
            ORDER  BY ticker, date
        """).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Factor construction
# ---------------------------------------------------------------------------

def build_market_factor(ohlcv: pd.DataFrame) -> pd.Series:
    """Nifty50 equal-weight daily return = mean of all 50 stock returns."""
    mkt = (
        ohlcv.groupby("date")["log_return_1d"]
        .mean()
        .rename("market_ret")
    )
    return mkt


def build_momentum_factor(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional momentum: each day, long top-decile 6-month returners,
    short bottom-decile. Factor return = long_avg - short_avg.
    """
    ohlcv = ohlcv.sort_values(["ticker", "date"])

    # 6-month (126-day) cumulative return, skip last month (21 days)
    ohlcv["mom"] = (
        ohlcv.groupby("ticker")["close"]
        .transform(lambda s: np.log(s.shift(MOMENTUM_SKIP) / s.shift(MOMENTUM_LOOKBACK)))
    )

    mom_factor = []
    for date_val, grp in ohlcv.dropna(subset=["mom"]).groupby("date"):
        if len(grp) < 10:
            continue
        grp   = grp.sort_values("mom")
        n     = max(1, len(grp) // 10)
        longs  = grp.tail(n)["log_return_1d"].mean()
        shorts = grp.head(n)["log_return_1d"].mean()
        mom_factor.append({"date": date_val, "momentum_ret": longs - shorts})

    return pd.DataFrame(mom_factor).set_index("date")["momentum_ret"]


def build_value_factor(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Value factor: price / 52-week high (low ratio = cheap = value stock).
    Long bottom-decile (cheapest), short top-decile.
    """
    ohlcv = ohlcv.sort_values(["ticker", "date"])

    # 52-week high
    ohlcv["hi_52w"] = (
        ohlcv.groupby("ticker")["close"]
        .transform(lambda s: s.rolling(252, min_periods=60).max())
    )
    ohlcv["value_score"] = ohlcv["close"] / ohlcv["hi_52w"]

    val_factor = []
    for date_val, grp in ohlcv.dropna(subset=["value_score"]).groupby("date"):
        if len(grp) < 10:
            continue
        grp   = grp.sort_values("value_score")
        n     = max(1, len(grp) // 10)
        longs  = grp.head(n)["log_return_1d"].mean()   # cheapest = value
        shorts = grp.tail(n)["log_return_1d"].mean()    # most expensive = growth
        val_factor.append({"date": date_val, "value_ret": longs - shorts})

    return pd.DataFrame(val_factor).set_index("date")["value_ret"]


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _perf(returns: pd.Series, label: str) -> dict:
    r  = returns.dropna().values
    if len(r) < 5:
        return {"label": label}
    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum     = float(np.expm1(np.nansum(r)))   # total return from log returns

    # Max drawdown
    vals  = np.exp(np.nancumsum(r))
    peak  = np.maximum.accumulate(vals)
    mdd   = float(((vals - peak) / peak).min())

    return {
        "label":        label,
        "total_return": round(cum * 100, 2),
        "ann_return":   round(ann_ret * 100, 2),
        "ann_vol":      round(ann_vol * 100, 2),
        "sharpe":       round(sharpe, 4),
        "max_dd":       round(mdd * 100, 2),
        "n_days":       len(r),
    }


# ---------------------------------------------------------------------------
# Factor regression (alpha decomposition)
# ---------------------------------------------------------------------------

def factor_regression(
    strategy: pd.Series,
    market:   pd.Series,
    momentum: pd.Series,
    value:    pd.Series,
) -> dict:
    """
    OLS: strategy_ret = α + β_mkt × mkt + β_mom × mom + β_val × val + ε
    α is the true sentiment alpha after removing factor exposures.
    """
    combined = pd.concat([strategy, market, momentum, value], axis=1).dropna()
    combined.columns = ["strategy", "market", "momentum", "value"]

    if len(combined) < 20:
        return {"alpha": np.nan, "note": "Insufficient overlap between strategy and factors"}

    X = combined[["market", "momentum", "value"]].values
    y = combined["strategy"].values

    # Add intercept
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)

    alpha   = float(beta[0]) * 252   # annualise the daily alpha
    b_mkt   = float(beta[1])
    b_mom   = float(beta[2])
    b_val   = float(beta[3])

    # t-stats and p-values via OLS residual variance
    y_hat   = X_aug @ beta
    e       = y - y_hat
    n, k    = X_aug.shape
    s2      = np.dot(e, e) / (n - k)
    cov_b   = s2 * np.linalg.pinv(X_aug.T @ X_aug)
    se      = np.sqrt(np.diag(cov_b))
    t_alpha = float(beta[0] / se[0]) if se[0] > 0 else np.nan
    p_alpha = float(2 * stats.t.sf(abs(t_alpha), df=n - k))
    r2      = float(1 - np.dot(e, e) / np.dot(y - y.mean(), y - y.mean()))

    return {
        "alpha_daily":   round(float(beta[0]), 8),
        "alpha_ann_pct": round(alpha * 100, 4),
        "beta_market":   round(b_mkt, 4),
        "beta_momentum": round(b_mom, 4),
        "beta_value":    round(b_val, 4),
        "t_stat_alpha":  round(t_alpha, 4),
        "p_value_alpha": round(p_alpha, 4),
        "r_squared":     round(r2, 4),
        "n_obs":         n,
        "alpha_significant": p_alpha < 0.05,
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    db_path: Path = DB_PATH,
    days: int | None = None,
) -> dict:
    """
    Build all benchmarks, compute performance metrics, run factor regression.
    Returns and saves a results dict.
    """
    port   = _load_portfolio_returns(db_path)
    ohlcv  = _load_ohlcv(db_path)

    if port.empty:
        raise RuntimeError("No portfolio_snapshots found. Run Phase 4 backtester first.")

    if days:
        cutoff = port["date"].max() - pd.Timedelta(days=days)
        port   = port[port["date"] >= cutoff]
        ohlcv  = ohlcv[ohlcv["date"] >= cutoff]

    port = port.set_index("date")["strategy_ret"]

    # Build factors
    log.info("[comp] Building market factor ...")
    mkt = build_market_factor(ohlcv)

    log.info("[comp] Building momentum factor ...")
    mom = build_momentum_factor(ohlcv)

    log.info("[comp] Building value factor ...")
    val = build_value_factor(ohlcv)

    # Align all to common dates
    common = port.index.intersection(mkt.index)
    port_c = port.reindex(common)
    mkt_c  = mkt.reindex(common)
    mom_c  = mom.reindex(common).fillna(0)
    val_c  = val.reindex(common).fillna(0)

    # Performance table
    perf_results = {
        "strategy": _perf(port_c, "Sentiment strategy"),
        "market":   _perf(mkt_c,  "Nifty50 equal-weight"),
        "momentum": _perf(mom_c,  "Pure momentum"),
        "value":    _perf(val_c,  "Pure value"),
    }

    log.info("\n── Performance comparison ──────────────────────────────────")
    headers = ["Label", "Total ret%", "Ann ret%", "Ann vol%", "Sharpe", "Max DD%"]
    log.info(f"  {'Strategy':<28} {'Total%':>9} {'Ann%':>8} {'Vol%':>8} {'Sharpe':>8} {'MaxDD%':>8}")
    for k, p in perf_results.items():
        log.info(
            f"  {p.get('label',''):28} "
            f"{p.get('total_return', 0):>9.2f} "
            f"{p.get('ann_return',  0):>8.2f} "
            f"{p.get('ann_vol',     0):>8.2f} "
            f"{p.get('sharpe',      0):>8.4f} "
            f"{p.get('max_dd',      0):>8.2f}"
        )

    # Factor regression (alpha decomposition)
# Factor regression (alpha decomposition)
    log.info("\n[comp] Running factor regression ...")
    reg = factor_regression(port_c, mkt_c, mom_c, val_c)
    
    # --- FIX: Safe string formatting for missing data ---
    a_ann = reg.get('alpha_ann_pct')
    log.info(f"  Alpha (ann): {f'{a_ann:.4f}%' if isinstance(a_ann, (int, float)) else 'N/A'}")
    
    t_stat = reg.get('t_stat_alpha')
    log.info(f"  t-stat α:    {f'{t_stat:.4f}' if isinstance(t_stat, (int, float)) else 'N/A'}")
    
    p_val = reg.get('p_value_alpha')
    log.info(f"  p-value α:   {f'{p_val:.4f}' if isinstance(p_val, (int, float)) else 'N/A'}")
    
    log.info(f"  Significant: {reg.get('alpha_significant', False)}")
    # ----------------------------------------------------

    output = {
        "performance": perf_results,
        "factor_regression": reg,
        "date_range": {
            "start": str(common.min().date()) if len(common) else "N/A",
            "end":   str(common.max().date()) if len(common) else "N/A",
            "n_days": len(common),
        },
    }

    out_path = EVAL_DIR / "benchmark_comparison.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\n[comp] Results saved → {out_path}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy vs benchmark comparison")
    parser.add_argument("--db",   default=str(DB_PATH))
    parser.add_argument("--days", type=int, default=None,
                        help="Limit to last N days (default: all history)")
    args = parser.parse_args()
    run_comparison(Path(args.db), args.days)
