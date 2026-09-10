# metrics.py
# Financial performance metrics library.
# Pure functions — no DuckDB access, no side effects.
# Used by the dashboard, backtester, and report generator.

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (assumes risk-free rate = 0)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    mu  = r.mean() * periods_per_year
    sig = r.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / sig) if sig > 0 else 0.0


def sortino_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    mu      = r.mean() * periods_per_year
    neg_ret = r[r < 0]
    if len(neg_ret) < 2:
        return float("inf")
    downside = neg_ret.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / downside) if downside > 0 else 0.0


def max_drawdown(values: pd.Series | np.ndarray) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.12 = -12%)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return 0.0
    peak   = np.maximum.accumulate(v)
    dd     = (v - peak) / peak
    return float(dd.min())


def calmar_ratio(ann_return: float, max_dd: float) -> float:
    """Calmar = annualised return / |max drawdown|."""
    return float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0


def hit_rate(pnls: pd.Series | np.ndarray) -> float:
    """Fraction of trades with positive net P&L."""
    p = np.asarray(pnls, dtype=float)
    p = p[~np.isnan(p)]
    return float((p > 0).mean()) if len(p) > 0 else 0.0


def win_loss_ratio(pnls: pd.Series | np.ndarray) -> float:
    """Mean win / |mean loss|. Returns 0 if no losing trades."""
    p = np.asarray(pnls, dtype=float)
    wins   = p[p > 0]
    losses = p[p < 0]
    if len(losses) == 0 or len(wins) == 0:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def annualised_return(total_return: float, n_days: int, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from total return over n_days."""
    if n_days <= 0:
        return 0.0
    return float((1 + total_return) ** (periods_per_year / n_days) - 1)


def rolling_sharpe(returns: pd.Series, window: int = 30) -> pd.Series:
    """Rolling Sharpe ratio over a sliding window of trading days."""
    mu  = returns.rolling(window).mean() * 252
    sig = returns.rolling(window).std(ddof=1) * np.sqrt(252)
    return (mu / sig).fillna(0.0)


def information_coefficient(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Pearson correlation between predicted and actual returns."""
    mask = ~(np.isnan(predicted) | np.isnan(actual))
    if mask.sum() < 2:
        return 0.0
    return float(np.corrcoef(predicted[mask], actual[mask])[0, 1])


# ---------------------------------------------------------------------------
# Full summary — called by dashboard and report generator
# ---------------------------------------------------------------------------

def compute_full_metrics(
    snapshot_df: pd.DataFrame,
    trades_df:   pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    Given portfolio_snapshots and closed trades DataFrames,
    compute the full set of metrics used in the dashboard.

    Parameters
    ----------
    snapshot_df    : DataFrame from portfolio_snapshots table
    trades_df      : DataFrame from trades table (closed trades only)
    initial_capital: starting capital in INR

    Returns
    -------
    dict with all metrics as plain Python scalars
    """
    if snapshot_df.empty:
        return _empty_metrics()

    snap = snapshot_df.sort_values("snapshot_date").copy()
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])

    daily_rets = snap["daily_return"].values
    port_vals  = snap["portfolio_value"].values

    final_val    = float(port_vals[-1]) if len(port_vals) else initial_capital
    total_ret    = (final_val / initial_capital) - 1
    n_days       = len(snap)
    ann_ret      = annualised_return(total_ret, n_days)
    ann_vol      = float(np.std(daily_rets, ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0
    sharpe       = sharpe_ratio(daily_rets)
    sortino      = sortino_ratio(daily_rets)
    mdd          = max_drawdown(port_vals)
    calmar       = calmar_ratio(ann_ret, mdd)

    # P&L aggregates
    cum_pnl      = float(final_val - initial_capital)
    today_pnl    = float(snap["daily_pnl"].iloc[-1]) if not snap.empty else 0.0
    today_ret    = float(snap["daily_return"].iloc[-1]) if not snap.empty else 0.0

    # Trade stats
    closed = trades_df[trades_df["status"] == "closed"].copy() if not trades_df.empty else pd.DataFrame()
    n_closed    = len(closed)
    hr          = hit_rate(closed["pnl_net"].values) if n_closed > 0 else 0.0
    wl_ratio    = win_loss_ratio(closed["pnl_net"].values) if n_closed > 0 else 0.0
    total_costs = float(snap["transaction_costs"].sum())

    # Current open positions
    open_pos    = int(snap["open_positions"].iloc[-1]) if not snap.empty else 0
    open_longs  = int(snap["longs"].iloc[-1]) if not snap.empty else 0
    open_shorts = int(snap["shorts"].iloc[-1]) if not snap.empty else 0

    return {
        # Portfolio values
        "portfolio_value":    round(final_val, 2),
        "initial_capital":    round(initial_capital, 2),
        "cum_pnl":            round(cum_pnl, 2),
        "today_pnl":          round(today_pnl, 2),
        "today_return_pct":   round(today_ret * 100, 4),
        # Returns
        "total_return_pct":   round(total_ret * 100, 2),
        "ann_return_pct":     round(ann_ret * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        # Risk metrics
        "sharpe_ratio":       round(sharpe, 4),
        "sortino_ratio":      round(sortino, 4),
        "max_drawdown_pct":   round(mdd * 100, 2),
        "calmar_ratio":       round(calmar, 4),
        # Trade stats
        "total_trades":       n_closed,
        "hit_rate_pct":       round(hr * 100, 2),
        "win_loss_ratio":     round(wl_ratio, 2),
        "total_costs_inr":    round(total_costs, 2),
        # Open positions
        "open_positions":     open_pos,
        "open_longs":         open_longs,
        "open_shorts":        open_shorts,
        # Metadata
        "trading_days":       n_days,
    }


def _empty_metrics() -> dict:
    keys = [
        "portfolio_value", "initial_capital", "cum_pnl", "today_pnl",
        "today_return_pct", "total_return_pct", "ann_return_pct",
        "ann_volatility_pct", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "calmar_ratio", "total_trades", "hit_rate_pct",
        "win_loss_ratio", "total_costs_inr", "open_positions",
        "open_longs", "open_shorts", "trading_days",
    ]
    return {k: 0.0 for k in keys}
