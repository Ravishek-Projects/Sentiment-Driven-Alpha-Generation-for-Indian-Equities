# verify_phase5.py
# End-to-end checks for Phase 5 components.
# Uses synthetic data — no trained models or real DB needed.
#
# Usage (Windows Terminal, from phase5\ directory)
#   python verify_phase5.py

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "PHASE_3"))
PASS = "[OK]"
FAIL = "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        try:
            fn()
            results.append((name, True, ""))
            print(f"  {PASS}  {name}")
        except Exception as exc:
            results.append((name, False, str(exc)))
            print(f"  {FAIL}  {name}")
            print(f"         {exc}")
        return fn
    return decorator


print("\n── Phase 5 verification ──────────────────────────────────────")

# ── 1. Imports ────────────────────────────────────────────────────────────
print("\n[Imports]")

@check("config_phase5 importable")
def _(): import config_phase5

@check("walk_forward_eval importable")
def _(): import walk_forward_eval

@check("statistical_tests importable")
def _(): import statistical_tests

@check("benchmark_comparison importable")
def _(): import benchmark_comparison

@check("model_monitor importable")
def _(): import model_monitor

@check("incremental_retrain importable")
def _(): import incremental_retrain

@check("report_generator importable")
def _(): import report_generator

@check("scipy available")
def _(): from scipy import stats; from scipy.special import rel_entr

@check("matplotlib available (optional)")
def _():
    try:
        import matplotlib
        print("         (matplotlib found)")
    except ImportError:
        print("         (matplotlib not installed — charts in report will be skipped)")

# ── 2. Statistical tests ───────────────────────────────────────────────────
print("\n[Statistical tests — unit tests]")

@check("bootstrap_ci on known data")
def _():
    from statistical_tests import bootstrap_ci
    data = np.random.default_rng(0).normal(0.001, 0.01, 252)
    ci   = bootstrap_ci(data, np.mean, n_samples=500)
    assert "lower" in ci and "upper" in ci
    assert ci["lower"] < ci["observed"] < ci["upper"], f"CI doesn't bracket observed: {ci}"

@check("bootstrap_ci CI width narrows with more data")
def _():
    from statistical_tests import bootstrap_ci
    rng    = np.random.default_rng(1)
    small  = bootstrap_ci(rng.normal(0, 1, 30),  np.mean, n_samples=200)
    large  = bootstrap_ci(rng.normal(0, 1, 300), np.mean, n_samples=200)
    width_small = small["upper"] - small["lower"]
    width_large = large["upper"] - large["lower"]
    assert width_large < width_small, "Larger sample should give narrower CI"

@check("deflated_sharpe_ratio — higher n_trials lowers DSR")
def _():
    from statistical_tests import deflated_sharpe_ratio
    dsr1 = deflated_sharpe_ratio(0.1, n_trials=1,  n_observations=252)
    dsr4 = deflated_sharpe_ratio(0.1, n_trials=10, n_observations=252)
    assert dsr4 < dsr1, f"More trials should reduce DSR: {dsr4:.4f} >= {dsr1:.4f}"

@check("deflated_sharpe_ratio output in [0, 1]")
def _():
    from statistical_tests import deflated_sharpe_ratio
    dsr = deflated_sharpe_ratio(2.0, n_trials=4, n_observations=252)
    assert 0.0 <= dsr <= 1.0, f"DSR out of range: {dsr}"

@check("diebold_mariano_test — identical errors → p ≈ 1 (no difference)")
def _():
    from statistical_tests import diebold_mariano_test
    e = np.random.default_rng(42).normal(0, 0.01, 100)
    dm = diebold_mariano_test(e, e)   # same errors → no difference
    assert dm["p_value"] > 0.05, f"Expected p > 0.05 for identical errors, got {dm['p_value']}"

@check("diebold_mariano_test — clearly better model detected")
def _():
    from statistical_tests import diebold_mariano_test
    rng  = np.random.default_rng(7)
    good = rng.normal(0, 0.001, 200)    # very small errors
    bad  = rng.normal(0, 0.050, 200)    # large errors
    dm   = diebold_mariano_test(good, bad)
    assert dm["dm_stat"] < 0, "Good model should have negative DM stat (lower loss)"

@check("paired_hit_rate_test — detects significant difference")
def _():
    from statistical_tests import paired_hit_rate_test
    a = np.ones(30) * 0.60   # model A: 60% hit rate
    b = np.ones(30) * 0.50   # model B: 50% hit rate
    res = paired_hit_rate_test(a, b, "A", "B")
    assert res["p_value"] < 0.05, f"Expected significant result, got p={res['p_value']}"

# ── 3. Benchmark comparison ────────────────────────────────────────────────
print("\n[Benchmark comparison]")

@check("build_market_factor computes mean return by date")
def _():
    from benchmark_comparison import build_market_factor
    n = 100
    ohlcv = pd.DataFrame({
        "ticker": ["A"] * n + ["B"] * n,
        "date":   pd.date_range("2024-01-01", periods=n).tolist() * 2,
        "close":  np.random.uniform(100, 200, 2 * n),
        "log_return_1d": np.random.normal(0, 0.01, 2 * n),
    })
    mkt = build_market_factor(ohlcv)
    assert isinstance(mkt, pd.Series)
    assert len(mkt) == n

@check("factor_regression returns alpha and betas")
def _():
    from benchmark_comparison import factor_regression
    n = 200
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n)
    strategy = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
    market   = pd.Series(rng.normal(0.000, 0.01, n), index=dates)
    momentum = pd.Series(rng.normal(0.000, 0.005, n), index=dates)
    value    = pd.Series(rng.normal(0.000, 0.005, n), index=dates)
    reg = factor_regression(strategy, market, momentum, value)
    assert "alpha_ann_pct" in reg
    assert "t_stat_alpha"  in reg
    assert "r_squared"     in reg
    assert 0 <= reg["r_squared"] <= 1

# ── 4. Model monitor ───────────────────────────────────────────────────────
print("\n[Model monitor]")

@check("check_rolling_sharpe — high return series → OK")
def _():
    from model_monitor import check_rolling_sharpe
    # Add random noise so Standard Deviation is not zero
    returns = np.random.normal(0.003, 0.01, 40)
    snap = pd.DataFrame({
        "snapshot_date": pd.date_range("2024-01-01", periods=40),
        "portfolio_value": np.cumprod(1 + returns) * 1_000_000,
        "daily_return": returns,
        "daily_pnl": returns * 1_000_000,
        "open_positions": [3] * 40,
    })
    res = check_rolling_sharpe(snap)
    assert res["status"] == "ok", f"Expected OK, got {res['status']}"

@check("check_rolling_sharpe — flat returns → alert")
def _():
    from model_monitor import check_rolling_sharpe
    snap = pd.DataFrame({
        "snapshot_date":  pd.date_range("2024-01-01", periods=40),
        "portfolio_value":np.ones(40) * 1_000_000,
        "daily_return":   np.zeros(40),
        "daily_pnl":      np.zeros(40),
        "open_positions": [0] * 40,
    })
    res = check_rolling_sharpe(snap)
    assert res["status"] == "alert", f"Expected alert for zero returns, got {res['status']}"

@check("check_hit_rate — 70% wins → OK")
def _():
    from model_monitor import check_hit_rate
    trades = pd.DataFrame({
        "pnl_net":    [100, 200, -50, 150, 300, -80, 120, 90, -30, 200],
        "pred_ret":   np.random.randn(10) * 0.01,
        "actual_ret": np.random.randn(10) * 0.01,
    })
    res = check_hit_rate(trades)
    assert res["status"] == "ok"

@check("check_sentiment_drift — identical distributions → OK")
def _():
    from model_monitor import check_sentiment_drift
    n = 60
    # Duplicate exactly 30 values twice so the two halves perfectly match
    sent = pd.DataFrame({
        "trading_date": pd.date_range("2024-01-01", periods=n),
        "sent_pos":  np.tile(np.linspace(0.3, 0.7, 30), 2),
        "sent_neg":  np.tile(np.linspace(0.1, 0.3, 30), 2),
        "sent_spread": np.tile(np.linspace(-0.2, 0.3, 30), 2),
    })
    res = check_sentiment_drift(sent, 30)
    assert res["status"] in ("ok", "insufficient_data")

# ── 5. Incremental retrain (dry-run) ──────────────────────────────────────
print("\n[Incremental retrain]")

@check("run_retrain dry-run returns without error")
def _():
    import tempfile, duckdb
    from incremental_retrain import run_retrain
    from config_phase5 import FEATURE_COLS

    # Create minimal temp DB
    tmp = Path(tempfile.mkdtemp()) / "verify5.duckdb"
    con = duckdb.connect(str(tmp))
    cols_def = ", ".join(f"{c} DOUBLE" for c in FEATURE_COLS)
    con.execute(f"""
        CREATE TABLE feature_matrix (
            ticker VARCHAR, date DATE, {cols_def},
            target_ret_1d DOUBLE
        )
    """)
    con.close()

    result = run_retrain(days=30, epochs=2, db_path=tmp, dry_run=True)
    assert isinstance(result, dict)

# ── 6. Report generator ───────────────────────────────────────────────────
print("\n[Report generator]")

@check("generate_report creates HTML file with empty DB")
def _():
    import tempfile, duckdb
    from report_generator import generate_report

    tmp    = Path(tempfile.mkdtemp()) / "rep_test.duckdb"
    con    = duckdb.connect(str(tmp))
    con.execute("""
        CREATE TABLE portfolio_snapshots (
            snapshot_date DATE, portfolio_value DOUBLE, daily_return DOUBLE,
            cum_return DOUBLE, daily_pnl DOUBLE, open_positions INTEGER,
            cash DOUBLE, longs INTEGER, shorts INTEGER,
            daily_turnover DOUBLE, transaction_costs DOUBLE, created_at TIMESTAMPTZ
        );
        CREATE TABLE benchmark_snapshots (
            snapshot_date DATE, portfolio_value DOUBLE,
            daily_return DOUBLE, cum_return DOUBLE, created_at TIMESTAMPTZ
        );
        CREATE TABLE trades (
            trade_id VARCHAR, ticker VARCHAR, direction VARCHAR,
            signal_date DATE, pnl_net DOUBLE, actual_ret DOUBLE,
            status VARCHAR
        );
    """)
    con.close()

    out = Path(tempfile.mkdtemp()) / "test_report.html"
    path = generate_report(db_path=tmp, out_path=out)
    assert path.exists(), "Report file not created"
    content = path.read_text(encoding="utf-8")
    assert "Alpha Strategy Report" in content

# ── Summary ───────────────────────────────────────────────────────────────
print("\n── Summary ────────────────────────────────────────────────────")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  {passed} passed  /  {failed} failed  /  {len(results)} total\n")

if failed:
    print("Failed checks:")
    for name, ok, msg in results:
        if not ok:
            print(f"  {FAIL}  {name}: {msg}")
    sys.exit(1)
else:
    print("All Phase 5 checks passed. Ready to run evaluation pipeline.")
    sys.exit(0)
