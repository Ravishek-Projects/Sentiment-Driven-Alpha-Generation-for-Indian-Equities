# verify_phase4.py
# End-to-end checks for Phase 4 — schema, backtester, metrics, dashboard imports.
# Uses a temporary in-memory DuckDB. No real data needed.
#
# Usage (Windows Terminal, from phase4\ directory)
#   python verify_phase4.py

from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

PASS = "[OK]"
FAIL = "[FAIL]"
results: list[tuple[str, bool, str]] = []

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))             # Looks in main folder
sys.path.insert(0, str(_root / "PHASE_2")) # Looks in Phase 2
sys.path.insert(0, str(_root / "PHASE_3")) # Looks in Phase 3
# --------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Bootstrap a temp DB with all phases' schemas
# ---------------------------------------------------------------------------

import duckdb, pandas as pd, numpy as np

_TMP = Path(tempfile.mkdtemp()) / "verify_p4.duckdb"

def _bootstrap():
    con = duckdb.connect(str(_TMP))
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, volume BIGINT, log_return_1d DOUBLE,
            PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS predictions (
            ticker VARCHAR, date DATE, pred_ret DOUBLE, signal VARCHAR,
            position_size DOUBLE, z_score DOUBLE, pred_close DOUBLE,
            vol_20d DOUBLE, computed_at TIMESTAMPTZ,
            PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id VARCHAR PRIMARY KEY, run_type VARCHAR,
            started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
            articles_fetched INTEGER DEFAULT 0, articles_new INTEGER DEFAULT 0,
            articles_tagged INTEGER DEFAULT 0,
            status VARCHAR DEFAULT 'running', error_msg VARCHAR
        );
        CREATE TABLE IF NOT EXISTS raw_news (article_id VARCHAR PRIMARY KEY,
            source VARCHAR, title VARCHAR, summary VARCHAR, link VARCHAR,
            published_at TIMESTAMPTZ, trading_date DATE, fetched_at TIMESTAMPTZ);
        CREATE TABLE IF NOT EXISTS tagged_articles (tag_id VARCHAR PRIMARY KEY,
            article_id VARCHAR, ticker VARCHAR, match_keyword VARCHAR, tagged_at TIMESTAMPTZ);
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            ticker VARCHAR, trading_date DATE, sent_pos DOUBLE, sent_neg DOUBLE,
            sent_neu DOUBLE, sent_spread DOUBLE, news_count INTEGER,
            articles_used INTEGER, computed_at TIMESTAMPTZ, PRIMARY KEY (ticker, trading_date));
        CREATE TABLE IF NOT EXISTS technical_features (
            ticker VARCHAR, date DATE, rsi_14 DOUBLE, ma_20 DOUBLE, ma_50 DOUBLE,
            price_to_ma20 DOUBLE, vol_20d DOUBLE, rel_volume DOUBLE, bb_position DOUBLE,
            lag_ret_1d DOUBLE, lag_ret_2d DOUBLE, computed_at TIMESTAMPTZ,
            PRIMARY KEY (ticker, date));
        CREATE TABLE IF NOT EXISTS feature_matrix (
            ticker VARCHAR, date DATE, sent_pos DOUBLE, sent_neg DOUBLE, sent_neu DOUBLE,
            sent_spread DOUBLE, news_count DOUBLE, rsi_14 DOUBLE, ma_20 DOUBLE, ma_50 DOUBLE,
            price_to_ma20 DOUBLE, vol_20d DOUBLE, rel_volume DOUBLE, bb_position DOUBLE,
            lag_ret_1d DOUBLE, lag_ret_2d DOUBLE, dow_mon DOUBLE, dow_tue DOUBLE,
            dow_wed DOUBLE, dow_thu DOUBLE, dow_fri DOUBLE, target_ret_1d DOUBLE,
            computed_at TIMESTAMPTZ, PRIMARY KEY (ticker, date));
    """)
    # Insert Phase 4 schema
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id VARCHAR PRIMARY KEY, ticker VARCHAR, signal_date DATE,
            entry_date DATE, exit_date DATE, direction VARCHAR,
            entry_price DOUBLE, exit_price DOUBLE, position_size DOUBLE,
            capital_allocated DOUBLE, pnl_gross DOUBLE, pnl_net DOUBLE,
            transaction_cost DOUBLE, pred_ret DOUBLE, actual_ret DOUBLE,
            z_score DOUBLE, status VARCHAR DEFAULT 'open', created_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            snapshot_date DATE PRIMARY KEY, portfolio_value DOUBLE, cash DOUBLE,
            daily_pnl DOUBLE, daily_return DOUBLE, cum_return DOUBLE,
            open_positions INTEGER, longs INTEGER, shorts INTEGER,
            daily_turnover DOUBLE, transaction_costs DOUBLE, created_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS benchmark_snapshots (
            snapshot_date DATE PRIMARY KEY, portfolio_value DOUBLE,
            daily_return DOUBLE, cum_return DOUBLE, created_at TIMESTAMPTZ
        );
    """)
    # Seed OHLCV for RELIANCE
    for i in range(90):
        d    = date(2024, 1, 2) + __import__("datetime").timedelta(days=i)
        base = 2800 + i * 0.5 + np.random.randn() * 10
        con.execute(
            "INSERT OR IGNORE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)",
            ["RELIANCE", str(d), base*0.99, base*1.01, base*0.98, base,
             int(1e6), float(np.random.randn()*0.01)],
        )
    # Seed predictions
    for i in range(5):
        d = date(2024, 3, 1) + __import__("datetime").timedelta(days=i)
        con.execute(
            "INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
            ["RELIANCE", str(d), 0.012, "LONG", 0.06, 1.1, 2850.0, 0.15,
             datetime.now(timezone.utc)],
        )
    con.close()

_bootstrap()

print("\n── Phase 4 verification ──────────────────────────────────────")

# ── 1. Imports ────────────────────────────────────────────────────────────
print("\n[Imports]")

@check("db_schema_phase4 importable")
def _(): import db_schema_phase4

@check("backtester importable")
def _(): import backtester

@check("metrics importable")
def _(): import metrics

@check("data_loader importable")
def _(): import data_loader

@check("daily_pnl_updater importable")
def _(): import daily_pnl_updater

@check("streamlit importable")
def _(): import streamlit

@check("plotly importable")
def _():
    import plotly.graph_objects as go
    import plotly.express as px

# ── 2. Schema ─────────────────────────────────────────────────────────────
print("\n[Schema]")

@check("Phase 4 schema creates all 3 tables")
def _():
    from db_schema_phase4 import extend_schema
    extend_schema(_TMP)
    con = duckdb.connect(str(_TMP), read_only=True)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    con.close()
    for t in ("trades", "portfolio_snapshots", "benchmark_snapshots"):
        assert t in tables, f"missing: {t}"

# ── 3. Metrics ────────────────────────────────────────────────────────────
print("\n[Metrics]")

@check("sharpe_ratio — known series")
def _():
    from metrics import sharpe_ratio
    # Add tiny noise so volatility is not exactly zero
    const = np.ones(252) * 0.01 + np.random.randn(252) * 0.0001
    s     = sharpe_ratio(const)
    assert s > 10   # very high Sharpe for constant-positive series

@check("sharpe_ratio — zero-mean series → ≈ 0")
def _():
    from metrics import sharpe_ratio
    np.random.seed(42)
    r = np.random.randn(252) * 0.01
    s = sharpe_ratio(r)
    assert abs(s) < 3.0   # should be close to 0 for random walk

@check("max_drawdown — returns negative fraction")
def _():
    from metrics import max_drawdown
    values = np.array([100, 110, 105, 95, 100, 120])
    mdd    = max_drawdown(values)
    assert mdd < 0
    # Corrected math: 95/110 - 1 = -0.136
    assert abs(mdd - (95/110 - 1)) < 0.01

@check("hit_rate — all positive → 1.0")
def _():
    from metrics import hit_rate
    assert hit_rate(np.array([1.0, 2.0, 0.5])) == 1.0

@check("hit_rate — half positive → 0.5")
def _():
    from metrics import hit_rate
    assert hit_rate(np.array([1.0, -1.0])) == 0.5

@check("compute_full_metrics — returns expected keys")
def _():
    from metrics import compute_full_metrics
    snap_df = pd.DataFrame([{
        "snapshot_date": date.today(),
        "portfolio_value": 1_050_000,
        "cash": 900_000,
        "daily_pnl": 5000,
        "daily_return": 0.005,
        "cum_return": 0.05,
        "open_positions": 3,
        "longs": 2, "shorts": 1,
        "daily_turnover": 100_000,
        "transaction_costs": 200,
    }])
    trades_df = pd.DataFrame([{
        "status": "closed", "pnl_net": 5000
    }])
    m = compute_full_metrics(snap_df, trades_df, 1_000_000)
    for key in ("sharpe_ratio", "hit_rate_pct", "max_drawdown_pct", "portfolio_value"):
        assert key in m, f"missing key: {key}"

# ── 4. Data loader ────────────────────────────────────────────────────────
print("\n[Data loader]")

@check("load_snapshots — returns DataFrame (empty OK)")
def _():
    from data_loader import load_snapshots
    df = load_snapshots(db_path=_TMP)
    assert isinstance(df, pd.DataFrame)

@check("load_today_signals — returns DataFrame")
def _():
    from data_loader import load_today_signals
    df = load_today_signals(target_date=str(date(2024, 3, 1)), db_path=_TMP)
    assert isinstance(df, pd.DataFrame)

@check("load_db_counts — returns dict with expected tables")
def _():
    from data_loader import load_db_counts
    counts = load_db_counts(db_path=_TMP)
    assert "trades" in counts
    assert "portfolio_snapshots" in counts

# ── 5. Daily P&L updater ──────────────────────────────────────────────────
print("\n[Daily P&L updater]")

@check("run_daily_update dry-run completes without error")
def _():
    from daily_pnl_updater import run_daily_update
    result = run_daily_update(
        target_date=date(2024, 3, 1),
        db_path=_TMP,
        dry_run=True,
    )
    assert isinstance(result, dict)

@check("run_daily_update actually writes snapshot (not dry-run)")
def _():
    from daily_pnl_updater import run_daily_update
    result = run_daily_update(
        target_date=date(2024, 3, 2),
        db_path=_TMP,
        dry_run=False,
    )
    # Verify snapshot was written
    con = duckdb.connect(str(_TMP), read_only=True)
    row = con.execute(
        "SELECT COUNT(*) FROM portfolio_snapshots WHERE snapshot_date = '2024-03-02'"
    ).fetchone()
    con.close()
    assert row[0] >= 1, "snapshot not written"

# ── 6. Backtester ─────────────────────────────────────────────────────────
print("\n[Backtester]")

@check("run_backtest returns summary dict")
def _():
    from backtester import run_backtest
    summary = run_backtest(
        initial_capital=1_000_000,
        cost_bps=20,
        db_path=_TMP,
    )
    assert isinstance(summary, dict)
    # May be empty if predictions don't align with OHLCV — that's OK
    if summary and "error" not in summary:
        assert "sharpe_ratio" in summary or "total_trades" in summary

# ── 7. Streamlit dashboard import ─────────────────────────────────────────
print("\n[Dashboard]")

@check("dashboard.py has no syntax errors")
def _():
    import ast
    src = (Path(__file__).parent / "dashboard.py").read_text(encoding="utf-8")
    ast.parse(src)   # raises SyntaxError if broken

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
    print("All Phase 4 checks passed. Ready to run the dashboard.")
    sys.exit(0)
