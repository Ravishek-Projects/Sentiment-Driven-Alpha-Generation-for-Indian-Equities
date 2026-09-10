# data_loader.py
# All DuckDB read functions used by the Streamlit dashboard.
# Every function opens a READ-ONLY connection, queries, and closes immediately.
# Read-only connections never block concurrent writers.

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB = Path(__file__).parent.parent / "db" / "alpha.duckdb"


def _read(sql: str, params: list | None = None, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Thin helper: open read-only connection, execute, close."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Portfolio snapshots
# ---------------------------------------------------------------------------

def load_snapshots(db_path: Path = DEFAULT_DB, days: int | None = None) -> pd.DataFrame:
    """Load portfolio_snapshots, optionally limited to the last N days."""
    where = ""
    if days:
        cutoff = date.today() - timedelta(days=days)
        where  = f"WHERE snapshot_date >= '{cutoff}'"
    df = _read(f"""
        SELECT snapshot_date, portfolio_value, cash, daily_pnl,
               daily_return, cum_return, open_positions, longs, shorts,
               daily_turnover, transaction_costs
        FROM   portfolio_snapshots
        {where}
        ORDER  BY snapshot_date
    """, db_path=db_path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def load_benchmark(db_path: Path = DEFAULT_DB, days: int | None = None) -> pd.DataFrame:
    where = ""
    if days:
        cutoff = date.today() - timedelta(days=days)
        where  = f"WHERE snapshot_date >= '{cutoff}'"
    df = _read(f"""
        SELECT snapshot_date, portfolio_value, daily_return, cum_return
        FROM   benchmark_snapshots
        {where}
        ORDER  BY snapshot_date
    """, db_path=db_path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def load_trades(
    db_path: Path = DEFAULT_DB,
    status: str | None = None,
    ticker: str | None = None,
    days: int | None   = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Load trades with optional filters."""
    conditions = []
    if status:
        conditions.append(f"status = '{status}'")
    if ticker:
        conditions.append(f"ticker = '{ticker}'")
    if days:
        cutoff = date.today() - timedelta(days=days)
        conditions.append(f"signal_date >= '{cutoff}'")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    df = _read(f"""
        SELECT trade_id, ticker, signal_date, entry_date, exit_date,
               direction, entry_price, exit_price, position_size,
               capital_allocated, pnl_gross, pnl_net, transaction_cost,
               pred_ret, actual_ret, z_score, status
        FROM   trades
        {where}
        ORDER  BY signal_date DESC, ticker
        LIMIT  {limit}
    """, db_path=db_path)
    for col in ["signal_date", "entry_date", "exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load_open_trades(db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    return load_trades(db_path=db_path, status="open", limit=100)


def load_recent_closed(db_path: Path = DEFAULT_DB, days: int = 30) -> pd.DataFrame:
    return load_trades(db_path=db_path, status="closed", days=days)


# ---------------------------------------------------------------------------
# Today's signals
# ---------------------------------------------------------------------------

def load_today_signals(
    target_date: str | None = None,
    db_path: Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Load today's predictions from the predictions table."""
    if target_date is None:
        target_date = str(date.today())
    df = _read("""
        SELECT ticker, date, signal, pred_ret, position_size,
               z_score, pred_close, vol_20d
        FROM   predictions
        WHERE  date = ?
        ORDER  BY abs(z_score) DESC
    """, params=[target_date], db_path=db_path)
    return df


# ---------------------------------------------------------------------------
# Per-ticker analytics
# ---------------------------------------------------------------------------

def load_ticker_pnl(db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Aggregate closed P&L by ticker."""
    return _read("""
        SELECT ticker,
               COUNT(*)                                AS total_trades,
               SUM(pnl_net)                            AS total_pnl,
               AVG(pnl_net)                            AS avg_pnl,
               SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN pnl_net < 0 THEN 1 ELSE 0 END) AS losses,
               AVG(actual_ret)                         AS avg_actual_ret,
               AVG(pred_ret)                           AS avg_pred_ret
        FROM   trades
        WHERE  status = 'closed'
          AND  pnl_net IS NOT NULL
        GROUP  BY ticker
        ORDER  BY total_pnl DESC
    """, db_path=db_path)


def load_daily_pnl_by_ticker(db_path: Path = DEFAULT_DB, days: int = 30) -> pd.DataFrame:
    """Daily P&L breakdown by ticker (for heatmap)."""
    cutoff = date.today() - timedelta(days=days)
    return _read("""
        SELECT exit_date AS trade_date, ticker, SUM(pnl_net) AS pnl
        FROM   trades
        WHERE  status    = 'closed'
          AND  exit_date >= ?
          AND  pnl_net   IS NOT NULL
        GROUP  BY exit_date, ticker
        ORDER  BY exit_date
    """, params=[str(cutoff)], db_path=db_path)


# ---------------------------------------------------------------------------
# Pipeline health
# ---------------------------------------------------------------------------

def load_pipeline_runs(db_path: Path = DEFAULT_DB, limit: int = 20) -> pd.DataFrame:
    return _read(f"""
        SELECT run_id[1:8] AS id, run_type,
               (started_at::VARCHAR)[1:19]  AS started,
               (finished_at::VARCHAR)[1:19] AS finished,
               articles_fetched, articles_new, articles_tagged, status
        FROM   pipeline_runs
        ORDER  BY started_at DESC
        LIMIT  {limit}
    """, db_path=db_path)


def load_db_counts(db_path: Path = DEFAULT_DB) -> dict:
    """Quick row counts for all major tables — dashboard health panel."""
    tables = [
        "raw_news", "tagged_articles", "ohlcv",
        "sentiment_scores", "technical_features",
        "feature_matrix", "predictions", "trades",
        "portfolio_snapshots",
    ]
    counts = {}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for t in tables:
            try:
                counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                counts[t] = 0
    finally:
        con.close()
    return counts


# ---------------------------------------------------------------------------
# Sentiment and feature query helpers
# ---------------------------------------------------------------------------

def load_sentiment_history(
    ticker: str,
    db_path: Path = DEFAULT_DB,
    days: int = 60,
) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=days)
    return _read("""
        SELECT trading_date, sent_pos, sent_neg, sent_spread, news_count
        FROM   sentiment_scores
        WHERE  ticker       = ?
          AND  trading_date >= ?
        ORDER  BY trading_date
    """, params=[ticker, str(cutoff)], db_path=db_path)


def load_price_history(
    ticker: str,
    db_path: Path = DEFAULT_DB,
    days: int = 90,
) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=days)
    return _read("""
        SELECT date, close, volume, log_return_1d
        FROM   ohlcv
        WHERE  ticker = ?
          AND  date  >= ?
        ORDER  BY date
    """, params=[ticker, str(cutoff)], db_path=db_path)
