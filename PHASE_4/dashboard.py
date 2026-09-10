# dashboard.py
# Streamlit dashboard — live daily P&L tracker.
#
# Run (Windows Terminal, from phase4\ directory):
#   streamlit run dashboard.py
#   streamlit run dashboard.py --server.port 8501
#
# The dashboard auto-refreshes every 5 minutes during market hours.
# All data comes from DuckDB via read-only connections.

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Add parent dirs to path so we can import from phase1-3
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import (
    load_snapshots, load_benchmark, load_trades, load_open_trades,
    load_recent_closed, load_today_signals, load_ticker_pnl,
    load_daily_pnl_by_ticker, load_pipeline_runs, load_db_counts,
    load_sentiment_history, load_price_history, DEFAULT_DB,
)
from metrics import compute_full_metrics, rolling_sharpe

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Alpha Dashboard — NSE Sentiment Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auto-refresh every 5 minutes (300_000 ms)
# ---------------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300_000, key="main_refresh")
except ImportError:
    pass   # Works without it; user must refresh manually

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
GREEN  = "#1D9E75"
RED    = "#D85A30"
BLUE   = "#378ADD"
AMBER  = "#BA7517"
GREY   = "#888780"
BG     = "#F1EFE8"

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    db_path   = Path(st.text_input("Database path", value=str(DEFAULT_DB)))
    lookback  = st.selectbox("Chart lookback", [30, 60, 90, 180, 365], index=1)
    show_bench= st.checkbox("Show Nifty50 benchmark", value=True)
    ticker_filter = st.selectbox(
        "Ticker deep-dive",
        ["— all —"] + [
            "RELIANCE","TCS","HDFCBANK","INFY","BHARTIARTL",
            "ICICIBANK","KOTAKBANK","AXISBANK","LT","WIPRO",
            "SUNPHARMA","MARUTI","BAJFINANCE","TITAN","NESTLEIND",
        ],
    )
    st.markdown("---")
    st.caption(f"Last refresh: {date.today()}")
    if st.button("🔄  Refresh now"):
        st.rerun()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)   # cache for 60 seconds
def _snapshots(days): return load_snapshots(db_path, days)

@st.cache_data(ttl=60, show_spinner=False)
def _benchmark(days): return load_benchmark(db_path, days)

@st.cache_data(ttl=60, show_spinner=False)
def _open_trades(): return load_open_trades(db_path)

@st.cache_data(ttl=60, show_spinner=False)
def _closed_trades(days): return load_recent_closed(db_path, days)

@st.cache_data(ttl=60, show_spinner=False)
def _today_signals(): return load_today_signals(db_path=db_path)

@st.cache_data(ttl=300, show_spinner=False)
def _ticker_pnl(): return load_ticker_pnl(db_path)

@st.cache_data(ttl=300, show_spinner=False)
def _heatmap_data(days): return load_daily_pnl_by_ticker(db_path, days)

@st.cache_data(ttl=300, show_spinner=False)
def _db_counts(): return load_db_counts(db_path)

snap_df    = _snapshots(lookback)
bench_df   = _benchmark(lookback)
open_df    = _open_trades()
closed_df  = _closed_trades(lookback)
signals_df = _today_signals()
ticker_pnl = _ticker_pnl()

all_closed = load_trades(db_path=db_path, status="closed", limit=10000)
m = compute_full_metrics(snap_df, all_closed, initial_capital=1_000_000.0)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## 📈 Alpha Dashboard — NSE Sentiment Strategy")
col_hdr1, col_hdr2 = st.columns([3, 1])
with col_hdr1:
    st.caption(
        f"Universe: Nifty 50 &nbsp;·&nbsp; "
        f"Strategy: FinBERT sentiment + technical ensemble &nbsp;·&nbsp; "
        f"Updated: {date.today()}"
    )
with col_hdr2:
    if not snap_df.empty:
        last_date = snap_df["snapshot_date"].max().strftime("%d %b %Y")
        st.caption(f"Last P&L date: **{last_date}**")

st.divider()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

k1, k2, k3, k4, k5, k6 = st.columns(6)

def _delta_colour(v): return "normal" if v >= 0 else "inverse"

k1.metric(
    "Portfolio value",
    f"₹{m['portfolio_value']:,.0f}",
    f"₹{m['cum_pnl']:+,.0f} total",
)
k2.metric(
    "Today's P&L",
    f"₹{m['today_pnl']:+,.0f}",
    f"{m['today_return_pct']:+.3f}%",
    delta_color=_delta_colour(m['today_pnl']),
)
k3.metric(
    "Sharpe ratio",
    f"{m['sharpe_ratio']:.2f}",
    f"Ann. vol {m['ann_volatility_pct']:.1f}%",
)
k4.metric(
    "Hit rate",
    f"{m['hit_rate_pct']:.1f}%",
    f"{m['total_trades']} closed trades",
)
k5.metric(
    "Max drawdown",
    f"{m['max_drawdown_pct']:.1f}%",
    f"Calmar {m['calmar_ratio']:.2f}",
    delta_color="inverse",
)
k6.metric(
    "Open positions",
    str(m["open_positions"]),
    f"{m['open_longs']}L · {m['open_shorts']}S",
)

st.divider()

# ---------------------------------------------------------------------------
# Equity curve + daily bar chart
# ---------------------------------------------------------------------------

chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.markdown("#### Equity curve")
    if not snap_df.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=snap_df["snapshot_date"],
            y=snap_df["portfolio_value"],
            mode="lines",
            name="Strategy",
            line=dict(color=GREEN, width=2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.0f}<extra></extra>",
        ))

        if show_bench and not bench_df.empty:
            # Rescale benchmark to same starting capital
            if len(bench_df) and len(snap_df):
                start_port = snap_df["portfolio_value"].iloc[0]
                scale = start_port / bench_df["portfolio_value"].iloc[0] if bench_df["portfolio_value"].iloc[0] > 0 else 1
                fig.add_trace(go.Scatter(
                    x=bench_df["snapshot_date"],
                    y=bench_df["portfolio_value"] * scale,
                    mode="lines",
                    name="Nifty 50 (scaled)",
                    line=dict(color=GREY, width=1.5, dash="dot"),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.0f}<extra></extra>",
                ))

        fig.update_layout(
            height=280, margin=dict(l=0, r=0, t=8, b=0),
            legend=dict(orientation="h", y=1.1, x=0),
            yaxis=dict(tickprefix="₹", tickformat=",.0f"),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No portfolio snapshots yet. Run `python backtester.py` first.")

with chart_col2:
    st.markdown("#### Daily P&L (₹)")
    if not snap_df.empty:
        colours = [GREEN if v >= 0 else RED for v in snap_df["daily_pnl"]]
        fig2 = go.Figure(go.Bar(
            x=snap_df["snapshot_date"],
            y=snap_df["daily_pnl"],
            marker_color=colours,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:+,.0f}<extra></extra>",
        ))
        fig2.update_layout(
            height=280, margin=dict(l=0, r=0, t=8, b=0),
            yaxis=dict(tickprefix="₹", tickformat=",.0f"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Rolling Sharpe + drawdown
# ---------------------------------------------------------------------------

roll_col1, roll_col2 = st.columns(2)

with roll_col1:
    st.markdown("#### Rolling 30-day Sharpe")
    if not snap_df.empty and len(snap_df) >= 30:
        rs = rolling_sharpe(snap_df.set_index("snapshot_date")["daily_return"], 30)
        rs = rs.reset_index()
        rs.columns = ["date", "sharpe"]
        colours_rs = [GREEN if v >= 1 else (AMBER if v >= 0 else RED) for v in rs["sharpe"]]
        fig3 = go.Figure(go.Bar(
            x=rs["date"], y=rs["sharpe"], marker_color=colours_rs,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Sharpe: %{y:.2f}<extra></extra>",
        ))
        fig3.add_hline(y=1.0, line_dash="dot", line_color=GREY, annotation_text="1.0")
        fig3.update_layout(
            height=220, margin=dict(l=0, r=0, t=8, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.caption("Need ≥ 30 days of history for rolling Sharpe.")

with roll_col2:
    st.markdown("#### Drawdown")
    if not snap_df.empty:
        vals   = snap_df["portfolio_value"].values
        peak   = np.maximum.accumulate(vals)
        dd_pct = (vals - peak) / peak * 100
        fig4 = go.Figure(go.Scatter(
            x=snap_df["snapshot_date"], y=dd_pct,
            fill="tozeroy", fillcolor="rgba(216,90,48,0.15)",
            line=dict(color=RED, width=1.5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:.2f}%<extra></extra>",
        ))
        fig4.update_layout(
            height=220, margin=dict(l=0, r=0, t=8, b=0),
            yaxis=dict(ticksuffix="%"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Today's signals table
# ---------------------------------------------------------------------------

st.markdown("#### Today's signals")

if signals_df.empty:
    st.info(f"No signals for {date.today()}. Run `python signal_generator.py` in phase3\\.")
else:
    def _signal_badge(sig):
        if sig == "LONG":
            return "🟢 LONG"
        elif sig == "SHORT":
            return "🔴 SHORT"
        return "⚪ FLAT"

    display = signals_df.copy()
    display["signal"]       = display["signal"].map(_signal_badge)
    display["pred_ret"]     = (display["pred_ret"] * 100).round(3).astype(str) + "%"
    display["z_score"]      = display["z_score"].round(3)
    display["position_size"]= (display["position_size"] * 100).round(1).astype(str) + "%"
    display["pred_close"]   = "₹" + display["pred_close"].round(2).astype(str)
    display["vol_20d"]      = (display["vol_20d"] * 100).round(1).astype(str) + "%"

    display = display.rename(columns={
        "ticker": "Ticker", "signal": "Signal",
        "pred_ret": "Pred. return", "z_score": "Z-score",
        "position_size": "Position", "pred_close": "Pred. close",
        "vol_20d": "Vol 20d",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Open positions
# ---------------------------------------------------------------------------

st.markdown("#### Open positions")

if open_df.empty:
    st.caption("No open positions.")
else:
    op = open_df.copy()
    for col in ["entry_price", "capital_allocated", "pred_ret", "z_score"]:
        if col in op.columns:
            op[col] = op[col].round(4)
    op["capital_allocated"] = op["capital_allocated"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(op[[
        "ticker", "direction", "entry_date", "entry_price",
        "capital_allocated", "position_size", "pred_ret", "z_score"
    ]], use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Recent closed trades
# ---------------------------------------------------------------------------

st.markdown("#### Recent closed trades")

if closed_df.empty:
    st.caption("No closed trades in the selected window.")
else:
    ct = closed_df.copy()
    ct["pnl_net"]  = ct["pnl_net"].apply(lambda x: f"₹{x:+,.2f}" if pd.notna(x) else "—")
    ct["actual_ret"] = ct["actual_ret"].apply(
        lambda x: f"{x*100:+.3f}%" if pd.notna(x) else "—"
    )
    ct["pred_ret"]   = ct["pred_ret"].apply(
        lambda x: f"{x*100:+.3f}%" if pd.notna(x) else "—"
    )

    def _colour_pnl(val):
        if "+" in str(val): return "color: #1D9E75"
        if "−" in str(val) or "-" in str(val): return "color: #D85A30"
        return ""

    st.dataframe(
        ct[[
            "ticker", "direction", "signal_date", "exit_date",
            "entry_price", "exit_price", "pred_ret", "actual_ret", "pnl_net",
        ]].rename(columns={
            "signal_date": "Signal date", "exit_date": "Exit date",
            "entry_price": "Entry", "exit_price": "Exit",
            "pred_ret": "Pred", "actual_ret": "Actual", "pnl_net": "P&L (net)",
        }),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# P&L by ticker (bar chart)
# ---------------------------------------------------------------------------

st.markdown("#### Cumulative P&L by ticker")

if ticker_pnl.empty:
    st.caption("No completed trades yet.")
else:
    tp = ticker_pnl.sort_values("total_pnl", ascending=True)
    colours = [GREEN if v >= 0 else RED for v in tp["total_pnl"]]
    fig5 = go.Figure(go.Bar(
        x=tp["total_pnl"], y=tp["ticker"],
        orientation="h", marker_color=colours,
        hovertemplate="<b>%{y}</b><br>₹%{x:+,.0f}<extra></extra>",
    ))
    fig5.update_layout(
        height=max(300, len(tp) * 22),
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(tickprefix="₹", tickformat=",.0f"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------------------------------------------------
# Ticker deep-dive
# ---------------------------------------------------------------------------

if ticker_filter != "— all —":
    st.divider()
    st.markdown(f"#### {ticker_filter} — deep dive")

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("**Price history**")
        ph = load_price_history(ticker_filter, db_path=db_path, days=lookback)
        if not ph.empty:
            ph["date"] = pd.to_datetime(ph["date"])
            fig6 = go.Figure(go.Scatter(
                x=ph["date"], y=ph["close"], mode="lines",
                line=dict(color=BLUE, width=1.5),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>",
            ))
            fig6.update_layout(
                height=220, margin=dict(l=0, r=0, t=8, b=0),
                yaxis=dict(tickprefix="₹"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig6, use_container_width=True)

    with dcol2:
        st.markdown("**Sentiment history**")
        sh = load_sentiment_history(ticker_filter, db_path=db_path, days=lookback)
        if not sh.empty:
            sh["trading_date"] = pd.to_datetime(sh["trading_date"])
            fig7 = go.Figure()
            fig7.add_trace(go.Bar(
                x=sh["trading_date"], y=sh["sent_spread"],
                marker_color=[GREEN if v >= 0 else RED for v in sh["sent_spread"]],
                name="Sentiment spread",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Spread: %{y:.3f}<extra></extra>",
            ))
            fig7.update_layout(
                height=220, margin=dict(l=0, r=0, t=8, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig7, use_container_width=True)

    # Ticker trade history
    t_trades = load_trades(db_path=db_path, ticker=ticker_filter,
                           status="closed", days=365, limit=50)
    if not t_trades.empty:
        st.markdown(f"**{ticker_filter} trade history**")
        t_trades["pnl_net"] = t_trades["pnl_net"].round(2)
        st.dataframe(
            t_trades[["direction","signal_date","exit_date",
                       "entry_price","exit_price","pred_ret","actual_ret","pnl_net"]],
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------------------
# Metrics summary panel
# ---------------------------------------------------------------------------

st.divider()
st.markdown("#### Performance summary")

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Ann. return",      f"{m['ann_return_pct']:+.1f}%")
mc1.metric("Ann. volatility",  f"{m['ann_volatility_pct']:.1f}%")
mc2.metric("Sharpe ratio",     f"{m['sharpe_ratio']:.2f}")
mc2.metric("Sortino ratio",    f"{m['sortino_ratio']:.2f}")
mc3.metric("Max drawdown",     f"{m['max_drawdown_pct']:.1f}%")
mc3.metric("Calmar ratio",     f"{m['calmar_ratio']:.2f}")
mc4.metric("Win/loss ratio",   f"{m['win_loss_ratio']:.2f}×")
mc4.metric("Total costs",      f"₹{m['total_costs_inr']:,.0f}")

# ---------------------------------------------------------------------------
# Pipeline health
# ---------------------------------------------------------------------------

with st.expander("Pipeline health & DB stats", expanded=False):
    counts = _db_counts()
    cols   = st.columns(len(counts))
    for col, (tbl, cnt) in zip(cols, counts.items()):
        col.metric(tbl, f"{cnt:,}")

    st.markdown("**Recent pipeline runs**")
    runs_df = load_pipeline_runs(db_path=db_path)
    if not runs_df.empty:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

st.caption(
    "Phase 4 dashboard · Data sourced from alpha.duckdb · "
    "Not financial advice · For research purposes only"
)
