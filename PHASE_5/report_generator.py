# report_generator.py
# Generates a complete HTML performance report combining:
#   - Walk-forward evaluation results
#   - Statistical significance tests
#   - Benchmark comparison and factor regression
#   - Model health history
#   - Equity curve and drawdown charts (embedded as base64 PNG)
#
# The HTML report is self-contained (single file, no external deps).
# It can be opened in any browser or printed to PDF via Ctrl+P.
#
# Usage
#   python report_generator.py
#   python report_generator.py --out reports\my_report.html

from __future__ import annotations

import argparse
import base64
import json
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import duckdb
import numpy as np
import pandas as pd

from config_phase5 import (
    DB_PATH, EVAL_DIR, REPORTS_DIR, MONITOR_DIR,
)

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_snap(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            "SELECT snapshot_date, portfolio_value, daily_return, cum_return "
            "FROM portfolio_snapshots ORDER BY snapshot_date"
        ).df()
    finally:
        con.close()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _load_bench(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            "SELECT snapshot_date, portfolio_value "
            "FROM benchmark_snapshots ORDER BY snapshot_date"
        ).df()
    finally:
        con.close()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _load_trades(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            "SELECT ticker, direction, signal_date, pnl_net, actual_ret "
            "FROM trades WHERE status='closed' AND pnl_net IS NOT NULL"
        ).df()
    finally:
        con.close()
    return df


# ---------------------------------------------------------------------------
# Chart generators (base64-encoded PNG)
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _chart_equity(snap_df, bench_df) -> str:
    if not HAS_MPL or snap_df.empty:
        return ""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    ax1.plot(snap_df["snapshot_date"], snap_df["portfolio_value"],
             color="#1D9E75", lw=1.5, label="Strategy")
    if not bench_df.empty:
        # Rescale benchmark to same start
        scale = snap_df["portfolio_value"].iloc[0] / bench_df["portfolio_value"].iloc[0]
        ax1.plot(bench_df["snapshot_date"], bench_df["portfolio_value"] * scale,
                 color="#888780", lw=1.2, ls="--", label="Nifty50 (scaled)")
    ax1.set_ylabel("Portfolio value (₹)")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.grid(axis="y", alpha=0.3)

    # Drawdown
    vals = snap_df["portfolio_value"].values
    peak = np.maximum.accumulate(vals)
    dd   = (vals - peak) / peak * 100
    ax2.fill_between(snap_df["snapshot_date"], dd, 0, color="#D85A30", alpha=0.5)
    ax2.set_ylabel("Drawdown %")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _chart_wf_hit_rate(wf_df) -> str:
    if not HAS_MPL or wf_df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 3))
    x = range(len(wf_df))
    ax.bar(x, wf_df["hit_rate_ens"] * 100, color="#378ADD", alpha=0.7, label="Ensemble")
    ax.plot(x, wf_df["hit_rate_lstm"] * 100, "o--", color="#BA7517", ms=4, label="LSTM")
    ax.plot(x, wf_df["hit_rate_tf"]   * 100, "s--", color="#534AB7", ms=4, label="Transformer")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("Hit rate %")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _chart_pnl_ticker(trades_df) -> str:
    if not HAS_MPL or trades_df.empty:
        return ""
    tp = trades_df.groupby("ticker")["pnl_net"].sum().sort_values()
    colours = ["#1D9E75" if v >= 0 else "#D85A30" for v in tp.values]
    fig, ax = plt.subplots(figsize=(10, max(4, len(tp) * 0.35)))
    ax.barh(tp.index, tp.values, color=colours)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Cumulative P&L (₹)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:2rem;color:#222}
h1{font-size:1.6rem;font-weight:600;border-bottom:2px solid #1D9E75;padding-bottom:.5rem}
h2{font-size:1.2rem;font-weight:600;margin-top:2rem;color:#185FA5}
h3{font-size:1rem;font-weight:600;color:#3C3489;margin-top:1.5rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0}
th{background:#F1EFE8;padding:.4rem .8rem;text-align:left;border-bottom:1.5px solid #ccc;font-weight:600}
td{padding:.35rem .8rem;border-bottom:.5px solid #e0e0e0}
tr:hover td{background:#fafafa}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:1rem 0}
.kpi-card{background:#F1EFE8;border-radius:8px;padding:.75rem 1rem}
.kpi-label{font-size:.75rem;color:#666;margin:0}
.kpi-value{font-size:1.4rem;font-weight:600;margin:0}
.pos{color:#1D9E75}.neg{color:#D85A30}
.alert-box{background:#FAECE7;border-left:4px solid #D85A30;padding:.5rem 1rem;margin:.5rem 0;border-radius:4px;font-size:.9rem}
.ok-box{background:#EAF3DE;border-left:4px solid #1D9E75;padding:.5rem 1rem;margin:.5rem 0;border-radius:4px;font-size:.9rem}
img{width:100%;border-radius:6px;margin:.5rem 0}
.footer{font-size:.75rem;color:#aaa;margin-top:3rem;text-align:center}
"""


def _kpi(label, value, pos=True):
    cls = "pos" if pos else "neg"
    return f'<div class="kpi-card"><p class="kpi-label">{label}</p><p class="kpi-value {cls}">{value}</p></div>'


def _table(df: pd.DataFrame, cols=None) -> str:
    df = df[cols] if cols else df
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in df.values
    )
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def generate_report(
    db_path:  Path = DB_PATH,
    out_path: Path | None = None,
) -> Path:
    """Build the full HTML report and save it. Returns output path."""
    if out_path is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = REPORTS_DIR / f"alpha_report_{ts}.html"

    snap_df   = _load_snap(db_path)
    bench_df  = _load_bench(db_path)
    trades_df = _load_trades(db_path)

    # Load eval files
    wf_df     = pd.DataFrame()
    wf_path   = EVAL_DIR / "walk_forward_results.csv"
    if wf_path.exists():
        wf_df = pd.read_csv(wf_path)

    stat_data = {}
    stat_path = EVAL_DIR / "statistical_tests.json"
    if stat_path.exists():
        with open(stat_path) as f:
            stat_data = json.load(f)

    comp_data = {}
    comp_path = EVAL_DIR / "benchmark_comparison.json"
    if comp_path.exists():
        with open(comp_path) as f:
            comp_data = json.load(f)

    # Compute top-level metrics
    r = snap_df["daily_return"].dropna().values if not snap_df.empty else np.array([])
    final_val    = float(snap_df["portfolio_value"].iloc[-1]) if not snap_df.empty else 0
    total_ret    = (final_val / 1_000_000 - 1) * 100 if final_val else 0
    ann_ret      = float(r.mean() * 252 * 100) if len(r) > 1 else 0
    ann_vol      = float(r.std(ddof=1) * np.sqrt(252) * 100) if len(r) > 1 else 0
    sharpe       = ann_ret / ann_vol if ann_vol > 0 else 0
    n_closed     = len(trades_df)
    hit_rate     = float((trades_df["pnl_net"] > 0).mean() * 100) if n_closed > 0 else 0

    # Charts
    chart_eq  = _chart_equity(snap_df, bench_df)
    chart_wf  = _chart_wf_hit_rate(wf_df)
    chart_tkr = _chart_pnl_ticker(trades_df)

    def _img(b64): return f'<img src="data:image/png;base64,{b64}">' if b64 else "<p><em>Chart unavailable — install matplotlib: pip install matplotlib</em></p>"

    # Walk-forward summary table
    wf_table = ""
    if not wf_df.empty:
        display = wf_df[["fold","train_end","test_start","test_end",
                          "n_test_samples","hit_rate_ens","ic_ens","rmse_ens"]].copy()
        display["hit_rate_ens"] = (display["hit_rate_ens"] * 100).round(1).astype(str) + "%"
        display["ic_ens"]       = display["ic_ens"].round(4)
        display["rmse_ens"]     = display["rmse_ens"].round(6)
        wf_table = _table(display)

    # Statistical tests section
    stat_html = ""
    if stat_data:
        for key, val in stat_data.items():
            if isinstance(val, dict):
                rows = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in val.items())
                stat_html += f"<h3>{key.replace('_',' ').title()}</h3><table>{rows}</table>"

    # Benchmark table
    bench_html = ""
    if comp_data.get("performance"):
        perf_rows = []
        for k, p in comp_data["performance"].items():
            perf_rows.append({
                "Strategy":   p.get("label",""),
                "Total ret%": f"{p.get('total_return',0):.2f}%",
                "Ann ret%":   f"{p.get('ann_return',0):.2f}%",
                "Sharpe":     f"{p.get('sharpe',0):.4f}",
                "Max DD%":    f"{p.get('max_dd',0):.2f}%",
            })
        bench_html = _table(pd.DataFrame(perf_rows))

        reg = comp_data.get("factor_regression", {})
        if reg:
            sig   = reg.get("alpha_significant", False)
            box   = "ok-box" if sig else "alert-box"
            bench_html += f"""
            <h3>Factor regression (α decomposition)</h3>
            <div class="{box}">
              Annual alpha: <b>{reg.get('alpha_ann_pct',0):.4f}%</b> &nbsp;
              t-stat: <b>{reg.get('t_stat_alpha','N/A')}</b> &nbsp;
              p-value: <b>{reg.get('p_value_alpha','N/A')}</b> &nbsp;
              Significant: <b>{'Yes ✓' if sig else 'No ✗'}</b>
            </div>"""

    # Alert history
    alerts = []
    alert_file = MONITOR_DIR / "alerts.jsonl"
    if alert_file.exists():
        with open(alert_file) as f:
            for line in f:
                try: alerts.append(json.loads(line))
                except: pass

    alerts_html = ""
    if alerts:
        for a in alerts[-10:]:
            alerts_html += f"""
            <div class="alert-box">
              <b>{a.get('date','')} — {a.get('metric','?')}</b>:
              {a.get('message', a.get('conclusion',''))}
            </div>"""
    else:
        alerts_html = '<div class="ok-box">No alerts in history — model health OK.</div>'

    # ── Assemble HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Alpha Report — {date.today()}</title>
<style>{_CSS}</style></head><body>

<h1>📈 Alpha Strategy Report</h1>
<p style="color:#666;font-size:.9rem">
  NSE Nifty50 · FinBERT sentiment + technical ensemble ·
  Generated {datetime.now().strftime('%d %b %Y %H:%M')}
</p>

<h2>Portfolio Summary</h2>
<div class="kpi-grid">
  {_kpi("Portfolio value",   f"₹{final_val:,.0f}",    final_val > 1_000_000)}
  {_kpi("Total return",      f"{total_ret:+.2f}%",    total_ret > 0)}
  {_kpi("Ann. return",       f"{ann_ret:+.2f}%",      ann_ret > 0)}
  {_kpi("Sharpe ratio",      f"{sharpe:.2f}",         sharpe > 1)}
  {_kpi("Hit rate",          f"{hit_rate:.1f}%",      hit_rate > 52)}
  {_kpi("Closed trades",     str(n_closed),           True)}
</div>

<h2>Equity Curve &amp; Drawdown</h2>
{_img(chart_eq)}

<h2>Walk-forward Evaluation ({len(wf_df)} folds)</h2>
{_img(chart_wf)}
{wf_table if wf_table else "<p><em>Run walk_forward_eval.py to populate this section.</em></p>"}

{'<h3>Walk-forward mean metrics</h3><table><tr><th>Metric</th><th>LSTM</th><th>Transformer</th><th>Ensemble</th></tr>' + "".join(f"<tr><td>{m}</td><td>{wf_df[f'{m}_lstm'].mean():.4f}</td><td>{wf_df[f'{m}_tf'].mean():.4f}</td><td>{wf_df[f'{m}_ens'].mean():.4f}</td></tr>" for m in ['hit_rate','ic','rmse']) + '</table>' if not wf_df.empty else ""}

<h2>Statistical Significance Tests</h2>
{stat_html if stat_html else "<p><em>Run statistical_tests.py to populate this section.</em></p>"}

<h2>Benchmark Comparison</h2>
{bench_html if bench_html else "<p><em>Run benchmark_comparison.py to populate this section.</em></p>"}

<h2>P&amp;L by Ticker</h2>
{_img(chart_tkr)}

<h2>Model Health Alerts</h2>
{alerts_html}

<div class="footer">
  Alpha Dashboard · Research use only · Not financial advice ·
  alpha.duckdb · {date.today()}
</div>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[report] Report saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML performance report")
    parser.add_argument("--db",  default=str(DB_PATH))
    parser.add_argument("--out", default=None, help="Output HTML path")
    args = parser.parse_args()

    path = generate_report(
        db_path  = Path(args.db),
        out_path = Path(args.out) if args.out else None,
    )
    print(f"Open in browser: {path}")
