# Phase 4 — Daily P&L Dashboard
## Complete Windows Terminal Setup and Run Guide

---

## What Phase 4 does

Phase 4 is the final layer of the pipeline. It takes the trading signals
produced by Phase 3 and turns them into a live, interactive P&L dashboard.

```
predictions (DuckDB, written by Phase 3)
        │
        ├──► backtester.py         — historical simulation, populates trades table
        │
        ├──► daily_pnl_updater.py  — post-close: closes trades, opens new positions,
        │                             writes portfolio snapshot
        │
        ├──► scheduler_phase4.py   — runs daily_pnl_updater at 16:30 + 08:45 IST
        │
        └──► dashboard.py          — Streamlit app (equity curve, P&L, signals, trades)
```

The dashboard reads everything from DuckDB via read-only connections.
It auto-refreshes every 5 minutes and never blocks the schedulers.

---

## File structure

```
phase4\
├── db_schema_phase4.py     ← adds trades, portfolio_snapshots, benchmark_snapshots
├── backtester.py           ← historical walk-forward simulation
├── metrics.py              ← Sharpe, Sortino, max DD, hit rate, etc. (pure functions)
├── data_loader.py          ← all DuckDB read functions for the dashboard
├── daily_pnl_updater.py    ← post-close P&L accounting job
├── scheduler_phase4.py     ← APScheduler for daily update
├── dashboard.py            ← Streamlit dashboard
├── verify_phase4.py        ← automated checks
└── requirements_phase4.txt
```

---

## Prerequisites

Before starting Phase 4:

1. Phases 1, 2, and 3 are fully set up and running
2. Phase 3 training is complete (`checkpoints\lstm_best.pt` and `checkpoints\transformer_best.pt` exist)
3. `predictions` table has rows (run `python signal_generator.py` in `phase3\`)

---

## Windows Terminal — Step-by-Step

### Important: folder layout assumed

```
alpha_project\
├── phase1\          ← scheduler.py, db\alpha.duckdb
├── phase2\          ← scheduler_phase2.py
├── phase3\          ← scheduler_phase3.py, signal_generator.py
├── phase4\          ← dashboard.py  ← you are here
└── venv\            ← shared virtual environment
```

Every command below is run from the `phase4\` folder unless stated otherwise.

---

### Step 1 — Open Windows Terminal and navigate

```powershell
# Win + X → Windows Terminal
cd C:\Users\YourName\alpha_project\phase4

# Confirm location
dir
# Should show: dashboard.py  backtester.py  metrics.py  ...
```

---

### Step 2 — Activate the virtual environment

```powershell
..\venv\Scripts\activate

# Prompt changes to:
# (venv) PS C:\Users\YourName\alpha_project\phase4>
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
..\venv\Scripts\activate
```

---

### Step 3 — Install Phase 4 dependencies

```powershell
pip install -r requirements_phase4.txt
```

This installs streamlit, plotly, and streamlit-autorefresh.
Expected time: 1–3 minutes.

---

### Step 4 — Add Phase 4 tables to the database

Run this once only. It adds `trades`, `portfolio_snapshots`, and
`benchmark_snapshots` to `alpha.duckdb`.

```powershell
python db_schema_phase4.py
```

Expected output:
```
[schema_p4] Phase 4 tables ready in ..\phase1\db\alpha.duckdb
[schema_p4] Tables: trades, portfolio_snapshots, benchmark_snapshots
```

---

### Step 5 — Run the verification suite

```powershell
python verify_phase4.py
```

Expected output:
```
── Phase 4 verification ──────────────────────────────────────

[Imports]
  [OK]  db_schema_phase4 importable
  [OK]  backtester importable
  [OK]  metrics importable
  [OK]  data_loader importable
  [OK]  daily_pnl_updater importable
  [OK]  streamlit importable
  [OK]  plotly importable

[Schema]
  [OK]  Phase 4 schema creates all 3 tables

[Metrics]
  [OK]  sharpe_ratio — known series
  [OK]  sharpe_ratio — zero-mean series → ≈ 0
  [OK]  max_drawdown — returns negative fraction
  [OK]  hit_rate — all positive → 1.0
  [OK]  hit_rate — half positive → 0.5
  [OK]  compute_full_metrics — returns expected keys

[Data loader]
  [OK]  load_snapshots — returns DataFrame (empty OK)
  [OK]  load_today_signals — returns DataFrame
  [OK]  load_db_counts — returns dict with expected tables

[Daily P&L updater]
  [OK]  run_daily_update dry-run completes without error
  [OK]  run_daily_update actually writes snapshot (not dry-run)

[Backtester]
  [OK]  run_backtest returns summary dict

[Dashboard]
  [OK]  dashboard.py has no syntax errors

── Summary ────────────────────────────────────────────────────
  20 passed  /  0 failed  /  20 total

All Phase 4 checks passed. Ready to run the dashboard.
```

---

### Step 6 — Run the historical backtester (populates trades table)

The backtester replays all past `predictions` as simulated trades,
writing results into `trades` and `portfolio_snapshots`.
Run this once before launching the dashboard.

```powershell
python backtester.py
```

Example output:
```
2024-03-28 09:12:01 [INFO] [benchmark] Wrote 87 benchmark snapshot rows
2024-03-28 09:12:02 [INFO] [backtest] Wrote 243 trades, 87 snapshots

── Backtest Summary ────────────────────────────────────────
  initial_capital          : 1000000.0
  final_value              : 1214380.0
  total_return_pct         : 21.44
  ann_return_pct           : 28.12
  ann_volatility_pct       : 15.31
  sharpe_ratio             : 1.8372
  max_drawdown_pct         : -6.24
  hit_rate_pct             : 61.2
  win_loss_ratio           : 2.14
  total_trades             : 243
  trading_days             : 87
```

To start with a different capital amount:
```powershell
python backtester.py --capital 500000      # ₹5 lakhs
python backtester.py --capital 2000000     # ₹20 lakhs
python backtester.py --cost-bps 30         # higher transaction cost
```

---

### Step 7 — Test the daily P&L updater manually

Before running the scheduler, test the updater once:

```powershell
# Dry run — shows what would happen without writing anything
python daily_pnl_updater.py --dry-run

# Real run — closes yesterday's trades, opens today's positions, writes snapshot
python daily_pnl_updater.py
```

Example output:
```
2024-03-28 16:31:00 [INFO] [updater] Daily update for 2024-03-28  dry_run=False
2024-03-28 16:31:00 [INFO]   CLOSE LONG   RELIANCE     entry=₹2896.40  exit=₹2934.12  P&L=₹+2,316.00  ret=+1.301%
2024-03-28 16:31:00 [INFO]   CLOSE SHORT  BHARTIARTL   entry=₹1142.00  exit=₹1118.32  P&L=₹+1,388.00  ret=-2.103%
2024-03-28 16:31:00 [INFO]   OPEN  LONG   TCS          @ ₹3812.44  alloc=₹62,400  z=1.10
2024-03-28 16:31:00 [INFO]   OPEN  SHORT  ONGC         @ ₹284.60   alloc=₹48,000  z=-0.92
2024-03-28 16:31:00 [INFO] [updater] Done — portfolio=₹1,221,840  day_pnl=₹+7,240  closed=5  opened=4
```

---

### Step 8 — Launch the dashboard

```powershell
streamlit run dashboard.py
```

Streamlit prints:
```
  You can now view your Streamlit app in your browser.

  Local URL:  http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

Open `http://localhost:8501` in your browser (Chrome or Edge recommended).

**The dashboard will be empty if `portfolio_snapshots` has no data.**
Run the backtester first (Step 6) to populate it.

To run on a different port:
```powershell
streamlit run dashboard.py --server.port 8502
```

To make it accessible from other devices on your network:
```powershell
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

---

### Step 9 — Start the Phase 4 scheduler

Open a **fourth Windows Terminal tab** (Ctrl+Shift+T).

```powershell
..\venv\Scripts\activate
cd ..\phase4
python scheduler_phase4.py
```

This runs `daily_pnl_updater.py` automatically at:
- 16:30 IST — after NSE closes (OHLCV top-up runs at 16:15)
- 08:45 IST — morning catch-up in case evening job missed data

---

## All four terminal tabs — final setup

At this point you should have four terminal windows running:

| Tab | Directory | Command |
|---|---|---|
| 1 | `phase1\` | `python scheduler.py` |
| 2 | `phase2\` | `python scheduler_phase2.py` |
| 3 | `phase3\` | `python scheduler_phase3.py` |
| 4 | `phase4\` | `python scheduler_phase4.py` |

And one browser window open at `http://localhost:8501` showing the dashboard.

Start the Streamlit dashboard in a **fifth** window:
```powershell
..\venv\Scripts\activate
cd ..\phase4
streamlit run dashboard.py
```

---

## Dashboard sections explained

### KPI row (top)
Six metrics updated every time the page refreshes:
- **Portfolio value** — total INR value (cash + open positions mark-to-market)
- **Today's P&L** — net profit/loss after transaction costs
- **Sharpe ratio** — annualised, rolling over full backtest history
- **Hit rate** — % of closed trades with positive P&L
- **Max drawdown** — worst peak-to-trough decline
- **Open positions** — live count of long and short positions

### Equity curve
Strategy vs. Nifty50 equal-weight benchmark (rescaled to same starting capital).
Toggle the benchmark from the sidebar.

### Daily P&L bars
Green = profitable day, red = loss day. Hover for exact figures.

### Rolling 30-day Sharpe
Colour-coded: green ≥ 1.0, amber 0–1.0, red < 0.

### Drawdown chart
Filled area showing depth of each drawdown from the prior peak.

### Today's signals table
Live signals from Phase 3 for the current date. Includes predicted close price,
z-score, and position size.

### Open positions
All currently held trades with entry price and allocated capital.

### Recent closed trades
Last 30 days of completed trades with predicted vs. actual return.

### P&L by ticker
Horizontal bar chart showing which stocks contributed most to total P&L.

### Ticker deep-dive (sidebar)
Select any Nifty50 stock to see its price chart, sentiment history, and trade log.

### Pipeline health (expandable)
Row counts for every DuckDB table + recent scheduler run log.

---

## Running in the background permanently (Windows)

### Option A — Batch files (simplest)

Create `start_all.bat` in `alpha_project\`:
```bat
@echo off
start "Phase 1" cmd /k "cd phase1 && ..\venv\Scripts\activate && python scheduler.py"
start "Phase 2" cmd /k "cd phase2 && ..\venv\Scripts\activate && python scheduler_phase2.py"
start "Phase 3" cmd /k "cd phase3 && ..\venv\Scripts\activate && python scheduler_phase3.py"
start "Phase 4" cmd /k "cd phase4 && ..\venv\Scripts\activate && python scheduler_phase4.py"
start "Dashboard" cmd /k "cd phase4 && ..\venv\Scripts\activate && streamlit run dashboard.py"
```

Double-click `start_all.bat` to launch everything with one click.

### Option B — Windows Task Scheduler (auto-start on boot)

For each scheduler (replace paths with your actual paths):

1. Open Task Scheduler (`Win + S` → "Task Scheduler")
2. Create Basic Task → Name: `Alpha Phase 1`
3. Trigger: "When the computer starts"
4. Action: Start a program
   - Program: `C:\Users\YourName\alpha_project\venv\Scripts\python.exe`
   - Arguments: `scheduler.py`
   - Start in: `C:\Users\YourName\alpha_project\phase1`
5. Repeat for phases 2, 3, 4, and the dashboard

---

## Querying the dashboard data directly

```powershell
python -c "
import duckdb
con = duckdb.connect(r'..\phase1\db\alpha.duckdb', read_only=True)

# Portfolio summary
print('=== Portfolio Snapshots (last 5) ===')
print(con.execute('''
    SELECT snapshot_date, portfolio_value, daily_pnl, cum_return, open_positions
    FROM   portfolio_snapshots ORDER BY snapshot_date DESC LIMIT 5
''').df().to_string(index=False))

# Best and worst trades
print('\n=== Top 5 Winning Trades ===')
print(con.execute('''
    SELECT ticker, direction, entry_date, exit_date,
           entry_price, exit_price, pnl_net
    FROM   trades WHERE status = 'closed' AND pnl_net IS NOT NULL
    ORDER  BY pnl_net DESC LIMIT 5
''').df().to_string(index=False))

# Strategy vs benchmark
print('\n=== Strategy vs Benchmark ===')
print(con.execute('''
    SELECT p.snapshot_date,
           p.portfolio_value AS strategy,
           b.portfolio_value AS nifty50
    FROM   portfolio_snapshots p
    LEFT   JOIN benchmark_snapshots b USING (snapshot_date)
    ORDER  BY p.snapshot_date DESC LIMIT 5
''').df().to_string(index=False))

con.close()
"
```

---

## Troubleshooting (Windows-specific)

**Dashboard opens but shows "No portfolio snapshots yet"**
Run the backtester first: `python backtester.py`

**"Streamlit is not recognized as a command"**
Venv is not activated. Run: `..\venv\Scripts\activate`

**"ModuleNotFoundError: No module named 'data_loader'"**
You are not in the `phase4\` directory. Run: `cd C:\...\phase4`

**Dashboard shows stale data after running the updater**
Click the "🔄 Refresh now" button in the sidebar,
or wait up to 60 seconds for the cache to expire.

**"duckdb.IOException: database is locked"**
One of the write schedulers is mid-write. Retry in 5 seconds.
Read-only connections (dashboard) should never cause this —
if the dashboard itself causes a lock, ensure it only uses `read_only=True`.

**Port 8501 already in use**
```powershell
streamlit run dashboard.py --server.port 8502
```

**Drawdown chart is empty**
`portfolio_snapshots` has only one row.
Run the backtester over a longer history: `python backtester.py`

---

## Configuration reference

| File | Variable | Default | Effect |
|---|---|---|---|
| `backtester.py` | `INITIAL_CAPITAL` | `1_000_000` | Starting capital (₹10 lakhs) |
| `backtester.py` | `COST_BPS` | `20` | Round-trip transaction cost |
| `daily_pnl_updater.py` | `INITIAL_CAPITAL` | `1_000_000` | Must match backtester |
| `daily_pnl_updater.py` | `COST_RATE` | `20/10000` | Must match backtester |
| `dashboard.py` | `ttl=60` (cache) | 60 sec | How often data refreshes |
| `scheduler_phase4.py` | `16:30 IST` | — | Post-close update time |
