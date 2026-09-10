# Phase 5 — Evaluation, Monitoring & Retraining
## Complete Windows Terminal Setup and Run Guide

---

## What Phase 5 does

Phase 5 is the rigorous validation and continuous improvement layer.
It answers four questions every quantitative strategy must answer:

1. **Is the backtest honest?** → Walk-forward cross-validation
2. **Is the performance real or lucky?** → Statistical significance tests
3. **Does it beat simple benchmarks?** → Factor regression and alpha decomposition
4. **Does it stay good over time?** → Daily health monitoring + auto-retraining

```
Phase 4 output (trades, portfolio_snapshots, predictions)
        │
        ├──► walk_forward_eval.py    — expanding-window CV across all folds
        ├──► statistical_tests.py   — bootstrap CI, Deflated Sharpe, DM test
        ├──► benchmark_comparison.py— vs. Nifty50, momentum, value + α regression
        ├──► model_monitor.py       — daily Sharpe/hit-rate/IC/drift health check
        ├──► incremental_retrain.py — warm-start fine-tune on latest data
        ├──► report_generator.py    — self-contained HTML report
        └──► scheduler_phase5.py    — orchestrates all jobs on schedule
```

---

## File structure

```
phase5\
├── config_phase5.py          ← all thresholds, paths, window sizes
├── walk_forward_eval.py      ← expanding-window walk-forward evaluation
├── statistical_tests.py      ← bootstrap CI, Deflated Sharpe, DM test, paired t
├── benchmark_comparison.py   ← vs. Nifty50, momentum, value; factor α regression
├── model_monitor.py          ← daily Sharpe / hit-rate / IC / drift checks
├── incremental_retrain.py    ← warm-start retraining on latest N days
├── report_generator.py       ← self-contained HTML performance report
├── scheduler_phase5.py       ← APScheduler for all Phase 5 jobs
├── verify_phase5.py          ← automated checks (no trained model needed)
├── requirements_phase5.txt
│
├── eval_results\             ← auto-created
│   ├── walk_forward_results.csv
│   ├── statistical_tests.json
│   └── benchmark_comparison.json
│
├── monitor\                  ← auto-created
│   ├── alerts.jsonl
│   └── health_history.csv
│
├── reports\                  ← auto-created
│   └── alpha_report_YYYYMMDD_HHMMSS.html
│
└── logs\
    └── scheduler_phase5.log
```

---

## Prerequisites

Before starting Phase 5 you need:

1. Phase 3 training complete (`checkpoints\lstm_best.pt` and `checkpoints\transformer_best.pt` exist)
2. Phase 4 backtester run (`portfolio_snapshots` table populated)
3. At least **60 trading days** of portfolio history for meaningful metrics
4. At least **90 days** of `feature_matrix` rows with targets for walk-forward

---

## Windows Terminal — Step-by-Step

### Step 1 — Open Windows Terminal and navigate

```powershell
# Win + X → Windows Terminal (or search "Windows Terminal")
cd C:\Users\YourName\alpha_project\phase5

# Activate the shared virtual environment
..\venv\Scripts\activate

# Confirm location and activation
dir
# Should show: config_phase5.py  walk_forward_eval.py  ...
```

---

### Step 2 — Install Phase 5 dependencies

```powershell
pip install -r requirements_phase5.txt
```

This adds scipy and matplotlib (optional). Under 1 minute.

---

### Step 3 — Run the verification suite

```powershell
python verify_phase5.py
```

Expected output:
```
── Phase 5 verification ──────────────────────────────────────

[Imports]
  [OK]  config_phase5 importable
  [OK]  walk_forward_eval importable
  [OK]  statistical_tests importable
  [OK]  benchmark_comparison importable
  [OK]  model_monitor importable
  [OK]  incremental_retrain importable
  [OK]  report_generator importable
  [OK]  scipy available
  [OK]  matplotlib available (optional)

[Statistical tests — unit tests]
  [OK]  bootstrap_ci on known data
  [OK]  bootstrap_ci CI width narrows with more data
  [OK]  deflated_sharpe_ratio — higher n_trials lowers DSR
  [OK]  deflated_sharpe_ratio output in [0, 1]
  [OK]  diebold_mariano_test — identical errors → p ≈ 1 (no difference)
  [OK]  diebold_mariano_test — clearly better model detected
  [OK]  paired_hit_rate_test — detects significant difference

[Benchmark comparison]
  [OK]  build_market_factor computes mean return by date
  [OK]  factor_regression returns alpha and betas

[Model monitor]
  [OK]  check_rolling_sharpe — high return series → OK
  [OK]  check_rolling_sharpe — flat returns → alert
  [OK]  check_hit_rate — 70% wins → OK
  [OK]  check_sentiment_drift — identical distributions → OK

[Incremental retrain]
  [OK]  run_retrain dry-run returns without error

[Report generator]
  [OK]  generate_report creates HTML file with empty DB

── Summary ────────────────────────────────────────────────────
  22 passed  /  0 failed  /  22 total

All Phase 5 checks passed. Ready to run evaluation pipeline.
```

---

### Step 4 — Run the daily health check manually

Test the monitoring system against your live portfolio data:

```powershell
python model_monitor.py
```

Example output:
```
2024-03-28 17:00:01 [INFO] [monitor] Health check for 2024-03-28
2024-03-28 17:00:02 [INFO] [monitor] OK    — sharpe: Rolling 30-day Sharpe = 1.8372 (threshold 0.5)
2024-03-28 17:00:02 [INFO] [monitor] OK    — hit_rate: Rolling hit rate = 61.2% over 87 trades (threshold 48%)
2024-03-28 17:00:02 [INFO] [monitor] OK    — ic: Rolling IC = 0.0823 (threshold 0.02)
2024-03-28 17:00:02 [INFO] [monitor] OK    — bias: Wrong direction rate = 38.8% (>55% = systematic bias)
2024-03-28 17:00:02 [INFO] [monitor] OK    — drift: Sentiment KL divergence = 0.0412 (threshold 0.15)
2024-03-28 17:00:02 [INFO] [monitor] Done — all OK
```

If any metric fails its threshold, you'll see ALERT messages and the issue
gets logged to `monitor\alerts.jsonl`.

To run the monitor AND trigger retraining automatically if alerts fire:
```powershell
python model_monitor.py --retrain-on-alert
```

---

### Step 5 — Run walk-forward evaluation

This is the most important evaluation step. It simulates what happens when
you retrain on all available data up to date T and test on T+1 month,
repeating across the full history.

**Quick test (5 folds, ~5 minutes):**
```powershell
python walk_forward_eval.py --quick --no-retrain
```

**Full run with retraining per fold (recommended, but slow — hours):**
```powershell
python walk_forward_eval.py
```

**Full run WITHOUT fold retraining (much faster, reasonable results):**
```powershell
python walk_forward_eval.py --no-retrain
```

Example output:
```
2024-03-28 17:05:00 [INFO] [wf] Walk-forward evaluation on cpu
2024-03-28 17:05:01 [INFO] [wf] 312 unique trading dates, 50 tickers
2024-03-28 17:05:03 [INFO] [wf] Fold   1 | train ≤ 2023-07-03 (15000 rows) | test 2023-07-04 → 2023-07-25 (500 rows)
2024-03-28 17:05:11 [INFO] [wf] Fold   2 | train ≤ 2023-07-25 (15500 rows) | test 2023-07-26 → 2023-08-16 (500 rows)
...
── Walk-forward summary (mean across folds) ─────────────────
  hit_rate_lstm        : 0.5312
  hit_rate_tf          : 0.5401
  hit_rate_ens         : 0.5491
  ic_lstm              : 0.0621
  ic_tf                : 0.0734
  ic_ens               : 0.0812
  rmse_lstm            : 0.009241
  rmse_tf              : 0.008973
  rmse_ens             : 0.008741

Results saved to eval_results\walk_forward_results.csv
```

**What good results look like:**
- Hit rate > 52% consistently across folds (not just the final test set)
- IC > 0.04 on average (meaningful predictive correlation)
- Ensemble should beat both individual models on most folds

---

### Step 6 — Run statistical significance tests

Run this after Step 5 so it has the walk-forward CSV to work from:

```powershell
python statistical_tests.py
```

Example output:
```
2024-03-28 17:30:01 [INFO] [stat] Loaded 14 folds from eval_results\walk_forward_results.csv
2024-03-28 17:30:01 [INFO] [stat] Ensemble hit rate: 0.5491 [0.5124, 0.5853] 95% CI
2024-03-28 17:30:01 [INFO] [stat] Ensemble IC:       0.0812 [0.0341, 0.1283] 95% CI
2024-03-28 17:30:01 [INFO] [stat] Deflated Sharpe: SR=1.8372  DSR_prob=0.9124  significant=True
2024-03-28 17:30:02 [INFO] [stat] DM test (LSTM vs TF): Fail to reject H₀ (p=0.2341) — no significant difference
2024-03-28 17:30:02 [INFO] [stat] Paired t (Ensemble vs LSTM): Reject H₀ (p=0.0312) — Ensemble is significantly better
2024-03-28 17:30:02 [INFO] [stat] Results saved → eval_results\statistical_tests.json
```

**How to interpret these results:**

- `Ensemble hit rate: 0.5491 [0.5124, 0.5853] 95% CI` — the true hit rate is between 51% and 59% with 95% confidence. The lower bound > 50% means the model is genuinely better than random.
- `DSR_prob=0.9124` — 91% probability the Sharpe is genuine after adjusting for the 4 model variants tested. Above 0.95 is considered significant.
- `DM test p=0.2341` — LSTM and Transformer are NOT significantly different from each other. The ensemble helps precisely because they're complementary, not identical.
- `Paired t p=0.0312 — Ensemble significantly better` — the ensemble significantly outperforms LSTM alone across walk-forward folds.

---

### Step 7 — Run benchmark comparison

```powershell
python benchmark_comparison.py
```

Example output:
```
── Performance comparison ──────────────────────────────────
  Strategy               Total%  Ann%     Vol%   Sharpe  MaxDD%
  Sentiment strategy     +21.44  +28.12   15.31    1.84   -6.24
  Nifty50 equal-weight   +12.18  +15.89   14.87    1.07  -11.34
  Pure momentum           +8.43  +10.91   18.22    0.60  -14.52
  Pure value              +6.21   +8.04   16.44    0.49  -16.88

[comp] Running factor regression ...
  Alpha (ann): 0.1241%   ← 12.41% annualised daily alpha
  t-stat α:    2.4121
  p-value α:   0.0163
  Significant: True
```

**How to interpret:**

- Strategy beats Nifty50 by 12% total return — that's meaningful.
- The factor regression alpha of 12.41% per year (p=0.016) means even after explaining away market, momentum, and value exposures, the sentiment signal adds genuine return. This is your core research claim.
- t-stat > 2 and p < 0.05 makes it publishable-quality evidence.

Results saved to `eval_results\benchmark_comparison.json`.

---

### Step 8 — Run incremental retraining

Test a dry run first:
```powershell
python incremental_retrain.py --dry-run
```

Then run the actual retraining (uses last 365 days, 30 epochs):
```powershell
python incremental_retrain.py
```

Example output:
```
2024-03-28 17:45:00 [INFO] [retrain] Incremental retrain — last 365 days, 30 epochs
2024-03-28 17:45:01 [INFO] [retrain] 18250 rows, 50 tickers
2024-03-28 17:45:02 [INFO] [retrain] Archived lstm_best.pt → lstm_best_20240328_174501.pt
2024-03-28 17:45:02 [INFO] [retrain] Fine-tuning LSTM ...
2024-03-28 17:45:02 [INFO] [retrain] Loaded existing lstm_best checkpoint
  [lstm] Epoch   1/30  val_loss=0.000241
  [lstm] Epoch   5/30  val_loss=0.000238
  [lstm] Epoch  12/30  val_loss=0.000233
  [lstm] Early stop at epoch 19
2024-03-28 17:46:30 [INFO] [retrain] Fine-tuning Transformer ...
  [transformer] Epoch   1/30  val_loss=0.000228
  ...
2024-03-28 17:47:50 [INFO] [retrain] Complete — ensemble hit rate: 0.5612
2024-03-28 17:47:50 [INFO] [retrain] Log appended → ..\phase3\checkpoints\retrain_log.jsonl
```

After retraining, Phase 3's `signal_generator.py` automatically uses the new
model on its next run (it always loads from `lstm_best.pt`).

To retrain on fewer days (faster, less history):
```powershell
python incremental_retrain.py --days 180 --epochs 20
```

---

### Step 9 — Generate the HTML report

```powershell
python report_generator.py
```

Output:
```
[report] Report saved → reports\alpha_report_20240328_174900.html
Open in browser: reports\alpha_report_20240328_174900.html
```

Open the HTML file in Chrome or Edge. The report contains:
- Portfolio summary KPIs
- Equity curve vs Nifty50 (embedded chart)
- Walk-forward fold-by-fold hit rates (embedded chart)
- P&L by ticker (embedded chart)
- Statistical test results
- Benchmark comparison table + factor regression alpha
- Model health alert history

The HTML file is completely self-contained — you can email it or save it
as a PDF using Ctrl+P → "Save as PDF" in Chrome.

---

### Step 10 — Start the Phase 5 scheduler

Open a **fifth Windows Terminal tab** alongside the four Phase 1-4 tabs.

```powershell
..\venv\Scripts\activate
cd ..\phase5
python scheduler_phase5.py
```

Or, to auto-retrain when health alerts fire:
```powershell
python scheduler_phase5.py --retrain-on-alert
```

The scheduler runs:
- Daily at 17:00 IST — health check
- Every Sunday at 02:00 IST — full walk-forward evaluation
- Every Sunday at 04:00 IST — statistical tests
- Every Sunday at 05:00 IST — benchmark comparison
- Every Sunday at 05:30 IST — report generation

---

## All five terminal tabs — final state

| Tab | Directory | Command |
|---|---|---|
| 1 | `phase1\` | `python scheduler.py` |
| 2 | `phase2\` | `python scheduler_phase2.py` |
| 3 | `phase3\` | `python scheduler_phase3.py` |
| 4 | `phase4\` | `python scheduler_phase4.py` |
| 5 | `phase5\` | `python scheduler_phase5.py` |
| Browser | — | `http://localhost:8501` (Streamlit dashboard) |

---

## One-command startup (all phases)

Create `start_all.bat` in the `alpha_project\` root folder:

```bat
@echo off
title Alpha Pipeline

start "Phase1-RSS"       cmd /k "cd /d %~dp0phase1 && ..\venv\Scripts\activate && python scheduler.py"
timeout /t 5 /nobreak >nul
start "Phase2-Features"  cmd /k "cd /d %~dp0phase2 && ..\venv\Scripts\activate && python scheduler_phase2.py"
timeout /t 5 /nobreak >nul
start "Phase3-Signals"   cmd /k "cd /d %~dp0phase3 && ..\venv\Scripts\activate && python scheduler_phase3.py"
timeout /t 5 /nobreak >nul
start "Phase4-PnL"       cmd /k "cd /d %~dp0phase4 && ..\venv\Scripts\activate && python scheduler_phase4.py"
timeout /t 5 /nobreak >nul
start "Phase5-Eval"      cmd /k "cd /d %~dp0phase5 && ..\venv\Scripts\activate && python scheduler_phase5.py"
timeout /t 5 /nobreak >nul
start "Dashboard"        cmd /k "cd /d %~dp0phase4 && ..\venv\Scripts\activate && streamlit run dashboard.py"

echo All processes started. Close this window to keep them running.
```

Double-click `start_all.bat` → all six windows open automatically.

---

## Interpreting the monitoring alerts

### What each alert means and what to do

| Alert | Cause | Action |
|---|---|---|
| `sharpe < 0.5` | Model losing money consistently | Run `incremental_retrain.py` |
| `hit_rate < 48%` | Model predicting wrong direction | Check sentiment drift first; then retrain |
| `ic < 0.02` | No predictive correlation | Inspect recent news quality; check feed availability |
| `sentiment_drift > 0.15` | RSS content has shifted vocabulary | Retrain FinBERT fine-tune in Phase 2 |
| `wrong_dir_rate > 55%` | Systematic sign error in predictions | Check feature scaling; retrain |

### How to check recent alerts

```powershell
python -c "
from model_monitor import load_alerts
alerts = load_alerts(20)
for a in alerts:
    print(a['date'], a.get('metric','?'), a.get('status','?'), a.get('message',''))
"
```

### How to view health history

```powershell
python -c "
from model_monitor import load_health_history
df = load_health_history()
print(df.tail(10).to_string(index=False))
"
```

---

## Configuration quick reference

Edit `config_phase5.py` to tune Phase 5 behaviour:

| Variable | Default | Effect |
|---|---|---|
| `WF_INITIAL_TRAIN_DAYS` | 180 | Min training days before first fold |
| `WF_TEST_WINDOW_DAYS` | 21 | Trading days in each test fold (~1 month) |
| `SHARPE_RETRAIN_THRESH` | 0.50 | Rolling Sharpe below this triggers alert |
| `HIT_RATE_RETRAIN_THRESH` | 0.48 | Hit rate below this triggers alert |
| `IC_RETRAIN_THRESH` | 0.02 | IC below this triggers alert |
| `SENTIMENT_DRIFT_THRESH` | 0.15 | KL divergence above this triggers drift alert |
| `RETRAIN_LOOKBACK_DAYS` | 365 | Days of history used per retrain |
| `RETRAIN_EPOCHS` | 30 | Fine-tuning epochs (warm start) |
| `RETRAIN_LR_FACTOR` | 0.1 | LR = Phase3_LR × this (smaller = gentler update) |
| `MONITOR_WINDOW_DAYS` | 30 | Rolling window for all health metrics |
| `BOOTSTRAP_N_SAMPLES` | 1000 | Bootstrap resamples for CI computation |

---

## Troubleshooting

**`FileNotFoundError: walk_forward_results.csv not found`**
Run walk-forward evaluation first: `python walk_forward_eval.py --quick`

**`RuntimeError: feature_matrix is empty`**
Phase 2 backfills haven't been run. See Phase 2 README.

**`RuntimeError: No portfolio_snapshots found`**
Phase 4 backtester hasn't been run. See Phase 4 README.

**Walk-forward takes too long**
Use `--no-retrain` for a fast run (minutes instead of hours):
`python walk_forward_eval.py --no-retrain`

**Report charts are missing**
Install matplotlib: `pip install matplotlib`

**`ModuleNotFoundError: No module named 'config'`** (from Phase 3)
Phase 3 must be in the Python path. All Phase 5 scripts add it automatically.
If it still fails, run: `set PYTHONPATH=..\phase3;%PYTHONPATH%`

**Sentiment drift alert fires immediately**
Normal if you have less than 60 days of history — the two halves of the window
are too small for stable distribution estimates. Threshold can be raised temporarily
in `config_phase5.py`: `SENTIMENT_DRIFT_THRESH = 0.30`
