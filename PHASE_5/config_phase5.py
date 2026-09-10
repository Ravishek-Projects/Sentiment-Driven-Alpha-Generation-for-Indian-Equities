# config_phase5.py
# Central configuration for Phase 5 — Evaluation, Monitoring, and Retraining.
# Import this at the top of every Phase 5 module.
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared paths (mirror Phase 3 layout so we can load trained models)
# ---------------------------------------------------------------------------
BASE_DIR         = Path(__file__).parent
# Use the exact folder names from your Windows environment
PHASE3_DIR       = BASE_DIR.parent / "PHASE_3"
# --- FIX: Dynamically import FEATURE_COLS from Phase 3 ---
sys.path.append(str(PHASE3_DIR))
from config import FEATURE_COLS
# Point directly to the root db folder we established earlier
DB_PATH          = BASE_DIR.parent / "db" / "alpha.duckdb"
CHECKPOINTS_DIR  = PHASE3_DIR / "checkpoints"
SCALERS_DIR      = PHASE3_DIR / "scalers"

REPORTS_DIR      = BASE_DIR / "reports"
EVAL_DIR         = BASE_DIR / "eval_results"
MONITOR_DIR      = BASE_DIR / "monitor"
LOGS_DIR         = BASE_DIR / "logs"

for _d in [REPORTS_DIR, EVAL_DIR, MONITOR_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------
WF_INITIAL_TRAIN_DAYS = 180     # minimum history before first test window
WF_TEST_WINDOW_DAYS   = 21      # ~one calendar month per fold
WF_STEP_DAYS          = 21      # non-overlapping folds (expand-window)
WF_MIN_TEST_SAMPLES   = 10      # skip fold if fewer test samples than this

# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------
BOOTSTRAP_N_SAMPLES   = 1_000   # bootstrap resamples for CI
CONFIDENCE_LEVEL      = 0.95    # 95% confidence intervals
DM_TEST_H0_VALUE      = 0.0     # null hypothesis: no difference in forecast errors

# ---------------------------------------------------------------------------
# Model health monitoring thresholds (auto-trigger retraining)
# ---------------------------------------------------------------------------
MONITOR_WINDOW_DAYS    = 30     # rolling window for health metrics
SHARPE_RETRAIN_THRESH  = 0.50   # rolling Sharpe below this → alert
HIT_RATE_RETRAIN_THRESH= 0.48   # rolling hit rate below this → alert
IC_RETRAIN_THRESH      = 0.02   # rolling IC below this → alert
SENTIMENT_DRIFT_THRESH = 0.15   # KL divergence above this → sentiment drift alert

# ---------------------------------------------------------------------------
# Retraining schedule
# ---------------------------------------------------------------------------
RETRAIN_LOOKBACK_DAYS  = 365    # how many days of history to use each retrain
RETRAIN_EPOCHS         = 30     # quick retrain epochs (warm-start from checkpoint)
RETRAIN_PATIENCE       = 8
RETRAIN_LR_FACTOR      = 0.1    # multiply Phase 3 LR by this for fine-tuning

# ---------------------------------------------------------------------------
# Benchmark tickers for comparison (Phase 5 uses these directly)
# ---------------------------------------------------------------------------
NIFTY50_INDEX_YF = "^NSEI"      # yfinance symbol for Nifty 50 index
MOMENTUM_LOOKBACK = 126          # 6-month momentum (trading days)
MOMENTUM_SKIP     = 21           # skip most recent month (standard momentum)

# ---------------------------------------------------------------------------
# Report settings
# ---------------------------------------------------------------------------
REPORT_DATE_FORMAT = "%Y-%m-%d"
