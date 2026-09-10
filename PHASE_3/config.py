# config.py
# Single source of truth for all Phase 3 hyperparameters, paths, and constants.
# Import this at the top of every Phase 3 module instead of hard-coding values.

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
DB_PATH         = Path(__file__).parent.parent / "db" / "alpha.duckdb"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
SCALERS_DIR     = BASE_DIR / "scalers"
LOGS_DIR        = BASE_DIR / "logs"
PREDICTIONS_DIR = BASE_DIR / "predictions"

# Create dirs if they don't exist
for _d in [CHECKPOINTS_DIR, SCALERS_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature columns  (must match feature_matrix table exactly)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # sentiment (5)
    "sent_pos", "sent_neg", "sent_neu", "sent_spread", "news_count",
    # technical (7) — ALL features restored
    "rsi_14", "price_to_ma20", "vol_20d", "rel_volume", "bb_position",
    "lag_ret_1d", "lag_ret_2d", 
    # day-of-week (5)
    "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri",
]
N_FEATURES = len(FEATURE_COLS)   # 17  (ma_20, ma_50 stored but not used as model input)

TARGET_COL  = "target_ret_1d"
SEQ_LEN     = 30                 # look-back window in trading days

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70    # 70% train
VAL_RATIO   = 0.15    # 15% validation
TEST_RATIO  = 0.15    # 15% test  (walk-forward, chronological)

# ---------------------------------------------------------------------------
# LSTM hyperparameters
# ---------------------------------------------------------------------------
LSTM_HIDDEN   = 64     # was 128
LSTM_LAYERS   = 2      # was 2
LSTM_DROPOUT  = 0.25   # was 0.30
LSTM_LR = 1e-3  # Reduced from 1e-3
LSTM_WEIGHT_DECAY = 1e-4
LSTM_EPOCHS     = 150
LSTM_BATCH      = 256
LSTM_PATIENCE   = 20             # early-stopping patience

# ---------------------------------------------------------------------------
# Transformer hyperparameters
# ---------------------------------------------------------------------------
TF_D_MODEL      = 32             # embedding dimension (keep small — 17 features)
TF_NHEAD        = 4              # attention heads  (d_model must be divisible by nhead)
TF_DIM_FF       = 64           # feed-forward inner dim
TF_N_LAYERS     = 2              # encoder layers
TF_DROPOUT      = 0.2
TF_LR   = 5e-4   # Reduced from 5e-4
TF_WEIGHT_DECAY = 1e-4
TF_EPOCHS       = 150
TF_BATCH        = 64
TF_PATIENCE     = 20

# ---------------------------------------------------------------------------
# Ensemble weights  (tuned on validation set; Transformer slightly favoured)
# ---------------------------------------------------------------------------
ENSEMBLE_W_LSTM = 0.7
ENSEMBLE_W_TF   = 0.3

# ---------------------------------------------------------------------------
# Signal generation thresholds
# ---------------------------------------------------------------------------
SIGNAL_Z_THRESHOLD  = 0.5       # z-score above this → LONG, below -this → SHORT
SIGNAL_TARGET_VOL   = 0.15       # annualised target portfolio vol (15%)
SIGNAL_MAX_POSITION = 0.10       # max single-stock weight (10%)
SIGNAL_TOP_N        = 5          # max longs + max shorts per day

# ---------------------------------------------------------------------------
# Scaler checkpoint names
# ---------------------------------------------------------------------------
SCALER_NAME = "feature_scaler.pkl"
