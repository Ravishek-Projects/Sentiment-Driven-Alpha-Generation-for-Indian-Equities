# Phase 2 — Feature Engineering
## Complete Setup and Run Guide

---

## What Phase 2 does

Phase 2 reads the raw articles and price data stored by Phase 1 and produces
a clean, model-ready **feature matrix** — one row per (Nifty50 ticker, trading date)
containing 19 numerical features plus a prediction target.

```
Phase 1 tables                    Phase 2 tables
──────────────                    ──────────────
raw_news          ──┐
tagged_articles   ──┤──► sentiment_engine.py  ──► sentiment_scores
                    │
ohlcv             ──┤──► technical_features.py──► technical_features
                    │
                    └──► feature_assembler.py ──► feature_matrix  ◄── model reads this
```

### Feature vector (19 dimensions per row)

| Group | Features | Dim |
|---|---|---|
| Sentiment (FinBERT) | sent_pos, sent_neg, sent_neu, sent_spread, news_count | 5 |
| Technical | rsi_14, price_to_ma20, vol_20d, rel_volume, bb_position, lag_ret_1d, lag_ret_2d | 7 |
| Day-of-week (one-hot) | dow_mon … dow_fri | 5 |
| Stored (not model input) | ma_20, ma_50 | 2 |
| **Target** | target_ret_1d (filled post-close) | 1 |

---

## File structure

```
phase2/
├── db_schema_phase2.py       # Adds 3 new tables to alpha.duckdb
├── db_writer.py              # Updated db_writer (Phase 1 + Phase 2 writers)
├── sentiment_engine.py       # FinBERT inference + aggregation
├── technical_features.py     # RSI, Bollinger, MA, vol, etc.
├── feature_assembler.py      # Joins sentiment + technical → feature_matrix
├── scheduler_phase2.py       # APScheduler for daily Phase 2 pipeline
├── verify_phase2.py          # End-to-end checks
└── requirements_phase2.txt
```

---

## Step-by-step setup

### Prerequisites

- Phase 1 fully set up and `db/alpha.duckdb` exists
- The Phase 1 virtual environment (`venv`) is activated
- At least 4 GB RAM (FinBERT base model is ~440 MB)
- Internet access for first FinBERT download (~440 MB, cached after that)

### Step 1 — Activate the Phase 1 environment

```bash
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### Step 2 — Install Phase 2 dependencies

```bash
pip install -r requirements_phase2.txt
```

**CPU-only machines** (default): the above command installs the CPU build of PyTorch.
FinBERT inference takes ~2–5 seconds per 32 headlines on CPU — fast enough.

**GPU machines** (optional, much faster for large backtills):
```bash
# Check your CUDA version first
nvidia-smi

# Install GPU PyTorch (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_phase2.txt
```

### Step 3 — Replace db_writer.py

Copy the updated `db_writer.py` from this folder into your `phase1/` folder,
replacing the old one. It is backward-compatible (all Phase 1 functions unchanged).

```bash
cp phase2/db_writer.py phase1/db_writer.py
```

### Step 4 — Add Phase 2 tables to the database

Run this once. It adds `sentiment_scores`, `technical_features`, and `feature_matrix`
to the existing `db/alpha.duckdb`:

```bash
cd phase2
python db_schema_phase2.py
```

Expected output:
```
[schema_phase2] Phase 2 tables added to db/alpha.duckdb
[schema_phase2] New tables: sentiment_scores, technical_features, feature_matrix
```

### Step 5 — Backfill historical data (important before running the model)

Run these three backfills in order. Each reads from the previous step's output.

**5a — Technical features (last 252 trading days ≈ 1 year)**
```bash
python technical_features.py --backfill --days 365
```
This processes all Nifty50 tickers for each of the past 365 calendar days.
Rows with insufficient OHLCV history (< 60 days) are skipped automatically.
Expected time: 3–8 minutes.

**5b — Sentiment scoring (backfill using articles already in DB)**
```bash
python sentiment_engine.py --backfill --days 90
```
Runs FinBERT on all tagged articles from the past 90 days.
First run downloads the FinBERT model (~440 MB). Subsequent runs use the local cache.
Expected time: 5–20 minutes depending on article volume and CPU/GPU.

**5c — Assemble the feature matrix**
```bash
python feature_assembler.py --backfill --days 90
```
Joins sentiment + technical into `feature_matrix`. Also fills `target_ret_1d`
for all rows where the next day's OHLCV is available.
Expected time: < 1 minute.

### Step 6 — Verify Phase 2 setup

```bash
python verify_phase2.py
```

To skip the slow FinBERT model load (if you ran it successfully in step 5b):
```bash
python verify_phase2.py --skip-finbert
```

Expected output:
```
── Phase 2 verification ──────────────────────────────────────

[Imports]
  [OK]  torch
  [OK]  transformers
  [OK]  sentiment_engine importable
  [OK]  technical_features importable
  [OK]  feature_assembler importable
  [OK]  db_writer phase2 functions present

[Database schema]
  [OK]  Phase 2 tables created
  [OK]  sentiment_scores columns correct
  [OK]  feature_matrix has 19+ feature columns

[DB writer — Phase 2]
  [OK]  insert_sentiment_scores works
  [OK]  insert_technical_features works
  [OK]  insert_feature_matrix works

[Technical indicators — unit tests]
  [OK]  RSI computes correctly on known series
  [OK]  RSI returns nan for too-short series
  [OK]  Bollinger Band position in [0,1] for typical price
  [OK]  _compute_for_ticker returns None with insufficient history

[Sentiment aggregation logic]
  [OK]  _aggregate_scores means correctly
  [OK]  _aggregate_scores returns neutral for empty list

[Feature assembler logic]
  [OK]  _dow_encode gives correct one-hot
  [OK]  _ffill_sentiment forward-fills missing dates
  [OK]  _ffill_sentiment inserts neutral beyond MAX_FFILL_DAYS

[FinBERT model]
  [OK]  FinBERT loads and scores one headline

── Summary ────────────────────────────────────────────────────
  20 passed  /  0 failed  /  20 total

All Phase 2 checks passed.
```

### Step 7 — Run a single-cycle test

```bash
python scheduler_phase2.py --run-once
```

This runs the full pipeline once (technical → sentiment → assemble → fill targets)
and exits. Check the output carefully before starting the continuous scheduler.

### Step 8 — Start the continuous Phase 2 scheduler

Run this **in a second terminal window** alongside the Phase 1 scheduler:

```bash
# Terminal 1 (Phase 1)
python phase1/scheduler.py

# Terminal 2 (Phase 2)
cd phase2
python scheduler_phase2.py
```

---

## Running in the background

### Linux — two systemd services

Create `/etc/systemd/system/phase2.service`:
```ini
[Unit]
Description=Phase 2 Feature Engineering Pipeline
After=network.target phase1.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/phase2
ExecStart=/home/ubuntu/phase1/venv/bin/python scheduler_phase2.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable phase2
sudo systemctl start phase2
sudo systemctl status phase2
```

---

## Querying the feature matrix

```python
import duckdb
con = duckdb.connect("db/alpha.duckdb", read_only=True)

# Latest features for all tickers (today)
df = con.execute("""
    SELECT ticker, date, sent_spread, rsi_14, vol_20d, target_ret_1d
    FROM   feature_matrix
    WHERE  date = current_date
    ORDER  BY sent_spread DESC
""").df()
print(df)

# Historical feature matrix for the model (has targets)
training_df = con.execute("""
    SELECT *
    FROM   feature_matrix
    WHERE  target_ret_1d IS NOT NULL
    ORDER  BY ticker, date
""").df()
print(f"Training rows: {len(training_df)}")

# Sentiment history for RELIANCE
sent_df = con.execute("""
    SELECT trading_date, sent_pos, sent_neg, sent_spread, news_count
    FROM   sentiment_scores
    WHERE  ticker = 'RELIANCE'
    ORDER  BY trading_date DESC
    LIMIT  10
""").df()
print(sent_df)

con.close()
```

---

## How each script works internally

### sentiment_engine.py

1. Queries `tagged_articles JOIN raw_news` for the target trading dates
2. Groups by (ticker, trading_date)
3. For each group, collects up to 32 headlines (most recent if more exist)
4. Runs `ProsusAI/finbert` via HuggingFace `pipeline("text-classification")`
   in batches of 32 — returns (positive, negative, neutral) probability triples
5. Averages the triples across all articles in the group
6. Writes aggregated scores to `sentiment_scores`
7. For tickers with zero articles, writes neutral scores (1/3, 1/3, 1/3)

The FinBERT model is loaded once at process start and stays in memory.
On CPU, scoring 32 headlines takes ~2 seconds. On GPU, ~0.1 seconds.

### technical_features.py

Pulls the last 120 days of OHLCV for each ticker and computes:

- **RSI-14**: Wilder's method — EMA of gains and losses, no lookahead
- **MA-20 / MA-50**: Simple moving averages of close price
- **Price-to-MA20**: `close / MA20 - 1` — positive = above average (momentum)
- **Vol-20d**: Std of log returns × sqrt(252) — annualised realised vol
- **Rel-volume**: `today_volume / avg_volume_last_20_days` — unusual activity
- **BB-position**: `(close - lower_band) / (upper_band - lower_band)` — 0=bottom, 1=top
- **Lag returns**: Yesterday's and day-before-yesterday's log returns

All calculations use only data up to and including `target_date` — zero lookahead.

### feature_assembler.py

1. Loads sentiment from `sentiment_scores` with a 5-day lookback buffer
2. Loads technical features from `technical_features`
3. Forward-fills sentiment gaps up to 5 trading days (weekends, holidays, silent stocks)
4. Merges on (ticker, date) with a LEFT JOIN (technical is the anchor)
5. Applies `log1p` to `news_count` so heavy-news days don't dominate
6. Encodes day-of-week as five binary columns
7. Writes to `feature_matrix` (target is NULL until `fill_targets()` runs)
8. `fill_targets()` looks up T+1 log return from `ohlcv` and fills the target column

---

## Configuration reference

| File | Variable | Default | Description |
|---|---|---|---|
| `sentiment_engine.py` | `FINBERT_MODEL` | `ProsusAI/finbert` | HuggingFace model ID |
| `sentiment_engine.py` | `BATCH_SIZE` | `32` | Articles per inference call |
| `sentiment_engine.py` | `MAX_ARTICLES` | `32` | Max articles per (ticker, date) |
| `sentiment_engine.py` | `MAX_TOKEN_LEN` | `512` | Token truncation limit |
| `technical_features.py` | `MIN_HISTORY_DAYS` | `60` | Min OHLCV rows before computing |
| `feature_assembler.py` | `MAX_FFILL_DAYS` | `5` | Max days to forward-fill sentiment |

---

## Troubleshooting

**`OSError: model not found` or `ConnectionError` when loading FinBERT**
HuggingFace downloads models from the internet on first use.
Check your internet connection and retry. Models are cached in `~/.cache/huggingface/`.

**FinBERT is very slow on CPU**
Expected: ~2s per 32 headlines. For 50 tickers × 10 articles = 500 headlines,
that is ~30 seconds per scoring run. This is fine for the 30-minute batch cadence.
If you need it faster, use a GPU (see Step 2 above).

**`feature_matrix` rows have NULL for technical features**
This means `technical_features` has no rows for that date.
Run: `python technical_features.py --date YYYY-MM-DD --force`

**`target_ret_1d` still NULL after running fill_targets**
The next day's OHLCV hasn't been downloaded yet.
Run: `python ohlcv_downloader.py --mode topup` (from phase1/)
Then: `python feature_assembler.py --fill-targets`

**`insert_technical_features: missing columns [...]`**
You're running the old `db_writer.py` (Phase 1 only). Copy the Phase 2 version:
`cp phase2/db_writer.py phase1/db_writer.py`
