# Phase 3 — Model Training and Inference
## Complete Windows Terminal Setup and Run Guide

---

## What Phase 3 does

Phase 3 reads the feature matrix built in Phase 2 and trains two neural
network models that predict next-day stock returns for all Nifty50 stocks.

```
feature_matrix (DuckDB)
        │
        ├──► dataset.py          — sliding-window sequences + RobustScaler
        │
        ├──► train_lstm.py       — trains LSTMPredictor
        ├──► train_transformer.py— trains TransformerPredictor
        │
        ├──► ensemble.py         — blends both models, ablation study
        │
        └──► signal_generator.py — pred → z-score → LONG/SHORT/FLAT → DuckDB
```

The `predictions` table written to DuckDB is what the Phase 4 dashboard reads.

---

## File structure

```
phase3\
├── config.py                  ← hyperparameters, paths, feature columns
├── dataset.py                 ← PyTorch Dataset + DataLoader factory
├── models.py                  ← LSTMPredictor + TransformerPredictor
├── trainer.py                 ← shared training loop, checkpointing, logging
├── train_lstm.py              ← trains LSTM, saves checkpoints\lstm_best.pt
├── train_transformer.py       ← trains Transformer, saves checkpoints\transformer_best.pt
├── ensemble.py                ← blended inference + ablation study
├── signal_generator.py        ← daily inference → signals → DuckDB
├── scheduler_phase3.py        ← APScheduler for daily signal generation
├── verify_phase3.py           ← automated checks
├── requirements_phase3.txt
│
├── checkpoints\               ← auto-created by training scripts
│   ├── lstm_best.pt
│   ├── transformer_best.pt
│   ├── lstm_results.json
│   └── transformer_results.json
│
├── scalers\                   ← auto-created
│   └── feature_scaler.pkl
│
└── logs\                      ← auto-created
    ├── lstm_loss.csv
    └── transformer_loss.csv
```

---

## Prerequisites

Before starting Phase 3:

1. Phase 1 `scheduler.py` has been running for at least several days
   (so `raw_news` and `ohlcv` have enough data)
2. Phase 2 backfills have been completed:
   ```
   python technical_features.py --backfill --days 365
   python sentiment_engine.py   --backfill --days 90
   python feature_assembler.py  --backfill --days 90
   ```
3. `feature_matrix` has rows with `target_ret_1d IS NOT NULL`
   (verify: `python feature_assembler.py --fill-targets`)

---

## Windows Terminal — Step-by-Step

### Important Windows notes before you start

- Use **Windows Terminal** (not the old `cmd.exe`).
  Install from the Microsoft Store if you don't have it.
- All paths use backslashes `\` in Windows Terminal.
  Python `Path` objects handle both `\` and `/` transparently.
- Run every command from inside the `phase3\` folder unless told otherwise.
- Keep the virtual environment activated in every terminal window you open.

---

### Step 1 — Open Windows Terminal and navigate to the project

```powershell
# Open Windows Terminal (Win + X → Windows Terminal)
# Navigate to your project folder
cd C:\Users\YourName\alpha_project\phase3

# Confirm you are in the right place
dir
# You should see: config.py  dataset.py  models.py  trainer.py  ...
```

---

### Step 2 — Activate the virtual environment

The virtual environment was created in Phase 1. Activate it in every terminal window.

```powershell
# If your project is at C:\Users\YourName\alpha_project\
..\venv\Scripts\activate

# Your prompt should change to show (venv):
# (venv) PS C:\Users\YourName\alpha_project\phase3>
```

If activation fails with a script execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then re-run the activate command above
```

---

### Step 3 — Install Phase 3 dependencies

```powershell
pip install -r requirements_phase3.txt
```

scikit-learn is the only new addition — it installs in under a minute.

---

### Step 4 — Verify models and training loop work

Run the fast verification (skips the slow 5-epoch training test):

```powershell
python verify_phase3.py --skip-training
```

Expected output:
```
── Phase 3 verification ──────────────────────────────────────

[Imports]
  [OK]  torch + nn
  [OK]  config imports
  [OK]  models importable
  [OK]  trainer importable
  [OK]  dataset importable
  [OK]  signal_generator importable

[Model architecture]
  [OK]  LSTM forward pass — correct output shape
  [OK]  Transformer forward pass — correct output shape
  [OK]  LSTM parameter count reasonable (50K–500K)
  [OK]  Transformer parameter count reasonable (10K–200K)
  [OK]  Positional encoding shape correct

[Loss function]
  [OK]  MixedLoss forward pass
  [OK]  MixedLoss is lower for closer predictions

[Dataset — synthetic data]
  [OK]  StockSequenceDataset builds from synthetic data
  [OK]  chronological_split respects date ordering
  [OK]  DataLoader iterates without error

[Training loop — 5 epochs on synthetic data]
  [SKIP] Training test (--skip-training set)

[Signal logic]
  [OK]  z_score above threshold → LONG
  [OK]  z_score below negative threshold → SHORT
  [OK]  z_score inside band → FLAT
  [OK]  position_size capped at SIGNAL_MAX_POSITION

[Device]
  [OK]  get_device returns a valid torch.device
         (using: cpu)

── Summary ────────────────────────────────────────────────────
  19 passed  /  0 failed  /  19 total

All Phase 3 checks passed.
```

To run with the full training test (adds ~3–5 minutes):
```powershell
python verify_phase3.py
```

---

### Step 5 — Check you have enough training data

Before training, confirm the feature matrix has enough rows:

```powershell
python -c "
import duckdb
con = duckdb.connect(r'..\phase1\db\alpha.duckdb', read_only=True)
total = con.execute('SELECT COUNT(*) FROM feature_matrix WHERE target_ret_1d IS NOT NULL').fetchone()[0]
dates = con.execute('SELECT MIN(date), MAX(date) FROM feature_matrix WHERE target_ret_1d IS NOT NULL').fetchone()
tickers = con.execute('SELECT COUNT(DISTINCT ticker) FROM feature_matrix').fetchone()[0]
con.close()
print(f'Labelled rows : {total}')
print(f'Date range    : {dates[0]}  to  {dates[1]}')
print(f'Unique tickers: {tickers}')
"
```

You need at minimum **~3,000 labelled rows** (= roughly 60 tickers × 50+ days).
For good results aim for **10,000+ rows** (= 50 tickers × 200+ trading days ≈ 10 months).

If you have fewer rows, run the Phase 2 backfill for a longer period:
```powershell
cd ..\phase2
python technical_features.py --backfill --days 500 --force
python sentiment_engine.py   --backfill --days 365
python feature_assembler.py  --backfill --days 365
python feature_assembler.py  --fill-targets
cd ..\phase3
```

---

### Step 6 — Train the LSTM

```powershell
python train_lstm.py
```

This will print epoch-by-epoch progress. Example output:

```
============================================================
Phase 3 — LSTM Training
============================================================
[dataset] Rows — train: 8234  val: 1763  test: 1763
[dataset] Samples — train: 6834  val: 1463  test: 1463

LSTM parameters: 197,505

[trainer] Training LSTM on cpu
          epochs=100  lr=0.001  patience=15
          train batches=107  val batches=23
------------------------------------------------------------
  Epoch    1/100 | train=0.000312 | val=0.000298 | lr=1.00e-03 | 12s
  Epoch    5/100 | train=0.000287 | val=0.000271 | lr=9.85e-04 | 58s
  Epoch   10/100 | train=0.000265 | val=0.000259 | lr=9.55e-04 | 114s
  ...
  Epoch   42/100 | train=0.000241 | val=0.000253 | lr=7.20e-04 | 482s

[trainer] Early stopping at epoch 57 (no improvement for 15 epochs)
[trainer] Training complete — best val_loss=0.000249 at epoch 42
[trainer] Best checkpoint: checkpoints/lstm_best.pt

[train_lstm] Evaluating on test set ...

Test set metrics:
  MAE       : 0.006821  (mean abs error in log-return units)
  RMSE      : 0.009144
  Hit rate  : 0.5480  (54.8% correct direction)
  IC        : 0.0823  (Pearson corr pred vs actual)

[train_lstm] Results saved → checkpoints\lstm_results.json
[train_lstm] Test predictions saved → checkpoints\lstm_test_predictions.csv
```

**Typical training time:**
- CPU only: 20–60 minutes depending on dataset size
- GPU (CUDA): 3–8 minutes

**What good metrics look like:**
- Hit rate > 52% is better than random; > 55% is strong
- IC > 0.05 means meaningful predictive correlation
- If hit rate is stuck at ~50%, the model needs more training data

To speed up training with fewer epochs (for quick testing):
```powershell
python train_lstm.py --epochs 20 --patience 5
```

---

### Step 7 — Train the Transformer

Open a **second Windows Terminal tab** (Ctrl+Shift+T) while the LSTM trains,
or wait until LSTM finishes.

```powershell
# Activate venv in the new tab
..\venv\Scripts\activate
cd ..\phase3

python train_transformer.py
```

Same output format as LSTM. Transformer typically trains faster but uses the
same stopping criterion. Expected metrics are similar — the ensemble of both
models outperforms either alone.

---

### Step 8 — Evaluate the ensemble

Once **both** models have finished training:

```powershell
python ensemble.py
```

Example output:
```
[ensemble] Evaluating ensemble on test set ...

Model comparison on test set:
  [LSTM]        MAE=0.006821  RMSE=0.009144  HitRate=54.8%  IC=0.0823
  [Transformer] MAE=0.006692  RMSE=0.008971  HitRate=55.1%  IC=0.0891
  [Ensemble]    MAE=0.006510  RMSE=0.008743  HitRate=55.9%  IC=0.0934

Ensemble predictions saved → checkpoints\ensemble_test_predictions.csv
```

The ensemble should score better than either model alone.
If it doesn't, tune the blend weights:

```powershell
python ensemble.py --tune-weights
```

This prints the optimal weights to put in `config.py`:
```
[ensemble] Best weights → LSTM=0.3  TF=0.7  val RMSE=0.008601

Update config.py:
  ENSEMBLE_W_LSTM = 0.3
  ENSEMBLE_W_TF   = 0.7
```

---

### Step 9 — Run the ablation study

This shows the incremental value of each feature group:

```powershell
python ensemble.py --ablation
```

Example output:
```
── Ablation summary ────────────────────────────────────────
  Variant               HitRate       IC       RMSE
  tech_only              51.2%    0.0312   0.009841
  text_only              52.8%    0.0541   0.009623
  text_plus_tech         55.1%    0.0867   0.009012
  full                   55.9%    0.0934   0.008743
```

This proves the value of combining sentiment + technical features.
Save this table for your project report.

---

### Step 10 — Generate today's signals

```powershell
python signal_generator.py
```

Example output:
```
[signal] Generating signals for 2024-03-28 ...
[signal] Written 50 signal rows to DB.

[signal] Active signals: 12
  LONG : 5
  SHORT: 5
  FLAT : 40

Top signals:
     ticker signal  pred_ret  z_score  position_size  pred_close
   RELIANCE   LONG   0.01423   1.2341         0.0617     2934.12
        TCS   LONG   0.01201   1.0982         0.0549     3812.44
 BHARTIARTL  SHORT  -0.01876  -1.4210         0.0710     1118.32
  HDFCBANK    LONG   0.00934   0.8821         0.0441     1703.81
      ...
```

To generate signals for a specific past date (for backtesting):
```powershell
python signal_generator.py --date 2024-03-15
```

---

### Step 11 — Start the continuous Phase 3 scheduler

Open a **third Windows Terminal tab**. You should now have:
- Tab 1: Phase 1 scheduler running (`scheduler.py`)
- Tab 2: Phase 2 scheduler running (`scheduler_phase2.py`)
- Tab 3: Phase 3 scheduler (start now)

```powershell
..\venv\Scripts\activate
cd ..\phase3
python scheduler_phase3.py
```

This generates fresh signals at 08:30, every 30 minutes intraday, and 16:50 IST.

---

## Running all three schedulers persistently on Windows

For long-term operation, use Windows Task Scheduler to auto-start all three
schedulers on boot.

### Option A — Three `.bat` launcher files

Create `start_phase1.bat`:
```bat
@echo off
cd C:\Users\YourName\alpha_project\phase1
call ..\venv\Scripts\activate
python scheduler.py
pause
```

Create `start_phase2.bat`:
```bat
@echo off
cd C:\Users\YourName\alpha_project\phase2
call ..\venv\Scripts\activate
python scheduler_phase2.py
pause
```

Create `start_phase3.bat`:
```bat
@echo off
cd C:\Users\YourName\alpha_project\phase3
call ..\venv\Scripts\activate
python scheduler_phase3.py
pause
```

### Option B — Windows Task Scheduler (recommended)

1. Press `Win + S` → search "Task Scheduler" → Open it
2. Click "Create Basic Task" (right panel)
3. Name: `Alpha Phase 1`  →  Trigger: "When the computer starts"
4. Action: "Start a program"
   - Program: `C:\Users\YourName\alpha_project\venv\Scripts\python.exe`
   - Arguments: `scheduler.py`
   - Start in: `C:\Users\YourName\alpha_project\phase1`
5. Repeat steps 2–4 for Phase 2 and Phase 3

With Task Scheduler, all three pipelines restart automatically after a reboot.

---

## Querying signals from DuckDB

```powershell
python -c "
import duckdb
con = duckdb.connect(r'..\phase1\db\alpha.duckdb', read_only=True)

# Today's signals
df = con.execute('''
    SELECT ticker, signal, pred_ret, z_score, position_size, pred_close
    FROM   predictions
    WHERE  date = current_date
    ORDER  BY abs(z_score) DESC
''').df()
print(df.to_string(index=False))
con.close()
"
```

---

## Monitoring training with loss CSV

The training scripts write a CSV log at `logs\lstm_loss.csv` and `logs\transformer_loss.csv`.
Open them in Excel to plot training curves.

Columns: `epoch, train_loss, val_loss, lr, elapsed_s`

Signs of a healthy run:
- Both train_loss and val_loss decrease for at least the first 20 epochs
- val_loss stays within 2× of train_loss (gap = overfitting)
- Early stopping fires between epoch 30 and epoch 80

If val_loss explodes (goes to NaN or > 1.0):
- Reduce learning rate: `python train_lstm.py --lr 0.0001`
- Reduce batch size:    `python train_lstm.py --batch-size 32`

---

## Troubleshooting (Windows-specific)

**`ModuleNotFoundError: No module named 'config'`**
You are not in the `phase3\` directory. Run:
```powershell
cd C:\Users\YourName\alpha_project\phase3
```

**`FileNotFoundError: Database not found`**
The DB path in `config.py` is relative. Open `config.py` and check:
```python
DB_PATH = BASE_DIR.parent / "phase1" / "db" / "alpha.duckdb"
```
If your folder structure is different, pass `--db` explicitly:
```powershell
python train_lstm.py --db C:\full\path\to\alpha.duckdb
```

**`RuntimeError: feature_matrix is empty`**
Phase 2 backfills have not been run. Go to `phase2\`:
```powershell
cd ..\phase2
python technical_features.py --backfill --days 365
python sentiment_engine.py   --backfill --days 90
python feature_assembler.py  --backfill --days 90
python feature_assembler.py  --fill-targets
cd ..\phase3
```

**`RuntimeError: Trained models not found`**
You are running `signal_generator.py` or `ensemble.py` before training.
Run Steps 6 and 7 first.

**`AssertionError: d_model (64) must be divisible by nhead (4)`**
This should never happen with default config. If you changed `TF_D_MODEL`
or `TF_NHEAD` in `config.py`, ensure `TF_D_MODEL % TF_NHEAD == 0`.

**Training very slow on CPU**
Expected on CPU — LSTM takes 20–60 min, Transformer 15–40 min.
To check if you have a usable GPU:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```
If you have an NVIDIA GPU but `cuda.is_available()` returns `False`:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**`duckdb.IOException: database is locked`**
Another Python process has the DB open in write mode.
Close all other Python processes and retry.
The schedulers open DuckDB only briefly (write then close), so this is rare.

---

## Configuration quick reference

Edit `config.py` to tune the model. Common adjustments:

| Parameter | Default | What it does |
|---|---|---|
| `SEQ_LEN` | 60 | Days of history fed to model. More = more context, slower training |
| `LSTM_HIDDEN` | 128 | LSTM hidden size. Double for richer capacity |
| `LSTM_EPOCHS` | 100 | Max epochs (early stopping fires before this usually) |
| `LSTM_PATIENCE` | 15 | Epochs without val improvement before stopping |
| `TF_D_MODEL` | 64 | Transformer embedding size (must be divisible by TF_NHEAD) |
| `SIGNAL_Z_THRESHOLD` | 0.5 | z-score cutoff for signal. Higher = fewer but stronger signals |
| `SIGNAL_TOP_N` | 5 | Max LONG and SHORT positions per day |
| `SIGNAL_MAX_POSITION` | 0.10 | Max portfolio weight per stock (10%) |
| `ENSEMBLE_W_LSTM` | 0.40 | LSTM blend weight (update after running `--tune-weights`) |
| `ENSEMBLE_W_TF` | 0.60 | Transformer blend weight |
