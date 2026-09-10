# Project Description: Sentiment-Driven Return Prediction on Nifty50 Stocks
## Overview
This project presents an end-to-end, machine learning-driven algorithmic trading pipeline designed to predict the next-day returns of Nifty50 stocks. By combining natural language processing (NLP) of financial news headlines with quantitative technical indicators, the system forecasts equity price movements. 
Unlike many academic implementations that rely on static, pre-packaged datasets, this project features a fully automated, production-grade ingestion and processing pipeline that operates on live data.

## System Architecture
The project is structured into a modular, three-phase pipeline:
* **Phase 1: Automated Data Ingestion & Processing**
  A live scheduling system that fetches financial news from 9 major Indian RSS feeds (e.g., ET Markets, MoneyControl, Mint) every 30 minutes. It handles real-time deduplication using MinHash LSH, applies NSE market calendar rules to assign trading dates, and tags articles to specific Nifty50 tickers using FlashText. It also automatically tops up daily OHLCV price data from Yahoo Finance.
* **Phase 2: Feature Engineering & Sentiment Analysis**
  Applies the FinBERT language model to extract financial sentiment (positive/negative/neutral probabilities) from the ingested headlines. It combines these sentiment scores with quantitative momentum and volatility metrics to generate a dense, 17-dimensional feature vector per stock, per day.
* **Phase 3: Predictive Modeling (Deep Learning)**
  An ensemble modeling approach utilizing Long Short-Term Memory (LSTM) networks and Transformer encoder architectures. The models are trained using a custom directional loss function optimized specifically to predict next-day Close-to-Open returns, rather than minimizing generic regression errors.
* **Phase 4: Target Refinement:**
 Shifting the target variable strictly to "Close-to-Open" overnight returns to capture the immediate impact of after-hours news.
* **Phase 5: Dataset Expansion:**
 Scaling the pipeline from the Nifty50 to the Nifty500 to increase the training dataset size tenfold, allowing the complex neural networks to generalize better across different market sectors and volatility regimes.

# Phase 1 — RSS Ingestion Pipeline
## Setup and Run Guide

---

## What this phase does

Every 30 minutes, the scheduler:
1. Fetches 9 RSS feeds (ET Markets, MoneyControl, Business Standard, Mint, Financial Express) in parallel
2. Deduplicates near-identical headlines using MinHash LSH (Jaccard ≥ 0.75 → drop duplicate)
3. Assigns each article a `trading_date` using NSE market calendar rules
4. Tags articles with Nifty50 ticker symbols using flashtext keyword matching
5. Opens a DuckDB connection → bulk-inserts the entire batch → closes immediately

The OHLCV downloader runs separately at 07:00 and 16:15 IST to top-up price data.

---

## File structure

```
PHASE_1/
├── nifty50_tickers.py      # Nifty50 symbols + company alias dictionary
├── rss_feeds.py            # Feed URLs + configuration constants
├── db_init.py              # Create DuckDB schema (run once)
├── rss_fetcher.py          # Fetch + deduplicate RSS articles
├── tagger.py               # Trading date assignment + ticker tagging
├── db_writer.py            # All DuckDB writes (open → insert → close)
├── ohlcv_downloader.py     # yfinance OHLCV download
├── scheduler.py            # APScheduler — orchestrates everything
├── verify.py               # End-to-end sanity checks
├── requirements.txt
├── db/                     # Created automatically
│   └── alpha.duckdb        # The database file
└── logs/
    └── scheduler.log       # Rolling log file
```

---

## Step-by-step setup

### Step 1 — Python environment

Requires Python 3.10 or 3.11.

```bash
# Create a virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows PowerShell
```

### Step 2 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: feedparser, datasketch, flashtext, pandas-market-calendars,
pandas, numpy, yfinance, duckdb, APScheduler.

Expected install time: 2–4 minutes depending on internet speed.

### Step 3 — Create the database schema

Run this once. It creates `db/alpha.duckdb` with four tables:
`raw_news`, `tagged_articles`, `ohlcv`, `pipeline_runs`.

```bash
python PHASE_1/db_init.py
```

Expected output:
```
[db_init] Database ready at db/alpha.duckdb
[db_init] Tables: raw_news, tagged_articles, ohlcv, pipeline_runs
```

### Step 4 — Download historical OHLCV (2 years)

This downloads ~500 trading days × 50 stocks from yfinance.
Run once; takes about 60–90 seconds.

```bash
python PHASE_1/ohlcv_downloader.py --mode full
```

Expected output:
```
[ohlcv] Downloading 2y of data for 50 tickers ...
[ohlcv] Assembled 24850 rows for 50 tickers
[ohlcv] ohlcv: upserted 24850 rows
[ohlcv] Full download complete — 24850 rows upserted.
```

If yfinance hits a rate limit, wait 60 seconds and rerun — the upsert is idempotent.

### Step 5 — Run the verification suite

This checks every component end-to-end using a temporary database.
It makes live network calls to ET Markets and yfinance.

```bash
python PHASE_1/verify.py
```

All checks should print `[OK]`. If any print `[FAIL]`, fix that issue
before proceeding (see Troubleshooting below).

Expected output (truncated):
```
── Phase 1 verification ──────────────────────────────────────

[Imports]
  [OK]  feedparser
  [OK]  datasketch
  [OK]  flashtext
  [OK]  pandas_market_calendars
  [OK]  duckdb
  [OK]  yfinance
  [OK]  APScheduler
  [OK]  pandas / numpy

[Nifty50 data]
  [OK]  NIFTY50_TICKERS has 50 entries
  [OK]  TICKER_ALIASES non-empty

[Database]
  [OK]  db_init creates schema
  [OK]  all four tables exist

[RSS fetcher — live network call]
  [OK]  fetch ET Markets RSS (≥1 article)
  [OK]  fetch_and_deduplicate returns DataFrame

[Tagger]
  [OK]  flashtext tags RELIANCE correctly
  [OK]  assign_trading_dates handles after-hours correctly

[DB writer round-trip]
  [OK]  insert_raw_news → count increases
  [OK]  insert_tagged_articles works
  [OK]  pipeline_runs audit log

[OHLCV — live yfinance call, 2 tickers]
  [OK]  yfinance returns data for RELIANCE.NS

── Summary ────────────────────────────────────────────────────
  16 passed  /  0 failed  /  16 total

All checks passed. Phase 1 is ready to run.
```

### Step 6 — Test a single batch (optional but recommended)

Before starting the continuous scheduler, run one RSS batch manually
to confirm articles flow all the way into DuckDB:

```bash
python PHASE_1/scheduler.py --run-once
```

Then inspect the database:

```bash
python - <<'EOF'
import duckdb
con = duckdb.connect("db/alpha.duckdb")
print("raw_news:       ", con.execute("SELECT COUNT(*) FROM raw_news").fetchone()[0])
print("tagged_articles:", con.execute("SELECT COUNT(*) FROM tagged_articles").fetchone()[0])
print("Top 5 tagged tickers:")
print(con.execute("""
    SELECT ticker, COUNT(*) as n
    FROM tagged_articles
    GROUP BY ticker
    ORDER BY n DESC
    LIMIT 5
""").df())
con.close()
EOF
```

Expected output (numbers will vary by news volume):
```
raw_news:        347
tagged_articles: 89
Top 5 tagged tickers:
        ticker   n
0    RELIANCE  14
1         TCS  11
2    HDFCBANK   9
3        INFY   8
4  BHARTIARTL   7
```

### Step 7 — Start the continuous scheduler

```bash
python PHASE_1/scheduler.py
```

The scheduler will:
- Run an immediate RSS batch on startup
- Then repeat every 30 minutes
- Run OHLCV top-up at 07:00 and 16:15 IST daily
- Write all activity to `logs/scheduler.log`

To stop: press `Ctrl+C` — it shuts down gracefully.

---

## Running in the background (production)

### Option A — nohup (simple, Linux/macOS)

```bash
nohup python PHASE_1/scheduler.py > logs/nohup.out 2>&1 &
echo $! > logs/scheduler.pid
```

To stop:
```bash
kill $(cat logs/scheduler.pid)
```

### Option B — systemd service (recommended for servers)

Create `/etc/systemd/system/phase1.service`:

```ini
[Unit]
Description=Phase 1 RSS Ingestion Pipeline
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/your_project
ExecStart=/home/ubuntu/your_project/venv/bin/python PHASE_1/scheduler.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable phase1
sudo systemctl start phase1
sudo systemctl status phase1
```

Logs via: `journalctl -u phase1 -f`

### Option C — Windows Task Scheduler

Create a `.bat` file:
```bat
@echo off
cd C:\projects\your_project
call venv\Scripts\activate
python PHASE_1\scheduler.py
```

Schedule it to run at system startup using Task Scheduler.

---

## Querying the database

DuckDB can be queried from Python, the CLI, or any SQL tool.

### From Python

```python
import duckdb
con = duckdb.connect("db/alpha.duckdb", read_only=True)

# Articles about RELIANCE in the last 7 days
df = con.execute("""
    SELECT n.title, n.source, n.trading_date, t.ticker
    FROM   raw_news n
    JOIN   tagged_articles t USING (article_id)
    WHERE  t.ticker = 'RELIANCE'
    AND    n.trading_date >= current_date - INTERVAL 7 DAY
    ORDER  BY n.trading_date DESC
""").df()
print(df)
con.close()
```

### From DuckDB CLI

```bash
# Install DuckDB CLI (one-time)
pip install duckdb

# Open interactive shell
duckdb db/alpha.duckdb

# Example queries:
D SELECT COUNT(*) FROM raw_news;
D SELECT ticker, COUNT(*) FROM tagged_articles GROUP BY ticker ORDER BY 2 DESC LIMIT 10;
D SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 5;
D SELECT * FROM ohlcv WHERE ticker = 'RELIANCE' ORDER BY date DESC LIMIT 5;
```

---

## Monitoring

Check today's pipeline health:

```python
import duckdb
con = duckdb.connect("db/alpha.duckdb", read_only=True)

# Recent run log
print(con.execute("""
    SELECT run_id[:8] as id, run_type, started_at::VARCHAR[:19] as started,
           articles_fetched, articles_new, articles_tagged, status
    FROM   pipeline_runs
    ORDER  BY started_at DESC
    LIMIT  10
""").df().to_string(index=False))

# Today's articles by ticker
print(con.execute("""
    SELECT ticker, COUNT(*) as articles_today
    FROM   tagged_articles
    WHERE  tagged_at::DATE = current_date
    GROUP  BY ticker
    ORDER  BY 2 DESC
""").df().to_string(index=False))
con.close()
```

---

## Troubleshooting

**`ModuleNotFoundError`**
Make sure the virtual environment is activated: `source venv/bin/activate`

**`feedparser` returns empty entries**
Some feeds block bots. Try opening the URL in a browser to confirm it's live.
The pipeline continues with whichever feeds succeed — partial data is fine.

**`pandas_market_calendars` NSE calendar errors**
Upgrade: `pip install --upgrade pandas-market-calendars exchange-calendars`

**yfinance rate limit (`YFRateLimitError`)**
Wait 60–120 seconds and retry. Or add `time.sleep(1)` between ticker downloads
in `ohlcv_downloader.py` for very slow connections.

**DuckDB `database is locked` error**
This should not happen with the open→write→close pattern. If it does,
check that no other process has the DB open in read-write mode
(e.g. a DuckDB CLI session). Read-only connections (`read_only=True`) never block writes.

**Articles not tagged with expected tickers**
Check `TICKER_ALIASES` in `nifty50_tickers.py`. Add the missing alias
and restart the scheduler. Re-tagging old articles:
```python
# Re-run tagger on all existing raw_news rows
import duckdb, pandas as pd
from PHASE_1.tagger import tag_articles
from PHASE_1.db_writer import insert_tagged_articles, DEFAULT_DB
con = duckdb.connect(str(DEFAULT_DB))
df = con.execute("SELECT * FROM raw_news").df()
con.close()
tags = tag_articles(df)
insert_tagged_articles(tags)
```

---

## Configuration reference

| File | Variable | Default | Effect |
|---|---|---|---|
| `rss_feeds.py` | `BATCH_INTERVAL_MIN` | `30` | RSS fetch frequency (minutes) |
| `rss_feeds.py` | `FETCH_TIMEOUT_SEC` | `12` | Per-feed HTTP timeout |
| `rss_feeds.py` | `FETCH_WORKERS` | `6` | Parallel feed threads |
| `rss_fetcher.py` | `threshold` in `deduplicate()` | `0.75` | Jaccard sim cutoff |
| `db_init.py` | `DEFAULT_DB` | `db/alpha.duckdb` | Database file location |
