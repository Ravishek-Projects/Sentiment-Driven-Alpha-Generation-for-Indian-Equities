# Historical News Scraper — v2
## Complete Windows Terminal Guide

## What changed in v2

| Problem | v1 | v2 fix |
|---|---|---|
| ET inconsistent counts (79 vs 500+) | Single archive page per day | Paginate up to 10 pages → 200–400 articles consistently |
| Mint/MC/BS/FE returning 0 | Direct HTML hit firewalls | Google News RSS (no firewall, free) + improved direct scrapers |
| No Google News | Not present | Added: covers ALL sources, 11 rotating queries, no API key |
| Previously scraped ET data | Risk of loss | INSERT OR IGNORE everywhere — zero data loss |

## Files

| File | Status | Role |
|---|---|---|
| `et_scraper.py` | UPDATED | ET pagination fix: 200-400 articles/day |
| `google_news_scraper.py` | NEW | Google News RSS with date operators |
| `multi_source_scraper.py` | NEW | MoneyControl, Business Standard, Financial Express |
| `run_scraper.py` | UPDATED | Orchestrates all sources |
| `scraper_config.py` | UPDATED | Google News queries, all source registry |
| `requirements_scraper.txt` | UPDATED | Added feedparser |
| All other files | UNCHANGED | http_client, wayback, db_writer, retrain_pipeline |

## Windows Terminal — Exact Commands

```powershell
cd C:\Users\YourName\alpha_project\historical_scraper
..\venv\Scripts\activate

# Step 1: install deps (15 sec)
pip install -r requirements_scraper.txt

# Step 2: verify existing ET data is intact
python -c "import duckdb; con=duckdb.connect(r'..\phase1\db\alpha.duckdb',read_only=True); print(con.execute('SELECT source, COUNT(*) FROM raw_news GROUP BY source ORDER BY 2 DESC').df())"

# Step 3: re-scrape ET with pagination fix (fills the low-count days)
python run_scraper.py --sources et_stocks et_markets et_topnews

# Step 4: add Google News (covers all sources via RSS, 2-4 hours)
python run_scraper.py --sources google_news

# Step 5: add MoneyControl, BS, FE
python run_scraper.py --sources moneycontrol business_std fin_express

# Step 6: OR run everything at once
python run_scraper.py

# Step 7: verify coverage
python -c "
import duckdb
con = duckdb.connect(r'..\phase1\db\alpha.duckdb', read_only=True)
print(con.execute('SELECT source, COUNT(*) as n, MIN(trading_date) earliest, MAX(trading_date) latest FROM raw_news GROUP BY source ORDER BY n DESC').df().to_string())
con.close()
"

# Step 8: retrain models (Phase 2 → Phase 3)
python retrain_pipeline.py
```

## Key design decisions

### Why Google News RSS works when direct scraping doesn't
- Google News RSS endpoint (`news.google.com/rss/search`) is freely accessible
- Supports `after:YYYY-MM-DD before:YYYY-MM-DD` date operators
- No authentication, no API key, works through corporate firewalls
- Aggregates from ET, MoneyControl, Business Standard, Mint, FE, Reuters India, etc.
- 11 rotating queries × 104 weeks = ~1,144 RSS fetches for 2-year coverage

### Why ET counts were inconsistent
ET's archive system splits articles across multiple paginated URLs:
```
/markets/stocks/news/archivelist/starttime-{ts}.cms          ← page 1 (~40 articles)
/markets/stocks/news/archivelist/page-2/starttime-{ts}.cms   ← page 2 (~40 articles)
...
/markets/stocks/news/archivelist/page-10/starttime-{ts}.cms  ← page 10 (empty → stop)
```
v1 only fetched page 1. v2 paginates until a page returns 0 new links.

### Zero data loss guarantee
`db_writer_historical.py` uses:
```sql
INSERT INTO raw_news
SELECT b.* FROM _batch_hist b
WHERE b.article_id NOT IN (SELECT article_id FROM raw_news)
```
Every article has a deterministic `article_id = SHA1(source + URL)`.
Re-scraping the same URL from the same source always generates the same ID,
so it's silently skipped.

## Troubleshooting

**ET still 79 articles some days**
That day genuinely has fewer articles (holidays, low news volume). Check:
```powershell
python -c "
from et_scraper import fetch_archive_links
from datetime import date
links = fetch_archive_links(date(2024,3,15), 'et_stocks')
print(len(links), 'links found')
"
```

**Google News returning 0**
Check the URL manually in browser:
```powershell
python -c "
from google_news_scraper import _gn_rss_url
from datetime import date
print(_gn_rss_url('Nifty 50 India', date(2024,1,8), date(2024,1,14)))
"
```

**Checkpoint confusion after adding new sources**
New sources (google_news, moneycontrol, etc.) won't be in the old checkpoint —
they'll run from scratch. Existing ET keys are preserved and skipped.

**Force re-scrape a specific source without touching DB data**
```powershell
# Edit checkpoint.json and delete the keys for the source you want to redo
# OR use --reset-checkpoint (re-scrapes all weeks but preserves DB rows)
python run_scraper.py --sources google_news --reset-checkpoint
```
