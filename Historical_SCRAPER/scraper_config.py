# scraper_config.py  (v2 — updated)
# Central configuration for the 2-year historical news scraper.
# v2 changes: Google News added, MoneyControl/BS/FE added,
#             ET pagination fixed, existing data preserved.

from datetime import date, timedelta
from pathlib import Path

BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "raw_data"
PARQUET_DIR  = BASE_DIR / "parquet"
LOGS_DIR     = BASE_DIR / "logs"
CHECKPOINT   = BASE_DIR / "checkpoint.json"
DB_PATH      = BASE_DIR.parent  / "db" / "alpha.duckdb"

for _d in [DATA_DIR, PARQUET_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

SCRAPE_END_DATE   = date.today() - timedelta(days=1)
SCRAPE_START_DATE = SCRAPE_END_DATE - timedelta(days=730)

REQUEST_TIMEOUT = 20
RETRY_MAX       = 3
RETRY_BACKOFF   = 2.0

DOMAIN_DELAYS = {
    "economictimes.indiatimes.com": 3.0,
    "livemint.com":                 2.5,
    "moneycontrol.com":             3.0,
    "business-standard.com":        3.0,
    "financialexpress.com":         3.0,
    "news.google.com":              1.5,
    "web.archive.org":              1.5,
    "default":                      2.5,
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

# ET pagination fix: try up to 10 archive pages per day (~400 articles max)
ET_MAX_ARCHIVE_PAGES = 10

# Google News RSS settings
GOOGLE_NEWS_BASE  = "https://news.google.com/rss/search"
GOOGLE_NEWS_QUERIES = [
    "Nifty 50 stock market India",
    "BSE NSE India equity",
    "Indian stock market Sensex Nifty",
    "Indian banking stocks HDFC ICICI SBI",
    "Indian IT stocks TCS Infosys Wipro",
    "Indian pharma stocks Sun Pharma Cipla",
    "India corporate earnings quarterly results",
    "RBI monetary policy repo rate India",
    "NSE BSE bulk block deals FII DII",
    "Indian auto stocks Maruti Tata Motors",
    "Indian FMCG stocks HUL ITC Nestle",
]

SOURCE_REGISTRY = {
    "et_stocks":    {"method": "et_archive",  "section": "et_stocks",  "priority": 1},
    "et_markets":   {"method": "et_archive",  "section": "et_markets", "priority": 1},
    "et_topnews":   {"method": "et_archive",  "section": "et_topnews", "priority": 1},
    "google_news":  {"method": "google_news", "priority": 1},
    "moneycontrol": {"method": "mc_archive",  "priority": 1},
    "business_std": {"method": "bs_archive",  "priority": 1},
    "fin_express":  {"method": "fe_archive",  "priority": 1},
    "wayback_et":   {"method": "wayback",     "priority": 2},
}

DEFAULT_SOURCES = [
    "et_stocks", "et_markets", "et_topnews",
    "google_news",
    "moneycontrol",
    "business_std",
    "fin_express",
]

MIN_ARTICLES_PER_DAY = 15
DB_BATCH_SIZE        = 500

def et_archive_ts(d: date) -> str:
    """
    ET archive URLs use an ancient 'Excel Epoch' format.
    It is the number of days since December 30, 1899.
    """
    base_date = date(1899, 12, 30)
    delta = d - base_date
    return str(delta.days)
