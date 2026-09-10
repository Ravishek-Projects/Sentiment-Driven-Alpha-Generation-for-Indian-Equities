# rss_feeds.py
# All RSS/Atom feed URLs used for news ingestion.
# Add or remove feeds here without touching any other file.

RSS_FEEDS: dict[str, str] = {
    # Economic Times — Markets
    "et_markets":
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",

    # Economic Times — Stocks
    "et_stocks":
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",

    # Economic Times — Top news
    "et_topnews":
        "https://economictimes.indiatimes.com/rssfeedstopstories.cms",

    # MoneyControl — Top news
    "moneycontrol_top":
        "https://www.moneycontrol.com/rss/MCtopnews.xml",

    # MoneyControl — Business
    "moneycontrol_biz":
        "https://www.moneycontrol.com/rss/business.xml",

    # Business Standard — Markets
    "business_std_markets":
        "https://www.business-standard.com/rss/markets-106.rss",

    # Business Standard — Companies
    "business_std_companies":
        "https://www.business-standard.com/rss/companies-101.rss",

    # Mint — Markets
    "mint_markets":
        "https://www.livemint.com/rss/markets",

    # Financial Express — Market
    "fin_express":
        "https://www.financialexpress.com/market/feed/",
}

# Fetch timeout in seconds per feed
FETCH_TIMEOUT_SEC: int = 12

# How many worker threads for parallel feed fetching
FETCH_WORKERS: int = 6

# Batch insert interval (minutes). We collect articles for this long,
# then bulk-insert the whole batch into DuckDB in one shot.
BATCH_INTERVAL_MIN: int = 30
