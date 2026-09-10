# nifty50_tickers.py
# Single source of truth for all ticker/alias data used across Phase 1.
# yfinance uses the ".NS" suffix; internal keys use plain NSE symbols.

NIFTY50_TICKERS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN",
    "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TMCV",
    "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

# yfinance symbols (append .NS)
NIFTY50_YF = [f"{t}.NS" for t in NIFTY50_TICKERS]

# Alias map: lowercase keyword -> NSE symbol
# Used by flashtext for ticker tagging in news headlines.
# Multiple aliases map to the same symbol.
TICKER_ALIASES: dict[str, str] = {
    # ADANIENT
    "adani enterprises": "ADANIENT",
    "adani ent": "ADANIENT",
    "adani enterprise": "ADANIENT",

    # ADANIPORTS
    "adani ports": "ADANIPORTS",
    "apsez": "ADANIPORTS",
    "adani port": "ADANIPORTS",

    # APOLLOHOSP
    "apollo hospitals": "APOLLOHOSP",
    "apollo hospital": "APOLLOHOSP",
    "apollo health": "APOLLOHOSP",

    # ASIANPAINT
    "asian paints": "ASIANPAINT",
    "asian paint": "ASIANPAINT",

    # AXISBANK
    "axis bank": "AXISBANK",
    "axis financial": "AXISBANK",

    # BAJAJ-AUTO
    "bajaj auto": "BAJAJ-AUTO",
    "bajaj motorcycle": "BAJAJ-AUTO",
    "bajaj two wheeler": "BAJAJ-AUTO",

    # BAJFINANCE
    "bajaj finance": "BAJFINANCE",
    "bajaj fin": "BAJFINANCE",

    # BAJAJFINSV
    "bajaj finserv": "BAJAJFINSV",
    "bajaj financial services": "BAJAJFINSV",

    # BPCL
    "bpcl": "BPCL",
    "bharat petroleum": "BPCL",

    # BHARTIARTL
    "airtel": "BHARTIARTL",
    "bharti airtel": "BHARTIARTL",
    "bharti": "BHARTIARTL",

    # BRITANNIA
    "britannia": "BRITANNIA",
    "britannia industries": "BRITANNIA",

    # CIPLA
    "cipla": "CIPLA",

    # COALINDIA
    "coal india": "COALINDIA",
    "coal india ltd": "COALINDIA",
    "cil": "COALINDIA",

    # DIVISLAB
    "divi's laboratories": "DIVISLAB",
    "divis lab": "DIVISLAB",
    "divi laboratories": "DIVISLAB",

    # DRREDDY
    "dr reddy": "DRREDDY",
    "dr. reddy": "DRREDDY",
    "dr reddys": "DRREDDY",
    "drl": "DRREDDY",

    # EICHERMOT
    "eicher motors": "EICHERMOT",
    "royal enfield": "EICHERMOT",
    "eicher": "EICHERMOT",

    # GRASIM
    "grasim": "GRASIM",
    "grasim industries": "GRASIM",
    "aditya birla fashion": "GRASIM",

    # HCLTECH
    "hcl technologies": "HCLTECH",
    "hcl tech": "HCLTECH",
    "hcl": "HCLTECH",

    # HDFCBANK
    "hdfc bank": "HDFCBANK",
    "hdfc banking": "HDFCBANK",

    # HDFCLIFE
    "hdfc life": "HDFCLIFE",
    "hdfc standard life": "HDFCLIFE",

    # HEROMOTOCO
    "hero motocorp": "HEROMOTOCO",
    "hero moto": "HEROMOTOCO",
    "hero honda": "HEROMOTOCO",

    # HINDALCO
    "hindalco": "HINDALCO",
    "hindalco industries": "HINDALCO",
    "novelis": "HINDALCO",

    # HINDUNILVR
    "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "unilever india": "HINDUNILVR",

    # ICICIBANK
    "icici bank": "ICICIBANK",
    "icici": "ICICIBANK",

    # ITC
    "itc": "ITC",
    "itc limited": "ITC",
    "itc ltd": "ITC",

    # INDUSINDBK
    "indusind bank": "INDUSINDBK",
    "indusind": "INDUSINDBK",

    # INFY
    "infosys": "INFY",
    "infy": "INFY",

    # JSWSTEEL
    "jsw steel": "JSWSTEEL",
    "jsw": "JSWSTEEL",

    # KOTAKBANK
    "kotak mahindra bank": "KOTAKBANK",
    "kotak bank": "KOTAKBANK",
    "kotak": "KOTAKBANK",

    # LT
    "larsen": "LT",
    "larsen and toubro": "LT",
    "l&t": "LT",
    "larsen toubro": "LT",

    # LTIM
    "ltimindtree": "LTIM",
    "lti mindtree": "LTIM",
    "mindtree": "LTIM",

    # M&M
    "mahindra": "M&M",
    "mahindra and mahindra": "M&M",
    "m&m": "M&M",

    # MARUTI
    "maruti suzuki": "MARUTI",
    "maruti": "MARUTI",
    "msil": "MARUTI",

    # NESTLEIND
    "nestle india": "NESTLEIND",
    "nestle": "NESTLEIND",

    # NTPC
    "ntpc": "NTPC",
    "national thermal power": "NTPC",

    # ONGC
    "ongc": "ONGC",
    "oil and natural gas": "ONGC",
    "oil natural gas": "ONGC",

    # POWERGRID
    "power grid": "POWERGRID",
    "power grid corporation": "POWERGRID",
    "pgcil": "POWERGRID",

    # RELIANCE
    "reliance": "RELIANCE",
    "ril": "RELIANCE",
    "reliance industries": "RELIANCE",
    "jio": "RELIANCE",
    "mukesh ambani": "RELIANCE",

    # SBILIFE
    "sbi life": "SBILIFE",
    "sbi life insurance": "SBILIFE",

    # SHRIRAMFIN
    "shriram finance": "SHRIRAMFIN",
    "shriram": "SHRIRAMFIN",

    # SBIN
    "sbi": "SBIN",
    "state bank": "SBIN",
    "state bank of india": "SBIN",

    # SUNPHARMA
    "sun pharma": "SUNPHARMA",
    "sun pharmaceutical": "SUNPHARMA",
    "sun pharmacy": "SUNPHARMA",

    # TCS
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "tata consultancy services": "TCS",

    # TATACONSUM
    "tata consumer": "TATACONSUM",
    "tata consumer products": "TATACONSUM",
    "tata tea": "TATACONSUM",

    # TATAMOTORS
    "tata motors": "TATAMOTORS",
    "jaguar land rover": "TATAMOTORS",
    "jlr": "TATAMOTORS",

    # TATASTEEL
    "tata steel": "TATASTEEL",
    "tata steel europe": "TATASTEEL",

    # TECHM
    "tech mahindra": "TECHM",
    "tech m": "TECHM",

    # TITAN
    "titan": "TITAN",
    "titan company": "TITAN",
    "tanishq": "TITAN",

    # ULTRACEMCO
    "ultratech cement": "ULTRACEMCO",
    "ultratech": "ULTRACEMCO",

    # WIPRO
    "wipro": "WIPRO",
    "wipro technologies": "WIPRO",
}
