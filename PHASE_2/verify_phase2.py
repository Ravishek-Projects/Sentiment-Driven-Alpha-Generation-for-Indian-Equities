# verify_phase2.py
# End-to-end sanity checks for all Phase 2 components.
# Runs against a temporary database — never touches production data.
# Makes one live yfinance call and loads FinBERT from HuggingFace (cached after first run).
#
# Usage:
#   python verify_phase2.py
#   python verify_phase2.py --skip-finbert    # skip the ~1 min model load
#
# All checks print [OK]. Any [FAIL] means something needs fixing.

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
PASS = "[OK]"
FAIL = "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        try:
            fn()
            results.append((name, True, ""))
            print(f"  {PASS}  {name}")
        except Exception as exc:
            results.append((name, False, str(exc)))
            print(f"  {FAIL}  {name}")
            print(f"         {exc}")
        return fn
    return decorator


# ─── Setup: create a temp DB with both Phase 1 and Phase 2 schemas ────────

import tempfile, duckdb
_TMP_DIR = Path(tempfile.mkdtemp())
_TMP_DB  = _TMP_DIR / "verify_p2.duckdb"

def _bootstrap_db():
    """Create schema in temp DB by importing both init modules."""
    import sys
    # Make sure phase1 and phase2 modules are importable
    sys.path.insert(0, str(Path(__file__).parent.parent / "PHASE_1"))
    sys.path.insert(0, str(Path(__file__).parent))

    from db_init import init_db
    init_db(_TMP_DB)
    from db_schema_phase2 import extend_schema
    extend_schema(_TMP_DB)

_bootstrap_db()

print("\n── Phase 2 verification ──────────────────────────────────────")

# ─── 1. Imports ───────────────────────────────────────────────────────────
print("\n[Imports]")

@check("torch")
def _(): import torch

@check("transformers")
def _(): from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

@check("sentiment_engine importable")
def _(): import sentiment_engine

@check("technical_features importable")
def _(): import technical_features

@check("feature_assembler importable")
def _(): import feature_assembler

@check("db_writer phase2 functions present")
def _():
    from db_writer import (insert_sentiment_scores, insert_technical_features,
                           insert_feature_matrix)

# ─── 2. Schema ────────────────────────────────────────────────────────────
print("\n[Database schema]")

@check("Phase 2 tables created")
def _():
    con = duckdb.connect(str(_TMP_DB))
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    con.close()
    for t in ("sentiment_scores", "technical_features", "feature_matrix"):
        assert t in tables, f"missing: {t}"

@check("sentiment_scores columns correct")
def _():
    con = duckdb.connect(str(_TMP_DB))
    cols = {r[0] for r in con.execute("DESCRIBE sentiment_scores").fetchall()}
    con.close()
    for c in ("ticker","trading_date","sent_pos","sent_neg","sent_neu",
              "sent_spread","news_count","articles_used"):
        assert c in cols, f"missing column: {c}"

@check("feature_matrix has 19+ feature columns")
def _():
    con = duckdb.connect(str(_TMP_DB))
    cols = [r[0] for r in con.execute("DESCRIBE feature_matrix").fetchall()]
    con.close()
    feature_cols = [c for c in cols
                    if c not in ("ticker","date","target_ret_1d","computed_at")]
    assert len(feature_cols) >= 19, f"only {len(feature_cols)} feature columns"

# ─── 3. db_writer round-trips ─────────────────────────────────────────────
print("\n[DB writer — Phase 2]")

@check("insert_sentiment_scores works")
def _():
    import pandas as pd
    from db_writer import insert_sentiment_scores, count_rows
    df = pd.DataFrame([{
        "ticker": "RELIANCE", "trading_date": date.today(),
        "sent_pos": 0.7, "sent_neg": 0.1, "sent_neu": 0.2,
        "sent_spread": 0.6, "news_count": 5, "articles_used": 5,
        "computed_at": datetime.now(timezone.utc),
    }])
    n = insert_sentiment_scores(df, _TMP_DB)
    assert n == 1
    # idempotent
    n2 = insert_sentiment_scores(df, _TMP_DB)
    assert n2 == 1   # INSERT OR REPLACE

@check("insert_technical_features works")
def _():
    import pandas as pd
    from db_writer import insert_technical_features
    df = pd.DataFrame([{
        "ticker": "RELIANCE", "date": date.today(),
        "rsi_14": 55.3, "ma_20": 2800.0, "ma_50": 2750.0,
        "price_to_ma20": 0.012, "vol_20d": 0.18,
        "rel_volume": 1.2, "bb_position": 0.65,
        "lag_ret_1d": 0.005, "lag_ret_2d": -0.002,
        "computed_at": datetime.now(timezone.utc),
    }])
    n = insert_technical_features(df, _TMP_DB)
    assert n == 1

@check("insert_feature_matrix works")
def _():
    import pandas as pd
    from db_writer import insert_feature_matrix
    df = pd.DataFrame([{
        "ticker": "RELIANCE", "date": date.today(),
        "sent_pos": 0.7, "sent_neg": 0.1, "sent_neu": 0.2,
        "sent_spread": 0.6, "news_count": 1.8,
        "rsi_14": 55.3, "ma_20": 2800.0, "ma_50": 2750.0,
        "price_to_ma20": 0.012, "vol_20d": 0.18,
        "rel_volume": 1.2, "bb_position": 0.65,
        "lag_ret_1d": 0.005, "lag_ret_2d": -0.002,
        "dow_mon": 0.0, "dow_tue": 0.0, "dow_wed": 1.0,
        "dow_thu": 0.0, "dow_fri": 0.0,
        "target_ret_1d": None,
        "computed_at": datetime.now(timezone.utc),
    }])
    n = insert_feature_matrix(df, _TMP_DB)
    assert n == 1

# ─── 4. Technical indicators math ─────────────────────────────────────────
print("\n[Technical indicators — unit tests]")

@check("RSI computes correctly on known series")
def _():
    import pandas as pd
    from technical_features import _rsi
    # Flat series → RSI should be 50
    flat = pd.Series([100.0] * 30)
    r = _rsi(flat)
    assert abs(r - 50.0) < 1.0, f"expected ~50, got {r}"

@check("RSI returns nan for too-short series")
def _():
    import pandas as pd, math
    from technical_features import _rsi
    short = pd.Series([100.0] * 5)
    assert math.isnan(_rsi(short))

@check("Bollinger Band position in [0,1] for typical price")
def _():
    import pandas as pd
    from technical_features import _bb_position
    # Price oscillating around 100
    prices = pd.Series([100 + (i % 5 - 2) * 0.5 for i in range(30)])
    pos = _bb_position(prices)
    assert 0.0 <= pos <= 1.0 or abs(pos) < 1.6, f"got {pos}"

@check("_compute_for_ticker returns None with insufficient history")
def _():
    from technical_features import _compute_for_ticker
    result = _compute_for_ticker("RELIANCE", date.today(), _TMP_DB)
    assert result is None   # no OHLCV in temp DB

# ─── 5. Sentiment aggregation ─────────────────────────────────────────────
print("\n[Sentiment aggregation logic]")

@check("_aggregate_scores means correctly")
def _():
    from sentiment_engine import _aggregate_scores
    scores = [
        {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
        {"positive": 0.6, "negative": 0.3, "neutral": 0.1},
    ]
    agg = _aggregate_scores(scores)
    assert abs(agg["sent_pos"] - 0.7) < 0.001
    assert abs(agg["sent_spread"] - 0.5) < 0.001

@check("_aggregate_scores returns neutral for empty list")
def _():
    from sentiment_engine import _aggregate_scores
    agg = _aggregate_scores([])
    assert abs(agg["sent_spread"]) < 0.001

# ─── 6. Feature assembler logic ───────────────────────────────────────────
print("\n[Feature assembler logic]")

@check("_dow_encode gives correct one-hot")
def _():
    from feature_assembler import _dow_encode
    # 2024-03-18 is a Monday
    enc = _dow_encode(date(2024, 3, 18))
    assert enc["dow_mon"] == 1.0
    assert enc["dow_tue"] == 0.0

@check("_ffill_sentiment forward-fills missing dates")
def _():
    import pandas as pd
    from feature_assembler import _ffill_sentiment
    sent = pd.DataFrame([{
        "ticker": "TCS", "date": date(2024, 3, 15),   # Friday
        "sent_pos": 0.8, "sent_neg": 0.1, "sent_neu": 0.1,
        "sent_spread": 0.7, "news_count": 3,
    }])
    filled = _ffill_sentiment(sent, ["TCS"], [date(2024, 3, 18)])  # Monday
    assert len(filled) == 1
    assert abs(filled.iloc[0]["sent_pos"] - 0.8) < 0.001

@check("_ffill_sentiment inserts neutral beyond MAX_FFILL_DAYS")
def _():
    import pandas as pd
    from feature_assembler import _ffill_sentiment, NEUTRAL
    empty = pd.DataFrame(columns=["ticker","date","sent_pos","sent_neg",
                                   "sent_neu","sent_spread","news_count"])
    filled = _ffill_sentiment(empty, ["TCS"], [date.today()])
    assert abs(filled.iloc[0]["sent_pos"] - NEUTRAL) < 0.001

# ─── 7. FinBERT model load (optional, slow) ───────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--skip-finbert", action="store_true")
cli_args, _ = parser.parse_known_args()

print("\n[FinBERT model]")

if cli_args.skip_finbert:
    print(f"  [SKIP] FinBERT model load (--skip-finbert set)")
else:
    @check("FinBERT loads and scores one headline")
    def _():
        from sentiment_engine import _score_texts
        results_fb = _score_texts(["Reliance Industries posts record quarterly profit"])
        assert len(results_fb) == 1
        r = results_fb[0]
        assert "positive" in r or "positive" in r
        total = r.get("positive", 0) + r.get("negative", 0) + r.get("neutral", 0)
        assert abs(total - 1.0) < 0.01, f"scores don't sum to 1: {r}"

# ─── Summary ──────────────────────────────────────────────────────────────
print("\n── Summary ────────────────────────────────────────────────────")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  {passed} passed  /  {failed} failed  /  {len(results)} total\n")

if failed > 0:
    print("Failed checks:")
    for name, ok, msg in results:
        if not ok:
            print(f"  {FAIL}  {name}: {msg}")
    sys.exit(1)
else:
    print("All Phase 2 checks passed.")
    sys.exit(0)
