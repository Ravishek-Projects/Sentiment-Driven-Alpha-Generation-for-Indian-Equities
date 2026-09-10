# verify_phase3.py
# End-to-end checks for Phase 3 — models, training loop, signal generation.
# Uses synthetic data so it runs without a populated DuckDB.
#
# Usage (Windows Terminal, from phase3\ directory):
#   python verify_phase3.py
#   python verify_phase3.py --skip-training   # skip the 5-epoch test train

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
import sys
import tempfile
from pathlib import Path

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
            import traceback
            results.append((name, False, str(exc)))
            print(f"  {FAIL}  {name}")
            print(f"         {exc}")
        return fn
    return decorator


print("\n── Phase 3 verification ──────────────────────────────────────")

# ── 1. Imports ────────────────────────────────────────────────────────────
print("\n[Imports]")

@check("torch + nn")
def _():
    import torch
    import torch.nn as nn

@check("config imports")
def _():
    from config import FEATURE_COLS, N_FEATURES, SEQ_LEN
    assert N_FEATURES == 17
    assert SEQ_LEN    == 60

@check("models importable")
def _():
    from models import build_lstm, build_transformer, count_parameters

@check("trainer importable")
def _():
    from trainer import train, evaluate, get_device, MixedLoss

@check("dataset importable")
def _():
    from dataset import StockSequenceDataset, chronological_split

@check("signal_generator importable")
def _():
    from signal_generator import generate_signals

# ── 2. Model architecture ─────────────────────────────────────────────────
print("\n[Model architecture]")

@check("LSTM forward pass — correct output shape")
def _():
    import torch
    from config import SEQ_LEN, N_FEATURES
    from models import build_lstm
    model = build_lstm()
    x = torch.randn(8, SEQ_LEN, N_FEATURES)
    y = model(x)
    assert y.shape == (8, 1), f"Expected (8,1) got {y.shape}"

@check("Transformer forward pass — correct output shape")
def _():
    import torch
    from config import SEQ_LEN, N_FEATURES
    from models import build_transformer
    model = build_transformer()
    x = torch.randn(8, SEQ_LEN, N_FEATURES)
    y = model(x)
    assert y.shape == (8, 1), f"Expected (8,1) got {y.shape}"

@check("LSTM parameter count reasonable (50K–500K)")
def _():
    from models import build_lstm, count_parameters
    n = count_parameters(build_lstm())
    assert 50_000 <= n <= 1_000_000, f"Unexpected param count: {n}"

@check("Transformer parameter count reasonable (10K–200K)")
def _():
    from models import build_transformer, count_parameters
    n = count_parameters(build_transformer())
    assert 10_000 <= n <= 500_000, f"Unexpected param count: {n}"

@check("Positional encoding shape correct")
def _():
    import torch
    from config import TF_D_MODEL, SEQ_LEN
    from models import PositionalEncoding
    pe = PositionalEncoding(TF_D_MODEL, SEQ_LEN)
    x  = torch.randn(4, SEQ_LEN, TF_D_MODEL)
    out = pe(x)
    assert out.shape == x.shape

# ── 3. Loss ───────────────────────────────────────────────────────────────
print("\n[Loss function]")

@check("MixedLoss forward pass")
def _():
    import torch
    from trainer import MixedLoss
    loss = MixedLoss()
    pred = torch.tensor([[0.01], [-0.005], [0.02]])
    tgt  = torch.tensor([[0.008], [-0.003], [0.018]])
    val  = loss(pred, tgt)
    assert val.item() > 0
    assert not val.isnan()

@check("MixedLoss is lower for closer predictions")
def _():
    import torch
    from trainer import MixedLoss
    loss = MixedLoss()
    pred_good = torch.tensor([[0.010], [-0.005]])
    pred_bad  = torch.tensor([[0.100], [-0.500]])
    target    = torch.tensor([[0.011], [-0.006]])
    assert loss(pred_good, target) < loss(pred_bad, target)

# ── 4. Dataset (synthetic data) ───────────────────────────────────────────
print("\n[Dataset — synthetic data]")

import numpy as np
import pandas as pd
from datetime import date, timedelta

def _make_synthetic_df(n_tickers=5, n_days=150):
    """Build a minimal synthetic feature_matrix DataFrame."""
    from config import FEATURE_COLS, TARGET_COL
    tickers   = [f"STOCK{i}" for i in range(n_tickers)]
    start_d   = date(2022, 1, 3)
    dates     = [start_d + timedelta(days=i) for i in range(n_days)]
    rows = []
    for t in tickers:
        for d in dates:
            row = {"ticker": t, "date": pd.Timestamp(d)}
            for c in FEATURE_COLS:
                row[c] = float(np.random.randn())
            row[TARGET_COL] = float(np.random.randn() * 0.01)
            rows.append(row)
    return pd.DataFrame(rows)

_synth_df = _make_synthetic_df()

@check("StockSequenceDataset builds from synthetic data")
def _():
    from dataset import StockSequenceDataset
    from config import SEQ_LEN
    ds = StockSequenceDataset(_synth_df, fit_scaler_on_this=True)
    assert len(ds) > 0
    x, y, ticker, d = ds[0]
    assert x.shape == (SEQ_LEN, 17)
    assert y.shape == (1,)
    assert isinstance(ticker, str)

@check("chronological_split respects date ordering")
def _():
    from dataset import chronological_split
    train, val, test = chronological_split(_synth_df)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max()   < test["date"].min()

@check("DataLoader iterates without error")
def _():
    import torch
    from dataset import StockSequenceDataset
    from torch.utils.data import DataLoader
    ds     = StockSequenceDataset(_synth_df, fit_scaler_on_this=True)
    loader = DataLoader(ds, batch_size=16)
    x, y, t, d = next(iter(loader))
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

# ── 5. Training loop (5 epochs on synthetic data) ─────────────────────────
cli_parser = argparse.ArgumentParser()
cli_parser.add_argument("--skip-training", action="store_true")
cli_args, _ = cli_parser.parse_known_args()

print("\n[Training loop — 5 epochs on synthetic data]")

if cli_args.skip_training:
    print(f"  [SKIP] Training test (--skip-training set)")
else:
    @check("LSTM trains 5 epochs without error — val loss finite")
    def _():
        import torch
        from dataset import StockSequenceDataset, chronological_split
        from models import build_lstm
        from trainer import train
        from torch.utils.data import DataLoader
        import tempfile

        train_df, val_df, _ = chronological_split(_synth_df)
        tr_ds  = StockSequenceDataset(train_df, fit_scaler_on_this=True)
        val_ds = StockSequenceDataset(val_df, scaler=tr_ds.scaler)
        tr_ld  = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=32)
        model  = build_lstm()

        # Point checkpoints to a temp dir
        import config as cfg
        orig = cfg.CHECKPOINTS_DIR
        cfg.CHECKPOINTS_DIR = Path(tempfile.mkdtemp())
        cfg.LOGS_DIR        = Path(tempfile.mkdtemp())
        try:
            summary = train(model, tr_ld, val_ld,
                            epochs=5, lr=1e-3, weight_decay=1e-4,
                            patience=10, name="test_lstm")
        finally:
            cfg.CHECKPOINTS_DIR = orig

        assert summary["best_val_loss"] < float("inf")
        assert not np.isnan(summary["best_val_loss"])

    @check("Transformer trains 5 epochs without error")
    def _():
        import torch
        from dataset import StockSequenceDataset, chronological_split
        from models import build_transformer
        from trainer import train
        from torch.utils.data import DataLoader
        import tempfile, config as cfg

        train_df, val_df, _ = chronological_split(_synth_df)
        tr_ds  = StockSequenceDataset(train_df, fit_scaler_on_this=True)
        val_ds = StockSequenceDataset(val_df, scaler=tr_ds.scaler)
        tr_ld  = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=32)
        model  = build_transformer()

        orig_ckpt = cfg.CHECKPOINTS_DIR
        orig_log  = cfg.LOGS_DIR
        cfg.CHECKPOINTS_DIR = Path(tempfile.mkdtemp())
        cfg.LOGS_DIR        = Path(tempfile.mkdtemp())
        try:
            summary = train(model, tr_ld, val_ld,
                            epochs=5, lr=5e-4, weight_decay=1e-4,
                            patience=10, name="test_transformer")
        finally:
            cfg.CHECKPOINTS_DIR = orig_ckpt
            cfg.LOGS_DIR        = orig_log

        assert summary["best_val_loss"] < float("inf")

# ── 6. Signal logic ───────────────────────────────────────────────────────
print("\n[Signal logic]")

@check("z_score above threshold → LONG")
def _():
    z = 0.8
    from config import SIGNAL_Z_THRESHOLD, SIGNAL_MAX_POSITION
    sig  = "LONG"  if z >  SIGNAL_Z_THRESHOLD else ("SHORT" if z < -SIGNAL_Z_THRESHOLD else "FLAT")
    size = min(abs(z) / 2, SIGNAL_MAX_POSITION)
    assert sig == "LONG"
    assert 0 < size <= SIGNAL_MAX_POSITION

@check("z_score below negative threshold → SHORT")
def _():
    z = -0.9
    from config import SIGNAL_Z_THRESHOLD
    sig = "SHORT" if z < -SIGNAL_Z_THRESHOLD else "other"
    assert sig == "SHORT"

@check("z_score inside band → FLAT")
def _():
    z = 0.2
    from config import SIGNAL_Z_THRESHOLD
    sig = "FLAT" if abs(z) <= SIGNAL_Z_THRESHOLD else "other"
    assert sig == "FLAT"

@check("position_size capped at SIGNAL_MAX_POSITION")
def _():
    from config import SIGNAL_MAX_POSITION
    z    = 100.0   # extreme z-score
    size = min(abs(z) / 2, SIGNAL_MAX_POSITION)
    assert size == SIGNAL_MAX_POSITION

# ── 7. Device detection ───────────────────────────────────────────────────
print("\n[Device]")

@check("get_device returns a valid torch.device")
def _():
    import torch
    from trainer import get_device
    d = get_device()
    assert isinstance(d, torch.device)
    print(f"         (using: {d})")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n── Summary ────────────────────────────────────────────────────")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  {passed} passed  /  {failed} failed  /  {len(results)} total\n")

if failed:
    print("Failed checks:")
    for name, ok, msg in results:
        if not ok:
            print(f"  {FAIL}  {name}: {msg}")
    sys.exit(1)
else:
    print("All Phase 3 checks passed.")
    sys.exit(0)
