# dataset.py
# PyTorch Dataset that reads feature_matrix from DuckDB and builds
# sliding-window sequences for LSTM / Transformer training.
#
# Key design decisions
# --------------------
# 1. Per-ticker sequences: each stock is treated as an independent time series.
#    Window [t-SEQ_LEN : t] of features predicts target at t+1.
#
# 2. Chronological split: train / val / test are cut by date, never shuffled
#    across the time axis.  Shuffling within the training split is fine.
#
# 3. Scaling: RobustScaler is fit on the training split only, then applied
#    to val and test.  The fitted scaler is saved to disk so inference can
#    use it without re-fitting.
#
# 4. NaN handling: any feature with a NaN is forward-filled within the
#    ticker's series, then zero-filled if the very first row(s) are NaN.

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader

from config import (
    DB_PATH, FEATURE_COLS, TARGET_COL, SEQ_LEN,
    TRAIN_RATIO, VAL_RATIO,
    SCALERS_DIR, SCALER_NAME,
)


# ---------------------------------------------------------------------------
# Data loading from DuckDB
# ---------------------------------------------------------------------------

def load_feature_matrix(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Load all feature_matrix rows that have a filled target.
    Returns a DataFrame sorted by (ticker, date).
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(f"""
            SELECT ticker, date,
                   {", ".join(FEATURE_COLS)},
                   {TARGET_COL}
            FROM   feature_matrix
            WHERE  {TARGET_COL} IS NOT NULL
            ORDER  BY ticker, date
        """).df()
    finally:
        con.close()

    df["date"] = pd.to_datetime(df["date"])
    return df


def load_inference_rows(
    target_date: str,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Load the last SEQ_LEN rows up to and including target_date for every ticker.
    Used at inference time (target_ret_1d may be NULL — that's fine).
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(f"""
            SELECT ticker, date,
                   {", ".join(FEATURE_COLS)}
            FROM   feature_matrix
            WHERE  date <= ?
            ORDER  BY ticker, date
        """, [target_date]).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Chronological split helpers
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test by date boundary (not by row count).
    All tickers present in train are also present in val and test — no leakage.
    """
    all_dates = sorted(df["date"].unique())
    n = len(all_dates)
    train_end_idx = int(n * TRAIN_RATIO)
    val_end_idx   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_end = all_dates[train_end_idx - 1]
    val_end   = all_dates[val_end_idx   - 1]

    train = df[df["date"] <= train_end].copy()
    val   = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test  = df[df["date"] > val_end].copy()

    return train, val, test


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

def fit_scaler(train_df: pd.DataFrame) -> RobustScaler:
    """Fit a RobustScaler on training features and save to disk."""
    scaler = RobustScaler()
    scaler.fit(train_df[FEATURE_COLS].values)
    path = SCALERS_DIR / SCALER_NAME
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[dataset] Scaler saved → {path}")
    return scaler


def load_scaler() -> RobustScaler:
    """Load the saved RobustScaler from disk."""
    path = SCALERS_DIR / SCALER_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. "
            "Run train_lstm.py or train_transformer.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# NaN filling
# ---------------------------------------------------------------------------

def clean_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Fill NaN values with per-column median (not zero).
    Zero-filling RSI=0 injects false signals.
    """
    df = df.copy()
    col_medians = df[feature_cols].median()
    df[feature_cols] = (
        df.groupby("ticker")[feature_cols]
        .transform(lambda s: s.ffill().bfill())
    )
    df[feature_cols] = df[feature_cols].fillna(col_medians)
    return df


class StockSequenceDataset(Dataset):
    """
    Sliding-window dataset. Each sample:
        X : (SEQ_LEN, N_FEATURES)  — scaled features
        y : (1,)                   — raw target (Close-to-Open log return)
    
    NOTE: Target is NOT normalised here.
    The loss function works in raw return space.
    MAE will be ~0.005–0.015 (not 0.93).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[RobustScaler] = None,
        seq_len: int = SEQ_LEN,
        fit_scaler_on_this: bool = False,
        augment: bool = False,        # add noise during training
    ):
        self.seq_len  = seq_len
        self.augment  = augment
        df = clean_features(df, FEATURE_COLS)

        if fit_scaler_on_this:
            self.scaler = fit_scaler(df)
        elif scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = None

        if self.scaler is not None:
            df = df.copy()
            df[FEATURE_COLS] = self.scaler.transform(df[FEATURE_COLS].values)

        self.samples: list[tuple[np.ndarray, float, str, str]] = []
        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        for ticker, grp in df.groupby("ticker"):
            grp   = grp.sort_values("date").reset_index(drop=True)
            feats = grp[FEATURE_COLS].values.astype(np.float32)
            # Raw target — DO NOT normalise here
            target = grp[TARGET_COL].values.astype(np.float32)
            dates  = grp["date"].dt.strftime("%Y-%m-%d").tolist()

            # Skip if target is all NaN
            if np.all(np.isnan(target)):
                continue

            for i in range(self.seq_len, len(grp)):
                y = target[i]
                # Skip rows with NaN target
                if np.isnan(y):
                    continue
                # Skip extreme outliers (circuit breakers, data errors)
                # Daily Close-to-Open returns beyond ±10% are almost always data errors
                if abs(y) > 0.10:
                    continue
                x = feats[i - self.seq_len : i]
                d = dates[i]
                self.samples.append((x, float(y), str(ticker), d))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, ticker, date_str = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
            ticker,
            date_str,
        )
    
    # Flag toggled by trainer
    training_mode = True


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(
    db_path: Path = DB_PATH,
    batch_size: int = 64,
    seq_len: int = SEQ_LEN,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, RobustScaler]:
    df = load_feature_matrix(db_path)

    if df.empty:
        raise RuntimeError("feature_matrix is empty. Complete Phase 2 backfill first.")

    train_df, val_df, test_df = chronological_split(df)
    print(f"[dataset] Rows — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # Fit scaler on training set only
    train_ds = StockSequenceDataset(
        train_df, seq_len=seq_len,
        fit_scaler_on_this=True,
        # augment=True,    # noise augmentation for training only
    )
    scaler = train_ds.scaler

    val_ds  = StockSequenceDataset(val_df,  scaler=scaler, seq_len=seq_len, augment=False)
    test_ds = StockSequenceDataset(test_df, scaler=scaler, seq_len=seq_len, augment=False)

    print(f"[dataset] Samples — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
        scaler,
    )
