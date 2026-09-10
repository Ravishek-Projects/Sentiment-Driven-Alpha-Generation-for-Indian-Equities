# train_lstm.py
# Trains the LSTMPredictor and evaluates it on the test set.
#
# What this script does
# ─────────────────────
# 1. Loads feature_matrix from DuckDB
# 2. Splits chronologically (train / val / test)
# 3. Fits a RobustScaler on the training set and saves it
# 4. Trains the LSTM with early stopping
# 5. Evaluates on the test set and prints metrics
# 6. Saves the best checkpoint to checkpoints/lstm_best.pt
#
# Usage (from phase3\ directory in Windows Terminal)
#   python train_lstm.py
#   python train_lstm.py --epochs 50
#   python train_lstm.py --db ..\phase1\db\alpha.duckdb

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DB_PATH, CHECKPOINTS_DIR,
    LSTM_LR, LSTM_WEIGHT_DECAY, LSTM_EPOCHS, LSTM_BATCH, LSTM_PATIENCE,
)
from dataset import make_loaders
from models import build_lstm, count_parameters
from trainer import train, evaluate, get_device


def regression_metrics(preds: list[float], actuals: list[float]) -> dict:
    p = np.array(preds)
    a = np.array(actuals)
    mae  = float(np.mean(np.abs(p - a)))
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    # Directional accuracy — did model predict the correct sign?
    hit_rate = float(np.mean(np.sign(p) == np.sign(a)))
    # Information coefficient — Pearson correlation
    ic = float(np.corrcoef(p, a)[0, 1]) if len(p) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "hit_rate": hit_rate, "ic": ic}


def main(args):
    print("=" * 60)
    print("Phase 3 — LSTM Training")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────
    db_path = Path(args.db)
    train_loader, val_loader, test_loader, scaler = make_loaders(
        db_path=db_path,
        batch_size=args.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = build_lstm()
    print(f"\nLSTM parameters: {count_parameters(model):,}")

    # ── Train ─────────────────────────────────────────────────────
    summary = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = args.epochs,
        lr           = args.lr,
        weight_decay = LSTM_WEIGHT_DECAY,
        patience     = args.patience,
        name         = "lstm",
    )

    # ── Test evaluation ───────────────────────────────────────────
    print("\n[train_lstm] Evaluating on test set ...")
    device = get_device()
    preds, actuals, tickers, dates = evaluate(model, test_loader, device)

    metrics = regression_metrics(preds, actuals)
    print("\nTest set metrics:")
    print(f"  MAE       : {metrics['mae']:.6f}  (mean abs error in log-return units)")
    print(f"  RMSE      : {metrics['rmse']:.6f}")
    print(f"  Hit rate  : {metrics['hit_rate']:.4f}  ({metrics['hit_rate']*100:.1f}% correct direction)")
    print(f"  IC        : {metrics['ic']:.4f}  (Pearson corr pred vs actual)")

    # ── Save results ──────────────────────────────────────────────
    results = {**summary, **metrics, "model": "lstm"}
    results_path = CHECKPOINTS_DIR / "lstm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[train_lstm] Results saved → {results_path}")

    # ── Save test predictions to CSV for analysis ─────────────────
    pred_df = pd.DataFrame({
        "ticker": tickers, "date": dates,
        "predicted_ret": preds, "actual_ret": actuals,
    })
    pred_path = CHECKPOINTS_DIR / "lstm_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[train_lstm] Test predictions saved → {pred_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM return predictor")
    parser.add_argument("--db",         default=str(DB_PATH))
    parser.add_argument("--epochs",     type=int,   default=LSTM_EPOCHS)
    parser.add_argument("--lr",         type=float, default=LSTM_LR)
    parser.add_argument("--batch-size", type=int,   default=LSTM_BATCH)
    parser.add_argument("--patience",   type=int,   default=LSTM_PATIENCE)
    args = parser.parse_args()
    main(args)
