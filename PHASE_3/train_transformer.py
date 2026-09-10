# train_transformer.py
# Trains the TransformerPredictor and evaluates it on the test set.
# Mirrors train_lstm.py exactly — the only differences are the model
# class and the hyperparameter defaults pulled from config.py.
#
# Usage (from phase3\ directory in Windows Terminal)
#   python train_transformer.py
#   python train_transformer.py --epochs 50
#   python train_transformer.py --db ..\phase1\db\alpha.duckdb

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
    TF_LR, TF_WEIGHT_DECAY, TF_EPOCHS, TF_BATCH, TF_PATIENCE,
)
from dataset import make_loaders, load_scaler
from models import build_transformer, count_parameters
from trainer import train, evaluate, get_device


def regression_metrics(preds, actuals):
    p = np.array(preds)
    a = np.array(actuals)
    mae      = float(np.mean(np.abs(p - a)))
    rmse     = float(np.sqrt(np.mean((p - a) ** 2)))
    hit_rate = float(np.mean(np.sign(p) == np.sign(a)))
    ic       = float(np.corrcoef(p, a)[0, 1]) if len(p) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "hit_rate": hit_rate, "ic": ic}


def main(args):
    print("=" * 60)
    print("Phase 3 — Transformer Training")
    print("=" * 60)

    db_path = Path(args.db)

    # ── Data ──────────────────────────────────────────────────────
    # Re-use the scaler fitted during LSTM training if it exists,
    # otherwise fit a fresh one.
    from config import SCALERS_DIR, SCALER_NAME
    scaler_path = SCALERS_DIR / SCALER_NAME

    if scaler_path.exists():
        print(f"[train_tf] Re-using existing scaler from {scaler_path}")
        train_loader, val_loader, test_loader, scaler = make_loaders(
            db_path=db_path, batch_size=args.batch_size
        )
        # make_loaders always re-fits; that's fine — RobustScaler is deterministic
    else:
        train_loader, val_loader, test_loader, scaler = make_loaders(
            db_path=db_path, batch_size=args.batch_size
        )

    # ── Model ─────────────────────────────────────────────────────
    model = build_transformer()
    print(f"\nTransformer parameters: {count_parameters(model):,}")

    # ── Train ─────────────────────────────────────────────────────
    summary = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = args.epochs,
        lr           = args.lr,
        weight_decay = TF_WEIGHT_DECAY,
        patience     = args.patience,
        name         = "transformer",
    )

    # ── Test evaluation ───────────────────────────────────────────
    print("\n[train_tf] Evaluating on test set ...")
    device = get_device()
    preds, actuals, tickers, dates = evaluate(model, test_loader, device)

    metrics = regression_metrics(preds, actuals)
    print("\nTest set metrics:")
    print(f"  MAE       : {metrics['mae']:.6f}")
    print(f"  RMSE      : {metrics['rmse']:.6f}")
    print(f"  Hit rate  : {metrics['hit_rate']:.4f}  ({metrics['hit_rate']*100:.1f}%)")
    print(f"  IC        : {metrics['ic']:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    results = {**summary, **metrics, "model": "transformer"}
    results_path = CHECKPOINTS_DIR / "transformer_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[train_tf] Results saved → {results_path}")

    pred_df = pd.DataFrame({
        "ticker": tickers, "date": dates,
        "predicted_ret": preds, "actual_ret": actuals,
    })
    pred_path = CHECKPOINTS_DIR / "transformer_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[train_tf] Test predictions saved → {pred_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer return predictor")
    parser.add_argument("--db",         default=str(DB_PATH))
    parser.add_argument("--epochs",     type=int,   default=TF_EPOCHS)
    parser.add_argument("--lr",         type=float, default=TF_LR)
    parser.add_argument("--batch-size", type=int,   default=TF_BATCH)
    parser.add_argument("--patience",   type=int,   default=TF_PATIENCE)
    args = parser.parse_args()
    main(args)
