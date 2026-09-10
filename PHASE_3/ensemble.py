# ensemble.py
# Weighted ensemble of LSTM + Transformer predictions.
# Also runs the ablation study: text-only vs text+tech vs full.
#
# Usage
#   python ensemble.py                    # evaluate ensemble on test set
#   python ensemble.py --ablation         # run all 4 ablation variants
#   python ensemble.py --tune-weights     # grid-search optimal blend weights

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DB_PATH, CHECKPOINTS_DIR,
    ENSEMBLE_W_LSTM, ENSEMBLE_W_TF,
    FEATURE_COLS,
)
from dataset import make_loaders, StockSequenceDataset, chronological_split, load_feature_matrix, clean_features
from models import build_lstm, build_transformer
from trainer import load_checkpoint, evaluate, get_device


# ---------------------------------------------------------------------------
# Load both trained models
# ---------------------------------------------------------------------------

def load_trained_models(device):
    lstm = build_lstm().to(device)
    tf   = build_transformer().to(device)

    try:
        load_checkpoint(lstm, "lstm_best",        device=device)
        load_checkpoint(tf,   "transformer_best", device=device)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Trained model not found: {e}\n"
            "Run train_lstm.py and train_transformer.py first."
        )

    lstm.eval()
    tf.eval()
    return lstm, tf


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------

def ensemble_predict(
    loader:  DataLoader,
    w_lstm:  float = ENSEMBLE_W_LSTM,
    w_tf:    float = ENSEMBLE_W_TF,
    device:  torch.device | None = None,
) -> pd.DataFrame:
    """
    Run both models over `loader` and return a DataFrame with columns:
        ticker, date, pred_lstm, pred_tf, pred_ensemble, actual_ret
    """
    if device is None:
        device = get_device()

    lstm, tf = load_trained_models(device)

    p_lstm, actuals, tickers, dates = evaluate(lstm, loader, device)
    p_tf,   _,       _,       _     = evaluate(tf,   loader, device)

    p_ensemble = [w_lstm * l + w_tf * t for l, t in zip(p_lstm, p_tf)]

    return pd.DataFrame({
        "ticker":       tickers,
        "date":         dates,
        "pred_lstm":    p_lstm,
        "pred_tf":      p_tf,
        "pred_ensemble":p_ensemble,
        "actual_ret":   actuals,
    })


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def metrics(preds, actuals, label=""):
    p = np.array(preds)
    a = np.array(actuals)
    mae      = np.mean(np.abs(p - a))
    rmse     = np.sqrt(np.mean((p - a) ** 2))
    hit_rate = np.mean(np.sign(p) == np.sign(a))
    ic       = float(np.corrcoef(p, a)[0, 1]) if len(p) > 1 else 0.0
    tag      = f"[{label}] " if label else ""
    print(f"  {tag}MAE={mae:.6f}  RMSE={rmse:.6f}  "
          f"HitRate={hit_rate*100:.1f}%  IC={ic:.4f}")
    return {"label": label, "mae": float(mae), "rmse": float(rmse),
            "hit_rate": float(hit_rate), "ic": ic}


# ---------------------------------------------------------------------------
# Weight tuning (grid search on val set)
# ---------------------------------------------------------------------------

def tune_weights(val_loader, device) -> tuple[float, float]:
    """
    Grid-search ensemble weights over val set to minimise RMSE.
    Returns (w_lstm, w_tf).
    """
    lstm, tf = load_trained_models(device)
    p_lstm, actuals, _, _ = evaluate(lstm, val_loader, device)
    p_tf,   _,       _, _ = evaluate(tf,   val_loader, device)

    best_rmse = float("inf")
    best_w    = (ENSEMBLE_W_LSTM, ENSEMBLE_W_TF)

    for w in range(0, 11):               # w_lstm from 0.0 to 1.0
        wl = w / 10
        wt = 1.0 - wl
        ens  = [wl * l + wt * t for l, t in zip(p_lstm, p_tf)]
        rmse = float(np.sqrt(np.mean((np.array(ens) - np.array(actuals)) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w    = (round(wl, 2), round(wt, 2))

    print(f"\n[ensemble] Best weights → LSTM={best_w[0]}  TF={best_w[1]}  "
          f"val RMSE={best_rmse:.6f}")
    return best_w


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = {
    "text_only":      ["sent_pos", "sent_neg", "sent_neu", "sent_spread", "news_count",
                       "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"],
    "text_plus_tech": ["sent_pos", "sent_neg", "sent_neu", "sent_spread", "news_count",
                       "rsi_14", "price_to_ma20", "vol_20d", "rel_volume", "bb_position",
                       "lag_ret_1d", "lag_ret_2d",
                       "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"],
    "tech_only":      ["rsi_14", "price_to_ma20", "vol_20d", "rel_volume", "bb_position",
                       "lag_ret_1d", "lag_ret_2d",
                       "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"],
    "full":           FEATURE_COLS,   # all 17 features
}


def run_ablation(db_path: Path, device: torch.device) -> list[dict]:
    """
    For each ablation variant:
      1. Build fresh LSTM + Transformer with the reduced feature set
      2. Train (20 epochs quick-train for ablation purposes)
      3. Report test metrics
    Returns list of metric dicts.
    """
    from sklearn.preprocessing import RobustScaler
    import torch.nn as nn

    df_full  = load_feature_matrix(db_path)
    train_df, val_df, test_df = chronological_split(df_full)

    results = []
    for variant_name, feat_cols in ABLATION_VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"Ablation variant: {variant_name}  ({len(feat_cols)} features)")
        print(f"{'='*50}")

        n_feat = len(feat_cols)

        # Build variant models (smaller if fewer features)
        from models import LSTMPredictor, TransformerPredictor
        v_lstm = LSTMPredictor(n_features=n_feat).to(device)
        v_tf   = TransformerPredictor(n_features=n_feat).to(device)

        # Scale using only the variant's feature columns
        def _clean(df):
            return clean_features(df, feat_cols)

        scaler = RobustScaler()
        t_clean = _clean(train_df.copy())
        scaler.fit(t_clean[feat_cols].values)

        # Override FEATURE_COLS temporarily — build custom datasets
        import importlib, config as cfg_module
        original_cols = cfg_module.FEATURE_COLS
        cfg_module.FEATURE_COLS = feat_cols  # monkey-patch for Dataset

# Override FEATURE_COLS temporarily — build custom datasets
# Override FEATURE_COLS temporarily — build custom datasets
        import dataset
        import config  # <--- Add this line to grab your global settings
        
        original_cols = dataset.FEATURE_COLS
        dataset.FEATURE_COLS = feat_cols  # Patch the dataset namespace directly

        try:
            from dataset import StockSequenceDataset
            train_ds = StockSequenceDataset.__new__(StockSequenceDataset)
            
            # Dynamically link to your config file instead of hardcoding 60
            train_ds.seq_len = config.SEQ_LEN 
            
            train_ds.scaler  = scaler
            t_scaled = _clean(train_df.copy())
            t_scaled[feat_cols] = scaler.transform(t_scaled[feat_cols].values)
            train_ds.samples = []
            train_ds._build = StockSequenceDataset._build.__get__(train_ds)
            train_ds._build(t_scaled)

            test_ds = StockSequenceDataset.__new__(StockSequenceDataset)
            
            # Dynamically link here too
            test_ds.seq_len = config.SEQ_LEN 
            
            test_ds.scaler  = scaler
            test_ds.samples = []
            test_ds._build  = StockSequenceDataset._build.__get__(test_ds)
            te_scaled = _clean(test_df.copy())
            te_scaled[feat_cols] = scaler.transform(te_scaled[feat_cols].values)
            test_ds._build(te_scaled)
        finally:
            dataset.FEATURE_COLS = original_cols  # Restore it safely

        from torch.utils.data import DataLoader as DL
        tr_loader = DL(train_ds, batch_size=64, shuffle=True)
        te_loader = DL(test_ds,  batch_size=64, shuffle=False)

        # Quick 20-epoch train for ablation
        from trainer import MixedLoss
        for model_name, model in [("lstm", v_lstm), ("tf", v_tf)]:
            opt  = torch.optim.AdamW(model.parameters(), lr=1e-3)
            crit = MixedLoss().to(device)
            model.train()
            for ep in range(20):
                for xb, yb, _, _ in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = crit(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

        # Evaluate ensemble on test
        p_lstm, acts, ticks, dts = evaluate(v_lstm, te_loader, device)
        p_tf,   _,    _,     _   = evaluate(v_tf,   te_loader, device)
        p_ens = [0.4*l + 0.6*t for l, t in zip(p_lstm, p_tf)]

        m = metrics(p_ens, acts, label=variant_name)
        results.append(m)

    # Print comparison table
    print("\n── Ablation summary ────────────────────────────────────")
    print(f"  {'Variant':<20} {'HitRate':>10} {'IC':>8} {'RMSE':>10}")
    for r in results:
        print(f"  {r['label']:<20} {r['hit_rate']*100:>9.1f}% {r['ic']:>8.4f} {r['rmse']:>10.6f}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble evaluation and ablation")
    parser.add_argument("--db",           default=str(DB_PATH))
    parser.add_argument("--ablation",     action="store_true",
                        help="Run ablation study across feature groups")
    parser.add_argument("--tune-weights", action="store_true",
                        help="Grid-search optimal LSTM/TF blend weights on val set")
    args = parser.parse_args()

    db_path = Path(args.db)
    device  = get_device()

    _, val_loader, test_loader, _ = make_loaders(db_path=db_path)

    if args.tune_weights:
        w_lstm, w_tf = tune_weights(val_loader, device)
        print(f"\nUpdate config.py:")
        print(f"  ENSEMBLE_W_LSTM = {w_lstm}")
        print(f"  ENSEMBLE_W_TF   = {w_tf}")

    elif args.ablation:
        results = run_ablation(db_path, device)
        out_path = CHECKPOINTS_DIR / "ablation_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nAblation results saved → {out_path}")

    else:
        # Standard ensemble evaluation
        print("\n[ensemble] Evaluating ensemble on test set ...")
        df_results = ensemble_predict(test_loader, device=device)

        print("\nModel comparison on test set:")
        metrics(df_results["pred_lstm"].tolist(),     df_results["actual_ret"].tolist(), "LSTM")
        metrics(df_results["pred_tf"].tolist(),       df_results["actual_ret"].tolist(), "Transformer")
        metrics(df_results["pred_ensemble"].tolist(), df_results["actual_ret"].tolist(), "Ensemble")

        out_path = CHECKPOINTS_DIR / "ensemble_test_predictions.csv"
        df_results.to_csv(out_path, index=False)
        print(f"\nEnsemble predictions saved → {out_path}")
