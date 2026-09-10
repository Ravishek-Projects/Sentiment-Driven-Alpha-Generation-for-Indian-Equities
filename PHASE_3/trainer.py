# trainer.py
# Shared training engine used by both LSTMPredictor and TransformerPredictor.
#
# Features
# ─────────
# - Mixed MSE + Huber loss (smooths outlier sensitivity)
# - AdamW optimiser with cosine annealing LR schedule
# - Early stopping on validation loss
# - Checkpoint saving (best val-loss model + latest epoch)
# - TensorBoard-compatible CSV loss log
# - Gradient clipping (norm=1.0) — essential for LSTM stability
# - Device auto-selection: CUDA → MPS (Apple Silicon) → CPU

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, LOGS_DIR
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MixedLoss(nn.Module):
    """
    Loss for raw Close-to-Open log returns.
    
    Components:
    - Huber (delta=0.005): robust to overnight gap outliers, smooth near zero
    - Directional: penalises wrong-sign predictions directly
    
    We drop MSE entirely — Huber is strictly better for fat-tailed return distributions.
    MAE on good predictions should be ~0.003–0.008 (0.3%–0.8% in return space).
    """
    def __init__(self, huber_delta: float = 0.005):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber_loss = self.huber(pred, target)

        # Normalise pred and target to unit scale before directional loss
        # so the penalty is not negligible for small return values
        pred_norm   = pred   / (target.abs().mean().detach() + 1e-8)
        target_norm = target / (target.abs().mean().detach() + 1e-8)
        directional_loss = torch.mean(
            torch.nn.functional.relu(-pred_norm * target_norm)
        )

        return 0.6 * huber_loss + 0.4 * directional_loss


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    name: str,
) -> Path:
    path = CHECKPOINTS_DIR / f"{name}.pt"
    torch.save({
        "epoch":      epoch,
        "val_loss":   val_loss,
        "model_state":     model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
    }, path)
    return path


def load_checkpoint(
    model: nn.Module,
    name: str,
    optimiser: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    path = CHECKPOINTS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device or get_device(), weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    if optimiser and "optimiser_state" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser_state"])
    return ckpt


# ---------------------------------------------------------------------------
# CSV loss logger
# ---------------------------------------------------------------------------

class LossLogger:
    def __init__(self, name: str):
        self.path = LOGS_DIR / f"{name}_loss.csv"
        self._file  = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    def log(self, epoch: int, train_loss: float, val_loss: float,
            lr: float, elapsed: float) -> None:
        self._writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}",
                               f"{lr:.2e}", f"{elapsed:.1f}"])
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int,
    lr:           float,
    weight_decay: float,
    patience:     int,
    name:         str,               # used for checkpoint filename + log
) -> dict:
    """
    Train `model` and return a summary dict with best val_loss, best epoch, etc.

    Parameters
    ----------
    model        : LSTMPredictor or TransformerPredictor (already on correct device)
    train_loader : DataLoader (shuffled)
    val_loader   : DataLoader (not shuffled)
    epochs       : maximum training epochs
    lr           : initial learning rate
    weight_decay : L2 regularisation
    patience     : early-stopping patience (epochs without val improvement)
    name         : checkpoint / log file prefix (e.g. "lstm" or "transformer")
    """
    device = get_device()
    model  = model.to(device)

    criterion  = MixedLoss().to(device)
    optimiser  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )
    logger     = LossLogger(name)

    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    t_start        = time.time()

    print(f"\n[trainer] Training {name.upper()} on {device}")
    print(f"          epochs={epochs}  lr={lr}  patience={patience}")
    print(f"          train batches={len(train_loader)}  val batches={len(val_loader)}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, _, _ in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Add small Gaussian noise to inputs during training (standard regularisation)
            # Only during training, never during validation
            x_batch = x_batch + torch.randn_like(x_batch) * 0.005
            # In train(), inside the training loop, replace the backward pass:
            optimiser.zero_grad(set_to_none=True)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            # L1 regularisation on output weights (reduces overconfident predictions)
            l1_lambda = 1e-5
            l1_reg = sum(p.abs().sum() for name, p in model.named_parameters() if 'head' in name)
            loss = loss + l1_lambda * l1_reg

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # tighter clipping
            optimiser.step()
            train_loss += loss.item() * len(x_batch)

        train_loss /= len(train_loader.dataset)

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, _, _ in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred    = model(x_batch)
                val_loss += criterion(pred, y_batch).item() * len(x_batch)
        if len(val_loader.dataset) > 0:
            val_loss /= len(val_loader.dataset)
        else:
            val_loss = float('inf')
            print("\n[WARNING] Validation set is empty! Check your dataset splits and sequence length.")
# -----------------------------------------------------------------

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t_start

        logger.log(epoch, train_loss, val_loss, current_lr, elapsed)

        # Print every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"lr={current_lr:.2e} | {elapsed:.0f}s"
            )

        # ── Checkpoint & early stopping ───────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_count = 0
            save_checkpoint(model, optimiser, epoch, val_loss, f"{name}_best")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n[trainer] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimiser, epoch, val_loss, f"{name}_latest")

    logger.close()

    # Reload best weights into model before returning
    load_checkpoint(model, f"{name}_best", device=device)
    print(f"\n[trainer] Training complete — best val_loss={best_val_loss:.6f} at epoch {best_epoch}")
    print(f"[trainer] Best checkpoint: checkpoints/{name}_best.pt")

    return {
        "model_name":  name,
        "best_epoch":  best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     Optional[torch.device] = None,
) -> tuple[list[float], list[float], list[str], list[str]]:
    """
    Run model in eval mode over `loader`.
    Returns (predictions, actuals, tickers, dates) — all as plain Python lists.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    preds, actuals, tickers, dates = [], [], [], []
    with torch.no_grad():
        for x_batch, y_batch, ticker_batch, date_batch in loader:
            x_batch = x_batch.to(device)
            out     = model(x_batch).cpu().squeeze(-1).tolist()
            ys      = y_batch.squeeze(-1).tolist()
            preds.extend(out if isinstance(out, list) else [out])
            actuals.extend(ys if isinstance(ys, list) else [ys])
            tickers.extend(list(ticker_batch))
            dates.extend(list(date_batch))

    return preds, actuals, tickers, dates
