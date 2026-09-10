# models.py
# Two PyTorch model architectures for next-day return prediction.
#
# Architecture 1 — LSTMPredictor
# ─────────────────────────────────────────────────────────────────
# Input  : (batch, seq_len, n_features)
# Layers : Linear projection → 2-layer LSTM → dropout → FC(1)
# Output : (batch, 1)  — predicted log return
#
# Architecture 2 — TransformerPredictor
# ─────────────────────────────────────────────────────────────────
# Input  : (batch, seq_len, n_features)
# Layers : Linear embedding → learnable positional encoding →
#          N × TransformerEncoderLayer → mean-pool → FC(1)
# Output : (batch, 1)  — predicted log return
#
# Both models output a scalar log-return prediction.
# The calling code (train_*.py) handles the loss, optimiser, and checkpointing.

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import (
    N_FEATURES, SEQ_LEN,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    TF_D_MODEL, TF_NHEAD, TF_DIM_FF, TF_N_LAYERS, TF_DROPOUT,
)


# ===========================================================================
# Architecture 1 — LSTM
# ===========================================================================

class LSTMPredictor(nn.Module):
    """
    Smaller LSTM with stronger regularisation.
    Target: ~80K parameters for ~29K samples (ratio ~0.36).
    """
    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden:     int = 64,        # reduced from 128
        n_layers:   int = 1,         # reduced from 2 (single layer = less overfitting)
        dropout:    float = 0.40,    # increased from 0.20
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.norm       = nn.LayerNorm(hidden)
        self.lstm       = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0,             # dropout inside LSTM only works with n_layers>1
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.input_proj(x))
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last)


# ===========================================================================
# Architecture 2 — Transformer
# ===========================================================================

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding (not sinusoidal).
    Each of the SEQ_LEN positions gets its own trainable embedding.
    This works better than sinusoidal for short financial sequences.
    """

    def __init__(self, d_model: int, max_len: int = SEQ_LEN):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)"""
        positions = torch.arange(x.size(1), device=x.device)   # (T,)
        return x + self.pos_emb(positions)                      # broadcast over batch


class TransformerPredictor(nn.Module):
    """
    Smaller Transformer with stronger regularisation.
    """
    def __init__(
        self,
        n_features: int  = N_FEATURES,
        d_model:    int  = 32,       # reduced from 64
        nhead:      int  = 4,
        dim_ff:     int  = 64,       # reduced from 256
        n_layers:   int  = 2,
        dropout:    float = 0.30,    # increased
        seq_len:    int  = SEQ_LEN,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x      = self.pos_enc(self.input_proj(x))
        x      = self.encoder(x)
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.head(pooled)


# ===========================================================================
# Model factory
# ===========================================================================

def build_lstm() -> LSTMPredictor:
    return LSTMPredictor()


def build_transformer() -> TransformerPredictor:
    return TransformerPredictor()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke-test of both architectures
    B, T, F = 4, SEQ_LEN, N_FEATURES
    x = torch.randn(B, T, F)

    lstm = build_lstm()
    tf   = build_transformer()

    y_lstm = lstm(x)
    y_tf   = tf(x)

    print(f"Input shape  : {x.shape}")
    print(f"LSTM output  : {y_lstm.shape}  params: {count_parameters(lstm):,}")
    print(f"TF   output  : {y_tf.shape}    params: {count_parameters(tf):,}")
    assert y_lstm.shape == (B, 1)
    assert y_tf.shape   == (B, 1)
    print("models.py smoke-test passed.")
