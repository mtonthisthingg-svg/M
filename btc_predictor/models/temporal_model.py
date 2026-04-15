"""
1D CNN temporal model for BTC direction prediction.

Input:  (batch, SEQUENCE_LENGTH_S, n_features)  — 60 seconds of 1s bars
Output: P(UP) scalar

Architecture:
  Conv1D → BN → ReLU → MaxPool  (×3 layers, increasing dilation)
  GlobalAvgPool → Dense(64) → Dropout → Dense(1) → Sigmoid

The temporal model captures short-term momentum patterns that the
LGBM model may miss due to its flattened feature representation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from btc_predictor.config import (
    CNN_BATCH_SIZE,
    CNN_DROPOUT,
    CNN_EPOCHS,
    CNN_HIDDEN_CHANNELS,
    CNN_KERNEL_SIZE,
    CNN_LR,
    CNN_NUM_LAYERS,
    MODEL_DIR,
    SEQUENCE_LENGTH_S,
)


class BTCTemporalModel:
    """
    1D CNN for sequential feature prediction.
    Requires PyTorch.
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self._model: Optional["torch.nn.Module"] = None
        self._device: Optional["torch.device"] = None

    def build(self) -> None:
        import torch
        import torch.nn as nn

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model  = _CNN1D(
            n_features=self.n_features,
            hidden_channels=CNN_HIDDEN_CHANNELS,
            kernel_size=CNN_KERNEL_SIZE,
            n_layers=CNN_NUM_LAYERS,
            dropout=CNN_DROPOUT,
        ).to(self._device)
        logger.info(
            f"[CNN] Built model: n_features={self.n_features}, "
            f"params={sum(p.numel() for p in self._model.parameters()):,d}, "
            f"device={self._device}"
        )

    def train_model(
        self,
        X_seqs: np.ndarray,     # (N, T, F)
        y: np.ndarray,          # (N,)
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            self.build()

        X_t = torch.FloatTensor(X_seqs).to(self._device)
        y_t = torch.FloatTensor(y).to(self._device)
        ds  = TensorDataset(X_t, y_t)
        dl  = DataLoader(ds, batch_size=CNN_BATCH_SIZE, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=CNN_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CNN_EPOCHS)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(CNN_EPOCHS):
            self._model.train()
            train_loss = 0.0
            for xb, yb in dl:
                optimizer.zero_grad()
                pred = self._model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            train_loss /= len(dl)

            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"[CNN epoch {epoch+1}/{CNN_EPOCHS}] "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                    )

        # Restore best weights
        if best_state:
            self._model.load_state_dict(best_state)

        return {"best_val_loss": best_val_loss}

    def _eval_loss(self, X: np.ndarray, y: np.ndarray, criterion) -> float:
        import torch
        self._model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self._device)
            y_t = torch.FloatTensor(y).to(self._device)
            pred = self._model(X_t).squeeze(-1)
            return criterion(pred, y_t).item()

    def predict_proba(self, X_seqs: np.ndarray) -> np.ndarray:
        """
        Args:
            X_seqs: (N, T, F) or (T, F) for single sample.
        Returns:
            (N,) array of P(UP).
        """
        import torch

        if self._model is None:
            raise RuntimeError("Model not built/trained")

        single = X_seqs.ndim == 2
        if single:
            X_seqs = X_seqs[np.newaxis]   # (1, T, F)

        self._model.eval()
        with torch.no_grad():
            X_t  = torch.FloatTensor(X_seqs).to(self._device)
            pred = self._model(X_t).squeeze(-1).cpu().numpy()

        return pred[0:1] if single else pred

    def save(self, name: str = "cnn_model") -> Path:
        import torch
        path = Path(MODEL_DIR) / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._model.state_dict(),
            "n_features":  self.n_features,
        }, path)
        logger.info(f"[CNN] Saved to {path}")
        return path

    def load(self, name: str = "cnn_model") -> None:
        import torch
        path = Path(MODEL_DIR) / f"{name}.pt"
        data = torch.load(path, map_location="cpu")
        self.n_features = data["n_features"]
        self.build()
        self._model.load_state_dict(data["state_dict"])
        self._model.to(self._device)
        logger.info(f"[CNN] Loaded from {path}")


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def _CNN1D(n_features, hidden_channels, kernel_size, n_layers, dropout):
    """Build and return the CNN1D model as a PyTorch Module."""
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = n_features
            for i in range(n_layers):
                out_ch = hidden_channels * (2 ** min(i, 1))
                # Causal padding: pad left only so no future leakage
                pad = (kernel_size - 1)
                layers += [
                    nn.ConstantPad1d((pad, 0), 0),
                    nn.Conv1d(in_ch, out_ch, kernel_size, dilation=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_ch = out_ch

            self.conv = nn.Sequential(*layers)
            # Global average pooling then classifier
            self.classifier = nn.Sequential(
                nn.Linear(in_ch, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: (B, T, F) → (B, F, T) for Conv1D
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = x.mean(dim=-1)    # global average pool over T
            return self.classifier(x)

    return Model()
