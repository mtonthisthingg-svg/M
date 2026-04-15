"""
Ensemble model: weighted average of LGBM + 1D CNN, post-hoc calibrated.

Training flow:
  1. Split data: 70% train, 15% calibration holdout, 15% final test
  2. Train LGBM and CNN on train set (with purged CV inside)
  3. Fit isotonic calibrator on calibration holdout
  4. Evaluate on final test — report Brier score, calibration curve
  5. Save all components (LGBM pkl, CNN pt, calibrator pkl)

Inference flow (< 200ms):
  1. LightGBM predict_proba   (flat feature dict)
  2. CNN predict_proba         (sequence array)
  3. Weighted average
  4. Isotonic calibration
  5. Return float in [0.01, 0.99]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from btc_predictor.config import (
    ENSEMBLE_CNN_WEIGHT,
    ENSEMBLE_LGBM_WEIGHT,
    MODEL_DIR,
    PURGED_EMBARGO_S,
    PURGED_KFOLD_N_SPLITS,
)
from btc_predictor.models.calibration import IsotonicCalibrator
from btc_predictor.models.cv import PurgedKFold
from btc_predictor.models.lgbm_model import LGBMPredictor
from btc_predictor.models.temporal_model import BTCTemporalModel


class EnsemblePredictor:
    """
    Blends LGBM (tabular) + CNN (sequential) with isotonic calibration.
    """

    def __init__(
        self,
        lgbm_weight: float = ENSEMBLE_LGBM_WEIGHT,
        cnn_weight:  float = ENSEMBLE_CNN_WEIGHT,
    ) -> None:
        self._lgbm_w = lgbm_weight
        self._cnn_w  = cnn_weight
        self._lgbm   = LGBMPredictor()
        self._cnn:   Optional[BTCTemporalModel] = None
        self._cal    = IsotonicCalibrator()
        self._feature_names: List[str] = []
        self._n_seq_features: int = 0
        self._has_cnn = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_flat: pd.DataFrame,            # (N, n_features) — flat feature matrix
        X_seq:  np.ndarray,              # (N, T, F) — sequence for CNN; nan rows = no seq
        y:      pd.Series,               # (N,) binary labels
        timestamps: pd.Series,           # (N,) UTC epoch seconds
    ) -> dict:
        """
        Full training pipeline with purged CV, calibration, and evaluation.

        Returns a dict with cross-val and holdout metrics.
        """
        N = len(X_flat)
        assert N == len(y) == len(X_seq), "Shape mismatch"

        # 70 / 15 / 15 split (chronological)
        n_train = int(N * 0.70)
        n_cal   = int(N * 0.15)

        X_tr_flat = X_flat.iloc[:n_train]
        X_tr_seq  = X_seq[:n_train]
        y_tr      = y.iloc[:n_train]
        ts_tr     = timestamps.iloc[:n_train]

        X_cal_flat = X_flat.iloc[n_train:n_train + n_cal]
        X_cal_seq  = X_seq[n_train:n_train + n_cal]
        y_cal      = y.iloc[n_train:n_train + n_cal]

        X_test_flat = X_flat.iloc[n_train + n_cal:]
        X_test_seq  = X_seq[n_train + n_cal:]
        y_test      = y.iloc[n_train + n_cal:]

        logger.info(f"[Ensemble] Split: train={n_train}, cal={n_cal}, test={len(y_test)}")

        # ---- LGBM: purged k-fold CV then final fit ----
        cv = PurgedKFold(
            n_splits=PURGED_KFOLD_N_SPLITS,
            embargo_s=PURGED_EMBARGO_S,
            label_horizon_s=300,
        )
        cv_results = self._lgbm.cross_validate(X_tr_flat, y_tr, ts_tr, cv)

        # Final LGBM train on full train set (10% of train as ES val)
        n_es = max(200, n_train // 10)
        self._lgbm.train(
            X_tr_flat.iloc[:-n_es], y_tr.iloc[:-n_es],
            X_tr_flat.iloc[-n_es:], y_tr.iloc[-n_es:],
        )

        # ---- CNN ----
        # Filter rows that have valid sequences
        seq_mask = ~np.isnan(X_tr_seq).any(axis=(1, 2))
        if seq_mask.sum() >= 500:
            n_seq_features = X_tr_seq.shape[2]
            self._cnn = BTCTemporalModel(n_features=n_seq_features)
            self._cnn.build()
            n_cnn_es = max(100, int(seq_mask.sum() * 0.1))
            valid_idx = np.where(seq_mask)[0]
            tr_idx, es_idx = valid_idx[:-n_cnn_es], valid_idx[-n_cnn_es:]
            self._cnn.train_model(
                X_tr_seq[tr_idx], y_tr.values[tr_idx],
                X_tr_seq[es_idx], y_tr.values[es_idx],
            )
            self._has_cnn = True
            logger.info(f"[Ensemble] CNN trained on {len(tr_idx)} sequences")
        else:
            logger.warning("[Ensemble] Not enough sequences for CNN; using LGBM only")
            self._has_cnn = False

        # ---- Calibration ----
        cal_probs_raw = self._raw_blend(X_cal_flat, X_cal_seq)
        self._cal.fit(cal_probs_raw, y_cal.values)
        logger.info("[Ensemble] Calibrator fitted on holdout")

        # ---- Final evaluation ----
        test_raw   = self._raw_blend(X_test_flat, X_test_seq)
        test_cal   = self._cal.transform(test_raw)
        brier_raw  = float(np.mean((test_raw  - y_test.values) ** 2))
        brier_cal  = float(np.mean((test_cal  - y_test.values) ** 2))
        accuracy   = float(np.mean((test_cal > 0.5) == y_test.values))

        logger.info(
            f"[Ensemble] FINAL TEST — "
            f"brier_raw={brier_raw:.4f} brier_cal={brier_cal:.4f} "
            f"accuracy={accuracy:.3f} n={len(y_test)}"
        )

        return {
            "cv_results": cv_results,
            "brier_raw":  brier_raw,
            "brier_cal":  brier_cal,
            "accuracy":   accuracy,
            "feature_importance": self._lgbm.feature_importance().to_dict(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: Dict[str, float],
        sequence: Optional[np.ndarray] = None,
    ) -> float:
        """
        Single-sample inference. Must complete in < 200ms.

        Args:
            features: Flat feature dict from FeaturePipeline.
            sequence: (T, F) array or None.

        Returns:
            Calibrated P(UP) in [0.01, 0.99].
        """
        t0 = time.perf_counter()

        row = pd.DataFrame([features])
        # Align columns with training order
        for col in self._feature_names:
            if col not in row.columns:
                row[col] = 0.0
        row = row[self._feature_names]

        lgbm_prob = float(self._lgbm.predict_proba(row)[0])

        if self._has_cnn and self._cnn is not None and sequence is not None:
            cnn_prob = float(self._cnn.predict_proba(sequence)[0])
            raw = self._lgbm_w * lgbm_prob + self._cnn_w * cnn_prob
        else:
            raw = lgbm_prob

        cal_prob = float(self._cal.transform(np.array([raw]))[0])
        result   = float(np.clip(cal_prob, 0.01, 0.99))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 200:
            logger.warning(f"[Ensemble] Inference took {elapsed_ms:.0f}ms > 200ms budget")

        return result

    def _raw_blend(
        self,
        X_flat: pd.DataFrame,
        X_seq:  np.ndarray,
    ) -> np.ndarray:
        """Raw (uncalibrated) blended probability."""
        lgbm_probs = self._lgbm.predict_proba(X_flat)

        if self._has_cnn and self._cnn is not None:
            # Only predict where sequences are valid
            seq_mask = ~np.isnan(X_seq).any(axis=(1, 2))
            cnn_probs = np.full(len(X_flat), 0.5)
            if seq_mask.sum() > 0:
                cnn_probs[seq_mask] = self._cnn.predict_proba(X_seq[seq_mask])
            raw = self._lgbm_w * lgbm_probs + self._cnn_w * cnn_probs
        else:
            raw = lgbm_probs

        return raw

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        path = Path(MODEL_DIR)
        path.mkdir(parents=True, exist_ok=True)
        self._lgbm.save("lgbm_model")
        if self._has_cnn and self._cnn:
            self._cnn.save("cnn_model")
        joblib.dump(self._cal, path / "calibrator.pkl")
        joblib.dump({
            "feature_names": self._feature_names,
            "has_cnn": self._has_cnn,
            "lgbm_w":  self._lgbm_w,
            "cnn_w":   self._cnn_w,
        }, path / "ensemble_meta.pkl")
        logger.info(f"[Ensemble] Saved all components to {path}")

    def load(self) -> None:
        path = Path(MODEL_DIR)
        self._lgbm.load("lgbm_model")
        meta = joblib.load(path / "ensemble_meta.pkl")
        self._feature_names = meta["feature_names"]
        self._has_cnn       = meta["has_cnn"]
        self._lgbm_w        = meta["lgbm_w"]
        self._cnn_w         = meta["cnn_w"]
        self._cal           = joblib.load(path / "calibrator.pkl")
        if self._has_cnn:
            # n_features stored inside CNN model file
            import torch
            d = torch.load(path / "cnn_model.pt", map_location="cpu")
            self._cnn = BTCTemporalModel(n_features=d["n_features"])
            self._cnn.load("cnn_model")
        logger.info(f"[Ensemble] Loaded from {path}")
