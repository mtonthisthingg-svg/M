"""
LightGBM binary classifier for BTC direction prediction.

Target: sign(chainlink_close - chainlink_open) at 5-min window.
  1 = UP (close >= open), 0 = DOWN.

Training:
  - Uses purged k-fold CV with embargo (no look-ahead).
  - Feature importance exported for inspection.
  - SHAP values available for calibration analysis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from btc_predictor.config import LGBM_PARAMS, MODEL_DIR


class LGBMPredictor:
    """
    LightGBM wrapper with train/predict/save/load interface.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        self._params = {**LGBM_PARAMS, **(params or {})}
        self._model = None
        self._feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """
        Train LightGBM model with early stopping on validation set.

        Returns:
            Dict with training metrics.
        """
        import lightgbm as lgb

        self._feature_names = list(X_train.columns)

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

        params = {k: v for k, v in self._params.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self._params.get("early_stopping_rounds", 30),
                verbose=False,
            ),
            lgb.log_evaluation(period=50),
        ]

        evals_result: dict = {}
        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=self._params.get("n_estimators", 500),
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
            evals_result=evals_result,
        )

        best_iter = self._model.best_iteration
        best_val_loss = min(evals_result["val"]["binary_logloss"])
        logger.info(f"[LGBM] Best iter: {best_iter}, val logloss: {best_val_loss:.4f}")

        return {
            "best_iteration": best_iter,
            "best_val_logloss": best_val_loss,
            "feature_names": self._feature_names,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(UP) for each row. Shape (n,)."""
        if self._model is None:
            raise RuntimeError("Model not trained")
        return self._model.predict(X[self._feature_names])

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importances sorted descending."""
        if self._model is None:
            raise RuntimeError("Model not trained")
        imp = self._model.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)

    def save(self, name: str = "lgbm_model") -> Path:
        """Save model to MODEL_DIR/{name}.pkl"""
        path = Path(MODEL_DIR) / f"{name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "features": self._feature_names}, path)
        logger.info(f"[LGBM] Saved to {path}")
        return path

    def load(self, name: str = "lgbm_model") -> None:
        """Load model from MODEL_DIR/{name}.pkl"""
        path = Path(MODEL_DIR) / f"{name}.pkl"
        data = joblib.load(path)
        self._model         = data["model"]
        self._feature_names = data["features"]
        logger.info(f"[LGBM] Loaded from {path}")

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
        cv,
    ) -> List[dict]:
        """
        Run purged k-fold CV. Returns list of per-fold metrics.
        """
        from btc_predictor.models.calibration import IsotonicCalibrator

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, timestamps)):
            if len(train_idx) < 100 or len(val_idx) < 10:
                continue

            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_vl, y_vl = X.iloc[val_idx],   y.iloc[val_idx]

            # Use last 10% of train as internal val for early stopping
            n_inner_val = max(100, len(X_tr) // 10)
            X_tr_inner = X_tr.iloc[:-n_inner_val]
            y_tr_inner = y_tr.iloc[:-n_inner_val]
            X_es       = X_tr.iloc[-n_inner_val:]
            y_es       = y_tr.iloc[-n_inner_val:]

            metrics = self.train(X_tr_inner, y_tr_inner, X_es, y_es)
            raw_probs = self.predict_proba(X_vl)

            # Calibrate within-fold
            cal = IsotonicCalibrator()
            cal.fit(self.predict_proba(X_es), y_es.values)
            cal_probs = cal.transform(raw_probs)

            brier_raw = float(np.mean((raw_probs - y_vl.values) ** 2))
            brier_cal = float(np.mean((cal_probs - y_vl.values) ** 2))
            accuracy  = float(np.mean((raw_probs > 0.5) == y_vl.values))

            fold_results.append({
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "brier_raw": brier_raw,
                "brier_cal": brier_cal,
                "accuracy": accuracy,
                **metrics,
            })
            logger.info(
                f"[LGBM CV fold {fold_idx}] "
                f"acc={accuracy:.3f} brier_raw={brier_raw:.4f} brier_cal={brier_cal:.4f}"
            )

        return fold_results
