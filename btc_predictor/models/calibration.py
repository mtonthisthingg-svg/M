"""
Probability calibration via isotonic regression.

A model with raw accuracy of 55% but miscalibrated confidence loses money.
A model with 52% accuracy but well-calibrated probabilities makes money.

We use isotonic regression (non-parametric, monotone) fitted on a holdout set.
The reliability diagram (calibration curve) is exported for the dashboard.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """
    Post-hoc probability calibration using isotonic regression.

    Usage:
        cal = IsotonicCalibrator()
        cal.fit(holdout_probs, holdout_labels)
        calibrated = cal.transform(raw_probs)
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Fit on holdout set. probs in [0,1], labels in {0,1}."""
        self._model.fit(probs, labels)
        self._fitted = True

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probs
        return self._model.transform(probs).clip(0.01, 0.99)

    def fit_transform(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        self.fit(probs, labels)
        return self.transform(probs)

    def reliability_diagram(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Compute reliability diagram data.

        Returns:
            mean_predicted:  Mean predicted probability per bin.
            fraction_pos:    Fraction of positives per bin.
            counts:          Number of samples per bin.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        mean_pred, frac_pos, counts = [], [], []

        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                mean_pred.append(float(np.mean(probs[mask])))
                frac_pos.append(float(np.mean(labels[mask])))
                counts.append(int(mask.sum()))

        return mean_pred, frac_pos, counts

    def brier_score(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Brier score (lower is better, 0.25 = no skill, 0 = perfect)."""
        return float(np.mean((probs - labels) ** 2))
