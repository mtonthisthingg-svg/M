"""
Purged k-fold cross-validation with embargo.

Prevents leakage when labels are forward-looking (5-min return):
  - Purging: remove training samples whose label window overlaps with
    any test sample's feature window.
  - Embargo: add a gap after each test fold to prevent leakage from
    autocorrelated residuals.

Reference: Lopez de Prado, "Advances in Financial Machine Learning" ch.7
"""

from __future__ import annotations

from typing import Generator, List, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """
    Time-series cross-validator with purging and embargo.

    Args:
        n_splits:       Number of folds.
        embargo_s:      Number of seconds to remove after each test fold.
        label_horizon_s: Forward horizon of the label (e.g. 300 for 5-min).
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_s: int = 60,
        label_horizon_s: int = 300,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_s = embargo_s
        self.label_horizon_s = label_horizon_s

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield (train_idx, test_idx) pairs.

        Args:
            X:          Feature matrix (index = integer position).
            y:          Labels.
            timestamps: UTC epoch seconds for each sample (same index as X).
        """
        n = len(X)
        fold_size = n // self.n_splits
        indices = np.arange(n)
        ts_arr = timestamps.values if hasattr(timestamps, "values") else np.array(timestamps)

        for fold in range(self.n_splits):
            test_start_idx = fold * fold_size
            test_end_idx   = (fold + 1) * fold_size if fold < self.n_splits - 1 else n

            test_idx = indices[test_start_idx:test_end_idx]
            if len(test_idx) == 0:
                continue

            test_ts_start = ts_arr[test_idx[0]]
            test_ts_end   = ts_arr[test_idx[-1]]

            # Purge: remove training samples whose LABEL overlaps test window
            # A sample at time t has label window [t, t + label_horizon_s]
            # Overlap if t + label_horizon_s > test_ts_start
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start_idx:test_end_idx] = False

            # Embargo: remove samples within embargo_s after test end
            embargo_end_ts = test_ts_end + self.embargo_s
            for i in indices:
                if ts_arr[i] < test_ts_start:
                    # Before test: purge if label window bleeds into test
                    if ts_arr[i] + self.label_horizon_s > test_ts_start:
                        train_mask[i] = False
                elif ts_arr[i] <= embargo_end_ts:
                    # In embargo zone
                    train_mask[i] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits
