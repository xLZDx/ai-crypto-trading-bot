"""
PurgedKFold for financial time-series cross-validation (AFML Ch. 7, Lopez de Prado).

Two leakage controls:
  1. Embargo: drop a window of bars immediately before the test set so any
     train sample whose feature window touches the test region is excluded.
  2. t1-span purge: drop training samples whose label-end timestamp `t1` falls
     INSIDE the test window. Without this, a training row entered at time T
     whose Triple Barrier resolves at T' > test_start carries forward-looking
     information into the model.

Both are required. The previous implementation accepted `t1` in `__init__` but
never read it in `split()` (BUG-2). This file now implements true PurgedKFold.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Optional

import numpy as np
import pandas as pd


class PurgedKFold:
    """
    Walk-forward k-fold that purges training data around test boundaries.

    Each fold uses ONLY data that precedes the test window. Fold 0 is always
    skipped because it has no prior observations.

    Parameters:
        n_splits: Number of folds (fold 0 is skipped).
        t1: Series of label resolution timestamps (output of triple_barrier
            second return). Used for AFML-style label-span purging in split().
            Must align with X by position.
        pct_embargo: Fraction of n_samples to embargo before the test window.
            0.0 means no embargo. NO minimum is enforced — set explicitly.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.0,
    ) -> None:
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate walk-forward train/test index pairs with purging + embargo.

        For fold k:
          test  = [k * fold_size, (k+1) * fold_size)
          train = [0, test_start - embargo_size)
                  purged of any sample whose t1 >= test_window_start

        Yields:
            (train_indices, test_indices): numpy arrays of integer positions.
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        # BUG-N: previous max(1, ...) silently forced minimum 1-bar embargo
        # even when pct_embargo=0. Honour the caller's intent exactly.
        embargo_size = int(n_samples * self.pct_embargo)

        # Resolve X.index for t1 comparison if X is a DataFrame.
        x_index = X.index if hasattr(X, 'index') else None

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (
                (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            )
            test_idx = np.arange(test_start, test_end)

            # Walk-forward: train only on strictly past data (before embargo zone).
            train_end = max(0, test_start - embargo_size)
            train_idx = np.arange(0, train_end)

            if len(train_idx) == 0:
                # Fold 0 always produces an empty train set; skip it.
                continue

            # BUG-2 fix: apply AFML t1-span purging.
            # Remove training samples whose label-end time `t1[i]` overlaps
            # the test window starting at position `test_start`.
            if self.t1 is not None and x_index is not None:
                try:
                    test_window_start_ts = x_index[test_start]
                    # Materialize t1 values for train_idx positions
                    if isinstance(self.t1, pd.Series):
                        t1_vals = self.t1.iloc[train_idx].values
                    else:
                        t1_vals = np.asarray(self.t1)[train_idx]
                    # Convert to comparable dtypes
                    t1_vals = pd.to_datetime(t1_vals, errors='coerce')
                    test_window_start_ts = pd.to_datetime(test_window_start_ts)
                    # Keep only train samples whose label resolved BEFORE test_start
                    keep = t1_vals < test_window_start_ts
                    # NaT entries: keep=False (be conservative — drop the row)
                    keep = keep & ~pd.isna(t1_vals)
                    train_idx = train_idx[keep.values if hasattr(keep, 'values') else keep]
                except Exception:
                    # If purging is impossible (dtype mismatch, missing index),
                    # log and continue without purging — but the AFML guarantee
                    # is then weaker. Embargo still applies.
                    import logging
                    logging.getLogger(__name__).warning(
                        "PurgedKFold: could not apply t1-span purging at fold=%d; "
                        "embargo-only train set returned.", fold,
                    )

            if len(train_idx) == 0:
                continue

            yield train_idx, test_idx
