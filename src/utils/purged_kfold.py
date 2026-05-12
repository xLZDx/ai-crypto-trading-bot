"""
Lightweight PurgedKFold for financial time-series cross-validation.
Avoids lookahead bias by removing samples around test set boundaries.
"""
import numpy as np


class PurgedKFold:
    """
    Walk-forward k-fold that purges training data around test boundaries.

    Each fold uses ONLY data that precedes the test window (no future data).
    Fold 0 is always skipped because it has zero prior observations.

    Parameters:
        n_splits: Number of folds (the first fold is skipped, so effective
                  walk-forward folds = n_splits - 1 or more depending on embargo).
        t1: Series of closing timestamps (reserved for future t1-aware purging).
        pct_embargo: Fraction of data to embargo before the test window start.
    """

    def __init__(self, n_splits=5, t1=None, pct_embargo=0.0):
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Generate walk-forward train/test index pairs.

        For fold k:
          test  = [k * fold_size, (k+1) * fold_size)
          train = [0, test_start - embargo_size)

        Folds where train is empty are silently skipped (always fold 0,
        and any fold where embargo_size >= test_start).

        Yields:
            (train_indices, test_indices): Both numpy arrays of integer positions.
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = max(1, int(n_samples * self.pct_embargo))

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (
                (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            )
            test_idx = np.arange(test_start, test_end)

            # Walk-forward: train only on strictly past data (before embargo zone)
            train_end = max(0, test_start - embargo_size)
            train_idx = np.arange(0, train_end)

            if len(train_idx) == 0:
                # Fold 0 always produces an empty train set; skip it.
                continue

            yield train_idx, test_idx
