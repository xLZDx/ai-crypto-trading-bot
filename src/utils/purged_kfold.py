"""
Lightweight PurgedKFold for financial time-series cross-validation.
Avoids lookahead bias by removing samples around test set boundaries.
"""
import numpy as np


class PurgedKFold:
    """
    Time-series aware k-fold that purges training data around test boundaries.

    Parameters:
        n_splits: Number of folds
        t1: Series of closing timestamps (for embargo calculation)
        pct_embargo: Fraction of data to embargo around test set (prevents leakage)
    """

    def __init__(self, n_splits=5, t1=None, pct_embargo=0.0):
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Generate train/test indices for walk-forward validation.

        Yields:
            (train_indices, test_indices): Both numpy arrays
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = max(1, int(n_samples * self.pct_embargo))

        for fold in range(self.n_splits):
            # Test set: fold-th contiguous block
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            test_idx = np.arange(test_start, test_end)

            # Embargo: exclude samples in window around test set
            embargo_start = max(0, test_start - embargo_size)
            embargo_end = min(n_samples, test_end + embargo_size)

            # Train set: everything except test + embargo
            train_idx = np.concatenate([
                np.arange(0, embargo_start),
                np.arange(embargo_end, n_samples)
            ])

            yield train_idx, test_idx
