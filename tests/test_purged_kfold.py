"""
Tests for Phase B1 — PurgedKFold walk-forward rewrite.

Anti-leakage guarantee: for every fold yielded by split(), no test index
appears in that fold's train set, and train indices are strictly < test indices.
"""
import numpy as np
import pytest

from src.utils.purged_kfold import PurgedKFold


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_no_overlap(train_idx, test_idx):
    overlap = set(train_idx) & set(test_idx)
    assert not overlap, f"Train/test overlap: {overlap}"


def _check_no_lookahead(train_idx, test_idx):
    """All train indices must be strictly less than all test indices."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    assert train_idx.max() < test_idx.min(), (
        f"Lookahead: max(train)={train_idx.max()} >= min(test)={test_idx.min()}"
    )


# ---------------------------------------------------------------------------
# Basic walk-forward guarantee
# ---------------------------------------------------------------------------

class TestWalkForwardGuarantee:
    def test_no_overlap_any_fold(self):
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.0)
        for train_idx, test_idx in cv.split(X):
            _check_no_overlap(train_idx, test_idx)

    def test_no_lookahead_any_fold(self):
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.0)
        for train_idx, test_idx in cv.split(X):
            _check_no_lookahead(train_idx, test_idx)

    def test_no_overlap_with_embargo(self):
        X = np.arange(200).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.05)
        for train_idx, test_idx in cv.split(X):
            _check_no_overlap(train_idx, test_idx)

    def test_no_lookahead_with_embargo(self):
        X = np.arange(200).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.05)
        for train_idx, test_idx in cv.split(X):
            _check_no_lookahead(train_idx, test_idx)

    def test_train_only_past_data(self):
        """Train window must end before test window starts (minus embargo)."""
        n = 300
        X = np.arange(n).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.02)
        fold_size = n // 5
        embargo_size = max(1, int(n * 0.02))

        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
            # Train must end at or before test_start - embargo
            expected_train_end = test_idx.min() - embargo_size
            assert train_idx.max() < expected_train_end + 1, (
                f"fold {fold_i}: train goes too far: max={train_idx.max()} "
                f"expected < {expected_train_end + 1}"
            )


# ---------------------------------------------------------------------------
# Fold-0 skip guarantee
# ---------------------------------------------------------------------------

class TestFold0Skipped:
    def test_fold0_skipped_pct0(self):
        """With pct_embargo=0 fold-0 train is empty → must be skipped."""
        X = np.arange(50).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.0)
        folds = list(cv.split(X))
        # Fold 0 would have test_start=0 → train_end=max(0,0-1)=0 → empty → skipped
        # So we get at most n_splits-1 folds yielded
        assert len(folds) <= cv.n_splits - 1

    def test_first_yielded_fold_has_nonempty_train(self):
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.0)
        first_train, _ = next(iter(cv.split(X)))
        assert len(first_train) > 0

    def test_yields_at_least_one_fold(self):
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.0)
        folds = list(cv.split(X))
        assert len(folds) >= 1


# ---------------------------------------------------------------------------
# Embargo correctness
# ---------------------------------------------------------------------------

class TestEmbargoCorrectness:
    def test_embargo_reduces_train_end(self):
        """Higher embargo shrinks the available train window."""
        X = np.arange(500).reshape(-1, 1)
        cv_no_emb = PurgedKFold(n_splits=5, pct_embargo=0.0)
        cv_emb    = PurgedKFold(n_splits=5, pct_embargo=0.10)

        folds_no  = list(cv_no_emb.split(X))
        folds_emb = list(cv_emb.split(X))

        # The fold with the same test set should have a shorter train set
        # when embargo is active. Compare the last common fold by test_idx.
        for (tr_no, te_no), (tr_emb, te_emb) in zip(folds_no, folds_emb):
            if np.array_equal(te_no, te_emb):
                assert len(tr_emb) <= len(tr_no), (
                    "Embargo should reduce or equal train length"
                )

    def test_zero_embargo_max_train_coverage(self):
        """With pct_embargo=0, train spans exactly [0, test_start-1]."""
        n = 100
        X = np.arange(n).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.0)
        fold_size = n // 5
        embargo_size = max(1, int(n * 0.0))  # = 1 (min clamp)

        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
            test_start = test_idx.min()
            expected_end = max(0, test_start - embargo_size)
            assert list(train_idx) == list(np.arange(0, expected_end)), (
                f"fold {fold_i}: expected train=[0,{expected_end}) "
                f"got train=[0,{train_idx.max()+1 if len(train_idx) else 0})"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_small_dataset(self):
        """Should not crash on tiny datasets; may yield 0 folds."""
        X = np.arange(10).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.05)
        folds = list(cv.split(X))
        # Just ensure no crash and invariants hold
        for train_idx, test_idx in folds:
            _check_no_overlap(train_idx, test_idx)
            _check_no_lookahead(train_idx, test_idx)

    def test_large_embargo_shrinks_folds(self):
        """Large embargo may cause many folds to be skipped — no crash."""
        X = np.arange(50).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.50)
        folds = list(cv.split(X))
        for train_idx, test_idx in folds:
            _check_no_overlap(train_idx, test_idx)
            _check_no_lookahead(train_idx, test_idx)

    def test_n_splits_1(self):
        """Single-split: fold 0 skipped → zero folds yielded (only fold is fold-0)."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=1, pct_embargo=0.0)
        folds = list(cv.split(X))
        assert folds == [], "n_splits=1 always skips fold-0 → no folds"

    def test_indices_are_numpy_arrays(self):
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.0)
        for train_idx, test_idx in cv.split(X):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_split_accepts_pandas_dataframe(self):
        """split() accepts DataFrame X (uses len(X))."""
        import pandas as pd
        X = pd.DataFrame({'a': range(100)})
        cv = PurgedKFold(n_splits=5, pct_embargo=0.0)
        folds = list(cv.split(X))
        assert len(folds) > 0
        for train_idx, test_idx in folds:
            _check_no_overlap(train_idx, test_idx)
