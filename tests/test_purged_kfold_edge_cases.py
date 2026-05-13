"""
Edge-case behavioral tests for PurgedKFold (AFML Ch. 7 cross-validator).

All tests:
  - Import and call PurgedKFold directly.
  - Assert on observable behavior: yielded indices, lengths, set membership.
  - Use no string-match assertions as sole coverage.
  - Are fully independent (no shared mutable state).

Scenarios covered:
  EC-01  n_splits=2 minimum — exactly 1 fold yielded
  EC-02  n_splits=10 on n=1000 — fold sizes exactly 100 each
  EC-03  Empty DataFrame (len=0) — yields nothing, no exception
  EC-04  X smaller than n_splits — all folds skipped (fold_size=0 edge)
  EC-05  t1=None + pct_embargo=0 pure walk-forward, train=[0, test_start)
  EC-06  t1=None + pct_embargo=0.05 on n=1000, fold-1 train=[0,150)
  EC-07  t1 series shorter than X — folds where t1 covers train are purged,
         folds where t1 is out-of-range fall back to embargo-only (no crash)
  EC-08  t1 series with NaT values — NaT rows excluded from train
  EC-09  All t1 values BEFORE every test window — purging removes nothing
  EC-10  All t1 values AFTER every test window — all train purged, no folds
  EC-11  tz-aware t1 vs tz-naive X.index — pd.to_datetime strips tz, purging runs
  EC-12  X as numpy array (not DataFrame) — t1 purging skipped, sizes match no-t1
  EC-13  pct_embargo=1.0 — every train_end=0, no folds yielded
  EC-14  Test windows disjoint and cover [fold_size, n) exactly once
  EC-15  Train indices are strictly before test window start (no data leakage)
  EC-16  Last fold absorbs remainder (n not divisible by n_splits)
  EC-17  pct_embargo=0 + large data (n=10000) — all 9 folds 1000 bars each
  EC-18  n_splits=1 — fold_size=n, fold=0: train=[0,0)=empty -> no folds yielded
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.purged_kfold import PurgedKFold


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_df(n: int, freq: str = "1h") -> pd.DataFrame:
    """Return a DataFrame with DatetimeIndex of length n."""
    ts = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame({"feat": np.arange(n, dtype=float)}, index=ts)


def _make_t1(df: pd.DataFrame, offset_hours: int) -> pd.Series:
    """Return t1 = df.index + offset_hours, indexed like df."""
    return pd.Series(
        df.index + pd.Timedelta(hours=offset_hours), index=df.index
    )


# ─── EC-01: n_splits=2 minimum ───────────────────────────────────────────────

def test_ec01_n_splits_2_yields_exactly_one_fold():
    """
    With n_splits=2: fold 0 has empty train -> skip; fold 1 yields.
    Total folds = n_splits - 1 = 1.
    """
    X = _make_df(10)
    cv = PurgedKFold(n_splits=2, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))

    assert len(folds) == 1, f"Expected 1 fold, got {len(folds)}"


def test_ec01_n_splits_2_fold_covers_second_half():
    """
    The single yielded fold must have train=[0,5) and test=[5,10).
    """
    X = _make_df(10)
    cv = PurgedKFold(n_splits=2, t1=None, pct_embargo=0.0)
    (train_idx, test_idx) = list(cv.split(X))[0]

    assert list(train_idx) == list(range(5))
    assert list(test_idx) == list(range(5, 10))


# ─── EC-02: n_splits=10 on n=1000 ───────────────────────────────────────────

def test_ec02_fold_size_100_each():
    """Every test fold must be exactly 100 bars when n=1000, n_splits=10."""
    X = _make_df(1000)
    cv = PurgedKFold(n_splits=10, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))

    test_sizes = [len(te) for _, te in folds]
    assert all(s == 100 for s in test_sizes), f"Unequal fold sizes: {test_sizes}"


def test_ec02_9_folds_yielded():
    """Fold 0 is skipped; 9 of 10 folds yield data."""
    X = _make_df(1000)
    cv = PurgedKFold(n_splits=10, t1=None, pct_embargo=0.0)
    assert len(list(cv.split(X))) == 9


def test_ec02_train_grows_by_fold_size():
    """Train set for fold k must be k*100 bars (embargo=0, no t1)."""
    X = _make_df(1000)
    cv = PurgedKFold(n_splits=10, t1=None, pct_embargo=0.0)
    train_sizes = [len(tr) for tr, _ in cv.split(X)]
    expected = [k * 100 for k in range(1, 10)]
    assert train_sizes == expected, f"Train sizes {train_sizes} != expected {expected}"


# ─── EC-03: Empty DataFrame ───────────────────────────────────────────────────

def test_ec03_empty_dataframe_yields_nothing():
    """len(X)=0 must produce no folds and must not raise."""
    X = pd.DataFrame({"feat": []})
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))
    assert folds == []


def test_ec03_empty_dataframe_does_not_raise():
    """Calling split() on an empty DataFrame must complete without exception."""
    X = pd.DataFrame({"feat": []})
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    try:
        list(cv.split(X))
    except Exception as exc:
        pytest.fail(f"Unexpected exception on empty DataFrame: {exc!r}")


# ─── EC-04: X smaller than n_splits ─────────────────────────────────────────

def test_ec04_x_smaller_than_n_splits_all_folds_skipped():
    """
    n=3, n_splits=5 -> fold_size = 3//5 = 0.
    Every test_start=0, every train_end=0 -> no folds yielded.
    """
    X = _make_df(3)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))
    assert folds == [], f"Expected no folds, got {len(folds)}"


def test_ec04_x_exactly_n_splits_fold0_only_skipped():
    """n = n_splits = 5 -> fold_size=1; fold 0 has empty train, folds 1-4 yield."""
    X = _make_df(5)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))
    assert len(folds) == 4, f"Expected 4 folds, got {len(folds)}"


# ─── EC-05: t1=None + pct_embargo=0 pure walk-forward ───────────────────────

def test_ec05_pure_walk_forward_train_starts_at_zero():
    """Every train_idx[0] must be 0 (no data before start excluded)."""
    X = _make_df(100)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    for train_idx, _ in cv.split(X):
        assert train_idx[0] == 0, "train must start from index 0"


def test_ec05_pure_walk_forward_train_ends_at_test_start():
    """
    With embargo=0 and no t1, train_idx[-1] + 1 == test_idx[0].
    Train immediately precedes the test window.
    """
    X = _make_df(100)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    for train_idx, test_idx in cv.split(X):
        assert train_idx[-1] + 1 == test_idx[0], (
            f"Gap between train end {train_idx[-1]} and test start {test_idx[0]}"
        )


# ─── EC-06: t1=None + pct_embargo=0.05 on n=1000 ────────────────────────────

def test_ec06_embargo_size_50():
    """
    n=1000, pct_embargo=0.05 -> embargo_size = int(1000*0.05) = 50.
    Fold 1 (first yielded): test=[200,400), train=[0, 200-50) = [0,150).
    """
    X = _make_df(1000)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.05)
    folds = list(cv.split(X))

    # First yielded fold = fold-index 1 (fold 0 is skipped)
    train_idx, test_idx = folds[0]

    assert test_idx[0] == 200, f"Test start: {test_idx[0]}"
    assert len(train_idx) == 150, f"Train size: {len(train_idx)}"
    assert train_idx[0] == 0
    assert train_idx[-1] == 149


def test_ec06_embargo_creates_gap():
    """Gap between train_end and test_start must equal embargo_size."""
    X = _make_df(1000)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.05)
    embargo_size = int(1000 * 0.05)  # 50

    for train_idx, test_idx in cv.split(X):
        gap = test_idx[0] - (train_idx[-1] + 1)
        assert gap == embargo_size, f"Gap {gap} != embargo_size {embargo_size}"


# ─── EC-07: t1 series shorter than X ────────────────────────────────────────

def test_ec07_t1_shorter_no_crash():
    """t1 with 50 rows on n=100 must not raise; split() must complete."""
    X = _make_df(100)
    t1_short = pd.Series(
        X.index[:50] + pd.Timedelta(hours=5), index=X.index[:50]
    )
    cv = PurgedKFold(n_splits=5, t1=t1_short, pct_embargo=0.0)
    try:
        list(cv.split(X))
    except Exception as exc:
        pytest.fail(f"Unexpected exception with short t1: {exc!r}")


def test_ec07_t1_shorter_purging_applied_where_t1_covers_train():
    """
    Fold 1 (test=[20,40), train=[0,20)): all 20 train rows are within t1's 50
    rows -> purging runs. t1[i] = X.index[i] + 5h; test_window_start = X.index[20].
    Rows where t1 >= test_window_start are excluded.
    Expected: rows 15..19 purged (t1[i]+5h >= ts[20]), rows 0..14 kept = 15 rows.
    """
    X = _make_df(100)
    t1_short = pd.Series(
        X.index[:50] + pd.Timedelta(hours=5), index=X.index[:50]
    )
    cv = PurgedKFold(n_splits=5, t1=t1_short, pct_embargo=0.0)
    folds = list(cv.split(X))

    # First yielded fold = fold-index 1 (fold 0 skipped)
    train_idx, test_idx = folds[0]
    assert test_idx[0] == 20
    assert len(train_idx) == 15, (
        f"Expected 15 (rows 15..19 purged), got {len(train_idx)}"
    )


def test_ec07_t1_shorter_out_of_range_folds_fall_back_to_embargo_only():
    """
    Fold 3 (test=[60,80), train=[0,60)): t1.iloc[0..59] is out of range for
    a 50-row t1 -> IndexError caught -> embargo-only fallback -> train size = 60.
    """
    X = _make_df(100)
    t1_short = pd.Series(
        X.index[:50] + pd.Timedelta(hours=5), index=X.index[:50]
    )
    cv = PurgedKFold(n_splits=5, t1=t1_short, pct_embargo=0.0)
    folds = list(cv.split(X))

    # folds[2] = fold-index 3 (fold 0 skipped, folds 1,2,3,4 -> indices 0,1,2,3)
    train_idx, test_idx = folds[2]
    assert test_idx[0] == 60
    assert len(train_idx) == 60, (
        f"Expected 60 (embargo-only fallback), got {len(train_idx)}"
    )


# ─── EC-08: t1 with NaT values ───────────────────────────────────────────────

def test_ec08_nat_rows_excluded_from_train():
    """
    NaT entries in t1 must be excluded from train_idx (conservative drop).
    Rows 0, 1, 2 have NaT in t1; rest have t1 = ts + 1h.
    Fold 1 (test=[20,40), train=[0,20)): row 19 also purged (t1[19]==test_start).
    Remaining: rows 3..18 = 16 rows.
    """
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame({"feat": np.arange(n, dtype=float)}, index=ts)

    t1_vals = list(ts + pd.Timedelta(hours=1))
    t1_vals[0] = pd.NaT
    t1_vals[1] = pd.NaT
    t1_vals[2] = pd.NaT
    t1 = pd.Series(t1_vals, index=ts)

    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
    folds = list(cv.split(X))

    # First yielded fold
    train_idx, test_idx = folds[0]
    assert test_idx[0] == 20

    # Rows 0,1,2 (NaT) and row 19 (t1==test_start) purged -> 16 remain
    assert len(train_idx) == 16, (
        f"Expected 16 train rows (3 NaT + 1 boundary purged), got {len(train_idx)}"
    )
    # NaT rows must not appear in train
    assert 0 not in train_idx
    assert 1 not in train_idx
    assert 2 not in train_idx


def test_ec08_all_nat_t1_produces_no_train():
    """
    If every t1 value is NaT, the keep mask is all-False -> train_idx empty
    -> fold is skipped -> no folds yielded at all.
    """
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame({"feat": np.arange(n, dtype=float)}, index=ts)
    t1_all_nat = pd.Series([pd.NaT] * n, index=ts, dtype="datetime64[ns]")

    cv = PurgedKFold(n_splits=5, t1=t1_all_nat, pct_embargo=0.0)
    folds = list(cv.split(X))
    assert folds == [], f"Expected 0 folds, got {len(folds)}"


# ─── EC-09: All t1 BEFORE every test window ──────────────────────────────────

def test_ec09_t1_all_before_test_windows_purges_nothing():
    """
    t1 = X.index - 1 day (far in the past). Purging removes nothing.
    Train sizes must equal the no-t1 case.
    """
    X = _make_df(100)
    t1_past = pd.Series(X.index - pd.Timedelta(days=1), index=X.index)

    cv_purged = PurgedKFold(n_splits=5, t1=t1_past, pct_embargo=0.0)
    cv_clean  = PurgedKFold(n_splits=5, t1=None,    pct_embargo=0.0)

    sizes_purged = [len(tr) for tr, _ in cv_purged.split(X)]
    sizes_clean  = [len(tr) for tr, _ in cv_clean.split(X)]

    assert sizes_purged == sizes_clean, (
        f"Past-t1 purged sizes {sizes_purged} != clean sizes {sizes_clean}"
    )


def test_ec09_t1_all_before_test_same_fold_count():
    """Fold count must be unchanged when t1 never overlaps any test window."""
    X = _make_df(100)
    t1_past = pd.Series(X.index - pd.Timedelta(days=1), index=X.index)

    cv_purged = PurgedKFold(n_splits=5, t1=t1_past, pct_embargo=0.0)
    cv_clean  = PurgedKFold(n_splits=5, t1=None,    pct_embargo=0.0)

    assert len(list(cv_purged.split(X))) == len(list(cv_clean.split(X)))


# ─── EC-10: All t1 AFTER every test window ───────────────────────────────────

def test_ec10_t1_all_future_purges_all_train():
    """
    t1 = X.index + 365 days (far in the future, past all test windows).
    Every train sample is purged -> all folds skipped -> 0 folds yielded.
    """
    X = _make_df(100)
    t1_future = pd.Series(X.index + pd.Timedelta(days=365), index=X.index)

    cv = PurgedKFold(n_splits=5, t1=t1_future, pct_embargo=0.0)
    folds = list(cv.split(X))
    assert folds == [], f"Expected 0 folds when all t1 future, got {len(folds)}"


# ─── EC-11: tz-aware t1 vs tz-naive X.index ──────────────────────────────────

def test_ec11_tz_aware_t1_does_not_crash():
    """
    tz-aware t1 vs tz-naive X.index: pd.to_datetime() strips tz from numpy
    datetime64 values, so the comparison runs without raising. No crash expected.
    """
    n = 100
    ts_naive = pd.date_range("2024-01-01", periods=n, freq="1h")
    ts_aware = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    X = pd.DataFrame({"feat": np.arange(n, dtype=float)}, index=ts_naive)
    t1_aware = pd.Series(ts_aware + pd.Timedelta(hours=5), index=ts_naive)

    cv = PurgedKFold(n_splits=5, t1=t1_aware, pct_embargo=0.0)
    try:
        folds = list(cv.split(X))
    except Exception as exc:
        pytest.fail(f"tz-aware t1 raised unexpectedly: {exc!r}")

    assert len(folds) > 0, "Expected at least 1 fold"


def test_ec11_tz_aware_t1_purging_still_applied():
    """
    After tz strip, purging still runs. With t1=ts+5h (UTC stripped to naive),
    the effective comparison is ts[i]+5h < test_window_start_naive.
    Fold 1 (test=[20,40), train=[0,20)):
      rows where ts[i]+5h >= ts[20] are excluded -> rows 15..19 purged -> 15 kept.
    """
    n = 100
    ts_naive = pd.date_range("2024-01-01", periods=n, freq="1h")
    ts_aware = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    X = pd.DataFrame({"feat": np.arange(n, dtype=float)}, index=ts_naive)
    t1_aware = pd.Series(ts_aware + pd.Timedelta(hours=5), index=ts_naive)

    cv = PurgedKFold(n_splits=5, t1=t1_aware, pct_embargo=0.0)
    folds = list(cv.split(X))

    train_idx, test_idx = folds[0]
    assert test_idx[0] == 20
    # With tz stripped: t1_effective[i] = ts[i]+5h (naive)
    # t1[14] = 19:00 < 20:00 -> kept; t1[15] = 20:00 NOT < 20:00 -> purged
    assert len(train_idx) == 15, (
        f"Expected 15 after tz-stripped purging, got {len(train_idx)}"
    )


# ─── EC-12: X as numpy array ─────────────────────────────────────────────────

def test_ec12_numpy_x_does_not_crash():
    """PurgedKFold must accept a numpy ndarray for X without raising."""
    X_np = np.arange(100, dtype=float).reshape(100, 1)
    t1 = pd.Series(pd.date_range("2024-01-01", periods=100, freq="1h"))
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
    try:
        list(cv.split(X_np))
    except Exception as exc:
        pytest.fail(f"numpy X raised unexpectedly: {exc!r}")


def test_ec12_numpy_x_t1_purging_skipped():
    """
    When X has no .index, x_index is None and t1 purging is skipped.
    Train sizes must equal the no-t1 case.
    """
    X_np = np.arange(100, dtype=float).reshape(100, 1)
    t1 = pd.Series(pd.date_range("2024-01-01", periods=100, freq="1h"))

    cv_with_t1 = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
    cv_no_t1   = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)

    sizes_with = [len(tr) for tr, _ in cv_with_t1.split(X_np)]
    sizes_none = [len(tr) for tr, _ in cv_no_t1.split(X_np)]

    assert sizes_with == sizes_none, (
        f"numpy X should skip t1 purging; {sizes_with} != {sizes_none}"
    )


def test_ec12_numpy_x_correct_fold_count():
    """numpy X with n_splits=5 yields 4 folds (fold 0 skipped)."""
    X_np = np.arange(100, dtype=float).reshape(100, 1)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X_np))
    assert len(folds) == 4


# ─── EC-13: pct_embargo=1.0 ──────────────────────────────────────────────────

def test_ec13_pct_embargo_1_yields_nothing():
    """
    embargo_size = n, so train_end = max(0, test_start - n) = 0 for every fold.
    Every train is empty -> every fold skipped -> 0 folds.
    """
    X = _make_df(100)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=1.0)
    folds = list(cv.split(X))
    assert folds == [], f"Expected 0 folds with pct_embargo=1.0, got {len(folds)}"


def test_ec13_pct_embargo_1_does_not_raise():
    """pct_embargo=1.0 must complete without exception."""
    X = _make_df(100)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=1.0)
    try:
        list(cv.split(X))
    except Exception as exc:
        pytest.fail(f"pct_embargo=1.0 raised: {exc!r}")


# ─── EC-14: Test windows disjoint and cover [fold_size, n) ───────────────────

def test_ec14_test_windows_disjoint():
    """No index appears in more than one test window."""
    X = _make_df(1000)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    all_test = [i for _, te in cv.split(X) for i in te.tolist()]
    assert len(all_test) == len(set(all_test)), "Duplicate indices across test windows"


def test_ec14_test_windows_cover_non_zero_folds():
    """
    Union of all test indices covers exactly [fold_size, n).
    Fold 0 is skipped so its window [0, fold_size) is never a test set.
    """
    n = 1000
    n_splits = 5
    fold_size = n // n_splits  # 200
    X = _make_df(n)
    cv = PurgedKFold(n_splits=n_splits, t1=None, pct_embargo=0.0)
    all_test = set(i for _, te in cv.split(X) for i in te.tolist())
    expected = set(range(fold_size, n))
    assert all_test == expected, f"Test coverage {len(all_test)} != expected {len(expected)}"


# ─── EC-15: Train indices strictly before test window ────────────────────────

def test_ec15_no_train_index_overlaps_test():
    """For every fold, train_idx and test_idx must be disjoint."""
    X = _make_df(500)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X)):
        overlap = set(train_idx.tolist()) & set(test_idx.tolist())
        assert not overlap, (
            f"Fold {fold_num}: overlap between train and test: {overlap}"
        )


def test_ec15_train_indices_precede_test_start():
    """max(train_idx) < test_idx[0] for every fold."""
    X = _make_df(500)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.02)
    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X)):
        assert int(train_idx.max()) < int(test_idx[0]), (
            f"Fold {fold_num}: train extends past test_start"
        )


# ─── EC-16: Last fold absorbs remainder ──────────────────────────────────────

def test_ec16_last_fold_test_extends_to_n():
    """When n is not divisible by n_splits, the last fold test runs to n."""
    n = 103  # not divisible by 5
    X = _make_df(n)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))
    last_test = folds[-1][1]
    assert int(last_test[-1]) == n - 1, (
        f"Last test index {int(last_test[-1])} != {n - 1}"
    )


def test_ec16_all_but_first_fold_windows_covered():
    """
    Union of all test indices for n=103, n_splits=5 must cover
    [fold_size, 103) = [20, 103).
    """
    n = 103
    fold_size = n // 5  # 20
    X = _make_df(n)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    all_test = set(i for _, te in cv.split(X) for i in te.tolist())
    expected = set(range(fold_size, n))
    assert all_test == expected


# ─── EC-17: Large data performance sanity ────────────────────────────────────

def test_ec17_large_data_10k_fold_sizes():
    """n=10000, n_splits=10: each test fold = 1000 bars, 9 folds total."""
    X = _make_df(10000, freq="1min")
    cv = PurgedKFold(n_splits=10, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))

    assert len(folds) == 9
    assert all(len(te) == 1000 for _, te in folds)


def test_ec17_large_data_train_sizes_correct():
    """Train sizes for n=10000 must be [1000, 2000, ..., 9000]."""
    X = _make_df(10000, freq="1min")
    cv = PurgedKFold(n_splits=10, t1=None, pct_embargo=0.0)
    train_sizes = [len(tr) for tr, _ in cv.split(X)]
    expected = [k * 1000 for k in range(1, 10)]
    assert train_sizes == expected


# ─── EC-18: n_splits=1 ───────────────────────────────────────────────────────

def test_ec18_n_splits_1_yields_nothing():
    """
    n_splits=1: only fold 0 exists. fold_size=n, test_start=0, train_end=0
    -> train_idx empty -> fold skipped -> 0 folds.
    """
    X = _make_df(100)
    cv = PurgedKFold(n_splits=1, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))
    assert folds == [], f"Expected 0 folds for n_splits=1, got {len(folds)}"


def test_ec18_n_splits_1_does_not_raise():
    """n_splits=1 must complete without exception."""
    X = _make_df(100)
    cv = PurgedKFold(n_splits=1, t1=None, pct_embargo=0.0)
    try:
        list(cv.split(X))
    except Exception as exc:
        pytest.fail(f"n_splits=1 raised: {exc!r}")


# ─── additional behavioral correctness checks ─────────────────────────────────

def test_yielded_indices_are_numpy_arrays():
    """split() must yield numpy arrays, not lists or pandas objects."""
    X = _make_df(100)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    for train_idx, test_idx in cv.split(X):
        assert isinstance(train_idx, np.ndarray), type(train_idx)
        assert isinstance(test_idx, np.ndarray), type(test_idx)


def test_indices_are_integer_dtype():
    """Yielded index arrays must have integer dtype (usable for iloc/indexing)."""
    X = _make_df(100)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    for train_idx, test_idx in cv.split(X):
        assert np.issubdtype(train_idx.dtype, np.integer), train_idx.dtype
        assert np.issubdtype(test_idx.dtype, np.integer), test_idx.dtype


def test_pct_embargo_zero_integer_truncation():
    """
    embargo_size = int(n * 0.0) = 0.  Confirm the same as passing pct_embargo=0
    — no accidental floor that forces minimum 1-bar embargo.
    """
    n = 100
    X = _make_df(n)
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))
    # Fold 1 (first yielded): test=[20,40), train=[0,20)
    train_idx, test_idx = folds[0]
    assert test_idx[0] == 20
    assert train_idx[-1] == 19  # no gap at all


def test_t1_purging_boundary_exclusive():
    """
    t1[i] == test_window_start must be treated as NOT strictly less than ->
    the row is EXCLUDED. This ensures the boundary is conservatively purged.
    """
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame({"feat": np.arange(n, dtype=float)}, index=ts)

    # t1[19] == ts[20] exactly (the test window start for fold 1)
    t1_vals = list(ts - pd.Timedelta(hours=1))  # all well before test
    t1_vals[19] = ts[20]  # boundary row: should be purged
    t1 = pd.Series(t1_vals, index=ts)

    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
    folds = list(cv.split(X))

    train_idx, test_idx = folds[0]
    assert 19 not in train_idx, "Row 19 (t1 == test_start) must be purged"
    assert 18 in train_idx, "Row 18 (t1 < test_start) must be kept"


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
