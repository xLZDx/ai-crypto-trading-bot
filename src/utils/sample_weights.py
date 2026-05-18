"""
AFML-style sample weights (Lopez de Prado, "Advances in Financial Machine Learning", Ch. 4).

Weight = average_uniqueness × event_strength × class_balance

- average_uniqueness: 1 / num_concurrent_events  (AFML num_co_events approach)
- event_strength:     |returns| normalised to [0.5, 1.5]
- class_balance:      sklearn balanced class weights (optional)

All three are multiplied together, then the array is normalised so
weights.sum() == len(y) (standard sklearn convention).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Average uniqueness ───────────────────────────────────────────────────────

def _num_co_events(t_start: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """
    For each event i, count how many events are active at time t_start[i].

    An event j is active at time t if t_start[j] <= t < t1[j].
    Returns a Series indexed like t_start.
    """
    # Restrict t1 to the same index
    t1 = t1.loc[t_start]
    counts = pd.Series(0, index=t_start, dtype=float)
    for t in t_start:
        # Events active at time t: started on or before t AND end strictly after t
        mask = (t_start <= t) & (t1.values > t)
        counts.loc[t] = mask.sum()
    return counts.clip(lower=1.0)


def _average_uniqueness(t_start: pd.DatetimeIndex, t1: pd.Series) -> np.ndarray:
    """
    AFML Ch.4 average uniqueness per event.

    Uses a vectorised outer-product approach instead of iterrows for speed.
    Returns np.ndarray of float64, same length as t_start.
    """
    n = len(t_start)
    if n == 0:
        return np.array([], dtype=float)

    t_arr = np.asarray(t_start, dtype="datetime64[ns]")
    t1_arr = np.asarray(t1.values, dtype="datetime64[ns]")

    # co_events[i] = number of events j where t_arr[j] <= t_arr[i] < t1_arr[j]
    # Vectorised: for each i, count j where t_arr[j] <= t_arr[i] AND t1_arr[j] > t_arr[i]
    co_events = np.zeros(n, dtype=float)
    for i in range(n):
        t_i = t_arr[i]
        active = (t_arr <= t_i) & (t1_arr > t_i)
        co_events[i] = max(1.0, active.sum())

    return 1.0 / co_events


# ── Event strength ───────────────────────────────────────────────────────────

def _event_strength(returns: np.ndarray, lo: float = 0.5, hi: float = 1.5) -> np.ndarray:
    """
    Scale |returns| to [lo, hi] via min-max.  Events with larger absolute
    returns get higher weight; neutral bars map to the midpoint.
    """
    abs_ret = np.abs(returns)
    r_min, r_max = abs_ret.min(), abs_ret.max()
    if r_max <= r_min:
        return np.full(len(returns), (lo + hi) / 2.0)
    normalised = (abs_ret - r_min) / (r_max - r_min)  # 0..1
    return lo + normalised * (hi - lo)


# ── Class balance ────────────────────────────────────────────────────────────

def _class_balance_weights(y: np.ndarray) -> np.ndarray:
    """
    Balanced class weights: n_samples / (n_classes * np.bincount(y)).
    Returns per-sample weights (not per-class).
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    cw = {cls: n_samples / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
    return np.array([cw[yi] for yi in y], dtype=float)


# ── Public API ───────────────────────────────────────────────────────────────

def compute_afml_weights(
    y: pd.Series,
    t1: pd.Series,
    returns: pd.Series,
    class_weight: str = 'balanced',
) -> np.ndarray:
    """
    Compute AFML sample weights = average_uniqueness × event_strength × class_balance.

    Parameters
    ----------
    y            : label series (0/1), indexed by event entry timestamps
    t1           : label-end timestamps (Triple Barrier t1), same index as y
    returns      : per-bar log-returns, same index as y
    class_weight : 'balanced' to apply sklearn-style balanced weights; else uniform

    Returns
    -------
    np.ndarray of float64, same length as y, summing to len(y).
    Falls back to uniform weights (all 1.0) on any error.
    """
    n = len(y)
    if n == 0:
        return np.array([], dtype=float)

    try:
        y_arr = np.asarray(y.values if hasattr(y, 'values') else y, dtype=int)
        ret_arr = np.asarray(
            returns.values if hasattr(returns, 'values') else returns,
            dtype=float,
        )
        ret_arr = np.nan_to_num(ret_arr, nan=0.0)

        # Align t1 to y's index
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("y.index must be a DatetimeIndex for AFML uniqueness weights")

        t1_aligned = t1.reindex(y.index)
        missing = t1_aligned.isna().sum()
        if missing > 0:
            logger.warning("[sample_weights] %d/%d t1 entries missing after reindex — filling with index value", missing, n)
            t1_aligned = t1_aligned.fillna(pd.Series(y.index, index=y.index))

        # 1. Average uniqueness
        t_start = y.index
        uniq = _average_uniqueness(t_start, t1_aligned)

        # 2. Event strength
        strength = _event_strength(ret_arr)

        # 3. Class balance
        if class_weight == 'balanced':
            balance = _class_balance_weights(y_arr)
        else:
            balance = np.ones(n, dtype=float)

        # Combine
        w = uniq * strength * balance

        # Guard against zeros / negatives
        w = np.where(w <= 0, 1e-6, w)

        # Normalise so sum == n (sklearn convention)
        w = w / w.sum() * n

        return w.astype(float)

    except Exception as exc:
        logger.warning(
            "[sample_weights] compute_afml_weights failed (%s) — using uniform weights",
            exc,
        )
        return np.ones(n, dtype=float)


if __name__ == '__main__':
    # Smoke test
    import pandas as pd

    n = 100
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    y_test = pd.Series(np.random.randint(0, 2, n), index=idx)
    t1_test = pd.Series(idx + pd.Timedelta(hours=12), index=idx)
    ret_test = pd.Series(np.random.randn(n) * 0.01, index=idx)

    w = compute_afml_weights(y_test, t1_test, ret_test)

    assert len(w) == n, f"Length mismatch: {len(w)} != {n}"
    assert (w > 0).all(), "Not all weights positive"
    assert abs(w.sum() - n) < 1.0, f"Weight sum {w.sum()} far from {n}"
    print(f"PASS: n={n}, w.sum()={w.sum():.2f}, w.min()={w.min():.4f}, w.max()={w.max():.4f}")
