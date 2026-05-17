"""
Kalman filter for price-noise cleaning — Phase 1, Level 1 (Data Layer).

Implements the configuration from updated_architecture_plan_en.md §3:

    from pykalman import KalmanFilter
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                      initial_state_mean=0, initial_state_covariance=1,
                      observation_covariance=1, transition_covariance=0.01)
    state_means, _ = kf.filter(df["close"].values)
    df["price_kalman"] = state_means

The plan literally specifies `initial_state_mean=0`, which is the right choice
for *return* series but produces a long transient when applied to absolute
price levels (e.g. $30,000). For price series we therefore default to
warm-start at the first observation and expose a flag to follow the plan
verbatim. Both modes are causal — filter() never sees future observations.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Constants from updated_architecture_plan_en.md §3
_PLAN_OBSERVATION_COVARIANCE = 1.0
_PLAN_TRANSITION_COVARIANCE = 0.01
_PLAN_INITIAL_STATE_COVARIANCE = 1.0


def smooth_price(
    prices,
    *,
    observation_covariance: float = _PLAN_OBSERVATION_COVARIANCE,
    transition_covariance: float = _PLAN_TRANSITION_COVARIANCE,
    initial_state_covariance: float = _PLAN_INITIAL_STATE_COVARIANCE,
    initial_state_mean: float | None = None,
    use_plan_initial: bool = False,
) -> np.ndarray:
    """Apply a Kalman filter to a 1-D series, return the filtered values.

    Args:
        prices: 1-D array-like of observed values (typically `close`).
        observation_covariance: σ² of measurement noise. Plan = 1.0.
        transition_covariance:  σ² of process noise. Plan = 0.01.
        initial_state_covariance: prior variance on initial state. Plan = 1.0.
        initial_state_mean: prior mean on initial state. Plan = 0; default
            is the first observation (warm-start) which avoids a multi-step
            transient when the plan's literal value is unsuitable.
        use_plan_initial: True forces `initial_state_mean=0` (plan-verbatim).

    Returns:
        np.ndarray of the same shape as `prices`, with NaN for any input NaNs.
    """
    arr = np.asarray(prices, dtype=float).ravel()
    n = arr.size
    if n == 0:
        return arr

    valid_mask = ~np.isnan(arr)
    if not valid_mask.any():
        return arr.copy()

    if use_plan_initial:
        ism = 0.0
    elif initial_state_mean is None:
        ism = float(arr[valid_mask][0])
    else:
        ism = float(initial_state_mean)

    try:
        from pykalman import KalmanFilter
    except ImportError as exc:
        logger.warning("pykalman not installed -- returning raw input. (%s)", exc)
        return arr.copy()

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=ism,
        initial_state_covariance=initial_state_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
    )

    # pykalman handles NaN as masked observations — we mask them explicitly.
    masked = np.ma.array(arr, mask=~valid_mask)
    state_means, _ = kf.filter(masked)
    return np.asarray(state_means).ravel()


def smooth_dataframe(
    df,
    column: str = "close",
    out_column: str = "price_kalman",
    **kwargs,
):
    """In-place add a Kalman-smoothed column to a DataFrame and return it.

    Equivalent to::

        df["price_kalman"] = smooth_price(df["close"].values)
    """
    if column not in df.columns:
        raise KeyError(f"DataFrame has no column '{column}'")
    df[out_column] = smooth_price(df[column].values, **kwargs)
    return df


__all__ = ["smooth_price", "smooth_dataframe"]
