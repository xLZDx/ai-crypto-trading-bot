"""
Fractional Differencing — Lopez de Prado (Advances in Financial ML, Ch. 5).

Replaces pct_change() across all models. d=0.4 is the empirically optimal value
that achieves stationarity while retaining maximum trend memory.

Standard differencing (d=1) destroys all price memory.
Raw prices (d=0) cause model overfitting on non-stationary trends.
Fractional d∈(0,1) is the mathematically sound middle ground.

Formula: Δᵈx_t = Σ_{k=0}^{L} w_k * x_{t-k}
Weights:  w_0 = 1,  w_k = w_{k-1} * (d - k + 1) / k
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Compute binomial series weights until |w_k| < threshold."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1])  # oldest first


def fractional_diff(series: pd.Series, d: float = 0.4, threshold: float = 1e-4) -> pd.Series:
    """
    Apply fractional differencing to a price series.

    Args:
        series:    Raw close price series (or any price-like series).
        d:         Differencing order. 0.4 is the sweet-spot for crypto hourly data.
        threshold: Weight cutoff — weights below this are dropped (controls window length).

    Returns:
        Series of same length with NaN for the warm-up period.
    """
    weights = _compute_weights(d, threshold)
    width = len(weights)
    vals = series.values.astype(float)
    n = len(vals)

    result = np.full(n, np.nan)
    for i in range(width - 1, n):
        result[i] = float(np.dot(weights, vals[i - width + 1: i + 1]))

    return pd.Series(result, index=series.index, name=f"frac_diff_d{d:.2f}".replace(".", ""))


def add_fractional_diff(df: pd.DataFrame, d: float = 0.4, col: str = "close") -> pd.DataFrame:
    """Add fractional diff column to DataFrame. Used by all training scripts."""
    colname = f"frac_diff_d{int(d * 100):02d}"
    df[colname] = fractional_diff(df[col], d=d)
    return df


def find_min_d(series: pd.Series, d_range: tuple = (0.1, 1.0), step: float = 0.05) -> float:
    """
    Find the minimum d that makes the series stationary (ADF p < 0.05).
    Call this once offline — result is typically 0.35–0.45 for crypto.
    """
    from statsmodels.tsa.stattools import adfuller

    d = d_range[0]
    while d <= d_range[1]:
        diff = fractional_diff(series, d=d).dropna()
        if len(diff) < 30:
            d += step
            continue
        pval = adfuller(diff, maxlag=1)[1]
        if pval < 0.05:
            return round(d, 4)
        d = round(d + step, 4)
    return d_range[1]
