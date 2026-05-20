"""
Threshold optimizer — grid search for optimal prediction confidence threshold.

Searches [0.20, 0.80] in steps of 0.05, maximising the Sortino ratio on a
held-out calibration set. Extracted and generalised from the meta-labeler's
Sortino search (train_meta_labeler.py lines 68-91).

Usage (in trainer, after calibration):
    from src.utils.threshold_optimizer import find_optimal_threshold
    best_thr, best_score = find_optimal_threshold(
        calibrated, X_cal, y_cal, returns_cal
    )
    # save best_thr to meta JSON as 'optimal_threshold'

Usage (in ml_predictor.py, at inference):
    threshold = meta.get('optimal_threshold', 0.5)
    signal = 1 if proba >= threshold else 0
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_GRID = np.arange(0.20, 0.81, 0.05)  # [0.20, 0.25, ..., 0.80]
_MIN_TRADES = 10                       # minimum trades at a threshold to be eligible
_DEFAULT_THRESHOLD = 0.50


def _sortino(trade_returns: np.ndarray, annualise_factor: float = 8760.0) -> float:
    """Annualised Sortino ratio for a vector of trade returns."""
    if len(trade_returns) < _MIN_TRADES:
        return 0.0
    mean_r = trade_returns.mean()
    downside = trade_returns[trade_returns < 0]
    if len(downside) == 0:
        return float(mean_r * np.sqrt(annualise_factor) * 10.0)
    dsv = downside.std()
    if dsv == 0:
        return 0.0
    return float(mean_r / dsv * np.sqrt(annualise_factor))


def find_optimal_threshold(
    model,
    X_cal: pd.DataFrame | np.ndarray,
    y_cal: pd.Series | np.ndarray,
    returns_cal: pd.Series | np.ndarray,
    signals_cal: pd.Series | np.ndarray | None = None,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Grid-search the confidence threshold that maximises Sortino on the
    calibration split.

    Parameters
    ----------
    model       : fitted calibrated classifier with predict_proba()
    X_cal       : calibration features
    y_cal       : true labels (0/1)
    returns_cal : per-bar log-returns aligned with y_cal
    signals_cal : optional primary signals (-1/0/+1); if None, all non-zero
                  predictions are treated as active trades
    grid        : threshold grid to search; defaults to np.arange(0.20, 0.81, 0.05)

    Returns
    -------
    (best_threshold, best_sortino_score)
    Falls back to (0.50, 0.0) on any error.
    """
    if grid is None:
        grid = _GRID

    try:
        proba = model.predict_proba(X_cal)[:, 1]
    except Exception as exc:
        logger.warning("[threshold_optimizer] predict_proba failed: %s -- using default %.2f", exc, _DEFAULT_THRESHOLD)
        return _DEFAULT_THRESHOLD, 0.0

    y = np.asarray(y_cal, dtype=int)
    ret = np.asarray(returns_cal, dtype=float)
    if np.all(np.isnan(ret)):
        ret = np.zeros(len(y))
    ret = np.nan_to_num(ret, nan=0.0)

    if signals_cal is not None:
        signals = np.asarray(signals_cal, dtype=float)
    else:
        signals = np.ones(len(y), dtype=float)  # treat all as active

    best_thr = _DEFAULT_THRESHOLD
    best_score = -np.inf

    for thr in grid:
        # Approved = proba >= threshold AND signal is non-zero
        mask = (proba >= thr) & (signals != 0)
        if mask.sum() < _MIN_TRADES:
            continue
        # Trade return: +return when correct, -return when wrong
        trade_returns = np.where(y[mask] == 1, ret[mask], -ret[mask])
        score = _sortino(trade_returns)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    if best_score == -np.inf:
        logger.warning(
            "[threshold_optimizer] No threshold had >= %d trades -- using default %.2f",
            _MIN_TRADES, _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD, 0.0

    logger.info(
        "[threshold_optimizer] Best threshold=%.2f Sortino=%.4f (grid %d steps)",
        best_thr, best_score, len(grid),
    )
    return best_thr, best_score
