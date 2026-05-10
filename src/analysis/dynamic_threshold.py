"""
Dynamic confidence threshold — Phase 4, Level 4.

Per updated_architecture_plan_en.md §15:

    def find_best_threshold(y_true, probs, returns):
        best_thr, best_sharpe = 0.5, -np.inf
        for thr in np.linspace(0.5, 0.8, 30):
            preds = (probs > thr).astype(int)
            pnl = pd.Series(preds * returns)
            sharpe = pnl.mean() / (pnl.std() + 1e-9)
            if sharpe > best_sharpe:
                best_sharpe, best_thr = sharpe, thr
        return best_thr

Replaces the fixed `SIGNAL_THRESHOLD` constant in `src/utils/config.py` with
a value chosen on-the-fly from the most recent validation window. This makes
the bot adaptive: as model calibration drifts (regime shifts), the threshold
floats up or down to keep the realised Sharpe ratio high.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ThresholdSearchResult:
    best_threshold: float
    best_sharpe:    float
    best_pnl_mean:  float
    best_pnl_std:   float
    best_n_trades:  int
    grid:           list[tuple[float, float, int]]   # (thr, sharpe, n_trades)


def find_best_threshold(
    probs:   np.ndarray,
    returns: np.ndarray,
    *,
    grid_low:   float = 0.5,
    grid_high:  float = 0.8,
    grid_n:     int   = 30,
    min_trades: int   = 5,
    metric:     str   = "sharpe",
) -> ThresholdSearchResult:
    """Grid-search over `[grid_low, grid_high]` for the threshold that
    maximises out-of-sample Sharpe (or `pnl` / `sortino`).

    Args:
        probs:       Calibrated model P(win) for each sample.
        returns:     Realised returns aligned with `probs`. Multiplied by
                     the binary entry decision to get strategy PnL.
        grid_low / grid_high / grid_n:  search range and resolution.
        min_trades:  fall back to default 0.5 if no threshold yields at
                     least this many entries.
        metric:      "sharpe" (default), "sortino", or "pnl".

    Returns:
        ThresholdSearchResult with the chosen threshold and the full grid.
    """
    p = np.asarray(probs, dtype=float)
    r = np.asarray(returns, dtype=float)
    if p.size != r.size:
        raise ValueError("probs and returns must have the same length")

    best_thr, best_metric = 0.5, -np.inf
    best_pnl_mean, best_pnl_std, best_n = 0.0, 0.0, 0
    grid: list[tuple[float, float, int]] = []
    for thr in np.linspace(grid_low, grid_high, grid_n):
        preds = (p > thr).astype(int)
        n_trades = int(preds.sum())
        if n_trades < min_trades:
            grid.append((float(thr), float("-inf"), n_trades))
            continue
        pnl = preds * r
        if metric == "sharpe":
            score = float(pnl.mean() / (pnl.std() + 1e-9))
        elif metric == "sortino":
            downside = np.minimum(pnl, 0)
            d_std = float(np.sqrt(np.mean(downside ** 2)))
            score = float(pnl.mean() / (d_std + 1e-9))
        elif metric == "pnl":
            score = float(pnl.sum())
        else:
            raise ValueError(f"unknown metric: {metric}")
        grid.append((float(thr), score, n_trades))
        if score > best_metric:
            best_metric = score
            best_thr = float(thr)
            best_pnl_mean = float(pnl.mean())
            best_pnl_std  = float(pnl.std())
            best_n = n_trades

    return ThresholdSearchResult(
        best_threshold=best_thr,
        best_sharpe=best_metric,
        best_pnl_mean=best_pnl_mean,
        best_pnl_std=best_pnl_std,
        best_n_trades=best_n,
        grid=grid,
    )


def rolling_threshold(
    probs:   pd.Series,
    returns: pd.Series,
    *,
    window:  int = 1000,
    refit_every: int = 100,
    grid_low:  float = 0.5,
    grid_high: float = 0.8,
) -> pd.Series:
    """Online dynamic threshold: re-fit every `refit_every` bars on the
    trailing `window` bars. Returns a Series of thresholds aligned with
    `probs.index`. Bars before the warm-up are filled with 0.5.
    """
    p = probs.to_numpy()
    r = returns.to_numpy()
    n = len(p)
    out = np.full(n, 0.5, dtype=float)
    last_thr = 0.5
    for i in range(window, n):
        if (i - window) % refit_every == 0:
            res = find_best_threshold(
                p[i - window:i], r[i - window:i],
                grid_low=grid_low, grid_high=grid_high,
            )
            last_thr = res.best_threshold
        out[i] = last_thr
    return pd.Series(out, index=probs.index, name="dynamic_threshold")


__all__ = ["find_best_threshold", "rolling_threshold", "ThresholdSearchResult"]
