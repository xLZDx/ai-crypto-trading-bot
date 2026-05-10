"""
Event-Time Labeling Engine — Phase 2, Level 2 (Alpha Engine).

Replaces the candle-based Triple Barrier (`triple_barrier.py`) with a labeling
system that operates in *event time* using regime-normalized barriers.

Per updated_architecture_plan_en.md §5:

    # 1. Barriers based on volatility normalized by regime
    df["vol_norm"] = df["atr"] / df["atr"].rolling(100).mean()
    dynamic_tp = k_tp * atr * vol_norm
    dynamic_sl = k_sl * atr * vol_norm

    # 2. Removing timeouts for binary classification
    mask = labels != 0
    X_filtered = X[mask]
    y_filtered = (labels[mask] == 1).astype(int)  # TP hit vs SL hit

Why event-time, not bar-time:
    Bar-time labeling assumes uniform information flow. In crypto microstructure,
    information bursts (around news, large orders, funding events) carry far more
    predictive content per unit wall-clock time than quiet periods. Event-time
    labels treat each *price-touch event* as a sample, eliminating the implicit
    bar-frequency bias.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EventTimeLabels(NamedTuple):
    """Output bundle from `label_event_time`."""
    labels:    pd.Series   # +1 / -1 / 0  (0 = timeout)
    t1:        pd.Series   # resolution timestamps
    binary_y:  pd.Series   # 0/1 with timeouts dropped (filter via .index)
    binary_idx: pd.Index   # the index of `binary_y` (timeout-free subset)
    stats:     dict


def regime_normalized_barriers(
    df: pd.DataFrame,
    *,
    k_tp: float = 2.0,
    k_sl: float = 2.0,
    atr_col: str = "atr_14",
    vol_window: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (dynamic_tp, dynamic_sl, vol_norm) per the architecture plan.

        vol_norm = atr / atr.rolling(window).mean()
        dynamic_tp = k_tp * atr * vol_norm
        dynamic_sl = k_sl * atr * vol_norm
    """
    if atr_col in df.columns:
        atr = pd.to_numeric(df[atr_col], errors="coerce").bfill().ffill().to_numpy()
    else:
        prev_close = df["close"].shift(1).bfill().to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ])
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().bfill().to_numpy()

    atr_mean = pd.Series(atr).rolling(vol_window, min_periods=1).mean().bfill().to_numpy()
    vol_norm = np.where(atr_mean > 0, atr / atr_mean, 1.0)
    dynamic_tp = k_tp * atr * vol_norm
    dynamic_sl = k_sl * atr * vol_norm
    return dynamic_tp, dynamic_sl, vol_norm


def label_event_time(
    df: pd.DataFrame,
    *,
    k_tp: float = 2.0,
    k_sl: float = 2.0,
    max_horizon_bars: int = 240,
    atr_col: str = "atr_14",
) -> EventTimeLabels:
    """Generate event-time labels with regime-normalized barriers.

    Args:
        df:               Frame with at least `close`, `high`, `low` and a
                          `timestamp` column (or DatetimeIndex).
        k_tp, k_sl:       Barrier multipliers in units of ATR×vol_norm.
        max_horizon_bars: Cap on lookahead. Plan §5 says "remove timeouts" —
                          we still need a finite horizon to bound compute, then
                          drop timeouts for binary classification.
        atr_col:          ATR column name.

    Returns:
        EventTimeLabels(labels, t1, binary_y, binary_idx, stats)
    """
    n = len(df)
    if n < 2:
        empty = pd.Series(dtype=np.int8)
        return EventTimeLabels(empty, empty, empty.astype(np.int8), empty.index, {})

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low  = df["low"].to_numpy(dtype=float)

    dynamic_tp, dynamic_sl, vol_norm = regime_normalized_barriers(
        df, k_tp=k_tp, k_sl=k_sl, atr_col=atr_col,
    )

    labels = np.zeros(n, dtype=np.int8)
    t1_idx = np.zeros(n, dtype=np.int64)

    # Vectorized expansion of the search window; identical structure to
    # triple_barrier_labels_vectorized but with regime-normalized barriers.
    for offset in range(1, max_horizon_bars + 1):
        future_high = np.concatenate([high[offset:], np.full(offset, np.nan)])
        future_low  = np.concatenate([low[offset:],  np.full(offset, np.nan)])

        unresolved = labels == 0
        hit_upper = unresolved & (future_high >= close + dynamic_tp)
        hit_lower = unresolved & (future_low  <= close - dynamic_sl)

        labels[hit_upper & ~hit_lower] = 1
        labels[hit_lower & ~hit_upper] = -1
        both = hit_upper & hit_lower
        if both.any():
            both_idx = np.where(both)[0]
            tp_dist = dynamic_tp[both]
            sl_dist = dynamic_sl[both]
            labels[both_idx[tp_dist <= sl_dist]] = 1
            labels[both_idx[tp_dist >  sl_dist]] = -1

        just_resolved = unresolved & (labels != 0)
        if just_resolved.any():
            t1_idx[just_resolved] = np.arange(n)[just_resolved] + offset

    timeouts = labels == 0
    t1_idx[timeouts] = np.minimum(np.arange(n)[timeouts] + max_horizon_bars, n - 1)

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"]).to_numpy()
    else:
        ts = df.index.to_numpy()
    t1_times = ts[t1_idx]

    # Mask out the final tail where lookahead is incomplete.
    labels[max(0, n - max_horizon_bars):] = 0

    labels_s = pd.Series(labels, index=df.index, name="event_time_label")
    t1_s = pd.Series(t1_times, index=df.index, name="t1_timestamp")

    # Binary subset: drop timeouts, then label TP=1 / SL=0.
    keep = labels_s != 0
    binary_y = (labels_s[keep] == 1).astype(np.int8).rename("binary_y")

    stats = {
        "n":             int(n),
        "long_pct":      float(round((labels_s == 1).sum() / n * 100, 2)),
        "short_pct":     float(round((labels_s == -1).sum() / n * 100, 2)),
        "timeout_pct":   float(round((labels_s == 0).sum() / n * 100, 2)),
        "binary_n":      int(len(binary_y)),
        "binary_pos":    float(round(binary_y.mean() * 100, 2)) if len(binary_y) else 0.0,
        "vol_norm_mean": float(np.nanmean(vol_norm)),
    }
    return EventTimeLabels(
        labels=labels_s,
        t1=t1_s,
        binary_y=binary_y,
        binary_idx=binary_y.index,
        stats=stats,
    )


def filter_for_binary_classification(X: pd.DataFrame, labels: EventTimeLabels):
    """Match the plan's snippet:

        mask = labels != 0
        X_filtered = X[mask]
        y_filtered = (labels[mask] == 1).astype(int)
    """
    return X.loc[labels.binary_idx], labels.binary_y


__all__ = [
    "EventTimeLabels",
    "regime_normalized_barriers",
    "label_event_time",
    "filter_for_binary_classification",
]
