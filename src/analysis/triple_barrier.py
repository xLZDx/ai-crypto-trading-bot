"""
Triple Barrier Method — Lopez de Prado (Advances in Financial ML, Ch. 3).

Replaces all binary targets (close.shift(-n) > close) across every training script.

Instead of asking "will price be higher in N candles?" we ask:
  "Which barrier will price touch FIRST — profit target, stop loss, or timeout?"

Labels: +1 = profit target hit first  (upper barrier)
         -1 = stop loss hit first      (lower barrier)
          0 = timeout (no barrier hit within max_bars)

This gives the model real risk-profile information — it learns not just direction
but trade quality. A +1 signal with tight stops is worth far more than a +1 with
wide ones, and the model learns this distinction naturally.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def triple_barrier_labels_vectorized(
    df: pd.DataFrame,
    pt_multiplier: float = 2.0,
    sl_multiplier: float = 2.0,
    max_bars: int = 24,
    atr_col: str = "atr_14",
) -> tuple[pd.Series, pd.Series]:
    """
    Fast vectorized variant using dynamic volatility-based barriers.
    TP = entry + pt_multiplier * ATR * vol_norm
    SL = entry - sl_multiplier * ATR * vol_norm
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    if atr_col not in df.columns:
        prev_close = df["close"].shift(1).fillna(close[0]).values
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    else:
        atr = df[atr_col].bfill().ffill().values
        
    # Normalize ATR by regime volatility to avoid tiny stops in quiet markets
    atr_mean = pd.Series(atr).rolling(100, min_periods=1).mean().bfill().values
    vol_norm = np.where(atr_mean > 0, atr / atr_mean, 1.0)
    
    dynamic_tp = pt_multiplier * atr * vol_norm
    dynamic_sl = sl_multiplier * atr * vol_norm

    labels = np.zeros(n, dtype=np.int8)
    t1_idx = np.zeros(n, dtype=np.int32)

    for offset in range(1, max_bars + 1):
        future_high = np.concatenate([high[offset:], np.full(offset, np.nan)])
        future_low = np.concatenate([low[offset:], np.full(offset, np.nan)])

        unresolved = labels == 0
        hit_upper = unresolved & (future_high >= close + dynamic_tp)
        hit_lower = unresolved & (future_low <= close - dynamic_sl)

        # Profit target takes priority if both hit same bar
        labels[hit_upper & ~hit_lower] = 1
        labels[hit_lower & ~hit_upper] = -1
        # Both same bar: whichever is closer to entry
        both = hit_upper & hit_lower
        if both.any():
            dist_upper = dynamic_tp[both]
            dist_lower = dynamic_sl[both]
            idx = np.where(both)[0]
            labels[idx[dist_upper <= dist_lower]] = 1
            labels[idx[dist_upper > dist_lower]] = -1
            
        just_resolved = unresolved & (labels != 0)
        if just_resolved.any():
            t1_idx[just_resolved] = np.arange(n)[just_resolved] + offset

    timeouts = labels == 0
    t1_idx[timeouts] = np.minimum(np.arange(n)[timeouts] + max_bars, n - 1)
    
    # Safely map integer indexes back to absolute timestamps for strict cutoff evaluation
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"]).values
    else:
        timestamps = df.index.values
    t1_times = timestamps[t1_idx]

    labels[max(0, n - max_bars):] = 0
    return pd.Series(labels, index=df.index, name="triple_barrier_label"), pd.Series(t1_times, index=df.index, name="t1_timestamp")


def label_stats(labels: pd.Series) -> dict:
    """Return class distribution for logging."""
    counts = labels.value_counts().to_dict()
    total = len(labels)
    return {
        "long_pct": round(counts.get(1, 0) / total * 100, 1),
        "short_pct": round(counts.get(-1, 0) / total * 100, 1),
        "timeout_pct": round(counts.get(0, 0) / total * 100, 1),
        "total": total,
    }
