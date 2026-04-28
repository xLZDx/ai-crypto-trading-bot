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


def triple_barrier_labels(
    df: pd.DataFrame,
    profit_pct: float | None = None,
    loss_pct: float | None = None,
    atr_profit_mult: float = 2.0,
    atr_loss_mult: float = 1.0,
    max_bars: int = 24,
    atr_col: str = "atr_14",
) -> pd.Series:
    """
    Compute triple barrier labels for every row in df.

    Barrier logic (per bar i):
      upper = close[i] * (1 + profit_pct)   OR  close[i] + atr * atr_profit_mult
      lower = close[i] * (1 - loss_pct)     OR  close[i] - atr * atr_loss_mult
      Scan forward up to max_bars.
      First touched barrier determines label.

    Args:
        df:              OHLCV DataFrame (must have 'close', 'high', 'low').
        profit_pct:      Fixed profit barrier (e.g. 0.02 = 2%). If None, uses ATR.
        loss_pct:        Fixed stop barrier (e.g. 0.01 = 1%). If None, uses ATR.
        atr_profit_mult: ATR multiplier for upper barrier (used when profit_pct is None).
        atr_loss_mult:   ATR multiplier for lower barrier (used when loss_pct is None).
        max_bars:        Maximum bars to hold before timeout label (0).
        atr_col:         Column name for ATR (must be pre-computed in df).

    Returns:
        Integer Series with values in {-1, 0, 1}.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    use_atr = (profit_pct is None or loss_pct is None)
    if use_atr and atr_col not in df.columns:
        # Compute ATR on the fly (simple rolling TR mean)
        prev_close = df["close"].shift(1).values
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        atr = pd.Series(tr).rolling(14).mean().values
    else:
        atr = df[atr_col].values if use_atr else np.zeros(n)

    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        entry = close[i]
        if entry <= 0:
            continue

        a = atr[i] if (use_atr and not np.isnan(atr[i])) else entry * 0.01

        upper = entry + (atr_profit_mult * a) if profit_pct is None else entry * (1.0 + profit_pct)
        lower = entry - (atr_loss_mult * a)  if loss_pct is None  else entry * (1.0 - loss_pct)

        end = min(i + max_bars + 1, n)
        for j in range(i + 1, end):
            # Use high/low to detect intra-bar touches (more realistic)
            if high[j] >= upper:
                labels[i] = 1
                break
            if low[j] <= lower:
                labels[i] = -1
                break
        # else stays 0 (timeout)

    # Last max_bars rows cannot be reliably labeled
    labels[max(0, n - max_bars):] = 0

    return pd.Series(labels, index=df.index, name="triple_barrier_label")


def triple_barrier_labels_vectorized(
    df: pd.DataFrame,
    profit_pct: float = 0.02,
    loss_pct: float = 0.01,
    max_bars: int = 24,
) -> pd.Series:
    """
    Fast vectorized variant using fixed percentage barriers.
    ~50x faster than the loop version for large DataFrames.
    Use this for training (offline); loop version for live signal generation.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    labels = np.zeros(n, dtype=np.int8)
    upper_mult = 1.0 + profit_pct
    lower_mult = 1.0 - loss_pct

    for offset in range(1, max_bars + 1):
        future_high = np.concatenate([high[offset:], np.full(offset, np.nan)])
        future_low = np.concatenate([low[offset:], np.full(offset, np.nan)])

        unresolved = labels == 0
        # Profit target hit at this offset
        hit_upper = unresolved & (future_high >= close * upper_mult)
        # Stop hit at this offset
        hit_lower = unresolved & (future_low <= close * lower_mult)

        # Profit target takes priority if both hit same bar
        labels[hit_upper & ~hit_lower] = 1
        labels[hit_lower & ~hit_upper] = -1
        # Both same bar: whichever is closer to entry
        both = hit_upper & hit_lower
        if both.any():
            dist_upper = (close * upper_mult - close)[both]
            dist_lower = (close - close * lower_mult)[both]
            idx = np.where(both)[0]
            labels[idx[dist_upper <= dist_lower]] = 1
            labels[idx[dist_upper > dist_lower]] = -1

    labels[max(0, n - max_bars):] = 0
    return pd.Series(labels, index=df.index, name="triple_barrier_label")


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
