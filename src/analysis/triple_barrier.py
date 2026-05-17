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

AFML conformance (cross-checked against BryceMeng/mlfinlab_research_bryce):
  - ATR applied ONCE (not squared via vol_norm — fixed BUG-1)
  - Asymmetric barriers (pt=2.5, sl=1.5) — R/R = 1.67, AFML Ch.3 default
  - Tail rows zeroed only when label cannot resolve within available data
  - All-NaN ATR raises ValueError instead of silently producing zero labels
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def triple_barrier_labels_vectorized(
    df: pd.DataFrame,
    pt_multiplier: float = 2.5,
    sl_multiplier: float = 1.5,
    max_bars: int = 12,
    atr_col: str = "atr_14",
) -> tuple[pd.Series, pd.Series]:
    """
    Fast vectorized variant using ATR-based barriers (applied ONCE per AFML Ch.3).
      TP = entry + pt_multiplier * ATR
      SL = entry - sl_multiplier * ATR

    Defaults pt=2.5, sl=1.5 (asymmetric R/R = 1.67) and max_bars=12 per the AFML
    playbook. NO vol_norm intermediate — that produced ATR² scaling (see BUG-1).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    # Empty input — return empty series of the right dtype/name, no error.
    if n == 0:
        return (
            pd.Series([], dtype=np.int8, name="triple_barrier_label"),
            pd.Series([], dtype="datetime64[ns]", name="t1_timestamp"),
        )

    if atr_col not in df.columns:
        prev_close = df["close"].shift(1).fillna(close[0]).values
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    else:
        atr = df[atr_col].bfill().ffill().values

    # Guard against all-NaN ATR — silent all-zero labels was BUG-N4.
    # n>0 by this point, so this only fires on genuine all-NaN columns.
    if np.isnan(atr).all():
        raise ValueError(
            f"ATR column '{atr_col}' is entirely NaN -- cannot compute barriers. "
            "Verify feature engineering produced atr_14 before calling this."
        )
    # Patch any remaining NaN spots with median to avoid downstream propagation.
    if np.isnan(atr).any():
        median_atr = np.nanmedian(atr)
        atr = np.where(np.isnan(atr), median_atr, atr)

    # ATR applied ONCE. No vol_norm. Barriers scale linearly with volatility,
    # not quadratically (BUG-1 fixed).
    dynamic_tp = pt_multiplier * atr
    dynamic_sl = sl_multiplier * atr

    labels = np.zeros(n, dtype=np.int8)
    t1_idx = np.full(n, n, dtype=np.intp)  # default: unresolvable (= n)

    # Hoist np.arange out of the inner loop (BUG-N: redundant allocation per offset).
    indices = np.arange(n)

    for offset in range(1, max_bars + 1):
        # Build length-n future-shifted arrays. For offset >= n, the head
        # `high[offset:]` is empty, so we must clip to length n explicitly.
        if offset >= n:
            future_high = np.full(n, np.nan)
            future_low = np.full(n, np.nan)
        else:
            future_high = np.concatenate([high[offset:], np.full(offset, np.nan)])
            future_low = np.concatenate([low[offset:], np.full(offset, np.nan)])

        unresolved = labels == 0
        hit_upper = unresolved & (future_high >= close + dynamic_tp)
        hit_lower = unresolved & (future_low <= close - dynamic_sl)

        # Profit target takes priority if both hit same bar
        labels[hit_upper & ~hit_lower] = 1
        labels[hit_lower & ~hit_upper] = -1
        # Both same bar: whichever barrier is tighter (closer to entry)
        both = hit_upper & hit_lower
        if both.any():
            dist_upper = dynamic_tp[both]
            dist_lower = dynamic_sl[both]
            idx = indices[both]
            labels[idx[dist_upper <= dist_lower]] = 1
            labels[idx[dist_upper > dist_lower]] = -1

        just_resolved = unresolved & (labels != 0)
        if just_resolved.any():
            t1_idx[just_resolved] = indices[just_resolved] + offset

    # Timeout rows: record their resolution time as min(i + max_bars, n - 1).
    timeouts = labels == 0
    t1_idx_resolved = np.minimum(indices + max_bars, n - 1)
    t1_idx[timeouts] = t1_idx_resolved[timeouts]

    # Map integer indexes back to absolute timestamps for strict cutoff evaluation
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"]).values
    else:
        timestamps = df.index.values
    t1_times = timestamps[t1_idx]

    # ONLY zero labels that genuinely cannot resolve within remaining data.
    # BUG-N1 fix: only mark `tb_label = 0` for bars where t1_idx >= n
    # (i.e. label would resolve beyond available data) — NOT every last
    # `max_bars` rows unconditionally.
    unresolvable = (np.arange(n) + max_bars) >= n  # last max_bars positions
    # But keep the labels that genuinely fired before n:
    # if t1_idx[i] < n AND labels[i] != 0, the barrier resolved → keep label.
    # Only force-zero rows where label is still 0 AND would have needed beyond n.
    final_zero_mask = unresolvable & (labels == 0)
    labels[final_zero_mask] = 0  # explicit (already 0, but documented intent)

    return (
        pd.Series(labels, index=df.index, name="triple_barrier_label"),
        pd.Series(t1_times, index=df.index, name="t1_timestamp"),
    )


def label_stats(labels: pd.Series) -> dict:
    """Return class distribution for logging."""
    counts = labels.value_counts().to_dict()
    total = len(labels)
    if total == 0:
        return {"long_pct": 0.0, "short_pct": 0.0, "timeout_pct": 0.0, "total": 0}
    return {
        "long_pct": round(counts.get(1, 0) / total * 100, 1),
        "short_pct": round(counts.get(-1, 0) / total * 100, 1),
        "timeout_pct": round(counts.get(0, 0) / total * 100, 1),
        "total": total,
    }


# ────────────────────────────────────────────────────────────────────────────
#  Phase 1 — strict causal t1 audit
#  Refer to updated_architecture_plan_en.md §4 — point 4 in the anti-leakage
#  checklist: "t1 from Triple Barrier must not overlap with the test set".
# ────────────────────────────────────────────────────────────────────────────


def causal_t1_audit(
    t1_times: pd.Series,
    train_end: pd.Timestamp | str,
    test_start: pd.Timestamp | str | None = None,
) -> dict:
    """Verify no train-set label resolves *after* the train/test boundary.

    The Triple Barrier resolves each label at `t1` (when TP/SL/timeout fires).
    If a sample's `t1` lies after `train_end`, the model would be trained on
    information from the test period — classic temporal leakage.

    Args:
        t1_times: Series of resolution timestamps (output of
                  triple_barrier_labels_vectorized's second return value).
                  Must have a DatetimeIndex.
        train_end: Last timestamp included in the training window (inclusive).
        test_start: First timestamp of the test window. Defaults to
                    `train_end + 1ns` (immediate adjacency). Pass a later
                    value to enforce a purge gap (recommended ≥ 1 max_bars).

    Returns:
        {ok, n_violations, first_violation, recommended_purge_until}
    """
    t1 = pd.to_datetime(t1_times)
    train_end = pd.to_datetime(train_end)
    if test_start is None:
        test_start = train_end + pd.Timedelta(nanoseconds=1)
    else:
        test_start = pd.to_datetime(test_start)

    # BUG-N2 fix: the previous `hasattr(t1.index, "to_series")` check was always
    # True (all pandas Index types have to_series), so the ternary was dead-code.
    # Require an explicit DatetimeIndex for the mask, otherwise treat all t1
    # values as in-scope (no row-level filtering).
    if isinstance(t1.index, pd.DatetimeIndex):
        train_mask_idx = t1.index <= train_end
        train_t1 = t1.loc[train_mask_idx]
    else:
        # Non-datetime index — caller is responsible for slicing to train rows.
        train_t1 = t1

    violations = train_t1[train_t1 >= test_start]
    return {
        "ok":              violations.empty,
        "n_violations":    int(len(violations)),
        "first_violation": str(violations.iloc[0]) if not violations.empty else None,
        # If violations exist, drop everything that resolves into the gap.
        "recommended_purge_until": str(train_t1.max()) if violations.empty else str(violations.max()),
    }


def purge_overlapping_train(
    df: pd.DataFrame,
    t1_times: pd.Series,
    train_end: pd.Timestamp | str,
    test_start: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Drop train rows whose label resolution overlaps the test window.

    Returns a *copy* of `df` with the offending rows removed. Used in
    PurgedKFold-style splits to guarantee strict causality.
    """
    audit = causal_t1_audit(t1_times, train_end=train_end, test_start=test_start)
    if audit["ok"]:
        return df
    keep_mask = pd.to_datetime(t1_times) < pd.to_datetime(test_start or train_end)
    return df.loc[keep_mask].copy()
