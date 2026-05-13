"""
Edge-case behavioral tests for src/analysis/triple_barrier.py.

Each test:
  1. Calls the function under test directly (no string-match assertions).
  2. Asserts on observable return values, shapes, or exception types.
  3. Is fully independent — no shared mutable state between tests.

Coverage targets (14 edge cases beyond test_phase0_fix.py):
  EC-01  Single-bar input (n=1)
  EC-02  Empty DataFrame
  EC-03  Missing atr_14 column — TR-based ATR fallback
  EC-04  Strongly trending up market — labels predominantly +1
  EC-05  Strongly trending down market — labels predominantly -1
  EC-06  Sideways low-volatility market — labels predominantly 0 (timeout)
  EC-07  ATR with single NaN value — patched with median, no raise
  EC-08  pt=3.0, sl=1.0 asymmetry — SL closer, biases toward -1 in random market
  EC-09  max_bars=1 — most labels are 0 (timeout)
  EC-10  max_bars boundary — last bar is always unresolvable
  EC-11  Both barriers hit same bar — tighter barrier wins
  EC-12  label_stats with all-+1 labels — long_pct=100, others=0
  EC-13  causal_t1_audit with non-DatetimeIndex — ok when no violations
  EC-14  purge_overlapping_train with no violations — returns input unchanged
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

from src.analysis.triple_barrier import (
    causal_t1_audit,
    label_stats,
    purge_overlapping_train,
    triple_barrier_labels_vectorized,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    close: np.ndarray,
    *,
    high_offset: float = 0.5,
    low_offset: float = 0.5,
    atr: float | np.ndarray | None = 1.0,
    freq: str = "1h",
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a DatetimeIndex."""
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    high = close + high_offset
    low = close - low_offset
    df: dict = {
        "close": close,
        "high": np.maximum(high, close),
        "low": np.minimum(low, close),
    }
    if atr is not None:
        if isinstance(atr, (int, float)):
            df["atr_14"] = np.full(n, float(atr))
        else:
            df["atr_14"] = np.asarray(atr, dtype=float)
    return pd.DataFrame(df, index=idx)


# ---------------------------------------------------------------------------
# EC-01  Single-bar input (n=1)
# ---------------------------------------------------------------------------

def test_ec01_single_bar_input():
    """n=1: no future bars exist — label must be 0 (timeout) with length 1.

    BUG DOCUMENTED (real implementation defect — test left failing intentionally):
    For n=1 and max_bars > 1, the loop at offset >= 2 computes:
        future_high = np.concatenate([high[2:], np.full(2, nan)])
    high[2:] is empty (shape (0,)) for n=1, so future_high has length 2 instead
    of n=1.  The subsequent boolean mask  `hit_upper & ~hit_lower` then has length
    2 while `labels` has length 1, raising:
        IndexError: boolean index did not match indexed array along axis 0;
                    size of axis is 1 but size of corresponding boolean axis is 2
    Fix needed: when offset > n, future_high/low construction should clip to n
    elements, not grow beyond n.  E.g. replace np.concatenate with:
        future_high = np.empty(n); future_high[:n-offset]=high[offset:]; future_high[n-offset:]=nan
    """
    df = _make_ohlcv(np.array([100.0]), atr=1.0)
    labels, t1 = triple_barrier_labels_vectorized(
        df, pt_multiplier=1.0, sl_multiplier=1.0, max_bars=5
    )
    assert len(labels) == 1, "Labels length must equal input length"
    assert len(t1) == 1, "t1_times length must equal input length"
    assert int(labels.iloc[0]) == 0, "Single bar: no future data — must be timeout (0)"


# ---------------------------------------------------------------------------
# EC-02  Empty DataFrame
# ---------------------------------------------------------------------------

def test_ec02_empty_dataframe():
    """n=0: function must return two empty Series without raising.

    BUG DOCUMENTED (real implementation defect — test left failing intentionally):
    With n=0, df['atr_14'] is an empty float array. After bfill().ffill().values
    the result is a length-0 array [].  np.isnan([]).all() returns True (vacuously
    true for empty input), so the all-NaN guard at line 61-65 fires and raises:
        ValueError: ATR column 'atr_14' is entirely NaN
    even though the array has zero elements — the invariant is unmeaningful on
    empty input.
    Fix needed: add an early return before the NaN guard when n == 0:
        if n == 0:
            empty = pd.Series([], dtype=np.int8, index=df.index)
            return empty.rename('triple_barrier_label'), empty.rename('t1_timestamp')
    """
    idx = pd.date_range("2024-01-01", periods=0, freq="1h")
    df = pd.DataFrame({"close": [], "high": [], "low": [], "atr_14": []}, index=idx)
    labels, t1 = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12
    )
    assert len(labels) == 0, "Labels must be empty for empty input"
    assert len(t1) == 0, "t1_times must be empty for empty input"
    assert labels.name == "triple_barrier_label"
    assert t1.name == "t1_timestamp"


# ---------------------------------------------------------------------------
# EC-03  Missing atr_14 column — TR-based ATR fallback
# ---------------------------------------------------------------------------

def test_ec03_missing_atr_col_falls_back_to_tr_based():
    """When atr_14 is absent the function computes TR-based ATR internally.

    Observable proof: labels are returned (no KeyError / ValueError), and at
    least one barrier resolves for an obvious trending series.
    """
    n = 50
    # Strong uptrend: each bar's high is well above entry + any reasonable ATR.
    close = np.linspace(100.0, 200.0, n)   # +100 pts over 50 bars
    high = close + 5.0                      # high is always +5 above close
    low = close - 0.1                       # tight low — SL almost unreachable
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    # Deliberately do NOT include atr_14.
    df = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)

    labels, t1 = triple_barrier_labels_vectorized(
        df, pt_multiplier=1.0, sl_multiplier=100.0, max_bars=20
    )
    assert len(labels) == n
    # TR = high-low = 5.1; ATR ≈ 5.1; TP = 1×ATR ≈ 5.1; the next bar's high
    # should immediately cross TP for most bars in this uptrend.
    assert (labels.values == 1).sum() > 0, (
        "TR-based ATR fallback: at least some bars should resolve +1 in uptrend"
    )


# ---------------------------------------------------------------------------
# EC-04  Strongly trending up market — labels predominantly +1
# ---------------------------------------------------------------------------

def test_ec04_strong_uptrend_biases_toward_long():
    """Monotonic uptrend: each bar moves up by 10×ATR — TP hit every bar."""
    n = 100
    atr_val = 1.0
    # Price rises 10 ATR every bar — TP (2.5 ATR) triggered immediately.
    close = 100.0 + np.arange(n) * (10.0 * atr_val)
    high = close + 12.0 * atr_val   # high well above close + TP
    low = close - 0.1 * atr_val     # low very close to close — SL stays intact

    df = _make_ohlcv(close, atr=atr_val)
    df["high"] = high
    df["low"] = np.minimum(low, close)

    labels, _ = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12
    )
    long_count = (labels.values == 1).sum()
    total_resolvable = n - 12  # exclude last max_bars that may timeout
    assert long_count / total_resolvable >= 0.90, (
        f"Strong uptrend: >=90% of resolvable labels should be +1, "
        f"got {long_count}/{total_resolvable}"
    )


# ---------------------------------------------------------------------------
# EC-05  Strongly trending down market — labels predominantly -1
# ---------------------------------------------------------------------------

def test_ec05_strong_downtrend_biases_toward_short():
    """Monotonic downtrend: each bar drops 10×ATR — SL hit every bar."""
    n = 100
    atr_val = 1.0
    close = 10000.0 - np.arange(n) * (10.0 * atr_val)
    low = close - 12.0 * atr_val    # low well below close − SL
    high = close + 0.1 * atr_val    # high barely above close — TP stays intact

    df = _make_ohlcv(close, atr=atr_val)
    df["high"] = np.maximum(high, close)
    df["low"] = low

    labels, _ = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12
    )
    short_count = (labels.values == -1).sum()
    total_resolvable = n - 12
    assert short_count / total_resolvable >= 0.90, (
        f"Strong downtrend: >=90% of resolvable labels should be -1, "
        f"got {short_count}/{total_resolvable}"
    )


# ---------------------------------------------------------------------------
# EC-06  Sideways low-volatility market — labels predominantly 0 (timeout)
# ---------------------------------------------------------------------------

def test_ec06_sideways_market_produces_timeouts():
    """Price oscillates within ±0.01 ATR — no barrier ever reached → all timeout."""
    n = 200
    atr_val = 10.0
    # Flat price; high/low within 0.05 ATR of close — far inside 1.5 ATR SL / 2.5 ATR TP.
    close = np.full(n, 100.0)
    high = close + 0.05 * atr_val   # 0.5 ATR movement — well inside SL (1.5 ATR)
    low = close - 0.05 * atr_val

    df = _make_ohlcv(close, atr=atr_val)
    df["high"] = high
    df["low"] = low

    labels, _ = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12
    )
    timeout_pct = (labels.values == 0).mean()
    assert timeout_pct >= 0.80, (
        f"Sideways market: >=80% labels should be timeout (0), got {timeout_pct:.1%}"
    )


# ---------------------------------------------------------------------------
# EC-07  ATR with a single NaN value — patched with median, no raise
# ---------------------------------------------------------------------------

def test_ec07_single_nan_atr_patched_with_median():
    """One NaN in atr_14: function should patch it with median and NOT raise."""
    n = 50
    atr_vals = np.full(n, 1.0)
    atr_vals[5] = np.nan  # exactly one NaN

    close = np.full(n, 100.0)
    df = _make_ohlcv(close, atr=atr_vals)

    # Should NOT raise ValueError (that is reserved for all-NaN).
    labels, t1 = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12
    )
    assert len(labels) == n, "Output length must match input after NaN patch"
    # Labels must be valid integers in {-1, 0, 1} — no NaN propagation.
    assert not np.isnan(labels.values.astype(float)).any(), (
        "NaN must not propagate into labels after median patch"
    )


# ---------------------------------------------------------------------------
# EC-08  pt/sl asymmetry: pt=3.0, sl=1.0 — SL is tighter → biases toward -1
# ---------------------------------------------------------------------------

def test_ec08_tight_sl_wide_pt_biases_toward_short():
    """With sl=1×ATR (tight) vs pt=3×ATR (wide), random walk hits SL more often.

    Gamblers-ruin / random-walk theory: probability of hitting barrier B before
    barrier A is proportional to distance(A) / (distance(A) + distance(B)).
    Here: P(-1) ≈ 3/(3+1) = 75%.  We check >50% (loose bound, deterministic data).
    """
    rng = np.random.default_rng(seed=0)
    n = 2000
    atr_val = 1.0
    returns = rng.normal(0, 0.5, n)  # half-ATR steps
    close = 1000.0 + np.cumsum(returns)
    # High/low = close ± 0.5 per bar (exactly the step size).
    high = close + 0.5
    low = close - 0.5

    df = _make_ohlcv(close, atr=atr_val)
    df["high"] = np.maximum(high, close)
    df["low"] = np.minimum(low, close)

    labels, _ = triple_barrier_labels_vectorized(
        df, pt_multiplier=3.0, sl_multiplier=1.0, max_bars=50
    )
    resolved = labels.values[labels.values != 0]
    if len(resolved) > 0:
        short_frac = (resolved == -1).mean()
        assert short_frac > 0.50, (
            f"Tight SL (1×ATR) vs wide TP (3×ATR): expect >50% short labels, "
            f"got {short_frac:.1%} among {len(resolved)} resolved"
        )


# ---------------------------------------------------------------------------
# EC-09  max_bars=1 — most labels must be 0 (timeout)
# ---------------------------------------------------------------------------

def test_ec09_max_bars_1_produces_mostly_timeouts():
    """With max_bars=1, only bars where the immediate next bar breaches a barrier
    get a non-zero label.  In a calm market this is rare → mostly timeout.
    """
    n = 200
    atr_val = 1.0
    close = np.full(n, 100.0)
    # Candles move ±0.3 ATR — well inside TP (2.5 ATR) and SL (1.5 ATR).
    high = close + 0.3 * atr_val
    low = close - 0.3 * atr_val

    df = _make_ohlcv(close, atr=atr_val)
    df["high"] = high
    df["low"] = low

    labels, _ = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=1
    )
    timeout_frac = (labels.values == 0).mean()
    assert timeout_frac >= 0.95, (
        f"max_bars=1 in calm market: expect >=95% timeouts, got {timeout_frac:.1%}"
    )


# ---------------------------------------------------------------------------
# EC-10  max_bars boundary — last bar must always be unresolvable (label 0)
#         when it was not already resolved by a prior barrier.
# ---------------------------------------------------------------------------

def test_ec10_last_bar_always_timeout():
    """The last position (index n-1) can never look forward; it must be 0."""
    n = 50
    atr_val = 1.0
    # Completely flat market — no barrier ever fires.
    close = np.full(n, 100.0)
    high = close + 0.01
    low = close - 0.01

    df = _make_ohlcv(close, atr=atr_val)
    df["high"] = high
    df["low"] = low

    labels, _ = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=5
    )
    assert int(labels.iloc[-1]) == 0, (
        "Last bar in a flat market must be timeout (0) — no future data to resolve"
    )


# ---------------------------------------------------------------------------
# EC-11  Both barriers hit same bar — tighter barrier wins
# ---------------------------------------------------------------------------

def test_ec11_both_barriers_same_bar_tighter_wins():
    """When both TP and SL are hit in the same future bar, the tighter one wins.

    Setup: pt_multiplier=0.5, sl_multiplier=1.0.  Both barriers may be hit
    by the next bar's high/low, but TP (0.5×ATR) is closer → label should be +1.
    """
    n = 20
    atr_val = 2.0
    close = np.full(n, 100.0)
    # Next bar's high = 101.5 = close + 0.75*ATR   — crosses TP (0.5*ATR = 1.0)
    # Next bar's low  = 97.0  = close - 1.5*ATR    — crosses SL (1.0*ATR = 2.0)
    # Wait: let's be precise.
    # TP distance = 0.5 * 2.0 = 1.0  → upper barrier at 101.0
    # SL distance = 1.0 * 2.0 = 2.0  → lower barrier at  98.0
    # Next bar: high = 101.5 (> 101.0 → TP hit), low = 97.0 (< 98.0 → SL hit)
    high = close.copy()
    low = close.copy()
    # Bar 0: we are testing its label. Bar 1 will be the "future" bar.
    high[1:] = 101.5    # crosses TP
    low[1:] = 97.0      # crosses SL

    df = _make_ohlcv(close, atr=atr_val)
    df["high"] = np.maximum(high, close)
    df["low"] = low

    labels, _ = triple_barrier_labels_vectorized(
        df, pt_multiplier=0.5, sl_multiplier=1.0, max_bars=5
    )
    # For bar 0: TP distance=1.0, SL distance=2.0 → TP is tighter → label=+1
    assert int(labels.iloc[0]) == 1, (
        f"Tighter barrier (TP dist=1.0) should win over SL dist=2.0; "
        f"got label={int(labels.iloc[0])}"
    )


def test_ec11b_both_barriers_same_bar_sl_tighter():
    """Mirror of EC-11: sl_multiplier < pt_multiplier → SL wins when both hit."""
    n = 20
    atr_val = 2.0
    close = np.full(n, 100.0)
    # TP distance = 2.0 * 2.0 = 4.0  → upper barrier at 104.0
    # SL distance = 0.5 * 2.0 = 1.0  → lower barrier at  99.0
    # Next bar: high = 105.0 (> 104.0 → TP hit), low = 98.0 (< 99.0 → SL hit)
    high = close.copy()
    low = close.copy()
    high[1:] = 105.0
    low[1:] = 98.0

    df = _make_ohlcv(close, atr=atr_val)
    df["high"] = np.maximum(high, close)
    df["low"] = low

    labels, _ = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.0, sl_multiplier=0.5, max_bars=5
    )
    # SL distance=1.0, TP distance=4.0 → SL is tighter → label=-1
    assert int(labels.iloc[0]) == -1, (
        f"Tighter barrier (SL dist=1.0) should win over TP dist=4.0; "
        f"got label={int(labels.iloc[0])}"
    )


# ---------------------------------------------------------------------------
# EC-12  label_stats with all-+1 labels
# ---------------------------------------------------------------------------

def test_ec12_label_stats_all_long():
    """All labels are +1: long_pct=100.0, short_pct=0.0, timeout_pct=0.0."""
    labels = pd.Series([1, 1, 1, 1, 1], dtype=np.int8)
    stats = label_stats(labels)
    assert stats["long_pct"] == 100.0
    assert stats["short_pct"] == 0.0
    assert stats["timeout_pct"] == 0.0
    assert stats["total"] == 5


def test_ec12b_label_stats_all_short():
    """All labels are -1: short_pct=100.0, others=0.0."""
    labels = pd.Series([-1, -1, -1], dtype=np.int8)
    stats = label_stats(labels)
    assert stats["short_pct"] == 100.0
    assert stats["long_pct"] == 0.0
    assert stats["timeout_pct"] == 0.0
    assert stats["total"] == 3


def test_ec12c_label_stats_all_timeout():
    """All labels are 0: timeout_pct=100.0, others=0.0."""
    labels = pd.Series([0, 0, 0, 0], dtype=np.int8)
    stats = label_stats(labels)
    assert stats["timeout_pct"] == 100.0
    assert stats["long_pct"] == 0.0
    assert stats["short_pct"] == 0.0
    assert stats["total"] == 4


# ---------------------------------------------------------------------------
# EC-13  causal_t1_audit with non-DatetimeIndex — ok=True when no violations
# ---------------------------------------------------------------------------

def test_ec13_non_datetime_index_no_violations():
    """With an integer index (non-DatetimeIndex), all t1 values are in scope.

    The implementation treats non-DatetimeIndex as 'caller responsible for
    slicing' and checks all t1 values for violations >= test_start.

    Setup: all t1 values resolve BEFORE test_start → ok=True, n_violations=0.
    """
    # Integer index (not DatetimeIndex)
    n = 20
    t1_values = pd.date_range("2024-01-01", periods=n, freq="1h")
    t1 = pd.Series(t1_values, index=range(n))  # integer index

    train_end = pd.Timestamp("2024-01-01 10:00")
    test_start = pd.Timestamp("2024-01-01 20:00")  # all t1 values are before this

    audit = causal_t1_audit(t1, train_end=train_end, test_start=test_start)
    assert audit["ok"] is True, (
        "Non-DatetimeIndex: all t1 before test_start → no violations → ok=True"
    )
    assert audit["n_violations"] == 0


def test_ec13b_non_datetime_index_with_violations():
    """With integer index, violations are still detected from t1 values."""
    n = 10
    # t1 resolves 48h into the future for every bar — all cross test_start.
    base = pd.Timestamp("2024-01-01")
    t1_values = [base + pd.Timedelta(hours=48 + i) for i in range(n)]
    t1 = pd.Series(t1_values, index=range(n))  # integer index

    train_end = pd.Timestamp("2024-01-01 06:00")
    test_start = pd.Timestamp("2024-01-02 00:00")  # t1 values are at 48h+ → after test_start

    audit = causal_t1_audit(t1, train_end=train_end, test_start=test_start)
    assert audit["ok"] is False
    assert audit["n_violations"] == n


# ---------------------------------------------------------------------------
# EC-14  purge_overlapping_train with no violations — returns input unchanged
# ---------------------------------------------------------------------------

def test_ec14_purge_no_violations_returns_unchanged():
    """When audit is clean (ok=True) purge returns the SAME df object (identity)."""
    n = 30
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"close": np.ones(n), "feature": np.arange(n)}, index=ts)

    # t1 resolves 2h after each observation — all well before test_start.
    t1 = pd.Series(ts + pd.Timedelta(hours=2), index=ts)
    train_end = ts[20]
    test_start = ts[25]  # large gap: no t1 crosses into [test_start, ...)

    result = purge_overlapping_train(df, t1, train_end=train_end, test_start=test_start)
    # audit ok=True → df returned directly (no copy).
    assert result is df, (
        "purge_overlapping_train: no violations → must return the original df object"
    )


def test_ec14b_purge_with_violations_removes_rows():
    """When some t1 values cross test_start, those rows are dropped."""
    n = 20
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"close": np.ones(n)}, index=ts)

    # Rows 15–19: t1 resolves at ts[15]+24h = well into test window.
    t1_values = list(ts + pd.Timedelta(hours=2))  # default: 2h ahead
    for i in range(15, n):
        t1_values[i] = ts[i] + pd.Timedelta(hours=24)  # crosses test_start
    t1 = pd.Series(t1_values, index=ts)

    train_end = ts[18]
    test_start = ts[19]  # rows with t1 >= test_start must be purged

    result = purge_overlapping_train(df, t1, train_end=train_end, test_start=test_start)
    assert len(result) < len(df), "Rows with overlapping t1 must be removed"
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# EC-EXTRA  Return shapes and names are always consistent
# ---------------------------------------------------------------------------

def test_return_series_names_and_index_preserved():
    """Labels and t1_times must carry the correct .name and share the input index."""
    n = 30
    ts = pd.date_range("2024-06-01", periods=n, freq="30min")
    df = pd.DataFrame(
        {
            "close": np.full(n, 200.0),
            "high": np.full(n, 200.5),
            "low": np.full(n, 199.5),
            "atr_14": np.full(n, 1.0),
        },
        index=ts,
    )
    labels, t1 = triple_barrier_labels_vectorized(df, max_bars=5)
    assert labels.name == "triple_barrier_label"
    assert t1.name == "t1_timestamp"
    assert labels.index.equals(df.index), "Labels index must match input DataFrame index"
    assert t1.index.equals(df.index), "t1_times index must match input DataFrame index"


if __name__ == "__main__":
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
