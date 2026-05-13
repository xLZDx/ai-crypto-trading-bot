"""
Tests for src/analysis/microstructure.py — Phase X2 features.

Covers the three causal microstructure features:
  - Amihud illiquidity: rolling |return| / dollar_volume
  - Kyle's lambda: rolling OLS slope of return on signed volume
  - VPIN: volume-synchronized flow toxicity

All three MUST be causal: a value at row i must only depend on rows 0..i.
This is verified by running the function on a frame and confirming that
prepending future rows doesn't change the historical values.
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


def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    volume = np.abs(rng.normal(100.0, 20.0, n))
    taker = np.clip(rng.normal(0.5, 0.15, n), 0.0, 1.0)
    return pd.DataFrame({'close': close, 'volume': volume, 'taker_buy_ratio': taker})


# ── Amihud illiquidity ──────────────────────────────────────────────────────

def test_amihud_appends_column():
    from src.analysis.microstructure import add_amihud_illiquidity
    df = _make_df()
    out = add_amihud_illiquidity(df, window=10)
    assert 'amihud_illiq' in out.columns
    assert len(out) == len(df)
    assert out['amihud_illiq'].notna().all()


def test_amihud_is_causal():
    """Value at row i must not change when future rows are added/dropped."""
    from src.analysis.microstructure import add_amihud_illiquidity
    df = _make_df(n=200)
    full = add_amihud_illiquidity(df, window=10)
    truncated = add_amihud_illiquidity(df.iloc[:120], window=10)
    # The first 120 rows must match between the two runs.
    pd.testing.assert_series_equal(
        full['amihud_illiq'].iloc[:120],
        truncated['amihud_illiq'],
        check_names=False,
    )


def test_amihud_zero_volume_robust():
    """Zero dollar volume rows must not crash the rolling mean."""
    from src.analysis.microstructure import add_amihud_illiquidity
    df = _make_df()
    df.loc[10:15, 'volume'] = 0.0
    out = add_amihud_illiquidity(df, window=10)
    assert out['amihud_illiq'].notna().all()


# ── Kyle's lambda ───────────────────────────────────────────────────────────

def test_kyle_lambda_appends_column():
    from src.analysis.microstructure import add_kyle_lambda
    df = _make_df()
    out = add_kyle_lambda(df, window=30)
    assert 'kyle_lambda' in out.columns
    assert len(out) == len(df)
    # Some values will be NaN at the start due to rolling min_periods;
    # we backfill to 0, so the final column has no NaN.
    assert out['kyle_lambda'].notna().all()


def test_kyle_lambda_is_causal():
    from src.analysis.microstructure import add_kyle_lambda
    df = _make_df(n=200)
    full = add_kyle_lambda(df, window=30)
    truncated = add_kyle_lambda(df.iloc[:120], window=30)
    pd.testing.assert_series_equal(
        full['kyle_lambda'].iloc[:120],
        truncated['kyle_lambda'],
        check_names=False,
    )


def test_kyle_lambda_without_taker_ratio():
    """Falls back to sign(return) * volume when taker_buy_ratio is absent."""
    from src.analysis.microstructure import add_kyle_lambda
    df = _make_df().drop(columns=['taker_buy_ratio'])
    out = add_kyle_lambda(df, window=30)
    assert 'kyle_lambda' in out.columns


# ── VPIN ────────────────────────────────────────────────────────────────────

def test_vpin_appends_column_bounded():
    """VPIN values must be in [0, 1]."""
    from src.analysis.microstructure import add_vpin
    df = _make_df()
    out = add_vpin(df, n_buckets=20)
    assert 'vpin' in out.columns
    assert (out['vpin'] >= 0.0).all()
    assert (out['vpin'] <= 1.0001).all()  # tiny float tolerance


def test_vpin_is_causal():
    from src.analysis.microstructure import add_vpin
    df = _make_df(n=200)
    full = add_vpin(df, n_buckets=20)
    truncated = add_vpin(df.iloc[:120], n_buckets=20)
    pd.testing.assert_series_equal(
        full['vpin'].iloc[:120],
        truncated['vpin'],
        check_names=False,
    )


def test_vpin_extreme_flow_high():
    """An all-buy bar sequence should drive VPIN toward 1.0 (extreme imbalance)."""
    from src.analysis.microstructure import add_vpin
    df = _make_df()
    df['taker_buy_ratio'] = 1.0   # every bar is 100% buyer-initiated
    out = add_vpin(df, n_buckets=10)
    # After warmup, VPIN should be ~ 1.0.
    assert out['vpin'].iloc[-1] > 0.9


def test_vpin_balanced_flow_low():
    """50/50 flow should drive VPIN toward 0.0."""
    from src.analysis.microstructure import add_vpin
    df = _make_df()
    df['taker_buy_ratio'] = 0.5
    out = add_vpin(df, n_buckets=10)
    assert out['vpin'].iloc[-1] < 0.1


# ── Combined convenience ────────────────────────────────────────────────────

def test_add_all_microstructure():
    from src.analysis.microstructure import add_all_microstructure
    df = _make_df()
    out = add_all_microstructure(df)
    for col in ('amihud_illiq', 'kyle_lambda', 'vpin'):
        assert col in out.columns
        assert out[col].notna().all()


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
