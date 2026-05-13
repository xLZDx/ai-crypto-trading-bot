"""
Microstructure features — Phase X2.

Three causal, rolling-window features that add information value to the
existing kline + L2 feature set:

  - VPIN  : Volume-synchronized Probability of Informed Trading
            (Easley, López de Prado, O'Hara 2012).
            Spikes when whales hit the book → leading indicator of
            volatility expansion.

  - Kyle's lambda : price impact per unit of signed volume.
            Higher λ = less liquid market = bigger moves on small flow.

  - Amihud illiquidity : |return| / dollar_volume
            Simple, robust illiquidity proxy.

All three are computed CAUSALLY — every value at index i uses only data
at indices ≤ i. Verified by `feature_engineering.causal_audit`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Volume-floor to avoid division-by-zero on stale bars.
_EPS = 1e-12


def add_amihud_illiquidity(
    df: pd.DataFrame,
    window: int = 30,
    *,
    out_col: str = 'amihud_illiq',
) -> pd.DataFrame:
    """Append a rolling Amihud illiquidity column.

        amihud_i = |return_i| / dollar_volume_i
        feature = rolling mean over `window` bars

    Robust to: zero dollar volume rows (returns NaN that's forward-filled
    by `feature_engineering` downstream), missing 'return' column (computed
    from close if absent).

    Reference: Amihud, Y. (2002). "Illiquidity and stock returns."
    """
    out = df.copy()
    if 'return' not in out.columns:
        out['return'] = out['close'].pct_change().fillna(0.0)
    dollar_vol = (out['close'] * out['volume']).replace(0.0, np.nan)
    raw = (out['return'].abs() / dollar_vol).fillna(0.0)
    # Rolling mean smooths out single-bar spikes.
    out[out_col] = raw.rolling(window, min_periods=max(1, window // 4)).mean().fillna(0.0)
    return out


def add_kyle_lambda(
    df: pd.DataFrame,
    window: int = 60,
    *,
    out_col: str = 'kyle_lambda',
) -> pd.DataFrame:
    """Append rolling Kyle's lambda — price impact per unit of signed volume.

        lambda_t = OLS slope of  return_i  ON  signed_volume_i
                   over a rolling window ending at t

    Causal: each rolling window ends at the current bar (no look-ahead).
    The signed volume proxy uses (taker_buy_volume - taker_sell_volume) when
    `taker_buy_ratio` is present, else falls back to sign(return)*volume.

    Reference: Kyle, A. (1985). "Continuous auctions and insider trading."
    """
    out = df.copy()
    if 'return' not in out.columns:
        out['return'] = out['close'].pct_change().fillna(0.0)

    if 'taker_buy_ratio' in out.columns:
        # taker_buy_ratio is in [0, 1] — center it and scale by volume.
        signed_vol = (2.0 * out['taker_buy_ratio'] - 1.0) * out['volume']
    else:
        signed_vol = np.sign(out['return']) * out['volume']

    # Rolling OLS via sliding covariance / variance — fastest correct form.
    ret = out['return']
    cov = ret.rolling(window, min_periods=max(2, window // 4)).cov(signed_vol)
    var = signed_vol.rolling(window, min_periods=max(2, window // 4)).var()
    var_safe = var.where(var > _EPS, np.nan)
    out[out_col] = (cov / var_safe).fillna(0.0)
    return out


def add_vpin(
    df: pd.DataFrame,
    bucket_volume: float | None = None,
    n_buckets: int = 50,
    *,
    out_col: str = 'vpin',
) -> pd.DataFrame:
    """Append Volume-synchronized Probability of Informed Trading (VPIN).

        VPIN = mean over last n_buckets of  |V_buy - V_sell| / (V_buy + V_sell)

    This is a SIMPLIFIED VPIN that uses bar-level volume + a tick-rule sign
    proxy instead of true tick-level buy/sell classification. The original
    Easley/Lopez de Prado/O'Hara formulation requires tick data; this version
    is the standard kline-level approximation used in production crypto
    quant pipelines.

    bucket_volume: target per-bucket volume. None → auto-set to the median
                   of the rolling-30 volume (so a typical bar fills ~1 bucket).
    n_buckets:     number of historical buckets to average over.

    Returns:
        DataFrame with `out_col` ∈ [0, 1]. 0 = perfectly balanced flow,
        1 = entirely one-sided (informed traders).

    Reference: Easley, López de Prado, O'Hara (2012). "Flow Toxicity and
               Liquidity in a High-Frequency World."
    """
    out = df.copy()
    if 'return' not in out.columns:
        out['return'] = out['close'].pct_change().fillna(0.0)

    # Auto-pick bucket size if not specified.
    if bucket_volume is None or bucket_volume <= 0:
        bv = out['volume'].rolling(30, min_periods=1).median()
        bucket_volume = float(bv.median()) or 1.0

    if 'taker_buy_ratio' in out.columns:
        buy_frac = out['taker_buy_ratio'].clip(0.0, 1.0)
    else:
        # Tick-rule proxy: positive return ⇒ buyer-initiated.
        buy_frac = (out['return'] > 0).astype(float)

    v_buy  = out['volume'] * buy_frac
    v_sell = out['volume'] * (1.0 - buy_frac)

    # Per-bar imbalance ratio.
    denom = (v_buy + v_sell).replace(0.0, np.nan)
    bar_imbal = (v_buy - v_sell).abs() / denom
    # Volume-weighted rolling mean over the last n_buckets bars.
    out[out_col] = bar_imbal.rolling(n_buckets, min_periods=max(1, n_buckets // 4)).mean().fillna(0.0)
    return out


def add_all_microstructure(
    df: pd.DataFrame,
    *,
    amihud_window: int = 30,
    kyle_window: int = 60,
    vpin_buckets: int = 50,
) -> pd.DataFrame:
    """Convenience: append all three features in one call. Used by the
    trainer feature-engineering pipelines so a single import covers Phase
    X2 microstructure additions."""
    out = add_amihud_illiquidity(df, window=amihud_window)
    out = add_kyle_lambda(out, window=kyle_window)
    out = add_vpin(out, n_buckets=vpin_buckets)
    return out
