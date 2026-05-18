"""
Unified configuration for meta-labeler — single source of truth for the feature
list and confidence threshold. Both training (`src/engine/train_meta_labeler.py`)
and inference (`src/analysis/meta_labeler.py`) MUST import from this module.

Adding a new feature is a two-step process:
  1. Add it to META_FEATURES below.
  2. Retrain the meta-labeler so the saved model's `n_features_in_` matches.

Per AFML Ch.3 (Lopez de Prado) and the BryceMeng/mlfinlab_research_bryce playbook,
the feature set must include:
  - primary model outputs (prob_base, prob_trend, regime)
  - the primary signal itself (primary_signal) — meta-labeler is conditional on it
  - microstructure features (ofi_z, vwap_dist, taker_buy_ratio)
  - macro/derivatives features (funding_z, funding_positive, liq_proximity)
  - regime/volatility features (atr_pct, volatility_7, kc_width, don_pos_20)
  - stationarity (frac_diff_d40) — d-value found per asset via ADF sweep
  - timestamp features (hour, day_of_week)
  - per-rule signals so the meta-labeler learns which rule produced the signal
"""
from __future__ import annotations

META_FEATURES: list[str] = [
    # Primary model outputs (upstream)
    'prob_base',
    'prob_trend',
    'regime',
    'primary_signal',
    # Stationarity & volatility
    'frac_diff_d40',
    'volatility_7',
    'atr_pct',
    # Technical features
    'rsi_14',
    'macd_hist',
    'bb_pb',
    'kc_width',
    'don_pos_20',
    # Microstructure
    'ofi_z',
    'vwap_dist',
    'taker_buy_ratio',
    # Derivatives / macro
    'funding_z',
    'funding_positive',
    'liq_proximity',
    # Timestamp
    'hour',
    'day_of_week',
    # Per-rule signal flags (meta-labeler learns which rule fired)
    'signal_rsi',
    'signal_macd',
    'signal_bb',
    # Symbol identity
    'symbol_id',        # ordinal symbol encoding (1-based; 0 = unknown)
    # Explicit regime one-hot features
    'regime_bull', 'regime_bear', 'regime_chop', 'regime_high_vol',
]

# Default confidence threshold — overridden by `optimal_threshold` stored in
# each model's meta JSON after walk-forward Sortino search.
CONFIDENCE_THRESHOLD: float = 0.60

# Sortino-ratio threshold search range used by train_meta_labeler.py during
# walk-forward optimisation. Stored here so the same range is used by the
# CIO Agent (Optuna) when sweeping per-model thresholds.
THRESHOLD_SEARCH_RANGE: tuple[float, float] = (0.40, 0.70)
THRESHOLD_SEARCH_STEP: float = 0.01
