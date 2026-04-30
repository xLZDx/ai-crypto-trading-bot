"""
Meta-Labeler inference wrapper.

Usage in live trading and backtester:
    ml = MetaLabeler()
    decision, confidence = ml.filter(signal=1.0, features_row=df.iloc[-1])
    if decision == 'PASS':
        execute_trade(...)

The meta-labeler is stateless at inference — just loads the trained model and
runs predict_proba. It is fast enough to call on every candle.
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

META_FEATURES = [
    'frac_diff_d40',
    'volatility_7',
    'rsi_14',
    'macd_hist',
    'bb_pb',
    'ofi_z',
    'vwap_dist',
    'funding_z',
    'funding_positive',
    'liq_proximity',
    'kc_width',
    'don_pos_20',
    'hour', 'day_of_week',
    'taker_buy_ratio',
    'atr_pct',
    'primary_signal',
    'signal_rsi',
    'signal_macd',
    'signal_bb',
]

CONFIDENCE_THRESHOLD = 0.60  # block trade if P(win) < this


class MetaLabeler:
    """
    Second-layer signal filter.
    Loads the pre-trained meta_labeler.joblib and evaluates incoming signals.
    """

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'models', 'meta_labeler.joblib')
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning("Meta-labeler model not found at %s. Filter disabled (all signals pass).",
                           self.model_path)
            return
        try:
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            logger.info("Meta-labeler loaded from %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load meta-labeler: %s", e)

    def filter(
        self,
        signal: float,
        features: dict | pd.Series,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> Tuple[str, float]:
        """
        Evaluate whether a primary strategy signal should be acted on.

        Args:
            signal:    Primary signal value (+1 long, -1 short, 0 flat).
            features:  Dict or Series with the META_FEATURES keys at signal bar.
            threshold: Minimum P(win) to approve the trade (default 0.60).

        Returns:
            ('PASS', confidence) if trade should be taken.
            ('BLOCK', confidence) if meta-labeler rejects it.
        """
        if not self.is_loaded or self.model is None:
            return 'PASS', 0.5  # disabled — let everything through

        if abs(signal) < 0.1:
            return 'BLOCK', 0.0  # no signal to filter

        try:
            # This is complex in live mode. We need to run primary models first.
            # The `main.py` orchestrator should provide these probabilities.
            # For now, we assume `features` dict contains `prob_base`, `prob_trend`, etc.
            if not all(f in features for f in ['prob_base', 'prob_trend', 'regime']):
                logger.warning("Meta-labeler missing required probability/regime features. Passing signal.")
                return 'PASS', 0.5

            # Ensure all features are present
            feature_row = {}
            if isinstance(features, pd.Series):
                feature_row = features.to_dict()
            elif isinstance(features, list) and len(features) > 0 and isinstance(features[-1], dict):
                # Handle list of dicts from data loader
                feature_row = features[-1]
            elif isinstance(features, dict):
                feature_row = features

            for f in META_FEATURES:
                if f not in feature_row:
                    feature_row[f] = 0.0

            X = pd.DataFrame([feature_row])[META_FEATURES].fillna(0)
            proba = self.model.predict_proba(X)[0]
            # Index 1 = probability of win (class 1)
            p_win = float(proba[1]) if len(proba) > 1 else float(proba[0])

            decision = 'PASS' if p_win >= threshold else 'BLOCK'
            return decision, round(p_win, 4)

        except Exception as e:
            logger.debug("Meta-labeler inference error: %s", e)
            return 'PASS', 0.5  # fail open

    def batch_filter(
        self,
        signals: pd.Series,
        features_df: pd.DataFrame,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Vectorized version for backtester use.
        Returns DataFrame with columns: ['decision', 'confidence', 'filtered_signal'].
        filtered_signal = original signal if PASS, 0.0 if BLOCK.
        """
        if not self.is_loaded or self.model is None:
            out = pd.DataFrame({
                'decision': 'PASS',
                'confidence': 0.5,
                'filtered_signal': signals,
            }, index=features_df.index)
            return out

        try:
            feat_df = features_df.copy()
            # In backtesting, we assume the features_df already contains
            # 'prob_base', 'prob_trend', 'regime' etc. from a pre-computation step.

            missing = [f for f in META_FEATURES if f not in feat_df.columns]
            for col in missing:
                feat_df[col] = 0.0
            if missing:
                logger.warning("Meta-labeler backtest missing features, filled with 0: %s", missing)

            X = feat_df[META_FEATURES].fillna(0)
            proba = self.model.predict_proba(X)
            p_win = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

            decisions = np.where(p_win >= threshold, 'PASS', 'BLOCK')
            filtered = np.where(p_win >= threshold, signals.values, 0.0)

            return pd.DataFrame({
                'decision': decisions,
                'confidence': p_win.round(4),
                'filtered_signal': filtered,
            }, index=features_df.index)

        except Exception as e:
            logger.error("Meta-labeler batch filter error: %s", e)
            return pd.DataFrame({
                'decision': 'PASS',
                'confidence': 0.5,
                'filtered_signal': signals,
            }, index=features_df.index)
