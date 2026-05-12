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

import io
import logging
from src.utils.model_integrity import verify_and_load_bytes
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
            self.model = joblib.load(io.BytesIO(verify_and_load_bytes(self.model_path)))
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
            # Normalize input to a flat dict regardless of caller form.
            feature_row: dict = {}
            if isinstance(features, pd.Series):
                feature_row = features.to_dict()
            elif isinstance(features, list) and len(features) > 0 and isinstance(features[-1], dict):
                feature_row = features[-1]
            elif isinstance(features, dict):
                feature_row = dict(features)

            # `prob_base`, `prob_trend`, `regime` come from upstream primary
            # models. When the caller hasn't run them, fail open with neutral
            # priors so the meta-labeler still scores the technical features
            # (rsi/macd/bb/ofi/…) instead of bypassing the gate entirely.
            missing = [f for f in ('prob_base', 'prob_trend', 'regime') if f not in feature_row]
            if missing:
                if not getattr(self, '_warned_missing_probs', False):
                    logger.info(
                        "Meta-labeler scoring with neutral priors for %s — "
                        "callers can supply these for more accurate filtering.",
                        ', '.join(missing),
                    )
                    self._warned_missing_probs = True
                feature_row.setdefault('prob_base', 0.5)
                feature_row.setdefault('prob_trend', 0.5)
                feature_row.setdefault('regime', 0)

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
