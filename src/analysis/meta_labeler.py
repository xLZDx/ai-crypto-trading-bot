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

from src.utils.meta_config import META_FEATURES, CONFIDENCE_THRESHOLD


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
        self.confidence_threshold: float = CONFIDENCE_THRESHOLD
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            logger.error(
                "Meta-labeler model not found at %s -- second-layer filter is "
                "DISABLED and will BLOCK all trades until model is trained.",
                self.model_path,
            )
            return
        try:
            self.model = joblib.load(io.BytesIO(verify_and_load_bytes(self.model_path)))
            self.is_loaded = True
            logger.info("Meta-labeler loaded from %s", self.model_path)
            # Load the paired meta JSON to pick up the calibrated optimal_threshold.
            meta_json_path = self.model_path.replace('.joblib', '_meta.json')
            if os.path.exists(meta_json_path):
                try:
                    import json
                    with open(meta_json_path, 'r', encoding='utf-8') as fh:
                        meta = json.load(fh)
                    threshold = meta.get('optimal_threshold') or meta.get('confidence_threshold')
                    if threshold is not None:
                        self.confidence_threshold = float(threshold)
                        logger.info(
                            "Meta-labeler confidence threshold loaded from meta JSON: %.4f",
                            self.confidence_threshold,
                        )
                except Exception as e:
                    logger.warning(
                        "Could not load optimal_threshold from %s: %s -- using default %.2f",
                        meta_json_path, e, self.confidence_threshold,
                    )
        except Exception as e:
            logger.error("Failed to load meta-labeler: %s", e, exc_info=True)

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
            logger.error(
                "Meta-labeler model is not loaded (path=%s, is_loaded=%s) -- "
                "BLOCKING trade. The second-layer filter is non-functional.",
                self.model_path, self.is_loaded,
            )
            return 'BLOCK', 0.0  # fail-CLOSED: no model → no trade

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
            # models. When missing, we substitute neutral priors but WARN on
            # every call (no suppression) so the operator sees the degraded
            # filtering state in the logs.
            missing = [f for f in ('prob_base', 'prob_trend', 'regime') if f not in feature_row]
            if missing:
                logger.warning(
                    "Meta-labeler: primary-model features missing (%s) -- "
                    "scoring with neutral priors. Caller should supply these "
                    "for accurate filtering.",
                    ', '.join(missing),
                )
                feature_row.setdefault('prob_base', 0.5)
                feature_row.setdefault('prob_trend', 0.5)
                feature_row.setdefault('regime', 0)

            for f in META_FEATURES:
                if f not in feature_row:
                    feature_row[f] = 0.0

            X = pd.DataFrame([feature_row])[META_FEATURES].fillna(0)
            # Sanity check — guard the silent feature-count drift that BUG-3 hid
            expected = getattr(self.model, 'n_features_in_', None)
            if expected is not None and X.shape[1] != expected:
                logger.error(
                    "Meta-labeler feature count mismatch: model expects %d, got %d. "
                    "BLOCKING trade. (META_FEATURES probably drifted from training time.)",
                    expected, X.shape[1],
                )
                return 'BLOCK', 0.0

            proba = self.model.predict_proba(X)[0]
            # Index 1 = probability of win (class 1)
            p_win = float(proba[1]) if len(proba) > 1 else float(proba[0])

            # Use instance threshold if loaded from meta JSON, else caller-supplied default
            effective_threshold = getattr(self, 'confidence_threshold', None) or threshold
            decision = 'PASS' if p_win >= effective_threshold else 'BLOCK'
            return decision, round(p_win, 4)

        except Exception as e:
            logger.error(
                "Meta-labeler inference FAILED -- BLOCKING trade: %s",
                e, exc_info=True,
            )
            return 'BLOCK', 0.0  # fail-CLOSED on any exception

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
            logger.error(
                "Meta-labeler batch_filter: model not loaded -- BLOCKING all signals."
            )
            out = pd.DataFrame({
                'decision': 'BLOCK',
                'confidence': 0.0,
                'filtered_signal': 0.0,
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
            # Sanity check — guard against feature-count drift
            expected = getattr(self.model, 'n_features_in_', None)
            if expected is not None and X.shape[1] != expected:
                logger.error(
                    "Meta-labeler batch: feature count mismatch -- model expects %d, "
                    "got %d. BLOCKING all signals.",
                    expected, X.shape[1],
                )
                return pd.DataFrame({
                    'decision': 'BLOCK',
                    'confidence': 0.0,
                    'filtered_signal': 0.0,
                }, index=features_df.index)

            proba = self.model.predict_proba(X)
            p_win = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

            effective_threshold = getattr(self, 'confidence_threshold', None) or threshold
            decisions = np.where(p_win >= effective_threshold, 'PASS', 'BLOCK')
            filtered = np.where(p_win >= effective_threshold, signals.values, 0.0)

            return pd.DataFrame({
                'decision': decisions,
                'confidence': p_win.round(4),
                'filtered_signal': filtered,
            }, index=features_df.index)

        except Exception as e:
            logger.error(
                "Meta-labeler batch_filter FAILED -- BLOCKING all signals: %s",
                e, exc_info=True,
            )
            # Fail-CLOSED: return all-BLOCK so backtester treats this window as
            # entirely filtered (correct conservative behaviour vs all-PASS bug).
            return pd.DataFrame({
                'decision': 'BLOCK',
                'confidence': -1.0,  # sentinel: caller can detect error state
                'filtered_signal': 0.0,
            }, index=features_df.index)
