"""
SpotAgent — Spot Market Specialist.

Timeframe: 1h–4h
Models: Base model + Trend model
Strategies: Trend following, momentum, cross-sectional ranking
Risk: Drawdown-capped, no leverage
Fees: Binance spot (0.10% taker, 0.075% with BNB)

This agent operates at the slowest cadence of the three market agents.
It ignores 1m noise and focuses on multi-hour directional moves.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from src.engine.agents.agent_bus import BaseAgent, get_bus

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

CONFIDENCE_THRESHOLD = 0.62
MAX_HOLD_HOURS = 72
FEE_RATE = 0.001   # Binance spot taker


class SpotAgent(BaseAgent):
    NAME = "SpotAgent"

    def __init__(self, symbols: list[str], data_getter, bus=None,
                 interval_sec: float = 3600.0):
        super().__init__(bus=bus, interval_sec=interval_sec)
        self.symbols = symbols
        self.data_getter = data_getter
        self._base_model = None
        self._trend_model = None
        self._open_positions: dict[str, dict] = {}
        self._load_models()

    def _load_models(self) -> None:
        try:
            import io
            import joblib
            from src.utils.model_integrity import verify_and_load_bytes
            models_dir = os.path.join(PROJECT_ROOT, "models")
            base_path = os.path.join(models_dir, "btc_rf_model.joblib")
            trend_path = os.path.join(models_dir, "trend_model.joblib")
            if os.path.exists(base_path):
                self._base_model = joblib.load(io.BytesIO(verify_and_load_bytes(base_path)))
                logger.info("[SpotAgent] Base model loaded.")
            if os.path.exists(trend_path):
                self._trend_model = joblib.load(io.BytesIO(verify_and_load_bytes(trend_path)))
                logger.info("[SpotAgent] Trend model loaded.")
        except Exception as e:
            logger.warning("[SpotAgent] Model load error: %s", e)

    def _setup_subscriptions(self):
        self.bus.subscribe("signal", self._on_signal)
        self.bus.subscribe("regime", self._on_regime)

    def _current_regime(self) -> int:
        msg = self.bus.get_latest("regime")
        return (msg.payload or {}).get("regime", 0) if msg else 0

    def _on_regime(self, msg) -> None:
        payload = msg.payload or {}
        if payload.get("regime") == 2:
            logger.info("[SpotAgent] VOLATILE regime — reducing spot exposure.")

    def _on_signal(self, msg) -> None:
        payload = msg.payload or {}
        sym = payload.get("symbol", "")
        if sym not in self.symbols:
            return
        if not payload.get("meta_pass", True):
            return

        regime = payload.get("regime", 0)
        # Spot agent is active in RANGING (0) and TRENDING (1) — not volatile
        if regime == 2:
            logger.debug("[SpotAgent] Skipping %s in VOLATILE regime.", sym)
            return

        direction = int(payload.get("direction", 0))
        confidence = float(payload.get("confidence", 0.5))

        if direction == 0 or confidence < CONFIDENCE_THRESHOLD:
            return

        # Validate with spot-specific model if available
        ml_confidence = confidence
        if self._base_model is not None or self._trend_model is not None:
            ml_confidence = self._get_ml_confidence(sym, direction)
            if ml_confidence < CONFIDENCE_THRESHOLD:
                logger.debug("[SpotAgent] %s ML confidence %.2f too low — skip.", sym, ml_confidence)
                return

        logger.info("[SpotAgent] SIGNAL %s dir=%+d conf=%.2f (spot 1h, regime=%d)",
                    sym, direction, ml_confidence, regime)
        # Pass to RiskAgent via 'trade_signal' (NOT 'signal'). Publishing back
        # on 'signal' would re-enter this very handler synchronously through
        # the bus dispatcher — the 2026-05-13 runaway-orders bug.
        self.publish("trade_signal", {
            **payload,
            "market": "spot",
            "fee_preset": "spot",
            "confidence": ml_confidence,
            "max_hold_hours": MAX_HOLD_HOURS,
        })

    def _get_ml_confidence(self, sym: str, direction: int) -> float:
        """Get calibrated probability from base/trend model."""
        try:
            df = self.data_getter(sym)
            if df is None or len(df) < 50:
                return 0.5

            from src.analysis.ml_predictor import MLPredictor
            predictor = MLPredictor(model_filename="btc_rf_model.joblib", model_type="base")
            if not predictor.is_loaded or predictor.model is None:
                return 0.5

            # Use predict_proba if calibrated model
            from src.analysis.feature_engineering import (
                add_rsi, add_macd, add_bollinger_bands, add_roc, add_time_features,
                add_taker_and_trade_features, add_ofi, add_vwap, add_atr
            )
            from src.analysis.fractional_diff import add_fractional_diff

            df = df.copy()
            df["return"] = df["close"].pct_change()
            df = add_fractional_diff(df, d=0.4)
            df = add_rsi(df, 14)
            df = add_macd(df)
            df = add_bollinger_bands(df)
            df = add_roc(df, [3, 7, 14])
            df = add_time_features(df)
            df = add_taker_and_trade_features(df)
            df = add_atr(df)
            df = add_ofi(df)
            df = add_vwap(df)

            model = predictor.model
            if hasattr(model, "predict_proba"):
                # Build feature row (simplified — use available columns)
                last = df.iloc[[-1]].fillna(0)
                feat_cols = [c for c in last.columns
                             if c not in ("timestamp", "open", "high", "low", "close", "volume")]
                X = last[[c for c in feat_cols if c in last.columns]].fillna(0)
                try:
                    proba = model.predict_proba(X)[0]
                    return float(proba[1]) if len(proba) > 1 else 0.5
                except Exception:
                    return 0.5
        except Exception as e:
            logger.debug("[SpotAgent] ML confidence error: %s", e)
        return 0.5

    def _run_cycle(self) -> None:
        logger.debug("[SpotAgent] heartbeat — %d open positions", len(self._open_positions))
