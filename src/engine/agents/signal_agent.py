"""
SignalAgent — Technical Analyst.

Responsibilities:
  - Runs all strategy signals in real-time on every new candle
  - Applies RegimeClassifier to route signals to the appropriate strategy
  - Passes signals through MetaLabeler filter
  - Publishes final directional signal with confidence score
  - Publishes regime changes

Output message format (topic='signal'):
  {
    'symbol': 'BTC_USDT',
    'direction': 1 | -1 | 0,
    'confidence': 0.72,
    'strategy': 'MACD_Momentum',
    'regime': 1,
    'regime_name': 'TRENDING',
    'meta_pass': True,
    'raw_signals': {'signal_rsi': 0, 'signal_macd': 1, ...}
  }
"""
from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from src.engine.agents.agent_bus import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

CONFIDENCE_THRESHOLD = 0.60


class SignalAgent(BaseAgent):
    NAME = "SignalAgent"

    def __init__(self, symbols: list[str], data_getter, bus=None, interval_sec: float = 60.0):
        """
        Args:
            symbols:     List of symbol strings e.g. ['BTC_USDT', 'ETH_USDT']
            data_getter: Callable(symbol) → pd.DataFrame of recent OHLCV bars
        """
        super().__init__(bus=bus, interval_sec=interval_sec)
        self.symbols = symbols
        self.data_getter = data_getter
        self._regime_classifier = None
        self._meta_labeler = None
        self._last_regimes: dict[str, int] = {}
        self._init_models()

    def _init_models(self) -> None:
        try:
            from src.analysis.regime_classifier import RegimeClassifier
            self._regime_classifier = RegimeClassifier()
        except Exception as e:
            logger.warning("[SignalAgent] Regime classifier not available: %s", e)

        try:
            from src.analysis.meta_labeler import MetaLabeler
            self._meta_labeler = MetaLabeler()
        except Exception as e:
            logger.warning("[SignalAgent] Meta-labeler not available: %s", e)

    def _compute_signals(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute all rule-based strategy signals on df."""
        from src.analysis.feature_engineering import (
            add_rsi, add_macd, add_bollinger_bands, add_ofi,
            add_vwap, add_donchian, add_keltner, add_funding_zscore
        )
        try:
            df = add_rsi(df, 14)
            df = add_macd(df)
            df = add_bollinger_bands(df)
            df = add_ofi(df)
            df = add_vwap(df)
            df = add_donchian(df, n=20)
            df = add_keltner(df)
            df = add_funding_zscore(df)
        except Exception as e:
            logger.debug("[SignalAgent] Feature error: %s", e)

        last = df.iloc[-1]
        signals = {}

        try:
            rsi = float(last.get("rsi_14", 50))
            signals["signal_rsi"] = 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.0)
        except Exception:
            signals["signal_rsi"] = 0.0

        try:
            signals["signal_macd"] = 1.0 if float(last.get("macd_hist", 0)) > 0 else -1.0
        except Exception:
            signals["signal_macd"] = 0.0

        try:
            bb_pb = float(last.get("bb_pb", 0.5))
            signals["signal_bb"] = 1.0 if bb_pb < 0.1 else (-1.0 if bb_pb > 0.9 else 0.0)
        except Exception:
            signals["signal_bb"] = 0.0

        try:
            vwap_d = float(last.get("vwap_dist", 0))
            signals["signal_vwap"] = 1.0 if vwap_d < -0.005 else (-1.0 if vwap_d > 0.005 else 0.0)
        except Exception:
            signals["signal_vwap"] = 0.0

        try:
            don_pos = float(last.get("don_pos_20", 0.5))
            signals["signal_donchian"] = 1.0 if don_pos > 0.98 else (-1.0 if don_pos < 0.02 else 0.0)
        except Exception:
            signals["signal_donchian"] = 0.0

        try:
            funding = float(last.get("funding_rate", 0))
            signals["signal_funding"] = -1.0 if funding > 0.001 else (1.0 if funding < -0.0005 else 0.0)
        except Exception:
            signals["signal_funding"] = 0.0

        return signals

    def _select_strategy(self, signals: dict, regime: int) -> tuple[str, float]:
        """
        Select best strategy for current regime and return (strategy_name, signal_value).
        """
        from src.analysis.regime_classifier import REGIME_STRATEGY_MAP

        approved = REGIME_STRATEGY_MAP.get(regime, list(signals.keys()))
        strategy_signal_map = {
            "RSI_MeanReversion": signals.get("signal_rsi", 0),
            "BB_Reversion": signals.get("signal_bb", 0),
            "VWAP_Reversion": signals.get("signal_vwap", 0),
            "MACD_Momentum": signals.get("signal_macd", 0),
            "Donchian_Breakout": signals.get("signal_donchian", 0),
            "Keltner_Breakout": signals.get("signal_donchian", 0),
            "Funding_Arb": signals.get("signal_funding", 0),
        }

        # Pick strongest non-zero approved signal
        best_strat, best_val = "Ensemble", 0.0
        for strat in approved:
            val = strategy_signal_map.get(strat, 0.0)
            if abs(val) > abs(best_val):
                best_strat, best_val = strat, val

        # Fallback to ensemble if no approved signal fires
        if best_val == 0.0:
            vals = [strategy_signal_map.get(s, 0) for s in approved]
            best_val = sum(vals) / len(vals) if vals else 0.0
            best_strat = "Ensemble"

        return best_strat, float(best_val)

    def _run_cycle(self) -> None:
        for sym in self.symbols:
            try:
                df = self.data_getter(sym)
                if df is None or len(df) < 50:
                    continue

                # Regime
                regime = 0
                if self._regime_classifier and self._regime_classifier.is_ready:
                    regime = self._regime_classifier.predict(df)

                if self._last_regimes.get(sym) != regime:
                    from src.analysis.regime_classifier import RegimeClassifier
                    self._last_regimes[sym] = regime
                    self.publish("regime", {
                        "symbol": sym,
                        "regime": regime,
                        "regime_name": RegimeClassifier.regime_name(regime),
                    })
                    logger.info("[SignalAgent] %s regime → %s",
                                sym, RegimeClassifier.regime_name(regime))

                # Signals
                signals = self._compute_signals(df)
                strategy, raw_signal = self._select_strategy(signals, regime)

                if abs(raw_signal) < 0.1:
                    continue

                # Meta-labeler filter
                confidence = 0.5
                meta_pass = True
                if self._meta_labeler and self._meta_labeler.is_loaded:
                    decision, confidence = self._meta_labeler.filter(
                        raw_signal, df.iloc[-1].to_dict()
                    )
                    meta_pass = decision == "PASS"

                from src.analysis.regime_classifier import RegimeClassifier
                self.publish("signal", {
                    "symbol": sym,
                    "direction": int(round(raw_signal)),
                    "confidence": confidence,
                    "strategy": strategy,
                    "regime": regime,
                    "regime_name": RegimeClassifier.regime_name(regime),
                    "meta_pass": meta_pass,
                    "raw_signals": signals,
                    "size_mult": RegimeClassifier.size_multiplier(regime),
                })

                if meta_pass:
                    logger.info(
                        "[SignalAgent] %s | %s | dir=%+d | conf=%.2f | regime=%s",
                        sym, strategy, int(round(raw_signal)), confidence,
                        RegimeClassifier.regime_name(regime)
                    )

            except Exception as e:
                logger.error("[SignalAgent] Error processing %s: %s", sym, e)
