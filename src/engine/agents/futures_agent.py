"""
FuturesAgent — Futures / Perpetual Market Specialist.

Timeframe: 1h
Models: Futures Short model + funding Z-score signal
Strategies: Funding arbitrage, short signals, liquidation cascade
Risk: Liquidation-aware, uses actual margin math
Fees: Binance USDT-M futures (0.02% maker / 0.04% taker)

Key difference vs SpotAgent:
  - Monitors funding rate as primary signal (high funding → short bias)
  - Tracks open interest for liquidation cascade signals
  - Uses leverage (default 2x) — position size in notional terms
  - Exits on funding reversal, not just price-based signals
"""
from __future__ import annotations

import logging
import os

from src.engine.agents.agent_bus import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

CONFIDENCE_THRESHOLD = 0.60
MAX_HOLD_HOURS = 24
FUNDING_ARB_THRESHOLD = 0.001    # 0.1% funding = strong short signal
LEVERAGE = 2.0                    # conservative for safety
FEE_MAKER = 0.0002
FEE_TAKER = 0.0004


class FuturesAgent(BaseAgent):
    NAME = "FuturesAgent"

    def __init__(self, symbols: list[str], data_getter, bus=None,
                 interval_sec: float = 3600.0):
        super().__init__(bus=bus, interval_sec=interval_sec)
        self.symbols = symbols
        self.data_getter = data_getter
        self._futures_model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import joblib
            path = os.path.join(PROJECT_ROOT, "models", "futures_short_model.joblib")
            if os.path.exists(path):
                self._futures_model = joblib.load(path)
                logger.info("[FuturesAgent] Futures model loaded.")
        except Exception as e:
            logger.warning("[FuturesAgent] Model load error: %s", e)

    def _setup_subscriptions(self):
        self.bus.subscribe("signal", self._on_signal)
        self.bus.subscribe("regime", self._on_regime)

    def _on_regime(self, msg) -> None:
        regime = (msg.payload or {}).get("regime", 0)
        if regime == 2:
            logger.info("[FuturesAgent] VOLATILE — only funding arb allowed.")

    def _on_signal(self, msg) -> None:
        payload = msg.payload or {}
        sym = payload.get("symbol", "")
        if sym not in self.symbols:
            return
        if not payload.get("meta_pass", True):
            return

        regime = payload.get("regime", 0)
        direction = int(payload.get("direction", 0))
        confidence = float(payload.get("confidence", 0.5))

        # In volatile regime, only take funding-arb signals
        raw_signals = payload.get("raw_signals", {})
        funding_signal = float(raw_signals.get("signal_funding", 0))

        if regime == 2 and abs(funding_signal) < 0.5:
            logger.debug("[FuturesAgent] Skipping %s — VOLATILE, no funding signal.", sym)
            return

        if direction == 0 or confidence < CONFIDENCE_THRESHOLD:
            return

        # Liquidity sweep guard
        liq_prox = float(raw_signals.get("liq_proximity", 0))
        if liq_prox > 0.90:
            logger.info("[FuturesAgent] %s BLOCKED — near liquidation cluster (%.2f).", sym, liq_prox)
            return

        # Funding arbitrage override — strong directional edge
        if abs(funding_signal) > 0.5:
            logger.info("[FuturesAgent] %s funding arb signal: dir=%+d funding=%.4f",
                        sym, int(funding_signal), float(raw_signals.get("funding_rate", 0)))

        logger.info("[FuturesAgent] SIGNAL %s dir=%+d conf=%.2f lev=%.1fx (futures 1h)",
                    sym, direction, confidence, LEVERAGE)

        self.publish("signal", {
            **payload,
            "market": "futures",
            "fee_preset": "futures",
            "leverage": LEVERAGE,
            "max_hold_hours": MAX_HOLD_HOURS,
            "confidence": confidence,
        })

    def _run_cycle(self) -> None:
        # Proactively scan for funding arbitrage opportunities every hour
        for sym in self.symbols:
            try:
                df = self.data_getter(sym)
                if df is None or len(df) < 10:
                    continue
                if "funding_rate" not in df.columns:
                    continue
                last_funding = float(df["funding_rate"].iloc[-1])
                if abs(last_funding) > FUNDING_ARB_THRESHOLD:
                    direction = -1 if last_funding > 0 else 1
                    logger.info("[FuturesAgent] Funding arb opportunity: %s funding=%.4f dir=%+d",
                                sym, last_funding, direction)
                    self.publish("signal", {
                        "symbol": sym,
                        "direction": direction,
                        "confidence": 0.65,
                        "strategy": "Funding_Arb",
                        "market": "futures",
                        "fee_preset": "futures",
                        "meta_pass": True,
                        "raw_signals": {"signal_funding": float(direction)},
                        "size_mult": 1.0,
                        "regime": 0,
                        "leverage": LEVERAGE,
                    })
            except Exception as e:
                logger.debug("[FuturesAgent] Funding check error for %s: %s", sym, e)
