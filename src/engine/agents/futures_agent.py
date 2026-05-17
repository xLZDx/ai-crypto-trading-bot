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
from src.analysis.risk_manager import calc_liquidation_price
from src.analysis.live_funding import fetch_funding_rate

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

CONFIDENCE_THRESHOLD = 0.60
MAX_HOLD_HOURS = 24
FUNDING_ARB_THRESHOLD = 0.001    # 0.1% funding = strong short signal
LEVERAGE = 2.0                    # conservative for safety
FEE_MAKER = 0.0002
FEE_TAKER = 0.0004

# Minimum distance from entry to liquidation price (fraction of entry).
# Positions where liq is within 3% of entry are too close to sustain any
# adverse tick and will be rejected.
_MIN_LIQ_DISTANCE: float = 0.03

# Funding rates above this threshold (0.3%) in the opposing direction are
# a strong headwind; block the trade rather than fight the carry cost.
_MAX_ADVERSE_FUNDING: float = 0.003


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
            import io
            import joblib
            from src.utils.model_integrity import verify_and_load_bytes
            path = os.path.join(PROJECT_ROOT, "models", "futures_short_model.joblib")
            if os.path.exists(path):
                self._futures_model = joblib.load(io.BytesIO(verify_and_load_bytes(path)))
                logger.info("[FuturesAgent] Futures model loaded.")
        except Exception as e:
            logger.warning("[FuturesAgent] Model load error: %s", e)

    def _setup_subscriptions(self):
        self.bus.subscribe("signal", self._on_signal)
        self.bus.subscribe("regime", self._on_regime)

    def _on_regime(self, msg) -> None:
        regime = (msg.payload or {}).get("regime", 0)
        if regime == 2:
            logger.info("[FuturesAgent] VOLATILE -- only funding arb allowed.")

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
            logger.debug("[FuturesAgent] Skipping %s -- VOLATILE, no funding signal.", sym)
            return

        if direction == 0 or confidence < CONFIDENCE_THRESHOLD:
            return

        # Liquidity sweep guard
        liq_prox = float(raw_signals.get("liq_proximity", 0))
        if liq_prox > 0.90:
            logger.info("[FuturesAgent] %s BLOCKED -- near liquidation cluster (%.2f).", sym, liq_prox)
            return

        # ── G3: Live funding gate (fail-closed) ───────────────────────────
        # Fetch current funding rate. If unavailable (exchange error / network),
        # block the trade rather than entering with unknown carry cost.
        live_rate = fetch_funding_rate(sym)
        if live_rate is None:
            logger.warning(
                "[FuturesAgent] %s BLOCKED -- live funding rate unavailable (fail-closed).", sym
            )
            return

        # Block when funding strongly opposes the intended direction.
        # Long trades are hurt by high positive funding; shorts by high negative.
        side_str = "long" if direction > 0 else "short"
        adverse_funding = (live_rate > _MAX_ADVERSE_FUNDING and direction > 0) or \
                          (live_rate < -_MAX_ADVERSE_FUNDING and direction < 0)
        if adverse_funding:
            logger.info(
                "[FuturesAgent] %s BLOCKED -- adverse funding %.4f%% exceeds threshold (dir=%+d).",
                sym, live_rate * 100, direction,
            )
            return

        # ── G1: Liquidation proximity gate ────────────────────────────────
        # Resolve the current entry price: prefer explicit field, fall back to
        # live data. If price is unknown, skip the gate (don't block blindly).
        entry_price: float | None = payload.get("price") or payload.get("close")
        if entry_price is None:
            try:
                df = self.data_getter(sym)
                if df is not None and len(df) > 0:
                    entry_price = float(df["close"].iloc[-1])
            except Exception as exc:
                logger.debug("[FuturesAgent] Could not resolve entry price for %s: %s", sym, exc)

        if entry_price is not None and entry_price > 0:
            try:
                liq_price = calc_liquidation_price(entry_price, LEVERAGE, side_str)
                liq_dist = abs(liq_price - entry_price) / entry_price
                if liq_dist < _MIN_LIQ_DISTANCE:
                    logger.warning(
                        "[FuturesAgent] %s BLOCKED -- liq %.2f within %.1f%% of entry %.2f "
                        "(min %.0f%% required).",
                        sym, liq_price, liq_dist * 100, entry_price, _MIN_LIQ_DISTANCE * 100,
                    )
                    return
            except (ValueError, AssertionError) as exc:
                logger.warning("[FuturesAgent] liq calc skipped for %s: %s", sym, exc)

        # ── Funding arbitrage override — strong directional edge ───────────
        if abs(funding_signal) > 0.5:
            logger.info("[FuturesAgent] %s funding arb signal: dir=%+d funding=%.4f",
                        sym, int(funding_signal), float(raw_signals.get("funding_rate", 0)))

        logger.info("[FuturesAgent] SIGNAL %s dir=%+d conf=%.2f lev=%.1fx (futures 1h)",
                    sym, direction, confidence, LEVERAGE)

        # Pass to RiskAgent via 'trade_signal' (NOT 'signal'). Same recursion
        # break as SpotAgent — see agent_bus topic docs.
        self.publish("trade_signal", {
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
                    # Funding-arb is a self-generated signal (no upstream), but
                    # we still publish on 'trade_signal' so RiskAgent — the only
                    # consumer that converts a signal to an order — receives it.
                    self.publish("trade_signal", {
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
                # Was logger.debug — silent-failure-hunter flagged that a
                # programming error in the publish() call (bad payload key)
                # would never surface. logger.warning keeps the loop running
                # but makes the failure visible to the operator.
                logger.warning("[FuturesAgent] Funding check error for %s: %s", sym, e)
