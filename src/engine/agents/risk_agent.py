"""
RiskAgent — Risk Manager (extends AgenticLLM with Kelly sizing and circuit breaker).

Responsibilities:
  - Receives trade signals from SignalAgent
  - Applies Kelly Criterion position sizing
  - Monitors drawdown in real-time (circuit breaker on 3 consecutive losses)
  - Runs LLM macro/news veto (existing AgenticLLM)
  - Detects liquidity sweep risk — blocks entries near large stop clusters
  - Publishes approved trade orders to ExecutionAgent

Decision chain for each signal:
  1. Circuit breaker check (consecutive losses)
  2. Drawdown limit check (max drawdown guard)
  3. Liquidity sweep check (don't enter when price near liq cluster)
  4. LLM macro veto (news/sentiment)
  5. Kelly sizing → publish order
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.engine.agents.agent_bus import BaseAgent
from src.analysis.kelly_criterion import KellySizer

logger = logging.getLogger(__name__)

MAX_DRAWDOWN_PCT = 10.0       # halt trading if account drawdown > 10%
MAX_CONSECUTIVE_LOSSES = 3    # circuit breaker
LIQ_PROXIMITY_BLOCK = 0.85   # block if liq_proximity > this (too close to stop cluster)
BASE_POSITION_PCT = 0.10      # base position size (10% of capital)


class RiskAgent(BaseAgent):
    NAME = "RiskAgent"

    def __init__(self, initial_capital: float = 10_000.0, bus=None,
                 interval_sec: float = 1.0):
        super().__init__(bus=bus, interval_sec=interval_sec)
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self._consecutive_losses = 0
        self._kelly = KellySizer(window=50, half_kelly=True)
        self._circuit_open = False

        # Lazy-load LLM veto
        self._llm = None
        try:
            from src.engine.agentic_llm import AgenticLLM
            self._llm = AgenticLLM()
        except Exception as e:
            logger.warning("[RiskAgent] AgenticLLM unavailable: %s", e)

    def _setup_subscriptions(self):
        self.bus.subscribe("signal", self._on_signal)
        self.bus.subscribe("order", self._on_order_filled)

    def _on_order_filled(self, msg) -> None:
        payload = msg.payload or {}
        if payload.get("status") != "closed":
            return
        pnl = float(payload.get("pnl", 0))
        self._kelly.record_trade(pnl)
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        drawdown_pct = (self.peak_capital - self.capital) / self.peak_capital * 100
        if drawdown_pct > MAX_DRAWDOWN_PCT:
            self._circuit_open = True
            logger.warning("[RiskAgent] Max drawdown %.1f%% exceeded — circuit OPEN.", drawdown_pct)
        elif self._circuit_open and drawdown_pct < MAX_DRAWDOWN_PCT * 0.5:
            self._circuit_open = False
            logger.info("[RiskAgent] Drawdown recovered to %.1f%% — circuit CLOSED.", drawdown_pct)

    def _on_signal(self, msg) -> None:
        payload = msg.payload or {}
        if not payload.get("meta_pass", True):
            return  # meta-labeler already blocked it

        sym = payload.get("symbol", "")
        direction = int(payload.get("direction", 0))
        confidence = float(payload.get("confidence", 0.5))
        regime_size_mult = float(payload.get("size_mult", 1.0))
        liq_proximity = float(payload.get("raw_signals", {}).get("liq_proximity", 0))

        if direction == 0:
            return

        # ── 1. Circuit breaker ────────────────────────────────────────────
        if self._circuit_open:
            logger.info("[RiskAgent] %s BLOCKED — circuit open (max drawdown).", sym)
            return

        if self._kelly.circuit_breaker(self._consecutive_losses, MAX_CONSECUTIVE_LOSSES):
            logger.warning("[RiskAgent] %s BLOCKED — %d consecutive losses.",
                           sym, self._consecutive_losses)
            return

        # ── 2. Liquidity sweep guard ──────────────────────────────────────
        if liq_proximity > LIQ_PROXIMITY_BLOCK:
            logger.info("[RiskAgent] %s BLOCKED — liquidity proximity %.2f too high.", sym, liq_proximity)
            return

        # ── 3. LLM macro veto ────────────────────────────────────────────
        if self._llm and self._llm.is_active:
            action = "BUY" if direction > 0 else "SELL"
            decision, reason = self._llm.evaluate_trade(
                symbol=sym, action=action,
                technical_reason=payload.get("strategy", ""),
                headlines=[]
            )
            if decision == "REJECTED":
                logger.info("[RiskAgent] %s LLM-REJECTED: %s", sym, reason)
                return

        # ── 4. Kelly sizing ───────────────────────────────────────────────
        drawdown_pct = (self.peak_capital - self.capital) / self.peak_capital
        vol_scale = max(0.5, 1.0 - drawdown_pct * 2)  # reduce size as drawdown grows
        position_usdt = self._kelly.size(
            capital=self.capital,
            p_win=confidence,
            volatility_scale=vol_scale * regime_size_mult
        )

        # ── 5. Publish order ──────────────────────────────────────────────
        self.publish("order", {
            "symbol": sym,
            "direction": direction,
            "size_usdt": position_usdt,
            "confidence": confidence,
            "strategy": payload.get("strategy", ""),
            "regime": payload.get("regime", 0),
            "status": "pending",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        logger.info("[RiskAgent] ORDER approved: %s dir=%+d size=%.2f USDT conf=%.2f",
                    sym, direction, position_usdt, confidence)

    def _run_cycle(self) -> None:
        # Periodic health log
        drawdown_pct = (self.peak_capital - self.capital) / self.peak_capital * 100
        logger.debug("[RiskAgent] Capital=%.2f | Drawdown=%.1f%% | "
                     "ConsecLosses=%d | WR=%.1f%% | W/L=%.2f",
                     self.capital, drawdown_pct, self._consecutive_losses,
                     self._kelly.win_rate * 100, self._kelly.win_loss_ratio)
