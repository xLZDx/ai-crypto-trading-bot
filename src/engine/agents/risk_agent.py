"""
RiskAgent — Risk Manager (extends AgenticLLM with Kelly sizing and circuit breaker).

Responsibilities:
  - Receives trade signals from SignalAgent
  - Applies Kelly Criterion position sizing
  - Monitors drawdown in real-time (circuit breaker on 3 consecutive losses)
  - Runs LLM macro/news veto (existing AgenticLLM)
  - Detects liquidity sweep risk — blocks entries near large stop clusters
  - Phase 5: dynamic beta-neutrality gate — blocks new same-side trades when
    aggregate factor-β exposure would breach the cap (arch plan §17)
  - Publishes approved trade orders to ExecutionAgent

Decision chain for each signal:
  1. Circuit breaker check (consecutive losses)
  2. Drawdown limit check (max drawdown guard)
  3. Liquidity sweep check (don't enter when price near liq cluster)
  4. Beta-neutrality check (Phase 5)
  5. LLM macro veto (news/sentiment)
  6. Kelly sizing → publish order
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.engine.agents.agent_bus import BaseAgent
from src.analysis.kelly_criterion import KellySizer

logger = logging.getLogger(__name__)

MAX_DRAWDOWN_PCT = 10.0        # halt if cumulative account drawdown > 10%
MAX_DAILY_LOSS_PCT = 5.0       # hard stop if single-day loss > 5%
MAX_CONSECUTIVE_LOSSES = 3     # circuit breaker: consecutive losing trades
LIQ_PROXIMITY_BLOCK = 0.85    # block if too close to liquidity stop cluster
BASE_POSITION_PCT = 0.10       # base position size (10% of capital)
DATA_STALE_SEC = 300           # bar older than this (5 min) = data feed problem
API_LATENCY_LIMIT_MS = 500     # halt new entries if exchange RTT exceeds this


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

        # Phase 4 — daily drawdown breaker
        self._daily_start_capital = initial_capital
        self._daily_start_date = datetime.now(timezone.utc).date()

        # Phase 4 — data staleness breaker
        self._last_bar_ts: datetime | None = None   # updated by update_bar_timestamp()

        # Phase 4 — API latency breaker (updated externally by execution agent)
        self.last_api_latency_ms: float = 0.0

        # Phase 5 — dynamic beta-neutrality filter (lazy init; needs history)
        self._beta_filter = None  # set via attach_beta_filter(returns_history)

        # Lazy-load LLM veto (must stay inside __init__ — was unreachable
        # before due to a Phase 9 misplaced-method bug).
        self._llm = None
        try:
            from src.engine.agentic_llm import AgenticLLM
            self._llm = AgenticLLM()
        except Exception as e:
            logger.warning("[RiskAgent] AgenticLLM unavailable: %s", e)

    def attach_beta_filter(self, history_returns, *, factor: str = "BTC/USDT",
                           max_beta_exposure: float = 1.0) -> None:
        """Activate the Phase 5 beta-neutrality pre-trade gate.

        Call once after enough historical returns are loaded (≥100 rows).
        After this, `check_beta_neutrality(symbol, side, notional)` will
        block new orders whose addition would push aggregate |β| past the cap.
        """
        try:
            from src.analysis.beta_neutrality import BetaNeutralityFilter
            self._beta_filter = BetaNeutralityFilter(
                history_returns, factor=factor, max_beta_exposure=max_beta_exposure,
            )
            logger.info("BetaNeutralityFilter attached. factor=%s cap=%.2f",
                        factor, max_beta_exposure)
        except Exception as exc:
            logger.warning("Could not attach BetaNeutralityFilter: %s", exc)
            self._beta_filter = None

    def check_beta_neutrality(self, symbol: str, side: str, notional: float) -> bool:
        """Return True if trade is allowed by the β-neutrality gate.

        When no filter is attached (e.g., insufficient history), pass-through
        allows the trade. Logs blocks for the dashboard to surface.
        """
        if self._beta_filter is None:
            return True
        try:
            blocked = self._beta_filter.would_breach(symbol, side, notional)
            if blocked:
                snap = self._beta_filter.snapshot()
                logger.warning(
                    "[β-Neutrality] blocked %s %s %.0f — would push |β|>%.2f (current %+.2f)",
                    side, symbol, notional, self._beta_filter.max_beta_exposure,
                    snap.aggregate_beta,
                )
            return not blocked
        except Exception as exc:
            # INTENTIONALLY fail-open: an internal beta-filter crash should
            # not stop trading. But surface it at WARNING (was DEBUG) so the
            # operator can see when the gate is silently transparent —
            # silent-failure review flagged the original logger.debug as a
            # latent revenue-loss risk masked by a NaN-input bug in the
            # filter.
            logger.warning("[β-Neutrality] check failed (FAIL-OPEN — trade allowed): %s", exc)
            return True

    def _setup_subscriptions(self):
        # Subscribe to 'trade_signal' — the validated, market-augmented signal
        # published by SpotAgent/FuturesAgent/ScalpingAgent. The raw 'signal'
        # topic (from SignalAgent) is consumed by the market specialists; if
        # RiskAgent listened to it too, every signal would be converted to an
        # order twice (once raw, once after market validation). See agent_bus
        # topic docs for the topology.
        self.bus.subscribe("trade_signal", self._on_signal)
        self.bus.subscribe("order", self._on_order_filled)
        self.bus.subscribe("bar", self._on_bar)  # track data freshness

    def update_bar_timestamp(self, ts: datetime) -> None:
        """Call this whenever a fresh market bar arrives (WebSocket handler)."""
        self._last_bar_ts = ts

    def _on_bar(self, msg) -> None:
        payload = msg.payload or {}
        ts_raw = payload.get("timestamp")
        if ts_raw:
            try:
                self._last_bar_ts = datetime.fromisoformat(str(ts_raw)).replace(tzinfo=timezone.utc)
            except Exception as exc:
                # Silent-failure review: a bad timestamp here either skips the
                # staleness gate (if _last_bar_ts is None) or keeps using a
                # stale value, letting trades through on a dead feed. Surface
                # it at WARNING so operators see the bad bar source.
                logger.warning("[RiskAgent] _on_bar timestamp parse failed: %r (%s)",
                               ts_raw, exc)

    def _hard_kill(self, reason: str) -> None:
        """Publish flatten-all event and open the circuit breaker permanently."""
        self._circuit_open = True
        logger.critical("[RiskAgent] HARD KILL SWITCH ACTIVATED — reason: %s", reason)
        self.publish("risk_kill_switch", {
            "action": "flatten_all",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

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
            self._hard_kill(f"cumulative_drawdown_{drawdown_pct:.1f}pct")
        elif self._circuit_open and drawdown_pct < MAX_DRAWDOWN_PCT * 0.5:
            self._circuit_open = False
            logger.info("[RiskAgent] Drawdown recovered to %.1f%% — circuit CLOSED.", drawdown_pct)

        # Daily drawdown limit — hard stop if single-day loss exceeds threshold
        daily_loss_pct = (self._daily_start_capital - self.capital) / max(self._daily_start_capital, 1) * 100
        if daily_loss_pct > MAX_DAILY_LOSS_PCT:
            self._hard_kill(f"daily_loss_{daily_loss_pct:.1f}pct")

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

        # ── 0a. Data feed staleness ───────────────────────────────────────
        if self._last_bar_ts is not None:
            age_sec = (datetime.now(timezone.utc) - self._last_bar_ts).total_seconds()
            if age_sec > DATA_STALE_SEC:
                logger.warning(
                    "[RiskAgent] %s BLOCKED — data feed stale (last bar %.0fs ago).", sym, age_sec
                )
                return

        # ── 0b. API latency spike ─────────────────────────────────────────
        if self.last_api_latency_ms > API_LATENCY_LIMIT_MS:
            logger.warning(
                "[RiskAgent] %s BLOCKED — API latency spike %.0f ms > %d ms limit.",
                sym, self.last_api_latency_ms, API_LATENCY_LIMIT_MS,
            )
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
        # Daily drawdown counter reset at UTC midnight
        today = datetime.now(timezone.utc).date()
        if today != self._daily_start_date:
            self._daily_start_date = today
            self._daily_start_capital = self.capital
            logger.info("[RiskAgent] Daily P&L counter reset for %s  capital=%.2f", today, self.capital)

        # Periodic health log
        drawdown_pct = (self.peak_capital - self.capital) / self.peak_capital * 100
        daily_loss_pct = (self._daily_start_capital - self.capital) / max(self._daily_start_capital, 1) * 100
        logger.debug(
            "[RiskAgent] Capital=%.2f | DD=%.1f%% | DailyLoss=%.1f%% | "
            "ConsecLosses=%d | WR=%.1f%% | W/L=%.2f | Latency=%.0fms",
            self.capital, drawdown_pct, daily_loss_pct, self._consecutive_losses,
            self._kelly.win_rate * 100, self._kelly.win_loss_ratio,
            self.last_api_latency_ms,
        )
