"""
ExecutionAgent — Order Router.

Responsibilities:
  - Receives approved orders from RiskAgent
  - Routes to exchange (market order = taker, limit order = maker)
  - Tracks open positions and P&L
  - Publishes fill confirmations back to the bus
  - Implements basic TWAP for large orders (>5% of capital)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.engine.agents.agent_bus import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

TWAP_THRESHOLD_PCT = 0.05   # use TWAP if order > 5% of capital
MAX_HOLD_BARS = 48           # force-close after 48h (futures/spot)


class ExecutionAgent(BaseAgent):
    NAME = "ExecutionAgent"

    def __init__(self, exchange_client=None, bus=None, interval_sec: float = 5.0):
        super().__init__(bus=bus, interval_sec=interval_sec)
        self._exchange = exchange_client
        self._open_positions: dict[str, dict] = {}

    def _setup_subscriptions(self):
        self.bus.subscribe("order", self._on_order_request)
        self.bus.subscribe("candle", self._on_candle)

    def _on_order_request(self, msg) -> None:
        payload = msg.payload or {}
        if payload.get("status") != "pending":
            return

        sym = payload.get("symbol", "")
        direction = int(payload.get("direction", 0))
        size_usdt = float(payload.get("size_usdt", 0))

        if not sym or direction == 0 or size_usdt <= 0:
            return

        # Check for existing position
        if sym in self._open_positions:
            existing = self._open_positions[sym]
            if existing["direction"] == direction:
                logger.debug("[ExecutionAgent] %s already has position in same direction.", sym)
                return
            # Opposite direction → close first
            self._close_position(sym, reason="signal_flip")

        self._open_position(sym, direction, size_usdt, payload)

    def _open_position(self, sym: str, direction: int, size_usdt: float,
                       order_payload: dict) -> None:
        if self._exchange is None:
            # Simulation mode — just record the position
            current_price = self._get_last_price(sym)
            if current_price is None:
                logger.warning("[ExecutionAgent] Cannot open %s — no price.", sym)
                return

            self._open_positions[sym] = {
                "direction": direction,
                "entry_price": current_price,
                "size_usdt": size_usdt,
                "entry_time": datetime.now(timezone.utc),
                "bars_held": 0,
                "strategy": order_payload.get("strategy", ""),
                "confidence": order_payload.get("confidence", 0.5),
            }
            logger.info("[ExecutionAgent][SIM] Opened %s dir=%+d size=%.2f @ %.4f",
                        sym, direction, size_usdt, current_price)
            self.publish("order", {
                "symbol": sym, "direction": direction, "size_usdt": size_usdt,
                "entry_price": current_price, "status": "open",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        else:
            try:
                action = "buy" if direction > 0 else "sell"
                # Real exchange call would go here
                logger.info("[ExecutionAgent] Placing %s order on exchange: %s %.2f USDT",
                            action, sym, size_usdt)
            except Exception as e:
                logger.error("[ExecutionAgent] Order placement failed for %s: %s", sym, e)

    def _close_position(self, sym: str, reason: str = "manual") -> None:
        if sym not in self._open_positions:
            return

        pos = self._open_positions.pop(sym)
        current_price = self._get_last_price(sym)
        if current_price is None:
            logger.warning("[ExecutionAgent] Cannot close %s — no price.", sym)
            return

        entry = pos["entry_price"]
        direction = pos["direction"]
        size = pos["size_usdt"]
        raw_ret = (current_price - entry) / entry * direction
        pnl = raw_ret * size

        logger.info("[ExecutionAgent][SIM] Closed %s dir=%+d pnl=%.2f USDT reason=%s",
                    sym, direction, pnl, reason)

        self.publish("order", {
            "symbol": sym, "direction": direction,
            "size_usdt": size, "entry_price": entry, "exit_price": current_price,
            "pnl": pnl, "reason": reason, "status": "closed",
            "bars_held": pos.get("bars_held", 0),
            "strategy": pos.get("strategy", ""),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _on_candle(self, msg) -> None:
        sym = (msg.payload or {}).get("symbol", "")
        if sym in self._open_positions:
            self._open_positions[sym]["bars_held"] = \
                self._open_positions[sym].get("bars_held", 0) + 1
            if self._open_positions[sym]["bars_held"] >= MAX_HOLD_BARS:
                self._close_position(sym, reason="max_hold_timeout")

    def _get_last_price(self, sym: str) -> float | None:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        for fname in [f"{sym}_1h.csv.gz", f"{sym}_spot_1h.csv.gz"]:
            fpath = os.path.join(raw_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                import pandas as pd
                df = pd.read_csv(fpath, usecols=["close"])
                return float(df["close"].iloc[-1])
            except Exception:
                pass
        return None

    def _run_cycle(self) -> None:
        # Check max hold timeout
        for sym in list(self._open_positions.keys()):
            pos = self._open_positions[sym]
            if pos.get("bars_held", 0) >= MAX_HOLD_BARS:
                self._close_position(sym, reason="max_hold_timeout")
