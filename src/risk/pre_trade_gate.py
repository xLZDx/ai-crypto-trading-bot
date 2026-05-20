"""PreTradeGate — Phase 10: unified safety check before every order.

All safety conditions (kill-switch, ws_connected, warmup_complete,
SAFE_MODE, NaN/Inf guard) are checked in a single PreTradeGate.check()
call — not scattered across independent if-statements in process_kline.

Two-lock design:
    trading_lock  — serializes order placement (held during exchange call)
    flag_lock     — serializes flag mutations (ws_connected, SAFE_MODE)

    Why separate: a single trading_lock covering both order placement AND
    WebSocket flag writes causes deadlock when the WebSocket thread holds it
    while waiting on a network response while the trading loop waits for it.

SAFE_MODE transitions (set automatically, cleared only by operator):
    "live"      — normal trading
    "read_only" — signals computed + paper-logged, NO exchange orders
    "off"       — bot fully paused (manual, rare)

Warmup:
    After bot restart or model reload, orders are blocked until
    warmup_bars_seen >= warmup_bars_required. Each WebSocket tick
    increments the counter.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SAFE_MODE_LIVE      = "live"
SAFE_MODE_READ_ONLY = "read_only"
SAFE_MODE_OFF       = "off"
_VALID_MODES = frozenset({SAFE_MODE_LIVE, SAFE_MODE_READ_ONLY, SAFE_MODE_OFF})

# Minimum 14 bars for RSI to stabilise after restart.
DEFAULT_WARMUP_BARS = 14


@dataclass
class GateContext:
    """Input snapshot for a single gate check."""
    symbol:          str   = ""
    action:          str   = "open"   # "open" | "close" | "reduce"
    has_nan_inf:     bool  = False     # True if signal features contain NaN/Inf


@dataclass
class GateResult:
    """Return value from PreTradeGate.check()."""
    allow:   bool
    reason:  str = ""


class PreTradeGate:
    """
    Singleton-friendly pre-trade safety gate.

    Usage in the bot loop (simplified):

        gate = PreTradeGate()
        ...
        # On WebSocket disconnect:
        with gate.flag_lock:
            gate.ws_connected = False

        # On WebSocket reconnect (after state reconciliation):
        with gate.flag_lock:
            gate.ws_connected = True

        # Before placing an order:
        with gate.trading_lock:
            result = gate.check(GateContext(symbol=sym, action="open"))
            if not result.allow:
                logger.warning("[gate] blocked: %s", result.reason)
                return
            exchange.create_order(...)

    Note: close/reduce-only orders bypass ws_connected and warmup checks
    so existing positions can always be de-risked even during outages.
    """

    def __init__(
        self,
        warmup_bars_required: int = DEFAULT_WARMUP_BARS,
        kill_switch=None,       # optional KillSwitch instance (injected for testability)
    ):
        self.trading_lock = threading.Lock()
        self.flag_lock    = threading.Lock()

        self._ws_connected:     bool = True   # assume connected at startup
        self._safe_mode:        str  = SAFE_MODE_LIVE
        self._warmup_bars_seen: int  = 0
        self._warmup_bars_required = warmup_bars_required
        self._warmup_complete:  bool = False
        self._kill_switch = kill_switch

    # ── Flag setters (always call under flag_lock) ────────────────────────

    @property
    def ws_connected(self) -> bool:
        return self._ws_connected

    @ws_connected.setter
    def ws_connected(self, value: bool) -> None:
        self._ws_connected = value

    @property
    def safe_mode(self) -> str:
        return self._safe_mode

    @safe_mode.setter
    def safe_mode(self, value: str) -> None:
        if value not in _VALID_MODES:
            logger.warning("[gate] invalid safe_mode=%r -- keeping %r", value, self._safe_mode)
            return
        old = self._safe_mode
        self._safe_mode = value
        if old != value:
            logger.warning("[gate] SAFE_MODE %r -> %r", old, value)

    def record_warmup_tick(self) -> None:
        """Call once per received WebSocket kline to advance warmup counter.
        Thread-safe: only called from the asyncio event loop thread.
        """
        if not self._warmup_complete:
            self._warmup_bars_seen += 1
            if self._warmup_bars_seen >= self._warmup_bars_required:
                self._warmup_complete = True
                logger.info(
                    "[gate] warmup complete after %d bars", self._warmup_bars_seen
                )

    # ── Main gate check ───────────────────────────────────────────────────

    def check(self, ctx: Optional[GateContext] = None) -> GateResult:
        """Atomically evaluate all safety conditions.

        Must be called *inside* the caller's `with gate.trading_lock:` block
        (the caller holds trading_lock; this method reads flags under flag_lock
        internally so there is no cross-lock deadlock).
        """
        if ctx is None:
            ctx = GateContext()

        is_close = ctx.action in ("close", "reduce")

        with self.flag_lock:
            ws_connected    = self._ws_connected
            safe_mode       = self._safe_mode
            warmup_complete = self._warmup_complete

        # 1. Hard off — even close orders blocked
        if safe_mode == SAFE_MODE_OFF:
            return GateResult(allow=False, reason="SAFE_MODE=off")

        # 2. Read-only — paper trades only; close/reduce still allowed
        if safe_mode == SAFE_MODE_READ_ONLY and not is_close:
            return GateResult(allow=False, reason="SAFE_MODE=read_only — paper-trade only")

        # 3. Kill-switch — close/reduce allowed (de-risk path)
        if self._kill_switch is not None and not is_close:
            try:
                from src.risk.kill_switch import get_kill_switch
                ks = self._kill_switch if self._kill_switch is not None else get_kill_switch()
                paused, trigger = ks.evaluate()
                if paused:
                    return GateResult(
                        allow=False,
                        reason=f"kill_switch PAUSED (trigger={trigger})",
                    )
            except Exception as e:
                logger.warning("[gate] kill_switch check failed: %s -- allowing", e)

        # 4. WebSocket disconnected — close/reduce allowed
        if not ws_connected and not is_close:
            return GateResult(
                allow=False,
                reason="ws_connected=False — stale prices, blocking new positions",
            )

        # 5. Warmup — close/reduce always allowed
        if not warmup_complete and not is_close:
            return GateResult(
                allow=False,
                reason=f"warmup incomplete ({self._warmup_bars_seen}/{self._warmup_bars_required} bars)",
            )

        # 6. NaN/Inf in signal features — always block
        if ctx.has_nan_inf:
            return GateResult(allow=False, reason="NaN/Inf in signal features")

        return GateResult(allow=True)

    def state_dict(self) -> dict:
        """Snapshot for dashboard tile / API endpoint."""
        with self.flag_lock:
            return {
                "ws_connected":      self._ws_connected,
                "safe_mode":         self._safe_mode,
                "warmup_complete":   self._warmup_complete,
                "warmup_bars_seen":  self._warmup_bars_seen,
                "warmup_bars_required": self._warmup_bars_required,
            }


__all__ = [
    "PreTradeGate", "GateContext", "GateResult",
    "SAFE_MODE_LIVE", "SAFE_MODE_READ_ONLY", "SAFE_MODE_OFF",
    "DEFAULT_WARMUP_BARS",
]
