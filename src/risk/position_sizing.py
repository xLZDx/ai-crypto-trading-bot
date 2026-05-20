"""Position Sizing Gate — Phase 10.

Hard limits enforced at order-generation time, before any signal reaches
the exchange. Sizes down to the limit rather than skipping the trade
entirely; skips only when the sized-down quantity falls below MIN_NOTIONAL.

Thresholds (configurable; defaults follow the §S0-3 spec at $500 bankroll):
    max_risk_per_trade: 0.25–0.5% of bankroll  → $1.25–$2.50 per trade
    max_daily_risk:     2.0% of bankroll        → $10/day max loss
    max_open_positions: N per strategy category  → prevents correlated over-exposure

All thresholds are fractions of bankroll (0.0–1.0) not absolute USDT so
the gate self-adjusts as the account grows or shrinks without reconfiguration.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MIN_NOTIONAL_USDT = 5.0    # exchange minimum order value; skip if below this


@dataclass
class SizingConfig:
    """Threshold configuration. All fractions of bankroll (0.0–1.0)."""
    max_risk_per_trade_pct: float = 0.005    # 0.5% of bankroll per trade
    max_daily_risk_pct:     float = 0.02     # 2.0% of bankroll per calendar day
    max_open_positions:     int   = 6        # hard cap on concurrent open trades
    min_notional_usdt:      float = MIN_NOTIONAL_USDT  # skip if sized-down < this


@dataclass
class SizingDecision:
    """Return value from PositionSizingGate.check()."""
    allow:          bool
    original_usdt:  float
    sized_usdt:     float          # may equal original_usdt if no sizing needed
    reason:         str = ""       # non-empty only when adjusted or blocked
    was_adjusted:   bool = False   # True → sized_usdt < original_usdt


class PositionSizingGate:
    """
    Pre-order position sizing gate.

    Usage:
        gate = PositionSizingGate(cfg)
        decision = gate.check(
            trade_usdt=50.0,
            bankroll_usdt=500.0,
            open_position_count=2,
        )
        if not decision.allow:
            log(decision.reason); return
        actual_amount = decision.sized_usdt / price
        exchange.create_order(symbol, 'buy', actual_amount)

    Thread-safe: daily-risk accumulation uses an internal lock.
    """

    def __init__(self, cfg: Optional[SizingConfig] = None):
        self.cfg = cfg or SizingConfig()
        self._daily_risk_usdt: float = 0.0
        self._day_start_epoch: float = self._today_epoch()
        import threading
        self._lock = threading.Lock()

    @staticmethod
    def _today_epoch() -> float:
        """Epoch of start of today (UTC midnight)."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counter when UTC date rolls over."""
        if time.time() >= self._day_start_epoch + 86400:
            self._daily_risk_usdt = 0.0
            self._day_start_epoch = self._today_epoch()

    def record_trade(self, executed_usdt: float) -> None:
        """Call after a trade executes to accumulate daily risk.

        Should be called once per executed order with the USD value
        of the position opened.
        """
        with self._lock:
            self._reset_daily_if_needed()
            self._daily_risk_usdt += max(0.0, executed_usdt)

    def check(
        self,
        trade_usdt: float,
        bankroll_usdt: float,
        open_position_count: int = 0,
    ) -> SizingDecision:
        """Evaluate and size a proposed trade.

        Parameters
        ----------
        trade_usdt:          Proposed position value in USDT.
        bankroll_usdt:       Current total equity in USDT.
        open_position_count: Number of currently open positions (all strategies).

        Returns
        -------
        SizingDecision with allow=False if the trade must be skipped,
        allow=True otherwise. sized_usdt holds the (possibly reduced) amount.
        """
        with self._lock:
            self._reset_daily_if_needed()

            if bankroll_usdt <= 0:
                return SizingDecision(
                    allow=False, original_usdt=trade_usdt, sized_usdt=0.0,
                    reason="bankroll_usdt <= 0 — cannot compute risk fractions",
                )

            # 1. Max open positions
            if open_position_count >= self.cfg.max_open_positions:
                return SizingDecision(
                    allow=False, original_usdt=trade_usdt, sized_usdt=0.0,
                    reason=(f"max_open_positions={self.cfg.max_open_positions} reached "
                            f"(current={open_position_count})"),
                )

            # 2. Per-trade cap
            per_trade_cap = bankroll_usdt * self.cfg.max_risk_per_trade_pct
            sized = min(trade_usdt, per_trade_cap)
            was_adjusted = sized < trade_usdt

            # 3. Daily risk cap
            daily_cap = bankroll_usdt * self.cfg.max_daily_risk_pct
            remaining_daily = max(0.0, daily_cap - self._daily_risk_usdt)
            if remaining_daily <= 0:
                return SizingDecision(
                    allow=False, original_usdt=trade_usdt, sized_usdt=0.0,
                    was_adjusted=True,
                    reason=(f"daily risk budget exhausted "
                            f"(used={self._daily_risk_usdt:.2f}/{daily_cap:.2f} USDT)"),
                )
            sized = min(sized, remaining_daily)
            was_adjusted = was_adjusted or (sized < trade_usdt)

            # 4. MIN_NOTIONAL check (after sizing)
            if sized < self.cfg.min_notional_usdt:
                return SizingDecision(
                    allow=False, original_usdt=trade_usdt, sized_usdt=sized,
                    was_adjusted=True,
                    reason=(f"sized_usdt={sized:.2f} < min_notional={self.cfg.min_notional_usdt} "
                            f"— order below exchange minimum; skipping"),
                )

            if was_adjusted:
                reason = (f"sized {trade_usdt:.2f} → {sized:.2f} USDT "
                          f"(per_trade_cap={per_trade_cap:.2f}, "
                          f"daily_remaining={remaining_daily:.2f})")
                logger.info("[sizing] %s", reason)
            return SizingDecision(
                allow=True, original_usdt=trade_usdt, sized_usdt=sized,
                was_adjusted=was_adjusted,
                reason="" if not was_adjusted else reason,
            )

    def daily_risk_used(self) -> float:
        """Current session's cumulative risk in USDT (diagnostic)."""
        with self._lock:
            self._reset_daily_if_needed()
            return self._daily_risk_usdt

    def state_dict(self) -> dict:
        """Snapshot for dashboard tile."""
        with self._lock:
            self._reset_daily_if_needed()
            return {
                "daily_risk_usdt": self._daily_risk_usdt,
                "max_risk_per_trade_pct": self.cfg.max_risk_per_trade_pct,
                "max_daily_risk_pct": self.cfg.max_daily_risk_pct,
                "max_open_positions": self.cfg.max_open_positions,
            }


__all__ = ["SizingConfig", "SizingDecision", "PositionSizingGate", "MIN_NOTIONAL_USDT"]
