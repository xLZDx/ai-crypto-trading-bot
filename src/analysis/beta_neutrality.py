"""
Dynamic Beta-Neutrality Filter — Phase 5, Level 5.

Per updated_architecture_plan_en.md §17:

    "A module that monitors the correlation matrix of your open positions
     to prevent compounded losses. Prohibit opening new trades in the same
     direction (e.g., all Longs) if the portfolio's total exposure to a
     single systemic factor (like BTC beta) exceeds a predefined threshold."

The filter computes per-position β to a *factor proxy* (default = BTC) and
rejects new orders that push the aggregate β past `max_beta_exposure`. This
caps the worst-case crash exposure: if BTC drops 10 %, a portfolio with
β = 1.5 loses 15 %; the filter holds β to a value the operator chose.

Usage:
    bn = BetaNeutralityFilter(history)
    bn.update_position("ETH/USDT", side="long", notional=5_000)
    if bn.would_breach(symbol="SOL/USDT", side="long", notional=2_000):
        skip_trade()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class _Position:
    symbol:   str
    side:     str           # "long" | "short"
    notional: float

    @property
    def signed_notional(self) -> float:
        return self.notional if self.side == "long" else -self.notional


@dataclass
class BetaSnapshot:
    factor:           str
    aggregate_beta:   float
    per_position_beta: dict
    per_position_notional: dict
    n_positions:      int
    breach:           bool


class BetaNeutralityFilter:
    """Track aggregate factor-beta exposure of all open positions."""

    def __init__(
        self,
        history_returns: pd.DataFrame,
        *,
        factor: str = "BTC/USDT",
        max_beta_exposure: float = 1.0,   # |β_total| > this ⇒ reject new same-side
        min_history: int = 100,
    ):
        if factor not in history_returns.columns:
            raise ValueError(f"factor '{factor}' missing from history_returns")
        if len(history_returns) < min_history:
            raise ValueError(f"need at least {min_history} return samples for stable β")

        self.factor = factor
        self.max_beta_exposure = float(max_beta_exposure)
        self._history = history_returns.copy()
        self._betas: dict[str, float] = self._compute_betas(self._history, factor)
        self._positions: dict[str, _Position] = {}

    # ── β fitting ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_betas(history: pd.DataFrame, factor: str) -> dict:
        """OLS β of each column vs the factor column."""
        f = history[factor].to_numpy(dtype=float)
        f_demeaned = f - f.mean()
        denom = float((f_demeaned ** 2).sum()) or 1e-12
        out: dict = {}
        for col in history.columns:
            if col == factor:
                out[col] = 1.0
                continue
            x = history[col].to_numpy(dtype=float)
            x_demeaned = x - x.mean()
            num = float((x_demeaned * f_demeaned).sum())
            out[col] = num / denom
        return out

    # ── Position book updates ─────────────────────────────────────────────

    def update_position(self, symbol: str, side: str, notional: float) -> None:
        if notional <= 0:
            self._positions.pop(symbol, None)
            return
        self._positions[symbol] = _Position(symbol=symbol, side=side, notional=float(notional))

    def close_position(self, symbol: str) -> None:
        self._positions.pop(symbol, None)

    # ── Snapshot / breach check ───────────────────────────────────────────

    def aggregate_beta(self, *, hypothetical: _Position | None = None) -> float:
        positions = list(self._positions.values())
        if hypothetical is not None:
            positions = [p for p in positions if p.symbol != hypothetical.symbol] + [hypothetical]
        total_notional = sum(abs(p.signed_notional) for p in positions) or 1.0
        beta = 0.0
        for p in positions:
            b = self._betas.get(p.symbol, 1.0)
            beta += p.signed_notional * b / total_notional
        return float(beta)

    def snapshot(self) -> BetaSnapshot:
        positions = list(self._positions.values())
        total = sum(abs(p.signed_notional) for p in positions) or 1.0
        per_beta = {p.symbol: float(self._betas.get(p.symbol, 1.0)) for p in positions}
        per_notional = {p.symbol: float(p.signed_notional) for p in positions}
        agg = self.aggregate_beta()
        return BetaSnapshot(
            factor=self.factor,
            aggregate_beta=agg,
            per_position_beta=per_beta,
            per_position_notional=per_notional,
            n_positions=len(positions),
            breach=abs(agg) > self.max_beta_exposure,
        )

    def would_breach(self, symbol: str, side: str, notional: float) -> bool:
        """True iff opening this trade would push |aggregate β| past the cap."""
        hyp = _Position(symbol=symbol, side=side, notional=float(notional))
        agg = self.aggregate_beta(hypothetical=hyp)
        return abs(agg) > self.max_beta_exposure

    # ── Online β refit (call periodically) ────────────────────────────────

    def refit(self, fresh_returns: pd.DataFrame) -> None:
        """Re-estimate per-symbol β from a fresh history window."""
        if self.factor not in fresh_returns.columns:
            return
        self._history = fresh_returns.copy()
        self._betas = self._compute_betas(fresh_returns, self.factor)


__all__ = ["BetaNeutralityFilter", "BetaSnapshot"]
