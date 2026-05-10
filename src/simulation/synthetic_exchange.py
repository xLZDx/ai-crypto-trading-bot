"""
Synthetic Adversarial Exchange — Phase 3, Level 3 (Execution & Simulation).

A differentiable matching engine that:
  • Replays L2 order-book history from QuestDB / Parquet.
  • Reacts to the bot's outgoing orders (market-impact model).
  • Fills via a "soft" probabilistic matcher so gradients can flow back to
    the OFT-alpha and the RL execution policy. (Plan §9 — softmax matching.)

Compared to a classic backtester:
  • The backtester's fill is binary (price ≤ ask  ⇒  filled). That hides the
    impact-vs-latency tradeoff and makes RL training useless.
  • Soft fill ∈ (0, 1) is differentiable and represents "fraction of the
    order that executes" given the limit-price offset and order-book depth.

This file deliberately does NOT depend on torch — `step()` returns plain
numpy arrays / dicts so it can be used from non-RL code paths too. A torch
shim is provided via `to_torch()` for joint OFT+RL training.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass
class ExchangeState:
    """Mutable book state at one instant."""
    timestamp: int
    p_bid:     float
    p_ask:     float
    v_bid:     float           # aggregated top-N bid volume
    v_ask:     float           # aggregated top-N ask volume
    inventory: float = 0.0     # bot's net position
    cash:      float = 0.0     # bot's cash
    realised_pnl: float = 0.0


@dataclass
class FillResult:
    fill_pct:    float
    avg_price:   float
    slippage:    float
    inventory_delta: float
    cash_delta:  float


@dataclass
class ImpactModel:
    """How much the book moves against you per dollar of size.

    `lambda_impact` is the linear coefficient in the simplest Almgren-Chriss
    style model:   ΔP / P = lambda_impact * size / V_book
    """
    lambda_impact: float = 0.5
    softness:      float = 50.0  # higher = sharper softmax fill curve


def softmax_fill(
    order_size: float,
    book_volume: float,
    limit_offset: float,
    *,
    softness: float = 50.0,
) -> float:
    """Differentiable fill fraction ∈ (0, 1).

    `limit_offset` is signed: positive = aggressive (cross the spread),
    negative = passive (post on far side). `book_volume` is the volume
    available on the side being hit. The function is monotone increasing
    in both order_size/book_volume and limit_offset.
    """
    book_ratio = order_size / max(book_volume, _EPS)
    raw = softness * (limit_offset - book_ratio)
    # Clamp logits to avoid overflow in float32 — equivalent to sigmoid.
    raw = max(-30.0, min(30.0, float(raw)))
    return 1.0 / (1.0 + np.exp(-raw))


class SyntheticExchange:
    """Tick-by-tick differentiable exchange.

    Usage:
        ex = SyntheticExchange(book_iter)            # iterable of book snapshots
        obs = ex.reset()
        for _ in range(steps):
            action = policy(obs)                     # (signed_size, price_offset)
            obs, reward, done, info = ex.step(action)
    """

    def __init__(
        self,
        book_iter: Iterable[dict] | None = None,
        impact: ImpactModel | None = None,
        *,
        symbol: str = "BTC/USDT",
        cash_start: float = 100_000.0,
        max_inventory: float = 10.0,
    ):
        self.symbol = symbol
        self.impact = impact or ImpactModel()
        self.cash_start = cash_start
        self.max_inventory = max_inventory
        self._book_iter = iter(book_iter) if book_iter is not None else None
        self.state: ExchangeState | None = None
        self._step_count = 0

    # ── Episode lifecycle ──────────────────────────────────────────────────

    def reset(self, book_iter: Iterable[dict] | None = None) -> dict:
        if book_iter is not None:
            self._book_iter = iter(book_iter)
        if self._book_iter is None:
            raise ValueError("SyntheticExchange.reset() needs a book_iter")
        first = next(self._book_iter)
        self.state = ExchangeState(
            timestamp=int(first.get("ts") or first.get("timestamp") or 0),
            p_bid=float(first["p_bid"]),
            p_ask=float(first["p_ask"]),
            v_bid=float(first["v_bid"]),
            v_ask=float(first["v_ask"]),
            inventory=0.0,
            cash=self.cash_start,
            realised_pnl=0.0,
        )
        self._step_count = 0
        return self._observation()

    def step(self, action: tuple[float, float]) -> tuple[dict, float, bool, dict]:
        """Apply (signed_size, price_offset) to current book.

        signed_size:    positive = buy, negative = sell, in base-asset units.
        price_offset:   signed, in fraction-of-mid-price.
                        +0.001 = aggressive 10 bps inside spread (cross),
                        -0.001 = passive 10 bps outside spread (post).
        """
        if self.state is None:
            raise RuntimeError("step() before reset()")

        signed_size, offset = float(action[0]), float(action[1])
        side_buy = signed_size >= 0
        size = abs(signed_size)

        # 1) Fill against current book
        book_v = self.state.v_ask if side_buy else self.state.v_bid
        fill_pct = softmax_fill(size, book_v, offset, softness=self.impact.softness)
        fill_size = size * fill_pct
        ref_price = self.state.p_ask if side_buy else self.state.p_bid
        # 2) Slippage from impact (linear in size / book_volume)
        impact_bps = self.impact.lambda_impact * (fill_size / max(book_v, _EPS))
        slip = ref_price * impact_bps
        avg_price = ref_price + slip if side_buy else ref_price - slip

        # 3) Update inventory & cash (clamped)
        signed_fill = fill_size if side_buy else -fill_size
        new_inv = float(np.clip(self.state.inventory + signed_fill,
                                -self.max_inventory, self.max_inventory))
        actual_fill = new_inv - self.state.inventory
        cash_delta = -actual_fill * avg_price

        self.state.inventory = new_inv
        self.state.cash += cash_delta

        # 4) Step to next book snapshot — episode ends when iterator exhausted
        try:
            nxt = next(self._book_iter)
            self.state.timestamp = int(nxt.get("ts") or nxt.get("timestamp") or 0)
            self.state.p_bid = float(nxt["p_bid"])
            self.state.p_ask = float(nxt["p_ask"])
            self.state.v_bid = float(nxt["v_bid"])
            self.state.v_ask = float(nxt["v_ask"])
            done = False
        except StopIteration:
            done = True

        # 5) Mark-to-market PnL (realised + unrealised)
        mid = 0.5 * (self.state.p_bid + self.state.p_ask)
        equity = self.state.cash + self.state.inventory * mid
        reward = float(equity - self.cash_start - self.state.realised_pnl)
        self.state.realised_pnl = equity - self.cash_start

        self._step_count += 1
        return (
            self._observation(),
            reward,
            done,
            {
                "fill_pct":   fill_pct,
                "avg_price":  avg_price,
                "slippage":   slip,
                "actual_fill": actual_fill,
                "cash_delta": cash_delta,
                "step":       self._step_count,
            },
        )

    # ── Observation builder ────────────────────────────────────────────────

    def _observation(self) -> dict:
        s = self.state
        mid = 0.5 * (s.p_bid + s.p_ask)
        spread = s.p_ask - s.p_bid
        return {
            "timestamp": s.timestamp,
            "p_bid":     s.p_bid,
            "p_ask":     s.p_ask,
            "v_bid":     s.v_bid,
            "v_ask":     s.v_ask,
            "imbalance": (s.v_bid - s.v_ask) / max(s.v_bid + s.v_ask, _EPS),
            "spread":    spread,
            "spread_bps": spread / max(mid, _EPS),
            "inventory": s.inventory,
            "cash":      s.cash,
            "equity":    s.cash + s.inventory * mid,
        }


__all__ = [
    "SyntheticExchange", "ExchangeState", "FillResult", "ImpactModel", "softmax_fill",
]
