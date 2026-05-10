"""
Execution-Cost & Slippage Model — Phase 5, Level 5.

Per updated_architecture_plan_en.md §16:

    Real_Price = P_mid * (1 + Fee + Slippage(Size, Depth))

Slippage is *not* a random fudge factor — it's a deterministic function of
how deep the bot's order has to walk into the book to fill its size. This
module models that walk against an L2 book snapshot and exposes a
backtest-applicable correction.

Two modes:

  1. Linear-impact (cheap, no book required)
     slip_bps = lambda_impact * (size / book_volume)

  2. Book-walk (expensive, full L2 levels required)
     Eats levels one-by-one until `size` is filled, returning VWAP.

Both modes return *signed* slippage so it can be added to mid:
    buy   ⇒ executed_price = P_mid + |slip|
    sell  ⇒ executed_price = P_mid - |slip|
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Default Binance Spot taker fee = 10 bps; matches `src/utils/config.py`
# constants used by the backtester.
DEFAULT_FEE_BPS = 10.0


# ─── Linear impact (closed-form) ────────────────────────────────────────────

def linear_slippage_bps(
    size: float,
    book_volume: float,
    *,
    lambda_impact: float = 0.5,
) -> float:
    """Slippage in basis points under a linear-impact model.

    `lambda_impact = 0.5` ⇒ a market order eating 1× the book volume slips
    50 bps. Empirically consistent with Binance Spot/Futures top-of-book.
    """
    if book_volume <= 0:
        return 1e4  # 100% slip when book is empty
    ratio = float(size) / max(float(book_volume), 1e-12)
    return float(lambda_impact * ratio * 1e4)


# ─── Book walk (level-by-level) ─────────────────────────────────────────────

@dataclass
class BookWalkResult:
    fill_size:    float
    avg_price:    float
    slippage_bps: float
    levels_used:  int


def book_walk_slippage(
    size: float,
    side: str,                          # "buy" | "sell"
    bids: list[tuple[float, float]],    # [(price, qty), ...] descending
    asks: list[tuple[float, float]],    # [(price, qty), ...] ascending
    *,
    mid_price: float | None = None,
) -> BookWalkResult:
    """Walk a real L2 book and return the VWAP fill price.

    Args:
        size:  base-asset size to execute.
        side:  "buy" walks `asks`, "sell" walks `bids`.
        bids / asks: list of (price, qty) levels. Standard order: best price first.
        mid_price: if provided, used for slippage_bps. Else inferred as
                   midpoint of first bid/ask.

    Returns:
        BookWalkResult(fill_size, avg_price, slippage_bps, levels_used)
    """
    if size <= 0 or side not in {"buy", "sell"}:
        return BookWalkResult(0.0, 0.0, 0.0, 0)

    levels = asks if side == "buy" else bids
    if not levels:
        return BookWalkResult(0.0, 0.0, 1e4, 0)

    if mid_price is None:
        if not bids or not asks:
            mid_price = float(levels[0][0])
        else:
            mid_price = 0.5 * (float(bids[0][0]) + float(asks[0][0]))

    remaining = float(size)
    notional = 0.0
    filled = 0.0
    used = 0
    for price, qty in levels:
        if remaining <= 0:
            break
        take = min(remaining, float(qty))
        notional += take * float(price)
        filled += take
        remaining -= take
        used += 1

    if filled <= 0:
        return BookWalkResult(0.0, 0.0, 1e4, used)

    avg_price = notional / filled
    if side == "buy":
        slip_bps = (avg_price / mid_price - 1.0) * 1e4
    else:
        slip_bps = (1.0 - avg_price / mid_price) * 1e4
    return BookWalkResult(
        fill_size=filled, avg_price=avg_price,
        slippage_bps=float(slip_bps), levels_used=used,
    )


# ─── Total execution cost (slippage + fee) ──────────────────────────────────

def total_execution_cost_bps(
    size: float,
    book_volume: float,
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
    lambda_impact: float = 0.5,
) -> float:
    """Closed-form total cost = slippage + taker fee, in bps."""
    return linear_slippage_bps(size, book_volume, lambda_impact=lambda_impact) + fee_bps


def real_price(
    p_mid: float,
    side: str,
    size: float,
    book_volume: float,
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
    lambda_impact: float = 0.5,
) -> float:
    """Return the *expected* executed price including slippage + fee.

    Per architecture plan §16:
        Real_Price = P_mid * (1 + Fee + Slippage(Size, Depth))   for buys
        Real_Price = P_mid * (1 - Fee - Slippage(Size, Depth))   for sells
    """
    cost = total_execution_cost_bps(size, book_volume,
                                    fee_bps=fee_bps, lambda_impact=lambda_impact)
    factor = cost / 1e4
    return float(p_mid) * (1.0 + factor) if side == "buy" else float(p_mid) * (1.0 - factor)


# ─── Backtest helper ────────────────────────────────────────────────────────

def apply_slippage_to_pnl(
    pnl_series,
    sizes,
    volumes,
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
    lambda_impact: float = 0.5,
):
    """Subtract round-trip execution cost from a PnL series.

    Used by the backtester to make backtest curves realistic. Each trade
    pays cost on entry AND exit.
    """
    import numpy as np
    p = np.asarray(pnl_series, dtype=float)
    s = np.asarray(sizes,   dtype=float)
    v = np.asarray(volumes, dtype=float)
    cost_bps = np.array([
        total_execution_cost_bps(si, vi, fee_bps=fee_bps, lambda_impact=lambda_impact)
        for si, vi in zip(s, v)
    ])
    # Two-way cost (entry + exit). Convert bps → fraction of trade notional.
    return p - 2 * (cost_bps / 1e4) * np.abs(s) * np.where(v > 0, np.abs(p) / np.maximum(np.abs(s), 1e-12), 0.0)


__all__ = [
    "linear_slippage_bps", "book_walk_slippage", "BookWalkResult",
    "total_execution_cost_bps", "real_price", "apply_slippage_to_pnl",
    "DEFAULT_FEE_BPS",
]
