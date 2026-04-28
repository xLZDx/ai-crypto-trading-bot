"""
Kelly Criterion position sizing.

Full Kelly is theoretically optimal but practically too aggressive — a single
bad streak destroys the account. Half-Kelly (the industry standard) halves the
bet, sacrifices ~25% of growth rate, but cuts variance by ~75%.

Formula: f* = (p * (b + 1) - 1) / b
  p = probability of win (from model predict_proba)
  b = win/loss ratio (average_win / average_loss from trade history)

Dynamic Kelly: position size updates every trade using live win/loss ratio
from the trade log — so the bot auto-adjusts as market regime changes.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# Hard caps — never bet more than this fraction of capital regardless of model
MAX_KELLY_FRACTION = 0.25   # 25% max per trade
MIN_KELLY_FRACTION = 0.005  # 0.5% min (still open a small position)


def kelly_fraction(
    p_win: float,
    win_loss_ratio: float,
    half_kelly: bool = True,
    max_fraction: float = MAX_KELLY_FRACTION,
    min_fraction: float = MIN_KELLY_FRACTION,
) -> float:
    """
    Compute Kelly fraction.

    Args:
        p_win:          Model's predicted probability of a winning trade (0–1).
        win_loss_ratio: Ratio of avg_win / avg_loss from recent trade history.
                        Use 1.5 as default when no history is available.
        half_kelly:     Apply half-Kelly safety factor (default True).
        max_fraction:   Hard ceiling on position fraction.
        min_fraction:   Hard floor — always open at least this size.

    Returns:
        Fraction of capital to risk (0 to max_fraction).
    """
    if win_loss_ratio <= 0 or p_win <= 0:
        return min_fraction

    b = win_loss_ratio
    f = (p_win * (b + 1) - 1) / b

    if f <= 0:
        return 0.0  # Kelly says don't trade at all

    if half_kelly:
        f *= 0.5

    return float(max(min_fraction, min(f, max_fraction)))


def kelly_position_size(
    capital: float,
    p_win: float,
    win_loss_ratio: float,
    volatility_scale: float = 1.0,
    half_kelly: bool = True,
) -> float:
    """
    Return absolute position size in USDT.

    Args:
        capital:          Current account equity in USDT.
        p_win:            Model win probability.
        win_loss_ratio:   avg_win / avg_loss.
        volatility_scale: Reduction factor from HullRiskManager (0.25–3.0).
                          Kelly × vol_scale gives final bet.
        half_kelly:       Apply half-Kelly.

    Returns:
        Position size in USDT.
    """
    frac = kelly_fraction(p_win, win_loss_ratio, half_kelly=half_kelly)
    size = capital * frac * volatility_scale
    return round(max(size, 0.0), 2)


def compute_win_loss_ratio(trade_pnls: List[float]) -> float:
    """
    Compute average_win / average_loss from a list of trade PnLs.
    Falls back to 1.5 when insufficient history.
    """
    if not trade_pnls or len(trade_pnls) < 5:
        return 1.5  # neutral default

    wins = [p for p in trade_pnls if p > 0]
    losses = [abs(p) for p in trade_pnls if p < 0]

    if not wins or not losses:
        return 1.5

    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)

    ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
    # Cap between 0.5 and 5.0 to prevent extreme bets from outlier trades
    return float(max(0.5, min(ratio, 5.0)))


class KellySizer:
    """
    Stateful Kelly sizer that maintains a rolling trade history window
    and auto-updates win/loss ratio every trade.
    """

    def __init__(self, window: int = 50, half_kelly: bool = True):
        self.window = window
        self.half_kelly = half_kelly
        self._pnls: List[float] = []

    def record_trade(self, pnl: float) -> None:
        self._pnls.append(pnl)
        if len(self._pnls) > self.window:
            self._pnls.pop(0)

    @property
    def win_loss_ratio(self) -> float:
        return compute_win_loss_ratio(self._pnls)

    @property
    def win_rate(self) -> float:
        if not self._pnls:
            return 0.5
        return sum(1 for p in self._pnls if p > 0) / len(self._pnls)

    def size(
        self,
        capital: float,
        p_win: float,
        volatility_scale: float = 1.0,
    ) -> float:
        return kelly_position_size(
            capital=capital,
            p_win=p_win,
            win_loss_ratio=self.win_loss_ratio,
            volatility_scale=volatility_scale,
            half_kelly=self.half_kelly,
        )

    def circuit_breaker(self, consecutive_losses: int, threshold: int = 3) -> bool:
        """Return True if trading should be halted (too many consecutive losses)."""
        return consecutive_losses >= threshold
