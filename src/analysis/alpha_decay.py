"""
Alpha Decay — Phase 3, Level 3 (Execution & Simulation).

Replaces the hard `max_hold_bars` exit with an exponential signal-decay model
per updated_architecture_plan_en.md §12:

    def apply_alpha_decay(signal_strength, time_in_trade, decay_rate=0.1):
        return signal_strength * np.exp(-decay_rate * time_in_trade)

    # In the loop: if apply_alpha_decay(...) < threshold, close the position.

Why exponential, not linear:
    The information advantage of a microstructure signal decays multiplicatively
    as new orders arrive and the book repaints. Linear decay would over-hold
    in fast regimes and under-hold in slow ones.

Decay-rate guidance (`decay_rate`, units: 1/bars):
    Scalping (1m bars):   0.30 — half-life ≈ 2.3 bars
    Intraday (15m bars):  0.10 — half-life ≈ 6.9 bars  (plan default)
    Swing (1h bars):      0.04 — half-life ≈ 17  bars
"""
from __future__ import annotations

from typing import Iterable

import math
import numpy as np


def apply_alpha_decay(
    signal_strength: float,
    time_in_trade: float,
    decay_rate: float = 0.1,
) -> float:
    """Decayed signal value.

        decayed = signal_strength * exp(-decay_rate * time_in_trade)

    Both `time_in_trade` and `decay_rate` are in matching units (typically
    bars; sub-bar callers may pass fractional values).
    """
    return float(signal_strength) * math.exp(-float(decay_rate) * float(time_in_trade))


def half_life(decay_rate: float) -> float:
    """Number of time-units before signal halves: ln(2) / decay_rate."""
    if decay_rate <= 0:
        return float("inf")
    return math.log(2.0) / decay_rate


def should_exit(
    signal_strength: float,
    time_in_trade: float,
    decay_rate: float = 0.1,
    exit_threshold: float = 0.2,
) -> bool:
    """True iff the decayed signal has fallen below `exit_threshold`."""
    return apply_alpha_decay(signal_strength, time_in_trade, decay_rate) < exit_threshold


def decay_curve(signal_strength: float, decay_rate: float, t_max: int) -> np.ndarray:
    """Return the full decay curve over [0, t_max] for plotting / inspection."""
    t = np.arange(t_max + 1, dtype=float)
    return signal_strength * np.exp(-decay_rate * t)


__all__ = ["apply_alpha_decay", "half_life", "should_exit", "decay_curve"]
