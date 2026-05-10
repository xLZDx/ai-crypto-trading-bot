"""
RL execution agents — shared abstractions.

Phase 3, Level 3 (Execution & Simulation). Both SAC (primary) and PPO (backup)
inherit from `BaseExecutionAgent` and operate on the same:

    Observation:  features extracted from `SyntheticExchange._observation()`
    Action:       (size_fraction, limit_offset) — both ∈ [-1, +1]
    Reward:       PnL - λ * inventory²  (HFT-style inventory hedging, plan §11)

Selection logic (used by `src.engine.order_manager`):
    Default: SAC.
    Failover: switch to PPO if last 100 trades' Sharpe < 0 OR action variance
    spikes 3σ above training mean — indicates SAC has drifted.

This module deliberately stays small: it defines the contract and a shared
ReplayBuffer. The SAC and PPO files implement the actual update rules.
"""
from __future__ import annotations

import abc
import logging
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# Continuous action space dimensions: (size_fraction, limit_offset)
ACTION_DIM = 2

# Inventory penalty coefficient — plan §11 reward = PnL - λ * inventory²
DEFAULT_INVENTORY_LAMBDA = 0.05


# ─── Spaces ──────────────────────────────────────────────────────────────────

@dataclass
class ContinuousBox:
    low:  np.ndarray
    high: np.ndarray

    @property
    def shape(self):
        return self.low.shape

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.low, self.high)


def make_action_space() -> ContinuousBox:
    """Action ∈ [-1, +1]² — (size_fraction, limit_offset)."""
    return ContinuousBox(low=-np.ones(ACTION_DIM), high=np.ones(ACTION_DIM))


def make_observation_space(n_features: int) -> ContinuousBox:
    """Observation features are normalized — bounded to [-10, +10]."""
    return ContinuousBox(low=-10.0 * np.ones(n_features),
                         high=10.0 * np.ones(n_features))


# ─── Observation extractor ───────────────────────────────────────────────────

OBS_KEYS = (
    "imbalance", "spread_bps", "inventory", "v_bid", "v_ask",
)


def obs_dict_to_vector(obs: dict, *, max_inventory: float = 10.0) -> np.ndarray:
    """Project the dict obs from `SyntheticExchange` into a fixed-length vector."""
    return np.array([
        float(obs.get("imbalance",  0.0)),
        float(obs.get("spread_bps", 0.0)) * 1e4,                     # bps → "10s of bps"
        float(obs.get("inventory",  0.0)) / max(max_inventory, 1.0),  # ∈ [-1,1]
        np.log1p(float(obs.get("v_bid", 0.0))),
        np.log1p(float(obs.get("v_ask", 0.0))),
    ], dtype=np.float32)


# ─── Reward shaping (plan §11) ──────────────────────────────────────────────

def shaped_reward(
    raw_pnl: float,
    inventory: float,
    *,
    inventory_lambda: float = DEFAULT_INVENTORY_LAMBDA,
) -> float:
    """R = PnL - λ * inventory²  (HFT-style inventory hedging)."""
    return float(raw_pnl) - float(inventory_lambda) * (float(inventory) ** 2)


# ─── Replay buffer (off-policy SAC needs this; PPO does not) ─────────────────

class Transition(NamedTuple):
    obs:      np.ndarray
    action:   np.ndarray
    reward:   float
    next_obs: np.ndarray
    done:     bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, obs_dim: int = 5):
        self.capacity = capacity
        self._buf: deque[Transition] = deque(maxlen=capacity)
        self.obs_dim = obs_dim

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        if len(self._buf) < batch_size:
            raise ValueError("not enough samples")
        idx = rng.integers(0, len(self._buf), size=batch_size)
        batch = [self._buf[int(i)] for i in idx]
        return (
            np.stack([t.obs for t in batch]).astype(np.float32),
            np.stack([t.action for t in batch]).astype(np.float32),
            np.array([t.reward for t in batch], dtype=np.float32),
            np.stack([t.next_obs for t in batch]).astype(np.float32),
            np.array([t.done for t in batch], dtype=bool),
        )


# ─── Base agent contract ────────────────────────────────────────────────────

class BaseExecutionAgent(abc.ABC):
    """Common interface for SAC and PPO execution agents."""

    name: str = "base"

    def __init__(self, obs_dim: int, action_space: ContinuousBox | None = None):
        self.obs_dim = obs_dim
        self.action_space = action_space or make_action_space()

    @abc.abstractmethod
    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> np.ndarray: ...

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """One gradient step. Returns a dict of training metrics."""

    def save(self, path: str) -> None:  # default no-op for stubs
        pass

    def load(self, path: str) -> None:  # default no-op
        pass


__all__ = [
    "ContinuousBox", "make_action_space", "make_observation_space",
    "obs_dict_to_vector", "shaped_reward",
    "Transition", "ReplayBuffer",
    "BaseExecutionAgent",
    "ACTION_DIM", "DEFAULT_INVENTORY_LAMBDA", "OBS_KEYS",
]
