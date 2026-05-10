"""
Multi-Agent Self-Play Environment — Phase 3, Level 3.

Per updated_architecture_plan_en.md §9 (Multi-Agent Self-Play):
    "OFT-alpha, RL market makers, and other agents train by competing
    against each other."

This module builds on `SyntheticExchange` to host *multiple* policies that
share a single book. One policy is the "alpha" (signal-driven, fed by OFT
predictions) and the others are "adversaries" (random, momentum, or trained
RL market-makers). Their orders perturb the book between ticks, giving the
alpha a more realistic execution environment than a static replay.

For Phase 3 we ship the orchestration skeleton + a NoiseAgent and
MomentumAgent baseline so the core training loop can be exercised. Trained
RL market-makers slot in by name lookup.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .synthetic_exchange import SyntheticExchange, ImpactModel

logger = logging.getLogger(__name__)


# ─── Built-in baseline agents ────────────────────────────────────────────────

class NoiseAgent:
    """Random Gaussian-noise actions — useful as a sanity baseline."""
    name = "noise"

    def __init__(self, sigma: float = 0.1, rng: np.random.Generator | None = None):
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()

    def act(self, obs: dict) -> tuple[float, float]:
        return (
            float(np.clip(self.rng.normal(0, self.sigma), -1, 1)),
            float(np.clip(self.rng.normal(0, self.sigma), -1, 1)),
        )


class MomentumAgent:
    """Simple imbalance-driven momentum agent — buys when bids dominate."""
    name = "momentum"

    def __init__(self, k_size: float = 0.5, k_offset: float = 0.05):
        self.k_size = k_size
        self.k_offset = k_offset

    def act(self, obs: dict) -> tuple[float, float]:
        i = float(obs.get("imbalance", 0.0))
        return float(self.k_size * i), float(self.k_offset)


# ─── Multi-agent env ─────────────────────────────────────────────────────────

@dataclass
class MultiAgentSession:
    """Per-step bookkeeping for each participating agent."""
    name:        str
    cum_reward:  float = 0.0
    n_steps:     int = 0
    last_action: tuple[float, float] | None = None


class MultiAgentEnv:
    """Wraps `SyntheticExchange` with multiple competing policies.

    Usage:
        env = MultiAgentEnv(book_iter, alpha_agent=oft_alpha,
                            adversaries=[NoiseAgent(), MomentumAgent()])
        env.reset()
        while not env.done:
            metrics = env.step()
        report = env.report()
    """

    def __init__(
        self,
        book_iter: Iterable[dict],
        *,
        alpha_agent,
        adversaries: list | None = None,
        impact: ImpactModel | None = None,
    ):
        self.exchange = SyntheticExchange(book_iter, impact=impact)
        self.alpha_agent = alpha_agent
        self.adversaries = list(adversaries or [])
        self.sessions: dict[str, MultiAgentSession] = {}
        self.done = False
        self._obs: dict | None = None

    def reset(self) -> dict:
        self._obs = self.exchange.reset()
        self.done = False
        names = [getattr(self.alpha_agent, "name", "alpha")] + \
                [getattr(a, "name", f"adv{i}") for i, a in enumerate(self.adversaries)]
        self.sessions = {n: MultiAgentSession(name=n) for n in names}
        return self._obs

    def step(self) -> dict:
        """All agents act; their order flow accumulates against the book."""
        if self._obs is None:
            self.reset()
        if self.done:
            return {"done": True}

        # 1) Adversaries act first — their orders perturb the next state seen
        #    by the alpha. We accumulate their net impact via repeated step
        #    calls on the exchange (each agent gets their own micro-step).
        rewards: dict[str, float] = {}
        last_obs = self._obs
        for adv in self.adversaries:
            action = adv.act(last_obs)
            obs, r, done, info = self.exchange.step(action)
            sess = self.sessions[getattr(adv, "name", "adv")]
            sess.cum_reward += r
            sess.n_steps += 1
            sess.last_action = action
            rewards[sess.name] = r
            last_obs = obs
            if done:
                self.done = True
                self._obs = obs
                return {"done": True, "rewards": rewards}

        # 2) Alpha acts last on the perturbed book.
        alpha_action = self.alpha_agent.act(last_obs)
        if isinstance(alpha_action, np.ndarray):
            alpha_action = (float(alpha_action[0]), float(alpha_action[1]))
        obs, r, done, info = self.exchange.step(alpha_action)
        sess = self.sessions[getattr(self.alpha_agent, "name", "alpha")]
        sess.cum_reward += r
        sess.n_steps += 1
        sess.last_action = alpha_action
        rewards[sess.name] = r

        self._obs = obs
        self.done = done
        return {"done": done, "rewards": rewards, "obs": obs, "info": info}

    def report(self) -> dict:
        return {
            "agents": {
                s.name: {
                    "cum_reward":   s.cum_reward,
                    "n_steps":      s.n_steps,
                    "last_action":  s.last_action,
                }
                for s in self.sessions.values()
            },
            "exchange_state": self.exchange.state.__dict__ if self.exchange.state else None,
        }


__all__ = [
    "MultiAgentEnv", "MultiAgentSession",
    "NoiseAgent", "MomentumAgent",
]
