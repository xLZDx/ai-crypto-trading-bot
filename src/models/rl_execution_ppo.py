"""
PPO execution agent — Phase 3 BACKUP.

On-policy Proximal Policy Optimization. Used as the failover when SAC
drifts (Sharpe < 0 over last 100 trades, or action-variance spike). Stable,
predictable, slower-learning — the workhorse.

Same observation / action / reward contract as `SACAgent`. Different
optimisation regime: collects a fixed-size rollout, performs K epochs of
clipped-surrogate updates, then discards the rollout (no replay buffer).
"""
from __future__ import annotations

import logging
import math
from collections import namedtuple

import numpy as np

from .rl_base import (
    ACTION_DIM, BaseExecutionAgent, ContinuousBox, make_action_space,
)

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    _TORCH_OK = False


Rollout = namedtuple("Rollout", "obs actions log_probs returns advantages")


if _TORCH_OK:

    class _ActorCritic(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )
            self.mu      = nn.Linear(hidden, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
            self.value   = nn.Linear(hidden, 1)

        def forward(self, obs):
            h = self.shared(obs)
            return self.mu(h), self.log_std.expand_as(self.mu(h)), self.value(h).squeeze(-1)

        def policy(self, obs):
            mu, log_std, _ = self.forward(obs)
            return torch.distributions.Normal(mu, log_std.exp())

else:
    _ActorCritic = None  # type: ignore


class PPOAgent(BaseExecutionAgent):
    """Clipped PPO with GAE(λ) advantage."""

    name = "ppo"

    def __init__(
        self,
        obs_dim: int,
        action_space: ContinuousBox | None = None,
        *,
        hidden: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip: float = 0.2,
        lr: float = 3e-4,
        epochs: int = 4,
        device: str = "auto",
    ):
        super().__init__(obs_dim, action_space)
        if not _TORCH_OK:
            raise ImportError("PPOAgent requires torch.")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        self.device = (torch.device("cuda") if device == "auto" and torch.cuda.is_available()
                       else torch.device(device if device != "auto" else "cpu"))

        self.net = _ActorCritic(obs_dim, ACTION_DIM, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ── Inference ───────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            mu, log_std, _ = self.net(o)
            if deterministic:
                a = torch.tanh(mu)
            else:
                dist = torch.distributions.Normal(mu, log_std.exp())
                a = torch.tanh(dist.sample())
            return a.squeeze(0).cpu().numpy()

    # ── GAE ────────────────────────────────────────────────────────────────

    def compute_gae(self, rewards, values, dones, last_value):
        """Generalised Advantage Estimation."""
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = values[t + 1] if t + 1 < len(values) else last_value
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values[:len(adv)]
        return adv, returns

    # ── Update ─────────────────────────────────────────────────────────────

    def update(self, rollout: Rollout, batch_size: int = 64) -> dict:
        """Run K epochs of clipped-surrogate PPO on one rollout."""
        n = len(rollout.obs)
        if n == 0:
            return {"skipped": True}

        obs       = torch.as_tensor(rollout.obs, dtype=torch.float32, device=self.device)
        actions   = torch.as_tensor(rollout.actions, dtype=torch.float32, device=self.device)
        old_logp  = torch.as_tensor(rollout.log_probs, dtype=torch.float32, device=self.device)
        returns   = torch.as_tensor(rollout.returns, dtype=torch.float32, device=self.device)
        adv       = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idxs = np.arange(n)
        last_metrics = {}
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, batch_size):
                b = idxs[start:start + batch_size]
                mu, log_std, value = self.net(obs[b])
                dist = torch.distributions.Normal(mu, log_std.exp())
                # The agent stores tanh-squashed actions; for log-prob we
                # need the pre-squash latent. For Phase 3 sanity we ignore
                # the squash correction (small-action regime).
                logp = dist.log_prob(actions[b]).sum(dim=-1)
                ratio = torch.exp(logp - old_logp[b])
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, returns[b])
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                last_metrics = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss":  float(value_loss.item()),
                    "entropy":     float(entropy.item()),
                    "ratio_mean":  float(ratio.mean().item()),
                }
        return last_metrics

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "obs_dim": self.obs_dim}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])


__all__ = ["PPOAgent", "Rollout"]
