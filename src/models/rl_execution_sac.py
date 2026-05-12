"""
SAC execution agent — Phase 3 PRIMARY.

Soft Actor-Critic for the (size_fraction, limit_offset) continuous action
space defined in `rl_base.py`. Off-policy, sample-efficient — the right
choice for execution where samples are expensive to collect from the live
exchange.

Reward (plan §11):    R = PnL - λ * inventory²

Architecture is intentionally minimal (~250 lines): one Gaussian policy +
two Q-critics + automatic entropy tuning. PyTorch native, no
stable-baselines3 dependency.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np

from .rl_base import (
    ACTION_DIM, BaseExecutionAgent, ContinuousBox, ReplayBuffer, Transition,
    make_action_space,
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


_LOG_STD_MIN = -5.0
_LOG_STD_MAX =  2.0


if _TORCH_OK:

    class _GaussianPolicy(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.mu  = nn.Linear(hidden, action_dim)
            self.log_std = nn.Linear(hidden, action_dim)

        def forward(self, obs):
            h = F.relu(self.fc1(obs))
            h = F.relu(self.fc2(h))
            mu = self.mu(h)
            log_std = self.log_std(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
            return mu, log_std

        def sample(self, obs):
            """Reparameterized sample with tanh squash and log-prob correction."""
            mu, log_std = self.forward(obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mu, std)
            x = normal.rsample()           # pre-squash
            action = torch.tanh(x)
            # Clamp action away from ±1 BEFORE log to avoid log(0) which on
            # CUDA fires `device-side assert triggered`. The 1e-5 floor matches
            # SB3's recommended numerical-stability epsilon.
            action_clipped = action.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
            log_prob = normal.log_prob(x) - torch.log(1 - action_clipped.pow(2) + 1e-5)
            return action, log_prob.sum(dim=-1, keepdim=True), torch.tanh(mu)

    class _QNet(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim + action_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.q   = nn.Linear(hidden, 1)

        def forward(self, obs, action):
            h = torch.cat([obs, action], dim=-1)
            h = F.relu(self.fc1(h))
            h = F.relu(self.fc2(h))
            return self.q(h)

else:
    _GaussianPolicy = _QNet = None  # type: ignore


class SACAgent(BaseExecutionAgent):
    """Off-policy SAC for continuous execution actions."""

    name = "sac"

    def __init__(
        self,
        obs_dim: int,
        action_space: ContinuousBox | None = None,
        *,
        hidden: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        target_entropy: float | None = None,
        device: str = "auto",
    ):
        super().__init__(obs_dim, action_space)
        if not _TORCH_OK:
            raise ImportError("SACAgent requires torch.")
        self.gamma = gamma
        self.tau = tau
        self.device = (torch.device("cuda") if device == "auto" and torch.cuda.is_available()
                       else torch.device(device if device != "auto" else "cpu"))

        self.actor = _GaussianPolicy(obs_dim, ACTION_DIM, hidden).to(self.device)
        self.q1 = _QNet(obs_dim, ACTION_DIM, hidden).to(self.device)
        self.q2 = _QNet(obs_dim, ACTION_DIM, hidden).to(self.device)
        self.q1_target = _QNet(obs_dim, ACTION_DIM, hidden).to(self.device)
        self.q2_target = _QNet(obs_dim, ACTION_DIM, hidden).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=lr)

        # Automatic entropy tuning — target H = -|A|
        self.target_entropy = float(target_entropy
                                    if target_entropy is not None
                                    else -ACTION_DIM)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)

    # ── Inference ───────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        # Sanitize obs — upstream can emit NaN/Inf when ex.step returns
        # degenerate state (e.g. zero spreads, divide-by-zero microprice).
        obs = np.nan_to_num(np.asarray(obs, dtype=np.float32),
                            nan=0.0, posinf=1e6, neginf=-1e6)
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            sample, _, mu = self.actor.sample(o)
            a = mu if deterministic else sample
            a = a.clamp(-1.0, 1.0)
            return a.squeeze(0).cpu().numpy()

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, buffer: ReplayBuffer, batch_size: int = 64) -> dict:
        if len(buffer) < batch_size:
            return {"skipped": True}

        o, a, r, no, d = buffer.sample(batch_size)
        # Sanitize batch — a single NaN row can poison gradients, and on CUDA
        # produces opaque `device-side assert triggered` failures with no line.
        o  = np.nan_to_num(o,  nan=0.0, posinf=1e6, neginf=-1e6)
        a  = np.nan_to_num(a,  nan=0.0, posinf=1.0, neginf=-1.0)
        r  = np.nan_to_num(r,  nan=0.0, posinf=1e6, neginf=-1e6)
        no = np.nan_to_num(no, nan=0.0, posinf=1e6, neginf=-1e6)
        a  = np.clip(a, -1.0, 1.0).astype(np.float32)
        o = torch.as_tensor(o, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device).unsqueeze(-1)
        no = torch.as_tensor(no, device=self.device)
        d = torch.as_tensor(d.astype(np.float32), device=self.device).unsqueeze(-1)
        alpha = self.log_alpha.exp().detach()

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            na, na_logp, _ = self.actor.sample(no)
            tq1 = self.q1_target(no, na)
            tq2 = self.q2_target(no, na)
            tq = torch.min(tq1, tq2) - alpha * na_logp
            target = r + self.gamma * (1.0 - d) * tq

        q1_loss = F.mse_loss(self.q1(o, a), target)
        q2_loss = F.mse_loss(self.q2(o, a), target)
        self.opt_q1.zero_grad(); q1_loss.backward(); self.opt_q1.step()
        self.opt_q2.zero_grad(); q2_loss.backward(); self.opt_q2.step()

        # ── Actor update ──────────────────────────────────────────────────
        sampled, sampled_logp, _ = self.actor.sample(o)
        q1_pi = self.q1(o, sampled)
        q2_pi = self.q2(o, sampled)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * sampled_logp - q_pi).mean()
        self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

        # ── Alpha update ──────────────────────────────────────────────────
        alpha_loss = -(self.log_alpha *
                       (sampled_logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()

        # ── Target soft update ────────────────────────────────────────────
        with torch.no_grad():
            for t, s in zip(self.q1_target.parameters(), self.q1.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * s.data)
            for t, s in zip(self.q2_target.parameters(), self.q2.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * s.data)

        return {
            "q1_loss":     float(q1_loss.item()),
            "q2_loss":     float(q2_loss.item()),
            "actor_loss":  float(actor_loss.item()),
            "alpha":       float(alpha.item()),
            "alpha_loss":  float(alpha_loss.item()),
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor":      self.actor.state_dict(),
            "q1":         self.q1.state_dict(),
            "q2":         self.q2.state_dict(),
            "q1_target":  self.q1_target.state_dict(),
            "q2_target":  self.q2_target.state_dict(),
            "log_alpha":  self.log_alpha.detach().cpu(),
            "obs_dim":    self.obs_dim,
        }, path)

    def load(self, path: str) -> None:
        # Phase A7 (2026-05-12): weights_only=True — checkpoint
        # contains state_dicts + log_alpha tensor + obs_dim int,
        # all safe for weights_only mode.
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))


__all__ = ["SACAgent"]
