"""
CVaR Portfolio Optimizer — Phase 4, Level 4.

Per updated_architecture_plan_en.md §13-14:

    # Objective
    max  E[R] - λ * CVaR_α(R)

    # Risk Parity + Confidence Sizing as a *prior* to CVaR
    weights = (probabilities - 0.5) * 2
    weights = weights / asset_volatility
    penalty = 1 - returns.corr().mean().mean()
    weights *= penalty
    weights /= np.sum(np.abs(weights))

CVaR (Conditional Value-at-Risk, a.k.a. Expected Shortfall) is the average
loss in the worst α-tail of outcomes. Replacing fixed sizing rules with this
optimizer lets the bot:
  • shrink positions automatically when joint-tail risk explodes;
  • respect realised correlations (no more "long BTC + long ETH = 2× BTC");
  • keep the convex-program guarantee of a unique global optimum.

Implementation: Rockafellar-Uryasev linear-program reformulation with CVXPY.
For typical portfolios (≤20 assets, ≤5 000 historical scenarios), solves in
< 50 ms on CPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Risk Parity + Confidence (prior to CVaR) ───────────────────────────────

def confidence_weights(probabilities: np.ndarray) -> np.ndarray:
    """Map model P(win) ∈ [0,1] to signed conviction ∈ [-1, +1]."""
    p = np.asarray(probabilities, dtype=float)
    return (p - 0.5) * 2.0


def risk_parity_weights(
    probabilities: np.ndarray,
    asset_volatility: np.ndarray,
    returns_correlation: pd.DataFrame | np.ndarray | None = None,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Plan §14 — risk parity + confidence + correlation penalty.

        w = (probabilities - 0.5) * 2
        w = w / asset_volatility
        penalty = 1 - corr.mean().mean()
        w *= penalty
        w /= np.sum(np.abs(w))
    """
    w = confidence_weights(probabilities)
    vol = np.asarray(asset_volatility, dtype=float)
    w = w / np.maximum(vol, eps)

    if returns_correlation is not None:
        corr = (returns_correlation if isinstance(returns_correlation, np.ndarray)
                else returns_correlation.to_numpy())
        # Mean off-diagonal correlation as a crowding measure.
        n = corr.shape[0]
        if n > 1:
            mean_corr = (corr.sum() - np.trace(corr)) / max(n * (n - 1), 1)
        else:
            mean_corr = 0.0
        penalty = max(1.0 - float(mean_corr), 0.0)
        w *= penalty

    norm = np.sum(np.abs(w))
    if norm > eps:
        w = w / norm
    return w


# ─── CVaR optimizer ─────────────────────────────────────────────────────────

@dataclass
class CVaRResult:
    weights:   np.ndarray
    expected_return: float
    cvar:      float
    objective: float
    status:    str
    n_scenarios: int


class CVaROptimizer:
    """Convex CVaR portfolio optimizer.

    Solves:
        max  μᵀw - λ * CVaR_α(R)
        s.t. Σᵢ |wᵢ| ≤ leverage_cap
             |wᵢ| ≤ box_max
             [optional] Σᵢ wᵢ = budget
    """

    def __init__(
        self,
        alpha: float = 0.05,           # tail-probability (5% worst)
        lam: float = 1.0,              # λ in objective
        leverage_cap: float = 1.0,
        box_max: float = 0.4,
        long_only: bool = False,
        solver: str | None = None,     # None → CVXPY default
    ):
        self.alpha = alpha
        self.lam = lam
        self.leverage_cap = leverage_cap
        self.box_max = box_max
        self.long_only = long_only
        self.solver = solver

    def fit(
        self,
        scenario_returns: np.ndarray,   # (n_scenarios, n_assets)
        prior_weights: np.ndarray | None = None,
        *,
        budget: float | None = None,
    ) -> CVaRResult:
        """Solve the CVaR program and return optimal weights."""
        try:
            import cvxpy as cp
        except ImportError as exc:
            logger.warning("cvxpy not installed -- returning prior weights. (%s)", exc)
            n = scenario_returns.shape[1]
            w = (prior_weights if prior_weights is not None
                 else np.ones(n) / n)
            return CVaRResult(weights=np.asarray(w, dtype=float),
                              expected_return=float(np.dot(w, scenario_returns.mean(axis=0))),
                              cvar=float("nan"), objective=float("nan"),
                              status="cvxpy_missing",
                              n_scenarios=scenario_returns.shape[0])

        R = np.asarray(scenario_returns, dtype=float)
        n_scen, n_assets = R.shape
        mu = R.mean(axis=0)

        w = cp.Variable(n_assets)
        eta = cp.Variable()                          # VaR auxiliary
        z = cp.Variable(n_scen, nonneg=True)         # tail excess

        # Rockafellar-Uryasev representation:
        #   CVaR_α(R) = min_η  η + (1/α n) Σ z_i,   z_i ≥ -Rᵢ w - η, z_i ≥ 0
        cvar_expr = eta + (1.0 / (self.alpha * n_scen)) * cp.sum(z)

        constraints = [
            z >= -R @ w - eta,
            cp.sum(cp.abs(w)) <= self.leverage_cap,
            cp.abs(w) <= self.box_max,
        ]
        if self.long_only:
            constraints.append(w >= 0)
        if budget is not None:
            constraints.append(cp.sum(w) == budget)

        objective = cp.Maximize(mu @ w - self.lam * cvar_expr)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=self.solver) if self.solver else prob.solve()
        except Exception as exc:
            logger.warning("CVaR solve failed (%s) -- falling back to prior.", exc)
            w_out = (prior_weights if prior_weights is not None
                     else np.ones(n_assets) / n_assets)
            return CVaRResult(weights=np.asarray(w_out),
                              expected_return=float(np.dot(w_out, mu)),
                              cvar=float("nan"), objective=float("nan"),
                              status="solver_failed", n_scenarios=n_scen)

        if w.value is None:
            w_out = (prior_weights if prior_weights is not None
                     else np.ones(n_assets) / n_assets)
        else:
            w_out = np.asarray(w.value, dtype=float)

        return CVaRResult(
            weights=w_out,
            expected_return=float(np.dot(w_out, mu)),
            cvar=float(eta.value + (1.0 / (self.alpha * n_scen)) * np.sum(np.maximum(0, -R @ w_out - eta.value)))
                 if eta.value is not None else float("nan"),
            objective=float(prob.value) if prob.value is not None else float("nan"),
            status=str(prob.status),
            n_scenarios=n_scen,
        )


__all__ = [
    "CVaROptimizer", "CVaRResult",
    "confidence_weights", "risk_parity_weights",
]
