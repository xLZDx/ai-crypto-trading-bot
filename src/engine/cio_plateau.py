"""CIO plateau detection — refuse "spike" winners in favour of "plateau" winners.

Background (Lopez de Prado, AFML §11.6 + the operator's 2026-05-15 review):
    An Optuna study can find a hyper-parameter point with great Sharpe / Sortino
    that is surrounded by neighbours with much worse scores. That's a spike —
    almost certainly overfit, and in live trading it will collapse.

    The right answer is a *plateau*: a region in parameter space where many
    nearby points all score well. A plateau means the strategy is robust to
    the exact parameter values, which is what we want.

Algorithm
---------
For each completed trial t with score s_t and parameter vector p_t:
  1. Find the K nearest neighbours (by Euclidean distance in normalised
     numeric parameter space) among all OTHER completed trials.
  2. Compute the median (or mean) of the neighbours' scores.
  3. Compute neighbourhood width = std(neighbour scores).
  4. The plateau score is:
       plateau_score = median_neighbour_score - α * width
     where α (default 0.5) penalises noisy neighbourhoods.
  5. Rank trials by plateau_score, NOT by best_value.

This module is pure — it does NOT modify the Optuna study, only inspects
its trials and returns a re-ranking. Hook it into `CIOAgent.run()` after
`study.optimize(...)` completes:

    from src.engine.cio_plateau import select_plateau_winner
    plateau = select_plateau_winner(study, k=5, alpha=0.5)
    if plateau and plateau.score >= best.value * 0.85:
        # Plateau winner achieves ≥85% of spike-winner's score AND has
        # robust neighbours. Promote it.
        summary['plateau_winner'] = asdict(plateau)
        ...

The 85% acceptance threshold and α=0.5 width penalty are operator-tunable.
Defaults chosen so:
  - A solitary spike score must be ≥18% better than the best plateau to win.
  - A plateau with mild variance still beats an isolated peak.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class PlateauResult:
    """One trial's robustness analysis. Higher score = better plateau."""
    trial_id: int
    params: dict[str, Any]
    raw_value: float        # original Optuna objective value (typ. Sortino)
    neighbour_mean: float
    neighbour_median: float
    neighbour_std: float
    n_neighbours: int
    plateau_score: float    # neighbour_median - alpha * neighbour_std
    distance_to_best: float # Euclidean distance to the spike-winner in param space


@dataclass
class PlateauSelection:
    """Top-level result of select_plateau_winner."""
    spike_winner: PlateauResult         # the trial with highest raw_value
    plateau_winner: PlateauResult       # the trial with highest plateau_score
    plateau_beats_spike: bool           # whether plateau_winner.plateau_score >= spike.plateau_score
    plateau_recovery_ratio: float       # plateau.raw_value / spike.raw_value (∈ [0, 1])
    recommendation: str                 # 'plateau' | 'spike' | 'tie' | 'insufficient_data'
    all_results: list[PlateauResult]    # full ranking, plateau-score desc


def _normalise_params(
    trials: Sequence[Any],
    numeric_keys: list[str] | None = None,
) -> tuple[list[list[float]], list[str]]:
    """Build a normalised numeric feature matrix from trial.params.

    Categorical / string params are one-hot encoded. Numeric params are
    min-max scaled to [0, 1] across the trial set so distance is unit-free.
    Returns (matrix, ordered_feature_names).
    """
    # Inspect the union of params across trials so a categorical that
    # appears in only some trials still gets one-hot'd consistently.
    all_keys: dict[str, set] = {}
    for t in trials:
        for k, v in (t.params or {}).items():
            all_keys.setdefault(k, set()).add(v)
    numeric_keys_used: list[str] = []
    categorical_keys: dict[str, list] = {}
    for k, vs in all_keys.items():
        # Treat as numeric if ALL values are numeric.
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vs):
            numeric_keys_used.append(k)
        else:
            categorical_keys[k] = sorted({str(v) for v in vs})
    if numeric_keys is not None:
        numeric_keys_used = [k for k in numeric_keys if k in numeric_keys_used]
    # Compute min/max per numeric key.
    bounds: dict[str, tuple[float, float]] = {}
    for k in numeric_keys_used:
        vals = [float(t.params.get(k)) for t in trials
                if isinstance(t.params.get(k), (int, float))]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        if hi == lo:
            hi = lo + 1.0    # avoid 0-width — every trial collapses to 0.5
        bounds[k] = (lo, hi)
    feature_names: list[str] = []
    for k in numeric_keys_used:
        if k in bounds:
            feature_names.append(k)
    for k, values in categorical_keys.items():
        for v in values:
            feature_names.append(f"{k}::{v}")
    # Build matrix.
    matrix: list[list[float]] = []
    for t in trials:
        row: list[float] = []
        for k in numeric_keys_used:
            if k not in bounds:
                continue
            v = t.params.get(k)
            if isinstance(v, (int, float)):
                lo, hi = bounds[k]
                row.append((float(v) - lo) / (hi - lo))
            else:
                row.append(0.5)   # missing → midpoint
        for k, values in categorical_keys.items():
            tv = t.params.get(k)
            for v in values:
                row.append(1.0 if str(tv) == v else 0.0)
        matrix.append(row)
    return matrix, feature_names


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((ai - bi) * (ai - bi) for ai, bi in zip(a, b)))


def select_plateau_winner(
    study,
    *,
    k: int = 5,
    alpha: float = 0.5,
    min_trials: int = 8,
    numeric_keys: list[str] | None = None,
) -> PlateauSelection | None:
    """Compute the plateau winner from an Optuna study.

    Parameters
    ----------
    study : optuna.Study
        Completed (or partial) study; only trials with state COMPLETE and a
        non-None value contribute.
    k : int
        Number of nearest neighbours to consider per trial. Should be a
        small fraction of trials — 5 is sensible for n_trials ∈ [20, 200].
    alpha : float
        Width-penalty coefficient. Higher = stronger preference for narrow
        (uniform) plateaus. 0 = pure median-of-neighbours ranking.
    min_trials : int
        Below this trial count the result is unreliable; we return None.
    numeric_keys : list[str] | None
        Optional whitelist of param keys to treat as numeric. Defaults to
        auto-detection.

    Returns
    -------
    PlateauSelection or None if the study has < min_trials completed trials.
    """
    try:
        import optuna
    except Exception as e:
        logger.warning("cio_plateau: optuna unavailable (%s)", e)
        return None
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and math.isfinite(t.value)]
    if len(completed) < min_trials:
        logger.info("cio_plateau: only %d completed trials, need %d -- skipping",
                    len(completed), min_trials)
        return None

    feats, feature_names = _normalise_params(completed, numeric_keys=numeric_keys)
    if not feats or not feature_names:
        logger.warning("cio_plateau: empty feature matrix -- skipping")
        return None

    # Per-trial plateau scoring.
    results: list[PlateauResult] = []
    spike_trial = max(completed, key=lambda t: t.value)
    spike_idx = completed.index(spike_trial)
    spike_vec = feats[spike_idx]
    for i, t in enumerate(completed):
        my_vec = feats[i]
        # Distance to every other trial.
        dists = []
        for j, other in enumerate(completed):
            if j == i:
                continue
            dists.append((j, _euclidean(my_vec, feats[j])))
        dists.sort(key=lambda x: x[1])
        # k nearest by distance.
        k_eff = min(k, len(dists))
        neighbours = dists[:k_eff]
        if not neighbours:
            continue
        n_vals = [completed[j].value for j, _ in neighbours]
        n_mean = sum(n_vals) / len(n_vals)
        sorted_vals = sorted(n_vals)
        if len(sorted_vals) % 2 == 1:
            n_median = sorted_vals[len(sorted_vals) // 2]
        else:
            mid = len(sorted_vals) // 2
            n_median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        if len(n_vals) >= 2:
            n_var = sum((v - n_mean) ** 2 for v in n_vals) / (len(n_vals) - 1)
            n_std = math.sqrt(n_var)
        else:
            n_std = 0.0
        plateau_score = n_median - alpha * n_std
        dist_to_best = _euclidean(my_vec, spike_vec)
        results.append(PlateauResult(
            trial_id=t.number,
            params=dict(t.params),
            raw_value=float(t.value),
            neighbour_mean=float(n_mean),
            neighbour_median=float(n_median),
            neighbour_std=float(n_std),
            n_neighbours=k_eff,
            plateau_score=float(plateau_score),
            distance_to_best=float(dist_to_best),
        ))

    if not results:
        return None
    results.sort(key=lambda r: r.plateau_score, reverse=True)
    plateau_w = results[0]
    spike_w = next((r for r in results if r.trial_id == spike_trial.number), None)
    if spike_w is None:
        # Spike trial got filtered out (shouldn't happen) — fall back.
        spike_w = max(results, key=lambda r: r.raw_value)
    recovery = (plateau_w.raw_value / spike_w.raw_value) if spike_w.raw_value else 1.0
    plateau_beats = plateau_w.plateau_score > spike_w.plateau_score
    if plateau_w.trial_id == spike_w.trial_id:
        recommendation = "tie"
    elif plateau_beats and recovery >= 0.85:
        recommendation = "plateau"
    else:
        recommendation = "spike"
    return PlateauSelection(
        spike_winner=spike_w,
        plateau_winner=plateau_w,
        plateau_beats_spike=plateau_beats,
        plateau_recovery_ratio=float(recovery),
        recommendation=recommendation,
        all_results=results,
    )


def summarise_for_proposal(sel: PlateauSelection | None) -> dict:
    """Render a PlateauSelection as a dict suitable for cio_proposals.json."""
    if sel is None:
        return {"available": False,
                "reason": "insufficient_trials_or_optuna_unavailable"}
    return {
        "available": True,
        "recommendation": sel.recommendation,
        "plateau_beats_spike": sel.plateau_beats_spike,
        "plateau_recovery_ratio": sel.plateau_recovery_ratio,
        "spike_winner": asdict(sel.spike_winner),
        "plateau_winner": asdict(sel.plateau_winner),
        "top_5_plateau": [asdict(r) for r in sel.all_results[:5]],
        "method": {
            "k_neighbours": sel.spike_winner.n_neighbours,
            "alpha": "see source",
            "scoring": "plateau_score = median(k_neighbours.value) - alpha * std(k_neighbours.value)",
            "acceptance": "plateau_recovery_ratio >= 0.85 AND plateau_score > spike.plateau_score",
        },
    }


__all__ = [
    "PlateauResult", "PlateauSelection",
    "select_plateau_winner", "summarise_for_proposal",
]
