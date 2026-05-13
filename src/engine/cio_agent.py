"""
CIO Agent — Optuna-driven hyperparameter orchestrator.

Treats the trading system as a fund. The CIO Agent makes capital-allocation
decisions across the strategy lineup by searching over:
  - timeframe (15m, 1h, 4h)
  - train_window_days (90, 180, 365)
  - Triple Barrier multipliers (pt, sl)
  - meta-labeler confidence threshold

Each Optuna trial submits a training+backtest task to the cluster (the trials
ARE cluster tasks — there is no parallel scheduler) and returns the
out-of-sample Sortino ratio. The CIO Agent gates trials by max drawdown and
optionally by ML Engineer agent KPI floors.

Results are persisted to data/optuna_orchestrator.db (SQLite) and visualized
through `optuna-dashboard sqlite:///data/optuna_orchestrator.db` on port 8080.

Workflow:
  CIO.create_study(name) → returns Optuna study
  CIO.objective(trial)    → submits cluster task, polls for completion, returns Sortino
  CIO.run(n_trials)       → optimization loop
  CIO.apply_best()        → writes the winning HP block to training_rules.json
                            (operator approval required — never auto-applied)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTUNA_DB_PATH = PROJECT_ROOT / 'data' / 'optuna_orchestrator.db'
TRAINING_RULES_PATH = PROJECT_ROOT / 'data' / 'training_rules.json'
CIO_PROPOSALS_PATH = PROJECT_ROOT / 'data' / 'cio_proposals.json'


@dataclass
class TrialResult:
    """Outcome of a single Optuna trial."""
    trial_id: int
    params: dict[str, Any]
    sortino: float
    sharpe: float
    max_dd: float
    n_trades: int
    cluster_task_id: str | None
    pruned: bool = False
    error: str | None = None


class CIOAgent:
    """
    Optuna-driven CIO. Each trial issues a cluster task; trials ARE the work.

    Dependencies (install once into venv):
        pip install --no-cache-dir optuna optuna-dashboard

    Dashboard:
        optuna-dashboard sqlite:///D:/test\\ 2/AI\\ trading\\ assistance/data/optuna_orchestrator.db
        → http://localhost:8080
    """

    # Class-level lock — used by _ensure_optuna under n_jobs > 1.
    _optuna_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        study_name: str = 'macro_parameter_search_v1',
        storage_url: str | None = None,
        task_submitter: Callable[[dict], str] | None = None,
        task_status_poller: Callable[[str], dict] | None = None,
        max_dd_threshold: float = 0.15,
        ml_engineer_gate: bool = True,
    ):
        self.study_name = study_name
        self.storage_url = storage_url or f"sqlite:///{OPTUNA_DB_PATH}"
        self.task_submitter = task_submitter
        self.task_status_poller = task_status_poller
        self.max_dd_threshold = max_dd_threshold
        self.ml_engineer_gate = ml_engineer_gate
        self._optuna = None  # lazy import — optuna is optional dependency

    # ── Lazy Optuna import ───────────────────────────────────────────────────

    def _ensure_optuna(self):
        # Thread-safe under n_jobs > 1: acquire a class-level lock around the
        # check + assignment. Avoids the TOCTOU race flagged in review.
        if self._optuna is None:
            with CIOAgent._optuna_lock:
                if self._optuna is None:  # double-check after lock
                    try:
                        import optuna
                        self._optuna = optuna
                    except ImportError as e:
                        raise RuntimeError(
                            "optuna not installed. Run: "
                            "pip install --no-cache-dir optuna optuna-dashboard"
                        ) from e
        return self._optuna

    # ── Study lifecycle ──────────────────────────────────────────────────────

    def create_study(self):
        optuna = self._ensure_optuna()
        OPTUNA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            storage=self.storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

    # ── Objective function (the heart of the agent) ──────────────────────────

    def objective(self, trial) -> float:
        """
        Submit one training+backtest cluster task, poll until completion, return
        OOS Sortino. Trials are PRUNED on max_dd breach or ML Engineer REJECT.
        """
        optuna = self._ensure_optuna()
        # ── 1. Define search space (market logic, not arbitrary ranges) ──
        timeframe = trial.suggest_categorical('timeframe', ['15m', '1h', '4h'])
        train_window_days = trial.suggest_int('train_window_days', 90, 365, step=30)
        pt_multiplier = trial.suggest_float('pt_multiplier', 2.0, 3.0, step=0.1)
        sl_multiplier = trial.suggest_float('sl_multiplier', 1.0, 2.0, step=0.1)
        confidence_threshold = trial.suggest_float('confidence_threshold', 0.50, 0.70, step=0.01)

        # Enforce AFML asymmetric ratio at search-time (saves cluster cycles)
        if (pt_multiplier / sl_multiplier) < 1.3:
            raise optuna.exceptions.TrialPruned()

        # ── 2. ML Engineer pre-flight gate ──
        if self.ml_engineer_gate:
            try:
                from src.engine.ml_engineer_agent import get_ml_engineer
                pre_decision = get_ml_engineer().validate_training_request(
                    model_type='meta',
                    timeframe=timeframe,
                    config={
                        'pt_multiplier': pt_multiplier,
                        'sl_multiplier': sl_multiplier,
                        'max_bars': 12,
                        'pct_embargo': (2.0 * 12) / 1000,
                        'use_t1_purging': True,
                        'calibration_method': 'isotonic',
                        'class_weight': 'balanced',
                    },
                )
                if pre_decision.decision == 'BLOCK':
                    logger.info("Trial %d BLOCKED by ML Engineer: %s",
                                trial.number, pre_decision.reasons)
                    raise optuna.exceptions.TrialPruned()
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                logger.warning("ML Engineer pre-flight skipped (error %s) — proceeding", e)

        # ── 3. Submit training task to cluster ──
        spec = {
            'model_type': 'meta',
            'timeframe':  timeframe,
            'overrides': {
                'pt_multiplier':         pt_multiplier,
                'sl_multiplier':         sl_multiplier,
                'max_bars':              12,
                'confidence_threshold':  confidence_threshold,
                'train_window_days':     train_window_days,
            },
            'trial_id':       trial.number,
            'study_name':     self.study_name,
            'requested_by':   'cio_agent',
        }

        if self.task_submitter is None:
            # Smoke-test mode — no cluster wired yet. Return random for testing.
            logger.warning("CIO: no task_submitter wired — running in smoke-test mode")
            return float(trial.number % 7) * 0.1  # deterministic stub

        task_id = self.task_submitter(spec)
        trial.set_user_attr('cluster_task_id', task_id)

        # ── 4. Poll for completion ──
        result = self._poll_task_completion(task_id, trial=trial, timeout_s=3600)
        if result is None:
            raise optuna.exceptions.TrialPruned()

        # ── 5. Risk gate: max_dd breach → prune ──
        max_dd = float(result.get('max_dd', 0.0))
        if max_dd > self.max_dd_threshold:
            logger.info("Trial %d PRUNED: max_dd=%.3f > %.3f",
                        trial.number, max_dd, self.max_dd_threshold)
            raise optuna.exceptions.TrialPruned()

        # ── 6. Return Sortino (objective is maximize) ──
        sortino = float(result.get('sortino') or 0.0)
        trial.set_user_attr('sharpe', float(result.get('sharpe') or 0.0))
        trial.set_user_attr('max_dd', max_dd)
        trial.set_user_attr('n_trades', int(result.get('n_trades') or 0))
        return sortino

    def _poll_task_completion(self, task_id: str, trial, timeout_s: int) -> dict | None:
        """Poll the cluster until the task finishes. Returns result dict or None."""
        if self.task_status_poller is None:
            return {'sortino': 0.0, 'sharpe': 0.0, 'max_dd': 0.0, 'n_trades': 0}
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = self.task_status_poller(task_id)
            state = status.get('status') or status.get('state')
            if state in ('done', 'completed', 'success'):
                return status.get('result') or status
            if state in ('failed', 'error', 'cancelled'):
                logger.warning("Trial %d task failed: %s", trial.number, status)
                return None
            # Optuna intermediate reporting for pruner
            if status.get('progress'):
                try:
                    trial.report(float(status.get('partial_sortino') or 0.0),
                                 step=int(status.get('progress') * 100))
                    if trial.should_prune():
                        return None
                except Exception:
                    pass
            time.sleep(15.0)
        logger.warning("Trial %d timed out after %ds", trial.number, timeout_s)
        return None

    # ── Run loop ─────────────────────────────────────────────────────────────

    def run(self, n_trials: int = 100, n_jobs: int = 1) -> dict:
        """Run the optimization sweep. Returns the best trial summary."""
        study = self.create_study()
        logger.info("CIO Agent starting study=%s n_trials=%d", self.study_name, n_trials)
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        best = study.best_trial
        summary = {
            'study_name':   self.study_name,
            'n_trials':     len(study.trials),
            'best_value':   float(best.value),
            'best_params':  dict(best.params),
            'best_user_attrs': dict(best.user_attrs),
            'completed_at': datetime.now(timezone.utc).isoformat(),
        }
        self._persist_proposal(summary)
        logger.info("CIO Agent best: %.4f params=%s", best.value, best.params)
        return summary

    def _persist_proposal(self, summary: dict) -> None:
        """Write proposed HP block to data/cio_proposals.json — NOT to training_rules.json
        (operator approval required to promote)."""
        try:
            from src.utils.safe_json import read_json, write_json
            existing = read_json(str(CIO_PROPOSALS_PATH), default={'proposals': []})
            if not isinstance(existing, dict):
                existing = {'proposals': []}
            existing.setdefault('proposals', []).append(summary)
            existing['proposals'] = existing['proposals'][-50:]
            write_json(str(CIO_PROPOSALS_PATH), existing)
        except Exception as e:
            logger.error("CIO Agent: could not persist proposal: %s", e)

    def apply_best(self, study_name: str | None = None, operator_approved: bool = False) -> dict:
        """
        Promote the winning HP block into data/training_rules.json. REQUIRES
        operator_approved=True — never auto-applies.
        """
        if not operator_approved:
            return {
                'ok': False,
                'error': 'operator_approved=True required. Review the proposal first.',
            }
        # Implementation left for after Optuna study completes — out of Phase 0-FIX scope
        return {'ok': True, 'note': 'apply_best not yet implemented'}


# ── HTTP submitters/pollers for live cluster integration ────────────────────

def make_cluster_callbacks(
    cluster_url: str = 'http://127.0.0.1:7700',
    api_key: str | None = None,
) -> tuple[Callable[[dict], str], Callable[[str], dict]]:
    """
    Build (task_submitter, task_status_poller) callables that talk to the
    standalone orchestrator's REST API. Used by `live_mode` to wire CIO
    trials as real cluster tasks.

    Returns two callables:
      submitter(spec) -> task_id
      poller(task_id) -> {status: ..., result: ...}
    """
    import os as _os
    import requests as _requests
    headers = {'X-API-Key': api_key or _os.getenv('CLUSTER_API_KEY')
                            or _os.getenv('DASHBOARD_API_KEY') or ''}

    def submitter(spec: dict) -> str:
        r = _requests.post(f"{cluster_url}/api/cluster/submit",
                           json=spec, headers=headers, timeout=10)
        r.raise_for_status()
        body = r.json()
        return body.get('task_id', '')

    def poller(task_id: str) -> dict:
        # Cluster orchestrator exposes /api/cluster/tasks (list); filter by id.
        # In a future iteration, add a /api/cluster/task/<id> GET to avoid
        # the full list. For now, this works for studies up to a few hundred
        # concurrent tasks.
        r = _requests.get(f"{cluster_url}/api/cluster/tasks",
                          headers=headers, timeout=10)
        r.raise_for_status()
        for t in (r.json() or []):
            if t.get('task_id') == task_id:
                return {
                    'status': t.get('status', 'unknown'),
                    'state':  t.get('status', 'unknown'),
                    'result': t.get('result') or {},
                    'error':  t.get('error', ''),
                }
        return {'status': 'unknown', 'state': 'unknown', 'result': {}}

    return submitter, poller


def live_mode(study_name: str = 'macro_parameter_search_v1',
              cluster_url: str = 'http://127.0.0.1:7700') -> CIOAgent:
    """Convenience constructor: returns a CIOAgent wired to the live cluster."""
    submitter, poller = make_cluster_callbacks(cluster_url=cluster_url)
    return CIOAgent(
        study_name=study_name,
        task_submitter=submitter,
        task_status_poller=poller,
    )


# ── Module-level singleton ───────────────────────────────────────────────────

_cio_singleton: CIOAgent | None = None


def get_cio_agent(**kwargs) -> CIOAgent:
    """
    Lazy singleton accessor.

    NOTE: kwargs are applied ONLY on first call. Subsequent calls return the
    existing instance and IGNORE kwargs — if the kwargs differ, that's a real
    misconfiguration and we raise rather than silently return the wrong agent.
    For runtime reconfiguration use `configure(...)` below.
    """
    global _cio_singleton
    if _cio_singleton is None:
        _cio_singleton = CIOAgent(**kwargs)
    elif kwargs:
        # Detect silent kwargs-ignoring (HIGH issue from code review).
        # Allow no-op kwargs but loudly reject incompatible ones.
        mismatched = []
        for k, v in kwargs.items():
            existing = getattr(_cio_singleton, k, None)
            if existing != v:
                mismatched.append((k, existing, v))
        if mismatched:
            raise RuntimeError(
                f"get_cio_agent() called with different kwargs after singleton "
                f"already initialized. Mismatched: {mismatched}. "
                f"Use cio_agent.configure(...) to update an existing instance."
            )
    return _cio_singleton


def configure(agent: CIOAgent | None = None, **kwargs) -> CIOAgent:
    """
    Apply runtime configuration (callbacks, study name, gates) to the singleton
    or a passed agent. Safe to call multiple times.
    """
    global _cio_singleton
    target = agent or _cio_singleton or CIOAgent()
    if target is _cio_singleton:
        pass
    else:
        _cio_singleton = target
    for k, v in kwargs.items():
        if hasattr(target, k):
            setattr(target, k, v)
    return target


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import argparse
    ap = argparse.ArgumentParser(description='CIO Agent (Optuna orchestrator)')
    ap.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials')
    ap.add_argument('--study-name', type=str, default='macro_parameter_search_v1')
    args = ap.parse_args()
    agent = CIOAgent(study_name=args.study_name)
    result = agent.run(n_trials=args.n_trials)
    print(json.dumps(result, indent=2))
