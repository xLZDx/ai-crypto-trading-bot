"""
KPI Gate — Sprint 1A R2.

Reads the last 3 successful training runs for each (model, tf) cell and auto-
retires any cell whose runs ALL miss the configured `kpi_threshold` in
`data/training_rules.json`. Retired cells are marked `kpi_retired=true` in
`data/kpi_retired_registry.json` and skipped by the cluster orchestrator
on subsequent sweeps until the operator restores them via the
`/api/registry/<key>/restore` endpoint.

Design choices:
  - "Successful run" = TrainingResult.error is None AND artifact_path is set
  - Threshold check is OR over fields: a single missed field is enough to fail
    that run. A cell retires only when ALL 3 most-recent successful runs fail.
  - One Parquet file per (model, tf) at `data/training_runs/<model>__<tf>.parquet`
  - Append-only; the gate reads the last 3 by `finished_at`
  - This module is invoked by:
       * Orchestrator post-training gate (after `evaluate_trained_model`)
       * Cluster dispatch (pre-flight check to skip retired cells)

Integration with ML Engineer Agent:
  - ML Engineer enforces AFML methodology (BLOCK on bad config / KPI)
  - KPI Gate enforces empirical retirement across runs (3-strike rule)
  - Both run; they are complementary
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RULES_PATH = PROJECT_ROOT / 'data' / 'training_rules.json'
TRAINING_RUNS_DIR   = PROJECT_ROOT / 'data' / 'training_runs'
RETIRED_REGISTRY_PATH = PROJECT_ROOT / 'data' / 'kpi_retired_registry.json'

# Number of consecutive misses required for auto-retirement.
CONSECUTIVE_MISS_LIMIT = 3


# ── TrainingResult — single source of truth for what every trainer emits ────

@dataclass
class TrainingResult:
    """KPI blob emitted by every trainer. Same fields across all model types
    so cross-model comparison in the dashboard is apples-to-apples."""
    model_key: str
    tf: str
    started_at: float
    finished_at: float
    artifact_path: str | None = None  # None on failure
    n_samples: int = 0
    n_features: int = 0
    # ── KPI block ──
    wf_sharpe:       float | None = None
    wf_calmar:       float | None = None
    wf_max_dd:       float | None = None
    wf_win_rate:     float | None = None
    wf_expectancy:   float | None = None
    wf_total_trades: int   | None = None
    wf_acc:          float | None = None
    auc_roc:         float | None = None
    # ── Failure modes ──
    error:     str | None = None
    cancelled: bool = False

    @property
    def successful(self) -> bool:
        return self.error is None and self.artifact_path is not None and not self.cancelled


# ── Persistence ─────────────────────────────────────────────────────────────

def runs_file_for(model: str, tf: str) -> Path:
    """Path to the per-cell Parquet file (created on first append)."""
    safe = f"{model}__{tf}"
    return TRAINING_RUNS_DIR / f"{safe}.parquet"


def append_run(result: TrainingResult) -> None:
    """Append a TrainingResult to the per-cell Parquet file. Creates the
    file on first call. Thread-safe via append-then-rewrite (acceptable for
    the run cadence; we're not appending millions of rows)."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required for KPI gate append_run")
        return

    TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = runs_file_for(result.model_key, result.tf)
    row = pd.DataFrame([asdict(result)])
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, row], ignore_index=True)
        except Exception as e:
            logger.warning("KPI gate: failed to read existing runs at %s: %s — overwriting", path, e)
            combined = row
    else:
        combined = row
    try:
        combined.to_parquet(path, index=False)
    except Exception as e:
        logger.error("KPI gate: failed to write %s: %s", path, e, exc_info=True)


def last_n_successful(model: str, tf: str, n: int = CONSECUTIVE_MISS_LIMIT) -> list[TrainingResult]:
    """Return the last `n` successful TrainingResults for (model, tf), most
    recent first. Empty list if no successful runs yet."""
    try:
        import pandas as pd
    except ImportError:
        return []
    path = runs_file_for(model, tf)
    if not path.exists():
        return []
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("KPI gate: cannot read %s: %s", path, e)
        return []
    # Filter to successful and sort by finished_at desc
    df = df[df['error'].isna() & df['artifact_path'].notna() & ~df['cancelled'].fillna(False).astype(bool)]
    df = df.sort_values('finished_at', ascending=False).head(n)
    out: list[TrainingResult] = []
    for _, r in df.iterrows():
        d = r.to_dict()
        # Restore None for NaN/NaT
        for k, v in list(d.items()):
            try:
                import math
                if isinstance(v, float) and math.isnan(v):
                    d[k] = None
            except Exception:
                pass
        try:
            out.append(TrainingResult(**{k: v for k, v in d.items() if k in TrainingResult.__dataclass_fields__}))
        except Exception:
            continue
    return out


# ── Rules + retirement registry ─────────────────────────────────────────────

def _load_rules() -> dict:
    """Return the training_rules.json dict. Returns {} on read failure."""
    try:
        with open(TRAINING_RULES_PATH, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning("KPI gate: cannot read training_rules.json: %s", e)
        return {}


def thresholds_for(model: str) -> dict[str, float] | None:
    """Return the `kpi_threshold` dict for a model, or None if not configured."""
    rules = _load_rules()
    cell = (rules.get('models', {}) or {}).get(model, {})
    thr = cell.get('kpi_threshold')
    if not isinstance(thr, dict) or not thr:
        return None
    return thr


def _load_retired() -> dict:
    if not RETIRED_REGISTRY_PATH.exists():
        return {'retired': {}, 'last_updated': None}
    try:
        with open(RETIRED_REGISTRY_PATH, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        if not isinstance(d, dict):
            return {'retired': {}, 'last_updated': None}
        d.setdefault('retired', {})
        return d
    except Exception:
        return {'retired': {}, 'last_updated': None}


def _save_retired(data: dict) -> None:
    try:
        from src.utils.safe_json import write_json
        data['last_updated'] = datetime.now(timezone.utc).isoformat()
        write_json(str(RETIRED_REGISTRY_PATH), data)
    except Exception as e:
        logger.error("KPI gate: cannot save retired registry: %s", e)


def is_retired(model: str, tf: str) -> bool:
    """Has (model, tf) been auto-retired? Used by the orchestrator's pre-flight
    skip check."""
    data = _load_retired()
    return data.get('retired', {}).get(f"{model}__{tf}", {}).get('retired', False)


def restore(model: str, tf: str) -> dict:
    """Clear the retired flag for (model, tf). Called by the
    `/api/registry/<key>/restore` endpoint."""
    data = _load_retired()
    key = f"{model}__{tf}"
    if key in data.get('retired', {}):
        del data['retired'][key]
        _save_retired(data)
        logger.info("KPI gate: restored %s", key)
        return {'ok': True, 'restored': key}
    return {'ok': False, 'reason': f'{key} was not retired'}


# ── The gate ────────────────────────────────────────────────────────────────

def evaluate_run(result: TrainingResult) -> dict:
    """
    Persist the run, then check if 3 consecutive successful runs all missed
    thresholds → auto-retire. Returns:

        {
          'persisted': True,
          'retired_now': bool,
          'last_n_failures': int,
          'missed_fields': [...],   # fields that ALL last-N runs missed
          'reasons': [...],
        }
    """
    append_run(result)

    if not result.successful:
        return {'persisted': True, 'retired_now': False, 'last_n_failures': 0,
                'missed_fields': [], 'reasons': ['run failed; not evaluated']}

    thresholds = thresholds_for(result.model_key)
    if not thresholds:
        return {'persisted': True, 'retired_now': False, 'last_n_failures': 0,
                'missed_fields': [], 'reasons': [f'no kpi_threshold configured for {result.model_key}']}

    recent = last_n_successful(result.model_key, result.tf, n=CONSECUTIVE_MISS_LIMIT)
    if len(recent) < CONSECUTIVE_MISS_LIMIT:
        return {'persisted': True, 'retired_now': False,
                'last_n_failures': 0, 'missed_fields': [],
                'reasons': [f'only {len(recent)}/{CONSECUTIVE_MISS_LIMIT} runs available']}

    # Each run fails if ANY threshold field is below its floor.
    # Cell retires if ALL recent runs fail (consecutive_misses == LIMIT).
    failures: list[tuple[TrainingResult, list[str]]] = []
    for run in recent:
        missed = _check_thresholds(run, thresholds)
        if missed:
            failures.append((run, missed))

    if len(failures) < CONSECUTIVE_MISS_LIMIT:
        # At least one of the recent runs passed → reset the strike counter
        return {'persisted': True, 'retired_now': False,
                'last_n_failures': len(failures),
                'missed_fields': [],
                'reasons': [f'{len(failures)}/{CONSECUTIVE_MISS_LIMIT} recent runs failed thresholds — not yet retired']}

    # All N runs failed. Compute the union of missed fields for the audit trail.
    all_missed = sorted({m for _, miss in failures for m in miss})
    data = _load_retired()
    key = f"{result.model_key}__{result.tf}"
    data['retired'][key] = {
        'retired': True,
        'retired_at': datetime.now(timezone.utc).isoformat(),
        'reason': f'{CONSECUTIVE_MISS_LIMIT} consecutive runs missed thresholds',
        'missed_fields': all_missed,
        'thresholds':    thresholds,
    }
    _save_retired(data)
    logger.error(
        "[KPI gate] AUTO-RETIRED %s — %d consecutive misses on %s",
        key, CONSECUTIVE_MISS_LIMIT, all_missed,
    )
    return {'persisted': True, 'retired_now': True,
            'last_n_failures': len(failures),
            'missed_fields': all_missed,
            'reasons': [f'auto-retired after {CONSECUTIVE_MISS_LIMIT} consecutive threshold misses']}


def _check_thresholds(run: TrainingResult, thresholds: dict[str, float]) -> list[str]:
    """Return list of field names where `run` failed the threshold.

    Special cases:
      - `wf_max_dd` is a MAX threshold (run must be <=, not >=).
      - All other fields are MIN thresholds (run must be >=).
    """
    missed: list[str] = []
    for field_name, floor in thresholds.items():
        actual = getattr(run, field_name, None)
        if actual is None:
            missed.append(f"{field_name}:missing")
            continue
        try:
            actual_f = float(actual)
            floor_f  = float(floor)
        except (TypeError, ValueError):
            missed.append(f"{field_name}:not_numeric")
            continue
        if field_name == 'wf_max_dd':
            # Drawdown is a MAX (smaller is better)
            if actual_f > floor_f:
                missed.append(f"{field_name}:{actual_f:.3f}>{floor_f:.3f}")
        else:
            if actual_f < floor_f:
                missed.append(f"{field_name}:{actual_f:.3f}<{floor_f:.3f}")
    return missed


# ── Convenience for orchestrator integration ────────────────────────────────

def evaluate_from_meta_json(model_key: str, tf: str, meta_json_path: str) -> dict:
    """
    Build a TrainingResult from a model's meta JSON and run evaluate_run.
    Called from the orchestrator's update_task("done") handler.
    """
    try:
        with open(meta_json_path, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)
    except Exception as e:
        logger.error("KPI gate: cannot read meta JSON %s: %s", meta_json_path, e)
        return {'persisted': False, 'retired_now': False,
                'reasons': [f'meta JSON unreadable: {e}']}

    result = TrainingResult(
        model_key=model_key,
        tf=tf,
        started_at=meta.get('started_at', 0.0) or 0.0,
        finished_at=meta.get('finished_at_ts', 0.0)
                    or (datetime.now(timezone.utc).timestamp()),
        artifact_path=meta.get('artifact_path') or meta_json_path,
        n_samples=int(meta.get('n_samples', 0) or 0),
        n_features=int(meta.get('n_features', 0) or 0),
        wf_sharpe=meta.get('wf_sharpe') or meta.get('walk_forward_sharpe'),
        wf_calmar=meta.get('wf_calmar') or meta.get('walk_forward_calmar'),
        wf_max_dd=meta.get('wf_max_dd') or meta.get('walk_forward_max_dd'),
        wf_win_rate=meta.get('wf_win_rate') or meta.get('walk_forward_win_rate') or meta.get('win_rate_pct'),
        wf_expectancy=meta.get('wf_expectancy') or meta.get('walk_forward_expectancy'),
        wf_total_trades=meta.get('wf_total_trades') or meta.get('walk_forward_total_trades'),
        wf_acc=meta.get('wf_acc') or meta.get('walk_forward_mean_acc'),
        auc_roc=meta.get('auc_roc'),
    )
    return evaluate_run(result)
