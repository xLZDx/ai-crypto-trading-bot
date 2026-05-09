"""training_rules — single source of truth for which (model × TF) combos
make sense to train, where they run (CPU/GPU/exclusive), what params to
use, and how long they take.

Reads `data/training_rules.json` at module load. Cached for the process
lifetime; reload via `reload()` if the JSON is edited and the running
process needs to pick up changes.

Public surface:
    rules.applicable_tfs(model)        → list[str]   ("✅" cells)
    rules.experimental_tfs(model)      → list[str]   ("⚠" cells)
    rules.skip_tfs(model)              → list[str]   ("❌" cells)
    rules.cell_status(model, tf)       → 'applicable' | 'experimental' | 'skip' | 'unknown'
    rules.skip_reason(model, tf=None)  → str
    rules.params(model)                → dict
    rules.resource_kind(model)         → 'cpu' | 'gpu' | 'exclusive'
    rules.est_minutes(model)           → int
    rules.symbols()                    → list[str]
    rules.skip_if_fresh_s()            → int
    rules.estimated_total_minutes(plan)
                                       → int — wall-clock estimate for
                                         a list of (model, tf) tuples
                                         (sum, no parallelism factor)
    rules.estimated_parallel_minutes(plan, n_workers)
                                       → int — same plan, divided across
                                         workers respecting resource_kind

Operator overrides:
    rules.matrix(force_train=None, force_skip=None) → list[(model, tf, status)]
        — same applicability matrix BUT with operator-supplied force lists
          honored. Used by SweepCoordinator to build the per-sweep task list.

v4 Phase B0 — built 2026-05-09 alongside the hybrid pivot.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH   = PROJECT_ROOT / 'data' / 'training_rules.json'

_lock = threading.Lock()
_cache: dict | None = None


# Canonical TF order for matrix rendering — used by the dashboard rules
# editor card so columns are always in time-ascending order.
TF_ORDER = ('1m', '5m', '15m', '1h', '4h', '1d', '1w', '1mo')


def _load() -> dict:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        if not RULES_PATH.exists():
            raise FileNotFoundError(f'training_rules.json not found at {RULES_PATH}')
        with open(RULES_PATH, 'r', encoding='utf-8') as f:
            _cache = json.load(f)
        return _cache


def reload() -> dict:
    """Drop the cache and re-read from disk. Call after editing the JSON
    while a long-lived process is running."""
    global _cache
    with _lock:
        _cache = None
    return _load()


def all_models() -> list[str]:
    return sorted(_load()['models'].keys())


def _model_block(model: str) -> dict:
    blocks = _load()['models']
    if model not in blocks:
        raise KeyError(f'unknown model {model!r}; valid: {sorted(blocks.keys())}')
    return blocks[model]


def applicable_tfs(model: str) -> list[str]:
    return list(_model_block(model).get('applicable_tfs', []))


def experimental_tfs(model: str) -> list[str]:
    return list(_model_block(model).get('experimental_tfs', []))


def skip_tfs(model: str) -> list[str]:
    return list(_model_block(model).get('skip_tfs', []))


def cell_status(model: str, tf: str) -> str:
    blk = _model_block(model)
    if tf in blk.get('applicable_tfs', []):
        return 'applicable'
    if tf in blk.get('experimental_tfs', []):
        return 'experimental'
    if tf in blk.get('skip_tfs', []):
        return 'skip'
    return 'unknown'


def should_train(model: str, tf: str, *, include_experimental: bool = False) -> bool:
    s = cell_status(model, tf)
    if s == 'applicable':
        return True
    if s == 'experimental' and include_experimental:
        return True
    return False


def skip_reason(model: str, tf: str | None = None) -> str:
    """Why this combo is skipped. tf is optional — when given, returns the
    model-level reason if the cell is in skip_tfs (the JSON has a
    per-model skip_reason that covers the whole skip list)."""
    blk = _model_block(model)
    if tf is None or tf in blk.get('skip_tfs', []):
        return blk.get('skip_reason', '')
    if tf in blk.get('experimental_tfs', []):
        return f"experimental — only run when include_experimental=True"
    if tf in blk.get('applicable_tfs', []):
        return ''
    return f"unknown TF {tf!r} for model {model!r}"


def params(model: str) -> dict:
    return dict(_model_block(model).get('params', {}))


def resource_kind(model: str) -> str:
    return _model_block(model).get('resource_kind', 'cpu')


def est_minutes(model: str) -> int:
    return int(_model_block(model).get('est_minutes_per_run', 30))


def symbols() -> list[str]:
    return list(_load().get('global', {}).get('default_symbol_universe', []))


def skip_if_fresh_s() -> int:
    return int(_load().get('global', {}).get('skip_if_fresh_s', 48 * 3600))


def matrix(force_train: list | None = None,
           force_skip: list | None = None,
           include_experimental: bool = False) -> list[tuple[str, str, str]]:
    """Return a flat list of (model, tf, status) triples covering every
    cell in the model × TF matrix. force_train / force_skip are lists of
    (model, tf) tuples — operator overrides for one-off sweeps.

    status ∈ {'applicable','experimental','skip','force_train','force_skip','unknown'}
    """
    ft = {tuple(t) for t in (force_train or [])}
    fs = {tuple(t) for t in (force_skip or [])}
    rows: list[tuple[str, str, str]] = []
    for model in all_models():
        # Iterate the canonical TF set so unknown TFs surface as 'unknown'
        # in the UI matrix.
        for tf in TF_ORDER:
            if (model, tf) in fs:
                rows.append((model, tf, 'force_skip'))
                continue
            if (model, tf) in ft:
                rows.append((model, tf, 'force_train'))
                continue
            rows.append((model, tf, cell_status(model, tf)))
    return rows


def planned_combos(force_train: list | None = None,
                   force_skip: list | None = None,
                   include_experimental: bool = False) -> list[tuple[str, str]]:
    """The (model, tf) tuples that WILL be trained in a sweep, after
    applying operator overrides + experimental flag. Skip-if-fresh is
    NOT applied here — this is the full intended plan, the per-meta
    age check happens at submission time."""
    out: list[tuple[str, str]] = []
    for model, tf, status in matrix(force_train, force_skip, include_experimental):
        if status == 'applicable' or status == 'force_train':
            out.append((model, tf))
        elif status == 'experimental' and include_experimental:
            out.append((model, tf))
    return out


def estimated_total_minutes(plan: list[tuple[str, str]]) -> int:
    """Naive sum (no parallelism). Used as upper-bound ETA."""
    return sum(est_minutes(m) for m, _ in plan)


def estimated_parallel_minutes(plan: list[tuple[str, str]],
                               n_workers: int = 2) -> int:
    """Estimate wall-clock when split across n_workers, respecting
    resource_kind. The 'exclusive' kind takes the whole fleet; 'gpu'
    needs ≥1 GPU node (we assume n_workers/2 are GPU-capable, which
    matches today's master+1 worker setup since both have GPUs); 'cpu'
    is fully parallelizable.

    This is a rough estimator — used by the rules editor card to show
    'this configuration finishes in ~X hours' so operators can size
    their sweep before triggering."""
    n = max(1, int(n_workers))
    cpu_total = 0
    gpu_total = 0
    exclusive_total = 0
    for m, _ in plan:
        kind = resource_kind(m)
        mins = est_minutes(m)
        if kind == 'cpu':
            cpu_total += mins
        elif kind == 'gpu':
            gpu_total += mins
        elif kind == 'exclusive':
            exclusive_total += mins
        else:
            cpu_total += mins
    # cpu fully parallelizes; gpu fully parallelizes (we assume both
    # workers have GPUs); exclusive serializes
    return int(cpu_total / n + gpu_total / n + exclusive_total)


__all__ = [
    'TF_ORDER',
    'all_models',
    'applicable_tfs', 'experimental_tfs', 'skip_tfs',
    'cell_status', 'should_train', 'skip_reason',
    'params', 'resource_kind', 'est_minutes',
    'symbols', 'skip_if_fresh_s',
    'matrix', 'planned_combos',
    'estimated_total_minutes', 'estimated_parallel_minutes',
    'reload',
]
