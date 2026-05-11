"""src.engine.trainers._common — shared helpers for trainer wrappers.

Pattern: each train_<key>.py wrapper calls the existing trainer function,
then reads the freshly-written model meta JSON to populate TrainingResult.
This file centralizes that meta-reading + dataclass-building logic so
8 wrappers share ~30 LOC instead of duplicating ~150.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Callable

from .types import TrainingResult


def _to_pct(v) -> float | None:
    """Normalize accuracy to percent. Trainers vary — some save 0.486,
    others 48.6. Heuristic: any non-null value ≤ 1.0 is a fraction;
    multiply by 100. Mirrors src/dashboard/app.py:_to_pct."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f * 100.0 if 0.0 <= f <= 1.0 else f


def _read_meta(model_key: str, tf: str) -> dict:
    """Read the freshly-written model meta JSON. Returns {} if not found —
    most trainers DO write a meta file but a failed run might not.
    Resolution order matches existing dashboard code:
      1. src.utils.model_paths.artifact_paths(key, tf)['meta']
      2. Empty dict (trainer didn't persist meta — error path)
    """
    try:
        from src.utils.model_paths import artifact_paths
        p = artifact_paths(model_key, tf).get('meta')
        if p and Path(p).exists():
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _populate_from_meta(result: TrainingResult, meta: dict) -> None:
    """Fill TrainingResult KPI fields from a model meta dict. Treats missing
    fields as None — every model meta is welcome to have its own field set;
    only what's present gets mapped.
    """
    # Identity / artifact
    if 'artifact' in meta:
        result.artifact_path = str(meta['artifact'])
    # Data stats
    result.n_samples    = meta.get('n_samples')
    result.n_features   = meta.get('n_features')
    result.n_iterations = meta.get('n_iterations')
    # Accuracy block
    result.test_acc       = _to_pct(meta.get('accuracy'))
    result.wf_acc         = _to_pct(meta.get('walk_forward_mean_acc'))
    result.long_acc       = _to_pct(meta.get('long_accuracy'))
    result.short_acc      = _to_pct(meta.get('short_accuracy'))
    result.win_precision  = _to_pct(meta.get('win_precision'))
    # AUC stays raw (0-1)
    auc = meta.get('auc_roc')
    if auc is not None:
        try:
            result.auc_roc = round(float(auc), 4)
        except (TypeError, ValueError):
            pass
    # KPI block — these come from backtest results, not training meta.
    # Trainers don't compute Sharpe/Calmar themselves. Sprint 1a R2 will
    # add a post-backtest hook to populate wf_sharpe/calmar/etc from the
    # cell's BT results. For now these stay None unless the meta carries
    # them as a side-channel (some experimental trainers do).
    for key in ('wf_sharpe', 'wf_calmar', 'wf_max_dd', 'wf_win_rate',
                 'wf_expectancy', 'wf_total_trades'):
        if key in meta:
            v = meta[key]
            if v is not None:
                try:
                    setattr(result, key,
                             int(v) if key == 'wf_total_trades' else float(v))
                except (TypeError, ValueError):
                    pass


def run_trainer(model_key: str, tf: str, *, symbol: str,
                 train_fn: Callable, **kwargs) -> TrainingResult:
    """Shared body for every train_<key>.py wrapper.

    1. Build TrainingResult with started_at = now
    2. Invoke train_fn(timeframe=tf, **kwargs) inside try/except
    3. On success: read meta + populate KPI fields
    4. On failure: capture exc info to result.error
    5. Always set finished_at + elapsed_s
    """
    result = TrainingResult(
        model_key=model_key, tf=tf, symbol=symbol,
        started_at=time.time(),
    )
    try:
        train_fn(timeframe=tf, **kwargs)
    except Exception as exc:
        result.error = f'{type(exc).__name__}: {exc}'
        # Append a short traceback tail to extras — useful for drill-down,
        # bounded to ~1KB so the Parquet row stays small.
        tb = traceback.format_exc()
        result.extras['traceback_tail'] = tb[-1024:]
    finally:
        result.finished_at = time.time()
        result.elapsed_s   = round(result.finished_at - result.started_at, 2)

    if result.ok:
        meta = _read_meta(model_key, tf)
        _populate_from_meta(result, meta)

    return result
