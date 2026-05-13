"""
Shared helper for reading operator-approved CIO Agent overrides.

Trainers call `load_cio_overrides(model_key)` to fetch a dict of HP overrides
that CIOAgent.apply_best wrote to data/training_rules.json. Audit metadata
fields (`_applied_at`, `_study`, `_best_value`) are stripped before return.

Returns {} for any error path — graceful degradation: trainers always fall
back to their hardcoded defaults if overrides can't be read.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RULES_PATH = PROJECT_ROOT / 'data' / 'training_rules.json'


def load_cio_overrides(model_key: str) -> dict:
    """Return the cio_overrides dict for `model_key` with metadata stripped.

    Empty dict on:
      - missing rules file
      - missing models block
      - missing model_key
      - no cio_overrides for that model
      - malformed JSON

    The trainer is expected to use this as a fallback chain:
        kwargs (CLI / cluster spec)  →  cio_overrides  →  hardcoded defaults
    """
    try:
        with open(TRAINING_RULES_PATH, 'r', encoding='utf-8') as fh:
            rules = json.load(fh)
        overrides = (rules.get('models') or {}).get(model_key, {}).get('cio_overrides') or {}
        return {k: v for k, v in overrides.items() if not str(k).startswith('_')}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("[CIO overrides] could not read %s overrides: %s", model_key, e)
        return {}


def merge_with_defaults(
    model_key: str,
    defaults: dict,
    schema: dict,
) -> tuple[dict, dict]:
    """Schema-bounded MERGE of CIO overrides on top of hardcoded defaults.

    Used by trend / futures / scalping trainers (and any future trainer)
    so the operator's CIO Optuna proposals actually reach the model's
    hyperparameters. Pre-2026-05-13 the merge was implemented only in
    train_model.py:_load_model_params (base) — the other three trainers
    audit-logged the overrides but didn't apply them, defeating the
    learning loop.

    Args:
        model_key:  e.g. 'trend', 'futures', 'scalping'
        defaults:   {hp_name: default_value}
        schema:     {hp_name: (expected_type, lo, hi)}
                    lo=hi=None means no range check (e.g. string params).

    Returns:
        (merged_params, applied_overrides_dict)
        merged_params      — defaults with valid CIO overrides applied.
        applied_overrides_dict — subset of CIO overrides that survived
                                 schema validation; useful for persisting
                                 to the model meta JSON as
                                 cio_overrides_applied=... for audit.

    Overrides that fail type or range checks are skipped with a WARN. The
    fallback chain matches train_model.py:_load_model_params semantics
    exactly, so trainer behavior is consistent across all 4 model paths.
    """
    overrides = load_cio_overrides(model_key)
    if not overrides:
        return dict(defaults), {}

    merged = dict(defaults)
    applied: dict = {}
    for key, (expected_type, lo, hi) in schema.items():
        if key not in overrides:
            continue
        val = overrides[key]
        if not isinstance(val, expected_type):
            logger.warning(
                "[CIO override %s] '%s' expected %s got %s — skipping",
                model_key, key, expected_type.__name__, type(val).__name__,
            )
            continue
        if lo is not None and not (lo <= val <= hi):
            logger.warning(
                "[CIO override %s] '%s'=%s out of range [%s, %s] — skipping",
                model_key, key, val, lo, hi,
            )
            continue
        merged[key] = val
        applied[key] = val

    if applied:
        logger.info("[CIO override %s] merged into params: %s", model_key, applied)
    return merged, applied
