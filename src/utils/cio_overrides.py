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
