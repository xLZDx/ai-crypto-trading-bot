"""
cold_cache.py — persist the dashboard's slow-to-rebuild caches to disk
so a process restart serves cached data immediately instead of paying
the full recompute cost on the first request.

v3.1 step 15 (5A). Aligned with the operator memory
`feedback_disk_over_ram` — use D: drive Parquet/JSON for warm-start
state, no in-memory-only caches that disappear on restart.

What's persisted (all under data/cache/cold/):
  - typical_durations.json — _TYPICAL_DURATIONS rolling-avg map
                            (drives ETA on training rows; was
                            seeded with hand-measured times pre-PR-46
                            and self-corrected as jobs finished, so
                            losing it on restart undoes 1-2 weeks of
                            self-tuning)
  - db_status.json         — last successful /api/db/status payload
  - data_coverage.json     — last successful /api/data/coverage payload
  - monitor_services.json  — last QuestDB / training cluster scan

Each cache file carries a `_saved_at` epoch so the loader can decide
whether to use the cached value as a placeholder (instant boot) or
ignore it as too stale (e.g. >24h old).

Disk over RAM: caches are tiny JSON blobs (<50 KB total). No Parquet
needed; atomic writes via tmp+rename.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLD_DIR     = PROJECT_ROOT / 'data' / 'cache' / 'cold'

# Soft TTL — return cached value immediately on boot (so first hit is
# fast) but mark it as "ageing"; a background refresh thread should
# overwrite within seconds. After this many seconds, the cached value
# is dropped on read.
DEFAULT_MAX_AGE_S = 24 * 3600


def _path(name: str) -> Path:
    return COLD_DIR / f'{name}.json'


def save(name: str, value: Any) -> None:
    """Atomically persist `value` to data/cache/cold/<name>.json.

    Wraps the value in a small envelope with `_saved_at` so the loader
    can decide if it's still fresh enough. Errors are swallowed —
    cold-cache writes are best-effort, never user-blocking.
    """
    try:
        COLD_DIR.mkdir(parents=True, exist_ok=True)
        envelope = {'_saved_at': time.time(), 'value': value}
        target = _path(name)
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(envelope, default=str), encoding='utf-8')
        os.replace(tmp, target)
    except Exception:
        # Don't propagate — cold-cache misses are recoverable.
        pass


def load(name: str, default: Any = None, *,
         max_age_s: float = DEFAULT_MAX_AGE_S) -> Any:
    """Read and return the cached value, or `default` if missing /
    too old / unreadable."""
    p = _path(name)
    if not p.exists():
        return default
    try:
        env = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return default
    if not isinstance(env, dict) or '_saved_at' not in env:
        return default
    if (time.time() - float(env.get('_saved_at') or 0)) > max_age_s:
        return default
    return env.get('value', default)


def age_seconds(name: str) -> float | None:
    """Return seconds since the named cache was last saved, or None."""
    p = _path(name)
    if not p.exists():
        return None
    try:
        env = json.loads(p.read_text(encoding='utf-8'))
        return time.time() - float(env.get('_saved_at') or 0)
    except Exception:
        return None


def list_keys() -> list[str]:
    """All persisted cache key names (without .json suffix)."""
    if not COLD_DIR.exists():
        return []
    return sorted(p.stem for p in COLD_DIR.glob('*.json'))


__all__ = ['save', 'load', 'age_seconds', 'list_keys', 'COLD_DIR']
