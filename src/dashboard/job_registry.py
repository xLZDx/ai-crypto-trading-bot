"""Dashboard-wide job registry — single source of truth for long-running
button actions across every card/tab.

Why this exists
---------------
Pre-existing job tracking was fragmented across `_training_jobs`,
`_resample_jobs`, and `_cio_study_state` inside `app.py`, plus client-side
`_saveMCState` localStorage mirrors. A user running orchestration on the
Model Comparison tab and switching to another tab would lose visible
progress because the only persisted state was the rendered HTML in
localStorage — there was no way to *poll* and see if the work was still
in flight, or to rehydrate the UI on a fresh page load.

This module:
- Persists every dashboard-launched job (frontend OR backend-initiated)
  to `data/dashboard_jobs.json` via `safe_json` (atomic + filelock).
- Indexes jobs by `card_id` so each card on the dashboard can ask
  "what's running on me right now?" without coupling to other cards.
- Caps history at 50 jobs per card so the file stays small.
- Exposes `register`, `update`, `complete`, `fail`, `append_log`,
  `list_for_card`, `get` for the Flask handler to call.

Schema
------
{
  "schema_version": 1,
  "jobs": [
    {
      "job_id": "job_<ts>_<hex>",
      "card_id": "auto-orch",                # logical card key
      "label": "Auto-orchestration",         # human-readable
      "kind": "orchestration",               # category for grouping
      "status": "running",                   # queued | running | done | error | cancelled
      "created_at": 1747000000.0,
      "started_at": 1747000001.0,
      "finished_at": null,
      "progress_pct": null,                  # 0-100 if known
      "log": [
        {"ts": 1747000001.5, "msg": "Running bake-off..."},
        ...
      ],
      "result": null,                        # arbitrary dict when done
      "error": null                          # error string when error
    }, ...
  ]
}
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from src.utils.safe_json import read_json, write_json

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = _PROJECT_ROOT / "data" / "dashboard_jobs.json"

# Per-card cap. Keeps history bounded so the JSON stays small (~1 MB max
# even with hundreds of cards). When exceeded, oldest terminal jobs are
# pruned first; running jobs are never pruned.
MAX_JOBS_PER_CARD = 50

# Lock guards the in-process critical section. The filelock inside
# safe_json handles cross-process atomicity. Both are needed: Flask runs
# multi-threaded by default.
_lock = RLock()

_TERMINAL = {"done", "error", "cancelled"}


def _empty_state() -> dict:
    return {"schema_version": 1, "jobs": []}


def _load() -> dict:
    state = read_json(str(REGISTRY_PATH), default=_empty_state())
    if not isinstance(state, dict) or "jobs" not in state:
        return _empty_state()
    return state


def _save(state: dict) -> None:
    write_json(str(REGISTRY_PATH), state)


def _gen_job_id() -> str:
    return f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _prune(state: dict, card_id: str) -> None:
    """Drop oldest terminal jobs for `card_id` beyond MAX_JOBS_PER_CARD."""
    card_jobs = [j for j in state["jobs"] if j.get("card_id") == card_id]
    if len(card_jobs) <= MAX_JOBS_PER_CARD:
        return
    terminal = [j for j in card_jobs if j.get("status") in _TERMINAL]
    # Sort oldest-first so the oldest get dropped.
    terminal.sort(key=lambda j: j.get("finished_at") or j.get("created_at") or 0)
    to_drop = len(card_jobs) - MAX_JOBS_PER_CARD
    drop_ids = {j["job_id"] for j in terminal[:to_drop]}
    if drop_ids:
        state["jobs"] = [j for j in state["jobs"] if j["job_id"] not in drop_ids]


def register(
    card_id: str,
    *,
    label: str,
    kind: str = "generic",
    initial_log: str | None = None,
) -> str:
    """Create a new job for `card_id`. Returns the new job_id."""
    if not card_id:
        raise ValueError("card_id is required")
    with _lock:
        state = _load()
        now = time.time()
        job_id = _gen_job_id()
        log_entries = []
        if initial_log:
            log_entries.append({"ts": now, "msg": initial_log})
        state["jobs"].append({
            "job_id": job_id,
            "card_id": card_id,
            "label": label,
            "kind": kind,
            "status": "running",
            "created_at": now,
            "started_at": now,
            "finished_at": None,
            "progress_pct": None,
            "log": log_entries,
            "result": None,
            "error": None,
        })
        _prune(state, card_id)
        _save(state)
    logger.info("job_registry: registered %s for card=%s (%s)",
                job_id, card_id, label)
    return job_id


def _find(state: dict, job_id: str) -> dict | None:
    for j in state["jobs"]:
        if j.get("job_id") == job_id:
            return j
    return None


def append_log(job_id: str, msg: str) -> bool:
    """Append a progress line. Returns True if found."""
    with _lock:
        state = _load()
        job = _find(state, job_id)
        if not job:
            return False
        log = job.setdefault("log", [])
        log.append({"ts": time.time(), "msg": str(msg)})
        # Cap log lines per job to 500 to keep the file small.
        if len(log) > 500:
            del log[: len(log) - 500]
        _save(state)
        return True


def update(
    job_id: str,
    *,
    status: str | None = None,
    progress_pct: float | None = None,
    result: Any = None,
    error: str | None = None,
    log: str | None = None,
) -> bool:
    """Generic update. Any non-None field overwrites. Returns True if found."""
    with _lock:
        state = _load()
        job = _find(state, job_id)
        if not job:
            return False
        if status is not None:
            job["status"] = status
            if status in _TERMINAL and not job.get("finished_at"):
                job["finished_at"] = time.time()
        if progress_pct is not None:
            job["progress_pct"] = float(progress_pct)
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        if log is not None:
            job.setdefault("log", []).append({"ts": time.time(), "msg": str(log)})
        _save(state)
        return True


def complete(job_id: str, *, result: Any = None, log: str | None = None) -> bool:
    return update(job_id, status="done", result=result, log=log, progress_pct=100.0)


def fail(job_id: str, *, error: str, log: str | None = None) -> bool:
    return update(job_id, status="error", error=error, log=log)


def cancel(job_id: str, *, reason: str | None = None) -> bool:
    return update(job_id, status="cancelled", log=reason)


def get(job_id: str) -> dict | None:
    """Return a single job by id, or None."""
    state = _load()
    return _find(state, job_id)


def list_for_card(
    card_id: str,
    *,
    limit: int = 20,
    include_terminal: bool = True,
) -> list[dict]:
    """Return jobs for `card_id`, newest-first. Running jobs always come
    first, then terminal jobs ordered by finished_at desc.
    """
    state = _load()
    jobs = [j for j in state["jobs"] if j.get("card_id") == card_id]
    if not include_terminal:
        jobs = [j for j in jobs if j.get("status") not in _TERMINAL]
    # Sort: running first (by created_at desc), then terminal (by finished_at desc).
    running = [j for j in jobs if j.get("status") not in _TERMINAL]
    terminal = [j for j in jobs if j.get("status") in _TERMINAL]
    running.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    terminal.sort(key=lambda j: j.get("finished_at") or j.get("created_at") or 0,
                  reverse=True)
    return (running + terminal)[:limit]


def list_running() -> list[dict]:
    """All in-flight jobs across every card."""
    state = _load()
    return [j for j in state["jobs"] if j.get("status") not in _TERMINAL]


def cleanup_stale(max_age_s: float = 7 * 24 * 3600) -> int:
    """Drop terminal jobs older than `max_age_s`. Returns count dropped.
    Caller's responsibility to schedule. Default: 7 days.
    """
    with _lock:
        state = _load()
        now = time.time()
        before = len(state["jobs"])
        state["jobs"] = [
            j for j in state["jobs"]
            if j.get("status") not in _TERMINAL or
               (now - (j.get("finished_at") or j.get("created_at") or now)) <= max_age_s
        ]
        dropped = before - len(state["jobs"])
        if dropped:
            _save(state)
    return dropped


__all__ = [
    "REGISTRY_PATH",
    "register", "update", "complete", "fail", "cancel",
    "append_log", "get", "list_for_card", "list_running", "cleanup_stale",
]
