"""Live training-progress tracker — write per-epoch state to a JSON file
the dashboard polls.

Why this exists
---------------
Operator request 2026-05-15: "On the Model Training screen add the epoch
number to the status and estimated time to show what epoch is currently
running." For TFT (and any future iterative trainer) we need to expose:

  - current_epoch / n_epochs
  - last_epoch_duration_s  → drives ETA
  - eta_s = (n_epochs - current_epoch) * last_epoch_duration_s
  - total elapsed seconds

so the dashboard can render "epoch 4/12 · 18m elapsed · ~54m remaining"
without inspecting the worker process directly.

Storage: data/training_progress.json (atomic safe_json writes).

Schema
------
{
  "tasks": {
    "<task_id>": {
      "task_id": "...",
      "model": "tft",
      "tf": "1h",
      "started_at": 1747000000.0,
      "current_epoch": 4,
      "n_epochs": 12,
      "epochs_completed": 3,             # last-COMPLETED epoch
      "last_epoch_duration_s": 5400.0,   # 90 min in this example
      "mean_epoch_duration_s": 5400.0,   # rolling mean
      "elapsed_s": 16200.0,              # since started_at
      "eta_s": 48600.0,                  # estimated remaining
      "status": "running" | "done" | "error",
      "last_update_at": 1747016200.0,
      "trainer": "train_tft_model"
    }
  },
  "schema_version": 1
}

Public surface
--------------
start(task_id, model, tf, n_epochs, ...) -> dict
epoch_done(task_id, epoch_idx, epoch_duration_s) -> None
finish(task_id, status='done'|'error') -> None
get(task_id) -> dict | None
list_active() -> list[dict]
clear_stale(max_age_s=86400) -> int

The Lightning callback in `train_tft_model._EpochProgressCallback` is the
canonical caller; tabular trainers can use start+finish for a single
"epoch 1/1" record so they appear in the same UI.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from threading import RLock
from typing import Any

from src.utils.safe_json import read_json, write_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROGRESS_PATH = PROJECT_ROOT / "data" / "training_progress.json"

_lock = RLock()
_TERMINAL = {"done", "error", "cancelled"}


def _empty_state() -> dict:
    return {"schema_version": 1, "tasks": {}}


def _load() -> dict:
    state = read_json(str(PROGRESS_PATH), default=_empty_state())
    if not isinstance(state, dict) or "tasks" not in state:
        return _empty_state()
    return state


def _save(state: dict) -> None:
    write_json(str(PROGRESS_PATH), state)


def start(
    task_id: str,
    *,
    model: str,
    tf: str,
    n_epochs: int,
    trainer: str = "",
    extra: dict | None = None,
) -> dict:
    """Register a new training task. Returns the record."""
    now = time.time()
    record = {
        "task_id": task_id,
        "model": model,
        "tf": tf,
        "trainer": trainer,
        "started_at": now,
        "current_epoch": 0,
        "n_epochs": int(n_epochs),
        "epochs_completed": 0,
        "last_epoch_duration_s": None,
        "mean_epoch_duration_s": None,
        "elapsed_s": 0.0,
        "eta_s": None,
        "status": "running",
        "last_update_at": now,
    }
    if extra:
        record["extra"] = extra
    with _lock:
        state = _load()
        state["tasks"][task_id] = record
        _save(state)
    logger.info("[training_progress] start task=%s %s@%s n_epochs=%d",
                task_id, model, tf, n_epochs)
    return record


def epoch_done(task_id: str, epoch_idx: int, epoch_duration_s: float) -> bool:
    """Record completion of an epoch (1-indexed). Returns False if task unknown."""
    with _lock:
        state = _load()
        rec = state["tasks"].get(task_id)
        if not rec:
            return False
        now = time.time()
        completed = max(int(epoch_idx), int(rec.get("epochs_completed") or 0))
        rec["epochs_completed"] = completed
        rec["current_epoch"]    = completed
        rec["last_epoch_duration_s"] = float(epoch_duration_s)
        # Rolling mean for ETA stability — equal weights so far.
        prev_mean = rec.get("mean_epoch_duration_s")
        if prev_mean is None:
            rec["mean_epoch_duration_s"] = float(epoch_duration_s)
        else:
            n = completed if completed > 0 else 1
            rec["mean_epoch_duration_s"] = (prev_mean * (n - 1) + epoch_duration_s) / n
        elapsed = now - rec["started_at"]
        rec["elapsed_s"] = elapsed
        n_remaining = max(0, int(rec["n_epochs"]) - completed)
        rec["eta_s"] = n_remaining * rec["mean_epoch_duration_s"]
        rec["last_update_at"] = now
        _save(state)
    return True


def heartbeat(task_id: str) -> bool:
    """Refresh elapsed_s without committing an epoch. Used for slow-epoch
    visibility on the dashboard."""
    with _lock:
        state = _load()
        rec = state["tasks"].get(task_id)
        if not rec:
            return False
        now = time.time()
        rec["elapsed_s"] = now - rec["started_at"]
        rec["last_update_at"] = now
        _save(state)
    return True


def finish(task_id: str, status: str = "done", error: str | None = None) -> bool:
    with _lock:
        state = _load()
        rec = state["tasks"].get(task_id)
        if not rec:
            return False
        now = time.time()
        rec["status"] = status
        rec["finished_at"] = now
        rec["elapsed_s"] = now - rec["started_at"]
        rec["eta_s"] = 0.0
        rec["last_update_at"] = now
        if error:
            rec["error"] = str(error)
        _save(state)
    logger.info("[training_progress] finish task=%s status=%s elapsed=%.1fs",
                task_id, status, rec.get("elapsed_s") or 0)
    return True


def get(task_id: str) -> dict | None:
    return _load()["tasks"].get(task_id)


def list_active() -> list[dict]:
    state = _load()
    out = [r for r in state["tasks"].values() if r.get("status") not in _TERMINAL]
    out.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return out


def list_all(limit: int = 50) -> list[dict]:
    state = _load()
    out = list(state["tasks"].values())
    out.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return out[:limit]


def clear_stale(max_age_s: float = 86400.0) -> int:
    """Drop terminal records older than max_age_s. Returns count dropped."""
    with _lock:
        state = _load()
        now = time.time()
        before = len(state["tasks"])
        state["tasks"] = {
            tid: r for tid, r in state["tasks"].items()
            if r.get("status") not in _TERMINAL
               or (now - (r.get("finished_at") or r.get("last_update_at") or now)) <= max_age_s
        }
        dropped = before - len(state["tasks"])
        if dropped:
            _save(state)
    return dropped


__all__ = [
    "PROGRESS_PATH",
    "start", "epoch_done", "heartbeat", "finish",
    "get", "list_active", "list_all", "clear_stale",
]
