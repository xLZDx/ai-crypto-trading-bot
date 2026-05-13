"""
Training Orchestrator — runs on the master laptop.

Manages a cluster of worker nodes (other laptops running worker.py).
Assigns training tasks based on worker capabilities (GPU first, then CPU).
Exposes a REST API consumed by the dashboard monitor tab.

Start:
    python -m src.training.distributed.orchestrator
    python -m src.training.distributed.orchestrator --port 7700

Dashboard API (registered in app.py):
    GET  /api/cluster/status          — cluster overview
    GET  /api/cluster/workers         — list all workers
    POST /api/cluster/submit          — submit a training task
    POST /api/cluster/register        — worker heartbeat/register
    POST /api/cluster/task_update     — worker reports task result
    DELETE /api/cluster/task/<id>     — cancel a task
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator")

# Optional ML Engineer Agent — pre-flight + post-flight AFML gate. Imported at
# module scope so tests can monkeypatch `_get_ml_engineer`. Failures are
# non-fatal (the orchestrator runs even without the gate, with a CRITICAL log).
try:
    from src.engine.ml_engineer_agent import get_ml_engineer as _get_ml_engineer
except Exception as _mle_imp_err:  # pragma: no cover — defensive
    _get_ml_engineer = None
    logger.critical(
        "[Orch] ML Engineer agent unavailable at import time: %s — pre/post "
        "flight AFML gates will not run.", _mle_imp_err,
    )

# Optional KPI Gate (Sprint 1A R2) — auto-retire after 3 consecutive threshold
# misses. Importable at module scope for monkeypatch in tests. Non-fatal on
# import failure.
try:
    from src.engine import kpi_gate as _kpi_gate
except Exception as _kpi_imp_err:  # pragma: no cover
    _kpi_gate = None
    logger.warning(
        "[Orch] KPI gate unavailable at import time: %s — auto-retire disabled.",
        _kpi_imp_err,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ORCH_PORT       = 7700
WORKER_TIMEOUT  = 60    # mark worker offline after N seconds without heartbeat
MAX_TASK_RETRIES = 2

# 2026-05-10 — task progress watchdog (Phase 88).
# A task held "running" indefinitely by a zombie worker (e.g. trainer
# thread crashed silently while heartbeat keeps reporting busy) was the
# #1 cause of confusion during the 2026-05-10 distributed sweep. The
# watchdog scans running tasks every WATCHDOG_POLL_S seconds; if a task
# has been running longer than the per-lane TIMEOUT and hasn't received
# a status update in STALE_UPDATE_S, mark it failed (error =
# "watchdog_timeout") and free the worker so it picks up the next task.
# Phase C1 (2026-05-12): orchestrator state is volatile until persisted.
# A crash or restart used to lose the entire queue + worker registry.
# Persist after every mutation via safe_json atomic write; reload on
# __init__. Running tasks at shutdown time are re-queued because the
# worker that owned them may not survive the restart.
ORCH_STATE_PATH = PROJECT_ROOT / "data" / "orchestrator_state.json"
ORCH_STATE_SCHEMA_VERSION = 1

WATCHDOG_POLL_S            = 30
WATCHDOG_STALE_UPDATE_S    = 5 * 60       # 5 min without an update + over-budget = dead
WATCHDOG_TIMEOUT_BY_KIND   = {
    "cpu":       60 * 60,                 # CPU model: 60 min hard cap
    "gpu":       120 * 60,                # GPU model: 120 min hard cap
    "exclusive": 180 * 60,                # OFT-class: 180 min hard cap
    # 2026-05-11 — TFT @ 1h needs ~108 min per epoch at current dataset
    # size (19077 batches @ 2.94 it/s, observed in worker_razer_gpu.out.log
    # before watchdog killed a healthy run). The 120-min "gpu" cap can't
    # accommodate multi-epoch neural training. The "neural" kind gets a
    # 6-hour budget — paired with the worker-side defensive heartbeat, the
    # cluster will only kill genuinely-stuck neural trainers.
    "neural":    6 * 60 * 60,             # TFT / multi-epoch neural: 6h hard cap
}
WATCHDOG_DEFAULT_TIMEOUT_S = 60 * 60


class Orchestrator:
    """
    In-process orchestrator — can be embedded in the dashboard Flask app
    or run as a standalone process.
    """

    def __init__(self, state_path: Path | None = None):
        self._lock    = threading.Lock()
        self._workers: dict[str, dict] = {}      # node_id → WorkerInfo dict
        self._tasks:   dict[str, dict] = {}       # task_id → TrainingTask dict
        self._queue:   list[str]       = []       # task_ids in order
        self._running  = False
        self._schedule_thread: threading.Thread | None = None
        # Phase C1 — state persistence. Tests inject `state_path` to point
        # at a tmpdir; production uses the project-rooted default.
        self._state_path: Path = state_path if state_path is not None else ORCH_STATE_PATH
        self._load_state()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._schedule_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="orch-scheduler"
        )
        self._schedule_thread.start()
        # 2026-05-10 — task progress watchdog (Phase 88). Catches zombie
        # workers that report 'busy' on heartbeat but whose task thread
        # has crashed silently.
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="orch-watchdog"
        )
        self._watchdog_thread.start()
        logger.info("[Orch] Orchestrator started (scheduler + watchdog)")

    def stop(self) -> None:
        self._running = False

    # ── State persistence (Phase C1) ──────────────────────────────────────────

    @staticmethod
    def _is_safe_worker_entry(w: dict) -> bool:
        """Phase C reviewer fix (SSRF): persisted state could carry a worker
        IP that points outside the LAN. _send_task_to_worker POSTs to that
        IP; an attacker with file-write could re-target the orchestrator at
        an internal/cloud-metadata endpoint. Accept only loopback or RFC1918
        private addresses + a sane port range. Python's `is_private` includes
        link-local (169.254.0.0/16) — which is exactly the cloud metadata
        target — so we explicitly exclude link-local, multicast, reserved,
        and unspecified ranges.
        """
        import ipaddress
        ip = w.get("ip", "")
        port = w.get("port", 0)
        if not isinstance(port, int) or not (1024 <= port <= 65535):
            return False
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        if (addr.is_link_local or addr.is_multicast
                or addr.is_reserved or addr.is_unspecified):
            return False
        return addr.is_private or addr.is_loopback

    @staticmethod
    def _is_safe_local_path(p: str, allowed_root: Path) -> bool:
        """Phase C reviewer fix (path injection): persisted task fields
        `data_path` / `output_path` flow to the worker's pd.read_csv +
        joblib.dump. Reject anything outside the allowed project subtree.
        Empty string is treated as 'use default' and allowed."""
        if not p:
            return True
        try:
            abs_p = Path(p).resolve()
            abs_root = allowed_root.resolve()
            abs_p.relative_to(abs_root)
            return True
        except (ValueError, OSError):
            return False

    def _load_state(self) -> None:
        """Restore _workers / _tasks / _queue from disk if a saved state exists.

        Running tasks are re-queued because the worker that owned them may
        not survive the restart (worker heartbeats will refresh worker
        liveness within ~30 s; their tasks won't auto-resume).
        Phase C reviewer hardening: drop worker entries with non-LAN IPs and
        neutralize task path fields that escape the project tree.
        """
        try:
            from src.utils.safe_json import read_json
        except Exception:
            return
        data = read_json(str(self._state_path), default=None)
        if not isinstance(data, dict):
            return
        try:
            workers = data.get("workers", {}) or {}
            tasks   = data.get("tasks",   {}) or {}
            queue   = list(data.get("queue", []) or [])
            if not isinstance(workers, dict) or not isinstance(tasks, dict):
                return
            # Phase C reviewer fix (SSRF): filter worker registry to LAN-only IPs.
            data_root  = PROJECT_ROOT / "data"
            model_root = PROJECT_ROOT / "models"
            sanitized_workers: dict[str, dict] = {}
            for nid, w in workers.items():
                if not isinstance(w, dict):
                    continue
                if self._is_safe_worker_entry(w):
                    sanitized_workers[nid] = w
                else:
                    logger.warning(
                        "[Orch] _load_state: dropped worker %s with unsafe ip=%r port=%r",
                        nid, w.get("ip"), w.get("port"),
                    )
            workers = sanitized_workers

            # Phase C1 — re-queue any in-flight task whose worker is gone.
            # Phase C reviewer fix: neutralize untrusted path fields.
            for tid, t in list(tasks.items()):
                if not isinstance(t, dict):
                    tasks.pop(tid, None)
                    continue
                if not self._is_safe_local_path(t.get("data_path", ""), data_root):
                    logger.warning("[Orch] _load_state: blanking unsafe data_path on task %s", tid)
                    t["data_path"] = ""
                if not self._is_safe_local_path(t.get("output_path", ""), model_root):
                    logger.warning("[Orch] _load_state: blanking unsafe output_path on task %s", tid)
                    t["output_path"] = str(model_root)
                if t.get("status") == "running":
                    t["status"] = "pending"
                    t["assigned_to"] = ""
                    if tid not in queue:
                        queue.append(tid)
                    logger.info("[Orch] Re-queued task %s (was running at shutdown)", tid)
            # Drop queue entries that don't have a task dict any more.
            queue = [tid for tid in queue if tid in tasks]
            self._workers, self._tasks, self._queue = workers, tasks, queue
            logger.info(
                "[Orch] Loaded persisted state: %d workers, %d tasks, %d queued",
                len(self._workers), len(self._tasks), len(self._queue),
            )
        except Exception as exc:  # never let a corrupt state file block startup
            logger.warning("[Orch] _load_state failed, starting fresh: %s", exc)

    def _snapshot_state_locked(self) -> dict:
        """Caller MUST hold `self._lock`. Cheap shallow copies of the three
        in-memory dicts — keeps the snapshot consistent so the disk write
        can happen without holding the lock."""
        return {
            "schema_version": ORCH_STATE_SCHEMA_VERSION,
            "saved_at":       datetime.now(timezone.utc).isoformat(),
            "workers":        dict(self._workers),
            "tasks":          {tid: dict(t) for tid, t in self._tasks.items()},
            "queue":          list(self._queue),
        }

    def _write_snapshot(self, snapshot: dict) -> None:
        """Disk write. Safe to call without holding `self._lock` (and
        intentionally called that way — see _persist below). Failures are
        logged but never raise so the orchestrator stays operational even
        if disk is full / read-only."""
        try:
            from src.utils.safe_json import write_json
        except Exception:
            return
        try:
            write_json(str(self._state_path), snapshot, indent=2)
        except Exception as exc:
            logger.debug("[Orch] _persist failed: %s", exc)

    def _persist(self) -> None:
        """Atomic snapshot of in-memory state to `data/orchestrator_state.json`.

        Phase C reviewer fix: snapshot acquired under the lock (consistent
        state) but the disk write happens AFTER releasing the lock so a
        slow filelock acquire (e.g. another process reading the file)
        cannot stall the scheduler or watchdog.

        Caller MUST NOT be holding `self._lock` when calling this — we
        re-acquire it internally for the cheap dict-copy step.
        """
        with self._lock:
            snapshot = self._snapshot_state_locked()
        self._write_snapshot(snapshot)

    # ── Worker registration ───────────────────────────────────────────────────

    def register_worker(self, info: dict) -> None:
        # Phase 93 — heartbeat payload now carries live load fields
        # (cpu_percent, gpu_percent, gpu_mem_used_mb, gpu_mem_total_mb,
        # uptime_s). They flow through the {**prev, **info} merge below
        # without any per-field plumbing — the dashboard's Live Load
        # column reads them straight off the worker dict in
        # /api/cluster/status. Older workers without these fields just
        # leave them None and the UI degrades gracefully.
        node_id = info.get("node_id", "")
        if not node_id:
            return
        with self._lock:
            prev = self._workers.get(node_id, {})
            info["last_seen"] = datetime.now(timezone.utc).isoformat()
            # Don't overwrite status if we just assigned it a task
            if prev.get("status") == "busy" and info.get("status") == "idle" and prev.get("current_task"):
                info["status"] = "busy"
            self._workers[node_id] = {**prev, **info}
        # Persist OUTSIDE the lock — filelock contention must not stall the scheduler.
        self._persist()
        logger.debug("[Orch] Worker registered: %s (%s)", info.get("name", node_id), info.get("ip"))

    def list_workers(self) -> list[dict]:
        now = time.time()
        result = []
        with self._lock:
            for w in self._workers.values():
                w = dict(w)
                # Calculate seconds since last seen
                try:
                    ls = datetime.fromisoformat(w.get("last_seen", "").replace("Z", "+00:00"))
                    age = now - ls.timestamp()
                    w["online"] = age < WORKER_TIMEOUT
                    w["last_seen_ago"] = int(age)
                except Exception:
                    w["online"] = False
                    w["last_seen_ago"] = 9999
                result.append(w)
        return result

    # ── Task submission ───────────────────────────────────────────────────────

    def submit_task(self, task_spec: dict) -> str:
        """Submit a training task. Returns task_id.

        Phase C2 (2026-05-12): dedupes by (model_type, symbol, timeframe).
        If a pending or running task with the same triple already exists,
        the existing task_id is returned instead of queueing a duplicate.
        Done/failed/cancelled tasks are NOT deduped — resubmitting after
        completion is the legitimate retrain path.
        """
        model_type = task_spec.get("model_type", "btc_rf")
        symbol     = task_spec.get("symbol",     "BTC/USDT")
        timeframe  = task_spec.get("timeframe",  "1m")

        # ── KPI gate retirement check ──
        # If (model_type, timeframe) was auto-retired by Sprint 1A R2 KPI gate
        # (3 consecutive runs failed thresholds), refuse the task with a
        # sentinel and let the operator restore via /api/registry/<key>/restore.
        # `force=True` in task_spec overrides the retirement check (allows
        # operator-initiated retraining of a retired cell to attempt recovery).
        if _kpi_gate is not None and not task_spec.get("force", False):
            try:
                if _kpi_gate.is_retired(model_type, timeframe):
                    logger.error(
                        "[Orch] Task REFUSED — %s/%s is KPI-retired. "
                        "Restore via POST /api/registry/%s__%s/restore.",
                        model_type, timeframe, model_type, timeframe,
                    )
                    sentinel = f"retired-kpi-{int(datetime.now(timezone.utc).timestamp())}"
                    with self._lock:
                        self._tasks[sentinel] = {
                            "task_id":     sentinel,
                            "model_type":  model_type,
                            "symbol":      symbol,
                            "timeframe":   timeframe,
                            "status":      "retired",
                            "blocked_by":  "kpi_gate",
                            "reasons":     [f"{model_type}/{timeframe} auto-retired by KPI gate"],
                            "created_at":  datetime.now(timezone.utc).isoformat(),
                        }
                    return sentinel
            except Exception as e:
                logger.warning("[Orch] KPI is_retired check error: %s", e)

        # ── ML Engineer pre-flight gate ──
        # Validates AFML compliance before allocating a task_id. BLOCK = task
        # is refused and a structured error is returned via "blocked-<reason>"
        # task_id sentinel so the caller can surface the reason in the UI.
        # Module-level _get_ml_engineer is None when the agent module is broken
        # (logged CRITICAL at import time).
        if _get_ml_engineer is not None:
            try:
                ml_decision = _get_ml_engineer().validate_training_request(
                    model_type=model_type,
                    timeframe=timeframe,
                    config=task_spec.get("config", {}) | task_spec.get("overrides", {}),
                )
                if ml_decision.decision == 'BLOCK':
                    logger.error(
                        "[Orch] Task BLOCKED by ML Engineer agent: model=%s tf=%s reasons=%s",
                        model_type, timeframe, ml_decision.reasons,
                    )
                    # Store a blocked-task record so the caller's get_task() can
                    # find it (previously the sentinel vanished from get_task).
                    sentinel = f"blocked-mle-{int(datetime.now(timezone.utc).timestamp())}"
                    with self._lock:
                        self._tasks[sentinel] = {
                            "task_id":     sentinel,
                            "model_type":  model_type,
                            "symbol":      symbol,
                            "timeframe":   timeframe,
                            "status":      "blocked",
                            "blocked_by":  "ml_engineer",
                            "reasons":     list(ml_decision.reasons),
                            "warnings":    list(ml_decision.warnings),
                            "created_at":  datetime.now(timezone.utc).isoformat(),
                        }
                    return sentinel
            except Exception as e:
                # Never let the gate crash task submission. Log and continue.
                logger.warning("[Orch] ML Engineer pre-flight gate error: %s", e, exc_info=True)

        with self._lock:
            # Phase C2 — dedup before allocating a new task_id.
            for existing_id, existing in self._tasks.items():
                if (existing.get("status") in ("pending", "running")
                        and existing.get("model_type") == model_type
                        and existing.get("symbol")     == symbol
                        and existing.get("timeframe")  == timeframe):
                    logger.info(
                        "[Orch] Task dedup: %s/%s/%s already %s as %s — returning existing id",
                        model_type, symbol, timeframe, existing.get("status"), existing_id,
                    )
                    return existing_id

            task_id = str(uuid.uuid4())[:12]
            now_iso = datetime.now(timezone.utc).isoformat()
            task = {
                "task_id":         task_id,
                "model_type":      model_type,
                "symbol":          symbol,
                "timeframe":       timeframe,
                "config":          task_spec.get("config", {}),
                "data_path":       task_spec.get("data_path", ""),
                "output_path":     task_spec.get("output_path", str(PROJECT_ROOT / "models")),
                "status":          "pending",
                "assigned_to":     "",
                "created_at":      now_iso,
                "started_at":      "",
                "finished_at":     "",
                # Watchdog timestamp — refreshed on every update_task() and on
                # dispatch. If now - last_update_at > stale window AND elapsed
                # > timeout, the watchdog kills the task.
                "last_update_at":  now_iso,
                "result":          {},
                "error":           "",
                "retries":         0,
            }
            self._tasks[task_id] = task
            self._queue.append(task_id)
        # Persist OUTSIDE the lock.
        self._persist()
        logger.info("[Orch] Task submitted: %s / %s / %s", task_id, task["model_type"], task["symbol"])
        return task_id

    def update_task(self, task_id: str, status: str, node_id: str = "",
                    result: dict | None = None, error: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            # Watchdog timestamp — refresh on every status update from a
            # worker so stale-task detection knows the trainer is alive.
            task["last_update_at"] = datetime.now(timezone.utc).isoformat()
            # 2026-05-11 — "heartbeat" is a defensive worker-side signal
            # that ONLY refreshes the stale-window timer. The task's status,
            # started_at, assigned_to etc. stay as-is. Used by trainers that
            # don't naturally call back (e.g. Darts/Lightning's tqdm-only
            # progress); without this, the 5-min stale gate would mis-fire
            # on long-running but actively-progressing neural runs.
            if status == "heartbeat":
                return
            task["status"] = status
            # 2026-05-12 live-validation fix: a status transition back to
            # "pending" (e.g. from `_send_task_to_worker` after the worker
            # POST fails) was leaving the task orphaned — out of `_queue`
            # because `_dispatch_pending` had removed it on assignment,
            # but no caller re-added it. The scheduler then never picked
            # it up again. Re-add to the queue AND clear assigned_to so
            # the next dispatch cycle picks a different worker.
            if status == "pending":
                task["assigned_to"] = ""
                if task_id not in self._queue:
                    self._queue.append(task_id)
            elif node_id:
                task["assigned_to"] = node_id
            if result:
                task["result"] = result
            if error:
                task["error"] = error
            if status == "running":
                task["started_at"] = datetime.now(timezone.utc).isoformat()
                # Update worker status
                if node_id and node_id in self._workers:
                    self._workers[node_id]["status"] = "busy"
                    self._workers[node_id]["current_task"] = task_id
            elif status in ("done", "failed", "cancelled"):
                task["finished_at"] = datetime.now(timezone.utc).isoformat()
                # Free worker
                if node_id and node_id in self._workers:
                    self._workers[node_id]["status"] = "idle"
                    self._workers[node_id]["current_task"] = ""
                    if status == "done":
                        self._workers[node_id]["tasks_done"] = self._workers[node_id].get("tasks_done", 0) + 1
                    else:
                        self._workers[node_id]["tasks_failed"] = self._workers[node_id].get("tasks_failed", 0) + 1

                # ── ML Engineer post-training gate ──
                # On status='done', evaluate the model meta JSON against KPI
                # floors. REJECT moves the artifact to quarantine.
                if status == "done" and _get_ml_engineer is not None:
                    try:
                        from pathlib import Path as _P
                        model_type = task.get("model_type", "")
                        tf         = task.get("timeframe", "")
                        # Derive meta JSON path from model_paths helper
                        meta_path = None
                        try:
                            from src.utils.model_paths import artifact_paths
                            paths = artifact_paths(model_type, tf)
                            meta_path = paths.get('meta')
                        except Exception:
                            pass
                        if meta_path and _P(meta_path).exists():
                            post_decision = _get_ml_engineer().evaluate_trained_model(
                                model_type=model_type,
                                timeframe=tf,
                                meta_json_path=str(meta_path),
                            )
                            task["ml_engineer_decision"] = {
                                "decision": post_decision.decision,
                                "reasons":  post_decision.reasons,
                                "warnings": post_decision.warnings,
                                "metrics":  post_decision.metrics,
                            }
                            if post_decision.decision == 'REJECT':
                                logger.error(
                                    "[Orch] Model REJECTED by ML Engineer: %s/%s reasons=%s",
                                    model_type, tf, post_decision.reasons,
                                )
                                # Task remains 'done' (it ran) but KPI gate failed.
                                # The dashboard surfaces ml_engineer_decision so the
                                # operator can quarantine the artifact.
                    except Exception as e:
                        logger.warning("[Orch] ML Engineer post-flight gate skipped: %s", e)

                # ── KPI Gate post-flight (Sprint 1A R2) ──
                # Persists a TrainingResult row, checks last-3 strikes, and
                # auto-retires the (model, tf) cell on 3 consecutive misses.
                if status == "done" and _kpi_gate is not None:
                    try:
                        model_type = task.get("model_type", "")
                        tf         = task.get("timeframe", "")
                        meta_path  = None
                        try:
                            from src.utils.model_paths import artifact_paths
                            paths = artifact_paths(model_type, tf)
                            meta_path = paths.get('meta')
                        except Exception:
                            pass
                        if meta_path and Path(meta_path).exists():
                            kpi_outcome = _kpi_gate.evaluate_from_meta_json(
                                model_key=model_type,
                                tf=tf,
                                meta_json_path=str(meta_path),
                            )
                            task["kpi_gate"] = kpi_outcome
                            if kpi_outcome.get("retired_now"):
                                logger.error(
                                    "[Orch] KPI gate AUTO-RETIRED %s/%s after 3 strikes — "
                                    "subsequent submissions will be blocked until restore. "
                                    "Missed fields: %s",
                                    model_type, tf, kpi_outcome.get("missed_fields"),
                                )
                    except Exception as e:
                        logger.warning("[Orch] KPI gate post-flight skipped: %s", e)

                # Retry on failure
                if status == "failed" and task.get("retries", 0) < MAX_TASK_RETRIES:
                    task["retries"] = task.get("retries", 0) + 1
                    task["status"] = "pending"
                    task["assigned_to"] = ""
                    if task_id not in self._queue:
                        self._queue.append(task_id)
                    logger.warning("[Orch] Task %s failed — retry %d/%d", task_id, task["retries"], MAX_TASK_RETRIES)
        # Persist OUTSIDE the lock.
        self._persist()

    def cancel_task(self, task_id: str) -> bool:
        cancelled = False
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task["status"] == "pending":
                task["status"] = "cancelled"
                if task_id in self._queue:
                    self._queue.remove(task_id)
                cancelled = True
        if cancelled:
            self._persist()
        return cancelled

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            return dict(self._tasks[task_id]) if task_id in self._tasks else None

    def list_tasks(self, limit: int = 50) -> list[dict]:
        with self._lock:
            tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tasks[:limit]

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        while self._running:
            try:
                self._dispatch_pending()
            except Exception as exc:
                logger.debug("[Orch] Scheduler error: %s", exc)
            time.sleep(5)

    def _watchdog_loop(self) -> None:
        """Phase 88 watchdog. Every WATCHDOG_POLL_S, scan running tasks
        and kill any that have:
          1. exceeded the per-lane wall-clock timeout (started_at), AND
          2. not received a status update in the last STALE window
        For each killed task: status='failed', error='watchdog_timeout',
        free the worker (idle, current_task=''). The dispatcher then
        picks the next pending task for that worker on the next cycle.

        This is the server-side fix for 'zombie worker holds task running
        forever' — observed during the 2026-05-10 sweep when Ivan's
        trainer thread crashed silently while heartbeats kept lying.
        """
        while self._running:
            try:
                self._sweep_stale_tasks()
            except Exception as exc:
                logger.debug("[Orch] Watchdog error: %s", exc)
            time.sleep(WATCHDOG_POLL_S)

    def _sweep_stale_tasks(self) -> None:
        # Resolve each task's resource_kind to pick the right timeout.
        try:
            from src.training.training_rules import resource_kind as _rkind
        except Exception:
            _rkind = lambda _m: "cpu"
        now = datetime.now(timezone.utc)
        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if task.get("status") != "running":
                    continue
                # Wall-clock elapsed since the task started executing.
                started_iso = task.get("started_at") or task.get("created_at")
                try:
                    started = datetime.fromisoformat(started_iso.replace("Z", "+00:00"))
                except Exception:
                    continue
                elapsed_s = (now - started).total_seconds()
                # Stale-update window — refreshed on every update_task() call.
                last_update_iso = task.get("last_update_at") or started_iso
                try:
                    last_update = datetime.fromisoformat(last_update_iso.replace("Z", "+00:00"))
                except Exception:
                    last_update = started
                stale_s = (now - last_update).total_seconds()
                # Per-lane timeout from training_rules.
                kind = "cpu"
                try:
                    kind = _rkind(task.get("model_type", "")) or "cpu"
                except Exception:
                    pass
                timeout_s = WATCHDOG_TIMEOUT_BY_KIND.get(kind, WATCHDOG_DEFAULT_TIMEOUT_S)
                # Kill condition: elapsed over budget AND no recent update
                # from the worker. Both gates must trip — a long-running
                # but actively-progressing task should NOT be killed.
                if elapsed_s > timeout_s and stale_s > WATCHDOG_STALE_UPDATE_S:
                    logger.warning(
                        "[Orch] WATCHDOG killing stale task %s %s@%s "
                        "(elapsed=%ds > %ds AND stale=%ds > %ds)",
                        task_id, task.get("model_type"), task.get("timeframe"),
                        int(elapsed_s), timeout_s, int(stale_s), WATCHDOG_STALE_UPDATE_S,
                    )
                    task["status"]      = "failed"
                    task["error"]       = (
                        f"watchdog_timeout: elapsed={int(elapsed_s)}s exceeds "
                        f"{kind}-lane budget {timeout_s}s, last update {int(stale_s)}s ago"
                    )
                    task["finished_at"] = now.isoformat()
                    task["last_update_at"] = now.isoformat()
                    # Free the assigned worker so the dispatcher reassigns.
                    nid = task.get("assigned_to", "")
                    if nid and nid in self._workers:
                        self._workers[nid]["status"]       = "idle"
                        self._workers[nid]["current_task"] = ""
                        self._workers[nid]["tasks_failed"] = (
                            self._workers[nid].get("tasks_failed", 0) + 1
                        )
        # Persist OUTSIDE the lock.
        self._persist()

    def _dispatch_pending(self) -> None:
        with self._lock:
            if not self._queue:
                return
            # Find idle online workers
            idle = [
                w for w in self._workers.values()
                if w.get("status") == "idle"
                and w.get("online", True)
                and w.get("last_seen_ago", 0) < WORKER_TIMEOUT
            ]
            if not idle:
                return
            # Sort: GPU workers first, then by VRAM descending — used as the
            # tie-breaker after lane match.
            idle.sort(key=lambda w: (-int(w.get("cuda_available", False)), -w.get("gpu_vram_gb", 0)))

            # 2026-05-10 — lane-aware dispatch. Each task carries a model_type;
            # we look up its resource_kind from training_rules.json (cpu / gpu /
            # exclusive) and route to a worker whose lane MATCHES. Lane "any"
            # accepts every kind (legacy/back-compat).
            #
            # Mapping:
            #   resource_kind=cpu        -> lane in {cpu, any}
            #   resource_kind=gpu        -> lane in {gpu, any}
            #   resource_kind=exclusive  -> lane in {gpu, any}  (gpu lane runs OFT;
            #                                                    sweep coordinator
            #                                                    serialises so OFT
            #                                                    runs alone)
            #   resource_kind=neural     -> lane in {gpu, any}  (TFT-class — needs
            #                                                    GPU; gets the 6h
            #                                                    watchdog budget
            #                                                    from Phase 101.)
            try:
                from src.training.training_rules import resource_kind as _rkind
            except Exception:
                _rkind = lambda _m: "cpu"  # safe default

            def _lane_accepts(worker_lane: str, kind: str) -> bool:
                if worker_lane == "any":
                    return True
                if kind == "cpu":
                    return worker_lane == "cpu"
                if kind in ("gpu", "exclusive", "neural"):
                    return worker_lane == "gpu"
                return True

            pending = [tid for tid in self._queue if self._tasks.get(tid, {}).get("status") == "pending"]
            assigned_workers: set[str] = set()
            for task_id in pending:
                task = self._tasks[task_id]
                kind = "cpu"
                try:
                    kind = _rkind(task.get("model_type", "")) or "cpu"
                except Exception:
                    pass
                # Find first idle worker (GPU-sorted) whose lane accepts this kind
                # AND that we haven't already assigned this round.
                worker = next(
                    (w for w in idle
                     if w["node_id"] not in assigned_workers
                     and _lane_accepts(w.get("lane", "any"), kind)),
                    None,
                )
                if worker is None:
                    continue   # no compatible worker idle; task stays pending
                task["status"]      = "running"
                task["assigned_to"] = worker["node_id"]
                task["started_at"]  = datetime.now(timezone.utc).isoformat()
                worker["status"]        = "busy"
                worker["current_task"]  = task_id
                assigned_workers.add(worker["node_id"])
                if task_id in self._queue:
                    self._queue.remove(task_id)
                # Dispatch in background (don't hold lock during HTTP call)
                threading.Thread(
                    target=self._send_task_to_worker,
                    args=(worker, dict(task)),
                    daemon=True,
                ).start()
            _did_assign = bool(assigned_workers)
        # Persist OUTSIDE the lock so a crash mid-dispatch doesn't lose the
        # "assigned to X" record. Done only when something actually changed.
        if _did_assign:
            self._persist()

    def _send_task_to_worker(self, worker: dict, task: dict) -> None:
        import requests
        ip, port, node_id = worker["ip"], worker["port"], worker["node_id"]
        try:
            r = requests.post(f"http://{ip}:{port}/task", json=task, timeout=15)
            if r.status_code == 200:
                logger.info("[Orch] Task %s → %s (%s:%s)", task["task_id"], worker.get("name", node_id), ip, port)
            else:
                logger.warning("[Orch] Worker %s rejected task: %s", node_id, r.text[:200])
                self.update_task(task["task_id"], "failed", node_id, error=f"Worker rejected: {r.status_code}")
        except Exception as exc:
            logger.warning("[Orch] Cannot reach worker %s: %s", node_id, exc)
            self.update_task(task["task_id"], "pending", node_id)  # re-queue

    # ── Status summary ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        workers = self.list_workers()
        tasks   = self.list_tasks(100)
        return {
            "workers_total":  len(workers),
            "workers_online": sum(1 for w in workers if w.get("online")),
            "workers_idle":   sum(1 for w in workers if w.get("status") == "idle" and w.get("online")),
            "workers_busy":   sum(1 for w in workers if w.get("status") == "busy"),
            "tasks_pending":  sum(1 for t in tasks if t["status"] == "pending"),
            "tasks_running":  sum(1 for t in tasks if t["status"] == "running"),
            "tasks_done":     sum(1 for t in tasks if t["status"] == "done"),
            "tasks_failed":   sum(1 for t in tasks if t["status"] == "failed"),
            "workers":        workers,
            "recent_tasks":   tasks[:20],
        }

    # ── Batch job helpers ─────────────────────────────────────────────────────

    def submit_full_training_run(self, symbols: list[str] | None = None) -> list[str]:
        """Submit training tasks for all models across all symbols."""
        if symbols is None:
            watchlist_file = PROJECT_ROOT / "data" / "watchlist.json"
            symbols = json.loads(watchlist_file.read_text()) if watchlist_file.exists() else ["BTC/USDT"]

        model_configs = [
            {"model_type": "btc_rf",        "timeframe": "1m", "config": {"n_estimators": 200}},
            {"model_type": "trend",          "timeframe": "1h", "config": {}},
            {"model_type": "scalping",       "timeframe": "1m", "config": {}},
            {"model_type": "meta_labeler",   "timeframe": "1m", "config": {}},
            {"model_type": "futures_short",  "timeframe": "1m", "config": {}},
            {"model_type": "regime",         "timeframe": "1h", "config": {}},
            # OFT (Order Flow Transformer) — single-symbol single-machine in
            # current implementation, but listed so the cluster scheduler
            # picks it up once joint_oft_rl supports multi-worker sharding.
            {"model_type": "oft",            "timeframe": "1m", "config": {"epochs": 5, "skip_sac": True}},
        ]
        task_ids = []
        for sym in symbols:
            safe = sym.replace("/", "_")
            for mc in model_configs:
                data_path = str(PROJECT_ROOT / "data" / "raw" / f"{safe}_{mc['timeframe']}.csv.gz")
                tid = self.submit_task({
                    **mc,
                    "symbol":      sym,
                    "data_path":   data_path,
                    "output_path": str(PROJECT_ROOT / "models"),
                })
                task_ids.append(tid)
        return task_ids


# ─── Singleton for dashboard embedding ───────────────────────────────────────

_orch_instance: Orchestrator | None = None
_orch_lock = threading.Lock()


def get_orchestrator() -> Orchestrator:
    global _orch_instance
    if _orch_instance is None:
        with _orch_lock:
            if _orch_instance is None:
                _orch_instance = Orchestrator()
                _orch_instance.start()
    return _orch_instance


# ─── Standalone HTTP server ───────────────────────────────────────────────────

def _build_standalone_app(orch: Orchestrator):
    from flask import Flask, jsonify, request as freq, abort
    import hmac as _hmac
    import os as _os
    # Load .env so CLUSTER_API_KEY / DASHBOARD_API_KEY are picked up when the
    # orchestrator process is launched via Win32_Process.Create (which does
    # NOT inherit the parent shell's exported env vars).
    try:
        from dotenv import load_dotenv as _load_dotenv
        from pathlib import Path as _Path
        _env_path = _Path(__file__).resolve().parents[3] / ".env"
        if _env_path.exists():
            _load_dotenv(_env_path)
    except Exception:
        pass  # dotenv is optional; env vars may already be set
    app = Flask("orchestrator")

    # SEC-4 fix: shared-secret auth on mutation endpoints. Reads CLUSTER_API_KEY
    # or DASHBOARD_API_KEY (same key the dashboard uses). When unset, the
    # auth check fails open — matching the dashboard's existing policy — but
    # the operator gets a CRITICAL log line so they cannot miss it.
    _CLUSTER_KEY = (_os.getenv("CLUSTER_API_KEY")
                    or _os.getenv("API_KEY")
                    or _os.getenv("DASHBOARD_API_KEY")
                    or "").strip()
    if not _CLUSTER_KEY:
        logger.critical(
            "[Orch] No CLUSTER_API_KEY / DASHBOARD_API_KEY set — mutation "
            "endpoints (/submit, /register, /task_update, /cancel) are "
            "UNPROTECTED. Set the env var to enable auth."
        )

    def _require_cluster_auth():
        """Apply at the start of each mutation route handler."""
        if not _CLUSTER_KEY:
            return  # fail-open when key unset (logged at startup)
        provided = freq.headers.get("X-API-Key", "") or ""
        if not provided or not _hmac.compare_digest(provided, _CLUSTER_KEY):
            abort(401)

    @app.route("/api/cluster/status")
    def status():
        return jsonify(orch.get_status())

    # ── Phase 0 institutional upgrade: parquet store + ZMQ databus ────────
    @app.route("/api/parquet/status")
    def parquet_status():
        try:
            from src.database.parquet_store import get_store
            return jsonify(get_store().status())
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/databus/stats")
    def databus_stats():
        try:
            from src.transport.data_bus import get_data_bus
            return jsonify(get_data_bus().stats())
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/cluster/workers")
    def workers():
        return jsonify(orch.list_workers())

    @app.route("/api/cluster/tasks")
    def tasks():
        return jsonify(orch.list_tasks())

    @app.route("/api/cluster/submit", methods=["POST"])
    def submit():
        _require_cluster_auth()
        spec = freq.get_json(force=True) or {}
        tid  = orch.submit_task(spec)
        return jsonify({"ok": True, "task_id": tid})

    @app.route("/api/cluster/submit_all", methods=["POST"])
    def submit_all():
        _require_cluster_auth()
        body    = freq.get_json(force=True) or {}
        symbols = body.get("symbols")
        ids     = orch.submit_full_training_run(symbols)
        return jsonify({"ok": True, "task_ids": ids, "count": len(ids)})

    @app.route("/api/cluster/register", methods=["POST"])
    def register():
        _require_cluster_auth()
        body = freq.get_json(force=True) or {}
        # SEC-7 fix: validate IP at runtime — was only checked at state-load.
        if hasattr(orch, '_is_safe_worker_entry') and not orch._is_safe_worker_entry(body):
            logger.warning("[Orch] register_worker: rejected unsafe entry ip=%r", body.get("ip"))
            abort(400, description="unsafe worker entry")
        orch.register_worker(body)
        return jsonify({"ok": True})

    @app.route("/api/cluster/task_update", methods=["POST"])
    def task_update():
        _require_cluster_auth()
        body = freq.get_json(force=True) or {}
        orch.update_task(
            body.get("task_id", ""),
            body.get("status", ""),
            node_id=body.get("node_id", ""),
            result=body.get("result"),
            error=body.get("error", ""),
        )
        return jsonify({"ok": True})

    @app.route("/api/cluster/task/<task_id>", methods=["DELETE"])
    def cancel(task_id):
        _require_cluster_auth()
        ok = orch.cancel_task(task_id)
        return jsonify({"ok": ok})

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description="AI Trading — Training Orchestrator")
    parser.add_argument("--port", type=int, default=ORCH_PORT, help=f"HTTP port (default {ORCH_PORT})")
    # 2026-05-12 Phase A2: bind defaults to localhost. The previous
    # 0.0.0.0 bind exposed the orchestrator to the LAN with no auth on
    # any endpoint (cluster/register, cluster/task_update, etc.). For
    # single-machine operation this is the safe default. For cluster
    # mode (workers on separate machines like Ivan), the operator must
    # set ORCHESTRATOR_BIND_HOST=0.0.0.0 (or the master's LAN IP) in
    # .env or pass --host explicitly.
    import os as _os
    parser.add_argument("--host", type=str,
                        default=_os.getenv("ORCHESTRATOR_BIND_HOST", "127.0.0.1"),
                        help="Bind host (default 127.0.0.1; set 0.0.0.0 for cluster mode)")
    args = parser.parse_args()

    orch = Orchestrator()
    orch.start()
    app = _build_standalone_app(orch)

    local_ip = _local_ip()
    logger.info("=" * 60)
    logger.info("Training Orchestrator — bind=%s port=%d", args.host, args.port)
    logger.info("Master LAN IP (for workers to connect): %s", local_ip)
    if args.host == "127.0.0.1":
        # ASCII-safe — cp1252 console can't encode ⚠ on Windows default stream.
        logger.info("[!] LOCALHOST-ONLY MODE - remote workers cannot connect.")
        logger.info("  For cluster mode: set ORCHESTRATOR_BIND_HOST=0.0.0.0 in .env")
        logger.info("  or restart with --host 0.0.0.0")
    else:
        logger.info("Workers connect with:")
        logger.info("  python -m src.training.distributed.worker --master http://%s:%d", local_ip, args.port)
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


def _local_ip() -> str:
    """Return the 192.168.0.x LAN IP if available, otherwise any non-loopback IP."""
    import socket as _sock
    try:
        import psutil
        for iface_addrs in psutil.net_if_addrs().values():
            for addr in iface_addrs:
                if addr.family == _sock.AF_INET and addr.address.startswith("192.168.0."):
                    return addr.address
    except Exception:
        pass
    try:
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    main()
