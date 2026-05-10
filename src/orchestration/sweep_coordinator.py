"""Sweep Coordinator daemon — drives the model-by-model distributed
training sweep across cluster workers. Auto-starts a sweep on launch,
persists state to disk, refuses to start a second concurrent instance.

Architecture (consumes the existing cluster orchestrator on :7700):

  +--------------------------------------------------------------+
  |  sweep_coordinator (this process, port 7710 control plane)   |
  |  - reads training_rules.json -> 26 (model, TF) combos        |
  |  - PLAN_ORDER: cpu models first, gpu (tft, oft) last         |
  |  - submits all TFs of one model -> waits -> next model       |
  |  - retries transient failures once; permanent-fail after     |
  |  - triggers run_full_backtest() after all training done      |
  |  - persists data/sweep_state.json on every transition        |
  +-----------------------------+--------------------------------+
                                | HTTP submit / poll
                                v
  +--------------------------------------------------------------+
  |  cluster orchestrator on :7700 (existing, untouched)         |
  |  - queues tasks, fans across idle workers (5s dispatch)      |
  +-----------------+--------------+-----------------------------+
                    | POST /task   | POST /task
                    v              v
  +-----------------+--+    +------+--------------------+
  | LOCAL_RAZER (master) |  | WORKER-1 (Ivan, RTX 2060) |
  +----------------------+  +---------------------------+

Design constraints (per 2026-05-10 user instructions):
  1. ONE MODEL AT A TIME — submit all TFs of one model in parallel
     (cluster fans them across both workers), wait for all to finish,
     move to next model.
  2. SKIP-IF-FRESH 24h — if model meta on disk is < 24h old, skip the
     (model, tf) combo. Workers will train it again on the next sweep.
  3. AUTO-START — when this daemon launches with a fresh state, it
     immediately starts a sweep. If state shows a sweep in progress,
     it RESUMES from where it left off.
  4. SINGLE INSTANCE — refuses to start if another sweep_coordinator
     is already running (pidfile lock at data/sweep_coordinator.pid).
  5. PERSIST STATE — every transition writes data/sweep_state.json
     atomically so a crash + relaunch resumes correctly.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH   = PROJECT_ROOT / "data" / "sweep_state.json"
PIDFILE      = PROJECT_ROOT / "data" / "sweep_coordinator.pid"

# CPU models first (cheap, predictable), GPU/exclusive last so a TFT/OFT
# failure doesn't block the easier wins.
PLAN_ORDER = ["base", "trend", "futures", "scalping", "meta",
              "regime", "tft", "oft"]

CLUSTER_BASE_URL = "http://localhost:7700"

# Tunables
POLL_S                  = 10        # how often to poll cluster for task progress
MAX_RETRIES_PER_TASK    = 1         # one retry on transient failure
SKIP_IF_FRESH_HOURS     = 24        # retrain rule (user: every 24h min)
WORKER_REGISTER_WAIT_S  = 15        # give a freshly-spawned worker time to register
CONTROL_PORT            = 7710      # Flask control plane port

# Failure patterns that mark a task transient (eligible for retry).
# Anything not matching is treated as permanent fail.
_TRANSIENT_FAIL_PATTERNS = (
    "timeout",
    "out of memory",
    "OOM",
    "insufficient_vram",     # worker can ask to reroute via this
    "ConnectionError",
    "Cannot reach worker",
    "Worker rejected",
)


def _is_transient_failure(error: str) -> bool:
    if not error:
        return False
    e = error.lower()
    return any(p.lower() in e for p in _TRANSIENT_FAIL_PATTERNS)


# ── Pidfile lock ────────────────────────────────────────────────────────

def _acquire_pidfile() -> bool:
    """Refuse to start if another sweep_coordinator is alive. Returns
    True if we acquired the lock, False if another instance is running."""
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            old_pid = 0
        # Check the recorded PID is still alive AND really is our daemon
        if old_pid:
            try:
                from src.utils import process_health as _ph
                info = _ph.find_process(_ph.KIND_TRAIN_SUPERVISOR)
                if info is not None and info.pid == old_pid:
                    logger.error("Another sweep_coordinator is alive (pid=%d) — refusing to start", old_pid)
                    return False
            except Exception:
                # If process_health import fails, fall back to bare PID check.
                try:
                    import psutil
                    if psutil.pid_exists(old_pid):
                        logger.error("Stale-but-alive PID %d holds the pidfile — refusing to start", old_pid)
                        return False
                except Exception:
                    pass
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release_pidfile() -> None:
    try:
        if PIDFILE.exists():
            PIDFILE.unlink()
    except OSError:
        pass


# ── Cluster HTTP helpers ────────────────────────────────────────────────

def _http_post(path: str, body: dict, timeout: float = 5.0) -> Optional[dict]:
    import urllib.request, urllib.error
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(f"{CLUSTER_BASE_URL}{path}", data=data,
                                      method="POST",
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("POST %s failed: %s", path, exc)
        return None


def _http_get(path: str, timeout: float = 5.0) -> Optional[dict]:
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{CLUSTER_BASE_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("GET %s failed: %s", path, exc)
        return None


# ── Coordinator ─────────────────────────────────────────────────────────

class SweepCoordinator:
    def __init__(self):
        self._lock = threading.Lock()
        self._abort_requested = False
        self._paused           = False
        self.state             = self._load_or_init_state()

    # ── State persistence ───────────────────────────────────────────────

    def _fresh_state(self) -> dict:
        from src.training.training_rules import planned_combos
        combos = planned_combos()
        models: dict = {}
        for model, tf in combos:
            if model not in models:
                models[model] = {"status": "pending", "tfs": {}}
            models[model]["tfs"][tf] = {"status": "pending", "task_id": "",
                                          "retries": 0, "error": ""}
        return {
            "sweep_id":                  datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "started_at":                datetime.now(timezone.utc).isoformat(),
            "finished_at":               None,
            "status":                    "running",        # running / paused / done / aborted / failed
            "current_model":             "",
            "current_model_idx":         -1,
            "models":                    models,
            "tasks_submitted":           0,
            "tasks_done":                0,
            "tasks_failed":              0,
            "tasks_skipped_fresh":       0,
            "tasks_failed_permanently":  [],
            "next_phase":                "training",       # training / backtest / done
            "backtest":                  {"status": "pending", "started_at": None,
                                          "finished_at": None, "error": ""},
        }

    def _load_or_init_state(self) -> dict:
        if STATE_PATH.exists():
            try:
                state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                # Resume if last status was running/paused — fresh otherwise.
                if state.get("status") in ("running", "paused"):
                    logger.info("Resuming sweep %s from disk (status=%s)",
                                state.get("sweep_id"), state.get("status"))
                    return state
            except Exception as exc:
                logger.warning("Bad sweep_state.json (%s) — starting fresh", exc)
        return self._fresh_state()

    def _save_state(self) -> None:
        with self._lock:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, STATE_PATH)

    # ── Skip-if-fresh check ─────────────────────────────────────────────

    def _is_model_fresh(self, model: str, tf: str) -> bool:
        """Return True if the model meta on disk is < SKIP_IF_FRESH_HOURS old."""
        try:
            from src.utils.model_paths import artifact_paths, KEYS
        except Exception:
            return False
        canon = {"futures_short": "futures", "btc_rf": "base",
                 "meta_labeler": "meta"}.get(model, model)
        if canon not in KEYS:
            return False
        try:
            paths = artifact_paths(canon, tf)
            meta = paths.get("meta")
            if meta and meta.exists():
                age_h = (time.time() - meta.stat().st_mtime) / 3600
                return age_h < SKIP_IF_FRESH_HOURS
        except Exception:
            pass
        return False

    # ── Worker lifecycle ────────────────────────────────────────────────

    def _spawn_local_worker(self, lane: str, port: int, name: str) -> Optional[int]:
        """Spawn one local worker on master with a specific lane (cpu|gpu).
        Each lane runs in its own python process so master can do CPU and
        GPU work simultaneously. Uses PYTHONUNBUFFERED=1 so trainer logs
        flush in real time (the 2026-05-10 stdout-buffering issue we hit
        during the smoke test)."""
        import subprocess
        venv_py = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
        if not venv_py.exists():
            logger.warning("venv python not found at %s — skipping local worker spawn", venv_py)
            return None
        cmd = [str(venv_py), "-u", "-m", "src.training.distributed.worker",
               "--master", CLUSTER_BASE_URL,
               "--name", name,
               "--lane", lane,
               "--port", str(port)]
        env = os.environ.copy()
        env["PYTHONPATH"]      = str(PROJECT_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        # GPU lane: hide CUDA from CPU workers so they can't grab VRAM.
        if lane == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        log_out = open(PROJECT_ROOT / "logs" / f"local_worker_{lane}.log", "a", encoding="utf-8")
        log_err = open(PROJECT_ROOT / "logs" / f"local_worker_{lane}.err.log", "a", encoding="utf-8")
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                stdout=log_out, stderr=log_err, creationflags=flags)
        logger.info("Spawned %s worker (lane=%s, port=%d) pid=%d", name, lane, port, proc.pid)
        return proc.pid

    def _ensure_local_workers(self) -> None:
        """Spawn TWO local workers on master — one per lane (cpu, gpu) — so
        the master node can run a CPU model and a GPU model concurrently.
        Skips spawn if a worker is already registered for that lane."""
        try:
            from src.utils import process_health as ph
            existing = ph.find_process(ph.KIND_WORKER)
            if existing:
                logger.info("Local worker(s) already running (pid=%d) — skipping spawn",
                            existing.pid)
                # Still wait briefly so any in-flight registration completes
                time.sleep(3)
                return
        except Exception:
            pass
        # Spawn cpu lane on port 7701, gpu lane on port 7702.
        self._spawn_local_worker(lane="cpu", port=7701, name="LOCAL_RAZER_CPU")
        self._spawn_local_worker(lane="gpu", port=7702, name="LOCAL_RAZER_GPU")
        logger.info("Waiting %ds for both local workers to register", WORKER_REGISTER_WAIT_S)
        time.sleep(WORKER_REGISTER_WAIT_S)

    # ── Cluster wrappers ────────────────────────────────────────────────

    def _submit_task(self, spec: dict) -> Optional[str]:
        r = _http_post("/api/cluster/submit", spec)
        if r and r.get("ok"):
            return r.get("task_id")
        logger.warning("Submit failed for %s: %s", spec, r)
        return None

    def _get_task(self, task_id: str) -> Optional[dict]:
        all_tasks = _http_get("/api/cluster/tasks")
        if not isinstance(all_tasks, list):
            return None
        for t in all_tasks:
            if t.get("task_id") == task_id:
                return t
        return None

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("=== SweepCoordinator START sweep_id=%s ===", self.state["sweep_id"])
        self._save_state()

        # Step 1: spawn TWO local workers (cpu + gpu lanes) so master runs
        # CPU and GPU work concurrently. Ivan is expected to do the same
        # on his side (user-managed).
        self._ensure_local_workers()

        # Step 2: kick off ALL gpu-lane work UPFRONT so it runs in parallel
        # with the cpu-lane sweep. The orchestrator's lane-aware dispatch
        # routes these to gpu workers only; cpu workers ignore them.
        # OFT is `exclusive` (heavy GPU + CPU) — submitted last in the GPU
        # lane so a TFT failure doesn't keep the GPU lane idle waiting on
        # OFT to finish.
        self._submit_gpu_lane()

        # Step 3: walk through CPU models in PLAN_ORDER.
        for model_idx, model in enumerate(PLAN_ORDER):
            # Skip GPU models here — they were submitted upfront in step 2.
            try:
                from src.training.training_rules import resource_kind as _rk
                if _rk(model) in ("gpu", "exclusive"):
                    continue
            except Exception:
                pass
            if self._abort_requested:
                self.state["status"] = "aborted"
                self._save_state()
                return
            while self._paused:
                time.sleep(2)
                if self._abort_requested:
                    self.state["status"] = "aborted"
                    self._save_state()
                    return
            if model not in self.state["models"]:
                logger.info("Skipping %s — not in training_rules planned combos", model)
                continue
            self.state["current_model"]      = model
            self.state["current_model_idx"]  = model_idx
            self._save_state()
            self._run_one_model(model)

        # Step 4: wait for any still-running GPU lane tasks to finish
        # before triggering the final backtest (we want all models fresh).
        self._await_gpu_lane()

        # Step 5: backtest after all training done.
        if not self._abort_requested:
            self.state["next_phase"] = "backtest"
            self._save_state()
            self._run_backtest()

        self.state["status"]      = "aborted" if self._abort_requested else "done"
        self.state["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        logger.info("=== SweepCoordinator END status=%s ===", self.state["status"])

    def _submit_gpu_lane(self) -> None:
        """Submit all GPU/exclusive (model, tf) tasks immediately so they
        flow through the gpu-lane workers in parallel with the cpu sweep."""
        try:
            from src.training.training_rules import resource_kind as _rk
        except Exception:
            return
        for model in PLAN_ORDER:
            if model not in self.state["models"]:
                continue
            try:
                kind = _rk(model)
            except Exception:
                continue
            if kind not in ("gpu", "exclusive"):
                continue
            m_state = self.state["models"][model]
            for tf, tf_state in m_state["tfs"].items():
                if tf_state["status"] in ("done", "failed_permanently", "skipped_fresh"):
                    continue
                if self._is_model_fresh(model, tf):
                    tf_state["status"] = "skipped_fresh"
                    self.state["tasks_skipped_fresh"] += 1
                    logger.info("GPU-lane skip %s @ %s — fresh model on disk", model, tf)
                    continue
                spec = self._build_task_spec(model, tf)
                task_id = self._submit_task(spec)
                if task_id is None:
                    tf_state["status"] = "failed_permanently"
                    tf_state["error"]  = "submit_failed"
                    self.state["tasks_failed"] += 1
                    self.state["tasks_failed_permanently"].append(f"{model}@{tf}")
                    continue
                tf_state["task_id"]  = task_id
                tf_state["status"]   = "submitted"
                self.state["tasks_submitted"] += 1
                logger.info("GPU-lane submit %s @ %s → %s", model, tf, task_id)
        self._save_state()

    def _await_gpu_lane(self) -> None:
        """Block until every GPU/exclusive (model, tf) reaches a terminal
        state. Called after the CPU sweep finishes, before backtest."""
        try:
            from src.training.training_rules import resource_kind as _rk
        except Exception:
            return
        gpu_models = []
        for m in PLAN_ORDER:
            if m not in self.state["models"]:
                continue
            try:
                if _rk(m) in ("gpu", "exclusive"):
                    gpu_models.append(m)
            except Exception:
                pass
        if not gpu_models:
            return
        logger.info("Awaiting GPU-lane tasks for: %s", gpu_models)
        while True:
            if self._abort_requested:
                return
            all_done = True
            for model in gpu_models:
                m_state = self.state["models"][model]
                for tf, tf_state in m_state["tfs"].items():
                    if tf_state["status"] in ("done", "failed_permanently", "skipped_fresh"):
                        continue
                    all_done = False
                    if not tf_state.get("task_id"):
                        continue
                    task = self._get_task(tf_state["task_id"])
                    if task is None:
                        continue
                    cstatus = task.get("status", "")
                    if cstatus == "done":
                        tf_state["status"] = "done"
                        self.state["tasks_done"] += 1
                    elif cstatus in ("failed", "cancelled"):
                        err = task.get("error", "") or "(no error message)"
                        tf_state["error"] = err
                        if (cstatus == "failed"
                            and tf_state["retries"] < MAX_RETRIES_PER_TASK
                            and _is_transient_failure(err)):
                            tf_state["retries"] += 1
                            new_id = self._submit_task(self._build_task_spec(model, tf))
                            if new_id:
                                tf_state["task_id"] = new_id
                                tf_state["status"]  = "submitted"
                            else:
                                tf_state["status"] = "failed_permanently"
                                self.state["tasks_failed_permanently"].append(f"{model}@{tf}")
                                self.state["tasks_failed"] += 1
                        else:
                            tf_state["status"] = "failed_permanently"
                            self.state["tasks_failed_permanently"].append(f"{model}@{tf}")
                            self.state["tasks_failed"] += 1
            self._save_state()
            if all_done:
                break
            time.sleep(POLL_S)
        # Mark gpu models done if any TF succeeded.
        for model in gpu_models:
            m_state = self.state["models"][model]
            any_done = any(s["status"] == "done" for s in m_state["tfs"].values())
            any_skip = any(s["status"] == "skipped_fresh" for s in m_state["tfs"].values())
            m_state["status"] = "done" if (any_done or any_skip) else "failed"
        self._save_state()

    def _run_one_model(self, model: str) -> None:
        m_state = self.state["models"][model]
        m_state["status"] = "running"
        self._save_state()

        # Submit pending TFs (skip fresh ones)
        for tf, tf_state in m_state["tfs"].items():
            if tf_state["status"] in ("done", "failed_permanently", "skipped_fresh"):
                continue
            if self._is_model_fresh(model, tf):
                tf_state["status"] = "skipped_fresh"
                self.state["tasks_skipped_fresh"] += 1
                logger.info("Skipping %s @ %s — fresh model on disk (< %dh)",
                            model, tf, SKIP_IF_FRESH_HOURS)
                continue
            spec = self._build_task_spec(model, tf)
            task_id = self._submit_task(spec)
            if task_id is None:
                tf_state["status"] = "failed_permanently"
                tf_state["error"]  = "submit_failed"
                self.state["tasks_failed"] += 1
                self.state["tasks_failed_permanently"].append(f"{model}@{tf}")
                continue
            tf_state["task_id"]  = task_id
            tf_state["status"]   = "submitted"
            self.state["tasks_submitted"] += 1
        self._save_state()

        # Poll until all TFs done / permanently failed / skipped
        while True:
            if self._abort_requested:
                return
            while self._paused:
                time.sleep(2)
                if self._abort_requested:
                    return
            all_done = True
            for tf, tf_state in m_state["tfs"].items():
                if tf_state["status"] in ("done", "failed_permanently", "skipped_fresh"):
                    continue
                all_done = False
                task = self._get_task(tf_state["task_id"]) if tf_state["task_id"] else None
                if task is None:
                    continue
                cstatus = task.get("status", "")
                if cstatus == "done":
                    tf_state["status"] = "done"
                    tf_state["error"]  = ""
                    self.state["tasks_done"] += 1
                elif cstatus in ("failed", "cancelled"):
                    err = task.get("error", "") or "(no error message)"
                    tf_state["error"] = err
                    if (cstatus == "failed"
                        and tf_state["retries"] < MAX_RETRIES_PER_TASK
                        and _is_transient_failure(err)):
                        tf_state["retries"] += 1
                        new_id = self._submit_task(self._build_task_spec(model, tf))
                        if new_id:
                            tf_state["task_id"] = new_id
                            tf_state["status"]  = "submitted"
                            logger.info("Retry %d/%d for %s @ %s as %s",
                                        tf_state["retries"], MAX_RETRIES_PER_TASK,
                                        model, tf, new_id)
                        else:
                            tf_state["status"] = "failed_permanently"
                            self.state["tasks_failed_permanently"].append(f"{model}@{tf}")
                            self.state["tasks_failed"] += 1
                    else:
                        tf_state["status"] = "failed_permanently"
                        self.state["tasks_failed_permanently"].append(f"{model}@{tf}")
                        self.state["tasks_failed"] += 1
                        logger.warning("Permanent fail %s @ %s: %s", model, tf, err[:120])
            self._save_state()
            if all_done:
                break
            time.sleep(POLL_S)

        # Mark model done if any TF succeeded; else failed.
        any_done = any(s["status"] == "done" for s in m_state["tfs"].values())
        any_skip = any(s["status"] == "skipped_fresh" for s in m_state["tfs"].values())
        m_state["status"] = "done" if (any_done or any_skip) else "failed"
        self._save_state()

    def _build_task_spec(self, model: str, tf: str) -> dict:
        # OFT is per-symbol; everything else operates on ALL symbols.
        # The trainer's master_trainer wrapper handles the symbol universe.
        return {
            "model_type":   model,
            "timeframe":    tf,
            "symbol":       "ALL",
            "data_path":    "",
            "output_path":  "",
            "config":       {"use_master_trainer": True},
        }

    # ── Backtest stage ──────────────────────────────────────────────────

    def _run_backtest(self) -> None:
        bt = self.state["backtest"]
        bt["status"]      = "running"
        bt["started_at"]  = datetime.now(timezone.utc).isoformat()
        self._save_state()
        try:
            # Run backtest in this process (blocking) — the daemon's job
            # is to be the long-lived watcher, so blocking is fine here.
            from src.engine.backtester import run_full_backtest
            df = run_full_backtest(timeframes=("5m", "15m", "1h", "4h", "1d"))
            bt["status"]      = "done"
            bt["rows"]        = int(len(df)) if df is not None else 0
            bt["finished_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            logger.exception("Backtest failed: %s", exc)
            bt["status"]      = "failed"
            bt["error"]       = f"{type(exc).__name__}: {exc}"
            bt["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    # ── Control plane (Flask on :7710) ──────────────────────────────────

    def request_abort(self) -> None:
        self._abort_requested = True
        logger.info("Abort requested — will stop after current task batch")

    def request_pause(self) -> None:
        self._paused = True
        self.state["status"] = "paused"
        self._save_state()

    def request_resume(self) -> None:
        self._paused = False
        self.state["status"] = "running"
        self._save_state()


def _build_control_app(coord: SweepCoordinator):
    from flask import Flask, jsonify, request
    app = Flask("sweep_coordinator")

    @app.route("/api/sweep/status")
    def status():
        return jsonify(coord.state)

    @app.route("/api/sweep/pause", methods=["POST"])
    def pause():
        coord.request_pause()
        return jsonify({"ok": True, "status": coord.state["status"]})

    @app.route("/api/sweep/resume", methods=["POST"])
    def resume():
        coord.request_resume()
        return jsonify({"ok": True, "status": coord.state["status"]})

    @app.route("/api/sweep/abort", methods=["POST"])
    def abort():
        coord.request_abort()
        return jsonify({"ok": True, "status": "aborting"})

    return app


# ── Entrypoint ──────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not _acquire_pidfile():
        return 1
    try:
        coord = SweepCoordinator()
        # Start control plane in background thread.
        ctrl = _build_control_app(coord)
        threading.Thread(
            target=lambda: ctrl.run(host="127.0.0.1", port=CONTROL_PORT,
                                     debug=False, use_reloader=False),
            daemon=True, name="sweep-ctrl",
        ).start()
        # Auto-start the sweep — main thread blocks here until done.
        coord.run()
        return 0
    finally:
        _release_pidfile()


if __name__ == "__main__":
    sys.exit(main())
