"""master_agent — Layer 5 supervisor. The thing that should have existed
all day on 2026-05-10.

Closes the self-healing loop the Phase 88 watchdog left open:

  Phase 88 watchdog (orchestrator-side)
    detects zombie tasks (running > timeout, no recent update),
    marks them failed in cluster bookkeeping,
    frees the worker SLOT (status=idle).
  → BUT: the worker process itself keeps running its dead trainer
    thread. Heartbeat keeps reporting busy. Cluster never dispatches
    new tasks to that worker.

  master_agent (THIS module — process-side healer)
    every POLL_S seconds, scans the cluster:
    - For each ONLINE worker reporting busy + current_task:
        look up that task in cluster.
        If task status ∈ {failed, cancelled, done} → ZOMBIE WORKER.
        If task is missing entirely → PHANTOM (orchestrator restarted
        while the worker was busy; worker still holds the old ID).
    - For zombies under our control (hostname == this machine's):
        SIGKILL the worker process, spawn a fresh replacement on the
        same lane and port.
    - For zombies on remote machines (Ivan, future workers):
        emit a warning to logs + service.alerts topic.
    - Also ensures cluster_orchestrator is alive (respawns if dead).
    - Also ensures the local LOCAL_RAZER_CPU + LOCAL_RAZER_GPU lane
      workers exist.

This is the missing 'reliability' piece. With master_agent + the existing
watchdog, the cluster self-heals: zombie task → marked failed → master
agent kills the zombie worker → spawns fresh worker → sweep_coordinator
resubmits the task. No more 'why is Ivan idle?' debugging sessions.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_BASE_URL = "http://localhost:7700"

POLL_S                = 60          # how often to sweep
LOCAL_HOSTNAME        = socket.gethostname()

# How long to leave a "stale" non-busy worker alone before respawning.
# Workers can briefly disappear during heartbeat hiccups; we don't want
# to nuke them on a 5-second blip.
WORKER_OFFLINE_GRACE_S = 90

# Bug-fix 2026-05-10 — phantom-zombie detection requires the phantom
# state to persist for at least PHANTOM_CONFIRM_S before declaring
# zombie. Catches transient cases (orchestrator just restarted, fresh
# worker registering, direct /task POST that hasn't propagated to the
# cluster's task table yet). The first smoke_test of the session was
# killed at ~60s by the over-eager original detection.
PHANTOM_CONFIRM_S      = 120        # 2 cycles at POLL_S=60

# Local worker spec: (lane, port, name).
LOCAL_WORKER_SPECS = (
    ("cpu", 7701, "LOCAL_RAZER_CPU"),
    ("gpu", 7702, "LOCAL_RAZER_GPU"),
)


# ── HTTP helpers ────────────────────────────────────────────────────────

def _http_get(path: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{CLUSTER_BASE_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


# ── master_agent ────────────────────────────────────────────────────────

class MasterAgent:
    """Top-level supervisor. Runs as a separate process."""

    def __init__(self):
        self._running = False
        self._restart_counts: dict[str, int] = {}      # name → respawn count
        self._last_seen_idle: dict[str, float] = {}    # node_id → ts when first seen non-busy
        # Bug-fix 2026-05-10 (after Phase 90 commit 49dda74): the
        # original phantom detection killed any worker reporting
        # current_task that wasn't in the cluster's task table. That
        # misfires for legitimate cases:
        #   - direct /task POST (bypasses cluster's submit endpoint)
        #   - cluster orchestrator restart (in-memory task table cleared
        #     while workers still hold task IDs from before)
        # Fix: require the phantom state to persist across N consecutive
        # scan cycles before declaring zombie. Tracks first-seen-phantom
        # timestamp per node_id; only kills if observed phantom across
        # at least PHANTOM_CONFIRM_CYCLES scans.
        self._phantom_first_seen: dict[str, float] = {}

    # ── Process management primitives ───────────────────────────────────

    def _find_local_python_pids(self, name: str, lane: str) -> list[int]:
        """Find local python.exe PIDs running a worker with the given
        --name and --lane. Uses Win32_Process via wmic-like API."""
        pids: list[int] = []
        try:
            import psutil
        except ImportError:
            return pids
        try:
            for p in psutil.process_iter(["pid", "name", "cmdline"]):
                if not (p.info.get("name") or "").lower().startswith("python"):
                    continue
                cmd = " ".join(p.info.get("cmdline") or [])
                if "distributed.worker" in cmd and f"--name {name}" in cmd and f"--lane {lane}" in cmd:
                    pids.append(int(p.info["pid"]))
        except Exception:
            pass
        return pids

    def _kill_pids(self, pids: list[int]) -> None:
        try:
            import psutil
        except ImportError:
            return
        for pid in pids:
            try:
                p = psutil.Process(pid)
                p.kill()
                logger.warning("[master_agent] SIGKILL pid=%d (zombie cleanup)", pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def _spawn_local_worker(self, lane: str, port: int, name: str) -> Optional[int]:
        """Spawn one local worker. Same logic as sweep_coordinator —
        kept here so master_agent can independently respawn if
        sweep_coordinator isn't running yet."""
        venv_py = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
        if not venv_py.exists():
            logger.warning("venv python not found — skipping spawn of %s", name)
            return None
        cmd = [str(venv_py), "-u", "-m", "src.training.distributed.worker",
               "--master", CLUSTER_BASE_URL,
               "--name", name,
               "--lane", lane,
               "--port", str(port)]
        env = os.environ.copy()
        env["PYTHONPATH"]      = str(PROJECT_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        if lane == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        log_out = open(PROJECT_ROOT / "logs" / f"local_worker_{lane}.log", "a", encoding="utf-8")
        log_err = open(PROJECT_ROOT / "logs" / f"local_worker_{lane}.err.log", "a", encoding="utf-8")
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                    stdout=log_out, stderr=log_err, creationflags=flags)
        finally:
            # Parent process no longer needs these handles; the child inherits its own copies.
            log_out.close()
            log_err.close()
        self._restart_counts[name] = self._restart_counts.get(name, 0) + 1
        logger.info("[master_agent] Spawned %s (lane=%s, port=%d) pid=%d (respawn #%d)",
                    name, lane, port, proc.pid, self._restart_counts[name])
        return proc.pid

    # ── Health checks ────────────────────────────────────────────────────

    def _cluster_orchestrator_alive(self) -> bool:
        """True iff :7700 responds AND a python process for it exists."""
        if _http_get("/api/cluster/status", timeout=3) is None:
            return False
        try:
            from src.utils import process_health as ph
            return ph.find_process(ph.KIND_CLUSTER_ORCH) is not None
        except Exception:
            return True   # don't kill if we can't tell

    def _ensure_cluster_orchestrator(self) -> None:
        if self._cluster_orchestrator_alive():
            return
        logger.warning("[master_agent] cluster_orchestrator DEAD — respawning")
        venv_py = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
        if not venv_py.exists():
            return
        cmd = [str(venv_py), "-u", "-m", "src.training.distributed.orchestrator"]
        env = os.environ.copy()
        env["PYTHONPATH"]      = str(PROJECT_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        log_out = open(PROJECT_ROOT / "logs" / "orchestrator.log", "a", encoding="utf-8")
        log_err = open(PROJECT_ROOT / "logs" / "orchestrator.err.log", "a", encoding="utf-8")
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                         stdout=log_out, stderr=log_err, creationflags=flags)
        time.sleep(5)

    def _ensure_local_workers(self) -> None:
        """Make sure both local lane workers exist. Skip if a worker
        is already registered (sweep_coordinator may have spawned them
        first — first-writer wins, no double spawn)."""
        st = _http_get("/api/cluster/status")
        if st is None:
            return
        registered_local = {
            (w.get("name", ""), w.get("lane", ""))
            for w in st.get("workers", [])
            if w.get("hostname", "") == LOCAL_HOSTNAME and w.get("online")
        }
        for lane, port, name in LOCAL_WORKER_SPECS:
            if (name, lane) in registered_local:
                continue
            # Check there isn't already a process running for this lane
            # (process exists but cluster registration lagging).
            if self._find_local_python_pids(name, lane):
                continue
            self._spawn_local_worker(lane, port, name)

    def _sweep_zombie_workers(self) -> None:
        """Detect + heal zombie workers. Z2's two flavours:
          PHANTOM: worker reports busy with task_id that doesn't exist
                   in the cluster's task table (orchestrator was
                   restarted while worker was busy).
          DEAD-TASK: worker reports busy with task_id whose status is
                     in {failed, cancelled, done}.
        Either case → kill local worker + respawn; remote → log alert.
        """
        st = _http_get("/api/cluster/status")
        if st is None:
            return
        tasks = _http_get("/api/cluster/tasks")
        if tasks is None or not isinstance(tasks, list):
            return
        task_lookup = {t.get("task_id"): t for t in tasks}
        now = time.time()
        seen_phantom_this_cycle: set[str] = set()
        for worker in st.get("workers", []):
            if not worker.get("online"):
                continue
            if worker.get("status") != "busy":
                continue
            tid = worker.get("current_task", "")
            if not tid:
                continue
            node_id = worker.get("node_id", "")
            task = task_lookup.get(tid)
            if task is None:
                # PHANTOM — task_id not in cluster table. Could be
                # transient (orchestrator just restarted / direct /task
                # POST). Require state to persist for PHANTOM_CONFIRM_S
                # before killing.
                seen_phantom_this_cycle.add(node_id)
                first_seen = self._phantom_first_seen.setdefault(node_id, now)
                if now - first_seen >= PHANTOM_CONFIRM_S:
                    self._heal_zombie(worker, reason="phantom_task_id_persisted",
                                       task_id=tid)
                else:
                    logger.info("[master_agent] %s phantom task_id=%s observed for %ds "
                                "(< %ds confirm window) — waiting",
                                worker.get("name"), tid[:11],
                                int(now - first_seen), PHANTOM_CONFIRM_S)
            else:
                tstatus = task.get("status", "")
                if tstatus in ("failed", "cancelled", "done"):
                    # Dead-task zombies are unambiguous — kill immediately.
                    self._heal_zombie(worker, reason=f"task_status={tstatus}", task_id=tid)
        # Reset the phantom timer for any node that's no longer phantom
        # (task showed up in the cluster table, or worker became idle).
        for nid in list(self._phantom_first_seen.keys()):
            if nid not in seen_phantom_this_cycle:
                del self._phantom_first_seen[nid]

    def _heal_zombie(self, worker: dict, reason: str, task_id: str) -> None:
        host = worker.get("hostname", "")
        name = worker.get("name", "")
        lane = worker.get("lane", "")
        node_id = worker.get("node_id", "")
        if host == LOCAL_HOSTNAME:
            # Local: kill + respawn
            pids = self._find_local_python_pids(name, lane)
            logger.warning("[master_agent] LOCAL ZOMBIE: %s (%s) reason=%s task_id=%s "
                           "→ killing pids=%s + respawn",
                           name, node_id, reason, task_id[:11], pids)
            self._kill_pids(pids)
            time.sleep(2)
            # Find port for this lane spec.
            for spec_lane, spec_port, spec_name in LOCAL_WORKER_SPECS:
                if spec_name == name and spec_lane == lane:
                    self._spawn_local_worker(lane, spec_port, name)
                    break
        else:
            # Phase 93 — remote zombies can now self-heal via the
            # worker's /restart endpoint. POSTing here causes the worker
            # process to os.execv itself in 1 s; uptime resets, the
            # dead trainer thread is gone, and the next heartbeat
            # registers the fresh process. Pre-Phase-93 the only
            # remediation was a logged alert + manual SSH/RDP.
            ip   = worker.get("ip", "")
            port = worker.get("port", 0)
            triggered = False
            if ip and port:
                try:
                    # confirm=True is required by the worker's /restart
                    # gate (added after the 2026-05-10 accidental restart
                    # incident on Ivan).
                    body = json.dumps({
                        "confirm": True,
                        "reason":  reason,
                        "task_id": task_id,
                    }).encode("utf-8")
                    req  = urllib.request.Request(
                        f"http://{ip}:{port}/restart",
                        data=body,
                        method="POST",
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=5) as r:
                        triggered = (r.status == 200)
                except (urllib.error.URLError, OSError) as exc:
                    logger.warning("[master_agent] /restart POST to %s:%d failed: %s",
                                   ip, port, exc)
            if triggered:
                logger.warning("[master_agent] REMOTE RESTART: %s (host=%s, %s:%d) "
                               "reason=%s task_id=%s — worker will re-exec in 1s",
                               name, host, ip, port, reason, task_id[:11])
            else:
                logger.error("[master_agent] REMOTE ZOMBIE on host=%s (%s) lane=%s "
                             "reason=%s task_id=%s — /restart unavailable, "
                             "operator must restart this worker manually.",
                             host, name, lane, reason, task_id[:11])
            # Always emit the service.alerts topic entry so the
            # dashboard can show what happened (auto-healed vs needs
            # human).
            try:
                from src.orchestration.topics import topic, TOPIC_SERVICE_ALERTS
                topic(TOPIC_SERVICE_ALERTS).append({
                    "kind":      "remote_zombie_worker",
                    "host":      host,
                    "name":      name,
                    "lane":      lane,
                    "node_id":   node_id,
                    "task_id":   task_id,
                    "reason":    reason,
                    "auto_healed": triggered,
                    "needs":     "self_restart_via_endpoint" if triggered
                                 else "operator restart on remote machine",
                })
            except Exception:
                pass

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("=== master_agent START host=%s poll=%ds ===", LOCAL_HOSTNAME, POLL_S)
        self._running = True
        while self._running:
            try:
                self._ensure_cluster_orchestrator()
                self._ensure_local_workers()
                self._sweep_zombie_workers()
            except Exception:
                logger.exception("[master_agent] loop iteration error")
            time.sleep(POLL_S)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    agent = MasterAgent()
    agent.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
