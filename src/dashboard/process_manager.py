"""Process management for the dashboard's Monitor tab.

Provides:
  - ProcessManager class with list/kill/start methods
  - Background health-check loop (60s) with optional auto-kill on bad health
  - 13 known roles with launcher cmds + health-check strategy

Why a separate module from src/utils/process_registry.py:
  - process_registry tracks "who claimed which role" (singleton enforcement).
  - process_manager is the ACTION layer — knows how to KILL and START processes.
  - The role->launcher map lives here because only the start path needs it.

The dashboard's /api/processes/* endpoints proxy directly into this module.
The Monitor tab's "Processes" card polls /api/processes/list every 30s.

Auto-kill is OFF by default. Set AUTO_KILL_BAD_HEALTH=true to enable.
Even when enabled, only kills after 3 consecutive bad health checks to
avoid cascade-killing during a brief network blip.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
LOGS_DIR = PROJECT_ROOT / "logs"

# Health states the UI renders.
HEALTH_OK = "ok"           # green
HEALTH_STALE = "stale"     # yellow (no recent activity but PID alive)
HEALTH_DEAD = "dead"       # red (PID gone)
HEALTH_UNKNOWN = "unknown" # grey (never checked yet)


@dataclass
class RoleSpec:
    """One process role the operator can start/kill from the dashboard."""
    key: str
    label: str
    cmd: list[str]                              # subprocess.Popen argv
    log_file: str | None = None                 # logs/<file>.log for activity check
    http_health: str | None = None              # URL to GET for 'ok' check, e.g. "http://127.0.0.1:5000/api/monitor/health"
    health_kind: str = "pid+log"                # "http" | "pid+log" | "pid-only"
    log_stale_s: int = 300                      # >this many seconds without log writes -> stale


# Canonical role definitions. The order here drives the order in the UI table.
# Commands match restart_all.ps1's launcher blocks 1:1 so a "start" from the
# dashboard is indistinguishable from a fresh restart_all.ps1 launch.
def _python_module(mod: str, *extra: str) -> list[str]:
    return [str(VENV_PYTHON), "-m", mod, *extra]


def _python_script(rel: str) -> list[str]:
    return [str(VENV_PYTHON), str(PROJECT_ROOT / rel)]


ROLE_SPECS: dict[str, RoleSpec] = {
    "monitor": RoleSpec(
        key="monitor", label="Monitor (:5001)",
        cmd=_python_script("src/monitor/server.py"),
        log_file="monitor.log",
        # Monitor exposes only "/" and "/logs/<component>"; "/" is the
        # liveness probe — returns 200 with the HTML index when alive.
        http_health="http://127.0.0.1:5001/",
        health_kind="http",
    ),
    "dashboard": RoleSpec(
        key="dashboard", label="Dashboard (:5000)",
        cmd=_python_script("src/dashboard/app.py"),
        log_file="dashboard.log",
        # Cannot use http_health here — the dashboard would self-probe its
        # own port from the background loop thread, blocking a Flask worker
        # while waiting for the response. pid+log on dashboard.log is fine:
        # Werkzeug appends an access-log line to dashboard.log on every
        # request, so a stale dashboard.log mtime is a reliable "not serving"
        # signal.
        health_kind="pid+log",
    ),
    "bot": RoleSpec(
        key="bot", label="Bot (trading engine)",
        cmd=_python_script("src/main.py"),
        log_file="bot.log",
        health_kind="pid+log",
    ),
    "cluster_orch": RoleSpec(
        key="cluster_orch", label="Cluster Orchestrator (:7700)",
        cmd=_python_module("src.training.distributed.orchestrator", "--port", "7700"),
        log_file="cluster.log",
        # The orchestrator exposes the api routes under /api/cluster/.
        # /api/cluster/status is the canonical liveness probe.
        http_health="http://127.0.0.1:7700/api/cluster/status",
        health_kind="http",
    ),
    "realtime": RoleSpec(
        key="realtime", label="Realtime DB Writer",
        cmd=_python_module("src.data_ingestion.realtime_db_writer"),
        log_file="realtime_db.log",
        health_kind="pid+log",
    ),
    "data_orch": RoleSpec(
        key="data_orch", label="Data Governance Orchestrator",
        cmd=_python_module("src.data_governance.orchestrator"),
        log_file="data_orchestrator.log",
        health_kind="pid+log",
    ),
    "orderbook_collector": RoleSpec(
        key="orderbook_collector", label="L2 Orderbook Collector",
        cmd=_python_module("src.data_ingestion.orderbook_collector",
                           "--symbols", "BTC/USDT,ETH/USDT,SOL/USDT",
                           "--depth", "20", "--speed", "100ms"),
        log_file="orderbook_collector.log",
        health_kind="pid+log",
    ),
    "orderbook_writer": RoleSpec(
        key="orderbook_writer", label="L2 Orderbook Parquet Writer",
        cmd=_python_module("src.data_ingestion.orderbook_parquet_writer"),
        log_file="orderbook_parquet_writer.log",
        health_kind="pid+log",
    ),
    "watchlist": RoleSpec(
        key="watchlist", label="Watchlist Downloader",
        cmd=_python_module("src.data_ingestion.watchlist_downloader"),
        log_file="watchlist_downloader.log",
        health_kind="pid+log",
    ),
    "debug_supervisor": RoleSpec(
        key="debug_supervisor", label="Debug Supervisor",
        cmd=_python_module("scripts.debug_supervisor"),
        log_file="debug_supervisor.log",
        health_kind="pid-only",
    ),
    "dashboard_watchdog": RoleSpec(
        key="dashboard_watchdog", label="Dashboard Watchdog",
        cmd=_python_module("scripts.dashboard_watchdog"),
        log_file="dashboard_watchdog.log",
        health_kind="pid-only",
    ),
    "sweep_watchdog": RoleSpec(
        key="sweep_watchdog", label="Training Sweep Watchdog",
        cmd=_python_module("scripts.training_sweep_watchdog"),
        log_file="training_sweep_watchdog.log",
        health_kind="pid-only",
    ),
}


@dataclass
class HealthSnapshot:
    role: str
    pid: int | None = None
    status: str = HEALTH_UNKNOWN
    last_health_ts: float = 0.0
    last_log_mtime: float = 0.0
    uptime_s: int = 0
    bad_count: int = 0           # consecutive bad checks (for auto-kill)
    last_error: str | None = None


# Process discovery: scan psutil for a cmdline matching the role's spec.
def _find_role_pid(role: RoleSpec) -> int | None:
    """Return the PID of the running process for `role`, or None.

    Phase F (2026-05-14): support BOTH launch styles regardless of which
    one the spec uses. ``restart_all.ps1`` launches ``python src/main.py``
    (script-style); a manual operator restart often uses ``python -m src.main``
    (module-style). Without this dual-form matching, the live process is
    invisible to the Monitor table -> shows "dead" with no PID, and the
    Kill / Restart / Start buttons either no-op or spawn a duplicate.
    """
    try:
        import psutil
    except ImportError:
        return None
    needle_module: str | None = None
    needle_path_tail: str | None = None
    # Spec.cmd is a list. The discriminator depends on launch style.
    if len(role.cmd) >= 3 and role.cmd[1] == "-m":
        needle_module = role.cmd[2]
    elif len(role.cmd) >= 2:
        # For script-style launches, derive the project-relative path so
        # the match works regardless of whether the launch used absolute
        # or relative paths. `_python_script("src/dashboard/app.py")` and
        # `D:\proj\src\dashboard\app.py` both yield the tail `src/dashboard/app.py`.
        tail = Path(role.cmd[1]).as_posix()
        if "/src/" in tail:
            needle_path_tail = "src/" + tail.split("/src/", 1)[1]
        elif tail.startswith("src/"):
            needle_path_tail = tail
        else:
            parts = tail.split("/")
            needle_path_tail = "/".join(parts[-2:]) if len(parts) >= 2 else tail
    # Always also build the OTHER form so we match both launch styles.
    # module -> path tail
    if needle_module and not needle_path_tail:
        # "src.dashboard.app" -> "src/dashboard/app.py"
        # "src.main"          -> "src/main.py"
        needle_path_tail = needle_module.replace(".", "/") + ".py"
    # path tail -> module
    if needle_path_tail and not needle_module:
        # "src/dashboard/app.py" -> "src.dashboard.app"
        # "src/main.py"          -> "src.main"
        # "dashboard/app.py"     -> "src.dashboard.app" (project assumption)
        stem = needle_path_tail
        if stem.endswith(".py"):
            stem = stem[:-3]
        as_mod = stem.replace("/", ".")
        if not as_mod.startswith("src."):
            as_mod = "src." + as_mod
        needle_module = as_mod
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            # Normalize the full cmdline string with forward slashes for
            # OS-agnostic matching.
            joined = " ".join(cmdline).replace("\\", "/")
            if "python" not in (proc.info.get("name") or "").lower():
                continue
            if needle_module:
                # `-m src.main` lands as `... -m src.main` (with possible
                # trailing args). The endswith check covers the trailing-
                # arg case; the substring covers everything else.
                if (f"-m {needle_module}" in joined
                        or (f" {needle_module} " in joined and " -m " in joined)
                        or (joined.rstrip().endswith(f" {needle_module}") and " -m " in joined)):
                    return int(proc.info["pid"])
            if needle_path_tail and needle_path_tail in joined:
                return int(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def _http_health_ok(url: str, timeout: float = 2.0) -> bool:
    """GET the URL; return True iff response code < 400."""
    api_key = os.environ.get("DASHBOARD_API_KEY", "")
    req = urllib.request.Request(url, headers={"X-API-Key": api_key} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        p = psutil.Process(int(pid))
        return p.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except Exception:
        return False


def _uptime_s(pid: int | None) -> int:
    if not pid:
        return 0
    try:
        import psutil
        p = psutil.Process(int(pid))
        return max(0, int(time.time() - p.create_time()))
    except Exception:
        return 0


def _log_mtime(log_file: str | None) -> float:
    if not log_file:
        return 0.0
    p = LOGS_DIR / log_file
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


class ProcessManager:
    """Singleton process control surface for the dashboard."""

    def __init__(self) -> None:
        self._snapshots: dict[str, HealthSnapshot] = {
            k: HealthSnapshot(role=k) for k in ROLE_SPECS
        }
        self._lock = threading.Lock()
        self._loop_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── public API ────────────────────────────────────────────────────────

    def list(self) -> list[dict[str, Any]]:
        """Return one row per known role for the UI table."""
        with self._lock:
            snaps = {k: v for k, v in self._snapshots.items()}
        out: list[dict[str, Any]] = []
        for key, spec in ROLE_SPECS.items():
            snap = snaps.get(key) or HealthSnapshot(role=key)
            out.append({
                "role": key,
                "label": spec.label,
                "pid": snap.pid,
                "status": snap.status,
                "last_health_ts": snap.last_health_ts,
                "last_log_mtime": snap.last_log_mtime,
                "uptime_s": snap.uptime_s,
                "log_file": spec.log_file,
                "health_kind": spec.health_kind,
                "last_error": snap.last_error,
                "bad_count": snap.bad_count,
            })
        return out

    def refresh_one(self, role_key: str) -> HealthSnapshot:
        """Recompute the health snapshot for `role_key` immediately.
        Used by the health loop and by /api/processes/list to give the
        operator fresh data on every poll."""
        spec = ROLE_SPECS.get(role_key)
        if not spec:
            return HealthSnapshot(role=role_key, status=HEALTH_DEAD,
                                  last_error="unknown role")
        pid = _find_role_pid(spec)
        log_mtime = _log_mtime(spec.log_file)
        uptime = _uptime_s(pid)
        now = time.time()
        status = HEALTH_UNKNOWN
        last_error: str | None = None

        if spec.health_kind == "http":
            # HTTP-exposed: ping the URL. If down but PID alive, mark stale.
            if spec.http_health and _http_health_ok(spec.http_health):
                status = HEALTH_OK
            elif _pid_alive(pid):
                status = HEALTH_STALE
                last_error = f"HTTP health failed at {spec.http_health}"
            else:
                status = HEALTH_DEAD
                last_error = "PID not found and HTTP health failed"
        elif spec.health_kind == "pid+log":
            if not _pid_alive(pid):
                status = HEALTH_DEAD
                last_error = "PID not found"
            elif log_mtime and (now - log_mtime) > spec.log_stale_s:
                status = HEALTH_STALE
                last_error = (f"log {spec.log_file} not written in "
                              f"{int(now - log_mtime)}s (stale threshold "
                              f"{spec.log_stale_s}s)")
            else:
                status = HEALTH_OK
        else:  # pid-only
            status = HEALTH_OK if _pid_alive(pid) else HEALTH_DEAD
            if status == HEALTH_DEAD:
                last_error = "PID not found"

        with self._lock:
            snap = self._snapshots.setdefault(role_key, HealthSnapshot(role=role_key))
            snap.pid = pid
            snap.last_health_ts = now
            snap.last_log_mtime = log_mtime
            snap.uptime_s = uptime
            snap.last_error = last_error
            # bad_count tracking for auto-kill threshold
            if status in (HEALTH_DEAD, HEALTH_STALE):
                snap.bad_count += 1
            else:
                snap.bad_count = 0
            snap.status = status
            return HealthSnapshot(**snap.__dict__)  # return a copy

    def refresh_all(self) -> None:
        for key in ROLE_SPECS:
            self.refresh_one(key)

    def kill(self, pid: int) -> dict[str, Any]:
        """Kill a process by PID. Recursive — terminates children first.
        Returns {ok: bool, message: str, killed_pid: int|None}."""
        try:
            import psutil
        except ImportError:
            return {"ok": False, "error": "psutil unavailable"}
        try:
            p = psutil.Process(int(pid))
        except psutil.NoSuchProcess:
            return {"ok": True, "message": "PID not found (already gone)",
                    "killed_pid": None}
        try:
            for c in p.children(recursive=True):
                try:
                    c.kill()
                except Exception:
                    pass
            p.kill()
            try:
                p.wait(timeout=3)
            except Exception:
                pass
            return {"ok": True, "message": "killed", "killed_pid": int(pid)}
        except Exception as e:
            return {"ok": False, "error": f"kill failed: {e}"}

    def start(self, role_key: str) -> dict[str, Any]:
        """Spawn the role's launcher. Refuses if already alive (use kill first)."""
        spec = ROLE_SPECS.get(role_key)
        if not spec:
            return {"ok": False, "error": f"unknown role: {role_key}"}
        existing_pid = _find_role_pid(spec)
        if existing_pid and _pid_alive(existing_pid):
            return {"ok": False, "error": f"role {role_key} already alive (PID {existing_pid})",
                    "existing_pid": existing_pid}
        # Spawn detached. On Windows, DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP
        # ensures the child outlives the dashboard's parent process. Logs go
        # to the role's log file in logs/.
        log_path = LOGS_DIR / (spec.log_file or f"{role_key}.log")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            log_fp = open(log_path, "ab")
        except OSError as e:
            return {"ok": False, "error": f"could not open log file: {e}"}
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = (subprocess.DETACHED_PROCESS
                                 | subprocess.CREATE_NEW_PROCESS_GROUP)
            proc = subprocess.Popen(
                spec.cmd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                creationflags=creationflags,
                close_fds=False,
            )
        except Exception as e:
            try:
                log_fp.close()
            except Exception:
                pass
            return {"ok": False, "error": f"spawn failed: {e}"}
        # Don't wait — process is detached. Give it a brief moment to start
        # before we report a PID.
        time.sleep(0.5)
        return {
            "ok": True,
            "message": f"started role={role_key}",
            "pid": proc.pid,
            "log_file": str(log_path),
        }

    # ── background loop ───────────────────────────────────────────────────

    def start_health_loop(self, interval_s: int = 60) -> None:
        """Start the background health-check thread. Idempotent."""
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._stop.clear()
        self._loop_thread = threading.Thread(
            target=self._loop, args=(interval_s,),
            name="ProcessManager-HealthLoop", daemon=True,
        )
        self._loop_thread.start()

    def stop_health_loop(self) -> None:
        self._stop.set()

    def _loop(self, interval_s: int) -> None:
        bad_threshold = 3  # consecutive bad checks before auto-kill
        while not self._stop.is_set():
            # Re-read AUTO_KILL_BAD_HEALTH on every tick so the dashboard's
            # POST /api/processes/auto_kill toggle takes effect without
            # needing a process_manager restart. Before this, the loop
            # cached the boolean at thread start, so a runtime toggle was
            # a no-op until the next dashboard restart (operator-visible
            # bug 2026-05-14).
            auto_kill = (os.environ.get("AUTO_KILL_BAD_HEALTH", "false").lower()
                         in ("1", "true", "yes"))
            try:
                for key in ROLE_SPECS:
                    snap = self.refresh_one(key)
                    if (auto_kill and snap.bad_count >= bad_threshold
                            and snap.pid and snap.status != HEALTH_OK):
                        logger.critical(
                            "[process_manager] AUTO_KILL: role=%s pid=%s "
                            "bad_count=%d status=%s -- killing.",
                            snap.role, snap.pid, snap.bad_count, snap.status,
                        )
                        self.kill(snap.pid)
                        with self._lock:
                            self._snapshots[key].bad_count = 0
            except Exception as e:
                logger.warning("[process_manager] loop iteration failed: %s", e)
            # Sleep in 1s slices so stop_health_loop() is responsive.
            for _ in range(int(interval_s)):
                if self._stop.is_set():
                    return
                time.sleep(1.0)


# Module-level singleton — the dashboard wires this into Flask endpoints.
_manager: ProcessManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> ProcessManager:
    """Lazy singleton accessor used by the Flask routes."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ProcessManager()
        return _manager


__all__ = [
    "ProcessManager", "RoleSpec", "ROLE_SPECS", "HealthSnapshot",
    "HEALTH_OK", "HEALTH_STALE", "HEALTH_DEAD", "HEALTH_UNKNOWN",
    "get_manager",
]
