r"""Centralised process detection for all dashboard probes, supervisors,
and watchdogs.

Single source of truth for "is service X alive, on which PID, with what
RSS". Replaces the four duplicated cmdline-scan implementations that
caused the Component Health DEAD/Stopped false alarms in PR-32 and the
2026-05-09 incident.

Why this module exists
----------------------
Before this module, four places independently scanned `psutil` for python
processes by regex:

  1. src/dashboard/app.py::monitor_health           (Component Health card)
  2. src/dashboard/app.py::_pipeline_proc_alive     (training pipeline status)
  3. src/dashboard/error_monitor.py::_probe_processes (banner)
  4. scripts/training_sweep_watchdog.py             (sweep self-heal)

Each had its own regex, its own wrapper-vs-worker tie-breaker (or none),
its own fallback policy. When the bot launch style changed from
`python src/main.py` to `python -m src.main` (Start-Process invocation),
three of the four broke independently and at different times. Since
Phase 78 only tested error_monitor's copy, the dashboard and watchdog
copies regressed silently.

Contract
--------
- `find_process(kind)` returns the highest-RSS python process matching
  the kind's pattern, or None. Highest RSS reliably picks the real worker
  over the dormant Start-Process wrapper (worker ~100+ MB, wrapper ~1 MB).
- Both launch styles supported per kind:
    script-style:  ``python src/main.py``        → matches ``src[\\/]main\.py``
    module-style:  ``python -m src.main``        → matches ``-m\s+src\.main\b``
- `all_known_processes()` does ONE psutil scan and dispatches to all kinds
  (cheaper than N independent scans when the dashboard renders the full
  Component Health card).
- Pure read — never kills, never spawns. Supervisors do that.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# 2026-05-11 — TTL cache for the psutil scan. On Windows, iterating every
# python process and calling memory_info() per process is expensive
# (~150-300ms per process × ~20 procs = 3-6s cold). The dashboard polls
# /api/pipeline/status, /api/monitor/health, /api/state and friends every
# 5-10s, all sharing this code path. Without a cache the dashboard pegs
# itself; with a 10s TTL most polls cost ~0ms.
# Override via PROCESS_HEALTH_TTL_S env var (set 0 to disable).
_SCAN_TTL_S = float(os.environ.get("PROCESS_HEALTH_TTL_S", "10"))
_scan_lock  = threading.Lock()
_scan_cache: tuple[float, list[tuple[int, str, int]]] = (0.0, [])

# ── Service kinds (canonical names — used as keys everywhere) ──────────
KIND_BOT              = "bot"
KIND_DASH             = "dash"
KIND_TRAIN_ORCH       = "training_orchestrator"   # src.engine.pipeline_orchestrator
KIND_CLUSTER_ORCH     = "cluster_orchestrator"    # src.training.distributed.orchestrator
KIND_WORKER           = "worker"                  # src.training.distributed.worker
KIND_TRAIN_SUPERVISOR = "training_supervisor"     # src.orchestration.training_supervisor (Layer 3)
KIND_BT_SUPERVISOR    = "backtest_supervisor"     # src.orchestration.backtest_supervisor (Layer 4)
KIND_MASTER_AGENT     = "master_agent"            # src.orchestration.master_agent (Layer 5)

ALL_KINDS = (
    KIND_BOT, KIND_DASH, KIND_TRAIN_ORCH, KIND_CLUSTER_ORCH, KIND_WORKER,
    KIND_TRAIN_SUPERVISOR, KIND_BT_SUPERVISOR, KIND_MASTER_AGENT,
)

# Pattern per kind. Each pattern handles BOTH launch styles via
# alternation. Anchored with `\b` where ambiguous so `src.main` doesn't
# match `src.maintenance`.
_PATTERNS: dict[str, re.Pattern[str]] = {
    KIND_BOT:              re.compile(r"src[\\/]main\.py|-m\s+src\.main\b"),
    KIND_DASH:             re.compile(r"src[\\/]dashboard[\\/]app\.py|-m\s+src\.dashboard\.app\b"),
    KIND_TRAIN_ORCH:       re.compile(r"src[\\/]engine[\\/]pipeline_orchestrator\.py|-m\s+src\.engine\.pipeline_orchestrator\b|train_all_models\.py|pipeline_orchestrator"),
    KIND_CLUSTER_ORCH:     re.compile(r"-m\s+src\.training\.distributed\.orchestrator\b"),
    KIND_WORKER:           re.compile(r"-m\s+src\.training\.distributed\.worker\b"),
    KIND_TRAIN_SUPERVISOR: re.compile(r"-m\s+src\.orchestration\.training_supervisor\b"),
    KIND_BT_SUPERVISOR:    re.compile(r"-m\s+src\.orchestration\.backtest_supervisor\b"),
    KIND_MASTER_AGENT:     re.compile(r"-m\s+src\.orchestration\.master_agent\b"),
}


@dataclass(frozen=True)
class ProcessInfo:
    """Snapshot of a matched process at scan time. RSS in bytes."""
    pid: int
    rss_bytes: int
    cmdline: str

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 * 1024)


def _snapshot_python_procs() -> list[tuple[int, str, int]]:
    """Return a cached (PID, cmdline, rss) snapshot of running python procs.
    The underlying psutil scan refreshes only when older than _SCAN_TTL_S.
    Eliminates redundant scans inside the same dashboard poll window."""
    global _scan_cache
    if _SCAN_TTL_S > 0:
        now = time.monotonic()
        with _scan_lock:
            ts, cached = _scan_cache
            if now - ts < _SCAN_TTL_S:
                return cached
    fresh = list(_iter_python_procs())
    if _SCAN_TTL_S > 0:
        with _scan_lock:
            _scan_cache = (time.monotonic(), fresh)
    return fresh


def invalidate_scan_cache() -> None:
    """Drop the cached snapshot. Use after a process is known to have
    started or died (e.g. immediately after taskkill) so the next
    find_process()/all_known_processes() returns fresh data."""
    global _scan_cache
    with _scan_lock:
        _scan_cache = (0.0, [])


def _iter_python_procs():
    """Yield (pid, cmdline_str, rss_bytes) for every running python.exe.
    Single-pass — callers should iterate this once and dispatch, not call
    it per-kind."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return
    try:
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = (p.info.get("name") or "").lower()
                if not name.startswith("python"):
                    continue
                cmdline_list = p.info.get("cmdline") or []
                cmdline = " ".join(cmdline_list)
                try:
                    rss = p.memory_info().rss
                except Exception:
                    rss = 0
                yield int(p.info["pid"]), cmdline, rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as exc:
        logger.debug("psutil scan failed: %s", exc)


def find_process(kind: str) -> Optional[ProcessInfo]:
    """Return the highest-RSS python process matching `kind`, or None.

    `kind` must be one of the KIND_* constants. Higher-RSS wins so the
    real worker beats the dormant Start-Process wrapper.
    """
    pat = _PATTERNS.get(kind)
    if pat is None:
        raise ValueError(f"Unknown process kind: {kind!r}")
    best: Optional[ProcessInfo] = None
    for pid, cmdline, rss in _snapshot_python_procs():
        if pat.search(cmdline):
            if best is None or rss > best.rss_bytes:
                best = ProcessInfo(pid=pid, rss_bytes=rss, cmdline=cmdline)
    return best


def all_known_processes() -> dict[str, Optional[ProcessInfo]]:
    """Single-pass psutil scan + dispatch to all known kinds.

    Cheaper than N calls to find_process when the caller wants the full
    fleet snapshot (e.g. Component Health card render). Per-kind result
    is the highest-RSS match, or None.
    """
    best: dict[str, Optional[ProcessInfo]] = {k: None for k in _PATTERNS}
    for pid, cmdline, rss in _snapshot_python_procs():
        for kind, pat in _PATTERNS.items():
            if pat.search(cmdline):
                cur = best[kind]
                if cur is None or rss > cur.rss_bytes:
                    best[kind] = ProcessInfo(pid=pid, rss_bytes=rss, cmdline=cmdline)
    return best


def is_alive(pid: int) -> bool:
    """Cheap pid-exists check. Doesn't validate cmdline — for that, use
    find_process(kind) which combines liveness + identity in one step.
    """
    if not pid or pid < 1:
        return False
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(int(pid))
    except ImportError:
        return False
    except Exception:
        return False


def proc_stats(pid: int) -> dict:
    """RSS + uptime + CPU% sample for a known PID. Returns zeros on
    error so callers don't have to wrap. CPU% uses interval=None
    (non-blocking, returns 0.0 on first call per PID — this is the
    same trade-off the original _proc_stats helper made)."""
    try:
        import psutil  # type: ignore
        import time
        p = psutil.Process(int(pid))
        with p.oneshot():
            mem = p.memory_info().rss
            create = p.create_time()
            cpu = p.cpu_percent(interval=None)
        return {
            "cpu": round(float(cpu), 1),
            "mem_mb": round(mem / (1024 * 1024), 1),
            "uptime_s": int(time.time() - create),
        }
    except Exception:
        return {"cpu": 0, "mem_mb": 0, "uptime_s": 0}
