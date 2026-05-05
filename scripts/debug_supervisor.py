"""
debug_supervisor.py — fine-grained crash detector for project python processes.

Runs as its own python process (started by restart_all.ps1 at boot). Polls
data/process_ids.json every POLL_SECONDS and detects when a tracked PID
goes from alive→dead. On death:
  1. captures the last 100 lines of the role's log file (last words)
  2. writes a crash record to data/process_deaths.json (capped 200 entries)
  3. logs a one-line summary to logs/debug_supervisor.log
  4. samples RSS/CPU% of every tracked process so death-context is available

Independent of bot/dashboard so it survives THEIR crashes — that's the
whole point. Strictly project-scoped: never touches any other process.

Surface:
  - data/process_deaths.json — most-recent-first list of death records
  - logs/debug_supervisor.log — append-only summary log

Both are read by the dashboard's /api/debug/deaths endpoint and the
banner-aggregator's _probe_recent_deaths probe (next phase).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("debug_supervisor")

POLL_SECONDS = 5.0
DEATHS_PATH  = PROJECT_ROOT / "data" / "process_deaths.json"
LOG_PATH     = PROJECT_ROOT / "logs" / "debug_supervisor.log"
PIDS_PATH    = PROJECT_ROOT / "data" / "process_ids.json"
MAX_DEATHS   = 200

# Map role → log file (relative to PROJECT_ROOT). Every role we want death
# diagnostics for needs an entry here. Roles not listed still get their
# death recorded — just with empty log_tail.
_ROLE_LOG_FILES: dict[str, str] = {
    "bot":       "logs/bot.log",
    "dash":      "logs/dashboard.log",
    "realtime":  "logs/realtime_db.log",
    "orderbook": "logs/orderbook_collector.log",
    "training":  "logs/training_supervisor.log",
    "monitor":   "logs/monitor.log",
    "orch":      "logs/data_orchestrator.log",
    "fastapi":   "logs/fastapi.log",
}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _read_pid_map() -> dict[str, int]:
    """Read process_ids.json. Some entries are list[int] (multi-PID roles
    like fastapi, orderbook). Returns role→primary-PID for tracked roles."""
    out: dict[str, int] = {}
    try:
        if not PIDS_PATH.exists():
            return out
        with open(PIDS_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as exc:
        logger.debug("read pid file: %s", exc)
        return out
    for k, v in (d or {}).items():
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, int) and v > 0:
            out[k] = v
    return out


def _is_alive(pid: int) -> bool:
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(pid)
    except ImportError:
        # psutil missing — fallback uses os.kill(pid, 0)
        try:
            os.kill(pid, 0)
            return True
        except (PermissionError,):
            return True   # exists but not ours; still counts as alive
        except (ProcessLookupError, OSError):
            return False


def _proc_stats(pid: int) -> dict[str, Any]:
    """Returns {rss_mb, cpu_pct, age_s} for a live PID, or {} on failure."""
    try:
        import psutil  # type: ignore
        p = psutil.Process(pid)
        with p.oneshot():
            return {
                "rss_mb": round(p.memory_info().rss / (1024 * 1024), 1),
                "cpu_pct": round(p.cpu_percent(interval=None), 1),
                "age_s": round(time.time() - p.create_time(), 0),
            }
    except Exception:
        return {}


def _tail_log(role: str, n_lines: int = 100) -> list[str]:
    rel = _ROLE_LOG_FILES.get(role)
    if not rel:
        return []
    path = PROJECT_ROOT / rel
    if not path.exists():
        return []
    try:
        # Read up to last 200 KB so we don't load multi-GB logs
        size = path.stat().st_size
        cap = 200_000
        with open(path, "rb") as f:
            if size > cap:
                f.seek(size - cap)
                f.readline()  # discard partial line
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-n_lines:]
    except Exception as exc:
        logger.debug("tail %s: %s", path, exc)
        return []


def _load_deaths() -> list[dict]:
    if not DEATHS_PATH.exists():
        return []
    try:
        with open(DEATHS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []


def _save_deaths(deaths: list[dict]) -> None:
    try:
        DEATHS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEATHS_PATH, "w", encoding="utf-8") as f:
            json.dump(deaths, f, indent=2)
    except Exception as exc:
        logger.debug("save deaths: %s", exc)


def _append_log(line: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.debug("append log: %s", exc)


def _record_death(role: str, pid: int, last_stats: dict[str, Any]) -> None:
    log_tail = _tail_log(role, 100)
    last_log_line = log_tail[-1] if log_tail else ""

    # Try to extract a one-line "exit clue" — the last non-200-OK or last
    # non-status-line entry that might hint at why it died.
    exit_clue = ""
    for ln in reversed(log_tail[-30:]):
        low = ln.lower()
        if any(tok in low for tok in (
                "error", "traceback", "exception", "fatal", "killed",
                "memory", "out of memory", "broken pipe", "connection reset"
        )):
            exit_clue = ln.strip()[:200]
            break
    if not exit_clue:
        exit_clue = last_log_line.strip()[:200]

    record = {
        "role":          role,
        "pid":           pid,
        "died_at":       _now_iso(),
        "rss_mb":        last_stats.get("rss_mb"),
        "cpu_pct":       last_stats.get("cpu_pct"),
        "age_s":         last_stats.get("age_s"),
        "exit_clue":     exit_clue,
        "last_log_line": last_log_line.strip()[:200],
        "log_tail":      [ln[:240] for ln in log_tail[-20:]],
    }

    deaths = _load_deaths()
    deaths.insert(0, record)   # newest first
    if len(deaths) > MAX_DEATHS:
        deaths = deaths[:MAX_DEATHS]
    _save_deaths(deaths)

    summary = (f"{record['died_at']} {role} pid={pid} "
               f"age={record.get('age_s', '?')}s "
               f"rss={record.get('rss_mb', '?')}mb "
               f"clue={exit_clue[:120]}")
    _append_log(summary)
    logger.info(summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument("--once", action="store_true",
                        help="Single tick + exit (for tests / smoke checks)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # State: previous tick's PID map + per-pid stats for crash diagnostics.
    prev_pids: dict[str, int] = {}
    pid_stats: dict[int, dict[str, Any]] = {}

    _append_log(f"{_now_iso()} debug_supervisor STARTED (pid={os.getpid()}, poll={args.poll_seconds}s)")

    while True:
        try:
            current_pids = _read_pid_map()

            # Sample stats while processes are alive (so when they die, we
            # have a recent snapshot in pid_stats[pid]).
            for role, pid in current_pids.items():
                if _is_alive(pid):
                    stats = _proc_stats(pid)
                    if stats:
                        pid_stats[pid] = stats

            # Death detection: was tracked last tick, gone now.
            for role, prev_pid in prev_pids.items():
                # Two ways a death is "real":
                # (a) the PID is no longer in process_ids.json AND it's dead
                # (b) the role still maps to the same PID but it's dead
                cur_pid = current_pids.get(role)
                if cur_pid != prev_pid and not _is_alive(prev_pid):
                    _record_death(role, prev_pid, pid_stats.get(prev_pid, {}))
                    pid_stats.pop(prev_pid, None)
                elif cur_pid == prev_pid and not _is_alive(prev_pid):
                    _record_death(role, prev_pid, pid_stats.get(prev_pid, {}))
                    pid_stats.pop(prev_pid, None)

            prev_pids = current_pids

            if args.once:
                return 0

            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            _append_log(f"{_now_iso()} debug_supervisor STOPPED (KeyboardInterrupt)")
            return 0
        except Exception as exc:
            logger.warning("supervisor loop error: %s", exc)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
