"""Log retention sweeper.

Deletes files under `logs/` older than RETENTION_DAYS so the directory
doesn't accumulate forever. Configurable via env var. Safe to call from
multiple processes — only acts on files whose mtime is past the cutoff.

Usage:
    from src.utils.log_retention import sweep_once, start_retention_thread
    n = sweep_once()                 # run once, return deleted count
    start_retention_thread()         # background thread, sweeps every 12h
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"

RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "5"))
SWEEP_INTERVAL_SEC = float(os.getenv("LOG_SWEEP_INTERVAL_SEC", str(12 * 3600)))

# Files matching these globs are pruned. Anything else under logs/ is left
# alone (e.g. a user's manual screenshots / notes).
_PRUNE_GLOBS = ("*.log", "*.log.*", "*.err", "*.out", "*.txt")


def _candidate_paths() -> list[Path]:
    if not LOGS_DIR.exists():
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in _PRUNE_GLOBS:
        for p in LOGS_DIR.glob(pattern):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def sweep_once(retention_days: int | None = None) -> int:
    """Delete log files older than `retention_days`. Returns count deleted."""
    days = retention_days if retention_days is not None else RETENTION_DAYS
    cutoff = time.time() - (days * 86400)
    deleted = 0
    for path in _candidate_paths():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.debug("[log_retention] could not remove %s: %s", path, exc)
    if deleted:
        logger.info("[log_retention] pruned %d files older than %d days from %s",
                    deleted, days, LOGS_DIR)
    return deleted


_thread_lock = threading.Lock()
_thread: threading.Thread | None = None


def start_retention_thread() -> threading.Thread:
    """Idempotent — starts a single daemon thread that sweeps every 12h."""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return _thread
        def _loop():
            # First sweep happens after a short delay so we don't hold up
            # process startup.
            time.sleep(60)
            while True:
                try:
                    sweep_once()
                except Exception as exc:
                    logger.warning("[log_retention] sweep error: %s", exc)
                time.sleep(SWEEP_INTERVAL_SEC)
        t = threading.Thread(target=_loop, daemon=True, name="log-retention")
        t.start()
        _thread = t
        return t


if __name__ == "__main__":
    # Allow `python -m src.utils.log_retention` from restart_all.ps1.
    n = sweep_once()
    print(f"[log_retention] pruned {n} files older than {RETENTION_DAYS} days")
