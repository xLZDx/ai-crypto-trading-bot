"""
Process registry — singleton-enforcement for long-running roles.

Problem (2026-05-13 incident): the bot left zombie / duplicate processes after
restart_all. PID 29936 accumulated 12.8 hours of CPU while spamming orders;
restart_all spawned a fresh bot on top instead of detecting the old one. The
runaway loop that took down the dashboard came from this exact mode of
failure — old code in process, new code in source tree, no dispatcher
arbitrating.

Solution: every long-running process (bot, dashboard, cluster_orch, monitor,
worker, watchdogs) calls `claim_role()` at startup. If another PID already
owns the role AND that PID is alive AND has a recent heartbeat, the new
process logs and exits cleanly — no duplicate. Reaper sweeps dead PIDs out
of the registry on a 60s cadence.

Storage: `data/process_registry.json` with the same atomic filelock pattern
as `agent_status.json`. Each entry:

    {
      "roles": {
        "bot": {
          "pid": 33012,
          "cmdline": "python -m src.main",
          "started_at": "2026-05-13T20:08:11+00:00",
          "last_heartbeat": "2026-05-13T20:30:45+00:00",
          "host": "operator-pc"
        },
        "dashboard": { ... }
      },
      "audit": [
        {"ts": "...", "event": "claim", "role": "bot", "pid": 33012, "by": "src.main"},
        {"ts": "...", "event": "release", "role": "bot", "pid": 33012, "reason": "graceful shutdown"},
        {"ts": "...", "event": "reap", "role": "bot", "pid": 33012, "reason": "PID dead"}
      ]
    }

Audit ring is capped at 200 entries; older events fall off. For a queryable
permanent log, every event also goes to `logs/process_registry.log` (plain
text, append-only).

API:
    claim_role(role, by="") -> (ok: bool, info: dict)
        ok=True  → caller owns the role; record persisted.
        ok=False → another live PID already owns; info has its details.
                   Caller should log and exit.

    release_role(role, reason="graceful")
        Drop the role from the registry. Idempotent.

    heartbeat(role)
        Update last_heartbeat. Call periodically (every interval_sec of the
        owning process — same cadence as agent_bus heartbeats). Cheap (~5ms).

    list_active() -> dict
        Returns {role: entry} for every role with a live PID.

    reap_zombies() -> list[str]
        Sweep entries whose PIDs are dead OR whose last_heartbeat is older
        than ZOMBIE_AGE_S. Returns the list of role names reaped.
"""
from __future__ import annotations

import logging
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / 'data' / 'process_registry.json'
AUDIT_LOG_PATH = PROJECT_ROOT / 'logs' / 'process_registry.log'

# A heartbeat older than this is treated as a dead process even if the PID
# happens to be alive (could be a re-used PID from a different program).
ZOMBIE_AGE_S = 300.0   # 5 minutes
AUDIT_RING_SIZE = 200


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int | None) -> bool:
    """Return True if a process with this PID exists right now.

    Uses psutil when available (most accurate, also catches zombie state),
    falls back to os.kill(pid, 0) for environments without psutil.
    """
    if not pid or pid <= 0:
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        # Distinguish zombie/stopped from running. A zombie is a dead process
        # whose PID hasn't been reaped — we treat that as "not running".
        try:
            p = psutil.Process(pid)
            return p.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def _read_registry() -> dict:
    """Load the registry file. Returns a fresh skeleton if missing/malformed."""
    from src.utils.safe_json import read_json
    data = read_json(str(REGISTRY_PATH), default=None)
    if not isinstance(data, dict) or 'roles' not in data:
        return {'roles': {}, 'audit': []}
    data.setdefault('roles', {})
    data.setdefault('audit', [])
    return data


def _write_registry(data: dict) -> None:
    from src.utils.safe_json import write_json
    # Cap audit ring before persist so unbounded growth can't blow up the file.
    audit = data.get('audit', [])
    if len(audit) > AUDIT_RING_SIZE:
        data['audit'] = audit[-AUDIT_RING_SIZE:]
    write_json(str(REGISTRY_PATH), data, indent=2)


def _append_audit_log(entry: dict) -> None:
    """Append one event to logs/process_registry.log (best-effort)."""
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = (
            f"{entry.get('ts','')}  {entry.get('event','?'):8s}  "
            f"role={entry.get('role',''):20s}  pid={entry.get('pid','')}  "
            f"reason={entry.get('reason','')}  by={entry.get('by','')}\n"
        )
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line)
    except Exception as exc:
        logger.debug("[registry] audit log append failed: %s", exc)


def _record_event(data: dict, event: str, role: str, pid: int | None,
                  reason: str = '', by: str = '') -> None:
    entry = {
        'ts': _now_iso(),
        'event': event,
        'role': role,
        'pid': pid,
        'reason': reason,
        'by': by,
    }
    data.setdefault('audit', []).append(entry)
    _append_audit_log(entry)


def _current_cmdline() -> str:
    """Best-effort cmdline of the current process. Used for audit / display."""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return ' '.join(p.cmdline())[:300]
    except Exception:
        # Fallback — Python's argv is sometimes truncated but acceptable for audit.
        import sys
        return ' '.join(sys.argv)[:300]


def claim_role(role: str, by: str = '') -> tuple[bool, dict]:
    """Try to claim *role* for the current process.

    Returns:
        (True, my_entry)    — claim succeeded; caller now owns the role.
                              Caller should periodically call heartbeat(role).
        (False, other_entry) — another live PID already owns it; caller
                               should log and exit cleanly to avoid duplicate.

    The check is "live PID with a recent heartbeat" — a stale entry whose PID
    no longer exists (or whose heartbeat is older than ZOMBIE_AGE_S) is
    treated as released and the new caller takes over.

    Re-entrant: if the current process already owns the role, this returns
    (True, my_entry) without modifying the registry. Lets components call
    claim_role() defensively at multiple init points.
    """
    if not role or not isinstance(role, str):
        raise ValueError(f"claim_role: role must be a non-empty string, got {role!r}")

    data = _read_registry()
    existing = data['roles'].get(role)
    now = time.time()
    my_pid = os.getpid()

    if existing:
        ex_pid = existing.get('pid')
        ex_hb = existing.get('last_heartbeat_ts') or 0.0
        ex_alive = _pid_alive(ex_pid)
        ex_fresh = (now - float(ex_hb)) < ZOMBIE_AGE_S

        # Re-entrant: same process re-claiming → silent success.
        if ex_pid == my_pid:
            return True, existing

        if ex_alive and ex_fresh:
            logger.warning(
                "[registry] role %r already claimed by PID %s (cmd=%s, hb_age=%.0fs); "
                "this process should exit to avoid duplicate.",
                role, ex_pid, existing.get('cmdline', '?')[:60], now - float(ex_hb),
            )
            _record_event(data, 'claim_blocked', role, my_pid,
                          reason=f'live owner PID {ex_pid}', by=by)
            _write_registry(data)
            return False, existing

        # Otherwise the existing entry is stale (dead PID or no heartbeat).
        # Reap it and proceed with the new claim.
        _record_event(data, 'reap', role, ex_pid,
                      reason=f"alive={ex_alive} fresh={ex_fresh}", by=by)

    entry = {
        'pid': my_pid,
        'cmdline': _current_cmdline(),
        'host': socket.gethostname(),
        'started_at': _now_iso(),
        'last_heartbeat': _now_iso(),
        'last_heartbeat_ts': now,
        'by': by,
    }
    data['roles'][role] = entry
    _record_event(data, 'claim', role, my_pid, by=by)
    _write_registry(data)
    logger.info("[registry] claimed role %r (PID %s, by=%s)", role, my_pid, by)
    return True, entry


def release_role(role: str, reason: str = 'graceful') -> bool:
    """Release a role we own. Idempotent — returns True if we successfully
    released (or no entry existed); False if the entry belonged to someone
    else and we didn't touch it (safety: never release another process's
    claim).
    """
    data = _read_registry()
    existing = data['roles'].get(role)
    if not existing:
        return True
    if existing.get('pid') != os.getpid():
        logger.debug(
            "[registry] refusing to release role %r — owned by PID %s, we are PID %s",
            role, existing.get('pid'), os.getpid(),
        )
        return False
    data['roles'].pop(role, None)
    _record_event(data, 'release', role, os.getpid(), reason=reason)
    _write_registry(data)
    logger.info("[registry] released role %r (reason=%s)", role, reason)
    return True


def heartbeat(role: str) -> bool:
    """Update last_heartbeat for *role* IF we own it. Silently no-op otherwise.
    Returns True on success.

    Cheap (~5ms with the filelock). Designed to be called from the existing
    agent_bus heartbeat loop or any periodic timer in the owning process.
    """
    data = _read_registry()
    existing = data['roles'].get(role)
    if not existing or existing.get('pid') != os.getpid():
        return False
    now_iso = _now_iso()
    existing['last_heartbeat'] = now_iso
    existing['last_heartbeat_ts'] = time.time()
    data['roles'][role] = existing
    _write_registry(data)
    return True


def list_active() -> dict[str, dict]:
    """Return every role with a live + recent-heartbeat entry."""
    data = _read_registry()
    now = time.time()
    out: dict[str, dict] = {}
    for role, entry in (data.get('roles') or {}).items():
        if not isinstance(entry, dict):
            continue
        pid = entry.get('pid')
        hb_ts = float(entry.get('last_heartbeat_ts') or 0.0)
        if _pid_alive(pid) and (now - hb_ts) < ZOMBIE_AGE_S:
            out[role] = entry
    return out


def reap_zombies(by: str = 'reaper') -> list[str]:
    """Sweep entries whose PIDs are dead OR whose heartbeats are older than
    ZOMBIE_AGE_S. Returns the role names that were reaped. Safe to call
    concurrently from multiple processes (one filelock-serialized).
    """
    data = _read_registry()
    now = time.time()
    reaped: list[str] = []
    for role, entry in list((data.get('roles') or {}).items()):
        if not isinstance(entry, dict):
            data['roles'].pop(role, None)
            continue
        pid = entry.get('pid')
        hb_ts = float(entry.get('last_heartbeat_ts') or 0.0)
        alive = _pid_alive(pid)
        fresh = (now - hb_ts) < ZOMBIE_AGE_S
        if not alive or not fresh:
            data['roles'].pop(role, None)
            _record_event(data, 'reap', role, pid,
                          reason=f"alive={alive} fresh={fresh}", by=by)
            reaped.append(role)
    if reaped:
        _write_registry(data)
        logger.info("[registry] reaped %d zombie role(s): %s", len(reaped), reaped)
    return reaped


def get_audit_tail(n: int = 50) -> list[dict]:
    """Return the last N audit events. For dashboard display."""
    data = _read_registry()
    audit = data.get('audit') or []
    return audit[-n:]
