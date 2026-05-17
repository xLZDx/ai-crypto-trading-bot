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
    """Return True if a process with this PID exists AND is in a running
    state (not zombie / not dead). False otherwise.

    Uses psutil — the existing fallback to `os.kill(pid, 0)` returned True
    for zombie PIDs on POSIX (a zombie still holds its PID slot and signals
    succeed against it), which defeats the purpose of the registry. psutil
    is in requirements.txt; if it ever goes missing the import-error path
    treats every PID as DEAD so the registry fails-safe (operator can
    always reclaim a stale role).
    """
    if not pid or pid <= 0:
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        try:
            p = psutil.Process(pid)
            return p.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    except ImportError:
        # Fail-safe: without psutil we cannot distinguish zombies from live
        # processes, so report ALL PIDs as dead. This means a fresh claim
        # always succeeds and a previous run's leftover entry gets reaped.
        # The trade-off: two processes started simultaneously on a host
        # without psutil could BOTH succeed, vs. one falsely blocking forever.
        # Operators are warned to install psutil in this module's docstring.
        logger.critical(
            "[registry] psutil unavailable -- every PID will be reported as "
            "DEAD. Install psutil to get correct singleton-enforcement; "
            "without it, duplicate processes are possible."
        )
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


_AUDIT_LOG_WARN_ONCE = False  # module-level so we only warn once per process


def _append_audit_log(entry: dict) -> None:
    """Append one event to logs/process_registry.log (best-effort).

    Reviewer note: the FIRST failure now logs at WARNING so the operator
    sees that the audit-log path isn't writable. Subsequent failures stay
    quiet to avoid log spam.
    """
    global _AUDIT_LOG_WARN_ONCE
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
        if not _AUDIT_LOG_WARN_ONCE:
            logger.warning(
                "[registry] audit log append failed: %s -- "
                "logs/process_registry.log will not be written this run. "
                "(Suppressing subsequent failures to avoid spam.)", exc,
            )
            _AUDIT_LOG_WARN_ONCE = True
        else:
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
    """Try to claim *role* for the current process — ATOMIC.

    Returns:
        (True, my_entry)    — claim succeeded; caller now owns the role.
                              Caller should periodically call heartbeat(role).
        (False, other_entry) — another live PID already owns it; caller
                               should log and exit cleanly to avoid duplicate.

    Re-entrant: if the current process already owns the role, this returns
    (True, my_entry) without modifying the registry.

    Atomicity (X1 reviewer fix, 2026-05-13): the read-check-write sequence
    is held under a SINGLE filelock via `safe_json.transaction`. The previous
    pattern (read_registry → check → write_registry) acquired the lock
    twice, leaving a TOCTOU window where two concurrent claims for the same
    role could BOTH succeed.
    """
    if not role or not isinstance(role, str):
        raise ValueError(f"claim_role: role must be a non-empty string, got {role!r}")

    from src.utils.safe_json import transaction
    my_pid = os.getpid()
    now = time.time()
    audit_logs: list[dict] = []      # written to file log AFTER the lock releases
    info_log: str | None = None
    warn_log: str | None = None
    result: tuple[bool, dict] = (False, {})

    with transaction(str(REGISTRY_PATH), default={'roles': {}, 'audit': []}) as data:
        data.setdefault('roles', {})
        data.setdefault('audit', [])
        existing = data['roles'].get(role)

        if existing:
            ex_pid = existing.get('pid')
            ex_hb = existing.get('last_heartbeat_ts') or 0.0
            ex_alive = _pid_alive(ex_pid)
            ex_fresh = (now - float(ex_hb)) < ZOMBIE_AGE_S

            # Re-entrant: same process re-claiming → silent success.
            if ex_pid == my_pid:
                # No state change; transaction will write back identical state.
                return True, existing

            if ex_alive and ex_fresh:
                warn_log = (
                    f"[registry] role {role!r} already claimed by PID {ex_pid} "
                    f"(cmd={(existing.get('cmdline', '?') or '')[:60]}, "
                    f"hb_age={now - float(ex_hb):.0f}s); this process should "
                    "exit to avoid duplicate."
                )
                entry_blocked = {
                    'ts': _now_iso(), 'event': 'claim_blocked', 'role': role,
                    'pid': my_pid, 'reason': f'live owner PID {ex_pid}', 'by': by,
                }
                data['audit'].append(entry_blocked)
                audit_logs.append(entry_blocked)
                # Cap audit
                if len(data['audit']) > AUDIT_RING_SIZE:
                    data['audit'] = data['audit'][-AUDIT_RING_SIZE:]
                result = (False, existing)
                # Fall through to write the audit entry, then return below.

            else:
                # Stale entry: reap and proceed with the new claim.
                reap_entry = {
                    'ts': _now_iso(), 'event': 'reap', 'role': role,
                    'pid': ex_pid, 'reason': f'alive={ex_alive} fresh={ex_fresh}',
                    'by': by,
                }
                data['audit'].append(reap_entry)
                audit_logs.append(reap_entry)
                existing = None  # treat as no-existing for the next branch

        if existing is None and result == (False, {}):
            # New claim path.
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
            claim_entry = {
                'ts': _now_iso(), 'event': 'claim', 'role': role,
                'pid': my_pid, 'reason': '', 'by': by,
            }
            data['audit'].append(claim_entry)
            audit_logs.append(claim_entry)
            if len(data['audit']) > AUDIT_RING_SIZE:
                data['audit'] = data['audit'][-AUDIT_RING_SIZE:]
            info_log = f"[registry] claimed role {role!r} (PID {my_pid}, by={by})"
            result = (True, entry)

    # Lock released — now do best-effort side-effects (logger + file audit).
    if warn_log:
        logger.warning(warn_log)
    if info_log:
        logger.info(info_log)
    for entry in audit_logs:
        _append_audit_log(entry)
    return result


def release_role(role: str, reason: str = 'graceful') -> bool:
    """Release a role we own. Idempotent — returns True if we successfully
    released (or no entry existed); False if the entry belonged to someone
    else and we didn't touch it (safety: never release another process's
    claim).

    Reviewer fix 2026-05-13: wrong-owner refusal now logs at WARNING (was
    DEBUG, off by default). The refusal is an anomalous event — if a
    process's atexit fires in a subprocess context or the lifecycle wiring
    is broken, the operator needs to see it.
    """
    from src.utils.safe_json import transaction
    my_pid = os.getpid()
    audit: dict | None = None
    info: str | None = None
    warn: str | None = None
    result = True

    with transaction(str(REGISTRY_PATH), default={'roles': {}, 'audit': []}) as data:
        data.setdefault('roles', {})
        data.setdefault('audit', [])
        existing = data['roles'].get(role)
        if not existing:
            return True
        if existing.get('pid') != my_pid:
            warn = (
                f"[registry] refusing to release role {role!r} — owned by "
                f"PID {existing.get('pid')}, we are PID {my_pid}"
            )
            result = False
        else:
            data['roles'].pop(role, None)
            audit = {
                'ts': _now_iso(), 'event': 'release', 'role': role,
                'pid': my_pid, 'reason': reason, 'by': '',
            }
            data['audit'].append(audit)
            if len(data['audit']) > AUDIT_RING_SIZE:
                data['audit'] = data['audit'][-AUDIT_RING_SIZE:]
            info = f"[registry] released role {role!r} (reason={reason})"

    if warn: logger.warning(warn)
    if info: logger.info(info)
    if audit: _append_audit_log(audit)
    return result


def heartbeat(role: str) -> bool:
    """Update last_heartbeat for *role* IF we own it.

    Returns True on success, False if we no longer own the role OR if
    the registry lock could not be acquired within the timeout window.

    Reviewer fix 2026-05-13: returning False used to be a silent no-op,
    which masked the case where the reaper had evicted this process while
    it was still running (the runaway scenario). Now logs at WARNING when
    ownership has been lost so callers can react (e.g. exit the bot to
    let the watchdog claim cleanly).

    Operator fix 2026-05-15: heartbeat ran with the safe_json default 5 s
    filelock timeout. During training the registry lock is contended by
    many parallel worker subprocesses claiming/heartbeating, and the bot's
    heartbeat thread surfaced 'The file lock ... could not be acquired'
    in the dashboard warning banner. Bumped to 15 s because a heartbeat
    is periodic background work — a slow lock is fine; a missed heartbeat
    is fine; a noisy operator banner is not.
    """
    from filelock import Timeout as _LockTimeout
    from src.utils.safe_json import transaction
    my_pid = os.getpid()
    ok = False
    try:
        with transaction(str(REGISTRY_PATH),
                          default={'roles': {}, 'audit': []},
                          timeout=15.0) as data:
            data.setdefault('roles', {})
            existing = data['roles'].get(role)
            if existing and existing.get('pid') == my_pid:
                existing['last_heartbeat'] = _now_iso()
                existing['last_heartbeat_ts'] = time.time()
                data['roles'][role] = existing
                ok = True
    except _LockTimeout:
        # Contended lock during heavy training is expected; downgrade to
        # DEBUG so the dashboard banner isn't flooded with WARNINGs that
        # the operator can't act on. The bot's _hb_loop() bumps its own
        # consecutive-failure counter, so true persistent failure (5+
        # consecutive missed beats) still escalates to ERROR.
        logger.debug(
            "[registry] heartbeat(%r) lock timeout -- registry busy, will retry next tick",
            role,
        )
        return False
    if not ok:
        logger.warning(
            "[registry] heartbeat(%r) ignored -- current process (PID %s) "
            "no longer owns this role. Another process may have reclaimed it.",
            role, my_pid,
        )
    return ok


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
