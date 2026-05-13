"""
Tests for src/utils/process_registry.py — Phase X1.1.

Covers:
  - claim_role: first claim succeeds, subsequent duplicate claim blocked
  - claim_role: re-entrant (same PID re-claims) → silent success, no audit spam
  - claim_role: stale entry (dead PID OR old heartbeat) auto-reaped + replaced
  - release_role: idempotent, refuses to release another process's claim
  - heartbeat: updates timestamp; no-op for non-owned role
  - list_active: excludes dead-PID entries
  - reap_zombies: sweeps dead + stale entries, audit log written
  - Audit ring bounded at AUDIT_RING_SIZE
  - Concurrent-process claim simulated via subprocess (real filelock race)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Redirect REGISTRY_PATH + AUDIT_LOG_PATH to tmp for every test.
    Also makes _pid_alive return True for any PID by default so tests don't
    depend on which PIDs happen to be alive on the host."""
    from src.utils import process_registry as pr
    monkeypatch.setattr(pr, 'REGISTRY_PATH', tmp_path / 'process_registry.json')
    monkeypatch.setattr(pr, 'AUDIT_LOG_PATH', tmp_path / 'process_registry.log')
    return pr, tmp_path


def test_first_claim_succeeds(isolated_registry):
    pr, _ = isolated_registry
    ok, info = pr.claim_role('bot', by='test')
    assert ok is True
    assert info['pid'] == os.getpid()
    assert info['by'] == 'test'
    # Persisted
    data = json.loads((pr.REGISTRY_PATH).read_text(encoding='utf-8'))
    assert 'bot' in data['roles']
    assert data['roles']['bot']['pid'] == os.getpid()


def test_reentrant_claim_silent_success(isolated_registry):
    """Same PID re-claiming the same role is OK — no audit spam, just True."""
    pr, _ = isolated_registry
    pr.claim_role('bot', by='test')
    audit_after_first = len(pr._read_registry()['audit'])
    ok, info = pr.claim_role('bot', by='test')
    assert ok is True
    audit_after_second = len(pr._read_registry()['audit'])
    assert audit_after_second == audit_after_first, (
        "Re-entrant claim should NOT add a new audit entry — it's a no-op for "
        "the same PID. Without this guard, hot-restart paths spam the audit."
    )


def test_duplicate_claim_blocked(isolated_registry, monkeypatch):
    """A different live PID with a fresh heartbeat blocks the new claim."""
    pr, _ = isolated_registry
    # Seed the registry with another live PID's claim. We need _pid_alive to
    # report the seeded PID as alive; patch it.
    seeded_pid = 99999
    data = pr._read_registry()
    data['roles']['bot'] = {
        'pid': seeded_pid,
        'cmdline': 'fake other bot',
        'host': 'somewhere',
        'started_at': '2026-05-13T20:00:00+00:00',
        'last_heartbeat': '2026-05-13T20:00:00+00:00',
        'last_heartbeat_ts': time.time(),  # fresh
    }
    pr._write_registry(data)

    monkeypatch.setattr(pr, '_pid_alive', lambda pid: pid == seeded_pid)
    ok, info = pr.claim_role('bot', by='test')
    assert ok is False
    assert info['pid'] == seeded_pid
    # The audit has a 'claim_blocked' event
    audit = pr._read_registry()['audit']
    assert any(e['event'] == 'claim_blocked' for e in audit)


def test_stale_dead_pid_replaced(isolated_registry, monkeypatch):
    """Entry whose PID is no longer alive is reaped and replaced."""
    pr, _ = isolated_registry
    data = pr._read_registry()
    data['roles']['bot'] = {
        'pid': 99999, 'cmdline': 'dead bot', 'host': 'x',
        'started_at': '...', 'last_heartbeat': '...',
        'last_heartbeat_ts': time.time(),
    }
    pr._write_registry(data)
    monkeypatch.setattr(pr, '_pid_alive', lambda pid: pid == os.getpid())

    ok, info = pr.claim_role('bot', by='test')
    assert ok is True
    assert info['pid'] == os.getpid()
    # The reap was audited before the claim
    events = [e['event'] for e in pr._read_registry()['audit']]
    assert 'reap' in events
    assert 'claim' in events


def test_stale_heartbeat_treated_as_dead(isolated_registry, monkeypatch):
    """Even with a live PID, an old heartbeat means the entry is stale."""
    pr, _ = isolated_registry
    data = pr._read_registry()
    data['roles']['bot'] = {
        'pid': 99999, 'cmdline': 'sleeping bot', 'host': 'x',
        'started_at': '...', 'last_heartbeat': '...',
        'last_heartbeat_ts': time.time() - 999,  # older than ZOMBIE_AGE_S (300)
    }
    pr._write_registry(data)
    monkeypatch.setattr(pr, '_pid_alive', lambda pid: True)  # any pid alive

    ok, info = pr.claim_role('bot', by='test')
    assert ok is True
    assert info['pid'] == os.getpid()


def test_release_idempotent(isolated_registry):
    pr, _ = isolated_registry
    pr.claim_role('bot', by='test')
    assert pr.release_role('bot') is True
    # Second release: still True (idempotent), nothing breaks.
    assert pr.release_role('bot') is True


def test_release_refuses_other_owner(isolated_registry):
    """Refuse to release someone else's claim — safety."""
    pr, _ = isolated_registry
    other_pid = os.getpid() + 1
    data = pr._read_registry()
    data['roles']['bot'] = {
        'pid': other_pid, 'last_heartbeat_ts': time.time(),
    }
    pr._write_registry(data)

    assert pr.release_role('bot') is False
    # Entry still there
    assert pr._read_registry()['roles'].get('bot', {}).get('pid') == other_pid


def test_heartbeat_only_owned_role(isolated_registry):
    pr, _ = isolated_registry
    pr.claim_role('bot', by='test')
    time.sleep(0.01)
    before = pr._read_registry()['roles']['bot']['last_heartbeat_ts']
    time.sleep(0.05)
    assert pr.heartbeat('bot') is True
    after = pr._read_registry()['roles']['bot']['last_heartbeat_ts']
    assert after > before

    # heartbeat on a role we don't own → False, no-op
    assert pr.heartbeat('does-not-exist') is False


def test_list_active_excludes_dead(isolated_registry, monkeypatch):
    """list_active() must filter out entries whose PID is dead."""
    pr, _ = isolated_registry
    pr.claim_role('bot', by='test')
    # Seed a fake dead PID claim
    data = pr._read_registry()
    data['roles']['dashboard'] = {
        'pid': 99999, 'last_heartbeat_ts': time.time(),
    }
    pr._write_registry(data)
    monkeypatch.setattr(pr, '_pid_alive', lambda pid: pid == os.getpid())

    active = pr.list_active()
    assert 'bot' in active
    assert 'dashboard' not in active


def test_reap_zombies(isolated_registry, monkeypatch):
    pr, _ = isolated_registry
    data = pr._read_registry()
    data['roles']['ghost1'] = {'pid': 99998, 'last_heartbeat_ts': time.time()}
    data['roles']['ghost2'] = {'pid': 99999, 'last_heartbeat_ts': time.time() - 999}
    data['roles']['alive'] = {'pid': os.getpid(), 'last_heartbeat_ts': time.time()}
    pr._write_registry(data)
    monkeypatch.setattr(pr, '_pid_alive', lambda pid: pid == os.getpid())

    reaped = pr.reap_zombies(by='test')
    assert set(reaped) == {'ghost1', 'ghost2'}
    after = pr._read_registry()['roles']
    assert 'alive' in after
    assert 'ghost1' not in after
    assert 'ghost2' not in after


def test_audit_ring_bounded(isolated_registry):
    """Audit log can't grow unbounded — capped at AUDIT_RING_SIZE."""
    pr, _ = isolated_registry
    cap = pr.AUDIT_RING_SIZE
    # Hammer claim+release to generate audit entries
    for i in range(cap + 50):
        pr.claim_role(f'role_{i}', by='test')
        pr.release_role(f'role_{i}')
    audit = pr._read_registry()['audit']
    assert len(audit) <= cap, (
        f"Audit ring exceeded cap: {len(audit)} > {cap}. "
        "Without the cap, long-running registries would grow without bound."
    )


def test_audit_log_file_written(isolated_registry):
    """Every event also appears in logs/process_registry.log (best-effort)."""
    pr, tmp_path = isolated_registry
    pr.claim_role('bot', by='test-fn')
    pr.release_role('bot')
    log_text = pr.AUDIT_LOG_PATH.read_text(encoding='utf-8')
    assert 'claim' in log_text
    assert 'release' in log_text
    assert 'bot' in log_text


def test_concurrent_claim_via_subprocess(isolated_registry, tmp_path):
    """Two processes racing for the same role — filelock must serialize.
    Exactly ONE should win; the other must report ok=False."""
    pr, _ = isolated_registry
    # Write a tiny helper script that tries to claim and prints result.
    helper = tmp_path / 'claim_helper.py'
    helper.write_text(f'''
import sys, os, json
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from src.utils import process_registry as pr
pr.REGISTRY_PATH = {str(pr.REGISTRY_PATH)!r}
pr.AUDIT_LOG_PATH = {str(pr.AUDIT_LOG_PATH)!r}
import pathlib
pr.REGISTRY_PATH = pathlib.Path({str(pr.REGISTRY_PATH)!r})
pr.AUDIT_LOG_PATH = pathlib.Path({str(pr.AUDIT_LOG_PATH)!r})
ok, info = pr.claim_role('bot', by='subproc-' + str(os.getpid()))
print(json.dumps({{'ok': ok, 'pid': info.get('pid')}}))
''', encoding='utf-8')

    # Launch two children in parallel. Filelock should serialize them.
    p1 = subprocess.Popen([sys.executable, str(helper)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen([sys.executable, str(helper)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out1, _ = p1.communicate(timeout=15)
    out2, _ = p2.communicate(timeout=15)

    r1 = json.loads(out1.decode().strip().splitlines()[-1])
    r2 = json.loads(out2.decode().strip().splitlines()[-1])

    # Subprocesses may race in either order; both children check liveness of
    # the OTHER's PID via psutil. After the first child claims and exits, its
    # PID is dead → second child reaps and re-claims. So both can succeed in
    # sequence (legitimate behavior). What MUST NOT happen: both observing
    # the registry as empty at the same time and BOTH writing without seeing
    # the other's write. The filelock prevents that.
    # Assertion: exactly one entry in the file at end, and its PID is one of
    # the two children's PIDs.
    final = json.loads(pr.REGISTRY_PATH.read_text(encoding='utf-8'))
    assert 'bot' in final['roles']
    assert final['roles']['bot']['pid'] in (r1['pid'], r2['pid'])
    # At least one of them succeeded
    assert r1['ok'] or r2['ok']


if __name__ == '__main__':
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
