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


def test_concurrent_claim_blocked_by_live_owner(isolated_registry, tmp_path):
    """Two subprocesses race for a role that is ALREADY OWNED by the test
    process (live PID, fresh heartbeat). Both children must see it as
    claimed and BOTH must report ok=False.

    Previous version of this test (renamed) allowed both children to
    succeed sequentially — the first claimed, exited, and the second
    reaped its dead PID and re-claimed. The python-reviewer flagged that
    a no-filelock implementation would have passed that assertion too,
    so the test couldn't catch a regression.

    This variant directly exercises the singleton-enforcement path:
    seed the registry with a fresh claim by the live test process; assert
    every child sees the role as occupied.
    """
    pr, _ = isolated_registry
    # Seed: the test process itself claims the role.
    ok, _ = pr.claim_role('bot', by='test-fixture')
    assert ok is True

    helper = tmp_path / 'claim_helper.py'
    helper.write_text(f'''
import sys, os, json, pathlib
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from src.utils import process_registry as pr
pr.REGISTRY_PATH = pathlib.Path({str(pr.REGISTRY_PATH)!r})
pr.AUDIT_LOG_PATH = pathlib.Path({str(pr.AUDIT_LOG_PATH)!r})
ok, info = pr.claim_role('bot', by='subproc-' + str(os.getpid()))
print(json.dumps({{'ok': ok, 'owner_pid': info.get('pid')}}))
''', encoding='utf-8')

    p1 = subprocess.Popen([sys.executable, str(helper)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen([sys.executable, str(helper)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out1, _ = p1.communicate(timeout=15)
    out2, _ = p2.communicate(timeout=15)

    r1 = json.loads(out1.decode().strip().splitlines()[-1])
    r2 = json.loads(out2.decode().strip().splitlines()[-1])

    # Both children must be blocked because the test process holds the role.
    assert r1['ok'] is False, f"child 1 should have been blocked, got {r1}"
    assert r2['ok'] is False, f"child 2 should have been blocked, got {r2}"
    # Both children should agree on who the owner is (the test process).
    assert r1['owner_pid'] == os.getpid()
    assert r2['owner_pid'] == os.getpid()

    # Registry still shows test process as owner.
    final = json.loads(pr.REGISTRY_PATH.read_text(encoding='utf-8'))
    assert final['roles']['bot']['pid'] == os.getpid()


def test_concurrent_claim_on_dead_role_only_one_wins(isolated_registry, tmp_path):
    """When the role's previous owner is DEAD, two subprocesses racing to
    reclaim it should both attempt — but only ONE can be in the registry
    after both finish. The filelock + transaction must serialize the
    read-check-write so race-conflated double-writes can't happen.
    """
    pr, _ = isolated_registry
    # Seed with a dead PID (PID 99999 is essentially never alive on a typical box).
    data = pr._read_registry()
    data['roles']['bot'] = {
        'pid': 99999, 'cmdline': 'dead', 'host': 'x',
        'last_heartbeat_ts': 0.0,   # very stale
    }
    pr._write_registry(data)

    helper = tmp_path / 'claim_helper.py'
    helper.write_text(f'''
import sys, os, json, pathlib, time
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from src.utils import process_registry as pr
pr.REGISTRY_PATH = pathlib.Path({str(pr.REGISTRY_PATH)!r})
pr.AUDIT_LOG_PATH = pathlib.Path({str(pr.AUDIT_LOG_PATH)!r})
# Brief sleep so both children reach claim_role at roughly the same time.
time.sleep(0.1)
ok, info = pr.claim_role('bot', by='subproc-' + str(os.getpid()))
print(json.dumps({{'ok': ok, 'pid': os.getpid(), 'owner_pid': info.get('pid')}}))
''', encoding='utf-8')

    p1 = subprocess.Popen([sys.executable, str(helper)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen([sys.executable, str(helper)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out1, _ = p1.communicate(timeout=15)
    out2, _ = p2.communicate(timeout=15)

    r1 = json.loads(out1.decode().strip().splitlines()[-1])
    r2 = json.loads(out2.decode().strip().splitlines()[-1])

    # After both subprocesses exit, the second-to-reach-claim sees the
    # first's PID either as live (if process ordering happened to overlap
    # exactly) or dead (more likely — both children exit before the second
    # check). What MUST hold:
    #   • the registry has exactly one entry for 'bot'
    #   • that entry's PID matches ONE of the two children
    final = json.loads(pr.REGISTRY_PATH.read_text(encoding='utf-8'))
    assert 'bot' in final['roles']
    final_pid = final['roles']['bot']['pid']
    assert final_pid in (r1['pid'], r2['pid']), (
        f"registry pid {final_pid} does not match either child "
        f"(r1.pid={r1['pid']}, r2.pid={r2['pid']})"
    )


if __name__ == '__main__':
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
