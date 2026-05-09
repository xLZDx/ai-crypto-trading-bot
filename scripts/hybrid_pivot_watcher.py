"""hybrid_pivot_watcher — detect TFT @ 15m completion → kill master orchestrator
→ apply skip-TFT-and-OFT patch → respawn for meta + backtest.

Triggered manually 2026-05-09 to pivot the in-flight sweep so it finishes by
~01:00 UTC (04:00 Chișinău) instead of 17:00+ tomorrow.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TFT_15M_META = PROJECT_ROOT / 'models' / 'tft_15m_meta.json'
LOG_PATH     = PROJECT_ROOT / 'logs' / 'hybrid_pivot.log'

# Trip when meta is fresher than this anchor — orchestrator started 11:41 UTC
# so any tft_15m_meta.json mtime > the master's start time = freshly written.
ANCHOR_TS = 1778445000   # ~11:30 UTC 2026-05-09 (master orchestrator start)

POLL_S = 30


def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f'[{datetime.now(timezone.utc).isoformat()}] {msg}\n'
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line)
    print(line, end='', file=sys.stderr)


def _meta_fresh() -> bool:
    if not TFT_15M_META.exists():
        return False
    return TFT_15M_META.stat().st_mtime > ANCHOR_TS


def _kill_orchestrator() -> int:
    """Stop any pipeline_orchestrator process. Returns count killed."""
    try:
        import psutil
    except ImportError:
        _log('psutil unavailable; cannot kill cleanly')
        return 0
    killed = 0
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if not (p.info.get('name') or '').lower().startswith('python'):
                continue
            cmd = ' '.join(p.info.get('cmdline') or [])
            if 'pipeline_orchestrator' in cmd:
                _log(f'killing pipeline_orchestrator PID {p.info["pid"]}')
                p.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed


def _patch_train_all_skip_tft_oft() -> None:
    """Set env var so train_all_models honours a skip list. The trainer
    already iterates per_key_tfs, so passing AI_TRADER_TRAIN_SKIP_KEYS
    (comma-separated) is the cleanest non-invasive change. Add the env
    var read to train_all_models if not present."""
    src = PROJECT_ROOT / 'src' / 'engine' / 'train_all_models.py'
    text = src.read_text(encoding='utf-8')
    if 'AI_TRADER_TRAIN_SKIP_KEYS' in text:
        _log('skip-keys env support already in train_all_models — no patch needed')
        return
    inject_after = "if os.getenv('AI_TRADER_TRAIN_TF_MAP', '').lower() in ('strict', 'all', 'strict_all_all'):"
    inject_block = '''
# Skip-keys override (v4 hybrid pivot 2026-05-09): comma-separated model
# keys to skip entirely for THIS run. Used for hot pivots that need to
# defer expensive trainers (tft, oft) to a follow-up sweep.
_skip_keys = {k.strip().lower() for k in os.getenv('AI_TRADER_TRAIN_SKIP_KEYS', '').split(',') if k.strip()}
if _skip_keys:
    DEFAULT_PER_KEY_TFS = {k: v for k, v in DEFAULT_PER_KEY_TFS.items() if k not in _skip_keys}
    log.info("AI_TRADER_TRAIN_SKIP_KEYS override: skipping %s", sorted(_skip_keys))
'''
    if inject_after not in text:
        _log(f'WARN: anchor "{inject_after}" not found in train_all_models.py — patch skipped, fall back to manual edit')
        return
    new_text = text.replace(
        inject_after + '\n    _ALL_TFS = (\'1m\', \'5m\', \'15m\', \'1h\', \'4h\', \'1d\', \'1w\')\n    DEFAULT_PER_KEY_TFS = {k: _ALL_TFS for k in DEFAULT_PER_KEY_TFS}\n    log.info("AI_TRADER_TRAIN_TF_MAP override: strict all×all (49 combos)")',
        inject_after + '\n    _ALL_TFS = (\'1m\', \'5m\', \'15m\', \'1h\', \'4h\', \'1d\', \'1w\')\n    DEFAULT_PER_KEY_TFS = {k: _ALL_TFS for k in DEFAULT_PER_KEY_TFS}\n    log.info("AI_TRADER_TRAIN_TF_MAP override: strict all×all (49 combos)")\n' + inject_block
    )
    if new_text == text:
        _log('WARN: patch produced no change — anchor mismatch')
        return
    src.write_text(new_text, encoding='utf-8')
    _log('train_all_models.py patched with AI_TRADER_TRAIN_SKIP_KEYS support')


def _respawn() -> int:
    """Spawn fresh orchestrator with skip env vars set."""
    venv = PROJECT_ROOT / 'venv' / 'Scripts' / 'python.exe'
    log_out = PROJECT_ROOT / 'logs' / 'pipeline_orchestrator.log'
    log_err = PROJECT_ROOT / 'logs' / 'pipeline_orchestrator.err.log'
    env = os.environ.copy()
    env['AI_TRADER_TRAIN_SKIP_IF_FRESH_S'] = '172800'   # 48h
    env['AI_TRADER_SCALPING_SMOTE_MAX_ROWS'] = '500000'
    env['AI_TRADER_TRAIN_SKIP_KEYS'] = 'tft,oft'        # the pivot
    cmd = [str(venv), '-m', 'src.engine.pipeline_orchestrator']
    flags = 0
    if sys.platform == 'win32':
        # Detached + new process group so it survives this script exit.
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        stdout=open(log_out, 'a', encoding='utf-8'),
        stderr=open(log_err, 'a', encoding='utf-8'),
        creationflags=flags,
    )
    _log(f'respawned orchestrator PID {proc.pid} with SKIP_KEYS=tft,oft + 48h freshness')
    return proc.pid


def main() -> int:
    _log(f'hybrid_pivot_watcher starting; polling {TFT_15M_META} every {POLL_S}s')
    while True:
        if _meta_fresh():
            _log(f'TFT @ 15m meta fresh ({TFT_15M_META.stat().st_mtime}) — pivoting')
            _kill_orchestrator()
            time.sleep(3)
            _patch_train_all_skip_tft_oft()
            time.sleep(2)
            new_pid = _respawn()
            _log(f'pivot complete; new orchestrator PID {new_pid}')
            return 0
        time.sleep(POLL_S)


if __name__ == '__main__':
    sys.exit(main())
