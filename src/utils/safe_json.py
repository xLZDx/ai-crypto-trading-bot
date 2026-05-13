import json
import os
import tempfile
from contextlib import contextmanager

from filelock import FileLock


def _lock_path(filepath: str) -> str:
    return filepath + ".lock"


def read_json(filepath: str, default=None):
    """Read a JSON file with a shared lock to prevent reading mid-write."""
    lock = FileLock(_lock_path(filepath), timeout=5)
    try:
        with lock:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except FileNotFoundError:
        return default
    except Exception:
        return default


def write_json(filepath: str, data, indent: int = 4):
    """Write JSON atomically: write to a temp file then rename, with an exclusive lock."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    lock = FileLock(_lock_path(filepath), timeout=5)
    with lock:
        dir_name = os.path.dirname(os.path.abspath(filepath))
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


@contextmanager
def transaction(filepath: str, default=None, indent: int = 4, timeout: float = 5.0):
    """Atomic read-modify-write context manager.

    The plain read_json / write_json pair acquires the filelock TWICE — a
    classic TOCTOU. process_registry.claim_role uses this transaction to
    hold ONE lock across the read, the liveness check, and the write so
    two concurrent claims for the same role cannot both succeed.

    Usage:
        with transaction('data/foo.json', default={}) as state:
            # state is the parsed JSON (or `default` if missing/malformed).
            state['key'] = 'value'
            # On block exit, state is written back atomically.

    Notes:
        - Caller can mutate `state` in place; the manager picks up the
          changes via the same reference.
        - If the block raises, the file is NOT rewritten — exceptions
          propagate after the lock is released.
        - On clean exit, the write goes through the same atomic-rename
          pattern as write_json.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    lock = FileLock(_lock_path(filepath), timeout=timeout)
    with lock:
        # Read (or default).
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except FileNotFoundError:
            state = default if default is not None else {}
        except Exception:
            state = default if default is not None else {}

        yield state

        # Write back atomically.
        dir_name = os.path.dirname(os.path.abspath(filepath))
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=indent, ensure_ascii=False)
            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
