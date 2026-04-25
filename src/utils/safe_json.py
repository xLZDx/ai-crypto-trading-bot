import json
import os
import tempfile
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
