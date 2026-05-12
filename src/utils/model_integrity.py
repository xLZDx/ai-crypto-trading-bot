"""Model integrity verification via HMAC-SHA256 over file bytes.

Phase A8 (2026-05-12). Threat model: an attacker with write access to
`models/` could replace a `.joblib` file with one that runs arbitrary
code on `joblib.load()`. Phase A7 already locked `torch.load` to
`weights_only=True`, but joblib loads remain a pickle deserialization
RCE vector. This module signs every saved model with HMAC-SHA256
(keyed by `MODEL_MANIFEST_KEY`) and verifies the signature before
every load. A tampered file fails the HMAC check and the load is
refused.

Policy (fail-open when key unset, fail-closed on mismatch or malformed):
  - MODEL_MANIFEST_KEY unset       -> log WARNING once, bypass all checks.
  - Key set, manifest missing      -> bootstrap, allow load (next save signs).
  - Key set, no entry for this file -> log WARNING (untracked), allow load.
  - Key set, malformed manifest entry -> log CRITICAL, raise ModelIntegrityError.
  - Key set, HMAC mismatch          -> log CRITICAL, raise ModelIntegrityError.
  - Path is a symlink              -> log CRITICAL, raise ModelIntegrityError
    (a swapped symlink defeats untracked-entry allow; reject across the board).

Manifest lives at `models/manifest.json` (atomic via safe_json).

API:
  - `verify_model_or_raise(path)` — path-based verify (TOCTOU window exists
    between verify and the subsequent load by a third-party library that
    re-opens the path; use for libraries that demand a path).
  - `verify_and_load_bytes(path)` -> bytes — preferred: opens the file
    once, returns bytes after HMAC pass. Caller wraps in `io.BytesIO`
    and passes to `joblib.load` / `torch.load`. Closes the TOCTOU race.
  - `sign_model(path)` — call after `joblib.dump` / `torch.save`.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from src.utils.safe_json import read_json, write_json


logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1
_HMAC_LEN_HEX = 64  # sha256 hex digest length
_CHUNK_SIZE = 1 << 20  # 1 MiB streaming reads
_MIN_RECOMMENDED_KEY_LEN = 32

_KEY_ENV = "MODEL_MANIFEST_KEY"
_MANIFEST_REL_PATH = os.path.join("models", "manifest.json")

# Sentinel distinguishing "never read env" from "read env, key absent".
_UNSET: object = object()

_state_lock = threading.Lock()
# _cached_key holds: _UNSET (never read), None (read, absent), or bytes (read, present).
_cached_key: object = _UNSET
_key_warning_emitted = False
_short_key_warning_emitted = False


class ModelIntegrityError(RuntimeError):
    """Raised when a model file fails HMAC integrity verification."""


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _manifest_path() -> str:
    return os.path.join(_project_root(), _MANIFEST_REL_PATH)


def _load_key() -> Optional[bytes]:
    """Read MODEL_MANIFEST_KEY from env, normalize to 32 bytes via SHA-256.

    Cached per-process. Distinguishes "never checked" (_UNSET) from
    "checked, absent" (None) so the env is only read once, and the
    one-shot WARNING fires exactly once even when key stays unset.
    Returns None when unset (callers must treat as fail-open).
    """
    global _cached_key, _key_warning_emitted, _short_key_warning_emitted
    with _state_lock:
        if _cached_key is not _UNSET:
            return _cached_key  # type: ignore[return-value]
        raw = (os.environ.get(_KEY_ENV) or "").strip()
        if not raw:
            if not _key_warning_emitted:
                logger.warning(
                    "%s unset — model integrity checks bypassed (fail-open). "
                    "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\" "
                    "and set %s in .env to enforce.",
                    _KEY_ENV, _KEY_ENV,
                )
                _key_warning_emitted = True
            _cached_key = None
            return None
        if len(raw) < _MIN_RECOMMENDED_KEY_LEN and not _short_key_warning_emitted:
            logger.warning(
                "%s is only %d chars; use >=%d random chars (token_urlsafe(48) "
                "recommended) for adequate HMAC entropy.",
                _KEY_ENV, len(raw), _MIN_RECOMMENDED_KEY_LEN,
            )
            _short_key_warning_emitted = True
        _cached_key = hashlib.sha256(raw.encode("utf-8")).digest()
        return _cached_key  # type: ignore[return-value]


def _rel_key(path: str) -> str:
    """Manifest key = path relative to project root, forward slashes."""
    abs_path = os.path.abspath(path)
    root = _project_root()
    try:
        rel = os.path.relpath(abs_path, root)
    except ValueError:
        rel = abs_path
    return rel.replace(os.sep, "/")


def _reject_symlink(path: str) -> None:
    """Raise if `path` is a symlink. A swapped symlink could otherwise reach
    the 'untracked entry -> allow' branch and bypass integrity entirely."""
    if os.path.islink(path):
        rel = _rel_key(path)
        logger.critical(
            "Model integrity: refusing to load symlink %s — symlinks are "
            "rejected to prevent target-swap bypass.", rel,
        )
        raise ModelIntegrityError(f"refusing symlink: {rel}")


def _hmac_bytes(data: bytes, key: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def _hmac_file_streaming(path: str, key: bytes) -> str:
    h = hmac.new(key, digestmod=hashlib.sha256)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    data = read_json(_manifest_path(), default=None)
    if not isinstance(data, dict):
        return {"version": _MANIFEST_VERSION, "entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {"version": _MANIFEST_VERSION, "entries": entries}


def _save_manifest(manifest: dict) -> None:
    write_json(_manifest_path(), manifest, indent=2)


def _check_against_manifest(rel: str, actual_hmac: str) -> None:
    """Compare actual HMAC against the manifest entry. Raises on mismatch
    or malformed entry. Returns silently on missing-entry-allow path."""
    manifest = _load_manifest()
    entry = manifest["entries"].get(rel)
    if entry is None:
        logger.warning(
            "Model integrity: no manifest entry for %s (untracked, allowing). "
            "It will be signed on the next training save.", rel,
        )
        return
    expected = entry.get("hmac_sha256")
    if not isinstance(expected, str) or len(expected) != _HMAC_LEN_HEX:
        logger.critical(
            "Model integrity: malformed manifest entry for %s (missing or "
            "short hmac). Refusing to load — manifest may be corrupted "
            "or tampered.", rel,
        )
        raise ModelIntegrityError(f"malformed manifest entry for {rel}")
    if not hmac.compare_digest(expected.lower(), actual_hmac.lower()):
        logger.critical(
            "Model integrity FAILURE for %s: HMAC mismatch (expected=%s..., actual=%s...). "
            "Refusing to load.", rel, expected[:12], actual_hmac[:12],
        )
        raise ModelIntegrityError(f"HMAC mismatch for {rel}")


def verify_model_or_raise(path: str) -> None:
    """Verify model file at `path` against the manifest. See module docstring for policy.

    Note: TOCTOU — the file is read once for HMAC, then the caller opens
    it again. Prefer `verify_and_load_bytes` when the caller can deserialize
    from a buffer (joblib, torch). Use this entry point only when a third-
    party library requires a path (e.g., darts.models.TFTModel.load).
    """
    key = _load_key()
    if key is None:
        return  # fail-open
    if not os.path.isfile(path):
        # Defer file-not-found to caller's normal error handling.
        return
    _reject_symlink(path)
    actual = _hmac_file_streaming(path, key)
    _check_against_manifest(_rel_key(path), actual)


def verify_and_load_bytes(path: str) -> bytes:
    """Open `path` once, return bytes after HMAC verification.

    Eliminates the verify-then-load TOCTOU window. Caller wraps the
    result in `io.BytesIO` and passes to `joblib.load` / `torch.load`.

    Raises:
        FileNotFoundError: file does not exist (delegated to caller's
            existing error handling — same shape as `open` would raise).
        ModelIntegrityError: HMAC mismatch, malformed entry, or symlink.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    _reject_symlink(path)
    with open(path, "rb") as f:
        data = f.read()
    key = _load_key()
    if key is None:
        return data  # fail-open
    actual = _hmac_bytes(data, key)
    _check_against_manifest(_rel_key(path), actual)
    return data


def sign_model(path: str) -> bool:
    """Compute HMAC and store manifest entry for `path`. Idempotent.

    Returns True on success, False on skip (key unset or file missing).
    Refuses to sign a symlink (a signed symlink could be retargeted).
    """
    key = _load_key()
    if key is None:
        return False
    if not os.path.isfile(path):
        logger.warning("sign_model: file does not exist, skipping: %s", path)
        return False
    if os.path.islink(path):
        logger.error("sign_model: refusing to sign symlink: %s", path)
        return False
    digest = _hmac_file_streaming(path, key)
    rel = _rel_key(path)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = -1
    with _state_lock:
        manifest = _load_manifest()
        manifest["entries"][rel] = {
            "hmac_sha256": digest,
            "size_bytes": int(size),
            "signed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _save_manifest(manifest)
    return True


def _reset_for_tests() -> None:
    """Test-only: clear cached key + one-shot warning flags so env changes take effect."""
    global _cached_key, _key_warning_emitted, _short_key_warning_emitted
    with _state_lock:
        _cached_key = _UNSET
        _key_warning_emitted = False
        _short_key_warning_emitted = False


__all__ = [
    "ModelIntegrityError",
    "verify_model_or_raise",
    "verify_and_load_bytes",
    "sign_model",
]
