"""Dataset fingerprinting for Phase 9 Champion/Challenger baseline system.

Hashes *logical* data (column values), not file bytes, so fingerprints
survive Arrow re-encoding of identical data.

Key design:
- Streaming via iter_batches — never loads a full 48 GB DataFrame
- Fast-path: compare {path: (mtime, size)} before doing the full hash
- Cache writes are atomic (os.replace) to survive crashes mid-write
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _hash_parquet_file(path: Path) -> str:
    """Return SHA-256 of all column buffers in *path* using Arrow streaming."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    h = hashlib.sha256()
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=50_000):
        for col_name in batch.schema.names:
            col = batch.column(col_name)
            for buf in col.buffers():
                if buf is not None:
                    h.update(buf.to_pybytes())
    return h.hexdigest()


def _load_fingerprint_cache(cache_path: Path) -> dict[str, Any]:
    """Return cached fingerprint dict, or empty dict if missing/corrupt."""
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_fingerprint_cache_atomic(cache_path: Path, data: dict[str, Any]) -> None:
    """Write *data* to *cache_path* atomically to survive crashes."""
    tmp = str(cache_path) + ".tmp"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, str(cache_path))


def compute_dataset_fingerprint(
    parquet_dir: Path,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Compute or retrieve a logical fingerprint for all parquet files under
    *parquet_dir*.

    Uses fast-path: if every file's (mtime, size) matches the cache, returns
    the cached fingerprint without re-hashing. Full hash is done only when
    any file changed.

    Returns a dict with:
        schema_hash  — SHA-256 of all {path: hash} values
        file_count   — total parquet files
        total_bytes  — sum of file sizes
        computed_at  — Unix timestamp of this computation
    """
    files = sorted(parquet_dir.rglob("*.parquet"))
    if not files:
        return {
            "schema_hash": hashlib.sha256(b"").hexdigest(),
            "file_count": 0,
            "total_bytes": 0,
            "computed_at": time.time(),
        }

    # Build mtime+size manifest
    current_manifest: dict[str, tuple[float, int]] = {}
    total_bytes = 0
    for f in files:
        st = f.stat()
        current_manifest[str(f)] = (st.st_mtime, st.st_size)
        total_bytes += st.st_size

    # Fast-path: cache hit when manifest unchanged
    cache: dict[str, Any] = {}
    if cache_path is not None:
        cache = _load_fingerprint_cache(cache_path)
        if cache.get("manifest") == current_manifest:
            return cache["fingerprint"]

    # Full hash
    per_file: dict[str, str] = {str(f): _hash_parquet_file(f) for f in files}
    combined = hashlib.sha256(
        json.dumps(per_file, sort_keys=True).encode()
    ).hexdigest()

    fingerprint: dict[str, Any] = {
        "schema_hash": combined,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "computed_at": time.time(),
    }

    if cache_path is not None:
        _save_fingerprint_cache_atomic(
            cache_path,
            {"manifest": current_manifest, "fingerprint": fingerprint},
        )

    return fingerprint
