"""
questdb_client — back-compat shim (Phase 2 of QuestDB → ParquetClient migration).

The QuestDB Java daemon is being retired; this module now re-exports the
ParquetClient (DuckDB + partitioned Parquet) under the legacy name so the
11 callsites that still `from src.database.questdb_client import …` keep
working without per-file edits.

Surface preserved:
  - get_client()        → returns the ParquetClient singleton
  - QuestDBClient       → alias of ParquetClient (legacy class name)
  - _to_ns, _tag, _now_ns — pure ILP-format helpers some callers still
    use to emit ILP strings; ParquetClient.write_ilp() parses those.

Phase 5 cleanup deletes this file and renames the few remaining importers
to the canonical `from src.database.parquet_client import …`.

The legacy QuestDB-specific implementation lives at
src/database/_archived_questdb_client_legacy.py.bak in case rollback is
ever needed.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

# Re-export the new client so legacy `import get_client` works unchanged.
from src.database.parquet_client import (  # noqa: F401
    ParquetClient as _ParquetClient,
    get_client,
)

# Legacy class alias. Tests / type hints that reference QuestDBClient
# resolve to the new ParquetClient. Method surface is identical.
QuestDBClient = _ParquetClient


# ── Pure ILP-format helpers (preserved for callers that still emit ILP) ──

def _to_ns(ts) -> int | None:
    """Convert various timestamp forms to nanoseconds-since-epoch (UTC)."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1e9)
    if isinstance(ts, (int, float)):
        if ts < 1e12:
            return int(ts * 1e9)
        elif ts < 1e15:
            return int(ts * 1e6)
        return int(ts)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S.%f+00:00"):
            try:
                dt = datetime.strptime(ts.replace("+00:00", ""),
                                       fmt.replace("+00:00", ""))
                return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e9)
            except ValueError:
                continue
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp() * 1e9)
    return None


def _tag(v: Any) -> str:
    """Sanitise a value for use as an ILP tag (also our partition-safe form)."""
    return (str(v).replace(" ", "_")
                  .replace(",", "_")
                  .replace("=", "_")
                  .replace("/", "_"))


def _now_ns() -> int:
    return int(time.time() * 1e9)
