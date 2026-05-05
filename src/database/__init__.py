"""
Trading bot database layer.

Backed by ParquetClient (DuckDB + partitioned Parquet on data/db/).
File-based, no daemon — replaces QuestDB after the Phase 1–5 migration
(commits 43db156…).

Quick start:
    # No setup required. The store is implicit; first write creates the
    # data/db/ tree. To verify health from a Python REPL:
    python -c "from src.database import get_client; print(get_client().is_available())"

    # CSV.gz bulk ingest (legacy path, still works through ParquetClient):
    python -m src.database.ingest_pipeline --symbol BTC/USDT

    # Optional one-shot QuestDB → ParquetClient backfill (only useful if
    # the QuestDB daemon happens to still be running with data):
    python -m scripts.migrate_questdb_to_parquet
"""
from src.database.parquet_client import ParquetClient, get_client

# Legacy alias preserved for callers that still reference the old class name.
QuestDBClient = ParquetClient

__all__ = ["ParquetClient", "QuestDBClient", "get_client"]
