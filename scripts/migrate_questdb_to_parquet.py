"""
Optional one-shot QuestDB → ParquetClient data migrator.

Only useful if QuestDB happens to be running with data you want to
preserve. Skips gracefully when QuestDB is unreachable (the common case
once Phase 5 cleanup retires the daemon).

Usage:
    python -m scripts.migrate_questdb_to_parquet
    python -m scripts.migrate_questdb_to_parquet --table market_data
    python -m scripts.migrate_questdb_to_parquet --dry-run

Behavior:
- For each known table, count rows in QuestDB. If 0 (or table missing),
  skip with a one-line note.
- Otherwise stream rows in 50k batches via SELECT, then bulk-insert into
  ParquetClient.insert_rows(). Duplicates are NOT deduplicated — re-runs
  will accumulate. Use --dry-run first to verify counts.

Why this is optional:
  Route B is "no daemon" — most users will retire QuestDB without ever
  having put valuable data into it. The only data that actually matters
  on this project is the 48 GB of OHLCV history in data/parquet/, which
  is read by parquet_store.py (legacy path) and never went through
  QuestDB anyway.
"""
from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("migrate_questdb_to_parquet")


def _questdb_alive(host: str = "127.0.0.1", port: int = 9000) -> bool:
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/exec?query=SELECT%201"
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False
    except Exception:
        return False


def _qdb_query(sql: str, host: str = "127.0.0.1", port: int = 9000) -> list[dict]:
    """Tiny QuestDB REST wrapper — copied from the archived legacy client
    so this script doesn't depend on questdb_client.py (which is now a
    shim and won't talk to QuestDB)."""
    import json
    import urllib.parse
    qs = urllib.parse.urlencode({"query": sql, "limit": "0,1000000"})
    req = urllib.request.Request(f"http://{host}:{port}/exec?{qs}")
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        body = resp.read()
    data = json.loads(body)
    cols = [c["name"] for c in data.get("columns", [])]
    return [dict(zip(cols, row)) for row in data.get("dataset", [])]


def _coerce_qdb_row_for_parquet(row: dict[str, Any]) -> dict[str, Any]:
    """Translate one QuestDB row into ParquetClient-shaped form:
    - rename `timestamp` (QuestDB designated TS) → `ts` (Parquet schema)
    - parse ISO-string timestamps into tz-aware datetime
    """
    out = dict(row)
    if "timestamp" in out and "ts" not in out:
        out["ts"] = out.pop("timestamp")
    for k, v in list(out.items()):
        if k.endswith("_ts") or k == "ts":
            if isinstance(v, str):
                try:
                    out[k] = datetime.fromisoformat(
                        v.replace("Z", "+00:00")
                    )
                except Exception:
                    pass
    return out


def migrate_table(table: str, dry_run: bool = False, batch: int = 50_000) -> tuple[int, int]:
    """Returns (rows_in_questdb, rows_written)."""
    try:
        n_rows = _qdb_query(f"SELECT count() AS n FROM {table}")
        total = int(n_rows[0]["n"]) if n_rows else 0
    except Exception as exc:
        logger.info("  - %s: SKIP (table missing in QuestDB: %s)", table, exc)
        return (0, 0)

    if total == 0:
        logger.info("  - %s: 0 rows, skip", table)
        return (0, 0)

    if dry_run:
        logger.info("  - %s: %d rows (dry-run, would migrate)", table, total)
        return (total, 0)

    from src.database.parquet_client import get_client
    pc = get_client()
    written = 0
    for offset in range(0, total, batch):
        try:
            # QuestDB has no OFFSET keyword — uses `LIMIT lo,hi` for windowing.
            # Rows from a designated-timestamp table return in time order
            # without an explicit ORDER BY (and the schema column is named
            # `timestamp`, not `ts` — that's normalised in coercion).
            rows = _qdb_query(
                f"SELECT * FROM {table} LIMIT {offset},{offset + batch}"
            )
        except Exception as exc:
            logger.warning("  - %s: query failed at offset %d -- %s", table, offset, exc)
            break
        if not rows:
            break
        coerced = [_coerce_qdb_row_for_parquet(r) for r in rows]
        if pc.insert_rows(table, coerced):
            written += len(coerced)
        else:
            logger.warning("  - %s: insert failed at offset %d", table, offset)
            break
    pc.flush_all()
    logger.info("  - %s: %d / %d rows migrated", table, written, total)
    return (total, written)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--dry-run", action="store_true",
                        help="Probe row counts but don't write")
    parser.add_argument("--table", default=None,
                        help="Migrate just one table (default: all known)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not _questdb_alive(args.host, args.port):
        logger.info("QuestDB not reachable on %s:%s -- nothing to migrate. "
                    "(Skip the migration; this is normal once QuestDB has "
                    "been retired.)", args.host, args.port)
        return 0

    from src.database.parquet_client import _TABLES
    tables = [args.table] if args.table else list(_TABLES.keys())

    logger.info("Migrating %d table(s) from QuestDB -> ParquetClient%s",
                len(tables), " (dry-run)" if args.dry_run else "")
    grand_total = 0
    grand_written = 0
    for t in tables:
        total, written = migrate_table(t, dry_run=args.dry_run)
        grand_total += total
        grand_written += written

    logger.info("Done. Total: %d rows in QuestDB, %d migrated.",
                grand_total, grand_written)
    return 0 if (args.dry_run or grand_written == grand_total) else 1


if __name__ == "__main__":
    sys.exit(main())
