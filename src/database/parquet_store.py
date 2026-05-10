"""
Parquet Store — DuckDB-backed query layer over partitioned Parquet files.

This is the cold/historical data path for the institutional upgrade (Phase 0).
QuestDB stays as the hot/real-time path (last ~30 days). All 1-second tick
history lives here as monthly-partitioned Parquet, queried via DuckDB.

Layout on disk:
    data/parquet/{symbol}/{YYYY-MM}/data.parquet

API contract (designed to be swap-compatible with a future ClickHouse client):
    store = ParquetStore()
    store.ingest_csv("BTC_USDT_spot_1s.csv.gz", symbol="BTC/USDT")
    df = store.query("BTC/USDT", start="2025-01-01", end="2025-01-15")
    info = store.status()                  # size, freshness, partitions per symbol

The same `query(...)` signature will be implemented by `clickhouse_client.py`
when the migration trigger fires (Parquet > 500 GB or backtest > 5 min).
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger("parquet_store")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = PROJECT_ROOT / "data" / "parquet"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "historical"

DEFAULT_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass
class SymbolStatus:
    symbol: str
    partitions: int
    rows: int
    size_bytes: int
    earliest: str | None
    latest: str | None


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").upper()


# Supported timeframe identifiers. None preserves the legacy
# `{SYMBOL}/yyyymm=*/` layout (used by the in-flight 1-sec migration);
# everything else writes to `{SYMBOL}/{TIMEFRAME}/yyyymm=*/`.
SUPPORTED_TIMEFRAMES = ("1s", "1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M", "funding")


def _symbol_dir(base_dir: Path, symbol: str, timeframe: str | None = None) -> Path:
    sym = _safe_symbol(symbol)
    if timeframe is None:
        return base_dir / sym
    return base_dir / sym / timeframe


def _partition_glob(base_dir: Path, symbol: str, timeframe: str | None = None) -> str:
    """Hive-style glob.

    timeframe=None  ⇒  data/parquet/BTC_USDT/yyyymm=*/*.parquet         (legacy)
    timeframe='1m'  ⇒  data/parquet/BTC_USDT/1m/yyyymm=*/*.parquet
    """
    return (_symbol_dir(base_dir, symbol, timeframe) / "yyyymm=*" / "*.parquet").as_posix()


def _existing_months(sym_dir: Path) -> set[str]:
    """Read existing yyyymm partitions from disk."""
    if not sym_dir.exists():
        return set()
    return {
        d.name.split("=", 1)[1]
        for d in sym_dir.iterdir()
        if d.is_dir() and d.name.startswith("yyyymm=") and any(d.glob("*.parquet"))
    }


class ParquetStore:
    """Thin wrapper around DuckDB for partitioned-Parquet tick data."""

    def __init__(self, base_dir: Path | str = DEFAULT_BASE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._conn = None  # lazy

    # ── Connection ──────────────────────────────────────────────────────────

    def _conn_or_open(self):
        if self._conn is None:
            import duckdb
            self._conn = duckdb.connect(database=":memory:")
            self._conn.execute("PRAGMA threads=4")
            self._conn.execute("PRAGMA enable_progress_bar=false")
            # Bound memory and force disk-spill to D: drive (per project rule:
            # all cache/temp data on D: only). 6 GB is enough headroom for
            # sorting a year of 1-sec ticks from a 6 GB gzipped CSV without
            # OOMing; anything that exceeds it spills to temp_directory.
            temp_dir = (PROJECT_ROOT / "data" / "cache" / "duckdb_temp").as_posix()
            (PROJECT_ROOT / "data" / "cache" / "duckdb_temp").mkdir(parents=True, exist_ok=True)
            try:
                self._conn.execute("PRAGMA memory_limit='6GB'")
                self._conn.execute(f"PRAGMA temp_directory='{temp_dir}'")
            except Exception as exc:
                logger.warning("[ParquetStore] could not apply memory pragmas: %s", exc)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _reset_connection(self) -> None:
        """Drop and reopen — used after an OOM corrupts the connection state."""
        self.close()
        self._conn_or_open()

    # ── Ingest ──────────────────────────────────────────────────────────────

    def ingest_csv(
        self,
        csv_path: Path | str,
        symbol: str,
        timestamp_col: str = "timestamp",
        compression: str = "zstd",
        skip_existing: bool = True,
        timeframe: str | None = None,
    ) -> dict:
        """Convert a (gzipped) CSV of OHLCV ticks into Hive-partitioned Parquet.

        Layouts:
          timeframe=None  →  data/parquet/{symbol}/yyyymm=YYYY-MM/data_*.parquet
                              (legacy / used by in-flight 1-sec migration)
          timeframe='1m'  →  data/parquet/{symbol}/1m/yyyymm=YYYY-MM/data_*.parquet

        Single-pass: reads the CSV exactly once, writes all month partitions
        in one COPY statement. ~10× faster than per-month writes for large
        files (matters at the 34 GB scale).

        Returns: {symbol, timeframe, months_written, skipped_months, rows_total, out_dir}
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        conn = self._conn_or_open()
        sym = _safe_symbol(symbol)
        out_root = _symbol_dir(self.base_dir, symbol, timeframe)
        out_root.mkdir(parents=True, exist_ok=True)

        existing = _existing_months(out_root)
        skip_clause = ""
        if skip_existing and existing:
            not_in = ", ".join(f"'{m}'" for m in sorted(existing))
            skip_clause = f"AND strftime({timestamp_col}, '%Y-%m') NOT IN ({not_in})"

        csv_uri = csv_path.as_posix()
        out_uri = out_root.as_posix()
        # Note: no `ORDER BY` — Binance archive CSVs are already chronological.
        # Removing the sort eliminates the gigantic sort buffer that caused OOMs
        # on large files (ADA/ATOM/AVAX/BNB) and gives a 3-5× speedup on 1-sec
        # data. If a future input source delivers unsorted ticks, set the env
        # var `PARQUET_FORCE_SORT=1` to re-enable.
        force_sort = os.getenv("PARQUET_FORCE_SORT") == "1"
        order_clause = f"ORDER BY {timestamp_col}" if force_sort else ""
        try:
            conn.execute(
                f"""
                COPY (
                    SELECT *,
                           strftime({timestamp_col}, '%Y-%m') AS yyyymm
                    FROM read_csv_auto('{csv_uri}', header=True)
                    WHERE TRUE {skip_clause}
                    {order_clause}
                ) TO '{out_uri}' (
                    FORMAT 'parquet',
                    PARTITION_BY (yyyymm),
                    CODEC '{compression}',
                    OVERWRITE_OR_IGNORE
                )
                """
            )
        except Exception as exc:
            # Fallback for OOM on very large CSVs: discover months first,
            # then write each month independently. Slower but bounded memory.
            err = str(exc)
            if "Out of Memory" in err or "OutOfMemory" in err or "memory" in err.lower():
                logger.warning(
                    "[ParquetStore] %s OOM on single-pass — falling back to per-month writes",
                    sym,
                )
                try:
                    self._reset_connection()
                    conn = self._conn_or_open()
                    months = [
                        row[0] for row in conn.execute(
                            f"""
                            SELECT DISTINCT strftime({timestamp_col}, '%Y-%m') AS yyyymm
                            FROM read_csv_auto('{csv_uri}', header=True)
                            ORDER BY yyyymm
                            """
                        ).fetchall()
                        if row[0] and row[0] not in existing
                    ]
                    for mm in months:
                        target_dir = out_root / f"yyyymm={mm}"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target = target_dir / "data.parquet"
                        conn.execute(
                            f"""
                            COPY (
                                SELECT *
                                FROM read_csv_auto('{csv_uri}', header=True)
                                WHERE strftime({timestamp_col}, '%Y-%m') = '{mm}'
                                ORDER BY {timestamp_col}
                            ) TO '{target.as_posix()}' (FORMAT 'parquet', CODEC '{compression}')
                            """
                        )
                        logger.info("[ParquetStore] %s yyyymm=%s ingested (per-month)", sym, mm)
                except Exception as exc2:
                    raise RuntimeError(f"Failed to ingest {csv_path.name} (fallback): {exc2}") from exc2
            else:
                raise RuntimeError(f"Failed to ingest {csv_path.name}: {exc}") from exc

        # Determine what we just wrote vs what was already there
        all_now = _existing_months(out_root)
        new_months = sorted(all_now - existing)

        # Total rows now in the symbol (post-ingest)
        glob = _partition_glob(self.base_dir, symbol, timeframe)
        rows_total = 0
        if any(out_root.glob("yyyymm=*/*.parquet")):
            rows_total = int(
                conn.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]
            )

        tf_tag = timeframe or "(legacy)"
        for m in new_months:
            logger.info("[ParquetStore] %s %s yyyymm=%s ingested", sym, tf_tag, m)

        return {
            "symbol":         symbol,
            "timeframe":      timeframe,
            "months_written": len(new_months),
            "skipped_months": len(existing),
            "rows_total":     rows_total,
            "out_dir":        str(out_root),
        }

    # ── Query ───────────────────────────────────────────────────────────────

    def query(
        self,
        symbol: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
        timeframe: str | None = None,
    ):
        """Query a time-range slice of OHLCV data. Returns a pandas DataFrame.

        `start` and `end` accept ISO-8601 strings or datetime objects (UTC).
        `timeframe`: e.g. '1s', '1m', '1d'. None = legacy layout.
        """
        conn = self._conn_or_open()
        sym = _safe_symbol(symbol)
        sym_dir = _symbol_dir(self.base_dir, symbol, timeframe)
        glob = _partition_glob(self.base_dir, symbol, timeframe)

        # If no partitions exist, return an empty DataFrame with the right schema.
        if not sym_dir.exists() or not any(sym_dir.glob("yyyymm=*/*.parquet")):
            import pandas as pd
            cols = list(columns) if columns else list(DEFAULT_COLUMNS)
            return pd.DataFrame(columns=cols)

        col_sql = ", ".join(columns) if columns else "*"
        where = []
        params: list = []
        if start is not None:
            where.append("timestamp >= ?")
            params.append(_to_dt(start))
        if end is not None:
            where.append("timestamp < ?")
            params.append(_to_dt(end))
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        limit_sql = f"LIMIT {int(limit)}" if limit else ""

        sql = f"""
            SELECT {col_sql}
            FROM read_parquet('{glob}')
            {where_sql}
            ORDER BY timestamp
            {limit_sql}
        """
        return conn.execute(sql, params).df()

    # ── Status / introspection ──────────────────────────────────────────────

    def list_symbols(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name.replace("_", "/")
            for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def list_timeframes(self, symbol: str) -> list[str]:
        """Return the timeframes available for a symbol (e.g. ['1s','1m','1d'])."""
        sym_dir = self.base_dir / _safe_symbol(symbol)
        if not sym_dir.exists():
            return []
        return sorted(
            d.name for d in sym_dir.iterdir()
            if d.is_dir() and d.name in SUPPORTED_TIMEFRAMES
        )

    def symbol_status(self, symbol: str, timeframe: str | None = None) -> SymbolStatus:
        sym_dir = _symbol_dir(self.base_dir, symbol, timeframe)
        if not sym_dir.exists():
            return SymbolStatus(symbol, 0, 0, 0, None, None)

        partition_dirs = sorted(
            d for d in sym_dir.iterdir() if d.is_dir() and d.name.startswith("yyyymm=")
        )
        files = sorted(sym_dir.glob("yyyymm=*/*.parquet"))
        size_bytes = sum(f.stat().st_size for f in files)
        if not files:
            return SymbolStatus(symbol, 0, 0, 0, None, None)

        conn = self._conn_or_open()
        glob = _partition_glob(self.base_dir, symbol, timeframe)

        # Pick the time column. News partitions store `published_at`; the
        # rest store `timestamp`. Probe the schema first to keep this generic.
        try:
            cols = conn.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{glob}') LIMIT 1"
            ).fetchall()
            col_names = {c[0] for c in cols}
        except Exception:
            col_names = set()
        time_col = ('timestamp' if 'timestamp' in col_names
                    else ('published_at' if 'published_at' in col_names else None))

        if time_col is None:
            return SymbolStatus(
                symbol=symbol, partitions=len(partition_dirs),
                rows=0, size_bytes=int(size_bytes),
                earliest=None, latest=None,
            )

        try:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS rows,
                       MIN({time_col}) AS earliest,
                       MAX({time_col}) AS latest
                FROM read_parquet('{glob}')
                """
            ).fetchone()
        except Exception:
            row = None

        if row is None:
            return SymbolStatus(
                symbol=symbol, partitions=len(partition_dirs),
                rows=0, size_bytes=int(size_bytes),
                earliest=None, latest=None,
            )
        rows, earliest, latest = row
        return SymbolStatus(
            symbol=symbol,
            partitions=len(partition_dirs),
            rows=int(rows or 0),
            size_bytes=int(size_bytes),
            earliest=str(earliest) if earliest else None,
            latest=str(latest) if latest else None,
        )

    def status(self) -> dict:
        """Cluster-wide status — used by the FastAPI control plane."""
        symbols = self.list_symbols()
        per_symbol = []
        total_rows = 0
        total_size = 0
        for s in symbols:
            st = self.symbol_status(s)
            per_symbol.append(st.__dict__)
            total_rows += st.rows
            total_size += st.size_bytes
        return {
            "base_dir":   str(self.base_dir),
            "symbols":    len(symbols),
            "rows_total": total_rows,
            "size_bytes": total_size,
            "size_gb":    round(total_size / 1e9, 3),
            "per_symbol": per_symbol,
            "as_of":      datetime.now(timezone.utc).isoformat(),
        }

    # ── Utility: drop a symbol (for re-ingestion) ──────────────────────────

    def drop_symbol(self, symbol: str) -> bool:
        sym_dir = self.base_dir / _safe_symbol(symbol)
        if not sym_dir.exists():
            return False
        shutil.rmtree(sym_dir)
        return True


def _to_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    # Accept "2025-01-01" or full ISO
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


# ─── Singleton helper (matches questdb_client style) ─────────────────────────

_store_instance: ParquetStore | None = None


def get_store() -> ParquetStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = ParquetStore()
    return _store_instance


__all__ = ["ParquetStore", "SymbolStatus", "get_store", "DEFAULT_BASE_DIR"]
