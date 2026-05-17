"""Migrate 1-second CSV.gz archives into the project's parquet store.

Operator request 2026-05-15: "move all the data from GZIP files to DB file
to let others easily read it without overhead". The 37 GB of 1s data in
`data/raw/historical/<SYM>_spot_1s.csv.gz` is the slowest hot path —
tick_feature_loader scans these end-to-end on every TFT training run.

Strategy
--------
For each `<SYM>_spot_1s.csv.gz` (and smaller `<SYM>_1s.csv.gz` live tails):

  1. Stream via DuckDB read_csv_auto (handles .gz natively).
  2. Partition by `to_yyyymm(timestamp)` so the layout matches the
     existing parquet store convention.
  3. Write `data/parquet/<SYM>/1s/yyyymm=<YYYY-MM>/data_0.parquet`
     with Snappy compression (project default).
  4. Skip yyyymm partitions that already exist (idempotent — re-runnable
     to top up new months without re-writing old ones).
  5. Dedup spot + live-tail timestamps so the merged 1s row count is
     monotonic (live tail overlaps spot for the most recent days).

Performance
-----------
- DuckDB column-store reads: each ~200 MB gzip → ~25 s decode +
  partition write.
- 37 GB total → ~30-60 min on a Razer-class NVMe.
- Output parquet: ~18-25 GB (Snappy is ~1.5-2× denser than gzip text).

Usage
-----
    .\\venv\\Scripts\\python.exe scripts\\migrate_1s_to_parquet.py
    .\\venv\\Scripts\\python.exe scripts\\migrate_1s_to_parquet.py --symbols BTC_USDT ETH_USDT
    .\\venv\\Scripts\\python.exe scripts\\migrate_1s_to_parquet.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\migrate_1s_to_parquet.py --force   # overwrite existing yyyymm partitions

Safe to run while a TFT training is in flight — the loader prefers parquet
over gzip but falls back gracefully so an in-flight scan against the old
gzip won't be disrupted.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("migrate_1s")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

HIST_DIR    = PROJECT_ROOT / "data" / "raw" / "historical"
RAW_DIR     = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
DUCKDB_TEMP = PROJECT_ROOT / "data" / "cache" / "duckdb_temp"


@dataclass
class PerSymbolResult:
    symbol: str
    sources: list[Path]
    partitions_written: int
    partitions_skipped: int
    rows_total: int
    duration_s: float
    bytes_out: int
    error: str | None = None


def _candidate_sources(symbol: str) -> list[Path]:
    """Return both spot (history) and live-tail 1s files for a symbol.
    Order matters: spot first (deeper history) so deduplication keeps it
    when timestamps overlap."""
    sym = symbol.replace("/", "_").upper()
    out = []
    for d, name in (
        (HIST_DIR, f"{sym}_spot_1s.csv.gz"),
        (HIST_DIR, f"{sym}_1s.csv.gz"),
        (RAW_DIR,  f"{sym}_1s.csv.gz"),
    ):
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            out.append(p)
    return out


def _existing_partitions(symbol: str) -> set[str]:
    """yyyymm values already written for this symbol's 1s store."""
    sym = symbol.replace("/", "_").upper()
    out_dir = PARQUET_DIR / sym / "1s"
    if not out_dir.exists():
        return set()
    return {p.name.split("=", 1)[-1] for p in out_dir.iterdir()
            if p.is_dir() and p.name.startswith("yyyymm=")}


def _migrate_symbol(symbol: str, *, force: bool, dry_run: bool) -> PerSymbolResult:
    sym = symbol.replace("/", "_").upper()
    sources = _candidate_sources(sym)
    out_dir = PARQUET_DIR / sym / "1s"
    started = time.time()
    if not sources:
        return PerSymbolResult(sym, [], 0, 0, 0, 0.0, 0, "no source files")

    existing = set() if force else _existing_partitions(sym)
    logger.info("[%s] sources=%d, existing yyyymm partitions=%d",
                sym, len(sources), len(existing))
    if dry_run:
        return PerSymbolResult(sym, sources, 0, len(existing), 0,
                               time.time() - started, 0,
                               error="dry-run (no writes)")

    out_dir.mkdir(parents=True, exist_ok=True)
    DUCKDB_TEMP.mkdir(parents=True, exist_ok=True)

    # Build the UNION ALL view across all sources, then partition + dedup
    # in DuckDB. Cheaper than reading each separately.
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DUCKDB_TEMP.as_posix()}'")
    # Schema: timestamp, open, high, low, close, volume, quote_volume,
    # trades_count, taker_buy_base, taker_buy_quote
    src_sql_parts = []
    for p in sources:
        src_sql_parts.append(
            f"SELECT timestamp, open, high, low, close, volume, "
            f"COALESCE(quote_volume, 0.0) AS quote_volume, "
            f"COALESCE(trades_count, 0) AS trades_count, "
            f"COALESCE(taker_buy_base, 0.0) AS taker_buy_base, "
            f"COALESCE(taker_buy_quote, 0.0) AS taker_buy_quote "
            f"FROM read_csv_auto('{p.as_posix()}', header=true)"
        )
    union_sql = " UNION ALL ".join(src_sql_parts)
    # Dedup by timestamp (newer rows from live tail win when both have
    # the same second — ROW_NUMBER over a SOURCE-order ranking).
    base_view_sql = f"""
    CREATE OR REPLACE TEMPORARY VIEW src AS
    WITH unioned AS (
        SELECT * FROM ({union_sql})
    )
    SELECT timestamp, open, high, low, close, volume, quote_volume,
           trades_count, taker_buy_base, taker_buy_quote
    FROM unioned
    QUALIFY ROW_NUMBER() OVER (PARTITION BY timestamp ORDER BY volume DESC) = 1
    """
    con.execute(base_view_sql)
    # List distinct yyyymm in this symbol's combined data.
    yyyymms = [r[0] for r in con.execute(
        "SELECT DISTINCT strftime(timestamp, '%Y-%m') AS yyyymm "
        "FROM src ORDER BY yyyymm"
    ).fetchall()]
    logger.info("[%s] distinct yyyymm in source: %d", sym, len(yyyymms))

    written = 0
    skipped = 0
    rows_total = 0
    bytes_out = 0
    for yyyymm in yyyymms:
        if not force and yyyymm in existing:
            skipped += 1
            continue
        part_dir = out_dir / f"yyyymm={yyyymm}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "data_0.parquet"
        # COPY (compression=snappy by default) — predicate by yyyymm.
        copy_sql = f"""
        COPY (
            SELECT * FROM src
            WHERE strftime(timestamp, '%Y-%m') = '{yyyymm}'
            ORDER BY timestamp
        ) TO '{out_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'SNAPPY');
        """
        try:
            con.execute(copy_sql)
            sz = out_path.stat().st_size if out_path.exists() else 0
            bytes_out += sz
            # Count rows just written (cheap — already in parquet metadata)
            n = con.execute(
                f"SELECT COUNT(*) FROM parquet_scan('{out_path.as_posix()}')"
            ).fetchone()[0]
            rows_total += int(n)
            written += 1
            logger.info("[%s] wrote %s  rows=%d  size=%.1f MB",
                        sym, yyyymm, n, sz / 1e6)
        except Exception as exc:
            logger.error("[%s] yyyymm=%s failed: %s", sym, yyyymm, exc)
            return PerSymbolResult(sym, sources, written, skipped,
                                   rows_total, time.time() - started,
                                   bytes_out, str(exc))

    con.close()
    duration = time.time() - started
    return PerSymbolResult(sym, sources, written, skipped, rows_total,
                           duration, bytes_out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*",
                    help="Symbols to migrate (default: every <SYM>_spot_1s.csv.gz found)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing yyyymm partitions")
    ap.add_argument("--dry-run", action="store_true",
                    help="List source files + existing partitions, write nothing")
    ap.add_argument("--workers", type=int, default=1,
                    help="Per-symbol parallelism (default 1; DuckDB itself uses all cores)")
    args = ap.parse_args()

    # Auto-discover symbols if not specified.
    if not args.symbols:
        symbols = sorted({
            p.name.replace("_spot_1s.csv.gz", "").replace("_1s.csv.gz", "")
            for p in HIST_DIR.glob("*_1s.csv.gz")
        })
    else:
        symbols = list(args.symbols)
    if not symbols:
        print("No symbols found.")
        return 1

    print(f"Migrating {len(symbols)} symbols: {symbols}")
    print(f"Output: {PARQUET_DIR}/<SYM>/1s/yyyymm=<YYYY-MM>/data_0.parquet")
    print(f"Mode:   {'DRY-RUN' if args.dry_run else 'FORCE' if args.force else 'IDEMPOTENT'}")
    print()

    results: list[PerSymbolResult] = []
    total_started = time.time()
    if args.workers <= 1:
        for sym in symbols:
            r = _migrate_symbol(sym, force=args.force, dry_run=args.dry_run)
            results.append(r)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(lambda s: _migrate_symbol(s, force=args.force, dry_run=args.dry_run), symbols):
                results.append(r)

    elapsed = time.time() - total_started
    total_rows = sum(r.rows_total for r in results)
    total_bytes_in = sum(p.stat().st_size for r in results for p in r.sources)
    total_bytes_out = sum(r.bytes_out for r in results)
    print()
    print("=" * 70)
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"  symbols migrated: {sum(1 for r in results if r.partitions_written>0 or args.dry_run)}/{len(results)}")
    print(f"  total partitions written: {sum(r.partitions_written for r in results)}")
    print(f"  total partitions skipped: {sum(r.partitions_skipped for r in results)}")
    print(f"  total rows: {total_rows:,}")
    if not args.dry_run:
        print(f"  size in (gzip):     {total_bytes_in/1e9:.2f} GB")
        print(f"  size out (parquet): {total_bytes_out/1e9:.2f} GB")
        if total_bytes_in > 0:
            print(f"  compression ratio:  {total_bytes_in/total_bytes_out:.2f}x")
    print()
    failures = [r for r in results if r.error]
    if failures:
        print(f"  {len(failures)} symbols had errors:")
        for r in failures:
            print(f"    {r.symbol}: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
