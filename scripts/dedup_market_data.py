"""
One-shot deduplicator for ParquetClient market_data table.

After the QuestDB → ParquetClient migration ran concurrently with the
realtime_db_writer (Phase 4), some bars exist in multiple Parquet files
within a single partition dir. DuckDB happily returns them all, which
inflates rowcount-based metrics. This script rewrites each yyyymm
partition with deduplication on (ts) — bars within one (symbol, timeframe,
yyyymm) partition are unique by ts.

Strategy: per-partition in-place rewrite.

For each yyyymm partition dir under data/db/hot/market_data/:
  1. If only one .parquet file → already deduplicated, skip
  2. Concat all parquet files in the dir
  3. Drop rows with NaN ts (defensive — pandas' silent NaN-group drop
     in groupby was the bug the previous DuckDB-COPY+rebucket version hit)
  4. drop_duplicates(subset=['ts'], keep='first')
  5. Write to <part>/_dedup_tmp.parquet
  6. After write succeeds, remove old files
  7. Rename _dedup_tmp.parquet → data.parquet

Properties:
- No directory restructuring (path layout unchanged)
- Per-partition unit of work — failure of one doesn't lose others
- Idempotent: re-runs see len(files)==1 partitions and skip
- BOT MUST BE STOPPED — concurrent writes to a partition we're
  rewriting would race the rename. We don't lock the dir.

Usage:
    python -m scripts.dedup_market_data
    python -m scripts.dedup_market_data --dry-run    # report only
    python -m scripts.dedup_market_data --table=trade_events  # any other table
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("dedup_market_data")


# Per-table dedup key. market_data uses ts because partition dir
# already constrains (symbol, timeframe). Other tables would extend this.
_DEDUP_KEYS: dict[str, list[str]] = {
    "market_data":   ["ts"],
    "model_signals": ["ts"],
    # extend as needed
}


def _walk_partitions(table_dir: Path) -> Iterable[Path]:
    """Yield each leaf-level partition dir (e.g. yyyymm=YYYYMM/) under
    the table dir. A leaf dir is one whose immediate children are *.parquet
    files, not subdirs."""
    if not table_dir.exists():
        return
    for d in table_dir.rglob("*"):
        if not d.is_dir():
            continue
        children = list(d.iterdir())
        has_parquet = any(c.suffix == ".parquet" for c in children if c.is_file())
        has_subdirs = any(c.is_dir() for c in children)
        if has_parquet and not has_subdirs:
            yield d


def dedup_partition(part_dir: Path, keys: list[str], dry_run: bool = False) -> tuple[int, int]:
    """Dedup one partition dir. Returns (rows_before, rows_after)."""
    files = sorted(part_dir.glob("*.parquet"))
    if len(files) <= 1:
        return (0, 0)  # nothing to do; caller treats as no-op

    import pandas as pd
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:
            logger.warning("  read %s failed — skipping file: %s", f, exc)
    if not frames:
        return (0, 0)
    df = pd.concat(frames, ignore_index=True)
    rows_before = len(df)

    # Defensive: drop rows missing any dedup key BEFORE drop_duplicates,
    # so we know exactly what's being removed (pandas drop_duplicates
    # would treat NaN==NaN as identical → keep one of them randomly,
    # which is wrong if the underlying row payload differed).
    df_clean = df.dropna(subset=keys)
    n_dropped_nan = rows_before - len(df_clean)
    df_dedup = df_clean.drop_duplicates(subset=keys, keep="first")
    rows_after = len(df_dedup)

    if dry_run:
        return (rows_before, rows_after)

    if n_dropped_nan > 0:
        logger.info("  %s: dropped %d rows with NaN in dedup keys",
                    part_dir.name, n_dropped_nan)

    # Write to a tmp file, verify, swap.
    tmp = part_dir / "_dedup_tmp.parquet"
    if tmp.exists():
        tmp.unlink()
    try:
        df_dedup.to_parquet(tmp, compression="zstd", index=False)
    except Exception as exc:
        logger.error("  write %s failed: %s — partition unchanged", tmp, exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return (rows_before, rows_before)

    # Verify the temp file is readable + has the right rowcount before
    # nuking originals.
    try:
        verify = pd.read_parquet(tmp)
        if len(verify) != rows_after:
            raise RuntimeError(
                f"verify mismatch: wrote {rows_after} rows but file has {len(verify)}"
            )
    except Exception as exc:
        logger.error("  verify %s failed: %s — partition unchanged", tmp, exc)
        tmp.unlink(missing_ok=True)
        return (rows_before, rows_before)

    # Drop originals + rename tmp → data.parquet (or pick first non-collision name).
    target = part_dir / "data.parquet"
    for f in files:
        try:
            f.unlink()
        except Exception as exc:
            logger.warning("  could not remove %s: %s", f, exc)
    try:
        tmp.rename(target)
    except Exception as exc:
        logger.error("  rename failed: %s — leaving %s in place", exc, tmp)
        return (rows_before, rows_after)

    return (rows_before, rows_after)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report duplicate counts; don't rewrite anything")
    parser.add_argument("--table", default="market_data",
                        help="Which table to dedup (default: market_data)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    keys = _DEDUP_KEYS.get(args.table)
    if not keys:
        logger.error("No dedup keys defined for table %s — add to _DEDUP_KEYS",
                     args.table)
        return 1

    from src.database.parquet_client import get_client, _TABLES
    pc = get_client()
    if not pc.is_available():
        logger.error("ParquetClient unavailable")
        return 1
    if args.table not in _TABLES:
        logger.error("Unknown table %s", args.table)
        return 1

    table_dir = pc._table_dir(args.table)
    if not table_dir.exists():
        logger.warning("No %s dir at %s — nothing to dedup", args.table, table_dir)
        return 0

    partitions = list(_walk_partitions(table_dir))
    logger.info("Scanning %d partitions under %s",
                len(partitions), table_dir.relative_to(PROJECT_ROOT))

    total_before = 0
    total_after = 0
    n_changed = 0
    n_partitions_with_dupes = 0
    t0 = time.time()
    for part in partitions:
        before, after = dedup_partition(part, keys, dry_run=args.dry_run)
        if before == 0 and after == 0:
            continue   # single-file partition, nothing to do
        total_before += before
        total_after += after
        if after < before:
            n_partitions_with_dupes += 1
            n_changed += 1
            if args.dry_run:
                logger.info("  %s: %d → %d (would drop %d)",
                            part.relative_to(table_dir),
                            before, after, before - after)
            else:
                logger.info("  %s: %d → %d (dropped %d)",
                            part.relative_to(table_dir),
                            before, after, before - after)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done in %.1fs. Partitions: %d total, %d had duplicates%s",
                elapsed, len(partitions), n_partitions_with_dupes,
                " (dry-run; nothing changed)" if args.dry_run else "")
    logger.info("Rows: %s -> %s (saved %s)%s",
                f"{total_before:,}", f"{total_after:,}",
                f"{total_before - total_after:,}",
                " (dry-run)" if args.dry_run else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
