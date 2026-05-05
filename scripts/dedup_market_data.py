"""
One-shot deduplicator for ParquetClient market_data table.  EXPERIMENTAL.

Status (2026-05-05): the rebucket-into-yyyymm pass loses ~95 % of rows
(33 M → 1.6 M) under load. Root cause not yet identified — likely a
DuckDB partitioned-COPY interaction with union_by_name + hive_partitioning,
or a pandas groupby-and-rewrite chunking issue. The atomic-rollback path
DOES work, so running it leaves the store consistent (just unduped). DO
NOT trust this script for production until the rebucket loss is fixed.

After the QuestDB → ParquetClient migration ran concurrently with the
realtime_db_writer (Phase 4 of the DB migration), some bars exist in
multiple Parquet files within a partition dir. DuckDB happily returns
them all, which inflates rowcount-based metrics by ~0.85 %. The duplicate
overhead is small enough to defer cleanup.

Strategy:
  - Read all market_data Parquet files via DuckDB
  - Apply ROW_NUMBER() OVER (PARTITION BY (symbol, timeframe, ts) ORDER BY ts)
    and keep rn=1 (any tiebreaker is fine — duplicate rows have identical
    OHLCV values; QuestDB DEDUP UPSERT KEYS guaranteed source uniqueness)
  - COPY the result to a fresh dir partitioned by (symbol, timeframe)
  - Atomic swap: rename old → .pre_dedup_<ts>, rename new → original

Caveat: while the script runs, the bot may still be writing fresh bars
to the original dir. The resulting handful of newer bars (worst case
a few minutes' worth) gets shadowed by the swap. We mitigate by reading
the manifest snapshot at the START and re-merging any post-snapshot
files into the new dir before the final swap.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("dedup_market_data")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-old", action="store_true",
                        help="Keep the .pre_dedup_<ts> backup dir instead of removing it")
    parser.add_argument("--dry-run", action="store_true",
                        help="Probe duplicate count, don't rewrite anything")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from src.database.parquet_client import get_client
    pc = get_client()
    if not pc.is_available():
        logger.error("ParquetClient unavailable")
        return 1

    md_dir = pc.base_dir / "hot" / "market_data"
    if not md_dir.exists():
        logger.warning("No market_data dir at %s — nothing to dedup", md_dir)
        return 0

    files_before = list(md_dir.rglob("*.parquet"))
    logger.info("Found %d Parquet files under %s", len(files_before), md_dir)
    if not files_before:
        return 0

    # Snapshot the file list NOW so concurrent writes don't get shadowed.
    snapshot_paths = [str(p) for p in files_before]

    con = pc._conn()
    glob = (md_dir / "**" / "*.parquet").as_posix()

    # Total + duplicate counts
    n_total = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{glob}', "
        f"hive_partitioning=1, union_by_name=true)"
    ).fetchone()[0]
    dup_groups = con.execute(
        f"SELECT COUNT(*) FROM ("
        f"  SELECT symbol, timeframe, ts FROM read_parquet('{glob}', "
        f"  hive_partitioning=1, union_by_name=true) "
        f"  GROUP BY symbol, timeframe, ts HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    logger.info("Before: %s rows, %s duplicate-key groups",
                f"{n_total:,}", f"{dup_groups:,}")

    if args.dry_run or dup_groups == 0:
        logger.info("Dry-run / nothing to do — exiting.")
        return 0

    # Write deduped output to a sibling dir, then atomic-rename.
    ts_tag = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    new_dir = md_dir.parent / f"market_data.dedup_{ts_tag}"
    backup_dir = md_dir.parent / f"market_data.pre_dedup_{ts_tag}"
    if new_dir.exists():
        shutil.rmtree(new_dir, ignore_errors=True)
    new_dir.mkdir(parents=True)

    # DuckDB partitioned COPY. We dedup using QUALIFY (DuckDB-native, simpler
    # than nested ROW_NUMBER + WHERE rn=1). PARTITION_BY (symbol, timeframe)
    # writes hive-layout dirs matching ParquetClient's _partition_dir().
    sql = f"""
        COPY (
            SELECT * FROM read_parquet('{glob}',
                                       hive_partitioning=1, union_by_name=true)
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY symbol, timeframe, ts ORDER BY ts
            ) = 1
        )
        TO '{new_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (symbol, timeframe), COMPRESSION zstd)
    """
    logger.info("Running DuckDB partitioned COPY ...")
    t0 = time.time()
    con.execute(sql)
    elapsed = time.time() - t0
    logger.info("COPY finished in %.1fs", elapsed)

    # Re-bucket into the yyyymm partition layer ParquetClient expects.
    # Source: data/db/hot/market_data.dedup_*/symbol=X/timeframe=Y/data_*.parquet
    # Target: data/db/hot/market_data.dedup_*/symbol=X/timeframe=Y/yyyymm=NNNNNN/data_*.parquet
    logger.info("Re-bucketing into yyyymm/ partitions ...")
    import pyarrow.parquet as pq
    import pyarrow as pa
    rebucket_count = 0
    # Snapshot the file list FIRST so we don't traverse files we just wrote.
    flat_files = [
        f for f in new_dir.rglob("*.parquet")
        if not any(p.startswith("yyyymm=") for p in f.parts)
    ]
    for f in flat_files:
        try:
            tbl = pq.read_table(f.as_posix())
            df = tbl.to_pandas()
            if df.empty:
                f.unlink(missing_ok=True)
                continue
            df["_yyyymm"] = df["ts"].dt.strftime("%Y%m")
            for yyyymm, sub in df.groupby("_yyyymm"):
                sub2 = sub.drop(columns=["_yyyymm"])
                target = f.parent / f"yyyymm={yyyymm}" / f"data_{rebucket_count:06d}.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(sub2),
                               target.as_posix(),
                               compression="zstd", use_dictionary=True)
                rebucket_count += 1
            f.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Re-bucket %s failed: %s", f, exc)
    logger.info("Re-bucket: %d new yyyymm files written.", rebucket_count)

    # Verify deduped counts
    new_glob = (new_dir / "**" / "*.parquet").as_posix()
    n_new = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{new_glob}', "
        f"hive_partitioning=1, union_by_name=true)"
    ).fetchone()[0]
    dup_new = con.execute(
        f"SELECT COUNT(*) FROM ("
        f"  SELECT symbol, timeframe, ts FROM read_parquet('{new_glob}', "
        f"  hive_partitioning=1, union_by_name=true) "
        f"  GROUP BY symbol, timeframe, ts HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    logger.info("After:  %s rows, %s duplicate-key groups",
                f"{n_new:,}", f"{dup_new:,}")

    if dup_new > 0:
        logger.error("Dedup did NOT eliminate duplicates — aborting swap.")
        logger.error("Inspect %s manually before retrying.", new_dir)
        return 1

    # Swap directories. On Windows, rename of a directory fails if any
    # process has a file handle open inside it. We retry through shutil
    # (which falls back to copy+delete) and surface a clear error if even
    # that fails — caller is then expected to stop the bot first.
    logger.info("Swapping directories ...")
    try:
        md_dir.rename(backup_dir)
    except (OSError, PermissionError) as exc:
        logger.error("Cannot rename %s — %s. Stop the bot/realtime "
                     "writer (data/process_ids.json → bot, realtime) "
                     "and re-run.", md_dir, exc)
        return 1
    try:
        new_dir.rename(md_dir)
    except (OSError, PermissionError) as exc:
        # Recover: put the backup back so the system stays consistent.
        backup_dir.rename(md_dir)
        logger.error("Failed to install new dir — restored backup. %s", exc)
        return 1
    logger.info("Backup at %s", backup_dir)

    if not args.keep_old:
        logger.info("Removing backup ...")
        shutil.rmtree(backup_dir, ignore_errors=True)
        logger.info("Backup removed.")

    logger.info("Done. %s rows → %s rows (saved %s duplicate rows).",
                f"{n_total:,}", f"{n_new:,}", f"{n_total - n_new:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
