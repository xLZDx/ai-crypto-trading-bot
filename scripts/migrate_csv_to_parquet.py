"""
Migrate OHLCV + funding CSV.gz files from data/raw/ to data/parquet/.

Writes to: data/parquet/{SYM}/{tf}/yyyymm=YYYY-MM/data_0.parquet
Merges with existing parquet (dedup on timestamp, keep last).

Skips 1s files (already archived to Google Drive).

Usage:
    python scripts/migrate_csv_to_parquet.py
    python scripts/migrate_csv_to_parquet.py --dry-run
    python scripts/migrate_csv_to_parquet.py --symbols BTC_USDT ETH_USDT
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR  = PROJECT_ROOT / "data" / "parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OHLCV_COLS   = ["timestamp", "open", "high", "low", "close", "volume",
                 "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote"]
FUNDING_COLS = ["timestamp", "funding_rate"]

# Timeframes to migrate (skip 1s — already on Google Drive)
_TF_PATTERN = re.compile(
    r"^([A-Z0-9]+_USDT)_(1m|5m|15m|1h|4h|1d|1w|1mo|funding)\.csv\.gz$"
)


def _read_csv_gz(path: Path, is_funding: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    expected = FUNDING_COLS if is_funding else OHLCV_COLS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        log.warning("  %s missing columns %s -- skipping", path.name, missing)
        return pd.DataFrame()
    return df[expected].sort_values("timestamp").reset_index(drop=True)


def _write_partition(df: pd.DataFrame, dest_base: Path, dry_run: bool) -> int:
    """Write df split by YYYY-MM to dest_base/yyyymm=YYYY-MM/data_0.parquet.
    Merges with existing. Returns number of new rows written."""
    df = df.copy()
    df["_ym"] = df["timestamp"].dt.strftime("%Y-%m")
    total_new = 0

    for ym, chunk in df.groupby("_ym"):
        chunk = chunk.drop(columns=["_ym"]).reset_index(drop=True)
        part_dir = dest_base / f"yyyymm={ym}"
        out_file  = part_dir / "data_0.parquet"

        if not dry_run:
            part_dir.mkdir(parents=True, exist_ok=True)

        existing_rows = 0
        if out_file.exists():
            old = pd.read_parquet(out_file)
            old["timestamp"] = pd.to_datetime(old["timestamp"])
            before = len(chunk)
            chunk = pd.concat([old, chunk], ignore_index=True)
            chunk = (chunk
                     .sort_values("timestamp")
                     .drop_duplicates(subset=["timestamp"], keep="last")
                     .reset_index(drop=True))
            existing_rows = len(old)
            new_rows = len(chunk) - existing_rows
        else:
            new_rows = len(chunk)

        total_new += new_rows
        if not dry_run:
            chunk.to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")

    return total_new


def migrate(
    symbols: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    files = sorted(RAW_DIR.glob("*.csv.gz"))
    matched: list[tuple[Path, str, str, bool]] = []  # (path, sym, tf, is_funding)

    for f in files:
        m = _TF_PATTERN.match(f.name)
        if not m:
            continue
        sym, tf = m.group(1), m.group(2)
        if symbols and sym not in symbols:
            continue
        is_funding = (tf == "funding")
        matched.append((f, sym, tf, is_funding))

    log.info("Files to migrate: %d %s", len(matched), "(DRY RUN)" if dry_run else "")

    total_files  = 0
    total_rows   = 0
    skipped      = 0

    for path, sym, tf, is_funding in matched:
        log.info("  %s -> parquet/%s/%s/ ...", path.name, sym, tf)
        df = _read_csv_gz(path, is_funding)
        if df.empty:
            skipped += 1
            continue

        dest_base = PARQUET_DIR / sym / tf
        new_rows  = _write_partition(df, dest_base, dry_run)
        total_files += 1
        total_rows  += new_rows
        log.info("    %d rows in file, %d new rows written", len(df), new_rows)

    log.info(
        "Migration complete: %d files processed, %d skipped, %d new rows total %s",
        total_files, skipped, total_rows, "(DRY RUN -- nothing written)" if dry_run else "",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate CSV.gz OHLCV/funding to parquet")
    parser.add_argument("--dry-run",  action="store_true", help="Preview only, no writes")
    parser.add_argument("--symbols",  nargs="+", help="Limit to these symbols (e.g. BTC_USDT)")
    args = parser.parse_args()
    migrate(symbols=args.symbols, dry_run=args.dry_run)
