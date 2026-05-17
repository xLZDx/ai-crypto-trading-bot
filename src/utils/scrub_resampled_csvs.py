"""
scrub_resampled_csvs — one-shot fixer for the gap-row bug in PR-7.

Old `_write_csv_gz` wrote one row per period in the source date range. Where
the 1s archive had a gap (exchange downtime, listing pre-history), pandas
emitted NaN OHLC, which `to_csv` rendered as empty cells. Bot crashed on
`float('')`. PR-7 fixed the writer; this script repairs the existing
~150 files without re-running the 3-hour resample.

Usage:
    python -m src.utils.scrub_resampled_csvs                # scrub all
    python -m src.utils.scrub_resampled_csvs --dry-run      # report only
    python -m src.utils.scrub_resampled_csvs BTC_USDT_1h    # one file
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"


def _is_bad(row: dict) -> bool:
    for k in ("open", "high", "low", "close"):
        v = row.get(k, "")
        if v in ("", None) or v != v:   # empty or NaN
            return True
    return False


def scrub_one(path: Path, *, dry_run: bool = False) -> dict:
    """Scrub gap rows out of one .csv.gz. Returns counts."""
    if not path.exists():
        return {"path": str(path), "status": "missing", "rows": 0, "dropped": 0}
    kept = 0
    dropped = 0
    header: list[str] | None = None
    out_rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for row in reader:
            if _is_bad(row):
                dropped += 1
            else:
                out_rows.append(row)
                kept += 1
    if dropped == 0:
        return {"path": str(path), "status": "clean", "rows": kept, "dropped": 0}
    if dry_run:
        return {"path": str(path), "status": "would-scrub",
                "rows": kept, "dropped": dropped}
    # Atomic rewrite
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
        if not header:
            return {"path": str(path), "status": "empty", "rows": 0, "dropped": dropped}
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    os.replace(tmp, path)
    return {"path": str(path), "status": "scrubbed",
            "rows": kept, "dropped": dropped}


def scrub_all(*, pattern_filter: str | None = None,
              dry_run: bool = False) -> list[dict]:
    """Scrub every <SYM>_<tf>.csv.gz in data/raw/. Skips data/raw/historical/
    (those are the source 1s archives — don't touch)."""
    results: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.csv.gz")):
        if pattern_filter and pattern_filter not in path.stem:
            continue
        # Don't scrub the source 1s files — empties there are fine because
        # they get resampled away. Scrub only OHLCV timeframe outputs;
        # funding / metadata files have a different schema and would all
        # show as "bad" by the OHLC check (no OHLC columns).
        stem = path.stem.removesuffix(".csv")
        if stem.endswith("_1s") or stem.endswith("_funding"):
            continue
        t0 = time.time()
        r = scrub_one(path, dry_run=dry_run)
        r["elapsed_s"] = round(time.time() - t0, 2)
        print(f"  {r['status']:>11s} {path.name:<28s} kept={r['rows']:>8d} dropped={r['dropped']:>5d} in {r['elapsed_s']:>5.1f}s")
        results.append(r)
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Scrub gap rows from resampled CSVs")
    ap.add_argument("filter", nargs="?", default=None,
                    help="Optional substring filter (e.g. 'BTC' or '_1h')")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    print(f"[scrub] dir={RAW_DIR} dry_run={args.dry_run} filter={args.filter or '*'}")
    started = time.time()
    results = scrub_all(pattern_filter=args.filter, dry_run=args.dry_run)
    total_dropped = sum(r["dropped"] for r in results)
    total_rows = sum(r["rows"] for r in results)
    scrubbed = sum(1 for r in results if r["status"] in ("scrubbed", "would-scrub"))
    print(f"\n[scrub] done -- {len(results)} files, {scrubbed} needed scrubbing, "
          f"{total_dropped} gap rows removed ({total_rows} rows kept), "
          f"elapsed {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
