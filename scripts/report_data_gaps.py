"""
Report data coverage gaps across timeframes.

Scans:
  • data/raw/                         (1m, 1h, 1d, funding CSV.gz)
  • data/raw/historical/              (spot_1s archive)
  • data/parquet/                     (already-migrated Hive partitions)

For every (symbol, timeframe) pair, reports whether the file is present,
its size, and whether it has been migrated to Parquet. Files smaller than
`THIN_THRESHOLD_KB` are flagged as likely incomplete (e.g. partial
re-download in progress).

Usage:
    python scripts/report_data_gaps.py
    python scripts/report_data_gaps.py --json > gaps.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
PARQUET_DIR    = PROJECT_ROOT / "data" / "parquet"
WATCHLIST_FILE = PROJECT_ROOT / "data" / "watchlist.json"

# Files smaller than this are almost certainly partial / corrupted / placeholder.
THIN_THRESHOLD_KB = 50

# (timeframe id, suffix-in-data/raw/, where-historical-1s-lives)
TIMEFRAMES = [
    ("1m",      "_1m.csv.gz",      RAW_DIR),
    ("1h",      "_1h.csv.gz",      RAW_DIR),
    ("1d",      "_1d.csv.gz",      RAW_DIR),
    ("funding", "_funding.csv.gz", RAW_DIR),
    ("1s",      "_spot_1s.csv.gz", HISTORICAL_DIR),
]


def _watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_").upper()


def _all_known_symbols() -> set[str]:
    """Union of watchlist + every symbol observed in data/raw/."""
    syms = {_safe(s) for s in _watchlist()}
    if RAW_DIR.exists():
        for p in RAW_DIR.iterdir():
            for tf, suffix, _ in TIMEFRAMES:
                if p.name.endswith(suffix):
                    syms.add(p.name.removesuffix(suffix))
    if HISTORICAL_DIR.exists():
        for p in HISTORICAL_DIR.iterdir():
            for tf, suffix, _ in TIMEFRAMES:
                if p.name.endswith(suffix):
                    syms.add(p.name.removesuffix(suffix))
    return syms


def _parquet_status(symbol_safe: str, timeframe: str) -> tuple[bool, int, int]:
    """Return (migrated, n_partitions, total_bytes) for the parquet store."""
    sym_dir = PARQUET_DIR / symbol_safe
    if timeframe == "1s":
        # Legacy layout: yyyymm=* directly under sym_dir
        partitions = list(sym_dir.glob("yyyymm=*/*.parquet")) if sym_dir.exists() else []
        if not partitions:
            # Maybe already reorganised under 1s/
            partitions = list((sym_dir / "1s").glob("yyyymm=*/*.parquet")) if (sym_dir / "1s").exists() else []
    else:
        partitions = list((sym_dir / timeframe).glob("yyyymm=*/*.parquet")) if (sym_dir / timeframe).exists() else []
    n = len({p.parent for p in partitions})
    sz = sum(p.stat().st_size for p in partitions)
    return n > 0, n, sz


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="Output JSON instead of table")
    p.add_argument("--thin-kb", type=float, default=THIN_THRESHOLD_KB,
                   help=f"Flag CSVs below this size (default {THIN_THRESHOLD_KB})")
    args = p.parse_args()

    syms = sorted(_all_known_symbols())
    rows = []
    for sym in syms:
        for tf, suffix, src_dir in TIMEFRAMES:
            csv_path = src_dir / f"{sym}{suffix}"
            csv_exists = csv_path.exists()
            csv_kb = (csv_path.stat().st_size / 1024) if csv_exists else 0.0
            thin = csv_exists and csv_kb < args.thin_kb
            migrated, n_part, sz = _parquet_status(sym, tf)
            rows.append({
                "symbol":     sym,
                "timeframe":  tf,
                "csv":        str(csv_path) if csv_exists else None,
                "csv_kb":     round(csv_kb, 1),
                "thin":       thin,
                "migrated":   migrated,
                "partitions": n_part,
                "parquet_mb": round(sz / 1e6, 2),
            })

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    # Table output
    print(f"{'SYMBOL':<14}{'TF':<8}{'CSV_KB':>10}  {'Thin':>5}  {'Mig':>5}  {'Parts':>6}  {'Parquet':>10}")
    print("-" * 80)
    for r in rows:
        flag_thin = "X" if r["thin"] else ""
        flag_mig  = "OK" if r["migrated"] else "—"
        csv_kb = f"{r['csv_kb']:>9.1f}" if r["csv"] else f"{'(none)':>10}"
        print(f"{r['symbol']:<14}{r['timeframe']:<8}{csv_kb}  {flag_thin:>5}  {flag_mig:>5}  "
              f"{r['partitions']:>6}  {r['parquet_mb']:>9.2f}M")

    # Summary
    n_thin = sum(1 for r in rows if r["thin"])
    n_missing = sum(1 for r in rows if not r["csv"])
    n_unmigrated = sum(1 for r in rows if r["csv"] and not r["migrated"])
    print("-" * 80)
    print(f"thin files: {n_thin}   missing entirely: {n_missing}   "
          f"present but un-migrated: {n_unmigrated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
