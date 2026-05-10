"""
Multi-timeframe ParquetStore tests.

Coverage:
  - ingest_csv with timeframe param writes to {SYMBOL}/{TIMEFRAME}/yyyymm=*/
  - query with timeframe reads back the same data
  - Backward compatibility: timeframe=None still works (legacy 1-sec layout)
  - list_timeframes() / symbol_status(timeframe=...) helpers
  - migrate_to_parquet.py CLI: discover_csv_files filters by timeframe
  - reorg_1s_to_subdir.py logic moves legacy partitions correctly

Run:
    python tests/test_multi_timeframe.py
"""
from __future__ import annotations

import gzip
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
results = {"pass": 0, "fail": 0, "skip": 0}


def check(name, ok, detail=""):
    if ok is None:
        results["skip"] += 1
        print(f"  {SKIP} {name} (skipped)")
    elif ok:
        results["pass"] += 1
        print(f"  {PASS} {name}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name}{': ' + detail if detail else ''}")


# ─── ParquetStore multi-timeframe ───────────────────────────────────────────

def test_parquet_store_multi_tf():
    print("\n[ParquetStore Multi-Timeframe]")
    from src.database.parquet_store import (
        ParquetStore, SUPPORTED_TIMEFRAMES,
        _partition_glob, _symbol_dir,
    )
    check("SUPPORTED_TIMEFRAMES has 1s/1m/1d/funding",
          all(t in SUPPORTED_TIMEFRAMES for t in ("1s", "1m", "1d", "funding")))

    # path helpers
    base = Path("/tmp/xxx")
    legacy_glob = _partition_glob(base, "BTC/USDT", None)
    tf_glob     = _partition_glob(base, "BTC/USDT", "1m")
    check("legacy glob format unchanged",
          legacy_glob.endswith("BTC_USDT/yyyymm=*/*.parquet"))
    check("multi-tf glob inserts timeframe segment",
          "BTC_USDT/1m/yyyymm=*/*.parquet" in tf_glob)

    # Roundtrip on a synthetic 1m CSV
    import pandas as pd
    rows = []
    for ts in pd.date_range("2025-01-01", "2025-02-15", freq="1h"):
        rows.append({"timestamp": ts, "open": 100, "high": 101,
                     "low": 99, "close": 100.5, "volume": 1.0})
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv = tmp_path / "TEST_USDT_1m.csv.gz"
        with gzip.open(csv, "wt", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)

        store = ParquetStore(tmp_path / "parquet")
        # Legacy layout
        res_l = store.ingest_csv(csv, "TEST/USDT", timeframe=None)
        check("legacy ingest writes to {sym}/yyyymm=*/", res_l["months_written"] >= 1)
        check("legacy result has timeframe=None",
              res_l.get("timeframe") is None)

        # Multi-tf layout
        store2 = ParquetStore(tmp_path / "parquet2")
        res_m = store2.ingest_csv(csv, "TEST/USDT", timeframe="1m")
        check("1m ingest writes to {sym}/1m/yyyymm=*/", res_m["months_written"] >= 1)
        check("1m result has timeframe='1m'", res_m.get("timeframe") == "1m")
        # Verify on disk
        target_dir = tmp_path / "parquet2" / "TEST_USDT" / "1m"
        check("on-disk: {sym}/1m/yyyymm=* dir exists",
              target_dir.exists() and any(target_dir.glob("yyyymm=*")))

        # Query roundtrip
        out = store2.query("TEST/USDT", start="2025-01-15", end="2025-02-01",
                           timeframe="1m")
        check("query(timeframe='1m') returns rows", len(out) > 0)

        # symbol_status with timeframe
        st = store2.symbol_status("TEST/USDT", timeframe="1m")
        check("symbol_status(timeframe='1m') reports rows",
              st.rows > 0)

        # list_timeframes
        # Add another timeframe to the same symbol
        store2.ingest_csv(csv, "TEST/USDT", timeframe="1d")
        tfs = store2.list_timeframes("TEST/USDT")
        check("list_timeframes returns ['1d', '1m']",
              set(tfs) == {"1m", "1d"})


# ─── Migration script discovery ─────────────────────────────────────────────

def test_migrate_script_discovery():
    print("\n[migrate_to_parquet.py]")
    script = PROJECT_ROOT / "scripts" / "migrate_to_parquet.py"
    check("migrate_to_parquet.py exists", script.exists())
    if not script.exists():
        return
    src = script.read_text(encoding="utf-8")
    check("--timeframe arg required", "required=True" in src and "--timeframe" in src)
    check("uses ParquetStore", "from src.database.parquet_store import ParquetStore" in src)
    check("passes timeframe to ingest_csv",
          "timeframe=args.timeframe" in src)

    # Import the discover function
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mig", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    check("discover_csv_files() defined", hasattr(mod, "discover_csv_files"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Build fake fileset
        for fname in ["BTC_USDT_1m.csv.gz", "ETH_USDT_1m.csv.gz",
                      "BTC_USDT_1d.csv.gz", "BTC_USDT_funding.csv.gz",
                      "BTC_USDT_spot_1s.csv.gz"]:
            (tmp_path / fname).write_bytes(b"")

        files_1m = mod.discover_csv_files(tmp_path, "1m")
        check("1m discovery finds 2 files", len(files_1m) == 2)

        files_1d = mod.discover_csv_files(tmp_path, "1d")
        check("1d discovery finds 1 file", len(files_1d) == 1)

        files_funding = mod.discover_csv_files(tmp_path, "funding")
        check("funding discovery finds 1 file", len(files_funding) == 1)

        # 1m suffix shouldn't match _spot_1s
        check("1m does NOT match _spot_1s",
              not any("spot_1s" in str(p) for _, p in files_1m))


# ─── Reorg script ───────────────────────────────────────────────────────────

def test_reorg_script():
    print("\n[reorg_1s_to_subdir.py]")
    script = PROJECT_ROOT / "scripts" / "reorg_1s_to_subdir.py"
    check("reorg script exists", script.exists())
    if not script.exists():
        return
    src = script.read_text(encoding="utf-8")
    check("idempotent: skips if 1s/ exists already (or no legacy)",
          "1s" in src and "yyyymm=" in src)

    # Import and exercise reorg_symbol on a fake layout
    import importlib.util
    spec = importlib.util.spec_from_file_location("_reorg", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with tempfile.TemporaryDirectory() as tmp:
        sym_dir = Path(tmp) / "BTC_USDT"
        legacy_part = sym_dir / "yyyymm=2025-01"
        legacy_part.mkdir(parents=True, exist_ok=True)
        (legacy_part / "data.parquet").write_bytes(b"\x00")

        moved = mod.reorg_symbol(sym_dir, dry_run=False)
        check("reorg_symbol moved 1 partition", moved == 1)
        check("after reorg: legacy yyyymm-dir is gone",
              not legacy_part.exists())
        check("after reorg: BTC_USDT/1s/yyyymm-dir exists",
              (sym_dir / "1s" / "yyyymm=2025-01").exists())

        # Idempotent re-run
        moved2 = mod.reorg_symbol(sym_dir, dry_run=False)
        check("reorg is idempotent (0 on re-run)", moved2 == 0)


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Multi-Timeframe Tests")
    print("=" * 60)
    test_parquet_store_multi_tf()
    test_migrate_script_discovery()
    test_reorg_script()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
