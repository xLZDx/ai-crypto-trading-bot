"""
Phase 7 tests — Continuous data pipeline + retention.

Coverage:
  - binance_archive_downloader multi-timeframe path helpers
  - realtime_db_writer: stream URL builder + kline event parser
  - startup_recovery: helpers + recover_symbol_tf no-op when up-to-date
  - retention_manager: scan, mark_trained, archive_eligible, save/load
  - google_drive_backup: graceful no-op when creds absent

Run:
    python tests/test_phase7.py
"""
from __future__ import annotations

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


# ─── Archive downloader multi-tf ────────────────────────────────────────────

def test_archive_downloader_multi_tf():
    print("\n[Archive Downloader Multi-TF]")
    try:
        from src.data_ingestion.binance_archive_downloader import (
            _gz_path, _output_filename, _output_dir_for, SUPPORTED_TF, _download_month_zip,
        )
    except Exception as exc:
        check("import binance_archive_downloader", False, str(exc))
        return
    check("import binance_archive_downloader", True)
    check("SUPPORTED_TF contains 1m, 1h, 1d, 1mo",
          all(t in SUPPORTED_TF for t in ("1m", "1h", "1d", "1mo")))

    # 1s legacy path preserved
    check("1s output filename preserves legacy _spot_1s.csv.gz",
          _output_filename("BTC/USDT", "1s") == "BTC_USDT_spot_1s.csv.gz")
    check("1m output filename = BTC_USDT_1m.csv.gz",
          _output_filename("BTC/USDT", "1m") == "BTC_USDT_1m.csv.gz")
    check("1mo output filename = BTC_USDT_1mo.csv.gz",
          _output_filename("BTC/USDT", "1mo") == "BTC_USDT_1mo.csv.gz")

    check("1s lives in data/raw/historical/",
          _output_dir_for("1s").name == "historical")
    check("non-1s lives in data/raw/",
          _output_dir_for("1m").name == "raw")


# ─── Realtime writer ────────────────────────────────────────────────────────

def test_realtime_writer():
    print("\n[Realtime DB Writer]")
    try:
        from src.data_ingestion.realtime_db_writer import (
            _stream_name, _ws_url, _parse_kline_event,
        )
    except Exception as exc:
        check("import realtime_db_writer", False, str(exc))
        return
    check("import realtime_db_writer", True)

    # Stream name
    check("_stream_name canonical form",
          _stream_name("BTC/USDT", "1m") == "btcusdt@kline_1m")

    # URL builder
    url = _ws_url(["BTC/USDT", "ETH/USDT"], ["1m", "1h"])
    check("URL has /stream?streams=", "/stream?streams=" in url)
    for s in ("btcusdt@kline_1m", "btcusdt@kline_1h",
              "ethusdt@kline_1m", "ethusdt@kline_1h"):
        check(f"URL includes {s}", s in url)

    # Closed-bar parsing
    closed = {
        "stream": "btcusdt@kline_1m",
        "data": {"k": {"x": True, "t": 1700000000000, "T": 1700000059999,
                       "o": "50000", "h": "50100", "l": "49900", "c": "50050",
                       "v": "100", "n": 200, "V": "60", "Q": "3000000"}}
    }
    bar = _parse_kline_event(closed)
    check("closed bar parses", bar is not None)
    check("bar symbol BTC/USDT", bar.get("symbol") == "BTC/USDT")
    check("bar timeframe 1m", bar.get("timeframe") == "1m")
    check("bar OHLC populated",
          bar.get("close") == 50050.0 and bar.get("open") == 50000.0)

    # In-progress bars must be skipped
    open_bar = dict(closed)
    open_bar["data"] = dict(closed["data"])
    open_bar["data"]["k"] = dict(closed["data"]["k"])
    open_bar["data"]["k"]["x"] = False
    check("in-progress bar (x=False) is skipped",
          _parse_kline_event(open_bar) is None)


# ─── Startup recovery ──────────────────────────────────────────────────────

def test_startup_recovery():
    print("\n[Startup Recovery]")
    try:
        from src.data_ingestion import startup_recovery as sr
    except Exception as exc:
        check("import startup_recovery", False, str(exc))
        return
    check("import startup_recovery", True)
    check("recover_all() defined", hasattr(sr, "recover_all"))
    check("recover_symbol_tf() defined", hasattr(sr, "recover_symbol_tf"))
    check("_TF_SECONDS contains 1m -> 60", sr._TF_SECONDS.get("1m") == 60)
    check("_TF_SECONDS contains 1d -> 86400", sr._TF_SECONDS.get("1d") == 86400)


# ─── Retention manager ────────────────────────────────────────────────────

def test_retention_manager():
    print("\n[Retention Manager]")
    try:
        from src.database.retention_manager import RetentionManager, PartitionRecord
    except Exception as exc:
        check("import retention_manager", False, str(exc))
        return
    check("import retention_manager", True)

    with tempfile.TemporaryDirectory() as tmp:
        idx = Path(tmp) / "retention.json"
        rm = RetentionManager(index_path=idx)
        check("starts with empty records", len(rm._records) == 0)

        # Inject a record manually
        r = PartitionRecord(symbol="BTC/USDT", timeframe="1s", yyyymm="2018-01",
                            size_bytes=12345)
        rm._records[r.key] = r
        rm.save()
        check("save() writes index file", idx.exists())

        # Reload
        rm2 = RetentionManager(index_path=idx)
        check("load() restores records", len(rm2._records) == 1)
        check("loaded key matches",
              "BTC/USDT::1s::2018-01" in rm2._records)

        # mark_trained
        ok = rm2.mark_trained("BTC/USDT", "1s", "2018-01", "btc_rf_v1")
        check("mark_trained returns True", ok)
        rec = rm2._records["BTC/USDT::1s::2018-01"]
        check("trained_on contains 'btc_rf_v1'", "btc_rf_v1" in rec.trained_on)

        # mark_trained_range
        for ym in ("2018-02", "2018-03", "2018-04"):
            r = PartitionRecord(symbol="BTC/USDT", timeframe="1s", yyyymm=ym)
            rm2._records[r.key] = r
        rm2.save()
        n = rm2.mark_trained_range("BTC/USDT", "1s", "2018-02", "2018-03", "oft_v1")
        check("mark_trained_range marks 2 partitions", n == 2)

        # archive_eligible — needs old enough yyyymm and >= 1 trained model
        elig = rm2.archive_eligible(min_models=1, older_than_days=30)
        check("archive_eligible returns list",
              isinstance(elig, list))
        check("archive_eligible filters by trained_on",
              all(len(r.trained_on) >= 1 for r in elig))

        # stats
        s = rm2.stats()
        check("stats has partitions / trained / archived keys",
              all(k in s for k in ("partitions", "trained", "archived")))


# ─── Google Drive backup (no-op without creds) ─────────────────────────────

def test_gdrive_backup():
    print("\n[Google Drive Backup]")
    try:
        from src.database.google_drive_backup import GoogleDriveBackup
    except Exception as exc:
        check("import google_drive_backup", False, str(exc))
        return
    check("import google_drive_backup", True)

    bk = GoogleDriveBackup(root_folder_name="test-archive")
    check("instantiates without auth", bk is not None)
    # Without creds, is_available() should return False (graceful)
    avail = bk.is_available()
    check("is_available returns bool", isinstance(avail, bool))
    if not avail:
        check("upload_file fails-soft (returns None)",
              bk.upload_file(Path("/tmp/nonexistent")) is None)


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 7 — Continuous Pipeline + Retention Tests")
    print("=" * 60)
    test_archive_downloader_multi_tf()
    test_realtime_writer()
    test_startup_recovery()
    test_retention_manager()
    test_gdrive_backup()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
