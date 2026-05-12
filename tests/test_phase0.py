"""
Phase 0 tests — validate the institutional-upgrade foundation.

Coverage:
  - src/database/parquet_store.py — ingest synthetic CSV → query roundtrip
  - src/transport/zmq_config.py   — port allocation
  - src/transport/data_bus.py     — module imports, serialize/deserialize
  - src/transport/control_api.py  — module imports, FastAPI app builds
  - scripts/migrate_1sec_to_parquet.py — discover_csv_files() logic
  - orchestrator + worker — new endpoints / methods present

Run:
    python tests/test_phase0.py
    python -m pytest tests/test_phase0.py -v
"""
from __future__ import annotations

import gzip
import io
import os
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


# ─── parquet_store ───────────────────────────────────────────────────────────

def test_parquet_store():
    print("\n[Parquet Store]")
    try:
        from src.database.parquet_store import ParquetStore, get_store, DEFAULT_BASE_DIR
    except Exception as exc:
        check("import parquet_store", False, str(exc))
        return
    check("import parquet_store", True)
    check("DEFAULT_BASE_DIR points at data/parquet",
          DEFAULT_BASE_DIR == PROJECT_ROOT / "data" / "parquet")

    # Build a tiny synthetic CSV.gz with two months of data
    import pandas as pd
    rows = []
    for ts in pd.date_range("2025-01-01", "2025-02-15", freq="D"):
        rows.append({"timestamp": ts, "open": 100.0, "high": 101.0,
                     "low": 99.5, "close": 100.5, "volume": 1.0})
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "TEST_USDT_spot_1s.csv.gz"
        with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
        check("synthetic gzipped CSV created", csv_path.exists())

        store = ParquetStore(tmp / "parquet")
        try:
            res = store.ingest_csv(csv_path, "TEST/USDT")
        except Exception as exc:
            check("ingest_csv runs", False, str(exc))
            return
        check("ingest_csv runs", True)
        check("two months written (2025-01, 2025-02)", res["months_written"] == 2,
              f"got {res['months_written']}")
        check("rows_total > 0", res["rows_total"] > 0)

        # Re-ingest = idempotent
        res2 = store.ingest_csv(csv_path, "TEST/USDT")
        check("re-ingest is idempotent (skipped_months == 2)",
              res2["skipped_months"] == 2 and res2["months_written"] == 0)

        # Query roundtrip
        out = store.query("TEST/USDT", start="2025-01-15", end="2025-02-01")
        check("query returns DataFrame", out is not None and len(out) >= 1)
        check("query respects start filter",
              len(out) > 0 and str(out["timestamp"].min())[:10] >= "2025-01-15")

        # Status
        st = store.symbol_status("TEST/USDT")
        check("symbol_status partitions == 2", st.partitions == 2,
              f"got {st.partitions}")
        check("symbol_status rows > 0", st.rows > 0)

        all_status = store.status()
        check("status() returns size_bytes > 0", all_status["size_bytes"] > 0)
        check("TEST/USDT appears in symbols list",
              "TEST/USDT" in store.list_symbols())

        store.drop_symbol("TEST/USDT")
        check("drop_symbol clears the directory",
              "TEST/USDT" not in store.list_symbols())


# ─── zmq_config ──────────────────────────────────────────────────────────────

def test_zmq_config():
    print("\n[ZMQ Config]")
    try:
        from src.transport.zmq_config import (
            ORDERFLOW_PORT, TRAINING_BATCH_PORT, CONTROL_FANOUT_PORT,
            CONTROL_API_PORT, bind_addr, connect_addr,
        )
    except Exception as exc:
        check("import zmq_config", False, str(exc))
        return
    check("import zmq_config", True)
    check("ORDERFLOW_PORT == 5555", ORDERFLOW_PORT == 5555)
    check("TRAINING_BATCH_PORT == 5556", TRAINING_BATCH_PORT == 5556)
    check("CONTROL_FANOUT_PORT == 5557", CONTROL_FANOUT_PORT == 5557)
    check("CONTROL_API_PORT == 8100", CONTROL_API_PORT == 8100)
    check("bind_addr format", bind_addr(5555) == "tcp://*:5555")
    check("connect_addr format", connect_addr(5555, "192.168.0.5") == "tcp://192.168.0.5:5555")


# ─── data_bus ────────────────────────────────────────────────────────────────

def test_data_bus():
    print("\n[Data Bus]")
    try:
        from src.transport.data_bus import (
            DataBus, get_data_bus, _serialize, _deserialize,
        )
    except Exception as exc:
        check("import data_bus", False, str(exc))
        return
    check("import data_bus", True)

    # Roundtrip a simple dict (msgpack path)
    payload = {"symbol": "BTC/USDT", "bid": 1.0, "ask": 2.0, "ts": 12345}
    blob = _serialize(payload)
    out = _deserialize(blob)
    check("serialize/deserialize roundtrip (dict)", out == payload)

    # msgpack cannot serialize numpy arrays — callers must convert first (.tolist()).
    # Verify that attempting to serialize ndarray raises ValueError (not a silent failure).
    try:
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0])
        try:
            _serialize({"x": arr})
            check("numpy array serialization raises ValueError", False,
                  "expected ValueError but no exception raised")
        except ValueError:
            check("numpy array serialization raises ValueError", True)
    except ImportError:
        check("numpy serialization", None, "numpy not installed")

    bus = get_data_bus()
    check("get_data_bus returns DataBus", isinstance(bus, DataBus))
    stats = bus.stats()
    check("stats() exposes orderflow_port",
          stats.get("orderflow_port") == 5555)
    check("stats() exposes training_batch_port",
          stats.get("training_batch_port") == 5556)


# ─── control_api ─────────────────────────────────────────────────────────────

def test_control_api():
    print("\n[Control API]")
    try:
        from src.transport import control_api
    except Exception as exc:
        check("import control_api", False, str(exc))
        return
    check("import control_api", True)

    if control_api.app is None:
        check("FastAPI app builds", None, "fastapi not installed")
        return
    check("FastAPI app builds", True)

    # Verify routes are registered
    routes = {r.path for r in control_api.app.routes}
    for path in ["/health", "/parquet/status", "/parquet/symbols",
                 "/parquet/ingest", "/databus/stats"]:
        check(f"route {path} registered", path in routes)


# ─── migrate_1sec_to_parquet ─────────────────────────────────────────────────

def test_migration_script():
    print("\n[Migration Script]")
    script_path = PROJECT_ROOT / "scripts" / "migrate_1sec_to_parquet.py"
    check("script file exists", script_path.exists())

    import importlib.util
    spec = importlib.util.spec_from_file_location("_mig", script_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        check("script imports cleanly", False, str(exc))
        return
    check("script imports cleanly", True)
    check("discover_csv_files() defined", hasattr(mod, "discover_csv_files"))

    # Empty dir -> empty list
    with tempfile.TemporaryDirectory() as tmp:
        files = mod.discover_csv_files(Path(tmp))
        check("empty dir -> empty list", files == [])

        # Create a fake file
        fake = Path(tmp) / "BTC_USDT_spot_1s.csv.gz"
        fake.write_bytes(b"")
        files = mod.discover_csv_files(Path(tmp))
        check("matched file -> (BTC/USDT, path)",
              len(files) == 1 and files[0][0] == "BTC/USDT")


# ─── orchestrator + worker integration ───────────────────────────────────────

def test_orchestrator_endpoints():
    print("\n[Orchestrator Phase 0 endpoints]")
    src_path = PROJECT_ROOT / "src" / "training" / "distributed" / "orchestrator.py"
    src = src_path.read_text(encoding="utf-8")
    check("orchestrator.py exists", src_path.exists())
    check("/api/parquet/status route added", "/api/parquet/status" in src)
    check("/api/databus/stats route added", "/api/databus/stats" in src)
    check("imports get_store from parquet_store",
          "from src.database.parquet_store import get_store" in src)
    check("imports get_data_bus from data_bus",
          "from src.transport.data_bus import get_data_bus" in src)


def test_worker_transport():
    print("\n[Worker transport hook]")
    src_path = PROJECT_ROOT / "src" / "training" / "distributed" / "worker.py"
    src = src_path.read_text(encoding="utf-8")
    check("worker.py exists", src_path.exists())
    check("_transport_info() method added", "_transport_info(" in src)
    check("transport key in /health response", '"transport":' in src)


# ─── requirements.txt ────────────────────────────────────────────────────────

def test_requirements():
    print("\n[requirements.txt]")
    req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ["pyarrow", "pyzmq", "fastapi", "uvicorn", "msgpack"]:
        check(f"{pkg} in requirements.txt", pkg in req)


# ─── Plan + CLAUDE.md presence ───────────────────────────────────────────────

def test_plan_files():
    print("\n[Plan files]")
    check("INSTITUTIONAL_UPGRADE_PLAN.md at root",
          (PROJECT_ROOT / "INSTITUTIONAL_UPGRADE_PLAN.md").exists())
    check("CLAUDE.md at root",
          (PROJECT_ROOT / "CLAUDE.md").exists())
    claude = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    check("CLAUDE.md mentions approval gate",
          "approval" in claude.lower() or "approve" in claude.lower())


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 0 — Institutional Upgrade Foundation Tests")
    print("=" * 60)
    test_requirements()
    test_plan_files()
    test_parquet_store()
    test_zmq_config()
    test_data_bus()
    test_control_api()
    test_migration_script()
    test_orchestrator_endpoints()
    test_worker_transport()

    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
