"""F5 — Behavioral tests for src/database/parquet_store.py.

Each test exercises the real code path (ingest → query → status) against a
tmpdir-backed store.  No string-matching on source text.
"""
from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _csv_bytes(rows: list[dict]) -> bytes:
    """Serialise a list of row-dicts to CSV bytes."""
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue().encode()


def _make_ohlcv_rows(n: int = 5, start_ts: str = "2025-01-15 00:00:00") -> list[dict]:
    """Generate n synthetic OHLCV rows starting at start_ts (hourly)."""
    from datetime import timedelta
    base = datetime.strptime(start_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = base + timedelta(hours=i)
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low":   95.0 + i,
            "close": 102.0 + i,
            "volume": 1000.0 + i,
        })
    return rows


class _StoreTestCase(unittest.TestCase):
    """Base: create a ParquetStore in a clean tmpdir for each test."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        from src.database.parquet_store import ParquetStore
        self.store = ParquetStore(base_dir=self.tmp.name)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _write_csv(self, rows: list[dict], suffix: str = ".csv") -> Path:
        p = Path(self.tmp.name) / f"test_data{suffix}"
        p.write_bytes(_csv_bytes(rows))
        return p


# ══════════════════════════════════════════════════════════════════════════════
# Ingest
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestCsv(_StoreTestCase):
    def test_ingest_writes_parquet_file(self) -> None:
        rows = _make_ohlcv_rows(3)
        csv_path = self._write_csv(rows)
        result = self.store.ingest_csv(csv_path, "BTC/USDT", timeframe="1h")
        self.assertEqual(result["symbol"], "BTC/USDT")
        self.assertGreater(result["months_written"], 0)
        # Parquet files must exist on disk
        parquet_files = list(Path(self.tmp.name).rglob("*.parquet"))
        self.assertGreater(len(parquet_files), 0)

    def test_ingest_returns_correct_row_count(self) -> None:
        rows = _make_ohlcv_rows(5)
        csv_path = self._write_csv(rows)
        result = self.store.ingest_csv(csv_path, "ETH/USDT", timeframe="1h")
        self.assertEqual(result["rows_total"], 5)

    def test_ingest_missing_csv_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store.ingest_csv(
                Path(self.tmp.name) / "does_not_exist.csv",
                "BTC/USDT",
            )

    def test_skip_existing_months_on_reingest(self) -> None:
        rows = _make_ohlcv_rows(3)
        csv_path = self._write_csv(rows)
        r1 = self.store.ingest_csv(csv_path, "ADA/USDT", timeframe="1h",
                                   skip_existing=True)
        r2 = self.store.ingest_csv(csv_path, "ADA/USDT", timeframe="1h",
                                   skip_existing=True)
        # Second ingest must skip all months already present
        self.assertEqual(r2["months_written"], 0)
        self.assertGreater(r2["skipped_months"], 0)

    def test_unsafe_symbol_raises_value_error(self) -> None:
        rows = _make_ohlcv_rows(2)
        csv_path = self._write_csv(rows)
        with self.assertRaises(ValueError):
            # embedded quote would break DuckDB SQL
            self.store.ingest_csv(csv_path, "BTC'/USDT", timeframe="1h")

    def test_unsafe_csv_path_raises_value_error(self) -> None:
        # A path with a single quote would break DuckDB read_csv_auto()
        bad_path = Path(self.tmp.name) / "bad'file.csv"
        bad_path.write_bytes(_csv_bytes(_make_ohlcv_rows(2)))
        with self.assertRaises(ValueError):
            self.store.ingest_csv(bad_path, "BTC/USDT", timeframe="1h")


# ══════════════════════════════════════════════════════════════════════════════
# Query
# ══════════════════════════════════════════════════════════════════════════════

class TestQuery(_StoreTestCase):
    def setUp(self) -> None:
        super().setUp()
        rows = _make_ohlcv_rows(10, start_ts="2025-03-01 00:00:00")
        csv_path = self._write_csv(rows)
        self.store.ingest_csv(csv_path, "BTC/USDT", timeframe="1h")

    def test_query_returns_dataframe(self) -> None:
        import pandas as pd
        df = self.store.query("BTC/USDT", timeframe="1h")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_query_returns_correct_columns(self) -> None:
        df = self.store.query("BTC/USDT", timeframe="1h")
        for col in ("open", "high", "low", "close", "volume"):
            self.assertIn(col, df.columns, f"expected column {col!r} in result")

    def test_query_limit_respected(self) -> None:
        df = self.store.query("BTC/USDT", timeframe="1h", limit=3)
        self.assertEqual(len(df), 3)

    def test_query_missing_symbol_returns_empty_df(self) -> None:
        import pandas as pd
        df = self.store.query("NONEXIST/USDT", timeframe="1h")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)

    def test_query_start_filter(self) -> None:
        df_all = self.store.query("BTC/USDT", timeframe="1h")
        df_filtered = self.store.query(
            "BTC/USDT", start="2025-03-01 05:00:00", timeframe="1h"
        )
        self.assertLess(len(df_filtered), len(df_all))

    def test_query_column_subset(self) -> None:
        df = self.store.query("BTC/USDT", timeframe="1h", columns=("close", "volume"))
        self.assertIn("close", df.columns)
        self.assertIn("volume", df.columns)
        self.assertNotIn("open", df.columns)


# ══════════════════════════════════════════════════════════════════════════════
# Status / introspection
# ══════════════════════════════════════════════════════════════════════════════

class TestStatus(_StoreTestCase):
    def test_list_symbols_empty_store(self) -> None:
        self.assertEqual(self.store.list_symbols(), [])

    def test_list_symbols_after_ingest(self) -> None:
        rows = _make_ohlcv_rows(2)
        csv_path = self._write_csv(rows)
        self.store.ingest_csv(csv_path, "SOL/USDT", timeframe="1h")
        syms = self.store.list_symbols()
        self.assertIn("SOL/USDT", syms)

    def test_symbol_status_before_ingest(self) -> None:
        s = self.store.symbol_status("XRP/USDT", timeframe="1h")
        self.assertEqual(s.partitions, 0)
        self.assertEqual(s.rows, 0)

    def test_symbol_status_after_ingest(self) -> None:
        rows = _make_ohlcv_rows(4)
        csv_path = self._write_csv(rows)
        self.store.ingest_csv(csv_path, "BNB/USDT", timeframe="1h")
        s = self.store.symbol_status("BNB/USDT", timeframe="1h")
        self.assertGreater(s.partitions, 0)
        self.assertGreater(s.rows, 0)
        self.assertGreater(s.size_bytes, 0)

    def test_list_timeframes(self) -> None:
        rows = _make_ohlcv_rows(2)
        csv_path = self._write_csv(rows)
        self.store.ingest_csv(csv_path, "MATIC/USDT", timeframe="1m")
        tfs = self.store.list_timeframes("MATIC/USDT")
        self.assertIn("1m", tfs)


# ══════════════════════════════════════════════════════════════════════════════
# Thread safety
# ══════════════════════════════════════════════════════════════════════════════

class TestConcurrentAccess(_StoreTestCase):
    def setUp(self) -> None:
        super().setUp()
        rows = _make_ohlcv_rows(5)
        csv_path = self._write_csv(rows)
        self.store.ingest_csv(csv_path, "BTC/USDT", timeframe="1h")

    def test_concurrent_queries_do_not_crash(self) -> None:
        errors: list[Exception] = []

        def _query() -> None:
            try:
                df = self.store.query("BTC/USDT", timeframe="1h")
                if len(df) == 0:
                    errors.append(RuntimeError("query returned empty df"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_query) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(errors, [], f"concurrent queries raised: {errors}")

    def test_concurrent_ingest_and_query(self) -> None:
        """Ingest a different symbol while concurrently querying the first.
        The RLock on the connection must prevent corruption."""
        errors: list[Exception] = []

        def _ingest() -> None:
            try:
                rows = _make_ohlcv_rows(3, start_ts="2025-06-01 00:00:00")
                csv_path = self._write_csv(rows, suffix="_ingest2.csv")
                self.store.ingest_csv(csv_path, "ETH/USDT", timeframe="1h")
            except Exception as exc:
                errors.append(exc)

        def _query() -> None:
            try:
                for _ in range(5):
                    self.store.query("BTC/USDT", timeframe="1h")
            except Exception as exc:
                errors.append(exc)

        t_ingest = threading.Thread(target=_ingest)
        t_query  = threading.Thread(target=_query)
        t_ingest.start()
        t_query.start()
        t_ingest.join(timeout=15)
        t_query.join(timeout=15)

        self.assertEqual(errors, [], f"concurrent ingest+query raised: {errors}")


# ══════════════════════════════════════════════════════════════════════════════
# Input validation (Phase A5 injection protection)
# ══════════════════════════════════════════════════════════════════════════════

class TestInputValidation(unittest.TestCase):
    def test_safe_symbol_rejects_sql_metachar(self) -> None:
        from src.database.parquet_store import _safe_symbol
        with self.assertRaises(ValueError):
            _safe_symbol("BTC'; DROP TABLE x; --/USDT")

    def test_safe_symbol_normalises_slash(self) -> None:
        from src.database.parquet_store import _safe_symbol
        self.assertEqual(_safe_symbol("btc/usdt"), "BTC_USDT")

    def test_safe_timeframe_rejects_unknown(self) -> None:
        from src.database.parquet_store import _safe_timeframe
        with self.assertRaises(ValueError):
            _safe_timeframe("99x")

    def test_safe_timeframe_accepts_known(self) -> None:
        from src.database.parquet_store import _safe_timeframe, SUPPORTED_TIMEFRAMES
        for tf in SUPPORTED_TIMEFRAMES:
            result = _safe_timeframe(tf)
            self.assertEqual(result, tf)

    def test_safe_path_uri_rejects_quote(self) -> None:
        from src.database.parquet_store import _safe_path_uri
        with self.assertRaises(ValueError):
            _safe_path_uri("/data/foo'bar.csv")

    def test_safe_path_uri_rejects_semicolon(self) -> None:
        from src.database.parquet_store import _safe_path_uri
        with self.assertRaises(ValueError):
            _safe_path_uri("/data/foo;DROP.csv")

    def test_safe_path_uri_accepts_normal_path(self) -> None:
        from src.database.parquet_store import _safe_path_uri
        path = "/data/parquet/BTC_USDT/1m/yyyymm=2025-01/data.parquet"
        self.assertEqual(_safe_path_uri(path), path)


if __name__ == "__main__":
    unittest.main()
