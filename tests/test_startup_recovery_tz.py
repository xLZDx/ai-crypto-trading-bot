"""Regression test for the TZ comparison bug in startup_recovery._last_known.

Pre-2026-05-12 bug: `_last_known` did `max(a, b)` where `a` came from
ParquetClient (naive datetime) and `b` came from ParquetStore (TZ-aware
datetime parsed from an ISO string). max() raised
`TypeError: can't compare offset-naive and offset-aware datetimes`,
blocking startup_recovery for every (symbol, timeframe) pair on every
restart_all invocation. Discovered while attempting to live-validate
Phase C — restart_all.ps1 hung on step 0/6 with 602+ caught exceptions.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_ingestion import startup_recovery as sr  # noqa: E402


class TestLastKnownTZSafe(unittest.TestCase):
    def test_naive_and_aware_inputs_do_not_raise(self) -> None:
        naive   = datetime(2026, 5, 12, 10, 0, 0)              # tzinfo=None
        aware   = datetime(2026, 5, 12, 11, 0, 0, tzinfo=timezone.utc)
        with mock.patch.object(sr, "_questdb_last_ts", return_value=naive), \
             mock.patch.object(sr, "_parquet_last_ts", return_value=aware):
            # Pre-fix: this raises TypeError. Post-fix: returns the aware datetime.
            got = sr._last_known("BTC/USDT", "1h")
        self.assertIsNotNone(got)
        self.assertIsNotNone(got.tzinfo, "result must be TZ-aware after normalization")
        self.assertEqual(got, aware, "11:00 UTC is later than 10:00 (naive treated as UTC)")

    def test_both_naive_normalized(self) -> None:
        a = datetime(2026, 5, 10, 0, 0, 0)
        b = datetime(2026, 5, 11, 0, 0, 0)
        with mock.patch.object(sr, "_questdb_last_ts", return_value=a), \
             mock.patch.object(sr, "_parquet_last_ts", return_value=b):
            got = sr._last_known("BTC/USDT", "1h")
        self.assertEqual(got.tzinfo, timezone.utc)
        self.assertEqual(got.replace(tzinfo=None), b)

    def test_both_aware(self) -> None:
        a = datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
        b = datetime(2026, 5, 12, 0, 0, 0, tzinfo=timezone.utc)
        with mock.patch.object(sr, "_questdb_last_ts", return_value=a), \
             mock.patch.object(sr, "_parquet_last_ts", return_value=b):
            self.assertEqual(sr._last_known("BTC/USDT", "1h"), b)

    def test_one_none(self) -> None:
        a = datetime(2026, 5, 10, 0, 0, 0)  # naive
        with mock.patch.object(sr, "_questdb_last_ts", return_value=a), \
             mock.patch.object(sr, "_parquet_last_ts", return_value=None):
            got = sr._last_known("BTC/USDT", "1h")
        self.assertEqual(got.tzinfo, timezone.utc)
        self.assertEqual(got.replace(tzinfo=None), a)

    def test_both_none(self) -> None:
        with mock.patch.object(sr, "_questdb_last_ts", return_value=None), \
             mock.patch.object(sr, "_parquet_last_ts", return_value=None):
            self.assertIsNone(sr._last_known("BTC/USDT", "1h"))


if __name__ == "__main__":
    unittest.main()
