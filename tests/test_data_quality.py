"""Phase 4 (2026-05-14) — F1 Data Integrity tests.

Behavioral tests: each test calls validate_ohlcv() with a real DataFrame
and asserts on the returned report's flags + the raise-vs-return policy.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import data_quality as dq  # noqa: E402


def _good_df(n: int = 240, bar_s: int = 3600, start: str = "2026-01-01") -> pd.DataFrame:
    """Build a synthetic OHLCV frame that should pass every check."""
    ts = pd.date_range(start, periods=n, freq=f"{bar_s}s", tz=None)
    close = 100 + (pd.Series(range(n)) % 7) * 0.5  # mild oscillation
    return pd.DataFrame({
        "timestamp": ts,
        "open":   close - 0.1,
        "high":   close + 0.3,
        "low":    close - 0.4,
        "close":  close,
        "volume": [1000.0] * n,
    })


class _ModeReset(unittest.TestCase):
    """Reset DATA_QUALITY_MODE between tests so each starts fresh."""

    def setUp(self) -> None:
        self._old_mode = os.environ.pop(dq._MODE_ENV, None)

    def tearDown(self) -> None:
        if self._old_mode is not None:
            os.environ[dq._MODE_ENV] = self._old_mode
        else:
            os.environ.pop(dq._MODE_ENV, None)


class TestHappyPath(_ModeReset):
    def test_clean_frame_passes(self) -> None:
        df = _good_df()
        out, rep = dq.validate_ohlcv(df, symbol="TEST", timeframe="1h")
        self.assertTrue(rep.schema_ok)
        self.assertTrue(rep.bounds_ok)
        self.assertTrue(rep.monotonic_ts)
        self.assertEqual(rep.hard_errors, [])
        self.assertEqual(rep.n_rows_in, len(df))
        self.assertEqual(rep.n_rows_out, len(df))


class TestBoundsCheck(_ModeReset):
    def test_negative_price_raises_in_enforce(self) -> None:
        df = _good_df(50)
        df.loc[10, "close"] = -1.0
        with self.assertRaises(dq.DataQualityError):
            dq.validate_ohlcv(df, symbol="X", timeframe="1h")

    def test_negative_price_warns_in_warn_mode(self) -> None:
        os.environ[dq._MODE_ENV] = "warn"
        df = _good_df(50)
        df.loc[5, "close"] = -1.0
        with self.assertLogs("src.utils.data_quality", level="CRITICAL") as cap:
            out, rep = dq.validate_ohlcv(df, symbol="X", timeframe="1h")
        self.assertFalse(rep.bounds_ok)
        self.assertTrue(any("HARD failures" in m for m in cap.output))
        # warn mode returns the frame anyway
        self.assertIsNotNone(out)

    def test_high_lower_than_low_raises(self) -> None:
        df = _good_df(30)
        df.loc[5, "high"] = df.loc[5, "low"] - 0.5
        with self.assertRaises(dq.DataQualityError):
            dq.validate_ohlcv(df, symbol="X", timeframe="1h")

    def test_negative_volume_raises(self) -> None:
        df = _good_df(30)
        df.loc[7, "volume"] = -10
        with self.assertRaises(dq.DataQualityError):
            dq.validate_ohlcv(df, symbol="X", timeframe="1h")


class TestTimestampChecks(_ModeReset):
    def test_duplicate_ts_deduplicated_and_flagged(self) -> None:
        df = _good_df(20)
        # Duplicate row 5's timestamp into row 4
        df.loc[4, "timestamp"] = df.loc[5, "timestamp"]
        out, rep = dq.validate_ohlcv(df, symbol="X", timeframe="1h")
        self.assertEqual(rep.duplicate_ts, 1)
        # dedup'd
        self.assertEqual(rep.n_rows_out, len(df) - 1)
        self.assertTrue(any("deduplicated" in w for w in rep.soft_warnings))

    def test_non_monotonic_timestamps_raise(self) -> None:
        df = _good_df(20)
        # Swap rows 5 and 6 so ts goes backwards
        df.loc[5, "timestamp"], df.loc[6, "timestamp"] = (
            df.loc[6, "timestamp"], df.loc[5, "timestamp"],
        )
        with self.assertRaises(dq.DataQualityError):
            dq.validate_ohlcv(df, symbol="X", timeframe="1h")


class TestGapDetection(_ModeReset):
    def test_gap_flagged_as_soft(self) -> None:
        """A 10-bar gap is a soft warning, not a hard fail."""
        df = _good_df(40, bar_s=3600)
        # Shift rows 20+ forward by an extra 10 hours
        df.loc[20:, "timestamp"] = df.loc[20:, "timestamp"] + pd.Timedelta(hours=10)
        out, rep = dq.validate_ohlcv(df, symbol="X", timeframe="1h")
        self.assertTrue(rep.schema_ok)  # not a hard fail
        self.assertEqual(len(rep.gaps), 1)
        self.assertTrue(any("gap" in w for w in rep.soft_warnings))


class TestZeroVolumeRun(_ModeReset):
    def test_long_zero_volume_run_flagged(self) -> None:
        df = _good_df(200, bar_s=3600)
        df.loc[50:120, "volume"] = 0  # 71 consecutive zero-vol bars
        out, rep = dq.validate_ohlcv(df, symbol="X", timeframe="1h",
                                     max_zero_vol_bars=60)
        # Not a hard fail
        self.assertTrue(rep.bounds_ok)
        self.assertEqual(len(rep.zero_volume_runs), 1)
        self.assertTrue(any("zero-volume" in w for w in rep.soft_warnings))


class TestPriceSpikeDetection(_ModeReset):
    def test_giant_spike_flagged_as_soft(self) -> None:
        df = _good_df(50)
        # Inject a 60% spike at row 25
        df.loc[25, ["open", "high", "low", "close"]] = (
            df.loc[24, "close"] * 1.6,
        ) * 4
        out, rep = dq.validate_ohlcv(df, symbol="X", timeframe="1h",
                                     max_spike_pct=0.5)
        self.assertEqual(len(rep.price_spikes), 1)
        self.assertTrue(any("move" in w for w in rep.soft_warnings))


class TestOffMode(_ModeReset):
    def test_off_skips_everything(self) -> None:
        """DATA_QUALITY_MODE=off → no validation, no flags, just return."""
        os.environ[dq._MODE_ENV] = "off"
        df = _good_df(20)
        df.loc[10, "close"] = -1.0  # would normally fail
        out, rep = dq.validate_ohlcv(df, symbol="X", timeframe="1h")
        # Returns the frame unchecked, no hard error raised
        self.assertEqual(len(out), len(df))
        self.assertEqual(rep.hard_errors, [])


class TestReportSerialization(_ModeReset):
    def test_to_dict_serializes_for_meta_json(self) -> None:
        df = _good_df(30)
        out, rep = dq.validate_ohlcv(df, symbol="BTC_USDT", timeframe="1h")
        d = rep.to_dict()
        # JSON-serializable shape — no pandas objects in there
        import json
        json.dumps(d)
        self.assertEqual(d["symbol"], "BTC_USDT")
        self.assertEqual(d["timeframe"], "1h")
        self.assertTrue(d["schema_ok"])


if __name__ == "__main__":
    unittest.main()
