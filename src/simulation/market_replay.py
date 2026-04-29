"""
MarketReplay — streams bars from gzipped CSV files to simulate a live feed.

Design:
  - Never loads an entire GZ file into memory; uses pandas chunked reading.
  - Supports 1s, 1m, 1h, 1d timeframes from data/raw/{sym}_{tf}.csv.gz.
  - Speed control: real-time (speed=1) to ultra-fast (speed=10000, no sleep).
  - Optional date-range filtering: only yields bars within [start, end].
  - Funding rate injection: merges {sym}_funding.csv.gz by timestamp proximity.
  - Returns bars as plain dicts so they are JSON-serialisable and bus-safe.
"""
from __future__ import annotations

import gzip
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _to_utc_ts(dt) -> pd.Timestamp:
    """Convert any datetime/Timestamp to tz-aware UTC Timestamp."""
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

# Seconds per bar for each supported timeframe
_TF_SECONDS: dict[str, int] = {
    "1s": 1,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Expected OHLCV columns (Binance archive format)
_OHLCV_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote",
]


def _gz_path(symbol: str, timeframe: str) -> Path:
    """Resolve the GZ file path for a given symbol/timeframe."""
    candidates = [
        RAW_DIR / f"{symbol}_{timeframe}.csv.gz",
        RAW_DIR / f"{symbol.replace('_', '')}_{timeframe}.csv.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No GZ file for {symbol}/{timeframe} in {RAW_DIR}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _funding_path(symbol: str) -> Path | None:
    p = RAW_DIR / f"{symbol}_funding.csv.gz"
    return p if p.exists() else None


def _load_funding_index(symbol: str) -> pd.Series | None:
    """Load funding rates as a Series indexed by timestamp for fast lookup."""
    path = _funding_path(symbol)
    if path is None:
        return None
    try:
        df = pd.read_csv(path, compression="gzip", index_col=0, parse_dates=True)
        if "fundingRate" in df.columns:
            s = df["fundingRate"].astype(float)
        elif df.shape[1] >= 1:
            s = df.iloc[:, 0].astype(float)
        else:
            return None
        s.index = pd.to_datetime(s.index, utc=True)
        return s.sort_index()
    except Exception as exc:
        logger.warning("[MarketReplay] Funding load error for %s: %s", symbol, exc)
        return None


def _get_funding_at(funding: pd.Series | None, ts: pd.Timestamp) -> float:
    if funding is None or funding.empty:
        return 0.0
    idx = funding.index.get_indexer([ts], method="nearest")
    return float(funding.iloc[idx[0]]) if idx[0] >= 0 else 0.0


class MarketReplay:
    """
    Streams OHLCV bars from a GZ file as if they were arriving live.

    Usage::

        replay = MarketReplay("BTC_USDT", "1m", speed=100.0)
        for bar in replay.stream(start=datetime(2023,1,1), end=datetime(2023,6,1)):
            print(bar)
            if stop_requested:
                break
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        speed: float = 1.0,
        chunk_size: int = 10_000,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.speed = max(0.001, speed)
        self.chunk_size = chunk_size
        self._bar_seconds = _TF_SECONDS.get(timeframe, 60)
        self._funding = _load_funding_index(symbol)
        self._gz_path = _gz_path(symbol, timeframe)
        self._bars_emitted = 0

    @property
    def bars_emitted(self) -> int:
        return self._bars_emitted

    # ── public interface ─────────────────────────────────────────────────────

    def stream(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        stopped_flag: list[bool] | None = None,
    ) -> Generator[dict, None, None]:
        """
        Yield one bar dict per call, sleeping to simulate real-time pace.

        Args:
            start:        first bar to yield (inclusive). None = beginning of file.
            end:          last bar to yield (inclusive). None = end of file.
            stopped_flag: mutable list[bool]; set [0]=True externally to break.
        """
        start_ts = _to_utc_ts(start) if start else None
        end_ts = _to_utc_ts(end) if end else None
        stopped = stopped_flag or [False]

        sleep_per_bar = self._bar_seconds / self.speed if self.speed < 5000 else 0.0

        try:
            reader = pd.read_csv(
                self._gz_path,
                compression="gzip",
                chunksize=self.chunk_size,
                index_col=0,
                parse_dates=True,
            )
        except Exception as exc:
            logger.error("[MarketReplay] Cannot open %s: %s", self._gz_path, exc)
            return

        for chunk in reader:
            if stopped[0]:
                break

            # Normalise index to UTC
            if chunk.index.tz is None:
                chunk.index = chunk.index.tz_localize("UTC")
            else:
                chunk.index = chunk.index.tz_convert("UTC")

            # Ensure required columns exist
            for col in _OHLCV_COLS:
                if col not in chunk.columns:
                    chunk[col] = 0.0

            # Date-range filter at chunk level (skip entire chunk if out of range)
            if end_ts is not None and chunk.index[0] > end_ts:
                break
            if start_ts is not None and chunk.index[-1] < start_ts:
                continue

            for ts, row in chunk.iterrows():
                if stopped[0]:
                    return
                if start_ts is not None and ts < start_ts:
                    continue
                if end_ts is not None and ts > end_ts:
                    return

                bar = self._row_to_bar(ts, row)
                self._bars_emitted += 1

                yield bar

                if sleep_per_bar > 0:
                    time.sleep(sleep_per_bar)

    # ── internal ─────────────────────────────────────────────────────────────

    def _row_to_bar(self, ts: pd.Timestamp, row: pd.Series) -> dict:
        funding = _get_funding_at(self._funding, ts)
        return {
            "symbol":       self.symbol,
            "timeframe":    self.timeframe,
            "timestamp":    ts.isoformat(),
            "open":         float(row.get("open", 0)),
            "high":         float(row.get("high", 0)),
            "low":          float(row.get("low", 0)),
            "close":        float(row.get("close", 0)),
            "volume":       float(row.get("volume", 0)),
            "quote_volume": float(row.get("quote_volume", 0)),
            "trades_count": int(row.get("trades_count", 0)),
            "taker_buy_base":  float(row.get("taker_buy_base", 0)),
            "taker_buy_quote": float(row.get("taker_buy_quote", 0)),
            "funding_rate": funding,
            "source":       "simulator",
        }

    # ── file introspection ───────────────────────────────────────────────────

    def get_date_range(self) -> tuple[datetime | None, datetime | None]:
        """Return (first_ts, last_ts) by reading only the first and last chunks."""
        try:
            first_chunk = pd.read_csv(
                self._gz_path, compression="gzip",
                chunksize=1000, index_col=0, parse_dates=True,
            )
            first_df = next(iter(first_chunk))
            if first_df.index.tz is None:
                first_df.index = first_df.index.tz_localize("UTC")
            first_ts = first_df.index[0].to_pydatetime()

            # For the last timestamp, count total rows to seek near end
            # (reading the entire file is expensive — use a heuristic: read
            #  the last chunk by scanning with tail=True-like approach)
            last_ts = self._scan_last_ts()
            return first_ts, last_ts
        except Exception as exc:
            logger.warning("[MarketReplay] get_date_range error: %s", exc)
            return None, None

    def _scan_last_ts(self) -> datetime | None:
        """Read all chunks to find the last timestamp (efficient for index-seek)."""
        last_ts = None
        try:
            reader = pd.read_csv(
                self._gz_path, compression="gzip",
                chunksize=50_000, index_col=0, parse_dates=True,
            )
            for chunk in reader:
                if chunk.index.tz is None:
                    chunk.index = chunk.index.tz_localize("UTC")
                last_ts = chunk.index[-1].to_pydatetime()
        except Exception:
            pass
        return last_ts

    def estimate_total_bars(self) -> int:
        """Estimate total bars from file size without scanning the full file."""
        try:
            gz_bytes = self._gz_path.stat().st_size
            # Rough estimate: each gzip-compressed OHLCV row ≈ 15 bytes compressed
            return int(gz_bytes / 15)
        except Exception:
            return 0
