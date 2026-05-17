"""
Open Interest Downloader -- Binance Futures API (free, no API key required).

Downloads open interest for all watchlist perpetual futures symbols.
OI = total value of outstanding derivative contracts (long + short combined).

Signal value:
  - OI rising + price rising  -> strong trend (new money entering)
  - OI rising + price falling -> bearish continuation (new shorts)
  - OI falling + price rising -> short squeeze (shorts closing)
  - OI falling + price falling-> capitulation (longs exiting)

Data source: Binance Futures https://fapi.binance.com/futures/data/openInterestHist
  - Period: 1h (aligns with main OHLCV training data)
  - Binance only retains the last 30 days of OI history
  - Free, no authentication required
  - Rate limit: 50 requests/min (we stay well below)
  - Incremental: run daily to accumulate data going forward

Storage: data/db/hot/open_interest/symbol=BTC_USDT/yyyymm=YYYYMM/data.parquet
  Trainer does an exact timestamp join for 1h models only -- no resampling.

Usage:
    python -m src.data_ingestion.open_interest_downloader
    python -m src.data_ingestion.open_interest_downloader --symbols BTC/USDT ETH/USDT
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

from src.data_ingestion.parquet_writer import load_context_parquet, write_context_parquet

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_BASE_URL          = "https://fapi.binance.com/futures/data/openInterestHist"
_PERIOD            = "1h"
_LIMIT             = 500
_SLEEP_S           = 0.25
_MAX_LOOKBACK_DAYS = 29  # Binance retains ~30 days; stay within limit


def _load_watchlist() -> List[str]:
    wl = PROJECT_ROOT / "data" / "watchlist.json"
    if wl.exists():
        return json.loads(wl.read_text(encoding="utf-8"))
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _to_binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def download_open_interest(
    symbols: List[str] | None = None,
    days: int = _MAX_LOOKBACK_DAYS,
) -> None:
    """Download OI for each symbol and write to parquet (data/db/hot/open_interest/).

    Binance only retains 30 days of history. Run daily to accumulate data.
    Uses endTime-based forward pagination (startTime > 29d ago returns 400).
    """
    if symbols is None:
        symbols = _load_watchlist()

    now_ms         = int(datetime.now(timezone.utc).timestamp() * 1000)
    fetch_days     = min(days, _MAX_LOOKBACK_DAYS)
    fetch_start_ms = int((datetime.now(timezone.utc) - timedelta(days=fetch_days)).timestamp() * 1000)

    for symbol in symbols:
        bsym = _to_binance_symbol(symbol)

        # Incremental: resume from last saved timestamp if recent enough
        start_ms = fetch_start_ms
        existing = load_context_parquet("open_interest", symbol=symbol, ts_col_out="ts")
        if not existing.empty:
            last_ms = int(existing["ts"].max().timestamp() * 1000)
            if last_ms > fetch_start_ms:
                start_ms = last_ms + 3_600_000

        if start_ms >= now_ms:
            logger.info("%s OI already up to date.", symbol)
            continue

        logger.info("Fetching OI for %s from %s ...", symbol,
                    datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d"))

        all_rows: list[dict] = []
        cursor_ms = start_ms

        while True:
            end_ms = min(cursor_ms + _LIMIT * 3_600_000, now_ms)
            params = {
                "symbol":    bsym,
                "period":    _PERIOD,
                "limit":     _LIMIT,
                "startTime": cursor_ms,
                "endTime":   end_ms,
            }
            try:
                resp = requests.get(_BASE_URL, params=params, timeout=15, verify=False)
                resp.raise_for_status()
                rows = resp.json()
            except Exception as exc:
                logger.error("OI fetch failed for %s: %s", symbol, exc)
                break

            if not rows:
                break

            all_rows.extend(rows)
            last_ts_ms = rows[-1]["timestamp"]
            if last_ts_ms >= now_ms or end_ms >= now_ms:
                break
            cursor_ms = last_ts_ms + 3_600_000
            time.sleep(_SLEEP_S)

        if not all_rows:
            logger.warning("No OI data for %s (spot-only or too new)", symbol)
            continue

        new_df = pd.DataFrame(all_rows)
        new_df["ts"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        new_df = new_df.rename(columns={
            "sumOpenInterest":      "oi_base",
            "sumOpenInterestValue": "oi_usdt",
        })[["ts", "oi_base", "oi_usdt"]]
        new_df["oi_base"] = pd.to_numeric(new_df["oi_base"], errors="coerce")
        new_df["oi_usdt"] = pd.to_numeric(new_df["oi_usdt"], errors="coerce")

        write_context_parquet(new_df, "open_interest", symbol=symbol)
        logger.info("Saved %d OI records for %s -> parquet", len(new_df), symbol)
        time.sleep(_SLEEP_S)


def load_open_interest(symbol: str) -> pd.DataFrame:
    """Load OI data for a symbol from parquet. Returns empty DataFrame if not available."""
    df = load_context_parquet("open_interest", symbol=symbol, ts_col_out="timestamp")
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "oi_base", "oi_usdt"])
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download Binance open interest history")
    parser.add_argument("--symbols", nargs="+", help="Symbols to download (default: watchlist)")
    parser.add_argument("--days",    type=int, default=_MAX_LOOKBACK_DAYS,
                        help=f"Lookback days (Binance max: {_MAX_LOOKBACK_DAYS})")
    args = parser.parse_args()
    download_open_interest(symbols=args.symbols, days=args.days)
