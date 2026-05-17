"""
Liquidation Downloader -- Coinglass API (free tier).

Liquidations = forced closure of leveraged positions by exchanges.
Large liquidation events cause sharp price moves and are strong short-term signals.

Signal value:
  - Large long liquidations  -> sharp price drop incoming / underway
  - Large short liquidations -> short squeeze, sharp price rise

Data source:
  Coinglass Open API (free, requires API key from https://coinglass.com/pricing)
  Endpoint: GET https://open-api.coinglass.com/public/v2/liquidation
  Key env var: COINGLASS_API_KEY

  NOTE: Binance /fapi/v1/allForceOrders was removed by Binance (400 "out of maintenance").
  Without a Coinglass key, this downloader skips and training proceeds with
  zero liquidation features.

Storage: data/db/hot/liquidations/symbol=BTC_USDT/yyyymm=YYYYMM/data.parquet
  Trainer does an exact timestamp join for 1h models only -- no resampling.

Historical depth:
  - Coinglass: up to 2019 for BTC/ETH

Usage:
    # Set API key first (free at coinglass.com):
    # $env:COINGLASS_API_KEY = "your_key_here"  (add to .env)

    python -m src.data_ingestion.liquidation_downloader
    python -m src.data_ingestion.liquidation_downloader --symbols BTC/USDT --days 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

from src.data_ingestion.parquet_writer import load_context_parquet, write_context_parquet

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_COINGLASS_URL = "https://open-api.coinglass.com/public/v2/liquidation"
_SLEEP_S       = 0.5


def _load_watchlist() -> List[str]:
    wl = PROJECT_ROOT / "data" / "watchlist.json"
    if wl.exists():
        return json.loads(wl.read_text(encoding="utf-8"))
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _to_binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def _download_coinglass(symbol: str, api_key: str, days: int) -> pd.DataFrame:
    """Download liquidation history from Coinglass. Returns empty df on error."""
    bsym     = _to_binance_symbol(symbol)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_rows: list[dict] = []
    cursor   = since_ms

    while True:
        headers = {"coinglassSecret": api_key}
        params  = {"symbol": bsym, "startTime": cursor, "endTime": cursor + 86400_000 * 30}
        try:
            resp = requests.get(_COINGLASS_URL, headers=headers, params=params, timeout=15, verify=False)
            if resp.status_code == 401:
                logger.error("Coinglass: invalid API key. Get a free key at https://coinglass.com/pricing")
                return pd.DataFrame()
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Coinglass fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

        rows = data.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        cursor += 86400_000 * 30
        if cursor >= int(datetime.now(timezone.utc).timestamp() * 1000):
            break
        time.sleep(_SLEEP_S)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "t" not in df.columns:
        return pd.DataFrame()

    df["ts"]           = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
    df["liq_long_usd"]  = pd.to_numeric(df.get("buyUsdVolume",  0), errors="coerce").fillna(0)
    df["liq_short_usd"] = pd.to_numeric(df.get("sellUsdVolume", 0), errors="coerce").fillna(0)
    df["liq_total_usd"] = df["liq_long_usd"] + df["liq_short_usd"]
    return df[["ts", "liq_long_usd", "liq_short_usd", "liq_total_usd"]]


def download_liquidations(
    symbols: List[str] | None = None,
    days: int = 365 * 4,
    binance_only: bool = False,  # kept for CLI compat, ignored (Binance endpoint retired)
) -> None:
    """Download liquidation history for all symbols via Coinglass API."""
    if symbols is None:
        symbols = _load_watchlist()

    api_key = os.environ.get("COINGLASS_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "COINGLASS_API_KEY not set -- skipping liquidations (training uses zeros).\n"
            "  Binance /fapi/v1/allForceOrders was retired (400 'out of maintenance').\n"
            "  Get a Coinglass key at: https://coinglass.com/pricing\n"
            "  Then add to .env: COINGLASS_API_KEY=your_key"
        )
        return

    for symbol in symbols:
        logger.info("Fetching liquidation history for %s (Coinglass, %d days) ...", symbol, days)
        new_df = _download_coinglass(symbol, api_key, days)

        if new_df.empty:
            logger.warning("No liquidation data for %s", symbol)
            continue

        write_context_parquet(new_df, "liquidations", symbol=symbol)
        logger.info("Saved %d liq records for %s -> parquet", len(new_df), symbol)
        time.sleep(_SLEEP_S)


def load_liquidations(symbol: str) -> pd.DataFrame:
    """Load liquidation data for a symbol from parquet."""
    df = load_context_parquet("liquidations", symbol=symbol, ts_col_out="timestamp")
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "liq_long_usd", "liq_short_usd", "liq_total_usd"])
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download crypto liquidation history")
    parser.add_argument("--symbols", nargs="+", help="Symbols (default: watchlist)")
    parser.add_argument("--days",    type=int, default=365 * 4)
    args = parser.parse_args()
    download_liquidations(symbols=args.symbols, days=args.days)
