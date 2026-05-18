"""
Liquidation Downloader -- CoinGlass API v4 (STARTUP plan).

Liquidations = forced closure of leveraged positions by exchanges.
Large liquidation events cause sharp price moves and are strong short-term signals.

Data source:
  CoinGlass Open API v4 — https://open-api-v4.coinglass.com
  Endpoint: GET /api/futures/liquidation/history
  Key env var: COINGLASS_API_KEY (STARTUP plan, 1h interval available)

  NOTE: Binance /fapi/v1/allForceOrders was removed by Binance (400 "out of maintenance").
  NOTE: Old v2 API (open-api.coinglass.com/public/v2/liquidation) is deprecated.

Storage: data/db/hot/liquidations/symbol=BTC_USDT/yyyymm=YYYYMM/data.parquet
  Trainer does an exact timestamp join for 1h models only -- no resampling.

Historical depth (STARTUP plan):
  1h interval: 180 days   4h interval: 180 days   1d: all-time

Usage:
    python -m src.data_ingestion.liquidation_downloader
    python -m src.data_ingestion.liquidation_downloader --symbols BTC/USDT ETH/USDT
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
_BASE_URL  = "https://open-api-v4.coinglass.com"
_LIQ_PATH  = "/api/futures/liquidation/history"
_SLEEP_S   = 0.8
_CHUNK_MS  = 86_400_000 * 7   # 7-day chunks


def _load_watchlist() -> List[str]:
    wl = PROJECT_ROOT / "data" / "watchlist.json"
    if wl.exists():
        return json.loads(wl.read_text(encoding="utf-8"))
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _to_cg_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def _download_coinglass(symbol: str, api_key: str, days: int) -> pd.DataFrame:
    """Download liquidation history via CoinGlass v4 API."""
    bsym     = _to_cg_symbol(symbol)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_rows: list[dict] = []
    cursor   = since_ms

    while cursor < end_ms:
        headers = {"CG-API-KEY": api_key}
        params  = {
            "symbol":    bsym,
            "exchange":  "Binance",
            "interval":  "1h",
            "startTime": cursor,
            "endTime":   min(cursor + _CHUNK_MS, end_ms),
            "limit":     1000,
        }
        try:
            resp = requests.get(_BASE_URL + _LIQ_PATH, headers=headers,
                                params=params, timeout=20)
            if resp.status_code == 401:
                logger.error("CoinGlass: invalid API key or subscription expired.")
                return pd.DataFrame()
            if resp.status_code == 403:
                logger.warning("CoinGlass: liquidation 1h not on current plan — trying 4h.")
                params["interval"] = "4h"
                resp = requests.get(_BASE_URL + _LIQ_PATH, headers=headers,
                                    params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("CoinGlass fetch failed for %s: %s", symbol, exc)
            break

        if str(data.get("code")) != "0":
            logger.warning("CoinGlass error %s: %s", data.get("code"), data.get("msg"))
            break

        rows = data.get("data", [])
        if rows:
            all_rows.extend(rows)
        cursor += _CHUNK_MS
        time.sleep(_SLEEP_S)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "time" not in df.columns:
        return pd.DataFrame()

    df["ts"]            = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(None)
    df["liq_long_usd"]  = pd.to_numeric(df.get("long_liquidation_usd",  0), errors="coerce").fillna(0)
    df["liq_short_usd"] = pd.to_numeric(df.get("short_liquidation_usd", 0), errors="coerce").fillna(0)
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
