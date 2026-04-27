"""
Downloads historical funding rates for all watchlist symbols using ccxt.
Saves to data/raw/{SYMBOL}_funding.csv.gz for use in the backtester and TFT training.
Funding rates are paid every 8 hours on Binance perpetual futures.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone as _tz
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def _load_watchlist() -> List[str]:
    wl = os.path.join(PROJECT_ROOT, "data", "watchlist.json")
    if os.path.exists(wl):
        with open(wl, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]


def download_funding_rates(
    symbols: List[str] | None = None,
    days: int = 365 * 2,
    exchange_id: str = "binance",
) -> None:
    """
    Downloads funding rate history for each symbol and saves to CSV.gz.
    Skips symbols where perpetual futures don't exist (spot-only coins).
    """
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})

    if symbols is None:
        symbols = _load_watchlist()

    since_ts = int((datetime.now(_tz.utc) - timedelta(days=days)).timestamp() * 1000)

    for symbol in symbols:
        sym_key = symbol.replace("/", "_")
        out_path = os.path.join(RAW_DIR, f"{sym_key}_funding.csv.gz")

        # Infer perpetual symbol (Binance uses BTC/USDT:USDT format for perps)
        perp_symbol = f"{symbol}:{symbol.split('/')[1]}" if "/" in symbol else symbol

        try:
            logger.info("Fetching funding rates for %s ...", perp_symbol)
            all_records = []
            ts = since_ts

            while True:
                try:
                    rows = exchange.fetch_funding_rate_history(perp_symbol, since=ts, limit=1000)
                except Exception as e:
                    if "does not have" in str(e).lower() or "not supported" in str(e).lower():
                        logger.warning("Perpetual futures not available for %s, skipping.", symbol)
                        rows = []
                        break
                    raise

                if not rows:
                    break

                all_records.extend(rows)
                last_ts = rows[-1]["timestamp"]
                if last_ts <= ts:
                    break
                ts = last_ts + 1

                # Rate limit
                time.sleep(exchange.rateLimit / 1000)

            if not all_records:
                logger.info("No funding rate data for %s (spot-only asset, skipping).", symbol)
                continue

            df = pd.DataFrame(all_records)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
            df = df[["timestamp", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
            df = df.sort_values("timestamp").drop_duplicates("timestamp")
            df.to_csv(out_path, index=False, compression="gzip")
            logger.info("Saved %d funding rate records for %s → %s", len(df), symbol, out_path)

        except Exception as exc:
            logger.error("Failed to download funding rates for %s: %s", symbol, exc)


def load_funding_rates(symbol: str) -> pd.DataFrame:
    """
    Loads funding rates for a symbol from disk into a DataFrame indexed by timestamp.
    Returns empty DataFrame if file doesn't exist (spot asset).
    """
    sym_key = symbol.replace("/", "_")
    fpath = os.path.join(RAW_DIR, f"{sym_key}_funding.csv.gz")
    if not os.path.exists(fpath):
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.read_csv(fpath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def merge_funding_into_ohlcv(ohlcv_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Left-joins funding rates into an OHLCV DataFrame using merge_asof (backward fill).
    Adds a 'funding_rate' column; fills NaN with 0.0 for spot assets or gaps.
    """
    funding = load_funding_rates(symbol)
    if funding.empty:
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df["funding_rate"] = 0.0
        return ohlcv_df

    df = ohlcv_df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    df["timestamp"] = ts

    funding["timestamp"] = pd.to_datetime(funding["timestamp"])
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        funding.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_funding_rates()
