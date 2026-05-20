"""
OHLCV + funding loader from the parquet partition store.

Reads data/parquet/{SYM}/{tf}/yyyymm=YYYY-MM/data_0.parquet
Returns a single sorted DataFrame with deduped timestamps.

Raises FileNotFoundError when data is absent — no silent empty returns.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR  = PROJECT_ROOT / "data" / "parquet"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"


def _safe_sym(symbol: str) -> str:
    return symbol.replace("/", "_")


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data for symbol/timeframe from the parquet store.

    Returns DataFrame with columns:
      timestamp, open, high, low, close, volume, quote_volume,
      trades_count, taker_buy_base, taker_buy_quote
    sorted ascending by timestamp.

    Raises FileNotFoundError if no parquet partition exists. During the
    CSV.gz transition period, falls back to data/raw/*.csv.gz before raising.
    """
    sym = _safe_sym(symbol)
    part_dir = PARQUET_DIR / sym / timeframe

    if part_dir.exists():
        files = sorted(part_dir.glob("**/*.parquet"))
        if files:
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = (df.sort_values("timestamp")
                  .drop_duplicates(subset=["timestamp"], keep="last")
                  .reset_index(drop=True))
            log.debug("Loaded %d rows for %s/%s from parquet (%d files)",
                      len(df), sym, timeframe, len(files))
            return df

    # Fallback to CSV.gz
    csv_path = RAW_DIR / f"{sym}_{timeframe}.csv.gz"
    spot_path = RAW_DIR / f"{sym}_spot_{timeframe}.csv.gz"
    for p in (csv_path, spot_path):
        if p.exists():
            log.warning("Parquet missing for %s/%s -- falling back to %s", sym, timeframe, p.name)
            df = pd.read_csv(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)

    raise FileNotFoundError(
        f"No parquet data for {symbol}/{timeframe}. Run backfill first."
    )


def load_funding(symbol: str) -> pd.DataFrame:
    """Load funding rate data from the parquet store.

    Returns DataFrame with columns: timestamp, funding_rate.
    Raises FileNotFoundError if no parquet partition exists (spot-only
    assets like SHIB have no funding; callers should catch and skip).
    """
    sym = _safe_sym(symbol)
    part_dir = PARQUET_DIR / sym / "funding"

    if part_dir.exists():
        files = sorted(part_dir.glob("**/*.parquet"))
        if files:
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = (df.sort_values("timestamp")
                  .drop_duplicates(subset=["timestamp"], keep="last")
                  .reset_index(drop=True))
            return df

    raise FileNotFoundError(
        f"No parquet funding data for {symbol}. Run funding_rate_downloader first."
    )
