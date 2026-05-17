"""
Fear & Greed Index Downloader -- Alternative.me (free, no API key required).

Downloads the entire Crypto Fear & Greed Index history from 2018-02-01.
Index value 0-100: 0 = Extreme Fear, 100 = Extreme Greed.

Signal value for ML:
  - Extreme Fear (0-24)  -> historically good long entries (contrarian)
  - Extreme Greed (76-100) -> historically good exit / short signal
  - Only used in 1d models (daily granularity -- no forward-fill into intraday).

Data source: https://api.alternative.me/fng/?limit=0
  - Free, no API key
  - Daily granularity
  - Available from: 2018-02-01
  - Refreshed daily ~midnight UTC

Storage: data/db/hot/fear_greed/yyyymm=YYYYMM/data.parquet
  Trainer does an exact date join for 1d models only -- no forward-fill.

Usage:
    python -m src.data_ingestion.fear_greed_downloader
    python -m src.data_ingestion.fear_greed_downloader --days 30  # top-up
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests

from src.data_ingestion.parquet_writer import load_context_parquet, write_context_parquet

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_API_URL     = "https://api.alternative.me/fng/?limit=0&format=json"


def download_fear_greed(days: int | None = None) -> None:
    """Download full Fear & Greed history (or last N days) to parquet.

    Incremental: merges with existing records, deduplicates on ts.
    """
    url = f"https://api.alternative.me/fng/?limit={days}&format=json" if days else _API_URL

    logger.info("Fetching Fear & Greed index from Alternative.me ...")
    try:
        resp = requests.get(url, timeout=20, verify=False)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as exc:
        logger.error("Fear & Greed fetch failed: %s", exc)
        return

    if not data:
        logger.warning("No Fear & Greed data returned.")
        return

    new_df = pd.DataFrame(data)
    new_df["ts"]         = pd.to_datetime(new_df["timestamp"].astype(int), unit="s", utc=True).dt.tz_convert(None)
    new_df["fear_greed"] = new_df["value"].astype(int)
    new_df["fg_label"]   = new_df["value_classification"]
    new_df = new_df[["ts", "fear_greed", "fg_label"]].sort_values("ts")

    write_context_parquet(new_df, "fear_greed", symbol=None)
    date_range = f"{new_df['ts'].min().date()} -> {new_df['ts'].max().date()}"
    logger.info("Saved %d Fear & Greed records (%s) -> parquet", len(new_df), date_range)


def load_fear_greed() -> pd.DataFrame:
    """Load Fear & Greed index from parquet. Returns empty DataFrame if not downloaded."""
    df = load_context_parquet("fear_greed", symbol=None, ts_col_out="timestamp")
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "fear_greed", "fg_label"])
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download Crypto Fear & Greed Index")
    parser.add_argument("--days", type=int, default=None,
                        help="Last N days only (default: full history from 2018)")
    args = parser.parse_args()
    download_fear_greed(days=args.days)
