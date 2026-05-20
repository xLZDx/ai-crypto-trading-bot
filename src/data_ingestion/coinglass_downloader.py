"""
CoinGlass v4 downloader — fetches all accessible endpoints on STARTUP plan.

Storage layout:
  data/coinglass/futures/<SYMBOL>/<metric>_<interval>.parquet
  data/coinglass/spot/<SYMBOL>/<metric>_<interval>.parquet
  data/coinglass/macro/<metric>.parquet

STARTUP plan limits (historical depth by interval):
  1h / 4h  -> 180 days
  6h / 8h / 12h -> 360 days
  1d       -> all-time (back to 2018-2019)

Fallback when COINGLASS_API_KEY is missing or subscription expired:
  Fear & Greed -> https://api.alternative.me/fng/?limit=0 (free, no key needed)
  Funding rate -> Binance futures /fapi/v1/fundingRate     (uses existing API_KEY)

Usage:
    python -m src.data_ingestion.coinglass_downloader
    python -m src.data_ingestion.coinglass_downloader --symbols BTC/USDT ETH/USDT
    python -m src.data_ingestion.coinglass_downloader --macro-only
    python -m src.data_ingestion.coinglass_downloader --futures-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "coinglass"
BASE_URL     = "https://open-api-v4.coinglass.com"

# Rate limit: STARTUP = 80/min -> sleep 0.8s per request to stay safe
_SLEEP_S = 0.8
# Max rows per request (API-level limit appears to be ~4500)
_CHUNK_ROWS = 1000


# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------

# Per-symbol futures endpoints
# path: API path template (%s = symbol like BTCUSDT)
# time_field: field name for timestamp in response rows
# time_unit: 'ms' or 's'
# columns: {response_field -> output_column_name}
# exchange: Binance (default for all derivative metrics)
FUTURES_SYMBOL_ENDPOINTS = {
    "oi": {
        "path": "/api/futures/open-interest/history",
        "params": {"exchange": "Binance"},
        "time_field": "time", "time_unit": "ms",
        "columns": {"open": "oi_open", "high": "oi_high", "low": "oi_low", "close": "oi_close"},
        "intervals": ["1h", "4h", "1d"],
    },
    "ls_global": {
        "path": "/api/futures/global-long-short-account-ratio/history",
        "params": {"exchange": "Binance"},
        "time_field": "time", "time_unit": "ms",
        "columns": {
            "global_account_long_percent": "ls_long_pct",
            "global_account_short_percent": "ls_short_pct",
            "global_account_long_short_ratio": "ls_ratio",
        },
        "intervals": ["1h", "4h", "1d"],
    },
    "ls_top_trader": {
        "path": "/api/futures/top-long-short-position-ratio/history",
        "params": {"exchange": "Binance"},
        "time_field": "time", "time_unit": "ms",
        "columns": {},  # populated dynamically from response keys
        "intervals": ["4h", "1d"],
    },
    "funding_rate": {
        "path": "/api/futures/funding-rate/history",
        "params": {"exchange": "Binance"},
        "time_field": "time", "time_unit": "ms",
        "columns": {"open": "fr_open", "high": "fr_high", "low": "fr_low", "close": "fr_close"},
        "intervals": ["8h", "1d"],
    },
    "liquidations": {
        "path": "/api/futures/liquidation/history",
        "params": {"exchange": "Binance"},
        "time_field": "time", "time_unit": "ms",
        "columns": {
            "long_liquidation_usd": "liq_long_usd",
            "short_liquidation_usd": "liq_short_usd",
        },
        "intervals": ["1h", "4h"],
    },
    "fut_taker": {
        "path": "/api/futures/taker-buy-sell-volume/history",
        "params": {"exchange": "Binance"},
        "time_field": "time", "time_unit": "ms",
        "columns": {
            "taker_buy_volume_usd": "fut_taker_buy_usd",
            "taker_sell_volume_usd": "fut_taker_sell_usd",
        },
        "intervals": ["1h", "4h"],
    },
    "coinbase_premium": {
        "path": "/api/coinbase-premium-index",
        "params": {},  # no exchange param, no symbol param
        "time_field": "time", "time_unit": "s",  # NOTE: seconds not ms
        "columns": {
            "premium": "cbp_premium",
            "premium_rate": "cbp_premium_rate",
        },
        "intervals": ["1h", "4h"],
        "no_symbol": True,  # global, not per-symbol
    },
}

SPOT_SYMBOL_ENDPOINTS = {
    "spot_taker": {
        "path": "/api/spot/taker-buy-sell-volume/history",
        "params": {"exchange": "Binance"},
        "time_field": "time", "time_unit": "ms",
        "columns": {
            "taker_buy_volume_usd": "spot_taker_buy_usd",
            "taker_sell_volume_usd": "spot_taker_sell_usd",
        },
        "intervals": ["1h", "4h"],
    },
}

# Global macro endpoints (no symbol param)
MACRO_ENDPOINTS = {
    "fear_greed": {
        "path": "/api/index/fear-greed-history",
        "params": {},
        "format": "parallel_arrays",  # {data_list, time_list}
        "time_unit": "ms",
        "value_col": "fear_greed",
    },
    "btc_dominance": {
        "path": "/api/index/bitcoin-dominance",
        "params": {},
        "format": "list",
        "time_field": "timestamp", "time_unit": "ms",
        "columns": {"bitcoin_dominance": "btc_dominance", "market_cap": "total_mcap"},
    },
    "stablecoin_mcap": {
        "path": "/api/index/stableCoin-marketCap-history",
        "params": {},
        "format": "parallel_arrays_obj",  # {data_list: [{USDT: v}, ...], time_list}
        "time_unit": "ms",
        "value_col": "stablecoin_mcap_total",
    },
    "ahr999": {
        "path": "/api/index/ahr999",
        "params": {},
        "format": "date_string",
        "date_field": "date_string",
        "columns": {"ahr999_value": "ahr999", "current_value": "ahr999_current"},
    },
    "puell_multiple": {
        "path": "/api/index/puell-multiple",
        "params": {},
        "format": "list",
        "time_field": "timestamp", "time_unit": "ms",
        "columns": {"puell_multiple": "puell_multiple"},
    },
    "golden_ratio": {
        "path": "/api/index/golden-ratio-multiplier",
        "params": {},
        "format": "list",
        "time_field": "timestamp", "time_unit": "ms",
        "columns": {},
    },
    "etf_btc_flow": {
        "path": "/api/etf/bitcoin/flow-history",
        "params": {},
        "format": "list",
        "time_field": "timestamp", "time_unit": "ms",
        "columns": {"flow_usd": "etf_btc_flow_usd"},
    },
    "etf_eth_flow": {
        "path": "/api/etf/ethereum/flow-history",
        "params": {},
        "format": "list",
        "time_field": "timestamp", "time_unit": "ms",
        "columns": {"flow_usd": "etf_eth_flow_usd"},
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return os.environ.get("COINGLASS_API_KEY", "").strip()


def _get(path: str, params: dict, api_key: str) -> Optional[dict]:
    url = BASE_URL + path
    headers = {"CG-API-KEY": api_key}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20, verify=False)
        if resp.status_code == 401:
            logger.warning("CoinGlass 401 -- key invalid or subscription expired.")
            return None
        if resp.status_code == 403:
            logger.warning("CoinGlass 403 -- endpoint not on current plan: %s", path)
            return None
        resp.raise_for_status()
        data = resp.json()
        if str(data.get("code")) != "0":
            logger.warning("CoinGlass API error %s: %s | path=%s params=%s",
                           data.get("code"), data.get("msg"), path, params)
            return None
        return data
    except Exception as exc:
        logger.warning("CoinGlass request failed %s %s: %s", path, params, exc)
        return None


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------

def _rows_to_df(rows: list[dict], cfg: dict) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    tf = cfg["time_field"]
    if tf not in df.columns:
        return pd.DataFrame()
    ts = pd.to_numeric(df[tf], errors="coerce")
    if cfg["time_unit"] == "s":
        ts = ts * 1000
    df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True).dt.tz_convert(None)
    col_map = cfg.get("columns", {})
    if col_map:
        for src, dst in col_map.items():
            if src in df.columns:
                df[dst] = pd.to_numeric(df[src], errors="coerce")
        keep = ["timestamp"] + [dst for dst in col_map.values() if dst in df.columns]
    else:
        # auto-detect: keep all numeric columns
        num_cols = [c for c in df.columns if c not in (tf, "timestamp") and
                    pd.api.types.is_numeric_dtype(df[c])]
        keep = ["timestamp"] + num_cols
    return df[keep].sort_values("timestamp").reset_index(drop=True)


def _date_string_to_df(rows: list[dict], cfg: dict) -> pd.DataFrame:
    """Parse rows with a date_string field like '2011/02/01'."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    date_field = cfg.get("date_field", "date_string")
    if date_field not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df[date_field], format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    col_map = cfg.get("columns", {})
    if col_map:
        for src, dst in col_map.items():
            if src in df.columns:
                df[dst] = pd.to_numeric(df[src], errors="coerce")
        keep = ["timestamp"] + [dst for dst in col_map.values() if dst in df.columns]
    else:
        num_cols = [c for c in df.columns if c not in (date_field, "timestamp") and
                    pd.api.types.is_numeric_dtype(df[c])]
        keep = ["timestamp"] + num_cols
    return df[keep].sort_values("timestamp").reset_index(drop=True)


def _parallel_arrays_to_df(data: dict, cfg: dict) -> pd.DataFrame:
    dl = data.get("data_list", [])
    tl = data.get("time_list", [])
    if not dl or not tl:
        # some endpoints return just data_list without time_list (indexed by day from epoch)
        if dl and not tl:
            # generate daily timestamps starting from a reasonable base
            # fall back: treat index as day offset from 2010-01-01
            base = pd.Timestamp("2010-01-01")
            dates = [base + pd.Timedelta(days=i) for i in range(len(dl))]
            df = pd.DataFrame({"timestamp": dates, cfg["value_col"]: dl})
            return df
        return pd.DataFrame()
    ts_factor = 1 if cfg["time_unit"] == "ms" else 1000
    ts = [t * ts_factor for t in tl]
    if isinstance(dl[0], dict):
        # stablecoin mcap: list of {COIN: value} dicts -> sum all values
        values = [sum(d.values()) for d in dl]
    else:
        values = dl
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="ms", utc=True).tz_convert(None),
        cfg["value_col"]: pd.to_numeric(values, errors="coerce"),
    })
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch_symbol_metric(
    symbol_cg: str,        # e.g. "BTCUSDT"
    metric_key: str,
    cfg: dict,
    interval: str,
    api_key: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch one metric for one symbol, paginating in chunks."""
    all_rows: list[dict] = []
    # Build date range: start from max allowed history for this interval
    days_back = _max_days(interval)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor   = start_ms
    chunk_ms = limit * _interval_ms(interval)

    no_symbol = cfg.get("no_symbol", False)

    while cursor < end_ms:
        params = {
            **cfg.get("params", {}),
            "interval": interval,
            "startTime": cursor,
            "endTime": min(cursor + chunk_ms, end_ms),
            "limit": limit,
        }
        if not no_symbol:
            params["symbol"] = symbol_cg
        data = _get(cfg["path"], params, api_key)
        time.sleep(_SLEEP_S)
        if data is None:
            break
        rows = data.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        cursor += chunk_ms

    return _rows_to_df(all_rows, cfg)


def _fetch_macro_metric(metric_key: str, cfg: dict, api_key: str, limit: int = 5000) -> pd.DataFrame:
    params = {**cfg.get("params", {}), "limit": limit}
    data = _get(cfg["path"], params, api_key)
    time.sleep(_SLEEP_S)
    if data is None:
        return _fallback_macro(metric_key)
    fmt = cfg.get("format", "list")
    if fmt in ("parallel_arrays", "parallel_arrays_obj"):
        return _parallel_arrays_to_df(data.get("data", {}), cfg)
    if fmt == "date_string":
        return _date_string_to_df(data.get("data", []), cfg)
    # list format
    rows = data.get("data", [])
    if isinstance(rows, dict):
        rows = [rows]
    return _rows_to_df(rows, cfg)


# ---------------------------------------------------------------------------
# Fallback: free sources when subscription expires
# ---------------------------------------------------------------------------

def _fallback_macro(metric_key: str) -> pd.DataFrame:
    if metric_key == "fear_greed":
        return _fallback_fear_greed()
    return pd.DataFrame()


def _fallback_fear_greed() -> pd.DataFrame:
    """alternative.me Fear & Greed — completely free, no API key needed."""
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 0, "format": "json"},
            timeout=20,
            verify=False,
        )
        resp.raise_for_status()
        items = resp.json().get("data", [])
        rows = []
        for item in items:
            ts = pd.to_datetime(int(item["timestamp"]), unit="s", utc=True).tz_convert(None)
            rows.append({"timestamp": ts, "fear_greed": float(item["value"])})
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        logger.info("Fear & Greed fallback: %d rows from alternative.me", len(df))
        return df
    except Exception as exc:
        logger.warning("Fear & Greed fallback failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

def _interval_ms(interval: str) -> int:
    mapping = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
               "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
               "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
               "1d": 86_400_000}
    return mapping.get(interval, 3_600_000)


def _max_days(interval: str) -> int:
    """STARTUP plan maximum history depth per interval."""
    if interval in ("1m", "3m", "5m", "15m", "30m"):
        return 0   # not available on STARTUP
    if interval in ("1h", "2h", "4h"):
        return 180
    if interval in ("6h", "8h", "12h"):
        return 360
    return 3650   # 1d -> 10 years (all-time)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path.relative_to(PROJECT_ROOT))


def _load(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _merge_and_save(new_df: pd.DataFrame, path: Path) -> None:
    if new_df.empty:
        return
    existing = _load(path)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df
    _save(combined, path)


def _symbol_to_cg(symbol: str) -> str:
    """BTC/USDT -> BTCUSDT"""
    return symbol.replace("/", "")


def _symbol_to_coin(symbol: str) -> str:
    """BTC/USDT -> BTC"""
    return symbol.split("/")[0]


# ---------------------------------------------------------------------------
# Main download functions
# ---------------------------------------------------------------------------

def download_futures_metrics(symbols: list[str], api_key: str) -> None:
    for symbol in symbols:
        sym_cg   = _symbol_to_cg(symbol)
        sym_dir  = DATA_DIR / "futures" / sym_cg
        logger.info("Downloading futures metrics for %s ...", symbol)

        for metric_key, cfg in FUTURES_SYMBOL_ENDPOINTS.items():
            no_symbol = cfg.get("no_symbol", False)
            # coinbase_premium is global — save once, skip per-symbol loop
            if no_symbol and symbol != symbols[0]:
                continue

            for interval in cfg["intervals"]:
                if _max_days(interval) == 0:
                    continue
                df = _fetch_symbol_metric(sym_cg, metric_key, cfg, interval, api_key)
                if df.empty:
                    logger.debug("  %s %s %s -- no data", metric_key, symbol, interval)
                    continue
                if no_symbol:
                    out = DATA_DIR / "macro" / f"{metric_key}_{interval}.parquet"
                else:
                    out = sym_dir / f"{metric_key}_{interval}.parquet"
                _merge_and_save(df, out)


def download_spot_metrics(symbols: list[str], api_key: str) -> None:
    for symbol in symbols:
        sym_cg  = _symbol_to_cg(symbol)
        sym_dir = DATA_DIR / "spot" / sym_cg
        logger.info("Downloading spot metrics for %s ...", symbol)

        for metric_key, cfg in SPOT_SYMBOL_ENDPOINTS.items():
            for interval in cfg["intervals"]:
                if _max_days(interval) == 0:
                    continue
                df = _fetch_symbol_metric(sym_cg, metric_key, cfg, interval, api_key)
                if df.empty:
                    continue
                out = sym_dir / f"{metric_key}_{interval}.parquet"
                _merge_and_save(df, out)


def download_macro_metrics(api_key: str) -> None:
    logger.info("Downloading global macro metrics ...")
    macro_dir = DATA_DIR / "macro"

    for metric_key, cfg in MACRO_ENDPOINTS.items():
        df = _fetch_macro_metric(metric_key, cfg, api_key)
        if df.empty:
            logger.debug("  %s -- no data", metric_key)
            continue
        out = macro_dir / f"{metric_key}.parquet"
        _merge_and_save(df, out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(symbols: Optional[list[str]] = None,
         futures_only: bool = False,
         macro_only: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = _api_key()
    if not api_key:
        logger.warning(
            "COINGLASS_API_KEY not set in .env -- macro metrics will use free fallbacks only.\n"
            "Add: COINGLASS_API_KEY=<your_key>"
        )

    if symbols is None:
        wl_path = PROJECT_ROOT / "data" / "watchlist.json"
        symbols = json.loads(wl_path.read_text(encoding="utf-8")) if wl_path.exists() else ["BTC/USDT", "ETH/USDT"]

    logger.info("CoinGlass download -- %d symbols, futures=%s macro=%s",
                len(symbols), not macro_only, not futures_only)

    if not macro_only and api_key:
        download_futures_metrics(symbols, api_key)
        download_spot_metrics(symbols, api_key)

    if not futures_only:
        download_macro_metrics(api_key)

    logger.info("CoinGlass download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CoinGlass v4 data")
    parser.add_argument("--symbols", nargs="+", help="Symbols to download (default: watchlist)")
    parser.add_argument("--futures-only", action="store_true")
    parser.add_argument("--macro-only", action="store_true")
    args = parser.parse_args()
    main(symbols=args.symbols, futures_only=args.futures_only, macro_only=args.macro_only)
