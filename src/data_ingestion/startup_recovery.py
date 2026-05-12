"""
Startup Recovery — Phase 7.

Run on every system start. For each (symbol, timeframe) pair:

  1. Find `last_ts` in QuestDB (hot path) AND in Parquet (cold path).
  2. The greater of the two is our last-known data point.
  3. If the gap to "now" is bigger than the timeframe period, fetch missing
     bars from the Binance archive (`binance_archive_downloader`) for the
     completed months, then top up the most recent <1 month gap from the
     Binance public REST klines API.
  4. After recovery, the realtime WS writer can resume cleanly — no overlap,
     no holes.

This script is idempotent — safe to run any time. It is the FIRST thing
called by `restart_all.ps1` after QuestDB is up.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.parquet_client import get_client as get_questdb
from src.database.parquet_store import get_store
from src.data_ingestion.binance_archive_downloader import (
    download_symbol as archive_download_symbol,
    SUPPORTED_TF as ARCHIVE_TFS,
)

logger = logging.getLogger("startup_recovery")

# Map our timeframe → seconds per bar (used to detect gaps).
_TF_SECONDS = {
    "1s":  1,        "1m":  60,       "3m":  180,      "5m":  300,
    "15m": 900,      "30m": 1800,     "1h":  3600,     "2h":  7200,
    "4h":  14400,    "6h":  21600,    "8h":  28800,    "12h": 43200,
    "1d":  86400,    "3d":  259200,   "1w":  604800,   "1mo": 2592000,
}


def _parquet_last_ts(symbol: str, timeframe: str) -> datetime | None:
    """Latest timestamp present in the Parquet store for this (symbol, tf)."""
    try:
        store = get_store()
        st = store.symbol_status(symbol, timeframe=timeframe)
        if st.latest:
            return datetime.fromisoformat(str(st.latest).replace("Z", "+00:00"))
    except Exception as exc:
        logger.debug("parquet last_ts(%s, %s) failed: %s", symbol, timeframe, exc)
    return None


def _questdb_last_ts(symbol: str, timeframe: str) -> datetime | None:
    """Latest timestamp present in QuestDB for this (symbol, tf)."""
    try:
        qdb = get_questdb()
        return qdb.get_latest_candle_ts(symbol, timeframe)
    except Exception as exc:
        logger.debug("questdb last_ts(%s, %s) failed: %s", symbol, timeframe, exc)
    return None


def _as_utc(ts: datetime | None) -> datetime | None:
    """Coerce a possibly-naive datetime to UTC-aware so it can be compared.

    Phase C+ fix (2026-05-12): ParquetClient.get_latest_candle_ts returns a
    naive datetime; ParquetStore.symbol_status returns an ISO string we
    parse to a TZ-aware datetime. Mixing them under `max()` raised
    `TypeError: can't compare offset-naive and offset-aware datetimes`,
    blocking startup_recovery for every (symbol, tf) pair on every restart.
    """
    if ts is None:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _last_known(symbol: str, timeframe: str) -> datetime | None:
    """Take the max of QuestDB and Parquet last-timestamps."""
    a = _as_utc(_questdb_last_ts(symbol, timeframe))
    b = _as_utc(_parquet_last_ts(symbol, timeframe))
    if a and b:
        return max(a, b)
    return a or b


def _gap_seconds(last_ts: datetime | None) -> float:
    if last_ts is None:
        return float("inf")
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last_ts).total_seconds()


def recover_symbol_tf(symbol: str, timeframe: str, *, archive_only: bool = False) -> dict:
    """Bring (symbol, tf) up to the present.

    Strategy:
      - Find last-known timestamp.
      - If gap > 1 month, drive the archive downloader.
      - Top-up the rest via Binance public REST klines (recent 30 days are
        served at full granularity by the public endpoint).

    Returns: {symbol, timeframe, last_known, gap_hours, archive_months, rest_bars}
    """
    last = _last_known(symbol, timeframe)
    gap_h = _gap_seconds(last) / 3600.0
    res = {
        "symbol":     symbol,
        "timeframe":  timeframe,
        "last_known": last.isoformat() if last else None,
        "gap_hours":  round(gap_h, 2),
        "archive_months": 0,
        "rest_bars":  0,
    }

    if gap_h <= (_TF_SECONDS.get(timeframe, 60) / 3600.0) * 1.5:
        return res     # already up-to-date

    # 1) Archive (full months only)
    if timeframe in ARCHIVE_TFS:
        try:
            ar = archive_download_symbol(symbol, timeframe=timeframe)
            res["archive_months"] = ar.get("months_downloaded", 0)
        except Exception as exc:
            logger.warning("archive download failed for %s/%s: %s",
                           symbol, timeframe, exc)

    if archive_only:
        return res

    # 2) Top-up via REST for the recent <1mo window. We use the existing
    #    streaming downloader (`binance_downloader`) if it exposes a klines
    #    helper; otherwise hit the public endpoint directly.
    try:
        rest_bars = _rest_topup(symbol, timeframe, since=last)
        res["rest_bars"] = rest_bars
    except Exception as exc:
        logger.warning("REST top-up failed for %s/%s: %s", symbol, timeframe, exc)

    return res


def _rest_topup(symbol: str, timeframe: str, since: datetime | None) -> int:
    """Fetch recent bars via Binance public klines REST API and write to QuestDB."""
    import requests
    qdb = get_questdb()

    sym_bin = symbol.replace("/", "")
    interval = timeframe if timeframe != "1mo" else "1M"  # Binance REST uses '1M'
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = (int(since.timestamp() * 1000)
                if since else end_ms - 30 * 86400 * 1000)

    written = 0
    while start_ms < end_ms:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={sym_bin}&interval={interval}"
               f"&startTime={start_ms}&limit=1000")
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            rows = r.json()
        except Exception:
            break
        if not rows:
            break
        bars = [{
            "timestamp": int(row[0]),
            "open":  float(row[1]), "high": float(row[2]),
            "low":   float(row[3]), "close": float(row[4]),
            "volume": float(row[5]),
        } for row in rows]
        n = qdb.write_market_candles_bulk(symbol, timeframe, bars)
        written += n
        # advance start_ms to avoid re-fetch
        start_ms = int(rows[-1][0]) + 1
        time.sleep(0.1)
    return written


def recover_all(symbols: list[str] | None = None,
                timeframes: list[str] | None = None,
                *, archive_only: bool = False) -> list[dict]:
    """Run recovery for every (symbol, tf) pair."""
    if symbols is None:
        wl = PROJECT_ROOT / "data" / "watchlist.json"
        symbols = json.loads(wl.read_text(encoding="utf-8")) if wl.exists() else ["BTC/USDT"]
    if timeframes is None:
        timeframes = ["1m", "1h", "1d"]

    out = []
    for s in symbols:
        for tf in timeframes:
            try:
                out.append(recover_symbol_tf(s, tf, archive_only=archive_only))
            except Exception as exc:
                logger.exception("recover %s/%s failed: %s", s, tf, exc)
                out.append({"symbol": s, "timeframe": tf, "error": str(exc)})
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--tfs", nargs="+", default=None)
    p.add_argument("--archive-only", action="store_true",
                   help="Skip the REST top-up; only fill complete months")
    args = p.parse_args()

    res = recover_all(args.symbols, args.tfs, archive_only=args.archive_only)
    n_arc = sum(r.get("archive_months", 0) for r in res)
    n_rest = sum(r.get("rest_bars", 0) for r in res)
    logger.info("=" * 60)
    logger.info("Recovery done: %d (sym, tf) pairs, %d archive months, %d REST bars",
                len(res), n_arc, n_rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
