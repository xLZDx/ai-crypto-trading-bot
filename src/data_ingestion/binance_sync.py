"""
Binance Sync Orchestrator — Phase 8B.

The single entry point that chains:

    archive (data.binance.vision)  →  REST top-up (api.binance.com)  →
        cross-check vs current Binance state  →  hand off to realtime WS

This replaces the partial `startup_recovery.py` flow. Run on every system
boot (called by restart_all.ps1) and any time you want to make sure the
hot path (QuestDB) and cold path (Parquet) are coherent with Binance's
authoritative state.

All Binance HTTP calls go through `rate_limiter.get_limiter('binance.com')`
so concurrent runs (e.g. archive downloader + sync) share a global budget
and never get IP-banned.

Run:
    python -m src.data_ingestion.binance_sync
    python -m src.data_ingestion.binance_sync --symbols BTC/USDT --tfs 1m 1h
    python -m src.data_ingestion.binance_sync --skip-archive       # only top-up
    python -m src.data_ingestion.binance_sync --check-recent 200   # cross-check tail
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

from src.data_ingestion.binance_archive_downloader import (
    download_all_timeframes_parallel, download_symbol as archive_download_symbol,
    SUPPORTED_TF as ARCHIVE_TFS,
)
from src.data_ingestion.rate_limiter import get_limiter
from src.database.parquet_store import get_store
from src.database.parquet_client import get_client as get_questdb

logger = logging.getLogger("binance_sync")

WATCHLIST_FILE = PROJECT_ROOT / "data" / "watchlist.json"
DEFAULT_TFS    = ["1m", "5m", "1h", "4h", "1d"]


def _watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


# ─── Step 1 — archive pull (delegates to the parallelized downloader) ──────

def step_archive(symbols: list[str], timeframes: list[str]) -> dict:
    archive_tfs = [tf for tf in timeframes if tf in ARCHIVE_TFS]
    if not archive_tfs:
        logger.info("[sync] no archive-capable timeframes; skipping archive step")
        return {"step": "archive", "skipped": True}
    download_all_timeframes_parallel(symbols=symbols, timeframes=archive_tfs)
    return {"step": "archive", "ok": True, "timeframes": archive_tfs}


# ─── Step 2 — REST top-up (last 30 days, fills archive→now gap) ────────────

_BINANCE_INTERVAL = {
    "1s":  "1s",  "1m":  "1m",  "3m":  "3m",  "5m":  "5m", "15m": "15m",
    "30m": "30m", "1h":  "1h",  "2h":  "2h",  "4h":  "4h", "6h":  "6h",
    "8h":  "8h",  "12h": "12h", "1d":  "1d",  "3d":  "3d", "1w":  "1w",
    "1mo": "1M",  # Binance REST uses '1M', the archive uses '1mo'
}


def _fetch_klines(symbol: str, timeframe: str, start_ms: int, end_ms: int,
                  limit: int = 1000) -> list[list]:
    """Hit /api/v3/klines under the rate limiter. Empty list on failure."""
    import requests
    sym_bin = symbol.replace("/", "")
    interval = _BINANCE_INTERVAL.get(timeframe, timeframe)
    url = (f"https://api.binance.com/api/v3/klines?symbol={sym_bin}"
           f"&interval={interval}&startTime={start_ms}&endTime={end_ms}"
           f"&limit={min(int(limit), 1000)}")
    limiter = get_limiter("binance.com")
    weight = 2 if limit > 100 else 1   # /klines weight scales with limit
    with limiter.acquire(weight=weight):
        try:
            r = requests.get(url, timeout=15)
            limiter.react_to_response(r)
            if r.status_code != 200:
                return []
            return r.json()
        except requests.RequestException as exc:
            logger.warning("[REST %s/%s] %s", symbol, timeframe, exc)
            return []


def step_rest_topup(symbols: list[str], timeframes: list[str],
                    *, lookback_days: int = 30) -> dict:
    """Fill the recent gap from `last_known` to now via REST klines."""
    qdb = get_questdb()
    parquet = get_store()
    out: list[dict] = []

    for sym in symbols:
        for tf in timeframes:
            try:
                last_qdb = qdb.get_latest_candle_ts(sym, tf)
            except Exception:
                last_qdb = None
            try:
                last_pq = parquet.symbol_status(sym, timeframe=tf).latest
                last_pq_dt = (datetime.fromisoformat(str(last_pq).replace("Z", "+00:00"))
                              if last_pq else None)
            except Exception:
                last_pq_dt = None
            last = max([t for t in (last_qdb, last_pq_dt) if t is not None],
                       default=None)
            now_dt = datetime.now(timezone.utc)
            start = (last if last and last.tzinfo else
                     (last.replace(tzinfo=timezone.utc) if last else
                      now_dt - timedelta(days=lookback_days)))
            start_ms = int(start.timestamp() * 1000)
            end_ms   = int(now_dt.timestamp() * 1000)

            if (end_ms - start_ms) < 60 * 1000:
                continue   # gap is < 1 min, nothing to do

            written = 0
            cursor = start_ms
            while cursor < end_ms:
                rows = _fetch_klines(sym, tf, cursor, end_ms, limit=1000)
                if not rows:
                    break
                bars = [{
                    "timestamp": int(r[0]),
                    "open":  float(r[1]), "high": float(r[2]),
                    "low":   float(r[3]), "close": float(r[4]),
                    "volume": float(r[5]),
                } for r in rows]
                n = qdb.write_market_candles_bulk(sym, tf, bars)
                written += n
                cursor = int(rows[-1][0]) + 1

            out.append({"symbol": sym, "timeframe": tf,
                        "rest_bars_written": written,
                        "last_before": last.isoformat() if last else None})
            logger.info("[topup] %s/%s +%d bars", sym, tf, written)

    return {"step": "rest_topup", "results": out}


# ─── Step 3 — cross-check tail of QuestDB against Binance current state ───

def step_cross_check(symbols: list[str], timeframes: list[str],
                     *, last_n_bars: int = 100) -> dict:
    """Re-fetch the last N bars from REST and compare to QuestDB.

    If Binance has corrected a candle (it sometimes does within the first
    minute after close, e.g. for delayed tape data), overwrite our copy.
    """
    qdb = get_questdb()
    out: list[dict] = []
    for sym in symbols:
        for tf in timeframes:
            try:
                last_ts = qdb.get_latest_candle_ts(sym, tf)
            except Exception:
                last_ts = None
            if last_ts is None:
                continue
            end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
            interval_sec = {"1m": 60, "5m": 300, "15m": 900,
                            "1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 3600)
            start_ms = end_ms - last_n_bars * interval_sec * 1000
            rows = _fetch_klines(sym, tf, start_ms, end_ms, limit=last_n_bars)
            if not rows:
                continue
            bars = [{
                "timestamp": int(r[0]),
                "open":  float(r[1]), "high": float(r[2]),
                "low":   float(r[3]), "close": float(r[4]),
                "volume": float(r[5]),
            } for r in rows]
            n = qdb.write_market_candles_bulk(sym, tf, bars)
            out.append({"symbol": sym, "timeframe": tf, "cross_bars_refreshed": n})
    return {"step": "cross_check", "results": out}


# ─── Orchestration ─────────────────────────────────────────────────────────

def run(symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        *, skip_archive: bool = False, check_recent: int = 100,
        lookback_days: int = 30) -> dict:
    symbols = symbols or _watchlist()
    timeframes = timeframes or DEFAULT_TFS
    logger.info("=" * 60)
    logger.info("[binance_sync] %d symbols x %d timeframes", len(symbols), len(timeframes))
    logger.info("=" * 60)
    summary = {}
    if not skip_archive:
        summary["archive"]    = step_archive(symbols, timeframes)
    summary["rest_topup"]     = step_rest_topup(symbols, timeframes, lookback_days=lookback_days)
    if check_recent > 0:
        summary["cross_check"] = step_cross_check(symbols, timeframes, last_n_bars=check_recent)
    logger.info("[binance_sync] DONE")
    return summary


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--tfs", nargs="+", default=None)
    p.add_argument("--skip-archive", action="store_true")
    p.add_argument("--check-recent", type=int, default=100,
                   help="Re-fetch last N bars per (sym, tf); 0 to skip cross-check")
    p.add_argument("--lookback-days", type=int, default=30,
                   help="REST top-up window")
    args = p.parse_args()
    run(symbols=args.symbols, timeframes=args.tfs,
        skip_archive=args.skip_archive,
        check_recent=args.check_recent,
        lookback_days=args.lookback_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
