"""
onesec_topup — keep the 1s archives current.

The historical archives in data/raw/historical/<sym>_spot_1s.csv.gz come
from Binance's data.binance.vision dump and are typically a few days
behind real time. This module reads each archive's last timestamp and
fetches any missing 1s candles from the Binance REST API up to "now",
appending them as a new gzipped tail file:

  data/raw/<sym>_1s.csv.gz   (the live-tail file)

That tail file is already considered as a fallback source by the
resampler and audit modules. So once this top-up runs, the next resample
covers the full historical-archive → present span automatically.

Realistically: the top-up window is a few days × 86400 s × 20 symbols ≈
a few million rows total, well within Binance's 1000-candles-per-request
limit at ~5 req/s. Total runtime ~10 min for a fresh run, seconds when
already up to date.
"""
from __future__ import annotations

import csv
import gzip
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
RAW_HIST_DIR = PROJECT_ROOT / "data" / "raw" / "historical"

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
LIMIT_PER_REQ  = 1000   # Binance public-API max
SLEEP_BETWEEN  = 0.20   # 5 req/s — well under the public limit


def _read_last_ts(path: Path) -> datetime | None:
    """Stream-read the gzip and return the last timestamp parsed. Cheaper
    than loading the whole frame; same shape the resampler reads."""
    if not path.exists():
        return None
    last_line = None
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            f.readline()  # header
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line:
            return None
        ts_str = last_line.split(",", 1)[0].strip()
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except Exception as exc:
        logger.debug("read_last_ts %s: %s", path, exc)
        return None


def _archive_last_ts(symbol: str) -> datetime | None:
    """Return the latest timestamp across all 1s sources for this symbol —
    historical/<sym>_spot_1s, historical/<sym>_1s, raw/<sym>_1s. The
    biggest one is the deep history; the smaller two may have a fresher
    tail. We return the max so the top-up only fetches truly missing data."""
    tss: list[datetime] = []
    for p in (RAW_HIST_DIR / f"{symbol}_spot_1s.csv.gz",
              RAW_HIST_DIR / f"{symbol}_1s.csv.gz",
              RAW_DIR      / f"{symbol}_1s.csv.gz"):
        ts = _read_last_ts(p)
        if ts is not None:
            tss.append(ts)
    return max(tss) if tss else None


def topup_symbol(
    symbol: str,
    *,
    end_ts: datetime | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Fetch any missing 1s candles for `symbol` from its archive's last
    bar to `end_ts` (default: now). Writes/extends data/raw/<sym>_1s.csv.gz.

    Returns a small summary dict the caller can log."""
    end = end_ts or datetime.now(timezone.utc)
    last_archive = _archive_last_ts(symbol)
    if last_archive is None:
        logger.warning("%s: no archive on disk; cannot determine top-up start", symbol)
        return {"symbol": symbol, "status": "no-archive", "rows_added": 0}

    # We will append to the live-tail file. If it exists, read its own last
    # timestamp so we don't overwrite or duplicate.
    tail_path = RAW_DIR / f"{symbol}_1s.csv.gz"
    tail_last = _read_last_ts(tail_path)
    start = max(last_archive, tail_last) if tail_last else last_archive
    # Move to next 1s bucket so we don't refetch the last bar.
    start = start + timedelta(seconds=1)

    if start >= end:
        return {"symbol": symbol, "status": "fresh",
                "last_ts": start.isoformat(), "rows_added": 0}

    # If we have no tail file, write a fresh one with the standard header.
    write_header = not tail_path.exists()

    rows_added = 0
    sym_no_slash = symbol.replace("_", "")
    cur = start
    end_ms = int(end.timestamp() * 1000)
    started_at = time.time()
    # Open in append mode (gzip 'at' creates a new gzipped frame; readers
    # handle multi-frame gz files transparently).
    mode = "wt" if write_header else "at"
    with gzip.open(tail_path, mode=mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "open", "high", "low", "close", "volume",
                "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote",
            ])
        while cur.timestamp() * 1000 < end_ms:
            params = {
                "symbol":    sym_no_slash,
                "interval":  "1s",
                "startTime": int(cur.timestamp() * 1000),
                "limit":     LIMIT_PER_REQ,
            }
            try:
                r = requests.get(BINANCE_KLINES, params=params, timeout=10)
                if r.status_code == 429:
                    logger.warning("%s: 429 rate limited, sleeping 5s", symbol)
                    time.sleep(5)
                    continue
                r.raise_for_status()
                rows = r.json()
            except Exception as exc:
                logger.error("%s: top-up fetch failed: %s", symbol, exc)
                break
            if not rows:
                break
            for row in rows:
                ts = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                w.writerow([ts, row[1], row[2], row[3], row[4], row[5],
                            row[7], row[8], row[9], row[10]])
            rows_added += len(rows)
            cur = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc) + timedelta(seconds=1)
            if progress and rows_added % 10_000 < LIMIT_PER_REQ:
                progress({"phase": "topup", "symbol": symbol,
                          "rows_added": rows_added,
                          "cur_ts": cur.isoformat(),
                          "elapsed_s": time.time() - started_at})
            time.sleep(SLEEP_BETWEEN)

    return {
        "symbol":     symbol,
        "status":     "ok",
        "rows_added": rows_added,
        "last_ts":    cur.isoformat(),
        "elapsed_s":  round(time.time() - started_at, 1),
    }


def topup_all(
    symbols: tuple[str, ...] | None = None,
    *,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """Top up every symbol in `symbols`. Defaults to whatever data_audit
    discovers on disk."""
    if symbols is None:
        from src.utils.data_audit import discover_symbols
        symbols = discover_symbols()
    out: dict[str, dict] = {}
    for sym in symbols:
        if progress:
            progress({"phase": "symbol_start", "symbol": sym})
        try:
            out[sym] = topup_symbol(sym, progress=progress)
        except Exception as exc:
            out[sym] = {"_error": f"{type(exc).__name__}: {exc}"}
        if progress:
            progress({"phase": "symbol_done", "symbol": sym, "result": out[sym]})
    return out


if __name__ == "__main__":
    import argparse, json as _j, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Top up 1s archives to current time")
    ap.add_argument("--symbol")
    args = ap.parse_args()
    def _cb(ev): sys.stderr.write(_j.dumps(ev) + "\n")
    if args.symbol:
        print(_j.dumps(topup_symbol(args.symbol, progress=_cb), indent=2))
    else:
        print(_j.dumps(topup_all(progress=_cb), indent=2))
