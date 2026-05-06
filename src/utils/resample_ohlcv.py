"""
resample_ohlcv — convert 1s OHLCV archives into the higher timeframes used
by the bot, the trainers, and the strategies (1m, 5m, 15m, 1h, 4h, 1d, 1w,
1mo). Idempotent: re-running on the same source produces identical bytes.

Why resample instead of HTTP backfill from Binance?
  - The 1s archives in data/raw/historical/ already cover each symbol's
    listing date → present. BTC starts 2017-08-17, ETH 2017-08-17, ADA
    2018-04-17, SOL 2020-08-11 — every regime is present.
  - Internally consistent: every higher TF is derived from the same source,
    so there's no clock-skew between TFs.
  - No Binance rate limits (5m × 20 symbols × 6 years = 12M+ candles by API).
  - Reproducible: re-running gives identical output.

The 1s files are large — BTC is ~6 GB compressed / ~30 GB decompressed —
so this module reads in chunks and aggregates incrementally rather than
loading the whole frame.

Public surface:
  resample_symbol(symbol, source_path=None, timeframes=...) -> dict[tf, dict]
  resample_all(symbols=DEFAULT_SYMBOLS, ...)               -> dict[symbol, ...]
"""
from __future__ import annotations

import gzip
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
RAW_HIST_DIR   = PROJECT_ROOT / "data" / "raw" / "historical"

# Pandas resample rule per target timeframe. Anchored to the same right-edge
# convention Binance uses (label = bar OPEN time, closed='left'). Pandas
# 2.x deprecated 'H' and 'D' uppercase — use lowercase to silence the
# FutureWarning that was flooding the dashboard log.
_RESAMPLE_RULE = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
    "1w":  "1W-MON",   # week starts Monday — matches Binance UI default
    "1mo": "MS",       # month-start (well-defined; Binance 1M has quirks)
}

# Aggregation per OHLCV column. Source 1s rows are assumed to carry the
# Binance kline schema written by historical_backfill / archive_downloader:
# timestamp, open, high, low, close, volume, quote_volume, trades_count,
# taker_buy_base, taker_buy_quote.
_AGG = {
    "open":            "first",
    "high":            "max",
    "low":             "min",
    "close":           "last",
    "volume":          "sum",
    "quote_volume":    "sum",
    "trades_count":    "sum",
    "taker_buy_base":  "sum",
    "taker_buy_quote": "sum",
}

DEFAULT_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w", "1mo")
# 1m is excluded by default — we already have that on disk and re-deriving
# from 1s would shadow whatever live tail the bot is appending.

# Auto-discover symbols from the 1s archives present on disk. Drop a new
# <SYM>_spot_1s.csv.gz into data/raw/historical/ and DEFAULT_SYMBOLS picks
# it up next time the module imports (no code change needed).
from src.utils.data_audit import discover_symbols as _discover_symbols
DEFAULT_SYMBOLS: tuple[str, ...] = _discover_symbols()


def _candidate_source_paths(symbol: str) -> list[Path]:
    """Where to look for 1s data, in priority order. The historical archive
    is the deep one (years); the recent file in data/raw/ is the small live
    tail. We use whichever exists; if both, we'll prefer the bigger one."""
    return [
        RAW_HIST_DIR / f"{symbol}_spot_1s.csv.gz",
        RAW_HIST_DIR / f"{symbol}_1s.csv.gz",
        RAW_DIR      / f"{symbol}_1s.csv.gz",
    ]


def _pick_source(symbol: str, source_path: Path | None) -> Path | None:
    if source_path is not None:
        return source_path if source_path.exists() else None
    candidates = [p for p in _candidate_source_paths(symbol) if p.exists()]
    if not candidates:
        return None
    # Pick the largest — heuristic for "deepest history".
    return max(candidates, key=lambda p: p.stat().st_size)


def _stream_chunks(path: Path, chunksize: int = 250_000) -> Iterable[pd.DataFrame]:
    """Read the gzipped CSV in chunks. Each chunk is parsed with pandas
    (much faster than Python csv) but we don't hold the whole file at once
    — BTC's 1s archive is ~30 GB decompressed."""
    cols = ["timestamp", "open", "high", "low", "close", "volume",
            "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote"]
    dtype = {
        "open": "float32", "high": "float32", "low": "float32", "close": "float32",
        "volume": "float32", "quote_volume": "float32",
        "trades_count": "float32",  # written as float by some sources
        "taker_buy_base": "float32", "taker_buy_quote": "float32",
    }
    reader = pd.read_csv(
        path, compression="gzip",
        chunksize=chunksize, dtype=dtype,
        usecols=cols,
        parse_dates=["timestamp"],
    )
    for chunk in reader:
        chunk = chunk.set_index("timestamp", drop=True)
        yield chunk


def _resample_target(target_tf: str, agg_state: dict) -> pd.DataFrame:
    """Combine the per-chunk partial aggregates into final bars for this TF.
    agg_state[target_tf] is a list of partial DataFrames (each chunk's
    resample output). We concat them, then re-resample so any bar straddling
    chunk boundaries gets merged correctly."""
    parts = agg_state.get(target_tf) or []
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, axis=0)
    # The chunk-resample produced per-chunk partial bars. Re-aggregate so
    # cross-chunk bars merge properly. We do open/high/low/close as
    # first/max/min/last and sum the rest.
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum",
        "trades_count": "sum",
        "taker_buy_base": "sum", "taker_buy_quote": "sum",
    }
    return raw.groupby(level=0).agg(agg).sort_index()


def _write_csv_gz(df: pd.DataFrame, out_path: Path) -> int:
    """Write the result as gzipped CSV in the same schema as our existing
    files (timestamp as 'YYYY-MM-DD HH:MM:SS')."""
    if df.empty:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%d %H:%M:%S")
    out.index.name = "timestamp"
    # Atomic-ish write: write to .tmp, rename. Skips partial-file
    # corruption if the process dies mid-write.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out.to_csv(tmp, compression="gzip")
    os.replace(tmp, out_path)
    return len(out)


def resample_symbol(
    symbol: str,
    source_path: Path | None = None,
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    *,
    chunksize: int = 250_000,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """Resample one symbol's 1s archive into every requested timeframe.
    Returns {tf: {rows, out_path, status}} per target. Skips a TF if its
    rule isn't recognised or if the source is missing."""
    src = _pick_source(symbol, source_path)
    if src is None:
        return {
            tf: {"status": "no-source", "rows": 0, "out_path": None}
            for tf in timeframes
        }
    logger.info("resample %s: source=%s (%.1f MB)",
                symbol, src.name, src.stat().st_size / 1e6)

    # agg_state[tf] is a list of per-chunk partial-resampled DataFrames.
    agg_state: dict[str, list[pd.DataFrame]] = {tf: [] for tf in timeframes}
    rule_for_tf = {tf: _RESAMPLE_RULE[tf] for tf in timeframes
                   if tf in _RESAMPLE_RULE}

    bytes_total = src.stat().st_size
    bytes_seen  = 0
    chunks_seen = 0
    rows_seen   = 0
    start_t     = datetime.now(timezone.utc)
    for chunk in _stream_chunks(src, chunksize=chunksize):
        if chunk.empty:
            continue
        chunks_seen += 1
        rows_seen   += len(chunk)
        # Approximate progress by chunks (gzip can't tell us decompressed
        # bytes cheaply; multiplying chunksize by chunks_seen is close
        # enough for the UI).
        for tf, rule in rule_for_tf.items():
            partial = chunk.resample(rule, label="left", closed="left").agg(_AGG)
            if not partial.empty:
                agg_state[tf].append(partial)
        if progress and chunks_seen % 4 == 0:
            progress({
                "symbol":      symbol,
                "phase":       "stream",
                "chunks_seen": chunks_seen,
                "rows_seen":   rows_seen,
                "elapsed_s":   (datetime.now(timezone.utc) - start_t).total_seconds(),
            })

    # Merge per-chunk partials and write each TF.
    out: dict[str, dict] = {}
    for tf in timeframes:
        if tf not in rule_for_tf:
            out[tf] = {"status": "unsupported-rule", "rows": 0, "out_path": None}
            continue
        df = _resample_target(tf, agg_state)
        out_path = RAW_DIR / f"{symbol}_{tf}.csv.gz"
        n = _write_csv_gz(df, out_path)
        out[tf] = {"status": "written" if n else "empty",
                   "rows": n,
                   "out_path": str(out_path)}
        if progress:
            progress({"symbol": symbol, "phase": "wrote_" + tf, "rows": n})

    if progress:
        progress({
            "symbol":    symbol,
            "phase":     "done",
            "rows_seen": rows_seen,
            "elapsed_s": (datetime.now(timezone.utc) - start_t).total_seconds(),
        })
    return out


def resample_all(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    *,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """Run resample_symbol for every symbol in order. Returns a per-symbol
    summary dict for the UI. Skips symbols whose 1s archive is missing."""
    out: dict[str, dict] = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        if progress:
            progress({"phase": "symbol_start", "symbol": sym,
                      "i": i, "total": total})
        try:
            out[sym] = resample_symbol(sym, timeframes=timeframes,
                                       progress=progress)
        except Exception as exc:
            logger.exception("resample %s failed: %s", sym, exc)
            out[sym] = {"_error": f"{type(exc).__name__}: {exc}"}
        if progress:
            progress({"phase": "symbol_done", "symbol": sym,
                      "i": i + 1, "total": total})
    return out


# CLI for one-shot runs (operator can also trigger via the dashboard).
if __name__ == "__main__":
    import argparse, json as _json, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Resample 1s OHLCV → higher TFs")
    ap.add_argument("--symbol", help="Single symbol (e.g. BTC_USDT). Omit for all.")
    ap.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES),
                    help="Comma-separated list (5m,15m,1h,4h,1d,1w,1mo)")
    args = ap.parse_args()
    tfs = tuple(t.strip() for t in args.timeframes.split(",") if t.strip())
    def _cli_progress(ev: dict) -> None:
        sys.stderr.write(_json.dumps(ev) + "\n")
    if args.symbol:
        result = {args.symbol: resample_symbol(args.symbol, timeframes=tfs,
                                               progress=_cli_progress)}
    else:
        result = resample_all(timeframes=tfs, progress=_cli_progress)
    print(_json.dumps(result, indent=2, default=str))
