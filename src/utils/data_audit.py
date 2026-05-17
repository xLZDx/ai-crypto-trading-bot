"""
data_audit — read-only inventory of OHLCV / sentiment data on disk.

Used by the dashboard's Data Coverage panel and the backfill orchestrator
to decide which (symbol, timeframe) pairs are missing or stale.

The audit walks data/raw/*.csv.gz (current canonical layout) and reports
per (symbol, timeframe):
  - rows                — number of bars
  - first_ts, last_ts   — first / last timestamp parsed from the file
  - file_size_bytes     — gzip footprint
  - file_age_s          — seconds since the file was last modified
  - status              — 'present' | 'missing' | 'stale'
                          stale = last_ts more than max_lag_bars * tf_seconds
                          behind now (so daily files <2 days old are fresh
                          but minute files >1 hour old are stale).

For sentiment we audit data/parquet/_NEWS/news partitions and report the
yyyymm range plus row count.
"""
from __future__ import annotations

import gzip
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PARQUET_NEWS = PROJECT_ROOT / "data" / "parquet" / "_NEWS" / "news"

# Canonical watchlist for the audit grid. Used as a fallback when no 1s
# archives are discoverable on disk (e.g. fresh checkout). The live default
# is computed by discover_symbols() so dropping a new <SYM>_spot_1s.csv.gz
# into data/raw/historical/ extends coverage with no code change.
FALLBACK_SYMBOLS = (
    "BTC_USDT", "SOL_USDT", "ADA_USDT", "ETH_USDT", "BNB_USDT",
    "XRP_USDT", "DOGE_USDT", "TRX_USDT", "AVAX_USDT", "SHIB_USDT",
    "DOT_USDT", "LINK_USDT", "NEAR_USDT", "UNI_USDT", "LTC_USDT",
    "APT_USDT", "ATOM_USDT", "HBAR_USDT", "ICP_USDT", "SUI_USDT",
)


def discover_symbols(
    historical_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> tuple[str, ...]:
    """Scan disk for any 1s archive and return the symbols sorted.
    Looks for these patterns, in priority order:
      data/raw/historical/<SYM>_spot_1s.csv.gz   (deepest history)
      data/raw/historical/<SYM>_1s.csv.gz
      data/raw/<SYM>_1s.csv.gz                   (live tail)
    Returns the union — if a symbol has any 1s file we include it.
    Falls back to FALLBACK_SYMBOLS if no archives are present (which is
    really only the case on a fresh checkout)."""
    hist = Path(historical_dir) if historical_dir else (PROJECT_ROOT / "data" / "raw" / "historical")
    raw  = Path(raw_dir)        if raw_dir        else RAW_DIR
    found: set[str] = set()
    for d in (hist, raw):
        if not d.exists():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            n = p.name
            if not n.endswith("_1s.csv.gz"):
                continue
            # Strip the suffix and any "_spot" infix to recover the symbol.
            stem = n[:-len("_1s.csv.gz")]
            if stem.endswith("_spot"):
                stem = stem[:-len("_spot")]
            # Only accept conventional <BASE>_<QUOTE> shapes — guards
            # against accidentally matching "_funding" or other patterns.
            if "_" in stem and len(stem.split("_")) == 2:
                found.add(stem)
    if not found:
        return tuple(FALLBACK_SYMBOLS)
    return tuple(sorted(found))


# Live default — computed at import time. Callers who pass an explicit
# symbols=[...] list bypass this and get exactly what they ask for.
DEFAULT_SYMBOLS: tuple[str, ...] = discover_symbols()

# Timeframes we want covered for the multi-TF training initiative.
DEFAULT_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo")

# Seconds per bar for staleness math. 1mo is approximate (28-31 days).
_TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400,
    "1d": 86_400, "1w": 604_800, "1mo": 2_592_000,
}


def _mtime_to_ts(path: Path) -> str | None:
    """Convert the file's modification time to the audit's "YYYY-MM-DD HH:MM:SS"
    UTC format. Used as a cheap proxy for last_ts when fast=True (default).

    For ongoing-archive files (Binance dumps that the downloader keeps
    appending to), mtime ≈ time of the last write ≈ last bar boundary.
    For frozen historical archives, mtime is stable and "stale" classification
    still works correctly — it just reflects the file's last update rather
    than the last bar inside the file.

    Operator request 2026-05-15: the Data Coverage card never loaded because
    the previous implementation decompressed every gzip end-to-end (BTC 1m
    alone was 27s; 160 cells × 10 GB total → ~10 min cold scan that always
    exceeded the 2-min cache TTL). mtime is O(1) per file via stat().
    """
    try:
        dt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return None


def _parse_first_last_ts(path: Path) -> tuple[str | None, str | None, int]:
    """Two-line read: skip header, take the first data row's timestamp,
    then drain the file to the last non-empty line for `last_ts`. Returns
    (first, last, rows). rows is set to -1 because counting requires a
    full file traversal.

    SLOW PATH: this decompresses the entire gzip stream. Only call from
    `audit_coverage(..., precise=True)`. The default audit uses mtime
    instead (see _mtime_to_ts)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header:
                return (None, None, 0)
            first_line = f.readline()
            if not first_line:
                return (None, None, 0)
            first_ts = first_line.split(",", 1)[0].strip()
            tail = first_line
            for line in f:
                if line.strip():
                    tail = line
            last_ts = tail.split(",", 1)[0].strip()
            return (first_ts, last_ts, -1)
    except Exception as exc:
        logger.debug("data_audit: failed to read %s: %s", path, exc)
        return (None, None, 0)


def _staleness(tf: str, last_ts: str | None,
               now_utc: datetime | None = None) -> tuple[bool, float | None]:
    """Returns (is_stale, lag_seconds). is_stale = lag > 3 × bar_size."""
    if not last_ts:
        return (True, None)
    try:
        # Files store "YYYY-MM-DD HH:MM:SS" UTC.
        dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return (True, None)
    now = now_utc or datetime.now(timezone.utc)
    lag = (now - dt).total_seconds()
    bar = _TF_SECONDS.get(tf, 3600)
    return (lag > bar * 3, lag)


def audit_coverage(
    symbols: list[str] | tuple[str, ...] = DEFAULT_SYMBOLS,
    timeframes: list[str] | tuple[str, ...] = DEFAULT_TIMEFRAMES,
    *,
    raw_dir: Path | None = None,
    fast: bool = True,
    precise: bool = False,
) -> list[dict]:
    """Return [{symbol, timeframe, status, rows, first_ts, last_ts, lag_s,
    file_size_bytes, path}] for every cell in the grid.

    `fast=True` (default) skips full row counts and uses file mtime as the
    last-bar proxy. This makes the audit O(1) per file via stat(). With
    20 symbols × 8 TFs (~10 GB compressed) the audit finishes in <1s
    instead of ~10 min of gzip decompression. The Data Coverage card on
    the Strategy tab depends on this finishing inside its 2-min cache TTL.

    `precise=True` opt-in decompresses each gzip to read the exact first
    and last bar timestamps. Only use when you genuinely need the bar
    timestamps (e.g. a backfill orchestrator deciding what to download).
    With the project's 10 GB of compressed data this takes ~10 minutes.

    `fast=False` triggers full row counts AND precise=True semantics —
    even slower. Kept for backwards-compat with any caller that passes it.
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    now = datetime.now(timezone.utc)
    use_gzip_scan = precise or (not fast)
    out: list[dict] = []
    for sym in symbols:
        for tf in timeframes:
            path = raw_dir / f"{sym}_{tf}.csv.gz"
            row: dict = {
                "symbol":           sym,
                "timeframe":        tf,
                "path":             str(path),
                "exists":           path.exists(),
                "rows":             0,
                "first_ts":         None,
                "last_ts":          None,
                "lag_s":            None,
                "file_size_bytes":  0,
                "file_age_s":       None,
                "status":           "missing",
            }
            if not path.exists():
                out.append(row)
                continue
            try:
                stat = path.stat()
                row["file_size_bytes"] = stat.st_size
                row["file_age_s"]      = round((now.timestamp() - stat.st_mtime), 1)
            except OSError:
                pass
            first_ts: str | None = None
            last_ts:  str | None = None
            _rows: int = -1
            if use_gzip_scan:
                first_ts, last_ts, _rows = _parse_first_last_ts(path)
                if not fast:
                    # Full row count — slow but precise.
                    try:
                        with gzip.open(path, "rt", encoding="utf-8") as f:
                            next(f, None)  # header
                            _rows = sum(1 for _ in f)
                    except Exception:
                        _rows = -1
            else:
                # FAST PATH (default): use mtime as last_ts proxy. The Binance
                # archive downloader rewrites the .csv.gz every time it tops
                # up the file, so mtime closely tracks last bar boundary for
                # an actively-maintained archive. For frozen archives, mtime
                # is stable and the staleness check still classifies them
                # correctly (lag > 3 * bar_size → stale).
                last_ts = _mtime_to_ts(path)
            row["first_ts"] = first_ts
            row["last_ts"]  = last_ts
            row["rows"]     = _rows
            stale, lag = _staleness(tf, last_ts, now_utc=now)
            row["lag_s"]   = round(lag, 1) if lag is not None else None
            row["status"]  = "present" if not stale else "stale"
            out.append(row)
    return out


def audit_summary(rows: list[dict] | None = None) -> dict:
    """Bucket the matrix into a one-glance summary for the chip / banner."""
    rows = rows if rows is not None else audit_coverage()
    counts = {"present": 0, "stale": 0, "missing": 0}
    for r in rows:
        counts[r.get("status", "missing")] = counts.get(r.get("status", "missing"), 0) + 1
    return {
        "total":   len(rows),
        "present": counts["present"],
        "stale":   counts["stale"],
        "missing": counts["missing"],
        "pct_present": round(100 * counts["present"] / len(rows), 1) if rows else 0.0,
    }


def audit_sentiment() -> dict:
    """Inventory news / sentiment partitions. Reports yyyymm coverage so the
    operator can see how deep our sentiment history goes (currently shallow
    — we expect ~5 months from the existing CryptoCompare ingest)."""
    out: dict = {
        "available":     False,
        "partitions":    [],
        "yyyymm_first":  None,
        "yyyymm_last":   None,
        "rows_estimate": 0,
    }
    if not PARQUET_NEWS.exists():
        return out
    out["available"] = True
    parts = sorted(p.name for p in PARQUET_NEWS.iterdir()
                   if p.is_dir() and p.name.startswith("yyyymm="))
    out["partitions"] = parts
    if parts:
        out["yyyymm_first"] = parts[0].split("=", 1)[1]
        out["yyyymm_last"]  = parts[-1].split("=", 1)[1]
    # Fast row estimate: sum file sizes / ~rough avg row size. Skip if it's
    # going to be slow — operator can ask for a precise count later.
    try:
        total_bytes = sum(f.stat().st_size for f in PARQUET_NEWS.rglob("*.parquet"))
        # Sentiment news rows are ~200 bytes after parquet compression.
        out["rows_estimate"] = int(total_bytes // 200)
        out["bytes_total"]   = total_bytes
    except Exception:
        pass
    return out


# CLI: `python -m src.utils.data_audit` prints a quick summary.
if __name__ == "__main__":
    import json as _json
    rows = audit_coverage()
    summary = audit_summary(rows)
    sent = audit_sentiment()
    print(_json.dumps({"summary": summary, "sentiment": sent}, indent=2))
    # Print the matrix as a compact table.
    cols = list(DEFAULT_TIMEFRAMES)
    print(f"\n{'symbol':<10}" + "".join(f"{c:>6}" for c in cols))
    by_sym: dict[str, dict[str, str]] = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], {})[r["timeframe"]] = r["status"]
    glyph = {"present": "  Y  ", "stale": "  s  ", "missing": "  -  "}
    for sym in DEFAULT_SYMBOLS:
        line = f"{sym:<10}"
        for tf in cols:
            st = by_sym.get(sym, {}).get(tf, "missing")
            line += glyph.get(st, "  ?  ")
        print(line)
