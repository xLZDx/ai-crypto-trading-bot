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

# Canonical watchlist for the audit grid. Mirrors the bot's training symbols.
DEFAULT_SYMBOLS = (
    "BTC_USDT", "SOL_USDT", "ADA_USDT", "ETH_USDT", "BNB_USDT",
    "XRP_USDT", "DOGE_USDT", "TRX_USDT", "AVAX_USDT", "SHIB_USDT",
    "DOT_USDT", "LINK_USDT", "NEAR_USDT", "UNI_USDT", "LTC_USDT",
    "APT_USDT", "ATOM_USDT", "HBAR_USDT", "ICP_USDT", "SUI_USDT",
)

# Timeframes we want covered for the multi-TF training initiative.
DEFAULT_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo")

# Seconds per bar for staleness math. 1mo is approximate (28-31 days).
_TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400,
    "1d": 86_400, "1w": 604_800, "1mo": 2_592_000,
}


def _parse_first_last_ts(path: Path) -> tuple[str | None, str | None, int]:
    """Cheap two-line read: skip header, take the first data row's timestamp,
    then seek to ~last 2KB to grab the trailing line. Returns (first, last,
    rows). rows is set to -1 if we don't know cheaply (we don't count by
    default — full count requires reading the whole file)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header:
                return (None, None, 0)
            first_line = f.readline()
            if not first_line:
                return (None, None, 0)
            first_ts = first_line.split(",", 1)[0].strip()
            # Tail: read full tail (gzip doesn't support cheap seek-from-end).
            # We bound by reading only the last 4KB of the *decompressed*
            # stream by doing a small-buffer drain — still O(file) but no
            # full parse. For the audit panel this is fast enough; the
            # alternative is sampling, which gives wrong "last_ts".
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
) -> list[dict]:
    """Return [{symbol, timeframe, status, rows, first_ts, last_ts, lag_s,
    file_size_bytes, path}] for every cell in the grid. fast=True skips
    full row counts (which would require reading the entire gzip stream).
    Set fast=False for an accurate row count — significantly slower."""
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    now = datetime.now(timezone.utc)
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
            first_ts, last_ts, _rows = _parse_first_last_ts(path)
            if not fast:
                # Full row count — slow but precise.
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        next(f, None)  # header
                        _rows = sum(1 for _ in f)
                except Exception:
                    _rows = -1
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
