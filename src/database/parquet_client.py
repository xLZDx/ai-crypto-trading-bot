"""
ParquetClient — replaces QuestDBClient with a DuckDB + Parquet stack.

Drop-in API compatibility: every public method on QuestDBClient is mirrored
here with the same signature, so callers don't need to change code beyond
the import path.

Why this design (Route B):
  - No daemon. No Docker. No port. No "container died" failure mode.
  - Single-writer constraint matches the bot's sequential loop (CLAUDE.md).
  - DuckDB queries Parquet directly with predicate pushdown — fast on the
    bot's actual scale (< 1k writes/sec, < 100k rows/query).
  - Storage is plain files on D:. Backup = `cp -r data/db ~/backup`.

Storage layout:
    data/db/
        hot/
            market_data/symbol=BTC_USDT/timeframe=1h/yyyymm=202605/*.parquet
            model_signals/symbol=BTC_USDT/yyyymm=202605/*.parquet
            training_telemetry/model=tft/yyyymm=202605/*.parquet
            agent_heartbeats/yyyymmdd=20260505/*.parquet
            news_sentiment/source=cryptopanic/yyyymm=202605/*.parquet
        cold/
            trade_events/yyyymm=202605/*.parquet
            strategy_performance/yyyymm=202605/*.parquet
            backtest_results/yyyymm=202605/*.parquet
            csv_ingestion_log/yyyymm=202605/*.parquet
            training_runs/yyyymm=202605/*.parquet
            model_wf_folds/yyyymm=202605/*.parquet
            testnet_trades/yyyymm=202605/*.parquet
            testnet_session_stats/yyyymm=202605/*.parquet

Write strategy: append-buffer-flush
  - Each table has its own in-memory buffer (list[dict]).
  - Flushes happen when:
      (a) buffer hits PARQUET_DB_FLUSH_ROWS (default 5000), or
      (b) PARQUET_DB_FLUSH_S elapse since last flush (default 30 s),
      (c) close() / flush_all() is called explicitly.
  - Each flush writes a fresh `<UTC-iso>_<count>.parquet` file in the right
    partition. Compaction is a separate concern (cron / manual).

Read strategy:
  - DuckDB connection (in-memory) is created on first query.
  - SELECT statements run via `read_parquet('<glob>')` which gives DuckDB
    predicate pushdown over the partition columns.
  - "Latest" queries union the in-memory buffer so writes-since-last-flush
    are visible immediately.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Tuneables (mirror src.utils.config so a shared config import isn't required) ──

try:
    from src.utils.config import (
        PARQUET_DB_DIR        as _CFG_DB_DIR,
        PARQUET_DB_FLUSH_S    as _CFG_FLUSH_S,
        PARQUET_DB_FLUSH_ROWS as _CFG_FLUSH_ROWS,
    )
except Exception:
    _CFG_DB_DIR = "data/db"
    _CFG_FLUSH_S = 30.0
    _CFG_FLUSH_ROWS = 5000


# ── Schema descriptor ────────────────────────────────────────────────────────
# Each entry: (db, partition_keys, column_specs).
# partition_keys are columns we use to bucket rows into directories. The
# remainder of each row stays inside the Parquet file. Choose partition keys
# that match the most common WHERE clauses (so DuckDB can prune partitions).

# column type tokens: "ts" = DateTime64 → use _to_dt; "i" = int; "f" = float;
#                    "s" = string;       "b" = bool (stored as uint8).
_TABLES: dict[str, dict[str, Any]] = {

    # ── HOT ──────────────────────────────────────────────────────────────────

    "market_data": {
        "db": "hot", "partition": ("symbol", "timeframe", "yyyymm"),
        "cols": {
            "ts": "ts", "symbol": "s", "timeframe": "s",
            "open": "f", "high": "f", "low": "f", "close": "f",
            "volume": "f", "funding_rate": "f",
        },
    },
    "model_signals": {
        "db": "hot", "partition": ("symbol", "yyyymm"),
        "cols": {
            "ts": "ts", "symbol": "s",
            "signal_rsi": "f", "signal_macd": "f", "signal_bb": "f",
            "signal_ensemble": "f", "signal_vwap": "f", "signal_donchian": "f",
            "signal_keltner": "f", "signal_funding": "f",
            "signal_vol_breakout": "f", "signal_supertrend": "f",
            "signal_ml_long": "f", "signal_ml_short": "f",
            "signal_scalping": "f", "signal_trend": "f", "signal_meta": "f",
            "regime": "i", "meta_label": "i", "meta_prob": "f",
            "garch_vol": "f", "ou_zscore": "f", "close": "f",
        },
    },
    "training_telemetry": {
        "db": "hot", "partition": ("model", "yyyymm"),
        "cols": {
            "ts": "ts", "model": "s", "run_id": "s", "hardware": "s",
            "epoch": "i", "train_loss": "f", "val_loss": "f",
            "accuracy": "f", "sharpe": "f", "learning_rate": "f",
            "batch_size": "i", "seq_len": "i", "n_samples": "i",
        },
    },
    "agent_heartbeats": {
        "db": "hot", "partition": ("yyyymmdd",),
        "cols": {
            "ts": "ts", "agent": "s", "status": "s", "current_task": "s",
            "cpu_pct": "f", "mem_mb": "f",
        },
    },
    "news_sentiment": {
        "db": "hot", "partition": ("source", "yyyymm"),
        "cols": {
            "ts": "ts", "source": "s", "sentiment": "s", "coins": "s",
            "score": "f", "headline": "s", "url": "s",
        },
    },

    # ── COLD ─────────────────────────────────────────────────────────────────

    "trade_events": {
        "db": "cold", "partition": ("yyyymm",),
        "cols": {
            "ts": "ts", "trade_id": "s", "symbol": "s", "strategy": "s",
            "market": "s", "direction": "i",
            "entry_price": "f", "exit_price": "f", "size_usd": "f",
            "pnl_usd": "f", "fees_usd": "f", "bars_held": "i", "is_live": "b",
        },
    },
    "strategy_performance": {
        "db": "cold", "partition": ("yyyymm",),
        "cols": {
            "ts": "ts", "strategy": "s", "symbol": "s", "is_live": "b",
            "balance": "f", "total_pnl": "f", "pnl_pct": "f",
            "win_rate": "f", "n_trades": "i", "n_wins": "i",
            "sharpe": "f", "wf_mean_sharpe": "f", "wf_consistency": "f",
        },
    },
    "backtest_results": {
        "db": "cold", "partition": ("yyyymm",),
        "cols": {
            "ts": "ts", "run_id": "s", "strategy": "s", "symbol": "s",
            "total_pnl": "f", "gross_pnl": "f", "total_fees": "f",
            "sharpe": "f", "win_rate": "f", "max_drawdown": "f",
            "n_trades": "i", "wf_mean_sharpe": "f", "wf_consistency": "f",
            "wf_decay": "f",
        },
    },
    "csv_ingestion_log": {
        "db": "cold", "partition": ("yyyy",),
        "cols": {
            "ts": "ts", "filename": "s", "source_path": "s",
            "symbol": "s", "timeframe": "s",
            "rows_written": "i", "file_size_bytes": "i",
            "first_bar_ts": "ts", "last_bar_ts": "ts",
        },
    },
    "training_runs": {
        "db": "cold", "partition": ("yyyymm",),
        "cols": {
            "ts": "ts", "run_id": "s", "model_name": "s", "strategy": "s",
            "symbol": "s", "timeframe": "s", "trigger": "s",
            "start_ts": "ts", "end_ts": "ts",
            "duration_secs": "f", "train_rows": "i", "val_rows": "i",
            "n_wf_folds": "i", "best_epoch": "i",
            "final_train_loss": "f", "final_val_loss": "f",
            "early_stopped": "b",
            "oos_sharpe": "f", "oos_win_rate": "f", "oos_max_drawdown": "f",
            "n_oos_trades": "i",
            "hyperparams_json": "s", "feature_list_json": "s", "notes": "s",
        },
    },
    "model_wf_folds": {
        "db": "cold", "partition": ("yyyymm",),
        "cols": {
            "ts": "ts", "run_id": "s", "model_name": "s", "fold_index": "i",
            "train_start": "ts", "train_end": "ts",
            "test_start": "ts",  "test_end": "ts",
            "train_rows": "i", "test_rows": "i",
            "oos_sharpe": "f", "oos_pnl": "f", "oos_win_rate": "f",
            "oos_max_dd": "f", "n_trades": "i",
        },
    },
    "testnet_trades": {
        "db": "cold", "partition": ("yyyymm",),
        "cols": {
            "ts": "ts", "trade_id": "s", "symbol": "s", "strategy": "s",
            "model": "s", "exit_reason": "s", "direction": "i", "is_live": "b",
            "entry_ts": "ts", "exit_ts": "ts",
            "entry_price": "f", "exit_price": "f", "size_usd": "f",
            "pnl_usd": "f", "fees_usd": "f", "funding_pnl": "f",
            "net_pnl": "f", "bars_held": "i", "meta_label": "i", "regime": "i",
            "garch_vol_at_entry": "f", "stop_loss": "f", "take_profit": "f",
            "meta_prob": "f", "signal_strength": "f",
        },
    },
    "testnet_session_stats": {
        "db": "cold", "partition": ("yyyymm",),
        "cols": {
            "ts": "ts", "session_id": "s", "strategy": "s", "symbol": "s",
            "balance": "f", "total_pnl": "f", "unrealized_pnl": "f",
            "n_open_trades": "i", "n_closed_trades": "i",
            "win_rate": "f", "sharpe": "f", "max_drawdown": "f",
            "funding_collected": "f",
        },
    },

    # ── MARKET CONTEXT (external signals for ML features) ─────────────────

    "open_interest": {
        # Binance Futures 1h OI. Written by open_interest_downloader.py.
        # Trainer joins on exact timestamp for 1h models only (no resampling).
        "db": "hot", "partition": ("symbol", "yyyymm"),
        "cols": {
            "ts": "ts", "symbol": "s",
            "oi_base": "f",       # OI in base asset (e.g. BTC)
            "oi_usdt": "f",       # OI in USDT
        },
    },
    "fear_greed": {
        # Alternative.me daily Fear & Greed index (0-100). Global, no symbol.
        # Trainer joins on exact date for 1d models only (no forward-fill).
        "db": "hot", "partition": ("yyyymm",),
        "cols": {
            "ts": "ts",
            "fear_greed": "i",    # 0=Extreme Fear, 100=Extreme Greed
            "fg_label": "s",      # text label
        },
    },
    "liquidations": {
        # Per-symbol hourly liquidation volume. Source: Coinglass (with key)
        # or Bybit/OKX free APIs. Trainer joins on exact timestamp for 1h only.
        "db": "hot", "partition": ("symbol", "yyyymm"),
        "cols": {
            "ts": "ts", "symbol": "s",
            "liq_long_usd": "f",  # long liquidations (USD)
            "liq_short_usd": "f", # short liquidations (USD)
            "liq_total_usd": "f", # total
        },
    },
}


def _safe_partition(v: str) -> str:
    """ASCII-safe directory token."""
    return str(v).replace("/", "_").replace(" ", "_").replace(",", "_")


def _now_dt() -> datetime:
    return datetime.now(tz=timezone.utc)


def _to_dt(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)):
        if ts < 1e12:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        elif ts < 1e15:
            return datetime.fromtimestamp(ts / 1e3, tz=timezone.utc)
        else:
            return datetime.fromtimestamp(ts / 1e9, tz=timezone.utc)
    if isinstance(ts, str):
        s = ts.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(s.split("+")[0], fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            return None
    if hasattr(ts, "timestamp"):
        return datetime.fromtimestamp(ts.timestamp(), tz=timezone.utc)
    return None


def _coerce_row(row: dict[str, Any], cols: dict[str, str]) -> dict[str, Any]:
    """Coerce row values into the types declared by the schema. Missing
    columns get a zero-equivalent so the schema stays uniform across files."""
    out: dict[str, Any] = {}
    for col, kind in cols.items():
        v = row.get(col)
        if kind == "ts":
            out[col] = _to_dt(v) if v is not None else _now_dt()
        elif kind == "i":
            try:
                out[col] = int(v) if v is not None else 0
            except (TypeError, ValueError):
                out[col] = 0
        elif kind == "f":
            try:
                out[col] = float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                out[col] = 0.0
        elif kind == "b":
            out[col] = 1 if v else 0
        else:  # 's'
            out[col] = str(v) if v is not None else ""
    return out


# ── Singleton ────────────────────────────────────────────────────────────────

_client_instance: "ParquetClient | None" = None
_client_lock = threading.Lock()


def get_client() -> "ParquetClient":
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                _client_instance = ParquetClient()
    return _client_instance


# ── ParquetClient ────────────────────────────────────────────────────────────

class ParquetClient:
    """
    DuckDB + Parquet store. Drop-in for QuestDBClient.
    Single-writer (the bot's sequential loop). Thread-safe within one process.
    """

    def __init__(self,
                 base_dir: str | Path | None = None,
                 flush_s: float | None = None,
                 flush_rows: int | None = None,
                 legacy_parquet_dir: str | Path | None = None) -> None:
        base = Path(base_dir) if base_dir else (PROJECT_ROOT / _CFG_DB_DIR)
        self.base_dir = base.resolve()
        self.flush_s = float(flush_s) if flush_s is not None else float(_CFG_FLUSH_S)
        self.flush_rows = int(flush_rows) if flush_rows is not None else int(_CFG_FLUSH_ROWS)
        # Legacy parquet_store layout (long-history OHLCV). Overridable for
        # tests that want to feed a synthetic legacy dir.
        self._LEGACY_PARQUET_DIR = (
            Path(legacy_parquet_dir).resolve() if legacy_parquet_dir
            else (PROJECT_ROOT / "data" / "parquet").resolve()
        )

        self._buf_lock = threading.Lock()
        self._buffers: dict[str, list[dict]] = {t: [] for t in _TABLES}
        self._last_flush: dict[str, float] = {t: time.time() for t in _TABLES}

        self._duck_lock = threading.Lock()
        self._duck = None       # lazy

        self._available: bool | None = None
        # Eager bootstrap so reads return [] cleanly even before any write.
        self._ensure_dirs()

    # ── Availability ──────────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        try:
            (self.base_dir / "hot").mkdir(parents=True, exist_ok=True)
            (self.base_dir / "cold").mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("ParquetClient: cannot create %s -- %s", self.base_dir, exc)

    def is_available(self, force: bool = False) -> bool:
        # The store is "available" iff we can reach the data dir AND DuckDB
        # imports. Cached after the first probe; idempotent.
        if self._available is not None and not force:
            return self._available
        try:
            import duckdb  # noqa: F401
        except Exception as exc:
            logger.warning("ParquetClient: duckdb missing -- %s", exc)
            self._available = False
            return False
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._available = True
        except Exception as exc:
            logger.warning("ParquetClient: data dir not writable -- %s", exc)
            self._available = False
        return self._available

    # ── DuckDB lazy connection ────────────────────────────────────────────

    def _conn(self):
        with self._duck_lock:
            if self._duck is not None:
                return self._duck
            import duckdb
            con = duckdb.connect(":memory:")
            tmp = (self.base_dir / "_duckdb_tmp").as_posix()
            try:
                Path(tmp).mkdir(parents=True, exist_ok=True)
                con.execute(f"PRAGMA temp_directory='{tmp}'")
            except Exception:
                pass
            self._duck = con
            return con

    # ── Table layout helpers ──────────────────────────────────────────────

    def _table_dir(self, table: str) -> Path:
        spec = _TABLES.get(table)
        db = spec["db"] if spec else "cold"
        return self.base_dir / db / table

    def _glob(self, table: str) -> str:
        return (self._table_dir(table) / "**" / "*.parquet").as_posix()

    def _partition_dir(self, table: str, row: dict[str, Any]) -> Path:
        """Compute the on-disk partition directory for one row."""
        spec = _TABLES.get(table) or {}
        keys: tuple[str, ...] = spec.get("partition", ()) or ()
        d = self._table_dir(table)
        ts = row.get("ts")
        if not isinstance(ts, datetime):
            ts = _to_dt(ts) or _now_dt()
        for key in keys:
            if key == "yyyymm":
                d = d / f"yyyymm={ts.strftime('%Y%m')}"
            elif key == "yyyymmdd":
                d = d / f"yyyymmdd={ts.strftime('%Y%m%d')}"
            elif key == "yyyy":
                d = d / f"yyyy={ts.strftime('%Y')}"
            else:
                v = _safe_partition(row.get(key, "_"))
                d = d / f"{key}={v}"
        return d

    # ── Write path ────────────────────────────────────────────────────────

    def _enqueue(self, table: str, rows: list[dict]) -> bool:
        spec = _TABLES.get(table)
        if spec is None:
            logger.debug("ParquetClient: unknown table %s -- dropping %d rows",
                         table, len(rows))
            return False
        cols = spec["cols"]
        # When a string column is also a partition key, sanitise the row
        # value so the in-file column matches the directory token. DuckDB's
        # hive_partitioning reads the path-derived value (BTC_USDT) — if the
        # row carries 'BTC/USDT' verbatim, WHERE filters miss the partition.
        part_keys = [k for k in (spec.get("partition") or ())
                     if k in cols and cols[k] == "s"]
        if part_keys:
            for r in rows:
                for k in part_keys:
                    if r.get(k) is not None:
                        r[k] = _safe_partition(r[k])
        coerced = [_coerce_row(r, cols) for r in rows]
        do_flush = False
        with self._buf_lock:
            buf = self._buffers.setdefault(table, [])
            buf.extend(coerced)
            now = time.time()
            since = now - self._last_flush.get(table, 0.0)
            if len(buf) >= self.flush_rows or since >= self.flush_s:
                do_flush = True
        if do_flush:
            return self._flush(table)
        return True

    def _flush(self, table: str) -> bool:
        with self._buf_lock:
            rows = self._buffers.get(table) or []
            if not rows:
                self._last_flush[table] = time.time()
                return True
            self._buffers[table] = []
            self._last_flush[table] = time.time()
        # Group rows by partition dir.
        groups: dict[str, list[dict]] = {}
        for r in rows:
            d = self._partition_dir(table, r).as_posix()
            groups.setdefault(d, []).append(r)
        ok = True
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            logger.warning("ParquetClient: pyarrow missing -- %s", exc)
            return False
        for dpath, grows in groups.items():
            try:
                Path(dpath).mkdir(parents=True, exist_ok=True)
                ts_now = _now_dt().strftime("%Y%m%dT%H%M%S%f")
                fname = f"{ts_now}_{len(grows):05d}.parquet"
                fp = Path(dpath) / fname
                table_arrow = pa.Table.from_pylist(grows)
                pq.write_table(table_arrow, fp.as_posix(),
                               compression="zstd", use_dictionary=True)
            except Exception as exc:
                logger.warning("ParquetClient: write %s failed -- %s", dpath, exc)
                ok = False
        return ok

    def flush_all(self) -> bool:
        ok = True
        for t in list(_TABLES):
            if not self._flush(t):
                ok = False
        return ok

    def close(self) -> None:
        self.flush_all()
        with self._duck_lock:
            if self._duck is not None:
                try:
                    self._duck.close()
                except Exception:
                    pass
                self._duck = None

    def insert_rows(self, table: str, rows: list[dict]) -> bool:
        if not rows:
            return True
        if not self.is_available():
            return False
        return self._enqueue(table, rows)

    # ── Query path ────────────────────────────────────────────────────────

    def _has_any_files(self, table: str) -> bool:
        d = self._table_dir(table)
        if not d.exists():
            return False
        try:
            return any(p.suffix == ".parquet" for p in d.rglob("*.parquet"))
        except Exception:
            return False

    def query(self, sql: str, params: list | tuple | None = None) -> list[dict]:
        """Execute SQL against the Parquet store via DuckDB.

        Convention: callers reference tables by their bare name
        (e.g. `SELECT * FROM market_data WHERE symbol = ?`). We
        rewrite each known table reference to a `read_parquet(...)` glob
        before running the query. Tables with no files yet return [].

        Phase A5 (2026-05-12): added `params` for parameterized
        execution. Callers should pass values via `?` placeholders +
        params list instead of f-string interpolation. The existing
        f-string call sites are being migrated as part of Phase A5.
        """
        if not self.is_available():
            return []
        rewritten, missing = self._rewrite_table_refs(sql)
        if missing:
            logger.debug("ParquetClient: missing-table fallback for %s", missing)
            return []
        # Serialize execute() across threads. DuckDB's Python connection
        # is NOT thread-safe — concurrent execute() on the same con
        # triggers C++ assertion failures (e.g. "unique_ptr NULL"
        # INTERNAL Errors that abort the process). The Flask dashboard
        # has many concurrent request threads + an error-monitor thread
        # all reading via this client, so the abort took the whole
        # dashboard down on 2026-05-08. Holding _duck_lock for the
        # entire execute+fetch keeps everything single-threaded at the
        # DuckDB layer without us needing per-thread cursors.
        try:
            con = self._conn()
            with self._duck_lock:
                if params is None:
                    res = con.execute(rewritten).fetch_arrow_table()
                else:
                    res = con.execute(rewritten, list(params)).fetch_arrow_table()
            return res.to_pylist()
        except Exception as exc:
            logger.warning("ParquetClient: query failed -- %s | SQL: %s",
                           exc, rewritten[:200])
            return []

    def query_df(self, sql: str, params: list | tuple | None = None):
        """Same as `query()` but returns a pandas DataFrame.

        Phase A5 (2026-05-12): `params` mirror argument added for
        parameterized execution; pass values via `?` placeholders.
        """
        import pandas as pd
        if not self.is_available():
            return pd.DataFrame()
        rewritten, missing = self._rewrite_table_refs(sql)
        if missing:
            return pd.DataFrame()
        try:
            con = self._conn()
            with self._duck_lock:
                if params is None:
                    return con.execute(rewritten).df()
                return con.execute(rewritten, list(params)).df()
        except Exception as exc:
            logger.warning("ParquetClient: query_df failed -- %s", exc)
            return pd.DataFrame()

    def exec_ddl(self, sql: str) -> bool:
        """Schema-as-files: there are no real DDL statements to execute.
        Treat as a no-op for QuestDBClient compatibility."""
        return True

    def _has_legacy_parquet(self) -> bool:
        try:
            return (self._LEGACY_PARQUET_DIR.exists()
                    and any(self._LEGACY_PARQUET_DIR.rglob("*.parquet")))
        except Exception:
            return False

    def _market_data_legacy_subquery(self) -> str:
        """SELECT subquery exposing the legacy data/parquet/{SYM}/{TF}/yyyymm=*/
        layout with the SAME column shape as the new market_data schema:
        ts, symbol, timeframe, open, high, low, close, volume, funding_rate.

        Notes:
        - legacy column is `timestamp` (TIMESTAMP, no tz) — renamed to `ts`
        - symbol and timeframe come from the path, not the file (the legacy
          layout pre-dates hive_partitioning). DuckDB's `filename=true`
          option exposes __filename. We normalize backslashes to forward
          slashes (Windows path separators trip RE2's backslash escaping)
          and then use a simple `/`-separated regex.
        - legacy files lack funding_rate; we fill NULL.
        - extra legacy columns (quote_volume, trades_count, taker_buy_*)
          are dropped here so the UNION schemas match.
        """
        glob = (self._LEGACY_PARQUET_DIR / "*" / "*" / "yyyymm=*" / "*.parquet").as_posix()
        # chr(92) = backslash, used to keep the SQL string itself
        # backslash-free (DuckDB's RE2 inside string literals + Python
        # escape semantics make the obvious `[/\\]` regex unreliable).
        normalized_path = "replace(filename, chr(92), '/')"
        return (
            "SELECT "
            "  CAST(timestamp AS TIMESTAMP) AS ts, "
            f"  regexp_extract({normalized_path}, "
            "    'parquet/([^/]+)/([^/]+)/yyyymm=', 1) "
            "    AS symbol, "
            f"  regexp_extract({normalized_path}, "
            "    'parquet/([^/]+)/([^/]+)/yyyymm=', 2) "
            "    AS timeframe, "
            "  open, high, low, close, volume, "
            "  CAST(NULL AS DOUBLE) AS funding_rate "
            f"FROM read_parquet('{glob}', filename=true, union_by_name=true)"
        )

    def _rewrite_table_refs(self, sql: str) -> tuple[str, str | None]:
        """Replace bare table names with read_parquet() globs. Returns
        (rewritten_sql, first_missing_table_name | None).

        Special case: market_data queries get UNION'd with the legacy
        data/parquet/ store so backtest-history reads + new live writes
        appear under one table name."""
        import re
        out = sql
        for table in _TABLES:
            # word-boundary match so "market_data" doesn't match "model_market_data"
            pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(table) + r"(?![A-Za-z0-9_])")
            if not pattern.search(out):
                continue

            new_has = self._has_any_files(table)
            legacy_has = (table == "market_data" and self._has_legacy_parquet())

            if not new_has and not legacy_has:
                return out, table

            new_subquery = ""
            if new_has:
                glob = self._glob(table)
                # DuckDB needs single-quoted glob, hive_partitioning=1 surfaces
                # the partition columns as real columns.
                new_subquery = (f"read_parquet('{glob}', "
                                f"hive_partitioning=1, union_by_name=true)")

            if legacy_has:
                legacy_sub = self._market_data_legacy_subquery()
                if new_subquery:
                    # New rows + legacy long-history under the same table name.
                    replacement = (
                        "(SELECT ts, symbol, timeframe, open, high, low, close, "
                        "volume, funding_rate FROM " + new_subquery + " "
                        "UNION ALL " + legacy_sub + ")"
                    )
                else:
                    replacement = "(" + legacy_sub + ")"
            else:
                replacement = new_subquery

            out = pattern.sub(replacement, out)
        return out, None

    # ── Compatibility shim: write_ilp ─────────────────────────────────────
    # Some legacy callers still emit ILP strings. Parse + route to insert_rows.

    def write_ilp(self, lines: list[str]) -> bool:
        if not lines:
            return True
        if not self.is_available():
            return False
        by_table: dict[str, list[dict]] = {}
        for line in lines:
            parsed = _parse_ilp_line(line)
            if parsed is None:
                continue
            table, row = parsed
            by_table.setdefault(table, []).append(row)
        ok = True
        for table, rows in by_table.items():
            if not self.insert_rows(table, rows):
                ok = False
        return ok

    # ── High-level write helpers (mirror QuestDBClient) ───────────────────

    def write_market_candle(self, symbol: str, timeframe: str, bar: dict) -> bool:
        ts = _to_dt(bar.get("timestamp") or bar.get("ts"))
        if ts is None:
            return False
        return self.insert_rows("market_data", [{
            "ts": ts, "symbol": str(symbol), "timeframe": str(timeframe),
            "open":   bar.get("open", 0),
            "high":   bar.get("high", 0),
            "low":    bar.get("low", 0),
            "close":  bar.get("close", 0),
            "volume": bar.get("volume", 0),
            "funding_rate": bar.get("funding_rate") or 0,
        }])

    def write_market_candles_bulk(self, symbol: str, timeframe: str,
                                   bars: list[dict], batch: int = 5000) -> int:
        rows: list[dict] = []
        written = 0
        for bar in bars:
            ts = _to_dt(bar.get("timestamp") or bar.get("ts"))
            if ts is None:
                continue
            rows.append({
                "ts": ts, "symbol": str(symbol), "timeframe": str(timeframe),
                "open":   bar.get("open", 0),
                "high":   bar.get("high", 0),
                "low":    bar.get("low", 0),
                "close":  bar.get("close", 0),
                "volume": bar.get("volume", 0),
                "funding_rate": bar.get("funding_rate") or 0,
            })
            if len(rows) >= batch:
                if self.insert_rows("market_data", rows):
                    written += len(rows)
                rows = []
        if rows and self.insert_rows("market_data", rows):
            written += len(rows)
        return written

    def write_trade(self, trade: dict) -> bool:
        ts = _to_dt(trade.get("timestamp") or trade.get("ts")) or _now_dt()
        return self.insert_rows("trade_events", [{
            "ts": ts,
            "trade_id":    trade.get("trade_id", ""),
            "symbol":      trade.get("symbol", "UNKNOWN"),
            "strategy":    trade.get("strategy", "unknown"),
            "market":      trade.get("market", "FUTURES"),
            "direction":   trade.get("direction", 0),
            "entry_price": trade.get("entry_price", 0),
            "exit_price":  trade.get("exit_price", 0),
            "size_usd":    trade.get("size_usd", 0),
            "pnl_usd":     trade.get("pnl_usd", 0),
            "fees_usd":    trade.get("fees_usd", 0),
            "bars_held":   trade.get("bars_held", 0),
            "is_live":     trade.get("is_live", False),
        }])

    def write_signal(self, symbol: str, signals: dict, ts_val=None) -> bool:
        ts = _to_dt(ts_val) or _now_dt()
        row: dict[str, Any] = {"ts": ts, "symbol": str(symbol)}
        for k, v in signals.items():
            row[k] = v
        return self.insert_rows("model_signals", [row])

    def write_training_event(self, model_name: str, run_id: str, metrics: dict) -> bool:
        ts = _now_dt()
        return self.insert_rows("training_telemetry", [{
            "ts": ts,
            "model":    str(model_name),
            "run_id":   str(run_id),
            "hardware": metrics.get("hardware", "unknown"),
            "epoch":    metrics.get("epoch", 0),
            "train_loss":    metrics.get("train_loss", 0),
            "val_loss":      metrics.get("val_loss", 0),
            "accuracy":      metrics.get("accuracy", 0),
            "sharpe":        metrics.get("sharpe", 0),
            "learning_rate": metrics.get("learning_rate", 0),
            "batch_size":    metrics.get("batch_size", 0),
            "seq_len":       metrics.get("seq_len", 0),
            "n_samples":     metrics.get("n_samples", 0),
        }])

    def write_strategy_stats(self, stats: list[dict]) -> bool:
        ts = _now_dt()
        rows = [{"ts": ts, **s} for s in stats]
        return self.insert_rows("strategy_performance", rows) if rows else True

    def write_news_sentiment(self, item: dict) -> bool:
        ts = _to_dt(item.get("timestamp")) or _now_dt()
        return self.insert_rows("news_sentiment", [{
            "ts": ts,
            "source":    item.get("source", "unknown"),
            "sentiment": item.get("sentiment_label", "neutral"),
            "coins":     str(item.get("coins_mentioned", ""))[:50],
            "score":     item.get("sentiment_score", 0),
            "headline":  str(item.get("headline", ""))[:500],
            "url":       str(item.get("url", ""))[:500],
        }])

    # ── Analytics helpers (QuestDBClient parity) ──────────────────────────

    def get_latest_candle_ts(self, symbol: str, timeframe: str) -> datetime | None:
        # Symbols are stored in safe-partition form ('BTC_USDT'). Accept
        # either form from callers.
        rows = self.query(
            f"SELECT MAX(ts) AS max_ts FROM market_data "
            f"WHERE symbol = '{_safe_partition(symbol)}' "
            f"AND timeframe = '{_safe_partition(timeframe)}'"
        )
        if rows and rows[0].get("max_ts"):
            v = rows[0]["max_ts"]
            if isinstance(v, datetime):
                return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
            return _to_dt(v)
        return None

    def get_strategy_history(self, strategy: str, days: int = 7) -> list[dict]:
        return self.query(
            f"SELECT ts, balance, total_pnl, win_rate, n_trades "
            f"FROM strategy_performance "
            f"WHERE strategy = '{_safe_sql(strategy)}' "
            f"AND ts >= now() - INTERVAL {int(days)} DAY "
            f"ORDER BY ts"
        )

    def get_training_history(self, model_name: str, last_n_runs: int = 5) -> list[dict]:
        return self.query(
            f"SELECT * FROM training_telemetry "
            f"WHERE model = '{_safe_sql(model_name)}' "
            f"ORDER BY ts DESC LIMIT {int(last_n_runs) * 200}"
        )

    # ── Training-run / WF / testnet writes ────────────────────────────────

    def write_training_run(self, run: dict) -> bool:
        ts       = _to_dt(run.get("end_ts")) or _now_dt()
        start_dt = _to_dt(run.get("start_ts")) or ts
        end_dt   = _to_dt(run.get("end_ts"))   or ts
        return self.insert_rows("training_runs", [{
            "ts": ts,
            "run_id":           run.get("run_id", ""),
            "model_name":       run.get("model_name", "unknown"),
            "strategy":         run.get("strategy", "unknown"),
            "symbol":           run.get("symbol", "ALL"),
            "timeframe":        run.get("timeframe", "1h"),
            "trigger":          run.get("trigger", "scheduled"),
            "start_ts":         start_dt,
            "end_ts":           end_dt,
            "duration_secs":    run.get("duration_secs", 0),
            "train_rows":       run.get("train_rows", 0),
            "val_rows":         run.get("val_rows", 0),
            "n_wf_folds":       run.get("n_wf_folds", 0),
            "best_epoch":       run.get("best_epoch", 0),
            "final_train_loss": run.get("final_train_loss", 0),
            "final_val_loss":   run.get("final_val_loss", 0),
            "early_stopped":    run.get("early_stopped", False),
            "oos_sharpe":       run.get("oos_sharpe", 0),
            "oos_win_rate":     run.get("oos_win_rate", 0),
            "oos_max_drawdown": run.get("oos_max_drawdown", 0),
            "n_oos_trades":     run.get("n_oos_trades", 0),
            "hyperparams_json": str(run.get("hyperparams_json", "{}"))[:2000],
            "feature_list_json": str(run.get("feature_list_json", "[]"))[:2000],
            "notes":            str(run.get("notes", ""))[:500],
        }])

    def write_wf_fold(self, run_id: str, model_name: str, fold: dict) -> bool:
        ts = _now_dt()
        return self.insert_rows("model_wf_folds", [{
            "ts": ts,
            "run_id":       run_id,
            "model_name":   model_name,
            "fold_index":   fold.get("fold_index", 0),
            "train_start":  _to_dt(fold.get("train_start")) or ts,
            "train_end":    _to_dt(fold.get("train_end"))   or ts,
            "test_start":   _to_dt(fold.get("test_start"))  or ts,
            "test_end":     _to_dt(fold.get("test_end"))    or ts,
            "train_rows":   fold.get("train_rows", 0),
            "test_rows":    fold.get("test_rows", 0),
            "oos_sharpe":   fold.get("oos_sharpe", 0),
            "oos_pnl":      fold.get("oos_pnl", 0),
            "oos_win_rate": fold.get("oos_win_rate", 0),
            "oos_max_dd":   fold.get("oos_max_dd", 0),
            "n_trades":     fold.get("n_trades", 0),
        }])

    def write_testnet_trade(self, trade: dict) -> bool:
        ts = _to_dt(trade.get("exit_ts") or trade.get("ts")) or _now_dt()
        return self.insert_rows("testnet_trades", [{
            "ts": ts,
            "trade_id":    trade.get("trade_id", ""),
            "symbol":      trade.get("symbol", "UNKNOWN"),
            "strategy":    trade.get("strategy", "unknown"),
            "model":       trade.get("model", "unknown"),
            "exit_reason": trade.get("exit_reason", "unknown"),
            "direction":   trade.get("direction", 0),
            "is_live":     trade.get("is_live", False),
            "entry_ts":    _to_dt(trade.get("entry_ts")) or ts,
            "exit_ts":     _to_dt(trade.get("exit_ts"))  or ts,
            "entry_price": trade.get("entry_price", 0),
            "exit_price":  trade.get("exit_price", 0),
            "size_usd":    trade.get("size_usd", 0),
            "pnl_usd":     trade.get("pnl_usd", 0),
            "fees_usd":    trade.get("fees_usd", 0),
            "funding_pnl": trade.get("funding_pnl", 0),
            "net_pnl":     trade.get("net_pnl", 0),
            "bars_held":   trade.get("bars_held", 0),
            "meta_label":  trade.get("meta_label", 0),
            "regime":      trade.get("regime", 0),
            "garch_vol_at_entry": trade.get("garch_vol_at_entry", 0),
            "stop_loss":          trade.get("stop_loss", 0),
            "take_profit":        trade.get("take_profit", 0),
            "meta_prob":          trade.get("meta_prob", 0),
            "signal_strength":    trade.get("signal_strength", 0),
        }])

    def write_testnet_session_stats(self, stats: dict) -> bool:
        ts = _now_dt()
        return self.insert_rows("testnet_session_stats", [{
            "ts": ts,
            "session_id":        stats.get("session_id", ""),
            "strategy":          stats.get("strategy", "ALL"),
            "symbol":            stats.get("symbol", "ALL"),
            "balance":           stats.get("balance", 0),
            "total_pnl":         stats.get("total_pnl", 0),
            "unrealized_pnl":    stats.get("unrealized_pnl", 0),
            "n_open_trades":     stats.get("n_open_trades", 0),
            "n_closed_trades":   stats.get("n_closed_trades", 0),
            "win_rate":          stats.get("win_rate", 0),
            "sharpe":            stats.get("sharpe", 0),
            "max_drawdown":      stats.get("max_drawdown", 0),
            "funding_collected": stats.get("funding_collected", 0),
        }])


# ── ILP helpers (compatibility shim) ─────────────────────────────────────────

def _parse_ilp_line(line: str) -> tuple[str, dict] | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split(" ")
    if len(parts) < 2:
        return None
    head, fields_part = parts[0], parts[1]
    ts_str = parts[2] if len(parts) > 2 else None
    head_parts = head.split(",")
    table = head_parts[0]
    row: dict[str, Any] = {}
    for tag in head_parts[1:]:
        if "=" in tag:
            k, v = tag.split("=", 1)
            row[k] = v
    for fld in fields_part.split(","):
        if "=" not in fld:
            continue
        k, v = fld.split("=", 1)
        if v.endswith("i"):
            try: row[k] = int(v[:-1])
            except ValueError: row[k] = v[:-1]
        elif v in ("true", "false"):
            row[k] = (v == "true")
        elif v.startswith('"') and v.endswith('"'):
            row[k] = v[1:-1]
        else:
            try: row[k] = float(v)
            except ValueError: row[k] = v
    if ts_str:
        try:
            row["ts"] = datetime.fromtimestamp(int(ts_str) / 1e9, tz=timezone.utc)
        except Exception:
            row["ts"] = _now_dt()
    else:
        row["ts"] = _now_dt()
    return table, row


def _safe_sql(v: str) -> str:
    return str(v).replace("'", "").replace(";", "").replace("--", "")


# CLI: smoke-test the store from the command line.
if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO, format="%(levelname)s %(message)s")
    c = ParquetClient()
    print(f"ParquetClient base: {c.base_dir}")
    print(f"Available: {c.is_available()}")
    print(f"Tables: {len(_TABLES)}")
