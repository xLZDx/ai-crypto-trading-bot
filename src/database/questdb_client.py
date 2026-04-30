"""
QuestDB client — REST queries + ILP writes.

Two write modes:
  • ILP (InfluxDB Line Protocol, port 9009) — fastest, use for bulk/live data
  • REST (HTTP /exec, port 9000)           — SQL DDL + ad-hoc queries

Usage:
    from src.database.questdb_client import get_client
    db = get_client()
    if db.is_available():
        df = db.query("SELECT * FROM market_data LIMIT 10")
        db.write_rows("market_data", rows)
"""
from __future__ import annotations

import json
import logging
import socket
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "localhost"
_DEFAULT_HTTP_PORT = 9000
_DEFAULT_ILP_PORT  = 9009

# Module-level singleton
_client_instance: "QuestDBClient | None" = None
_client_lock = threading.Lock()


def get_client() -> "QuestDBClient":
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                _client_instance = QuestDBClient()
    return _client_instance


class QuestDBClient:
    """
    Thread-safe QuestDB client.
    Falls back gracefully when QuestDB is unavailable (logs warning, returns empty data).
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        http_port: int = _DEFAULT_HTTP_PORT,
        ilp_port: int = _DEFAULT_ILP_PORT,
        timeout: float = 10.0,
    ) -> None:
        self.host      = host
        self.http_port = http_port
        self.ilp_port  = ilp_port
        self.timeout   = timeout
        self._base_url = f"http://{host}:{http_port}"
        self._available: bool | None = None   # None = not yet checked
        self._last_check: float = 0.0
        self._check_interval = 30.0           # re-probe every 30 s

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and self._available is not None and (now - self._last_check) < self._check_interval:
            return self._available
        try:
            r = requests.get(f"{self._base_url}/health", timeout=3)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        self._last_check = now
        if not self._available:
            logger.debug("QuestDB not available at %s:%s — run: docker-compose up -d questdb",
                         self.host, self.http_port)
        return self._available

    # ── SQL query (REST) ──────────────────────────────────────────────────────

    def query(self, sql: str) -> list[dict]:
        """Execute SQL, return list of row dicts. Returns [] if DB unavailable."""
        if not self.is_available():
            return []
        try:
            r = requests.get(
                f"{self._base_url}/exec",
                params={"query": sql, "limit": "0,50000"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                logger.warning("QuestDB query error: %s | SQL: %s", data["error"], sql[:200])
                return []
            cols = [c["name"] for c in data.get("columns", [])]
            return [dict(zip(cols, row)) for row in data.get("dataset", [])]
        except Exception as exc:
            logger.warning("QuestDB query failed: %s", exc)
            return []

    def query_df(self, sql: str):
        """Execute SQL, return pandas DataFrame. Returns empty DF if unavailable."""
        import pandas as pd
        rows = self.query(sql)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def exec_ddl(self, sql: str) -> bool:
        """Execute DDL (CREATE TABLE, etc.). Returns True on success."""
        if not self.is_available():
            return False
        try:
            r = requests.get(
                f"{self._base_url}/exec",
                params={"query": sql},
                timeout=self.timeout,
            )
            data = r.json()
            if "error" in data:
                # Table already exists is not an error for us
                if "already exists" in str(data.get("error", "")):
                    return True
                logger.warning("QuestDB DDL error: %s | SQL: %s", data["error"], sql[:200])
                return False
            return True
        except Exception as exc:
            logger.warning("QuestDB DDL failed: %s", exc)
            return False

    # ── ILP write (fast bulk) ─────────────────────────────────────────────────

    def write_ilp(self, lines: list[str]) -> bool:
        """
        Write ILP (InfluxDB Line Protocol) lines to QuestDB via TCP socket.
        Each line format: table_name,tag1=v1 field1=v1,field2=v2 nanosec_ts
        Returns True on success.
        """
        if not lines:
            return True
        if not self.is_available():
            return False
        payload = "\n".join(lines) + "\n"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.host, self.ilp_port))
                s.sendall(payload.encode())
            return True
        except Exception as exc:
            logger.warning("QuestDB ILP write failed (%d lines): %s", len(lines), exc)
            return False

    # ── High-level write helpers ──────────────────────────────────────────────

    def write_market_candle(self, symbol: str, timeframe: str, bar: dict) -> bool:
        """Write one OHLCV bar."""
        ts = _to_ns(bar.get("timestamp") or bar.get("ts"))
        if ts is None:
            return False
        line = (
            f"market_data,symbol={_tag(symbol)},timeframe={timeframe} "
            f"open={float(bar.get('open', 0))},"
            f"high={float(bar.get('high', 0))},"
            f"low={float(bar.get('low', 0))},"
            f"close={float(bar.get('close', 0))},"
            f"volume={float(bar.get('volume', 0))},"
            f"funding_rate={float(bar.get('funding_rate') or 0)} "
            f"{ts}"
        )
        return self.write_ilp([line])

    def write_market_candles_bulk(self, symbol: str, timeframe: str, bars: list[dict], batch: int = 5000) -> int:
        """Bulk-write OHLCV bars. Returns number of rows written."""
        lines, written = [], 0
        for bar in bars:
            ts = _to_ns(bar.get("timestamp") or bar.get("ts"))
            if ts is None:
                continue
            lines.append(
                f"market_data,symbol={_tag(symbol)},timeframe={timeframe} "
                f"open={float(bar.get('open', 0))},"
                f"high={float(bar.get('high', 0))},"
                f"low={float(bar.get('low', 0))},"
                f"close={float(bar.get('close', 0))},"
                f"volume={float(bar.get('volume', 0))},"
                f"funding_rate={float(bar.get('funding_rate') or 0)} "
                f"{ts}"
            )
            if len(lines) >= batch:
                if self.write_ilp(lines):
                    written += len(lines)
                lines = []
        if lines:
            if self.write_ilp(lines):
                written += len(lines)
        return written

    def write_trade(self, trade: dict) -> bool:
        """Write one bot trade event."""
        ts = _to_ns(trade.get("timestamp") or trade.get("ts")) or _now_ns()
        sym = _tag(trade.get("symbol", "UNKNOWN"))
        strategy = _tag(trade.get("strategy", "unknown"))
        market = _tag(trade.get("market", "FUTURES"))
        line = (
            f"trade_events,symbol={sym},strategy={strategy},market={market} "
            f"direction={int(trade.get('direction', 0))}i,"
            f"entry_price={float(trade.get('entry_price', 0))},"
            f"exit_price={float(trade.get('exit_price', 0))},"
            f"size_usd={float(trade.get('size_usd', 0))},"
            f"pnl_usd={float(trade.get('pnl_usd', 0))},"
            f"fees_usd={float(trade.get('fees_usd', 0))} "
            f"{ts}"
        )
        return self.write_ilp([line])

    def write_signal(self, symbol: str, signals: dict, ts_val=None) -> bool:
        """Write one bar's signal snapshot."""
        ts = _to_ns(ts_val) or _now_ns()
        sym = _tag(symbol)
        fields = []
        for k, v in signals.items():
            if v is None:
                continue
            try:
                fields.append(f"{k}={float(v)}")
            except (TypeError, ValueError):
                pass
        if not fields:
            return False
        line = f"model_signals,symbol={sym} {','.join(fields)} {ts}"
        return self.write_ilp([line])

    def write_training_event(self, model_name: str, run_id: str, metrics: dict) -> bool:
        """Write one training epoch metrics row."""
        ts = _now_ns()
        mn = _tag(model_name)
        rid = _tag(run_id)
        hardware = _tag(metrics.get("hardware", "unknown"))
        fields = [
            f"epoch={int(metrics.get('epoch', 0))}i",
            f"train_loss={float(metrics.get('train_loss', 0))}",
            f"val_loss={float(metrics.get('val_loss', 0))}",
            f"accuracy={float(metrics.get('accuracy', 0))}",
            f"sharpe={float(metrics.get('sharpe', 0))}",
            f"learning_rate={float(metrics.get('learning_rate', 0))}",
        ]
        line = (
            f"training_telemetry,model={mn},run_id={rid},hardware={hardware} "
            f"{','.join(fields)} {ts}"
        )
        return self.write_ilp([line])

    def write_strategy_stats(self, stats: list[dict]) -> bool:
        """Write periodic strategy performance snapshot."""
        ts = _now_ns()
        lines = []
        for s in stats:
            strategy = _tag(s.get("strategy", "unknown"))
            symbol   = _tag(s.get("symbol", "ALL"))
            lines.append(
                f"strategy_performance,strategy={strategy},symbol={symbol} "
                f"balance={float(s.get('balance', 0))},"
                f"total_pnl={float(s.get('total_pnl', 0))},"
                f"pnl_pct={float(s.get('pnl_pct', 0))},"
                f"win_rate={float(s.get('win_rate', 0))},"
                f"n_trades={int(s.get('n_trades', 0))}i,"
                f"n_wins={int(s.get('n_wins', 0))}i "
                f"{ts}"
            )
        return self.write_ilp(lines) if lines else True

    def write_news_sentiment(self, item: dict) -> bool:
        """Write one news/sentiment item."""
        ts = _to_ns(item.get("timestamp")) or _now_ns()
        source = _tag(item.get("source", "unknown"))
        label  = _tag(item.get("sentiment_label", "neutral"))
        score  = float(item.get("sentiment_score", 0))
        headline = str(item.get("headline", ""))[:200].replace('"', '\\"').replace('\n', ' ')
        coins    = _tag(str(item.get("coins_mentioned", ""))[:50])
        line = (
            f'news_sentiment,source={source},sentiment={label},coins={coins} '
            f'score={score},headline="{headline}" '
            f'{ts}'
        )
        return self.write_ilp([line])

    # ── Analytics helpers ─────────────────────────────────────────────────────

    def get_latest_candle_ts(self, symbol: str, timeframe: str) -> datetime | None:
        """Return the latest stored candle timestamp for a symbol/timeframe."""
        rows = self.query(
            f"SELECT MAX(ts) as max_ts FROM market_data "
            f"WHERE symbol='{symbol}' AND timeframe='{timeframe}'"
        )
        if rows and rows[0].get("max_ts"):
            try:
                return datetime.fromisoformat(str(rows[0]["max_ts"]).replace("Z", "+00:00"))
            except Exception:
                pass
        return None

    def get_strategy_history(self, strategy: str, days: int = 7) -> list[dict]:
        """Return strategy PNL history for the last N days."""
        return self.query(
            f"SELECT ts, balance, total_pnl, win_rate, n_trades "
            f"FROM strategy_performance "
            f"WHERE strategy='{strategy}' "
            f"AND ts > dateadd('d', -{days}, now()) "
            f"ORDER BY ts"
        )

    def get_training_history(self, model_name: str, last_n_runs: int = 5) -> list[dict]:
        """Return training telemetry for last N runs of a model."""
        return self.query(
            f"SELECT * FROM training_telemetry "
            f"WHERE model='{model_name}' "
            f"ORDER BY ts DESC "
            f"LIMIT {last_n_runs * 200}"
        )

    # ── Analytics: training runs ──────────────────────────────────────────────

    def write_training_run(self, run: dict) -> bool:
        """Write one training run summary to training_runs."""
        ts       = _to_ns(run.get("end_ts")) or _now_ns()
        start_ns = _to_ns(run.get("start_ts")) or ts
        end_ns   = _to_ns(run.get("end_ts"))   or ts
        model    = _tag(run.get("model_name", "unknown"))
        strategy = _tag(run.get("strategy", "unknown"))
        symbol   = _tag(run.get("symbol", "ALL"))
        tf       = _tag(run.get("timeframe", "1h"))
        trigger  = _tag(run.get("trigger", "scheduled"))
        hp       = str(run.get("hyperparams_json", "{}")).replace('"', '\\"')[:500]
        feats    = str(run.get("feature_list_json", "[]")).replace('"', '\\"')[:500]
        notes    = str(run.get("notes", "")).replace('"', '\\"')[:200]
        es       = "true" if run.get("early_stopped") else "false"
        line = (
            f"training_runs,model_name={model},strategy={strategy},"
            f"symbol={symbol},timeframe={tf},trigger={trigger} "
            f'run_id="{run.get("run_id", "")}",'
            f"start_ts={start_ns}i,"
            f"end_ts={end_ns}i,"
            f"duration_secs={float(run.get('duration_secs', 0))},"
            f"train_rows={int(run.get('train_rows', 0))}i,"
            f"val_rows={int(run.get('val_rows', 0))}i,"
            f"n_wf_folds={int(run.get('n_wf_folds', 0))}i,"
            f"best_epoch={int(run.get('best_epoch', 0))}i,"
            f"final_train_loss={float(run.get('final_train_loss', 0))},"
            f"final_val_loss={float(run.get('final_val_loss', 0))},"
            f"early_stopped={es},"
            f"oos_sharpe={float(run.get('oos_sharpe', 0))},"
            f"oos_win_rate={float(run.get('oos_win_rate', 0))},"
            f"oos_max_drawdown={float(run.get('oos_max_drawdown', 0))},"
            f"n_oos_trades={int(run.get('n_oos_trades', 0))}i,"
            f'hyperparams_json="{hp}",'
            f'feature_list_json="{feats}",'
            f'notes="{notes}" '
            f"{ts}"
        )
        return self.write_ilp([line])

    def write_wf_fold(self, run_id: str, model_name: str, fold: dict) -> bool:
        """Write one Walk-Forward fold result to model_wf_folds."""
        ts          = _now_ns()
        train_start = _to_ns(fold.get("train_start")) or ts
        train_end   = _to_ns(fold.get("train_end"))   or ts
        test_start  = _to_ns(fold.get("test_start"))  or ts
        test_end    = _to_ns(fold.get("test_end"))     or ts
        model       = _tag(model_name)
        line = (
            f"model_wf_folds,model_name={model} "
            f'run_id="{run_id}",'
            f"fold_index={int(fold.get('fold_index', 0))}i,"
            f"train_start={train_start}i,"
            f"train_end={train_end}i,"
            f"test_start={test_start}i,"
            f"test_end={test_end}i,"
            f"train_rows={int(fold.get('train_rows', 0))}i,"
            f"test_rows={int(fold.get('test_rows', 0))}i,"
            f"oos_sharpe={float(fold.get('oos_sharpe', 0))},"
            f"oos_pnl={float(fold.get('oos_pnl', 0))},"
            f"oos_win_rate={float(fold.get('oos_win_rate', 0))},"
            f"oos_max_dd={float(fold.get('oos_max_dd', 0))},"
            f"n_trades={int(fold.get('n_trades', 0))}i "
            f"{ts}"
        )
        return self.write_ilp([line])

    def write_testnet_trade(self, trade: dict) -> bool:
        """Write one testnet/live trade to testnet_trades."""
        ts       = _to_ns(trade.get("exit_ts") or trade.get("ts")) or _now_ns()
        entry_ns = _to_ns(trade.get("entry_ts")) or ts
        exit_ns  = _to_ns(trade.get("exit_ts"))  or ts
        sym      = _tag(trade.get("symbol", "UNKNOWN"))
        strategy = _tag(trade.get("strategy", "unknown"))
        model    = _tag(trade.get("model", "unknown"))
        exit_r   = _tag(trade.get("exit_reason", "unknown"))
        is_live  = "true" if trade.get("is_live") else "false"
        tid      = str(trade.get("trade_id", "")).replace('"', '')
        line = (
            f"testnet_trades,symbol={sym},strategy={strategy},"
            f"model={model},exit_reason={exit_r} "
            f'trade_id="{tid}",'
            f"direction={int(trade.get('direction', 0))}i,"
            f"is_live={is_live},"
            f"entry_ts={entry_ns}i,"
            f"exit_ts={exit_ns}i,"
            f"entry_price={float(trade.get('entry_price', 0))},"
            f"exit_price={float(trade.get('exit_price', 0))},"
            f"size_usd={float(trade.get('size_usd', 0))},"
            f"pnl_usd={float(trade.get('pnl_usd', 0))},"
            f"fees_usd={float(trade.get('fees_usd', 0))},"
            f"funding_pnl={float(trade.get('funding_pnl', 0))},"
            f"net_pnl={float(trade.get('net_pnl', 0))},"
            f"bars_held={int(trade.get('bars_held', 0))}i,"
            f"meta_label={int(trade.get('meta_label', 0))}i,"
            f"regime={int(trade.get('regime', 0))}i,"
            f"garch_vol_at_entry={float(trade.get('garch_vol_at_entry', 0))},"
            f"stop_loss={float(trade.get('stop_loss', 0))},"
            f"take_profit={float(trade.get('take_profit', 0))},"
            f"meta_prob={float(trade.get('meta_prob', 0))},"
            f"signal_strength={float(trade.get('signal_strength', 0))} "
            f"{ts}"
        )
        return self.write_ilp([line])

    def write_testnet_session_stats(self, stats: dict) -> bool:
        """Write one hourly testnet session snapshot."""
        ts       = _now_ns()
        strategy = _tag(stats.get("strategy", "ALL"))
        symbol   = _tag(stats.get("symbol", "ALL"))
        sid      = str(stats.get("session_id", "")).replace('"', '')
        line = (
            f"testnet_session_stats,strategy={strategy},symbol={symbol} "
            f'session_id="{sid}",'
            f"balance={float(stats.get('balance', 0))},"
            f"total_pnl={float(stats.get('total_pnl', 0))},"
            f"unrealized_pnl={float(stats.get('unrealized_pnl', 0))},"
            f"n_open_trades={int(stats.get('n_open_trades', 0))}i,"
            f"n_closed_trades={int(stats.get('n_closed_trades', 0))}i,"
            f"win_rate={float(stats.get('win_rate', 0))},"
            f"sharpe={float(stats.get('sharpe', 0))},"
            f"max_drawdown={float(stats.get('max_drawdown', 0))},"
            f"funding_collected={float(stats.get('funding_collected', 0))} "
            f"{ts}"
        )
        return self.write_ilp([line])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tag(v: str) -> str:
    """Sanitise ILP tag value (no spaces, commas, equals)."""
    return str(v).replace(" ", "_").replace(",", "_").replace("=", "_").replace("/", "_")


def _now_ns() -> int:
    return int(time.time() * 1e9)


def _to_ns(ts) -> int | None:
    """Convert various timestamp formats to nanoseconds-since-epoch."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # Could be seconds, millis, or nanos
        if ts < 1e12:         # seconds
            return int(ts * 1e9)
        elif ts < 1e15:       # milliseconds
            return int(ts * 1e6)
        else:                  # nanoseconds
            return int(ts)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f+00:00"):
            try:
                dt = datetime.strptime(ts.replace("+00:00", ""), fmt.replace("+00:00", ""))
                return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e9)
            except ValueError:
                continue
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp() * 1e9)
    return None
