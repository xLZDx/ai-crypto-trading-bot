"""
SimulatorDataStore — DuckDB-backed persistence for the live-feed simulator.

Stores:
  sim_scenarios       — metadata for each replay session
  sim_paper_trades    — paper trades executed during simulation
  sim_training_events — per-cycle training metrics (loss, accuracy, sharpe)
  sim_model_metrics   — latest metrics per model (upserted)

Thread-safety: all writes serialised through a Python threading.Lock();
DuckDB's default single-writer mode is preserved.
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "simulator.duckdb"

_DDL = """
CREATE TABLE IF NOT EXISTS sim_scenarios (
    id          VARCHAR PRIMARY KEY,
    scenario_type VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    timeframe   VARCHAR NOT NULL,
    start_ts    TIMESTAMP,
    end_ts      TIMESTAMP,
    bars_replayed BIGINT DEFAULT 0,
    speed_mult  DOUBLE DEFAULT 1.0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sim_paper_trades (
    id          VARCHAR PRIMARY KEY,
    scenario_id VARCHAR,
    symbol      VARCHAR,
    direction   INTEGER,
    entry_price DOUBLE,
    exit_price  DOUBLE,
    size_usd    DOUBLE,
    pnl_usd     DOUBLE,
    entry_ts    TIMESTAMP,
    exit_ts     TIMESTAMP,
    strategy    VARCHAR,
    model_ver   VARCHAR
);

CREATE TABLE IF NOT EXISTS sim_training_events (
    id          VARCHAR PRIMARY KEY,
    model_name  VARCHAR NOT NULL,
    scenario_id VARCHAR,
    bars_trained BIGINT,
    train_loss  DOUBLE,
    val_loss    DOUBLE,
    accuracy    DOUBLE,
    sharpe      DOUBLE,
    trained_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sim_model_metrics (
    model_name      VARCHAR PRIMARY KEY,
    total_bars      BIGINT DEFAULT 0,
    last_accuracy   DOUBLE,
    last_loss       DOUBLE,
    last_sharpe     DOUBLE,
    last_trained_at TIMESTAMP,
    cycles          INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sim_pattern_db (
    id              VARCHAR PRIMARY KEY,
    model_name      VARCHAR NOT NULL,
    scenario_type   VARCHAR NOT NULL,
    pattern_label   VARCHAR NOT NULL,
    occurrences     INTEGER DEFAULT 1,
    wins            INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    avg_pnl         DOUBLE DEFAULT 0.0,
    last_seen       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class SimulatorDataStore:
    """Thread-safe DuckDB store for simulator state and training results."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ── internals ─────────────────────────────────────────────────────────────

    def _connect(self):
        import duckdb
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._lock:
            try:
                con = self._connect()
                for stmt in _DDL.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        con.execute(stmt)
                con.commit()
                con.close()
            except Exception as exc:
                logger.error("[SimStore] Schema init failed: %s", exc)

    # ── scenarios ─────────────────────────────────────────────────────────────

    def start_scenario(
        self,
        scenario_type: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        speed: float = 1.0,
    ) -> str:
        sid = str(uuid.uuid4())
        with self._lock:
            con = self._connect()
            con.execute(
                """INSERT INTO sim_scenarios
                   (id, scenario_type, symbol, timeframe, start_ts, end_ts, speed_mult)
                   VALUES (?,?,?,?,?,?,?)""",
                [sid, scenario_type, symbol, timeframe, start_ts, end_ts, speed],
            )
            con.commit()
            con.close()
        return sid

    def update_scenario_bars(self, scenario_id: str, bars: int) -> None:
        with self._lock:
            con = self._connect()
            con.execute(
                "UPDATE sim_scenarios SET bars_replayed = ? WHERE id = ?",
                [bars, scenario_id],
            )
            con.commit()
            con.close()

    # ── paper trades ──────────────────────────────────────────────────────────

    def record_paper_trade(
        self,
        scenario_id: str,
        symbol: str,
        direction: int,
        entry_price: float,
        exit_price: float,
        size_usd: float,
        entry_ts: datetime,
        exit_ts: datetime,
        strategy: str,
        model_ver: str = "unknown",
    ) -> None:
        if entry_price <= 0:
            return
        pnl = (exit_price - entry_price) / entry_price * direction * size_usd
        with self._lock:
            con = self._connect()
            con.execute(
                """INSERT INTO sim_paper_trades
                   (id, scenario_id, symbol, direction, entry_price, exit_price,
                    size_usd, pnl_usd, entry_ts, exit_ts, strategy, model_ver)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    str(uuid.uuid4()), scenario_id, symbol, direction,
                    entry_price, exit_price, size_usd, pnl,
                    entry_ts, exit_ts, strategy, model_ver,
                ],
            )
            con.commit()
            con.close()

    # ── training events ───────────────────────────────────────────────────────

    def record_training_event(
        self,
        model_name: str,
        scenario_id: str,
        bars_trained: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
        sharpe: float = 0.0,
    ) -> None:
        with self._lock:
            con = self._connect()
            con.execute(
                """INSERT INTO sim_training_events
                   (id, model_name, scenario_id, bars_trained,
                    train_loss, val_loss, accuracy, sharpe)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [
                    str(uuid.uuid4()), model_name, scenario_id,
                    bars_trained, train_loss, val_loss, accuracy, sharpe,
                ],
            )
            # Upsert aggregate metrics
            existing = con.execute(
                "SELECT total_bars, cycles FROM sim_model_metrics WHERE model_name = ?",
                [model_name],
            ).fetchone()
            if existing:
                con.execute(
                    """UPDATE sim_model_metrics SET
                       total_bars = total_bars + ?,
                       last_accuracy = ?, last_loss = ?, last_sharpe = ?,
                       last_trained_at = CURRENT_TIMESTAMP,
                       cycles = cycles + 1
                       WHERE model_name = ?""",
                    [bars_trained, accuracy, train_loss, sharpe, model_name],
                )
            else:
                con.execute(
                    """INSERT INTO sim_model_metrics
                       (model_name, total_bars, last_accuracy, last_loss,
                        last_sharpe, last_trained_at, cycles)
                       VALUES (?,?,?,?,?,CURRENT_TIMESTAMP,1)""",
                    [model_name, bars_trained, accuracy, train_loss, sharpe],
                )
            con.commit()
            con.close()

    # ── query helpers ─────────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            try:
                con = self._connect()
                sc = con.execute(
                    "SELECT COUNT(*), COALESCE(MAX(bars_replayed),0), MAX(created_at) FROM sim_scenarios"
                ).fetchone()
                tr = con.execute(
                    "SELECT COUNT(*), COALESCE(SUM(pnl_usd),0), COALESCE(AVG(pnl_usd),0) FROM sim_paper_trades"
                ).fetchone()
                mm = con.execute(
                    """SELECT model_name, last_accuracy, last_loss, last_sharpe,
                              total_bars, cycles, last_trained_at
                       FROM sim_model_metrics ORDER BY model_name"""
                ).fetchall()
                con.close()
                return {
                    "total_scenarios": sc[0] or 0,
                    "max_bars_replayed": sc[1] or 0,
                    "last_scenario_at": sc[2].isoformat() if sc[2] else None,
                    "total_paper_trades": tr[0] or 0,
                    "total_pnl_usd": round(float(tr[1] or 0), 2),
                    "avg_pnl_usd": round(float(tr[2] or 0), 4),
                    "models": [
                        {
                            "name": r[0],
                            "accuracy": round(float(r[1] or 0), 4),
                            "loss": round(float(r[2] or 0), 6),
                            "sharpe": round(float(r[3] or 0), 3),
                            "bars_trained": r[4] or 0,
                            "cycles": r[5] or 0,
                            "last_trained": r[6].isoformat() if r[6] else None,
                        }
                        for r in mm
                    ],
                }
            except Exception as exc:
                logger.error("[SimStore] get_summary error: %s", exc)
                return {}

    def get_recent_training_events(
        self, model_name: str | None = None, limit: int = 100
    ) -> list[dict]:
        with self._lock:
            try:
                con = self._connect()
                if model_name:
                    rows = con.execute(
                        """SELECT model_name, bars_trained, train_loss, val_loss,
                                  accuracy, sharpe, trained_at
                           FROM sim_training_events WHERE model_name = ?
                           ORDER BY trained_at DESC LIMIT ?""",
                        [model_name, limit],
                    ).fetchall()
                else:
                    rows = con.execute(
                        """SELECT model_name, bars_trained, train_loss, val_loss,
                                  accuracy, sharpe, trained_at
                           FROM sim_training_events ORDER BY trained_at DESC LIMIT ?""",
                        [limit],
                    ).fetchall()
                con.close()
                return [
                    {
                        "model": r[0], "bars": r[1],
                        "train_loss": r[2], "val_loss": r[3],
                        "accuracy": r[4], "sharpe": r[5],
                        "at": r[6].isoformat() if r[6] else None,
                    }
                    for r in rows
                ]
            except Exception as exc:
                logger.error("[SimStore] get_recent_training_events error: %s", exc)
                return []

    def record_pattern(
        self,
        model_name: str,
        scenario_type: str,
        pattern_label: str,
        won: bool,
        pnl: float,
    ) -> None:
        """Upsert a pattern observation into the pattern DB."""
        with self._lock:
            try:
                con = self._connect()
                key = f"{model_name}|{scenario_type}|{pattern_label}"
                row = con.execute(
                    "SELECT id, occurrences, wins, losses, avg_pnl FROM sim_pattern_db WHERE id = ?",
                    [key],
                ).fetchone()
                if row:
                    new_occ = row[1] + 1
                    new_wins = row[2] + (1 if won else 0)
                    new_loss = row[3] + (0 if won else 1)
                    new_avg  = (row[4] * row[1] + pnl) / new_occ
                    con.execute(
                        """UPDATE sim_pattern_db SET occurrences=?, wins=?, losses=?,
                           avg_pnl=?, last_seen=CURRENT_TIMESTAMP WHERE id=?""",
                        [new_occ, new_wins, new_loss, new_avg, key],
                    )
                else:
                    con.execute(
                        """INSERT INTO sim_pattern_db
                           (id, model_name, scenario_type, pattern_label,
                            occurrences, wins, losses, avg_pnl)
                           VALUES (?,?,?,?,1,?,?,?)""",
                        [key, model_name, scenario_type, pattern_label,
                         1 if won else 0, 0 if won else 1, pnl],
                    )
                con.commit()
                con.close()
            except Exception as exc:
                logger.error("[SimStore] record_pattern error: %s", exc)

    def get_pattern_db(self, model_name: str | None = None, limit: int = 100) -> list[dict]:
        """Return pattern observations sorted by occurrences descending."""
        with self._lock:
            try:
                con = self._connect()
                if model_name:
                    rows = con.execute(
                        """SELECT model_name, scenario_type, pattern_label,
                                  occurrences, wins, losses, avg_pnl, last_seen
                           FROM sim_pattern_db WHERE model_name = ?
                           ORDER BY occurrences DESC LIMIT ?""",
                        [model_name, limit],
                    ).fetchall()
                else:
                    rows = con.execute(
                        """SELECT model_name, scenario_type, pattern_label,
                                  occurrences, wins, losses, avg_pnl, last_seen
                           FROM sim_pattern_db ORDER BY occurrences DESC LIMIT ?""",
                        [limit],
                    ).fetchall()
                con.close()
                return [
                    {
                        "model": r[0], "scenario": r[1], "pattern": r[2],
                        "n": r[3], "wins": r[4], "losses": r[5],
                        "avg_pnl": round(float(r[6] or 0), 4),
                        "win_rate": round(r[4] / r[3] * 100, 1) if r[3] else 0,
                        "last_seen": r[7].isoformat() if r[7] else None,
                    }
                    for r in rows
                ]
            except Exception as exc:
                logger.error("[SimStore] get_pattern_db error: %s", exc)
                return []

    def get_paper_pnl_series(self, limit: int = 500) -> list[dict]:
        """Cumulative P&L series for charting."""
        with self._lock:
            try:
                con = self._connect()
                rows = con.execute(
                    """SELECT symbol, pnl_usd, exit_ts, strategy
                       FROM sim_paper_trades ORDER BY exit_ts ASC LIMIT ?""",
                    [limit],
                ).fetchall()
                con.close()
                cumulative = 0.0
                result = []
                for r in rows:
                    cumulative += float(r[1] or 0)
                    result.append({
                        "symbol": r[0], "pnl": round(float(r[1] or 0), 4),
                        "cumulative": round(cumulative, 4),
                        "ts": r[2].isoformat() if r[2] else None,
                        "strategy": r[3],
                    })
                return result
            except Exception as exc:
                logger.error("[SimStore] get_paper_pnl_series error: %s", exc)
                return []
