"""
SimulatorAgent — orchestrates the market-data replay and paper-trading loop.

State machine:
  IDLE → LOADING → RUNNING ⇄ PAUSED → COMPLETE

Publishes on AgentBus topic 'sim_candle':
  {"symbol", "timeframe", "timestamp", "open", "high", "low", "close",
   "volume", "quote_volume", "trades_count", "taker_buy_base",
   "taker_buy_quote", "funding_rate", "source": "simulator"}

Also publishes 'sim_signal' when the meta-labeler approves a paper trade,
and writes simulator state to data/simulator_state.json for the dashboard.

Config (set via REST POST /api/simulator/config):
  symbol      — symbol to replay (default "BTC_USDT")
  timeframe   — bar timeframe (default "1m")
  speed       — replay speed multiplier (default 1000.0 for fast training)
  scenario    — scenario type or "AUTO" (default "AUTO")
  start_date  — ISO date string (optional)
  end_date    — ISO date string (optional)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.engine.agents.agent_bus import BaseAgent, _write_agent_status

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_FILE   = PROJECT_ROOT / "data" / "simulator_state.json"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

# Simulator FSM states
IDLE      = "IDLE"
LOADING   = "LOADING"
RUNNING   = "RUNNING"
PAUSED    = "PAUSED"
COMPLETE  = "COMPLETE"
ERROR     = "ERROR"


class SimulatorAgent(BaseAgent):
    """
    Manages one replay session at a time; sessions cycle continuously when
    auto_cycle=True so training agents never starve.
    """

    NAME = "SimulatorAgent"

    def __init__(self, bus=None, auto_cycle: bool = True):
        super().__init__(bus=bus, interval_sec=5.0)
        self.auto_cycle = auto_cycle

        # Runtime state
        self._state = IDLE
        self._config: dict[str, Any] = {
            "symbol":    "BTC_USDT",
            "timeframe": "1m",
            "speed":     1000.0,
            "scenario":  "AUTO",
            "start_date": None,
            "end_date":   None,
        }
        self._current_scenario: dict | None = None
        self._current_scenario_id: str | None = None
        self._bars_emitted: int = 0
        self._current_ts: str = ""
        self._bars_per_sec: float = 0.0
        self._error_msg: str = ""

        # Replay control flag (mutable list for passing into generator)
        self._stop_flag: list[bool] = [False]
        self._replay_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Lazy imports (avoid heavy deps at import time)
        self._store = None
        self._scenario_mgr = None

        # Rate tracking (bars/sec)
        self._bar_times: deque[float] = deque(maxlen=200)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        """Called every interval_sec by BaseAgent._loop; ensures replay stays running."""
        self._flush_state()
        with self._lock:
            is_running = self._state == RUNNING
            replay_alive = (
                self._replay_thread is not None
                and self._replay_thread.is_alive()
            )

        if is_running and not replay_alive:
            # Replay thread finished — start next scenario
            if self.auto_cycle:
                self._start_replay_thread()
            else:
                with self._lock:
                    self._state = COMPLETE

    # ── public control API (called from Flask endpoints) ─────────────────────

    def configure(self, cfg: dict) -> None:
        """Update replay configuration. Safe to call while IDLE or between sessions."""
        with self._lock:
            for k, v in cfg.items():
                if k in self._config:
                    self._config[k] = v
        logger.info("[SimulatorAgent] Config updated: %s", cfg)

    def start(self) -> None:
        """Begin replay. Idempotent if already running."""
        with self._lock:
            if self._state in (RUNNING, LOADING):
                return
            self._state = LOADING
            self._stop_flag = [False]

        self._lazy_init()
        self._start_replay_thread()

    def pause(self) -> None:
        """Pause after the current bar finishes."""
        with self._lock:
            if self._state == RUNNING:
                self._state = PAUSED
                self._stop_flag[0] = True

    def resume(self) -> None:
        """Resume from paused state."""
        with self._lock:
            if self._state != PAUSED:
                return
            self._state = LOADING
            self._stop_flag = [False]
        self._start_replay_thread()

    def stop(self) -> None:
        """Stop replay entirely."""
        with self._lock:
            self._stop_flag[0] = True
            self._state = IDLE

    def get_status(self) -> dict:
        with self._lock:
            return {
                "state":       self._state,
                "config":      dict(self._config),
                "scenario":    self._current_scenario,
                "bars_emitted": self._bars_emitted,
                "current_ts":  self._current_ts,
                "bars_per_sec": round(self._bars_per_sec, 1),
                "error":       self._error_msg,
            }

    # ── private ───────────────────────────────────────────────────────────────

    def _lazy_init(self) -> None:
        if self._store is None:
            from src.simulation.data_store import SimulatorDataStore
            self._store = SimulatorDataStore()
        if self._scenario_mgr is None:
            from src.simulation.scenario_manager import ScenarioManager
            self._scenario_mgr = ScenarioManager()

    def _start_replay_thread(self) -> None:
        t = threading.Thread(
            target=self._replay_loop,
            name="SimReplayThread",
            daemon=True,
        )
        with self._lock:
            self._replay_thread = t
            self._state = RUNNING
        t.start()

    def _replay_loop(self) -> None:
        """One full scenario replay; runs in a dedicated thread."""
        try:
            from src.simulation.market_replay import MarketReplay

            with self._lock:
                cfg = dict(self._config)
                stop_flag = self._stop_flag

            # Pick scenario
            if cfg["scenario"] == "AUTO" and self._scenario_mgr:
                scenario = self._scenario_mgr.next_scenario(timeframe=cfg["timeframe"])
            else:
                scenario = {
                    "type":      cfg["scenario"],
                    "symbol":    cfg["symbol"],
                    "timeframe": cfg["timeframe"],
                    "start":     cfg.get("start_date"),
                    "end":       cfg.get("end_date"),
                }

            with self._lock:
                self._current_scenario = scenario
                self._bars_emitted = 0

            symbol    = scenario.get("symbol",    cfg["symbol"])
            timeframe = scenario.get("timeframe", cfg["timeframe"])
            speed     = float(cfg["speed"])
            start_dt  = _parse_dt(scenario.get("start"))
            end_dt    = _parse_dt(scenario.get("end"))

            # Check GZ exists — fall back to BTC if missing
            gz = RAW_DIR / f"{symbol}_{timeframe}.csv.gz"
            if not gz.exists():
                logger.warning("[SimulatorAgent] Missing %s, falling back to BTC_USDT", gz.name)
                symbol = "BTC_USDT"

            # Register scenario in DB
            sid = None
            if self._store:
                sid = self._store.start_scenario(
                    scenario_type=scenario.get("type", "COMPOSITE"),
                    symbol=symbol,
                    timeframe=timeframe,
                    start_ts=start_dt,
                    end_ts=end_dt,
                    speed=speed,
                )
            with self._lock:
                self._current_scenario_id = sid

            replay = MarketReplay(symbol, timeframe, speed=speed)
            bars_since_db_update = 0
            t_window_start = time.monotonic()
            bars_window = 0

            for bar in replay.stream(
                start=start_dt, end=end_dt, stopped_flag=stop_flag
            ):
                if stop_flag[0]:
                    break

                # Publish to AgentBus
                self.publish("sim_candle", bar)

                # Track state
                t_now = time.monotonic()
                self._bar_times.append(t_now)
                elapsed = t_now - t_window_start
                bars_window += 1
                if elapsed >= 2.0:
                    with self._lock:
                        self._bars_per_sec = bars_window / elapsed
                        self._current_ts   = bar["timestamp"]
                        self._bars_emitted = replay.bars_emitted
                    t_window_start = t_now
                    bars_window = 0

                # Periodic DB update (every 5k bars)
                bars_since_db_update += 1
                if bars_since_db_update >= 5000 and self._store and sid:
                    self._store.update_scenario_bars(sid, replay.bars_emitted)
                    bars_since_db_update = 0

                _write_agent_status(
                    self.NAME, "running",
                    f"Replay {symbol}/{timeframe} bar {replay.bars_emitted}",
                    self.interval_sec,
                )

            # Final DB update
            if self._store and sid:
                self._store.update_scenario_bars(sid, replay.bars_emitted)

            with self._lock:
                self._bars_emitted = replay.bars_emitted
                if not stop_flag[0]:
                    self._state = RUNNING  # allow _run_cycle to start next
                logger.info(
                    "[SimulatorAgent] Scenario %s/%s complete — %d bars",
                    symbol, timeframe, replay.bars_emitted,
                )

        except Exception as exc:
            logger.error("[SimulatorAgent] Replay error: %s", exc, exc_info=True)
            with self._lock:
                self._state = ERROR
                self._error_msg = str(exc)

        finally:
            self._flush_state()

    def _flush_state(self) -> None:
        """Write current state to JSON for dashboard polling."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            # get_status() acquires self._lock internally — re-acquiring the
            # non-reentrant lock here would deadlock the agent thread on
            # every cycle, which then hung /api/simulator/status forever.
            data = self.get_status()
            STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
