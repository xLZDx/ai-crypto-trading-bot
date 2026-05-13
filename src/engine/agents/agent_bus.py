"""
Agent Message Bus.

Thread-safe publish/subscribe message bus that decouples agents.
Each agent runs at its own frequency and communicates via typed messages.

Topics:
  'candle'    — new OHLCV candle available (from DataAgent)
  'signal'    — directional signal (from SignalAgent)
  'regime'    — market regime change (from SignalAgent)
  'risk'      — risk event: drawdown alert, circuit breaker (from RiskAgent)
  'order'     — order placed/filled/cancelled (from ExecutionAgent)
  'perf'      — performance alert: live/backtest divergence (from QuantAgent)
  'retrain'   — model retraining requested (from DataAgent)
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

# ── Agent status file ─────────────────────────────────────────────────────────
_STATUS_FILE = Path(__file__).resolve().parents[3] / 'data' / 'agent_status.json'
_status_write_lock = threading.Lock()
# Phase C7 (2026-05-12): history cap bumped 10 -> 100 so the operator has
# meaningful context when diagnosing why an agent stalled. The previous 10
# entries covered <1 minute of activity for fast agents.
_MAX_HISTORY = 100


def _write_agent_status(name: str, status: str, task: str, interval_sec: float) -> None:
    """Atomic per-agent heartbeat write — read+merge+write under one lock.

    Previous "Phase C reviewer fix" released `_status_write_lock` BEFORE the
    write_json call to avoid stalling other agents on the filelock. That
    optimization introduced a classic read-modify-write race:

      T1: thread A locks → reads disk → merges {A_new} → unlocks → write A
      T2: thread B locks → reads disk → still sees A_old → merges {B_new}
          → unlocks → write B  ← OVERWRITES A's new heartbeat

    At boot, 8 agents launched in succession all suffer this race. Fast-
    interval agents (60-300s) recover on their next cycle within seconds.
    Slow-interval agents (3600s) stay STALE in the dashboard for a full hour.
    DataAgent + SpotAgent at 3600s were flagged STALE every restart because
    of this.

    Fix: hold the lock through the write. Filelock contention is bounded
    (write_json is ~5ms), and 8 agents serializing on boot adds <100ms
    total — well within acceptable startup overhead. Heartbeat reliability
    is more valuable than the marginal parallelism the old code traded
    correctness for.
    """
    from src.utils.safe_json import read_json, write_json  # atomic JSON I/O
    now = datetime.now(timezone.utc).isoformat()
    now_ts = datetime.now(timezone.utc).timestamp()
    try:
        with _status_write_lock:
            data = read_json(str(_STATUS_FILE), default={}) or {}
            if not isinstance(data, dict):
                data = {}
            prev = data.get(name, {})
            history: list = list(prev.get('history', []))
            if prev.get('current_task') and prev['current_task'] != task:
                history.append({
                    'task': prev['current_task'],
                    'ts':   prev.get('last_heartbeat_ts', now_ts - interval_sec),
                    'status': prev.get('status', 'idle'),
                })
            history = history[-_MAX_HISTORY:]
            data[name] = {
                'status': status,
                'current_task': task,
                'last_heartbeat': now,
                'last_heartbeat_ts': now_ts,
                'interval_sec': interval_sec,
                'history': history,
            }
            # Write under the lock — closes the race.
            write_json(str(_STATUS_FILE), data, indent=2)
    except Exception as exc:
        logger.debug("[agent_bus] _write_agent_status(%s) failed: %s", name, exc)


@dataclass
class Message:
    topic: str
    sender: str
    payload: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AgentBus:
    """
    Central message bus. Agents subscribe to topics and publish messages.
    All operations are thread-safe.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        self._history: List[Message] = []
        self._max_history = 1000

    def subscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def publish(self, topic: str, sender: str, payload: Any) -> None:
        msg = Message(topic=topic, sender=sender, payload=payload)
        with self._lock:
            callbacks = list(self._subscribers.get(topic, []))
            self._history.append(msg)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        for cb in callbacks:
            try:
                cb(msg)
            except Exception as e:
                logger.error("[BUS] Callback error on topic '%s' from '%s': %s",
                             topic, sender, e)

    def get_latest(self, topic: str) -> Message | None:
        with self._lock:
            for msg in reversed(self._history):
                if msg.topic == topic:
                    return msg
        return None

    def get_history(self, topic: str, n: int = 10) -> List[Message]:
        with self._lock:
            return [m for m in self._history if m.topic == topic][-n:]


# Global singleton bus — all agents share this instance
_bus: AgentBus | None = None
_bus_lock = threading.Lock()


def get_bus() -> AgentBus:
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = AgentBus()
    return _bus


class BaseAgent:
    """
    Base class for all agents.
    Agents are background threads that subscribe to topics and publish results.
    """

    NAME = "BaseAgent"

    def __init__(self, bus: AgentBus | None = None, interval_sec: float = 60.0):
        self.bus = bus or get_bus()
        self.interval_sec = interval_sec
        self._running = False
        self._thread: threading.Thread | None = None
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Override in subclass to subscribe to bus topics."""

    def _run_cycle(self) -> None:
        """Override in subclass with the agent's main work."""

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=self.NAME, daemon=True)
        self._thread.start()
        logger.info("[%s] started (interval=%.0fs)", self.NAME, self.interval_sec)

    def stop(self) -> None:
        self._running = False
        logger.info("[%s] stopping...", self.NAME)

    def _loop(self) -> None:
        while self._running:
            try:
                _write_agent_status(self.NAME, 'running', 'Executing cycle', self.interval_sec)
                self._run_cycle()
                _write_agent_status(self.NAME, 'idle', 'Waiting for next cycle', self.interval_sec)
            except Exception as e:
                logger.error("[%s] cycle error: %s", self.NAME, e)
                _write_agent_status(self.NAME, 'error', f'Error: {e}', self.interval_sec)
            time.sleep(self.interval_sec)

    def publish(self, topic: str, payload: Any) -> None:
        self.bus.publish(topic, self.NAME, payload)
