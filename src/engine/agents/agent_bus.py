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


def _write_agent_status(name: str, status: str, task: str, interval_sec: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _status_write_lock:
        try:
            data: dict = {}
            if _STATUS_FILE.exists():
                try:
                    data = json.loads(_STATUS_FILE.read_text(encoding='utf-8'))
                except Exception:
                    pass
            data[name] = {
                'status': status,
                'current_task': task,
                'last_heartbeat': now,
                'interval_sec': interval_sec,
            }
            _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATUS_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception:
            pass


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
