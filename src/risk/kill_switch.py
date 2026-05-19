"""
Automated Kill Switch — Sprint 0 §S0-3.

Polls five trigger sources on every trade-loop tick and pauses order submission
when any trigger fires. Operator can `reset()` via the dashboard, recording
who/why/when in `data/risk/kill_switch_state.json`.

State machine:
  RUNNING  →  PAUSED (auto: any trigger fires)
  PAUSED   →  RUNNING (manual: operator reset via /api/risk/kill_switch/reset)

Triggers (configurable thresholds in KillSwitchConfig):
  - daily_loss_R_multiple: today's realized PnL <= -3 × R (avg daily ATR baseline)
  - max_consecutive_losses: 5 closed losing trades in a row
  - latency_p99_ms: rolling 5-min p99 exchange latency > 500ms
  - drawdown_pct: equity drawdown from peak > 8%
  - calibration_brier_z: model Brier score z-score vs 30-day baseline > 2.0
  - slippage_pct: actual fill slippage vs expected price > 0.5% (Phase 10)

Persistence:
  data/risk/kill_switch_state.json   — current state, last trigger, reset history
  data/risk/consecutive_losses.json  — running count of consecutive losing trades

Integration:
  - Trade loop calls `get_kill_switch().evaluate(now)` before every order.
  - When paused, order submission is skipped (logged at WARNING level).
  - The risk_agent feeds the consecutive_losses counter on every closed trade.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / 'data' / 'risk' / 'kill_switch_state.json'
LOSSES_FILE = PROJECT_ROOT / 'data' / 'risk' / 'consecutive_losses.json'


@dataclass
class KillSwitchConfig:
    """Threshold configuration. Defaults follow the §S0-3 spec."""
    daily_loss_R_multiple:        float = 3.0     # pause if daily PnL <= -3 × R
    max_consecutive_losses:       int   = 5       # pause after 5 losers in a row
    latency_p99_ms_threshold:     float = 500.0   # pause if p99 latency > 500ms
    drawdown_pct_threshold:       float = 0.08    # pause at 8% peak-to-trough drawdown
    calibration_brier_z_threshold: float = 2.0    # pause when Brier z > 2σ
    slippage_pct_threshold:        float = 0.005  # pause when fill slippage > 0.5%
    rolling_window_minutes:       int   = 5       # window for latency & brier
    enabled:                      bool  = True    # master enable flag


@dataclass
class KillSwitchState:
    """Persisted state — JSON-serialized to STATE_FILE."""
    paused: bool = False
    paused_at: Optional[str] = None
    last_trigger: Optional[str] = None
    last_reset_at: Optional[str] = None
    last_reset_by: Optional[str] = None
    last_reset_reason: Optional[str] = None
    trigger_history: list[dict] = field(default_factory=list)  # last 20 trigger events


class KillSwitch:
    """
    Singleton trade-loop gate. Operator-reset only on PAUSE → RUNNING.

    Thread-safe: state mutations and persistence go through `_lock`.
    """

    _MAX_TRIGGER_HISTORY = 20

    def __init__(self, cfg: Optional[KillSwitchConfig] = None):
        self.cfg = cfg or KillSwitchConfig()
        self._lock = threading.Lock()
        self._state = self._load_state()

    # ── Public API ───────────────────────────────────────────────────────────

    def is_paused(self) -> bool:
        return self._state.paused

    def evaluate(self, ts: Optional[datetime] = None,
                 metrics: Optional[dict] = None) -> tuple[bool, Optional[str]]:
        """
        Check all triggers; return (paused, reason).

        `metrics` (optional) lets callers inject live values for the triggers
        they have. Missing keys fall back to file-based readers. Available keys:
            daily_pnl_R           float  — today's realized PnL in R units
            consecutive_losses    int    — current streak count
            latency_p99_ms        float
            drawdown_pct          float
            calibration_brier_z   float
        """
        if not self.cfg.enabled:
            return False, None

        ts = ts or datetime.now(timezone.utc)
        m = metrics or {}

        with self._lock:
            # If already paused, stay paused. Manual reset required.
            if self._state.paused:
                return True, self._state.last_trigger

            # Evaluate each trigger in order
            for trigger_name, trigger_fired in self._iter_triggers(m):
                if trigger_fired:
                    self._record_trigger(trigger_name, ts)
                    self._state.paused = True
                    self._state.paused_at = ts.isoformat()
                    self._state.last_trigger = trigger_name
                    self._persist()
                    logger.critical(
                        "[KillSwitch] PAUSED by trigger=%s at %s", trigger_name, ts.isoformat(),
                    )
                    return True, trigger_name
            return False, None

    def pause(self, reason: str, ts: Optional[datetime] = None) -> None:
        """Manual pause (operator panic button or test injection)."""
        ts = ts or datetime.now(timezone.utc)
        with self._lock:
            self._state.paused = True
            self._state.paused_at = ts.isoformat()
            self._state.last_trigger = f'manual: {reason}'
            self._record_trigger(f'manual: {reason}', ts)
            self._persist()
        logger.critical("[KillSwitch] manually PAUSED: %s", reason)

    def reset(self, operator: str, reason: str = '',
              ts: Optional[datetime] = None) -> dict:
        """Operator clears the pause. Returns the new state."""
        if not operator:
            raise ValueError('operator name required for reset')
        ts = ts or datetime.now(timezone.utc)
        with self._lock:
            was_paused = self._state.paused
            self._state.paused = False
            self._state.paused_at = None
            self._state.last_reset_at = ts.isoformat()
            self._state.last_reset_by = operator
            self._state.last_reset_reason = reason
            self._persist()
        logger.warning(
            "[KillSwitch] RESET by %s (reason=%s, was_paused=%s)",
            operator, reason, was_paused,
        )
        return self.state()

    def state(self) -> dict:
        """Snapshot for dashboard tile."""
        with self._lock:
            s = asdict(self._state)
        s['config'] = asdict(self.cfg)
        return s

    def record_trade_outcome(self, won: bool) -> None:
        """Update consecutive-loss counter on every closed trade.

        Called by the risk agent / order manager when a position closes.
        """
        try:
            count = self._read_losses_count()
            count = 0 if won else (count + 1)
            self._write_losses_count(count)
        except Exception as e:
            logger.warning("[KillSwitch] could not update consecutive losses: %s", e)

    # ── Internals ────────────────────────────────────────────────────────────

    def _iter_triggers(self, m: dict):
        """Yield (trigger_name, fired) tuples in evaluation order."""
        # 1. Daily loss
        daily_pnl_R = m.get('daily_pnl_R')
        if daily_pnl_R is not None and daily_pnl_R <= -self.cfg.daily_loss_R_multiple:
            yield 'daily_loss_R_multiple', True
            return
        # 2. Consecutive losses
        cons = m.get('consecutive_losses')
        if cons is None:
            try:
                cons = self._read_losses_count()
            except Exception:
                cons = 0
        if cons is not None and cons >= self.cfg.max_consecutive_losses:
            yield 'max_consecutive_losses', True
            return
        # 3. Latency p99
        p99 = m.get('latency_p99_ms')
        if p99 is not None and p99 > self.cfg.latency_p99_ms_threshold:
            yield 'latency_p99_ms', True
            return
        # 4. Drawdown
        dd = m.get('drawdown_pct')
        if dd is not None and dd >= self.cfg.drawdown_pct_threshold:
            yield 'drawdown_pct', True
            return
        # 5. Calibration Brier z-score
        brier_z = m.get('calibration_brier_z')
        if brier_z is not None and abs(brier_z) >= self.cfg.calibration_brier_z_threshold:
            yield 'calibration_brier_z', True
            return
        # 6. Slippage
        slip = m.get('slippage_pct')
        if slip is not None and slip > self.cfg.slippage_pct_threshold:
            yield 'slippage_pct', True
            return
        yield None, False

    def _record_trigger(self, name: str, ts: datetime) -> None:
        """Append a trigger event to the bounded history."""
        self._state.trigger_history.append({
            'trigger': name,
            'at': ts.isoformat(),
        })
        self._state.trigger_history = self._state.trigger_history[-self._MAX_TRIGGER_HISTORY:]

    def _read_losses_count(self) -> int:
        if not LOSSES_FILE.exists():
            return 0
        try:
            d = json.loads(LOSSES_FILE.read_text(encoding='utf-8'))
            return int(d.get('count', 0))
        except Exception:
            return 0

    def _write_losses_count(self, count: int) -> None:
        LOSSES_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            from src.utils.safe_json import write_json
            write_json(str(LOSSES_FILE), {
                'count': count,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            # Fall back to direct write — safe_json is optional
            LOSSES_FILE.write_text(json.dumps({'count': count}), encoding='utf-8')

    def _load_state(self) -> KillSwitchState:
        if not STATE_FILE.exists():
            return KillSwitchState()
        try:
            d = json.loads(STATE_FILE.read_text(encoding='utf-8'))
            if not isinstance(d, dict):
                return KillSwitchState()
            d.pop('config', None)  # don't restore stale config
            # Only retain known fields
            known = {f for f in KillSwitchState.__dataclass_fields__}
            d = {k: v for k, v in d.items() if k in known}
            return KillSwitchState(**d)
        except Exception as e:
            logger.warning("[KillSwitch] could not load state: %s -- starting fresh", e)
            return KillSwitchState()

    def _persist(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            from src.utils.safe_json import write_json
            write_json(str(STATE_FILE), asdict(self._state))
        except Exception:
            STATE_FILE.write_text(json.dumps(asdict(self._state), indent=2),
                                  encoding='utf-8')


# ── Module-level singleton ───────────────────────────────────────────────────

_singleton: Optional[KillSwitch] = None
_singleton_lock = threading.Lock()


def get_kill_switch(cfg: Optional[KillSwitchConfig] = None) -> KillSwitch:
    """Lazy-initialized singleton. `cfg` only honored on first call."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = KillSwitch(cfg=cfg)
    return _singleton


def reset_singleton_for_tests() -> None:
    """Tests-only: wipe the singleton + state file so each test starts clean."""
    global _singleton
    with _singleton_lock:
        _singleton = None
        if STATE_FILE.exists():
            STATE_FILE.unlink(missing_ok=True)
        if LOSSES_FILE.exists():
            LOSSES_FILE.unlink(missing_ok=True)
