"""
Regression test for the signal-topic recursion bug (2026-05-13).

Original incident: SpotAgent + FuturesAgent both SUBSCRIBED to 'signal' AND
re-PUBLISHED to 'signal' after their market-specialist validation. Because the
bus dispatches subscribers synchronously inside `publish()`, each upstream
signal re-entered the same handler chain — recursing until Python's stack
depth limit, AND placing one order per recursion level via RiskAgent →
ExecutionAgent. Free-tier Gemini was hammered 11×880ms per recursion level
trying to evaluate every dispatched signal.

Fix: market specialists publish to 'trade_signal' (NOT 'signal'); RiskAgent
subscribes to 'trade_signal' only.

This test guards the topology so a future refactor cannot silently reintroduce
the loop.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def fresh_bus(monkeypatch):
    """A bus with no subscribers — every test starts from clean state.

    Stubs out two production dependencies that would otherwise make the
    topology assertions non-deterministic in CI:

    - FuturesAgent's fail-closed funding-rate gate (returns a small positive
      funding so the gate passes). The gate itself is covered by its own
      unit tests; topology tests only care about bus dispatch.
    - SpotAgent's _get_ml_confidence path. python-reviewer flagged that the
      ml-confidence path can silently return 0.5 (below the 0.62 threshold)
      when the real model files are loaded — the test would then 'pass'
      with call_count==1, masking a real regression. Force-return a high
      confidence so SpotAgent always publishes when the rest of the gates
      pass.
    """
    from src.engine.agents.agent_bus import AgentBus
    import src.engine.agents.futures_agent as fut_mod
    monkeypatch.setattr(fut_mod, 'fetch_funding_rate',
                        lambda sym: 0.0001)
    # Patch SpotAgent's ML confidence so the decision doesn't depend on
    # whether btc_rf_model.joblib is loadable in CI.
    import src.engine.agents.spot_agent as spot_mod
    monkeypatch.setattr(spot_mod.SpotAgent, '_get_ml_confidence',
                        lambda self, sym, direction: 0.85)
    return AgentBus()


def _make_spot(bus, symbols=None):
    from src.engine.agents.spot_agent import SpotAgent
    return SpotAgent(symbols=symbols or ['BTC_USDT'],
                     data_getter=lambda s: None, bus=bus, interval_sec=999)


def _make_futures(bus, symbols=None):
    from src.engine.agents.futures_agent import FuturesAgent
    return FuturesAgent(symbols=symbols or ['BTC_USDT'],
                        data_getter=lambda s: None, bus=bus, interval_sec=999)


def _make_risk(bus):
    from src.engine.agents.risk_agent import RiskAgent
    # interval_sec=999 so background thread (if start() ever called) doesn't fire
    r = RiskAgent(initial_capital=10_000.0, bus=bus, interval_sec=999)
    # Stub LLM out completely — the production bug compounded with LLM 429s
    r._llm = None
    return r


def test_market_specialists_publish_trade_signal_not_signal(fresh_bus):
    """SpotAgent/FuturesAgent/ScalpingAgent must publish on 'trade_signal'."""
    spot = _make_spot(fresh_bus)
    futures = _make_futures(fresh_bus)

    # If they publish on 'signal', this listener would receive their output AND
    # re-trigger their own _on_signal — the recursion. After the fix, 'signal'
    # subscribers should NEVER hear from a market specialist.
    raw_signal_callback = MagicMock()
    trade_signal_callback = MagicMock()
    fresh_bus.subscribe('signal', raw_signal_callback)
    fresh_bus.subscribe('trade_signal', trade_signal_callback)

    # Inject one upstream signal (as SignalAgent would). meta_pass=True so the
    # market specialists don't drop it; confidence > both Spot (0.62) and
    # Futures (0.60) thresholds.
    fresh_bus.publish('signal', 'SignalAgent', {
        'symbol': 'BTC_USDT',
        'direction': 1,
        'confidence': 0.85,
        'meta_pass': True,
        'regime': 0,
        'raw_signals': {},
        'size_mult': 1.0,
    })

    # Raw-signal subscriber sees EXACTLY the one upstream message.
    assert raw_signal_callback.call_count == 1, (
        f"'signal' was re-published {raw_signal_callback.call_count - 1} extra times — "
        "market specialists are leaking back onto the 'signal' topic, "
        "which is the original recursion bug."
    )
    # Trade-signal subscriber sees one publish per specialist that approved it
    # (both Spot and Futures whitelist BTC_USDT, so both publish).
    assert trade_signal_callback.call_count == 2, (
        f"Expected 2 trade_signal publishes (Spot + Futures), got "
        f"{trade_signal_callback.call_count}. Check that both specialists "
        f"actually publish after validation."
    )


def test_risk_agent_runs_once_per_specialist_not_per_recursion(fresh_bus):
    """A single upstream signal must invoke RiskAgent N times (one per matching
    market specialist), NOT Python's recursion limit number of times."""
    spot = _make_spot(fresh_bus)
    futures = _make_futures(fresh_bus)
    risk = _make_risk(fresh_bus)

    # Spy on RiskAgent's handler — capture call count without changing behavior
    original_on_signal = risk._on_signal
    call_count = {'n': 0}

    def counting_on_signal(msg):
        call_count['n'] += 1
        return original_on_signal(msg)
    # Re-subscribe with the spy
    fresh_bus._subscribers['trade_signal'] = [counting_on_signal]

    fresh_bus.publish('signal', 'SignalAgent', {
        'symbol': 'BTC_USDT',
        'direction': 1,
        'confidence': 0.85,
        'meta_pass': True,
        'regime': 0,
        'raw_signals': {},
        'size_mult': 1.0,
    })

    # Before the fix this was unbounded (recursed to Python's limit, ~1000).
    # After the fix: one call per market specialist that matches the symbol.
    assert call_count['n'] == 2, (
        f"RiskAgent invoked {call_count['n']} times for a single upstream "
        f"signal. Expected exactly 2 (Spot + Futures publish to "
        f"'trade_signal'). >2 indicates the recursion loop returned."
    )


def test_spot_agent_does_not_recursively_invoke_itself(fresh_bus):
    """The handler must be invoked AT MOST ONCE per upstream signal."""
    spot = _make_spot(fresh_bus)
    futures = _make_futures(fresh_bus)

    call_count = {'spot': 0, 'futures': 0}
    original_spot = spot._on_signal
    original_fut = futures._on_signal

    def spy_spot(msg):
        call_count['spot'] += 1
        return original_spot(msg)

    def spy_fut(msg):
        call_count['futures'] += 1
        return original_fut(msg)

    fresh_bus._subscribers['signal'] = [spy_spot, spy_fut]

    fresh_bus.publish('signal', 'SignalAgent', {
        'symbol': 'BTC_USDT',
        'direction': 1,
        'confidence': 0.85,
        'meta_pass': True,
        'regime': 0,
        'raw_signals': {},
        'size_mult': 1.0,
    })

    assert call_count['spot'] == 1, (
        f"SpotAgent._on_signal called {call_count['spot']} times — must be 1. "
        "If >1, SpotAgent.publish is feeding back into its own subscription."
    )
    assert call_count['futures'] == 1, (
        f"FuturesAgent._on_signal called {call_count['futures']} times — must "
        "be 1. If >1, FuturesAgent.publish is feeding back."
    )


def test_off_whitelist_symbol_does_not_reach_risk_agent(fresh_bus):
    """Spot/Futures must filter by their `symbols` whitelist before publishing.
    If neither specialist owns the symbol, RiskAgent should never see it."""
    spot = _make_spot(fresh_bus, symbols=['BTC_USDT'])
    futures = _make_futures(fresh_bus, symbols=['BTC_USDT'])
    risk = _make_risk(fresh_bus)

    trade_signal_callback = MagicMock()
    fresh_bus.subscribe('trade_signal', trade_signal_callback)

    fresh_bus.publish('signal', 'SignalAgent', {
        'symbol': 'PEPE_USDT',   # not in either whitelist
        'direction': 1,
        'confidence': 0.85,
        'meta_pass': True,
        'regime': 0,
        'raw_signals': {},
        'size_mult': 1.0,
    })

    assert trade_signal_callback.call_count == 0, (
        "An off-whitelist symbol leaked through to 'trade_signal'. "
        "Market specialists must drop signals whose symbol is not in their "
        "self.symbols list."
    )


def test_meta_pass_false_signal_dropped_by_specialists(fresh_bus):
    """Signals with meta_pass=False (meta-labeler rejected) must not reach the
    'trade_signal' topic at all."""
    spot = _make_spot(fresh_bus)
    futures = _make_futures(fresh_bus)

    trade_signal_callback = MagicMock()
    fresh_bus.subscribe('trade_signal', trade_signal_callback)

    fresh_bus.publish('signal', 'SignalAgent', {
        'symbol': 'BTC_USDT',
        'direction': 1,
        'confidence': 0.85,
        'meta_pass': False,   # meta-labeler said no
        'regime': 0,
        'raw_signals': {},
        'size_mult': 1.0,
    })

    assert trade_signal_callback.call_count == 0, (
        "meta_pass=False signal leaked to 'trade_signal'. Specialists must "
        "honour the meta-labeler veto."
    )


def test_database_agent_still_captures_raw_signals(fresh_bus):
    """DBAgent listens to 'signal' for analytics — split topology must NOT
    silence the analytics pipeline."""
    # Inline lightweight DBAgent stand-in to avoid importing the real DB stack
    captured = []

    def db_signal_capture(msg):
        captured.append(msg.payload)
    fresh_bus.subscribe('signal', db_signal_capture)

    spot = _make_spot(fresh_bus)

    fresh_bus.publish('signal', 'SignalAgent', {
        'symbol': 'BTC_USDT',
        'direction': 1,
        'confidence': 0.85,
        'meta_pass': True,
        'regime': 0,
        'raw_signals': {},
        'size_mult': 1.0,
    })

    assert len(captured) == 1, (
        f"DB-style 'signal' subscriber saw {len(captured)} messages — must be "
        "exactly 1 (one upstream signal in, one captured)."
    )


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
