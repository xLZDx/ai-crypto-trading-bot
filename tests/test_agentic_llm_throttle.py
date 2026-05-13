"""
Tests for the AgenticLLM per-(symbol, action) decision cache + all-cooled-down
short-circuit.

Both are guards against the 2026-05-13 incident where signal-recursion caused
the same trade to be re-evaluated by 11 fallback Gemini models, all of which
were 429-quota-exceeded. Even with the recursion fix in place, signal storms
at 1+/sec per (symbol, side) would still flood the LLM endpoint without these
throttles.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def reset_module_state():
    """Each test starts with empty caches + no model cooldowns."""
    from src.engine import agentic_llm as a
    a._decision_cache.clear()
    a._model_cooldown_until.clear()
    yield
    a._decision_cache.clear()
    a._model_cooldown_until.clear()


def _make_llm_with_mock_client():
    """Build an AgenticLLM bypassing real Gemini init, with a recorded client."""
    from src.engine.agentic_llm import AgenticLLM
    inst = AgenticLLM.__new__(AgenticLLM)
    inst.api_key = "fake"
    inst.is_active = True
    inst._client = MagicMock()
    # Default response: APPROVED
    resp = MagicMock()
    resp.text = '{"decision": "APPROVED", "reason": "ok"}'
    inst._client.models.generate_content.return_value = resp
    return inst


def test_decision_cached_within_ttl():
    llm = _make_llm_with_mock_client()
    # First call → hits the client
    d1, r1 = llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    # Second call same symbol/action → served from cache, no client hit
    d2, r2 = llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    assert d1 == d2 == "APPROVED"
    assert llm._client.models.generate_content.call_count == 1, (
        "Same (symbol, action) within TTL must not call the LLM again."
    )


def test_different_action_not_cached_together():
    llm = _make_llm_with_mock_client()
    llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    llm.evaluate_trade("BTC_USDT", "SELL", "tech", [])
    assert llm._client.models.generate_content.call_count == 2


def test_different_symbol_not_cached_together():
    llm = _make_llm_with_mock_client()
    llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    llm.evaluate_trade("ETH_USDT", "BUY", "tech", [])
    assert llm._client.models.generate_content.call_count == 2


def test_all_models_cooled_down_short_circuits():
    """If every model is in cooldown, return APPROVED without any API call."""
    from src.engine import agentic_llm as a
    # Push every model into cooldown. Uses monotonic to match the production
    # clock; mixing wall-clock here would silently mask a real cooldown bug.
    expires = time.monotonic() + 3600
    for m in a._ALL_MODELS:
        a._model_cooldown_until[m] = expires

    llm = _make_llm_with_mock_client()
    decision, reason = llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    assert decision == "APPROVED"
    assert "cooled down" in reason.lower()
    assert llm._client.models.generate_content.call_count == 0, (
        "All-cooled-down branch must NOT call the LLM."
    )


def test_partial_cooldown_still_tries_active_models():
    """If even one model is alive, the cascade still runs (no full short-circuit).

    Pinned to keep exactly ONE model live regardless of _ALL_MODELS ordering
    — earlier the test depended on `_ALL_MODELS[-1]` being the live one,
    which would silently flip semantics if the list was reordered.
    """
    from src.engine import agentic_llm as a
    live_model = a._ALL_MODELS[0]  # cool down everything except the first model
    expires = time.monotonic() + 3600
    for m in a._ALL_MODELS:
        if m == live_model:
            continue
        a._model_cooldown_until[m] = expires

    llm = _make_llm_with_mock_client()
    decision, _ = llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    assert decision == "APPROVED"
    # Exactly one call — the live model returned APPROVED on first try.
    assert llm._client.models.generate_content.call_count == 1, (
        f"Expected exactly 1 LLM call (the one live model), got "
        f"{llm._client.models.generate_content.call_count}. If 0, the "
        "all-cooled-down branch fired incorrectly; if >1, the cascade is "
        "calling cooled-down models too."
    )


def test_failure_cached_to_prevent_storm():
    """If the LLM call itself raises, the resulting APPROVED is still cached
    so a signal storm does not re-fire the full fallback cascade per tick."""
    llm = _make_llm_with_mock_client()
    # Make EVERY model raise a non-transient error — cascades break after 1st
    llm._client.models.generate_content.side_effect = RuntimeError("boom")

    d1, _ = llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    calls_after_first = llm._client.models.generate_content.call_count
    d2, _ = llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])

    assert d1 == "APPROVED"
    assert d2 == "APPROVED"
    assert llm._client.models.generate_content.call_count == calls_after_first, (
        "Failure decision must be cached too, otherwise signal storms during "
        "an LLM outage still cycle every model on every tick."
    )


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
