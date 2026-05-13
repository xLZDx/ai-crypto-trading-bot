"""
Tests for the AgenticLLM per-(symbol, action) decision cache + all-cooled-down
short-circuit + Phase 2 (2026-05-14) cheap-first cascade + budget guard.

Both are guards against the 2026-05-13 incident where signal-recursion caused
the same trade to be re-evaluated by 11 fallback Gemini models, all of which
were 429-quota-exceeded. Even with the recursion fix in place, signal storms
at 1+/sec per (symbol, side) would still flood the LLM endpoint without these
throttles. The budget guard adds a USD-spend cap on top of the rate cap so a
runaway bug can't blow through the operator's monthly budget.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def reset_module_state(tmp_path, monkeypatch):
    """Each test starts with empty caches + no model cooldowns + isolated budget state."""
    from src.engine import agentic_llm as a
    a._decision_cache.clear()
    a._model_cooldown_until.clear()
    a._reset_budget_state_for_tests()
    # Redirect budget state file into a per-test tmpdir so writes don't
    # pollute the real data/llm_budget_state.json.
    budget_path = tmp_path / "llm_budget_state.json"
    monkeypatch.setattr(a, "_BUDGET_STATE_PATH", str(budget_path))
    yield budget_path
    a._decision_cache.clear()
    a._model_cooldown_until.clear()
    a._reset_budget_state_for_tests()


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


def _make_llm_with_usage_metadata(model_id_used: str, input_tokens: int = 200, output_tokens: int = 50):
    """Mock client whose generate_content returns a response with usage_metadata."""
    llm = _make_llm_with_mock_client()
    resp = MagicMock()
    resp.text = '{"decision": "APPROVED", "reason": "ok"}'
    usage = MagicMock()
    usage.prompt_token_count = input_tokens
    usage.candidates_token_count = output_tokens
    resp.usage_metadata = usage
    llm._client.models.generate_content.return_value = resp
    return llm


# ============================================================
# Phase 2 budget-guard tests
# ============================================================


def test_gemma_call_does_not_increment_budget_state(reset_module_state):
    """Gemma is on the free-quota pool; calls must NOT write the state file."""
    from src.engine import agentic_llm as a
    llm = _make_llm_with_usage_metadata("gemma-3-27b-it", input_tokens=200, output_tokens=50)
    # Force the cascade to start with Gemma (it's already first in _ALL_MODELS).
    llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    # State file should NOT have been written (zero-cost short-circuit).
    assert not reset_module_state.exists(), (
        f"Gemma call must not write budget state; got {reset_module_state.read_text() if reset_module_state.exists() else '<missing>'}"
    )


def test_paid_call_increments_budget_state(reset_module_state, monkeypatch):
    """A call to a paid model (Gemini 2.0 Flash) must update MTD spend."""
    from src.engine import agentic_llm as a
    # Pin the cascade to a paid model only so the first call routes there.
    monkeypatch.setattr(a, "_ALL_MODELS", ["gemini-2.0-flash"])
    llm = _make_llm_with_usage_metadata("gemini-2.0-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    assert reset_module_state.exists(), "Paid call must write budget state"
    import json
    state = json.loads(reset_module_state.read_text())
    # 1M input * $0.075 + 1M output * $0.30 = $0.375
    assert state["spent_usd"] == pytest.approx(0.375, rel=1e-3), (
        f"Expected ~$0.375 for 1M+1M Flash tokens, got {state['spent_usd']}"
    )
    assert state["calls_by_model"]["gemini-2.0-flash"] == 1
    assert state["tokens"]["input"] == 1_000_000
    assert state["tokens"]["output"] == 1_000_000


def test_budget_filter_below_80pct_keeps_full_cascade(monkeypatch):
    """MTD spend under 80% of cap → no filtering, full cascade returned."""
    from src.engine import agentic_llm as a
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "1.00")
    monkeypatch.setattr(a, "_read_budget_state", lambda: {
        "month_key": "2026-05", "spent_usd": 0.50,  # 50%
        "calls_by_model": {}, "tokens": {"input": 0, "output": 0},
    })
    result = a._budget_filter(list(a._ALL_MODELS))
    assert result == list(a._ALL_MODELS), "Below 80% MTD must keep every model"


def test_budget_filter_at_80pct_drops_pro_keeps_flash_and_gemma(monkeypatch):
    """80-94.99% MTD → Pro removed; paid Flash + Gemma stay."""
    from src.engine import agentic_llm as a
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "1.00")
    monkeypatch.setattr(a, "_read_budget_state", lambda: {
        "month_key": "2026-05", "spent_usd": 0.85,  # 85%
        "calls_by_model": {}, "tokens": {"input": 0, "output": 0},
    })
    result = a._budget_filter(list(a._ALL_MODELS))
    assert all("-pro" not in m for m in result), f"Pro must be dropped: {result}"
    # Sanity: Gemma + paid Flash still present
    assert any(m.startswith("gemma-") for m in result)
    assert any(m == "gemini-2.0-flash" for m in result)


def test_budget_filter_at_95pct_drops_all_paid_keeps_gemma(monkeypatch):
    """95%+ MTD → only Gemma survives."""
    from src.engine import agentic_llm as a
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "1.00")
    monkeypatch.setattr(a, "_read_budget_state", lambda: {
        "month_key": "2026-05", "spent_usd": 0.96,  # 96%
        "calls_by_model": {}, "tokens": {"input": 0, "output": 0},
    })
    result = a._budget_filter(list(a._ALL_MODELS))
    assert all(m.startswith("gemma-") for m in result), (
        f"At 95%+ MTD only Gemma must survive: {result}"
    )
    # Sanity: Gemma is not empty (we still have a working cascade)
    assert len(result) >= 1


def test_budget_filter_at_100pct_returns_gemma_only(monkeypatch):
    """100%+ MTD → same gate as 95% (Gemma only)."""
    from src.engine import agentic_llm as a
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "1.00")
    monkeypatch.setattr(a, "_read_budget_state", lambda: {
        "month_key": "2026-05", "spent_usd": 1.50,  # 150%
        "calls_by_model": {}, "tokens": {"input": 0, "output": 0},
    })
    result = a._budget_filter(list(a._ALL_MODELS))
    assert all(m.startswith("gemma-") for m in result)


def test_budget_filter_zero_cap_disables_guard(monkeypatch):
    """LLM_MONTHLY_BUDGET_USD=0 → guard off, cascade unchanged regardless of spend."""
    from src.engine import agentic_llm as a
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "0")
    monkeypatch.setattr(a, "_read_budget_state", lambda: {
        "month_key": "2026-05", "spent_usd": 999.99,
        "calls_by_model": {}, "tokens": {"input": 0, "output": 0},
    })
    result = a._budget_filter(list(a._ALL_MODELS))
    assert result == list(a._ALL_MODELS), "cap=0 must disable the guard"


def test_evaluate_trade_short_circuits_when_only_paid_models_and_over_cap(reset_module_state, monkeypatch):
    """If the operator's cascade has zero Gemma entries AND spend >= 95% MTD,
    evaluate_trade must fall back to APPROVED without calling any model."""
    from src.engine import agentic_llm as a
    monkeypatch.setenv("LLM_MONTHLY_BUDGET_USD", "1.00")
    monkeypatch.setattr(a, "_ALL_MODELS", ["gemini-2.5-pro", "gemini-2.0-flash"])
    monkeypatch.setattr(a, "_read_budget_state", lambda: {
        "month_key": "2026-05", "spent_usd": 0.99,  # 99%
        "calls_by_model": {}, "tokens": {"input": 0, "output": 0},
    })
    llm = _make_llm_with_mock_client()
    decision, reason = llm.evaluate_trade("BTC_USDT", "BUY", "tech", [])
    assert decision == "APPROVED"
    assert "budget cap" in reason.lower()
    assert llm._client.models.generate_content.call_count == 0, (
        "Budget short-circuit must NOT call the LLM"
    )


def test_month_rollover_resets_spent(reset_module_state, monkeypatch):
    """When the persisted month_key doesn't match current month, spend resets to 0."""
    from src.engine import agentic_llm as a
    import json
    # Write a state file with a stale month
    stale = {"month_key": "1999-01", "spent_usd": 999.0,
             "calls_by_model": {"gemini-2.5-pro": 100},
             "tokens": {"input": 1, "output": 1}}
    reset_module_state.write_text(json.dumps(stale))
    state = a._read_budget_state()
    # Current month != "1999-01" so state is freshly zeroed
    assert state["spent_usd"] == 0.0
    assert state["calls_by_model"] == {}


def test_unknown_model_defaults_to_zero_rate(monkeypatch):
    """An unrecognized model ID gets (0.0, 0.0) rates — safer than raising or
    using an arbitrary default. Worst case: one untracked call before the
    operator updates the rate table."""
    from src.engine import agentic_llm as a
    rates = a._model_rates_usd_per_1m("some-future-model-not-in-table")
    assert rates == (0.0, 0.0)


def test_extract_token_counts_falls_back_to_char_estimate():
    """Response without usage_metadata → use 4-char-per-token heuristic."""
    from src.engine import agentic_llm as a
    resp = MagicMock()
    resp.text = "x" * 400  # 400 chars / 4 = 100 tokens
    resp.usage_metadata = None
    in_tok, out_tok = a._extract_token_counts(resp, prompt="y" * 200)
    assert in_tok == 50  # 200 / 4
    assert out_tok == 100  # 400 / 4


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
