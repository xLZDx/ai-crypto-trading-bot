import os
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from src.utils.safe_json import read_json, write_json

logger = logging.getLogger(__name__)

# Single lock covering BOTH module-level dicts. The 2026-05-13 review flagged
# that `_decision_cache.get` + later `.pop` plus the cascade-read of
# `_model_cooldown_until` from RiskAgent's thread are not atomic. CPython's
# GIL makes individual dict ops thread-safe but the read-check-then-act
# sequences are not — two threads on a cache miss would both fire the full
# 11-model cascade. Same lock for both dicts because `_cached_decision` and
# the cooldown short-circuit are both consulted on every evaluate_trade.
_cache_lock = threading.Lock()

_VALID_DECISIONS = {"APPROVED", "REJECTED"}
# Includes model-not-found codes so the loop falls through to the next model
# instead of breaking on stale/unavailable model IDs.
_TRANSIENT = [
    '429', 'quota', 'resource_exhausted',
    '503', 'unavailable', 'high demand', 'overloaded',
    '404', 'not found', 'invalid argument', 'unknown model',
]

# Models that recently 429/503'd are skipped for COOLDOWN_S to stop the
# log-spam cascade where every signal retries dead models in a row.
# Two tiers:
#   - Generic transient (503, server overload): short cooldown, recovery likely
#   - Free-tier quota (429 + free_tier in error body): long cooldown, the model
#     literally has limit:0 in our plan and won't recover until a billing change
_MODEL_COOLDOWN_S        = 300.0   # 5 min for generic 429/503
_FREE_TIER_COOLDOWN_S    = 3600.0  # 1 hour for free-tier=0 quota walls
_model_cooldown_until: dict[str, float] = {}

# Per-(symbol, action) decision cache. The 2026-05-13 incident showed that
# the same (AVAX_USDT, SELL) hit AgenticLLM hundreds of times per second once
# the signal-recursion + market-specialist loop kicked in. Even with the
# topology fix in place, a single specialist sending one signal per second
# does not need a fresh LLM evaluation each tick — the macro/news picture
# does not change at that rate. The 2026-05-14 Tier 1 plan bumped this from
# 60s to 300s (5 min) — macro/news context does not flip in 60 seconds and
# the 5x reduction in calls saves quota on Pro / paid tiers.
_DECISION_TTL_S = 300.0
_decision_cache: dict[tuple[str, str], tuple[float, str, str]] = {}
# key -> (expires_at_monotonic, decision, reason)

# Budget guard (Phase 2 of the 2026-05-14 Tier 1 quota plan).
# Tracks month-to-date USD spend across paid Gemini Flash / Pro calls and
# progressively drops the more expensive tiers from the cascade as spend
# approaches the configured cap. Gemma 3 calls cost $0 on the free-quota
# pool so they're never counted (and never blocked by the cap).
_BUDGET_ENV = "LLM_MONTHLY_BUDGET_USD"
_DEFAULT_BUDGET_USD = 1.25  # $15/year as a soft monthly ceiling
_BUDGET_STATE_PATH = "data/llm_budget_state.json"

# Approximate Tier 1 rates per Google's published pricing as of 2026-05-14.
# Verify in console; pricing may change. Pro is intentionally last in the
# cascade so we rarely touch the high output-token rate.
# Tuple: (input_$_per_1M, output_$_per_1M).
_TIER_1_RATES_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-3.1-pro-preview":        (1.50, 12.00),
    "gemini-2.5-pro":                (1.25, 10.00),
    "gemini-3-pro-preview":          (1.25, 10.00),
    "gemini-3.1-flash-lite-preview": (0.10, 0.40),
    "gemini-3-flash-preview":        (0.10, 0.40),
    "gemini-2.5-flash":              (0.10, 0.40),
    "gemini-2.5-flash-lite":         (0.075, 0.30),
    "gemini-2.0-flash":              (0.075, 0.30),
    "gemini-2.0-flash-lite":         (0.075, 0.30),
    "gemini-2.0-flash-001":          (0.075, 0.30),
    "gemini-2.0-flash-lite-001":     (0.075, 0.30),
    # Gemma family is free-quota; explicit 0.0 entries keep the dict lookup
    # cheap and document the policy. Models absent from this dict default
    # to (0.0, 0.0) — see _model_rates_usd_per_1m for the fallback.
    "gemma-3-27b-it": (0.0, 0.0),
    "gemma-3-12b-it": (0.0, 0.0),
    "gemma-3-4b-it":  (0.0, 0.0),
    "gemma-3-2b-it":  (0.0, 0.0),
    "gemma-3-1b-it":  (0.0, 0.0),
}

# Budget guard one-shot warning flag so we don't spam logs every call.
_budget_cap_warned_at_pct: float = 0.0


def _model_rates_usd_per_1m(model_id: str) -> tuple[float, float]:
    """Return (input_rate, output_rate) per 1M tokens for `model_id`.
    Unknown models default to (0.0, 0.0) — they won't trip the cap, and
    if it's a paid model we missed, the worst case is one untracked call
    until the operator adds the rate to the table."""
    return _TIER_1_RATES_USD_PER_1M.get(model_id, (0.0, 0.0))


def _is_free_quota_model(model_id: str) -> bool:
    """Gemma family is the free-quota tier on Tier 1."""
    return model_id.startswith("gemma-")


def _is_pro_model(model_id: str) -> bool:
    """Pro models are the most expensive output tokens — first to be dropped."""
    return "-pro" in model_id


def _read_budget_state() -> dict:
    """Read MTD spend tracker. Auto-resets at month boundary."""
    raw = read_json(_BUDGET_STATE_PATH, default={})
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    if not isinstance(raw, dict) or raw.get("month_key") != current_month:
        return {
            "month_key": current_month,
            "spent_usd": 0.0,
            "calls_by_model": {},
            "tokens": {"input": 0, "output": 0},
            "last_call_iso": None,
        }
    return raw


def _budget_cap_usd() -> float:
    """Read LLM_MONTHLY_BUDGET_USD from env. Returns the cap or 0.0 if
    invalid / disabled. 0.0 means "no cap" — every paid model stays in
    the cascade."""
    raw = (os.environ.get(_BUDGET_ENV) or "").strip()
    if not raw:
        return _DEFAULT_BUDGET_USD
    try:
        val = float(raw)
        return max(val, 0.0)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using default %.2f.",
            _BUDGET_ENV, raw, _DEFAULT_BUDGET_USD,
        )
        return _DEFAULT_BUDGET_USD


def _record_call_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Update the MTD spend tracker. Returns the cost of THIS call in USD.
    Skips disk I/O for zero-cost (Gemma) calls to keep the hot path cheap."""
    in_rate, out_rate = _model_rates_usd_per_1m(model_id)
    cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0
    if cost <= 0.0:
        return 0.0
    with _cache_lock:
        state = _read_budget_state()
        state["spent_usd"] = float(state.get("spent_usd", 0.0)) + cost
        calls = state.setdefault("calls_by_model", {})
        calls[model_id] = int(calls.get(model_id, 0)) + 1
        toks = state.setdefault("tokens", {"input": 0, "output": 0})
        toks["input"] = int(toks.get("input", 0)) + int(input_tokens)
        toks["output"] = int(toks.get("output", 0)) + int(output_tokens)
        state["last_call_iso"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            write_json(_BUDGET_STATE_PATH, state)
        except Exception as e:
            # Don't crash on budget-state write failures — the cap is a
            # convenience guard, not a safety-critical control.
            logger.warning("LLM budget state write failed: %s", e)
    return cost


def _budget_filter(models: list[str]) -> list[str]:
    """Progressively drop paid tiers from the cascade as MTD spend approaches
    the cap. Gemma always passes through (free quota).

    Thresholds (% of LLM_MONTHLY_BUDGET_USD):
      <80%       : full cascade
      80-94.99%  : drop Pro models (expensive output tokens)
      95-99.99%  : drop ALL paid models (Pro + Flash); Gemma only
      >=100%     : Gemma only (paid cascade frozen until month rollover)
    """
    cap = _budget_cap_usd()
    if cap <= 0.0:
        return models  # no cap configured
    spent = float(_read_budget_state().get("spent_usd", 0.0))
    pct = spent / cap
    _maybe_warn_budget(pct, spent, cap)
    if pct >= 0.95:
        return [m for m in models if _is_free_quota_model(m)]
    if pct >= 0.80:
        return [m for m in models if not _is_pro_model(m)]
    return models


def _maybe_warn_budget(pct: float, spent: float, cap: float) -> None:
    """One-shot per threshold band. Resets if pct drops below the band
    (e.g., month rollover)."""
    global _budget_cap_warned_at_pct
    thresholds = (1.0, 0.95, 0.80)
    for t in thresholds:
        if pct >= t and _budget_cap_warned_at_pct < t:
            level = logging.CRITICAL if t >= 1.0 else logging.WARNING
            logger.log(
                level,
                "LLM budget at %.0f%% MTD ($%.4f / $%.2f cap). "
                "%s",
                pct * 100, spent, cap,
                "Paid models frozen until month rollover; Gemma-only cascade active." if t >= 0.95
                else ("Pro models dropped from cascade; paid Flash + Gemma still active." if t >= 0.80
                      else ""),
            )
            _budget_cap_warned_at_pct = t
            return
    if pct < 0.80 and _budget_cap_warned_at_pct > 0.0:
        _budget_cap_warned_at_pct = 0.0  # rolled over to a new month


def _extract_token_counts(response, prompt: str) -> tuple[int, int]:
    """Pull (input, output) token counts from the Gemini response.
    Falls back to a 4-char-per-token estimate if usage_metadata is absent."""
    try:
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            pt = getattr(um, "prompt_token_count", None)
            ct = getattr(um, "candidates_token_count", None)
            if pt is not None and ct is not None:
                return int(pt), int(ct)
    except Exception:
        pass
    in_est = max(1, len(prompt) // 4)
    out_est = max(1, len(getattr(response, "text", "") or "") // 4)
    return in_est, out_est


def _reset_budget_state_for_tests() -> None:
    """Test-only: clear in-memory budget warning flag."""
    global _budget_cap_warned_at_pct
    _budget_cap_warned_at_pct = 0.0


def _is_cooled_down(model_id: str) -> bool:
    # time.monotonic() is used for BOTH dicts — clocks must not be mixed
    # across the two caches even though they are read independently. Mixing
    # time.time() (wall-clock, can jump on NTP sync) with time.monotonic()
    # is a latent correctness trap flagged by the python-reviewer.
    with _cache_lock:
        return _model_cooldown_until.get(model_id, 0.0) > time.monotonic()


def _mark_cooldown(model_id: str, seconds: float = _MODEL_COOLDOWN_S) -> None:
    with _cache_lock:
        _model_cooldown_until[model_id] = time.monotonic() + seconds


def _cached_decision(symbol: str, action: str) -> tuple[str, str] | None:
    """Return a cached (decision, reason) if the same (symbol, action) was
    evaluated within the last _DECISION_TTL_S. None otherwise.

    REJECTED decisions reuse the same TTL as APPROVED — operators trade off
    rejection-stickiness against signal-storm suppression. If a stale reject
    becomes painful, lower _REJECT_DECISION_TTL_S separately.
    """
    with _cache_lock:
        entry = _decision_cache.get((symbol, action))
        if entry is None:
            return None
        expires, decision, reason = entry
        if expires <= time.monotonic():
            _decision_cache.pop((symbol, action), None)
            return None
        return decision, reason


def _cache_decision(symbol: str, action: str, decision: str, reason: str) -> None:
    with _cache_lock:
        # Cap the cache at 500 entries. In normal operation the bot trades a
        # fixed symbol set (O(10) × 3 actions = ~30 entries) so this never
        # triggers — but if the upstream symbol set ever becomes dynamic, the
        # cap prevents unbounded memory growth flagged by security-reviewer.
        if len(_decision_cache) >= 500:
            # Drop the oldest by expiry time. O(n) but n ≤ 500 and we hit
            # this branch only when the cap is breached.
            oldest_key = min(_decision_cache, key=lambda k: _decision_cache[k][0])
            _decision_cache.pop(oldest_key, None)
        _decision_cache[(symbol, action)] = (
            time.monotonic() + _DECISION_TTL_S, decision, reason,
        )

# Cheap-first cascade per the 2026-05-14 Tier 1 quota plan.
# Gemma 3 (free quota, 72,000 RPD combined) carries the typical case at $0.
# Gemini 2.0 Flash (Unlimited RPD on Tier 1) is the cheapest paid fallback.
# Mid Flash and Pro are reserved for the rare cases where Tiers A+B+C are
# all rate-limited simultaneously — virtually never happens in operation.
# The budget guard (_budget_filter) progressively drops paid entries as MTD
# spend approaches LLM_MONTHLY_BUDGET_USD.
_ALL_MODELS = [
    # Tier A — free quota
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    # Tier B — cheap paid (Unlimited RPD)
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    # Tier C — Gemma small fallback
    "gemma-3-2b-it",
    # Tier D — mid Flash
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    # Tier E — Pro (last resort; first to be dropped by budget guard at 80% MTD)
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
]


class AgenticLLM:
    """LLM-based risk manager that vetos trades on severe macro/news risk."""

    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        load_dotenv(os.path.join(project_root, '.env'))
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.is_active = False
        self._client = None

        if self.api_key and self.api_key != "your_api_key_here":
            try:
                from google import genai as _genai
                self._client = _genai.Client(api_key=self.api_key)
                self.is_active = True
                logger.info("Agentic LLM initialized (google.genai, gemini-3.1-pro-preview).")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("GEMINI_API_KEY not set. Agentic LLM disabled — trades auto-approved.")

    def evaluate_trade(self, symbol: str, action: str, technical_reason: str, headlines: list, telegram_monitor=None) -> tuple[str, str]:
        if not self.is_active or self._client is None:
            return "APPROVED", "Agent disabled (no API key), trade auto-approved."

        # Per-(symbol, action) decision cache — collapses repeated identical
        # evaluations within the TTL into one LLM call. Big win against signal
        # storms and per-tick re-evaluation.
        cached = _cached_decision(symbol, action)
        if cached is not None:
            return cached

        # Skip the entire 11-model cascade if every candidate is currently in
        # cooldown — there is nothing live to call. Returning APPROVED here
        # matches the existing "LLM unavailable -> fail-open" contract that
        # the catch-all at the bottom of this method also uses. Without this
        # short-circuit, sustained 429 storms would still try every cooled-
        # down model in sequence (was the 2026-05-13 banner cascade).
        all_models_dead = all(_is_cooled_down(m) for m in _ALL_MODELS)
        if all_models_dead:
            decision = "APPROVED"
            reason = "LLM models all cooled down (429/503) — trade auto-approved."
            _cache_decision(symbol, action, decision, reason)
            return decision, reason

        news_text = "\n".join(headlines[:20]) if headlines else "No recent news."

        telegram_text = "No recent Telegram analysis."
        if telegram_monitor and telegram_monitor.is_active:
            tg_messages = telegram_monitor.get_recent_messages()
            if tg_messages:
                telegram_text = "\n".join(f"- {msg}" for msg in tg_messages)

        prompt = (
            f"You are a strict AI Risk Manager for a crypto hedge fund.\n"
            f"The quantitative system wants to execute a {action} order for {symbol}.\n"
            f"Technical justification: {technical_reason}\n\n"
            f"Recent market news headlines:\n{news_text}\n\n"
            f"Proprietary Telegram Channel Analysis (Secondary source — use only if consistent with technicals):\n"
            f"{telegram_text}\n\n"
            f"VETO (REJECT) the trade only if news indicates a severe crash, hack, regulatory ban, "
            f"or massive macroeconomic risk. Otherwise APPROVE it.\n"
            f'Respond ONLY in valid JSON: {{"decision": "APPROVED" or "REJECTED", "reason": "1 short sentence"}}'
        )

        from google.genai import types as _gntypes
        
        ctrl = read_json('data/control.json', default={})
        selected_model = ctrl.get('selected_ai_model')
        if selected_model and selected_model not in _ALL_MODELS:
            models_to_try = [selected_model] + _ALL_MODELS
        elif selected_model:
            models_to_try = [selected_model] + [m for m in _ALL_MODELS if m != selected_model]
        else:
            models_to_try = list(_ALL_MODELS)

        # Filter out models cooling down from a recent 429/503. If every
        # candidate is cooled down (rare), retry the whole list anyway.
        active = [m for m in models_to_try if not _is_cooled_down(m)]
        if active:
            models_to_try = active

        # Budget guard — progressively drops paid models as MTD spend climbs.
        # Gemma (free quota) always passes through. If the operator's cascade
        # has zero free-quota models AND we're over the cap, this list goes
        # empty and we fall through to the fail-open contract below.
        models_to_try = _budget_filter(models_to_try)
        if not models_to_try:
            decision = "APPROVED"
            cap = _budget_cap_usd()
            reason = (f"LLM monthly budget cap (${cap:.2f}) reached — "
                      f"trade auto-approved.")
            _cache_decision(symbol, action, decision, reason)
            return decision, reason

        last_err = None
        for model_id in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=_gntypes.GenerateContentConfig()
                )
                # Record actual token usage for the budget tracker. Done
                # before parsing so even unparseable responses count against
                # the cap (we paid for the tokens regardless of usefulness).
                in_tok, out_tok = _extract_token_counts(response, prompt)
                _record_call_cost(model_id, in_tok, out_tok)
                raw_text = response.text
                match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
                if not match:
                    logger.warning(f"Agentic LLM non-JSON response: {raw_text[:200]}")
                    return "APPROVED", "LLM response unparseable — trade auto-approved."
                data = json.loads(match.group(0))
                decision = data.get("decision", "").upper()
                reason = data.get("reason", "No reason provided.")
                if decision not in _VALID_DECISIONS:
                    logger.warning(f"Unexpected LLM decision '{decision}' — defaulting to APPROVED.")
                    decision = "APPROVED"
                _cache_decision(symbol, action, decision, reason)
                return decision, reason
            except Exception as e:
                last_err = e
                err_lower = str(e).lower()
                if any(x in err_lower for x in _TRANSIENT):
                    if any(x in err_lower for x in ('429', 'quota', '503', 'unavailable', 'overloaded')):
                        # Free-tier quota walls are persistent (limit:0). Treat
                        # them with a long cooldown so we don't keep retrying
                        # ~3× per signal. Generic transient errors get the
                        # shorter recovery window.
                        is_free_tier = (
                            'free_tier' in err_lower
                            or 'free tier' in err_lower
                            or 'limit: 0' in err_lower
                        )
                        cd = _FREE_TIER_COOLDOWN_S if is_free_tier else _MODEL_COOLDOWN_S
                        _mark_cooldown(model_id, cd)
                    logger.debug(f"Agentic LLM: {model_id} transient error, trying fallback...")
                    continue
                break

        logger.error(f"Agentic LLM Error: {last_err}")
        # Cache the failure decision too — otherwise a signal storm during an
        # outage still cycles all 11 models per tick before falling through.
        fallback_reason = "LLM connection error — trade auto-approved."
        _cache_decision(symbol, action, "APPROVED", fallback_reason)
        return "APPROVED", fallback_reason

    def query(self, prompt: str, *, max_chars: int = 4000) -> dict[str, str | None]:
        """Free-form query path for non-trade callers (Wizard chat, training
        guidance, etc). Reuses the same Tier-1 cheap-first cascade,
        cooldown table, and MTD budget guard as evaluate_trade — the only
        difference is that we don't ask for a JSON {decision, reason}
        envelope, we return the raw model text.

        Phase C (2026-05-14) — wizard chat was previously routed through
        evaluate_trade(), which forced the LLM into APPROVE/VETO mode and
        produced answers like "this is a valid operational query and does
        not fall under VETO criteria" instead of actually answering the
        user's question. This bypass solves that.

        Returns dict with keys: answer (str), model_used (str|None), source (str).
        """
        if not self.is_active or self._client is None:
            return {"answer": "Agent disabled (no API key).",
                    "model_used": None, "source": "no_api_key"}

        all_models_dead = all(_is_cooled_down(m) for m in _ALL_MODELS)
        if all_models_dead:
            return {"answer": "LLM models all cooled down (429/503) — try again in a few minutes.",
                    "model_used": None, "source": "cooled_down"}

        ctrl = read_json('data/control.json', default={})
        selected_model = ctrl.get('selected_ai_model')
        if selected_model and selected_model not in _ALL_MODELS:
            models_to_try = [selected_model] + _ALL_MODELS
        elif selected_model:
            models_to_try = [selected_model] + [m for m in _ALL_MODELS if m != selected_model]
        else:
            models_to_try = list(_ALL_MODELS)

        active = [m for m in models_to_try if not _is_cooled_down(m)]
        if active:
            models_to_try = active
        models_to_try = _budget_filter(models_to_try)
        if not models_to_try:
            cap = _budget_cap_usd()
            return {"answer": f"LLM monthly budget cap (${cap:.2f}) reached — Q&A temporarily disabled.",
                    "model_used": None, "source": "budget_cap"}

        from google.genai import types as _gntypes
        last_err = None
        for model_id in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_id, contents=prompt,
                    config=_gntypes.GenerateContentConfig(),
                )
                in_tok, out_tok = _extract_token_counts(response, prompt)
                _record_call_cost(model_id, in_tok, out_tok)
                text = (getattr(response, 'text', None) or '').strip()
                if not text:
                    last_err = ValueError(f"{model_id}: empty response")
                    continue
                return {"answer": text[:max_chars],
                        "model_used": model_id, "source": "llm"}
            except Exception as e:
                last_err = e
                err_lower = str(e).lower()
                if any(x in err_lower for x in _TRANSIENT):
                    if any(x in err_lower for x in ('429', 'quota', '503', 'unavailable', 'overloaded')):
                        is_free_tier = ('free_tier' in err_lower or 'free tier' in err_lower
                                        or 'limit: 0' in err_lower)
                        cd = _FREE_TIER_COOLDOWN_S if is_free_tier else _MODEL_COOLDOWN_S
                        _mark_cooldown(model_id, cd)
                    continue
                break
        logger.error(f"AgenticLLM.query failed: {last_err}")
        return {"answer": f"LLM error: {last_err}", "model_used": None,
                "source": "llm_error"}
