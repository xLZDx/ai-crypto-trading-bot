import os
import json
import logging
import re
import threading
import time
from dotenv import load_dotenv
from src.utils.safe_json import read_json

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
# does not change at that rate. A 60s TTL collapses 60 signals into 1 call.
_DECISION_TTL_S = 60.0
_decision_cache: dict[tuple[str, str], tuple[float, str, str]] = {}
# key -> (expires_at_monotonic, decision, reason)


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

# Paid / most capable models first for trade decisions; free tier as fallback.
_ALL_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
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

        last_err = None
        for model_id in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=_gntypes.GenerateContentConfig()
                )
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
