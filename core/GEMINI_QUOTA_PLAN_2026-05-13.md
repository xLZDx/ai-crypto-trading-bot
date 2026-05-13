# Gemini API quota plan — stop the 429 wall

**Date:** 2026-05-13
**Trigger:** operator screenshot of Gemini API Rate Limit dashboard showing **5 different Gemini Flash models in the red** (RPM and RPD both exceeded for the day) while the trading bot was running its AgenticLLM macro-veto cascade.

---

## What the dashboard says (28-day peak window)

| Model | Category | RPM | TPM | RPD | Status |
|---|---|---:|---:|---:|---|
| **Gemini 2.5 Flash** | Text-out | 7 / 5 | 22.8K / 250K | 28 / 20 | 🔴 OVER (RPM + RPD) |
| **Gemini 3 Flash** | Text-out | 7 / 5 | 31.2K / 250K | 32 / 30 | 🔴 OVER (RPM + RPD) |
| **Gemini 3.1 Flash Lite** | Text-out | 20 / 15 | 31.3K / 250K | **507 / 500** | 🔴 OVER (RPM + RPD) |
| **Gemini 2.5 Flash Lite** | Text-out | 13 / 10 | 31.7K / 250K | 30 / 20 | 🔴 OVER (RPM + RPD) |
| Gemini 2.5 / 3 / 3.1 Pro | Text-out | 0 / 0 | 0 / 0 | 0 / 0 | NOT GRANTED on free tier |
| **Gemma 3 1B / 4B / 12B / 27B / 2B** | Other | 0 / 30 | 0 / 15K | **0 / 14,400** | ✅ UNUSED — 720× more RPD headroom |
| **Gemma 4 26B / 31B** | Other | 0 / 15 | 0 / Unlimited | **0 / 1,500** | ✅ UNUSED — 75× more RPD |
| **Gemini 2.5 Flash Native Audio Dialog** | Live API | 0 / **Unlimited** | 0 / 1M | 0 / **Unlimited** | ✅ UNUSED — UNLIMITED quota |
| **Gemini 3 Flash Live** | Live API | 0 / **Unlimited** | 0 / 65K | 0 / **Unlimited** | ✅ UNUSED — UNLIMITED quota |
| Gemini Embedding 1 / 2 | Other | 0 / 100 | 0 / 30K | 0 / 1,000 | ✅ UNUSED — separate quota pool |

**Net read:** the bot is hammering the lowest-quota tier (Gemini Flash family with 20-30 RPD per model) while the highest-quota free-tier family (Gemma 3 with 14,400 RPD per model) sits idle.

---

## Where the bot's calls go today

[src/engine/agentic_llm.py:100-110](../src/engine/agentic_llm.py#L100) `_ALL_MODELS` cascade:

```
1.  gemini-3.1-pro-preview        ← 0 RPD on free tier (NOT GRANTED)
2.  gemini-3-pro-preview          ← 0 RPD on free tier
3.  gemini-2.5-pro                ← 0 RPD on free tier
4.  gemini-3.1-flash-lite-preview ← 500 RPD (hit today: 507/500)
5.  gemini-3-flash-preview        ← 30 RPD (hit: 32/30)
6.  gemini-2.5-flash              ← 20 RPD (hit: 28/20)
7.  gemini-2.5-flash-lite         ← 20 RPD (hit: 30/20)
8.  gemini-2.0-flash              ← 200 RPD (likely fine)
9.  gemini-2.0-flash-001          ← 200 RPD
10. gemini-2.0-flash-lite         ← 200 RPD
11. gemini-2.0-flash-lite-001     ← 200 RPD
```

Three of the top 7 candidates are NOT GRANTED on free tier and 4 of them hit RPD walls within 24h. After the cascade falls through, the bot caches `APPROVED` for 60s and lives off the cache.

---

## Plan — 4 changes, ranked by cost/benefit

### Change 1 — Add Gemma 3 to the cascade (FREE, highest impact)

Add `gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-4b-it`, `gemma-3-2b-it` to `_ALL_MODELS` BEFORE the throttled Gemini Flash entries. Gemma 3 is served through the same Gemini API endpoint, free tier, **14,400 RPD per model × 5 model variants = ~72,000 RPD** combined headroom.

For a binary "APPROVED / REJECTED" classification with a 200-token prompt, Gemma 3 27B is comparable in quality to Gemini Flash. The trade-off:
- Smaller model = slightly higher false-negative rate on nuanced macro-veto cases.
- Acceptable because the LLM is a *secondary* gate (the 9-gate risk stack runs first; LLM only vetoes on news/macro that the technical gates can't see).

**Effort:** 3 lines in `_ALL_MODELS` list. No retraining, no new code path.

```python
_ALL_MODELS = [
    "gemini-3.1-pro-preview",        # paid tier only
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemma-3-27b-it",                # NEW — 14,400 RPD free-tier
    "gemma-3-12b-it",                # NEW — 14,400 RPD free-tier
    "gemma-3-4b-it",                 # NEW — 14,400 RPD free-tier
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    # ...
    "gemma-3-2b-it",                 # NEW — last-ditch fallback
]
```

### Change 2 — Bump decision cache TTL from 60 s to 300 s (FREE, complementary)

[src/engine/agentic_llm.py:46](../src/engine/agentic_llm.py#L46): `_DECISION_TTL_S = 60.0` → `_DECISION_TTL_S = 300.0`.

The macro/news picture doesn't change in 60 seconds. 300 s is still well under the bot's 1 h trading cadence on the spot/futures market specialists, but cuts LLM calls per (symbol, action) tuple by 5×. Combined with Change 1, the daily LLM budget effectively becomes:

```
72,000 RPD (Gemma 3) × 5 (5-min cache) = 360,000 logical decisions/day
```

vs. today's ~800 RPD combined free-tier limit.

**Risk:** a fresh news headline takes up to 5 min to flip a cached APPROVED to REJECTED. Mitigation: there's already a separate `Validate Application Logs` rule on the dashboard banner, so operator-visible if it happens.

### Change 3 — Mirror the cascade into the Aider fallback wrapper (FREE)

[tools/aider_or_claude.py:31](../tools/aider_or_claude.py#L31) `DEFAULT_CHAIN` currently:

```python
DEFAULT_CHAIN = [
    "gemini/gemini-2.5-pro",        # 0 RPD free
    "gemini/gemini-2.5-flash",      # 20 RPD free — exhausted today
    "gemini/gemini-2.0-flash",      # 200 RPD free
    "gemini/gemini-2.0-flash-lite", # 200 RPD free
    "anthropic/claude-sonnet-4-6",  # only if key set
]
```

Updated chain — Gemma 3 inserted between Pro (paid) and Flash (low quota):

```python
DEFAULT_CHAIN = [
    "gemini/gemini-2.5-pro",
    "gemini/gemma-3-27b-it",        # 14.4K RPD
    "gemini/gemma-3-12b-it",        # 14.4K RPD
    "gemini/gemini-2.5-flash",
    "gemini/gemma-3-4b-it",         # 14.4K RPD
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite",
    "gemini/gemma-3-2b-it",         # 14.4K RPD last ditch
    "anthropic/claude-sonnet-4-6",
]
```

For Aider-grade code edits, Gemma 3 27B is a reasonable midpoint — won't match Claude/Gemini-Pro on long refactors but handles the kind of mechanical edits Aider is best at (cross-file renames, import additions, threshold tweaks).

### Change 4 — Set up paid Tier 1 billing (operator decision)

Free-tier walls don't lift without billing. The dashboard's "Set up billing" link does this in 2 minutes. Tier 1 pricing (per Google's published rates as of 2026-05-13):
- Gemini Flash: **~$0.075 / 1M input tokens · $0.30 / 1M output tokens**
- RPD on paid tier: **10,000 per Flash model** (vs. 20-500 free)
- RPM on paid tier: **1,000 per Flash model** (vs. 5-15 free)

The bot's LLM workload is ~200 input tokens × 50 output tokens × ~500 calls/day after Change 1+2 caching. That's:
- ~100,000 input + 25,000 output tokens/day
- ≈ $0.008 input + $0.0075 output = **$0.015/day = $5.50/year**

The cost is negligible relative to a single trade fee. The reason this is operator-decision: requires entering payment info in Google Cloud Console, not a code change.

---

## Recommended sequencing

1. **Now**: Changes 1 + 2 + 3 — code-only, free, 10 lines total across two files. Cuts daily LLM exhaustion risk from "1 hour of trading" to "weeks of trading" without a single paid call.
2. **This week**: Set up Tier 1 billing for resilience. $5.50/yr is essentially free; the value is the safety net for an unusual high-news day where even 360K cached decisions isn't enough.
3. **Next month**: evaluate Gemini 2.5 Flash Native Audio Dialog (UNLIMITED quota) for the operator-chat assistant on the dashboard. Different use case from AgenticLLM but cheap headroom for the chat feature.

---

## What this plan does NOT solve

- **Gemini Pro (paid)** access for higher-quality vetoes. The Gemini 2.5/3/3.1 Pro models all show 0/0 on free tier and would need Tier 2+ billing to use. Not worth it for binary classification.
- **Real-time news veto.** AgenticLLM today consumes headline tail from `live_news_buffer.py`; refreshing the headline window is independent of the LLM call rate.
- **Embedding-based cache lookup.** Could use `gemini-embedding-001` (free 1K RPD) to find near-duplicate prior decisions instead of exact (symbol, action) key match. Not in this plan — TTL bump in Change 2 captures most of the value.

---

## Files this plan would touch

| File | Lines | What changes |
|---|---|---|
| [src/engine/agentic_llm.py](../src/engine/agentic_llm.py) | ~10 | Add 4 Gemma 3 entries to `_ALL_MODELS`; bump `_DECISION_TTL_S` 60 → 300 |
| [tools/aider_or_claude.py](../tools/aider_or_claude.py) | ~5 | Add 4 Gemma 3 entries to `DEFAULT_CHAIN` |
| `.env` | 0 | (none — `GEMINI_API_KEY` already routes to Gemma) |
| `data/training_rules.json` | 0 | (none — LLM model selection is bot-side, not rules-side) |

**No tests to update** — the existing `test_agentic_llm_throttle.py` 6 tests cover the cache + cooldown behaviour and pass regardless of which model IDs are in `_ALL_MODELS`.

---

## Decision needed from operator

Three Y/N choices:
1. **Apply Changes 1 + 2 + 3 now?** (free, ~10-line change, immediate impact)
2. **Apply Change 4 today?** (set up paid Tier 1 billing — operator action in Google Cloud Console, ~$5/year)
3. **Use the same agents-first / Aider-ask-first flow for the code change?** Or just direct edit since it's a 10-line tweak.
