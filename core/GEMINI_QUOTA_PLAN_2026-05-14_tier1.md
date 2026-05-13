# Gemini API quota plan — Tier 1 update

**Date:** 2026-05-14
**Supersedes:** [`GEMINI_QUOTA_PLAN_2026-05-13.md`](./GEMINI_QUOTA_PLAN_2026-05-13.md)
**Trigger:** operator upgraded to Tier 1 billing. Free-tier walls lifted; cascade strategy needs re-ranking.

---

## What changed (Tier 1 vs free tier)

| Model | Free RPD | Tier 1 RPD | Tier 1 RPM | Tier 1 TPM | Δ |
|---|---:|---:|---:|---:|---|
| **gemini-3.1-pro-preview** | 0 | **250** | 25 | 2M | newly reachable |
| **gemini-2.5-pro** | 0 | **1,000** | 150 | 2M | newly reachable |
| **gemini-3.1-flash-lite-preview** | 500 | **100,000** | 4K | 4M | 200x |
| **gemini-3-flash-preview** | 30 | **10,000** | 1K | 2M | 333x |
| **gemini-2.5-flash** | 20 | **10,000** | 1K | 1M | 500x |
| **gemini-2.5-flash-lite** | 20 | **10,000** | 4K | 4M | 500x |
| **gemini-2.0-flash** | 200 | **Unlimited** | 2K | 4M | infinite |
| **gemini-2.0-flash-lite** | 200 | **Unlimited** | 4K | 4M | infinite |
| gemma-3-27b-it | 14,400 | 14,400 | 30 | 15K | unchanged |
| gemma-3-12b-it | 14,400 | 14,400 | 30 | 15K | unchanged |
| gemma-3-4b-it | 14,400 | 14,400 | 30 | 15K | unchanged |
| gemma-3-2b-it | 14,400 | 14,400 | 30 | 15K | unchanged |
| Live API (audio dialog, flash live) | Unlimited | Unlimited | Unlimited | 1M / 150K | unchanged |

**Net read:** Pro is now reachable for high-stakes vetoes. Gemini 2 Flash / 2 Flash Lite are the never-fail tier (Unlimited RPD). Gemma 3 demoted from "primary safety net" to "last-ditch backup".

---

## Budget discipline — $15 / year ($1.25 / month) hard cap

Operator constraint: **total Gemini API spend must stay at or below $15 / year (~$1.25 / month).** Tier 1 default cap is $250/month — far too loose. Three layers of defense:

1. **Cascade order: cheap-first.** Gemma 3 family (free-quota pool, 72,000 RPD combined) sits at the top — primary working set. Cheapest paid Flash (Gemini 2.0 Flash / Flash Lite — Unlimited RPD at ~$0.075/1M) sits next. Mid Flash after that. Pro models sit at the **bottom** — last-resort only.
2. **In-code budget guard.** New `LLM_MONTHLY_BUDGET_USD` env var (default `1.25`). Persisted MTD spend tracker in `data/llm_budget_state.json`. When MTD spend ≥ 80% of cap ($1.00), AgenticLLM hard-skips every Pro entry. When ≥ 95% ($1.19), it hard-skips paid Flash entries too — falls through to Gemma 3 only. When ≥ 100% ($1.25), AgenticLLM short-circuits with `APPROVED` (fail-open after exhaustion).
3. **Google Cloud Console budget alert** (operator side, not code). Set a $1.25/month budget at https://console.cloud.google.com/billing/budgets — Google emails you if real billing exceeds the cap. Supplementary safety net.

Together: cheap-first cascade keeps the typical case at $0; budget guard ramps down to Gemma-only as spend approaches cap; cloud alert pages you if the in-code tracker is wrong.

---

## Updated cascade for AgenticLLM (cheap-first, $15/year budget)

[src/engine/agentic_llm.py](../src/engine/agentic_llm.py) `_ALL_MODELS`:

```python
_ALL_MODELS = [
    # Tier A — free quota (Gemma 3, 72,000 RPD combined). Primary working set.
    "gemma-3-27b-it",                # 14,400 RPD — biggest, best quality
    "gemma-3-12b-it",                # 14,400 RPD
    "gemma-3-4b-it",                 # 14,400 RPD

    # Tier B — cheap paid (Gemini 2.0, Unlimited RPD, ~$0.075/1M input). Fallback when Gemma rate-limits.
    "gemini-2.0-flash-lite",         # Unlimited RPD — cheapest paid
    "gemini-2.0-flash",              # Unlimited RPD

    # Tier C — Gemma small fallback
    "gemma-3-2b-it",                 # 14,400 RPD — fastest

    # Tier D — mid Flash (only reached if Tiers A+B both rate-limited; budget guard may block at >80% MTD)
    "gemini-2.5-flash-lite",         # 10,000 RPD, ~$0.10/1M
    "gemini-3.1-flash-lite-preview", # 100,000 RPD, ~$0.10/1M
    "gemini-2.5-flash",              # 10,000 RPD, ~$0.10/1M
    "gemini-3-flash-preview",        # 10,000 RPD, ~$0.10/1M

    # Tier E — Pro (last resort; blocked by budget guard at >80% MTD)
    "gemini-2.5-pro",                # 1,000 RPD, $1.25-$10/1M
    "gemini-3.1-pro-preview",        # 250 RPD
]
```

**Removed from prior cascade:**
- `gemini-3-pro-preview` — not visible on Tier 1 dashboard (only 3.1 Pro is granted)
- `gemini-2.0-flash-001`, `gemini-2.0-flash-lite-001` — share quota with unversioned aliases; redundant

**Why this order:** AgenticLLM walks top-to-bottom; first-success wins. Gemma 3 27B handles the typical veto case at $0. Only when Gemma rate-limits (rare given 72,000 RPD capacity) does the bot reach paid Gemini 2.0 Flash (~$0.075/1M, cheapest paid option). Pro is reserved for the last-resort case where every Gemma + every cheap Flash + every mid Flash has rate-limited — virtually never happens in normal operation.

**Quality trade-off:** Gemma 3 27B catches the obvious cases. Pro would catch ~5-10% more nuanced vetoes (e.g., subtle SEC enforcement language). Acceptable because the LLM is a *secondary* gate — the 9-gate risk stack runs first; LLM only vetoes news/macro the technical gates can't see.

**Capacity headroom:** 72,000 free RPD + Unlimited cheap Flash / 500 cache-misses per day = effectively unbounded. Budget cap will never bind in normal operation.

---

## Updated Aider chain (cheap-first → Claude-harness fallback)

[tools/aider_or_claude.py](../tools/aider_or_claude.py) `DEFAULT_CHAIN`:

```python
DEFAULT_CHAIN = [
    # Free-quota first
    "gemini/gemma-3-27b-it",                 # 14,400 RPD
    "gemini/gemma-3-12b-it",                 # 14,400 RPD

    # Cheap-paid workhorse (good code quality at low rate)
    "gemini/gemini-2.0-flash",               # Unlimited RPD, ~$0.075/1M
    "gemini/gemini-2.0-flash-lite",          # Unlimited RPD

    # Mid Flash if needed
    "gemini/gemini-3.1-flash-lite-preview",  # 100,000 RPD

    # Last-resort Pro (only on hard multi-file refactors)
    "gemini/gemini-2.5-pro",                 # 1,000 RPD, $1.25-$10/1M
    # No anthropic/claude-sonnet entry — Claude Code session below already covers paid-Claude capacity.
]
```

**Why no Claude API entry:** the existing Claude Code session (this conversation, running on the operator's Claude Code subscription) is already paying for Claude capacity. Calling `anthropic/claude-sonnet-4-6` from Aider would double-bill. The cleaner architecture: when every Gemini entry exhausts (exit 42), control returns to the Claude Code session → it finishes the work directly. No incremental Claude cost.

**Aider trade-off:** for code refactors, Gemma 3 27B is weaker than Pro on long multi-file work. The cheap-first ordering means Aider may take more passes for the same edit, but each pass is essentially free. Pro is reserved for the hard cases where the operator explicitly wants Aider to one-shot a complex refactor.

Exit 42 still signals "every remote model exhausted" → Claude Code session takes over directly per the Agents-First Routing carve-out.

---

## Cache TTL still applies

Change 2 from the original plan is unchanged: bump `_DECISION_TTL_S` 60 -> 300 s in [src/engine/agentic_llm.py:46](../src/engine/agentic_llm.py#L46). Reasoning is unchanged — macro / news veto context does not shift in 60s. The 5x reduction in LLM call volume still saves Pro quota for the cases where it matters.

---

## Cost projection (Tier 1, $15/year budget)

Per Google's published Tier 1 rates as of 2026-05-14 (approximate; verify in console):

- **Gemma 3 family:** typically $0 at low volume on the free-quota pool (verify in console)
- **Gemini 2.0 Flash:** ~$0.075 / 1M input + ~$0.30 / 1M output
- **Gemini 2.5 Flash:** ~$0.10 / 1M input + ~$0.40 / 1M output
- **Gemini 2.5 Pro:** ~$1.25 / 1M input + ~$10.00 / 1M output (10-30x Flash on output)

Bot's LLM workload with Change 2 caching active (`_DECISION_TTL_S = 300`):
- ~500 cache-misses/day × 200 input + 50 output tokens = ~100K input + 25K output tokens/day = ~3.75M tokens/month

**Realistic monthly spend with the cheap-first cascade** (95% Gemma, 5% Gemini 2.0 Flash, 0% Pro):
- Gemma 95%: ~$0 (assumed free quota)
- Gemini 2.0 Flash 5%: ~5K input + 1.25K output tokens/day at ~$0.075/1M + $0.30/1M = ~$0.01/month
- **Total: under $0.05/month = $0.60/year** — leaves ~$14.40/year of the $15 budget as headroom

**Headroom usage:**
- One bad day where Gemma rate-limits hard and traffic spikes to Pro: even 50 Pro calls × 250 tokens × $1.25/1M input + $10/1M output = ~$0.15. The $14.40 reserve covers ~95 such incidents per year.
- New use cases (wizard free-text Q&A, embedding-based cache) can spend ~$1/month each within budget.

**Cap-breach scenarios:**
- If Gemma 3 family is unexpectedly billed at full Flash rates: ~$0.40/month = $4.80/year — well under $15.
- If a code bug spams LLM calls (no cache, no TTL): in-code guard halts Pro at 80% MTD ($1.00), halts paid Flash at 95% MTD ($1.19), short-circuits at 100% ($1.25/month). Bot stays alive on Gemma, no overage.
- Pro routing is rare-by-design (Tier E in cascade) — protects against the high-token-output blow-up that could spike spend.

**If the operator wants more Pro quality on news/macro vetoes:** add a tag at the AgenticLLM call site (`priority="news_macro"`) that promotes Pro to the head of the cascade for that specific call. The budget guard still applies. Not in this round — cheap-first cascade adequately covers >99% of vetoes.

---

## What's left to apply (code changes)

| File | Lines | What changes |
|---|---|---|
| [src/engine/agentic_llm.py](../src/engine/agentic_llm.py) | ~12 | Replace `_ALL_MODELS` with the new cascade; bump `_DECISION_TTL_S` 60 -> 300 |
| [tools/aider_or_claude.py](../tools/aider_or_claude.py) | ~7 | Replace `DEFAULT_CHAIN` with the new chain |
| `.env` | 0 | (none — `GEMINI_API_KEY` already routes to all Gemini models incl. Gemma) |
| `data/training_rules.json` | 0 | (none — LLM model selection is bot-side) |

No tests to update — `test_agentic_llm_throttle.py` covers cache + cooldown behavior and is model-id-agnostic.

---

## Headroom for new use cases (Tier 1)

With the quota walls lifted, two formerly-deferred use cases are now affordable:

1. **Training wizard free-text Q&A** (planned in [SPRINT_1A](./SPRINT_1A_PER_MODEL_AGENTS_AND_KPI.md)) — can route to gemini-2.5-pro for high-quality model improvement advice without quota panic.
2. **Dashboard chat assistant** — can use `gemini-2.5-flash-native-audio-dialog` (Live API, Unlimited) for operator voice chat. Different from AgenticLLM macro-veto.

Both stay as deferred follow-ups; no code in this round.

---

## What this plan still does NOT solve

- **Real-time news veto latency.** AgenticLLM still consumes headline tail from `live_news_buffer.py`; refreshing the headline window is independent of the LLM call rate.
- **Embedding-based cache lookup.** Gemini Embedding 1 has Unlimited RPD on Tier 1 — could now use it to find near-duplicate prior decisions instead of exact `(symbol, action)` key match. Worth a separate plan; not in this round.
- **TPM ceiling on Gemma 3 (15K).** Gemma 3 sustained throughput is bounded by TPM, not RPD. If a single decision burns >15K tokens it won't fit. For 200-token prompts this is irrelevant; for future long-context wizard sessions it might bind.

---

## Application order (within Phase 2 of the consolidated plan)

1. Edit [tools/aider_or_claude.py](../tools/aider_or_claude.py) `DEFAULT_CHAIN` (Change 3, ~7 lines) — via Aider-ask-first flow.
2. Edit [src/engine/agentic_llm.py](../src/engine/agentic_llm.py) — replace `_ALL_MODELS` and bump TTL (Changes 1 + 2, ~12 lines).
3. Run `pytest tests/test_agentic_llm_throttle.py` (6 tests, must stay at 0 failures).
4. Live smoke: trigger one AgenticLLM macro-veto on the dashboard and confirm the cascade log shows the new ordering.

Total: ~20 min coding + ~5 min validation.
