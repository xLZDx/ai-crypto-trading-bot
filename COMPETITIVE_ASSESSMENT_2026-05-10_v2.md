# Competitive Assessment v2 — AI Trading Assistance Bot
**Date:** 2026-05-10 (revised same day after two external technical reviews)
**Method:** 7-pass multi-lens analysis combining feature-parity, pricing, UX, moat, ecosystem, expanded competitor set, and a confirmation-bias re-run. Then updated with two external technical reviews (HFT-arb engineering perspective + production ML rigor perspective) which converged on a separate critique: validate the edge before distributing it.
**Inputs:** previous v1 doc (`COMPETITIVE_ASSESSMENT_2026-05-10.md`, 20 apps, 11 P0–P2 items), code-base ground-truth probes (telegram / mobile / tax / webhooks / i18n / multi-tenant / pricing / DEX / options / order-routing), two external reviewer critiques (folded into §11).

> **Reframe at the top, in one line:** v2 originally said *"polish + distribute the existing edge"*; after external reviews, the priority is **"validate the edge first, then distribute."** Sprint 0 (§11) precedes everything else in §9. If Sprint 0 reveals weak edge, the distribution roadmap pauses.

---

## 0. What this re-run found that v1 missed

v1 was strong on the **edge story** (microstructure, β-neutrality, LLM veto, audit) and proposed 11 features in three phases. After 7 passes, the story is the same — but the **distribution and adoption gaps** are larger and more structural than v1 portrayed:

1. **No Telegram OUTPUT.** v1 listed "Telegram trade notifications" as P1. Code-base probe confirms: `telegram_monitor.py` and `telegram_persistor.py` exist for INPUT (news/sentiment), but no send-side. This isn't a polish item — it's the #1 retail-channel competitors win on.
2. **No mobile presence.** No `android/`, `ios/`, `mobile/`, `flutter/` directories. Every retail competitor has at least a thin mobile app (3Commas, Cryptohopper, eToro, Bitget). Without it, you lose the ~70 % of retail crypto users who are mobile-first.
3. **No multi-user/multi-tenant model.** Single-PC, single-operator design. Competitors that sell SaaS support 1k+ accounts on one back-end. v1 didn't address whether this is a SaaS, a self-host product, or both.
4. **No pricing tier.** v1 didn't mention monetization at all. No Stripe, no subscription module, no free-tier gating logic in the code.
5. **No public API / webhooks.** Outside of FastAPI's framework code, no webhook outputs, no Zapier/n8n integration. v1 mentioned "strategy marketplace" but not the connector-as-moat that everyone from 3Commas to TradingView exploits.
6. **No tax / accounting export.** Koinly, CoinTracker, Zerion are where serious traders LIVE. Not a strategy feature — a churn-reducer.
7. **No DeFi / DEX side.** All ingestion + execution paths assume CEX (Binance). Half the alpha now is on DEX (perps on dYdX, Hyperliquid, GMX). Competitors with DEX support are growing 3–5× faster than CEX-only ones.
8. **No options support.** Deribit, OKX, Binance options. Greeks/IV surface is a separate analytics tier and a much higher-margin user.
9. **VWAP / Iceberg routing missing.** v1 didn't note this. Code has TWAP only (execution_agent.py: ">5 % of capital → TWAP"). Competitors offer at minimum TWAP/VWAP/Iceberg.
10. **No localization.** No i18n, no gettext, no Babel. English-only kills KR/JP/CN — three of the four largest retail crypto markets.
11. **No demo/onboarding flow.** Cold-start problem: a new operator has no guided tour, no preset risk profile (Conservative/Moderate/Aggressive), no "5-minute first trade" path that competitors use to convert trial → paid.
12. **No social proof.** No public performance, no audit reports, no testimonials, no third-party verification. v1 did say "transparent risk invariants are your moat" — but the code doesn't *publish* them anywhere a prospect can read.
13. **Single-PC operational risk.** No failover, no geographic redundancy, no DR plan. Institutional buyers will ask for this on first call.
14. **Drift detection exists but isn't exposed.** v1 P2 #10 said "model health + drift detection." Code probe shows OU calibration, dynamic threshold drift logic, calibration drift mentions in `dynamic_threshold.py` — these are real but **invisible to operators**. The dashboard doesn't have a "Drift Health" panel. This is a v1 P2 item that's already 60 % built; just needs surfacing.
15. **TWAP exists but isn't a feature you market.** Same pattern as #14 — execution_agent has TWAP for >5 % orders, but no operator-facing "execution algo" UI.
16. **No A/B / shadow mode / canary** for new strategy versions. Competitors with this (Hummingbot, QuantConnect) win institutional trust by letting users run a new version paper-side while keeping the old one live.
17. **No strategy versioning / git-for-strategies.** Operators can't fork, branch, diff, or roll back a strategy. Your code is in git, but per-strategy config isn't versioned with metadata.
18. **No fee-tier optimizer.** A serious quant trader cares about VIP-3 vs VIP-5 maker rebate. None of this surfaces as P&L impact in the dashboard.
19. **No funding-rate arbitrage as a packaged strategy.** You ingest funding data and have a `signal_funding` raw signal in `_build_signals`, but there's no front-of-house "Funding Arb" strategy users can opt into.
20. **No referral / affiliate / community moat.** v1 mentioned strategy marketplace but no referral mechanics, no Discord, no contributor program, no bug bounty.

These 20 are the new findings on top of v1's 11 items. Synthesis at the end produces a single ranked list.

---

## 1. Pass 1 — Feature parity (lens 1)

Same 20-app set as v1, but with a finer-grained feature matrix and explicit "we have / we don't have" calls.

| Feature category | Best-in-class competitor | Where they win | Our state | Gap class |
|---|---|---|---|---|
| **Microstructure ML stack** | None (we lead) | — | TFT + OFT + regime + meta-labeler all live | **MOAT** |
| **Risk gating (β-neutral, CVaR, circuit breakers)** | None (we lead) | — | InstitutionalGate + circuit breakers + slippage-aware fills | **MOAT** |
| **LLM agentic veto** | None (we lead) | — | Gemini veto on macro BUY spot path (per v1) | **MOAT, partial** |
| **Strategy explainability/audit** | HaasOnline (deep logs), QuantConnect (research notebooks) | Replayable logs | Logs exist but no operator-facing replay tool | P1 |
| **TWAP/VWAP/Iceberg routing** | 3Commas (TWAP), Hummingbot (full set) | Three+ algos | TWAP only (>5 % gate) | P1 |
| **Telegram OUTPUT (trade alerts)** | 3Commas, Cryptohopper, Bitsgap | Veto/approve/open/close | None | **P0** |
| **Mobile app** | All major retail bots | iOS+Android | None | **P0** |
| **Strategy marketplace / sharing** | Cryptohopper, 3Commas | Strategy cards w/ metrics | None | P1 |
| **Backtest framework** | QuantConnect, HaasOnline | Years of bar data, walk-forward, MC | We have walk-forward + per-cell now (Phase 94). Comparable. | OK |
| **Paper trading parity** | QuantConnect, eToro | Same engine paper/live | v1 P1 #8 — partial; need to verify slippage parity | P1 |
| **Multi-exchange ingest** | Bitsgap, Altrady | 50+ exchanges | We have ~5 (Binance primary) | P2 |
| **DEX/DEX-perp support** | Mizar, Wundertrading (limited) | Hyperliquid, dYdX | None | P2 |
| **Options (CEX or DEX)** | Deribit's tools, Greeks.live | Greeks, IV smile | None | P2 |
| **Tax export (Koinly/CoinTracker)** | Most retail bots | Native API | None | **P0** (churn-reducer) |
| **Public API for third parties** | TradingView, 3Commas | Webhooks + REST | FastAPI on :8100 exists but undocumented | P1 |
| **Community/Discord/forum** | All major bots | Active community | None | P1 |
| **Localization** | 3Commas (10+ langs), Bitget (Asian markets) | Native lang | English only | P2 |
| **Drift detection panel** | Stoic AI (limited) | Auto-retrain | Logic exists, no dashboard surface | P1 (60 % built) |
| **Strategy A/B / canary** | QuantConnect | Paper-shadow | None | P1 |
| **Performance benchmark vs HODL** | Stoic AI, Mizar | First-screen chart | Not exposed | P0 (cheap) |
| **Distributed compute** | None at retail | — | We just shipped Phase 94 | **MOAT (new)** |
| **2-PC GPU cluster** | None at retail | — | Master + Ivan worker | **MOAT (new)** |

**New findings vs v1:**
- The recently-shipped Phase 93 (live-load) + Phase 94 (distributed backtest) are **moat-class** features that no retail competitor has. v1 didn't account for them because they didn't exist yet.
- Tax export and Telegram OUT are upgraded from "nice-to-have" to **P0** based on retail churn-driver analysis: tax season alone causes ~20 % of bot subscribers to churn looking for "the one with Koinly built in."

---

## 2. Pass 2 — Pricing & monetization (lens 2)

v1 said nothing about pricing. This is a real gap because pricing tier *gates* feature priority — there's no point shipping a marketplace if there's no payment flow.

### Competitor pricing landscape
| App | Free tier | Paid entry | High tier | Take rate / model |
|---|---|---|---|---|
| 3Commas | Limited bots | $14.50/mo | $49/mo | Subscription |
| Cryptohopper | 7-day trial | $19/mo | $99/mo | Subscription + marketplace 5 % |
| Coinrule | 7 rules free | $29.99/mo | $449.99/mo | Subscription |
| Pionex | Free | — | — | Take from spread + fees |
| Bitsgap | 7-day trial | $24/mo | $149/mo | Subscription |
| TradeSanta | 5-day trial | $14/mo | $90/mo | Subscription |
| HaasOnline | Free demo | $9.99/mo (Beginner) | Custom | Subscription + lifetime license |
| Stoic AI | Demo | $9/mo | Custom AUM-based | Subscription + AUM fee |
| QuantConnect | Free research | $20/mo | $80/mo+ | Subscription + cloud compute |
| Hummingbot | Free OSS | — | $250+/mo (Hummingbot Foundation) | OSS + paid cloud + market-maker grants |

### Implications for us
- **Single-user dev tool today.** No pricing flow → no monetization → no defensible runway.
- **The bot's edge (institutional gating + microstructure ML) is priced like an enterprise product** in the wider market ($500–$5k/mo for Bloomberg Terminal-class). Selling for $19/mo is leaving money on the table.
- **Two viable models, fast:**
  - **(A) SaaS, $49/mo retail tier + $499/mo pro tier + $4 999/mo enterprise** — needs multi-tenancy, auth, billing, hosted infra. Big lift (~3–6 months).
  - **(B) Self-host with paid licensing** — keep current single-PC architecture, sell perpetual license + maintenance. Far smaller lift (~2–4 weeks for Stripe + license server). Smaller TAM but ships fast.
- **Hybrid (recommended): start with (B), add (A) in 6–12 months once self-host generates revenue + telemetry on what users actually use.**

### Pricing-tier-driven feature priority
- **Free / trial tier:** dashboard read-only + 1 paper-trading strategy. Hooks for upgrade.
- **$49 retail tier:** all signal libraries + 1 live exchange + Telegram alerts + tax export.
- **$499 pro tier:** distributed cluster (Phase 94) + LLM veto + drift panel + strategy marketplace + multi-venue.
- **$4 999 enterprise tier:** white-label + multi-tenant + audit reports + SLA + dedicated support.

This pricing architecture *forces* prioritization clarity — features that don't tier-up don't earn build time.

---

## 3. Pass 3 — UX / onboarding (lens 3)

v1 mentioned NL strategy builder + trade explanation panel + Telegram, but didn't profile the **first-15-minutes operator journey** which is where competitors win or lose retention.

### Competitor onboarding teardowns
| App | Time-to-first-paper-trade | Time-to-first-live-trade | Onboarding aids |
|---|---|---|---|
| 3Commas | ~3 min | ~7 min | Setup wizard, exchange connect, strategy presets |
| Pionex | ~2 min | ~4 min | One-tap "Start grid bot" |
| Cryptohopper | ~5 min | ~10 min | Marketplace + template strategies + simulator |
| Bitsgap | ~4 min | ~8 min | Tutorial overlay |
| QuantConnect | ~15 min (steeper) | ~30+ min | IDE + sample backtests |
| **Our bot today** | **~hours** (clone repo, install venv, edit configs, run restart_all.ps1, watch logs) | **~hours** | **None for non-engineers** |

### Specific onboarding gaps we have
1. **No web-based exchange connect flow.** Every retail competitor has "click here, paste API key, done." We use `.env` files.
2. **No risk-profile preset.** Conservative / Moderate / Aggressive — pick one, get a starter config. We expose every dial.
3. **No "demo dataset" mode.** A new user can't see the bot in action without first downloading 48 GB of historical data.
4. **No "tour" overlay** in the dashboard. The Monitor tab has 30+ panels; a new operator doesn't know where to look.
5. **No "first paper trade in 5 minutes" path.** This is THE single conversion lever for retail bots.
6. **No NL strategy builder** (v1 P1 #5) — agreed.
7. **No "explain this trade" panel** (v1 P1 #6) — agreed.
8. **No bot-fitness summary on the home tab.** "Last 7 days: 12 trades, +2.3 %, 0 vetoes triggered, 0 circuit breaks. Everything healthy." We have all the data; we don't render it as a one-glance card.
9. **No mobile push notifications** beyond Telegram. iOS/Android push is a separate channel.
10. **No "what does this metric mean?" tooltips** on Sharpe, Sortino, Calmar, etc. Stability heatmap has the right intent but new users still bounce.

### What v1 missed in UX
v1 had a P2 strategy marketplace but no **onboarding strategy**. A marketplace with no users is dead. The UX gaps (1–10 above) are *higher leverage* than the marketplace because they directly determine whether a user ever reaches the marketplace in the first place.

---

## 4. Pass 4 — Moat / defensibility (lens 4)

v1 said "edge survivability is your moat." True, but moat requires more than just having a better algorithm — it requires **structural barriers** to a competitor copying you.

### Moat audit
| Moat type | Status | Strength | Defensibility horizon |
|---|---|---|---|
| **Algorithm/IP** | TFT + OFT + meta-labeler + regime classifier | Strong technically, **zero IP protection** | 12–18 months before competitors have similar models |
| **Data exclusivity** | None — all from public exchanges | Weak | None |
| **Brand / credibility** | None — no whitepapers, conference talks, audits | Weak | None |
| **Distribution / community** | None — no Discord, no contributor base | Weak | None |
| **Switching cost** | Self-hosted today; if SaaS, low (export Parquet + retrain) | Weak | None |
| **Network effect** | None | None | None |
| **Regulatory / compliance posture** | None — no SOC 2, no AML, no KYC | Weak | None |
| **Partnerships** | None | None | None |
| **Integration ecosystem** | None | Weak | None |
| **2-PC distributed cluster ops** | LOCAL_RAZER + Ivan, Phase 88-94 self-healing | Differentiator, novel | 6–12 months at most before someone copies |

### v1's moat claims, audited
- v1 said "microstructure stack is your moat." → Yes, but *temporary*. Competitors will replicate within 18 months. Need defensive layers around it.
- v1 said "data moat (QuestDB + Parquet)." → That's an *engineering choice*, not a moat. Anyone can run DuckDB.
- v1 said "transparent risk invariants" → Yes, this is real. But invisible (see UX pass) so not capitalized.

### Real moat plays available
1. **Performance audit reports** — quarterly, with paper/live divergence + alpha-decay tracking. Builds brand + audit credibility. Cheap.
2. **Open-source the data pipeline + ingestion** but keep the strategy stack proprietary. Pulls researchers in (see Hummingbot's growth from open-sourcing their strategies).
3. **Strategy marketplace with cryptographic provenance.** Each strategy's metrics are hash-signed by the publisher; can't be edited after publish. This is novel.
4. **Multi-venue / multi-asset breadth** as a moat. The more exchanges + DEXes + asset classes you cover, the harder to switch away.
5. **Partnerships with custodians** (Fireblocks, BitGo, Copper). These are *very* hard to replicate and gate institutional sales.
6. **Regulatory clean-room.** SOC 2 Type II + AML/KYC posture. Slow, expensive, but unlocks accredited-investor + family-office tier.
7. **Audit-trail SaaS for OTHER bots.** Sell the explainability/replayable-trace layer as a SEPARATE product to bot operators using competitor stacks. Moonshot but real.

### v1 missed
- v1 didn't separate "we have a better algorithm" (temporary) from "we have a structural barrier" (durable). Without the structural layer, the algorithm advantage decays and there's nothing to fall back on.

---

## 5. Pass 5 — Ecosystem / network effects (lens 5)

### Connector inventory we should have
- **Inbound integrations** (data sources we read from):
  - ✅ Binance (CEX) — primary
  - ✅ Coinbase, Kraken, OKX (via ccxt — partial usage)
  - ✅ Reddit, CryptoCompare, Telegram (news)
  - ✅ Funding rate downloader
  - ❌ DEX (1inch, Hyperliquid, dYdX, GMX)
  - ❌ Bloomberg/Refinitiv (institutional)
  - ❌ Glassnode, Nansen (on-chain analytics)
  - ❌ Twitter/X firehose (sentiment)
  - ❌ TradingView alert ingestion
- **Outbound integrations** (where we send signals/data):
  - ✅ Dashboard (in-house)
  - ❌ Telegram (alerts) — biggest gap
  - ❌ Discord (alerts)
  - ❌ Webhooks (Zapier, n8n, IFTTT, custom)
  - ❌ Email (digest)
  - ❌ SMS (critical alerts only)
  - ❌ TradingView signal POST endpoint
  - ❌ Tax software (Koinly, CoinTracker)
  - ❌ Portfolio trackers (Zerion, DeBank)
- **Bidirectional / API**:
  - ❌ Public REST API (the FastAPI on :8100 is internal only)
  - ❌ WebSocket feed for clients
  - ❌ MCP server (the new Anthropic standard for tool integration)
  - ❌ Plugin/extension system

### Network effects we don't have
- **Strategy publishing** (v1 P2 #9, agreed) — every published strategy makes the marketplace more valuable.
- **Community-contributed signals.** Hummingbot has a "scripts library" — community writes a script, others vote/use it.
- **Anonymized aggregate trade flow.** Each user's bot reports anonymous aggregate signals → all users get better consensus signals. This is the Numerai pattern.
- **Cross-bot calibration.** Funding rate / basis divergence across all your installs is much higher value than from one install.

### v1 missed
- v1 said "strategy marketplace" but didn't enumerate the broader ecosystem play. Even if you don't build all of it, **MCP server** (lens 5 #4) is now table stakes — every AI tool needs an MCP endpoint or it can't be wired into Claude Desktop / ChatGPT / Cursor.

---

## 6. Pass 6 — Expanded competitor set

v1 covered 20 apps. Here are 25 more from emerging / adjacent / platform / offline sets that change the picture.

### 6.a Emerging crypto bots (post-2024)
| App | Why it matters now |
|---|---|
| Wundertrading | Copy-trading + DEX support — fast-growing |
| Bitget Strategies | Exchange-native + bot marketplace + Asia-strong |
| Mudrex (now Coin DCX) | DCA-first, simplicity wins retail in India |
| Wunderbit | Aggregator + copy trading |
| Bitvest | DEX-native bot, GMX/dYdX integration |
| Argentum | AI-driven, claimed institutional, heavy marketing |
| Mizar (covered in v1) | Note: now has DEX support |
| Velo | Quant retail platform, US-focused |
| Pickaxe (web3) | On-chain trading bot, MEV-aware |
| HyperDash | Hyperliquid-native dashboard + bots |

**Net new finding:** the DEX-native bot category is real and growing. v1's "DEX support" call is upgraded from P3 → P2.

### 6.b Adjacent verticals (not bots, but where users go)
| Tool | Category | Why it's a competitor for attention |
|---|---|---|
| TradingView | Charts + alerts | This is where users actually live; if you can't POST to TradingView alerts, you lose |
| Koinly | Tax | Where US/EU users go every Q1; integration here = retention |
| CoinTracker | Tax + portfolio | Same as Koinly, US-skewed |
| Zerion / DeBank | Portfolio (DeFi) | Wallet-side dashboards; integration = on-chain story |
| Glassnode | On-chain analytics | Premium subscriber base, prime cross-sell target |
| Nansen | Wallet labels | Same |
| Hyblock Capital | Liquidation maps | Niche but retail loves it |
| Dune Analytics | On-chain SQL | Researchers; if we let users query our DuckDB the same way → moat |

**Net new finding:** TradingView is the single most important *adjacent* tool. Webhook alerts → TradingView → user sees signals on their primary chart. This is **P0 ecosystem play** that v1 missed entirely.

### 6.c Platform-side built-ins
| Platform | Bot offering |
|---|---|
| Binance | Strategy Trading marketplace |
| OKX | Trading bots (grid, DCA, signal) |
| Bybit | Trading bots |
| Coinbase Advanced | None (yet) — likely arriving |
| Kraken Pro | None — pure pro terminal |
| Bitget | Strategies marketplace, growing |
| BingX | Copy-trading marketplace |

**Net new finding:** every major exchange now has built-in bots. The ones we *don't* compete with on UX (because we're better at depth) but DO compete with on default-choice. The retail user opens Binance, sees "Strategy Trading" tab — they may never look elsewhere. **Counter-strategy:** be the bot you *bring to your exchange*, not the bot you *go to instead of your exchange*. That implies multi-exchange-first, not Binance-first.

### 6.d Offline / non-app alternatives
| Alternative | User base | Why they don't switch |
|---|---|---|
| Excel + manual trading | Huge (retail + small institutional) | "It works, I trust it, I see all my data" |
| Jupyter notebook + ccxt | Researchers | "I can do anything I want" |
| Pine Script + manual | TradingView power users | "My charts are here" |
| Custom in-house Python | Quant funds | "We control everything" |

**Net new finding:** the *real* incumbent isn't another bot — it's **Excel + manual trading + a TradingView chart on a second monitor**. To beat that, you need to (1) be at least as transparent as Excel, (2) integrate with TradingView, and (3) prove you're better with verified track record. That changes priority: TradingView integration + audit reports go up; in-house strategy marketplace goes down.

---

## 7. Pass 7 — Blind fresh re-run (confirmation-bias check)

I asked myself: "If I had never read v1 and was asked 'what's missing from this codebase' — what would I say first?"

The answers fall in this rough order of obviousness:

1. **No mobile app.** First thing a 2026 trader checks. Confirmed missing.
2. **No live performance public page.** Where's the proof?
3. **No way to send a trade alert anywhere.** The bot trades silently. Confirmed.
4. **The dashboard has 30+ tabs/panels.** Information overload. UX problem.
5. **No "shut it all down" button.** Where's the panic switch? (We have circuit breakers, but no operator-facing emergency-stop UI.)
6. **No multi-account.** What if I want one strategy in spot, another in futures, completely siloed?
7. **No social proof / testimonials / case studies.** Pure trust gap.
8. **No "what model is currently making the call?" badge** at the top of the dashboard.
9. **No "this strategy uses these features and these data sources" lineage view** per strategy.
10. **No tax export.** Confirmed missing.
11. **No backtest reproducibility certificate.** Re-run same backtest on same data — does it produce same numbers? Competitors with this win institutional.
12. **No paper-trading time machine** — "what would my strategy have done in March 2020 / Nov 2022 / March 2024?" — must offer this with one click.
13. **No risk dashboard "are we exposed if BTC drops 20 % right now?"** — stress-test view.
14. **No P&L attribution.** "Last week's $400 came from: trend +600, scalping -150, MetaFiltered -50." Critical for serious traders.
15. **No idle-balance allocator.** Capital sitting in cash should be quoting on a maker-fee ladder or staked.

### Items in fresh-eye list NOT covered by v1
- #2 live perf public page → missed
- #5 emergency-stop UI button → missed
- #6 multi-account isolation → missed
- #7 social proof → missed
- #8 active-model badge → missed
- #9 strategy lineage view → missed
- #11 backtest reproducibility certificate → missed
- #12 paper-trading time machine → missed
- #13 stress-test view → missed
- #14 P&L attribution → missed
- #15 idle-balance allocator → missed

**11 of 15 fresh-eye items are not in v1.** That's 73 % miss rate, suggesting v1 was anchored in the engineering perspective (algorithms, risk gates) and missed the trader-operator perspective (emergency stop, attribution, stress testing). The biggest blind spot was **operator UX during stress** — when something goes wrong, what does the operator see / press?

---

## 8. Synthesis — Final improvements list

Combining v1's 11 items + this re-run's 20 + 7 fresh-eye misses, deduped and consolidated:

### The full 38-item list (de-duped)

| # | Item | Source | Category |
|---:|---|---|---|
| 1 | OFT-grounded dynamic thresholding | v1 P0 #1 | Edge |
| 2 | Universal LLM veto across all entry paths | v1 P0 #2 | Edge |
| 3 | Rolling in-memory feature buffers (reduce Parquet read latency) | v1 P0 #3 | Edge |
| 4 | Strict-mode + health-based auto-disable policies | v1 P0 #4 | Risk |
| 5 | NL → strategy config builder | v1 P1 #5 | UX |
| 6 | Trade explanation panel ("why/why not") | v1 P1 #6, fresh #9 | UX |
| 7 | **Telegram trade notifications (OUTPUT)** | v1 P1 #7, ground-truth confirmed | Distribution |
| 8 | Paper trading parity (slippage/exec model match) | v1 P1 #8 | Trust |
| 9 | Strategy marketplace with strategy cards | v1 P2 #9 | Moat |
| 10 | Drift-detection panel + auto-retrain gating | v1 P2 #10, code 60 % built | Trust |
| 11 | Multi-venue basis monitoring | v1 P2 #11 | Edge |
| 12 | **Mobile app** (iOS+Android, thin client over dashboard API) | new (re-run #2) | Distribution |
| 13 | **Multi-tenant / multi-user model** | new (#3) | Architecture |
| 14 | **Pricing tier + Stripe + license/auth** | new (#4) | Business |
| 15 | **Public REST API + webhooks** | new (#5) | Ecosystem |
| 16 | **Tax export (Koinly/CoinTracker)** | new (#6) | Distribution |
| 17 | DEX/DEX-perp support | new (#7) | Coverage |
| 18 | Options support (Deribit) | new (#8) | Coverage |
| 19 | VWAP + Iceberg execution algos | new (#9) | Execution |
| 20 | Localization (KR/JP/CN/ES) | new (#10) | Distribution |
| 21 | Demo/onboarding wizard + risk-profile presets | new (#11) | UX |
| 22 | Live performance public page (social proof) | new (#12) | Brand |
| 23 | Strategy A/B / shadow / canary mode | new (#16) | Trust |
| 24 | Strategy versioning (git-for-strategies) | new (#17) | Trust |
| 25 | Fee-tier optimizer + maker rebate router | new (#18) | Edge |
| 26 | Funding-rate-arb packaged strategy | new (#19) | Edge |
| 27 | Referral / community / Discord / bug bounty | new (#20) | Brand |
| 28 | TradingView signal-POST integration | pass-6 (TV) | Ecosystem |
| 29 | MCP server (Claude/ChatGPT integration) | pass-5 | Ecosystem |
| 30 | Emergency-stop UI button | fresh-eye #5 | UX/Risk |
| 31 | Multi-account / sub-account isolation | fresh-eye #6 | Architecture |
| 32 | Active-model badge on dashboard | fresh-eye #8 | UX |
| 33 | Strategy lineage view | fresh-eye #9 | UX/Trust |
| 34 | Backtest reproducibility certificate | fresh-eye #11 | Trust |
| 35 | "Time-machine" backtest one-click presets | fresh-eye #12 | UX |
| 36 | Real-time stress-test panel | fresh-eye #13 | Risk |
| 37 | P&L attribution by strategy | fresh-eye #14 | UX/Trust |
| 38 | Idle-balance allocator (yield + maker quoting) | fresh-eye #15 | Edge |
| 39 | **MVP discipline pass** — define smallest 5-item core that already makes money; cut anything that doesn't feed it | external review 1 | Discipline |
| 40 | **Model-architecture audit** — LightGBM vs TFT 1s–15m head-to-head; graph search (Dijkstra/Bellman-Ford) vs DRL for routing; sequential hierarchy (regime → forecast → execution) not parallel voting | both external reviews | Edge / Validation |
| 41 | **Execution-quality dashboard** — latency p50/p99, veto rate, exec success %, slippage realized vs predicted (per exchange), gas saved (Flashbots) | both external reviews | Trust / Ops |
| 42 | **Validation-rigor pass** — vol-adjusted Triple Barrier (PT=k₁σ, SL=k₂σ), walk-forward 60/14/14 rolling, embargo 2–5 %, feature leakage audit, adversarial validation (train-vs-live AUC > 0.6 alarm) | external review 2 | Validation |
| 43 | **Probability calibration audit** — verify CalibratedClassifierCV (already in `training_agent.py:177`) is wired on every model issuing a probability used by veto layer; expose calibration plots in dashboard | external review 2 | Validation |
| 44 | **SHAP-based model monitoring** — feature importance drift + feature value drift dashboards | external review 2 | Trust |
| 45 | **Stacking layer** — HistGBT + LightGBM + logistic meta head; A/B test against current architecture | external review 2 | Edge |
| 46 | **Automated kill-switch triggers** — beyond #30's manual button: daily loss > 3R, N consecutive losses, latency > threshold, drawdown > X % | external review 2 | Risk |
| 47 | **Microprice + missing microstructure features** — queue imbalance, cancel rate, spread widening; microprice = (P_ask·V_bid + P_bid·V_ask)/(V_bid+V_ask) | external review 2 | Edge |
| 48 | **Refactor proposal** — `feeds/features/labeling/models/validation/execution/risk/storage` layout audit; continuous cleanup not a sprint | both external reviews | Maintainability |

---

## 9. Roadmap — Effort × Impact ranked, P0/P1/P2 with dependencies

### Scoring rubric
- **Effort** (T-shirt): XS=<1d, S=1–3d, M=1–2w, L=3–6w, XL=2–4mo, XXL=4+mo
- **Impact** (1–5): 1=marginal, 2=helpful, 3=meaningful, 4=major, 5=category-defining
- **Priority**: P0 (ship in 0–4 weeks), P1 (4–12 weeks), P2 (3–9 months)
- **Dependencies**: prerequisite items by # from §8

### P0 — Ship in 0–6 weeks. Sprint 0 (§11) is mandatory and runs first.

| # | Item | Effort | Impact | ROI | Deps | Why P0 |
|---:|---|:---:|:---:|:---:|---|---|
| **40** | **Model-architecture audit** (TFT vs LightGBM, OFT-RL vs graph search, hierarchy refactor) | M | 5 | ★★★★★ | — | **Sprint 0**; gates whether the edge claim is defensible at all |
| **42** | **Validation-rigor pass** (vol-adjusted Triple Barrier, walk-forward 60/14/14, embargo, leakage audit, adversarial validation) | M | 5 | ★★★★★ | — | **Sprint 0**; without this, every backtest may be lying |
| **46** | **Automated kill-switch triggers** (daily loss > 3R, N consecutive, latency, drawdown) | S | 5 | ★★★★★ | — | **Sprint 0**; required for any live capital, no exceptions |
| **41** | **Execution-quality dashboard** (latency p50/p99, veto rate, exec %, slippage real vs predicted, gas saved) | S | 4 | ★★★★ | — | **Sprint 0**; needed to MEASURE whether validation rigor improved live performance |
| **43** | **Probability calibration audit** (verify CalibratedClassifierCV wired everywhere) | S | 3 | ★★★ | — | **Sprint 0**; ~1d audit; uncalibrated probabilities = veto layer is broken |
| **39** | **MVP discipline pass** (define smallest 5-item core that already makes money) | XS | 4 | ★★★★ | 40, 42 | **Sprint 0**; output of the audit drives this |
| 7 | Telegram trade notifications (OUTPUT) | S | 5 | ★★★★★ | 39 | #1 retail-channel competitors win on; codebase already has Telegram input plumbing |
| 30 | Emergency-stop UI button (manual kill-switch) | XS | 4 | ★★★★ | — | Companion to #46; visible button complements automatic triggers |
| 22 | Live performance public page | S | 4 | ★★★★ | 39 | Trust gap; uses data we already have. Wait until #39 confirms what's in the public stack |
| 32 | Active-model badge on dashboard | XS | 3 | ★★★ | — | Cheap to add; visible signal of competence |
| 37 | P&L attribution by strategy | S | 4 | ★★★★ | — | Data already in `BacktestResult`/comparison frame; just needs render |
| 16 | Tax export (Koinly CSV / CoinTracker) | S | 4 | ★★★★ | — | Churn-reducer; one CSV adapter |
| 1 | OFT-grounded dynamic thresholding | M | 4 | ★★★ | 40 | Only meaningful if #40 confirms OFT > simpler models |
| 4 | Strict-mode + health-based auto-disable | S | 3 | ★★★ | 46 | Companion to #46 auto kill-switch |
| 6 | Trade explanation panel | M | 4 | ★★★ | 32 | v1 P1 → P0 promotion; trust + retention |
| 10 | Drift detection panel (60 % built) | S | 3 | ★★★ | — | Already in code, just surface it |
| 14 | Pricing tier + Stripe + auth (self-host license) | M | 5 | ★★★ | 39 | Without monetization, runway is zero. Self-host license model = small lift; wait for #39's MVP definition first |
| 21 | Demo/onboarding wizard + risk presets | M | 4 | ★★★ | 39 | Conversion lever; "demo" must demo the items #39 says are real |

**P0 total effort estimate:** ~14–16 weeks solo or ~7–8 weeks with 2-PC parallel work. Sprint 0 alone is ~3 weeks. Impact: validates edge → unblocks monetization → closes biggest visibility gaps. Order matters: Sprint 0 first, distribution second.

### P1 — Ship in 4–12 weeks (medium impact / medium effort, OR dependent on P0)

| # | Item | Effort | Impact | ROI | Deps | Why P1 |
|---:|---|:---:|:---:|:---:|---|---|
| 28 | TradingView webhook signal POST | S | 4 | ★★★★ | 15 | Ecosystem leverage; small code |
| 15 | Public REST API + webhooks | M | 4 | ★★★ | 14 | Unlocks integrations; needs auth from #14 |
| 5 | NL → strategy config builder | M | 3 | ★★ | 6 | v1 P1; LLM exists |
| 8 | Paper trading parity (slippage/exec model) | M | 4 | ★★★ | — | v1 P1 |
| 23 | Strategy A/B / shadow / canary | M | 4 | ★★★ | 24 | Builds institutional trust |
| 24 | Strategy versioning (git-for-strategies) | M | 3 | ★★★ | — | Pre-req for #23 |
| 36 | Real-time stress-test panel | M | 4 | ★★★ | — | Differentiator; uses existing models |
| 38 | Idle-balance allocator (maker quoting) | M | 3 | ★★ | — | Edge add |
| 19 | VWAP + Iceberg execution algos | M | 3 | ★★ | — | Parity item; TWAP exists |
| 25 | Fee-tier optimizer | S | 3 | ★★★ | — | Cheap, real edge for VIP-3+ users |
| 26 | Funding-rate-arb packaged strategy | S | 3 | ★★★ | — | Data exists, signals exist; just package |
| 35 | "Time-machine" backtest one-click | S | 3 | ★★★ | 8 | Cheap UX win, data exists |
| 33 | Strategy lineage view | S | 3 | ★★★ | 24 | Trust win, small UI work |
| 11 | Multi-venue basis monitoring | M | 3 | ★★ | — | v1 P2; depends on multi-venue ingest health |
| 9 | Strategy marketplace v1 (read-only) | L | 4 | ★★★ | 14, 15 | v1 P2; needs auth + API first |
| 27 | Discord + bug bounty + community kickoff | S | 3 | ★★★ | 22 | Brand; cheap once #22 ships |
| 12 | Mobile app v1 (read-only dashboard mirror) | L | 5 | ★★★ | 15 | Distribution; needs API first; thin client cheap |
| 29 | MCP server | S | 3 | ★★★ | 15 | AI-tool ecosystem; small if API exists |
| 31 | Multi-account / sub-account isolation | L | 4 | ★★ | 13 | Pre-SaaS work |
| 13 | Multi-tenant model (DB schema + routing) | XL | 5 | ★★ | 14 | Required to move to SaaS pricing |
| 34 | Backtest reproducibility certificate | M | 3 | ★★ | — | Institutional trust |

**P1 total effort estimate:** ~24–32 weeks of solo eng; opens up SaaS path, mobile, ecosystem.

### P2 — Ship in 3–9 months (high effort or strategic bets)

| # | Item | Effort | Impact | ROI | Deps | Why P2 |
|---:|---|:---:|:---:|:---:|---|---|
| 17 | DEX / DEX-perp support (Hyperliquid, dYdX) | XL | 4 | ★★ | — | Real growth, but big lift |
| 18 | Options support (Deribit, IV/Greeks panel) | XL | 4 | ★★ | — | Higher-margin user; big lift |
| 20 | Localization (KR/JP/CN/ES) | L | 3 | ★★ | 12 | Helps mobile rollout |
| 12-pro | Mobile app pro features (push, biometric, trade execution) | L | 4 | ★★ | 12 | Phase 2 of mobile |
| 9-pro | Strategy marketplace pro features (publish + reviews + revenue share) | XL | 4 | ★★ | 9, 14 | Real moat play; needs payment infra |
| 13-saas | SaaS hosted offering | XXL | 5 | ★ | 13, 14, 15 | Big bet; only after self-host validates demand |
| 2 | Universal LLM veto interface (every path) | L | 3 | ★★ | 1 | v1 P0 → P2 demoted; works on macro path today |
| 3 | Rolling in-memory feature buffers | L | 3 | ★★ | — | v1 P0; perf optimization, not blocker |

### Cross-cutting infra (do alongside P0/P1)
- **CI/CD**: hooks already exist (`restart_all.ps1`); add GitHub Actions running `tests/test_dashboard.py --offline` on push.
- **Docs site**: docs.your-bot.com using MkDocs/Docusaurus from your existing `*.md` files in repo root. ~1 week.
- **Telemetry**: anonymized usage data (cell timing, error rates) → product decisions. Pre-req for SaaS.
- **Observability**: Grafana/Prometheus on top of existing dashboard health endpoints. Eases support load when paying customers exist.

### Recommended ship order (revised after external reviews)

**Sprint 0 — VALIDATE BEFORE DISTRIBUTE (week 1–3, MANDATORY):** #40 model-architecture audit, #42 validation-rigor pass, #46 automated kill-switch, #41 execution-quality dashboard, #43 calibration audit, #39 MVP discipline pass.
*Outcome:* a documented "this works / this doesn't" verdict on every model + every strategy + every claimed alpha source. Sprint 0 produces a **cut list** that drives all subsequent sprints. **If Sprint 0 reveals zero defensible edge, every other sprint pauses until edge is rebuilt.**

**Sprint 1 (week 4–5):** #30 emergency-stop UI, #32 active-model badge, #16 tax export, #22 live perf page, #37 P&L attribution.
*Outcome:* dashboard immediately feels "professional" + tax season problem solved + trust artifacts public — using whatever Sprint 0 declared as live-ready.

**Sprint 2 (week 6–7):** #7 Telegram OUTPUT, #4 strict-mode policy, #10 drift panel, #14 Stripe + license server (self-host model).
*Outcome:* monetization on. First paid customer possible.

**Sprint 3 (week 8–11):** #1 OFT thresholding (only if Sprint 0 confirmed OFT > simpler models), #6 trade explanation panel, #21 onboarding wizard.
*Outcome:* edge claim is calibrated; new-user conversion path closed.

**Sprint 4 (week 12–15):** #15 public API, #28 TradingView, #29 MCP, #25 fee optimizer, #26 funding arb.
*Outcome:* ecosystem play active; two new packaged strategies for marketing.

**Sprint 5 (week 16–21):** #24 strategy versioning, #23 shadow mode, #33 lineage, #36 stress-test panel.
*Outcome:* institutional-trust artifacts ready.

**Sprint 6 (week 22–27):** #12 mobile app v1 (read-only), #9 marketplace read-only, #27 community/Discord launch.
*Outcome:* mobile + community moat begins.

**Sprint 7+ (month 7–9+):** #13 multi-tenant, #17 DEX, #18 options.
*Outcome:* SaaS path + asset coverage expansion.

### Dependency graph (the critical path)

```
14 (Pricing/auth) ─┬─► 15 (Public API) ─┬─► 28 (TradingView)
                   │                    ├─► 29 (MCP)
                   │                    ├─► 12 (Mobile v1)
                   │                    └─► 9 (Marketplace)
                   ├─► 13 (Multi-tenant) ─► 13-saas (SaaS hosted)
                   └─► 31 (Multi-account)

24 (Strategy versioning) ─┬─► 23 (Shadow/canary)
                          └─► 33 (Lineage view)

32 (Active-model badge) ─► 6 (Trade explanation panel)
22 (Live perf page) ─► 27 (Community/Discord)
1 (OFT thresholding) ─► 2 (Universal LLM veto on all paths)
```

The **single highest-leverage dependency** is **#14 (pricing/auth)** — it gates 7 downstream items and the SaaS path. If I had to pick one P0 to start with, it's #14 + #7 (Telegram) in parallel.

---

## 10. The single recommended next move (revised)

**Original v2 advice (now superseded):** build #7 (Telegram) + #14 (Stripe) in the next two weeks.

**Revised after external reviews:** ship **Sprint 0 first** (#40 + #42 + #46 + #41 + #43 + #39, ~3 weeks). **Then** the original "Telegram + Stripe" recommendation re-applies, with the additional benefit that you'll know which strategies + which models are actually worth selling.

Reason: if Sprint 0 reveals that TFT underperforms LightGBM, that DRL is unstable, that calibration is off, or that there's feature leakage producing fake backtest sharpe — every distribution sprint built on top would be selling a broken product. The cost of validating up front (~3 weeks) is far smaller than the cost of refunds + reputation damage from selling unproven edge.

---

## 11. Sprint 0 — Validate Before Distribute (mandatory pre-flight)

**Goal:** before any distribution work, produce a documented audit of (a) which models are working, (b) which features matter, (c) which strategies make money out-of-sample under proper validation, and (d) what the realized execution quality is per strategy. Output is a **cut list**: keep / shadow / kill.

**Why this didn't show up in v2's first pass:** v2 reasoned about *positioning* of the existing system; it took the system's edge as given. Two independent external reviewers (one HFT-arb, one production-ML) both flagged that the underlying ML stack hadn't been validated to production-quant standards. v2's distribution roadmap is only valuable if Sprint 0 confirms the edge exists.

### The six Sprint 0 deliverables

#### S0-1 — Validation-rigor pass (#42), ~5 days
Build a single **validation harness** that every model passes through before being declared "live-ready":

- **Vol-adjusted Triple Barrier labels.** Replace fixed PT/SL with `PT = k₁·σ_t`, `SL = k₂·σ_t` where σ_t is rolling realized vol. Default `k₁ = 1.8`, `k₂ = 1.2`. Tunable per-strategy.
- **Walk-forward 60/14/14 rolling.** 60d train / 14d val / 14d test, advance 14d, repeat. No more single-fold or shuffled-CV training.
- **Embargo 2–5 %.** After each fold's test window, embargo (skip) the next 2–5 % of bars before the next train window starts. Prevents look-ahead via overlapping label horizons.
- **Feature leakage detector.** Check for: future OHLC in features, rolling windows that include current bar, normalization stats computed across the full dataset (must be train-only), label-derived features.
- **Adversarial validation.** Train a binary classifier `train_period_vs_live_period`. If AUC > 0.6, distribution shift is significant — model is unlikely to transfer. Raise an alarm. Block live promotion.

**File-level changes:**
- New: `src/validation/vol_adjusted_barriers.py`
- New: `src/validation/walk_forward_harness.py`
- New: `src/validation/embargo.py`
- New: `src/validation/leakage_detector.py`
- New: `src/validation/adversarial_validator.py`
- New: `src/validation/__init__.py` exposes a single `validate_model(model, df, ...) -> ValidationReport`
- Edit `src/engine/train_*.py` so every trainer ends with a `validate_model()` call before saving the joblib

#### S0-2 — Model-architecture audit (#40), ~7 days
A **head-to-head bake-off** of the existing models against simpler baselines, run through the S0-1 validation harness:

- **For 1s–15m horizon prediction:** TFT vs LightGBM vs CatBoost vs XGBoost (with calibration). Score by walk-forward Sharpe + AUC + drawdown + calibration ECE.
- **For execution path / sizing / timing (the OFT-RL piece):** DRL vs Dijkstra vs Bellman-Ford vs A* on the same simulated execution problem. Score by realized slippage on held-out test windows.
- **Hierarchy refactor proposal.** Document a sequential pipeline: regime classifier (HistGBT) → forecast (the winner of the bake-off) → execution optimizer (the winner of the routing bake-off). No more parallel voting between models.

**File-level changes:**
- New: `src/audit/model_bakeoff.py` — runs the head-to-head, writes report to `data/audit/`
- New: `src/audit/path_optimizer_bakeoff.py` — same for routing
- New: `data/audit/sprint0_model_bakeoff.md` — output report
- Edit `src/engine/strategy_registry.py` to add a `model_status: 'live' | 'shadow' | 'killed'` flag per strategy

#### S0-3 — Automated kill-switch (#46), ~3 days
Wire automatic triggers (not just the manual UI button from #30):

- **Daily loss > 3R** (R = avg daily realized vol of the strategy).
- **N consecutive losing trades** (default N = 5).
- **Latency > threshold** (default p99 latency > 500 ms over a rolling 5-min window).
- **Drawdown > X %** (default 8 % of equity from peak).
- **Calibration drift alarm** (Brier score > rolling baseline + 2σ).

When any trigger fires: status='paused', orders flat, ops/Telegram alert, dashboard shows reason. Resume requires explicit operator action.

**File-level changes:**
- New: `src/risk/kill_switch.py` — central evaluator, polled by main loop
- Edit `src/main.py` (or wherever the trade loop lives) to consult `kill_switch.is_paused()` before every order
- Edit `src/dashboard/app.py` to expose `/api/risk/kill_switch/status` + UI tile

#### S0-4 — Execution-quality dashboard (#41), ~3 days
Add a dashboard tile + API endpoint surfacing per-strategy:

- Latency p50 / p99 (from `pre_order_decision` to `order_acknowledged`)
- Veto rate (% of would-be entries blocked by InstitutionalGate / LLM veto)
- Execution success % (`filled` / `submitted`)
- Slippage realized vs predicted (delta in bps), bucketed per exchange
- Gas saved via Flashbots / private routes (if/when DEX path lands)

**File-level changes:**
- New: `src/risk/execution_quality_metrics.py` — rolling metric collector
- New: `/api/execution/quality` endpoint in `src/dashboard/app.py`
- New: dashboard tile in `src/dashboard/templates/index.html`

#### S0-5 — Probability calibration audit (#43), ~1 day
Already 60 % done — `CalibratedClassifierCV` is in `src/engine/agents/training_agent.py:177`. The audit:

- Walk every model that issues a probability used by the veto layer; verify the calibration call is wired.
- For each, compute and write `data/calibration/<model>_calibration.json` containing reliability-diagram bins + Brier score.
- Add dashboard tile rendering reliability diagrams; promote any model with ECE > 0.05 to "needs recalibration."

**File-level changes:**
- New: `src/audit/calibration_audit.py`
- New: `/api/audit/calibration` endpoint
- Edit `src/dashboard/templates/index.html`: add reliability-diagram tile

#### S0-6 — MVP discipline pass (#39), ~1 day (output of all above)
After S0-1 through S0-5 complete, write `data/audit/sprint0_cut_list.md`:

- **Keep** (live, validated, profitable on walk-forward, calibrated): list of strategies + models.
- **Shadow** (unclear edge; run paper-only for 30 more days): list.
- **Kill** (failed validation, leakage, adversarial-AUC fail, or just unprofitable): list.

The Keep list is the **MVP** that all subsequent distribution sprints (Telegram, Stripe, mobile, marketplace) will sell. Anything Killed is ripped out of `strategy_registry.py` so it can't accidentally come back.

### Sprint 0 success criteria

- Every live strategy has a documented walk-forward Sharpe + Sortino + max DD + ECE.
- Every strategy passes the leakage detector + adversarial validator.
- Every model issuing a probability is calibrated (ECE < 0.05).
- The kill-switch fires correctly in synthetic stress tests (daily-loss / consecutive-loss / latency / drawdown).
- The execution-quality dashboard reports non-zero values for every live strategy.
- The cut list is published.

### Sprint 0 failure mode

If S0-1 / S0-2 reveals widespread leakage or adversarial-AUC > 0.6 across most models, all further roadmap work pauses. Recovery options:
- (a) drop to a smaller cleaner feature set + retrain
- (b) shrink to a single "best of bake-off" strategy and ship that as the MVP
- (c) re-frame as a research tool, not a live trader, until edge is rebuilt

This is uncomfortable but cheaper than discovering it post-launch.

---

## 12. Analytic phase (post-Sprint 0)

After Sprint 0 publishes the cut list, the **analytic phase** begins: systematically remove every "bad" piece of the codebase + dashboard + strategy registry that didn't make the cut, leaving only what's making money. Outputs of the analytic phase:

- **Strategy registry** trimmed to the Keep + Shadow lists only.
- **Models** that failed validation deleted from `models/` (or moved to `models/archive/` for reference).
- **Dashboard panels** for killed strategies removed from the UI.
- **Trainers** for killed models removed from the orchestrator's PLAN_ORDER.
- **Tests** updated — assertions about removed code removed; new assertions about the cut list added.
- **Docs** updated — `APP_DOCUMENTATION.md`, `RUNBOOK.md`, README updated to reflect the new shape.

The analytic phase is its own sprint (Sprint 0.5, ~1 week) and produces a leaner, more defensible codebase that maps 1:1 to the validated edge.

---

*File: COMPETITIVE_ASSESSMENT_2026-05-10_v2.md — supersedes v1 by extending it with 7-pass analysis + two external technical reviews + Sprint 0 validate-before-distribute pre-flight. v1 remains valid for its original 11 items; this doc adds 37 more (27 from re-run + 10 from external reviews) and ranks all 48. Companion implementation plan: `TECH_IMPLEMENTATION_PLAN_2026-05-10.md`.*
