# Post-Production Tuning Roadmap

**Created:** 2026-05-20  
**Status:** FUTURE — do not start until VPS migration (PLAN_VPS_CLEAN_SLATE.md) is fully complete and live  
**Trigger:** Begin this roadmap only after champion/challenger baseline is established and bot is live on VPS

---

## Context

After the VPS clean-slate migration + retrain, the system will have:
- 7+ model types (Base RF, Trend RF, Futures Short, Scalping, Meta-labeler, Regime GMM, TFT, OFT)
- Multiple timeframes per model
- Strategy control layer (pure rules + ML-driven + filters + routers)
- GARCH sizing, SMA-200 macro filter, Regime router, Meta-labeler
- Champion/challenger system with canary deployment
- Training dashboard with ETA, health, WF accuracy, bull/bear metrics

The bottleneck at that point shifts from "add more models" to **signal coordination, capital allocation, and strategy orchestration**.

---

## Target Trading Style

**Medium-frequency crypto quant system** — this is the correct classification for this architecture.

| Parameter | Target |
|-----------|--------|
| Holding time | Minutes → hours (occasionally days) |
| Trades per day | 10–100 |
| Edge source | Regime adaptation, volatility expansion, funding anomalies, liquidation rebounds |
| Leverage | 1x–2x effective |
| Max risk/trade | 0.5% of bankroll |

**What this system is NOT:**
- Not HFT (no nanosecond latency, no FPGA, no colocation)
- Not a market maker (no inside-spread capture, no inventory management)
- Not a wick-catcher (too execution-sensitive; retail API = last in queue)
- Not ultra-short-latency arbitrage

**Why this matters:** ML accuracy matters less than execution quality and capital allocation when trading frequency increases. At 10–100 trades/day, `positive expectancy after fees` is the only metric that matters — not win rate. Slippage and spread become enemy #1 at any scale below $50k bankroll.

---

## Main Directive: Stop Expanding, Start Orchestrating

```
1. Do NOT add new models
2. Promote 3 core strategies
3. Demote everything else to filters or experimental
4. Build Master Allocator
5. Add Strategy Score per strategy
6. Enable/disable strategies based on regime
7. Collect execution audit + live P&L attribution per strategy
```

---

## Recommended Strategy Taxonomy

### Core (receive capital allocation)

| Strategy | Timeframes | Notes |
|----------|-----------|-------|
| Trend Momentum RF | 1h / 4h | Primary directional signal |
| Volatility Breakout | 15m / 1h | Regime: trending or high-vol |
| Mean Reversion | 5m / 15m | Only in chop/sideways regime |

### Filters (gate/scale core signals, no independent capital)

| Filter | Role |
|--------|------|
| Meta-labeler | Binary confidence filter on core signals |
| Regime classifier (GMM) | Routes capital to matching core strategy |
| GARCH sizing | Dynamic position sizing based on realized vol |
| SMA-200 macro filter | Blocks mean reversion in strong trends |
| Liquidity filter | Blocks execution in thin markets |
| Correlation gate | Caps exposure in correlated clusters |

### Experimental (paper trade only, no live capital)

| Model | Reason |
|-------|--------|
| TFT | Novel architecture, needs live canary validation |
| OFT | Fine-tuned from TFT, same canary requirements |
| Extra Base RF timeframes | Redundant with core RF after regime routing |

---

## Capital-Level Architecture

Optimal allocation changes significantly with bankroll. Two target states:

### $1,000 Bankroll — Survive + Compound

**Goal:** avoid fee/slippage death, maximize signal quality over signal quantity.

| Bucket | Allocation | Strategy | Timeframes |
|--------|-----------|----------|-----------|
| Core directional | 60% | Trend Momentum RF | 1h / 4h |
| Volatility | 20% | Volatility Expansion / Breakout | 15m / 1h |
| Regime-gated | 10% | Mean Reversion | 5m / 15m (chop only) |
| Cash reserve | 10% | Idle | — |

**Symbols:** BTC + ETH + SOL only. Over-diversification at $1k = noise, not diversification.

**Risk settings:**
- Max risk/trade: 0.25–0.5% ($1.25–$2.50)
- Effective leverage: 1x–2x
- Max simultaneous positions: 3–4
- Max daily drawdown: 2% → kill-switch

**What to skip at $1k:** full cross-exchange arbitrage (capital fragmentation + fees kill it), market making, 20+ symbols.

---

### $10,000 Bankroll — Small Systematic Desk

**Goal:** introduce carry yield, portfolio allocation becomes serious, multi-venue intelligence.

| Bucket | Allocation | Strategy | Notes |
|--------|-----------|----------|-------|
| Core directional | 40% | Trend Momentum RF | 1h / 4h |
| Volatility | 20% | Volatility Expansion / Breakout | 15m / 1h |
| Carry yield | 15% | Funding/Basis Carry | spot long + perp short, delta-neutral |
| Regime-gated | 10% | Mean Reversion | chop only |
| Event-driven | 10% | Post-liquidation Reversal | crypto-native edge |
| Experimental | 5% | TFT / OFT / new alpha | paper → small live |

**New capabilities unlocked at $10k:**
- Funding carry (spot long + perp short) becomes fee-efficient
- Multi-venue intelligence: treat Binance vs Bybit spread as a signal (not execution arb)
- Cross-market lead/lag: BTC futures impulse → alt lag → signal
- Statistical spread models: ETH/BTC spread, SOL/ETH relative momentum

**Still NOT viable at $10k:** latency arbitrage, DEX↔CEX arb, flash wick catching, market making.

---

## Master Allocator + Strategy Decay Monitor

**The most important single improvement.**

The system must automatically understand:
```
Trend currently working       → increase capital allocation
Mean reversion breaking down  → reduce / pause allocation
Scalping expensive on slippage → cut allocation
Regime = chop                 → suppress trend signal
```

### Strategy Score (per strategy, updated hourly)

Composite score from:
- Rolling 7d live Sharpe (Tier 1, weight 40%)
- Sharpe deviation from backtest baseline (Tier 1, weight 30%)
- Win rate trend (5d EMA of win rate, weight 15%)
- Slippage cost trend (weight 15%)

Score range: 0–100. Allocation weight proportional to score.

### Decay Monitor

Detect strategy degradation before it hits drawdown limits:
- Rolling 3d Sharpe < 0.5 AND declining → "Degrading" state
- Rolling 3d Sharpe < 0 → "Suspended" state (paper only)
- Recovery: 5d Sharpe > 1.0 after suspension → manual re-enable review

### Capital Allocation Logic

```python
# Pseudocode for Master Allocator
total_capital = account_balance
active_strategies = [s for s in strategies if s.score > 30 and regime.allows(s)]
weights = softmax([s.score for s in active_strategies])
for s, w in zip(active_strategies, weights):
    s.capital_limit = total_capital * w * (1 - correlation_penalty(s))
```

---

## Advanced Alpha Roadmap

Funding rate windows create microstructure dislocations — especially in BTC, ETH, SOL. This system already has the required stack (ML, regime, volatility, execution gating, funding data). The roadmap is 4 levels:

### Level 1 — Funding Blackout Windows ✅ (already in Phase 10)

```
07:58–08:02 UTC
15:58–16:02 UTC
23:58–00:02 UTC
```
Block new entries, widen slippage assumption 2×, allow exits. This is already implemented in `PLAN_VPS_CLEAN_SLATE.md` Phase 10. No further action needed — just confirm it stays enabled.

### Level 2 — Funding Anomaly as Meta/Regime Feature (build after baseline v1)

Add to meta model and regime layer:
- `funding_percentile` — where current funding sits vs 30d history
- `funding_z_score` — standardized deviation from rolling mean
- `oi_delta_near_funding` — OI change in the 30 min before settlement

These become **filter features**, not a standalone strategy. They improve confidence on existing signals (suppress trend entry when funding is extreme and exhaustion is likely).

### Level 3 — Post-Funding Reversal Alpha (experimental bucket, small allocation)

Logic:
```
extreme funding (positive or negative)
+ OI spike near settlement
+ price vertical move into settlement
→ probability of mean reversion ↑ after settlement
```

This is a **meta alpha feature first** — feed it into the meta-labeler as an additional signal. Only promote to standalone small allocation after ≥90 days of paper data shows statistical edge (Sharpe > 1.0).

This does NOT require second-level timing or HFT execution — the reversal plays out over minutes to an hour.

### Level 4 — Full Funding Carry Engine (only at $10k+, after stable infra)

Delta-neutral: `spot long + perp short`, collecting funding yield.

**Do NOT build until:**
- [ ] Bankroll ≥ $10,000 (at $500 the yield is eaten by fees)
- [ ] VPS infra stable ≥ 6 months with no critical incidents
- [ ] Risk engine (kill-switch, liquidity filter, safe mode) battle-tested
- [ ] Position reconciliation (Phase 10 WS reconnect) fully validated

**Why it's good but not now:** Jump Trading, Wintermute, Alameda all run this. It's structurally sound. But at $500–$1k: fees + funding changes + hedge sync complexity produce negative expectancy. At $10k it becomes viable.

---

## Live P&L Attribution

Currently impossible to answer: "which strategy made money this week?"

Required: tag every filled order with `strategy_id` in execution_audit.jsonl. Dashboard card: P&L per strategy per day/week/month.

Without attribution, the Master Allocator has no signal to act on.

---

## Technical Debt to Address Before This Roadmap

These CRITICAL items from the 6-agent review of v10 must be resolved first:
- DuckDB `PRAGMA` → `SET` syntax fix
- DuckDB singleton connection pattern
- datetime64[ns] → datetime64[us]
- Per-column hash streaming (not full load)
- ntpdate → chronyc makestep
- OOS run_id isolation
- Single unified PreTradeGate.check()
- VPS hardening (ufw, fail2ban, SSH)

See `PLAN_VPS_CLEAN_SLATE.md` Phase 0–11 for the full list.

---

## What NOT to Build

Explicit exclusion list — these are tempting but wrong for this system at current scale:

| Strategy | Why NOT |
|----------|---------|
| HFT spread capture / inside-spread scalping | Competing against Jane Street, Jump Trading, Wintermute with FPGA + colocation. Retail API = last in queue. |
| Flash wick / vacuum liquidity catching | Fill rarely happens; when it does, reversal often doesn't. Too execution-sensitive. |
| Market making | Requires inventory management, queue priority, adverse selection model. Not viable without VIP tier + colocation. |
| DEX MEV competition | Requires mempool access, custom nodes, gas optimization. Different engineering domain entirely. |
| Ultra-low latency arbitrage | Latency between Tokyo VPS and exchange matching engine is already a structural disadvantage vs colocated competitors. |
| 100+ symbols | Over-diversification at this bankroll = noise. Stick to top 10 liquid majors. |
| 50x leverage | One bad trade = account gone. Structural risk incompatible with compound growth goal. |
| "One big bet" moonshot approach | Antithetical to the system's design. This is a statistical edge machine, not a gambler. |
| Constant model expansion | Adding models without fixing orchestration = diminishing returns. Master Allocator > 8th model type. |

**The honest answer:** biggest gains after the baseline is live will NOT come from another model, another indicator, or another transformer. They will come from execution quality, capital allocation, regime adaptation, risk management, and avoiding catastrophic losses.

---

## Baseline Trading Analysis (2026-04-25 → 2026-05-17)

Source: `data/trades.json` (1350 closed), `models/_baseline_2026-05-16/*_meta.json`, `src/engine/strategy_registry.py`

### All ML Models in System

| Model | Algorithm | Artifact | Canonical TF | WF Acc | AUC-ROC | Win Rate | Samples | Role |
|-------|-----------|----------|-------------|--------|---------|---------|---------|------|
| Base RF | HistGBT + Calibrated | `btc_rf_model.joblib` | 1h | 52.2% | 0.504 | 49.5% | 707k | Directional buy |
| Trend RF | HistGBT + Calibrated | `trend_model.joblib` | 4h | 50.4% | 0.505 | 35.0% | 789k | Trend-follower |
| Futures Short RF | HistGBT + Calibrated | `futures_short_model.joblib` | 1h | 51.9% | 0.513 | 49.6% | 533k | Short-side |
| Scalping RF | HistGBT + SMOTE | `scalping_model.joblib` | 1m | 51.6% | 0.536 | 49.8% | 7.1M | Sub-min mean-rev |
| Meta-Labeler | HistGBT + Calibrated | `meta_labeler.joblib` | 1h | 57.9% | **0.641** | 35.1% | 582k | TP-probability gate |
| Regime GMM | Bayesian GMM | `regime_classifier.joblib` | 1h | 45%+ | — | — | 817k | bull/bear/chop router |
| TFT Neural | Darts TFT | `tft_model.pt` | 1h | — | — | — | 61k | Multi-horizon forecast |
| OFT | Order-Flow Transformer | `oft_model.pt` | 1m | — | — | — | — | Microstructure µ/σ |

Per-TF variants also trained: `base_15m`, `base_4h`, `trend_15m`, `trend_4h`, `futures_15m`, `futures_4h`, `meta_15m`, `meta_4h`, `scalping_1m`, `tft_1h`.

---

### Strategy × Model × Market P&L (all historical trades)

**Period:** 2026-04-25 → 2026-05-17 (22 days, testnet) | **Total P&L: −1,112 USDT** | **Win rate: 32.3%** | Avg win: +$0.41 / Avg loss: −$1.57

| Strategy (log name) | Model | Market | TF | Trades | P&L (USDT) | Win Rate | P&L % vol | Status now |
|---------------------|-------|--------|----|--------|-----------|---------|----------|-----------|
| Scalping_Short | `scalping_model` | SCALPING | 1m | 280 | **−912** | 23% | −5.92% | ❌ disabled |
| Scalping_Long | `scalping_model` | SPOT_SCALPING | 1m | 631 | **−124** | 36% | −0.36% | ❌ disabled |
| Scalping_Long | `scalping_model` | SCALPING | 1m | 197 | −54 | 26% | −0.50% | ❌ disabled |
| Ichimoku_Cloud | none (rule) | SPOT | rule | 53 | −17 | 30% | −0.57% | ❌ disabled |
| MACD_Divergence | none (rule) | SPOT | rule | 17 | −6 | 12% | −0.63% | ❌ disabled |
| Elliott_Wave_Correction | `btc_rf_model` | FUTURES | 1h | 16 | −4 | 38% | −0.49% | ✅ live |
| VWAP_Reversion | none (rule) | SPOT | rule | 7 | −2 | 29% | −0.64% | ❌ disabled |
| Volatility_Breakout | none (rule) | SPOT | rule | 3 | −2 | 33% | −1.00% | ❌ disabled |
| Elliott_Wave_Impulse | `btc_rf_model` | SPOT | 1h | 1 | −1 | 0% | −0.99% | ✅ live |
| Keltner_Breakout | none (rule) | SPOT | rule | 1 | −1 | 0% | −0.97% | ❌ disabled |
| Supertrend | none (rule) | SPOT | rule | 2 | ≈0 | 50% | −0.01% | ❌ disabled |
| ML_Trend_Following | `trend_model` | SPOT | 4h | 31 | **+1** | 29% | +0.05% | ✅ live |
| Donchian_Breakout | none (rule) | SPOT | rule | 2 | +1 | 50% | +0.80% | ❌ disabled |
| OU_Entry | none (rule) | SPOT | rule | 48 | **+4** | **71%** | +0.14% | ❌ disabled |
| MACD_Momentum | none (rule) | SPOT | rule | 61 | **+6** | 28% | +0.17% | ❌ disabled |

---

### Key Observations from This Data

**1. The ML system has never actually traded.**
Not a single trade from Base_ML, Trend_ML (as primary signal), TFT, OFT, or Meta-Labeler appears in the log. All 1350 trades are legacy rule-based or Scalping_ML (now disabled). The ML architecture is unproven in live conditions — AUC and WF accuracy are offline metrics only.

**2. Scalping_ML destroyed 98% of losses (−$1,090 of −$1,112).**
Root cause: avg_loss / avg_win = 3.8× at 23–36% win rate = guaranteed negative EV. The model may predict short-term moves correctly, but fees + spread + slippage eliminate the edge entirely. 1m scalping on retail API with no colocation is structurally a death zone. Move to paper/experimental permanently.

**3. `model_confidence` = NULL on all 1350 trades.**
The field exists in `trades_enriched.json` but was never populated. Meta-Labeler was supposedly live but left no trace in any trade record. Impossible to know retrospectively whether it filtered anything, helped, or was bypassed.

**4. `timeframe` field missing from trade log.**
No `timeframe` column in `trades.json` or `trades_enriched.json`. TF mapping in table above was reconstructed from `strategy_registry.py`. The new `execution_audit.jsonl` (Phase 9) must include `timeframe` as a required field.

**5. Rule-based strategies outperformed ML in this period.**
OU_Entry (71% win rate, rule-based), MACD_Momentum (+$6, rule-based), Donchian (+$1) all beat every ML combination. This does not mean ML is worse — it means ML has not yet been properly connected to the trading loop. Crypto structurally rewards simple robust logic; complex models need more careful integration.

**6. Per-model assessment:**
- **Base RF** (AUC 0.504): effectively random. Tiny edge possibly hidden by noisy Triple-Barrier labels. Unproven.
- **Trend RF** (live P&L slightly positive, 31 trades): strongest current ML candidate. Trend-following is structurally compatible with crypto.
- **Scalping RF**: remove from core capital permanently. Paper/execution research only.
- **Meta-Labeler** (AUC 0.641 — only meaningful AUC in entire system): potentially very important, but was never instrumented. First priority for proper telemetry.
- **Regime GMM**: likely the strongest architectural component. Non-stationarity in crypto is severe; routing matters more than model accuracy.
- **TFT / OFT**: unproven research alpha. Not production engine. Keep in experimental bucket only.

---

## Priorities for Live Alpha Validation

These must be completed BEFORE building the Master Allocator (no allocation signal without trade data).

### Priority 1 — Proper Telemetry on Every Trade (blocking everything else)

Every entry in `execution_audit.jsonl` must include:
```json
{
  "strategy":        "Trend_ML",
  "model":           "trend_model_4h",
  "timeframe":       "4h",
  "regime":          "trending",
  "model_confidence": 0.73,
  "meta_passed":     true,
  "expected_ev":     0.0042,
  "slippage_pct":    0.00018,
  "spread_pct":      0.00031,
  "exit_reason":     "TP"
}
```
Without this, the interaction matrix (Priority 4) is impossible to compute. `model_confidence = NULL` killed all retrospective analysis of this period.

### Priority 2 — Scalping to Paper/Sandbox Only

`Scalping_ML` must never receive real capital again until it demonstrates positive EV after fees in a dedicated paper-only canary with at least 500 trades and 30 days of data. Current evidence: −$1,090 on 1108 trades = structural loser at this infra level.

### Priority 3 — First Live Combo to Validate

The highest-priority combination to test with real capital (small, 5% allocation):
```
Trend RF (4h)
  + Meta-Labeler filter (confidence threshold 0.54+)
  + Regime Router (TRENDING regime only)
  + GARCH sizing
  + Funding blackout windows
```
This is the strongest candidate from available evidence. Run as champion canary for ≥30 days before evaluating.

### Priority 4 — Interaction Matrix

Before building the Master Allocator, map which combinations actually produce Sharpe > 0:

| Strategy | Meta filter | Regime gate | Sharpe (live) | Notes |
|----------|-------------|-------------|--------------|-------|
| Trend RF | None | Any | ? | Baseline |
| Trend RF | Meta | Any | ? | Does meta help? |
| Trend RF | Meta | TRENDING only | ? | Core hypothesis |
| Trend RF | Meta | RANGING (blocked) | ? | Expected: 0 trades |
| Base RF | None | Any | ? | Baseline |
| Base RF | Meta | TRENDING | ? | |
| OU_Entry | None | RANGING only | ? | Rule-based already positive |

Each cell requires ≥30 trades to be statistically interpretable. At $500 bankroll this takes weeks — start immediately after VPS baseline.

### Priority 5 — Portfolio Allocator

Only build after Priority 1–4 produce at least 3 cells with Sharpe > 0 from the interaction matrix. The Allocator needs real performance signals to weight — without them it just shuffles capital randomly.

---

## One-Time Security Tasks (complete before first live trade)

These were deferred from the VPS migration plan and must be done before real capital is deployed:

| Task | Action | Where |
|------|--------|--------|
| Binance API key — IP whitelist | In Binance web UI: API Management → edit key → restrict to VPS IP `5.104.81.27`. Disable Withdrawals + Universal Transfer. | binance.com (manual, browser only) |

---

## Execution Gate

**Do not start this roadmap until:**
- [ ] VPS migration complete (all 11 phases done)
- [ ] First clean retrain complete + baseline v1 established
- [ ] Bot live on VPS with real canary data (≥14 days)
- [ ] execution_audit.jsonl has ≥100 live trades with strategy_id tagged
- [ ] P&L attribution working for at least 3 strategies
