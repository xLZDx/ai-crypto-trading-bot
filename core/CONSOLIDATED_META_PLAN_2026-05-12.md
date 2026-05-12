# Consolidated Meta-Plan — AI Trading Assistance — 2026-05-12

**Status:** DRAFT — pending specialist agent review + operator final approval.
**Supersedes:** `REVISED_PLAN_2026-05-12.md` (stabilization-only), `TECH_IMPLEMENTATION_PLAN_2026-05-10.md` (Sprint 0/0a/0b/0c), `COMPETITIVE_ASSESSMENT_2026-05-10_v2.md` §11/§12 (Sprint 0 spec).
**Authoring inputs:**
1. Codebase audit by code-explorer (2026-05-12)
2. Three specialist reviews (security-reviewer, architect, python-reviewer) on the stabilization plan
3. TECH_IMPLEMENTATION_PLAN_2026-05-10.md (Sprint 0 + 0a/0b/0c + analytic phase)
4. COMPETITIVE_ASSESSMENT_2026-05-10_v2.md (48-item roadmap)
5. Personal-use reframe (kills distribution items)
6. Operator rejection of Telegram (2026-05-12)
7. `unicorn-binance-suite` (https://github.com/oliver-zehentleitner/unicorn-binance-suite) — 5 sub-packages, all MIT, reviewed for replacement value

---

## 0. North Star

**The bot is personal-use only.** Capital preservation + profit for the operator. Single laptop + Ivan worker, no SaaS, no multi-tenant, no Telegram, no public surface.

The entire plan optimizes for:
1. **Trust the numbers.** Every backtest, every "live ready" verdict, every alpha claim must be defensible.
2. **Preserve capital.** Auto kill-switches, position caps, exchange-side stops, reconciliation on restart, multi-exchange spread.
3. **Stay simple.** No over-engineering for a single-operator system. Delete what's not load-bearing.
4. **Don't ship unreviewed code.** Specialist agent review before AND after every non-trivial change (per the new global rule).

---

## 1. Six-layer sequence (Layer 7 demoted to conditional follow-up)

```
LAYER 1   Stabilization (A-G)                          ~10 d  ← blocking everything
LAYER 2   KPI gate + Model Comparison                  ~8 d
LAYER 3   Sprint 0 — Validate Before Trust             ~17 d
LAYER 4   Risk hardening (0a + 0b + 0c)                ~17 d  (much parallel-able)
LAYER 5   Analytic phase §S0.5 — execute cut list      ~5 d
LAYER 6   Personal-use UX surface                      ~6 d
─────────────────────────────────────────────────────────
LAYER 7   External library eval (CONDITIONAL/DEFERRED) skip by default
─────────────────────────────────────────────────────────
TOTAL                                                  ~63 d serial / ~37-42 d parallel
```

Critical path: Layer 1 → Layer 3 → Layer 5. Layers 2/4/6 contain partly-parallelizable work.

**Layer 7 is deferred** per operator decision 2026-05-12. Current `orderbook_collector.py` (top-20 partial streams) is sufficient for personal-use trade sizes (≤ 5% equity ≤ ~$5k). Top-20 covers $200k-$1M liquidity on major pairs — far exceeds bot's trade sizes. See §8 for trigger conditions if Layer 7 is later reactivated.

---

## 2. Layer 1 — Stabilization (A–G phases)

**Source:** `core/REVISED_PLAN_2026-05-12.md`. **Status:** Specialist-reviewed (security + architect + python-reviewer). **Approved in principle 2026-05-12.**

Summary of phase scopes (full details in source doc):

| Phase | Scope | Effort |
|---|---|---|
| **A** | Security hardening — auth on 50+ routes incl. 3 NEW criticals (`/api/scheduler/*`, `/api/cluster/worker_restart`, `/api/cluster/register`); parameterized SQL; ZMQ HMAC; `weights_only=True`; HMAC integrity on 12 model loads (9 joblib + 3 torch); bind services to 127.0.0.1; delete `control_plane.py:8100` | 2 d |
| **C** | Cluster state persistence (`data/orchestrator_state.json`); task dedup; delete `src/agents/` toy; `ParquetStore._duck_lock`; Phase 100b/c finish | 2-3 d |
| **B** | ML correctness — `models/` snapshot, walk-forward PurgedKFold, 70/85/100 splits, HP-from-JSON with 4 guards; retrain all 22 models | 1 d + overnight |
| **D** | Delete `src/engine/trainers/` wrappers (INVERTED from original); delete `train_model_v2.py`; delete `src/tools/binance_archive_downloader.py` dup | 0.5 d |
| **E** | Performance — psutil dedup; parquet size manifest; chat cache; supervisor handle close | 1 d |
| **F** | Test rebuild — `test_safe_json`, `test_purged_kfold`, `test_dashboard_api`, `test_orchestrator`, `test_parquet_store`, `test_model_integrity`; demote string-matches | 2 d (parallel) |
| **G** | Risk-mgmt — module-level `calc_liquidation_price` (with taker fee + accumulated funding + margin_type assertion); `size_from_stop_distance` (notional in quote ccy, type hints); `live_funding.py` (shared ccxt, locked TTL, fail-closed); wire into `futures_agent`; `test_risk.py` | 1 d |

**Exit criteria:** All routes authenticated. No volatile orchestrator state. ML pipeline has no temporal leakage and no calibration/test overlap. Tests are behavioral, not string-match. Risk management has liquidation + stop-distance sizing + live funding gate. 0 regressions across full test suite.

---

## 3. Layer 2 — KPI gate + Model Comparison (Sprint 1a R2 + R3)

**Source:** `core/SPRINT_1A_PER_MODEL_AGENTS_AND_KPI.md`. **Status:** Not yet started. Sprint 1a R1 partially absorbed into Layer 1 (Phase C + D); R2/R3 still pending.

### R2 — KPI gate per training run (~3 d)
Add `kpi_threshold` block to `data/training_rules.json`:
- `walk_forward_sharpe` (min: e.g. 1.0)
- `calmar` (min)
- `max_drawdown_pct` (max)
- `win_rate` (min)
- `expectancy` (min)
- `min_total_trades` (sanity)

Logic: each training run writes a KPI record. **3 consecutive misses → auto-retire model** (status flag in `strategy_registry.py`). Operator can unretire after investigation.

### R3 — Model Comparison dashboard tab (~5 d)
New "Model Comparison" tab in `src/dashboard/templates/index.html`:
- Sortable KPI grid (one row per (model, symbol, tf))
- Drill-down: per-fold Sharpe / drawdown / equity curve
- Action buttons: Promote / Retire / Restore
- Same template reused for Strategy and Combo comparisons

**Exit criteria:** Every training run produces a KPI record. Models failing 3 runs are flagged retired. Dashboard tab renders sortable grid with promote/retire actions.

---

## 4. Layer 3 — Sprint 0 (Validate Before Trust)

**Source:** `TECH_IMPLEMENTATION_PLAN_2026-05-10.md` §0–§S0-6. Six sub-phases.

### §0 — Cross-cutting setup (~1 d)
- New `data/audit/` directory tree
- New `src/audit/` + `src/validation/` packages
- `ValidationReport` dataclass (verdict ∈ {live, shadow, kill})
- `validation_status` field on strategy registry
- Dashboard "Audit" tab placeholder

### §S0-1 — Validation rigor pipeline (~5 d) **— DEPENDS ON LAYER 1 PHASE B**
- `vol_adjusted_barriers.py` — `PT = k₁·σ_t`, `SL = k₂·σ_t` triple-barrier labels
- `walk_forward_harness.py` — 60/14/14 train/val/test, advancing window (uses Layer 1 B1's true walk-forward)
- `leakage_detector.py` — 3 checks: future-bar correlation, rolling-window scan, normalize-on-full-dataset scan
- `adversarial_validator.py` — train_vs_live AUC; ≥0.65 = kill
- `validate_model()` single entry point; trainers gain a final `validate_model()` step that refuses to save if verdict == kill

### §S0-2 — Model bake-off (~7 d)
- Forecast bake-off — TFT vs LightGBM vs CatBoost vs XGBoost across 1m / 5m / 15m horizons (cluster-parallel via Phase 94 infra)
- Path-optimizer bake-off — OFT-RL vs Dijkstra / Bellman-Ford / A* / Single-venue / Round-robin
- Hierarchy proposal — sequential pipeline: regime → forecast → execution

### §S0-3 — Automated kill-switch (~3 d)
- `src/risk/kill_switch.py` — triggers: daily loss > 3R, N consecutive losses, latency p99 > 500ms, drawdown > 8%, calibration Brier z > 2σ
- Wired into trade loop (singleton `get_kill_switch()`)
- `/api/risk/kill_switch/status` + reset endpoint (requires `confirm=true`)
- **Mandatory pass before any live capital.**

### §S0-4 — Execution-quality dashboard (~3 d)
- `src/risk/execution_quality_metrics.py` — rolling per-strategy latency p50/p99, veto rate, exec success %, slippage real-vs-predicted by exchange
- `/api/execution/quality` endpoint
- Dashboard tile in Audit tab

### §S0-5 — Calibration audit (~1 d)
- Scan every model issuing probabilities; compute reliability bins + Brier + ECE
- Models with ECE > 0.05 flagged for recalibration

### §S0-6 — MVP discipline pass / cut list (~1 d)
- `cut_list_builder.py` consumes all reports; applies rubric → live / shadow / kill
- Output: `data/audit/sprint0_cut_list.md`
- Sprint 0 complete when operator signs off on cut list

---

## 5. Layer 4 — Risk hardening (Sprint 0a + 0b + 0c)

**Source:** `TECH_IMPLEMENTATION_PLAN_2026-05-10.md` §S0a / §S0b / §S0c. 0a + 0b can run in parallel; 0c partially depends on 0b.

### §S0a — Market-risk (~5 d)
- **M1** Position caps (per-trade 1%, per-symbol 10%, total 50%, hard ceiling 5%) — `data/risk_caps.json`
- **M2** Strategy correlation monitor (alarm if any pair > 0.7) — Audit tab heatmap
- **M3** Leverage cap enforcer + auto-deleverage (close worst-performing first)
- **M4** Tick circuit breaker — 4σ single-tick freeze, 60s

**Note:** M1/M2/M3/M4 partially overlap with Layer 1 Phase G (G2 stop-distance sizing). Layer 1 G is the *minimum* before any trading; Layer 4 0a is the *full hardened* version with config-reloadable thresholds and dashboard surfaces.

### §S0b — Operational-risk (~7 d)
- **O1** State backups (hourly/daily/weekly/monthly retention to `D:/backups/`)
- **O2** Offsite encrypted snapshots (AES-256-GCM → Backblaze B2; weekly full + daily incrementals; key from `.env`)
- **O3** Restart state reconciler — pulls live positions + open orders from every exchange; refuses trade until operator ACKs mismatches
- **O4** Network outage SAFE MODE — 3 consecutive heartbeat fails → freeze new orders (existing positions kept; **require exchange-side stops first**)
- **O5** UPS / power-out runbook (doc + light `nut` integration, optional)

**Mandatory pass before live capital:** O1, O3, O4. O2 strongly recommended. O5 doc-only.

### §S0c — Counterparty-risk (~5 d)
- **C1** Multi-exchange capital split — Binance + Bybit + OKX routing via `pick_exchange_for_order(...)`; per-exchange health gates routing
- **C2** Auto-withdraw to cold storage — weekly, hardware-wallet whitelist (whitelist edited ONLY via `data/cold_storage_config.json` direct edit, never via dashboard); 2FA at exchange; default `enabled=false`
- **C3** Exchange health monitor — API latency + ack rate + withdrawal queue + news-keyword scan; 3 reds → quarantine
- **C4** Custodian integration — DEFERRED (only relevant >$1M AUM; doc-only)

**Mandatory pass:** C1 + C3. C2 only after operator whitelists. C4 deferred.

---

## 6. Layer 5 — Analytic phase §S0.5 (~5 d)

**Source:** `TECH_IMPLEMENTATION_PLAN_2026-05-10.md` §S0.5. Executes the Sprint 0 cut list.

1. **Strategy registry trim** — `live` → `validation_status='live'`; `shadow` → paper-only; `kill` → DELETE from registry, remove `signal_*` columns, archive joblib to `models/archive/`, log in `data/audit/sprint0_kill_log.md`
2. **Dashboard cleanup** — remove panels for killed strategies; add Live/Shadow/Killed badge
3. **Trainer/orchestrator cleanup** — remove killed entries from `_MASTER_TRAINER_DISPATCH`, `PLAN_ORDER`, `training_rules.json`
4. **Test cleanup** — remove assertions for killed artifacts; add assertion: every `live` strategy has a passing ValidationReport
5. **Documentation update** — `APP_DOCUMENTATION.md`, `RUNBOOK.md`, `README.md`; new `MODEL_AUDIT_2026-05-12.md` summary

---

## 7. Layer 6 — Personal-use UX surface (~6 d)

From COMPETITIVE_ASSESSMENT v2 item list, only those that survive the personal-use reframe AND aren't already in earlier layers.

| # | Item | Effort | Why kept |
|---|---|---|---|
| 30 | Emergency-stop UI button | XS | Kill-switch from §S0-3 needs an operator-facing button |
| 32 | Active-model badge | XS | "Which model is currently making the call?" — cheap visibility |
| 33 | Strategy lineage view | S | "What features + data sources feed this strategy?" — debugging aid |
| 36 | Real-time stress-test panel | M | "What if BTC drops 20% right now?" — risk visibility for operator |
| 37 | P&L attribution by strategy | S | "Last week's $X came from: trend +Y, scalping -Z" — operator wants this |
| 38 | Idle-balance allocator | M | Cash should be on maker ladder, not idle (DEFERRED if complex) |

**Killed by personal-use reframe (DO NOT BUILD):**
- #7 Telegram OUTPUT (rejected 2026-05-12)
- #12 Mobile app (single-operator, single-laptop)
- #13 Multi-tenant (personal use)
- #14 Pricing/Stripe (personal use)
- #15 Public REST API (personal use; localhost-only is enough)
- #16 Tax export (operator handles tax separately)
- #20 Localization (single English operator)
- #21 Onboarding wizard (operator IS the operator)
- #22 Live perf public page (no public)
- #27 Discord / community (no community)
- #28 TradingView webhook (no TradingView signal source)
- #29 MCP server (no AI tool integration needed)
- #31 Multi-account (single account)
- #9 Strategy marketplace (no market)

---

## 8. Layer 7 — External library evaluation (CONDITIONAL / DEFERRED)

**Operator decision 2026-05-12: SKIP by default.**

Rationale (full analysis in conversation transcript 2026-05-12):
- Current `orderbook_collector.py` ([src/data_ingestion/orderbook_collector.py](src/data_ingestion/orderbook_collector.py)) uses Binance partial top-20 streams (`@depth20@100ms`) — stateless consumer, 100ms snapshots, no gap-detection needed because each message is a full top-20 replacement
- Personal-use trade sizes (1-5% equity, max ~$5k on $100k equity) never escape top-20 liquidity on major pairs (top-20 covers $200k-$1M typically)
- §S0-2 path-optimizer bake-off is a *ranking* exercise — relative algorithm ranking doesn't change between top-20 and top-1000 for trade sizes ≤ $5k
- UBLDC's gap-detection sophistication is HFT-specific; not solving a problem we have
- Stateless top-20 consumer = robust under network instability; stateful UBLDC adds failure modes (gap detection, reinit storms)
- Skipping saves ~5 days from critical path

### Layer 7 trigger conditions (only if any of these become true)

1. **§S0-2 path bake-off reveals depth-related ranking changes** — OFT-RL / Dijkstra / A* delta vs top-1000 baseline > 2 bps on realistic-size orders
2. **Operator equity exceeds $1M** — trade sizes start probing beyond top-20 typical liquidity
3. **Bot starts trading low-liquidity pairs** — pairs where top-20 covers < $5k each side
4. **Future microstructure features need deeper book** — OFI > level 20, queue-position dynamics, etc.

### Layer 7 deferred plan (only reactivated on trigger above)

| Sub-package | Decision |
|---|---|
| **UBLDC** + **UBWA** (bundle) | Spike when triggered — 5d eval branch, retire `orderbook_collector.py` only if gap-events observed vs ccxt REST snapshots |
| **UBRA** | Permanently skip — ccxt covers this |
| **UnicornFy** | Permanently skip — our `_parse_depth_event` works |
| **UBTSL** | Permanently skip — Phase G + §S0a M3 cover stop-loss |

---

## 9. Cumulative timeline (with parallelism)

```
Week 1   ─ Layer 1 Phase A (2d) + Phase C (3d)
Week 2   ─ Layer 1 Phase B (1d + overnight) + D (0.5d) + E (1d) + F (parallel 2d) + G (1d)  → STABILIZATION DONE
Week 3   ─ Layer 2 R2 (3d) + R3 (5d) [parallel]; Layer 7 L7-1 + L7-2 read-audit
Week 4   ─ Layer 3 §0 (1d) + §S0-1 (5d)
Week 5   ─ Layer 3 §S0-2 bake-off (cluster, 7d) || Layer 4 §S0a starts (parallel with cluster wait)
Week 6   ─ Layer 3 §S0-3 + §S0-4 + §S0-5 (parallel, 7d) || Layer 4 §S0b (parallel)
Week 7   ─ Layer 3 §S0-6 cut list (1d); Layer 4 §S0c (5d) || Layer 7 L7-3 spike
Week 8   ─ Layer 5 §S0.5 analytic phase (5d) — execute cut list
Week 9   ─ Layer 6 personal-use UX (6d)  ← BOT IS NOW PRODUCTION-READY FOR LIVE CAPITAL
```

**Mandatory-pass gates** (will not progress past these until met):
- After Week 2: stabilization regression suite green
- After Week 4: validation rigor harness operational; deliberate-leakage fixture detected
- After Week 6: kill switch fires under synthetic stress; all 4 0a items wired
- After Week 7: O1 + O3 + O4 mandatory passes; C1 + C3 mandatory passes
- After Week 8: cut list reviewed + signed by operator

---

## 10. Cross-cutting global rules now in force

These are **always-on** during every layer (per `D:\test 2\CLAUDE.md`):

1. **Approval gate** — written plan + double-ask before code execution
2. **No guessing, cite source inline** — every factual claim has a file:line or command output
3. **Agent review mandatory** (NEW 2026-05-12) — specialist reviewer agent(s) BEFORE writing AND AFTER completion of every non-trivial change; routing table in CLAUDE.md
4. **Validate logs before claiming success** — last 50 lines of every relevant log file
5. **Functional tests prove behavior** — no string-match-only assertions
6. **Regression test maintenance** — every change ships with a test; 0 failures gates push
7. **Git lifecycle** — commit before phases; commit+push after completion; todo list in every commit body
8. **D:-drive only** — no C: drive writes
9. **Plan persistence** — every multi-phase plan in `<project>/core/*.md` + memory pointer

---

## 11. Open decisions for operator

Before final approval, operator confirms:

1. **Sprint 1a R1 absorbed into Layer 1.** Original Sprint 1a R1 (5-7 d, per-model agent refactor) overlaps with Layer 1 Phase C + D. We treat R1 as DONE-via-Layer-1; only R2 + R3 remain. Confirm.

2. **Layer 7 — RESOLVED 2026-05-12.** Skip by default; reactivate only if §S0-2 reveals depth-related ranking changes (see §8 trigger conditions).

3. **Idle-balance allocator (Layer 6 #38).** Real edge but moderate effort. Keep in Layer 6 or move to a Layer 8 follow-up?

4. **Restart-script-is-supervisor-of-supervisor.** `restart_all.ps1` is the only thing that respawns `master_agent.py`. Acceptable for personal-use single-laptop? Or add a Windows scheduled task?

5. **Order book replacement (Layer 7) — RESOLVED.** Layer 7 skipped per operator decision; see §8.

6. **Sprint 0 abort criteria.** If §S0-1 leakage detector fires on >50% of strategies, Sprint 0 aborts and we fix the framework leakage first. Confirm we accept this stop-the-line rule.

7. **Cold storage chain (Layer 4 §S0c C2).** TRC20 vs ERC20 — both, or one? Default: skip C2 entirely until operator has a hardware wallet whitelisted.

---

## 12. What I'm asking the specialist reviewers to validate

(Spawning agents in parallel after this doc is written — they review THIS document.)

1. **architect** — Is the 7-layer sequence correct? Any phase dependencies missed? Is Sprint 1a R1 truly absorbed into Layer 1?
2. **security-reviewer** — Layer 4 0c (multi-exchange + cold storage) introduces new attack surface (exchange API keys, withdrawal addresses, B2 keys). Are the safeguards (whitelist via direct file edit, 2FA, fail-closed) sufficient?
3. **python-reviewer** — Layer 2 R2 KPI gate, Layer 3 §S0-1 walk_forward / leakage detector / adversarial validator, Layer 4 0a M4 tick circuit breaker — are these mathematically + Python-idiom sound as scoped?
4. **planner** — Realism check on the 9-week timeline. Critical-path validity. Parallelism claims hold?
5. **code-architect** — Concrete file/module blueprints for each new package (`src/validation/`, `src/audit/`, `src/risk/`, `src/ops/`) — are the boundaries clean? Any module that would grow into a monolith?

---

## 13. After review

Once all five reviewers return, this doc will be updated with their findings. The final version becomes the canonical roadmap, and execution begins with Layer 1 Phase A (already approved in principle as part of `core/REVISED_PLAN_2026-05-12.md`).

---

## 14. Specialist review findings (2026-05-12)

Five specialist agents reviewed §0–§13 in parallel: architect, planner, security-reviewer, python-reviewer, code-architect. **31 mandatory amendments** consolidated below.

### 14a. Architect amendments (5)

A1. **L2 ordering is backwards** — KPI thresholds (`wf_sharpe ≥ 1.0`) get calibrated by L3 §S0-1's rigorous harness, not L1 Phase B's minimal one. **Fix:** L2 R2 ships **collect-only mode** in Week 3; auto-retire activates ONLY after §S0-6 cut list signed.
A2. **Sprint 1a R1 agent layer is SUPERSEDED (not absorbed) by cluster routing.** File-per-model split landed at `src/engine/trainers/__init__.py:25-55`. Per-model SUPERVISED agents (`src/agents/trainers/`) never built and **shouldn't be** — cluster workers ARE the per-model executors. Rewrite §11 Q1.
A3. **L7-3 UBLDC spike = evaluate-only branch.** If UBLDC lands right before §S0-2 path-optimizer bake-off, slippage verdicts are computed on OLD code and immediately stale. Full-switch decision moves to Layer 5 cleanup.
A4. **Add §15 Rollback Playbook** — per-layer recovery procedures. Layer 5 is the deletion phase (highest rollback risk).
A5. **Insert 5-day paper-trade observation gate between L5 and L6** with quantitative pass criterion: PnL signature within 2σ of pre-Sprint-0 baseline.

### 14b. Planner amendments (4)

P1. **Timeline is overoptimistic.** Solo-operator parallelism = "cluster runs while operator codes" only. Realistic with 1.5× buffer: **~95-100 d serial / ~75-85 d with real cluster overlap → 16-20 calendar weeks, not 9 weeks.**
P2. **Top-3 likely to slip:** L1 Phase B retrain (first run will fail; 22 models × new walk-forward × new HP guards), L3 §S0-2 bake-off (realistic 10-14 d not 7 d), L4 §S0c C1 multi-exchange (needs real Bybit + OKX accounts + funded balances).
P3. **Add §16 Live-bot continuity matrix** — 5 mandatory downtime windows:
   - L1 Phase C (orchestrator state schema change) — DOWN
   - L3 §S0-3 kill-switch wiring into `src/main.py` trade loop — DOWN
   - L4 §S0a M1 position-caps gate in order path — DOWN
   - L4 §S0c C1 multi-exchange routing swap — DOWN
   - L5 §S0.5 strategy registry trim — DOWN
   - (L1 Phase B retrain is NOT downtime — bot stays on OLD models; swap atomic at end)
P4. **Add §17 Layer 3 Outcome Decision Gate.** §S0-2 may collapse half of L4 (e.g., TFT loses on 75%+ of cells → retire TFT → L7 OFT-RL UBWA cascade-drops). Explicit fork after L3 §S0-6 before L4 starts.

### 14c. Security amendments (6, 4 mandatory pre-impl)

S1. **CRITICAL — Adversarial news input can redirect ALL capital to one exchange.** [news_scraper.py:14,204](src/data_ingestion/news_scraper.py#L14) uses stdlib `xml.etree` (no XXE/billion-laughs protection); 7 hardcoded RSS URLs with no signature verification. **Fix:** C3 news signals = advisory-only; require hard metric co-trigger (API latency p99 OR ack-rate drop) before quarantine fires. Switch to `defusedxml` (1-line change).
S2. **HIGH — O2 AES-256-GCM spec under-specified.** Must specify: (a) `os.urandom(12)` IV per encryption, (b) `scrypt` KDF for `OFFSITE_BACKUP_KEY`, (c) B2 application key write-only scope (`writeFiles` capability only).
S3. **HIGH — C2 cold-storage whitelist has no integrity check.** Process control ("edit via file") fails against any code execution path. **Fix:** HMAC-sign `data/cold_storage_config.json` with a key held outside bot process (hardware token OR file at separate path bot user can't write). Verify HMAC on every withdrawal run.
S4. **HIGH — Kill-switch reset is just `{"confirm": true}`.** Highest-risk operator action during market crisis. **Fix:** require TOTP in body: `{"confirm": true, "totp": "123456"}`. Use `pyotp` (10 lines).
S5. **MEDIUM — Multi-exchange API keys must be trade-only.** Startup assertion via each exchange's key-info API; refuse to start if any key has `withdraw` permission.
S6. **MEDIUM — defusedxml fix in news_scraper.py** — same as S1.

### 14d. Python-reviewer amendments (1 CRITICAL + 11 HIGH + 5 MEDIUM)

**CRITICAL:**
PY1. **§S0a M4 off-by-one bug.** `deque(maxlen=30)` appending current return BEFORE computing sigma means the extreme bar inflates σ_t and partially masks itself. **Fix:** compute sigma from window state BEFORE appending; append after the comparison.

**HIGH:**
PY2. KPI "3 consecutive misses" — time window unspecified. **Fix:** store `kpi_miss_history: list[{cycle_id, ts, metrics}]`; auto-retire on 3 misses in `max_miss_window_days=90` window OR staleness gate.
PY3. Calmar window ambiguous (rolling 252d vs full history). **Fix:** rolling 252-cal-day max DD inside each walk-forward fold; store `calmar_rolling_252d` + `calmar_full_history`; gate on rolling.
PY4. `vol_window=30` wrong for both 1m and 1d. **Fix:** per-timeframe defaults: `{"1m": 120, "5m": 60, "15m": 48, "1h": 30, "4h": 20, "1d": 20}`.
PY5. Walk-forward `min_folds=4` blocks short-history altcoins. **Fix:** add `strict: bool = True` param; on insufficient history, auto-halve once and WARN before raising.
PY6. Leakage AST walk misses alias imports, `.expanding()`, generator exprs. **Fix:** add runtime wrapper — monkeypatch `pd.Series.rolling` in test fixture, record `closed='left'` enforcement.
PY7. Spearman `|ρ|>0.95` misses cumulative/lagged leakage. **Fix:** add LightGBM R² check per feature; flag if `R² > 0.05` predicting `close.shift(-1)` from single feature.
PY8. M4 freeze spec gap — must NOT block existing-position stop-loss management. **Fix:** separate `is_management_frozen(symbol)` always returns False; `observe_tick()` freeze flag applies only to new order initiation.
PY9. M3 "worst-performing" undefined. **Fix:** `sort_key: Literal["pnl_pct","pnl_abs","distance_to_stop"]="pnl_pct"`.
PY10. M3 death-spiral risk. **Fix:** `_deleveraging: bool` flag OR `cooldown_seconds=30` after first close action.
PY11. O3 order ID matching undefined. **Fix:** two-phase write — `{local_uuid, exchange_id=None, status="submitting"}` then on ACK update with `exchange_id`. Reconciler matches on `exchange_id`. Add 4th category: "unconfirmed submission" (`exchange_id=None` + age > 30s).
PY12. Missing behavioral test for `_lane_accepts` neural routing — only string-match exists at [tests/test_dashboard.py:8712](tests/test_dashboard.py#L8712). Already on Layer 1 Phase F backlog.

**MEDIUM:**
PY13. KPI set missing Sortino — add `walk_forward_sortino` (min) to `kpi_threshold`.
PY14. Auto-retire atomicity — confirm KPI-miss counter + status-flag flip is single `safe_json.atomic_write`.
PY15. Adversarial AUC reason — distinguish "regime shift" from "look-ahead leakage". Add `shift_type: Literal["regime","leakage","unknown"]` field to `AdversarialReport`.
PY16. M4 4σ global threshold too sensitive on altcoins. **Fix:** `sigma_threshold: float | dict[str, float]`; per-symbol override from `data/risk_caps.json`.
PY17. ccxt rate-limit burst on startup (3 exchanges × multiple endpoints). **Fix:** sequential per-exchange; `reconciliation_timeout_seconds=120`; surface timeout warning.

### 14e. Code-architect amendments (5)

CA1. **Move `capital_allocator.py` + `exchange_health.py` from `ops/` → `src/risk/`.** Order-path business logic, not operational housekeeping.
CA2. **Rename `src/ops/` → `src/operational_risk/`.** Mirrors plan's term; parallel with `src/risk/` (market) vs `src/operational_risk/` (system). Generic `ops/` invites future misclassification.
CA3. **Export `ValidationReport` from `src/validation/__init__.py`** alongside `validate_model()`. One import surface.
CA4. **Migrate existing risk surface into `src/risk/`** before/during Layer 4: [src/analysis/risk_manager.py](src/analysis/risk_manager.py) (`HullRiskManager`) and [src/engine/institutional_gate.py](src/engine/institutional_gate.py). Otherwise `src/main.py` imports from two parallel risk namespaces.
CA5. **No sub-packages in `src/risk/` yet.** 8 flat files readable. Trip-wire: split at 12+ modules OR second variant of a concern.

---

## 15. Rollback playbook (per-layer)

| Layer | Rollback procedure |
|---|---|
| L1 (any phase) | `git revert <pre-phase commit hash>`; `restart_all.ps1`; verify regression suite |
| L1 Phase B retrain | Restore from `models/archive/<date>/` snapshot (B0); commit pre-B `_meta.json` files |
| L2 R2 KPI gate | Edit `data/training_rules.json` `kpi_threshold` block; `/api/registry/<key>/restore` to unretire flagged models |
| L3 §S0-1 to §S0-6 | Pre-phase commit; trainers stop calling `validate_model()` if revert; cut list document delete |
| L4 §S0a (caps + monitors) | Config-edit revert in `data/risk_caps.json` (caps are reload-able without restart) |
| L4 §S0b O3 reconciler | RECONCILE_REQUIRED is itself a safety state; SAFE MODE exit if reconciler broken — fall back to operator manual confirmation |
| L4 §S0c C1 multi-exchange | Disable additional exchanges in `data/capital_allocation.json` (set `enabled=false`); router falls back to single-exchange |
| **L5 §S0.5 (DELETION phase — HIGHEST RISK)** | Pre-L5 commit hash MANDATORY. Restore deleted strategies from `models/archive/<date>/` + revert `strategy_registry.py` + revert `_MASTER_TRAINER_DISPATCH` + revert `PLAN_ORDER` + revert `training_rules.json` blocks + revert `tests/test_dashboard.py` assertion deletions |
| L6 (UX) | Template-only; revert by deleting tabs |
| L7 UBLDC | Branch-only until cut list signed; merge decision in Layer 5 |

---

## 16. Live-bot continuity matrix

| Action | Bot state | Why |
|---|---|---|
| L1 Phase A (auth headers + parameterize SQL) | UP | Dashboard requires header refresh; no order-path impact |
| L1 Phase B retrain | UP (on OLD models) | Atomic model swap at end of retrain |
| **L1 Phase C orchestrator state migration** | **DOWN** | Schema change to `data/orchestrator_state.json`; in-flight tasks must complete or be re-queued |
| L1 Phase D trainer cleanup | UP | Cluster dispatch table edited; restart cluster only |
| L1 Phase E performance | UP | Internal optimizations |
| L1 Phase F tests | UP | Test-only |
| L1 Phase G risk-mgmt wiring | UP | Gate added to entry path; no state break |
| L2 R3 dashboard tab | UP | Template-only |
| L2 R2 KPI gate (collect-only) | UP | Observation-only |
| L3 §0 stubs | UP | New empty packages |
| L3 §S0-1 trainer instrumentation | UP | Trainers gain a final validate_model() call |
| L3 §S0-2 bake-off | UP (trainers idle, cluster busy) | Cluster-only |
| **L3 §S0-3 kill-switch wiring into `src/main.py`** | **DOWN** | Trade-loop ingress change |
| L3 §S0-4 exec-quality dashboard | UP | Observation-only |
| L3 §S0-5 calibration audit | UP | Read-only |
| L3 §S0-6 cut list | UP | Document write |
| **L4 §S0a M1 position-caps gate** | **DOWN** | Gate inserted into every order submission |
| L4 §S0a M2/M3/M4 | UP except M3 wire-in | Monitors + breakers |
| L4 §S0b O1/O2 backups | UP | Background |
| L4 §S0b O3 reconciler | UP (RECONCILE_REQUIRED held) | Reconciler runs on next startup |
| L4 §S0b O4/O5 | UP | Heartbeat + doc |
| **L4 §S0c C1 multi-exchange router swap** | **DOWN** | `pick_exchange_for_order()` inserted into every order path |
| L4 §S0c C2/C3 | UP | Cold storage default-disabled; health monitor observation-only |
| **L5 §S0.5 strategy registry trim** | **DOWN** | Strategies deleted; signal columns removed; trainers archived |
| L6 (UX) | UP | Template-only |
| L7 UBLDC | UP | Branch-only |

**5 mandatory downtime windows** flagged in bold. Operator should plan these as deliberate maintenance slots.

---

## 17. Layer 3 outcome decision gate

After §S0-6 cut list lands and BEFORE Layer 4 starts, the operator reviews these forks:

**Fork F-1: §S0-2 forecast bake-off outcome**
- IF TFT loses to LightGBM on >75% of cells → retire TFT; remove from `_MASTER_TRAINER_DISPATCH`; Layer 7 OFT-RL UBWA integration cascade-drops; §S0-2.3 hierarchy proposal becomes 1-line "use LightGBM"
- IF TFT wins majority → keep; proceed to L4 as planned
- IF mixed (no clear winner) → keep both, use regime-conditional selection

**Fork F-2: §S0-2 path-optimizer bake-off outcome**
- IF DRL loses to graph search on >75% of cells → retire OFT-RL; Layer 7 UBWA spike loses value; defer Layer 7 entirely
- IF DRL wins → keep; Layer 7 UBLDC integration still warranted

**Fork F-3: §S0-1 leakage detector outcome**
- IF fires on >50% of strategies → ABORT. Fix framework-level leakage at the feature-builder layer before any Layer 4 work
- IF fires on <50% → fix flagged strategies in §S0.5 analytic phase; proceed

**Fork F-4: §S0-1 adversarial validator outcome**
- IF AUC ≥ 0.65 on most models → extend training window or add regime-conditioned features (NOT auto-retire; per amendment PY15 the `shift_type` field distinguishes regime shift from leakage)
- IF most pass → proceed

**Document forks taken in `data/audit/sprint0_decisions_2026-XX-XX.md`** with: which fork triggered, what was retired, downstream layer impact, sign-off.

---

## 18. Updated timeline (post-review)

```
Cal Week 1-2   ─ Layer 1 (Stabilization)               ~15 d realistic (was 10)
Cal Week 3     ─ Layer 2 R3 only (UI, no auto-retire)  ~5 d realistic (was 8)
Cal Week 4-7   ─ Layer 3 (Sprint 0)                    ~26 d realistic (was 17)
Cal Week 7     ─ Layer 3 outcome decision gate (§17)
Cal Week 8-11  ─ Layer 4 (Risk hardening)              ~26 d realistic (was 17)
Cal Week 12    ─ Layer 5 (Analytic phase)              ~8 d realistic (was 5)
Cal Week 13    ─ 5-day paper-trade observation gate
Cal Week 14-15 ─ Layer 6 (UX) + L2 R2 auto-retire activation  ~9 d realistic (was 6)
   (Layer 7 — SKIPPED per operator decision 2026-05-12)
```

**Realistic total: ~15-19 calendar weeks (3.5-4.5 months).** Solo operator, with global rules (approval gate, double-ask, log validation, agent review before+after) in force.

---

## 19. Final approval — what operator confirms

Reply with one of:
- **"approved, full v2 plan"** → Execute Layer 1 Phase A in next turn with all 31 amendments applied
- **"approved with changes: …"** → tell me which amendments to drop/modify
- **"approve item-by-item"** → I list each of the 31 amendments and you accept/reject one by one

No code will be written until this final approval.
