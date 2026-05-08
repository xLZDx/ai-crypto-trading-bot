# PLAN — 2026-05-08 outstanding work (v3.1 — execution in progress)

This is the v3.1 implementation plan.

**v3 → v3.1 delta** (applied 2026-05-09 mid-execution after step 1
landed):
- **P0.A flipped back to curated** ("applicable based on model logic"
  per operator clarification) instead of strict all×all.
- **New item 1M added** — OFT (Microstructure) sweep coverage, since
  the dashboard surfaces it as `NOT STARTED` and the operator wants
  all 15 currently-shown model rows retrained.

v3 folded in three new dashboard items surfaced during the 2026-05-09
review (mode-aware Portfolio, per-market All-Markets breakdown,
"MAINNET → REAL CASH" rename), re-ordered steps by **priority +
complexity + dependencies**, and put the overnight retrain+backtest
sweep **at the very end** of the implementation work per the operator's
direction.

Status: **everything stopped** since the operator's last "stop all".
Nothing relaunches without your reply on this plan. No in-progress
training will be killed (per `feedback_dont_relaunch_inflight_training`).

---

## §0 · Cross-reference: every original item is in v3.1

Per the lesson learned on 2026-05-08 (P2.2 nearly slipped), every
original item ID is mapped explicitly to its new step. **No item is
dropped.** Four new items were added (1K, 1L, 1M, scope-expansion of 1I).

| Original ID | What it is | New step | Notes |
|---|---|---|---|
| **P0.A** | TF coverage policy | §1.P0.A | **Curated** (v3.1) — "applicable based on model logic" per operator clarification |
| **P0.B** | Multi-TF trading architecture | §1.P0.B | Default unchanged (cross-TF confirmation gate) |
| **P0.C** | Canonical model file policy | §1.P0.C | Default unchanged (canonical + per-TF variants) |
| **1A** (TF map) | Update DEFAULT_PER_KEY_TFS | **Step 2** | — |
| **1B** (P2.1 TFT) | TFT `cannot reindex` deeper fix | **Step 4** | — |
| **1B′** (P2.2) | TFT regression test | **Step 5** | depends on 1B |
| **1C** (P2.X) | Scalping label rebalance | **Step 6** | — |
| **1D** (P1.2 fwd) | Trade-row enrichment fields going-forward | **Step 12** | blocks 1E |
| **1E** (P1.2 back) | Backfill 912 historical trades | **Step 13** | depends on 1D |
| **1F** (P2.5) | Per-model `run_full_backtest` filter | **Step 7** | scope expanded — also accepts per-TF filter |
| **1G** (P2.4) | 1s archive coverage check | **Step 8** | gates 1H |
| **1H** (P2.6) | 1s archive refill (only if gap) | **Step 14** | depends on 1G |
| **1I** (P2.8) | Mode-switch Portfolio panel wiring | **Step 9** | **scope EXPANDED** — see §2.1I |
| **1J** (P2.7) | "Backfill missing data" button | **Step 11** | depends on 1H |
| **2A** | Overnight retrain sweep | **Step 16** | moved to END per user direction |
| **2B** (P2.3) | Post-retrain accuracy audit | **Step 17** | after 2A |
| **2C** | P1.2 backfill validation | **Step 18** | — |
| **3A** | Multi-TF trading (cross-TF gate) | **Step 19** | depends on 2A producing per-TF variants |
| **3B** | 1-week paper-trading validation | **Step 20** | depends on 3A |
| **4A** (P1.1) | Analytical dashboard (7 sections) | **Step 21** | depends on 2A + 1E |
| **5A** (P2.9) | Persistent cold-start disk cache | **Step 15** | parallel-safe |
| **5B** (P2.10) | FastAPI process separation | **Step 22** | depends on 5A |
| **NEW · 1K** | "MAINNET" → "REAL CASH" UI rename | **Step 1** ✅ | done 2026-05-09 (commit 2dd612b) |
| **NEW · 1L** | "All Markets" per-market Signal/Risk panels | **Step 10** | new — fixes screenshot flagged 2026-05-09 |
| **NEW · 1M** | OFT (Microstructure) sweep coverage | **Step 8** | new (v3.1) — covers 15th dashboard row currently `NOT STARTED` |
| **NEW · 1I expanded** | API-driven Balances + kill mode-blind PnL writes | **Step 9** | scope-expansion of existing 1I |

**Total steps**: 22 (was 18 in v2; +1 from 1B′ split, +3 from 1K/1L/1M, 1I scope-expanded in place).

---

## §1 · Decisions taken (operator may override)

### P0.A — TF coverage policy
**Decision (default, v3.1 — corrected per operator clarification 2026-05-09)**: **curated — "all 15 dashboard models trained on every TF *applicable based on model logic*", not naive all×all.**

```python
DEFAULT_PER_KEY_TFS = {
    'base':     ('5m', '15m', '1h', '4h', '1d'),     # 5 TFs — directional signals across intraday→swing
    'trend':    ('15m', '1h', '4h', '1d', '1w'),     # 5 TFs — trend lives at 15m+
    'futures':  ('5m', '15m', '1h', '4h', '1d'),     # 5 TFs — same logic as base
    'scalping': ('1m', '5m'),                         # 2 TFs — sub-minute mean reversion only
    'meta':     ('5m', '15m', '1h', '4h'),           # 4 TFs — gates entry signals (5m–4h)
    'tft':      ('15m', '1h', '4h'),                  # 3 TFs — swing horizons; 1m/5m → noise
    'regime':   ('1h',),                              # 1 TF — features TF-invariant
}
# 7 keys × ~3.6 TFs avg = 25 (model × TF) tabular combos
# + OFT (item 1M): microstructure model on L2/L3 events, single canonical TF (1m)
# Total: ~26 sweep entries; ~6-12 h wall clock @ 10 cores
```

**Why this is the v3.1 default** (corrected from v3's strict all×all):
operator clarified 2026-05-09 — *"all 15 models trained on all
timeframes if applicable based on model logic"*. v3's strict all×all
ignored the "applicable" clause and would have wasted ~6-12 h of
compute on combos that converge to noise (TFT @ 1m, scalping @ 1d) or
that add no information (regime × extra TFs — features are
TF-invariant). This map encodes the per-model logic:

- `tft` only at 15m+ — TFT input_chunk_length=168 → ~3 h history per
  inference; designed for swing horizons, on minute bars it just fits
  noise.
- `scalping` only at 1m/5m — sub-minute mean reversion; on daily bars
  it produces a biased trend follower with bad accuracy.
- `regime` only at 1h — GMM clusters use TF-invariant features (vol,
  ADX, returns z-score); extra TFs add no information.
- `meta` only at 5m–4h — meta-labeler gates entry signals on those
  TFs; 1d/1w meta has no consumer in the bot loop.
- `base` / `trend` / `futures` — every TF where the directional thesis
  has historical evidence.

Per-trainer `try/except` in `_train_loop` still isolates any
unexpected combo failure, so the sweep can't be brought down by a
single trainer crash.

**Coverage of the 15 currently-shown dashboard rows**: the curated map
covers all 14 OHLCV-bar rows (futures × {1d, 4h, 1h, "short"}, trend ×
{1d, 4h, 1h}, base × {1d, 4h, 1h}, tft × 1h, meta × 1h, regime × 1h,
scalping × 1m) — and adds 11 new variants (e.g. trend @ 15m, base @ 5m,
meta @ 4h). The 15th row (OFT, currently `NOT STARTED`) is covered by
the new item 1M.

The `Futures Short RF @ short` row is a `short`-horizon label variant
(not a TF) produced by the existing futures trainer; it's preserved by
the trainer's own logic — no separate sweep entry needed.

Override:
- `use strict all×all` → 49-combo sweep (v3 default; longer + lower-quality models on the noise combos).
- `drop X / Y from P0.A` → name specific (key, tf) pairs to skip.

### P0.B — Multi-TF trading architecture
**Decision (default, unchanged)**: **cross-TF confirmation gate**.
Single agent at canonical TF; signal must be confirmed by ≥1 higher
TF before entry. Lowest portfolio-risk delta, highest signal quality,
minimal architecture change.

Concrete rules:
- Each existing agent (`FuturesAgent`, `ScalpingAgent`, `SpotAgent`)
  keeps its canonical TF; adds one config line listing **confirmation
  TFs** to check before entry.
- LONG entry on canonical TF only fires if ≥1 confirmation TF says
  LONG (or HOLD — never SHORT).
- SHORT entry mirrors.
- Conflicts (canonical=LONG, all confirms=SHORT) → flat, no trade.
- Capital sizing unchanged.

Override: "use multi-TF agent (option 1)" or "per-TF agent instances
(option 2)".

### P0.C — Canonical model file policy
**Decision (default, unchanged)**: **keep canonical + add per-TF
variants alongside.** Inference path keeps loading
`models/<key>_model.joblib`; new variants land at
`models/<key>_<tf>_*.joblib`. Inference can opt in later.

---

## §2 · Implementation phases (re-ordered by priority + complexity + deps)

The order below is what step 1, 2, 3 … of the execution table consumes.
Numbering of *items* (1A, 1B, …, 1L) is the original taxonomy preserved
for cross-reference; numbering of *steps* (1 … 22) is execution order.

### Phase 1 — Quick fixes & blockers (small, high-priority, do first)

#### 1K. NEW — "MAINNET" → "REAL CASH" UI rename (label only)
- Display text only; backend wire value `'mainnet'` stays everywhere
  (control.json, balance_real.json filename, ccxt config — no
  cascading rename).
- Edits in `src/dashboard/templates/index.html`:
  - Button text on the mode-switcher: `⚡ MAINNET` → `⚡ REAL CASH`
  - Tooltip: "Bot trades on Binance live exchange — REAL CASH at
    risk on every order. Requires explicit confirm."
  - Status mapping: `mainnet: '⚠ REAL CASH — live Binance, real money'`
  - Confirm dialog: "⚠ Switch to REAL CASH (live Binance)?"
- Button id `lt-btn-mainnet`, CSS class `.lt-mode-btn.active.mainnet`,
  control.json key — all stay `mainnet` (zero risk of breaking
  control plane).
- **Files**: `src/dashboard/templates/index.html`,
  `tests/test_dashboard.py` (Phase 71 part 1).
- **Effort**: 15 min
- **Accept**: visible button reads `⚡ REAL CASH`; POST to
  `/api/control/trade_mode` with `{mode:'mainnet'}` still works
  unchanged.

#### 1A. Update `DEFAULT_PER_KEY_TFS` per P0.A (curated, v3.1)
- Replace current map with the 25-combo curated map from §1.P0.A.
- Add comment explaining the curation logic ("applicable based on
  model logic") and the override path (`use strict all×all`).
- **Files**: `src/engine/train_all_models.py`
- **Effort**: 10 min
- **Accept**: dry-run import; no syntax error; `_train_loop` iterates
  the per-key TFs from the curated map.

#### 1B. P2.1 — TFT `cannot reindex on duplicate labels` deeper fix
- `train_tft_model(dry_run=True)` against one symbol; print
  `df['timestamp'].duplicated().sum()` BEFORE and AFTER `engineer_frame()`
  to localise the duplicate source.
- Most likely fix: dedupe immediately before each of the THREE
  `from_dataframe()` calls inside `build_series_bundle` (target,
  past_covariates, future_covariates) — the v1 fix only deduped once
  at the top.
- If `asfreq` path is the source: set `freq` based on actual median
  bar interval rather than hard-coded `'1h' / '1min'`.
- **Files**: `src/engine/train_tft_model.py`
- **Effort**: 1–2 h
- **Accept**: TFT dry-run runs to `done`; `tft_model_meta.json` mtime
  fresh; no `cannot reindex` in any subsequent run.

#### 1B′. P2.2 — TFT regression test (locks the 1B fix)
- Synthetic-data unit test: small dataframe with duplicate timestamps
  and irregular gaps (mimicking the legacy/new market_data UNION),
  call `build_series_bundle(df, freq='1h')`, assert no `ValueError`
  and that the three TimeSeries objects come back with consistent
  indices.
- Test fails on pre-1B code path; passes after 1B.
- **Files**: `tests/test_dashboard.py` (Phase 70 —
  `test_phase70_pr45_tft_dedupe_regression`).
- **Effort**: 1 h
- **Deps**: 1B fixed first.
- **Accept**: pytest passes; reverting 1B's dedupe makes the test fail.

#### 1C. P2.X — Scalping label rebalance
- Inspect current `target_scalp` distribution (~88/12 short/long).
- Add `class_weight='balanced'` to the HistGBT classifier in
  `src/engine/train_scalping_model.py`.
- If still imbalanced after class_weight, layer in `imblearn.SMOTE`
  (synthetic oversampling) on the **training fold only** (test left
  untouched).
- Validate post-train: `long_accuracy ≥ 50 %` AND
  `short_accuracy ≥ 50 %` AND walk-forward acc ≥ baseline.
- Emit `accuracy_warning` field **only when the warning is actually
  warranted post-balance** — currently it's set unconditionally.
- **Files**: `src/engine/train_scalping_model.py`,
  `requirements.txt` (add `imbalanced-learn` if absent).
- **Effort**: 2 h
- **Accept**: scalping meta has long_acc ≥ 50 %, short_acc ≥ 50 %, no
  class-imbalance warning emitted on balanced data.

#### 1F. P2.5 — Per-model + per-TF filter for `run_full_backtest`
- Add `models: tuple[str, ...] | None = None` and
  `timeframes: tuple[str, ...] | None = None` params (v3 expansion —
  also lets user backtest "all models on all TFs" as a single chained
  call from `2A`).
- Skip strategies whose underlying (model, TF) tuple isn't in the
  filter.
- Update `_spawn_followup_backtest` to forward both filters.
- **Files**: `src/engine/backtester.py`,
  `src/engine/strategy_registry.py` (read-only),
  `src/dashboard/app.py`,
  `tests/test_dashboard.py` (Phase 71 part 2).
- **Effort**: 2 h
- **Accept**: train-then-refresh on `trend @ 4h` finishes <2 min, only
  updates trend×4h heatmap rows; `run_full_backtest()` with no
  filters runs the full sweep.

#### 1M. NEW — OFT (Microstructure) sweep coverage
- Currently the dashboard surfaces `OFT (Microstructure) — L2/L3 — NOT
  STARTED` as the 15th model row. The trainer already exists at
  `src/training/joint_oft_rl.py::train_oft(symbol, timeframe, ...)`
  and the dashboard's "Train OFT" button can fire it, but the
  pipeline orchestrator never includes OFT in the all-models sweep.
- Add OFT to `train_all_models.train_all()`:
  - Single canonical TF: `'1m'` (microstructure detail at higher TFs
    averages out — L2/L3 events are sub-minute).
  - Per-symbol loop is already inside `train_oft`, so train_all just
    invokes once per symbol from the universe (or for the 3 canonical
    symbols BTC/SOL/ETH and let the operator extend later).
  - Wrap in the same `try/except` pattern as other trainers so an OFT
    failure can't crash the sweep.
  - Skip-if-fresh resume guard works as long as `models/oft_*_meta.json`
    is what the trainer writes — verify path and add the meta-age
    check.
- Surface on dashboard: ensure `_RESOURCE_KIND['oft'] = 'exclusive'`
  remains (OFT uses GPU+CPU heavily; can't share a lane with TFT).
- **Files**: `src/engine/train_all_models.py`,
  `src/training/joint_oft_rl.py` (read-only — verify entry signature),
  `src/dashboard/app.py` (read-only — verify resource-kind mapping),
  `tests/test_dashboard.py` (Phase 71 part 3 — assert OFT in
  `DEFAULT_PER_KEY_TFS` or in the dispatch table).
- **Effort**: 1.5 h
- **Deps**: 1A landed (so the curated map exists to extend).
- **Accept**: dashboard shows `OFT (Microstructure)` with status
  `OK` and a fresh `last_trained` after the sweep completes (instead
  of `NOT STARTED`); training does not collide with TFT or other
  GPU work via the exclusive resource lane.

#### 1G. P2.4 — 1s archive coverage check (audit only — gates 1H)
- Per symbol: `gunzip -c data/raw/historical/<sym>_USDT_spot_1s.csv.gz | tail -1`.
- For 6 stale symbols (BNB / DOGE / ETH / LINK / TRX / XRP), record
  actual last timestamp.
- Output: `data/audit_reports/data_coverage_check_2026_05_08.md`.
- **Effort**: 30 min
- **Accept**: report identifies which symbols have a refill gap (or
  confirms none do, in which case 1H is dropped).

### Phase 2 — Dashboard "Overall Bot Status — All Markets" rebuild

This phase fixes the 2026-05-09 screenshot: the card titled "All
Markets" was showing single-market BTC/USDT data and mode-blind
balances. Three coupled steps in `src/dashboard/templates/index.html`:

#### 1I. P2.8 — Mode-aware Portfolio loader (scope EXPANDED in v3)
- v2 scope ("call /api/portfolio in ltSetMode") **plus**:
  - Add `loadPortfolioByMode()` — fetches `/api/portfolio?mode=` +
    `_ltCurrentMode`, caches as `_lastPortfolioPayload`.
  - Writes the 7 portfolio fields directly from payload — **no more
    `tradesData` summation in JS**.
  - Renders Balances table dynamically from `payload.balances[]`.
    PAPER → only USDT shown (no testnet 0.999 BTC / 5 SOL / 1885 ADA
    bleed-through).
  - Wire from `ltLoadAll()`, `ltSetMode()` (after POST succeeds), and
    the existing hourly auto-refresh.
  - **Gate the legacy mode-blind paths**: in
    `updateBalancesPanel(state)` and the portfolio-PnL block at
    [index.html:3267-3290](src/dashboard/templates/index.html#L3267-L3290),
    skip writes when `_lastPortfolioPayload && _lastPortfolioPayload.mode === _ltCurrentMode`.
    Falls back to legacy compute only when payload is missing/error.
  - Static `<tr>` rows for BTC/SOL/ADA in `bal-wallet-tbody` move to
    JS rendering so they only appear when a non-paper payload returns
    them.
- Backend `/api/portfolio?mode=<m>` already shipped.
- **Files**: `src/dashboard/templates/index.html`,
  `tests/test_dashboard.py` (Phase 72 — mode-switch wiring).
- **Effort**: 2 h
- **Accept**: clicking PAPER flips Total Capital from $10 K to $100 K,
  USDT row from 10058 to 100000, BTC/SOL/ADA rows disappear; clicking
  TESTNET restores them from the live exchange payload.

#### 1L. NEW — "All Markets" per-market Signal & Risk panels
- The card header reads "Overall Bot Status — All Markets" but the
  Signal and Risk panels render **single-market BTC/USDT SPOT** data
  only (per your 2026-05-09 screenshot). Fix:
- **Signal panel** — replace single HOLD/Sentiment/RSI/Symbol stack
  with **3 stacked rows**, one per market:
  ```
  SPOT     · BTC/USDT  · HOLD  · Sent -0.04 · RSI 81.9
  FUTURES  · ETH/USDT  · SELL  · Sent +0.12 · RSI 28.4
  SCALPING · SOL/USDT  · BUY   · Sent +0.05 · RSI 64.1
  ```
  Data source: `state.market_data.{SPOT|FUTURES|SCALPING}` (already
  returned by `/api/state`); each row reads its market's
  `active_symbol` / `last_signal` / `sentiment` / `rsi`.
- **Risk panel** — replace single Vol/Size/Open with per-market rows:
  ```
  SPOT      · Vol 93.2 % · Size 55 USDT · Open 8
  FUTURES   · Vol 41.5 % · Size 30 USDT · Open 3
  SCALPING  · Vol 12.1 % · Size 12 USDT · Open 2
  ─────────────────────────────────────────────
  TOTAL OPEN POSITIONS                       13
  ```
  Per-market open count =
  `tradesData.filter(t => OPEN && t.market === <M>).length`. Total
  stays as the aggregate so the bot loop's "Open positions" badge
  has a single home.
- **Files**: `src/dashboard/templates/index.html`,
  `tests/test_dashboard.py` (Phase 72 — per-market layout).
- **Effort**: 1.5 h
- **Accept**: panels render 3 rows each in DOM regardless of which
  symbol is selected in the chart; per-market open counts sum to
  total.

### Phase 3 — Trade-row enrichment

#### 1D. P1.2 — Trade-row enrichment fields (going-forward)
- Add fields to every trade write (drives 1L's per-market filtering
  precision and 4A analytics):
  - `mode` — paper / testnet / mainnet (drives 1I strict filter once
    populated; until then 1I shows all closed PnL for non-paper).
  - `regime_at_entry` — TRENDING / RANGING / VOLATILE label at entry
    bar.
  - `model_confidence` — calibrated probability from the meta-labeler.
  - `mfe_pct` / `mae_pct` — max favorable / adverse excursion during
    hold.
  - `slippage_pct` — fill price vs intended price.
  - `exit_reason` — TP / SL / trailing / regime_flip / manual / timeout.
- **Files**: `src/engine/trade_tracker.py`,
  `src/engine/paper_book.py`.
- **Effort**: 4 h
- **Accept**: every NEW trade row has all 7 enrichment fields; old
  rows untouched (1E handles them).

#### 1E. P1.2 — Backfill 912 historical trades (best-effort)
- Read `data/trades.json` (912 rows, all untagged).
- For each:
  - `mode` → assume `'testnet'` (most likely; mark as inferred).
  - `regime_at_entry` → re-run `RegimeClassifier` on the 1h bar at
    entry timestamp.
  - `mfe_pct` / `mae_pct` → load 1m bars between entry and exit;
    compute high/low excursions.
  - `model_confidence` → unrecoverable; set `None`.
  - `slippage_pct` → unrecoverable; set `None`.
  - `exit_reason` → infer from `t.status` + `pnl_pct` sign + timing.
- Write to `data/trades_enriched.json` (don't mutate original).
- **Files**: new `scripts/backfill_trade_enrichment.py`.
- **Effort**: 4 h
- **Deps**: 1D landed first (so backfill schema matches forward schema).
- **Accept**: 912 enriched rows; regime + MFE/MAE populated for >90 %.

### Phase 4 — Optional data refill + refill UX (only if 1G shows gap)

#### 1H. P2.6 — 1s archive refill
- `python -m src.data_ingestion.binance_archive_downloader --symbols <list> --start 2024-12-31 --end <today>`
- Re-run `/api/data/resample` for those symbols.
- **Effort**: 30 min trigger + 4–8 h network-bound.
- **Deps**: 1G confirmed gap (else skip this step entirely).

#### 1J. P2.7 — "Backfill missing data" button on Data Coverage card
- New endpoint `/api/data/backfill` chains archive top-up → resample.
- UI button + progress chip in the Data Coverage card.
- **Files**: `src/dashboard/app.py`,
  `src/dashboard/templates/index.html`,
  `tests/test_dashboard.py` (Phase 73).
- **Effort**: 4 h
- **Deps**: 1H validates the chain (else this step still ships but
  has nothing to refill).

### Phase 5 — Performance / cleanup (parallel-safe — can run alongside 1D-1J)

#### 5A. P2.9 — Persistent disk cache for cold-start speed
- New `src/dashboard/cold_cache.py`.
- Save snapshots of `_db_status_cache`, `_monitor_services_cache`,
  `_dl_status_cache`, `_data_coverage_cache`, `_TYPICAL_DURATIONS`
  rolling-avg map to `data/cache/cold/`.
- Load on dashboard boot — first hit serves cached, background
  refresh keeps it warm.
- ~50 MB total on D:.
- **Effort**: 3 h
- **Accept**: cold-start `/api/db/status` <100 ms (vs ~20 s today).

### Phase 6 — Overnight retrain + multi-TF backtest sweep (the END of impl work)

#### 2A. Overnight all×all sweep (49 combos) + chained multi-TF backtest
- Triggered via `/api/pipeline/run` (scheduler exclusive lane,
  watchdog-protected, skip-if-fresh resume, CPU 10/14).
- 49 (model × TF) combos sequentially per `_train_loop`'s
  per-trainer try/except.
- **Chained backtest**: as soon as the 49-combo training finishes,
  pipeline calls `run_full_backtest()` (now per-TF capable via 1F)
  with no filters → full strategy × symbol × TF heatmap.
- ~12–24 h wall clock (training) + ~1–2 h (backtest).
- **Effort**: 0 (operational; just trigger).
- **Accept**:
  - Up to 49 fresh `<key>_<tf>_meta.json` files (failures isolated).
  - Dashboard shows AUC + WinPrec for every classifier row.
  - Backtest writes new `data/backtest/comparison_<ts>.csv` covering
    every (strategy × symbol × TF).

#### 2B. P2.3 — Post-retrain accuracy audit
- Read every meta JSON.
- Flag models with WF acc <51 % as "needs feature/label review".
- Flag models with `auc_roc <0.55` as "no discrimination".
- Cross-tab per-TF — which TFs work for which model.
- Output: `data/audit_reports/post_retrain_2026_05_08.md`.
- **Effort**: 30 min
- **Deps**: 2A finished.
- **Accept**: 1-page report identifying production-ready vs rework
  combos; recommendations for v4 plan revision.

#### 2C. P1.2 backfill validation
- Confirm `data/trades_enriched.json` has populated regime / MFE / MAE
  for >90 % rows.
- If not, debug the inference path.
- **Effort**: 30 min

### Phase 7 — Post-sweep follow-up (kicks off only after 2A produces results)

#### 3A. Multi-TF trading — cross-TF confirmation gate
- New section in `data/strategy_config.json`:
  ```json
  "FuturesAgent":  {"canonical_tf": "1h", "confirm_tfs": ["4h"]},
  "ScalpingAgent": {"canonical_tf": "1m", "confirm_tfs": ["5m", "15m"]},
  "SpotAgent":     {"canonical_tf": "1h", "confirm_tfs": ["4h", "1d"]}
  ```
- Each agent reads its config + uses the matching per-TF model variant
  for confirmation (variants now exist from 2A).
- **Files**: every `src/engine/agents/*.py`,
  `data/strategy_config.json`,
  `src/engine/inference_engine.py` (load per-TF variant).
- **Effort**: 2 days
- **Deps**: 2A (per-TF model variants must exist).
- **Accept**: agents emit signals with
  `confirmation_status: confirmed/conflicted/single-tf-only`; trades
  only fire on `confirmed`.

#### 3B. 1-week paper-trading validation
- Run for 1 calendar week in paper mode.
- Compare Sharpe / win-rate / max-DD vs single-TF baseline.
- Decision: ship to testnet if multi-TF Sharpe ≥ single-TF Sharpe.
- **Effort**: 1 calendar week wall + 30 min review.

#### 4A. P1.1 — Analytical dashboard (7 sections + decision panel)
- Sections (zero-noise, hourly refresh, no live polling):
  1. Strategy P&L matrix (heatmap strategy × symbol × TF; cell=Sharpe).
  2. Regime conditional performance (same matrix partitioned by regime).
  3. Training history (WF acc / AUC / WinPrec per model×TF over time).
  4. Calibration plots (predicted prob vs realized win rate per model).
  5. Trade lifecycle distribution (entry → MFE/MAE → exit histograms).
  6. Slippage + fee impact (cumulative drag on Sharpe).
  7. Correlation matrix (strategy returns; diversification check).
  8. Decision panel (top): "Today: X strategies underperforming. Y
     models overdue retrain. Z symbols stale data."
- **Files**: new `src/dashboard/templates/analytics.html`,
  new `src/dashboard/analytics_routes.py`,
  new `src/analytics/` package,
  data sources: `data/db/` ParquetClient + `data/trades_enriched.json`,
  nightly aggregation job: new `scripts/analytics_aggregate.py` writes
  `data/analytics_*.parquet` for fast dashboard reads.
- **Effort**: 2 weeks
- **Deps**: 2A + 1E (need fresh metas + enriched trades).

#### 5B. P2.10 — FastAPI process separation
- Move `/api/db/status`, `/api/db/market_stats`,
  `/api/db/training_history` to FastAPI control plane (:8100).
- Flask UI proxies. Runaway DuckDB query can no longer kill the UI.
- **Effort**: 2–3 days
- **Deps**: 5A landed first; only ship 5B if 5A alone isn't enough.

---

## §3 · Step-by-step execution order

| # | Step | Phase | Effort | Wait |
|---|---|---|---|---|
| 1 | **1K — REAL CASH UI rename** (NEW) ✅ done | 1 | 15 min | — |
| 2 | 1A — Update `DEFAULT_PER_KEY_TFS` (curated, v3.1) | 1 | 10 min | — |
| 3 | 1F — Per-model + per-TF backtest filter | 1 | 2 h | — |
| 4 | 1B — TFT dedupe deeper fix + dry-run verify | 1 | 1–2 h | — |
| 5 | **1B′ — TFT regression test (P2.2; locks 1B)** | 1 | 1 h | after 4 |
| 6 | 1C — Scalping label rebalance (class_weight + SMOTE) | 1 | 2 h | — |
| 7 | 1G — 1s archive coverage check (audit only) | 1 | 30 min | — |
| 8 | **1M — OFT (Microstructure) sweep coverage** (NEW v3.1) | 1 | 1.5 h | after 2 |
| 9 | **1I — Mode-aware Portfolio loader (scope expanded)** | 2 | 2 h | — |
| 10 | **1L — Per-market Signal & Risk panels** (NEW) | 2 | 1.5 h | — |
| 11 | 1D — Trade enrichment fields (going-forward) | 3 | 4 h | — |
| 12 | 1E — Backfill 912 historical trades (best-effort) | 3 | 4 h | after 11 |
| 13 | 1H — 1s archive refill (only if 7 confirms gap) | 4 | 30 min trigger + 4–8 h | after 7 |
| 14 | 1J — Backfill button on Data Coverage card | 4 | 4 h | after 13 |
| 15 | 5A — Cold-start disk cache (parallel-safe) | 5 | 3 h | — |
| 16 | **2A — Overnight curated sweep (~26 entries) + chained backtest** | 6 | 0 trigger + 7–13 h wall | after 1-15 ✱ |
| 17 | 2B — Post-retrain accuracy audit | 6 | 30 min | after 16 |
| 18 | 2C — Trade enrichment backfill validation | 6 | 30 min | — |
| 19 | 3A — Multi-TF cross-TF confirmation gate | 7 | 2 days | after 16 |
| 20 | **3B — 1-week paper-trading validation** | 7 | 1 calendar week | after 19 |
| 21 | 4A — Analytical dashboard build (7 sections) | 7 | 2 weeks | after 16 + 12 |
| 22 | 5B — FastAPI process separation | 7 | 2–3 days | after 15 |

✱ Step 16 fires once steps 1-15 are green. Steps 17-22 are post-sweep
follow-up; 20 (1-week wall) and 21 (2-week build) are calendar-bound.

**Total focused dev work for steps 1-15**: ~28-32 h (was ~26-30 h before 1M).
**Realistic calendar (steps 1-16 incl. sweep wall)**: 4-6 days (curated sweep 7-13 h vs v3's 13-26 h all×all).
**End-to-end including 20/21 calendar waits**: 4-5 weeks.

---

## §4 · Acceptance criteria (full v3 sweep)

After all phases complete, the dashboard must show:
1. ✅ **All 15 currently-shown model rows** retrained today: every
   `last_trained` is fresh, AUC + WinPrec populated where applicable.
   - The `OFT (Microstructure) — NOT STARTED` row flips to `OK`.
   - The `Scalping RF (1m) — FAILED` row flips to `OK`.
2. ✅ Per-TF variant rows for every (model × TF) combo in the curated
   map — ~26 rows total (was ~15; +11 from extending `base` / `trend` /
   `futures` / `meta` / `tft` to additional applicable TFs).
3. ✅ Scalping rows: long_acc ≥ 50 %, short_acc ≥ 50 %, no
   class-imbalance warning.
4. ✅ TFT rows: status `OK` at every TF in the curated map (15m / 1h /
   4h); no `cannot reindex` in any logs.
5. ✅ Mode-switch in Performance Overview card: PAPER shows $100 K
   USDT only, no BTC/SOL/ADA bleed; TESTNET shows live exchange
   balance + open positions; REAL CASH (formerly MAINNET) renders
   correctly with cached `balance_real.json` data.
6. ✅ "Overall Bot Status — All Markets" card actually shows
   per-market data: SPOT/FUTURES/SCALPING all visible in Signal
   panel and Risk panel simultaneously.
7. ✅ Mode-switcher button reads `⚡ REAL CASH`, not `⚡ MAINNET`.
8. ✅ Stability Heatmap refreshed with the new TFs.
9. ✅ Bot trading multi-TF in paper mode for 1 week before flipping
   to testnet/REAL CASH.
10. ✅ New Analytical tab loads <1 s, shows 7 decision-support
    sections, no flickering.
11. ✅ Data Coverage shows 0 stale rows for all 20 symbols.
12. ✅ Cold-start dashboard restart serves `/api/db/status` <100 ms.

---

## §5 · What to reply

**To approve as-is**: reply `approved` — I execute steps 1 → 22 in
order. The plan halts before **step 16** (the overnight sweep) for a
"ready to fire?" confirmation; curated sweep is 7-13 h wall clock
(was 13-26 h in v3 strict all×all). Anything blocking on a v4
decision (3A confirmation TFs, 4A dashboard sections, …) I re-confirm
at that step.

**To override defaults**: tell me which one. Common overrides:
- `use strict all×all` → 49-combo sweep (v3 default; longer + lower-quality models on the noise combos).
- `drop X / Y from P0.A` → name specific (key, tf) pairs to skip.
- `use multi-TF agent (option 1)` for P0.B.
- `drop 4A sections X / Y / Z` — listed by number.
- `skip 1J — I'll trigger backfill manually`.
- `do steps 1-10 only this week, defer the rest`.

**To stop**: reply `abort` — everything stays stopped, no code
changes.
