# PLAN — 2026-05-08 outstanding work (priority-ordered)

Captures everything left from today's session and the closest-value follow-ups,
ordered strictly by priority. Each item is self-contained: scope, files,
effort, deps, acceptance criteria.

Priority bands:
- **P0** — close today's loose ends (must finish before any new work)
- **P1** — fixes broken functionality (TFT trainer, model accuracy review)
- **P2** — operator quality-of-life (per-model backtest, data backfill)
- **P3** — performance / longer refactors
- **P4** — long-term initiatives (separate roadmap)

---

## P0 — Loose ends from this session

### P0.1 · Retrain `scalping` 🟥
- **Why**: meta dated `2026-04-29`. Today's session retrained 5 of 8 model
  families but never finished scalping (4.5 M-sample trainer takes ~30 min and
  got pre-empted by zombies / restarts).
- **Scope**: trigger `/api/training/run/scalping` once with `force=true`,
  watch the row's elapsed/ETA, confirm `models/scalping_model_meta.json`
  mtime updates and `walk_forward_mean_acc` is real.
- **Files**: none (operational).
- **Effort**: 30 min wall clock, mostly waiting.
- **Deps**: dashboard up; one CPU lane free.
- **Accept**: meta mtime is today AND `walk_forward_mean_acc > 50%` AND no
  `data/training_jobs.json` entry left in `error` / `lost` for `model='scalping'`.

### P0.2 · Retrain `meta_labeler` post-import-fix 🟥
- **Why**: meta dated `2026-04-29`. PR-30 (`36eb047`) fixed the
  `_compute_regime_features` import path but the trainer was never
  re-triggered after the fix, so we don't actually know if it runs end-to-end.
- **Scope**: trigger `/api/training/run/meta` with `force=true` and `tf=1h`.
  If a NEW error surfaces (the trainer hasn't been exercised in 10+ days),
  capture from `logs/train/<job_id>.log` and patch.
- **Files**: possibly `src/engine/train_meta_labeler.py` for the next bug.
- **Effort**: 15 min if import fix was sufficient; +1–2 h if a new bug surfaces.
- **Deps**: P0.1 ideally finishes first to free the CPU lane.
- **Accept**: meta mtime is today AND `auc_roc` and `win_precision` populated.

### P0.3 · OFT subprocess hygiene 🟥
- **Why**: 4 zombie OFT subprocesses (PIDs 27860 / 31380 / 46280 / 67404, ages
  43–78 min when last sampled) running unsupervised since before PR-40 added
  the persistence layer. PR-41's orphan sweep made them visible, but if any
  finishes it'll write `models/oft_model_meta.json` and `models/oft_model.pt`
  unconditionally — **last writer wins** with no tracking of which was the
  canonical run.
- **Scope**:
  1. Kill all four with `Stop-Process` (or let them finish if you prefer the
     "first to win" outcome).
  2. Delete `models/oft_model.pt` if it exists, so a clean retrain doesn't
     blend epochs from different runs.
  3. Trigger fresh `/api/training/run/oft` (resource_kind=exclusive — will
     wait for any other GPU/exclusive job).
- **Files**: none (operational).
- **Effort**: 5 min cleanup + 140 min wait for a new OFT to finish.
- **Deps**: P1.1 (TFT fix) done first so TFT and OFT don't both retrain at
  peak GPU load.
- **Accept**: exactly one `oft_model.pt` with mtime today, no zombie OFT
  subprocesses, fresh metrics in `oft_model_meta.json`.

---

## P1 — Fixes broken functionality

### P1.1 · TFT `cannot reindex on an axis with duplicate labels` 🟧
- **Why**: my one-line dedupe in `build_series_bundle` (commit `8545c54`) was
  insufficient — TFT still errors with the same pandas message. Duplicates
  appear AFTER `engineer_frame()` adds features, OR `darts.TimeSeries.from_dataframe(fill_missing_dates=True)`
  generates its own duplicates via `asfreq(freq)` when the source index has
  irregular gaps.
- **Scope**:
  1. Run `train_tft_model(dry_run=True)` against one symbol to isolate the
     offending dataframe.
  2. Print `df['timestamp'].duplicated().sum()` BEFORE and AFTER
     `engineer_frame()` to localize.
  3. If duplicates emerge from `engineer_frame()`, check
     `add_taker_and_trade_features` and `add_ofi` for joins that fan out rows.
  4. If from the asfreq path, set `freq` based on the actual median bar
     interval in the source data (vs hard-coded `'1h' / '1min'`).
  5. Add `df.drop_duplicates('timestamp', keep='last')` immediately before
     each of the THREE `from_dataframe()` calls inside `build_series_bundle`
     (target, past_covariates, future_covariates).
- **Files**: `src/engine/train_tft_model.py`.
- **Effort**: 1–2 h.
- **Deps**: GPU lane free.
- **Accept**: TFT job goes `done`; `tft_model_meta.json` mtime is today; no
  `cannot reindex` error in any future run.

### P1.2 · TFT regression test 🟧
- **Why**: once P1.1 fixes the dedupe, lock the fix with a synthetic test that
  catches the duplicate-labels failure in CI without GPU.
- **Scope**: synthetic-data unit test that:
  1. Builds a small dataframe with duplicate timestamps and irregular gaps
     (mimicking the legacy/new market_data UNION).
  2. Calls `build_series_bundle(df, freq='1h')`.
  3. Asserts no `ValueError`.
- **Files**: `tests/test_tft_dedupe.py` (new) or extend `tests/test_dashboard.py`
  Phase 70.
- **Effort**: 1 h.
- **Deps**: P1.1 must be solved first.
- **Accept**: pytest passes; test would fail on the pre-P1.1 code path.

### P1.3 · Post-retrain accuracy audit 🟧
- **Why**: today's regime/base/trend/futures retrains finished, but
  preview screenshots showed several rows with **WF Acc% in 50.4–52.2%**.
  That's barely above coin-flip — features don't carry signal, or labels
  are too noisy, or both. Need to know which.
- **Scope**: read each retrained meta JSON (`base_*`, `trend_*`, `futures_*`,
  `regime_classifier`, `scalping`, `meta_labeler`, `tft`, `oft`),
  extract `walk_forward_mean_acc`, `walk_forward_std_acc`, `n_samples`,
  `n_features`, write a 1-page report.
- **Files**: read-only; output to
  `data/audit_reports/post_retrain_2026_05_08.md`.
- **Effort**: 30 min.
- **Deps**: P0.1, P0.2, P0.3, P1.1 ideally complete so we have full picture.
- **Accept**: report exists, flags every model with WF acc <51% as
  "needs feature/label review".

---

## P2 — Operator quality of life

### P2.1 · Verify 1s archive coverage for 6 stale symbols 🟨
- **Why**: original symptom was BNB / DOGE / ETH / LINK / TRX / XRP at
  5 m / 15 m / 4 h / 1 d / 1 w / 1 mo timeframes ending **2024-12-31**
  (16-month gap). The resample today filled higher TFs FROM the 1 s archive —
  but if the 1 s archive itself has the gap, the resample silently produced
  gap-aware output. Need to confirm.
- **Scope**:
  1. For each of the 6 symbols:
     `gunzip -c data/raw/historical/<sym>_USDT_spot_1s.csv.gz | tail -1`.
  2. If `<sym>_spot_1s` ends before 2026-01, the archive has the gap.
  3. Document findings.
- **Files**: none (read-only check); output to
  `data/audit_reports/data_coverage_check_2026_05_08.md`.
- **Effort**: 30 min.
- **Deps**: none.
- **Accept**: report tells us whether each of the 6 symbols actually has the
  gap or whether the resample just didn't run for them.

### P2.2 · Per-model filter for `run_full_backtest` 🟨
- **Why**: PR-42 added "refresh stats after train" toggle, but
  `run_full_backtest(timeframes=(tf,))` re-runs every strategy at that TF —
  no per-model filter. Clicking "Train trend @ 4h + refresh stats" runs the
  backtest for all 9 ML strategies + 18 pure-rule strategies at 4h. ~5–10 min
  when goal was to refresh just one row.
- **Scope**:
  1. Add `models: tuple[str, ...] | None = None` parameter to
     `run_full_backtest` in `src/engine/backtester.py`.
  2. Skip strategies whose underlying model isn't in the filter (look at
     `model_key` attribute or `strategy_to_model` mapping in
     `src/engine/strategy_registry.py`).
  3. `_spawn_followup_backtest` in `src/dashboard/app.py` passes
     `models=(model_key,)`.
  4. Pure-rule strategies always skipped in this code path — they don't
     change when a model retrains.
- **Files**: `src/engine/backtester.py`, `src/engine/strategy_registry.py`
  (read-only), `src/dashboard/app.py` (call-site update),
  `tests/test_dashboard.py` (Phase 70 regression).
- **Effort**: 2 h.
- **Deps**: none.
- **Accept**: clicking Train+refresh on `trend @ 4h` triggers a backtest
  finishing in <2 min, only updating `data/backtest/wf_results.json` rows
  with `model='trend' AND timeframe='4h'`.

### P2.3 · Refill missing 1 s archive (if P2.1 confirms gap) 🟨
- **Why**: if P2.1 confirms the 6 symbols have a 16-month gap in the 1 s
  archive, no resample / retrain on this data is meaningful for those
  symbols × those TFs.
- **Scope**:
  ```
  python -m src.data_ingestion.binance_archive_downloader \
      --symbols BNB,DOGE,ETH,LINK,TRX,XRP \
      --start 2024-12-31 --end <today>
  ```
  After it finishes, re-run `/api/data/resample` for the same six
  symbols to backfill the higher TFs.
- **Files**: none (operational).
- **Effort**: 30 min trigger + 4–8 h download (network-bound).
- **Deps**: P2.1 confirms gap is real.
- **Accept**: `/api/data/coverage` shows zero `stale` entries with
  `lag_s > 7*86400` for those six symbols.

### P2.4 · "Backfill missing data" button 🟨
- **Why**: when the operator sees a gap on the dashboard, today they have to
  manually run two separate steps (archive top-up + resample). Should be one
  click.
- **Scope**: new endpoint `/api/data/backfill` that chains
  `binance_archive_downloader` → `data/resample` for the symbols/TFs marked
  stale. UI button in the Data Coverage card.
- **Files**: `src/dashboard/app.py`, `src/dashboard/templates/index.html`,
  `tests/test_dashboard.py` (Phase 72).
- **Effort**: 4 h.
- **Deps**: P2.3 validates the underlying tools work.
- **Accept**: one click kicks off both phases as scheduler jobs; operator
  sees them complete without manual follow-up.

---

## P3 — Performance / longer refactors

### P3.1 · Persistent disk cache for cold-start speed 🟦
- **Why**: I quoted ~50–100 MB of disk cache that would skip 30 s of dashboard
  cold-start work (parquet file index, db row counts, service probes,
  training history). Useful but not urgent — current cold start is ~5–10 s
  already.
- **Scope**:
  1. New `src/dashboard/cold_cache.py` module.
  2. `save_snapshot(name, value)` writes `data/cache/cold/<name>.json`
     atomically.
  3. `load_snapshot(name, max_age_s)` reads + age-checks.
  4. Wire into `_db_status_cache`, `_monitor_services_cache`,
     `_dl_status_cache`, `_data_coverage_cache`, and `_TYPICAL_DURATIONS`
     (so ETA self-calibration survives restart).
  5. Cap each cache file at 5 MB; total <50 MB.
- **Files**: `src/dashboard/cold_cache.py` (new), `src/dashboard/app.py`,
  `tests/test_dashboard.py` (Phase 71).
- **Effort**: 3 h.
- **Deps**: none.
- **Accept**: dashboard cold-start serves `/api/db/status` in <100 ms on
  first hit (vs ~20 s today).

### P3.2 · Process separation — heavy reads to FastAPI :8100 🟦
- **Why**: most dashboard endpoints are now <100 ms (PR-35 / PR-37 / PR-39
  fixes), but `/api/db/market_stats` and other ParquetClient queries still
  occasionally take seconds. Long-term right design: Flask UI delegates
  ParquetClient reads to the FastAPI control plane (already running on :8100),
  so a runaway query can't take down the UI.
- **Scope**: bigger refactor — separate PR plan needed before starting.
- **Files**: TBD.
- **Effort**: 2–3 days.
- **Deps**: validate P3.1 alone isn't enough first (it might be).
- **Accept**: TBD.

---

## P4 — Long-term initiative (separate roadmap)

Per memory `project_institutional_upgrade.md` and `INSTITUTIONAL_UPGRADE_PLAN.md`:
- 5-level upgrade to quant hedge fund quality
- 2-PC GPU cluster
- 1-second tick data ingestion
- Full dashboard rework
- 18-point spec, multi-week effort

Out of scope for this plan. Listed here so the next agent doesn't re-plan it.

---

## Recommended PR grouping

| PR  | Items | Type | Time |
|-----|-------|------|------|
| 43  | P0.1 + P0.2 + P0.3 | Operational (no code) | ~3.5 h wall clock |
| 44  | P1.1 + P1.2 | Code + test | 2–3 h |
| 45  | P1.3 | Audit report | 30 min |
| 46  | P2.1 | Audit report | 30 min |
| 47  | P2.2 | Code + test | 2 h |
| 48  | P2.3 | Operational | 4–8 h network |
| 49  | P2.4 | Code + test | 4 h |
| 50  | P3.1 | Code + test | 3 h |
| 51+ | P3.2 | Bigger refactor | 2–3 days |

## Total effort to clear P0 + P1 + P2

Sum of code/test work: ~9 hours.
Wall clock (with training waits): ~1 day with parallelism, ~2 days serial.
Network-bound (P2.3): up to 8 hours additional, runs in background.
