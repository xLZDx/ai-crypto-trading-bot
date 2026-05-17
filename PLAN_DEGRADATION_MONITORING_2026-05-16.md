# Degradation Monitoring — Implementation Plan
**Date:** 2026-05-16  
**Status:** Reviewed — pending execution  
**Reviewed by:** ML-engineer agent + architect agent

---

## Overview

Five independent improvements to the model health system, ordered by delivery priority.

| ID | Name | Files changed | Order |
|----|------|---------------|-------|
| P2 | Two-tier drift enforcement | `drift_psi.py`, `drift_monitor.py`, `.env.example` | 1st — independent |
| P5 | Per-strategy regression guard | `auto_retrain.py` | 2nd — independent |
| P3+P4 | Overfitting ratio + per-fold scores | 5 trainers, `kpi_gate.py`, `training_rules.json` | 3rd — share trainer edits, one PR |
| P1 | Live performance monitor | new `live_performance_monitor.py`, `main.py`, dashboard | 4th — largest, ships last |

---

## P2 — Two-tier drift enforcement

### Problem
`LLM_DRIFT_PAUSE=warn` (default) means PSI ≥ 0.25 on `ofi_z` logs a WARNING but does not halt a single trade. The infrastructure exists; it is switched off.

Flipping globally to `enforce` is wrong — `atr_14` drift (volatility expansion) would halt trading as aggressively as `ofi_z` drift (orderbook regime change).

### Design
Two tiers inside `DRIFT_HARD_FEATURES`:

- **Enforce-tier** (`DRIFT_ENFORCE_FEATURES` env var): halt immediately on first hourly poll with PSI ≥ 0.25. Default: `ofi_z,funding_z,macd_hist,frac_diff_d40`
- **Confirm-tier** (remaining `DRIFT_HARD_FEATURES`): halt only after **3 consecutive** hourly polls with PSI ≥ 0.25. Single-spike false positives do not halt.

Architect confirmed: `safe_json` atomic rename pattern + single-writer thread means adding a consecutive counter to `CellState` is race-safe.

### File changes

**`src/risk/drift_psi.py`**
- Add `_enforce_features() -> frozenset[str]`: reads `DRIFT_ENFORCE_FEATURES` env var, intersects with `DRIFT_HARD_FEATURES`. Falls back to empty set if unset (preserve current `warn` behaviour).
- `check_drift()`: pass the enforce-tier set to `DriftFinding` as `is_enforce_feature: bool`.

**`src/risk/drift_monitor.py`**
- `CellState`: add `consecutive_pause_count: int = 0`.
- `_run_one_cell()`: read previous cell state from cached JSON before running. If new `pause_count > 0`, increment `consecutive_pause_count`; if clean, reset to 0.
- `is_drift_paused()`: return `True` when (a) any enforce-tier feature is in `pause` severity (immediate), OR (b) `consecutive_pause_count >= 3` for confirm-tier features. Currently returns on first pause finding — gate now distinguishes tier.

**`.env.example`**
- Document `DRIFT_ENFORCE_FEATURES=ofi_z,funding_z,macd_hist,frac_diff_d40`
- Document that `LLM_DRIFT_PAUSE=enforce` must also be set; without it `is_drift_paused()` returns False regardless.

### Tests
- Unit: `check_drift()` with a feature in enforce-tier vs confirm-tier at PSI ≥ 0.25 — assert `is_enforce_feature` flag set correctly.
- Unit: `is_drift_paused()` with `consecutive_pause_count = 2` → returns False; `= 3` → returns True.

---

## P5 — Per-strategy regression guard

### Problem
`auto_retrain.py:_avg(before)` averages WF Sharpe across all strategies. A strong improvement in one strategy can mask a regression in another. Single-strategy regression is invisible until it compounds.

### Design
After computing `before` and `after` dicts: run per-strategy comparison in addition to the system-wide check. A strategy that regresses individually fails the retrain verdict even if the system-wide average holds.

New strategies in `after` but not `before` are excluded from the check (no baseline) and logged as `"new_strategies"` — they do not fail the verdict. Architect confirmed this is correct.

### File changes

**`src/engine/auto_retrain.py`**
- Extract per-strategy regression check into `_per_strategy_regressions(before, after, tolerance) -> list[str]`: returns list of strategy names that individually dropped below `old × (1 - tolerance)`.
- In `run_auto_retrain()`: call this after computing `before`/`after`. If any strategy regresses individually, verdict = `"regression"` regardless of `_avg()` comparison.
- Add `"per_strategy_before"`, `"per_strategy_after"`, `"per_strategy_delta"`, `"regressed_strategies"`, `"new_strategies"` to the output dict and regression report JSON.

### Tests
- Unit: `_per_strategy_regressions()` where one strategy drops 10% and another improves 20% — assert the regressions list contains the failing strategy, verdict = `"regression"`.
- Unit: new strategy in `after` not in `before` — assert excluded from list, no failure.

---

## P3 + P4 — Overfitting ratio + per-fold WF scores

Delivered together because both touch the WF fold loop in all 5 trainers.

### P3 — Overfitting ratio

#### Problem
In-sample accuracy is not saved. `overfit_ratio = (in_sample_acc - wf_acc) / in_sample_acc` is never computed. A model with Train=0.72 / WF=0.51 looks fine on the KPI gate (wf_acc > 50%) but is memorising noise.

#### Design
In each trainer's WF fold loop, after fitting on the training fold, call `base_clf.score(X_train_fold, y_train_fold)` to get in-sample accuracy for that fold.

**Architect confirmed:** `.score()` is called on the raw `HistGradientBoostingClassifier` used inside the WF loop — calibration (`CalibratedClassifierCV`) only wraps the **final** model after the loop completes (`train_model.py:399`). No `.base_estimator` workaround needed. Verify the same pattern in each of the other 4 trainers before implementing.

Thresholds (from ML-engineer agent):
- `overfit_ratio > 0.10` → WARNING at training time
- `overfit_ratio > 0.20` → ERROR at training time + `kpi_gate` retirement eligible
- Rationale: HistGBT on financial time-series legitimately shows 5–12% gap from purge effects. 10% is the meaningful signal threshold; 20% means the model is memorising regime structure.

#### File changes

**All 5 trainers** (`train_model.py`, `train_futures_model.py`, `train_trend_model.py`, `train_scalping_model.py`, `train_meta_labeler.py`):
- In WF fold loop: collect `in_sample_fold_accs: list[float]` alongside existing `fold_accuracies`.
- After loop: `in_sample_mean_acc = mean(in_sample_fold_accs)`, `overfit_ratio = (in_sample_mean_acc - wf_mean_acc) / in_sample_mean_acc`.
- Log WARNING/ERROR per threshold.
- Save `in_sample_mean_acc` and `overfit_ratio` to meta.json.

**`src/engine/kpi_gate.py`**
- `TrainingResult`: add `overfit_ratio: float | None = None`.
- `_check_thresholds()`: handle `overfit_ratio` as a MAX field (same pattern as `wf_max_dd` at line 302): fail if `overfit_ratio > floor`.
- `evaluate_from_meta_json()`: read `overfit_ratio` from meta JSON.

**`data/training_rules.json`**
- Add `"overfit_ratio": 0.20` to `kpi_threshold` for each model.

### P4 — Per-fold WF scores + slope gate

#### Problem
WF results are aggregated (mean ± std). Individual fold performance over time is hidden. A model with fold scores `[0.58, 0.57, 0.55, 0.52, 0.48]` has mean 0.54 — passes the 50% gate — but is clearly degrading across folds. (AFML Chapter 7: the time-ordered fold sequence is the signal; aggregate statistics hide it.)

#### Design
Save per-fold scores to meta.json as `wf_fold_scores: [f1, f2, ..., fn]` in time order.

Add slope check to `kpi_gate._check_thresholds()`: if `wf_fold_scores` is present and `len >= 3`, compute linear slope via `numpy.polyfit`. Slope threshold (architect correction): `slope / mean_acc < -0.02` (relative, not absolute) AND `wf_fold_scores[-1] < wf_fold_scores[0]` (last fold worse than first — sanity gate against noisy mid-series dip). If both conditions met → add `"wf_fold_slope:negative"` to missed fields list, counting toward the 3-strike retirement.

#### File changes

**All 5 trainers**:
- In WF fold loop: collect existing per-fold accuracy into `wf_fold_accs: list[float]` (architect confirmed `train_model.py:360-372` already collects `fold_accuracies` — only the save step is missing; verify in the other 4 trainers).
- Save `"wf_fold_scores": wf_fold_accs` to meta.json.

**`src/engine/kpi_gate.py`**
- `TrainingResult`: add `wf_fold_scores: list[float] | None = None`.
- `_check_thresholds()`: add slope check block after existing field loop.
- `evaluate_from_meta_json()`: read `wf_fold_scores` from meta JSON.

### Tests (P3 + P4 together)
- Unit trainer test: mock 3-fold WF; assert `overfit_ratio` and `wf_fold_scores` present in meta dict with correct values.
- Unit `_check_thresholds()`: `overfit_ratio=0.25` with threshold `0.20` → assert `"overfit_ratio"` in missed fields.
- Unit `_check_thresholds()`: `wf_fold_scores=[0.58, 0.54, 0.50, 0.46]` → assert `"wf_fold_slope:negative"` in missed fields.
- Unit `_check_thresholds()`: `wf_fold_scores=[0.52, 0.54, 0.50, 0.53]` (noisy, last > first) → assert slope flag NOT set.
- Unit: 3-strike scenario where overfit_ratio triggers retirement after 3 consecutive runs.

---

## P1 — Live performance monitor

Delivered last because (a) it is the largest item and (b) the overfit/slope gates (P3+P4) should be in place as upstream defenses before live halts are introduced.

### Problem
The system detects that input features look different from training (feature drift), but cannot detect that the model's predictions are now wrong against actual price outcomes. No rolling live prediction accuracy exists; concept drift is only visible to the operator via trading logs.

### Design

**`src/risk/live_performance_monitor.py`** — new file

`record_signal(model_key, tf, predicted_direction, entry_bar_ts, entry_close, atr_at_entry, pt_mult, sl_mult, max_bars)`:
- Assigns a `signal_id` (UUID).
- Stores in `self._pending: dict[str, dict]` (protected by `threading.Lock` — close callbacks arrive on the WebSocket thread, not the main loop thread; architect confirmed a lock or `queue.Queue` is required).

`record_outcome(signal_id, bars_df)`:
- Receives the OHLC bars from `entry_bar_ts` to the barrier horizon.
- Calls `triple_barrier_labels_vectorized()` from `src/analysis/triple_barrier.py` (reuse — do NOT reimplement) on a 1-row-entry-anchored slice, passing the snapshotted `atr_at_entry`, `pt_mult`, `sl_mult`, `max_bars`.
- Labels the prediction correct (predicted direction matches barrier exit sign) or wrong.
- Appends to per-cell rolling deque (7 calendar days, min 30 samples).
- Writes updated state to `data/risk/live_perf_state.json` via `safe_json`.

`is_live_perf_halted(model_key, tf) -> tuple[bool, str]`:
- Reads `live_perf_state.json` (cached, not recomputed per call).
- Returns `(True, reason)` if rolling accuracy < `wf_acc × 0.90` for 2 consecutive evaluation windows.
- `wf_acc` sourced from the model's meta.json at monitor startup and refreshed on hot-reload.

Alert path: CRITICAL log + write `data/risk/live_perf_halt.json` per halted cell.

**`src/main.py`** — integration:
- After `MultiTFPredictor` instantiation: `LivePerfMonitor.start()` (background flush thread).
- At signal generation: `monitor.record_signal(...)` — snapshots entry bar context.
- At position close: `monitor.record_outcome(signal_id, bars_slice)`.
- Before emitting a signal to `OrderManager`: call `is_live_perf_halted(model, tf)` — if True, skip with WARNING log.

**Dashboard** — new endpoint `POST /api/live_perf/reset/<model>/<tf>`:
- Clears the halt flag for a cell without requiring retrain.
- Returns current rolling accuracy and window count.

**State file:** `data/risk/live_perf_state.json`
```json
{
  "cells": {
    "base__1h": {
      "rolling_acc": 0.48,
      "window_count": 2,
      "consecutive_below_threshold": 2,
      "halted": true,
      "halt_reason": "rolling_acc 0.48 < wf_acc 0.61 × 0.90 for 2 windows",
      "sample_count": 74,
      "window_start_iso": "2026-05-09T00:00:00Z"
    }
  }
}
```

### Thread-safety design
- `self._pending` dict: protected by `threading.Lock`. Signal emission (main loop thread) acquires on write; WebSocket close callback acquires on read+delete. No bare list append.
- Alternative: `queue.Queue` from close handler to a dedicated outcome-processor thread — cleaner if the WebSocket callback must be non-blocking.

### Tests
- Unit: `record_signal()` + `record_outcome()` on a synthetic bar slice where the triple-barrier exits at profit-take — assert prediction labeled correct when predicted_direction matches exit sign.
- Unit: 30 correct + 5 wrong predictions → rolling_acc = 30/35 ≈ 0.857 — assert `is_live_perf_halted()` returns False with `wf_acc = 0.61` (threshold = 0.549).
- Unit: inject 2 consecutive windows below threshold → assert halt flag set, CRITICAL logged.
- Unit: `POST /api/live_perf/reset/base/1h` → assert halt cleared, sample history preserved.
- Unit: confirm threading lock prevents race between signal writer and outcome reader.

---

## Shared constraints

- All new state files written via `src/utils/safe_json.write_json` (atomic temp-file rename).
- All new `data/risk/*.json` and `data/retrain_regressions/*.json` excluded from git (already covered by `data/*.tmp` pattern — confirm or add explicit exclusions to `.gitignore`).
- All trainers: in-sample `.score()` called on the raw `base_clf` inside the WF loop, not on the calibrated wrapper. Verify per-trainer before implementing P3.
- `auto_retrain.py` new-strategy handling: strategies appearing in `after` but not `before` → excluded from per-strategy regression check, logged as `"new_strategies"` in report.

---

## Execution order

```
Phase 1 — Independent, small
  P2  drift enforcement (2 files + .env.example)
  P5  per-strategy regression guard (1 file)

Phase 2 — Shared trainer edits (one PR)
  P3  overfitting ratio (5 trainers + kpi_gate + training_rules)
  P4  per-fold scores + slope gate (5 trainers + kpi_gate)

Phase 3 — Largest, ships after Phase 2 defenses are live
  P1  live performance monitor (new file + main.py + dashboard endpoint)
```

## Risk register

| Severity | Item | Risk | Mitigation |
|----------|------|------|------------|
| HIGH | P1 | Triple-barrier signature mismatch — naive close-in-N-bars produces biased labels | `record_signal()` must snapshot `atr_at_entry, pt_mult, sl_mult, max_bars`; `record_outcome()` calls `triple_barrier_labels_vectorized()` from `src/analysis/triple_barrier.py` |
| HIGH | P1 | Thread safety — WebSocket close callbacks fire on different thread than signal emission | `threading.Lock` around `_pending` dict, or `queue.Queue` from close handler |
| HIGH | P3 | `.score()` on wrong object — calling on `CalibratedClassifierCV` instead of raw clf | Confirmed: WF loop uses raw `base_clf` (`train_model.py:360-372`); calibration wraps only the final model. Verify in all 5 trainers before implementing. |
| MEDIUM | P4 | Absolute slope threshold fires on noisy classifiers | Use `slope / mean_acc < -0.02` (relative) + `last_fold < first_fold` sanity gate |
| LOW | P5 | New strategy in `after` not in `before` causes KeyError | Explicit exclusion + log as `"new_strategies"` |
