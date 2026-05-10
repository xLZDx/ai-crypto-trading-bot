# Sprint 1a — Per-Model Agent Refactor + KPI Dashboard

**Date:** 2026-05-10
**Author:** operator + Claude
**Slot:** Between Sprint 0 stabilization and Sprint 1 (Sprint 0 cross-cutting setup is postponed until Sprint 1a foundations land)
**Estimated effort:** ~13–15 days
**Status:** Plan approved 2026-05-10; not yet started

---

## Why this sprint exists

Operator screenshot 2026-05-10 22:55:
- Pipeline orchestrator running, GPUs at 70% — clear ground truth that training is happening.
- Every Model Training row showed "OK" — no row reported RUNNING.
- One CRITICAL banner ("debug died 491s ago") that never got investigated.

Root cause was diagnosed and fixed in Phase 97c (orphan record refresh + canonical-row fallback), but the operator's takeaway was correct:

> *"we have a lot of issues with model training because the logic and execution patterns are all in one file for everything … we need to separate the process … each model gets its own file and its own agent/daemon … responsible only for one part / one model e2e … will add more control, more robust execution without guessing if the pid is zombie or not."*

`src/engine/train_all_models.py` is a 30+ KB monolith that:
- Hosts every model's data load, label, fit, eval, persist
- Forks subprocesses for each model under the dashboard's process tree
- Leaves orphan PIDs that the dashboard has to detect via psutil scans
- Couples failure modes — a TFT bug stalls scalping, a meta crash hides a regime success
- Has no per-model contract for "did this run succeed by KPI standards" — only "did the script exit 0"

The strategic answer: **one file per model, one supervised agent per model, KPI gates per model run, granular comparison dashboard for real decision-making.**

> *"till now we were only playing and we still do not have measurement results, but moving forward we will need to introduce KPI that model should meet to win against other models."*

---

## R1 — File-per-model + agent-per-model

### Goal
Replace the monolith with one self-contained training file per model, each owned by a long-running supervised agent process.

### Files to create
```
src/engine/trainers/
    __init__.py
    train_base.py          # Base RF (1h SPOT)
    train_trend.py         # Trend RF
    train_futures.py       # Futures Short RF
    train_scalping.py      # Scalping RF (1m)
    train_tft.py           # TFT Neural
    train_oft.py           # OFT (Microstructure)
    train_meta.py          # Meta-Labeler
    train_regime.py        # Regime Classifier

src/agents/trainers/
    __init__.py
    base_trainer_agent.py  # Shared base class — handles topic subscribe, KPI emit, lifecycle
    trainer_base_agent.py
    trainer_trend_agent.py
    ... one per model
```

### Each `train_<key>.py` contract
- Pure function `train(timeframe: str, *, force: bool = False) -> TrainingResult`
- Self-contained: data load → labels → CV folds → fit → eval → persist artifact
- Returns a typed `TrainingResult` dataclass (see R2 for fields)
- No global state, no dashboard awareness, no subprocess management
- Importable from tests for unit-level coverage
- Replaces the ad-hoc `_train_loop` / `_train_tft` / `_train_oft` / `_train_meta` / `_train_regime` paths in `train_all_models.py`

### Each `trainer_<key>_agent.py` contract
- Long-running process; one per model; supervised by `master_agent` (existing infra)
- Subscribes to its model's topic on the existing pubsub (`src/pubsub`)
- Listens for `{action: "train", tf: "5m", force: false}` commands
- Calls the corresponding `train_<key>.train(tf=...)`
- Publishes `TrainingResult` back on a result topic
- Owns its own log file, heartbeat, PID — supervised exit/restart by master_agent
- No more orphan-PID detection in the dashboard — the agent itself is the source of truth

### Pipeline orchestrator becomes a thin dispatcher
- Reads the model × TF matrix (existing `data/training_rules.json`)
- For each (model, tf) cell: posts `{action: "train", tf}` to that agent's topic
- Awaits result topic with timeout
- Records outcome + KPI to `data/training_runs/`
- Writes `data/training_current.json` from the dispatch loop (no longer the trainer's responsibility)

### Manual override (`/api/training/run/<key>`)
- Posts directly to the target agent's topic with `{action: "train", tf, force}`
- Same return shape as today
- Bypasses pipeline; `master_agent` still owns the agent's lifecycle

### What goes away
- `src/engine/train_all_models.py` (deleted; CLI shim re-exports for backward compatibility during migration only, then deleted)
- `_detect_orphan_training_subprocesses` and the Phase 97c refresh loop (no orphans to detect when every trainer has a known supervisor)
- The "dashboard launches subprocesses" path inside `app.py` (`_run_trainer_blocking`, `_run_followup_backtest_blocking`) — replaced by topic posts

### Estimated effort: 5–7 days
- D1: extract `train_base.py` + `trainer_base_agent.py` skeleton + base class. Get one model working e2e through the agent path.
- D2–D3: extract remaining 7 trainers using the pattern.
- D4: rewrite `pipeline_orchestrator` as topic dispatcher.
- D5: rewrite `/api/training/run/<key>` as topic post; deprecate dashboard-side scheduler for trainers.
- D6: backwards-compat shim for `train_all_models` CLI; rip out orphan detection.
- D7: tests + buffer.

---

## R2 — KPI gate per model run

### Goal
Every training run emits a KPI blob. Models that miss KPI thresholds 3 consecutive times get auto-retired (disabled in registry until operator re-enables).

### `TrainingResult` dataclass
```python
@dataclass
class TrainingResult:
    model_key: str
    tf: str
    started_at: float
    finished_at: float
    artifact_path: str | None  # None on failure
    n_samples: int
    n_features: int
    # KPI block — same fields for every model so comparison is apples-to-apples
    wf_sharpe: float | None        # walk-forward Sharpe ratio
    wf_calmar: float | None        # walk-forward Calmar (Sharpe / max DD)
    wf_max_dd: float | None        # walk-forward max drawdown (fraction)
    wf_win_rate: float | None      # walk-forward fold win rate
    wf_expectancy: float | None    # mean PnL per trade, net of costs
    wf_total_trades: int | None    # count across folds
    wf_acc: float | None           # walk-forward classification accuracy (existing)
    auc_roc: float | None          # for probabilistic classifiers
    # Failure modes
    error: str | None
    cancelled: bool
```

### `data/training_rules.json` extension — `kpi_threshold` block
```json
{
  "models": {
    "trend": {
      "applicable_tfs": ["1h","4h","1d"],
      "kpi_threshold": {
        "wf_sharpe":     1.0,
        "wf_calmar":     1.5,
        "wf_win_rate":   50.0,
        "wf_expectancy": 0.0,
        "wf_total_trades": 30
      }
    }
  }
}
```

### Auto-retirement loop
- New file `src/engine/kpi_gate.py`
- Runs after every `TrainingResult` write
- Reads last 3 successful runs for (model, tf) from `data/training_runs/`
- If all 3 miss any threshold → set registry flag `kpi_retired=true` for that strategy/model
- Operator can `/api/registry/<key>/restore` to clear the flag

### Persistence
- One Parquet file per (model, tf) at `data/training_runs/<model>__<tf>.parquet`
- Append-only; partitioned by month for fast filtering
- Indexed by `finished_at`

### Estimated effort: 3 days
- D1: TrainingResult dataclass + KPI emit hook in base trainer agent.
- D2: training_rules.json schema update + kpi_gate.py loop.
- D3: registry hook + auto-retire test + restore endpoint.

---

## R3 — Granular analytic dashboard

### Goal
Side-by-side KPI comparison so the operator can decide which models to keep, retire, or promote, based on real measurement instead of vibes.

### New tab: "Model Comparison"
Sortable grid with one row per (model, tf):

| Model | TF | WF Sharpe | Calmar | Max DD | Win Rate | Expectancy | Trades | KPI |
|-------|----|-----------|--------|--------|----------|------------|--------|-----|
| trend | 1h | 1.42 ✓   | 2.10 ✓ | 12.0%  | 53.4%    | $4.30      | 412    | PASS |
| trend | 4h | 0.65 ✗   | 0.91 ✗ | 18.2%  | 48.1%    | -$1.20     | 88     | FAIL |
| futures | 5m | 1.05 ✓ | 1.62 ✓ | 22.1%  | 51.2%    | $0.80      | 1450   | PASS |
| ...   |    |          |        |        |          |            |        |     |

- KPI cell colored by pass/fail vs threshold
- Click any row → drill-down: WF folds, equity curve, per-symbol breakdown, last 5 runs trend
- Promote / Retire buttons in the action column
- Filter by KPI status (pass / fail / retired), model, tf

### Same template applied to:
- "Strategy Comparison" — pure-rule + ML-driven strategies, same KPI fields
- "Combo Comparison" — model+strategy pairs (the meta_filtered combinations)

### Implementation
- New API: `/api/kpi/comparison?bucket=models|strategies|combos`
  - Reads `data/training_runs/*.parquet` (models)
  - Reads `data/backtest/wf_results.json` (strategies)
  - Reads combo registry for combos
  - Returns rows + per-cell pass/fail vs threshold
- New tab in dashboard template; reuses existing sortable-table component
- Drill-down modal reuses existing chart components

### Estimated effort: 5 days
- D1–D2: API endpoint + query layer over training_runs Parquet
- D3: tab UI + sortable grid
- D4: drill-down modal + per-symbol breakdown
- D5: Promote / Retire / Restore buttons + tests

---

## R4 — Sequencing + acceptance gates

### Order
1. **Sprint 0 stabilization** (ongoing — Phase 97c just landed; banner alert dedup is next)
2. **Sprint 1a R1** (file-per-model + agent-per-model) — 5–7 d
3. **Sprint 1a R2** (KPI gate) — 3 d
4. **Sprint 1a R3** (analytic dashboard) — 5 d
5. **Sprint 0 §0 cross-cutting setup** (was originally pre-Sprint-1; now sequenced post-1a)
6. **Sprint 1** original scope continues (validation rigor, model bake-off, …)

### Acceptance gates
- **R1 done when:** every trainer is its own file + agent; pipeline orchestrator is <200 LOC; orphan detection deleted; `/api/training/run/<key>` is a topic post; full regression green.
- **R2 done when:** every successful training run writes a TrainingResult Parquet row; kpi_gate retires a synthetic always-failing model after 3 runs in a test; operator can restore via API.
- **R3 done when:** Model Comparison tab renders all model × tf cells with KPI pass/fail color-coded; sorting works on every column; drill-down opens; Promote / Retire / Restore buttons function.

### Non-goals for Sprint 1a
- No new ML models added
- No new strategies added
- No live-trading behavior changes
- No infra changes outside the trainer / agent / dashboard surface

### Hard rule: stabilize current solution first
> *"I guess we need to postpone on new plan implementation before we stabilize the current solution."*

Sprint 1a starts ONLY after:
- Phase 97c verified working in operator screenshots (1+ pipeline cycle observed end-to-end)
- Banner alert dedup (debug-restart deaths suppressed) — small follow-up phase
- 0 net new failing tests in the offline regression suite

---

## Files this plan will create / change (forward-looking inventory)

### New files (R1)
- `src/engine/trainers/__init__.py`
- `src/engine/trainers/train_base.py`
- `src/engine/trainers/train_trend.py`
- `src/engine/trainers/train_futures.py`
- `src/engine/trainers/train_scalping.py`
- `src/engine/trainers/train_tft.py`
- `src/engine/trainers/train_oft.py`
- `src/engine/trainers/train_meta.py`
- `src/engine/trainers/train_regime.py`
- `src/agents/trainers/__init__.py`
- `src/agents/trainers/base_trainer_agent.py`
- `src/agents/trainers/trainer_base_agent.py`  (× 8 — one per model)
- `src/engine/kpi_gate.py` (R2)
- `data/training_runs/` (R2 — Parquet output dir)

### Modified files
- `src/engine/pipeline_orchestrator.py` — rewrite as topic dispatcher
- `src/dashboard/app.py` — `/api/training/run/<key>` becomes topic post; trainer scheduler deleted; orphan detection deleted; new `/api/kpi/comparison`
- `src/dashboard/templates/index.html` — new "Model Comparison" tab + reused for Strategy/Combo
- `data/training_rules.json` — `kpi_threshold` block per model
- `tests/test_dashboard.py` — assertions for the agent path, KPI gate, comparison tab

### Deleted (after migration shim removed)
- `src/engine/train_all_models.py`

---

## Open questions deferred for later

- **Cross-machine agents.** The current cluster has 4 lanes (LOCAL_RAZER CPU/GPU + Ivan WORKER-1 CPU/GPU). Should each trainer agent be cluster-aware (run on whichever lane is free) or pinned per machine? **Default: pinned per machine for simplicity in v1; add lane-routing in v2.**
- **Agent restart semantics.** If a trainer agent crashes mid-fit, master_agent restarts it — but does the in-flight training resume or re-start? **Default: re-start from scratch in v1; checkpoint resume in v2.**
- **Backtest agent.** Should backtest also become its own agent? **Default: yes in v1 — backtest is the natural follow-up to training and shares the same lifecycle pattern.** Add `src/agents/backtest_agent.py` to R1.

---

## References

- Operator directive 2026-05-10 22:55 (chat transcript)
- Phase 97c fix (commit forthcoming — backend orphan-refresh daemon + frontend canonical-row fallback)
- `TECH_IMPLEMENTATION_PLAN_2026-05-10.md` — original Sprint 0 / 1 sequence; this doc inserts Sprint 1a between them
- `data/training_rules.json` — current model × TF matrix, will gain `kpi_threshold` block in R2
- Existing `src/agents/master_agent.py` — supervisor infra reused for trainer agents
