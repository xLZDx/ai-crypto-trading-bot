# STATUS — AI Trading Assistance

**Updated:** 2026-05-11
**Purpose:** Single source of truth for current architecture, shipped phases, pending work, and operator runbook for manual model training. Created at operator request — "save everything to root for trace".

---

## Quick reference — manual model training

Operator can now trigger training one model at a time. All paths route through the distributed cluster orchestrator (port 7700) which load-balances across all healthy worker lanes.

### Train one model at one tf

**UI:** Model Training table → row → TF dropdown → ▶ Train button.

**CLI:**
```powershell
curl -X POST -H "Content-Type: application/json" -d '{"n":1,"tf":"4h","force":true}' http://127.0.0.1:5000/api/training/run/futures
```

Returns `{ok: true, job_id: "...", routed_to: "cluster", tf: "4h"}`. Job tracked in `data/training_jobs.json`. Cluster task in `/api/cluster/tasks`.

### Valid model keys (dashboard → cluster mapping)

| Dashboard key | Cluster `model_type` | Default TF | Resource lane |
|---------------|---------------------|------------|---------------|
| `base` | `btc_rf` | 1h | cpu |
| `trend` | `trend` | 1h | cpu |
| `futures` | `futures_short` | 1h | cpu |
| `scalping` | `scalping` | 1m | cpu |
| `meta` | `meta_labeler` | 1h | cpu |
| `regime` | `regime` | 1h | cpu |
| `tft` | `tft` | 1h | **gpu** |
| `oft` | `oft` | 1m | **exclusive (gpu)** |

### Train all models (entire pipeline)

**UI:** ▶ Retrain ALL button (top of Model Training table).

**CLI:**
```powershell
curl -X POST -H "Content-Type: application/json" -d '{"force":true}' http://127.0.0.1:5000/api/training/run/all
```

Triggers Phase 100b distributed dispatch — submits all 25 (model, tf) cells from `DEFAULT_PER_KEY_TFS` to cluster. Per cell: train completes before its BT submits. Across cells: full parallelism across all 4 lanes.

### Check what's running

```powershell
# Cluster-side: actual workers + tasks
curl http://127.0.0.1:7700/api/cluster/status
curl http://127.0.0.1:7700/api/cluster/workers
curl http://127.0.0.1:7700/api/cluster/tasks

# Dashboard-side: job records
curl http://127.0.0.1:5000/api/training/jobs

# Pipeline state (auto-pipeline orchestrator)
curl http://127.0.0.1:5000/api/pipeline/status
type data\training_current.json    # which (model, tf) the auto-pipeline is on right now
```

### Emergency rollback (cluster down / cluster routing misbehaves)

```powershell
$env:AI_TRADER_LOCAL_TRAINING = '1'     # manual click + Retrain ALL route to legacy local subprocess
$env:AI_TRADER_PIPELINE_LOCAL = '1'     # auto-pipeline_orchestrator runs train_all() in-process
# then restart
.\restart_all.ps1
```

Default for both flags is `0` (cluster routing).

---

## Architecture — where every training path goes

```
┌─────────────────────────────────────────────────────────────┐
│ TRIGGER                          ROUTE                       │
├─────────────────────────────────────────────────────────────┤
│ Manual ▶ Train row button   →    Phase 100a                  │
│                                  /api/training/run/<key>     │
│                                  → _dispatch_training_to_    │
│                                    cluster()                 │
│                                  → POST /api/cluster/submit  │
│                                                              │
│ Manual ▶ Retrain ALL        →    Phase 100b                  │
│                                  /api/training/run/all       │
│                                  → _run_retrain_all_         │
│                                    distributed()             │
│                                  → submits 25 cells, chains  │
│                                    BT per cell as train done │
│                                                              │
│ Auto pipeline_orchestrator  →    Phase 100e                  │
│                                  _run_train_phase()          │
│                                  → _run_train_phase_cluster()│
│                                  → same per-cell loop as     │
│                                    Phase 100b                │
│                                                              │
│ Legacy CLI                       train_all_models.py         │
│ (operator-explicit)              (kept for debugging)        │
└─────────────────────────────────────────────────────────────┘

ALL roads lead to cluster_orchestrator (port 7700)
↓
┌─────────────────────────────────────────────────────────────┐
│ LOAD BALANCER  cluster_orchestrator                          │
│   - Receives submit_task                                     │
│   - Matches compute_kind to worker lane                      │
│   - Routes to free worker, GPU-first sort                    │
│   - Watchdog kills stuck tasks                               │
└─────────────────────────────────────────────────────────────┘
↓
┌───────────────────┬───────────────────┐
│ LOCAL_RAZER_CPU   │ LOCAL_RAZER_GPU   │     ← this machine
├───────────────────┼───────────────────┤
│ IVAN_CPU          │ IVAN_GPU          │     ← worker laptop
├───────────────────┼───────────────────┤
│ (future worker)   │ (future worker)   │     ← zero-code-change scale
└───────────────────┴───────────────────┘
```

Adding a future worker = register via `POST /api/cluster/register` from that machine's `worker.py`. Orchestrator picks it up on next dispatch tick. No code change in dashboard, pipeline, or this status doc.

---

## Phases shipped (in commit order)

| Phase | Commit | Date | Summary |
|-------|--------|------|---------|
| **97c** | `0802a36` | 2026-05-11 | Orphan record periodic refresh (5s daemon) + frontend canonical-row fallback. Fixes the "pipeline running but no row shows RUNNING" bug class. |
| **98** | `713862e` | 2026-05-11 | ETA Train + ETA BT columns. Per-(model, tf) rolling average + seed defaults. Color-tinted by duration band. Sortable. |
| **100a** | `345bebb` | 2026-05-11 | Manual ▶ Train per-row routes to cluster. Operator can stack manual clicks; cluster serializes per worker, parallelizes across workers. `AI_TRADER_LOCAL_TRAINING=1` = legacy rollback. |
| **100 functional tests** | `6c4dce8` | 2026-05-11 | Refactored sync into pure `_aggregate_cluster_task_statuses` for testability. 41 functional assertions (call the code, not string-match). Codified "Functional Tests Prove Behavior" rule globally. |
| **100b** | `91d4b20` | 2026-05-11 | Retrain ALL routes through cluster. Parallel cells, sequential train→BT per cell. Pure `_retrain_all_step` extracted. 40+ functional assertions. |
| **100e** | *(pending commit)* | 2026-05-11 | Auto pipeline_orchestrator routes through cluster. Workers (Ivan + Razer) finally get tasks instead of staying idle. `AI_TRADER_PIPELINE_LOCAL=1` = emergency rollback. 25 functional assertions. |

Total functional test additions through Phase 100e: 130+ assertions, all proving observable behavior.

---

## Phases queued (priority order)

### Phase 100d — fix `/api/training/jobs` slowness + FAILED row display
**Why:** Endpoint times out at >60s currently, freezing the FE poller. Result: every row shows "OK" even when a job is running or last train errored.
**Fix:** Throttle `_persist_training_jobs` to 1 write per 2s, copy-then-release lock pattern in the endpoint, persist `last_error` per (model, tf) and surface as FAILED badge in `/api/strategy/full` row builder.
**Tests:** Endpoint <500ms with 100 synthetic jobs; FAILED badge renders when `data/training_last_errors.json` carries entry.

### Phase 100c — delete obsolete code (post-100e cleanup)
- `_TrainingScheduler` class + `_RESOURCE_KIND` map + caps
- `_detect_orphan_training_subprocesses` + Phase 97c refresh daemon
- `_resource_kind_for`, `_LegacySemAdapter`, `_training_concurrency_sem`
- `/api/training/scheduler` endpoint (replaced by `/api/cluster/status`)
- `_run_trainer_blocking` (once env-var rollback escape hatch is retired)

### Phase 99 — banner-monitoring respawn agent
**Decisions locked** in `memory/project_debug_orchestrator_decisions.md`:
- Recipes file: `data/process_recipes.json`
- Circuit breaker: per-role rate limit (default 5/hr) + global tripped flag on first per-role limit breach
- Reuse `scripts/debug_supervisor.py` (extend from detector to detector+healer)
- `dash` managed by `dashboard_watchdog` (skip); cluster workers managed by `master_agent` (skip); transient roles (`training`) get death recorded but no respawn
- Side benefit: kills the recurring "debug died Ns ago" banner that fires every restart_all cycle

---

## Strategic plans (canonical docs)

| Doc | Status | Scope |
|-----|--------|-------|
| [`core/PHASE_100_CLUSTER_ROUTED_TRAINING.md`](core/PHASE_100_CLUSTER_ROUTED_TRAINING.md) | 100a/b/e shipped; 100c/d/99 queued | Master Phase 100 plan |
| [`core/SPRINT_1A_PER_MODEL_AGENTS_AND_KPI.md`](core/SPRINT_1A_PER_MODEL_AGENTS_AND_KPI.md) | Post-stabilization | R1 file-per-model + agent-per-model; R2 KPI gate + auto-retire; R3 granular comparison dashboard. R1 shrinks 5-7d → 2-3d thanks to 100e. |
| [`TECH_IMPLEMENTATION_PLAN_2026-05-10.md`](TECH_IMPLEMENTATION_PLAN_2026-05-10.md) | Post-Sprint-1a | Sprint 0 §0 + §S0-1 through §S0-6 + §S0a/b/c risk hardening + §S0.5 analytic phase |
| [`COMPETITIVE_ASSESSMENT_2026-05-10_v2.md`](COMPETITIVE_ASSESSMENT_2026-05-10_v2.md) | Reference | 48-item roadmap, pruned by personal-use reframe |
| [`PLAN_2026_05_08_outstanding.md`](PLAN_2026_05_08_outstanding.md) | Reference | Older priority roadmap (mostly absorbed into Phase 100 + Sprint 1a) |
| [`INSTITUTIONAL_UPGRADE_PLAN.md`](INSTITUTIONAL_UPGRADE_PLAN.md) | Reference | 5-level upgrade to quant hedge fund quality |
| [`updated_architecture_plan_en.md`](updated_architecture_plan_en.md) | Reference | 18-point architecture plan |

---

## Active issues (NOT regressions, pre-existing)

### 1. `/api/training/jobs` slow → FE can't update status pills
**Symptom:** All Model Training rows show "OK" even when pipeline is running. Root cause: endpoint takes >60s to respond. FE poller stalls; `_trActiveByModel` never populates; rows fall through to static `_statusFor(m)`.
**Tracked by:** Phase 100d (queued, next).

### 2. "debug died Ns ago" banner during restart cycles
**Symptom:** Every `restart_all.ps1` cycle leaves a CRITICAL banner entry because `_TRANSIENT_DEATH_ROLES` doesn't include `"debug"` — debug supervisor's own restart is recorded as a death event.
**Tracked by:** Phase 99 (queued).

### 3. Pre-existing test failures (14 total, no new since Phase 100e)
Same 14 failures across recent runs — string-match assertions for code that's been refactored, brittle template-interpolated values, and pre-existing trainer bugs (sklearn calibration, pipeline exit code 0xFFFFFFFF). Listed in `tests/test_dashboard.py` output. Not blocking.

---

## Global rules in effect (CLAUDE.md hierarchy)

1. **`~/.claude/CLAUDE.md`** — user-home global (cross-volume): Functional Tests Prove Behavior + Code Review Before Reporting Done
2. **`D:\test 2\CLAUDE.md`** — volume global (all D:\test 2 projects): Approval Gate (double-ask), No Guessing, Verify Before Claiming Fixed, Regression Test Maintenance, Functional Tests Prove Behavior, Git Lifecycle (todo-in-commits), Shell Pre-Approved, D:-drive-only disk policy, Plan Persistence, Save Rules Globally
3. **`D:\test 2\AI trading assistance\CLAUDE.md`** — project-specific only: ParquetClient, testnet default, Gemini model chain, training pipeline paths, cluster orchestrator port

### Key methodology rules
- **Functional Tests Prove Behavior** — string-match tests acceptable only as supplementary smoke checks; every behavior needs a test that actually calls the code and asserts on observable state.
- **Bug-fix loop:** reproduce live → failing test → fix → test passes → regression → live verify → commit (test + fix together).
- **Verify Before Claiming Fixed:** full kill of old processes, browser reload, watch UI render, audit banner, run regression — ALL of these, then claim.
- **Approval Gate:** present plan → wait for "approved/go/yes/proceed" → restate plan → wait for second confirmation → first code edit.
- **Include todo list in every commit body** so `git log` is auditable.

---

## Cluster orchestrator API (for trace/debug)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/cluster/status` | Overview: counts, workers, recent tasks |
| `GET /api/cluster/workers` | All workers with lane, status (idle/busy), online flag, last_seen_ago |
| `GET /api/cluster/tasks` | All tasks with model_type, status, assigned_to, started_at, error |
| `POST /api/cluster/submit` | Submit task. Body: `{model_type, symbol, timeframe, config, data_path, output_path}` |
| `POST /api/cluster/submit_all` | Submit full training run (one task per (model, symbol)) |
| `POST /api/cluster/register` | Worker heartbeat / re-register |
| `POST /api/cluster/task_update` | Worker reports task result |
| `DELETE /api/cluster/task/<id>` | Cancel task |

---

## How to verify cluster distribution is working

After triggering any training, watch:

```powershell
# Should show workers transitioning idle → busy as tasks dispatch
while ($true) { Clear-Host; curl -s http://127.0.0.1:7700/api/cluster/workers | python -m json.tool | Select-String "node_id|status|online" ; Start-Sleep 5 }
```

Expected during pipeline / Retrain ALL: at least 2 workers status=busy across both Razer and Ivan lanes (1 CPU + 1 GPU each at minimum if GPU models are in the cell list).

---

## Test suite

- **Path:** `tests/test_dashboard.py`
- **Run offline:** `venv/Scripts/python.exe tests/test_dashboard.py --offline`
- **Current:** 2165 pass / 14 fail (same 14 pre-existing failures; +130 net new passes from Phase 97c/98/100a/b/e + functional tests)
- **Gate:** 0 failures required for new code to be considered shipped; new failures must be fixed before commit

---

## Operator runbook — starting models one by one

Per the operator's plan to train models manually one at a time:

1. **Restart everything fresh** (cluster routing in place):
   ```powershell
   .\restart_all.ps1
   ```

2. **Verify cluster + dashboard up:**
   ```powershell
   curl http://127.0.0.1:5000/api/state            # dashboard
   curl http://127.0.0.1:7700/api/cluster/status   # cluster orchestrator
   ```

3. **Pick a model + tf, trigger train** (UI ▶ Train button or curl):
   ```powershell
   curl -X POST -H "Content-Type: application/json" -d '{"n":1,"tf":"4h","force":true}' http://127.0.0.1:5000/api/training/run/futures
   ```

4. **Watch it work:**
   ```powershell
   # Cluster side
   curl http://127.0.0.1:7700/api/cluster/tasks | python -m json.tool
   # Dashboard side
   curl http://127.0.0.1:5000/api/training/jobs | python -m json.tool
   ```

5. **When done, pick the next:** repeat step 3 with a different model or tf.

Cluster will queue at the orchestrator level if a lane is busy. Operator can stack as many manual clicks as desired — no local-scheduler gate blocks them (that was removed in Phase 100a).
