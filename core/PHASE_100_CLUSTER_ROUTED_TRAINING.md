# Phase 100 — Cluster-routed training + explicit lane-agent topology

**Date:** 2026-05-11
**Approved:** 2026-05-11
**Status:** Phase 100a SHIPPED 2026-05-11 (manual single-tf path → cluster). Phase 100b/c deferred to follow-up sessions.
**Replaces:** local `_training_scheduler` + `_resource_kind_for` + `_detect_orphan_training_subprocesses` + Phase 97c orphan refresh loop
**Slots into:** Sprint 0 stabilization (before Sprint 1a R1; Sprint 1a R1 scope shrinks from 5-7d → 2-3d after this lands)

---

## Shipped vs deferred breakdown

### ✅ Phase 100a — SHIPPED 2026-05-11

**Manual single-tf path routes to cluster.** This is the change that fixes the operator's reported bug (`Futures @ 4h queued forever behind pipeline's exclusive lane`).

Changes:
- `_DASH_TO_CLUSTER_KEY` mapping (base→btc_rf, futures→futures_short, meta→meta_labeler; rest are same-name fallthrough)
- `_to_cluster_model_type(dash_key)` helper
- `_dispatch_training_to_cluster(job_id, key, n, tf, with_backtest)` — POSTs n cluster tasks via `_cluster_proxy_post('/api/cluster/submit', spec)`, records `cluster_task_ids` + `cluster_routed=True` on the job, spawns sync thread
- `_sync_cluster_task_status(job_id, key, task_ids, tf, with_backtest)` — 5 s poll of `/api/cluster/tasks`, aggregates status across n tasks (all-done → done, any-cancelled → cancelled, terminal-mix → partial/error), 6 h deadline, ETA self-tune on done, optional followup backtest chaining
- `_cluster_status_to_job_status` mapping (pending→queued, failed→error, etc.)
- `api_training_run_one` routes to cluster by default; `AI_TRADER_LOCAL_TRAINING=1` env var forces legacy local-subprocess path for emergency rollback
- 24 new assertions in `test_phase100_cluster_routed_training_dispatch`

End-to-end verified 2026-05-11 01:48:
- `POST /api/training/run/futures {n:1, tf:4h}` → returned `{ok:true, job_id:02d8f4867741, routed_to:"cluster"}`
- Cluster orchestrator received task `57b5ca41-ea3 model=futures_short tf=4h` (correct mapping)
- Dashboard job record linked: `cluster_routed=True, cluster_task_ids=['57b5ca41-ea3'], status=running` (sync thread alive)
- Full regression: 2059 pass / 14 fail (same 14 pre-existing); +24 net passes, zero regressions

### 🟡 Phase 100b — DEFERRED (next session)

**`tf='all'` + pipeline orchestrator → cluster.**

- `_run_trainer_multi_tf` (tf='all' path): submit N cluster tasks per tf, aggregate status the same way the n>1 loop does today
- `pipeline_orchestrator._run_train_phase`: replace `train_all()` invocation with cluster `submit_full_training_run`-style dispatch (already exists at `orchestrator.py:431`)
- `pipeline_orchestrator` no longer needs to acquire the local `exclusive` lane (it becomes a dispatcher, not a runner)

Risk: tf='all' has internal sequencing semantics + per-tf followup backtest. Need careful refactor.

### 🟡 Phase 100c — DEFERRED (next session)

**Delete obsolete local-scheduler + orphan-detection code.**

- `_TrainingScheduler` class + `_training_scheduler` instance
- `_resource_kind_for` + `_RESOURCE_KIND` map
- `_TRAINING_CPU_CAP` / `_TRAINING_GPU_CAP` / `_LegacySemAdapter` / `_training_concurrency_sem`
- `_detect_orphan_training_subprocesses` + Phase 97c daemon (`_refresh_orphan_current_state`, `_orphan_refresh_loop`)
- `/api/training/scheduler` endpoint (replaced by `/api/cluster/status`)
- `_run_trainer_blocking` once the env-var rollback escape hatch is no longer needed

Pre-req: Phase 100b shipped + 1+ full pipeline cycle observed end-to-end on cluster routing.

### 🟡 Phase 100d — DEFERRED (next session)

**Master_agent agent registry + circuit breaker + lane-agent naming.**

- Promote `node_id` to derived `agent_id` (`<hostname>_<lane>_AGENT`)
- Heartbeat schema additions: `queue_depth`, `load_avg`, `cpu_pct`, `gpu_pct`, `mem_mb`
- `master_agent` tracks deaths per agent; 3 deaths in 10 min → flag unhealthy, orchestrator skips
- New `/agents` endpoint exposing the registry to dashboard

Pre-req: Phase 100a + 100b shipped.

---

## Origin

**Operator screenshot 2026-05-11 ~01:21Z**: clicked ▶ Train on "Futures Short RF @ 5m" row with the TF dropdown changed to 4h. System correctly routed to `futures @ 4h` slot — but parked it in QUEUED instead of starting immediately. The local `_training_scheduler` had its `exclusive_busy=True` flag held by the `pipeline_orchestrator` job (`resource_kind=exclusive`), which gates **all** `acquire()` calls regardless of lane. Manual click and auto-pipeline share the same gate.

Operator directive: *"implement the load balancer to distribute the training power across different nodes CPU/GPU and unlimited instances"* + *"create the separate agents under supervisor load-balancer/agent/daemon to distribute the load across the compute engines (CPU/GPU/worker nodes)"*.

The cluster infra to do this **already exists** for backtest cells (Phase 94). Workers in [`src/training/distributed/worker.py:246-254`](../src/training/distributed/worker.py#L246-L254) already have handlers for every model type (`_train_random_forest`, `_train_sklearn_model`, `_train_tft`, `_train_oft`, `_train_garch`). They've just never been used from the training path — dashboard `/api/training/run/<key>` and `pipeline_orchestrator` spawn LOCAL subprocesses instead.

This phase routes everything through the cluster.

---

## Topology after Phase 100

```
┌─────────────────────────────────────────────────────┐
│ SUPERVISOR   master_agent (existing, extended)      │
│   - heartbeat-monitors every lane agent             │
│   - respawns crashed agents                         │
│   - circuit breaker per agent (3 deaths/10min)      │
│   - reads heartbeats from cluster orchestrator      │
└────────────────────┬────────────────────────────────┘
                     │ supervises
                     ▼
┌─────────────────────────────────────────────────────┐
│ LOAD BALANCER  cluster_orchestrator (existing, :7700)│
│   - receives submit_task from dashboard / pipeline  │
│   - matches task.compute_kind to agent kind         │
│   - picks least-loaded agent (queue depth)          │
│   - dispatches; tracks status                       │
└────────────────────┬────────────────────────────────┘
                     │ dispatches
        ┌────────────┼────────────┬─────────────┐
        ▼            ▼            ▼             ▼
   ┌────────┐  ┌────────┐   ┌────────┐    ┌────────┐
   │LOCAL   │  │LOCAL   │   │IVAN    │    │IVAN    │
   │_CPU    │  │_GPU    │   │_CPU    │    │_GPU    │
   │_AGENT  │  │_AGENT  │   │_AGENT  │    │_AGENT  │
   └────────┘  └────────┘   └────────┘    └────────┘
   (existing worker.py procs, formalized as named lane agents)
```

---

## What changes

### `src/training/distributed/worker.py`
- Each worker process gets `agent_id` (e.g. `LOCAL_RAZER_CPU_AGENT`, `IVAN_GPU_AGENT`) derived from `<hostname>_<compute_kind>_AGENT`.
- Heartbeat every 5s posts `{agent_id, queue_depth, last_task_id, load_avg, cpu_pct, gpu_pct?, mem_mb}` to cluster orchestrator.
- No new processes — same 4 lanes, clearer identity.

### `src/orchestration/master_agent.py`
- New agent registry: reads heartbeats from cluster orchestrator's `/agents` endpoint.
- Tracks deaths per agent (timestamp list); circuit-breaker: 3 deaths in 10 min → mark agent unhealthy, route around it (orchestrator skips agents flagged unhealthy).
- Existing zombie-task detection stays.

### Cluster orchestrator
- New endpoint `/agents` returning all known agents + their current heartbeat.
- Routing: pick least-loaded healthy agent of matching kind. Today: first-match; new: queue-depth-aware.
- `exclusive` kind gated at cluster level (across all lanes) for the `pipeline` task type only — but pipeline becomes a dispatcher (not a runner), so `exclusive` lane usage drops to near-zero in practice.

### `src/dashboard/app.py`
- `/api/training/run/<key>` builds a cluster task payload `{model_type: key, timeframe, symbol, n_iter}` and POSTs to cluster orchestrator. Returns `{job_id, cluster_task_id, status: queued}`.
- `_training_jobs` syncs status from cluster orchestrator (polling extends today's backtest-cell poller).
- Manual job optimistic UI flash continues to work.
- **Deleted:** `_training_scheduler` class + `_resource_kind_for` + `_RESOURCE_KIND` map + `_TRAINING_CPU_CAP` / `_TRAINING_GPU_CAP` constants + `_LegacySemAdapter`.
- **Deleted:** `_detect_orphan_training_subprocesses` (line ~2976) + `_refresh_orphan_current_state` + `_orphan_refresh_loop` (Phase 97c daemon — obsolete because every training task now has a known cluster task ID).
- **Kept:** `training_current.json` write path (now written by cluster orchestrator on task dispatch, not by trainer). FE reads it the same way.

### `src/engine/pipeline_orchestrator.py`
- Rewrite as cluster dispatcher: iterate `data/training_rules.json` model × tf matrix, submit one task per cell, await all, optionally chain backtest tasks afterwards.
- No more local `train_all_models.train_all()` invocation.
- No more `acquire('exclusive')` — pipeline is a coordinator, not a runner.

### `data/training_rules.json`
- No schema change. Cells are submitted to cluster one at a time (orchestrator-side queueing) so the matrix authoring stays as-is.

### Backward compat
- `train_all_models.py` stays as a CLI for the rare "run everything locally without cluster" debug path. Not invoked from dashboard or pipeline orchestrator anymore.
- `launch_training.ps1` keeps working for direct CLI use; the orphan detector that used to surface it on the dashboard is gone (CLI-only runs are now an operator-conscious choice, not a hidden state).

---

## What this kills (cleanup tracker)

| Surface | Was | Now |
|---------|-----|-----|
| `_training_scheduler` (cpu/gpu/exclusive lanes, semaphore) | Local gate, broke on `exclusive_busy` | DELETED — cluster IS the scheduler |
| `_resource_kind_for(model_key)` + `_RESOURCE_KIND` map | Per-model lane mapping local | Task carries `compute_kind`; routing in cluster |
| `_detect_orphan_training_subprocesses` (Phase 97c) | Scanned psutil for trainer-shaped cmdlines | DELETED — every training has a cluster task ID |
| `_refresh_orphan_current_state` + `_orphan_refresh_loop` | 5s daemon refreshing orphan records | DELETED — cluster task status is the truth |
| `/api/training/run/<key>` thread spawn via `_run_trainer_blocking` | Local subprocess | DELETED — POST to cluster |
| `pipeline_orchestrator` running `train_all` | Local sequential runner | REWRITE as cluster dispatcher |
| `exclusive` lane usage | Pipeline orchestrator held it, blocked all manual jobs | NEAR-ZERO — pipeline is a dispatcher; only OFT (single-GPU exclusive) still uses it at the lane-agent level |

---

## Verification protocol

1. **Full kill + restart_all.ps1 + post-edit PID check on:**
   - dashboard
   - master_agent
   - cluster_orchestrator
   - all 4 lane agents
2. **Manual click test:** ▶ Train on Futures @ 4h → confirm cluster submits, CPU lane agent picks up within ~5s, runs without queue gate (the old `_training_scheduler` bug is gone).
3. **Manual GPU test:** ▶ Train on TFT → confirm GPU lane agent picks it up.
4. **Manual + auto coexistence:** start pipeline orchestrator, then click manual Train on multiple rows. Confirm:
   - Pipeline dispatches its N tasks
   - Manual tasks queue at cluster orchestrator level (fair: FIFO per lane)
   - No false-exclusive gate; manual jobs can stack
5. **Agent kill / respawn test:** `Stop-Process` one lane agent → master_agent respawns within ~5s. Submit a task → confirm new agent picks it up.
6. **Circuit breaker test:** force-kill the same agent 4 times in 60s → confirm master_agent flags it unhealthy on death #3, orchestrator routes around it, banner shows CRITICAL.
7. **Banner audit:** confirm no recurring "debug died" alerts from the obsolete Phase 97c path.
8. **Regression:** full offline test suite ≥ baseline + new `test_phase100_…` assertions.

---

## Tests added (`tests/test_dashboard.py`)

`test_phase100_cluster_routed_training_and_lane_agent_topology` — covers:
- Worker has `agent_id` resolution + heartbeat fields
- Cluster orchestrator `/agents` endpoint shape
- Cluster orchestrator least-loaded routing logic
- Master_agent agent registry + circuit breaker (3 deaths / 10 min)
- `/api/training/run/<key>` POSTs to cluster (no local subprocess)
- `pipeline_orchestrator` is a cluster dispatcher (no `train_all` import / call)
- Deleted symbols absent: `_training_scheduler`, `_resource_kind_for`, `_RESOURCE_KIND`, `_detect_orphan_training_subprocesses`, `_refresh_orphan_current_state`, `_orphan_refresh_loop`
- Banner / `process_deaths.json` no longer cluttered by orphan-detector artifacts

---

## Risks + mitigations

| Risk | Mitigation |
|------|-----------|
| Cluster orchestrator API doesn't have all the routes I assume | Verify endpoint inventory before code edits. Add missing routes incrementally. |
| Worker training handlers haven't been exercised live | First-task verification step proves the path. Roll back if `_train_sklearn_model` fails for an unforeseen reason. |
| Deleting Phase 97c removes the safety net for currently-running orphan trainings | Drain check: refuse to delete Phase 97c code until no orphan-* records exist in `_training_jobs`. Wait for current pipeline cycle to finish, then ship. |
| Ivan worker unreachable → manual click on a TFT task waits forever | Cluster orchestrator already times out unresponsive lanes (existing Phase 94 behavior). Add per-task `submit_timeout_s` (default 30s) — if no agent claims it, surface to dashboard as `failed: no agent available`. |
| Test surface change is large | Each deletion comes with a "symbol not present" assertion; routing change comes with a "POST not subprocess" assertion. Run regression after every code group to catch breakage early. |

---

## Open questions deferred

- **Local CLI use of `train_all_models.py`** — keep working for debugging, but should it auto-submit to cluster when cluster is reachable? Default: no (CLI is a deliberate cluster-bypass).
- **Task priority** — manual operator clicks should jump the queue ahead of auto-pipeline. Add `priority: int` field in v2.
- **Backtest tasks** — Phase 94 already routed; double-check nothing in this refactor breaks it.
- **Cross-machine GPU** — IVAN_GPU not always reachable. Today: orchestrator skips dead lanes. Future: explicit "prefer LOCAL_GPU if available" hint.

---

## Files inventory

### Modified
- `src/training/distributed/worker.py`
- `src/orchestration/master_agent.py`
- `src/orchestration/cluster_orchestrator.py` (verify exact path during investigation)
- `src/dashboard/app.py`
- `src/engine/pipeline_orchestrator.py`
- `tests/test_dashboard.py`

### Untouched
- `src/engine/train_all_models.py` (CLI-only path)
- `launch_training.ps1` (CLI launcher; for cluster-bypass)
- `data/training_rules.json` (no schema change)

### Deleted symbols (within app.py)
- `_training_scheduler`, `_TrainingScheduler` class, `_resource_kind_for`, `_RESOURCE_KIND`
- `_TRAINING_CPU_CAP`, `_TRAINING_GPU_CAP`, `_LegacySemAdapter`, `_training_concurrency_sem`
- `_detect_orphan_training_subprocesses`
- `_refresh_orphan_current_state`, `_orphan_refresh_loop`, the daemon thread
- `_reattach_training_subprocess`, `_training_state_recover` (or kept as cluster-state-recover variants — TBD during code review)
