# Architecture Flows — deep Mermaid diagrams

**Date:** 2026-05-13
**Purpose:** Eight detailed flow diagrams complementing the high-level PNG diagrams at [core/diagrams/](diagrams/). The PNGs show class structure; these diagrams show runtime sequence + dependency depth.

Source files cited inline. Each diagram references real method names so operators can grep the codebase from a node label.

---

## 1. Models load — `MLPredictor` from `<key>_meta.json` → predict

```mermaid
sequenceDiagram
    autonumber
    participant Agent as SpotAgent / FuturesAgent / SignalAgent
    participant MP as MLPredictor.__init__
    participant FS as filesystem (D:/models)
    participant META as <key>_meta.json
    participant MI as model_integrity.verify_and_load_bytes
    participant JL as joblib.load
    participant MODEL as scikit/HGB calibrated classifier

    Agent->>MP: new MLPredictor(model_filename="btc_rf_model.joblib", model_type="base")
    MP->>FS: open <root>/models/btc_rf_model.joblib
    alt file missing
        FS-->>MP: FileNotFoundError
        MP-->>Agent: is_loaded=False, model=None
    end
    MP->>META: open btc_rf_model_meta.json
    META-->>MP: dict {features, optimal_threshold, signature_hex}

    Note over MP,MI: Phase A8 HMAC integrity gate
    MP->>MI: verify_and_load_bytes(path)
    MI->>MI: hmac.compare_digest(HMAC-SHA256(bytes), signature_hex)
    alt signature mismatch
        MI-->>MP: SignatureError → REFUSE to load
        MP-->>Agent: is_loaded=False
    end
    MI-->>MP: verified bytes
    MP->>JL: joblib.load(BytesIO(bytes))
    JL-->>MP: CalibratedClassifierCV
    MP->>MP: is_loaded=True

    Note over Agent,MODEL: Inference time (per signal)
    Agent->>MP: predict_proba(features_dict)
    MP->>MODEL: predict_proba(X[meta.features])
    MODEL-->>MP: [[p_loss, p_win]]
    MP-->>Agent: float p_win

    opt MetaLabeler.filter path
        Agent->>MP: filter(raw_signal, feats)
        MP-->>Agent: PASS if p_win >= optimal_threshold else BLOCK
    end
```

---

## 2. Infrastructure topology — 11 roles + ports + storage

```mermaid
flowchart TB
    OP([Operator])
    OP -->|restart_all.ps1<br/>early-kill + spawn| BLOCK[11 detached processes]

    subgraph Tier1[Tier-1 operational]
        MON["monitor :5001<br/>component_health_probe"]
        CO["cluster_orch :7700<br/>training queue, 4 lanes"]
        DASH["dashboard :5000<br/>Flask + session_TTL 1h"]
        BOT["bot src.main<br/>trading engine + agent bus"]
        RT["realtime_db_writer<br/>Binance WS klines"]
        OBC["orderbook_collector<br/>depth20@100ms → ZMQ"]
    end

    subgraph Tier2[Tier-2 ancillary]
        OBW["orderbook_writer X1.2<br/>ZMQ → Parquet _L2"]
        WDL["watchlist_downloader<br/>archive top-up"]
        DOR["data_orchestrator<br/>multi-source governance"]
        DEB["debug_supervisor<br/>process crash detector"]
        DW["dashboard_watchdog<br/>auto-respawn dashboard"]
        SW["sweep_watchdog<br/>training stall detector"]
    end

    BLOCK --> MON & CO & DASH & BOT & RT & OBC
    BLOCK --> OBW & WDL & DOR & DEB & DW & SW

    subgraph PROC_REG[Process registry — X1.1]
        REG[(data/process_registry.json)]
        TX[safe_json.transaction<br/>atomic claim+release]
    end

    MON  -.->|claim_role| TX
    CO   -.->|claim_role| TX
    DASH -.->|claim_role| TX
    BOT  -.->|claim_role| TX
    OBW  -.->|claim_role| TX
    TX --> REG

    subgraph Storage["D:/ persistence (NEVER C:)"]
        PARQ[(data/parquet/<br/>OHLCV, _L2, _NEWS)]
        STATE[(data/*.json<br/>agent_status, control,<br/>error_state, training_rules,<br/>process_registry)]
        MODELS_DIR[(models/<br/>*.joblib HMAC-signed<br/>*_meta.json siblings)]
        OPTDB[(data/optuna_orchestrator.db<br/>SQLite, CIO study)]
        RUNS[(data/training_runs/<br/>&lt;model&gt;__&lt;tf&gt;.parquet)]
    end

    RT  -->|klines| PARQ
    OBC -->|ZMQ| OBW -->|batched flush| PARQ
    BOT -->|read+write| STATE
    BOT -->|load joblib| MODELS_DIR
    DASH -->|read| STATE & RUNS & MODELS_DIR
    CO -->|train| MODELS_DIR & RUNS & OPTDB
```

---

## 3. Trading business flow — WS tick → fill → P&L

```mermaid
sequenceDiagram
    autonumber
    participant WS as Binance WebSocket
    participant MA as MarketAnalyzer
    participant RC as RegimeClassifier
    participant SA as SignalAgent
    participant ML as MetaLabeler
    participant BUS as AgentBus
    participant SPOT as SpotAgent
    participant FUT as FuturesAgent
    participant SCALP as ScalpingAgent
    participant DB as DatabaseAgent
    participant RA as RiskAgent
    participant LLM as AgenticLLM (60s cache)
    participant KS as KillSwitch
    participant EX as ExecutionAgent
    participant BIN as Binance API (testnet)

    WS->>MA: tick(symbol, price)
    MA->>RC: classify(features)
    RC->>BUS: publish 'regime'
    MA->>SA: compute_raw_signal()
    SA->>ML: filter(raw_signal, feats)
    ML-->>SA: meta_pass = PASS/BLOCK
    SA->>BUS: publish 'signal'

    par
        BUS->>SPOT: signal
        SPOT->>SPOT: whitelist + regime + conf>=0.62
        SPOT->>BUS: publish 'trade_signal' market=spot
    and
        BUS->>FUT: signal
        FUT->>FUT: live_funding + liq_proximity + adverse_funding + conf>=0.60
        FUT->>BUS: publish 'trade_signal' market=futures
    and
        BUS->>SCALP: signal
        SCALP->>SCALP: ROUND_TRIP_FEE + conf>=0.65
        SCALP->>BUS: publish 'trade_signal' market=scalping
    and
        BUS->>DB: signal (analytics)
    end

    BUS->>RA: trade_signal
    Note over RA: 9 gates traversed in order — see diagram 7

    RA->>LLM: evaluate_trade(symbol, action)
    LLM-->>RA: APPROVED/REJECTED (cache or fresh call)
    RA->>RA: Kelly.size(capital, p_win)
    RA->>BUS: publish 'order' pending

    BUS->>EX: order pending
    EX->>KS: paused?
    alt paused
        KS-->>EX: True → block
    else
        EX->>BIN: place_order (CCXT)
        BIN-->>EX: filled
        EX->>BUS: publish 'order' open
    end

    Note over WS,DB: On position close
    BIN-->>EX: closed
    EX->>BUS: publish 'order' closed

    par
        BUS->>RA: order closed
        RA->>RA: Kelly.record_trade, capital+=pnl, check kill_switch triggers
    and
        BUS->>DB: order closed → cold.trade_events Parquet
    end
```

**Invariants** (tests/test_signal_topic_topology.py):
- One `signal` → one `_on_signal` per matching specialist.
- One `trade_signal` → RiskAgent invoked exactly once.
- One approved order → ExecutionAgent invoked exactly once.

---

## 4. Training business flow — operator → retire decision

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator
    participant DASH as Dashboard
    participant ORCH as pipeline_orchestrator
    participant CLU as cluster_orch :7700
    participant KGRET as KPIGate.is_retired
    participant MLE as MLEngineerAgent
    participant LANE as Lane scheduler
    participant TA as TrainerAgent
    participant PQ as ParquetClient
    participant RULES as training_rules.json
    participant TB as TripleBarrier
    participant PKF as PurgedKFold
    participant CAL as CalibratedClassifierCV
    participant SRT as Sortino threshold search
    participant META as <key>_meta.json
    participant RUNS as training_runs/*.parquet
    participant BO as BakeOff
    participant CIO as CIOAgent

    OP->>DASH: click "Train all"
    DASH->>ORCH: POST /api/training/run
    ORCH->>CLU: POST /api/cluster/submit per (model, tf)
    CLU->>KGRET: is_retired(model, tf)?
    alt retired
        KGRET-->>CLU: True → reject
    else
        CLU->>MLE: validate_training_request
        MLE->>MLE: 5 validators (freshness, label_imbalance, nan_density, drift z, feature_count)
        alt any fail
            MLE-->>CLU: reject with reason
        else all pass
            CLU->>LANE: schedule
            LANE->>TA: dispatch (lane 0=meta+regime, 1=base+trend, 2=futures+scalping, 3=oft+tft)

            Note over TA,META: 9-step pipeline
            TA->>PQ: query data
            TA->>TA: feature engineering
            TA->>TB: pt=2.5, sl=1.5, max_bars=12
            TA->>RULES: read params + cio_overrides
            TA->>TA: merge_with_defaults (schema-bounded)
            TA->>TA: 60/20/20 temporal split + 12-bar purge
            TA->>PKF: 5-fold on train portion, pct_embargo=2*max_bars/N
            loop 5 folds
                PKF-->>TA: (train_idx, val_idx)
                TA->>TA: HGB.fit + accuracy
            end
            TA->>CAL: fit on calibration window (never seen)
            TA->>SRT: search [0.40, 0.70] step 0.05
            SRT-->>TA: best_threshold, best_sortino
            TA->>TA: test → wf_acc, AUC, win_rate, wf_max_dd, wf_sharpe
            TA->>META: HMAC-sign joblib + write meta JSON
            TA-->>CLU: (ok, info, meta_path)

            CLU->>MLE: evaluate_trained_model (PSR Bailey-LdP)
            MLE-->>CLU: pass/review

            CLU->>RUNS: append run row
            CLU->>RUNS: read last 3 successful
            CLU->>CLU: evaluate vs kpi_threshold
            alt 3 consec fails
                CLU-->>DASH: RETIRE (add to retired_models.json)
            else
                CLU-->>DASH: KEEP / REVIEW
            end
        end
    end

    Note over OP,CIO: Operator follow-up
    OP->>DASH: Run bake-off
    DASH->>BO: GET /api/bake_off
    BO->>RUNS: read all cells
    BO-->>DASH: cut list {keep, review, retire}

    OP->>DASH: Start CIO study
    DASH->>CIO: POST /api/cio/start
    CIO->>CIO: Optuna TPE search

    OP->>DASH: Apply best
    DASH->>CIO: POST /api/cio/apply_best
    CIO->>RULES: write cio_overrides
    Note over RULES,TA: Next retrain merges via merge_with_defaults
```

---

## 5. BaseAgent lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant Main as src/main.py
    participant BA as BaseAgent.__init__
    participant Bus as AgentBus
    participant Reg as ProcessRegistry
    participant Thread as Background thread (daemon)
    participant Status as agent_status.json

    Main->>BA: SpotAgent(symbols, data_getter, bus, interval_sec=3600)
    BA->>BA: self.bus, self.interval_sec, self._running=False
    BA->>BA: _setup_subscriptions()
    BA->>Bus: subscribe('signal', self._on_signal)
    BA->>Bus: subscribe('regime', self._on_regime)

    Main->>BA: agent.start()
    BA->>BA: self._running=True
    BA->>Thread: Thread(target=_loop, daemon=True).start()
    BA-->>Main: log "Agent started"
    opt registry-aware
        BA-->>Reg: heartbeat('bot') every 60s
    end

    loop while self._running
        Thread->>Status: _write_agent_status(NAME, 'running', task, interval_sec)
        Note over Thread,Status: holds _status_write_lock through read+merge+write
        Thread->>Thread: _run_cycle()
        alt cycle raises
            Thread->>Status: write status='error', task=<msg>
            Thread-->>Thread: continue
        end
        Thread->>Status: write status='idle'
        Thread->>Thread: time.sleep(interval_sec)
    end

    Main->>BA: agent.stop()
    BA->>BA: self._running=False
    Note over Thread: loop exits on next sleep

    opt heartbeat returns False (registry evicted)
        Reg-->>Main: logger.warning + consec += 1
        opt consec >= 3
            Main->>Main: os._exit(0) so watchdog claims cleanly
        end
    end
```

---

## 6. Trainer dispatch — factory → train → meta JSON

```mermaid
sequenceDiagram
    autonumber
    participant CO as cluster_orch
    participant FAC as get_trainer_agent factory
    participant REG as TRAINER_AGENT_REGISTRY
    participant TA as TrainerXAgent
    participant Func as train_meta_labeler /<br/>train_base_model / etc.
    participant CIO as cio_overrides.merge_with_defaults
    participant RULES as training_rules.json
    participant Sig as model_integrity.sign_model
    participant META as <key>_meta.json

    CO->>FAC: get_trainer_agent("trend")
    FAC->>REG: REGISTRY["trend"]
    alt unknown key
        REG-->>FAC: KeyError "No trainer agent for 'unknown'..."
    end
    REG-->>FAC: TrainerTrendAgent class
    FAC->>FAC: instantiate fresh (not singleton)
    FAC-->>CO: TrainerTrendAgent instance

    CO->>TA: train(rules_version, n_samples_min)
    TA->>RULES: read params + cio_overrides
    TA->>CIO: merge_with_defaults('trend', DEFAULTS, SCHEMA)
    CIO->>CIO: drop wrong-type / out-of-range / non-allowlist
    CIO-->>TA: (merged_params, applied_dict)
    TA->>Func: train_trend_model(timeframe, ...) with merged_params

    Note over Func,META: 9-step pipeline (see diagram 4)
    Func->>Sig: sign_model(joblib_path)
    Func->>META: write meta with cio_overrides_applied=applied_dict
    Func-->>TA: (ok=True, info)

    alt train raised
        Func-->>TA: Exception
        TA->>TA: last_result = {ok: False, error: ...}
        TA-->>CO: (False, {error})
    else
        TA->>TA: last_result = {ok: True, ...}
        TA-->>CO: (True, info, meta_path)
    end

    CO->>CO: update_task(task_id, status='done')
```

---

## 7. Risk subsystem — RiskAgent traverses 9 gates

```mermaid
sequenceDiagram
    autonumber
    participant Bus as AgentBus
    participant RA as RiskAgent._on_signal
    participant G1 as data_freshness (≤300s)
    participant G2 as API latency (<500ms)
    participant G3 as circuit breaker (3 consec)
    participant G4 as cum drawdown (<10%)
    participant G5 as daily loss (<5%)
    participant G6 as liquidity proximity (<0.85)
    participant G7 as beta neutrality
    participant G8 as AgenticLLM macro veto
    participant G9 as Kelly sizing
    participant ORDER as publish 'order'

    Bus->>RA: trade_signal
    RA->>RA: meta_pass? direction != 0?

    RA->>G1: _last_bar_ts age vs DATA_STALE_SEC
    alt fail
        G1-->>RA: BLOCK (log warning)
    end

    RA->>G2: last_api_latency_ms
    alt > API_LATENCY_LIMIT_MS
        G2-->>RA: BLOCK
    end

    RA->>G3: _circuit_open OR kelly.circuit_breaker
    alt either trips
        G3-->>RA: BLOCK
    end

    RA->>G4: drawdown_pct vs MAX_DRAWDOWN_PCT
    alt over
        G4-->>RA: _hard_kill('cumulative_drawdown')<br/>publish 'risk_kill_switch' flatten_all
    end

    RA->>G5: daily_loss_pct vs MAX_DAILY_LOSS_PCT
    alt over
        G5-->>RA: _hard_kill('daily_loss')
    end

    RA->>G6: liq_proximity from raw_signals
    alt > 0.85
        G6-->>RA: BLOCK
    end

    RA->>G7: BetaNeutralityFilter.would_breach
    alt would push |β| past cap
        G7-->>RA: BLOCK (fail-OPEN with WARNING if filter not attached)
    end

    RA->>G8: AgenticLLM.evaluate_trade(sym, action)
    G8->>G8: (symbol, action) cache (60s TTL, threading.Lock)
    alt cache hit
        G8-->>RA: cached decision
    else miss
        G8->>G8: all 11 Gemini models cooled down?
        alt all dead
            G8-->>RA: APPROVED (fail-OPEN, cache decision)
        else
            G8->>G8: 11-model fallback chain
            opt success
                G8-->>RA: APPROVED/REJECTED (cached)
            end
        end
    end
    alt REJECTED
        G8-->>RA: BLOCK
    end

    RA->>G9: kelly.size(capital, p_win, vol_scale × regime_size_mult)
    G9-->>RA: position_usdt

    RA->>ORDER: publish 'order' pending
```

**Constants** (src/engine/agents/risk_agent.py):
- `MAX_DRAWDOWN_PCT = 10.0`
- `MAX_DAILY_LOSS_PCT = 5.0`
- `MAX_CONSECUTIVE_LOSSES = 3`
- `LIQ_PROXIMITY_BLOCK = 0.85`
- `DATA_STALE_SEC = 300`
- `API_LATENCY_LIMIT_MS = 500`

KillSwitch (sticky pause, 5 triggers) is separate and checked at ExecutionAgent — see src/risk/kill_switch.py.

---

## 8. Process registry — `claim_role` under atomic transaction

```mermaid
sequenceDiagram
    autonumber
    participant Caller as src/main.py / dashboard / cluster_orch
    participant CR as claim_role(role, by)
    participant TX as safe_json.transaction<br/>(FileLock + read + yield + write)
    participant FILE as data/process_registry.json
    participant PSU as _pid_alive (psutil)
    participant AUD as _append_audit_log
    participant LOGF as logs/process_registry.log
    participant LOGGER as logging
    participant Caller2 as Caller resumes

    Caller->>CR: claim_role("bot", by="src.main")
    CR->>TX: with transaction(REGISTRY_PATH) as data:
    TX->>FILE: acquire FileLock (5s timeout)
    TX->>FILE: open + json.load (or default {})
    TX-->>CR: data = {roles, audit}

    CR->>CR: existing = data.roles.get("bot")
    alt existing.pid == os.getpid()
        Note over CR: re-entrant — silent success, NO audit entry
        CR-->>Caller: (True, existing)
    end

    opt existing exists
        CR->>PSU: _pid_alive(existing.pid)
        PSU->>PSU: psutil.pid_exists AND status NOT in (ZOMBIE, DEAD)
        PSU-->>CR: ex_alive bool
        CR->>CR: ex_fresh = (now - last_heartbeat_ts) < 300s

        alt ex_alive AND ex_fresh
            CR->>CR: warn_log = "role 'bot' already claimed by PID..."
            CR->>CR: data.audit.append(claim_blocked entry)
            CR->>CR: result = (False, existing)
        else stale
            CR->>CR: data.audit.append(reap entry)
            CR->>CR: existing = None (treat as no-existing)
        end
    end

    opt new claim path
        CR->>CR: new entry = {pid, cmdline, host, started_at, heartbeat_ts, by}
        CR->>CR: data.roles["bot"] = new entry
        CR->>CR: data.audit.append(claim entry)
        CR->>CR: cap audit at AUDIT_RING_SIZE=200
        CR->>CR: info_log = "claimed role 'bot' (PID, by)"
        CR->>CR: result = (True, new entry)
    end

    CR->>TX: end of with block
    TX->>FILE: atomic write (tempfile + os.replace)
    TX->>FILE: release FileLock

    Note over CR: side-effects AFTER lock release (keeps lock-hold tight)
    opt warn_log
        CR->>LOGGER: logger.warning(warn_log)
    end
    opt info_log
        CR->>LOGGER: logger.info(info_log)
    end
    loop audit_logs queued
        CR->>AUD: _append_audit_log(entry)
        AUD->>LOGF: append "{ts} {event} role={role} pid={pid}..."
        opt write fails (only ONCE per process lifetime)
            AUD->>LOGGER: logger.warning("audit log not writable")
        end
    end

    CR-->>Caller: result

    Note over Caller,Caller2: At process atexit
    Caller->>Caller2: release_role("bot", reason="atexit")
    CR->>TX: with transaction
    alt other PID owns
        CR-->>Caller2: False (with WARNING, was DEBUG)
    else we own
        CR->>CR: pop role + audit release
        CR-->>Caller2: True
    end

    Note over Caller,LOGGER: Periodic heartbeat (60s)
    loop forever
        Caller2->>CR: heartbeat("bot")
        alt ownership lost
            CR-->>LOGGER: WARNING "heartbeat ignored — no longer owner"
            CR-->>Caller2: False
            opt 3 consecutive false
                Caller2->>Caller2: os._exit(0)
            end
        else
            CR->>CR: update last_heartbeat_ts
            CR-->>Caller2: True
        end
    end

    Note over Caller,LOGGER: Dashboard reaper (60s)
    Caller2->>CR: reap_zombies(by="dashboard-reaper")
    CR->>TX: with transaction
    loop each role
        CR->>PSU: _pid_alive
        alt not alive OR not fresh
            CR->>CR: pop + audit reap
        end
    end
    CR-->>Caller2: list of reaped role names
```

**Atomicity guarantee** (X1 reviewer fix 2026-05-13): read-check-write runs inside a single `FileLock` acquisition via `safe_json.transaction`. Previous two-lock pattern had TOCTOU window where two concurrent claims could both succeed.

---

## Companion documents

- [SYSTEM_WORKFLOWS_AND_TRAINING_ROADMAP_2026-05-13.md](SYSTEM_WORKFLOWS_AND_TRAINING_ROADMAP_2026-05-13.md) — high-level + X1-X5 roadmap.
- [UML_CLASS_DIAGRAMS_2026-05-13.md](UML_CLASS_DIAGRAMS_2026-05-13.md) — 11 class diagrams (static).
- [../ARCHITECTURE.md](../ARCHITECTURE.md) — Optuna → HistGBT → Risk Gate → Sharpe.
- [diagrams/*.png](diagrams/) — 8 PNG renderings.

## How this document was produced

Aider was installed (v0.86.2 at `D:/tools/aider-env` via `uv` + Python 3.12) and configured against Gemini. Free-tier daily quota for both `gemini-2.5-pro` AND `gemini-2.0-flash` was exhausted by the trading bot's own AgenticLLM today. The doc-generation carve-out in the agents-first rule + the new fallback-to-Claude rule permitted direct Claude generation.

The Aider→Gemini→Claude fallback wrapper is `tools/aider_or_claude.py`.

## Update history
- 2026-05-13 — initial.
