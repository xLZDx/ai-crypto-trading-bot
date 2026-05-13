# UML Class Diagrams — Trading Bot System

**Date:** 2026-05-13
**Style:** Same UML inheritance + methods notation as the MetaGPT
SoftwareCompany example (Boss / ProductManager / Engineer / QA inheriting from
SoftwareCompany).
**Rendering:** Mermaid `classDiagram` blocks — GitHub web + VS Code Markdown
Preview Enhanced + Obsidian all render these natively.

---

## Diagram 1 — Trading Agents (live bus subscribers)

The bot loop is an in-process AgentBus topology. Every agent inherits from
`BaseAgent`, runs its own thread (or runs on-demand callbacks), and
communicates only via the bus topics.

```mermaid
classDiagram
    class BaseAgent {
        +NAME: str
        +interval_sec: float
        +start()
        +stop()
        +_run_cycle()
        +_setup_subscriptions()
        +heartbeat()
        +publish(topic, payload)
    }

    class DataAgent {
        +interval_sec=3600
        +fetch_candles()
        +publish_bar()
        +publish_candle()
    }

    class SignalAgent {
        +interval_sec=3600
        +_compute_raw_signal()
        +_apply_meta_labeler()
        +publish_signal()
        +publish_regime()
    }

    class SpotAgent {
        +symbols: list
        +CONFIDENCE_THRESHOLD=0.62
        +MAX_HOLD_HOURS=72
        +_on_signal(msg)
        +_get_ml_confidence()
        +_load_models()
    }

    class FuturesAgent {
        +symbols: list
        +LEVERAGE=2.0
        +CONFIDENCE_THRESHOLD=0.60
        +_on_signal(msg)
        +_funding_gate()
        +_liquidation_proximity_gate()
        +_adverse_funding_block()
        +_funding_arb_scan()
    }

    class ScalpingAgent {
        +symbols: list
        +CONFIDENCE_THRESHOLD=0.65
        +MAX_HOLD_BARS=5
        +ROUND_TRIP_FEE=0.0008
        +_run_cycle()
        +_compute_scalping_signal()
        +_load_model()
    }

    class RiskAgent {
        +capital: float
        +interval_sec=1
        +_on_signal(msg)
        +_on_order_filled(msg)
        +_on_bar(msg)
        +check_beta_neutrality()
        +_hard_kill(reason)
        +attach_beta_filter()
    }

    class ExecutionAgent {
        +interval_sec=5
        +TWAP_THRESHOLD_PCT=0.05
        +_on_order_request(msg)
        +_open_position()
        +_close_position()
        +_on_candle()
    }

    class DatabaseAgent {
        +HEARTBEAT_SEC=30
        +STATS_SEC=60
        +_on_candle()
        +_on_signal()
        +_on_trade()
        +_on_strategy_pnl()
        +_on_training_event()
        +_on_news()
        +_flush_loop()
    }

    class QuantAgent {
        +interval_sec=14400
        +monitor_drift()
        +monitor_divergence()
        +publish_perf_alert()
    }

    BaseAgent <|-- DataAgent
    BaseAgent <|-- SignalAgent
    BaseAgent <|-- SpotAgent
    BaseAgent <|-- FuturesAgent
    BaseAgent <|-- ScalpingAgent
    BaseAgent <|-- RiskAgent
    BaseAgent <|-- ExecutionAgent
    BaseAgent <|-- DatabaseAgent
    BaseAgent <|-- QuantAgent
```

---

## Diagram 2 — Trainer Agent Hierarchy (X1 Sprint 1A R1)

Each model type has its own concrete trainer. The cluster orchestrator dispatches
training jobs to the right trainer via the `TRAINER_AGENT_REGISTRY` factory.

```mermaid
classDiagram
    class BaseTrainerAgent {
        +MODEL_KEY: str
        +last_result: dict
        +train(rules_version, n_samples_min) tuple
        +train_async() Thread
    }

    class TrainerMetaAgent {
        +MODEL_KEY="meta"
        +train()
    }

    class TrainerBaseAgent {
        +MODEL_KEY="base"
        +train()
    }

    class TrainerTrendAgent {
        +MODEL_KEY="trend"
        +train()
    }

    class TrainerFuturesAgent {
        +MODEL_KEY="futures"
        +train()
    }

    class TrainerScalpingAgent {
        +MODEL_KEY="scalping"
        +train()
    }

    class TRAINER_AGENT_REGISTRY {
        <<dict>>
        meta: TrainerMetaAgent
        base: TrainerBaseAgent
        trend: TrainerTrendAgent
        futures: TrainerFuturesAgent
        scalping: TrainerScalpingAgent
    }

    class get_trainer_agent {
        <<factory>>
        get_trainer_agent(model_key) BaseTrainerAgent
    }

    BaseTrainerAgent <|-- TrainerMetaAgent
    BaseTrainerAgent <|-- TrainerBaseAgent
    BaseTrainerAgent <|-- TrainerTrendAgent
    BaseTrainerAgent <|-- TrainerFuturesAgent
    BaseTrainerAgent <|-- TrainerScalpingAgent
    get_trainer_agent ..> TRAINER_AGENT_REGISTRY : reads
    TRAINER_AGENT_REGISTRY ..> TrainerMetaAgent : contains
    TRAINER_AGENT_REGISTRY ..> TrainerBaseAgent : contains
    TRAINER_AGENT_REGISTRY ..> TrainerTrendAgent : contains
    TRAINER_AGENT_REGISTRY ..> TrainerFuturesAgent : contains
    TRAINER_AGENT_REGISTRY ..> TrainerScalpingAgent : contains
```

---

## Diagram 3 — Training Decision Layer (Pre/Post-flight + Gate)

The training pipeline has three orchestrating "manager" classes that decide
whether a training job runs and whether its result is accepted.

```mermaid
classDiagram
    class MLEngineerAgent {
        <<pre/post-flight gate>>
        +validate_training_request(model, tf) bool
        +evaluate_trained_model(meta) dict
        +compute_psr(sharpe, n, skew, kurt) float
        +check_data_freshness()
        +check_label_imbalance()
        +check_nan_density()
        +check_distribution_drift()
        +check_feature_count()
    }

    class KPIGate {
        <<3-strike retirement>>
        +THRESHOLDS_PATH
        +RETIRED_PATH
        +is_retired(model, tf) bool
        +evaluate_run(result) str
        +append_run(result)
        +last_n_successful(model, tf, n) list
        +thresholds_for(model) dict
        +restore(key)
        +evaluate_from_meta_json(meta)
    }

    class CIOAgent {
        <<Optuna hyperparameter search>>
        +start_study(model, n_trials, live_mode)
        +get_status() dict
        +get_proposals() list
        +apply_best(model_key)
        +make_cluster_callbacks()
    }

    class BakeOff {
        <<ranked cut list>>
        +INVERTED_METRICS
        +run_bake_off(metric, retire_pct) dict
        +_read_latest_run(model, tf)
        +_enumerate_cells()
    }

    class TrainingRules {
        <<config: data/training_rules.json>>
        +_version: str
        +models: dict
        +matrix: dict
        +params: per_model
        +kpi_threshold: per_model
        +cio_overrides: per_model
    }

    class TrainingRunsParquet {
        <<KPI gate log>>
        +path: training_runs/&lt;model&gt;__&lt;tf&gt;.parquet
        +schema: TrainingResult
        +append(row)
        +query_last_n(model, tf, n)
    }

    MLEngineerAgent ..> TrainingRules : reads thresholds
    KPIGate ..> TrainingRules : reads thresholds
    KPIGate ..> TrainingRunsParquet : appends + reads
    BakeOff ..> TrainingRunsParquet : reads
    BakeOff ..> KPIGate : asks is_retired
    CIOAgent ..> TrainingRules : writes cio_overrides
```

---

## Diagram 4 — Risk Subsystem (gates between RiskAgent and ExecutionAgent)

Risk is layered: each gate has veto power and they're traversed in order.

```mermaid
classDiagram
    class KillSwitch {
        <<sticky pause + operator reset>>
        +daily_loss_R=3.0
        +max_consecutive_losses=5
        +latency_p99_ms=500
        +drawdown_pct=0.08
        +brier_z=2.0
        +paused: bool
        +check_triggers()
        +manual_pause(operator, reason)
        +reset(operator, reason)
    }

    class ValidationGate {
        <<pre-flight data validators>>
        +run(model, tf, df) dict
        +data_freshness_check()
        +label_imbalance_check()
        +nan_density_check()
        +distribution_drift_check()
    }

    class DriftBaseline {
        <<feature distribution snapshot>>
        +save_baseline(model, tf, df) dict
        +load_baseline(model, tf, max_age_days)
        +baseline_age_days(model, tf)
    }

    class BetaNeutralityFilter {
        <<position-level beta cap>>
        +max_beta_exposure=1.0
        +factor="BTC/USDT"
        +would_breach(symbol, side, notional) bool
        +snapshot() dict
    }

    class AgenticLLM {
        <<macro/news veto>>
        +is_active: bool
        +_DECISION_TTL_S=60
        +evaluate_trade(symbol, action) tuple
        +_cached_decision()
        +_cache_decision()
    }

    class OrderManager {
        <<exchange routing>>
        +execute_spot_order()
        +execute_futures_order()
        +_kill_switch_blocks()
        +get_balance()
    }

    ValidationGate ..> DriftBaseline : loads
    OrderManager ..> KillSwitch : checks before order
    OrderManager ..> AgenticLLM : macro veto
```

---

## Diagram 5 — Process Registry (singleton-enforced long-running roles)

Every long-running process registers a role. Duplicates blocked at startup.
Dead PIDs / stale heartbeats reaped every 60s.

```mermaid
classDiagram
    class ProcessRegistry {
        <<src/utils/process_registry.py>>
        +ZOMBIE_AGE_S=300
        +AUDIT_RING_SIZE=200
        +claim_role(role, by) tuple
        +release_role(role, reason) bool
        +heartbeat(role) bool
        +list_active() dict
        +reap_zombies(by) list
        +get_audit_tail(n) list
    }

    class SafeJsonTransaction {
        <<atomic read-modify-write>>
        +transaction(filepath, default, timeout) Context
    }

    class ClaimedRole {
        +pid: int
        +cmdline: str
        +host: str
        +started_at: iso
        +last_heartbeat: iso
        +last_heartbeat_ts: float
        +by: str
    }

    class bot_role { <<role: bot>> }
    class dashboard_role { <<role: dashboard>> }
    class cluster_orch_role { <<role: cluster_orch>> }
    class orderbook_writer_role { <<role: orderbook_writer>> }

    ProcessRegistry ..> SafeJsonTransaction : uses for atomicity
    ProcessRegistry ..> ClaimedRole : creates / stores
    bot_role --|> ClaimedRole : claimed by src.main
    dashboard_role --|> ClaimedRole : claimed by src.dashboard.app
    cluster_orch_role --|> ClaimedRole : claimed by orchestrator
    orderbook_writer_role --|> ClaimedRole : claimed by L2 writer
```

---

## Diagram 6 — Data Ingestion Layer

Five processes feed candles + L2 + funding + news + sentiment into the
Parquet store on the `D:/` volume.

```mermaid
classDiagram
    class RealtimeDBWriter {
        <<klines: Binance WS to Parquet>>
        +symbols
        +timeframes
        +on_kline()
        +flush_to_parquet()
    }

    class OrderbookCollector {
        <<L2 to ZeroMQ bus>>
        +symbols
        +depth=20
        +speed=100ms
        +stream_loop()
        +parse_depth_event()
        +publish_orderflow()
    }

    class OrderbookParquetWriter {
        <<bus to Parquet>>
        +batch_size=1000
        +flush_sec=30
        +_MAX_BUF_ROWS=100000
        +on_snapshot()
        +flush() int
        +_snap_to_row()
    }

    class WatchlistDownloader {
        <<historical archive top-up>>
        +tick_every
        +backfill_missing()
    }

    class DataOrchestrator {
        <<governance across sources>>
        +symbols
        +schedule()
        +health_check()
    }

    class LiveFunding {
        <<Binance funding live>>
        +TTL=300s
        +fetch_funding_rate(symbol)
        +_to_ccxt_perpetual()
        +clear_cache()
    }

    class LiveOpenInterest {
        <<X2: Binance OI live>>
        +TTL=300s
        +fetch_open_interest(symbol)
    }

    class LiveLongShortRatio {
        <<X2: Binance L/S ratio>>
        +TTL=300s
        +fetch_long_short_ratio(symbol, period)
    }

    class ParquetClient {
        <<DuckDB plus partitioned Parquet>>
        +base_dir
        +query(sql)
        +write_ilp(lines)
        +is_available()
    }

    RealtimeDBWriter ..> ParquetClient : writes klines
    OrderbookCollector ..> OrderbookParquetWriter : ZMQ publish/subscribe
    OrderbookParquetWriter ..> ParquetClient : writes L2 partitions
    WatchlistDownloader ..> ParquetClient : backfills
```

---

## Diagram 7 — Feature Engineering Stack (training + inference)

Every model loads candles, then enriches with these features in order. X2
microstructure features were added 2026-05-13.

```mermaid
classDiagram
    class FeatureEngineering {
        <<src/analysis/feature_engineering.py>>
        +add_rsi(df, period)
        +add_macd(df, fast, slow, signal)
        +add_bollinger_bands(df, window)
        +add_roc(df, periods)
        +add_time_features(df)
        +add_taker_and_trade_features(df)
        +add_ofi(df, window)
        +add_vwap(df)
        +add_atr(df)
        +add_keltner(df, ema, atr_mult, atr_period)
        +add_orderbook_features(df)
        +causal_audit(df) dict
    }

    class FractionalDiff {
        +add_fractional_diff(df, d)
    }

    class OrderbookFeatures {
        <<L2/L3 microstructure>>
        +imbalance(v_bid, v_ask)
        +microprice(p_bid, p_ask, v_bid, v_ask)
        +aggregate_levels(snapshot, depth)
        +add_orderbook_features(df)
    }

    class Microstructure {
        <<X2: VPIN/Kyle/Amihud>>
        +add_amihud_illiquidity(df, window)
        +add_kyle_lambda(df, window)
        +add_vpin(df, n_buckets)
        +add_all_microstructure(df)
    }

    class TripleBarrier {
        <<AFML labeling>>
        +pt_multiplier=2.5
        +sl_multiplier=1.5
        +max_bars=12
        +apply(close, atr, t_events)
    }

    class PurgedKFold {
        <<AFML walk-forward CV>>
        +n_splits=5
        +t1
        +pct_embargo
        +split()
    }

    class MetaConfig {
        <<unified META_FEATURES>>
        +META_FEATURES: 23 names
        +CONFIDENCE_THRESHOLD=0.60
        +THRESHOLD_SEARCH_RANGE=(0.40, 0.70)
    }

    FeatureEngineering ..> FractionalDiff : composes
    FeatureEngineering ..> OrderbookFeatures : composes
    FeatureEngineering ..> Microstructure : composes (X2)
```

---

## Diagram 8 — Dashboard API + UI (operator surfaces)

Each card on the dashboard wraps one or more backend endpoints; each
endpoint is protected by `@require_api_key`.

```mermaid
classDiagram
    class FlaskApp {
        <<src/dashboard/app.py>>
        +DASHBOARD_API_KEY
        +require_api_key(decorator)
        +_mint_session_token()
        +_is_session_token_valid()
    }

    class ControlEndpoints {
        +/api/control/run (POST)
        +/api/control/trade_mode (GET, POST)
    }

    class BalanceEndpoints {
        +/api/balance/real
        +/api/balance/virtual
        +/api/balance/by_mode
        +/api/balance/refresh (POST)
        +/api/portfolio
    }

    class TrainingEndpoints {
        +/api/training/rules
        +/api/training/run (POST)
        +/api/training/jobs
        +/api/cluster/submit (POST)
        +/api/cluster/status
    }

    class ModelComparisonEndpoints {
        +/api/model_comparison
        +/api/registry/retired
        +/api/registry/&lt;key&gt;/restore (POST)
        +/api/bake_off
        +/api/cio/start (POST)
        +/api/cio/status
        +/api/cio/proposals
        +/api/cio/apply_best (POST)
    }

    class MonitorEndpoints {
        +/api/monitor/health
        +/api/monitor/services
        +/api/errors/recent
        +/api/errors/dismiss (POST)
        +/api/process/registry
        +/api/process/registry/reap (POST)
    }

    class DataEndpoints {
        +/api/data/coverage
        +/api/data/resample (POST)
        +/api/data/backfill (POST)
        +/api/strategy/stability
        +/api/strategy/full
        +/api/parquet/coverage
    }

    class RiskEndpoints {
        +/api/risk/kill_switch/status
        +/api/risk/kill_switch/pause (POST)
        +/api/risk/kill_switch/reset (POST)
        +/api/risk/overrides
    }

    FlaskApp <|-- ControlEndpoints
    FlaskApp <|-- BalanceEndpoints
    FlaskApp <|-- TrainingEndpoints
    FlaskApp <|-- ModelComparisonEndpoints
    FlaskApp <|-- MonitorEndpoints
    FlaskApp <|-- DataEndpoints
    FlaskApp <|-- RiskEndpoints
```

---

## Diagram 9 — Live trading sequence (one cycle, runtime view)

```mermaid
sequenceDiagram
    autonumber
    participant WS as Binance WS
    participant MA as MarketAnalyzer
    participant RC as RegimeClassifier
    participant SA as SignalAgent
    participant ML as MetaLabeler
    participant BUS as AgentBus
    participant SPOT as SpotAgent
    participant FUT as FuturesAgent
    participant RA as RiskAgent
    participant LLM as AgenticLLM
    participant KS as KillSwitch
    participant EX as ExecutionAgent
    participant BIN as Binance API
    participant DB as DatabaseAgent

    WS->>MA: tick(symbol, price)
    MA->>RC: classify regime
    RC->>BUS: publish 'regime'
    MA->>SA: compute_raw_signal
    SA->>ML: filter(meta_pass?)
    ML-->>SA: PASS/BLOCK
    SA->>BUS: publish 'signal'

    par market specialists
        BUS->>SPOT: signal
        SPOT-->>BUS: publish 'trade_signal' (market=spot)
    and
        BUS->>FUT: signal
        FUT-->>BUS: publish 'trade_signal' (market=futures)
    and
        BUS->>DB: signal (analytics)
    end

    BUS->>RA: trade_signal
    RA->>RA: 9-gate stack<br/>(freshness, latency, circuit,<br/>drawdown, daily loss, liquidity,<br/>beta, Kelly)
    RA->>LLM: evaluate_trade (60s cache)
    LLM-->>RA: APPROVED / REJECTED
    RA->>BUS: publish 'order' (pending)

    BUS->>EX: order pending
    EX->>KS: check_kill_switch
    KS-->>EX: not paused
    EX->>BIN: place order
    BIN-->>EX: filled
    EX->>BUS: publish 'order' (open/closed)
    BUS->>RA: order filled (P&L)
    BUS->>DB: order (persist to Parquet)
    RA->>RA: update Kelly + drawdown<br/>check kill_switch triggers
```

---

## Diagram 10 — Training sequence (one cell, retraining view)

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator
    participant DASH as Dashboard
    participant CO as ClusterOrch
    participant KG as KPIGate
    participant MLE as MLEngineerAgent
    participant TA as TrainerAgent
    participant RULES as training_rules.json
    participant PQ as ParquetClient
    participant RUNS as training_runs/*.parquet
    participant BO as BakeOff
    participant CIO as CIOAgent

    OP->>DASH: Click 'Train all'
    DASH->>CO: POST /api/cluster/submit
    CO->>KG: is_retired(model, tf)?
    alt retired
        KG-->>CO: True
        CO-->>DASH: reject (operator must restore)
    else not retired
        KG-->>CO: False
        CO->>MLE: validate_training_request
        MLE->>RULES: read thresholds
        MLE->>PQ: data freshness check
        MLE-->>CO: PASS / FAIL with reason
        CO->>TA: train(model, tf)
        TA->>RULES: load HPs + cio_overrides
        TA->>PQ: query data
        TA->>TA: Triple Barrier labels
        TA->>TA: PurgedKFold CV
        TA->>TA: CalibratedClassifierCV
        TA->>TA: Sortino threshold search
        TA->>TA: HMAC-SHA256 sign joblib
        TA-->>CO: model + meta JSON
        CO->>MLE: evaluate_trained_model
        MLE->>MLE: compute PSR
        MLE-->>CO: pass/review
        CO->>KG: evaluate_run(result)
        KG->>RUNS: append run row
        KG->>RUNS: last_n_successful(3)
        alt 3 consec below threshold
            KG-->>CO: RETIRE cell
        else
            KG-->>CO: KEEP / REVIEW
        end
        CO-->>DASH: result + KEEP/REVIEW/RETIRE
    end

    Note over OP,CIO: Later (operator-triggered)
    OP->>DASH: Click 'Run bake-off'
    DASH->>BO: GET /api/bake_off
    BO->>RUNS: read all cells
    BO-->>DASH: cut list (keep/review/retire)

    OP->>DASH: Click 'Start CIO study'
    DASH->>CIO: POST /api/cio/start
    CIO->>CIO: TPE search (Optuna)
    OP->>DASH: Click 'Apply best'
    DASH->>CIO: POST /api/cio/apply_best
    CIO->>RULES: write cio_overrides
    Note over RULES,TA: Next retrain merges via _HP_SCHEMA
```

---

## Diagram 11 — Test layer (regression suites)

What each test file covers and which production class it gates.

```mermaid
classDiagram
    class test_signal_topic_topology {
        <<6 tests, X1 fix>>
        +test_specialists_publish_trade_signal_not_signal
        +test_risk_agent_runs_once_per_specialist
        +test_spot_agent_does_not_recurse
        +test_off_whitelist_drop
        +test_meta_pass_false_drop
        +test_db_still_captures_raw_signals
    }

    class test_live_funding_symbol {
        <<4 tests>>
        +test_internal_format_translated
        +test_already_ccxt_passthrough
        +test_spot_format_passthrough
        +test_unknown_format_passthrough
    }

    class test_agentic_llm_throttle {
        <<6 tests>>
        +test_decision_cached_within_ttl
        +test_different_action_not_cached
        +test_different_symbol_not_cached
        +test_all_models_cooled_short_circuits
        +test_partial_cooldown_still_tries
        +test_failure_cached
    }

    class test_process_registry {
        <<14 tests, X1.1>>
        +test_first_claim_succeeds
        +test_reentrant_silent
        +test_duplicate_blocked
        +test_stale_dead_pid_replaced
        +test_stale_heartbeat_replaced
        +test_release_idempotent
        +test_release_refuses_other
        +test_heartbeat_only_owned
        +test_list_active_excludes_dead
        +test_reap_zombies
        +test_audit_ring_bounded
        +test_audit_log_file_written
        +test_concurrent_blocked_by_owner
        +test_concurrent_only_one_wins
    }

    class test_orderbook_parquet_writer {
        <<6 tests, X1.2>>
        +test_snap_to_row_happy
        +test_snap_to_row_malformed
        +test_flush_writes_parquet
        +test_flush_empty_noop
        +test_flush_failure_rebuffers
        +test_batch_spans_month_boundary
    }

    class test_cio_merge_with_defaults {
        <<7 tests, X1.3>>
        +test_no_overrides_returns_defaults
        +test_valid_overrides_merged
        +test_wrong_type_skipped
        +test_out_of_range_skipped
        +test_audit_metadata_filtered
        +test_unknown_model_returns_defaults
        +test_per_model_isolation
    }

    class test_microstructure {
        <<11 tests, X2>>
        +test_amihud_appends
        +test_amihud_is_causal
        +test_amihud_zero_volume_robust
        +test_kyle_lambda_appends
        +test_kyle_lambda_is_causal
        +test_kyle_lambda_no_taker_ratio
        +test_vpin_bounded_0_1
        +test_vpin_is_causal
        +test_vpin_extreme_flow_high
        +test_vpin_balanced_flow_low
        +test_add_all_microstructure
    }

    class test_kill_switch {
        <<X §S0-3>>
        +test_daily_loss_R_trigger
        +test_consec_loss_trigger
        +test_drawdown_trigger
        +test_brier_z_trigger
        +test_sticky_pause
        +test_operator_reset
    }

    class test_kpi_gate {
        <<Layer 2>>
        +test_append_run_round_trip
        +test_3_strike_retire
        +test_threshold_per_model
        +test_restore_un_retires
        +test_evaluate_from_meta_json
    }

    test_signal_topic_topology ..> BaseAgent : guards
    test_process_registry ..> ProcessRegistry : guards
    test_orderbook_parquet_writer ..> OrderbookParquetWriter : guards
    test_cio_merge_with_defaults ..> CIOAgent : guards
    test_microstructure ..> Microstructure : guards
    test_kpi_gate ..> KPIGate : guards
    test_kill_switch ..> KillSwitch : guards
```

---

## How to read the diagrams

| Notation | Meaning |
|---|---|
| `ClassA <\|-- ClassB` | ClassB inherits from ClassA |
| `ClassA ..> ClassB` | ClassA depends on / uses ClassB (no inheritance) |
| `<<stereotype>>` | Class role tag (e.g. `<<factory>>`, `<<config>>`) |
| `+method()` | Public method |
| `+attr: type` | Public attribute with type |
| Sequence diagram `participant X as Y` | Y is the lifeline label, X is the alias |
| Sequence `par … and … and …` | Parallel branches (all happen) |
| Sequence `alt … else …` | Conditional branches |

This file pairs with [SYSTEM_WORKFLOWS_AND_TRAINING_ROADMAP_2026-05-13.md](SYSTEM_WORKFLOWS_AND_TRAINING_ROADMAP_2026-05-13.md):
that one covers process topology + decision logic, this one covers the
class structure + method surface that backs each decision.

## Update history
- 2026-05-13 — initial UML diagrams (post X1+X2 ship).
