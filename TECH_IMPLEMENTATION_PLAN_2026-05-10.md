# Tech Implementation Plan — Sprint 0 + Analytic Phase
**Date:** 2026-05-10
**Companion to:** [`COMPETITIVE_ASSESSMENT_2026-05-10_v2.md`](COMPETITIVE_ASSESSMENT_2026-05-10_v2.md) §11 (Sprint 0) and §12 (Analytic phase).
**Goal:** scaffold the validation + audit + kill-switch infrastructure required to answer "what is actually working and making money?" before any v2 distribution sprint runs. Output of Sprint 0 is a **cut list**; the analytic phase consumes the cut list and removes everything that didn't make it.

---

## How to read this document
- Each S0-N section maps 1:1 to §11 in the assessment doc.
- File paths are absolute relative to project root (`d:\test 2\AI trading assistance\`).
- "New file" = create. "Edit file" = surgical change to existing file.
- Each section ends with **success criteria** (binary: pass / fail) and **decision gate** (what triggers proceeding to the next section vs. blocking the whole sprint).
- Time estimates assume one developer working sequentially. Several can parallelize on the 2-PC cluster, but watch the dependencies in §0.

---

## §0 — Cross-cutting setup (~1 day, day 0 of Sprint 0)

These are prerequisites every later section depends on.

### 0.1 Audit output directory
**New dir:** `data/audit/`
- `data/audit/sprint0_model_bakeoff.md` — output of S0-2
- `data/audit/sprint0_validation_reports/` — one subdir per model
- `data/audit/sprint0_calibration/` — calibration diagrams
- `data/audit/sprint0_cut_list.md` — final output (S0-6)

Add `data/audit/` to `.gitignore` *only if* it grows beyond ~50 MB. Otherwise track it — the audit output IS the deliverable.

### 0.2 New top-level package: `src/audit/` and `src/validation/`
- `src/validation/__init__.py` — exposes `validate_model(model, X_train, y_train, X_test, y_test, **kwargs) -> ValidationReport`
- `src/audit/__init__.py` — exposes `run_bakeoff(...)`, `run_calibration_audit(...)`, `run_leakage_check(...)`

These are the public surfaces every trainer + every test will import.

### 0.3 Shared types
**New file:** `src/validation/types.py`
```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ValidationReport:
    model_name: str
    timeframe: str
    symbol: str
    walk_forward_sharpe: float
    walk_forward_sortino: float
    walk_forward_max_dd: float
    walk_forward_n_folds: int
    leakage_detected: bool
    leakage_features: list[str] = field(default_factory=list)
    adversarial_auc: float = 0.0
    adversarial_alarm: bool = False
    calibration_ece: float | None = None  # None if not a probabilistic model
    embargo_pct: float = 0.0
    label_scheme: Literal["fixed", "vol_adjusted"] = "fixed"
    pt_k1: float | None = None  # vol multiplier for profit target
    sl_k2: float | None = None  # vol multiplier for stop loss
    # Decision
    verdict: Literal["live", "shadow", "kill"] = "kill"
    verdict_reason: str = ""
```

Every validation harness call returns one of these. The cut-list builder consumes them.

### 0.4 New strategy_registry status field
**Edit:** `src/engine/strategy_registry.py` — add `validation_status: Literal['live', 'shadow', 'killed', 'unaudited']` to the strategy spec dict, default `'unaudited'`. Don't change anything else yet; the analytic phase will set the values.

### 0.5 Dashboard "Sprint 0" tab placeholder
**Edit:** `src/dashboard/templates/index.html` — add a single new tab `Audit` (collapsed by default) with empty divs that S0-1 through S0-5 will populate. This avoids 5 separate "add a tab" PRs.

Tab structure:
```
[Audit]
  ├── [Validation reports]   (S0-1 fills)
  ├── [Model bake-off]       (S0-2 fills)
  ├── [Kill switch]          (S0-3 fills)
  ├── [Execution quality]    (S0-4 fills)
  ├── [Calibration]          (S0-5 fills)
  └── [Cut list]             (S0-6 fills)
```

**§0 success criteria:** tab renders with empty divs; `from src.validation import validate_model` and `from src.audit import run_bakeoff` import without error (functions not implemented yet, just stubs).

---

## §S0-1 — Validation rigor pipeline (~5 days)

The single most important piece of Sprint 0. Every other section relies on this harness producing trustworthy verdicts.

### S0-1.1 Vol-adjusted Triple Barrier (~1 day)

**New file:** `src/validation/vol_adjusted_barriers.py`

Replaces fixed PT/SL with `PT = k₁·σ_t`, `SL = k₂·σ_t`. `σ_t` is rolling realized vol on the same bar grid as the labels (default 30-bar window for 1m, scale proportionally for higher TFs).

```python
def vol_adjusted_triple_barrier(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    vol_window: int = 30,
    k1: float = 1.8,   # profit target multiplier
    k2: float = 1.2,   # stop loss multiplier
    side_col: str | None = None,  # if None, both sides; else use {-1, +1} from this col
    price_col: str = "close",
) -> pd.Series:
    """Return label series in {-1, 0, +1} matching the Lopez de Prado spec.

    +1 = profit target hit before stop or horizon
    -1 = stop hit before profit or horizon
     0 = horizon hit first (no clean signal)

    σ_t computed from rolling-window realized log-return std.
    Both barriers scale with σ_t so labels are consistent across regimes.
    """
```

Replaces (or shadows) any current label-generation in trainers. The trainer needs a `--label-scheme=vol_adjusted` flag with `fixed` as the back-compat default.

**Edits:**
- `src/engine/train_meta_labeler.py` (this is where labels are generated for the meta-model)
- `src/engine/train_*.py` (any trainer that builds labels — touch each only enough to add the flag)

### S0-1.2 Walk-forward harness (~1 day)

**New file:** `src/validation/walk_forward_harness.py`

```python
@dataclass
class WalkForwardConfig:
    train_days: int = 60
    val_days:   int = 14
    test_days:  int = 14
    embargo_pct: float = 0.03   # 3 % of bars between test end and next train start
    min_folds:   int = 4

def walk_forward(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    model_factory,           # callable -> a fresh model instance
    cfg: WalkForwardConfig,
) -> list[FoldResult]:
    """Roll the (train, val, test) window forward; return list of per-fold
    results (Sharpe, AUC, calibration ECE, n_train, n_test, etc.).
    """
```

**Decision rule for `min_folds`:** if `len(df) / (train_days + val_days + test_days)` < 4, raise — there isn't enough data to walk-forward at this TF. Bot operator must either use a longer history or accept that the strategy can't be walk-forward validated yet.

### S0-1.3 Embargo (~0.5 day, integrated with §S0-1.2)

The embargo is computed in `walk_forward()` itself: after each fold's test window, skip `embargo_pct` of total dataset bars before starting the next train window.

Add a unit test: a strategy with deliberately overlapping label horizons should have its Sharpe DROP when embargo is enabled vs. disabled. If Sharpe doesn't drop, the embargo isn't being applied correctly.

### S0-1.4 Feature leakage detector (~1 day)

**New file:** `src/validation/leakage_detector.py`

Three checks, one report:

1. **Future-bar reference.** Walk the feature columns. For each feature, compute Spearman correlation with `close.shift(-k)` for k = 1..5. If `|ρ| > 0.95`, flag as leakage.
2. **Rolling-window includes-current-bar.** AST-walk the trainer module to find rolling windows; verify all use `closed='left'` or `.shift(1)` after rolling. Flag any that don't.
3. **Normalization computed across full dataset.** Find any `StandardScaler.fit(X)` or `df.mean()` / `df.std()` calls that aren't inside the train fold. Flag.

```python
def detect_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    trainer_module: types.ModuleType,
) -> LeakageReport:
    """Return list of leak-suspected feature names + AST findings."""
```

**Severity:** any `severity='high'` finding blocks model promotion to live; `severity='medium'` puts the model in shadow mode.

### S0-1.5 Adversarial validation (~1 day)

**New file:** `src/validation/adversarial_validator.py`

Train a binary classifier `period == train_period` on combined (train_period_df, recent_live_df). Use the same features the strategy uses. If `AUC > 0.6`, distribution shift between train and live is significant — model is unlikely to transfer.

```python
def adversarial_auc(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    feature_cols: list[str],
    classifier_factory=None,    # default LightGBM
) -> AdversarialReport:
    """Return AUC + per-feature drift score (which features are doing the
    most to separate train from live)."""
```

**Decision rule:**
- `AUC < 0.55`: pass.
- `0.55 ≤ AUC < 0.65`: warn; model goes to shadow.
- `AUC ≥ 0.65`: block; model goes to kill list. Operator must rebuild features or extend train window.

### S0-1.6 Single entry point: `validate_model()`

**Edit:** `src/validation/__init__.py`

```python
def validate_model(
    model_factory,
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    *,
    trainer_module=None,
    use_vol_adjusted_barriers: bool = True,
    walk_forward_cfg: WalkForwardConfig | None = None,
    name: str = "",
    timeframe: str = "",
    symbol: str = "",
) -> ValidationReport:
    """Run the full Sprint 0 validation gauntlet on one model.

    Order of checks (cheapest first, fail fast):
      1. Leakage detection (static + correlation)  — blocks if high severity
      2. Walk-forward + embargo (Sharpe / Sortino / DD / AUC)
      3. Calibration ECE (only for probabilistic models)
      4. Adversarial validation (AUC vs live)
    Returns a report; sets verdict ∈ {live, shadow, kill}.
    """
```

### S0-1.7 Wire into trainers

Each trainer (`src/engine/train_*.py`) gains, at the end:

```python
from src.validation import validate_model
report = validate_model(
    model_factory=lambda: <trainer's model class>(),
    df=df,
    feature_cols=FEATURE_COLS,
    label_col="target",
    trainer_module=sys.modules[__name__],
    name="trend",
    timeframe=tf,
    symbol=symbol,
)
report_path = PROJECT_ROOT / "data" / "audit" / "sprint0_validation_reports" / f"{name}_{tf}_{symbol}.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    json.dump(asdict(report), f, indent=2)
```

If `report.verdict == "kill"` and live mode requested, the trainer raises `RuntimeError("validation failed: …")` instead of writing the joblib. This stops a known-bad model from going live.

### §S0-1 success criteria
- `validate_model()` runs on every trainer without error (even if some return `kill`).
- A deliberate-leakage test fixture (a feature = `close.shift(-1)` masquerading as a real feature) is detected by `leakage_detector` with high severity.
- Embargo-on vs embargo-off Sharpe differ on at least one strategy with overlapping label horizons.
- Adversarial-validation produces non-trivial AUC (>0.5) on at least one strategy when fed deliberately mismatched train/live windows.

### §S0-1 decision gate
**If any live model raises `kill` verdict from leakage detection:** block all subsequent Sprint 0 work; fix the trainer first.
**If most models raise `kill` from adversarial validation:** flag for §S0-2 to retest with simpler models on shorter horizons.

---

## §S0-2 — Model architecture audit / bake-off (~7 days)

Two head-to-head competitions, each running through the §S0-1 harness.

### S0-2.1 Forecast-model bake-off (~4 days)

For 1s, 5s, 1m, 5m, 15m horizons, run TFT vs LightGBM vs CatBoost vs XGBoost (all with calibration) on the same data, same features, same labels.

**New file:** `src/audit/forecast_bakeoff.py`

```python
def run_forecast_bakeoff(
    symbols: list[str],
    horizons: list[str],   # ["1m", "5m", "15m"]
    feature_cols: list[str],
    output_dir: Path,
) -> BakeoffReport:
    """For each (symbol, horizon, model_kind):
      - train via walk_forward harness
      - record Sharpe, AUC, calibration ECE, drawdown
      - record training time + inference latency p50/p99
      - record leakage / adversarial findings
    Output: ranked table per (symbol, horizon).
    """
```

**Models to include:**
- Existing: TFT (`darts`), OFT (`joint_oft_rl`)
- New baselines: LightGBM, CatBoost, XGBoost (all with `CalibratedClassifierCV` wrapping)

**Output:** `data/audit/sprint0_model_bakeoff.md` — markdown table:

```
| symbol | horizon | model | walk_fwd_sharpe | auc | ECE | infer_p99_ms | verdict |
|---|---|---|---:|---:|---:|---:|---|
| BTC_USDT | 1m | LightGBM-cal | 1.34 | 0.62 | 0.04 | 1.2 | live |
| BTC_USDT | 1m | TFT          | 1.18 | 0.59 | 0.07 | 41.0 | shadow |
...
```

**Implementation approach:**
- Reuse the Phase 94 distributed backtest infrastructure: each (symbol, horizon, model_kind) cell becomes one cluster task. The orchestrator already knows how to fan out per-cell work.
- Add a new `model_type='audit_cell'` handler in `src/training/distributed/worker.py` (mirror of `_run_backtest_cell` but runs validation harness instead of backtest).

### S0-2.2 Path-optimizer bake-off (~2 days)

For order routing / sizing / timing, compare DRL (current OFT-RL) vs deterministic algorithms.

**New file:** `src/audit/path_optimizer_bakeoff.py`

Algorithms to include:
- Current: OFT-RL (`joint_oft_rl.train_oft`)
- Deterministic baselines:
  - Dijkstra (treats venues as nodes, fees+slippage as edge weights)
  - Bellman-Ford (handles negative weights, e.g. maker rebates)
  - A* (admissible heuristic = expected best-case fill)
- Naive baselines:
  - Single-venue (always Binance)
  - Round-robin

Score by realized slippage on held-out test windows + execution success %.

**Output:** `data/audit/sprint0_path_bakeoff.md`.

### S0-2.3 Hierarchy refactor proposal (~1 day, doc-only)

After S0-2.1 and S0-2.2 complete, write a 1-page markdown design proposal:

`data/audit/sprint0_hierarchy_proposal.md`

Should specify:
- The exact sequential pipeline: regime classifier (HistGBT) → forecast (winner of bake-off) → execution optimizer (winner of routing bake-off).
- Which existing model files get retired.
- Migration steps (don't execute now — that's the analytic phase).

### §S0-2 success criteria
- Both bake-offs complete with at least one passing (non-`kill`) model per symbol × horizon cell.
- The output reports rank models by walk-forward Sharpe AND highlight any model with calibration ECE > 0.05.
- The hierarchy proposal names exactly one model per layer (no parallel-voting ambiguity).

### §S0-2 decision gate
**If TFT loses to LightGBM on >75 % of cells:** TFT goes to kill list. Project loses one of its claimed differentiators — that's information, not a problem.
**If DRL loses to graph search on >75 % of cells:** DRL goes to kill list. Same logic.
**If no model passes on a (symbol, horizon) cell:** that cell is a "no-edge" cell — strategy registry should not reference it.

---

## §S0-3 — Automated kill-switch (~3 days)

### S0-3.1 Trigger evaluator (~1.5 days)

**New file:** `src/risk/kill_switch.py`

```python
@dataclass
class KillSwitchConfig:
    daily_loss_R_multiple: float = 3.0      # R = avg daily realized vol
    max_consecutive_losses: int  = 5
    latency_p99_ms_threshold: float = 500.0
    drawdown_pct_threshold: float = 0.08    # 8 %
    calibration_brier_z_threshold: float = 2.0
    rolling_window_minutes: int = 5

class KillSwitch:
    def __init__(self, cfg: KillSwitchConfig): ...
    def is_paused(self) -> bool: ...
    def evaluate(self, ts: datetime) -> tuple[bool, str | None]:
        """Check all triggers; return (paused, reason).
        Reason is None when not paused, else a human-readable trigger label."""
    def pause(self, reason: str) -> None: ...
    def reset(self, operator: str, reason: str) -> None: ...
    def state(self) -> dict: ...   # for dashboard
```

**Trigger sources** (this evaluator polls these on every tick):
- `daily_loss`: read from `src/engine/risk_aggregator` (or wherever realized PnL aggregates live)
- `consecutive_losses`: tracked in `data/risk/consecutive_losses.json` — incremented on closed losing trade, reset on closed winning trade
- `latency_p99`: rolling histogram from §S0-4 metrics collector
- `drawdown_pct`: equity peak vs current equity
- `calibration_brier_z`: from §S0-5 audit, compared to rolling 30-day baseline

### S0-3.2 Wiring into the trade loop (~0.5 day)

**Edit:** `src/main.py` — find the spot where every order goes through (likely in `engine/trading_engine.py` or `agents/execution_agent.py`):

```python
from src.risk.kill_switch import get_kill_switch

ks = get_kill_switch()
paused, reason = ks.evaluate(now)
if paused:
    logger.warning("[trade-loop] BLOCKED by kill switch: %s", reason)
    return  # don't submit any order
```

Singleton via `get_kill_switch()` so the dashboard, the trade loop, and the test suite share state.

### S0-3.3 Dashboard surface (~1 day)

**Edit:** `src/dashboard/app.py`
```python
@app.route('/api/risk/kill_switch/status')
def kill_switch_status():
    from src.risk.kill_switch import get_kill_switch
    return jsonify(get_kill_switch().state())

@app.route('/api/risk/kill_switch/reset', methods=['POST'])
def kill_switch_reset():
    body = request.get_json() or {}
    if body.get('confirm') is not True:
        return jsonify({'error': 'confirm flag required'}), 400
    from src.risk.kill_switch import get_kill_switch
    get_kill_switch().reset(operator='dashboard', reason=body.get('reason', ''))
    return jsonify({'ok': True})
```

**Edit:** `src/dashboard/templates/index.html` — Audit/Kill-switch tile shows current paused/active status, last trigger reason, last reset timestamp + operator, and a [Reset] button (with confirm flag, like the worker /restart hardening).

### §S0-3 success criteria
- Synthetic test: feed the evaluator a series of losing trades > 3R; `is_paused()` flips to `True`.
- Synthetic test: latency series with p99 > 500 ms over a 5-min window pauses.
- Reset endpoint requires `{"confirm": true}` and records operator + timestamp.
- Live trade loop visibly skips order submission when kill switch is engaged (verify in logs).

### §S0-3 decision gate
**Mandatory pass.** No live capital should run without auto kill-switch.

---

## §S0-4 — Execution-quality dashboard (~3 days)

### S0-4.1 Metrics collector (~1.5 days)

**New file:** `src/risk/execution_quality_metrics.py`

```python
class ExecutionQualityMetrics:
    """Rolling per-strategy metrics. Updated on every order event.
    Persisted to data/exec_quality/<strategy>.json on every flush."""

    def record_decision(self, strategy: str, ts: datetime): ...
    def record_submitted(self, strategy: str, ts: datetime, predicted_slippage_bps: float): ...
    def record_filled(self, strategy: str, ts: datetime, realized_slippage_bps: float, exchange: str): ...
    def record_rejected(self, strategy: str, ts: datetime, reason: str): ...   # InstitutionalGate / LLM veto / etc.
    def record_failed(self, strategy: str, ts: datetime): ...                  # network / venue error

    def snapshot(self) -> dict:
        return {
          'strategies': {
             strategy_name: {
               'latency_p50_ms': ...,
               'latency_p99_ms': ...,
               'veto_rate': vetoed / decisions,
               'exec_success_pct': filled / submitted,
               'slippage_realized_vs_predicted_bps': mean(realized - predicted),
               'slippage_by_exchange': {ex: mean(...)},
               'gas_saved_total_usd': ...,    # placeholder until DEX path lands
             },
             ...
          }
        }
```

Implementation notes:
- Use `collections.deque(maxlen=10_000)` for sliding windows (latency, slippage). Constant memory.
- Numpy `percentile` for p50/p99.
- Persistence: every 60 s (or N events) write JSON to `data/exec_quality/`. Reload on startup.

### S0-4.2 Wiring (~0.5 day)

Find every order-event hook in the codebase (`agents/execution_agent.py` is a strong candidate) and add `metrics.record_*` calls. Use a single `from src.risk.execution_quality_metrics import get_metrics` singleton.

### S0-4.3 API + dashboard tile (~1 day)

**Edit:** `src/dashboard/app.py`
```python
@app.route('/api/execution/quality')
def execution_quality():
    from src.risk.execution_quality_metrics import get_metrics
    return jsonify(get_metrics().snapshot())
```

**Edit:** `src/dashboard/templates/index.html` — Audit/Execution-quality tile renders:
- Per-strategy table (one row per strategy)
- Sparkline of latency p99 over last 1h
- Slippage real-vs-predicted bar chart (exchanges as x-axis)
- Veto rate gauge

Refresh interval: 5s (matches existing dashboard cadence).

### §S0-4 success criteria
- Every strategy in `strategy_registry.py` produces non-zero metrics within 5 minutes of bot startup.
- Latency p99 renders in the dashboard at <500 ms for paper trades on master.
- Slippage realized-vs-predicted delta is non-zero (i.e., we're actually computing the predicted side correctly).

### §S0-4 decision gate
**Soft.** If a strategy never produces metrics it gets flagged but doesn't block S0 progress.

---

## §S0-5 — Probability calibration audit (~1 day)

### S0-5.1 Calibration scanner (~0.5 day)

**New file:** `src/audit/calibration_audit.py`

```python
def audit_all_models() -> list[CalibrationReport]:
    """For every joblib in models/ that produces probabilities:
      - Load model
      - Walk forward on holdout
      - Compute reliability diagram bins
      - Compute Brier + ECE
      - Write data/audit/sprint0_calibration/<model>_calibration.json
    Returns list of reports."""
```

Cross-reference against `src/engine/agents/training_agent.py:177` (where `CalibratedClassifierCV` is already used) — confirm every model that should be calibrated IS calibrated. List any that aren't and recommend the fix.

### S0-5.2 Dashboard tile (~0.5 day)

**Edit:** `src/dashboard/app.py` — `/api/audit/calibration` endpoint reads from `data/audit/sprint0_calibration/`.

**Edit:** `src/dashboard/templates/index.html` — Audit/Calibration tile renders reliability diagrams (Chart.js or D3) per model.

### §S0-5 success criteria
- Every model issuing a probability has a calibration report.
- Models with `ECE > 0.05` are flagged in the dashboard with a warning badge.

### §S0-5 decision gate
**Soft.** Models with high ECE go to shadow; those that fail recalibration after retry go to kill.

---

## §S0-6 — MVP discipline pass / cut list (~1 day)

After S0-1 through S0-5 complete, run a single script that consumes all reports and produces the cut list.

### S0-6.1 Cut-list builder (~0.5 day)

**New file:** `src/audit/cut_list_builder.py`

```python
def build_cut_list(audit_dir: Path = Path("data/audit")) -> CutList:
    """Aggregate every ValidationReport, BakeoffReport, CalibrationReport,
    KillSwitchTest. Apply this rubric per (model × strategy):

      live   if all of: walk_forward_sharpe > 1.0
                       AND no high-severity leakage
                       AND adversarial_auc < 0.6
                       AND ECE < 0.05 (if probabilistic)
                       AND in top-3 of bake-off for its horizon

      shadow if any of: 0.5 < walk_forward_sharpe ≤ 1.0
                       OR 0.6 ≤ adversarial_auc < 0.65
                       OR 0.05 ≤ ECE < 0.10

      kill   if any of: walk_forward_sharpe ≤ 0.5
                       OR high-severity leakage
                       OR adversarial_auc ≥ 0.65
                       OR ECE ≥ 0.10
                       OR ranked last in bake-off
    """
```

Thresholds tunable via `data/audit/sprint0_cutlist_thresholds.json` if the operator wants to be more or less strict.

### S0-6.2 Output: `data/audit/sprint0_cut_list.md`

Markdown report with three tables:

```
## LIVE (validated, profitable, calibrated)
| strategy | model | symbol | tf | sharpe | drawdown | ece | adv_auc |

## SHADOW (paper-only for 30 more days)
| strategy | model | symbol | tf | reason |

## KILL (failed validation)
| strategy | model | symbol | tf | reason |
```

Plus a summary header: `Live: N  |  Shadow: M  |  Kill: K`.

### S0-6.3 Dashboard tile (~0.5 day)

**Edit:** `src/dashboard/templates/index.html` — Audit/Cut-list tile renders the three lists.

### §S0-6 success criteria
- The cut list compiles without manual editing.
- Every entry references a specific ValidationReport / BakeoffReport that supports its verdict.

### §S0-6 decision gate
**Sprint 0 is complete when the cut list exists and is reviewed by the operator.** If the operator disagrees with a verdict, the dispute is resolved by re-tuning thresholds in `sprint0_cutlist_thresholds.json` — never by manually editing the cut list.

---

---

## §S0a — Market-risk / fat-tail hardening (~5 days, post §S0)

These four items together prevent **avoidable market-side losses** that the Sprint-0 model audit + kill-switch don't cover. The kill-switch (§S0-3) reacts to losses already realized; §S0a stops orders BEFORE they're submitted when caps would be breached, and adds a tick-level circuit breaker faster than the 5-min watchdog.

### S0a-M1 — Position-sizing enforcer audit + ratchet (~1 day)

**Goal:** every order goes through a hard cap gate. No exceptions.

**Existing code to audit:** `src/risk/institutional_gate.py` (or wherever the InstitutionalGate lives) — confirm what caps are enforced today, what's missing.

**New file:** `src/risk/position_caps.py`

```python
@dataclass
class PositionCaps:
    per_trade_pct_equity:           float = 0.01    # default 1 %
    per_symbol_pct_equity:          float = 0.10    # 10 % cap on any one symbol
    total_open_exposure_pct_equity: float = 0.50    # 50 % cap across all open
    per_strategy_max_concurrent_trades: int = 5
    HARD_CEILING_per_trade_pct: float = 0.05        # immutable ceiling — never above 5 %

def validate_order_size(
    symbol: str, side: str, size_usdt: float,
    current_equity: float,
    open_positions: list[dict],
    caps: PositionCaps,
) -> tuple[bool, str]:
    """Return (ok, reason). Reason populated only on rejection.
    Checked: per-trade, per-symbol total, total open exposure, concurrent count."""
```

**Config:** `data/risk_caps.json` — operator-tunable values, schema validated on load.

**Wiring:** every order submission path imports `validate_order_size` and refuses to place if `ok=False`. Orders rejected here log a `risk_cap_breach` event to the kill-switch's input stream — repeated breaches feed §S0-3 trigger #3 (latency/operational degradation pattern).

**Success criteria:** synthetic test — submit an order at 2 % equity; gate rejects with reason "per_trade_pct_equity 0.02 > cap 0.01."

### S0a-M2 — Strategy correlation monitor (~1 day)

**New file:** `src/risk/correlation_monitor.py`

Rolling 30-day per-strategy daily P&L window → pairwise Spearman correlation matrix. Alarm when any pair > 0.7 (you think you have N diversified strategies; you actually have 1 with N hats).

**Dashboard tile:** correlation heatmap in the Audit tab (added in §0).

**Decision rule:** correlation > 0.7 → log alert, dashboard turns yellow. Doesn't auto-pause — operator decision (might be correct in regime where everything moves together).

### S0a-M3 — Leverage cap enforcer (~1 day)

**New file:** `src/risk/leverage_cap.py`

```python
@dataclass
class LeverageCaps:
    max_total_leverage:         float = 3.0
    max_per_strategy_leverage:  float = 2.0
    max_per_symbol_leverage:    float = 1.5

def evaluate_leverage(open_positions, equity, caps) -> tuple[float, str | None]:
    """Returns (current_total_leverage, breach_reason or None)."""

def deleverage_actions(open_positions, caps) -> list[dict]:
    """Returns ordered list of close-actions to bring leverage under cap.
    Strategy: close worst-performing position first."""
```

**Wiring:** kill-switch §S0-3 polls this on every tick; if breach detected → freeze new entries + execute deleverage_actions one at a time (not all at once — avoid liquidity self-impact).

### S0a-M4 — Fat-tail tick circuit breaker (~2 days)

**New file:** `src/risk/tick_circuit_breaker.py`

Faster than §S0-3's 5-min window. Runs on every price tick:

```python
class TickCircuitBreaker:
    def __init__(self, sigma_threshold: float = 4.0,
                 vol_window_bars: int = 30,
                 freeze_seconds: int = 60): ...

    def observe_tick(self, symbol: str, price: float, ts: datetime) -> bool:
        """Return True if order submission for this symbol is currently
        frozen due to a recent extreme tick."""
```

Logic:
- Maintain rolling 30-bar realized stddev per symbol (cheap, deque).
- On each tick, compute log-return vs prev tick.
- If `|log_return| > sigma_threshold * sigma` → freeze symbol for `freeze_seconds`.
- After freeze elapses: re-enable. Log every fire to `data/audit/circuit_breaker_log.jsonl`.

**Default 4σ.** A 4σ move is a ~1-in-31,000 event under normal distribution — far rarer in reality, so when it fires the market really IS in stress mode.

**Wiring:** every order submission consults `tick_breaker.observe_tick(...)` for the target symbol; refuses if frozen.

**Success criteria:** synthetic test — feed a price series with a 5σ spike; the symbol freezes; within 60s of return-to-normal, freeze releases.

### §S0a — success criteria + decision gate

- All four items wired into the order-submission path.
- Synthetic stress tests pass (rejection on per-trade cap, deleverage on leverage cap, freeze on tick spike).
- Configurable thresholds in `data/risk_caps.json` reload-able without restart.

**Mandatory pass.** No live capital should run without these four.

---

## §S0b — Operational-risk hardening (~7 days, post §S0a)

### S0b-O1 — Automated state backups (~1 day)

**New file:** `src/ops/state_backup.py`

```python
def run_backup(now: datetime, root: Path = PROJECT_ROOT,
               dest: Path = Path("D:/backups")) -> Path:
    """Tar.gz snapshot of:
      data/  (excluding data/parquet/ — too big, separate path)
      models/  (excluding models/archive/)
      config files in root
    Output: D:/backups/<YYYYMMDD_HHMMSS>/snapshot.tar.gz + manifest.json"""
```

**Schedule:** every 4 hours via existing scheduler, configurable in `data/backup_config.json`.

**Retention:**
- Hourly: keep last 24
- Daily: keep last 7
- Weekly: keep last 4
- Monthly: keep forever

Rotation script `scripts/rotate_backups.py` runs after each new backup.

**Dashboard tile:** "Last backup: 2026-05-10 14:00 UTC ✓ · Next: 18:00 UTC" in Audit tab.

### S0b-O2 — Offsite encrypted snapshot (~2 days)

**New file:** `src/ops/offsite_snapshot.py`

- AES-256-GCM encrypt the `state_backup.py` tar.gz output. Key from `.env` (`OFFSITE_BACKUP_KEY`, **never** in git).
- Provider: Backblaze B2 default ($0.005/GB/month, S3-compatible API). Configurable via `data/offsite_config.json`.
- Daily incremental + weekly full.
- Retention: weekly fulls 12 weeks, daily incrementals 14 days.

**New script:** `scripts/restore_from_offsite.py` — operator runs manually. Pulls latest snapshot, decrypts, verifies hash, extracts to `D:/restore_staging/<timestamp>/`. Operator reviews before swapping into live `data/` and `models/`.

**RUNBOOK section:** "Restore from offsite snapshot" — exact commands, expected duration, how to verify integrity.

### S0b-O3 — Restart state reconciliation (~2 days)

**New file:** `src/ops/state_reconciler.py`

```python
def reconcile_on_startup(exchange_clients: dict, local_state: dict) -> ReconciliationReport:
    """Pull live positions + open orders from every connected exchange.
    Compare to local state. Returns report with:
      - positions_match: bool
      - orphan_orders: list[dict]   (exchange-side orders we don't track)
      - phantom_positions: list[dict] (we think we have, exchange doesn't)
      - mismatched_sizes: list[dict]
    If any non-empty: bot enters RECONCILE_REQUIRED status and refuses to trade
    until operator dashboard ACKs."""
```

**Wiring:** `src/main.py` calls reconciler before any trade-loop iteration. Status surfaces in Audit tab + Telegram alert if running.

**Decision rule:** **mandatory pass.** Auto-reconcile (cancel orphans, mark phantoms closed, log) only on operator's explicit click. No silent fix-ups — every reconciliation event is auditable.

### S0b-O4 — ISP / network outage SAFE MODE (~1 day)

**New file:** `src/ops/network_health.py`

- Heartbeat to each exchange API every 30s.
- 3 consecutive failures → SAFE MODE: refuse new order submissions; existing positions unchanged.
- Pre-condition: every existing position must have an **exchange-side stop-loss** (server-side OCO order). Audit existing position-management code to ensure this is the default; if not, add to position open path.
- When network recovers: run reconciler (§S0b-O3); if clean, exit SAFE MODE.

**Dashboard tile:** network-health status per exchange (last heartbeat, consecutive fails, current mode).

### S0b-O5 — UPS / power-out runbook (~1 day, doc + light code)

**Edit:** `RUNBOOK.md` — new section:
- Recommended UPS: APC Back-UPS Pro 1500VA or equivalent (15+ min runtime under bot's load)
- Power-loss test procedure (pull plug, observe graceful shutdown, restore power, verify reconciler runs clean)
- Expected behavior: bot detects low-battery signal from UPS via `nut` (Network UPS Tools); enters SAFE MODE; saves checkpoint; flat-orders if config says so; clean shutdown.
- **Light code:** new file `src/ops/ups_monitor.py` polling `nut` if installed; degrades to no-op if not. Optional (default disabled).

**Warm-standby on Ivan worker DEFERRED** — heavy lift (active-passive replication, exchange API key sharing, conflict resolution). Not core to capital safety.

### §S0b — success criteria + decision gate

- Backup ran 4 times in the last 24h, all verified by hash.
- One offsite restore-test PASSED (extract to staging, hash compare to source).
- Synthetic test: kill the bot mid-trade; on restart, reconciler fires; bot stays in `RECONCILE_REQUIRED` until ACKed.
- Network heartbeat fails 3x → SAFE MODE engages within ~90s.

**Mandatory: O1, O3, O4 must pass before live capital. O2 strongly recommended. O5 doc-only, no gate.**

---

## §S0c — Counterparty-risk hardening (~5 days, post §S0b)

### S0c-C1 — Multi-exchange capital split (~2 days)

**Existing code to audit:** any references to "binance" in `src/exchange/`, `src/agents/`, `src/main.py`. Document which paths assume single-exchange.

**New file:** `src/risk/capital_allocator.py`

```python
@dataclass
class ExchangeAllocation:
    name:                 str            # 'binance', 'bybit', 'okx', ...
    target_pct:           float          # 0.5 = 50 % of total
    min_balance_usdt:     float          # don't drain below this
    enabled:              bool = True

def pick_exchange_for_order(strategy: str, side: str, size_usdt: float,
                             exchange_balances: dict[str, float],
                             allocations: list[ExchangeAllocation],
                             health_status: dict[str, str],  # green/yellow/red
                             ) -> str | None:
    """Returns the exchange name to route this order to, or None if all
    suitable exchanges are unhealthy / too low balance."""
```

**Config:** `data/capital_allocation.json`.

**Wiring:** strategy code calls `pick_exchange_for_order(...)` before order submission. Per-exchange health (from §S0c-C3) gates routing.

### S0c-C2 — Auto-withdraw to cold storage (~2 days)

**New file:** `src/ops/cold_storage_withdraw.py`

```python
@dataclass
class ColdStorageConfig:
    enabled:                       bool   = False    # OFF until operator whitelists
    schedule_cron:                 str    = "0 3 * * 0"   # Sun 03:00 UTC
    min_balance_floor_usdt:        float  = 5000.0
    min_withdrawal_threshold_usdt: float  = 1000.0
    addresses_by_chain:            dict   = field(default_factory=dict)

def schedule_weekly_withdrawals(cfg: ColdStorageConfig,
                                exchange_clients: dict) -> None:
    """For each exchange:
      - Skip if balance ≤ floor + threshold
      - Compute amount = balance - floor
      - Generate pre-submit alert (dashboard + Telegram)
      - Wait up to 5 min for operator dashboard ACK
      - On ACK: submit withdrawal
      - On timeout: cancel"""
```

**Audit log:** `data/cold_storage_audit.jsonl` — every event (skipped, alerted, ACKed, submitted, completed, failed).

**Dashboard tile:** Audit tab — last withdrawal, scheduled next, current floor balances per exchange.

**Hard gates:**
- Cold address must be on whitelist BEFORE any withdrawal can fire.
- Whitelist edit requires editing `data/cold_storage_config.json` directly (not via dashboard) — prevents API-key-compromise-then-redirect.
- 2FA at exchange level still applies (every withdrawal goes through exchange's 2FA).

**`enabled: false` by default** — operator sets up hardware wallet, gets a USDT-receivable address, edits config, sets `enabled: true`.

### S0c-C3 — Exchange health monitor (~1 day)

**New file:** `src/ops/exchange_health.py`

Per-exchange metrics:
- API latency p99 (rolling 5-min)
- Order acknowledgement rate (orders accepted within 2s / orders submitted)
- Withdrawal queue health (Binance/Bybit expose this; others fall back to "unknown")
- News-signal: keyword match across `data/news/*.jsonl` for `<exchange>` + `outage|halt|insolvency|paused|suspended`
- Three RED → exchange enters `quarantined` status; §S0c-C1 stops routing there

**Dashboard tile:** per-exchange status with sparklines.

### S0c-C4 — Custodian integration — DEFERRED

Document only. New section in `RUNBOOK.md`:

> Custodial cold storage (Fireblocks, Copper, Casa) is the standard for institutional-tier capital safety. It removes counterparty risk almost entirely. Cost: $5k+/month + setup fee. Cost-benefit: positive when AUM > ~$1M. For personal capital under that threshold, hardware-wallet self-custody (S0c-C2) is the right tier.

### §S0c — success criteria + decision gate

- C1: synthetic test — Binance flagged red; orders route to Bybit + OKX automatically.
- C2: dry-run with `enabled=true` and a tiny test address — pre-submit alert fires, ACK works, withdrawal submits at the exchange.
- C3: synthetic test — feed the news classifier a fake outage headline; exchange flips to YELLOW after 1 signal, RED after 3.

**C1 + C3 mandatory pass. C2 mandatory only after operator whitelists a cold-storage address. C4 deferred — doc-only.**

---

## §S0.5 — Analytic phase (~5 days, post Sprint 0)

The analytic phase **executes** the cut list. Sprint 0 is verdict-only; the analytic phase does the actual deletions.

### S0.5.1 Strategy registry trim (~1 day)
For each entry in the cut list:
- `live` → set `validation_status='live'` in `strategy_registry.py`
- `shadow` → `validation_status='shadow'`; route to paper-only execution path; do NOT include in live order generation
- `kill` → DELETE the registry entry; remove the corresponding `signal_*` column build from `_build_signals()` in `src/engine/backtester.py`; remove the trainer file from `src/engine/`; move the joblib(s) to `models/archive/`; document the kill in `data/audit/sprint0_kill_log.md` with reason

### S0.5.2 Dashboard cleanup (~1 day)
- Remove dashboard panels for killed strategies.
- Add a "Status: Live / Shadow / Killed" badge next to every remaining strategy.
- Update Stability Heatmap rendering to skip killed cells.

### S0.5.3 Trainer / orchestrator cleanup (~1 day)
- `src/orchestration/sweep_coordinator.py` PLAN_ORDER no longer includes killed models.
- `src/training/distributed/worker.py` `_MASTER_TRAINER_DISPATCH` removes killed entries.
- `data/training_rules.json` removes killed model blocks.

### S0.5.4 Test cleanup (~1 day)
- Remove `tests/test_dashboard.py` assertions for killed-strategy artifacts.
- Add new assertions: every `live` strategy in cut list has a passing ValidationReport.
- Run `tests/test_dashboard.py --offline` — confirm 0 new failures.

### S0.5.5 Documentation update (~1 day)
- `APP_DOCUMENTATION.md`: replace strategy list with the cut list.
- `RUNBOOK.md`: document the kill switch + execution-quality dashboard.
- `README.md`: update the headline strategy claims.
- New file: `MODEL_AUDIT_2026-05-10.md` — a one-page summary of the bake-off results, suitable for marketing.

### §S0.5 success criteria
- `git diff` shows N deletions for each killed strategy (consistent across registry, signals, trainer, joblib archive, test, docs).
- Test suite passes with the new assertions.
- Bot restarts cleanly (`restart_all.ps1`) with the trimmed code path; first 30 minutes show no errors related to removed strategies.

---

## §A — End-of-Sprint-0 deliverables checklist

Before declaring Sprint 0 + risk-hardening + analytic phase complete:

**§S0 — Validation core**
- [ ] `data/audit/sprint0_validation_reports/` — one per (model × symbol × tf)
- [ ] `data/audit/sprint0_model_bakeoff.md` — forecast + path bake-off rankings
- [ ] `data/audit/sprint0_calibration/` — reliability diagrams
- [ ] `data/audit/sprint0_cut_list.md` — Keep / Shadow / Kill verdicts
- [ ] `data/audit/sprint0_kill_log.md` — removed strategies + reasons
- [ ] `data/audit/sprint0_hierarchy_proposal.md` — sequential pipeline design
- [ ] Dashboard `Audit` tab populated with live data
- [ ] Kill switch (§S0-3) firing correctly under synthetic stress
- [ ] Execution-quality dashboard showing non-zero per-strategy metrics

**§S0a — Market-risk hardening**
- [ ] `data/risk_caps.json` schema + reload-able
- [ ] `src/risk/position_caps.py` wired into every order path
- [ ] `src/risk/correlation_monitor.py` + heatmap tile
- [ ] `src/risk/leverage_cap.py` + auto-deleverage
- [ ] `src/risk/tick_circuit_breaker.py` + per-symbol freeze log

**§S0b — Operational-risk hardening**
- [ ] `src/ops/state_backup.py` running on schedule, ≥4 successful runs in last 24h
- [ ] `src/ops/offsite_snapshot.py` + at least one verified restore dry-run
- [ ] `src/ops/state_reconciler.py` blocks trade-loop until clean
- [ ] `src/ops/network_health.py` SAFE-MODE switch tested
- [ ] `RUNBOOK.md` "Power outage" section + recommended UPS

**§S0c — Counterparty-risk hardening**
- [ ] `src/risk/capital_allocator.py` routing across ≥2 exchanges
- [ ] `src/ops/cold_storage_withdraw.py` (enabled=false until operator whitelists)
- [ ] `src/ops/exchange_health.py` + per-exchange status tiles
- [ ] `RUNBOOK.md` "Custodian integration" deferred-options section

**§S0.5 — Analytic phase**
- [ ] Strategy registry trimmed to Keep + Shadow only
- [ ] Killed model joblibs moved to `models/archive/`
- [ ] Dashboard panels for killed strategies removed
- [ ] Test suite green
- [ ] `restart_all.ps1` runs clean for 24 hours with the trimmed code
- [ ] `MODEL_AUDIT_2026-05-10.md` written (audit summary)

When all of these are checked, **then** the personal-use roadmap from `COMPETITIVE_ASSESSMENT_2026-05-10_v2.md` (revised Sprint 1+ from the personal-use reframe) becomes safe to begin.

---

## §B — Estimated effort summary

| Phase | Section | Days | Parallelizable? |
|---|---|---:|---|
| Sprint 0 | §0 cross-cutting setup | 1 | no |
| Sprint 0 | §S0-1 validation rigor | 5 | partial |
| Sprint 0 | §S0-2 model bake-off | 7 | **yes** (cluster-parallel via Phase 94) |
| Sprint 0 | §S0-3 kill switch | 3 | partial |
| Sprint 0 | §S0-4 exec-quality dashboard | 3 | yes (with S0-3) |
| Sprint 0 | §S0-5 calibration audit | 1 | yes (with S0-3/S0-4) |
| Sprint 0 | §S0-6 cut list | 1 | no |
| **Sprint 0a** | M1 position-sizing enforcer | 1 | yes |
| **Sprint 0a** | M2 correlation monitor | 1 | yes |
| **Sprint 0a** | M3 leverage cap enforcer | 1 | yes |
| **Sprint 0a** | M4 tick circuit breaker | 2 | yes (with O1/O2) |
| **Sprint 0b** | O1 state backups | 1 | yes |
| **Sprint 0b** | O2 offsite snapshots | 2 | yes |
| **Sprint 0b** | O3 restart reconciler | 2 | partial |
| **Sprint 0b** | O4 network outage SAFE MODE | 1 | yes |
| **Sprint 0b** | O5 UPS runbook | 1 | yes |
| **Sprint 0c** | C1 multi-exchange split | 2 | partial |
| **Sprint 0c** | C2 cold-storage withdraw | 2 | yes (with C1) |
| **Sprint 0c** | C3 exchange health monitor | 1 | yes |
| Analytic | §S0.5 cut-list execution | 5 | no |
| **TOTAL** | | **~43 days serial** / **~25–28 days with 2-PC parallelism** | |

**Parallelism note:** Sprints 0a + 0b can run concurrently (different code areas — risk vs ops). Sprint 0c depends on 0b-O4 (network health) for C3 inputs. Critical path is roughly: §0 → §S0-1 → §S0-2 → §S0-6 (~14 days) → analytic (5 days), with 0a/0b/0c overlapping the bake-off wait time.

---

---

## §C — Critical decisions you'll have to make in flight

These are the points where the plan can't decide for you:

1. **Adversarial-AUC threshold.** The defaults (0.6 warn, 0.65 block) are conservative. If they kill too many models, raise to 0.65 / 0.70.
2. **Walk-forward window size.** 60/14/14 assumes ~3 years of data. For shorter histories, drop to 30/7/7 — but then `min_folds` may not be reachable.
3. **Vol-adjusted barrier multipliers (k₁, k₂).** 1.8 / 1.2 is the standard. If labels come out too sparse (>50 % horizon hits = label 0), narrow the bands.
4. **Sharpe threshold for `live` verdict.** 1.0 is moderate-conservative. If ALL your strategies are coming out below 1.0 on walk-forward, the bar may be wrong for your TF/asset — consider 0.7.
5. **What counts as "in the cut" for the MVP.** §S0-6 ranks; you decide where the bar is. Default: top-5 by walk-forward Sharpe across all (strategy × symbol × tf) cells.
6. **Per-trade equity cap (§S0a M1).** Default 1 %. The hard ceiling 5 % is immutable; below that, operator chooses based on risk tolerance.
7. **Total exposure cap (§S0a M1).** Default 50 %. Tighter (e.g. 30 %) for stronger drawdown control; looser (70 %) only with high-confidence audited strategies.
8. **Tick circuit-breaker σ threshold (§S0a M4).** Default 4σ. If too sensitive in your asset universe (high-vol meme coins), raise to 5σ before lowering coverage.
9. **Backup retention (§S0b O1).** Default hourly×24 / daily×7 / weekly×4 / monthly×∞. Storage cost minimal at this scale; loosen only if disk pressure.
10. **Offsite provider (§S0b O2).** Backblaze B2 (cheapest), AWS S3 (most reliable), OneDrive (if you already pay M365). I'll default to B2 unless overridden in `data/offsite_config.json`.
11. **Cold-storage chain (§S0c C2).** TRC20 (Tron, low fees) vs ERC20 (Ethereum, more secure but higher gas) vs both. Operator picks based on hardware-wallet support.
12. **Exchange allocation split (§S0c C1).** Default 50/30/20 across Binance/Bybit/OKX assumes you have all three set up. If not, drop to single-exchange + queue C1 for later.

---

## §D — When to ABORT Sprint 0 and reconsider

Sprint 0 should be aborted (not paused) only if:

- §S0-1's leakage detector fires on >50 % of strategies. That's a code-base-level problem, not a strategy-level one — fix the leakage at the framework layer first.
- §S0-2 reveals that EVERY model fails on EVERY (symbol × horizon) cell. That's a data problem — investigate data quality / feature engineering before continuing.
- A kill-switch trigger fires repeatedly under normal conditions. That means a threshold is wrong; tune in `kill_switch_config.json` and continue.

Otherwise: push through. The discomfort of finding what doesn't work is exactly the point.

---

*File: TECH_IMPLEMENTATION_PLAN_2026-05-10.md — companion to COMPETITIVE_ASSESSMENT_2026-05-10_v2.md §11–§12. Sprint 0 + 0a/0b/0c risk hardening + analytic phase = ~5–7 weeks of work that gates everything in the personal-use roadmap. Personal-use reframe applied: kill switches + capital preservation prioritized over distribution / monetization features.*
