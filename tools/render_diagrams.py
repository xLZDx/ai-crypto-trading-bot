"""
Render PNG diagrams in the MetaGPT UML aesthetic (operator request 2026-05-13).

Produces every diagram in `core/diagrams/`:
  - models_hierarchy.png           — ML model taxonomy + interfaces
  - infrastructure_topology.png    — 11 long-running processes + ports + storage
  - trading_business_flow.png      — live signal → order → fill
  - training_business_flow.png     — operator click → KPI gate → retrain decision
  - agents_class_hierarchy.png     — BaseAgent + 9 concrete subclasses
  - trainer_agents_class_hierarchy.png — BaseTrainerAgent + 5 subclasses + registry
  - risk_subsystem.png             — KillSwitch + ValidationGate + DriftBaseline + ...
  - process_registry.png           — singleton enforcement state

Style match for MetaGPT: lavender fill, dark-purple header bar, +method() body
text in monospace, open-triangle inheritance arrows, dashed dependency arrows.

Run:
    python tools/render_diagrams.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).resolve().parents[1] / 'core' / 'diagrams'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MetaGPT-style palette ───────────────────────────────────────────────────
FILL_LAVENDER  = '#dcd6f7'    # box body fill (matches the screenshot)
HEADER_BAR     = '#dcd6f7'    # header bar (same hue, separated by line)
BORDER         = '#1f2937'    # near-black
TEXT_DARK      = '#0f172a'
TEXT_SECONDARY = '#475569'

# Variations for visual grouping
FILL_GREEN     = '#bbf7d0'
FILL_BLUE      = '#bfdbfe'
FILL_YELLOW    = '#fde68a'
FILL_RED       = '#fecaca'
FILL_GRAY      = '#e2e8f0'


@dataclass
class UMLBox:
    name: str
    methods: list[str] = field(default_factory=list)
    attrs:   list[str] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    width: float = 2.5
    fill: str = FILL_LAVENDER
    stereotype: Optional[str] = None   # e.g. "<<factory>>"

    @property
    def total_lines(self) -> int:
        lines = 0
        if self.stereotype:
            lines += 1
        lines += 1  # name
        lines += max(1, len(self.attrs))
        lines += max(1, len(self.methods))
        return lines

    @property
    def height(self) -> float:
        # Line height 0.32; padding 0.2 top/bottom + separator lines
        return 0.32 * self.total_lines + 0.4


def _draw_box(ax, box: UMLBox):
    """Render one UML class box with header bar + attributes + methods."""
    x, y, w, h = box.x, box.y, box.width, box.height
    # Body rectangle
    body = mpatches.FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.3, edgecolor=BORDER, facecolor=box.fill, zorder=2,
    )
    ax.add_patch(body)

    # Compute section offsets
    line_h = 0.32
    cursor = y - 0.12

    # Stereotype (optional)
    if box.stereotype:
        ax.text(
            x + w / 2, cursor, box.stereotype, ha='center', va='top',
            fontsize=8, color=TEXT_SECONDARY, fontstyle='italic', zorder=3,
        )
        cursor -= line_h * 0.85

    # Class name (bold, slightly larger)
    ax.text(
        x + w / 2, cursor, box.name, ha='center', va='top',
        fontsize=11, fontweight='bold', color=TEXT_DARK, zorder=3,
    )
    cursor -= line_h * 1.05

    # Divider line below name
    ax.add_line(Line2D(
        [x + 0.05, x + w - 0.05], [cursor + 0.05, cursor + 0.05],
        color=BORDER, linewidth=0.8, zorder=3,
    ))

    # Attrs
    if box.attrs:
        for a in box.attrs:
            ax.text(
                x + 0.12, cursor, a, ha='left', va='top',
                fontsize=8.5, color=TEXT_DARK, family='monospace', zorder=3,
            )
            cursor -= line_h * 0.95
    else:
        cursor -= line_h * 0.4

    # Divider line between attrs and methods
    ax.add_line(Line2D(
        [x + 0.05, x + w - 0.05], [cursor + 0.05, cursor + 0.05],
        color=BORDER, linewidth=0.8, zorder=3,
    ))

    # Methods
    for m in box.methods:
        ax.text(
            x + 0.12, cursor, m, ha='left', va='top',
            fontsize=8.5, color=TEXT_DARK, family='monospace', zorder=3,
        )
        cursor -= line_h * 0.95


def _draw_inheritance(ax, child: UMLBox, parent: UMLBox):
    """Open-triangle inheritance arrow (child → parent)."""
    # Bottom-center of parent (target) and top-center of child (source).
    px, py = parent.x + parent.width / 2, parent.y - parent.height
    cx, cy = child.x + child.width / 2, child.y
    # If parent is above child, arrow goes from child top → parent bottom.
    if parent.y > child.y:
        sx, sy = cx, cy
        tx, ty = px, py
    else:
        sx, sy = child.x + child.width / 2, child.y - child.height
        tx, ty = parent.x + parent.width / 2, parent.y
    ax.annotate(
        '', xy=(tx, ty), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle='-|>', color=BORDER, lw=1.2,
            mutation_scale=18,
            fc='white',  # open triangle = white fill
        ),
        zorder=1,
    )


def _draw_dependency(ax, src: UMLBox, dst: UMLBox, label: str = ''):
    """Dashed dependency arrow (src ..> dst)."""
    sx = src.x + src.width / 2
    sy = src.y - src.height / 2
    tx = dst.x + dst.width / 2
    ty = dst.y - dst.height / 2
    ax.annotate(
        '', xy=(tx, ty), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle='->', color='#64748b', lw=1.0,
            linestyle='--', mutation_scale=15,
        ),
        zorder=1,
    )
    if label:
        mx, my = (sx + tx) / 2, (sy + ty) / 2
        ax.text(mx, my, label, fontsize=7, color=TEXT_SECONDARY,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.0))


def _new_canvas(width=18.0, height=10.0, title: str = ''):
    fig, ax = plt.subplots(figsize=(width, height), dpi=130)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold',
                     color=TEXT_DARK, y=0.985)
    return fig, ax


def _save(fig, fname: str):
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=130)
    plt.close(fig)
    print(f"  [OK] {out}")
    return out


# ───────────────────────────────────────────────────────────────────────────
# DIAGRAM 1 — Models hierarchy (the operator's specific ask)
# ───────────────────────────────────────────────────────────────────────────
def render_models_hierarchy():
    fig, ax = _new_canvas(width=20.0, height=12.0,
                          title='ML Model Taxonomy — Trained models, interfaces, and inference path')

    # Parent: abstract Model interface
    parent = UMLBox(
        name='TrainedModel',
        stereotype='<<abstract interface>>',
        attrs=['+model_path: Path', '+meta_path: Path', '+last_trained: iso',
               '+features: list[str]', '+model_type: str', '+timeframe: str'],
        methods=['+fit(X, y)', '+predict(X) → array', '+predict_proba(X) → array',
                 '+verify_signature() bool'],
        x=8.0, y=11.5, width=4.0, fill=FILL_LAVENDER,
    )

    # RF-based models (tree ensemble)
    base_rf = UMLBox(
        name='BaseRFModel',
        stereotype='<<RandomForest>>',
        attrs=['+timeframes: [5m,15m,1h,4h,1d]', '+pt=2.5  sl=1.5  max_bars=12'],
        methods=['+predict_proba(X)', '+train_via TrainerBaseAgent'],
        x=0.5, y=7.0, width=3.6, fill=FILL_GREEN,
    )
    trend_rf = UMLBox(
        name='TrendRFModel',
        stereotype='<<RandomForest>>',
        attrs=['+timeframes: [15m,1h,4h,1d,1w]', '+labels: triple_barrier_long_win'],
        methods=['+predict_proba(X)', '+train_via TrainerTrendAgent'],
        x=4.6, y=7.0, width=3.6, fill=FILL_GREEN,
    )
    futures_rf = UMLBox(
        name='FuturesShortRFModel',
        stereotype='<<RandomForest>>',
        attrs=['+timeframes: [5m,15m,1h,4h,1d,1w]', '+leverage=2.0'],
        methods=['+predict_proba(X)', '+train_via TrainerFuturesAgent'],
        x=8.7, y=7.0, width=3.7, fill=FILL_GREEN,
    )

    # HistGradientBoosting models
    scalping_hgb = UMLBox(
        name='ScalpingHGBModel',
        stereotype='<<HistGradientBoosting>>',
        attrs=['+timeframe=1m', '+horizon=5 bars', '+round_trip_fee=0.08%'],
        methods=['+predict_proba(X)', '+train_via TrainerScalpingAgent'],
        x=12.9, y=7.0, width=3.7, fill=FILL_BLUE,
    )
    meta_labeler = UMLBox(
        name='MetaLabeler',
        stereotype='<<HGB + Calibrated isotonic>>',
        attrs=['+confidence_threshold=0.40', '+search_range=[0.40, 0.70]'],
        methods=['+filter(signal, feats) → PASS/BLOCK',
                 '+batch_filter()', '+train_via TrainerMetaAgent'],
        x=17.1, y=7.0, width=2.7, fill=FILL_BLUE,
    )

    # Specialized models (non-RF/HGB)
    regime = UMLBox(
        name='RegimeClassifier',
        stereotype='<<unsupervised clustering>>',
        attrs=['+regimes: {0:RANGING, 1:TRENDING, 2:VOLATILE}',
               '+size_multipliers: per_regime'],
        methods=['+classify(features) → int', '+regime_name(int) → str',
                 '+size_multiplier(int) → float'],
        x=0.5, y=2.5, width=4.0, fill=FILL_YELLOW,
    )
    oft = UMLBox(
        name='OrderFlowTransformer',
        stereotype='<<PyTorch nn.Module>>',
        attrs=['+input: L2 microstructure', '+p_move_threshold=0.50',
               '+timeframe=1m / 5m'],
        methods=['+forward(orderbook_seq)', '+predict_p_move()',
                 '+train_via joint_oft_rl.py'],
        x=5.0, y=2.5, width=4.5, fill=FILL_RED,
    )
    tft = UMLBox(
        name='TFTNeural',
        stereotype='<<Darts TFT>>',
        attrs=['+forecast_horizon: 24 bars', '+covariates: time + macro',
               '+timeframes: [1h, 4h, 1d]'],
        methods=['+fit(time_series)', '+predict() → quantile forecast',
                 '+train_via train_tft.py'],
        x=10.0, y=2.5, width=4.5, fill=FILL_RED,
    )
    mlp = UMLBox(
        name='MLPredictor',
        stereotype='<<inference wrapper>>',
        attrs=['+is_loaded: bool', '+meta: dict (from <key>_meta.json)'],
        methods=['+predict_proba(features)', '+_get_model_features()',
                 '+load_from_meta(path)'],
        x=15.0, y=2.5, width=4.5, fill=FILL_GRAY,
    )

    # Draw boxes
    for b in (parent, base_rf, trend_rf, futures_rf, scalping_hgb, meta_labeler,
              regime, oft, tft, mlp):
        _draw_box(ax, b)

    # Inheritance arrows: all concrete models → TrainedModel
    for child in (base_rf, trend_rf, futures_rf, scalping_hgb, meta_labeler,
                  regime, oft, tft):
        _draw_inheritance(ax, child, parent)

    # MLPredictor uses any model
    _draw_dependency(ax, mlp, parent, label='loads')

    # Legend
    ax.text(0.5, 0.55, 'RandomForest models', fontsize=9, color=TEXT_DARK,
            bbox=dict(facecolor=FILL_GREEN, edgecolor=BORDER, pad=3))
    ax.text(5.5, 0.55, 'HistGradientBoosting', fontsize=9, color=TEXT_DARK,
            bbox=dict(facecolor=FILL_BLUE, edgecolor=BORDER, pad=3))
    ax.text(10.0, 0.55, 'Unsupervised', fontsize=9, color=TEXT_DARK,
            bbox=dict(facecolor=FILL_YELLOW, edgecolor=BORDER, pad=3))
    ax.text(13.5, 0.55, 'Neural (PyTorch / Darts)', fontsize=9, color=TEXT_DARK,
            bbox=dict(facecolor=FILL_RED, edgecolor=BORDER, pad=3))
    ax.text(17.5, 0.55, 'Inference wrapper', fontsize=9, color=TEXT_DARK,
            bbox=dict(facecolor=FILL_GRAY, edgecolor=BORDER, pad=3))

    return _save(fig, 'models_hierarchy.png')


# ───────────────────────────────────────────────────────────────────────────
# DIAGRAM 2 — Infrastructure topology
# ───────────────────────────────────────────────────────────────────────────
def render_infrastructure_topology():
    fig, ax = _new_canvas(width=22.0, height=13.0,
                          title='Infrastructure Topology — 11 long-running processes + ports + storage')

    operator = UMLBox(
        name='Operator',
        stereotype='<<human>>',
        attrs=['+browser: Chrome', '+terminal: PowerShell'],
        methods=['+click(restart_all.ps1)', '+POST /api/control/run',
                 '+POST /api/control/trade_mode'],
        x=9.0, y=12.5, width=4.0, fill=FILL_BLUE,
    )

    # Long-running processes (Tier 1)
    monitor = UMLBox(
        name='monitor',
        stereotype='<<role>>',
        attrs=['+port: 5001', '+heartbeat: 60s'],
        methods=['+component_health_probe()'],
        x=0.3, y=8.5, width=3.0, fill=FILL_LAVENDER,
    )
    cluster_orch = UMLBox(
        name='cluster_orch',
        stereotype='<<role>>',
        attrs=['+port: 7700', '+lanes: 4'],
        methods=['+submit_task()', '+schedule_to_lane()'],
        x=3.7, y=8.5, width=3.0, fill=FILL_LAVENDER,
    )
    dashboard = UMLBox(
        name='dashboard',
        stereotype='<<role>>',
        attrs=['+port: 5000', '+session_TTL: 1h'],
        methods=['+require_api_key', '+reaper_thread()'],
        x=7.1, y=8.5, width=3.0, fill=FILL_LAVENDER,
    )
    bot = UMLBox(
        name='bot (src.main)',
        stereotype='<<role>>',
        attrs=['+trade_mode: testnet', '+watchlist: 20 symbols'],
        methods=['+heartbeat 60s', '+os._exit on registry eviction'],
        x=10.5, y=8.5, width=3.0, fill=FILL_LAVENDER,
    )
    realtime = UMLBox(
        name='realtime_db_writer',
        stereotype='<<role>>',
        attrs=['+Binance WS klines'],
        methods=['+flush_to_parquet()'],
        x=13.9, y=8.5, width=3.0, fill=FILL_LAVENDER,
    )
    orderbook = UMLBox(
        name='orderbook_collector',
        stereotype='<<role>>',
        attrs=['+depth: 20', '+speed: 100ms'],
        methods=['+publish_orderflow()'],
        x=17.3, y=8.5, width=3.0, fill=FILL_LAVENDER,
    )

    # Long-running processes (Tier 2)
    ob_writer = UMLBox(
        name='orderbook_writer',
        stereotype='<<role X1.2>>',
        attrs=['+batch: 1000', '+_MAX_BUF: 100k'],
        methods=['+writes _L2/<sym>/yyyymm=.parquet'],
        x=0.3, y=4.5, width=3.0, fill=FILL_GREEN,
    )
    watchlist_dl = UMLBox(
        name='watchlist_downloader',
        stereotype='<<role>>',
        attrs=['+archive top-up'],
        methods=['+backfill_missing()'],
        x=3.7, y=4.5, width=3.0, fill=FILL_LAVENDER,
    )
    data_orch = UMLBox(
        name='data_orchestrator',
        stereotype='<<role>>',
        attrs=['+multi-source feeds'],
        methods=['+governance()', '+schedule()'],
        x=7.1, y=4.5, width=3.0, fill=FILL_LAVENDER,
    )
    debug_sup = UMLBox(
        name='debug_supervisor',
        stereotype='<<role>>',
        attrs=['+crash detector'],
        methods=['+poll_process_ids()',
                 '+capture_log_tail()'],
        x=10.5, y=4.5, width=3.0, fill=FILL_LAVENDER,
    )
    dash_watchdog = UMLBox(
        name='dashboard_watchdog',
        stereotype='<<role>>',
        attrs=['+health_check'],
        methods=['+respawn_dashboard()'],
        x=13.9, y=4.5, width=3.0, fill=FILL_LAVENDER,
    )
    sweep_watchdog = UMLBox(
        name='sweep_watchdog',
        stereotype='<<role>>',
        attrs=['+training stall detector'],
        methods=['+respawn_stalled_trainer()'],
        x=17.3, y=4.5, width=3.0, fill=FILL_LAVENDER,
    )

    # Storage layer
    parquet_store = UMLBox(
        name='ParquetClient',
        stereotype='<<storage: D:/data>>',
        attrs=['+DuckDB + partitioned Parquet',
               '+data/parquet/{ohlcv, _L2, _NEWS}'],
        methods=['+query(sql)', '+write_ilp(lines)'],
        x=2.0, y=0.5, width=5.0, fill=FILL_YELLOW,
    )
    state_files = UMLBox(
        name='JSON state files',
        stereotype='<<storage: D:/data>>',
        attrs=['+agent_status.json', '+control.json',
               '+process_registry.json', '+training_rules.json'],
        methods=['+safe_json.read/write/transaction'],
        x=8.0, y=0.5, width=5.5, fill=FILL_YELLOW,
    )
    models_dir = UMLBox(
        name='models/*.joblib',
        stereotype='<<storage: D:/models>>',
        attrs=['+HMAC-SHA256 signed', '+<key>_meta.json siblings'],
        methods=['+joblib.dump / verify_and_load'],
        x=14.5, y=0.5, width=5.5, fill=FILL_YELLOW,
    )

    for b in (operator, monitor, cluster_orch, dashboard, bot, realtime, orderbook,
              ob_writer, watchlist_dl, data_orch, debug_sup, dash_watchdog,
              sweep_watchdog, parquet_store, state_files, models_dir):
        _draw_box(ax, b)

    # operator → restart_all → all processes (dashed)
    for proc in (monitor, cluster_orch, dashboard, bot, realtime, orderbook):
        _draw_dependency(ax, operator, proc)

    # bot / dashboard / cluster → storage
    for proc in (bot, dashboard, cluster_orch, ob_writer):
        _draw_dependency(ax, proc, parquet_store)
        _draw_dependency(ax, proc, state_files)

    # cluster_orch / bot → models
    _draw_dependency(ax, cluster_orch, models_dir, label='trains')
    _draw_dependency(ax, bot, models_dir, label='loads')

    return _save(fig, 'infrastructure_topology.png')


# ───────────────────────────────────────────────────────────────────────────
# DIAGRAM 3 — Trading business flow
# ───────────────────────────────────────────────────────────────────────────
def render_trading_business_flow():
    fig, ax = _new_canvas(width=22.0, height=12.0,
                          title='Trading Business Flow — one cycle: market tick → fill → P&L')

    # Stage 1: ingress
    ws_tick = UMLBox(
        name='Binance WebSocket',
        stereotype='<<ingress>>',
        attrs=['+klines: 1m/5m/1h/...', '+depth20@100ms'],
        methods=['+tick(symbol, price)'],
        x=0.3, y=11.0, width=3.7, fill=FILL_BLUE,
    )
    market_analyzer = UMLBox(
        name='MarketAnalyzer',
        stereotype='<<bot.py>>',
        attrs=['+state.market_data', '+regime + funding cache'],
        methods=['+handle_tick()', '+update_context()'],
        x=4.5, y=11.0, width=3.7, fill=FILL_BLUE,
    )

    # Stage 2: signal generation
    regime = UMLBox(
        name='RegimeClassifier',
        attrs=['+0=RANGING 1=TRENDING 2=VOLATILE'],
        methods=['+classify()', '+publish "regime"'],
        x=9.0, y=11.0, width=3.7, fill=FILL_YELLOW,
    )
    signal_agent = UMLBox(
        name='SignalAgent',
        stereotype='<<27 strategies>>',
        attrs=['+strategies: RSI/MACD/BB/OFI/...'],
        methods=['+compute_raw_signal()', '+meta_labeler.filter()',
                 '+publish "signal"'],
        x=13.2, y=11.0, width=4.0, fill=FILL_YELLOW,
    )

    # Stage 3: market specialists (parallel)
    spot = UMLBox(
        name='SpotAgent',
        stereotype='<<market specialist>>',
        attrs=['+conf >= 0.62', '+spot fees: 0.10%'],
        methods=['+_on_signal()', '+_get_ml_confidence()',
                 '+publish "trade_signal" market=spot'],
        x=0.5, y=7.5, width=4.5, fill=FILL_GREEN,
    )
    futures = UMLBox(
        name='FuturesAgent',
        stereotype='<<market specialist>>',
        attrs=['+conf >= 0.60', '+leverage: 2.0x'],
        methods=['+_funding_gate()', '+_liquidation_proximity_gate()',
                 '+publish "trade_signal" market=futures'],
        x=5.3, y=7.5, width=4.7, fill=FILL_GREEN,
    )
    scalping = UMLBox(
        name='ScalpingAgent',
        stereotype='<<market specialist>>',
        attrs=['+conf >= 0.65', '+max_hold: 5 bars'],
        methods=['+_compute_scalping_signal()', '+round_trip_fee check',
                 '+publish "trade_signal" market=scalping'],
        x=10.3, y=7.5, width=4.7, fill=FILL_GREEN,
    )

    # Stage 4: risk + decision
    risk = UMLBox(
        name='RiskAgent',
        stereotype='<<9-gate stack>>',
        attrs=['+capital / drawdown / Kelly',
               '+circuit / latency / liquidity / beta / LLM'],
        methods=['+_on_signal()', '+check_kill_switch_triggers()',
                 '+publish "order" (pending)'],
        x=15.3, y=7.5, width=4.7, fill=FILL_RED,
    )

    # Stage 5: execution
    execution = UMLBox(
        name='ExecutionAgent',
        stereotype='<<order router>>',
        attrs=['+TWAP threshold: 5%', '+dedup by symbol+direction'],
        methods=['+_on_order_request()', '+_open_position()',
                 '+_close_position()'],
        x=5.5, y=3.5, width=4.5, fill=FILL_LAVENDER,
    )
    binance = UMLBox(
        name='Binance Exchange',
        stereotype='<<testnet by default>>',
        attrs=['+CCXT v4', '+OrderManager.kill_switch_blocks'],
        methods=['+place_order()', '+returns fill'],
        x=10.3, y=3.5, width=4.5, fill=FILL_BLUE,
    )

    # Stage 6: persistence
    risk_pnl = UMLBox(
        name='RiskAgent._on_order_filled',
        stereotype='<<P&L feedback>>',
        attrs=['+capital += pnl', '+Kelly.record_trade'],
        methods=['+check_hard_kill()', '+update consec_losses'],
        x=15.1, y=3.5, width=5.0, fill=FILL_RED,
    )
    db_agent = UMLBox(
        name='DatabaseAgent',
        stereotype='<<persistence>>',
        attrs=['+candle / signal / trade / news buffers'],
        methods=['+_flush_loop()',
                 '+writes to Parquet via ILP'],
        x=10.3, y=0.4, width=4.5, fill=FILL_YELLOW,
    )

    for b in (ws_tick, market_analyzer, regime, signal_agent,
              spot, futures, scalping, risk, execution, binance,
              risk_pnl, db_agent):
        _draw_box(ax, b)

    # Flow arrows
    _draw_dependency(ax, ws_tick, market_analyzer, label='tick')
    _draw_dependency(ax, market_analyzer, regime)
    _draw_dependency(ax, market_analyzer, signal_agent)
    _draw_dependency(ax, signal_agent, spot, label='signal')
    _draw_dependency(ax, signal_agent, futures, label='signal')
    _draw_dependency(ax, signal_agent, scalping, label='signal')
    _draw_dependency(ax, spot, risk, label='trade_signal')
    _draw_dependency(ax, futures, risk, label='trade_signal')
    _draw_dependency(ax, scalping, risk, label='trade_signal')
    _draw_dependency(ax, risk, execution, label='order')
    _draw_dependency(ax, execution, binance, label='place')
    _draw_dependency(ax, binance, risk_pnl, label='fill')
    _draw_dependency(ax, execution, db_agent, label='log')
    _draw_dependency(ax, risk_pnl, db_agent, label='log P&L')

    return _save(fig, 'trading_business_flow.png')


# ───────────────────────────────────────────────────────────────────────────
# DIAGRAM 4 — Training business flow
# ───────────────────────────────────────────────────────────────────────────
def render_training_business_flow():
    fig, ax = _new_canvas(width=22.0, height=13.0,
                          title='Training Business Flow — operator → KPI gate → trainer → retire decision')

    # Trigger
    operator = UMLBox(
        name='Operator',
        stereotype='<<trigger>>',
        attrs=['+click "Train all" OR Auto-orchestrate',
               '+OR CIO apply_best chain'],
        methods=['+POST /api/cluster/submit'],
        x=9.0, y=12.5, width=4.0, fill=FILL_BLUE,
    )
    cluster_orch = UMLBox(
        name='ClusterOrchestrator :7700',
        stereotype='<<dispatcher>>',
        attrs=['+lanes: 4 (meta/regime, base/trend, futures/scalping, oft/tft)',
               '+heartbeat 60s'],
        methods=['+submit_task()', '+update_task("done")'],
        x=8.5, y=10.0, width=5.0, fill=FILL_LAVENDER,
    )

    # Pre-flight gates
    kpi_gate_pre = UMLBox(
        name='KPIGate.is_retired',
        stereotype='<<pre-flight #1>>',
        attrs=['+reads retired_models.json'],
        methods=['+True → reject (operator restore)'],
        x=0.5, y=7.0, width=4.2, fill=FILL_YELLOW,
    )
    mle_pre = UMLBox(
        name='MLEngineerAgent (pre)',
        stereotype='<<pre-flight #2>>',
        attrs=['+5 validators: freshness, label_imbalance,',
               '+nan_density, drift z, feature_count'],
        methods=['+validate_training_request()'],
        x=5.2, y=7.0, width=5.0, fill=FILL_YELLOW,
    )
    schedule = UMLBox(
        name='Lane scheduler',
        stereotype='<<dispatch>>',
        attrs=['+lane 0: meta+regime', '+lane 1: base+trend',
               '+lane 2: futures+scalping', '+lane 3: oft+tft (GPU)'],
        methods=['+route(model_key) → lane_id'],
        x=10.7, y=7.0, width=4.5, fill=FILL_LAVENDER,
    )

    # Trainer pipeline
    trainer = UMLBox(
        name='TrainerAgent',
        stereotype='<<concrete: meta/base/trend/futures/scalping>>',
        attrs=['+9-step pipeline'],
        methods=['+1. ParquetClient.query data',
                 '+2. feature_engineering',
                 '+3. Triple Barrier (pt=2.5, sl=1.5, mb=12)',
                 '+4. cio_overrides MERGE (schema-bounded)',
                 '+5. Walk-forward CV (60/20/20)',
                 '+6. PurgedKFold (real t1-purge)',
                 '+7. CalibratedClassifierCV + Sortino threshold',
                 '+8. HMAC-SHA256 sign joblib',
                 '+9. write <key>_meta.json'],
        x=15.7, y=7.0, width=5.5, fill=FILL_GREEN,
    )

    # Post-flight
    mle_post = UMLBox(
        name='MLEngineerAgent (post)',
        stereotype='<<post-flight>>',
        attrs=['+Bailey-LdP PSR formula',
               '+walk-forward consistency check',
               '+baseline comparison'],
        methods=['+evaluate_trained_model() → PASS/REVIEW'],
        x=0.5, y=2.7, width=5.0, fill=FILL_YELLOW,
    )
    kpi_eval = UMLBox(
        name='KPIGate.evaluate_run',
        stereotype='<<3-strike rule>>',
        attrs=['+wf_acc / win_rate / total_trades vs thresholds',
               '+per-model thresholds from training_rules.json'],
        methods=['+append to training_runs/<m>__<tf>.parquet',
                 '+if last 3 all below: RETIRE',
                 '+else: KEEP / REVIEW'],
        x=5.7, y=2.7, width=5.5, fill=FILL_RED,
    )

    # Dashboard / follow-up
    dashboard = UMLBox(
        name='Dashboard surfaces',
        stereotype='<<operator visibility>>',
        attrs=['+/api/model_comparison', '+Strong/Weak card',
               '+badges: KEEP / REVIEW / RETIRE'],
        methods=['+restore via /api/registry/<key>/restore'],
        x=11.7, y=2.7, width=4.5, fill=FILL_BLUE,
    )
    bake_off = UMLBox(
        name='BakeOff',
        stereotype='<<operator-triggered>>',
        attrs=['+rank cells by metric',
               '+cut list: keep / review / retire'],
        methods=['+run_bake_off()',
                 '+/api/bake_off'],
        x=16.7, y=2.7, width=4.5, fill=FILL_GREEN,
    )
    cio = UMLBox(
        name='CIOAgent (Optuna)',
        stereotype='<<operator-triggered>>',
        attrs=['+TPE sampler + SQLite',
               '+search: pt/sl/threshold/HPs'],
        methods=['+start_study()', '+apply_best → cio_overrides'],
        x=11.7, y=0.4, width=5.0, fill=FILL_LAVENDER,
    )

    for b in (operator, cluster_orch, kpi_gate_pre, mle_pre, schedule,
              trainer, mle_post, kpi_eval, dashboard, bake_off, cio):
        _draw_box(ax, b)

    _draw_dependency(ax, operator, cluster_orch)
    _draw_dependency(ax, cluster_orch, kpi_gate_pre)
    _draw_dependency(ax, kpi_gate_pre, mle_pre, label='not retired')
    _draw_dependency(ax, mle_pre, schedule, label='all pass')
    _draw_dependency(ax, schedule, trainer)
    _draw_dependency(ax, trainer, mle_post)
    _draw_dependency(ax, mle_post, kpi_eval)
    _draw_dependency(ax, kpi_eval, dashboard)
    _draw_dependency(ax, dashboard, bake_off, label='operator click')
    _draw_dependency(ax, bake_off, cio, label='operator click')
    _draw_dependency(ax, cio, cluster_orch, label='next retrain')

    return _save(fig, 'training_business_flow.png')


# ───────────────────────────────────────────────────────────────────────────
# DIAGRAM 5 — Agents class hierarchy (BaseAgent + 9 subclasses)
# ───────────────────────────────────────────────────────────────────────────
def render_agents_class_hierarchy():
    fig, ax = _new_canvas(width=22.0, height=11.5,
                          title='Trading Agents — BaseAgent inheritance + 9 concrete subscribers')

    parent = UMLBox(
        name='BaseAgent',
        stereotype='<<abstract bus subscriber>>',
        attrs=['+NAME: str', '+interval_sec: float', '+_running: bool',
               '+bus: AgentBus'],
        methods=['+start()', '+stop()', '+_run_cycle()',
                 '+_setup_subscriptions()', '+heartbeat()',
                 '+publish(topic, payload)'],
        x=9.0, y=11.0, width=4.0, fill=FILL_LAVENDER,
    )

    data = UMLBox(
        name='DataAgent',
        attrs=['+interval_sec=3600'],
        methods=['+fetch_candles()', '+publish_bar()'],
        x=0.3, y=6.0, width=2.6, fill=FILL_BLUE,
    )
    signal = UMLBox(
        name='SignalAgent',
        attrs=['+interval_sec=3600', '+strategies: 27'],
        methods=['+_compute_raw_signal()', '+_apply_meta_labeler()',
                 '+publish "signal"', '+publish "regime"'],
        x=3.1, y=6.0, width=3.0, fill=FILL_BLUE,
    )
    spot = UMLBox(
        name='SpotAgent',
        attrs=['+CONFIDENCE_THRESHOLD=0.62'],
        methods=['+_on_signal(msg)', '+_get_ml_confidence()',
                 '+publish "trade_signal"'],
        x=6.3, y=6.0, width=3.0, fill=FILL_GREEN,
    )
    futures = UMLBox(
        name='FuturesAgent',
        attrs=['+LEVERAGE=2.0', '+CONFIDENCE_THRESHOLD=0.60'],
        methods=['+_on_signal(msg)', '+_funding_gate()',
                 '+_liquidation_proximity_gate()'],
        x=9.5, y=6.0, width=3.2, fill=FILL_GREEN,
    )
    scalping = UMLBox(
        name='ScalpingAgent',
        attrs=['+CONFIDENCE_THRESHOLD=0.65', '+MAX_HOLD_BARS=5'],
        methods=['+_run_cycle()', '+_compute_scalping_signal()'],
        x=12.9, y=6.0, width=3.4, fill=FILL_GREEN,
    )
    risk = UMLBox(
        name='RiskAgent',
        attrs=['+capital: float', '+_kelly: KellySizer'],
        methods=['+_on_signal(msg)', '+_on_order_filled(msg)',
                 '+check_beta_neutrality()', '+_hard_kill()'],
        x=16.5, y=6.0, width=3.5, fill=FILL_RED,
    )

    # Second row
    execution = UMLBox(
        name='ExecutionAgent',
        attrs=['+TWAP_THRESHOLD_PCT=0.05'],
        methods=['+_on_order_request()',
                 '+_open_position()', '+_close_position()'],
        x=3.0, y=1.5, width=3.5, fill=FILL_LAVENDER,
    )
    db_agent = UMLBox(
        name='DatabaseAgent',
        attrs=['+HEARTBEAT_SEC=30', '+STATS_SEC=60'],
        methods=['+_flush_loop()', '+writes to Parquet via ILP',
                 '+listens 8 topics'],
        x=7.0, y=1.5, width=4.0, fill=FILL_YELLOW,
    )
    quant = UMLBox(
        name='QuantAgent',
        attrs=['+interval_sec=14400'],
        methods=['+monitor_drift()', '+monitor_divergence()',
                 '+publish_perf_alert()'],
        x=12.0, y=1.5, width=3.5, fill=FILL_LAVENDER,
    )

    for b in (parent, data, signal, spot, futures, scalping, risk,
              execution, db_agent, quant):
        _draw_box(ax, b)

    for child in (data, signal, spot, futures, scalping, risk,
                  execution, db_agent, quant):
        _draw_inheritance(ax, child, parent)

    return _save(fig, 'agents_class_hierarchy.png')


# ───────────────────────────────────────────────────────────────────────────
# DIAGRAM 6 — Trainer agents class hierarchy
# ───────────────────────────────────────────────────────────────────────────
def render_trainer_agents_class_hierarchy():
    fig, ax = _new_canvas(width=18.0, height=10.0,
                          title='Trainer Agents — BaseTrainerAgent + 5 concrete + registry factory')

    parent = UMLBox(
        name='BaseTrainerAgent',
        stereotype='<<abstract>>',
        attrs=['+MODEL_KEY: str (class attr)',
               '+last_result: dict | None'],
        methods=['+train(rules_version, n_samples_min) → tuple',
                 '+train_async() → Thread'],
        x=6.5, y=9.0, width=5.0, fill=FILL_LAVENDER,
    )

    meta = UMLBox(
        name='TrainerMetaAgent',
        stereotype='<<meta-labeler>>',
        attrs=['+MODEL_KEY="meta"', '+model: HGB+isotonic'],
        methods=['+train() → meta_labeler.joblib'],
        x=0.5, y=5.0, width=3.2, fill=FILL_BLUE,
    )
    base = UMLBox(
        name='TrainerBaseAgent',
        stereotype='<<long bias>>',
        attrs=['+MODEL_KEY="base"', '+model: RandomForest'],
        methods=['+train() → btc_rf_model.joblib + per-TF'],
        x=4.0, y=5.0, width=3.5, fill=FILL_GREEN,
    )
    trend = UMLBox(
        name='TrainerTrendAgent',
        stereotype='<<trend-following>>',
        attrs=['+MODEL_KEY="trend"', '+l2_reg in [0, 10]'],
        methods=['+train() → trend_model.joblib + per-TF'],
        x=7.8, y=5.0, width=3.5, fill=FILL_GREEN,
    )
    futures = UMLBox(
        name='TrainerFuturesAgent',
        stereotype='<<short bias>>',
        attrs=['+MODEL_KEY="futures"', '+leverage-aware labels'],
        methods=['+train() → futures_short_model + per-TF'],
        x=11.6, y=5.0, width=4.0, fill=FILL_GREEN,
    )
    scalping = UMLBox(
        name='TrainerScalpingAgent',
        stereotype='<<1m horizon>>',
        attrs=['+MODEL_KEY="scalping"', '+SMOTE oversampling'],
        methods=['+train() → scalping_model.joblib'],
        x=15.9, y=5.0, width=3.5, fill=FILL_BLUE,
    )

    registry = UMLBox(
        name='TRAINER_AGENT_REGISTRY',
        stereotype='<<dict + factory>>',
        attrs=['+keys: meta / base / trend / futures / scalping'],
        methods=['+get_trainer_agent(model_key) → BaseTrainerAgent',
                 '+unknown key → KeyError'],
        x=6.5, y=0.9, width=5.0, fill=FILL_YELLOW,
    )

    for b in (parent, meta, base, trend, futures, scalping, registry):
        _draw_box(ax, b)

    for child in (meta, base, trend, futures, scalping):
        _draw_inheritance(ax, child, parent)
        _draw_dependency(ax, registry, child, label='')

    return _save(fig, 'trainer_agents_class_hierarchy.png')


# ───────────────────────────────────────────────────────────────────────────
# DIAGRAM 7 — Risk subsystem
# ───────────────────────────────────────────────────────────────────────────
def render_risk_subsystem():
    fig, ax = _new_canvas(width=20.0, height=11.0,
                          title='Risk Subsystem — gates between signal and exchange (fail-closed by default)')

    risk_agent = UMLBox(
        name='RiskAgent',
        stereotype='<<orchestrator>>',
        attrs=['+capital / peak / drawdown', '+_kelly: KellySizer'],
        methods=['+_on_signal() → 9-gate stack',
                 '+_on_order_filled() → update Kelly'],
        x=8.0, y=10.0, width=5.0, fill=FILL_LAVENDER,
    )

    # First row of gates
    freshness = UMLBox(
        name='Data freshness gate',
        stereotype='<<bar age check>>',
        attrs=['+DATA_STALE_SEC=300'],
        methods=['+block if last bar > 5 min old'],
        x=0.3, y=6.5, width=4.0, fill=FILL_YELLOW,
    )
    latency = UMLBox(
        name='API latency gate',
        attrs=['+API_LATENCY_LIMIT_MS=500'],
        methods=['+block if p99 > 500 ms'],
        x=4.5, y=6.5, width=3.5, fill=FILL_YELLOW,
    )
    circuit = UMLBox(
        name='Circuit breaker',
        attrs=['+MAX_CONSECUTIVE_LOSSES=3'],
        methods=['+block if 3 consec losses'],
        x=8.2, y=6.5, width=3.5, fill=FILL_YELLOW,
    )
    drawdown = UMLBox(
        name='Drawdown limit',
        attrs=['+MAX_DRAWDOWN_PCT=10'],
        methods=['+hard_kill if dd > 10%'],
        x=11.9, y=6.5, width=3.5, fill=FILL_RED,
    )
    daily_loss = UMLBox(
        name='Daily loss limit',
        attrs=['+MAX_DAILY_LOSS_PCT=5'],
        methods=['+hard_kill if dl > 5%'],
        x=15.6, y=6.5, width=4.0, fill=FILL_RED,
    )

    # Second row
    liquidity = UMLBox(
        name='Liquidity proximity',
        attrs=['+LIQ_PROXIMITY_BLOCK=0.85'],
        methods=['+block near stop cluster'],
        x=0.3, y=2.8, width=3.5, fill=FILL_YELLOW,
    )
    beta = UMLBox(
        name='BetaNeutralityFilter',
        attrs=['+max_beta_exposure=1.0', '+factor=BTC/USDT'],
        methods=['+would_breach() → bool',
                 '+fail-OPEN (logged)'],
        x=4.0, y=2.8, width=4.0, fill=FILL_GREEN,
    )
    llm = UMLBox(
        name='AgenticLLM macro veto',
        attrs=['+_DECISION_TTL_S=60', '+threading.Lock + LRU 500'],
        methods=['+evaluate_trade() → APPROVED/REJECTED',
                 '+fail-OPEN on outage'],
        x=8.2, y=2.8, width=4.5, fill=FILL_GREEN,
    )
    kelly = UMLBox(
        name='KellySizer',
        attrs=['+half_kelly=True', '+window=50'],
        methods=['+size(capital, p_win, vol_scale) → USDT'],
        x=12.9, y=2.8, width=3.5, fill=FILL_BLUE,
    )

    # Bottom row
    kill_switch = UMLBox(
        name='KillSwitch',
        stereotype='<<sticky pause>>',
        attrs=['+daily_loss_R=3.0', '+max_consec=5',
               '+latency_p99=500ms', '+drawdown_pct=0.08',
               '+brier_z=2.0'],
        methods=['+paused: bool', '+check_triggers()',
                 '+manual_pause(operator, reason)',
                 '+reset(operator, reason)'],
        x=3.0, y=0.7, width=5.0, fill=FILL_RED,
    )
    validation_gate = UMLBox(
        name='ValidationGate',
        stereotype='<<pre-train>>',
        attrs=['+freshness, label_imbalance,',
               '+nan_density, drift z-score'],
        methods=['+run(model, tf, df) → dict'],
        x=8.5, y=0.7, width=4.5, fill=FILL_YELLOW,
    )
    drift_baseline = UMLBox(
        name='DriftBaseline',
        stereotype='<<snapshot>>',
        attrs=['+max_age_days=30', '+per-feature {μ, σ, q05, q95}'],
        methods=['+save_baseline()', '+load_baseline()'],
        x=13.5, y=0.7, width=4.5, fill=FILL_BLUE,
    )

    for b in (risk_agent, freshness, latency, circuit, drawdown, daily_loss,
              liquidity, beta, llm, kelly, kill_switch, validation_gate,
              drift_baseline):
        _draw_box(ax, b)

    # RiskAgent traverses gates
    for gate in (freshness, latency, circuit, drawdown, daily_loss,
                 liquidity, beta, llm, kelly):
        _draw_dependency(ax, risk_agent, gate)

    return _save(fig, 'risk_subsystem.png')


# ───────────────────────────────────────────────────────────────────────────
# DIAGRAM 8 — Process registry topology
# ───────────────────────────────────────────────────────────────────────────
def render_process_registry():
    fig, ax = _new_canvas(width=18.0, height=10.0,
                          title='Process Registry — singleton enforcement (X1.1, 2026-05-13)')

    registry = UMLBox(
        name='ProcessRegistry',
        stereotype='<<src/utils/process_registry.py>>',
        attrs=['+ZOMBIE_AGE_S=300', '+AUDIT_RING_SIZE=200',
               '+REGISTRY_PATH: data/process_registry.json'],
        methods=['+claim_role(role, by) → (ok, info)',
                 '+release_role(role, reason) → bool',
                 '+heartbeat(role) → bool',
                 '+list_active() → dict',
                 '+reap_zombies(by) → list',
                 '+get_audit_tail(n) → list'],
        x=6.5, y=9.5, width=5.0, fill=FILL_LAVENDER,
    )

    transaction = UMLBox(
        name='SafeJsonTransaction',
        stereotype='<<atomic read-modify-write>>',
        attrs=['+filelock.FileLock', '+timeout=5s'],
        methods=['+@contextmanager transaction(filepath, default)',
                 '+ALL claim/release/heartbeat ops are ATOMIC'],
        x=12.5, y=9.5, width=5.0, fill=FILL_BLUE,
    )

    # Roles claimed
    bot_role = UMLBox(
        name='bot',
        stereotype='<<role>>',
        attrs=['+pid=33012', '+by=src.main'],
        methods=['+heartbeat 60s', '+os._exit on eviction'],
        x=0.3, y=5.0, width=3.0, fill=FILL_GREEN,
    )
    dash_role = UMLBox(
        name='dashboard',
        stereotype='<<role>>',
        attrs=['+pid=26024', '+reaper inside it'],
        methods=['+heartbeat 60s', '+reap_zombies every 60s'],
        x=3.5, y=5.0, width=3.0, fill=FILL_GREEN,
    )
    co_role = UMLBox(
        name='cluster_orch',
        stereotype='<<role>>',
        attrs=['+pid=4008', '+port 7700'],
        methods=['+heartbeat 60s'],
        x=6.7, y=5.0, width=3.0, fill=FILL_GREEN,
    )
    obw_role = UMLBox(
        name='orderbook_writer',
        stereotype='<<role X1.2>>',
        attrs=['+L2 → Parquet'],
        methods=['+heartbeat 60s'],
        x=9.9, y=5.0, width=3.0, fill=FILL_GREEN,
    )

    # Dashboard endpoints
    api_get = UMLBox(
        name='GET /api/process/registry',
        stereotype='<<dashboard endpoint>>',
        attrs=['+returns: active roles + last 50 audit events'],
        methods=['+Process Registry card on Monitor tab'],
        x=2.0, y=1.5, width=6.0, fill=FILL_BLUE,
    )
    api_reap = UMLBox(
        name='POST /api/process/registry/reap',
        stereotype='<<manual reaper trigger>>',
        attrs=['+by="operator-manual"'],
        methods=['+UI: 🧹 Reap zombies button'],
        x=9.0, y=1.5, width=6.5, fill=FILL_YELLOW,
    )

    for b in (registry, transaction, bot_role, dash_role, co_role, obw_role,
              api_get, api_reap):
        _draw_box(ax, b)

    _draw_dependency(ax, registry, transaction, label='atomic R-M-W')
    for role in (bot_role, dash_role, co_role, obw_role):
        _draw_dependency(ax, role, registry, label='claim/heartbeat')
    _draw_dependency(ax, api_get, registry, label='list_active')
    _draw_dependency(ax, api_reap, registry, label='reap_zombies')

    return _save(fig, 'process_registry.png')


def main():
    print('Rendering MetaGPT-style PNG diagrams to', OUT_DIR)
    render_models_hierarchy()
    render_infrastructure_topology()
    render_trading_business_flow()
    render_training_business_flow()
    render_agents_class_hierarchy()
    render_trainer_agents_class_hierarchy()
    render_risk_subsystem()
    render_process_registry()
    print('Done.')


if __name__ == '__main__':
    main()
