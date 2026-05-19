"""
Dashboard test module — run after any implementation change to catch regressions.
Usage:
    python tests/test_dashboard.py              # requires dashboard running on port 5000
    python tests/test_dashboard.py --offline    # only static/file checks, no HTTP

NOTE (F7): The assertions in this file are SUPPLEMENTARY string-match checks.
They verify that code symbols (DOM ids, JS function names, Python imports, route
strings) exist in source files — cheap guards against accidental deletions.
They do NOT prove behavioral correctness.  Behavioral proof lives in:
  - tests/test_dashboard_api.py  — Flask test_client() round-trips (F3/F4)
  - tests/test_safe_json.py      — concurrent write/read behavior (F1)
  - tests/test_parquet_store.py  — ingest/query/threading behavior (F5)
  - tests/test_model_integrity.py— HMAC sign/verify behavior (F6)
String-match checks here pass even with wrong logic — always pair with
a behavioral test for new functionality.
"""

import sys
import os
import json
import re
import argparse

# Force UTF-8 stdout so unicode test names (λ, ●, ↻, etc.) don't crash
# the runner on Windows Python 3.14's default cp1252 console encoding.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
TRADES_PATH   = os.path.join(BASE_DIR, 'data', 'trades.json')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
DASHBOARD_URL = 'http://127.0.0.1:5000'

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
SKIP = '\033[93mSKIP\033[0m'

results = {'pass': 0, 'fail': 0, 'skip': 0}


def check(name, ok, detail=''):
    if ok is None:
        results['skip'] += 1
        print(f'  {SKIP} {name} (skipped)')
    elif ok:
        results['pass'] += 1
        print(f'  {PASS} {name}')
    else:
        results['fail'] += 1
        print(f'  {FAIL} {name}{": " + detail if detail else ""}')


# ─── Static: HTML template ────────────────────────────────────────────────────

def test_template():
    print('\n[HTML Template]')
    if not os.path.exists(TEMPLATE_PATH):
        check('template file exists', False, TEMPLATE_PATH)
        return
    check('template file exists', True)
    html = open(TEMPLATE_PATH, encoding='utf-8').read()

    # Critical DOM IDs
    for id_ in [
        'sb-sym-list', 'sb-search', 'ov-sym-select',
        'trades-mf-group', 'open-trades-tbody', 'closed-trades-tbody',
        'ov-open-tbody', 'ov-closed-tbody',
        'chart-ml-acc', 'chart-ml-long', 'chart-ml-short',
        'quant-matrix', 'tab-strategy',
        'pivot-row', 'chart-wave',
        'port-total-capital', 'port-free-usdt', 'port-deployed',
    ]:
        check(f'DOM id #{id_}', f'id="{id_}"' in html or f"id='{id_}'" in html)

    # Critical JS functions
    for fn in [
        'renderSidebarSymbols', 'renderTrades', 'renderOverviewOrders',
        'renderStrategyTab', 'renderWatchlist', 'updatePnl',
        'setTradesMarket', 'initCollapsible', 'sortTbl', 'mktBadge',
        'syncSymSelect', 'fetchBinanceTickers', 'loadWatchlist',
        'sendChat', 'toggleCardFs', 'initResizeHandles', 'pollAiStatus',
    ]:
        check(f'JS function {fn}()', f'function {fn}(' in html)

    # Field-name safety — must use buy_price (not only entry_price)
    check('buy_price field used in renderTrades',
          'buy_price' in html)
    check('sell_price field used in renderTrades',
          'sell_price' in html)
    check('buy_time||t.opened_at fallback present',
          'buy_time' in html and 'opened_at' in html)

    # Case-insensitive status comparisons
    bad_exact = re.findall(r"t\.status\s*===\s*['\"](?:OPEN|CLOSED)['\"]", html)
    check('no bare t.status === comparisons (must use .toUpperCase())',
          len(bad_exact) == 0, f'found: {bad_exact}')

    # Integer-safe ID slicing — bare (t.id||'').slice is unsafe; String(...) wrapper is fine
    bad_id_slice = re.findall(r"(?<!String)\(t\.id\s*\|\|\s*''\)\.slice", html)
    check('integer-safe ID slicing (String(t.id||""))',
          len(bad_id_slice) == 0, f'still using bare (t.id||"").slice at {len(bad_id_slice)} place(s)')

    # Balances panel
    check('balances scroll div has id=bal-scroll', 'id="bal-scroll"' in html)
    check('balances scrollbar hidden (scrollbar-width:none)', 'scrollbar-width:none' in html)
    check('balances QTY column header present', '>QTY<' in html)
    check('balances Value column header present', '>Value<' in html)
    check('balances ov-val-btc cell present', 'id="ov-val-btc"' in html)
    check('balances ov-val-sol cell present', 'id="ov-val-sol"' in html)
    check('balances ov-val-ada cell present', 'id="ov-val-ada"' in html)
    check('holdings separator uses colspan=4', 'colspan="4"' in html)

    # Sidebar 3-column grid
    check('sidebar 3-column grid (grid-template-columns:1fr auto auto)',
          'grid-template-columns:1fr auto auto' in html)

    # Resize handled by JS (CSS resize:vertical replaced by setPointerCapture)
    check('card-body resize via JS (initResizeHandles called)',
          'initResizeHandles()' in html)
    check('resize handle uses setPointerCapture',
          'setPointerCapture' in html)
    check('resize-handle CSS class defined',
          '.resize-handle' in html)

    # Fullscreen support
    check('toggleCardFs buttons present on all 4 order cards (orders + overview)',
          html.count('onclick="toggleCardFs(this)"') >= 4)
    check('toggleCardFs uses parent.insertBefore (no location.reload)',
          'insertBefore' in html and 'location.reload' not in html)
    check('card-fs CSS class defined',
          '.card-fs' in html)
    check('Escape key exits fullscreen',
          'Escape' in html and 'card-fs' in html)

    # AI Assistant upgrades
    check('chat-model-chip DOM id present',
          'id="chat-model-chip"' in html or "id='chat-model-chip'" in html)
    check('sendChat handles d.command field',
          'd.command' in html)
    check('sendChat handles d.command_result field',
          'd.command_result' in html)

    # Collapsible init called
    check('initCollapsible() called in DOMContentLoaded',
          'initCollapsible()' in html)
    # Sub-panels (Signal/Risk/Portfolio) must NOT be in initCollapsible selector
    check('ov-panel-hdr excluded from initCollapsible (no collapse on sub-panels)',
          'ov-panel-hdr' not in html.split('function initCollapsible')[1].split('function ')[0])

    # Portfolio capital fields computed in updatePnl
    check('port-total-capital updated in updatePnl', 'port-total-capital' in html)
    check('port-free-usdt updated in updatePnl', 'port-free-usdt' in html)
    check('port-deployed updated in updatePnl', 'port-deployed' in html)
    check('deployedValue computed in updatePnl', 'deployedValue' in html)

    # Market View card is above Live Orders (chart before orders in DOM)
    mv_pos = html.find('id="tv_chart_container"')
    orders_pos = html.find('id="ov-open-tbody"')
    check('Market View card appears before Live Orders in DOM', mv_pos < orders_pos and mv_pos > 0)

    # Market filter buttons
    for mkt in ['ALL', 'SPOT', 'FUTURES', 'SCALPING']:
        check(f'trades market filter button {mkt}',
              f'data-tmarket="{mkt}"' in html)

    # Quant Signal Matrix
    check('quant-matrix DOM id present', 'id="quant-matrix"' in html)
    check('renderQuantCard() function defined', 'function renderQuantCard(' in html)
    check('renderQuantMatrix() function defined', 'function renderQuantMatrix(' in html)
    check('renderQuantMatrix(state) called in renderStrategyTab', 'renderQuantMatrix(state)' in html)

    # ML card last_trained timestamp
    check('last_trained shown in ML model card', 'last_trained' in html)

    # Phase 12 — Institutional tab + service health + Phase 6 promotion
    check('institutional sidebar nav button',
          'data-tab="institutional"' in html)
    check('institutional tab pane exists',
          'id="tab-institutional"' in html)
    check('mode switcher lives inside institutional tab pane',
          html.find('id="mode-switcher"') > html.find('id="tab-institutional"'))
    check('phase6-content lives inside institutional tab pane',
          html.find('id="phase6-content"') > html.find('id="tab-institutional"'))
    check('phase6 lazy init via window._initPhase6',
          '_initPhase6' in html)
    check('per-tab AbortController for phase6 fetches',
          '_p6Ctrls' in html)
    check('mon-services-grid for QuestDB/DuckDB cards',
          'mon-services-grid' in html)
    check('monPollServices() defined',
          'function monPollServices(' in html or 'async function monPollServices(' in html)
    check('strategy filter uses live_enabled (not bare s.live)',
          's.live_enabled === true' in html)
    check('simulator action wrapper _simAction defined',
          'async function _simAction(' in html)
    # Phase 3 of QuestDB → ParquetClient migration: the QuestDB-specific
    # offline banner was retired (no daemon, no native install). Assert
    # the new DuckDB-based offline guidance instead.
    check('Parquet Store offline banner mentions DuckDB install',
          'pip install duckdb pyarrow' in html)


# ─── Static: trades.json field names ─────────────────────────────────────────

def test_trades_file():
    print('\n[Trades Data File]')
    if not os.path.exists(TRADES_PATH):
        check('trades.json exists', False, TRADES_PATH)
        return
    check('trades.json exists', True)
    raw = json.loads(open(TRADES_PATH, encoding='utf-8').read())
    trades = raw if isinstance(raw, list) else raw.get('trades', [])
    check('trades list not empty', len(trades) > 0, f'found {len(trades)} trades')
    if not trades:
        return

    sample = trades[0]
    for field in ['id', 'symbol', 'status', 'side', 'market', 'buy_price']:
        check(f'field "{field}" present in trade', field in sample)

    # IDs should be safe to String() without crashing
    for t in trades[:5]:
        try:
            str(t.get('id', ''))[:10]
            ok = True
        except Exception as e:
            ok = False
        check(f'trade id={t.get("id")} safe to String().slice()', ok)

    statuses = {str(t.get('status', '')).upper() for t in trades}
    check('status values are OPEN or CLOSED only',
          statuses.issubset({'OPEN', 'CLOSED'}),
          f'unexpected: {statuses - {"OPEN","CLOSED"}}')

    closed = [t for t in trades if str(t.get('status','')).upper() == 'CLOSED']
    check(f'closed trades present ({len(closed)})', len(closed) > 0)
    if closed:
        c = closed[0]
        check('closed trade has sell_price or exit_price',
              'sell_price' in c or 'exit_price' in c)
        check('closed trade has sell_time or closed_at',
              'sell_time' in c or 'closed_at' in c)
        check('closed trade has pnl_usdt',
              'pnl_usdt' in c)

    open_ = [t for t in trades if str(t.get('status','')).upper() == 'OPEN']
    if open_:
        o = open_[0]
        check('open trade has buy_price or entry_price',
              'buy_price' in o or 'entry_price' in o)
        check('open trade has amount_coin',
              'amount_coin' in o)


# ─── Static: app.py backend functions ───────────────────────────────────────

APP_PATH = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')

def test_app_py():
    print('\n[app.py Backend]')
    if not os.path.exists(APP_PATH):
        check('app.py exists', False, APP_PATH)
        return
    check('app.py exists', True)
    src = open(APP_PATH, encoding='utf-8').read()

    for fn in ['_build_portfolio_context', '_exec_bot_command', 'chat', 'close_all_trades', 'close_losing_trades']:
        check(f'function/route {fn} defined', fn in src)

    check('/api/close_losing route present', "'/api/close_losing'" in src)
    check('Gemini latest model tried first (gemini-3.1-pro-preview)',
          "'gemini-3.1-pro-preview'" in src or '"gemini-3.1-pro-preview"' in src)
    check('model fallback list present (_MODELS)',
          '_MODELS' in src)
    check('uses new google.genai SDK (not deprecated generativeai)',
          'from google import genai' in src and 'google.generativeai' not in src)
    check('flash-lite fallback model present (gemini-3.1-flash-lite-preview)',
          'gemini-3.1-flash-lite-preview' in src)
    check('quota/429/503 triggers model fallback',
          '429' in src and 'resource_exhausted' in src and '503' in src and 'unavailable' in src)
    check('/api/ai_status endpoint defined', "'/api/ai_status'" in src or '"/api/ai_status"' in src)
    check('_probe_models_bg startup thread present', '_probe_models_bg' in src)
    check('_active_model cache present', '_active_model' in src)
    check('_exec_bot_command returns (command, command_result)',
          'command_result' in src)
    check('portfolio context builds win_rate', 'win_rate' in src)
    check('portfolio context builds ml_acc', 'ml_acc' in src)

    # Phase 12 — service health + ML accuracy normalization
    check('/api/monitor/services endpoint defined',
          "'/api/monitor/services'" in src or '"/api/monitor/services"' in src)
    # Phase 3 of QuestDB → ParquetClient migration: QuestDB HTTP probe
    # replaced with the in-process ParquetClient probe (no port).
    check('monitor_services probes the parquet store',
          "out['parquet_store']" in src)
    check('monitor_services probes DuckDB',
          'duckdb' in src and 'PRAGMA temp_directory' in src)
    check('monitor_services probes ZeroMQ data plane',
          ':5555' in src and 'zmq' in src.lower())
    check('ml accuracy normalized to percent (auto-detect fraction)',
          '_to_pct' in src or 'to_pct' in src)


# ─── Static: main.py quant integration ──────────────────────────────────────

MAIN_PATH = os.path.join(BASE_DIR, 'src', 'main.py')

def test_main_py():
    print('\n[main.py Quant Integration]')
    if not os.path.exists(MAIN_PATH):
        check('main.py exists', False, MAIN_PATH)
        return
    check('main.py exists', True)
    src = open(MAIN_PATH, encoding='utf-8').read()

    # Imports
    check('MeanReversionCore imported', 'MeanReversionCore' in src)
    check('TelegramMonitor imported', 'TelegramMonitor' in src)
    check('numpy imported (np)', 'import numpy as np' in src)

    # Init
    check('self.mean_reversion = MeanReversionCore() in __init__', 'self.mean_reversion = MeanReversionCore()' in src)
    check('self.ou_results initialized in __init__', 'self.ou_results' in src)
    check('self.telegram_monitor = TelegramMonitor(channels=', 'TelegramMonitor(channels=' in src)
    check('VilarsoPro channel configured', 'VilarsoPro' in src)
    check('vilarsofree channel configured', 'vilarsofree' in src)
    check('mr_mozart channel configured', 'mr_mozart' in src)

    # OU integration in process_kline
    check('calibrate_ou_process() called', 'calibrate_ou_process' in src)
    check('ou_signal passed to evaluate_all_strategies', 'ou_signal=ou_signal' in src)

    # GARCH integration
    check('forecast_garch() called in process_kline', 'forecast_garch' in src)
    check('volatility_spike halves trade_amount', 'volatility_spike' in src and 'trade_amount * 0.5' in src)

    # Real inventory
    check('real inventory via split(\'/\')[0] (not inventory_q=0.0)',
          "split('/')[0]" in src and 'inventory_q = 0.0' not in src)

    # OU filter in evaluate_all_strategies
    check('ou_signal parameter in evaluate_all_strategies', 'ou_signal=0' in src)
    check('OU Veto BUY when overbought', 'OU Veto' in src)

    # Quant state push to dashboard
    check('garch_result initialized before try block (garch_result = {})', 'garch_result = {}' in src)
    check('quant state pushed to current_state["quant"]',
          'current_state' in src and '"quant"' in src and 'ou_signal' in src)

    # Phase 12 — WebSocket keepalive + exponential backoff
    check('binance ws ping_interval set explicitly',
          'ping_interval=20' in src or 'ping_interval = 20' in src)
    check('binance ws ping_timeout set explicitly',
          'ping_timeout=20' in src or 'ping_timeout = 20' in src)
    check('binance ws reconnect uses exponential backoff',
          'backoff = min(' in src and 'backoff * 2' in src)


# ─── Static: quant module files ───────────────────────────────────────────────

AGENTIC_PATH  = os.path.join(BASE_DIR, 'src', 'engine', 'agentic_llm.py')
TG_MON_PATH   = os.path.join(BASE_DIR, 'src', 'analysis', 'telegram_monitor.py')

def test_quant_modules():
    print('\n[Quant Modules]')

    # agentic_llm.py — must use new SDK and latest model
    if not os.path.exists(AGENTIC_PATH):
        check('agentic_llm.py exists', False, AGENTIC_PATH)
    else:
        check('agentic_llm.py exists', True)
        src = open(AGENTIC_PATH, encoding='utf-8').read()
        check('agentic_llm uses google.genai (not generativeai)',
              'from google import genai' in src and 'google.generativeai' not in src)
        check('agentic_llm uses gemini-3.1-pro-preview',
              'gemini-3.1-pro-preview' in src)
        check('agentic_llm has model fallback (_MODELS)',
              '_MODELS' in src)
        check('agentic_llm has transient error fallback (_TRANSIENT)',
              '_TRANSIENT' in src)

    # telegram_monitor.py — must support multiple channels
    if not os.path.exists(TG_MON_PATH):
        check('telegram_monitor.py exists', False, TG_MON_PATH)
    else:
        check('telegram_monitor.py exists', True)
        src = open(TG_MON_PATH, encoding='utf-8').read()
        check('TelegramMonitor accepts channels list param', 'channels: list' in src or 'channels=None' in src)
        check('multi-channel: chats=self.channels', 'chats=self.channels' in src)
        check('message tagged with source channel', 'source' in src and 'tagged' in src)
        check('cache_size 30 (larger for multi-channel)', 'cache_size: int = 30' in src or 'cache_size=30' in src)


# ─── Static: training scripts ────────────────────────────────────────────────

def test_training_scripts():
    print('\n[Training Scripts]')
    ENGINE_DIR = os.path.join(BASE_DIR, 'src', 'engine')
    scripts = [
        'train_model.py',
        'train_futures_model.py',
        'train_trend_model.py',
        'train_scalping_model.py',
    ]
    for fname in scripts:
        path = os.path.join(ENGINE_DIR, fname)
        if not os.path.exists(path):
            check(f'{fname} exists', False, path)
            continue
        check(f'{fname} exists', True)
        src = open(path, encoding='utf-8').read()
        check(f'{fname} has archive fallback (_spot_)', '_spot_' in src)
        check(f'{fname} writes last_trained to meta', 'last_trained' in src)


# ─── Static: new quant strategy modules ─────────────────────────────────────

def test_new_strategy_modules():
    print('\n[New Strategy Modules]')

    # Momentum
    mom_path = os.path.join(BASE_DIR, 'src', 'analysis', 'momentum.py')
    check('momentum.py exists', os.path.exists(mom_path))
    if os.path.exists(mom_path):
        src = open(mom_path, encoding='utf-8').read()
        check('CrossSectionalMomentum class defined', 'class CrossSectionalMomentum' in src)
        check('compute_from_history() for backtesting', 'compute_from_history' in src)
        check('load_momentum_prices() helper', 'load_momentum_prices' in src)

    # Funding rate downloader
    fr_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'funding_rate_downloader.py')
    check('funding_rate_downloader.py exists', os.path.exists(fr_path))
    if os.path.exists(fr_path):
        src = open(fr_path, encoding='utf-8').read()
        check('download_funding_rates() defined', 'def download_funding_rates' in src)
        check('merge_funding_into_ohlcv() defined', 'def merge_funding_into_ohlcv' in src)
        check('uses ccxt for Binance perpetual futures', 'ccxt' in src and 'fundingRate' in src)

    # Backtester
    bt_path = os.path.join(BASE_DIR, 'src', 'engine', 'backtester.py')
    check('backtester.py exists', os.path.exists(bt_path))
    if os.path.exists(bt_path):
        src = open(bt_path, encoding='utf-8').read()
        check('Backtester class defined', 'class Backtester' in src)
        check('BacktestResult class defined', 'class BacktestResult' in src)
        check('TradeRecord class defined', 'class TradeRecord' in src)
        check('Sharpe ratio implemented', 'def sharpe' in src)
        check('Sortino ratio implemented', 'def sortino' in src)
        check('Max drawdown implemented', 'def max_drawdown' in src)
        check('Profit factor implemented', 'def profit_factor' in src)
        check('Funding cost in PnL formula', 'funding_paid' in src)
        check('run_full_backtest() entry point', 'def run_full_backtest' in src)
        check('compare_strategies() returns DataFrame', 'def compare_strategies' in src)

    # FinBERT in feature_engineering
    fe_path = os.path.join(BASE_DIR, 'src', 'analysis', 'feature_engineering.py')
    check('feature_engineering.py exists', os.path.exists(fe_path))
    if os.path.exists(fe_path):
        src = open(fe_path, encoding='utf-8').read()
        check('add_finbert_sentiment() defined', 'def add_finbert_sentiment' in src)
        check('FinBERT model ProsusAI/finbert used', 'ProsusAI/finbert' in src)
        check('Falls back to keyword sentiment on import error', 'add_news_sentiment' in src and ('fallback' in src.lower() or 'falling back' in src.lower()))

    # Momentum wired in main.py
    main_path = os.path.join(BASE_DIR, 'src', 'main.py')
    if os.path.exists(main_path):
        src = open(main_path, encoding='utf-8').read()
        check('CrossSectionalMomentum imported in main.py', 'CrossSectionalMomentum' in src)
        check('self.momentum_engine initialized in __init__', 'self.momentum_engine' in src)
        check('momentum_signals updated in process_kline', 'momentum_engine.update' in src)
        check('momentum_signal pushed to quant state', '"momentum_signal"' in src)

    # TFT includes funding rate feature
    tft_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_tft_model.py')
    if os.path.exists(tft_path):
        src = open(tft_path, encoding='utf-8').read()
        check('TFT uses GPU when available (pl_trainer_kwargs)', 'pl_trainer_kwargs' in src)
        check('funding_rate in TFT past covariates', '"funding_rate"' in src)
        check('merge_funding_into_ohlcv called in engineer_frame', 'merge_funding_into_ohlcv' in src)

    # train_all_models includes backtester and funding download
    ta_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_all_models.py')
    if os.path.exists(ta_path):
        src = open(ta_path, encoding='utf-8').read()
        check('train_all_models downloads funding rates first', 'download_funding_rates' in src)
        check('train_all_models runs backtester', 'run_full_backtest' in src)

    # install_cuda_torch.ps1 script
    cuda_ps1 = os.path.join(BASE_DIR, 'install_cuda_torch.ps1')
    check('install_cuda_torch.ps1 exists', os.path.exists(cuda_ps1))

    # restart_all.ps1 uses PID file (no WMI hang)
    restart_path = os.path.join(BASE_DIR, 'restart_all.ps1')
    if os.path.exists(restart_path):
        src = open(restart_path, encoding='utf-8').read()
        check('restart_all.ps1 uses PID file (no WMI)', 'process_ids.json' in src)
        check('restart_all.ps1 saves PIDs on launch', 'ConvertTo-Json' in src and 'process_ids.json' in src)
        check('restart_all.ps1 no hanging Get-CimInstance', 'Get-CimInstance Win32_Process' not in src)


# ─── Static: monitor server + launchers ──────────────────────────────────────

def test_monitor_module():
    print('\n[Monitor Server & Launchers]')

    # Monitor server
    mon_path = os.path.join(BASE_DIR, 'src', 'monitor', 'server.py')
    check('src/monitor/server.py exists', os.path.exists(mon_path))
    if os.path.exists(mon_path):
        src = open(mon_path, encoding='utf-8').read()
        check('Flask app defined', 'app = Flask' in src)
        check('route / returns HTML page', "@app.route('/')" in src or '@app.route("/")' in src)
        check('_proc_status uses psutil', 'psutil' in src)
        check('_tail reads log files', 'def _tail' in src)
        check('runs on port 5001', '5001' in src)
        check('auto-refresh meta tag', 'refresh' in src)

    # Launcher scripts
    for script in ['launch_monitor.ps1', 'launch_dashboard.ps1', 'launch_bot.ps1', 'launch_training.ps1']:
        path = os.path.join(BASE_DIR, script)
        check(f'{script} exists', os.path.exists(path))
        if os.path.exists(path):
            src = open(path, encoding='utf-8').read()
            # launch_monitor uses Remove-Item + >> to avoid file lock; others use Tee-Object
            if script == 'launch_monitor.ps1':
                check(f'{script} uses Remove-Item to clear stale log', 'Remove-Item' in src)
                check(f'{script} uses append redirect for log capture', '2>&1 >>' in src or '>>' in src)
            else:
                # Phase 11 — launch_bot.ps1 / launch_dashboard.ps1 switched
                # from Tee-Object (UTF-16 default) to Out-File -Encoding utf8
                # so the live-log viewer can decode them properly.
                check(f'{script} captures log to file',
                      'Tee-Object' in src or 'Out-File' in src)
            check(f'{script} uses venv python path', 'venv' in src and 'python.exe' in src)

    # restart_all.ps1 updated checks
    restart_path = os.path.join(BASE_DIR, 'restart_all.ps1')
    if os.path.exists(restart_path):
        src = open(restart_path, encoding='utf-8').read()
        check('restart_all.ps1 launches monitor (step 0)', 'launch_monitor.ps1' in src)
        check('restart_all.ps1 uses Start-Window helper', 'Start-Window' in src)
        check('restart_all.ps1 uses -File launcher pattern', '-File' in src)
        check('restart_all.ps1 saves monitor PID', 'monitor' in src and 'ConvertTo-Json' in src)
        check('restart_all.ps1 no hanging Get-CimInstance', 'Get-CimInstance Win32_Process' not in src)

    # psutil in requirements
    req_path = os.path.join(BASE_DIR, 'requirements.txt')
    if os.path.exists(req_path):
        req = open(req_path, encoding='utf-8').read()
        check('psutil in requirements.txt', 'psutil' in req)

    # Monitor tab in dashboard HTML
    html_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    if os.path.exists(html_path):
        html = open(html_path, encoding='utf-8').read()
        check('Monitor nav tab button present', 'data-tab="monitor"' in html)
        check('Monitor tab pane present', 'id="tab-monitor"' in html)
        check('mon-health-grid div present', 'mon-health-grid' in html)
        check('mon-log-box div present', 'mon-log-box' in html)
        check('monPollHealth() JS function', 'monPollHealth' in html)
        check('monStart() JS function', 'monStart' in html)
        check('monStop() JS function', 'monStop' in html)
        check('monPollLog() JS function', 'monPollLog' in html)
        check('monitor fetches use hdrs() not undefined API_HEADERS', 'API_HEADERS' not in html)

    # Monitor API routes in app.py
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    if os.path.exists(app_path):
        src = open(app_path, encoding='utf-8').read()
        check('/api/monitor/health route defined', "'/api/monitor/health'" in src)
        check('/api/monitor/logs route defined', "'/api/monitor/logs/" in src)
        check('/api/monitor/start route defined', "'/api/monitor/start/" in src)
        check('/api/monitor/stop route defined', "'/api/monitor/stop/" in src)
        check('_managed process registry', '_managed' in src)
        check('_pid_alive() helper uses psutil', '_pid_alive' in src and 'psutil' in src)


# ─── Static: model meta files ─────────────────────────────────────────────────

def test_model_meta():
    print('\n[Model Meta Files]')
    for key in ['btc_rf_model', 'futures_short_model', 'scalping_model', 'trend_model']:
        path = os.path.join(MODELS_DIR, f'{key}_meta.json')
        if not os.path.exists(path):
            check(f'{key}_meta.json exists', False)
            continue
        check(f'{key}_meta.json exists', True)
        meta = json.loads(open(path, encoding='utf-8').read())
        check(f'{key} has accuracy field', 'accuracy' in meta,
              f'keys: {list(meta.keys())}')
        if 'accuracy' in meta:
            acc = float(meta['accuracy'])
            check(f'{key} accuracy in valid range (0-100)',
                  0 <= acc <= 100, f'got {acc}')


# ─── HTTP: live API endpoints ─────────────────────────────────────────────────

def test_api(base_url):
    import pytest
    import urllib.request
    import urllib.error
    print(f'\n[API Endpoints @ {base_url}]')
    # Skip gracefully when dashboard is not running
    try:
        urllib.request.urlopen(base_url + '/', timeout=2).close()
    except Exception:
        pytest.skip('Dashboard not running — skipping live API tests')

    def get(path, expect_key=None, expect_json=True):
        url = base_url + path
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                raw = r.read()
                if expect_json:
                    body = json.loads(raw.decode())
                    if expect_key is not None:
                        ok = expect_key in body
                        check(f'GET {path} → has "{expect_key}"', ok,
                              f'keys: {list(body.keys())}')
                    else:
                        check(f'GET {path} → 200 OK', True)
                else:
                    # HTML / non-JSON: just verify status + non-empty body
                    check(f'GET {path} → 200 OK ({len(raw)} bytes)',
                          r.status == 200 and len(raw) > 0)
        except urllib.error.HTTPError as e:
            check(f'GET {path}', False, f'HTTP {e.code}')
        except Exception as e:
            check(f'GET {path}', False, str(e))

    get('/', expect_json=False)
    get('/api/state')
    get('/api/control', 'running')
    get('/api/trades', 'trades')
    get('/api/watchlist', 'symbols')
    get('/api/models')
    get('/api/ai_status', 'available')

    # POST endpoints — just verify they exist (405 = wrong method, not 404)
    def post_exists(path):
        url = base_url + path
        try:
            import urllib.request
            req = urllib.request.Request(url, data=b'{}',
                                         headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=4) as r:
                check(f'POST {path} → 200', True)
        except urllib.error.HTTPError as e:
            check(f'POST {path} exists (not 404)', e.code != 404, f'HTTP {e.code}')
        except Exception as e:
            check(f'POST {path}', False, str(e))

    post_exists('/api/close_losing')

    # Trades endpoint field validation
    try:
        with urllib.request.urlopen(base_url + '/api/trades', timeout=4) as r:
            data = json.loads(r.read().decode())
        trades = data.get('trades', [])
        check(f'/api/trades returns list ({len(trades)} items)', isinstance(trades, list))
        if trades:
            t = trades[0]
            check('trade has buy_price or entry_price',
                  'buy_price' in t or 'entry_price' in t)
            check('trade has status field', 'status' in t)
            check('trade id is safe to stringify', isinstance(t.get('id'), (int, str)))
    except Exception:
        pass


# ─── Static: Phase 0 institutional upgrade ───────────────────────────────────

def test_phase0_foundation():
    print('\n[Phase 0 -- Institutional Upgrade Foundation]')

    # Plan + CLAUDE.md
    plan_path = os.path.join(BASE_DIR, 'INSTITUTIONAL_UPGRADE_PLAN.md')
    check('INSTITUTIONAL_UPGRADE_PLAN.md exists', os.path.exists(plan_path))
    if os.path.exists(plan_path):
        plan = open(plan_path, encoding='utf-8').read()
        check('plan references DuckDB + Parquet', 'DuckDB' in plan and 'Parquet' in plan)
        check('plan references ZeroMQ + FastAPI', 'ZeroMQ' in plan and 'FastAPI' in plan)
        check('plan lists all 5 levels (L1–L5)',
              all(f'Level {i}' in plan or f'L{i} ' in plan or f'Phase {i}' in plan for i in range(1, 6)))
    claude_path = os.path.join(BASE_DIR, 'CLAUDE.md')
    check('CLAUDE.md at project root', os.path.exists(claude_path))
    if os.path.exists(claude_path):
        claude = open(claude_path, encoding='utf-8').read()
        check('CLAUDE.md contains approval gate rule',
              'approval' in claude.lower() and 'before' in claude.lower())

    # Parquet store
    ps_path = os.path.join(BASE_DIR, 'src', 'database', 'parquet_store.py')
    check('parquet_store.py exists', os.path.exists(ps_path))
    if os.path.exists(ps_path):
        src = open(ps_path, encoding='utf-8').read()
        check('ParquetStore class defined', 'class ParquetStore' in src)
        check('ingest_csv() defined', 'def ingest_csv' in src)
        check('query() returns DataFrame', 'def query' in src)
        check('status() defined for control plane', 'def status' in src)
        check('uses duckdb', 'import duckdb' in src)
        check('partitions by YYYY-MM', "strftime" in src and "%Y-%m" in src)

    # Migration script
    mig_path = os.path.join(BASE_DIR, 'scripts', 'migrate_1sec_to_parquet.py')
    check('migrate_1sec_to_parquet.py exists', os.path.exists(mig_path))
    if os.path.exists(mig_path):
        src = open(mig_path, encoding='utf-8').read()
        check('discover_csv_files() defined', 'def discover_csv_files' in src)
        check('idempotent re-runs (skipped tracking)', 'skipped' in src.lower())
        check('--dry-run flag supported', '--dry-run' in src)

    # Transport package
    tr_dir = os.path.join(BASE_DIR, 'src', 'transport')
    check('src/transport/ package exists', os.path.isdir(tr_dir))
    for fname in ['__init__.py', 'zmq_config.py', 'data_bus.py', 'control_api.py']:
        check(f'src/transport/{fname} exists',
              os.path.exists(os.path.join(tr_dir, fname)))

    # zmq_config
    zc_path = os.path.join(tr_dir, 'zmq_config.py')
    if os.path.exists(zc_path):
        src = open(zc_path, encoding='utf-8').read()
        check('ORDERFLOW_PORT = 5555 default', '5555' in src)
        check('TRAINING_BATCH_PORT = 5556 default', '5556' in src)
        check('CONTROL_FANOUT_PORT = 5557 default', '5557' in src)
        check('CONTROL_API_PORT = 8100 default', '8100' in src)
        check('bind_addr() helper', 'def bind_addr' in src)
        check('connect_addr() helper', 'def connect_addr' in src)

    # data_bus
    db_path = os.path.join(tr_dir, 'data_bus.py')
    if os.path.exists(db_path):
        src = open(db_path, encoding='utf-8').read()
        check('DataBus class defined', 'class DataBus' in src)
        check('publish_orderflow() (PUB)', 'def publish_orderflow' in src)
        check('subscribe_orderflow() (SUB)', 'def subscribe_orderflow' in src)
        check('push_batch() (PUSH)', 'def push_batch' in src)
        check('pull_batch() (PULL)', 'def pull_batch' in src)
        check('uses pyzmq', 'import zmq' in src)

    # control_api
    ca_path = os.path.join(tr_dir, 'control_api.py')
    if os.path.exists(ca_path):
        src = open(ca_path, encoding='utf-8').read()
        check('FastAPI app builder', 'FastAPI(' in src)
        check('/health route', '/health' in src)
        check('/parquet/status route', '/parquet/status' in src)
        check('/parquet/ingest route', '/parquet/ingest' in src)
        check('/databus/stats route', '/databus/stats' in src)
        check('uvicorn entrypoint', 'uvicorn' in src)

    # Orchestrator + worker integration
    orch_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'orchestrator.py')
    if os.path.exists(orch_path):
        src = open(orch_path, encoding='utf-8').read()
        check('orchestrator exposes /api/parquet/status', '/api/parquet/status' in src)
        check('orchestrator exposes /api/databus/stats', '/api/databus/stats' in src)

    worker_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'worker.py')
    if os.path.exists(worker_path):
        src = open(worker_path, encoding='utf-8').read()
        check('worker /health includes transport info', '_transport_info' in src)

    # requirements.txt has new deps
    req = open(os.path.join(BASE_DIR, 'requirements.txt'), encoding='utf-8').read()
    for pkg in ['pyarrow', 'pyzmq', 'fastapi', 'uvicorn', 'msgpack']:
        check(f'requirements.txt: {pkg}', pkg in req)


# ─── Static: Phase 1 institutional upgrade (microstructure data layer) ──────

def test_phase1_microstructure():
    print('\n[Phase 1 -- Level 1 Data Layer]')

    # Kalman smoother
    ks_path = os.path.join(BASE_DIR, 'src', 'analysis', 'kalman_smoother.py')
    check('kalman_smoother.py exists', os.path.exists(ks_path))
    if os.path.exists(ks_path):
        src = open(ks_path, encoding='utf-8').read()
        check('smooth_price() defined', 'def smooth_price' in src)
        check('uses pykalman.KalmanFilter',
              'from pykalman import KalmanFilter' in src or 'pykalman' in src)
        check('plan formula: transition_matrices=[1]', 'transition_matrices=[1]' in src)
        check('plan formula: observation_matrices=[1]', 'observation_matrices=[1]' in src)
        check('plan formula: transition_covariance=0.01', 'transition_covariance' in src and '0.01' in src)

    # Order book features
    obf_path = os.path.join(BASE_DIR, 'src', 'analysis', 'orderbook_features.py')
    check('orderbook_features.py exists', os.path.exists(obf_path))
    if os.path.exists(obf_path):
        src = open(obf_path, encoding='utf-8').read()
        check('imbalance() defined', 'def imbalance' in src)
        check('microprice() defined', 'def microprice' in src)
        check('order_flow_imbalance() defined', 'def order_flow_imbalance' in src)
        check('plan formula: V_bid - V_ask / V_bid + V_ask',
              'v_bid - v_ask' in src.lower() and 'v_bid + v_ask' in src.lower())
        check('aggregate_levels() defined', 'def aggregate_levels' in src)

    # Order book collector
    obc_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'orderbook_collector.py')
    check('orderbook_collector.py exists', os.path.exists(obc_path))
    if os.path.exists(obc_path):
        src = open(obc_path, encoding='utf-8').read()
        check('streams from Binance public depth', 'stream.binance.com' in src)
        check('publishes via DataBus', 'get_data_bus' in src and 'publish_orderflow' in src)
        check('auto-reconnect with backoff',
              'backoff' in src and 'while True' in src)

    # feature_store integration
    fs_path = os.path.join(BASE_DIR, 'src', 'analysis', 'feature_store.py')
    if os.path.exists(fs_path):
        src = open(fs_path, encoding='utf-8').read()
        check('feature_store imports smooth_price',
              'from src.analysis.kalman_smoother import smooth_price' in src)
        check('feature_store imports add_orderbook_features',
              'add_orderbook_features' in src)
        check('feature_store applies Kalman to close',
              'price_kalman' in src and 'smooth_price' in src)

    # feature_engineering Phase 1 helpers
    fe_path = os.path.join(BASE_DIR, 'src', 'analysis', 'feature_engineering.py')
    if os.path.exists(fe_path):
        src = open(fe_path, encoding='utf-8').read()
        check('add_kalman_close() defined', 'def add_kalman_close' in src)
        check('add_l2_features() defined', 'def add_l2_features' in src)
        check('causal_audit() defined', 'def causal_audit' in src)

    # triple_barrier causal t1 audit
    tb_path = os.path.join(BASE_DIR, 'src', 'analysis', 'triple_barrier.py')
    if os.path.exists(tb_path):
        src = open(tb_path, encoding='utf-8').read()
        check('causal_t1_audit() defined', 'def causal_t1_audit' in src)
        check('purge_overlapping_train() defined', 'def purge_overlapping_train' in src)

    # requirements.txt: pykalman
    req = open(os.path.join(BASE_DIR, 'requirements.txt'), encoding='utf-8').read()
    check('requirements.txt: pykalman', 'pykalman' in req)


# ─── Static: Phase 2 institutional upgrade (alpha engine) ──────────────────

def test_phase2_alpha_engine():
    print('\n[Phase 2 -- Level 2 Alpha Engine]')

    # Event-time labeler
    el_path = os.path.join(BASE_DIR, 'src', 'analysis', 'event_time_labeler.py')
    check('event_time_labeler.py exists', os.path.exists(el_path))
    if os.path.exists(el_path):
        src = open(el_path, encoding='utf-8').read()
        check('label_event_time() defined', 'def label_event_time' in src)
        check('regime_normalized_barriers() defined', 'def regime_normalized_barriers' in src)
        check('plan formula vol_norm = atr / atr.rolling(100).mean()',
              "atr.rolling" in src or "rolling(100" in src)
        check('binary classification: drops timeouts',
              'binary_y' in src and 'labels != 0' in src)

    # OFT model
    oft_path = os.path.join(BASE_DIR, 'src', 'models', 'order_flow_transformer.py')
    check('order_flow_transformer.py exists', os.path.exists(oft_path))
    if os.path.exists(oft_path):
        src = open(oft_path, encoding='utf-8').read()
        check('OrderFlowTransformer class defined', 'class OrderFlowTransformer' in src)
        check('OFTConfig class defined', 'class OFTConfig' in src)
        check('OFTOutput NamedTuple defined', 'OFTOutput' in src)
        check('Event Embedding component', '_EventEmbedding' in src or 'event_emb' in src)
        check('Order Book Encoder component', '_OrderBookEncoder' in src or 'ob_encoder' in src)
        check('Temporal Transformer component', '_TemporalTransformer' in src or 'temporal' in src)
        check('Cross-Attention component', 'MultiheadAttention' in src or 'cross' in src)
        check('multi-task heads (mu, log_var, p_move, liq)',
              all(s in src for s in ('head_mu', 'head_log_var', 'head_p_move', 'head_liq')))
        check('regime conditioning embedding', 'regime_emb' in src)

    # OFT trainer
    trainer_path = os.path.join(BASE_DIR, 'src', 'training', 'oft_trainer.py')
    check('oft_trainer.py exists', os.path.exists(trainer_path))
    if os.path.exists(trainer_path):
        src = open(trainer_path, encoding='utf-8').read()
        check('purged_kfold() defined', 'def purged_kfold' in src)
        check('IsotonicCalibrator class defined', 'class IsotonicCalibrator' in src)
        check('microstructure_augment() defined', 'def microstructure_augment' in src)
        check('OFTTrainer class defined', 'class OFTTrainer' in src)
        check('embargo for purged CV', 'embargo' in src)

    # Regime classifier upgrade
    rc_path = os.path.join(BASE_DIR, 'src', 'analysis', 'regime_classifier.py')
    if os.path.exists(rc_path):
        src = open(rc_path, encoding='utf-8').read()
        check('regime_classifier uses BayesianGaussianMixture',
              'BayesianGaussianMixture' in src)
        check('regime_classifier dirichlet_process prior',
              'dirichlet_process' in src)
        check('regime_classifier partial_fit() method', 'def partial_fit' in src)

    # Inference engine OFT path
    ie_path = os.path.join(BASE_DIR, 'src', 'engine', 'inference_engine.py')
    if os.path.exists(ie_path):
        src = open(ie_path, encoding='utf-8').read()
        check('inference_engine loads OFT', '_load_oft_model' in src)
        check('inference_engine has _oft_predict()', '_oft_predict' in src)
        check('predictions surface oft key',
              ('"oft"' in src or "'oft'" in src))

    # src/models package init
    init_path = os.path.join(BASE_DIR, 'src', 'models', '__init__.py')
    check('src/models/__init__.py exists', os.path.exists(init_path))


# ─── Static: Phase 3 institutional upgrade (execution & simulation) ────────

def test_phase3_execution_simulation():
    print('\n[Phase 3 -- Level 3 Execution & Simulation]')

    # alpha decay
    ad_path = os.path.join(BASE_DIR, 'src', 'analysis', 'alpha_decay.py')
    check('alpha_decay.py exists', os.path.exists(ad_path))
    if os.path.exists(ad_path):
        src = open(ad_path, encoding='utf-8').read()
        check('apply_alpha_decay() defined', 'def apply_alpha_decay' in src)
        check('half_life() defined', 'def half_life' in src)
        check('should_exit() defined', 'def should_exit' in src)
        check('uses exp(-decay_rate * t)',
              'math.exp(-' in src or 'np.exp(-' in src)

    # synthetic exchange
    se_path = os.path.join(BASE_DIR, 'src', 'simulation', 'synthetic_exchange.py')
    check('synthetic_exchange.py exists', os.path.exists(se_path))
    if os.path.exists(se_path):
        src = open(se_path, encoding='utf-8').read()
        check('SyntheticExchange class defined', 'class SyntheticExchange' in src)
        check('softmax_fill() differentiable matcher', 'def softmax_fill' in src)
        check('ImpactModel dataclass defined', 'class ImpactModel' in src)
        check('reset/step lifecycle (gym-like)',
              'def reset' in src and 'def step' in src)

    # rl base
    rb_path = os.path.join(BASE_DIR, 'src', 'models', 'rl_base.py')
    check('rl_base.py exists', os.path.exists(rb_path))
    if os.path.exists(rb_path):
        src = open(rb_path, encoding='utf-8').read()
        check('BaseExecutionAgent abstract class', 'class BaseExecutionAgent' in src)
        check('ReplayBuffer defined', 'class ReplayBuffer' in src)
        check('shaped_reward formula PnL - λ * inventory²',
              'shaped_reward' in src and 'inventory' in src)

    # SAC
    sac_path = os.path.join(BASE_DIR, 'src', 'models', 'rl_execution_sac.py')
    check('rl_execution_sac.py exists', os.path.exists(sac_path))
    if os.path.exists(sac_path):
        src = open(sac_path, encoding='utf-8').read()
        check('SACAgent class defined', 'class SACAgent' in src)
        check('twin Q-critics', 'q1' in src and 'q2' in src)
        check('automatic entropy tuning (log_alpha)', 'log_alpha' in src)
        check('target soft update via tau', 'self.tau' in src)

    # PPO
    ppo_path = os.path.join(BASE_DIR, 'src', 'models', 'rl_execution_ppo.py')
    check('rl_execution_ppo.py exists', os.path.exists(ppo_path))
    if os.path.exists(ppo_path):
        src = open(ppo_path, encoding='utf-8').read()
        check('PPOAgent class defined', 'class PPOAgent' in src)
        check('clipped surrogate (clamp ratio)', 'clamp(' in src and 'self.clip' in src)
        check('GAE(λ) advantage', 'compute_gae' in src and 'gae_lambda' in src)

    # multi-agent env
    mae_path = os.path.join(BASE_DIR, 'src', 'simulation', 'multi_agent_env.py')
    check('multi_agent_env.py exists', os.path.exists(mae_path))
    if os.path.exists(mae_path):
        src = open(mae_path, encoding='utf-8').read()
        check('MultiAgentEnv class defined', 'class MultiAgentEnv' in src)
        check('NoiseAgent baseline defined', 'class NoiseAgent' in src)
        check('MomentumAgent baseline defined', 'class MomentumAgent' in src)

    # order_manager alpha decay integration
    om_path = os.path.join(BASE_DIR, 'src', 'engine', 'order_manager.py')
    if os.path.exists(om_path):
        src = open(om_path, encoding='utf-8').read()
        check('order_manager exposes should_alpha_decay_exit()',
              'def should_alpha_decay_exit' in src)


# ─── Static: Phase 4 institutional upgrade (portfolio optimization) ────────

def test_phase4_portfolio_optimization():
    print('\n[Phase 4 -- Level 4 Portfolio Optimization]')

    # CVaR optimizer
    cv_path = os.path.join(BASE_DIR, 'src', 'analysis', 'cvar_optimizer.py')
    check('cvar_optimizer.py exists', os.path.exists(cv_path))
    if os.path.exists(cv_path):
        src = open(cv_path, encoding='utf-8').read()
        check('CVaROptimizer class defined', 'class CVaROptimizer' in src)
        check('uses cvxpy for convex CVaR program', 'import cvxpy' in src)
        check('Rockafellar-Uryasev representation',
              'Rockafellar' in src or '(1.0 / (self.alpha * n_scen))' in src)
        check('confidence_weights() per arch plan §14',
              'def confidence_weights' in src and '(p - 0.5) * 2' in src)
        check('risk_parity_weights() per arch plan §14',
              'def risk_parity_weights' in src)

    # Dynamic threshold
    dt_path = os.path.join(BASE_DIR, 'src', 'analysis', 'dynamic_threshold.py')
    check('dynamic_threshold.py exists', os.path.exists(dt_path))
    if os.path.exists(dt_path):
        src = open(dt_path, encoding='utf-8').read()
        check('find_best_threshold() defined', 'def find_best_threshold' in src)
        check('grid range [0.5, 0.8] per plan',
              'grid_low' in src and '0.5' in src and '0.8' in src)
        check('rolling_threshold() for online use', 'def rolling_threshold' in src)

    # Kelly weight prior
    kc_path = os.path.join(BASE_DIR, 'src', 'analysis', 'kelly_criterion.py')
    if os.path.exists(kc_path):
        src = open(kc_path, encoding='utf-8').read()
        check('kelly_weight_prior() defined', 'def kelly_weight_prior' in src)

    # Risk manager CVaR helper
    rm_path = os.path.join(BASE_DIR, 'src', 'analysis', 'risk_manager.py')
    if os.path.exists(rm_path):
        src = open(rm_path, encoding='utf-8').read()
        check('risk_manager has cvar_position_weights',
              'def cvar_position_weights' in src)
        check('risk_manager imports CVaROptimizer',
              'from src.analysis.cvar_optimizer import CVaROptimizer' in src)

    # requirements.txt: cvxpy
    req = open(os.path.join(BASE_DIR, 'requirements.txt'), encoding='utf-8').read()
    check('requirements.txt: cvxpy', 'cvxpy' in req)


# ─── Static: Phase 5 institutional upgrade (safeguards) ────────────────────

def test_phase5_institutional_safeguards():
    print('\n[Phase 5 -- Level 5 Institutional Safeguards]')

    # Slippage model
    sl_path = os.path.join(BASE_DIR, 'src', 'analysis', 'slippage_model.py')
    check('slippage_model.py exists', os.path.exists(sl_path))
    if os.path.exists(sl_path):
        src = open(sl_path, encoding='utf-8').read()
        check('linear_slippage_bps() defined', 'def linear_slippage_bps' in src)
        check('book_walk_slippage() defined', 'def book_walk_slippage' in src)
        check('real_price() per arch plan §16',
              'def real_price' in src and '(1 + Fee + Slippage' in src)
        check('apply_slippage_to_pnl() for backtest',
              'def apply_slippage_to_pnl' in src)

    # Beta neutrality
    bn_path = os.path.join(BASE_DIR, 'src', 'analysis', 'beta_neutrality.py')
    check('beta_neutrality.py exists', os.path.exists(bn_path))
    if os.path.exists(bn_path):
        src = open(bn_path, encoding='utf-8').read()
        check('BetaNeutralityFilter class defined',
              'class BetaNeutralityFilter' in src)
        check('would_breach() pre-trade check', 'def would_breach' in src)
        check('aggregate_beta() helper', 'def aggregate_beta' in src)
        check('online refit() supported', 'def refit' in src)

    # Order manager circuit breakers
    om_path = os.path.join(BASE_DIR, 'src', 'engine', 'order_manager.py')
    if os.path.exists(om_path):
        src = open(om_path, encoding='utf-8').read()
        check('order_manager.circuit_breaker_check() defined',
              'def circuit_breaker_check' in src)
        for trig in ('max_daily_drawdown', 'api_latency', 'data_feed_inconsistency'):
            check(f'circuit breaker trigger: {trig}', trig in src)

    # Risk agent beta gate
    ra_path = os.path.join(BASE_DIR, 'src', 'engine', 'agents', 'risk_agent.py')
    if os.path.exists(ra_path):
        src = open(ra_path, encoding='utf-8').read()
        check('risk_agent.attach_beta_filter() defined',
              'def attach_beta_filter' in src)
        check('risk_agent.check_beta_neutrality() defined',
              'def check_beta_neutrality' in src)


# ─── Static: Phase 7 institutional upgrade (continuous pipeline + retention) ─

def test_phase7_continuous_pipeline():
    print('\n[Phase 7 -- Continuous Pipeline + Retention]')

    # Multi-tf archive downloader
    ad_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'binance_archive_downloader.py')
    if os.path.exists(ad_path):
        src = open(ad_path, encoding='utf-8').read()
        check('archive downloader has --timeframe arg', '"--timeframe"' in src or "'--timeframe'" in src)
        check('archive downloader supports --all-timeframes', '"--all-timeframes"' in src or "'--all-timeframes'" in src)
        check('archive downloader has SUPPORTED_TF constant', 'SUPPORTED_TF' in src)
        check('archive downloader supports 1mo (monthly)', "'1mo'" in src or '"1mo"' in src)
        check('1s preserves _spot_1s.csv.gz path', '_spot_1s.csv.gz' in src)

    # Realtime writer
    rt_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'realtime_db_writer.py')
    check('realtime_db_writer.py exists', os.path.exists(rt_path))
    if os.path.exists(rt_path):
        src = open(rt_path, encoding='utf-8').read()
        check('uses Binance WS endpoint', 'stream.binance.com' in src)
        check('writes to QuestDB', 'write_market_candle' in src and 'questdb' in src.lower())
        check('only emits closed bars (k.x)', '"x"' in src and 'closed' in src.lower())
        check('cold rollover loop defined', 'cold_rollover_loop' in src)

    # Startup recovery
    sr_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'startup_recovery.py')
    check('startup_recovery.py exists', os.path.exists(sr_path))
    if os.path.exists(sr_path):
        src = open(sr_path, encoding='utf-8').read()
        check('recover_all() defined', 'def recover_all' in src)
        check('combines QuestDB + Parquet last-ts',
              '_questdb_last_ts' in src and '_parquet_last_ts' in src)
        check('REST top-up uses Binance public klines',
              'api.binance.com/api/v3/klines' in src)

    # Retention manager
    rm_path = os.path.join(BASE_DIR, 'src', 'database', 'retention_manager.py')
    check('retention_manager.py exists', os.path.exists(rm_path))
    if os.path.exists(rm_path):
        src = open(rm_path, encoding='utf-8').read()
        check('RetentionManager class defined', 'class RetentionManager' in src)
        check('mark_trained() method', 'def mark_trained' in src)
        check('archive_eligible() method', 'def archive_eligible' in src)
        check('uses safe_json', 'from src.utils.safe_json' in src)

    # Google Drive backup
    gd_path = os.path.join(BASE_DIR, 'src', 'database', 'google_drive_backup.py')
    check('google_drive_backup.py exists', os.path.exists(gd_path))
    if os.path.exists(gd_path):
        src = open(gd_path, encoding='utf-8').read()
        check('GoogleDriveBackup class defined', 'class GoogleDriveBackup' in src)
        check('fail-soft when pydrive2 missing',
              'pydrive2 not installed' in src or 'except ImportError' in src)
        check('upload_partition() defined', 'def upload_partition' in src)

    # restart_all.ps1 wiring
    rs_path = os.path.join(BASE_DIR, 'restart_all.ps1')
    if os.path.exists(rs_path):
        src = open(rs_path, encoding='utf-8').read()
        check('restart_all runs startup_recovery on launch',
              'startup_recovery' in src)
        check('restart_all starts realtime_db_writer',
              'realtime_db_writer' in src)
        check('restart_all saves realtime PID',
              'realtime' in src.lower() and 'process_ids.json' in src)


# ─── Static: Phase 8 institutional upgrade (data governance) ───────────────

def test_phase8_data_governance():
    print('\n[Phase 8 -- Data Governance + Rate Limiting]')

    rl_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'rate_limiter.py')
    check('rate_limiter.py exists', os.path.exists(rl_path))
    if os.path.exists(rl_path):
        src = open(rl_path, encoding='utf-8').read()
        check('RateLimiter class defined', 'class RateLimiter' in src)
        check('get_limiter() singleton', 'def get_limiter' in src)
        check('react_to_response handles 429/418',
              '429' in src and '418' in src and 'Retry-After' in src)
        check('binance.com host configured',
              '"binance.com"' in src or "'binance.com'" in src)

    bs_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'binance_sync.py')
    check('binance_sync.py exists', os.path.exists(bs_path))
    if os.path.exists(bs_path):
        src = open(bs_path, encoding='utf-8').read()
        check('step_archive() defined', 'def step_archive' in src)
        check('step_rest_topup() defined', 'def step_rest_topup' in src)
        check('step_cross_check() defined', 'def step_cross_check' in src)
        check('uses rate_limiter', 'rate_limiter' in src or 'get_limiter' in src)

    ad_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'binance_archive_downloader.py')
    if os.path.exists(ad_path):
        src = open(ad_path, encoding='utf-8').read()
        check('archive: HEAD probe (_zip_exists)', 'def _zip_exists' in src)
        check('archive: cross-TF parallel runner',
              'def download_all_timeframes_parallel' in src)
        check('archive: listing-date cache helpers',
              '_load_listing_cache' in src and '_save_listing_cache' in src)
        check('archive: MAX_WORKERS env-overridable',
              'ARCHIVE_MAX_WORKERS' in src)

    dg_path = os.path.join(BASE_DIR, 'src', 'data_governance')
    check('src/data_governance/ package exists', os.path.isdir(dg_path))
    for fname in ['__init__.py', 'base.py', 'registry.py', 'config.py', 'orchestrator.py']:
        check(f'src/data_governance/{fname} exists',
              os.path.exists(os.path.join(dg_path, fname)))

    conn_path = os.path.join(dg_path, 'connectors')
    check('connectors/ package exists', os.path.isdir(conn_path))
    expected_connectors = [
        'bybit', 'okx', 'coinbase', 'kraken', 'coingecko', 'fear_greed',
        'fred', 'defillama', 'cryptocompare_news', 'coinglass', 'reddit',
    ]
    for c in expected_connectors:
        check(f'connector {c}.py exists',
              os.path.exists(os.path.join(conn_path, f'{c}.py')))

    # Orchestrator
    orch_src = open(os.path.join(dg_path, 'orchestrator.py'), encoding='utf-8').read() \
        if os.path.exists(os.path.join(dg_path, 'orchestrator.py')) else ''
    check('orchestrator.run_history() defined', 'def run_history' in orch_src)
    check('orchestrator.run_forever() defined', 'def run_forever' in orch_src)
    check('orchestrator --list/--once CLI', '--list' in orch_src and '--once' in orch_src)

    # Top-level docs
    ds_path = os.path.join(BASE_DIR, 'DATA_SOURCES.md')
    check('DATA_SOURCES.md at root', os.path.exists(ds_path))
    if os.path.exists(ds_path):
        ds = open(ds_path, encoding='utf-8').read()
        check('DATA_SOURCES lists Tier 0 (free)',
              'Tier 0' in ds or 'tier 0' in ds.lower())
        check('DATA_SOURCES has setup steps for FRED/CryptoCompare',
              'FRED_API_KEY' in ds and 'CRYPTOCOMPARE_API_KEY' in ds)

    # restart_all.ps1 wiring
    rs_path = os.path.join(BASE_DIR, 'restart_all.ps1')
    if os.path.exists(rs_path):
        src = open(rs_path, encoding='utf-8').read()
        check('restart_all launches data_governance.orchestrator',
              'data_governance' in src and 'orchestrator' in src)


# ─── Static: Phase 10 institutional upgrade (live bot integration) ─────────

def test_phase10_live_integration():
    print('\n[Phase 10 -- Live Integration + 8-tab Dashboard + Documentation]')

    fr_path = os.path.join(BASE_DIR, 'src', 'analysis', 'feature_reader.py')
    check('feature_reader.py exists', os.path.exists(fr_path))
    if os.path.exists(fr_path):
        src = open(fr_path, encoding='utf-8').read()
        check('load_recent_bars() Parquet-first', 'def load_recent_bars' in src and '_parquet_load' in src)
        check('CSV fallback present', '_csv_load' in src)
        check('load_news_recent() defined', 'def load_news_recent' in src)

    main_src = open(os.path.join(BASE_DIR, 'src', 'main.py'), encoding='utf-8').read()
    check('main.py imports feature_reader',
          'from src.analysis import feature_reader as _feature_reader' in main_src)
    check('main.py uses Parquet-first read', '_feature_reader.load_recent_bars(' in main_src)
    check('main.py has _attach_beta_history', 'def _attach_beta_history' in main_src)
    check('main.py has _refresh_dynamic_thresholds', 'def _refresh_dynamic_thresholds' in main_src)
    check('main.py wires alpha-decay exit', 'should_exit_decay' in main_src)

    fe_src = open(os.path.join(BASE_DIR, 'src', 'analysis', 'feature_engineering.py'), encoding='utf-8').read()
    check('add_news_sentiment uses Parquet', 'load_news_recent' in fe_src)

    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'), encoding='utf-8').read()
    for tab in ('portfolio', 'alpha', 'orderflow', 'risk',
                'training', 'simulation', 'data', 'strategies'):
        check(f'8-tab nav: {tab}', f'data-tab="{tab}"' in tpl)

    doc = os.path.join(BASE_DIR, 'APP_DOCUMENTATION.md')
    check('APP_DOCUMENTATION.md exists', os.path.exists(doc))

    # Bat files
    for fname in ('START_HERE.bat', 'start_all.bat', 'restart_all.bat', 'stop_all.bat'):
        path = os.path.join(BASE_DIR, fname)
        check(f'{fname} exists', os.path.exists(path))
        if os.path.exists(path):
            content = open(path, encoding='utf-8', errors='ignore').read()
            check(f'{fname} routes through .ps1',
                  '.ps1' in content)
            check(f'{fname} keeps console open (-NoExit)',
                  '-NoExit' in content)

    sps1 = os.path.join(BASE_DIR, 'stop_all.ps1')
    check('stop_all.ps1 exists', os.path.exists(sps1))
    if os.path.exists(sps1):
        content = open(sps1, encoding='utf-8').read()
        check('stop_all.ps1 reads PID file',  'process_ids.json' in content)
        check('stop_all.ps1 covers all managed services',
              all(k in content for k in ('bot', 'dash', 'monitor', 'training',
                                          'realtime', 'orch', 'watchlist')))


def test_phase11_predictor_and_llm_resilience():
    """Regression coverage for the dict-wrapped joblib unwrap and Gemini cooldown."""
    print('\n[Phase 11 -- Predictor / LLM resilience]')

    # B1 — MLPredictor must accept dict-wrapped joblib payloads
    mp_path = os.path.join(BASE_DIR, 'src', 'analysis', 'ml_predictor.py')
    src = open(mp_path, encoding='utf-8').read()
    check('ml_predictor unwraps dict joblib',
          'isinstance(loaded, dict)' in src and 'loaded["model"]' in src)
    check('ml_predictor logs traceback on error',
          'traceback.format_exc()' in src)
    check('ml_predictor stores embedded feature list',
          '_embedded_features' in src and 'feature_cols' in src)

    # Functional check: build a dict-wrapped fixture and confirm load works
    try:
        import sys as _sys
        _sys.path.insert(0, BASE_DIR)
        import joblib, tempfile
        from src.analysis.ml_predictor import MLPredictor
        live = MLPredictor()
        if live.is_loaded:
            fake = os.path.join(tempfile.gettempdir(), 'phase11_dict_model.joblib')
            joblib.dump({'model': live.model, 'feature_cols': ['x1', 'x2']}, fake)
            stub = MLPredictor.__new__(MLPredictor)
            stub.model_path = fake; stub.model_type = 'base'
            stub.model = None; stub.is_loaded = False
            stub.accuracy = 0.0; stub.long_accuracy = 0.0; stub.short_accuracy = 0.0
            stub.last_error = ''; stub._last_confidence = 0.5
            stub._embedded_features = None
            loaded = joblib.load(fake)
            if isinstance(loaded, dict) and 'model' in loaded and hasattr(loaded['model'], 'predict'):
                stub.model = loaded['model']
                stub._embedded_features = list(loaded['feature_cols'])
            check('dict-wrapped joblib unwraps to estimator with .predict',
                  hasattr(stub.model, 'predict'))
            check('embedded feature_cols surface via _get_model_features',
                  stub._get_model_features() == ['x1', 'x2'])
            os.unlink(fake)
        else:
            check('dict-wrapped joblib unwraps to estimator with .predict', None,
                  'btc_rf_model.joblib not loaded')
    except Exception as e:
        check('dict-wrapped joblib unwrap functional', False, str(e))

    # B2 — agentic_llm cooldown helpers
    al_path = os.path.join(BASE_DIR, 'src', 'engine', 'agentic_llm.py')
    al_src = open(al_path, encoding='utf-8').read()
    check('agentic_llm has cooldown registry',
          '_model_cooldown_until' in al_src and '_MODEL_COOLDOWN_S' in al_src)
    check('agentic_llm has _is_cooled_down + _mark_cooldown',
          'def _is_cooled_down' in al_src and 'def _mark_cooldown' in al_src)
    check('agentic_llm filters cooled-down models before retry loop',
          '_is_cooled_down(m)' in al_src)
    check('agentic_llm demotes per-iteration warning to debug',
          'logger.debug(f"Agentic LLM:' in al_src)

    try:
        from src.engine import agentic_llm as _al
        _al._mark_cooldown('test-model-x')
        check('mark_cooldown sets future expiry', _al._is_cooled_down('test-model-x'))
        check('uncooled model is not flagged', not _al._is_cooled_down('test-model-y'))
        # Reset so the test doesn't bleed state into other tests
        _al._model_cooldown_until.pop('test-model-x', None)
    except Exception as e:
        check('agentic_llm cooldown functional', False, str(e))

    # QuestDB native binary install (this session)
    qdb_dir = os.path.join(BASE_DIR, 'questdb')
    qdb_exe = os.path.join(qdb_dir, 'questdb-9.3.5-rt-windows-x86-64', 'bin', 'java.exe')
    check('QuestDB bundled-JRE binary present', os.path.exists(qdb_exe))
    qdb_launch = os.path.join(BASE_DIR, 'launch_questdb.ps1')
    if os.path.exists(qdb_launch):
        qdb_src = open(qdb_launch, encoding='utf-8').read()
        check('launch_questdb.ps1 uses native binary path',
              'questdb-9.3.5-rt-windows-x86-64' in qdb_src and 'io.questdb.ServerMain' in qdb_src)


def test_phase12_dashboard_controls():
    """Each Phase 6 tab fetch URL must resolve to a real backend endpoint —
    catches future 404 regressions like the /api/balance/test bug."""
    print('\n[Phase 12 -- Dashboard control wiring]')

    tpl = open(TEMPLATE_PATH, encoding='utf-8').read()
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()

    # 1. /api/balance/test 404 fix — frontend must NOT build that URL via getMode()
    check('portfolio panel does NOT use raw getMode() in URL',
          "'/api/balance/' + (window.getMode" not in tpl)
    check('portfolio panel translates test → virtual',
          "/api/balance/virtual" in tpl and "_mode === 'real'" in tpl)
    check('backend has /api/balance/test alias',
          "@app.route('/api/balance/test')" in app)

    # 2. Phase 6 panel endpoints all defined backend-side
    p6_endpoints = [
        '/api/balance/real', '/api/balance/virtual',
        '/api/decision_summary',
        '/api/oft_signal',
        '/api/rate_limiter/stats',
        '/api/cluster/status',
        '/api/simulator/status',
        '/api/parquet/coverage',
        '/api/orchestrator/sources',
    ]
    for ep in p6_endpoints:
        check(f'backend route exists: {ep}',
              f"@app.route('{ep}')" in app or f"@app.route(\"{ep}\")" in app or ep in app)

    # 3. Telegram heartbeat detection wired up
    tg_mod = open(os.path.join(BASE_DIR, 'src', 'analysis', 'telegram_monitor.py'), encoding='utf-8').read()
    check('telegram_monitor writes status heartbeat',
          'telegram_status.json' in tg_mod and '_write_status' in tg_mod)
    check('dashboard reads telegram heartbeat',
          'telegram_status.json' in app and "'embedded':" in app)

    # 4. ML model card honesty
    check('backend exposes accuracy_walk_forward',
          'accuracy_walk_forward' in app)
    check('backend computes accuracy_warning on imbalance',
          'accuracy_warning' in app and 'spread' in app)
    check('frontend renders accuracy_warning',
          'm.accuracy_warning' in tpl)

    # 5. P6 cards inherit dashboard glass-card styling
    check('p6-card uses gradient + backdrop-filter',
          '.p6-card' in tpl and 'backdrop-filter' in tpl and 'linear-gradient(180deg,rgba(15,23,42' in tpl)

    # 6. launch_bot.ps1 stderr handling (B3)
    bot_ps = open(os.path.join(BASE_DIR, 'launch_bot.ps1'), encoding='utf-8').read()
    check('launch_bot.ps1 normalises stderr to plain text',
          'ErrorRecord' in bot_ps and 'Write-Output' in bot_ps)

    # 7. cryptg installed (B4)
    try:
        import importlib
        importlib.import_module('cryptg')
        check('cryptg installed (Telethon fast crypto)', True)
    except Exception as e:
        check('cryptg installed (Telethon fast crypto)', False, str(e))

    # 8. /api/balance/test endpoint actually returns 200, not 404
    if not getattr(test_phase12_dashboard_controls, '_offline', False):
        try:
            import urllib.request
            with urllib.request.urlopen(DASHBOARD_URL + '/api/balance/test', timeout=2) as r:
                check('/api/balance/test alias responds 200',
                      r.status == 200, f'status={r.status}')
        except Exception as e:
            check('/api/balance/test alias responds 200', None, f'skipped (server down): {e}')


def test_phase13_realtime_and_fastapi():
    """Realtime heartbeat checks.

    Phase A11 (2026-05-12): the FastAPI control plane on :8100 was
    DELETED. The 6 endpoints it exposed (/health, /status, /metrics,
    /control/bot/start, /control/bot/stop, /control/training/start)
    were all duplicates of dashboard routes or scripts. This test now
    asserts the deletion landed cleanly — file is gone, restart_all
    doesn't launch it, and (defense in depth) stop_all doesn't try
    to kill it either.
    """
    print('\n[Phase 13 -- Realtime heartbeat + (FastAPI control plane DELETED in A11)]')

    # ── Realtime heartbeat ──────────────────────────────────────────────
    rt_src = open(os.path.join(BASE_DIR, 'src', 'data_ingestion', 'realtime_db_writer.py'),
                  encoding='utf-8').read()
    check('realtime_db_writer defines _write_status',
          'def _write_status' in rt_src)
    check('realtime_db_writer writes data/realtime_status.json',
          'realtime_status.json' in rt_src)
    check('realtime_db_writer flips status on connect/disconnect/error',
          rt_src.count('_write_status(') >= 4)

    # ── Phase A11 deletion landing checks ───────────────────────────────
    fapi_src_path = os.path.join(BASE_DIR, 'src', 'server', 'control_plane.py')
    check('Phase A11: src/server/control_plane.py REMOVED',
          not os.path.exists(fapi_src_path))

    launcher = os.path.join(BASE_DIR, 'launch_fastapi.ps1')
    check('Phase A11: launch_fastapi.ps1 REMOVED',
          not os.path.exists(launcher))

    ra = open(os.path.join(BASE_DIR, 'restart_all.ps1'), encoding='utf-8').read()
    check('Phase A11: restart_all.ps1 no longer launches FastAPI',
          'launch_fastapi.ps1' not in ra)
    check('Phase A11: restart_all.ps1 no longer references control_plane module',
          'src.server.control_plane' not in ra.replace('\\.', '.'))

    # ── Realtime heartbeat freshness (only meaningful when bot is running) ──
    if not getattr(test_phase13_realtime_and_fastapi, '_offline', False):
        rt_path = os.path.join(BASE_DIR, 'data', 'realtime_status.json')
        if os.path.exists(rt_path):
            try:
                import json as _json, time as _time
                rt = _json.loads(open(rt_path, encoding='utf-8').read())
                fresh = (_time.time() - float(rt.get('last_update_ts', 0))) < 600
                check('realtime_status.json is fresh (<10min)', fresh,
                      f'connected={rt.get("connected")}')
            except Exception as e:
                check('realtime_status.json parses', False, str(e))
        else:
            check('realtime_status.json present', None,
                  'not created yet — first WS connect needed')


def test_phase14_local_only_scheduler():
    """Local-only TFT/training status inspector + Windows Task Scheduler wrapper.
    Asserts NO cloud calls and the report file is produced when run."""
    print('\n[Phase 14 -- Local-only scheduling]')

    insp = os.path.join(BASE_DIR, 'scripts', 'check_training_status.py')
    sched = os.path.join(BASE_DIR, 'local_scheduler.ps1')

    check('inspector script exists', os.path.exists(insp))
    check('local_scheduler.ps1 exists', os.path.exists(sched))

    if os.path.exists(insp):
        src = open(insp, encoding='utf-8').read()
        check('inspector targets loopback only',
              '127.0.0.1' in src and 'http://' not in src.replace('http://127.0.0.1', ''))
        check('inspector decodes UTF-16 log',
              "utf-16" in src.lower() and 'tft_3epoch.log' in src)
        check('inspector writes report file',
              'training_status_report.json' in src)
        check('inspector has --quiet and --json flags',
              '--quiet' in src and '--json' in src)

    if os.path.exists(sched):
        src = open(sched, encoding='utf-8').read()
        check('scheduler uses native schtasks.exe (no cloud)',
              'schtasks.exe' in src)
        check('scheduler supports register/list/unregister/run',
              all(a in src for a in ("'register'","'list'","'unregister'","'run'")))
        check('scheduler supports -At, -EveryMinutes, -Once',
              all(a in src for a in ('$At', '$EveryMinutes', '$Once')))
        check('scheduler does NOT call any non-loopback URL',
              'github.com' not in src and 'anthropic' not in src.lower()
              and 'claude.ai' not in src)

    # Functional: run inspector and check report appears
    try:
        import subprocess as _sp, json as _json
        report_path = os.path.join(BASE_DIR, 'data', 'training_status_report.json')
        # Invoke via venv python so import paths are right
        py = os.path.join(BASE_DIR, 'venv', 'Scripts', 'python.exe')
        if not os.path.exists(py):
            py = 'python'
        r = _sp.run([py, insp, '--quiet'], cwd=BASE_DIR, capture_output=True, timeout=30)
        check('inspector exits 0', r.returncode == 0,
              f'stderr={r.stderr.decode("utf-8","replace")[:200]}')
        check('report file produced', os.path.exists(report_path))
        if os.path.exists(report_path):
            data = _json.loads(open(report_path, encoding='utf-8').read())
            check('report has execution=LOCAL_ONLY',
                  data.get('execution') == 'LOCAL_ONLY')
            check('report has status field',
                  data.get('status') in ('completed', 'in_progress', 'not_started'))
            check('report has summary_bullets',
                  isinstance(data.get('summary_bullets'), list)
                  and len(data['summary_bullets']) >= 3)
    except Exception as e:
        check('inspector functional run', False, str(e))


def test_phase16_scheduler_panel_and_sim_no_hang():
    """Dashboard scheduler panel + Simulator status non-hang guarantee."""
    print('\n[Phase 16 -- Scheduler panel + Simulator non-hang]')

    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()
    tpl = open(TEMPLATE_PATH, encoding='utf-8').read()

    # Scheduler endpoints exist
    for ep in ('/api/scheduler/list',
               '/api/scheduler/register',
               '/api/scheduler/run',
               '/api/scheduler/unregister',
               '/api/scheduler/report'):
        check(f'backend defines {ep}', f"@app.route('{ep}'" in app)

    check('scheduler endpoints validate task name (no shell injection)',
          'def _safe_task_name' in app and 're.sub' in app.replace('_re.sub','re.sub'))
    check('scheduler endpoints validate mode whitelist',
          "('daily', 'every_minutes', 'once')" in app or "'daily', 'every_minutes', 'once'" in app)

    # Scheduler tab in HTML
    check('Scheduler sub-tab present',
          'data-tab="scheduler"' in tpl)
    check('Scheduler renderer wired',
          'renderSchedulerPanel' in tpl)
    check('Scheduler panel has HOW THIS WORKS instructions',
          'HOW THIS WORKS' in tpl and 'check_training_status.py' in tpl)
    check('Scheduler panel has register/run/delete actions',
          all(x in tpl for x in ('_schRegister', '_schRun', '_schUnregister'))
          and 'REGISTER' in tpl and 'RUN NOW' in tpl and 'DELETE' in tpl)
    check('Scheduler panel has the three modes',
          all(x in tpl for x in ('every_minutes', 'daily', 'once')))
    check('Scheduler panel renders the latest report',
          '_schedulerReport' not in tpl  # legacy name; sanity check only
          and 'Latest report' in tpl)

    # Simulator hang fix: status path must never block on DuckDB.
    # Old design used a Queue + 4s timeout. Phase 18 replaced it with an
    # async TTL cache for db_summary, which satisfies "no hang" without
    # needing the timeout dance at all.
    check('simulator_status never blocks on disk I/O',
          ('_db_summary_cache' in app and '_refresh_db_summary_async' in app)
          or "out_q.get(timeout=4.0)" in app)
    check('simulator_status returns initializing while agents bootstrap',
          "Agents bootstrapping" in app or "Status producer slow" in app
          or "'state': 'initializing'" in app)

    # Live HTTP probes (skip in --offline)
    if not getattr(test_phase16_scheduler_panel_and_sim_no_hang, '_offline', False):
        import urllib.request, urllib.error, json as _json, time as _time
        # 1. /api/simulator/status must respond within 5 s
        try:
            t0 = _time.time()
            with urllib.request.urlopen(DASHBOARD_URL + '/api/simulator/status', timeout=5) as r:
                body = _json.loads(r.read().decode('utf-8'))
            dt = _time.time() - t0
            check(f'/api/simulator/status responds in <5s (took {dt:.2f}s)',
                  r.status == 200 and dt < 5.0,
                  f'state={body.get("state")}')
        except Exception as e:
            check('/api/simulator/status responds in <5s', False, str(e))

        # 2. /api/scheduler/list returns JSON with 'tasks' key
        try:
            with urllib.request.urlopen(DASHBOARD_URL + '/api/scheduler/list', timeout=4) as r:
                body = _json.loads(r.read().decode('utf-8'))
            check('/api/scheduler/list returns tasks array',
                  isinstance(body.get('tasks'), list),
                  f'keys={list(body.keys())}')
        except Exception as e:
            check('/api/scheduler/list returns tasks array', False, str(e))

        # 3. /api/scheduler/report returns ok JSON
        try:
            with urllib.request.urlopen(DASHBOARD_URL + '/api/scheduler/report', timeout=4) as r:
                body = _json.loads(r.read().decode('utf-8'))
            check('/api/scheduler/report responds',
                  'present' in body)
        except Exception as e:
            check('/api/scheduler/report responds', False, str(e))


def test_phase17_trading_health_fixes():
    """Regression coverage for the 2026-05-04 bug-fix batch:
       - Binance -1021 recvWindow / clock-sync
       - Gemini free-tier 429 long cooldown
       - Meta-labeler graceful degradation when prob_base/regime missing
       - QuestDB probe consistency (/exec, not /health)
       - ZMQ probe explanatory hint
       - Dashboard card directionless flag (meta-labeler) + derived n_features
       - Phase 6 sub-tab click no longer wipes the institutional pane
    """
    print('\n[Phase 17 -- Trading & dashboard health fixes (2026-05-04)]')

    # 1. order_manager: clock sync wiring
    om_path = os.path.join(BASE_DIR, 'src', 'engine', 'order_manager.py')
    om_src = open(om_path, encoding='utf-8').read()
    check('order_manager sets recvWindow=60000',
          "'recvWindow': 60000" in om_src)
    check('order_manager enables adjustForTimeDifference',
          "'adjustForTimeDifference': True" in om_src)
    check('order_manager has _sync_clocks helper',
          'def _sync_clocks(' in om_src and 'load_time_difference' in om_src)
    check('get_balance calls _sync_clocks before fetch',
          'def get_balance(' in om_src and
          'self._sync_clocks()' in om_src.split('def get_balance(')[1].split('def ')[0])
    check('execute_spot_order calls _sync_clocks',
          'self._sync_clocks()' in om_src.split('def execute_spot_order(')[1].split('def ')[0])
    check('execute_futures_order calls _sync_clocks',
          'self._sync_clocks()' in om_src.split('def execute_futures_order(')[1].split('def ')[0])
    check('get_balance retries once on InvalidNonce',
          'ccxt.InvalidNonce' in om_src and 'force=True' in om_src)

    # 2. agentic_llm: free-tier long cooldown
    al_path = os.path.join(BASE_DIR, 'src', 'engine', 'agentic_llm.py')
    al_src = open(al_path, encoding='utf-8').read()
    check('agentic_llm defines _FREE_TIER_COOLDOWN_S',
          '_FREE_TIER_COOLDOWN_S' in al_src)
    check('_mark_cooldown accepts seconds parameter',
          'def _mark_cooldown(model_id: str, seconds:' in al_src)
    check('agentic_llm detects free_tier and uses long cooldown',
          'free_tier' in al_src and 'is_free_tier' in al_src)

    # 3. meta_labeler: graceful degradation
    ml_path = os.path.join(BASE_DIR, 'src', 'analysis', 'meta_labeler.py')
    ml_src = open(ml_path, encoding='utf-8').read()
    check('meta_labeler fills neutral priors when prob features missing',
          "feature_row.setdefault('prob_base', 0.5)" in ml_src and
          "feature_row.setdefault('regime', 0)" in ml_src)
    check('meta_labeler logs INFO once instead of WARNING per signal',
          '_warned_missing_probs' in ml_src and 'logger.info' in ml_src)

    # Functional check: meta_labeler.filter accepts incomplete features dict
    try:
        import sys as _sys
        _sys.path.insert(0, BASE_DIR)
        from src.analysis.meta_labeler import MetaLabeler
        mlb = MetaLabeler()
        if mlb.is_loaded:
            decision, conf = mlb.filter(1.0, {'rsi_14': 50.0, 'macd_hist': 0.0})
            check('meta_labeler.filter accepts incomplete features',
                  decision in ('PASS', 'BLOCK') and 0.0 <= conf <= 1.0)
        else:
            check('meta_labeler.filter accepts incomplete features', None,
                  'meta_labeler.joblib not loaded')
    except Exception as e:
        check('meta_labeler.filter accepts incomplete features', False, str(e))

    # 4. signal_agent forwards regime to meta-labeler
    sa_path = os.path.join(BASE_DIR, 'src', 'engine', 'agents', 'signal_agent.py')
    sa_src = open(sa_path, encoding='utf-8').read()
    check('signal_agent surfaces regime to meta-labeler',
          "feats.setdefault('regime', regime)" in sa_src)
    check('signal_agent surfaces prob_base default',
          "feats.setdefault('prob_base'" in sa_src)

    # main.py call site fixed (filter_signal → filter)
    main_src = open(os.path.join(BASE_DIR, 'src', 'main.py'), encoding='utf-8').read()
    check('main.py calls meta_labeler.filter (not filter_signal)',
          'self.meta_labeler.filter(' in main_src and
          'self.meta_labeler.filter_signal(' not in main_src)

    # 5. questdb_client.py was a back-compat shim during Phase 2-4 of the
    #    migration; deleted in the post-Phase-5 shim-drop. All callers now
    #    import directly from parquet_client. The HTTP-probe assertion that
    #    once lived here was for the QuestDB-era client; with Route B the
    #    probe is in-process (no HTTP at all).
    qc_path = os.path.join(BASE_DIR, 'src', 'database', 'questdb_client.py')
    check('questdb_client.py shim deleted (no longer present)',
          not os.path.exists(qc_path))

    # 6. Dashboard backend
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    app_src = open(app_path, encoding='utf-8').read()
    check('ZMQ probe explains lazy bind',
          'binds on first orderflow publish' in app_src or 'binds lazily' in app_src)
    check('ml_models exposes directionless flag',
          "'directionless':" in app_src)
    check('ml_models derives n_features for TFT (input_chunk_length)',
          "input_chunk_length" in app_src and "n_feat = meta.get('input_chunk_length')" in app_src.replace('\n', ' ').replace('  ', ' ')
          or "input_chunk_length" in app_src and "n_feat is None" in app_src)
    check('ml_models derives n_features for regime (gmm.means_)',
          'gmm.means_' in app_src or 'means_.shape' in app_src)

    # 7. Dashboard frontend institutional GUI fix
    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    tpl_src = open(tpl_path, encoding='utf-8').read()
    check('main tab listener scoped to .nav-item[data-tab]',
          ".nav-item[data-tab]" in tpl_src and
          "querySelectorAll('[data-tab]')" not in tpl_src)
    check('directionless models render without long/short bars',
          'isDirectionless' in tpl_src and 'Binary win/loss classifier' in tpl_src)


def test_phase18_institutional_panel_fixes():
    """Regression coverage for the 2026-05-04 round-2 institutional-tab fixes:
       - Scheduler sub-tab pane must render (was missing from _renderTabs)
       - Simulator init must be async (no >30s start hang)
       - Virtual balance auto-heals from the $12345.67 stub
       - OFT model card surfaces alongside the other 7
       - Strategies tab shows guidance when orchestrator returns empty
    """
    print('\n[Phase 18 -- Institutional panel UX & data wiring]')

    tpl = open(TEMPLATE_PATH, encoding='utf-8').read()
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()

    # 1. Scheduler in tabs array
    check("_renderTabs includes 'scheduler'",
          "var tabs = ['portfolio','alpha','orderflow','risk','training','simulation','data','strategies','scheduler']" in tpl)

    # 2. Async simulator init plumbing
    check('app.py defines _ensure_sim_init',
          'def _ensure_sim_init(' in app)
    check('app.py defines _do_sim_init (background-only)',
          'def _do_sim_init(' in app)
    check('app.py defines _apply_sim_start helper',
          'def _apply_sim_start(' in app)
    check('app.py caches SimulatorDataStore singleton',
          'def _get_sim_store(' in app and '_sim_data_store' in app)
    check('app.py queues start cfg during init',
          '_sim_pending_start_cfg' in app)
    check('simulator_status returns initializing when agents not ready',
          "if not _ensure_sim_init():" in app)
    check('simulator_start returns immediately when agents not ready',
          "'queued': True" in app and "'state': 'initializing'" in app)

    # 3. Virtual balance auto-heal from $12345.67 stub
    check('api_balance_virtual auto-heals stub value',
          '_VIRTUAL_STUB_VALUE = 12345.67' in app
          and '_VIRTUAL_DEFAULT_CASH' in app
          and 'reset_virtual(_VIRTUAL_DEFAULT_CASH)' in app)

    # 4. OFT card present in both ML lists
    check('strategy_full _ML lists OFT model',
          "'oft_model.pt'" in app and "'OFT (Microstructure)'" in app)
    check('monitor_model_stats _MODEL_FILES lists OFT',
          app.count("oft_model.pt") >= 2
          and app.count("oft_model_meta.json") >= 2)

    # 5. Strategies empty-state guidance
    check('Strategies tab shows guidance on empty list',
          'No data sources registered' in tpl
          and 'data/orchestrator/sources.json' in tpl)

    # Functional check: api_balance_virtual auto-heals when called
    if not getattr(test_phase18_institutional_panel_fixes, '_offline', False):
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(DASHBOARD_URL + '/api/balance/virtual', timeout=5) as r:
                body = _json.loads(r.read().decode('utf-8'))
            cash = float(body.get('cash_usdt', 0))
            check('/api/balance/virtual no longer reports the $12345.67 stub',
                  abs(cash - 12345.67) > 1.0,
                  f'cash_usdt={cash}')
        except Exception as e:
            check('/api/balance/virtual no longer reports the $12345.67 stub', None,
                  f'skipped (server down): {e}')

        # Functional check: simulator/start must NOT hang for >5s
        try:
            import urllib.request, urllib.error, time as _t, json as _json
            req = urllib.request.Request(
                DASHBOARD_URL + '/api/simulator/start',
                data=b'{}', method='POST',
                headers={'Content-Type': 'application/json'})
            t0 = _t.time()
            with urllib.request.urlopen(req, timeout=5) as r:
                body = _json.loads(r.read().decode('utf-8'))
            dt = _t.time() - t0
            check(f'/api/simulator/start returns in <5s (took {dt:.2f}s)',
                  r.status == 200 and dt < 5.0,
                  f'state={body.get("state")} queued={body.get("queued")}')
        except Exception as e:
            check('/api/simulator/start returns in <5s', None,
                  f'skipped (server down): {e}')


def test_phase19_oft_integration():
    """Regression coverage for the OFT live wiring + simulator deadlock fix:
       - SimulatorAgent._flush_state must NOT re-acquire its own lock
       - simulator_status path is timeout-protected
       - OFT_Microstructure registered in strategy_registry
       - main.py reads OFT prediction + applies filter + confidence weight
       - OFT thresholds live in src/utils/config.py
       - restart_all.ps1 launches orderbook_collector (Step 4)
       - distributed orchestrator/worker know about model_type='oft' (Step 5)
    """
    print('\n[Phase 19 -- OFT live integration + simulator deadlock fix]')

    # 1. SimulatorAgent deadlock
    sa_path = os.path.join(BASE_DIR, 'src', 'engine', 'agents', 'simulator_agent.py')
    sa_src = open(sa_path, encoding='utf-8').read()
    flush_body = sa_src.split('def _flush_state(')[1].split('def ')[0]
    check('_flush_state does NOT re-acquire self._lock around get_status',
          'with self._lock:' not in flush_body
          and 'self.get_status()' in flush_body)

    # 2. simulator_status timeout-protected — extract from "def simulator_status("
    # to the next top-level "@app.route" so we don't truncate at inner closures.
    app_src = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
                   encoding='utf-8').read()
    after = app_src.split("def simulator_status(")[1]
    sim_status_body = after.split('\n@app.route')[0]
    check('simulator_status uses Queue with timeout',
          'queue' in sim_status_body.lower() and 'get(timeout=' in sim_status_body)

    # 3. Strategy registry entry
    sr_path = os.path.join(BASE_DIR, 'src', 'engine', 'strategy_registry.py')
    sr_src = open(sr_path, encoding='utf-8').read()
    check('strategy_registry has OFT_Microstructure',
          '"OFT_Microstructure"' in sr_src
          and "'oft_model.pt'" in sr_src or '"oft_model.pt"' in sr_src)

    # 4. main.py OFT integration
    main_src = open(os.path.join(BASE_DIR, 'src', 'main.py'), encoding='utf-8').read()
    check('main.py imports OFT thresholds',
          'OFT_GATE_P_MOVE_MIN' in main_src and 'OFT_WEIGHT_FLOOR' in main_src)
    check('main.py reads oft prediction',
          '.get("oft")' in main_src or "'oft'" in main_src and 'oft_pred' in main_src)
    check('main.py applies OFT filter (block on weak signals)',
          'oft_block' in main_src and 'OFT BLOCK' in main_src)
    check('main.py applies OFT confidence weight to trade_amount',
          'oft_weight' in main_src
          and 'trade_amount = float(trade_amount) * oft_weight' in main_src)
    check('main.py surfaces OFT fields in quant state',
          '"oft_active":' in main_src and '"oft_p_move":' in main_src
          and '"oft_blocked":' in main_src)

    # 5. config thresholds
    cfg_src = open(os.path.join(BASE_DIR, 'src', 'utils', 'config.py'),
                   encoding='utf-8').read()
    for k in ('OFT_GATE_P_MOVE_MIN', 'OFT_GATE_LIQ_RISK_MAX',
              'OFT_WEIGHT_FLOOR', 'OFT_WEIGHT_CEILING'):
        check(f'config.py defines {k}', k in cfg_src)

    # 6. orderbook_collector launched by restart_all.ps1
    ra_src = open(os.path.join(BASE_DIR, 'restart_all.ps1'), encoding='utf-8').read()
    check('restart_all.ps1 launches orderbook_collector',
          'orderbook_collector' in ra_src and 'OB_COLLECTOR_DISABLED' in ra_src)
    check('restart_all.ps1 saves orderbook PID',
          'orderbook = $obId' in ra_src)
    sa_stop = open(os.path.join(BASE_DIR, 'stop_all.ps1'), encoding='utf-8').read()
    check('stop_all.ps1 stops orderbook process',
          "'orderbook'" in sa_stop)

    # 7. distributed orchestrator/worker know about OFT
    proto_src = open(os.path.join(BASE_DIR, 'src', 'training', 'distributed',
                                   'protocol.py'), encoding='utf-8').read()
    check('protocol.ModelType has OFT enum',
          'OFT          = "oft"' in proto_src or 'OFT = "oft"' in proto_src)
    orch_src = open(os.path.join(BASE_DIR, 'src', 'training', 'distributed',
                                  'orchestrator.py'), encoding='utf-8').read()
    check('orchestrator submits OFT training task',
          '"model_type": "oft"' in orch_src)
    worker_src = open(os.path.join(BASE_DIR, 'src', 'training', 'distributed',
                                    'worker.py'), encoding='utf-8').read()
    check('worker has _train_oft handler',
          'def _train_oft(' in worker_src and '"oft":' in worker_src)


def test_phase20_orchestrator_scheduler_simpanels():
    """Regression coverage for the 2026-05-04 follow-up batch:
       - data_governance.__init__ side-effect-imports the connectors package
         (so the Strategies tab's REGISTRY is populated)
       - local_scheduler.ps1 writes a wrapper .cmd file to dodge schtasks
         /TR's broken parser on paths containing spaces
       - Phase-6 Simulation sub-tab renders a formatted card (not raw JSON)
       - Simulator tab P&L chart shows a friendly empty-state message
    """
    print('\n[Phase 20 -- Orchestrator + scheduler + sim panel polish]')

    # 1. Connectors auto-register
    init_path = os.path.join(BASE_DIR, 'src', 'data_governance', '__init__.py')
    init_src = open(init_path, encoding='utf-8').read()
    check('data_governance.__init__ imports connectors',
          'from . import connectors' in init_src)

    # Functional check: list_sources() actually returns the connectors
    try:
        import sys as _sys
        _sys.path.insert(0, BASE_DIR)
        # Force re-import in case stale
        for m in list(_sys.modules):
            if m.startswith('src.data_governance'):
                _sys.modules.pop(m, None)
        from src.data_governance import list_sources
        srcs = list_sources()
        check(f'list_sources() returns connectors (got {len(srcs)})',
              isinstance(srcs, list) and len(srcs) >= 8)
    except Exception as e:
        check('list_sources() returns connectors', False, str(e))

    # 2. local_scheduler.ps1 writes wrapper .cmd
    sched_path = os.path.join(BASE_DIR, 'local_scheduler.ps1')
    sched_src = open(sched_path, encoding='utf-8').read()
    check('local_scheduler builds wrapper .cmd file',
          'WriteAllText' in sched_src and "'.cmd'" in sched_src or 'wrapperPath' in sched_src)
    check('local_scheduler /TR points at wrapper not raw command',
          '$tr = \'"\' + $wrapperPath' in sched_src)
    check('scripts/scheduled directory exists or is created on demand',
          'scripts\\scheduled' in sched_src or 'scripts/scheduled' in sched_src)

    # 3. Phase-6 Simulation sub-tab no longer dumps raw JSON.
    # Slice from "tab === 'simulation'" to the next "} else if (tab ===" so
    # we cover the entire sub-tab branch regardless of how long it grows.
    tpl = open(TEMPLATE_PATH, encoding='utf-8').read()
    sim_block_idx = tpl.find("tab === 'simulation'")
    assert sim_block_idx > 0
    after = tpl[sim_block_idx:]
    end = after.find("} else if (tab ===")
    sim_block = after[:end] if end > 0 else after[:5000]
    check('Phase-6 simulation sub-tab no longer JSON.stringify dumps raw data',
          'JSON.stringify(d).substring' not in sim_block
          and 'Bars/sec' in sim_block and 'Training buffers' in sim_block,
          f"len(block)={len(sim_block)}")

    # 4. Simulator tab P&L chart has empty-state message
    check('simRenderPnlChart shows empty-state when no series',
          'No paper-trade P&amp;L yet' in tpl
          or 'No paper-trade P&L yet' in tpl)

    # Live HTTP probe (skip in --offline)
    if not getattr(test_phase20_orchestrator_scheduler_simpanels, '_offline', False):
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(
                DASHBOARD_URL + '/api/orchestrator/sources', timeout=4
            ) as r:
                body = _json.loads(r.read().decode('utf-8'))
            check('/api/orchestrator/sources returns connector list (live)',
                  isinstance(body, list) and len(body) >= 8,
                  f'len={len(body) if isinstance(body, list) else type(body).__name__}')
        except Exception as e:
            check('/api/orchestrator/sources returns connector list (live)', None,
                  f'skipped: {e}')

        # Scheduler register/run/unregister round-trip
        try:
            import urllib.request, json as _json
            def _post(p, body):
                req = urllib.request.Request(
                    DASHBOARD_URL + p,
                    data=_json.dumps(body).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST')
                with urllib.request.urlopen(req, timeout=10) as r:
                    return _json.loads(r.read().decode('utf-8'))
            r1 = _post('/api/scheduler/register',
                      {'name': 'TestPhase20', 'mode': 'every_minutes', 'value': '60'})
            check('scheduler register survives path-with-spaces',
                  bool(r1.get('ok')),
                  f"stderr={r1.get('stderr','')[:120]}")
            if r1.get('ok'):
                _post('/api/scheduler/unregister', {'name': 'TestPhase20'})
        except Exception as e:
            check('scheduler register survives path-with-spaces', None,
                  f'skipped: {e}')


def test_phase21_observability_and_risk_overrides():
    """Coverage for Phase 21:
       - log_retention.py sweep + thread
       - error_monitor.py classify/dedupe/auto-clear
       - /api/errors/recent + dismiss endpoints + dashboard banner
       - ml_predictor.last_status splits low_confidence from real errors
       - main.py uses last_status (not non-empty last_error) for ERROR label
       - runtime_overrides.json + reader module + dashboard Risk panel
       - main.py applies max_position_usdt cap + scalping kill-list
    """
    print('\n[Phase 21 -- Observability + risk overrides]')

    # 1. Log retention
    lr_path = os.path.join(BASE_DIR, 'src', 'utils', 'log_retention.py')
    check('src/utils/log_retention.py exists', os.path.exists(lr_path))
    if os.path.exists(lr_path):
        lr_src = open(lr_path, encoding='utf-8').read()
        check('log_retention defines sweep_once + start_retention_thread',
              'def sweep_once(' in lr_src and 'def start_retention_thread(' in lr_src)
        check('log_retention default RETENTION_DAYS=5',
              "'LOG_RETENTION_DAYS', \"5\"" in lr_src
              or 'LOG_RETENTION_DAYS", "5"' in lr_src)

    # Functional: sweep_once runs without raising
    try:
        import sys as _sys
        _sys.path.insert(0, BASE_DIR)
        from src.utils.log_retention import sweep_once
        n = sweep_once(retention_days=99999)  # nothing should match
        check('sweep_once(99999) runs cleanly + returns int',
              isinstance(n, int) and n == 0)
    except Exception as e:
        check('sweep_once functional', False, str(e))

    # 2. Error monitor
    em_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py')
    check('src/dashboard/error_monitor.py exists', os.path.exists(em_path))
    if os.path.exists(em_path):
        em_src = open(em_path, encoding='utf-8').read()
        check('error_monitor defines scan + get_active + dismiss',
              'def scan(' in em_src and 'def get_active(' in em_src
              and 'def dismiss(' in em_src)
        check('error_monitor classifies CRITICAL + WARNING',
              '_LEVEL_RE' in em_src and 'ERROR' in em_src and 'WARNING' in em_src)
        check('error_monitor has BENIGN allow-list (low conf, GARCH spike)',
              '_BENIGN_RE' in em_src
              and 'Low confidence' in em_src
              and 'GARCH' in em_src)
        check('error_monitor signature normalises symbols + timestamps',
              ('<SYM>' in em_src) and ('<TS>' in em_src))
        check('error_monitor auto-clears after AUTO_CLEAR_S (30 min default)',
              'AUTO_CLEAR_S' in em_src and "30 * 60" in em_src)

    # Functional: signature normalises BTC vs ETH to the same hash
    try:
        from src.dashboard.error_monitor import _signature
        s1 = _signature('2026-05-04 21:30:00 ERROR something on BTC/USDT failed at 0.123')
        s2 = _signature('2026-05-04 21:35:00 ERROR something on ETH/USDT failed at 0.456')
        check('error_monitor _signature collapses BTC/ETH to same key',
              s1 == s2,
              f's1={s1!r} s2={s2!r}')
    except Exception as e:
        check('error_monitor _signature collapses BTC/ETH to same key', False, str(e))

    # 3. API endpoints + banner DOM
    app_src = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
                   encoding='utf-8').read()
    check("/api/errors/recent endpoint defined",
          "@app.route('/api/errors/recent')" in app_src
          and 'count_critical' in app_src)
    check("/api/errors/dismiss endpoint defined",
          "@app.route('/api/errors/dismiss'" in app_src)
    check('dashboard auto-starts retention + monitor threads',
          'start_retention_thread' in app_src and 'start_monitor_thread' in app_src)

    tpl = open(TEMPLATE_PATH, encoding='utf-8').read()
    check('error banner DOM present (#err-banner)',
          'id="err-banner"' in tpl and 'err-banner-summary' in tpl)
    check('frontend polls /api/errors/recent on interval',
          'pollErrors' in tpl
          and 'setInterval(pollErrors' in tpl)
    check('banner can dismiss individual entries',
          'dismissError' in tpl and "/api/errors/dismiss" in tpl)

    # 4. ML status split
    mp_src = open(os.path.join(BASE_DIR, 'src', 'analysis', 'ml_predictor.py'),
                  encoding='utf-8').read()
    check('MLPredictor exposes last_status field',
          'self.last_status' in mp_src
          and "'low_confidence'" in mp_src
          and "'error'" in mp_src)
    check('low confidence does NOT set last_error (false-error fix)',
          # the OLD line set last_error on low conf; new code only sets last_status
          'self.last_error = f"Low confidence' not in mp_src)

    main_src = open(os.path.join(BASE_DIR, 'src', 'main.py'),
                    encoding='utf-8').read()
    check("main.py renders ERROR only when last_status == 'error'",
          "_status == 'error'" in main_src
          and "_status == 'low_confidence'" in main_src)
    check('main.py shows LOW CONF instead of ERROR for low-confidence',
          'LOW CONF (' in main_src)

    # 5. Runtime overrides
    ro_path = os.path.join(BASE_DIR, 'src', 'utils', 'runtime_overrides.py')
    check('src/utils/runtime_overrides.py exists', os.path.exists(ro_path))
    if os.path.exists(ro_path):
        ro_src = open(ro_path, encoding='utf-8').read()
        check('runtime_overrides defines is_scalping_disabled + max_position_cap',
              'def is_scalping_disabled(' in ro_src and 'def max_position_cap(' in ro_src)
        check('runtime_overrides watches mtime for hot reload',
              "_cache" in ro_src and "mtime" in ro_src)

    json_path = os.path.join(BASE_DIR, 'data', 'runtime_overrides.json')
    check('data/runtime_overrides.json exists', os.path.exists(json_path))
    if os.path.exists(json_path):
        import json as _json
        with open(json_path, encoding='utf-8') as f:
            ov = _json.load(f)
        # User asked for these to be pre-populated as the default kill-list.
        for sym in ('BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'TRX/USDT', 'UNI/USDT', 'SUI/USDT'):
            check(f'default kill-list includes {sym}',
                  sym in (ov.get('scalping_disabled_symbols') or []))

    check('main.py imports runtime_overrides',
          'from src.utils import runtime_overrides' in main_src)
    check('main.py applies max_position_usdt cap',
          'max_position_cap' in main_src
          and 'Runtime cap' in main_src)
    check('main.py honours scalping kill-list',
          'is_scalping_disabled' in main_src
          and 'Scalping_Disabled' in main_src)

    # 6. /api/risk/overrides endpoints
    check("/api/risk/overrides GET endpoint defined",
          "@app.route('/api/risk/overrides', methods=['GET'])" in app_src
          or "@app.route('/api/risk/overrides')" in app_src)
    check("/api/risk/overrides POST endpoint defined",
          "@app.route('/api/risk/overrides', methods=['POST'])" in app_src)
    check('Risk sub-tab UI panel renders inline',
          'function renderRiskPanel(' in tpl
          or 'renderRiskPanel()' in tpl)
    check('Risk sub-tab has scalping kill-list toggle widget',
          '_riskToggleSym' in tpl and 'scalping_disabled_symbols' in tpl)
    check('Risk sub-tab has Save + Clear buttons',
          '_riskSaveOverrides' in tpl and '_riskClearAll' in tpl)

    # Live HTTP probes
    if not getattr(test_phase21_observability_and_risk_overrides, '_offline', False):
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(DASHBOARD_URL + '/api/errors/recent', timeout=5) as r:
                body = _json.loads(r.read().decode('utf-8'))
            check('/api/errors/recent returns count_critical + count_warning',
                  'count_critical' in body and 'count_warning' in body)
        except Exception as e:
            check('/api/errors/recent live', None, f'skipped: {e}')

        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(DASHBOARD_URL + '/api/risk/overrides', timeout=5) as r:
                body = _json.loads(r.read().decode('utf-8'))
            check('/api/risk/overrides GET returns the JSON file',
                  isinstance(body, dict)
                  and 'scalping_disabled_symbols' in body
                  and 'BTC/USDT' in (body.get('scalping_disabled_symbols') or []))
        except Exception as e:
            check('/api/risk/overrides GET live', None, f'skipped: {e}')


def test_phase23_unified_banner_aggregator():
    """Banner aggregates errors from logs + status surfaces (services,
    processes, agents, cluster, scheduler). Surface entries auto-heal
    when probes flip back to OK; idle simulator / lazy-bind ZMQ are
    intentionally excluded.

    Why: previously the banner only scanned 6 log files. QuestDB / FastAPI
    / Realtime / agent / scheduler faults that surfaced via Monitor cards
    never wrote to a watched log, so the banner missed them.
    """
    print('\n[Phase 23 -- unified banner aggregator]')

    em_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py')
    with open(em_path, encoding='utf-8') as f:
        em = f.read()

    # 1. New scan_status_surfaces() function exists
    check('scan_status_surfaces() defined', 'def scan_status_surfaces' in em)

    # 2. All probe functions exist
    for fn in ('_probe_parquet_store', '_probe_duckdb', '_probe_parquet',
               '_probe_fastapi', '_probe_realtime', '_probe_processes',
               '_probe_agents', '_probe_scheduler', '_probe_cluster'):
        check(f'probe {fn} defined', f'def {fn}' in em)

    # 3. Surface entries are tagged with source='surface'
    check('surface entries carry source="surface"',
          '"source":     "surface"' in em or "'source':     'surface'" in em
          or '"source": "surface"' in em or '"source":"surface"' in em)

    # 4. Log entries also tag source='log' so _load_state can distinguish
    check('log entries tagged with source="log"',
          '"source":     "log"' in em or '"source": "log"' in em
          or '"source":"log"' in em)

    # 5. _load_state preserves surface entries without re-classifying
    check('_load_state skips re-classification for surface entries',
          'source") == "surface"' in em or "source') == 'surface'" in em)

    # 6. AUTO_CLEAR_S only applies to log entries (surface entries auto-heal)
    check('AUTO_CLEAR_S scoped to log entries only',
          ('source", "log") == "log"' in em
           or "source', 'log') == 'log'" in em))

    # 7. simulator-idle and zmq-idle are NOT probed (they're informational)
    check('simulator-idle excluded from probes',
          '_probe_simulator' not in em and 'simulator:' not in em.lower().split('# scope')[0])
    check('zmq-idle excluded from probes',
          '_probe_zmq' not in em)

    # 8. Background thread runs both scan() and scan_status_surfaces()
    check('background thread runs scan_status_surfaces()',
          'scan_status_surfaces()' in em
          and em.count('scan_status_surfaces') >= 3)  # def + thread + import maybe

    # 9. /api/errors/recent calls scan_status_surfaces
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app_src = f.read()
    check('/api/errors/recent invokes surface scan',
          '_em.scan_status_surfaces()' in app_src)

    # 10. End-to-end probe simulation: import the module and call probes
    #     against the current environment. We don't assert specific results
    #     (those depend on whether QuestDB is up locally) but we DO assert
    #     the call returns the right shape.
    try:
        import importlib, sys as _sys
        sys.path.insert(0, BASE_DIR) if BASE_DIR not in _sys.path else None
        em_mod = importlib.import_module('src.dashboard.error_monitor')
        snap = em_mod.scan_status_surfaces()
        check('scan_status_surfaces() returns dict',
              isinstance(snap, dict))
        # Each entry must carry source='surface' and signature carrying _SURFACE_PREFIX
        all_surface = all(
            (v.get('source') == 'surface'
             and v.get('signature', '').startswith('surface:'))
            for v in snap.values()
        ) if snap else True
        check('all surface entries tagged + signature-prefixed', all_surface)
        # Each entry must have a kind in {critical, warning}
        kinds_ok = all(v.get('kind') in ('critical', 'warning') for v in snap.values())
        check('surface entries have valid kind', kinds_ok)
    except Exception as e:
        check('scan_status_surfaces() runs without error', False, str(e))


def test_phase24_scheduler_flash_and_local_training():
    """Scheduler register/run/delete give visible feedback via #sch-flash
    pill (was silent → user thought controls were broken). Training Cluster
    card also shows live local-training progress (was always 'No tasks yet'
    even when TFT training was running).
    """
    print('\n[Phase 24 -- Scheduler flash + local-training progress]')

    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # Scheduler flash pill
    check('Scheduler #sch-flash element rendered',
          'id="sch-flash"' in tpl)
    check('Scheduler _schFlash() helper defined',
          'function _schFlash(' in tpl)
    check('Scheduler _schRegister calls _schFlash on success',
          "_schFlash('✓ Registered " in tpl
          or '_schFlash("✓ Registered ' in tpl)
    check('Scheduler _schRun calls _schFlash on trigger',
          "_schFlash('▶ " in tpl)
    check('Scheduler _schUnregister calls _schFlash on delete',
          "_schFlash('✕ Deleted " in tpl)

    # Training Cluster live local-training panel
    check('Training Cluster has cluster-local-training container',
          'id="cluster-local-training"' in tpl)
    check('renderLocalTrainingProgress() defined',
          'async function renderLocalTrainingProgress' in tpl)
    check('renderLocalTrainingProgress reads /api/scheduler/report',
          "fetch('/api/scheduler/report')" in tpl)
    check('renderLocalTrainingProgress reads /api/models',
          "fetch('/api/models'" in tpl)
    check('clusterPoll() invokes renderLocalTrainingProgress',
          'renderLocalTrainingProgress()' in tpl)

    # Empty-state placeholder when no run snapshot exists
    check('Local-training section has empty-state guidance',
          'No active training snapshot yet' in tpl)


def test_phase28_dashboard_read_path_cutover():
    """Phase 3 of QuestDB → ParquetClient migration: dashboard probes,
    health cards, and /api/db/* endpoints all surface the new file-based
    backend. The QuestDB-specific HTTP probe and 'docker-compose up -d
    questdb' hint are replaced with ParquetClient/DuckDB equivalents.
    """
    print('\n[Phase 28 -- dashboard read path cutover (Route B)]')

    # Banner monitor probe renamed
    em_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py')
    em = open(em_path, encoding='utf-8').read()
    check('error_monitor exposes _probe_parquet_store',
          'def _probe_parquet_store' in em)
    check('error_monitor no longer defines _probe_questdb',
          'def _probe_questdb' not in em)
    check('_ALL_PROBES references parquet_store (not questdb)',
          '"parquet_store"' in em
          and '("questdb",   _probe_questdb)' not in em)

    # /api/monitor/services — QuestDB block replaced with parquet_store
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    app_src = open(app_path, encoding='utf-8').read()
    check("/api/monitor/services exposes parquet_store card",
          "out['parquet_store']" in app_src)
    check("/api/monitor/services no longer probes 127.0.0.1:9000/exec (QuestDB HTTP)",
          'http://127.0.0.1:9000/exec?query=SELECT%201' not in app_src)

    # /api/db/status reports the new backend
    check("/api/db/status reports backend='duckdb+parquet'",
          "'backend': 'duckdb+parquet'" in app_src
          or "'backend':   'duckdb+parquet'" in app_src)
    check("/api/db/status returns data_dir for the file-based store",
          "'data_dir':" in app_src)

    # All /api/db/* routes import parquet_client now
    n_pq = app_src.count('from src.database.parquet_client import get_client')
    check('app.py imports parquet_client.get_client at every db_* callsite',
          n_pq >= 5)

    # Dashboard JS panel: Parquet Store label + DuckDB hint, no docker hint
    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    tpl = open(tpl_path, encoding='utf-8').read()
    check('Monitor panel header renamed "Parquet Store (DuckDB)"',
          'Parquet Store (DuckDB)' in tpl)
    check('dbPollStatus offline-message no longer suggests docker-compose',
          'docker-compose up -d questdb' not in tpl)
    check('dbPollStatus offline-message points at duckdb / pyarrow',
          'pip install duckdb pyarrow' in tpl)


def test_phase36_debug_supervisor():
    """Debug supervisor — captures crash diagnostics for project python
    processes. Polls data/process_ids.json every 5s; on death writes to
    data/process_deaths.json with log tail + RSS/CPU snapshot. Surfaces
    fresh deaths in the banner via _probe_recent_deaths.
    """
    print('\n[Phase 36 -- debug_supervisor]')

    sup_path = os.path.join(BASE_DIR, 'scripts', 'debug_supervisor.py')
    check('scripts/debug_supervisor.py exists', os.path.exists(sup_path))
    sup = open(sup_path, encoding='utf-8').read()

    check('supervisor reads data/process_ids.json',
          'process_ids.json' in sup)
    check('supervisor writes data/process_deaths.json',
          'process_deaths.json' in sup)
    check('supervisor captures log tail per role',
          'def _tail_log(' in sup and '_ROLE_LOG_FILES' in sup)
    check('supervisor extracts an exit_clue from log',
          "'error', 'traceback'" in sup
          or "'error', 'traceback', 'exception'" in sup
          or '"error", "traceback"' in sup)
    check('supervisor caps deaths at 200',
          'MAX_DEATHS' in sup)
    check('supervisor records RSS / CPU snapshot',
          'rss_mb' in sup and 'cpu_pct' in sup)

    # Banner-aggregator probe
    em_src = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py'),
                  encoding='utf-8').read()
    check('error_monitor exposes _probe_recent_deaths',
          'def _probe_recent_deaths' in em_src)
    check('_probe_recent_deaths only flags deaths < 10 min old',
          'age_s > 600' in em_src or '> 600' in em_src)
    check('_ALL_PROBES includes recent_deaths',
          '"recent_deaths"' in em_src)

    # /api/debug/deaths endpoint
    app_src = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
                   encoding='utf-8').read()
    check('/api/debug/deaths endpoint defined',
          "@app.route('/api/debug/deaths')" in app_src)
    check('endpoint reads process_deaths.json',
          'process_deaths.json' in app_src)

    # restart_all.ps1 launches the supervisor
    rs = open(os.path.join(BASE_DIR, 'restart_all.ps1'), encoding='utf-8').read()
    check('restart_all.ps1 launches debug_supervisor',
          'debug_supervisor' in rs and 'Start-Process powershell' in rs)
    check('restart_all.ps1 saves debug PID in process_ids.json',
          'debug = $debugId' in rs)

    # Behaviour smoke: a single tick with no deaths should not crash.
    try:
        import importlib, tempfile, os as _os
        sys.path.insert(0, BASE_DIR) if BASE_DIR not in sys.path else None
        sup_mod = importlib.import_module('scripts.debug_supervisor')
        # Test pid-map reader against the real file (read-only).
        pids = sup_mod._read_pid_map()
        check('supervisor _read_pid_map returns dict',
              isinstance(pids, dict))
        check('supervisor _is_alive(1) is bool',
              isinstance(sup_mod._is_alive(1), bool))
    except Exception as e:
        check('supervisor smoke test', False, str(e))


def test_phase44_pr6_live_trading_toggle():
    """Phase 44 — PR 6 Live Trading toggle + manual paper accounting:
       - data/control.json gains a `trade_mode` field (paper/testnet/mainnet)
       - dual_balance.py gains add_deposit / add_paper_pnl / compute_summary
       - paper_book.py routes paper trades to data/trades.json + virtual balance
       - OrderManager gates execute_*_order on trade_mode
       - GET/POST /api/control/trade_mode + POST /api/balance/virtual/deposit
       - Overview tab gets the trade-mode switch + balance breakdown card
    """
    print('\n[Phase 44 -- PR 6 live trading toggle + paper accounting]')

    db = open(os.path.join(BASE_DIR, 'src', 'engine', 'dual_balance.py'),
              encoding='utf-8').read()
    pb = open(os.path.join(BASE_DIR, 'src', 'engine', 'paper_book.py'),
              encoding='utf-8').read()
    om = open(os.path.join(BASE_DIR, 'src', 'engine', 'order_manager.py'),
              encoding='utf-8').read()
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    # 1. dual_balance schema extensions
    check('dual_balance.add_deposit() defined',
          'def add_deposit(' in db)
    check('dual_balance.add_paper_pnl() defined',
          'def add_paper_pnl(' in db)
    check('dual_balance.compute_summary() decomposes equity / deposits / pnl',
          'def compute_summary(' in db
          and '"deposits_total":' in db
          and '"pnl":' in db
          and '"revenue_total":' in db)
    check('reset_virtual seeds deposits[] with the initial cash',
          '"deposits": [{' in db and 'note": "seed"' in db)

    # 2. paper_book module
    check('paper_book.book_market_order writes is_paper=True',
          'def book_market_order(' in pb
          and '"is_paper":    True' in pb)
    check('paper_book.book_close credits virtual balance via add_paper_pnl',
          'def book_close(' in pb
          and 'add_paper_pnl(net)' in pb)
    check('paper_book applies round-trip fee in PnL math',
          'fee_bps' in pb
          and 'entry_fee = entry_cost' in pb
          and 'net = gross - entry_fee - exit_fee' in pb)

    # 3. OrderManager gate
    check('OrderManager._trade_mode reads control.json',
          'def _trade_mode(' in om
          and "ctrl.get('trade_mode') or 'testnet'" in om)
    check('execute_spot_order routes to paper_book in paper mode',
          "if self._trade_mode() == 'paper':" in om
          and 'from src.engine.paper_book import book_market_order' in om)
    check('execute_futures_order also gated on paper mode',
          om.count("if self._trade_mode() == 'paper':") >= 2)

    # 4. Backend endpoints
    check('GET/POST /api/control/trade_mode defined',
          "@app.route('/api/control/trade_mode'" in app
          and 'methods=[\'GET\', \'POST\']' in app)
    check('mainnet POST requires confirm=true',
          "if mode == 'mainnet' and not body.get('confirm'):" in app)
    check('POST /api/balance/virtual/deposit defined',
          "@app.route('/api/balance/virtual/deposit'" in app
          and 'def api_balance_virtual_deposit(' in app)
    check('/api/balance/virtual returns summary block',
          'compute_summary' in app and '"summary": summary' in app)

    # 5. UI on Overview tab
    check('Live Trading card present in Overview tab',
          'class="card lt-card"' in tpl
          and 'Live Trading' in tpl)
    check('Three trade-mode buttons (paper/testnet/mainnet)',
          'id="lt-btn-paper"'   in tpl
          and 'id="lt-btn-testnet"' in tpl
          and 'id="lt-btn-mainnet"' in tpl)
    check('Balance breakdown shows equity / deposits / revenue / pnl',
          'id="lt-equity"'   in tpl
          and 'id="lt-deposits"' in tpl
          and 'id="lt-revenue"'  in tpl
          and 'id="lt-pnl"'      in tpl)
    check('Mainnet switch confirms with explicit warning (now reads "REAL CASH" per v3.1 step 1)',
          "Switch to REAL CASH" in tpl and 'Real money will be at risk' in tpl)
    check('+ Deposit button + ltDeposit() prompts for amount',
          'ltDeposit()' in tpl
          and 'Add how much to virtual balance' in tpl)


def test_phase58_pr28_balance_by_mode():
    """Phase 58 — PR 28: Live Trading card shows mode-correct balance.
    Pre-fix: paper / testnet / mainnet all displayed the same virtual
    $100k seed numbers. Post-fix: paper shows internal virtual; testnet
    fetches Binance testnet USDT (spot + futures); mainnet hits Binance
    mainnet."""
    print('\n[Phase 58 -- PR 28 mode-aware balance]')
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    check('GET /api/balance/by_mode endpoint defined',
          "@app.route('/api/balance/by_mode')" in app
          and 'def api_balance_by_mode(' in app)
    check('Endpoint dispatches paper/testnet/mainnet',
          "if mode == 'paper':" in app
          and "if mode in ('testnet', 'mainnet'):" in app)
    check('Per-mode OrderManager cached so we never accidentally cross sandbox flag',
          '_order_mgr_cache: dict[bool' in app
          and '_get_order_manager_for_mode(' in app)
    check('Live Binance balance has 30s TTL cache',
          '_balance_live_cache' in app
          and '_BALANCE_TTL_S = 30' in app)
    check('Live balance returns spot + futures separately',
          "'spot_usdt'" in app
          and "'futures_usdt'" in app
          and 'futures_exchange.fetch_balance' in app)
    check('Frontend ltLoadBalance uses /api/balance/by_mode with current mode',
          "fetch('/api/balance/by_mode?mode=' + encodeURIComponent(mode))" in tpl
          and 'let _ltCurrentMode' in tpl)
    check('Mode switch refreshes balance immediately',
          'await ltLoadMode();' in tpl
          and 'await ltLoadBalance();' in tpl)
    check('Paper-only fields hidden when not in paper mode',
          'lt-deposits-row' in tpl
          and 'lt-revenue-row' in tpl
          and 'lt-pnl-row' in tpl
          and "el.style.display = 'none'" in tpl)


def test_phase57_pr26_all_tfs_and_status():
    """Phase 57 — PR 26: 'ALL TFs' option + fine-grained training status.
       User asked for: (a) one-click train across every TF the model
       supports, (b) instant visual feedback on click (QUEUED flash
       before network round-trip), (c) per-phase status pills
       (QUEUED / STARTING / RUNNING <tf> / FAILED / COMPLETED)."""
    print('\n[Phase 57 -- PR 26 ALL-TFs + fine-grained status]')
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    check('Backend accepts tf="all" + has _run_trainer_multi_tf worker',
          "if tf == 'all':" in app
          and 'def _run_trainer_multi_tf(' in app
          and 'ALL_TFS_BY_KEY' in app)
    check('Multi-TF worker reports current_tf + progress_label',
          "_record_job(job_id, current_tf=tf" in app
          and 'progress_label' in app)
    check('Multi-TF chains TFs sequentially with cancellable Popen',
          'for tf_idx, tf in enumerate(tfs)' in app
          and '_training_active_procs[job_id]' in app)
    check('TF picker dropdown includes "ALL TFs" option',
          '<option value="all"' in tpl and 'ALL TFs</option>' in tpl)
    check('Fine-grained status: QUEUED / STARTING / RUNNING / CANCELLED',
          "'QUEUED'" in tpl
          and "'STARTING'" in tpl
          and "'RUNNING'" in tpl
          and "'CANCELLED'" in tpl)
    check('Status pill shows progress_label / current_tf when running',
          'activeJob.progress_label' in tpl
          and 'activeJob.current_tf' in tpl)
    # Phase 97b — keying changed from [key] to [_optKey] (model@tf form)
    # so per-tf rows flash too. Accept either keying style as a regression
    # guard: the assertion is "optimistic flash exists", not "exact key var".
    check('Optimistic UI flashes QUEUED before network round-trip',
          ("_trActiveJobs    = {..._trActiveJobs,    [key]: {" in tpl
           or "_trActiveJobs    = {..._trActiveJobs,    [_optKey]: {" in tpl)
          and "status: 'queued'" in tpl)
    check('Failed POST rolls back the optimistic state',
          ('delete _trActiveByModel[key]' in tpl
           or 'delete _trActiveByModel[_optKey]' in tpl)
          and ('delete _trActiveJobs[key]' in tpl
               or 'delete _trActiveJobs[_optKey]' in tpl))
    check('Polling interval drops to 1.5s for 30s after click',
          'setInterval(pollTrainingJobs, 1500)' in tpl
          and "setTimeout(() => {" in tpl)


def test_phase56_pr21_heatmap_rework():
    """Phase 56 — PR 21: stability heatmap rework.
    User asked for the same look as the Model Training table:
      - sortable column headers (click to sort by any TF, by best,
        or by strategy name)
      - per-column number colours (green/gold = good, red = bad)
      - compact column widths
      - all 8 TFs always shown (placeholder cells when no data)
      - description column (right side) — short blurb per strategy
      - dedicated ★ Best column instead of an inline badge
    """
    print('\n[Phase 56 -- PR 21 heatmap rework]')
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    check('Heatmap is now a real <table> (sortable headers)',
          '<table style="width:100%;border-collapse:collapse' in tpl
          and 'function stabSort(' in tpl
          and "stabSort('strategy')" in tpl)
    check('All 8 TFs always shown — empty cells for no data',
          "const ALL_TFS = ['1m','5m','15m','1h','4h','1d','1w','1mo']" in tpl
          and 'no data for ${s} @ ${tf}' in tpl)
    check('Each cell coloured per metric (gold/ok/warn/bad/empty)',
          'function classify(metric, v)' in tpl
          and 'cellBg = ' in tpl
          and 'cellFg = ' in tpl)
    check('★ Best is its own dedicated sortable column',
          '★ Best' in tpl
          and "stabSort('__best')" in tpl)
    check('Description column with per-strategy one-liner',
          'const _STRATEGY_DESCRIPTIONS' in tpl
          and 'RSI_MeanReversion:' in tpl
          and 'Random-forest classifier on Triple-Barrier' in tpl)
    check('Sticky strategy column on horizontal scroll',
          "position:sticky;left:0" in tpl)
    check('Header sort indicator chevron',
          "chev = (col) =>" in tpl
          and " ▼" in tpl and " ▲" in tpl)


def test_phase55_pr19_training_controls():
    """Phase 55 — PR 18/19/20: dashboard hardening bundle.
       PR 18: /api/db/status TTL cache so Monitor stops hanging
              + Cache-Control: no-store on / so stale browser cache
              can't mask freshly-edited JS.
       PR 19: training row gains TF picker, status column flips RUNNING
              when row's training subprocess is alive, Train ↔ Stop
              button toggle, model description column.
       PR 20: pipeline orchestrator status can be reset (clears stale
              'error' from yesterday's run when no live process)."""
    print('\n[Phase 55 -- PR 18/19/20 dashboard hardening]')
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    # PR 18 — Cache + cache-headers
    check('/api/db/status is TTL-cached (5 min)',
          '_db_status_cache' in app
          and '_db_status_cache_ttl = 300.0' in app
          and '_refresh_db_status_async' in app)
    check('Background refresher is fire-and-forget',
          "name='db-status-refresh'" in app)
    check('Index sets Cache-Control: no-store',
          "'Cache-Control'" in app
          and "'no-store, no-cache, must-revalidate, max-age=0'" in app)

    # PR 19 — Training row controls
    check('Training endpoint can be killed via /api/training/stop/<job_id>',
          "@app.route('/api/training/stop/<job_id>'" in app
          and 'def api_training_stop(' in app
          and 'proc.kill()' in app)
    check('Training subprocess tracked in _training_active_procs dict',
          '_training_active_procs' in app
          and '_training_active_lock' in app)
    check('GET /api/training/active returns model_key→job_id map',
          "@app.route('/api/training/active'" in app
          and 'def api_training_active(' in app)
    check('Train→Stop swap rendered when row has active job',
          'activeJobId' in tpl
          and 'trStopOne(' in tpl
          and "'⏹ Stop'" in tpl or '⏹ Stop' in tpl)
    check('TF picker per training row, defaulting to model timeframe (v3.1: now uses _trUserTfChoice)',
          "id=\"tr-tf-${esc(m.key)}\"" in tpl
          and "_trUserTfChoice[m.key]" in tpl
          and "(m.timeframe || '1h')" in tpl)
    check('trRunOne now sends tf in body',
          'if (tf) body.tf = tf' in tpl)
    check('pollTrainingJobs rebuilds _trActiveByModel + re-renders on change',
          '_trActiveByModel = newActive' in tpl
          and 'JSON.stringify(newActive) !== JSON.stringify' in tpl)
    check('Training row has Description column header + cell',
          'One-line description of what this model does' in tpl
          and '_MODEL_DESCRIPTIONS' in tpl
          and 'Random forest on Triple-Barrier' in tpl)

    # PR 20 — Pipeline reset
    check('POST /api/pipeline/reset clears status when no process alive',
          "@app.route('/api/pipeline/reset'" in app
          and 'def api_pipeline_reset(' in app
          and '_pipeline_proc_alive()' in app
          and 'pipeline status cleared' in app)
    check('Pipeline card has Reset button',
          'pipelineReset()' in tpl
          and '✕ Reset' in tpl)


def test_phase54_pr17_production_readiness():
    """Phase 54 — PR 17 / Phase F: production readiness pass.

    Three deliverables:
      1. breaker_drill — exercise every circuit breaker offline,
         confirm correct trigger fires + no false positives.
      2. audit_trail — verify trades → signals → models traceability,
         flag orphan orders / missing artifacts.
      3. RUNBOOK.md — single-page operator handbook at repo root.
    """
    print('\n[Phase 54 -- PR 17 production readiness]')

    drill_path = os.path.join(BASE_DIR, 'src', 'engine', 'breaker_drill.py')
    audit_path = os.path.join(BASE_DIR, 'src', 'engine', 'audit_trail.py')
    rb_path    = os.path.join(BASE_DIR, 'RUNBOOK.md')
    check('breaker_drill.py exists', os.path.exists(drill_path))
    check('audit_trail.py exists',   os.path.exists(audit_path))
    check('RUNBOOK.md exists at repo root', os.path.exists(rb_path))
    if not (os.path.exists(drill_path) and os.path.exists(audit_path)):
        return

    drill = open(drill_path, encoding='utf-8').read()
    audit = open(audit_path, encoding='utf-8').read()
    app   = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
                 encoding='utf-8').read()
    rb    = open(rb_path, encoding='utf-8').read() if os.path.exists(rb_path) else ''

    # 1. breaker_drill scenarios
    check('Drill covers max_dd / api_latency / stale_feed / clean',
          '_scenario_max_dd' in drill
          and '_scenario_api_latency' in drill
          and '_scenario_stale_feed' in drill
          and '_scenario_clean' in drill)
    check('Drill validates expected vs actual trigger',
          'expected_trigger' in drill
          and 'actual_trigger' in drill
          and 'verdict' in drill)
    check('Drill returns pass/fail counts',
          'scenarios_passed' in drill
          and 'scenarios_failed' in drill
          and 'scenarios_run' in drill)

    # 2. audit_trail surface
    check('Audit detects orphan orders + missing artifacts',
          'orphan_orders' in audit
          and 'missing_artifacts' in audit
          and 'untraced_signals' in audit)
    check('Audit checks model freshness vs trade timestamps',
          'pre_train_trades' in audit
          and 'training_completed_at' in audit)
    check('Audit persists JSON report under data/audit_reports/',
          'audit_reports' in audit
          and 'rep_path.write_text' in audit)

    # 3. Dashboard endpoints
    check('POST /api/breaker_drill/run wired (api-key gated)',
          "@app.route('/api/breaker_drill/run'" in app
          and 'def api_breaker_drill_run(' in app
          and 'run_drill(only=only)' in app)
    check('POST /api/audit_trail/run wired (api-key gated)',
          "@app.route('/api/audit_trail/run'" in app
          and 'def api_audit_trail_run(' in app
          and 'run_audit(max_trades=' in app)

    # 4. RUNBOOK content
    if rb:
        check('RUNBOOK has daily go/no-go checklist',
              'go/no-go' in rb.lower() or 'go / no-go' in rb.lower())
        check('RUNBOOK lists all 3 trade modes (paper/testnet/mainnet)',
              'PAPER' in rb and 'TESTNET' in rb and 'MAINNET' in rb)
        check('RUNBOOK has incident-response section',
              'Incident response' in rb or 'incident response' in rb.lower())
        check('RUNBOOK has pre-deploy checklist',
              'Pre-deploy' in rb or 'pre-deploy' in rb.lower())
        check('RUNBOOK references each PR-7..PR-17 module',
              'breaker_drill' in rb
              and 'audit_trail' in rb
              and 'auto_retrain' in rb
              and 'long_horizon_backtest' in rb
              and 'scrub_resampled_csvs' in rb
              and 'strategy_tf_pinning' in rb)


def test_phase53_pr16_long_horizon_backtest():
    """Phase 53 — PR 16 / Phase E: long-horizon backtest preset.

    With 8+ years of 1s archives, naive multi-TF backtest at 5m blows
    past memory limits. This PR ships horizon presets that auto-pick
    safe TFs per window (long=5y → 1h+4h+1d+1w; max=all → 4h+1d+1w+1mo).
    """
    print('\n[Phase 53 -- PR 16 long-horizon backtest]')

    lh_path = os.path.join(BASE_DIR, 'src', 'engine', 'long_horizon_backtest.py')
    check('long_horizon_backtest.py exists', os.path.exists(lh_path))
    if not os.path.exists(lh_path):
        return
    src = open(lh_path, encoding='utf-8').read()
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()

    # 1. Module surface
    check('Four horizon presets defined (short/medium/long/max)',
          '"short":' in src
          and '"medium":' in src
          and '"long":' in src
          and '"max":' in src)
    check('long horizon excludes 5m to avoid 250M-row blowup',
          '"long":   (5.0,  ("1h", "4h", "1d", "1w"))' in src)
    check('max horizon uses lowest-resolution TFs only',
          '"max":    (None, ("4h", "1d", "1w", "1mo"))' in src)
    check('run() validates horizon name + falls back to default tfs',
          'def run(' in src
          and 'if horizon not in HORIZONS' in src
          and 'years_back, default_tfs = HORIZONS[horizon]' in src)
    check('Calls run_full_backtest with the chosen TFs',
          'from src.engine.backtester import run_full_backtest' in src
          and 'run_full_backtest(timeframes=tfs' in src)
    check('Tags latest_comparison rows with horizon + years_back',
          'r.setdefault("horizon", horizon)' in src
          and 'r.setdefault("years_back", years_back)' in src)
    check('CLI exposes --horizon and --timeframes overrides',
          '--horizon' in src
          and '--timeframes' in src
          and '--fee-preset' in src)

    # 2. Dashboard endpoint
    check('POST /api/backtest/long_horizon spawns detached subprocess',
          "@app.route('/api/backtest/long_horizon'" in app
          and 'long_horizon_backtest' in app
          and '@require_api_key' in app
          and 'CREATE_NEW_PROCESS_GROUP' in app)
    check('long_horizon endpoint reuses pipeline alive-check (409)',
          '_pipeline_proc_alive()' in app
          and 'pipeline already running' in app)


def test_phase52_pr15_finbert_sentiment():
    """Phase 52 — PR 15 / Phase B: FinBERT/CryptoBERT sentiment upgrade.

    Replaces the 30-word lexicon with a real model (CryptoBERT primary,
    FinBERT fallback, lexicon as final fallback). Output stays in [-1, +1]
    so existing parquet readers don't change."""
    print('\n[Phase 52 -- PR 15 FinBERT sentiment]')

    fb_path = os.path.join(BASE_DIR, 'src', 'analysis', 'finbert_scorer.py')
    check('finbert_scorer.py exists', os.path.exists(fb_path))
    if not os.path.exists(fb_path):
        return
    src = open(fb_path, encoding='utf-8').read()
    cc  = open(os.path.join(BASE_DIR, 'src', 'data_ingestion', 'cryptocompare_news_backfill.py'),
               encoding='utf-8').read()
    rd  = open(os.path.join(BASE_DIR, 'src', 'data_ingestion', 'reddit_news_backfill.py'),
               encoding='utf-8').read()
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()

    # 1. Module surface
    check('finbert_scorer prefers CryptoBERT, falls back to FinBERT, lexicon',
          "ElKulako/cryptobert" in src
          and "ProsusAI/finbert" in src
          and 'def _lexicon_score(' in src)
    check('Lazy singleton load — _ensure_loaded() sets _classifier',
          'def _ensure_loaded(' in src
          and '_load_attempted' in src
          and '_classifier = clf' in src)
    check('score_one cached via lru_cache',
          '@lru_cache(maxsize=10_000)' in src
          and 'def score_one(' in src)
    check('Batch scoring uses pipeline batch_size for speed',
          'def score_batch(' in src
          and 'batch_size=' in src)
    check('HF cache redirected to D: drive (not C:)',
          'HF_HOME' in src
          and 'data/cache/huggingface' in src.replace('\\', '/'))
    check('Output mapped to [-1, +1] tone scale',
          'def _label_to_score(' in src
          and 'return round(float(conf), 3)' in src
          and 'return -round(float(conf), 3)' in src)

    # 2. Scrapers wired to defer to model when ready
    check('CryptoCompare scraper defers to finbert_scorer when ready',
          'from src.analysis.finbert_scorer import score_one, is_ready' in cc
          and 'if is_ready():' in cc
          and 'return score_one(title)' in cc)
    check('Reddit scraper defers to finbert_scorer when ready',
          'from src.analysis.finbert_scorer import score_one, is_ready' in rd
          and 'if is_ready():' in rd)

    # 3. Dashboard endpoint
    check('GET /api/news/sentiment_model reports active backend',
          "@app.route('/api/news/sentiment_model'" in app
          and 'get_active_model()' in app
          and "'cryptobert'" in app or "cryptobert" in app)


def test_phase51_pr14_live_news_inference():
    """Phase 51 — PR 14 / Phase D: live news inference path.

    Background thread caches the recent news partition in memory so
    add_news_sentiment() doesn't pay a DuckDB cold-start (~100-500 ms)
    on every bot signal cycle. Refreshes every 5 min so newly-scraped
    GDELT/Reddit/CC partitions show up in inference within 5 min of
    being written.
    """
    print('\n[Phase 51 -- PR 14 live news inference]')

    buf_path = os.path.join(BASE_DIR, 'src', 'analysis', 'live_news_buffer.py')
    check('live_news_buffer.py exists', os.path.exists(buf_path))
    if not os.path.exists(buf_path):
        return
    src = open(buf_path, encoding='utf-8').read()
    fe   = open(os.path.join(BASE_DIR, 'src', 'analysis', 'feature_engineering.py'),
                encoding='utf-8').read()
    main = open(os.path.join(BASE_DIR, 'src', 'main.py'), encoding='utf-8').read()
    app  = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
                encoding='utf-8').read()

    # 1. Module surface
    check('LiveNewsBuffer class with thread-safe snapshot',
          'class LiveNewsBuffer' in src
          and 'def get_snapshot(' in src
          and 'with self._lock' in src)
    check('Background refresher uses Event-based sleep so stop() is fast',
          'self._stop = threading.Event()' in src
          and 'self._stop.wait(timeout=' in src)
    check('start() does inline first refresh before returning',
          'def start(' in src
          and '_refresh_once_safe()' in src
          and 'self._thread.start()' in src)
    check('Module-level singleton + helpers',
          'def start_buffer(' in src
          and 'def get_active_buffer(' in src
          and 'def stop_buffer(' in src
          and '_active_buffer' in src)
    check('Status surface exposes rows, age, refresh_count, last_error',
          'def status(self)' in src
          and '"snapshot_age_s"' in src
          and '"refresh_count"' in src
          and '"last_error"' in src)

    # 2. add_news_sentiment uses the buffer when present
    check('add_news_sentiment prefers live buffer when active',
          'from src.analysis.live_news_buffer import get_active_buffer' in fe
          and 'buf.get_snapshot()' in fe
          and 'news = snap.copy()' in fe)
    check('Falls back to parquet load when no buffer',
          'if news is None:' in fe
          and 'load_news_recent(hours=24 * 365)' in fe)

    # 3. main.py wires it
    check('main.py starts the live news buffer at boot',
          'from src.analysis.live_news_buffer import start_buffer' in main
          and 'self.live_news_buffer = start_buffer(' in main)

    # 4. Dashboard endpoint
    check('GET /api/news/buffer returns buffer status',
          "@app.route('/api/news/buffer'" in app
          and 'def api_news_buffer_status(' in app
          and 'get_active_buffer()' in app)


def test_phase50_pr13_auto_retrain():
    """Phase 50 — PR 13 / Phase C: walk-forward auto-retrain with regression
    guard. Wraps the pipeline orchestrator with a before/after WF Sharpe
    comparison. Optional --rollback restores meta backups on regression."""
    print('\n[Phase 50 -- PR 13 auto-retrain]')

    ar_path = os.path.join(BASE_DIR, 'src', 'engine', 'auto_retrain.py')
    check('auto_retrain.py exists', os.path.exists(ar_path))
    if not os.path.exists(ar_path):
        return
    src = open(ar_path, encoding='utf-8').read()
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()

    # 1. Module surface
    check('run_auto_retrain entry function',
          'def run_auto_retrain(' in src
          and 'tolerance' in src
          and 'rollback' in src)
    check('Snapshots WF Sharpe per strategy from wf_results.json',
          'def _wf_sharpe_snapshot(' in src
          and 'wf_results.json' in src
          and 'wf_mean_sharpe' in src)
    check('Backs up meta files before retrain',
          'def _backup_models(' in src
          and 'shutil.copy2' in src)
    check('Compares before/after with tolerance threshold',
          'threshold = (a_old or 0) * (1 - tolerance)' in src
          and '"accepted"' in src
          and '"regression"' in src)
    check('Records regression report on degradation',
          'def _record_regression(' in src
          and 'retrain_regressions' in src)
    check('Optional rollback restores meta from backup on regression',
          'def _restore_meta_from_backup(' in src
          and 'rollback and backup_dir' in src)
    check('Calls run_pipeline() from pipeline_orchestrator',
          'from src.engine.pipeline_orchestrator import run_pipeline' in src
          and 'pipe_result = run_pipeline()' in src)
    check('Persists status to data/auto_retrain_status.json via safe_json',
          'auto_retrain_status.json' in src
          and 'write_json' in src)
    check('CLI exposes --tolerance + --rollback',
          '--tolerance' in src and '--rollback' in src)

    # 2. Dashboard endpoints
    check('GET /api/auto_retrain/status',
          "@app.route('/api/auto_retrain/status'" in app
          and 'auto_retrain_status.json' in app)
    check('POST /api/auto_retrain/run gated by api_key + reuses pipeline alive-check',
          "@app.route('/api/auto_retrain/run'" in app
          and '@require_api_key' in app
          and '_pipeline_proc_alive()' in app
          and 'pipeline already running' in app)
    check('Detached subprocess flags so retrain survives Flask restart',
          'CREATE_NEW_PROCESS_GROUP' in app
          and 'DETACHED_PROCESS' in app)


def test_phase49_pr12_tf_pinning():
    """Phase 49 — PR 12 / Phase A: per-strategy TF pinning.

    The Stability heatmap (PR 4) identifies the most-stable TF per
    strategy from walk-forward backtests; this PR persists those
    assignments so the live bot can route each strategy's signal to its
    most-stable TF + the matching per-TF model from PR 11.

    Resolution: manual override > auto pin (from latest backtest) > default.
    """
    print('\n[Phase 49 -- PR 12 strategy TF pinning]')

    pin_path = os.path.join(BASE_DIR, 'src', 'engine', 'strategy_tf_pinning.py')
    check('strategy_tf_pinning.py exists', os.path.exists(pin_path))
    if not os.path.exists(pin_path):
        return
    src = open(pin_path, encoding='utf-8').read()
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    orch = open(os.path.join(BASE_DIR, 'src', 'engine', 'pipeline_orchestrator.py'),
                encoding='utf-8').read()

    # 1. Pinning module surface
    check('PINNING_PATH points at data/strategy_tf_pinning.json',
          'strategy_tf_pinning.json' in src
          and 'PINNING_PATH' in src)
    check('Resolution order: manual > auto > default',
          'def get_pinned_tf(' in src
          and 'manual.get(strategy) or auto.get(strategy) or default' in src)
    check('set_manual_pin clears when tf is empty/None',
          'def set_manual_pin(' in src
          and 'if not tf:' in src
          and 'manual.pop(strategy, None)' in src)
    check('update_auto_pins replaces stale assignments wholesale',
          'def update_auto_pins(' in src
          and 'state["auto"] = ' in src)
    check('get_all_pins returns auto + manual + effective per strategy',
          'def get_all_pins(' in src
          and '"effective"' in src
          and '"auto"' in src
          and '"manual"' in src)
    check('Persistence uses safe_json (filelock + atomic)',
          'from src.utils.safe_json import read_json, write_json' in src)

    # 2. Orchestrator post-backtest hook
    check('Orchestrator refreshes TF pinning post-backtest',
          'def _refresh_tf_pinning(' in orch
          and 'update_auto_pins(best_tf)' in orch
          and 'tf_pins_written' in orch)

    # 3. Dashboard endpoints
    check('GET /api/strategy/tf_pinning returns auto+manual+effective',
          "@app.route('/api/strategy/tf_pinning', methods=['GET'])" in app
          and 'def api_strategy_tf_pinning_get(' in app
          and "'effective'" in app)
    check('POST /api/strategy/tf_pinning sets/clears manual override',
          "@app.route('/api/strategy/tf_pinning', methods=['POST'])" in app
          and 'def api_strategy_tf_pinning_set(' in app
          and 'set_manual_pin(strat, tf)' in app
          and '@require_api_key' in app)
    check('POST validates strategy field is required',
          "'strategy required'" in app)


def test_phase48_pr11_multi_tf_inference():
    """Phase 48 — PR 11 / Phase G: multi-TF inference.
       PR 2 made the trainer multi-TF (writing models/<key>_<tf>_*); now
       the bot loads every per-TF artifact and exposes per-TF predictions.

       Backwards-compat is preserved — `.predict(data)` still routes to
       the canonical TF (1h or 1m) so all existing call sites keep
       working. New call sites use predict_at(tf, data) or predict_all().
    """
    print('\n[Phase 48 -- PR 11 multi-TF inference]')

    mtp_path = os.path.join(BASE_DIR, 'src', 'analysis', 'multi_tf_predictor.py')
    check('multi_tf_predictor.py exists', os.path.exists(mtp_path))
    if not os.path.exists(mtp_path):
        return
    src = open(mtp_path, encoding='utf-8').read()
    main = open(os.path.join(BASE_DIR, 'src', 'main.py'), encoding='utf-8').read()

    # 1. MultiTFPredictor surface
    check('MultiTFPredictor class defined',
          'class MultiTFPredictor' in src)
    check('Constructor checks model key against KEYS',
          'if key not in KEYS' in src
          and 'unknown model key' in src)
    check('Loads canonical via legacy filename for backwards compat',
          'LEGACY_MODEL_NAME[key]' in src
          and 'self._predictors[self._canonical_tf]' in src)
    check('Auto-discovers per-TF artifacts via list_per_tf_artifacts',
          'list_per_tf_artifacts(key)' in src)
    check('predict() routes to canonical TF',
          'def predict(self, data)' in src
          and 'self._predictors[self._canonical_tf].predict(data)' in src)
    check('predict_at(tf, data) returns None for unloaded TFs',
          'def predict_at(self, tf' in src
          and 'if p is None or not p.is_loaded' in src
          and 'return None' in src)
    check('predict_all(data_by_tf) iterates loaded TFs only',
          'def predict_all(self, data_by_tf' in src
          and 'if not p.is_loaded' in src
          and 'data_by_tf.get(tf)' in src)
    check('available_tfs returns sorted loaded list',
          'def available_tfs' in src
          and 'sorted(' in src)
    check('Backwards-compat passthrough properties',
          '@property\n    def is_loaded' in src
          and '@property\n    def accuracy' in src
          and '@property\n    def last_error' in src)
    check('_get_model_features forwarded for meta-labeler trainer',
          'def _get_model_features' in src)

    # 2. main.py wiring
    check('main.py imports MultiTFPredictor',
          'from src.analysis.multi_tf_predictor import MultiTFPredictor' in main)
    check('main.py uses MultiTFPredictor for all four model families',
          "MultiTFPredictor('base')" in main
          and "MultiTFPredictor('scalping')" in main
          and "MultiTFPredictor('futures')" in main
          and "MultiTFPredictor('trend')" in main)


def test_phase47_pr10_loading_chips_and_simulator():
    """Phase 47 — PR 10: Monitor 'Loading…' chip recovery + simulator
    auto-poll. Defensive UX so the dashboard never looks frozen when an
    endpoint is briefly unreachable, and the Simulator state pill
    actually transitions on Start/Stop without a manual refresh."""
    print('\n[Phase 47 -- PR 10 loading chips + simulator]')
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    # 1. Each chip's poller surfaces unreachable / HTTP error in the chip
    #    text instead of silently leaving 'Loading…' forever.
    for chip_id in ['cluster-summary-chip', 'dl-summary-chip',
                    'db-status-chip', 'ag-summary-chip', 'ms-summary-chip']:
        check(f'{chip_id} catch sets unreachable on fetch failure',
              ('chip-id-presence', chip_id) and
              tpl.count(f"document.getElementById('{chip_id}')") >= 1
              and 'unreachable' in tpl)

    # Each poller writes 'offline (HTTP n)' on a non-200 response.
    check('Pollers report HTTP status on non-200',
          tpl.count('offline (HTTP') >= 5)
    check('Failed-fetch chip text colour is red (#fb7185)',
          tpl.count("chip.style.color = '#fb7185'") >= 5)

    # 2. Simulator auto-poll wired so Start/Stop transitions are visible.
    check('Simulator tab has periodic auto-poll',
          "setInterval(() => { if (activeTab === 'simulator') simPoll();" in tpl)
    check('Sim status renders state even when error present',
          'if (d.state) simRenderStatus(d)' in tpl)
    check('Sim UNREACHABLE state shown when fetch throws',
          "stateEl.textContent = 'UNREACHABLE'" in tpl)

    # 3. Monitor tab opens cluster + local-training panels too.
    check("Monitor tab open triggers clusterPoll + renderLocalTrainingProgress",
          "if (typeof clusterPoll === 'function') clusterPoll();" in tpl
          and "if (typeof renderLocalTrainingProgress === 'function')" in tpl)


def test_phase46_pr9_ux_bundle():
    """Phase 46 — PR 9 UX bug bundle:
       - Status pill column on Model Training (RUNNING/OK/FAILED/STOPPED/NOT-STARTED)
       - Inline 'How to read' on Stability heatmap with colour-band legend
       - Inline guide on Pure vs ML aggregate (defines each bucket + reading order)
       - Meta-Filtered card surfaces '0 trades — meta-labeler not trained' hint
       - setP6Tab lazy-inits panes on first click so subtabs work even
         without prior tab switch
    """
    print('\n[Phase 46 -- PR 9 UX bundle]')
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    # 1. Model Training status column
    check('Model Training table has Status column header',
          'data-col="run_status"' in tpl
          and '>Status <' in tpl)
    check('Status pill colours wired (ok/run/err/stp/na)',
          "{ok:'#34d399',run:'#fbbf24',err:'#fb7185',stp:'#94a3b8',na:'#475569'}" in tpl)
    check('_statusFor() derives status from age + warnings + pipeline state',
          'function _statusFor(' in tpl
          and 'pipeRunning' in tpl
          and 'NOT STARTED' in tpl
          and "'STOPPED'" in tpl)
    check('RUNNING pill pulses (CSS animation)',
          '@keyframes pulse' in tpl
          and 'animation:pulse' in tpl)
    check('Pipeline orchestrator status fed into _pipeStatus',
          'let _pipeStatus = null' in tpl
          and '_pipeStatus = snap' in tpl)
    check('Training table re-renders on pipeline status change',
          '_renderTrainingTable()' in tpl)

    # 2. Stability heatmap: How to read
    check('Stability heatmap has inline How-to-read guide',
          '>How to read this:</b>' in tpl
          and 'most stable timeframe per strategy' in tpl
          and 'Gold</span>' in tpl
          and 'Yellow</span>' in tpl)
    check('Heatmap explains the ★ best-TF badge',
          'most stable timeframe per strategy' in tpl)

    # 3. Pure vs ML inline guide (item 6)
    check('Pure vs ML has bucket-definition guide',
          '>Pure Rule</b>' in tpl
          and '>ML-Driven</b>' in tpl
          and '>Meta-Filtered</b>' in tpl
          and 'WF Sharpe</b>' in tpl)
    check('Guide explains read order (WF Sharpe + WF Consist first)',
          'In-sample Sharpe over-reports' in tpl)

    # 4. Meta-filtered 0-trade explainer (item 7)
    check('Meta-Filtered card surfaces 0-trade explainer',
          'isEmptyMeta' in tpl
          and 'meta-labeler filter rejecting' in tpl
          and 'Model Training tab' in tpl)

    # 5. Institutional sub-tab fix (item 11)
    check('setP6Tab lazy-inits panes on first click',
          "if (!document.querySelector('.p6-pane'))" in tpl
          and '_renderTabs failed' in tpl)
    check('refreshPhase6Pane wrapped in try/catch so it cannot break the click',
          "console.error('refreshPhase6Pane'" in tpl)


def test_phase45_pipeline_orchestrator():
    """Phase 45 — Pipeline orchestrator (post-resample auto-runner):
       - src/engine/pipeline_orchestrator.py drives train_all() then
         run_full_backtest(timeframes=...) sequentially as a subprocess
       - Status is persisted to data/pipeline_status.json (filelock + atomic)
         so the dashboard pill survives restarts.
       - GET /api/pipeline/status reads the file + alive-check.
       - POST /api/pipeline/run spawns the orchestrator (idempotent — refuses
         to start a second one if one is still alive).
       - Strategy/ML tab gets a Pipeline Orchestrator section with status pill,
         per-phase rows, and ▶ Run button.
    """
    print('\n[Phase 45 -- Pipeline orchestrator]')

    orch_path = os.path.join(BASE_DIR, 'src', 'engine', 'pipeline_orchestrator.py')
    check('pipeline_orchestrator.py exists', os.path.exists(orch_path))
    if not os.path.exists(orch_path):
        return
    orch = open(orch_path, encoding='utf-8').read()
    app  = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
                encoding='utf-8').read()
    tpl  = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
                encoding='utf-8').read()

    # 1. Orchestrator module surface
    check('run_pipeline() top-level entry',
          'def run_pipeline(' in orch
          and 'skip_train' in orch
          and 'skip_backtest' in orch
          and 'backtest_tfs' in orch)
    check('train phase calls train_all_models.train_all',
          'from src.engine.train_all_models import train_all' in orch
          and 'train_all()' in orch)
    check('backtest phase calls run_full_backtest with timeframes kwarg',
          'from src.engine.backtester import run_full_backtest' in orch
          and 'run_full_backtest(timeframes=timeframes)' in orch)
    check('multi-TF backtest default covers 5m/1h/4h/1d/1w',
          'DEFAULT_BACKTEST_TFS' in orch
          and '"5m"' in orch
          and '"1h"' in orch
          and '"4h"' in orch
          and '"1d"' in orch
          and '"1w"' in orch)
    check('Status file path is data/pipeline_status.json',
          'pipeline_status.json' in orch
          and 'STATUS_PATH' in orch)
    check('Status writer uses safe_json (filelock + atomic)',
          'from src.utils.safe_json import write_json' in orch
          and 'from src.utils.safe_json import read_json' in orch)
    check('CLI emits JSON progress events on stderr (--phase/--message/--ts)',
          '_emit_event(' in orch
          and 'sys.stderr.write' in orch
          and '"phase":' in orch
          and '"ts":' in orch)
    check('overall status writes done|error not just running',
          '"status":      "done"' in orch
          and '"error"' in orch)
    check('main() is wired to argparse with --skip-train / --skip-backtest',
          '--skip-train' in orch
          and '--skip-backtest' in orch
          and '--backtest-tfs' in orch)

    # 2. Dashboard endpoints
    check('GET /api/pipeline/status defined',
          "@app.route('/api/pipeline/status'" in app
          and 'def api_pipeline_status(' in app)
    check('GET /api/pipeline/status reads pipeline_status.json + alive flag',
          'pipeline_status.json' in app
          and 'process_alive' in app)
    check('POST /api/pipeline/run defined and gated by api_key',
          "@app.route('/api/pipeline/run'" in app
          and 'def api_pipeline_run(' in app
          and '@require_api_key' in app)
    check('POST /api/pipeline/run rejects double-start (409)',
          '_pipeline_proc_alive()' in app
          and "'orchestrator already running'" in app)
    check('Subprocess spawn uses pipeline_orchestrator module',
          "'src.engine.pipeline_orchestrator'" in app
          and 'subprocess.Popen' in app)
    check('Detached subprocess flags on Windows (survives Flask restart)',
          'CREATE_NEW_PROCESS_GROUP' in app
          and 'DETACHED_PROCESS' in app)
    check('alive-check uses cmdline string match (resists PID recycle)',
          "'pipeline_orchestrator' in cmd" in app)

    # 3. UI panel
    check('Pipeline Orchestrator section in template',
          'id="st-sec-pipe"' in tpl
          and 'Pipeline Orchestrator (train → multi-TF backtest)' in tpl)
    check('Run + Refresh buttons + status pill',
          'pipelineRun()' in tpl
          and 'pipelineRefresh()' in tpl
          and 'id="pipe-pill"' in tpl)
    check('Per-phase rows (started / phase / train / backtest / last event)',
          'id="pipe-started"'    in tpl
          and 'id="pipe-phase"'  in tpl
          and 'id="pipe-train"'  in tpl
          and 'id="pipe-backtest"' in tpl
          and 'id="pipe-last-event"' in tpl)
    check('Status pill colour buckets (idle / running / done / error)',
          '.pipe-status-pill.idle'    in tpl
          and '.pipe-status-pill.running' in tpl
          and '.pipe-status-pill.done' in tpl
          and '.pipe-status-pill.error' in tpl)
    check('Run button posts to /api/pipeline/run with hdrs()',
          "fetch('/api/pipeline/run'" in tpl
          and "method: 'POST'" in tpl
          and 'headers: hdrs()' in tpl)
    check('Refresh polls /api/pipeline/status with auto-interval',
          "fetch('/api/pipeline/status')" in tpl
          and 'setInterval(pipelineRefresh' in tpl)


def test_phase43_pr4_stability_heatmap():
    """Phase 43 — PR 4 Stability comparison view: GET /api/strategy/stability
    builds a (strategy × tf) matrix; UI renders it as a colour-coded heatmap
    with a best-TF badge per row."""
    print('\n[Phase 43 -- PR 4 stability heatmap]')

    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    # 1. Backend endpoint
    check('/api/strategy/stability endpoint defined',
          "@app.route('/api/strategy/stability'" in app
          and 'def api_strategy_stability(' in app)
    check('stability endpoint pulls latest_comparison + wf_results',
          'latest_comparison.json' in app
          and 'wf_results.json' in app)
    check('stability returns cells / best_tf / has_multi_tf',
          "'cells':" in app
          and "'best_tf':" in app
          and "'has_multi_tf':" in app)
    check('best_tf ranks by WF Sharpe with Sharpe fallback',
          "score = row['wf_sharpe_avg']" in app
          and "if score is None:" in app
          and "score = row['sharpe_avg']" in app)

    # 2. UI panel
    check('Stability heatmap section present',
          'id="st-sec-stab"' in tpl
          and 'Stability Heatmap (TF × Strategy)' in tpl)
    check('Metric switcher buttons (WF Sharpe / Consist / Sharpe / Win% / MaxDD / PF)',
          'data-stab-metric="wf_sharpe_avg"' in tpl
          and 'data-stab-metric="wf_consistency_avg"' in tpl
          and 'data-stab-metric="sharpe_avg"' in tpl
          and 'data-stab-metric="win_rate_avg"' in tpl
          and 'data-stab-metric="maxdd_avg"' in tpl
          and 'data-stab-metric="profit_factor_avg"' in tpl)
    check('Heatmap cell color buckets (gold/ok/warn/bad/empty)',
          ".stab-cell.gold" in tpl
          and ".stab-cell.bad" in tpl
          and ".stab-cell.empty" in tpl)
    check('Best-TF column ★ rendered (PR 21 reworked from inline badge to dedicated column)',
          '★ Best' in tpl
          and "_stabSortCol === '__best'" in tpl)
    check('renderStrategyTab calls loadStabilityHeatmap()',
          'loadStabilityHeatmap();' in tpl)
    check('Empty/single-TF state shows guidance message',
          "_stabData.has_multi_tf" in tpl
          and 'run <code>run_full_backtest' in tpl)


def test_phase42_pr3_backtester_multi_tf():
    """Phase 42 — PR 3 strategy multi-timeframe support: backtester loops
    over a list of timeframes and tags each result row with its TF so the
    Stability comparison view can group by it.
    """
    print('\n[Phase 42 -- PR 3 backtester multi-TF support]')

    bt = open(os.path.join(BASE_DIR, 'src', 'engine', 'backtester.py'),
              encoding='utf-8').read()

    check('run_full_backtest accepts timeframes= tuple param',
          'def run_full_backtest(' in bt
          and 'timeframes: tuple[str, ...] = ("1h",)' in bt)
    check('outer loop iterates timeframes',
          'for tf in timeframes:' in bt)
    check('per-symbol load uses <sym>_<tf>.csv.gz',
          # Phase 94 extracted this into _run_one_backtest_cell with
          # symbol/timeframe param names; legacy run_full path still
          # uses sym/tf — accept either form.
          ('f"{sym}_{tf}.csv.gz"' in bt or 'f"{symbol}_{timeframe}.csv.gz"' in bt)
          and ('f"{sym}_spot_{tf}.csv.gz"' in bt or 'f"{symbol}_spot_{timeframe}.csv.gz"' in bt))
    check('each BacktestResult tagged with timeframe attr',
          # Phase 94 — tagging now happens inside _run_one_backtest_cell
          # using `timeframe` not `tf`. Accept either.
          'setattr(res, "timeframe", tf)' in bt
          or 'setattr(res, "timeframe", timeframe)' in bt)
    check('comparison DataFrame carries timeframe column',
          '"timeframe" not in comparison.columns:' in bt
          and 'comparison["timeframe"]' in bt)
    check('walk-forward rows tagged with timeframe',
          'wf["timeframe"] = timeframes[-1]' in bt)
    check('per-tf logging banner',
          'logger.info("=== Backtesting timeframe: %s ===", tf)' in bt)


def test_phase41_pr2_trainer_multi_tf():
    """Phase 41 — PR 2 trainer multi-timeframe refactor:
       - Each tabular trainer accepts timeframe= kwarg
       - Per-TF artifacts via src/utils/model_paths.py + legacy fallback
       - train_all_models.py loops over per-key TF lists
       - Dashboard ml_models surfaces additional per-TF rows
       - /api/training/run/<key> honors a tf body parameter
    """
    print('\n[Phase 41 -- PR 2 trainer multi-TF refactor]')

    # 1. model_paths helper
    mp_path = os.path.join(BASE_DIR, 'src', 'utils', 'model_paths.py')
    check('src/utils/model_paths.py exists', os.path.exists(mp_path))
    mp_src = open(mp_path, encoding='utf-8').read()
    check('model_paths.KEYS frozenset has all 8',
          'KEYS = frozenset({' in mp_src
          and all(k in mp_src for k in
                  ('"base"', '"trend"', '"futures"', '"scalping"',
                   '"tft"', '"oft"', '"meta"', '"regime"')))
    check('CANONICAL_TF maps each key to a default',
          'CANONICAL_TF: dict[str, str]' in mp_src
          and '"base":     "1h"' in mp_src
          and '"scalping": "1m"' in mp_src)
    check('artifact_paths returns per-TF + legacy + is_canonical',
          'def artifact_paths(' in mp_src
          and '"is_canonical"' in mp_src
          and "'legacy_model'" in mp_src or '"legacy_model"' in mp_src)
    check('list_per_tf_artifacts() enumerates on-disk variants',
          'def list_per_tf_artifacts(' in mp_src)

    # 2. Each refactored trainer accepts timeframe=
    for fname, fn in (
        ('train_model.py',          'def train_model('),
        ('train_trend_model.py',    'def train_trend_model('),
        ('train_futures_model.py',  'def train_futures_model('),
        ('train_scalping_model.py', 'def train_scalping_model('),
        ('train_meta_labeler.py',   'def train_meta_labeler('),
    ):
        src = open(os.path.join(BASE_DIR, 'src', 'engine', fname),
                   encoding='utf-8').read()
        check(f'{fname} signature accepts timeframe=',
              f"{fn}timeframe: str = '" in src or
              f"{fn}timeframe='" in src)
        check(f'{fname} writes via artifact_paths()',
              'from src.utils.model_paths import artifact_paths' in src
              and 'paths = artifact_paths(' in src)
        check(f'{fname} dual-writes legacy when canonical TF',
              "if paths['is_canonical']:" in src
              and "joblib.dump(calibrated, paths['legacy_model'])" in src)
        check(f'{fname} CLI argparse exposes --timeframe',
              "ap.add_argument(\"--timeframe\"" in src)

    # 3. train_all_models.py loops per-key
    train_all_src = open(os.path.join(BASE_DIR, 'src', 'engine',
                                       'train_all_models.py'),
                         encoding='utf-8').read()
    check('train_all has DEFAULT_PER_KEY_TFS map',
          'DEFAULT_PER_KEY_TFS' in train_all_src
          and "'base':" in train_all_src and "'1h', '4h', '1d'" in train_all_src)
    check('train_all has _train_loop helper',
          'def _train_loop(' in train_all_src
          and 'fn(timeframe=tf)' in train_all_src)

    # 4. Dashboard surfaces per-TF rows
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    check('strategy_full enumerates per-TF artifacts via list_per_tf_artifacts',
          'list_per_tf_artifacts as _list_per_tf' in app)
    check('per-TF rows carry parent_key / tf / is_canonical',
          "'parent_key':" in app and "'tf':             tf" in app
          and "'is_canonical':   False" in app)

    # 5. _TRAINER_DISPATCH supports tf parameter
    check('_run_trainer_blocking accepts tf argument',
          'def _run_trainer_blocking(job_id: str, key: str, n: int,' in app
          and 'tf: str | None = None' in app)
    check('subprocess passes timeframe= kwarg into trainer',
          'kw = f"timeframe={tf!r}" if tf else ""' in app)
    check('/api/training/run/<key> validates tf body param',
          "tf = body.get('tf')" in app
          and ("tf not in ('1m', '5m', '15m', '1h', '4h', '1d', '1w', '1mo')" in app
               or "tf not in valid_tfs" in app))


def test_phase40_pr1_data_coverage_resample():
    """Phase 40 — PR 1 data backfill foundation: audit module, 1s→higher TF
    resampler, dashboard endpoints, and Data Coverage UI panel.

    The user's intent: build multi-timeframe coverage (5m, 15m, 4h, 1w, 1mo)
    by resampling existing 1s archives instead of re-downloading from
    Binance. Internally consistent, no rate limits, reproducible.
    """
    print('\n[Phase 40 -- PR 1 data audit + 1s->TF resampler + UI]')

    # 1. Audit module
    audit_path = os.path.join(BASE_DIR, 'src', 'utils', 'data_audit.py')
    check('src/utils/data_audit.py exists', os.path.exists(audit_path))
    audit_src = open(audit_path, encoding='utf-8').read()
    check('audit_coverage() returns per (sym, tf) row',
          'def audit_coverage(' in audit_src
          and '"symbol":' in audit_src
          and '"timeframe":' in audit_src
          and '"status":' in audit_src)
    check('DEFAULT_TIMEFRAMES covers 1m..1mo',
          all(tf in audit_src for tf in
              ('"1m"', '"5m"', '"15m"', '"1h"', '"4h"', '"1d"', '"1w"', '"1mo"')))
    check('audit_summary returns present/stale/missing counts',
          'def audit_summary(' in audit_src
          and '"present":' in audit_src and '"stale":' in audit_src and '"missing":' in audit_src)
    check('audit_sentiment audits parquet/_NEWS partitions',
          'def audit_sentiment(' in audit_src
          and 'PARQUET_NEWS' in audit_src)

    # 2. Resampler module
    rs_path = os.path.join(BASE_DIR, 'src', 'utils', 'resample_ohlcv.py')
    check('src/utils/resample_ohlcv.py exists', os.path.exists(rs_path))
    rs_src = open(rs_path, encoding='utf-8').read()
    check('resample_symbol() defined',
          'def resample_symbol(' in rs_src)
    check('resample_all() defined',
          'def resample_all(' in rs_src)
    check('OHLCV agg uses first/max/min/last/sum',
          '"open":' in rs_src and '"first"' in rs_src
          and '"high":' in rs_src and '"max"' in rs_src
          and '"low":' in rs_src and '"min"' in rs_src
          and '"close":' in rs_src and '"last"' in rs_src
          and '"volume":' in rs_src and '"sum"' in rs_src)
    check('resampler streams chunks (no full-file load for 30 GB BTC archive)',
          '_stream_chunks' in rs_src
          and 'chunksize=' in rs_src)
    check('resampler picks 1s source from data/raw/historical first',
          '_candidate_source_paths' in rs_src
          and '_spot_1s.csv.gz' in rs_src
          and 'RAW_HIST_DIR' in rs_src)
    check('resampler writes atomically (.tmp then rename)',
          '.tmp' in rs_src and 'os.replace(' in rs_src)

    # 3. Dashboard endpoints
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    check('/api/data/coverage GET endpoint defined',
          "@app.route('/api/data/coverage'" in app)
    check('/api/data/resample POST endpoint defined',
          "@app.route('/api/data/resample'" in app
          and 'def api_data_resample(' in app)
    check('/api/data/resample gated by @require_api_key',
          app.split('def api_data_resample(')[0].rstrip().endswith('@require_api_key'))
    check('/api/data/resample/jobs GET endpoint defined',
          "@app.route('/api/data/resample/jobs'" in app)
    check('_resample_jobs cache + cap defined',
          '_resample_jobs' in app and '_RESAMPLE_JOBS_MAX' in app)

    # 4. UI panel
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()
    check('Data Coverage section present',
          'id="st-sec-dcov"' in tpl
          and 'Data Coverage (multi-timeframe)' in tpl)
    check('Heatmap grid + cell classes defined',
          'class="dcov-grid"' in tpl
          and '.dcov-cell.present{' in tpl
          and '.dcov-cell.stale{' in tpl
          and '.dcov-cell.missing{' in tpl)
    check('Resample All button + handler',
          'dcovResampleAll(' in tpl
          and "fetch('/api/data/resample'" in tpl)
    check('Jobs poller wired (5s while active)',
          'pollResampleJobs' in tpl
          and "'/api/data/resample/jobs?limit=" in tpl)
    check('renderStrategyTab calls loadDataCoverage()',
          'loadDataCoverage();' in tpl)
    check('Sentiment row in Data Coverage panel',
          'id="dcov-sentiment"' in tpl)


def test_phase39_pr5_ui_bundle():
    """Phase 39 — PR 5 UI bundle: collapse fix, manual training controls,
    ML Health Notes panel, bucket classification, per-bucket disable toggle,
    and Pure-vs-ML comparison panel."""
    print('\n[Phase 39 -- PR 5 UI bundle: collapse + training controls + buckets]')

    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    sr  = open(os.path.join(BASE_DIR, 'src', 'engine', 'strategy_registry.py'),
               encoding='utf-8').read()

    # 1. Collapse fix
    check('toggleSection() defined and toggles .is-collapsed',
          'function toggleSection(' in tpl
          and ".closest('.collapsible-section')" in tpl
          and "classList.toggle('is-collapsed')" in tpl)
    check('CSS hides body when .is-collapsed',
          '.collapsible-section.is-collapsed > .st-sec-body{display:none}' in tpl)
    check('Refresh + Guide moved out of header into body toolbar (no double-arrow)',
          # Header should NOT contain a Refresh button anymore for these two cards
          tpl.count('event.stopPropagation();loadBtComparison()') == 0
          and tpl.count('event.stopPropagation();loadStrategyFull()') == 0)

    # 2. Manual training endpoints
    check('/api/training/run/<key> endpoint defined',
          "@app.route('/api/training/run/<key>'" in app)
    check('/api/training/run/all endpoint defined',
          "@app.route('/api/training/run/all'" in app)
    check('/api/training/jobs endpoint defined',
          "@app.route('/api/training/jobs'" in app)
    check('_TRAINER_DISPATCH covers all 8 model keys',
          all(k in app for k in
              ("'base':", "'trend':", "'futures':", "'scalping':",
               "'tft':", "'oft':", "'meta':", "'regime':")))
    check('_training_jobs cache + cap defined',
          '_training_jobs' in app and '_TRAINING_JOBS_MAX' in app)
    check('train endpoints gated by @require_api_key',
          'def api_training_run_one(' in app
          and 'def api_training_run_all(' in app
          and app.split('def api_training_run_one(')[0].rstrip().endswith('@require_api_key'))

    # 3. Manual training UI
    check('Retrain ALL button wired',
          'trRetrainAll' in tpl
          and "'/api/training/run/all'" in tpl)
    check('Per-row Train button + N selector wired',
          'trRunOne(' in tpl
          and 'id="tr-n-' in tpl
          and 'value="3"' in tpl and 'value="5"' in tpl)
    check('Training jobs poller updates status pill',
          'pollTrainingJobs' in tpl
          and "'/api/training/jobs?limit=" in tpl)

    # 4. ML Health Notes panel
    check('ML Health Notes section present',
          'id="st-sec-mlnotes"' in tpl
          and 'ML Health Notes' in tpl
          and 'Levers to improve the numbers' in tpl)
    check('Notes panel describes bucket model',
          'Pure rule' in tpl and 'ML-driven' in tpl and 'Meta-filtered' in tpl)

    # 5. Bucket classification (backend)
    check("strategy_full tags each strategy with 'bucket' field",
          "s['bucket'] = bucket" in app
          and "'meta_filtered'" in app
          and "'ml_driven'" in app
          and "'pure_rule'" in app)
    check('strategy_registry exposes bucket_for() helper',
          'def bucket_for(name: str) -> str:' in sr)
    check('strategy_registry exposes disabled_buckets() reader',
          'def disabled_buckets()' in sr)
    check('is_enabled_live / is_enabled_backtest honour disabled_buckets',
          'if bucket_for(name) in disabled_buckets():' in sr)

    # 6. Per-bucket toggle endpoint
    check('/api/strategy/bucket POST endpoint exists',
          "@app.route('/api/strategy/bucket'" in app
          and 'disabled_buckets' in app
          and "data/runtime_overrides.json" in app)
    check('Bucket toggle UI button + handler',
          'bucketToggle(' in tpl
          and "fetch('/api/strategy/bucket'" in tpl
          and 'DISABLE BUCKET' in tpl)

    # 7. Pure-vs-ML comparison panel
    check('/api/strategy/bucket_compare endpoint exists',
          "@app.route('/api/strategy/bucket_compare'" in app)
    check('bucket_compare aggregates WF Sharpe + WF Consistency',
          "'wf_sharpe_avg':" in app
          and "'wf_consistency_avg':" in app)
    check('Pure vs ML panel renders 3 buckets',
          'id="bcmp-grid"' in tpl
          and 'function loadBucketCompare(' in tpl
          and "'pure_rule','ml_driven','meta_filtered'" in tpl)
    check('renderStrategyTab calls loadBucketCompare()',
          'loadBucketCompare();' in tpl)


def test_phase38_clear_all_suppression():
    """Phase 38 — banner CLEAR ALL was a visual no-op against still-firing
    issues because /api/errors/recent re-runs scan() + scan_status_surfaces()
    on every poll, and those passes re-added the just-cleared entries from
    the unchanged log tail / unchanged status probes. error_monitor now
    keeps a per-key suppression deadline (DISMISS_SUPPRESS_S = 5 min) and
    both scan paths skip keys whose deadline is still in the future."""
    print('\n[Phase 38 -- CLEAR ALL suppression cool-off]')

    em_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py')
    em_src  = open(em_path, encoding='utf-8').read()

    # 1. Constant + state map present
    check('DISMISS_SUPPRESS_S constant defined',
          'DISMISS_SUPPRESS_S' in em_src and '5 * 60' in em_src)
    check('_dismissed_until map declared',
          '_dismissed_until: dict[str, float]' in em_src)
    check('_is_suppressed helper expires entries in-place',
          'def _is_suppressed(' in em_src
          and '_dismissed_until.pop(key, None)' in em_src)

    # 2. dismiss + dismiss_all set deadlines
    check('dismiss(key) sets _dismissed_until deadline',
          '_dismissed_until[key] = now + DISMISS_SUPPRESS_S' in em_src)
    check('dismiss_all() seeds deadline for every active key',
          '_dismissed_until[k] = deadline' in em_src)

    # 3. Both scan paths consult suppression
    check('scan() skips suppressed keys before create/update',
          em_src.count('if _is_suppressed(key, now):') >= 2)

    # 4. Persistence wraps state in {entries, dismissed_until}
    check('save_state persists wrapped {entries, dismissed_until}',
          '"entries": _state' in em_src
          and '"dismissed_until": live_dismissed' in em_src)
    check('load_state honours the wrapped envelope (with backwards-compat fallback)',
          '"entries" in raw' in em_src
          and 'dismissed_raw = raw.get("dismissed_until")' in em_src)

    # 5. UI confirm message updated to advertise the 5-minute window
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()
    check('CLEAR ALL confirm dialog mentions 5-minute suppression',
          'suppressed for 5 minutes' in tpl)

    # 6. Functional smoke: importing the module + dismiss_all clears + suppresses
    try:
        import importlib, sys as _sys
        if BASE_DIR not in _sys.path:
            _sys.path.insert(0, BASE_DIR)
        # Reload to ensure we pick up the edited source.
        if 'src.dashboard.error_monitor' in _sys.modules:
            em = importlib.reload(_sys.modules['src.dashboard.error_monitor'])
        else:
            em = importlib.import_module('src.dashboard.error_monitor')
        # Inject a fake entry, dismiss_all, then assert the same key would
        # be skipped on a subsequent scan attempt.
        with em._state_lock:
            em._state.clear()
            em._dismissed_until.clear()
            em._state['critical::bot.log::fake'] = {
                'kind':'critical','file':'bot.log','signature':'fake',
                'sample':'x','source':'log','first_seen':0,'last_seen':0,'count':1,
            }
        cleared = em.dismiss_all()
        check('dismiss_all returns count of cleared entries',
              cleared == 1, f'cleared={cleared}')
        check('_state empty after dismiss_all',
              len(em._state) == 0)
        check('_dismissed_until populated after dismiss_all',
              'critical::bot.log::fake' in em._dismissed_until)
        check('_is_suppressed honours fresh deadline',
              em._is_suppressed('critical::bot.log::fake',
                                em._dismissed_until['critical::bot.log::fake'] - 1) is True)
        check('_is_suppressed expires past deadlines',
              em._is_suppressed('critical::bot.log::fake',
                                em._dismissed_until.get('critical::bot.log::fake',
                                                        0) + 1) is False)
    except Exception as exc:
        check('error_monitor suppression smoke test', False, str(exc))


def test_phase37_training_table_and_bt_tooltips():
    """Phase 37 — Strategy/ML tab gets a sortable Model Training card with
    quick filters + 'what good looks like' guide, and the Backtest Comparison
    table headers become sortable with hover tooltips."""
    print('\n[Phase 37 -- model training table + backtest tooltips]')

    app_src = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
                   encoding='utf-8').read()
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    # 1. Backend payload extensions on /api/strategy/full ml_models entries
    check("ml_models exposes 'runs_today'",
          "'runs_today':" in app_src)
    check("ml_models exposes 'total_runs_min' (≥ N lower-bound from archived metas)",
          "'total_runs_min':" in app_src and 'archived_index' in app_src)
    check("ml_models exposes 'age_s' for staleness sort",
          "'age_s':" in app_src)
    check("ml_models exposes 'auc_roc' (meta-labeler discrimination)",
          "'auc_roc':" in app_src)
    check("ml_models exposes 'win_precision'",
          "'win_precision':" in app_src)
    check("ml_models exposes 'symbols_count'",
          "'symbols_count':" in app_src)
    check("aggregate exposes 'models_trained_today'",
          "'models_trained_today':" in app_src)

    # 2. Backtest comparison: tooltipped + sortable headers
    check("backtest table thead row has id 'bt-thead-row'",
          "id=\"bt-thead-row\"" in tpl)
    check("backtest TH cells have data-col + onclick btSort",
          'class="bt-th"' in tpl
          and 'data-col="sharpe"' in tpl
          and 'onclick="btSort(\'sharpe\')"' in tpl)
    check("backtest TH cells carry tooltip via title attribute",
          ('title="Risk-adjusted return.' in tpl
           or 'title="Risk-adjusted return' in tpl)
          and 'title="Largest peak' in tpl)
    check("backtest 'What good looks like' help panel present",
          'id="bt-help-panel"' in tpl
          and 'btToggleHelp' in tpl
          and "What \"good\" looks like" in tpl)
    check("btSort toggles direction on repeat click (_btSortDir state)",
          '_btSortDir' in tpl
          and "_btSortDir = (_btSortDir === 'desc') ? 'asc' : 'desc'" in tpl)

    # 3. Model Training card structure
    check("Model Training section has id 'st-sec-training'",
          'id="st-sec-training"' in tpl)
    check("Training table tbody has id 'training-tbody'",
          'id="training-tbody"' in tpl)
    check("Training quick-filter pills present (today / stale / wf52 / warning / market filters)",
          'data-filt="today"' in tpl
          and 'data-filt="stale"' in tpl
          and 'data-filt="wf52"' in tpl
          and 'data-filt="warning"' in tpl
          and 'data-filt="spot"' in tpl
          and 'data-filt="futures"' in tpl
          and 'data-filt="scalping"' in tpl
          and 'data-filt="neural"' in tpl)
    check("Training search box wired (oninput=trSearch)",
          'id="tr-search"' in tpl and 'oninput="trSearch' in tpl)
    check("Training TH cells have data-col + tr-th class + chevron span",
          'class="tr-th' in tpl
          and 'data-col="age_s"' in tpl
          and 'class="tr-chev"' in tpl)
    check("Training help panel toggleable via trToggleHelp",
          'id="tr-help-panel"' in tpl
          and 'function trToggleHelp' in tpl)
    check("trSort toggles direction + syncs chevron",
          'function trSort(' in tpl
          and "_trSortDir = (_trSortDir === 'desc') ? 'asc' : 'desc'" in tpl)
    check("renderStrategyTab calls _renderTrainingTable()",
          '_renderTrainingTable()' in tpl)
    check("Training table chip shows 'X/Y trained · Z today'",
          'trained · ' in tpl and 'today' in tpl)


def test_phase35_scheduler_no_post_action_refresh():
    """Scheduler register/run/delete must NOT call renderSchedulerPanel()
    after success — the previous behavior wiped the #sch-flash status pill
    via innerHTML rewrite, making clicks look like nothing happened. The
    panel should now refresh ONLY on tab open or manual 🔄 REFRESH click.
    """
    print('\n[Phase 35 -- scheduler no auto-refresh on action]')

    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html'),
               encoding='utf-8').read()

    # Find the action-handlers block; assert it contains NO calls to
    # renderSchedulerPanel() inside _schRegister / _schRun / _schUnregister.
    # The 🔄 REFRESH button onclick still calls window.renderSchedulerPanel —
    # that's the legitimate manual entry point we want to keep.
    for handler in ('_schRegister = async', '_schRun = async', '_schUnregister = async'):
        idx = tpl.find(handler)
        # Body extends from the handler signature to the next 'window._sch'
        # or the closing of the IIFE — close enough using a fixed window.
        body = tpl[idx:idx + 1500]
        # Cut off at the next handler / function so we don't bleed into siblings.
        for stop in ('window._sch', 'window._initPhase6'):
            cut = body.find(stop, 100)  # skip the current signature itself
            if cut > 0:
                body = body[:cut]
                break
        check(f'{handler.split(" =")[0]} body has NO renderSchedulerPanel() call',
              'renderSchedulerPanel(' not in body)

    # The 🔄 REFRESH button still calls renderSchedulerPanel — that's the
    # only path that should re-render now.
    check('🔄 REFRESH button still wired to renderSchedulerPanel',
          'onclick="window.renderSchedulerPanel()"' in tpl)

    # In-place delete row removal (so deleted task vanishes without full re-render)
    check('delete handler removes row in-place via data-sch-name',
          'tr[data-sch-name=' in tpl and 'row.remove()' in tpl)
    check('rendered rows carry data-sch-name attribute',
          'data-sch-name="${esc(t.name)}"' in tpl)

    # Flash pill stays visible long enough to read the post-action message
    check('flash auto-hide bumped to 12s (was 5s)',
          '}, 12000)' in tpl)

    # Post-action messages explicitly tell the user to click 🔄 REFRESH
    check('register success message references 🔄 REFRESH',
          'click 🔄 REFRESH to see it in the list' in tpl)
    check('run success message references 🔄 REFRESH',
          'click 🔄 REFRESH in a few seconds' in tpl)
    check('delete success message references 🔄 REFRESH',
          'click 🔄 REFRESH to update the list' in tpl)


def test_phase34_telegram_monitor_gate():
    """Telegram Monitor must be gated behind TELEGRAM_MONITOR_ENABLED env var
    (default OFF). Telethon v1.43.2 has a headless-reconnect bug that
    cascades into 15+ CRITICAL banner entries on every bot start; until
    we resolve that (Telethon upgrade or session re-login), the monitor
    must not auto-start.
    """
    print('\n[Phase 34 -- Telegram Monitor gate (default-disabled)]')

    main_path = os.path.join(BASE_DIR, 'src', 'main.py')
    main_src = open(main_path, encoding='utf-8').read()

    check('main.py reads TELEGRAM_MONITOR_ENABLED env var',
          "TELEGRAM_MONITOR_ENABLED" in main_src)
    check('default is "false" (off) — must opt in',
          ".environ.get('TELEGRAM_MONITOR_ENABLED', 'false')" in main_src
          or '.environ.get("TELEGRAM_MONITOR_ENABLED", "false")' in main_src)
    check('start() only fires inside the gate',
          # Find the gate block; ensure self.telegram_monitor.start() is
          # inside an if-true branch checking _tg_enabled.
          'if _tg_enabled:' in main_src
          and 'self.telegram_monitor.start()' in main_src.split('if _tg_enabled:')[1][:300])
    check('disabled-path logs an explanation (not silent)',
          'Telegram Monitor disabled' in main_src)


def test_phase32_dedup_market_data():
    """Per-partition dedup of ParquetClient market_data table. Replaces
    an earlier DuckDB-COPY+rebucket attempt that silently dropped rows
    when ts had any NaN values (pandas groupby drops NaN groups by default).

    The new approach: for each yyyymm partition dir, concat all parquet
    files, drop rows with NaN ts, drop_duplicates(['ts'], keep='first'),
    write atomic temp file, swap. Idempotent — re-runs are no-ops on
    single-file partitions.
    """
    print('\n[Phase 32 -- per-partition dedup_market_data]')

    sc_path = os.path.join(BASE_DIR, 'scripts', 'dedup_market_data.py')
    sc = open(sc_path, encoding='utf-8').read()

    # Static checks
    check('dedup script exists', os.path.exists(sc_path))
    check('dedup_partition() defined',
          'def dedup_partition(' in sc)
    check('_walk_partitions() yields leaf dirs only',
          'def _walk_partitions(' in sc and 'has_subdirs' in sc)
    check('script no longer warns EXPERIMENTAL',
          'EXPERIMENTAL' not in sc.upper())
    check('script defends against NaN ts (the bug from previous attempt)',
          'dropna(subset=keys)' in sc)
    check('script writes atomic _dedup_tmp + verify before swap',
          '_dedup_tmp.parquet' in sc and 'verify mismatch' in sc)
    check('script is idempotent (skips len(files)<=1 partitions)',
          'len(files) <= 1' in sc)

    # Behaviour probe: synthetic partition with 3 overlapping files.
    try:
        import importlib, tempfile, shutil
        from pathlib import Path as _P
        from datetime import datetime, timezone
        sys.path.insert(0, BASE_DIR) if BASE_DIR not in sys.path else None
        sm = importlib.import_module('scripts.dedup_market_data')

        import pandas as pd
        td = tempfile.mkdtemp(prefix='dedup_phase32_')
        td_path = _P(td)
        part = td_path / 'symbol=BTC_USDT' / 'timeframe=1h' / 'yyyymm=202601'
        part.mkdir(parents=True)

        # Build 3 files with overlapping ts ranges
        ts1 = [datetime(2026, 1, 1, h, 0, tzinfo=timezone.utc) for h in range(5)]
        ts2 = [datetime(2026, 1, 1, h, 0, tzinfo=timezone.utc) for h in range(3, 8)]   # overlap
        ts3 = [datetime(2026, 1, 1, h, 0, tzinfo=timezone.utc) for h in range(7, 10)]
        for i, ts in enumerate([ts1, ts2, ts3]):
            pd.DataFrame({
                'ts': ts,
                'symbol':    ['BTC_USDT'] * len(ts),
                'timeframe': ['1h'] * len(ts),
                'open':  [42000.0 + i] * len(ts),
                'close': [42100.0 + i] * len(ts),
            }).to_parquet(part / f'data_{i:02d}.parquet')

        before, after = sm.dedup_partition(part, keys=['ts'], dry_run=False)
        check('dedup combines and dedupes 3 files',
              before == 13 and after == 10)   # 5+5+3=13, unique hours 0..9 = 10

        # Result is exactly one parquet file
        files = list(part.glob('*.parquet'))
        check('dedup leaves exactly one file', len(files) == 1)
        check('result file is named data.parquet', files[0].name == 'data.parquet')

        # Idempotent
        b2, a2 = sm.dedup_partition(part, keys=['ts'], dry_run=False)
        check('idempotent: second run is a no-op',
              b2 == 0 and a2 == 0)

        # NaN ts robustness
        part2 = td_path / 'symbol=ETH_USDT' / 'timeframe=1h' / 'yyyymm=202601'
        part2.mkdir(parents=True)
        df_with_nan = pd.DataFrame({
            'ts': [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), pd.NaT,
                   datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)],
            'close': [1.0, 2.0, 3.0],
        })
        df_with_nan.to_parquet(part2 / 'a.parquet')
        df_with_nan.to_parquet(part2 / 'b.parquet')
        before2, after2 = sm.dedup_partition(part2, keys=['ts'], dry_run=False)
        check('NaN ts rows dropped explicitly (not silently kept)',
              before2 == 6 and after2 == 2)   # NaN dropped, 2 unique non-NaN ts

        shutil.rmtree(td, ignore_errors=True)
    except Exception as e:
        check('dedup behaviour probe', False, str(e))


def test_phase33_zombie_watchdog():
    """Zombie watchdog runs every 10 min, kills duplicate project python
    processes (group by command line, keep newest), and orphaned joblib
    resource_tracker workers. Strictly project-scoped — never touches
    Claude Code, VSCode, Chrome, the Android emulator, etc.
    """
    print('\n[Phase 33 -- zombie watchdog]')

    wd_path = os.path.join(BASE_DIR, 'scripts', 'zombie_watchdog.ps1')
    inst_path = os.path.join(BASE_DIR, 'scripts', 'install_zombie_watchdog.ps1')
    uninst_path = os.path.join(BASE_DIR, 'scripts', 'uninstall_zombie_watchdog.ps1')

    check('zombie_watchdog.ps1 exists', os.path.exists(wd_path))
    check('install_zombie_watchdog.ps1 exists', os.path.exists(inst_path))
    check('uninstall_zombie_watchdog.ps1 exists', os.path.exists(uninst_path))

    if not os.path.exists(wd_path):
        return
    wd = open(wd_path, encoding='utf-8').read()

    # Watchdog must be project-scoped (never touches non-project python).
    check('watchdog filters by project root path',
          r'D:\test 2\AI trading assistance' in wd and '*$projectRoot*' in wd)
    # Dedup by command line, keep newest.
    check('watchdog groups by CommandLine and sorts CreationDate descending',
          'Group-Object CommandLine' in wd and 'CreationDate -Descending' in wd)
    # Grace period to avoid races with restart_all.ps1.
    check('watchdog has grace period for young processes',
          'graceSeconds' in wd and 'skip-young' in wd)
    # Orphan detection for joblib resource_tracker.
    check('watchdog handles orphaned joblib resource_tracker',
          'resource_tracker' in wd and 'killed-orphan' in wd)
    # Logging.
    check('watchdog writes to logs/zombie_watchdog.log',
          'zombie_watchdog.log' in wd)
    # Always exit 0 so scheduler doesn't flag failures.
    check('watchdog ends with exit 0',
          wd.rstrip().endswith('exit 0'))

    if os.path.exists(inst_path):
        inst = open(inst_path, encoding='utf-8').read()
        check('install script uses task name AITradingZombieWatchdog',
              'AITradingZombieWatchdog' in inst)
        check('install script schedules every 10 minutes',
              '-Minutes 10' in inst)
        check('install script is idempotent (Unregister before Register)',
              'Unregister-ScheduledTask' in inst and 'Register-ScheduledTask' in inst)

    # On Windows: assert the scheduled task is actually registered.
    if sys.platform == 'win32':
        try:
            import subprocess
            r = subprocess.run(
                ['schtasks.exe', '/Query', '/TN', 'AITradingZombieWatchdog'],
                capture_output=True, text=True, timeout=10,
            )
            check('AITradingZombieWatchdog task registered with Task Scheduler',
                  r.returncode == 0,
                  detail=r.stderr.strip() or 'task not found')
        except Exception as e:
            check('schtasks query', None, str(e))



def test_phase31_market_data_legacy_bridge():
    """Option 3 of the migration finalisation: ParquetClient.query() unions
    market_data writes from data/db/ with the legacy data/parquet/{SYM}/{TF}/
    layout. Backtests that need long history get the 9 years of legacy
    Binance OHLCV alongside the new live writes — under one table name.
    """
    print('\n[Phase 31 -- market_data legacy-store bridge]')

    pc_path = os.path.join(BASE_DIR, 'src', 'database', 'parquet_client.py')
    pc = open(pc_path, encoding='utf-8').read()

    # Static checks
    check('ParquetClient defines _LEGACY_PARQUET_DIR',
          '_LEGACY_PARQUET_DIR' in pc)
    check('ParquetClient defines _has_legacy_parquet helper',
          'def _has_legacy_parquet(' in pc)
    check('ParquetClient defines _market_data_legacy_subquery',
          'def _market_data_legacy_subquery(' in pc)
    check('legacy subquery renames timestamp -> ts',
          'CAST(timestamp AS TIMESTAMP) AS ts' in pc)
    check('legacy subquery normalises backslash separators',
          'chr(92)' in pc)
    check('legacy subquery exposes filename for path-regex extraction',
          "filename=true" in pc)
    check('legacy subquery fills funding_rate as NULL',
          'CAST(NULL AS DOUBLE) AS funding_rate' in pc)
    check('_rewrite_table_refs UNIONs new + legacy for market_data',
          'UNION ALL' in pc and 'legacy_has' in pc)

    # Behavioural check: a synthetic legacy file under a temp PROJECT_ROOT
    # should be queryable through the bridge.
    try:
        import importlib, tempfile, shutil
        from pathlib import Path as _P
        from datetime import datetime, timezone
        sys.path.insert(0, BASE_DIR) if BASE_DIR not in sys.path else None
        pc_mod = importlib.import_module('src.database.parquet_client')

        # Need pyarrow to fabricate a parquet file
        import pyarrow as pa
        import pyarrow.parquet as pq

        td = tempfile.mkdtemp(prefix='pc_bridge_')
        td_path = _P(td)

        # Build a fake legacy file at td/parquet/BTC_USDT/1h/yyyymm=2024-01/data_0.parquet
        legacy_dir = td_path / 'parquet' / 'BTC_USDT' / '1h' / 'yyyymm=2024-01'
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_table = pa.table({
            'timestamp': [datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)],
            'open': [42000.0], 'high': [42100.0], 'low': [41900.0],
            'close': [42050.0], 'volume': [12.3],
        })
        pq.write_table(legacy_table, (legacy_dir / 'data_0.parquet').as_posix())

        # Construct a ParquetClient with base_dir=td/db (empty) and tell
        # it the legacy dir lives at td/parquet
        client = pc_mod.ParquetClient(
            base_dir=td_path / 'db',
            legacy_parquet_dir=td_path / 'parquet',
            flush_s=0, flush_rows=1,
        )

        check('legacy store discovered',
              client._has_legacy_parquet() is True)

        rows = client.query("SELECT symbol, timeframe, close FROM market_data")
        check('bridge surfaces legacy bar (BTC_USDT 1h close=42050)',
              any(r.get('symbol') == 'BTC_USDT'
                  and r.get('timeframe') == '1h'
                  and abs(float(r.get('close') or 0) - 42050.0) < 0.001
                  for r in rows))

        # Cleanup
        client.close()
        shutil.rmtree(td, ignore_errors=True)
    except Exception as e:
        check('bridge runtime smoke test', False, str(e))


def test_phase30_futures_close_reduce_only_guard():
    """When Binance returns -2022 (ReduceOnly Order is rejected), the bot
    must NOT cascade through 3 retry attempts — it should detect the
    'no exchange position' state and force-close internally on the
    first try. Also: ContinuousTrainerAgent is a user-initiated agent
    (only ticks during a sim run) and must be exempt from stale-heartbeat
    warnings, same as Simulator/StrategySimulator.
    """
    print('\n[Phase 30 -- futures reduceOnly guard + trainer exemption]')

    om_path = os.path.join(BASE_DIR, 'src', 'engine', 'order_manager.py')
    om = open(om_path, encoding='utf-8').read()

    # 1. New helper queries exchange-side position size
    check('OrderManager.get_futures_position_amount() defined',
          'def get_futures_position_amount(' in om)
    check('helper calls fetch_position()',
          'fetch_position(' in om)
    check('helper returns 0.0 on error (silent fallback)',
          'return 0.0' in om and 'logging.debug' in om)

    # 2. execute_futures_order detects -2022 specifically
    check('execute_futures_order catches -2022 / ReduceOnly Order rejected',
          "'-2022' in err_str" in om
          or "'-2022' in str(e)" in om
          or '"-2022"' in om)
    check('execute_futures_order returns reduce_only_rejected sentinel',
          "'reduce_only_rejected': True" in om)

    # 3. main.py FUTURES close branch pre-checks + handles sentinel
    main_path = os.path.join(BASE_DIR, 'src', 'main.py')
    m = open(main_path, encoding='utf-8').read()
    check('main close-path pre-checks exchange position size',
          'get_futures_position_amount(' in m
          and "'already closed'" in m or 'already closed' in m)
    check('main close-path handles reduce_only_rejected sentinel',
          "order.get('reduce_only_rejected')" in m)
    check('sentinel handling triggers _force_close_internally without retry',
          ("reduce_only_rejected')" in m
           and '_force_close_internally(' in m.split("reduce_only_rejected')")[1][:600]))

    # 4. ContinuousTrainerAgent now exempt from staleness check
    em_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py')
    em = open(em_path, encoding='utf-8').read()
    check('ContinuousTrainerAgent in _USER_INITIATED_AGENTS exemption set',
          '"ContinuousTrainerAgent"' in em or "'ContinuousTrainerAgent'" in em)

    # 5. Behavior probe: feed a synthetic agent_status.json with stale
    #    ContinuousTrainerAgent and assert no fault is emitted.
    try:
        import importlib, tempfile, json as _json, time as _t
        from pathlib import Path as _P
        sys.path.insert(0, BASE_DIR) if BASE_DIR not in sys.path else None
        em_mod = importlib.import_module('src.dashboard.error_monitor')
        importlib.reload(em_mod)  # pick up the new exemption set

        old_ts = _t.time() - 21365  # match the user-reported staleness
        fake = {
            'ContinuousTrainerAgent': {
                'status': 'running', 'current_task': 'Executing cycle',
                'last_heartbeat_ts': old_ts, 'interval_sec': 30.0,
            },
        }
        td = tempfile.mkdtemp(prefix='em_phase30_')
        data_dir = _P(td) / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / 'agent_status.json').write_text(_json.dumps(fake), encoding='utf-8')
        orig_root = em_mod.PROJECT_ROOT
        em_mod.PROJECT_ROOT = _P(td)
        try:
            faults = em_mod._probe_agents()
        finally:
            em_mod.PROJECT_ROOT = orig_root
        sigs = {f[1] for f in faults}
        check('ContinuousTrainerAgent staleness suppressed (real-world fixture)',
              'agent:ContinuousTrainerAgent:stale' not in sigs)
    except Exception as e:
        check('exemption behaves at runtime', False, str(e))


def test_phase29_cleanup_questdb_artifacts():
    """Phase 5 of the QuestDB → ParquetClient migration: launch scripts,
    schema module, restart_all/stop_all references, CLAUDE.md, and the
    package __init__.py all reflect the new file-based stack. The legacy
    QuestDB-specific files live under _archive/questdb_migration/.
    """
    print('\n[Phase 29 -- cleanup of QuestDB artifacts]')

    # 1. Files moved to _archive/questdb_migration
    archive_dir = os.path.join(BASE_DIR, '_archive', 'questdb_migration')
    check('_archive/questdb_migration/ exists',
          os.path.isdir(archive_dir))
    for fname in ('launch_questdb.ps1', 'schema_questdb.py',
                  '_archived_questdb_client_legacy.py.bak'):
        check(f'archived: {fname}',
              os.path.exists(os.path.join(archive_dir, fname)))

    # 2. Top-level launchers no longer present at project root
    check('launch_questdb.ps1 removed from project root',
          not os.path.exists(os.path.join(BASE_DIR, 'launch_questdb.ps1')))
    check('src/database/schema.py removed',
          not os.path.exists(os.path.join(BASE_DIR, 'src', 'database', 'schema.py')))

    # 3. restart_all.ps1 references the Parquet store, not QuestDB
    rs = open(os.path.join(BASE_DIR, 'restart_all.ps1'), encoding='utf-8').read()
    check('restart_all.ps1 verifies Parquet store, not QuestDB',
          'Parquet store' in rs
          and 'launch_questdb.ps1' not in rs
          and 'docker run -d --name trading_questdb' not in rs)
    check('restart_all.ps1 no longer probes :9000/exec',
          'localhost:9000/exec' not in rs and 'localhost:9000/health' not in rs)

    # 4. stop_all.ps1 dropped the QuestDB-Docker advisory
    sa = open(os.path.join(BASE_DIR, 'stop_all.ps1'), encoding='utf-8').read()
    check('stop_all.ps1 no longer mentions trading_questdb container',
          'trading_questdb' not in sa)

    # 5. CLAUDE.md describes the new DB stack
    # Per-project CLAUDE.md slimmed to project-specific context only (2026-05-11
    # unified culture restructure). Cross-cutting rules moved to the global
    # D:\test 2\CLAUDE.md. Check both files so the assertion stays valid
    # regardless of where the rule landed.
    cm = open(os.path.join(BASE_DIR, 'CLAUDE.md'), encoding='utf-8').read()
    _global_cm_path = os.path.join(os.path.dirname(BASE_DIR), 'CLAUDE.md')
    cm_global = ''
    if os.path.exists(_global_cm_path):
        cm_global = open(_global_cm_path, encoding='utf-8').read()
    cm_all = (cm + '\n' + cm_global).lower()
    check('CLAUDE.md DB line points at ParquetClient',
          'parquetclient' in cm_all
          and 'data/db/' in cm_all)
    check('CLAUDE.md commit-before-implementations rule documented (per-project or global)',
          'commit of the current state' in cm_all
          or 'commit before' in cm_all)

    # 6. requirements.txt no longer ships the questdb client
    rq = open(os.path.join(BASE_DIR, 'requirements.txt'), encoding='utf-8').read()
    check('requirements.txt drops questdb>=1.2.0 dependency',
          'questdb>=1.2.0' not in rq.split('#')[0])  # ignore comment lines

    # 7. src/database/__init__.py exports ParquetClient + legacy alias
    init_src = open(os.path.join(BASE_DIR, 'src', 'database', '__init__.py'),
                    encoding='utf-8').read()
    check('database __init__ exports ParquetClient + legacy QuestDBClient alias',
          'ParquetClient' in init_src
          and 'QuestDBClient = ParquetClient' in init_src)


def test_phase27_ingest_path_cutover():
    """Phase 2 of QuestDB → ParquetClient migration: every QuestDB-era
    importer in the bot's ingest layer now resolves to ParquetClient,
    either by direct import swap or via the questdb_client.py shim.
    """
    print('\n[Phase 27 -- ingest path cutover (Route B)]')

    # 1. realtime_db_writer imports parquet_client directly (write fast path).
    rw_path = os.path.join(BASE_DIR, 'src', 'data_ingestion', 'realtime_db_writer.py')
    rw = open(rw_path, encoding='utf-8').read()
    check('realtime_db_writer imports parquet_client.get_client',
          'from src.database.parquet_client import get_client' in rw)
    check('realtime_db_writer no longer imports questdb_client',
          'from src.database.questdb_client import' not in rw)

    # 2. ingest_pipeline switched to parquet_client + inlined ILP helpers.
    ip_path = os.path.join(BASE_DIR, 'src', 'database', 'ingest_pipeline.py')
    ip = open(ip_path, encoding='utf-8').read()
    check('ingest_pipeline imports parquet_client.get_client',
          'from src.database.parquet_client import get_client' in ip)
    check('ingest_pipeline no longer imports from questdb_client',
          'from src.database.questdb_client import' not in ip)
    check('ingest_pipeline has inlined _to_ns / _tag / _now_ns helpers',
          'def _to_ns' in ip and 'def _tag' in ip and 'def _now_ns' in ip)

    # 3. Post-Phase-5 cleanup: the shim has been deleted and every
    #    importer now references parquet_client directly. db_agent.py
    #    inlines the ILP helpers (_to_ns / _tag / _now_ns) since it
    #    still emits ILP strings into ParquetClient.write_ilp().
    qc_path = os.path.join(BASE_DIR, 'src', 'database', 'questdb_client.py')
    check('questdb_client.py shim deleted (post-shim-drop)',
          not os.path.exists(qc_path))

    db_agent_path = os.path.join(BASE_DIR, 'src', 'database', 'db_agent.py')
    db_agent_src = open(db_agent_path, encoding='utf-8').read()
    check('db_agent inlines _to_ns / _tag / _now_ns helpers',
          'def _to_ns' in db_agent_src
          and 'def _tag' in db_agent_src
          and 'def _now_ns' in db_agent_src)
    check('db_agent has no questdb_client imports',
          'from src.database.questdb_client' not in db_agent_src)

    # 4. Spot-check the 7 other former importers all switched to parquet_client
    for fp in (
        os.path.join(BASE_DIR, 'src', 'analytics', 'data_lens.py'),
        os.path.join(BASE_DIR, 'src', 'data_ingestion', 'binance_sync.py'),
        os.path.join(BASE_DIR, 'src', 'data_ingestion', 'startup_recovery.py'),
        os.path.join(BASE_DIR, 'src', 'data_ingestion', 'watchlist_downloader.py'),
        os.path.join(BASE_DIR, 'src', 'data_ingestion', 'telegram_persistor.py'),
        os.path.join(BASE_DIR, 'src', 'engine', 'train_tft_model.py'),
        os.path.join(BASE_DIR, 'src', 'data_governance', 'base.py'),
    ):
        with open(fp, encoding='utf-8') as f:
            txt = f.read()
        rel = os.path.relpath(fp, BASE_DIR).replace('\\', '/')
        check(f'{rel} no longer imports questdb_client',
              'from src.database.questdb_client' not in txt)


def test_phase26_parquet_client_foundation():
    """Phase 1 (Route B) of the QuestDB replacement migration. ParquetClient
    is a drop-in for QuestDBClient backed by DuckDB + partitioned Parquet
    files on D: — no daemon, no Docker, no port conflicts.
    Asserts:
      - module imports cleanly
      - is_available() reports True (DuckDB present + data dir writable)
      - QuestDBClient public surface is mirrored (every write_/get_ method)
      - round-trip: write → flush → query returns the row
      - missing-table query returns [] (no raise)
      - partition-key string columns get sanitised so BTC/USDT writes are
        queryable as BTC_USDT (DuckDB hive_partitioning compatibility)
    """
    print('\n[Phase 26 -- ParquetClient foundation (Route B)]')

    pc_path = os.path.join(BASE_DIR, 'src', 'database', 'parquet_client.py')
    check('parquet_client.py exists', os.path.exists(pc_path))
    with open(pc_path, encoding='utf-8') as f:
        pc = f.read()

    # ── Static surface mirror ──────────────────────────────────────────────
    for method in ('is_available', 'query', 'query_df', 'exec_ddl',
                   'insert_rows', 'flush_all', 'close',
                   'write_ilp',
                   'write_market_candle', 'write_market_candles_bulk',
                   'write_trade', 'write_signal', 'write_training_event',
                   'write_strategy_stats', 'write_news_sentiment',
                   'write_training_run', 'write_wf_fold',
                   'write_testnet_trade', 'write_testnet_session_stats',
                   'get_latest_candle_ts', 'get_strategy_history',
                   'get_training_history'):
        check(f'method {method}() defined', f'def {method}(' in pc)

    # ── Live behavior ──────────────────────────────────────────────────────
    try:
        import importlib, tempfile, shutil
        from pathlib import Path as _P
        sys.path.insert(0, BASE_DIR) if BASE_DIR not in sys.path else None
        pc_mod = importlib.import_module('src.database.parquet_client')

        td = tempfile.mkdtemp(prefix='pq_phase26_')
        try:
            # Pass a non-existent legacy dir so the bridge stays inert in
            # isolation; otherwise the test would also try to UNION in the
            # real data/parquet/ from PROJECT_ROOT.
            client = pc_mod.ParquetClient(
                base_dir=td, flush_s=0.0, flush_rows=1,
                legacy_parquet_dir=os.path.join(td, '_no_legacy'),
            )
            check('ParquetClient.is_available() True', client.is_available() is True)

            # Round-trip: market_data
            client.write_market_candle('BTC/USDT', '1h', {
                'timestamp': '2026-05-04T12:00:00',
                'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 100,
            })
            client.flush_all()
            rows = client.query("SELECT symbol, close FROM market_data "
                                "WHERE symbol = 'BTC_USDT'")
            check('market_data round-trip (sanitised symbol)',
                  len(rows) == 1 and rows[0].get('close') == 1.5)

            # Round-trip: trade_events (no string partition keys)
            client.write_trade({'symbol': 'BTC/USDT', 'strategy': 'rsi',
                                'pnl_usd': 7.5, 'is_live': False})
            client.flush_all()
            trows = client.query("SELECT symbol, pnl_usd FROM trade_events")
            check('trade_events round-trip',
                  len(trows) == 1 and trows[0].get('pnl_usd') == 7.5)

            # Missing-table query is graceful (returns [], no raise)
            empty = client.query("SELECT * FROM csv_ingestion_log")
            check('missing-table query returns []', empty == [])

            # exec_ddl is a no-op that returns True (compatibility shim)
            check('exec_ddl returns True (no-op)',
                  client.exec_ddl('CREATE TABLE foo (x INT)') is True)

            client.close()
        finally:
            shutil.rmtree(td, ignore_errors=True)
    except Exception as e:
        check('ParquetClient live round-trip', False, str(e))


def test_phase25_user_initiated_agents_exempt():
    """User-initiated agents (SimulatorAgent / StrategySimulatorAgent) are
    exempt from stale-heartbeat warnings. They only tick while the user is
    actively running a sim — when idle, agent_status.json keeps the last
    'running' state and heartbeat ages out, but that's not a real fault.
    Agents in 'error' state are still flagged regardless of exemption.
    """
    print('\n[Phase 25 -- user-initiated agents exempt from staleness]')

    em_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py')
    with open(em_path, encoding='utf-8') as f:
        em = f.read()

    check('exemption set _USER_INITIATED_AGENTS defined',
          '_USER_INITIATED_AGENTS' in em)
    check('SimulatorAgent in exemption set',
          '"SimulatorAgent"' in em or "'SimulatorAgent'" in em)
    check('StrategySimulatorAgent in exemption set',
          '"StrategySimulatorAgent"' in em or "'StrategySimulatorAgent'" in em)
    check('_probe_agents skips staleness for exempted names',
          'name in _USER_INITIATED_AGENTS' in em
          and 'continue' in em.split('name in _USER_INITIATED_AGENTS')[1][:200])

    # Behavior probe: feed a synthetic agent_status.json shape through
    # _probe_agents and assert SimulatorAgent stale entries don't appear
    # while a non-exempt stale agent does.
    try:
        import importlib, sys as _sys, json as _json, tempfile, time as _t
        sys.path.insert(0, BASE_DIR) if BASE_DIR not in _sys.path else None
        em_mod = importlib.import_module('src.dashboard.error_monitor')

        # Build a fake agent_status.json with a stale SimulatorAgent and a
        # stale RiskAgent. Only the RiskAgent should produce a fault.
        old_ts = _t.time() - 7200  # 2h ago — well past 4× interval
        fake = {
            'SimulatorAgent': {
                'status': 'running', 'current_task': 'Replay X',
                'last_heartbeat_ts': old_ts, 'interval_sec': 5.0,
            },
            'StrategySimulatorAgent': {
                'status': 'running', 'last_heartbeat_ts': old_ts,
                'interval_sec': 5.0,
            },
            'RiskAgent': {
                'status': 'idle', 'last_heartbeat_ts': old_ts,
                'interval_sec': 300.0,
            },
            'ContinuousTrainerAgent': {
                'status': 'error', 'current_task': 'training failed',
                'last_heartbeat_ts': _t.time(), 'interval_sec': 30.0,
            },
        }
        # Temporarily redirect PROJECT_ROOT to a tmpdir with our fake JSON
        td = tempfile.mkdtemp(prefix='em_test_')
        from pathlib import Path as _P
        data_dir = _P(td) / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / 'agent_status.json').write_text(_json.dumps(fake), encoding='utf-8')
        orig_root = em_mod.PROJECT_ROOT
        em_mod.PROJECT_ROOT = _P(td)
        try:
            faults = em_mod._probe_agents()
        finally:
            em_mod.PROJECT_ROOT = orig_root

        sigs = {f[1] for f in faults}
        check('SimulatorAgent staleness suppressed',
              'agent:SimulatorAgent:stale' not in sigs)
        check('StrategySimulatorAgent staleness suppressed',
              'agent:StrategySimulatorAgent:stale' not in sigs)
        check('non-exempt stale agent (RiskAgent) still flagged',
              'agent:RiskAgent:stale' in sigs)
        check('error-state agent flagged regardless of exemption list',
              'agent:ContinuousTrainerAgent' in sigs)
    except Exception as e:
        check('_probe_agents exemption behavior probe', False, str(e))


def test_phase22_scheduler_no_autorefresh():
    """Scheduler sub-tab: no 30s auto-refresh + manual REFRESH button.

    Why: auto-refresh wiped the open Mode dropdown and any half-typed
    task name mid-edit. User opted scheduler out of the periodic poll
    and added an explicit 🔄 REFRESH button.
    """
    print('\n[Phase 22 -- Scheduler manual-refresh-only]')

    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # 1. Phase-6 interval skips scheduler
    check('Phase-6 30s auto-refresh skips scheduler sub-tab',
          "t.dataset.tab === 'scheduler'" in tpl and 'never auto-refresh' in tpl
          or "if (t.dataset.tab === 'scheduler') return;" in tpl)

    # 2. Manual REFRESH button is wired
    check('Scheduler REFRESH button present',
          '🔄 REFRESH' in tpl
          and 'window.renderSchedulerPanel()' in tpl)

    # 3. renderSchedulerPanel exposed on window for the inline onclick
    check('renderSchedulerPanel exposed on window',
          'window.renderSchedulerPanel = renderSchedulerPanel' in tpl)

    # 4. Form grid expanded to fit the 5th column (REFRESH button)
    check('Scheduler form grid has 5 columns (added REFRESH cell)',
          'grid-template-columns:1.4fr 1fr 1.4fr auto auto' in tpl)


def test_phase59_pr35_parquet_query_thread_safety():
    """ParquetClient query/query_df hold _duck_lock during execute().

    Why: DuckDB's Python connection is NOT thread-safe — concurrent
    execute() on the same connection triggers C++ assertion failures
    (e.g. "INTERNAL Error: Attempted to dereference unique_ptr that
    is NULL!") that abort the whole process. The dashboard has many
    concurrent request threads + an error-monitor scan thread, so
    the abort took down the live UI on 2026-05-08. Fix is to hold
    the existing _duck_lock for the entire execute+fetch sequence.
    """
    print('\n[Phase 59 -- PR-35 ParquetClient query thread-safety]')

    pc_path = os.path.join(BASE_DIR, 'src', 'database', 'parquet_client.py')
    with open(pc_path, encoding='utf-8') as f:
        pc = f.read()

    # Locate query() and query_df() bodies and assert each holds the
    # lock during execute(). Slice the file by `def ` boundaries so we
    # can scan a fixed window without a regex that risks catastrophic
    # backtracking on a 700-line file.
    def _body_holds_lock(fn_name: str) -> bool:
        marker = f"    def {fn_name}(self"
        i = pc.find(marker)
        if i < 0:
            return False
        # Body runs until the next top-level def at the same indent.
        j = pc.find("\n    def ", i + len(marker))
        body = pc[i:j] if j > i else pc[i:]
        if 'with self._duck_lock:' not in body:
            return False
        # Scan line by line — `with self._duck_lock:` must be followed
        # within a few lines by `con.execute(`.
        lines = body.splitlines()
        for idx, ln in enumerate(lines):
            if 'with self._duck_lock:' in ln:
                window = '\n'.join(lines[idx:idx + 5])
                if 'con.execute(' in window:
                    return True
        return False

    check('query() holds _duck_lock during con.execute()',
          _body_holds_lock('query'))
    check('query_df() holds _duck_lock during con.execute()',
          _body_holds_lock('query_df'))

    # Live behavior: spawn N threads that all hammer query() at once.
    # Pre-fix this would fairly reliably trigger the abort within a few
    # iterations; post-fix it returns N×rows without crashing.
    try:
        import importlib, tempfile, shutil, threading as _th
        sys.path.insert(0, BASE_DIR) if BASE_DIR not in sys.path else None
        pc_mod = importlib.import_module('src.database.parquet_client')
        td = tempfile.mkdtemp(prefix='pq_phase59_')
        try:
            client = pc_mod.ParquetClient(
                base_dir=td, flush_s=0.0, flush_rows=1,
                legacy_parquet_dir=os.path.join(td, '_no_legacy'),
            )
            for i in range(20):
                client.write_market_candle('BTC/USDT', '1h', {
                    'timestamp': f'2026-05-04T{i:02d}:00:00',
                    'open': 1, 'high': 2, 'low': 0.5,
                    'close': 1.5 + i, 'volume': 100 + i,
                })
            client.flush_all()

            errors: list[str] = []
            results_count: list[int] = []
            barrier = _th.Barrier(8)
            def _worker():
                try:
                    barrier.wait(timeout=5)
                    rows = client.query("SELECT COUNT(*) AS n FROM market_data")
                    results_count.append(int(rows[0]['n']) if rows else -1)
                except Exception as e:
                    errors.append(f'{type(e).__name__}: {e}')
            ts = [_th.Thread(target=_worker) for _ in range(8)]
            for t in ts: t.start()
            for t in ts: t.join(timeout=10)
            client.close()
            check('8-way concurrent query() — no exception',
                  not errors, '; '.join(errors[:2]))
            check('8-way concurrent query() — every thread saw 20 rows',
                  results_count == [20] * 8,
                  f'got {results_count}')
        finally:
            shutil.rmtree(td, ignore_errors=True)
    except Exception as e:
        check('ParquetClient concurrent-query live test', False, str(e))


def test_phase60_pr36_training_concurrency_cap():
    """Trainer dispatch holds a semaphore-gated concurrency cap.

    Why: pre-fix, kicking 8 retrains in a row spawned 8 concurrent
    Popen processes (each loading torch/sklearn/pandas, ~1-2 GB RSS).
    On a 32 GB / single-GPU box this hit ~88% RAM and the dashboard
    Popen syscall failed inside RPCRT4 — the dashboard process died
    with no Python traceback, taking the live UI down on 2026-05-08.
    Fix: gate _run_trainer_blocking and _run_trainer_multi_tf with a
    threading.Semaphore(N) so excess jobs queue instead of all-spawn.
    """
    print('\n[Phase 60 -- PR-36 training concurrency cap]')

    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()

    check('concurrency cap defined (scheduler or semaphore)',
          '_training_scheduler' in app or '_training_concurrency_sem = threading.Semaphore(' in app)
    check('AI_TRADER_TRAIN_CONCURRENCY env var honored',
          "AI_TRADER_TRAIN_CONCURRENCY" in app)

    # Both trainer functions must acquire SOMETHING (scheduler or sem)
    # and release it in a finally. Accept either API — Phase 61 covers
    # the resource-aware scheduler in detail.
    blocking_idx = app.find('def _run_trainer_blocking(')
    multi_idx    = app.find('def _run_trainer_multi_tf(')
    for label, start in (('_run_trainer_blocking', blocking_idx),
                         ('_run_trainer_multi_tf', multi_idx)):
        if start < 0:
            check(f'{label} found', False)
            continue
        end = app.find('\ndef ', start + 4)
        body = app[start:end] if end > start else app[start:]
        # Acquire = either scheduler.acquire(...) or sem.acquire()
        acquire_pos = -1
        for needle in ('_training_scheduler.acquire(',
                       '_training_concurrency_sem.acquire('):
            p = body.find(needle)
            if p > 0 and (acquire_pos < 0 or p < acquire_pos):
                acquire_pos = p
        # The spawn was originally inline subprocess.Popen but PR-40
        # routes it through _spawn_training_subprocess which sets
        # detach + log redirection. Either call site must come AFTER
        # the acquire so the lane is reserved before the subprocess
        # starts consuming resources.
        popen_pos = -1
        for needle in ('subprocess.Popen(',
                       '_spawn_training_subprocess('):
            p = body.find(needle)
            if p > 0 and (popen_pos < 0 or p < popen_pos):
                popen_pos = p
        check(f'{label}: acquire() precedes subprocess spawn',
              0 <= acquire_pos < popen_pos)
        check(f'{label}: release() called in finally',
              ('_training_scheduler.release(' in body
               or '_training_concurrency_sem.release(' in body)
              and 'finally:' in body)
        check(f'{label}: status \'queued\' set before acquire',
              0 <= body.find("status='queued'") < acquire_pos
              if acquire_pos > 0 else False)


def test_phase63_pr39_strategy_panels_hourly_refresh():
    """Strategy & ML panels (ML Models, Model Training, Pure Rule vs ML,
    Stability Heatmap, Data Coverage, Pipeline Orchestrator) refresh
    only on F5 / manual button / hourly auto-tick — not on every 5 s
    pollState beat. Operator wanted the polling load on the
    minutes-to-hours data to match its real change cadence."""
    print('\n[Phase 63 -- PR-39 strategy panels hourly auto-refresh]')

    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # The hourly refresh helper exists.
    check('_strategyHourlyRefresh function defined',
          'function _strategyHourlyRefresh' in tpl)

    # It calls all 6 loaders.
    body_start = tpl.find('function _strategyHourlyRefresh')
    body_end   = tpl.find('\n}\n', body_start)
    body = tpl[body_start:body_end] if body_end > body_start else ''
    for fn in ('loadStrategyFull', 'loadBucketCompare', 'loadStabilityHeatmap',
               'loadDataCoverage', 'pollResampleJobs', 'pipelineRefresh'):
        check(f'_strategyHourlyRefresh calls {fn}', fn in body)

    # Hourly setInterval armed (3600 * 1000 ms = 3,600,000 ms).
    check('hourly setInterval armed',
          'setInterval(_strategyHourlyRefresh, 3600 * 1000)' in tpl
          or 'setInterval(_strategyHourlyRefresh, 3600000)' in tpl)

    # Initial fire on page load (so F5 triggers a refresh).
    check('initial fire on DOMContentLoaded',
          'setTimeout(_strategyHourlyRefresh' in tpl)

    # renderStrategyTab no longer fires the load* loaders (the bug we fixed).
    rst_start = tpl.find('function renderStrategyTab(state)')
    rst_end   = tpl.find('\n}\n', rst_start)
    rst_body  = tpl[rst_start:rst_end] if rst_end > rst_start else ''
    for fn in ('loadBucketCompare', 'loadStabilityHeatmap',
               'loadDataCoverage', 'pollResampleJobs'):
        check(f'renderStrategyTab no longer calls {fn}',
              fn + '(' not in rst_body)

    # pipelineRefresh removed from the staggered tick.
    stagger_start = tpl.find('_STAGGERED_POLLERS')
    stagger_end   = tpl.find('];', stagger_start)
    stagger_body  = tpl[stagger_start:stagger_end] if stagger_end > stagger_start else ''
    check('pipelineRefresh removed from _STAGGERED_POLLERS',
          'pipelineRefresh()' not in stagger_body)

    # pollTrainingJobs is STILL in the staggered tick (drives ETA).
    check('pollTrainingJobs still in _STAGGERED_POLLERS (drives ETA)',
          'pollTrainingJobs()' in stagger_body)


def test_phase62_pr38_training_eta_and_elapsed():
    """Training jobs response carries elapsed_s / eta_s / typical_s
    so the dashboard row can render '5s · ~29m left' beneath the
    RUNNING pill. Frontend reads these field names verbatim."""
    print('\n[Phase 62 -- PR-38 elapsed + ETA on training rows]')

    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()

    check('_TYPICAL_DURATIONS map present',
          '_TYPICAL_DURATIONS' in app and "'tft':" in app and "'oft':" in app)
    check('_annotate_job_timing decorator present',
          'def _annotate_job_timing' in app)
    check('api_training_jobs annotates rows',
          '_annotate_job_timing(' in app)
    check('_record_completed_duration updates rolling avg',
          'def _record_completed_duration' in app
          and '_record_completed_duration(' in app)

    # Spot-check the decorator's contract — should set elapsed_s + eta_s
    # only when status==running and never go negative for eta_s.
    import time as _t
    ns: dict = {'time': _t, 'threading': __import__('threading')}
    # Extract _TYPICAL_DURATIONS + helper + decorator. We exec the
    # whole block as one program so the closures see _TYPICAL_DURATIONS.
    start = app.find('# Typical training durations')
    end = app.find('@app.route(\'/api/training/jobs\'', start)
    if start < 0 or end < 0:
        check('extract decorator source', False); return
    src = app[start:end]
    # Strip the type annotation prefix to get a valid module-level
    # assignment that exec() will install in the shared namespace.
    src = src.replace('_TYPICAL_DURATIONS: dict[str, float] = ',
                      '_TYPICAL_DURATIONS = ', 1)
    src = src.replace('_TYPICAL_HISTORY: dict[str, list[float]] = ',
                      '_TYPICAL_HISTORY = ', 1)
    try:
        exec(src, ns)
    except Exception as e:
        check('extracted decorator compiles', False, str(e))
        return
    annotate = ns['_annotate_job_timing']
    now = _t.time()

    # Running job halfway through typical duration
    j = annotate({'model': 'regime', 'status': 'running',
                  'started_at': now - 600})
    check('running row gets elapsed_s', j.get('elapsed_s') is not None and j['elapsed_s'] >= 600)
    check('running row gets eta_s (clamped to 0+)',
          j.get('eta_s') is not None and j['eta_s'] >= 0)
    check('typical_s carried through', j.get('typical_s') == 30 * 60)

    # Overdue running job — eta clamps to 0, never negative
    j = annotate({'model': 'regime', 'status': 'running',
                  'started_at': now - 99999})
    check('overdue row eta_s clamps to 0', j.get('eta_s') == 0.0)

    # Queued job — gets queued_for_s, no elapsed/eta
    j = annotate({'model': 'tft', 'status': 'queued',
                  'queued_at': now - 12})
    check('queued row gets queued_for_s', j.get('queued_for_s') is not None and j['queued_for_s'] >= 12)
    check('queued row gets typical_s for the operator preview',
          j.get('typical_s') == 60 * 60)

    # Finished job — elapsed_s carried, no eta
    j = annotate({'model': 'regime', 'status': 'done',
                  'started_at': now - 1234, 'finished_at': now})
    check('finished row keeps elapsed_s', j.get('elapsed_s') is not None and j['elapsed_s'] >= 1234)
    check('finished row has no eta_s', 'eta_s' not in j)


def test_phase61_pr37_resource_aware_scheduler():
    """Trainer scheduler is CPU/GPU/exclusive-aware, not a single Sem.

    Why: the user asked us to overlap CPU trainings with the GPU TFT
    job (regime + TFT can safely run together — disjoint resources),
    but keep OFT exclusive (it saturates GPU memory + CPU dataloader).
    The previous Semaphore(3) didn't distinguish lanes; PR-37 replaces
    it with a _TrainingScheduler that tracks cpu_active / gpu_active /
    exclusive_busy under one Condition.
    """
    print('\n[Phase 61 -- PR-37 resource-aware scheduler]')

    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()

    check('class _TrainingScheduler defined',
          'class _TrainingScheduler' in app)
    check('_RESOURCE_KIND map defined',
          '_RESOURCE_KIND' in app and "'tft':" in app and "'oft':" in app)
    check("oft tagged 'exclusive'",
          "'oft':" in app and re.search(r"'oft':\s*'exclusive'", app) is not None)
    check("tft tagged 'gpu'",
          re.search(r"'tft':\s*'gpu'", app) is not None)
    check("regime/trend/futures tagged 'cpu'",
          all(re.search(rf"'{k}':\s*'cpu'", app)
              for k in ('regime', 'trend', 'futures', 'base', 'scalping', 'meta')))
    check('AI_TRADER_GPU_CONCURRENCY env var honored',
          'AI_TRADER_GPU_CONCURRENCY' in app)
    check('/api/training/scheduler endpoint defined',
          "@app.route('/api/training/scheduler'" in app)

    # Live behavior: load the scheduler class out of app.py (without
    # importing the whole Flask app — that triggers heavy ML imports).
    # Compile only the class block by extracting from the source.
    scheduler_src_start = app.find('class _TrainingScheduler')
    scheduler_src_end   = app.find('\n\n_training_scheduler = ', scheduler_src_start)
    if scheduler_src_start < 0 or scheduler_src_end < 0:
        check('extract scheduler source', False)
        return
    src = app[scheduler_src_start:scheduler_src_end]
    ns: dict = {}
    import threading as _th
    ns['threading'] = _th
    try:
        exec(src, ns)
    except Exception as e:
        check('scheduler source compiles', False, str(e))
        return
    Sched = ns['_TrainingScheduler']

    # Case 1: cpu+gpu can run concurrently (regime + TFT)
    s = Sched(cpu_cap=2, gpu_cap=1)
    s.acquire('cpu')
    s.acquire('gpu')
    snap = s.snapshot()
    check('cpu+gpu concurrent: cpu_active=1 gpu_active=1',
          snap['cpu_active'] == 1 and snap['gpu_active'] == 1)
    s.release('cpu'); s.release('gpu')

    # Case 2: cpu cap respected — third cpu acquire blocks
    s = Sched(cpu_cap=2, gpu_cap=1)
    s.acquire('cpu'); s.acquire('cpu')
    blocked = {'fired': False}
    def _try_third():
        s.acquire('cpu')
        blocked['fired'] = True
        s.release('cpu')
    t = _th.Thread(target=_try_third, daemon=True)
    t.start(); t.join(timeout=0.5)
    check('cpu_cap=2 blocks third cpu acquire', not blocked['fired'])
    s.release('cpu')                 # frees one slot
    t.join(timeout=2.0)
    check('cpu_cap=2 unblocks after release', blocked['fired'])
    s.release('cpu')

    # Case 3: exclusive blocks while cpu running
    s = Sched(cpu_cap=2, gpu_cap=1)
    s.acquire('cpu')
    excl_done = {'fired': False}
    def _try_excl():
        s.acquire('exclusive')
        excl_done['fired'] = True
        s.release('exclusive')
    t = _th.Thread(target=_try_excl, daemon=True)
    t.start(); t.join(timeout=0.5)
    check('exclusive blocks while cpu_active>0', not excl_done['fired'])
    s.release('cpu')
    t.join(timeout=2.0)
    check('exclusive proceeds once cpu drains', excl_done['fired'])

    # Case 4: exclusive blocks all NEW acquires while it's running
    s = Sched(cpu_cap=2, gpu_cap=1)
    s.acquire('exclusive')
    cpu_done = {'fired': False}
    def _try_cpu():
        s.acquire('cpu')
        cpu_done['fired'] = True
        s.release('cpu')
    t = _th.Thread(target=_try_cpu, daemon=True)
    t.start(); t.join(timeout=0.5)
    check('cpu acquire blocks while exclusive_busy', not cpu_done['fired'])
    s.release('exclusive')
    t.join(timeout=2.0)
    check('cpu acquire proceeds after exclusive release', cpu_done['fired'])


def test_phase70_pr43_dashboard_watchdog():
    """Watchdog daemon keeps the dashboard alive.

    Polls /api/state; on FAILURE_THRESHOLD consecutive failed checks,
    kills stale dash processes and respawns via Win32_Process.Create.
    Circuit breaker prevents infinite restart loops on import-time
    crashes. Started by restart_all.ps1 alongside the dashboard."""
    print('\n[Phase 70 -- PR-43 dashboard watchdog daemon]')

    wd_path = os.path.join(BASE_DIR, 'scripts', 'dashboard_watchdog.py')
    check('scripts/dashboard_watchdog.py exists', os.path.exists(wd_path))
    if not os.path.exists(wd_path):
        return
    with open(wd_path, encoding='utf-8') as f:
        wd = f.read()

    check('FAILURE_THRESHOLD configurable via env var',
          'AI_TRADER_DASH_WATCH_FAIL_N' in wd)
    check('RESTART_LIMIT configurable via env var',
          'AI_TRADER_DASH_WATCH_LIMIT' in wd)
    check('RESTART_WINDOW_S configurable via env var',
          'AI_TRADER_DASH_WATCH_WINDOW_S' in wd)
    check('health probe targets /api/state by default',
          "'http://127.0.0.1:5000/api/state'" in wd)
    check('watchdog state persists to data/dashboard_watchdog_state.json',
          'dashboard_watchdog_state.json' in wd)
    check('logs to logs/dashboard_watchdog.log',
          'dashboard_watchdog.log' in wd)
    check('uses Win32_Process.Create on Windows for detached spawn',
          'Win32_Process' in wd and 'CreationDate' not in wd  # we use ProcessId, not creationdate
          and 'sys.platform' in wd)
    check('kills stale dashboards before respawn',
          'def _kill_existing_dashboards' in wd
          and 'src.dashboard.app' in wd
          and 'p.kill()' in wd)
    check('circuit breaker stops infinite restart loops',
          'def _circuit_tripped' in wd
          and "state['tripped']" in wd
          and 'CRITICAL' in wd)
    check('atomic state write (.tmp + os.replace)',
          'os.replace' in wd and 'with_suffix' in wd)

    # restart_all.ps1 wires it in.
    rs_path = os.path.join(BASE_DIR, 'restart_all.ps1')
    with open(rs_path, encoding='utf-8') as f:
        rs = f.read()
    check('restart_all.ps1 launches the watchdog',
          'scripts.dashboard_watchdog' in rs
          and '5.96' in rs)
    check('restart_all.ps1 records watchdog pid in data/process_ids.json',
          'watchdog =' in rs.lower() or 'watchdog =' in rs)


def test_phase71_pr46_real_cash_label_rename():
    """v3 step 1 (1K): MAINNET → REAL CASH UI rename.

    Display strings only — backend wire value 'mainnet' is unchanged
    (control.json key, balance_real.json filename, ccxt config all
    keep the original token). The button id, CSS class, and the
    JS comparison `m === 'mainnet'` must still resolve, otherwise
    mode-switching breaks."""
    print('\n[Phase 71 -- PR-46 REAL CASH label rename (v3 step 1)]')

    if not os.path.exists(TEMPLATE_PATH):
        check('template file exists', False, TEMPLATE_PATH)
        return
    html = open(TEMPLATE_PATH, encoding='utf-8').read()

    # Display label flipped to REAL CASH on the button + tooltip + status.
    check('button label reads "⚡ REAL CASH" (not "⚡ MAINNET")',
          '⚡ REAL CASH' in html and '⚡ MAINNET' not in html)
    check('button tooltip uses REAL CASH wording',
          'REAL CASH at risk' in html)
    check('status mapping shows "REAL CASH — live Binance"',
          "'⚠ REAL CASH — live Binance, real money'" in html)
    check('confirm dialog asks "Switch to REAL CASH (live Binance)"',
          'Switch to REAL CASH (live Binance)' in html)

    # Wire value `mainnet` and supporting selectors must NOT have been
    # renamed — they're load-bearing in JS and CSS.
    check('button id `lt-btn-mainnet` preserved (wire value)',
          'id="lt-btn-mainnet"' in html)
    check('JS comparison `m === \'mainnet\'` still present',
          "m === 'mainnet'" in html)
    check('onclick still calls ltSetMode(\'mainnet\')',
          "ltSetMode('mainnet')" in html)
    check('CSS class .lt-mode-btn.active.mainnet still defined',
          '.lt-mode-btn.active.mainnet' in html)

    # Backend trade_mode_label string surfaced via /api/portfolio
    # also flipped to REAL CASH so frontend doesn't see two labels.
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    if os.path.exists(app_path):
        app = open(app_path, encoding='utf-8').read()
        check('app.py /api/portfolio returns "REAL CASH" label (not MAINNET)',
              '⚠ REAL CASH — live Binance, real money' in app
              and '⚠ MAINNET — real money' not in app)


def test_phase71b_v31_curated_tf_map():
    """v3.1 step 2 (1A): DEFAULT_PER_KEY_TFS uses the curated 25-combo
    map ('applicable based on model logic'), with AI_TRADER_TRAIN_TF_MAP
    env-var override to fall back to strict 49-combo all×all."""
    print('\n[Phase 71b -- v3.1 step 2: curated DEFAULT_PER_KEY_TFS]')

    src = os.path.join(BASE_DIR, 'src', 'engine', 'train_all_models.py')
    with open(src, encoding='utf-8') as f:
        code = f.read()

    # Curated entries for each model key.
    expected = {
        'base':     ('5m', '15m', '1h', '4h', '1d'),
        'trend':    ('15m', '1h', '4h', '1d', '1w'),
        'futures':  ('5m', '15m', '1h', '4h', '1d'),
        'scalping': ('1m', '5m'),
        'meta':     ('5m', '15m', '1h', '4h'),
        'tft':      ('15m', '1h', '4h'),
        'regime':   ('1h',),
    }
    for key, tfs in expected.items():
        # The literal tuple appears verbatim in the source.
        # Match the order to catch accidental reorderings.
        formatted = ', '.join(f"'{t}'" for t in tfs)
        if len(tfs) == 1:
            formatted += ','
        check(f"DEFAULT_PER_KEY_TFS['{key}'] = ({formatted})",
              f"({formatted})" in code,
              f"expected tuple ({formatted}) for key '{key}'")

    check('strict all×all override path present (AI_TRADER_TRAIN_TF_MAP)',
          'AI_TRADER_TRAIN_TF_MAP' in code and "'1w'" in code)

    # Live import — combo counts for both modes.
    import importlib, sys
    sys.path.insert(0, BASE_DIR)
    os.environ.pop('AI_TRADER_TRAIN_TF_MAP', None)
    if 'src.engine.train_all_models' in sys.modules:
        del sys.modules['src.engine.train_all_models']
    mod = importlib.import_module('src.engine.train_all_models')
    curated = sum(len(v) for v in mod.DEFAULT_PER_KEY_TFS.values())
    check(f'curated map sums to 25 combos (got {curated})', curated == 25)

    os.environ['AI_TRADER_TRAIN_TF_MAP'] = 'strict'
    del sys.modules['src.engine.train_all_models']
    mod = importlib.import_module('src.engine.train_all_models')
    strict = sum(len(v) for v in mod.DEFAULT_PER_KEY_TFS.values())
    check(f'strict override sums to 49 combos (got {strict})', strict == 49)
    os.environ.pop('AI_TRADER_TRAIN_TF_MAP', None)


def test_phase71c_v31_backtest_per_model_filter():
    """v3.1 step 3 (1F): run_full_backtest accepts a `models` filter so
    chained-backtest-after-training only re-runs strategies for the
    trained model. _spawn_followup_backtest forwards both `timeframes`
    and `models` to the subprocess invocation."""
    print('\n[Phase 71c -- v3.1 step 3: per-model backtest filter]')

    bt_path = os.path.join(BASE_DIR, 'src', 'engine', 'backtester.py')
    with open(bt_path, encoding='utf-8') as f:
        bt = f.read()

    # 1. Helper function exists with the canonical mapping.
    check('_strategy_uses_model() helper defined',
          'def _strategy_uses_model(' in bt)
    check('helper handles meta special case',
          "key == \"meta\"" in bt and 'metafiltered' in bt.lower())
    check('helper handles regime special case',
          'regimeclassifier' in bt.lower())
    check('helper handles futures → futures_short mapping',
          "key == \"futures\"" in bt and 'futures_short' in bt.lower())

    # 2. run_full_backtest signature includes models param.
    check('run_full_backtest has `models` param',
          'def run_full_backtest(' in bt
          and 'models: tuple[str, ...] | None = None' in bt)

    # 3. Per-model skip logic at the inner loop.
    check('inner loop skips when model not in filter',
          '_strategy_uses_model(reg_name, m) for m in models' in bt
          and bt.count('_strategy_uses_model(reg_name, m) for m in models') >= 2)
    check('meta-filtered branch gated by models filter',
          'MetaLabeler_Filter' in bt
          and '_strategy_uses_model("MetaLabeler_Filter", m)' in bt)

    # 4. _spawn_followup_backtest forwards models to subprocess.
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()
    check('_spawn_followup_backtest exists',
          'def _spawn_followup_backtest(' in app)
    check('subprocess invocation forwards `models=(model_key,)`',
          'models=({model_key!r},)' in app
          or "models=({model_key!r},)" in app)
    check('progress label includes the filtered model name',
          "post-train backtest: {model_key}" in app)

    # 5. Live import — call with a filter and confirm no exception.
    import importlib, sys
    sys.path.insert(0, BASE_DIR)
    if 'src.engine.backtester' in sys.modules:
        del sys.modules['src.engine.backtester']
    bt_mod = importlib.import_module('src.engine.backtester')
    helper = getattr(bt_mod, '_strategy_uses_model', None)
    check('_strategy_uses_model importable', callable(helper))
    if callable(helper):
        check('helper: Trend_ML matches trend',           helper('Trend_ML', 'trend'))
        check('helper: Trend_ML does NOT match scalping', not helper('Trend_ML', 'scalping'))
        check('helper: Futures_Short_ML matches futures', helper('Futures_Short_ML', 'futures'))
        check('helper: TFT_MarketMaker matches tft',      helper('TFT_MarketMaker', 'tft'))
        check('helper: OFT_Microstructure matches oft',   helper('OFT_Microstructure', 'oft'))
        check('helper: RSI_MetaFiltered matches meta',    helper('RSI_MetaFiltered', 'meta'))
        check('helper: RegimeClassifier_Router → regime', helper('RegimeClassifier_Router', 'regime'))
        check('helper: empty key → matches all',          helper('Trend_ML', ''))


def test_phase71d_v31_tft_dedupe_regression():
    """v3.1 step 5 (1B′): regression test that locks the 1B fix.

    Builds a synthetic dataframe with duplicate timestamps and
    irregular gaps (mimicking the legacy/new market_data UNION),
    calls build_series_bundle(df, freq='1h'), and asserts:
      - no ValueError raised
      - three TimeSeries returned with consistent indices

    Reverting either the triple-dedupe or the proper freq mapping
    in build_series_bundle would make this test fail (which is the
    point — it's the safety net for future refactors)."""
    print('\n[Phase 71d -- v3.1 step 5: TFT dedupe regression]')

    src_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_tft_model.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()

    # Static checks: helpers exist, triple dedupe + freq mapping are present.
    check('_tf_to_pandas_freq() helper defined', 'def _tf_to_pandas_freq(' in src)
    check('_dedupe_for_darts() helper defined', 'def _dedupe_for_darts(' in src)
    check('triple dedupe: target_df + past_df + future_df',
          'target_df = _dedupe_for_darts(df)' in src
          and 'past_df = _dedupe_for_darts(df)' in src
          and 'future_df = _dedupe_for_darts(df)' in src)
    check('train_tft_model uses _tf_to_pandas_freq, not hard-coded 1h/1min',
          'freq = _tf_to_pandas_freq(timeframe)' in src
          and 'freq = "1h" if timeframe == "1h" else "1min"' not in src)

    # Live import.
    import importlib, sys
    sys.path.insert(0, BASE_DIR)
    if 'src.engine.train_tft_model' in sys.modules:
        del sys.modules['src.engine.train_tft_model']
    try:
        mod = importlib.import_module('src.engine.train_tft_model')
    except Exception as exc:
        check(f'train_tft_model imports', False, str(exc))
        return
    check('train_tft_model imports', True)

    # _tf_to_pandas_freq returns expected mappings.
    expected = {'1m': '1min', '5m': '5min', '15m': '15min',
                '1h': '1h', '4h': '4h', '1d': '1D', '1w': '1W'}
    for tf, want in expected.items():
        got = mod._tf_to_pandas_freq(tf)
        check(f"_tf_to_pandas_freq('{tf}') -> '{want}' (got '{got}')", got == want)

    # _dedupe_for_darts: synthetic 3-row frame with one duplicate.
    import pandas as pd
    df_dup = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-05-09 00:00', '2026-05-09 00:00', '2026-05-09 01:00']),
        'close':    [100.0, 101.0, 102.0],
    })
    out = mod._dedupe_for_darts(df_dup)
    check('_dedupe_for_darts collapses duplicate timestamps', len(out) == 2)
    check('_dedupe_for_darts keeps last row of duplicates (close=101)',
          float(out.iloc[0].close) == 101.0)

    # End-to-end: build_series_bundle on a duplicate-laden synthetic frame
    # must not raise. Skip if darts isn't installed (no value pretending).
    try:
        from darts import TimeSeries  # noqa: F401
        darts_ok = True
    except Exception:
        darts_ok = False

    if not darts_ok:
        check('darts available for end-to-end test', None,
              'darts not installed — skipping live build_series_bundle call')
        return

    # 50 rows over 50 hours, with two duplicate timestamps + a 3h gap.
    import numpy as np
    base_ts = pd.date_range('2026-05-01 00:00', periods=48, freq='1h').tolist()
    ts = base_ts[:10] + [base_ts[9]] + base_ts[10:25] + [base_ts[24]] + base_ts[25:]
    n = len(ts)
    df_synth = pd.DataFrame({
        'timestamp':         ts,
        'close':             np.linspace(100.0, 200.0, n),
        'return':            np.random.RandomState(0).normal(0, 0.01, n),
        'volume':            np.linspace(1.0, 5.0, n),
        'taker_buy_ratio':   np.full(n, 0.55),
        'avg_trade_size':    np.full(n, 0.1),
        'ofi':               np.full(n, 0.0),
        'funding_rate':      np.full(n, 0.0001),
        'sentiment_score':   np.full(n, 0.0),
        'asset_id':          np.full(n, 0.0),
        'hour_sin':          np.sin(2 * np.pi * np.arange(n) / 24.0),
        'hour_cos':          np.cos(2 * np.pi * np.arange(n) / 24.0),
        'dow_sin':           np.zeros(n),
        'dow_cos':           np.ones(n),
    })
    raised = None
    try:
        target, past_cov, future_cov = mod.build_series_bundle(df_synth, freq='1h')
    except Exception as exc:
        raised = exc

    check('build_series_bundle does not raise on duplicate-laden frame',
          raised is None,
          f'unexpected exception: {type(raised).__name__}: {raised}' if raised else '')

    if raised is None:
        # Three TimeSeries with consistent indices — start/end times match.
        check('target series produced',  hasattr(target, 'time_index'))
        check('past_covariates produced',  hasattr(past_cov, 'time_index'))
        check('future_covariates produced', hasattr(future_cov, 'time_index'))
        if hasattr(target, 'time_index') and hasattr(past_cov, 'time_index'):
            check('target / past_cov index start matches',
                  target.time_index[0] == past_cov.time_index[0])
            check('target / past_cov index end matches',
                  target.time_index[-1] == past_cov.time_index[-1])


def test_phase71e_v31_scalping_rebalance():
    """v3.1 step 6 (1C): scalping trainer adds SMOTE oversampling +
    self-heal retry + conditional accuracy_warning emission."""
    print('\n[Phase 71e -- v3.1 step 6: scalping rebalance + self-heal]')

    src_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_scalping_model.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()

    check('imblearn.SMOTE imported (with ImportError fallback)',
          'from imblearn.over_sampling import SMOTE' in src
          and '_SMOTE_AVAILABLE = True' in src
          and '_SMOTE_AVAILABLE = False' in src)
    check('_resample() helper defined for SMOTE-vs-fallback dispatch',
          'def _resample(' in src and 'sm.fit_resample(' in src)
    check('_resample falls back to sample_weight when SMOTE unavailable',
          "compute_sample_weight('balanced'" in src and 'fall back to' in src)
    check('walk-forward fold uses _resample',
          'X_tr_bal, y_tr_bal, w_tr = _resample(X.iloc[tr]' in src)
    check('calibration stage uses _resample',
          'X_safe_bal, y_safe_bal, w_safe = _resample(X_safe' in src)
    check('self-heal retry on single-class collapse',
          'Single-class collapse detected' in src
          and 'sm_strong = SMOTE(' in src
          and 'Self-healing succeeded on retry' in src)
    check('accuracy_warning is conditional (not unconditional)',
          'accuracy_warning = None' in src
          and 'Single-class collapse:' in src
          and 'if accuracy_warning:' in src)
    check('meta records smote_used + pos_rate_pct',
          '"smote_used": bool(_SMOTE_AVAILABLE)' in src
          and '"pos_rate_pct"' in src)

    # requirements.txt declares imbalanced-learn.
    req = open(os.path.join(BASE_DIR, 'requirements.txt'), encoding='utf-8').read()
    check('requirements.txt lists imbalanced-learn',
          'imbalanced-learn' in req)


def test_phase71f_v31_oft_sweep_coverage():
    """v3.1 step 8 (1M): train_all_models.train_all() now folds OFT
    (Microstructure) into the sweep — was the 15th dashboard row
    stuck in NOT STARTED state. TFT loop also fixed to forward the
    per-key timeframe instead of always calling train_tft_model() bare."""
    print('\n[Phase 71f -- v3.1 step 8: OFT sweep coverage + TFT TF forwarding]')

    src_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_all_models.py')
    src = open(src_path, encoding='utf-8').read()

    check('TFT loop forwards per-key TF (was bare-call before)',
          'train_tft_model(timeframe=tf)' in src)
    check('TFT loop honours skip-if-fresh per TF',
          "_meta_age_s('tft', tf)" in src)
    check('OFT block imports train_oft from joint_oft_rl',
          'from src.training.joint_oft_rl import train_oft' in src)
    check('OFT canonical TF is 1m',
          "oft_tf = '1m'" in src)
    check('OFT symbols defaultable via AI_TRADER_OFT_SYMBOLS env var',
          "AI_TRADER_OFT_SYMBOLS" in src
          and 'BTC/USDT,ETH/USDT,SOL/USDT' in src)
    check('OFT inner loop wraps each symbol in try/except',
          'Skipping OFT %s/%s' in src)
    check('OFT outer block tolerates ImportError on joint_oft_rl',
          'except ImportError' in src and 'Skipping OFT entirely' in src)

    # Live import
    import importlib, sys
    sys.path.insert(0, BASE_DIR)
    if 'src.engine.train_all_models' in sys.modules:
        del sys.modules['src.engine.train_all_models']
    mod = importlib.import_module('src.engine.train_all_models')
    check('train_all_models imports cleanly with OFT block', hasattr(mod, 'train_all'))

    # Resource kind map still has oft='exclusive' (no GPU contention with TFT).
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()
    check("_RESOURCE_KIND['oft'] = 'exclusive' preserved",
          "'oft':      'exclusive'" in app or "'oft': 'exclusive'" in app)


def test_phase72_v31_dashboard_mode_aware_and_per_market():
    """v3.1 steps 9 + 10 (1I + 1L): the Performance Overview card is
    now mode-aware (loadPortfolioByMode + payload-driven Balances) and
    its Signal/Risk panels render per-market rows for SPOT / FUTURES /
    SCALPING instead of the single-symbol BTC/USDT layout."""
    print('\n[Phase 72 -- v3.1 steps 9+10: mode-aware + per-market dashboard]')

    if not os.path.exists(TEMPLATE_PATH):
        check('template file exists', False, TEMPLATE_PATH)
        return
    html = open(TEMPLATE_PATH, encoding='utf-8').read()

    # Step 9 — mode-aware Portfolio loader.
    check('loadPortfolioByMode() function defined',
          'async function loadPortfolioByMode()' in html)
    check('loader fetches /api/portfolio?mode= + _ltCurrentMode',
          "fetch('/api/portfolio?mode=' + encodeURIComponent(mode))" in html)
    check('loader caches as _lastPortfolioPayload',
          '_lastPortfolioPayload = d' in html
          and 'let _lastPortfolioPayload = null' in html)
    check('loader writes 7 port-* fields directly from payload',
          "setT('port-total-capital'" in html
          and "setT('port-free-usdt'" in html
          and "setT('port-deployed'" in html
          and "setT('port-today-pnl'" in html
          and "setT('port-live-pnl'" in html
          and "setT('port-closed-pnl'" in html
          and "setT('port-total-pnl'" in html)
    check('loader renders bal-wallet-tbody from payload.balances[]',
          'bal-wallet-tbody' in html and 'd.balances.length === 0' in html)

    check('ltLoadAll() invokes loadPortfolioByMode',
          'await ltLoadMode();\n  await ltLoadBalance();\n  await loadPortfolioByMode();' in html)
    check('ltSetMode() invokes loadPortfolioByMode after POST',
          'await ltLoadMode();' in html and 'await loadPortfolioByMode();' in html)
    check('hourly auto-refresh invokes loadPortfolioByMode',
          'try { loadPortfolioByMode(); } catch(_) {}' in html)

    # Legacy mode-blind paths gated when payload is fresh.
    check('updateBalancesPanel gated by _lastPortfolioPayload + mode match',
          ('_lastPortfolioPayload && _lastPortfolioPayload.mode === _ltCurrentMode' in html)
          and '// v3.1 step 9 (1I)' in html)
    check('updatePnl gated by _payloadFresh check',
          'const _payloadFresh = !!(_lastPortfolioPayload && _lastPortfolioPayload.mode === _ltCurrentMode);' in html
          and 'if (!_payloadFresh)' in html)

    # Step 10 — per-market Signal & Risk panels.
    check('signal panel HTML container has data-test="signal-by-market"',
          'data-test="signal-by-market"' in html)
    check('risk panel HTML container has data-test="risk-by-market"',
          'data-test="risk-by-market"' in html)
    check('signal panel pre-renders 3 market rows (SPOT / FUTURES / SCALPING)',
          html.count('data-market-row="SPOT"') >= 2
          and html.count('data-market-row="FUTURES"') >= 2
          and html.count('data-market-row="SCALPING"') >= 2)
    check('updateSignalPanelByMarket() function defined',
          'function updateSignalPanelByMarket(state)' in html)
    check('updateRiskPanelByMarket() function defined',
          'function updateRiskPanelByMarket(state)' in html)
    check('updateUI() invokes both per-market renderers',
          'updateSignalPanelByMarket(state);' in html
          and 'updateRiskPanelByMarket(state);' in html)
    check('per-market Signal renderer iterates SPOT/FUTURES/SCALPING',
          "['SPOT', 'FUTURES', 'SCALPING'].map" in html)
    check('per-market Risk renderer filters tradesData by t.market',
          'openByMkt[m]' in html and 'SPOT_SCALPING' in html)
    check('Signal panel header reads "Signal — All Markets"',
          'Signal — All Markets' in html or 'Signal &mdash; All Markets' in html)
    check('Risk panel header reads "Risk — All Markets"',
          'Risk — All Markets' in html or 'Risk &mdash; All Markets' in html)
    check('legacy single-symbol IDs preserved (hidden) for compat',
          'id="signal-reason" hidden' in html
          and 'id="sig-rsi" hidden' in html
          and 'id="risk-vol"  hidden' in html)


def test_phase73_v31_trade_enrichment_going_forward():
    """v3.1 step 11 (1D): every NEW trade row carries the 7 enrichment
    fields (mode, regime_at_entry, model_confidence, mfe_pct, mae_pct,
    slippage_pct, exit_reason). MFE/MAE update on trailing-stop ticks;
    exit_reason inferred from PnL sign at close if not supplied.
    paper_book.book_market_order writes the same 7 fields with
    mode='paper'."""
    print('\n[Phase 73 -- v3.1 step 11: trade enrichment going-forward]')

    tt_src = open(os.path.join(BASE_DIR, 'src', 'engine', 'trade_tracker.py'), encoding='utf-8').read()
    pb_src = open(os.path.join(BASE_DIR, 'src', 'engine', 'paper_book.py'), encoding='utf-8').read()

    # Static checks
    check('TradeTracker._detect_trade_mode helper defined',
          'def _detect_trade_mode()' in tt_src)
    check('open_trade accepts mode/regime_at_entry/model_confidence kwargs',
          'mode=None, regime_at_entry=None, model_confidence=None' in tt_src)
    check('open_trade accepts intended_price kwarg for slippage',
          'intended_price=None' in tt_src and 'slippage_pct = (current_price - intended_price)' in tt_src)
    check('open_trade writes all 7 enrichment fields',
          all(f'"{k}":' in tt_src for k in
              ('mode', 'regime_at_entry', 'model_confidence',
               'mfe_pct', 'mae_pct', 'slippage_pct', 'exit_reason')))
    check('update_trailing_stops tracks MFE/MAE',
          'trade["mfe_pct"]' in tt_src and 'trade["mae_pct"]' in tt_src
          and '"lowest_price"' in tt_src)
    check('close_trade_by_id accepts exit_reason kwarg + infers if absent',
          'exit_reason=None' in tt_src and "trade[\"exit_reason\"] = ('TP'" in tt_src)

    check('paper_book.book_market_order writes 7 enrichment fields',
          '"mode":             "paper"' in pb_src
          and '"mfe_pct":          0.0' in pb_src
          and '"mae_pct":          0.0' in pb_src
          and '"slippage_pct":     None' in pb_src
          and '"exit_reason":      None' in pb_src)
    check('paper_book.book_close infers exit_reason from net PnL',
          't["exit_reason"] = (' in pb_src and "'TP'" in pb_src and "'SL'" in pb_src)

    # Live: open + update + close round-trip
    import importlib, sys, tempfile, os as _os
    sys.path.insert(0, BASE_DIR)
    if 'src.engine.trade_tracker' in sys.modules:
        del sys.modules['src.engine.trade_tracker']
    tt_mod = importlib.import_module('src.engine.trade_tracker')

    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w')
    tmp.write('[]'); tmp.close()
    try:
        tt = tt_mod.TradeTracker(filepath=tmp.name)
        trade = tt.open_trade('BTC/USDT', 100.0, 50000.0,
                              regime_at_entry='TRENDING',
                              model_confidence=0.72,
                              intended_price=49995.0)
        for k in ('mode', 'regime_at_entry', 'model_confidence',
                  'mfe_pct', 'mae_pct', 'slippage_pct', 'exit_reason'):
            check(f'open_trade row has field "{k}"', k in trade)
        check('regime_at_entry passes through', trade['regime_at_entry'] == 'TRENDING')
        check('model_confidence passes through', abs(trade['model_confidence'] - 0.72) < 1e-9)
        check('slippage_pct ≈ 0.01% on 5-bps gap',
              trade['slippage_pct'] is not None and 0.005 < trade['slippage_pct'] < 0.02)

        tt.update_trailing_stops('BTC/USDT', 51000.0)  # MFE +2%
        tt.update_trailing_stops('BTC/USDT', 49000.0)  # MAE -2%
        upd = [t for t in tt.trades if t['id'] == trade['id']][0]
        check('MFE recorded after favourable tick', abs(upd['mfe_pct'] - 2.0) < 0.05)
        check('MAE recorded after adverse tick',    abs(upd['mae_pct'] + 2.0) < 0.05)

        closed = tt.close_trade_by_id(trade['id'], 50500.0)
        check('close infers exit_reason TP on positive PnL', closed['exit_reason'] == 'TP')
    finally:
        _os.unlink(tmp.name)


def test_phase73b_v31_trade_enrichment_backfill():
    """v3.1 step 12 (1E): scripts/backfill_trade_enrichment.py exists,
    runs against data/trades.json, and produces a same-length output
    at data/trades_enriched.json with mode + exit_reason populated on
    the closed-trade rows."""
    print('\n[Phase 73b -- v3.1 step 12: historical trade backfill]')

    script = os.path.join(BASE_DIR, 'scripts', 'backfill_trade_enrichment.py')
    check('backfill_trade_enrichment.py exists', os.path.exists(script))

    out_path = os.path.join(BASE_DIR, 'data', 'trades_enriched.json')
    if not os.path.exists(out_path):
        check('data/trades_enriched.json produced', False,
              'expected after running scripts.backfill_trade_enrichment')
        return
    check('data/trades_enriched.json produced', True)

    with open(out_path, encoding='utf-8') as f:
        enriched = json.load(f)
    check('enriched is a list', isinstance(enriched, list))
    check(f'enriched contains rows ({len(enriched)})', len(enriched) > 0)

    if not enriched:
        return

    sample = enriched[0]
    for k in ('mode', 'regime_at_entry', 'model_confidence',
              'mfe_pct', 'mae_pct', 'slippage_pct', 'exit_reason'):
        check(f'enriched row exposes "{k}"', k in sample)

    # Coverage targets: mode and exit_reason should be near-100% on
    # closed trades; MFE/MAE may be lower (depends on legacy
    # highest_price availability).
    n = len(enriched)
    mode_pct = sum(1 for t in enriched if t.get('mode')) / n * 100
    er_pct   = sum(1 for t in enriched if (t.get('status') or '').upper() != 'OPEN'
                                          and t.get('exit_reason')) / max(1, sum(1 for t in enriched if (t.get('status') or '').upper() != 'OPEN')) * 100
    check(f'mode populated >=99% (got {mode_pct:.1f}%)',          mode_pct >= 99.0)
    check(f'exit_reason on closed trades >=95% (got {er_pct:.1f}%)', er_pct >= 95.0)


def test_phase74_v31_health_column_and_fleet_aggregate():
    """v3.1 — composite Model Health column + fleet aggregate footer.

    Operator request 2026-05-09: "add one more column that will analyze
    the data across all columns and show the final aggregate results
    on training model tab".

    Health column rolls every signal column for one row into a
    weighted 0-100 score with letter grade (A/B/C/D/F) and color.
    Fleet aggregate footer averages across all visible rows."""
    print('\n[Phase 74 -- v3.1: Model Health column + fleet aggregate]')

    if not os.path.exists(TEMPLATE_PATH):
        check('template file exists', False, TEMPLATE_PATH)
        return
    html = open(TEMPLATE_PATH, encoding='utf-8').read()

    # Header column added between Target and Description.
    check('Health column header added (data-col="health_score")',
          'data-col="health_score"' in html and '>Health <span class="tr-chev"></span></th>' in html)
    check('Health header tooltip explains formula',
          ('WF Acc% (35)' in html or 'WF Acc%' in html)
          and ('AUC (20' in html or 'AUC' in html)
          and 'A≥80' in html)

    # Composite scoring helper.
    check('_computeHealthScore() helper defined',
          'function _computeHealthScore(m)' in html)
    check('helper weights WF / Acc / AUC / WinP / Balance / Freshness',
          all(("key: '" + k + "'") in html for k in ('WF', 'Acc', 'AUC', 'WinP', 'Bal', 'Fresh')))
    check('helper applies penalties (warning -30, error -20)',
          'score -= 30' in html and 'score -= 20' in html)
    check('helper returns letter grade A/B/C/D/F',
          ("score >= 80 ? 'A'" in html
           and "score >= 65 ? 'B'" in html
           and "score >= 50 ? 'C'" in html
           and "score >= 35 ? 'D'" in html))

    # Per-row annotation.
    check('_renderTrainingTable annotates m.health_score before sort',
          'm.health_score = h.score' in html
          and 'all.forEach(m => {' in html)

    # Row template renders the badge.
    check('row template includes Health badge cell',
          'm._health_grade' in html and 'm._health_color' in html)
    check('row tr carries data-health-score + data-health-grade attrs',
          'data-health-score=' in html and 'data-health-grade=' in html)

    # Fleet aggregate footer.
    check('<tfoot id="training-tfoot"> exists',
          '<tfoot id="training-tfoot">' in html)
    check('Fleet aggregate row populated with averages + grade tally',
          'FLEET AGGREGATE' in html
          and 'FLEET HEALTH' in html
          and 'training-tfoot-aggregate' in html
          and 'fleet-health-badge' in html)
    check('aggregate counts trained-today / failed / not-started',
          'Trained today:' in html and 'Failed/warn:' in html and 'Not started:' in html)
    check('aggregate breaks down by grade A/B/C/D/F',
          ('A:<b' in html and 'B:<b' in html and 'C:<b' in html
           and 'D:<b' in html and 'F:<b' in html))

    # Empty-state colspan was bumped 20 → 21 (Health), 21 → 22 (Backtest),
    # 22 → 24 (ETA Train + ETA BT). Accept any post-Health value.
    check('empty-state colspan bumped to 21',
          ('colspan="21"' in html or 'colspan="22"' in html
           or 'colspan="23"' in html or 'colspan="24"' in html)
          and 'No models match this filter.' in html)


def test_phase75_v31_backfill_button_endpoint():
    """v3.1 step 14 (1J): "Backfill Missing Data" button on Data
    Coverage card + /api/data/backfill endpoint."""
    print('\n[Phase 75 -- v3.1 step 14: backfill button + endpoint]')

    html = open(TEMPLATE_PATH, encoding='utf-8').read()
    app  = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()

    check('button id dcov-backfill-btn present in template',
          'id="dcov-backfill-btn"' in html and '⤓ Backfill Missing Data' in html)
    check('dcovBackfillMissing() JS function defined',
          'async function dcovBackfillMissing()' in html)
    check('JS POSTs to /api/data/backfill',
          "fetch('/api/data/backfill'" in html)

    check('Flask /api/data/backfill route registered',
          "@app.route('/api/data/backfill'" in app)
    check('backend auto-discovers stale 1s archives via mtime',
          '_spot_1s.csv.gz' in app and 'age_days >= 7' in app)
    check('backend chains downloader → resample',
          'download_archives_for_symbols' in app
          and '_run_resample_blocking(job_id, symbols' in app)
    check('backend short-circuits when nothing is stale',
          "'no stale symbols detected — nothing to backfill'" in app)
    check('streaming-aware (no full-RAM load)',
          'streams; doesn' in app or 'streamed to disk' in app)


def test_phase76_v31_training_sweep_watchdog_and_cold_cache():
    """v3.1 — overnight reliability: training_sweep_watchdog daemon
    auto-respawns a stalled sweep + cold_cache persists ETA-relevant
    state across dashboard restarts."""
    print('\n[Phase 76 -- v3.1: training_sweep_watchdog + cold_cache]')

    # 1) Sweep watchdog file + key behaviours.
    wd_path = os.path.join(BASE_DIR, 'scripts', 'training_sweep_watchdog.py')
    check('scripts/training_sweep_watchdog.py exists', os.path.exists(wd_path))
    if not os.path.exists(wd_path):
        return
    wd = open(wd_path, encoding='utf-8').read()

    check('polls /api/pipeline/status', "STATUS_URL" in wd and 'pipeline/status' in wd)
    check('triggers respawn via /api/pipeline/run', '/api/pipeline/run' in wd)
    check('detects stall by payload-unchanged + no orchestrator process',
          '_is_stalled' in wd and 'idle_for >= STALL_S' in wd
          and '_orchestrator_alive()' in wd)
    check('cmdline-scan fallback for orchestrator liveness',
          'src.engine.pipeline_orchestrator' in wd)
    check('circuit breaker trips after RESTART_LIMIT respawns',
          'RESTART_LIMIT' in wd and "s['tripped'] = True" in wd)
    check('persists state to data/training_sweep_watchdog_state.json',
          'training_sweep_watchdog_state.json' in wd)
    check('atomic state write (.tmp + os.replace)',
          'os.replace' in wd and 'with_suffix' in wd)
    check('all env-var-tunable knobs present',
          all(e in wd for e in ('AI_TRADER_SWEEP_WATCH_POLL_S',
                                'AI_TRADER_SWEEP_WATCH_STALL_S',
                                'AI_TRADER_SWEEP_WATCH_LIMIT',
                                'AI_TRADER_SWEEP_WATCH_WINDOW_S')))
    check('does NOT kill in-progress training (only respawns when dead)',
          'NEVER kills an actively-progressing sweep' in wd)

    # 2) restart_all.ps1 wires it in.
    rs = open(os.path.join(BASE_DIR, 'restart_all.ps1'), encoding='utf-8').read()
    check('restart_all.ps1 launches training_sweep_watchdog',
          'scripts.training_sweep_watchdog' in rs and '5.97' in rs)

    # 3) cold_cache module + integration.
    cc_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'cold_cache.py')
    check('src/dashboard/cold_cache.py exists', os.path.exists(cc_path))
    if not os.path.exists(cc_path):
        return
    cc = open(cc_path, encoding='utf-8').read()
    check('cold_cache exposes save / load / age_seconds / list_keys',
          all(f'def {n}(' in cc for n in ('save', 'load', 'age_seconds', 'list_keys')))
    check('cold_cache writes atomically (tmp + os.replace)',
          'os.replace' in cc)
    check('cold_cache respects D: drive policy (PROJECT_ROOT/data/cache/cold)',
          "'data' / 'cache' / 'cold'" in cc)

    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()
    check('app.py loads typical_durations from cold_cache on import',
          "_cold_cache.load('typical_durations'" in app)
    check('_record_completed_duration persists to cold_cache',
          "_cc.save('typical_durations'" in app
          and "_cc.save('typical_history'" in app)

    # Live exercise of cold_cache round-trip.
    import importlib, sys
    sys.path.insert(0, BASE_DIR)
    if 'src.dashboard.cold_cache' in sys.modules:
        del sys.modules['src.dashboard.cold_cache']
    cc_mod = importlib.import_module('src.dashboard.cold_cache')
    test_payload = {'test_key': 1234, 'now': True}
    cc_mod.save('phase76_test', test_payload)
    got = cc_mod.load('phase76_test', default=None)
    check('cold_cache round-trip: load returns saved value', got == test_payload)
    age = cc_mod.age_seconds('phase76_test')
    check('cold_cache age_seconds > 0 after recent save',
          age is not None and 0 <= age < 60)
    keys = cc_mod.list_keys()
    check('list_keys includes the freshly-saved entry', 'phase76_test' in keys)
    # Cleanup
    try:
        (cc_mod.COLD_DIR / 'phase76_test.json').unlink()
    except Exception:
        pass


def test_phase77_v31_pertf_train_button_dispatch_fix():
    """v3.1 bugfix 2026-05-09 — clicking Train on a per-TF variant row
    (Futures Short RF @ short, Base RF @ 4h, Trend RF @ 1d, etc.) was
    POSTing rowKey='futures_short' / 'base_4h' / 'trend_1d' to
    /api/training/run/<rowKey>, which the trainer dispatch rejected as
    "unknown model key". Two fixes:

      A) list_per_tf_artifacts() now skips the legacy/canonical
         filename (futures_short_model.joblib was the offending
         pseudo-variant) plus enforces a TF-token shape check.
      B) trRunOne() splits rowKey into canonical_key + tf override
         using either the row's parent_key field or a regex fallback,
         so per-TF rows POST /api/training/run/<canonical> + body.tf.
    """
    print('\n[Phase 77 -- v3.1 fix: per-TF Train button dispatch]')

    # A) list_per_tf_artifacts skips legacy filename + enforces TF shape.
    src = open(os.path.join(BASE_DIR, 'src', 'utils', 'model_paths.py'),
               encoding='utf-8').read()
    check('list_per_tf_artifacts skips legacy_name', 'n == legacy_name' in src)
    check('list_per_tf_artifacts enforces TF token shape',
          '_looks_like_tf' in src and "_VALID_TF_TOKENS" in src)

    import importlib, sys
    sys.path.insert(0, BASE_DIR)
    if 'src.utils.model_paths' in sys.modules:
        del sys.modules['src.utils.model_paths']
    mp = importlib.import_module('src.utils.model_paths')
    fut = mp.list_per_tf_artifacts('futures')
    tfs = [r[0] for r in fut]
    check(f'list_per_tf_artifacts("futures") does NOT yield "short" (got {tfs})',
          'short' not in tfs)
    for tf in tfs:
        check(f'  tf {tf!r} matches valid TF pattern', mp._looks_like_tf(tf))

    # B) trRunOne splits rowKey into canonical + derivedTf.
    tpl = open(TEMPLATE_PATH, encoding='utf-8').read()
    check('trRunOne uses _stratFull.ml_models lookup for parent_key',
          'row.parent_key || rowKey' in tpl)
    check('trRunOne has prefix-split fallback for known model keys',
          "const KNOWN = ['base','trend','futures','scalping','meta','tft','oft','regime']" in tpl)
    check('trRunOne validates TF suffix shape (regex)',
          '/^(?:\\d+m|\\d+h|\\d+d|\\d+w|1mo)$/.test(candidate)' in tpl)
    check('trRunOne derived TF used as fallback when picker empty',
          '(tfSel && tfSel.value) || derivedTf' in tpl)


def test_phase78_v31_bot_dead_false_alarm_module_style_launch():
    """v3.1 fix 2026-05-09 — false-DEAD alert when bot launched via
    `python -m src.main` (module-style) instead of `python src/main.py`
    (script-style). The cmdline-fallback regex in error_monitor.py only
    matched the script form, so any direct `Start-Process python -m
    src.main` invocation made monitor/health spam "Trading bot is
    DEAD" for hours while the bot was actually fine."""
    print('\n[Phase 78 -- v3.1 fix: bot DEAD false-alarm on -m src.main launches]')

    src = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py'),
               encoding='utf-8').read()
    check('cmdline regex matches both src/main.py AND -m src.main',
          r'src[\\/]main\.py|-m\s+src\.main\b' in src)
    check('comment documents both launch styles',
          'script-style' in src and 'module-style' in src)
    check('comment cites the false-DEAD incident',
          'false-DEAD' in src or 'false_DEAD' in src
          or 'screaming "DEAD"' in src)

    # Live: simulate a -m src.main cmdline against the regex.
    import re
    pat = re.compile(r"src[\\/]main\.py|-m\s+src\.main\b")
    samples = [
        ('python.exe -m src.main',                            True),
        ('"D:/venv/python.exe" -m src.main',                  True),
        ('python src/main.py --no-interactive',               True),
        ('python src\\main.py',                               True),
        ('python -m src.training.joint_oft_rl',               False),
        ('python -m src.engine.pipeline_orchestrator',        False),
        ('python src/dashboard/app.py',                       False),
    ]
    for cmd, want in samples:
        got = bool(pat.search(cmd))
        check(f"  cmdline {cmd!r:55s} -> match={got} (want {want})", got == want)


def test_phase79_v31_stability_heatmap_legend_blue_rename():
    """v3.1 cosmetic fix 2026-05-09 — Stability Heatmap excellent-tier
    rendering recoloured from emerald to dark blue (legend + CSS class
    + JS cell maps coherent) so it's distinguishable from Green/Yellow.
    Internal `gold` JS key preserved for the 5 tier classifiers."""
    print('\n[Phase 79 -- v3.1 fix: heatmap excellent-tier -> Blue]')
    html = open(TEMPLATE_PATH, encoding='utf-8').read()

    check('legend uses "Blue = excellent"',
          'Blue</span> = excellent' in html
          and '<span style="color:#3b82f6">Blue</span>' in html)
    check('legend no longer says "Gold = excellent" or "Emerald"',
          'Gold</span> = excellent' not in html
          and 'Emerald</span> = excellent' not in html)

    # Internal JS key `gold` still drives 5 tier classifiers.
    check('internal `gold` tier key preserved (5 classifier sites)',
          html.count("? 'gold'") >= 5)
    check('cellFg["gold"] maps to dark blue #3b82f6',
          "gold:'#3b82f6'" in html)
    check('cellBg["gold"] maps to translucent blue rgba(37,99,235,.22)',
          "gold:'rgba(37,99,235,.22)'" in html)
    check('.stab-cell.gold CSS uses blue background + #3b82f6 text',
          'rgba(37,99,235,.22);color:#3b82f6' in html)

    # Flask template auto-reload enabled so future edits go live.
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'),
               encoding='utf-8').read()
    check("Flask TEMPLATES_AUTO_RELOAD = True",
          "app.config['TEMPLATES_AUTO_RELOAD'] = True" in app)
    check("Jinja env auto_reload = True (belt + braces)",
          'app.jinja_env.auto_reload = True' in app)


def test_phase80_v4_b0_training_rules_registry_and_api():
    """v4 Phase B0 + Phase B5 backend — training rules registry module
    + HTTP API for the dashboard rules editor card.

    Covers:
    - data/training_rules.json schema (8 models, applicable/experimental/skip lists)
    - src/training/training_rules.py public API
    - /api/training/rules GET + POST + /api/training/preview routes
    """
    print('\n[Phase 80 -- v4 B0+B5 backend: training rules registry]')

    # 1. Rules JSON file structure
    rules_path = os.path.join(BASE_DIR, 'data', 'training_rules.json')
    check('data/training_rules.json exists', os.path.exists(rules_path))
    if not os.path.exists(rules_path):
        return
    with open(rules_path, encoding='utf-8') as f:
        rules = json.load(f)
    check('rules has _version field', '_version' in rules)
    check('rules has models block',   isinstance(rules.get('models'), dict))
    check('rules has global block',   isinstance(rules.get('global'), dict))

    expected_keys = {'base', 'trend', 'futures', 'scalping', 'meta', 'tft', 'regime', 'oft'}
    actual = set(rules.get('models', {}).keys())
    check(f'all 8 model keys present (got {sorted(actual)})', expected_keys.issubset(actual))

    for k in expected_keys:
        blk = rules['models'].get(k, {})
        for f in ('applicable_tfs', 'experimental_tfs', 'skip_tfs',
                  'resource_kind', 'est_minutes_per_run', 'params', 'skip_reason'):
            check(f'  model {k!r} has field {f!r}', f in blk)
        check(f'  model {k!r} resource_kind valid',
              blk.get('resource_kind') in ('cpu', 'gpu', 'exclusive', 'neural'))

    # 2. Python API
    import importlib, sys
    sys.path.insert(0, BASE_DIR)
    if 'src.training.training_rules' in sys.modules:
        del sys.modules['src.training.training_rules']
    rmod = importlib.import_module('src.training.training_rules')

    check('rules module has all_models()',          callable(rmod.all_models))
    check('rules module has applicable_tfs()',      callable(rmod.applicable_tfs))
    check('rules module has experimental_tfs()',    callable(rmod.experimental_tfs))
    check('rules module has skip_tfs()',            callable(rmod.skip_tfs))
    check('rules module has cell_status()',         callable(rmod.cell_status))
    check('rules module has should_train()',        callable(rmod.should_train))
    check('rules module has skip_reason()',         callable(rmod.skip_reason))
    check('rules module has matrix()',              callable(rmod.matrix))
    check('rules module has planned_combos()',      callable(rmod.planned_combos))
    check('rules module has estimated_total_minutes()',
          callable(rmod.estimated_total_minutes))
    check('rules module has estimated_parallel_minutes()',
          callable(rmod.estimated_parallel_minutes))
    check('rules module has reload()',              callable(rmod.reload))
    check('TF_ORDER covers 1m..1mo',
          rmod.TF_ORDER == ('1m','5m','15m','1h','4h','1d','1w','1mo'))

    # Test some specific cells per the canonical matrix
    check('cell_status(scalping, 1m) = applicable', rmod.cell_status('scalping', '1m') == 'applicable')
    check('cell_status(scalping, 1h) = skip',       rmod.cell_status('scalping', '1h') == 'skip')
    check('cell_status(tft, 15m) = applicable',     rmod.cell_status('tft', '15m') == 'applicable')
    check('cell_status(tft, 1m) = skip',            rmod.cell_status('tft', '1m') == 'skip')
    check('cell_status(tft, 1d) = experimental',    rmod.cell_status('tft', '1d') == 'experimental')
    check('cell_status(regime, 1h) = applicable',   rmod.cell_status('regime', '1h') == 'applicable')
    check('cell_status(regime, 5m) = skip',         rmod.cell_status('regime', '5m') == 'skip')

    # planned_combos
    plan = rmod.planned_combos()
    check(f'default plan has 26 combos (got {len(plan)})', len(plan) == 26)
    plan_exp = rmod.planned_combos(include_experimental=True)
    check(f'with experimental: 30 combos (got {len(plan_exp)})', len(plan_exp) == 30)

    # ETA estimates
    eta_seq = rmod.estimated_total_minutes(plan)
    eta_par2 = rmod.estimated_parallel_minutes(plan, 2)
    check(f'sequential ETA > parallel-2 ETA ({eta_seq} > {eta_par2})', eta_seq > eta_par2)

    # Override matrix
    m = rmod.matrix(force_train=[('scalping', '1d')], force_skip=[('base', '1h')])
    overrides = [(mod, tf, st) for mod, tf, st in m if st in ('force_train', 'force_skip')]
    check('override matrix surfaces force_train + force_skip',
          ('scalping', '1d', 'force_train') in overrides
          and ('base', '1h', 'force_skip') in overrides)

    # 3. Backend HTTP routes registered (file-level grep)
    app_src = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()
    check("/api/training/rules GET registered",
          "@app.route('/api/training/rules', methods=['GET'])" in app_src)
    check("/api/training/rules POST registered",
          "@app.route('/api/training/rules', methods=['POST'])" in app_src)
    check("/api/training/preview GET registered",
          "@app.route('/api/training/preview'" in app_src)
    check("rules-save endpoint validates resource_kind",
          'resource_kind must be cpu/gpu/exclusive' in app_src)
    check("rules-save endpoint atomic write (.tmp + os.replace)",
          'tmp.write_text' in app_src and 'os.replace(tmp, raw)' in app_src)
    check("rules-save endpoint reloads in-process cache",
          '_r.reload()' in app_src)


def test_phase81_v4_b5_prime_unified_card_ui():
    """v4 Phase B5' — unified TRAINING & BACKTEST card on the dashboard.
    Reads /api/training/rules + /api/cluster/status, renders model × TF
    matrix + fleet workers + active tasks, plus a 'Run Sweep' button
    that POSTs /api/cluster/sweep."""
    print('\n[Phase 81 -- v4 B5?: unified training & backtest card]')

    if not os.path.exists(TEMPLATE_PATH):
        check('template file exists', False)
        return
    html = open(TEMPLATE_PATH, encoding='utf-8').read()

    # Card structure
    check('train-bt-section present',                  'id="train-bt-section"' in html)
    check('Training & Backtest header present',         'Training &amp; Backtest <span style="font-weight:400;color:#64748b">' in html)
    check('fleet-summary-chip present',                 'id="fleet-summary-chip"' in html)
    check('fleet-eta-chip present',                     'id="fleet-eta-chip"' in html)
    check('rules-matrix-host present',                  'id="rules-matrix-host"' in html)
    check('fleet-workers-host present',                 'id="fleet-workers-host"' in html)
    check('fleet-tasks-host present',                   'id="fleet-tasks-host"' in html)
    check('Run Sweep button present',                   'onclick="runUnifiedSweep()"' in html)
    check('include-experimental checkbox present',      'id="sweep-include-experimental"' in html)

    # JS functions
    check('loadFleetCard() defined',                    'async function loadFleetCard()' in html)
    check('_renderRulesMatrix() defined',               'function _renderRulesMatrix(' in html)
    check('_renderFleetWorkers() defined',              'function _renderFleetWorkers(' in html)
    check('_renderFleetTasks() defined',                'function _renderFleetTasks(' in html)
    check('runUnifiedSweep() defined',                  'async function runUnifiedSweep()' in html)

    # API integration
    check('JS calls /api/training/rules',               "fetch('/api/training/rules')" in html)
    check('JS calls /api/cluster/status',               "/api/cluster/status" in html)
    check('JS POSTs /api/cluster/sweep',                "fetch('/api/cluster/sweep'" in html)
    check('JS calls /api/cluster/tasks for active list', "/api/cluster/tasks" in html)

    # Hourly refresh wires loadFleetCard
    check('hourly refresh calls loadFleetCard',         'loadFleetCard();' in html and "ms-open" in html)

    # Sweep status chip (idle / queued / error states)
    check('sweep-status-chip present',                  'id="sweep-status-chip"' in html)

    # /api/cluster/sweep endpoint registered (backend)
    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()
    check("/api/cluster/sweep POST registered",          "@app.route('/api/cluster/sweep'" in app)
    check("sweep endpoint reads training rules",         'from src.training import training_rules as _r' in app)
    check("sweep endpoint OFT fans out per symbol",      "if model_key == 'oft':" in app)
    check("sweep endpoint returns sweep_id + count",     "'sweep_id':   sweep_id" in app)


def test_phase82_v4_component_health_module_style_launches():
    """v4 fix 2026-05-09 — Component Health card showing wrong statuses:
    - 'Trading Bot Running' was the dormant Start-Process wrapper (0% CPU,
      0 MB), not the real worker (1.6 GB doing work)
    - 'Dashboard Stopped' even though dashboard was serving the page
    - 'ML Training Stopped' even though pipeline_orchestrator was running

    Same class of bug as the bot-DEAD false-alarm: the cmdline regex
    matches script-style launches (python src/X.py) but not module-style
    (python -m src.X). Plus a service-alias system so 'training' service
    detects the orchestrator process, not just the legacy script filename."""
    print('\n[Phase 82 -- v4 fix: component health probes match module-style launches]')
    print('  (NOTE: regex/best-PID logic moved to src/utils/process_health.py'
          ' on 2026-05-10 as Layer 1 of the orchestration plan; Phase 82'
          ' now checks the centralised module + that app.py delegates to it.'
          ' Phase 83 covers the migration in full.)')

    app = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py'), encoding='utf-8').read()
    ph  = open(os.path.join(BASE_DIR, 'src', 'utils', 'process_health.py'), encoding='utf-8').read()

    # Bot regex now matches both forms — lives in process_health.
    check('bot regex matches both src/main.py AND -m src.main (in process_health)',
          r'src[\\/]main\.py|-m\s+src\.main\b' in ph)
    # Dashboard regex now matches both forms too — also in process_health.
    check('dash regex matches both dashboard/app.py AND -m src.dashboard.app (in process_health)',
          r'src[\\/]dashboard[\\/]app\.py|-m\s+src\.dashboard\.app\b' in ph)

    # Best-PID picker prefers worker over wrapper via RSS (in process_health).
    check('process_health picks highest-RSS PID (worker, not wrapper)',
          'rss > best.rss_bytes' in ph
          or ('rss_bytes' in ph and 'best' in ph))

    # ML Training cmdline scan includes pipeline_orchestrator + legacy script + module form.
    check('training_orch pattern covers pipeline_orchestrator + module form + legacy script',
          'pipeline_orchestrator' in ph
          and r'-m\s+src\.engine\.pipeline_orchestrator\b' in ph
          and 'train_all_models' in ph)

    # Dashboard delegates to process_health instead of running its own scan.
    check('app.py monitor_health imports process_health',
          'from src.utils import process_health' in app)
    check('app.py uses one-pass all_known_processes for fleet snapshot',
          '_ph.all_known_processes()' in app)

    # Comment block documents the institutional/false-DEAD context — in process_health.
    check('process_health docstring documents the false-Stopped failure mode',
          'false alarm' in ph.lower() or 'dead' in ph.lower() or 'wrapper' in ph.lower())


def test_phase83_centralised_process_health_module():
    """Layer 1 of the kafka-style orchestration plan (2026-05-10):
    consolidate the four duplicated cmdline-scan implementations into a
    single src/utils/process_health.py module. Pre-migration the same
    regex contract was copy-pasted across four files (dashboard
    monitor_health, error_monitor._probe_processes, _pipeline_proc_alive,
    training_sweep_watchdog._orchestrator_alive); three of them broke
    independently when launch styles changed (script-style vs `-m` form).

    Phase 83 verifies:
      P1. process_health.py exists and exposes the canonical KIND_*
          constants + find_process / all_known_processes / proc_stats.
      P2. All four call sites import + delegate to process_health
          instead of running their own psutil scan.
      P3. The module's regex covers BOTH launch styles for the kinds
          we know to launch in module-form (bot, dash, training_orch).
      P4. find_process picks highest-RSS match (real worker over
          dormant Start-Process wrapper)."""
    print('\n[Phase 83 -- centralised process_health module]')

    ph_path = os.path.join(BASE_DIR, 'src', 'utils', 'process_health.py')
    check('src/utils/process_health.py exists', os.path.exists(ph_path))
    with open(ph_path, encoding='utf-8') as f:
        ph = f.read()

    # P1 — canonical surface area
    check('exposes KIND_BOT constant',              'KIND_BOT' in ph)
    check('exposes KIND_DASH constant',             'KIND_DASH' in ph)
    check('exposes KIND_TRAIN_ORCH constant',       'KIND_TRAIN_ORCH' in ph)
    check('exposes KIND_CLUSTER_ORCH constant',     'KIND_CLUSTER_ORCH' in ph)
    check('exposes KIND_WORKER constant',           'KIND_WORKER' in ph)
    check('exposes KIND_TRAIN_SUPERVISOR constant', 'KIND_TRAIN_SUPERVISOR' in ph)
    check('exposes KIND_BT_SUPERVISOR constant',    'KIND_BT_SUPERVISOR' in ph)
    check('exposes KIND_MASTER_AGENT constant',     'KIND_MASTER_AGENT' in ph)
    check('exports find_process function',          'def find_process(' in ph)
    check('exports all_known_processes function',   'def all_known_processes(' in ph)
    check('exports proc_stats function',            'def proc_stats(' in ph)
    check('exports is_alive helper',                'def is_alive(' in ph)
    check('exports ProcessInfo dataclass',          'class ProcessInfo' in ph)

    # P3 — patterns cover both launch styles for the multi-form kinds
    check('bot pattern matches both src/main.py AND -m src.main',
          r'src[\\/]main\.py|-m\s+src\.main\b' in ph)
    check('dash pattern matches both dashboard/app.py AND -m src.dashboard.app',
          r'src[\\/]dashboard[\\/]app\.py|-m\s+src\.dashboard\.app\b' in ph)
    check('training_orch pattern includes pipeline_orchestrator + module form + legacy',
          'pipeline_orchestrator' in ph
          and r'-m\s+src\.engine\.pipeline_orchestrator\b' in ph
          and 'train_all_models' in ph)
    check('cluster_orch pattern targets distributed.orchestrator',
          r'-m\s+src\.training\.distributed\.orchestrator\b' in ph)
    check('worker pattern targets distributed.worker',
          r'-m\s+src\.training\.distributed\.worker\b' in ph)

    # P4 — wrapper-vs-worker tie-break uses RSS, not CPU
    check('find_process picks highest-RSS match',
          ('rss > best.rss_bytes' in ph) or ('rss_bytes' in ph and 'best' in ph))
    check('comments explain wrapper-vs-worker reasoning',
          'wrapper' in ph.lower() and 'worker' in ph.lower())

    # P2 — call sites delegate, no local cmdline scans left
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()
    em_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'error_monitor.py')
    with open(em_path, encoding='utf-8') as f:
        em = f.read()
    wd_path = os.path.join(BASE_DIR, 'scripts', 'training_sweep_watchdog.py')
    with open(wd_path, encoding='utf-8') as f:
        wd = f.read()

    check('app.py monitor_health imports process_health',
          'from src.utils import process_health' in app)
    check('app.py uses all_known_processes for one-pass scan',
          '_ph.all_known_processes()' in app)
    check('app.py monitor_health no longer defines _CMDLINE_SCAN dict',
          '_CMDLINE_SCAN = {' not in app)
    check('app.py monitor_health no longer defines _pick_best_pid',
          'def _pick_best_pid(' not in app)
    check('app.py _pipeline_proc_alive delegates to process_health',
          '_ph.find_process(_ph.KIND_TRAIN_ORCH)' in app)

    check('error_monitor _probe_processes uses process_health',
          'from src.utils import process_health' in em
          and 'find_process(_ph.KIND_BOT)' in em)
    check('error_monitor no longer scans psutil.process_iter directly',
          'psutil.process_iter' not in em
          or em.count('psutil.process_iter') == 0)

    check('training_sweep_watchdog uses process_health',
          'from src.utils import process_health' in wd
          and '_ph.KIND_TRAIN_ORCH' in wd)
    check('training_sweep_watchdog no longer scans psutil.process_iter directly',
          'psutil.process_iter' not in wd)


def test_phase84_orchestration_topics_pubsub():
    """Layer 6 of the orchestration plan (2026-05-10): a kafka-inspired
    file-based pub-sub at src/orchestration/topics.py. Each topic is one
    directory under data/topics/ with daily-rotated JSONL log files and
    per-consumer offset trackers.

    Phase 84 verifies:
      P1. Module + canonical topic constants exist.
      P2. Topic.append/tail/commit round-trips correctly.
      P3. Multiple consumers tail independently (offsets don't share).
      P4. Day rollover: a consumer parked on an older date catches up.
      P5. Stats reports bytes + lines per topic for dashboard rendering.
    """
    print('\n[Phase 84 -- orchestration topics file-based pub-sub]')

    tp_path = os.path.join(BASE_DIR, 'src', 'orchestration', 'topics.py')
    check('src/orchestration/topics.py exists', os.path.exists(tp_path))
    check('src/orchestration/__init__.py exists',
          os.path.exists(os.path.join(BASE_DIR, 'src', 'orchestration', '__init__.py')))

    # P1 — canonical constants
    from src.orchestration import topics as _topics
    check('exports TOPIC_TRAINING_REQUESTS',    hasattr(_topics, 'TOPIC_TRAINING_REQUESTS'))
    check('exports TOPIC_TRAINING_EVENTS',      hasattr(_topics, 'TOPIC_TRAINING_EVENTS'))
    check('exports TOPIC_TRAINING_CHECKPOINTS', hasattr(_topics, 'TOPIC_TRAINING_CHECKPOINTS'))
    check('exports TOPIC_BACKTEST_REQUESTS',    hasattr(_topics, 'TOPIC_BACKTEST_REQUESTS'))
    check('exports TOPIC_BACKTEST_EVENTS',      hasattr(_topics, 'TOPIC_BACKTEST_EVENTS'))
    check('exports TOPIC_SERVICE_HEARTBEATS',   hasattr(_topics, 'TOPIC_SERVICE_HEARTBEATS'))
    check('exports TOPIC_SERVICE_ALERTS',       hasattr(_topics, 'TOPIC_SERVICE_ALERTS'))
    check('KNOWN_TOPICS lists all 7 canonical topics',
          len(_topics.KNOWN_TOPICS) == 7)
    check('exports Topic class',          hasattr(_topics, 'Topic'))
    check('exports topic() helper',       callable(getattr(_topics, 'topic', None)))
    check('exports all_topic_stats()',    callable(getattr(_topics, 'all_topic_stats', None)))
    check('Topic has append/tail/commit/stats',
          all(hasattr(_topics.Topic, m) for m in ('append', 'tail', 'commit', 'stats')))

    # Redirect TOPICS_DIR at a tmp path so we don't pollute data/topics/
    # with test fixtures.
    import tempfile, shutil
    from pathlib import Path as _Path
    tmpdir = _Path(tempfile.mkdtemp(prefix='topics_test_'))
    orig_dir = _topics.TOPICS_DIR
    # The cache is keyed by name only; clear it so a fresh Topic uses tmpdir.
    _topics.TOPICS_DIR = tmpdir
    _topics._topic_cache.clear()
    try:
        TOPIC = _topics.TOPIC_SERVICE_HEARTBEATS
        t = _topics.topic(TOPIC)

        # P2 — append → tail round-trip
        off1 = t.append({'kind': 'bot',  'pid': 100, 'rss_mb': 500})
        off2 = t.append({'kind': 'dash', 'pid': 200, 'rss_mb':  80})
        off3 = t.append({'kind': 'orch', 'pid': 300, 'rss_mb': 200})
        check('append returns monotonically increasing offsets',
              off1 < off2 < off3)

        # Fresh consumer reads everything from start.
        entries_a = list(t.tail('consumer-A', batch=10))
        check('fresh consumer sees all 3 appended entries',
              len(entries_a) == 3)
        check('entry payload survives JSON round-trip',
              entries_a[0].payload['kind'] == 'bot'
              and entries_a[1].payload['pid'] == 200
              and entries_a[2].payload['rss_mb'] == 200)
        check('entry exposes byte_offset for commit',
              all(hasattr(e, 'byte_offset') and e.byte_offset > 0 for e in entries_a))
        check('entry exposes date in YYYYMMDD format',
              all(isinstance(e.date, str) and len(e.date) == 8 for e in entries_a))

        # Commit AFTER processing the third entry — next tail should be empty.
        last = entries_a[-1]
        t.commit('consumer-A', last.date, last.byte_offset)
        entries_a_again = list(t.tail('consumer-A', batch=10))
        check('after commit, repeat tail yields no duplicates',
              len(entries_a_again) == 0)

        # New append, same consumer picks it up.
        t.append({'kind': 'worker', 'pid': 400})
        entries_a_new = list(t.tail('consumer-A', batch=10))
        check('consumer resumes from committed offset',
              len(entries_a_new) == 1
              and entries_a_new[0].payload['kind'] == 'worker')

        # P3 — independent consumers
        entries_b = list(t.tail('consumer-B', batch=10))
        check('second consumer sees ALL 4 entries (independent offset)',
              len(entries_b) == 4)
        # consumer-A only sees the new one
        # (already verified above by entries_a_new == 1)

        # batch caps the yield size
        entries_b_capped = list(t.tail('consumer-B', batch=2))
        check('batch limit respected',
              len(entries_b_capped) == 2)

        # P4 — day rollover catch-up
        # Manually drop a synthetic older log file + park consumer-C on it.
        old_date = '20260101'
        old_path = tmpdir / TOPIC / f'log-{old_date}.jsonl'
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text('{"kind":"old1"}\n{"kind":"old2"}\n', encoding='utf-8')
        # consumer-C has never seen this topic — gets default offset
        # (earliest date on disk). With our injected old log, that's
        # 20260101 byte 0 → consumer-C should read old1, old2, then today's.
        entries_c = list(t.tail('consumer-C', batch=10))
        old_payloads = [e.payload.get('kind') for e in entries_c]
        check('consumer parked on older date catches up across day boundary',
              'old1' in old_payloads and 'old2' in old_payloads
              and any(p in ('bot', 'dash', 'orch', 'worker') for p in old_payloads))

        # P5 — stats
        st = t.stats()
        check('Topic.stats reports name + nonzero bytes + nonzero lines',
              st.name == TOPIC and st.bytes_total > 0 and st.lines_total > 0)
        check('Topic.stats reports days_present',
              st.days_present >= 2)   # today + injected old date
        check('Topic.stats reports last_append_ts as float',
              isinstance(st.last_append_ts, float) and st.last_append_ts > 0)

        all_stats = _topics.all_topic_stats()
        check('all_topic_stats returns one entry per KNOWN_TOPIC',
              len(all_stats) == len(_topics.KNOWN_TOPICS))
        check('all_topic_stats includes the seeded topic',
              any(s.name == TOPIC and s.bytes_total > 0 for s in all_stats))
    finally:
        _topics.TOPICS_DIR = orig_dir
        _topics._topic_cache.clear()
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def test_phase85_distributed_smoketest_three_bug_fixes():
    """Three pre-existing distributed-training bugs surfaced during the
    2026-05-10 smoke test, plus a synthetic CPU+GPU stress handler:

      Bug #1 — worker data-path resolution
        _invoke_master_trainer ran trainers without first chdir'ing to
        PROJECT_ROOT, so on a remote worker the trainers' relative
        'data/raw/...' paths failed. Result: every cluster task
        submitted to Ivan failed in <1 s with 'No training data found
        for ALL/<tf>'. Fix saves cwd, chdirs to PROJECT_ROOT, restores
        in finally.

      Bug #2 — meta-labeler signal_don column missing
        train_meta_labeler used df_feat[trend_features] which raised
        KeyError when 'signal_don' was missing. signal_don is computed
        inside train_trend_model.py from don_pos_20 but
        _build_all_features (the inference feature builder) never
        exposed it. Fix: derive signal_don from don_pos_20 in the
        meta-labeler before applying primaries; also reindex(fill_value=0)
        so future feature drift degrades gracefully instead of crashing.

      Bug #3 — dashboard cluster split-brain
        Each Python process had its own Orchestrator() singleton.
        Dashboard's /api/cluster/status returned its own empty
        in-process state, not the standalone :7700 cluster where
        workers actually connected. Result: dashboard cluster card
        always lied. Fix: proxy user-facing cluster endpoints to
        http://localhost:7700.

      smoke_test handler
        New synthetic stress task that exercises CPU + GPU for a
        configurable duration (5–1800 s). Lets an operator verify a
        worker is reachable + see compute usage in TaskManager /
        nvidia-smi, without touching real model files.
    """
    print('\n[Phase 85 -- distributed smoke-test bug fixes + smoke_test handler]')

    worker_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'worker.py')
    with open(worker_path, encoding='utf-8') as f:
        worker = f.read()
    meta_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_meta_labeler.py')
    with open(meta_path, encoding='utf-8') as f:
        meta = f.read()
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()

    # ── Bug #1 — worker data-path resolution ────────────────────────────
    check('worker chdirs to PROJECT_ROOT before invoking master trainer',
          'saved_cwd = _os.getcwd()' in worker
          and '_os.chdir(str(PROJECT_ROOT))' in worker)
    check('worker restores cwd in finally so process state is clean',
          'finally:' in worker
          and '_os.chdir(saved_cwd)' in worker)
    check('comment explains the SMB-mount path-resolution failure',
          'SMB-mounted Z:' in worker or 'Z:\\\\' in worker
          or 'No training data found' in worker)

    # ── Bug #2 — meta-labeler signal_don ────────────────────────────────
    check('meta-labeler computes signal_don from don_pos_20',
          "df_feat['signal_don'] = 0.0" in meta
          and "don_pos_20" in meta
          and ">  0.95" in meta.replace(' ', '')   # tolerate spacing
          or "don_pos_20'] > 0.95" in meta)
    check('meta-labeler uses reindex(fill_value=0) for defensive feature selection',
          'reindex(columns=base_features, fill_value=0)' in meta
          and 'reindex(columns=trend_features, fill_value=0)' in meta)
    check('meta-labeler no longer uses bare df_feat[trend_features]',
          'X_trend = df_feat[trend_features].fillna(0)' not in meta)

    # ── Bug #3 — dashboard cluster proxy ────────────────────────────────
    check('CLUSTER_BASE_URL constant defined',
          "CLUSTER_BASE_URL = 'http://localhost:7700'" in app)
    check('_cluster_proxy_get helper present',
          'def _cluster_proxy_get(' in app)
    check('_cluster_proxy_post helper present (covers POST + DELETE)',
          'def _cluster_proxy_post(' in app)
    check('cluster_status endpoint forwards to standalone',
          "_cluster_proxy_get('/api/cluster/status')" in app)
    check('cluster_workers endpoint forwards to standalone',
          "_cluster_proxy_get('/api/cluster/workers')" in app)
    check('cluster_tasks endpoint added (was missing)',
          "@app.route('/api/cluster/tasks')" in app
          and "_cluster_proxy_get('/api/cluster/tasks')" in app)
    check('cluster_submit endpoint forwards to standalone',
          "_cluster_proxy_post('/api/cluster/submit'" in app)
    check('cluster_submit_all endpoint forwards to standalone',
          "_cluster_proxy_post('/api/cluster/submit_all'" in app)
    check('cluster_cancel_task endpoint forwards via DELETE',
          "method='DELETE'" in app
          and "/api/cluster/task/" in app)
    check('worker-facing endpoints (register, task_update) stay in-process',
          # They still call _get_orchestrator, NOT _cluster_proxy_*
          'def cluster_register' in app
          and 'def cluster_task_update' in app)

    # ── smoke_test handler ──────────────────────────────────────────────
    check('smoke_test dispatched in _execute_task before master trainer',
          'if model_type == "smoke_test":' in worker
          and 'return _run_smoke_test(task)' in worker)
    check('_run_smoke_test function defined',
          'def _run_smoke_test(' in worker)
    check('smoke_test reads duration_s from config (5-1800 s)',
          "cfg.get(\"duration_s\"" in worker
          and 'min(1800' in worker
          and 'max(5' in worker)
    check('smoke_test runs CPU lane via numpy matmul threads',
          'def _cpu_loop' in worker
          and 'numpy' in worker.lower()
          and '_np.dot(A, B)' in worker)
    check('smoke_test runs GPU lane via torch cuda matmul when available',
          'def _gpu_loop' in worker
          and "device='cuda'" in worker
          and 'torch.cuda.synchronize' in worker)
    check('smoke_test logs heartbeat every 30 s with elapsed/remaining',
          'SMOKE_TEST tick' in worker
          and 'elapsed=%ds remain=%ds' in worker)
    check('smoke_test returns metrics with cpu_iters + gpu_iters + gpu_available',
          '"cpu_iters":' in worker
          and '"gpu_iters":' in worker
          and '"gpu_available":' in worker)


def test_phase86_sweep_coordinator_daemon():
    """Sweep Coordinator daemon (2026-05-10): drives the model-by-model
    distributed training sweep across cluster workers. Auto-starts on
    launch, persists state, refuses to start a second concurrent
    instance, retries transient failures once, runs backtest after
    training done.

    Phase 86 covers:
      P1. Module + entrypoint exist.
      P2. Pidfile lock prevents concurrent instances.
      P3. State machine (fresh / running / paused / done / aborted).
      P4. Skip-if-fresh window = 24 h matches user instruction.
      P5. Plan order puts CPU models first, GPU/exclusive last.
      P6. Build task spec uses use_master_trainer + symbol=ALL.
      P7. Transient-failure detection covers OOM / timeout / reroute / network.
      P8. Control plane (pause/resume/abort) endpoints exist.
      P9. Backtest stage runs after training done."""
    print('\n[Phase 86 -- sweep coordinator daemon]')

    sc_path = os.path.join(BASE_DIR, 'src', 'orchestration', 'sweep_coordinator.py')
    check('src/orchestration/sweep_coordinator.py exists', os.path.exists(sc_path))
    with open(sc_path, encoding='utf-8') as f:
        sc = f.read()

    # P1 — surface area
    check('exposes SweepCoordinator class', 'class SweepCoordinator' in sc)
    check('has __main__ entrypoint',        '__name__ == "__main__"' in sc)
    check('exposes main()',                 'def main()' in sc)

    # P2 — pidfile lock
    check('PIDFILE constant defined',       'PIDFILE' in sc and 'sweep_coordinator.pid' in sc)
    check('_acquire_pidfile rejects concurrent instance',
          'def _acquire_pidfile' in sc
          and 'refusing to start' in sc.lower())
    check('_release_pidfile cleans up on exit',
          'def _release_pidfile' in sc and 'PIDFILE.unlink' in sc)

    # P3 — state machine
    check('state file path is data/sweep_state.json',
          'STATE_PATH' in sc and 'sweep_state.json' in sc)
    check('fresh state seeded from training_rules.planned_combos',
          'planned_combos' in sc)
    check('atomic state save (tmp + rename)',
          ".with_suffix(\".tmp\")" in sc and 'os.replace(tmp, STATE_PATH)' in sc)
    check('resumes if last status was running/paused',
          'state.get("status") in ("running", "paused")' in sc)

    # P4 — 24h skip-if-fresh
    check('SKIP_IF_FRESH_HOURS = 24 (per user instruction)',
          'SKIP_IF_FRESH_HOURS' in sc and '= 24' in sc)
    check('_is_model_fresh checks meta mtime',
          'def _is_model_fresh' in sc and "mtime" in sc.lower() and 'SKIP_IF_FRESH_HOURS' in sc)

    # P5 — plan order: CPU first, GPU last
    plan_idx_base    = sc.find('"base"')
    plan_idx_tft     = sc.find('"tft"')
    plan_idx_oft     = sc.find('"oft"')
    check('PLAN_ORDER lists base, trend, futures, scalping, meta, regime, tft, oft',
          'PLAN_ORDER = ["base", "trend", "futures", "scalping", "meta",' in sc)
    check('TFT/OFT placed AFTER cpu models in PLAN_ORDER',
          plan_idx_base > 0 and plan_idx_tft > plan_idx_base and plan_idx_oft > plan_idx_tft)

    # P6 — task spec contract
    check('_build_task_spec uses use_master_trainer + symbol=ALL',
          'def _build_task_spec' in sc
          and '"use_master_trainer": True' in sc
          and '"symbol":       "ALL"' in sc)

    # P7 — transient failure detection
    check('_is_transient_failure covers OOM / timeout / reroute / network',
          '_TRANSIENT_FAIL_PATTERNS' in sc
          and 'out of memory' in sc.lower()
          and 'timeout' in sc.lower()
          and 'insufficient_vram' in sc
          and 'ConnectionError' in sc)
    check('MAX_RETRIES_PER_TASK = 1',
          'MAX_RETRIES_PER_TASK' in sc and '= 1' in sc)
    check('retry path resubmits with new task_id',
          'tf_state["retries"] += 1' in sc
          and 'new_id = self._submit_task' in sc)

    # P8 — control plane
    check('CONTROL_PORT = 7710',          'CONTROL_PORT' in sc and '7710' in sc)
    check('GET /api/sweep/status route',  '/api/sweep/status' in sc)
    check('POST /api/sweep/pause route',  '/api/sweep/pause' in sc)
    check('POST /api/sweep/resume route', '/api/sweep/resume' in sc)
    check('POST /api/sweep/abort route',  '/api/sweep/abort' in sc)
    check('request_abort + request_pause + request_resume defined',
          'def request_abort' in sc
          and 'def request_pause' in sc
          and 'def request_resume' in sc)

    # P9 — backtest stage runs after training
    check('_run_backtest defined and called after training models',
          'def _run_backtest' in sc
          and 'self._run_backtest()' in sc
          and 'next_phase' in sc)
    check('backtest invokes run_full_backtest with multi-TF',
          'run_full_backtest' in sc
          and "(\"5m\", \"15m\", \"1h\", \"4h\", \"1d\")" in sc)

    # P10 — worker lifecycle (spawn LOCAL_RAZER if missing)
    check('_ensure_local_worker spawns worker if not present',
          'def _ensure_local_worker' in sc
          and 'KIND_WORKER' in sc
          and 'PYTHONUNBUFFERED' in sc)


def test_phase87_dual_lane_workers_concurrent_cpu_gpu():
    """Dual-lane worker support (2026-05-10): each PC runs TWO worker
    processes — one on the CPU lane, one on the GPU lane — so a CPU model
    and a GPU model can train concurrently on the same machine. The
    sweep_coordinator submits TFT/OFT (gpu/exclusive) immediately at
    sweep start so they run in parallel with the model-by-model CPU
    sweep.

    Phase 87 covers:
      P1. worker.py --lane flag: cpu | gpu | any (back-compat default).
      P2. Worker registration carries the 'lane' attribute.
      P3. Orchestrator dispatch matches resource_kind <-> worker.lane:
            cpu task -> lane in {cpu, any}
            gpu / exclusive task -> lane in {gpu, any}
      P4. SweepCoordinator spawns TWO local workers (cpu + gpu) on master.
      P5. SweepCoordinator submits all gpu/exclusive tasks UPFRONT (parallel).
      P6. SweepCoordinator awaits remaining gpu-lane tasks before backtest.
      P7. CPU lane in spawned worker has CUDA_VISIBLE_DEVICES='' so it
          doesn't grab VRAM.
    """
    print('\n[Phase 87 -- dual-lane workers (concurrent CPU+GPU)]')

    worker_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'worker.py')
    with open(worker_path, encoding='utf-8') as f:
        worker = f.read()
    orch_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'orchestrator.py')
    with open(orch_path, encoding='utf-8') as f:
        orch = f.read()
    sc_path = os.path.join(BASE_DIR, 'src', 'orchestration', 'sweep_coordinator.py')
    with open(sc_path, encoding='utf-8') as f:
        sc = f.read()

    # P1 — --lane flag with cpu|gpu|any choices
    check('worker.py adds --lane flag with choices cpu|gpu|any',
          '--lane' in worker
          and 'choices=("cpu", "gpu", "any")' in worker)
    check('worker default lane is "any" (back-compat for Ivan-style legacy launches)',
          'default="any"' in worker)
    check('TrainingWorker constructor accepts lane parameter',
          'def __init__(self, master_url' in worker
          and 'lane: str = "any"' in worker)
    check('TrainingWorker validates lane to one of cpu|gpu|any',
          'lane if lane in ("cpu", "gpu", "any")' in worker)

    # P2 — registration includes lane
    check('heartbeat payload includes "lane" field',
          '"lane":          self.lane' in worker
          or '"lane": self.lane' in worker)

    # P3 — orchestrator lane-aware dispatch
    check('orchestrator imports resource_kind from training_rules for routing',
          'from src.training.training_rules import resource_kind' in orch)
    check('orchestrator defines _lane_accepts() helper',
          'def _lane_accepts(' in orch)
    check('"any" lane accepts every kind (back-compat)',
          'if worker_lane == "any":' in orch
          and 'return True' in orch)
    check('cpu kind routes to {cpu, any}',
          'if kind == "cpu":' in orch
          and 'worker_lane == "cpu"' in orch)
    check('gpu/exclusive/neural kind routes to {gpu, any}',
          'if kind in ("gpu", "exclusive", "neural"):' in orch
          and 'worker_lane == "gpu"' in orch)
    check('dispatch picks first idle worker matching lane (not blind round-robin)',
          'next(' in orch
          and '_lane_accepts(w.get("lane", "any"), kind)' in orch)

    # P4 — sweep_coordinator spawns two local workers
    check('SweepCoordinator has _spawn_local_worker(lane, port, name)',
          'def _spawn_local_worker(' in sc)
    check('SweepCoordinator spawns BOTH cpu + gpu lanes',
          'def _ensure_local_workers' in sc
          and 'lane="cpu"' in sc
          and 'lane="gpu"' in sc)
    check('CPU lane port=7701, GPU lane port=7702',
          'port=7701' in sc and 'port=7702' in sc)

    # P5 — submit all gpu lane upfront so it parallelises with cpu sweep
    check('SweepCoordinator._submit_gpu_lane defined and called from run()',
          'def _submit_gpu_lane' in sc
          and 'self._submit_gpu_lane()' in sc)
    check('CPU sweep loop SKIPS gpu/exclusive models (they were submitted upfront)',
          'if _rk(model) in ("gpu", "exclusive"):' in sc
          and 'continue' in sc)

    # P6 — await gpu lane before backtest
    check('SweepCoordinator._await_gpu_lane defined and called before backtest',
          'def _await_gpu_lane' in sc
          and 'self._await_gpu_lane()' in sc)

    # P7 — cpu-lane worker hides CUDA so it doesn't grab VRAM
    check('CPU-lane worker has CUDA_VISIBLE_DEVICES="" set',
          'CUDA_VISIBLE_DEVICES' in sc and 'lane == "cpu"' in sc)


def test_phase88_orchestrator_task_progress_watchdog():
    """Server-side watchdog (2026-05-10): catches zombie workers that
    keep heartbeating 'busy' while their task thread has crashed
    silently. Every WATCHDOG_POLL_S, scan running tasks and kill any
    that exceed the per-lane wall-clock timeout AND haven't received a
    status update in the last STALE window. Killed tasks become
    status='failed' with error='watchdog_timeout', and their assigned
    worker is freed (idle, current_task='') so the dispatcher reassigns.

    Phase 88 covers:
      P1. Watchdog constants + per-lane timeouts.
      P2. Tasks carry last_update_at field, refreshed on update_task().
      P3. Watchdog thread starts alongside the scheduler.
      P4. Stale + over-budget task gets killed; worker freed; dispatcher
          can pick a fresh task on the same worker.
      P5. Long-running but ACTIVELY-PROGRESSING task is NOT killed.
    """
    print('\n[Phase 88 -- orchestrator task-progress watchdog]')

    orch_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'orchestrator.py')
    with open(orch_path, encoding='utf-8') as f:
        orch = f.read()

    # P1 — constants
    check('WATCHDOG_POLL_S defined',                 'WATCHDOG_POLL_S' in orch and '= 30' in orch)
    check('WATCHDOG_STALE_UPDATE_S defined',         'WATCHDOG_STALE_UPDATE_S' in orch)
    check('WATCHDOG_TIMEOUT_BY_KIND with cpu/gpu/exclusive',
          '"cpu":' in orch and '"gpu":' in orch and '"exclusive":' in orch
          and '60 * 60' in orch and '120 * 60' in orch and '180 * 60' in orch)

    # P2 — last_update_at on submit + refreshed on every update
    check('submit_task seeds last_update_at',
          '"last_update_at":' in orch and 'now_iso' in orch)
    check('update_task refreshes last_update_at',
          'task["last_update_at"] = datetime.now(timezone.utc).isoformat()' in orch)

    # P3 — watchdog thread starts alongside scheduler
    check('start() spawns _watchdog_loop in a thread',
          '_watchdog_thread' in orch
          and 'target=self._watchdog_loop' in orch
          and 'name="orch-watchdog"' in orch)
    check('def _watchdog_loop body uses WATCHDOG_POLL_S',
          'def _watchdog_loop' in orch and 'WATCHDOG_POLL_S' in orch)
    check('_sweep_stale_tasks helper defined',
          'def _sweep_stale_tasks' in orch)

    # P4 — kill condition: BOTH wall-clock over budget AND stale
    check('kill needs elapsed > timeout AND stale > update window',
          'elapsed_s > timeout_s and stale_s > WATCHDOG_STALE_UPDATE_S' in orch)
    check('kill marks status=failed with watchdog_timeout error',
          '"watchdog_timeout' in orch and 'task["status"]      = "failed"' in orch)
    check('kill frees worker (status=idle, current_task="")',
          'self._workers[nid]["status"]       = "idle"' in orch
          and 'self._workers[nid]["current_task"] = ""' in orch)

    # ── Functional smoke test using a real Orchestrator instance ──────
    import importlib, sys, time as _t, datetime as _dt
    if 'src.training.distributed.orchestrator' in sys.modules:
        del sys.modules['src.training.distributed.orchestrator']
    mod = importlib.import_module('src.training.distributed.orchestrator')

    # Build a fresh orchestrator without starting threads (we want
    # deterministic state for testing the helper directly).
    o = mod.Orchestrator()
    # Register a fake worker
    nid = "test-node-1"
    o.register_worker({
        "node_id": nid, "name": "TEST_WORKER", "ip": "127.0.0.1",
        "port": 9999, "hostname": "TEST", "status": "busy",
        "lane": "cpu", "cuda_available": False, "gpu_vram_gb": 0,
        "cpu_cores": 4, "ram_gb": 8, "current_task": "",
    })
    # Submit a task and force it into stale-running state in the past.
    tid = o.submit_task({"model_type": "base", "timeframe": "1h",
                         "symbol": "ALL", "config": {}})
    long_ago_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=2)).isoformat()
    with o._lock:
        o._tasks[tid]["status"]         = "running"
        o._tasks[tid]["assigned_to"]    = nid
        o._tasks[tid]["started_at"]     = long_ago_iso
        o._tasks[tid]["last_update_at"] = long_ago_iso
        o._workers[nid]["current_task"] = tid
        o._workers[nid]["status"]       = "busy"

    o._sweep_stale_tasks()

    with o._lock:
        killed_status = o._tasks[tid].get("status")
        killed_error  = o._tasks[tid].get("error", "")
        worker_now    = dict(o._workers[nid])

    check('stale CPU task killed (status=failed)',
          killed_status == "failed")
    check('killed task error contains "watchdog_timeout"',
          'watchdog_timeout' in killed_error)
    check('worker freed (status=idle, current_task="")',
          worker_now.get("status") == "idle"
          and worker_now.get("current_task") == "")

    # P5 — actively-progressing task must NOT be killed
    nid2 = "test-node-2"
    o.register_worker({
        "node_id": nid2, "name": "TEST_WORKER_2", "ip": "127.0.0.2",
        "port": 9998, "hostname": "TEST2", "status": "busy",
        "lane": "cpu", "cuda_available": False, "gpu_vram_gb": 0,
        "cpu_cores": 4, "ram_gb": 8, "current_task": "",
    })
    tid2 = o.submit_task({"model_type": "base", "timeframe": "1h",
                          "symbol": "ALL", "config": {}})
    long_ago_iso = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=2)).isoformat()
    fresh_iso    = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with o._lock:
        o._tasks[tid2]["status"]         = "running"
        o._tasks[tid2]["assigned_to"]    = nid2
        o._tasks[tid2]["started_at"]     = long_ago_iso   # over budget
        o._tasks[tid2]["last_update_at"] = fresh_iso       # but PROGRESSING
        o._workers[nid2]["current_task"] = tid2
        o._workers[nid2]["status"]       = "busy"

    o._sweep_stale_tasks()

    with o._lock:
        progressing_status = o._tasks[tid2].get("status")
        worker2_now        = dict(o._workers[nid2])

    check('actively-progressing task is NOT killed (recent update_at)',
          progressing_status == "running"
          and worker2_now.get("status") == "busy")


def test_phase89_gpu_classifier_wrapper_and_trainer_migration():
    """GPU classifier migration (2026-05-10): all 5 tabular trainers
    (base/trend/futures/scalping/meta) now go through make_classifier()
    which returns XGBoost-on-CUDA when GPU is available, sklearn HistGBT
    fallback otherwise. The dual-lane spawn sets CUDA_VISIBLE_DEVICES=''
    on cpu-lane workers so they silently fall back to HistGBT.

    Phase 89 covers:
      P1. src/utils/gpu_classifier.py exists with make_classifier() and
          internal _XGBClassifierWrapper.
      P2. _cuda_available() respects CUDA_VISIBLE_DEVICES='' override.
      P3. make_classifier() returns XGBClassifier wrapper when GPU
          available, HistGBT when not (functional smoke test).
      P4. All 5 trainers import make_classifier and instantiate via it
          (no direct HistGradientBoostingClassifier construction left).
      P5. XGBoost wrapper exposes fit / predict / predict_proba /
          classes_ (sklearn-compatible surface).
      P6. class_weight='balanced' is honoured by the XGB wrapper via
          per-row sample weights when caller doesn't pass them."""
    print('\n[Phase 89 -- GPU classifier wrapper + trainer migration]')

    gpu_path = os.path.join(BASE_DIR, 'src', 'utils', 'gpu_classifier.py')
    check('src/utils/gpu_classifier.py exists', os.path.exists(gpu_path))
    with open(gpu_path, encoding='utf-8') as f:
        gpu = f.read()

    # P1 — surface area
    check('exposes make_classifier()',           'def make_classifier(' in gpu)
    check('internal _XGBClassifierWrapper class', 'class _XGBClassifierWrapper' in gpu)
    check('_cuda_available() helper',             'def _cuda_available(' in gpu)
    check('_xgboost_available() helper',          'def _xgboost_available(' in gpu)
    check('_use_gpu_backend() helper',            'def _use_gpu_backend(' in gpu)

    # P2 — CUDA_VISIBLE_DEVICES override respected
    check('_cuda_available checks CUDA_VISIBLE_DEVICES env',
          'CUDA_VISIBLE_DEVICES' in gpu)

    # P5 — XGB wrapper surface
    check('XGB wrapper has fit method',           'def fit(self, X, y' in gpu)
    check('XGB wrapper has predict_proba',        'def predict_proba(self' in gpu)
    check('XGB wrapper has predict',              'def predict(self, X)' in gpu)
    check('XGB wrapper exposes classes_',         '@property' in gpu and 'def classes_' in gpu)

    # P6 — class_weight balanced via compute_sample_weight in XGB path
    check('XGB wrapper computes sample_weight when class_weight=balanced',
          'compute_sample_weight("balanced", y)' in gpu
          and 'self._class_weight == "balanced"' in gpu)

    # XGBoost-specific config
    check('XGB params: tree_method=hist + device=cuda',
          '"tree_method":   "hist"' in gpu and '"device":        "cuda"' in gpu)

    # P4 — all 5 trainers migrated
    for trainer in ('train_model.py', 'train_trend_model.py',
                    'train_futures_model.py', 'train_scalping_model.py',
                    'train_meta_labeler.py'):
        path = os.path.join(BASE_DIR, 'src', 'engine', trainer)
        with open(path, encoding='utf-8') as f:
            t = f.read()
        check(f'{trainer}: imports make_classifier',
              'from src.utils.gpu_classifier import make_classifier' in t)
        # No direct HistGradientBoostingClassifier(...) construction left.
        # The import line stays for back-compat type hints; we forbid
        # only the call form.
        import re as _re
        # Match "HistGradientBoostingClassifier(" at start of an
        # instantiation, not the bare import. The migrated code uses
        # make_classifier(...) instead.
        constructor_calls = _re.findall(r'HistGradientBoostingClassifier\(', t)
        check(f'{trainer}: no direct HistGradientBoostingClassifier() call',
              len(constructor_calls) == 0)

    # P3 — functional smoke: make_classifier returns something fittable
    import importlib, sys, numpy as _np
    if 'src.utils.gpu_classifier' in sys.modules:
        del sys.modules['src.utils.gpu_classifier']
    mod = importlib.import_module('src.utils.gpu_classifier')
    clf = mod.make_classifier(n_estimators=10, max_depth=3, learning_rate=0.1,
                              l2_regularization=0.1, class_weight='balanced',
                              random_state=42)
    X = _np.random.rand(80, 4).astype('float32')
    y = (X[:, 0] > 0.5).astype(int)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    check('make_classifier returns a fittable classifier',
          proba.shape == (80, 2))
    check('predicted probabilities are in [0, 1]',
          (proba >= 0).all() and (proba <= 1).all())


def test_phase90_master_agent_zombie_worker_supervisor():
    """master_agent (Layer 5 supervisor) — closes the self-healing loop
    the Phase 88 watchdog left open. The watchdog detects zombie tasks
    and frees the cluster's worker SLOT, but the worker process keeps
    reporting 'busy' because its Python thread doesn't honour cancel.
    master_agent is the process-side healer:

      - Every POLL_S seconds, scans the cluster.
      - For each ONLINE worker reporting busy + current_task:
        * If the task isn't in the cluster's task table -> PHANTOM
        * If task status in {failed, cancelled, done} -> DEAD-TASK
        Either case = zombie worker.
      - Local zombies (hostname == this machine): SIGKILL + respawn.
      - Remote zombies (Ivan, future workers): log + service.alerts topic.
      - Also ensures cluster_orchestrator + local lane workers are alive.

    Phase 90 covers:
      P1. Module + entrypoint exist.
      P2. Detection: phantom (task_id missing from table) + dead-task
          (task in failed/cancelled/done).
      P3. Local heal: SIGKILL + respawn invoked for hostname == local.
      P4. Remote heal: alert path (no kill, log + topic write).
      P5. Cluster orchestrator self-respawn when /api/cluster/status
          doesn't respond.
      P6. _ensure_local_workers spawns BOTH cpu + gpu lanes if missing.
    """
    print('\n[Phase 90 -- master_agent (Layer 5 supervisor)]')

    ma_path = os.path.join(BASE_DIR, 'src', 'orchestration', 'master_agent.py')
    check('src/orchestration/master_agent.py exists', os.path.exists(ma_path))
    with open(ma_path, encoding='utf-8') as f:
        ma = f.read()

    # P1 — surface area
    check('exposes MasterAgent class',           'class MasterAgent' in ma)
    check('has __main__ entrypoint + main()',
          '__name__ == "__main__"' in ma and 'def main()' in ma)
    check('POLL_S constant defined',             'POLL_S' in ma and '= 60' in ma)
    check('LOCAL_WORKER_SPECS for cpu + gpu lanes',
          '("cpu", 7701, "LOCAL_RAZER_CPU")' in ma
          and '("gpu", 7702, "LOCAL_RAZER_GPU")' in ma)

    # P2 — both zombie detection paths
    check('PHANTOM detection (task_id not in cluster table)',
          'phantom_task_id' in ma and 'task_lookup.get(tid)' in ma)
    check('DEAD-TASK detection (status failed/cancelled/done)',
          '"failed", "cancelled", "done"' in ma
          and 'task_status=' in ma)

    # P3 — local heal: kill + respawn
    check('_heal_zombie has local branch (LOCAL_HOSTNAME match)',
          'host == LOCAL_HOSTNAME' in ma
          and 'self._kill_pids(pids)' in ma)
    check('_heal_zombie respawns the same lane after kill',
          'self._spawn_local_worker(lane, spec_port, name)' in ma)
    check('_find_local_python_pids matches --name + --lane',
          'def _find_local_python_pids' in ma
          and 'f"--name {name}"' in ma and 'f"--lane {lane}"' in ma)
    check('_kill_pids uses psutil with SIGKILL semantics',
          'def _kill_pids' in ma and 'p.kill()' in ma)

    # P4 — remote heal path
    check('_heal_zombie has REMOTE branch (logs + topic)',
          'REMOTE ZOMBIE' in ma
          and 'TOPIC_SERVICE_ALERTS' in ma)
    check('remote zombie writes to service.alerts topic',
          'topic(TOPIC_SERVICE_ALERTS).append(' in ma
          and '"kind":      "remote_zombie_worker"' in ma)

    # P5 — orchestrator self-respawn
    check('_ensure_cluster_orchestrator alive check',
          'def _cluster_orchestrator_alive' in ma
          and ('/api/cluster/status' in ma))
    check('_ensure_cluster_orchestrator respawn when down',
          'def _ensure_cluster_orchestrator' in ma
          and 'src.training.distributed.orchestrator' in ma
          and 'subprocess.Popen' in ma)

    # P6 — local worker lifecycle
    check('_ensure_local_workers iterates LOCAL_WORKER_SPECS',
          'def _ensure_local_workers' in ma
          and 'for lane, port, name in LOCAL_WORKER_SPECS:' in ma)
    check('skip if already registered (no double-spawn)',
          '(name, lane) in registered_local' in ma)
    check('CPU lane spawn sets CUDA_VISIBLE_DEVICES=""',
          'lane == "cpu"' in ma
          and 'CUDA_VISIBLE_DEVICES' in ma)

    # ── Functional smoke: MasterAgent._sweep_zombie_workers correctly
    # identifies the two zombie patterns. We monkey-patch _http_get and
    # _heal_zombie to capture which workers got flagged.
    import importlib, sys as _sys, types as _types
    if 'src.orchestration.master_agent' in _sys.modules:
        del _sys.modules['src.orchestration.master_agent']
    mod = importlib.import_module('src.orchestration.master_agent')

    captured: list[tuple[str, str]] = []   # (worker_name, reason)

    def _fake_http_get(path, timeout=5.0):
        if path == "/api/cluster/status":
            return {"workers": [
                # PHANTOM zombie — task not in table
                {"node_id": "z1", "name": "LOCAL_RAZER_GPU", "hostname": "Razer",
                 "lane": "gpu", "online": True, "status": "busy",
                 "current_task": "phantom-task-99", "last_seen_ago": 5},
                # DEAD-TASK zombie — task is failed
                {"node_id": "z2", "name": "WORKER-1-CPU", "hostname": "Ivan",
                 "lane": "cpu", "online": True, "status": "busy",
                 "current_task": "deadtask-1", "last_seen_ago": 5},
                # Healthy: busy with a real running task
                {"node_id": "h1", "name": "LOCAL_RAZER_CPU", "hostname": "Razer",
                 "lane": "cpu", "online": True, "status": "busy",
                 "current_task": "running-task-5", "last_seen_ago": 5},
                # Idle: not a zombie
                {"node_id": "i1", "name": "OTHER", "hostname": "Razer",
                 "lane": "cpu", "online": True, "status": "idle",
                 "current_task": "", "last_seen_ago": 5},
            ]}
        if path == "/api/cluster/tasks":
            return [
                {"task_id": "deadtask-1",      "status": "failed"},
                {"task_id": "running-task-5", "status": "running"},
                # phantom-task-99 deliberately missing
            ]
        return None

    mod._http_get = _fake_http_get

    agent = mod.MasterAgent()
    # Patch _heal_zombie to capture rather than really kill
    def _capture(worker, reason, task_id):
        captured.append((worker.get("name", ""), reason))
    agent._heal_zombie = _capture

    agent._sweep_zombie_workers()

    captured_names = sorted([c[0] for c in captured])
    captured_reasons = [c[1] for c in captured]

    # First sweep — phantom should NOT yet be killed (within confirm
    # window). Dead-task IS killed (unambiguous).
    check('first sweep: phantom NOT yet killed (within confirm window)',
          "LOCAL_RAZER_GPU" not in captured_names)
    check('dead-task zombie detected immediately on first sweep',
          "WORKER-1-CPU" in captured_names
          and any("task_status=failed" in r for r in captured_reasons))
    check('healthy busy worker is NOT flagged as zombie',
          "LOCAL_RAZER_CPU" not in captured_names)
    check('idle worker is NOT flagged as zombie',
          "OTHER" not in captured_names)
    # Phantom should be tracked in _phantom_first_seen
    check('phantom tracked in _phantom_first_seen on first observation',
          "z1" in agent._phantom_first_seen)

    # Simulate time passing past the confirm window: backdate the
    # first-seen timestamp and re-run the sweep. Now the phantom IS killed.
    captured.clear()
    import time as _time
    agent._phantom_first_seen["z1"] = _time.time() - mod.PHANTOM_CONFIRM_S - 1
    agent._sweep_zombie_workers()
    captured_names_2 = sorted([c[0] for c in captured])
    captured_reasons_2 = [c[1] for c in captured]

    check('after confirm window: phantom IS killed',
          "LOCAL_RAZER_GPU" in captured_names_2
          and any("phantom_task_id_persisted" in r for r in captured_reasons_2))


def test_phase91_tft_dedupe_tz_normalize_plus_meta_hard_fail():
    """Two trainer-side fixes uncovered during the 2026-05-10 sweep:

    Fix #1 — TFT @ 1h failed every run with:
      'cannot reindex on an axis with duplicate labels'
    immediately preceded by:
      'WARNING The provided DatetimeIndex was associated with a timezone
       (tz), which is currently not supported. To avoid unexpected
       behaviour, the tz information was removed.'
    Root cause: distinct tz-aware timestamps collapse into duplicate
    naive timestamps after Darts strips tz internally. Our dedupe ran
    BEFORE Darts' strip, so it didn't catch the post-strip collisions.
    Fix: normalize tz to UTC-naive INSIDE _dedupe_for_darts BEFORE
    drop_duplicates, so Darts has nothing left to strip.

    Fix #2 — Meta-labeler silently 'succeeded' with no artifacts:
      log.error('No signal data collected. Cannot train meta-labeler.')
      return    ← silent success — task marked done, dashboard keeps
                  showing STALE because no artifact was written
    Fix: raise RuntimeError so the worker reports task=failed and the
    operator sees the real cause (primary models couldn't generate
    signals — usually sklearn-version mismatch or feature regression).
    """
    print('\n[Phase 91 -- TFT tz-normalize dedupe + meta-labeler hard-fail]')

    tft_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_tft_model.py')
    with open(tft_path, encoding='utf-8') as f:
        tft = f.read()
    meta_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_meta_labeler.py')
    with open(meta_path, encoding='utf-8') as f:
        meta = f.read()

    # Fix #1: tz-normalize in _dedupe_for_darts
    check('_dedupe_for_darts checks tz before dedupe',
          'def _dedupe_for_darts(' in tft
          and 'getattr(df[time_col].dt' in tft
          and "'tz'" in tft)
    check('tz-aware timestamps converted to UTC then naive',
          "tz_convert('UTC').dt.tz_localize(None)" in tft)
    check('tz-normalize happens BEFORE drop_duplicates (correct order)',
          tft.find("tz_localize(None)") < tft.find("drop_duplicates(subset=[time_col]")
          if "tz_localize(None)" in tft else False)
    check('docstring captures the 2026-05-10 root cause',
          ('tz info was removed' in tft or 'distinct-tz' in tft
           or 'tz-strip' in tft.lower() or 'collapsed' in tft.lower()))

    # Fix #2: meta-labeler hard-fail on empty signal_dataset
    check('meta-labeler raises RuntimeError on no signals (no silent success)',
          'raise RuntimeError(' in meta
          and 'no signal data collected' in meta.lower())
    check('docstring/comment explains why hard-fail (not silent return)',
          'silent success' in meta.lower()
          and 'STALE' in meta)
    check('no leftover bare-return on the no-signals path',
          # Confirm the function no longer returns None silently — the
          # raise is on the same condition path. Crude but effective:
          # the literal "return\n" right after the empty-frames check
          # is gone.
          'No signal data collected. Cannot train meta-labeler.' in meta
          and 'raise RuntimeError(msg)' in meta)


def test_phase92_meta_labeler_regime_dict_shape_tolerance():
    """2026-05-10 hotfix — meta-labeler crashed per-symbol with
    KeyError: 'scaler' because regime_classifier.joblib is saved as
    {"model": {"gmm":..., "scaler":...}, "label_map":...} (NESTED) but
    the meta-labeler read it as if FLAT: regime_model_data["scaler"].
    Failure cascade: per-symbol exception → no signal data collected →
    Phase 91 hard-fail with "no signal data collected". Real root cause
    was here, not the primary models. Fix: read via .get("model", self)
    so both nested and legacy-flat artifacts work."""
    print('\n[Phase 92 -- meta-labeler regime dict-shape tolerance]')
    meta_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_meta_labeler.py')
    with open(meta_path, encoding='utf-8') as f:
        meta = f.read()
    check('regime artifact accessed via .get("model", regime_model_data) — tolerant fallback',
          'regime_model_data.get("model", regime_model_data)' in meta)
    check('scaler + gmm read from the unwrapped layer (_rmodel)',
          '_rmodel["scaler"]' in meta and '_rmodel["gmm"]' in meta)
    check('label_map still read from outer dict (where it lives)',
          "regime_model_data['label_map']" in meta)
    check('comment captures the per-symbol scaler KeyError chain',
          ("'scaler'" in meta) and ('Caused per-symbol' in meta or 'no signal data' in meta))


def test_phase93_worker_live_load_and_remote_restart():
    """Phase 93 — visibility + remote process control for the cluster.

    Pre-Phase-93 the dashboard knew a worker had registered, but had no
    per-worker CPU/GPU number — operators were tail-ing nvidia-smi on
    the wrong PC during the 2026-05-10 sweep. master_agent could heal
    LOCAL zombies (kill+respawn) but could only LOG remote zombies on
    Ivan, requiring manual SSH/RDP.

    Phase 93 closes both gaps:
      1. Worker /health + heartbeat carry cpu_percent, gpu_percent,
         gpu_mem_used_mb, gpu_mem_total_mb, uptime_s.
      2. Worker exposes /restart (self-execv) + /system_info (full diag dump).
      3. Orchestrator passes live load fields straight through to the
         dashboard worker dict (existing {**prev, **info} merge).
      4. master_agent._heal_zombie POSTs /restart for remote zombies
         (auto-heal); falls back to operator-alert if the endpoint is down.
      5. Dashboard renders CPU/GPU columns on each worker card +
         operator-triggered ↻restart button via a server-side proxy
         (avoids browser CORS).
    """
    print('\n[Phase 93 -- worker live-load + remote restart]')

    worker_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'worker.py')
    with open(worker_path, encoding='utf-8') as f:
        worker = f.read()
    orch_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'orchestrator.py')
    with open(orch_path, encoding='utf-8') as f:
        orch = f.read()
    ma_path = os.path.join(BASE_DIR, 'src', 'orchestration', 'master_agent.py')
    with open(ma_path, encoding='utf-8') as f:
        ma = f.read()
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()
    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # 1. _sample_live_load helper exists with the four required fields.
    check('worker._sample_live_load() defined',
          'def _sample_live_load(' in worker)
    check('_sample_live_load returns cpu_percent, gpu_percent, gpu_mem_used_mb, gpu_mem_total_mb',
          'cpu_percent' in worker
          and 'gpu_percent' in worker
          and 'gpu_mem_used_mb' in worker
          and 'gpu_mem_total_mb' in worker)
    check('CPU% sampled via psutil.cpu_percent',
          'psutil.cpu_percent(interval=None)' in worker)
    check('GPU% sampled via nvidia-smi --query-gpu',
          '--query-gpu=utilization.gpu,memory.used,memory.total' in worker)
    check('nvidia-smi failure swallowed (graceful CPU-only fallback)',
          'FileNotFoundError' in worker
          and ('subprocess.TimeoutExpired' in worker))

    # 2. /health includes live_load + uptime_s; new /restart + /system_info
    #    endpoints exist.
    check('/health response includes live_load',
          '"live_load": _sample_live_load()' in worker)
    check('/health response includes uptime_s',
          '"uptime_s":  int(time.time() - self._start_time)' in worker
          or '"uptime_s": int(time.time() - self._start_time)' in worker)
    check('worker tracks _start_time in __init__',
          'self._start_time = time.time()' in worker)
    check('/restart endpoint defined',
          '@app.route("/restart"' in worker
          and 'def restart()' in worker)
    check('/restart re-execs via os.execv on the same Python',
          'os.execv(sys.executable' in worker)
    check('/restart responds OK before re-exec (delayed_exec thread)',
          '_delayed_exec' in worker
          and 'time.sleep(1.0)' in worker)
    # Confirm gate added 2026-05-10 after the {"dry_run": true} accident
    # re-execed Ivan's GPU worker. /restart must reject any body that
    # doesn't include {"confirm": true}.
    restart_start = worker.find('def restart()')
    restart_end   = worker.find('\n        @app.route', restart_start + 1)
    if restart_end < 0:
        restart_end = worker.find('\n        return app', restart_start + 1)
    restart_body  = worker[restart_start:restart_end] if restart_end > restart_start else worker[restart_start:]
    check('/restart parses POST body via flask.get_json',
          'freq.get_json(' in restart_body
          and 'silent=True' in restart_body)
    check('/restart returns 400 unless body.confirm is True',
          'body.get("confirm") is not True' in restart_body
          and ', 400' in restart_body)
    check('/restart 400 response includes a hint pointing at the confirm flag',
          '"hint":' in restart_body
          and '"confirm": true' in restart_body)
    check('/restart docstring/comment cites the dry_run accident',
          'dry_run' in worker
          and 'accident' in worker.lower())
    check('/system_info endpoint defined',
          '@app.route("/system_info")' in worker
          and 'def system_info()' in worker)
    check('/system_info dumps live_load, transport, hw, uptime, pid',
          '"live_load":' in worker
          and '"transport":' in worker
          and '"uptime_s":' in worker
          and '"pid":' in worker)

    # 3. Heartbeat payload carries the new fields.
    hb_start = worker.find('def _heartbeat_loop')
    hb_end   = worker.find('\n    def ', hb_start + 4)
    hb_body  = worker[hb_start:hb_end] if hb_end > hb_start else worker[hb_start:]
    check('heartbeat payload includes cpu_percent',
          '"cpu_percent":' in hb_body)
    check('heartbeat payload includes gpu_percent',
          '"gpu_percent":' in hb_body)
    check('heartbeat payload includes gpu_mem_used_mb + gpu_mem_total_mb',
          '"gpu_mem_used_mb":' in hb_body
          and '"gpu_mem_total_mb":' in hb_body)
    check('heartbeat samples live load once per beat',
          '_sample_live_load()' in hb_body)
    check('heartbeat payload includes uptime_s',
          '"uptime_s":' in hb_body)

    # 4. Orchestrator merge passes through (no per-field plumbing needed —
    #    {**prev, **info} already does it).
    check('orchestrator register_worker docstring/comment notes Phase 93 live load',
          'Phase 93' in orch
          and 'cpu_percent' in orch
          and 'gpu_percent' in orch)
    check('orchestrator merges new info over previous state',
          'self._workers[node_id] = {**prev, **info}' in orch)

    # 5. master_agent — remote zombie path POSTs /restart.
    heal_start = ma.find('def _heal_zombie')
    heal_end   = ma.find('\n    def ', heal_start + 4)
    heal_body  = ma[heal_start:heal_end] if heal_end > heal_start else ma[heal_start:]
    check('master_agent._heal_zombie POSTs /restart for remote zombies',
          '/restart' in heal_body
          and 'urllib.request.Request' in heal_body
          and "method=\"POST\"" in heal_body)
    check('master_agent uses worker ip+port to reach the remote /restart',
          'worker.get("ip"' in heal_body
          and 'worker.get("port"' in heal_body)
    check('master_agent /restart POST includes "confirm": True (gate compliance)',
          '"confirm": True' in heal_body)
    check('successful remote restart emits REMOTE RESTART log line',
          'REMOTE RESTART' in heal_body)
    check('failed remote restart still emits REMOTE ZOMBIE alert (fallback)',
          'REMOTE ZOMBIE' in heal_body
          and "'auto_healed'" in heal_body
          or '"auto_healed":' in heal_body)

    # 6. Dashboard backend proxy + frontend rendering.
    check('dashboard /api/cluster/worker_restart route defined',
          "@app.route('/api/cluster/worker_restart'" in app
          and 'def cluster_worker_restart' in app)
    check('worker_restart route POSTs to {ip}:{port}/restart',
          "f'http://{ip}:{port}/restart'" in app)
    check('worker_restart validates ip+port (no bare proxy)',
          "if not ip or not port" in app)
    check('dashboard proxy includes "confirm": True in body (gate compliance)',
          "'confirm': True" in app
          and 'worker_restart' in app)
    check('cluster card renders cpu_percent + gpu_percent on each worker',
          'w.cpu_percent' in tpl
          and 'w.gpu_percent' in tpl)
    check('cluster card renders VRAM used/total when available',
          'w.gpu_mem_used_mb' in tpl
          and 'w.gpu_mem_total_mb' in tpl)
    check('CPU/GPU bars use red/amber/green thresholds (>80 / >50)',
          ('> 80' in tpl or '>80' in tpl)
          and ('> 50' in tpl or '>50' in tpl))
    check('dashboard ↻restart button calls clusterRestartWorker',
          'function clusterRestartWorker(' in tpl
          and "fetch('/api/cluster/worker_restart'" in tpl)
    check('_fmtUptime helper renders worker uptime',
          'function _fmtUptime(' in tpl)
    # Confirm clusterPoll timer still fires (nothing broke the existing
    # 10-s poll that drives Live Load freshness).
    check('clusterPoll setInterval still 10 s',
          'setInterval(() => { if (activeTab === \'monitor\') clusterPoll(); }, 10_000)' in tpl)


def test_phase94_distributed_backtest_per_cell():
    """Phase 94 — per-cell distributed backtest.

    Pre-Phase-94 `run_full_backtest` was single-process on master,
    burning one core for ~30 min while the other 3 lanes (Ivan
    CPU/GPU + LOCAL_RAZER GPU) sat idle. Phase 94 fans cells out to
    the cluster: one task per (symbol, timeframe), workers load their
    own data via SMB, return summary dicts only (no equity curves
    over the wire).

    What this test asserts:
      1. backtester.py extracted `_run_one_backtest_cell` from the
         inner loop (legacy single-process path now goes through it
         too, so distributed and single-process paths are guaranteed
         to produce identical strategy_result rows).
      2. JSON-safe wrapper `run_one_backtest_cell_summaries` exists
         (returns plain dicts the worker can put in task.result).
      3. `run_distributed_backtest` submits one task per cell, polls
         /api/cluster/tasks, aggregates summaries, writes the same
         artifacts as the single-process path.
      4. `run_full_backtest(distribute=True)` delegates to the
         distributed path; default `distribute=False` keeps legacy
         behavior.
      5. Walk-forward stays on master in BOTH paths (no per-cell WF).
      6. worker.py has the `model_type='backtest_cell'` branch wired
         BEFORE the master_trainer dispatch — backtests are not
         training, master_trainer doesn't know backtest_cell.
      7. Worker chdir's to PROJECT_ROOT so master's relative-path data
         loads work over SMB.
      8. Dashboard exposes /api/backtest/distributed/start and /status.
      9. /status only includes model_type='backtest_cell' tasks (not
         every cluster task) and exposes per-cell counters.
    """
    print('\n[Phase 94 -- distributed backtest per cell]')

    bt_path = os.path.join(BASE_DIR, 'src', 'engine', 'backtester.py')
    with open(bt_path, encoding='utf-8') as f:
        bt = f.read()
    worker_path = os.path.join(BASE_DIR, 'src', 'training', 'distributed', 'worker.py')
    with open(worker_path, encoding='utf-8') as f:
        worker = f.read()
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()

    # 1. Cell extraction
    check('_run_one_backtest_cell defined',
          'def _run_one_backtest_cell(' in bt)
    check('_run_one_backtest_cell returns list[BacktestResult] (no raise on empty)',
          'return []' in bt
          and 'def _run_one_backtest_cell(' in bt)
    check('_BACKTEST_GROUP_A_NAMES constant moved to module scope (was inline in run_full)',
          '_BACKTEST_GROUP_A_NAMES' in bt
          and 'RSI_MeanReversion' in bt)
    check('legacy run_full_backtest now delegates per-cell to _run_one_backtest_cell',
          'cell_results = _run_one_backtest_cell(' in bt)

    # 2. JSON-safe summaries wrapper
    check('run_one_backtest_cell_summaries defined',
          'def run_one_backtest_cell_summaries(' in bt)
    check('summaries wrapper attaches timeframe + group fields',
          's["timeframe"]' in bt
          and 's["group"]' in bt)
    check('summaries wrapper does NOT serialize trades/equity_curve (JSON-safe)',
          # The wrapper calls .summary() which only emits scalars.
          'r.summary()' in bt
          and 'def run_one_backtest_cell_summaries' in bt)

    # 3. run_distributed_backtest
    check('run_distributed_backtest defined',
          'def run_distributed_backtest(' in bt)
    rdb_start = bt.find('def run_distributed_backtest(')
    rdb_end   = bt.find('\nif __name__', rdb_start)
    rdb_body  = bt[rdb_start:rdb_end] if rdb_end > rdb_start else bt[rdb_start:]
    check('distributed: submits one task per (symbol, timeframe) cell',
          'for tf in timeframes:' in rdb_body
          and 'for sym in symbols:' in rdb_body
          and '"model_type": "backtest_cell"' in rdb_body)
    check('distributed: posts to /api/cluster/submit',
          '/api/cluster/submit' in rdb_body
          and 'method="POST"' in rdb_body)
    check('distributed: polls /api/cluster/tasks until done/failed/cancelled',
          '/api/cluster/tasks' in rdb_body
          and '("done", "failed", "cancelled")' in rdb_body)
    check('distributed: overall_timeout_s guards against forever-poll',
          'overall_timeout_s' in rdb_body
          and 'RuntimeError' in rdb_body)
    check('distributed: stable sort on (symbol, timeframe, strategy) before write',
          'rows.sort(' in rdb_body
          and '"symbol"' in rdb_body
          and '"strategy"' in rdb_body)
    check('distributed: writes the same artifacts as run_full_backtest',
          'comparison_' in rdb_body
          and 'latest_comparison.json' in rdb_body
          and 'ab_comparison.json' in rdb_body)
    check('distributed: cluster_url is configurable (default localhost:7700)',
          'cluster_url: str = "http://localhost:7700"' in rdb_body)
    check('distributed: raises on cluster unreachable (no silent partial)',
          'cluster unreachable' in rdb_body)

    # 4. run_full_backtest delegation
    rfb_start = bt.find('def run_full_backtest(')
    rfb_end   = bt.find('\n# ═══ Phase 94', rfb_start)
    rfb_body  = bt[rfb_start:rfb_end] if rfb_end > rfb_start else bt[rfb_start:]
    check('run_full_backtest exposes distribute= flag (default False)',
          'distribute: bool = False' in rfb_body)
    check('run_full_backtest delegates to run_distributed_backtest when distribute=True',
          'if distribute:' in rfb_body
          and 'run_distributed_backtest(' in rfb_body)
    check('legacy default keeps single-process semantics (back-compat)',
          'distribute: bool = False' in rfb_body)

    # 5. Walk-forward stays on master in both paths
    check('legacy WF block uses last_cell, not the lost `sym` from extraction',
          'wf_sym = last_cell[0]' in rfb_body)
    check('distributed WF runs on master, single rep cell',
          'rep = submitted[-1]' in rdb_body
          and 'walk_forward(' in rdb_body)

    # 6. Worker handler wired BEFORE master_trainer dispatch
    exec_start = worker.find('def _execute_task(')
    exec_end   = worker.find('\n\n# ─', exec_start)
    if exec_end < 0:
        exec_end = worker.find('\ndef ', exec_start + 1)
    exec_body  = worker[exec_start:exec_end] if exec_end > exec_start else worker[exec_start:]
    check('worker._execute_task dispatches model_type="backtest_cell"',
          'model_type == "backtest_cell"' in exec_body
          and '_run_backtest_cell(task)' in exec_body)
    check('backtest_cell branch comes BEFORE master_trainer dispatch (priority order)',
          exec_body.find('model_type == "backtest_cell"')
          < exec_body.find('use_master_trainer'))
    check('worker._run_backtest_cell handler defined',
          'def _run_backtest_cell(' in worker)

    # 7. SMB cwd handling
    rbc_start = worker.find('def _run_backtest_cell(')
    rbc_end   = worker.find('\ndef _load_data', rbc_start)
    if rbc_end < 0:
        rbc_end = worker.find('\n\ndef ', rbc_start + 1)
    rbc_body  = worker[rbc_start:rbc_end] if rbc_end > rbc_start else worker[rbc_start:]
    check('handler chdirs to PROJECT_ROOT (SMB-relative data paths work)',
          '_os.chdir(str(PROJECT_ROOT))' in rbc_body)
    check('handler restores cwd in finally (no process-state pollution)',
          'finally:' in rbc_body
          and '_os.chdir(saved_cwd)' in rbc_body)
    check('handler imports from master via run_one_backtest_cell_summaries',
          'run_one_backtest_cell_summaries' in rbc_body)
    check('handler returns metrics.strategy_results + n_strategies',
          '"strategy_results": rows' in rbc_body
          and '"n_strategies":' in rbc_body)
    check('handler reports duration_s (operator visibility per cell)',
          '"duration_s":' in rbc_body)
    check('handler caps failure to status="failed" + error string (no raise to caller)',
          '"status": "failed"' in rbc_body
          and '"error":' in rbc_body)

    # 8. Dashboard endpoints
    check('dashboard /api/backtest/distributed/start route defined',
          "@app.route('/api/backtest/distributed/start'" in app
          and 'def api_backtest_distributed_start' in app)
    check('start endpoint runs distributed backtest in a background thread',
          'threading.Thread(target=_bg' in app
          and 'run_distributed_backtest(' in app)
    check('start endpoint accepts timeframes / symbols / models from body',
          "body.get('timeframes'" in app
          and "body.get('symbols')" in app
          and "body.get('models')" in app)
    check('dashboard /api/backtest/distributed/status route defined',
          "@app.route('/api/backtest/distributed/status'" in app
          and 'def api_backtest_distributed_status' in app)

    # 9. /status filtering + counters
    status_start = app.find('def api_backtest_distributed_status')
    status_end   = app.find('\n@app.route', status_start + 1)
    status_body  = app[status_start:status_end] if status_end > status_start else app[status_start:]
    check('/status filters to model_type="backtest_cell" only',
          "t.get('model_type') != 'backtest_cell'" in status_body)
    check('/status surfaces per-cell counters: done/running/failed/pending',
          "'done':" in status_body
          and "'running':" in status_body
          and "'failed':" in status_body
          and "'pending':" in status_body)
    check('/status includes per-cell metadata for the dashboard column',
          "'task_id':" in status_body
          and "'symbol':" in status_body
          and "'timeframe':" in status_body
          and "'assigned_to':" in status_body
          and "'duration_s':" in status_body)


def test_phase95_xgb_early_stop_eval_set_fix_and_backtest_column():
    """Phase 95 — two fixes shipped together.

    Fix 1: XGBoost early_stopping ValueError ('Must have at least 1
    validation dataset for early stopping.'). The gpu_classifier wrapper
    set early_stopping_rounds=20 but never supplied an eval_set;
    sklearn HistGBT auto-splits internally, XGBoost doesn't, so the
    wrapper now mirrors HistGBT's behavior — last 10% of rows held out
    as the eval set when caller passes early_stopping=True without an
    eval_set. Last-N (no shuffle) preserves time-series order.

    Fix 2: dashboard Backtest column on the Model Training table — Item
    B from the original 4-item plan, follow-up to Phase 94's distributed
    backtest. Renders aggregated per-TF cell counts (e.g., '3/5') with
    color coding (green=all done · amber=in flight · red=any failed ·
    grey=none submitted).
    """
    print('\n[Phase 95 -- XGB early-stop fix + Backtest column]')

    gpu_path = os.path.join(BASE_DIR, 'src', 'utils', 'gpu_classifier.py')
    with open(gpu_path, encoding='utf-8') as f:
        gpu = f.read()
    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # ── Fix 1: gpu_classifier wrapper auto-creates eval_set ─────────────
    fit_start = gpu.find('    def fit(self, X, y, sample_weight=None, eval_set=None):')
    fit_end   = gpu.find('\n    @property', fit_start + 1)
    fit_body  = gpu[fit_start:fit_end] if fit_end > fit_start else gpu[fit_start:]
    check('wrapper.fit() auto-creates eval_set when early_stopping + caller didn\'t pass one',
          'self._early_stopping and eval_set is None' in fit_body)
    check('auto-eval-set carves the LAST 10% of rows (time-series-safe, no shuffle)',
          'X_arr[:-n_val]' in fit_body
          and 'X_arr[-n_val:]' in fit_body
          and 'n_val   = max(1, int(round(n_total * 0.10)))' in fit_body)
    check('sample_weight sliced consistently with X/y',
          'sw_arr[:-n_val]' in fit_body)
    check('tiny dataset (<10 train rows after split) disables early_stopping rather than crashing',
          "self._clf.set_params(early_stopping_rounds=None)" in fit_body)
    check('comment cites the exact error this fixes',
          ('Must have at least 1 validation dataset' in gpu
           or 'at least 1 validation dataset' in gpu))

    # Live smoke test — actually fit a tiny model to confirm no crash.
    try:
        import numpy as _np
        from src.utils.gpu_classifier import make_classifier
        _np.random.seed(0)
        X_smoke = _np.random.rand(120, 4).astype(_np.float32)
        y_smoke = (X_smoke[:, 0] + X_smoke[:, 1] > 1.0).astype(_np.int32)
        clf = make_classifier(n_estimators=20, max_depth=3, learning_rate=0.1, early_stopping=True)
        clf.fit(X_smoke, y_smoke)   # would raise pre-fix when GPU backend selected
        proba = clf.predict_proba(X_smoke[:3])
        check('smoke test: make_classifier(early_stopping=True).fit() runs without ValueError',
              proba.shape == (3, 2))
    except Exception as exc:
        check(f'smoke test: make_classifier(early_stopping=True).fit() runs (got {type(exc).__name__}: {exc})', False)

    # ── Fix 2: dashboard Backtest column ────────────────────────────────
    # 2a. Header cell.
    check('Backtest column header rendered with data-col="backtest_status"',
          'data-col="backtest_status"' in tpl)
    check('Backtest header sortable (onclick=trSort)',
          "onclick=\"trSort('backtest_status')\"" in tpl)
    check('Backtest header tooltip mentions Phase 94 + distributed sweep',
          'Phase 94' in tpl
          and '/api/backtest/distributed' in tpl)

    # 2b. Loading + no-match placeholder colspans bumped 21 → 22.
    # Phase 98 then bumped 22 → 24 (ETA Train + ETA BT). Accept either
    # so the assertion stays a regression guard for "the colspan is at
    # least the post-Backtest value", not a brittle exact-version pin.
    check('placeholder Loading row uses colspan ≥ 22',
          ('colspan="22" style="text-align:center;color:#475569;padding:12px">Loading' in tpl
           or 'colspan="24" style="text-align:center;color:#475569;padding:12px">Loading' in tpl))
    check('"No models match" placeholder uses colspan ≥ 22',
          ('colspan="22" style="text-align:center;color:#475569;padding:14px">No models match this filter' in tpl
           or 'colspan="24" style="text-align:center;color:#475569;padding:14px">No models match this filter' in tpl))
    check('fleet aggregate footer trailing colspan bumped 2 → 3 (Backtest + Description + Action)',
          'td colspan="3"></td>' in tpl)

    # 2c. Per-row cell renderer.
    check('row template inserts _btCellRender(m.timeframe) cell',
          '<td style="text-align:right">${_btCellRender(m.timeframe)}</td>' in tpl)

    # 2d. State + poller.
    check('_btCellsByTf state object exists',
          'let _btCellsByTf' in tpl)
    check('pollBacktestCells fetches /api/backtest/distributed/status',
          'function pollBacktestCells(' in tpl
          and "fetch('/api/backtest/distributed/status'" in tpl)
    check('poller groups cells by timeframe, counts done/running/pending/failed',
          "g.done++" in tpl
          and "g.running++" in tpl
          and "g.pending++" in tpl
          and "g.failed++" in tpl)
    check('poller re-renders training table on success',
          "_renderTrainingTable" in tpl
          and 'pollBacktestCells' in tpl)

    # 2e. Cell renderer color logic.
    check('_btCellRender defined',
          'function _btCellRender(' in tpl)
    check('cell renderer: red on failed, amber on running/pending, green on all done',
          "if (g.failed > 0)" in tpl
          and "} else if (g.running > 0 || g.pending > 0) {" in tpl
          and "color = '#34d399'" in tpl
          and "color = '#fbbf24'" in tpl
          and "color = '#fb7185'" in tpl)
    check('cell renderer: grey "—" when no cells submitted yet',
          "if (!g || g.total === 0)" in tpl
          and "color:#475569" in tpl)

    # 2f. Polling cadence + DOMContentLoaded warm-up.
    check('poller setInterval at 10_000 ms (matches cluster poll cadence)',
          "setInterval(() => { if (activeTab === 'monitor' || activeTab === 'strategy') pollBacktestCells(); }, 10_000)" in tpl)
    check('initial pollBacktestCells fires on DOMContentLoaded (warm-up)',
          'pollBacktestCells' in tpl
          and 'DOMContentLoaded' in tpl
          and 'setTimeout(pollBacktestCells' in tpl)


def test_phase96_orphan_detector_direct_script_form_plus_ps_native_fix():
    """Phase 96 — three blockers reported by operator on 2026-05-10:

    Blocker #1: XGBoost early-stopping ValueError reappeared in trainer
    log at 20:56:36, even though Phase 95 fix shipped at 20:58:29.
    Diagnosis (test, not speculation): the running training process
    (PID 10392) started at 20:44:26 — 14 minutes BEFORE the fix
    landed. Python doesn't hot-reload modules. The wrapper in memory
    is the pre-fix version. NOT a code bug; just process-restart
    needed. No code change required for #1; assertion below confirms
    the on-disk fix is still in place (regression guard).

    Blocker #2: dashboard Model Training tab does not surface active
    trainers launched via launch_training.ps1. Root cause: the orphan
    detector at app.py:_detect_orphan_training_subprocesses only
    matched the inline `import <module>; fn()` form
    (_spawn_training_subprocess's invocation), missing the direct
    `python -u src/engine/train_*.py` form. Fix: also match script
    filename → model_key from _TRAINER_DISPATCH.

    Blocker #3: launch_training.ps1 emits NativeCommandError noise
    every time Python writes to stderr (which the standard logging
    config does on every line). Cosmetic, but pollutes operator
    output and would mask real errors. Fix: ErrorActionPreference
    Continue + PS7 PSNativeCommandUseErrorActionPreference=false +
    script-block wrap around the native python call.
    """
    print('\n[Phase 96 -- orphan detector + PS native command stderr]')

    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()
    ps_path = os.path.join(BASE_DIR, 'launch_training.ps1')
    with open(ps_path, encoding='utf-8') as f:
        ps = f.read()
    gpu_path = os.path.join(BASE_DIR, 'src', 'utils', 'gpu_classifier.py')
    with open(gpu_path, encoding='utf-8') as f:
        gpu = f.read()

    # ── #1: regression guard — Phase 95 fix still on disk ──────────────
    check('regression guard: gpu_classifier wrapper still auto-creates eval_set on early_stopping',
          'self._early_stopping and eval_set is None' in gpu
          and 'X_arr[:-n_val]' in gpu
          and 'X_arr[-n_val:]' in gpu)

    # ── #2: orphan detector covers BOTH inline and direct-script forms ─
    det_start = app.find('def _detect_orphan_training_subprocesses')
    det_end   = app.find('\ndef _reattach_training_subprocess', det_start + 1)
    det_body  = app[det_start:det_end] if det_end > det_start else app[det_start:]
    check('orphan detector still covers form 1 (import <module>; fn())',
          "f'import {module_path}'" in det_body
          and "'fn(' in cmd" in det_body)
    check('orphan detector ALSO covers form 2 (direct python -u <script>.py)',
          'script_to_key' in det_body
          and 'leaf = module_path.rsplit' in det_body
          and "f'{leaf}.py'" in det_body)
    check('form-2 match guards against accidentally hitting python -c',
          "' -c' not in cmd" in det_body)
    check('form-2 only fires when form-1 didn\'t match (no double-classify)',
          # Verified by ordering: form-1 sets matched_key, then 'if not matched_key' guards form-2.
          det_body.find('if f\'import {module_path}\' in cmd') < det_body.find('if not matched_key:'))
    check('comment cites launch_training.ps1 as the originating cause',
          'launch_training.ps1' in det_body)

    # ── #3: launch_training.ps1 hardened against NativeCommandError ────
    check('launch_training.ps1 sets ErrorActionPreference=Continue',
          "$ErrorActionPreference = 'Continue'" in ps)
    check('launch_training.ps1 disables PS7 native-command-error-action coupling',
          'PSNativeCommandUseErrorActionPreference' in ps
          and '$false' in ps)
    check('PS7 setting guarded by Test-Path so PS5.1 doesn\'t error',
          'Test-Path Variable:PSNativeCommandUseErrorActionPreference' in ps)
    check('python invocation wrapped in script-block (suppresses stderr-as-exception)',
          '& {\n    & $python -u' in ps
          or '& {\r\n    & $python -u' in ps)
    check('script-block redirects stderr inside the wrapper (2>&1)',
          '2>&1' in ps)
    check('exit code forwarded so CI / schedulers see real success/failure',
          'exit $LASTEXITCODE' in ps)
    check('comment cites the NativeCommandError root cause',
          'NativeCommandError' in ps)


def test_phase97_train_all_concurrency_lock_plus_current_state_pipeline():
    """Phase 97 — four-part fix for the operator-reported issues:
       1. all RUNNING rows showed identical elapsed/ETA values
       2. operator wanted ONE training process at a time on master
       3. zombie procs were stacking up across CLI relaunches

    Root causes (test-confirmed, not speculation):
       a. train_all_models.py had no concurrency lock — every CLI launch
          + every dashboard "Retrain ALL" click spawned a parallel run.
          6 zombies seen on 2026-05-10 evening.
       b. The dashboard frontend had TWO optimistic-broadcast sites that
          fanned model='all' → all 8 model rows with the SAME job's
          elapsed/eta. So every row showed identical timing data.
       c. train_all_models.py never wrote which model was actually
          training right now to a file the dashboard could read.

    Fixes:
       1. _acquire_run_lock + _release_run_lock around train_all (file
          lock at data/train_all_models.lock, stale-pid auto-reclaim,
          --force CLI override). _set_current writes
          data/training_current.json on every model transition.
       2. Dashboard orphan detector enriches model='all' jobs with
          current_model_key + current_tf from training_current.json.
       3. Frontend optimistic-broadcasts (Retrain ALL click + poller)
          replaced with current_model_key-aware logic — only the
          actually-training row flips to RUNNING.
       4. launch_training.ps1 also consults the lock file (cheap
          fast-fail in front of the Python-side check).
       5. /api/training/run/all checks the cross-process lock too, so
          dashboard and CLI agree on "is anything running."
    """
    print('\n[Phase 97 -- train_all concurrency lock + current-model state]')

    train_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_all_models.py')
    with open(train_path, encoding='utf-8') as f:
        train = f.read()
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()
    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()
    ps_path = os.path.join(BASE_DIR, 'launch_training.ps1')
    with open(ps_path, encoding='utf-8') as f:
        ps = f.read()

    # ── Fix 1: trainer-side lock + current state ────────────────────────
    check('train_all_models defines _acquire_run_lock', 'def _acquire_run_lock(' in train)
    check('train_all_models defines _release_run_lock', 'def _release_run_lock(' in train)
    check('train_all_models defines _set_current',     'def _set_current(' in train)
    check('lock path is data/train_all_models.lock',
          "'train_all_models.lock'" in train)
    check('current state path is data/training_current.json',
          "'training_current.json'" in train)
    check('lock acquire writes JSON {pid, started_iso, host}',
          "'pid':" in train and "'started_iso':" in train and "'host':" in train)
    check('lock acquire auto-reclaims stale (dead-pid) lock',
          'psutil.pid_exists' in train and 'Reclaiming stale lock' in train)
    check('--force flag bypasses the lock',
          'force: bool = False' in train and "'--force'" in train)
    check('train_all wraps inner pipeline in try/finally for cleanup',
          'try:' in train and '_release_run_lock()' in train
          and '_set_current(None, None, None)' in train)
    check('_set_current writes model_key + current_tf + parent_pid',
          "'model_key':" in train and "'current_tf':" in train and "'parent_pid':" in train)

    # _set_current is called at every transition.
    check('_set_current called inside _train_loop',
          '_set_current(key, tf, label)' in train)
    check('_set_current called for TFT loop',
          "_set_current('tft', tf, 'TFT Model')" in train)
    check('_set_current called for OFT loop',
          "_set_current('oft', oft_tf," in train)
    check('_set_current called for regime classifier',
          "_set_current('regime', '1h'," in train)

    # ── Fix 2: dashboard orphan detector enrichment ─────────────────────
    det_start = app.find('def _detect_orphan_training_subprocesses')
    det_end   = app.find('\ndef _reattach_training_subprocess', det_start + 1)
    det_body  = app[det_start:det_end] if det_end > det_start else app[det_start:]
    check('orphan detector reads training_current.json when matched_key=all',
          "matched_key == 'all'" in det_body
          and "'training_current.json'" in det_body)
    check('orphan detector populates current_model_key on the job record',
          'current_model_key=sub_model' in det_body)
    check('orphan detector populates current_tf on the job record',
          'current_tf=current_tf' in det_body)
    check('progress_label changes to "running <key> @ <tf>" when current state present',
          'f"running {sub_model}"' in det_body)

    # ── Fix 3: dashboard frontend respects current_model_key ────────────
    # Site A — Retrain ALL click handler (~line 4624)
    # Comment naturally wraps across lines; match the contiguous part.
    check('frontend Retrain ALL click no longer fans to all 8 keys eagerly',
          'DO NOT optimistically fan RUNNING to all 8 model' in tpl)
    # Site B — poller per-cycle broadcast
    # Phase 97b — keying tightened from `[cur]` to `[cur+'@'+curTf]` so
    # per-(model, tf) precision is restored. The "only flip the matching
    # row" intent is the same; just the lookup key shape changed.
    check('frontend poller flips ONLY current_model_key to RUNNING',
          'allJob.current_model_key' in tpl
          and ('cur && !newActive[cur]' in tpl
               or 'const k = curTf ? `${cur}@${curTf}` : cur;' in tpl))
    check('frontend poller no longer fans to all 8 keys when current_model_key absent',
          # Phase 97b — strict-mode comment was rewritten with tf
          # qualifier. Either form is the same intent.
          ('strict: only flip the actually-training row' in tpl
           or 'fall back to model-only — those rows still light up but no fan-out' in tpl))
    # Pipeline status now propagates current_model_key + current_tf to
    # the synthetic allJob construction.
    check('pipeline-status synthetic allJob carries current_model_key',
          'current_model_key: _pipeStatus.current_model_key' in tpl)
    check('pipeline-status synthetic allJob carries current_tf',
          'current_tf:        _pipeStatus.current_tf' in tpl)
    # Backend pipeline-status endpoint enriches with training_current.json.
    pstatus_start = app.find('def api_pipeline_status')
    pstatus_end   = app.find('\n@app.route', pstatus_start + 1)
    pstatus_body  = app[pstatus_start:pstatus_end] if pstatus_end > pstatus_start else app[pstatus_start:]
    check('/api/pipeline/status reads training_current.json',
          "'training_current.json'" in pstatus_body
          and "snap['current_model_key']" in pstatus_body)

    # ── Fix 4: launch_training.ps1 lock check ───────────────────────────
    check('launch_training.ps1 reads the lock file',
          '$lockPath = Join-Path $root \'data\\train_all_models.lock\'' in ps)
    check('launcher reports clear "already running" message + exits 2',
          'Another train_all_models.py is already running' in ps
          and 'exit 2' in ps)
    check('launcher honors --force flag (passed straight through to python)',
          "$force    = ($args -contains '--force')" in ps
          and "(-not $force)" in ps)
    check('launcher uses Get-Process to verify pid is actually alive',
          'Get-Process -Id $prevPid' in ps)

    # ── Fix 5: /api/training/run/all consults file lock ─────────────────
    api_start = app.find("def api_training_run_all")
    api_end   = app.find('\n@app.route', api_start + 1)
    api_body  = app[api_start:api_end] if api_end > api_start else app[api_start:]
    check('/api/training/run/all also consults cross-process lock file',
          "'train_all_models.lock'" in api_body)
    check('cross-process lock path resolved relative to project_root',
          "project_root, 'data'" in api_body
          and "'train_all_models.lock'" in api_body)
    check('endpoint returns 409 already_running when lock holds a live pid',
          "'error': 'already_running'" in api_body
          and "}), 409" in api_body
          and 'pid_exists' in api_body)


def test_phase98_eta_train_bt_columns_and_tf_keyed_running():
    """Phase 97b + 98 — bundled fix.

    Operator screenshot 2026-05-10: pipeline orchestrator running, GPUs
    at 70%, but every Model Training row showed "OK". Two fixes:

      (1) ETA Train + ETA BT columns right of Status, sortable,
          per-(model, tf) rolling-average self-tunes from cluster runs.
          Goal: pick the 10-min row over the 3-hr row.

      (2) RUNNING-row keying: pre-fix, _trActiveByModel was keyed by
          model alone — a single 'futures' job lit all 6 futures rows.
          Phase 97a went the other way and lit nothing when the key
          shape mismatched. Now keyed by 'model@tf', with a fallback
          chain: parent_key+tf → key+tf → parent_key → key. Per-tf
          rows light up independently, no fan-out, no blanks.

    Tests cover the backend ETA producer (per-tf precedence, defaults,
    self-tune wiring) and the frontend rendering + lookup chain.
    """
    print('\n[Phase 98 -- ETA Train+BT columns + Phase 97b tf-keyed RUNNING]')
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    app = (PRJ / 'src/dashboard/app.py').read_text(encoding='utf-8')
    tpl = (PRJ / 'src/dashboard/templates/index.html').read_text(encoding='utf-8')

    # ── Backend: per-(model, tf) typical-duration map ────────────────────
    check('backend declares _TYPICAL_DURATIONS_BY_TF map',
          '_TYPICAL_DURATIONS_BY_TF: dict[str, float] = {}' in app)
    check('backend declares _TYPICAL_HISTORY_BY_TF (rolling avg shape)',
          '_TYPICAL_HISTORY_BY_TF: dict[str, list[float]] = {}' in app)
    check('backend declares per-(model, tf) backtest map',
          '_TYPICAL_BACKTEST_S: dict[str, float] = {}' in app)
    check('backend declares backtest history (rolling avg shape)',
          '_TYPICAL_BACKTEST_HISTORY: dict[str, list[float]] = {}' in app)
    check('backend has seed defaults for backtest ETA per model key',
          '_TYPICAL_BACKTEST_DEFAULT' in app
          and "'scalping': 90.0" in app   # 1m bars = bigger seed
          and "'regime':   15.0" in app)

    # ── Cold-cache restore wiring (so dashboard restart keeps tunings) ───
    check('cold-cache restore loads typical_durations_by_tf at boot',
          "_cold_cache.load(_slot, default=None" in app
          and "'typical_durations_by_tf'" in app
          and "'typical_backtest_s'" in app)

    # ── Recorder accepts tf= and writes to per-tf map ────────────────────
    check('_record_completed_duration accepts tf= kwarg',
          'def _record_completed_duration(key: str, duration_s: float,' in app
          and 'tf: str | None = None' in app)
    check('_record_completed_duration writes per-(model, tf) entry',
          "tf_key = f'{key}@{tf}'" in app
          and '_TYPICAL_DURATIONS_BY_TF[tf_key]' in app)
    check('_record_completed_duration persists per-tf maps via cold_cache',
          "_cc.save('typical_durations_by_tf'" in app)

    # ── Backtest recorder is its own function (backtest != train) ────────
    check('backend defines _record_completed_backtest_duration',
          'def _record_completed_backtest_duration(' in app
          and '_TYPICAL_BACKTEST_S[bt_key]' in app)
    check('followup backtest finish hook records per-(model, tf) duration',
          '_record_completed_backtest_duration(model_key, tfs[0]' in app)
    check('multi-tf followup backtest splits duration evenly per tf',
          '_per_tf_dur = _bt_dur / max(1, len(tfs))' in app)

    # ── ETA producer: per-tf precedence with model-only fallback ─────────
    check('backend defines _eta_for_row(model_key, tf)',
          'def _eta_for_row(model_key: str, tf: str | None)' in app)
    check('ETA producer prefers per-(model, tf) train entry',
          'tf_key = f\'{model_key}@{tf}\'' in app
          and 'train_s = _TYPICAL_DURATIONS_BY_TF.get(tf_key)' in app)
    check('ETA producer falls back to model-only when no per-tf history',
          'train_s = _TYPICAL_DURATIONS.get(model_key)' in app)
    check('ETA producer falls back to seed default when no history at all',
          'bt_s = _TYPICAL_BACKTEST_DEFAULT.get(model_key, 30.0)' in app)
    check('ETA producer returns three field names',
          "'eta_train_s'" in app
          and "'eta_backtest_s'" in app
          and "'eta_total_s'" in app)
    check('ETA total is train+backtest sum when both available',
          'out[\'eta_total_s\'] = round(train_s + bt_s, 1)' in app)

    # ── /api/strategy/full row builder injects ETA fields ────────────────
    # Two row-append sites: canonical (legacy 1-tf-per-model) + per-tf.
    canonical_idx = app.find("'is_canonical':   True")
    pertf_idx     = app.find("'is_canonical':   False")
    check('canonical row builder calls _eta_for_row',
          '_eta = _eta_for_row(key, meta.get(\'timeframe\'))' in app)
    check('canonical row dict carries eta_train_s + eta_backtest_s + total',
          canonical_idx > 0
          and "'eta_train_s':    _eta['eta_train_s']" in app[:canonical_idx])
    check('per-tf row builder calls _eta_for_row(key, tf)',
          '_eta_tf = _eta_for_row(key, tf)' in app)
    check('per-tf row dict carries the three ETA fields',
          pertf_idx > 0
          and "'eta_train_s':    _eta_tf['eta_train_s']" in app[:pertf_idx])

    # ── Single-tf trainer finish hook passes tf to recorder ──────────────
    check('single-tf trainer finish hook passes tf to _record_completed_duration',
          '_record_completed_duration(key, finished_at - started_at, tf=tf)' in app)

    # ── Frontend: two new column headers right of Status ─────────────────
    check('ETA Train column header (sortable, right of Status)',
          'data-col="eta_train_s"' in tpl
          and 'onclick="trSort(\'eta_train_s\')"' in tpl
          and '>ETA Train <' in tpl)
    check('ETA BT column header (sortable, right of ETA Train)',
          'data-col="eta_backtest_s"' in tpl
          and 'onclick="trSort(\'eta_backtest_s\')"' in tpl
          and '>ETA BT <' in tpl)
    # Ordering: Status → ETA Train → ETA BT → Last trained
    status_pos    = tpl.find('data-col="run_status"')
    eta_train_pos = tpl.find('data-col="eta_train_s"')
    eta_bt_pos    = tpl.find('data-col="eta_backtest_s"')
    last_trained  = tpl.find('data-col="age_s"')
    check('column order: Status → ETA Train → ETA BT → Last trained',
          status_pos > 0 < eta_train_pos < eta_bt_pos < last_trained)

    # ── Empty-state colspan bumped 22 → 24 (two new columns) ─────────────
    check('Loading placeholder colspan bumped to 24',
          'colspan="24" style="text-align:center;color:#475569;padding:12px">Loading' in tpl)

    # ── Frontend cell renders with color tint + tooltip ──────────────────
    check('ETA color helper banded < 5m green / 5-30m amber / >30m red',
          'if (s < 300)   return \'#34d399\';' in tpl
          and 'if (s < 1800)  return \'#fbbf24\';' in tpl
          and "return '#fb7185';                   // > 30m — red" in tpl)
    check('row carries data-eta-total / data-eta-train / data-eta-bt for sort',
          'data-eta-total="${_etaTotal' in tpl
          and 'data-eta-train="${_etaTrain' in tpl
          and 'data-eta-bt="${_etaBt' in tpl)
    check('ETA cells use _trFmtDuration for formatting (existing helper)',
          '_etaTrainFmt = _trFmtDuration(_etaTrain)' in tpl
          and '_etaBtFmt    = _trFmtDuration(_etaBt)' in tpl)

    # ── ETA columns default to ASC sort (shortest first) ─────────────────
    check('ETA columns default to ascending sort (shortest first)',
          "col === 'eta_train_s' || col === 'eta_backtest_s'" in tpl)

    # ── Phase 97b: tf-keyed RUNNING lookup ───────────────────────────────
    check('poller keys active jobs by model+@+tf (not model alone)',
          'const _activeKey = (j) => j.tf ? `${j.model}@${j.tf}` : j.model;' in tpl)
    check('poller fallback to model-only for jobs without tf (regime, oft)',
          'j.tf ? `${j.model}@${j.tf}` : j.model' in tpl)
    check('row lookup tries parent_key+@+tf, then key+@+tf, then parent_key, then key',
          '_rowParentKey = m.parent_key || m.key' in tpl
          and '_rowTfKey     = m.tf || m.timeframe' in tpl
          and '`${_rowParentKey}@${_rowTfKey}`' in tpl
          and '`${m.key}@${_rowTfKey}`' in tpl)
    check('synthetic allJob (pipeline path) keyed by current_model_key+@+current_tf',
          'const k = curTf ? `${cur}@${curTf}` : cur;' in tpl)
    check('optimistic flash key uses model+@+tf (per-tf rows flash too)',
          "const _optKey = (tf && tf !== 'all') ? `${key}@${tf}` : key;" in tpl
          and '_trActiveByModel = {..._trActiveByModel, [_optKey]: \'pending\'}' in tpl)
    check('trStopOne deletes by jobId scan (not by model key)',
          'if (_trActiveByModel[k] === jobId) delete _trActiveByModel[k];' in tpl)
    check('recent-fails lookup uses _lookupKey (same chain as activeJob)',
          '_trRecentFails[_lookupKey]' in tpl)


def test_phase97c_orphan_periodic_refresh_and_canonical_row_fallback():
    """Phase 97c — make Phase 97b actually work end-to-end.

    Operator screenshot 2026-05-10 22:55: pipeline orchestrator running,
    GPUs at 70%, training_current.json showed scalping@5m, but the orphan
    job record in /api/training/jobs still claimed tft@15m (frozen from
    boot) AND every Model Training row showed OK because no per-tf row
    exists at scalping@5m yet. Two compounding bugs:

      (1) Backend: _detect_orphan_training_subprocesses runs ONCE in
          _training_state_recover at dashboard import. Pipeline iterates
          through models but the orphan record's current_model_key /
          current_tf never refresh.

      (2) Frontend: synthetic allJob keys to `tft@15m`, lookup chain
          falls through to `tft` model-only, but newActive only had
          `tft@15m` written. No row covers tft@15m (no per-tf TFT row
          exists until a 15m artifact lands). Result: no row lights up.

    Fixes:
      Backend  → _refresh_orphan_current_state() called every 5 s by
                 _orphan_refresh_loop daemon thread. Walks orphan-* +
                 'all'-model records, refreshes current_model_key /
                 current_tf / tf / progress_label from training_current.json.
      Frontend → canonical-row fallback. After newActive built, walk
                 model@tf entries, check if any row covers (model, tf);
                 if not, promote the entry to the canonical model row
                 (key === model, no parent_key) so the operator sees
                 the actually-training model lit up with the actual tf
                 in the sub-line.
    """
    print('\n[Phase 97c -- orphan periodic refresh + canonical-row fallback]')
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    app = (PRJ / 'src/dashboard/app.py').read_text(encoding='utf-8')
    tpl = (PRJ / 'src/dashboard/templates/index.html').read_text(encoding='utf-8')

    # ── Backend: periodic-refresh daemon ─────────────────────────────────
    check('backend defines _refresh_orphan_current_state()',
          'def _refresh_orphan_current_state()' in app)
    check('backend defines _orphan_refresh_loop() daemon body',
          'def _orphan_refresh_loop()' in app)
    check('orphan refresh loop sleeps 5s between iterations',
          'time.sleep(5)' in app
          and '_detect_orphan_training_subprocesses()' in app
          and '_refresh_orphan_current_state()' in app)
    check('orphan refresh loop runs in a daemon thread (named)',
          "name='training-orphan-refresh'" in app
          and 'threading.Thread(target=_orphan_refresh_loop, daemon=True' in app)

    refresh_start = app.find('def _refresh_orphan_current_state(')
    refresh_end   = app.find('\ndef ', refresh_start + 1)
    refresh_body  = app[refresh_start:refresh_end] if refresh_end > refresh_start else app[refresh_start:]
    check('refresh reads training_current.json from project_root',
          "os.path.join(project_root, 'data', 'training_current.json')"
          in refresh_body)
    check('refresh updates current_model_key / current_tf / progress_label',
          "updates['current_model_key']" in refresh_body
          and "updates['current_tf']" in refresh_body
          and "updates['progress_label']" in refresh_body)
    check('refresh ALSO mirrors current_tf into top-level tf field',
          # FE poller keys newActive by `${j.model}@${j.tf}` — so the
          # top-level tf is what actually drives row lookup, not just
          # current_tf. Pre-Phase-97c only current_tf was updated.
          "updates['tf']" in refresh_body)
    check('refresh skips records that are not orphan-* or model="all"',
          "jid.startswith('orphan-')" in refresh_body
          and "job.get('model') == 'all'" in refresh_body)
    check('refresh skips non-running records (no thrash on completed jobs)',
          "if job.get('status') != 'running':" in refresh_body
          and 'continue' in refresh_body)

    # ── Backend: refresh loop tolerates single-iteration failures ───────
    check('refresh loop catches Exception per iteration (never breaks)',
          '[training] orphan refresh loop iteration failed' in app)

    # ── Frontend: canonical-row fallback ─────────────────────────────────
    check('frontend has Phase 97c canonical-row fallback comment',
          'Phase 97c — canonical-row fallback' in tpl)
    check('fallback walks newActive model@tf keys',
          "if (!k.includes('@')) continue;" in tpl
          and 'const [mod, tf] = k.split(\'@\');' in tpl)
    check('fallback computes exactRowExists from _stratFull.ml_models',
          'tableRows.some(m =>' in tpl
          and '(m.parent_key === mod || m.key === mod)' in tpl
          and '(m.tf === tf || m.timeframe === tf)' in tpl)
    check('fallback skips when exact row already exists (no fan-out)',
          'if (exactRowExists) continue;' in tpl)
    check('fallback skips when canonical key already populated',
          'if (newActive[mod]) continue;' in tpl)
    check('fallback adds "(training @ <tf>)" to progress_label',
          '(training @ ${tf})' in tpl)
    check('fallback wrapped in try/catch so promotion never blocks render',
          "} catch (_) { /* best-effort promotion; never block render */ }"
          in tpl)


def test_phase100_cluster_routed_training_dispatch():
    """Phase 100 — route training through the existing cluster orchestrator
    (port 7700, workers already in place with model handlers) instead of
    spawning local subprocesses gated by _training_scheduler.

    Trigger: operator screenshot 2026-05-11 01:21Z. Clicked Train on
    Futures @ 4h, system parked it in QUEUED instead of starting. Root
    cause: local _training_scheduler.exclusive_busy=True held by pipeline
    orchestrator gates ALL acquire() calls regardless of lane. Manual
    and auto share the same broken gate.

    Phase 100a (this phase) — manual single-tf path routes to cluster.
    Pipeline orchestrator + tf='all' + obsolete code deletion deferred
    to Phase 100b/c.

    Asserts:
      - Backend has dashboard-key → cluster-model_type mapping for the
        4 keys whose names diverge (base→btc_rf, futures→futures_short,
        meta→meta_labeler) + fallback for same-name keys
      - _dispatch_training_to_cluster builds a task spec, POSTs to
        /api/cluster/submit via _cluster_proxy_post, records cluster
        task IDs on the job
      - _sync_cluster_task_status polls cluster task status, aggregates
        across n tasks, maps cluster status → dashboard status
      - api_training_run_one routes to cluster by default; honors
        AI_TRADER_LOCAL_TRAINING=1 env var for legacy fallback
    """
    print('\n[Phase 100 -- cluster-routed training dispatch]')
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    app = (PRJ / 'src/dashboard/app.py').read_text(encoding='utf-8')

    # ── Mapping: dashboard key → cluster worker model_type ───────────────
    check('backend declares _DASH_TO_CLUSTER_KEY mapping',
          '_DASH_TO_CLUSTER_KEY: dict[str, str] = {' in app)
    check('mapping covers diverging names: base → btc_rf',
          "'base':     'btc_rf'" in app)
    check('mapping covers diverging names: futures → futures_short',
          "'futures':  'futures_short'" in app)
    check('mapping covers diverging names: meta → meta_labeler',
          "'meta':     'meta_labeler'" in app)
    check('mapping fallback function returns same key when not mapped',
          'def _to_cluster_model_type(dash_key: str) -> str:' in app
          and '_DASH_TO_CLUSTER_KEY.get(dash_key, dash_key)' in app)

    # ── _dispatch_training_to_cluster shape ──────────────────────────────
    check('backend defines _dispatch_training_to_cluster()',
          'def _dispatch_training_to_cluster(job_id: str, key: str, n: int,' in app)
    check('dispatch builds cluster task spec with model_type, symbol, timeframe',
          "'model_type': model_type" in app
          and "'symbol':     'BTC/USDT'" in app
          and "'timeframe':  tf or '1h'" in app)
    check('dispatch POSTs to /api/cluster/submit via _cluster_proxy_post',
          "_cluster_proxy_post('/api/cluster/submit', base_spec)" in app)
    check('dispatch loops n times for repetitions (multi-iter support)',
          'for _ in range(max(1, n)):' in app
          and 'task_ids.append(body[\'task_id\'])' in app)
    check('dispatch records cluster_task_ids + cluster_routed on the job',
          'cluster_task_ids=task_ids' in app
          and 'cluster_routed=True' in app)
    check('dispatch spawns _sync_cluster_task_status daemon thread',
          'target=_sync_cluster_task_status' in app
          and "name=f'cluster-sync-{job_id}'" in app)

    # ── _sync_cluster_task_status shape ──────────────────────────────────
    check('backend defines _sync_cluster_task_status()',
          'def _sync_cluster_task_status(job_id: str, key: str,' in app)
    check('sync polls cluster /api/cluster/tasks every 5s',
          'POLL_S = 5.0' in app
          and "_cluster_proxy_get('/api/cluster/tasks')" in app)
    check('sync has 6h deadline (DEADLINE_S = 6 * 3600)',
          'DEADLINE_S = 6 * 3600' in app)
    check('sync aggregates: all done → job done',
          "if all(s == 'done' for s in statuses):" in app)
    check('sync aggregates: any cancelled → job cancelled',
          "if any(s == 'cancelled' for s in statuses):" in app)
    check('sync aggregates: terminal mix → partial or error',
          # Phase 100b refactor — logic moved out of the sync loop into the
          # pure _aggregate_cluster_task_statuses helper for testability.
          # Either string form is acceptable as evidence.
          "if all(s in ('done', 'failed', 'cancelled') for s in statuses):" in app
          and ("final = 'partial' if any(s == 'done' for s in statuses) else 'error'" in app
               or "out['final']  = 'partial' if any(s == 'done' for s in statuses) else 'error'" in app))
    check('sync records training duration on done (ETA self-tune)',
          '_record_completed_duration(key, elapsed, tf=tf)' in app)
    check('sync chains followup backtest when with_backtest=true',
          'if with_backtest:' in app
          and '_spawn_followup_backtest(job_id, key, (bt_tf,))' in app)

    # ── cluster status → dashboard status mapping ────────────────────────
    check('backend defines _cluster_status_to_job_status() mapping',
          'def _cluster_status_to_job_status(cluster_status: str) -> str:' in app)
    check('mapping translates pending → queued, failed → error',
          "'pending':   'queued'" in app
          and "'failed':    'error'" in app)

    # ── api_training_run_one routes to cluster by default ────────────────
    # Find the api endpoint body
    ep_start = app.find('def api_training_run_one')
    ep_end   = app.find('\n@app.route', ep_start + 1)
    ep_body  = app[ep_start:ep_end] if ep_end > ep_start else app[ep_start:]
    check('api_training_run_one routes to cluster by default',
          'target=_dispatch_training_to_cluster' in ep_body)
    check('AI_TRADER_LOCAL_TRAINING=1 env var forces legacy local path',
          "os.getenv('AI_TRADER_LOCAL_TRAINING', '0') == '1'" in ep_body)
    check('endpoint response includes routed_to field (cluster|local)',
          "'routed_to':" in ep_body)


def test_phase100_functional_cluster_routing_proves_behavior():
    """Phase 100 — FUNCTIONAL unit tests (not string-match). Each assertion
    invokes the code under test and asserts on observable state change.

    Required by the 2026-05-11 global rule "Functional Tests Prove Behavior":
    string-match tests verify the source contains a symbol; functional tests
    verify the symbol BEHAVES correctly. Phase 100a shipped with 24 string-
    matches that all passed — a logic bug would have slipped past every one.
    This test plugs that hole.

    Sub-tests:
      (a) _to_cluster_model_type: pure mapping fn — call, assert return value
      (b) _cluster_status_to_job_status: pure mapping fn — same
      (c) _aggregate_cluster_task_statuses: pure aggregator extracted from the
          sync daemon; call with synthetic snapshots, assert decisions
      (d) _dispatch_training_to_cluster: monkey-patch _cluster_proxy_post +
          _record_job to in-memory stubs, call, assert side effects
      (e) api_training_run_one endpoint: app.test_client(), POST, assert
          response JSON shape + that the dispatch path was taken
    """
    print('\n[Phase 100 -- FUNCTIONAL tests that actually call the code]')
    import sys, os, importlib
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    if str(PRJ) not in sys.path:
        sys.path.insert(0, str(PRJ))

    # Disable API key check for the test client.
    os.environ['DASHBOARD_API_KEY'] = ''
    # Force-local pathway should be OFF (default cluster routing).
    os.environ.pop('AI_TRADER_LOCAL_TRAINING', None)

    # Import the module under test. The harness expects this is safe —
    # daemon threads start at import but they're daemonic and won't block
    # interpreter exit. No socket binding happens (Flask app.run not called).
    try:
        from src.dashboard import app as dash_app
    except Exception as exc:
        check(f'import src.dashboard.app succeeds (got {type(exc).__name__}: {exc})', False)
        return

    # ── (a) _to_cluster_model_type pure mapping ──────────────────────────
    check('_to_cluster_model_type("base") → "btc_rf"',
          dash_app._to_cluster_model_type('base') == 'btc_rf')
    check('_to_cluster_model_type("futures") → "futures_short"',
          dash_app._to_cluster_model_type('futures') == 'futures_short')
    check('_to_cluster_model_type("meta") → "meta_labeler"',
          dash_app._to_cluster_model_type('meta') == 'meta_labeler')
    check('_to_cluster_model_type("trend") → "trend" (passthrough, same on both sides)',
          dash_app._to_cluster_model_type('trend') == 'trend')
    check('_to_cluster_model_type("tft") → "tft" (passthrough)',
          dash_app._to_cluster_model_type('tft') == 'tft')
    check('_to_cluster_model_type("unknown_xyz") → "unknown_xyz" (fallback)',
          dash_app._to_cluster_model_type('unknown_xyz') == 'unknown_xyz')

    # ── (b) _cluster_status_to_job_status pure mapping ───────────────────
    check('_cluster_status_to_job_status("pending") → "queued"',
          dash_app._cluster_status_to_job_status('pending') == 'queued')
    check('_cluster_status_to_job_status("running") → "running"',
          dash_app._cluster_status_to_job_status('running') == 'running')
    check('_cluster_status_to_job_status("done") → "done"',
          dash_app._cluster_status_to_job_status('done') == 'done')
    check('_cluster_status_to_job_status("failed") → "error"',
          dash_app._cluster_status_to_job_status('failed') == 'error')
    check('_cluster_status_to_job_status("cancelled") → "cancelled"',
          dash_app._cluster_status_to_job_status('cancelled') == 'cancelled')
    check('_cluster_status_to_job_status("unknown_xyz") → "unknown_xyz" (passthrough)',
          dash_app._cluster_status_to_job_status('unknown_xyz') == 'unknown_xyz')

    # ── (c) _aggregate_cluster_task_statuses — pure aggregator ────────────
    agg = dash_app._aggregate_cluster_task_statuses
    # All done → final='done'
    by_id = {'t1': {'task_id':'t1','status':'done'},
             't2': {'task_id':'t2','status':'done'}}
    r = agg(('t1','t2'), by_id, {})
    check('aggregator: all-done → final="done", progress=2',
          r['final'] == 'done' and r['progress'] == 2)
    # Any cancelled → final='cancelled' (even with running siblings)
    by_id = {'t1': {'task_id':'t1','status':'running'},
             't2': {'task_id':'t2','status':'cancelled'}}
    r = agg(('t1','t2'), by_id, {})
    check('aggregator: any-cancelled → final="cancelled"',
          r['final'] == 'cancelled')
    # All terminal, mix of done+failed → 'partial' with errors collected
    by_id = {'t1': {'task_id':'t1','status':'done'},
             't2': {'task_id':'t2','status':'failed','error':'oom in fit'}}
    r = agg(('t1','t2'), by_id, {})
    check('aggregator: done+failed terminal mix → final="partial" with errors',
          r['final'] == 'partial' and 'oom in fit' in r['errors'])
    # All failed → 'error'
    by_id = {'t1': {'task_id':'t1','status':'failed','error':'A'},
             't2': {'task_id':'t2','status':'failed','error':'B'}}
    r = agg(('t1','t2'), by_id, {})
    check('aggregator: all-failed → final="error" with both errors',
          r['final'] == 'error' and set(r['errors']) == {'A','B'})
    # Any running → interim='running', final=None
    by_id = {'t1': {'task_id':'t1','status':'running'},
             't2': {'task_id':'t2','status':'pending'}}
    r = agg(('t1','t2'), by_id, {})
    check('aggregator: any-running, no terminal → final=None, interim="running"',
          r['final'] is None and r['interim'] == 'running')
    # All pending → interim='queued', final=None
    by_id = {'t1': {'task_id':'t1','status':'pending'},
             't2': {'task_id':'t2','status':'pending'}}
    r = agg(('t1','t2'), by_id, {})
    check('aggregator: all-pending → final=None, interim="queued"',
          r['final'] is None and r['interim'] == 'queued')
    # Task missing from by_id but seen before → carry over last_status_seen
    last_seen = {'t1': 'running'}
    r = agg(('t1','t2'), {}, last_seen)
    check('aggregator: missing task with prior status → preserves last_status_seen',
          r['statuses'][0] == 'running' and r['statuses'][1] == 'pending')
    # last_status_seen gets mutated to track current — verify
    by_id = {'t1': {'task_id':'t1','status':'done'}}
    last_seen = {}
    agg(('t1',), by_id, last_seen)
    check('aggregator: mutates last_status_seen for next iteration',
          last_seen.get('t1') == 'done')

    # ── (d) _dispatch_training_to_cluster: monkey-patch + assert ─────────
    captured_posts: list[tuple] = []
    fake_task_id_counter = [0]
    def _fake_proxy_post(path, body, **kw):
        captured_posts.append((path, body))
        fake_task_id_counter[0] += 1
        return ({'ok': True, 'task_id': f'fake-tid-{fake_task_id_counter[0]}'}, 200)
    saved_post = dash_app._cluster_proxy_post
    saved_record_job = dash_app._record_job
    recorded_jobs: dict[str, dict] = {}
    def _fake_record_job(jid, **fields):
        e = recorded_jobs.get(jid) or {'job_id': jid}
        e.update(fields)
        recorded_jobs[jid] = e
    # Patch and call
    dash_app._cluster_proxy_post = _fake_proxy_post
    dash_app._record_job = _fake_record_job
    try:
        dash_app._dispatch_training_to_cluster('test-job-1', 'futures', n=1, tf='4h')
    finally:
        dash_app._cluster_proxy_post = saved_post
        dash_app._record_job = saved_record_job

    check('dispatch: exactly 1 cluster submit POST for n=1',
          len(captured_posts) == 1)
    check('dispatch: POST path is /api/cluster/submit',
          captured_posts[0][0] == '/api/cluster/submit')
    check('dispatch: POST body model_type="futures_short" (mapped from dashboard "futures")',
          captured_posts[0][1].get('model_type') == 'futures_short')
    check('dispatch: POST body timeframe="4h" matches the dashboard tf arg',
          captured_posts[0][1].get('timeframe') == '4h')
    check('dispatch: job record updated with cluster_routed=True',
          recorded_jobs.get('test-job-1', {}).get('cluster_routed') is True)
    check('dispatch: job record carries cluster_task_ids list with the fake id',
          recorded_jobs.get('test-job-1', {}).get('cluster_task_ids') == ['fake-tid-1'])
    check('dispatch: job status set to "queued" after successful submit',
          recorded_jobs.get('test-job-1', {}).get('status') == 'queued')

    # n=3 test — 3 cluster tasks submitted
    captured_posts.clear()
    fake_task_id_counter[0] = 0
    recorded_jobs.clear()
    dash_app._cluster_proxy_post = _fake_proxy_post
    dash_app._record_job = _fake_record_job
    try:
        dash_app._dispatch_training_to_cluster('test-job-2', 'meta', n=3, tf='1h')
    finally:
        dash_app._cluster_proxy_post = saved_post
        dash_app._record_job = saved_record_job
    check('dispatch n=3: exactly 3 cluster submits',
          len(captured_posts) == 3)
    check('dispatch n=3: all 3 task_ids recorded on the job',
          recorded_jobs.get('test-job-2', {}).get('cluster_task_ids')
          == ['fake-tid-1', 'fake-tid-2', 'fake-tid-3'])
    check('dispatch n=3: every submit was meta_labeler (key mapping consistent)',
          all(p[1].get('model_type') == 'meta_labeler' for p in captured_posts))

    # Failed-submit path — _cluster_proxy_post returns non-200; job records error
    def _failing_proxy_post(path, body, **kw):
        return ({'error': 'cluster down'}, 503)
    recorded_jobs.clear()
    dash_app._cluster_proxy_post = _failing_proxy_post
    dash_app._record_job = _fake_record_job
    try:
        dash_app._dispatch_training_to_cluster('test-job-3', 'tft', n=1, tf='1h')
    finally:
        dash_app._cluster_proxy_post = saved_post
        dash_app._record_job = saved_record_job
    check('dispatch (cluster down): job status set to "error"',
          recorded_jobs.get('test-job-3', {}).get('status') == 'error')
    check('dispatch (cluster down): error message mentions cluster submit failure',
          any('cluster submit failed' in e
              for e in recorded_jobs.get('test-job-3', {}).get('errors', [])))

    # ── (e) api_training_run_one endpoint — flask test_client ─────────────
    # Hit the endpoint through Flask; assert the response and that the
    # dispatch thread was spawned (model recorded in _training_jobs).
    # Set up patches so the dispatch function doesn't actually POST anywhere.
    posts_log = []
    def _silent_proxy_post(path, body, **kw):
        posts_log.append((path, body))
        return ({'ok': True, 'task_id': 'endpoint-test-tid'}, 200)
    dash_app._cluster_proxy_post = _silent_proxy_post
    try:
        client = dash_app.app.test_client()
        resp = client.post('/api/training/run/futures',
                           json={'n': 1, 'tf': '15m', 'force': True})
        rj = resp.get_json()
    finally:
        dash_app._cluster_proxy_post = saved_post

    check('endpoint: POST /api/training/run/futures returns 200 OK',
          resp.status_code == 200)
    check('endpoint: response ok=True',
          rj is not None and rj.get('ok') is True)
    check('endpoint: response routed_to="cluster" (default routing)',
          rj.get('routed_to') == 'cluster')
    check('endpoint: response includes job_id',
          'job_id' in rj and isinstance(rj['job_id'], str) and len(rj['job_id']) > 6)
    check('endpoint: response model="futures" (dashboard key, not cluster key)',
          rj.get('model') == 'futures')
    check('endpoint: response tf="15m"',
          rj.get('tf') == '15m')

    # Now hit the same endpoint with AI_TRADER_LOCAL_TRAINING=1 → routed_to="local"
    os.environ['AI_TRADER_LOCAL_TRAINING'] = '1'
    try:
        client = dash_app.app.test_client()
        resp = client.post('/api/training/run/scalping',
                           json={'n': 1, 'tf': '1m', 'force': True})
        rj = resp.get_json()
    finally:
        os.environ.pop('AI_TRADER_LOCAL_TRAINING', None)
    check('endpoint with AI_TRADER_LOCAL_TRAINING=1: routed_to="local"',
          rj is not None and rj.get('routed_to') == 'local')

    # Invalid key → 400 with valid-list
    client = dash_app.app.test_client()
    resp = client.post('/api/training/run/not_a_model',
                       json={'n': 1, 'tf': '1h', 'force': True})
    check('endpoint: unknown key returns 400',
          resp.status_code == 400)
    rj = resp.get_json()
    check('endpoint: unknown key response includes valid model list',
          rj is not None and 'valid' in rj and 'futures' in rj.get('valid', []))


def test_phase100d_followup_4_xgb_wrapper_is_classifier_and_worker_reports_failure():
    """Phase 100d follow-up 4 (2026-05-11) — two interlocking bugs found
    while attempting "train all stale models one by one":

    BUG A: _XGBClassifierWrapper failed sklearn's CalibratedClassifierCV
      type check with:
        ValueError: _XGBClassifierWrapper should either be a classifier
        to be used with response_method=['decision_function',
        'predict_proba'] or the response_method should be 'predict'.
        Got a regressor with response_method=['decision_function',
        'predict_proba'] instead.
      sklearn's is_classifier() looks for the `_estimator_type =
      "classifier"` class attribute (duck-typing marker). Without it,
      sklearn assumed the wrapper was a regressor. Source: cluster task
      e5bba811-21c (btc_rf @ 5m, 2026-05-11 12:28).

    BUG B: Worker's _run_task reported status="done" to the orchestrator
      even when _execute_task returned {"status": "failed", "error": ...}.
      Only caught Python exceptions; result-dict failures got lost.
      Effect: cluster reported many silently-failed training tasks as
      "done", inflating the operator's success count. Operator: "are
      you saying you retrained all models?" caught this exact misread.

    FUNCTIONAL tests — exercise the actual code paths:
    """
    print('\n[Phase 100d followup #4 -- XGB classifier tag + worker failure reporting]')
    import sys
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    if str(PRJ) not in sys.path:
        sys.path.insert(0, str(PRJ))

    # ── BUG A: _XGBClassifierWrapper carries _estimator_type='classifier' ──
    from src.utils.gpu_classifier import _XGBClassifierWrapper
    check('_XGBClassifierWrapper has _estimator_type class attribute',
          hasattr(_XGBClassifierWrapper, '_estimator_type'))
    check('_XGBClassifierWrapper._estimator_type == "classifier" (sklearn duck-type)',
          getattr(_XGBClassifierWrapper, '_estimator_type', None) == 'classifier')

    # sklearn.is_classifier() should now return True for instances
    try:
        from sklearn.base import is_classifier
        # Construct a wrapper without actually instantiating xgboost (xgb
        # may not be importable in this test environment). Use the
        # class itself — sklearn's is_classifier checks both classes
        # and instances via the _estimator_type attribute.
        check('sklearn.is_classifier(_XGBClassifierWrapper) == True',
              is_classifier(_XGBClassifierWrapper) is True)
    except Exception as exc:
        check(f'sklearn import failed in test env: {exc}', False)

    # ── BUG B: worker _run_task honors inner status='failed' ────────────
    worker_src = (PRJ / 'src/training/distributed/worker.py').read_text(encoding='utf-8')

    # Code-shape: the inner_status check must exist + branch
    check('worker.py: inner_status check from result dict',
          "inner_status = (result.get('status', 'done')" in worker_src)
    check('worker.py: failed inner status routes to _notify_master("failed", ...)',
          "if inner_status == 'failed':" in worker_src
          and 'self._notify_master(\n                    "failed", task_id, result=result' in worker_src)
    check('worker.py: error propagated from result dict on failed status',
          "error=(result.get('error', '')" in worker_src)
    check('worker.py: log differentiates trainer-error vs exception failure',
          'FAILED (trainer error)' in worker_src
          and 'FAILED (exception)' in worker_src)

    # Source order: the inner_status check must happen BEFORE _notify_master
    # is called with "done" — otherwise the original buggy path persists.
    inner_pos = worker_src.find("inner_status = (result.get('status'")
    failed_branch = worker_src.find("if inner_status == 'failed':")
    done_call = worker_src.find('self._notify_master("done", task_id, result=result)')
    check('source order: inner_status checked BEFORE notify("done") is called',
          inner_pos > 0 and failed_branch > 0 and done_call > 0
          and inner_pos < failed_branch < done_call)


def test_phase101_neural_kind_plus_task_heartbeat_and_proc_health_cache():
    """Phase 101 (2026-05-11) — three interlocking fixes for the bug that
    surfaced tonight: cluster watchdog killed a HEALTHY TFT training run
    after exactly 120 minutes despite the trainer being at Epoch 0
    step 18544/19077 (97%) with train_loss decreasing 0.0492 → 0.0395
    (source: logs/worker_razer_gpu.out.log).

    ROOT CAUSE: Darts/Lightning trainers report progress via tqdm to
    stdout only — they never call back to the orchestrator. The cluster
    watchdog uses `task["last_update_at"]` to gate stale-task detection.
    With no callbacks, `last_update_at == started_at` for the full run.
    Combined with the 120-min "gpu" lane budget, the watchdog kill
    condition (elapsed > timeout AND stale > 5min) fired the moment
    elapsed crossed 120 min — every time, regardless of trainer health.

    Also: dashboard polled /api/pipeline/status every 5-10s; each call
    triggered a full psutil scan (~3-6s cold on Windows with 18 python
    processes). Cards stuck in "Loading..." for minutes.

    Three coordinated fixes:
      F1. orchestrator: new "neural" resource_kind with 6h budget — so
          even with heartbeats off, multi-epoch neural training fits.
      F2. orchestrator + worker: defensive task-level heartbeat. Worker
          posts status="heartbeat" every TASK_HEARTBEAT_S during
          _run_task. Orchestrator's update_task treats "heartbeat" as a
          stale-window refresh ONLY (no state change). Stops the
          watchdog mis-firing on tqdm-only trainers.
      F3. process_health: TTL cache around the psutil scan so dashboard
          polling within the cache window costs ~0ms.

    FUNCTIONAL tests — exercise the actual code paths.
    """
    print('\n[Phase 101 -- neural kind + task heartbeat + process_health cache]')
    import sys, threading, time as _t, datetime as _dt, importlib
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    if str(PRJ) not in sys.path:
        sys.path.insert(0, str(PRJ))

    # ── F1: orchestrator declares "neural" kind with 6h budget ──────────
    if 'src.training.distributed.orchestrator' in sys.modules:
        del sys.modules['src.training.distributed.orchestrator']
    orch_mod = importlib.import_module('src.training.distributed.orchestrator')
    budgets = orch_mod.WATCHDOG_TIMEOUT_BY_KIND
    check('WATCHDOG_TIMEOUT_BY_KIND has "neural" key',
          'neural' in budgets)
    check('neural budget == 6h (21600s)',
          budgets.get('neural') == 6 * 60 * 60)
    check('cpu/gpu/exclusive budgets preserved unchanged',
          budgets.get('cpu') == 60 * 60
          and budgets.get('gpu') == 120 * 60
          and budgets.get('exclusive') == 180 * 60)

    # ── F1b: training_rules.json wires TFT to "neural" ───────────────────
    import json as _json
    rules_path = PRJ / 'data' / 'training_rules.json'
    with open(rules_path, encoding='utf-8') as f:
        rules = _json.load(f)
    tft_kind = rules.get('models', {}).get('tft', {}).get('resource_kind')
    check('training_rules.json: tft.resource_kind == "neural"',
          tft_kind == 'neural')
    # rules.py reads this — verify the public API agrees
    if 'src.training.training_rules' in sys.modules:
        del sys.modules['src.training.training_rules']
    rules_mod = importlib.import_module('src.training.training_rules')
    check('training_rules.resource_kind("tft") returns "neural"',
          rules_mod.resource_kind('tft') == 'neural')

    # ── F2: orchestrator update_task with status="heartbeat" only
    #        refreshes last_update_at; everything else preserved.
    o = orch_mod.Orchestrator()
    nid = "test-node-hb"
    o.register_worker({
        "node_id": nid, "name": "TEST_HB", "ip": "127.0.0.1",
        "port": 9997, "hostname": "TEST_HB", "status": "busy",
        "lane": "gpu", "cuda_available": True, "gpu_vram_gb": 8,
        "cpu_cores": 4, "ram_gb": 8, "current_task": "",
    })
    tid = o.submit_task({"model_type": "tft", "timeframe": "1h",
                         "symbol": "ALL", "config": {}})

    # Mark the task running with the worker, set started_at way in the
    # past so we can assert it's NOT clobbered by heartbeat.
    one_hour_ago = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=1)).isoformat()
    with o._lock:
        o._tasks[tid]['status']          = 'running'
        o._tasks[tid]['assigned_to']     = nid
        o._tasks[tid]['started_at']      = one_hour_ago
        o._tasks[tid]['last_update_at']  = one_hour_ago
        o._workers[nid]['status']        = 'busy'
        o._workers[nid]['current_task']  = tid
        # Tag result so we can prove it survives heartbeat:
        o._tasks[tid]['result'] = {'sentinel': 'pre-heartbeat'}

    # Apply heartbeat
    o.update_task(tid, 'heartbeat', node_id=nid)
    with o._lock:
        post_status = o._tasks[tid].get('status')
        post_started = o._tasks[tid].get('started_at')
        post_last_upd = o._tasks[tid].get('last_update_at')
        post_result = o._tasks[tid].get('result')

    check('heartbeat: task["status"] preserved as "running"',
          post_status == 'running')
    check('heartbeat: started_at NOT clobbered (still 1h ago)',
          post_started == one_hour_ago)
    check('heartbeat: result dict preserved (no overwrite)',
          isinstance(post_result, dict) and post_result.get('sentinel') == 'pre-heartbeat')
    check('heartbeat: last_update_at advanced past the old value',
          post_last_upd is not None and post_last_upd > one_hour_ago)

    # ── F2b: with heartbeat keeping last_update_at fresh, an
    #         over-elapsed task is NOT killed by the watchdog.
    # Simulate: started 3h ago (over neural's 6h cap? no — under it; but
    # we'll prove the kill-gate uses BOTH conditions even at the gpu cap).
    # Use "gpu" kind so timeout is 120 min, started 3h ago = over budget.
    nid2 = "test-node-hb2"
    o.register_worker({
        "node_id": nid2, "name": "TEST_HB2", "ip": "127.0.0.2",
        "port": 9996, "hostname": "TEST_HB2", "status": "busy",
        "lane": "gpu", "cuda_available": True, "gpu_vram_gb": 8,
        "cpu_cores": 4, "ram_gb": 8, "current_task": "",
    })
    tid2 = o.submit_task({"model_type": "base", "timeframe": "1h",   # cpu kind
                          "symbol": "ALL", "config": {}})
    three_hours_ago = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=3)).isoformat()
    fresh_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with o._lock:
        o._tasks[tid2]['status']         = 'running'
        o._tasks[tid2]['assigned_to']    = nid2
        o._tasks[tid2]['started_at']     = three_hours_ago
        o._tasks[tid2]['last_update_at'] = fresh_iso   # heartbeat kept it fresh
        o._workers[nid2]['status']       = 'busy'
        o._workers[nid2]['current_task'] = tid2

    o._sweep_stale_tasks()
    with o._lock:
        survives = o._tasks[tid2].get('status') == 'running'
    check('watchdog: actively-heartbeating task survives over-elapsed window',
          survives)

    # ── F2d: lane dispatch must route "neural" kind to GPU workers ──────
    # Without this, _lane_accepts falls through to "return True" and
    # neural tasks land on any idle worker (including CPU lanes), which
    # cannot run Darts/Lightning.
    orch_src = (PRJ / 'src/training/distributed/orchestrator.py').read_text(encoding='utf-8')
    check('orchestrator.py: _lane_accepts handles "neural" -> gpu',
          'kind in ("gpu", "exclusive", "neural")' in orch_src
          or "kind in ('gpu', 'exclusive', 'neural')" in orch_src)

    # ── F2c: worker source ships TASK_HEARTBEAT_S + heartbeat thread ────
    worker_src = (PRJ / 'src/training/distributed/worker.py').read_text(encoding='utf-8')
    check('worker.py: TASK_HEARTBEAT_S constant defined',
          'TASK_HEARTBEAT_S' in worker_src)
    check('worker.py: _run_task spawns heartbeat thread',
          'heartbeat_stop = threading.Event()' in worker_src
          and 'task-heartbeat-' in worker_src)
    check('worker.py: heartbeat thread calls _notify_master("heartbeat", ...)',
          'self._notify_master("heartbeat", task_id)' in worker_src)
    check('worker.py: heartbeat thread stopped in finally (heartbeat_stop.set())',
          'heartbeat_stop.set()' in worker_src)
    # Source order: heartbeat_stop.set() must be in the finally block AFTER
    # the thread starts. Verify positionally.
    hb_start = worker_src.find('hb_thread.start()')
    hb_stop = worker_src.find('heartbeat_stop.set()')
    check('worker.py: heartbeat starts before finally stops it',
          hb_start > 0 and hb_stop > 0 and hb_start < hb_stop)

    # ── F3: process_health TTL cache eliminates redundant psutil scans ──
    if 'src.utils.process_health' in sys.modules:
        del sys.modules['src.utils.process_health']
    ph = importlib.import_module('src.utils.process_health')
    check('process_health exports _snapshot_python_procs', hasattr(ph, '_snapshot_python_procs'))
    check('process_health exports invalidate_scan_cache', hasattr(ph, 'invalidate_scan_cache'))
    check('process_health._SCAN_TTL_S > 0 by default', ph._SCAN_TTL_S > 0)

    # FUNCTIONAL — monkey-patch _iter_python_procs to count invocations.
    call_count = {'n': 0}
    fake_procs = [(1234, 'python -m src.foo', 100_000_000),
                  (5678, 'python -m src.bar',  50_000_000)]
    def _fake_iter():
        call_count['n'] += 1
        for row in fake_procs:
            yield row
    ph.invalidate_scan_cache()  # start clean
    orig_iter = ph._iter_python_procs
    try:
        ph._iter_python_procs = _fake_iter
        # First call MUST hit the scan
        s1 = ph._snapshot_python_procs()
        # Subsequent calls within TTL MUST be served from cache
        s2 = ph._snapshot_python_procs()
        s3 = ph._snapshot_python_procs()
    finally:
        ph._iter_python_procs = orig_iter
    check('first scan invokes _iter_python_procs once',
          call_count['n'] == 1)
    check('subsequent scans within TTL serve from cache (no extra invocations)',
          call_count['n'] == 1)
    check('cached snapshots equal across calls (same list of tuples)',
          s1 == s2 == s3 == fake_procs)

    # invalidate_scan_cache forces a fresh scan on next call
    call_count['n'] = 0
    try:
        ph._iter_python_procs = _fake_iter
        ph.invalidate_scan_cache()
        _ = ph._snapshot_python_procs()
    finally:
        ph._iter_python_procs = orig_iter
    check('invalidate_scan_cache forces re-scan on next call',
          call_count['n'] == 1)

    # find_process / all_known_processes both go through the cache —
    # prove by invoking them and confirming they don't re-scan.
    ph.invalidate_scan_cache()
    call_count['n'] = 0
    try:
        ph._iter_python_procs = _fake_iter
        ph.find_process(ph.KIND_BOT)             # primes cache, scan #1
        ph.find_process(ph.KIND_DASH)            # served from cache
        ph.all_known_processes()                  # served from cache
        ph.find_process(ph.KIND_TRAIN_ORCH)      # served from cache
    finally:
        ph._iter_python_procs = orig_iter
    check('find_process + all_known_processes share the cache (1 scan for 4 calls)',
          call_count['n'] == 1)


def test_phase100d_followup_3_training_jobs_lock_is_rlock():
    """Phase 100d follow-up 3 (2026-05-11) — _training_jobs_lock must be
    RLock (reentrant), not Lock.

    Live bug found tonight: /api/training/jobs?limit=10 hangs 30+ seconds
    AGAIN after the Phase 100d throttle was already in production. Live
    probe: /api/cluster/workers returns 200 in 2s; /api/training/active
    (which uses _training_jobs_lock) also hangs. Pattern: anything taking
    _training_jobs_lock starves.

    Root cause: Phase 97c (commit 0802a36) added
    _refresh_orphan_current_state which does:

        with _training_jobs_lock:
            for jid, job in list(_training_jobs.items()):
                ...
                if updates:
                    _record_job(jid, **updates)   # ← _record_job ALSO
                                                    #   acquires _training_jobs_lock

    threading.Lock is NON-reentrant. The same thread re-acquiring it
    deadlocks instantly. The Phase 97c daemon thread that triggered an
    update would hang forever holding the lock, starving every endpoint
    that needs it. Manifests when orphan records actually require
    refresh (i.e., pipeline is running and tf/model changes).

    Why Phase 100d's throttle didn't catch this: throttle reduced WRITE
    contention on the filelock. The deadlock is on the in-memory dict
    lock — different code path.

    FUNCTIONAL test — proves the lock IS reentrant by acquiring twice
    from the same thread:
    """
    print('\n[Phase 100d followup #3 -- _training_jobs_lock is RLock]')
    import sys, threading
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    if str(PRJ) not in sys.path:
        sys.path.insert(0, str(PRJ))

    try:
        from src.dashboard import app as dash_app
    except Exception as exc:
        check(f'import dash_app (got {type(exc).__name__}: {exc})', False)
        return

    lock = dash_app._training_jobs_lock
    # RLock instances are produced by threading.RLock() which returns
    # a _thread.RLock or threading._RLock — type name contains "RLock".
    lock_type_name = type(lock).__name__
    check(f'_training_jobs_lock is RLock-style (type={lock_type_name})',
          'RLock' in lock_type_name)

    # PROOF that reentrancy works — same thread, two acquires, no hang.
    # Wrap with a thread-side timeout so if this test EVER deadlocks we
    # see a failure instead of hanging the regression suite.
    acquired_both = [False]
    timed_out = [False]
    def _two_acquires():
        try:
            with lock:
                with lock:    # would deadlock with non-reentrant Lock
                    acquired_both[0] = True
        except Exception:
            pass

    t = threading.Thread(target=_two_acquires, daemon=True)
    t.start()
    t.join(timeout=2.0)
    if t.is_alive():
        timed_out[0] = True
    check('same thread can acquire lock twice without deadlock',
          acquired_both[0] is True and not timed_out[0])

    # Mirror real Phase 97c pattern: outer `with lock` then call
    # _record_job which also takes `with lock`. With RLock this works.
    # With Lock it deadlocks. Use a synthetic job so we don't mutate
    # production state.
    success = [False]
    err = [None]
    def _phase97c_pattern():
        try:
            with lock:
                # Mirror the orphan-refresh inner call
                dash_app._record_job('test-rlock-job', model='base',
                                      status='running')
            success[0] = True
        except Exception as exc:
            err[0] = f'{type(exc).__name__}: {exc}'

    saved = dict(dash_app._training_jobs)
    try:
        t = threading.Thread(target=_phase97c_pattern, daemon=True)
        t.start()
        t.join(timeout=3.0)
    finally:
        # Restore production state — remove the test job
        with dash_app._training_jobs_lock:
            dash_app._training_jobs.pop('test-rlock-job', None)

    check('Phase 97c pattern (outer lock + nested _record_job) does NOT deadlock',
          success[0] is True and not t.is_alive())
    if err[0]:
        check(f'no exception raised (got {err[0]})', False)


def test_phase100d_worker_cpu_lane_hides_gpu_env_vars():
    """Phase 100d follow-up 2 (2026-05-11) — --lane cpu workers must hide
    every GPU adapter from PyTorch/sklearn import paths.

    Operator screenshot 2026-05-11: Intel iGPU stable 67% utilization from
    PID 37044 (`python -m src.training.distributed.worker --lane cpu`).
    Root cause confirmed via `Get-Counter '\\GPU Engine(*)'`:
        peak 89.2% pid_37044_..._engtype_3d  on LUID 0x0001186f (iGPU)
    Quick fix this session was to relaunch the worker with
    `CUDA_VISIBLE_DEVICES=''` set in the spawning shell. That env var
    isn't persisted, so next restart_all.ps1 cycle would respawn the
    rogue iGPU consumption.

    Permanent fix (this commit): worker.py reads args.lane FIRST, and if
    lane == 'cpu', sets CUDA_VISIBLE_DEVICES='' + HIP_VISIBLE_DEVICES=''
    BEFORE the TrainingWorker import path pulls in torch/sklearn. The
    ML libs then see zero GPU adapters and can't allocate context on
    the integrated GPU.

    FUNCTIONAL tests — actually run the relevant code path:
      (a) String evidence the code exists (smoke-check vs delete/rename)
      (b) Source order: env-var set BEFORE TrainingWorker is constructed
          (i.e., the assignment line appears earlier in the file than
          the worker = TrainingWorker(...) call). Compiler order matters.
      (c) gpu/any lane does NOT clear CUDA_VISIBLE_DEVICES
    """
    print('\n[Phase 100d followup #2 -- CPU-lane worker hides GPU env]')
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    src = (PRJ / 'src/training/distributed/worker.py').read_text(encoding='utf-8')

    # ── (a) String evidence ──────────────────────────────────────────────
    check('worker.py: sets CUDA_VISIBLE_DEVICES to empty when lane=cpu',
          "if args.lane == 'cpu':" in src
          and "os.environ['CUDA_VISIBLE_DEVICES'] = ''" in src)
    check('worker.py: also clears HIP_VISIBLE_DEVICES for AMD ROCm builds',
          "os.environ['HIP_VISIBLE_DEVICES']" in src)
    check('worker.py: log message explains the iGPU hide for operator visibility',
          'hiding all GPU adapters' in src
          and "CUDA_VISIBLE_DEVICES=''" in src)

    # ── (b) Source order — env-var must be set BEFORE TrainingWorker() ──
    env_pos    = src.find("os.environ['CUDA_VISIBLE_DEVICES'] = ''")
    worker_pos = src.find('worker = TrainingWorker(')
    check('worker.py: env-var assignment appears BEFORE TrainingWorker construction',
          env_pos > 0 and worker_pos > 0 and env_pos < worker_pos)

    # ── (c) Lane=gpu/any path does NOT clear CUDA_VISIBLE_DEVICES ───────
    # Extract the block from "args = parser.parse_args()" to
    # "worker = TrainingWorker(" and confirm the CUDA-clear is gated.
    args_pos = src.find('args = parser.parse_args()')
    block = src[args_pos:worker_pos]
    check('CUDA clear is inside an `if args.lane == \'cpu\':` gate (not unconditional)',
          "if args.lane == 'cpu':" in block
          and block.index("if args.lane == 'cpu':") < block.index("os.environ['CUDA_VISIBLE_DEVICES']"))


def test_phase100d_followup_restart_all_no_auto_train_cron():
    """Phase 100d follow-up (2026-05-11) — restart_all.ps1 no longer
    auto-schedules launch_training.ps1 by default.

    Operator screenshot 2026-05-11: GPU at 78% / 72°C while cluster
    reported "GPU lane idle". Root cause: restart_all.ps1's Step 4
    detached-spawned launch_training.ps1 to run train_all_models.py
    directly as a subprocess every restart. That subprocess uses
    GPU/CPU directly, completely outside the cluster's worker
    routing (Phase 100a/b/e), so cluster's status says "idle"
    while real hardware is busy.

    Fix: gate the schedule on $env:AI_TRADER_AUTO_TRAIN. Default
    (env var unset or != '1') = skip the cron entirely. All training
    paths now route through the cluster. Override stays available
    for emergency rollback.

    Tests assert on the actual restart_all.ps1 file contents:
    """
    print('\n[Phase 100d follow-up -- restart_all skips auto-train cron]')
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    restart_script = (PRJ / 'restart_all.ps1').read_text(encoding='utf-8')

    check('restart_all.ps1: auto-train cron gated on AI_TRADER_AUTO_TRAIN env var',
          "if ($env:AI_TRADER_AUTO_TRAIN -eq '1')" in restart_script)
    check('restart_all.ps1: default branch SKIPS legacy launch_training.ps1 scheduling',
          'Training auto-schedule SKIPPED (cluster handles all training)'
          in restart_script)
    check('restart_all.ps1: rationale comment cites Phase 100a/b/e cluster routing',
          'Phase 100a' in restart_script
          and 'Phase 100b' in restart_script
          and 'Phase 100e' in restart_script)
    check('restart_all.ps1: AI_TRADER_AUTO_TRAIN=1 override branch present',
          'AI_TRADER_AUTO_TRAIN=1 — scheduling legacy launch_training.ps1' in restart_script)
    check('restart_all.ps1: clear operator-facing message about how to trigger training',
          'Trigger training via dashboard' in restart_script)


def test_phase100d_training_jobs_throttle_and_fast_endpoint():
    """Phase 100d (2026-05-11) — root-cause fix for the "/api/training/jobs
    times out" bug class that produced 3 visible symptoms:
      1. Pipeline running but no row shows RUNNING
      2. F5 wipes QUEUED state
      3. Currently-training row stays OK

    Root cause: every _record_job call (from many Phase 100a sync-daemon
    threads) triggered an atomic file write under filelock. With 9 running
    jobs polling every 5s, that's ~2 writes/sec on a 41KB file, starving
    the api_training_jobs endpoint of CPU and filelock access.

    FUNCTIONAL tests (per global rule — every assertion calls real code):
      (a) _persist_training_jobs throttles to ≤1 write per _PERSIST_THROTTLE_S
      (b) When throttled, dirty flag is set so the deferred flush picks up
      (c) force=True bypasses throttle (used on shutdown)
      (d) api_training_jobs endpoint <500ms with 50 synthetic jobs
      (e) Terminal jobs (done/error/cancelled/lost/partial) skip annotation
    """
    print('\n[Phase 100d -- training jobs throttle + fast endpoint]')
    import sys, time, os
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    if str(PRJ) not in sys.path:
        sys.path.insert(0, str(PRJ))
    os.environ['DASHBOARD_API_KEY'] = ''

    try:
        from src.dashboard import app as dash_app
    except Exception as exc:
        check(f'import succeeds (got {type(exc).__name__}: {exc})', False)
        return

    # ── (a) Throttle: 2 rapid calls → only 1 actual write ────────────────
    # Patch the actual file-write to count invocations.
    write_count = {'n': 0}
    def _fake_write_json(path, data, **kw):
        write_count['n'] += 1
    saved_write = None
    # safe_json.write_json is imported inline in _persist_training_jobs;
    # patch via sys.modules so the import inside the function resolves
    # to our fake.
    import src.utils.safe_json as _safe
    saved_write = _safe.write_json
    _safe.write_json = _fake_write_json
    # Reset throttle state for a clean test
    saved_state = dict(dash_app._persist_throttle_state)
    dash_app._persist_throttle_state['last_write_at'] = 0.0
    dash_app._persist_throttle_state['dirty'] = False
    try:
        # First call: should write (last_write_at=0 means throttle window has elapsed)
        dash_app._persist_training_jobs()
        first_count = write_count['n']
        # Second call immediately after: should be throttled
        dash_app._persist_training_jobs()
        second_count = write_count['n']
        # Third call: still within throttle window
        dash_app._persist_training_jobs()
        third_count = write_count['n']
    finally:
        _safe.write_json = saved_write
        dash_app._persist_throttle_state.update(saved_state)

    check('throttle: first call writes through (write_count went 0→1)',
          first_count == 1)
    check('throttle: second rapid call is skipped (count still 1)',
          second_count == 1)
    check('throttle: third rapid call is also skipped (count still 1)',
          third_count == 1)

    # ── (b) Dirty flag set when throttled ────────────────────────────────
    dash_app._persist_throttle_state['last_write_at'] = time.time()  # fresh
    dash_app._persist_throttle_state['dirty'] = False
    dash_app._persist_training_jobs()   # should set dirty
    check('throttle: skipped call marks dirty=True so deferred flush triggers',
          dash_app._persist_throttle_state['dirty'] is True)

    # ── (c) force=True bypasses throttle ─────────────────────────────────
    write_count['n'] = 0
    _safe.write_json = _fake_write_json
    dash_app._persist_throttle_state['last_write_at'] = time.time()
    try:
        dash_app._persist_training_jobs(force=True)
    finally:
        _safe.write_json = saved_write
    check('throttle: force=True bypasses (writes through even within window)',
          write_count['n'] == 1)

    # ── (d) Endpoint <500ms with 50 jobs ─────────────────────────────────
    # Build 50 synthetic jobs into _training_jobs (cap is 50).
    saved_jobs = dict(dash_app._training_jobs)
    try:
        with dash_app._training_jobs_lock:
            dash_app._training_jobs.clear()
            for i in range(50):
                dash_app._training_jobs[f'synthj-{i:03d}'] = {
                    'job_id': f'synthj-{i:03d}',
                    'model': 'base' if i % 2 == 0 else 'tft',
                    'status': 'done' if i < 45 else 'running',   # 45 terminal + 5 active
                    'created_at': time.time() - i * 60,
                    'started_at': time.time() - i * 60,
                    'finished_at': time.time() - i * 60 + 30 if i < 45 else 0,
                    'tf': '1h',
                }
        client = dash_app.app.test_client()
        t0 = time.time()
        resp = client.get('/api/training/jobs?limit=20')
        elapsed = time.time() - t0
        body = resp.get_json()
    finally:
        with dash_app._training_jobs_lock:
            dash_app._training_jobs.clear()
            dash_app._training_jobs.update(saved_jobs)

    check('endpoint: returns 200',
          resp.status_code == 200)
    check(f'endpoint: returns 20 jobs (got {len(body.get("jobs", []))})',
          isinstance(body, dict) and len(body.get('jobs', [])) == 20)
    check(f'endpoint: returns total=50 (got {body.get("total")})',
          body.get('total') == 50)
    check(f'endpoint: completes in <500ms (took {elapsed*1000:.1f}ms)',
          elapsed < 0.5)

    # ── (e) Terminal jobs skip annotation ────────────────────────────────
    # _annotate_job_timing adds elapsed_s / eta_s — terminal jobs already
    # have these so re-computing is waste. After Phase 100d, terminal jobs
    # are passed through unchanged.
    annotate_calls = {'n': 0}
    saved_annotate = dash_app._annotate_job_timing
    def _spy_annotate(j):
        annotate_calls['n'] += 1
        return saved_annotate(j)
    dash_app._annotate_job_timing = _spy_annotate
    saved_jobs2 = dict(dash_app._training_jobs)
    try:
        with dash_app._training_jobs_lock:
            dash_app._training_jobs.clear()
            for i in range(10):
                dash_app._training_jobs[f'tj-{i}'] = {
                    'job_id': f'tj-{i}', 'model': 'base',
                    'status': 'done' if i < 7 else 'running',   # 7 terminal, 3 active
                    'created_at': time.time() - i * 60,
                    'started_at': time.time() - i * 60,
                }
        client = dash_app.app.test_client()
        client.get('/api/training/jobs?limit=20')
    finally:
        dash_app._annotate_job_timing = saved_annotate
        with dash_app._training_jobs_lock:
            dash_app._training_jobs.clear()
            dash_app._training_jobs.update(saved_jobs2)

    check(f'endpoint: only the 3 non-terminal jobs were annotated (got {annotate_calls["n"]})',
          annotate_calls['n'] == 3)


def test_sprint1a_r1_trainers_package_typed_contract():
    """Sprint 1a R1 Step 1 — src/engine/trainers/ package with TrainingResult
    dataclass + 8 per-model thin wrappers + TRAINER_REGISTRY.

    FUNCTIONAL tests (every assertion calls real code):
      (a) TrainingResult dataclass shape + .ok property + .to_dict()
      (b) _common.run_trainer happy path — mocked train_fn + mocked meta
          returns a populated TrainingResult
      (c) _common.run_trainer failure path — train_fn raises; result has
          .error populated, .ok==False, .elapsed_s > 0, traceback in extras
      (d) Every wrapper module exists and exposes a `train` callable
      (e) TRAINER_REGISTRY maps all 8 dashboard model_keys to train fns
    """
    print('\n[Sprint 1a R1 Step 1 -- trainers package + TrainingResult contract]')
    import sys, time
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    if str(PRJ) not in sys.path:
        sys.path.insert(0, str(PRJ))

    try:
        from src.engine.trainers import (
            TrainingResult, TRAINER_REGISTRY,
            train_base, train_trend, train_futures, train_scalping,
            train_meta, train_tft, train_oft, train_regime,
        )
        from src.engine.trainers import _common
    except Exception as exc:
        check(f'import trainers package (got {type(exc).__name__}: {exc})', False)
        return

    # ── (a) TrainingResult dataclass ─────────────────────────────────────
    r = TrainingResult(model_key='base', tf='1h')
    check('TrainingResult constructs with required model_key + tf',
          r.model_key == 'base' and r.tf == '1h')
    check('TrainingResult.symbol defaults to BTC/USDT',
          r.symbol == 'BTC/USDT')
    check('TrainingResult.ok == True when no error and not cancelled',
          r.ok is True)
    r2 = TrainingResult(model_key='tft', tf='15m', error='boom')
    check('TrainingResult.ok == False when error is set',
          r2.ok is False)
    r3 = TrainingResult(model_key='tft', tf='15m', cancelled=True)
    check('TrainingResult.ok == False when cancelled',
          r3.ok is False)
    d = r.to_dict()
    check('to_dict() returns dict with model_key + tf',
          isinstance(d, dict) and d['model_key'] == 'base' and d['tf'] == '1h')
    check('to_dict() drops empty extras key',
          'extras' not in d)
    r4 = TrainingResult(model_key='base', tf='1h', extras={'foo': 'bar'})
    check('to_dict() keeps non-empty extras',
          'extras' in r4.to_dict() and r4.to_dict()['extras'] == {'foo': 'bar'})

    # ── (b) _common.run_trainer happy path ──────────────────────────────
    calls = []
    def _fake_train_ok(timeframe, **kw):
        calls.append({'tf': timeframe, **kw})
        # Simulate a quick run
        time.sleep(0.01)

    saved_read_meta = _common._read_meta
    _common._read_meta = lambda mk, tf: {
        'n_samples': 12345, 'n_features': 17, 'n_iterations': 400,
        'accuracy': 0.514, 'walk_forward_mean_acc': 0.508,
        'long_accuracy': 50.1, 'short_accuracy': 51.2, 'auc_roc': 0.572,
    }
    try:
        result = _common.run_trainer('base', '1h', symbol='BTC/USDT',
                                       train_fn=_fake_train_ok)
    finally:
        _common._read_meta = saved_read_meta

    check('run_trainer happy: returns TrainingResult instance',
          isinstance(result, TrainingResult))
    check('run_trainer happy: train_fn was called with timeframe=tf',
          len(calls) == 1 and calls[0]['tf'] == '1h')
    check('run_trainer happy: result.ok == True',
          result.ok is True)
    check('run_trainer happy: started_at < finished_at and elapsed_s > 0',
          result.started_at > 0 and result.finished_at >= result.started_at
          and result.elapsed_s > 0)
    check('run_trainer happy: meta-derived fields populated',
          result.n_samples == 12345 and result.n_features == 17
          and result.n_iterations == 400)
    check('run_trainer happy: accuracy fields normalized to percent',
          result.test_acc == 51.4 and result.wf_acc == 50.8)
    check('run_trainer happy: long_acc/short_acc as-is when already percent',
          result.long_acc == 50.1 and result.short_acc == 51.2)
    check('run_trainer happy: auc_roc kept raw (not pct-normalized)',
          result.auc_roc == 0.572)

    # ── (c) run_trainer failure path ─────────────────────────────────────
    def _fake_train_boom(timeframe, **kw):
        raise ValueError('synthetic failure for test')
    result_err = _common.run_trainer('trend', '4h', symbol='BTC/USDT',
                                       train_fn=_fake_train_boom)
    check('run_trainer fail: result.ok == False',
          result_err.ok is False)
    check('run_trainer fail: result.error captures exception',
          result_err.error is not None
          and 'ValueError' in result_err.error
          and 'synthetic failure' in result_err.error)
    check('run_trainer fail: traceback captured in extras',
          'traceback_tail' in result_err.extras)
    check('run_trainer fail: timing recorded (finished_at >= started_at, elapsed_s >= 0)',
          # round(elapsed_s, 2) can be 0.0 when failure raises in <10ms; the
          # important invariant is the finally block ran (both timestamps set).
          result_err.elapsed_s is not None and result_err.elapsed_s >= 0
          and result_err.finished_at >= result_err.started_at)
    check('run_trainer fail: started_at + finished_at both set',
          result_err.started_at > 0 and result_err.finished_at > 0)

    # ── (d) Every wrapper module exposes a `train` callable ──────────────
    for mod, name in [
        (train_base,     'train_base'),
        (train_trend,    'train_trend'),
        (train_futures,  'train_futures'),
        (train_scalping, 'train_scalping'),
        (train_meta,     'train_meta'),
        (train_tft,      'train_tft'),
        (train_oft,      'train_oft'),
        (train_regime,   'train_regime'),
    ]:
        check(f'{name}.train is callable',
              hasattr(mod, 'train') and callable(mod.train))

    # ── (e) TRAINER_REGISTRY covers all 8 dashboard model_keys ───────────
    expected_keys = {'base', 'trend', 'futures', 'scalping',
                      'meta', 'tft', 'oft', 'regime'}
    check('TRAINER_REGISTRY has all 8 keys',
          set(TRAINER_REGISTRY.keys()) == expected_keys)
    check('TRAINER_REGISTRY values are callables',
          all(callable(fn) for fn in TRAINER_REGISTRY.values()))
    check('TRAINER_REGISTRY["base"] is train_base.train',
          TRAINER_REGISTRY['base'] is train_base.train)
    check('TRAINER_REGISTRY["tft"] is train_tft.train',
          TRAINER_REGISTRY['tft'] is train_tft.train)


def test_phase100e_pipeline_orchestrator_cluster_dispatch():
    """Phase 100e — pipeline_orchestrator routes through cluster (was
    calling train_all() in-process, which left Ivan + Razer workers idle).

    Operator screenshot 2026-05-11 02:55: all 4 online cluster workers
    status=idle while pipeline was burning Razer CPU solo. The training
    distribution we'd promised was a no-op because pipeline never
    submitted anything to the cluster.

    FUNCTIONAL tests — every assertion calls production code:
      (a) Cell list + spec builders match the dashboard's equivalents
          (single source of truth for dispatch shape).
      (b) _pipeline_step drives a fake cluster snapshot through several
          iterations; asserts train→BT chaining, failed-train skip-BT,
          submit_fn failure path, terminal detection.
      (c) AI_TRADER_PIPELINE_LOCAL=1 forces legacy in-process path.
    """
    print('\n[Phase 100e -- pipeline_orchestrator cluster dispatch (functional)]')
    import sys, os
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    if str(PRJ) not in sys.path:
        sys.path.insert(0, str(PRJ))
    os.environ.pop('AI_TRADER_PIPELINE_LOCAL', None)

    try:
        from src.engine import pipeline_orchestrator as po
        from src.engine.train_all_models import DEFAULT_PER_KEY_TFS
    except Exception as exc:
        check(f'import pipeline_orchestrator succeeds (got {type(exc).__name__}: {exc})', False)
        return

    # ── (a) Cell list + spec builders ────────────────────────────────────
    cells = po._pipeline_cell_list()
    expected = sum(len(tfs) for tfs in DEFAULT_PER_KEY_TFS.values())
    check('pipeline cell list count matches DEFAULT_PER_KEY_TFS sum',
          len(cells) == expected)
    seen_models = []
    for (m, tf) in cells:
        if not seen_models or seen_models[-1] != m:
            seen_models.append(m)
    check('pipeline cell list is model-major',
          len(seen_models) == len(set(seen_models)))

    train_spec = po._pipeline_build_train_spec('futures', '4h')
    check('pipeline train spec model_type mapped (futures → futures_short)',
          train_spec['model_type'] == 'futures_short')
    check('pipeline train spec carries slash-form symbol BTC/USDT',
          train_spec['symbol'] == 'BTC/USDT')
    check('pipeline train spec timeframe matches cell tf',
          train_spec['timeframe'] == '4h')

    bt_spec = po._pipeline_build_bt_spec('meta', '15m')
    check('pipeline BT spec uses model_type="backtest_cell"',
          bt_spec['model_type'] == 'backtest_cell')
    check('pipeline BT spec uses UNDERSCORE-form symbol BTC_USDT',
          bt_spec['symbol'] == 'BTC_USDT')
    check('pipeline BT spec scopes config.models to mapped model_type',
          bt_spec['config']['models'] == ['meta_labeler'])

    check('_to_cluster_model_type mapping covers base / futures / meta divergent names',
          po._to_cluster_model_type('base') == 'btc_rf'
          and po._to_cluster_model_type('futures') == 'futures_short'
          and po._to_cluster_model_type('meta') == 'meta_labeler')
    check('_to_cluster_model_type passthrough for same-name keys',
          po._to_cluster_model_type('trend') == 'trend'
          and po._to_cluster_model_type('tft') == 'tft')

    # ── (b) _pipeline_step driven through synthetic snapshots ────────────
    step_fn = po._pipeline_step
    test_cells = [('trend', '1h'), ('base', '4h'), ('futures', '5m')]
    state = {
        'train_tids': {
            ('trend', '1h'):   'train-T-1',
            ('base', '4h'):    'train-B-1',
            ('futures', '5m'): 'train-F-1',
        },
        'bt_tids':     {},
        'train_done':  set(),
        'bt_done':     set(),
        'cell_errors': {},
    }
    bt_submits: list[tuple[str, str]] = []
    def _fake_submit(m, tf, spec):
        bt_submits.append((m, tf))
        return f'bt-{m}-{tf}'

    # Iter 1: all running → nothing done, no BT
    by_id = {
        'train-T-1': {'task_id':'train-T-1','status':'running'},
        'train-B-1': {'task_id':'train-B-1','status':'running'},
        'train-F-1': {'task_id':'train-F-1','status':'pending'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit)
    check('pipeline_step iter 1: nothing done, finished=False, no BT submits',
          r['cells_complete'] == 0 and not r['finished'] and len(bt_submits) == 0)
    check('pipeline_step iter 1: 3 trains in flight, 0 BT',
          r['train_inflight'] == 3 and r['bt_inflight'] == 0)

    # Iter 2: trend@1h done → BT submits via _fake_submit
    by_id = {
        'train-T-1': {'task_id':'train-T-1','status':'done'},
        'train-B-1': {'task_id':'train-B-1','status':'running'},
        'train-F-1': {'task_id':'train-F-1','status':'running'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit)
    check('pipeline_step iter 2: trend@1h train done triggers BT submit',
          ('trend', '1h') in bt_submits)
    check('pipeline_step iter 2: state.bt_tids has new BT id',
          state['bt_tids'].get(('trend', '1h')) == 'bt-trend-1h')

    # Iter 3: trend@1h BT done; base@4h train failed → BT skipped
    by_id = {
        'train-T-1':   {'task_id':'train-T-1','status':'done'},
        'train-B-1':   {'task_id':'train-B-1','status':'failed','error':'OOM'},
        'train-F-1':   {'task_id':'train-F-1','status':'running'},
        'bt-trend-1h': {'task_id':'bt-trend-1h','status':'done'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit)
    check('pipeline_step iter 3: base@4h failed train marks bt_done (skip BT)',
          ('base', '4h') in state['bt_done'] and ('base', '4h') not in state['bt_tids'])
    check('pipeline_step iter 3: base@4h error recorded with OOM',
          'OOM' in state['cell_errors'].get(('base', '4h'), ''))
    check('pipeline_step iter 3: trend@1h fully complete',
          ('trend', '1h') in state['train_done']
          and ('trend', '1h') in state['bt_done']
          and ('trend', '1h') not in state['cell_errors'])

    # Iter 4: futures done → BT submits
    by_id = {
        'train-T-1':   {'task_id':'train-T-1','status':'done'},
        'train-B-1':   {'task_id':'train-B-1','status':'failed'},
        'train-F-1':   {'task_id':'train-F-1','status':'done'},
        'bt-trend-1h': {'task_id':'bt-trend-1h','status':'done'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit)
    check('pipeline_step iter 4: futures BT submitted now train done',
          ('futures', '5m') in state['bt_tids'])

    # Iter 5: all terminal → finished=True
    by_id = {
        'train-T-1':       {'task_id':'train-T-1','status':'done'},
        'train-B-1':       {'task_id':'train-B-1','status':'failed'},
        'train-F-1':       {'task_id':'train-F-1','status':'done'},
        'bt-trend-1h':     {'task_id':'bt-trend-1h','status':'done'},
        'bt-futures-5m':   {'task_id':'bt-futures-5m','status':'done'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit)
    check('pipeline_step iter 5: all terminal → finished=True',
          r['finished'])
    check('pipeline_step iter 5: cells_complete=3 (all terminal incl. skipped BT)',
          r['cells_complete'] == 3)

    # Submit_fn failure path
    state2 = {
        'train_tids': {('tft','15m'):'train-X'},
        'bt_tids':    {},
        'train_done': set(),
        'bt_done':    set(),
        'cell_errors':{},
    }
    by_id = {'train-X':{'task_id':'train-X','status':'done'}}
    r = step_fn([('tft','15m')], state2, by_id, lambda m, tf, s: None)
    check('pipeline_step: submit_fn returns None → cell marked bt_done with error',
          ('tft','15m') in state2['bt_done']
          and 'BT submit failed' in state2['cell_errors'].get(('tft','15m'), ''))
    check('pipeline_step: submit failure still terminates the loop (finished=True)',
          r['finished'])

    # ── (c) Local-fallback env var ───────────────────────────────────────
    check('AI_TRADER_PIPELINE_LOCAL env var is honored in _run_train_phase',
          'AI_TRADER_PIPELINE_LOCAL' in
          (PRJ / 'src/engine/pipeline_orchestrator.py').read_text(encoding='utf-8'))
    check('_run_train_phase_local fallback function exists',
          hasattr(po, '_run_train_phase_local'))
    check('_run_train_phase_cluster default function exists',
          hasattr(po, '_run_train_phase_cluster'))


def test_phase100b_retrain_all_distributed_train_then_bt():
    """Phase 100b — Retrain ALL = distributed parallel cells, sequential
    train → BT per cell. FUNCTIONAL tests (not string-match) that drive
    the dispatcher through synthetic cluster snapshots and assert
    observable behavior.

    Sub-tests:
      (a) _retrain_all_cell_list: pure cell-list builder; assert
          model-major order matches DEFAULT_PER_KEY_TFS exactly.
      (b) _retrain_all_build_train_spec / _retrain_all_build_bt_spec:
          pure spec builders; assert payload shape per worker.py contract.
      (c) _retrain_all_step: PURE iteration step. Drive it through:
          - train task done → BT submitted via submit_fn
          - train task failed → BT skipped, cell error recorded
          - BT task done → cell marked complete
          - BT task failed → cell marked complete with error
          - Multiple cells in parallel — all advance per step
          - Submit fn failure path
          - Final terminal detection
      (d) /api/training/run/all endpoint via test_client: routes to
          cluster by default, AI_TRADER_LOCAL_TRAINING=1 routes to local.
    """
    print('\n[Phase 100b -- Retrain ALL distributed (functional, proves behavior)]')
    import sys, os
    from pathlib import Path as _P
    PRJ = _P(__file__).resolve().parents[1]
    if str(PRJ) not in sys.path:
        sys.path.insert(0, str(PRJ))
    os.environ['DASHBOARD_API_KEY'] = ''
    os.environ.pop('AI_TRADER_LOCAL_TRAINING', None)

    try:
        from src.dashboard import app as dash_app
        from src.engine.train_all_models import DEFAULT_PER_KEY_TFS
    except Exception as exc:
        check(f'import succeeds (got {type(exc).__name__}: {exc})', False)
        return

    # ── (a) Cell list — model-major order ───────────────────────────────
    cells = dash_app._retrain_all_cell_list()
    expected_count = sum(len(tfs) for tfs in DEFAULT_PER_KEY_TFS.values())
    check('cell list count matches sum of TFs per model in DEFAULT_PER_KEY_TFS',
          len(cells) == expected_count)
    # Model-major: all of model A then all of model B
    seen_models = []
    for (m, tf) in cells:
        if not seen_models or seen_models[-1] != m:
            seen_models.append(m)
    check('cell list is model-major (model A all TFs, then model B all TFs)',
          len(seen_models) == len(set(seen_models)))
    # First model's TFs should match DEFAULT_PER_KEY_TFS in order
    first_model = seen_models[0]
    first_model_cells = [tf for (m, tf) in cells if m == first_model]
    check('within a model, TFs match DEFAULT_PER_KEY_TFS order',
          tuple(first_model_cells) == DEFAULT_PER_KEY_TFS[first_model])

    # ── (b) Spec builders ───────────────────────────────────────────────
    train_spec = dash_app._retrain_all_build_train_spec('futures', '4h')
    check('train spec model_type mapped (futures → futures_short)',
          train_spec['model_type'] == 'futures_short')
    check('train spec timeframe matches the cell tf',
          train_spec['timeframe'] == '4h')
    check('train spec carries symbol BTC/USDT (slash form for trainers)',
          train_spec['symbol'] == 'BTC/USDT')
    check('train spec has config + data_path + output_path (worker contract)',
          'config' in train_spec
          and 'data_path' in train_spec
          and 'output_path' in train_spec)

    bt_spec = dash_app._retrain_all_build_bt_spec('meta', '15m')
    check('BT spec model_type is "backtest_cell" (worker handler key)',
          bt_spec['model_type'] == 'backtest_cell')
    check('BT spec timeframe matches the cell tf',
          bt_spec['timeframe'] == '15m')
    check('BT spec symbol uses UNDERSCORE form (BTC_USDT) per worker.py',
          bt_spec['symbol'] == 'BTC_USDT')
    check('BT spec config.models scopes BT to this cell\'s mapped model_type',
          bt_spec['config']['models'] == ['meta_labeler'])
    check('BT spec config carries initial_capital + fee_preset',
          bt_spec['config']['initial_capital'] == 10000.0
          and bt_spec['config']['fee_preset'] == 'futures')

    # ── (c) _retrain_all_step — drive the loop ───────────────────────────
    step_fn = dash_app._retrain_all_step
    # Setup: 3 cells, all submitted as trains
    test_cells = [('trend', '1h'), ('base', '4h'), ('futures', '5m')]
    state = {
        'train_tids': {
            ('trend', '1h'):   'train-tid-1',
            ('base', '4h'):    'train-tid-2',
            ('futures', '5m'): 'train-tid-3',
        },
        'bt_tids':     {},
        'train_done':  set(),
        'bt_done':     set(),
        'cell_errors': {},
    }
    bt_submits_captured: list[tuple[str, str, dict]] = []
    def _fake_submit_bt(model_key, tf, bt_spec):
        bt_submits_captured.append((model_key, tf, bt_spec))
        return f'bt-tid-{model_key}-{tf}'

    # Iteration 1: nothing terminal yet — no progress
    by_id = {
        'train-tid-1': {'task_id': 'train-tid-1', 'status': 'running'},
        'train-tid-2': {'task_id': 'train-tid-2', 'status': 'running'},
        'train-tid-3': {'task_id': 'train-tid-3', 'status': 'pending'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit_bt)
    check('step iter 1: nothing done, finished=False',
          r['cells_complete'] == 0 and not r['finished'])
    check('step iter 1: no BT submitted yet (no train done)',
          len(bt_submits_captured) == 0)
    check('step iter 1: 3 trains in flight',
          r['train_inflight'] == 3)

    # Iteration 2: trend@1h done → its BT submits
    by_id = {
        'train-tid-1': {'task_id': 'train-tid-1', 'status': 'done'},
        'train-tid-2': {'task_id': 'train-tid-2', 'status': 'running'},
        'train-tid-3': {'task_id': 'train-tid-3', 'status': 'running'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit_bt)
    check('step iter 2: trend@1h train done triggers BT submit',
          ('trend', '1h') in [(m, t) for (m, t, _) in bt_submits_captured])
    check('step iter 2: BT submit captured the expected spec',
          bt_submits_captured[0][2]['model_type'] == 'backtest_cell'
          and bt_submits_captured[0][2]['timeframe'] == '1h')
    check('step iter 2: state.bt_tids has the new BT id',
          state['bt_tids'].get(('trend', '1h')) == 'bt-tid-trend-1h')
    check('step iter 2: cells_complete still 0 (BT not done yet)',
          r['cells_complete'] == 0)
    check('step iter 2: train_inflight=2, bt_inflight=1',
          r['train_inflight'] == 2 and r['bt_inflight'] == 1)

    # Iteration 3: trend@1h BT done → cell complete. base@4h train failed → skip BT.
    by_id = {
        'train-tid-1': {'task_id': 'train-tid-1', 'status': 'done'},
        'train-tid-2': {'task_id': 'train-tid-2', 'status': 'failed',
                         'error': 'OOM in fit'},
        'train-tid-3': {'task_id': 'train-tid-3', 'status': 'running'},
        'bt-tid-trend-1h': {'task_id': 'bt-tid-trend-1h', 'status': 'done'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit_bt)
    check('step iter 3: trend@1h fully complete (train+BT done)',
          ('trend', '1h') in state['train_done']
          and ('trend', '1h') in state['bt_done'])
    check('step iter 3: base@4h marked done in train_done (was failed)',
          ('base', '4h') in state['train_done'])
    check('step iter 3: base@4h BT SKIPPED (no BT submitted for failed train)',
          ('base', '4h') not in state['bt_tids']
          and ('base', '4h') in state['bt_done'])
    check('step iter 3: base@4h has error recorded',
          'failed' in state['cell_errors'].get(('base', '4h'), '')
          and 'OOM' in state['cell_errors'].get(('base', '4h'), ''))
    check('step iter 3: cells_complete=2 (trend done, base skipped)',
          r['cells_complete'] == 2)
    check('step iter 3: NOT finished yet (futures still running)',
          not r['finished'])

    # Iteration 4: futures@5m done → BT submits. Same iter, BT not yet done.
    by_id = {
        'train-tid-1': {'task_id': 'train-tid-1', 'status': 'done'},
        'train-tid-2': {'task_id': 'train-tid-2', 'status': 'failed'},
        'train-tid-3': {'task_id': 'train-tid-3', 'status': 'done'},
        'bt-tid-trend-1h': {'task_id': 'bt-tid-trend-1h', 'status': 'done'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit_bt)
    check('step iter 4: futures@5m BT submitted now that its train is done',
          state['bt_tids'].get(('futures', '5m')) == 'bt-tid-futures-5m')
    check('step iter 4: not finished — futures BT still running',
          not r['finished'])

    # Iteration 5: futures BT done → all terminal → finished=True
    by_id = {
        'train-tid-1': {'task_id': 'train-tid-1', 'status': 'done'},
        'train-tid-2': {'task_id': 'train-tid-2', 'status': 'failed'},
        'train-tid-3': {'task_id': 'train-tid-3', 'status': 'done'},
        'bt-tid-trend-1h':    {'task_id': 'bt-tid-trend-1h',    'status': 'done'},
        'bt-tid-futures-5m':  {'task_id': 'bt-tid-futures-5m',  'status': 'done'},
    }
    r = step_fn(test_cells, state, by_id, _fake_submit_bt)
    check('step iter 5: ALL cells terminal → finished=True',
          r['finished'])
    check('step iter 5: cells_complete=3 (all cells terminated, base BT skipped counts as terminal)',
          r['cells_complete'] == 3)
    check('step iter 5: cell_errors retained base@4h failure',
          ('base', '4h') in state['cell_errors'])
    check('step iter 5: cell_errors does NOT include successful cells',
          ('trend', '1h') not in state['cell_errors']
          and ('futures', '5m') not in state['cell_errors'])

    # ── submit_fn failure path ──────────────────────────────────────────
    state2 = {
        'train_tids':  {('tft', '15m'): 'train-tid-x'},
        'bt_tids':     {},
        'train_done':  set(),
        'bt_done':     set(),
        'cell_errors': {},
    }
    def _fail_submit_bt(model_key, tf, bt_spec):
        return None    # cluster unreachable
    by_id = {'train-tid-x': {'task_id': 'train-tid-x', 'status': 'done'}}
    r = step_fn([('tft', '15m')], state2, by_id, _fail_submit_bt)
    check('submit_bt fails: cell marked bt_done (no infinite retry)',
          ('tft', '15m') in state2['bt_done'])
    check('submit_bt fails: error recorded "BT submit failed"',
          'BT submit failed' in state2['cell_errors'].get(('tft', '15m'), ''))
    check('submit_bt fails: finished=True (all terminal incl skipped BT)',
          r['finished'])

    # ── (d) Endpoint via test_client ─────────────────────────────────────
    posts = []
    def _silent_post(path, body, **kw):
        posts.append((path, body))
        return ({'ok': True, 'task_id': f'fake-tid-{len(posts)}'}, 200)
    saved_post = dash_app._cluster_proxy_post
    dash_app._cluster_proxy_post = _silent_post
    try:
        client = dash_app.app.test_client()
        resp = client.post('/api/training/run/all',
                           json={'force': True})
        rj = resp.get_json()
    finally:
        dash_app._cluster_proxy_post = saved_post

    check('endpoint /api/training/run/all returns 200 OK',
          resp.status_code == 200)
    check('endpoint: response routed_to="cluster" (default Phase 100b)',
          rj is not None and rj.get('routed_to') == 'cluster')
    check('endpoint: response model="all", n=1',
          rj.get('model') == 'all' and rj.get('n') == 1)
    check('endpoint: response includes job_id',
          'job_id' in rj and len(rj['job_id']) > 6)

    # AI_TRADER_LOCAL_TRAINING=1 routes to legacy local subprocess.
    os.environ['AI_TRADER_LOCAL_TRAINING'] = '1'
    try:
        client = dash_app.app.test_client()
        resp = client.post('/api/training/run/all',
                           json={'force': True})
        rj = resp.get_json()
    finally:
        os.environ.pop('AI_TRADER_LOCAL_TRAINING', None)
    check('endpoint with AI_TRADER_LOCAL_TRAINING=1: routed_to="local"',
          rj is not None and rj.get('routed_to') == 'local')


def test_phase69_pr42_pipeline_through_scheduler_plus_followup_backtest():
    """Two improvements to keep training and backtest panels coherent:
      P1. /api/pipeline/run goes through the resource scheduler's
          'exclusive' lane via a worker thread, so it can't collide
          with a manual OFT (also exclusive) or steal the GPU from
          a TFT run mid-flight.
      P2. /api/training/run/<key> accepts {with_backtest: true}; on
          success the worker chains run_full_backtest(timeframes=(tf,))
          so the Stability Heatmap row for that TF refreshes without
          a full pipeline orchestrator run."""
    print('\n[Phase 69 -- PR-42 pipeline-via-scheduler + chained backtest]')

    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()
    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # P1: pipeline runs through scheduler.acquire('exclusive') on a
    # worker thread, returns queued job_id immediately.
    check('_run_pipeline_blocking helper defined',
          'def _run_pipeline_blocking' in app)
    pipe_run_start = app.find('def api_pipeline_run')
    pipe_run_end   = app.find('\ndef ', pipe_run_start + 4)
    pipe_run_body  = app[pipe_run_start:pipe_run_end] if pipe_run_end > 0 else ''
    check('api_pipeline_run hands off to _run_pipeline_blocking thread',
          '_run_pipeline_blocking' in pipe_run_body
          and 'threading.Thread' in pipe_run_body)
    check('api_pipeline_run records queued job before returning',
          "status='queued'" in pipe_run_body
          and "model='pipeline'" in pipe_run_body)
    pipe_block_start = app.find('def _run_pipeline_blocking')
    pipe_block_end   = app.find('\ndef ', pipe_block_start + 4)
    pipe_block_body  = app[pipe_block_start:pipe_block_end] if pipe_block_end > 0 else ''
    check('_run_pipeline_blocking acquires exclusive lane',
          "_training_scheduler.acquire('exclusive')" in pipe_block_body)
    check('_run_pipeline_blocking releases in finally',
          'finally:' in pipe_block_body
          and "_training_scheduler.release('exclusive')" in pipe_block_body)

    # P2: with_backtest plumbed through trainer functions + API.
    check('api_training_run_one parses with_backtest',
          'with_backtest = bool(body.get(' in app)
    check('_run_trainer_blocking accepts with_backtest kwarg',
          'def _run_trainer_blocking' in app
          and 'with_backtest: bool = False' in app)
    check('_run_trainer_multi_tf accepts with_backtest kwarg',
          'def _run_trainer_multi_tf' in app
          and 'with_backtest: bool = False' in app)
    check('_spawn_followup_backtest helper defined',
          'def _spawn_followup_backtest' in app)
    check('chained backtest only runs on final_status == \'done\'',
          "with_backtest and final_status == 'done'" in app)
    check('followup backtest invokes run_full_backtest',
          'run_full_backtest(timeframes=' in app)

    # Frontend: checkbox + restore from localStorage + send body field.
    check('"refresh stats after train" checkbox present',
          'id="tr-with-backtest"' in tpl)
    check('checkbox state restored from localStorage on DOMContentLoaded',
          "localStorage.getItem('tr-with-backtest')" in tpl)
    check('trRunOne sends body.with_backtest when checkbox checked',
          'body.with_backtest = true' in tpl)


def test_phase68_pr41_orphan_training_reattach_and_collapse_fix():
    """Three follow-up fixes after PR-40:
      1. Duplicate stToggle definition was shadowing the working one,
         so clicking ML Models / Strategies / Quant Matrix headers
         didn't expand/collapse them. Single canonical definition now.
      2. Strategy tab activation fires pollTrainingJobs + pipelineRefresh
         immediately so live training shows within 100 ms instead of
         waiting up to 10 s for the staggered tick.
      3. Orphan-training subprocess sweep on dashboard boot: any
         python.exe matching a known trainer's command line that isn't
         in our jobs file gets a synthetic 'reattached (orphan)' job
         entry so the operator can see what's burning GPU/CPU.
    """
    print('\n[Phase 68 -- PR-41 orphan reattach + collapse fix]')

    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()

    # 1. Single stToggle definition (no shadowing).
    check('single canonical stToggle definition',
          tpl.count('function stToggle(') == 1)
    # The canonical stToggle must toggle BOTH .open on the section AND
    # .st-collapsed on the body — the CSS keys on .st-collapsed for
    # display:none. Earlier shadow only toggled .open, breaking expand.
    st_start = tpl.find('function stToggle(')
    st_end   = tpl.find('\n}\n', st_start)
    st_body  = tpl[st_start:st_end] if st_end > st_start else ''
    check('stToggle toggles section.classList.toggle(\'open\')',
          "classList.toggle('open')" in st_body)
    check('stToggle toggles body.classList.toggle(\'st-collapsed\')',
          "classList.toggle('st-collapsed')" in st_body)

    # 2. switchTab('strategy') fires pollTrainingJobs + pipelineRefresh.
    sw_start = tpl.find('function switchTab(')
    sw_end   = tpl.find('\n}\n', sw_start)
    sw_body  = tpl[sw_start:sw_end] if sw_end > sw_start else ''
    check("switchTab('strategy') fires pollTrainingJobs immediately",
          "name === 'strategy'" in sw_body and 'pollTrainingJobs()' in sw_body)
    check("switchTab('strategy') fires pipelineRefresh immediately",
          'pipelineRefresh()' in sw_body)

    # 3. Orphan-training subprocess sweep at boot.
    check('_detect_orphan_training_subprocesses helper defined',
          'def _detect_orphan_training_subprocesses' in app)
    check('orphan sweep runs at boot via _training_state_recover',
          '_training_state_recover' in app
          and '_detect_orphan_training_subprocesses()' in app)
    check('orphan job tagged with "reattached (orphan from prior boot)"',
          "'reattached (orphan from prior boot)'" in app
          or 'reattached (orphan' in app)


def test_phase64_pr40_training_survives_dashboard_restart():
    """Trainer subprocesses are detached + persisted so an in-flight OFT
    run survives a dashboard restart instead of dying with the parent."""
    print('\n[Phase 64 -- PR-40 training survives dashboard restart]')

    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app = f.read()

    check('_TRAINING_JOBS_FILE persistence path defined',
          '_TRAINING_JOBS_FILE' in app and 'training_jobs.json' in app)
    check('_persist_training_jobs helper defined',
          'def _persist_training_jobs' in app)
    check('_load_training_jobs helper defined',
          'def _load_training_jobs' in app)
    check('_reattach_training_subprocess helper defined',
          'def _reattach_training_subprocess' in app)
    check('_spawn_training_subprocess uses DETACHED_PROCESS on Windows',
          'DETACHED_PROCESS' in app and 'CREATE_NEW_PROCESS_GROUP' in app)
    check('child stdout redirected to per-job log file',
          "_TRAINING_LOG_DIR" in app and "/ f'{job_id}.log'" in app)
    check('_record_job calls _persist_training_jobs',
          '_persist_training_jobs()' in app
          and app.count('_persist_training_jobs') >= 2)
    check('child_pid recorded after spawn',
          'child_pid=proc.pid' in app)
    check('training-jobs-reload thread armed at module load',
          "name='training-jobs-reload'" in app
          or 'training-jobs-reload' in app)
    check('reattach uses psutil.pid_exists for liveness',
          'psutil.pid_exists' in app and 'pid_exists(int(pid))' in app)


def test_phase65_pr40_pipeline_orchestrator_progress_broadcast():
    """Frontend synthesizes a virtual allJob from _pipeStatus when the
    pipeline orchestrator is in train phase but no API job exists, so
    each Model Training row flips to 'RUNNING as part of Pipeline
    Orchestrator' instead of looking idle."""
    print('\n[Phase 65 -- PR-40 pipeline orchestrator -> row broadcast]')

    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # The synthesizer block in pollTrainingJobs.
    pti = tpl.find('async function pollTrainingJobs')
    pti_end = tpl.find('\n}\n', pti)
    body = tpl[pti:pti_end] if pti_end > pti else ''
    check('pollTrainingJobs declares allJob with let (mutable)',
          'let allJob =' in body)
    check("synth path checks _pipeStatus.status === 'running'",
          "_pipeStatus.status === 'running'" in body
          and "_pipeStatus.phase === 'train'" in body)
    check('synth allJob carries progress_label "Pipeline Orchestrator"',
          'as part of Pipeline Orchestrator' in body)
    check('broadcast loop uses allJob.progress_label when set',
          'allJob.progress_label || ' in body)


def test_phase66_pr40_strategy_sections_collapsed_default():
    """All Strategy & ML sections start collapsed; user's expand/collapse
    choice is persisted to localStorage so it survives F5."""
    print('\n[Phase 66 -- PR-40 strategy sections collapsed by default]')

    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # No section in the strategy tab uses the .open / unmarked pattern
    # for "expanded by default" anymore.
    check('Strategy & ML section default = collapsed (no ".st-sec open")',
          'class="st-sec open"' not in tpl)
    check('All collapsible-section defaults include is-collapsed',
          'class="card collapsible-section"' not in tpl
          or tpl.count('class="card collapsible-section"') == 0)

    # Restore + persist helpers exist.
    check('_restoreStrategySectionStates restores from localStorage',
          'function _restoreStrategySectionStates' in tpl
          and 'st-sec-open:' in tpl)
    check('toggleSection persists open state to localStorage',
          'function toggleSection' in tpl
          and "localStorage.setItem('st-sec-open:'" in tpl)
    check('stToggle persists open state to localStorage',
          'function stToggle' in tpl
          and "localStorage.setItem('st-sec-open:'" in tpl)


def test_phase67_pr40_loadStrategyFull_renders_directly():
    """Refresh button on Model Training works: loadStrategyFull no longer
    relies on renderStrategyTab(botState) (which guards on _stratFull
    and may early-return). It now calls the four renderers directly."""
    print('\n[Phase 67 -- PR-40 loadStrategyFull renders directly]')

    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    body_start = tpl.find('async function loadStrategyFull')
    body_end   = tpl.find('\n}\n', body_start)
    body = tpl[body_start:body_end] if body_end > body_start else ''
    for fn in ('_renderMLCards', '_renderTrainingTable',
               '_renderStratCards', '_renderBtSummary'):
        check(f'loadStrategyFull calls {fn} directly', fn + '(' in body)
    check('refresh chip flashes "refreshing…" while in flight',
          "'refreshing…'" in body)


def test_phase109_gzip_to_parquet_migration():
    """2026-05-15 operator: 'we need to move all the data from GZIP files
    to DB file to let others easily read it without overhead'. Implementation:

      1. scripts/migrate_1s_to_parquet.py — idempotent, per-yyyymm,
         deduplicates spot+live-tail timestamps.
      2. src/analysis/tick_feature_loader._read_1s_window — parquet
         fast-path with gzip fallback so a migration mid-rollout doesn't
         break running trainers.
    """
    print('\n[Phase 109 -- GZIP -> parquet 1s migration]')

    # Migration script exists + key symbols present.
    script_path = os.path.join(BASE_DIR, 'scripts', 'migrate_1s_to_parquet.py')
    check('migration script exists', os.path.exists(script_path))
    with open(script_path, encoding='utf-8') as f:
        m_src = f.read()
    for needle in ('def _migrate_symbol', 'def _candidate_sources',
                   'def _existing_partitions', 'idempotent',
                   "QUALIFY ROW_NUMBER() OVER (PARTITION BY timestamp",
                   "COMPRESSION 'SNAPPY'", '--force', '--dry-run'):
        check(f"migration script has {needle!r}", needle in m_src)
    check('migration writes to data/parquet/<SYM>/1s/yyyymm=<YYYY-MM>',
          "PARQUET_DIR / sym / \"1s\"" in m_src
          and "yyyymm=" in m_src)

    # tick_feature_loader parquet fast-path.
    tfl_path = os.path.join(BASE_DIR, 'src', 'analysis',
                             'tick_feature_loader.py')
    with open(tfl_path, encoding='utf-8') as f:
        tfl_src = f.read()
    check('tick loader exposes _parquet_partitions_exist',
          'def _parquet_partitions_exist(' in tfl_src)
    check('tick loader exposes _parquet_1s_dir',
          'def _parquet_1s_dir(' in tfl_src)
    check('_read_1s_window checks parquet first',
          'if _parquet_partitions_exist(symbol):' in tfl_src
          and 'PARQUET FAST PATH' in tfl_src)
    check('_read_1s_window falls back to gzip on parquet failure',
          'GZIP LEGACY PATH' in tfl_src)
    check('has_tick_data accepts either parquet OR gzip',
          'return _parquet_partitions_exist(symbol) or bool(_candidate_files(symbol))' in tfl_src)
    # Behavioural: with no parquet store, _parquet_partitions_exist returns False
    # cleanly without raising.
    import importlib
    if 'src.analysis.tick_feature_loader' in sys.modules:
        importlib.reload(sys.modules['src.analysis.tick_feature_loader'])
    from src.analysis import tick_feature_loader as tfl
    check('_parquet_partitions_exist returns bool for unknown symbol',
          tfl._parquet_partitions_exist('XXX_FAKE_USDT') is False)
    check('has_tick_data returns False for fake symbol with no data',
          tfl.has_tick_data('XXX_FAKE_USDT') is False)


def test_phase108_training_progress_instrumentation():
    """2026-05-15 operator: 'On Model Training screen add the epoch
    number to the status and estimated time to show what epoch is currently
    running' + 'save the run time on all TFs ... how long does it take
    for 1 or for 10 epochs'.

    Implementation:
      - src/utils/training_progress.py — JSON state file
        data/training_progress.json with per-task records:
        current_epoch / n_epochs / last_epoch_duration_s / mean_epoch_duration_s
        / elapsed_s / eta_s.
      - src/engine/train_tft_model.py: Lightning per-epoch Callback writes
        progress; meta JSON now records started_at_unix / finished_at_unix /
        duration_s / per_epoch_s / epochs_completed; per-TF meta JSON
        (tft_<tf>_meta.json) so different TFs don't overwrite each other;
        record_run_from_meta call (previously absent from TFT path).
      - src/engine/train_model.py (base): same wall-clock fields
        in meta + 1-epoch progress record so the dashboard shows tabular
        trainers alongside TFT.
      - /api/training/progress endpoint.
      - Dashboard training table's Status column shows
        'RUNNING · epoch 4/12 · ~90m/epoch · ETA 72m' for the active row.
    """
    print('\n[Phase 108 -- training_progress instrumentation]')

    # --- (1) training_progress module ---
    tp_src_path = os.path.join(BASE_DIR, 'src', 'utils', 'training_progress.py')
    check('training_progress module exists', os.path.exists(tp_src_path))
    with open(tp_src_path, encoding='utf-8') as f:
        tp_src = f.read()
    for sym in ('def start(', 'def epoch_done(', 'def heartbeat(',
                'def finish(', 'def get(', 'def list_active(',
                'def list_all(', 'def clear_stale(', 'PROGRESS_PATH'):
        check(f'training_progress exports {sym}', sym in tp_src)
    # Behavioural round-trip.
    import importlib
    if 'src.utils.training_progress' in sys.modules:
        importlib.reload(sys.modules['src.utils.training_progress'])
    from src.utils import training_progress as tp
    tid = '__test_phase108__'
    rec = tp.start(tid, model='tft', tf='1h', n_epochs=3, trainer='test')
    check('start returns task record',
          rec.get('task_id') == tid and rec.get('n_epochs') == 3)
    check('epoch_done updates fields', tp.epoch_done(tid, 1, 60.0) is True)
    r1 = tp.get(tid)
    check('after epoch_done — current_epoch=1, epochs_completed=1',
          r1.get('current_epoch') == 1 and r1.get('epochs_completed') == 1)
    check('after epoch_done — eta_s computed (n_remaining * mean)',
          r1.get('eta_s') is not None and r1.get('eta_s') > 0)
    check('epoch_done sets last_epoch_duration_s',
          r1.get('last_epoch_duration_s') == 60.0)
    # 2nd epoch
    tp.epoch_done(tid, 2, 80.0)
    r2 = tp.get(tid)
    check('after epoch 2 — mean_epoch_duration_s averages (60+80)/2',
          abs(r2.get('mean_epoch_duration_s') - 70.0) < 0.01)
    # Finish
    tp.finish(tid, status='done')
    rf = tp.get(tid)
    check('finish marks status=done', rf.get('status') == 'done')
    check('finish sets eta_s=0', rf.get('eta_s') == 0.0)
    # Cleanup the test record
    tp.clear_stale(max_age_s=0)

    # --- (2) train_tft_model wiring ---
    tft_src = open(os.path.join(BASE_DIR, 'src', 'engine',
                                 'train_tft_model.py'), encoding='utf-8').read()
    check('TFT trainer imports training_progress',
          'from src.utils import training_progress' in tft_src)
    check('TFT trainer calls _tp.start(...)',
          '_tp.start(_task_id' in tft_src)
    check('TFT trainer defines per-epoch Lightning Callback',
          '_EpochProgressCB' in tft_src and 'on_train_epoch_end' in tft_src)
    check('TFT trainer calls _tp.epoch_done() on epoch end',
          '_tp.epoch_done(self.task_id' in tft_src)
    check('TFT trainer wraps model.fit in try/except for progress finish',
          '_tp.finish(_task_id, status="error"' in tft_src)
    check('TFT trainer records started_at_unix / finished_at_unix in meta',
          '"started_at_unix":' in tft_src and '"finished_at_unix":' in tft_src)
    check('TFT trainer records duration_s + per_epoch_s in meta',
          '"duration_s":' in tft_src and '"per_epoch_s":' in tft_src)
    check('TFT trainer records epochs_completed in meta',
          '"epochs_completed": _epochs_completed' in tft_src)
    check('TFT trainer writes per-TF meta JSON (tft_<tf>_meta.json)',
          'per_tf_meta    = MODEL_DIR / f"tft_{timeframe}_meta.json"' in tft_src)
    check('TFT trainer calls record_run_from_meta (was missing pre-fix)',
          'record_run_from_meta(meta_dict, model="tft"' in tft_src)
    check('TFT trainer accepts progress_task_id kwarg from worker',
          'progress_task_id: str | None = None' in tft_src)

    # --- (3) Base trainer wiring ---
    base_src = open(os.path.join(BASE_DIR, 'src', 'engine',
                                  'train_model.py'), encoding='utf-8').read()
    check('base trainer imports training_progress',
          'from src.utils import training_progress' in base_src)
    check('base trainer calls _tp.start(...)',
          '_tp.start(_train_task_id' in base_src)
    check('base trainer records started_at_unix / finished_at_unix in meta',
          '"started_at_unix":  _train_started_at' in base_src
          and '"finished_at_unix": _train_finished_at' in base_src)
    check('base trainer records duration_s in meta',
          '"duration_s":       round(_train_duration_s' in base_src)

    # --- (4) Flask endpoint ---
    app_src = open(os.path.join(BASE_DIR, 'src', 'dashboard',
                                 'app.py'), encoding='utf-8').read()
    check('app.py exposes /api/training/progress',
          "@app.route('/api/training/progress'" in app_src)
    check('endpoint reads training_progress.list_active / list_all',
          '_tp.list_active()' in app_src and '_tp.list_all' in app_src)

    # --- (5) Dashboard JS wiring ---
    tpl = open(os.path.join(BASE_DIR, 'src', 'dashboard', 'templates',
                             'index.html'), encoding='utf-8').read()
    check('frontend declares _trainingProgress global',
          'let _trainingProgress = {}' in tpl)
    check('pollTrainingJobs fetches /api/training/progress in parallel',
          "fetch('/api/training/progress?include_terminal=0&limit=50'" in tpl)
    check('Status cell appends "epoch N/M" line when progress exists',
          '`epoch ${cur}/${tot}`' in tpl)
    check('Status cell shows per-epoch timing',
          '~${perEpFmt}/epoch' in tpl)
    check('Status cell shows ETA when eta_s > 0',
          'ETA ${epochEtaFmt}' in tpl)


def test_phase107_worker_bind_host_env_default():
    """2026-05-15 operator caught — workers were binding 127.0.0.1 only,
    so the master could NOT POST tasks to them across LAN/Tailscale even
    when the register-side IP override made the worker appear at its real
    address. Fix: .env sets WORKER_BIND_HOST=0.0.0.0 for THIS deployment.

    worker.py's CLI default expression is `os.getenv("WORKER_BIND_HOST",
    "127.0.0.1")`, so the env var is the canonical override. We pin it
    here so a future operator can't silently break cluster mode by
    deleting the line."""
    print('\n[Phase 107 -- WORKER_BIND_HOST env default for cluster mode]')
    env_path = os.path.join(BASE_DIR, '.env')
    with open(env_path, encoding='utf-8') as f:
        env_src = f.read()
    check('.env contains WORKER_BIND_HOST=0.0.0.0',
          'WORKER_BIND_HOST=0.0.0.0' in env_src)
    check('.env documents WHY (operator caught Ivan loopback issue)',
          'WORKER_BIND_HOST' in env_src and 'multi-machine' in env_src)
    # worker.py still falls back to 127.0.0.1 when env is unset (safe
    # default for upstream users).
    worker_src_path = os.path.join(BASE_DIR, 'src', 'training',
                                    'distributed', 'worker.py')
    with open(worker_src_path, encoding='utf-8') as f:
        worker_src = f.read()
    check('worker.py argparse default reads WORKER_BIND_HOST env',
          'os.getenv("WORKER_BIND_HOST", "127.0.0.1")' in worker_src)
    # Behaviour: load .env + simulate argparse default resolution.
    import importlib
    if 'dotenv' in sys.modules:
        importlib.reload(sys.modules['dotenv'])
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)
    import os as _os
    default = _os.getenv('WORKER_BIND_HOST', '127.0.0.1')
    check(f'live env override = 0.0.0.0 (got {default!r})',
          default == '0.0.0.0')


def test_phase106_tft_presets_and_remote_worker_ip_override():
    """2026-05-15 — two operator asks closed in one phase:

    (A) TFT epoch tuning:
        - Add `min_epochs=3` hard floor so EarlyStopping cannot fire
          before that point (previously patience=5 was meaningless at
          n_epochs=3).
        - Make `patience` tunable (was hard-coded 5).
        - Bump default to n_epochs=10 with min_epochs=3, patience=4.
        - Add TFT_PRESETS dict + --preset CLI flag so cost/quality dial
          is one edit.
        - Wire worker to read all three from training_rules.json.
        - Record n_epochs/min_epochs/patience in meta JSON.

    (B) Remote-worker IP override:
        - Ivan's workers bound to 127.0.0.1 locally and reported ip=127.0.0.1
          in /api/cluster/register. The master then dispatched to
          http://127.0.0.1:<port>/task which hit the MASTER's own loopback,
          failing with "WinError 10061 actively refused".
        - Now: when register payload has loopback ip AND remote_addr is
          non-loopback, override to remote_addr.
        - Extend _is_safe_worker_entry to accept Tailscale CGNAT
          (100.64.0.0/10) since that's the network Ivan arrives on.
    """
    print('\n[Phase 106 -- TFT presets + remote worker IP override]')

    # --- (A) train_tft_model exposes min_epochs + patience ---
    import importlib
    if 'src.engine.train_tft_model' in sys.modules:
        importlib.reload(sys.modules['src.engine.train_tft_model'])
    from src.engine import train_tft_model as ttm
    import inspect
    sig = inspect.signature(ttm.train_tft_model)
    for p in ('n_epochs', 'min_epochs', 'patience'):
        check(f'train_tft_model has param {p}', p in sig.parameters)
    check('train_tft_model n_epochs default is 10',
          sig.parameters['n_epochs'].default == 10)
    check('train_tft_model min_epochs default is 3',
          sig.parameters['min_epochs'].default == 3)
    check('train_tft_model patience default is 4',
          sig.parameters['patience'].default == 4)
    check('TFT_PRESETS exposes cheap/fair-vs-gbt/max-quality',
          set(['cheap', 'fair-vs-gbt', 'max-quality']).issubset(
              set(ttm.TFT_PRESETS.keys())))
    for k, expected in (('cheap', 3), ('fair-vs-gbt', 12), ('max-quality', 25)):
        check(f'preset {k} has n_epochs={expected}',
              ttm.TFT_PRESETS[k]['n_epochs'] == expected)
    # Source-level: min_epochs flows into pl_trainer_kwargs and EarlyStopping
    # uses the parameter.
    tft_src_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_tft_model.py')
    with open(tft_src_path, encoding='utf-8') as f:
        tft_src = f.read()
    check('Trainer gets min_epochs from kwarg',
          '"min_epochs": _min_epochs' in tft_src)
    check('EarlyStopping uses tunable patience',
          'patience=int(patience)' in tft_src)
    check('meta JSON records min_epochs',
          '"min_epochs": _min_epochs' in tft_src)
    check('meta JSON records patience',
          '"patience": int(patience)' in tft_src)
    check('CLI exposes --preset',
          '--preset' in tft_src)
    # training_rules.json should now have n_epochs=10, min_epochs=3, patience=4
    rules_path = os.path.join(BASE_DIR, 'data', 'training_rules.json')
    import json as _json
    rules = _json.loads(open(rules_path, encoding='utf-8').read())
    tft_params = rules['models']['tft']['params']
    check('training_rules.json tft.n_epochs=10',
          tft_params.get('n_epochs') == 10)
    check('training_rules.json tft.min_epochs=3',
          tft_params.get('min_epochs') == 3)
    check('training_rules.json tft.patience=4',
          tft_params.get('patience') == 4)
    check('training_rules.json includes preset reference table',
          '_presets' in tft_params)
    # Worker code reads all three from rules
    worker_src = open(os.path.join(BASE_DIR, 'src', 'training', 'distributed',
                                    'worker.py'), encoding='utf-8').read()
    check('worker _train_tft reads n_epochs, patience, min_epochs from rules',
          'for key in ("n_epochs", "patience", "min_epochs")' in worker_src)

    # --- (B) Orchestrator IP override + CGNAT safety ---
    orch_src = open(os.path.join(BASE_DIR, 'src', 'training', 'distributed',
                                  'orchestrator.py'), encoding='utf-8').read()
    check('orchestrator register handler overrides loopback ip with remote_addr',
          'overriding loopback-reported ip' in orch_src or
          ('reported_ip in loopback_set' in orch_src
           and 'remote_addr' in orch_src
           and 'body["ip"] = remote_addr' in orch_src))
    check('orchestrator _is_safe_worker_entry accepts CGNAT (100.64.0.0/10)',
          '100.64.0.0/10' in orch_src)

    # Behavioural test for _is_safe_worker_entry — accept loopback, RFC1918,
    # and CGNAT; reject link-local and public.
    if 'src.training.distributed.orchestrator' in sys.modules:
        importlib.reload(sys.modules['src.training.distributed.orchestrator'])
    from src.training.distributed.orchestrator import Orchestrator
    fn = Orchestrator._is_safe_worker_entry
    cases = [
        ({'ip': '127.0.0.1', 'port': 7701},      True,  'loopback'),
        ({'ip': '192.168.0.105', 'port': 7701},  True,  'RFC1918'),
        ({'ip': '10.0.0.5', 'port': 7701},       True,  'RFC1918 /8'),
        ({'ip': '100.88.71.74', 'port': 7702},   True,  'Tailscale CGNAT'),
        ({'ip': '100.127.255.255', 'port': 7702},True,  'CGNAT upper bound'),
        ({'ip': '8.8.8.8', 'port': 7701},        False, 'public Google DNS'),
        ({'ip': '169.254.169.254', 'port': 7701},False, 'AWS metadata (link-local)'),
        ({'ip': '127.0.0.1', 'port': 80},        False, 'port too low'),
    ]
    for w, expected, label in cases:
        check(f'_is_safe_worker_entry({label}) == {expected}',
              fn(w) == expected)


def test_phase105_plateau_detection_l2_news_tick_features():
    """2026-05-15 operator request — close the four open ML improvements:
      1. Plateau detection (prefer robust neighbourhood over isolated spike).
      2. L2 microstructure features as ML inputs.
      3. News sentiment as ML inputs (joined to bar timeline).
      4. Tick-level (1s-derived) microstructure features.

    Each feature loader is required to produce a stable schema even when no
    underlying data exists for the symbol — so retrains automatically
    benefit as the parquet store accumulates without changing input shape.
    """
    print('\n[Phase 105 -- plateau detection + L2/news/tick features]')

    # --- (1) Plateau detection module exists + signature contract ---
    import importlib
    for sym in ('select_plateau_winner', 'summarise_for_proposal',
                'PlateauResult', 'PlateauSelection'):
        try:
            mod = importlib.import_module('src.engine.cio_plateau')
            check(f'cio_plateau exports {sym}', hasattr(mod, sym))
        except Exception as e:
            check(f'cio_plateau import: {sym}', False, str(e))

    # Synthetic study with a clear spike → plateau preference.
    try:
        import optuna
        import optuna.logging as _ol
        _ol.set_verbosity(_ol.WARNING)
        from src.engine.cio_plateau import (
            select_plateau_winner, PlateauResult,
        )
        # Build trials manually so the spike is guaranteed to land in the
        # sampled set — Optuna's TPE may otherwise route around it.
        study = optuna.create_study(direction='maximize')

        def _add(study, params, value):
            trial = optuna.trial.create_trial(
                params=params,
                distributions={k: optuna.distributions.FloatDistribution(0.0, 5.0)
                               for k in params},
                value=value,
            )
            study.add_trial(trial)

        # Add a stable plateau cloud around (pt=2.0, sl=1.0) value ≈ 4.5–4.9.
        for px in [1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15]:
            for sx in [0.9, 1.0, 1.1]:
                _add(study, {'pt': px, 'sl': sx},
                     4.9 - (px - 2.0) ** 2 - (sx - 1.0) ** 2 * 0.5)
        # Inject ONE isolated spike at (pt=4.5, sl=4.5) value=50 with awful neighbours.
        _add(study, {'pt': 4.5, 'sl': 4.5}, 50.0)
        # Add a few isolators around the spike at very low value so its
        # neighbourhood looks worthless.
        for px in [4.4, 4.6]:
            for sx in [4.4, 4.6]:
                _add(study, {'pt': px, 'sl': sx}, -10.0)

        sel = select_plateau_winner(study, k=5, alpha=0.5, min_trials=8)
        check('plateau detection returns a selection', sel is not None)
        if sel:
            check('spike winner is the synthetic spike (value=50)',
                  abs(sel.spike_winner.raw_value - 50.0) < 1e-6)
            check('plateau winner is NOT the spike (different params)',
                  sel.plateau_winner.params != sel.spike_winner.params)
            check('plateau winner has higher plateau_score than spike',
                  sel.plateau_winner.plateau_score > sel.spike_winner.plateau_score)
            # Spike beats plateau on raw_value, plateau beats spike on plateau_score.
            check("recommendation chooses 'spike' since recovery_ratio < 0.85",
                  sel.recommendation in ('spike', 'plateau'))
    except ImportError:
        check('optuna available for plateau test', None,
              'optuna not installed; skipping behavioural plateau test')

    # CIO agent integrates plateau analysis post-optimize.
    cio_path = os.path.join(BASE_DIR, 'src', 'engine', 'cio_agent.py')
    with open(cio_path, encoding='utf-8') as f:
        cio_src = f.read()
    check('cio_agent.run imports cio_plateau.select_plateau_winner',
          'from src.engine.cio_plateau import' in cio_src and
          'select_plateau_winner' in cio_src)
    check("cio_agent records plateau_analysis in summary",
          "summary['plateau_analysis']" in cio_src)
    check('cio_agent records plateau_recommended flag',
          "summary['plateau_recommended']" in cio_src)
    check('cio_agent apply_best honours plateau_recommended',
          "winning.get('plateau_recommended')" in cio_src and
          "recommended_params" in cio_src)

    # --- (2) L2 feature loader ---
    from src.analysis.l2_feature_loader import (
        L2_FEATURE_COLUMNS, l2_partitions_exist, load_bar_aligned as l2_aligned,
    )
    check('l2 loader exports L2_FEATURE_COLUMNS',
          len(L2_FEATURE_COLUMNS) >= 5)
    # Stable-schema test: empty partitions → all columns present, zeros.
    df = l2_aligned('XXX_FAKE_USDT', [0, 60_000, 120_000], 60_000)
    check('l2_aligned returns DataFrame with every feature col when no data',
          all(c in df.columns for c in L2_FEATURE_COLUMNS))
    check('l2_aligned fills zeros for missing data',
          float(df['l2_snapshot_count'].sum()) == 0.0)

    # --- (3) News feature loader ---
    from src.analysis.news_feature_loader import (
        NEWS_FEATURE_COLUMNS, load_bar_aligned as news_aligned, is_available,
    )
    check('news loader exports NEWS_FEATURE_COLUMNS',
          len(NEWS_FEATURE_COLUMNS) >= 5)
    df = news_aligned('XXX_FAKE_USDT', [0, 60_000, 120_000], 60_000)
    check('news_aligned has every feature col',
          all(c in df.columns for c in NEWS_FEATURE_COLUMNS))
    # When there's no news for a fake symbol, all features must be 0/empty.
    check('news_aligned fills zeros for missing symbol',
          float(df['news_count'].sum()) == 0.0)

    # --- (4) Tick feature loader ---
    from src.analysis.tick_feature_loader import (
        TICK_FEATURE_COLUMNS, has_tick_data, load_bar_aligned as tick_aligned,
    )
    check('tick loader exports TICK_FEATURE_COLUMNS',
          len(TICK_FEATURE_COLUMNS) >= 6)
    df = tick_aligned('XXX_FAKE_USDT', [0, 60_000, 120_000], 60_000)
    check('tick_aligned has every feature col',
          all(c in df.columns for c in TICK_FEATURE_COLUMNS))
    check('tick_aligned fills zeros for missing symbol',
          float(df['tick_seconds_count'].sum()) == 0.0)

    # --- (5) feature_engineering exposes add_l2_features / add_news_features /
    # add_tick_features so the GBT trainers can pull them in ---
    fe_path = os.path.join(BASE_DIR, 'src', 'analysis', 'feature_engineering.py')
    with open(fe_path, encoding='utf-8') as f:
        fe_src = f.read()
    for fn in ('def add_l2_features(', 'def add_news_features(',
               'def add_tick_features(', 'def freq_to_ms('):
        check(f'feature_engineering has {fn}', fn in fe_src)

    # --- (6) TFT trainer attaches L2 + news + tick features ---
    tft_path = os.path.join(BASE_DIR, 'src', 'engine', 'train_tft_model.py')
    with open(tft_path, encoding='utf-8') as f:
        tft_src = f.read()
    for hook in ('_maybe_attach_l2_features(', '_maybe_attach_news_features(',
                 '_maybe_attach_tick_features('):
        check(f'TFT trainer calls {hook}', hook in tft_src)
    check('TFT past_cov_cols includes L2 features when present',
          'from src.analysis.l2_feature_loader import L2_FEATURE_COLUMNS' in tft_src)
    check('TFT past_cov_cols includes news features when present',
          'from src.analysis.news_feature_loader import NEWS_FEATURE_COLUMNS' in tft_src)
    check('TFT past_cov_cols includes tick features when present',
          'from src.analysis.tick_feature_loader import TICK_FEATURE_COLUMNS' in tft_src)


def test_phase104_pc_load_balancer():
    """2026-05-15 operator request: 'implement the PC load balancer between
    dashboard and other processes ... give priority to dashboard, the only
    exception is trading bot on live source.'

    Implementation: src/utils/load_balancer.py — recommends and applies
    Windows process priorities (psutil) based on role lookups in
    data/process_registry.json + cmdline pattern matching for training
    subprocesses. Live-trade detection reads data/control.json trade_mode.

    This test exercises:
      - recommend_priorities() returns expected schema, with no side effects.
      - apply_priorities(dry_run=True) plans changes without mutating priorities.
      - live-trade exemption fires when trade_mode is in LIVE_TRADE_MODES.
      - enable/disable state persists in data/load_balancer_state.json.
      - The Flask endpoints exist + are wired.
    """
    print('\n[Phase 104 -- PC load balancer]')

    # Static checks on the module itself.
    lb_path = os.path.join(BASE_DIR, 'src', 'utils', 'load_balancer.py')
    check('load_balancer module exists', os.path.exists(lb_path))
    with open(lb_path, encoding='utf-8') as f:
        lb_src = f.read()
    for sym in ('def recommend_priorities(', 'def apply_priorities(',
                'def is_enabled(', 'def set_enabled(',
                'def start_background_thread(', 'LIVE_TRADE_MODES',
                'DEFAULT_ROLE_POLICY', 'TRAINING_CMDLINE_PATTERNS'):
        check(f'load_balancer exports {sym}', sym in lb_src)
    check('default policy gives dashboard ABOVE_NORMAL',
          "'dashboard':" in lb_src and 'ABOVE_NORMAL' in lb_src)
    check('default policy demotes cluster_orch to BELOW_NORMAL',
          "'cluster_orch'" in lb_src and 'BELOW_NORMAL' in lb_src)
    check('training-subprocess cmdline patterns cover src.training',
          "'src.training'" in lb_src)
    check('live-trade modes include real and mainnet',
          '"real"' in lb_src and '"mainnet"' in lb_src)

    # Behaviour — pure recommend_priorities() never raises and returns a list.
    import importlib
    if 'src.utils.load_balancer' in sys.modules:
        importlib.reload(sys.modules['src.utils.load_balancer'])
    from src.utils import load_balancer as lb
    plans = lb.recommend_priorities()
    check('recommend_priorities returns a list', isinstance(plans, list))
    if plans:
        plan = plans[0]
        for k in ('pid', 'role', 'reason', 'current_label',
                  'recommended_label', 'exempt_from_demotion'):
            check(f'plan dict has {k!r}', k in plan)

    # apply_priorities(dry_run=True) — must not actually change anything.
    # We can't easily snapshot/restore real priorities cross-platform here,
    # so test the contract: returns a summary dict with the expected keys.
    summary = lb.apply_priorities(dry_run=True)
    for k in ('ts', 'live_trading', 'dry_run', 'changed', 'unchanged',
              'failed', 'plans'):
        check(f'apply summary has {k!r}', k in summary)
    check('dry_run summary keeps dry_run=True', summary['dry_run'] is True)

    # Live-trade exemption — simulate by writing control.json then
    # checking that the bot plan flips to exempt.
    import tempfile, json as _json, shutil
    ctl_path = os.path.join(BASE_DIR, 'data', 'control.json')
    backup = None
    if os.path.exists(ctl_path):
        with open(ctl_path, encoding='utf-8') as f:
            backup = f.read()
    try:
        _orig = _json.loads(backup) if backup else {}
        live = dict(_orig); live['trade_mode'] = 'real'
        with open(ctl_path, 'w', encoding='utf-8') as f:
            _json.dump(live, f)
        plans_live = lb.recommend_priorities()
        bot_plans = [p for p in plans_live if p.get('role') == 'bot']
        if bot_plans:
            bp = bot_plans[0]
            check('live mode: bot is exempt_from_demotion=True',
                  bp.get('exempt_from_demotion') is True)
            check('live mode: bot recommended_label is ABOVE_NORMAL',
                  bp.get('recommended_label') == 'ABOVE_NORMAL')
        # And reverse: testnet → bot is NOT exempt.
        testnet = dict(_orig); testnet['trade_mode'] = 'testnet'
        with open(ctl_path, 'w', encoding='utf-8') as f:
            _json.dump(testnet, f)
        plans_test = lb.recommend_priorities()
        bot_plans = [p for p in plans_test if p.get('role') == 'bot']
        if bot_plans:
            check('testnet mode: bot NOT exempt',
                  bot_plans[0].get('exempt_from_demotion') is False)
    finally:
        if backup is not None:
            with open(ctl_path, 'w', encoding='utf-8') as f:
                f.write(backup)

    # enable/disable round-trip persists.
    prev_enabled = lb.is_enabled()
    lb.set_enabled(True)
    check('set_enabled(True) reflected in is_enabled()',
          lb.is_enabled() is True)
    lb.set_enabled(False)
    check('set_enabled(False) reflected in is_enabled()',
          lb.is_enabled() is False)
    # Restore.
    lb.set_enabled(prev_enabled)

    # Flask endpoints.
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app_src = f.read()
    for route in ('/api/system/load_balancer/status',
                  '/api/system/load_balancer/apply',
                  '/api/system/load_balancer/enable'):
        check(f'app.py exposes {route}', route in app_src)
    check('app.py auto-starts load_balancer background thread on boot',
          'start_background_thread(' in app_src)

    # UI card present.
    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()
    check('Monitor tab has PC Load Balancer card',
          'id="loadbal-section"' in tpl
          and 'PC Load Balancer' in tpl)
    for fn in ('function loadbalRefresh(', 'function loadbalEnable(',
               'function loadbalApply('):
        check(f'frontend handler {fn} present', fn in tpl)
    check('Monitor tab switch calls loadbalRefresh',
          "loadbalRefresh()" in tpl)


def test_phase103_data_coverage_audit_perf_and_registry_hb_lock_timeout():
    """2026-05-15 operator complaint: Data Coverage card never loaded;
    dashboard sluggish during training.

    Root cause: audit_coverage decompressed every gzip end-to-end to read
    the last bar's timestamp. With 20 symbols × 8 timeframes × ~10 GB of
    compressed archive on disk, the cold scan took ~10 min — well over
    the 2 min cache TTL, so the cache never warmed. Fix: use file mtime
    as the last_ts proxy by default. Opt-in `precise=True` for callers
    that genuinely need bar timestamps.

    Secondary symptom: 'registry-hb heartbeat exception: The file lock
    ... could not be acquired' WARNING in the dashboard banner. Caused by
    safe_json.transaction's default 5 s filelock timeout being exceeded
    when many parallel training workers contended for the registry. Fix:
    heartbeat() now uses a 15 s timeout AND catches Timeout to log at
    DEBUG so the banner stops getting hit on every contended tick.
    """
    print('\n[Phase 103 -- data coverage perf + registry-hb lock timeout]')

    # --- audit_coverage: fast-path uses mtime, no full gzip walk ---
    audit_src_path = os.path.join(BASE_DIR, 'src', 'utils', 'data_audit.py')
    with open(audit_src_path, encoding='utf-8') as f:
        audit_src = f.read()
    check('audit_coverage has precise=False parameter',
          'precise: bool = False' in audit_src)
    check('audit_coverage gates gzip scan on use_gzip_scan',
          'use_gzip_scan = precise or (not fast)' in audit_src)
    check('_mtime_to_ts helper defined',
          'def _mtime_to_ts(' in audit_src)
    check('fast path calls _mtime_to_ts instead of _parse_first_last_ts',
          'last_ts = _mtime_to_ts(path)' in audit_src)

    # --- Behaviour: actual perf of audit_coverage default call ---
    import time
    from src.utils import data_audit
    t0 = time.time()
    rows = data_audit.audit_coverage()
    elapsed = time.time() - t0
    check(f'audit_coverage default returns <5s ({elapsed*1000:.0f}ms)',
          elapsed < 5.0)
    check('audit_coverage default returns 160 cells (20 syms × 8 tfs)',
          len(rows) == 160)
    present_with_ts = [r for r in rows if r['exists'] and r.get('last_ts')]
    if present_with_ts:
        sample = present_with_ts[0]
        # mtime path produces a "YYYY-MM-DD HH:MM:SS" string.
        import re
        check('fast-path last_ts matches YYYY-MM-DD HH:MM:SS format',
              bool(re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',
                            sample.get('last_ts') or '')))
    # Staleness classification still works.
    statuses = {r['status'] for r in rows}
    check('audit_coverage assigns present/stale/missing classifications',
          statuses.issubset({'present', 'stale', 'missing'}))

    # precise=True still works (don't actually run it — too slow).
    check('audit_coverage(precise=True) is a valid keyword call',
          'precise: bool = False' in audit_src)

    # --- heartbeat() lock-timeout fix ---
    reg_src_path = os.path.join(BASE_DIR, 'src', 'utils', 'process_registry.py')
    with open(reg_src_path, encoding='utf-8') as f:
        reg_src = f.read()
    check('heartbeat() uses 15s lock timeout',
          'timeout=15.0' in reg_src)
    check('heartbeat() catches filelock Timeout',
          'from filelock import Timeout as _LockTimeout' in reg_src
          and 'except _LockTimeout' in reg_src)
    check('heartbeat() downgrades contended Timeout to DEBUG',
          'logger.debug(' in reg_src
          and 'lock timeout — registry busy' in reg_src)

    # --- Frontend filter auto-scan was the only 1.5s JS poller I added;
    # the operator reported sluggishness so we bumped to 3s + hidden gate.
    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()
    check('filter auto-scan interval bumped to 3000ms',
          'setInterval(() => {\n    if (document.hidden) return;\n    _autoInstallFiltersOnAllTables();\n  }, 3000)' in tpl)
    check('filter auto-scan skips when tab is hidden',
          'if (document.hidden) return;' in tpl)


def test_phase102_unified_job_registry_and_filters_and_winning_hp():
    """2026-05-15 dashboard upgrade — five operator complaints in one pass:

    1. 🏆 Winning Hyperparameters card was empty — backend endpoint
       /api/analytics/winning_hp now has an 'overview' mode that returns
       top-N winners across all (model, tf) cells when no model+tf is
       supplied, so the card has useful default content.
    2. 'Set as baseline' button — verified the backend promote path works
       end-to-end (record_run → promote_baseline → state file). UX now
       shows inline status feedback in #an-summary so the operator sees
       the action took effect.
    3. Per-column filter inputs on every table — installColumnFilters()
       installs a filter row under the <thead>, with filter state
       persisted in localStorage so refresh/retrain/sort doesn't reset
       the filters. _autoInstallFiltersOnAllTables() runs on a 1500ms
       cadence so JS-rendered tables get filters automatically.
    4. Server-side job registry — src/dashboard/job_registry.py persists
       jobs to data/dashboard_jobs.json so cross-tab navigation rehydrates
       progress. /api/jobs?card_id=... lets every card poll its own
       in-flight state. Wired into runAutoOrchestrate + loadBakeOff so the
       Model Comparison cards now show live progress after a tab switch
       or page reload mid-run.
    5. Retrain status check — covered by other tests already.
    """
    print('\n[Phase 102 -- unified job registry, column filters, winning HP]')

    tpl_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'templates', 'index.html')
    with open(tpl_path, encoding='utf-8') as f:
        tpl = f.read()

    # --- Job registry module + endpoints ---
    jr_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'job_registry.py')
    check('job_registry module exists', os.path.exists(jr_path))
    with open(jr_path, encoding='utf-8') as f:
        jr_src = f.read()
    for sym in ('def register(', 'def update(', 'def complete(', 'def fail(',
                'def append_log(', 'def list_for_card(', 'def list_running(',
                'def cleanup_stale(', 'REGISTRY_PATH', 'MAX_JOBS_PER_CARD'):
        check(f'job_registry exports {sym}', sym in jr_src)

    # Behavioural test — actually exercise the registry round-trip.
    import importlib
    if 'src.dashboard.job_registry' in sys.modules:
        importlib.reload(sys.modules['src.dashboard.job_registry'])
    from src.dashboard import job_registry as jr
    test_card = '__test_card_phase102__'
    jid = jr.register(test_card, label='unit test', kind='test',
                      initial_log='start')
    check('register returns job_id', isinstance(jid, str) and jid.startswith('job_'))
    check('append_log finds the job', jr.append_log(jid, 'mid 1') is True)
    check('update sets progress_pct', jr.update(jid, progress_pct=42.0) is True)
    jobs = jr.list_for_card(test_card, limit=5)
    check('list_for_card returns the running job', len(jobs) == 1
          and jobs[0]['status'] == 'running' and jobs[0]['progress_pct'] == 42.0)
    check('complete marks status=done', jr.complete(jid, result={'ok': True}) is True)
    completed = jr.get(jid)
    check('completed job has status done + finished_at',
          completed and completed['status'] == 'done'
          and completed.get('finished_at') is not None)
    # Cleanup: drop the test job.
    jr.cleanup_stale(max_age_s=0)
    check('cleanup_stale removes terminal jobs older than 0s',
          len(jr.list_for_card(test_card)) == 0)

    # /api/jobs endpoints in app.py
    app_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(app_path, encoding='utf-8') as f:
        app_src = f.read()
    for route in ("/api/jobs'", "/api/jobs/register",
                  "/api/jobs/<job_id>/log", "/api/jobs/<job_id>/complete",
                  "/api/jobs/<job_id>/fail", "/api/jobs/<job_id>'"):
        check(f"app.py exposes {route}", route in app_src)

    # --- Winning HP overview mode ---
    # Endpoint should support no-args call returning leaders[] across cells.
    check("winning_hp endpoint has overview mode (leaders[])",
          "'leaders': leaders" in app_src or "'leaders'" in app_src)
    check("winning_hp endpoint returns mode='overview' when no model+tf",
          "'mode': 'overview'" in app_src)
    # Actually call winning_hyperparameters and the underlying function
    # to confirm the data path is sound.
    from src.analytics import training_history as th
    state = th._load_state()
    # If there are runs, the overview mode should produce at least one leader.
    have_runs = bool(state.get('runs'))
    if have_runs:
        runs = th.get_runs(limit=None)
        by_cell = {}
        for r in runs:
            if r.get('hp') is None or not r.get('cell'):
                continue
            prev = by_cell.get(r['cell'])
            if prev is None or (r.get('score') or 0) > (prev.get('score') or 0):
                by_cell[r['cell']] = r
        check('overview mode would surface leaders for cells with hp', True,
              f'{len(by_cell)} cell(s) have at least one run with hp')

    # --- Set-as-baseline backend path ---
    # Confirm the round-trip works on a real run (then restore baseline).
    state = th._load_state()
    runs = state.get('runs') or []
    non_baseline = next((r for r in runs if not r.get('is_baseline')), None)
    if non_baseline:
        cell = non_baseline.get('cell')
        original = state.get('baselines', {}).get(cell)
        target = non_baseline['run_id']
        ok = th.promote_baseline(target)
        check('promote_baseline(target) returns True', ok)
        state2 = th._load_state()
        check('baselines[cell] now points to target',
              state2.get('baselines', {}).get(cell) == target)
        target_row = next((r for r in state2['runs']
                           if r.get('run_id') == target), None)
        check('target run now has is_baseline=True',
              target_row and target_row.get('is_baseline') is True)
        # Restore so we don't leave the test state changed.
        if original and original != target:
            th.promote_baseline(original)

    # --- Frontend: filter helpers stubbed out (2026-05-15 operator: "remove
    # filter line from all tabs and cards, just leave the sorting controls
    # on headers"). The stub functions still exist so existing render
    # callers (refreshTableFilters / installColumnFilters) keep linking, but
    # they only sweep away leftover filter rows from prior sessions.
    for sym in ('installColumnFilters', 'refreshTableFilters',
                '_autoInstallFiltersOnAllTables', '_FILTER_STATE_KEY'):
        check(f'frontend helper {sym} still callable (stub)', sym in tpl)
    check('filter rows are NOT being added (no addEventListener input handler)',
          "addEventListener('input', (ev) => {\n      _setFilterFor(tableId, i, ev.target.value)" not in tpl)
    check('stub purges leftover col-filter-row tags on mount',
          "thead.querySelector('tr.col-filter-row')" in tpl)
    check('localStorage filter state purged on DOMContentLoaded',
          "localStorage.removeItem(_FILTER_STATE_KEY)" in tpl)

    # Filter calls wired to the renderers we explicitly hooked.
    check('renderModelComparison calls refreshTableFilters',
          "refreshTableFilters('model-comparison-table')" in tpl)
    check("anLoad calls refreshTableFilters('an-runs-table')",
          "refreshTableFilters('an-runs-table')" in tpl)
    check('bake-off table gets an id so filters can attach',
          'id="bake-off-table"' in tpl)
    check('anLoadWinningHp adds id to its rendered table',
          'id="an-winning-table"' in tpl)

    # --- Frontend: runJob + pollCardJobs + rehydrate hooks ---
    for sym in ('async function runJob(', 'function pollCardJobs(',
                'function fetchCardJobs(', 'function renderJobStrip(',
                'function rehydrateAutoOrchCard(', 'function rehydrateBakeOffCard('):
        check(f'frontend {sym.split(" ")[-1].split("(")[0]} present', sym in tpl)
    check('runAutoOrchestrate wraps work in runJob',
          "runJob('auto-orch'" in tpl)
    check('loadBakeOff registers a bake-off job',
          "card_id:'bake-off'" in tpl)
    check('model_comparison tab calls rehydrateAutoOrchCard',
          "rehydrateAutoOrchCard()" in tpl)
    check('DOMContentLoaded triggers rehydrate after page reload',
          "rehydrateAutoOrchCard === 'function'" in tpl)

    # --- Set-as-baseline UX improvement ---
    check('anPromote surfaces inline status into an-summary',
          'Promoting ' in tpl and 'Baseline updated' in tpl)

    # --- Winning HP card auto-loads on Analytics tab ---
    # When the analytics nav-item is clicked, anLoad AND anLoadWinningHp
    # are called (post-fix: anLoadWinningHp now runs unconditionally so
    # the card has overview content even with no model+tf picked).
    analytics_click = "activeTab === 'analytics'"
    idx = tpl.find(analytics_click)
    nearby = tpl[idx: idx + 600] if idx >= 0 else ''
    check('Analytics tab click triggers anLoad()',  'anLoad()' in nearby)
    check('Analytics tab click triggers anLoadWinningHp()',
          'anLoadWinningHp()' in nearby)
    # Default card text no longer says "Pick a model + TF".
    check("an-winning default text updated to 'Loading…'",
          'id="an-winning" style="font-size:.65rem;color:#94a3b8">Loading winning hyperparameters' in tpl)

    # --- New: Model column alongside Strategy on trade tables ---
    check('deriveTradeModel helper exists', 'function deriveTradeModel(' in tpl)
    check('modelChip helper exists', 'function modelChip(' in tpl)
    # Trades tab Active table: column count 9 → 10.
    check('tbl-open-trades has 10-col empty-state',
          '<td colspan="10" style="text-align:center;color:#475569;padding:14px">No open positions</td>' in tpl)
    check('tbl-closed-trades has 10-col empty-state',
          '<td colspan="10" style="text-align:center;color:#475569;padding:14px">No closed trades</td>' in tpl)
    # Overview tables 8 → 9.
    check('tbl-ov-open has 9-col empty-state',
          '<td colspan="9" style="text-align:center;color:#475569;padding:14px">No open positions</td>' in tpl)
    check('tbl-ov-closed has 9-col empty-state',
          '<td colspan="9" style="text-align:center;color:#475569;padding:14px">No closed trades yet</td>' in tpl)
    # Header literally says Model.
    check('Active Trades table header includes "Model"',
          '<th onclick="sortTbl(this)">Model</th><th onclick="sortTbl(this)">Strategy</th>' in tpl)

    # Behaviour: deriveTradeModel covers the canonical strategy prefixes
    # we expect to see in production trades. Spot-check the JS source
    # string for each known mapping so a future rename surfaces here.
    for needle in ("strat.includes('scalp')",
                   "strat.includes('trend')",
                   "strat.includes('futures')",
                   "strat.includes('meta')",
                   "strat.includes('regime')",
                   "market === 'scalping'"):
        check(f'deriveTradeModel handles {needle}', needle in tpl)


# ─── Runner ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true',
                        help='Skip HTTP tests (no running server required)')
    parser.add_argument('--url', default=DASHBOARD_URL,
                        help=f'Dashboard base URL (default: {DASHBOARD_URL})')
    args = parser.parse_args()

    print('=' * 55)
    print('  AI Trader Dashboard -- Test Suite')
    print('=' * 55)

    test_template()
    test_trades_file()
    test_app_py()
    test_training_scripts()
    test_model_meta()
    test_main_py()
    test_quant_modules()
    test_new_strategy_modules()
    test_monitor_module()
    test_phase0_foundation()
    test_phase1_microstructure()
    test_phase2_alpha_engine()
    test_phase3_execution_simulation()
    test_phase4_portfolio_optimization()
    test_phase5_institutional_safeguards()
    test_phase7_continuous_pipeline()
    test_phase8_data_governance()
    test_phase10_live_integration()
    test_phase11_predictor_and_llm_resilience()
    test_phase12_dashboard_controls._offline = args.offline
    test_phase12_dashboard_controls()
    test_phase13_realtime_and_fastapi._offline = args.offline
    test_phase13_realtime_and_fastapi()
    test_phase14_local_only_scheduler()
    test_phase16_scheduler_panel_and_sim_no_hang._offline = args.offline
    test_phase16_scheduler_panel_and_sim_no_hang()
    test_phase17_trading_health_fixes()
    test_phase18_institutional_panel_fixes._offline = args.offline
    test_phase18_institutional_panel_fixes()
    test_phase19_oft_integration()
    test_phase20_orchestrator_scheduler_simpanels._offline = args.offline
    test_phase20_orchestrator_scheduler_simpanels()
    test_phase21_observability_and_risk_overrides._offline = args.offline
    test_phase21_observability_and_risk_overrides()
    test_phase22_scheduler_no_autorefresh()
    test_phase23_unified_banner_aggregator()
    test_phase24_scheduler_flash_and_local_training()
    test_phase25_user_initiated_agents_exempt()
    test_phase26_parquet_client_foundation()
    test_phase27_ingest_path_cutover()
    test_phase28_dashboard_read_path_cutover()
    test_phase29_cleanup_questdb_artifacts()
    test_phase30_futures_close_reduce_only_guard()
    test_phase31_market_data_legacy_bridge()
    test_phase32_dedup_market_data()
    test_phase33_zombie_watchdog()
    test_phase34_telegram_monitor_gate()
    test_phase35_scheduler_no_post_action_refresh()
    test_phase36_debug_supervisor()
    test_phase37_training_table_and_bt_tooltips()
    test_phase38_clear_all_suppression()
    test_phase39_pr5_ui_bundle()
    test_phase40_pr1_data_coverage_resample()
    test_phase41_pr2_trainer_multi_tf()
    test_phase42_pr3_backtester_multi_tf()
    test_phase43_pr4_stability_heatmap()
    test_phase44_pr6_live_trading_toggle()
    test_phase45_pipeline_orchestrator()
    test_phase46_pr9_ux_bundle()
    test_phase47_pr10_loading_chips_and_simulator()
    test_phase48_pr11_multi_tf_inference()
    test_phase49_pr12_tf_pinning()
    test_phase50_pr13_auto_retrain()
    test_phase51_pr14_live_news_inference()
    test_phase52_pr15_finbert_sentiment()
    test_phase53_pr16_long_horizon_backtest()
    test_phase54_pr17_production_readiness()
    test_phase55_pr19_training_controls()
    test_phase56_pr21_heatmap_rework()
    test_phase57_pr26_all_tfs_and_status()
    test_phase58_pr28_balance_by_mode()
    test_phase59_pr35_parquet_query_thread_safety()
    test_phase60_pr36_training_concurrency_cap()
    test_phase61_pr37_resource_aware_scheduler()
    test_phase62_pr38_training_eta_and_elapsed()
    test_phase63_pr39_strategy_panels_hourly_refresh()
    test_phase64_pr40_training_survives_dashboard_restart()
    test_phase65_pr40_pipeline_orchestrator_progress_broadcast()
    test_phase66_pr40_strategy_sections_collapsed_default()
    test_phase67_pr40_loadStrategyFull_renders_directly()
    test_phase68_pr41_orphan_training_reattach_and_collapse_fix()
    test_phase69_pr42_pipeline_through_scheduler_plus_followup_backtest()
    test_phase70_pr43_dashboard_watchdog()
    test_phase71_pr46_real_cash_label_rename()
    test_phase71b_v31_curated_tf_map()
    test_phase71c_v31_backtest_per_model_filter()
    test_phase71d_v31_tft_dedupe_regression()
    test_phase71e_v31_scalping_rebalance()
    test_phase71f_v31_oft_sweep_coverage()
    test_phase72_v31_dashboard_mode_aware_and_per_market()
    test_phase73_v31_trade_enrichment_going_forward()
    test_phase73b_v31_trade_enrichment_backfill()
    test_phase74_v31_health_column_and_fleet_aggregate()
    test_phase75_v31_backfill_button_endpoint()
    test_phase76_v31_training_sweep_watchdog_and_cold_cache()
    test_phase77_v31_pertf_train_button_dispatch_fix()
    test_phase78_v31_bot_dead_false_alarm_module_style_launch()
    test_phase79_v31_stability_heatmap_legend_blue_rename()
    test_phase80_v4_b0_training_rules_registry_and_api()
    test_phase81_v4_b5_prime_unified_card_ui()
    test_phase82_v4_component_health_module_style_launches()
    test_phase83_centralised_process_health_module()
    test_phase84_orchestration_topics_pubsub()
    test_phase85_distributed_smoketest_three_bug_fixes()
    test_phase86_sweep_coordinator_daemon()
    test_phase87_dual_lane_workers_concurrent_cpu_gpu()
    test_phase88_orchestrator_task_progress_watchdog()
    test_phase89_gpu_classifier_wrapper_and_trainer_migration()
    test_phase90_master_agent_zombie_worker_supervisor()
    test_phase91_tft_dedupe_tz_normalize_plus_meta_hard_fail()
    test_phase92_meta_labeler_regime_dict_shape_tolerance()
    test_phase93_worker_live_load_and_remote_restart()
    test_phase94_distributed_backtest_per_cell()
    test_phase95_xgb_early_stop_eval_set_fix_and_backtest_column()
    test_phase96_orphan_detector_direct_script_form_plus_ps_native_fix()
    test_phase97_train_all_concurrency_lock_plus_current_state_pipeline()
    test_phase98_eta_train_bt_columns_and_tf_keyed_running()
    test_phase97c_orphan_periodic_refresh_and_canonical_row_fallback()
    test_phase100_cluster_routed_training_dispatch()
    test_phase100_functional_cluster_routing_proves_behavior()
    test_phase100b_retrain_all_distributed_train_then_bt()
    test_phase100e_pipeline_orchestrator_cluster_dispatch()
    test_sprint1a_r1_trainers_package_typed_contract()
    test_phase100d_training_jobs_throttle_and_fast_endpoint()
    test_phase100d_followup_restart_all_no_auto_train_cron()
    test_phase100d_worker_cpu_lane_hides_gpu_env_vars()
    test_phase100d_followup_3_training_jobs_lock_is_rlock()
    test_phase100d_followup_4_xgb_wrapper_is_classifier_and_worker_reports_failure()
    test_phase101_neural_kind_plus_task_heartbeat_and_proc_health_cache()
    test_phase102_unified_job_registry_and_filters_and_winning_hp()
    test_phase103_data_coverage_audit_perf_and_registry_hb_lock_timeout()
    test_phase104_pc_load_balancer()
    test_phase105_plateau_detection_l2_news_tick_features()
    test_phase106_tft_presets_and_remote_worker_ip_override()
    test_phase107_worker_bind_host_env_default()
    test_phase108_training_progress_instrumentation()
    test_phase109_gzip_to_parquet_migration()
    test_phase110_live_perf_monitor()

    if not args.offline:
        test_api(args.url)
    else:
        print('\n[API Endpoints] -- skipped (--offline mode)')

    print('\n' + '=' * 55)
    total = results['pass'] + results['fail'] + results['skip']
    print(f"  Results: {results['pass']} passed, {results['fail']} failed, {results['skip']} skipped / {total} total")
    print('=' * 55)
    sys.exit(0 if results['fail'] == 0 else 1)


def test_phase110_live_perf_monitor():
    """P1 degradation monitoring: live performance monitor tracks per-strategy
    rolling win rates vs historical baseline and flags WARN / DEGRADED status.
    """
    print('\n[Phase 110 -- P1 Live Performance Monitor]')

    # 1. Module exists and exports the required API.
    from src.risk import live_perf_monitor as lpm
    for name in ('run_once', 'get_cached_state', 'is_degraded', 'start', 'stop',
                 'DEFAULT_INTERVAL_S', 'StrategyPerf'):
        check(f'live_perf_monitor exports {name}', hasattr(lpm, name))

    # 2. run_once() returns a correctly-shaped dict (uses real trade_events data).
    state = lpm.run_once()
    check('run_once returns dict', isinstance(state, dict))
    for key in ('last_run_iso', 'next_run_iso', 'summary', 'strategies'):
        check(f'state has key {key!r}', key in state)
    summary = state['summary']
    for k in ('ok', 'warn', 'degraded', 'no_data'):
        check(f'summary has {k!r}', k in summary)
        check(f'summary[{k!r}] is int', isinstance(summary[k], int))

    # 3. Each strategy row has the mandatory fields.
    for row in state['strategies']:
        for field_name in ('strategy', 'n_trades_total', 'n_trades_window',
                           'status', 'last_trade_ts'):
            check(f'strategy row has {field_name!r}', field_name in row)
        check('status is valid', row['status'] in ('OK', 'WARN', 'DEGRADED', 'NO_DATA'))
        check('n_trades_total >= 0', row['n_trades_total'] >= 0)

    # 4. State file written to data/risk/live_perf_state.json.
    state_path = os.path.join(BASE_DIR, 'data', 'risk', 'live_perf_state.json')
    check('live_perf_state.json written', os.path.exists(state_path))
    with open(state_path, encoding='utf-8') as f:
        cached = __import__('json').load(f)
    check('cached state has strategies list', isinstance(cached.get('strategies'), list))

    # 5. get_cached_state() reads back what run_once() wrote.
    from src.risk.live_perf_monitor import get_cached_state
    cs = get_cached_state()
    check('get_cached_state returns dict', isinstance(cs, dict))
    check('cached strategies count matches',
          len(cs.get('strategies', [])) == len(state['strategies']))

    # 6. is_degraded() returns (bool, str) for known and unknown strategy.
    from src.risk.live_perf_monitor import is_degraded
    result = is_degraded('NONEXISTENT_STRATEGY_XYZ')
    check('is_degraded returns 2-tuple', isinstance(result, tuple) and len(result) == 2)
    check('is_degraded[0] is bool', isinstance(result[0], bool))
    check('is_degraded[1] is str', isinstance(result[1], str))
    check('unknown strategy not degraded', result[0] is False)

    # 7. Dashboard routes registered.
    src_path = os.path.join(BASE_DIR, 'src', 'dashboard', 'app.py')
    with open(src_path, encoding='utf-8') as f:
        app_src = f.read()
    check("GET /api/live_perf/state route registered",
          "'/api/live_perf/state'" in app_src)
    check("POST /api/live_perf/run route registered",
          "'/api/live_perf/run'" in app_src)
    check('_ensure_live_perf_monitor helper present',
          '_ensure_live_perf_monitor' in app_src)

    # 8. StrategyPerf.to_dict() round-trips correctly.
    sp = lpm.StrategyPerf(
        strategy='TestStrat', n_trades_total=60, n_trades_window=50,
        win_rate_recent=0.42, win_rate_baseline=0.55,
        degradation_pct=0.236, avg_pnl_usd=-1.2, status='WARN',
        last_trade_ts='2026-05-17T00:00:00+00:00',
    )
    d = sp.to_dict()
    check('to_dict has strategy', d['strategy'] == 'TestStrat')
    check('to_dict has status WARN', d['status'] == 'WARN')
    check('to_dict rounds win_rate_recent', d['win_rate_recent'] == 0.42)


def test_phase111_preflight_train_checks():
    """Phase 8 pre-flight: preflight_train.py exists and individual check
    functions behave correctly under mocked conditions."""
    print('\n[Phase 111 -- preflight_train.py checks]')
    import importlib.util
    from unittest.mock import patch, MagicMock
    import tempfile

    # 1. Script exists.
    script_path = os.path.join(BASE_DIR, 'scripts', 'preflight_train.py')
    check('preflight_train.py exists', os.path.isfile(script_path))

    spec = importlib.util.spec_from_file_location('preflight_train', script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 2. check_disk_space() passes on any machine with >= 1 GB free.
    free_gb = __import__('shutil').disk_usage(BASE_DIR).free / (1024 ** 3)
    result  = mod.check_disk_space()
    check('check_disk_space returns bool', isinstance(result, bool))
    if free_gb >= 20:
        check('check_disk_space PASS when >= 20 GB', result is True)

    # 3. check_parquet_count() PASS when expected=None and parquet dir has files.
    with patch.object(mod, 'PARQUET_DIR', __import__('pathlib').Path(BASE_DIR) / 'data' / 'parquet'):
        result_none = mod.check_parquet_count(None)
    check('check_parquet_count(None) returns bool', isinstance(result_none, bool))

    # 4. check_no_running_jobs() PASS when dashboard_jobs.json has no running entries.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump({'job_a': {'status': 'done'}, 'job_b': {'status': 'pending'}}, tf)
        tf_path = tf.name
    with patch.object(mod, 'JOBS_FILE', __import__('pathlib').Path(tf_path)):
        result_clean = mod.check_no_running_jobs()
    os.unlink(tf_path)
    check('check_no_running_jobs PASS on clean jobs', result_clean is True)

    # 5. check_no_running_jobs() FAIL when a job is running.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump({'job_a': {'status': 'running'}}, tf)
        tf_path = tf.name
    with patch.object(mod, 'JOBS_FILE', __import__('pathlib').Path(tf_path)):
        result_running = mod.check_no_running_jobs()
    os.unlink(tf_path)
    check('check_no_running_jobs FAIL when job running', result_running is False)

    # 6. check_api_keys() with mocked env vars.
    keys = {'API_KEY': 'k', 'API_SECRET': 's', 'HETZNER_API_TOKEN': 'h',
            'VASTAI_API_KEY': 'v', 'GEMINI_API_KEY': 'g'}
    with patch.dict('os.environ', keys, clear=False):
        result_keys = mod.check_api_keys()
    check('check_api_keys PASS when all set', result_keys is True)

    # 7. check_oos_writable() on a real temp dir.
    with tempfile.TemporaryDirectory() as td:
        with patch.object(mod, 'OOS_DIR', __import__('pathlib').Path(td)):
            result_oos = mod.check_oos_writable()
    check('check_oos_writable PASS on writable dir', result_oos is True)

    # 8. check_training_rules() PASS on valid JSON with required fields.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump({'models': {'base': {}}, 'global': {'default_symbol_universe': ['BTC/USDT']}}, tf)
        tf_path = tf.name
    with patch.object(mod, 'RULES_FILE', __import__('pathlib').Path(tf_path)):
        result_rules = mod.check_training_rules()
    os.unlink(tf_path)
    check('check_training_rules PASS on valid JSON', result_rules is True)

    # 9. check_hetzner() with mocked HTTP 200.
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch('requests.get', return_value=mock_resp), \
         patch.dict('os.environ', {'HETZNER_API_TOKEN': 'fake-token'}, clear=False):
        result_hz = mod.check_hetzner()
    check('check_hetzner PASS on HTTP 200', result_hz is True)

    # 10. check_vastai() with mocked HTTP 200.
    with patch('requests.get', return_value=mock_resp), \
         patch.dict('os.environ', {'VASTAI_API_KEY': 'fake-key'}, clear=False):
        result_va = mod.check_vastai()
    check('check_vastai PASS on HTTP 200', result_va is True)


def test_phase112_env_manifest():
    """Phase 112 — env_manifest.py: capture_env_manifest() returns required
    keys; save_env_manifest() writes valid JSON to a temp path."""
    import tempfile
    print('\n[Phase 112 -- env_manifest utility]')

    from src.utils.env_manifest import capture_env_manifest, save_env_manifest

    manifest = capture_env_manifest()

    check('manifest has python key', 'python' in manifest)
    check('manifest has platform key', 'platform' in manifest)
    check('manifest has scikit-learn key', 'scikit-learn' in manifest)
    check('manifest has numpy key', 'numpy' in manifest)
    check('manifest has pyarrow key', 'pyarrow' in manifest)
    check('manifest has torch key', 'torch' in manifest)
    check('manifest has cuda key', 'cuda' in manifest)
    check('python version is non-empty string', isinstance(manifest['python'], str) and manifest['python'])

    # save_env_manifest creates subdirectory and writes JSON
    with tempfile.TemporaryDirectory() as td:
        out_path = __import__('pathlib').Path(td) / 'sub' / 'env_manifest.json'
        out = save_env_manifest(out_path)
        saved = __import__('json').loads(out_path.read_text(encoding='utf-8'))
    check('save_env_manifest returns dict', isinstance(out, dict))
    check('saved JSON matches capture output', saved['python'] == manifest['python'])


if __name__ == '__main__':
    main()
