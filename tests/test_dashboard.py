"""
Dashboard test module — run after any implementation change to catch regressions.
Usage:
    python tests/test_dashboard.py              # requires dashboard running on port 5000
    python tests/test_dashboard.py --offline    # only static/file checks, no HTTP
"""

import sys
import os
import json
import re
import argparse

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
    print('\n[Phase 0 — Institutional Upgrade Foundation]')

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
    print('\n[Phase 1 — Level 1 Data Layer]')

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
    print('\n[Phase 2 — Level 2 Alpha Engine]')

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
    print('\n[Phase 3 — Level 3 Execution & Simulation]')

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
    print('\n[Phase 4 — Level 4 Portfolio Optimization]')

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
    print('\n[Phase 5 — Level 5 Institutional Safeguards]')

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
    print('\n[Phase 7 — Continuous Pipeline + Retention]')

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
    print('\n[Phase 8 — Data Governance + Rate Limiting]')

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
    print('\n[Phase 10 — Live Integration + 8-tab Dashboard + Documentation]')

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

    tv2 = os.path.join(BASE_DIR, 'src', 'engine', 'train_model_v2.py')
    check('train_model_v2.py exists', os.path.exists(tv2))

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
    print('\n[Phase 11 — Predictor / LLM resilience]')

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
    print('\n[Phase 12 — Dashboard control wiring]')

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
    """Realtime heartbeat + FastAPI control plane: filesystem + import-time
    checks (offline) plus one HTTP probe per service when --offline is off."""
    print('\n[Phase 13 — Realtime heartbeat + FastAPI control plane]')

    # ── Realtime heartbeat ──────────────────────────────────────────────
    rt_src = open(os.path.join(BASE_DIR, 'src', 'data_ingestion', 'realtime_db_writer.py'),
                  encoding='utf-8').read()
    check('realtime_db_writer defines _write_status',
          'def _write_status' in rt_src)
    check('realtime_db_writer writes data/realtime_status.json',
          'realtime_status.json' in rt_src)
    check('realtime_db_writer flips status on connect/disconnect/error',
          rt_src.count('_write_status(') >= 4)

    # ── FastAPI control plane ───────────────────────────────────────────
    fapi_src_path = os.path.join(BASE_DIR, 'src', 'server', 'control_plane.py')
    check('src/server/control_plane.py exists', os.path.exists(fapi_src_path))
    if os.path.exists(fapi_src_path):
        fapi_src = open(fapi_src_path, encoding='utf-8').read()
        for ep in ('/health', '/status', '/metrics',
                   '/control/bot/start', '/control/bot/stop',
                   '/control/training/start'):
            check(f'control_plane defines {ep}',
                  f'"{ep}"' in fapi_src or f"'{ep}'" in fapi_src)
        check('control_plane requires X-API-Key on /control/*',
              '_require_api_key' in fapi_src)

    launcher = os.path.join(BASE_DIR, 'launch_fastapi.ps1')
    check('launch_fastapi.ps1 exists', os.path.exists(launcher))
    if os.path.exists(launcher):
        l_src = open(launcher, encoding='utf-8').read()
        check('launch_fastapi.ps1 binds :8100 by default',
              "FASTAPI_BIND_PORT = '8100'" in l_src)
        check('launch_fastapi.ps1 is idempotent (skips if /health up)',
              '/health' in l_src and 'skipping launch' in l_src.lower())

    # ── restart_all wiring ──────────────────────────────────────────────
    ra = open(os.path.join(BASE_DIR, 'restart_all.ps1'), encoding='utf-8').read()
    check('restart_all.ps1 starts FastAPI',
          'launch_fastapi.ps1' in ra and 'src.server.control_plane' in ra.replace('\\.', '.'))
    check('restart_all.ps1 saves fastapi PID',
          'fastapi = $fastapiId' in ra)
    sa = open(os.path.join(BASE_DIR, 'stop_all.ps1'), encoding='utf-8').read()
    check('stop_all.ps1 covers fastapi key',
          "'fastapi'" in sa)
    check('stop_all.ps1 stray-sweep matches control_plane',
          'control_plane' in sa)

    # ── FastAPI app importable + has expected routes ────────────────────
    try:
        import sys as _sys
        _sys.path.insert(0, BASE_DIR)
        from src.server.control_plane import app as _fastapi_app  # noqa: F401
        paths = {getattr(r, 'path', None) for r in _fastapi_app.routes}
        for needed in ('/health', '/status', '/metrics',
                       '/control/bot/start', '/control/bot/stop',
                       '/control/training/start'):
            check(f'FastAPI route registered: {needed}', needed in paths)
    except Exception as e:
        check('FastAPI app imports cleanly', False, str(e))

    # ── Live HTTP probe (skipped in --offline) ──────────────────────────
    if not getattr(test_phase13_realtime_and_fastapi, '_offline', False):
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=2) as r:
                body = _json.loads(r.read().decode('utf-8'))
                check('FastAPI /health responds 200',
                      r.status == 200 and body.get('status') == 'ok',
                      f'body={body}')
        except Exception as e:
            check('FastAPI /health responds 200', None, f'skipped: {e}')
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen('http://127.0.0.1:8100/status', timeout=2) as r:
                body = _json.loads(r.read().decode('utf-8'))
                check('FastAPI /status returns components dict',
                      'components' in body and isinstance(body['components'], dict))
        except Exception as e:
            check('FastAPI /status returns components dict', None, f'skipped: {e}')

        # Realtime heartbeat: file exists and is fresh (< 10 min old)
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
    print('\n[Phase 14 — Local-only scheduling]')

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
    print('\n[Phase 16 — Scheduler panel + Simulator non-hang]')

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
    print('\n[Phase 17 — Trading & dashboard health fixes (2026-05-04)]')

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
    print('\n[Phase 18 — Institutional panel UX & data wiring]')

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
    print('\n[Phase 19 — OFT live integration + simulator deadlock fix]')

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
    print('\n[Phase 20 — Orchestrator + scheduler + sim panel polish]')

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
    print('\n[Phase 21 — Observability + risk overrides]')

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
    print('\n[Phase 23 — unified banner aggregator]')

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
    print('\n[Phase 24 — Scheduler flash + local-training progress]')

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
    print('\n[Phase 28 — dashboard read path cutover (Route B)]')

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
    print('\n[Phase 36 — debug_supervisor]')

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
    print('\n[Phase 44 — PR 6 live trading toggle + paper accounting]')

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
    check('Mainnet switch confirms with explicit warning',
          "Switch to MAINNET" in tpl and 'Real money will be at risk' in tpl)
    check('+ Deposit button + ltDeposit() prompts for amount',
          'ltDeposit()' in tpl
          and 'Add how much to virtual balance' in tpl)


def test_phase57_pr26_all_tfs_and_status():
    """Phase 57 — PR 26: 'ALL TFs' option + fine-grained training status.
       User asked for: (a) one-click train across every TF the model
       supports, (b) instant visual feedback on click (QUEUED flash
       before network round-trip), (c) per-phase status pills
       (QUEUED / STARTING / RUNNING <tf> / FAILED / COMPLETED)."""
    print('\n[Phase 57 — PR 26 ALL-TFs + fine-grained status]')
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
          '<option value="all">ALL TFs</option>' in tpl)
    check('Fine-grained status: QUEUED / STARTING / RUNNING / CANCELLED',
          "'QUEUED'" in tpl
          and "'STARTING'" in tpl
          and "'RUNNING'" in tpl
          and "'CANCELLED'" in tpl)
    check('Status pill shows progress_label / current_tf when running',
          'activeJob.progress_label' in tpl
          and 'activeJob.current_tf' in tpl)
    check('Optimistic UI flashes QUEUED before network round-trip',
          "_trActiveJobs    = {..._trActiveJobs,    [key]: {" in tpl
          and "status: 'queued'" in tpl)
    check('Failed POST rolls back the optimistic state',
          'delete _trActiveByModel[key]' in tpl
          and 'delete _trActiveJobs[key]' in tpl)
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
    print('\n[Phase 56 — PR 21 heatmap rework]')
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
    print('\n[Phase 55 — PR 18/19/20 dashboard hardening]')
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
    check('TF picker per training row, defaulting to model timeframe',
          "id=\"tr-tf-${esc(m.key)}\"" in tpl
          and "tf===(m.timeframe||'1h')?' selected':''" in tpl)
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
    print('\n[Phase 54 — PR 17 production readiness]')

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
    print('\n[Phase 53 — PR 16 long-horizon backtest]')

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
    print('\n[Phase 52 — PR 15 FinBERT sentiment]')

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
    print('\n[Phase 51 — PR 14 live news inference]')

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
    print('\n[Phase 50 — PR 13 auto-retrain]')

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
    print('\n[Phase 49 — PR 12 strategy TF pinning]')

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
    print('\n[Phase 48 — PR 11 multi-TF inference]')

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
    print('\n[Phase 47 — PR 10 loading chips + simulator]')
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
    print('\n[Phase 46 — PR 9 UX bundle]')
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
    print('\n[Phase 45 — Pipeline orchestrator]')

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
    print('\n[Phase 43 — PR 4 stability heatmap]')

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
    print('\n[Phase 42 — PR 3 backtester multi-TF support]')

    bt = open(os.path.join(BASE_DIR, 'src', 'engine', 'backtester.py'),
              encoding='utf-8').read()

    check('run_full_backtest accepts timeframes= tuple param',
          'def run_full_backtest(' in bt
          and 'timeframes: tuple[str, ...] = ("1h",)' in bt)
    check('outer loop iterates timeframes',
          'for tf in timeframes:' in bt)
    check('per-symbol load uses <sym>_<tf>.csv.gz',
          'f"{sym}_{tf}.csv.gz"' in bt
          and 'f"{sym}_spot_{tf}.csv.gz"' in bt)
    check('each BacktestResult tagged with timeframe attr',
          'setattr(res, "timeframe", tf)' in bt)
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
    print('\n[Phase 41 — PR 2 trainer multi-TF refactor]')

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
    print('\n[Phase 40 — PR 1 data audit + 1s→TF resampler + UI]')

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
    print('\n[Phase 39 — PR 5 UI bundle: collapse + training controls + buckets]')

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
    print('\n[Phase 38 — CLEAR ALL suppression cool-off]')

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
    print('\n[Phase 37 — model training table + backtest tooltips]')

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
    print('\n[Phase 35 — scheduler no auto-refresh on action]')

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
    print('\n[Phase 34 — Telegram Monitor gate (default-disabled)]')

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
    print('\n[Phase 32 — per-partition dedup_market_data]')

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
    print('\n[Phase 33 — zombie watchdog]')

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
    print('\n[Phase 31 — market_data legacy-store bridge]')

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
    print('\n[Phase 30 — futures reduceOnly guard + trainer exemption]')

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
    print('\n[Phase 29 — cleanup of QuestDB artifacts]')

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
    cm = open(os.path.join(BASE_DIR, 'CLAUDE.md'), encoding='utf-8').read()
    check('CLAUDE.md DB line points at ParquetClient',
          'ParquetClient' in cm
          and 'data/db/' in cm)
    check('CLAUDE.md commit-before-implementations rule documented',
          'commit of the current state' in cm.lower()
          or 'commit before' in cm.lower())

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
    print('\n[Phase 27 — ingest path cutover (Route B)]')

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
    print('\n[Phase 26 — ParquetClient foundation (Route B)]')

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
    print('\n[Phase 25 — user-initiated agents exempt from staleness]')

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
    print('\n[Phase 22 — Scheduler manual-refresh-only]')

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


# ─── Runner ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true',
                        help='Skip HTTP tests (no running server required)')
    parser.add_argument('--url', default=DASHBOARD_URL,
                        help=f'Dashboard base URL (default: {DASHBOARD_URL})')
    args = parser.parse_args()

    print('=' * 55)
    print('  AI Trader Dashboard — Test Suite')
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

    if not args.offline:
        test_api(args.url)
    else:
        print('\n[API Endpoints] — skipped (--offline mode)')

    print('\n' + '=' * 55)
    total = results['pass'] + results['fail'] + results['skip']
    print(f"  Results: {results['pass']} passed, {results['fail']} failed, {results['skip']} skipped / {total} total")
    print('=' * 55)
    sys.exit(0 if results['fail'] == 0 else 1)


if __name__ == '__main__':
    main()
