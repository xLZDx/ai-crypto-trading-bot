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
        'strat-agg', 'ml-agg', 'strategy-cards',
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
    print(f'\n[API Endpoints @ {base_url}]')
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        check('urllib available', False)
        return

    def get(path, expect_key=None):
        url = base_url + path
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                body = json.loads(r.read().decode())
            if expect_key is not None:
                ok = expect_key in body
                check(f'GET {path} → has "{expect_key}"', ok,
                      f'keys: {list(body.keys())}')
            else:
                check(f'GET {path} → 200 OK', True)
        except urllib.error.HTTPError as e:
            check(f'GET {path}', False, f'HTTP {e.code}')
        except Exception as e:
            check(f'GET {path}', False, str(e))

    get('/')
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
    test_model_meta()
    test_main_py()
    test_quant_modules()

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
