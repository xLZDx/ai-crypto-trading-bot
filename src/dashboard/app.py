import os
import sys
import threading
import subprocess
import re
from functools import wraps
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Redirect caches to project drive to protect C: drive
cache_dir = os.path.join(project_root, 'data', 'cache')
os.makedirs(os.path.join(cache_dir, 'temp'), exist_ok=True)
os.environ['TMP'] = os.path.join(cache_dir, 'temp')
os.environ['TEMP'] = os.path.join(cache_dir, 'temp')
os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')
os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')

from src.utils.safe_json import read_json, write_json

load_dotenv()
app = Flask(__name__)

# ─── Gemini model health cache ────────────────────────────────────────────────
# Ordered lightest-quota first so free-tier API keys succeed more often.
_AI_MODELS_CONFIG = [
    {"id": "gemini-2.0-flash-lite",          "name": "Gemini 2.0 Flash Lite",         "cost": "Free Tier", "thinking": "MED"},
    {"id": "gemini-2.0-flash-lite-001",      "name": "Gemini 2.0 Flash Lite 001",     "cost": "Free Tier", "thinking": "MED"},
    {"id": "gemini-2.0-flash",               "name": "Gemini 2.0 Flash",              "cost": "Free Tier", "thinking": "MED"},
    {"id": "gemini-2.0-flash-001",           "name": "Gemini 2.0 Flash 001",          "cost": "Free Tier", "thinking": "MED"},
    {"id": "gemini-2.5-flash-lite",          "name": "Gemini 2.5 Flash Lite",         "cost": "Free Tier", "thinking": "HIGH"},
    {"id": "gemini-2.5-flash",               "name": "Gemini 2.5 Flash",              "cost": "Free Tier", "thinking": "HIGH"},
    {"id": "gemini-3-flash-preview",         "name": "Gemini 3 Flash Preview",        "cost": "Free Tier", "thinking": "HIGH"},
    {"id": "gemini-3.1-flash-lite-preview",  "name": "Gemini 3.1 Flash Lite Preview", "cost": "Free Tier", "thinking": "HIGH"},
    {"id": "gemini-2.5-pro",                 "name": "Gemini 2.5 Pro",                "cost": "Paid / Rate Limited", "thinking": "HIGH"},
    {"id": "gemini-3-pro-preview",           "name": "Gemini 3 Pro Preview",          "cost": "Paid / Rate Limited", "thinking": "HIGH"},
    {"id": "gemini-3.1-pro-preview",         "name": "Gemini 3.1 Pro Preview",        "cost": "Paid / Rate Limited", "thinking": "HIGH"},
]

_GEMINI_MODELS = [m['id'] for m in _AI_MODELS_CONFIG]
_active_model: str | None = None   # best model confirmed available (not quota-tested)
_model_lock = threading.Lock()

def _probe_models_bg():
    """Discover available Gemini models via models.list() — consumes zero quota."""
    global _active_model
    api_key = os.getenv('GEMINI_API_KEY', '')
    if not api_key or api_key == 'your_api_key_here':
        return
    try:
        from google import genai as _gp
        client = _gp.Client(api_key=api_key)
        available = {
            m.name.replace('models/', '')
            for m in client.models.list()
            if 'generateContent' in (m.supported_actions or [])
        }
        for model_id in _GEMINI_MODELS:
            if model_id in available:
                with _model_lock:
                    _active_model = model_id
                return
    except Exception:
        pass
    with _model_lock:
        _active_model = None

# Probe on startup — non-blocking background thread
threading.Thread(target=_probe_models_bg, daemon=True).start()

# Re-probe every 30 minutes (was 5 min — reduced to avoid unnecessary API overhead)
def _schedule_probe():
    import time
    while True:
        time.sleep(1800)
        _probe_models_bg()
threading.Thread(target=_schedule_probe, daemon=True).start()

DASHBOARD_API_KEY = os.getenv("DASHBOARD_API_KEY", "")

if not DASHBOARD_API_KEY:
    import logging
    logging.getLogger(__name__).warning(
        "DASHBOARD_API_KEY is not set in .env — dashboard API is unprotected!"
    )


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not DASHBOARD_API_KEY:
            return f(*args, **kwargs)
        token = request.headers.get("X-API-Key", "")
        if token != DASHBOARD_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/')
def index():
    return render_template('index.html', api_key=DASHBOARD_API_KEY or '')


@app.route('/api/state')
@require_api_key
def get_state():
    state = read_json('data/state.json', default={"status": "No data", "last_signal": "UNKNOWN"})
    return jsonify(state)


@app.route('/api/control', methods=['GET'])
@require_api_key
def get_control():
    ctrl = read_json('data/control.json', default={"running": True})
    return jsonify(ctrl)


@app.route('/api/control', methods=['POST'])
@require_api_key
def set_control():
    try:
        data = request.json
        write_json('data/control.json', data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/trades')
@require_api_key
def get_trades():
    trades = read_json('data/trades.json', default=[])
    return jsonify({"trades": trades})


@app.route('/api/logs')
@require_api_key
def get_logs():
    try:
        with open('logs/trading.log', 'r', encoding='utf-8') as f:
            # Seek to end, read last 50 KB to avoid loading huge files
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 51200))
            tail = f.read()
        lines = tail.splitlines()
        # If we didn't read from the start, the first line may be partial — drop it
        if size > 51200 and lines:
            lines = lines[1:]
        return jsonify({"logs": lines[-500:]})
    except FileNotFoundError:
        return jsonify({"logs": ["No logs yet..."]})
    except Exception as e:
        return jsonify({"logs": [f"Error reading logs: {e}"]})


def _build_portfolio_context():
    try:
        state = read_json('data/state.json', default={})
        trades_raw = read_json('data/trades.json', default=[])
        trades = trades_raw if isinstance(trades_raw, list) else trades_raw.get('trades', [])
        
        safe_state = {k: v for k, v in state.items() if 'key' not in k.lower() and 'secret' not in k.lower()}

        from collections import Counter
        open_trades   = [t for t in trades if str(t.get('status', '')).upper() == 'OPEN']
        closed_trades = [t for t in trades if str(t.get('status', '')).upper() == 'CLOSED']

        wins      = [t for t in closed_trades if float(t.get('pnl_usdt') or 0) > 0]
        win_rate  = round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0
        total_pnl = sum(float(t.get('pnl_usdt') or 0) for t in closed_trades)

        strat_counts = Counter(t.get('market', 'SPOT') for t in closed_trades)
        strat_pnl = {}
        for t in closed_trades:
            m = t.get('market', 'SPOT')
            strat_pnl[m] = strat_pnl.get(m, 0) + float(t.get('pnl_usdt') or 0)

        _META = {
            'spot':     'models/btc_rf_model_meta.json',
            'scalping': 'models/scalping_model_meta.json',
            'futures':  'models/futures_short_model_meta.json',
            'trend':    'models/trend_model_meta.json',
            'tft':      'models/tft_model_meta.json',
        }
        ml_acc = {k: read_json(v, default={}).get('accuracy', 'N/A') for k, v in _META.items()}

        open_summary = []
        for t in open_trades[:10]:
            bp   = float(t.get('buy_price') or 0)
            cp   = float(t.get('current_price') or bp)
            amt  = float(t.get('amount_coin') or 0)
            upnl = float(t.get('unrealized_pnl') or ((cp - bp) * amt if bp else 0))
            open_summary.append(
                f"{t.get('symbol','?')} {t.get('side','LONG')}/{t.get('market','SPOT')} "
                f"entry={bp} cur={cp} upnl={round(upnl, 2)}"
            )

        context = (
            f"BOT STATE: {safe_state}\n"
            f"TOTAL CLOSED TRADES: {len(closed_trades)} | WIN RATE: {win_rate}%\n"
            f"TOTAL REALIZED PNL: {round(total_pnl, 2)} USDT\n"
            f"STRATEGY BREAKDOWN: {dict(strat_counts)} | PNL/STRATEGY: "
            f"{dict((k, round(v, 2)) for k, v in strat_pnl.items())}\n"
            f"ML MODEL ACCURACY: {ml_acc}\n"
            f"OPEN POSITIONS ({len(open_trades)}): {open_summary}\n"
            f"RECENT TRADES (last 20): {trades[-20:]}"
        )
        return trades, safe_state, context
    except Exception as e:
        return [], {}, f"Error building context: {str(e)}"


def _exec_bot_command(lower_msg):
    if any(w in lower_msg for w in ['close all', 'sell all', 'close everything']):
        try:
            from src.engine.trade_tracker import TradeTracker
            tracker = TradeTracker()
            closed, pnl = 0, 0.0
            for t in list(tracker.get_open_trades()):
                price = t.get('current_price') or t.get('buy_price', 0)
                if price:
                    r = tracker.close_trade_by_id(t['id'], float(price))
                    if r:
                        closed += 1
                        pnl += r.get('pnl_usdt', 0) or 0
            threading.Thread(
                target=lambda: subprocess.run(
                    [sys.executable, 'src/engine/train_all_models.py'],
                    capture_output=True, timeout=600
                ), daemon=True
            ).start()
            return 'close_all', f'Closed {closed} positions. Realized PnL: {round(pnl, 2)} USDT. ML retraining started.'
        except Exception as e:
            return 'close_all', f'Error: {e}'

    if any(w in lower_msg for w in ['close losing', 'close loss', 'cut losses', 'close red']):
        try:
            from src.engine.trade_tracker import TradeTracker
            tracker = TradeTracker()
            closed, pnl = 0, 0.0
            for t in list(tracker.get_open_trades()):
                bp = float(t.get('buy_price') or 0)
                cp = float(t.get('current_price') or bp)
                if bp and cp < bp:
                    r = tracker.close_trade_by_id(t['id'], cp)
                    if r:
                        closed += 1
                        pnl += r.get('pnl_usdt', 0) or 0
            return 'close_losing', f'Closed {closed} losing positions. Realized PnL: {round(pnl, 2)} USDT.'
        except Exception as e:
            return 'close_losing', f'Error: {e}'

    if any(w in lower_msg for w in ['stop bot', 'pause bot', 'stop trading', 'pause trading']):
        try:
            ctrl = read_json('data/control.json', default={'running': True})
            ctrl['running'] = False
            write_json('data/control.json', ctrl)
            return 'stop_bot', 'Bot stopped (control.running = False).'
        except Exception as e:
            return 'stop_bot', f'Error: {e}'

    if any(w in lower_msg for w in ['start bot', 'resume bot', 'start trading', 'resume trading']):
        try:
            ctrl = read_json('data/control.json', default={'running': True})
            ctrl['running'] = True
            write_json('data/control.json', ctrl)
            return 'start_bot', 'Bot started (control.running = True).'
        except Exception as e:
            return 'start_bot', f'Error: {e}'

    if any(w in lower_msg for w in ['retrain', 'train model', 'train all']):
        threading.Thread(
            target=lambda: subprocess.run(
                [sys.executable, 'src/engine/train_all_models.py'],
                capture_output=True, timeout=600
            ), daemon=True
        ).start()
        return 'retrain', 'ML model retraining started in background.'

    return None, None


@app.route('/api/chat', methods=['POST'])
@require_api_key
def chat():
    global _active_model
    try:
        try:
            from google import genai as _genai
            from google.genai import types as _gtypes
        except ImportError:
            return jsonify({"response": "The google-genai library is not installed. Run: pip install google-genai"})

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            return jsonify({"response": "⚠️ **Error:** Add `GEMINI_API_KEY=your_key` to the `.env` file."})

        req_data = request.get_json(silent=True) or {}
        user_message = str(req_data.get('message', '')).strip()
        if not user_message:
            return jsonify({"response": "Empty message received."})

        command, command_result = _exec_bot_command(user_message.lower())
        trades, safe_state, context = _build_portfolio_context()

        # Tool: article / YouTube link analysis
        url_match = re.search(r'(https?://[^\s]+)', user_message)
        if url_match:
            try:
                from importlib import import_module
                scraper = import_module('src.tools.web_scraper_bot')
                url = url_match.group(1)
                if 'youtube.com' in url or 'youtu.be' in url:
                    extracted_text = scraper.get_youtube_transcript(url)
                else:
                    extracted_text = scraper.get_article_text(url)
                user_message += f"\n\n[SYSTEM: Extracted content from link:\n{extracted_text[:30000]}]"
            except Exception as e:
                user_message += f"\n\n[SYSTEM: Could not extract link content: {e}]"

        if command:
            user_message += f"\n\n[SYSTEM: Bot command '{command}' executed — {command_result}]"

        system_prompt = (
            "You are an advanced AI Trading Assistant embedded in a crypto trading dashboard. "
            "You provide deep portfolio analytics, strategy analysis, and market insights. "
            "You CAN execute bot commands (close positions, stop/start bot, retrain models) — "
            "when a command is detected the system executes it and reports the result to you. "
            "Analyse the portfolio data, identify which strategies or ML models need improvement, "
            "and give actionable, data-driven advice. Be concise and professional.\n\n"
            f"PORTFOLIO CONTEXT:\n{context}"
        )

        _TRANSIENT = ['not found', '404', 'invalid argument', 'unknown model',
                      '429', 'quota', 'resource_exhausted',
                      '503', 'unavailable', 'high demand', 'overloaded']

        ctrl = read_json('data/control.json', default={})
        selected_model = ctrl.get('selected_ai_model')

        # Build model priority list.
        # If user selected a model, try it first; on quota/rate errors fall through
        # to the full fallback list so chat still works on free-tier keys.
        with _model_lock:
            preferred = _active_model
        if selected_model:
            rest = [m for m in _GEMINI_MODELS if m != selected_model]
            models_to_try = [selected_model] + rest
        elif preferred:
            models_to_try = [preferred] + [m for m in _GEMINI_MODELS if m != preferred]
        else:
            models_to_try = _GEMINI_MODELS

        client = _genai.Client(api_key=api_key)
        last_err = None
        transient_fail = False
        used_model = None
        for model_id in models_to_try:
            try:
                resp = client.models.generate_content(
                    model=model_id,
                    contents=user_message,
                    config=_gtypes.GenerateContentConfig(
                        system_instruction=system_prompt,
                    ),
                )
                used_model = model_id
                with _model_lock:
                    _active_model = model_id
                return jsonify({"response": resp.text, "model": model_id,
                                "command": command, "command_result": command_result})
            except Exception as e:
                last_err = e
                err_s = str(e).lower()
                if any(x in err_s for x in _TRANSIENT):
                    transient_fail = True
                    continue
                break  # Non-transient error (auth, bad request) — stop immediately

        # All models failed — re-probe in background so next request gets fresh routing
        threading.Thread(target=_probe_models_bg, daemon=True).start()

        if transient_fail:
            ai_msg = (f"⚠ **Gemini API Error:** All models quota-limited or unavailable.\n\n"
                      f"Last error: `{str(last_err)[:200]}`\n\n"
                      "*(Free-tier daily limits may be exhausted. Try again in a few minutes or tomorrow.)*")
        else:
            ai_msg = f"Gemini API Error: {str(last_err)}"

        return jsonify({"response": ai_msg, "model": None,
                        "command": command, "command_result": command_result})
    except Exception as e:
        import traceback
        app.logger.error(f"Chat API critical error: {traceback.format_exc()}")
        return jsonify({
            "response": f"Dashboard Internal Error: Could not process request. {str(e)}",
            "model": None,
            "command": None,
            "command_result": None
        })


@app.route('/api/ai_status')
@require_api_key
def ai_status():
    with _model_lock:
        model = _active_model
        
    ctrl = read_json('data/control.json', default={})
    selected = ctrl.get('selected_ai_model')
    display = model or selected
    return jsonify({
        'active_model': model,
        'model': display,          # alias used by pollAiStatus() in the frontend
        'selected_model': selected,
        'available_models': _AI_MODELS_CONFIG,
        'available': display is not None
    })


_WATCHLIST_FILE = 'data/watchlist.json'
_DEFAULT_WATCHLIST = ['BTC/USDT', 'SOL/USDT', 'ADA/USDT', 'ETH/USDT']

_TOP20_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'TRX/USDT', 'AVAX/USDT', 'SHIB/USDT',
    'DOT/USDT', 'LINK/USDT', 'NEAR/USDT', 'UNI/USDT', 'LTC/USDT',
    'APT/USDT', 'ATOM/USDT', 'HBAR/USDT', 'ICP/USDT', 'SUI/USDT',
]


@app.route('/api/watchlist', methods=['GET'])
@require_api_key
def get_watchlist():
    symbols = read_json(_WATCHLIST_FILE, default=_DEFAULT_WATCHLIST)
    return jsonify({'symbols': symbols})


@app.route('/api/watchlist/add', methods=['POST'])
@require_api_key
def add_watchlist():
    import logging as _log
    symbol = (request.json or {}).get('symbol', '').upper().strip()
    if '/' not in symbol or len(symbol) < 5:
        return jsonify({'error': 'Invalid symbol — use format BTC/USDT'}), 400
    symbols = read_json(_WATCHLIST_FILE, default=_DEFAULT_WATCHLIST)
    if symbol not in symbols:
        symbols.append(symbol)
        write_json(_WATCHLIST_FILE, symbols)

        def _bg_download():
            try:
                from src.tools.binance_archive_downloader import bulk_download_for_symbol as archive_dl
                from src.data_ingestion.binance_downloader import download_history
                # Full 1h and 1d history from archive (resumes from last downloaded month)
                archive_dl(symbol, '1h', start_year=2017)
                archive_dl(symbol, '1d', start_year=2017)
                # 1m limited to recent 2 years to avoid massive downloads
                archive_dl(symbol, '1m', start_year=2023)
                # Patch latest candles via REST API
                download_history(symbol=symbol, timeframe='1h', limit=1000)
                download_history(symbol=symbol, timeframe='1m', limit=1000)
            except Exception as exc:
                _log.getLogger(__name__).error(f'Watchlist archive download {symbol}: {exc}')

        threading.Thread(target=_bg_download, daemon=True).start()
    return jsonify({'symbols': symbols, 'added': symbol})


@app.route('/api/watchlist/top20', methods=['GET'])
@require_api_key
def get_top20():
    return jsonify({'symbols': _TOP20_SYMBOLS})


@app.route('/api/models', methods=['GET'])
@require_api_key
def get_models():
    """Return accuracy metadata for all 4 ML models."""
    _MODEL_FILES = {
        'spot':     'models/btc_rf_model_meta.json',
        'scalping': 'models/scalping_model_meta.json',
        'futures':  'models/futures_short_model_meta.json',
        'trend':    'models/trend_model_meta.json',
        'tft':      'models/tft_model_meta.json',
    }
    result = {}
    for key, path in _MODEL_FILES.items():
        result[key] = read_json(path, default={})
    return jsonify(result)


@app.route('/api/watchlist/remove', methods=['POST'])
@require_api_key
def remove_watchlist():
    symbol = (request.json or {}).get('symbol', '').upper().strip()
    symbols = read_json(_WATCHLIST_FILE, default=_DEFAULT_WATCHLIST)
    symbols = [s for s in symbols if s != symbol]
    write_json(_WATCHLIST_FILE, symbols)
    return jsonify({'symbols': symbols})


@app.route('/api/close_all', methods=['POST'])
@require_api_key
def close_all_trades():
    """Close every open position at current price and trigger ML retraining."""
    try:
        from src.engine.trade_tracker import TradeTracker
        tracker = TradeTracker()
        open_trades = tracker.get_open_trades()
        closed_count = 0
        total_pnl = 0.0
        for trade in list(open_trades):
            sell_price = trade.get('current_price') or trade.get('buy_price', 0)
            if sell_price:
                result = tracker.close_trade_by_id(trade['id'], float(sell_price))
                if result:
                    closed_count += 1
                    total_pnl += result.get('pnl_usdt', 0) or 0

        def _retrain():
            try:
                subprocess.run(
                    [sys.executable, 'src/engine/train_all_models.py'],
                    capture_output=True, timeout=600
                )
            except Exception:
                pass

        threading.Thread(target=_retrain, daemon=True).start()
        return jsonify({'success': True, 'closed': closed_count, 'total_pnl': round(total_pnl, 4)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/close_losing', methods=['POST'])
@require_api_key
def close_losing_trades():
    """Close every open position currently in loss (current_price < buy_price)."""
    try:
        from src.engine.trade_tracker import TradeTracker
        tracker = TradeTracker()
        closed_count = 0
        total_pnl = 0.0
        for trade in list(tracker.get_open_trades()):
            bp = float(trade.get('buy_price') or 0)
            cp = float(trade.get('current_price') or bp)
            if bp and cp < bp:
                result = tracker.close_trade_by_id(trade['id'], cp)
                if result:
                    closed_count += 1
                    total_pnl += result.get('pnl_usdt', 0) or 0
        return jsonify({'success': True, 'closed': closed_count, 'total_pnl': round(total_pnl, 4)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─── Monitor: process registry ───────────────────────────────────────────────
import time as _time
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
_LOG_DIR      = _PROJECT_ROOT / 'logs'
_PID_FILE     = _PROJECT_ROOT / 'data' / 'process_ids.json'

_SERVICES = {
    'training':  {'label': 'ML Training',           'script': 'src/engine/train_all_models.py'},
    'download':  {'label': 'Data Downloader',       'script': 'src/data_ingestion/run_full_download.py'},
    'news':      {'label': 'News Scraper',           'script': 'src/data_ingestion/news_scraper.py'},
    'telegram':  {'label': 'Telegram Monitor',      'script': 'src/data_ingestion/telegram_scraper.py'},
    'watchlist': {'label': 'Watchlist Downloader',  'script': 'src/data_ingestion/watchlist_downloader.py'},
}
# Script fragment used to detect externally-launched processes by cmdline scan
_EXTERNAL_SCRIPTS = {k: v['script'].split('/')[-1] for k, v in _SERVICES.items()}
_LOG_MAP = {
    'bot': 'bot.log', 'dash': 'dashboard.log', 'monitor': 'monitor.log',
    **{k: f'{k}.log' for k in _SERVICES}
}

_managed: dict = {}
_managed_lock = threading.Lock()


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        import psutil
        p = psutil.Process(int(pid))
        return p.status() not in ('zombie', 'dead')
    except Exception:
        return False


def _proc_stats(pid) -> dict:
    try:
        import psutil
        p = psutil.Process(int(pid))
        return {
            'cpu': round(p.cpu_percent(interval=0.05), 1),
            'mem_mb': p.memory_info().rss // (1024 * 1024),
            'uptime_s': max(0, int(_time.time() - p.create_time())),
        }
    except Exception:
        return {'cpu': 0, 'mem_mb': 0, 'uptime_s': 0}


@app.route('/api/monitor/health')
def monitor_health():
    pids = read_json('data/process_ids.json', default={})
    out = {}

    # Externally launched components (PIDs saved by restart_all.ps1)
    for key, label in [('bot', 'Trading Bot'), ('dash', 'Dashboard')]:
        pid = pids.get(key)
        alive = _pid_alive(pid)
        entry = {'label': label, 'running': alive, 'pid': pid, 'managed': False}
        if alive:
            entry.update(_proc_stats(pid))
        out[key] = entry

    # Services managed by this dashboard (started via /api/monitor/start)
    # Also detect externally-launched processes by scanning cmdlines
    def _find_external_pid(script_name):
        try:
            import psutil
            for p in psutil.process_iter(['pid', 'cmdline']):
                cmd = ' '.join(p.info.get('cmdline') or [])
                if script_name in cmd:
                    return p.info['pid']
        except Exception:
            pass
        return None

    with _managed_lock:
        for svc_key, svc in _SERVICES.items():
            proc = _managed.get(svc_key)
            running = proc is not None and proc.poll() is None
            pid = proc.pid if running else None
            # Also check if script is running externally (e.g. via launch_training.ps1)
            if not running:
                ext_pid = _find_external_pid(_EXTERNAL_SCRIPTS.get(svc_key, ''))
                if ext_pid:
                    running, pid = True, ext_pid
            entry = {'label': svc['label'], 'running': running, 'pid': pid, 'managed': proc is not None and proc.poll() is None}
            if running:
                entry.update(_proc_stats(pid))
            out[svc_key] = entry

    return jsonify(out)


@app.route('/api/monitor/logs/<component>')
def monitor_logs(component):
    log_file = _LOG_MAP.get(component)
    if not log_file:
        return jsonify({'error': 'unknown component'}), 404
    path = _LOG_DIR / log_file
    if not path.exists():
        return jsonify({'lines': [], 'size': 0})
    try:
        # Read last 40 KB to avoid loading huge files
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 40960))
            chunk = f.read()
        # Handle UTF-16 LE (PowerShell Tee-Object default) and UTF-8
        if chunk[:2] == b'\xff\xfe':
            raw = chunk[2:].decode('utf-16-le', errors='replace').replace('\x00', '')
        else:
            raw = chunk.decode('utf-8', errors='replace')
        lines = raw.splitlines()
        if size > 40960 and lines:
            lines = lines[1:]   # first line may be partial
        return jsonify({'lines': lines[-300:], 'size': size})
    except Exception as e:
        return jsonify({'lines': [str(e)], 'size': 0})


@app.route('/api/monitor/start/<service>', methods=['POST'])
def monitor_start(service):
    svc = _SERVICES.get(service)
    if not svc:
        return jsonify({'error': 'unknown service'}), 404
    with _managed_lock:
        proc = _managed.get(service)
        if proc is not None and proc.poll() is None:
            return jsonify({'ok': False, 'msg': 'already running', 'pid': proc.pid})
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / f'{service}.log'
        log_fh = open(log_path, 'a', encoding='utf-8')
        new_proc = subprocess.Popen(
            [sys.executable, str(_PROJECT_ROOT / svc['script'])],
            stdout=log_fh, stderr=log_fh,
            cwd=str(_PROJECT_ROOT),
        )
        _managed[service] = new_proc
    return jsonify({'ok': True, 'pid': new_proc.pid})


@app.route('/api/monitor/stop/<service>', methods=['POST'])
def monitor_stop(service):
    if service not in _SERVICES:
        return jsonify({'error': 'unknown service'}), 404
    killed = False
    # Kill dashboard-managed instance
    with _managed_lock:
        proc = _managed.pop(service, None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        killed = True
    # Also kill any externally-launched instance (e.g. via launch_training.ps1)
    script_frag = _EXTERNAL_SCRIPTS.get(service, '')
    if script_frag:
        try:
            import psutil
            for p in psutil.process_iter(['pid', 'cmdline']):
                cmd = ' '.join(p.info.get('cmdline') or [])
                if script_frag in cmd:
                    try:
                        p.terminate()
                        killed = True
                    except Exception:
                        pass
        except Exception:
            pass
    return jsonify({'ok': killed, 'msg': 'stopped' if killed else 'not running'})


_AGENT_CONFIG = {
    'DataAgent':      {'label': 'Data Agent',      'desc': 'Data freshness monitor, retraining trigger',    'interval': 3600,  'market': 'core',     'color': '#6366f1'},
    'SignalAgent':    {'label': 'Signal Agent',    'desc': 'Regime-aware signal generator + meta-filter',   'interval': 3600,  'market': 'core',     'color': '#3b82f6'},
    'QuantAgent':     {'label': 'Quant Agent',     'desc': 'Rolling backtest + correlation shift detector',  'interval': 14400, 'market': 'core',     'color': '#8b5cf6'},
    'RiskAgent':      {'label': 'Risk Agent',      'desc': 'Kelly sizing, circuit breaker, liquidity guard', 'interval': 300,   'market': 'core',     'color': '#f59e0b'},
    'ExecutionAgent': {'label': 'Execution Agent', 'desc': 'Position management & order simulation',        'interval': 60,    'market': 'core',     'color': '#10b981'},
    'SpotAgent':      {'label': 'Spot Agent',      'desc': '1h spot — RANGING+TRENDING (conf≥0.62)',        'interval': 3600,  'market': 'spot',     'color': '#1d4ed8'},
    'FuturesAgent':   {'label': 'Futures Agent',   'desc': '1h futures — funding arb, 2× leverage',        'interval': 3600,  'market': 'futures',  'color': '#7c3aed'},
    'ScalpingAgent':  {'label': 'Scalping Agent',  'desc': '1m micro — OFI+VWAP, BTC/ETH/SOL (conf≥0.65)', 'interval': 60,   'market': 'scalping', 'color': '#059669'},
}


@app.route('/api/agents')
@require_api_key
def get_agents():
    import json as _json
    status_path = _PROJECT_ROOT / 'data' / 'agent_status.json'
    live: dict = {}
    if status_path.exists():
        try:
            live = _json.loads(status_path.read_text(encoding='utf-8'))
        except Exception:
            pass

    now_ts = _time.time()
    result = []
    for name, cfg in _AGENT_CONFIG.items():
        entry = live.get(name, {})
        last_hb_iso = entry.get('last_heartbeat', '')
        last_hb_ts: float | None = None
        if last_hb_iso:
            try:
                from datetime import datetime as _dt
                last_hb_ts = _dt.fromisoformat(last_hb_iso.replace('Z', '+00:00')).timestamp()
            except Exception:
                pass

        interval = float(entry.get('interval_sec', cfg['interval']))
        status   = entry.get('status', 'offline')
        if last_hb_ts and (now_ts - last_hb_ts) > interval * 3:
            status = 'stale'
        elif not last_hb_ts:
            status = 'offline'

        # Task timeline: real history + current
        timeline = []
        history = entry.get('history', [])
        for h in history[-5:]:  # last 5 past tasks
            timeline.append({'ts': h['ts'], 'type': 'completed', 'task': h['task']})
        if last_hb_ts:
            timeline.append({
                'ts': last_hb_ts,
                'type': 'current',
                'task': entry.get('current_task', 'Executing cycle'),
            })

        result.append({
            'name':         name,
            'label':        cfg['label'],
            'desc':         cfg['desc'],
            'market':       cfg['market'],
            'color':        cfg['color'],
            'interval_sec': interval,
            'status':       status,
            'current_task': entry.get('current_task', '—'),
            'last_heartbeat': last_hb_iso,
            'timeline':     timeline,
        })

    return jsonify({'agents': result, 'ts': now_ts})


@app.route('/api/monitor/model_stats')
def monitor_model_stats():
    models_dir = _PROJECT_ROOT / 'models'
    _MODEL_FILES = [
        ('base',     'btc_rf_model_meta.json',       'Base Model',        'btc_rf_model.joblib'),
        ('trend',    'trend_model_meta.json',         'Trend Following',   'trend_model.joblib'),
        ('futures',  'futures_short_model_meta.json', 'Futures Short',     'futures_short_model.joblib'),
        ('scalping', 'scalping_model_meta.json',      'Scalping (1m)',     'scalping_model.joblib'),
        ('tft',      'tft_model_meta.json',           'TFT (Neural)',      'tft_model.pt'),
        ('meta',     'meta_labeler_meta.json',        'Meta-Labeler',      'meta_labeler.joblib'),
        ('regime',   'regime_classifier_meta.json',   'Regime Classifier', 'regime_classifier.joblib'),
    ]
    result = []
    for key, meta_file, label, model_file in _MODEL_FILES:
        meta_path  = models_dir / meta_file
        model_path = models_dir / model_file
        exists = model_path.exists()
        meta = {}
        if meta_path.exists():
            try:
                import json as _json
                meta = _json.loads(meta_path.read_text())
            except Exception:
                pass
        raw_acc_m = meta.get('accuracy')
        result.append({
            'key': key, 'label': label,
            'model_exists': exists,
            'accuracy':              round(raw_acc_m, 2) if raw_acc_m is not None else None,
            'long_accuracy':         round(meta.get('long_accuracy', 0), 2),
            'short_accuracy':        round(meta.get('short_accuracy', 0), 2),
            'n_samples':             meta.get('n_samples'),
            'n_train':               meta.get('n_train'),
            'n_test':                meta.get('n_test'),
            'n_features':            meta.get('n_features'),
            'n_iterations':          meta.get('n_iterations'),
            'symbols':               meta.get('symbols', []),
            'timeframe':             meta.get('timeframe', '--'),
            'last_trained':          meta.get('last_trained', ''),
            'walk_forward_mean_acc': meta.get('walk_forward_mean_acc'),
            'target':                meta.get('target', ''),
        })

    # CUDA / GPU info
    cuda = {'available': False, 'device': 'CPU only', 'version': None}
    try:
        import torch
        if torch.cuda.is_available():
            cuda = {
                'available': True,
                'device': torch.cuda.get_device_name(0),
                'version': torch.version.cuda,
            }
    except Exception:
        pass

    return jsonify({'models': result, 'cuda': cuda})


@app.route('/api/strategy/full')
@require_api_key
def strategy_full():
    """
    Single endpoint for the Strategy/ML tab.
    Returns: ml_models (7 entries), strategies (from registry), trade_stats.
    """
    import json as _json

    # ── ML models ─────────────────────────────────────────────────────────────
    models_dir = _PROJECT_ROOT / 'models'
    _ML = [
        ('base',    'btc_rf_model_meta.json',       'Base RF (1h)',         'btc_rf_model.joblib',         '🧠', 'SPOT'),
        ('trend',   'trend_model_meta.json',         'Trend RF',            'trend_model.joblib',           '🌊', 'SPOT'),
        ('futures', 'futures_short_model_meta.json', 'Futures Short RF',    'futures_short_model.joblib',   '📉', 'FUTURES'),
        ('scalping','scalping_model_meta.json',      'Scalping RF (1m)',    'scalping_model.joblib',        '⚡', 'SCALPING'),
        ('tft',     'tft_model_meta.json',           'TFT Neural (1h)',     'tft_model.pt',                 '🔮', 'SPOT'),
        ('meta',    'meta_labeler_meta.json',        'Meta-Labeler',        'meta_labeler.joblib',          '🔍', 'ALL'),
        ('regime',  'regime_classifier_meta.json',   'Regime Classifier',   'regime_classifier.joblib',     '🎯', 'ALL'),
    ]
    ml_models = []
    for key, mf, label, model_file, icon, market in _ML:
        meta_path  = models_dir / mf
        model_path = models_dir / model_file
        meta = {}
        if meta_path.exists():
            try: meta = _json.loads(meta_path.read_text())
            except Exception: pass
        raw_acc = meta.get('accuracy')
        ml_models.append({
            'key': key, 'label': label, 'icon': icon, 'market': market,
            'model_exists':   model_path.exists(),
            'accuracy':       round(raw_acc, 2) if raw_acc is not None else None,
            'long_accuracy':  round(meta.get('long_accuracy', 0), 2),
            'short_accuracy': round(meta.get('short_accuracy', 0), 2),
            'accuracy_note':  meta.get('accuracy_note'),
            'model_type':     meta.get('model_type'),
            'n_samples':      meta.get('n_samples'),
            'n_features':     meta.get('n_features'),
            'n_iterations':   meta.get('n_iterations'),
            'symbols':        meta.get('symbols', []),
            'timeframe':      meta.get('timeframe', '--'),
            'last_trained':   meta.get('last_trained', ''),
            'target':         meta.get('target', ''),
        })

    # ── Strategy registry ──────────────────────────────────────────────────────
    try:
        from src.engine.strategy_registry import get_sync_report
        sync = get_sync_report()
        strategies = sync['strategies']
        summary    = sync['summary']
    except Exception as e:
        strategies = []
        summary    = {}

    # ── Trade stats per strategy ───────────────────────────────────────────────
    # Live trades (tagged with strategy field)
    trades = read_json('data/trades.json', default=[])
    if isinstance(trades, dict):
        trades = trades.get('trades', [])
    trade_stats: dict[str, dict] = {}
    for t in trades:
        if str(t.get('status', '')).upper() != 'CLOSED':
            continue
        k = t.get('strategy', 'Unknown')
        pnl = t.get('pnl_usdt', 0) or 0
        s = trade_stats.setdefault(k, {'n': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'source': 'live'})
        s['n'] += 1
        s['pnl'] += pnl
        if pnl > 0:
            s['wins'] += 1
        else:
            s['losses'] += 1
    for k, s in trade_stats.items():
        s['win_rate'] = round(s['wins'] / s['n'] * 100, 1) if s['n'] else 0.0

    # Backtest stats — aggregate latest_comparison.json, keyed by registry name
    import re as _re
    bt_path = _PROJECT_ROOT / 'data' / 'backtest' / 'latest_comparison.json'
    if bt_path.exists():
        try:
            import json as _j
            bt_rows = _j.loads(bt_path.read_text())
            # Build label → registry_name reverse map from the strategy list
            label_to_key: dict[str, str] = {}
            for s in strategies:
                lbl = s.get('label', '')
                key = s.get('name', '')
                if lbl and key:
                    label_to_key[lbl] = key
            bt_agg: dict[str, dict] = {}
            for row in bt_rows:
                raw = row.get('strategy', '')
                bt_label = _re.sub(r'^[AB]_', '', raw).strip()
                # Resolve to registry key; fall back to the label itself
                reg_key = label_to_key.get(bt_label, bt_label)
                a = bt_agg.setdefault(reg_key, {'n': 0, 'wins': 0, 'pnl': 0.0,
                                                 'sharpe_sum': 0.0, 'sharpe_cnt': 0,
                                                 'win_rate_sum': 0.0, 'win_rate_cnt': 0})
                n = int(row.get('n_trades', 0))
                wr = float(row.get('win_rate_pct', 0))
                a['n']            += n
                a['wins']         += round(n * wr / 100)
                a['pnl']          += float(row.get('total_pnl_usdt', 0))
                sh = row.get('sharpe')
                if sh is not None:
                    a['sharpe_sum'] += float(sh)
                    a['sharpe_cnt'] += 1
                a['win_rate_sum'] += wr
                a['win_rate_cnt'] += 1
            for reg_key, a in bt_agg.items():
                if reg_key not in trade_stats:   # don't overwrite live data
                    trade_stats[reg_key] = {
                        'n':        a['n'],
                        'wins':     a['wins'],
                        'losses':   a['n'] - a['wins'],
                        'pnl':      round(a['pnl'], 2),
                        'win_rate': round(a['win_rate_sum'] / a['win_rate_cnt'], 1) if a['win_rate_cnt'] else 0.0,
                        'sharpe':   round(a['sharpe_sum'] / a['sharpe_cnt'], 3) if a['sharpe_cnt'] else None,
                        'source':   'backtest',
                    }
        except Exception:
            pass

    # ── Aggregate ──────────────────────────────────────────────────────────────
    trained_count = sum(1 for m in ml_models if m['model_exists'])
    live_count    = sum(1 for s in strategies if s.get('live_enabled'))

    return jsonify({
        'ml_models':   ml_models,
        'strategies':  strategies,
        'trade_stats': trade_stats,
        'summary':     summary,
        'aggregate': {
            'models_trained': trained_count,
            'models_total':   len(ml_models),
            'strategies_live': live_count,
            'strategies_total': len(strategies),
        },
    })


@app.route('/api/strategy-sync', methods=['GET'])
def strategy_sync_get():
    """Return full strategy registry with sync status and enabled flags."""
    try:
        from src.engine.strategy_registry import get_sync_report
        return jsonify(get_sync_report())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/strategy-sync', methods=['POST'])
def strategy_sync_post():
    """
    Toggle a strategy's live/backtest flag.
    Body: {"name": "RSI_MeanReversion", "live": true, "backtest": false}
    Or bulk: {"strategies": [{"name": ..., "live": ..., "backtest": ...}, ...]}
    """
    try:
        from src.engine.strategy_registry import update_strategy, get_sync_report
        data = request.get_json(force=True) or {}

        updates = data.get('strategies', [data] if 'name' in data else [])
        results = []
        for upd in updates:
            name = upd.get('name')
            if not name:
                continue
            entry = update_strategy(
                name,
                live     = upd['live']     if 'live'     in upd else None,
                backtest = upd['backtest'] if 'backtest' in upd else None,
            )
            results.append({'name': name, **entry})

        return jsonify({'updated': results, **get_sync_report()})
    except KeyError as e:
        return jsonify({'error': f'Unknown strategy: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Quant matrix (independent of live bot) ──────────────────────────────────
import time as _time_mod
_qm_cache: dict = {}
_qm_cache_ts: float = 0.0
_QM_TTL = 300  # 5-minute cache


@app.route('/api/quant_matrix')
@require_api_key
def quant_matrix():
    """
    Compute OU deviation, GARCH vol, and RF signal for all watchlist pairs
    directly from GZ files — works without main.py running.
    Results are cached for 5 minutes.
    """
    global _qm_cache, _qm_cache_ts
    now = _time_mod.time()
    if _qm_cache and (now - _qm_cache_ts) < _QM_TTL:
        return jsonify(_qm_cache)

    symbols = read_json(_WATCHLIST_FILE, default=_DEFAULT_WATCHLIST)
    raw_dir = _PROJECT_ROOT / 'data' / 'raw'
    result: dict = {}

    try:
        from src.analysis.ml_predictor import MLPredictor
        from src.analysis.feature_engineering import add_rsi, add_macd
        _ml = MLPredictor('btc_rf_model.joblib', 'base')
    except Exception:
        _ml = None

    for sym in symbols:
        key = sym.replace('/', '_').replace('-', '_')
        gz = raw_dir / f'{key}_1h.csv.gz'
        if not gz.exists():
            gz = raw_dir / f'{key.replace("_USDT", "")}USDT_1h.csv.gz'
        if not gz.exists():
            result[sym] = {'signal': 'NO_DATA', 'ou_dev': 0.0,
                           'garch_vol': 0.0, 'ml_return': None, 'as_spread': None}
            continue
        try:
            df = _read_last_n_bars(gz, 200)
            if df is None or len(df) < 50:
                continue

            # OU deviation (200-bar window, OLS fit)
            prices = df['close'].values
            ou_dev = _calc_ou_dev(prices)

            # GARCH volatility proxy (realized vol ratio)
            rets = df['close'].pct_change().dropna()
            vol5  = float(rets.tail(5).std() * 100) if len(rets) >= 5 else 0.0
            vol60 = float(rets.tail(60).std() * 100) if len(rets) >= 60 else 0.0
            garch_flag = 'HIGH' if vol5 > vol60 * 1.8 else 'NORMAL'
            garch_vol  = round(vol60, 3)

            # Base RF signal → expected return proxy
            ml_return = None
            if _ml and _ml.is_loaded:
                try:
                    p = _ml.predict_proba_long(df.tail(60).to_dict('records'))
                    ml_return = round((p - 0.5) * 4, 2)  # map [0,1] → [-2%,+2%]
                except Exception:
                    pass

            # Overall signal
            if ou_dev <= -2.0 and ml_return is not None and ml_return > 0:
                sig = 'OVERSOLD'
            elif ou_dev >= 2.0 and ml_return is not None and ml_return < 0:
                sig = 'OVERBOUGHT'
            elif ou_dev <= -2.5 or (ml_return is not None and ml_return < -1.0):
                sig = 'OVERSOLD'
            elif ou_dev >= 2.5 or (ml_return is not None and ml_return > 1.0):
                sig = 'OVERBOUGHT'
            else:
                sig = 'NEUTRAL'

            result[sym] = {
                'signal':    sig,
                'ou_dev':    round(ou_dev, 2),
                'ou_mu':     len(df),
                'garch_vol': garch_vol,
                'garch_flag': garch_flag,
                'ml_return': ml_return,
                'as_spread': None,
            }
        except Exception as exc:
            logger.warning('quant_matrix %s: %s', sym, exc)
            result[sym] = {'signal': 'ERROR', 'ou_dev': 0.0,
                           'garch_vol': 0.0, 'ml_return': None, 'as_spread': None}

    _qm_cache = {'quant': result, 'computed_at': _time_mod.time()}
    _qm_cache_ts = now
    return jsonify(_qm_cache)


def _read_last_n_bars(gz_path, n: int = 200):
    """Read last N rows from a gzipped CSV without loading the full file."""
    try:
        import pandas as pd
        chunks = []
        reader = pd.read_csv(gz_path, compression='gzip', chunksize=50_000,
                             index_col=0, parse_dates=True)
        for chunk in reader:
            chunks.append(chunk)
        if not chunks:
            return None
        df = pd.concat(chunks).tail(n)
        df.columns = [c.lower() for c in df.columns]
        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['close'])
    except Exception:
        return None


def _calc_ou_dev(prices) -> float:
    """OU deviation in sigma units via OLS: X_t+1 = a + b*X_t."""
    import numpy as np
    if len(prices) < 10:
        return 0.0
    x = prices[:-1]
    y = prices[1:]
    try:
        b = np.cov(x, y)[0, 1] / np.var(x)
        a = np.mean(y) - b * np.mean(x)
        mu = a / (1 - b) if abs(1 - b) > 1e-9 else np.mean(prices)
        resid = y - (a + b * x)
        sigma = float(resid.std())
        return float((prices[-1] - mu) / sigma) if sigma > 0 else 0.0
    except Exception:
        return 0.0


# ─── Simulator agent singletons (lazy-started on first /start call) ───────────
_simulator_agent = None
_trainer_agent   = None
_sim_lock        = threading.Lock()


def _get_simulator():
    global _simulator_agent, _trainer_agent
    with _sim_lock:
        if _simulator_agent is None:
            from src.engine.agents.simulator_agent import SimulatorAgent
            from src.engine.agents.training_agent  import ContinuousTrainerAgent
            _simulator_agent = SimulatorAgent(auto_cycle=True)
            _trainer_agent   = ContinuousTrainerAgent()
    return _simulator_agent, _trainer_agent


@app.route('/api/simulator/status', methods=['GET'])
def simulator_status():
    """Return current simulator state, config, and per-model training metrics."""
    try:
        sim, trainer = _get_simulator()
        status = sim.get_status()
        status['trainer_stats'] = trainer.get_stats()

        # Augment with DB summary if available
        try:
            from src.simulation.data_store import SimulatorDataStore
            store = SimulatorDataStore()
            status['db_summary'] = store.get_summary()
        except Exception:
            status['db_summary'] = {}

        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/start', methods=['POST'])
def simulator_start():
    """Start or resume the simulator replay."""
    try:
        sim, trainer = _get_simulator()
        # Apply any config from the request body
        cfg = request.get_json(force=True) or {}
        if cfg:
            sim.configure(cfg)
        # Configure trainer models from request
        train_models = cfg.pop('train_models', None)
        if train_models and isinstance(train_models, list):
            trainer.configure_models(train_models)
        # Start trainer (idempotent)
        if not trainer._running:
            trainer.start()
        sim.start()
        return jsonify({'ok': True, 'status': sim.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/pause', methods=['POST'])
def simulator_pause():
    try:
        sim, _ = _get_simulator()
        sim.pause()
        return jsonify({'ok': True, 'status': sim.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/resume', methods=['POST'])
def simulator_resume():
    try:
        sim, _ = _get_simulator()
        sim.resume()
        return jsonify({'ok': True, 'status': sim.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/stop', methods=['POST'])
def simulator_stop():
    try:
        sim, trainer = _get_simulator()
        sim.stop()
        trainer.stop()
        return jsonify({'ok': True, 'status': sim.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/config', methods=['POST'])
def simulator_config():
    """Update simulator config (symbol, timeframe, speed, scenario, date range)."""
    try:
        sim, _ = _get_simulator()
        cfg = request.get_json(force=True) or {}
        allowed = {'symbol', 'timeframe', 'speed', 'scenario', 'start_date', 'end_date'}
        clean = {k: v for k, v in cfg.items() if k in allowed}
        sim.configure(clean)
        return jsonify({'ok': True, 'config': sim.get_status()['config']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/training_history', methods=['GET'])
def simulator_training_history():
    """Return recent training events and cumulative paper P&L series."""
    try:
        from src.simulation.data_store import SimulatorDataStore
        store  = SimulatorDataStore()
        model  = request.args.get('model')
        limit  = int(request.args.get('limit', 200))
        events = store.get_recent_training_events(model_name=model, limit=limit)
        pnl    = store.get_paper_pnl_series(limit=500)
        return jsonify({'events': events, 'pnl_series': pnl})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/patterns', methods=['GET'])
def simulator_patterns():
    """Return top patterns from the training pattern DB."""
    try:
        from src.simulation.data_store import SimulatorDataStore
        store = SimulatorDataStore()
        model = request.args.get('model')
        limit = int(request.args.get('limit', 50))
        patterns = store.get_pattern_db(model_name=model, limit=limit)
        return jsonify({'patterns': patterns, 'total': len(patterns)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/available_data', methods=['GET'])
def simulator_available_data():
    """List all GZ files available for replay."""
    try:
        raw_dir = os.path.join(project_root, 'data', 'raw')
        files = []
        if os.path.isdir(raw_dir):
            for f in sorted(os.listdir(raw_dir)):
                if f.endswith('.csv.gz') and '_funding' not in f:
                    parts = f.replace('.csv.gz', '').rsplit('_', 1)
                    if len(parts) == 2:
                        symbol, tf = parts
                        size_mb = round(os.path.getsize(os.path.join(raw_dir, f)) / 1e6, 1)
                        files.append({'symbol': symbol, 'timeframe': tf,
                                      'file': f, 'size_mb': size_mb})
        return jsonify({'files': files, 'total': len(files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
