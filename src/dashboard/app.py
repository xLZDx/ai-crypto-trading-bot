import os
import sys
import threading
import subprocess
import re
import time
import uuid
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
# Pick up template edits without a full dashboard restart. Default Flask
# only auto-reloads templates when debug=True; we run in production-like
# mode (no debug) so without this flag the Jinja env caches the parsed
# index.html for the dashboard process's lifetime, and operator-facing
# UI fixes don't show up until the watchdog respawns. Cheap to enable
# (mtime check per render). v3.1 fix 2026-05-09.
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# ─── Gemini model health cache ────────────────────────────────────────────────
# Paid / most capable models first; free-tier as fallback when paid unavailable.
_AI_MODELS_CONFIG = [
    {"id": "gemini-3.1-pro-preview",         "name": "Gemini 3.1 Pro Preview",        "cost": "Paid",      "thinking": "HIGH"},
    {"id": "gemini-3-pro-preview",           "name": "Gemini 3 Pro Preview",          "cost": "Paid",      "thinking": "HIGH"},
    {"id": "gemini-2.5-pro",                 "name": "Gemini 2.5 Pro",                "cost": "Paid",      "thinking": "HIGH"},
    {"id": "gemini-3.1-flash-lite-preview",  "name": "Gemini 3.1 Flash Lite Preview", "cost": "Free Tier", "thinking": "HIGH"},
    {"id": "gemini-3-flash-preview",         "name": "Gemini 3 Flash Preview",        "cost": "Free Tier", "thinking": "HIGH"},
    {"id": "gemini-2.5-flash",               "name": "Gemini 2.5 Flash",              "cost": "Free Tier", "thinking": "HIGH"},
    {"id": "gemini-2.5-flash-lite",          "name": "Gemini 2.5 Flash Lite",         "cost": "Free Tier", "thinking": "HIGH"},
    {"id": "gemini-2.0-flash",               "name": "Gemini 2.0 Flash",              "cost": "Free Tier", "thinking": "MED"},
    {"id": "gemini-2.0-flash-001",           "name": "Gemini 2.0 Flash 001",          "cost": "Free Tier", "thinking": "MED"},
    {"id": "gemini-2.0-flash-lite",          "name": "Gemini 2.0 Flash Lite",         "cost": "Free Tier", "thinking": "MED"},
    {"id": "gemini-2.0-flash-lite-001",      "name": "Gemini 2.0 Flash Lite 001",     "cost": "Free Tier", "thinking": "MED"},
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
    """Render the main dashboard. Forces no-store cache headers so the
    operator's browser always fetches the freshly-edited template — the
    pre-fix UX bug was that browser cache kept stale onclick handlers
    after restart_all.ps1 reloaded the bot, making refresh buttons look
    broken when really the JS was just from a prior dashboard process."""
    from flask import make_response
    resp = make_response(render_template('index.html', api_key=DASHBOARD_API_KEY or ''))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma']        = 'no-cache'
    resp.headers['Expires']       = '0'
    return resp


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


@app.route('/api/system/restart_all', methods=['POST'])
@require_api_key
def api_system_restart_all():
    """Operator-triggered full-stack restart. Spawns restart_all.ps1 in
    a detached PowerShell window so it survives this dashboard process
    being killed (which it WILL be — the script kills/respawns the
    dashboard). The browser will see the connection drop for ~30s
    until the new dashboard comes up; the JS auto-reconnects on its
    next poll."""
    import subprocess as _sp, os as _os
    script = os.path.join(project_root, 'restart_all.ps1')
    if not os.path.exists(script):
        return jsonify({'ok': False,
                        'error': f'restart_all.ps1 not found at {script}'}), 500
    try:
        # Detached on Windows so we survive being killed mid-execution.
        creationflags = 0
        if _os.name == 'nt':
            creationflags = (_sp.CREATE_NEW_PROCESS_GROUP |
                             getattr(_sp, 'DETACHED_PROCESS', 0x00000008))
        log_path = os.path.join(project_root, 'logs', f'restart_all_{int(time.time())}.log')
        log_fp = open(log_path, 'a', encoding='utf-8')
        proc = _sp.Popen(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-File', script],
            cwd=project_root, stdout=log_fp, stderr=log_fp,
            creationflags=creationflags, close_fds=True,
        )
        return jsonify({'ok': True, 'pid': proc.pid,
                        'log_path': log_path,
                        'message': 'restart_all spawned; dashboard will go offline ~10-30s while it relaunches'})
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), 500


@app.route('/api/control', methods=['POST'])
@require_api_key
def set_control():
    """Merge incoming fields into control.json — never overwrite the whole
    file. Older callers POSTed `{"running": false}` here and silently wiped
    `trade_mode` (and any other field added later). Merge keeps every
    previously-set field and only touches what the caller specified.
    """
    try:
        data = request.json
        if not isinstance(data, dict):
            return jsonify({"success": False,
                            "error": "expected JSON object body"}), 400
        existing = read_json('data/control.json', default={}) or {}
        if not isinstance(existing, dict):
            existing = {}
        existing.update(data)
        write_json('data/control.json', existing)
        return jsonify({"success": True, "control": existing})
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
    'training':     {'label': 'ML Training',              'script': 'src/engine/train_all_models.py'},
    'download':     {'label': 'Data Downloader',          'script': 'src/data_ingestion/run_full_download.py'},
    'news':         {'label': 'News Scraper',              'script': 'src/data_ingestion/news_scraper.py'},
    'telegram':     {'label': 'Telegram Monitor',         'script': 'src/data_ingestion/telegram_scraper.py'},
    'watchlist':    {'label': 'Watchlist Downloader',     'script': 'src/data_ingestion/watchlist_downloader.py'},
    'historical_dl':{'label': 'Historical Archive (pre-2026)', 'script': 'src/data_ingestion/binance_archive_downloader.py'},
}
# Script fragment used to detect externally-launched processes by cmdline scan
_EXTERNAL_SCRIPTS = {k: v['script'].split('/')[-1] for k, v in _SERVICES.items()}
_LOG_MAP = {
    'bot': 'bot.log', 'dash': 'dashboard.log', 'monitor': 'monitor.log',
    **{k: f'{k}.log' for k in _SERVICES}
}

_managed: dict = {}
_managed_lock = threading.Lock()


def _find_external_pid(script_name: str):
    """Scan all running processes for one whose cmdline contains script_name."""
    if not script_name:
        return None
    try:
        import psutil
        for p in psutil.process_iter(['pid', 'cmdline']):
            cmd = ' '.join(p.info.get('cmdline') or [])
            if script_name in cmd:
                return p.info['pid']
    except Exception:
        pass
    return None


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

    # Externally launched components (PIDs saved by restart_all.ps1).
    # If the recorded PID is dead, fall back to a cmdline scan so a bot
    # or dashboard that was relaunched directly (e.g. via Start-Process,
    # bypassing restart_all) still shows as 'Running'. Pre-fix:
    # process_ids.json's PIDs went stale after any direct relaunch and
    # the Component Health card kept showing 'Stopped' for things that
    # were obviously alive (the user was looking at the dashboard!).
    _CMDLINE_SCAN = {
        'bot':  r'src[\\/]main\.py',
        'dash': r'src[\\/]dashboard[\\/]app\.py',
    }
    for key, label in [('bot', 'Trading Bot'), ('dash', 'Dashboard')]:
        pid = pids.get(key)
        alive = _pid_alive(pid)
        if not alive and key in _CMDLINE_SCAN:
            try:
                import psutil, re as _re
                pat = _re.compile(_CMDLINE_SCAN[key])
                for p in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if (p.info.get('name') or '').lower().startswith('python'):
                            cmd = ' '.join(p.info.get('cmdline') or [])
                            if pat.search(cmd):
                                pid = p.info['pid']; alive = True
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception:
                pass
        entry = {'label': label, 'running': alive, 'pid': pid, 'managed': False}
        if alive:
            entry.update(_proc_stats(pid))
        out[key] = entry

    # Services managed by this dashboard (started via /api/monitor/start)
    # Also detect externally-launched processes by scanning cmdlines
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

    # Telegram is embedded in the bot process — detect via heartbeat file.
    # Without this, the card always shows "Stopped" because no standalone
    # telegram_scraper.py process exists.
    try:
        import json as _json
        tg_path = _PROJECT_ROOT / 'data' / 'telegram_status.json'
        if tg_path.exists():
            tg = _json.loads(tg_path.read_text(encoding='utf-8'))
            fresh = (_time.time() - float(tg.get('last_update_ts', 0))) < 600
            if fresh and tg.get('connected'):
                channels = tg.get('channels', [])
                detail = f"embedded in bot · {len(channels)} ch" if channels else 'embedded in bot'
                out['telegram'] = {
                    'label': 'Telegram Monitor', 'running': True,
                    'pid': out.get('bot', {}).get('pid'),
                    'managed': False, 'embedded': True, 'detail': detail,
                }
    except Exception:
        pass

    return jsonify(out)


# TTL cache for /api/monitor/services. Pre-fix: every poll re-ran 7
# probes (rglob 7K parquet files + stat-sum 50 GB + DuckDB connect +
# psutil + HTTP). With dashboard polling every few seconds × 138 hung
# threads, this contributed to the lock-storm that wedged the dashboard
# on 2026-05-08. Cache for 30 s — service health doesn't change faster
# than that anyway, and the operator's eye can't refresh a card faster.
_monitor_services_cache: dict = {'value': None, 'ts': 0.0}
_monitor_services_cache_ttl = 30.0
_monitor_services_lock = threading.Lock()


def _build_monitor_services_snapshot() -> dict:
    """Run every service probe and return the {key: status} dict.
    Pure read; no shared state mutation. Called by monitor_services()
    behind a TTL gate so concurrent dashboard polls don't fan-out."""
    import socket
    import urllib.request
    import urllib.error

    out: dict[str, dict] = {}

    # ── ParquetClient store (replaces QuestDB) ─────────────────────────────
    # File-based, no daemon. Healthy iff DuckDB imports + data/db is writable.
    try:
        from src.database.parquet_client import get_client as _get_pq
        pq = _get_pq()
        up = pq.is_available(force=True)
        # Count tables that have at least one parquet file (rough freshness signal).
        try:
            from src.database.parquet_client import _TABLES as _PQ_TABLES
            populated = sum(1 for t in _PQ_TABLES if pq._has_any_files(t))
        except Exception:
            populated = 0
        out['parquet_store'] = {
            'label': 'Parquet Store (DuckDB)',
            'up': up,
            'detail': f'{pq.base_dir.relative_to(_PROJECT_ROOT)} · '
                      f'{populated} populated tables · in-process query',
            'hint': None if up else 'install duckdb / ensure D:/data/db is writable',
        }
    except Exception as e:
        out['parquet_store'] = {
            'label': 'Parquet Store (DuckDB)', 'up': False,
            'error': type(e).__name__,
            'hint': 'pip install duckdb pyarrow clickhouse-connect',
        }

    # ── DuckDB (in-process; check both library + temp dir writable) ─────────
    try:
        import duckdb  # noqa: F401
        tmp = _PROJECT_ROOT / 'data' / 'cache' / 'duckdb_temp'
        tmp.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(':memory:')
        try:
            con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
            ver = con.execute('SELECT version()').fetchone()[0]
        finally:
            con.close()
        out['duckdb'] = {
            'label': 'DuckDB (cold path / parquet)', 'up': True,
            'detail': f'in-process · v{ver} · temp → data/cache/duckdb_temp',
        }
    except Exception as e:
        out['duckdb'] = {
            'label': 'DuckDB (cold path / parquet)', 'up': False,
            'error': f'{type(e).__name__}: {e}',
            'hint': 'pip install duckdb>=0.10.0',
        }

    # ── Parquet store (count partitions, sum size) ─────────────────────────
    try:
        pq_root = _PROJECT_ROOT / 'data' / 'parquet'
        if pq_root.exists():
            files = list(pq_root.rglob('*.parquet'))
            size_gb = sum(f.stat().st_size for f in files) / 1e9
            symbols = len({p.parts[len(pq_root.parts)] for p in files if len(p.parts) > len(pq_root.parts)})
            out['parquet'] = {
                'label': 'Parquet Store',
                'up': len(files) > 0,
                'detail': f'{len(files):,} files · {symbols} symbols · {size_gb:.2f} GB',
            }
        else:
            out['parquet'] = {'label': 'Parquet Store', 'up': False,
                              'error': 'data/parquet missing',
                              'hint': 'run scripts/migrate_news_to_parquet.py / archive downloader'}
    except Exception as e:
        out['parquet'] = {'label': 'Parquet Store', 'up': False, 'error': str(e)}

    # ── Simulator (read /api/simulator/status state) ───────────────────────
    try:
        sim_status_path = _PROJECT_ROOT / 'data' / 'sim_state.json'
        if sim_status_path.exists():
            import json as _j
            st = _j.loads(sim_status_path.read_text())
            running = bool(st.get('running'))
            out['simulator'] = {
                'label': 'Synthetic Exchange / Simulator',
                'up': running,
                'detail': f"state: {st.get('state','idle')} · scenario: {st.get('scenario','--')}",
            }
        else:
            out['simulator'] = {
                'label': 'Synthetic Exchange / Simulator', 'up': False,
                'error': 'idle (no run started)',
                'hint': 'open Simulator tab → ▶ Start',
            }
    except Exception as e:
        out['simulator'] = {'label': 'Synthetic Exchange / Simulator', 'up': False, 'error': str(e)}

    # ── ZeroMQ control plane (probe :5555 PUB) ─────────────────────────────
    # The data-bus binds 5555 lazily on the first publish_orderflow() call —
    # so port-closed isn't a fault. Only orderbook_collector / distributed
    # training PUBLISH; the standard bot loop doesn't, so the port stays
    # closed by design unless the user enables L2 streaming.
    def _tcp_open(host, port, timeout=0.5):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False
    zmq_up = _tcp_open('127.0.0.1', 5555)
    out['zmq'] = {
        'label': 'ZeroMQ Data Plane',
        'up': zmq_up,
        'detail': ('tcp://127.0.0.1:5555 · bound, streaming'
                   if zmq_up else 'tcp://127.0.0.1:5555 · idle (binds on first orderflow publish)'),
        'hint': None if zmq_up else 'enable orderbook_collector / distributed training to bind',
    }

    # ── FastAPI control plane (:8100) ──────────────────────────────────────
    try:
        with urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=0.5) as resp:
            out['fastapi'] = {'label': 'FastAPI Control Plane', 'up': resp.status == 200,
                              'detail': 'localhost:8100'}
    except Exception:
        out['fastapi'] = {'label': 'FastAPI Control Plane', 'up': False,
                          'error': 'unreachable', 'detail': 'localhost:8100'}

    # ── Realtime feed (Binance L2) — check status JSON ─────────────────────
    try:
        rt_path = _PROJECT_ROOT / 'data' / 'realtime_status.json'
        if rt_path.exists():
            import json as _j
            st = _j.loads(rt_path.read_text())
            up = bool(st.get('connected'))
            last = st.get('last_msg_iso', '')
            sym  = st.get('symbol', '--')
            out['realtime'] = {
                'label': 'Realtime Feed (Binance L2)', 'up': up,
                'detail': f'sym: {sym} · last: {last}',
            }
        else:
            out['realtime'] = {'label': 'Realtime Feed (Binance L2)', 'up': False,
                               'error': 'no status file',
                               'hint': 'started by orderbook_realtime.py'}
    except Exception as e:
        out['realtime'] = {'label': 'Realtime Feed (Binance L2)', 'up': False, 'error': str(e)}

    return out


def _refresh_monitor_services_async():
    """Background-thread refresh of the monitor-services snapshot. Same
    pattern as _refresh_db_status_async — keeps the cache warm without
    blocking the route on a slow probe pass."""
    def _job():
        try:
            value = _build_monitor_services_snapshot()
            with _monitor_services_lock:
                _monitor_services_cache['value'] = value
                _monitor_services_cache['ts'] = time.time()
        except Exception as exc:
            with _monitor_services_lock:
                _monitor_services_cache['value'] = {'_error': str(exc)}
                _monitor_services_cache['ts'] = time.time()
    threading.Thread(target=_job, daemon=True, name='monitor-services-refresh').start()


@app.route('/api/monitor/services')
def monitor_services():
    """ParquetClient store / DuckDB / Simulator / ZMQ / FastAPI / Realtime
    feed status. TTL-cached (30 s) — see _monitor_services_cache_ttl."""
    with _monitor_services_lock:
        cached = _monitor_services_cache.get('value')
        cache_age = time.time() - (_monitor_services_cache.get('ts') or 0)
    if cached is None or cache_age > _monitor_services_cache_ttl:
        _refresh_monitor_services_async()
    if cached is None:
        # First call — return a placeholder so the chip doesn't hang.
        return jsonify({'_warming': True})
    out = dict(cached)
    out['_cache_age_s'] = round(cache_age, 1)
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
        # Read last 40 KB to avoid loading huge files. PowerShell Tee-Object
        # writes UTF-16 LE; for tail reads the BOM is at file-start, not in
        # our chunk — detect by null-byte ratio instead.
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 40960))
            chunk = f.read()

        is_utf16 = chunk[:2] == b'\xff\xfe'
        if not is_utf16 and len(chunk) > 200:
            null_ratio = chunk.count(b'\x00') / len(chunk)
            is_utf16 = null_ratio > 0.30          # UTF-16 ASCII ≈ 50% nulls

        if is_utf16:
            # If we landed mid-character, drop the first byte to align.
            start = 2 if chunk[:2] == b'\xff\xfe' else (1 if size % 2 == 1 else 0)
            raw = chunk[start:].decode('utf-16-le', errors='replace').replace('\x00', '')
        else:
            raw = chunk.decode('utf-8', errors='replace')

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if size > 40960 and lines:
            lines = lines[1:]
        return jsonify({'lines': lines[-300:], 'size': size, 'encoding': 'utf-16' if is_utf16 else 'utf-8'})
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
        # Also check externally-launched instances to avoid duplicates
        ext_pid = _find_external_pid(_EXTERNAL_SCRIPTS.get(service, ''))
        if ext_pid:
            return jsonify({'ok': False, 'msg': f'already running externally (PID {ext_pid})', 'pid': ext_pid})
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

    def _kill_tree(pid):
        """Terminate process and all its children (handles GPU subprocs)."""
        try:
            import psutil
            parent = psutil.Process(int(pid))
            for child in parent.children(recursive=True):
                try:
                    child.terminate()
                except Exception:
                    pass
            parent.terminate()
            return True
        except Exception:
            return False

    # Kill dashboard-managed instance (+ its children)
    with _managed_lock:
        proc = _managed.pop(service, None)
    if proc is not None and proc.poll() is None:
        killed = _kill_tree(proc.pid)
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
                    if _kill_tree(p.info['pid']):
                        killed = True
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
        ('oft',      'oft_model_meta.json',           'OFT (Microstructure)','oft_model.pt'),
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
        ('oft',     'oft_model_meta.json',           'OFT (Microstructure)','oft_model.pt',                 '🌊', 'L2/L3'),
        ('meta',    'meta_labeler_meta.json',        'Meta-Labeler',        'meta_labeler.joblib',          '🔍', 'ALL'),
        ('regime',  'regime_classifier_meta.json',   'Regime Classifier',   'regime_classifier.joblib',     '🎯', 'ALL'),
    ]
    def _to_pct(v):
        # Normalize accuracy to percent: trainers vary — some save 0.486, others 48.6.
        # Heuristic: any non-null value ≤ 1.0 is a fraction; multiply by 100.
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f * 100.0 if 0.0 <= f <= 1.0 else f

    # Archived runs directory — used for "≥ N total runs" lower bound.
    # Trainers don't yet emit training_runs Parquet rows, so we count
    # archived metadata files matching the model key as historical runs.
    archived_dir = models_dir / '_archived'
    archived_index: dict[str, int] = {}
    if archived_dir.exists():
        try:
            for f in archived_dir.iterdir():
                if not f.is_file():
                    continue
                fname = f.name.lower()
                # Match patterns like "btc_rf_model_meta_20260501.json" or
                # "scalping_model_*.joblib" — bucket by leading model key.
                for k, mf_, *_ in _ML:
                    base = mf_.replace('_meta.json', '').lower()
                    if fname.startswith(base + '_') or fname.startswith(base + '.'):
                        archived_index[k] = archived_index.get(k, 0) + 1
                        break
        except Exception:
            pass

    import time as _time
    _now_s = _time.time()

    ml_models = []
    for key, mf, label, model_file, icon, market in _ML:
        meta_path  = models_dir / mf
        model_path = models_dir / model_file
        meta = {}
        if meta_path.exists():
            try: meta = _json.loads(meta_path.read_text())
            except Exception: pass
        acc_pct  = _to_pct(meta.get('accuracy'))
        long_pct = _to_pct(meta.get('long_accuracy', 0)) or 0.0
        shrt_pct = _to_pct(meta.get('short_accuracy', 0)) or 0.0
        wf_pct = _to_pct(meta.get('walk_forward_mean_acc'))
        auc_roc = meta.get('auc_roc')
        win_precision = _to_pct(meta.get('win_precision'))
        win_rate_pct = _to_pct(meta.get('win_rate_pct'))
        confidence_threshold = meta.get('confidence_threshold')

        # Derive missing display fields for models whose trainers don't write
        # the standard n_features/n_iterations keys (TFT, GMM regime).
        n_feat = meta.get('n_features')
        n_iter = meta.get('n_iterations')
        n_samp = meta.get('n_samples')
        if key == 'tft':
            # Darts TFT meta currently lacks n_features/n_samples — surface
            # what we DO know so the card isn't all dashes.
            n_iter = n_iter or meta.get('n_epochs')
            if n_feat is None and meta.get('input_chunk_length'):
                n_feat = meta.get('input_chunk_length')  # sequence length proxy
        elif key == 'regime':
            # Probe the GMM joblib once for n_features (cheap — pickled covar).
            # Trainer wraps it as {'model': {'gmm': GaussianMixture, 'scaler': ...}, 'label_map': ...}
            if n_feat is None and model_path.exists():
                try:
                    import joblib as _jl
                    blob = _jl.load(model_path)
                    gmm = None
                    if hasattr(blob, 'means_'):
                        gmm = blob
                    elif isinstance(blob, dict):
                        cand = blob.get('model', blob)
                        if hasattr(cand, 'means_'):
                            gmm = cand
                        elif isinstance(cand, dict):
                            gmm = cand.get('gmm') or cand.get('model')
                    if gmm is not None and hasattr(gmm, 'means_'):
                        n_feat = int(gmm.means_.shape[1])
                        n_iter = n_iter or getattr(gmm, 'n_iter_', None)
                except Exception:
                    pass

        # Class imbalance check: if one direction is near-zero while the other is
        # near the headline, the test-set accuracy is misleading. Prefer the
        # walk-forward mean (when present) and surface a warning to the UI.
        accuracy_warning = None
        headline_acc = acc_pct
        # Meta-labeler is a binary win-or-not classifier — long/short fields
        # don't apply, so suppress the "Long 0% Short 0%" rendering and don't
        # treat the perfect 0/0 split as imbalance.
        is_directionless = (key == 'meta')
        if acc_pct is not None and (long_pct or shrt_pct) and not is_directionless:
            spread = abs(long_pct - shrt_pct)
            if spread >= 30 and min(long_pct, shrt_pct) < 10:
                accuracy_warning = (
                    f"Class-imbalance: long={long_pct:.1f}% short={shrt_pct:.1f}% — "
                    f"the headline {acc_pct:.1f}% over-reports. "
                    + (f"Walk-forward mean {wf_pct:.1f}% is more honest." if wf_pct is not None else "")
                ).strip()
                if wf_pct is not None:
                    headline_acc = wf_pct
        # Training-history derived fields. We don't yet have a training_runs
        # table populated, so we derive a lower-bound from archived metas
        # plus the current artifact mtime for "trained today / staleness".
        meta_mtime = None
        try:
            if meta_path.exists():
                meta_mtime = meta_path.stat().st_mtime
            elif model_path.exists():
                meta_mtime = model_path.stat().st_mtime
        except Exception:
            pass
        age_s = (_now_s - meta_mtime) if meta_mtime else None
        runs_today = 1 if (age_s is not None and age_s <= 86400) else 0
        archived_n = archived_index.get(key, 0)
        total_runs_min = archived_n + (1 if model_path.exists() else 0)

        ml_models.append({
            'key': key, 'label': label, 'icon': icon, 'market': market,
            'model_exists':   model_path.exists(),
            'accuracy':       round(headline_acc, 2) if headline_acc is not None else None,
            'accuracy_test':  round(acc_pct, 2) if acc_pct is not None else None,
            'accuracy_walk_forward': round(wf_pct, 2) if wf_pct is not None else None,
            'accuracy_warning': accuracy_warning,
            'long_accuracy':  round(long_pct, 2),
            'short_accuracy': round(shrt_pct, 2),
            'accuracy_note':  meta.get('accuracy_note'),
            'model_type':     meta.get('model_type'),
            'directionless':  is_directionless,
            'auc_roc':        round(float(auc_roc), 4) if auc_roc is not None else None,
            'win_precision':  round(win_precision, 2) if win_precision is not None else None,
            'win_rate_pct':   round(win_rate_pct, 2) if win_rate_pct is not None else None,
            'confidence_threshold': confidence_threshold,
            'n_samples':      n_samp,
            'n_train':        meta.get('n_train'),
            'n_test':         meta.get('n_test'),
            'n_features':     n_feat,
            'n_iterations':   n_iter,
            'symbols':        meta.get('symbols', []),
            'symbols_count':  len(meta.get('symbols', []) or []),
            'timeframe':      meta.get('timeframe', '--'),
            'last_trained':   meta.get('last_trained', ''),
            'target':         meta.get('target', ''),
            'age_s':          int(age_s) if age_s is not None else None,
            'runs_today':     runs_today,
            'total_runs_min': total_runs_min,
            'is_canonical':   True,   # this row is the legacy/canonical TF
        })

    # ── Multi-TF model variants ────────────────────────────────────────────────
    # The canonical rows above represent the legacy artifact per key (1h for
    # most, 1m for scalping). PR 2 adds <key>_<tf>_*.{joblib,json} alongside
    # the legacy file when trainers are run at non-canonical TFs. Enumerate
    # each per-TF artifact and add one extra row per (key, tf) so the
    # Stability comparison view (PR 4) and the Model Training table can
    # surface them. Keys map to display labels via _ML.
    try:
        from src.utils.model_paths import (
            list_per_tf_artifacts as _list_per_tf,
            CANONICAL_TF as _CANONICAL_TF,
        )
        per_key_label = {k: (lbl, icon, mkt)
                         for k, _mf, lbl, _mfile, icon, mkt in _ML}
        for key in per_key_label:
            for tf, mp, mtp in _list_per_tf(key):
                if tf == _CANONICAL_TF.get(key):
                    # Skip — already represented by the legacy row above.
                    continue
                m: dict = {}
                if mtp.exists():
                    try: m = _json.loads(mtp.read_text())
                    except Exception: pass
                acc_p  = _to_pct(m.get('accuracy'))
                long_p = _to_pct(m.get('long_accuracy', 0)) or 0.0
                shrt_p = _to_pct(m.get('short_accuracy', 0)) or 0.0
                wf_p   = _to_pct(m.get('walk_forward_mean_acc'))
                lbl, icon, mkt = per_key_label[key]
                try:
                    age_v = _now_s - mtp.stat().st_mtime if mtp.exists() else (
                        _now_s - mp.stat().st_mtime if mp.exists() else None
                    )
                except Exception:
                    age_v = None
                ml_models.append({
                    'key':            f'{key}_{tf}',
                    'parent_key':     key,
                    'tf':             tf,
                    'label':          f'{lbl} @ {tf}',
                    'icon':           icon,
                    'market':         mkt,
                    'model_exists':   mp.exists(),
                    'accuracy':       round(acc_p, 2) if acc_p is not None else None,
                    'accuracy_test':  round(acc_p, 2) if acc_p is not None else None,
                    'accuracy_walk_forward': round(wf_p, 2) if wf_p is not None else None,
                    'accuracy_warning': None,
                    'long_accuracy':  round(long_p, 2),
                    'short_accuracy': round(shrt_p, 2),
                    'directionless':  (key == 'meta'),
                    'auc_roc':        round(float(m['auc_roc']), 4) if m.get('auc_roc') is not None else None,
                    'win_precision':  _to_pct(m.get('win_precision')),
                    'win_rate_pct':   _to_pct(m.get('win_rate_pct')),
                    'n_samples':      m.get('n_samples'),
                    'n_train':        m.get('n_train'),
                    'n_test':         m.get('n_test'),
                    'n_features':     m.get('n_features'),
                    'n_iterations':   m.get('n_iterations'),
                    'symbols':        m.get('symbols', []),
                    'symbols_count':  len(m.get('symbols', []) or []),
                    'timeframe':      tf,
                    'last_trained':   m.get('last_trained', ''),
                    'target':         m.get('target', ''),
                    'age_s':          int(age_v) if age_v is not None else None,
                    'runs_today':     1 if (age_v is not None and age_v <= 86400) else 0,
                    'total_runs_min': 1 if mp.exists() else 0,
                    'is_canonical':   False,
                })
    except Exception as _exc:
        # Multi-TF enumeration is best-effort; legacy rows are still served.
        import logging as _lg
        _lg.getLogger(__name__).debug("multi-TF enum failed: %s", _exc)

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

    # ── Simulator paper stats (from StrategySimulatorAgent) ──────────────────
    paper_stats: dict[str, dict] = {}
    try:
        _, _, strat_sim = _get_simulator()
        for row in strat_sim.get_stats():
            sname = row.get("strategy")
            if sname and row.get("n_trades", 0) > 0:
                paper_stats[sname] = {
                    "n":        row.get("n_trades",  0),
                    "wins":     row.get("n_wins",    0),
                    "losses":   row.get("n_losses",  0),
                    "pnl":      row.get("total_pnl", 0.0),
                    "pnl_pct":  row.get("pnl_pct",   0.0),
                    "win_rate": row.get("win_rate",   0.0),
                    "balance":  row.get("balance",    10_000.0),
                }
    except Exception:
        pass

    # ── Walk-forward results ──────────────────────────────────────────────────
    wf_stats: dict[str, dict] = {}
    wf_path = _PROJECT_ROOT / 'data' / 'backtest' / 'wf_results.json'
    if wf_path.exists():
        try:
            import json as _wfj
            for row in _wfj.loads(wf_path.read_text()):
                key = row.get('strategy', '')
                if key:
                    wf_stats[key] = row
        except Exception:
            pass

    # ── Bucket classification for strategies ──────────────────────────────────
    # Three exclusive buckets:
    #   meta_filtered: name ends with _MetaFiltered (passes another signal
    #                  through the meta-labeler — bucket-disable kills these
    #                  even if the underlying primary is enabled).
    #   ml_driven:     uses any model artifact (non-empty 'models' list, or
    #                  the group is 'ML' which the registry already tags).
    #   pure_rule:     everything else (RSI, MACD, BB, VWAP, Donchian, OFI…).
    # Computed here so the registry doesn't need a new column and so any
    # rename / addition automatically buckets correctly.
    bucket_overrides = read_json('data/runtime_overrides.json',
                                 default={}) or {}
    disabled_buckets = set(bucket_overrides.get('disabled_buckets') or [])
    for s in strategies:
        nm = s.get('name', '') or ''
        models_used = s.get('models') or []
        group = s.get('group', '')
        if nm.endswith('_MetaFiltered'):
            bucket = 'meta_filtered'
        elif models_used or group == 'ML':
            bucket = 'ml_driven'
        else:
            bucket = 'pure_rule'
        s['bucket'] = bucket
        s['bucket_disabled'] = bucket in disabled_buckets

    # ── Aggregate ──────────────────────────────────────────────────────────────
    trained_count = sum(1 for m in ml_models if m['model_exists'])
    today_count   = sum(1 for m in ml_models if m.get('runs_today'))
    live_count    = sum(1 for s in strategies
                       if s.get('live_enabled') and not s.get('bucket_disabled'))
    bucket_counts: dict[str, int] = {}
    bucket_live:   dict[str, int] = {}
    for s in strategies:
        b = s.get('bucket', 'pure_rule')
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
        if s.get('live_enabled') and not s.get('bucket_disabled'):
            bucket_live[b] = bucket_live.get(b, 0) + 1

    return jsonify({
        'ml_models':   ml_models,
        'strategies':  strategies,
        'trade_stats': trade_stats,
        'paper_stats': paper_stats,
        'wf_stats':    wf_stats,
        'summary':     summary,
        'buckets': {
            'counts':           bucket_counts,
            'live':             bucket_live,
            'disabled_buckets': sorted(disabled_buckets),
        },
        'aggregate': {
            'models_trained':       trained_count,
            'models_trained_today': today_count,
            'models_total':         len(ml_models),
            'strategies_live':      live_count,
            'strategies_total':     len(strategies),
        },
    })


@app.route('/api/backtest/summary', methods=['GET'])
def backtest_summary():
    """
    Aggregate latest_comparison.json + wf_results.json into one sortable table.
    Each row: strategy, n_trades, win_rate_pct, total_pnl_usdt, sharpe, sortino,
               max_drawdown_pct, calmar, profit_factor, wf_mean_sharpe, wf_consistency, symbols_count
    Symbols are aggregated by averaging ratio metrics and summing count/pnl metrics.
    """
    import json as _j, re as _re
    bt_path  = _PROJECT_ROOT / 'data' / 'backtest' / 'latest_comparison.json'
    wf_path  = _PROJECT_ROOT / 'data' / 'backtest' / 'wf_results.json'
    hist_dir = _PROJECT_ROOT / 'data' / 'backtest'

    # Aggregate per-strategy across symbols
    agg: dict[str, dict] = {}
    if bt_path.exists():
        try:
            for row in _j.loads(bt_path.read_text()):
                strat = _re.sub(r'^[AB]_', '', row.get('strategy', '')).strip()
                if not strat:
                    continue
                a = agg.setdefault(strat, {
                    'strategy': strat, 'n_trades': 0, 'total_pnl_usdt': 0.0,
                    'gross_pnl_usdt': 0.0, 'total_fees_usdt': 0.0,
                    '_wins': 0, '_symbols': set(),
                    '_sharpe': [], '_sortino': [], '_calmar': [],
                    '_max_dd': [], '_pf': [], '_wr': [],
                })
                n   = int(row.get('n_trades', 0))
                wr  = float(row.get('win_rate_pct', 0))
                a['n_trades']        += n
                a['total_pnl_usdt']  += float(row.get('total_pnl_usdt', 0))
                a['gross_pnl_usdt']  += float(row.get('gross_pnl_usdt', 0))
                a['total_fees_usdt'] += float(row.get('total_fees_usdt', 0))
                a['_wins']           += round(n * wr / 100)
                sym = row.get('symbol', '')
                if sym:
                    a['_symbols'].add(sym)
                for key, lst in [('sharpe','_sharpe'),('sortino','_sortino'),
                                  ('calmar','_calmar'),('max_drawdown_pct','_max_dd'),
                                  ('profit_factor','_pf'),('win_rate_pct','_wr')]:
                    v = row.get(key)
                    if v is not None:
                        a[lst].append(float(v))
        except Exception:
            pass

    # Load walk-forward results
    wf_map: dict[str, dict] = {}
    if wf_path.exists():
        try:
            for row in _j.loads(wf_path.read_text()):
                k = row.get('strategy', '')
                if k:
                    wf_map[k] = row
        except Exception:
            pass

    # Build final rows
    rows = []
    def _avg(lst): return round(sum(lst)/len(lst), 3) if lst else None
    for strat, a in agg.items():
        wf = wf_map.get(strat, {})
        rows.append({
            'strategy':        strat,
            'n_trades':        a['n_trades'],
            'win_rate_pct':    round(_avg(a['_wr']) or 0, 1),
            'total_pnl_usdt':  round(a['total_pnl_usdt'], 2),
            'gross_pnl_usdt':  round(a['gross_pnl_usdt'], 2),
            'total_fees_usdt': round(a['total_fees_usdt'], 2),
            'sharpe':          _avg(a['_sharpe']),
            'sortino':         _avg(a['_sortino']),
            'max_drawdown_pct':_avg(a['_max_dd']),
            'calmar':          _avg(a['_calmar']),
            'profit_factor':   _avg(a['_pf']),
            'symbols_count':   len(a['_symbols']),
            'wf_mean_sharpe':  wf.get('wf_mean_sharpe'),
            'wf_consistency':  wf.get('wf_consistency'),
            'wf_decay':        wf.get('wf_decay'),
        })

    # Historical runs list (filenames only, for UI reference)
    hist_files = sorted(
        [f.name for f in hist_dir.glob('comparison_*.csv')],
        reverse=True
    )[:20]

    rows.sort(key=lambda r: (r.get('sharpe') or -999), reverse=True)
    return jsonify({'rows': rows, 'history': hist_files})


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
# Init runs entirely on a background thread so neither /api/simulator/status
# nor /api/simulator/start ever block the Flask worker. The previous design
# called _get_simulator() synchronously, which timed out the Start request
# (>30s) because SimulatorAgent + ContinuousTrainerAgent + StrategySimulatorAgent
# + DatabaseAgent + DuckDB schema replay are all heavy at construction.
_simulator_agent   = None
_trainer_agent     = None
_strategy_sim      = None
_db_agent          = None
_sim_lock          = threading.Lock()
_sim_init_thread   = None
_sim_init_error: str | None = None
_sim_pending_start_cfg: dict | None = None  # config queued during init
_sim_data_store    = None  # SimulatorDataStore singleton (caches DuckDB schema)


def _ensure_sim_init() -> bool:
    """Kick off async simulator construction. Returns True if agents are ready,
    False if init is still in progress. Never blocks for more than the lock."""
    global _sim_init_thread
    if _simulator_agent is not None:
        return True
    with _sim_lock:
        if _simulator_agent is not None:
            return True
        if _sim_init_thread is None or not _sim_init_thread.is_alive():
            _sim_init_thread = threading.Thread(
                target=_do_sim_init, daemon=True, name='sim-init')
            _sim_init_thread.start()
    return False


def _do_sim_init() -> None:
    """Background-only constructor. Installs agents atomically, then drains
    any start config queued via /api/simulator/start during init."""
    global _simulator_agent, _trainer_agent, _strategy_sim, _db_agent
    global _sim_init_error, _sim_pending_start_cfg
    try:
        from src.engine.agents.simulator_agent    import SimulatorAgent
        from src.engine.agents.training_agent     import ContinuousTrainerAgent
        from src.engine.agents.strategy_simulator import StrategySimulatorAgent
        sim   = SimulatorAgent(auto_cycle=True)
        train = ContinuousTrainerAgent()
        strat = StrategySimulatorAgent()
        db = None
        try:
            from src.database.db_agent import DatabaseAgent
            db = DatabaseAgent(bus=sim.bus)
            db.start()
        except Exception as _dbe:
            import logging as _lg
            _lg.getLogger(__name__).debug("DatabaseAgent not started: %s", _dbe)
        with _sim_lock:
            _simulator_agent = sim
            _trainer_agent   = train
            _strategy_sim    = strat
            _db_agent        = db
            queued_cfg = _sim_pending_start_cfg
            _sim_pending_start_cfg = None
        _sim_init_error = None
        if queued_cfg is not None:
            try:
                _apply_sim_start(sim, train, strat, queued_cfg)
            except Exception as e:
                _sim_init_error = f"queued start failed: {e}"
    except Exception as e:
        _sim_init_error = str(e)
        import logging as _lg
        _lg.getLogger(__name__).error("[sim-init] failed: %s", e)


def _apply_sim_start(sim, trainer, strat_sim, cfg: dict) -> None:
    """Apply /api/simulator/start config to already-constructed agents."""
    cfg = dict(cfg or {})
    train_models = cfg.pop('train_models', None)
    if cfg:
        sim.configure(cfg)
    if train_models and isinstance(train_models, list):
        trainer.configure_models(train_models)
    if not trainer._running:
        trainer.start()
    if not strat_sim._running:
        strat_sim.start()
    sim.start()


def _get_sim_store():
    """Cached SimulatorDataStore — DuckDB schema-replay runs once per process,
    not once per status poll (which was eating most of the 4-second budget)."""
    global _sim_data_store
    if _sim_data_store is None:
        with _sim_lock:
            if _sim_data_store is None:
                from src.simulation.data_store import SimulatorDataStore
                _sim_data_store = SimulatorDataStore()
    return _sim_data_store


def _get_simulator():
    """Legacy entry point — agents are required. Kicks off async init if
    needed and waits up to 250ms; raises if still warming up so callers can
    return a clean 'initializing' response instead of hanging."""
    import time as _t
    if _simulator_agent is not None:
        return _simulator_agent, _trainer_agent, _strategy_sim
    _ensure_sim_init()
    deadline = _t.time() + 0.25
    while _t.time() < deadline:
        if _simulator_agent is not None:
            return _simulator_agent, _trainer_agent, _strategy_sim
        _t.sleep(0.01)
    raise RuntimeError("Simulator initializing — retry in a moment")


_db_summary_cache: dict = {'value': {}, 'updated': 0.0}
_DB_SUMMARY_TTL_S = 5.0


def _refresh_db_summary_async():
    """Refresh the DuckDB summary on a background thread. Uses a TTL so the
    UI poll path never waits on DuckDB; the cached value is stale for at
    most _DB_SUMMARY_TTL_S seconds."""
    import time as _t
    if (_t.time() - _db_summary_cache['updated']) < _DB_SUMMARY_TTL_S:
        return
    if _db_summary_cache.get('refreshing'):
        return
    _db_summary_cache['refreshing'] = True
    def _run():
        try:
            val = _get_sim_store().get_summary()
            _db_summary_cache['value'] = val or {}
        except Exception:
            pass
        finally:
            _db_summary_cache['updated'] = _t.time()
            _db_summary_cache['refreshing'] = False
    threading.Thread(target=_run, daemon=True, name='sim-db-summary').start()


@app.route('/api/simulator/status', methods=['GET'])
def simulator_status():
    """Non-blocking. Returns sim.get_status() inline (fast in-memory dict
    access) and a TTL-cached DuckDB summary refreshed on a background
    thread. The Simulator tab polls this every few seconds, so this path
    must never wait on disk I/O.

    Defensive timeout: even though get_status() should be sub-millisecond,
    we still run it on a worker thread with a 2.5s budget so any future
    lock-contention regression in SimulatorAgent can't hang Flask."""
    import queue as _q
    if not _ensure_sim_init():
        msg = ('Agents bootstrapping (first call only, ~5-10s).'
               if _sim_init_error is None
               else f'Init failed: {_sim_init_error}')
        return jsonify({
            'state': 'initializing' if _sim_init_error is None else 'error',
            'message': msg,
            'trainer_stats': {}, 'db_summary': {},
        })
    out_q: _q.Queue = _q.Queue(maxsize=1)
    def _build():
        try:
            sim, trainer = _simulator_agent, _trainer_agent
            st = sim.get_status()
            st['trainer_stats'] = trainer.get_stats() if trainer else {}
            out_q.put(('ok', st))
        except Exception as exc:
            out_q.put(('err', str(exc)))
    threading.Thread(target=_build, daemon=True, name='sim-status').start()
    try:
        kind, payload = out_q.get(timeout=2.5)
    except _q.Empty:
        return jsonify({
            'state': 'busy',
            'message': 'sim.get_status() did not return within 2.5s — agent may be busy.',
            'trainer_stats': {}, 'db_summary': {},
        })
    if kind == 'err':
        return jsonify({'error': payload, 'state': 'error'}), 500
    _refresh_db_summary_async()
    payload['db_summary'] = _db_summary_cache.get('value', {}) or {}
    return jsonify(payload)


@app.route('/api/simulator/start', methods=['POST'])
def simulator_start():
    """Start or resume the simulator replay. Returns immediately — if agents
    are still initializing, the start config is queued and applied as soon
    as init completes."""
    global _sim_pending_start_cfg
    cfg = request.get_json(force=True) or {}
    if _simulator_agent is None:
        with _sim_lock:
            _sim_pending_start_cfg = dict(cfg)
        _ensure_sim_init()
        return jsonify({
            'ok': True, 'queued': True, 'state': 'initializing',
            'message': 'Simulator agents are bootstrapping; start command queued.',
        })
    try:
        _apply_sim_start(_simulator_agent, _trainer_agent, _strategy_sim, cfg)
        return jsonify({'ok': True, 'status': _simulator_agent.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/pause', methods=['POST'])
def simulator_pause():
    try:
        sim, _, _ = _get_simulator()
        sim.pause()
        return jsonify({'ok': True, 'status': sim.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/resume', methods=['POST'])
def simulator_resume():
    try:
        sim, _, _ = _get_simulator()
        sim.resume()
        return jsonify({'ok': True, 'status': sim.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/stop', methods=['POST'])
def simulator_stop():
    try:
        sim, trainer, strat_sim = _get_simulator()
        sim.stop()
        trainer.stop()
        strat_sim.stop()
        return jsonify({'ok': True, 'status': sim.get_status()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/config', methods=['POST'])
def simulator_config():
    """Update simulator config (symbol, timeframe, speed, scenario, date range)."""
    try:
        sim, _, _ = _get_simulator()
        cfg = request.get_json(force=True) or {}
        allowed = {'symbol', 'timeframe', 'speed', 'scenario', 'start_date', 'end_date'}
        clean = {k: v for k, v in cfg.items() if k in allowed}
        sim.configure(clean)
        return jsonify({'ok': True, 'config': sim.get_status()['config']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/strategy_stats', methods=['GET'])
def simulator_strategy_stats():
    """Return per-strategy virtual account performance (live from StrategySimulatorAgent)."""
    try:
        _, _, strat_sim = _get_simulator()
        live = strat_sim.get_stats()
        # If agent hasn't seen candles yet, fall back to DB
        has_data = any(r.get("n_trades", 0) > 0 for r in live)
        if not has_data:
            try:
                from src.simulation.data_store import SimulatorDataStore
                live = SimulatorDataStore().get_strategy_stats() or live
            except Exception:
                pass
        return jsonify({'stats': live, 'virtual_capital': 10_000.0})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulator/strategy_reset', methods=['POST'])
def simulator_strategy_reset():
    """Reset all virtual strategy accounts to initial capital."""
    try:
        _, _, strat_sim = _get_simulator()
        strat_sim.reset_stats()
        return jsonify({'ok': True})
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


# TTL cache for /api/monitor/downloader/status. Pre-fix: every poll did
# two full psutil.process_iter() scans (one for archive, one for
# watchlist) plus iterdir() over data/raw + data/raw/historical. With
# 100+ python subprocesses around (training, bot, agents) on this host,
# psutil.process_iter took 5-18 s. Cache for 10 s so the operator's
# refresh button still feels live without burning psutil twice per click.
_dl_status_cache: dict = {'value': None, 'ts': 0.0}
_dl_status_cache_ttl = 10.0
_dl_status_lock = threading.Lock()


def _build_downloader_status_snapshot() -> dict:
    """Pure read of the downloader state — fs scan + 2× psutil.
    Wrapped by the TTL-cached route so concurrent polls fan in."""
    import json as _json

    watchlist = read_json('data/watchlist.json', default=['BTC/USDT','ETH/USDT','SOL/USDT','ADA/USDT'])
    raw_dir  = _PROJECT_ROOT / 'data' / 'raw'
    hist_dir = raw_dir / 'historical'

    def _scan_dir(d: _Path) -> dict:
        if not d.exists():
            return {}
        result = {}
        for f in d.iterdir():
            if not f.name.endswith('.csv.gz'):
                continue
            st = f.stat()
            result[f.name] = {
                'size_mb': round(st.st_size / 1e6, 1),
                'mtime':   _time.strftime('%Y-%m-%d %H:%M', _time.localtime(st.st_mtime)),
            }
        return result

    hist_files = _scan_dir(hist_dir)
    curr_files = _scan_dir(raw_dir)

    # Which watchlist symbols are missing from historical (no _spot_1s file)
    missing_historical = []
    for sym in watchlist:
        safe = sym.replace('/', '_')
        if f'{safe}_spot_1s.csv.gz' not in hist_files:
            missing_historical.append(sym)

    # Which watchlist symbols are missing current 1m data
    missing_current_1m = []
    for sym in watchlist:
        safe = sym.replace('/', '_')
        if f'{safe}_1m.csv.gz' not in curr_files:
            missing_current_1m.append(sym)

    # Single psutil scan (was two). Pre-fix: _svc_running was called
    # twice (archive + watchlist), each iterating ALL python processes
    # on the box — under load (100+ training subprocs) this took 5-18 s
    # PER request. Now we walk once and check both substrings.
    archive_running = False
    watchlist_running = False
    try:
        import psutil
        for p in psutil.process_iter(['cmdline']):
            cmd = ' '.join(p.info.get('cmdline') or [])
            if not archive_running and 'binance_archive_downloader' in cmd:
                archive_running = True
            if not watchlist_running and 'watchlist_downloader' in cmd:
                watchlist_running = True
            if archive_running and watchlist_running:
                break
    except Exception:
        pass

    # Load saved folder state
    state_path = _PROJECT_ROOT / 'data' / 'downloader_state.json'
    saved_state = {}
    if state_path.exists():
        try:
            saved_state = _json.loads(state_path.read_text(encoding='utf-8'))
        except Exception:
            pass

    # Save current state
    new_state = {
        'historical_files': len(hist_files),
        'current_files': len(curr_files),
        'missing_historical': missing_historical,
        'missing_current_1m': missing_current_1m,
        'last_checked': _time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    try:
        state_path.write_text(_json.dumps(new_state, indent=2), encoding='utf-8')
    except Exception:
        pass

    return {
        'historical': {
            'dir': str(hist_dir),
            'file_count': len(hist_files),
            'files': dict(list(hist_files.items())[:10]),   # first 10 for display
            'missing_symbols': missing_historical,
            'running': archive_running,
        },
        'current': {
            'dir': str(raw_dir),
            'file_count': len(curr_files),
            'missing_1m': missing_current_1m,
            'running': watchlist_running,
        },
        'saved_state': saved_state,
    }


def _refresh_dl_status_async():
    def _job():
        try:
            value = _build_downloader_status_snapshot()
            with _dl_status_lock:
                _dl_status_cache['value'] = value
                _dl_status_cache['ts'] = time.time()
        except Exception as exc:
            with _dl_status_lock:
                _dl_status_cache['value'] = {'_error': str(exc)}
                _dl_status_cache['ts'] = time.time()
    threading.Thread(target=_job, daemon=True, name='dl-status-refresh').start()


@app.route('/api/monitor/downloader/status')
def monitor_downloader_status():
    """Returns the state of both data folders. TTL-cached (10 s) — see
    _dl_status_cache_ttl. Pre-fix this route did two psutil.process_iter()
    scans per request and took 5-18 s under load."""
    with _dl_status_lock:
        cached = _dl_status_cache.get('value')
        cache_age = time.time() - (_dl_status_cache.get('ts') or 0)
    if cached is None or cache_age > _dl_status_cache_ttl:
        _refresh_dl_status_async()
    if cached is None:
        return jsonify({'_warming': True})
    out = dict(cached)
    out['_cache_age_s'] = round(cache_age, 1)
    return jsonify(out)


@app.route('/api/monitor/downloader/migrate', methods=['POST'])
def monitor_migrate_to_historical():
    """
    Move healthy *_spot_1s.csv.gz files from data/raw/ to data/raw/historical/.
    Deletes corrupted ones.  Safe to call when archive downloader is NOT active.
    Returns a summary of what was moved / deleted.
    """
    import gzip as _gz
    import shutil as _sh
    import collections as _col

    raw_dir  = _PROJECT_ROOT / 'data' / 'raw'
    hist_dir = raw_dir / 'historical'
    hist_dir.mkdir(parents=True, exist_ok=True)

    moved, deleted, skipped = [], [], []
    for f in raw_dir.glob('*_spot_1s.csv.gz'):
        try:
            last = list(_col.deque(_gz.open(f, 'rt', encoding='utf-8'), maxlen=1))
            if not last:
                f.unlink()
                deleted.append(f.name)
                continue
        except Exception:
            f.unlink()
            deleted.append(f.name)
            continue

        dest = hist_dir / f.name
        if dest.exists():
            skipped.append(f.name)
            continue
        _sh.move(str(f), str(dest))
        moved.append(f.name)

    return jsonify({'moved': moved, 'deleted': deleted, 'skipped': skipped})


# ─── Parquet Store / Database endpoints (was QuestDB pre-Phase-3) ────────────

_db_status_cache: dict = {'value': None, 'ts': 0.0}
_db_status_cache_ttl = 300.0   # seconds — row counts change in batches, not in real time
_db_status_lock = threading.Lock()


def _refresh_db_status_async():
    """Background-thread refresh of the parquet row-count cache. The
    Monitor tab polls /api/db/status every 30s; without caching, each poll
    burned ~60s scanning 89M+ rows × 7 tables and timed the request out
    (empty body → JS chip stuck on 'Loading…'). Cache TTL is 5 min — row
    counts don't change in real time anyway."""
    def _job():
        try:
            from src.database.parquet_client import get_client
            c = get_client()
            available = c.is_available(force=True)
            tables = {}
            if available:
                for tbl in ['market_data', 'trade_events', 'model_signals',
                            'training_telemetry', 'strategy_performance',
                            'news_sentiment', 'backtest_results']:
                    try:
                        rows = c.query(f"SELECT COUNT(*) as n FROM {tbl}")
                        tables[tbl] = rows[0]['n'] if rows else 0
                    except Exception:
                        tables[tbl] = None
            value = {
                'available': available,
                'backend':   'duckdb+parquet',
                'host':      'in-process',
                'http_port': None,
                'ilp_port':  None,
                'data_dir':  str(c.base_dir),
                'tables':    tables,
            }
            with _db_status_lock:
                _db_status_cache['value'] = value
                _db_status_cache['ts'] = time.time()
        except Exception as exc:
            with _db_status_lock:
                _db_status_cache['value'] = {'available': False, 'error': str(exc)}
                _db_status_cache['ts'] = time.time()
    threading.Thread(target=_job, daemon=True, name='db-status-refresh').start()


@app.route('/api/db/status')
def db_status():
    """ParquetClient store status + table row counts.

    Returns the most-recent TTL-cached snapshot in O(1). A background
    thread refreshes the row counts every 5 minutes; the dashboard's
    poll cadence (30s) never blocks on the slow scan."""
    with _db_status_lock:
        cached = _db_status_cache.get('value')
        cache_age = time.time() - (_db_status_cache.get('ts') or 0)
    if cached is None or cache_age > _db_status_cache_ttl:
        _refresh_db_status_async()
    if cached is None:
        # First call — return a placeholder so the chip doesn't hang
        return jsonify({
            'available': True,
            'backend':   'duckdb+parquet',
            'host':      'in-process',
            'http_port': None,
            'ilp_port':  None,
            'data_dir':  '',
            'tables':    {},
            'cache_warming': True,
        })
    cached['cache_age_s'] = round(cache_age, 1)
    return jsonify(cached)


@app.route('/api/db/query', methods=['POST'])
def db_query():
    """Execute arbitrary SQL against QuestDB (read-only SELECT only)."""
    body = request.get_json(force=True) or {}
    sql = (body.get('sql') or '').strip()
    if not sql:
        return jsonify({'error': 'sql required'}), 400
    # Safety: only allow SELECT
    if not sql.upper().lstrip().startswith('SELECT'):
        return jsonify({'error': 'Only SELECT queries allowed'}), 403
    try:
        from src.database.parquet_client import get_client
        c = get_client()
        rows = c.query(sql)
        return jsonify({'rows': rows, 'count': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/db/strategy_history')
def db_strategy_history():
    """Return PNL history for one strategy over last N days."""
    strategy = request.args.get('strategy', '')
    days = int(request.args.get('days', 7))
    if not strategy:
        return jsonify({'error': 'strategy required'}), 400
    try:
        from src.database.parquet_client import get_client
        rows = get_client().get_strategy_history(strategy, days)
        return jsonify({'rows': rows, 'strategy': strategy, 'days': days})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Stability comparison: per-strategy × per-timeframe heatmap data ─────────
@app.route('/api/strategy/stability', methods=['GET'])
def api_strategy_stability():
    """Build a (strategy × timeframe) matrix from latest_comparison.json
    and wf_results.json. Drives the Stability heatmap on the Strategy tab.

    Each cell aggregates across symbols (mean Sharpe, mean WF Sharpe,
    mean Win%, sum Trades). 'Best TF' per strategy is the TF with the
    highest WF Sharpe (or Sharpe if WF is missing).

    Returns:
      {strategies: [...], timeframes: [...],
       cells: {(strategy, tf): {sharpe_avg, wf_sharpe_avg, ...}},
       best_tf: {strategy: tf}}

    When the backtester hasn't yet been run with multi-TF, this returns
    just the single-TF rows (defaulting to 1h) so the UI shows a usable
    column even before PR 3's multi-TF run lands."""
    import json as _j
    from src.engine import strategy_registry as _sr

    bt_path = _PROJECT_ROOT / 'data' / 'backtest' / 'latest_comparison.json'
    wf_path = _PROJECT_ROOT / 'data' / 'backtest' / 'wf_results.json'
    bt_rows = _j.loads(bt_path.read_text()) if bt_path.exists() else []
    wf_rows = _j.loads(wf_path.read_text()) if wf_path.exists() else []

    # label → registry name, so latest_comparison labels (which sometimes
    # carry "A_" / "B_" prefixes) round-trip cleanly to the registry key
    label_to_key: dict[str, str] = {}
    for nm, info in _sr.REGISTRY.items():
        label_to_key[info.get('label', nm)] = nm

    timeframes_seen: set[str] = set()
    strategies_seen: set[str] = set()
    # cells[strategy][tf] = aggregator dict
    cells: dict[str, dict[str, dict]] = {}

    def _safe_float(v):
        try: return float(v)
        except (TypeError, ValueError): return None

    for r in bt_rows:
        raw  = (r.get('strategy') or '').strip()
        clean = re.sub(r'^[AB]_', '', raw)
        reg_key = label_to_key.get(clean, clean)
        tf = (r.get('timeframe') or '1h').strip() or '1h'
        timeframes_seen.add(tf)
        strategies_seen.add(reg_key)
        bucket = cells.setdefault(reg_key, {}).setdefault(tf, {
            'n_symbols':       0, 'n_trades_total': 0, 'pnl_total': 0.0,
            'win_rate_sum':    0.0, 'win_rate_n':    0,
            'sharpe_sum':      0.0, 'sharpe_n':      0,
            'maxdd_sum':       0.0, 'maxdd_n':       0,
            'pf_sum':          0.0, 'pf_n':          0,
            'wf_sharpe_sum':   0.0, 'wf_sharpe_n':   0,
            'wf_consist_sum':  0.0, 'wf_consist_n':  0,
        })
        bucket['n_symbols'] += 1
        bucket['n_trades_total'] += int(r.get('n_trades') or 0)
        bucket['pnl_total']      += float(r.get('total_pnl_usdt') or 0.0)
        for src_key, sum_key, n_key in (
            ('win_rate_pct',     'win_rate_sum',  'win_rate_n'),
            ('sharpe',           'sharpe_sum',    'sharpe_n'),
            ('max_drawdown_pct', 'maxdd_sum',     'maxdd_n'),
            ('profit_factor',    'pf_sum',        'pf_n'),
        ):
            v = _safe_float(r.get(src_key))
            if v is not None:
                bucket[sum_key] += v
                bucket[n_key]   += 1

    for r in wf_rows:
        reg_key = (r.get('strategy') or '').strip()
        tf = (r.get('timeframe') or '1h').strip() or '1h'
        if reg_key not in cells:
            continue
        bucket = cells[reg_key].get(tf)
        if not bucket:
            continue
        for src_key, sum_key, n_key in (
            ('wf_mean_sharpe', 'wf_sharpe_sum',  'wf_sharpe_n'),
            ('wf_consistency', 'wf_consist_sum', 'wf_consist_n'),
        ):
            v = _safe_float(r.get(src_key))
            if v is not None:
                bucket[sum_key] += v
                bucket[n_key]   += 1

    # Flatten to {strategy: {tf: avgs}} + compute best_tf per strategy
    flat_cells: dict[str, dict[str, dict]] = {}
    best_tf: dict[str, str] = {}
    for strat, by_tf in cells.items():
        flat_cells[strat] = {}
        ranked: list[tuple[float, str]] = []
        for tf, b in by_tf.items():
            avg = lambda s, n: round(b[s] / b[n], 3) if b[n] else None
            row = {
                'tf':                  tf,
                'n_symbols':           b['n_symbols'],
                'n_trades_total':      b['n_trades_total'],
                'pnl_total':           round(b['pnl_total'], 2),
                'sharpe_avg':          avg('sharpe_sum',     'sharpe_n'),
                'win_rate_avg':        avg('win_rate_sum',   'win_rate_n'),
                'maxdd_avg':           avg('maxdd_sum',      'maxdd_n'),
                'profit_factor_avg':   avg('pf_sum',         'pf_n'),
                'wf_sharpe_avg':       avg('wf_sharpe_sum',  'wf_sharpe_n'),
                'wf_consistency_avg':  avg('wf_consist_sum', 'wf_consist_n'),
            }
            flat_cells[strat][tf] = row
            # Use WF Sharpe as the ranking signal; fall back to Sharpe
            score = row['wf_sharpe_avg']
            if score is None:
                score = row['sharpe_avg']
            if score is not None:
                ranked.append((score, tf))
        if ranked:
            ranked.sort(reverse=True)
            best_tf[strat] = ranked[0][1]

    return jsonify({
        'strategies':  sorted(strategies_seen),
        'timeframes':  sorted(timeframes_seen),
        'cells':       flat_cells,
        'best_tf':     best_tf,
        'has_multi_tf': len(timeframes_seen) > 1,
    })


# ─── Strategy TF pinning (Phase A) ────────────────────────────────────────────
@app.route('/api/strategy/tf_pinning', methods=['GET'])
def api_strategy_tf_pinning_get():
    """Return the current pinning state (auto + manual + effective per
    strategy). The orchestrator writes 'auto' after each multi-TF backtest;
    the operator can override via POST below."""
    from src.engine import strategy_tf_pinning as _tp
    state = _tp.read_state()
    return jsonify({
        'auto':       state.get('auto')   or {},
        'manual':     state.get('manual') or {},
        'effective':  _tp.get_all_pins(),
        'updated_at': state.get('updated_at') or '',
        'default_tf': _tp.DEFAULT_TF,
    })


@app.route('/api/strategy/tf_pinning', methods=['POST'])
@require_api_key
def api_strategy_tf_pinning_set():
    """Set or clear a manual TF override for one strategy.
    Body: {"strategy": "RSI_MeanReversion", "tf": "4h"}.
    Pass tf="" or null to clear (falls back to auto / default)."""
    body = request.get_json(silent=True) or {}
    strat = (body.get('strategy') or '').strip()
    tf = body.get('tf')
    if tf is not None:
        tf = str(tf).strip() or None
    if not strat:
        return jsonify({'ok': False, 'error': 'strategy required'}), 400
    from src.engine import strategy_tf_pinning as _tp
    state = _tp.set_manual_pin(strat, tf)
    return jsonify({'ok': True, 'strategy': strat, 'tf': tf,
                    'state': {'auto': state.get('auto'),
                              'manual': state.get('manual')}})


# ─── Bucket aggregate comparison (Pure rule vs ML-driven vs Meta-filtered) ───
@app.route('/api/strategy/bucket_compare', methods=['GET'])
def api_strategy_bucket_compare():
    """Aggregate WF Sharpe, WF consistency, Win%, MaxDD, total trades, total
    PnL per bucket. Pulls from data/backtest/latest_comparison.json (in-sample)
    and data/backtest/wf_results.json (out-of-sample). Drives the
    'Pure vs ML' card on the Strategy tab."""
    import json as _j
    from src.engine import strategy_registry as _sr
    bt_path = _PROJECT_ROOT / 'data' / 'backtest' / 'latest_comparison.json'
    wf_path = _PROJECT_ROOT / 'data' / 'backtest' / 'wf_results.json'
    bt_rows = _j.loads(bt_path.read_text()) if bt_path.exists() else []
    wf_rows = _j.loads(wf_path.read_text()) if wf_path.exists() else []
    # Build label → registry name map (latest_comparison uses labels)
    label_to_key: dict[str, str] = {}
    for nm, info in _sr.REGISTRY.items():
        label_to_key[info.get('label', nm)] = nm

    buckets: dict[str, dict] = {
        b: {'bucket': b, 'n_strategies': 0, 'n_trades_total': 0, 'pnl_total': 0.0,
            'win_rate_sum': 0.0, 'win_rate_n': 0,
            'sharpe_sum': 0.0, 'sharpe_n': 0,
            'maxdd_sum': 0.0, 'maxdd_n': 0,
            'pf_sum': 0.0, 'pf_n': 0,
            'wf_sharpe_sum': 0.0, 'wf_sharpe_n': 0,
            'wf_consist_sum': 0.0, 'wf_consist_n': 0}
        for b in ('pure_rule', 'ml_driven', 'meta_filtered')
    }

    def _add(b: str, key: str, val, count_key: str, sum_key: str):
        if val is None: return
        try: f = float(val)
        except (TypeError, ValueError): return
        buckets[b][sum_key] += f
        buckets[b][count_key] += 1

    for r in bt_rows:
        raw = (r.get('strategy') or '').strip()
        # latest_comparison rows look like "A_RSI_MeanReversion" with prefix
        clean = re.sub(r'^[AB]_', '', raw)
        reg_key = label_to_key.get(clean, clean)
        bucket = _sr.bucket_for(reg_key)
        if bucket not in buckets: continue
        buckets[bucket]['n_strategies'] += 1
        buckets[bucket]['n_trades_total'] += int(r.get('n_trades') or 0)
        buckets[bucket]['pnl_total']     += float(r.get('total_pnl_usdt') or 0.0)
        _add(bucket, reg_key, r.get('win_rate_pct'),       'win_rate_n',  'win_rate_sum')
        _add(bucket, reg_key, r.get('sharpe'),             'sharpe_n',    'sharpe_sum')
        _add(bucket, reg_key, r.get('max_drawdown_pct'),   'maxdd_n',     'maxdd_sum')
        _add(bucket, reg_key, r.get('profit_factor'),      'pf_n',        'pf_sum')

    for r in wf_rows:
        reg_key = (r.get('strategy') or '').strip()
        bucket = _sr.bucket_for(reg_key)
        if bucket not in buckets: continue
        _add(bucket, reg_key, r.get('wf_mean_sharpe'),  'wf_sharpe_n',  'wf_sharpe_sum')
        _add(bucket, reg_key, r.get('wf_consistency'),  'wf_consist_n', 'wf_consist_sum')

    out = []
    for b, x in buckets.items():
        avg = lambda s, n: round(x[s] / x[n], 3) if x[n] else None
        out.append({
            'bucket':           b,
            'n_strategies':     x['n_strategies'],
            'n_trades_total':   x['n_trades_total'],
            'pnl_total':        round(x['pnl_total'], 2),
            'win_rate_avg':     avg('win_rate_sum',  'win_rate_n'),
            'sharpe_avg':       avg('sharpe_sum',    'sharpe_n'),
            'maxdd_avg':        avg('maxdd_sum',     'maxdd_n'),
            'profit_factor_avg':avg('pf_sum',        'pf_n'),
            'wf_sharpe_avg':    avg('wf_sharpe_sum', 'wf_sharpe_n'),
            'wf_consistency_avg': avg('wf_consist_sum', 'wf_consist_n'),
        })
    return jsonify({'buckets': out})


# ─── Bucket disable toggle ───────────────────────────────────────────────────
_VALID_BUCKETS = ('pure_rule', 'ml_driven', 'meta_filtered')


@app.route('/api/strategy/bucket', methods=['POST'])
@require_api_key
def api_strategy_bucket_toggle():
    """Enable / disable an entire strategy bucket. Persists to
    data/runtime_overrides.json under the 'disabled_buckets' key. The bot
    and backtester read this list before activating a strategy."""
    body = request.get_json(silent=True) or {}
    bucket = (body.get('bucket') or '').strip()
    enabled = bool(body.get('enabled', True))
    if bucket not in _VALID_BUCKETS:
        return jsonify({'ok': False,
                        'error': f'invalid bucket: {bucket}',
                        'valid': list(_VALID_BUCKETS)}), 400
    overrides = read_json('data/runtime_overrides.json', default={}) or {}
    if not isinstance(overrides, dict):
        overrides = {}
    disabled = set(overrides.get('disabled_buckets') or [])
    if enabled:
        disabled.discard(bucket)
    else:
        disabled.add(bucket)
    overrides['disabled_buckets'] = sorted(disabled)
    write_json('data/runtime_overrides.json', overrides)
    return jsonify({'ok': True, 'bucket': bucket, 'enabled': enabled,
                    'disabled_buckets': overrides['disabled_buckets']})


# ─── Manual training triggers ────────────────────────────────────────────────
# Per-process job log. Capped at _TRAINING_JOBS_MAX so it can't grow forever
# even if someone hammers the Train button. Keyed by job_id (uuid hex).
_training_jobs: dict[str, dict] = {}
_training_jobs_lock = threading.Lock()
_TRAINING_JOBS_MAX = 50

# Maps the model key shown in the UI to (module, callable, accepts_tf).
# accepts_tf=True means the function takes a `timeframe=` kwarg and the
# trainer should pass it through; False means the function ignores TF
# (e.g. train_all takes per_key_tfs dict, regime classifier takes
# symbols= only, OFT main() is argparse-driven). Pre-fix bug:
# 'regime' dispatched to train_all which DOESN'T accept timeframe= and
# every regime training run failed with TypeError.
_TRAINER_DISPATCH = {
    'base':     ('src.engine.train_model',           'train_model',          True),
    'trend':    ('src.engine.train_trend_model',     'train_trend_model',    True),
    'futures':  ('src.engine.train_futures_model',   'train_futures_model',  True),
    'scalping': ('src.engine.train_scalping_model',  'train_scalping_model', True),
    'tft':      ('src.engine.train_tft_model',       'train_tft_model',      True),
    # OFT entry point lives in joint_oft_rl (oft_trainer.py is the library
    # module — it exposes OFTTrainer/OFTTrainerConfig classes but no main()).
    # Pre-fix the trainer dispatch tried to call oft_trainer.main() and
    # bailed with AttributeError every time the user clicked Train.
    'oft':      ('src.training.joint_oft_rl',         'main',                 False),
    'meta':     ('src.engine.train_meta_labeler',    'train_meta_labeler',   True),
    # Regime classifier has its own standalone trainer.
    'regime':   ('src.analysis.regime_classifier',   'train_regime_classifier', False),
    # train_all() takes per_key_tfs dict, not a single tf — pass nothing
    # and let it use its own DEFAULT_PER_KEY_TFS map.
    'all':      ('src.engine.train_all_models',      'train_all',            False),
}


def _record_job(job_id: str, **fields) -> None:
    with _training_jobs_lock:
        e = _training_jobs.get(job_id) or {'job_id': job_id}
        e.update(fields)
        _training_jobs[job_id] = e
        # Cap: drop the oldest by created_at if over.
        if len(_training_jobs) > _TRAINING_JOBS_MAX:
            oldest = min(_training_jobs.values(),
                         key=lambda x: x.get('created_at', 0))
            _training_jobs.pop(oldest['job_id'], None)
    # Persist on every state change so an operator who started OFT
    # (140 min run) survives an in-flight dashboard restart.
    try:
        _persist_training_jobs()
    except Exception:
        pass


_training_active_procs: dict[str, "subprocess.Popen"] = {}
_training_active_lock = threading.Lock()

# Persisted training-jobs state. Survives dashboard restarts so an
# operator who just kicked off OFT (140 min run) doesn't lose visibility
# when the dashboard process dies for any reason. Each entry is the
# same record shape as _training_jobs but on disk; we reload at startup
# and rejoin any subprocess whose pid is still alive.
_TRAINING_JOBS_FILE = _Path(project_root) / 'data' / 'training_jobs.json'
_TRAINING_LOG_DIR = _Path(project_root) / 'logs' / 'train'


def _persist_training_jobs() -> None:
    """Atomic write of _training_jobs to disk. Called from _record_job."""
    try:
        _TRAINING_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _training_jobs_lock:
            snap = {jid: dict(j) for jid, j in _training_jobs.items()}
        # Strip the live Popen handle (not JSON-serialisable) and any
        # transient fields the dict may carry.
        for j in snap.values():
            j.pop('_proc', None)
        from src.utils.safe_json import write_json
        # safe_json.write_json builds the lockfile path via string concat,
        # so it can't take a pathlib.Path — pass the stringified form.
        write_json(str(_TRAINING_JOBS_FILE),
                   {'jobs': snap, 'saved_at': time.time()})
    except Exception as exc:
        logger = __import__('logging').getLogger(__name__)
        logger.debug('[training] persist failed: %s', exc)


def _load_training_jobs() -> None:
    """Reload persisted jobs at startup. Reconcile each 'running' entry
    against psutil — if the recorded child_pid is still alive, attach a
    poller thread to wait for it; if dead, infer completion from the
    log file tail and mark 'done' / 'lost' accordingly."""
    if not _TRAINING_JOBS_FILE.exists():
        return
    try:
        from src.utils.safe_json import read_json
        raw = read_json(str(_TRAINING_JOBS_FILE), default={}) or {}
        saved = raw.get('jobs') or {}
    except Exception:
        return
    if not isinstance(saved, dict) or not saved:
        return
    now = time.time()
    with _training_jobs_lock:
        for jid, j in saved.items():
            if not isinstance(j, dict):
                continue
            _training_jobs[jid] = dict(j)
    # Reconcile in-flight subprocesses. Done outside the lock so the
    # poller threads can update _training_jobs as they progress.
    try:
        import psutil
    except ImportError:
        return
    for jid, j in list(saved.items()):
        if j.get('status') not in ('running', 'queued', 'starting'):
            continue
        pid = j.get('child_pid')
        if not pid:
            # No PID recorded — can't reconnect. Mark as lost.
            _record_job(jid, status='lost', finished_at=now,
                        errors=['dashboard restart — pid not recorded; '
                                'subprocess orphaned, check logs/train/'
                                + jid + '.log'])
            continue
        try:
            alive = psutil.pid_exists(int(pid))
        except Exception:
            alive = False
        if alive:
            # Subprocess survived the dashboard restart. Spawn a poller
            # thread to wait for it and write the final status when it
            # exits.
            threading.Thread(
                target=_reattach_training_subprocess,
                args=(jid, int(pid)),
                daemon=True,
                name=f'train-reattach-{jid}',
            ).start()
        else:
            # Subprocess died while the dashboard was down. Snapshot
            # the log tail so the operator can see what happened.
            tail = ''
            log_path = _TRAINING_LOG_DIR / f'{jid}.log'
            try:
                if log_path.exists():
                    with open(log_path, 'rb') as f:
                        f.seek(max(0, log_path.stat().st_size - 2048))
                        tail = f.read().decode('utf-8', 'replace')[-1500:]
            except Exception:
                pass
            _record_job(jid, status='lost', finished_at=now,
                        errors=[f'subprocess pid {pid} died while dashboard '
                                f'was offline. Log tail:\n{tail}'])


def _detect_orphan_training_subprocesses() -> None:
    """Scan all running python.exe subprocesses for trainer-shaped command
    lines that aren't tracked in _training_jobs. Synthesize a job entry
    for each so the dashboard shows them and reattaches a poller.

    This rescues subprocesses that were spawned by a previous dashboard
    instance which didn't yet have the persistence layer (PR-40), or by
    manual command-line testing — without this they'd burn GPU/CPU
    invisibly until they finish or the operator notices in Task Manager.
    """
    try:
        import psutil
    except ImportError:
        return
    # Inverse map of trainer module name → model key, lifted from
    # _TRAINER_DISPATCH so we stay in sync if new trainers are added.
    module_to_key = {mod: k for k, (mod, _, _) in _TRAINER_DISPATCH.items()}
    with _training_jobs_lock:
        tracked_pids = {j.get('child_pid') for j in _training_jobs.values()
                        if j.get('child_pid')}
    found = 0
    now = time.time()
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if not (p.info.get('name') or '').lower().startswith('python'):
                continue
            cmd = ' '.join(p.info.get('cmdline') or [])
            pid = p.info['pid']
            if pid in tracked_pids:
                continue
            # Match `import <module> as m; ... fn()` — the standard
            # invocation _spawn_training_subprocess uses.
            matched_key = None
            for module_path, key in module_to_key.items():
                if f'import {module_path}' in cmd and 'fn(' in cmd:
                    matched_key = key
                    break
            if not matched_key:
                continue
            # Synthesize a job record. created_at = the subprocess's
            # actual start time so elapsed_s is accurate from boot.
            jid = f'orphan-{pid}'
            created = float(p.info.get('create_time') or now)
            _record_job(
                jid,
                model=matched_key,
                status='running',
                created_at=created,
                queued_at=created,
                started_at=created,
                child_pid=pid,
                resource_kind=_resource_kind_for(matched_key),
                progress=0,
                total=1,
                tf=None,
                progress_label='reattached (orphan from prior boot)',
            )
            # Spawn poller to wait for it like a normal job.
            threading.Thread(
                target=_reattach_training_subprocess,
                args=(jid, pid),
                daemon=True,
                name=f'train-orphan-{pid}',
            ).start()
            found += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if found:
        import logging as _lg
        _lg.getLogger(__name__).info(
            '[training] reattached %d orphan training subprocess(es)', found)


def _reattach_training_subprocess(job_id: str, pid: int) -> None:
    """Wait for an orphaned-but-alive training subprocess to finish,
    then record the result. Used after dashboard restart picks up a
    persisted 'running' job whose child is still alive."""
    try:
        import psutil
        try:
            p = psutil.Process(pid)
        except psutil.NoSuchProcess:
            _record_job(job_id, status='done', finished_at=time.time(),
                        errors=['subprocess exited between persistence '
                                'load and reattach'])
            return
        # Block until the process exits (no timeout — training can run
        # for hours; the dashboard already had a 1 h cap pre-detach
        # and that's now enforced inside the subprocess wrapper).
        try:
            p.wait()
        except psutil.NoSuchProcess:
            pass
        rc = None
        try:
            rc = p.returncode
        except Exception:
            pass
        # Read the log tail for any error message.
        tail = ''
        log_path = _TRAINING_LOG_DIR / f'{job_id}.log'
        try:
            if log_path.exists():
                sz = log_path.stat().st_size
                with open(log_path, 'rb') as f:
                    f.seek(max(0, sz - 2048))
                    tail = f.read().decode('utf-8', 'replace')[-1500:]
        except Exception:
            pass
        if rc == 0 or rc is None:
            _record_job(job_id, status='done', finished_at=time.time(),
                        successes=1)
        else:
            _record_job(job_id, status='error', finished_at=time.time(),
                        errors=[tail or f'exit code {rc}'])
    except Exception as exc:
        _record_job(job_id, status='error', finished_at=time.time(),
                    errors=[f'{type(exc).__name__}: {exc}'])


def _read_training_log_tail(job_id: str, n_bytes: int = 800) -> str:
    """Return the last n_bytes of the per-job training log file as a
    UTF-8 string. Used to populate error fields on the job record once
    the subprocess exits non-zero (we redirect stderr to this file in
    _spawn_training_subprocess; the old PIPE-based path is gone)."""
    log_path = _TRAINING_LOG_DIR / f'{job_id}.log'
    if not log_path.exists():
        return ''
    try:
        sz = log_path.stat().st_size
        with open(log_path, 'rb') as f:
            f.seek(max(0, sz - n_bytes))
            data = f.read()
        return data.decode('utf-8', 'replace')[-n_bytes:]
    except Exception:
        return ''


def _spawn_training_subprocess(job_id: str, cmd: list) -> "subprocess.Popen":
    """Spawn a trainer subprocess detached from the dashboard's console
    group, with stdout/stderr redirected to a per-job log file. The
    detach is what lets the subprocess survive a dashboard restart.

    Windows: DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP + CREATE_BREAKAWAY_FROM_JOB
    Unix:    start_new_session=True (Python wraps setsid())
    """
    _TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _TRAINING_LOG_DIR / f'{job_id}.log'
    log_fp = open(log_path, 'wb', buffering=0)
    kw: dict = {
        'cwd':    project_root,
        'stdout': log_fp,
        'stderr': log_fp,
        'stdin':  subprocess.DEVNULL,
        'close_fds': True,
    }
    if sys.platform == 'win32':
        # 0x00000008 DETACHED_PROCESS · 0x00000200 CREATE_NEW_PROCESS_GROUP ·
        # 0x01000000 CREATE_BREAKAWAY_FROM_JOB. Together: child is
        # NOT in the dashboard's console group and NOT in any inherited
        # job, so Stop-Process -Force on the dashboard does not
        # propagate to it.
        kw['creationflags'] = (
            getattr(subprocess, 'DETACHED_PROCESS', 0x08)
            | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200)
            | 0x01000000
        )
    else:
        kw['start_new_session'] = True
    return subprocess.Popen(cmd, **kw)


# Reload persisted training-jobs state at module import time, then sweep
# for orphan subprocesses (older dashboard instances or manual runs that
# aren't in our jobs file). Both run in a daemon thread so we don't slow
# startup on disk I/O / psutil enumeration.
def _training_state_recover() -> None:
    try: _load_training_jobs()
    except Exception: pass
    try: _detect_orphan_training_subprocesses()
    except Exception: pass


threading.Thread(target=_training_state_recover, daemon=True,
                 name='training-jobs-reload').start()


# Resource-aware scheduler for trainer subprocesses.
#
# Each model declares whether its training run is CPU-bound (sklearn /
# HistGBT — torch CPU at most for some preprocessors), GPU-bound (Darts
# TFT runs neural net on the single CUDA device), or exclusive (OFT
# joint training saturates both GPU memory AND the CPU dataloader, so
# it must run alone).
#
# Rules:
#   • ≤ AI_TRADER_TRAIN_CONCURRENCY cpu jobs run at once (default 2).
#   • ≤ AI_TRADER_GPU_CONCURRENCY    gpu jobs run at once (default 1).
#   • An exclusive job blocks until cpu_active==0 AND gpu_active==0,
#     then blocks every new acquire until it finishes.
#
# Why this design (not a single Semaphore): on a 32 GB / 1× GPU host,
# regime+TFT can safely overlap (CPU and GPU are disjoint), but OFT+TFT
# would OOM the GPU and throw "CUDA out of memory" partway in. The
# 2026-05-08 incident also showed an undifferentiated cap is fragile —
# 3 concurrent CPU jobs killed the dashboard via Win32 RPC subprocess
# failure under memory pressure, so the CPU cap drops to 2.
import os as _os

_RESOURCE_KIND: dict[str, str] = {
    # CPU-bound (HistGBT / sklearn-style trainers)
    'base':     'cpu',
    'trend':    'cpu',
    'futures':  'cpu',
    'scalping': 'cpu',
    'meta':     'cpu',
    'regime':   'cpu',
    # GPU-bound (single-GPU exclusive among GPU jobs, but CPU jobs may run)
    'tft':      'gpu',
    # Exclusive (must run alone — saturates GPU + heavy CPU dataloader)
    'oft':      'exclusive',
    # `all` sequences every trainer internally → treat as exclusive so
    # nothing else queues alongside it.
    'all':      'exclusive',
}

_TRAINING_CPU_CAP = max(1, int(_os.getenv('AI_TRADER_TRAIN_CONCURRENCY', '2')))
_TRAINING_GPU_CAP = max(1, int(_os.getenv('AI_TRADER_GPU_CONCURRENCY',   '1')))
# Kept for backwards-compat with the older Semaphore API and the test
# suite's static-grep checks.
_TRAINING_CONCURRENCY = _TRAINING_CPU_CAP


class _TrainingScheduler:
    """Resource-aware acquire/release for trainer subprocesses.

    Three counters guarded by one Condition:
      cpu_active     — CPU-bound trainers currently running
      gpu_active     — GPU-bound trainers currently running
      exclusive_busy — True while an `exclusive` trainer is running

    acquire(kind) blocks until the requested kind fits under the rules
    (see module docstring above). release(kind) decrements and broadcasts
    so any waiters can re-evaluate their predicate.
    """

    def __init__(self, cpu_cap: int, gpu_cap: int):
        self._cond = threading.Condition()
        self._cpu_active = 0
        self._gpu_active = 0
        self._exclusive_busy = False
        self.cpu_cap = cpu_cap
        self.gpu_cap = gpu_cap

    def _can_acquire(self, kind: str) -> bool:
        if self._exclusive_busy:
            return False
        if kind == 'exclusive':
            return self._cpu_active == 0 and self._gpu_active == 0
        if kind == 'gpu':
            return self._gpu_active < self.gpu_cap
        # default 'cpu' (or any unknown kind — fail safe to CPU lane)
        return self._cpu_active < self.cpu_cap

    def acquire(self, kind: str) -> None:
        kind = kind if kind in ('cpu', 'gpu', 'exclusive') else 'cpu'
        with self._cond:
            while not self._can_acquire(kind):
                self._cond.wait()
            if kind == 'exclusive':
                self._exclusive_busy = True
            elif kind == 'gpu':
                self._gpu_active += 1
            else:
                self._cpu_active += 1

    def release(self, kind: str) -> None:
        kind = kind if kind in ('cpu', 'gpu', 'exclusive') else 'cpu'
        with self._cond:
            if kind == 'exclusive':
                self._exclusive_busy = False
            elif kind == 'gpu':
                self._gpu_active = max(0, self._gpu_active - 1)
            else:
                self._cpu_active = max(0, self._cpu_active - 1)
            self._cond.notify_all()

    def snapshot(self) -> dict:
        with self._cond:
            return {
                'cpu_active': self._cpu_active,
                'gpu_active': self._gpu_active,
                'exclusive_busy': self._exclusive_busy,
                'cpu_cap': self.cpu_cap,
                'gpu_cap': self.gpu_cap,
            }


_training_scheduler = _TrainingScheduler(
    cpu_cap=_TRAINING_CPU_CAP, gpu_cap=_TRAINING_GPU_CAP,
)


def _resource_kind_for(model_key: str) -> str:
    return _RESOURCE_KIND.get(model_key, 'cpu')


# Legacy alias — some older callers reach for `_training_concurrency_sem`.
# Wrap the new scheduler so they keep working as a CPU-only sem. The Phase
# 60 test asserts both .acquire() and .release() are called, so we forward
# to the scheduler with the CPU lane.
class _LegacySemAdapter:
    def acquire(self) -> None:
        _training_scheduler.acquire('cpu')

    def release(self) -> None:
        _training_scheduler.release('cpu')


_training_concurrency_sem = _LegacySemAdapter()


def _spawn_followup_backtest(parent_job_id: str, model_key: str,
                             tfs: tuple[str, ...]) -> None:
    """After a manual training succeeds, optionally refresh the Stability
    Heatmap data for the touched timeframe(s). run_full_backtest now
    accepts both `timeframes` and `models` filters (PR-46 v3.1 step 3),
    so this followup-backtest scopes to the trained (model, tf) tuple
    only — finishes in <2 min instead of <10 min. Spawns a detached
    subprocess so it survives a dashboard restart, matching the
    trainer's pattern.

    Recorded as a separate job (resource_kind='cpu') so the scheduler
    queues it behind any in-flight training.
    """
    if not tfs:
        return
    # Synthetic job for the backtest leg — operator sees it queued.
    bt_job_id = uuid.uuid4().hex[:12]
    _record_job(bt_job_id, model=f'{model_key}-backtest', status='queued',
                queued_at=time.time(), created_at=time.time(),
                resource_kind='cpu',
                progress_label=f'queued (post-train backtest: {model_key} @ {",".join(tfs)})',
                tf=','.join(tfs), total=1, n=1,
                parent_job_id=parent_job_id)
    threading.Thread(
        target=_run_followup_backtest_blocking,
        args=(bt_job_id, model_key, tfs),
        daemon=True,
        name=f'post-train-bt-{bt_job_id}',
    ).start()


def _run_followup_backtest_blocking(job_id: str, model_key: str,
                                    tfs: tuple[str, ...]) -> None:
    """Worker body for the chained-backtest job. Goes through the same
    scheduler as a CPU-lane training (so it can't fight a TFT GPU run
    or an OFT exclusive run). Spawns a Python subprocess invoking
    run_full_backtest with the trained TFs."""
    _training_scheduler.acquire('cpu')
    try:
        _record_job(job_id, status='running', started_at=time.time(),
                    progress_label=f'post-train backtest: {model_key} @ {",".join(tfs)}')
        tf_arg = ','.join(repr(t) for t in tfs)
        # Per-model filter — only re-run strategies whose underlying ML
        # model matches the just-trained model_key. Cuts backtest time
        # from ~10 min to <2 min for a single-model retrain.
        cmd = [
            sys.executable, '-c',
            'from src.engine.backtester import run_full_backtest; '
            f'run_full_backtest(timeframes=({tf_arg},), models=({model_key!r},))',
        ]
        try:
            proc = _spawn_training_subprocess(job_id, cmd)
        except Exception as exc:
            _record_job(job_id, status='error', finished_at=time.time(),
                        errors=[f'{type(exc).__name__}: {exc}'])
            return
        _record_job(job_id, child_pid=proc.pid)
        # Same poll loop as _run_trainer_blocking. Backtest is bounded
        # by the same 1-hour ceiling.
        deadline = time.time() + 3600
        rc = None
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if time.time() > deadline:
                proc.kill()
                rc = -1
                break
            time.sleep(2)
        finished_at = time.time()
        if rc == 0:
            _record_job(job_id, status='done', finished_at=finished_at)
        else:
            tail = _read_training_log_tail(job_id, 600)
            _record_job(job_id, status='error',
                        finished_at=finished_at,
                        errors=[tail or f'exit code {rc}'])
    finally:
        _training_scheduler.release('cpu')


def _run_trainer_multi_tf(job_id: str, key: str, n: int,
                          tfs: tuple[str, ...],
                          with_backtest: bool = False) -> None:
    """Multi-TF wrapper around _run_trainer_blocking. Trains the model at
    each TF sequentially. Status reflects which TF is currently running so
    the dashboard can show e.g. 'running base @ 4h (2/3 TFs done)'."""
    successes_by_tf: dict[str, int] = {}
    errors_by_tf: dict[str, list[str]] = {}
    cancelled = False
    spec = _TRAINER_DISPATCH.get(key)
    if not spec:
        _record_job(job_id, status='error',
                    error=f'unknown model key {key}',
                    finished_at=time.time())
        return
    module_path, fn_name, accepts_tf = spec
    # Same resource-aware gate as _run_trainer_blocking — queue under the
    # CPU/GPU/exclusive lane appropriate for this model.
    res_kind = _resource_kind_for(key)
    _record_job(job_id, status='queued', queued_at=time.time(),
                total=len(tfs) * n, tfs=list(tfs), tf_done=0, current_tf=None,
                resource_kind=res_kind,
                progress_label=f'queued ({res_kind} lane)')
    _training_scheduler.acquire(res_kind)
    try:
        _record_job(job_id, status='running', started_at=time.time(),
                    progress=0, progress_label=None)
        iter_count = 0
        for tf_idx, tf in enumerate(tfs):
            if cancelled:
                break
            _record_job(job_id, current_tf=tf, status='running',
                        progress_label=f'{key} @ {tf} ({tf_idx+1}/{len(tfs)})')
            for i in range(n):
                if cancelled:
                    break
                # Some trainers don't accept timeframe= (regime, oft, all) —
                # pass nothing for those even if a TF is in the chain.
                kw = f"timeframe={tf!r}" if accepts_tf else ""
                cmd = [
                    sys.executable, '-c',
                    f'import {module_path} as m; '
                    f'fn = getattr(m, {fn_name!r}); fn({kw})',
                ]
                try:
                    proc = _spawn_training_subprocess(job_id, cmd)
                    with _training_active_lock:
                        _training_active_procs[job_id] = proc
                    # Persist child_pid so a dashboard restart can reattach.
                    _record_job(job_id, child_pid=proc.pid)
                    # poll() in a loop instead of communicate() — keeps the
                    # subprocess detached (its stdout/stderr go to the log
                    # file, not our pipe) and lets us bail cleanly on
                    # TimeoutExpired / cancellation without holding the
                    # subprocess hostage to our process tree.
                    deadline = time.time() + 3600
                    while True:
                        rc = proc.poll()
                        if rc is not None:
                            break
                        if time.time() > deadline:
                            proc.kill()
                            errors_by_tf.setdefault(tf, []).append(f'iter {i+1} timed out')
                            rc = -1
                            break
                        time.sleep(2)
                    with _training_active_lock:
                        _training_active_procs.pop(job_id, None)
                    if rc == 0:
                        successes_by_tf[tf] = successes_by_tf.get(tf, 0) + 1
                    elif rc == -9 or rc == 1:
                        cancelled = True
                        errors_by_tf.setdefault(tf, []).append(f'iter {i+1} cancelled')
                    elif rc is not None and rc != 0:
                        # Read tail of log file for the error message.
                        msg = _read_training_log_tail(job_id, 600)
                        errors_by_tf.setdefault(tf, []).append(msg)
                except Exception as exc:
                    errors_by_tf.setdefault(tf, []).append(f'{type(exc).__name__}: {exc}')
                    with _training_active_lock:
                        _training_active_procs.pop(job_id, None)
                iter_count += 1
                _record_job(job_id, progress=iter_count, tf_done=tf_idx)
            _record_job(job_id, tf_done=tf_idx + 1)
        total_succ = sum(successes_by_tf.values())
        total_err = sum(len(v) for v in errors_by_tf.values())
        final_status = ('cancelled' if cancelled else
                        'done' if total_err == 0 else
                        'partial' if total_succ > 0 else 'error')
        finished_at = time.time()
        _record_job(job_id, status=final_status,
                    successes=total_succ,
                    successes_by_tf=successes_by_tf,
                    errors_by_tf={k: v[-2:] for k, v in errors_by_tf.items()},
                    current_tf=None,
                    finished_at=finished_at)
        # Update the per-key rolling-average duration so future ETA
        # estimates reflect this hardware. Only record successes — failed
        # / cancelled runs are noisy datapoints that would skew the avg.
        with _training_jobs_lock:
            entry = _training_jobs.get(job_id) or {}
        started_at = entry.get('started_at') or 0
        if final_status == 'done' and started_at > 0:
            _record_completed_duration(key, finished_at - started_at)
        with _training_active_lock:
            _training_active_procs.pop(job_id, None)
        # P-2: chain a per-TF backtest after a fully-successful multi-TF
        # train so the Stability Heatmap data refreshes for the touched
        # TFs. Only run when the operator explicitly opted in via
        # with_backtest=true on the API call. Failed runs skip this —
        # backtesting a broken model wastes CPU.
        if with_backtest and final_status == 'done':
            tfs_done = tuple(t for t, c in successes_by_tf.items() if c > 0)
            if tfs_done:
                _spawn_followup_backtest(job_id, key, tfs_done)
    finally:
        _training_scheduler.release(res_kind)


def _run_trainer_blocking(job_id: str, key: str, n: int,
                          tf: str | None = None,
                          with_backtest: bool = False) -> None:
    """Worker thread body: invoke the matching trainer N times sequentially.
    tf — optional per-TF override (defaults to the trainer's own default).
    Used for manual per-TF training from the dashboard.

    with_backtest — if True and the run finishes 'done', chain a
    run_full_backtest(timeframes=(tf,)) subprocess so the Stability
    Heatmap refreshes for the trained TF.

    Each iteration is launched as a Popen we track in
    _training_active_procs so /api/training/stop/<job_id> can kill it.
    Falls through to next iteration on stop (treated as 'cancelled').
    """
    spec = _TRAINER_DISPATCH.get(key)
    if not spec:
        _record_job(job_id, status='error',
                    error=f'unknown model key {key}',
                    finished_at=time.time())
        return
    module_path, fn_name, accepts_tf = spec
    successes, errors = 0, []
    cancelled = False
    # Wait for a resource slot. The scheduler enforces CPU/GPU/exclusive
    # rules — see _TrainingScheduler. Mark the job as 'queued' while
    # waiting so the dashboard pill shows what's blocking.
    res_kind = _resource_kind_for(key)
    _record_job(job_id, status='queued', queued_at=time.time(),
                total=n, tf=tf, resource_kind=res_kind,
                progress_label=f'queued ({res_kind} lane)')
    _training_scheduler.acquire(res_kind)
    try:
        _record_job(job_id, status='running', started_at=time.time(),
                    progress=0, progress_label=None)
        for i in range(n):
            if cancelled:
                break
            # Skip the timeframe kwarg entirely for trainers that don't
            # accept it (regime / oft / all), even if the UI passed a TF.
            kw = f"timeframe={tf!r}" if (tf and accepts_tf) else ""
            cmd = [
                sys.executable, '-c',
                f'import {module_path} as m; '
                f'fn = getattr(m, {fn_name!r}); fn({kw})',
            ]
            proc = None
            try:
                proc = _spawn_training_subprocess(job_id, cmd)
                with _training_active_lock:
                    _training_active_procs[job_id] = proc
                _record_job(job_id, child_pid=proc.pid)
                # Same poll-loop pattern as _run_trainer_multi_tf — see
                # the comment there for why this replaced communicate().
                deadline = time.time() + 3600
                rc = None
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        break
                    if time.time() > deadline:
                        proc.kill()
                        errors.append(f'iteration {i+1} timed out (>1h)')
                        rc = -1
                        break
                    time.sleep(2)
                with _training_active_lock:
                    _training_active_procs.pop(job_id, None)
                if rc == 0:
                    successes += 1
                elif rc == -9:
                    cancelled = True
                    errors.append(f'iteration {i+1} cancelled by operator')
                elif rc is not None and rc != 0:
                    tail = _read_training_log_tail(job_id, 800)
                    if 'killed' in tail.lower():
                        cancelled = True
                        errors.append(f'iteration {i+1} cancelled by operator')
                    else:
                        errors.append(tail)
            except Exception as exc:
                errors.append(f'iteration {i+1} crashed: {type(exc).__name__}: {exc}')
                with _training_active_lock:
                    _training_active_procs.pop(job_id, None)
            _record_job(job_id, progress=i + 1)
        final_status = ('cancelled' if cancelled else
                        'done' if not errors else
                        'partial' if successes else 'error')
        finished_at = time.time()
        _record_job(job_id, status=final_status,
                    successes=successes, errors=errors[-3:],
                    finished_at=finished_at)
        # Same rolling-average update as multi-tf path; only record
        # successful single-iteration runs.
        with _training_jobs_lock:
            entry = _training_jobs.get(job_id) or {}
        started_at = entry.get('started_at') or 0
        if final_status == 'done' and started_at > 0:
            _record_completed_duration(key, finished_at - started_at)
        with _training_active_lock:
            _training_active_procs.pop(job_id, None)
        # P-2: chain a per-TF backtest if the operator opted in.
        if with_backtest and final_status == 'done':
            # tf may be None — use the model's canonical TF in that case.
            from src.engine.train_all_models import DEFAULT_PER_KEY_TFS
            bt_tf = tf or (DEFAULT_PER_KEY_TFS.get(key, ('1h',)) or ('1h',))[0]
            _spawn_followup_backtest(job_id, key, (bt_tf,))
    finally:
        _training_scheduler.release(res_kind)


@app.route('/api/training/run/<key>', methods=['POST'])
@require_api_key
def api_training_run_one(key: str):
    """Kick off a training run for one model key.
    Body / query params:
      n             — repetitions (default 1, clamped 1..20)
      tf            — optional timeframe (5m, 15m, 1h, 4h, 1d, 1w, 1mo).
                      Default is the trainer's own default. Passed
                      straight through as the timeframe= kwarg, so
                      'base @ 4h' writes models/base_4h_*.
      with_backtest — bool. When true, after the training finishes
                      'done', spawn run_full_backtest(timeframes=(tf,))
                      so the Stability Heatmap row for that TF refreshes
                      without requiring a full pipeline orchestrator run.
    """
    if key not in _TRAINER_DISPATCH:
        return jsonify({'ok': False,
                        'error': f'unknown model key: {key}',
                        'valid': sorted(_TRAINER_DISPATCH.keys())}), 400
    body = request.get_json(silent=True) or {}
    try:
        n = int(body.get('n') or request.args.get('n') or 1)
    except (TypeError, ValueError):
        n = 1
    n = max(1, min(n, 20))   # clamp so a stray 1000 doesn't pin a CPU all day
    tf = body.get('tf') or request.args.get('tf') or None
    with_backtest = bool(body.get('with_backtest'))
    # Duplicate-job guard. If this same model is already training, return
    # 409 with the existing job_id unless the caller explicitly opted in
    # via {"force":true}. Operator's UI shows a confirm dialog when 409
    # comes back so they can pick 'retrain anyway'.
    force = bool(body.get('force'))
    if not force:
        with _training_jobs_lock:
            for j in _training_jobs.values():
                if (j.get('model') == key
                        and j.get('status') in ('queued', 'running', 'starting')):
                    return jsonify({
                        'ok': False,
                        'error': 'already_running',
                        'message': f'{key} is already training (job {j.get("job_id")}, status={j.get("status")})',
                        'existing_job_id': j.get('job_id'),
                        'existing_status':  j.get('status'),
                    }), 409
    # 'all' expands to every TF the trainer supports for this model — we
    # spawn one job per TF and let the worker chain them sequentially.
    valid_tfs = ('1m', '5m', '15m', '1h', '4h', '1d', '1w', '1mo')
    if tf == 'all':
        # Default per-key TF set mirrors src.engine.train_all_models.
        ALL_TFS_BY_KEY = {
            'base':     ('1h', '4h', '1d'),
            'trend':    ('1h', '4h', '1d'),
            'futures':  ('1h', '4h', '1d'),
            'scalping': ('1m',),
            'meta':     ('1h',),
            'tft':      ('1h',),
            'oft':      ('1m',),
            'regime':   ('1h',),
        }
        tfs = ALL_TFS_BY_KEY.get(key, ('1h',))
        # Single job that loops the TFs internally for clean status reporting.
        job_id = uuid.uuid4().hex[:12]
        _record_job(job_id, model=key, n=n, tf='all', tfs=list(tfs),
                    with_backtest=with_backtest,
                    status='queued', created_at=time.time())
        threading.Thread(
            target=_run_trainer_multi_tf,
            args=(job_id, key, n, tfs),
            kwargs={'with_backtest': with_backtest},
            daemon=True, name=f'train-{key}-all-{job_id}',
        ).start()
        return jsonify({'ok': True, 'job_id': job_id, 'model': key,
                        'n': n, 'tf': 'all', 'tfs': list(tfs),
                        'with_backtest': with_backtest})
    if tf and tf not in valid_tfs:
        return jsonify({'ok': False, 'error': f'invalid tf: {tf}'}), 400
    job_id = uuid.uuid4().hex[:12]
    _record_job(job_id, model=key, n=n, tf=tf,
                with_backtest=with_backtest,
                status='queued', created_at=time.time())
    threading.Thread(
        target=_run_trainer_blocking,
        args=(job_id, key, n, tf),
        kwargs={'with_backtest': with_backtest},
        daemon=True, name=f'train-{key}-{tf or "default"}-{job_id}',
    ).start()
    return jsonify({'ok': True, 'job_id': job_id, 'model': key, 'n': n, 'tf': tf,
                    'with_backtest': with_backtest})


@app.route('/api/training/stop/<job_id>', methods=['POST'])
@require_api_key
def api_training_stop(job_id: str):
    """Kill the subprocess backing this training job. Returns ok=true if
    we either killed something or the job was already finished. Returns
    404 only if the job_id was never seen."""
    with _training_jobs_lock:
        rec = _training_jobs.get(job_id)
    if not rec:
        return jsonify({'ok': False, 'error': f'unknown job_id {job_id}'}), 404
    with _training_active_lock:
        proc = _training_active_procs.pop(job_id, None)
    if proc is None:
        return jsonify({'ok': True, 'message': 'job already finished',
                        'status': rec.get('status')})
    try:
        proc.kill()
        return jsonify({'ok': True, 'message': 'killed', 'job_id': job_id})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/training/active', methods=['GET'])
def api_training_active():
    """Return a {model_key: job_id} map of currently-active training jobs.
    The Strategy/ML row JS uses this to decide whether to render Train or
    Stop on each row."""
    out: dict[str, str] = {}
    with _training_jobs_lock:
        rows = list(_training_jobs.values())
    for r in rows:
        if r.get('status') in ('running', 'queued'):
            out[r.get('model', '')] = r.get('job_id', '')
    return jsonify({'active': out})


@app.route('/api/training/scheduler', methods=['GET'])
def api_training_scheduler():
    """Expose the resource-aware scheduler snapshot (cpu/gpu counters +
    caps + exclusive flag) plus the per-model resource kind. Used by the
    UI to render lane occupancy and explain why a job is queued."""
    return jsonify({
        'snapshot':      _training_scheduler.snapshot(),
        'resource_kind': dict(_RESOURCE_KIND),
    })


@app.route('/api/pipeline/reset', methods=['POST'])
@require_api_key
def api_pipeline_reset():
    """Clear stale pipeline status. Refuses to clear if a process is
    actually alive — the operator's expectation should match disk."""
    if _pipeline_proc_alive():
        return jsonify({'ok': False,
                        'error': 'orchestrator process is still alive — refuse to clear status',
                        'pid': _pipeline_proc_pid}), 409
    from src.utils.safe_json import write_json
    try:
        write_json(_pipeline_status_path(), {})
        return jsonify({'ok': True, 'message': 'pipeline status cleared'})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/training/run/all', methods=['POST'])
@require_api_key
def api_training_run_all():
    """Run train_all_models.py once — the canonical full-pipeline retrain.

    Refuses to spawn a duplicate when ANY training subprocess (single
    model OR another 'all' run) is currently active — pre-fix the user
    clicked Retrain ALL, saw nothing happen, and kept clicking, ending
    up with 10 train_all subprocesses competing for CPU/GPU."""
    body = request.get_json(silent=True) or {}
    force = bool(body.get('force'))
    if not force:
        with _training_jobs_lock:
            for j in _training_jobs.values():
                if j.get('status') in ('queued', 'running', 'starting'):
                    return jsonify({
                        'ok': False,
                        'error': 'already_running',
                        'message': f"a training job is already in flight (model={j.get('model')}, status={j.get('status')}). Pass force=true to spawn a parallel run anyway.",
                        'existing_job_id': j.get('job_id'),
                        'existing_model':  j.get('model'),
                        'existing_status': j.get('status'),
                    }), 409
    job_id = uuid.uuid4().hex[:12]
    _record_job(job_id, model='all', n=1,
                status='queued', created_at=time.time())
    threading.Thread(
        target=_run_trainer_blocking,
        args=(job_id, 'all', 1),
        daemon=True, name=f'train-all-{job_id}',
    ).start()
    return jsonify({'ok': True, 'job_id': job_id, 'model': 'all', 'n': 1})


# Typical training durations (seconds) per model key. Seeded from the
# 2026-05-08 measured run set (most numbers are wall-clock from runs that
# ran with 0-2 other trainings sharing CPU/GPU). The map auto-updates
# below as jobs finish — running average over the last 5 completions per
# key, so the operator's UI estimate self-corrects on this hardware.
_TYPICAL_DURATIONS: dict[str, float] = {
    'regime':   30 * 60,        # RF on 776K samples
    'base':     30 * 60,        # HistGBT on 635K samples (writes btc_rf_*)
    'trend':    27 * 60,        # HistGBT on 615K
    'futures':  25 * 60,        # HistGBT on 504K
    'scalping': 30 * 60,        # HistGBT on 4.5M (1m candles)
    'meta':     12 * 60,        # meta-labeler on 1.14M
    'tft':      60 * 60,        # Darts neural, GPU
    'oft':     140 * 60,        # joint OFT, 5-fold purged kfold (May 4 run)
    'all':     180 * 60,        # train_all_models sequential
}
_TYPICAL_HISTORY: dict[str, list[float]] = {}
_TYPICAL_LOCK = threading.Lock()

# v3.1 step 15 (5A) — load persisted self-tuned values from data/cache/cold/
# at import time. The rolling-avg map self-corrects as jobs finish; without
# this restore, every dashboard restart resets to the hand-seeded numbers
# and the operator's ETA accuracy drops for several training cycles.
try:
    from src.dashboard import cold_cache as _cold_cache
    _persisted_typ = _cold_cache.load('typical_durations', default=None,
                                       max_age_s=30 * 86400)
    if isinstance(_persisted_typ, dict):
        for _k, _v in _persisted_typ.items():
            try:
                _v = float(_v)
                if 1 < _v < 6 * 3600:
                    _TYPICAL_DURATIONS[_k] = _v
            except (TypeError, ValueError):
                continue
    # Persisted history (so the rolling avg keeps shape, not just current value).
    _persisted_hist = _cold_cache.load('typical_history', default=None,
                                        max_age_s=30 * 86400)
    if isinstance(_persisted_hist, dict):
        for _k, _vlist in _persisted_hist.items():
            if isinstance(_vlist, list):
                _TYPICAL_HISTORY[_k] = [float(x) for x in _vlist if isinstance(x, (int, float))][-5:]
except Exception:
    pass


def _record_completed_duration(key: str, duration_s: float) -> None:
    """Update _TYPICAL_DURATIONS with a rolling average of recent runs.
    Persists to data/cache/cold/typical_durations.json + _history.json on
    each update so a dashboard restart doesn't reset the self-tuned values
    (v3.1 step 15)."""
    if not key or duration_s <= 0 or duration_s > 6 * 3600:
        return
    with _TYPICAL_LOCK:
        hist = _TYPICAL_HISTORY.setdefault(key, [])
        hist.append(duration_s)
        if len(hist) > 5:
            del hist[0]
        _TYPICAL_DURATIONS[key] = sum(hist) / len(hist)
        # Cold-cache snapshot — best-effort, never blocks the caller.
        try:
            from src.dashboard import cold_cache as _cc
            _cc.save('typical_durations', dict(_TYPICAL_DURATIONS))
            _cc.save('typical_history', {k: list(v) for k, v in _TYPICAL_HISTORY.items()})
        except Exception:
            pass


def _annotate_job_timing(j: dict) -> dict:
    """Decorate a job dict with elapsed_s / eta_s fields. Pure read; the
    underlying job entry is not mutated. ETA uses the typical duration
    map; if a job is past the typical estimate, eta is reported as 0
    (rather than negative) so the UI shows '… overdue' without confusing
    arithmetic."""
    out = dict(j)
    started_at = j.get('started_at') or 0
    queued_at  = j.get('queued_at')  or 0
    finished   = j.get('finished_at')
    status     = j.get('status', '')
    now = time.time()
    if status in ('running', 'queued') and not finished:
        if status == 'running' and started_at > 0:
            elapsed = max(0.0, now - started_at)
            out['elapsed_s'] = round(elapsed, 1)
            typ = _TYPICAL_DURATIONS.get(j.get('model', ''))
            if typ:
                out['eta_s'] = round(max(0.0, typ - elapsed), 1)
                out['typical_s'] = round(typ, 1)
        elif status == 'queued' and queued_at > 0:
            out['queued_for_s'] = round(max(0.0, now - queued_at), 1)
            typ = _TYPICAL_DURATIONS.get(j.get('model', ''))
            if typ:
                out['typical_s'] = round(typ, 1)
    elif finished and started_at > 0:
        out['elapsed_s'] = round(max(0.0, finished - started_at), 1)
    return out


# ─── v4 Phase B5 — training rules registry HTTP API ───────────────────────────
# Operator-facing endpoints so the dashboard's TRAINING & BACKTEST card can
# read + edit data/training_rules.json without code changes. Three endpoints:
#   GET  /api/training/rules            → current rules content + matrix
#   POST /api/training/rules            → save edited rules (validated)
#   GET  /api/training/preview          → planned (model, tf) list for
#                                          a hypothetical sweep, with ETA

@app.route('/api/training/rules', methods=['GET'])
def api_training_rules():
    """Return the canonical training rules registry + a flat matrix the
    UI can render directly. The UI calls this on load + after any save."""
    try:
        from src.training import training_rules as _r
        _r.reload()
        # Build a matrix for the UI: per (model, tf) tuple, status + reason
        # so cells render with accurate tooltips even on skip/experimental.
        cells = []
        for model in _r.all_models():
            for tf in _r.TF_ORDER:
                status = _r.cell_status(model, tf)
                cells.append({
                    'model':   model,
                    'tf':      tf,
                    'status':  status,
                    'reason':  _r.skip_reason(model, tf) if status == 'skip' else '',
                })
        from pathlib import Path as _P
        raw = _P(PROJECT_ROOT) / 'data' / 'training_rules.json'
        rules_obj = json.loads(raw.read_text(encoding='utf-8')) if raw.exists() else {}
        return jsonify({
            'ok':         True,
            'tf_order':   list(_r.TF_ORDER),
            'models':     _r.all_models(),
            'matrix':     cells,
            'rules':      rules_obj,
            'sweep_eta':  {
                'default_combo_count':       len(_r.planned_combos()),
                'with_experimental_count':   len(_r.planned_combos(include_experimental=True)),
                'sequential_minutes':        _r.estimated_total_minutes(_r.planned_combos()),
                'parallel_2_workers_min':    _r.estimated_parallel_minutes(_r.planned_combos(), 2),
                'parallel_4_workers_min':    _r.estimated_parallel_minutes(_r.planned_combos(), 4),
            },
        })
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), 500


@app.route('/api/training/rules', methods=['POST'])
@require_api_key
def api_training_rules_save():
    """Persist edited rules atomically. Body: full rules JSON object
    (same shape as data/training_rules.json). Validates required keys
    + that every model has an applicable_tfs / experimental_tfs / skip_tfs
    list before writing — bad input doesn't corrupt the rules file."""
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict) or 'models' not in body:
        return jsonify({'ok': False, 'error': 'expected {"models": {...}, ...}'}), 400
    models = body.get('models') or {}
    if not isinstance(models, dict) or not models:
        return jsonify({'ok': False, 'error': 'models must be a non-empty object'}), 400
    for k, blk in models.items():
        for req_field in ('applicable_tfs', 'experimental_tfs', 'skip_tfs'):
            if req_field not in blk or not isinstance(blk[req_field], list):
                return jsonify({'ok': False,
                                'error': f'model {k!r} missing list field {req_field!r}'}), 400
        if 'resource_kind' not in blk or blk['resource_kind'] not in ('cpu', 'gpu', 'exclusive'):
            return jsonify({'ok': False,
                            'error': f'model {k!r} resource_kind must be cpu/gpu/exclusive'}), 400
    body['_updated_at'] = datetime.now(_tz.utc).isoformat()
    from pathlib import Path as _P
    raw = _P(PROJECT_ROOT) / 'data' / 'training_rules.json'
    tmp = raw.with_suffix('.tmp')
    tmp.write_text(json.dumps(body, indent=2), encoding='utf-8')
    os.replace(tmp, raw)
    # Reload the in-process cache so the next /api/training/rules GET
    # returns the freshly-saved data without a process restart.
    try:
        from src.training import training_rules as _r
        _r.reload()
    except Exception:
        pass
    return jsonify({'ok': True, 'updated_at': body['_updated_at']})


@app.route('/api/cluster/sweep', methods=['POST'])
@require_api_key
def api_cluster_sweep():
    """v4 Phase B5 — submit a rules-aware sweep to the local cluster.

    Reads training_rules.json, applies operator overrides from the request
    body, expands the planned_combos list, and POSTs each (model, tf)
    combo as a cluster task to localhost:7700.

    Body (all optional):
        force_train:           [["base","1w"], ["scalping","15m"], ...]
        force_skip:            [["tft","4h"], ...]
        include_experimental:  bool (default False)
        symbols:               list — overrides the default symbol universe
                               for THIS sweep only

    Returns: {ok, sweep_id, task_ids, count, eta_min}
    """
    body = request.get_json(silent=True) or {}
    ft  = body.get('force_train') or []
    fs  = body.get('force_skip') or []
    inc = bool(body.get('include_experimental', False))
    syms_override = body.get('symbols')

    try:
        from src.training import training_rules as _r
        _r.reload()
        plan = _r.planned_combos(force_train=ft, force_skip=fs,
                                  include_experimental=inc)
        symbols = syms_override or _r.symbols()
        if not plan:
            return jsonify({'ok': False, 'error': 'plan is empty after overrides'}), 400
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'rules-load: {type(exc).__name__}: {exc}'}), 500

    sweep_id = uuid.uuid4().hex[:12]
    submitted: list[str] = []
    failed: list[dict] = []
    cluster_url = os.getenv('AI_TRADER_CLUSTER_URL', 'http://192.168.0.105:7700')

    import urllib.request as _req
    for (model_key, tf) in plan:
        # Per-symbol fan-out only for OFT (tick-level microstructure). All
        # other trainers iterate symbols internally, so one task covers
        # the whole universe.
        if model_key == 'oft':
            for sym in symbols:
                spec = {
                    'model_type': model_key, 'timeframe': tf, 'symbol': sym,
                    'data_path': '', 'output_path': '',
                    'config': {'use_master_trainer': True, 'sweep_id': sweep_id},
                }
                _try_submit(cluster_url, spec, submitted, failed)
        else:
            spec = {
                'model_type': model_key, 'timeframe': tf, 'symbol': 'ALL',
                'data_path': '', 'output_path': '',
                'config': {'use_master_trainer': True, 'sweep_id': sweep_id,
                           'symbols': symbols},
            }
            _try_submit(cluster_url, spec, submitted, failed)

    eta_par2 = _r.estimated_parallel_minutes(plan, 2)
    return jsonify({
        'ok':         True,
        'sweep_id':   sweep_id,
        'task_ids':   submitted,
        'count':      len(submitted),
        'failed':     failed,
        'eta_min':    eta_par2,
        'plan':       plan,
        'symbols':    symbols,
    })


def _try_submit(cluster_url: str, spec: dict, submitted: list, failed: list) -> None:
    """Helper for api_cluster_sweep — POSTs one task spec, appends to
    submitted list on success, failed list on error."""
    try:
        body = json.dumps(spec).encode('utf-8')
        import urllib.request as _req
        rq = _req.Request(f'{cluster_url}/api/cluster/submit', data=body,
                          method='POST',
                          headers={'Content-Type': 'application/json'})
        with _req.urlopen(rq, timeout=10) as r:
            d = json.loads(r.read().decode('utf-8'))
            if d.get('ok'):
                submitted.append(d['task_id'])
            else:
                failed.append({'spec': spec, 'error': d.get('error', 'unknown')})
    except Exception as exc:
        failed.append({'spec': spec, 'error': f'{type(exc).__name__}: {exc}'})


@app.route('/api/training/preview', methods=['GET'])
def api_training_preview():
    """Preview which (model, tf) combos a sweep would actually run, given
    operator overrides. Used by the UI's "Run Sweep" button to show
    operator the plan before submission. Query params:
       include_experimental=true|false
       force_train=base:1w,trend:1m   (comma list of model:tf)
       force_skip=tft:4h
    """
    inc_exp = request.args.get('include_experimental', 'false').lower() in ('1', 'true', 'yes')
    def _parse(arg: str) -> list:
        out = []
        for tok in arg.split(',') if arg else []:
            tok = tok.strip()
            if ':' in tok:
                m, t = tok.split(':', 1)
                out.append([m.strip(), t.strip()])
        return out
    ft = _parse(request.args.get('force_train', ''))
    fs = _parse(request.args.get('force_skip', ''))
    try:
        from src.training import training_rules as _r
        _r.reload()
        plan = _r.planned_combos(force_train=ft, force_skip=fs,
                                  include_experimental=inc_exp)
        return jsonify({
            'ok':              True,
            'plan':            plan,
            'count':           len(plan),
            'sequential_min':  _r.estimated_total_minutes(plan),
            'parallel_min': {
                str(n): _r.estimated_parallel_minutes(plan, n) for n in (1, 2, 4)
            },
        })
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), 500


@app.route('/api/training/jobs', methods=['GET'])
def api_training_jobs():
    """Most-recent N training jobs, newest first. Each row carries
    elapsed_s + eta_s when running so the UI can show "3m 12s · ~7m left"
    without re-deriving the arithmetic on the client."""
    try:
        limit = int(request.args.get('limit', 20))
    except (TypeError, ValueError):
        limit = 20
    with _training_jobs_lock:
        rows = list(_training_jobs.values())
    rows.sort(key=lambda x: x.get('created_at', 0), reverse=True)
    annotated = [_annotate_job_timing(r) for r in rows[:limit]]
    return jsonify({'jobs': annotated, 'total': len(rows)})


# ─── Data coverage + 1s→higher-TF resample endpoints ────────────────────────
# Operator-triggered backfill of missing timeframes. We resample from the
# canonical 1s archives in data/raw/historical/ rather than re-downloading
# from Binance, which would take hours and hit rate limits.

_resample_jobs: dict[str, dict] = {}
_resample_jobs_lock = threading.Lock()
_RESAMPLE_JOBS_MAX = 10


def _record_resample_job(job_id: str, **fields) -> None:
    with _resample_jobs_lock:
        e = _resample_jobs.get(job_id) or {'job_id': job_id}
        e.update(fields)
        _resample_jobs[job_id] = e
        if len(_resample_jobs) > _RESAMPLE_JOBS_MAX:
            oldest = min(_resample_jobs.values(),
                         key=lambda x: x.get('created_at', 0))
            _resample_jobs.pop(oldest['job_id'], None)


def _run_resample_blocking(job_id: str, symbols: list[str],
                           timeframes: list[str]) -> None:
    """Worker body: spawn the resampler as a SUBPROCESS per symbol so a
    pandas memory blowup or runaway can't take the Flask process down with
    it (which it did the first time we ran this in-thread). The supervisor
    thread here just polls the subprocess's stderr line-by-line — each
    progress JSON line lands in the job record for the dashboard to read.
    """
    import json as _j
    total = len(symbols)
    _record_resample_job(job_id, status='running',
                         started_at=time.time(),
                         total_symbols=total,
                         done_symbols=0,
                         current_symbol=None,
                         results={})
    results: dict = {}
    for i, sym in enumerate(symbols):
        _record_resample_job(job_id, current_symbol=sym, done_symbols=i)
        cmd = [
            sys.executable, '-m', 'src.utils.resample_ohlcv',
            '--symbol', sym,
            '--timeframes', ','.join(timeframes),
        ]
        try:
            proc = subprocess.Popen(
                cmd, cwd=project_root,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
            # Drain stderr (where the resampler writes one JSON line per event)
            # so the buffer never fills and the child can't deadlock on its
            # own progress reporting. The last successful event is recorded
            # to the job dict for the dashboard pill.
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = _j.loads(line)
                    if isinstance(ev, dict) and ev.get('phase'):
                        _record_resample_job(job_id, last_event=ev)
                except Exception:
                    # Non-JSON stderr (warnings, tracebacks) — ignore
                    pass
            stdout, _ = proc.communicate(timeout=60)
            if proc.returncode == 0:
                try:
                    results[sym] = _j.loads(stdout) if stdout else {}
                except Exception:
                    results[sym] = {'_warning': 'no-stdout-json'}
            else:
                results[sym] = {'_error': f'exit={proc.returncode}'}
        except subprocess.TimeoutExpired:
            results[sym] = {'_error': 'timeout (>1h per symbol)'}
        except Exception as exc:
            results[sym] = {'_error': f'{type(exc).__name__}: {exc}'}
    _record_resample_job(job_id,
                         status='done',
                         done_symbols=total,
                         current_symbol=None,
                         finished_at=time.time(),
                         results=results)


@app.route('/api/data/archive_topup', methods=['GET'])
def api_data_archive_topup():
    """Snapshot of the 1s archive top-up — file sizes + last-modified
    times for every <sym>_spot_1s.csv.gz in data/raw/historical/.
    Lets the operator see download progress without tailing logs."""
    import os as _os
    from datetime import datetime, timezone
    out = []
    from pathlib import Path as _P
    hist_dir = _P(project_root) / 'data' / 'raw' / 'historical'
    if not hist_dir.exists():
        return jsonify({'files': [], 'total_size_gb': 0.0,
                        'message': f'no {hist_dir} dir'})
    total = 0
    now = time.time()
    for p in sorted(hist_dir.glob('*_spot_1s.csv.gz')):
        try:
            st = p.stat()
            sym = p.name.replace('_spot_1s.csv.gz', '')
            out.append({
                'symbol': sym,
                'path':   str(p),
                'size_mb': round(st.st_size / (1024*1024), 1),
                'modified_age_s': round(now - st.st_mtime, 1),
                'modified_iso': datetime.fromtimestamp(
                    st.st_mtime, tz=timezone.utc).isoformat(),
                'recent_activity': (now - st.st_mtime) < 600,
            })
            total += st.st_size
        except OSError:
            pass
    return jsonify({
        'files': out,
        'total_size_gb': round(total / (1024**3), 2),
        'recent_activity_count': sum(1 for f in out if f.get('recent_activity')),
    })


_data_coverage_cache: dict = {'value': None, 'ts': 0.0}
_data_coverage_lock = threading.Lock()
_DATA_COVERAGE_TTL = 120.0


def _refresh_data_coverage_async() -> None:
    """Background-thread refresh of the data-coverage matrix.
    audit_coverage scans every (symbol × tf) gzip CSV's first + last
    line; with BTC 1m at 211MB compressed × 8 TFs × 20 symbols the
    cold-path took 30s+. Cache it on a 2-min TTL so the dashboard's
    Data Coverage panel never blocks the operator."""
    def _job():
        try:
            from src.utils.data_audit import (
                audit_coverage, audit_summary, audit_sentiment,
                discover_symbols, FALLBACK_SYMBOLS, DEFAULT_TIMEFRAMES,
            )
            symbols = discover_symbols()
            rows = audit_coverage(symbols=symbols)
            value = {
                'symbols':    list(symbols),
                'timeframes': list(DEFAULT_TIMEFRAMES),
                'rows':       rows,
                'summary':    audit_summary(rows),
                'sentiment':  audit_sentiment(),
                'discovery': {
                    'discovered_count': len(symbols),
                    'fallback_count':   len(FALLBACK_SYMBOLS),
                    'using_fallback':   set(symbols) == set(FALLBACK_SYMBOLS)
                                        and len(symbols) == len(FALLBACK_SYMBOLS),
                },
            }
            with _data_coverage_lock:
                _data_coverage_cache['value'] = value
                _data_coverage_cache['ts'] = time.time()
        except Exception as exc:
            with _data_coverage_lock:
                _data_coverage_cache['value'] = {'error': f'{type(exc).__name__}: {exc}'}
                _data_coverage_cache['ts'] = time.time()
    threading.Thread(target=_job, daemon=True, name='data-coverage-refresh').start()


@app.route('/api/data/coverage', methods=['GET'])
def api_data_coverage():
    """Return the (symbol × timeframe) coverage matrix.
    TTL-cached (2 min) on a background thread so a slow gzip walk
    (200+ MB files at 1m) never blocks the dashboard's poll cadence."""
    with _data_coverage_lock:
        cached = _data_coverage_cache.get('value')
        cache_age = time.time() - (_data_coverage_cache.get('ts') or 0)
    if cached is None or cache_age > _DATA_COVERAGE_TTL:
        _refresh_data_coverage_async()
    if cached is None:
        return jsonify({
            'symbols':    [], 'timeframes': [], 'rows': [],
            'summary':    {},  'sentiment':  {},
            'cache_warming': True,
        })
    if isinstance(cached, dict):
        cached = {**cached, 'cache_age_s': round(cache_age, 1)}
    return jsonify(cached)


@app.route('/api/data/resample', methods=['POST'])
@require_api_key
def api_data_resample():
    """Kick off a 1s→higher-TF resample. Body: {symbols: [...], timeframes: [...]}.
    Symbols default is auto-discovered from data/raw/historical/<sym>_spot_1s.csv.gz
    (drop new archives there to extend coverage with no code change).
    Timeframes default to (5m,15m,1h,4h,1d,1w,1mo)."""
    from src.utils.data_audit import discover_symbols
    from src.utils.resample_ohlcv import DEFAULT_TIMEFRAMES
    body = request.get_json(silent=True) or {}
    symbols = body.get('symbols') or list(discover_symbols())
    timeframes = body.get('timeframes') or list(DEFAULT_TIMEFRAMES)
    job_id = uuid.uuid4().hex[:12]
    _record_resample_job(job_id,
                         status='queued',
                         created_at=time.time(),
                         symbols=symbols,
                         timeframes=timeframes,
                         total_symbols=len(symbols),
                         done_symbols=0)
    threading.Thread(
        target=_run_resample_blocking,
        args=(job_id, symbols, timeframes),
        daemon=True, name=f'resample-{job_id}',
    ).start()
    return jsonify({'ok': True, 'job_id': job_id,
                    'symbols': symbols, 'timeframes': timeframes})


@app.route('/api/data/backfill', methods=['POST'])
@require_api_key
def api_data_backfill():
    """v3.1 step 14 (1J) — chain: archive top-up → resample for any
    symbols flagged stale or missing.

    Body (all optional): {
      symbols: ['BTC/USDT', ...],   # default: every stale 1s archive
      since:   '2024-12-31',         # default: today − 14 days
    }

    The job runs as a detached subprocess so dashboard restarts don't
    interrupt it. Status surfaces via the existing
    /api/data/resample/jobs poller (re-using job_id).
    """
    body = request.get_json(silent=True) or {}
    symbols = body.get('symbols') or []
    since   = body.get('since') or ''

    if not symbols:
        # Auto-discover stale symbols from the 1s archive mtime.
        try:
            from pathlib import Path as _P
            from datetime import datetime as _dt, timezone as _tz
            base = _P(PROJECT_ROOT) / 'data' / 'raw' / 'historical'
            now_ts = _dt.now(_tz.utc).timestamp()
            stale = []
            for fp in base.glob('*_spot_1s.csv.gz'):
                age_days = (now_ts - fp.stat().st_mtime) / 86400
                if age_days >= 7:
                    sym = fp.name.replace('_spot_1s.csv.gz', '').replace('_USDT', '/USDT')
                    stale.append(sym)
            symbols = stale
        except Exception:
            symbols = []

    job_id = uuid.uuid4().hex[:12]
    _record_resample_job(job_id,
                         status='queued',
                         created_at=time.time(),
                         symbols=symbols,
                         timeframes=['__backfill__'],  # marker; resample worker recognises
                         total_symbols=len(symbols),
                         done_symbols=0,
                         job_kind='backfill',
                         since=since)

    if not symbols:
        # No-op fast path — nothing flagged stale. Still return ok=True
        # with zero count so the UI's chip can show "0 symbols stale —
        # nothing to do".
        _record_resample_job(job_id, status='done', finished_at=time.time(),
                             progress_label='no stale symbols')
        return jsonify({'ok': True, 'job_id': job_id, 'symbols': [],
                        'note': 'no stale symbols detected — nothing to backfill'})

    # Spawn the chain: binance_archive_downloader → resample. Each
    # symbol streamed to disk; no pre-load into RAM.
    def _backfill_chain():
        try:
            _record_resample_job(job_id, status='running', started_at=time.time(),
                                 progress_label=f'archive top-up: {len(symbols)} symbols')
            try:
                from src.data_ingestion.binance_archive_downloader import \
                    download_archives_for_symbols
                # The downloader streams; doesn't load all data into RAM.
                # Falls through if module unavailable (older code path).
                download_archives_for_symbols(symbols, since=since)
            except ImportError:
                _record_resample_job(job_id, progress_label='downloader unavailable; skipping to resample')
            # Then resample for the same symbols.
            _record_resample_job(job_id, progress_label=f'resampling: {len(symbols)} symbols')
            from src.utils.resample_ohlcv import DEFAULT_TIMEFRAMES
            _run_resample_blocking(job_id, symbols, list(DEFAULT_TIMEFRAMES))
        except Exception as exc:
            _record_resample_job(job_id, status='error',
                                 finished_at=time.time(),
                                 errors=[f'{type(exc).__name__}: {exc}'])

    threading.Thread(target=_backfill_chain, daemon=True,
                     name=f'backfill-{job_id}').start()
    return jsonify({'ok': True, 'job_id': job_id, 'symbols': symbols, 'since': since})


@app.route('/api/data/resample/jobs', methods=['GET'])
def api_data_resample_jobs():
    """Most-recent N resample jobs, newest first. Used by the status pill
    in the Data Coverage panel."""
    try:
        limit = int(request.args.get('limit', 10))
    except (TypeError, ValueError):
        limit = 10
    with _resample_jobs_lock:
        rows = list(_resample_jobs.values())
    rows.sort(key=lambda x: x.get('created_at', 0), reverse=True)
    return jsonify({'jobs': rows[:limit], 'total': len(rows)})


# ─── Pipeline orchestrator (train → multi-TF backtest) ──────────────────
# The orchestrator runs as a subprocess so memory pressure during training
# can't take Flask down. We track its PID and the on-disk status file so
# the operator can see progress without an in-process supervisor.
_pipeline_proc_pid: int | None = None
_pipeline_proc_lock = threading.Lock()


def _pipeline_status_path() -> str:
    return os.path.join(project_root, 'data', 'pipeline_status.json')


def _pipeline_proc_alive() -> bool:
    """Return True if a pipeline orchestrator subprocess is running.

    Two-step check (mirrors PR-32's bot DEAD cmdline-scan fallback):
      1. Recorded PID: fast path. If our last-known PID is alive AND its
         cmdline confirms it's the orchestrator, return True.
      2. Cmdline scan: if (1) misses (recorded PID is None / stale /
         pre-PID-recycle), scan psutil for any python process running
         `-m src.engine.pipeline_orchestrator`. If found, adopt that
         PID as the new _pipeline_proc_pid so subsequent calls hit (1).

    Without (2), the dashboard reports the pipeline as DEAD whenever
    the orchestrator was started outside this dashboard process —
    e.g. a fresh dashboard boot, a manual relaunch, or the operator
    kicking it off via PowerShell directly.
    """
    global _pipeline_proc_pid
    try:
        import psutil
    except ImportError:
        return False
    # ── (1) recorded PID fast path ────────────────────────────────────
    pid = _pipeline_proc_pid
    if pid is not None:
        try:
            if psutil.pid_exists(pid):
                p = psutil.Process(pid)
                try:
                    cmd = ' '.join(p.cmdline())
                    if 'pipeline_orchestrator' in cmd:
                        return True
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
        except Exception:
            pass
        # PID stale — fall through to scan.
        with _pipeline_proc_lock:
            _pipeline_proc_pid = None
    # ── (2) cmdline scan fallback ─────────────────────────────────────
    try:
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if not (p.info.get('name') or '').lower().startswith('python'):
                continue
            cmd = p.info.get('cmdline') or []
            if (len(cmd) >= 3 and cmd[1] == '-m'
                    and cmd[2] == 'src.engine.pipeline_orchestrator'):
                with _pipeline_proc_lock:
                    _pipeline_proc_pid = int(p.info['pid'])
                return True
    except Exception:
        pass
    return False


@app.route('/api/pipeline/status', methods=['GET'])
def api_pipeline_status():
    """Return the orchestrator status snapshot. Combines the on-disk status
    file (survives dashboard restarts) with the live subprocess-alive
    check so the dashboard pill shows 'running' even if status writes lag."""
    from src.utils.safe_json import read_json
    snap = read_json(_pipeline_status_path(), default={}) or {}
    snap['process_alive'] = _pipeline_proc_alive()
    if snap.get('status') == 'running' and not snap['process_alive']:
        # Orchestrator died without writing a final status — surface this
        # so the operator can re-launch instead of waiting forever.
        snap['status'] = 'error'
        # NB: setdefault returns the existing value if the key is present —
        # even if that value is None (which the orchestrator initialises
        # `last_event` to). Coerce to a dict before mutating.
        last_event = snap.get('last_event')
        if not isinstance(last_event, dict):
            last_event = {}
        # Always set both `phase` and `message` so the frontend renderer
        # `${last_event.phase}: ${last_event.message}` doesn't print
        # 'undefined: …'. Use 'orchestrator' as the phase tag — it's
        # the orchestrator that died, not any one trainer phase.
        last_event.setdefault('phase', 'orchestrator')
        last_event['message'] = 'orchestrator process exited without finalising'
        snap['last_event'] = last_event
    return jsonify(snap)


@app.route('/api/breaker_drill/run', methods=['POST'])
@require_api_key
def api_breaker_drill_run():
    """Run the synthetic circuit-breaker drill in-process. Cheap (~ms);
    no subprocess. Returns per-scenario verdicts and pass/fail counts."""
    try:
        from src.engine.breaker_drill import run_drill
        body = request.get_json(silent=True) or {}
        only = body.get('scenario') if body.get('scenario') in (
            'max_dd', 'api_latency', 'stale_feed', 'clean') else None
        return jsonify(run_drill(only=only))
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), 500


@app.route('/api/audit_trail/run', methods=['POST'])
@require_api_key
def api_audit_trail_run():
    """Audit trades → signals → models for orphans / missing refs.
    Body: {"max_trades": 100} to limit scan to recent N (default: all)."""
    try:
        from src.engine.audit_trail import run_audit
        body = request.get_json(silent=True) or {}
        max_trades = body.get('max_trades')
        try: max_trades = int(max_trades) if max_trades is not None else None
        except (TypeError, ValueError): max_trades = None
        return jsonify(run_audit(max_trades=max_trades))
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), 500


@app.route('/api/backtest/long_horizon', methods=['POST'])
@require_api_key
def api_backtest_long_horizon():
    """Spawn a long-horizon backtest as a detached subprocess.
    Body: {"horizon": "short|medium|long|max"}. Auto-picks safe TFs
    per horizon (5y window won't try 5m bars → 250M rows blowup)."""
    global _pipeline_proc_pid
    if _pipeline_proc_alive():
        return jsonify({'ok': False,
                        'error': 'pipeline already running — backtest shares its slot',
                        'pid': _pipeline_proc_pid}), 409
    body = request.get_json(silent=True) or {}
    horizon = (body.get('horizon') or 'long').strip()
    cmd = [sys.executable, '-m', 'src.engine.long_horizon_backtest',
           '--horizon', horizon]
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"long_horizon_{int(time.time())}.log")
    try:
        log_fp = open(log_path, 'a', encoding='utf-8')
        creationflags = 0
        if os.name == 'nt':
            creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP |
                             getattr(subprocess, 'DETACHED_PROCESS', 0x00000008))
        proc = subprocess.Popen(
            cmd, cwd=project_root,
            stdout=log_fp, stderr=log_fp,
            creationflags=creationflags,
            close_fds=True,
        )
        with _pipeline_proc_lock:
            _pipeline_proc_pid = proc.pid
        return jsonify({'ok': True, 'pid': proc.pid,
                        'horizon': horizon,
                        'log_path': log_path, 'cmd': cmd})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/news/sentiment_model', methods=['GET'])
def api_news_sentiment_model():
    """Return which sentiment backend is active (cryptobert / finbert /
    lexicon). Useful for the operator to confirm whether new scraper runs
    are using a real model or the fallback."""
    try:
        from src.analysis.finbert_scorer import get_active_model, is_ready
        # Trigger lazy load so the answer is meaningful even on first call.
        ready = is_ready()
        return jsonify({
            'model':  get_active_model(),
            'ready':  ready,
            'note':   ('A real model is loaded — new scraper writes use it.'
                       if ready
                       else 'Falling back to lexicon — install `transformers` and '
                            'first scraper run will download the model.')
        })
    except Exception as exc:
        return jsonify({'model': 'unknown',
                        'ready': False,
                        'error': str(exc)}), 500


@app.route('/api/news/buffer', methods=['GET'])
def api_news_buffer_status():
    """Return the live news buffer's status (rows cached, snapshot age,
    refresh count, last error). Returns ready=false when the bot hasn't
    started a buffer (training / backtest workers run without one)."""
    try:
        from src.analysis.live_news_buffer import get_active_buffer
        buf = get_active_buffer()
        if buf is None:
            return jsonify({'ready': False, 'rows': 0,
                            'snapshot_age_s': None,
                            'refresh_count': 0,
                            'last_error': '',
                            'message': 'no buffer active in this process — bot must be running for live news inference'})
        return jsonify(buf.status())
    except Exception as exc:
        return jsonify({'ready': False, 'error': str(exc)}), 500


@app.route('/api/auto_retrain/status', methods=['GET'])
def api_auto_retrain_status():
    """Return the last auto-retrain result (or empty when never run)."""
    from src.utils.safe_json import read_json
    snap = read_json(os.path.join(project_root, 'data', 'auto_retrain_status.json'),
                     default={}) or {}
    return jsonify(snap)


@app.route('/api/auto_retrain/run', methods=['POST'])
@require_api_key
def api_auto_retrain_run():
    """Spawn auto_retrain as a detached subprocess. Body (optional):
        {"tolerance": 0.05, "rollback": false}
    Idempotent — refuses to spawn a second auto-retrain if one is alive."""
    global _pipeline_proc_pid
    # auto_retrain runs through the same pipeline orchestrator, so reuse
    # its alive-check to prevent overlapping cycles.
    if _pipeline_proc_alive():
        return jsonify({'ok': False,
                        'error': 'pipeline already running — auto-retrain shares its process slot',
                        'pid': _pipeline_proc_pid}), 409

    body = request.get_json(silent=True) or {}
    cmd = [sys.executable, '-m', 'src.engine.auto_retrain']
    try:
        cmd += ['--tolerance', str(float(body.get('tolerance', 0.05)))]
    except (TypeError, ValueError):
        pass
    if body.get('rollback'):
        cmd.append('--rollback')

    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"auto_retrain_{int(time.time())}.log")
    try:
        log_fp = open(log_path, 'a', encoding='utf-8')
        creationflags = 0
        if os.name == 'nt':
            creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP |
                             getattr(subprocess, 'DETACHED_PROCESS', 0x00000008))
        proc = subprocess.Popen(
            cmd, cwd=project_root,
            stdout=log_fp, stderr=log_fp,
            creationflags=creationflags,
            close_fds=True,
        )
        with _pipeline_proc_lock:
            _pipeline_proc_pid = proc.pid
        return jsonify({'ok': True, 'pid': proc.pid,
                        'log_path': log_path, 'cmd': cmd})
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), 500


def _run_pipeline_blocking(job_id: str, cmd: list) -> None:
    """Worker-thread body for a pipeline-orchestrator job. Acquires the
    scheduler's exclusive slot (blocking — may wait minutes if OFT or
    another exclusive job is in flight), spawns the orchestrator, waits
    for it, then releases the slot. Runs on its own daemon thread so
    api_pipeline_run returns immediately with a queued job_id."""
    global _pipeline_proc_pid
    _training_scheduler.acquire('exclusive')
    try:
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"pipeline_{int(time.time())}.log")
        log_fp = open(log_path, 'a', encoding='utf-8')
        creationflags = 0
        if os.name == 'nt':
            creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP |
                             getattr(subprocess, 'DETACHED_PROCESS', 0x00000008))
        try:
            proc = subprocess.Popen(
                cmd, cwd=project_root,
                stdout=log_fp, stderr=log_fp,
                creationflags=creationflags,
                close_fds=True,
            )
        except Exception as exc:
            _record_job(job_id, status='error', finished_at=time.time(),
                        errors=[f'{type(exc).__name__}: {exc}'])
            return
        with _pipeline_proc_lock:
            _pipeline_proc_pid = proc.pid
        _record_job(job_id, status='running', started_at=time.time(),
                    progress_label='pipeline orchestrator',
                    log_path=log_path, child_pid=proc.pid)
        # Wait for orchestrator to exit. We poll the proc rather than
        # use psutil.wait so we can also notice the dashboard shutting
        # down (the daemon thread will be terminated then).
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(2)
        _record_job(job_id,
                    status=('done' if rc == 0 else 'error'),
                    finished_at=time.time(),
                    errors=([] if rc == 0 else [f'orchestrator exit code {rc}']))
    finally:
        _training_scheduler.release('exclusive')


@app.route('/api/pipeline/run', methods=['POST'])
@require_api_key
def api_pipeline_run():
    """Queue a pipeline orchestrator run through the resource scheduler.

    Body (all optional):
        {"skip_train": bool, "skip_backtest": bool,
         "backtest_tfs": ["5m","1h","4h","1d","1w"]}

    Idempotent — refuses to spawn a second orchestrator if one is already
    running. Goes through the 'exclusive' lane so it can't collide with
    a manual OFT run (also exclusive) or take the GPU out from under a
    TFT run. Returns immediately with a queued job_id; the actual
    spawn happens on a worker thread once the lane is free."""
    global _pipeline_proc_pid
    if _pipeline_proc_alive():
        return jsonify({'ok': False,
                        'error': 'orchestrator already running',
                        'pid':   _pipeline_proc_pid}), 409

    body = request.get_json(silent=True) or {}
    cmd = [sys.executable, '-m', 'src.engine.pipeline_orchestrator']
    if body.get('skip_train'):
        cmd.append('--skip-train')
    if body.get('skip_backtest'):
        cmd.append('--skip-backtest')
    tfs = body.get('backtest_tfs')
    if isinstance(tfs, list) and tfs:
        cmd += ['--backtest-tfs', ','.join(str(t) for t in tfs)]

    # Synthetic job entry so the operator sees the orchestrator queued
    # behind any in-flight CPU/GPU/exclusive work, with the same UI
    # treatment as a training job.
    job_id = uuid.uuid4().hex[:12]
    _record_job(job_id, model='pipeline', status='queued',
                queued_at=time.time(), created_at=time.time(),
                resource_kind='exclusive',
                progress_label='queued (pipeline orchestrator)',
                tf=None, total=1, n=1)
    threading.Thread(
        target=_run_pipeline_blocking,
        args=(job_id, cmd),
        daemon=True,
        name=f'pipeline-{job_id}',
    ).start()
    return jsonify({'ok': True,
                    'job_id': job_id,
                    'status': 'queued',
                    'cmd':    cmd})


@app.route('/api/db/training_history')
def db_training_history():
    """Return training telemetry for one model."""
    model = request.args.get('model', '')
    runs = int(request.args.get('runs', 5))
    if not model:
        return jsonify({'error': 'model required'}), 400
    try:
        from src.database.parquet_client import get_client
        rows = get_client().get_training_history(model, runs)
        return jsonify({'rows': rows, 'model': model})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/db/market_stats')
def db_market_stats():
    """Return stored market data summary per symbol/timeframe."""
    try:
        from src.database.parquet_client import get_client
        c = get_client()
        if not c.is_available():
            return jsonify({'available': False, 'rows': []})
        rows = c.query(
            "SELECT symbol, timeframe, "
            "COUNT(*) as row_count, "
            "MIN(ts) as first_ts, MAX(ts) as last_ts "
            "FROM market_data "
            "GROUP BY symbol, timeframe "
            "ORDER BY symbol, timeframe"
        )
        return jsonify({'available': True, 'rows': rows})
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})


@app.route('/api/db/ingest', methods=['POST'])
def db_ingest():
    """Trigger background CSV.gz → QuestDB ingestion for given symbols."""
    body = request.get_json(force=True) or {}
    symbols    = body.get('symbols') or None
    timeframes = body.get('timeframes') or None
    since_str  = body.get('since') or None

    def _run():
        try:
            from src.database.ingest_pipeline import run
            from datetime import datetime, timezone
            since_dt = None
            if since_str:
                since_dt = datetime.strptime(since_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            run(symbols=symbols, timeframes=timeframes, since=since_dt)
        except Exception as exc:
            import logging as _lg
            _lg.getLogger(__name__).error("DB ingest error: %s", exc)

    threading.Thread(target=_run, daemon=True, name="db-ingest").start()
    return jsonify({'ok': True, 'msg': 'Ingestion started in background'})


# ─── Training Cluster endpoints ──────────────────────────────────────────────

def _get_orchestrator():
    try:
        from src.training.distributed.orchestrator import get_orchestrator
        return get_orchestrator()
    except Exception:
        return None


@app.route('/api/cluster/status')
def cluster_status():
    orch = _get_orchestrator()
    if orch is None:
        return jsonify({'error': 'Orchestrator not available'}), 503
    return jsonify(orch.get_status())


@app.route('/api/cluster/workers')
def cluster_workers():
    orch = _get_orchestrator()
    if orch is None:
        return jsonify([])
    return jsonify(orch.list_workers())


@app.route('/api/cluster/submit', methods=['POST'])
def cluster_submit():
    orch = _get_orchestrator()
    if orch is None:
        return jsonify({'error': 'Orchestrator not available'}), 503
    spec = request.get_json(force=True) or {}
    tid  = orch.submit_task(spec)
    return jsonify({'ok': True, 'task_id': tid})


@app.route('/api/cluster/submit_all', methods=['POST'])
def cluster_submit_all():
    orch = _get_orchestrator()
    if orch is None:
        return jsonify({'error': 'Orchestrator not available'}), 503
    body    = request.get_json(force=True) or {}
    symbols = body.get('symbols')
    ids     = orch.submit_full_training_run(symbols)
    return jsonify({'ok': True, 'task_ids': ids, 'count': len(ids)})


@app.route('/api/cluster/register', methods=['POST'])
def cluster_register():
    orch = _get_orchestrator()
    if orch is None:
        return jsonify({'error': 'Orchestrator not available'}), 503
    orch.register_worker(request.get_json(force=True) or {})
    return jsonify({'ok': True})


@app.route('/api/cluster/task_update', methods=['POST'])
def cluster_task_update():
    orch = _get_orchestrator()
    if orch is None:
        return jsonify({'ok': True})   # silently accept
    body = request.get_json(force=True) or {}
    orch.update_task(
        body.get('task_id', ''),
        body.get('status', ''),
        node_id=body.get('node_id', ''),
        result=body.get('result'),
        error=body.get('error', ''),
    )
    return jsonify({'ok': True})


@app.route('/api/cluster/task/<task_id>', methods=['DELETE'])
def cluster_cancel_task(task_id):
    orch = _get_orchestrator()
    if orch is None:
        return jsonify({'ok': False})
    ok = orch.cancel_task(task_id)
    return jsonify({'ok': ok})


# ────────────────────────────────────────────────────────────────────────────
#  Phase 9 — dual-balance, news, OFT signal, orchestrator stats, retention,
#  rate-limiter usage. These power the REAL vs TEST/TRAIN mode switcher.
# ────────────────────────────────────────────────────────────────────────────


@app.route('/api/balance/real')
def api_balance_real():
    try:
        from src.engine.dual_balance import read_real
        return jsonify(read_real())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# Cached balance snapshots — 30s TTL keeps the dashboard's poll cadence
# cheap without ever blocking on a slow exchange call.
_balance_live_cache: dict[str, dict] = {}
_balance_live_lock = threading.Lock()
_BALANCE_TTL_S = 30


def _live_binance_balance(testnet: bool) -> dict:
    """Return Binance balance for the requested mode.

    History: an earlier version constructed a fresh ccxt.binance per mode
    on the dashboard's hot path. Switching modes rapidly (paper ↔ testnet
    ↔ mainnet) raced on the USE_TESTNET env-var mutation AND tried to
    authenticate mainnet using testnet keys (which hangs ccxt for many
    seconds) — the dashboard worker thread blocked, then the operator's
    next request returned 'Failed to fetch'. Repeated → bot/dashboard
    instability.

    Fix: never construct ccxt on demand. Read the cached
    data/balance_real.json snapshot the bot itself maintains, and only
    actively refresh when the bot can do it safely (i.e. when its own
    OrderManager already exists in this process). Mainnet specifically
    is NEVER auto-fetched from the dashboard — it requires the operator
    to actually be running the bot in mainnet mode."""
    from datetime import datetime, timezone
    ckey = 'testnet' if testnet else 'mainnet'
    with _balance_live_lock:
        cached = _balance_live_cache.get(ckey)
        if cached and (time.time() - cached.get('_ts', 0)) < _BALANCE_TTL_S:
            return cached

    # Safety: mainnet never gets a live fetch from the dashboard. The
    # dashboard's OrderManager (if any) is configured by the bot's
    # USE_TESTNET env var at process start; flipping it on a running
    # ccxt instance can corrupt the auth context.
    out: dict = {
        'available':  False,
        'mode':       ckey,
        'cash_usdt':  None, 'spot_usdt': None,
        'futures_usdt': None, 'equity_usdt': None,
        '_ts':        time.time(),
    }
    if testnet is False:
        # Mainnet — show cached balance_real.json if the bot is in
        # mainnet mode, otherwise tell the operator to switch the bot.
        try:
            from src.engine.dual_balance import read_real
            snap = read_real() or {}
            if snap and snap.get('cash_usdt') is not None:
                out.update({
                    'available':    True,
                    'cash_usdt':    float(snap.get('cash_usdt') or 0),
                    'spot_usdt':    float(snap.get('cash_usdt') or 0),
                    'futures_usdt': 0.0,
                    'equity_usdt':  float(snap.get('equity_usdt') or 0),
                    'fetched_at':   snap.get('timestamp') or '',
                    'source':       'cached (data/balance_real.json)',
                })
            else:
                out['error'] = 'mainnet balance unavailable — start the bot with USE_TESTNET=False to populate balance_real.json'
        except Exception as exc:
            out['error'] = f'{type(exc).__name__}: {exc}'
    else:
        # Testnet — try the cached file first; only do a live fetch when
        # the bot's own OrderManager singleton is already loaded in this
        # process (lazy-loaded by other endpoints like /api/balance/real
        # via refresh_real_from_binance). Avoids constructing a new
        # ccxt instance on the dashboard's hot path.
        try:
            from src.engine.dual_balance import read_real
            snap = read_real() or {}
            usdt_spot = float(snap.get('cash_usdt') or 0)
            usdt_eq   = float(snap.get('equity_usdt') or 0)
            out.update({
                'available':    bool(snap.get('cash_usdt') is not None),
                'cash_usdt':    usdt_spot,
                'spot_usdt':    usdt_spot,
                'futures_usdt': 0.0,
                'equity_usdt':  usdt_eq,
                'fetched_at':   snap.get('timestamp') or '',
                'source':       'cached (data/balance_real.json)',
            })
            if not out.get('available'):
                out['error'] = 'no cached testnet balance — bot has not refreshed it yet'
        except Exception as exc:
            out['error'] = f'{type(exc).__name__}: {exc}'

    with _balance_live_lock:
        _balance_live_cache[ckey] = out
    return out


@app.route('/api/balance/refresh', methods=['POST'])
@require_api_key
def api_balance_refresh():
    """Operator-triggered live refresh of the cached balance file.
    Spawns refresh_real_from_binance on a background thread so the HTTP
    request returns instantly. Result lands in data/balance_real.json
    within seconds."""
    def _job():
        try:
            from src.engine.dual_balance import refresh_real_from_binance
            refresh_real_from_binance()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[balance/refresh] failed: %s", exc)
    threading.Thread(target=_job, daemon=True, name='balance-refresh').start()
    # Invalidate caches so the next /api/balance/by_mode poll picks up the new file.
    with _balance_live_lock:
        _balance_live_cache.clear()
    return jsonify({'ok': True, 'message': 'refresh queued; cached values may take a few seconds to update'})


@app.route('/api/portfolio')
def api_portfolio():
    """Mode-aware Performance Overview snapshot.

    Returns a single payload the dashboard's Balances / Risk / Portfolio
    cards can consume directly, populated from the right backing store
    for the requested mode:

        paper    — data/balance_virtual.json + zero open positions
        testnet  — live Binance testnet balance + filter trades.json
        mainnet  — live Binance mainnet balance + filter trades.json

    Replaces the pre-PR-46 path where those cards read from /api/state
    (mode-blind) and tradesData (also mode-blind), so switching the
    operator's mode toggle didn't refresh the numbers below.

    Trades.json doesn't yet carry a per-row mode tag (~912 rows
    untagged), so for non-paper modes we report all closed trades
    without filtering. Going forward the bot can stamp `mode` on each
    trade write — once that lands, this endpoint can filter cleanly.
    """
    mode = (request.args.get('mode') or '').strip().lower()
    if not mode:
        ctrl = read_json('data/control.json', default={}) or {}
        mode = (ctrl.get('trade_mode') or 'testnet').lower()

    out = {
        'mode':                  mode,
        'balances':              [],   # [{asset, qty, value, price}]
        'total_capital':         0.0,
        'free_usdt':             0.0,
        'in_positions_value':    0.0,
        'live_pnl':              0.0,
        'closed_pnl':            0.0,
        'today_pnl':             0.0,
        'open_position_count':   0,
        'closed_position_count': 0,
        'trade_mode_label':      '',
    }

    if mode == 'paper':
        # Paper: read the virtual balance directly. No exchange holdings.
        # Closed PnL comes from balance_virtual's revenue_total (sum of
        # closed paper-trade pnl). Live PnL = 0 by construction (paper
        # trades are booked atomically — there are no open positions
        # carried in balance_virtual.json).
        try:
            from src.engine.dual_balance import read_virtual, compute_summary
            snap = read_virtual() or {}
            summary = compute_summary() or {}
        except Exception:
            snap, summary = {}, {}
        cash    = float(snap.get('cash_usdt', 0) or 0)
        equity  = float(snap.get('equity_usdt', 0) or 0)
        revenue = float(snap.get('revenue_total', 0) or 0)
        out['balances'] = [{
            'asset': 'USDT', 'qty': cash, 'value': cash, 'price': 1.0,
        }]
        # Surface paper holdings if any exist (the schema supports it
        # even if today the dict is usually empty).
        for asset, qty in (snap.get('holdings') or {}).items():
            try:
                qty_f = float(qty)
                out['balances'].append({
                    'asset': asset, 'qty': qty_f,
                    'value': qty_f, 'price': None,
                })
            except Exception:
                continue
        out['total_capital']      = equity
        out['free_usdt']          = cash
        out['in_positions_value'] = max(0.0, equity - cash)
        out['live_pnl']           = 0.0
        out['closed_pnl']         = revenue
        out['today_pnl']          = float(snap.get('pnl_24h', 0) or 0)
        out['trade_mode_label']   = 'paper — internal-only booking, no real orders'
        return jsonify(out)

    # Live exchange (testnet / mainnet) — use the same helper that
    # /api/balance/by_mode uses, then merge in trades.json for PnL.
    try:
        live = _live_binance_balance(testnet=(mode == 'testnet'))
    except Exception as exc:
        return jsonify({**out, 'error': f'{type(exc).__name__}: {exc}'}), 200
    if not live or live.get('available') is False:
        out['error'] = live.get('error', 'exchange unavailable') if isinstance(live, dict) else 'exchange unavailable'
        return jsonify(out), 200

    spot_usdt   = float(live.get('spot_usdt', 0)    or 0)
    equity      = float(live.get('equity_usdt', 0)  or 0)
    holdings    = (live.get('balances') or live.get('holdings') or {}) or {}
    out['free_usdt']          = spot_usdt
    out['total_capital']      = equity
    out['in_positions_value'] = max(0.0, equity - spot_usdt)
    out['balances']           = [{
        'asset': 'USDT', 'qty': spot_usdt, 'value': spot_usdt, 'price': 1.0,
    }]
    if isinstance(holdings, dict):
        for asset, qty in holdings.items():
            try:
                qty_f = float(qty)
                if qty_f == 0 or asset == 'USDT':
                    continue
                out['balances'].append({
                    'asset': asset, 'qty': qty_f,
                    'value': None, 'price': None,
                })
            except Exception:
                continue

    # Trades-derived live + closed PnL. Trades.json has no per-row mode
    # tag yet, so for testnet/mainnet we report all closed trades
    # (most existing 912 rows are testnet anyway). When the bot starts
    # stamping `mode` on writes, we can switch to a strict filter here.
    try:
        from src.utils.safe_json import read_json as _rj
        raw = _rj('data/trades.json', default=[]) or []
        if isinstance(raw, dict):
            raw = raw.get('trades') or []
        if not isinstance(raw, list):
            raw = []
        # When `mode` IS present on rows, prefer strict filter — that's
        # the eventual end state once the bot tags every write.
        tagged = [t for t in raw if isinstance(t, dict) and t.get('mode')]
        if tagged:
            tradeset = [t for t in tagged if t.get('mode') == mode]
        else:
            tradeset = [t for t in raw if isinstance(t, dict)]
    except Exception:
        tradeset = []

    from datetime import datetime as _dt
    today_ts = _dt.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    open_cnt, closed_cnt = 0, 0
    live_pnl, closed_pnl, today_pnl = 0.0, 0.0, 0.0
    for t in tradeset:
        st = (t.get('status') or '').upper()
        if st == 'OPEN':
            open_cnt += 1
            live_pnl += float(t.get('unrealized_pnl', 0) or 0)
        elif st == 'CLOSED':
            closed_cnt += 1
            p = float(t.get('pnl_usdt', 0) or 0)
            closed_pnl += p
            try:
                close_str = t.get('sell_time') or t.get('closed_at') or ''
                close_ts = _dt.fromisoformat(str(close_str).replace('Z', '+00:00')).timestamp()
                if close_ts >= today_ts:
                    today_pnl += p
            except Exception:
                pass
    out['live_pnl']              = live_pnl
    out['closed_pnl']            = closed_pnl
    out['today_pnl']             = today_pnl
    out['open_position_count']   = open_cnt
    out['closed_position_count'] = closed_cnt
    out['trade_mode_label']      = ('testnet — Binance fake-money exchange'
                                    if mode == 'testnet'
                                    else '⚠ REAL CASH — live Binance, real money')
    return jsonify(out)


@app.route('/api/balance/by_mode')
def api_balance_by_mode():
    """Return the balance appropriate for the active trade mode.
       paper   → internal virtual balance (with deposits/revenue split)
       testnet → live Binance testnet balance (USDT spot + futures)
       mainnet → live Binance mainnet balance (USDT spot + futures)
    Mode comes from query string (?mode=...) or, if absent, from
    data/control.json so the dashboard can call without arguments."""
    from datetime import datetime, timezone
    mode = (request.args.get('mode') or '').strip().lower()
    if not mode:
        ctrl = read_json('data/control.json', default={}) or {}
        mode = (ctrl.get('trade_mode') or 'testnet').lower()
    if mode == 'paper':
        # Re-use the virtual balance endpoint payload exactly.
        from src.engine.dual_balance import read_virtual, compute_summary
        snap = read_virtual()
        snap['summary'] = compute_summary()
        snap['mode'] = 'paper'
        return jsonify(snap)
    if mode in ('testnet', 'mainnet'):
        return jsonify(_live_binance_balance(testnet=(mode == 'testnet')))
    return jsonify({'error': f'unknown mode: {mode}',
                    'valid': ['paper', 'testnet', 'mainnet']}), 400


# ─── Local scheduler endpoints (Windows Task Scheduler wrapper) ──────────────
# All execution stays on the local machine. No cloud, no remote agents.
# The dashboard's Scheduler panel calls these to register/run/list/remove
# native Windows scheduled tasks that invoke scripts/check_training_status.py.

_SCHEDULER_PS1 = _PROJECT_ROOT / 'local_scheduler.ps1'
_DEFAULT_TASK_PREFIX = 'AI-Trader-'
_TRAINING_REPORT_PATH = _PROJECT_ROOT / 'data' / 'training_status_report.json'


def _safe_task_name(name: str) -> str:
    """Restrict task names to alphanum + dash/underscore + AI-Trader- prefix.
    Prevents shell injection via the task name."""
    import re as _re
    name = (name or '').strip()
    name = _re.sub(r'[^A-Za-z0-9_\-]', '', name)
    if not name:
        name = 'TrainingStatus'
    if not name.startswith(_DEFAULT_TASK_PREFIX):
        name = _DEFAULT_TASK_PREFIX + name
    return name[:120]


def _run_schtasks(args: list[str]) -> dict:
    """Run schtasks.exe and return {ok, code, stdout, stderr}."""
    import subprocess as _sp
    try:
        r = _sp.run(['schtasks.exe', *args], capture_output=True,
                    text=True, timeout=15)
        return {
            'ok': r.returncode == 0,
            'code': r.returncode,
            'stdout': (r.stdout or '').strip(),
            'stderr': (r.stderr or '').strip(),
        }
    except Exception as exc:
        return {'ok': False, 'code': -1, 'stdout': '', 'stderr': str(exc)}


@app.route('/api/scheduler/list', methods=['GET'])
def api_scheduler_list():
    """Return all Windows scheduled tasks whose name starts with AI-Trader-."""
    res = _run_schtasks(['/Query', '/FO', 'CSV', '/NH'])
    if not res['ok']:
        return jsonify({'tasks': [], 'error': res['stderr']}), 200
    tasks = []
    for line in res['stdout'].splitlines():
        cols = [c.strip('"') for c in line.split('","')]
        if not cols or len(cols) < 3:
            continue
        name = cols[0].lstrip('\\')
        if not name.startswith(_DEFAULT_TASK_PREFIX):
            continue
        tasks.append({
            'name':       name,
            'next_run':   cols[1] if len(cols) > 1 else '',
            'status':     cols[2] if len(cols) > 2 else '',
        })
    return jsonify({'tasks': tasks})


@app.route('/api/scheduler/register', methods=['POST'])
def api_scheduler_register():
    """body: {name, mode: 'daily'|'every_minutes'|'once', value}"""
    body = request.get_json(silent=True) or {}
    name = _safe_task_name(body.get('name', ''))
    mode = (body.get('mode') or '').strip().lower()
    value = str(body.get('value') or '').strip()

    if not _SCHEDULER_PS1.exists():
        return jsonify({'ok': False, 'error': f'launcher missing: {_SCHEDULER_PS1}'}), 500
    if mode not in ('daily', 'every_minutes', 'once'):
        return jsonify({'ok': False, 'error': "mode must be daily/every_minutes/once"}), 400

    cmd = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
           '-File', str(_SCHEDULER_PS1), 'register', '-Name', name]
    if mode == 'daily':
        # Validate HH:MM
        import re as _re
        if not _re.fullmatch(r'\d{2}:\d{2}', value):
            return jsonify({'ok': False, 'error': "value must be HH:MM"}), 400
        cmd += ['-At', value]
    elif mode == 'every_minutes':
        if not value.isdigit() or not (1 <= int(value) <= 1440):
            return jsonify({'ok': False, 'error': "value must be 1..1440 minutes"}), 400
        cmd += ['-EveryMinutes', value]
    else:  # once
        # Accept ISO YYYY-MM-DDTHH:MM or YYYY-MM-DDTHH:MM:SS
        import re as _re
        if not _re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?', value):
            return jsonify({'ok': False, 'error': "value must be YYYY-MM-DDTHH:MM[:SS]"}), 400
        cmd += ['-Once', value]

    import subprocess as _sp
    try:
        r = _sp.run(cmd, capture_output=True, text=True, timeout=20,
                    cwd=str(_PROJECT_ROOT))
        return jsonify({
            'ok': r.returncode == 0,
            'name': name,
            'mode': mode,
            'value': value,
            'stdout': (r.stdout or '').strip(),
            'stderr': (r.stderr or '').strip(),
        })
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/scheduler/run', methods=['POST'])
def api_scheduler_run():
    """body: {name}"""
    body = request.get_json(silent=True) or {}
    name = _safe_task_name(body.get('name', ''))
    res = _run_schtasks(['/Run', '/TN', name])
    return jsonify({
        'ok': res['ok'], 'name': name,
        'stdout': res['stdout'], 'stderr': res['stderr'],
    })


@app.route('/api/scheduler/unregister', methods=['POST'])
def api_scheduler_unregister():
    """body: {name}"""
    body = request.get_json(silent=True) or {}
    name = _safe_task_name(body.get('name', ''))
    res = _run_schtasks(['/Delete', '/TN', name, '/F'])
    return jsonify({
        'ok': res['ok'], 'name': name,
        'stdout': res['stdout'], 'stderr': res['stderr'],
    })


@app.route('/api/scheduler/report', methods=['GET'])
def api_scheduler_report():
    """Return the latest training_status_report.json (the file the scheduled
    task writes). Includes file mtime + age so the UI can show 'last run'."""
    if not _TRAINING_REPORT_PATH.exists():
        return jsonify({'present': False, 'hint': 'Run a task at least once first.'})
    try:
        import json as _json, time as _t
        from datetime import datetime as _dt, timezone as _tz
        st = _TRAINING_REPORT_PATH.stat()
        return jsonify({
            'present':     True,
            'age_s':       round(_t.time() - st.st_mtime, 1),
            'mtime_iso':   _dt.fromtimestamp(st.st_mtime, tz=_tz.utc).isoformat(),
            'size_bytes':  st.st_size,
            'report':      _json.loads(_TRAINING_REPORT_PATH.read_text(encoding='utf-8')),
        })
    except Exception as exc:
        return jsonify({'present': True, 'error': str(exc)}), 500


# ─── End scheduler endpoints ──────────────────────────────────────────────────


_VIRTUAL_STUB_VALUE = 12345.67  # historical placeholder written by an early dev fixture
_VIRTUAL_DEFAULT_CASH = 100_000.0


@app.route('/api/balance/virtual')
@app.route('/api/balance/test')  # legacy alias — frontend uses 'test' mode label
def api_balance_virtual():
    try:
        from src.engine.dual_balance import read_virtual, reset_virtual, compute_summary
        snap = read_virtual()
        # Auto-heal: an early stub wrote $12345.67 to the virtual balance file.
        # Replace it once with a sensible $100k so the Portfolio panel doesn't
        # display a bogus default until the simulator generates real PnL.
        if (abs(float(snap.get('cash_usdt', 0)) - _VIRTUAL_STUB_VALUE) < 1e-3
                and abs(float(snap.get('equity_usdt', 0)) - _VIRTUAL_STUB_VALUE) < 1e-3
                and not snap.get('holdings')):
            snap = reset_virtual(_VIRTUAL_DEFAULT_CASH)
        # Decompose into deposits / revenue / pnl for the Overview panel.
        summary = compute_summary()
        snap = {**snap, "summary": summary}
        return jsonify(snap)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/balance/virtual/deposit', methods=['POST'])
@require_api_key
def api_balance_virtual_deposit():
    """Operator manually adds funds to the virtual balance. The internal
    paper account never auto-syncs from the exchange — every increase
    is either a closed paper trade's PnL (auto) or an explicit deposit
    here. Total P&L is then equity − sum(deposits)."""
    try:
        from src.engine.dual_balance import add_deposit
        body = request.get_json(silent=True) or {}
        amount = float(body.get('amount', 0))
        if amount == 0:
            return jsonify({'ok': False, 'error': 'amount required'}), 400
        note = str(body.get('note', '') or '')[:120]
        return jsonify(add_deposit(amount, note=note))
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/control/trade_mode', methods=['GET', 'POST'])
def api_control_trade_mode():
    """Get / set the live-trading mode.

    Three values:
      paper   — orders never hit the exchange; routed to paper_book.
                Bot still consumes live Binance feed + generates signals.
      testnet — orders go to Binance testnet (legacy default).
      mainnet — orders go to Binance mainnet (real money). POST to mainnet
                requires confirm=true in the body to discourage misclicks.
    """
    if request.method == 'GET':
        ctrl = read_json('data/control.json', default={}) or {}
        return jsonify({
            'trade_mode': (ctrl.get('trade_mode') or 'testnet').lower(),
            'valid':      ['paper', 'testnet', 'mainnet'],
        })
    # POST — manually enforce auth (decorator can't gate a single method
    # of a multi-method route).
    if DASHBOARD_API_KEY and request.headers.get('X-API-Key', '') != DASHBOARD_API_KEY:
        return jsonify({'ok': False, 'error': 'unauthorized'}), 401
    body = request.get_json(silent=True) or {}
    mode = (body.get('mode') or '').strip().lower()
    if mode not in ('paper', 'testnet', 'mainnet'):
        return jsonify({'ok': False,
                        'error': f'invalid mode: {mode}',
                        'valid': ['paper', 'testnet', 'mainnet']}), 400
    if mode == 'mainnet' and not body.get('confirm'):
        return jsonify({'ok': False,
                        'error': 'mainnet requires confirm=true in body — '
                                 'real money will be at risk'}), 400
    ctrl = read_json('data/control.json', default={}) or {}
    if not isinstance(ctrl, dict):
        ctrl = {}
    ctrl['trade_mode'] = mode
    write_json('data/control.json', ctrl)
    logger = __import__('logging').getLogger(__name__)
    logger.warning("[control] trade_mode → %s (operator action)", mode)
    return jsonify({'ok': True, 'trade_mode': mode})


@app.route('/api/balance/virtual/reset', methods=['POST'])
def api_balance_virtual_reset():
    try:
        from src.engine.dual_balance import reset_virtual
        body = request.get_json(silent=True) or {}
        cash = float(body.get('cash', 100_000.0))
        return jsonify(reset_virtual(initial_cash=cash))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/news')
def api_news():
    """Most recent news from `_NEWS/news/yyyymm=*/`. The news partition uses
    `published_at` (not `timestamp`) as its time column — query directly."""
    import traceback as _tb
    try:
        from src.database.parquet_store import _partition_glob, get_store
        store = get_store()
        glob = _partition_glob(store.base_dir, "_NEWS", "news")
        from pathlib import Path
        if not list(Path(store.base_dir).glob("_NEWS/news/yyyymm=*/*.parquet")):
            return jsonify([])
        # Use a direct DuckDB query — the generic .query() expects a `timestamp` col.
        sql = f"SELECT * FROM read_parquet('{glob}') ORDER BY published_at DESC LIMIT 50"
        df = store._conn_or_open().execute(sql).df()
        if df is None or df.empty:
            return jsonify([])
        # Convert datetime columns to strings for JSON serialisation
        for c in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df[c] = df[c].astype(str)
        return jsonify(df.to_dict(orient='records'))
    except Exception as exc:
        print('[/api/news] FAILED:', exc, flush=True)
        _tb.print_exc()
        return jsonify({'error': str(exc), 'trace': _tb.format_exc()[-800:]}), 500


@app.route('/api/oft_signal/<path:symbol>')
def api_oft_signal(symbol):
    """Return the latest OFT prediction for a symbol from inference_engine."""
    try:
        # Inference engine is held by the live bot; we read its
        # state via the existing state.json (set by the main loop).
        from src.utils.safe_json import read_json
        st = read_json('data/state.json', default={}) or {}
        oft = (st.get('quant', {}) or {}).get(symbol, {}).get('oft', None)
        if oft:
            return jsonify(oft)
        return jsonify({'available': False})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/orchestrator/sources')
def api_orchestrator_sources():
    try:
        from src.data_governance import list_sources
        return jsonify(list_sources())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/orchestrator/config')
def api_orchestrator_config():
    try:
        from src.data_governance import GovernanceConfig
        cfg = GovernanceConfig.load()
        return jsonify({
            "history_days":         cfg.history_days,
            "store_local":          cfg.store_local,
            "google_drive_archive": cfg.google_drive_archive,
            "sources": {n: s.__dict__ for n, s in cfg.sources.items()},
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/retention/stats')
def api_retention_stats():
    try:
        from src.database.retention_manager import RetentionManager
        return jsonify(RetentionManager().stats())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/rate_limiter/stats')
def api_rate_limiter_stats():
    try:
        from src.data_ingestion.rate_limiter import stats as rl_stats
        return jsonify(rl_stats())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/decision_summary/<path:symbol>')
def api_decision_summary(symbol):
    try:
        from src.analytics import DecisionMetrics
        tf = request.args.get('tf', '1h')
        return jsonify(DecisionMetrics().summarize(symbol=symbol, timeframe=tf).to_dict())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ─── Error monitor + log retention ─────────────────────────────────────────────

@app.route('/api/errors/recent')
def api_errors_recent():
    """Return active error/warning entries (last 30 min). The dashboard
    banner polls this and shows critical entries until they auto-clear."""
    try:
        from src.dashboard import error_monitor as _em
        # Force a fresh scan if the cached state is stale (>60s) so the
        # banner doesn't lag behind a brand-new failure. We probe both
        # log files AND live status surfaces (services / processes /
        # agents / cluster / scheduler) so any non-OK card shows up here.
        _em.scan()
        _em.scan_status_surfaces()
        rows = _em.get_active()
        crit = [r for r in rows if r.get('kind') == 'critical']
        warn = [r for r in rows if r.get('kind') == 'warning']
        return jsonify({
            'critical': crit,
            'warning':  warn,
            'count_critical': len(crit),
            'count_warning':  len(warn),
        })
    except Exception as exc:
        return jsonify({'error': str(exc),
                        'critical': [], 'warning': [],
                        'count_critical': 0, 'count_warning': 0}), 200


@app.route('/api/errors/dismiss', methods=['POST'])
def api_errors_dismiss():
    """Manual dismiss from the UI. Body: {key}."""
    try:
        from src.dashboard import error_monitor as _em
        body = request.get_json(silent=True) or {}
        key = body.get('key', '').strip()
        if not key:
            return jsonify({'ok': False, 'error': 'key required'}), 400
        ok = _em.dismiss(key)
        return jsonify({'ok': ok})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/errors/dismiss_all', methods=['POST'])
def api_errors_dismiss_all():
    """Wipe every active entry. Used by the banner's 'Clear All' button."""
    try:
        from src.dashboard import error_monitor as _em
        n = _em.dismiss_all()
        return jsonify({'ok': True, 'cleared': n})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/debug/deaths')
def api_debug_deaths():
    """Recent process deaths captured by scripts/debug_supervisor.py.

    Returns the newest-first list (capped at 50 in the response). The
    supervisor itself caps storage at 200. Each record has:
        role, pid, died_at, age_s, rss_mb, cpu_pct, exit_clue,
        last_log_line, log_tail (last 20 lines).
    `running` reflects whether the supervisor itself is alive (probed
    via process_ids.json). `present` reflects whether deaths.json exists
    (i.e. at least one death has been captured ever).
    """
    try:
        import json as _json
        deaths_path = _PROJECT_ROOT / 'data' / 'process_deaths.json'
        pids = read_json('data/process_ids.json', default={})
        sup_pid = pids.get('debug')
        sup_running = bool(sup_pid) and _pid_alive(sup_pid)
        deaths = []
        present = deaths_path.exists()
        if present:
            try:
                deaths = _json.loads(deaths_path.read_text(encoding='utf-8') or '[]')
            except Exception:
                deaths = []
        return jsonify({
            'deaths':  deaths[:50],
            'count':   len(deaths),
            'present': present,
            'running': sup_running,
            'hint':   None if sup_running
                      else 'debug_supervisor not running — re-run restart_all.ps1',
        })
    except Exception as exc:
        return jsonify({'deaths': [], 'error': str(exc)}), 500


# ─── Runtime risk overrides ────────────────────────────────────────────────────
_RUNTIME_OVERRIDES_PATH = _PROJECT_ROOT / 'data' / 'runtime_overrides.json'

_RUNTIME_OVERRIDES_DEFAULT = {
    "max_position_usdt":           None,    # None = no cap, override Kelly+GARCH+OFT
    "scalping_disabled_symbols":   [
        "BTC/USDT", "ETH/USDT", "DOGE/USDT", "TRX/USDT", "UNI/USDT", "SUI/USDT",
    ],
    "trailing_stop_pct_scalping":  None,    # None = use DEFAULT_TRAILING_STOP_PCT
    "_updated_at":                 "",
    "_updated_by":                 "",
}


def _read_runtime_overrides() -> dict:
    """Load with defaults — never raises, never returns None."""
    try:
        if _RUNTIME_OVERRIDES_PATH.exists():
            with open(_RUNTIME_OVERRIDES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f) or {}
        else:
            data = {}
    except Exception:
        data = {}
    out = dict(_RUNTIME_OVERRIDES_DEFAULT)
    out.update({k: v for k, v in data.items() if k in _RUNTIME_OVERRIDES_DEFAULT})
    return out


def _write_runtime_overrides(payload: dict) -> dict:
    from datetime import datetime, timezone
    cur = _read_runtime_overrides()
    # Whitelist-merge so unknown keys can't sneak in.
    for k in ('max_position_usdt', 'scalping_disabled_symbols',
              'trailing_stop_pct_scalping'):
        if k in payload:
            cur[k] = payload[k]
    cur['_updated_at'] = datetime.now(timezone.utc).isoformat()
    cur['_updated_by'] = (payload.get('_updated_by') or 'dashboard')[:80]
    _RUNTIME_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RUNTIME_OVERRIDES_PATH, 'w', encoding='utf-8') as f:
        json.dump(cur, f, indent=2)
    return cur


@app.route('/api/risk/overrides', methods=['GET'])
def api_risk_overrides_get():
    return jsonify(_read_runtime_overrides())


@app.route('/api/risk/overrides', methods=['POST'])
@require_api_key
def api_risk_overrides_set():
    body = request.get_json(silent=True) or {}
    # Soft validation
    cap = body.get('max_position_usdt')
    if cap is not None:
        try:
            cap = float(cap)
            if cap < 0 or cap > 1_000_000:
                return jsonify({'ok': False, 'error': 'max_position_usdt out of range'}), 400
            body['max_position_usdt'] = cap
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'max_position_usdt must be numeric or null'}), 400
    syms = body.get('scalping_disabled_symbols')
    if syms is not None:
        if not isinstance(syms, list) or not all(isinstance(s, str) for s in syms):
            return jsonify({'ok': False, 'error': 'scalping_disabled_symbols must be list[str]'}), 400
        body['scalping_disabled_symbols'] = [s.strip() for s in syms if s.strip()]
    tstop = body.get('trailing_stop_pct_scalping')
    if tstop is not None:
        try:
            tstop = float(tstop)
            if tstop <= 0 or tstop > 50:
                return jsonify({'ok': False, 'error': 'trailing_stop_pct_scalping out of range (0-50)'}), 400
            body['trailing_stop_pct_scalping'] = tstop
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'trailing_stop_pct_scalping must be numeric or null'}), 400
    saved = _write_runtime_overrides(body)
    return jsonify({'ok': True, 'overrides': saved})


_parquet_coverage_cache = {'ts': 0.0, 'data': None}

@app.route('/api/parquet/coverage')
def api_parquet_coverage():
    """Coverage iterates 25+ symbols × multiple timeframes × COUNT/MIN/MAX.
    First call is slow (~30-60 s); cache the result for 5 minutes."""
    import time as _time, traceback as _tb
    now = _time.time()
    if _parquet_coverage_cache['data'] is not None and (now - _parquet_coverage_cache['ts']) < 300:
        return jsonify(_parquet_coverage_cache['data'])
    try:
        from src.database.parquet_store import get_store
        data = get_store().status()
        _parquet_coverage_cache.update({'ts': now, 'data': data})
        return jsonify(data)
    except Exception as exc:
        print('[parquet/coverage] FAILED:', exc, flush=True)
        _tb.print_exc()
        return jsonify({'error': str(exc), 'trace': _tb.format_exc()[-800:]}), 500


# ────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    # Phase 21 — start log retention + error monitor as daemon threads inside
    # the dashboard process. Both are idempotent and silent if logs/ is empty.
    try:
        from src.utils.log_retention import start_retention_thread, sweep_once
        sweep_once()  # one prune at boot
        start_retention_thread()
    except Exception as _e:
        print(f"[dashboard] log_retention init failed: {_e}")
    try:
        from src.dashboard.error_monitor import start_monitor_thread
        start_monitor_thread()
    except Exception as _e:
        print(f"[dashboard] error_monitor init failed: {_e}")

    # Phase 11 — bind to a dedicated IP via env var. Defaults to 0.0.0.0
    # so the dashboard remains reachable on every interface unless the
    # operator explicitly binds it to e.g. 192.168.0.99.
    _host = os.getenv('DASHBOARD_BIND_HOST', '0.0.0.0')
    _port = int(os.getenv('DASHBOARD_BIND_PORT', '5000'))
    print(f"[dashboard] binding {_host}:{_port}")
    app.run(host=_host, port=_port, debug=False)
