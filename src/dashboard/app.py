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

from src.utils.safe_json import read_json, write_json

load_dotenv()
app = Flask(__name__)

# ─── Gemini model health cache ────────────────────────────────────────────────
_GEMINI_MODELS = ['gemini-3.1-pro-preview', 'gemini-3.1-flash-lite-preview']
_active_model: str | None = None   # best model confirmed working
_model_lock = threading.Lock()

def _probe_models_bg():
    """Test each Gemini model with a tiny request; cache the first that responds."""
    global _active_model
    api_key = os.getenv('GEMINI_API_KEY', '')
    if not api_key or api_key == 'your_api_key_here':
        return
    try:
        from google import genai as _gp
        from google.genai import types as _gtp
        client = _gp.Client(api_key=api_key)
        for model_id in _GEMINI_MODELS:
            try:
                client.models.generate_content(
                    model=model_id,
                    contents='ping',
                    config=_gtp.GenerateContentConfig(
                        http_options=_gtp.HttpOptions(timeout=8000),
                    ),
                )
                with _model_lock:
                    _active_model = model_id
                return
            except Exception:
                continue
    except Exception:
        pass
    with _model_lock:
        _active_model = None   # all models unavailable right now

# Probe on startup — non-blocking background thread
threading.Thread(target=_probe_models_bg, daemon=True).start()

# Re-probe every 5 minutes so status stays fresh
def _schedule_probe():
    while True:
        import time; time.sleep(300)
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
    state = read_json('data/state.json', default={})
    trades = read_json('data/trades.json', default=[])
    safe_state = {k: v for k, v in state.items() if 'key' not in k.lower() and 'secret' not in k.lower()}

    from collections import Counter
    open_trades   = [t for t in trades if str(t.get('status', '')).upper() == 'OPEN']
    closed_trades = [t for t in trades if str(t.get('status', '')).upper() == 'CLOSED']

    wins      = [t for t in closed_trades if (t.get('pnl_usdt') or 0) > 0]
    win_rate  = round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0
    total_pnl = sum(t.get('pnl_usdt') or 0 for t in closed_trades)

    strat_counts = Counter(t.get('market', 'SPOT') for t in closed_trades)
    strat_pnl = {}
    for t in closed_trades:
        m = t.get('market', 'SPOT')
        strat_pnl[m] = strat_pnl.get(m, 0) + (t.get('pnl_usdt') or 0)

    _META = {
        'spot':     'models/btc_rf_model_meta.json',
        'scalping': 'models/scalping_model_meta.json',
        'futures':  'models/futures_short_model_meta.json',
        'trend':    'models/trend_model_meta.json',
    }
    ml_acc = {k: read_json(v, default={}).get('accuracy', 'N/A') for k, v in _META.items()}

    open_summary = []
    for t in open_trades[:10]:
        bp   = t.get('buy_price', 0)
        cp   = t.get('current_price') or bp
        upnl = t.get('unrealized_pnl') or ((cp - bp) * (t.get('amount_coin') or 0) if bp else 0)
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
    try:
        from google import genai as _genai
        from google.genai import types as _gtypes
    except ImportError:
        return jsonify({"response": "The google-genai library is not installed. Run: pip install google-genai"})

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return jsonify({"response": "⚠️ **Error:** Add `GEMINI_API_KEY=your_key` to the `.env` file."})

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"response": "Empty message received."})

    command, command_result = _exec_bot_command(user_message.lower())
    trades, safe_state, context = _build_portfolio_context()

    # Tool: article / YouTube link analysis
    url_match = re.search(r'(https?://[^\s]+)', user_message)
    if url_match:
        try:
            from importlib import import_module
            scraper = import_module('src.analysis.web_scraper_bot')
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

    # Use the probe-cached model first (fastest path), then try the rest
    with _model_lock:
        preferred = _active_model
    if preferred:
        models_to_try = [preferred] + [m for m in _GEMINI_MODELS if m != preferred]
    else:
        models_to_try = _GEMINI_MODELS

    client = _genai.Client(api_key=api_key)
    last_err = None
    transient_fail = False
    used_fallback = False
    for i, model_id in enumerate(models_to_try):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=user_message,
                config=_gtypes.GenerateContentConfig(
                    system_instruction=system_prompt,
                    http_options=_gtypes.HttpOptions(timeout=60000),
                ),
            )
            # Update cached model if we fell back to a different one
            if used_fallback or model_id != preferred:
                with _model_lock:
                    _active_model = model_id
            return jsonify({"response": resp.text, "model": model_id,
                            "command": command, "command_result": command_result})
        except Exception as e:
            last_err = e
            err_s = str(e).lower()
            if any(x in err_s for x in _TRANSIENT):
                transient_fail = True
                used_fallback = True
                continue
            break

    # All models failed — re-probe in background so next request gets fresh routing
    threading.Thread(target=_probe_models_bg, daemon=True).start()

    if transient_fail:
        ai_msg = ("⚠ All Gemini models are temporarily overloaded. "
                  "Your command was executed — see the result above. "
                  "The system will retry routing on your next message.")
    else:
        ai_msg = f"Gemini API Error: {str(last_err)}"

    return jsonify({"response": ai_msg, "model": None,
                    "command": command, "command_result": command_result})


@app.route('/api/ai_status')
def ai_status():
    with _model_lock:
        model = _active_model
    return jsonify({'model': model, 'available': model is not None})


_WATCHLIST_FILE = 'data/watchlist.json'
_DEFAULT_WATCHLIST = ['BTC/USDT', 'SOL/USDT', 'ADA/USDT']

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
