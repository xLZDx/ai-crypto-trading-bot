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


@app.route('/api/chat', methods=['POST'])
@require_api_key
def chat():
    try:
        import google.generativeai as genai
    except ImportError:
        return jsonify({"response": "The google-generativeai library is not installed. Run pip install google-generativeai."})

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return jsonify({"response": "⚠️ **Error:** Add `GEMINI_API_KEY=your_key` to the `.env` file."})

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"response": "Empty message received."})

    # Tool: model retraining trigger
    if any(word in user_message.lower() for word in ["train", "retrain"]):
        def run_train():
            subprocess.run([sys.executable, "src/engine/train_model.py"])
        threading.Thread(target=run_train, daemon=True).start()
        return jsonify({"response": "⚙️ **Command accepted:** Background ML model retraining started!"})

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

    # Load bot context — strip API key fields before sending to Gemini
    state = read_json('data/state.json', default={})
    trades = read_json('data/trades.json', default=[])
    # Remove sensitive fields from the state before embedding in the prompt
    safe_state = {k: v for k, v in state.items() if 'key' not in k.lower() and 'secret' not in k.lower()}

    system_prompt = (
        "You are an AI Assistant embedded in a crypto trading dashboard. "
        "You give advice, analyze the market, and help the trader. "
        "You CANNOT place or cancel trades — you are strictly an analytical assistant. "
        f"CURRENT BOT STATE: {safe_state}\n"
        f"TRADE HISTORY (last 20): {trades[-20:]}\n"
        "Answer concisely and professionally."
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_prompt)
        response = model.generate_content(user_message, request_options={"timeout": 30})
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"Gemini API Error: {str(e)}"})


_WATCHLIST_FILE = 'data/watchlist.json'
_DEFAULT_WATCHLIST = ['BTC/USDT', 'SOL/USDT', 'ADA/USDT']


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
                from src.data_ingestion.binance_downloader import download_history
                download_history(symbol=symbol, timeframe='1h', limit=1000)
                download_history(symbol=symbol, timeframe='1m', limit=1000)
            except Exception as exc:
                _log.getLogger(__name__).error(f'Watchlist download {symbol}: {exc}')

        threading.Thread(target=_bg_download, daemon=True).start()
    return jsonify({'symbols': symbols, 'added': symbol})


@app.route('/api/watchlist/remove', methods=['POST'])
@require_api_key
def remove_watchlist():
    symbol = (request.json or {}).get('symbol', '').upper().strip()
    symbols = read_json(_WATCHLIST_FILE, default=_DEFAULT_WATCHLIST)
    symbols = [s for s in symbols if s != symbol]
    write_json(_WATCHLIST_FILE, symbols)
    return jsonify({'symbols': symbols})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
