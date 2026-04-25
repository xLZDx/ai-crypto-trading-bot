import json
import os
import sys
import threading
import subprocess
import re
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    try:
        with open('data/state.json', 'r', encoding='utf-8') as f:
            state = json.load(f)
        return jsonify(state)
    except Exception:
        return jsonify({"status": "No data", "last_signal": "UNKNOWN"})

@app.route('/api/control', methods=['GET'])
def get_control():
    try:
        with open('data/control.json', 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({"running": True})

@app.route('/api/control', methods=['POST'])
def set_control():
    try:
        data = request.json
        with open('data/control.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/trades')
def get_trades():
    try:
        with open('data/trades.json', 'r', encoding='utf-8') as f:
            trades = json.load(f)
        return jsonify({"trades": trades})
    except Exception:
        return jsonify({"trades": []})

@app.route('/api/logs')
def get_logs():
    try:
        # Read the last 500 lines of the log
        with open('logs/trading.log', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Remove newline characters and take from the end
            logs = [line.strip() for line in lines[-500:]]
            return jsonify({"logs": logs})
    except Exception:
        return jsonify({"logs": ["No logs yet..."]})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        import google.generativeai as genai
    except ImportError:
        return jsonify({"response": "The google-generativeai library is installing. Restart via restart_all.bat!"})

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return jsonify({"response": "⚠️ **Error:** Add `GEMINI_API_KEY=your_key` to the `.env` file in the project root for AI to work."})
        
    user_message = request.json.get('message', '')
    
    # Tool: Model training trigger
    trigger_words = ["train", "retrain"]
    if any(word in user_message.lower() for word in trigger_words):
        def run_train():
            subprocess.run([sys.executable, "src/engine/train_model.py"])
        threading.Thread(target=run_train).start()
        return jsonify({"response": "⚙️ **Command accepted:** Background ML model retraining started! This will take some time."})

    # Tool: Web scraper (Analyze links to articles and YouTube)
    url_match = re.search(r'(https?://[^\s]+)', user_message)
    if url_match:
        try:
            from src.tools.web_scraper_bot import get_youtube_transcript, get_article_text
            url = url_match.group(1)
            if 'youtube.com' in url or 'youtu.be' in url:
                extracted_text = get_youtube_transcript(url)
            else:
                extracted_text = get_article_text(url)
            
            user_message += f"\n\n[SYSTEM MESSAGE: User sent a link. Text successfully extracted:\n{extracted_text[:30000]}\nPlease analyze this information in the context of crypto trading.]"
        except Exception as e:
            user_message += f"\n\n[SYSTEM MESSAGE: An error occurred extracting text from the link: {e}]"

    # Load bot context for AI
    try:
        with open('data/state.json', 'r', encoding='utf-8') as f: state = f.read()
        with open('data/trades.json', 'r', encoding='utf-8') as f: trades = f.read()
    except Exception:
        state, trades = "No data", "No data"

    system_prompt = f"""You are an AI Assistant (Gemini), a powerful AI embedded in a crypto trading dashboard.
    Your task is to give advice, analyze the market, and help the trader.
    CURRENT BOT STATE: {state}
    TRADE HISTORY: {trades}
    Answer briefly, professionally, in English. Analyze PnL and Signals if the user asks about them."""

    try:
        genai.configure(api_key=api_key)
        # Use a modern fast model
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_prompt)
        response = model.generate_content(user_message)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"Gemini API Error: {str(e)}"})

if __name__ == '__main__':
    # Start dashboard on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
