import os
import sys
import json
import threading
import time
from mcp.server.fastmcp import FastMCP

# Add project root to path for imports to work correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.engine.train_model import train_model
from src.data_ingestion.binance_downloader import download_history
from src.tools.web_scraper_bot import get_youtube_transcript, get_article_text

# Initialize FastMCP server
mcp = FastMCP("AITrader_MCP")

def update_data_and_train():
    print("--- [MCP] Downloading fresh 1-day candles for ML training ---", file=sys.stderr)
    download_history(symbol='BTC/USDT', timeframe='1d', limit=1000)
    print("--- [MCP] Data updated. Starting model training... ---", file=sys.stderr)
    train_model()

@mcp.tool()
def trigger_training() -> str:
    """Downloads the latest real-time daily data and triggers Random Forest ML model training."""
    try:
        update_data_and_train()
        return "Real-time data downloaded and model training completed successfully."
    except Exception as e:
        return f"Error during training: {str(e)}"

@mcp.tool()
def get_bot_status() -> str:
    """Retrieves the current status, balances, and signals from the live trading bot."""
    state_path = os.path.join(project_root, 'data', 'state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        return json.dumps(state, indent=2, ensure_ascii=False)
    return "Bot state not found."

@mcp.tool()
def extract_text_from_url(url: str) -> str:
    """Extracts text content from a YouTube video (via subtitles) or a web article. Use this when the user asks to analyze a specific link."""
    try:
        if 'youtube.com' in url or 'youtu.be' in url:
            return get_youtube_transcript(url)
        else:
            return get_article_text(url)
    except Exception as e:
        return f"Error extracting text from {url}: {e}"

def background_training_loop():
    """Background process: fetches fresh candles and retrains the model every 6 hours."""
    while True:
        time.sleep(21600)  # Sleep 6 hours
        try:
            update_data_and_train()
            print("--- [MCP] Background training completed ---", file=sys.stderr)
        except Exception as e:
            print(f"Background training error: {e}", file=sys.stderr)

if __name__ == "__main__":
    print("Starting AI Trader MCP Server...", file=sys.stderr)
    
    # Start background training loop in a separate thread
    training_thread = threading.Thread(target=background_training_loop, daemon=True)
    training_thread.start()
    
    # Start MCP server via stdio
    mcp.run()