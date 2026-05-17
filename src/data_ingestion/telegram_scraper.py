import os
import sys
import asyncio
import pandas as pd
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

async def scrape_channel_history():
    """
    Scrapes historical messages from a specified Telegram channel and saves them to a CSV.
    Requires telethon and configuration in the .env file.
    """
    try:
        from telethon.sync import TelegramClient
    except ImportError:
        logger.error("Telethon is not installed. Please run: pip install telethon pandas")
        return

    load_dotenv()
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    session_name = os.getenv("TELEGRAM_SESSION_NAME", "telegram_session")
    channel_name = os.getenv("TELEGRAM_CHANNEL_NAME")
    limit = int(os.getenv("TELEGRAM_HISTORY_LIMIT", 10000))

    if not all([api_id, api_hash, channel_name]):
        logger.error("Missing TELEGRAM_API_ID, TELEGRAM_API_HASH, or TELEGRAM_CHANNEL_NAME in .env file.")
        return

    output_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'telegram_analytics.csv')

    logger.info(f"Connecting to Telegram as user session '{session_name}'...")
    
    messages_data = []
    
    async with TelegramClient(session_name, int(api_id), api_hash) as client:
        logger.info(f"Successfully connected. Scraping last {limit} messages from '{channel_name}'...")
        try:
            async for message in client.iter_messages(channel_name, limit=limit):
                if message.text:
                    messages_data.append({"timestamp": message.date.strftime('%Y-%m-%d %H:%M:%S'), "message_id": message.id, "text": message.text.replace('\n', ' ')})
        except Exception as e:
            logger.error(f"Error scraping messages (is '{channel_name}' a valid channel username/ID?): {e}")
            return

    df = pd.DataFrame(messages_data).sort_values('timestamp')
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"OK  Successfully scraped {len(df)} messages and saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if sys.platform == "win32": asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(scrape_channel_history())