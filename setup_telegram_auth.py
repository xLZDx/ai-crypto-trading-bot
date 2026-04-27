"""
One-time Telegram authorization.
Run this ONCE manually in a terminal to create the session file.
After this, the bot will connect automatically without prompts.

Usage:
    .\\venv\\Scripts\\python.exe setup_telegram_auth.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

API_ID   = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION  = os.getenv("TELEGRAM_SESSION_NAME", "trading_session")

if not API_ID or not API_HASH:
    print("ERROR: TELEGRAM_API_ID or TELEGRAM_API_HASH missing from .env")
    sys.exit(1)

CHANNELS = ['VilarsoPro', 'vilarsofree', 'mr_mozart']

async def main():
    try:
        from telethon import TelegramClient
    except ImportError:
        print("ERROR: telethon not installed. Run: pip install telethon")
        sys.exit(1)

    print(f"\nConnecting to Telegram (session: {SESSION})...")
    async with TelegramClient(SESSION, int(API_ID), API_HASH) as client:
        me = await client.get_me()
        print(f"\n✅ Authorized as: {me.first_name} (@{me.username})")
        print(f"Session file saved: {SESSION}.session")
        print("\nVerifying channel access...")
        for ch in CHANNELS:
            try:
                entity = await client.get_entity(ch)
                print(f"  ✅ {ch} — accessible ({getattr(entity, 'title', ch)})")
            except Exception as e:
                print(f"  ⚠️  {ch} — {e}")
        print("\nSetup complete. You can now start the bot normally.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
