import os
import asyncio
import threading
import logging
from collections import deque
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class TelegramMonitor:
    """
    Connects to Telegram in a background thread to listen for new messages
    from multiple channels simultaneously, caching them for the AgenticLLM.

    Channels can be passed directly (e.g. ['VilarsoPro', 'mr_mozart']) or
    configured via TELEGRAM_CHANNEL_NAME env variable (single channel fallback).
    Messages are tagged with the source channel name.
    """

    def __init__(self, channels: list = None, cache_size: int = 30):
        load_dotenv()
        self.api_id = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.session_name = os.getenv("TELEGRAM_SESSION_NAME", "telegram_session")

        # Multi-channel list: prefer explicit arg, fall back to env single-channel
        env_channel = os.getenv("TELEGRAM_CHANNEL_NAME")
        if channels:
            self.channels = list(channels)
        elif env_channel:
            self.channels = [env_channel]
        else:
            self.channels = []

        self.is_active = all([self.api_id, self.api_hash]) and bool(self.channels)
        self.recent_messages: deque = deque(maxlen=cache_size)
        self._is_running = False
        self.lock = threading.Lock()

    async def _event_handler(self, event):
        """Callback for new messages from any monitored channel."""
        message_text = event.message.text
        if not message_text:
            return
        try:
            chat = await event.get_chat()
            source = getattr(chat, 'username', None) or getattr(chat, 'title', 'Telegram')
        except Exception:
            source = 'Telegram'

        tagged = f"[{source}] {message_text}"
        with self.lock:
            self.recent_messages.append(tagged)
        logger.info(f"Telegram Monitor: new message from '{source}'")

    async def _run_client(self):
        from telethon import TelegramClient, events
        async with TelegramClient(self.session_name, int(self.api_id), self.api_hash) as client:
            client.add_event_handler(
                self._event_handler,
                events.NewMessage(chats=self.channels)
            )
            logger.info(f"Telegram Monitor listening to channels: {self.channels}")
            await client.run_until_disconnected()

    def _start_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_client())
        except Exception as e:
            logger.error(f"Telegram Monitor thread failed: {e}")

    def start(self):
        if not self.is_active:
            logger.warning(
                "Telegram Monitor disabled — missing TELEGRAM_API_ID/TELEGRAM_API_HASH "
                "in .env or no channels configured."
            )
            return
        self._is_running = True
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def get_recent_messages(self) -> list:
        with self.lock:
            return list(self.recent_messages)
