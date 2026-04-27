import os
import json
import logging
import re
from dotenv import load_dotenv
from src.utils.safe_json import read_json

logger = logging.getLogger(__name__)

_VALID_DECISIONS = {"APPROVED", "REJECTED"}
# Includes model-not-found codes so the loop falls through to the next model
# instead of breaking on stale/unavailable model IDs.
_TRANSIENT = [
    '429', 'quota', 'resource_exhausted',
    '503', 'unavailable', 'high demand', 'overloaded',
    '404', 'not found', 'invalid argument', 'unknown model',
]

# Lightest-quota models first; gives free-tier keys the best chance.
_ALL_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
]


class AgenticLLM:
    """LLM-based risk manager that vetos trades on severe macro/news risk."""

    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        load_dotenv(os.path.join(project_root, '.env'))
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.is_active = False
        self._client = None

        if self.api_key and self.api_key != "your_api_key_here":
            try:
                from google import genai as _genai
                self._client = _genai.Client(api_key=self.api_key)
                self.is_active = True
                logger.info("Agentic LLM initialized (google.genai, gemini-3.1-pro-preview).")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("GEMINI_API_KEY not set. Agentic LLM disabled — trades auto-approved.")

    def evaluate_trade(self, symbol: str, action: str, technical_reason: str, headlines: list, telegram_monitor=None) -> tuple:
        if not self.is_active or self._client is None:
            return "APPROVED", "Agent disabled (no API key), trade auto-approved."

        news_text = "\n".join(headlines[:20]) if headlines else "No recent news."

        telegram_text = "No recent Telegram analysis."
        if telegram_monitor and telegram_monitor.is_active:
            tg_messages = telegram_monitor.get_recent_messages()
            if tg_messages:
                telegram_text = "\n".join(f"- {msg}" for msg in tg_messages)

        prompt = (
            f"You are a strict AI Risk Manager for a crypto hedge fund.\n"
            f"The quantitative system wants to execute a {action} order for {symbol}.\n"
            f"Technical justification: {technical_reason}\n\n"
            f"Recent market news headlines:\n{news_text}\n\n"
            f"Proprietary Telegram Channel Analysis (Secondary source — use only if consistent with technicals):\n"
            f"{telegram_text}\n\n"
            f"VETO (REJECT) the trade only if news indicates a severe crash, hack, regulatory ban, "
            f"or massive macroeconomic risk. Otherwise APPROVE it.\n"
            f'Respond ONLY in valid JSON: {{"decision": "APPROVED" or "REJECTED", "reason": "1 short sentence"}}'
        )

        from google.genai import types as _gntypes
        
        ctrl = read_json('data/control.json', default={})
        selected_model = ctrl.get('selected_ai_model')
        models_to_try = [selected_model] if selected_model else _ALL_MODELS
        
        last_err = None
        for model_id in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=_gntypes.GenerateContentConfig()
                )
                raw_text = response.text
                match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
                if not match:
                    logger.warning(f"Agentic LLM non-JSON response: {raw_text[:200]}")
                    return "APPROVED", "LLM response unparseable — trade auto-approved."
                data = json.loads(match.group(0))
                decision = data.get("decision", "").upper()
                reason = data.get("reason", "No reason provided.")
                if decision not in _VALID_DECISIONS:
                    logger.warning(f"Unexpected LLM decision '{decision}' — defaulting to APPROVED.")
                    decision = "APPROVED"
                return decision, reason
            except Exception as e:
                last_err = e
                if any(x in str(e).lower() for x in _TRANSIENT):
                    logger.warning(f"Agentic LLM: {model_id} transient error, trying fallback...")
                    continue
                break

        logger.error(f"Agentic LLM Error: {last_err}")
        return "APPROVED", f"LLM connection error — trade auto-approved."
