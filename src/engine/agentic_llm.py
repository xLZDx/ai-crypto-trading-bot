import os
import json
import logging
import re
import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_VALID_DECISIONS = {"APPROVED", "REJECTED"}


class AgenticLLM:
    """LLM-based risk manager that vetos trades on severe macro/news risk."""

    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        load_dotenv(os.path.join(project_root, '.env'))
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.is_active = False
        self.model = None

        if self.api_key and self.api_key != "your_api_key_here":
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.is_active = True
                logger.info("Agentic LLM initialized (gemini-2.5-flash).")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("GEMINI_API_KEY not set. Agentic LLM disabled — trades auto-approved.")

    def evaluate_trade(self, symbol: str, action: str, technical_reason: str, headlines: list) -> tuple:
        if not self.is_active or self.model is None:
            return "APPROVED", "Agent disabled (no API key), trade auto-approved."

        news_text = "\n".join(headlines[:20]) if headlines else "No recent news."
        prompt = (
            f"You are a strict AI Risk Manager for a crypto hedge fund.\n"
            f"The quantitative system wants to execute a {action} order for {symbol}.\n"
            f"Technical justification: {technical_reason}\n\n"
            f"Recent market news headlines:\n{news_text}\n\n"
            f"VETO (REJECT) the trade only if news indicates a severe crash, hack, regulatory ban, "
            f"or massive macroeconomic risk. Otherwise APPROVE it.\n"
            f'Respond ONLY in valid JSON: {{"decision": "APPROVED" or "REJECTED", "reason": "1 short sentence"}}'
        )

        try:
            response = self.model.generate_content(prompt, request_options={"timeout": 20})
            raw_text = response.text

            match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
            if not match:
                logger.warning(f"Agentic LLM returned non-JSON response: {raw_text[:200]}")
                return "APPROVED", "LLM response unparseable — trade auto-approved."

            data = json.loads(match.group(0))

            decision = data.get("decision", "").upper()
            reason = data.get("reason", "No reason provided.")

            if decision not in _VALID_DECISIONS:
                logger.warning(f"Unexpected LLM decision value '{decision}' — defaulting to APPROVED.")
                decision = "APPROVED"

            return decision, reason

        except json.JSONDecodeError as e:
            logger.error(f"Agentic LLM JSON parse error: {e}")
            return "APPROVED", f"LLM JSON error: {e}"
        except Exception as e:
            logger.error(f"Agentic LLM Error: {e}")
            return "APPROVED", f"LLM connection error: {e}"
