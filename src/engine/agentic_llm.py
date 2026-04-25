import os
import json
import logging
import re
import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class AgenticLLM:
    """
    Agentic AI (LLM Overlay) acts as a Risk Manager.
    It makes the final decision (VETO) based on news and macroeconomics.
    """
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(project_root, '.env')
        load_dotenv(env_path)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.is_active = False
        
        if self.api_key and self.api_key != "your_api_key_here":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.is_active = True
        else:
            logger.warning("GEMINI_API_KEY not found in .env file. Agentic LLM Overlay is disabled.")

    def evaluate_trade(self, symbol: str, action: str, technical_reason: str, headlines: list) -> tuple:
        if not self.is_active:
            return "APPROVED", "Agent disabled (no API key), trade auto-approved."
            
        prompt = f"""You are a strict AI Risk Manager for a crypto hedge fund.
The quantitative system wants to execute a {action} order for {symbol}.
Technical justification: {technical_reason}

Recent market news headlines:
{chr(10).join(headlines) if headlines else 'No recent news.'}

Your job is to VETO (REJECT) the trade if the news indicates a severe crash, hack, regulatory ban, or massive macroeconomic risk. Otherwise, APPROVE it.
Respond ONLY in valid JSON format exactly like this: {{"decision": "APPROVED" or "REJECTED", "reason": "1 short sentence explanation in English"}}"""
        
        try:
            response = self.model.generate_content(prompt)
            # Clear response of possible Markdown markup
            text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
            data = json.loads(text)
            return data.get("decision", "APPROVED"), data.get("reason", "No reason provided")
        except Exception as e:
            logger.error(f"Agentic LLM Error: {e}")
            return "APPROVED", f"LLM connection error: {e}"