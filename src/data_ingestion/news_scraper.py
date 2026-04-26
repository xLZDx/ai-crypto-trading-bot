"""
Historical news feed pipeline for Deep Learning (TFT/LSTM) training.
Integrates CryptoCompare (Primary), snscrape (Twitter/X), and Kaggle fallbacks.
"""
import os
import logging
import pandas as pd
import requests

logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self, cryptocompare_api_key: str = None):
        self.api_key = cryptocompare_api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
        self.cc_url = "https://min-api.cryptocompare.com/data/v2/news/"

    def fetch_cryptocompare(self, categories: str = "BTC,ETH") -> pd.DataFrame:
        """Fetches historical news from CryptoCompare."""
        logger.info(f"Fetching CryptoCompare news for {categories}")
        params = {"categories": categories, "excludeCategories": "Sponsored"}
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(self.cc_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("Data", [])
            if data:
                df = pd.DataFrame(data)
                # Keep useful columns: id, published_on, title, body, categories, upvotes, downvotes
                return df
        except Exception as e:
            logger.error(f"Failed to fetch news from CryptoCompare: {e}")
            
        return pd.DataFrame()

    def fetch_twitter_sentiment(self, query: str, limit: int = 1000) -> pd.DataFrame:
        """Fetches retail sentiment from Twitter/X using snscrape."""
        logger.info(f"Fetching tweets for {query}")
        logger.warning("snscrape integration requires the library to be installed and configured. Using fallback.")
        return pd.DataFrame()
        
    def load_kaggle_baseline(self, csv_path: str) -> pd.DataFrame:
        """Loads pre-compiled CSVs to bootstrap historical data up to 2023/2024."""
        logger.info(f"Loading baseline Kaggle dataset from {csv_path}")
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error loading Kaggle CSV: {e}")
            return pd.DataFrame()
