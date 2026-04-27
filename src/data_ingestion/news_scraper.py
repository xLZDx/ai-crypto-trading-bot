"""
Historical news feed pipeline for Deep Learning (TFT/LSTM) training.
Integrates CryptoCompare (Primary), RSS/news fallbacks, and Kaggle bootstrap data.
"""
import json
import logging
import os
import time
import urllib.request
from typing import List, Optional

import pandas as pd
import requests
import xml.etree.ElementTree as ET
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://news.bitcoin.com/feed/",
    "https://coinjournal.net/news/feed/",
    "https://www.newsbtc.com/feed/",
    "https://cryptoslate.com/feed/",
    "https://bitcoinist.com/feed/",
]

STANDARD_COLUMNS = [
    "source",
    "source_detail",
    "categories",
    "title",
    "summary",
    "url",
    "published_at",
]


class NewsScraper:
    def __init__(self, cryptocompare_api_key: str = None):
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=True)
            logger.info("Loaded environment variables from %s", env_path)
        else:
            logger.warning("Could not find a .env file. Continuing without a CryptoCompare key.")

        self.api_key = cryptocompare_api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
        
        if not self.api_key:
            print("\n" + "="*60)
            print("🔑 CryptoCompare API Key is missing or not readable!")
            user_key = input("Please paste your API key here (or press Enter to skip): ").strip('"\' ')
            if user_key:
                self.api_key = user_key
                target_env = env_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
                try:
                    with open(target_env, 'a', encoding='utf-8') as f:
                        f.write(f'\nCRYPTOCOMPARE_API_KEY="{self.api_key}"\n')
                    print(f"✅ Key automatically saved to {target_env}!")
                except Exception as e:
                    print(f"⚠️ Could not save to {target_env}: {e}")
            print("="*60 + "\n")

        if self.api_key:
            self.api_key = self.api_key.strip('"\' ')
            logger.info("CryptoCompare API key loaded.")
        else:
            logger.warning(
                "No CRYPTOCOMPARE_API_KEY found. CryptoCompare will be skipped and RSS/news fallbacks will be used."
            )

        self.cc_url = "https://min-api.cryptocompare.com/data/v2/news/"
        self.rss_feeds = RSS_FEEDS

    @staticmethod
    def _clean_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    @staticmethod
    def _normalize_categories(categories: str) -> List[str]:
        return [term.strip().lower() for term in categories.split(",") if term.strip()]

    def _empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    def _finalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return self._empty_frame()

        result = frame.copy()

        for column in STANDARD_COLUMNS:
            if column not in result.columns:
                result[column] = ""

        result["published_at"] = pd.to_datetime(result["published_at"], errors="coerce", utc=True)
        result = result[STANDARD_COLUMNS]

        sort_key = result["published_at"]
        if sort_key.notna().any():
            result = result.sort_values(by="published_at", ascending=False, na_position="last")

        result = result.drop_duplicates(subset=["title", "url", "published_at"], keep="first")
        return result.reset_index(drop=True)

    def fetch_cryptocompare(self, categories: str = "BTC,ETH", days_back: int = 3650) -> pd.DataFrame:
        """Fetches historical news from CryptoCompare."""
        if not self.api_key:
            logger.warning("Skipping CryptoCompare fetch because no API key is configured.")
            return self._empty_frame()

        logger.info("Fetching CryptoCompare news for %s (up to %d days back)", categories, days_back)
        headers = {"authorization": f"Apikey {self.api_key}"}
        
        target_ts = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp())
        current_lts = None
        rows = []

        try:
            while True:
                params = {"categories": categories, "excludeCategories": "Sponsored"}
                params["api_key"] = self.api_key  # Fallback authentication
                if current_lts:
                    params["lTs"] = current_lts

                response = requests.get(self.cc_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()

                json_resp = response.json()
                data = json_resp.get("Data", [])

                if not data:
                    if not rows:
                        logger.warning(
                            "CryptoCompare returned no articles. Message: %s",
                            json_resp.get("Message", "Unknown response"),
                        )
                    break

                for item in data:
                    # Avoid duplicates on boundary
                    if current_lts and item.get("published_on") == current_lts:
                        continue
                    rows.append({
                        "source": "cryptocompare",
                        "source_detail": self._clean_text(item.get("source")),
                        "categories": categories,
                        "title": self._clean_text(item.get("title")),
                        "summary": self._clean_text(item.get("body")),
                        "url": self._clean_text(item.get("url")),
                        "published_at": pd.to_datetime(
                            item.get("published_on"), unit="s", errors="coerce", utc=True
                        ),
                    })

                last_ts = data[-1].get("published_on")
                
                # Break if we reached the target date or if API stops paginating backward
                if not last_ts or last_ts <= target_ts or last_ts == current_lts:
                    break
                    
                current_lts = last_ts
                
                if len(rows) % 500 < 50:  # Print progress roughly every 500 articles
                    dt_str = pd.to_datetime(current_lts, unit="s").strftime("%Y-%m-%d")
                    logger.info("... downloaded %d articles so far (reached %s)", len(rows), dt_str)
                    
                time.sleep(0.1)  # Rate limit protection (10 req/sec)

            return self._finalize_frame(pd.DataFrame(rows))
        except Exception as e:
            logger.error("Failed to fetch news (saving what was downloaded): %s", e)
            return self._finalize_frame(pd.DataFrame(rows)) if rows else self._empty_frame()

    def fetch_rss_news(
        self,
        categories: str = "BTC,ETH",
        max_items_per_feed: int = 50,
        allow_unfiltered_fallback: bool = True,
    ) -> pd.DataFrame:
        """Fetches news headlines from RSS feeds as a fallback source."""
        terms = self._normalize_categories(categories)
        rows = []

        for feed_url in self.rss_feeds:
            try:
                request = urllib.request.Request(
                    feed_url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        )
                    },
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    xml_data = response.read()

                root = ET.fromstring(xml_data)
                matched_rows = []

                for item in root.findall(".//item"):
                    title = self._clean_text(item.findtext("title"))
                    summary = self._clean_text(item.findtext("description"))
                    url = self._clean_text(item.findtext("link"))
                    published_raw = self._clean_text(item.findtext("pubDate") or item.findtext("published"))
                    text_blob = f"{title} {summary}".lower()

                    if terms and not any(term in text_blob for term in terms):
                        continue

                    matched_rows.append(
                        {
                            "source": "rss",
                            "source_detail": feed_url,
                            "categories": categories,
                            "title": title,
                            "summary": summary,
                            "url": url,
                            "published_at": pd.to_datetime(published_raw, errors="coerce", utc=True),
                        }
                    )

                    if len(matched_rows) >= max_items_per_feed:
                        break

                if matched_rows:
                    logger.info("Fetched %d RSS headlines from %s", len(matched_rows), feed_url)
                    rows.extend(matched_rows)
                else:
                    logger.info("No category-matched RSS headlines found in %s", feed_url)

            except Exception as e:
                logger.warning("Error loading news from %s: %s", feed_url, e)

        if not rows and terms and allow_unfiltered_fallback:
            logger.info("No category-matched RSS headlines found. Retrying without category filter.")
            return self.fetch_rss_news(
                categories="",
                max_items_per_feed=max_items_per_feed,
                allow_unfiltered_fallback=False,
            )

        return self._finalize_frame(pd.DataFrame(rows) if rows else self._empty_frame())

    def fetch_twitter_sentiment(self, query: str, limit: int = 1000) -> pd.DataFrame:
        """Fetches retail sentiment from Twitter/X using snscrape."""
        logger.info("Fetching tweets for %s", query)
        logger.warning("snscrape integration requires the library to be installed and configured. Using fallback.")
        return self._empty_frame()

    def load_kaggle_baseline(self, csv_path: str) -> pd.DataFrame:
        """Loads pre-compiled CSVs to bootstrap historical data up to 2023/2024."""
        logger.info("Loading baseline Kaggle dataset from %s", csv_path)
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.error("Error loading Kaggle CSV: %s", e)
            return self._empty_frame()

    def build_news_dataset(
        self,
        categories: str = "BTC,ETH",
        baseline_csv_path: Optional[str] = None,
        days_back: int = 3650
    ) -> pd.DataFrame:
        """Builds a single normalized news dataset using the available sources."""
        frames = []

        cryptocompare_df = self.fetch_cryptocompare(categories=categories, days_back=days_back)
        if not cryptocompare_df.empty:
            frames.append(cryptocompare_df)

        rss_df = self.fetch_rss_news(categories=categories)
        if not rss_df.empty:
            frames.append(rss_df)

        if baseline_csv_path and os.path.exists(baseline_csv_path):
            baseline_df = self.load_kaggle_baseline(baseline_csv_path)
            if not baseline_df.empty:
                baseline = baseline_df.copy()
                for column in STANDARD_COLUMNS:
                    if column not in baseline.columns:
                        baseline[column] = ""
                if baseline["source"].eq("").all():
                    baseline["source"] = "kaggle"
                if baseline["source_detail"].eq("").all():
                    baseline["source_detail"] = baseline_csv_path
                if baseline["categories"].eq("").all():
                    baseline["categories"] = categories
                if baseline["title"].eq("").all():
                    for candidate in ("headline", "news_title", "name"):
                        if candidate in baseline.columns:
                            baseline["title"] = baseline[candidate].fillna("")
                            break
                if baseline["summary"].eq("").all():
                    for candidate in ("description", "text", "body"):
                        if candidate in baseline.columns:
                            baseline["summary"] = baseline[candidate].fillna("")
                            break
                if baseline["url"].eq("").all():
                    for candidate in ("link", "source_url", "article_url"):
                        if candidate in baseline.columns:
                            baseline["url"] = baseline[candidate].fillna("")
                            break
                if baseline["published_at"].eq("").all():
                    for candidate in ("published_at", "date", "published", "timestamp"):
                        if candidate in baseline.columns:
                            baseline["published_at"] = baseline[candidate]
                            break
                frames.append(self._finalize_frame(baseline[STANDARD_COLUMNS]))

        if not frames:
            return self._empty_frame()

        combined = pd.concat(frames, ignore_index=True, sort=False)
        return self._finalize_frame(combined)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    scraper = NewsScraper()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    watchlist_path = os.path.join(project_root, "data", "watchlist.json")

    categories = "BTC,ETH"
    if os.path.exists(watchlist_path):
        try:
            with open(watchlist_path, "r", encoding="utf-8") as f:
                symbols = json.load(f)
                base_coins = [sym.split("/")[0] for sym in symbols]
                categories = ",".join(list(dict.fromkeys(base_coins)))
        except Exception as e:
            logger.error("Failed to read watchlist: %s", e)

    baseline_candidates = [
        os.path.join(project_root, "data", "raw", "kaggle_news.csv"),
        os.path.join(project_root, "data", "raw", "news.csv"),
        os.path.join(project_root, "data", "raw", "news_baseline.csv"),
    ]
    baseline_csv_path = next((path for path in baseline_candidates if os.path.exists(path)), None)

    news_df = scraper.build_news_dataset(categories=categories, baseline_csv_path=baseline_csv_path, days_back=3650)

    output_path = os.path.join(project_root, "data", "raw", "cryptocompare_news.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if news_df.empty:
        logger.warning(
            "No news data was collected from CryptoCompare, RSS, or Kaggle fallback. "
            "Writing an empty CSV with the standard headers."
        )
        news_df = pd.DataFrame(columns=STANDARD_COLUMNS)

    news_df.to_csv(output_path, index=False, encoding="utf-8")

    if not news_df.empty:
        logger.info("✅ Successfully grabbed and saved %d articles to %s", len(news_df), output_path)
    else:
        logger.info("✅ Created empty placeholder CSV at %s", output_path)
