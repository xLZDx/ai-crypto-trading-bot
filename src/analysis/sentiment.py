import urllib.request
import xml.etree.ElementTree as ET
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import time

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://news.bitcoin.com/feed/",
            "https://coinjournal.net/news/feed/",
            "https://www.newsbtc.com/feed/",
            "https://cryptoslate.com/feed/",
            "https://bitcoinist.com/feed/"
        ]
        self.cached_sentiment = 0.0
        self.last_fetch_time = 0
        self.cached_headlines = []

    def fetch_news(self):
        headlines = []
        for url in self.rss_feeds:
            try:
                # Simulate a real browser so anti-bot filters don't block us
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    xml_data = response.read()
                    root = ET.fromstring(xml_data)
                    for item in root.findall('.//item'):
                        title = item.find('title')
                        if title is not None and title.text:
                            headlines.append(title.text)
            except Exception as e:
                logger.warning(f"Error loading news from {url}: {e}")
        return headlines

    def get_average_sentiment(self):
        current_time = time.time()
        # Cache result for 15 minutes (900 seconds) to avoid IP blocks from RSS servers
        if current_time - self.last_fetch_time < 900:
            return self.cached_sentiment

        headlines = self.fetch_news()
        if not headlines:
            logger.warning("Failed to load news. Sentiment = 0.0")
            self.last_fetch_time = current_time # Update timer even on error to avoid spamming
            self.cached_sentiment = 0.0
            return 0.0
        
        scores = [self.analyzer.polarity_scores(hl)['compound'] for hl in headlines]
        self.cached_headlines = headlines
        self.cached_sentiment = sum(scores) / len(scores)
        self.last_fetch_time = current_time
        
        logger.info(f"Successfully loaded {len(headlines)} headlines. Overall sentiment: {self.cached_sentiment:.2f}")
        return self.cached_sentiment