import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion.news_scraper import NewsScraper

scraper = NewsScraper()

watchlist_path = os.path.join(PROJECT_ROOT, "data", "watchlist.json")
categories = "BTC,ETH"

if os.path.exists(watchlist_path):
    try:
        with open(watchlist_path, "r", encoding="utf-8") as f:
            symbols = json.load(f)
            base_coins = [symbol.split("/")[0] for symbol in symbols]
            categories = ",".join(list(dict.fromkeys(base_coins)))
    except Exception:
        pass

baseline_candidates = [
    os.path.join(PROJECT_ROOT, "data", "raw", "kaggle_news.csv"),
    os.path.join(PROJECT_ROOT, "data", "raw", "news.csv"),
    os.path.join(PROJECT_ROOT, "data", "raw", "news_baseline.csv"),
]
baseline_csv_path = next((path for path in baseline_candidates if os.path.exists(path)), None)

df = scraper.build_news_dataset(
    categories=categories,
    baseline_csv_path=baseline_csv_path,
    days_back=3650,
)

output_path = os.path.join(PROJECT_ROOT, "data", "raw", "cryptocompare_news.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8")

print(df.columns.tolist())
print(df.shape)
print(output_path)
