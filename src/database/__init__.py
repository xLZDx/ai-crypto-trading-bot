"""
Trading bot database layer — QuestDB time-series store.

Quick start:
    docker-compose up -d questdb          # start container
    python -m src.database.schema         # create tables
    python -m src.database.ingest_pipeline --symbol BTC/USDT  # import CSV.gz
"""
from src.database.questdb_client import QuestDBClient, get_client

__all__ = ["QuestDBClient", "get_client"]
