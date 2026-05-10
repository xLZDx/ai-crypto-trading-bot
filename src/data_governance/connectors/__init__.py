"""All registered connector implementations.

Importing this package side-effect-registers every connector with the
governance REGISTRY. Add new connectors as `connectors/<name>.py` and
import them below.
"""
# Tier 0/1 (free, no auth or optional auth)
from . import bybit, okx, coinbase, kraken, coingecko, fear_greed, fred, defillama
from . import cryptocompare_news, theblock_rss
# Tier 2 (auth required)
from . import coinglass, reddit, glassnode, santiment, newsapi
from . import youtube_transcripts, etherscan

__all__ = []
