"""DeFiLlama — DeFi TVL by chain & protocol. Free, no auth."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class DefiLlamaConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="defillama", host="api.llama.fi",
        priority=1, requires_auth=False,
        default_poll_interval_sec=3600,
        category="onchain",
        description="DeFi TVL across all chains; total + per-chain breakdown.",
    )
    BASE = "https://api.llama.fi"

    CHAINS = ("Ethereum", "Bitcoin", "Solana", "Arbitrum",
              "BSC", "Avalanche", "Polygon")

    def is_available(self) -> bool:
        return True

    def pull_history(self, **kw) -> int:
        qdb = self._qdb()
        # Snapshot of total TVL by chain — point-in-time, written each poll.
        r = self._http_get(f"{self.BASE}/v2/chains")
        if r is None or r.status_code != 200:
            return 0
        chains = r.json() or []
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        n = 0
        for c in chains:
            name = c.get("name")
            tvl  = c.get("tvl")
            if name in self.CHAINS and tvl is not None:
                qdb.write_signal(f"tvl_{name.lower()}",
                                 {"tvl_usd": float(tvl)}, ts_val=ts_ms)
                n += 1
        logger.info("[defillama] %d TVL points written", n)
        return n
