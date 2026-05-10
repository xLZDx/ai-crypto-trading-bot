"""Etherscan large-transfer monitor — free 5 req/sec.

Uses `txlist` for a watchlist of "smart-money" wallets at
`data/etherscan_wallets.json`. Each tx is written as a `whale_tx` signal.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WATCHLIST    = PROJECT_ROOT / "data" / "etherscan_wallets.json"


@register
class EtherscanConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="etherscan", host="api.etherscan.io",
        priority=2, requires_auth=True,
        default_poll_interval_sec=900,
        category="onchain",
        description="Etherscan watchlist-wallet large transfers. "
                    "Set ETHERSCAN_API_KEY env var (free 5 req/sec).",
    )
    BASE = "https://api.etherscan.io/api"

    def is_available(self) -> bool:
        return bool(os.getenv("ETHERSCAN_API_KEY")) and WATCHLIST.exists()

    def _wallets(self) -> list[str]:
        try:
            return list(json.loads(WATCHLIST.read_text(encoding="utf-8")) or [])
        except Exception:
            return []

    def pull_history(self, **kw) -> int:
        api_key = os.getenv("ETHERSCAN_API_KEY")
        if not api_key:
            return 0
        qdb = self._qdb()
        n = 0
        for addr in self._wallets():
            url = (f"{self.BASE}?module=account&action=txlist&address={addr}"
                   f"&page=1&offset=20&sort=desc&apikey={api_key}")
            r = self._http_get(url)
            if r is None or r.status_code != 200:
                continue
            for tx in (r.json() or {}).get("result") or []:
                try:
                    val_eth = float(int(tx["value"]) / 1e18)
                except Exception:
                    continue
                if val_eth < 100:    # only "whale" txs
                    continue
                ts_ms = int(tx["timeStamp"]) * 1000
                qdb.write_signal(f"etherscan_whale_{addr[:10]}",
                                 {"value_eth": val_eth},
                                 ts_val=ts_ms)
                n += 1
        logger.info("[etherscan] %d whale txs written", n)
        return n
