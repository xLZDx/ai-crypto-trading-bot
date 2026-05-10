"""FRED (St. Louis Fed) macro indicators — free with API key."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.data_governance.base import DataSourceConnector, ConnectorMeta
from src.data_governance.registry import register

logger = logging.getLogger(__name__)


@register
class FREDConnector(DataSourceConnector):
    META = ConnectorMeta(
        name="fred", host="api.stlouisfed.org",
        priority=1, requires_auth=True,
        default_poll_interval_sec=14400,
        category="macro",
        description="FRED macro series (DXY, VIX, US10Y, gold, oil, M2). "
                    "Set FRED_API_KEY env var (free at fred.stlouisfed.org).",
    )
    BASE = "https://api.stlouisfed.org/fred/series/observations"

    # Top picks for crypto-correlation features.
    SERIES = {
        "DTWEXBGS":   "dxy",         # Trade-weighted USD
        "VIXCLS":     "vix",         # CBOE Volatility Index
        "DGS10":      "us_10y",      # 10-year Treasury yield
        "DCOILWTICO": "wti_oil",
        "GOLDAMGBD228NLBM": "gold",
        "M2NS":       "m2",          # Money supply
    }

    def is_available(self) -> bool:
        return bool(os.getenv("FRED_API_KEY"))

    def pull_history(self, *, since=None, **kw) -> int:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            logger.info("[fred] no FRED_API_KEY — skipping.")
            return 0
        qdb = self._qdb()
        rows_total = 0
        obs_start = (since or "2017-01-01")
        if hasattr(obs_start, "strftime"):
            obs_start = obs_start.strftime("%Y-%m-%d")
        for sid, alias in self.SERIES.items():
            url = (f"{self.BASE}?series_id={sid}&api_key={api_key}"
                   f"&file_type=json&observation_start={obs_start}")
            r = self._http_get(url)
            if r is None or r.status_code != 200:
                continue
            obs = (r.json() or {}).get("observations") or []
            for o in obs:
                try:
                    val = float(o["value"])
                except (ValueError, TypeError):
                    continue
                ts_ms = int(datetime.fromisoformat(o["date"])
                            .replace(tzinfo=timezone.utc).timestamp() * 1000)
                qdb.write_signal(f"fred_{alias}", {"value": val}, ts_val=ts_ms)
                rows_total += 1
        logger.info("[fred] %d observations written", rows_total)
        return rows_total
