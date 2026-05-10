"""
GovernanceConfig — reads `data/data_governance.json`.

Schema:
{
  "sources": {
    "bybit":          {"enabled": true,  "priority": 0, "poll_sec": 300},
    "okx":            {"enabled": true,  "priority": 0, "poll_sec": 300},
    "coingecko":      {"enabled": true,  "priority": 0, "poll_sec": 600},
    "coinbase":       {"enabled": true,  "priority": 0, "poll_sec": 300},
    "kraken":         {"enabled": true,  "priority": 0, "poll_sec": 300},
    "fear_greed":     {"enabled": true,  "priority": 0, "poll_sec": 3600},
    "fred":           {"enabled": true,  "priority": 1, "poll_sec": 14400},
    "defillama":      {"enabled": true,  "priority": 1, "poll_sec": 3600},
    "cryptocompare_news": {"enabled": true, "priority": 1, "poll_sec": 600},
    "coinglass":      {"enabled": false, "priority": 2, "poll_sec": 600},
    "reddit":         {"enabled": false, "priority": 2, "poll_sec": 1800}
  },
  "history_days": 365,
  "store_local":  true,
  "google_drive_archive": false
}

Missing entries default to enabled=True with priority=1 + poll=3600.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "data_governance.json"


@dataclass
class SourceSetting:
    enabled:  bool = True
    priority: int  = 1
    poll_sec: int  = 3600


@dataclass
class GovernanceConfig:
    sources:        dict[str, SourceSetting] = field(default_factory=dict)
    history_days:   int  = 365
    store_local:    bool = True
    google_drive_archive: bool = False
    config_path:    Path = DEFAULT_CONFIG_PATH

    @classmethod
    def load(cls, path: Path | None = None) -> "GovernanceConfig":
        path = path or DEFAULT_CONFIG_PATH
        if not path.exists():
            cfg = cls.default()
            cfg.save(path)
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("could not parse %s (%s) — using defaults", path, exc)
            return cls.default()
        sources = {
            name: SourceSetting(**s) for name, s in (data.get("sources") or {}).items()
        }
        return cls(
            sources=sources,
            history_days=int(data.get("history_days", 365)),
            store_local=bool(data.get("store_local", True)),
            google_drive_archive=bool(data.get("google_drive_archive", False)),
            config_path=path,
        )

    def save(self, path: Path | None = None) -> None:
        path = path or self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sources": {n: s.__dict__ for n, s in self.sources.items()},
            "history_days":         self.history_days,
            "store_local":          self.store_local,
            "google_drive_archive": self.google_drive_archive,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self, name: str) -> SourceSetting:
        return self.sources.get(name, SourceSetting())

    @classmethod
    def default(cls) -> "GovernanceConfig":
        # Tier-1 (no auth): on by default. Tier-2 (auth): off by default.
        defaults = {
            "bybit":              SourceSetting(True,  0, 300),
            "okx":                SourceSetting(True,  0, 300),
            "coinbase":           SourceSetting(True,  0, 300),
            "kraken":             SourceSetting(True,  0, 300),
            "coingecko":          SourceSetting(True,  0, 600),
            "fear_greed":         SourceSetting(True,  0, 3600),
            "fred":               SourceSetting(True,  1, 14400),
            "defillama":          SourceSetting(True,  1, 3600),
            "cryptocompare_news": SourceSetting(True,  1, 600),
            "coinglass":          SourceSetting(False, 2, 600),
            "reddit":             SourceSetting(False, 2, 1800),
        }
        return cls(sources=defaults)


__all__ = ["GovernanceConfig", "SourceSetting", "DEFAULT_CONFIG_PATH"]
