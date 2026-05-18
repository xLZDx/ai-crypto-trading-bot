"""
Champion / Challenger registry for ML models.

Each (model_key, timeframe) cell has at most one champion and one challenger.
When a new model trains it registers as the challenger. If its walk-forward
accuracy beats the current champion by at least PROMOTION_DELTA, it is
automatically promoted; otherwise it stays as the pending challenger until
the next training run or a manual promotion call.

Registry is persisted to data/champion_registry.json via safe_json atomic writes.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "champion_registry.json"
PROMOTION_DELTA = 0.005  # challenger must beat champion by at least 0.5 pp WF acc


def _load() -> dict:
    from src.utils.safe_json import read_json
    return read_json(str(REGISTRY_PATH), default={})


def _save(registry: dict) -> None:
    from src.utils.safe_json import write_json
    write_json(str(REGISTRY_PATH), registry)


class ChampionRegistry:
    """Thread-safe champion/challenger registry backed by champion_registry.json."""

    def register_challenger(
        self,
        model_key: str,
        timeframe: str,
        meta: dict[str, Any],
    ) -> bool:
        """
        Register a freshly trained model as challenger for (model_key, timeframe).

        Automatically promotes challenger to champion if:
          1. No champion exists yet, OR
          2. challenger WF accuracy > champion WF accuracy + PROMOTION_DELTA

        Parameters
        ----------
        model_key  : e.g. 'base', 'futures', 'trend', 'scalping', 'meta'
        timeframe  : e.g. '1h', '1m'
        meta       : trainer meta dict (must contain 'walk_forward_mean_acc' in %)

        Returns True if challenger was auto-promoted to champion.
        """
        registry = _load()
        registry.setdefault(model_key, {}).setdefault(timeframe, {})
        cell = registry[model_key][timeframe]

        challenger_acc = float(meta.get("walk_forward_mean_acc", 0.0)) / 100.0
        challenger_entry: dict = {
            "model_path": meta.get("model_path", ""),
            "wf_acc":     challenger_acc,
            "optimal_threshold": meta.get("optimal_threshold"),
            "last_trained":      meta.get("last_trained", ""),
        }
        cell["challenger"] = challenger_entry
        log.info(
            "[champion_challenger] %s/%s challenger registered: wf_acc=%.4f",
            model_key, timeframe, challenger_acc,
        )

        promoted = False
        champion = cell.get("champion")
        if champion is None:
            cell["champion"] = challenger_entry.copy()
            cell.pop("challenger", None)
            promoted = True
            log.info(
                "[champion_challenger] %s/%s first champion auto-set: wf_acc=%.4f",
                model_key, timeframe, challenger_acc,
            )
        elif challenger_acc > float(champion.get("wf_acc", 0.0)) + PROMOTION_DELTA:
            old_acc = champion.get("wf_acc", 0.0)
            cell["champion"] = challenger_entry.copy()
            cell.pop("challenger", None)
            promoted = True
            log.info(
                "[champion_challenger] %s/%s challenger PROMOTED: %.4f > %.4f + delta",
                model_key, timeframe, challenger_acc, old_acc,
            )
        else:
            log.info(
                "[champion_challenger] %s/%s challenger KEPT (wf %.4f <= champion %.4f + %.3f)",
                model_key, timeframe,
                challenger_acc,
                float(champion.get("wf_acc", 0.0)),
                PROMOTION_DELTA,
            )

        _save(registry)
        return promoted

    def promote_challenger(self, model_key: str, timeframe: str) -> bool:
        """Manually promote the current challenger to champion.

        Returns True if promotion happened, False if no challenger exists.
        """
        registry = _load()
        cell = registry.get(model_key, {}).get(timeframe, {})
        challenger = cell.get("challenger")
        if not challenger:
            log.warning(
                "[champion_challenger] %s/%s promote: no challenger to promote",
                model_key, timeframe,
            )
            return False
        cell["champion"] = challenger.copy()
        cell.pop("challenger", None)
        registry.setdefault(model_key, {})[timeframe] = cell
        _save(registry)
        log.info("[champion_challenger] %s/%s challenger manually promoted", model_key, timeframe)
        return True

    def get_active_artifact(self, model_key: str, timeframe: str) -> str | None:
        """Return the model_path of the current champion, or None if no champion."""
        registry = _load()
        return registry.get(model_key, {}).get(timeframe, {}).get("champion", {}).get("model_path")

    def get_champion_meta(self, model_key: str, timeframe: str) -> dict | None:
        """Return the champion's stored metadata dict, or None."""
        registry = _load()
        return registry.get(model_key, {}).get(timeframe, {}).get("champion")

    def summary(self) -> dict:
        """Return the full registry for dashboards / reporting."""
        return _load()
