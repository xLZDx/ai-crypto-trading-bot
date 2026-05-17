"""
Retention Manager — Phase 7.

Tracks which (symbol, timeframe, yyyymm) partitions have been "fully
consumed" by the model trainer, indexing them as eligible for archival
and (optionally) backup to Google Drive.

Index file: data/retention_index.json
Schema:
    {
        "partitions": [
            {
                "symbol":     "BTC/USDT",
                "timeframe":  "1s",
                "yyyymm":     "2018-01",
                "rows":       2682384,
                "size_bytes": 12345678,
                "trained_on": ["btc_rf_v1", "oft_v1"],
                "last_seen":  "2026-05-01T03:21:00Z",
                "archived":   false,
                "archive_url": null
            },
            ...
        ]
    }

A partition can be marked `archived=True` after a successful Google Drive
upload. The local copy can then be deleted by `prune_archived(...)` once
the user wants to reclaim space.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

from src.utils.safe_json import read_json, write_json

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PARQUET_DIR   = PROJECT_ROOT / "data" / "parquet"
RETENTION_IDX = PROJECT_ROOT / "data" / "retention_index.json"


@dataclass
class PartitionRecord:
    symbol:      str
    timeframe:   str
    yyyymm:      str
    rows:        int = 0
    size_bytes:  int = 0
    trained_on:  list[str] = field(default_factory=list)
    last_seen:   str = ""
    archived:    bool = False
    archive_url: str | None = None

    @property
    def key(self) -> str:
        return f"{self.symbol}::{self.timeframe}::{self.yyyymm}"


class RetentionManager:
    """Stateful index over the parquet partitions."""

    def __init__(self, index_path: Path = RETENTION_IDX):
        self.index_path = index_path
        self._records: dict[str, PartitionRecord] = {}
        self.load()

    # ── Load / save ────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.index_path.exists():
            self._records = {}
            return
        try:
            data = read_json(str(self.index_path), default={"partitions": []}) or {}
            for r in data.get("partitions", []):
                rec = PartitionRecord(**r)
                self._records[rec.key] = rec
            logger.info("Retention index loaded: %d partitions", len(self._records))
        except Exception as exc:
            logger.warning("Could not read retention index (%s) -- starting empty.", exc)
            self._records = {}

    def save(self) -> None:
        payload = {
            "partitions": [asdict(r) for r in self._records.values()],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(str(self.index_path), payload)

    # ── Discover partitions in the parquet store ──────────────────────────

    def scan(self) -> int:
        """Walk `data/parquet/` and add any new partitions to the index."""
        if not PARQUET_DIR.exists():
            return 0
        added = 0
        now = datetime.now(timezone.utc).isoformat()
        for sym_dir in PARQUET_DIR.iterdir():
            if not sym_dir.is_dir():
                continue
            symbol = sym_dir.name.replace("_", "/", 1)

            # Layout 1: legacy 1s under {sym}/yyyymm=*/
            for part in sym_dir.glob("yyyymm=*"):
                if not part.is_dir():
                    continue
                added += int(self._upsert_partition(
                    symbol, "1s", part.name.split("=", 1)[1], part, now,
                ))

            # Layout 2: multi-tf under {sym}/{tf}/yyyymm=*/
            for tf_dir in sym_dir.iterdir():
                if not tf_dir.is_dir() or tf_dir.name.startswith("yyyymm="):
                    continue
                tf = tf_dir.name
                for part in tf_dir.glob("yyyymm=*"):
                    if not part.is_dir():
                        continue
                    added += int(self._upsert_partition(
                        symbol, tf, part.name.split("=", 1)[1], part, now,
                    ))
        if added:
            self.save()
        logger.info("Retention scan done: +%d new partitions (total %d)",
                    added, len(self._records))
        return added

    def _upsert_partition(self, symbol: str, timeframe: str, yyyymm: str,
                          part_dir: Path, now_iso: str) -> bool:
        key = f"{symbol}::{timeframe}::{yyyymm}"
        files = list(part_dir.glob("*.parquet"))
        size = sum(f.stat().st_size for f in files)
        is_new = key not in self._records
        if is_new:
            self._records[key] = PartitionRecord(
                symbol=symbol, timeframe=timeframe, yyyymm=yyyymm,
                size_bytes=size, last_seen=now_iso,
            )
        else:
            r = self._records[key]
            r.size_bytes = size
            r.last_seen = now_iso
        return is_new

    # ── Public ops ────────────────────────────────────────────────────────

    def mark_trained(self, symbol: str, timeframe: str, yyyymm: str,
                     model_name: str) -> bool:
        key = f"{symbol}::{timeframe}::{yyyymm}"
        rec = self._records.get(key)
        if rec is None:
            return False
        if model_name not in rec.trained_on:
            rec.trained_on.append(model_name)
        self.save()
        return True

    def mark_trained_range(self, symbol: str, timeframe: str,
                           start_ym: str, end_ym: str, model_name: str) -> int:
        """Mark all partitions between two YYYY-MM strings (inclusive)."""
        n = 0
        for r in self._records.values():
            if r.symbol == symbol and r.timeframe == timeframe \
               and start_ym <= r.yyyymm <= end_ym:
                if model_name not in r.trained_on:
                    r.trained_on.append(model_name)
                    n += 1
        if n:
            self.save()
        return n

    def archive_eligible(self, *, min_models: int = 1,
                         older_than_days: int = 30) -> list[PartitionRecord]:
        """Return partitions that have been trained on at least
        `min_models` distinct models AND have not been archived yet."""
        cutoff = (datetime.now(timezone.utc) -
                  timedelta(days=older_than_days)).strftime("%Y-%m")
        return [r for r in self._records.values()
                if not r.archived
                and len(r.trained_on) >= min_models
                and r.yyyymm < cutoff]

    def mark_archived(self, key: str, archive_url: str) -> bool:
        rec = self._records.get(key)
        if rec is None:
            return False
        rec.archived = True
        rec.archive_url = archive_url
        self.save()
        return True

    def prune_archived(self, *, dry_run: bool = True) -> list[Path]:
        """Delete local parquet partitions that are marked archived.

        Set `dry_run=False` to actually remove. Use only when you've
        verified the Google Drive backup is restorable.
        """
        removed: list[Path] = []
        for r in self._records.values():
            if not r.archived or not r.archive_url:
                continue
            sym_safe = r.symbol.replace("/", "_")
            if r.timeframe == "1s":
                path = PARQUET_DIR / sym_safe / f"yyyymm={r.yyyymm}"
                path_alt = PARQUET_DIR / sym_safe / "1s" / f"yyyymm={r.yyyymm}"
            else:
                path = PARQUET_DIR / sym_safe / r.timeframe / f"yyyymm={r.yyyymm}"
                path_alt = None
            for p in (path, path_alt):
                if p and p.exists():
                    if not dry_run:
                        shutil.rmtree(p)
                    removed.append(p)
        return removed

    def stats(self) -> dict:
        n = len(self._records)
        archived = sum(1 for r in self._records.values() if r.archived)
        trained = sum(1 for r in self._records.values() if r.trained_on)
        size_total = sum(r.size_bytes for r in self._records.values())
        size_archived = sum(r.size_bytes for r in self._records.values() if r.archived)
        return {
            "partitions":         n,
            "trained":            trained,
            "archived":           archived,
            "size_total_gb":      round(size_total / 1e9, 2),
            "size_archived_gb":   round(size_archived / 1e9, 2),
        }


__all__ = ["RetentionManager", "PartitionRecord", "RETENTION_IDX"]
