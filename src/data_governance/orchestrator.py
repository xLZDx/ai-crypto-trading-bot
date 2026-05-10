"""
Data orchestrator — Phase 8.

Loads `GovernanceConfig`, instantiates every enabled connector, runs:
  1. `pull_history()` once at startup (parallel across sources).
  2. `realtime_loop()` per source on its configured poll interval (one
     thread per connector — they're mostly I/O-bound).

Storage policy:
  • Hot: QuestDB ILP (each connector writes via its `_qdb()` helper).
  • Cold: when `cfg.store_local`, parquet rollover happens via the
    existing `realtime_db_writer.cold_rollover_loop`.
  • Archive: when `cfg.google_drive_archive`, eligible partitions go to
    Google Drive via `RetentionManager.archive_eligible()` → `GoogleDriveBackup`.

Run:
    python -m src.data_governance.orchestrator
    python -m src.data_governance.orchestrator --once     # history only, no realtime
    python -m src.data_governance.orchestrator --list     # list registered sources
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_governance import GovernanceConfig, REGISTRY, list_sources
from src.data_governance import connectors  # side-effect: registers them all

logger = logging.getLogger("orchestrator")


def _instantiate_enabled(cfg: GovernanceConfig) -> list:
    """Build connector instances for each enabled source that's available."""
    out = []
    for name, cls in REGISTRY.items():
        setting = cfg.get(name)
        if not setting.enabled:
            continue
        try:
            inst = cls()
        except Exception as exc:
            logger.warning("[orch] cannot instantiate %s: %s", name, exc)
            continue
        if not inst.is_available():
            logger.info("[orch] %s skipped (unavailable / missing creds)", name)
            continue
        out.append((inst, setting))
    return out


def run_history(cfg: GovernanceConfig | None = None,
                names: list[str] | None = None) -> dict:
    """Run pull_history() concurrently across all enabled sources."""
    cfg = cfg or GovernanceConfig.load()
    instances = _instantiate_enabled(cfg)
    if names:
        instances = [(i, s) for (i, s) in instances if i.name in names]
    if not instances:
        logger.warning("[orch] no enabled+available connectors")
        return {}

    logger.info("[orch] history phase: %d sources", len(instances))
    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(instances))) as pool:
        futures = {pool.submit(i.pull_history): i for (i, _) in instances}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                n = fut.result() or 0
                results[i.name] = int(n)
                logger.info("[orch] %s: %d rows", i.name, n)
            except Exception as exc:
                logger.exception("[orch] %s failed: %s", i.name, exc)
                results[i.name] = -1
    return results


def run_forever(cfg: GovernanceConfig | None = None) -> None:
    """One thread per connector running its configured poll interval."""
    cfg = cfg or GovernanceConfig.load()
    instances = _instantiate_enabled(cfg)
    if not instances:
        logger.warning("[orch] nothing to run; sleeping")
        while True:
            time.sleep(60)

    threads: list[threading.Thread] = []
    stop_evt = threading.Event()

    def _wrap(inst, poll):
        return inst.realtime_loop(
            poll_interval_sec=poll,
            on_stop=stop_evt.is_set,
        )

    for inst, setting in instances:
        t = threading.Thread(
            target=_wrap, args=(inst, setting.poll_sec),
            name=f"src-{inst.name}", daemon=True,
        )
        t.start()
        threads.append(t)
        logger.info("[orch] started %s (poll=%ds)", inst.name, setting.poll_sec)

    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("[orch] stop requested")
        stop_evt.set()
        for t in threads:
            t.join(timeout=5)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser()
    p.add_argument("--list",  action="store_true", help="list registered sources and exit")
    p.add_argument("--once",  action="store_true", help="run history phase only, no realtime")
    p.add_argument("--names", nargs="+", help="run only the named sources")
    args = p.parse_args()

    if args.list:
        for s in list_sources():
            mark = "auth-required" if s["requires_auth"] else "free"
            print(f"  [{s['priority']}] {s['name']:<22} {s['category']:<12} "
                  f"{mark:<14} -- {s['description']}")
        return 0

    cfg = GovernanceConfig.load()
    run_history(cfg, names=args.names)
    if args.once:
        return 0
    run_forever(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
