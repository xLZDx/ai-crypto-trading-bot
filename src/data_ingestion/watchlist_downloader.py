"""
WatchlistDownloader — daemon that keeps OHLCV data fresh for all watchlist coins.

Downloads 3 timeframes per symbol:
  1s  — last 24 h of 1-second bars, refreshed every 5 min (scalping tick data)
  1m  — 3 years of 1-minute bars, synced every hour
  1M  — 10 years of monthly bars, synced daily (macro regime features)

Behaviour:
  - Polls watchlist every 30 s for additions / removals
  - New coin added  → immediate parallel backfill of all 3 TFs, then training
  - Coin removed    → keep existing files (training value preserved), stop syncing
  - After new-coin backfill → writes download_manifest.json, git-commits it,
    then launches ML training + simulator training in background subprocesses

Usage:
  python -m src.data_ingestion.watchlist_downloader   # runs forever
  WatchlistDownloader().run()
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
WATCHLIST_FILE = PROJECT_ROOT / "data" / "watchlist.json"
MANIFEST_FILE  = PROJECT_ROOT / "data" / "download_manifest.json"

_DEFAULT_WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]

# ── Timeframe configuration ───────────────────────────────────────────────────
# history_days : how far back to backfill on first encounter
# sync_every   : seconds between incremental syncs
# limit        : candles per API call (Binance max 1000)
TF_CONFIG: dict[str, dict] = {
    "1s": {
        "history_days": 2,        # Binance keeps ≈24–48 h of 1s data (API hard limit)
        "sync_every":   300,      # refresh every 5 min
        "limit":        1000,
    },
    "1m": {
        "history_days": 365 * 10, # full available history (~10 years back to 2017)
        "sync_every":   3600,     # sync every hour
        "limit":        1000,
    },
    "1h": {
        "history_days": 365 * 10, # full 10-year 1h history for model training
        "sync_every":   3600,     # sync every hour
        "limit":        1000,
    },
    "1M": {
        "history_days": 365 * 15, # full monthly history (Binance data from 2017)
        "sync_every":   86400,    # sync daily
        "limit":        200,
    },
}

# Training is triggered at most once per this many seconds after new data
_TRAIN_COOLDOWN = 600   # 10 min


# ── Binance helpers ───────────────────────────────────────────────────────────

_RETRY_CODES = {429, 418}


def _get(url: str, timeout: int = 15) -> list:
    """GET with exponential back-off on rate-limit responses."""
    delay = 2.0
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code in _RETRY_CODES:
                wait = int(r.headers.get("Retry-After", delay))
                logger.warning("Rate-limited (%s). Waiting %ss …", r.status_code, wait)
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d): %s", attempt + 1, exc)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"API unreachable after retries: {url}")


def _last_ts_ms(gz_path: Path) -> int | None:
    """Read last timestamp (ms) from gzipped CSV without loading the whole file."""
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            last = deque(f, maxlen=1)[0]
        if last and not last.startswith("timestamp"):
            ts_str = last.split(",")[0].strip()
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
    except Exception:
        pass
    return None


def _row_to_csv(row: list) -> list:
    dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return [dt, float(row[1]), float(row[2]), float(row[3]), float(row[4]),
            float(row[5]), float(row[7]), float(row[8]), float(row[9]), float(row[10])]


_CSV_HEADER = ["timestamp", "open", "high", "low", "close", "volume",
               "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote"]


def _write_to_db(symbol: str, timeframe: str, raw_rows: list) -> None:
    """Write raw Binance kline rows to ParquetClient (best-effort, non-blocking)."""
    try:
        from src.database.parquet_client import get_client
        db = get_client()
        if not db.is_available():
            return
        bars = []
        for row in raw_rows:
            dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
            bars.append({
                "timestamp": dt,
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
                "funding_rate": 0.0,
            })
        written = db.write_market_candles_bulk(symbol, timeframe, bars)
        logger.debug("[%s/%s] Wrote %d bars to QuestDB", symbol, timeframe, written)
    except Exception as exc:
        logger.debug("[%s/%s] QuestDB write skipped: %s", symbol, timeframe, exc)


def _fetch_klines(symbol: str, interval: str, start_ms: int | None, limit: int) -> list:
    safe = symbol.replace("/", "")
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={safe}&interval={interval}&limit={limit}")
    if start_ms is not None:
        url += f"&startTime={start_ms}"
    return _get(url)


# ── Per-symbol download logic ─────────────────────────────────────────────────

def _gz_path(symbol: str, timeframe: str) -> Path:
    return RAW_DIR / f"{symbol.replace('/', '_')}_{timeframe}.csv.gz"


def backfill(symbol: str, timeframe: str, history_days: int, limit: int) -> int:
    """
    Full historical backfill for a symbol/TF.  Skips if file already exists
    (use sync() to append newer candles).  Returns number of candles written.
    """
    path = _gz_path(symbol, timeframe)
    if path.exists():
        logger.debug("[%s/%s] file exists — using sync() instead", symbol, timeframe)
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=history_days)).timestamp() * 1000)
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    total    = 0

    logger.info("[%s/%s] Backfilling %d days …", symbol, timeframe, history_days)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        while since_ms < now_ms:
            try:
                rows = _fetch_klines(symbol, timeframe, since_ms, limit)
            except Exception as exc:
                logger.warning("[%s/%s] Fetch error: %s", symbol, timeframe, exc)
                break
            if not rows:
                break
            for row in rows:
                writer.writerow(_row_to_csv(row))
            # Mirror batch to QuestDB (non-blocking, best-effort)
            _write_to_db(symbol, timeframe, rows)
            total    += len(rows)
            since_ms  = rows[-1][0] + 1
            if len(rows) < limit:
                break
            time.sleep(0.15)

    logger.info("[%s/%s] Backfill done — %d candles", symbol, timeframe, total)
    return total


def sync(symbol: str, timeframe: str, limit: int) -> int:
    """
    Append only candles newer than the last stored timestamp.
    Returns number of candles appended.
    """
    path = _gz_path(symbol, timeframe)
    last = _last_ts_ms(path) if path.exists() else None
    start_ms = (last + 1) if last else None

    try:
        rows = _fetch_klines(symbol, timeframe, start_ms, limit)
    except Exception as exc:
        logger.warning("[%s/%s] Sync error: %s", symbol, timeframe, exc)
        return 0

    if not rows:
        return 0

    mode = "at" if path.exists() else "wt"
    with gzip.open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "wt":
            writer.writerow(_CSV_HEADER)
        for row in rows:
            writer.writerow(_row_to_csv(row))

    # Mirror to QuestDB (non-blocking, best-effort)
    _write_to_db(symbol, timeframe, rows)

    logger.info("[%s/%s] Synced %d new candles", symbol, timeframe, len(rows))
    return len(rows)


def download_all_tfs(symbol: str) -> dict[str, int]:
    """Backfill then sync all 3 timeframes for one symbol. Returns counts."""
    results: dict[str, int] = {}
    for tf, cfg in TF_CONFIG.items():
        n = backfill(symbol, tf, cfg["history_days"], cfg["limit"])
        if n == 0:
            n = sync(symbol, tf, cfg["limit"])
        results[tf] = n
    return results


# ── Manifest ──────────────────────────────────────────────────────────────────

_manifest_lock = threading.Lock()


def _load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(data: dict) -> None:
    with _manifest_lock:
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _update_manifest(symbol: str, counts: dict[str, int]) -> None:
    manifest = _load_manifest()
    manifest[symbol] = {
        "timeframes": {
            tf: {
                "file": str(_gz_path(symbol, tf)),
                "exists": _gz_path(symbol, tf).exists(),
                "last_sync": datetime.now(timezone.utc).isoformat(),
                "candles_added": counts.get(tf, 0),
            }
            for tf in TF_CONFIG
        },
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)


# ── Post-download hooks ───────────────────────────────────────────────────────

def _git_commit_manifest() -> None:
    """Add download_manifest.json to git and commit."""
    try:
        subprocess.run(
            ["git", "add", str(MANIFEST_FILE)],
            cwd=str(PROJECT_ROOT), check=True, capture_output=True,
        )
        msg = f"data: update download_manifest.json [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC]"
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("[Manifest] git commit: %s", msg)
        else:
            logger.debug("[Manifest] git commit skipped (nothing to commit)")
    except Exception as exc:
        logger.warning("[Manifest] git commit failed: %s", exc)


def _launch_training() -> None:
    """Launch train_all_models.py in a detached subprocess."""
    python = str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
    if not os.path.exists(python):
        python = sys.executable
    script = str(PROJECT_ROOT / "src" / "engine" / "train_all_models.py")
    try:
        proc = subprocess.Popen(
            [python, script],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
        logger.info("[Training] Launched train_all_models.py (PID %d)", proc.pid)
    except Exception as exc:
        logger.warning("[Training] Failed to launch training: %s", exc)


def _launch_simulator_training() -> None:
    """Start the simulator agent which triggers ContinuousTrainerAgent."""
    try:
        r = requests.post(
            "http://127.0.0.1:5000/api/simulator/start",
            json={"speed": 5000, "train_models": ["ScalpingML", "OU_Filter"]},
            timeout=5,
        )
        if r.ok:
            logger.info("[SimTraining] Simulator training started via dashboard API")
        else:
            logger.debug("[SimTraining] Dashboard not running (%s)", r.status_code)
    except Exception:
        logger.debug("[SimTraining] Dashboard not reachable — skipping sim trigger")


# ── Watchlist daemon ──────────────────────────────────────────────────────────

class WatchlistDownloader:
    """
    Continuously maintains OHLCV data for all watchlist symbols.

    Thread layout:
      - Main thread: watchlist poll loop (every 30 s)
      - ThreadPoolExecutor: parallel per-symbol downloads
      - One-shot threads: training triggers (debounced)
    """

    POLL_INTERVAL = 30       # seconds between watchlist checks
    MAX_WORKERS   = 4        # parallel symbol downloads

    def __init__(self):
        self._known: set[str] = set()          # symbols seen so far
        self._last_sync: dict[str, dict[str, float]] = {}  # [sym][tf] = epoch
        self._last_train: float = 0.0          # epoch of last training trigger
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS,
                                            thread_name_prefix="dl")
        self._stop = threading.Event()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("WatchlistDownloader started. Watching %s every %ds",
                    WATCHLIST_FILE, self.POLL_INTERVAL)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.error("Tick error: %s", exc, exc_info=True)
            self._stop.wait(self.POLL_INTERVAL)

    def stop(self) -> None:
        self._stop.set()
        self._executor.shutdown(wait=False)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_watchlist(self) -> list[str]:
        if WATCHLIST_FILE.exists():
            try:
                return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return list(_DEFAULT_WATCHLIST)

    def _tick(self) -> None:
        current = set(self._load_watchlist())
        new_coins = current - self._known
        removed   = self._known - current

        if removed:
            logger.info("Coins removed from watchlist (data kept): %s", removed)

        if new_coins:
            logger.info("New coins detected: %s — starting immediate download", new_coins)
            self._download_new(new_coins)

        # Ongoing syncs for existing symbols
        self._schedule_due_syncs(current)

        self._known = current

    def _download_new(self, symbols: set[str]) -> None:
        """Backfill all 3 TFs for newly added symbols, then trigger training."""
        futures = {
            self._executor.submit(self._full_download, sym): sym
            for sym in symbols
        }
        all_counts: dict[str, dict] = {}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                counts = fut.result()
                all_counts[sym] = counts
                _update_manifest(sym, counts)
            except Exception as exc:
                logger.error("Download failed for %s: %s", sym, exc)

        if all_counts:
            _git_commit_manifest()
            self._trigger_training()

    def _full_download(self, symbol: str) -> dict[str, int]:
        """Backfill + sync all TFs for one symbol, initialise sync timers."""
        logger.info("[%s] Starting full download (1s / 1m / 1M) …", symbol)
        counts = download_all_tfs(symbol)
        now = time.time()
        self._last_sync.setdefault(symbol, {})
        for tf in TF_CONFIG:
            self._last_sync[symbol][tf] = now
        total = sum(counts.values())
        logger.info("[%s] Full download done — %d candles across 3 TFs", symbol, total)
        return counts

    def _schedule_due_syncs(self, symbols: set[str]) -> None:
        """For known symbols, run syncs whose interval has elapsed."""
        now = time.time()
        futures = {}
        for sym in symbols:
            if sym in self._known:  # only already-known symbols
                self._last_sync.setdefault(sym, {})
                for tf, cfg in TF_CONFIG.items():
                    last = self._last_sync[sym].get(tf, 0)
                    if now - last >= cfg["sync_every"]:
                        futures[self._executor.submit(
                            self._run_sync, sym, tf, cfg["limit"]
                        )] = (sym, tf)

        for fut in as_completed(futures):
            sym, tf = futures[fut]
            try:
                n = fut.result()
                self._last_sync[sym][tf] = time.time()
                if n > 0:
                    manifest = _load_manifest()
                    if sym in manifest:
                        manifest[sym]["timeframes"][tf]["last_sync"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                        manifest[sym]["timeframes"][tf]["candles_added"] = n
                        _save_manifest(manifest)
            except Exception as exc:
                logger.warning("Sync %s/%s failed: %s", sym, tf, exc)

    def _run_sync(self, symbol: str, tf: str, limit: int) -> int:
        return sync(symbol, tf, limit)

    def _trigger_training(self) -> None:
        now = time.time()
        if now - self._last_train < _TRAIN_COOLDOWN:
            logger.debug("Training trigger skipped (cooldown)")
            return
        self._last_train = now
        t = threading.Thread(target=self._do_training, daemon=True)
        t.start()

    def _do_training(self) -> None:
        logger.info("Triggering ML training pipeline …")
        _launch_training()
        time.sleep(5)
        logger.info("Triggering simulator training …")
        _launch_simulator_training()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from src.utils.hw_config import configure as _hw
    _hw(verbose=False)

    d = WatchlistDownloader()
    try:
        d.run()
    except KeyboardInterrupt:
        logger.info("Interrupted — stopping.")
        d.stop()


if __name__ == "__main__":
    # Make project root importable when run directly
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
