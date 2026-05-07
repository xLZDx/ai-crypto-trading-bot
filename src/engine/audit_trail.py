"""
audit_trail — verify every order in trades.json is traceable to a
signal + a model artifact.

Phase F production-readiness: 'every order in data/trades.json
traceable to a signal in data/signals.json traceable to a model
artifact + input-data hash'.

Implementation today:
  - data/trades.json  is a flat list of order dicts written by
    book_market_order / OrderManager.
  - data/signals.json is the bot's most-recent-N signals log.
  - models/*_meta.json carries each model's training metadata.

This module checks for orphans:
  1. Every order with strategy='X' should reference a signal whose
     strategy='X' and timestamp within ±60s of the order's timestamp.
  2. Every signal that triggered an order should reference a model
     (or be marked rules-only) — and the model artifact should exist.
  3. Trades older than the model's last training timestamp should be
     flagged as 'pre-train' (informational, not a violation).

Output: data/audit_report_<ts>.json with verdict counts.

Usage:
    python -m src.engine.audit_trail
    python -m src.engine.audit_trail --max-trades 100
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRADES_PATH  = PROJECT_ROOT / "data" / "trades.json"
SIGNALS_PATH = PROJECT_ROOT / "data" / "signals.json"
MODELS_DIR   = PROJECT_ROOT / "models"


def _load_json_list(path: Path, default=None):
    if not path.exists():
        return default if default is not None else []
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            # Some bots write {"trades": [...]} — accept either
            for k in ("trades", "signals", "items", "data"):
                if k in data and isinstance(data[k], list):
                    return data[k]
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.warning("[audit_trail] failed to parse %s: %s", path, exc)
    return default if default is not None else []


def _model_metas() -> dict[str, dict]:
    """Map model_key → meta dict. model_key here is the file stem."""
    out: dict[str, dict] = {}
    if not MODELS_DIR.exists():
        return out
    for p in MODELS_DIR.glob("*_meta.json"):
        try:
            out[p.stem.replace("_meta", "")] = json.loads(p.read_text())
        except Exception:
            pass
    return out


def _ts_of(row) -> float | None:
    """Best-effort timestamp parser. Accepts unix-seconds, unix-ms, or ISO."""
    for k in ("timestamp", "ts", "time", "created_at", "executed_at"):
        v = row.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v) if v < 1e12 else float(v) / 1000.0
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
    return None


def run_audit(max_trades: int | None = None) -> dict:
    started = time.time()
    trades = _load_json_list(TRADES_PATH)
    signals = _load_json_list(SIGNALS_PATH)
    metas = _model_metas()

    if max_trades is not None and max_trades > 0:
        trades = trades[-int(max_trades):]

    orphan_orders = 0       # no matching signal
    untraced_signals = 0    # signal references no model AND not rules-only
    missing_artifacts = 0   # signal references a model whose file is gone
    pre_train_trades = 0    # trade older than the model's last_train_ts

    # Build a quick {(strategy, ~ts_bucket_minute): [signals]} index.
    sig_by_strat: dict[str, list[dict]] = {}
    for s in signals:
        k = (s.get("strategy") or s.get("strategy_name") or "").strip()
        sig_by_strat.setdefault(k, []).append(s)

    findings = []
    for tr in trades:
        strat = (tr.get("strategy") or tr.get("strategy_name") or "").strip()
        ts = _ts_of(tr)
        if not strat:
            findings.append({"kind": "no_strategy_field",
                              "trade_id": tr.get("id") or tr.get("order_id")})
            continue
        sigs = sig_by_strat.get(strat, [])
        # Match by closest timestamp within 60 s
        match = None
        if ts is not None:
            best = None
            for s in sigs:
                sts = _ts_of(s)
                if sts is None:
                    continue
                dt = abs(sts - ts)
                if dt <= 60 and (best is None or dt < best[0]):
                    best = (dt, s)
            if best:
                match = best[1]
        if match is None:
            orphan_orders += 1
            findings.append({"kind": "orphan_order",
                              "strategy": strat, "ts": ts,
                              "trade_id": tr.get("id") or tr.get("order_id")})
            continue
        # Validate the signal's model reference
        model_ref = (match.get("model")
                     or match.get("model_key")
                     or match.get("source_model"))
        rules_only = bool(match.get("rules_only"))
        if model_ref:
            # Look for any meta file whose stem starts with the ref
            ref = str(model_ref).replace(".joblib", "").replace("_model", "")
            found = any(k.startswith(ref) or ref in k for k in metas.keys())
            if not found:
                missing_artifacts += 1
                findings.append({"kind": "missing_artifact",
                                  "strategy": strat,
                                  "model_ref": model_ref})
            else:
                # Check freshness
                meta_key = next((k for k in metas.keys()
                                 if k.startswith(ref) or ref in k), None)
                if meta_key:
                    meta = metas[meta_key]
                    train_ts_raw = meta.get("training_completed_at") or meta.get("trained_at")
                    if train_ts_raw and ts is not None:
                        try:
                            tt = datetime.fromisoformat(
                                str(train_ts_raw).replace("Z", "+00:00")
                            ).timestamp()
                            if ts < tt:
                                pre_train_trades += 1
                        except Exception:
                            pass
        elif not rules_only:
            untraced_signals += 1
            findings.append({"kind": "untraced_signal",
                              "strategy": strat, "ts": ts})

    out = {
        "ok":                  (orphan_orders == 0
                                and untraced_signals == 0
                                and missing_artifacts == 0),
        "trades_checked":      len(trades),
        "signals_indexed":     len(signals),
        "models_indexed":      len(metas),
        "orphan_orders":       orphan_orders,
        "untraced_signals":    untraced_signals,
        "missing_artifacts":   missing_artifacts,
        "pre_train_trades":    pre_train_trades,
        "findings":            findings[:50],          # cap for log size
        "more_findings":       max(0, len(findings) - 50),
        "started_at":          datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "elapsed_s":           round(time.time() - started, 3),
    }
    # Persist a copy
    rep_dir = PROJECT_ROOT / "data" / "audit_reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / f"audit_{int(time.time())}.json"
    rep_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    out["report_path"] = str(rep_path)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Audit trail: trades → signals → models")
    ap.add_argument("--max-trades", type=int, default=None,
                    help="Audit only the most recent N trades (faster)")
    args = ap.parse_args(argv)
    res = run_audit(max_trades=args.max_trades)
    print(json.dumps(res, default=str, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
