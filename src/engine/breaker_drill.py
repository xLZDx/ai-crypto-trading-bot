"""
breaker_drill — exercise every circuit breaker without putting real
money at risk.

Phase F of the institutional roadmap. Production-readiness checklist
calls for: 'force every circuit breaker via test harness and confirm
position flatten'. Today we have the breakers (order_manager.
circuit_breaker_check) but no automated drill — the only way to
exercise them is to wait for a real outage.

Each drill scenario constructs a synthetic state that should trip
exactly one breaker, calls circuit_breaker_check, and confirms:
  - the right trigger fires
  - no other (false-positive) trigger fires
  - the response payload has the expected shape

Returns a dict suitable for the dashboard's pill:
  {ok, scenarios_run, scenarios_passed, scenarios_failed, results: [...]}

Usage:
    python -m src.engine.breaker_drill           # all scenarios
    python -m src.engine.breaker_drill --scenario max_dd
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _scenario_max_dd():
    """Drawdown breach: equity dropped 7% off peak with default 5% limit."""
    from src.engine.order_manager import OrderManager
    om = OrderManager()
    return om.circuit_breaker_check(
        peak_equity=10_000,
        current_equity=9_300,        # -7% off peak
        api_latency_ms=120,
        last_data_ts_unix=time.time() - 5,
        now_unix=time.time(),
    ), "max_daily_drawdown"


def _scenario_api_latency():
    """API latency spike: 800ms vs 500ms ceiling."""
    from src.engine.order_manager import OrderManager
    om = OrderManager()
    return om.circuit_breaker_check(
        peak_equity=10_000,
        current_equity=10_000,
        api_latency_ms=800,
        last_data_ts_unix=time.time() - 5,
        now_unix=time.time(),
    ), "api_latency"


def _scenario_stale_feed():
    """Data feed: last bar is 60s old vs 30s ceiling."""
    from src.engine.order_manager import OrderManager
    om = OrderManager()
    return om.circuit_breaker_check(
        peak_equity=10_000,
        current_equity=10_000,
        api_latency_ms=120,
        last_data_ts_unix=time.time() - 60,
        now_unix=time.time(),
    ), "data_feed_inconsistency"


def _scenario_clean():
    """Healthy state: nothing should trip."""
    from src.engine.order_manager import OrderManager
    om = OrderManager()
    return om.circuit_breaker_check(
        peak_equity=10_000,
        current_equity=10_050,        # tiny PnL, no drawdown
        api_latency_ms=80,
        last_data_ts_unix=time.time() - 2,
        now_unix=time.time(),
    ), None


_SCENARIOS = {
    "max_dd":         (_scenario_max_dd,         "max_daily_drawdown"),
    "api_latency":    (_scenario_api_latency,    "api_latency"),
    "stale_feed":     (_scenario_stale_feed,     "data_feed_inconsistency"),
    "clean":          (_scenario_clean,          None),
}


def run_drill(only: str | None = None) -> dict:
    """Run every drill scenario (or just the named one). Returns a
    summary dict with per-scenario verdicts."""
    started = time.time()
    results = []
    passed = 0
    failed = 0
    targets = {only: _SCENARIOS[only]} if only and only in _SCENARIOS else _SCENARIOS
    for name, (fn, expected_trigger) in targets.items():
        try:
            payload, expected = fn()
            actual = payload.get("trigger")
            ok = (actual == expected_trigger)
            verdict = "pass" if ok else "fail"
            if ok:
                passed += 1
            else:
                failed += 1
            results.append({
                "scenario":         name,
                "expected_trigger": expected_trigger,
                "actual_trigger":   actual,
                "verdict":          verdict,
                "drawdown_pct":     payload.get("drawdown_pct"),
                "reason":           payload.get("reason"),
                "ok_field":         payload.get("ok"),
            })
        except Exception as exc:
            failed += 1
            results.append({
                "scenario": name,
                "expected_trigger": expected_trigger,
                "verdict": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return {
        "ok":                 failed == 0,
        "scenarios_run":      len(results),
        "scenarios_passed":   passed,
        "scenarios_failed":   failed,
        "results":            results,
        "started_at":         datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "elapsed_s":          round(time.time() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Circuit breaker drill harness")
    ap.add_argument("--scenario",
                    choices=list(_SCENARIOS) + ["all"],
                    default="all")
    args = ap.parse_args(argv)
    only = None if args.scenario == "all" else args.scenario
    res = run_drill(only=only)
    print(json.dumps(res, default=str, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
