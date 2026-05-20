#!/usr/bin/env python3
"""
Pre-flight checks before starting a retrain run.

All checks must PASS before the orchestrator proceeds.
Exit code 0 = all pass. Non-zero = at least one failure.

Usage (on VPS):
    python scripts/preflight_train.py
    python scripts/preflight_train.py --expected-parquet-count 13951
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR  = PROJECT_ROOT / "data" / "parquet"
OOS_DIR      = PROJECT_ROOT / "data" / "oos_signals"
JOBS_FILE    = PROJECT_ROOT / "data" / "dashboard_jobs.json"
RULES_FILE   = PROJECT_ROOT / "data" / "training_rules.json"

PASS  = "\033[32mPASS\033[0m"
FAIL  = "\033[31mFAIL\033[0m"
WARN  = "\033[33mWARN\033[0m"


def _check(label: str, result: bool, detail: str = "") -> bool:
    status = PASS if result else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return result


def check_disk_space() -> bool:
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
    return _check("Disk free >= 20 GB", free_gb >= 20, f"{free_gb:.1f} GB free")


def check_parquet_count(expected: int | None) -> bool:
    actual = sum(1 for _ in PARQUET_DIR.rglob("*.parquet"))
    if expected is None:
        return _check("Parquet files exist", actual > 0, f"{actual} files found")
    ok = actual >= expected * 0.95  # allow 5% tolerance
    return _check(
        f"Parquet file count >= {expected}",
        ok,
        f"found {actual}, expected >= {int(expected * 0.95)}",
    )


def check_schema() -> bool:
    sample_files = list(PARQUET_DIR.glob("*/*/yyyymm=*/data_0.parquet"))
    if not sample_files:
        sample_files = list(PARQUET_DIR.glob("**/data_0.parquet"))
    if not sample_files:
        return _check("Parquet schema valid", False, "no parquet files found")
    errors: list[str] = []
    for f in sample_files[:5]:
        try:
            pq.read_schema(str(f))
        except Exception as e:
            errors.append(f"{f.name}: {e}")
    return _check(
        "Parquet schema valid (sample 5 files)",
        len(errors) == 0,
        f"errors: {errors}" if errors else f"checked {min(5, len(sample_files))} files",
    )


def check_no_running_jobs() -> bool:
    if not JOBS_FILE.exists():
        return _check("No running jobs", True, "dashboard_jobs.json missing — treating as clean")
    try:
        jobs = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        running = [k for k, v in jobs.items() if isinstance(v, dict) and v.get("status") == "running"]
        return _check("No running jobs", len(running) == 0, f"running: {running}" if running else "clean")
    except Exception as e:
        return _check("No running jobs", False, str(e))


def check_api_keys() -> bool:
    required = ["API_KEY", "API_SECRET", "HETZNER_API_TOKEN", "VASTAI_API_KEY", "GEMINI_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    return _check("Required .env keys present", len(missing) == 0, f"missing: {missing}" if missing else "all present")


def check_gdrive_backup() -> bool:
    try:
        result = subprocess.run(
            ["rclone", "lsd", "gdrive:trading-bot-backup/"],
            capture_output=True, text=True, timeout=30,
        )
        ok = result.returncode == 0 and bool(result.stdout.strip())
        return _check("GDrive backup exists", ok, result.stderr.strip()[:80] if not ok else "entries found")
    except FileNotFoundError:
        return _check("GDrive backup exists", False, "rclone not found")
    except subprocess.TimeoutExpired:
        return _check("GDrive backup exists", False, "rclone timed out")


def check_training_rules() -> bool:
    if not RULES_FILE.exists():
        return _check("training_rules.json valid", False, "file missing")
    try:
        rules = json.loads(RULES_FILE.read_text(encoding="utf-8"))
        required_fields = ["models", "global"]
        missing = [f for f in required_fields if f not in rules]
        return _check(
            "training_rules.json valid",
            len(missing) == 0,
            f"missing fields: {missing}" if missing else f"{len(rules.get('models', {}))} models",
        )
    except json.JSONDecodeError as e:
        return _check("training_rules.json valid", False, str(e))


def check_oos_writable() -> bool:
    OOS_DIR.mkdir(parents=True, exist_ok=True)
    test_file = OOS_DIR / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
        return _check("OOS directory writable", True)
    except Exception as e:
        return _check("OOS directory writable", False, str(e))


def check_hetzner() -> bool:
    token = os.environ.get("HETZNER_API_TOKEN", "")
    if not token:
        return _check("Hetzner credentials", False, "HETZNER_API_TOKEN not set")
    try:
        r = requests.get(
            "https://api.hetzner.cloud/v1/servers",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        return _check("Hetzner credentials", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as e:
        return _check("Hetzner credentials", False, str(e)[:80])


def check_vastai() -> bool:
    key = os.environ.get("VASTAI_API_KEY", "")
    if not key:
        return _check("Vast.ai credentials", False, "VASTAI_API_KEY not set")
    try:
        r = requests.get(
            "https://cloud.vast.ai/api/v0/instances/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        return _check("Vast.ai credentials", r.status_code in (200, 404), f"HTTP {r.status_code}")
    except Exception as e:
        return _check("Vast.ai credentials", False, str(e)[:80])


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-flight checks before retrain")
    parser.add_argument("--expected-parquet-count", type=int, default=None,
                        help="Expected number of parquet files (from last verified sync)")
    args = parser.parse_args()

    # Load .env if dotenv available
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    print("=" * 60)
    print("Pre-flight checks (retrain gate)")
    print("=" * 60)

    checks = [
        check_disk_space(),
        check_parquet_count(args.expected_parquet_count),
        check_schema(),
        check_no_running_jobs(),
        check_api_keys(),
        check_gdrive_backup(),
        check_training_rules(),
        check_oos_writable(),
        check_hetzner(),
        check_vastai(),
    ]

    print("=" * 60)
    passed = sum(checks)
    total  = len(checks)
    if passed == total:
        print(f"\033[32mALL {total} CHECKS PASSED -- safe to retrain\033[0m")
        sys.exit(0)
    else:
        print(f"\033[31m{total - passed}/{total} CHECKS FAILED -- fix before retraining\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
