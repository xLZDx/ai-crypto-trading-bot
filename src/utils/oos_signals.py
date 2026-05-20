"""OOS signal utilities for meta-labeler training (Phase 8).

After base / trend / futures trainers complete, each saves OOS predictions to:
    data/oos_signals/<run_id>/base.parquet
    data/oos_signals/<run_id>/trend.parquet
    data/oos_signals/<run_id>/futures.parquet

run_id format: UTC timestamp with no colons (e.g. '2026-05-20_140000') so
the directory name is valid on all OS (Windows disallows colons in paths).

The meta trainer calls validate_oos_signals(run_id) before starting.
A missing file or run_id mismatch is a hard stop — never train meta on
mixed-run OOS data.
"""
from __future__ import annotations

from pathlib import Path

_REQUIRED_MODELS = ("base", "trend", "futures")


def oos_dir(root: Path, run_id: str) -> Path:
    return root / "oos_signals" / run_id


def save_oos_signal(root: Path, run_id: str, model_key: str,
                    df: "pd.DataFrame") -> Path:  # noqa: F821 — avoid circular
    """Write OOS predictions for *model_key* under *run_id*."""
    import pandas as pd  # noqa: PLC0415

    dest = oos_dir(root, run_id) / f"{model_key}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["run_id"] = run_id
    df.to_parquet(dest, index=False)
    return dest


def validate_oos_signals(root: Path, run_id: str) -> None:
    """Raise RuntimeError if any required OOS signal file is missing or stale.

    Checks:
    1. All three files exist under data/oos_signals/<run_id>/.
    2. Each file's run_id column matches the expected run_id (detects stale
       files from a previous partial run left behind under a new run_id dir).
    """
    import pandas as pd  # noqa: PLC0415

    d = oos_dir(root, run_id)
    missing = [m for m in _REQUIRED_MODELS if not (d / f"{m}.parquet").exists()]
    if missing:
        raise RuntimeError(
            f"OOS signals missing for run_id={run_id}: {missing}. "
            "Cannot train meta-labeler on incomplete OOS data."
        )

    for model_key in _REQUIRED_MODELS:
        df = pd.read_parquet(d / f"{model_key}.parquet", columns=["run_id"])
        bad_rows = df["run_id"].ne(run_id).sum()
        if bad_rows > 0:
            raise RuntimeError(
                f"OOS file {model_key}.parquet has {bad_rows} rows with "
                f"run_id != '{run_id}' -- likely a stale file from a prior run."
            )
