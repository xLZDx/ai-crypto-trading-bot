from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any


_PACKAGES = [
    "scikit-learn",
    "lightgbm",
    "pyarrow",
    "numpy",
    "pandas",
    "duckdb",
    "darts",
    "optuna",
]


def capture_env_manifest() -> dict[str, Any]:
    """Return versions of key packages for training reproducibility."""
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    for pkg in _PACKAGES:
        try:
            manifest[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            manifest[pkg] = "not-installed"

    try:
        import torch  # noqa: PLC0415
        manifest["torch"] = torch.__version__
        manifest["cuda"] = torch.version.cuda  # None if CPU-only build
    except ImportError:
        manifest["torch"] = "not-installed"
        manifest["cuda"] = None

    return manifest


def save_env_manifest(path: Path) -> dict[str, Any]:
    """Capture manifest and write it as JSON to *path*."""
    manifest = capture_env_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
