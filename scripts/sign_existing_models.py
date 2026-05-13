"""
One-shot: sign every existing model artifact in models/ with the current
MODEL_MANIFEST_KEY. Run after first setting the key in .env so existing
models pass HMAC verify without needing a retrain.

Usage:
    python scripts/sign_existing_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Load .env so MODEL_MANIFEST_KEY is visible
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_integrity import sign_model, _load_key

models_dir = PROJECT_ROOT / 'models'
if not models_dir.exists():
    print(f"ERROR: {models_dir} does not exist")
    sys.exit(1)

if _load_key() is None:
    print("ERROR: MODEL_MANIFEST_KEY is not set in .env — cannot sign models.")
    sys.exit(1)

# All model extensions used in this project
exts = ('.joblib', '.pt', '.ckpt', '.pkl')
signed, skipped, failed = 0, 0, 0

for path in sorted(models_dir.iterdir()):
    if path.is_dir():
        continue
    if not path.suffix.lower() in exts:
        continue
    if path.is_symlink():
        print(f"SKIP (symlink): {path.name}")
        skipped += 1
        continue
    try:
        ok = sign_model(str(path))
        if ok:
            print(f"OK:   {path.name}")
            signed += 1
        else:
            print(f"SKIP: {path.name}")
            skipped += 1
    except Exception as e:
        print(f"FAIL: {path.name}  ({e})")
        failed += 1

print()
print(f"Done. signed={signed}  skipped={skipped}  failed={failed}")
