"""Probe what the loaded scalping model actually expects."""
import sys, joblib, io
sys.path.insert(0, r'D:/test 2/AI trading assistance')
from src.utils.model_integrity import verify_and_load_bytes

m = joblib.load(io.BytesIO(verify_and_load_bytes(
    r'D:/test 2/AI trading assistance/models/scalping_model.joblib')))
print(f'wrapper: {type(m).__name__}')

# Drill into calibrated_classifiers_
if hasattr(m, 'calibrated_classifiers_'):
    cc = m.calibrated_classifiers_[0]
    print(f'cc: {type(cc).__name__}')
    if hasattr(cc, 'estimator'):
        e = cc.estimator
        print(f'inner: {type(e).__name__}')
        print(f'inner.n_features_in_: {getattr(e, "n_features_in_", None)}')
        names = getattr(e, "feature_names_in_", None)
        print(f'inner.feature_names_in_ len: {len(names) if names is not None else None}')
        if names is not None:
            print(f'  names: {list(names)}')

# Try a real predict_proba with the FEATURE_COLUMNS to see what happens
import pandas as pd
from src.engine.train_scalping_model import FEATURE_COLUMNS
print(f'\nFEATURE_COLUMNS len: {len(FEATURE_COLUMNS)}')
df = pd.DataFrame([{c: 0.0 for c in FEATURE_COLUMNS}])
try:
    p = m.predict_proba(df)
    print(f'predict_proba(21cols) OK shape={p.shape}')
except Exception as e:
    print(f'predict_proba(21cols) ERROR: {e}')

# Also try with just 17
short_cols = FEATURE_COLUMNS[:17]
df17 = pd.DataFrame([{c: 0.0 for c in short_cols}])
try:
    p = m.predict_proba(df17)
    print(f'predict_proba(17cols) OK shape={p.shape}')
except Exception as e:
    print(f'predict_proba(17cols) ERROR: {e}')
