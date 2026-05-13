"""Trace EXACTLY what _get_model_features returns for the live scalping model."""
import sys, os, joblib, io
sys.path.insert(0, r'D:/test 2/AI trading assistance')
os.environ.setdefault('MODEL_MANIFEST_KEY', '')

from src.analysis.ml_predictor import MLPredictor
mp = MLPredictor(model_filename='scalping_model.joblib', model_type='scalping')
print(f'is_loaded: {mp.is_loaded}')
print(f'_embedded_features: {mp._embedded_features}')

# Test each path in _get_model_features
print('\n--- Path 0: embedded ---')
print(f'embedded: {mp._embedded_features}')

print('\n--- Path 1: meta JSON ---')
import json
meta_path = mp.model_path.replace('.joblib', '_meta.json')
with open(meta_path) as f:
    meta = json.load(f)
print(f'meta has "features" key: {"features" in meta}')
print(f'meta has "feature_names" key: {"feature_names" in meta}')

print('\n--- Path 2: recursive search ---')
def find_features(obj, depth=0, path=''):
    if depth > 5 or obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        names = getattr(obj, 'feature_names_in_')
        if names is not None and len(names) > 0:
            print(f'  found feature_names_in_ at {path}: {len(names)} names')
            return list(names)
    for attr in ["estimator", "base_estimator", "best_estimator_", "model", "_final_estimator", "step"]:
        if hasattr(obj, attr):
            sub = getattr(obj, attr)
            if sub is not None:
                res = find_features(sub, depth + 1, path + '.' + attr)
                if res: return res
    if hasattr(obj, "calibrated_classifiers_"):
        for i, clf in enumerate(getattr(obj, "calibrated_classifiers_")):
            res = find_features(clf, depth + 1, path + f'.cc[{i}]')
            if res: return res
    if hasattr(obj, "steps"):
        for name, step in getattr(obj, "steps"):
            res = find_features(step, depth + 1, path + f'.step[{name}]')
            if res: return res
    # Try XGBoost feature_names attribute
    if hasattr(obj, "feature_names"):
        names = getattr(obj, 'feature_names')
        if names is not None and len(names) > 0:
            print(f'  found XGB.feature_names at {path}: {len(names)} names')
            return list(names)
    if hasattr(obj, "get_booster"):
        try:
            booster = obj.get_booster()
            if booster.feature_names:
                print(f'  found booster.feature_names at {path}: {len(booster.feature_names)} names')
                return list(booster.feature_names)
        except Exception:
            pass
    return None

res = find_features(mp.model, path='model')
print(f'\nFinal _get_model_features result:')
result = mp._get_model_features()
print(f'  count: {len(result)}')
print(f'  features: {result}')
