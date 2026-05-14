"""Phase A (2026-05-14) — regression test for the feature-shape-mismatch bug.

Bug: bot loaded a 20-feature joblib model. A concurrent trainer rewrote the
meta JSON to declare 22 features. MLPredictor's _get_model_features() re-read
the meta JSON on every predict call, so it started returning 22 columns to an
in-memory 20-feature model -> XGBoost raised "Feature shape mismatch,
expected: 20, got 22" ~2300 times.

Fix: cache the feature list at __init__ so it stays in lockstep with the
joblib payload that was loaded into memory.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _train_toy_model(n_features: int, feature_names: list[str]) -> RandomForestClassifier:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, n_features))
    y = (X.sum(axis=1) > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    import pandas as pd
    clf.fit(pd.DataFrame(X, columns=feature_names), y)
    return clf


class TestFeatureListLockedAtInit(unittest.TestCase):
    """The feature list MUST be frozen at __init__ so that a post-init
    meta JSON rewrite by another process cannot make us pass the wrong
    number of columns to an in-memory model."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        # Sit ourselves inside a fake models/ directory so MLPredictor's
        # path construction works. MLPredictor's __file__ lives in
        # src/analysis/, and it walks three levels up to find the project
        # root and append "models". We monkey-patch model_path directly
        # via the public attribute instead.
        self._features_v1 = [f"f{i}" for i in range(20)]
        self._features_v2 = [f"f{i}" for i in range(22)]
        self._model_path = self.tmp_dir / "toy_model.joblib"
        self._meta_path = self.tmp_dir / "toy_model_meta.json"
        clf_20 = _train_toy_model(20, self._features_v1)

        # Sign the joblib via the same path verify_and_load_bytes() expects.
        # Easiest: use the model_integrity helper's sign_model() if it's a
        # simple wrap. If signing is unavailable in the test env, fall back
        # to a direct joblib.dump and patch verify_and_load_bytes to just
        # read the file unchanged.
        try:
            from src.utils.model_integrity import sign_model
            sign_model(clf_20, str(self._model_path))
        except Exception:
            joblib.dump(clf_20, self._model_path)

        # Write meta JSON declaring the OLD 20-feature shape (matches model)
        self._meta_path.write_text(json.dumps({
            "n_features": 20,
            "features": list(self._features_v1),
            "accuracy": 65.0,
        }))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_predictor(self):
        # MLPredictor builds its model_path from `models/<filename>` relative
        # to the project root. Easiest path: build it normally with an
        # unused filename, then point model_path at our tmp file before
        # the joblib load happens. Since __init__ does the load up front,
        # we instead monkey-patch the directory.
        from src.analysis.ml_predictor import MLPredictor
        from unittest import mock
        with mock.patch("src.analysis.ml_predictor.os.path.exists", return_value=True), \
             mock.patch("src.analysis.ml_predictor.verify_and_load_bytes",
                        return_value=self._model_path.read_bytes()), \
             mock.patch("src.analysis.ml_predictor.read_json",
                        side_effect=lambda p, default=None: json.loads(self._meta_path.read_text())):
            p = MLPredictor(model_filename="toy_model.joblib", model_type="base")
        # After __init__ returns, the predictor should have cached features.
        # Point its meta_path lookups at our tmp meta so subsequent reads
        # (the bug behavior) would hit our v2 rewrite.
        p.model_path = str(self._model_path)
        return p

    def test_features_cached_at_init(self) -> None:
        p = self._make_predictor()
        self.assertTrue(p.is_loaded)
        self.assertIsNotNone(p._features)
        self.assertEqual(len(p._features), 20, f"expected 20 features, got {len(p._features)}")

    def test_meta_rewrite_does_not_change_cached_features(self) -> None:
        """The core regression: after init, rewriting meta JSON to declare
        a different feature shape MUST NOT change what _get_model_features
        returns. The cached list must stay frozen."""
        p = self._make_predictor()
        cached_before = list(p._get_model_features())
        # Simulate concurrent trainer rewriting meta JSON to 22 features.
        self._meta_path.write_text(json.dumps({
            "n_features": 22,
            "features": list(self._features_v2),
            "accuracy": 65.0,
        }))
        cached_after = list(p._get_model_features())
        self.assertEqual(cached_before, cached_after,
                         "feature list must be frozen at init — "
                         "this is exactly the 2026-05-14 ×2315-error bug")
        self.assertEqual(len(cached_after), 20)

    def test_resolve_features_falls_back_for_unloaded_predictor(self) -> None:
        """If __init__ couldn't load (file missing), _get_model_features
        still works for callers that need the feature list — falls back
        through embedded -> meta -> hardcoded."""
        from src.analysis.ml_predictor import MLPredictor
        p = MLPredictor(model_filename="nonexistent_xyz.joblib", model_type="trend")
        self.assertFalse(p.is_loaded)
        # _features is None (no model loaded) but _get_model_features
        # should still return the hardcoded trend list (20 entries).
        feats = p._get_model_features()
        self.assertGreater(len(feats), 0)


class TestStaleMetaJsonDoesNotCauseShapeMismatch(unittest.TestCase):
    """Phase E (2026-05-14) — second-order regression. After Phase A locked
    features at init, the operator hit a NEW case where the meta JSON was
    rewritten to a STALE shape mid-session (n_features=20, features=[])
    while the in-memory model expected 22. The fix: _resolve_features now
    consults model.n_features_in_ FIRST and rejects candidate lists whose
    length doesn't match.
    """

    def test_resolve_features_matches_model_n_features_in(self) -> None:
        from src.analysis.ml_predictor import MLPredictor
        from unittest import mock
        # Build a fake model that exposes n_features_in_=22
        fake_model = mock.MagicMock()
        fake_model.n_features_in_ = 22
        # Strip attrs find_features would normally walk through.
        fake_model.feature_names_in_ = None
        # Construct the predictor without going through the real __init__.
        p = MLPredictor.__new__(MLPredictor)
        p.model = fake_model
        p.is_loaded = True
        p._embedded_features = None
        p._features = None
        p.model_path = "/nonexistent/trend_model.joblib"
        p.model_type = "trend"
        p.last_error = ""
        p.last_status = "init"
        p._last_confidence = 0.5
        feats = p._resolve_features()
        self.assertEqual(len(feats), 22,
                         f"Expected 22-length list (matches model.n_features_in_=22), got {len(feats)}")


if __name__ == "__main__":
    unittest.main()
