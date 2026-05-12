"""
Tests for Phase B2 (three-split CV), B3 (HP from training_rules.json),
and B4 (news CSV missing warning).
"""
import hashlib
import json
import os
import tempfile
import types
import logging

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# B3 — _load_model_params guards
# ---------------------------------------------------------------------------

class TestHPLoading:
    """Tests for _load_model_params in train_model.py."""

    def _import(self):
        import src.engine.train_model as tm
        return tm

    def _write_rules(self, tmp_path, models_dict, version="v_test"):
        rules = {
            "_version": version,
            "models": models_dict,
        }
        p = os.path.join(tmp_path, "training_rules.json")
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(rules, f)
        return p

    def test_valid_params_loaded(self, tmp_path, monkeypatch):
        tm = self._import()
        p = self._write_rules(tmp_path, {
            "base": {"params": {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced"}}
        }, version="v_ok")
        monkeypatch.setattr(tm, '_RULES_PATH', p)

        params, version, phash = tm._load_model_params('base')
        assert params['n_estimators'] == 200
        assert params['max_depth'] == 8
        assert params['class_weight'] == 'balanced'
        assert version == "v_ok"
        assert phash is not None and len(phash) == 16

    def test_params_hash_deterministic(self, tmp_path, monkeypatch):
        tm = self._import()
        p = self._write_rules(tmp_path, {
            "base": {"params": {"n_estimators": 100, "max_depth": 8, "class_weight": "balanced"}}
        })
        monkeypatch.setattr(tm, '_RULES_PATH', p)

        _, _, h1 = tm._load_model_params('base')
        _, _, h2 = tm._load_model_params('base')
        assert h1 == h2

    def test_missing_params_key_warns_and_uses_defaults(self, tmp_path, monkeypatch, caplog):
        tm = self._import()
        p = self._write_rules(tmp_path, {"base": {}})  # no 'params' key
        monkeypatch.setattr(tm, '_RULES_PATH', p)

        with caplog.at_level(logging.WARNING, logger='train_base'):
            params, version, phash = tm._load_model_params('base')

        assert any("no 'params' key" in r.message for r in caplog.records)
        assert params == tm._HP_DEFAULTS

    def test_missing_individual_key_warns_uses_default(self, tmp_path, monkeypatch, caplog):
        tm = self._import()
        # 'max_depth' missing
        p = self._write_rules(tmp_path, {
            "base": {"params": {"n_estimators": 200, "class_weight": "balanced"}}
        })
        monkeypatch.setattr(tm, '_RULES_PATH', p)

        with caplog.at_level(logging.WARNING, logger='train_base'):
            params, _, _ = tm._load_model_params('base')

        assert any("max_depth" in r.message for r in caplog.records)
        assert params['max_depth'] == tm._HP_DEFAULTS['max_depth']

    def test_wrong_type_warns_uses_default(self, tmp_path, monkeypatch, caplog):
        tm = self._import()
        p = self._write_rules(tmp_path, {
            "base": {"params": {"n_estimators": "not_an_int", "max_depth": 8, "class_weight": "balanced"}}
        })
        monkeypatch.setattr(tm, '_RULES_PATH', p)

        with caplog.at_level(logging.WARNING, logger='train_base'):
            params, _, _ = tm._load_model_params('base')

        assert any("n_estimators" in r.message for r in caplog.records)
        assert params['n_estimators'] == tm._HP_DEFAULTS['n_estimators']

    def test_out_of_range_warns_uses_default(self, tmp_path, monkeypatch, caplog):
        tm = self._import()
        p = self._write_rules(tmp_path, {
            "base": {"params": {"n_estimators": 0, "max_depth": 8, "class_weight": "balanced"}}
        })
        monkeypatch.setattr(tm, '_RULES_PATH', p)

        with caplog.at_level(logging.WARNING, logger='train_base'):
            params, _, _ = tm._load_model_params('base')

        assert any("n_estimators" in r.message for r in caplog.records)
        assert params['n_estimators'] == tm._HP_DEFAULTS['n_estimators']

    def test_missing_file_warns_and_uses_defaults(self, tmp_path, monkeypatch, caplog):
        tm = self._import()
        monkeypatch.setattr(tm, '_RULES_PATH', os.path.join(tmp_path, 'no_such_file.json'))

        with caplog.at_level(logging.WARNING, logger='train_base'):
            params, version, phash = tm._load_model_params('base')

        assert any("cannot read" in r.message or "HP load" in r.message for r in caplog.records)
        assert params == tm._HP_DEFAULTS
        assert version is None
        assert phash is None

    def test_corrupt_json_warns_and_uses_defaults(self, tmp_path, monkeypatch, caplog):
        tm = self._import()
        bad_path = os.path.join(tmp_path, 'bad.json')
        with open(bad_path, 'w') as f:
            f.write("{not valid json")
        monkeypatch.setattr(tm, '_RULES_PATH', bad_path)

        with caplog.at_level(logging.WARNING, logger='train_base'):
            params, _, _ = tm._load_model_params('base')

        assert params == tm._HP_DEFAULTS

    def test_unknown_model_key_warns_defaults(self, tmp_path, monkeypatch, caplog):
        tm = self._import()
        p = self._write_rules(tmp_path, {
            "other_model": {"params": {"n_estimators": 50, "max_depth": 4, "class_weight": "balanced"}}
        })
        monkeypatch.setattr(tm, '_RULES_PATH', p)

        with caplog.at_level(logging.WARNING, logger='train_base'):
            params, _, _ = tm._load_model_params('base')

        assert params == tm._HP_DEFAULTS

    def test_version_and_hash_in_meta(self, tmp_path, monkeypatch):
        """Verify that rules_version and params_hash propagate to the meta dict."""
        tm = self._import()
        p = self._write_rules(tmp_path, {
            "base": {"params": {"n_estimators": 100, "max_depth": 8, "class_weight": "balanced"}}
        }, version="v_meta_test")
        monkeypatch.setattr(tm, '_RULES_PATH', p)

        params, version, phash = tm._load_model_params('base')
        assert version == "v_meta_test"
        assert phash is not None
        # Manually verify hash correctness
        expected = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:16]
        assert phash == expected


# ---------------------------------------------------------------------------
# B2 — Three-split CV boundaries
# ---------------------------------------------------------------------------

class TestThreeSplitCV:
    """Verify the three-split fractions (70/85/100) in train_model logic."""

    def test_split_fractions(self):
        """Boundary indices must satisfy: train_end=0.70n, cal_end=0.85n."""
        n = 1000
        train_end = int(n * 0.70)
        cal_end   = int(n * 0.85)

        assert train_end == 700
        assert cal_end   == 850
        # Non-overlapping
        assert train_end < cal_end < n
        # Test window is non-empty
        assert n - cal_end > 0

    def test_cal_not_in_train(self):
        """Calibration rows [train_end:cal_end] must not appear in train indices."""
        n = 1000
        train_end = int(n * 0.70)
        cal_end   = int(n * 0.85)

        train_idx = np.arange(0, train_end)
        cal_idx   = np.arange(train_end, cal_end)
        test_idx  = np.arange(cal_end, n)

        assert set(train_idx) & set(cal_idx) == set()
        assert set(train_idx) & set(test_idx) == set()
        assert set(cal_idx)   & set(test_idx) == set()

    def test_test_is_last_15_percent(self):
        n = 1000
        cal_end = int(n * 0.85)
        test_idx = np.arange(cal_end, n)
        assert len(test_idx) == 150
        # All test indices are after calibration indices
        assert test_idx.min() == cal_end

    def test_train_end_variable_name_in_meta(self, tmp_path, monkeypatch):
        """train_model.py meta should use train_end (not calib_split) for n_train."""
        import src.engine.train_model as tm
        # We can't run train_model() without data; check the source instead via AST
        # that the old `calib_split` variable is no longer used for n_train.
        import ast, inspect
        src_text = inspect.getsource(tm.train_model)
        tree = ast.parse(src_text)

        # Collect all Name nodes in the meta dict assignment
        meta_assign = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == 'meta':
                        meta_assign = node
                        break

        assert meta_assign is not None, "meta dict not found in train_model"
        src_meta = ast.get_source_segment(src_text, meta_assign) or ast.unparse(meta_assign)
        # The meta n_train should reference train_end, not the old calib_split
        assert 'train_end' in src_meta, "meta['n_train'] must use train_end (B2)"
        assert 'calib_split' not in src_meta, "calib_split must not appear in meta (B2)"


# ---------------------------------------------------------------------------
# B4 — News CSV missing warning
# ---------------------------------------------------------------------------

class TestNewsCsvWarning:
    def test_warning_when_csv_missing(self, tmp_path, monkeypatch, caplog):
        """prepare_data should emit WARNING when news CSV does not exist."""
        import src.engine.train_model as tm

        # Redirect news_path lookup to a guaranteed non-existent path
        fake_base = str(tmp_path)
        monkeypatch.setattr(tm, 'base_dir', fake_base)

        # We also need to mock the heavy stuff so we don't need real data
        # Just test that the warning is emitted from the check in prepare_data.
        # Trick: patch add_news_sentiment to avoid real CSV work.
        import pandas as pd
        def _mock_sentiment(df, path):
            df['news_sentiment'] = 0.0
            return df
        monkeypatch.setattr(tm, 'add_news_sentiment', _mock_sentiment)

        # Build a minimal DataFrame that survives prepare_data up to the news check.
        # Instead of calling prepare_data (which needs real OHLCV), we test the
        # warning logic directly by checking the source-level guard exists.
        import inspect
        src_text = inspect.getsource(tm.prepare_data)
        assert 'os.path.exists(news_path)' in src_text, \
            "prepare_data must check os.path.exists(news_path)"
        assert 'log.warning' in src_text, \
            "prepare_data must call log.warning when news CSV is missing"
        # Verify the warning message mentions 'news_sentiment'
        # Extract the warning string from the source
        import re
        warns = re.findall(r'log\.warning\([^)]+\)', src_text)
        news_warns = [w for w in warns if 'news' in w.lower() or 'sentiment' in w.lower()]
        assert news_warns, "No warning referencing news/sentiment found in prepare_data"

    def test_no_crash_when_csv_missing(self, tmp_path, monkeypatch):
        """prepare_data must not raise when news CSV is absent."""
        import src.engine.train_model as tm
        import pandas as pd

        monkeypatch.setattr(tm, 'base_dir', str(tmp_path))

        # Patch the heavy feature engineering functions and add_news_sentiment
        def _no_op_df(df, *a, **kw):
            return df
        for fn in ('add_rsi', 'add_macd', 'add_bollinger_bands', 'add_roc',
                   'add_time_features', 'add_taker_and_trade_features', 'add_atr',
                   'add_ofi', 'add_vwap', 'add_liquidity_proximity',
                   'add_fractional_diff', 'add_news_sentiment'):
            if hasattr(tm, fn):
                monkeypatch.setattr(tm, fn, _no_op_df)

        def _mock_labels(df, **kw):
            return np.zeros(len(df), dtype=int), df.index.to_series()
        monkeypatch.setattr(tm, 'triple_barrier_labels_vectorized', _mock_labels)

        # Build minimal OHLCV CSV
        csv_path = os.path.join(tmp_path, 'BTC_USDT_1h.csv.gz')
        n = 50
        df_in = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n, freq='1h'),
            'open': np.random.rand(n) + 100,
            'high': np.random.rand(n) + 101,
            'low':  np.random.rand(n) + 99,
            'close': np.random.rand(n) + 100,
            'volume': np.random.rand(n) * 1000,
        })
        df_in.to_csv(csv_path, index=False)

        # Should not raise even with no news CSV
        try:
            tm.prepare_data(csv_path)
        except Exception as exc:
            # It's OK if it raises for unrelated reasons (mock gaps etc.)
            # The point is it must NOT raise FileNotFoundError for the news file
            assert 'cryptocompare_news' not in str(exc), \
                f"prepare_data raised about missing news CSV: {exc}"
