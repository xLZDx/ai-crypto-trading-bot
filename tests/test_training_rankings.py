"""Phase B (2026-05-14) — /api/training/rankings tests.

Validates the composite-score endpoint that drives the Top-5 / Bottom-5
efficiency badges on the Model Training card.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dashboard.app import app, _ranking_score  # noqa: E402


class TestRankingScore(unittest.TestCase):
    """Pure function — exercise the scoring math directly."""

    def test_full_metrics_weighted_correctly(self) -> None:
        # acc=60 (w 0.5), auc=0.6 -> 60 (w 0.3), wr=70 (w 0.2)
        s = _ranking_score({'accuracy_test': 60.0, 'auc_roc': 0.6, 'bull_wr': 70.0})
        # 0.5*60 + 0.3*60 + 0.2*70 = 30 + 18 + 14 = 62
        self.assertAlmostEqual(s, 62.0, places=3)

    def test_missing_auc_renormalizes(self) -> None:
        s = _ranking_score({'accuracy_test': 60.0, 'auc_roc': None, 'bull_wr': 70.0})
        # 0.5*60 + 0.2*70 / 0.7 = (30 + 14) / 0.7 = 62.857
        self.assertAlmostEqual(s, (30.0 + 14.0) / 0.7, places=3)

    def test_only_accuracy_returns_accuracy(self) -> None:
        s = _ranking_score({'accuracy_test': 55.0})
        self.assertAlmostEqual(s, 55.0, places=3)

    def test_falls_back_to_win_precision_when_no_bull_wr(self) -> None:
        s = _ranking_score({'accuracy_test': 50.0, 'auc_roc': 0.5, 'win_precision': 60.0})
        # 0.5*50 + 0.3*50 + 0.2*60 = 25 + 15 + 12 = 52
        self.assertAlmostEqual(s, 52.0, places=3)

    def test_no_metrics_returns_none(self) -> None:
        s = _ranking_score({})
        self.assertIsNone(s)

    def test_invalid_values_ignored(self) -> None:
        s = _ranking_score({'accuracy_test': 'not_a_number', 'auc_roc': 0.6})
        # acc was invalid -> only auc counts -> score = 50 + (0.6-0.5)*100 = 60
        self.assertAlmostEqual(s, 60.0, places=3)

    def test_auc_below_random_scores_below_50(self) -> None:
        s = _ranking_score({'auc_roc': 0.4})
        # 50 + (0.4 - 0.5)*100 = 40
        self.assertAlmostEqual(s, 40.0, places=3)


class TestRankingEndpoint(unittest.TestCase):
    """End-to-end: hit /api/training/rankings with mocked strategy_full."""

    def setUp(self) -> None:
        app.config['TESTING'] = True
        self.client = app.test_client()
        api_key = os.environ.get('DASHBOARD_API_KEY')
        self.headers = {'X-API-Key': api_key} if api_key else {}
        # Stub payload — 7 models, varying performance
        self._stub_payload = {
            'ml_models': [
                {'key': 'a', 'label': 'A', 'timeframe': '1h', 'market': 'SPOT',
                 'accuracy_test': 70, 'auc_roc': 0.7, 'bull_wr': 70, 'model_exists': True},
                {'key': 'b', 'label': 'B', 'timeframe': '1h', 'market': 'SPOT',
                 'accuracy_test': 65, 'auc_roc': 0.65, 'bull_wr': 65, 'model_exists': True},
                {'key': 'c', 'label': 'C', 'timeframe': '1h', 'market': 'SPOT',
                 'accuracy_test': 60, 'auc_roc': 0.6, 'bull_wr': 60, 'model_exists': True},
                {'key': 'd', 'label': 'D', 'timeframe': '1h', 'market': 'SPOT',
                 'accuracy_test': 55, 'auc_roc': 0.55, 'bull_wr': 55, 'model_exists': True},
                {'key': 'e', 'label': 'E', 'timeframe': '1h', 'market': 'SPOT',
                 'accuracy_test': 50, 'auc_roc': 0.5, 'bull_wr': 50, 'model_exists': True},
                {'key': 'f', 'label': 'F', 'timeframe': '1h', 'market': 'SPOT',
                 'accuracy_test': 45, 'auc_roc': 0.45, 'bull_wr': 45, 'model_exists': True},
                {'key': 'g', 'label': 'G', 'timeframe': '1h', 'market': 'SPOT',
                 'accuracy_test': 40, 'auc_roc': 0.4, 'bull_wr': 40, 'model_exists': True},
                # An untrained row — should be excluded by default
                {'key': 'h', 'label': 'H', 'timeframe': '1h', 'market': 'SPOT',
                 'accuracy_test': 50, 'auc_roc': 0.5, 'model_exists': False},
            ]
        }

    def _patch_strategy_full(self):
        class _StubResponse:
            def __init__(self, payload): self._payload = payload
            def get_json(self): return self._payload
        return mock.patch('src.dashboard.app.strategy_full',
                          return_value=_StubResponse(self._stub_payload))

    def test_returns_200_and_shape(self) -> None:
        with self._patch_strategy_full():
            r = self.client.get('/api/training/rankings', headers=self.headers)
            self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
            data = r.get_json()
            self.assertIn('ranked', data)
            self.assertIn('top_n', data)
            self.assertIn('bottom_n', data)
            self.assertIn('score_formula', data)
            self.assertEqual(data['n_rankable'], 7)  # h is excluded

    def test_top_5_are_highest_scoring(self) -> None:
        with self._patch_strategy_full():
            r = self.client.get('/api/training/rankings', headers=self.headers)
            data = r.get_json()
            top_keys = [row['key'] for row in data['top_n']]
            self.assertEqual(top_keys, ['a', 'b', 'c', 'd', 'e'])
            self.assertEqual(data['top_n'][0]['badge'], 'TOP_1')
            self.assertEqual(data['top_n'][4]['badge'], 'TOP_5')

    def test_bottom_5_worst_first(self) -> None:
        with self._patch_strategy_full():
            r = self.client.get('/api/training/rankings', headers=self.headers)
            data = r.get_json()
            bot_keys = [row['key'] for row in data['bottom_n']]
            # BOT_1 = worst (g, score~40). Then f, e, d, c.
            self.assertEqual(bot_keys[0], 'g')
            self.assertEqual(data['bottom_n'][0]['badge'], 'BOT_1')

    def test_top_param_caps_count(self) -> None:
        with self._patch_strategy_full():
            r = self.client.get('/api/training/rankings?top=2&bottom=1',
                                headers=self.headers)
            data = r.get_json()
            self.assertEqual(len(data['top_n']), 2)
            self.assertEqual(len(data['bottom_n']), 1)

    def test_include_untrained_adds_h(self) -> None:
        with self._patch_strategy_full():
            r = self.client.get('/api/training/rankings?include_untrained=1',
                                headers=self.headers)
            data = r.get_json()
            self.assertEqual(data['n_rankable'], 8)

    def test_score_in_response_matches_pure_function(self) -> None:
        with self._patch_strategy_full():
            r = self.client.get('/api/training/rankings', headers=self.headers)
            data = r.get_json()
            top = data['top_n'][0]
            # Row 'a' = acc 70, auc 0.7, bull_wr 70
            expected = _ranking_score({
                'accuracy_test': 70, 'auc_roc': 0.7, 'bull_wr': 70,
            })
            self.assertAlmostEqual(top['score'], round(expected, 3), places=2)


if __name__ == '__main__':
    unittest.main()
