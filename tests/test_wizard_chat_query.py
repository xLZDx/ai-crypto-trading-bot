"""Phase C (2026-05-14) — Wizard chat regression tests.

Bug: ask_llm() routed free-text questions through AgenticLLM.evaluate_trade()
which forced an APPROVE/VETO JSON envelope. The LLM responded with
"this is a valid operational query and does not fall under the VETO criteria"
instead of answering. Fix: route through new AgenticLLM.query() free-form
path.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestAskLlmRoutesThroughQueryNotEvaluateTrade(unittest.TestCase):
    """Core regression — ask_llm MUST call AgenticLLM.query, NOT evaluate_trade."""

    def test_ask_llm_calls_query_method(self) -> None:
        from src.dashboard import wizard
        fake_llm = mock.MagicMock()
        fake_llm.is_active = True
        fake_llm.query.return_value = {
            'answer': 'Try increasing trees from 100 to 300 and add purged CV.',
            'model_used': 'gemini-2.0-flash', 'source': 'llm',
        }
        with mock.patch.object(wizard, 'AgenticLLM', create=True, return_value=fake_llm), \
             mock.patch('src.engine.agentic_llm.AgenticLLM', return_value=fake_llm):
            result = wizard.ask_llm(
                'why does trend underperform during high volatility?',
                context={'model': 'trend', 'tf': '1h'},
            )
        # The free-form query path must be called.
        self.assertTrue(fake_llm.query.called, 'ask_llm must call query() — not evaluate_trade()')
        # evaluate_trade must NOT be called — that's the bug path.
        fake_llm.evaluate_trade.assert_not_called()
        # And we must return the answer the LLM produced.
        self.assertEqual(result['answer'],
                         'Try increasing trees from 100 to 300 and add purged CV.')
        self.assertEqual(result['source'], 'llm')

    def test_ask_llm_empty_question_returns_empty(self) -> None:
        from src.dashboard import wizard
        result = wizard.ask_llm('   ')
        self.assertEqual(result['source'], 'empty')

    def test_ask_llm_inactive_llm_returns_friendly_message(self) -> None:
        from src.dashboard import wizard
        fake_llm = mock.MagicMock()
        fake_llm.is_active = False
        with mock.patch('src.engine.agentic_llm.AgenticLLM', return_value=fake_llm):
            result = wizard.ask_llm('any question')
        self.assertEqual(result['source'], 'no_api_key')
        self.assertIn('GEMINI_API_KEY', result['answer'])


class TestQueryMethodOnAgenticLLM(unittest.TestCase):
    """The new query() method must reuse the cascade + budget guard."""

    def test_query_method_exists(self) -> None:
        from src.engine.agentic_llm import AgenticLLM
        self.assertTrue(hasattr(AgenticLLM, 'query'),
                        'AgenticLLM must have a query() method for free-form Q&A')

    def test_query_inactive_returns_disabled_message(self) -> None:
        from src.engine.agentic_llm import AgenticLLM
        # Build an instance and force is_active=False so we don't hit the API.
        with mock.patch.object(AgenticLLM, '__init__', return_value=None):
            llm = AgenticLLM()
            llm.is_active = False
            llm._client = None
            llm.api_key = None
            r = llm.query('test')
        self.assertEqual(r['source'], 'no_api_key')

    def test_query_returns_text_when_llm_responds(self) -> None:
        from src.engine import agentic_llm as al
        # Mock the whole cascade — force the first model to succeed.
        with mock.patch.object(al, '_ALL_MODELS', ['fake-model']), \
             mock.patch.object(al, '_is_cooled_down', return_value=False), \
             mock.patch.object(al, '_budget_filter', side_effect=lambda x: x), \
             mock.patch.object(al, '_record_call_cost'), \
             mock.patch.object(al, '_extract_token_counts', return_value=(10, 50)), \
             mock.patch.object(al, 'read_json', return_value={}):
            fake_resp = mock.MagicMock()
            fake_resp.text = '  Tune n_estimators to 300 and reduce learning_rate.  '
            fake_client = mock.MagicMock()
            fake_client.models.generate_content.return_value = fake_resp
            with mock.patch.object(al.AgenticLLM, '__init__', return_value=None):
                llm = al.AgenticLLM()
                llm.is_active = True
                llm._client = fake_client
                llm.api_key = 'fake'
                r = llm.query('How do I improve the trend model?')
        self.assertEqual(r['source'], 'llm')
        self.assertEqual(r['answer'], 'Tune n_estimators to 300 and reduce learning_rate.')
        self.assertEqual(r['model_used'], 'fake-model')


if __name__ == '__main__':
    unittest.main()
