"""
Behavioral tests for the CIO Agent live-mode cluster wiring.

Covers `make_cluster_callbacks` HTTP submitter/poller pair + `live_mode`
constructor. Uses requests-mock-style stubs (monkeypatch on _requests.post/get).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_make_cluster_callbacks_returns_two_callables():
    from src.engine.cio_agent import make_cluster_callbacks
    sub, poll = make_cluster_callbacks()
    assert callable(sub)
    assert callable(poll)


def test_submitter_posts_to_cluster_submit(monkeypatch):
    """submitter(spec) → POST /api/cluster/submit and returns task_id."""
    from src.engine import cio_agent
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = {'ok': True, 'task_id': 'abc-123'}
    fake_post.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr('requests.post', fake_post)
    monkeypatch.setattr('requests.get', MagicMock())

    sub, _poll = cio_agent.make_cluster_callbacks(cluster_url='http://test:7700', api_key='k')
    tid = sub({'model_type': 'meta', 'timeframe': '1h'})
    assert tid == 'abc-123'
    args, kwargs = fake_post.call_args
    assert args[0] == 'http://test:7700/api/cluster/submit'
    assert kwargs['json']['model_type'] == 'meta'
    assert kwargs['headers']['X-API-Key'] == 'k'


def test_poller_filters_task_list_by_id(monkeypatch):
    """poller(task_id) → GET /api/cluster/tasks and returns the matching row."""
    from src.engine import cio_agent
    fake_get = MagicMock()
    fake_get.return_value.json.return_value = [
        {'task_id': 'abc-123', 'status': 'done',  'result': {'sortino': 1.7}, 'error': ''},
        {'task_id': 'xyz-999', 'status': 'failed', 'result': {}, 'error': 'boom'},
    ]
    fake_get.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr('requests.get', fake_get)
    monkeypatch.setattr('requests.post', MagicMock())

    _sub, poll = cio_agent.make_cluster_callbacks(cluster_url='http://test:7700', api_key='k')
    res = poll('abc-123')
    assert res['status'] == 'done'
    assert res['result']['sortino'] == 1.7

    res_other = poll('xyz-999')
    assert res_other['status'] == 'failed'

    res_missing = poll('does-not-exist')
    assert res_missing['status'] == 'unknown'


def test_live_mode_constructs_agent_with_callbacks(monkeypatch):
    """live_mode() returns a CIOAgent with both callbacks wired."""
    from src.engine import cio_agent
    monkeypatch.setattr('requests.post', MagicMock())
    monkeypatch.setattr('requests.get',  MagicMock())
    agent = cio_agent.live_mode(study_name='test_live')
    assert agent.study_name == 'test_live'
    assert agent.task_submitter is not None
    assert agent.task_status_poller is not None
    assert callable(agent.task_submitter)
    assert callable(agent.task_status_poller)


def test_cluster_callbacks_use_api_key_from_env(monkeypatch):
    """When api_key=None, callbacks fall back to env CLUSTER_API_KEY or
    DASHBOARD_API_KEY."""
    from src.engine import cio_agent
    monkeypatch.setenv('CLUSTER_API_KEY', 'env-cluster-key')
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = {'task_id': 'tid'}
    fake_post.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr('requests.post', fake_post)
    monkeypatch.setattr('requests.get', MagicMock())

    sub, _poll = cio_agent.make_cluster_callbacks(cluster_url='http://test:7700')
    sub({'x': 1})
    _args, kwargs = fake_post.call_args
    assert kwargs['headers']['X-API-Key'] == 'env-cluster-key'


def test_cluster_callbacks_falls_back_to_dashboard_key(monkeypatch):
    from src.engine import cio_agent
    monkeypatch.delenv('CLUSTER_API_KEY', raising=False)
    monkeypatch.setenv('DASHBOARD_API_KEY', 'env-dash-key')
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = {'task_id': 'tid'}
    fake_post.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr('requests.post', fake_post)
    monkeypatch.setattr('requests.get', MagicMock())

    sub, _poll = cio_agent.make_cluster_callbacks(cluster_url='http://test:7700')
    sub({'x': 1})
    _args, kwargs = fake_post.call_args
    assert kwargs['headers']['X-API-Key'] == 'env-dash-key'


def test_submitter_propagates_http_error(monkeypatch):
    """raise_for_status on 4xx/5xx must bubble up so CIO can prune the trial."""
    from src.engine import cio_agent
    fake_post = MagicMock()
    fake_post.return_value.raise_for_status.side_effect = RuntimeError('400 Bad Request')
    monkeypatch.setattr('requests.post', fake_post)
    monkeypatch.setattr('requests.get', MagicMock())

    sub, _poll = cio_agent.make_cluster_callbacks(cluster_url='http://test:7700', api_key='k')
    with pytest.raises(RuntimeError, match='400'):
        sub({'x': 1})


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
