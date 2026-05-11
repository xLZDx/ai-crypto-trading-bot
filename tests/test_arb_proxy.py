"""
Regression test for the trading-bot dashboard's arbitrage reverse-proxy.

Verifies that /arb/* and /api/arb/* on port 5000 forward correctly to the
arbitrage_strategy dashboard at port 5002 (per Q5 of the arb plan, with
the user-chosen reverse-proxy integration approach).

These tests use Flask's test client + mock the upstream HTTP call so they
work without a live arb dashboard process.

Run:
  python -m pytest tests/test_arb_proxy.py -v
  python tests/test_arb_proxy.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _client():
    from src.dashboard.app import app
    app.config["TESTING"] = True
    return app.test_client()


def _mock_response(status: int, body: bytes, headers: dict | None = None):
    m = MagicMock()
    m.status_code = status
    m.content = body
    raw = MagicMock()
    raw.headers.items.return_value = list((headers or {}).items())
    m.raw = raw
    return m


# --- /arb/ HTML proxy ----------------------------------------------------


def test_arb_html_proxied_when_upstream_returns_200() -> None:
    body = b"<html>arb dashboard</html>"
    mock = _mock_response(200, body, {"Content-Type": "text/html"})
    with patch("requests.request", return_value=mock):
        c = _client()
        r = c.get("/arb/")
    assert r.status_code == 200
    assert b"arb dashboard" in r.data


def test_arb_html_proxied_with_path() -> None:
    body = b"static-content"
    mock = _mock_response(200, body)
    with patch("requests.request", return_value=mock) as m:
        _client().get("/arb/some/static.css")
    args, kwargs = m.call_args
    assert kwargs["url"].endswith("/some/static.css"), kwargs["url"]


def test_arb_root_path_no_trailing_slash_redirects_or_proxies() -> None:
    """/arb (no slash) → either Flask redirects to /arb/ (308) or the route
    matches and proxies upstream. Both are acceptable; the failure mode is
    a 404."""
    mock = _mock_response(200, b"ok")
    with patch("requests.request", return_value=mock):
        c = _client()
        r = c.get("/arb", follow_redirects=True)
    assert r.status_code == 200, f"got HTTP {r.status_code}"


# --- /api/arb/* JSON proxy ----------------------------------------------


def test_api_arb_health_proxied() -> None:
    body = b'{"status":"ok","mode":"SHADOW"}'
    mock = _mock_response(200, body, {"Content-Type": "application/json"})
    with patch("requests.request", return_value=mock):
        r = _client().get("/api/arb/health")
    assert r.status_code == 200
    assert b'"status":"ok"' in r.data


def test_api_arb_with_query_string() -> None:
    mock = _mock_response(200, b"[]")
    with patch("requests.request", return_value=mock) as m:
        _client().get("/api/arb/opportunities?n=10&decision=GO")
    args, kwargs = m.call_args
    # The proxy passes params via the params kwarg
    assert "params" in kwargs
    assert kwargs["params"] == {"n": ["10"], "decision": ["GO"]}


def test_api_arb_post_method_forwarded() -> None:
    mock = _mock_response(200, b"ok")
    with patch("requests.request", return_value=mock) as m:
        _client().post("/api/arb/some_action", json={"x": 1})
    args, kwargs = m.call_args
    assert kwargs["method"] == "POST"
    assert b'"x"' in kwargs["data"]


# --- error handling -----------------------------------------------------


def test_proxy_returns_502_when_upstream_unreachable() -> None:
    import requests
    with patch(
        "requests.request",
        side_effect=requests.exceptions.ConnectionError("boom"),
    ):
        r = _client().get("/api/arb/health")
    assert r.status_code == 502
    body = r.get_json()
    assert body["error"] == "arb_dashboard_unreachable"
    assert "5002" in body["detail"] or "ARB_DASHBOARD_URL" in body["detail"]


def test_proxy_returns_502_on_other_errors() -> None:
    with patch("requests.request", side_effect=RuntimeError("oops")):
        r = _client().get("/api/arb/health")
    assert r.status_code == 502
    body = r.get_json()
    assert body["error"] == "arb_proxy_failed"
    assert "oops" in body["detail"]


# --- upstream URL configuration -----------------------------------------


def test_arb_upstream_url_default_is_5002() -> None:
    from src.dashboard.app import ARB_UPSTREAM_URL
    assert "5002" in ARB_UPSTREAM_URL


def test_proxy_strips_hop_headers() -> None:
    """Connection / Transfer-Encoding etc must not leak through."""
    body = b"ok"
    mock = _mock_response(200, body, {
        "Content-Type": "text/plain",
        "Transfer-Encoding": "chunked",
        "Connection": "keep-alive",
    })
    with patch("requests.request", return_value=mock):
        r = _client().get("/api/arb/health")
    # Hop-by-hop headers must NOT be in the proxied response
    assert "Transfer-Encoding" not in r.headers
    assert "Connection" not in r.headers
    assert "Content-Type" in r.headers


# --- index.html tab ------------------------------------------------------


def test_index_html_contains_arbitrage_tab() -> None:
    """Verify the Arbitrage nav button + tab pane were added to the template."""
    tmpl = (REPO_ROOT / "src" / "dashboard" / "templates" / "index.html").read_text(
        encoding="utf-8"
    )
    assert 'data-tab="arbitrage"' in tmpl
    assert 'id="tab-arbitrage"' in tmpl
    assert "/arb/" in tmpl  # iframe src
    assert "arbitrage:" in tmpl  # TAB_TITLES entry


def _run_all() -> int:
    failures: list[tuple[str, str]] = []
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} / {len(tests)} FAILED")
        return 1
    print(f"{len(tests)} / {len(tests)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
