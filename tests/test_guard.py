"""The request guard: websites must not be able to drive CIU.

Run with: python -m pytest tests/test_guard.py
"""
from fastapi.testclient import TestClient

from ciu.server import app

c = TestClient(app, base_url="http://127.0.0.1:8674")


def code(method, path, **kw):
    return c.request(method, path, **kw).status_code


def test_non_browser_clients_allowed():
    assert code("GET", "/api/status") == 200


def test_extensions_allowed():
    for o in ("chrome-extension://abc", "moz-extension://abc"):
        assert code("POST", "/v1/chat/completions", headers={"origin": o}, json={}) != 403


def test_own_page_allowed():
    assert code("POST", "/api/unload", headers={"origin": "http://127.0.0.1:8674"}) == 200


def test_websites_refused():
    for o in ("https://evil.example", "http://localhost:3000", "null"):
        assert code("POST", "/api/unload", headers={"origin": o}) == 403
        assert code("POST", "/v1/chat/completions", headers={
            "origin": o, "content-type": "text/plain"}, content="{}") == 403


def test_dns_rebinding_refused():
    assert code("GET", "/api/status", headers={"host": "rebind.example:8674"}) == 403


def test_allowlist_env(monkeypatch):
    monkeypatch.setenv("CIU_ALLOWED_ORIGINS", "http://localhost:3000")
    assert code("POST", "/api/unload", headers={"origin": "http://localhost:3000"}) == 200
