"""The request guard: websites must not be able to drive CIU.

Run with: python -m pytest tests/test_guard.py

/api/status is used for the allowed cases because it is a GET that always
returns 200 and changes nothing. Asserting an exact 200 matters: an earlier
version of these tests asserted "not 403", which would also have passed if
the route had been deleted, or if it had thrown a 500.
"""
import pytest
from fastapi.testclient import TestClient

from ciu.server import PORT, app

c = TestClient(app, base_url=f"http://127.0.0.1:{PORT}")

EXTENSION = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"


def code(method, path, **kw):
    return c.request(method, path, **kw).status_code


# ------------------------------------------------------------- allowed

def test_non_browser_clients_allowed():
    """curl, the Blender addon, the Godot addon: no Origin header at all."""
    assert code("GET", "/api/status") == 200


def test_own_page_allowed():
    assert code("GET", "/api/status",
                headers={"origin": f"http://127.0.0.1:{PORT}"}) == 200


def test_own_page_allowed_via_localhost():
    assert code("GET", "/api/status",
                headers={"origin": f"http://localhost:{PORT}"}) == 200


def test_trailing_slash_on_origin_allowed():
    assert code("GET", "/api/status",
                headers={"origin": f"http://127.0.0.1:{PORT}/"}) == 200


def test_ipv6_loopback_host_allowed():
    assert code("GET", "/api/status", headers={"host": f"[::1]:{PORT}"}) == 200


# ------------------------------------------------------------- refused

def test_websites_refused():
    for o in ("https://evil.example", "http://localhost:3000", "null"):
        assert code("GET", "/api/status", headers={"origin": o}) == 403
        assert code("POST", "/api/unload", headers={"origin": o}) == 403
        # A text/plain POST needs no preflight, so this is the shape a site
        # would actually use against the proxy.
        assert code("POST", "/v1/chat/completions", headers={
            "origin": o, "content-type": "text/plain"}, content="{}") == 403


def test_unknown_extensions_refused():
    """Not every extension the user has installed gets the GPU."""
    for o in ("chrome-extension://someotherextensionidgoeshere00",
              "moz-extension://aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
              "safari-web-extension://abc"):
        assert code("GET", "/api/status", headers={"origin": o}) == 403


def test_dns_rebinding_refused():
    assert code("GET", "/api/status",
                headers={"host": f"rebind.example:{PORT}"}) == 403


def test_rebound_host_refused_even_with_good_origin():
    """Host is checked first, so a forged-looking Origin does not help."""
    assert code("GET", "/api/status", headers={
        "host": f"rebind.example:{PORT}",
        "origin": f"http://127.0.0.1:{PORT}"}) == 403


# ------------------------------------------------------------- allowlists

def test_extension_allowlist_env(monkeypatch):
    monkeypatch.setenv("CIU_EXTENSION_ORIGINS", EXTENSION)
    assert code("GET", "/api/status", headers={"origin": EXTENSION}) == 200


def test_extension_allowlist_is_exact(monkeypatch):
    monkeypatch.setenv("CIU_EXTENSION_ORIGINS", EXTENSION)
    assert code("GET", "/api/status",
                headers={"origin": "chrome-extension://somethingelse"}) == 403


def test_web_allowlist_env(monkeypatch):
    monkeypatch.setenv("CIU_ALLOWED_ORIGINS", "http://localhost:3000")
    assert code("GET", "/api/status",
                headers={"origin": "http://localhost:3000"}) == 200


def test_web_allowlist_accepts_several(monkeypatch):
    monkeypatch.setenv("CIU_ALLOWED_ORIGINS",
                       "http://localhost:3000, http://localhost:5173/")
    for o in ("http://localhost:3000", "http://localhost:5173"):
        assert code("GET", "/api/status", headers={"origin": o}) == 200


# ------------------------------------------------------- route invariant

def test_no_get_route_changes_state():
    """The guard lets through cross-site GETs, because they carry no Origin
    and that is how curl looks. That is only safe while every GET is
    read-only, so this asserts the property the guard depends on."""
    safe = {"/api/status", "/", "/v1/{path:path}",
            "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        if "GET" in methods:
            assert route.path in safe, (
                f"{route.path} answers GET. If it changes state it must be "
                f"POST: a website can issue a cross-site GET with no Origin "
                f"header and the guard cannot tell it from curl.")
