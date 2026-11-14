"""CIU: one local model, shared by everything on the machine.

Serves three things on one port:
  /            a status page
  /api/*       CIU's own endpoints for the page
  /v1/*        an OpenAI-compatible API, proxied to llama-server

The point of the proxy is a stable address. Applications point at
127.0.0.1:8674/v1 once and keep working when the model behind it changes.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from .budget import (CONTEXT_STEPS, GIB, budget_for, context_options,
                     max_fitting_context, round_context)
from .catalog import CATALOG, by_id
from .gguf_read import read_shape
from .gguf_remote import file_size as remote_size, read_remote
from .hardware import detect
from .runner import LlamaRunner, RunConfig, find_llama_server
from . import bootstrap

PORT = 8674
MODEL_DIR = Path.home() / ".ciu" / "models"

app = FastAPI(title="CIU")
runner = LlamaRunner()


# ---------------------------------------------------------- request guard
#
# CIU listens on loopback, but loopback is not private from the browser. Any
# website the user visits can send requests to 127.0.0.1:8674: a form post to
# /api/unload, or a text/plain POST to /v1/chat/completions that burns the
# GPU. With DNS rebinding a site can even read the responses. Two checks
# close that:
#
#   Host   must name loopback. A rebound domain arrives with its own name in
#          Host, so this alone defeats DNS rebinding.
#   Origin browsers attach it to every cross-origin request and to all POSTs,
#          and pages cannot forge it. No Origin means a non-browser client
#          (curl, a Blender or Godot addon, a script), which is allowed.
#          CIU's own page is allowed, and so are the specific extensions
#          named in CIU_EXTENSION_ORIGINS. Everything else is refused.
#
# Two things this deliberately does NOT do.
#
# It does not refuse requests that carry no Origin, because that is every
# non-browser client CIU exists to serve. The consequence is that a website
# can still reach CIU with a cross-site GET, which carries no Origin: an
# <img src> or a top-level navigation. That is safe only while no GET
# endpoint changes state, which is true today (/api/status and / are the
# only ones, and both are read-only). Any new state-changing endpoint must
# be POST. This is a rule about the routes, not something the guard can
# enforce.
#
# It does not trust every browser extension. An earlier draft allowed any
# chrome-extension:// origin, which hands a local GPU and the user's model
# to every extension the user has ever installed, including one that changes
# hands later. Only the IDs named below are allowed.
#
# CIU_ALLOWED_ORIGINS adds web origins, comma separated, for a local web app
# the user wants to connect on purpose.
# CIU_EXTENSION_ORIGINS replaces the extension allowlist, which is how you
# work on an unpacked extension whose ID changes on every reload.

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

# The CIU Browser extension. Confirmed by header capture: its GET /api/status
# carries no Origin at all, but POST /v1/chat/completions carries this one, so
# without this entry the panel would report "Connected" and then fail every
# task with a 403.
#
# This is the UNPACKED id, which Chrome derives from the folder path. It will
# differ on another machine and will change again when the extension is
# published, unless a "key" is pinned in its manifest. Set
# CIU_EXTENSION_ORIGINS to override, comma separated.
_DEFAULT_EXTENSION_ORIGINS = "chrome-extension://iafbfooieeiknhppomkeikcjcihodnge"


def _env_origins(name: str, default: str = "") -> set[str]:
    return {o.strip().lower().rstrip("/")
            for o in os.environ.get(name, default).split(",")
            if o.strip()}


def _host_name(host_header: str) -> str:
    """The name part of a Host header, without the port.

    IPv6 literals arrive bracketed ("[::1]:8674"), which is why the bracket
    case is handled first: splitting an unbracketed "::1" on the last colon
    would yield "::", and the request would be refused.
    """
    h = host_header.strip().lower()
    if h.startswith("["):
        return h[: h.find("]") + 1]
    if h.count(":") > 1:          # bare IPv6 literal, no port
        return f"[{h}]"
    return h.rsplit(":", 1)[0] if ":" in h else h


def _origin_allowed(origin: str) -> bool:
    origin = origin.strip().lower().rstrip("/")
    own = {f"http://{h}:{PORT}" for h in ("127.0.0.1", "localhost", "[::1]")}
    if origin in own:
        return True
    if origin in _env_origins("CIU_EXTENSION_ORIGINS",
                              _DEFAULT_EXTENSION_ORIGINS):
        return True
    return origin in _env_origins("CIU_ALLOWED_ORIGINS")


@app.middleware("http")
async def guard(request: Request, call_next):
    host = request.headers.get("host", "")
    if _host_name(host) not in _LOOPBACK_HOSTS:
        return JSONResponse(status_code=403, content={"error": {
            "message": "CIU only answers requests addressed to this machine.",
            "type": "ciu_forbidden_host"}})
    origin = request.headers.get("origin")
    if origin is not None and not _origin_allowed(origin):
        return JSONResponse(status_code=403, content={"error": {
            "message": (f"Requests from {origin} are not allowed. Set "
                        f"CIU_ALLOWED_ORIGINS to allow a local web app, or "
                        f"CIU_EXTENSION_ORIGINS for a browser extension."),
            "type": "ciu_forbidden_origin"}})
    return await call_next(request)


_download = {"active": False, "model_id": None, "pct": 0.0, "error": None}

# Engine setup reports the same way a model download does, so the page can
# render both with one code path.
_engine_setup = {"active": False, "pct": 0.0, "error": None, "note": ""}


def model_path(model) -> Path:
    return MODEL_DIR / model.filename


# Measured values are cached, because reading a local GGUF memory-maps the
# whole file and reading a remote one costs a network round trip. Neither
# belongs on a status poll that runs every 1.5 seconds.
#
# Remote reads happen on a background thread and NEVER inside a request. A
# machine with nothing downloaded yet would otherwise fetch metadata for every
# catalogue entry on the first poll, and if HuggingFace is slow or
# unreachable the page hangs instead of loading. That is the state every new
# user starts in, so it is the worst possible place to block.
_shape_cache: dict[str, object] = {}
_size_cache: dict[str, int] = {}
_remote_pending: set[str] = set()
_remote_lock = threading.Lock()


def _fetch_remote(model):
    """Read a model's real shape and size from HuggingFace, off-thread."""
    key = f"{model.repo}/{model.filename}"
    try:
        shape, _ = read_remote(model.repo, model.filename)
    except Exception:
        shape = None
    try:
        size = remote_size(model.repo, model.filename)
    except Exception:
        size = None

    with _remote_lock:
        _shape_cache[key] = shape
        if size:
            _size_cache[key] = size
        _remote_pending.discard(key)


def _want_remote(model):
    """Ask for a remote read if one is not already done or running."""
    key = f"{model.repo}/{model.filename}"
    with _remote_lock:
        if key in _shape_cache or key in _remote_pending:
            return
        _remote_pending.add(key)
    threading.Thread(target=_fetch_remote, args=(model,), daemon=True).start()


def shape_for(model):
    """The model's real shape, preferring measurement over hand-recorded values.

    Local file, then whatever the background read has returned, then the
    catalogue. The catalogue is a fallback rather than the source of truth:
    hand-typed values go stale and are easy to get wrong, and the numbers
    shown before download are what someone uses to decide whether to spend
    the bandwidth. But it is what gets shown until the network answers.
    """
    path = model_path(model)
    key = str(path)

    if key in _shape_cache:
        return _shape_cache[key] or model.shape

    if path.is_file():
        try:
            shape, _ = read_shape(key)
        except Exception:
            shape = None
        _shape_cache[key] = shape
        return shape or model.shape

    remote_key = f"{model.repo}/{model.filename}"
    if remote_key in _shape_cache:
        return _shape_cache[remote_key] or model.shape

    _want_remote(model)
    return model.shape


def size_for(model) -> int:
    """Actual file size, from disk or from a completed background read,
    falling back to the catalogue. A wrong size here is a wrong fit
    decision, but a hung page is worse than a slightly stale number."""
    path = model_path(model)
    if path.is_file():
        return path.stat().st_size

    key = f"{model.repo}/{model.filename}"
    if key in _size_cache:
        return _size_cache[key] or model.size_bytes

    _want_remote(model)
    return model.size_bytes


# ------------------------------------------------------ context and batch

# The micro-batch and the context are not independent. llama.cpp's largest
# batch-scaled allocation is the logits buffer, n_ubatch x n_vocab x 4 bytes,
# which on Qwen's 152k vocabulary is 311 MiB at 512 and 155 MiB at 256.
#
# On a machine with room, 512 is free prefill speed and there is no reason to
# give it up. On a tight one that 155 MiB is the difference between offering
# 4k of context and offering none. So the larger batch is tried first and the
# smaller is a fallback, rather than either being a fixed default.
#
# Measured on an Apple M1 8GB with Qwen3.5-9B-NF4DQ: at 512 even 8k fails to
# load; at 256, 8k loads and 12k does not.
PREFERRED_BATCH = 512
FALLBACK_BATCH = 256


def plan_for(model, available: int, n_ctx: int | None = None):
    """Budget, micro-batch and training context for one model.

    Returns the batch that was chosen alongside the budget computed with it,
    so that what the page shows and what the runner launches can never
    disagree. They did disagree before: the budget assumed 256 while a
    non-tight load passed no batch flags at all and got llama-server's 512.
    """
    shape = shape_for(model)
    weights = size_for(model)
    n_ctx_train = getattr(shape, "n_ctx_train", None) or model.n_ctx_train

    plan = None
    for n_batch in (PREFERRED_BATCH, FALLBACK_BATCH):
        b = budget_for(weights, shape, available, n_batch=n_batch)
        plan = (b, n_batch, n_ctx_train)
        target = n_ctx or max_fitting_context(b, n_ctx_train)
        if target and b.fixed_bytes + b.kv_bytes_at(target) <= available:
            return plan
    # Nothing fits at either batch. Return the smaller one: it is the closest
    # to fitting, and the estimate is advisory anyway.
    return plan


# ----------------------------------------------------------------- CIU API

def _resident_bytes() -> int:
    """Memory the loaded model is holding, weights plus cache plus overhead."""
    if runner.state != "ready" or not runner.config:
        return 0
    model = next((m for m in CATALOG
                  if runner.config.model_path.endswith(m.filename)), None)
    if not model:
        return 0
    shape = shape_for(model)
    # Use the batch the runner was actually started with, not a default.
    n_batch = runner.config.n_batch or PREFERRED_BATCH
    b = budget_for(size_for(model), shape, 0, n_batch=n_batch)
    return b.fixed_bytes + b.kv_bytes_at(runner.config.n_ctx)


@app.get("/api/status")
def status():
    # Runs on every poll, so everything in here must be cheap.
    resident = _resident_bytes()
    hw = detect(resident)

    # free_bytes now excludes whatever the loaded model holds, so this is
    # genuinely what a second model, or a longer context, could use.
    available = hw.free_bytes

    models = []
    for m in CATALOG:
        path = model_path(m)
        downloaded = path.is_file()
        shape = shape_for(m)
        b, n_batch, n_ctx_train = plan_for(m, available)
        opts = context_options(b, n_ctx_train)
        # Default to the largest context that fits entirely. Zero means the
        # model will not load at all on the memory currently free.
        max_ctx = max_fitting_context(b, n_ctx_train)
        default_ctx = max_ctx or (opts[0]["n_ctx"] if opts else 0)

        entry = m.to_dict()
        entry["size_gib"] = round(size_for(m) / GIB, 2)
        entry.update({
            "downloaded": downloaded,
            "budget": b.to_dict(),
            "suggested_ctx": round_context(b.max_context),
            "kv_per_token": b.kv_per_token,
            "contexts": opts,
            "default_ctx": default_ctx,
            "max_ctx": max_ctx,
            "n_batch": n_batch,
            "n_layer": shape.n_layer,
            "n_ctx_train": n_ctx_train,
        })
        models.append(entry)

    return {
        "hardware": hw.to_dict(),
        "resident_gib": round(resident / GIB, 2),
        "models": models,
        "runner": runner.status(),
        "download": _download,
        "llama_server": find_llama_server(),
        "engine": {**bootstrap.describe(), **_engine_setup},
        "base_url": f"http://127.0.0.1:{PORT}/v1",
    }


@app.post("/api/setup-engine")
def setup_engine():
    """Fetch llama-server for this machine.

    Runs on a thread and reports through /api/status, the same shape as a
    model download, because the Windows CUDA archive is over 200 MB and a
    request that blocks for that long is indistinguishable from a hang.
    """
    if _engine_setup["active"]:
        raise HTTPException(409, "engine setup is already running")
    if find_llama_server():
        return {"started": False, "reason": "already installed"}

    def work():
        _engine_setup.update(active=True, pct=0.0, error=None, note="")

        def progress(got, total):
            _engine_setup["pct"] = round(100 * got / total, 1)

        try:
            p = bootstrap.ensure_llama_server(
                progress=progress,
                log=lambda m: _engine_setup.update(note=m))
            _engine_setup.update(pct=100.0, note=f"Engine ready at {p}")
        except Exception as e:
            _engine_setup["error"] = str(e)
        finally:
            _engine_setup["active"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@app.post("/api/download/{model_id}")
def download(model_id: str):
    model = by_id(model_id)
    if not model:
        raise HTTPException(404, "unknown model")
    if _download["active"]:
        raise HTTPException(409, "a download is already running")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def work():
        _download.update(active=True, model_id=model_id, pct=0.0, error=None)
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=model.repo,
                filename=model.filename,
                local_dir=str(MODEL_DIR),
            )
            _download["pct"] = 100.0
        except Exception as e:
            _download["error"] = str(e)
        finally:
            _download["active"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@app.post("/api/load/{model_id}")
def load(model_id: str, n_ctx: int | None = None, mtp: bool = False):
    model = by_id(model_id)
    if not model:
        raise HTTPException(404, "unknown model")

    path = model_path(model)
    if not path.is_file():
        raise HTTPException(400, "model not downloaded")

    hw = detect(_resident_bytes())
    b, n_batch, n_ctx_train = plan_for(model, hw.free_bytes, n_ctx)
    max_ctx = max_fitting_context(b, n_ctx_train)

    # The estimate is advisory, never a veto. It is built from a model of how
    # llama.cpp allocates, and that model is approximate: an earlier version
    # of it refused everything on an 8GB machine that in fact runs the model
    # fine. So CIU says what it expects, then lets the backend be the judge.
    # If it really does not fit, the OOM is caught and reported plainly.
    ctx = n_ctx or max_ctx or CONTEXT_STEPS[0]

    expected = (b.fixed_bytes + b.kv_bytes_at(ctx)) / GIB
    tight = expected > hw.free_gib

    # n_batch is always passed explicitly. Leaving it unset means
    # llama-server picks 512, which would silently contradict whatever the
    # budget above was computed with.
    runner.start(RunConfig(
        model_path=str(path),
        n_ctx=ctx,
        backend=hw.backend,
        use_mtp=mtp and model.has_mtp,
        n_batch=n_batch,
    ))
    return {
        "loading": True,
        "n_ctx": ctx,
        "n_batch": n_batch,
        "expected_gib": round(expected, 2),
        "free_gib": hw.free_gib,
        "tight": tight,
    }


@app.post("/api/unload")
def unload():
    runner.stop()
    return {"stopped": True}


# ------------------------------------------------- OpenAI-compatible proxy

@app.api_route("/v1/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    """Forward to llama-server, preserving streaming.

    Applications hold this URL. Which model answers is CIU's business.
    """
    if runner.state != "ready":
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": f"No model loaded (state: {runner.state}). "
                           f"Open http://127.0.0.1:{PORT} to load one.",
                "type": "ciu_not_ready",
            }},
        )

    url = f"{runner.base_url}/v1/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(request.method, url, content=body,
                               headers=headers,
                               params=request.query_params)
    upstream = await client.send(req, stream=True)

    # A request that overruns the loaded context comes back from llama-server
    # as a 400 with a message about tokens. Applications tend to surface that
    # raw, which tells the user nothing useful, so it is rewritten here into
    # the thing they actually need to do.
    if upstream.status_code == 400 and runner.config:
        raw = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        text = raw.decode("utf-8", "replace").lower()
        if "context" in text or "n_ctx" in text or "exceed" in text:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": (
                        f"This conversation has filled the "
                        f"{runner.config.n_ctx}-token context. Start a new "
                        f"conversation, or reload the model with a larger "
                        f"context if memory allows."),
                    "type": "ciu_context_full",
                    "n_ctx": runner.config.n_ctx,
                }},
            )
        return JSONResponse(status_code=400,
                            content={"error": {"message": raw.decode("utf-8", "replace")}})

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items()
                 if k.lower() not in ("content-length", "transfer-encoding")},
    )


# ------------------------------------------------------------------- page

def _resource(*parts) -> Path:
    """Locate a bundled data file.

    PyInstaller unpacks data to a temporary directory and points sys._MEIPASS
    at it, so __file__ is inside a build tree that does not exist at runtime.
    Reading the page relative to __file__ works when running from source and
    raises FileNotFoundError in the packaged build, which is the sort of thing
    that only shows up after shipping.
    """
    import sys
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    if getattr(sys, "_MEIPASS", None):
        return base.joinpath("ciu", *parts)
    return base.joinpath(*parts)


@app.get("/", response_class=HTMLResponse)
def index():
    page = _resource("static", "index.html")
    # The charset must be explicit. Without it Edge falls back to Windows-1252
    # and every non-ASCII character in the page renders as mojibake.
    return HTMLResponse(page.read_text(encoding="utf-8"),
                        media_type="text/html; charset=utf-8")


def main():
    import uvicorn
    import webbrowser

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{PORT}"

    print("CIU")
    print(f"  page  {url}")
    print(f"  API   {url}/v1")
    engine = find_llama_server()
    if engine:
        print(f"  engine {engine}")
    else:
        print("  engine not installed yet, the page will offer to fetch it")
    print()
    print("Leave this window open. Closing it stops CIU.")

    # Someone who double-clicked an application expects a window, not a
    # terminal. The delay is because the browser must not arrive before
    # uvicorn is accepting connections.
    if os.environ.get("CIU_NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
