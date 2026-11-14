#!/usr/bin/env python3
"""CIU entry point.

`ciu --selftest` checks that the things a packaged build silently loses are
present: data files that nothing imports, and modules imported lazily inside
functions. Both build cleanly and fail later, in front of a user, so the
release workflow runs this against the binary before publishing it.
"""
import sys


def selftest() -> int:
    ok = True

    def check(label, fn):
        nonlocal ok
        try:
            detail = fn()
            print(f"  ok    {label}{f'  {detail}' if detail else ''}")
        except Exception as e:
            ok = False
            print(f"  FAIL  {label}: {type(e).__name__}: {e}")

    print("CIU self-test")

    # Read at request time, so nothing in the import graph points at it.
    def page():
        from ciu.server import _resource
        p = _resource("static", "index.html")
        n = len(p.read_text(encoding="utf-8"))
        if n < 1000:
            raise RuntimeError(f"index.html is only {n} bytes")
        return f"index.html {n} bytes"

    # Imported inside functions to keep startup quick, so PyInstaller cannot
    # see them from the call sites.
    def hub():
        from huggingface_hub import hf_hub_download  # noqa: F401
        return "huggingface_hub"

    def http():
        import httpx  # noqa: F401
        return "httpx"

    def protocols():
        import uvicorn.protocols.http.h11_impl  # noqa: F401
        import uvicorn.lifespan.on  # noqa: F401
        import uvicorn.loops.auto  # noqa: F401
        return "uvicorn protocol and lifespan implementations"

    def modules():
        from ciu import bootstrap, budget, catalog, hardware, runner  # noqa
        return f"{len(catalog.CATALOG)} models in the catalogue"

    def engine_plan():
        from ciu.bootstrap import plan
        p = plan()
        return f"{p.system} {p.arch} -> {p.asset or 'no asset: ' + p.reason}"

    check("bundled page", page)
    check("huggingface_hub", hub)
    check("httpx", http)
    check("uvicorn internals", protocols)
    check("ciu modules", modules)
    check("engine plan", engine_plan)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    from ciu.server import main as serve
    serve()


if __name__ == "__main__":
    main()
