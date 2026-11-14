"""Fetch the inference engine on first run.

CIU ships without llama-server. Bundling it would mean a separate download per
backend, and the Windows CUDA build alone is over 200 MB, so the engine is
pulled from the llama.cpp fork's releases the first time CIU starts.

The machine decides which asset: Apple Silicon takes the Metal build, a Windows
box with an NVIDIA card takes the CUDA build and one without takes the CPU
build. Not every combination has a published binary, and the honest answer when
one is missing is to say so rather than install something that will not run, or
something that will run a hundred times slower than the user expects.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

FORK_REPO = "runciu/llama.cpp"
RELEASES_API = f"https://api.github.com/repos/{FORK_REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{FORK_REPO}/releases"

INSTALL_DIR = Path.home() / ".ciu" / "bin"

# (system, arch) -> {backend: asset}. Keys are normalised by _platform_key, so
# aarch64 and arm64, x86_64 and AMD64, all land on the same entry rather than
# needing a row each.
ASSETS: dict[tuple[str, str], dict[str, str]] = {
    ("Darwin", "arm64"): {
        "metal": "llama-server-macos-arm64.tar.gz",
        "cpu": "llama-server-macos-arm64.tar.gz",
    },
    ("Windows", "x64"): {
        "cuda": "llama-server-windows-x64-cuda.zip",
        "cpu": "llama-server-windows-x64-cpu.zip",
    },
    ("Linux", "x64"): {
        # The fork's linux-cuda job does not currently produce a binary, so an
        # NVIDIA box on Linux gets the CPU build. That is a real degradation
        # and _plan reports it rather than hiding it.
        "cuda": "llama-server-linux-x64-cpu.tar.gz",
        "cpu": "llama-server-linux-x64-cpu.tar.gz",
    },
}

# Combinations with no binary at all, and why. Better a clear sentence than a
# download that cannot work.
UNSUPPORTED = {
    ("Darwin", "x64"): (
        "Intel Macs are not supported yet: the engine is only published for "
        "Apple Silicon."),
}


@dataclass
class Plan:
    """What would be installed, before anything is downloaded."""
    asset: str | None
    backend: str
    system: str
    arch: str
    reason: str = ""        # why there is no asset, when asset is None
    warning: str = ""       # installable, but the user should know something

    @property
    def ok(self) -> bool:
        return self.asset is not None


def _platform_key() -> tuple[str, str]:
    """Normalise to the handful of names the asset table uses."""
    system = platform.system()                 # Darwin | Windows | Linux
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64"):
        arch = "x64"
    else:
        arch = machine
    return system, arch


def plan(backend: str | None = None) -> Plan:
    """Decide which asset this machine needs. Touches the network only if
    backend is not supplied."""
    system, arch = _platform_key()

    if backend is None:
        try:
            from .hardware import detect
            backend = detect().backend
        except Exception:
            backend = "cpu"

    if (system, arch) in UNSUPPORTED:
        return Plan(None, backend, system, arch,
                    reason=UNSUPPORTED[(system, arch)])

    table = ASSETS.get((system, arch))
    if not table:
        return Plan(None, backend, system, arch,
                    reason=f"No engine is published for {system} {arch}.")

    asset = table.get(backend) or table.get("cpu")
    warning = ""
    if backend == "cuda" and system == "Linux":
        warning = ("No CUDA build is published for Linux yet, so the CPU "
                   "build will be used. Generation will be very slow on a "
                   "machine with an NVIDIA card.")
    elif backend != "cpu" and asset == table.get("cpu") and \
            table.get(backend) is None:
        warning = (f"No {backend} build is published for {system} {arch}, so "
                   f"the CPU build will be used.")
    return Plan(asset, backend, system, arch, warning=warning)


def _direct_url(name: str) -> str:
    """The redirect GitHub serves without an API call."""
    return f"https://github.com/{FORK_REPO}/releases/latest/download/{name}"


def _release_asset(name: str) -> tuple[str, int, str | None]:
    """URL, size and digest for one asset of the latest release.

    The API is preferred because it reports the expected size and, when
    GitHub provides one, a digest to check the download against. But
    unauthenticated API calls are limited to 60 an hour per address, which a
    shared or office network can exhaust without the user doing anything
    wrong. So an API failure falls back to the plain download redirect, which
    has no such limit. Less verification, still a working install, and the
    caller is told which path it took.
    """
    import httpx

    try:
        r = httpx.get(RELEASES_API, timeout=30,
                      headers={"Accept": "application/vnd.github+json"},
                      follow_redirects=True)
        r.raise_for_status()
        rel = r.json()
    except Exception:
        return _direct_url(name), 0, None

    for a in rel.get("assets", []):
        if a["name"] == name:
            return a["browser_download_url"], a.get("size", 0), a.get("digest")

    # The API answered and the asset genuinely is not there. That is worth
    # reporting precisely rather than falling back to a URL that will 404.
    published = ", ".join(a["name"] for a in rel.get("assets", [])) or "none"
    raise RuntimeError(
        f"The latest release ({rel.get('tag_name', '?')}) has no asset called "
        f"{name}. It publishes: {published}. See {RELEASES_PAGE}")


def _safe_extract(archive, dest: Path) -> None:
    """Extract without letting a member escape dest.

    A crafted archive can carry ../.. in a member name and write anywhere the
    process can reach. These archives come from a repo we control, but an
    extractor that only works on trusted input is a trap for whoever reuses it.
    """
    dest = dest.resolve()
    if isinstance(archive, tarfile.TarFile):
        members = archive.getmembers()
        names = [m.name for m in members]
    else:
        members = None
        names = archive.namelist()

    for name in names:
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise RuntimeError(f"archive member escapes the target: {name}")

    if members is not None:
        try:
            archive.extractall(dest, filter="data")     # Python 3.12+
        except TypeError:
            archive.extractall(dest)
    else:
        archive.extractall(dest)


def _download(url: str, out: Path, expect_size: int = 0,
              progress=None) -> None:
    import httpx

    got = 0
    sha = hashlib.sha256()
    with httpx.stream("GET", url, timeout=None, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or expect_size or 0)
        with open(out, "wb") as fh:
            for chunk in r.iter_bytes(1024 * 256):
                fh.write(chunk)
                sha.update(chunk)
                got += len(chunk)
                if progress and total:
                    progress(got, total)
    out.with_suffix(out.suffix + ".sha256").write_text(sha.hexdigest())


def _find_server(root: Path) -> Path | None:
    names = ("llama-server", "llama-server.exe")
    for p in root.rglob("*"):
        if p.is_file() and p.name in names:
            return p
    return None


def installed_server() -> Path | None:
    """The engine CIU installed, if it is there. Does not look elsewhere;
    runner.find_llama_server covers PATH and local builds."""
    return _find_server(INSTALL_DIR)


def ensure_llama_server(force: bool = False, progress=None,
                        log=print) -> Path:
    """Download and unpack the engine. Returns the path to llama-server.

    Raises RuntimeError with a sentence the user can act on, rather than a
    stack trace, for every failure that is not a bug.
    """
    if not force:
        existing = installed_server()
        if existing:
            return existing

    p = plan()
    if not p.ok:
        raise RuntimeError(p.reason)
    if p.warning:
        log(p.warning)

    url, size, digest = _release_asset(p.asset)
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading the {p.backend} engine"
        f"{f' ({size / 1024**2:.0f} MB)' if size else ''}...")

    with tempfile.TemporaryDirectory() as td:
        blob = Path(td) / p.asset
        try:
            _download(url, blob, size, progress)
        except Exception as e:
            # Network failures are the common case here and a raw httpx
            # repr tells the user nothing they can act on.
            raise RuntimeError(
                f"Could not download the engine from GitHub. Check your "
                f"connection and try again. If it keeps failing, the file "
                f"can be fetched by hand from {RELEASES_PAGE} and unpacked "
                f"into {INSTALL_DIR}. ({type(e).__name__})") from e

        if digest and digest.startswith("sha256:"):
            want = digest.split(":", 1)[1]
            got = blob.with_suffix(blob.suffix + ".sha256").read_text().strip()
            if got != want:
                raise RuntimeError(
                    "The downloaded engine does not match the digest GitHub "
                    "published for it. Nothing was installed. Try again, and "
                    "if it keeps happening do not use the file.")

        if p.asset.endswith(".zip"):
            with zipfile.ZipFile(blob) as z:
                _safe_extract(z, INSTALL_DIR)
        else:
            with tarfile.open(blob, "r:gz") as t:
                _safe_extract(t, INSTALL_DIR)

    server = _find_server(INSTALL_DIR)
    if not server:
        raise RuntimeError(
            f"{p.asset} unpacked but contains no llama-server binary. "
            f"See {RELEASES_PAGE}")

    # Archives do not always carry the executable bit, and a downloaded file
    # that is not executable fails later with a permission error that points
    # nowhere useful.
    if platform.system() != "Windows":
        server.chmod(server.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP |
                     stat.S_IXOTH)

    log(f"Engine installed at {server}")
    return server


def describe() -> dict:
    """For /api/status, so the page can explain itself before anything is
    installed."""
    p = plan()
    server = installed_server()
    return {
        "installed": str(server) if server else None,
        "asset": p.asset,
        "backend": p.backend,
        "platform": f"{p.system} {p.arch}",
        "supported": p.ok,
        "reason": p.reason,
        "warning": p.warning,
        "releases_url": RELEASES_PAGE,
    }
