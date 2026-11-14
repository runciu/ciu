"""Start, watch and stop llama-server.

CIU is not an inference engine. llama-server already speaks the OpenAI API,
handles streaming and manages parallel slots. What it does not do is decide
which model to load, work out what fits, or stay at one address while the
model behind it changes. That is what this supervises.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Backends that can hold layers in device memory. CPU never can.
GPU_BACKENDS = ("cuda", "metal")

# A backend that runs out of memory does not necessarily kill the process.
# On Metal the command buffer fails, the backend latches into an error state,
# and llama-server still prints "listening" and answers /health with 200.
# Measured: every failing load in testing reported a healthy server. So the
# log is the only reliable signal and it has to override the health check.
OOM_MARKERS = (
    "kIOGPUCommandBufferCallbackErrorOutOfMemory",   # Metal
    "Insufficient Memory",                            # Metal
    "CUDA error: out of memory",                      # CUDA
    "cudaMalloc failed",                              # CUDA
    "failed to allocate",                             # generic ggml
)


def _is_oom(line: str) -> bool:
    return any(m in line for m in OOM_MARKERS)


def find_llama_server() -> str | None:
    """Locate llama-server: PATH, env override, or the usual build locations."""
    env = os.environ.get("CIU_LLAMA_SERVER")
    if env and Path(env).is_file():
        return env

    found = shutil.which("llama-server")
    if found:
        return found

    # .exe matters: on Windows shutil.which finds it through PATHEXT, but a
    # direct path check does not, so the engine CIU installs into ~/.ciu/bin
    # was invisible to this function on the one platform that needs the
    # suffix.
    exe = ".exe" if os.name == "nt" else ""
    for candidate in (
        Path.home() / f"llama.cpp/build/bin/llama-server{exe}",
        Path.home() / f".ciu/bin/llama-server{exe}",
        Path(f"/usr/local/bin/llama-server{exe}"),
    ):
        if candidate.is_file():
            return str(candidate)

    # The Windows CUDA archive unpacks into a subdirectory alongside its DLLs
    # rather than dropping the binary at the top level.
    bindir = Path.home() / ".ciu" / "bin"
    if bindir.is_dir():
        for found in bindir.rglob(f"llama-server{exe}"):
            if found.is_file():
                return str(found)
    return None


def _library_env(binary: str) -> dict:
    """Environment that lets llama-server find the libraries shipped with it.

    The published builds put their shared libraries in the same directory as
    the executable, but the loader does not search that directory on its own.
    Worse, the Linux release has a RUNPATH baked to the CI machine's build
    tree:

        Library runpath: [/home/runner/work/llama.cpp/llama.cpp/build/bin:]

    which exists nowhere outside the GitHub runner. The binary then fails with
    "error while loading shared libraries", and because llama-server dies
    before printing anything CIU recognises, it surfaces as a bare "exited
    during load".

    Pointing the platform's library search variable at the binary's own
    directory fixes it regardless of what the publisher baked in, which is
    better than depending on a particular RPATH being correct. Windows needs
    no help, since it always searches the executable's own directory, but
    adding the entry is harmless and keeps one code path.
    """
    env = os.environ.copy()
    bindir = str(Path(binary).resolve().parent)
    if sys.platform == "darwin":
        key = "DYLD_LIBRARY_PATH"
    elif os.name == "nt":
        key = "PATH"
    else:
        key = "LD_LIBRARY_PATH"
    existing = env.get(key, "")
    env[key] = bindir + (os.pathsep + existing if existing else "")
    return env


def free_port(start: int = 8700) -> int:
    for port in range(start, start + 200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port in range")


@dataclass
class RunConfig:
    model_path: str
    n_ctx: int
    backend: str
    use_mtp: bool = False
    draft_n_max: int = 4
    flash_attn: bool = True
    # 99 offloads everything. A smaller number splits the model, keeping the
    # first N layers on the GPU and running the rest on CPU.
    n_gpu_layers: int = 99
    # Defaults to 256, not None. None meant the flags were never passed and
    # llama-server's own default of 512 applied, which costs an extra 155 MiB
    # of logits buffer and was the difference between 2k and 8k on an M1.
    n_batch: int | None = 256


class LlamaRunner:
    """One llama-server process, with its lifecycle and log."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.config: RunConfig | None = None
        self.log: list[str] = []
        self.state = "stopped"      # stopped | loading | ready | error
        self.error: str | None = None
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}" if self.port else None

    def _build_args(self, binary: str, cfg: RunConfig, port: int) -> list[str]:
        args = [
            binary,
            "-m", cfg.model_path,
            "-c", str(cfg.n_ctx),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--jinja",
        ]
        ngl = cfg.n_gpu_layers if cfg.backend in GPU_BACKENDS else 0
        args += ["-ngl", str(ngl)]

        if cfg.flash_attn and ngl > 0:
            args += ["-fa", "on"]

        # One sequence slot. llama-server defaults to 4, and -c is PER SLOT,
        # so the default quadruples the KV cache to serve parallelism CIU
        # never uses: four slots at 8k hold 32k tokens. On Qwen3.5-9B that is
        # 1.07 GiB instead of 0.27. Largest single saving available, and it
        # costs nothing in quality or speed.
        args += ["-np", "1"]

        # The compute buffer is sized by the micro-batch, not by the context,
        # and its largest component is the logits buffer at
        # n_ubatch x n_vocab x 4 bytes. On Qwen's 152k vocab that is 311 MiB
        # at 512 and 155 MiB at 256. Smaller batches slow prefill but leave
        # decode and quality untouched.
        if cfg.n_batch:
            args += ["-b", str(cfg.n_batch), "-ub", str(cfg.n_batch)]
        if cfg.use_mtp:
            # Measured 1.79x on an L4, 1.54x on an A100. Depth 4 was the peak
            # of a sweep; 8 collapsed badly, so it is not exposed as a dial.
            args += ["--spec-type", "draft-mtp",
                     "--spec-draft-n-max", str(cfg.draft_n_max)]
        return args

    def start(self, cfg: RunConfig, timeout: int = 900) -> None:
        with self._lock:
            if self.state in ("loading", "ready"):
                self.stop()

            binary = find_llama_server()
            if not binary:
                self.state = "error"
                self.error = (
                    "llama-server not found. Build the NF4DQ fork, or set "
                    "CIU_LLAMA_SERVER to its path."
                )
                return

            self.port = free_port()
            self.config = cfg
            self.log = []
            self.error = None
            self.state = "loading"

            args = self._build_args(binary, cfg, self.port)
            self.proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=str(Path(binary).parent),
                env=_library_env(binary),
            )

        threading.Thread(target=self._pump_log, daemon=True).start()
        threading.Thread(target=self._await_ready, args=(timeout,),
                         daemon=True).start()

    def _pump_log(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            self.log.append(line.rstrip())
            # Loading a 13GiB model produces a lot of output; keep the tail.
            if len(self.log) > 400:
                del self.log[:200]

    def _saw_oom(self) -> bool:
        # Snapshot: _pump_log appends from another thread.
        return any(_is_oom(line) for line in list(self.log))

    def _await_ready(self, timeout: int) -> None:
        """Poll /health. The port opens long before the weights are loaded, so
        watching the socket alone reports ready far too early. A passing
        /health is also not sufficient: a backend that has run out of memory
        still answers 200, so the log is checked first on every pass."""
        import httpx

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                self.state = "error"
                self.error = "llama-server exited during load"
                return

            if self._saw_oom():
                self.state = "error"
                self.error = (
                    "Out of memory while loading. Try a shorter context, or "
                    "close other applications."
                )
                self.stop()
                return

            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    # One last look: the OOM can land between the check above
                    # and the response, and a server reported ready here will
                    # fail on the user's first message instead.
                    if self._saw_oom():
                        self.state = "error"
                        self.error = (
                            "Out of memory while loading. Try a shorter "
                            "context, or close other applications."
                        )
                        self.stop()
                        return
                    self.state = "ready"
                    return
            except Exception:
                pass
            time.sleep(1)

        self.state = "error"
        self.error = f"model did not load within {timeout}s"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.port = None
        if self.state != "error":
            self.state = "stopped"

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "base_url": self.base_url,
            "model_path": self.config.model_path if self.config else None,
            "n_ctx": self.config.n_ctx if self.config else None,
            "mtp": self.config.use_mtp if self.config else False,
            "n_gpu_layers": self.config.n_gpu_layers if self.config else None,
            "log_tail": self.log[-25:],
        }
