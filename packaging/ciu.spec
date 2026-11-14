# PyInstaller spec for CIU.
#
# Run from the repo root:  pyinstaller packaging/ciu.spec
#
# Two things here are not optional and both fail only at runtime, after the
# build has reported success:
#
#   datas         index.html is read at request time, not imported, so nothing
#                 in the dependency graph tells PyInstaller it exists. Without
#                 this line the binary builds and then 500s on the first page
#                 load. server._resource knows to look under sys._MEIPASS/ciu.
#
#   hiddenimports uvicorn loads its protocol and lifespan implementations by
#                 string name, and httpx picks a transport the same way, so
#                 static analysis cannot see them. Missing ones surface as an
#                 ImportError the moment a request arrives rather than at
#                 startup.

import os

block_cipher = None

ROOT = os.path.abspath(os.getcwd())

hiddenimports = [
    # uvicorn resolves these from strings in its config
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # CIU pulls these in lazily, inside functions, to keep startup quick
    "httpx",
    "huggingface_hub",
    "ciu.bootstrap",
    "ciu.catalog",
    "ciu.gguf_read",
    "ciu.gguf_remote",
    "ciu.hardware",
]

a = Analysis(
    ["../run.py"],
    pathex=[ROOT],
    binaries=[],
    datas=[("../ciu/static", "ciu/static")],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Nothing here draws a chart or opens a window, and leaving them in adds
    # tens of megabytes to a download whose whole point is being small.
    excludes=["tkinter", "matplotlib", "numpy.distutils", "pytest",
              "PIL", "scipy", "pandas"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ciu",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-packed binaries trip antivirus heuristics, and
                        # an unsigned download has enough of that already
    runtime_tmpdir=None,
    console=True,       # the window is the status display and the stop button
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
