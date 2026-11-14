Download the zip for your machine and unzip it.

On macOS, drag CIU.app to Applications. On Windows, run `ciu.exe`.

| Platform | File |
|---|---|
| macOS, Apple Silicon | `ciu-macos-arm64.zip` |
| Windows 10/11, 64-bit | `ciu-windows-x64.zip` |

Intel Macs and Linux are not supported yet.

**These builds are not code-signed,** so the first run needs one extra step.
FIRST-RUN.txt in the zip has it, and it is two lines on either platform.

CIU does not include the inference engine. On first run the page offers to
fetch the right llama-server build for your machine, around 20 MB on macOS
and up to 200 MB for Windows with CUDA.
