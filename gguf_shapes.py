"""Check whether a model can take NF4DQ, without downloading it.

NF4DQ has one hard constraint: QK_NF4DQ is 1024, and ggml requires the block
size to divide the row length, so every quantisable tensor's first dimension
must be a multiple of 1024. A model that fails this cannot be quantised at
all, and finding out costs a 66 GiB download if you do it the obvious way.

GGUF stores tensor names, shapes and types in the header, immediately after
the key-value block and before any tensor data. So a range request for the
first few megabytes answers the question.

Usage:
    python gguf_shapes.py ornith-ai/Ornith-1.5-35B-A3B-GGUF Ornith-1.5-35B-BF16.gguf
"""

from __future__ import annotations

import struct
import sys

import httpx

MAGIC = b"GGUF"
CHUNK = 4 << 20          # 4 MiB is usually enough; vocab dominates the header
MAX_READ = 64 << 20

(T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL,
 T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64) = range(13)

_FIXED = {T_UINT8: ("<B", 1), T_INT8: ("<b", 1), T_UINT16: ("<H", 2),
          T_INT16: ("<h", 2), T_UINT32: ("<I", 4), T_INT32: ("<i", 4),
          T_FLOAT32: ("<f", 4), T_BOOL: ("<?", 1), T_UINT64: ("<Q", 8),
          T_INT64: ("<q", 8), T_FLOAT64: ("<d", 8)}


class _Cursor:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def take(self, n):
        if self.pos + n > len(self.buf):
            raise EOFError
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def scalar(self, tag):
        if tag in _FIXED:
            fmt, size = _FIXED[tag]
            return struct.unpack(fmt, self.take(size))[0]
        if tag == T_STRING:
            return self.string()
        raise ValueError(f"unknown value type {tag}")

    def value(self):
        tag = self.u32()
        if tag != T_ARRAY:
            return self.scalar(tag)
        elem, count = self.u32(), self.u64()
        # Token lists run to hundreds of thousands of entries. Skip rather
        # than materialise; nothing here needs them.
        if elem in _FIXED:
            self.take(_FIXED[elem][1] * count)
        elif elem == T_STRING:
            for _ in range(count):
                self.take(self.u64())
        else:
            raise ValueError(f"unknown array element type {elem}")
        return f"<{count} items>"


def parse_header(buf):
    """Return (kv, tensors). Raises EOFError if buf stops short."""
    c = _Cursor(buf)
    if c.take(4) != MAGIC:
        raise ValueError("not a GGUF file")
    version = c.u32()
    n_tensors, n_kv = c.u64(), c.u64()

    kv = {}
    for _ in range(n_kv):
        key = c.string()
        kv[key] = c.value()

    tensors = []
    for _ in range(n_tensors):
        name = c.string()
        n_dims = c.u32()
        dims = [c.u64() for _ in range(n_dims)]
        dtype = c.u32()
        c.u64()                      # offset, not needed here
        tensors.append({"name": name, "shape": dims, "type": dtype})

    kv["_version"] = version
    return kv, tensors


def fetch(repo, filename, revision="main"):
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    size = CHUNK
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        while size <= MAX_READ:
            r = client.get(url, headers={"Range": f"bytes=0-{size - 1}"})
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            try:
                kv, tensors = parse_header(r.content)
                return kv, tensors, len(r.content)
            except EOFError:
                if r.status_code == 200:
                    raise RuntimeError("header truncated")
                size *= 2
    raise RuntimeError("header larger than the read limit")


# ggml refuses to quantise 1-D tensors, and llama.cpp leaves some 2-D ones
# alone by convention. These are the ones that actually get a block type.
def quantisable(t):
    return len(t["shape"]) > 1


def report(repo, filename, qk=1024):
    kv, tensors, nbytes = fetch(repo, filename)
    arch = kv.get("general.architecture", "?")

    print(f"{repo}/{filename}")
    print(f"read {nbytes / 1024**2:.1f} MiB of header, {len(tensors)} tensors\n")
    print(f"architecture      {arch}")
    for k in ("block_count", "embedding_length", "feed_forward_length",
              "attention.head_count", "attention.head_count_kv",
              "expert_count", "expert_used_count",
              "expert_feed_forward_length", "context_length"):
        v = kv.get(f"{arch}.{k}")
        if v is not None:
            print(f"{k:26s} {v}")

    bad = [t for t in tensors if quantisable(t) and t["shape"][0] % qk]
    print(f"\ndivisibility by {qk}: "
          f"{len(bad)} of {sum(1 for t in tensors if quantisable(t))} "
          f"quantisable tensors fail")

    if bad:
        # Group by shape: hundreds of layers repeat the same few shapes, and
        # the distinct shapes are what decides whether a smaller block size
        # would fix it.
        from collections import Counter
        counts = Counter(tuple(t["shape"]) for t in bad)
        print("\nfailing shapes, most common first:")
        for shape, n in counts.most_common(12):
            print(f"  {str(list(shape)):28s} x{n:<4}  "
                  f"{shape[0]} % {qk} = {shape[0] % qk}")
        print("\nexample names:")
        for t in bad[:6]:
            print(f"  {t['name']}")

        for smaller in (512, 256, 128):
            still = [t for t in tensors
                     if quantisable(t) and t["shape"][0] % smaller]
            print(f"\nat QK={smaller}: {len(still)} would still fail")
            if not still:
                print(f"  -> QK_NF4DQ={smaller} would work for this model")
                break
    else:
        print("  -> every quantisable tensor divides; NF4DQ can take it")

    # Routers are the MoE risk: a routing error flips expert selection rather
    # than degrading gently, so these are the tensors to exclude by hand.
    routers = [t for t in tensors
               if any(k in t["name"] for k in ("ffn_gate_inp", "router",
                                               "gate_inp", "exp_probs"))]
    if routers:
        print(f"\n{len(routers)} router tensors found. These should NOT be "
              f"quantised:")
        from collections import Counter
        for shape, n in Counter(tuple(t["shape"]) for t in routers).most_common(5):
            print(f"  {str(list(shape)):28s} x{n}")
        print(f"  example: {routers[0]['name']}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    report(sys.argv[1], sys.argv[2])
