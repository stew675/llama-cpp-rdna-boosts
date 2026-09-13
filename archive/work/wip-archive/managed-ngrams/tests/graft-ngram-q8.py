#!/usr/bin/env python3
"""
graft-ngram-q8.py -- swap the ngram table of a merged GGUF for a Q8_0 copy

Replaces the `per_layer_token_embd.weight` tensor of IN_GGUF (typically the
merged Unsloth UD-Q4_K_XL build, whose ngram is IQ4_NL) with the Q8_0 ngram
table taken from NGRAM_GGUF (any of the Unsloth Q8_0/Q5_K_XL/Q6_K_XL builds'
*-00003-of-00006.gguf file; those three files are byte-identical).

All other tensors are byte-copied, so the model weights stay bit-exact.

Usage:
    python3 graft-ngram-q8.py IN_GGUF NGRAM_GGUF OUT_GGUF
"""

import struct
import sys

GGML_TYPE_Q8_0 = 8
GGML_TYPE_IQ4_NL = 20

# bytes per element, matching the current llama.cpp ggml definitions
SIZE = {
    0: 4.0,       # f32
    1: 2.0,       # f16
    2: 18/32,     # q4_0
    3: 20/32,     # q4_1
    6: 22/32,     # q5_0
    7: 24/32,     # q5_1
    8: 34/32,     # q8_0
    9: 36/32,     # q8_1
    10: 84/256,   # q2_k
    11: 110/256,  # q3_k
    12: 144/256,  # q4_k
    13: 176/256,  # q5_k
    14: 210/256,  # q6_k
    15: 292/256,  # q8_k
    16: 66/256,   # iq2_xxs
    17: 74/256,   # iq2_xs
    18: 98/256,   # iq3_xxs
    19: 50/256,   # iq1_s
    20: 18/32,    # iq4_nl
    21: 110/256,  # iq3_s
    22: 82/256,   # iq2_s
    23: 136/256,  # iq4_xs
    24: 1.0,      # i8
    25: 2.0,      # i16
    26: 4.0,      # i32
    27: 8.0,      # i64
    28: 8.0,      # f64
    29: 56/256,   # iq1_m
    30: 2.0,      # bf16
}

ALIGN = 32
TENSOR_NAME = b"per_layer_token_embd.weight"


def pad32(n):
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def read_str(buf, off):
    n = struct.unpack_from("<Q", buf, off)[0]
    off += 8
    return buf[off:off + n], off + n


def skip_val(buf, off, t):
    """Skip one GGUF value of type t; return the offset just past it."""
    if t == 8:  # string
        _, off = read_str(buf, off)
    elif t == 9:  # array
        et = struct.unpack_from("<I", buf, off)[0]
        off += 4
        n = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        for _ in range(n):
            off = skip_val(buf, off, et)
    else:
        off += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t]
    return off


def parse_header(path):
    """Return (kv_end, tensors, data_start).  Tensors are dicts with raw names."""
    f = open(path, "rb")
    buf = f.read(64 << 20)  # the header always fits in 64 MiB
    f.close()
    assert buf[:4] == b"GGUF", "not a GGUF file: %s" % path
    n_tensors = struct.unpack_from("<Q", buf, 8)[0]
    n_kv = struct.unpack_from("<Q", buf, 16)[0]
    off = 24
    for _ in range(n_kv):
        _, off = read_str(buf, off)
        t = struct.unpack_from("<I", buf, off)[0]
        off += 4
        off = skip_val(buf, off, t)
    kv_end = off
    tensors = []
    for _ in range(n_tensors):
        name, off = read_str(buf, off)
        nd = struct.unpack_from("<I", buf, off)[0]
        off += 4
        dims = struct.unpack_from("<%dQ" % nd, buf, off)
        off += 8 * nd
        typ = struct.unpack_from("<I", buf, off)[0]
        off += 4
        toff = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        tensors.append({"name": name, "dims": dims, "type": typ, "offset": toff})
    header_end = off
    data_start = pad32(header_end)
    for t in tensors:
        if t["type"] not in SIZE:
            raise ValueError("unsupported tensor type %d for %s" % (t["type"], t["name"]))
    offs = [t["offset"] for t in tensors]
    if offs != sorted(offs):
        raise ValueError("tensors are not sorted by offset")
    return kv_end, tensors, data_start


def nbytes(t):
    nelems = 1
    for d in t["dims"]:
        nelems *= d
    return int(round(nelems * SIZE[t["type"]]))


def copy_range(src, src_abs, dst, n, chunk=64 << 20):
    src.seek(src_abs)
    remaining = n
    while remaining > 0:
        take = min(chunk, remaining)
        data = src.read(take)
        if len(data) != take:
            raise IOError("unexpected EOF while copying tensor data")
        dst.write(data)
        remaining -= take


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    in_path, ngram_path, out_path = sys.argv[1:]

    kv_end, tensors, data_start = parse_header(in_path)
    _, ngram_tensors, data_start2 = parse_header(ngram_path)
    in_file = open(in_path, "rb")
    ngram_file = open(ngram_path, "rb")

    src = [t for t in ngram_tensors if t["name"] == TENSOR_NAME]
    assert len(src) == 1, "ngram file must contain exactly one per_layer_token_embd.weight"
    src = src[0]
    assert src["type"] == GGML_TYPE_Q8_0, "ngram source is not Q8_0"

    dst = [t for t in tensors if t["name"] == TENSOR_NAME]
    assert len(dst) == 1, "input has no per_layer_token_embd.weight"
    dst = dst[0]
    assert dst["dims"] == src["dims"], "ngram shape mismatch: %r vs %r" % (dst["dims"], src["dims"])
    print("replacing per_layer_token_embd.weight: type %d (%d bytes) -> q8_0 (%d bytes)"
          % (dst["type"], nbytes(dst), nbytes(src)))

    # --- write the output header (raw prefix + re-emitted tensor infos) ---
    out_file = open(out_path, "wb")
    in_file.seek(0)
    out_file.write(in_file.read(kv_end))  # magic + version + counts + KV section, verbatim

    new_infos = []
    for t in tensors:
        if t["name"] == TENSOR_NAME:
            t = dict(t)
            t["type"] = GGML_TYPE_Q8_0
        new_infos.append(t)

    data_off = 0
    for t in new_infos:
        out_file.write(struct.pack("<Q", len(t["name"])))
        out_file.write(t["name"])
        out_file.write(struct.pack("<I", len(t["dims"])))
        out_file.write(struct.pack("<%dQ" % len(t["dims"]), *t["dims"]))
        out_file.write(struct.pack("<I", t["type"]))
        out_file.write(struct.pack("<Q", data_off))
        data_off += nbytes(t)

    data_start_out = pad32(out_file.tell())
    out_file.write(b"\0" * (data_start_out - out_file.tell()))
    assert out_file.tell() == data_start_out

    # --- copy tensor data ---
    for t in new_infos:
        if t["name"] == TENSOR_NAME:
            copy_range(ngram_file, data_start2 + src["offset"], out_file, nbytes(t))
        else:
            copy_range(in_file, data_start + t["offset"], out_file, nbytes(t))

    out_file.close()
    print("wrote %s: %d tensors, %d bytes of data" % (out_path, len(new_infos), data_off))


if __name__ == "__main__":
    main()
