#!/usr/bin/env python3
"""Minimal GGUF header parser: dumps KV keys and tensors (name, shape, type, size)."""
import struct, sys

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_S", 5: "Q5_0", 6: "Q5_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}
VAL_TYPES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool",
             8: "str", 9: "array", 10: "u64", 11: "i64", 12: "f64"}

def read_str(buf, off):
    n = struct.unpack_from("<Q", buf, off)[0]; off += 8
    s = buf[off:off+n].decode("utf-8", "replace"); return s, off + n

def read_val(buf, off, t):
    if t == 6: return struct.unpack_from("<f", buf, off)[0], off+4
    if t == 4: return struct.unpack_from("<I", buf, off)[0], off+4
    if t == 5: return struct.unpack_from("<i", buf, off)[0], off+4
    if t == 10: return struct.unpack_from("<Q", buf, off)[0], off+8
    if t == 11: return struct.unpack_from("<q", buf, off)[0], off+8
    if t == 7: return bool(buf[off]), off+1
    if t == 0: return buf[off], off+1
    if t == 1: return struct.unpack_from("<b", buf, off)[0], off+1
    if t == 2: return struct.unpack_from("<H", buf, off)[0], off+2
    if t == 3: return struct.unpack_from("<h", buf, off)[0], off+2
    if t == 12: return struct.unpack_from("<d", buf, off)[0], off+8
    if t == 8:
        s, off = read_str(buf, off); return s, off
    if t == 9:
        et = struct.unpack_from("<I", buf, off)[0]; off += 4
        n = struct.unpack_from("<Q", buf, off)[0]; off += 8
        items = []
        for _ in range(n):
            v, off = read_val(buf, off, et); items.append(v)
        return items, off
    raise ValueError(f"unknown val type {t}")

def parse(path, show_kv=False, only_tensors=None):
    data = open(path, "rb").read()
    assert data[:4] == b"GGUF", "not a GGUF"
    ver = struct.unpack_from("<I", data, 4)[0]
    n_tensors = struct.unpack_from("<Q", data, 8)[0]
    n_kv = struct.unpack_from("<Q", data, 16)[0]
    off = 24
    print(f"== {path}: ver={ver} n_tensors={n_tensors} n_kv={n_kv}")
    if show_kv:
        for _ in range(n_kv):
            k, off = read_str(data, off)
            t = struct.unpack_from("<I", data, off)[0]; off += 4
            v, off = read_val(data, off, t)
            print(f"   KV {k} = {v if len(str(v)) < 80 else str(v)[:77]+'...'}")
    else:
        # skip KV but remember interesting keys
        interesting = {}
        for _ in range(n_kv):
            k, off = read_str(data, off)
            t = struct.unpack_from("<I", data, off)[0]; off += 4
            v, off = read_val(data, off, t)
            if k in ("general.file_type", "general.architecture", "qwen4exp.ple.layers",
                     "qwen4exp.ple.ngram_size", "qwen4exp.ple.heads_per_ngram",
                     "qwen4exp.embedding_length_per_layer_input", "split.count", "split.no"):
                interesting[k] = v
        for k, v in interesting.items():
            print(f"   KV {k} = {v}")
    print(f"   --- tensors ---")
    total = 0
    for i in range(n_tensors):
        name, off = read_str(data, off)
        nd = struct.unpack_from("<I", data, off)[0]; off += 4
        dims = struct.unpack_from(f"<{nd}Q", data, off); off += 8*nd
        t = struct.unpack_from("<I", data, off)[0]; off += 4
        toff = struct.unpack_from("<Q", data, off)[0]; off += 8
        ne = "x".join(str(d) for d in reversed(dims))
        if only_tensors and not any(f in name for f in only_tensors):
            continue
        total += 1
        print(f"   {name:60s} {GGML_TYPES.get(t, t):10s} [{ne:>24s}] off={toff}")
    return off  # header size

if __name__ == "__main__":
    path = sys.argv[1]
    show_kv = len(sys.argv) > 2 and sys.argv[2] == "kv"
    filt = sys.argv[3:] if len(sys.argv) > 3 else None
    end = parse(path, show_kv=show_kv, only_tensors=filt)
    print(f"   (header ends at {end} bytes)")
