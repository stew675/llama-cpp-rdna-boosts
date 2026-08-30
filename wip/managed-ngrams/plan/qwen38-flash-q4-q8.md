# Qwen3.8-Flash-Next: UD-Q4_K_XL weights + Q8_0 ngram table (the "frankenstein")

Goal: run Qwen3.8-Flash-Next on a 96 GB VRAM / ~32 GB system RAM machine (e.g. a
Strix Halo with a 96 GB carve-out), with the **Unsloth UD-Q4_K_XL model weights**
(~82.5 GB, fits VRAM) and the **Q8_0 ngram table** (54.4 GB, kept on disk and
mmap'd into system RAM on demand).

Unsloth does not ship this combination:

| Build | Weights (fits 96 GB VRAM?) | Ngram |
|---|---|---|
| UD-Q4_K_XL | 82.5 GB yes | IQ4_NL (28.8 GB) |
| UD-Q5_K_XL / UD-Q6_K_XL | 104+ GB no | Q8_0 (54.4 GB) |
| Q8_0 | 134 GB no | Q8_0 (54.4 GB) |

The good news: the Q8_0 ngram is **byte-identical across the Q8_0 / Q5_K_XL /
Q6_K_XL builds** (same LFS blob, sha256 `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20`),
and in each build it sits alone in file `00003` as a single tensor
`per_layer_token_embd.weight` `Q8_0 [160 x 320001536]`. So we graft that one
tensor into a merged UD-Q4_K_XL file. Everything else is copied byte-for-byte,
so **both halves stay bit-exact**.

Result: 136.9 GB (82.5 GB weights + 54.4 GB ngram), split into 3 files.

---

## 1. Prerequisites

### Downloads (165 GB total)

| File | Size | Purpose |
|---|---|---|
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` | 111.3 GB | model weights |
| any of `Q8_0/` , `UD-Q5_K_XL/` , `UD-Q6_K_XL/` `...-00003-of-00006.gguf` | 54.4 GB | the Q8_0 ngram (all three are the same blob) |

```
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
    --local-dir unsloth-gq4 \
    --include "UD-Q4_K_XL/*"
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
    --local-dir unsloth-gq8 \
    --include "Q8_0/*00003-of-00006*"
```

### Disk

The two-step method below needs roughly:

| step | peak |
|---|---|
| merge | 111 + 111 = 222 GB |
| graft | 111 + 137 = 248 GB |
| re-split | 137 + 137 = 274 GB |

Plan for ~300 GB free, or delete intermediates (`merged.gguf`, `grafted.gguf`)
after each step. (A one-pass script variant that writes the three output splits
directly from the four input files needs ~250 GB; see section 6.)

### Tools

- `llama-gguf-split` (build from this repo: `cmake --build build --target llama-gguf-split`)
- `python3` (stdlib only for the graft script)

---

## 2. Step 1: merge UD-Q4_K_XL into a single file

```
llama-gguf-split --merge \
    unsloth-gq4/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    merged.gguf
```

The merged file has `split.count = 0`, which llama.cpp treats as a single file.

## 3. Step 2: graft the Q8_0 ngram in

Save the script below as `graft-ngram-q8.py` and run:

```
python3 graft-ngram-q8.py merged.gguf \
    unsloth-gq8/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf \
    grafted.gguf
```

It rewrites `merged.gguf` to `grafted.gguf`, copying every tensor's bytes
verbatim except `per_layer_token_embd.weight`, which is replaced by the Q8_0
data from the ngram file (streamed in 64 MB chunks, so RAM usage stays small).
The output header keeps all metadata KV pairs and the same tensor order; only
the ngram tensor's type field changes (`IQ4_NL` -> `Q8_0`) and all data offsets
are recomputed.

```python
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
```

## 4. Step 3: re-split and verify

```
llama-gguf-split --split --split-max-size 50G grafted.gguf out/
```

This produces 3 files (`out-00001-of-00003.gguf`, ...). The ngram (54.4 GB)
ends up in its own split, which is expected.

Sanity checks:

```
# 1. tensor listing shows per_layer_token_embd.weight as Q8_0 [160 x 320001536]
llama-server -m out/out-00001-of-00003.gguf --no-webui   # or any load

# 2. structural check (reuse the parser from the script):
python3 - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("g", "graft-ngram-q8.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
for f in ["out/out-00001-of-00003.gguf", "out/out-00002-of-00003.gguf", "out/out-00003-of-00003.gguf"]:
    kv_end, tensors, ds = g.parse_header(f)
    for t in tensors:
        if t["name"].endswith(b"per_layer_token_embd.weight"):
            print(f, "ngram:", t["type"], t["dims"])
EOF
```

Expected output: `ngram: 8 (160, 320001536)` on the split that holds it.

---

## 5. Running it

```
llama-cli -m out/out-00001-of-00003.gguf -ngl 99 --lazy-mode auto
```

- `--lazy-mode auto` (the default) keeps `per_layer_token_embd.weight` mmap'd:
  its 54.4 GB are excluded from load-time prefetch and faulted in from disk on
  demand, so it never touches VRAM and does not need to be read at load time.
  The other 82.5 GB go into the 96 GB VRAM carve-out.
- The PLE layer's small projections (`ple_key`, `ple_value`, ...) are regular
  weights and are offloaded like everything else; only the ngram *table* is
  RAM-backed.

**Important caveat for this machine (96 GB VRAM carve-out -> ~32 GB system RAM):**
the Q8_0 ngram is 54.4 GB, which *cannot* fit in system RAM all at once, and the
current lazy mmap has no residency cap (the kernel's page cache decides). So
today the kernel will thrash between the hot ngram rows and everything else.
The IQ4_NL ngram of the stock UD-Q4_K_XL (28.8 GB) at least fits. **The Q8_0
graft pays off once the bounded "lazy window" (user-defined mmap buffer size +
LRU eviction) lands in llama.cpp**; with a 10 GB window it becomes strictly
better than stock: 22 GB of system RAM left free, hot rows cached LRU, PLE at
twice the precision.

---

## 6. Notes

### Why Unsloth doesn't ship this

- GGUF is distributed as a single model family; splits exist to beat file-size
  limits, not as a composition mechanism, and there is no official
  "swap one tensor" tool (`llama-gguf-split` only merges/splits). The script
  above is effectively that missing utility.
- The "UD" (Dynamic 3.0) scheme deliberately couples ngram precision to build
  tier: Q8_0 ngram in the top builds, IQ4_NL in everything from Q4_K_XL down
  (even the 1-bit IQ1_M build keeps the ngram at 4.5 bits). So a
  "Q4_K_XL weights + Q8_0 ngram" tier simply isn't one of their matrix points.
- GGUF's per-tensor type support is exactly what makes the graft possible: the
  loader reads each tensor's own type, and the PLE gather (`ggml_get_rows`)
  dequantizes Q8_0 (or any type) on the fly.

### Reference: UD-Q4_K_XL per-tensor map (useful if you instead re-quantize)

| Tensor group | Type |
|---|---|
| `token_embd`, `output`, `output_hc_*` | Q8_0 |
| all attention (`attn_q/k/v/qkv/gate/output`), `ssm_out`, `hc_attn_*`/`hc_ffn_*` down/up, `ple_key`, `ple_value`, all `*_shexp` | Q8_0 |
| `ffn_down_exps` | Q5_1, except layers 2, 4, 30, 46, 47 -> Q8_0 |
| `ffn_gate_exps`, `ffn_up_exps` | Q4_K, except layer 2 -> Q5_K |
| `per_layer_token_embd` | IQ4_NL (grafted to Q8_0 here) |
| `indexer.k/q_proj` | BF16 |
| norms, `ffn_gate_inp*`, `ssm_conv1d`, `ssm_alpha`, `ssm_beta`, `hc_*_inject`, `ple_conv1d` | F32 |

### Alternative: re-quantize the Q8_0 build (no script)

`llama-quantize` accepts regex per-tensor overrides (`--tensor-type
Q8_0=per_layer_token_embd` etc.) and can reproduce this map from the Q8_0
build, but: it needs the full 188 GB Q8_0 download, the ngram goes
Q8_0 -> f32 -> Q8_0 (near-lossless, not bit-exact), and you must get ~15
override patterns right. The graft is simpler and exact.

### One-pass variant

The script can be adapted to read the four UD-Q4_K_XL splits and write the
three output splits directly (no `merged.gguf` / `grafted.gguf`
intermediates), saving ~250 GB of peak disk. It needs to re-emit the split
metadata KV keys (`split.count`, `split.no`, `split.tensors.count`) and pack
tensors into files of ~50 GB in tensor order. Ask if you want this version.
