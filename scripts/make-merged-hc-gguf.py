#!/usr/bin/env python3
"""Rewrite a pre-#31028 qwen4exp GGUF into the merged hc_*_down_inject layout.

WHY THIS EXISTS
  Upstream #31028's converter writes `hc_{attn,ffn}_down` and `hc_{attn,ffn}_inject` as ONE
  `blk.N.hc_{attn,ffn}_down_inject` tensor, so `llama_model_loader::create_merged()` leaves
  `w_down == nullptr` and `llama_model_qwen4exp::graph::build_hc_mix()` takes its merged branch.
  This fleet has no such checkpoint (the local Flash-Next GGUFs are all separate-tensor), and that
  was the reason the merged path shipped unexercised for six releases and then segfaulted at
  `sched_reserve` (contributor PR #321: the fused-mixer gate dereferenced `w_down` before testing
  it).  This tool builds the fixture from a GGUF the fleet does have, so the merged layout can be
  gated with `scripts/gate-hc-fusions.sh MODEL.gguf` and `test-llama-archs -a qwen4exp`.

WHAT IT DOES
  For every block that carries both `hc_*_down` and `hc_*_inject` (same type, same ne[0]), it
  writes one `hc_*_down_inject` whose row-major data is the down rows followed by the inject rows
  -- exactly what `create_merged(..., n_in, n_a, n_b)` and `build_hc_mix`'s split (`n_lo =
  ne[1] - hc`, inject = the `hc` tail rows) expect.  Every other tensor is copied byte for byte,
  the shards are concatenated into one file, and the split keys are dropped.

  The output is a TEST FIXTURE, not a redistribution artifact: it is the same weights twice over
  once (both layouts) and takes the model's size again on disk.

USAGE
  ./make-merged-hc-gguf.py OUT.gguf SHARD0.gguf [SHARD1.gguf ...]
  ./make-merged-hc-gguf.py --list IN.gguf          # report what would be merged, write nothing
  ./make-merged-hc-gguf.py --verify IN.gguf        # check an existing merged GAINS nothing

  Exit: 0 = ok, 1 = the input is not a qwen4exp GGUF / nothing to merge, 2 = usage or IO error.

NOTES
  * The block size / type size table below is verified against the file: every tensor's computed
    byte size must fit between its data offset and the next tensor's (the slack is the alignment
    padding, `< general.alignment`), so a wrong table entry is an error, never silent corruption.
  * A tensor whose pair is missing (an MTP block whose inject lives elsewhere, a synthetic model)
    is left alone; `create_merged` then falls back to the separate tensors for that block.
  * Only the type table's entries are understood; anything else is refused rather than guessed.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

# ggml type -> (block_size, type_size).  Mirrors ggml/src/ggml.c's type_traits; verified against
# the source file's offsets, so an entry that is wrong for this file cannot pass silently.
QUANT_SIZES = {
    0:  (1, 4),      # F32
    1:  (1, 2),      # F16
    2:  (32, 18),    # Q4_0
    3:  (32, 20),    # Q4_1
    6:  (32, 22),    # Q5_0
    7:  (32, 24),    # Q5_1
    8:  (32, 34),    # Q8_0
    9:  (32, 36),    # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    # the fork's newer types (ggml.h: 34/35 TQ1_0/TQ2_0, 39 MXFP4, 40 NVFP4, 41 Q1_0, 42 Q2_0,
    # 43 F8_E4M3; block sizes from ggml-common.h, verified against the file's offsets)
    34: (256, 54),   # TQ1_0
    35: (256, 66),   # TQ2_0
    39: (32, 17),    # MXFP4
    40: (64, 36),    # NVFP4
    41: (128, 18),   # Q1_0
    42: (64, 18),    # Q2_0
    43: (1, 1),      # F8_E4M3
}

T_STRING = 8
T_ARRAY = 9


def die(msg: str, code: int = 2) -> "NoReturn":  # noqa: F821
    print("make-merged-hc-gguf: " + msg, file=sys.stderr)
    raise SystemExit(code)


def read_str(f) -> bytes:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def skip_value(f, t: int) -> None:
    if t in (0, 1, 7):
        f.read(1)
    elif t in (2, 3):
        f.read(2)
    elif t in (4, 5, 6):
        f.read(4)
    elif t in (10, 11, 12):
        f.read(8)
    elif t == T_STRING:
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n)
    elif t == T_ARRAY:
        at, n = struct.unpack("<IQ", f.read(12))
        for _ in range(n):
            skip_value(f, at)
    else:
        die("unknown GGUF value type %d" % t)


def nbytes(dims, tt: int) -> int:
    if tt not in QUANT_SIZES:
        die("GGUF type id %d is not in this tool's table (add it from ggml.c's type_traits)" % tt)
    blck, size = QUANT_SIZES[tt]
    ne = 1
    for d in dims:
        ne *= int(d)
    if ne % blck:
        die("tensor with %d elements is not a multiple of the block size %d of type %d" % (ne, blck, tt))
    return ne // blck * size


class Shard:
    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                die("%s is not a GGUF file" % path)
            (self.version,) = struct.unpack("<I", f.read(4))
            if self.version != 3:
                die("%s: GGUF version %d (only 3 is understood)" % (path, self.version))
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            self.kv = []          # (key, type, raw value bytes)
            self.kv_map = {}
            for _ in range(n_kv):
                key = read_str(f).decode("utf-8", "replace")
                (t,) = struct.unpack("<I", f.read(4))
                start = f.tell()
                skip_value(f, t)
                end = f.tell()
                f.seek(start)
                raw = f.read(end - start)
                self.kv.append((key, t, raw))
                if t == T_STRING:
                    (n,) = struct.unpack("<Q", raw[:8])
                    self.kv_map[key] = raw[8:8 + n].decode("utf-8", "replace")
                elif t == 4:
                    self.kv_map[key] = struct.unpack("<I", raw[:4])[0]
            self.tensors = []      # (name, dims, type, data offset)
            for _ in range(n_tensors):
                name = read_str(f).decode("utf-8", "replace")
                (nd,) = struct.unpack("<I", f.read(4))
                dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
                (tt,) = struct.unpack("<I", f.read(4))
                (off,) = struct.unpack("<Q", f.read(8))
                self.tensors.append((name, dims, tt, off))
            self.alignment = int(self.kv_map.get("general.alignment", 32))
            self.data_start = f.tell() + ((-f.tell()) % self.alignment)
        self.data_len = self.size - self.data_start
        # the type table is verified here: sizes must tile the data section
        ordered = sorted(self.tensors, key=lambda t: t[3])
        for i, (name, dims, tt, off) in enumerate(ordered):
            want = nbytes(dims, tt)
            limit = (ordered[i + 1][3] - off) if i + 1 < len(ordered) else (self.data_len - off)
            if want > limit:
                die("%s: tensor %s needs %d bytes but only %d remain before the next tensor "
                    "(the type table or the file is wrong)" % (path, name, want, limit))
            if limit - want >= self.alignment:
                die("%s: %d unexplained bytes after %s (table/file mismatch)"
                    % (path, limit - want, name))

    def close(self) -> None:
        os.close(self.fd)


def merge_plan(shards):
    """[(merged name, dims, type, [(shard, offset, nbytes), ...])], plus the names to drop."""
    drops: set[str] = set()
    merged = []
    for s in shards:
        for name, dims, tt, off in s.tensors:
            if not (name.endswith(".hc_attn_down.weight") or name.endswith(".hc_ffn_down.weight")):
                continue
            inject = name[: -len("_down.weight")] + "_inject.weight"
            sj = next((x for x in shards if any(t[0] == inject for t in x.tensors)), None)
            if sj is None:
                continue
            tj = next(t for t in sj.tensors if t[0] == inject)
            if tj[2] != tt or tj[1][0] != dims[0] or len(dims) != 2 or len(tj[1]) != 2:
                die("%s and %s are not a mergeable pair (type %d/%d, shapes %s/%s)"
                    % (name, inject, tt, tj[2], dims, tj[1]))
            merged.append((name[: -len("_down.weight")] + "_down_inject.weight",
                           [dims[0], dims[1] + tj[1][1]], tt,
                           [(s, off, nbytes(dims, tt)), (sj, tj[3], nbytes(tj[1], tj[2]))]))
            drops.add(name)
            drops.add(inject)
    return merged, drops


def cmd_list(path: str) -> int:
    s = Shard(path)
    merged, _ = merge_plan([s])
    if not merged:
        print("%s: no hc_*_down/hc_*_inject pair -- nothing to merge" % path)
        return 1
    print("%s: %d pair(s) to merge, alignment %d, data %.2f GiB"
          % (path, len(merged), s.alignment, s.data_len / 2**30))
    for name, dims, tt, src in merged:
        print("  %-44s [%d, %d] type %d  <- %d byte(s)"
              % (name, dims[0], dims[1], tt, sum(x[2] for x in src)))
    s.close()
    return 0


def cmd_merge(out_path: str, shard_paths, verify: bool) -> int:
    shards = [Shard(p) for p in shard_paths]
    if all("split.no" in s.kv_map for s in shards):
        shards.sort(key=lambda s: int(s.kv_map["split.no"]))
    alignment = shards[0].alignment
    for s in shards:
        if s.alignment != alignment:
            die("%s: alignment %d differs from %s's %d" % (s.path, s.alignment, shards[0].path, alignment))

    merged, drops = merge_plan(shards)
    if not merged:
        die("no hc_*_down/hc_*_inject pair: this is not a pre-#31028 qwen4exp GGUF", 1)

    # metadata: every key once, minus the split markers (the output is one file)
    meta, seen = [], set()
    for s in shards:
        for key, t, raw in s.kv:
            if key in ("split.count", "split.no", "split.tensors.count") or key in seen:
                continue
            seen.add(key)
            meta.append((key, t, raw))

    entries = []       # (name, dims, type, [(shard, offset, nbytes)])
    for s in shards:
        for name, dims, tt, off in s.tensors:
            if name not in drops:
                entries.append((name, dims, tt, [(s, off, nbytes(dims, tt))]))
    for name, dims, tt, src in merged:
        entries.append((name, dims, tt, src))

    offset = 0
    placed = []
    for name, dims, tt, src in entries:
        offset += (-offset) % alignment
        placed.append((name, dims, tt, src, offset, sum(x[2] for x in src)))
        offset += placed[-1][5]

    with open(out_path, "wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", 3))
        out.write(struct.pack("<QQ", len(placed), len(meta)))
        for key, t, raw in meta:
            kb = key.encode()
            out.write(struct.pack("<Q", len(kb)))
            out.write(kb)
            out.write(struct.pack("<I", t))
            out.write(raw)
        for name, dims, tt, _src, off, _nb in placed:
            nb = name.encode()
            out.write(struct.pack("<Q", len(nb)))
            out.write(nb)
            out.write(struct.pack("<I", len(dims)))
            out.write(struct.pack("<%dQ" % len(dims), *dims))
            out.write(struct.pack("<I", tt))
            out.write(struct.pack("<Q", off))
        out.write(b"\0" * ((-out.tell()) % alignment))
        data_start = out.tell()

        chunk = 1 << 24
        buf = bytearray(chunk)
        done = 0
        for name, _dims, _tt, src, off, nb in placed:
            target = data_start + off
            out.write(b"\0" * (target - out.tell()))
            for shard, soff, sbytes in src:
                pos = shard.data_start + soff
                left = sbytes
                while left:
                    got = os.preadv(shard.fd, [memoryview(buf)[: min(left, chunk)]], pos)
                    if got <= 0:
                        die("%s: short read while copying %s" % (shard.path, name))
                    out.write(buf[:got])
                    pos += got
                    left -= got
            done += nb
            if done // (4 << 30) != (done - nb) // (4 << 30):
                print("  %.1f GiB written" % (done / 2**30), file=sys.stderr)
        out.flush()
        os.fsync(out.fileno())

    for s in shards:
        s.close()
    print("%s: %d tensors (%d merged), %.2f GiB"
          % (out_path, len(placed), len(merged), os.path.getsize(out_path) / 2**30))

    if verify:
        return cmd_verify(out_path, merged)
    return 0


def cmd_verify(merged_path: str, expect=None) -> int:
    """Re-read the written file: the header parses, the merged tensors' rows are their sources'."""
    s = Shard(merged_path)          # the type table is re-verified against the output's offsets
    got = {t[0]: t for t in s.tensors}
    n = 0
    for name, dims, tt, _src in (expect or []):
        if name not in got:
            die("%s: %s is missing from the merged file" % (merged_path, name), 1)
        gdims, gtt = got[name][1], got[name][2]
        if [int(d) for d in gdims] != [int(d) for d in dims] or gtt != tt:
            die("%s: %s is [%s] type %d, expected [%s] type %d"
                % (merged_path, name, gdims, gtt, dims, tt), 1)
        n += 1
    separate = [k for k in got if k.endswith(".hc_attn_down.weight") or k.endswith(".hc_ffn_down.weight")]
    if separate:
        die("%s: still carries a separate %s -- the merge did not drop the pair"
            % (merged_path, separate[0]), 1)
    print("%s: verified %d merged tensor(s), no separate hc_*_down left, offsets tile the data section"
          % (merged_path, n))
    s.close()
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="report the pairs a file would merge, write nothing")
    ap.add_argument("--verify", action="store_true", help="re-read the OUTPUT after writing and check it")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)

    if a.list:
        if len(a.paths) != 1:
            die("--list takes exactly one file")
        return cmd_list(a.paths[0])
    if len(a.paths) < 2:
        die("usage: %s [--verify] OUT.gguf SHARD0.gguf [SHARD1.gguf ...]" % sys.argv[0])
    return cmd_merge(a.paths[0], a.paths[1:], a.verify)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
