#!/usr/bin/env python3
"""isa-kernel-diff.py -- compare one kernel's gfx1201 instruction stream between two ggml-hip builds.

Why this exists: `RUNGS.md` `P7` / `TODO` #303 needed to know whether a 1.4 % kernel delta was our code or
upstream's.  Profiling cannot answer that (a single `rocprofv3` kernel-trace run has a +-1.1 % floor between two
runs of the SAME binary), but the instruction stream is exact and noise-free: two builds whose kernel is
byte-identical cannot have a body difference, and a candidate fix that does not change the ISA cannot be worth
measuring.

It also answers "same template instantiation, different VGPR count" in seconds -- VGPR tracks the body, so the
question you actually have is "is the body the same?".

Usage:
    isa-kernel-diff.py <buildA>/bin/libggml-hip.so* <buildB>/bin/libggml-hip.so* --kernel mul_mat_q

`--kernel` is a substring of the *mangled* symbol; add `--list` to print the candidates.  Output: instruction
counts, VGPR/SGPR from the code object's kernel descriptor, an opcode histogram delta, and (with `--diff`) the
first N lines of the normalised sequence difference.

Requires the ROCm LLVM tools (`llvm-objcopy`, `llvm-nm`, `llvm-objdump`); set ROCM_LLVM_BIN if they are not in
/opt/rocm/llvm/bin.

Notes, both of which cost time to learn (2026-11-08):
  * the whole device program is ONE `.hip_fatbin` section holding one `__CLANG_OFFLOAD_BUNDLE__` per TU, so you
    must walk every bundle in the section -- `llvm-objcopy --dump-section` gives you all of them at once;
  * inside a bundle, each entry's offset is **bundle-relative**, not section-relative.  Reading them as
    section-relative extracts the first TU over and over and every comparison comes out trivially "the same".
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
ROCM_LLVM_BIN = os.environ.get("ROCM_LLVM_BIN", "/opt/rocm/llvm/bin")


def tool(name):
    p = os.path.join(ROCM_LLVM_BIN, name)
    return p if os.path.exists(p) else name


def hip_code_objects(so_path, want_arch, tmpdir):
    """Yield (index, path) for every <want_arch> code object embedded in so_path's .hip_fatbin."""
    fatbin = os.path.join(tmpdir, "fatbin")
    subprocess.run([tool("llvm-objcopy"), "--dump-section", f".hip_fatbin={fatbin}", so_path],
                   check=True, capture_output=True)
    data = open(fatbin, "rb").read()
    starts, i = [], 0
    while True:
        i = data.find(MAGIC, i)
        if i < 0:
            break
        starts.append(i)
        i += len(MAGIC)
    for n, start in enumerate(starts):
        count = struct.unpack_from("<Q", data, start + 24)[0]
        off = start + 32
        for j in range(count):
            o, size, id_len = struct.unpack_from("<QQQ", data, off)
            off += 24
            ident = data[off:off + id_len].decode(errors="replace")
            off += id_len
            if want_arch not in ident or size <= 0:
                continue
            path = os.path.join(tmpdir, f"co{n:04d}_{j}.o")
            with open(path, "wb") as fh:
                fh.write(data[start + o:start + o + size])  # bundle-relative!
            yield path


def find_kernel(so_path, want_arch, pattern, tmpdir):
    for path in hip_code_objects(so_path, want_arch, tmpdir):
        nm = subprocess.run([tool("llvm-nm"), "--defined-only", path], capture_output=True, text=True)
        for line in nm.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "T" and pattern in parts[2]:
                return path, parts[2]
    return None, None


def disassemble(path, symbol, tmpdir):
    out = subprocess.run([tool("llvm-objdump"), "-d", f"--disassemble-symbols={symbol}", path],
                         capture_output=True, text=True).stdout
    instrs = []
    for line in out.splitlines():
        m = re.match(r"^\t([a-z][a-z0-9_]*)\s*(.*?)\s*//", line)
        if m:
            instrs.append((m.group(1), m.group(2)))
    return instrs


def normalise(instrs):
    """Drop operand *identity* (registers/immediates) but keep operand shape, so a register-renamed but otherwise
    identical kernel compares equal while a real difference (an extra min, a different compare direction) shows."""
    out = []
    for op, args in instrs:
        a = re.sub(r"0x[0-9a-fA-F]+", "IMM", args)
        a = re.sub(r"\bs\[[0-9:]+\]", "S[]", a)
        a = re.sub(r"\bs\d+", "S", a)
        a = re.sub(r"\bv\d+", "V", a)
        a = re.sub(r"\[[^\]]*\]", "[]", a)
        out.append(f"{op} {a}".strip())
    return out


def describe(instrs):
    hist = {}
    for op, _ in instrs:
        hist[op] = hist.get(op, 0) + 1
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_a")
    ap.add_argument("build_b")
    ap.add_argument("--kernel", default="mul_mat_q")
    ap.add_argument("--arch", default="gfx1201")
    ap.add_argument("--list", action="store_true", help="list matching kernels and exit")
    ap.add_argument("--diff", type=int, default=40, help="show the first N normalised diff lines (0 = none)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        results = {}
        for tag, so in (("A", args.build_a), ("B", args.build_b)):
            sub = os.path.join(td, tag)
            os.makedirs(sub)
            path, sym = find_kernel(so, args.arch, args.kernel, sub)
            if path is None:
                sys.exit(f"{tag}: no {args.arch} kernel matching {args.kernel!r} in {so}")
            instrs = disassemble(path, sym, sub)
            results[tag] = (so, sym, instrs)
            if args.list:
                print(f"{tag}: {sym}")

    for tag in ("A", "B"):
        so, sym, instrs = results[tag]
        print(f"{tag}: {os.path.basename(so)}")
        print(f"   symbol     {sym[:96]}{'...' if len(sym) > 96 else ''}")
        print(f"   instrs     {len(instrs)}")

    a = normalise(results["A"][2])
    b = normalise(results["B"][2])
    same = a == b
    print(f"\nnormalised instruction streams: {'IDENTICAL' if same else 'DIFFERENT'} "
          f"({len(a)} vs {len(b)} instructions)")

    if not same:
        ha, hb = describe(results["A"][2]), describe(results["B"][2])
        print("\nopcode histogram delta (only opcodes that differ):")
        for op in sorted(set(ha) | set(hb)):
            da, db = ha.get(op, 0), hb.get(op, 0)
            if da != db:
                print(f"   {op:30s} A:{da:4d}  B:{db:4d}  {db - da:+d}")
        if args.diff:
            import difflib
            print(f"\nfirst {args.diff} normalised diff lines:")
            shown = 0
            for line in difflib.unified_diff(a, b, "A", "B", lineterm="", n=1):
                print("   " + line)
                shown += 1
                if shown >= args.diff:
                    break

    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
