#!/usr/bin/env python3
"""Fold a rocprofv3 kernel-trace CSV into per-kernel and per-class GPU time.

gfx1201 note: rocprofv3 PMC counters return 0 on this box, so this uses only the
kernel-trace timestamps (Start/End), which do work.

Usage:
    fold-kernel-trace.py <kernel_trace.csv> [--top N] [--class]
"""
import csv
import sys
import collections
import re

# Kernel-name -> coarse class.  Order matters (first match wins).
CLASSES = [
    ("attn",    r"flash_attn|fattn|paged_attn|attention"),
    ("mmq",     r"mul_mat_q"),
    ("mmvq",    r"mul_mat_vec"),
    ("mmf",     r"mul_mat_f|mul_mat_vec_f"),
    ("quant",   r"quantize|cvt_|convert"),
    ("gdn",     r"gated_delta_net|gdn_|ssm_|mamba"),
    ("moe",     r"moe|topk|expert|router"),
    ("arreduce",r"allreduce|nccl|all_reduce|reduce_scatter|allgather"),
    ("norm",    r"rms_norm|norm_|layernorm"),
    ("rope",    r"rope"),
    ("copy",    r"copy|memcpy|pad|cont|permute|transpose"),
    ("glue",    r"get_rows|set_rows|dup|repeat|concat|view|sum_rows|argsort|soft_max|top_k"),
]


def classify(name: str) -> str:
    n = name.lower()
    for cls, pat in CLASSES:
        if re.search(pat, n):
            return cls
    return "other"


def short(name: str) -> str:
    # keep the primary template tag, strip the rest
    m = re.search(r"mul_mat_q<\(ggml_type\)(\d+)", name)
    if m:
        return "mul_mat_q"
    return name[:70]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    top = 30
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])

    per_kernel = collections.Counter()
    per_class = collections.Counter()
    n_kernels = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Kernel_Name", "")
            if not name:
                continue
            try:
                dur = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            except (KeyError, ValueError):
                continue
            n_kernels += 1
            per_kernel[short(name)] += dur
            per_class[classify(name)] += dur

    total = sum(per_kernel.values())
    if total == 0:
        print("no kernel durations found")
        return 1
    ms_total = total / 1e6

    print(f"# dispatches: {n_kernels}   total GPU kernel time: {ms_total:.1f} ms")
    print()
    print("## by class")
    print(f"{'class':10s} {'ms':>10s} {'share':>7s}")
    for cls, ns in per_class.most_common():
        print(f"{cls:10s} {ns/1e6:10.1f} {100*ns/total:6.1f}%")
    print()
    print(f"## top {top} kernels")
    print(f"{'ms':>10s} {'share':>7s}  kernel")
    for name, ns in per_kernel.most_common(top):
        print(f"{ns/1e6:10.1f} {100*ns/total:6.1f}%  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
