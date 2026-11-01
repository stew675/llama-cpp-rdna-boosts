#!/usr/bin/env python3
"""Decode-band kernel census from two rocprofv3 runs of the SAME arm at two -n lengths (BA-POOL-DESIGN.md §8.4).

    rocprofv3 --kernel-trace --stats -f csv -o short -- <llama-cli ...> -n 2
    rocprofv3 --kernel-trace --stats -f csv -o long  -- <llama-cli ...> -n 130

    # one arm: per-token decode GPU time, per kernel and per bucket
    ./kv-kernel-census.py short_kernel_stats.csv long_kernel_stats.csv 2 130

    # two arms, same shape: the per-kernel DELTA (e.g. pool - dense)
    ./kv-kernel-census.py short_kernel_stats.csv long_kernel_stats.csv 2 130 \
                         --vs pool-short pool-long 2 130

The prefill is identical in the two runs of an arm and the one-time lazy code-object loads happen in both, so
(t_long - t_short)/(n_long - n_short) is the steady-state DECODE cost per token, kernel by kernel.  Only kernels
whose CALL COUNT scales with tokens are kept: a prefill-only kernel drifting between two runs otherwise
contributes milliseconds per token of noise (that filter is what makes the method usable at all).

Plain decode (no speculator) has 1 token = 1 step, so its per-token numbers ARE its per-step numbers and there is
no acceptance confound -- that is the arm to use when the question is what the mechanism costs per step.
"""
import collections
import csv
import sys

CLASSES = [
    ("A indexer score chain", ("indexer_score", "idx_relu_sum", "k_get_rows", "k_bin_bcast", "rope_multi",
                               "rms_rows", "mul_mat_vec_f<float, float, 20", "mul_mat_vec_f_vb",
                               "mul_mat_vec_f<__hip_bfloat16, float, 20")),
    ("B radix top-k",         ("indexer_topk",)),
    ("P pool resolve",        ("kv_pool",)),
    ("C FA",                  ("flash_attn",)),
    ("D copies",              ("copyBuffer", "Memcpy", "memcpy", "fillBuffer")),
]
MIN_CALLS = 0.3     # decode-band kernels only


def classify(name):
    for label, keys in CLASSES:
        if any(k in name for k in keys):
            return label
    return "E everything else"


def load(path):
    dur, calls = collections.Counter(), collections.Counter()
    with open(path, newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            dur[row["Name"]] += int(row["TotalDurationNs"])
            calls[row["Name"]] += int(row["Calls"])
    return dur, calls


def per_token(short, long_, ns, nl):
    """{kernel: ms/token}, {kernel: calls/token} for the decode band of one arm."""
    ds, cs = load(short)
    dl, cl = load(long_)
    ntok = nl - ns
    keys = set(ds) | set(dl)
    dur = {k: (dl[k] - ds[k]) / 1e6 / ntok for k in keys}
    calls = {k: (cl[k] - cs[k]) / ntok for k in keys}
    return dur, calls


def print_arm(dur, calls, label):
    rows = [(classify(k), v, calls[k], k) for k, v in dur.items() if calls[k] >= MIN_CALLS]
    rows.sort(key=lambda r: -r[1])
    tot = sum(r[1] for r in rows)
    print(f"\n# {label}: {tot:.3f} ms/token of decode GPU kernel time "
          f"(kernel-only ceiling {1000.0/tot if tot else 0:.1f} tok/s)")
    print(f"{'bucket':<24} {'ms/token':>9} {'%':>6} {'calls/tok':>10}  kernel")
    for cls, ms, c, k in rows:
        if ms > 0.02 or c > 1.0:
            print(f"{cls:<24} {ms:>9.3f} {100*ms/tot:>5.1f}% {c:>10.2f}  {k[:88]}")
    by = collections.Counter()
    for cls, ms, c, k in rows:
        by[cls] += ms
    print("  -- by bucket")
    for cls in sorted(by):
        print(f"{cls:<24} {by[cls]:>9.3f} {100*by[cls]/tot:>5.1f}%")
    print(f"{'TOTAL':<24} {tot:>9.3f}")
    return tot


def print_delta(dur, calls, odur, ocalls, label):
    rows = []
    for k in set(dur) | set(odur):
        if max(calls.get(k, 0.0), ocalls.get(k, 0.0)) < MIN_CALLS:
            continue
        rows.append((dur.get(k, 0.0) - odur.get(k, 0.0), odur.get(k, 0.0), dur.get(k, 0.0),
                     ocalls.get(k, 0.0), calls.get(k, 0.0), k))
    rows.sort(key=lambda r: -r[0])
    td, to = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"\n# DELTA {label}: {to:.3f} vs {td:.3f} ms/token = {to-td:+.3f} "
          f"({100*(to-td)/td if td else 0:+.1f} %)")
    print(f"{'bucket':<24} {'delta':>8} {'other':>7} {'this':>7} {'calls':>8} {'calls':>8}  kernel")
    for d, o, t, oc, tc, k in rows:
        if abs(d) > 0.02 or tc > 1.0:
            print(f"{classify(k):<24} {d:>8.3f} {o:>7.3f} {t:>7.3f} {oc:>8.2f} {tc:>8.2f}  {k[:80]}")
    by = collections.Counter()
    for d, o, t, oc, tc, k in rows:
        by[classify(k)] += d
    print("  -- by bucket")
    for cls in sorted(by):
        print(f"{cls:<24} {by[cls]:>8.3f}")


def main():
    argv = sys.argv[1:]
    if "--vs" in argv:
        i = argv.index("--vs")
        mine, other = argv[:i], argv[i + 1:]
    else:
        mine, other = argv, None
    if len(mine) != 4 or (other is not None and len(other) != 4):
        print(__doc__)
        return 2
    dur, calls = per_token(mine[0], mine[1], int(mine[2]), int(mine[3]))
    tot = print_arm(dur, calls, f"{mine[0]} -> {mine[1]}")
    if other is not None:
        odur, ocalls = per_token(other[0], other[1], int(other[2]), int(other[3]))
        # the caller passes the *arm* pairs in a fixed order; the label reads "this minus --vs"
        print_delta(dur, calls, odur, ocalls, f"{mine[0]} minus {other[0]}")
        print(f"\nthis arm {tot:.3f} ms/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
