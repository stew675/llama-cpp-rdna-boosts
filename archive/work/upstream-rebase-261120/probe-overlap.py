#!/usr/bin/env python3
"""Measure, per delivery block, how much UPSTREAM churn lands on the files that block owns.

WHY: probe-conflicts.py answers "does the text merge?"; this answers "is upstream rewriting the
subsystem sideways?" -- the class of collision that merges cleanly and is still a semantic
decision (upstream's refactor vs our second path).  A block whose files were touched by 30
upstream commits is a different rung from one whose files nobody touched, even when neither has a
text conflict.

USAGE:  python3 archive/work/upstream-rebase-261120/probe-overlap.py [/path/to/llama.cpp]
Writes overlap-map.json beside itself, prints a per-block table + the upstream commits per block.
"""
import json
import pathlib
import re
import subprocess
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path.home() / "llama.cpp")
DELIVERY = pathlib.Path(__file__).resolve().parents[2]
rel = json.load(open(DELIVERY / "release.json"))
OLD_BASE, OLD_TIP = rel["base"], rel["tip"]
NEW_REF = "master"

# subsystems the delivery owns -> a regex over upstream commit subjects.  Used to flag "upstream
# is moving into our space" even when no file overlaps.
SPACE = {
    "FA / flash attention": r"\bfattn|flash.attn|\bFA\b|\bMMA\b.*attn|kv.*attn",
    "quantized KV": r"quantized (KV|k/v)|\bKV\b.*quant|iq4_nl",
    "BF16": r"bf16|bfloat16",
    "GDN / SSM / recurrent": r"\bGDN\b|gated.delta|SSM|recurrent|mamba",
    "MMQ / MMVQ / MMID / MMB": r"\bMMQ\b|\bMMVQ\b|\bMMID\b|\bMMB\b|mul_mat_q|mul_mat_id|few-row",
    "MoE / host-resident experts": r"\bMoE\b|expert|mul_mat_id|topk",
    "MTP / spec decode": r"\bMTP\b|speculat|draft|EAGLE|DFlash|DFLASH|nextn",
    "qwen4exp / QSA / indexer": r"qwen4exp|\bQSA\b|indexer|hyper.connection",
    "fusion": r"fus(e|ion|ed)",
    "graph budget / nodes": r"node budget|graph node|realloc",
    "server admission / slots": r"server.*(slot|sched|admission|prompt)|slot_sched|fair|park",
    "tensor split / all-reduce": r"tensor.split|all.reduce|split_state|-sm\b",
    "KV cache / streaming": r"kv.cache|kv.cell|k-pool|K/V",
    "KV window / residency": r"window|residen|evict|stream",
    "load mode / host tiers": r"load.mode|mlock|mmap|O_DIRECT|staging buffer|buffered I/O",
}


def g(*args, check=True):
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def main():
    blocks = []
    shas = g("log", "--reverse", "--format=%H", f"{OLD_BASE}..{OLD_TIP}").split()
    for i, c in enumerate(shas):
        subj = g("log", "-1", "--format=%s", c).strip()
        files = [f for f in g("diff", "--name-only", f"{c}^", c).split("\n") if f]
        blocks.append(dict(i=i, sha=c[:9], subj=subj, files=files))

    up = []
    for line in g("log", "--format=%H%x09%s", f"{OLD_BASE}..{NEW_REF}").split("\n"):
        if not line.strip():
            continue
        sha, subj = line.split("\t", 1)
        files = [f for f in g("diff", "--name-only", f"{sha}^", sha).split("\n") if f]
        up.append(dict(sha=sha[:9], subj=subj, files=files))

    up_by_file = {}
    for u in up:
        for f in u["files"]:
            up_by_file.setdefault(f, []).append(u["sha"])

    out = []
    for b in blocks:
        overlap = {f: up_by_file[f] for f in b["files"] if f in up_by_file}
        churn = sum(len(v) for v in overlap.values())
        # upstream commits that touch >=1 of this block's files
        commits = sorted({s for v in overlap.values() for s in v})
        out.append(dict(i=b["i"], sha=b["sha"], subj=b["subj"], nfiles=len(b["files"]),
                        n_shared=len(overlap), churn=churn, n_up_commits=len(commits),
                        top=sorted(overlap.items(), key=lambda kv: -len(kv[1]))[:12],
                        commits=commits))

    print(f"{'blk':>3} {'sha':9} {'files':>5} {'shared':>6} {'churn':>5} {'upcmt':>5}  subject")
    for r in out:
        print(f"{r['i']:3d} {r['sha']:9} {r['nfiles']:5d} {r['n_shared']:6d} {r['churn']:5d} "
              f"{r['n_up_commits']:5d}  {r['subj'][:60]}")
    print()
    for r in out:
        print(f"--- block {r['i']:02d} {r['sha']} :: {r['subj'][:70]}")
        for f, shas in r["top"]:
            print(f"    {len(shas):3d}  {f}")
    print()
    print("=== upstream commits whose SUBJECT enters a delivery space ===")
    for space, pat in SPACE.items():
        rx = re.compile(pat, re.I)
        hits = [u for u in up if rx.search(u["subj"])]
        if hits:
            print(f"\n## {space}  ({len(hits)} commits)")
            for u in hits:
                print(f"   {u['sha']} {u['subj'][:100]}")
    json.dump(out, open(pathlib.Path(__file__).with_name("overlap-map.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
