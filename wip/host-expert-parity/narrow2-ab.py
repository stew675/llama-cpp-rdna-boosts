#!/usr/bin/env python3
"""Pull the TODO #279 A/B numbers out of the `narrow2-ab.sh` logs.

usage: narrow2-ab.py /tmp/narrow2-ab [--hash]

Prints one row per arm: the pin branch the run took (from the slab's own lines), the arena MiB, the cache
hit rate, the evict census and prefill/decode t/s.  `--hash` also hashes the generated text with
`scripts/extract-generated.py`, which is the text-identity half of the A/B.
"""
import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
EXTRACT = os.path.join(REPO, "scripts", "extract-generated.py")


def arm_row(path, do_hash):
    log = open(path, encoding="utf-8", errors="replace").read()
    branch = "-"
    if "stays on the cudaMalloc path (a" in log:
        branch = "malloc(pref)"
    elif "taken from the arena (tables evicted)" in log:
        branch = "evict"
    elif "unmapped VA spare" in log:
        branch = "spare"
    narrowed = re.findall(r"narrow-2 \(draft compute\) pinned.*?\(([\d.]+ GiB)", log)
    evicted = re.findall(r"evicted (\d+(?:\.\d+)?) MiB of arena tables", log)
    # the pin's own eviction is the FIRST one (the wide work view evicts later); keep the whole list too
    hit = re.findall(r"MoE arena = ([\d.]+)", log)
    arena = re.findall(r"arena ([\d.]+) MiB", log)
    pp = re.findall(r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second", log)
    tg = re.findall(r"(?<!prompt )eval time =\s+[\d.]+ ms /\s+(\d+) tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second", log)
    sha = ""
    if do_hash:
        out = subprocess.run([sys.executable, EXTRACT, path], capture_output=True, text=True)
        sha = out.stdout.strip() if out.returncode == 0 else "HASH-FAIL"
    return {
        "arm": os.path.basename(path).replace(".log", ""),
        "branch": branch,
        "narrow2": narrowed[0] if narrowed else "",
        "evict_first": evicted[0] if evicted else "0",
        "hit": hit[-1] if hit else "",
        "arena": arena[-1] if arena else "",
        "pp": pp[-1][1] if pp else "",
        "tg": tg[-1][1] if tg else "",
        "sha": sha,
    }


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/narrow2-ab"
    do_hash = "--hash" in sys.argv
    rows = [arm_row(p, do_hash) for p in sorted(glob.glob(os.path.join(d, "pref*-r*.log")))]
    if not rows:
        raise SystemExit("no arm logs in " + d)
    cols = ["arm", "branch", "narrow2", "evict_first", "arena", "hit", "pp", "tg"] + (["sha"] if do_hash else [])
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))
    for key, name in (("pp", "prefill"), ("tg", "decode")):
        vals = {}
        for r in rows:
            if r[key]:
                vals.setdefault(r["arm"].split("-r")[0], []).append(float(r[key]))
        if not vals:
            continue
        base = vals.get("pref0")
        for arm, v in sorted(vals.items()):
            mean = sum(v) / len(v)
            extra = ""
            if base and arm != "pref0":
                b = sum(base) / len(base)
                extra = f"  ({100.0 * (mean / b - 1.0):+.1f} % vs pref0)"
            print(f"{name:8s} {arm}: n={len(v)} mean={mean:.2f} values={[round(x, 2) for x in v]}{extra}")


if __name__ == "__main__":
    main()
