#!/usr/bin/env python3
"""Pull the PLAN P1.3 (`MOE_EXPERT_CACHE_TOUCH` 2 vs 3) A/B numbers out of the `touch-ab.sh` logs.

usage: touch-ab.py /tmp/touch-ab [--hash]

One row per arm: prefill/decode t/s, the arena hit rate, the sink census (demoted, MiB, reuse, declines) and
the access split (hits/fills/colds), which is where TOUCH is visible.  `--hash` also hashes the generated text
with `scripts/extract-generated.py` (run those arms with `VERBOSE=` -- the default verbosity -- because any
`-lv` interleaves the statistics block into llama-cli's generated text).
"""
import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
EXTRACT = os.path.join(REPO, "scripts", "extract-generated.py")


def num(pat, text, default=""):
    m = re.findall(pat, text)
    return m[-1] if m else default


def arm_row(path, do_hash):
    log = open(path, encoding="utf-8", errors="replace").read()
    sha = ""
    if do_hash:
        out = subprocess.run([sys.executable, EXTRACT, path], capture_output=True, text=True)
        sha = out.stdout.strip().split("sha=")[-1] if out.returncode == 0 else "HASH-FAIL"
    return {
        "arm": os.path.basename(path).replace(".log", ""),
        "pp": num(r"prompt eval time =\s+[\d.]+ ms /\s+\d+ tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second", log),
        "tg": num(r"(?<!prompt )eval time =\s+[\d.]+ ms /\s+\d+ tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second", log),
        "arena_h": num(r"MoE arena = ([\d.]+)", log),
        "cache_h": num(r"MoE expert cache: h=([\d.]+)", log),
        "arena": num(r"arena=([\d.]+) MiB total", log),
        "demoted": num(r"(\d+) demoted, [\d.]+ MiB", log),
        "demoted_mib": num(r"\d+ demoted, ([\d.]+) MiB", log),
        "declined": num(r"declined: (\d+) no-free-slot", log),
        "uses": num(r"reuse: (\d+) uses", log),
        "reuse_pct": num(r"\(([\d.]+) % used", log),
        "colds": num(r"colds=(\d+)", log),
        "cold_pct": num(r"cold=([\d.]+)%", log),
        "fills": num(r"fills=(\d+) evictions", log),
        "mirror": num(r"mirror tables \(full coverage -> exchange INERT, spare=0\): (\d+)", log),
        "sha": sha,
    }


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/touch-ab"
    do_hash = "--hash" in sys.argv
    rows = [arm_row(p, do_hash) for p in sorted(glob.glob(os.path.join(d, "lme*-t*-r*.log")))]
    if not rows:
        raise SystemExit("no arm logs in " + d)
    cols = ["arm", "pp", "tg", "arena_h", "cache_h", "arena", "demoted", "demoted_mib", "uses", "reuse_pct",
            "declined", "colds", "cold_pct", "fills", "mirror"] + (["sha"] if do_hash else [])
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))
    for key, name in (("pp", "prefill"), ("tg", "decode"), ("colds", "colds"), ("demoted", "demoted")):
        cells = {}
        for r in rows:
            if r[key]:
                cells.setdefault("-".join(r["arm"].split("-")[:2]), []).append(float(r[key]))
        for cell, v in sorted(cells.items()):
            print(f"{name:8s} {cell}: n={len(v)} mean={sum(v)/len(v):.1f} values={[round(x, 1) for x in v]}")


if __name__ == "__main__":
    main()
