#!/usr/bin/env python3
"""Phase 2: pinned oracle on c4/p1, then a first bucket-constant candidate sweep."""
import os, re, subprocess
from collections import defaultdict

MODEL = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
CORPUS = "/home/stew675/llama-cpp-rdna-boosts/wip/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/tune-logs"; os.makedirs(LOGD, exist_ok=True)

def parse(log):
    tps = tok = acc = ml = ""
    for ln in open(log, errors="ignore"):
        if "eval time" in ln and "prompt eval time" not in ln:
            m = re.search(r"/\s*(\d+) tokens\s+\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)", ln)
            if m: tok, tps = m.group(1), m.group(2)
        if "draft acceptance" in ln:
            m = re.search(r"draft acceptance = ([\d.]+).*?mean len =\s*([\d.]+)", ln)
            if m: acc, ml = m.groups()
    return tps, tok, acc, ml

def run(label, axis, fname, specargs, env_extra):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
    env["HIP_VISIBLE_DEVICES"] = "0"
    env.update(env_extra)
    log = f"{LOGD}/{label}_{fname}.log"
    cmd = [BIN, "-m", MODEL, "--reasoning", "on" if axis == "reasoning" else "off",
           *specargs, "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42", "--temp", "0",
           "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048", "-ub", "2048",
           "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4"]
    try:
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    return parse(log) if rc == 0 else ("", "", "", "")

# ---- pinned oracle (bucket build, depth forced; controller bypassed)
print("== pinned oracle ==")
print(f"{'prompt':18s} {'D':>2s} {'tps':>7s} {'ml':>6s}")
oracle = defaultdict(dict)
for fname, axis in (("c4-code.txt", "code"), ("p1-prose.txt", "prose")):
    for D in (6, 8, 10, 12):
        tps, tok, acc, ml = run(f"pin{D}", axis, fname,
            ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-min-adaptive", str(D),
             "--spec-draft-n-max", str(D)], {})
        oracle[fname][D] = tps
        print(f"{fname:18s} {D:2d} {tps:>7} {ml:>6}", flush=True)

# ---- candidate sweep
CANDS = {
    "base":       {},
    "drop_tbl":   {"GGML_MTP_DROP_FLOOR": "20", "GGML_MTP_DROP_SLOPE": "5"},
    "drop_mid":   {"GGML_MTP_DROP_FLOOR": "40", "GGML_MTP_DROP_SLOPE": "7"},
    "climb_fast": {"GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"},
    "cold0":      {"GGML_MTP_COLD_START": "0"},
    "combo1":     {"GGML_MTP_DROP_FLOOR": "30", "GGML_MTP_DROP_SLOPE": "5",
                   "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"},
}
PROMPTS = [("code", "c1-code.txt"), ("code", "c2-code.txt"), ("code", "c3-code.txt"),
           ("code", "c4-code.txt"), ("recall", "k1-recall.txt"), ("prose", "p1-prose.txt")]
# table ground state from /tmp/ground.tsv
table = {}
for ln in open("/tmp/ground.tsv"):
    if ln.startswith("axis"): continue
    a, p, arm, tps, tok, acc, ml = ln.rstrip("\n").split("\t")
    if arm == "table": table[p] = float(tps)

print("\n== candidate sweep (ratio vs table) ==")
res = defaultdict(dict)
for cname, envx in CANDS.items():
    ratios = []
    for axis, fname in PROMPTS:
        tps, tok, acc, ml = run(f"{cname}", axis, fname,
            ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12"], envx)
        if tps:
            r = float(tps) / table[fname]
            res[cname][fname] = (float(tps), r, ml)
            ratios.append(r)
    import math
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 0
    worst = min(ratios) if ratios else 0
    print(f"{cname:11s} geo={geo:.3f} worst={worst:.3f}  " +
          " ".join(f"{p.split('-')[0]}={res[cname][p][1]:.2f}" for _, p in PROMPTS), flush=True)
print("tune1 done", flush=True)
