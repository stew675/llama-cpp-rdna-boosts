#!/usr/bin/env python3
"""The earlier-derived ngram-mod + adaptive-MTP optimum (c9 s9 nm45) on the 4x4 corpus,
Q8_0 2-GPU tensor.  Arms: mtp12 (delivery default), c9s9 (same cap/start, no ngram),
c9s9nm45 (the combo).  c9s9 isolates the ngram contribution from the cap/start change."""
import os, re, subprocess, math

CORPUS = "/home/stew675/llama-cpp-rdna-boosts/archive/work/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/ngram-logs"; os.makedirs(LOGD, exist_ok=True)
OUT = "/tmp/ngram-q8t2.tsv"
MODEL = "/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
BASE = ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12"]
ARMS = {
 "mtp12":     BASE,
 "c9s9":      ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "9", "--spec-draft-n-start", "9"],
 "c9s9nm45":  ["--spec-type", "draft-mtp-adaptive,ngram-mod", "--spec-ngram-mod-n-match", "45",
               "--spec-draft-n-max", "9", "--spec-draft-n-start", "9"],
}
FILES = []
for axis, pre in (("reasoning", "r"), ("prose", "p"), ("code", "c"), ("recall", "k")):
    for i in range(1, 5):
        for f in sorted(os.listdir(CORPUS)):
            if f.startswith(f"{pre}{i}-"):
                FILES.append((axis, f))

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

if not os.path.exists(OUT):
    open(OUT, "w").write("axis\tprompt\tarm\ttps\ttok\tacc\tmeanlen\n")

for arm, spec in ARMS.items():
    for axis, fname in FILES:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
        env["HIP_VISIBLE_DEVICES"] = "1,2"
        log = f"{LOGD}/{arm}_{fname}.log"
        cmd = [BIN, "-m", MODEL, "--reasoning", "on" if axis == "reasoning" else "off",
               "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42", "--temp", "0",
               "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048", "-ub", "2048",
               "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4",
               "-sm", "tensor", "-ts", "1/1"] + spec
        try:
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        tps, tok, acc, ml = parse(log) if rc == 0 else ("", "", "", "")
        open(OUT, "a").write(f"{axis}\t{fname}\t{arm}\t{tps}\t{tok}\t{acc}\t{ml}\n")
        print(f"{arm:9s} {axis:9s} {fname:20s} rc={rc} {tps:>7} t/s tok={tok:>5} acc={acc:>8} ml={ml}", flush=True)

# summary
d = {}
for ln in open(OUT):
    if ln.startswith("axis"): continue
    a, p, arm, tps, tok, acc, ml = ln.rstrip("\n").split("\t")
    if tps: d[(a, p, arm)] = (float(tps), float(acc or 0), float(ml or 0))
print("\n=== per-axis t/s and ratio vs mtp12 ===")
print(f"{'axis':10s} {'prompt':20s} " + " ".join(f"{a:>16s}" for a in ARMS))
for axis, fname in FILES:
    row = []
    for arm in ARMS:
        if (axis, fname, arm) in d:
            t, ac, ml = d[(axis, fname, arm)]
            b = d.get((axis, fname, "mtp12"), (0,))[0]
            r = t/b if b else 0
            row.append(f"{t:8.2f}({r:.3f})")
        else: row.append(" " * 16)
    print(f"{axis:10s} {fname:20s} " + " ".join(f"{x:>16s}" for x in row))
print(f"\n{'AXIS GEO vs mtp12':30s} " + " ".join(f"{a:>10s}" for a in ARMS))
for axis in ("reasoning", "prose", "code", "recall"):
    row = []
    for arm in ARMS:
        rs = [d[(axis, f, arm)][0]/d[(axis, f, "mtp12")][0] for (a, f, _) in d if a == axis and (axis, f, arm) in d and d.get((axis, f, "mtp12"), (0,))[0]]
        row.append(f"{math.exp(sum(math.log(x) for x in rs)/len(rs)):10.3f}" if rs else " " * 10)
    print(f"{axis:30s} " + " ".join(row))
print("ngram done", flush=True)
