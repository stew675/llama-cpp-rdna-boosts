#!/usr/bin/env python3
"""Q4_K_XL single GPU, combo with n_match 42 (a multiple of both 6 and 7):
  c6s6nm42  and  c7s7nm42
vs the delivery default (mtp12 = base bucket, plain adaptive, cap 12) from /tmp/ground.tsv.
"""
import os, re, subprocess, math

CORPUS = "/home/stew675/llama-cpp-rdna-boosts/wip/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/dense-combo-logs"; os.makedirs(LOGD, exist_ok=True)
OUT = "/tmp/dense-combo.tsv"
MODEL = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"

ARMS = {
 "c6s6nm42": ("6", "42"),
 "c7s7nm42": ("7", "42"),
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

for arm, (cap, nm) in ARMS.items():
    for axis, fname in FILES:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
        env["HIP_VISIBLE_DEVICES"] = "0"
        log = f"{LOGD}/{arm}_{fname}.log"
        cmd = [BIN, "-m", MODEL, "--reasoning", "on" if axis == "reasoning" else "off",
               "--spec-type", "draft-mtp-adaptive,ngram-mod", "--spec-ngram-mod-n-match", nm,
               "--spec-draft-n-max", cap, "--spec-draft-n-start", cap,
               "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42", "--temp", "0",
               "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048", "-ub", "2048",
               "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4"]
        try:
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        tps, tok, acc, ml = parse(log) if rc == 0 else ("", "", "", "")
        open(OUT, "a").write(f"{axis}\t{fname}\t{arm}\t{tps}\t{tok}\t{acc}\t{ml}\n")
        print(f"{arm:9s} {axis:9s} {fname:20s} rc={rc} {tps:>7} t/s tok={tok:>5} acc={acc:>8} ml={ml}", flush=True)

base = {}
for ln in open("/tmp/ground.tsv"):
    if ln.startswith("axis"): continue
    a, p, arm, tps, tok, acc, ml = ln.rstrip("\n").split("\t")
    if arm == "bucket": base[p] = float(tps)
d = {}
for ln in open(OUT):
    if ln.startswith("axis"): continue
    a, p, arm, tps, tok, acc, ml = ln.rstrip("\n").split("\t")
    if tps: d[(a, p, arm)] = (float(tps), acc, ml)
print("\n=== dense Q4_K_XL 1 GPU: combo vs delivery default (mtp12) ===")
print(f"{'axis':10s} {'prompt':20s} {'mtp12':>8s} " + " ".join(f"{a:>18s}" for a in ARMS))
for axis, fname in FILES:
    b = base.get(fname, 0)
    row = []
    for arm in ARMS:
        if (axis, fname, arm) in d:
            t = d[(axis, fname, arm)][0]
            row.append(f"{t:8.2f}({t/b:.3f})" if b else f"{t:8.2f}")
        else: row.append(" " * 18)
    print(f"{axis:10s} {fname:20s} {b:8.2f} " + " ".join(f"{x:>18s}" for x in row))
print(f"\n{'AXIS GEO vs mtp12':30s} " + " ".join(f"{a:>10s}" for a in ARMS))
for axis in ("reasoning", "prose", "code", "recall"):
    row = []
    for arm in ARMS:
        rs = [d[(a, f, arm)][0]/base[f] for (a, f, _) in d if a == axis and f in base and (axis, f, arm) in d]
        row.append(f"{math.exp(sum(math.log(x) for x in rs)/len(rs)):10.3f}" if rs else " " * 10)
    print(f"{axis:30s} " + " ".join(row))
print("dense combo done", flush=True)
