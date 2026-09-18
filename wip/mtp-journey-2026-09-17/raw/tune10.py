#!/usr/bin/env python3
"""Full 16-prompt validation of the leading controllers (dense, 1 GPU, cap 12).
Guards against repeating the single-prompt-per-axis overfit."""
import os, re, subprocess, math

MODEL = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
CORPUS = "/home/stew675/llama-cpp-rdna-boosts/wip/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/valid-logs"; os.makedirs(LOGD, exist_ok=True)
OUT = "/tmp/target-valid.tsv"

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

T = {"GGML_ADAPTIVE_TARGET": "1", "GGML_MTP_TARGET_STRIDE": "4"}
CANDS = {
 "ref_d":   {"GGML_MTP_DROP_FLOOR": "250", "GGML_MTP_DROP_SLOPE": "40",
             "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"},
 "t50w32":  dict(T, GGML_MTP_TARGET_PCT="50", GGML_MTP_TARGET_WINDOW="32"),
 "t55w64":  dict(T, GGML_MTP_TARGET_PCT="55", GGML_MTP_TARGET_WINDOW="64"),
}

if not os.path.exists(OUT):
    open(OUT, "w").write("axis\tprompt\tarm\ttps\ttok\tacc\tmeanlen\n")

for cn, envx in CANDS.items():
    for axis, fname in FILES:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
        env["HIP_VISIBLE_DEVICES"] = "0"
        env.update(envx)
        log = f"{LOGD}/{cn}_{fname}.log"
        cmd = [BIN, "-m", MODEL, "--reasoning", "on" if axis == "reasoning" else "off",
               "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12",
               "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42", "--temp", "0",
               "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048", "-ub", "2048",
               "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4"]
        try:
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        tps, tok, acc, ml = parse(log) if rc == 0 else ("", "", "", "")
        open(OUT, "a").write(f"{axis}\t{fname}\t{cn}\t{tps}\t{tok}\t{acc}\t{ml}\n")
        print(f"{cn:8s} {axis:9s} {fname:20s} rc={rc} {tps:>7} t/s acc={acc:>8} ml={ml}", flush=True)

# summary vs table
table = {}
for ln in open("/tmp/ground.tsv"):
    if ln.startswith("axis"): continue
    a, p, arm, tps, tok, acc, ml = ln.rstrip("\n").split("\t")
    if arm == "table": table[p] = float(tps)
data = {}
for ln in open(OUT):
    if ln.startswith("axis"): continue
    a, p, arm, tps, tok, acc, ml = ln.rstrip("\n").split("\t")
    if tps: data[(a, p, arm)] = float(tps)
print("\n=== summary (ratio vs table, geometric mean per axis) ===")
print(f"{'cand':8s} {'overall':>8s} {'worst-axis':>10s}  " + " ".join(f"{x:>9s}" for x in ("reasoning","prose","code","recall")))
for cn in CANDS:
    axis_geo = {}
    for axis in ("reasoning","prose","code","recall"):
        rs = [data[(a,p,cn)]/table[p] for (a,p,c) in data if a==axis and c==cn and (a,p,"table") not in data]
        rs = [data[(a,p,cn)]/table[p] for (a,p,c) in data if a==axis and c==cn]
        axis_geo[axis] = math.exp(sum(math.log(r) for r in rs)/len(rs)) if rs else 0
    ov = math.exp(sum(math.log(v) for v in axis_geo.values())/4)
    print(f"{cn:8s} {ov:8.3f} {min(axis_geo.values()):10.3f}  " + " ".join(f"{axis_geo[x]:9.3f}" for x in ("reasoning","prose","code","recall")), flush=True)
print("valid done", flush=True)
