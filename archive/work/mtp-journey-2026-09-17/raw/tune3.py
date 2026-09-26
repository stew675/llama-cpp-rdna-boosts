#!/usr/bin/env python3
"""Sweep higher drop thresholds; must fix code without breaking reasoning/prose."""
import os, re, subprocess, math
MODEL = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
CORPUS = "/home/stew675/llama-cpp-rdna-boosts/archive/work/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/tune-logs"

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

def run(cn, axis, fname, envx):
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"; env["HIP_VISIBLE_DEVICES"] = "0"
    env.update(envx)
    log = f"{LOGD}/s3_{cn}_{fname}.log"
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
    return parse(log) if rc == 0 else ("", "", "", "")

table = {}
for ln in open("/tmp/ground.tsv"):
    if ln.startswith("axis"): continue
    a, p, arm, tps, tok, acc, ml = ln.rstrip("\n").split("\t")
    if arm == "table": table[p] = float(tps)

CANDS = {
 "base": {},
 "hi1":  {"GGML_MTP_DROP_FLOOR": "120", "GGML_MTP_DROP_SLOPE": "20"},
 "hi2":  {"GGML_MTP_DROP_FLOOR": "200", "GGML_MTP_DROP_SLOPE": "30"},
 "hi3":  {"GGML_MTP_DROP_FLOOR": "200", "GGML_MTP_DROP_SLOPE": "30",
          "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"},
 "hi4":  {"GGML_MTP_DROP_FLOOR": "300", "GGML_MTP_DROP_SLOPE": "0",
          "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"},
}
PROMPTS = [("reasoning", "r1-reasoning.txt"), ("prose", "p1-prose.txt"),
           ("code", "c3-code.txt"), ("code", "c4-code.txt"), ("recall", "k1-recall.txt")]
print(f"{'cand':6s} {'geo':>6s} {'worst':>6s}  " + " ".join(f"{p.split('-')[0]:>6s}" for _, p in PROMPTS))
for cn, envx in CANDS.items():
    rs = []
    cells = []
    for axis, fname in PROMPTS:
        tps, tok, acc, ml = run(cn, axis, fname, envx)
        r = float(tps) / table[fname] if tps else 0
        rs.append(r); cells.append(f"{r:6.2f}")
    geo = math.exp(sum(math.log(x) for x in rs) / len(rs))
    print(f"{cn:6s} {geo:6.3f} {min(rs):6.3f}  " + " ".join(cells), flush=True)
print("tune3 done", flush=True)
