#!/usr/bin/env python3
"""Decisive test for the split proposal: is the multi-GPU prose/reasoning cost of
ref_d robust across prompts, or a p1 artifact?  2-GPU q8t2 + 3-GPU q8t2, p1-p4 + r1-r4."""
import os, re, subprocess, math

CORPUS = "/home/stew675/llama-cpp-rdna-boosts/archive/work/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/valid-logs"; os.makedirs(LOGD, exist_ok=True)
D8 = "/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
REFD = {"GGML_MTP_DROP_FLOOR": "250", "GGML_MTP_DROP_SLOPE": "40",
        "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"}

def parse(log):
    tps = ""
    for ln in open(log, errors="ignore"):
        if "eval time" in ln and "prompt eval time" not in ln:
            m = re.search(r"([\d.]+) tokens per second\)", ln)
            if m: tps = m.group(1)
    return tps

def run(tag, workers, axis, fname, envx):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
    env["HIP_VISIBLE_DEVICES"] = workers
    env.update(envx)
    log = f"{LOGD}/{tag}_{fname}.log"
    cmd = [BIN, "-m", D8, "--reasoning", "on" if axis == "reasoning" else "off",
           "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12",
           "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42", "--temp", "0",
           "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048", "-ub", "2048",
           "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4", "-sm", "tensor"]
    try:
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    return parse(log) if rc == 0 else ""

# known p1/r1 from tune12/tune15 (same build/flags)
KNOWN = {
 ("q8", "p1-prose.txt"):    (81.24, 77.26),
 ("q8", "r1-reasoning.txt"):(61.03, 60.12),
 ("g3", "p1-prose.txt"):    (94.31, 86.10),
 ("g3", "r1-reasoning.txt"):(69.73, 70.31),
}
PROMPTS = [("prose", f"p{i}-prose.txt") for i in range(1, 5)] + \
          [("reasoning", f"r{i}-reasoning.txt") for i in range(1, 5)]

for label, workers in (("q8 (2 GPU)", "1,2"), ("g3 (3 GPU)", "0,1,2")):
    print(f"\n=== {label}: base vs ref_d ===")
    print(f"{'prompt':22s} {'base':>8s} {'ref_d':>8s} {'ratio':>7s}")
    key = "q8" if label.startswith("q8") else "g3"
    rs = []
    for axis, fname in PROMPTS:
        if (key, fname) in KNOWN:
            b, d = KNOWN[(key, fname)]
            b = f"{b:.2f}"; d = f"{d:.2f}"
            rb, rd = float(b), float(d)
        else:
            rb = float(run(f"x_{key}_base", workers, axis, fname, {}) or 0)
            rd = float(run(f"x_{key}_refd", workers, axis, fname, REFD) or 0)
        r = rd/rb if rb and rd else 0
        rs.append(r)
        print(f"{fname:22s} {rb:8.2f} {rd:8.2f} {r:7.3f}", flush=True)
    print(f"  prose axis geo    = {math.exp(sum(math.log(x) for x in rs[:4])/4):.3f}")
    print(f"  reasoning axis geo= {math.exp(sum(math.log(x) for x in rs[4:])/4):.3f}")
    print(f"  combined geo      = {math.exp(sum(math.log(x) for x in rs)/len(rs)):.3f}")
print("tune16 done", flush=True)
