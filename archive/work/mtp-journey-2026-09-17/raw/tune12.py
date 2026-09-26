#!/usr/bin/env python3
"""Risk cells: 2-GPU q8t2 (base vs ref_d) + dense phase-switch guard."""
import os, re, subprocess, math

CORPUS = "/home/stew675/llama-cpp-rdna-boosts/archive/work/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/valid-logs"; os.makedirs(LOGD, exist_ok=True)

D8  = "/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
D4K = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"

REFD = {"GGML_MTP_DROP_FLOOR": "250", "GGML_MTP_DROP_SLOPE": "40",
        "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"}

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

def run(tag, model, workers, extra, axis, fname, envx):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
    env["HIP_VISIBLE_DEVICES"] = workers
    env.update(envx)
    log = f"{LOGD}/{tag}_{fname}.log"
    cmd = [BIN, "-m", model, "--reasoning", "on" if axis == "reasoning" else "off",
           "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12",
           "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42", "--temp", "0",
           "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048", "-ub", "2048",
           "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4"] + extra
    try:
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    return (parse(log) if rc == 0 else ("", "", "", "")) + (rc,)

Q8T2_EXTRA = ["-sm", "tensor", "-ts", "1/1"]
Q8T2 = [("reasoning", "r1-reasoning.txt"), ("prose", "p1-prose.txt"),
        ("code", "c1-code.txt"), ("recall", "k1-recall.txt")]

print("=== 2-GPU q8t2: base vs ref_d (ratio ref_d/base) ===")
print(f"{'prompt':20s} {'base':>8s} {'ref_d':>8s} {'ratio':>7s}")
for axis, fname in Q8T2:
    b = run("q8_base", D8, "1,2", Q8T2_EXTRA, axis, fname, {})
    d = run("q8_refd", D8, "1,2", Q8T2_EXTRA, axis, fname, REFD)
    r = float(d[0]) / float(b[0]) if b[0] and d[0] else 0
    print(f"{fname:20s} {b[0]:>8} {d[0]:>8} {r:>7.3f}   (base acc={b[2]} ml={b[3]}; ref_d acc={d[2]} ml={d[3]})", flush=True)

print("\n=== dense phase-switch guard (1 GPU, reasoning off) ===")
for arm, envx in (("base", {}), ("ref_d", REFD)):
    tps, tok, acc, ml, rc = run(f"ps_{arm}", D4K, "0", [], "code", "phase-switch.txt", envx)
    print(f"{arm:6s} rc={rc} {tps:>7} t/s acc={acc:>8} ml={ml}", flush=True)
print("tune12 done", flush=True)
