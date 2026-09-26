#!/usr/bin/env python3
"""Two decisive tests:
 A) phase-switch robustness on 1 GPU (ps1/ps2/ps3, base vs ref_d)
 B) 2-GPU -sm layer (no cross-device AR): does ref_d's multi-GPU prose cost
    come from the tensor split / AR, or from multi-GPU itself?
"""
import os, re, subprocess

CORPUS = "/home/stew675/llama-cpp-rdna-boosts/archive/work/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/valid-logs"; os.makedirs(LOGD, exist_ok=True)
D4K = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
D8  = "/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
REFD = {"GGML_MTP_DROP_FLOOR": "250", "GGML_MTP_DROP_SLOPE": "40",
        "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"}

def parse(log):
    tps = acc = ml = ""
    for ln in open(log, errors="ignore"):
        if "eval time" in ln and "prompt eval time" not in ln:
            m = re.search(r"([\d.]+) tokens per second\)", ln)
            if m: tps = m.group(1)
        if "draft acceptance" in ln:
            m = re.search(r"draft acceptance = ([\d.]+).*?mean len =\s*([\d.]+)", ln)
            if m: acc, ml = m.groups()
    return tps, acc, ml

def run(tag, model, workers, split, axis, fname, envx):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
    env["HIP_VISIBLE_DEVICES"] = workers
    env.update(envx)
    log = f"{LOGD}/{tag}_{fname}.log"
    cmd = [BIN, "-m", model, "--reasoning", "on" if axis == "reasoning" else "off",
           "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12",
           "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42", "--temp", "0",
           "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048", "-ub", "2048",
           "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4"] + split
    try:
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    return (parse(log) if rc == 0 else ("", "", "")) + (rc,)

print("=== A) 1-GPU phase-switch robustness (dense Q4_K_XL) ===")
print(f"{'prompt':22s} {'base':>8s} {'ref_d':>8s} {'ratio':>7s}")
for fname in ("phase-switch.txt", "ps2-phase.txt", "ps3-phase.txt"):
    b = run("psA_base", D4K, "0", [], "code", fname, {})
    d = run("psA_refd", D4K, "0", [], "code", fname, REFD)
    r = float(d[0])/float(b[0]) if b[0] and d[0] else 0
    print(f"{fname:22s} {b[0]:>8} {d[0]:>8} {r:7.3f}   (base acc={b[1]} ml={b[2]}; ref_d acc={d[1]} ml={d[2]})", flush=True)

print("\n=== B) 2-GPU -sm layer (no AR) vs the tensor result, prose ===")
print(f"{'prompt':22s} {'layer base':>10s} {'layer ref_d':>11s} {'ratio':>7s}   [tensor ref_d/base]")
TENSOR = {"p1-prose.txt": 0.951, "p2-prose.txt": 0.900}
for fname in ("p1-prose.txt", "p2-prose.txt"):
    b = run("lay_base", D8, "1,2", ["-sm", "layer"], "prose", fname, {})
    d = run("lay_refd", D8, "1,2", ["-sm", "layer"], "prose", fname, REFD)
    r = float(d[0])/float(b[0]) if b[0] and d[0] else 0
    print(f"{fname:22s} {b[0]:>10} {d[0]:>11} {r:7.3f}   [{TENSOR[fname]:.3f}]", flush=True)
print("tune17 done", flush=True)
