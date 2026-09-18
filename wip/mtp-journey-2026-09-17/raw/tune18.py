#!/usr/bin/env python3
"""Two decisive tests:
 A) phase-switch robustness on 1 GPU (ps1/ps2/ps3, base vs ref_d)
 B) 2-GPU -sm layer (no cross-device AR): does ref_d's multi-GPU prose cost
    come from the tensor split / AR, or from multi-GPU itself?
"""
import os, re, subprocess

CORPUS = "/home/stew675/llama-cpp-rdna-boosts/wip/mtp-journey-2026-09-17/corpus"
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

print("=== 1-GPU Q8_0 (the quant used on every multi-GPU cell): base vs ref_d ===")
print(f"{'prompt':22s} {'base':>8s} {'ref_d':>8s} {'ratio':>7s}   [2-GPU tensor / 2-GPU layer / 1-GPU Q4_K_XL]")
REF = {"p1-prose.txt": "0.951 / 0.957 / 1.174", "p2-prose.txt": "0.900 / 0.943 / 1.062",
       "p3-prose.txt": "0.942 /   --  / 1.022", "c1-code.txt": "1.013 /   --  / 1.010"}
for axis, fname in (("prose","p1-prose.txt"),("prose","p2-prose.txt"),("prose","p3-prose.txt"),("code","c1-code.txt")):
    b = run("q1_base", D8, "0", [], axis, fname, {})
    d = run("q1_refd", D8, "0", [], axis, fname, REFD)
    r = float(d[0])/float(b[0]) if b[0] and d[0] else 0
    print(f"{fname:22s} {b[0]:>8} {d[0]:>8} {r:7.3f}   [{REF.get(fname,'')}]", flush=True)
print("tune18 done", flush=True)
