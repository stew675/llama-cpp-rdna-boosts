#!/usr/bin/env python3
"""Diagnostic: is the c4 deficit the drop (sinking) or the climb (can't recover)?"""
import os, re, subprocess
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

def run(label, axis, fname, env_extra):
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"; env["HIP_VISIBLE_DEVICES"] = "0"
    env.update(env_extra)
    log = f"{LOGD}/diag_{label}_{fname}.log"
    cmd = [BIN, "-m", MODEL, "--reasoning", "off", "--spec-type", "draft-mtp-adaptive",
           "--spec-draft-n-max", "12", "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42",
           "--temp", "0", "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048",
           "-ub", "2048", "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4"]
    try:
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    tps, tok, acc, ml = parse(log) if rc == 0 else ("", "", "", "")
    nt = sum(1 for _ in open(log, errors="ignore") if "adaptive draft depth" in _)
    return tps, ml, nt

CANDS = {
    "neverdrop":  {"GGML_MTP_DROP_FLOOR": "1000000", "GGML_MTP_DROP_SLOPE": "0"},   # hold at start (cap-3=9)
    "hold12":     {"GGML_MTP_DROP_FLOOR": "1000000", "GGML_MTP_DROP_SLOPE": "0", "GGML_MTP_COLD_START": "0"},
    "climbtop":   {"GGML_MTP_DROP_FLOOR": "1000000", "GGML_MTP_DROP_SLOPE": "0",
                   "GGML_MTP_CLIMB_BASE": "2", "GGML_MTP_CLIMB_SLOPE": "0"},        # climb to cap and hold
    "credit0":    {"GGML_MTP_CREDIT_MINUS": "0"},
}
for cn, envx in CANDS.items():
    row = []
    for axis, fname in (("code", "c4-code.txt"), ("recall", "k1-recall.txt")):
        tps, ml, nt = run(cn, axis, fname, envx)
        row.append(f"{fname.split('-')[0]} {tps:>7} ml={ml} tr={nt}")
    print(f"{cn:11s} " + " | ".join(row), flush=True)
print("diag done", flush=True)
