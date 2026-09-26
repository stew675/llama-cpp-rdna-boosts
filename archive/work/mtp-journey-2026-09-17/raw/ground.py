#!/usr/bin/env python3
"""Ground state: table vs current bucket across the 4x4 corpus, 1 GPU, adaptive cap 12."""
import os, re, subprocess, time

MODEL = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
CORPUS = "/home/stew675/llama-cpp-rdna-boosts/archive/work/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/ground-logs"; os.makedirs(LOGD, exist_ok=True)
OUT = "/tmp/ground.tsv"

FILES = []
for axis, pre in (("reasoning", "r"), ("prose", "p"), ("code", "c"), ("recall", "k")):
    for i in range(1, 5):
        for f in os.listdir(CORPUS):
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

for axis, fname in FILES:
    for arm, tbl in (("table", "1"), ("bucket", None)):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
        env["HIP_VISIBLE_DEVICES"] = "0"
        if tbl: env["GGML_ADAPTIVE_TABLE"] = tbl
        else: env.pop("GGML_ADAPTIVE_TABLE", None)
        log = f"{LOGD}/{arm}_{fname}.log"
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
        open(OUT, "a").write(f"{axis}\t{fname}\t{arm}\t{tps}\t{tok}\t{acc}\t{ml}\n")
        print(f"{axis:9s} {fname:20s} {arm:6s} rc={rc} {tps:>7} t/s tok={tok:>5} acc={acc:>8} ml={ml}", flush=True)
print("ground done", flush=True)
