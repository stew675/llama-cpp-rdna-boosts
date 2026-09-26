#!/usr/bin/env python3
"""Three-way MTP journey harness: stock | upstream+PR#27210 | current delivery.

Runs llama-cli one process at a time (no parallel benches), parses the generation
eval line and the draft-acceptance line, and appends to a TSV.
"""
import os, re, subprocess, sys, time

REPO = "/home/stew675/llama-cpp-rdna-boosts"
PROMPTS = f"{REPO}/prompts"
ROCM = "/opt/rocm-7.14-gfx1201"
OUT = "/tmp/journey.tsv"
LOGDIR = "/tmp/journey-logs"
os.makedirs(LOGDIR, exist_ok=True)

ARMS = {
    "stock": "/home/stew675/stock-9113/build-rocm/bin/llama-cli",
    "pr":    "/home/stew675/pr27210/build-rocm/bin/llama-cli",
    "deliv": "/home/stew675/llama.cpp/build-rocm-current/bin/llama-cli",
}

SPECS = {
    "plain":       ["--spec-type", "none"],
    "static_n3":   ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
    "adaptive_n7": ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "7"],
    "adaptive_n12":["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12"],
}
ARM_SPECS = {
    "stock": ["plain", "static_n3"],
    "pr":    ["plain", "static_n3", "adaptive_n7", "adaptive_n12"],
    "deliv": ["plain", "static_n3", "adaptive_n7", "adaptive_n12"],
}

CELLS = {
    # dense 27B UD-Q4_K_XL, one GPU (the delivery's documented reference cell)
    "dense1": dict(
        model="/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        hip="0", extra=[],
        axes=["reasoning", "prose-rdna-boosts", "code-python", "recall"],
    ),
    # MoE 35B-A3B UD-Q4_K_M, one GPU (acceptance + verify-width cell)
    "moe1": dict(
        model="/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        hip="0", extra=[],
        axes=["reasoning", "prose-rdna-boosts", "code-python", "recall"],
    ),
    # dense 27B Q8_0, two-GPU tensor (the controller-tuning cell, issue #35)
    "q8t2": dict(
        model="/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf",
        hip="1,2", extra=["-sm", "tensor", "-ts", "1/1"],
        axes=["code-python", "prose-rdna-boosts"],
    ),
}

GEN_RE = re.compile(r"/\s*(\d+) tokens\s+\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)")
ACC_RE = re.compile(r"draft acceptance = ([\d.]+)\s*\(\s*(\d+) accepted\s*/\s*(\d+) generated\),\s*mean len =\s*([\d.]+)")


def parse(log):
    tps = tok = acc = acn = gen = ml = ""
    for ln in open(log, errors="ignore"):
        if "eval time" in ln and "prompt eval time" not in ln:
            m = GEN_RE.search(ln)
            if m:
                tok, tps = m.group(1), m.group(2)
        if "draft acceptance" in ln:
            m = ACC_RE.search(ln)
            if m:
                acc, acn, gen, ml = m.groups()
    return tps, tok, acc, acn, gen, ml


def run(cell, cellname, arm, axis, spec):
    c = CELLS[cellname]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{ROCM}/lib"
    env["HIP_VISIBLE_DEVICES"] = c["hip"]
    rea = "on" if axis == "reasoning" else "off"
    log = f"{LOGDIR}/{cellname}_{arm}_{axis}_{spec}.log"
    cmd = [ARMS[arm], "-m", c["model"], "--reasoning", rea, *SPECS[spec], *c["extra"],
           "-f", f"{PROMPTS}/{axis}.txt",
           "-n", "3000", "--seed", "42", "--temp", "0", "--single-turn", "--no-display-prompt",
           "-c", "32768", "-b", "2048", "-ub", "2048", "-ctk", "f16", "-ctv", "f16",
           "-fa", "auto", "-ngl", "99", "-lv", "4"]
    t0 = time.time()
    try:
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    dt = time.time() - t0
    tps, tok, acc, acn, gen, ml = parse(log) if rc == 0 else ("", "", "", "", "", "")
    row = f"{cellname}\t{arm}\t{axis}\t{spec}\t{rc}\t{tps}\t{tok}\t{acc}\t{ml}\t{acn}\t{gen}\t{dt:.1f}\n"
    with open(OUT, "a") as fh:
        fh.write(row)
    print(f"  {cellname} {arm:5s} {axis:18s} {spec:12s} rc={rc} {tps:>7} t/s tok={tok:>5} acc={acc:>8} ml={ml}", flush=True)


if __name__ == "__main__":
    want = sys.argv[1:] or list(CELLS)
    if not os.path.exists(OUT):
        with open(OUT, "w") as fh:
            fh.write("cell\tarm\taxis\tspec\trc\ttps\ttok\tacc\tmeanlen\taccepted\tgen\tdt\n")
    for cellname in want:
        c = CELLS[cellname]
        print(f"=== cell {cellname} ({os.path.basename(c['model'])}, hip={c['hip']}) ===", flush=True)
        for axis in c["axes"]:
            for arm in ARMS:
                for spec in ARM_SPECS[arm]:
                    run(c, cellname, arm, axis, spec)
    print("done", flush=True)
