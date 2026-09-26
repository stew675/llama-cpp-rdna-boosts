#!/usr/bin/env python3
"""Controller A/B on ONE delivery build: bucket (default) vs table (GGML_ADAPTIVE_TABLE=1)."""
import os, re, subprocess, time
import importlib.util

spec = importlib.util.spec_from_file_location("m", "/tmp/matrix.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGDIR = "/tmp/ab-logs"; os.makedirs(LOGDIR, exist_ok=True)
SPECS = {
    "adaptive_n12": ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12"],
    "adaptive_n7":  ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "7"],
}
CELL_AXES = {
    "dense1": ["prose-rdna-boosts", "code-python", "recall"],
    "q8t2":   ["prose-rdna-boosts", "code-python"],
    "moe1":   ["code-python", "recall"],
}

def parse(log):
    tps = tok = acc = ml = ""
    for ln in open(log, errors="ignore"):
        if "eval time" in ln and "prompt eval time" not in ln:
            mm = re.search(r"/\s*(\d+) tokens\s+\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)", ln)
            if mm: tok, tps = mm.group(1), mm.group(2)
        if "draft acceptance" in ln:
            mm = re.search(r"draft acceptance = ([\d.]+).*?mean len =\s*([\d.]+)", ln)
            if mm: acc, ml = mm.groups()
    return tps, tok, acc, ml

for specname, specargs in SPECS.items():
    for cell, axes in CELL_AXES.items():
        c = m.CELLS[cell]
        for axis in axes:
            for mode in ("bucket", "table"):
                env = dict(os.environ)
                env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
                env["HIP_VISIBLE_DEVICES"] = c["hip"]
                if mode == "table":
                    env["GGML_ADAPTIVE_TABLE"] = "1"
                else:
                    env.pop("GGML_ADAPTIVE_TABLE", None)
                rea = "on" if axis == "reasoning" else "off"
                log = f"{LOGDIR}/{cell}_{mode}_{axis}_{specname}.log"
                cmd = [BIN, "-m", c["model"], "--reasoning", rea, *specargs, *c["extra"],
                       "-f", f"{m.PROMPTS}/{axis}.txt", "-n", "3000", "--seed", "42",
                       "--temp", "0", "--single-turn", "--no-display-prompt",
                       "-c", "32768", "-b", "2048", "-ub", "2048", "-ctk", "f16", "-ctv", "f16",
                       "-fa", "auto", "-ngl", "99", "-lv", "4"]
                try:
                    with open(log, "w") as fh:
                        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
                except subprocess.TimeoutExpired:
                    rc = "TIMEOUT"
                tps, tok, acc, ml = parse(log) if rc == 0 else ("", "", "", "")
                print(f"{cell:7s} {specname:13s} {mode:6s} {axis:18s} rc={rc} {tps:>7} t/s tok={tok:>5} acc={acc:>8} ml={ml}", flush=True)
print("ab done", flush=True)
