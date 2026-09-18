#!/usr/bin/env python3
"""Conservative retunes (drop floor only) on the risk cells that ref_d regressed."""
import os, re, subprocess

CORPUS = "/home/stew675/llama-cpp-rdna-boosts/wip/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/valid-logs"; os.makedirs(LOGD, exist_ok=True)
D8  = "/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
D4K = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
MOE = "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

CANDS = {
 "hi2": {"GGML_MTP_DROP_FLOOR": "200", "GGML_MTP_DROP_SLOPE": "30"},
 "hi3": {"GGML_MTP_DROP_FLOOR": "200", "GGML_MTP_DROP_SLOPE": "30",
         "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"},
}
# (label, model, workers, extra, axis, fname, baseline)
CELLS = [
 ("dense-ps", D4K, "0", [], "code", "phase-switch.txt", 49.03),
 ("q8t2-p1",  D8,  "1,2", ["-sm","tensor","-ts","1/1"], "prose", "p1-prose.txt", 81.24),
 ("q8t2-c1",  D8,  "1,2", ["-sm","tensor","-ts","1/1"], "code",  "c1-code.txt", 96.10),
 ("moe1-p1",  MOE, "0", [], "prose", "p1-prose.txt", 149.00),
 ("moe1-c1",  MOE, "0", [], "code",  "c1-code.txt", 166.58),
]

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
    return (parse(log) if rc == 0 else ("", "", "")) + (rc,)

print(f"{'cell':10s} {'base':>8s} " + " ".join(f"{c:>14s}" for c in CANDS))
for label, model, workers, extra, axis, fname, base in CELLS:
    row = []
    for cn, envx in CANDS.items():
        tps, acc, ml, rc = run(f"{cn}_{label}", model, workers, extra, axis, fname, envx)
        r = float(tps)/base if tps else 0
        row.append(f"{tps:>7} ({r:.3f})" if tps else "   FAIL")
    print(f"{label:10s} {base:8.2f} " + " ".join(f"{x:>14s}" for x in row), flush=True)
print("tune13 done", flush=True)
