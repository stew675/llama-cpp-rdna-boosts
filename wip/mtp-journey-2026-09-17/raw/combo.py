#!/usr/bin/env python3
"""Does the bucket's drop/climb optimum move when ngram-mod covers recall?

All arms are the combo (`--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 45`)
with different MTP cap/start and/or bucket constants.  Q8_0 2-GPU tensor, 4x4 corpus.
Reference rows (base constants, cap 9/start 9) are already in /tmp/ngram-q8t2.tsv as c9s9nm45.

usage: combo.py <group>   with group in {consts, caps, all}
"""
import os, re, subprocess, sys, math

CORPUS = "/home/stew675/llama-cpp-rdna-boosts/wip/mtp-journey-2026-09-17/corpus"
BIN = "/home/stew675/deliv-ab/build-rocm/bin/llama-cli"
LOGD = "/tmp/combo-logs"; os.makedirs(LOGD, exist_ok=True)
OUT = "/tmp/combo-q8t2.tsv"
MODEL = "/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"

# (cap, start, env constants)
CANDS = {
 "c9_base": (9, 9, {}),
 "c9_refd": (9, 9, {"GGML_MTP_DROP_FLOOR": "250", "GGML_MTP_DROP_SLOPE": "40",
                    "GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"}),
 "c9_climb": (9, 9, {"GGML_MTP_CLIMB_BASE": "10", "GGML_MTP_CLIMB_SLOPE": "3"}),
 "c9_hi2":  (9, 9, {"GGML_MTP_DROP_FLOOR": "200", "GGML_MTP_DROP_SLOPE": "30"}),
 "c8_base": (8, 8, {}),
 "c7_base": (7, 7, {}),
 "c6_base": (6, 6, {}),
 "c5_base": (5, 5, {}),
 "c4_base": (4, 4, {}),
 "c3_base": (3, 3, {}),
 "c9_s6": (9, 6, {}),
 "c9_s3": (9, 3, {}),
}
GROUP = {"consts": ["c9_refd", "c9_climb", "c9_hi2"],
         "caps": ["c8_base", "c7_base", "c6_base"],
         "caps2": ["c5_base", "c4_base", "c3_base"],
         "start": ["c9_s6", "c9_s3"],
         "all": list(CANDS)}

PROMPTS = [("reasoning", f"r{i}-reasoning.txt") for i in range(1, 5)] + \
          [("prose", f"p{i}-prose.txt") for i in range(1, 5)] + \
          [("code", f"c{i}-code.txt") for i in range(1, 5)] + \
          [("recall", "k1-recall.txt")]

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

for arm in GROUP[sys.argv[1]]:
    cap, start, envx = CANDS[arm]
    for axis, fname in PROMPTS:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
        env["HIP_VISIBLE_DEVICES"] = "1,2"
        env.update(envx)
        log = f"{LOGD}/{arm}_{fname}.log"
        cmd = [BIN, "-m", MODEL, "--reasoning", "on" if axis == "reasoning" else "off",
               "--spec-type", "draft-mtp-adaptive,ngram-mod", "--spec-ngram-mod-n-match", "45",
               "--spec-draft-n-max", str(cap), "--spec-draft-n-start", str(start),
               "-f", f"{CORPUS}/{fname}", "-n", "3000", "--seed", "42", "--temp", "0",
               "--single-turn", "--no-display-prompt", "-c", "32768", "-b", "2048", "-ub", "2048",
               "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "-ngl", "99", "-lv", "4",
               "-sm", "tensor", "-ts", "1/1"]
        try:
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        tps, tok, acc, ml = parse(log) if rc == 0 else ("", "", "", "")
        open(OUT, "a").write(f"{axis}\t{fname}\t{arm}\t{tps}\t{tok}\t{acc}\t{ml}\n")
        print(f"{arm:8s} {axis:9s} {fname:20s} rc={rc} {tps:>7} t/s acc={acc:>8} ml={ml}", flush=True)
print("combo group done", flush=True)
