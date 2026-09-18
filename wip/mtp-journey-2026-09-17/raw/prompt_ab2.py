#!/usr/bin/env python3
"""Prose-prompt sensitivity: bucket vs table on ONE delivery build, plus the PR arm."""
import os, re, subprocess, time

MODEL = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
PROMPTS = {
    "wk5":   "/tmp/prose_p5.txt",
    "wk6":   "/tmp/prose_p6.txt",
    "wk7":   "/tmp/prose_p7.txt",
    "wk8":   "/tmp/prose_p8.txt",
}
ARMS = {
    # (pr dropped for the wider pass)
    "bucket": ("/home/stew675/deliv-ab/build-rocm/bin/llama-cli", None),
    "table":  ("/home/stew675/deliv-ab/build-rocm/bin/llama-cli", "1"),
}
SPEC = ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12"]
LOGD = "/tmp/prose-ab-logs"; os.makedirs(LOGD, exist_ok=True)

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

print(f"{'prompt':6s} {'arm':7s} {'tps':>7s} {'tok':>5s} {'acc':>8s} {'meanlen':>7s}")
for pname, ppath in PROMPTS.items():
    for aname, (binp, envflag) in ARMS.items():
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-7.14-gfx1201/lib"
        env["HIP_VISIBLE_DEVICES"] = "0"
        if envflag: env["GGML_ADAPTIVE_TABLE"] = envflag
        else: env.pop("GGML_ADAPTIVE_TABLE", None)
        log = f"{LOGD}/{pname}_{aname}.log"
        cmd = [binp, "-m", MODEL, "--reasoning", "off", *SPEC, "-f", ppath,
               "-n", "3000", "--seed", "42", "--temp", "0", "--single-turn", "--no-display-prompt",
               "-c", "32768", "-b", "2048", "-ub", "2048", "-ctk", "f16", "-ctv", "f16",
               "-fa", "auto", "-ngl", "99", "-lv", "4"]
        try:
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        tps, tok, acc, ml = parse(log) if rc == 0 else ("", "", "", "")
        print(f"{pname:6s} {aname:7s} {tps:>7s} {tok:>5s} {acc:>8s} {ml:>7s}", flush=True)
print("prompt-ab done", flush=True)
