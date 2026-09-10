#!/usr/bin/env python3
"""Issue #25 repro harness: greedy output vs MTP verify batch width (gfx1151).

Faithful to the reporter's server driver:
  - Qwen3.6-27B Q8_0 (dense qwen35), f16 KV, flash-attn auto, greedy temp 0 seed 42
  - one warmup per arm, then one request per prompt, sha1 of reasoning_content+content
Arms are selected by name on the command line; a fresh server is started per arm
(spec flags are startup-only).

Usage:  repro.py <arm> [prompt-index ...]
        repro.py matrix        # all arms x all prompts
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = "/home/stew675/llama-cpp-rdna-boosts"
OUTROOT = os.environ.get("OUTROOT", os.path.join(ROOT, "wip/strix-halo/issue25/runs"))
BIN = os.environ.get("LLAMA_BIN", "/home/stew675/llama.cpp/build-rocm/bin/llama-server")
MODEL = os.environ.get("LLAMA_MODEL", "/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf")
ROCM_LIB = os.environ.get("ROCM_LIB", "/opt/rocm-7.14-gfx1151/lib")
PORT = 8997
URL = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "700"))
CTX = int(os.environ.get("CTX", "16384"))
FA = os.environ.get("FA", "auto")
WARMUP = os.environ.get("WARMUP", "1") != "0"

PROMPTS = [
    "Explain in detail how a hash map works internally, including collision handling, load factor, and resizing. Then write a complete implementation in Python.",
    "Write a bash script that backs up a directory to a dated tarball, keeps the last 7 backups, and logs to syslog. Explain each step.",
    "Compare TCP and QUIC for a video streaming service. Cover head-of-line blocking, connection migration, and congestion control. Recommend one and justify it.",
    "Implement an LRU cache in Rust with O(1) get and put. Explain the data structure choices and show tests.",
    "A farmer has 17 sheep. All but 9 run away. Then he buys twice as many as remain. How many sheep does he have? Show your reasoning step by step, then write the same logic as a TypeScript function.",
]

if os.environ.get("PROMPT_FILE"):
    with open(os.environ["PROMPT_FILE"]) as _f:
        PROMPTS = [_f.read()]

ARMS = {
    "none": ["--spec-type", "none"],
    "n1": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"],
    "n2": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    "n4": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
    "n6": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
    "adaptive": ["--spec-type", "draft-mtp-adaptive"],
}


def start_server(arm):
    outdir = os.path.join(OUTROOT, arm)
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "server.log"), "w")
    cmd = [
        BIN, "--port", str(PORT), "-ngl", "99", "--jinja", "--flash-attn", FA,
        "-m", MODEL, "-ctk", "f16", "-ctv", "f16", "-np", "1", "-c", str(CTX),
        "--cache-reuse", "256", "--reasoning-format", "auto",
        "--chat-template-kwargs", '{"enable_thinking": true}', "--temp", "0",
        "--no-webui",
    ] + ARMS[arm]
    env = dict(os.environ, HIP_VISIBLE_DEVICES="0",
               LD_LIBRARY_PATH=ROCM_LIB)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(600):
        if p.poll() is not None:
            raise RuntimeError(f"server exited early (rc={p.returncode}); see {outdir}/server.log")
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=2) as r:
                if r.status == 200:
                    return p, log
        except Exception:
            pass
        time.sleep(1)
    p.kill()
    raise RuntimeError("server did not become ready")


def req(prompt):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "top_k": 1,
        "seed": 42,
        "cache_prompt": False,
        "logprobs": True,
        "top_logprobs": 8,
    }).encode()
    r = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as resp:
        return json.load(resp)


def run_arm(arm, idxs):
    outdir = os.path.join(OUTROOT, arm)
    print(f"### arm {arm}: {ARMS[arm]}", flush=True)
    p, log = start_server(arm)
    try:
        # warmup (reporter: one warmup request before the measured ones)
        if WARMUP:
            req(PROMPTS[0])
        rows = []
        for i in idxs:
            r = req(PROMPTS[i])
            with open(os.path.join(outdir, f"p{i}.json"), "w") as f:
                json.dump(r, f)
            msg = r["choices"][0]["message"]
            text = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
            h = hashlib.sha1(text.encode()).hexdigest()
            with open(os.path.join(outdir, f"p{i}.txt"), "w") as f:
                f.write(text)
            t = r.get("timings", {})
            dn = t.get("draft_n")
            acc = f"{t.get('draft_n_accepted', 0) / dn:.3f}" if dn else "-"
            rows.append((i, h, acc, len(text)))
            print(f"p{i} sha={h[:8]} acc={acc} chars={len(text)}", flush=True)
        with open(os.path.join(outdir, "hashes.txt"), "w") as f:
            for i, h, acc, n in rows:
                f.write(f"p{i}\t{h}\tacc={acc}\tchars={n}\n")
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    arg = sys.argv[1]
    if arg == "matrix":
        arms = list(ARMS)
        idxs = list(range(len(PROMPTS)))
    else:
        arms = [arg]
        idxs = [int(x) for x in sys.argv[2:]] or list(range(len(PROMPTS)))
    for a in arms:
        if a not in ARMS:
            print(f"unknown arm {a}; known: {list(ARMS)}")
            sys.exit(1)
        run_arm(a, idxs)


if __name__ == "__main__":
    main()
