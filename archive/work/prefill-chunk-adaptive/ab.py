#!/usr/bin/env python3
"""Uncontended-prefill A/B for block 16's admission chunk (wip, NOT part of the delivery).

WHAT IT MEASURES
  One long prompt, ONE request, nothing decoding: prompt throughput, the number of
  prompt-carrying steps, and the per-step width (`added=` from the `-lv 5` fair-share line).
  That is the arm the chunk is NOT supposed to bound -- the cap exists to bound a decoder's
  inter-token latency, and with nothing generating there is no decoder to protect.

  The interesting comparison is across BINARIES, because r301's "cap always" is not reachable
  from the patched tree by any env arm:
      old r305 binary, CHUNK=512      -> cap always (the shipped default, ub 512)
      old r305 binary, CHUNK=0        -> no cap            (the throughput ceiling)
      new binary,      CHUNK=512      -> cap only while decoding (the proposal)
      new binary,      CHUNK=0        -> no cap            (the kill switch: must equal old/0)

WHY -lv 5
  `fair share: budget = N, added = N, batch = N / n_batch` is printed per slot per step by
  update_slots() (LOG_DBG = -lv 5, NOT -lv 4).  It is the only per-step instrument, so it is
  also what the gate's width assertion will read.

Usage:
  ./ab.py --arm "old:512:/home/stew675/llama.cpp/build-rocm/bin/llama-server:LLAMA_SERVER_PREFILL_CHUNK=512" \
          --arm "new:512:/home/stew675/llama-scratch/build-rocm/bin/llama-server:LLAMA_SERVER_PREFILL_CHUNK=512" \
          ... [--reps N] [--b 2048] [--ub 512] [--ctx 16384] [--gpu 0] [--model PATH]
              [--server-args "..."] [--keep]

Exit: 0 always (this is a measurement tool, it reports rather than judges).
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("ab: python 'requests' is required")

STEP_RE = re.compile(r"fair share: budget =\s*(-?\d+), added =\s*(\d+), batch =\s*(\d+) / (\d+)")

# ~14 tokens per repetition, the same shape gate-prefill-duty.py uses
LINE = ("The capital of France is Paris. A short paragraph that repeats to make a long prompt, "
        "with ordinary prose so the tokenizer produces ordinary tokens. ")


def build_prompt(reps):
    return LINE * reps


def run_arm(a, label, server, env_extra, prompt):
    port = a.port
    log_path = os.path.join(a.outdir, f"{label}.log")
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(a.gpu)
    env.update(env_extra)
    cmd = [server, "-m", a.model, "-ngl", "99", "-np", "1", "-c", str(a.ctx),
           "-b", str(a.b), "-ub", str(a.ub), "-t", str(a.threads),
           "--no-warmup", "--port", str(port), "-lv", "5"] + shlex.split(a.server_args)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + a.load_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited during load (rc={proc.returncode}); see {log_path}")
            try:
                if requests.get(base + "/health", timeout=2).json().get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError(f"server not ready in {a.load_timeout}s; see {log_path}")

        # one request, nothing else on the server
        t0 = time.time()
        r = requests.post(base + "/completion", timeout=1800, json={
            "prompt": prompt, "n_predict": a.n_predict, "stream": False, "temperature": 0, "seed": 42,
            "cache_prompt": False, "id_slot": 0,
            # a repetition-heavy prompt can make a model emit EOG on the first token, which would make the
            # text oracle the empty string; the sampled tokens are still greedy and comparable
            "ignore_eos": True})
        wall = time.time() - t0
        r.raise_for_status()
        body = r.json()
        tm = body.get("timings") or {}
        # a reasoning model puts its answer in reasoning_content; hash both, so the oracle is
        # never silently the empty string (`e3b0c442...` is sha256(""))
        text = (body.get("content") or "") + (body.get("reasoning_content") or "")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()

    added = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                added.append(int(m.group(2)))
    steps = len(added)
    return {
        "label": label,
        "server": server,
        "env": env_extra,
        "prompt_n": tm.get("prompt_n"),
        "prompt_ms": tm.get("prompt_ms"),
        "prompt_tps": tm.get("prompt_per_second"),
        "wall_s": wall,
        "steps": steps,
        "added": added,
        "max_added": max(added) if added else None,
        "mean_added": (sum(added) / len(added)) if added else None,
        "text_len": len(text),
        "text_sha": hashlib.sha256(text.encode()).hexdigest()[:12],
        "log": log_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help='LABEL:SERVER:ENV=VALUE[,ENV=VALUE...]  (omit the env part for "no extra env")')
    ap.add_argument("--model", default="/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf")
    ap.add_argument("--gpu", default="0,1")
    ap.add_argument("--port", type=int, default=8177)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--b", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=1,
                    help="1 for a pure prefill timing; >1 to also hash the generated text")
    ap.add_argument("--reps", type=int, default=600, help="long-prompt repetitions (~35 tokens each)")
    ap.add_argument("--server-args", default="")
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--outdir", default="/tmp/prefill-chunk-ab")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if not a.arm:
        sys.exit("ab: at least one --arm LABEL:SERVER is required")
    os.makedirs(a.outdir, exist_ok=True)
    prompt = build_prompt(a.reps)

    rows = []
    for spec in a.arm:
        parts = spec.split(":", 2)
        label, server = parts[0], parts[1]
        env_extra = {}
        if len(parts) > 2 and parts[2]:
            for kv in parts[2].split(","):
                k, _, v = kv.partition("=")
                env_extra[k] = v
        print(f"==> {label}  {server}  {env_extra}", flush=True)
        row = run_arm(a, label, server, env_extra, prompt)
        rows.append(row)
        print(f"    prompt {row['prompt_n']} tok  {row['prompt_ms']:.0f} ms  "
              f"{row['prompt_tps']:.1f} t/s  steps {row['steps']}  "
              f"added max {row['max_added']} mean {row['mean_added']}", flush=True)

    print()
    print(f"{'arm':<16} {'prompt_n':>9} {'ms':>8} {'t/s':>8} {'steps':>6} {'max added':>10} "
          f"{'mean added':>11} {'text':>14}")
    for r in rows:
        print(f"{r['label']:<16} {r['prompt_n']:>9} {r['prompt_ms']:>8.0f} {r['prompt_tps']:>8.1f} "
              f"{r['steps']:>6} {str(r['max_added']):>10} {r['mean_added']:>11.1f} "
              f"{str(r['text_len']) + '/' + r['text_sha']:>14}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
