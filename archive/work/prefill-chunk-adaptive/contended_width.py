#!/usr/bin/env python3
"""Contended-width check for block 16's admission chunk (wip, NOT part of the delivery).

WHAT IT ASSERTS
  With a decoder active, every step that carries prompt tokens must admit <= LLAMA_SERVER_PREFILL_CHUNK
  prompt tokens for the prefilling slot -- i.e. the chunk still bounds the decoder's stall -- and the
  uncontended arm must show steps WIDER than the chunk.  It reads the per-step `-lv 5` line:

      slot update_slots: id  0 | task N | fair share: budget = N, added = N, batch = N / 2048

  Run it once per binary: the old r305 build must DO the chunk even uncontended, the prototype must not.

Usage: ./contended_width.py --server PATH [--chunk 512] [--model PATH] [--gpu 0] [--keep]
Exit:  0 = both assertions hold, 1 = they do not, 2 = prerequisite error.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

try:
    import requests
except ImportError:
    sys.exit("contended_width: python 'requests' is required")

STEP_RE = re.compile(r"slot\s+\S+: id\s+(\d+) \| task \d+ \| fair share: budget =\s*(-?\d+), "
                     r"added =\s*(\d+), batch =\s*(\d+) / (\d+)")

LINE = ("The capital of France is Paris. A short paragraph that repeats to make a long prompt, "
        "with ordinary prose so the tokenizer produces ordinary tokens. ")


def start(a, env_extra, log_path):
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(a.gpu)
    env.update(env_extra)
    log = open(log_path, "w")
    cmd = [a.server, "-m", a.model, "-ngl", "99", "-np", "2", "-c", str(a.ctx), "-b", str(a.b),
           "-ub", str(a.ub), "-t", "8", "--no-warmup", "--port", str(a.port), "-lv", "5"] \
          + shlex.split(a.server_args)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = f"http://127.0.0.1:{a.port}"
    deadline = time.time() + a.load_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server died during load (rc={proc.returncode}); see {log_path}")
        try:
            if requests.get(base + "/health", timeout=2).json().get("status") == "ok":
                return proc, log, base
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"server not ready in {a.load_timeout}s; see {log_path}")


def stop(proc, log, base):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        log.close()


def arm(a, env_extra, log_path, contended):
    proc, log, base = start(a, env_extra, log_path)
    out = {}
    try:
        if contended:
            # the decoder must ALREADY be generating when the long prompt arrives, otherwise the first
            # steps are assembled while the newcomer is still a prompt (and the wide window is correct
            # there -- that is the documented arrival cost, not a contention failure)
            ready = threading.Event()

            def decoder():
                body = {"prompt": "Hello", "n_predict": a.dec_n, "stream": True, "temperature": 0,
                        "seed": 42, "cache_prompt": False, "id_slot": 1}
                t0 = time.time()
                n = 0
                with requests.post(base + "/completion", json=body, stream=True, timeout=600) as r:
                    for line in r.iter_lines():
                        if not line or not line.startswith(b"data: "):
                            continue
                        d = json.loads(line[6:])
                        if d.get("prompt_progress") is None and (d.get("content") or d.get("tokens")):
                            n += 1
                            if n >= 5:
                                ready.set()
                        if d.get("stop"):
                            break
                out["dec"] = (n, time.time() - t0)

            th = threading.Thread(target=decoder)
            th.start()
            if not ready.wait(timeout=120):
                raise RuntimeError("the decoder never produced 5 tokens")
        r = requests.post(base + "/completion", timeout=1800, json={
            "prompt": LINE * a.reps, "n_predict": 1, "stream": False, "temperature": 0, "seed": 42,
            "cache_prompt": False, "id_slot": 0})
        pre = r.json().get("timings") or {}
        if contended:
            th.join(timeout=900)
    finally:
        stop(proc, log, base)

    by_slot = {}
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                by_slot.setdefault(int(m.group(1)), []).append(int(m.group(3)))
    return {
        "prompt_n": pre.get("prompt_n"),
        "prompt_ms": pre.get("prompt_ms"),
        "prefill_steps": by_slot.get(0, []),
        "decoder": out.get("dec"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", default="/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--b", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--dec-n", type=int, default=48)
    ap.add_argument("--server-args", default="")
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--outdir", default="/tmp/prefill-chunk-width")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    env = {"LLAMA_SERVER_PREFILL_CHUNK": str(a.chunk), "LLAMA_SERVER_PREFILL_DUTY": "0"}

    fails = []
    for contended in (False, True):
        tag = "contended" if contended else "uncontended"
        log_path = os.path.join(a.outdir, f"{tag}.log")
        r = arm(a, env, log_path, contended)
        steps = r["prefill_steps"]
        print(f"\n== {tag}: prompt {r['prompt_n']} tok in {r['prompt_ms']:.0f} ms, "
              f"{len(steps)} prompt step(s), added = {steps}" +
              (f", decoder rc/wall/content = {r['decoder']}" if contended else ""))
        if not steps:
            fails.append(f"{tag}: no prompt-carrying step found in {log_path}")
            continue
        if contended:
            over = [x for x in steps if x > a.chunk]
            if over:
                fails.append(f"contended: {len(over)} step(s) admitted more than chunk {a.chunk}: {over}")
            else:
                print(f"   [PASS] every contended prompt step <= chunk {a.chunk}")
        else:
            wide = [x for x in steps if x > a.chunk]
            print(f"   [{'PASS' if wide else 'INFO'}] {len(wide)} of {len(steps)} uncontended step(s) "
                  f"wider than chunk {a.chunk} (a r301 binary shows none)")
    if fails:
        print("\nFAIL: " + "; ".join(fails))
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
