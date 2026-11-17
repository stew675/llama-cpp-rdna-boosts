"""Several concurrent PROMPTS with nothing decoding (wip, NOT part of the delivery).

The chunk is a contended cap, so the number of concurrent prefills must not matter: with no slot
in SLOT_STATE_GENERATING every prompt is admitted into the wide (n_batch) window, still shared
max-min fairly.  Reports the per-slot `-lv 5` added= lines and the wall time.  Run once per binary.

Usage: ./multi_prompt.py     (edit NEW/OLD at the bottom; -np 4, 4 x ~7000-token prompts, 1 GPU)
"""
import os, re, json, subprocess, time, threading, requests
STEP = re.compile(r"slot\s+\S+: id\s+(\d+) \| task \d+ \| fair share: budget =\s*(-?\d+), added =\s*(\d+), batch =\s*(\d+) / (\d+)")
LINE = ("The capital of France is Paris. A short paragraph that repeats to make a long prompt, "
        "with ordinary prose so the tokenizer produces ordinary tokens. ")
def run(server, env_extra, tag):
    env = dict(os.environ); env["HIP_VISIBLE_DEVICES"] = "0"; env.update(env_extra)
    log_path = f"/tmp/multi4-{tag}.log"; log = open(log_path, "w")
    cmd = [server, "-m", "/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf", "-ngl", "99", "-np", "4",
           "-c", "32768", "-b", "2048", "-ub", "512", "-t", "8", "--no-warmup", "--port", "8199", "-lv", "5"]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = "http://127.0.0.1:8199"
    for _ in range(300):
        try:
            if requests.get(base + "/health", timeout=2).json().get("status") == "ok": break
        except Exception: pass
        time.sleep(1)
    res = {}
    def one(i):
        r = requests.post(base + "/completion", timeout=1800, json={"prompt": LINE * 200, "n_predict": 1,
            "temperature": 0, "seed": 42, "cache_prompt": False, "id_slot": i})
        res[i] = r.json().get("timings") or {}
    ts = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.time() - t0
    p.terminate(); p.wait(timeout=20); log.close()
    by = {}
    for line in open(log_path, errors="replace"):
        m = STEP.search(line)
        if m: by.setdefault(int(m.group(1)), []).append(int(m.group(3)))
    steps = sum(len(v) for v in by.values())
    mx = max((max(v) for v in by.values()), default=0)
    print(f"{tag:>10}: 4 concurrent prompts, nothing decoding -> {steps} step(s) total, max added {mx}, "
          f"wall {wall:.2f}s, slots {sorted(by)}")
    for i in sorted(by): print(f"            slot {i}: {by[i]}")
NEW = "/home/stew675/llama-scratch/build-rocm/bin/llama-server"
OLD = "/home/stew675/llama.cpp/build-rocm/bin/llama-server"
run(OLD, {"LLAMA_SERVER_PREFILL_CHUNK": "512"}, "r305-512")
run(NEW, {"LLAMA_SERVER_PREFILL_CHUNK": "512"}, "new-512")
