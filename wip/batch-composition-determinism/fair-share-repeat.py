#!/usr/bin/env python3
"""Two-prompt repeat harness (block 16's fair share): does ONE server repeat itself?

One /completion carrying an ARRAY of two prompts = two tasks admitted in the same step, so the
fair-share split of the window decides how each prompt is cut into ubatches.  Sent N times to ONE
server process, greedy, cache_prompt false.  Prints a hash of each reply, the wall time and the
per-prompt rates; the verdict line is whether the replies repeated.

This is what PR #383's claim reduces to.  On r378 the answer is YES at the default settings, and
with `--skew N` (LLAMA_SERVER_PREFILL_SKEW > 1) plus `--same` it also stays YES even though the
per-step split mirrors: see wip/batch-composition-determinism/README.md.

Usage: fair-share-repeat.py --server PATH [--model PATH] [--repeat 3] [--n-predict 640]
                            [--skew N] [--same] [--duty K] [--tag NAME] [--extra "-ncmoe 47"]
"""
import argparse, hashlib, json, os, signal, subprocess, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPTS = os.path.join(REPO, "prompts")

def load(name):
    with open(os.path.join(PROMPTS, name), "r", encoding="utf-8") as f:
        return f.read()

def build_prompts():
    prose   = load("prose-rdna-boosts.txt")
    reason  = load("reasoning.txt")
    code    = load("code-python.txt")
    recall  = load("recall.txt")
    mixed   = load("code-reasoning-mixed.txt")
    p1 = prose + "\n\n" + prose + "\n\n" + reason            # ~10.9K tokens
    p2 = prose                                                      # ~5.2K tokens
    return p1, p2

def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def wait_health(base, proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited with %s" % proc.returncode)
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("server did not become healthy")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", default="/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=640)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--port", type=int, default=18083)
    ap.add_argument("--extra", default="", help="extra llama-server args, space separated")
    ap.add_argument("--duty", default="1")
    ap.add_argument("--skew", default="")
    ap.add_argument("--same", action="store_true")
    args = ap.parse_args()

    p1, p2 = build_prompts()
    if args.same:
        p2 = p1
    logdir = os.path.join(HERE, "logs")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "%s.log" % args.tag)

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["ROCBLAS_USE_HIPBLASLT"] = "0"          # INVARIANTS §44: a fresh-process comparison needs it
    if args.duty:
        env["LLAMA_SERVER_PREFILL_DUTY"] = args.duty
    if args.skew:
        env["LLAMA_SERVER_PREFILL_SKEW"] = args.skew

    cmd = [args.server, "-m", args.model, "-c", "65536", "-ngl", "99",
           "--parallel", "2", "--kv-unified", "-ub", "2048",
           "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
           "--port", str(args.port), "--host", "127.0.0.1", "-lv", "4"]
    if args.extra:
        cmd += args.extra.split()

    with open(logpath, "wb") as logf:
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        try:
            took = wait_health("http://127.0.0.1:%d" % args.port, proc)
            print("server up in %.1fs (log %s)" % (took, logpath))
            url = "http://127.0.0.1:%d/completion" % args.port
            rows = []
            for i in range(args.repeat):
                payload = {
                    "prompt": [p1, p2],
                    "n_predict": args.n_predict,
                    "temperature": 0.0,
                    "top_k": 1,
                    "seed": 42,
                    "cache_prompt": False,
                    "stream": False,
                }
                t0 = time.time()
                res = post(url, payload, timeout=1800)
                wall = time.time() - t0
                tim = []
                for r in res:
                    t = r.get("timings", {})
                    tim.append("%.0f/%.1f" % (t.get("prompt_per_second", -1), t.get("predicted_per_second", -1)))
                hs = []
                for r in res:
                    c = r.get("content", "") + "\x00" + r.get("reasoning_content", "") 
                    h = hashlib.sha256(c.encode()).hexdigest()[:8]
                    hs.append("%s(%dch,%drc,%dtok)" % (h, len(r.get("content", "")),
                                                       len(r.get("reasoning_content", "") or ""),
                                                       r.get("tokens_predicted", -1)))
                rows.append(hs)
                print("  request %d: %s  wall %.2fs  (prompt t/s per prompt: %s)" % (i + 1, "  ".join(hs), wall, " ".join(tim)), flush=True)
            print("\n%s" % args.tag)
            same = all(r == rows[0] for r in rows)
            print("  replies identical across requests: %s" % ("YES" if same else "NO"))
            print("  prompt1: %s" % " ".join(r[0].split("(")[0] for r in rows))
            print("  prompt2: %s" % " ".join(r[1].split("(")[0] for r in rows))
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()

if __name__ == "__main__":
    main()
