#!/usr/bin/env python3
"""Prefill fairness gate for `llama-server --parallel N` (wip/session-scheduling).

WHY THIS EXISTS
  Every other gate in this repo runs ONE request at a time, and `llama-batched-bench` measures throughput,
  which cannot see a latency distribution.  The defect this gate covers is only visible with two sessions:
  `update_slots()` assembles one batch per step and lets the lowest slot fill it, so a request's prompt waits
  for another request's ENTIRE prompt (measured: a 7-token prompt behind a 13204-token one waits 1.91 s), and
  a decoding slot inherits the other slot's prefill chunk as its inter-token latency (10 ms -> 336 ms).

WHAT IT CHECKS  (two arms on the same binary, `-np 2`, one GPU)
  arm `baseline`   FAIR=0 CHUNK=0   today's slot-order fill (the reference)
  arm `candidate`  FAIR=1 CHUNK=<ubatch>   max-min fair share + a per-step chunk cap

  HARD  * the short prompt's TTFT behind a long one improves by >= --min-ttft-ratio (default 3x)
  HARD  * the candidate's worst decoder ITL during a foreign prefill is <= --max-itl-ms (default 120)
  HARD  * a lone long prompt's prefill throughput is within --max-prefill-loss of the baseline (3 %)
  HARD  * ONE request's greedy text is byte-identical between the arms (the single-slot oracle; a two-slot
          pair is NOT a text oracle -- co-batching changes the ubatch shape and hence the kernel family,
          GREEDY-PURITY.md §63 / issue #234)
  HARD  * each arm's startup census says the configuration it claims (`prefill admission: fair share = ...`)
  HARD  * no abort / assert in either log, and `scripts/check-gpu-faults.sh` reads both clean
  SOFT  * aggregate decode at concurrency 2 within --max-decode-loss (5 %) of the baseline

  `--self-test` runs the candidate AS the baseline configuration and requires the HARD fairness checks to
  FAIL.  A gate that cannot reject the unfixed build is not a gate; run this before trusting a green run.

Usage:  ./gate.py [--quick] [--keep] [--self-test] [--server PATH] [--model PATH] [--gpu 0]
Exit:   0 = pass, 1 = fail, 2 = usage/prerequisite error.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

try:
    import requests
except ImportError:
    sys.exit("gate: python 'requests' is required")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

TEXT_PROMPTS = [
    "[P1] Explain why the sky is blue in two sentences.",
    "[P2] List the first eight prime numbers, comma separated.",
]
SHORT_PROMPT = "[SHORT] Name three planets."
ABORT_RE = re.compile(r"GGML_ASSERT|GGML_ABORT|terminate called|Segmentation fault|fatal error", re.I)


class Server:
    def __init__(self, args, env_extra, log_path):
        self.args = args
        self.env_extra = env_extra
        self.log_path = log_path
        self.proc = None
        self.base = f"http://127.0.0.1:{args.port}"

    def start(self):
        env = dict(os.environ)
        env["HIP_VISIBLE_DEVICES"] = str(self.args.gpu)
        env.update(self.env_extra)
        cmd = [
            self.args.server, "-m", self.args.model, "-ngl", "99",
            "-np", "2", "-c", str(self.args.ctx), "-b", str(self.args.b),
            "-ub", str(self.args.ub), "-t", str(self.args.threads),
            "--no-warmup", "--port", str(self.args.port), "-lv", "4",
        ]
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        deadline = time.time() + self.args.load_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during load (rc={self.proc.returncode}); see {self.log_path}")
            try:
                if requests.get(self.base + "/health", timeout=2).json().get("status") == "ok":
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"server not ready within {self.args.load_timeout}s; see {self.log_path}")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        try:
            self.log.close()
        except Exception:
            pass

    def log_text(self):
        with open(self.log_path, "r", errors="replace") as f:
            return f.read()


def sentence(reps, tag=""):
    return (f"[{tag}] " if tag else "") + "The history of computing is a long and tangled story. " * reps


BASE = None  # set by measure(); one server is up at a time


def stream(prompt, n_predict, out, key, delay=0.0):
    time.sleep(delay)
    t0 = time.time()
    arrivals = []
    timings = None
    with requests.post(BASE + "/completion", json={
        "prompt": prompt, "n_predict": n_predict, "stream": True,
        "temperature": 0, "seed": 42, "cache_prompt": False,
    }, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode()
            if not s.startswith("data: "):
                continue
            d = json.loads(s[6:])
            if d.get("content"):
                arrivals.append(time.time())
            if d.get("stop"):
                timings = d.get("timings")
                break
    out[key] = {"arrivals": arrivals, "timings": timings, "wall": time.time() - t0,
                "ttft": (arrivals[0] - t0) if arrivals else None}


def run_pair(arms):
    out = {}
    th = [threading.Thread(target=stream, args=(a[0], a[1], out, a[2]) + tuple(a[3:])) for a in arms]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def measure(srv, reps, dec_n, agg_n):
    """The five measurements of one arm.  `reps` is the long prompt -- never shortened by --quick,
    because the prompt length IS the arm."""
    global BASE
    BASE = srv.base
    res = {"text": {}}

    # 1. single-request greedy text (the purity oracle)
    for p in TEXT_PROMPTS:
        r = requests.post(srv.base + "/completion", json={
            "prompt": p, "n_predict": 64, "stream": False,
            "temperature": 0, "seed": 42, "cache_prompt": False,
        }).json()
        c = r["content"]
        res["text"][p[:6]] = hashlib.sha256(c.encode()).hexdigest()[:16]

    # 2. starvation: a long prompt, then a 7-token prompt 50 ms later
    long_p = sentence(reps, "LONG")
    o = run_pair([(long_p, 4, "L"), (SHORT_PROMPT, 4, "S", 0.05)])
    res["short_ttft"] = o["S"]["ttft"]
    res["long_ttft"] = o["L"]["ttft"]

    # 3. decoder ITL while another slot prefills (fired 350 ms into the decode)
    dec_p = "[DEC] Count slowly from one to five hundred, one number per line."
    o = run_pair([(dec_p, dec_n, "dec"), (long_p, 2, "pf", 0.35)])
    a = o["dec"]["arrivals"]
    gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    res["max_itl"] = max(gaps) if gaps else None

    # 4. lone long-prompt prefill throughput (the cost of the policy)
    o = run_pair([(long_p, 2, "solo")])
    t = o["solo"]["timings"] or {}
    res["lone_tps"] = t.get("prompt_per_second", 0.0)

    # 5. aggregate decode at concurrency 2
    p = sentence(200, "T")
    o = run_pair([(p, agg_n, "a"), (p, agg_n, "b")])
    wall = max(v["wall"] for v in o.values())
    res["agg_tps"] = (2.0 * agg_n) / wall if wall else 0.0
    return res


def check_arm_config(log, fair, chunk):
    m = re.search(r"prefill admission: fair share = (\w+), chunk = (\d+)", log)
    if not m:
        return None
    return (m.group(1), int(m.group(2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.path.expanduser("~/llama.cpp/build-rocm/bin/llama-server"))
    ap.add_argument("--model", default="/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8151)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--b", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=0, help="candidate chunk; 0 = the ubatch size (the free point)")
    ap.add_argument("--reps", type=int, default=1200, help="long-prompt repetitions (~11 tokens each)")
    ap.add_argument("--min-ttft-ratio", type=float, default=3.0)
    ap.add_argument("--max-itl-ms", type=float, default=120.0)
    ap.add_argument("--max-prefill-loss", type=float, default=0.03)
    ap.add_argument("--max-decode-loss", type=float, default=0.05)
    ap.add_argument("--load-timeout", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="run the candidate AS the baseline config and require the HARD fairness checks to FAIL")
    a = ap.parse_args()

    # --quick trims only the decode-only measurements; the long prompt is the arm and is never shortened
    dec_n = 80 if a.quick else 200
    agg_n = 80 if a.quick else 200
    if a.chunk == 0 and not a.self_test:
        a.chunk = a.ub
    if not os.access(a.server, os.X_OK):
        sys.exit(f"gate: no executable llama-server at {a.server}")
    if not os.path.exists(a.model):
        sys.exit(f"gate: no model at {a.model}")

    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    logdir = os.path.join(HERE, "gate-logs")
    os.makedirs(logdir, exist_ok=True)
    faults = os.path.join(REPO, "scripts", "check-gpu-faults.sh")

    arms = [("baseline", {"LLAMA_SERVER_PREFILL_FAIR": "0", "LLAMA_SERVER_PREFILL_CHUNK": "0"}, False, 0)]
    cand_env = {"LLAMA_SERVER_PREFILL_FAIR": "0" if a.self_test else "1",
                "LLAMA_SERVER_PREFILL_CHUNK": "0" if a.self_test else str(a.chunk)}
    arms.append(("candidate", cand_env, True, a.chunk))

    results = {}
    logs = {}
    try:
        for name, env, is_cand, chunk in arms:
            log_path = os.path.join(logdir, f"{name}.log")
            print(f"\n=== arm {name}: FAIR={env['LLAMA_SERVER_PREFILL_FAIR']} "
                  f"CHUNK={env['LLAMA_SERVER_PREFILL_CHUNK']} -> {log_path}")
            srv = Server(a, env, log_path)
            try:
                srv.start()
                cfg = check_arm_config(srv.log_text(), env["LLAMA_SERVER_PREFILL_FAIR"], chunk)
                results[name] = measure(srv, a.reps, dec_n, agg_n)
                results[name]["census"] = cfg
            finally:
                srv.stop()
            logs[name] = log_path
            r = results[name]
            print(f"    short TTFT {r['short_ttft']:.3f}s  long TTFT {r['long_ttft']:.3f}s  "
                  f"max ITL {(r['max_itl'] or 0) * 1000:.0f}ms  lone prefill {r['lone_tps']:.0f} t/s  "
                  f"agg decode {r['agg_tps']:.0f} t/s  census {r['census']}")
    finally:
        pass

    b, c = results["baseline"], results["candidate"]
    fails, warns = [], []
    print("\n=== checks")

    def hard(ok, name, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    def soft(ok, name, detail):
        print(f"  [{'PASS' if ok else 'WARN'}] {name}: {detail}")
        if not ok:
            warns.append(name)

    ratio = (b["short_ttft"] / c["short_ttft"]) if c["short_ttft"] else 0.0
    hard(ratio >= a.min_ttft_ratio, "fairness",
         f"short-prompt TTFT {b['short_ttft']:.3f}s -> {c['short_ttft']:.3f}s "
         f"({ratio:.1f}x, need >= {a.min_ttft_ratio:.1f}x)")

    itl_ok = c["max_itl"] is not None and c["max_itl"] * 1000.0 <= a.max_itl_ms
    hard(itl_ok, "decode ITL",
         f"max {1000.0 * (c['max_itl'] or 0):.0f}ms (need <= {a.max_itl_ms:.0f}ms); "
         f"baseline {1000.0 * (b['max_itl'] or 0):.0f}ms")

    pf = (c["lone_tps"] / b["lone_tps"]) if b["lone_tps"] else 0.0
    hard(pf >= 1.0 - a.max_prefill_loss, "lone prefill",
         f"{b['lone_tps']:.0f} -> {c['lone_tps']:.0f} t/s ({100 * (pf - 1):+.1f} %, "
         f"limit -{100 * a.max_prefill_loss:.0f} %)")

    hard(c["text"] == b["text"], "single-request text",
         f"{c['text']} vs {b['text']} (must be identical)")

    ok_census = b["census"] == ("off", 0) and c["census"] == (
        "off" if a.self_test else "on", 0 if a.self_test else a.chunk)
    hard(ok_census, "census", f"baseline {b['census']}, candidate {c['census']}")

    aborts = [n for n, p in logs.items() if ABORT_RE.search(open(p, errors="replace").read())]
    hard(not aborts, "no abort", f"clean ({', '.join(logs)})" if not aborts else f"abort in {aborts}")

    if os.access(faults, os.X_OK):
        # bound the kernel-log window to THIS run: an unbounded journal carries yesterday's faults and
        # the gate would fail on someone else's crash
        args = [faults, "--quiet", "--since", t_start] + sum([["--log", p] for p in logs.values()], [])
        rc = subprocess.run(args, capture_output=True, text=True).returncode
        hard(rc == 0, "GPU faults", "clean" if rc == 0 else "check-gpu-faults.sh reported a fault")
    else:
        warns.append("GPU faults")
        print(f"  [WARN] GPU faults: no {faults}; not checked")

    dg = (c["agg_tps"] / b["agg_tps"]) if b["agg_tps"] else 0.0
    soft(dg >= 1.0 - a.max_decode_loss, "aggregate decode",
         f"{b['agg_tps']:.0f} -> {c['agg_tps']:.0f} t/s ({100 * (dg - 1):+.1f} %)")

    if a.self_test:
        # the candidate IS the baseline config: the gate must reject it
        fairness_failed = "fairness" in fails or "decode ITL" in fails
        print(f"\n=== self-test: gate must reject the unfixed build -> "
              f"{'PASS (it does)' if fairness_failed else 'FAIL (it accepts the defect)'}")
        if not a.keep:
            shutil.rmtree(logdir, ignore_errors=True)
        return 0 if fairness_failed else 1

    print(f"\n=== {'PASS' if not fails else 'FAIL'} ({len(fails)} hard failure(s), {len(warns)} warning(s))")
    if not a.keep and not fails:
        shutil.rmtree(logdir, ignore_errors=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(2)
