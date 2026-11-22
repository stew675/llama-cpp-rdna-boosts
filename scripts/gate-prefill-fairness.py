#!/usr/bin/env python3
"""Prefill fairness gate for `llama-server --parallel N` (block 12, r300).

WHY THIS EXISTS
  Every other gate in this repo runs ONE request at a time, and `llama-batched-bench` measures throughput,
  which cannot see a latency distribution.  The defect this gate covers is only visible with two sessions:
  `update_slots()` assembles one batch per step and lets the lowest slot fill it, so a request's prompt waits
  for another request's ENTIRE prompt (measured: a 7-token prompt behind a 13204-token one waits 1.91 s), and
  a decoding slot inherits the other slot's prefill chunk as its inter-token latency (10 ms -> 336 ms).

WHAT IT CHECKS  (two arms on the same binary, `-np 2`, one GPU)
  arm `baseline`   FAIR=0 CHUNK=0 DUTY=0   today's slot-order fill (the reference)
  arm `candidate`  FAIR=1 CHUNK=<ubatch> DUTY=<--duty>   max-min fair share + a per-step chunk cap + the
                   prefill/decode duty cycle (16 = the shipped default; pinned on BOTH arms, so the baseline
                   stays the r299 fill and the self-test reproduces it)

  HARD  * ADMISSION: the short prompt's TTFT <= (its own measured admission step count) x (its own measured
          per-step cost) x --min-admission-slack.  This is the policy claim ("the wait is the request's own
          steps, not the other request's prompt") and it is tight: the r299 fill reads 6x its own floor.
  HARD  * FAIRNESS: the TTFT ratio >= min(--min-ttft-ratio, 0.8 x this shape's own ceiling), where the ceiling
          is the baseline's wait over that floor.  The bound is DERIVED, not a fixed 3x (TODO #338): the fixed
          number failed the streamed host-expert arm (2.7x) where the arm was at its real ceiling.
  HARD  * the candidate's worst decoder ITL during a foreign prefill is <= --max-itl-ms (default 120), and
  HARD  * with --arms-band K (the TWO-ARM server, block 17): the worst in-window ITL is also <= K x the
          candidate's OWN median step.  The shipped criterion above is derived from a SHARED step
          (ITL <= baseline_max x chunk/b), and on the pair there is no shared step: the decoder has its own
          context and thread, so the honest bound is its own step plus an acceptance band.  Measured on the
          pair that derived bound reads 13 ms against the decode arm's own 28-30 ms step, i.e. the gate was red
          on a CORRECT configuration.  --arms-band requires the candidate's --server-args to name the pair
          (a single-context run with the arms criterion would be vacuous).
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
import shlex
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
REPO = os.path.abspath(os.path.join(HERE, ".."))

TEXT_PROMPTS = [
    "[P1] Explain why the sky is blue in two sentences.",
    "[P2] List the first eight prime numbers, comma separated.",
]
SHORT_PROMPT = "[SHORT] Name three planets."
STREAMING_RE = re.compile(r"WINDOWED KV ENABLED|taking the ATTENDANCE POOL|is_streaming|pool .*enabled", re.I)
SPEC_RE = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)")
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
        ] + shlex.split(self.args.server_args)
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


def stream(prompt, n_predict, out, key, delay=0.0, id_slot=None):
    time.sleep(delay)
    t0 = time.time()
    arrivals = []
    progress = []
    timings = None
    body = {
        "prompt": prompt, "n_predict": n_predict, "stream": True,
        "temperature": 0, "seed": 42, "cache_prompt": False,
        # a per-step progress chunk: one per step that placed prompt tokens, plus one at arrival.  This is how
        # the gate MEASURES the short prompt's step count instead of assuming it (TODO #338): a 7-token prompt
        # is split 3+4 by the checkpoint tail rule, so its step count is a property of the model, not of K.
        "return_progress": True,
    }
    if id_slot is not None:
        # PIN the slot.  Slot assignment is LRU/similarity-based, and the fill policy gives the LOWEST slot
        # index priority -- so an unpinned arm silently measures the lucky case whenever the long prompt
        # lands on slot 1 (measured: baseline short TTFT 1.907 s pinned vs 0.828 s unpinned).
        body["id_slot"] = id_slot
    with requests.post(BASE + "/completion", json=body, stream=True) as r:
        if r.status_code != 200:
            # the server's own message is the useful one (context size, batch size, ...)
            raise RuntimeError(f"{r.status_code} from /completion: {r.text[:300]}")
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode()
            if not s.startswith("data: "):
                continue
            d = json.loads(s[6:])
            # a progress chunk is not always `content`: a reasoning model streams `reasoning_content`, and a
            # chunk can carry token ids with an empty text field.  Count any of them, or a reasoning arm
            # measures a TTFT of None.
            prog = d.get("prompt_progress")
            if prog is not None:
                progress.append(prog)
            # An ARRIVAL is a generation chunk.  A prompt-progress chunk is NOT one, and it carries
            # `tokens: [0]` with an empty `content`, so the old "any of content/reasoning/tokens" rule counted
            # the arrival progress chunk and collapsed TTFT to the launch time (measured 2026-11-16: a 7-token
            # prompt "arrived" at 0.002 s and generated at 0.052 s).  Progress chunks are skipped instead.
            if prog is None and (d.get("content") or d.get("reasoning_content") or d.get("tokens")):
                arrivals.append(time.time())
            if d.get("stop"):
                timings = d.get("timings")
                break
    out[key] = {"arrivals": arrivals, "timings": timings, "wall": time.time() - t0,
                "progress": progress,
                "ttft": (arrivals[0] - t0) if arrivals else None}


def run_pair(arms):
    out = {}
    th = [threading.Thread(target=stream, args=(a[0], a[1], out, a[2]) + tuple(a[3:])) for a in arms]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def measure_solo(srv, reps):
    """Lone long-prompt prefill throughput, on its OWN fresh server.

    It must not share a server with the interaction arms: the arena/graph state left by the earlier arms
    differs between a 512-token-step arm and a 2048-token-step one, and that moved this number by +25 % on a
    host-expert shape in one run.  Isolated on four fresh loads, the two configs read 2065-2073 t/s -- i.e.
    identical.  Measuring it in the 4th position would have shipped a 25 % phantom win.
    """
    global BASE
    BASE = srv.base
    o = run_pair([(sentence(reps, "SOLO"), 2, "solo", 0.0, 0)])
    return (o["solo"]["timings"] or {}).get("prompt_per_second", 0.0)


def measure(srv, reps, dec_n, agg_n):
    """The five measurements of one arm.  `reps` is the long prompt -- never shortened by --quick,
    because the prompt length IS the arm."""
    global BASE
    BASE = srv.base
    res = {"text": {}}

    # 1. decoder ITL while another slot prefills -- FIRST, on a fresh server.  The first prefill step after a
    # previous request's KV state costs a one-off (~1.8 s fair-on, ~2.5 s fair-off on a host-expert shape),
    # which is a pre-existing transition cost, not this policy; measuring it first keeps the arm comparable.
    # DISTINCT prompt text per activity: the same text run twice on one server is served from the prompt
    # cache / lazy idle slot, which silently shortens the second arm (measured: the starvation arm's baseline
    # TTFT fell 1.907 -> 0.839 s when the ITL arm ran first with the same text).
    long_p = sentence(reps, "ITL")
    dec_p = "[DEC] Count slowly from one to five hundred, one number per line."
    o = run_pair([(dec_p, dec_n, "dec", 0.0, 0), (long_p, 2, "pf", 0.35, 1)])
    a = o["dec"]["arrivals"]
    gaps = sorted((a[i + 1] - a[i]) * 1000.0 for i in range(len(a) - 1))
    if gaps:
        res["itl"] = {
            "median": gaps[len(gaps) // 2],
            "p95":    gaps[int(0.95 * (len(gaps) - 1))],
            "max":    gaps[-1],
            "n":      len(gaps),
        }
    else:
        res["itl"] = None

    # (the greedy-text oracle is pass 1b, on its own fresh server -- see measure_text)

    # 3. starvation: a long prompt, then a 7-token prompt 50 ms later
    long_s = sentence(reps, "STARVE")
    o = run_pair([(long_s, 4, "L", 0.0, 0), (SHORT_PROMPT, 4, "S", 0.05, 1)])
    res["short_ttft"] = o["S"]["ttft"]
    res["long_ttft"] = o["L"]["ttft"]
    # the short prompt's own admission steps: the arrival chunk (processed = 0) plus one per step that placed
    # tokens, so k == len(progress).  `prompt_ms` is the wall time of those same steps (k-1 of them), i.e. the
    # per-step cost the shape really charges -- which is what the bound below is derived from.
    res["short_steps"] = len(o["S"]["progress"])
    res["short_prompt_ms"] = (o["S"]["timings"] or {}).get("prompt_ms") or 0.0
    res["short_prompt_n"] = (o["S"]["timings"] or {}).get("prompt_n") or 0
    res["long_n"] = (o["L"]["timings"] or {}).get("prompt_n") or 0

    # 5. aggregate decode at concurrency 2
    p = sentence(200, "T")
    o = run_pair([(p, agg_n, "a", 0.0, 0), (p, agg_n, "b", 0.0, 1)])
    # Use the GENERATION phase only.  Wall time here includes both requests' prefill, which on a shape whose
    # prefill streams routed experts is seconds against 80 decode tokens -- that made this metric mostly
    # prefill variance ("-8 %" on one arm, "+22.7 %" on another, both meaningless).  predicted_ms is each
    # request's own decode span, so max() of them is the concurrent decode window.
    # ... and the span is first-arrival to last-arrival per request, which excludes prefill entirely.
    # (`timings.predicted_ms` is NOT usable for a streamed completion -- it reads back as a fraction of a
    # millisecond here, which reported 2e6 t/s.)
    spans = [(v["arrivals"][-1] - v["arrivals"][0], len(v["arrivals"]))
             for v in o.values() if len(v["arrivals"]) > 1]
    if spans:
        res["agg_tps"] = sum(n for _, n in spans) / max(sp for sp, _ in spans)
    else:
        res["agg_tps"] = 0.0
    return res


def spec_stats(log):
    """(accepted, generated, n_lines) from the per-request `draft acceptance` lines, or None if no spec ran."""
    acc = gen = n = 0
    for m in SPEC_RE.finditer(log):
        acc += int(m.group(2))
        gen += int(m.group(3))
        n += 1
    return (acc, gen, n) if n else None


def measure_text(srv):
    """The single-slot greedy text oracle, on its OWN fresh server.

    It must not share the interaction server: on a POOLED / streamed cache the KV cells a request gets follow
    the pages the earlier requests left resident, and FA's summation depends on the cells' tile indices, so a
    near tie can flip between two arms that did nothing wrong (GREEDY-PURITY.md section 52 and 63).  Measured
    2026-11-16 on Flash-Next + `--kv-attendance-pool`: P1 read 6436... in one arm and 79bd... in the other when
    hashed after the interaction arms, and 6436... on a fresh server in BOTH arms.  A fresh server keeps the
    oracle strict instead of documenting it away.
    """
    global BASE
    BASE = srv.base
    out = {}
    for p in TEXT_PROMPTS:
        r = requests.post(srv.base + "/completion", json={
            "prompt": p, "n_predict": 64, "stream": False,
            "temperature": 0, "seed": 42, "cache_prompt": False,
        }).json()
        c = (r.get("content") or "") + (r.get("reasoning_content") or "")
        out[p[:6]] = hashlib.sha256(c.encode()).hexdigest()[:16]
    return out


def check_arm_config(log, fair, chunk):
    # chunk is `off`, `n_ubatch` or an explicit count (it stopped being an integer when the default
    # became "follow the ubatch")
    m = re.search(r"prefill admission: fair share = (\w+), chunk = (\w+), skew = (\d+), duty = (\d+)", log)
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3), m.group(4))


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
    ap.add_argument("--server-args", default="",
                    help='extra llama-server args, e.g. "-ncmoe 48" or "-sm tensor --device 0,1,2"')
    ap.add_argument("--tag", default="", help="log-name tag, e.g. moe35b or tensor3gpu")
    ap.add_argument("--env", default="", help='extra env for BOTH arms, e.g. "MOE_EXPERT_CACHE_MIB=8192"')
    ap.add_argument("--chunk", type=int, default=0, help="candidate chunk; 0 = the ubatch size (the free point)")
    ap.add_argument("--reps", type=int, default=1200, help="long-prompt repetitions (~11 tokens each)")
    ap.add_argument("--min-ttft-ratio", type=float, default=3.0,
                    help="the CAP on the short-TTFT ratio; the bound actually applied is this or the shape's "
                         "own ceiling, whichever is smaller (derived, not fixed -- TODO #338)")
    ap.add_argument("--duty", type=int, default=16, help="the candidate's LLAMA_SERVER_PREFILL_DUTY")
    ap.add_argument("--min-admission-slack", type=float, default=1.25,
                    help="slack over the short prompt's own step count x its measured per-step cost")
    ap.add_argument("--max-itl-ms", type=float, default=0.0,
                    help="absolute ITL bound; 0 = DERIVE it from the candidate's own prefill rate (default)")
    ap.add_argument("--arms-band", type=float, default=0.0,
                    help="TWO-ARM path only (block 17): also require the worst in-window ITL to be within this "
                         "factor of the candidate's own median step.  0 = off (the shipped shared-step "
                         "criterion).  Needs the pair in --server-args; see the header")
    ap.add_argument("--itl-slack", type=float, default=1.75,
                    help="slack over one chunk's prefill time when the bound is derived (default 1.75)")
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
    logdir = os.path.join(HERE, "runs", "gate-prefill-fairness")
    os.makedirs(logdir, exist_ok=True)
    faults = os.path.join(REPO, "scripts", "check-gpu-faults.sh")

    # DUTY is pinned on BOTH arms: the baseline must be the r299 fill, and the self-test must reproduce it
    # with the duty off too, or the "unfixed build" it claims to reject is not the unfixed build.
    shared = {}
    for kv in filter(None, a.env.split(",")):
        k, _, v = kv.partition("=")
        shared[k.strip()] = v.strip()
    arms = [("baseline", dict(shared, LLAMA_SERVER_PREFILL_FAIR="0", LLAMA_SERVER_PREFILL_CHUNK="0",
                              LLAMA_SERVER_PREFILL_DUTY="0"), False, 0)]
    cand_env = dict(shared,
                    LLAMA_SERVER_PREFILL_FAIR="0" if a.self_test else "1",
                    LLAMA_SERVER_PREFILL_CHUNK="0" if a.self_test else str(a.chunk),
                    LLAMA_SERVER_PREFILL_DUTY="0" if a.self_test else str(a.duty))
    arms.append(("candidate", cand_env, True, a.chunk))

    results = {}
    logs = {}
    try:
        for name, env, is_cand, chunk in arms:
            log_path = os.path.join(logdir, f"{a.tag + '.' if a.tag else ''}{name}.log")
            print(f"\n=== arm {name}: FAIR={env['LLAMA_SERVER_PREFILL_FAIR']} "
                  f"CHUNK={env['LLAMA_SERVER_PREFILL_CHUNK']} {a.server_args} -> {log_path}")
            tag = f"{a.tag + '.' if a.tag else ''}{name}"
            # pass 1: lone prefill on its own fresh server (see measure_solo)
            srv = Server(a, env, os.path.join(logdir, f"{tag}.solo.log"))
            try:
                srv.start()
                lone_tps = measure_solo(srv, a.reps)
            finally:
                srv.stop()

            # pass 1b: the text oracle on its own fresh server (see measure_text)
            srv = Server(a, env, os.path.join(logdir, f"{tag}.text.log"))
            try:
                srv.start()
                text = measure_text(srv)
            finally:
                srv.stop()

            # pass 2: the interaction arms
            srv = Server(a, env, log_path)
            try:
                srv.start()
                cfg = check_arm_config(srv.log_text(), env["LLAMA_SERVER_PREFILL_FAIR"], chunk)
                results[name] = measure(srv, a.reps, dec_n, agg_n)
                results[name]["text"] = text
                results[name]["census"] = cfg
                results[name]["lone_tps"] = lone_tps
                results[name]["spec"] = spec_stats(srv.log_text())
                results[name]["spec_on"] = "adding speculative implementation" in srv.log_text()
                results[name]["streaming"] = bool(STREAMING_RE.search(srv.log_text()))
            finally:
                srv.stop()
            logs[name] = log_path
            r = results[name]
            itl = r["itl"] or {}
            print(f"    short TTFT {r['short_ttft']}  long TTFT {r['long_ttft']}  "
                  f"ITL med/p95/max {itl.get('median', 0):.0f}/{itl.get('p95', 0):.0f}/"
                  f"{itl.get('max', 0):.0f}ms  lone prefill {r['lone_tps']:.0f} t/s  "
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

    # ---- TODO #338: the ratio bound is DERIVED, not 3.0.
    # The claim is "the short prompt waits for its own admission steps, not for the long prompt".  k is that
    # step count, MEASURED from the short request's progress events (a 7-token prompt is split 3+4 by the
    # checkpoint tail rule, so k is a property of the model); p is the per-step cost the shape really charges,
    # measured as the short prompt's own prompt_ms / (k-1).  Their product is the floor.
    #   * the ADMISSION check is the policy claim, and it is tight: a late admission cannot pass it (the
    #     r299 fill measures 6x its own floor);
    #   * the RATIO check is the user-visible claim, whose ceiling is the baseline's own wait over that floor.
    #     L/(k*chunk) alone is NOT that ceiling: it assumes the baseline's step is chunk-wide and its per-step
    #     fixed cost is zero.  On the streamed host-expert shape the baseline's step is n_batch-wide (4x the
    #     tokens, so 4x the a-part but the same b), which is why that shape measured 5.3x where L/(k*chunk)
    #     promises 7.3x -- the arm was at its real ceiling all along, and the fixed 3.0 was the thing that was
    #     wrong (measured 2026-11-16, TODO #338).
    k = c["short_steps"] or 0
    p_step_s = (c["short_prompt_ms"] / 1000.0 / (k - 1)) if k > 1 else 0.0
    # k-1 placement steps plus one ARRIVAL-ALIGNMENT step: a request that arrives inside a step cannot join
    # it, and that step is not in its own prompt_ms.  The slack covers arriving part-way through it.
    floor = (k + 1) * p_step_s
    admitted = bool(floor) and c["short_ttft"] <= floor * a.min_admission_slack
    hard(admitted, "admission",
         f"the short prompt's TTFT {c['short_ttft']:.3f}s <= {floor:.3f}s x {a.min_admission_slack:.2f} "
         f"({k - 1} placement + 1 alignment step(s) x {p_step_s * 1000:.0f} ms, its own measured cost; a "
         f"late admission cannot pass)")
    ceiling = (b["short_ttft"] / floor) if floor else 0.0
    bound = min(a.min_ttft_ratio, ceiling * 0.8)
    hard(ratio >= bound, "fairness",
         f"short-prompt TTFT {b['short_ttft']:.3f}s -> {c['short_ttft']:.3f}s ({ratio:.1f}x); bound "
         f"{bound:.1f}x = min({a.min_ttft_ratio:.1f}, 0.8 x this shape's own ceiling {ceiling:.1f}x)")
    naive = (f"{c['long_n'] / (k * a.chunk):.1f}x" if a.chunk > 0 else "n/a (no chunk)")
    if ceiling < 1.5:
        print(f"  [INFO] this shape has NO room to improve: its per-step cost is dominated by a fixed term "
              f"(a pooled/streamed KV re-reads spilled cells once per STEP, so a smaller chunk buys nothing "
              f"there), so its ceiling is {ceiling:.1f}x and the bound above is the shape's own, not 3x.  The "
              f"arm is judged on ADMISSION, which is what the policy actually promises")
    print(f"  [INFO] shape: a {c['short_prompt_n']}-token short prompt over a {c['long_n']}-token one, "
          f"chunk {a.chunk}, {k} admission step(s) at {p_step_s * 1000:.0f} ms each.  L/(k*chunk) would "
          f"promise {naive}; the ceiling above is the MEASURED one, and it is lower because the baseline's "
          f"step is n_batch-wide, not chunk-wide")

    # A decoder's ITL is bounded by the step it waits on, so the honest bound is "one chunk of prompt at
    # this shape's prefill rate" -- NOT a fixed millisecond count, which would fail a slow host-expert model
    # for a legitimate reason.  Derived from the candidate's own measured prefill rate unless overridden.
    # `max`, not `p95`: a cap trades a FEW huge stalls for MANY small ones, so max alone would reject the fix
    # and p95 alone would hide the 1 s spike the baseline has.  Both are reported; the HARD check is max,
    # because "the wait is bounded by one chunk" is the claim, and p95 legitimately rises.
    ci, bi = c["itl"] or {}, b["itl"] or {}
    itl_bound_ms = 0.0   # unset on the "no cap, no explicit bound" path below (the self-test's arm)
    if a.chunk <= 0 and a.max_itl_ms <= 0:
        # no cap and no explicit bound: nothing smaller than the step bounds the decoder, so this is not a
        # check.  Say so instead of inventing a threshold (the self-test's candidate has chunk 0).
        warns.append("ITL unbounded")
        print(f"  [WARN] decode ITL: no --chunk and no --max-itl-ms, so there is no bound to check "
              f"(candidate max {ci.get('max', 0):.0f}ms) -- the cap is what bounds this")
    else:
        # The claim is "a decoder's wait is proportional to the prompt tokens in the step", so the bound is
        # RELATIVE to the baseline's own worst step, scaled by chunk/n_batch.  Deriving it from the lone-prefill
        # rate instead is wrong on a bandwidth-shared box: on gfx1151 (UMA) a co-batched prefill runs at ~1550
        # t/s, not the 2604 t/s it reaches alone, which made a correct 405 ms look like a 344 ms failure.
        streaming = b.get("streaming") or c.get("streaming")
        if a.max_itl_ms > 0:
            itl_bound_ms, itl_why = a.max_itl_ms, f"--max-itl-ms {a.max_itl_ms:.0f}"
        elif streaming:
            # A windowed / pooled KV re-reads the spilled cells IN PLACE once per STEP, and that read does not
            # shrink with the chunk: fitting two chunk sizes here gives step = 0.158 ms/token x chunk + ~167 ms
            # at full depth, so the achievable ratio floors at ~0.34 instead of chunk/b = 0.25.  The bound
            # therefore asks for a 25 % improvement rather than a proportional one -- and says why.
            itl_bound_ms = bi.get("max", 0) * 0.75
            itl_why = (f"streaming KV: baseline max ({bi.get('max', 0):.0f}ms) x 0.75 -- a per-step spill "
                       f"floor (~167 ms at full depth here) does not scale with the chunk")
        else:
            frac = (a.chunk / a.b) if a.b else 1.0
            itl_bound_ms = (bi.get("max", 0) * frac * a.itl_slack) if bi.get("max") else float("inf")
            itl_why = (f"{a.itl_slack:.2f} x baseline max ({bi.get('max', 0):.0f}ms) x chunk/b "
                       f"({a.chunk}/{a.b} = {frac:.3f})")
        itl_ok = bool(ci) and ci["max"] <= itl_bound_ms
        # SUPERSEDED ON THE TWO-ARM SERVER (r346, block 17): the derived form scales the BASELINE's max by
        # chunk/b, and on the pair the baseline's max is the decode arm's own idle step -- so it reads a bound
        # below that step (10 ms against 20 ms, measured 2026-11-22) and calls a CORRECT pair a failure.  With
        # --arms-band the arms criterion below IS the criterion; this line states the old bound and says why it
        # is not asserted, so a reader can still see both numbers.
        on_pair = "prefill-decode-arms" in (a.server_args or "")
        if on_pair and a.arms_band > 0 and a.max_itl_ms <= 0:
            print(f"  [INFO] decode ITL max (superseded): the shared-step bound would read "
                  f"{itl_bound_ms:.0f}ms ({itl_why}) against a candidate max of {ci.get('max', 0):.0f}ms -- on the "
                  f"pair there is no shared step, so the arms criterion below is the asserted one")
        else:
            hard(itl_ok, "decode ITL max",
                 f"{ci.get('max', 0):.0f}ms <= {itl_bound_ms:.0f}ms ({itl_why}); "
                 f"ratio {ci.get('max', 0) / bi['max']:.2f} of baseline" if bi.get("max") else
                 f"{ci.get('max', 0):.0f}ms <= {itl_bound_ms:.0f}ms ({itl_why})")
    # THE ARMS-AWARE CRITERION (r346, block 17).  On the pair the decoder owns a context and a thread, so the
    # wait it experiences is its OWN step (its graph, its sampling) plus whatever the other arm's step
    # contends for -- not a fraction of a shared prefill step.  The median of the candidate's own gaps IS that
    # step (the run is dominated by uncontended decode steps), so the bound is median x band, and the band is
    # the acceptance criterion rather than a tuned constant: on a correct pair the measured max sat at 64-78 ms
    # against a 47-49 ms median, i.e. ~1.6x, and a band of 2.0 leaves room for the prefill's own tail without
    # accepting an unbounded stall.
    if a.arms_band > 0:
        # THE ARMS-AWARE CRITERION (r346, block 17).  Same CLAIM as the shipped one -- "a decoder's wait is
        # bounded by one chunk of a foreign prompt at this shape's own rate" -- but the pair needs it expressed
        # per ARM, because the shipped form scales the BASELINE's max by chunk/b and the baseline's max on the
        # pair is the decode arm's own idle step (23 ms here), not a shared prefill step.  Measured 2026-11-22
        # on the 4B: that form reads a 10 ms bound against a correct pair whose max is 72 ms, and 72 ms IS one
        # 512-token chunk at the shape's own 7357 t/s.  So the bound is
        #       decode arm's own step  +  band x one chunk of foreign prefill
        # which is decidable from numbers this gate already measures, and which FAILS a pair that lets a
        # prefill step run long (a broken chunk cap) instead of passing it.
        chunk = a.chunk if a.chunk > 0 else a.ub
        if "prefill-decode-arms" not in (a.server_args or ""):
            print("  [FAIL] arms ITL: --arms-band needs the candidate to run the two-arm server "
                  "(--server-args '... --prefill-decode-arms ...'); a single-context run cannot exercise it")
            fails.append("arms ITL (not applicable)")
        elif not ci or not (c.get("lone_tps") or 0):
            print("  [FAIL] arms ITL: no candidate gaps or no lone-prefill rate to judge")
            fails.append("arms ITL (no data)")
        else:
            step_ms  = 1000.0 * chunk / c["lone_tps"]
            arms_bound = ci["median"] + a.arms_band * step_ms
            hard(ci["max"] <= arms_bound, "arms ITL max",
                 f"{ci['max']:.0f}ms <= {arms_bound:.0f}ms = the decode arm's own step {ci['median']:.0f}ms "
                 f"+ {a.arms_band:.2f} x one {chunk}-token chunk of foreign prefill ({step_ms:.0f}ms at the "
                 f"shape's own {c['lone_tps']:.0f} t/s)")
            print(f"  [INFO] arms ITL: max/step = {ci['max']/max(ci['median'],1):.1f}x; the wait is one chunk, "
                  f"so this ratio is a property of chunk-vs-model-speed, not of the pair")

    # a max close to the bound is expected (it IS one chunk of prompt); only a max that overshoots it while
    # still passing is worth naming.  Comparing against p95 instead produced a false "single-event stall" on
    # an arm whose median was 0 ms and whose max was simply the last chunk (88 ms of a 144 ms bound).
    if ci and itl_bound_ms > 0 and ci["max"] > 0.9 * itl_bound_ms and not ("prefill-decode-arms" in (a.server_args or "") and a.arms_band > 0 and a.max_itl_ms <= 0):
        warns.append("ITL near bound")
        print(f"  [WARN] ITL near the bound: max {ci['max']:.0f}ms of {itl_bound_ms:.0f}ms -- within limits "
              f"but with little headroom; the cap is what sets this, so lower --chunk to buy margin")
    # The cap BOUNDS the tail but touches MORE steps (it converts a few 1 s stalls into many short ones), so
    # p95 legitimately gets worse while max gets better.  Both are reported; only max is asserted, because
    # "the wait is bounded by one chunk" is the claim being made.
    print(f"  [INFO] ITL med/p95/max: candidate {ci.get('median', 0):.0f}/{ci.get('p95', 0):.0f}/"
          f"{ci.get('max', 0):.0f}ms over {ci.get('n', 0)} gaps; baseline {bi.get('median', 0):.0f}/"
          f"{bi.get('p95', 0):.0f}/{bi.get('max', 0):.0f}ms -- the cap trades a few long stalls for more "
          f"short ones, so p95 may rise while max falls")

    pf = (c["lone_tps"] / b["lone_tps"]) if b["lone_tps"] else 0.0
    hard(pf >= 1.0 - a.max_prefill_loss, "lone prefill",
         f"{b['lone_tps']:.0f} -> {c['lone_tps']:.0f} t/s ({100 * (pf - 1):+.1f} %, "
         f"limit -{100 * a.max_prefill_loss:.0f} %)")

    hard(c["text"] == b["text"], "single-request text",
         f"{c['text']} vs {b['text']} (must be identical)")

    ok_census = b["census"] == ("off", "off", "1", "1") and c["census"] == (
        "off" if a.self_test else "on", "off" if a.self_test else str(a.chunk), "1",
        str(a.duty) if not a.self_test else "1")
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

    # speculative decoding: with `--spec-type ...` in --server-args, assert that it really ran in BOTH arms
    # (a spec arm that silently failed to initialize would pass every other check) and that acceptance did not
    # collapse.  Acceptance is NOT asserted equal: the fair share changes the ubatch composition, and a verify
    # width is a numerics reference (GREEDY-PURITY.md §63), so it is reported and only a collapse is a failure.
    if c["spec_on"] or b["spec_on"]:
        ok_on = b["spec_on"] and c["spec_on"]
        hard(ok_on, "spec active", f"baseline {b['spec_on']}, candidate {c['spec_on']}")
        if ok_on and b["spec"] and c["spec"]:
            br, cr = b["spec"][0] / b["spec"][1], c["spec"][0] / c["spec"][1]
            hard(cr >= 0.5 * br, "spec acceptance",
                 f"{br:.3f} ({b['spec'][0]}/{b['spec'][1]}) -> {cr:.3f} ({c['spec'][0]}/{c['spec'][1]}); "
                 f"need >= {0.5 * br:.3f} (a collapse, not equality -- §63)")

    if b["agg_tps"] and c["agg_tps"]:
        dg = c["agg_tps"] / b["agg_tps"]
        soft(dg >= 1.0 - a.max_decode_loss, "aggregate decode",
             f"{b['agg_tps']:.0f} -> {c['agg_tps']:.0f} tok/s ({100 * (dg - 1):+.1f} %)")
    else:
        # a metric that cannot be measured must not produce a verdict: a shape that streams one chunk per
        # request has no inter-token span to measure, and an earlier wall-clock version silently measured
        # PREFILL instead (which is how "-8 %" and "+22.7 %" both appeared on the same arm).
        print("  [INFO] aggregate decode: not measurable on this shape (no inter-token span); skipped")

    if a.self_test:
        # the candidate IS the baseline config: the gate must reject it
        fairness_failed = ("fairness" in fails or "decode ITL" in fails) and "admission" in fails
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
