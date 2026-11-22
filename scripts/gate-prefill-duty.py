#!/usr/bin/env python3
"""Prefill/decode duty-cycle gate (block 16, the server's prompt admission policy).

WHY THIS EXISTS
  A decoding slot gets exactly one token per step, so its rate is 1/step_time -- and a step that carries
  prompt tokens is the slow one.  On the field shape (3 GPU tensor, one 16k step) a decoder sharing a server
  with a 20k-token prefill measured 5 tokens over a 6 s prefill, i.e. 0.25 t/s for 5 s at a time.  r300's
  per-step chunk bounds the damage; the duty cycle attacks it from the other end: it leaves steps decode-only
  and puts the prompt into one step per cycle.

  The mechanism is a TRADE, so both of its sides are asserted here: the decode burst must be real, and the
  prefill's own wall time must stay inside the bound the design claims.  Without that second half a duty
  cycle is just a prefill slowdown.

WHAT IT CHECKS  (two arms on ONE binary, `-np 2`, one GPU, both slots pinned)
  arm `baseline`   LLAMA_SERVER_PREFILL_DUTY=0
  arm `candidate`  LLAMA_SERVER_PREFILL_DUTY=<--duty> (default 16 = the shipped default)

  HARD  * decode tokens delivered WHILE the other slot prefills: candidate >= --min-decode-gain x baseline,
          OR -- with --inert-band F -- within +-F of the baseline, in which case the mechanism is a deliberate
          no-op on that path and must not cost prefill either.  The two-arm server (block 17) is that case: the
          decoder has its own context and thread, so the duty cycle is DEPRECATED there and this gate runs on
          it with --inert-band 0.10 (which is a symmetric check, so a duty cycle that LOST tokens also fails).
          This is the feature's whole claim; a latency median cannot see it (the tokens arrive in a burst).
  HARD  * the candidate's concurrent prefill wall is <= --max-prefill-stretch x the baseline's.  The design
          bounds the idle time by the previous prefill step's own duration, so the analytic bound is 2x; the
          slack covers a dirty first cycle (the budget is unknown until one prefill step has been measured).
  HARD  * a prompt arriving with NOTHING decoding is prefilled at the unchanged rate (<= --max-lone-loss):
          the deferral must never fire into an empty batch -- decode() calls that "no tokens to decode" and
          aborts after four in a row, which is also why an idle server is untouched by the policy.
  HARD  * one request's greedy text, alone on the server, is byte-identical between the arms (the single-slot
          oracle; a two-slot pair is NEVER a text oracle, GREEDY-PURITY.md §63).
  HARD  * each arm's startup census states the duty it claims (`prefill admission: ... duty = N`).
  HARD  * no "no tokens to decode" line and no abort/assert in the candidate arm, and
          scripts/check-gpu-faults.sh clean with --since bounded to this run.
  HARD  * with --spec-args (an MTP/DFlash drafter): the decode gain holds under the speculator too, the text
          is identical, and draft acceptance does not collapse (>= 0.5x -- a verify width is a numerics
          reference, so it is judged for collapse, not equality).
  SOFT  * the decoder's worst inter-token gap during the prefill is not worse than the baseline's.

Usage:  ./gate-prefill-duty.sh [--quick] [--keep] [--duty N] [--server PATH] [--model PATH] [--gpu 0]
                               [--server-args "-ncmoe 47"] [--spec-args "--spec-type draft-mtp --spec-draft-n-max 3"]
Example (the release runs both):  the dense 4B shape, and the 27B + draft-mtp shape.
Exit:   0 = pass, 1 = fail, 2 = usage/prerequisite error.
"""
import argparse
import hashlib
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
    sys.exit("gate: python 'requests' is required")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ABORT_RE = re.compile(r"GGML_ASSERT|GGML_ABORT|terminate called|Segmentation fault|fatal error", re.I)
CENSUS_RE = re.compile(r"prefill admission: fair share = (\w+), chunk = (\w+), skew = (\d+), duty = (\d+)")
# `-lv 5`: the per-step admission line (LOG_DBG, so -lv 4 does NOT show it).  `batch - added` is exactly the
# decode tokens the step already carried, because these passes fill ONE prompt slot.
WIDTH_RE = re.compile(r"slot\s+\S+: id\s+(\d+) \| task \d+ \| fair share: budget =\s*(-?\d+), "
                      r"added =\s*(\d+), batch =\s*(\d+) / (\d+)")
SPEC_RE = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)")

BASE = None


class Server:
    def __init__(self, a, env_extra, log_path):
        self.a = a
        self.env_extra = env_extra
        self.log_path = log_path
        self.proc = None
        self.base = f"http://127.0.0.1:{a.port}"

    def start(self):
        env = dict(os.environ)
        env["HIP_VISIBLE_DEVICES"] = str(self.a.gpu)
        env.update(self.env_extra)
        cmd = [self.a.server, "-m", self.a.model, "-ngl", "99", "-np", "2", "-c", str(self.a.ctx),
               "-b", str(self.a.b), "-ub", str(self.a.ub), "-t", str(self.a.threads),
               "--no-warmup", "--port", str(self.a.port), "-lv", "5"] \
              + shlex.split(self.a.server_args) + shlex.split(self.a.spec_args)
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        # poll the PID as well as /health: a crashed server is not a slow one, and a harness that cannot tell
        # "not ready" from "dead" invents numbers
        deadline = time.time() + self.a.load_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during load (rc={self.proc.returncode}); see {self.log_path}")
            try:
                if requests.get(self.base + "/health", timeout=2).json().get("status") == "ok":
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"server not ready within {self.a.load_timeout}s; see {self.log_path}")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()

    def text(self):
        with open(self.log_path, "r", errors="replace") as f:
            return f.read()


def stream(prompt, n_predict, out, key, delay=0.0, id_slot=None):
    time.sleep(delay)
    t0 = time.time()
    arrivals = []
    timings = None
    body = {"prompt": prompt, "n_predict": n_predict, "stream": True, "temperature": 0, "seed": 42,
            "cache_prompt": False}
    if id_slot is not None:
        body["id_slot"] = id_slot  # PIN the slot: an unpinned two-slot arm measures a race
    with requests.post(BASE + "/completion", json=body, stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} from /completion: {r.text[:300]}")
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            d = json.loads(line[6:])
            # An ARRIVAL is a generation chunk.  A prompt-progress chunk is NOT one even though it carries
            # `tokens: [0]` with an empty `content` (measured 2026-11-16: counting it collapses TTFT to the
            # request's launch time).
            if d.get("prompt_progress") is None and (
                    d.get("content") or d.get("reasoning_content") or d.get("tokens")):
                arrivals.append(time.time())
            if d.get("stop"):
                timings = d.get("timings")
                break
    out[key] = {"t0": t0, "arrivals": arrivals, "timings": timings,
                "ttft": arrivals[0] - t0 if arrivals else None}


def run_pair(arms):
    out = {}
    th = [threading.Thread(target=stream, args=a) for a in arms]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def sentence(reps, tag):
    # distinct text per activity: the same text run twice on one server is served from the prompt cache and
    # silently shortens the second arm
    return f"[{tag}] The history of computing is a long and tangled story. " * reps


def measure(srv, a):
    global BASE
    BASE = srv.base
    r = {"text": {}}

    # 1. the mixed arm: a decoder on slot 0, then the long prompt on slot 1.  FIRED FRESH, and the decoder gets
    # a distinct prompt from the lone-prefill arm's text.
    out = {}
    t = [threading.Thread(target=stream, args=("[DEC] Count slowly from one to five hundred, one number per line.",
                                               a.dec_n, out, "dec", 0.0, 0)),
         threading.Thread(target=stream, args=(sentence(a.reps, "MIX"), 2, out, "pf", 0.4, 1))]
    for x in t:
        x.start()
    for x in t:
        x.join()
    dec, pf = out["dec"], out["pf"]
    arr = dec["arrivals"]
    gaps = [(arr[i + 1] - arr[i]) * 1000.0 for i in range(len(arr) - 1)]
    pf_end = pf["t0"] + (pf["ttft"] or 0.0)
    during = [x for x in arr if pf["t0"] <= x <= pf_end]
    win = [(arr[i + 1] - arr[i]) * 1000.0 for i in range(len(arr) - 1) if pf["t0"] <= arr[i + 1] <= pf_end]
    r["prefill_wall"] = pf["ttft"]
    r["prefill_tps"] = (pf["timings"] or {}).get("prompt_per_second")
    r["prompt_n"] = (pf["timings"] or {}).get("prompt_n")
    r["during"] = len(during)
    r["itl_max"] = max(gaps) if gaps else 0.0
    r["itl_median"] = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    r["win_median"] = sorted(win)[len(win) // 2] if win else 0.0
    r["win_max"] = max(win) if win else 0.0

    # 2. a lone long prompt on its own (no decoder): the duty must not fire at all
    out = {}
    t = [threading.Thread(target=stream, args=(sentence(a.reps, "LONE"), 2, out, "pf", 0.0, 0))]
    for x in t:
        x.start()
    for x in t:
        x.join()
    r["lone_wall"] = out["pf"]["ttft"]
    r["lone_tps"] = (out["pf"]["timings"] or {}).get("prompt_per_second")

    # 3. one request's greedy text, alone on the server (the purity oracle)
    for p in ("[P1] Explain why the sky is blue in two sentences.",
              "[P2] List the first eight prime numbers, comma separated."):
        d = requests.post(srv.base + "/completion", json={
            "prompt": p, "n_predict": 64, "stream": False, "temperature": 0, "seed": 42,
            "cache_prompt": False}).json()
        c = (d.get("content") or "") + (d.get("reasoning_content") or "")
        r["text"][p[:6]] = hashlib.sha256(c.encode()).hexdigest()[:16]
    return r


def spec_stats(log):
    acc = gen = n = 0
    for m in SPEC_RE.finditer(log):
        acc += int(m.group(2))
        gen += int(m.group(3))
        n += 1
    return (acc, gen, n) if n else None


def width_pass(a, env, log_path, contended):
    """One server, one request shape, no other traffic -- the r306 two-sided width check.

    With NOTHING decoding a step must take the whole `-b` window (r301 capped it at the interactive chunk,
    which cost the field shape half of its uncontended prefill), and a step that DOES carry decode tokens
    must admit at most the chunk.  One prompt slot per pass, so a line's `batch - added` is exactly the
    decode tokens that step already carried -- that is the contention test, and it does not depend on
    guessing when the decoder's first token landed.
    """
    global BASE
    srv = Server(a, env, log_path)
    srv.start()
    try:
        BASE = srv.base
        out = {}
        th = None
        if contended:
            # the decoder must ALREADY be generating when the prompt is assembled, or the first steps are
            # legitimately uncontended (a newcomer's first step is the arrival cost, not a violation)
            th = threading.Thread(target=stream, args=(
                "[DEC] Count slowly from one to five hundred, one number per line.",
                a.dec_n, out, "dec", 0.0, 0))
            th.start()
            time.sleep(1.5)
        stream(sentence(a.reps, "W" + ("C" if contended else "U")), 2, out, "pf", 0.0, 1)
        if th is not None:
            th.join()
        log = srv.text()
    finally:
        srv.stop()

    m = CENSUS_RE.search(log)
    chunk = a.ub
    if m is not None:
        chunk = a.b if m.group(2) == "off" else a.ub if m.group(2) == "n_ubatch" else int(m.group(2))
    # only the PROMPT slot (pinned to 1 above): the decoder's own prefill lines are a prompt too, but mixing
    # them in would mis-read `batch - added` on the step where both are being filled
    rows = [(int(x.group(3)), int(x.group(4))) for x in WIDTH_RE.finditer(log) if int(x.group(1)) == 1]
    return {
        "chunk": chunk,
        "rows": rows,
        "wide": [ad for ad, ba in rows if ba - ad == 0 and ad > chunk],
        "overlap": [1 for ad, ba in rows if ba - ad > 0],
        "violations": [(ad, ba) for ad, ba in rows if ba - ad > 0 and ad > chunk],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.path.expanduser("~/llama.cpp/build-rocm/bin/llama-server"))
    ap.add_argument("--model", default="/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8153)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--b", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--server-args", default="")
    ap.add_argument("--spec-args", default="", help="drafter args; the composition arms run only when given")
    ap.add_argument("--env", default="", help='extra env for BOTH arms, e.g. "MOE_EXPERT_CACHE_MIB=8192"')
    ap.add_argument("--tag", default="")
    ap.add_argument("--duty", type=int, default=16)
    ap.add_argument("--reps", type=int, default=400, help="long prompt repetitions (~14 tokens each)")
    ap.add_argument("--dec-n", type=int, default=400)
    ap.add_argument("--min-decode-gain", type=float, default=3.0)
    ap.add_argument("--inert-band", type=float, default=0.0,
                    help="two-sided INERT band (fraction): a mechanism whose decode gain is within "
                         "[1-band, 1+band] of the baseline reads PASS as INERT (it changed nothing, so it "
                         "cannot fail its own claim) and MUST NOT pay prefill for it.  0 = off (the strict "
                         "one-sided criterion).  Use it where the mechanism is deliberately a no-op -- the "
                         "two-arm server makes the duty cycle redundant, so block 17 runs it on that path "
                         "with --inert-band 0.10")
    ap.add_argument("--max-prefill-stretch", type=float, default=2.5,
                    help="the design bounds the idle time by one prefill step (2x); the slack covers the "
                         "first, still-uncalibrated cycle")
    ap.add_argument("--max-lone-loss", type=float, default=0.03)
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.dec_n = a.dec_n // 2
    if not os.access(a.server, os.X_OK):
        sys.exit(f"gate: no executable llama-server at {a.server}")
    if not os.path.exists(a.model):
        sys.exit(f"gate: no model at {a.model}")

    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    logdir = os.path.join(HERE, "runs", "gate-prefill-duty")
    os.makedirs(logdir, exist_ok=True)
    tag = f"{a.tag + '.' if a.tag else ''}"

    shared = {}
    for kv in filter(None, a.env.split(",")):
        k, _, v = kv.partition("=")
        shared[k.strip()] = v.strip()
    arms = [("baseline", dict(shared, LLAMA_SERVER_PREFILL_DUTY="0"), 1),
            ("candidate", dict(shared, LLAMA_SERVER_PREFILL_DUTY=str(a.duty)), a.duty)]
    res, logs = {}, {}
    for name, env, want_duty in arms:
        log_path = os.path.join(logdir, f"{tag}{name}.log")
        print(f"\n=== arm {name}: DUTY={env['LLAMA_SERVER_PREFILL_DUTY']} {a.server_args} {a.spec_args}"
              f" -> {log_path}")
        srv = Server(a, env, log_path)
        try:
            srv.start()
            res[name] = measure(srv, a)
            log = srv.text()
            m = CENSUS_RE.search(log)
            res[name]["census"] = m.groups() if m else None
            res[name]["empty"] = log.count("no tokens to decode")
            res[name]["spec"] = spec_stats(log)
            res[name]["spec_on"] = "adding speculative implementation" in log
            res[name]["want_duty"] = want_duty
        finally:
            srv.stop()
        logs[name] = log_path
        r = res[name]
        print(f"    prefill wall {r['prefill_wall']:.2f}s ({r['prefill_tps'] or 0:.0f} t/s, "
              f"{r['prompt_n']} tokens)  decode tokens during it {r['during']}  "
              f"decoder ITL med/max {r['itl_median']:.0f}/{r['itl_max']:.0f}ms  "
              f"win med/max {r['win_median']:.0f}/{r['win_max']:.0f}ms  lone {r['lone_tps']:.0f} t/s")

    # the r306 width passes, on the candidate env, each in its own server so the log is unambiguous
    width = {}
    for tag, cont in (("width.uncontended", False), ("width.contended", True)):
        log_path = os.path.join(logdir, f"{tag}.log")
        print(f"\n=== width pass {tag} -> {log_path}")
        width[cont] = width_pass(a, arms[1][1], log_path, cont)
        w = width[cont]
        print(f"    chunk {w['chunk']}, {len(w['rows'])} prompt step(s), "
              f"added {[ad for ad, _ in w['rows']]}")

    b, c = res["baseline"], res["candidate"]
    fails, warns = [], []

    def hard(ok, name, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    def soft(ok, name, detail):
        print(f"  [{'PASS' if ok else 'WARN'}] {name}: {detail}")
        if not ok:
            warns.append(name)

    print("\n=== checks")
    gain = (c["during"] / b["during"]) if b["during"] else float("inf")
    stretch = (c["prefill_wall"] / b["prefill_wall"]) if b["prefill_wall"] else float("inf")
    # TWO-SIDED (r346, block 17): a mechanism that is deliberately a no-op on this path must be able to read
    # PASS.  On the two-arm server the decoder has its own context and thread, so there is nothing for a duty
    # cycle to win: the honest outcome is "no change", and a one-sided ">= 3x" criterion calls that a failure
    # of a CORRECT configuration.  The band is symmetric because BOTH directions matter -- a duty cycle that
    # *lost* tokens on this path would be a defect, and prefill must not pay for an inert mechanism.
    if a.inert_band > 0 and abs(gain - 1.0) <= a.inert_band:
        hard(stretch <= 1.0 + a.inert_band, "decode burst (INERT)",
             f"{b['during']} -> {c['during']} tokens ({gain:.2f}x) is inside the +-{a.inert_band:.0%} inert "
             f"band: the mechanism changed nothing here (deprecated on this path) and the prefill must not "
             f"pay for it")
        warns.append("inert")
    else:
        hard(gain >= a.min_decode_gain, "decode burst",
             f"{b['during']} -> {c['during']} tokens delivered during the other slot's prefill ({gain:.1f}x, "
             f"need >= {a.min_decode_gain:.1f}x or inside the inert band)")

    hard(stretch <= a.max_prefill_stretch, "prefill stretch",
         f"{b['prefill_wall']:.2f} -> {c['prefill_wall']:.2f} s ({stretch:.2f}x, bound "
         f"{a.max_prefill_stretch:.1f}x -- the idle time is capped by one prefill step)")

    lone = (c["lone_tps"] / b["lone_tps"]) if b["lone_tps"] else 0.0
    hard(lone >= 1.0 - a.max_lone_loss, "lone prefill",
         f"{b['lone_tps']:.0f} -> {c['lone_tps']:.0f} t/s ({100 * (lone - 1):+.1f} %, limit "
         f"-{100 * a.max_lone_loss:.0f} %; a prompt with nothing decoding must not be deferred)")

    hard(c["text"] == b["text"], "single-request text", f"{c['text']} vs {b['text']} (must be identical)")

    ok_census = (b["census"] is not None and c["census"] is not None
                 and int(b["census"][3]) == 1 and int(c["census"][3]) == a.duty)
    hard(ok_census, "census", f"baseline {b['census']}, candidate {c['census']} (want duty 1 / {a.duty})")

    hard(c["empty"] == 0, "no empty batch",
         f"{c['empty']} 'no tokens to decode' line(s) in the candidate arm (the deferral must never fire "
         f"into an empty batch)")
    aborts = [n for n, p in logs.items() if ABORT_RE.search(open(p, errors="replace").read())]
    hard(not aborts, "no abort", "clean" if not aborts else f"abort in {aborts}")

    wu, wc = width[False], width[True]
    if wu["chunk"] < a.b:
        hard(bool(wu["wide"]), "uncontended window",
             f"{-1 if not wu['wide'] else len(wu['wide'])} step(s) wider than the chunk {wu['chunk']} with "
             f"nothing decoding (r301 capped every step; a wide step is the whole -b window)")
    else:
        soft(True, "uncontended window", f"chunk {wu['chunk']} == -b, nothing to widen")
    hard(not wc["violations"], "contended width",
         f"{len(wc['violations'])} step(s) admitted more than the chunk {wc['chunk']} while carrying decode "
         f"tokens {wc['violations'][:4]}{' (the cap must bound a waiting decoder)' if wc['violations'] else ''}"
         f", over {len(wc['overlap'])} contended step(s)")
    if not wc["overlap"]:
        warns.append("contended width not reached")
        print("  [WARN] contended width not reached: no step carried decode tokens; the assertion was vacuous")

    soft(c["itl_max"] <= b["itl_max"] * 1.1, "decoder worst gap",
         f"{b['itl_max']:.0f} -> {c['itl_max']:.0f}ms (the burst's last gap is one prefill step, so it may "
         f"not be worse than the baseline's)")

    if a.spec_args:
        hard(b["spec_on"] and c["spec_on"], "spec active", f"baseline {b['spec_on']}, candidate {c['spec_on']}")
        if b["spec"] and c["spec"]:
            br, cr = b["spec"][0] / b["spec"][1], c["spec"][0] / c["spec"][1]
            hard(cr >= 0.5 * br, "spec composition",
                 f"acceptance {br:.5f} ({b['spec'][0]}/{b['spec'][1]}) -> {cr:.5f} "
                 f"({c['spec'][0]}/{c['spec'][1]}) -- judged for collapse, not equality (§63); the decode "
                 f"gain above already holds under the drafter")

    faults = os.path.join(REPO, "scripts", "check-gpu-faults.sh")
    if os.access(faults, os.X_OK):
        args = [faults, "--quiet", "--since", t_start] + sum([["--log", p] for p in logs.values()], [])
        rc = subprocess.run(args, capture_output=True, text=True).returncode
        hard(rc == 0, "GPU faults", "clean" if rc == 0 else "check-gpu-faults.sh reported a fault")
    else:
        warns.append("GPU faults")
        print(f"  [WARN] GPU faults: no {faults}; not checked")

    print(f"\n=== {'PASS' if not fails else 'FAIL'} ({len(fails)} hard failure(s), {len(warns)} warning(s))")
    if not a.keep and not fails:
        import shutil
        shutil.rmtree(logdir, ignore_errors=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(2)
