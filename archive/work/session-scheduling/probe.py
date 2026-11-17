#!/usr/bin/env python3
"""Session-scheduling probe: the five arms of wip/session-scheduling/README.md §2.

Measures the *existing* `llama-server` admission behaviour.  It changes nothing and needs no
privileges; it only needs a running `llama-server` (see README §2 for the invocation) and `requests`.

    ./probe.py --port 8099 decode     # 2.1  decode is batched: merged token arrival timestamps
    ./probe.py --port 8099 prefill    # 2.2/2.3 prompt co-batching vs serialisation
    ./probe.py --port 8099 starve     # 2.4 admission order decides a 7-token prompt's TTFT
    ./probe.py --port 8099 itl        # 2.5 a decoder's ITL while another slot prefills
    ./probe.py --port 8099 tps        # 2.6 aggregate throughput at concurrency 1 and 2
    ./probe.py --port 8099 agg        # 2.7 co-batched prefill (server at -b 16384)
    ./probe.py --port 8099 skew       # 4.2.2 a heterogeneous prompt mix: who finishes first, and when
                                      #       (needs -np 4; compare LLAMA_SERVER_PREFILL_SKEW=1 vs 5 vs 20)

Every arm is greedy (temperature 0, seed 42) with `cache_prompt: false`, because the server's prompt cache
would otherwise make the second request instant and hide the effect.  Prompts are synthetic (a repeated
sentence, ~11 tokens/repetition) so the token counts are exact; a gate should use `prompts/` instead.
"""

import argparse
import json
import sys
import threading
import time

import requests

BASE = ""


def toklen(text):
    return len(requests.post(BASE + "/tokenize", json={"content": text}).json()["tokens"])


def sentence(n_rep, tag=""):
    return (f"[{tag}] " if tag else "") + "The history of computing is a long and tangled story. " * n_rep


def stream(prompt, n_predict, out, key, delay=0.0, id_slot=None):
    """One streaming completion.  Records (token arrival times, final timings).

    `id_slot` PINS the slot.  Slot assignment is LRU/similarity-based and the fill policy gives the LOWEST
    slot index priority, so an unpinned starvation arm measures whichever side of the race it happens to get:
    measured 1.907 s pinned (long in slot 0) vs 0.828 s unpinned (the long prompt landing in slot 1).
    """
    time.sleep(delay)
    t0 = time.time()
    ev = []
    timings = None
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": True,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
    }
    if id_slot is not None:
        body["id_slot"] = id_slot
    with requests.post(BASE + "/completion", json=body, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode()
            if not s.startswith("data: "):
                continue
            d = json.loads(s[6:])
            if d.get("content"):
                ev.append(round(time.time() - t0, 3))
            if d.get("stop"):
                timings = d.get("timings")
                break
    out[key] = (ev, timings, round(time.time() - t0, 3))


def run_pair(arms):
    out = {}
    th = [threading.Thread(target=stream, args=a) for a in arms]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def arm_decode(reps):
    p = sentence(reps, "D")
    out = {}
    run_pair([(p, 40, out, "A"), (p, 40, out, "B")])
    merged = sorted((t, k) for k in out for t in out[k][0])
    print("tokens per stream:", {k: len(v[0]) for k, v in out.items()})
    print("wall:", max(v[2] for v in out.values()))
    print("first 30 arrivals:", " ".join(f"{k}@{t}" for t, k in merged[:30]))
    print("\ninterleaved arrivals at identical instants == one forward pass per step (batched decode).")


def arm_prefill(reps):
    p = sentence(reps)
    n = toklen(p)
    print(f"prompt tokens: {n} each, solo arm first")
    out = {}
    stream(p, 4, out, "solo")
    print("  solo   TTFT/total:", out["solo"][0][:1], out["solo"][2],
          "prompt_ms:", round((out["solo"][1] or {}).get("prompt_ms", 0)))
    out = {}
    run_pair([(sentence(reps, "A"), 4, out, "A"), (sentence(reps, "B"), 4, out, "B")])
    for k in ("A", "B"):
        t = out[k][1] or {}
        print(f"  2x     {k}: TTFT {out[k][0][:1]} total {out[k][2]} "
              f"prompt_ms {round(t.get('prompt_ms', 0))} pps {round(t.get('prompt_per_second', 0))}")
    print("\nBoth TTFTs near the solo value == co-batched.  Second TTFT ~= 2x solo == serialised.")


def arm_starve(reps, delay):
    long_p = sentence(reps, "LONG")
    short_p = "[SHORT] Name three planets."
    print(f"long {toklen(long_p)} tokens, short {toklen(short_p)} tokens, second request +{delay}s")
    for label, order in (("long admitted first", ("L", "S")), ("short admitted first", ("S", "L"))):
        out = {}
        p = {"L": long_p, "S": short_p}
        # the second entry is delayed so the admission order is deterministic
        # pin: the first-admitted request gets slot 0 (the priority slot), the other slot 1
        arms = [(p[order[0]], 4, out, order[0], 0.0, 0),
                (p[order[1]], 4, out, order[1], delay, 1)]
        run_pair(arms)
        s = out["S"][1] or {}
        print(f"  {label:22s} long TTFT {out['L'][0][:1]}  short TTFT {out['S'][0][:1]} "
              f"short prompt_ms {round(s.get('prompt_ms', 0))} pps {round(s.get('prompt_per_second', 0))}")
    print("\nshort-first TTFT ~1 step and long-first ~the long prompt's prefill == slot-order admission.")
    print("Re-run with a smaller -b on the server: the short TTFT does NOT improve.")


def arm_itl(reps, delay, n_predict):
    long_p = sentence(reps, "LONG")
    dec_p = "[DEC] Count slowly from one to five hundred, one number per line."
    out = {}
    run_pair([(dec_p, n_predict, out, "dec", 0.0, 0), (long_p, 2, out, "pf", delay, 1)])
    ev = out["dec"][0]
    gaps = [round(ev[i + 1] - ev[i], 3) for i in range(len(ev) - 1)]
    if not gaps:
        print("decoder produced too few tokens; raise -n")
        return
    print(f"decoder tokens: {len(ev)}  median gap: {sorted(gaps)[len(gaps) // 2]}s")
    big = [(i, ev[i], g) for i, g in enumerate(gaps) if g > 0.05]
    print("gaps > 50 ms (index, t, gap):", big[:12])
    print(f"total decode wall: {ev[-1]}s")


def arm_tps(reps, n_predict):
    p = sentence(reps, "T")
    for k in (1, 2):
        out = {}
        th = [threading.Thread(target=stream, args=(p, n_predict, out, i)) for i in range(k)]
        t0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - t0
        print(f"  concurrency {k}: wall {wall:.2f}s  aggregate {k * n_predict / wall:.1f} t/s")
    print("\n1.4-1.5x at concurrency 2 on one GPU is the bandwidth wall, not a scheduling defect.")


def arm_agg():
    """2.7: is co-batching two prompt sequences a cost or a saving?  Needs -b 16384 on the server."""
    def one(label, prompts):
        out = {}
        th = [threading.Thread(target=stream, args=(p, 2, out, i)) for i, p in enumerate(prompts)]
        t0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - t0
        rows = sum(((v[1] or {}).get("prompt_n", 0)) for v in out.values())
        pms = [round((v[1] or {}).get("prompt_ms", 0)) for v in out.values()]
        print(f"  {label:34s} rows={rows:6d} wall={wall:5.2f}s aggregate={rows / wall:7.1f} t/s  "
              f"prompt_ms={pms}")

    one("1 x 8803-token prompt", [sentence(800, "A")])
    one("2 x 4403-token prompts", [sentence(400, "A"), sentence(400, "B")])
    one("1 x 4403-token prompt", [sentence(400, "C")])
    one("2 x 2203-token prompts", [sentence(200, "C"), sentence(200, "D")])
    print("\nEach request ~1.87x its solo time (that is the work), and the aggregate is NEUTRAL within "
          "run-to-run spread: a second sequence in the batch costs no throughput.")


def arm_skew():
    """4.2.2: four prompts of different sizes, all at once.  Prints each request's time-to-first-token and
    completion, so a skew policy can be judged on who benefits and whether anyone is starved."""
    sizes = [545, 272, 136, 36]   # ~6000 / 3000 / 1500 / 400 tokens
    prompts = [sentence(r) for r in sizes]
    print("    prompt tokens:", [toklen(p) for p in prompts])
    out = {}
    th = [threading.Thread(target=stream, args=(p, 4, out, i, 0.0, i)) for i, p in enumerate(prompts)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print("    size   slot   ttft(s)   done(s)")
    for i, p in enumerate(prompts):
        ev, _, wall = out[i]
        print(f"    {toklen(p):5d}  {i:4d}   {ev[0]:7.3f}   {wall:7.3f}")
    n = len(prompts)
    print(f"    mean ttft {sum(out[i][0][0] for i in range(n)) / n:.3f}s   "
          f"mean done {sum(out[i][2] for i in range(n)) / n:.3f}s   "
          f"max done {max(out[i][2] for i in range(n)):.3f}s")
    print("\n    starvation check: every request must appear above with a finite ttft; a 0.000 or a missing")
    print("    row means a slot was denied admission.")


def main():
    global BASE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=["decode", "prefill", "starve", "itl", "tps", "agg", "skew"])
    ap.add_argument("--port", default="8099")
    ap.add_argument("--reps", type=int, default=1200, help="sentence repetitions (11 tokens each)")
    ap.add_argument("--delay", type=float, default=0.05, help="delay of the second request, seconds")
    ap.add_argument("--n", type=int, default=200, help="decode tokens for the itl/tps arms")
    a = ap.parse_args()
    BASE = f"http://127.0.0.1:{a.port}"

    try:
        requests.get(BASE + "/health", timeout=2).raise_for_status()
    except Exception:
        sys.exit(f"no llama-server on {BASE} (see the README's invocation)")

    if a.arm == "decode":
        arm_decode(12)
    elif a.arm == "prefill":
        arm_prefill(a.reps)
    elif a.arm == "starve":
        arm_starve(a.reps, a.delay)
    elif a.arm == "itl":
        arm_itl(a.reps, a.delay, a.n)
    elif a.arm == "tps":
        arm_tps(300, a.n)
    elif a.arm == "agg":
        arm_agg()
    elif a.arm == "skew":
        arm_skew()


if __name__ == "__main__":
    main()
