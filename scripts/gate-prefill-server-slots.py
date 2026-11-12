#!/usr/bin/env python3
"""Drive the two-slot decoder-vs-prefill scenario for `gate-prefill-server-slots.sh` (stdlib only).

Why it is a separate helper: the measurement is *concurrent streaming*, which a shell script cannot do.  It
issues request A (a long decode) against `llama-server -np 2`, waits `--slot-wait` seconds, then issues
request B (the ~250-token prompt) so that B's PREFILL lands in the middle of A's DECODE, and reports:

  a_sha        A's generated text hash -- it must be identical across arms (a placement policy never changes
               WHICH bytes are computed)
  a_tps        A's decode rate over the whole request
  a_win_tps    A's decode rate inside B's lifetime (the interference window): an average can hide a stall
  a_max_gap_ms A's longest single inter-token gap (the stall instrument)
  b_*          B's latency and text hash (B has to survive too)

Everything is greedy (`temperature 0`, `seed 42`, `cache_prompt false`) so the text is a per-arm reference.

Usage: gate-prefill-server-slots.py --port 8137 --a-prompt FILE --b-prompt FILE --n-predict 160 --out a.json
"""

import argparse
import hashlib
import json
import threading
import time
import urllib.request


def stream_completion(port, prompt, n_predict, timeout=3600):
    """One streaming /completion -> (text, [arrival monotonic times], t_start, t_end)."""
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        "cache_prompt": False,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % port, body,
                                 {"Content-Type": "application/json"})
    out = []
    stamps = []
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            piece = obj.get("content")
            if piece:
                out.append(piece)
                stamps.append(time.monotonic())
    return "".join(out), stamps, t0, time.monotonic()


def rate(stamps, t_from, t_to):
    n = sum(1 for s in stamps if t_from <= s <= t_to)
    return n / max(1e-9, t_to - t_from)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--a-prompt", required=True)
    ap.add_argument("--b-prompt", required=True)
    ap.add_argument("--n-predict", type=int, default=160)
    ap.add_argument("--n-predict-b", type=int, default=8)
    ap.add_argument("--slot-wait", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.a_prompt, "r", encoding="utf-8", errors="replace") as f:
        prompt_a = f.read()
    with open(a.b_prompt, "r", encoding="utf-8", errors="replace") as f:
        prompt_b = f.read()

    # A starts; B is fired `slot_wait` seconds later so its PREFILL overlaps A's DECODE.  `a_res` is filled by
    # the thread; the main thread owns B, so B's window is measured from the caller's own clock.
    a_res = {}

    def run_a():
        a_res["v"] = stream_completion(a.port, prompt_a, a.n_predict)

    th = threading.Thread(target=run_a)
    th.start()
    time.sleep(a.slot_wait)
    b_text, b_stamps, b_t0, b_t1 = stream_completion(a.port, prompt_b, a.n_predict_b)
    th.join()

    text_a, stamps_a, t_a0, t_a1 = a_res["v"]
    gaps = [stamps_a[i + 1] - stamps_a[i] for i in range(len(stamps_a) - 1)] or [0.0]
    out = {
        "a_sha": hashlib.sha256(text_a.encode()).hexdigest()[:12],
        "a_chars": len(text_a),
        "a_tokens": len(stamps_a),
        "a_tps": rate(stamps_a, t_a0, t_a1),
        "a_win_tps": rate(stamps_a, b_t0, b_t1),
        "a_max_gap_ms": 1000.0 * max(gaps),
        "b_sha": hashlib.sha256(b_text.encode()).hexdigest()[:12],
        "b_chars": len(b_text),
        "b_secs": b_t1 - b_t0,
    }
    print("A: %d tok %.2f tok/s (in B's window %.2f tok/s, longest gap %.0f ms) sha=%s"
          % (out["a_tokens"], out["a_tps"], out["a_win_tps"], out["a_max_gap_ms"], out["a_sha"]))
    print("B: %d chars in %.1f s sha=%s" % (out["b_chars"], out["b_secs"], out["b_sha"]))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
