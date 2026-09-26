#!/usr/bin/env python3
"""MTP decode benchmark: drives an already-running llama-server.

Usage: bench_mtp.py --port 18080 [--reps 5] [--predict 256] [--prompt "..."]
                    [--warmup 2]
Prints per-rep predicted tok/s, draft stats, and the median (as the reporter does).
"""
import argparse, json, statistics, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=18080)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--predict", type=int, default=256)
ap.add_argument("--prompt", default="Write a detailed technical essay about the history of the Roman Empire from its founding to its fall.")
ap.add_argument("--cache-prompt", type=int, default=1)
args = ap.parse_args()

url = f"http://{args.host}:{args.port}/completion"
rates, accs = [], []
for i in range(args.reps + args.warmup):
    body = json.dumps({
        "prompt": args.prompt,
        "n_predict": args.predict,
        "temperature": 0.0,
        "cache_prompt": bool(args.cache_prompt),
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    t = resp.get("timings", {})
    pps = t.get("predicted_per_second", 0.0)
    dn = t.get("draft_n", 0)
    dna = t.get("draft_n_accepted", 0)
    acc = (dna / dn) if dn else float("nan")
    tag = "warm" if i < args.warmup else "meas"
    print(f"  [{tag} {i+1}] {pps:.2f} tok/s  draft_n={dn} accepted={dna} acc={acc:.4f}  prompt_n={t.get('prompt_n')}", flush=True)
    if i >= args.warmup:
        rates.append(pps)
        accs.append(acc)

med = statistics.median(rates)
print(f"MEDIAN {med:.2f} tok/s  (n={len(rates)}, min={min(rates):.2f}, max={max(rates):.2f})")
print(f"ACC    {statistics.mean(accs):.4f} (mean over measured reps)")
