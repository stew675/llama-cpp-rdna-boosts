#!/usr/bin/env python3
"""gate16.py - the gfx1151 masked-column sign-leak determinism gate.

16 sequential identical greedy /completion requests against ONE running
llama-server (cache_prompt on, so freed cells from run N are read-masked
during run N+1). Any (run,pos) difference in the generated token stream or
the per-position top-8 logprobs (compared at full JSON/float64 precision)
fails the gate.

Usage: gate16.py <base_url> <prompt_file> <n_predict> <out_dir>
  base_url    e.g. http://127.0.0.1:8080
  prompt_file text file with the (raw) prompt
  n_predict   tokens per request (e.g. 129); a second 512-token pass uses n_predict 512
  out_dir     per-run JSON snapshots + a verdict file are written here
"""
import json, os, sys, urllib.request

BASE = sys.argv[1].rstrip('/')
PROMPT_F = sys.argv[2]
N_PREDICT = int(sys.argv[3])
OUT = sys.argv[4]
N_RUNS = 16
TOP_K = 8

os.makedirs(OUT, exist_ok=True)
with open(PROMPT_F) as f:
    prompt = f.read()

def post(path, body):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def probe_ok():
    try:
        with urllib.request.urlopen(BASE + '/health', timeout=5) as r:
            return r.status == 200
    except Exception:
        return False

def req(run):
    body = {
        'prompt': prompt,
        'n_predict': N_PREDICT,
        'temperature': 0.0,
        'top_k': 1,
        'cache_prompt': True,
        'n_probs': TOP_K,
        'return_tokens': True,
    }
    return post('/completion', body)

def snap_key(pos, entry):
    """A run-position identity: (top-8 ids+logprobs at full precision)."""
    toks = []
    for t in entry.get('top_logprobs', []):
        toks.append((t['id'], repr(t['logprob'])))
    return json.dumps(toks, sort_keys=True)

def main():
    if not probe_ok():
        print('server not healthy at', BASE); sys.exit(2)
    runs = []
    for i in range(1, N_RUNS + 1):
        r = req(i)
        probs = r.get('completion_probabilities') or r.get('probs') or []
        tokens = r.get('tokens') or []
        # probs may exclude the final position(s); use its own length for the compare grid
        grid = []
        for pos, entry in enumerate(probs):
            grid.append(snap_key(pos, entry))
        runs.append({'run': i, 'tokens': tokens, 'grid': grid,
                     'pn': (r.get('timings') or {}).get('prompt_n')})
        with open(os.path.join(OUT, f'run{i:02d}.json'), 'w') as f:
            json.dump({'tokens': tokens, 'probs': probs}, f, indent=1)
        print(f'run {i:2d}: prompt_n={runs[-1]["pn"]} ntok={len(tokens)} nprobs={len(grid)}', flush=True)

    # grid length should equal n_predict (or n_predict-1 if the final token is the stop)
    print(f'grid lengths: {sorted(set(len(r["grid"]) for r in runs))}', flush=True)

    diffs = 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            if a['tokens'] != b['tokens'] or a['grid'] != b['grid']:
                for pos, (ta, tb) in enumerate(zip(a['tokens'], b['tokens'])):
                    if ta != tb:
                        print(f'  DIFF token {pos}: run{i+1}={ta} run{j+1}={tb}')
                for pos, (ga, gb) in enumerate(zip(a['grid'], b['grid'])):
                    if ga != gb:
                        print(f'  DIFF probs {pos}: run{i+1}={ga[:120]}... run{j+1}={gb[:120]}...')
                diffs += 1
                if diffs <= 3:
                    print(f'  mismatched pair: run {i+1} vs run {j+1}')
    n_pairs = N_RUNS * (N_RUNS - 1) // 2
    verdict = 'PASS' if diffs == 0 else f'FAIL ({diffs}/{n_pairs} pairs differ)'
    print(f'VERDICT: {verdict}')
    with open(os.path.join(OUT, 'verdict.txt'), 'w') as f:
        f.write(verdict + '\n')
    return 0 if diffs == 0 else 1

sys.exit(main())
