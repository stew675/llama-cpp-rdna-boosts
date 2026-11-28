#!/usr/bin/env python3
# save-only probe: A (long), then B (short) on the same slot -> the server saves A's state to the
# prompt cache.  One save per run, at a chosen state size.
import json, sys, time, hashlib, urllib.request
port, srcf, nch = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
src = open(srcf, encoding="utf-8", errors="replace").read()
A = "Summarise this source file in one sentence.\n\n" + src[:nch]
B = "Reply with the single word OK."
def req(prompt, n):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": True, "id_slot": 0}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
    t = r["timings"]
    return (time.time() - t0) * 1000, t["prompt_n"], t["prompt_ms"], hashlib.sha256(r["content"].encode()).hexdigest()[:12]
for name, p, n in (("A", A, 8), ("B", B, 8)):
    dt, pn, pms, sha = req(p, n)
    print("%-3s wall %8.1f ms | prompt %6d tok %8.1f ms | sha %s" % (name, dt, pn, pms, sha), flush=True)
