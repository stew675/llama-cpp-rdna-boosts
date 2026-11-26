#!/usr/bin/env python3
# rndrep.py PORT SRC: prompt-cache save + load of long states on one slot (stock r366 repro of the state-copy rounds
# patch): A (long), B (another long prompt on the same slot: A is saved to the prompt cache), A again (restored), B again.
# Per request: wall time, prompt tokens evaluated, greedy sha of a 32-token reply.
import json, sys, time, hashlib, urllib.request
port, srcf = sys.argv[1], sys.argv[2]; nch = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
src = open(srcf).read()
A = "Summarise this source file in one sentence.\n\n" + src[:nch]
B = "List three functions defined in this file.\n\n" + src[nch:2*nch]
def req(prompt):
    body = {"prompt": prompt, "n_predict": 32, "temperature": 0, "top_k": 1, "cache_prompt": True, "id_slot": 0}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
    t = r["timings"]
    return (time.time() - t0) * 1000, t["prompt_n"], t["prompt_ms"], hashlib.sha256(r["content"].encode()).hexdigest()[:12]
for name, p in (("A", A), ("B", B), ("A again", A), ("B again", B)):
    dt, pn, pms, sha = req(p)
    print("%-8s wall %7.1f ms | prompt %5d tok %7.1f ms | sha %s" % (name, dt, pn, pms, sha), flush=True)
