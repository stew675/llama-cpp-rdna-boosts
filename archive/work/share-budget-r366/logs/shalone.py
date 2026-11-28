#!/usr/bin/env python3
# shalone.py PORT: the four shrep.py prompts one at a time (fresh, n_predict 16 greedy): reply sha per prompt.
import hashlib, json, random, sys, time, urllib.request
port = sys.argv[1]
A_ = "pump valve filter gauge sensor relay bearing rotor stator flange gasket impeller coupling manifold actuator solenoid".split()
B_ = "inspected replaced calibrated cleaned tightened lubricated tested rebuilt adjusted logged flagged measured".split()
def text(seed, n):
    R = random.Random(seed)
    return " ".join("Entry %d: the %s at station %d was %s by crew %d on day %d, reading %.2f." % (i, R.choice(A_), R.randrange(9000), R.choice(B_), R.randrange(97), R.randrange(365), R.random()*500) for i in range(n)) + "\nSummarize the entries above in one sentence."
for name, seed, n in (("P1", 71, 1040), ("P2", 72, 910), ("Q1", 81, 1560), ("Q2", 82, 1560)):
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=json.dumps({"prompt": text(seed, n), "n_predict": 16, "temperature": 0, "top_k": 1, "cache_prompt": True}).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
    t = r["timings"]
    print("%-3s alone | prompt %6d tok @ %5.0f t/s | %5.1f s | sha %s" % (name, t["prompt_n"], t["prompt_per_second"], time.time() - t0, hashlib.sha256(r["content"].encode()).hexdigest()[:8]), flush=True)
