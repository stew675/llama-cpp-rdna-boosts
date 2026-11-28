#!/usr/bin/env python3
# shrep.py PORT: two prompts prefilled at once (fresh), n_predict 16 greedy: pair 1 ~40K + ~35K, pair 2 ~60K + ~60K.
# Per request: end time, prompt tokens and t/s, reply sha; pair wall time.
import hashlib, json, random, sys, threading, time, urllib.request
port = sys.argv[1]
A_ = "pump valve filter gauge sensor relay bearing rotor stator flange gasket impeller coupling manifold actuator solenoid".split()
B_ = "inspected replaced calibrated cleaned tightened lubricated tested rebuilt adjusted logged flagged measured".split()
def text(seed, n):
    R = random.Random(seed)
    return " ".join("Entry %d: the %s at station %d was %s by crew %d on day %d, reading %.2f." % (i, R.choice(A_), R.randrange(9000), R.choice(B_), R.randrange(97), R.randrange(365), R.random()*500) for i in range(n)) + "\nSummarize the entries above in one sentence."
def one(name, p, out):
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=json.dumps({"prompt": p, "n_predict": 16, "temperature": 0, "top_k": 1, "cache_prompt": True}).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
    t = r["timings"]
    out[name] = "%-3s end %6.1f s | prompt %6d tok @ %5.0f t/s | sha %s" % (name, time.time() - T0, t["prompt_n"], t["prompt_per_second"], hashlib.sha256(r["content"].encode()).hexdigest()[:8])
for title, jobs in (("pair 1 (~40K + ~35K)", [("P1", text(71, 1040)), ("P2", text(72, 910))]),
                    ("pair 2 (~60K + ~60K)", [("Q1", text(81, 1560)), ("Q2", text(82, 1560))])):
    out = {}; T0 = time.time()
    th = [threading.Thread(target=one, args=(n, p, out)) for n, p in jobs]
    for x in th: x.start()
    for x in th: x.join()
    print("== %s: wall %.1f s" % (title, time.time() - T0))
    for n, _ in jobs: print("   " + out[n])
    sys.stdout.flush()
