#!/usr/bin/env python3
# agpar.py PORT LOG (FINDINGS: two long agents, parallel vs taking turns).  Per arm: agents A and B build their context
# (cold, one after the other), then 3 rounds in which both send their next turn at once (+40 entries ~1.5K tokens, 256 gen).
# Arm "par": base 3140 entries (~121K -> ~125.5K, both fit 256K); arm "turn": 3375 entries (~130K, they cannot both fit).
import hashlib, json, random, re, sys, threading, time, urllib.request
port, log = sys.argv[1], sys.argv[2]
A_ = "pump valve filter gauge sensor relay bearing rotor stator flange gasket impeller coupling manifold actuator solenoid".split()
B_ = "inspected replaced calibrated cleaned tightened lubricated tested rebuilt adjusted logged flagged measured".split()
def text(seed, n):
    R = random.Random(seed)
    return " ".join("Entry %d: the %s at station %d was %s by crew %d on day %d, reading %.2f." % (i, R.choice(A_), R.randrange(9000), R.choice(B_), R.randrange(97), R.randrange(365), R.random()*500) for i in range(n)) + "\nSummarize the entries above in one sentence."
def logpos(): return sum(1 for _ in open(log, errors="replace"))
def newl(p): return [l for i, l in enumerate(open(log, errors="replace")) if i >= p]
def one(name, p, ngen, out, T0):
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=json.dumps({"prompt": p, "n_predict": ngen, "temperature": 0, "top_k": 1, "cache_prompt": True}).encode(), headers={"Content-Type": "application/json"}), timeout=3600))
        t = r["timings"]; e = time.time() - T0
        out[name] = (e, "%s done %6.1f s | ctx %6d, prefilled %6d @ %5.0f t/s (%5.1f s) | gen %3d @ %5.1f t/s | sha %s" % (name, e, r.get("tokens_evaluated", 0), t["prompt_n"], t["prompt_per_second"], t["prompt_ms"] / 1000, t["predicted_n"], t["predicted_per_second"], hashlib.sha256(r["content"].encode()).hexdigest()[:8]))
    except Exception as ex:
        out[name] = (time.time() - T0, "%s ERROR %s" % (name, ex))
def run(title, jobs):
    p0 = logpos(); out = {}; T0 = time.time()
    th = [threading.Thread(target=one, args=(n, p, g, out, T0)) for n, p, g in jobs]
    for x in th: x.start(); time.sleep(0.1)
    for x in th: x.join()
    w = time.time() - T0
    print("== %s: wall %.1f s" % (title, w))
    for n, _, _ in jobs: print("   " + out[n][1])
    ev = [l for l in newl(p0) if re.search(r"KV admission|prompt cache (save|restore|load)|restored|making room|error|exceed", l, re.I) and "ROCm pool" not in l]
    for l in ev[:8]: print("      log: " + re.sub(r"^[0-9.]+ [A-Z] ", "", l.strip())[:170])
    sys.stdout.flush(); return w
for arm, base in (("par", 3140), ("turn", 3375)):
    print("##### arm %s: base %d entries" % (arm, base))
    run("%s build A" % arm, [("A", text(101, base), 16)])
    run("%s build B" % arm, [("B", text(202, base), 16)])
    tot = 0
    for k in range(1, 4):
        n = base + 40 * k
        tot += run("%s round %d (both at once)" % (arm, k), [("A", text(101, n), 256), ("B", text(202, n), 256)])
    print("##### arm %s: 3 rounds total %.1f s" % (arm, tot)); sys.stdout.flush()
