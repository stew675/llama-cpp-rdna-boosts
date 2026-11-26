#!/usr/bin/env python3
# kfsrep.py PORT LOG: KV full stop repro at -c 65536 on stock r366 (no MTP): two ~12K requests at once (fit; output
# must not change), two ~28K requests at once with long forced replies (ignore_eos, n_predict 8000) that are both
# admitted (LLAMA_SERVER_KV_ADMIT_RESERVE=512) and overflow the cache mid-reply, then one short request.
import hashlib, json, random, re, sys, threading, time, urllib.request, urllib.error
port, log = sys.argv[1], sys.argv[2]
A_ = "pump valve filter gauge sensor relay bearing rotor stator flange gasket impeller coupling manifold actuator solenoid".split()
B_ = "inspected replaced calibrated cleaned tightened lubricated tested rebuilt adjusted logged flagged measured".split()
def text(seed, n, q):
    R = random.Random(seed)
    return " ".join("Entry %d: the %s at station %d was %s by crew %d on day %d, reading %.2f." % (i, R.choice(A_), R.randrange(9000), R.choice(B_), R.randrange(97), R.randrange(365), R.random()*500) for i in range(n)) + "\n" + q
def logpos(): return sum(1 for _ in open(log, errors="replace"))
def newl(p): return [l for i, l in enumerate(open(log, errors="replace")) if i >= p]
def one(name, p, ngen, ign, out):
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=json.dumps({"prompt": p, "n_predict": ngen, "temperature": 0, "top_k": 1, "cache_prompt": True, "ignore_eos": ign}).encode(), headers={"Content-Type": "application/json"}), timeout=3600))
        t = r["timings"]
        out[name] = "%-3s done %6.1f s | prompt %6d tok | gen %5d @ %5.1f t/s | stop_type %s, truncated %s | sha %s" % (name, time.time() - T0, t["prompt_n"], t["predicted_n"], t["predicted_per_second"], r.get("stop_type"), r.get("truncated"), hashlib.sha256(r["content"].encode()).hexdigest()[:8])
    except urllib.error.HTTPError as e:
        out[name] = "%-3s HTTP %d after %.1f s: %s" % (name, e.code, time.time() - T0, e.read()[:160])
    except Exception as e:
        out[name] = "%-3s ERROR %s after %.1f s" % (name, e, time.time() - T0)
def run(title, jobs):
    global T0
    p0 = logpos(); out = {}; T0 = time.time()
    th = [threading.Thread(target=one, args=(n, p, g, i, out)) for n, p, g, i in jobs]
    for x in th: x.start(); time.sleep(0.2)
    for x in th: x.join()
    print("== %s (wall %.1f s)" % (title, time.time() - T0))
    for j in jobs: print("   " + out[j[0]])
    ev = [l for l in newl(p0) if re.search(r"KV cache full|KV admission|failed to find|retrying|Context size|exception|making room", l, re.I)]
    for l in ev[:6]: print("      log: " + re.sub(r"^[0-9.]+ [A-Z] ", "", l.strip())[:180])
    sys.stdout.flush()
S = "Summarize the entries above in one sentence."
C = "List every entry above again, one per line, in order."
run("two ~12K at once (fit)", [("M1", text(21, 310, S), 128, False), ("M2", text(22, 310, S), 128, False)])
run("two ~28K at once, long replies overflow 64K", [("L1", text(51, 730, C), 8000, True), ("L2", text(52, 730, C), 8000, True)])
run("one short after", [("S1", text(14, 25, S), 64, False)])
