# depth2.py PORT TAG [n_entries]: depth.py with the text saved, one run.
import hashlib, json, os, random, sys, urllib.request
port, tag = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 650; R = random.Random(20261106)
A = "pump valve filter gauge sensor relay bearing rotor stator flange gasket impeller coupling manifold actuator solenoid regulator thermostat compressor condenser".split()
B = "inspected replaced calibrated cleaned tightened lubricated tested rebuilt adjusted logged flagged measured".split()
C = "north south east west upper lower inner outer primary secondary spare auxiliary".split()
body = " ".join("Entry %d: the %s %s at station %d was %s by crew %d on day %d, reading %.2f, cost %d." % (i, R.choice(C), R.choice(A), R.randrange(9000), R.choice(B), R.randrange(97), R.randrange(365), R.random()*500, R.randrange(100000)) for i in range(n))
req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=json.dumps({"prompt": body + "\nSummarize the entries above in three sentences.", "n_predict": 256, "temperature": 0, "top_k": 1, "cache_prompt": False}).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=900)); t = r["timings"]; c = r["content"]
open(os.path.expanduser("~/w1100/kt223/%s.txt" % tag), "w").write(c)
print("%s n=%d: prompt %d tok, gen %d, sha %s" % (tag, n, t["prompt_n"], t["predicted_n"], hashlib.sha256(c.encode()).hexdigest()[:12]))
