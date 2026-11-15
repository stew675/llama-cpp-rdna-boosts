#!/usr/bin/env python3
# spinert.py PORT MODE: decode inertness of sequence parallelism.  save: prefill a 31.5k prompt (token-split when seqpar is
# on) and save slot 0.  restore: restore that slot, then 512 greedy tokens from it (no prefill left), print the sha.
# fresh: 512 greedy tokens after a fresh prefill.  Same KV in, same decode graphs => same text, whatever LLAMA_SEQPAR says.
import hashlib, json, sys, urllib.request, urllib.error
port, mode = int(sys.argv[1]), sys.argv[2]
para = ("The quarterly maintenance log records pump pressure, filter changes, valve inspections and the technician's notes "
        "for building %d, including any anomalies observed during the walk-through and the follow-up actions assigned. ")
P = "".join(para % i for i in range(790)) + "\n\nHow many buildings are mentioned? Then summarize the log."
def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(r, timeout=3600))
    except urllib.error.HTTPError as e:
        print("HTTP %d on %s: %s" % (e.code, path, e.read().decode()[:300])); sys.exit(1)
gen = {"prompt": P, "n_predict": 512, "temperature": 0, "top_k": 1, "cache_prompt": True, "id_slot": 0, "ignore_eos": True}
if mode == "save":
    r = post("/completion", dict(gen, n_predict=1))
    s = post("/slots/0?action=save", {"filename": "spinert.bin"})
    print("save: prompt %d tok, saved %s tokens, %s bytes" % (r["timings"]["prompt_n"], s.get("n_saved"), s.get("n_written")))
else:
    if mode == "restore":
        s = post("/slots/0?action=restore", {"filename": "spinert.bin"})
        print("restore: %s tokens" % s.get("n_restored"), end="; ")
    r = post("/completion", gen if mode == "restore" else dict(gen, cache_prompt=False))
    t = r["timings"]
    print("%s: prompt %d tok (evaluated), gen %d, %.1f t/s, sha %s" % (mode, t["prompt_n"], t["predicted_n"], t["predicted_per_second"],
          hashlib.sha256(r["content"].encode()).hexdigest()[:12]))
