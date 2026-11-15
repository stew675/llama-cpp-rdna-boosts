#!/usr/bin/env python3
"""POST one /completion and print the fields the A/B is judged on, as JSON."""
import json, sys, urllib.request

url = sys.argv[1]
body = json.load(open(sys.argv[2]))
req = urllib.request.Request(url, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    d = json.load(r)
t = d.get("timings", {})
text = d.get("content", "")
import hashlib
out = {
    "prompt_n":        t.get("prompt_n"),
    "pp_tps":          round(t.get("prompt_per_second") or 0, 2),
    "predicted_n":     t.get("predicted_n"),
    "tg_tps":          round(t.get("predicted_per_second") or 0, 2),
    "draft_n":         t.get("draft_n"),
    "draft_n_accepted": t.get("draft_n_accepted"),
    "text_chars":      len(text),
    "text_sha256_12":  hashlib.sha256(text.encode()).hexdigest()[:12],
    "stop":            d.get("stop_type"),
}
print(json.dumps(out))
