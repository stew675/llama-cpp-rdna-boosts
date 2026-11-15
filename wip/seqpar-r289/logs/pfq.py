#!/usr/bin/env python3
# pfq.py PORT NPARA: one fresh chat prompt of NPARA maintenance-log paragraphs, max_tokens 1; prints prompt timings
import json, sys, urllib.request
port, n = int(sys.argv[1]), int(sys.argv[2])
para = ("The quarterly maintenance log records pump pressure, filter changes, valve inspections and the technician\x27s notes "
        "for building %d, including any anomalies observed during the walk-through and the follow-up actions assigned. ")
b = {"messages": [{"role": "user", "content": "".join(para % i for i in range(n)) + "\n\nHow many buildings are mentioned?"}], "max_tokens": 1, "cache_prompt": False}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=3600))
tm = r["timings"]; print("prefill %d tok in %.1f s = %.1f t/s" % (tm["prompt_n"], tm["prompt_ms"]/1000, tm["prompt_per_second"]))
