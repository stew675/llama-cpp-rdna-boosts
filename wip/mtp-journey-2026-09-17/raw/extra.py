#!/usr/bin/env python3
"""Run the settled-config specs (MTP n9/s9 and the ngram-mod combo) on top of matrix.py."""
import importlib.util

spec = importlib.util.spec_from_file_location("m", "/tmp/matrix.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.SPECS["mtp_n9"]           = ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "9"]
m.SPECS["mtp_n9s9"]         = ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "9", "--spec-draft-n-start", "9"]
m.SPECS["combo_n9nm45"]     = ["--spec-type", "draft-mtp-adaptive,ngram-mod", "--spec-ngram-mod-n-match", "45", "--spec-draft-n-max", "9"]
m.SPECS["combo_n9s9nm45"]   = ["--spec-type", "draft-mtp-adaptive,ngram-mod", "--spec-ngram-mod-n-match", "45", "--spec-draft-n-max", "9", "--spec-draft-n-start", "9"]

RUNS = []
for cell in ("dense1", "q8t2"):
    for axis in m.CELLS[cell]["axes"]:
        for s in ("mtp_n9", "combo_n9nm45"):
            RUNS.append((cell, "pr", axis, s))
        for s in ("mtp_n9", "mtp_n9s9", "combo_n9nm45", "combo_n9s9nm45"):
            RUNS.append((cell, "deliv", axis, s))

print(f"{len(RUNS)} runs", flush=True)
for cell, arm, axis, s in RUNS:
    m.run(None, cell, arm, axis, s)
print("extra done", flush=True)
