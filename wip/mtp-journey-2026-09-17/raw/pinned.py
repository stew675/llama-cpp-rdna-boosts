#!/usr/bin/env python3
"""Pinned-depth diagnostic: force both adaptive arms to the same depth so the
controller is removed and only kernel/numerics speed remains.  Also completes
the q8t2 (issue #35) adaptive_n12 columns."""
import importlib.util

spec = importlib.util.spec_from_file_location("m", "/tmp/matrix.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.SPECS["pinned12"] = ["--spec-type", "draft-mtp-adaptive",
                       "--spec-draft-n-min-adaptive", "12", "--spec-draft-n-max", "12"]
m.SPECS["pinned8"]  = ["--spec-type", "draft-mtp-adaptive",
                       "--spec-draft-n-min-adaptive", "8", "--spec-draft-n-max", "8"]
m.SPECS["adaptive_n12"] = ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "12"]

RUNS = []
for cell in ("dense1", "q8t2"):
    for axis in m.CELLS[cell]["axes"]:
        for arm in ("pr", "deliv"):
            RUNS.append((cell, arm, axis, "pinned12"))
            RUNS.append((cell, arm, axis, "pinned8"))
for axis in m.CELLS["q8t2"]["axes"]:
    for arm in ("pr", "deliv"):
        RUNS.append(("q8t2", arm, axis, "adaptive_n12"))

print(f"{len(RUNS)} runs", flush=True)
for cell, arm, axis, s in RUNS:
    m.run(None, cell, arm, axis, s)
print("pinned done", flush=True)
