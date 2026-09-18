#!/usr/bin/env python3
"""Merge the two measurement TSVs into one canonical table and print pivots."""
import os, re, collections

LOGS1 = "/tmp/mtp-journey-logs"
rows = []  # dict

# ---- first pass (dense1) : build axis spec tps acc meanlen accepted generated
for ln in open("/tmp/mtp-journey.tsv"):
    p = ln.rstrip("\n").split("\t")
    if len(p) < 8 or p[0] == "build":
        continue
    build, axis, spec, tps, acc, ml, acn, gen = p[:8]
    arm = {"delivery": "deliv"}.get(build, build)
    tok = ""
    log = f"{LOGS1}/{build}_{axis}_{spec}.log"
    if os.path.exists(log):
        m = re.search(r"/\s*(\d+) tokens\s+\(\s*[\d.]+ ms per token", open(log, errors="ignore").read())
        # take the generation (last) match
        ms = re.findall(r"/\s*(\d+) tokens\s+\(\s*[\d.]+ ms per token", open(log, errors="ignore").read())
        if ms:
            tok = ms[-1]
    rows.append(dict(cell="dense1", arm=arm, axis=axis, spec=spec, rc="0", tps=tps, tok=tok,
                     acc=acc, ml=ml, acn=acn, gen=gen))

# ---- matrix / extra / pinned journey.tsv
for ln in open("/tmp/journey.tsv"):
    p = ln.rstrip("\n").split("\t")
    if len(p) < 11 or p[0] == "cell":
        continue
    cell, arm, axis, spec, rc, tps, tok, acc, ml, acn, gen = p[:11]
    rows.append(dict(cell=cell, arm=arm, axis=axis, spec=spec, rc=rc, tps=tps, tok=tok,
                     acc=acc, ml=ml, acn=acn, gen=gen))

SPEC_ORDER = ["plain", "static_n3", "mtp_n9", "mtp_n9s9", "combo_n9nm45", "combo_n9s9nm45",
              "adaptive_n7", "adaptive_n12", "pinned8", "pinned12"]
AXES = ["reasoning", "prose-rdna-boosts", "code-python", "recall"]
ARMS = ["stock", "pr", "deliv"]

out = open("/tmp/mtp-journey-results/canonical.tsv", "w")
out.write("cell\tarm\taxis\tspec\trc\ttps\ttok\tacc\tmeanlen\taccepted\tgen\n")
for r in sorted(rows, key=lambda r: (r["cell"], r["axis"], SPEC_ORDER.index(r["spec"]) if r["spec"] in SPEC_ORDER else 99, ARMS.index(r["arm"]) if r["arm"] in ARMS else 99)):
    out.write("\t".join(r[k] for k in ("cell", "arm", "axis", "spec", "rc", "tps", "tok", "acc", "ml", "acn", "gen")) + "\n")
out.close()

for cell in ("dense1", "moe1", "q8t2"):
    axes = [a for a in AXES if any(r["cell"] == cell and r["axis"] == a for r in rows)]
    print(f"\n################## {cell} ##################")
    for axis in axes:
        print(f"\n--- {axis} ---")
        print(f"{'spec':16s} " + " ".join(f"{a:>18s}" for a in ARMS))
        for spec in SPEC_ORDER:
            cells = []
            for arm in ARMS:
                m = [r for r in rows if r["cell"] == cell and r["axis"] == axis and r["spec"] == spec and r["arm"] == arm and r["rc"] == "0"]
                if m:
                    r = m[-1]
                    cells.append(f"{r['tps']}({r['ml']})".rjust(18))
                else:
                    cells.append("".rjust(18))
            if any(c.strip() for c in cells):
                print(f"{spec:16s} " + " ".join(cells))
print("\n(tps(meanlen); canonical.tsv written)")
