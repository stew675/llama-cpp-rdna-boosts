#!/usr/bin/env python3
"""Replay the delivery's block commits onto a new upstream base and report the conflicts, block by block.

WHY: the ladder in RUNGS.md was laid out from a measured conflict map, not from a guess.  `git cherry-pick`
would answer the same question but stops at the first conflict and needs a human; `git merge-tree
--write-tree --messages` performs the same three-way merge without touching any worktree or index, and it
prints the conflicted paths.  Chaining those trees (commit-tree) gives the per-block map for all 16 blocks in
one unattended run.

USAGE (run it from anywhere; it only reads the fork checkout):
    python3 wip/upstream-rebase-261108/probe-conflicts.py [/path/to/llama.cpp]

It reads the old/new bases from the delivery repo's release.json + the first argument or $HOME/llama.cpp.  To
probe a different pair, edit OLD_BASE / NEW_REF below (they are the only inputs).

CAVEAT, by construction: a conflicted block leaves conflict MARKERS in its tree, and the chain continues from
that tree, so later counts are slightly pessimistic.  Read "0 conflicts" as solid and a high count as
indicative.  This is deliberately NOT a resolution tool: the real climb is `git cherry-pick` per rung
(RUNGS.md), because a real resolution must remove the markers before the next block merges.
"""
import json
import pathlib
import re
import subprocess
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path.home() / "llama.cpp")
DELIVERY = pathlib.Path(__file__).resolve().parents[2]
OLD_BASE = json.load(open(DELIVERY / "release.json"))["base"]
NEW_REF = "master"          # the pristine snapshot of upstream in the fork checkout
OLD_TIP = json.load(open(DELIVERY / "release.json"))["tip"]


def g(*args, check=False):
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr}")
    return r


def rev(spec):
    return g("rev-parse", spec, check=True).stdout.strip()


def main():
    blocks = g("log", "--reverse", "--format=%H", f"{OLD_BASE}..{OLD_TIP}", check=True).stdout.split()
    base = rev(NEW_REF)
    report = []
    for i, c in enumerate(blocks):
        parent = rev(c + "^")
        subj = g("log", "-1", "--format=%s", c, check=True).stdout.strip()
        out = g("merge-tree", "--write-tree", "--messages", f"--merge-base={parent}", base, c).stdout
        lines = out.split("\n")
        tree = lines[0].strip() if re.fullmatch(r"[0-9a-f]{40}", lines[0].strip()) else None
        conf = [l.split("Merge conflict in ")[1].strip() for l in lines if "CONFLICT (content)" in l]
        other = [l for l in lines if l.startswith("CONFLICT") and "content" not in l]
        nfiles = len(g("diff", "--name-only", parent, c, check=True).stdout.split())
        report.append(dict(i=i, commit=c[:9], subj=subj, nfiles=nfiles, nconf=len(conf), conf=conf, other=other))
        print(f"[{i:02d}] {c[:9]} conflicts={len(conf):3d} files={nfiles:3d}  {subj[:70]}", flush=True)
        if conf:
            print("      " + ", ".join(conf[:20]), flush=True)
        if other:
            print("      OTHER: " + "; ".join(other[:5]), flush=True)
        if tree is None:
            sys.exit("merge-tree produced no tree; stopping")
        base = g("commit-tree", tree, "-p", base, "-m", "probe:" + subj, check=True).stdout.strip()
    out = pathlib.Path(__file__).with_name("conflict-map.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"total conflicted paths: {sum(x['nconf'] for x in report)}  -> {out}")
    print(f"FINAL synthetic tree commit: {base}")


if __name__ == "__main__":
    main()
