#!/usr/bin/env python3
"""Summarise one 56 GiB-rig server log: the prefill passes, and the taxes that are not compute.

The `prompt eval time` the client sees is NOT the prefill compute: after the last ubatch the server may
re-lay-out the expert pool (`pool_rerank_from_heat_locked`), re-arm stood-down arena tables after a
compute-buffer drop (`moe_cache_rearm`), and take a slab range from the arena (`moe_cache_evict_slab_range`).
Those land between the last `progress = 1.00` line and the `prompt eval time` line, so the difference is
measured here as `tail_s`.  Run:  python3 pp-analysis.py runs/<tag>/server.log [...]
"""
import re
import sys

TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) (.?)(?: \S+ *:)? ?(.*)$")
PROG = re.compile(r"task (\d+) \| prompt processing, n_tokens = *(\d+), progress = ([0-9.]+), t = *([0-9.]+) s / *([0-9.]+) tokens per second")
PEVAL = re.compile(r"task (\d+) \| prompt eval time = *([0-9.]+) ms / *(\d+) tokens")
REARM = re.compile(r"re-armed (\d+) stood-down expert-cache tables.*\(arena now ([0-9.]+) MiB\)")
EVICT = re.compile(r"evicted ([0-9.]+) MiB of arena tables.*\(arena now ([0-9.]+) MiB\)")
DUMP = re.compile(r"(\d+) slots requested, only (\d+) fit; sizing the arena to (\d+) slots")
FAIL = re.compile(r"arena alloc failed for layer=(\d+) role=(\S+) \((\d+) slots x (\d+) B\); cache disabled")
RERANK = re.compile(r"pool_rerank_from_heat_locked: .*: (\d+) filled, (\d+) freed")

def secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def last_prog(text):
    """the last cumulative progress row for the largest token count (the whole prompt)."""
    rows = PROG.findall(text)
    if not rows:
        return None
    top = max(int(r[1]) for r in rows)
    for r in rows:
        if int(r[1]) == top and float(r[2]) >= 0.999:
            return r
    return None

for path in sys.argv[1:]:
    lines = open(path, errors="replace").read().splitlines()
    tag = path.split("/")[-2]
    by_task = {}
    t0 = None
    events = []
    arena = []
    for ln in lines:
        m = TS.match(ln)
        if not m:
            continue
        t = secs(*m.groups()[:4])
        t0 = t if t0 is None else t0
        body = ln
        if "prompt eval time" in body:
            q = PEVAL.search(body)
            if q:
                by_task.setdefault(q.group(1), {})["peval_s"] = float(q.group(2)) / 1000.0
                by_task[q.group(1)]["ptokens"] = int(q.group(3))
        if "progress = 1.00" in body:
            q = PROG.search(body)
            if q:
                by_task.setdefault(q.group(1), {}).update(
                    t_s=float(q.group(4)), tps=float(q.group(5)), ntok=int(q.group(2)))
        if "moe_cache_rearm:" in body:
            q = REARM.search(body)
            events.append(("rearm", t, q.groups() if q else (body,)))
        if "evicted" in body and "arena tables" in body:
            q = EVICT.search(body)
            events.append(("evict", t, q.groups() if q else (body,)))
        if "sizing the arena to" in body:
            q = DUMP.search(body)
            events.append(("dump", t, q.groups() if q else (body,)))
        if "cache disabled for this table" in body:
            q = FAIL.search(body)
            events.append(("dis", t, q.groups() if q else (body,)))
        if "pool_rerank_from_heat_locked" in body:
            q = RERANK.search(body)
            if q:
                events.append(("rerank", t, (int(q.group(1)), int(q.group(2)))))
        if "MoE arena =" in body:
            q = re.search(r"MoE arena = ([0-9.]+) .*arena ([0-9.]+) MiB", body)
            if q:
                arena.append(q.groups())
    print("== %s ==" % tag)
    for task in sorted(by_task, key=lambda k: by_task[k].get("t_s", 0)):
        d = by_task[task]
        if "peval_s" not in d:
            continue
        tail = d["peval_s"] - d.get("t_s", 0)
        print("  task %-4s ntok=%-6s progress=%6.1fs  peval=%6.1fs  tail=%5.2fs  (peval %.1f t/s, processing %.1f t/s, arena_hit=%s)"
              % (task, d.get("ntok"), d.get("t_s", 0), d["peval_s"], tail,
                 d["ptokens"] / d["peval_s"] if d["peval_s"] else 0,
                 d.get("tps", 0), arena[len(arena) - 1][0] if arena else "?"))
    for kind, t, g in events:
        if kind == "rerank":
            continue
        print("  event %-7s t=%8.1f %s" % (kind, t - t0, g))
    rr = [g for k, t, g in events if k == "rerank"]
    if rr:
        print("  rerank: %d tables touched, %d slots filled, %d freed" % (len(rr), sum(a for a, b in rr), sum(b for a, b in rr)))
    if arena:
        print("  arena hit rows: %s" % (arena[-4:],))
