#!/usr/bin/env python3
"""One markdown row per arm of the parity matrix, from `runs/<tag>/{parity.txt,server.log}`.

Columns, and why each is there:
  * `gen3` -- the third 600-token pass (the warm decode number the campaign quotes; the first pass of a
    server is always slower).
  * `pp50k_1` -- the client-visible first-pass 50k prefill (`prompt eval time`, i.e. TTFT to the first token).
  * `proc` -- the same pass's *processing* rate from the last `progress = 1.00` row: the compute+staging part.
  * `tail` -- `pp50k_1 - proc`: the band-transition tax (pool re-layout + arena re-arm), which is NOT
    prefill compute but IS charged to the first token.  Naming it separately is the whole point: on a
    non-mirror tier it is 2-22 s and it is the biggest single term in the field prefill gap.
  * `MiB1` -- storage bytes the server read (`/proc/<pid>/io read_bytes`) during that pass.
  * `fill` -- pool slots re-filled by `pool_rerank_from_heat_locked` over the run (each one expert copy).
  * `slots` -- the tier the pool actually got (read back from the placement log: never trust the knob).
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def one(tag):
    p = os.path.join(ROOT, "runs", tag, "parity.txt")
    log = os.path.join(ROOT, "runs", tag, "server.log")
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read()
    d = {"tag": tag}
    ms = re.findall(r"=== %s\s+LME=(\S+) SINK=(\S+) DIO=(\S+) SPAN=(\S+) RANK=(\S+)" % re.escape(tag), t)
    if ms:
        # the LAST header: an aborted arm that was restarted appends a second one to the same parity.txt
        d["lme"], d["sink"], d["dio"], d["span"], d["rank"] = ms[-1]
    m = re.search(r"applied tier: slots=(\d+)/(\d+)", t)
    d["slots"] = "%s/%s" % m.groups() if m else "?"
    gens = re.findall(r"^gen\d prompt_n=\S+ prompt_tps=\S+ pred_tps=([0-9.]+)", t, re.M)
    d["gen"] = gens[-1] if gens else "?"
    pp = re.findall(r"^pp50k_pass(\d) prompt_n=\S+ prompt_tps=([0-9.]+).*wall_s=([0-9.]+)", t, re.M)
    d["pp"] = {int(a): (float(b), float(c)) for a, b, c in pp}
    st = re.findall(r"^storage_reads_MiB: pp50k_pass(\d)=(\d+)", t, re.M)
    d["stor"] = {int(a): int(b) for a, b in st}
    m = re.search(r"^storage_reads_MiB: gen=(\d+)", t, re.M)
    d["gen_stor"] = int(m.group(1)) if m else 0
    m = re.search(r"host pool \(L2\): \d+ tensors, (\d+) slots, ([0-9.]+) MiB pinned", t)
    d["pinned"] = float(m.group(2)) / 1024.0 if m else 0.0
    m = re.search(r"sink \(MOE_HOST_POOL_SINK\): (\d+) demoted", t)
    d["demoted"] = int(m.group(1)) if m else 0
    m = re.search(r"sink reuse: (\d+) uses against (\d+) demotions \(([0-9.]+) %", t)
    d["reuse"] = float(m.group(3)) if m else 0.0
    m = re.search(r"^anon (\d+)", t, re.M)
    d["anon"] = int(m.group(1)) / 2**30 if m else 0
    # processing rate and tail, per pass, from the server log
    d["proc"], d["tail"] = {}, {}
    if os.path.exists(log):
        txt = open(log, errors="replace").read()
        prog = {}
        for m in re.finditer(r"task (\d+) \| prompt processing, n_tokens = *(\d+), progress = 1\.00, t = *([0-9.]+) s / *([0-9.]+)", txt):
            prog[m.group(1)] = (float(m.group(3)), float(m.group(4)))
        pev = {}
        for m in re.finditer(r"task (\d+) \| prompt eval time = *([0-9.]+) ms / *(\d+) tokens", txt):
            pev[m.group(1)] = (float(m.group(2)) / 1000.0, int(m.group(3)))
        # order the passes: the ones with the big token counts, by start time is unknown, so pair them by
        # their (processing, peval) and let the caller see both.
        rows = []
        for task, (pt, pts) in prog.items():
            if task in pev and pev[task][1] > 10000:
                rows.append((int(task), pt, pts, pev[task][0]))
        rows.sort()   # by task id = request order (NOT by duration: pass 2 is usually faster)
        for i, (_task, pt, pts, pe) in enumerate(rows[:2]):
            d["proc"][i + 1] = pts
            d["tail"][i + 1] = pe - pt
        fills = 0
        for m in re.finditer(r"pool_rerank_from_heat_locked.*?: (\d+) filled", txt):
            fills += int(m.group(1))
        d["fills"] = fills
    return d

def main():
    tags = sys.argv[1:] or [os.path.basename(x) for x in sorted(glob.glob(os.path.join(ROOT, "runs", "*")))]
    print("| arm | -lme | SINK | DIO | SPAN | slots | pinned GiB | gen3 t/s | gen MiB | pp50k#1 | proc | tail s | MiB#1 | pp50k#2 | tail s | MiB#2 | fills | demoted | reuse % | anon GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in tags:
        d = one(tag)
        if not d:
            continue
        p1 = d["pp"].get(1, (0, 0))
        p2 = d["pp"].get(2, (0, 0))
        print("| %s | %s | %s | %s | %s | %s | %.1f | %s | %s | %.0f | %.0f | %.2f | %s | %.0f | %.2f | %s | %s | %s | %.1f | %.1f |" % (
            tag, d.get("lme", "?"), d.get("sink", "?"), d.get("dio", "?"), d.get("span", "?"), d["slots"], d.get("pinned", 0.0),
            d["gen"], d.get("gen_stor", 0), p1[0], d["proc"].get(1, 0), d["tail"].get(1, 0), d["stor"].get(1, "?"),
            p2[0], d["tail"].get(2, 0), d["stor"].get(2, "?"), d.get("fills", "?"), d.get("demoted", 0),
            d.get("reuse", 0.0), d["anon"]))

if __name__ == "__main__":
    main()
