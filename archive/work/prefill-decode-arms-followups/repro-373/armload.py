#!/usr/bin/env python3
"""armload.py -- drive the crash shape of crash #373 (TODO #373 / campaign W9).

Three slots on the two-arm server, exactly the maintainer's workload shape:
  slot 0: a short prompt + a LONG generation (keeps the decode arm applying for ~1 min)
  slot 1: a ~71k-token prompt
  slot 2: a ~23k-token prompt, launched ~13 s after slot 1 (as in the crash log)

Every round is a fresh set of requests, so the request boundary and the prefills recur.
Nothing here asserts; it only drives and prints.  The server log is the artifact.
"""
import json
import os
import sys
import threading
import time
import urllib.request

HOST = os.environ.get("HOST", "http://127.0.0.1:8033")
LONG_TOK = int(os.environ.get("LONG_TOK", 71000))
MED_TOK = int(os.environ.get("MED_TOK", 23000))
HEAD_TOK = int(os.environ.get("HEAD_TOK", 2000))
GEN_TOK = int(os.environ.get("GEN_TOK", 800))
ROUNDS = int(os.environ.get("ROUNDS", 1))
MED_DELAY = float(os.environ.get("MED_DELAY", 13.0))
MED_GEN = int(os.environ.get("MED_GEN", 8))
# TEXTDUMP=<dir>: write every response body to <dir>/<round>-<tag>.txt, so two arms driving the SAME workload
# (e.g. the batched state move against LLAMA_RS_FORCE_SERIALISED=1) can be compared with diff(1) -- a text
# reference is per device count, per split and per residency class (`GREEDY-PURITY.md` §38/§89/§105).
TEXTDUMP = os.environ.get("TEXTDUMP", "")
# ARM_TEMP=0 makes the arm's text an ORACLE (NOT `TEMP`: exporting TEMP to a GPU process breaks the ROCm runtime --
# `libLLVM`'s JIT, i.e. hipBLASLt's Tensile cache path, dies with a stack-write fault at load; `WORKLOG.md` 261122.014): at the runme's 0.7 a tiny numeric difference flips a sampled token, and a
# CO-BATCHED pair is stress only, never a text claim (`GREEDY-PURITY.md` §63).  SERIAL=1 fires the requests one at a
# time instead of overlapping them, which makes the pair's step composition -- and therefore the cell placement of
# every handover -- reproducible across two runs, so two arms can be compared byte for byte.
ARM_TEMP = float(os.environ.get("ARM_TEMP", 0.7))
SERIAL = os.environ.get("SERIAL", "") not in ("", "0")
SEQ = [0]  # a request counter: the slot a request lands on is the server's choice, the order is ours
TEXT = os.environ.get("TEXT", "/llm/models/wikitext-2-raw/wiki.train.raw")


def post(path, payload, timeout=1800):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def n_tokens(text):
    try:
        return len(post("/tokenize", {"content": text}, timeout=120)["tokens"])
    except Exception as e:  # noqa: BLE001
        print(f"  (tokenize failed: {e})", flush=True)
        return -1


def prompt_of(target_tok, tag):
    """A deterministic chunk of wikitext sized to ~target_tok tokens."""
    with open(TEXT, "r", encoding="utf-8", errors="ignore") as f:
        blob = f.read(4 * 1024 * 1024)
    # ~3.6 chars/token for prose; refine once with the real tokenizer
    n_chars = int(target_tok * 3.6)
    text = blob[:n_chars]
    got = n_tokens(text)
    if got > 0 and got != target_tok:
        text = blob[:int(len(text) * target_tok / got)]
        got = n_tokens(text)
    print(f"[{tag}] target {target_tok} tok -> {len(text)} chars, {got} tok", flush=True)
    return text


def fire(tag, prompt, n_predict, out):
    t0 = time.time()
    try:
        r = post("/completion", {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": ARM_TEMP,
            "top_k": 20,
            "min_p": 0.01,
            "seed": 42,
            "cache_prompt": False,
            "stream": False,
        })
        out[tag] = ("ok", time.time() - t0, len(r.get("content", "")), r.get("tokens_predicted"))
        if TEXTDUMP:
            os.makedirs(TEXTDUMP, exist_ok=True)
            with open(os.path.join(TEXTDUMP, f"{tag}.txt"), "w", encoding="utf-8") as f:
                f.write(r.get("content", ""))
    except Exception as e:  # noqa: BLE001
        out[tag] = ("ERR", time.time() - t0, str(e), 0)
    print(f"[{tag}] {out[tag][0]} in {out[tag][1]:.1f} s "
          f"(predicted={out[tag][3]}) {'' if out[tag][0] == 'ok' else out[tag][2]}", flush=True)


def main():
    head = prompt_of(HEAD_TOK, "head")
    longp = prompt_of(LONG_TOK, "long")
    medp = prompt_of(MED_TOK, "med")

    for rnd in range(ROUNDS):
        print(f"=== round {rnd} {time.strftime('%H:%M:%S')}", flush=True)
        out = {}
        if SERIAL:
            fire("head", head, GEN_TOK, out)
            fire("long", longp, MED_GEN, out)
            if MED_TOK > 0:
                fire("med", medp, MED_GEN, out)
        else:
            th = [threading.Thread(target=fire, args=("head", head, GEN_TOK, out))]
            th[0].start()
            time.sleep(1.0)
            th.append(threading.Thread(target=fire, args=("long", longp, MED_GEN, out)))
            th[1].start()
            if MED_TOK > 0:
                time.sleep(MED_DELAY)
                th.append(threading.Thread(target=fire, args=("med", medp, MED_GEN, out)))
                th[2].start()
            for t in th:
                t.join()
        bad = [k for k, v in out.items() if v[0] != "ok"]
        print(f"=== round {rnd} done {time.strftime('%H:%M:%S')} "
              f"{'FAILED: ' + ','.join(bad) if bad else 'all ok'}", flush=True)
        if bad:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
