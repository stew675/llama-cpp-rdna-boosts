#!/usr/bin/env python3
"""One arm's row from a `llama-cli -v` log (its internal server prints the timings as JSON).

    ./arm-row.py <log> <tag>

tokens/step is printed because a speculative arm's t/s is (tokens/step) x (step rate): two MTP arms may only be
compared where that column matches (HANDOVER.md §5 rule 9).  steps = n_predict - draft_n_accepted.
"""
import json
import os
import re
import sys


def timings(path):
    txt = open(path, errors="replace").read()
    m = None
    for m in re.finditer(r'"timings":\{[^{}]*\}', txt):
        pass
    if m is None:
        return None
    return json.loads("{" + m.group(0) + "}")["timings"]


def main():
    path, tag = sys.argv[1], sys.argv[2]
    t = timings(path)
    if t is None:
        print(f"{tag:<24} FAILED (no timings in {os.path.basename(path)})")
        return 1
    dn = t.get("draft_n", 0)
    da = t.get("draft_n_accepted", 0)
    pred = t.get("predicted_n", 0)
    steps = max(pred - da, 1)
    print(f"{tag:<24} prompt_n {t.get('prompt_n',0):>7}  prefill {t.get('prompt_per_second',0):>7.1f} t/s  "
          f"gen {t.get('predicted_per_second',0):>7.2f} t/s  "
          f"{pred/steps:>5.2f} tok/step  draft/acc {dn}/{da} "
          f"({100.0*da/dn if dn else 0:.1f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
