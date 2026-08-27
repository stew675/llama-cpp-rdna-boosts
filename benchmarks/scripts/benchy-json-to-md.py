#!/usr/bin/env python3
"""benchy-json-to-md.py - render the md summary table from a saved
llama-benchy JSON report.

usage: benchy-json-to-md.py <report.json> [<out.md>]
  reads a --save-result --format json output and writes the same style of
  pipe table llama-benchy prints for --format md (mean ± std).
"""
import json
import sys


def fmt(metric, ndigits=2):
    if metric is None:
        return ""
    return f"{metric['mean']:.{ndigits}f} ± {metric['std']:.{ndigits}f}"


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        d = json.load(f)

    rows = []
    for run in d.get("benchmarks", []):
        if run.get("is_context_prefill_phase"):
            continue
        ctx = run["context_size"]
        pp, tg = run["prompt_size"], run["response_size"]
        for test, pk in (("pp", "pp_throughput"), ("tg", "tg_throughput")):
            if run.get(pk):
                name = f"{test}{pp if test == 'pp' else tg}" + (f" @ d{ctx}" if ctx else "")
                rows.append([
                    d.get("model", ""),
                    name,
                    fmt(run[pk]),
                    fmt(run.get("peak_throughput")),
                    fmt(run.get("ttfr")),
                    fmt(run.get("est_ppt")),
                    fmt(run.get("e2e_ttft")),
                ])

    out = sys.stdout
    close_out = False
    if len(sys.argv) >= 3:
        out = open(sys.argv[2], "w")
        close_out = True

    headers = ["model", "test", "t/s", "peak t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]
    out.write("| " + " | ".join(headers) + " |\n")
    out.write("|" + "---|" * len(headers) + "\n")
    for r in rows:
        out.write("| " + " | ".join(r) + " |\n")

    if close_out:
        out.close()
        print(f"wrote {sys.argv[2]}")
    else:
        sys.stdout.write("")


if __name__ == "__main__":
    main()
