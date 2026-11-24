#!/usr/bin/env bash
# Normalise a llama-server / llama-cli log into its SHAPE so two runs can be compared line by line.
#
# Why: "the default path prints exactly the same lines" is an assertion this project makes at every
# step, but a raw log differs on every run (timestamps, rates, ms, tick counts, ports).  What must not
# change is the sequence of (level, component, function, message) — a missing/extra INFO line, a new
# WARN, a changed decision.  This strips precisely the run-to-run noise and nothing else:
#
#   * the leading `D.DD.DDD.DDD ` timestamp;
#   * decimals (rates, ms, MiB) but NOT integers: `n_slots = 2` and `n_ctx_slot = 8192` are decisions
#     that must be compared, while `1049.48 tokens per second` is noise (--mask-ints drops integers too,
#     which over-normalises and can hide a real change -- use it only for request-chatter lines);
#   * the port in a `listening on http://127.0.0.1:PORT` line;
#   * everything from a termination line onwards (a killed server's shutdown tail is not a decision);
#   * optional (--drop-prefix RE): repeated per-request chatter (slot launch/timing/release) — use it
#     when the arm count differs between the two runs and only the STARTUP is being compared.
#
# Usage: norm-log.sh LOG [--drop-prefix RE] [--mask-ints]
set -uo pipefail

LOG="${1:-}"; [ -f "$LOG" ] || { echo "usage: norm-log.sh LOG [--drop-prefix RE]" >&2; exit 2; }
DROP=""; MASK_INTS=0
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --drop-prefix) DROP="${2:-}"; shift 2 ;;
        --mask-ints) MASK_INTS=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

python3 - "$LOG" "$DROP" "$MASK_INTS" <<'PY'
import re, sys
log, drop, mask_ints = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
# "<ts> <level> <component> <function>: <msg>"  (server / common)
pre  = re.compile(r'^\d+\.\d+\.\d+\.\d+ ([IWE]) +(\S+) +(\S+): (.*)$')
# "<ts> <level> <tag>: <msg>"                   (the library's own lines: llama_context:, llama_kv_cache:,
#                                                notify_borrowed:, resolve_fused_ops:, report_locked:, ...)
pre1 = re.compile(r'^\d+\.\d+\.\d+\.\d+ ([IWE]) +(\S+): (.*)$')
term = re.compile(r'Received (first|second) interrupt|terminating immediately|Exiting\.\.\.')
out = []
for raw in open(log, encoding="utf-8", errors="replace"):
    line = raw.replace("\r", "").replace("\b", "").rstrip("\n")
    m = pre.match(line)
    if m:
        lvl, comp, fn, msg = m.groups()
    else:
        m1 = pre1.match(line)
        if m1:
            lvl, comp, msg = m1.groups()
            fn  = "-"
        elif line.startswith(("ggml_", "llama_", "load_", "print_info", "srv  ", "slot ")):
            lvl, comp, fn, msg = "?", "?", "?", line          # plain lines (no timestamp prefix)
        else:
            continue
    if term.search(msg):
        break
    msg = re.sub(r'127\.0\.0\.1:\d+', '127.0.0.1:PORT', msg)
    msg = re.sub(r'://[^ ]*:\d+', '://HOST:PORT', msg)
    msg = re.sub(r'0x[0-9a-f]+', 'PTR', msg)                 # addresses are per-run
    msg = re.sub(r'\b\d+\.\d+\b', 'N', msg)                # decimals (rates, ms, MiB)
    if mask_ints:
        msg = re.sub(r'\b\d+\b', 'N', msg)
    msg = re.sub(r'\s+', ' ', msg).strip()
    shape = "%s %s %s: %s" % (lvl, comp, fn, msg)
    if drop and re.search(drop, shape):
        continue
    out.append(shape)
print("\n".join(out))
PY
