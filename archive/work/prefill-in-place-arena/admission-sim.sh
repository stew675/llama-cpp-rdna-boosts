#!/usr/bin/env bash
# Does admitting a prompt's experts on PREFILL cost the following DECODE more than it saves?
# (`TODO` #317 rung 3: a prefill miss may reclaim the coldest resident THIS PASS DOES NOT ROUTE, ranked by the
# decode heat prior.)  Answered offline, from ONE routing trace, with `scripts/moe-trace-replay.py`'s
# `--prefill-admit` model -- no GPU, no 30 s arena seed, and one trace sweeps every prompt size.
#
# The trace (captured on `halo`, Qwen3.5-122B-A10B **Q8_0**, the model that does NOT fit: 133 GiB of weights,
# 123 GiB RAM, `-lme 90` => 200 of 256 slots/table, 78 %):
#   MOE_EXPERT_TRACE=traces/q8-lme90-code533-n200.trace MOE_EXPERT_HEATMAP=load \
#     llama-cli -m ...Q8_0-00001-of-00004.gguf -lme 90 -ub 4096 -b 4096 -ngl all --fit off \
#       --seed 42 --temp 0 --no-display-prompt --single-turn -c 8192 -n 200 -f prompts/code-python.txt
#   538-token prompt (ONE prefill pass, "up to 256 used expert(s)"), 200 decode tokens.
#
# Arms (all on the same trace, same seed, same slots):
#   copy      the delivered path (r265): the prefill band stages the used set, D2D-copying every resident.
#   reserve   in place, landing reserve only -- the strict "never evict" reading: 12 of 200 slots per table.
#   ladder    in place with the reserve, then the coldest NON-ROUTED resident (the delivered policy).
#   nolanding in place and NO reserve: every admission reclaims a resident (the harshest decode test).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO: walk up to the delivery repo root (this campaign moved to archive/work/ after r266, so a fixed
# "../.." is wrong -- find scripts/moe-trace-replay.py instead).
REPO="$HERE"
while [ ! -f "$REPO/scripts/moe-trace-replay.py" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
TRACE="${TRACE:-$HERE/traces/q8-lme90-code533-n200.trace}"
HM="${HM:-/llm/models/Qwen3.5/122B-A10B/Q8_0/Qwen3.5-122B-A10B-Q8_0.moe-heatmap}"
ARENA_MIB="${ARENA_MIB:-91800}"    # per device; 200 slots/table on this trace
SIZES="${SIZES:-200 300 400 538}"
TMP="$(mktemp -d)"

run() { python3 "$REPO/scripts/moe-trace-replay.py" "$TRACE" --seed "sidecar:$HM" --arena-mib "$ARENA_MIB" \
            --policies shipped --quiet --json "$1" "${@:2}" >/dev/null; }

for P in $SIZES; do
    run "$TMP/copy-$P.json"  --prefill-admit off     --prefill-prompt-tokens "$P"
    run "$TMP/res-$P.json"   --prefill-admit landing --landing-frac 16 --prefill-prompt-tokens "$P"
    run "$TMP/lad-$P.json"   --prefill-admit ladder  --landing-frac 16 --prefill-prompt-tokens "$P"
    run "$TMP/nol-$P.json"   --prefill-admit ladder  --landing-frac 0  --prefill-prompt-tokens "$P"
    run "$TMP/copy16-$P.json" --prefill-admit off    --prefill-prompt-tokens "$P" --decode-scoped 16
    run "$TMP/lad16-$P.json"  --prefill-admit ladder --landing-frac 16 --prefill-prompt-tokens "$P" --decode-scoped 16
done

python3 - "$TMP" $SIZES <<'PY'
import json, sys
tmp, sizes = sys.argv[1], [int(x) for x in sys.argv[2:]]
def r(f): return json.load(open("%s/%s.json" % (tmp, f)))["results"]["shipped"]
mib = lambda b: b / 1048576.0
print("%-6s %-36s %8s %8s %9s %10s %9s %9s %8s"
      % ("prompt", "arm", "h", "misses", "d miss", "prefill D2D", "admitted", "reclaim", "declined"))
for P in sizes:
    base = r("copy-%d" % P)
    for tag, name in (("copy", "copy path (r265)"),
                      ("res", "in place: landing reserve only"),
                      ("lad", "in place: ladder (delivered)"),
                      ("nol", "in place: ladder, no reserve")):
        x = r("%s-%d" % (tag, P))
        print("%-6d %-36s %8.4f %8d %+8d %10.1f %9d %9d %8d"
              % (P, name, x["h"], x["misses"], x["misses"] - base["misses"],
                 x["prefill_copy_mib"] if tag == "copy" else 0.0,
                 x["prefill_admitted"], x["prefill_reclaimed"], x["prefill_declined"]))
    a, b = r("copy16-%d" % P), r("lad16-%d" % P)
    print("%-6d %-36s %8.4f %8d %+8d" % (P, "  first 16 decode passes (follow-up)",
                                         0.0, b["misses"], b["misses"] - a["misses"]))
    print()
print("`prefill D2D` is the arena->input_cpy copy the copy path makes on the prefill band; the in-place arms")
print("make none.  `read` (the misses both arms read from storage) is identical in every arm and not printed.")
PY
rm -rf "$TMP"
