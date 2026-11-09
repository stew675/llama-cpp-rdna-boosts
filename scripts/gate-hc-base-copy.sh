#!/usr/bin/env bash
# The hyper-connection base-copy pool-lifetime gate (issue #290, r240).
#
# WHY THIS EXISTS
#   The `hc_combine_norm` [repeat] matcher fuses `repeat -> mul -> add -> rms_norm -> mul(gamma)` into one
#   kernel.  The narrow base (`block_out`, [n_embd,1,T] F32) is not pinned -- it dies at the repeat, so the
#   allocator may hand its buffer to `add` or `mulg`, which the same kernel writes -- so when either output
#   overlaps it the matcher reads the base from a POOL COPY instead of falling back to the unfused chain.
#
#   That copy is a kernel argument and nothing else ever reads it, but r236 registered it on the device
#   pool stack BEFORE the `set_bf16*()` calls of the same window, which push blocks of their own (the mmb
#   activation reserve, `bf16_src1_cache_put`).  The pool is a LIFO stack: a killed block is only freed
#   when it reaches the top, so the copy sat under those blocks and stayed live until the END of the graph.
#   Under `-sm layer` one graph covers every layer of a device, so a prefill graph held one copy per layer:
#   the reporter (2x R9700, Flash-Next GSQ, `-ub 2048`) saw 26 live entries on device 0, ~2.2 GiB, and a
#   `ROCm error: out of memory` from `ggml_cuda_pool_leg::alloc` on a 259.6k-token prompt.  r235 (a scoped
#   alloc, freed after the launch) passed.
#
#   The same class sat in the two per-graph activation caches, whose evictions LEAPFROG: the BF16 src1 cache
#   holds one block per graph and the MMB activation cache `cache_max` of them, but each replaced its previous
#   block only after the other had pushed a newer one above it, so no killed block ever reached the top and
#   both chains were freed only at `pool_stack_release_all` -- the ~1.5 GiB the reporter measured as
#   "remaining" after their own one-line fix (their follow-up snapshot, 41 dead / 1 live).  Both caches now
#   REUSE their block instead of replacing it: one BF16 block per graph, and the MMB cache's eviction re-keys
#   its oldest block as the newest entry.
#
# WHAT IT CHECKS
#   `GGML_CUDA_POOL_STACK_CENSUS=1` prints, once per graph per device, what the pool stack still holds when
#   the graph reaches its end: `live=<entries> dead=<entries>/<MiB> bytes=<MiB> | hc=<live>/<MiB>
#   made=<copies made> mmb=<live>/<MiB> bf16=<live>/<MiB>`.  The per-kind counts are LIVE entries; `dead` is
#   the killed-but-blocked pile (the LIFO class this gate exists for) and `bytes` is what the graph really
#   still holds.
#
#   A. THE TEXT ARMS (no `-lv`: a text arm must not have its stream polluted).  The same-seed greedy text
#      must equal the recorded anchor below, with the delivery defaults and with `GGML_CUDA_BF16_SRC1_CACHE=0`
#      -- the cache's own A/B, so a "fix" that changes numerics fails here.
#   B. THE CENSUS ARM (delivery defaults).  For every device, over every graph of the run:
#        - `hc` LIVE must be 0: no base copy may survive to the end of the graph.  This is the bug.
#        - `made` must be > 0 on the devices that carry layers: the matcher really made copies, so `hc=0`
#          is the fix and not a vacuous arm (a shape that stops reaching the matcher must not pass).
#        - `bf16` LIVE must be <= 1 and `mmb` LIVE <= MMB_LIVE_MAX: the src1 cache holds one block and the
#          MMB activation cache holds `cache_max` (4 on RDNA4), so anything above that is churn.
#        - the retained bytes must stay under MAX_RETAINED_MIB: the per-graph activation caches are
#          legitimate, a per-layer pile of dead-but-blocked blocks is not.
#   C. THE CACHE-OFF CENSUS ARM (`GGML_CUDA_BF16_SRC1_CACHE=0`): `hc` 0 and `bf16` 0 -- the reuse is what
#      keeps the count at one, and with the cache off nothing may be cached at all.
#
#   A GPU-fault scan over this gate's own window is the last step.
#
#   NEGATIVE-TESTED (r240, pre-fix build = the same tree with the pre-r240 ordering in `hyperconn.cu`):
#   the pre-fix build FAILS B with `hc=31/620.0` live on device 0 (10 on device 1, 7 on device 2) and
#   1442 MiB retained in total, against `hc=0`, `mmb=4`, `bf16=1`, `dead=0` and 292 MiB post-fix.  A build
#   without the census prints no lines and fails ("no census lines"), never passes silently.  The cache-off
#   arms are the kill switches for the two reuses (`GGML_CUDA_BF16_SRC1_CACHE=0`, `GGML_CUDA_MMB_CACHE=0` is
#   the manual check -- the MMB cache cannot be off here, MMB is the prefill GEMM path on this model).
#
# Usage:
#   ./gate-hc-base-copy.sh
#   ./gate-hc-base-copy.sh --no-faults
# Environment:
#   BUILD_DIR    llama.cpp build tree     (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI    llama-cli               (default: $BUILD_DIR/bin/llama-cli)
#   MODEL        a qwen4exp GGUF         (default: Flash-Next GSQ-RCO-IQ3_XXS, the BF16-hc class)
#   GPU          HIP_VISIBLE_DEVICES     (default: 0,1,2 -- `-sm layer` needs 2+ devices to reproduce,
#                                         and the model needs all three here)
#   TS           --tensor-split          (default: 60,20,20 -- concentrates the layers, and so the copies,
#                                         on device 0, which is the shape the report measured)
#   CTX/UB       context / ubatch        (default: 8192 / 2048)
#   PROMPT_FILE  the prefill corpus      (default: prompts/prose-rdna-boosts.txt, 5246 tokens)
#   PROMPT       the text arm's prompt   (default: a short fixed prompt)
#   NGEN         generated tokens        (default: 32)
#   ANCHOR       the text arm's expected sha (default: 8447c7032117, this model/shape; a moved anchor is a
#                delivery output change, not a number to re-record)
#   MAX_RETAINED_MIB  the per-graph retention ceiling, all devices summed (default: 512; measured 292)
#   MMB_LIVE_MAX the floor on live MMB activation-cache blocks (default: 8; RDNA4 holds 4)
#   MIN_MADE     the floor on copies made per device (default: 4; measured 31 / 10 / 7)
#   KEEPOUT      dir for the logs        (default: mktemp -d)
#   TIME_LIMIT   per llama-cli timeout   (default: 1800 s)
#   A_ARM/B_ARM/C_ARM=0 skips that section (A is the text arms, B/C the census arms)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
GPU="${GPU:-0,1,2}"
TS="${TS:-60,20,20}"
CTX="${CTX:-8192}"
UB="${UB:-2048}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
PROMPT="${PROMPT:-Write a short paragraph about the Eiffel Tower.}"
NGEN="${NGEN:-32}"
ANCHOR="${ANCHOR:-8447c7032117}"
MAX_RETAINED_MIB="${MAX_RETAINED_MIB:-512}"
MMB_LIVE_MAX="${MMB_LIVE_MAX:-8}"
MIN_MADE="${MIN_MADE:-4}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
A_ARM="${A_ARM:-1}"
B_ARM="${B_ARM:-1}"
C_ARM="${C_ARM:-1}"

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

FAULTS=1
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--no-faults]" >&2; exit 2 ;;
    esac
done

echo "==> hyper-connection base-copy pool-lifetime gate (issue #290)"
echo "    build=$BUILD_DIR"
echo "    model=$MODEL"
echo "    split=$GPU (-sm layer -ts $TS)"
echo "    logs=$KEEPOUT"
mkdir -p "$KEEPOUT"

[ -x "$LLAMA_CLI" ]   || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]       || { echo "FAIL: model not found: $MODEL (set MODEL)" >&2; exit 2; }
[ -f "$PROMPT_FILE" ] || { echo "FAIL: prompt file not found: $PROMPT_FILE" >&2; exit 2; }

# one armed llama-cli run.  A TEXT arm passes no -lv (the same stream carries the generated text); a CENSUS
# arm passes -lv 2, because the census is a WARN and llama-cli starts at errors-only (AGENTS.md, -lv table).
run_text() { # <name> <env...>
    local name="$1"; shift
    env "$@" HIP_VISIBLE_DEVICES="$GPU" \
        timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL" -ngl 99 -sm layer -ts "$TS" \
        -c "$CTX" -b "$UB" -ub "$UB" -fa on -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 \
        --no-display-prompt --single-turn > "$KEEPOUT/$name.log" 2>&1
    local rc=$?
    echo "    $name: exit $rc, $(extract "$name")"
}

run_census() { # <name> <env...>
    local name="$1"; shift
    env "$@" HIP_VISIBLE_DEVICES="$GPU" GGML_CUDA_POOL_STACK_CENSUS=1 \
        timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL" -ngl 99 -sm layer -ts "$TS" \
        -c "$CTX" -b "$UB" -ub "$UB" -fa on -n 0 -f "$PROMPT_FILE" \
        --no-display-prompt --single-turn -lv 2 > "$KEEPOUT/$name.log" 2>&1
    echo "    $name: exit $?, $(grep -c POOL_STACK_CENSUS "$KEEPOUT/$name.log") census lines"
}

extract() { # <chars> <sha> of an arm's generated text (the delivery's own extractor)
    python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.log" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1
}

# census_report <name> -- parse the census lines and assert the contract.  Prints the worst graph per device.
census_report() {
    python3 - "$KEEPOUT/$1.log" "$MAX_RETAINED_MIB" "$MIN_MADE" "$MMB_LIVE_MAX" <<'PY'
import re, sys
path, max_mib, min_made, mmb_live_max = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
pat = re.compile(r"POOL_STACK_CENSUS dev=(\d+) live=(\d+) dead=(\d+)/([\d.]+) bytes=([\d.]+) MiB \| "
                 r"hc=(\d+)/([\d.]+) made=(\d+) mmb=(\d+)/([\d.]+) bf16=(\d+)/([\d.]+)")
worst = {}   # dev -> dict of the graph with the largest retained bytes
made  = {}   # dev -> max copies made in one graph
nlines = 0
for line in open(path, errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    nlines += 1
    d = int(m.group(1))
    rec = dict(live=int(m.group(2)), dead=int(m.group(3)), deadb=float(m.group(4)), bytes=float(m.group(5)),
               hc=int(m.group(6)), hcb=float(m.group(7)), made=int(m.group(8)),
               mmb=int(m.group(9)), mmbb=float(m.group(10)), bf16=int(m.group(11)), bfb=float(m.group(12)))
    made[d] = max(made.get(d, 0), rec["made"])
    if d not in worst or rec["bytes"] > worst[d]["bytes"]:
        worst[d] = rec
fails = 0
if nlines == 0:
    print("FAIL: no POOL_STACK_CENSUS lines: this build has no census (rebuild it) or the run never computed a graph")
    sys.exit(1)
total = sum(r["bytes"] for r in worst.values())
for d in sorted(worst):
    r = worst[d]
    print(f"    dev{d}: worst graph live={r['live']} dead={r['dead']}/{r['deadb']:.1f} MiB "
          f"retained={r['bytes']:.1f} MiB | hc={r['hc']}/{r['hcb']:.1f} made={made[d]} "
          f"mmb={r['mmb']}/{r['mmbb']:.1f} bf16={r['bf16']}/{r['bfb']:.1f}")
    if r["hc"] != 0:
        print(f"FAIL: dev{d} retains {r['hc']} hc base cop{'y' if r['hc']==1 else 'ies'} "
              f"({r['hcb']:.1f} MiB) at the end of a graph: a base copy only has to live for one statement "
              f"(issue #290)")
        fails += 1
    if r["bf16"] > 1:
        print(f"FAIL: dev{d} retains {r['bf16']} live bf16 src1 blocks at the end of a graph: the cache holds "
              f"one, so the rest are replacements that could not be freed (issue #290)")
        fails += 1
    if r["mmb"] > mmb_live_max:
        print(f"FAIL: dev{d} retains {r['mmb']} live MMB activation-cache blocks (max {mmb_live_max}): the "
              f"eviction is allocating instead of re-keying (issue #290)")
        fails += 1
print(f"    retained at graph end, all devices: {total:.1f} MiB (ceiling {max_mib:.0f})")
if total > max_mib:
    print(f"FAIL: {total:.1f} MiB retained across devices is above the {max_mib:.0f} MiB ceiling: a block class "
          f"is outliving its statement")
    fails += 1
if max(made.values() or [0]) < min_made:
    print(f"FAIL: at most {max(made.values() or [0])} base copies were made in a graph (floor {min_made}): the "
          f"arm no longer reaches the matcher's copy path, so hc=0 proves nothing")
    fails += 1
sys.exit(1 if fails else 0)
PY
}

if [ "$A_ARM" -eq 1 ]; then
    echo
    echo "==> A. the text arms: defaults, and the BF16 src1 cache off (its own A/B)"
    run_text a_text
    read -r a_chars a_sha < <(extract a_text)
    if [ -z "${a_sha:-}" ]; then
        fail "the default text arm produced no text (log: $KEEPOUT/a_text.log)"
    elif [ "${a_sha:-}" != "$ANCHOR" ]; then
        fail "the default text arm moved: ${a_chars} chars sha=${a_sha}, anchor ${ANCHOR} (a delivery output change, not a gate number to update)"
    else
        note "a_text: ${a_chars} chars ${a_sha} == anchor ${ANCHOR}"
    fi
    if [ -n "${a_sha:-}" ]; then
        run_text b_text GGML_CUDA_BF16_SRC1_CACHE=0
        read -r b_chars b_sha < <(extract b_text)
        if [ "${b_sha:-}" != "${a_sha:-}" ] || [ "${b_chars:-0}" != "${a_chars:-0}" ]; then
            fail "the cache-off arm differs from the default arm (${b_chars:-?} chars ${b_sha:-none} vs ${a_chars} chars ${a_sha}): the BF16 src1 reuse is not numerics-neutral"
        else
            note "b_text: ${b_chars} chars ${b_sha} == a_text (the reuse is numerics-neutral)"
        fi
    fi
fi

if [ "$B_ARM" -eq 1 ]; then
    echo
    echo "==> B. the census arm (defaults): every copy released, one src1 block per graph"
    run_census c_census
    census_report c_census || fail "the default census arm is above (pre-fix signature: hc=31/620.0 on device 0)"
fi

if [ "$C_ARM" -eq 1 ]; then
    echo
    echo "==> C. the census arm with the BF16 src1 cache off: nothing cached, nothing retained"
    run_census d_census GGML_CUDA_BF16_SRC1_CACHE=0
    census_report d_census || fail "the cache-off census arm is above"
    if grep -q "bf16=0/0.0" "$KEEPOUT/d_census.log"; then
        note "d_census: no bf16 block at all with the cache off"
    else
        fail "the cache-off census arm still holds a bf16 block: GGML_CUDA_BF16_SRC1_CACHE=0 did not take effect"
    fi
fi

if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (since ${GATE_SINCE_TS:-the journal start})"
    since_args=()
    [ -n "${GATE_SINCE_TS:-}" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /' || fail "a GPU fault was recorded during this gate's arms"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: the hc base copy is released per statement and the text is unchanged"
    exit 0
fi
echo "FAILED: $fails check(s) -- see $KEEPOUT"
exit 1
