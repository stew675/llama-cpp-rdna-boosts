#!/usr/bin/env bash
# gate-arms-default-inert -- with the pair OFF, the default path is byte-identical: the startup SHAPE and the
# greedy TEXT, on four shapes.
#
# WHY (invariant I7 of DESIGN.md): every STEP 2-7 change is supposed to be inert unless a second context exists
# (the borrow, the one allocator, the Meta view slot, the recurrent placement, the slab aux region).  "Inert" is
# an assertion, not a memory, and it has two instruments that see different things:
#
#   * the STARTUP SHAPE -- the normalised sequence of (level, component, function, message) lines up to
#     `listening on`.  It catches a new INFO/WARN, a changed decision (n_slots, n_ctx_slot, the admission line),
#     a changed cache geometry; it does NOT see a numerical change;
#   * the GREEDY TEXT -- same prompt, seed 42, temp 0, one fresh process per arm.  It catches a numerical change
#     on the decode path; a uniform *prefill*-logit shift needs `scripts/gate-prefill-logits.sh` (a different
#     gate, STEP 11), and a latency change is invisible to both (that is `gate-arms-itl.sh`).
#
# SHAPES (the four the campaign validates on):
#   4b-1gpu            Qwen3.5 4B Q8_0, 1 device        -- dense, GPU-resident
#   phi4-1gpu          phi-4 IQ4_XS, 1 device           -- dense, a second architecture family
#   27b-3gpu-tensor    Qwen3.8 27B Q8_0, 3 devices `-sm tensor -mg 0` -- the comm layer is reachable
#   35b-a3b-host       Qwen3.6 35B-A3B Q8_0, 1 device `-ncmoe 40`     -- the slab + host expert tier is armed
#
# THE ANCHORS ARE PRE-CHANGE WHERE THEY EXIST, and the gate SAYS which is which (every line prints its
# provenance).  A1/A2/A3 and the 4B server shape were recorded at STEP 0 on the r353 delivery tree, before any
# campaign patch existed; the other rows were recorded by `--record` at STEP 8 and are therefore a *lock* (they
# catch a later regression) and NOT by themselves evidence of inertness.
#
# NON-VACUITY (the gate must be able to fail):
#   * SENSITIVITY -- on `SENS_SHAPE` (default 4b-1gpu) the SAME server is also started with the pair ON, and the
#     gate asserts (a) its shape DIFFERS from the pair-OFF shape and equals the recorded pair-ON shape (so the
#     shape comparison sees a change when one exists, and the pair's own logging is locked too), and (b) its
#     responses are byte-identical to the pair-OFF arm's (the pair changes latency, not answers).
#   * RUN-STABILITY -- the cli arm on `SENS_SHAPE` is read twice and both reads must agree; a text anchor that
#     is not run-stable cannot catch a change (the same-seed near-tie class, GREEDY-PURITY.md).
#
# USAGE
#   gate-arms-default-inert.sh [--shapes LIST] [--record] [--force] [--keep] [--no-cli] [--no-srv]
#     --shapes LIST   comma list: 4b-1gpu,phi4-1gpu,27b-3gpu-tensor,35b-a3b-host (default: all four)
#     --record        write a MISSING anchor row / shape file from what the arm just read, and say so loudly.
#                     Never overwrites an existing anchor unless --force (a re-recorded anchor is not evidence).
#     --no-cli        skip the cli text arms      --no-srv  skip the server shape arms
#     --keep          keep the logs (they are kept under $OUT either way; this prints the path)
#
# ENVIRONMENT: HIP_VISIBLE_DEVICES is set per shape (never inherited); ROCBLAS_USE_HIPBLASLT=0 (fresh-process
# text comparison, INVARIANTS §44); MOE_EXPERT_HEATMAP=load on the host-expert shape (AGENTS rule 5).  One GPU
# job at a time is ENFORCED with an exclusive flock for the whole gate.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$V2/../.." && pwd)"
BASE="$V2/baseline"
BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-rocm/bin}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-default-inert}"
ANCHORS="$BASE/inert-anchors.tsv"
SHAPES_DIR="$BASE/inert-shapes"
SHAPES="4b-1gpu,phi4-1gpu,27b-3gpu-tensor,35b-a3b-host"
SENS_SHAPE="${SENS_SHAPE:-4b-1gpu}"
NPRED_CLI_DEFAULT=64
RECORD=0; FORCE=0; KEEP=0; DO_CLI=1; DO_SRV=1

while [ $# -gt 0 ]; do
    case "$1" in
        --shapes) SHAPES="$2"; shift 2 ;;
        --record) RECORD=1; shift ;;
        --force)  FORCE=1; shift ;;
        --keep)   KEEP=1; shift ;;
        --no-cli) DO_CLI=0; shift ;;
        --no-srv) DO_SRV=0; shift ;;
        -h|--help) sed -n '2,48p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT" "$SHAPES_DIR"

# -----------------------------------------------------------------------------------------------------------
# the shape table: model, devices, cli args, server args, server env, cli -n, and where each anchor comes from
# -----------------------------------------------------------------------------------------------------------
shape_var() {  # $1 = shape, $2 = which field; echoes the value (empty is legal for extras/env)
    case "$1:$2" in
        4b-1gpu:model)     echo /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf ;;
        phi4-1gpu:model)   echo /llm/models/phi-4/IQ4_XS/phi-4-IQ4_XS.gguf ;;
        27b-3gpu-tensor:model) echo /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf ;;
        35b-a3b-host:model)    echo /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf ;;

        4b-1gpu:devices)      echo 0 ;;
        phi4-1gpu:devices)    echo 0 ;;
        27b-3gpu-tensor:devices) echo 0,1,2 ;;
        35b-a3b-host:devices) echo 0 ;;

        4b-1gpu:extra)        echo "" ;;
        phi4-1gpu:extra)      echo "" ;;
        27b-3gpu-tensor:extra) echo "-sm tensor -mg 0" ;;
        35b-a3b-host:extra)   echo "-ncmoe 40" ;;

        4b-1gpu:env)          echo "" ;;
        phi4-1gpu:env)        echo "" ;;
        27b-3gpu-tensor:env)  echo "" ;;
        35b-a3b-host:env)     echo "MOE_EXPERT_HEATMAP=load" ;;   # AGENTS rule 5: pin the residency prior

        4b-1gpu:n)            echo "$NPRED_CLI_DEFAULT" ;;
        phi4-1gpu:n)          echo "$NPRED_CLI_DEFAULT" ;;
        27b-3gpu-tensor:n)    echo 96 ;;     # the A2 anchor's -n
        35b-a3b-host:n)       echo 64 ;;     # the A3 anchor's -n

        # where the startup-shape expectation for this shape lives, and where it came from
        4b-1gpu:shape)        echo "$BASE/A6-startup-shape.txt" ;;
        phi4-1gpu:shape)      echo "$SHAPES_DIR/phi4-1gpu.shape" ;;
        27b-3gpu-tensor:shape) echo "$SHAPES_DIR/27b-3gpu-tensor.shape" ;;
        35b-a3b-host:shape)   echo "$SHAPES_DIR/35b-a3b-host.shape" ;;
        *) echo "" ;;
    esac
}
PAIR_ON_SHAPE="$SHAPES_DIR/$SENS_SHAPE.pair-on.shape"

# -----------------------------------------------------------------------------------------------------------
lock="${MTP_LOCK:-$HOME/.cache/rdna-boosts/gpu.lock}"
mkdir -p "$(dirname "$lock")"
exec 9>"$lock"
if ! flock -w "${LOCK_WAIT:-1800}" 9; then
    echo "FAIL: another GPU job holds $lock (waited ${LOCK_WAIT:-1800}s)" >&2
    exit 1
fi

fails=0
hard() { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }
skip() { echo "SKIP  $1"; }
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
LOGS=()

# the normalised STARTUP shape: every line up to (and including) `listening on`, `#` comments dropped
shape_of() {
    "$BASE/norm-log.sh" "$1" | awk '/listening on/{print; exit} {print}' | grep -v '^#'
}

anchor_of() {  # $1 = shape; prints "<chars> <sha> <provenance>" for the cli anchor (empty if absent)
    [ -f "$ANCHORS" ] || return 0
    awk -v s="$1" 'BEGIN{FS="\t"} !/^#/ && $1==s {print $2, $3, $4; exit}' "$ANCHORS"
}

record_anchor() {  # $1 = shape, $2 = chars, $3 = sha
    local prov="recorded $(date +%F) by --record (POST-CHANGE lock; inertness must come from a pre-change row)"
    if [ -f "$ANCHORS" ] && awk -v s="$1" 'BEGIN{FS="\t"} !/^#/ && $1==s {f=1} END{exit !f}' "$ANCHORS"; then
        if [ "$FORCE" = 1 ]; then
            grep -v "^$1	" "$ANCHORS" > "$ANCHORS.tmp" && mv "$ANCHORS.tmp" "$ANCHORS"
        else
            echo "      (anchor for $1 already exists -- NOT re-recorded; --force to replace it deliberately)"
            return 0
        fi
    fi
    [ -f "$ANCHORS" ] || printf '# shape\tchars\tsha\tprovenance\n' > "$ANCHORS"
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$prov" >> "$ANCHORS"
    echo "      RECORDED anchor: $1 chars=$2 sha=$3  ($prov)"
}

run_cli() {  # $1 = shape, $2 = tag; writes $OUT/$2.log and echoes the extracted "<n> chars sha=..." line
    local s="$1" tag="$2" model dev extra env n
    model="$(shape_var "$s" model)"; dev="$(shape_var "$s" devices)"
    extra="$(shape_var "$s" extra)"; env="$(shape_var "$s" env)"; n="$(shape_var "$s" n)"
    # shellcheck disable=SC2086
    local rc
    # `env` is explicit because `$env` is an expanded ASSIGNMENT LIST (`MOE_EXPERT_HEATMAP=load`): written as
    # `FOO=1 $env cmd` the shell treats the expansion's first word as the COMMAND (rc=127, measured).
    env HIP_VISIBLE_DEVICES="$dev" ROCBLAS_USE_HIPBLASLT=0 $env timeout 1800 "$BIN/llama-cli" \
        -m "$model" -ngl 99 $extra \
        -p "$(cat "$REPO/prompts/prose-rdna-boosts.txt")" -n "$n" --seed 42 --temp 0 \
        --no-display-prompt --single-turn > "$OUT/$tag.log" 2>&1
    rc=$?
    if [ "$rc" != 0 ]; then
        echo "CLI-FAILED rc=$rc" >&2
        tail -5 "$OUT/$tag.log" >&2
        return 1
    fi
    "$REPO/scripts/extract-generated.py" "$OUT/$tag.log"
}

run_srv() {  # $1 = shape, $2 = tag, $3 = extra arm args
    local s="$1" tag="$2" arm="${3:-}" model dev extra env
    model="$(shape_var "$s" model)"; dev="$(shape_var "$s" devices)"
    extra="$(shape_var "$s" extra)"; env="$(shape_var "$s" env)"
    HIP_VISIBLE_DEVICES="$dev" timeout 1800 "$BASE/server-arm.sh" --out "$OUT" --tag "$tag" \
        --model "$model" --np 2 --ctx 8192 --n-predict 48 --ready 300 --req-timeout 600 \
        --env "$env" --extra "$extra $arm" > "$OUT/$tag.out" 2>&1
}

echo "==> default-inert: pair OFF, shapes [$SHAPES]; sensitivity shape=$SENS_SHAPE; anchors=$ANCHORS"
IFS=',' read -r -a want <<<"$SHAPES"

for s in "${want[@]}"; do
    [ -n "$(shape_var "$s" model)" ] || { echo "FAIL  unknown shape '$s'"; fails=$((fails+1)); continue; }
    echo
    echo "-- $s  ($(basename "$(shape_var "$s" model)"), devices $(shape_var "$s" devices) '$(shape_var "$s" extra)')"

    # ---- 1. the greedy TEXT -------------------------------------------------------------------------------
    if [ "$DO_CLI" = 1 ]; then
        reads=1
        [ "$s" = "$SENS_SHAPE" ] && reads=2        # RUN-STABILITY on the sensitivity shape
        got=""; stab=1
        for i in $(seq 1 "$reads"); do
            line="$(run_cli "$s" "$s-cli$i")" || { stab=0; break; }
            echo "      cli[$i]: $line"
            LOGS+=("$OUT/$s-cli$i.log")
            if [ -z "$got" ]; then got="$line"; elif [ "$line" != "$got" ]; then stab=0; fi
        done
        if [ "${reads}" -gt 1 ]; then
            hard "$stab" "the cli text is RUN-STABLE over $reads fresh processes (an unstable anchor cannot catch a change)"
        fi
        if [ "$stab" = 1 ]; then
            chars="$(sed -n 's/^\([0-9]*\) chars sha=.*/\1/p' <<<"$got")"
            sha="$(sed -n 's/.*sha=//p' <<<"$got")"
            read -r a_chars a_sha a_prov <<<"$(anchor_of "$s")"
            if [ -z "${a_sha:-}" ]; then
                if [ "$RECORD" = 1 ]; then
                    record_anchor "$s" "$chars" "$sha"
                    hard 0 "no cli anchor for $s -- recorded it; the next run asserts it"
                else
                    echo "FAIL  no cli anchor for $s in $ANCHORS (run this gate with --record once)"; fails=$((fails+1))
                fi
            else
                echo "      anchor: $a_chars chars sha=$a_sha  [$a_prov]"
                hard "$([ "$chars" = "$a_chars" ] && [ "$sha" = "$a_sha" ] && echo 1 || echo 0)" \
                    "the pair-OFF cli text is the anchor's ($chars/$sha vs $a_chars/$a_sha)"
            fi
        fi
    fi

    # ---- 2. the startup SHAPE -----------------------------------------------------------------------------
    if [ "$DO_SRV" = 1 ]; then
        if ! run_srv "$s" "$s-srv-off"; then
            echo "FAIL  the pair-OFF server arm did not complete (see $OUT/$s-srv-off.out)"
            tail -5 "$OUT/$s-srv-off.out" 2>/dev/null | sed 's/^/      /'
            fails=$((fails+1))
        else
            LOGS+=("$OUT/$s-srv-off.log")
        fi
        exp="$(shape_var "$s" shape)"
        if [ -s "$OUT/$s-srv-off.log" ]; then
            shape_of "$OUT/$s-srv-off.log" > "$OUT/$s.shape.off"
            if [ -s "$exp" ]; then
                echo "      shape expectation: ${exp#$BASE/} ($(grep -vc '^#' "$exp") lines)"
                if diff -u <(grep -v '^#' "$exp") "$OUT/$s.shape.off" > "$OUT/$s.shape.diff" 2>&1; then
                    hard 1 "the pair-OFF startup shape is byte-identical to the recorded shape ($(wc -l < "$OUT/$s.shape.off") lines)"
                else
                    hard 0 "the pair-OFF startup shape CHANGED -- see $OUT/$s.shape.diff"
                    sed 's/^/      /' "$OUT/$s.shape.diff" | head -30
                fi
            elif [ "$RECORD" = 1 ]; then
                { echo "# recorded $(date +%F) by --record on the arms-v2 tree (POST-CHANGE lock)";
                  echo "# shape: $(basename "$(shape_var "$s" model)") devices $(shape_var "$s" devices) extra '$(shape_var "$s" extra)' np=2 ctx=8192";
                  cat "$OUT/$s.shape.off"; } > "$exp"
                echo "      RECORDED shape: ${exp#$BASE/} ($(wc -l < "$OUT/$s.shape.off") lines)"
                hard 0 "no recorded startup shape for $s -- recorded it; the next run asserts it"
            else
                echo "FAIL  no recorded startup shape for $s ($exp) -- run this gate with --record once"
                fails=$((fails+1))
            fi
        fi

        # ---- 3. SENSITIVITY: the same server with the pair ON --------------------------------------------
        if [ "$s" = "$SENS_SHAPE" ] && [ -s "$OUT/$s-srv-off.log" ]; then
            if ! run_srv "$s" "$s-srv-on" "--prefill-decode-arms"; then
                hard 0 "the pair-ON sensitivity arm did not complete (see $OUT/$s-srv-on.out)"
            else
                LOGS+=("$OUT/$s-srv-on.log")
                shape_of "$OUT/$s-srv-on.log" > "$OUT/$s.shape.on"
                if diff -q "$OUT/$s.shape.off" "$OUT/$s.shape.on" >/dev/null 2>&1; then
                    hard 0 "the pair-ON startup shape is IDENTICAL to the pair-OFF shape -- the shape comparison is VACUOUS"
                else
                    hard 1 "the shape comparison SEES the pair: the pair-ON shape differs ($(diff "$OUT/$s.shape.off" "$OUT/$s.shape.on" | grep -c '^[<>]') line(s))"
                    if [ -s "$PAIR_ON_SHAPE" ]; then
                        if diff -u <(grep -v '^#' "$PAIR_ON_SHAPE") "$OUT/$s.shape.on" > "$OUT/$s.pair-on.diff" 2>&1; then
                            hard 1 "the pair-ON shape is the recorded pair-ON shape (${PAIR_ON_SHAPE#$BASE/})"
                        else
                            hard 0 "the pair-ON shape CHANGED -- see $OUT/$s.pair-on.diff"
                            sed 's/^/      /' "$OUT/$s.pair-on.diff" | head -30
                        fi
                    elif [ "$RECORD" = 1 ]; then
                        { echo "# recorded $(date +%F) by --record (SENSITIVITY fixture: the same shape with the pair ON)";
                          cat "$OUT/$s.shape.on"; } > "$PAIR_ON_SHAPE"
                        echo "      RECORDED shape: ${PAIR_ON_SHAPE#$BASE/}"
                    else
                        echo "FAIL  no recorded pair-ON shape ($PAIR_ON_SHAPE) -- run with --record once"
                        fails=$((fails+1))
                    fi
                fi
                # the pair changes latency, not answers
                off_h="$(sed 's/.*sha=//' "$OUT/$s-srv-off.hashes" 2>/dev/null | sort | tr '\n' ' ')"
                on_h="$(sed 's/.*sha=//' "$OUT/$s-srv-on.hashes" 2>/dev/null | sort | tr '\n' ' ')"
                echo "      responses: off={$off_h} on={$on_h}"
                hard "$([ -n "${off_h// /}" ] && [ "$off_h" = "$on_h" ] && echo 1 || echo 0)" \
                    "the pair-ON arm read the same responses as the pair-OFF arm (the pair is a latency change)"
            fi
        fi
    fi
done

echo
echo "--> GPU faults (+ the shape files are the decision record)"
fault_args=()
for l in "${LOGS[@]}"; do fault_args+=(--log "$l"); done
if [ "${#fault_args[@]}" -eq 0 ]; then
    skip "no logs to check"
elif "$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" "${fault_args[@]}" > "$OUT/faults.txt" 2>&1; then
    hard 1 "kernel journal and every arm log clean"
else
    hard 0 "GPU fault(s) -- see $OUT/faults.txt"; tail -3 "$OUT/faults.txt" | sed 's/^/      /'
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: with the pair OFF the default path prints the same shape and reads the same text"
    [ "$KEEP" = 1 ] && echo "    logs kept in $OUT"
    exit 0
fi
echo "==> FAIL ($fails check(s))  (logs: $OUT)"
exit 1
