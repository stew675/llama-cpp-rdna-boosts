#!/usr/bin/env bash
# Fused-op probe gate: a probed feature may not enable a graph the acting backend cannot execute
# (r308, the `--kv-unified` kv_rows defect), and a probe that cannot verify an op may not report
# "enabled" anyway (r311, TODO #345 -- the vacuity CLASS, not just that one probe).
#
# WHY THIS EXISTS
#   `llama_context::resolve_fused_ops()` decides a fused feature by building a PROBE graph on the reserve
#   memory context and checking where the fused node LANDED (a node the backend refuses goes to the CPU, so
#   the probe reads "not supported" and disables the feature).  That is only sound when the probe graph
#   actually CONTAINS the node.  Two independent ways it can fail to:
#     * the reserve context's `n_kv` is `kv->get_size()`, i.e. the whole PADDED cache extent, not the used
#       depth, and a model's decode arm can depend on that extent: qwen4exp (QSA) decodes dense only while
#       `n_kv < qsa_dense_decode_until`, which is 32768 on gfx1151 (RDNA3_5) and unbounded on gfx1100/
#       gfx1201.  At `-c 32768` on gfx1151 the kv_rows probe took the SPARSE arm, which builds no
#       `ggml_flash_attn_ext_rows` node at all -- and because that probe did not require the node it
#       reported "enabled", while at decode time (a shallow used extent) the DENSE arm built the node, no
#       AMD kernel reads the rows form, and every joint attention layer ran on the CPU.  Silent: correct
#       text, 25 backend splits, plain decode 32.5 -> 11.0 t/s and MTP `draft-mtp n3` 18.7 -> 62-65 t/s
#       (post-fix) on Qwen3.8 Flash-Next IQ4_NL at 15.5K depth; prefill unaffected.  FIX: r308's
#       `require_observed` -- a probe that cannot verify an op DISABLES it.
#     * the op can be PREFILL-only, or belong to another arch entirely, on a shape the probe's 1-token
#       graph can never reach: qwen4exp's DSV4 HC pre/post ops only exist above `HC_FUSED_MAX_TOKENS` (its
#       single-token arm runs the fused HC_MIX/HC_COMBINE band), and TOP_K_MASK is built by the DSA arches
#       while qwen4exp's GGUF still carries `indexer_top_k` and therefore still arms that probe.  Those
#       cannot use `require_observed` (it would disable a feature that WORKS, and for DSV4_HC_POST on a
#       tensor split that is a numeric regression, not a perf one -- r309: the unfused chain reads NMSE
#       1.74e-01 under the Meta device), so r311's fix asks the acting DEVICE instead, via
#       `ggml_backend_dev_supports_op`.  Every probe in that class now says HOW it was verified.
#
# WHAT IT CHECKS (HARD unless marked INFO)
#   * no arm aborts, and `scripts/check-gpu-faults.sh` reads the kernel journal clean over this gate's window
#   * **the rows form never runs on the CPU.**  That is the r307 defect, and it is graded where it is visible:
#     the probe's placement line (`... but Flash Attention over K/V rows is assigned to device CPU`) and the
#     `splits = N` bound below.  Whether `enabled` is ACCEPTED on a box depends on the box's HIP FA kernels: since
#     r358 (issue #363) an AMD MMA kernel (CDNA MFMA / RDNA WMMA, GQA-optimised layout) reads the rows form, so on
#     an RDNA box `enabled` is the CORRECT verdict and the defect this gate hunts is a CPU placement, not the
#     verdict.  On a box whose HIP backend cannot read it, the same placement line must appear (the decline) --
#     `KVROWS_EXPECT=enable|obs|place` pins which box you mean, `auto` accepts either as long as no CPU placement
#     and no extra split appears.  A bare `enabled` claim is still impossible to fake: `require_observed` (r308)
#     makes the probe disable the feature when its graph never built the node.
#   * no `sched_graph_realloc` line in an arm reports more backend segments than its LAYOUT can produce --
#     `SPLITS_MAX`, 1 on a single device or a tensor split (the Meta backend is one backend) and `NGPU + 1`
#     on a layer split (the output layer is placed back on the first device, so 3 devices read 4).  A fused
#     node refused by a device inserts an extra CPU segment and breaks the bound; r307's decode graph read
#     25 against a bound of 1.
#   * **(r311) every `enabled` line for the audited probes carries a VERDICT.**  `...(the probe graph
#     contained the node and its placement was checked)` or `...(the probe graph built no such node; the
#     acting device accepts the op, verified by ggml_backend_dev_supports_op)`.  A bare
#     `fused <probe> enabled` FAILS: that form is exactly what r308 and r311 removed, and it is
#     indistinguishable from a vacuous pass.  Applies to the three DSV4 HC probes and the top-k mask.
#   * the FIXTURE LADDER (`test-llama-archs -a ARCH`) records, per (arch, probe), whether the node was
#     OBSERVED in the probe graph or the device was QUERIED -- the reachability audit as an assertion table
#     (`PROBE_AUDIT` below).  A change to a model builder, to a descriptor, or to a `cparams.fused_*`
#     default that moves a verdict fails here.  This is the cheap arm: no model load, one arch per arm,
#     ~2-5 s each, and it covers every arch that builds these ops.
#   * **(r359, issue #365) the DELIVERY DEFAULT takes the DENSE form**: the `default` arm sets no env at all
#     and must log `K/V rows attention: dense form` and never a rows-form claim.  The upstream form became
#     opt-in (`LLAMA_KV_ROWS_DISABLE=0`) after it measured 17-59 % of decode at 90k depth on gfx1201 and
#     cost the derived (mask-free) kq mask -- measurements in `INVARIANTS.md` r365.  Every arm that
#     exercises the rows form below therefore ASKS for it; `rows-off` is the same default spelled out and
#     must read the same text as `default`.
#   * the REAL-MODEL arms (llama-server, the default): the kv_rows behaviour r308 fixed --
#     `dense-always` arm (the QSA policy gfx1100/gfx1201 use) and the `sparse` arm (`LLAMA_QSA_DENSE_
#     DECODE_UNTIL=0`, which forces the same vacuity on ANY arch): `sparse` declines the op BY OBSERVATION
#     (`was not used in the probe graph`) -- that line IS the fix firing -- and `dense-always` declines BY
#     PLACEMENT (`is assigned to device CPU`) on a box whose HIP backend cannot read the rows form, or reads
#     `enabled` on one that can (r358: the AMD MMA kernel does).  Either way the hard assertions are that the
#     rows form is never PLACED ON THE CPU and that no graph gains a CPU segment.  The MTP arms: the verify band
#     (`n_tokens <= max(8, n_rs_batch)`) takes the same dense arm as W=1, so a drafter multiplies the defect by
#     every verify step and adds the drafter's OWN unified context.  Greedy text is byte-identical across the
#     arms that decode the same regime.  INFO generation t/s per arm; NOT a threshold.
#
#   WHAT IT CANNOT FORCE.  The kv_rows vacuity needs a model whose probe arm differs from its runtime arm,
#   i.e. qwen4exp (QSA) on gfx1151 at `-c >= 32768`; `MODEL` defaults to a Flash-Next GGUF for that reason.
#   **The r308 ARMS ARE BOX-DEPENDENT AS SHIPPED** -- `plain`/`mtp` leave `LLAMA_QSA_DENSE_DECODE_UNTIL` at
#   the box default, which is the SPARSE policy on gfx1151 (32768) and the DENSE policy on gfx1201
#   (unbounded), so `plain` declines by OBSERVATION on halo and by PLACEMENT on the R9700 box.  The
#   expectation is therefore derived from the observed verdict rather than pinned: use
#   `KVROWS_EXPECT=obs|place` (or leave it `auto`) and run the arm on the box whose policy you mean.  The
#   fixture ladder has no such dependence and is the class-wide instrument.
#   `--log FILE` grades an existing log instead of running arms; that is the negative test: a log carrying
#   r307's two decisive lines (`Flash Attention over K/V rows enabled`, `splits = 25`) must FAIL.
#
# Usage:
#   ./gate-fused-probe-splits.sh [--log FILE]... [--keep] [--no-faults] [--no-mtp]
#                                [--no-fixture] [--fixture-only] [--fixture-archs "A B C"]
#
# Environment:
#   LLAMA_SERVER  llama-server binary (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   TEST_ARCHS    test-llama-archs binary (default: ~/llama.cpp/build-rocm/bin/test-llama-archs)
#   MODEL         a qwen4exp (QSA) GGUF   (default: Qwen3.8 Flash-Next IQ4_NL, shard 1)
#   DRAFT         MTP drafter GGUF for the MTP arms (default: the model dir's `mtp-*.gguf`; "" disables)
#   PROMPT        prose prompt (default: prompts/prose-rdna-boosts.txt, sha256-pinned there)
#   CTX           -c (default 32768 -- the gfx1151 QSA dense-decode crossover, which is what makes the default
#                 arm the vacuous one; `CTX=16384` makes it take the dense probe arm instead)
#   NGEN          decoded tokens (default 48)
#   NGPU          GPU count for HIP_VISIBLE_DEVICES (default 1)
#   KV            KV cache type (default bf16 -- the field report's `-ctk bf16 -ctv bf16`)
#   EXTRA_ARGS    extra llama-server args (default: the field report's MoE flags for the default MODEL)
#   KVROWS_EXPECT obs|place|enable|auto -- the kv_rows verdict `plain`/`mtp` must produce.  `auto` (default) derives
#                 it from the arm's own log, so the gate is box-independent; `enable` pins a box whose HIP FA kernel
#                 reads the rows form (r358: RDNA3/3.5/4 WMMA and CDNA MFMA with the GQA-optimised layout), `place`
#                 a box whose backend refuses it (the pre-r358 AMD state).  `dense-always`/`mtp-rdna34` accept either
#                 -- the CPU-placement check and the `splits` bound are the hard part, never the verdict alone.
#   SPLITS_MAX    the backend-segment bound (default: 1 on NGPU=1, NGPU+1 otherwise -- see above)
#   ARMS          a space-separated subset of: default plain dense-always sparse rows-off mtp mtp-rdna34
#                 (`default` = no env = the r359 delivery default, dense; every other arm but `rows-off`
#                 asks for the rows form with LLAMA_KV_ROWS_DISABLE=0)
#   PORT          base port (default 18113; each arm takes PORT+index)
#
# Exit: 0 = PASS, 1 = FAIL (gate), 2 = usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
TEST_ARCHS="${TEST_ARCHS:-$HOME/llama.cpp/build-rocm/bin/test-llama-archs}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf}"
if [ -z "${DRAFT+y}" ]; then
    DRAFT="$(ls /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-*.gguf 2>/dev/null | head -1)"
fi
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
CTX="${CTX:-32768}"
NGEN="${NGEN:-48}"
NGPU="${NGPU:-1}"
KV="${KV:-bf16}"
PORT="${PORT:-18113}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KVROWS_EXPECT="${KVROWS_EXPECT:-auto}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"

# The reachability audit, as an assertion table (r311, TODO #345).  Columns:
#   arch | HC pre | HC comb | HC post | top-k mask
# obs   = the probe graph CONTAINED the node, so the placement verdict decided it
# query = the probe graph did NOT contain it, so the acting device was asked directly
# any   = not pinned.  HC post is `any` because `cparams.fused_dsv4_hc_post = meta_split_active`, i.e. it is
#         DISABLED (and logs nothing) on a single device and only probed on a Meta/tensor arm -- with
#         NGPU >= 2 the gate pins it separately below.
# `xing4-0` and `hy-v4` also build DSV4-HC / LIGHTNING_INDEXER nodes but `test-llama-archs` has no
# fixture for them and this repo has no GGUF of either: record them as an uncovered arch, do not add
# a row that cannot run.
# NOTE the qwen4exp row is the FIXTURE's verdict, and the fixture's hc weights are F32/F16, so
# `build_hc_mix` does not take its Q8_0/BF16 fused band and the 1-token graph builds DSV4_HC_PRE after
# all (obs).  The REAL Flash-Next GGUF takes the band and reads `query` -- that arm is the llama-server
# one, and the audit's qwen4exp-HC-pre vacuity is only reproducible there.
PROBE_AUDIT="
qwen4exp|obs|query|any|query
deepseek4|obs|obs|any|obs
deepseek41|obs|obs|any|query
glm5-next|obs|obs|any|obs
kimi-k31|query|query|any|obs
deepseek32|query|query|any|obs
dots3note|query|query|any|obs
glm-dsa|query|query|any|obs
mistral-large4|query|query|any|obs
"

LOG_ONLY=()
KEEP=0
FAULTS=1
MTP=1
FIXTURE=1
FIXTURE_ONLY=0
FIXTURE_ARCHS="qwen4exp deepseek4 deepseek41 glm5-next kimi-k31 deepseek32 dots3note glm-dsa mistral-large4"
while [ $# -gt 0 ]; do
    case "$1" in
        --log)          [ $# -ge 2 ] || { echo "--log needs a file" >&2; exit 2; }; LOG_ONLY+=("$2"); shift 2 ;;
        --keep)         KEEP=1; shift ;;
        --no-faults)    FAULTS=0; shift ;;
        --no-mtp)       MTP=0; shift ;;
        --no-fixture)   FIXTURE=0; shift ;;
        --fixture-only) FIXTURE_ONLY=1; shift ;;
        --fixture-archs) [ $# -ge 2 ] || { echo "--fixture-archs needs a list" >&2; exit 2; }
                         FIXTURE_ARCHS="$2"; shift 2 ;;
        -h|--help)      sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

FAIL=0
fail() { echo "FAIL: $*" >&2; FAIL=1; }
pass() { echo "PASS: $*"; }
info() { echo "INFO: $*"; }

# ---- the log assertions, shared by the live arms, the fixture ladder and --log mode ----------------
# The split bound.  `ggml_backend_sched` counts one segment per backend BOUNDARY, so the number is a
# property of the LAYOUT, not of the probe: 1 on a single device and on a tensor split (the Meta backend
# is one backend), and N+1 on an N-device layer split (the output layer is placed back on the first
# device, so the tail is a fourth segment at NGPU=3 -- measured 4, not 3).  A fused node refused by a
# device inserts an extra CPU segment, which the bound still catches.  SPLITS_MAX overrides it.
if [ -z "${SPLITS_MAX:-}" ]; then
    if [ "$NGPU" = 1 ]; then SPLITS_MAX=1; else SPLITS_MAX=$((NGPU + 1)); fi
fi

CHECKS=()

# ---- the log assertions, shared by the live arms, the fixture ladder and --log mode ----------------
# the four probes whose descriptor sets `query_supports_op` (r311): for these a bare `enabled` is the
# r308/r311 defect, because it does not say whether the graph contained the node or the device was asked
AUDITED_PROBES=(
    "fused DeepSeek V4 HC pre"
    "fused DeepSeek V4 HC comb"
    "fused DeepSeek V4 HC post"
    "fused top-k mask"
)

grade_log() { # tag logfile [want_splits]
    local tag="$1" log="$2" want_splits="${3:-1}"
    if [ ! -s "$log" ]; then
        fail "$tag: no log at $log"
        return
    fi
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault|Aborted" "$log"; then
        fail "$tag: the process aborted (see $log)"
    fi
    if grep -q "but Flash Attention over K/V rows is assigned to device CPU" "$log"; then
        fail "$tag: the rows form was placed on the CPU -- that is the r307 defect (one CPU attention launch per joint layer per step, 3x decode loss); see $log"
    fi
    if [ "$want_splits" = 1 ]; then
        local worst
        worst="$(sed -nE 's/.*sched_graph_realloc: graph reallocation: .*splits = ([0-9]+).*/\1/p' "$log" | sort -n | tail -1)"
        worst="${worst:-0}"
        if [ "$worst" -gt "$SPLITS_MAX" ]; then
            fail "$tag: a graph was split across $worst backends (the layout allows $SPLITS_MAX) -- a fused node was refused by the device and placed on the CPU"
        else
            pass "$tag: every graph stayed within the layout's $SPLITS_MAX segment(s) (worst splits = ${worst:-none})"
        fi
    fi

    # r311: an `enabled` claim for an audited probe must carry its verdict
    local name bad=0
    for name in "${AUDITED_PROBES[@]}"; do
        local line
        while IFS= read -r line; do
            [ -n "$line" ] || continue
            case "$line" in
                *"$name enabled ("*) : ;;
                *) fail "$tag: UNVERIFIED 'enabled' for '$name' -- the line carries no verdict, so it cannot be told from r308's vacuous pass: $line"
                   bad=1 ;;
            esac
        done < <(grep -F ": $name enabled" "$log")
    done
    [ "$bad" = 0 ] && pass "$tag: every audited-probe 'enabled' line carries its verdict"
}

# probe_verdict log name -> a space-separated set of: obs query off (empty = never logged)
probe_verdict() { # logfile name
    local log="$1" name="$2" out=""
    grep -qF ": $name enabled (the probe graph contained"       "$log" && out="$out obs"
    grep -qF ": $name enabled (the probe graph built no such"   "$log" && out="$out query"
    # a refusal is also a verdict (the device was asked, or the node was seen on the CPU)
    grep -qF "$name not supported, set to disabled"             "$log" && out="$out off"
    grep -qF "$name: layer "                                    "$log" && out="$out off"
    echo $out
}

# ---- the fixture ladder: the reachability audit as an assertion table ------------------------------
case "$NGPU" in 1) DEVS=0 ;; 2) DEVS=0,1 ;; 3) DEVS=0,1,2 ;; 4) DEVS=0,1,2,3 ;; *) DEVS="$NGPU" ;; esac
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

run_fixture() {
    local arch="$1" log="$KEEPOUT/fixture-$arch.log"
    echo "  -> $arch"
    HIP_VISIBLE_DEVICES="$DEVS" timeout 900 "$TEST_ARCHS" -a "$arch" -v 4 > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "fixture/$arch: test-llama-archs exited $rc (see $log)"
        return
    fi
    if ! grep -q "Summary: all" "$log"; then
        fail "fixture/$arch: no test summary -- the fixture did not run (see $log)"
        return
    fi
    if grep -q "no tests executed" "$log"; then
        fail "fixture/$arch: test-llama-archs executed no tests for this arch (see $log)"
        return
    fi
    grade_log "fixture/$arch" "$log" 0

    # HC post is only enabled on a Meta/tensor arm (`cparams.fused_dsv4_hc_post = meta_split_active`), so
    # a multi-device run that actually builds one is the only way the gate reaches that probe -- pin it
    # there.  Only some arches support a tensor split in the fixture (`llama_arch_supports_sm_tensor`), so
    # the condition is "this log built a Meta device", not "NGPU > 1".
    if grep -q "creating a Meta device" "$log"; then
        local hp
        hp="$(probe_verdict "$log" "fused DeepSeek V4 HC post")"
        if [ -z "$hp" ]; then
            fail "fixture/$arch: 'fused DeepSeek V4 HC post' was never reported although this run built a Meta device -- the tensor arm should enable and probe it (see $log)"
        else
            pass "fixture/$arch: 'fused DeepSeek V4 HC post' verdict='$hp' on the Meta arm"
        fi
    fi

    local row pre comb post topk
    row="$(echo "$PROBE_AUDIT" | grep -E "^${arch}\|" | head -1)"
    if [ -z "$row" ]; then
        fail "fixture/$arch: no PROBE_AUDIT row for this arch -- record the audit before adding it"
        return
    fi
    IFS='|' read -r _ pre comb post topk <<< "$row"
    local names=("fused DeepSeek V4 HC pre" "fused DeepSeek V4 HC comb" "fused DeepSeek V4 HC post" "fused top-k mask")
    local wants=("$pre" "$comb" "$post" "$topk")
    local idx name want got
    for idx in 0 1 2 3; do
        name="${names[$idx]}"; want="${wants[$idx]}"
        got="$(probe_verdict "$log" "$name")"
        if [ "$want" = "any" ]; then
            info "fixture/$arch: '$name' verdict='${got:-none}' (not pinned)"
            continue
        fi
        if [ -z "$got" ]; then
            fail "fixture/$arch: '$name' was never reported -- the probe did not run (see $log)"
            continue
        fi
        case " $got " in
            *" $want "*) pass "fixture/$arch: '$name' verdict=$want (the recorded audit)" ;;
            *) fail "fixture/$arch: '$name' verdict is '$got', the recorded audit says '$want' -- a builder, a descriptor or a cparams default moved; re-audit and update PROBE_AUDIT (see $log)" ;;
        esac
    done
}

if [ "${#LOG_ONLY[@]}" -gt 0 ]; then
    echo "==> fused-op probe gate: grading ${#LOG_ONLY[@]} existing log(s)"
    for f in "${LOG_ONLY[@]}"; do grade_log "$(basename "$f")" "$f"; done
    echo
    [ "$FAIL" = 0 ] && { echo "RESULT: PASS"; exit 0; }
    echo "RESULT: FAIL"; exit 1
fi

if [ "$FIXTURE" = 1 ]; then
    if [ ! -x "$TEST_ARCHS" ]; then
        echo "ERROR: no test-llama-archs at $TEST_ARCHS (set TEST_ARCHS, or --no-fixture)" >&2
        exit 2
    fi
    echo "==> fused-op probe gate (r311): fixture ladder -- the reachability audit, one arch per arm"
    echo "    test-llama-archs=$TEST_ARCHS  devices=$DEVS  archs='$FIXTURE_ARCHS'"
    echo "    the fixture builds each arch's graph from metadata (no model load); xing4-0/hy-v4 are uncovered"
    for arch in $FIXTURE_ARCHS; do
        run_fixture "$arch"
    done
    echo
fi

if [ "$FIXTURE_ONLY" = 1 ]; then
    if [ "$FAULTS" = 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
        flogs=(); for arch in $FIXTURE_ARCHS; do flogs+=(--log "$KEEPOUT/fixture-$arch.log"); done
        if "$REPO_DIR/scripts/check-gpu-faults.sh" ${GATE_SINCE_TS:+--since "$GATE_SINCE_TS"} "${flogs[@]}" > "$KEEPOUT/faults.log" 2>&1; then
            pass "no GPU faults in the kernel journal over the fixture ladder"
        else
            fail "GPU faults reported (see $KEEPOUT/faults.log)"
            sed -n '1,20p' "$KEEPOUT/faults.log" >&2
        fi
    fi
    echo
    if [ "$FAIL" = 0 ]; then
        echo "RESULT: PASS"
        [ "$KEEP" = 1 ] && echo "logs kept in $KEEPOUT"
        exit 0
    fi
    echo "RESULT: FAIL"
    [ "$KEEP" = 1 ] && echo "logs kept in $KEEPOUT"
    exit 1
fi

# ---- the real-model arms (r308's kv_rows shape) ----------------------------------------------------
[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$PROMPT" ]       || { echo "ERROR: no prompt at $PROMPT" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required" >&2; exit 2; }
command -v curl   >/dev/null 2>&1 || { echo "ERROR: curl is required" >&2; exit 2; }

if [ "$MTP" = 1 ] && { [ -z "$DRAFT" ] || [ ! -f "$DRAFT" ]; }; then
    echo "note: no MTP drafter (DRAFT='${DRAFT:-}'), the mtp arms are skipped" >&2
    MTP=0
fi

ARMS="${ARMS:-default plain dense-always sparse rows-off}"
[ "$MTP" = 1 ] && ARMS="$ARMS mtp mtp-rdna34"

# arm -> env assignment (empty = none).  The policy arms re-create another arch's QSA policy on this box:
# 1000000000 == "dense always" (gfx1100/gfx1201) and 0 == "sparse always" (forces the vacuity anywhere).
arm_env() {
    # r359: the rows form is OPT-IN (`LLAMA_KV_ROWS_DISABLE=0` -- upstream's "do not disable"), so every arm
    # that is meant to EXERCISE it asks for it, `rows-off` is the same delivery default spelled out, and
    # `default` is the delivery default with no env at all (the arm that asserts the opt-in is off).
    case "$1" in
        default)      echo "" ;;
        plain)        echo "LLAMA_KV_ROWS_DISABLE=0" ;;
        dense-always) echo "LLAMA_KV_ROWS_DISABLE=0 LLAMA_QSA_DENSE_DECODE_UNTIL=1000000000" ;;
        sparse)       echo "LLAMA_KV_ROWS_DISABLE=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0" ;;
        rows-off)     echo "LLAMA_KV_ROWS_DISABLE=1" ;;
        mtp)          echo "LLAMA_KV_ROWS_DISABLE=0" ;;
        mtp-rdna34)   echo "LLAMA_KV_ROWS_DISABLE=0 LLAMA_QSA_DENSE_DECODE_UNTIL=1000000000" ;;
        *)            echo "ERROR" ;;
    esac
}
arm_draft() { case "$1" in mtp|mtp-rdna34) echo 1 ;; *) echo 0 ;; esac; }

SRV_PID=""
stop_arm() {
    if [ -n "${SRV_PID:-}" ]; then
        kill "$SRV_PID" 2>/dev/null
        for _ in $(seq 1 120); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
        SRV_PID=""
    fi
}
trap 'stop_arm; [ "$KEEP" = 1 ] || rm -rf "$KEEPOUT"' EXIT

# one arm = one server start: the census comes from the LOG (-lv 4 is safe here because the generated text
# comes back over HTTP -- AGENTS.md's -lv table), the text from the /completion response.
run_arm() { # arm port env-assignment
    local arm="$1" port="$2" envs="$3"
    local log="$KEEPOUT/$arm.log"
    local draft=()
    [ "$(arm_draft "$arm")" = 1 ] && draft=(--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-model "$DRAFT")
    # shellcheck disable=SC2086
    env HIP_VISIBLE_DEVICES="$DEVS" $envs \
        "$LLAMA_SERVER" -m "$MODEL" -ngl 99 -c "$CTX" -t 8 -lv 4 \
        --host 127.0.0.1 --port "$port" --parallel 2 --kv-unified \
        -ctk "$KV" -ctv "$KV" --temp 0 --top-k 1 "${draft[@]}" $EXTRA_ARGS > "$log" 2>&1 &
    SRV_PID=$!

    local up=0
    for _ in $(seq 1 1800); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then up=1; break; fi
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        fail "$arm: the server did not come up (see $log)"
        stop_arm
        return 1
    fi

    python3 - "$port" "$NGEN" "$PROMPT" "$KEEPOUT/$arm.text" <<'PY'
import json, sys, urllib.request
port, ngen, prompt_file, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
body = json.dumps({"prompt": open(prompt_file, encoding="utf-8").read(), "n_predict": ngen,
                   "temperature": 0, "top_k": 1, "cache_prompt": False}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                             headers={"Content-Type": "application/json"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=7200))
    open(out, "w", encoding="utf-8").write(d.get("content") or "")
    t = (d.get("timings") or {})
    print("    tg=%.1f t/s" % (t.get("predicted_per_second") or 0))
except Exception as e:
    open(out, "w", encoding="utf-8").write("")
    print("    request failed: %s" % e)
    sys.exit(1)
PY
    local rc=$?
    stop_arm
    sleep 2
    return $rc
}

echo "==> fused-op probe gate: kv_rows probe + backend splits (r308), real-model arms"
echo "    server=$LLAMA_SERVER"
echo "    model=$MODEL"
[ "$MTP" = 1 ] && echo "    draft=$DRAFT"
echo "    devices=$DEVS  ctx=$CTX  ngen=$NGEN  kv=$KV  extra='$EXTRA_ARGS'"
echo "    arms=$ARMS  kvrows_expect=$KVROWS_EXPECT"

i=0
for arm in $ARMS; do
    port=$((PORT + i)); i=$((i + 1))
    envs="$(arm_env "$arm")"
    if [ "$envs" = "ERROR" ]; then fail "unknown arm '$arm'"; continue; fi
    run_arm "$arm" "$port" "$envs" || continue
    grade_log "$arm" "$KEEPOUT/$arm.log"
done

text_of() {
    local f="$KEEPOUT/$1.text"
    [ -s "$f" ] || { echo "none"; return; }
    sha256sum "$f" | cut -c1-12
}
decline_of() { # arm -> "obs" | "place" | "both" | "enable" | "none" | "missing"  (kv_rows probe only)
    local log="$KEEPOUT/$1.log"
    [ -s "$log" ] || { echo missing; return; }
    # scoped to the kv_rows probe: other probes print the same strings (the derived kq mask observes-and-
    # declines on many shapes), and grepping them unscoped reports a decline that belongs to a different
    # probe -- the r308 gate only had one probe that could print either
    local o=0 p=0 e=0
    grep -q "Flash Attention over K/V rows was not used in the probe graph, set to disabled" "$log" && o=1
    grep -q "is assigned to device CPU (usually due to missing support)"                     "$log" && p=1
    grep -q "Flash Attention over K/V rows enabled"                                          "$log" && e=1
    if [ "$o" = 1 ] && [ "$p" = 1 ]; then echo both
    elif [ "$o" = 1 ]; then echo obs
    elif [ "$p" = 1 ]; then echo place
    elif [ "$e" = 1 ]; then echo enable
    else echo none; fi
}

echo
for arm in $ARMS; do
    echo "    $arm: text sha=$(text_of "$arm")  decline=$(decline_of "$arm")  probes: $(probe_verdict "$KEEPOUT/$arm.log" "fused top-k mask" | tr -d '\n')"
done

# per-arm expectations that DISTINGUISH the decline mechanisms from a genuine enable
#   obs    declined BY OBSERVATION (r308: the probe graph contained no rows node)
#   place  declined BY PLACEMENT (the node landed on the CPU -- the box's HIP backend cannot read it)
#   enable the node was PROBED AND PLACED on the acting device, i.e. this box's HIP kernel reads the rows form
#          (r358, issue #363: the AMD MMA kernel does; the no-CPU-placement check above is what makes this safe)
#   any    any of the three, as long as the arm's log is clean (auto: KVROWS_EXPECT)
expect() { # arm want
    local arm="$1" want="$2" got
    got="$(decline_of "$arm")"
    case "$want" in
        any)  [ "$got" = none ] && fail "$arm: the probe neither observed, placed nor enabled the node -- it did not run (see $KEEPOUT/$arm.log)" \
                              || pass "$arm: kv_rows verdict = $got" ;;
        obs)  case "$got" in obs|both) pass "$arm: declined by OBSERVATION (was not used in the probe graph)" ;;
                                *) fail "$arm: expected the observation decline, got '$got' (see $KEEPOUT/$arm.log)" ;; esac ;;
        place) case "$got" in place|both) pass "$arm: declined by PLACEMENT (the node landed on the CPU)" ;;
                                  enable) info "$arm: enabled -- this box's HIP backend reads the rows form (r358); no CPU placement" ;;
                                  *) fail "$arm: expected the placement decline or an enable, got '$got' (see $KEEPOUT/$arm.log)" ;; esac ;;
        enable) case "$got" in enable) pass "$arm: enabled, i.e. the rows form runs on this box's GPU (r358)" ;;
                                   *) fail "$arm: expected 'enabled' (KVROWS_EXPECT=enable pins a box whose HIP kernel reads the rows form), got '$got'" ;; esac ;;
    esac
}
# `plain`/`mtp` run the BOX's own QSA policy, and that policy differs per arch: the sparse arm on gfx1151
# (dense-decode crossover 32768) and the dense arm on gfx1100/gfx1201 (unbounded).  So the expectation is
# derived from the arm itself unless KVROWS_EXPECT pins it -- a gate that only passes on one box is a
# hazard, and r308's own numbers were halo points.  Since r358 a SECOND box property decides the verdict
# (does the box's HIP FA kernel read the rows form), and `dense-always` used to decline BY PLACEMENT only
# because no AMD arch could -- it is box-derived now too, with the CPU-placement check above as the hard part.
box_want() { # arm -> obs|place|auto|enable|skip
    case "$1" in
        dense-always|mtp-rdna34) echo place ;;   # place on a box that cannot read the rows form, enable on one that can
        sparse)                  echo obs ;;     # arch-independent: the probe graph builds no rows node
        rows-off)                echo skip ;;    # LLAMA_KV_ROWS_DISABLE=1: the probe never runs (control)
        default)                 echo skip ;;    # r359: no env at all -- the opt-in must be OFF (asserted below)
        *)                       echo "$KVROWS_EXPECT" ;;
    esac
}
for arm in $ARMS; do
    want="$(box_want "$arm")"
    case "$want" in
        skip) : ;;
        auto) got="$(decline_of "$arm")"
              case "$got" in
                  obs|both|place|enable) info "$arm: box-policy verdict = $got (derived; KVROWS_EXPECT pins it)" ;;
                  none)  fail "$arm: the box's own policy neither observed, placed nor enabled the node -- the probe did not run (see $KEEPOUT/$arm.log)" ;;
                  *)     : ;;   # missing: already failed by run_arm
              esac ;;
        *)    expect "$arm" "$want" ;;
    esac
done
# the `rows-off` control must not build an enabled claim at all
case " $ARMS " in
    *" rows-off "*) grep -q "Flash Attention over K/V rows enabled" "$KEEPOUT/rows-off.log" 2>/dev/null \
        && fail "rows-off: the control arm reported the rows form enabled" || pass "rows-off: no rows-form claim (control)" ;;
esac
# r359: the DELIVERY DEFAULT must be the dense form, with the probe never run.  This is the regression
# (issue #365: the rows form cost 17-59 % of decode at depth, and its materialised compacted mask cost the
# derived kq mask), so it is asserted, not reported.
case " $ARMS " in
    *" default "*)
        if grep -q "Flash Attention over K/V rows enabled" "$KEEPOUT/default.log" 2>/dev/null; then
            fail "default: the delivery default reported the rows form enabled -- it is opt-in since r359 (LLAMA_KV_ROWS_DISABLE=0), see INVARIANTS.md r365"
        elif grep -q "K/V rows attention: dense form" "$KEEPOUT/default.log" 2>/dev/null; then
            pass "default: the delivery default takes the dense form and never probes the rows form (opt-in, r359)"
        else
            fail "default: no 'K/V rows attention: dense form' line -- the r359 opt-in decision did not run (see $KEEPOUT/default.log)"
        fi
        if grep -q "K/V rows attention: requested" "$KEEPOUT/default.log" 2>/dev/null; then
            fail "default: the opt-in was requested on an arm that sets no env -- something else turned the rows form on"
        else
            pass "default: nothing turned the rows form on without the opt-in"
        fi ;;
esac

# the text must be identical across arms that decode the SAME regime.  `rows-off` is deliberately the OTHER
# regime (the dense mask form), so it is only comparable when `plain` declined the rows form too (a box whose
# HIP backend cannot read it, i.e. the pre-r358 state); otherwise the two are different kernel forms and a
# near-tie may legitimately flip, which is reported and never asserted.
same_text() { # label a b
    local ta tb
    ta="$(text_of "$1")"; tb="$(text_of "$2")"
    if [ "$ta" = none ] || [ "$tb" = none ]; then
        fail "$3: one of the arms produced no text ($1=$ta $2=$tb)"
    elif [ "$ta" = "$tb" ]; then
        pass "$3: identical text (sha=$ta)"
    else
        fail "$3: text differs ($1=$ta $2=$tb)"
    fi
}
HAVE() { case " $ARMS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
rows_regime() { # arm -> rows|dense
    case "$(decline_of "$1")" in enable) echo rows ;; *) echo dense ;; esac
}
if HAVE default && HAVE rows-off; then same_text default rows-off "default vs rows-off (both the dense form)"; fi
if HAVE plain && HAVE rows-off && HAVE dense-always; then
    if [ "$(rows_regime plain)" = "$(rows_regime rows-off)" ]; then
        same_text plain rows-off "plain vs rows-off (same rows regime)"
    else
        info "plain(rows sha=$(text_of plain)) vs rows-off(dense sha=$(text_of rows-off)): different K/V row forms by design, text reported not asserted (r358)"
    fi
    same_text plain dense-always "plain vs dense-always (same QSA policy AND same rows regime on this box)"
elif HAVE plain && HAVE rows-off; then
    if [ "$(rows_regime plain)" = "$(rows_regime rows-off)" ]; then
        same_text plain rows-off "plain vs rows-off (same rows regime)"
    else
        info "plain(rows sha=$(text_of plain)) vs rows-off(dense sha=$(text_of rows-off)): different K/V row forms by design, text reported not asserted (r358)"
    fi
fi
if HAVE plain && HAVE dense-always && ! HAVE rows-off; then same_text plain dense-always "plain vs dense-always"; fi
if HAVE mtp   && HAVE mtp-rdna34; then same_text mtp   mtp-rdna34   "mtp vs mtp-rdna34";   fi
if HAVE sparse; then info "sparse arm text sha=$(text_of sparse) (a different decode regime by design; reported, not asserted)"; fi

if [ "$FAULTS" = 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    logs=(); for arm in $ARMS; do logs+=(--log "$KEEPOUT/$arm.log"); done
    if "$REPO_DIR/scripts/check-gpu-faults.sh" ${GATE_SINCE_TS:+--since "$GATE_SINCE_TS"} "${logs[@]}" > "$KEEPOUT/faults.log" 2>&1; then
        pass "no GPU faults in the kernel journal over this gate's window"
    else
        fail "GPU faults reported (see $KEEPOUT/faults.log)"
        sed -n '1,20p' "$KEEPOUT/faults.log" >&2
    fi
elif [ "$FAULTS" = 1 ]; then
    echo "note: scripts/check-gpu-faults.sh not executable, fault check skipped"
fi

echo
if [ "$FAIL" = 0 ]; then
    echo "RESULT: PASS"
    [ "$KEEP" = 1 ] && echo "logs kept in $KEEPOUT"
    exit 0
fi
echo "RESULT: FAIL"
[ "$KEEP" = 1 ] && echo "logs kept in $KEEPOUT"
exit 1
