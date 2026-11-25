#!/usr/bin/env bash
# The TQ1_0 / TQ2_0 (ternary) GPU gate -- block 17 (contributor PR #358, @overdoingism, r356).
#
# WHY THIS EXISTS
#   Block 17 gives the CUDA / HIP backend a MUL_MAT (and GET_ROWS, and the fp16/fp32/bf16 conversions) for the
#   ternary weight types, so a BitNet-style model runs on the GPU instead of leaving every matmul on the CPU
#   backend.  The block's whole design claim is that its result is BIT-IDENTICAL to the CPU backend's x86 AVX2
#   path -- the quantisation, the eight-lane integer dot, the per-lane accumulation and the final reduction all
#   reproduce `ggml_vec_dot_tq{1,2}_0_q8_K`, so a TQ model's text is the CPU's text and not a near-tie.
#
#   That claim rests on ONE detail nobody can see: the CPU's
#       sumf = _mm256_add_ps(_mm256_mul_ps(cvtepi32_ps(sumi0), d), sumf);
#   is NOT contracted into an fma by a Release clang/GCC build of ggml-cpu (the disassembly of both
#   ggml_vec_dot_tq*_0_q8_K has `vmulps` + `vaddps` and no `vfmadd`), so the per-block accumulation must be an
#   unfused mul and add as well.  `fmaf` there is a 1-ULP-per-block difference that leaves every hash and every
#   greedy text identical and only shows up in test-backend-ops' NMSE column -- exactly the class this delivery
#   records as a "plausible shift" (GREEDY-PURITY.md §109).  The r356 integration shipped the contributor's
#   `fmaf` first and the PR's own "every reported error is exactly 0" did not reproduce here: 25 of 61 MUL_MAT
#   cases read ~1e-14, and the split was diagnostic (every k=256 case 0, every k>256 case not).  This gate is the
#   instrument for that failure, so it cannot happen silently again.
#
#   The second class is structural: the block adds a fusor to block 10's dispatcher (`ggml_cuda_tq_try_fuse`) and
#   a node-reordering pass to graph_optimize, both of which run for EVERY graph.  A non-TQ model is the
#   default-path neutrality arm.
#
# WHAT IT CHECKS
#   HARD  1. the op oracle with NMSE: `test-backend-ops -o MUL_MAT -p type_a=tq` 61/61 and
#            `-o GET_ROWS -p type=tq` 16/16, and with LLAMA_TEST_OP_NMSE=1 **every** reported error is exactly 0
#            (305 MUL_MAT values, 84 GET_ROWS values here).  A failure is diagnosed: when every k=256 case is 0
#            and every k>256 case is nonzero, the accumulation has been fused again (`fmaf` in tq.cu).
#   HARD  2. the text: the default and `GGML_CUDA_TQ_FUSE=0` (the decode fusions and the graph_optimize reorder
#            off) must read ONE text on TQ2_0 and on TQ1_0, at a shallow and at a 200-token generation, and on the
#            multi-device split.  Those two arms have the SAME op placement, so a difference is a fusion that is
#            not the separate ops.  The `GGML_CUDA_TQ=0` (CPU-backend) arm is run and REPORTED, not gated: it is a
#            different placement, not a reference -- moving the TQ matmuls to the CPU drags the neighbouring ops
#            there too, so a hash comparison across the two is not an oracle (`GREEDY-PURITY.md` §38/§82), and
#            the bit-identity claim is arm 1's NMSE column (exactly 0).  Measured r356: gfx1201 agreed at 64 AND
#            200 tokens, gfx1151 agreed at 64 and flipped one token by 200 with the NMSE still 0/305 -- the
#            divergence is in a non-TQ op the placement moved, which is exactly why it cannot be the gate.
#            On the multi-device split the CPU arm is not run at all: `-ngl 99` still uploads the split weights to
#            the devices, so it is a per-op host round-trip of the weight slab rather than "the pre-block
#            behaviour", and on r355 -- where every TQ matmul took exactly that path -- it does not even reach a
#            token: `ggml-backend-meta.cpp:1460: GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]))` aborts
#            (measured, r356).  Gating it would gate a pre-existing pathology this block REMOVES.
#   HARD  3. the mechanism / non-vacuity (`GGML_CUDA_TQ_CENSUS=1`, one line per device at teardown):
#              default arm   : small>0, vec>0, vec_fused>0 (the decode fusions fired) and, on RDNA4, mma>0 (the
#                              int8-WMMA prefill kernel) -- a census of 0s means the block is not running;
#              `GGML_CUDA_TQ=0`: ALL FIVE counters 0 -- a kill switch that leaves the path live is not a switch.
#   HARD  4. default-path neutrality: a non-TQ model's greedy text is byte-identical to the recorded reference
#            for that device count (`--dense-ref`), so the fusor and the reorder changed nothing for the models
#            the delivery actually validates.  Throughput per arm is printed (INFO, not gated).
#   HARD  5. no abort, and `scripts/check-gpu-faults.sh` clean over this gate's own window.
#   INFO  * the TQ throughput against the `TQ=0` arm (llama-bench) is printed when `--bench` is given; the block's
#            claim is 8-37x, but a rate is not a correctness signal (`AGENTS.md`).
#
# NOT COVERED (deliberate, recorded in patches/README.md): MUL_MAT_ID (a TQ MoE keeps its experts on the CPU
#   backend, so there is nothing on the GPU to check), draft/MTP decoding with a TQ model, the CUDA (nvcc) build
#   of tq.cu -- only HIP compiles here -- and a CPU reference other than x86 AVX2 (the oracle IS the AVX2 path).
#
# USAGE:
#   ./gate-tq.sh [options]
#     --build DIR        the build to test (default $BUILD, else ~/llama.cpp/build-rocm)
#     --tq2 FILE/--tq1 FILE   the two TQ models (defaults under /llm/models/Llama3-8B-1.58/)
#     --dense FILE       the non-TQ neutrality model (default the Qwen3.5-4B Q8_0)
#     --dense-ref SHA    the recorded 1-GPU sha for the dense model (default: the value r356 recorded);
#                        pass an empty string to skip the neutrality comparison
#     --device LIST      1-GPU arm (default 0)
#     --split-device LIST  multi-device arm (default 0,1,2); empty/`--no-split` skips it
#     --prompt FILE      the 200-token prompt (default prompts/prose-rdna-boosts.txt)
#     --quick            ops + the TQ2_0 three-arm text at the shallow width only (a smoke, not a verdict)
#     --bench            also print llama-bench pp512/tg128 for the default and TQ=0 arms
#     --no-faults        skip the kernel-journal check (a box with no readable journal)
#     --self-test        point the NMSE diagnosis at a KNOWN-BAD binary and require it to FAIL (the
#                        negative control: `--build` a pre-r356 integration, or the fixed build with
#                        GGML_CUDA_TQ_FMA=... is not exposed -- use the original PR build if you have one)
#   Exit: 0 every arm passed, 1 a check failed, 2 usage / prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD:-$HOME/llama.cpp/build-rocm}"
TQ2_MODEL="${TQ2_MODEL:-/llm/models/Llama3-8B-1.58/TQ2_0.gguf}"
TQ1_MODEL="${TQ1_MODEL:-/llm/models/Llama3-8B-1.58/TQ1_0.gguf}"
DENSE_MODEL="${DENSE_MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
# The r356 record's 1-GPU, 20-token, seed-42 greedy sha for the dense model ("The capital of France is").
DENSE_REF="${DENSE_REF-1c5d32ac537d}"
DEVICE="${DEVICE:-0}"
SPLIT_DEVICE="${SPLIT_DEVICE:-0,1,2}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
QUICK=0
BENCH=0
FAULTS=1
SELF_TEST=0

while [ $# -gt 0 ]; do
    case "$1" in
        --build)         BUILD="$2"; shift 2 ;;
        --tq2)           TQ2_MODEL="$2"; shift 2 ;;
        --tq1)           TQ1_MODEL="$2"; shift 2 ;;
        --dense)         DENSE_MODEL="$2"; shift 2 ;;
        --dense-ref)     DENSE_REF="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --split-device)  SPLIT_DEVICE="$2"; shift 2 ;;
        --no-split)      SPLIT_DEVICE=""; shift ;;
        --prompt)        PROMPT="$2"; shift 2 ;;
        --quick)         QUICK=1; shift ;;
        --bench)         BENCH=1; shift ;;
        --no-faults)     FAULTS=0; shift ;;
        --self-test)     SELF_TEST=1; shift ;;
        -h|--help)       sed -n '2,66p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

LLAMA_CLI="$BUILD/bin/llama-cli"
LLAMA_BENCH="$BUILD/bin/llama-bench"
TEST_OPS="$BUILD/bin/test-backend-ops"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
GATE_SINCE_TS="${GATE_SINCE_TS:-$(date +%s)}"
mkdir -p "$KEEPOUT"

n_pass=0
n_fail=0
GATE_SINCE_TS="$(date +%s)"
pass() { echo "  PASS  $*"; n_pass=$((n_pass + 1)); }
fail() { echo "  FAIL  $*"; n_fail=$((n_fail + 1)); }
note() { echo "  info  $*"; }

for f in "$LLAMA_CLI" "$TEST_OPS"; do
    [ -x "$f" ] || { echo "ERROR: no $f -- build the tree first (see --build)" >&2; exit 2; }
done
for f in "$TQ2_MODEL" "$TQ1_MODEL"; do
    [ -f "$f" ] || { echo "ERROR: no TQ model at $f (a missing model is a prerequisite error, never a silent skip)" >&2; exit 2; }
done

echo "==> TQ1_0 / TQ2_0 gate (block 17, r356)"
echo "    build=$BUILD"
echo "    tq2=$TQ2_MODEL"
echo "    tq1=$TQ1_MODEL"
echo "    dense=$DENSE_MODEL (ref ${DENSE_REF:-none})"
echo "    devices=$DEVICE split=${SPLIT_DEVICE:-none} logs=$KEEPOUT"

# ---------------------------------------------------------------------------
# 0. self-test: the NMSE check must have teeth.  The original PR build (before the unfused accumulation) reads
#    nonzero on every k>256 case, so `--build <that>` must FAIL arm 1 here.
# ---------------------------------------------------------------------------
nmse_scan() {  # nmse_scan LOG OPNAME -> "<total> <nonzero> <k256_total> <k256_nonzero>"
    python3 - "$1" "$2" <<'PY'
import sys
log, op = sys.argv[1], sys.argv[2]
def err(l):
    for tok in l.split():
        if tok.startswith("err="):
            return float(tok[4:])
    return None
lines = [l for l in open(log, errors="replace") if "[NMSE]" in l and (" %s " % op) in l]
zs = [l for l in lines if err(l) == 0.0]
k  = [l for l in lines if "k=256," in l]
knz = [l for l in k if err(l) not in (None, 0.0)]
print(len(lines), len(lines) - len(zs), len(k), len(knz))
PY
}

ops_arm() {
    HIP_VISIBLE_DEVICES="$DEVICE" timeout 1800 "$TEST_OPS" -o MUL_MAT -p type_a=tq > "$KEEPOUT/ops-mulmat.log" 2>&1
    local rc=$?
    HIP_VISIBLE_DEVICES="$DEVICE" LLAMA_TEST_OP_NMSE=1 timeout 1800 "$TEST_OPS" -o MUL_MAT -p type_a=tq > "$KEEPOUT/ops-mulmat-nmse.log" 2>&1
    local rc2=$?
    HIP_VISIBLE_DEVICES="$DEVICE" LLAMA_TEST_OP_NMSE=1 timeout 1800 "$TEST_OPS" -o GET_ROWS -p type=tq > "$KEEPOUT/ops-getrows-nmse.log" 2>&1
    local rc3=$?
    local n_mm n_gr tot nz k_tot k_nz gtot gnz
    n_mm="$(grep -oE '[0-9]+/[0-9]+ tests passed' "$KEEPOUT/ops-mulmat.log" | tail -1)"
    n_gr="$(grep -oE '[0-9]+/[0-9]+ tests passed' "$KEEPOUT/ops-getrows-nmse.log" | tail -1)"
    if [ $rc -eq 0 ] && [ "$n_mm" = "61/61 tests passed" ]; then
        pass "op oracle: MUL_MAT $n_mm"
    else
        fail "op oracle: MUL_MAT ${n_mm:-<none>} (rc=$rc) -- see $KEEPOUT/ops-mulmat.log"
    fi
    if [ $rc3 -eq 0 ] && [ "$n_gr" = "16/16 tests passed" ]; then
        pass "op oracle: GET_ROWS $n_gr"
    else
        fail "op oracle: GET_ROWS ${n_gr:-<none>} (rc=$rc3) -- see $KEEPOUT/ops-getrows-nmse.log"
    fi
    read -r tot nz k_tot k_nz <<<"$(nmse_scan "$KEEPOUT/ops-mulmat-nmse.log" MUL_MAT)"
    read -r gtot gnz _ _ <<<"$(nmse_scan "$KEEPOUT/ops-getrows-nmse.log" GET_ROWS)"
    if [ "${nz:-1}" = 0 ] && [ "${gnz:-1}" = 0 ]; then
        pass "bit-identity to the CPU backend: all $tot MUL_MAT cases and all $gtot GET_ROWS cases read NMSE exactly 0"
    else
        fail "bit-identity: $nz/$tot MUL_MAT values read a nonzero NMSE ($k_nz/$k_tot at k=256) and $gnz/$gtot GET_ROWS"
        if [ "${nz:-0}" -gt 0 ] && [ "${k_nz:-1}" = 0 ]; then
            echo "        DIAGNOSIS: every single-block (k=256) case is exact and every multi-block case is not,"
            echo "        i.e. the per-Q8_K-block accumulation is fused again.  The CPU's"
            echo "        _mm256_add_ps(_mm256_mul_ps(..), sumf) is not contracted (vmulps+vaddps, no vfmadd), so"
            echo "        tq.cu must accumulate with TQ_FMUL/TQ_FADD, not fmaf (GREEDY-PURITY.md §109)."
        fi
    fi
}

if [ "$SELF_TEST" = 1 ]; then
    echo "==> self-test: a build WITHOUT the unfused accumulation must fail arm 1"
    ops_arm
    if [ "$n_fail" -gt 0 ]; then
        echo "==> PASS (self-test): the gate rejects this build ($n_fail failure(s)) -- $KEEPOUT"
        exit 0
    fi
    echo "==> FAIL (self-test): the gate accepted this build, so its NMSE check has no teeth (or the build"
    echo "    already carries the fix).  Point --build at a pre-fix binary."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. the op oracle + the bit-identity claim
# ---------------------------------------------------------------------------
echo "--> arm 1: op oracle + NMSE (bit-identity to the CPU's AVX2 path)"
ops_arm

# ---------------------------------------------------------------------------
# 2. the text: default / TQ_FUSE=0 / TQ=0 must read ONE text on both TQ models
# ---------------------------------------------------------------------------
text_run() {  # text_run TAG MODEL EXTRA_PROMPT_ARGS... ENV...
    local tag="$1" model="$2"; shift 2
    local log="$KEEPOUT/text-${tag//\//-}.log"
    : > "$log"
    HIP_VISIBLE_DEVICES="${TEXT_DEVICE:-$DEVICE}" timeout 1800 \
        env ${TEXT_ENV[@]+"${TEXT_ENV[@]}"} "$LLAMA_CLI" -m "$model" -ngl 99 ${TEXT_SPLIT:+-sm $TEXT_SPLIT} \
        "${TEXT_PROMPT_ARGS[@]}" --seed 42 --temp 0 --no-display-prompt --single-turn > "$log" 2>&1
    if [ $? -ne 0 ]; then echo "FAILED"; return; fi
    python3 "$REPO_DIR/scripts/extract-generated.py" "$log" --text 2>/dev/null > "${log%.log}.txt"
    python3 "$REPO_DIR/scripts/extract-generated.py" "$log" 2>/dev/null
}
sha_of() { echo "$1" | sed -n 's/.*sha=\([0-9a-f]*\).*/\1/p'; }

text_arm() {  # text_arm MODEL LABEL [no-cpu]  (runs the env arms on the current TEXT_* settings)
    local model="$1" label="$2" with_cpu="${3:-cpu}"
    local g f c
    TEXT_ENV=()                                   g="$(text_run "$label-G" "$model")"
    TEXT_ENV=(GGML_CUDA_TQ_FUSE=0)                 f="$(text_run "$label-F" "$model")"
    c="(not run)"
    if [ "$with_cpu" = cpu ]; then
        TEXT_ENV=(GGML_CUDA_TQ=0)                  c="$(text_run "$label-C" "$model")"
    fi
    if [ "$g" = FAILED ] || [ "$f" = FAILED ] || [ "$c" = FAILED ]; then
        fail "$label: an arm did not complete"; return
    fi
    if [ "$with_cpu" = cpu ]; then
        # HARD: default == fuse-off.  The two arms have the SAME op placement (only the fusion differs), so a
        # text difference is a fusion that is not the separate ops, which is the claim.
        local sg2 sf2
        sg2="$(sha_of "$g")"; sf2="$(sha_of "$f")"
        if [ -n "$sg2" ] && [ "$sg2" = "$sf2" ]; then
            pass "$label: the default and GGML_CUDA_TQ_FUSE=0 arms read one text (sha $sg2)"
        else
            fail "$label: default ($sg2) != GGML_CUDA_TQ_FUSE=0 ($sf2) -- a fused result is not the separate ops'"
        fi
        # INFO: the CPU-backend arm is a DIFFERENT PLACEMENT, not a reference: GGML_CUDA_TQ=0 drags the whole
        # attention/FFN block's neighbours onto the CPU backend too, so the two arms are not one configuration and
        # a hash comparison across them is not an oracle (GREEDY-PURITY.md §38/§82).  The bit-identity claim is
        # arm 1's NMSE column, which reads exactly 0; this arm is reported for the record.  Measured r356: gfx1201
        # agreed at 64 and 200 tokens, gfx1151 agreed at 64 and flipped one token by 200 (with NMSE still 0/305
        # there) -- which is what a non-oracle looks like.
        note "$label: CPU-backend arm (different placement, INFO not an oracle): ${c:-<none>}"
        if [ -s "$KEEPOUT/text-${label//\//-}-G.txt" ] && [ -s "$KEEPOUT/text-${label//\//-}-C.txt" ]; then
            note "$label: shared prefix with the CPU arm: $(python3 - "$KEEPOUT/text-${label//\//-}-G.txt" "$KEEPOUT/text-${label//\//-}-C.txt" <<'PY'
import sys
a = open(sys.argv[1], errors="replace").read(); b = open(sys.argv[2], errors="replace").read()
n = 0
for x, y in zip(a, b):
    if x != y: break
    n += 1
print("%d of %d/%d chars" % (n, len(a), len(b)))
PY
)"
        fi
        return
    fi
    if [ "$with_cpu" != cpu ]; then
        # A MULTI-DEVICE split is not a CPU-backend configuration: `-ngl 99` still uploads the split weights to
        # the devices, so GGML_CUDA_TQ=0 turns every matmul into a per-op host round-trip of the weight slab --
        # and on r355, where every TQ matmul took that path, the Meta backend aborts at
        # `ggml-backend-meta.cpp:1460: GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]))` before the first
        # token.  The CPU oracle is therefore a 1-GPU arm (arm 2 above); here the fusion pair is the claim.
        local sg sf
        sg="$(sha_of "$g")"; sf="$(sha_of "$f")"
        if [ -n "$sg" ] && [ "$sg" = "$sf" ]; then
            pass "$label: default == TQ_FUSE=0 (sha $sg; the CPU arm is a 1-GPU arm, see the gate header)"
        else
            fail "$label: the fusion arms disagree (default $sg, fuse-off $sf)"
        fi
        return
    fi
    note "$label: G=$g"
    note "$label: F=$f"
    note "$label: C=$c"
    local sg sf sc
    sg="$(sha_of "$g")"; sf="$(sha_of "$f")"; sc="$(sha_of "$c")"
    if [ -n "$sg" ] && [ "$sg" = "$sf" ] && [ "$sg" = "$sc" ]; then
        pass "$label: default == TQ_FUSE=0 == TQ=0 (sha $sg)"
    else
        fail "$label: the arms disagree (default $sg, fuse-off $sf, cpu $sc) -- the GPU result is not the CPU's"
    fi
}

SHORT=(-p "The capital of France is" -n 64)
LONG=(-f "$PROMPT" -n 200)
TEXT_PROMPT_ARGS=("${SHORT[@]}")
TEXT_DEVICE="$DEVICE"
TEXT_SPLIT=""

echo "--> arm 2: the text, three arms per model"
text_arm "$TQ2_MODEL" "tq2_0/short/1gpu"
[ "$QUICK" = 1 ] || text_arm "$TQ1_MODEL" "tq1_0/short/1gpu"
if [ "$QUICK" = 0 ]; then
    [ -f "$PROMPT" ] || { echo "ERROR: no prompt at $PROMPT" >&2; exit 2; }
    TEXT_PROMPT_ARGS=("${LONG[@]}")
    text_arm "$TQ2_MODEL" "tq2_0/long200/1gpu"
    text_arm "$TQ1_MODEL" "tq1_0/long200/1gpu"
    if [ -n "$SPLIT_DEVICE" ]; then
        TEXT_DEVICE="$SPLIT_DEVICE"
        TEXT_PROMPT_ARGS=("${LONG[@]}")
        for sm in layer tensor; do
            TEXT_SPLIT="$sm"
            text_arm "$TQ1_MODEL" "tq1_0/long200/$sm" no-cpu
        done
        TEXT_SPLIT=""
        TEXT_DEVICE="$DEVICE"
    fi
else
    TEXT_PROMPT_ARGS=("${SHORT[@]}")
fi

# ---------------------------------------------------------------------------
# 3. the mechanism: census on the default arm, all-zero on the kill switch
# ---------------------------------------------------------------------------
census_run() {  # census_run TAG MODEL  -> one SUMMED census line per device
    local tag="$1" model="$2"
    local log="$KEEPOUT/census-$tag.log"
    # the prompt file is ~350 tokens, i.e. a PREFILL wider than TQ_SMALL_MAX (16), which is what makes the
    # RDNA4 int8-WMMA kernel reachable at all: a 6-token prompt is decode-only and reads mma=0 vacuously.
    HIP_VISIBLE_DEVICES="${CENSUS_DEVICE:-$DEVICE}" timeout 1800 \
        env ${CENSUS_ENV[@]+"${CENSUS_ENV[@]}"} GGML_CUDA_TQ_CENSUS=1 "$LLAMA_CLI" -m "$model" -ngl 99 \
        -f "$PROMPT" -n 8 --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 > "$log" 2>&1
    if [ $? -ne 0 ]; then echo "FAILED"; return; fi
    # SUMMED over every line: llama-cli can create and destroy more than one CUDA context, and the first line is
    # not always the one that did the work (measured on gfx1151: two lines, the first all-zero).
    python3 - "$log" <<'PY'
import re, sys
tot = {}
for l in open(sys.argv[1], errors="replace"):
    m = re.search(r"TQ_CENSUS dev=(\d+) mul_mat small=(\d+) mma=(\d+) get_rows=(\d+) vec=(\d+) vec_fused=(\d+)", l)
    if not m:
        continue
    dev = int(m.group(1))
    v = [int(x) for x in m.groups()[1:]]
    if dev not in tot:
        tot[dev] = v
    else:
        tot[dev] = [a + b for a, b in zip(tot[dev], v)]
for dev in sorted(tot):
    s, mma, gr, vec, vf = tot[dev]
    print("TQ_CENSUS dev=%d mul_mat small=%d mma=%d get_rows=%d vec=%d vec_fused=%d" % (dev, s, mma, gr, vec, vf))
PY
}

echo "--> arm 3: the census (the block ran; the kill switch is not vacuous)"
# RDNA4 (gfx12xx) is the only family with the int8-WMMA prefill kernel; ask the build which device it sees.
RDNA4=0
if HIP_VISIBLE_DEVICES="$DEVICE" timeout 120 "$LLAMA_CLI" --list-devices 2>/dev/null | grep -qi 'gfx12'; then
    RDNA4=1
fi
note "RDNA4 device selected: $RDNA4 (gates the mma counter)"
CENSUS_ENV=()
CENSUS_DEVICE="$DEVICE"
on="$(census_run on-tq2 "$TQ2_MODEL")"
CENSUS_ENV=(GGML_CUDA_TQ_FUSE=0)
off="$(census_run off-tq2 "$TQ2_MODEL")"
CENSUS_ENV=(GGML_CUDA_TQ=0)
off_cpu="$(census_run cpu-tq2 "$TQ2_MODEL")"
CENSUS_ENV=()
if [ "$on" = FAILED ] || [ -z "$on" ]; then
    fail "census: the default arm printed no TQ_CENSUS line (is GGML_CUDA_TQ_CENSUS wired?)"
else
    note "default: $(echo "$on" | head -1)"
    sm="$(echo "$on" | head -1 | sed -n 's/.*small=\([0-9]*\).*/\1/p')"
    mm="$(echo "$on" | head -1 | sed -n 's/.*mma=\([0-9]*\).*/\1/p')"
    vc="$(echo "$on" | head -1 | sed -n 's/.*vec=\([0-9]*\).*/\1/p')"
    vf="$(echo "$on" | head -1 | sed -n 's/.*vec_fused=\([0-9]*\).*/\1/p')"
    if [ "${sm:-0}" -gt 0 ] && [ "${vc:-0}" -gt 0 ]; then
        pass "census: the TQ path ran (small=$sm vec=$vc)"
    else
        fail "census: small=$sm vec=$vc -- the block did not run on a TQ model"
    fi
    if [ "${vf:-0}" -gt 0 ]; then
        pass "census: the decode fusions fired (vec_fused=$vf)"
    else
        fail "census: vec_fused=0 -- the default arm's fusion is vacuous (GGML_CUDA_TQ_FUSE is on by default)"
    fi
    if [ "${RDNA4:-0}" = 1 ]; then
        if [ "${mm:-0}" -gt 0 ]; then
            pass "census: the RDNA4 int8-WMMA prefill kernel ran (mma=$mm)"
        else
            fail "census: mma=0 on an RDNA4 device -- the int8-WMMA prefill kernel was never reached (is the prefill > 16 tokens?)"
        fi
    else
        note "mma=$mm: not RDNA4, so the WMMA kernel is absent by design and the small-batch kernel takes every batch"
    fi
    if [ -n "$SPLIT_DEVICE" ] && [ "$QUICK" = 0 ]; then
        CENSUS_DEVICE="$SPLIT_DEVICE"
        multi="$(census_run multi-tq1 "$TQ1_MODEL")"
        CENSUS_DEVICE="$DEVICE"
        if [ "$multi" = FAILED ] || [ "$(echo "$multi" | wc -l)" -lt 2 ]; then
            fail "census: the multi-device arm did not print one line per device"
        else
            pass "census: one line per device on the split arm ($(echo "$multi" | wc -l) devices)"
        fi
    fi
fi
if [ "$off" = FAILED ] || [ -z "$off" ]; then
    fail "census: no line on the TQ_FUSE=0 arm"
else
    note "fuse-off: $(echo "$off" | head -1)"
    case "$off" in
        *vec_fused=0*) pass "census: GGML_CUDA_TQ_FUSE=0 takes the fusions away (vec_fused=0)" ;;
        *) fail "census: the fuse-off arm still reports fused launches -- the switch does nothing" ;;
    esac
fi
if [ "$off_cpu" = FAILED ] || [ -z "$off_cpu" ]; then
    fail "census: no line on the TQ=0 arm"
else
    note "cpu: $(echo "$off_cpu" | head -1)"
    if echo "$off_cpu" | grep -qE 'small=0 mma=0 get_rows=0 vec=0 vec_fused=0'; then
        pass "census: GGML_CUDA_TQ=0 leaves every TQ counter at 0 (the types are refused again)"
    else
        fail "census: the TQ=0 arm still launched a TQ kernel -- the kill switch is vacuous"
    fi
fi

# ---------------------------------------------------------------------------
# 4. default-path neutrality: a non-TQ model's text is the recorded one
# ---------------------------------------------------------------------------
echo "--> arm 4: default-path neutrality (a non-TQ model)"
if [ ! -f "$DENSE_MODEL" ]; then
    fail "neutrality: no dense model at $DENSE_MODEL"
elif [ -z "$DENSE_REF" ]; then
    note "neutrality: skipped (--dense-ref '')"
else
    TEXT_DEVICE="$DEVICE"; TEXT_SPLIT=""; TEXT_ENV=()
    TEXT_PROMPT_ARGS=(-p "The capital of France is" -n 20)
    d="$(text_run dense "$DENSE_MODEL")"
    if [ "$d" = FAILED ]; then
        fail "neutrality: the dense arm did not complete"
    else
        note "dense: $d"
        if [ "$(sha_of "$d")" = "$DENSE_REF" ]; then
            pass "neutrality: the dense model still reads $DENSE_REF (the fusor and the reorder are inert for it)"
        else
            fail "neutrality: the dense model reads $(sha_of "$d"), not the recorded $DENSE_REF -- the block moved a default path"
        fi
    fi
    if [ -n "$SPLIT_DEVICE" ]; then
        # A greedy text is a PER-DEVICE-COUNT reference (AGENTS.md, "Verify"): the split arm has no 1-GPU
        # oracle, so this arm only proves the default path still runs on a split without an abort, and its
        # text is recorded for the release entry.
        TEXT_DEVICE="$SPLIT_DEVICE"; TEXT_SPLIT="tensor"
        d="$(text_run dense-tensor "$DENSE_MODEL")"
        TEXT_SPLIT=""; TEXT_DEVICE="$DEVICE"
        if [ "$d" = FAILED ]; then
            fail "neutrality: the 3-GPU tensor arm did not complete (the default path must still run on a split)"
        else
            note "dense/-sm tensor (INFO, per-device-count reference): $d"
            pass "neutrality: the default path runs on a 3-GPU tensor split"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 5. INFO: the rate (never a correctness signal)
# ---------------------------------------------------------------------------
if [ "$BENCH" = 1 ] && [ -x "$LLAMA_BENCH" ]; then
    echo "--> info: llama-bench -fa 1 -p 512 -n 128 -r 3 (default vs TQ=0)"
    for arm in on off; do
        if [ "$arm" = off ]; then e=(env GGML_CUDA_TQ=0); else e=(); fi
        HIP_VISIBLE_DEVICES="$DEVICE" timeout 1800 "${e[@]}" "$LLAMA_BENCH" -m "$TQ2_MODEL" -ngl 99 -fa 1 \
            -p 512 -n 128 -r 3 > "$KEEPOUT/bench-$arm.log" 2>&1
        if [ $? -eq 0 ]; then
            note "bench $arm: $(grep -E 'pp512|tg128' "$KEEPOUT/bench-$arm.log" | awk -F'|' '{gsub(/ /,"",$9); printf "%s %s; ", $8, $9}')"
        else
            note "bench $arm: FAILED (see $KEEPOUT/bench-$arm.log)"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 6. the kernel journal
# ---------------------------------------------------------------------------
if [ "$FAULTS" = 1 ]; then
    logs=()
    while IFS= read -r l; do logs+=(--log "$l"); done < <(find "$KEEPOUT" -maxdepth 1 -name '*.log' | sort)
    if bash "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$GATE_SINCE_TS" "${logs[@]}" > "$KEEPOUT/faults.log" 2>&1; then
        pass "kernel journal and every run log clean"
    else
        fail "GPU faults found ($KEEPOUT/faults.log)"
    fi
else
    note "fault check skipped (--no-faults)"
fi

echo
echo "==> TQ gate: $n_pass passed, $n_fail failed (logs in $KEEPOUT)"
if [ "$n_fail" -ne 0 ]; then
    echo "==> FAIL"
    exit 1
fi
if [ "$QUICK" = 1 ]; then
    echo "==> PASS (QUICK: the 200-token, multi-device and split arms did not run)"
else
    echo "==> PASS"
fi
exit 0
