#!/usr/bin/env bash
# The shared-expert down epilogue's ROUNDING gate -- block 08 (`shexp_down_gated_q8_0`, issue #377).
#
# WHY THIS EXISTS
#   `shexp_down_gated_q8_0` stands in for three graph nodes of the MoE shared-expert output chain:
#       gated   = down * gate_sig      (a MUL kernel)
#       ffn_out = gated + moe_out      (an ADD kernel)
#       dst     = ffn_out + ffn_residual (an ADD kernel)
#   and its epilogue computed `__fmul_rn(sum_down, gate_sig) + moe_out + ffn_residual`.  HIP clang contracts that
#   into `v_fma_f32` / `v_fmac_f32` -- `__fmul_rn` is NOT a barrier, it lowers to a plain op (GREEDY-PURITY.md
#   §102.2) -- so the epilogue rounded ONCE where the chain rounds THREE times, and the comment above it claimed
#   the opposite.  Every instantiation contracted the same way, so decode == verify was never at risk: this is a
#   "the code says X, the ISA says Y" defect, and it is invisible to every instrument the repo usually runs.
#   Measured on gfx1201 / ROCm 10.2, Qwen3.6-35B-A3B Q8_0 (the fused epilogue fires 717x per 4-token run):
#     * the delivery's canonical prefill-logit point reads **0.000000 KLD / 100.000 %** between the two
#       arithmetics -- because the fusion only arms at <= MMVQ_MAX_BATCH_SIZE tokens, so a batched prefill
#       (`--ub 512`) runs the epilogue only on its small tail ubatch;
#     * same-seed greedy text is byte-identical (2f79e37ca208 / a2cdef23588f, 1-GPU host-expert and 3-GPU
#       `-sm tensor` arms);
#     * at `-ub 1` -- where EVERY token runs the fused epilogue and 512 sequential steps amplify a 1e-7 shift --
#       the same pair reads mean KLD 0.001354 / same-top-p 98.431 %, i.e. the change is real but rounding-level.
#   So the only instrument that decides this class is the EMITTED ISA, plus a census that says the kernel ran at
#   all (GREEDY-PURITY.md §102.5).  That is what this gate is.
#
# WHAT IT CHECKS
#   HARD  1. ISA (the mechanism).  Unbundle the gfx code object out of the built `mmvq.cu.o`, disassemble it, and
#            for EVERY `shexp_down_gated_q8_0<nwarps, ncols>` instantiation: the epilogue must carry no
#            `v_fma_f32` / `v_fmac_f32` / `v_fma_mix_f32`, and at least `ncols` `v_mul_f32` (a separate product
#            per token).  Non-vacuity of the scan itself: (a) all 32 instantiations must be found, so a mangled
#            name or a missing arch is a FAIL and not an empty pass; (b) the SAME regex must find fma/fmac
#            elsewhere in the same function (the `vec_dot_q8_0_q8_1` accumulation uses them), so "none in the
#            epilogue" is a measured absence and not a broken pattern.  The hyper-connection kernels
#            (`mul_mat_id_{q8_0,iq4_nl}_weighted_rdna3_5`) are counted too: they share the helper, and they
#            must keep theirs.
#   HARD  2. the census / non-vacuity (`GGML_CUDA_SHEXP_GATE_DEBUG=1`, one line per fused launch):
#              default arm : `SHEXP DOWN GATE <out> ncols=N exact=1` printed at least once -- a MoE model whose
#                            `ffn_down_shexp` is Q8_0 takes this fusion at every decode step;
#              kill switch : `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` prints NOTHING (a switch that leaves the
#                            kernel live is not a switch).
#   HARD  3. no abort, and the two arms' same-seed greedy text is REPORTED and must be reproducible: each arm is
#            run twice and must read ONE text.  The fused and the unfused arm are NOT expected to agree -- the
#            fused gate reduction (`shexp_gate_sigmoid`) does not reproduce the standalone MUL_MAT's order
#            (`ggml-cuda-try-fuse.cu` says so), so the two arms are different placements, not two arithmetics of
#            one (`GREEDY-PURITY.md` §38).  The purity claim is arm 1.
#   HARD  4. `scripts/check-gpu-faults.sh` clean over this gate's own window.
#
# NOT COVERED (deliberate): the CUDA (nvcc) build -- this box has no NVIDIA device, and nvcc never contracts
#   `__fmul_rn`, so the non-AMD path keeps the arithmetic it always had (it is not what this gate is about); a
#   non-gfx1201 target (the epilogue is the same source and the same helper; the gate takes `--arch`); and the
#   FP64/`-ffp-contract=fast` compiler variant a downstream distro might use.
#
# USAGE
#   ./gate-shexp-down-gate.sh [options]
#     --build DIR     the build to test (default ~/llama.cpp/build-rocm)
#     --model FILE    a MoE gguf whose ffn_down_shexp is Q8_0 (default Qwen3.6-35B-A3B Q8_0)
#     --gpu LIST      HIP_VISIBLE_DEVICES for the census arms (default 0,1,2 -- the whole model fits)
#     --ncmoe N       pass `-ncmoe N` (+ `--host-experts pool`) for a single-GPU arm (default: none)
#     --arch ARCH     the code-object arch to inspect (default gfx1201)
#     --ngen N        generated tokens per text arm (default 64)
#     --keep          keep the log directory (default: kept only on failure)
#     --no-faults     skip the kernel-journal fault scan (a host with no readable journal)
#
# Exit: 0 PASS, 1 FAIL, 2 usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf}"
GPU="${GPU:-0,1,2}"
NCMOE="${NCMOE:-}"
ARCH="${ARCH:-gfx1201}"
NGEN="${NGEN:-64}"
KEEP=0
FAULTS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --build)     BUILD_DIR="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --gpu)       GPU="$2"; shift 2 ;;
        --ncmoe)     NCMOE="$2"; shift 2 ;;
        --arch)      ARCH="$2"; shift 2 ;;
        --ngen)      NGEN="$2"; shift 2 ;;
        --keep)      KEEP=1; shift ;;
        --no-faults) FAULTS=0; shift ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--build DIR] [--model FILE] [--gpu LIST] [--ncmoe N] [--arch A] [--ngen N] [--keep] [--no-faults]" >&2; exit 2 ;;
    esac
done

CLI="$BUILD_DIR/bin/llama-cli"
OBJ="$BUILD_DIR/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o"
CACHE="$BUILD_DIR/CMakeCache.txt"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

echo "==> shared-expert down epilogue gate (shexp_down_gated_q8_0, block 08, issue #377)"
echo "    build=$BUILD_DIR"
echo "    model=$MODEL"
echo "    gpu=$GPU${NCMOE:+ (ncmoe $NCMOE)}  arch=$ARCH"
echo "    logs=$KEEPOUT"
mkdir -p "$KEEPOUT"
[ -x "$CLI" ]  || { echo "FAIL: llama-cli not executable: $CLI" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "FAIL: model not found: $MODEL" >&2; exit 2; }
[ -f "$OBJ" ]  || { echo "FAIL: object not found: $OBJ (build ggml-hip first)" >&2; exit 2; }

# --------------------------------------------------------------------------------------- 1. the ISA
# The toolchain is the build's own (the CMake cache pins it), so the disassembly is of the object this build
# actually produced.  No tool, no verdict: a mechanism gate that skips its mechanism is worse than no gate.
echo
echo "==> 1. the emitted ISA of every shexp_down_gated_q8_0 instantiation"
HIPC="$(sed -n 's/^CMAKE_HIP_COMPILER:FILEPATH=//p' "$CACHE" 2>/dev/null | head -1)"
TOOLDIR="${HIPC:+$(dirname "$HIPC")}"
OBJCOPY="$TOOLDIR/llvm-objcopy"; BUNDLER="$TOOLDIR/clang-offload-bundler"; OBJDUMP="$TOOLDIR/llvm-objdump"
if [ -z "${TOOLDIR:-}" ] || [ ! -x "$OBJDUMP" ] || [ ! -x "$BUNDLER" ] || [ ! -x "$OBJCOPY" ]; then
    fail "the build's LLVM tools were not found next to CMAKE_HIP_COMPILER ('${HIPC:-<unset>}'): cannot read the ISA"
else
    note "tools: $TOOLDIR"
    ( cd "$KEEPOUT" &&
      "$OBJCOPY" --dump-section=.hip_fatbin=fat.bin "$OBJ" fat.o &&
      "$BUNDLER" --unbundle --type=o --targets="hip-amdgcn-amd-amdhsa--$ARCH" --input=fat.bin --output=dev.o &&
      "$OBJDUMP" -d dev.o > dev.s ) > "$KEEPOUT/isa.log" 2>&1
    if [ ! -s "$KEEPOUT/dev.s" ]; then
        fail "could not disassemble $OBJ for $ARCH (log: $KEEPOUT/isa.log)"
    else
        python3 - "$KEEPOUT/dev.s" <<'PY'
import re, sys
lines = open(sys.argv[1]).read().splitlines()
cur, blocks = None, {}
for line in lines:
    m = re.match(r'^[0-9a-f]{16} <(_ZL21shexp_down_gated_q8_0[^>]*)>:', line)
    if m:
        cur = m.group(1); blocks[cur] = []; continue
    if cur is not None:
        blocks[cur].append(line)
hc = [l for l in lines if re.search(r'<[^>]*mul_mat_id_(q8_0|iq4_nl)_weighted_rdna3_5', l)]
kern = re.compile(r'v_fma_f32|v_fmac|v_fma_mix')          # the contracted forms
# The epilogue is a loop over tokens, so each output value is a chain that ENDS IN A STORE.  Anchor on the store:
# look back for the token's separate product (v_mul_f32) and require that nothing between it and the store is a
# contracted form.  A window anchored on the last `v_add_f32_dpp` cannot do this job -- the compiler interleaves
# the eight-token reduction with the dot loop and software-pipelines the stores, so a window that is right for
# <1,1>..<4,8> is wrong for <8,8> (measured on both the r370 and the r371 object).
LOOKBACK = 30
bad, ok, noepi, short = [], 0, [], []
for name, body in sorted(blocks.items()):
    dims = re.search(r'ILi(\d+)ELi(\d+)E', name)
    ncols = int(dims.group(2)) if dims else 0
    # non-vacuity of the pattern itself: the dot-product accumulation in the same function DOES use fma forms,
    # so "no contracted form in the epilogue" below is a measured absence and not a broken regex.
    if not kern.search(' '.join(body)):
        noepi.append(name + " (no contracted form anywhere in the kernel -- the scan pattern is broken)")
        continue
    stores = 0
    for i, l in enumerate(body):
        if 'global_store' not in l:
            continue
        stores += 1
        mul = next((j for j in range(i - 1, max(0, i - LOOKBACK) - 1, -1) if 'v_mul_f32' in body[j]), None)
        if mul is None:
            bad.append("%s: store@%d has no v_mul_f32 in the %d instructions before it (the product was contracted)"
                       % (name, i, LOOKBACK))
        elif kern.search(' '.join(body[mul:i])):
            bad.append("%s: store@%d has a contracted form between its v_mul_f32@%d and the store" % (name, i, mul))
        else:
            ok += 1
    if stores < ncols:
        short.append("%s: %d store(s) for %d token(s)" % (name, stores, ncols))
print("    instantiations found      : %d (expect 32 = 4 nwarps x 8 ncols)" % len(blocks))
print("    epilogue chains verified  : %d (uncontracted, each ending in a store)" % ok)
print("    hyper-connection kernels  : %d" % len(hc))
failed = False
if len(blocks) != 32:
    print("    FAIL: expected 32 instantiations, found %d" % len(blocks)); failed = True
if noepi:
    print("    FAIL: the scan did not reach the kernel (%d):" % len(noepi))
    for n in noepi:
        print("          %s" % n)
    failed = True
if bad:
    print("    FAIL: %d epilogue chain(s) are contracted or unanchored:" % len(bad))
    for n in bad[:12]:
        print("          %s" % n)
    failed = True
if short:
    print("    FAIL: %d instantiation(s) store fewer values than their token count:" % len(short))
    for n in short:
        print("          %s" % n)
    failed = True
if not hc:
    print("    FAIL: the hyper-connection kernels were not found -- the object is not the one this gate is about")
    failed = True
sys.exit(1 if failed else 0)
PY
        [ $? -eq 0 ] || fails=$((fails + 1))
    fi
fi

# ------------------------------------------------------------------------------------- 2. the census
echo
echo "==> 2. the census: the fused launch happens, and the kill switch takes it away"
run_cli() { # <tag> <env...> -- <cli args...>
    local tag="$1"; shift
    local envs=() args=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift || true
    env "${envs[@]}" HIP_VISIBLE_DEVICES="$GPU" timeout "$TIME_LIMIT" "$CLI" "$@" > "$KEEPOUT/$tag.log" 2>&1
}
# The band arm: `-sm tensor` over the pinned list.  A single-device arm needs -ncmoe (the model is larger than one
# card), and then the split mode is the CLI's own `layer`.
split_args=(-sm tensor)
[ -n "$NCMOE" ] && split_args=(-sm layer -ncmoe "$NCMOE" --host-experts pool)
base=(-m "$MODEL" -ngl 99 "${split_args[@]}" -fa 1 -c 4096 -p "The capital of France is" --seed 42 --temp 0
      --no-display-prompt --single-turn -t 8)
# A text/hash arm takes NO -lv (the same stream carries the generated text); a census arm takes -lv 4.
census_args=("${base[@]}" -n 4 -lv 4)
text_args=("${base[@]}" -n "$NGEN")
run_cli census_on  GGML_CUDA_SHEXP_GATE_DEBUG=1 -- "${census_args[@]}"
run_cli census_off GGML_CUDA_SHEXP_GATE_DEBUG=1 GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1 -- "${census_args[@]}"
on="$(grep -c 'SHEXP DOWN GATE' "$KEEPOUT/census_on.log" 2>/dev/null || true)"
off="$(grep -c 'SHEXP DOWN GATE' "$KEEPOUT/census_off.log" 2>/dev/null || true)"
note "default: ${on:-0} fused launch(es); GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1: ${off:-0}"
grep -m2 'SHEXP DOWN GATE' "$KEEPOUT/census_on.log" 2>/dev/null | sed 's/^/    first: /' || true
if [ "${on:-0}" -lt 1 ]; then
    fail "the fused epilogue never ran on $MODEL -- either ffn_down_shexp is not Q8_0 on this file (then point --model at one that is) or the fusion is not arming (log: $KEEPOUT/census_on.log)"
fi
if [ "${on:-0}" -gt 0 ] && [ "${off:-0}" -gt 0 ]; then
    fail "GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1 still fused ${off} launch(es): the kill switch does not reach the matcher"
fi
if grep -q 'exact=0' "$KEEPOUT/census_on.log" 2>/dev/null; then
    fail "the census reports exact=0 -- the epilogue is not the exact-rounding form this gate is about"
fi

# --------------------------------------------------------------------------------------- 3. the text
echo
echo "==> 3. same-seed greedy text, each arm twice (the arms are different placements; the claim is arm 1)"
extract() { python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null | sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\2/p' | head -1; }
for arm in fused chain; do
    envs=(); [ "$arm" = chain ] && envs=(GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1)
    for rep in 1 2; do
        run_cli "text_${arm}_$rep" ${envs[@]+"${envs[@]}"} -- "${text_args[@]}"
    done
    h1="$(extract "$KEEPOUT/text_${arm}_1.log")"; h2="$(extract "$KEEPOUT/text_${arm}_2.log")"
    note "$arm: ${h1:-<none>} / ${h2:-<none>}"
    if [ -z "${h1:-}" ] || [ "$h1" != "$h2" ]; then
        fail "the $arm arm's text is not reproducible (${h1:-none} vs ${h2:-none}) -- the arm is not an oracle"
    fi
done

# ------------------------------------------------------------------------------------- the faults
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (since ${GATE_SINCE_TS:-the journal start})"
    since_args=(); [ -n "${GATE_SINCE_TS:-}" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=(); for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" | sed 's/^/    /' ||
        fail "a GPU fault was recorded during this gate's arms"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: every shexp_down_gated_q8_0 instantiation rounds its epilogue on its own, the fused launch runs,"
    echo "      and the kill switch removes it"
    [ "$KEEP" -eq 1 ] || rm -rf "$KEEPOUT"
    exit 0
fi
echo "FAILED: $fails check(s) -- see $KEEPOUT"
exit 1
