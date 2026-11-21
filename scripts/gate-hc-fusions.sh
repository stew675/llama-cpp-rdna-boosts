#!/usr/bin/env bash
# Hyper-connection fusion-equivalence gate for qwen4exp / Flash-Next (issue #228, r148).
#
# WHY THIS EXISTS
#   The qwen4exp residual stream is carried by `hc` parallel streams and steered by a
#   per-stream SCATTER LOGIT: build_hc_mix computes it (the hc_inject LoRA product) and
#   build_hc_combine turns it into `w = 2*sigmoid(logit)` for the residual add.  Both halves
#   exist twice -- as a plain ggml chain (upstream's model definition) and as a fused op
#   (GGML_OP_HC_MIX / GGML_OP_HC_COMBINE) -- and each half can be selected independently:
#
#       mix = fused  iff cparams.fused_hc_mix     (LLAMA_FUSED_HC_MIX)
#       comb = fused iff cparams.fused_hc_combine (LLAMA_FUSED_HC_COMBINE)
#
#   through r147 the fused MIXER published the RAW dot product as its inject tail while the
#   fused COMBINE divided that tail by hc again, so the two halves were only consistent with
#   EACH OTHER.  The model's own chain scales the logit by 1/hc exactly once (build_hc_mix's
#   SCALE), which made two of the four combinations wrong:
#
#     fused mix + fused comb (Q8_0/BF16 hc weights)  correct  -- the combination every
#                                                              dense/GSQ validation ran
#     chain mix + fused comb (any other hc quant)    /hc TWICE -> the model degenerates
#     fused mix + chain comb                          /hc MISSING -> the model degenerates
#     chain mix + chain comb                          correct  -- the reference chain
#
#   The first real-GGUF victim was a Q5_K_M Flash-Next requant (issue #228: hc_attn_down is
#   Q5_K, so the mixer takes the chain while the combine stayed fused): greedy repetition
#   collapse and a kl ~15 logit drift vs v14.  The second was Flash-Next IQ4_NL -- the model
#   most of the delivery's own qwen4exp measurements used.  `LLAMA_FUSED_HC_COMBINE=0` (the
#   documented A/B kill switch) was the only rescue, and on a BF16-hc model that same switch
#   produced an EMPTY generation.
#
#   test-backend-ops cannot see this class: HC_MIX and HC_COMBINE are checked against a CPU
#   reference that encoded the same (self-consistent, chain-inconsistent) convention, so the
#   pair passed while the fused arms disagreed with the chain.  This gate is the oracle that
#   compares the ARMS against each other on a real GGUF.
#
# ONE SWITCH AT A TIME (r309, `TODO` #341).  Until r309 the arms did NOT isolate the switch they
# named: they left the two DSV4 HC switches at their defaults, and those defaults are not benign.
#   * `LLAMA_FUSED_DSV4_HC_PRE`/`_POST` select GGML_OP_DSV4_HC_PRE / _POST, which own the HC16 bf16
#     producer slots -- a DOCUMENTED numeric re-baseline, not an alias (see NUMERIC below).
#   * `fused_dsv4_hc_post` DEFAULTS TO `meta_split_active` (src/llama-context.cpp), i.e. ON whenever a
#     tensor split puts a Meta device in the graph, and the DSV4 ops are MUTUALLY EXCLUSIVE with the
#     mix/combine ops: build_hc_mix/build_hc_combine take the DSV4 branch only on their CHAIN path.
#   So on a `-sm tensor` run the old `chain_chain` arm was not the chain at all -- it was "chain mixer +
#   DSV4_HC_POST", and every `LLAMA_FUSED_HC_COMBINE=1` arm bypassed that op and ran GGML_OP_HC_COMBINE
#   instead.  The gate then compared TWO numeric fusions and reported the difference as a combine
#   disagreement: Flash-Next IQ4_NL read 446 `a5d0ec7b809d` (every combine-op arm) vs 459 `9f8819c4c2c5`
#   (`chain_chain`) at r303, and the same pair on the r302 build, which is why it looked pre-existing.
#   It was: the DSV4_HC_POST default became split-dependent in the r242 era, and the gate kept a
#   reference arm that silently changed meaning.
#   Fix: every arm below PINS ALL FOUR switches (the table is checked at startup), the reference arm is
#   the true chain (all four off), and the states that differ by design are NUMERIC arms.
#   Measured 2026-11-17 (r308 build) on IQ4_NL, 3 devices `-sm tensor`: true chain 450 `bc46095cf4c6`;
#   mix_op / comb_op / both_ops / cuda_fuse_off all 450, i.e. on THAT class both ops and the CUDA
#   matchers ARE byte-identical to the chain; the DSV4 pre op 444 `7ea23f309705`, the post op 454
#   `9b4b033b33f7`, all four on 446 `a5d0ec7b809d`.  So `TODO` #341 is a gate defect, not a numerics
#   defect -- and `446`/`a5d0ec7b809d` is what the shipping default of that class reads.
#
# WHAT IT CHECKS
#   HARD: the reference arm is the TRUE unfused chain (`LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0
#         LLAMA_FUSED_DSV4_HC_PRE=0 LLAMA_FUSED_DSV4_HC_POST=0`), it must not be degenerate
#         (>= MIN_CHARS and no `////` run), and the hard arms must produce the same text byte for byte:
#           comb_op       -- the combine op, with both DSV4 ops off, vs the chain
#           both_ops      -- vs `mix_op`, so the combine op is still exact with the mixer fused
#           cuda_fuse_off -- the CUDA chain matchers vs the chain
#           mix_op        -- on a CHAIN-mixer class only, where the switch is inert (`LLAMA_FUSED_HC_MIX`
#                            selects a mixer this model never builds; the verdict says so instead of
#                            implying coverage)
#         A mismatch names the arm.
#   NUMERIC (sha-anchored, r236): the DeepSeek-V4 HC pre/post op pair (which qwen4exp also
#         builds through) does NOT reproduce its ggml chain and must not be required to: those
#         two ops own the HC16 bf16 producer slots (`dsv4-hc.cu`: "a numerics change vs the F32
#         chain, approved as a prefill re-baseline"), so `LLAMA_FUSED_DSV4_HC_PRE`/`_POST` are
#         numeric arms, not aliases.  Measured with `gate-prefill-logits.sh --ab` on the
#         BF16-hc class (Flash-Next GSQ-IQ3_XXS, 3 devices `-sm layer`, wikitext 512 x 40,
#         base PPL 2.8489): pre op **0.013217 / 96.392 %** and post op **0.013198 / 96.186 %**
#         against their own chain -- the documented re-baseline class, and *smaller* KLD than
#         the IQ4_NL reference below.  r156's IQ4_NL point, for scale: 0.027936 / 96.471 % at
#         base PPL 1.7888 (`MANIFESTS.md` gate 7 carries a second IQ4_NL figure, 0.021542 /
#         95.814 %, from its own unnamed point -- both are historical, neither is a target).
#         So byte-identity is NOT expected on any class, and the honest regression detector is
#         the arm's OWN recorded text: each arm is deterministic per class, so `SOFT_EXPECT`
#         anchors it and a real numeric change in either op FAILS the gate instead of printing
#         "reported" (the hole `TODO` #292 exposed).
#         The two SHIPPING states are numeric arms as well, because each is a combination that
#         includes those ops: `all_ops` (the default on a Meta/tensor-split run, where
#         `fused_dsv4_hc_post` auto-enables) and `ops_no_post` (the default on one device, where it
#         auto-disables).  The `defaults` arm runs with NO env at all and must equal one of them --
#         which is also what proves WHICH auto state the run took, so a flip in that default is a
#         FAIL instead of a silently different arm set.
#   NUMERIC, mixer (r309): on a FUSED-mixer class `mix_op` is a numeric arm as well, and it is the one
#         finding that came out of isolating the switches.  The fused mixer's *inject* half is the fused
#         band dot kernel (`hc_mix_down_dots`), where the chain's is `build_lora_mm(w_inject, xn)` plus a
#         SCALE, so the two do not reduce in the same order: measured against the true chain on
#         GSQ-IQ3_XXS, 3 devices `-sm tensor`, wikitext 20 x 512, `llama-perplexity --kl-divergence`:
#         **mean KLD 0.000000 +- 0.000000 / same-top-p 100.000 %** -- rounding level, NOT a defect, and
#         it does not move PPL.  It can still flip a greedy near-tie (that class reads 449
#         `d62340fbf566` through the chain and 449 `6a102dbc62d6` through the mixer under `-sm tensor`,
#         while under `-sm layer` the two coincide), which is why the arm is anchored instead of hard.
#         The `LLAMA_FUSED_HC_MIX` convention itself (the /hc that issue #228 was about) is a LARGE
#         drift and still fails here: an arm that moved by /hc cannot land on its anchor.
#   The arms cover both halves independently and the CUDA-side chain fusions.
#
#   The hc weight TYPE decides which mixer runs, so run this once per class:
#     * a Q8_0 or BF16 hc_attn_down GGUF  -> the fused BF16/Q8_0 mixer (fused mix arm is real)
#     * a post-#31028 MERGED hc_*_down_inject GGUF -> ALSO fused-capable since r303 (the op takes two
#       row views of the merged tensor), so its class line says so instead of "chain"
#     * any other quant (Q5_K, IQ4_NL, ...) -> the chain mixer (`mix_op` is INERT: the model never
#       builds GGML_OP_HC_MIX for it, so that arm passes without exercising the mixer fusion -- the
#       gate says so in its class line and the arm's verdict instead of implying coverage)
#   `scripts/gate-hc-fusions.sh --types MODEL.gguf` prints the class of a model (and its
#   blk.0 hc_attn_* tensor types) without running anything.
#
# ANCHORS ARE PER (class, split state, 3 devices).  A generated text is a per-split, per-device-count
# reference, so the built-in anchor sets cover the two states of the normal 3-device invocation:
# `meta` (a tensor split, where `fused_dsv4_hc_post` auto-enables) and `local` (one device or
# `-sm layer`), both at CTX 8192 / NGEN 96 / `EXTRA_ARGS` as the gate defaults it, on the 2026-11-17
# build.  The `defaults` arm decides which state a run is in, so a `-sm tensor` and a `-sm layer` run
# are both anchored out of the box; another device count, model or `EXTRA_ARGS` needs its own anchors
# via `SOFT_EXPECT` (or prints `reported`), and the HARD arms must agree in every case.
#
# Usage:
#   ./gate-hc-fusions.sh [--no-faults] [MODEL.gguf]
#   ./gate-hc-fusions.sh --types MODEL.gguf
# Environment:
#   MODEL        GGUF path          (default: $MODEL_ROOT/GSQ-IQ3_XXS/... the BF16-hc class)
#   MODEL_ROOT   qwen4exp model root (default: /llm/models/Qwen3.8/Flash-Next)
#   LLAMA_CLI    llama-cli          (default: ~/llama.cpp/build-rocm/bin/llama-cli)
#   NGEN         generated tokens    (default: 96)
#   CTX          context size        (default: 8192)
#   MIN_CHARS    reference-arm floor (default: 64)
#   EXTRA_ARGS   extra llama-cli args (e.g. "-ngl 99 -sm tensor -ncmoe 40")
#   KEEPOUT      dir for the per-arm logs (default: mktemp -d)
#   TIME_LIMIT   per-arm timeout, seconds (default: 3000)
#   --no-faults  skip the kernel-journal fault check (a box with no readable kernel log)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL_ROOT="${MODEL_ROOT:-/llm/models/Qwen3.8/Flash-Next}"
NGEN="${NGEN:-96}"
CTX="${CTX:-8192}"
MIN_CHARS="${MIN_CHARS:-64}"
EXTRA_ARGS="${EXTRA_ARGS:--ngl 99 -fa on}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-3000}"

# A short, fixed prompt: the collapse this gate exists for is unmistakable at this length
# (the r147 Q5_K arm looped "The user is asking for a short paragraph about the Eiffel Tower").
PROMPT='Write a short paragraph about the Eiffel Tower.'

types_of() { # print the blk.0 hc_attn_* tensor types of a GGUF (no llama build needed)
    python3 - "$1" <<'PY'
import glob, os, re, struct, sys
TN = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',
      12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',
      20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',
      29:'IQ1_M',30:'BF16'}
def s(f):
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n).decode('utf-8', 'replace')
def v(f, t):
    if t in (0, 1, 7):  return f.read(1)[0]
    if t == 2:  return struct.unpack('<H', f.read(2))[0]
    if t == 3:  return struct.unpack('<h', f.read(2))[0]
    if t == 4:  return struct.unpack('<I', f.read(4))[0]
    if t == 5:  return struct.unpack('<i', f.read(4))[0]
    if t == 6:  return struct.unpack('<f', f.read(4))[0]
    if t == 8:  return s(f)
    if t == 9:
        at = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
        return [v(f, at) for _ in range(n)]
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    raise SystemExit('unknown gguf value type %d' % t)

def scan(path):
    """(arch, {tensor: type}) for one GGUF shard."""
    f = open(path, 'rb')
    if f.read(4) != b'GGUF':
        return None, {}
    struct.unpack('<I', f.read(4))[0]
    nt, nkv = struct.unpack('<Q', f.read(8))[0], struct.unpack('<Q', f.read(8))[0]
    arch = None
    for _ in range(nkv):
        k = s(f); t = struct.unpack('<I', f.read(4))[0]; val = v(f, t)
        if k == 'general.architecture': arch = val
    found = {}
    for _ in range(nt):
        name = s(f); nd = struct.unpack('<I', f.read(4))[0]
        struct.unpack('<%dQ' % nd, f.read(8*nd))
        tt = struct.unpack('<I', f.read(4))[0]; struct.unpack('<Q', f.read(8))[0]
        if name.startswith('blk.0.hc_attn_') and name.endswith('.weight'):
            found[name] = TN.get(tt, str(tt))
    return arch, found

arch, found = scan(sys.argv[1])
if arch != 'qwen4exp':
    print('arch=%s (not qwen4exp -- this gate does not apply)' % arch); sys.exit(0)
if not found:
    # a split GGUF whose shard 0 carries only metadata: scan the sibling shards for blk.0
    d, base = os.path.split(os.path.abspath(sys.argv[1]))
    m = re.match(r'^(.*?)-\d{5}-of-\d{5}\.gguf$', base)
    if m:
        for cand in sorted(glob.glob(os.path.join(d, m.group(1) + '-*.gguf'))):
            if cand == sys.argv[1]:
                continue
            a2, f2 = scan(cand)
            if f2:
                found = f2
                print('(blk.0 tensors read from %s)' % os.path.basename(cand))
                break
for k in sorted(found):
    print('%-34s %s' % (k, found[k]))
dn = found.get('blk.0.hc_attn_down.weight')
di = found.get('blk.0.hc_attn_down_inject.weight')
if dn is not None:
    print('class: %s mixer (%s hc_attn_down)' %
          ('FUSED' if dn in ('Q8_0', 'BF16') else 'CHAIN', dn))
elif di is not None:
    # post-#31028 converter layout: the down and inject products share ONE tensor.  Since r303 that is
    # no longer a chain-only shape -- build_hc_mix hands the fused op two row views of the merged tensor
    # (the kernel addresses every weight row packed at ggml_row_size(ne[0]), and ne[0] is unchanged), so
    # the mixed layout keeps the fused decode band.  Name the layout AND say the mixer is reachable,
    # because reading this line as "chain" is how the shape went unexercised for six releases.
    print('class: FUSED-capable mixer (merged hc_attn_down_inject %s, post-#31028 layout)' % di)
else:
    print('class: unknown (no blk.0 hc_attn_down / hc_attn_down_inject tensor)')
PY
}

if [ "${1:-}" = "--types" ]; then
    [ -n "${2:-}" ] || { echo "usage: $0 --types MODEL.gguf" >&2; exit 2; }
    types_of "$2"
    exit $?
fi

FAULTS=1
MODEL_ARG=""
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        -*) echo "usage: $0 [--no-faults] [MODEL.gguf]" >&2; exit 2 ;;
        *) MODEL_ARG="$a" ;;
    esac
done

MODEL="${MODEL_ARG:-${MODEL:-$MODEL_ROOT/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}}"

[ -x "$LLAMA_CLI" ] || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "FAIL: model not found: $MODEL" >&2; exit 2; }
command -v python3 >/dev/null || { echo "FAIL: python3 required" >&2; exit 2; }
[ -f "$REPO_DIR/scripts/extract-generated.py" ] || { echo "FAIL: extractor not found" >&2; exit 2; }

mkdir -p "$KEEPOUT"
# The kernel journal is shared between runs, so the fault window starts here, before the first arm (#294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
PROMPT_SHA="$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"

# A fresh-process comparison needs the library pinned (`INVARIANTS.md`, r302 §44): hipBLASLt's
# per-process selection reorders the prefill GEMMs, so ONE unpinned arm in the set reads different
# text -- measured 2026-11-17 on GSQ-IQ3_XXS all-VRAM `-sm tensor`: 449 `6a102dbc62d6` on the two
# chain-mixer arms and 459 `7c5e5463fd6a` on the other five, identical to the arm set with every
# other env pinned, i.e. a library artefact, not a fusion disagreement.  With the pin the whole
# (then seven-arm) set is byte-identical to the chain.
export ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"
# The MoE residency prior is an INPUT (`GREEDY-PURITY.md` §89): a learned heat-map selects a
# different expert kernel set, and the sets are not bit-identical.
export MOE_EXPERT_HEATMAP="${MOE_EXPERT_HEATMAP:-load}"
# The HC BF16-stream marks are INPUTS too (they change the combine's producer/consumer precision),
# and a model directory's own `runme` sets them, so an inherited value must not silently re-baseline
# every arm (the anchors below assume them at their defaults).
export LLAMA_HC_BLK16="${LLAMA_HC_BLK16:-0}"
export LLAMA_HC_RES16="${LLAMA_HC_RES16:-0}"
export GGML_CUDA_MMB_DOWN16="${GGML_CUDA_MMB_DOWN16:-0}"

echo "==> qwen4exp hyper-connection fusion-equivalence gate"
echo "    cli=$LLAMA_CLI"
echo "    model=$MODEL"
types_of "$MODEL" | sed 's/^/    /'
echo "    ctx=$CTX ngen=$NGEN extra='$EXTRA_ARGS'"
echo "    ROCBLAS_USE_HIPBLASLT=$ROCBLAS_USE_HIPBLASLT MOE_EXPERT_HEATMAP=$MOE_EXPERT_HEATMAP"
echo "    LLAMA_HC_BLK16=$LLAMA_HC_BLK16 LLAMA_HC_RES16=$LLAMA_HC_RES16 GGML_CUDA_MMB_DOWN16=$GGML_CUDA_MMB_DOWN16"
echo "    prompt sha256=$PROMPT_SHA  (extractor: scripts/extract-generated.py)"
echo "    logs: $KEEPOUT"
echo

# The DSV4_HC_PRE/_POST ops are NUMERIC arms, not exact reimplementations: `dsv4_hc_pre_f32`'s
# `wbf16`/`xbf16` arms read the HC16 bf16 producer slots deliberately (see the header), so
# `LLAMA_FUSED_DSV4_HC_PRE=0` swaps the producer precision as well as the fusion -- BF16-hc class,
# measured r236: mean KLD 0.013217 / same-top-p 96.392 % (pre) and 0.013198 / 96.186 % (post)
# against the chain.  Byte-identity is therefore NOT expected on any class.  Each arm IS
# deterministic per class, so its text is anchored instead: SOFT_EXPECT (below) makes a real
# numeric change FAIL rather than print "reported" -- the hole `TODO` #292 exposed.
# EVERY PINNED ARM NAMES ALL FOUR SWITCHES, in the order MIX COMBINE PRE POST (r309, `TODO` #341).
# An arm that leaves one at its default is not the arm it claims to be: `fused_dsv4_hc_post` defaults
# to `meta_split_active`, and the DSV4 ops are unreachable from a mix/combine-op graph, so a
# "combine on" arm that leaves the DSV4 switches alone is also a "DSV4 post op OFF" arm.
HARD_ARMS=(
    "chain_chain   :LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0 LLAMA_FUSED_DSV4_HC_PRE=0 LLAMA_FUSED_DSV4_HC_POST=0"
    "mix_op        :LLAMA_FUSED_HC_MIX=1 LLAMA_FUSED_HC_COMBINE=0 LLAMA_FUSED_DSV4_HC_PRE=0 LLAMA_FUSED_DSV4_HC_POST=0"
    "comb_op       :LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=1 LLAMA_FUSED_DSV4_HC_PRE=0 LLAMA_FUSED_DSV4_HC_POST=0"
    "both_ops      :LLAMA_FUSED_HC_MIX=1 LLAMA_FUSED_HC_COMBINE=1 LLAMA_FUSED_DSV4_HC_PRE=0 LLAMA_FUSED_DSV4_HC_POST=0"
    "cuda_fuse_off :LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0 LLAMA_FUSED_DSV4_HC_PRE=0 LLAMA_FUSED_DSV4_HC_POST=0 GGML_CUDA_DISABLE_HC_FUSION=1"
)
SOFT_ARMS=(
    "dsv4_pre_op   :LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0 LLAMA_FUSED_DSV4_HC_PRE=1 LLAMA_FUSED_DSV4_HC_POST=0"
    "dsv4_post_op  :LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0 LLAMA_FUSED_DSV4_HC_PRE=0 LLAMA_FUSED_DSV4_HC_POST=1"
    "all_ops       :LLAMA_FUSED_HC_MIX=1 LLAMA_FUSED_HC_COMBINE=1 LLAMA_FUSED_DSV4_HC_PRE=1 LLAMA_FUSED_DSV4_HC_POST=1"
    "ops_no_post   :LLAMA_FUSED_HC_MIX=1 LLAMA_FUSED_HC_COMBINE=1 LLAMA_FUSED_DSV4_HC_PRE=1 LLAMA_FUSED_DSV4_HC_POST=0"
)
# the shipping state itself: no env prefix at all.  It must equal one of the two explicit states
# (`all_ops` on a Meta/tensor-split run, `ops_no_post` on one device), which is also what reports
# WHICH side of the split-dependent `fused_dsv4_hc_post` default this run took.
DEFAULT_ARM="defaults:-"
ARMS=("${HARD_ARMS[@]}" "${SOFT_ARMS[@]}" "$DEFAULT_ARM")

# Every pinned arm must name all four switches, with a 0/1 value: a typo or a dropped variable would
# silently turn that arm into the default run (the `env -` class of mistake, r303).
for arm in "${HARD_ARMS[@]}" "${SOFT_ARMS[@]}"; do
    n=0
    for sw in LLAMA_FUSED_HC_MIX LLAMA_FUSED_HC_COMBINE LLAMA_FUSED_DSV4_HC_PRE LLAMA_FUSED_DSV4_HC_POST; do
        if printf '%s' "${arm#*:}" | grep -Eq "(^|[[:space:]])$sw=[01]([[:space:]]|$)"; then
            n=$((n+1))
        fi
    done
    if [ "$n" -ne 4 ]; then
        echo "FAIL: arm '${arm%%:*}' pins $n of the 4 LLAMA_FUSED_* switches (all four are required)" >&2
        exit 2
    fi
done
unset n sw arm

# NUMERIC-ARM ANCHORS (see the header).  They are recorded per (hc-down class, split state), because
# a generated text is a per-split, per-device-count reference (INVARIANTS: a tensor-split text is a
# reference for ITS device count).  `meta` is the state a TENSOR split takes -- a Meta device in the
# graph, which is also what turns `fused_dsv4_hc_post` on -- and `local` is one device or `-sm layer`.
# The `defaults` arm (below) is what decides which state this run is in, and every built-in set is a
# 3-device measurement on the 2026-11-17 (r308) build:
#   fused-mixer class (BF16/Q8_0 hc_attn_down), Flash-Next GSQ-IQ3_XXS
#     `-sm tensor`: true chain 449 d62340fbf566 | mix_op/comb_op/both_ops/cuda_fuse_off all 449
#                   dsv4_pre_op 449 6a102dbc62d6 | dsv4_post_op 447 3f049f67781c
#                   all_ops 449 6a102dbc62d6 | ops_no_post 449 6a102dbc62d6
#     `-sm layer` : true chain 449 6a102dbc62d6 | the four hard arms all 449 (identical)
#                   dsv4_pre_op 444 a512ee68c4b3 | dsv4_post_op 449 6a102dbc62d6
#                   all_ops 449 6a102dbc62d6 | ops_no_post 444 a512ee68c4b3
#   chain-mixer class (any other hc quant), Flash-Next IQ4_NL -- the `TODO` #341 file
#     `-sm tensor`: true chain 450 bc46095cf4c6 | the four hard arms all 450 (identical)
#                   dsv4_pre_op 444 7ea23f309705 | dsv4_post_op 454 9b4b033b33f7
#                   all_ops 446 a5d0ec7b809d | ops_no_post 444 7ea23f309705
#     `-sm layer` : true chain 453 f242246a382b | the four hard arms all 453 (identical)
#                   dsv4_pre_op 453 f242246a382b | dsv4_post_op 444 7ea23f309705
#                   all_ops 459 9f8819c4c2c5 | ops_no_post 453 f242246a382b
# RE-ANCHORED 2026-11-21 (r328) -- read this before re-anchoring anything again.
# The two `fusedmixer` rows above were recorded on r308 and had gone stale underneath a r290..r319 change, so
# this gate was RED on the RELEASED r327 already: r327's own binaries read exactly the r328 numbers
#   local: chain 444 a512ee68c4b3 | mix 447 3f049f67781c | pre 443 c8db61d5dcc8 | post 449 6a102dbc62d6
#   meta : chain 449 6a102dbc62d6 | mix 449 6a102dbc62d6   | pre 444 a512ee68c4b3 | post 447 3f049f67781c
# (measured with `LLAMA_CLI=~/llama.cpp/build-rocm/bin-r327/llama-cli`; the whole gate run against that build
# reproduces the same three FAILs), i.e. the r328 re-base moved NOTHING here and there is no HC-purity defect
# to fix.  What moved is the class's TEXT: every arm (chain included) shifted inside r290..r319 -- `bin-r289`
# still reads the r308 row, `bin-r320` already reads the current one -- so the rows above are re-derived from
# the shipped behaviour and a further move still fails.
# The lesson, and the reason this note exists: an HC-class generated text is NOT portable across a numerics
# change, and this gate used to leave the REFERENCE arm unanchored.  With no anchor on the chain, a moved
# chain silently turns every other arm's anchor into a statement about a text nobody re-measured, and the only
# visible symptom is "numeric arms left their anchors" -- which is exactly how the r328 re-base came to be
# blamed for it (`RUNGS.md` R18.4).  `chain_chain` is anchored from here on, and a moved reference arm fails
# as a moved reference arm.  Re-derive a row TOGETHER, never one arm at a time.
#
# `SOFT_EXPECT="arm=sha ..."` replaces the selected set (another model, another device count); an empty
# value disables anchoring for a class that has none recorded.
HC_TYPE="$(types_of "$MODEL" | sed -n 's/^blk\.0\.hc_attn_down\.weight[[:space:]]*//p' | head -1)"
# which mixer the MODEL builds is a property of the class line, not of the anchor key (a Q5_K file is a
# chain mixer with no anchors, a merged file is fused-capable, an unknown file could be either -- treat
# an unknown as fused so `mix_op` is never waved through as "inert")
case "$(types_of "$MODEL" | sed -n 's/^class: //p' | head -1)" in
    FUSED*)   mixer_builds=fused ;;
    *merged*) mixer_builds=fused ;;
    CHAIN*)   mixer_builds=chain ;;
    *)        mixer_builds=unknown ;;
esac
case "$HC_TYPE" in
    BF16|Q8_0) HC_CLASS=fusedmixer ;;
    IQ4_NL)    HC_CLASS=iq4nl ;;
    *)         HC_CLASS=none ;;
esac
anchors_for() { # $1 = hc class, $2 = split state (meta|local)
    case "$1/$2" in
        # re-derived 2026-11-21 (r328) from the RELEASED r327 binaries, which read these values byte for
        # byte -- see the "RE-ANCHORED" note in the header.  `chain_chain` is anchored as well now.
        fusedmixer/meta)  echo "chain_chain=6a102dbc62d6 mix_op=6a102dbc62d6 dsv4_pre_op=a512ee68c4b3 dsv4_post_op=3f049f67781c all_ops=6a102dbc62d6 ops_no_post=a512ee68c4b3" ;;
        fusedmixer/local) echo "chain_chain=a512ee68c4b3 mix_op=3f049f67781c dsv4_pre_op=c8db61d5dcc8 dsv4_post_op=6a102dbc62d6 all_ops=6a102dbc62d6 ops_no_post=c8db61d5dcc8" ;;
        iq4nl/meta)       echo "dsv4_pre_op=7ea23f309705 dsv4_post_op=9b4b033b33f7 all_ops=a5d0ec7b809d ops_no_post=7ea23f309705" ;;
        iq4nl/local)      echo "dsv4_pre_op=f242246a382b dsv4_post_op=7ea23f309705 all_ops=9f8819c4c2c5 ops_no_post=f242246a382b" ;;
        *)                echo "" ;;
    esac
}
mixer_class="$mixer_builds"
[ "$mixer_class" = "chain" ]   && echo "    mixer: chain (this class never builds GGML_OP_HC_MIX: the mix_op arm is inert)"
[ "$mixer_class" = "fused" ]   && echo "    mixer: fused (the model builds GGML_OP_HC_MIX in the decode band)"
[ "$mixer_class" = "unknown" ] && echo "    mixer: UNKNOWN (no hc_attn_down type line; mix_op is graded as numeric, never as inert)"

declare -A RC_OF=()
declare -A SHA_OF=()
declare -A CHARS_OF=()
for arm in "${ARMS[@]}"; do
    name="${arm%%:*}"
    envs="${arm#*:}"
    [ "$envs" = "-" ] && envs=""          # the default arm: no env prefix at all (see below)
    name="$(echo "$name" | tr -d ' ')"
    log="$KEEPOUT/$name.log"
    # The default arm carries no env prefix, so it must run WITHOUT `env`: GNU `env -`
    # (which `env $envs` with envs="-" expands to) is `env -i` and wipes the caller's
    # environment -- HIP_VISIBLE_DEVICES included -- so the baseline arm would enumerate
    # the iGPU under `-sm tensor` and die with `ROCm error: invalid kernel file` while
    # every other arm ran.  (#321's session found this; the arm had been silently failing
    # on any pinned multi-device run.)
    # shellcheck disable=SC2086
    ${envs:+env $envs} timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL" -c "$CTX" \
        $EXTRA_ARGS -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 \
        --no-display-prompt --single-turn > "$log" 2>&1
    rc=$?
    read -r chars sha < <(python3 "$REPO_DIR/scripts/extract-generated.py" "$log" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1)
    RC_OF[$name]="$rc"
    SHA_OF[$name]="${sha:-none}"
    CHARS_OF[$name]="${chars:-0}"
done

# ---- verdicts.  Two passes, because the `defaults` arm is what decides the split state, and the
# state picks the anchor set for the numeric arms: the arms all run first, then are graded.
STATE=""
if [ "${SHA_OF[defaults]:-x}" = "${SHA_OF[all_ops]:-y}" ] && \
   [ "${CHARS_OF[defaults]:-0}" = "${CHARS_OF[all_ops]:-1}" ]; then
    STATE=meta
elif [ "${SHA_OF[defaults]:-x}" = "${SHA_OF[ops_no_post]:-y}" ] && \
     [ "${CHARS_OF[defaults]:-0}" = "${CHARS_OF[ops_no_post]:-1}" ]; then
    STATE=local
fi
ANCHOR_SET="${SOFT_EXPECT-$(anchors_for "$HC_CLASS" "$STATE")}"
declare -A ANCHOR_MAP=()
for kv in $ANCHOR_SET; do ANCHOR_MAP["${kv%%=*}"]="${kv#*=}"; done
if [ -n "$STATE" ]; then
    echo "    split state: $STATE (dsv4_hc_post auto=$([ "$STATE" = meta ] && echo ON || echo OFF))"
else
    echo "    split state: UNKNOWN (the 'defaults' arm matched neither explicit state)"
fi
if [ -n "$ANCHOR_SET" ]; then echo "    numeric-arm anchors: $ANCHOR_SET"; else echo "    numeric-arm anchors: none"; fi

ref_sha="${SHA_OF[chain_chain]:-}"
ref_chars="${CHARS_OF[chain_chain]:-0}"
fails=0
warns=0
echo
printf '%-14s %-6s %8s %14s  %s\n' arm rc chars sha verdict
for arm in "${ARMS[@]}"; do
    name="$(echo "${arm%%:*}" | tr -d ' ')"
    rc="${RC_OF[$name]:-1}"; chars="${CHARS_OF[$name]:-0}"; sha="${SHA_OF[$name]:-none}"
    soft=0
    for s in "${SOFT_ARMS[@]}"; do [ "${s%%:*}" = "${arm%%:*}" ] && soft=1; done
    # on a class that BUILDS the fused mixer, mix_op is a numeric arm too: the op's inject half is
    # the fused band dot kernel where the chain's is the lora GEMM, so its logits are rounding-level
    # but not bit-identical (measured: KLD 0.000000 / same-top-p 100.000 %, 20 x 512 wikitext), and a
    # greedy near-tie can flip.  On a chain-mixer class the switch is inert and it stays hard.
    if [ "$name" = "mix_op" ] && [ "$mixer_class" != "chain" ]; then soft=1; fi
    log="$KEEPOUT/$name.log"
    verdict=""
    if [ "$rc" -ne 0 ]; then
        verdict="FAIL (exit $rc)"
    elif [ "$name" = "chain_chain" ]; then
        # THE REFERENCE ARM IS ANCHORED TOO (r328, 2026-11-21).  It was not, and that hole is exactly what
        # made the r328 re-base mis-read a PRE-EXISTING chain move as "the numeric arms left their
        # anchors": with no anchor on the chain, every other arm's anchor silently became a statement about
        # a text nobody re-measured.  A moved reference arm must fail HERE and say so, because it makes
        # every other anchor stale at once -- re-derive the whole row together, never one arm at a time.
        exp="${ANCHOR_MAP[chain_chain]:-}"
        if [ "$chars" -lt "$MIN_CHARS" ]; then
            verdict="FAIL (reference arm generated $chars < $MIN_CHARS chars)"
        elif grep -q '////' "$log"; then
            verdict="FAIL (reference arm shows the '////' corruption)"
        elif [ -n "$exp" ] && [ "$sha" != "$exp" ]; then
            verdict="FAIL (the REFERENCE arm moved: $chars chars $sha, anchor $exp -- every anchor in this row is stale; re-derive them together, see the header)"
        else
            verdict="reference (true chain)"
        fi
    elif [ "$name" = "defaults" ]; then
        # the shipping state must equal one of the two explicit states; naming which one is what
        # proves the split-dependent `fused_dsv4_hc_post` default took the side it should
        if [ -n "$STATE" ]; then
            best=$([ "$STATE" = meta ] && echo all_ops || echo ops_no_post)
            verdict="ok (== $best: dsv4_hc_post auto=$([ "$STATE" = meta ] && echo ON || echo OFF))"
        else
            verdict="FAIL (equals neither of 'all_ops ops_no_post': $chars chars $sha)"
        fi
    elif [ "$name" = "both_ops" ] && [ "$mixer_class" != "chain" ]; then
        # the mixer op is on here AND in mix_op, so this pair isolates the COMBINE op with the
        # mixer fused -- the combine must still be byte-identical (it is pure elementwise)
        if [ "$sha" = "${SHA_OF[mix_op]:-x}" ] && [ "$chars" = "${CHARS_OF[mix_op]:-y}" ]; then
            verdict="ok (== mix_op: the combine op is exact with the fused mixer on)"
        else
            verdict="FAIL (differs from mix_op: $chars chars $sha)"
        fi
    elif [ "$sha" = "$ref_sha" ] && [ "$chars" = "$ref_chars" ]; then
        if [ "$name" = "mix_op" ] && [ "$mixer_class" = "chain" ]; then
            verdict="ok (inert: this class has no fused mixer)"
        elif [ "$soft" -eq 1 ]; then
            # a numeric arm may coincide with the chain on a class where its bf16 slots do not engage;
            # that is the documented state for that (class, split), so it is a pass, not "reported"
            verdict="ok (numeric arm coincides with the chain on this class/state)"
        else
            verdict="ok"
        fi
    elif [ "$soft" -eq 1 ]; then
        # a numeric arm: it is not expected to match the chain, but its own text is deterministic per
        # (class, split state), so a recorded anchor turns "the op's numerics moved" into a FAIL
        exp="${ANCHOR_MAP[$name]:-}"
        if [ -z "$exp" ]; then
            verdict="reported (numeric arm, differs -- no anchor recorded for this class/state)"
        elif [ "$sha" = "$exp" ]; then
            verdict="reported (numeric arm, anchor $exp)"
        else
            verdict="FAIL (numeric arm moved: $chars chars $sha, anchor $exp; for another model/split shape pass SOFT_EXPECT)"
        fi
    else
        verdict="FAIL (differs from chain_chain)"
    fi
    case "$verdict" in
        FAIL*) fails=$((fails+1));;
        reported*) warns=$((warns+1));;
    esac
    printf '%-14s %-6s %8s %14s  %s\n' "$name" "$rc" "$chars" "$sha" "$verdict"
done

echo
# the kernel journal, scoped to this gate's own window (#294)
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        echo "PASS: no GPU fault in the kernel journal (since ${GATE_SINCE_TS:-start}) or the arm logs"
    else
        echo "FAIL: a GPU fault was recorded during the arms (since ${GATE_SINCE_TS:-the journal start}):"
        echo "$out" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
fi

if [ "$fails" -ne 0 ]; then
    echo "FAIL: $fails arm(s) failed their grading -- a hard arm moved, a numeric arm left its anchor," \
         "or the shipping default matched neither explicit state (see $KEEPOUT)"
    exit 1
fi
echo "PASS: every hard arm is byte-identical to the arm it is graded against (true chain: $ref_chars" \
     "chars, sha $ref_sha), the shipping default matches an explicit state, and $warns numeric arm(s)" \
     "differ from the chain as documented and match their anchors"
