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
# WHAT IT CHECKS
#   HARD: every arm produces the same generated text (byte-identical, same sha) as the
#         chain+chain reference arm, and the reference arm is not degenerate (>= MIN_CHARS
#         and no `////` run).  A mismatch names the arm.
#   REPORTED: the DeepSeek-V4 HC pre/post op pair (which qwen4exp also builds through) is
#         printed with its sha and divergence but does not fail the gate.  Those two ops are
#         NOT an exact reimplementation of their ggml chain: they own the HC16 bf16 producer
#         slots (`dsv4-hc.cu`: "a numerics change vs the F32 chain, approved as a prefill
#         re-baseline"), so `LLAMA_FUSED_DSV4_HC_PRE`/`_POST` are numeric arms, not aliases.
#         Baseline on the IQ4_NL class (r156, `gate-prefill-logits.sh --ab`): the fused pre op
#         reads mean KLD 0.027936 / same-top-p 96.471 % against its own chain at base PPL
#         1.7888, the same order the combine+norm fusion has always shown (0.0185-0.0213 /
#         95.6 %, WORKLOG 261025.014).  A *change* in that magnitude is the regression signal;
#         byte-identity is not expected there.  A class where they DO agree (BF16-hc) prints
#         equal shas, so nothing is lost.
#   The arms cover both halves independently and the CUDA-side chain fusions.
#
#   The hc weight TYPE decides which mixer runs, so run this once per class:
#     * a Q8_0 or BF16 hc_attn_down GGUF  -> the fused BF16/Q8_0 mixer (fused mix arm is real)
#     * any other quant (Q5_K, IQ4_NL, ...) -> the chain mixer (fused mix arm is a no-op)
#   `scripts/gate-hc-fusions.sh --types MODEL.gguf` prints the class of a model (and its
#   blk.0 hc_attn_* tensor types) without running anything.
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
if dn is not None:
    print('class: %s mixer (%s hc_attn_down)' %
          ('FUSED' if dn in ('Q8_0', 'BF16') else 'CHAIN', dn))
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

echo "==> qwen4exp hyper-connection fusion-equivalence gate"
echo "    cli=$LLAMA_CLI"
echo "    model=$MODEL"
types_of "$MODEL" | sed 's/^/    /'
echo "    ctx=$CTX ngen=$NGEN extra='$EXTRA_ARGS'"
echo "    prompt sha256=$PROMPT_SHA  (extractor: scripts/extract-generated.py)"
echo "    logs: $KEEPOUT"
echo

# The DSV4_HC_PRE/_POST ops are REPORTED arms (r156).  They are numeric re-baselines, not exact
# reimplementations: `dsv4_hc_pre_f32`'s `wbf16`/`xbf16` arms read the HC16 bf16 producer slots
# deliberately (see the header), so `LLAMA_FUSED_DSV4_HC_PRE=0` swaps the producer precision as well
# as the fusion.  Measured on the IQ4_NL class at r156: mean KLD 0.027936 / same-top-p 96.471 %
# (base PPL 1.7888), i.e. the same class of difference the combine+norm fusion has always shown
# (0.0185-0.0213 / 95.6 %, WORKLOG 261025.014) -- so it is reported, and a *change* in it is the
# signal.  Byte-identity IS expected on the BF16-hc class, where both arms still agree (449 chars
# `6a102dbc62d6`).
HARD_ARMS=(
    "chain_chain   :LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0"
    "mixed_comb    :LLAMA_FUSED_HC_COMBINE=0"
    "mixfused_comb :LLAMA_FUSED_HC_MIX=0"
    "fused_fused   :-"
    "no_cuda_fuse  :GGML_CUDA_DISABLE_HC_FUSION=1"
)
SOFT_ARMS=(
    "dsv4_pre_chain:LLAMA_FUSED_DSV4_HC_PRE=0"
    "dsv4_post_op  :LLAMA_FUSED_DSV4_HC_POST=1"
)
ARMS=("${HARD_ARMS[@]}" "${SOFT_ARMS[@]}")

ref_sha=""
ref_chars=0
fails=0
warns=0
printf '%-14s %-6s %8s %14s  %s\n' arm rc chars sha verdict
for arm in "${ARMS[@]}"; do
    name="${arm%%:*}"
    envs="${arm#*:}"
    name="$(echo "$name" | tr -d ' ')"
    soft=0
    for s in "${SOFT_ARMS[@]}"; do [ "${s%%:*}" = "${arm%%:*}" ] && soft=1; done
    log="$KEEPOUT/$name.log"
    # shellcheck disable=SC2086
    env $envs timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL" -c "$CTX" \
        $EXTRA_ARGS -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 \
        --no-display-prompt --single-turn > "$log" 2>&1
    rc=$?
    read -r chars sha < <(python3 "$REPO_DIR/scripts/extract-generated.py" "$log" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1)
    chars="${chars:-0}"; sha="${sha:-none}"
    verdict=""
    if [ "$rc" -ne 0 ]; then
        verdict="FAIL (exit $rc)"
    elif [ "$name" = "chain_chain" ]; then
        ref_sha="$sha"; ref_chars="$chars"
        if [ "$chars" -lt "$MIN_CHARS" ]; then
            verdict="FAIL (reference arm generated $chars < $MIN_CHARS chars)"
        elif grep -q '////' "$log"; then
            verdict="FAIL (reference arm shows the '////' corruption)"
        else
            verdict="reference"
        fi
    elif [ "$sha" = "$ref_sha" ] && [ "$chars" = "$ref_chars" ]; then
        verdict="ok"
    elif [ "$soft" -eq 1 ]; then
        verdict="reported (numeric arm, differs -- see the header)"
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
    echo "FAIL: $fails arm(s) disagreed with the unfused chain (see $KEEPOUT)"
    exit 1
fi
if [ "$warns" -ne 0 ]; then
    echo "PASS: all $(( ${#ARMS[@]} - warns )) hard arms are byte-identical to the unfused chain" \
         "($ref_chars chars, sha $ref_sha); $warns reported arm(s) differ (documented numeric re-baseline)"
else
    echo "PASS: all ${#ARMS[@]} arms are byte-identical to the unfused chain ($ref_chars chars, sha $ref_sha)"
fi
