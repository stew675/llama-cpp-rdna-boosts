#!/usr/bin/env bash
# Prefill-logit KLD gate (issue #113).
#
# Automation of `benchmarks/prefill-logit-methodology.md`: compare the current build's
# prefill logits against a recorded known-good base with `llama-perplexity
# --kl-divergence`, and fail unless the divergence is rounding-level
# (mean KLD <= MAX_KLD, same-top-p >= MIN_TOPP).
#
# WHY THIS EXISTS:
#   Same-seed coherence and the MTP gate sample the decode distribution *from the
#   post-prefill state*; a uniform prefill shift is invisible to both.  The
#   default-on BF16/WMMA chunked GDN shipped that way in the 16-block set (issue
#   #113: mean KLD 0.032 on gfx1201, 0.62 on gfx1100) and passed every existing
#   gate.  This is the instrument that sees it.
#
# MODES:
#   (default)        compare the current build against $BASE (must already exist)
#   --record         (re)record $BASE from the current build, then compare it to
#                    itself (a self-test of the harness; mean KLD ~ 0)
#   --ab "K=V ..."   exact-fallback A/B: record $BASE with the given env applied
#                    (the known-good/fallback arm), then compare the *default*
#                    arm against it.  This is the check that catches issue #113:
#                    --ab "GGML_CUDA_GDN_CHUNKED_BF16=0" on a build whose default
#                    is the bf16 kernel.
#   --family-switch  judge a *kernel-family* switch (a different reduction order,
#                    e.g. tile -> WMMA FA rows) rather than a change inside one
#                    family.  See "FAMILY-SWITCH MODE" below: the point must be
#                    in the gate's calibrated domain and the result is reported
#                    against a same-kernel reordering control measured on the
#                    spot, not only against the absolute limits.
#
# FAMILY-SWITCH MODE (--family-switch):
#   The absolute limits decide a change *inside* a kernel family (a quant type, a
#   precision, an approximate path).  A family switch does not share a reduction
#   order with the kernel it replaces, so its divergence is the tuning, not a
#   defect -- but on a point the model is not confident about, EVERY reordering
#   (including a benign ubatch change) measures far outside the limits, so such a
#   point cannot decide the question at all (GREEDY-PURITY.md §46: a same-kernel
#   -ub 512 vs -ub 256 change measures 0.105 KLD / 85.0 % top-p at the wikitext
#   point and 0.0140 / 97.2 % at a confident one).  This mode therefore:
#     * refuses a point whose base PPL is outside the calibrated domain
#       (--family-max-ppl, default 10; the gate was pinned at base PPL 5.93);
#     * records, next to the fallback base, a CONTROL base built with the same
#       fallback arm at a different ubatch (--control-ub, default 256);
#     * measures the candidate and the control and prints both;
#     * PASSES if the candidate meets the absolute limits, or if it is no worse
#       than the control on both metrics (the §46 calibration).
#   Usage:  --record --ab "<fallback env>" --family-switch ...  (records both bases)
#           --family-switch ...                                  (compares)
#
# Usage:
#   ./gate-prefill-logits.sh [options]
#
# Options / environment (the environment variable is the default for the flag):
#   --base FILE       BASE           base logits file (default:
#                                    $BASE_DIR/prefill-logit-<arch>-<modelstem>.kld)
#   --base-dir DIR    BASE_DIR       default base directory
#                                    (default: ~/.cache/rdna-boosts/gates)
#   --model FILE      MODEL          model gguf (default: auto: Q8_0 if the largest
#                                    single card has >= 28 GiB, else the first
#                                    available 24-GiB-class quant)
#   --corpus FILE     CORPUS         default /llm/models/wikitext-2-raw/wiki.test.raw
#   --ppl FILE        LLAMA_PPL      default ~/llama.cpp/build-rocm/bin/llama-perplexity (the canonical
#                                    ~/bin/build-llama-rocm tree)
#   --ctx N           CTX            default 512
#   --chunks N        CHUNKS         default 40
#   --ub N            UB             pass `-ub N` (default: the tool's own ubatch, i.e. moot)
#                                    A non-default ubatch gets its own base file, because the
#                                    captured-graph band (`-ub 8`/`-ub 1`) is where a
#                                    scheduler/upload defect shows up and its logits differ.
#   --ngl N           NGL            default 99
#   --fa on|off|auto  FA             default on
#   --kv TYPE         KV             default f16.  The KV cache type, passed as `-ctk TYPE -ctv TYPE`.
#                                    `f16` is the plain reference arm, but it *skips* the
#                                    quantized-cache prefill paths entirely (no F16 staging arena,
#                                    no native q8_0/bf16 tile read in the FA kernels), so any change
#                                    to those is invisible at f16 -- judge it with `--kv q8_0`,
#                                    which gets its own base file and a meta pin.
#   --threads N       THREADS        default 8
#   --max-kld F       MAX_KLD        default 0.005   (fail if mean KLD > F)
#   --min-top-p F     MIN_TOPP       default 98.0    (fail if same-top-p < F)
#   --family-switch   FAMILY_SWITCH  default 0       (see FAMILY-SWITCH MODE)
#   --family-max-ppl F FAMILY_MAX_PPL default 10     (refuse a family-switch point
#                                    whose base PPL is above this)
#   --control-ub N    CONTROL_UB     default 256     (ubatch of the control base;
#                                    must differ from the default ubatch)
#   --candidate-env S CAND_ENV       env for the candidate arm (default: none, i.e. the
#                                    build's own default).  Use it to judge one switch of
#                                    a set, e.g. --candidate-env
#                                    "GGML_CUDA_FA_WMMA_256=0" for "both off",
#                                    "GGML_CUDA_FA_RDNA3_WMMA_512=0" for the head-256 rows
#                                    alone against a fully-tile base.
#   --allow-control-pass  ALLOW_CONTROL_PASS default 0.  By default the ABSOLUTE limits are
#                                    the criterion and the control is evidence; with this
#                                    flag a candidate that is no worse than the control
#                                    passes even if the limits are missed (the §46
#                                    calibration, for a switch that provably cannot meet
#                                    them).  Off by default so a bad change cannot pass
#                                    just because the point is insensitive.
#   --device LIST     DEVICE         HIP_VISIBLE_DEVICES (default 0; a list selects a multi-device arm)
#   --sm MODE         SM             pass `-sm MODE` to llama-perplexity (default: unset = the CLI
#                                    default, `layer`).  Needed for anything that changes the
#                                    ALL-REDUCE: a layer split needs none, so an AR change is invisible
#                                    to this gate unless the arm is `--sm tensor` on >= 2 devices.
#   --timeout N       TIMEOUT        per-run seconds (default 1800)
#   --keep            KEEP=1         keep the per-run logs
#   --no-faults       FAULTS=0       skip the kernel-journal fault check (a box with no readable
#                                    kernel log).  By default the journal is checked over this gate's
#                                    own window, i.e. from a timestamp taken before the first run (#294)
#   --extra "ARGS"    PPL_EXTRA      extra llama-perplexity arguments, word-split (default: none).
#                                    Needed to gate a configuration that is not the all-GPU default --
#                                    e.g. a host-expert arm: --extra "-ncmoe 40 --host-experts pool".
#                                    Folded into the base file name, because a base recorded on one
#                                    configuration is not a reference for another.
#   --allow-base-mismatch            skip the base-meta compatibility check
#   --record, --ab SPEC
#
# Exit: 0 = PASS, 1 = FAIL (gate), 2 = usage/prerequisite error.
#
# This is a pre-release gate, not CI: it needs a ROCm GPU, the pinned model and a
# multi-GB base file.  Record the base from the previous release's build once, then
# run the candidate (default, and --ab for any new approximate kernel) before
# tagging.  Record the result in the release's WORKLOG.md entry.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_PPL="${LLAMA_PPL:-$HOME/llama.cpp/build-rocm/bin/llama-perplexity}"
CORPUS="${CORPUS:-/llm/models/wikitext-2-raw/wiki.test.raw}"
BASE_DIR="${BASE_DIR:-$HOME/.cache/rdna-boosts/gates}"
CTX="${CTX:-512}"
CHUNKS="${CHUNKS:-40}"
UB="${UB:-}"
NGL="${NGL:-99}"
FA="${FA:-on}"
KV="${KV:-f16}"
THREADS="${THREADS:-8}"
MAX_KLD="${MAX_KLD:-0.005}"
MIN_TOPP="${MIN_TOPP:-98.0}"
FAMILY_SWITCH="${FAMILY_SWITCH:-0}"
FAMILY_MAX_PPL="${FAMILY_MAX_PPL:-10}"
CONTROL_UB="${CONTROL_UB:-256}"
CAND_ENV="${CAND_ENV:-}"
ALLOW_CONTROL_PASS="${ALLOW_CONTROL_PASS:-0}"
DEVICE="${DEVICE:-0}"
SM="${SM:-}"
PPL_EXTRA="${PPL_EXTRA:-}"
TIMEOUT="${TIMEOUT:-1800}"
KEEP="${KEEP:-0}"
FAULTS="${FAULTS:-1}"
ALLOW_BASE_MISMATCH=0

RECORD=0
RECORD_ENV=""
BASE="${BASE:-}"
MODEL="${MODEL:-}"
MODEL_GIVEN=0

usage() { sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --base)                 BASE="$2"; shift 2 ;;
        --base-dir)             BASE_DIR="$2"; shift 2 ;;
        --model)                MODEL="$2"; MODEL_GIVEN=1; shift 2 ;;
        --corpus)               CORPUS="$2"; shift 2 ;;
        --ppl)                  LLAMA_PPL="$2"; shift 2 ;;
        --ctx)                  CTX="$2"; shift 2 ;;
        --chunks)               CHUNKS="$2"; shift 2 ;;
        --ub)                   UB="$2"; shift 2 ;;
        --ngl)                  NGL="$2"; shift 2 ;;
        --fa)                   FA="$2"; shift 2 ;;
        --kv)                   KV="$2"; shift 2 ;;
        --threads)              THREADS="$2"; shift 2 ;;
        --max-kld)              MAX_KLD="$2"; shift 2 ;;
        --min-top-p)            MIN_TOPP="$2"; shift 2 ;;
        --family-switch)        FAMILY_SWITCH=1; shift ;;
        --family-max-ppl)       FAMILY_MAX_PPL="$2"; shift 2 ;;
        --control-ub)           CONTROL_UB="$2"; shift 2 ;;
        --candidate-env)        CAND_ENV="$2"; shift 2 ;;
        --allow-control-pass)   ALLOW_CONTROL_PASS=1; shift ;;
        --device)               DEVICE="$2"; shift 2 ;;
        --sm)                   SM="$2"; shift 2 ;;
        --timeout)              TIMEOUT="$2"; shift 2 ;;
        --keep)                 KEEP=1; shift ;;
        --no-faults)            FAULTS=0; shift ;;
        --allow-base-mismatch)  ALLOW_BASE_MISMATCH=1; shift ;;
        --extra)                PPL_EXTRA="$2"; shift 2 ;;
        --record)               RECORD=1; RECORD_ENV=""; shift ;;
        --ab)                   RECORD=1; RECORD_ENV="${2:-}"; shift 2 ;;
        -h|--help)              usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

fail() { echo "FAIL: $*" >&2; KEEP=1; exit 1; }
die()  { echo "ERROR: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# prerequisites
# ---------------------------------------------------------------------------
[ -x "$LLAMA_PPL" ] || die "llama-perplexity not executable: $LLAMA_PPL (set LLAMA_PPL)"
[ -f "$CORPUS" ]    || die "corpus not found: $CORPUS"
command -v awk >/dev/null || die "awk required"

ARCH="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
ARCH="${ARCH:-gpu}"

vram_gib() {
    local best=0 f b gib
    for f in /sys/class/drm/card*/device/mem_info_vram_total; do
        [ -r "$f" ] || continue
        b="$(cat "$f" 2>/dev/null)" || continue
        case "$b" in ''|*[!0-9]*) continue ;; esac
        gib=$((b / 1073741824))
        [ "$gib" -gt "$best" ] && best="$gib"
    done
    echo "$best"
}

# Auto model: Q8_0 when a single card can hold it, else the first available
# 24-GiB-class 27B quant (the same lineage the methodology pins for gfx1100).
resolve_model() {
    local q8=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
    local cands=(
        /llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf
        /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
        /llm/models/Qwen3.8/27B/Q5_K/Qwen3.8-27B-UD-Q5_K_M.gguf
        /llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf
    )
    if [ -n "$MODEL" ]; then return 0; fi
    local gib; gib="$(vram_gib)"
    if [ "$gib" -ge 28 ] && [ -f "$q8" ]; then
        MODEL="$q8"
        return 0
    fi
    local c
    for c in "${cands[@]}"; do
        if [ -f "$c" ]; then MODEL="$c"; return 0; fi
    done
    die "no auto model found (set MODEL); biggest single card = ${gib} GiB"
}
resolve_model
[ "$FAMILY_SWITCH" -eq 1 ] && [ "$MODEL_GIVEN" -eq 0 ] && die "family-switch mode needs an explicit --model
       the auto-chosen model has no head-256/512 attention, so it cannot exercise the
       rows the switch is about; use e.g.
         --model /llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
[ -f "$MODEL" ] || die "model not found: $MODEL"

MODEL_STEM="$(basename "$MODEL" .gguf)"
if [ -z "$BASE" ]; then
    mkdir -p "$BASE_DIR" || die "cannot create base dir: $BASE_DIR"
    # A family-switch judgement uses its own in-domain corpus/context, so it gets its own
    # base name; the default (wikitext, 512-token chunk) base is a different point.
    fs_tag=""
    [ "$FAMILY_SWITCH" -eq 1 ] && fs_tag="-familyswitch"
    # A quantized KV cache reaches the prefill paths an f16 cache never builds (the F16 staging
    # arena, the native tile read), so it is a different point and gets its own base file.
    kv_tag=""
    [ "$KV" != "f16" ] && kv_tag="-kv${KV}"
    # A non-default configuration (a host-expert arm, a split mode) is its own point.
    xt_tag=""
    if [ -n "$PPL_EXTRA" ]; then
        xt_tag="-x$(printf '%s' "$PPL_EXTRA" | tr -cs 'A-Za-z0-9.' '-' | sed 's/^-*//;s/-*$//')"
    fi
    BASE="$BASE_DIR/prefill-logit-${ARCH}-${MODEL_STEM}${fs_tag}${kv_tag}${xt_tag}${UB:+-ub${UB}}.kld"
fi
META="$BASE.meta"
CONTROL_BASE="${BASE%.kld}-control-ub${CONTROL_UB}.kld"
CONTROL_META="$CONTROL_BASE.meta"

CORPUS_SHA="$(sha256sum "$CORPUS" | awk '{print $1}')"

if [ "$RECORD" -eq 0 ] && [ ! -f "$BASE" ]; then
    die "base not found: $BASE
       record it first from a known-good build with: $0 --record
       (or point --base at an existing file)"
fi

LOGDIR="${LOGDIR:-$(mktemp -d)}"
mkdir -p "$LOGDIR"
# The kernel journal is shared between runs, so the fault window starts here, before the first run (#294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$LOGDIR"; }
trap cleanup EXIT

echo "==> prefill-logit KLD gate (issue #113)"
echo "    ppl=$LLAMA_PPL"
echo "    model=$MODEL"
echo "    corpus=$CORPUS  (sha256=${CORPUS_SHA:0:16}...)"
echo "    arch=$ARCH  device=$DEVICE  ctx=$CTX chunks=$CHUNKS fa=$FA kv=$KV ngl=$NGL threads=$THREADS${PPL_EXTRA:+  extra='$PPL_EXTRA'}"
echo "    base=$BASE$( [ -f "$META" ] && { r="$(sed -n 's/^rocm=//p' "$META" | head -1)"; [ -n "$r" ] && printf '  (recorded with ROCm %s)' "$r"; } )"
echo "    thresholds: mean KLD <= $MAX_KLD, same-top-p >= ${MIN_TOPP}%"
[ "$FAMILY_SWITCH" -eq 1 ] && echo "    mode=family-switch (control ubatch=$CONTROL_UB, max point PPL=$FAMILY_MAX_PPL)"
[ "$RECORD" -eq 1 ] && echo "    mode=record${RECORD_ENV:+ (base arm: $RECORD_ENV)}"
[ -n "$RECORD_ENV" ] && echo "    --ab: base arm = '$RECORD_ENV', candidate arm = default"
echo "    logs=$LOGDIR"

# ---------------------------------------------------------------------------
# run helper
# ---------------------------------------------------------------------------
run_ppl() {
    local extra="$1" out="$2"; shift 2
    local ub_args=()
    [ -n "$UB" ] && ub_args=(-ub "$UB" -b "$UB")
    local sm_args=()
    [ -n "$SM" ] && sm_args=(-sm "$SM")
    # shellcheck disable=SC2086  # $extra is an intended word-split K=V list
    local xt_args=()
    # shellcheck disable=SC2206  # an intended word-split argument list
    [ -n "$PPL_EXTRA" ] && xt_args=($PPL_EXTRA)
    timeout "$TIMEOUT" env HIP_VISIBLE_DEVICES="$DEVICE" ${extra:-} \
        "$LLAMA_PPL" -m "$MODEL" -f "$CORPUS" \
        -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" -fa "$FA" -ctk "$KV" -ctv "$KV" -t "$THREADS" "${ub_args[@]}" "${sm_args[@]}" \
        "${xt_args[@]}" "$@" >"$out" 2>&1
}

mean_kld()  { sed -n 's/^Mean[[:space:]]*KLD:[[:space:]]*\([-0-9.][0-9.]*\).*/\1/p' "$1" | head -1; }
same_topp() { sed -n 's/^Same top p:[[:space:]]*\([0-9.][0-9.]*\).*/\1/p' "$1" | head -1; }
final_ppl() { sed -n 's/.*Final estimate: *PPL *= *\([0-9.][0-9.]*\).*/\1/p' "$1" | head -1; }
# A build reference that identifies the BINARY, not the checkout beside it.
#   `llama-perplexity --version` prints `version: ... (build N, commit <sha>)`, so ask the binary first;
#   falling back to the enclosing checkout's HEAD is wrong whenever a build dir was configured from a
#   different commit than the one checked out now (r138 hit exactly that: the gate labelled an r138 run
#   `e8bd52069` because the tree had been switched to build r137 afterwards).  Last resort: the binary's
#   hash, which is always correct but not human-readable.
build_ref() {
    local dir ver commit
    # --version goes to stderr (it is a GGML log line), so capture 2>&1.
    if ver="$(cd "$(dirname "$LLAMA_PPL")" 2>/dev/null && timeout 60 ./"$(basename "$LLAMA_PPL")" --version 2>&1)"; then
        commit="$(printf '%s\n' "$ver" | sed -n 's/.*commit \([0-9a-f]\{7,\}\).*/\1/p' | head -1)"
        if [ -n "$commit" ]; then
            printf '%s\n' "$commit"
            return
        fi
    fi
    dir="$(cd "$(dirname "$LLAMA_PPL")/../.." 2>/dev/null && pwd)"
    if [ -n "$dir" ] && git -C "$dir" rev-parse --short=12 HEAD >/dev/null 2>&1; then
        git -C "$dir" rev-parse --short=12 HEAD
    else
        sha256sum "$LLAMA_PPL" | awk '{print substr($1,1,12)}'
    fi
}
# The ROCm line the BINARY links (not the one on PATH, and not the checkout's CMakeCache): a base
# recorded on one line is not a rounding-level reference for another.  Measured 2026-10-31 at the
# 35B-A3B point: the same r191 tree reads 0.005657 / 96.794 % on 10.1.0 and 0.005688 / 96.765 % on
# 7.14.1 -- a different compiler moves the low bits of every prefill kernel, so a cross-line compare
# reports a "divergence" that is the toolchain (TODO #126: the 35B base was an r125-era recording of
# the pre-r126 FA prefill row, and it read as a gate failure for six releases).
rocm_line() {
    local p
    p="$(ldd "$LLAMA_PPL" 2>/dev/null | sed -n 's#.*=> \(/opt/rocm[^/]*\)/.*#\1#p' | head -1)"
    if [ -z "$p" ] && command -v readelf >/dev/null 2>&1; then
        p="$(readelf -d "$LLAMA_PPL" 2>/dev/null | sed -n 's#.*\[\(/opt/rocm[^]:]*\)[:/].*#\1#p' | head -1)"
    fi
    printf '%s\n' "${p##*/}"
}
le() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 <= b+0) }'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 >= b+0) }'; }

# ---------------------------------------------------------------------------
# base compatibility (model/corpus/ctx/chunks must match to be a valid compare)
# ---------------------------------------------------------------------------
base_meta_field() { [ -f "$META" ] && sed -n "s/^$1=//p" "$META" | head -1 || true; }

check_base_compat() {
    [ -f "$META" ] || { echo "    WARN: no base metadata ($META); cannot verify the base matches this model/corpus"; return 0; }
    local problems=0 k v want
    for k in model corpus_sha256 ctx chunks kv; do
        v="$(base_meta_field "$k")"
        case "$k" in
            model)         want="$MODEL" ;;
            corpus_sha256) want="$CORPUS_SHA" ;;
            ctx)           want="$CTX" ;;
            chunks)        want="$CHUNKS" ;;
            kv)            want="$KV" ;;
        esac
        if [ -n "$v" ] && [ "$v" != "$want" ]; then
            echo "    base meta mismatch: $k='$v' != current '$want'" >&2
            problems=1
        fi
    done
    if [ "$problems" -eq 1 ]; then
        if [ "$ALLOW_BASE_MISMATCH" -eq 1 ]; then
            echo "    WARN: base does not match the current model/corpus/settings (--allow-base-mismatch)"
        else
            fail "base $BASE does not match the current model/corpus/settings; re-record it (--record) or pass --allow-base-mismatch"
        fi
    fi

    # the TOOLCHAIN the base was recorded with: a legacy base has no `rocm=` field and is left alone
    local b_rocm c_rocm
    b_rocm="$(base_meta_field rocm)"
    c_rocm="$(rocm_line)"
    if [ -n "$b_rocm" ] && [ -n "$c_rocm" ] && [ "$b_rocm" != "$c_rocm" ]; then
        if [ "$ALLOW_BASE_MISMATCH" -eq 1 ]; then
            echo "    WARN: base was recorded with $b_rocm, this build links $c_rocm (--allow-base-mismatch)"
        else
            fail "base $BASE was recorded with ROCm '$b_rocm' but this build links '$c_rocm'
       a cross-toolchain compare is not rounding-level (the low bits of every prefill kernel move);
       record one base per ROCm line (--record), or pass --allow-base-mismatch to see the number anyway"
        fi
    fi
}

# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
if [ "$RECORD" -eq 1 ]; then
    rec_log="$LOGDIR/base-record.log"
    echo
    echo "==> recording base with arm '${RECORD_ENV:-<default>}' (may take a few minutes)"
    [ -f "$BASE" ] && echo "    WARN: overwriting existing base $BASE"
    if ! run_ppl "$RECORD_ENV" "$rec_log" --kl-divergence-base "$BASE"; then
        echo "--- tail of $rec_log ---" >&2; tail -20 "$rec_log" >&2
        fail "base recording run failed"
    fi
    [ -f "$BASE" ] || fail "base file was not written: $BASE"
    grep -q "Final estimate" "$rec_log" || { tail -20 "$rec_log" >&2; fail "base recording did not complete (no 'Final estimate')"; }

    b_ppl="$(final_ppl "$rec_log")"
    b_ref="$(build_ref)"
    b_sha=""
    if [ "${SKIP_BASE_HASH:-0}" != "1" ]; then
        b_sha="$(sha256sum "$BASE" | awk '{print $1}')"
    fi
    {
        echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host=$(hostname)"
        echo "arch=$ARCH"
        echo "rocm=$(rocm_line)"
        echo "build=$b_ref"
        echo "model=$MODEL"
        echo "corpus=$CORPUS"
        echo "corpus_sha256=$CORPUS_SHA"
        echo "ctx=$CTX"
        echo "chunks=$CHUNKS"
        echo "ngl=$NGL"
        echo "fa=$FA"
        echo "kv=$KV"
        echo "arm=${RECORD_ENV:-<default>}"
        echo "base_ppl=$b_ppl"
        echo "base_sha256=$b_sha"
    } > "$META"
    echo "    base PPL=$b_ppl build=${b_ref:-?} sha256=${b_sha:-<skipped>}"
    echo "    wrote $META"

    if [ "$FAMILY_SWITCH" -eq 1 ]; then
        if ge "$b_ppl" "$FAMILY_MAX_PPL"; then
            fail "family-switch point is out of the gate's calibrated domain: base PPL $b_ppl > $FAMILY_MAX_PPL
       a family switch cannot be decided at a point where the model is not confident
       (at such a point even a benign same-kernel ubatch change is far outside the limits);
       pass --corpus with in-domain text, or --family-max-ppl to override deliberately."
        fi
        ctl_log="$LOGDIR/control-record.log"
        echo "==> recording the same-kernel control base (arm '${RECORD_ENV:-<default>}', -ub $CONTROL_UB)"
        if ! ( UB="$CONTROL_UB"; run_ppl "$RECORD_ENV" "$ctl_log" --kl-divergence-base "$CONTROL_BASE" ); then
            echo "--- tail of $ctl_log ---" >&2; tail -20 "$ctl_log" >&2
            fail "control base recording run failed"
        fi
        [ -f "$CONTROL_BASE" ] || fail "control base file was not written: $CONTROL_BASE"
        c_ppl="$(final_ppl "$ctl_log")"
        c_sha=""
        [ "${SKIP_BASE_HASH:-0}" != "1" ] && c_sha="$(sha256sum "$CONTROL_BASE" | awk '{print $1}')"
        {
            echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "host=$(hostname)"
            echo "arch=$ARCH"
            echo "build=$b_ref"
            echo "model=$MODEL"
            echo "corpus=$CORPUS"
            echo "corpus_sha256=$CORPUS_SHA"
            echo "ctx=$CTX"
            echo "chunks=$CHUNKS"
            echo "ngl=$NGL"
            echo "fa=$FA"
            echo "kv=$KV"
            echo "ub=$CONTROL_UB"
            echo "arm=${RECORD_ENV:-<default>}"
            echo "base_ppl=$c_ppl"
            echo "base_sha256=$c_sha"
        } > "$CONTROL_META"
        echo "    control base PPL=$c_ppl (same kernel, -ub $CONTROL_UB); wrote $CONTROL_META"
    fi
    # If we recorded with an extra env arm, comparing the default against it is
    # the whole point; if we recorded the default, the compare is a self-test.
fi

if [ "$RECORD" -eq 0 ]; then
    check_base_compat
fi

# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
cmp_log="$LOGDIR/candidate-compare.log"
echo
echo "==> comparing candidate (${CAND_ENV:-default}${RECORD_ENV:+ vs base arm '$RECORD_ENV'})"
if ! run_ppl "$CAND_ENV" "$cmp_log" --kl-divergence-base "$BASE" --kl-divergence; then
    echo "--- tail of $cmp_log ---" >&2; tail -20 "$cmp_log" >&2
    fail "candidate run failed (see $cmp_log)"
fi
grep -qE '^Mean[[:space:]]*KLD:' "$cmp_log" || { tail -20 "$cmp_log" >&2; fail "candidate run did not complete (no mean KLD)"; }

KLD="$(mean_kld "$cmp_log")"
TOPP="$(same_topp "$cmp_log")"
CAND_PPL="$(final_ppl "$cmp_log")"
CAND_REF="$(build_ref)"
[ -n "$KLD" ]  || { tail -30 "$cmp_log" >&2; fail "could not parse mean KLD from $cmp_log"; }
[ -n "$TOPP" ] || { tail -30 "$cmp_log" >&2; fail "could not parse same-top-p from $cmp_log"; }

echo
[ -n "$CAND_PPL" ] && ppl_note=" PPL=$CAND_PPL" || ppl_note=""
echo "    candidate: build=${CAND_REF:-?}${ppl_note}"
echo "    mean KLD = $KLD   (limit $MAX_KLD)"
echo "    same top p = $TOPP %   (floor $MIN_TOPP %)"

rc=0
if ! le "$KLD" "$MAX_KLD"; then
    echo "FAIL: mean KLD $KLD > $MAX_KLD" >&2; rc=1
fi
if ! ge "$TOPP" "$MIN_TOPP"; then
    echo "FAIL: same top p $TOPP % < $MIN_TOPP %" >&2; rc=1
fi

if [ "$FAMILY_SWITCH" -eq 1 ]; then
    # The absolute limits decide a within-family change.  A family switch is judged
    # against a *same-kernel reordering* measured on the spot: at any point the model
    # finds hard, that control is the floor of what any reordering costs, so a
    # candidate that is no worse than it carries no evidence of a defect.  Both are
    # always printed; the absolute limits stay authoritative when they are met.
    base_arm="$(base_meta_field arm)"
    [ "$base_arm" = "<default>" ] && base_arm=""
    if [ ! -f "$CONTROL_BASE" ]; then
        fail "family-switch control base not found: $CONTROL_BASE
       record it first: $0 --record --family-switch --ab \"<fallback env>\" ..."
    fi
    point_ppl="$(base_meta_field base_ppl)"
    family_rc=0
    if [ -n "$point_ppl" ] && ge "$point_ppl" "$FAMILY_MAX_PPL"; then
        echo "FAIL: family-switch point is out of the calibrated domain: base PPL $point_ppl > $FAMILY_MAX_PPL" >&2
        family_rc=1
    fi
    ctl_log2="$LOGDIR/control-compare.log"
    echo
    echo "==> control: same kernel (arm '${base_arm:-<default>}'), ${CTX}-token chunks vs -ub $CONTROL_UB"
    if ! run_ppl "$base_arm" "$ctl_log2" --kl-divergence-base "$CONTROL_BASE" --kl-divergence; then
        echo "--- tail of $ctl_log2 ---" >&2; tail -20 "$ctl_log2" >&2
        fail "control comparison run failed"
    fi
    C_KLD="$(mean_kld "$ctl_log2")"
    C_TOPP="$(same_topp "$ctl_log2")"
    [ -n "$C_KLD" ] && [ -n "$C_TOPP" ] || fail "could not parse the control result"
    echo "    control (same-kernel reordering): mean KLD = $C_KLD   same top p = $C_TOPP %"
    if [ "$rc" -ne 0 ]; then
        # Absolute limits missed.  The control says whether that is the candidate's fault or
        # the point's; it only decides the verdict when the caller opts in.
        if le "$KLD" "$C_KLD" && ge "$TOPP" "$C_TOPP"; then
            if [ "$ALLOW_CONTROL_PASS" -eq 1 ]; then
                echo "    NOTE: absolute limits missed, but the candidate is no worse than the"
                echo "          same-kernel reordering control on both metrics (GREEDY-PURITY.md §46)."
                rc=0
            else
                echo "    NOTE: the candidate is no worse than the same-kernel control, so the excess" >&2
                echo "          is the point's, not the switch's — but the absolute limits are the" >&2
                echo "          criterion.  Re-run with --allow-control-pass to accept the calibration." >&2
            fi
        fi
    fi
    rc=$(( rc | family_rc ))
fi

echo
# the kernel journal, scoped to this gate's own window (#294).  A prefill-logit gate that ran on a card that
# faulted earlier in the session is a verdict about the journal, not about the logits.
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$LOGDIR"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        echo "    GPU faults: kernel journal (since ${GATE_SINCE_TS:-start}) and every run log clean"
    else
        echo "FAIL: a GPU fault was recorded during this gate (since ${GATE_SINCE_TS:-the journal start}):" >&2
        echo "$out" | sed 's/^/      /' >&2
        rc=1
        KEEP=1
    fi
fi

echo
if [ "$rc" -eq 0 ]; then
    echo "==> PASS: prefill logits within rounding tolerance of the base"
    echo "    base=$BASE"
    [ "$KEEP" -eq 1 ] && echo "    logs kept in $LOGDIR"
    exit 0
fi
echo "==> FAIL: prefill-logit divergence (issue #113 class)" >&2
echo "    logs in $LOGDIR (set KEEP=1 to retain on success too)" >&2
KEEP=1
exit 1
