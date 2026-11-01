#!/usr/bin/env bash
# GPU validation gate -- the heavy per-push gate that no hosted runner can run.
#
# WHY THIS EXISTS
#   The repo's gating has three tiers, and only two could run anywhere:
#     * tier 1  `.github/workflows/validate.yml`  -- apply + tree/marker checks, ~20 s, no GPU;
#     * tier 2  THIS SCRIPT on the off-host CI worker (radiance, gfx1151) -- apply, the
#               per-block `--fast` ladder, the op-test oracles and the prefill-logit KLD gate;
#     * tier 3  the release flow -- the `all` build, coherence, MTP Protocol A, arch sweeps.
#   Tier 2 is the one that actually catches a bad patch set, because almost every delivery
#   block is a CUDA/HIP kernel change and a hosted runner has no ROCm.  It is the automation
#   behind "a re-base is a decode AND prefill change" (INVARIANTS.md): it re-applies the set from
#   scratch on the box, so a patch that no longer applies, a block whose tree does not build,
#   a broken op and a shifted prefill all fail here, unattended.
#
# WHAT IT DOES (each step is skippable; see the env list)
#   1. materialises the set: delivery repo checked out at REV, `apply-all.sh` in LLAMA_REPO,
#      applied tree == that rev's `release.json.tree` (strict `git am`);
#   2. `gate-per-block-build.sh --fast` over every block commit (the ladder);
#   3. builds the tip's test-binary and runs the op oracles (`FLASH_ATTN_EXT`,
#      `GATED_DELTA_NET`, `FLASH_ATTN_QSA`, `MUL_MAT`, `MUL_MAT_ID`, `TOP_K_MASK`, `HC_MIX`,
#      `HC_COMBINE`, `HC_COMBINE_MIX`);
#   4. `gate-prefill-logits.sh` on the tip -- compares against this box's recorded base, or
#      RECORDS one on the first run (a fresh worker box has no base; the gate then exits with
#      the recorded result, and later runs compare);
#   4b. `gate-hc-fusions.sh` on the qwen4exp GGUF(s) present here -- the four mix/combine arms
#      must all reproduce the unfused chain (issue #228: the op-vs-CPU oracle cannot see that
#      class).  Skipped when no qwen4exp GGUF is present or the applied tree touched no HC path;
#   4c. `test-logits-width-probe` at the VERIFY widths against the r183 GDN direct-conv folds --
#      those folds reorder a multi-token step's launches without changing its arithmetic, so
#      greedy text and the `-ub 1` prefill-logit arm are both blind to them (r80 + PR #177 was
#      run-to-run unstable at P=3584 with clean text and KLD 0.000000).  Fails unless the default
#      arm is run-to-run stable, bit-identical to its all-folds-off arm, and really reaches the
#      fold (`GGML_CUDA_CONV_DEBUG=1` -> `snap=1`) and the reference arm must really have taken
#      CONCAT+SSM_CONV+CPY.  Runs one single-device arm and, on a 2+ GPU box, `-sm tensor` and
#      `-sm layer` arms (r186 made the fusion device-count independent, so those are the ones that
#      can catch a regression).  Skipped when no GDN GGUF is present or the tree touched no
#      conv-fold source;
#   5. writes a one-screen summary to $STATE_DIR/runs/<rev>/summary.txt and, on success,
#      records $STATE_DIR/last-ok.
#
# USAGE
#   validate-gpu.sh [REV]              REV defaults to origin/main
#   validate-gpu.sh --force [REV]      re-run even if REV is the last success
#
# ENV  REPO (delivery checkout)          default the repo this script lives in
#      LLAMA_REPO (long-lived checkout)  default $HOME/llama-ci   <- keep ONE path (ccache keys on it)
#      STATE_DIR                         default $HOME/.cache/rdna-validate
#      JOBS                              default nproc
#      ROCM_DIR / GPU_TARGETS            auto-detected from /opt/rocm-10.1.0-gfxNNNN
#      KLD_MODEL, KLD_CTX (512), KLD_CHUNKS (40), KLD_BASE_DIR
#      HC_MODELS                     space-separated qwen4exp GGUFs for the fusion gate (both
#                                    hc-down classes); HC_ALWAYS=1 forces it
#      GDN_WIDTH_MODELS              space-separated GDN/hybrid GGUFs for the conv-fold width gate
#                                    (a missing path is skipped); GDN_WIDTH_P ("3584 4096"),
#                                    GDN_WIDTH_RUNS (3), GDN_WIDTH_TEXT, GDN_WIDTH_ALWAYS=1
#      SKIP_LADDER=1 / SKIP_OPTESTS=1 / SKIP_KLD=1 / SKIP_HC=1 / SKIP_GDN_WIDTH=1   narrow the run
#      MAX_ATTEMPTS (3, per rev)         KEEP_RUNS (5)
#      KLD_ALWAYS=1                      run the prefill-logit gate even when the applied tree
#                                        did not touch a prefill path (see below)
#
# COST (measured on radiance, gfx1151 APU, 16 threads, 2026-10-25)
#   cold ccache, whole chain        87m37   <- ONE-TIME; the FA instantiation sets dominate: the
#                                            ladder was 4882 s and blocks 05-12 each compile their
#                                            own set cold (520-610 s per block; 00-04 are 10-114 s)
#   warm ladder, whole chain        89 s
#   unchanged delivery content      0.3 s   skipped on base:tree:n_blocks, not on the commit
#   warm full run (patch set same)  5m36    ladder 6 s (only the tip) + ~5 min op oracles
#   prefill-logit, when relevant    ~6 min  a 9B Q8_0 base at ctx 512 x 40 chunks
#   => a release-shaped push costs ~6 min, ~12 min if it moves a prefill kernel, and a docs or
#      benchmarks push costs 0.3 s.  The cold figure is bring-up only: do not plan around it.
#      CI_LOCK (the container watcher's lock)  default $HOME/.cache/rdna-containers/.lock
#
# Exit: 0 = validated (or nothing to do / already given up on this rev); 1 = a step failed.
set -uo pipefail

export PATH="$HOME/.local/bin:$HOME/bin:/usr/local/bin:/usr/bin:/bin"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# Installed copies live OUTSIDE the checkout (the worker installs to ~/bin, like the container
# watcher), so "the directory above the script" is not the repo there: fall back to the
# conventional clone location and fail loudly rather than fetching in $HOME.
_default_repo() {
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
    if git -C "$here" rev-parse --git-dir >/dev/null 2>&1; then printf '%s' "$here"; return; fi
    if git -C "$HOME/llama-cpp-rdna-boosts" rev-parse --git-dir >/dev/null 2>&1; then
        printf '%s' "$HOME/llama-cpp-rdna-boosts"; return
    fi
    printf '%s' "$here"
}
REPO="${REPO:-$(_default_repo)}"
LLAMA_REPO="${LLAMA_REPO:-$HOME/llama-ci}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/rdna-validate}"
JOBS="${JOBS:-$(nproc)}"
SKIP_LADDER="${SKIP_LADDER:-0}"
SKIP_OPTESTS="${SKIP_OPTESTS:-0}"
SKIP_KLD="${SKIP_KLD:-0}"
SKIP_HC="${SKIP_HC:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
KEEP_RUNS="${KEEP_RUNS:-5}"
FORCE=0
REV=""
CI_LOCK="${CI_LOCK:-$HOME/.cache/rdna-containers/.lock}"

# The KLD arm is slow on an APU; keep the default point small and stable.
KLD_MODEL="${KLD_MODEL:-/llm/models/Qwen3.5/9B/Q8_0/Qwen3.5-9B-Q8_0.gguf}"
KLD_CTX="${KLD_CTX:-512}"
KLD_CHUNKS="${KLD_CHUNKS:-40}"
KLD_ALWAYS="${KLD_ALWAYS:-0}"
# qwen4exp GGUFs for the hyper-connection fusion gate (issue #228).  The gate's subject is the
# four mix/combine arcs, and the hc-down weight TYPE decides which mixer runs, so covering both
# classes needs one GGUF of each: a Q8_0/BF16 hc_attn_down one (the default below, the class the
# delivery's own qwen4exp records use) and any other quant (Q5_K/IQ4_NL/...) -- add it here, e.g.
#   HC_MODELS="/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-...-00001-of-00002.gguf \
#              /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf"
# A path that does not exist is skipped (each class is optional).
HC_MODELS="${HC_MODELS:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
HC_ALWAYS="${HC_ALWAYS:-0}"
SKIP_GDN_WIDTH="${SKIP_GDN_WIDTH:-0}"
# GDN/hybrid GGUFs for the conv-fold width gate (r183).  The subjects are the direct conv's three
# folds and the shared-expert gate fold, which only act on multi-token steps, so the probe has to
# run at the verify widths.  Both a MoE (35B-A3B) and a dense (27B) GDN model are wanted: the folds
# pick up on the dense shape too, and only the MoE one carries the routed-expert path.  A path that
# does not exist is skipped; a model that never reaches the fold FAILS (a vacuous pass is exactly
# the blind-probe failure this gate exists to prevent), so keep this list GDN/hybrid models only.
GDN_WIDTH_MODELS="${GDN_WIDTH_MODELS:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf}"
GDN_WIDTH_TEXT="${GDN_WIDTH_TEXT:-$REPO/prompts/prose-rdna-boosts.txt}"
GDN_WIDTH_P="${GDN_WIDTH_P:-3584 4096}"
GDN_WIDTH_RUNS="${GDN_WIDTH_RUNS:-3}"
GDN_WIDTH_ALWAYS="${GDN_WIDTH_ALWAYS:-0}"
# The arm list is "<HIP_VISIBLE_DEVICES>:<split>".  r186 made the conv fusion device-count independent, so the
# MULTI-device arms are the ones that can catch a regression; they are clamped to the GPUs actually present.
GW_SINGLE_ARM="0:layer"
GDN_WIDTH_ARMS="${GDN_WIDTH_ARMS:-}"

# Paths whose change makes the prefill-logit gate relevant (AGENTS: "any change to a prefill
# kernel ... or a default-on approximate path").
PREFILL_RE='^(ggml/src/ggml-cuda/|ggml/src/ggml\.c|src/llama-graph\.cpp|src/models/|src/llama-kv-cache)'
# A change to a qwen4exp hyper-connection op/chain/matcher (issue #228) or to the graph that emits it.
HC_RE='^(ggml/src/ggml-cuda/hc-mix|ggml/src/ggml-cuda/hyperconn|ggml/src/ggml-cuda/ggml-cuda-try-fuse|ggml/src/ggml-cpu/ops\.cpp|ggml/include/ggml\.h|ggml/src/ggml\.c|src/models/qwen4exp\.cpp)'
# A change to a GDN direct-conv fold, its matchers or the batched-copy run it stands down (r183).
# `ggml-cuda-try-fuse.cu` carries the dispatcher that emits the fold, `cpy-batch.cu` the run stop.
GDN_WIDTH_RE='^(ggml/src/ggml-cuda/gdn-conv|ggml/src/ggml-cuda/cpy-batch|ggml/src/ggml-cuda/ggml-cuda-try-fuse)'

log() { echo "$(date -Is) $*"; }
die() { log "ERROR: $*"; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *)  REV="$1"; shift ;;
    esac
done

for t in git jq rsync; do command -v "$t" >/dev/null || die "$t is required"; done
command -v ccache >/dev/null && : || log "WARN: ccache not found -- the ladder will be slow and cold"

# ROCm: pick the installed 10.1 tree and its target, so this works on any gfx box.
if [ -z "${ROCM_DIR:-}" ]; then
    ROCM_DIR="$(ls -d /opt/rocm-10.1.0-gfx* 2>/dev/null | head -1)"
    [ -n "$ROCM_DIR" ] || die "no /opt/rocm-10.1.0-gfx* install found (set ROCM_DIR)"
fi
[ -n "${GPU_TARGETS:-}" ] || GPU_TARGETS="$(basename "$ROCM_DIR" | sed 's/^.*-//')"

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/.lock"
if ! flock -n 9; then log "another validation run is in progress; skipping"; exit 0; fi

# Never contend with a release image build: that is the worker's other, rarer job and it is
# already slow on an APU.  The watcher holds this lock for the whole build+push.
if [ -e "$CI_LOCK" ] && command -v flock >/dev/null; then
    if ! flock -n "$CI_LOCK" true 2>/dev/null; then
        log "a container build is running; skipping this pass"; exit 0
    fi
fi

git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || die "not a git checkout: $REPO (set REPO=)"
cd "$REPO" || die "cannot enter $REPO"
git fetch --quiet --tags origin || die "git fetch failed in $REPO"

if [ -z "$REV" ]; then
    REV="$(git rev-parse origin/main)"; REV_LABEL="origin/main"
else
    REV="$(git rev-parse --verify -q "${REV}^{commit}")" || die "cannot resolve $REV"
    REV_LABEL="$REV"
fi
short="$(git rev-parse --short "$REV")"

# Idempotence is keyed on the delivery CONTENT (base + applied tree), not the commit: a docs or
# benchmarks push changes origin/main but not the patch set, so it must cost nothing.  The rev is
# recorded alongside for the summary.
CONTENT_KEY="$(git show "$REV:release.json" | jq -r '"\(.base):\(.tree):\(.n_blocks)"')"
last_key="$(cat "$STATE_DIR/last-ok" 2>/dev/null || true)"
last_rev="$(cat "$STATE_DIR/last-ok-rev" 2>/dev/null || true)"
if [ "$CONTENT_KEY" = "$last_key" ] && [ "$FORCE" -eq 0 ]; then
    log "up to date -- delivery content $CONTENT_KEY already validated (at ${last_rev:0:7}); nothing to do"
    exit 0
fi

fail_file="$STATE_DIR/failed-$short"
attempt=$(( $(cat "$fail_file" 2>/dev/null || echo 0) + 1 ))
if [ -f "$fail_file" ] && [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    log "ERROR: $short has failed $((attempt - 1)) times; not retrying (fix it, then remove $fail_file)"
    exit 0
fi

RUN="$STATE_DIR/runs/$short"
rm -rf "$RUN"; mkdir -p "$RUN"
SUMMARY="$RUN/summary.txt"
{
    echo "rdna-boosts GPU validation"
    echo "rev      : $short ($REV_LABEL)"
    echo "box      : $(hostname), $GPU_TARGETS, ROCm $(basename "$ROCM_DIR"), $(nproc) threads"
    echo "attempt  : $attempt/$MAX_ATTEMPTS"
    echo "started  : $(date -Is)"
    echo
} > "$SUMMARY"
say() { echo "$*" | tee -a "$SUMMARY"; }

fail() {
    echo "$attempt" > "$fail_file"
    say "RESULT   : FAIL -- $*"
    log "ERROR: $short failed (attempt $attempt/$MAX_ATTEMPTS): $*"
    log "summary: $SUMMARY"
    exit 1
}

# --- 1. materialise the set ------------------------------------------------------------------
# The delivery repo must be AT the rev under test before apply-all.sh runs, because apply-all
# reads release.json from its own checkout (a stale checkout would apply the wrong set).
git checkout --quiet --force --detach "$REV" || fail "cannot check out $short in $REPO"
[ "$(git rev-parse HEAD)" = "$REV" ] || fail "HEAD is not $short after checkout"
base="$(jq -r '.base'     release.json)"
tree="$(jq -r '.tree'     release.json)"
n_blocks="$(jq -r '.n_blocks' release.json)"
say "release  : $(jq -r '.release' release.json)  base=$base  blocks=$n_blocks"

# Keep the PREVIOUS applied chain as a ref before re-applying: `--changed-from` needs it in this
# repo, and the tree delta old..new decides whether the prefill-logit gate is relevant.
prev_tip=""
prev_base=""
if git -C "$LLAMA_REPO" rev-parse --git-dir >/dev/null 2>&1; then
    prev_tip="$(git -C "$LLAMA_REPO" rev-parse -q --verify refs/ci/last-applied 2>/dev/null || true)"
    prev_base="$(cat "$STATE_DIR/last-applied-base" 2>/dev/null || true)"
fi
if [ -n "$prev_tip" ] && [ "$prev_base" != "$base" ]; then
    log "note: base changed ($prev_base -> $base): full ladder, full prefill gate"
    prev_tip=""
fi

if [ ! -d "$LLAMA_REPO/.git" ]; then
    say "clone    : $LLAMA_REPO (first run; keep ONE path -- ccache keys on it)"
    git clone --quiet "$(git -C "$REPO" remote get-url origin | sed 's|llama-cpp-rdna-boosts|llama.cpp|')" "$LLAMA_REPO" \
        || fail "cannot clone the fork into $LLAMA_REPO"
fi
git -C "$LLAMA_REPO" fetch --quiet origin || fail "git fetch failed in $LLAMA_REPO"
git -C "$LLAMA_REPO" cat-file -e "${base}^{commit}" 2>/dev/null || fail "base $base not in $LLAMA_REPO (fetch the fork)"
git -C "$LLAMA_REPO" checkout --quiet --force --detach "$base" || fail "cannot check out base $base"
git -C "$LLAMA_REPO" branch -D rdna-boosts >/dev/null 2>&1 || true
if ! bash "$REPO/scripts/apply-all.sh" "$LLAMA_REPO" >"$RUN/apply.log" 2>&1; then
    fail "apply-all.sh failed (log: $RUN/apply.log)"
fi
applied="$(git -C "$LLAMA_REPO" rev-parse HEAD^{tree})"
[ "$applied" = "$tree" ] || fail "applied tree $applied != release.json.tree $tree"
tip="$(git -C "$LLAMA_REPO" rev-parse HEAD)"
say "apply    : strict, tree $applied == release.json.tree"

# --- 2. per-block build ladder ----------------------------------------------------------------
if [ "$SKIP_LADDER" = 1 ]; then
    say "ladder   : skipped (SKIP_LADDER=1)"
else
    ladder_args=(--repo "$LLAMA_REPO" --base "$base" --tip "$tip" --rocm "$ROCM_DIR"
                 --gpu-target "$GPU_TARGETS" --jobs "$JOBS" --fast --keep-going)
    [ -n "$prev_tip" ] && ladder_args+=(--changed-from "$prev_tip")
    if bash "$REPO/scripts/gate-per-block-build.sh" "${ladder_args[@]}" >"$RUN/ladder.log" 2>&1; then
        say "ladder   : $(grep -E '^==> result:' "$RUN/ladder.log" | tail -1 | sed 's/^==> result: //')"
    else
        cp "$RUN/ladder.log" "$STATE_DIR/last-ladder-fail.log"
        fail "per-block ladder (log: $RUN/ladder.log)"
    fi
fi

BUILD="$LLAMA_REPO/build-per-block"   # the gate's build dir, holding the TIP tree after the ladder
# --- 3. op oracles ----------------------------------------------------------------------------
if [ "$SKIP_OPTESTS" = 1 ]; then
    say "op tests : skipped (SKIP_OPTESTS=1)"
else
    if ! cmake --build "$BUILD" --target test-backend-ops llama-perplexity llama-cli -j "$JOBS" \
            >"$RUN/optest-build.log" 2>&1; then
        fail "tip build for the op tests (log: $RUN/optest-build.log)"
    fi
    opts_failed=""
    for op in FLASH_ATTN_EXT GATED_DELTA_NET FLASH_ATTN_QSA MUL_MAT MUL_MAT_ID TOP_K_MASK HC_MIX HC_COMBINE HC_COMBINE_MIX; do
        line="$(HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$BUILD/bin:$ROCM_DIR/lib" \
                timeout 3600 "$BUILD/bin/test-backend-ops" -o "$op" 2>&1 | grep -E 'tests passed|FAIL' | tail -1)"
        say "op       : $(printf '%-16s' "$op") ${line:-<no result>}"
        case "$line" in *passed*) ;; *) opts_failed="$opts_failed $op" ;; esac
    done
    [ -z "$opts_failed" ] || fail "op tests failed:$opts_failed"
fi

# --- 4. prefill-logit -------------------------------------------------------------------------
if [ "$SKIP_KLD" = 1 ]; then
    say "prefill  : skipped (SKIP_KLD=1)"
elif [ ! -f "$KLD_MODEL" ]; then
    say "prefill  : skipped (no model at $KLD_MODEL)"
else
    kld_args=(--ppl "$BUILD/bin/llama-perplexity" --model "$KLD_MODEL" --ctx "$KLD_CTX" --chunks "$KLD_CHUNKS")
    [ -n "${KLD_BASE_DIR:-}" ] && kld_args+=(--base-dir "$KLD_BASE_DIR")
    # A fresh worker has no recorded base: record one, say so, and let the NEXT run compare.
    have_base=0
    if [ -n "${KLD_BASE_DIR:-}" ]; then
        ls "$KLD_BASE_DIR"/prefill-logit-*.kld >/dev/null 2>&1 && have_base=1
    else
        ls "$HOME/.cache/rdna-boosts/gates"/prefill-logit-*.kld >/dev/null 2>&1 && have_base=1
    fi
    if [ "$have_base" -eq 0 ]; then
        kld_args+=(--record)
        say "prefill  : no recorded base on this box -> RECORDING one now (later runs compare)"
    fi
    # Relevance: the prefill-logit gate only means something if the applied tree moved a prefill
    # kernel (or the base moved, or this box has no base yet).  Otherwise skip -- it is the most
    # expensive step (~6 min here) and a docs/coverage change does not justify it.
    if [ -z "$prev_tip" ] || [ "$KLD_ALWAYS" = 1 ]; then
        kld_want=1
    else
        delta="$(git -C "$LLAMA_REPO" diff --name-only "$prev_tip" HEAD 2>/dev/null || echo ALL)"
        if printf '%s\n' "$delta" | grep -Eq "$PREFILL_RE"; then kld_want=1; else kld_want=0; fi
        [ "$kld_want" = 1 ] || say "prefill  : skipped (applied tree touched no prefill path; KLD_ALWAYS=1 forces it)"
    fi
    if [ "$kld_want" = 1 ]; then
        if LD_LIBRARY_PATH="$BUILD/bin:$ROCM_DIR/lib" bash "$REPO/scripts/gate-prefill-logits.sh" \
                "${kld_args[@]}" >"$RUN/kld.log" 2>&1; then
            say "prefill  : $(grep -E 'mean KLD|same top p' "$RUN/kld.log" | tr -s ' ' | tr '\n' ' ')"
        else
            cp "$RUN/kld.log" "$STATE_DIR/last-kld-fail.log"
            fail "prefill-logit gate (log: $RUN/kld.log)"
        fi
    fi
fi

# --- 4b. qwen4exp hyper-connection fusion equivalence (issue #228) ----------------------------
# `test-backend-ops -o HC_MIX,HC_COMBINE` compares each op against a CPU reference that can encode
# the same convention, so it cannot see an op that disagrees with the model's own CHAIN.  This runs
# the arms instead.  Two weight classes: Q8_0/BF16 hc_attn_down takes the fused mixer, any other
# quant takes the chain, and the defect class only shows on the cross arms.
if [ "$SKIP_HC" = 1 ]; then
    say "hc gate  : skipped (SKIP_HC=1)"
elif [ -z "$prev_tip" ] || [ "$HC_ALWAYS" = 1 ]; then
    hc_want=1
else
    delta="$(git -C "$LLAMA_REPO" diff --name-only "$prev_tip" HEAD 2>/dev/null || echo ALL)"
    if printf '%s\n' "$delta" | grep -Eq "$HC_RE"; then hc_want=1; else hc_want=0; fi
    [ "$hc_want" = 1 ] || say "hc gate  : skipped (applied tree touched no qwen4exp HC path; HC_ALWAYS=1 forces it)"
fi
if [ "$hc_want" = 1 ]; then
    hc_ran=0
    for hc_model in $HC_MODELS; do
        [ -f "$hc_model" ] || { say "hc gate  : no qwen4exp GGUF at $hc_model"; continue; }
        hc_log="$RUN/hc-$(basename "$hc_model").log"
        if LD_LIBRARY_PATH="$BUILD/bin:$ROCM_DIR/lib" HIP_VISIBLE_DEVICES=0 EXTRA_ARGS="-ngl 99 -fa on" \
                bash "$REPO/scripts/gate-hc-fusions.sh" "$hc_model" >"$hc_log" 2>&1; then
            hc_ran=1
            say "hc gate  : $(basename "$hc_model") $(grep -E '^(PASS|FAIL)' "$hc_log" | tail -1)"
        else
            cp "$hc_log" "$STATE_DIR/last-hc-fail.log"
            fail "qwen4exp HC fusion gate on $hc_model (log: $hc_log)"
        fi
    done
    [ "$hc_ran" = 1 ] || say "hc gate  : skipped (no qwen4exp GGUF on this box; set HC_MODELS)"
fi

# --- 4c. GDN direct-conv fold width purity (r183) ---------------------------------------------
# The three GDN decode folds (the direct conv also writing the snapshot tails, the conv-state gather
# deferral, the 1-token direct conv) and the shared-expert gate fold change the LAUNCH ORDER of a
# multi-token step, never its arithmetic.  Greedy text and the prefill-logit gate at `-ub 1` are both
# blind to that class: r80 + PR #177 was run-to-run unstable at P=3584 while its greedy text, its
# prefill-logit KLD (0.000000 / 100.000 %) and its WITHIN-run width purity all read clean -- the
# between-runs difference was the whole signal (TODO #75).  `test-logits-width-probe` at the verify
# widths is the instrument that saw it, so it is the arm a change to these folds owes.  r183 shipped
# the folds without it; r185 re-checked them with it (see the WORKLOG entry).

# How many GPUs this box exposes.  rocm-smi is not required: if the count cannot be read we report 1 and only
# the single-device arm runs (never a multi-device claim we did not actually exercise).
gdn_gpu_count() {
    local n
    n=$("$ROCM_DIR/bin/rocm-smi" -i 2>/dev/null || rocm-smi -i 2>/dev/null) || n=""
    n=$(printf '%s\n' "$n" | grep -oE '^GPU\[[0-9]+\]' | sort -u | wc -l)
    [ "${n:-0}" -ge 1 ] && printf '%s' "$n" || printf '1'
}

# "<W8 hash> <row-0 hash>" from a probe log -- the two values a determinism/purity claim is made of.
# Prints NOTHING when either half is missing (a bare `printf '%s %s'` of two empty strings is a space,
# which would make the "no hash" guard below unreachable).
gw_hashes() {
    local w8 r0
    w8="$(grep -oE '^W=8 rows=[0-9]+ hash=[0-9a-f]+' "$1" | tail -1 | sed 's/.*hash=//')"
    r0="$(grep -oE 'row0_row1_hashes: 1:[0-9a-f]+' "$1" | tail -1 | sed 's/.*1://')"
    [ -n "$w8" ] && [ -n "$r0" ] && printf '%s %s' "$w8" "$r0"
    return 0
}

gdn_width_step() {
    local probe="$BUILD/bin/test-logits-width-probe"
    local gw_ran=0 gw_fired=0 gw_model gw_arm gw_devs gw_split gw_p gw_r gw_ref gw_on gw_off gw_log gw_tag gw_ngpu arms
    if ! cmake --build "$BUILD" --target test-logits-width-probe -j "$JOBS" >"$RUN/gdn-width-build.log" 2>&1; then
        fail "building test-logits-width-probe (log: $RUN/gdn-width-build.log)"
    fi
    gw_ngpu="$(gdn_gpu_count)"
    for gw_model in $GDN_WIDTH_MODELS; do
        gw_tag="$(basename "$gw_model" .gguf)"
        if [ ! -f "$gw_model" ]; then
            say "gdn width: no GDN GGUF at $gw_tag (skipped)"
            continue
        fi
        gw_ran=1
        if [ -n "$GDN_WIDTH_ARMS" ]; then
            arms="$GW_SINGLE_ARM $GDN_WIDTH_ARMS"
        elif [ "$gw_ngpu" -ge 3 ]; then
            arms="$GW_SINGLE_ARM 0,1,2:tensor 0,1,2:layer"
        elif [ "$gw_ngpu" -eq 2 ]; then
            arms="$GW_SINGLE_ARM 0,1:tensor 0,1:layer"
        else
            arms="$GW_SINGLE_ARM"
            say "gdn width: single-GPU box, the multi-device arms are skipped (r186 made the fusion device-count independent, so they are the ones that matter -- run this on a 2+ GPU box before a release)"
        fi
        for gw_arm in $arms; do
            gw_devs="${gw_arm%%:*}"
            gw_split="${gw_arm##*:}"
            for gw_p in $GDN_WIDTH_P; do
                gw_ref=""
                for gw_r in $(seq 1 "$GDN_WIDTH_RUNS"); do
                    gw_log="$RUN/gdn-width-$gw_tag-$gw_devs-$gw_split-P$gw_p-r$gw_r"
                    # default arm: every fold on
                    if ! HIP_VISIBLE_DEVICES="$gw_devs" SPLIT="$gw_split" LD_LIBRARY_PATH="$BUILD/bin:$ROCM_DIR/lib" \
                            GGML_CUDA_CONV_DEBUG=1 \
                            timeout 3600 "$probe" "$gw_model" "$GDN_WIDTH_TEXT" "$gw_p" 512 >"$gw_log.on" 2>&1; then
                        fail "width probe, folds on ($gw_tag $gw_devs $gw_split P=$gw_p; log: $gw_log.on)"
                    fi
                    # kill-switch arm: every fold off (`GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0` too: it is the
                    # fourth fold and a BF16/F32-gate model is the only one it fires on).  r186: every fusion
                    # rides GGML_CUDA_DISABLE_CONV_FUSION, so the conv itself is stood down here as well --
                    # that is the reference path the fused arm must reproduce.
                    if ! HIP_VISIBLE_DEVICES="$gw_devs" SPLIT="$gw_split" LD_LIBRARY_PATH="$BUILD/bin:$ROCM_DIR/lib" \
                            GGML_CUDA_CONV_DEBUG=1 GGML_CUDA_DISABLE_CONV_FUSION=1 \
                            GGML_CUDA_DISABLE_GDN_CONV_SNAP=1 GGML_CUDA_FUSE_GDN_CONV_GATHER=0 \
                            GGML_CUDA_FUSE_GDN_CONV_DECODE=0 GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0 \
                            timeout 3600 "$probe" "$gw_model" "$GDN_WIDTH_TEXT" "$gw_p" 512 >"$gw_log.off" 2>&1; then
                        fail "width probe, folds off ($gw_tag $gw_devs $gw_split P=$gw_p; log: $gw_log.off)"
                    fi
                    gw_on="$(gw_hashes "$gw_log.on")"
                    gw_off="$(gw_hashes "$gw_log.off")"
                    # both halves present, else the log did not carry a probe result at all
                    case "$gw_on" in "") fail "width probe produced no W8/row-0 hash ($gw_tag $gw_devs $gw_split P=$gw_p; log: $gw_log.on)";; esac
                    case "$gw_off" in "") fail "width probe produced no W8/row-0 hash ($gw_tag $gw_devs $gw_split P=$gw_p; log: $gw_log.off)";; esac
                    grep -q 'GDN_CONV_DIRECT ' "$gw_log.on" \
                        || fail "the GDN direct conv never ran ($gw_tag $gw_devs $gw_split P=$gw_p) -- a multi-device arm needs 2+ visible devices (GDN_WIDTH_ARMS) and a GDN model (GDN_WIDTH_MODELS); log: $gw_log.on"
                    # `(^|[[:space:]])snap=1` and not a bare `snap=1`: the same line also carries `n_snap=1`,
                    # which would make the stand-down check below pass on the OFF arm by accident.
                    grep -qE '(^|[[:space:]])snap=1([[:space:]]|$)' "$gw_log.on" && gw_fired=1
                    # The comparison is only meaningful if the two arms really ran different paths: the
                    # reference arm must have taken CONCAT + SSM_CONV + CPY, with no direct conv at all.
                    if grep -q 'GDN_CONV_DIRECT ' "$gw_log.off"; then
                        fail "GGML_CUDA_DISABLE_CONV_FUSION=1 did not stand the conv down, so both arms ran the fused path and this arm is VACUOUS ($gw_tag $gw_devs $gw_split P=$gw_p; log: $gw_log.off)"
                    fi
                    if [ -z "$gw_ref" ]; then
                        gw_ref="$gw_on"
                    elif [ "$gw_on" != "$gw_ref" ]; then
                        fail "width probe is NOT run-to-run stable: $gw_tag $gw_devs $gw_split P=$gw_p run $gw_r gave '$gw_on', run 1 gave '$gw_ref' (logs: $gw_log.*)"
                    fi
                    [ "$gw_on" = "$gw_off" ] \
                        || fail "the conv fusion is NOT bit-identical to its reference path: $gw_tag $gw_devs $gw_split P=$gw_p fused '$gw_on' vs reference '$gw_off' (logs: $gw_log.*)"
                done
                say "gdn width: $gw_tag $gw_devs $gw_split P=$gw_p  W8/row0 $gw_ref  (== reference path, $GDN_WIDTH_RUNS runs)"
            done
        done
    done
    if [ "$gw_ran" = 1 ] && [ "$gw_fired" != 1 ]; then
        fail "no model reached the GDN snapshot fold (never logged snap=1), so the width gate was VACUOUS -- set GDN_WIDTH_MODELS to GDN/hybrid GGUFs or SKIP_GDN_WIDTH=1"
    fi
    [ "$gw_ran" = 1 ] || say "gdn width: skipped (no GDN GGUF on this box; set GDN_WIDTH_MODELS)"
}

if [ "$SKIP_GDN_WIDTH" = 1 ]; then
    say "gdn width: skipped (SKIP_GDN_WIDTH=1)"
else
    if [ -z "$prev_tip" ] || [ "$GDN_WIDTH_ALWAYS" = 1 ]; then
        gdn_width_step
    else
        delta="$(git -C "$LLAMA_REPO" diff --name-only "$prev_tip" HEAD 2>/dev/null || echo ALL)"
        if printf '%s\n' "$delta" | grep -Eq "$GDN_WIDTH_RE"; then
            gdn_width_step
        else
            say "gdn width: skipped (applied tree touched no GDN conv-fold source; GDN_WIDTH_ALWAYS=1 forces it)"
        fi
    fi
fi

# --- 5. record --------------------------------------------------------------------------------
# A partial run must NOT mark the rev validated: that would make the next cron pass skip a rev
# that was never fully gated (hit during the first radiance bring-up).
if [ "$SKIP_LADDER" = 1 ] || [ "$SKIP_OPTESTS" = 1 ] || [ "$SKIP_KLD" = 1 ] || [ "$SKIP_HC" = 1 ] \
        || [ "$SKIP_GDN_WIDTH" = 1 ]; then
    say "RESULT   : PASS (a step was skipped, so no up-to-date marker was written -- the next run resumes)"
    log "$short partially validated (summary: $SUMMARY)"
    exit 0
fi
echo "$CONTENT_KEY" > "$STATE_DIR/last-ok"
echo "$REV"        > "$STATE_DIR/last-ok-rev"
# The next run diffs against this chain (`--changed-from` + the prefill relevance test).
git -C "$LLAMA_REPO" update-ref refs/ci/last-applied "$tip"
echo "$base" > "$STATE_DIR/last-applied-base"
rm -f "$fail_file"
ls -1d "$STATE_DIR"/runs/*/ 2>/dev/null | sort | head -n -"$KEEP_RUNS" | xargs -r rm -rf
say "RESULT   : PASS  ($(date -Is))"
log "$short validated (summary: $SUMMARY)"
exit 0
