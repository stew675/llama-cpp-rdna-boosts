#!/usr/bin/env bash
# The GDN output-norm gate (block 10, r353).
#
# WHY THIS EXISTS
#   Block 10 skips a GDN layer's output `RMS_NORM -> MUL(w)` pair and lets the `gate -> Q8_1` kernel compute the
#   norm itself (`ggml_cuda_gdn_norm_gate_*`, `unary_gated_q8_1_norm_kernel`), which saves two launches per GDN
#   layer per step.  r102 matched a SIGMOID gate only, i.e. qwen4exp; r353 widens the matcher to SILU as well,
#   which is what qwen35 / qwen35moe / qwen3next / clover-moe emit (`build_norm_gated()` -> `ggml_silu`).
#
#   The change is structurally invisible to every existing instrument: a matcher that stops matching costs two
#   launches and changes NO hash, and a numeric change inside the folded norm is a plausible-shift class that
#   only a KLD point or a bit-identity arm sees.  So this gate is a MECHANISM gate first -- the census must prove
#   the fusion fired, and the OFF arm must prove the switch is not vacuous -- and a bit-identity gate second.
#
# WHAT IT CHECKS
#   HARD  * per model, the mechanism: with `GGML_CUDA_FUSE_GDN_NORM_GATE_CENSUS=1` the ON arm must log
#           `GDN_NORM_GATE_CENSUS ... silu=on fused sigmoid=<n> silu=<m> declined silu=0` with **m > 0**, at the
#           decode width (`-ub 1`) AND in the verify/prefill band (`-ub 8`) -- a SILU model whose fusion fires at
#           one width and not the other is a defect the text arms cannot see.  The `-ub 1` arm is driven with
#           `llama-perplexity` because `llama-cli -b 1` aborts in `common.cpp`'s `get_sub_batch` (pre-existing,
#           unrelated to this feature -- the same reason gate-gdn-epi.sh has no `-ub 1` text arm).
#   HARD  * per model, the OFF arm (`GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0`) must log `silu=off ... silu=0` with
#           **declined silu > 0**: the counter is taken at the same gate node as the fused one, so a non-zero value
#           proves the OFF arm still reached the decision.  (The two counts are NOT required to be equal -- a
#           two-slot server walk counts them differently, measured 384 vs 672 -- so equality would be a brittle
#           gate; non-vacuity is what matters.)
#   HARD  * per model and per ubatch (2, 4, 8, 512: the mmvq verify band, its 8-token upper edge, and the prefill
#           band where the kernel is inert) the ON text is byte-identical to the OFF text.  The folded norm reduces
#           exactly like `rms_norm_f32`'s strided path (GREEDY-PURITY §60), so the oracle is byte identity, not a
#           tolerance.  There is no llama-cli arm at `-ub 1` (see above); that width's oracle is the `--ub 1` KLD
#           point of `scripts/gate-prefill-logits.sh`, which a release runs alongside this gate (`--kld` here).
#   HARD  * the SIGMOID path is untouched: on an arm where the fused count is sigmoid (a qwen4exp shape) the
#           census must still report it under `fused sigmoid`.  This gate cannot build that model (Flash-Next needs
#           ~100 GB), so what it enforces instead is the *strict* silu/sigmoid split on the SILU models: `fused
#           sigmoid` must be 0 there (a non-zero value means the matcher mis-attributed the op).
#   HARD  * no abort, and `scripts/check-gpu-faults.sh` clean over this gate's own window.
#   INFO  * generation t/s per arm, so the (small) win is visible in the record -- NOT gated on.
#   OPT   * `--kld` also runs `scripts/gate-prefill-logits.sh --ab "GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0"` at
#           `--ub 1` and `--ub 4`, which must read 0.000000 mean KLD / 100.000 % same-top-p (the exact-fallback
#           contract of GREEDY-PURITY §60).  It is off by default because it is a 4-run, multi-GB, ~20-minute
#           arm; a RELEASE owes it (MANIFESTS.md gate 41), and the gate prints the exact command when skipped.
#
# WHAT IT CANNOT FORCE
#   (a) A model whose GDN output norm is not 128-wide or whose gate shape differs: the matcher's shape guards
#       (`w->ne[0] == 128`, `xn->ne[3] == 1`, the use counts, the window's memory-range test) are checked by
#       construction, and the `xn->ne[3] != 1` guard is what keeps a multi-sequence ubatch out (the r108 class).
#   (b) Text identity cannot see a *uniform* logit shift; `gate-prefill-logits.sh` is that instrument.
#   (c) Windows / RDNA3 / NVIDIA: no runner here.
#
# Usage:
#   ./gate-gdn-norm-gate.sh [--no-faults] [--kld] [--keep]
#   ./gate-gdn-norm-gate.sh --self-test [PREFIX_CLI]   # negative test: a PRE-r353 build must FAIL the census
#                                                      # check (it has no census at all, so it logs nothing)
#
# Environment:
#   BUILD_DIR     llama.cpp build tree       (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI     llama-cli                  (default: $BUILD_DIR/bin/llama-cli)
#   MODEL_DENSE   dense GDN GGUF (qwen35)    (default: Qwen3.8-27B UD-Q4_K_XL)
#   MODEL_MOE     MoE GDN GGUF (qwen35moe)   (default: Qwen3.6-35B-A3B UD-Q4_K_M)
#   UB_LIST       text sweep                 (default: "2 4 8 512" -- no `-ub 1`: `llama-cli -b 1` aborts upstream)
#   NGEN          generated tokens           (default: 40; the census arms use 32)
#   PROMPT        prompt string              (default: below)
#   GPU           HIP_VISIBLE_DEVICES        (default: 0; the iGPU must never be enumerated)
#   TIME_LIMIT    per llama-cli timeout      (default: 1200 s)
#   KEEPOUT       dir for the logs           (default: mktemp -d)
#   KLD_BASE_DIR  --kld's base directory     (default: ~/.cache/rdna-boosts/gates/gdn-norm-gate -- its own
#                                            directory, so a gate run never overwrites the delivery's canonical
#                                            `-ub1` / `-ub4` bases, which other gates' decode-width probes read)
#   DENSE_ARM=0 / MOE_ARM=0                  skip that model
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL_DENSE="${MODEL_DENSE:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
MODEL_MOE="${MODEL_MOE:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
UB_LIST="${UB_LIST:-2 4 8 512}"
NGEN="${NGEN:-40}"
PROMPT="${PROMPT:-The capital of France is Paris, and the capital of Germany is Berlin. Explain, in a few sentences, why river crossings decided where the first cities of Europe were built.}"
GPU="${GPU:-0}"
TIME_LIMIT="${TIME_LIMIT:-1200}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
DENSE_ARM="${DENSE_ARM:-1}"
MOE_ARM="${MOE_ARM:-1}"
KLD_BASE_DIR="${KLD_BASE_DIR:-$HOME/.cache/rdna-boosts/gates/gdn-norm-gate}"

FAULTS=1
KLD=0
SELF_TEST=0
SELF_TEST_CLI=""
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        --kld)       KLD=1 ;;
        --keep)      : ;;
        --self-test) SELF_TEST=1 ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \?//; /^set -uo/d'; exit 0 ;;
        -*)          echo "unknown option: $a" >&2; exit 2 ;;
        *)           SELF_TEST_CLI="$a" ;;
    esac
done

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

[ -x "$LLAMA_CLI" ] || { echo "ERROR: no llama-cli at $LLAMA_CLI" >&2; exit 2; }

LLAMA_PPL="${LLAMA_PPL:-$BUILD_DIR/bin/llama-perplexity}"

# The 1-token-width census arm: llama-perplexity (llama-cli -b 1 aborts in get_sub_batch, pre-existing).
census_ppl() {   # census_ppl TAG MODEL ENV...
    local tag="$1" model="$2"; shift 2
    local log="$KEEPOUT/$tag.log"
    ( export HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_FUSE_GDN_NORM_GATE_CENSUS=1
      env "$@" timeout "$TIME_LIMIT" "$LLAMA_PPL" -m "$model" -ngl 99 -fa on -c 512 --chunks 1 -ub 1 \
          -t 8 -lv 2 ) > "$log" 2>&1
    if [ $? -ne 0 ]; then
        echo "FAILED $log"
        return
    fi
    grep -o 'GDN_NORM_GATE_CENSUS .*' "$log" | tail -1
}

# The census arm at any llama-cli-reachable width: one run with the census on, printing its census line.
census_line() {   # census_line TAG CLI MODEL UB ENV...
    local tag="$1" cli="$2" model="$3" ub="$4"; shift 4
    local log="$KEEPOUT/$tag.log"
    ( export HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_FUSE_GDN_NORM_GATE_CENSUS=1
      env "$@" timeout "$TIME_LIMIT" "$cli" -m "$model" -ngl 99 -sm layer \
          -b "$ub" -ub "$ub" -p "$PROMPT" -n 32 --seed 42 --temp 0 \
          --no-display-prompt --single-turn -lv 2 ) > "$log" 2>&1
    if [ $? -ne 0 ]; then
        echo "FAILED $log"
        return
    fi
    grep -o 'GDN_NORM_GATE_CENSUS .*' "$log" | tail -1
}

# one text arm; prints "<n> chars sha=<hex> <t/s>" via scripts/extract-generated.py (no -lv at all: -lv 4 would
# interleave the census into the text stream).  Like gate-gdn-epi.sh, this must not call fail() itself: the
# caller captures its stdout, so a subshell's `fails` increment would be discarded (a false PASS).
text_arm() {   # text_arm TAG MODEL UB ENV...
    local tag="$1" model="$2" ub="$3"; shift 3
    local log="$KEEPOUT/$tag.log"
    ( export HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0
      env "$@" timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$model" -ngl 99 -sm layer \
          -b "$ub" -ub "$ub" -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 \
          --no-display-prompt --single-turn ) > "$log" 2>&1
    if [ $? -ne 0 ]; then
        echo "FAILED $log"
        return
    fi
    local out rate
    out="$(python3 "$REPO_DIR/scripts/extract-generated.py" "$log" 2>/dev/null)"
    rate="$(grep -oE 'Generation: *[0-9.]+ t/s' "$log" | tail -1 | grep -oE '[0-9.]+')"
    echo "$out ${rate:--}"
}

text_of()  { echo "$1" | awk '{print $3}' | sed 's/sha=//'; }
rate_of()  { echo "$1" | awk '{print $4}'; }
failed()   { case "$1" in FAILED*) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------------------
# --self-test: the mechanism check must have teeth.  A pre-r353 build has no census switch at all, so it logs
# no GDN_NORM_GATE_CENSUS line and the ON arm's `silu > 0` requirement must fail.
# ---------------------------------------------------------------------------
if [ "$SELF_TEST" = 1 ]; then
    cli="${SELF_TEST_CLI:-$LLAMA_CLI}"
    model="$MODEL_DENSE"
    echo "==> GDN output-norm gate --self-test (a pre-r353 build must FAIL the census check)"
    echo "    cli=$cli"
    [ -x "$cli" ] || { echo "ERROR: no llama-cli at $cli" >&2; exit 2; }
    line="$(census_line selftest "$cli" "$model" 8 X=1)"
    if failed "$line"; then
        echo "ERROR: the self-test arm did not complete ($line)" >&2; exit 2
    fi
    if [ -z "$line" ]; then
        echo "    pre-fix arm: no census line at all -> the gate's mechanism check FAILS on it.  PASS (self-test)"
        exit 0
    fi
    echo "    pre-fix arm printed: $line"
    echo "==> FAIL (self-test): this build has the census, so it is not a pre-fix binary (or the feature was"
    echo "    backported); point the self-test at a build from before r353."
    exit 1
fi

echo "==> GDN output-norm gate (block 10, r353)"
echo "    build=$BUILD_DIR"
echo "    model_dense=$MODEL_DENSE"
echo "    model_moe=$MODEL_MOE"
echo "    ubs=$UB_LIST  ngen=$NGEN  logs=$KEEPOUT"

check_model() {   # check_model NAME MODEL
    local name="$1" model="$2"
    [ -f "$model" ] || { note "SKIP $name: no such model $model"; return; }

    # 1. the mechanism at the decode width (perplexity: the 1-token graph) and in the verify band (cli)
    local ub line fsil fsig dec
    for ub in 1 8; do
        if [ "$ub" = 1 ]; then
            line="$(census_ppl "$name-census-on-ub1" "$model" X=1)"
        else
            line="$(census_line "$name-census-on-ub$ub" "$LLAMA_CLI" "$model" "$ub" X=1)"
        fi
        if failed "$line"; then fail "$name -ub $ub: the census arm did not complete ($line)"; continue; fi
        if [ -z "$line" ]; then
            fail "$name -ub $ub: no GDN_NORM_GATE_CENSUS line -- the census did not run"
            continue
        fi
        note "$name -ub $ub ON : $line"
        case "$line" in *"silu=on"*) ;; *) fail "$name -ub $ub: the census does not report silu=on on the default arm" ;; esac
        case "$line" in *"declined silu=0"*) ;; *) fail "$name -ub $ub: the ON arm declined a SILU gate (the switch is on)" ;; esac
        fsil="$(echo "$line" | sed -n 's/.*fused sigmoid=\([0-9]*\) silu=\([0-9]*\).*/\2/p')"
        fsig="$(echo "$line" | sed -n 's/.*fused sigmoid=\([0-9]*\) silu=\([0-9]*\).*/\1/p')"
        if [ -z "$fsil" ] || [ "${fsil:-0}" -le 0 ]; then
            fail "$name -ub $ub: fused silu=$fsil -- the SILU norm-gate fusion did not fire (the arm is vacuous)"
        fi
        if [ "${fsig:-0}" -ne 0 ]; then
            fail "$name -ub $ub: fused sigmoid=$fsig on a SILU-gate model -- the matcher mis-attributed the gate op"
        fi
    done

    # 2. the OFF arm: the switch must decline the same decision (non-vacuity)
    line="$(census_line "$name-census-off" "$LLAMA_CLI" "$model" 8 GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0)"
    if failed "$line"; then
        fail "$name: the OFF census arm did not complete ($line)"
    elif [ -z "$line" ]; then
        fail "$name: no census line on the OFF arm"
    else
        note "$name   OFF: $line"
        case "$line" in *"silu=off"*) ;; *) fail "$name: the OFF arm does not report silu=off" ;; esac
        case "$line" in *"fused sigmoid=0 silu=0"*) ;; *) fail "$name: the OFF arm fused a SILU gate -- the switch does not disable the widening" ;; esac
        dec="$(echo "$line" | sed -n 's/.*declined silu=\([0-9]*\).*/\1/p')"
        if [ -z "$dec" ] || [ "${dec:-0}" -le 0 ]; then
            fail "$name: declined silu=$dec -- the OFF arm never reached the decision (a vacuous arm cannot judge the switch)"
        fi
    fi

    # 3. bit-identical text on vs off, per ubatch
    local a b ha hb ra rb
    for ub in $UB_LIST; do
        a="$(text_arm "$name-on-ub$ub"  "$model" "$ub" X=1)"
        b="$(text_arm "$name-off-ub$ub" "$model" "$ub" GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0)"
        if failed "$a" || failed "$b"; then
            fail "$name -ub $ub: an arm did not complete ($(echo "$a" | head -1) / $(echo "$b" | head -1))"
            continue
        fi
        ha="$(text_of "$a")"; hb="$(text_of "$b")"
        ra="$(rate_of "$a")"; rb="$(rate_of "$b")"
        if [ -z "$ha" ] || [ -z "$hb" ]; then
            fail "$name -ub $ub: no text hash from one of the arms"
            continue
        fi
        if [ "$ha" != "$hb" ]; then
            fail "$name -ub $ub: on ($ha) != off ($hb) -- the folded norm is not bit-identical to the chain"
        else
            note "$name -ub $ub: text $ha (on == off)   t/s on=$ra off=$rb  (info, not gated)"
        fi
    done
}

[ "$DENSE_ARM" = 1 ] && check_model dense "$MODEL_DENSE"
[ "$MOE_ARM" = 1 ]   && check_model moe   "$MODEL_MOE"

# ---------------------------------------------------------------------------
# the exact-fallback KLD point (a release owes it; skipped unless --kld)
# ---------------------------------------------------------------------------
if [ "$KLD" = 1 ]; then
    echo "==> prefill-logit exact-fallback arm (mean KLD must be 0.000000, same-top-p 100.000 %)"
    for ub in 1 4; do
        if ( cd "$REPO_DIR" && timeout "$TIME_LIMIT" ./scripts/gate-prefill-logits.sh --ppl "$BUILD_DIR/bin/llama-perplexity" \
                --base-dir "$KLD_BASE_DIR" --ub "$ub" --chunks 8 --ab "GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0" ) > "$KEEPOUT/kld-ub$ub.log" 2>&1; then
            note "-ub $ub: PASS ($(grep -oE 'mean KLD = [-0-9.]+' "$KEEPOUT/kld-ub$ub.log" | tail -1), $(grep -oE 'same top p = [0-9.]+ %' "$KEEPOUT/kld-ub$ub.log" | tail -1))"
        else
            fail "-ub $ub: gate-prefill-logits failed ($KEEPOUT/kld-ub$ub.log)"
        fi
    done
else
    echo "==> prefill-logit arm SKIPPED (pass --kld).  A release owes:"
    echo "    ./scripts/gate-prefill-logits.sh --ppl $BUILD_DIR/bin/llama-perplexity --base-dir $KLD_BASE_DIR --ub 1 --chunks 8 --ab \"GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0\""
    echo "    ./scripts/gate-prefill-logits.sh --ppl $BUILD_DIR/bin/llama-perplexity --base-dir $KLD_BASE_DIR --ub 4 --chunks 8 --ab \"GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0\""
fi

# ---------------------------------------------------------------------------
# GPU faults over this gate's own window
# ---------------------------------------------------------------------------
echo "==> GPU faults"
if [ "$FAULTS" = 1 ]; then
    logs=()
    while IFS= read -r l; do logs+=(--log "$l"); done < <(find "$KEEPOUT" -maxdepth 1 -name '*.log' | sort)
    if bash "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$GATE_SINCE_TS" "${logs[@]}" > "$KEEPOUT/faults.log" 2>&1; then
        note "kernel journal and every run log clean"
    else
        fail "GPU faults found ($KEEPOUT/faults.log)"
    fi
else
    note "skipped (--no-faults)"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: the SILU gate folds into the norm kernel, bit-identically, and the switch declines it"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed (logs in $KEEPOUT)"
exit 1
