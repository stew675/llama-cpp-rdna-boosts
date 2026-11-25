#!/usr/bin/env bash
# The --fit FA-staging-arena reserve gate (GitHub issue #367, r360).
#
# WHY THIS EXISTS
#   `llama_context::memory_breakdown()`'s no_alloc (--fit) path charges a per-device compute reserve for
#   the FA **prefill staging arena**, because that arena is a raw device allocation outside the
#   compute-graph reserve and a deep prefill can grow it after the fit has sized the context.  The
#   backend answered with a FLAT `2 * GGML_CUDA_FA_STAGE_MAX_MB` (1024 MiB at the 512 MiB default) for
#   every device except RDNA3_5, i.e. for **every context with `-fa on` and a quantized K cache** --
#   including the delivery default `-ctk q8_0 -ctv q8_0`, where the kernels read the cache natively and
#   nothing is ever staged, so the arena never grows at all.
#
#   On a card where the model barely fits, that reserve is not free: `--fit` spends it on nothing and
#   offloads fewer layers.  Measured at an ~11 GiB budget with a 27B UD-IQ3_S (`-c 16384 -fa on
#   -ctk q8_0 -ctv q8_0`): compute tier 1539 MiB and `-ngl 52` with the flat reserve, 515 MiB and
#   `-ngl 60` without it, against the same-tree upstream build's 541 MiB / `-ngl 59`.  On the reporter's
#   12 GiB gfx1031 that is the difference between 4.0 t/s and upstream's 15.5 (issue #367).
#
#   The reserve is still REAL where a conversion happens: a KV pair the kernels cannot read natively
#   (a mixed pair, bf16 without bf16 MMA, or the `GGML_CUDA_FA_KV_NATIVE=0` staged arm) is converted
#   into the arena.  r360 therefore made the bound ask the launcher's own predicate
#   (`ggml_cuda_fattn_kv_pair_native_host`, i.e. `ggml_cuda_fattn_tile_type` applied to a probe tensor)
#   and charge 0 for a native pair -- the same "the allocator and the launcher agree" rule TODO 71
#   established, and the reason a policy switch turns both on together.
#
# WHAT IT CHECKS  (`llama-fit-params` only: no compute, no kernels, ~10 s.  Every arm pins the device.)
#   A. NATIVE PAIR => NO RESERVE.  `-fa on -ctk q8_0 -ctv q8_0` must project a compute tier within
#      COMPUTE_SLACK_MIB of the f16-KV arm's (which never charges), and at a tight `--fit-target` the
#      two arms must pick the SAME `-ngl`.  The f16 arm is the control: it charges nothing in any build.
#   B. THE STAGED PAIR STILL CHARGES.  `GGML_CUDA_FA_KV_NATIVE=0` (the arm that really converts) must
#      project >= RESERVE_MIN_MIB more than the native arm and pick FEWER layers.  Without this arm the
#      gate is vacuous -- an unconditional 0 would pass A.
#   C. THE RESERVE STILL FOLLOWS ITS CAP.  `GGML_CUDA_FA_STAGE_MAX_MB=0` (unbounded) must charge the
#      whole-F16-KV estimate, i.e. MORE than the default cap's reserve, so the accounting is still the
#      cap (and the `f16_kv_bytes` ceiling), not a constant that happens to match.
#   D. THE LIVE PATH IS UNTOUCHED.  A real llama-cli run prints its own compute buffer size; the fit's
#      reserve is an ESTIMATE only and must not appear in the run (the run must stay at the ubatch-
#      scaled figure it had before).  The gate reads it from `-lv 4` and only requires that it is
#      smaller than the fit's projection for the same configuration -- the reserve exists to cover an
#      allocation the compute buffer does not contain, so a run that already spent it would mean the
#      reserve is being double-counted somewhere else.
#
#   A green gate does NOT license changing the native-pair predicate: it proves the accounting agrees
#   with the launcher for the pair it tests.  A new KV type, a new staging arm or a change to
#   `ggml_cuda_fattn_tile_type` owes its own arm here (and `GREEDY-PURITY.md` §96's rule that a
#   family switch is not a text claim does not apply -- nothing here changes a kernel).
#
# USAGE
#   scripts/gate-fit-staging.sh [-m MODEL.gguf] [--gpu IDS] [--fitt MIB]
#   MODEL      a model whose KV the kernels read natively at q8_0 (default: the 27B UD-IQ3_S)
#   --gpu      HIP_VISIBLE_DEVICES for every arm (default 0)
#   --fitt     the --fit-target MiB that makes the arms placement-sensitive; default: free - 11264,
#              i.e. an ~11 GiB budget on whatever card is pinned.  The gate FITS it, it does not guess.
#   BUILD_DIR  the build to test (default $HOME/llama.cpp/build-rocm)
#
# PASS/FAIL lines per arm, exit 0 only when every assertion holds.

set -u

MODEL="${MODEL:-/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf}"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
GPU="${GPU:-0}"
CTX="${CTX:-16384}"
FITT="${FITT:-}"
COMPUTE_SLACK_MIB="${COMPUTE_SLACK_MIB:-8}"
RESERVE_MIN_MIB="${RESERVE_MIN_MIB:-900}"

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        --gpu)      GPU="$2"; shift 2 ;;
        --fitt)     FITT="$2"; shift 2 ;;
        -h|--help)  sed -n '2,60p' "$0"; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
done

FIT="$BUILD_DIR/bin/llama-fit-params"
CLI="$BUILD_DIR/bin/llama-cli"
for f in "$FIT" "$CLI"; do
    [ -x "$f" ] || { echo "error: $f not found (BUILD_DIR=$BUILD_DIR)" >&2; exit 2; }
done
[ -f "$MODEL" ] || { echo "error: model not found: $MODEL" >&2; exit 2; }

fail=0
note() { printf '  %s\n' "$*"; }
ok()   { printf 'PASS  %s\n' "$*"; }
bad()  { printf 'FAIL  %s\n' "$*"; fail=1; }

# one fit arm: echoes "<model_mib> <context_mib> <compute_mib> <free_mib> <ngl>" ("-1" ngl = fits as-is).
# TWO invocations, because llama-fit-params prints EITHER the estimated-memory table (`-fitp on`) OR the
# fitted CLI arguments, never both.
fit_arm() { # env... -- args...
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local mb ngl out
    mb=$(env "${envs[@]}" HIP_VISIBLE_DEVICES="$GPU" timeout 600 "$FIT" -m "$MODEL" "$@" -fitp on -lv 4 2>&1)
    ngl=$(env "${envs[@]}" HIP_VISIBLE_DEVICES="$GPU" timeout 600 "$FIT" -m "$MODEL" "$@" 2>&1 \
          | grep -E '^-c ' | head -1 | grep -oE '\-ngl -?[0-9]+' | awk '{print $2}')
    # the device row is "(self = model + context + compute)" -- keep the four numbers only
    local row
    row=$(printf '%s\n' "$mb" | grep -oE '\([0-9]+ = [0-9]+ \+ +[0-9]+ \+ +[0-9]+\)' | head -1 | tr -dc '0-9 ')
    if [ -z "$row" ]; then
        echo "0 0 0 0 ${ngl:--1}"
        return
    fi
    # shellcheck disable=SC2086
    set -- $row
    local free
    free=$(printf '%s\n' "$mb" | grep -oE '[0-9]+ = [0-9]+ \+ \(' | head -1 | awk '{print $3}')
    echo "$2 $3 $4 ${free:-0} ${ngl:--1}"
}

COMMON=( -c "$CTX" -fa on )

echo "=== gate-fit-staging: $MODEL"
echo "    build=$BUILD_DIR  gpu=$GPU  ctx=$CTX"

# The --fit-target must be a budget TIGHT enough that a 1 GiB reserve can change the layer count, so it
# is derived from the card's free VRAM (an ~11 GiB budget) rather than assumed.
if [ -z "$FITT" ]; then
    read -r _m _c _k _free _n <<<"$(fit_arm X=1 -- "${COMMON[@]}")"
    [ "${_k:-0}" -gt 0 ] || { echo "error: the f16 control arm produced no breakdown" >&2; exit 1; }
    FITT=$(( _free - 11264 ))
    [ "$FITT" -gt 0 ] || { echo "error: cannot derive a tight --fit-target from free=${_free} MiB" >&2; exit 1; }
fi
echo "    fitt=${FITT} MiB"
TIGHT=( "${COMMON[@]}" -ctk q8_0 -ctv q8_0 -fitt "$FITT" )

# f16 KV is the control: a quantized V cache requires -fa, and f16 never stages in any build.  It runs at
# the SAME --fit-target so its layer count is the like-for-like number for A2.
read -r _m _c ctl_compute _f ctl_ngl <<<"$(fit_arm X=1 -- "${COMMON[@]}" -fitt "$FITT")"
read -r _m _c nat_compute _f nat_ngl <<<"$(fit_arm X=1 -- "${TIGHT[@]}")"
read -r _m _c std_compute _f std_ngl <<<"$(fit_arm GGML_CUDA_FA_KV_NATIVE=0 -- "${TIGHT[@]}")"
read -r _m _c big_compute _f big_ngl <<<"$(fit_arm GGML_CUDA_FA_KV_NATIVE=0 GGML_CUDA_FA_STAGE_MAX_MB=0 -- "${TIGHT[@]}")"
[ "${ctl_compute:-0}" -gt 0 ] || { echo "error: the f16 control arm produced no breakdown" >&2; exit 1; }

note "control (f16 KV)          compute=${ctl_compute} MiB  ngl=${ctl_ngl}"
note "native pair (q8_0/q8_0)   compute=${nat_compute} MiB  ngl=${nat_ngl}"
note "staged  (KV_NATIVE=0)     compute=${std_compute} MiB  ngl=${std_ngl}"
note "staged + unbounded cap    compute=${big_compute} MiB  ngl=${big_ngl}"

# A. a native pair charges no staging reserve, and it does not cost layers.
delta=$(( nat_compute - ctl_compute ))
if awk -v d="$delta" -v s="$COMPUTE_SLACK_MIB" 'BEGIN{exit !(d <= s && d >= -s)}'; then
    ok "A1 native q8_0 pair charges no staging reserve (compute ${nat_compute} vs f16 control ${ctl_compute} MiB)"
else
    bad "A1 native q8_0 pair charges ${delta} MiB of staging reserve (limit ${COMPUTE_SLACK_MIB}) -- the arena is only ever filled by a conversion, and this pair has none"
fi
if [ "$nat_ngl" = -1 ] || [ "$std_ngl" = -1 ]; then
    bad "A2 the arm is VACUOUS: -ngl is -1 on a native or staged arm (nat=${nat_ngl} staged=${std_ngl}) -- --fitt=${FITT} is not tight enough for this model, lower it (or pass a bigger model)"
elif [ "$nat_ngl" -ge "$ctl_ngl" ]; then
    # The q8_0 cache is SMALLER than the f16 control's, so the native arm may buy layers -- it must never
    # buy FEWER than the arm that reserves nothing at all.
    ok "A2 a native pair offloads at least the f16 control's layers (${nat_ngl} >= ${ctl_ngl}; the q8_0 cache is the smaller one)"
else
    bad "A2 native pair picked -ngl ${nat_ngl}, FEWER than the f16 control's ${ctl_ngl} -- the native pair is still paying for a reserve the control does not have"
fi

# B. the staged arm still reserves (non-vacuity for A).
if [ $(( std_compute - nat_compute )) -ge "$RESERVE_MIN_MIB" ]; then
    ok "B1 the staged arm still reserves $(( std_compute - nat_compute )) MiB (>= ${RESERVE_MIN_MIB}) for the arena it really uses"
else
    bad "B1 the staged arm (GGML_CUDA_FA_KV_NATIVE=0) reserved only $(( std_compute - nat_compute )) MiB over the native arm -- the reserve was removed unconditionally, not made predicate-driven"
fi
[ "$std_ngl" -lt "$nat_ngl" ] && ok "B2 the staged reserve visibly costs layers (${std_ngl} < ${nat_ngl})" \
                              || bad "B2 the staged arm did not lose a layer (${std_ngl} vs ${nat_ngl}) -- the placement consequence is not reproduced"

# C. the reserve still follows GGML_CUDA_FA_STAGE_MAX_MB.
if [ "$big_compute" -gt "$std_compute" ]; then
    ok "C1 an unbounded cap reserves more than the capped staged arm (${big_compute} > ${std_compute} MiB)"
else
    bad "C1 GGML_CUDA_FA_STAGE_MAX_MB=0 did not raise the reserve (${big_compute} vs ${std_compute} MiB) -- the accounting no longer follows the cap"
fi

# D. the live path: the run's own compute buffer must not contain the fit's reserve.
run_compute=$(HIP_VISIBLE_DEVICES="$GPU" timeout 900 "$CLI" -m "$MODEL" "${COMMON[@]}" -ctk q8_0 -ctv q8_0 -c "$CTX" -ngl 99 -p "hi" -n 1 \
    --no-display-prompt --single-turn -lv 4 2>&1 | grep -oE 'compute buffer size = +[0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+')
if [ -z "$run_compute" ]; then
    bad "D1 the live arm printed no compute buffer size (llama-cli failed?)"
elif awk -v run="$run_compute" -v f16="$ctl_compute" 'BEGIN{exit !(run < f16)}'; then
    ok "D1 the live run's compute buffer (${run_compute} MiB) stays below the fit's projection (${ctl_compute} MiB): the reserve is an estimate, not a double count"
else
    bad "D1 the live run's compute buffer (${run_compute} MiB) is at or above the fit's projection (${ctl_compute} MiB)"
fi

echo
if [ "$fail" = 0 ]; then
    echo "gate-fit-staging: PASS"
else
    echo "gate-fit-staging: FAIL"
fi
exit "$fail"
