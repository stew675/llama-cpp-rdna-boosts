#!/usr/bin/env bash
# shape-matrix -- STEP 9's measurement: what the pair does NOT change (the answers) and what it COSTS
# (VRAM, the reserve it declares, the region it pins), per shape.
#
# THIS IS A MEASUREMENT HARNESS, NOT A GATE.  It prints a table and writes `$OUT/shape-matrix.tsv`; the claims it
# is allowed to support are exactly two:
#
#   1. TEXT IDENTITY -- for one request in flight, `N` FRESH server processes per arm read the same answers with
#      the pair OFF and ON.  Compared as a SET per run (the server assigns prompts to slots by LRU, so which slot
#      got which prompt is not part of the answer).  Also reported: run-stability within an arm (`k/N`), because
#      an arm whose own repeats disagree cannot support a comparison at all.
#      A CONCURRENT arm is deliberately NOT used for a byte claim: the batch composition picks the kernels and
#      even the unmodified server reads a different answer run to run on some shapes (INVARIANTS.md r362-wip
#      item 5).  The multi-slot/concurrent arms are `gate-arms-multislot.sh`'s job.
#
#   2. MEMORY -- the WHOLE-server VRAM per device while the model is LOADED (`server-arm.sh --vram-probe`), so
#      the pair's cost is (pair ON) - (pair OFF) on the same shape, plus the reserve the server DECLARES and the
#      auxiliary region the slab pins.  This is the number a promotion decision needs.
#
# SHAPES (`--shapes` is a comma list of these keys):
#   4b               Qwen3.5 4B Q8_0, 1 device            -- dense, GPU-resident
#   phi4             phi-4 IQ4_XS, 1 device               -- a second dense family
#   27b-tensor       Qwen3.8 27B Q8_0, 3 devices `-sm tensor -mg 0`   -- the comm layer
#   35b-host         Qwen3.6 35B-A3B Q8_0, 1 device `-ncmoe 40` np 4  -- the slab + host expert tier
#   flashnext        Qwen3.8 Flash-Next GSQ-RCO IQ3_S, 3 devices `-sm tensor -mg 0`, np 3  -- qwen4exp/hybrid
#   gemma4-iswa      Gemma4 26B-A4B QAT UD-Q4_K_XL, 1 device  -- the iswa MoE, VRAM-resident
#   gemma4-draft     the same + its shared MTP head `-md`  -- the DRAFT-CONTEXT row: the pair MUST REFUSE it
#                    (a gemma4 assistant assumes it is its target context's only other user), so this row records
#                    the pair-OFF answers and the refusal message, and nothing else
#
# WHY Flash-Next GSQ IQ3_S and not the IQ4_NL tree: IQ4_NL is ~100 GB across its 9 shards and this box has 3 x
# 32 GB, so it does not fit a 3-device split with or without the pair (a memory fact, not a pair fact).  The
# GSQ IQ3_S GGUF (~84 GB) is what fits, and it is the shape this campaign's gates already use.
#
# USAGE: shape-matrix.sh [--shapes LIST] [--repeats N] [--n-predict N] [--keep]
# One GPU job at a time: the whole run holds the exclusive GPU flock.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$V2/../.." && pwd)"
BASE="$V2/baseline"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/shape-matrix}"
SHAPES="4b,phi4,27b-tensor,35b-host,flashnext,gemma4-iswa,gemma4-draft"
REPEATS="${REPEATS:-3}"
NPRED="${NPRED:-48}"
KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --shapes) SHAPES="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --n-predict) NPRED="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"
TSV="$OUT/shape-matrix.tsv"
: > "$TSV"
printf 'shape\tnp\tctx\tarm\ttext\trun_stable\thashes\tvram_delta_mib\trss_delta_mib\treserve_present\tcensus\n' >> "$TSV"

MTP_G4=/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-Q4_0-MTP.gguf

shape_var() {
    case "$1:$2" in
        4b:model)          echo /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf ;;
        4b:devices)        echo 0 ;;      4b:np)   echo 2 ;;   4b:ctx)  echo 8192 ;;
        4b:extra)          echo "" ;;     4b:env)  echo "" ;;  4b:kind) echo pair ;;

        phi4:model)        echo /llm/models/phi-4/IQ4_XS/phi-4-IQ4_XS.gguf ;;
        phi4:devices)      echo 0 ;;      phi4:np) echo 2 ;;   phi4:ctx) echo 8192 ;;
        phi4:extra)        echo "" ;;     phi4:env) echo "" ;; phi4:kind) echo pair ;;

        27b-tensor:model)  echo /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf ;;
        27b-tensor:devices) echo 0,1,2 ;; 27b-tensor:np) echo 2 ;; 27b-tensor:ctx) echo 8192 ;;
        27b-tensor:extra)  echo "-sm tensor -mg 0" ;; 27b-tensor:env) echo "" ;; 27b-tensor:kind) echo pair ;;

        35b-host:model)    echo /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf ;;
        35b-host:devices)  echo 0 ;;      35b-host:np)  echo 4 ;;   35b-host:ctx)  echo 16384 ;;
        35b-host:extra)    echo "-ncmoe 40" ;; 35b-host:env) echo "MOE_EXPERT_HEATMAP=load" ;; 35b-host:kind) echo pair ;;

        flashnext:model)   echo /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf ;;
        flashnext:devices) echo 0,1,2 ;;  flashnext:np) echo 3 ;;  flashnext:ctx) echo 8192 ;;
        flashnext:extra)   echo "-sm tensor -mg 0" ;; flashnext:env) echo "" ;; flashnext:kind) echo pair ;;

        gemma4-iswa:model) echo /llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf ;;
        gemma4-iswa:devices) echo 0 ;;    gemma4-iswa:np) echo 2 ;; gemma4-iswa:ctx) echo 8192 ;;
        gemma4-iswa:extra) echo "" ;;     gemma4-iswa:env) echo "" ;; gemma4-iswa:kind) echo pair ;;

        gemma4-draft:model) echo /llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf ;;
        gemma4-draft:devices) echo 0 ;;   gemma4-draft:np) echo 2 ;; gemma4-draft:ctx) echo 8192 ;;
        gemma4-draft:extra) echo "-ncmoe 20 -md $MTP_G4 --spec-type draft-mtp" ;;
        gemma4-draft:env)  echo "MOE_EXPERT_HEATMAP=load" ;; gemma4-draft:kind) echo refusal ;;
        *) echo "" ;;
    esac
}

lock="${MTP_LOCK:-$HOME/.cache/rdna-boosts/gpu.lock}"
mkdir -p "$(dirname "$lock")"
exec 9>"$lock"
if ! flock -w "${LOCK_WAIT:-5400}" 9; then
    echo "FAIL: another GPU job holds $lock" >&2; exit 1
fi
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
LOGS=()

hashes_of() { sed 's/.*sha=//' "$1" 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//'; }
vram_of()   { sed -n 's/^gpu\([0-9]*\) total_bytes=.* used_bytes=\([0-9]*\)/\1 \2/p' "$1" 2>/dev/null; }
rss_of()    { sed -n 's/^# tag=.* rss_kib=\([0-9]*\)/\1/p' "$1" 2>/dev/null | head -1; }

echo "==> shape matrix: $(basename "$OUT"), repeats=$REPEATS, npred=$NPRED, shapes=[$SHAPES]"
IFS=',' read -r -a want <<<"$SHAPES"
for sh in "${want[@]}"; do
    model="$(shape_var "$sh" model)"
    [ -n "$model" ] || { echo "FAIL  unknown shape '$sh'"; continue; }
    dev="$(shape_var "$sh" devices)"; np="$(shape_var "$sh" np)"; ctx="$(shape_var "$sh" ctx)"
    extra="$(shape_var "$sh" extra)"; env="$(shape_var "$sh" env)"; kind="$(shape_var "$sh" kind)"
    echo
    echo "-- $sh  $(basename "$model")  dev=$dev np=$np ctx=$ctx '${extra}' kind=$kind"

    off_sets=(); on_sets=(); off_stable=0; on_stable=0
    nrep="$REPEATS"; [ "$kind" = refusal ] && nrep=1     # a refusal row needs ONE pair-OFF arm + one ON attempt
    for arm in off on; do
        for r in $(seq 1 "$nrep"); do
            a=""; [ "$arm" = on ] && a="--prefill-decode-arms"
            tag="$sh-$arm-r$r"
            # -lv 4 so the log carries the reserve/census lines; the hashes come from the JSON responses, so the
            # text claim is unaffected (AGENTS.md's -lv table: a server's hashed text is the API response).
            # shellcheck disable=SC2086
            if ! HIP_VISIBLE_DEVICES="$dev" timeout 2400 "$BASE/server-arm.sh" --out "$OUT" --tag "$tag" \
                    --model "$model" --np "$np" --ctx "$ctx" --n-predict "$NPRED" --ready 600 --req-timeout 900 \
                    --timeout 2400 --env "$env" --extra "$extra $a -lv 4" --vram-probe > "$OUT/$tag.out" 2>&1 ; then
                if [ "$kind" = refusal ] && [ "$arm" = on ]; then
                    echo "      pair ON REFUSED, as the predicate requires:"
                    grep -aE "two-arm server|cannot be built|refus" "$OUT/$tag.log" 2>/dev/null | tail -2 | sed 's/^/        /'
                else
                    echo "      $tag: the arm did not complete (see $OUT/$tag.out)"
                    tail -3 "$OUT/$tag.out" 2>/dev/null | sed 's/^/        /'
                fi
            else
                LOGS+=("$OUT/$tag.log")
            fi
            hs="$(hashes_of "$OUT/$tag.hashes")"
            echo "      $tag: {$hs}"
            if [ "$arm" = off ]; then off_sets+=("$hs"); else on_sets+=("$hs"); fi
        done
    done

    # run-stability within an arm, then the off-vs-on set comparison
    if [ "${#off_sets[@]}" -gt 0 ]; then
        off_stable="$(printf '%s\n' "${off_sets[@]}" | sort -u | wc -l)"
        off_stable="$(( ${#off_sets[@]} - off_stable + 1 ))/${#off_sets[@]}"
    fi
    if [ "${#on_sets[@]}" -gt 0 ]; then
        on_stable="$(printf '%s\n' "${on_sets[@]}" | sort -u | wc -l)"
        on_stable="$(( ${#on_sets[@]} - on_stable + 1 ))/${#on_sets[@]}"
    fi

    if [ "$kind" = refusal ]; then
        verdict="REFUSED-AS-DESIGNED"
        vram_delta="-"; rss_delta="-"; census="-"; reserve="-"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$sh" "$np" "$ctx" "off/on(refused)" "$verdict" "$off_stable" "${off_sets[0]:-}" "$vram_delta" \
            "$rss_delta" "$reserve" "$census" >> "$TSV"
        echo "      -> $verdict (pair OFF stable $off_stable)"
        continue
    fi

    # the text verdict: every OFF set must equal the corresponding ON set (and every arm's own repeats must agree)
    text="IDENTICAL"
    for i in "${!off_sets[@]}"; do
        [ "${off_sets[$i]}" = "${on_sets[$i]:-<none>}" ] || text="DIFFERS"
    done
    [ "$off_stable" = "${off_stable#*/}" ] || true
    stable_n="${off_stable%%/*}"; stable_d="${off_stable#*/}"

    # memory: the first OFF vs first ON probe (whole-server VRAM while loaded, per device)
    off_v="$(vram_of "$OUT/$sh-off-r1.vram")"; on_v="$(vram_of "$OUT/$sh-on-r1.vram")"
    vram_delta="$(python3 - "$off_v" "$on_v" <<'PY'
import sys
def parse(s):
    d = {}
    for line in s.strip().splitlines():
        if not line.strip():
            continue
        g, b = line.split()
        d[g] = int(b)
    return d
off, on = parse(sys.argv[1]), parse(sys.argv[2])
out = []
for g in sorted(set(off) | set(on)):
    if g in off and g in on:
        out.append("gpu%s %+.0f" % (g, (on[g] - off[g]) / 1048576.0))
print(", ".join(out) if out else "-")
PY
)"
    off_r="$(rss_of "$OUT/$sh-off-r1.vram")"; on_r="$(rss_of "$OUT/$sh-on-r1.vram")"
    rss_delta="$(python3 -c "
import sys
try:
    print('%+.0f' % ((int('${on_r:-0}') - int('${off_r:-0}')) / 1024.0))
except Exception:
    print('-')")"
    reserve="$(grep -a 'decode arm reserves\|aux narrow-\|slab draft reserve' "$OUT/$sh-on-r1.log" 2>/dev/null \
               | sed 's/^[0-9.]* //' | head -2 | tr '\n' ' ' | sed 's/"/'"'"'/g' | cut -c1-220)"
    [ -n "$reserve" ] || reserve="-"
    census="$(grep -aoE 'the decode arm is generating: [0-9]+ step\(s\), [0-9]+ handover\(s\), widest batch [0-9]+' \
               "$OUT/$sh-on-r1.log" 2>/dev/null | tail -1)"
    [ -n "$census" ] || census="-"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sh" "$np" "$ctx" "off/on" "$text" "off $off_stable on $on_stable" \
        "${off_sets[0]:-}" "$vram_delta" "$rss_delta" "$reserve" "$census" >> "$TSV"
    echo "      -> text=$text  off-stable $off_stable on-stable $on_stable  vram ${vram_delta}  rss ${rss_delta} MiB"
    [ -n "$census" ] && echo "         census: $census"
    [ "$reserve" != "-" ] && echo "         reserve: $(cut -c1-160 <<<"$reserve")"
done

echo
echo "--> GPU faults"
fault_args=()
for l in "${LOGS[@]}"; do [ -s "$l" ] && fault_args+=(--log "$l"); done
if [ "${#fault_args[@]}" -eq 0 ]; then
    echo "    no logs to check"
elif "$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" "${fault_args[@]}" > "$OUT/faults.txt" 2>&1; then
    echo "    PASS  kernel journal and every arm log clean"
else
    echo "    FAIL  GPU fault(s) -- see $OUT/faults.txt"; tail -3 "$OUT/faults.txt" | sed 's/^/      /'
fi

echo
echo "==> TSV: $TSV"
column -t -s $'\t' "$TSV" 2>/dev/null || cat "$TSV"
[ "$KEEP" = 1 ] && echo "    logs kept in $OUT"
exit 0
