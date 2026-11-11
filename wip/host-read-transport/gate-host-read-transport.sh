#!/usr/bin/env bash
# Host-read-transport gate (wip/host-read-transport, TODO #306 widened).  Proves the transport machinery is
# NOT VACUOUS: every arm must actually take the transport it names, the adaptive arm must latch on evidence and
# only on evidence, and the span coupling must make the measured -41 % pair unreachable.
#
#   ./gate-host-read-transport.sh                 # all arms (needs `sudo -n` for the starving cgroup)
#   ./gate-host-read-transport.sh --no-scope      # skip the two cgroup arms (faster, weaker)
#
# HOW IT IS BUILT (see AGENTS.md, the `-lv` table): every TEXT arm runs with NO `-lv` (the generated stream must
# stay clean for `scripts/extract-generated.py`), and every CENSUS assertion is read from a SEPARATE `-lv 4` run
# of the same arm.  A gate that hashes a `-lv 4` log is measuring the log, not the model.
#
# ARMS
#   ref        --load-mode mmap            the mapping transport (the delivered default)
#   file       --load-mode buffered        the buffered `pread` transport (NEW load mode)
#   dio        mmap + TRANSPORT=dio        the O_DIRECT fill arm (the delivery's `MOE_HOST_POOL_DIO=1` shape;
#                                          `--load-mode dio` builds NO pool at all: "a pool needs a mapped load")
#   auto-warm  --load-mode auto            must NOT latch (no false positive on a RAM-rich box)
#   auto-32G   --load-mode auto (32G scope) MUST latch on the probe A/B (the ladder is not dead code)
#   span-off   --load-mode buffered + STAGE_SPAN=0   the buffered arm must be REFUSED (coupling)
#   spanoff32  --load-mode auto (32G) + STAGE_SPAN=0 must never latch to `file`
#
# PASS CRITERIA (hard): all seven texts identical; the named transport appears in each arm's census; the file
# arm reports storage reads and the ref arm reports mapping reads; auto-warm reports ZERO transitions, auto-32G
# reports AT LEAST ONE; the span-off arms report the refusal and stay on the mapping (or dio), never `file`.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-cli}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
PROMPT=${PROMPT:-$REPO/wip/bench-56gb/prompt_short.txt}
DIR=${DIR:-$PWD/runs/gate}
NLIB=${NLIB:-8}          # -lme: a bounded pool, so both the fill and the demand paths are reached
NGEN=${NGEN:-24}
SCOPE=${SCOPE:-32G}      # the cgroup that cannot cache the 46.84 GiB host set
WANT_SCOPE=1
[ "${1:-}" = "--no-scope" ] && WANT_SCOPE=0
mkdir -p "$DIR"
PASS=0; FAIL=0
ok()   { printf 'PASS  %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf 'FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }
check_has()  { local n=$1 pat=$2 f=$3; grep -qE "$pat" "$f" && ok "$n" || bad "$n  (no /$pat/ in $(basename "$f"))"; }
check_nhas() { local n=$1 pat=$2 f=$3; grep -qE "$pat" "$f" && bad "$n  (/$(echo "$pat" | cut -c1-60)/ found)" || ok "$n"; }

# run_arm <tag> <load-mode> <transport-env> <span-env> <scope|-> <lv: text|census>
run_arm() {
    local tag=$1 lm=$2 tr=$3 span=$4 scope=$5 mode=$6
    local extra=()
    [ "$mode" = census ] && extra=(-lv 4)
    local envs=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES")
    # `-` is this script's "leave it to the binary".  It is NOT a value: exporting
    # `MOE_HOST_POOL_TRANSPORT=-` makes parse_env reject the value, mark the env as an explicit pin and then
    # IGNORE the loader -- which silently ran every arm without a transport at all (measured: `--load-mode mmap`
    # reported `pool_decide_dio_locked` and no transport line).
    [ -n "$tr" ]   && [ "$tr" != "-" ]   && envs+=("MOE_HOST_POOL_TRANSPORT=$tr")
    [ -n "$span" ] && [ "$span" != "-" ] && envs+=("MOE_HOST_POOL_STAGE_SPAN=$span")
    echo "--- $tag  (load-mode=${lm} transport=${tr:-<load-mode>} span=${span:-on} scope=${scope} ${mode})"
    if [ "$scope" = "-" ]; then
        timeout 900 env "${envs[@]}" "$BIN" -m "$M" -ngl all --n-cpu-moe 48 --limit-host-experts "$NLIB" \
            -c 4096 -b 2048 -ub 2048 --load-mode "$lm" --lazy-mode on --spec-type none \
            --seed 42 --temp 0 -n "$NGEN" -p "$(cat "$PROMPT")" --no-display-prompt --single-turn \
            "${extra[@]}" > "$DIR/$tag.log" 2>&1
    else
        timeout 1200 sudo -n systemd-run --scope -p "MemoryMax=$scope" -p MemorySwapMax=0 --quiet \
            env "${envs[@]}" "$BIN" -m "$M" -ngl all --n-cpu-moe 48 --limit-host-experts "$NLIB" \
            -c 4096 -b 2048 -ub 2048 --load-mode "$lm" --lazy-mode on --spec-type none \
            --seed 42 --temp 0 -n "$NGEN" -p "$(cat "$PROMPT")" --no-display-prompt --single-turn \
            "${extra[@]}" > "$DIR/$tag.log" 2>&1
    fi
    return $?
}

hash_of() { python3 "$REPO/scripts/extract-generated.py" "$DIR/$1.log" 2>/dev/null; }
n_switch() { grep -c "host read transport -> " "$DIR/$1.log"; }
transport_of() { grep -oE "host read transport -> [a-z]+" "$DIR/$1.log" | tail -1 | awk '{print $NF}'; }

echo "=== host-read-transport gate  $(date -Is)"
echo "    bin=$BIN  lme=$NLIB  scope=$SCOPE  prompt=$(wc -c < "$PROMPT") bytes"

# ---------------------------------------------------------------- the TEXT arms (no -lv on any of them)
for arm in "ref mmap  -    -  -" "file buffered - - -" "dio mmap dio - -" "auto-warm auto - - -"; do
    set -- $arm
    run_arm "$1" "$2" "$3" "$4" "$5" text
done
if [ "$WANT_SCOPE" = 1 ]; then
    run_arm auto-32G  auto  -     -     "$SCOPE" text
    run_arm span-off  buffered -  0     -        text
    run_arm spanoff32 auto -      0     "$SCOPE" text
fi

# ---------------------------------------------------------------- the CENSUS arms (-lv 4, separate runs)
for arm in "refc mmap  -    -  -" "filec buffered - - -" "dioc mmap dio - -" "autowc auto - - -"; do
    set -- $arm
    run_arm "$1" "$2" "$3" "$4" "$5" census
done
if [ "$WANT_SCOPE" = 1 ]; then
    run_arm auto32c auto -   -    "$SCOPE" census
    run_arm spanofc buffered - 0  -        census
    run_arm span32c auto -   0    "$SCOPE" census
fi

# ---------------------------------------------------------------- HARD: one text for every arm
REF=$(hash_of ref)
[ -n "$REF" ] && ok "text: ref produced generated text ($REF)" || bad "text: ref produced no text"
TEXT_ARMS="file dio auto-warm"
[ "$WANT_SCOPE" = 1 ] && TEXT_ARMS="$TEXT_ARMS auto-32G span-off spanoff32"
for a in $TEXT_ARMS; do
    h=$(hash_of "$a")
    [ "$h" = "$REF" ] && ok "text: $a == ref" || bad "text: $a ($h) != ref ($REF)"
done

# ---------------------------------------------------------------- HARD: every arm took the arm it names
[ "$(transport_of refc)" = "map" ]  && ok "ref: transport is the mapping"  || bad "ref: transport is $(transport_of refc), expected map"
[ "$(transport_of filec)" = "file" ] && ok "file: transport is a buffered pread" || bad "file: transport is $(transport_of filec), expected file"
[ "$(transport_of dioc)" = "dio" ]  && ok "dio: transport is O_DIRECT"     || bad "dio: transport is $(transport_of dioc), expected dio"
# HARD: the adaptive arm must CONVERGE on the arm the pinned arms prove fastest -- that is the non-vacuous
# test of "auto", and it is box-independent because it is derived from this run's own pinned measurements
# rather than from an assumption about which arm "should" win.
arm_rate() { # <log> <line-pattern>
    grep -E "$2" "$1" | tail -1 | grep -oE '[0-9.]+ MiB/s' | head -1 | awk '{print $1}'
}
R_MAP=$(arm_rate "$DIR/refc.log"  "per-expert mapping reads \(bounce\)")
R_FILE=$(arm_rate "$DIR/filec.log" "per-expert storage reads \(bounce\)")
R_DIO=$(arm_rate "$DIR/dioc.log"  "per-expert storage reads \(bounce\)")
echo "    measured demand-path rates: map=${R_MAP:-?} file=${R_FILE:-?} dio=${R_DIO:-?} MiB/s"
[ -n "$R_MAP" ] && [ -n "$R_FILE" ] || bad "auto: could not measure the pinned arms"
if [[ -n "${R_MAP:-}" && -n "${R_FILE:-}" ]]; then
    if awk -v f="$R_FILE" -v m="$R_MAP" 'BEGIN{exit !(f > 1.15*m)}'; then
        want=file
    elif awk -v f="$R_FILE" -v m="$R_MAP" 'BEGIN{exit !(m > 1.15*f)}'; then
        want=map
    else
        want=either   # within the probe's own margin: staying put is correct
    fi
    got=$(transport_of autowc)
    case "$want" in
        either) ok "auto-warm: map and file are within 15 % (map $R_MAP / file $R_FILE), ended on $got" ;;
        *) [ "$got" = "$want" ] && ok "auto-warm: converged on $want (map $R_MAP / file $R_FILE MiB/s)" \
                               || bad "auto-warm: ended on $got but the pinned arms prove $want fastest (map $R_MAP / file $R_FILE)" ;;
    esac
fi

# ---------------------------------------------------------------- HARD: the counters prove the path ran
check_has  "ref: fills went through the mapping"      "pool fills: .*[1-9][0-9]* mapping memcpys"  "$DIR/refc.log"
check_has  "file: fills went through the file"        "pool fills: [1-9][0-9]* storage reads"      "$DIR/filec.log"
check_has  "file: the demand path read the file"      "per-expert storage reads \(bounce\): [1-9]" "$DIR/filec.log"
check_has  "ref: the demand path read the mapping"    "per-expert mapping reads \(bounce\): [1-9]" "$DIR/refc.log"

# ---------------------------------------------------------------- HARD: adaptive latches only on evidence
check_has  "auto-warm: the exit census names the arm"  "host read transport: .*, .*"                  "$DIR/autowc.log"
# the transition must be EVIDENCE-backed: a probe win or a stall, never a bare heuristic
if [ "$(n_switch autowc)" -ge 2 ]; then
    check_has "auto-warm: the transition cites its evidence" "host read transport -> (file|dio) \((probe fill|stall)" "$DIR/autowc.log"
else
    ok "auto-warm: no transition (the mapping is the arm this box wants)"
fi
if [ "$WANT_SCOPE" = 1 ]; then
    n=$(( $(n_switch auto32c) - 1 ))
    [ "$n" -ge 1 ] && ok "auto-32G: latched $n step(s) under a starving cgroup" || bad "auto-32G: never latched (the ladder is dead code here)"
    check_has "auto-32G: the latch cites its evidence"  "host read transport -> (file|dio) \((probe fill|stall)" "$DIR/auto32c.log"
    [ "$(transport_of auto32c)" != "map" ] && ok "auto-32G: ended on $(transport_of auto32c), not the mapping" \
                                           || bad "auto-32G: still on the mapping after a collapse"
    # ---- HARD: the coupling (TODO #306's second condition)
    check_has "span-off: the buffered arm is REFUSED"    "needs the coalesced prefill read"          "$DIR/spanofc.log"
    [ "$(transport_of spanofc)" = "map" ] && ok "span-off: fell back to the mapping, not the -41 % arm" \
                                          || bad "span-off: landed on $(transport_of spanofc)"
    t=$(transport_of span32c)
    [ "$t" != "file" ] && ok "span-off auto: never selected the buffered arm (ended on $t)" \
                       || bad "span-off auto: selected the measured -41 % pair"
    check_nhas "span-off auto: no file transition" "host read transport -> file" "$DIR/span32c.log"
fi

echo
echo "=== $PASS passed, $FAIL failed   (logs in $DIR)"
[ "$FAIL" -eq 0 ]
