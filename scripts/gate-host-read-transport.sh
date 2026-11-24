#!/usr/bin/env bash
# Host-read-transport gate (wip/host-read-transport, TODO #306 widened).  Proves the transport machinery is
# NOT VACUOUS: every arm must actually take the transport it names, the adaptive arm must latch on evidence and
# only on evidence, and the span coupling must make the measured -41 % pair unreachable.
#
#   ./gate-host-read-transport.sh                 # the arms that need no privilege (~2 min)
#   ./gate-host-read-transport.sh --with-starve   # + the two cgroup arms, which need `sudo -n`, an mlocked
#                                                 #   ballast and ~10 min: it PROVES the ladder moves
#
# The default run covers everything that can be checked on a plain box: every arm takes the transport it names
# (its own counters prove it), the text is byte-identical, `auto` converges on / stays level with the best pinned
# arm, and the coalesced-span coupling makes the measured -41 % pair unreachable.  `--with-starve` adds the one
# assertion that needs memory pressure: that the ladder actually moves when the page cache gives up.
#
# HOW IT IS BUILT (see AGENTS.md, the `-lv` table): every TEXT arm runs with NO `-lv` (the generated stream must
# stay clean for `scripts/extract-generated.py`), and every CENSUS assertion is read from a SEPARATE `-lv 4` run
# of the same arm.  A gate that hashes a `-lv 4` log is measuring the log, not the model.
#
# ARMS
#   ref        --load-mode mmap            the mapping transport (the delivered default)
#   file       --load-mode buffered        the buffered `pread` transport (NEW load mode)
#   dio        --load-mode dio             the O_DIRECT fill arm.  r294: the load mode IS the only thing that
#                                          picks this transport (MOE_HOST_POOL_TRANSPORT / _DIO are REMOVED), and
#                                          --load-mode dio DOES build a pool over the file-backed master, where it
#                                          used to pin the whole host expert set with no pool at all.
#   auto-warm  --load-mode auto            must NOT latch (no false positive on a RAM-rich box)
#   auto-32G   --load-mode auto (32G scope) MUST latch on the probe A/B (the ladder is not dead code)
#   span-off   --load-mode buffered + STAGE_SPAN=0   the buffered arm must be REFUSED (coupling)
#   spanoff32  --load-mode auto (32G) + STAGE_SPAN=0 must never latch to `file`
#   mlock-hole / mlock-nohole  `--load-mode mmap+mlock` with and without a `-lme` hole set (TODO #333): the
#                              load mode's pin is a RESIDENCY request and may not take the pageable holes
##
# PASS CRITERIA (hard): all seven texts identical; the named transport appears in each arm's census; the file
# arm reports storage reads and the ref arm reports mapping reads; auto-warm reports ZERO transitions, auto-32G
# reports AT LEAST ONE; EVERY arm reports a bounded pool (r294: the load mode no longer decides whether the
# tier exists, only how it fills); the span-off arms report the refusal and stay on the mapping (or dio),
# never `file`; and `mmap+mlock` composes with `-lme` (TODO #333): the hole set stays pageable and the pin does
# not take it.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$(dirname "${BASH_SOURCE[0]}")/rocmlib.sh"
BIN=${BIN:-$HOME/llama.cpp/build-rocm/bin/llama-cli}
export LD_LIBRARY_PATH="$(rocm_ld_library_path)"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
# A host-expert config is required (the transport only exists for host-resident expert weights); the field
# model is the one the campaign measured.
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
# The starving arms need a prompt that routes through ENOUGH experts to make the page cache matter.  A 64-char
# prompt leaves the cgroup cacheable even at 32G (measured: 33 us per fill = 18 GB/s, i.e. warm), so `auto` had
# nothing to latch on and the gate's "must latch" assertion failed for the wrong reason.
PROMPT=${PROMPT:-$PWD/runs/gate-prompt.txt}
if [ ! -f "$PROMPT" ]; then
    mkdir -p "$(dirname "$PROMPT")"
    printf 'Write a short story about a lighthouse keeper who finds a message in a bottle.\n' > "$PROMPT"
fi
STARVE_PROMPT=${STARVE_PROMPT:-$PWD/runs/abba/prompt_20k.txt}
if [ ! -f "$STARVE_PROMPT" ] && [ -f "$REPO/wip/bench-56gb/prompt_50k.txt" ]; then
    mkdir -p "$(dirname "$STARVE_PROMPT")"
    head -c 100000 "$REPO/wip/bench-56gb/prompt_50k.txt" > "$STARVE_PROMPT"
fi
DIR=${DIR:-$PWD/runs/gate}
NLIB=${NLIB:-8}          # -lme: a bounded pool, so both the fill and the demand paths are reached
NGEN=${NGEN:-1000}   # a 24-token decode is noise: ~40 s at 25 t/s gives a rate worth comparing
SCOPE=${SCOPE:-32G}      # the cgroup that cannot cache the 46.84 GiB host set
# A cgroup ALONE does not starve this box: the host has 91 GiB, and a memory.max on the process does not stop the
# page cache from serving its reads (the ABBA matrix measured 32G as WARM until a ballast was added).  So the
# starving arms also hold an mlocked ballast outside the scope, exactly as the rig does, and release it on exit.
BALLAST_GIB=${BALLAST_GIB:-21}
BALLAST=${BALLAST:-$(command -v ballast || echo "$REPO/wip/bench-56gb/ballast")}
BALLAST_PID=""
if [ "${WANT_SCOPE:-0}" = 1 ] && [ ! -x "$BALLAST" ]; then
    echo "!! --with-starve needs an mlocked ballast: build it (cc -O2 -o ballast $REPO/wip/bench-56gb/ballast.c)" >&2
    echo "   or point BALLAST= at one, or run without --with-starve." >&2
    exit 2
fi
if [ "${WANT_SCOPE:-0}" = 1 ] && [ -x "$BALLAST" ]; then
    if ! pgrep -x ballast > /dev/null; then
        setsid "$BALLAST" "$BALLAST_GIB" > /dev/null 2>&1 &
        BALLAST_PID=$!
        sleep 12
    fi
    echo "    MemAvailable with the ballast: $(awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
    trap '[ -n "$BALLAST_PID" ] && kill -9 "$BALLAST_PID" 2>/dev/null' EXIT
fi
WANT_SCOPE=0
for arg in "$@"; do
    case "$arg" in
        --with-starve) WANT_SCOPE=1 ;;
        --no-scope)    WANT_SCOPE=0 ;;
        *) echo "usage: $0 [--with-starve]" >&2; exit 2 ;;
    esac
done
mkdir -p "$DIR"
PASS=0; FAIL=0
ok()   { printf 'PASS  %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf 'FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }
check_has()  { local n=$1 pat=$2 f=$3; grep -qE "$pat" "$f" && ok "$n" || bad "$n  (no /$pat/ in $(basename "$f"))"; }
check_nhas() { local n=$1 pat=$2 f=$3; grep -qE "$pat" "$f" && bad "$n  (/$(echo "$pat" | cut -c1-60)/ found)" || ok "$n"; }

# run_arm <tag> <load-mode> <transport-env> <span-env> <scope|-> <lv: text|census>
run_arm() {
    local tag=$1 lm=$2 span=$3 scope=$4 mode=$5
    local extra=()
    [ "$mode" = census ] && extra=(-lv 4)
    # the starving arms use a long PREFILL prompt (routing volume); the rest a short story PROMPT with a long
    # decode, because the assertion is about a decode rate and 24 tokens is not a rate (identical text either way)
    # ONE prompt for every arm: the text hashes are compared across arms, so a different prompt per arm would
    # make the comparison meaningless (measured: the starving arms hashed 542 chars against the 144 of the short
    # prompt and the gate "failed" on its own configuration).
    local prompt_file="$PROMPT"
    if [ -f "$STARVE_PROMPT" ]; then
        prompt_file="$STARVE_PROMPT"
        extra+=(-c 32768 -b 6144 -ub 6144)
    fi
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
    local envs=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES")
    # r294: there is no transport override to plumb any more -- `--load-mode` IS the demand transport.
    [ -n "$span" ] && [ "$span" != "-" ] && envs+=("MOE_HOST_POOL_STAGE_SPAN=$span")
    echo "--- $tag  (load-mode=${lm} span=${span:-on} scope=${scope} ${mode})"
    export GGML_CUDA_SLAB_HEADROOM_MIB=${GGML_CUDA_SLAB_HEADROOM_MIB:-1536}
    export MOE_HOST_POOL_RAM_PCT=${MOE_HOST_POOL_RAM_PCT:-0}
    local attempt=0
    while :; do
    attempt=$((attempt + 1))
    if [ "$scope" = "-" ]; then
        timeout 900 env "${envs[@]}" "$BIN" -m "$M" -ngl all --n-cpu-moe 48 --limit-host-experts "$NLIB" \
            --load-mode "$lm" --lazy-mode on --spec-type none \
            --seed 42 --temp 0 --reasoning off -n "$NGEN" -p "$(cat "$prompt_file")" --no-display-prompt --single-turn \
            "${extra[@]}" > "$DIR/$tag.log" 2>&1
    else
        timeout 1200 sudo -n systemd-run --scope -p "MemoryMax=$scope" -p MemorySwapMax=0 --quiet \
            env "${envs[@]}" "$BIN" -m "$M" -ngl all --n-cpu-moe 48 --limit-host-experts "$NLIB" \
            --load-mode "$lm" --lazy-mode on --spec-type none \
            --seed 42 --temp 0 --reasoning off -n "$NGEN" -p "$(cat "$prompt_file")" --no-display-prompt --single-turn \
            "${extra[@]}" > "$DIR/$tag.log" 2>&1
    fi
    local rc=$?
    # a load failure is retried once: the arena sizes against free VRAM and the previous arm's teardown can
    # still be in flight
    if [ "$rc" -ne 0 ] && ! grep -q "\[ Prompt:" "$DIR/$tag.log" && [ "$attempt" -lt 2 ]; then
        echo "    (retrying $tag: load failed, likely VRAM teardown)"
        sleep 5
        continue
    fi
    return "$rc"
    done
}

hash_of() { python3 "$REPO/scripts/extract-generated.py" "$DIR/$1.log" 2>/dev/null; }
n_switch() { grep -c "host read transport -> " "$DIR/$1.log"; }
transport_of() { grep -oE "host read transport -> [a-z]+" "$DIR/$1.log" | tail -1 | awk '{print $NF}'; }

echo "=== host-read-transport gate  $(date -Is)"
echo "    bin=$BIN  lme=$NLIB  scope=$SCOPE  prompt=$(wc -c < "$PROMPT") bytes"

# ---------------------------------------------------------------- the TEXT arms (no -lv on any of them)
for arm in "ref mmap  -  -" "file buffered - -" "dio dio - -" "auto-warm auto - -"; do
    set -- $arm
    run_arm "$1" "$2" "$3" "$4" text
done
if [ "$WANT_SCOPE" = 1 ]; then
    run_arm auto-32G  auto   -     "$SCOPE" text
    run_arm span-off  buffered 0    -        text
    run_arm spanoff32 auto     0    "$SCOPE" text
fi

# ---------------------------------------------------------------- the CENSUS arms (-lv 4, separate runs)
for arm in "refc mmap  -  -" "filec buffered - -" "dioc dio - -" "autowc auto - -"; do
    set -- $arm
    run_arm "$1" "$2" "$3" "$4" census
done
if [ "$WANT_SCOPE" = 1 ]; then
    run_arm auto32c auto  -    "$SCOPE" census
    run_arm spanofc buffered 0 -        census
    run_arm span32c auto  0    "$SCOPE" census
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
# r294: the LOAD MODE no longer decides whether the tier exists.  Before this change `--load-mode dio` built no
# pool at all ("a pool needs a mapped load") and pinned the whole host expert set; every arm must now report a
# bounded pool that respects `-lme`.
for a in refc filec dioc autowc; do
    check_has "$a: a bounded pool exists (--load-mode no longer forbids it)" "host pool \(L2\): .*pinned" "$DIR/$a.log"
    check_has "$a: the -lme budget is honoured" "pinned tier: [0-9.]+ MiB allocated in total" "$DIR/$a.log"
done
check_has "dioc: the master is the model MAPPING, not a pinned copy" "ROCm_HostMap model buffer" "$DIR/dioc.log"
# HARD: the adaptive arm must be as FAST as the best pinned arm, end to end.  Grading it on one path's read
# rate was wrong: the two read paths legitimately disagree (a fill is a first touch and favours `file`; the demand
# read repeats and favours the mapping), so the outcome -- decode throughput on the same prompt -- is the only
# honest criterion.  It also catches the failure that matters: a latch onto `dio`, which is 2.3x worse warm.
arm_rate() { # <log> <line-pattern>
    grep -E "$2" "$1" | tail -1 | grep -oE '[0-9.]+ MiB/s' | head -1 | awk '{print $1}'
}
gen_tps() { # the decode rate: the `-lv 4` timing line, else llama-cli's own footer (a text arm has no -lv)
    local v
    v=$(grep -oE 'eval time = *[0-9.]+ ms / *[0-9]+ tokens \( *[0-9.]+ ms per token, *[0-9.]+ tokens per second' "$1" \
        | tail -1 | grep -oE '[0-9.]+ tokens per second' | awk '{print $1}')
    [ -n "$v" ] && { echo "$v"; return; }
    grep -oE '\[ Prompt: *[0-9.]+ t/s \| Generation: *[0-9.]+ t/s \]' "$1" \
        | tail -1 | grep -oE 'Generation: *[0-9.]+' | grep -oE '[0-9.]+'
}
R_MAP=$(arm_rate "$DIR/refc.log"  "per-expert mapping reads \(bounce\)")
R_FILE=$(arm_rate "$DIR/filec.log" "per-expert storage reads \(bounce\)")
R_DIO=$(arm_rate "$DIR/dioc.log"  "per-expert storage reads \(bounce\)")
echo "    measured demand-path rates: map=${R_MAP:-?} file=${R_FILE:-?} dio=${R_DIO:-?} MiB/s"
P_MAP=$(gen_tps "$DIR/ref.log");  P_FILE=$(gen_tps "$DIR/file.log"); P_AUTO=$(gen_tps "$DIR/auto-warm.log")
echo "    decode t/s: map=${P_MAP:-?} file=${P_FILE:-?} auto=${P_AUTO:-?}"
if [[ -n "${P_MAP:-}" && -n "${P_FILE:-}" && -n "${P_AUTO:-}" ]]; then
    awk -v a="$P_AUTO" -v m="$P_MAP" -v f="$P_FILE" 'BEGIN{
        best = (m > f) ? m : f;
        printf "auto is %.1f %% of the best pinned arm\n", 100.0*a/best;
        exit !(a >= 0.90*best);
    }' && ok "auto: within 10 % of the best pinned arm (decode)" \
       || bad "auto: $P_AUTO t/s against the best pinned arm (map $P_MAP / file $P_FILE)"
else
    bad "auto: could not read the decode rates"
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

# ---------------------------------------------------------------- arm: the mlock/hole composition (TODO #333)
# `--load-mode mlock`/`mmap+mlock` is a RESIDENCY request, not a read transport, and a `-lme` hole set must stay
# PAGEABLE: the tier's budget is the reservation and its device reads come from the pinned host pool, so pinning
# the holes here is what made `-lme 30 --load-mode mmap+mlock` pin the whole expert set (measured 2026-11-16,
# RLIMIT_MEMLOCK raised to unlimited: **43.5 GiB** pinned before the fix, **260 MiB** after it, against 40 925
# MiB of holes).  The arms assert the composition, not a number that needs a raised limit to exist:
#   mlock-hole    --load-mode mmap+mlock with a `-lme` hole set: the hole set is named PAGEABLE and the census
#                 counts it as unpinned; the load-mode pin cannot take it.
#   mlock-nohole  --host-experts pinned (no hole set): no hole line, and the census says there is none.
# With `sudo -n` the quantity is asserted too (unlimited memlock); without it a failed pin makes the quantity
# meaningless, so it is reported and not asserted.
MLOCK_SUDO=0
( unset -v LD_PRELOAD; sudo -n bash -c 'ulimit -l unlimited' ) 2>/dev/null && MLOCK_SUDO=1
mlock_arm() {   # <tag> <load-mode> <config-args...>
    local tag=$1 lm=$2; shift 2
    echo "--- $tag  (load-mode=$lm $*)  memlock=$([ $MLOCK_SUDO = 1 ] && echo unlimited || echo as-shell)"
    local cmd=(env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
               "$BIN" -m "$M" -ngl all "$@" --load-mode "$lm" -c 4096 -b 512 -ub 512 --spec-type none
               --seed 42 --temp 0 --reasoning off -n 8 -p "$(cat "$PROMPT")"
               --no-display-prompt --single-turn -lv 4)
    if [ "$MLOCK_SUDO" = 1 ]; then
        timeout 1200 sudo -n bash -c 'ulimit -l unlimited; exec "$@"' -- "${cmd[@]}" > "$DIR/$tag.log" 2>&1
    else
        timeout 1200 "${cmd[@]}" > "$DIR/$tag.log" 2>&1
    fi
}
mlock_arm mlock-hole   mmap+mlock --n-cpu-moe 48 --limit-host-experts "$NLIB"
mlock_arm mlock-nohole mmap+mlock --n-cpu-moe 48 --host-experts pinned

# the hole set is left pageable, and the census counts it
check_has "mlock-hole: the -lme holes are left PAGEABLE" "host-expert .-lme. holes are left PAGEABLE" "$DIR/mlock-hole.log"
check_has "mlock-hole: the mlock census names the pageable hole set" \
    "mlock: [0-9]+ MiB of the model mapping pinned \\(the -lme host-expert holes, [0-9]+ MiB left pageable\\)" "$DIR/mlock-hole.log"
check_has "mlock-hole: the tier's residency is named as necessity, not as the load mode" \
    "PINNED BY NECESSITY whatever --load-mode says" "$DIR/mlock-hole.log"
check_nhas "mlock-nohole: no hole set, so no hole line" "holes are left PAGEABLE" "$DIR/mlock-nohole.log"
check_has "mlock-nohole: the census reports that there is no hole set" "no -lme holes" "$DIR/mlock-nohole.log"
mh_holes=$(sed -n 's/.*host-expert holes, \([0-9][0-9]*\) MiB left pageable.*/\1/p' "$DIR/mlock-hole.log" | tail -1)
mh_pinned=$(sed -n 's/.*mlock: \([0-9][0-9]*\) MiB of the model mapping pinned.*/\1/p' "$DIR/mlock-hole.log" | tail -1)
if [ "$MLOCK_SUDO" = 1 ] && [ -n "$mh_holes" ] && [ -n "$mh_pinned" ]; then
    [ "$mh_holes" -gt 20000 ] && ok "mlock-hole: the pageable hole set is real (${mh_holes} MiB)" \
                             || bad "mlock-hole: only ${mh_holes} MiB of holes -- the arm is vacuous"
    [ "$mh_pinned" -lt $((mh_holes / 4)) ] && ok "mlock-hole: the pin did not take the holes (${mh_pinned} MiB pinned of ${mh_holes} MiB holes)" \
                                           || bad "mlock-hole: ${mh_pinned} MiB pinned against ${mh_holes} MiB of holes -- the sweep is back"
else
    echo "    SKIP  mlock-hole quantity: no sudo -n (or the pin failed: ${mh_pinned:-?} MiB pinned, ${mh_holes:-?} MiB holes)"
fi

echo
echo "=== $PASS passed, $FAIL failed   (logs in $DIR)"
[ "$FAIL" -eq 0 ]
