#!/usr/bin/env bash
# Coherence check for a llama-cli log: print the LAST N characters of the generated text.
#
# Why the tail: a coherence divergence often shows up half-way through generation, so a prefix (or a
# hash taken at `-lv 4`, where extract-generated.py's span also contains the interleaved log lines and
# their timings) can look fine while the run is broken.  Text/hash arms take NO `-lv` at all
# (AGENTS.md, Verify) -- a transport CASUS arm is a separate run with `-lv 4`.
#
#   usage: ./tailcheck.sh LOG [N=200]
set -uo pipefail
LOG=${1:?usage: tailcheck.sh LOG [N]}; N=${2:-200}
HERE=$(dirname "$(readlink -f "$0")")
if grep -qE "^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ [IWED] " "$LOG"; then
  echo "WARNING: this log carries -lv lines inside the token stream; the extracted span is NOT clean." >&2
  echo "         Re-run this arm with no VERBOSE for a text/coherence check." >&2
fi
TXT=$(python3 "$HERE/../../../scripts/extract-generated.py" "$LOG" --text)
printf '%s' "$TXT" | tr -s '[:space:]' ' ' | tail -c "$N"
echo
printf '%s' "$TXT" | tail -c 400 | grep -qE "(.)\1{20,}" && echo "SUSPECT: a character repeated 20+ times in the tail"
