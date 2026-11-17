#!/usr/bin/env bash
# Prefill/decode duty-cycle gate: a decoder sharing a server with a prompt keeps a burst of tokens
# (block 16, the server's prompt admission policy).
#
# WHY THIS EXISTS
#   A decoding slot gets one token per step, so its rate is 1/step_time -- and the step that carries prompt
#   tokens is the slow one.  Measured on the field shape (Flash-Next, 3 GPU tensor, one 16k step): a decoder
#   sharing the server with a 20k-token prefill delivered 5 tokens over 6.0 s at a 4.3 s worst gap.  r300's
#   per-step chunk bounds that stall; the duty cycle leaves most steps decode-only and puts the prompt into
#   one step per cycle, so the decoder gets `K-1` tokens at the decode-only step time.
#
#   It is a TRADE, so the gate asserts BOTH sides: the decode burst is real, AND the prefill's own wall time
#   stays inside the bound the design claims (the idle time is capped by the previous prefill step's own
#   duration, so the analytic bound is 2x).  A duty cycle that is only checked for its latency win is a
#   prefill slowdown with a good median.
#
# WHAT IT CHECKS  (two arms on one binary, `-np 2`, one GPU, both slots pinned)
#   arm `baseline`   DUTY=0
#   arm `candidate`  DUTY=<--duty> (16 = the shipped default)
#   HARD  decode tokens delivered while the other slot prefills >= 3x the baseline
#   HARD  the concurrent prefill's wall time <= 2.5x the baseline's (2x analytic + one uncalibrated cycle)
#   HARD  a lone prompt (nothing decoding) prefilled within 3 % -- the deferral never fires into an empty batch
#   HARD  one request's greedy text byte-identical between the arms
#   HARD  each arm's census states its duty; 0 "no tokens to decode" lines; no abort
#   HARD  with --spec-args: the gain holds under the drafter and acceptance does not collapse
#   SOFT  the decoder's worst gap during the prefill is not worse than the baseline's
#   plus scripts/check-gpu-faults.sh with --since bounded to this run.
#
# Usage:  ./gate-prefill-duty.sh [--quick] [--keep] [--duty N] [--server PATH] [--model PATH] [--gpu 0]
#                                [--server-args "-ncmoe 47"]
#                                [--spec-args "--spec-type draft-mtp --spec-draft-n-max 3"]
# The release runs it on the dense 4B shape and on the 27B + draft-mtp shape.
# Exit:   0 = pass, 1 = fail, 2 = usage/prerequisite error.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/gate-prefill-duty.py" "$@"
