#!/usr/bin/env bash
# Prefill fairness gate: `llama-server --parallel N` must share the batch window fairly (block 12, r300).
#
# WHY THIS EXISTS
#   Every other gate in this repo runs ONE request at a time, and `llama-batched-bench` measures throughput,
#   which cannot see a latency distribution.  The defect this gate covers needs two sessions: update_slots()
#   adds the decode tokens and then fills the batch in SLOT INDEX order, so the lowest slot took the whole
#   window and every other prompt waited for that slot's ENTIRE prompt (measured: a 7-token prompt behind a
#   13204-token one waited 1.91 s), while a decoding slot inherited the other slot's prefill chunk as its
#   inter-token latency (10 ms -> 266-331 ms).  `-b` does not help: it changes the grab size, not the order.
#
# WHAT IT CHECKS  (two arms on the same binary, `-np 2`, one GPU)
#   arm `baseline`   LLAMA_SERVER_PREFILL_FAIR=0 CHUNK=0    the old slot-order fill
#   arm `candidate`  FAIR=1 CHUNK=<ubatch>                  the max-min fair share + a per-step cap
#   HARD  short-prompt TTFT improves >= --min-ttft-ratio (3x)
#   HARD  worst decoder ITL <= baseline_max x (chunk/n_batch) x 1.75 (and x0.75 on a streaming KV, where a
#         per-step spill read does not scale with the chunk)
#   HARD  a lone long prompt's prefill within --max-prefill-loss (3 %) of the baseline
#   HARD  ONE request's greedy text byte-identical between the arms (the single-slot oracle; a two-slot pair
#         is NOT a text oracle -- co-batching changes the ubatch shape, GREEDY-PURITY.md §63)
#   HARD  each arm's startup census states the configuration it claims
#   HARD  no abort/assert, and scripts/check-gpu-faults.sh clean with --since bounded to this run
#   SOFT  aggregate decode within --max-decode-loss (5 %); skipped when unmeasurable
#
#   --self-test runs the candidate AS the baseline and requires the HARD fairness checks to FAIL: a gate that
#   cannot reject the unfixed build is not a gate.  Run it before trusting a green run.
#
# Usage:  ./gate-prefill-fairness.sh [--quick] [--keep] [--self-test] [--server PATH] [--model PATH]
#                                    [--gpu 0] [--server-args "-ncmoe 47"] [--chunk N] ...
# Exit:   0 = pass, 1 = fail, 2 = usage/prerequisite error.
#
# Env: LLAMA_SERVER (binary override), plus everything the python helper documents (--help).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/gate-prefill-fairness.py" "$@"
