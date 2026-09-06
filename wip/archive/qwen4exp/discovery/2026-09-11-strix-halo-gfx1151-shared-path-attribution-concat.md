# Strix Halo (RDNA3.5 / gfx1151) — remaining pp gap attributed to the shared path; concat + swiglu-quantize ports landed

Date: 2026-09-11. Commits: `2f8864cc8` (LLAMA_QSA_OFF gate) + `2bd516bab` (concat port) +
`7a6a2e97b` (fused swiglu-input quantize) on `3cb9168be`. Follows the prefill fix records
(2026-09-11-*-ple-host-gather, -*-managed-ple-batched-fetch).

## Finding: both A and B run the SAME QSA architecture; the remaining gap is the shared path

A surprise check (B has lightning-indexer.cu + flash_attn_qsa + the same
LLAMA_QSA_DENSE_SHORTCUT env) corrects the earlier working assumption that A's depth lead came
from having QSA where B did not. Corrected model: both trees implement the qwen4exp QSA
sparse-attention architecture; A's sparse path wins at LARGE n_kv (depth-0 pp16384 A 626 vs B
600 and A wins the 32k rows; B's rate collapses 776->734->679->600 as pp grows 2048->16384 while
A stays ~flat 655->646->637->626), while B wins the SMALL/mid rows where the shared path and
launch overhead dominate.

New gate knob LLAMA_QSA_OFF=1 forces the dense no-indexer regime (no store/scoring/sparse at
any layer) - a plain-dense reference, default OFF. Probe (same session, depth-0 pp2048): A-on
590 == A-off 589, B 754 -> the ~1.28x dense-regime gap is entirely the SHARED path (projections,
MoE, norms, concat, host/launch); the indexer store is ~free and QSA is not the deficit.

## Concat port (2bd516bab)

The recurrent conv-state concat (state [hist,C] || transpose(x) [T,C] along dim 0) leaves src1
with the transposed layout (nb1 == sizeof, nb0 == ne1*sizeof); A's generic concat_non_cont ran
it at ~0.14 s/pass (pp2048) vs B's ~0.045 s/pass. Ported B's concat_transposed_src1_dim0
(shared-memory tile transpose, tile_y 16 on RDNA3.5 else 8) + layout guard in concat_cuda.
Fires (rocprof: concat_transposed_src1_dim0 74 calls / 0.109 s over the 2-decode capture at
pp2048; concat time ~0.055 s/pass). Text byte-identical (same-seed cli). Same-session depth-0
r3 A-vs-B (warm order 8192,4096,2048):

| row | A now | A before | B | gap now | gap before |
|---|---|---|---|---|---|
| pp8192 | 631.7 (noisy +/-28) | 637.0 | 678.5 | ~0.93x | 0.94x |
| pp4096 | 661.4 | 645.8 | 731.6 | 0.90x | 0.88x |
| pp2048 | 672.4 | 654.6 | 770.8 | 0.87x | 0.84x |

The concat recovery (~0.08 s/pass, the expected amount once the multi-pass profile span is
accounted for) shows up as +2.4-2.7% at pp4096/2048.

## Swiglu-input quantize port (7a6a2e97b)

The qwen4exp MoE down-feed (silu(gate)*up -> quantize_mmq_q8_1 for the 512-expert mmq) ran as
a materialized f32 GLU (gated-silu kernel) + a separate quantize pass. Ported B's fusion: a
try_fuse matcher on SWIGLU GLU nodes whose next node is the qwen4exp/Qwen3.6-shape expert
mul_mat/mul_mat_id (weights Q8_0 or qwen4exp IQ4_NL, RDNA3.5 + mmq heuristics) calls
ggml_cuda_mul_mat_q_swiglu; the quantize step computes silu(gate)*up inline (per-expert
scatter via ids_src1) with no GLU materialization. swiglu threaded as an optional param
through ggml_cuda_mul_mat_q alongside the existing epilogue-gate fusion args. mmq only
engages at prefill, so decode's MoE-tail fusions are untouched. Fires (rocprof pp2048:
quantize_mmq_q8_1_swiglu 188 calls / ~94 per pass; the gated-silu GLU kernels are gone); text
byte-identical. Same-session depth-0 r3 vs B: pp2048 675.0 vs 770.8, pp4096 667.0 vs 731.6,
pp8192 ~flat. Win is smaller than concat's (~0.4-0.8%: the net is the GLU f32 write/read +
one launch; the quantize itself still runs).

## Remaining shared-path deltas at pp2048 (next, in order)

1. RE-DERIVE on fresh same-session rocprof pair (A-now vs B-now at pp2048) - the earlier
   per-kernel deltas (concat/swiglu/Q8_0 feed) were measured across separate sessions and
   pre-fix builds. Then attack whatever is largest: the Q8_0 gate/up mul_mat_q J128 (+13%,
   ~520 ms/pass - largest single kernel - needs confirming), mm_ids_helper launches, host
   per-ubatch submit.
2. Keep QSA ON (default) throughout; re-check the sparse crossover + depth lead after each
   landing. Decode followup (per-op kernel-mix profile) afterwards.

Raw: /tmp/gateA/probe-{Aoff,Aon,B}-2048.txt, concat-{A,B}.txt, concat-cli.txt, /tmp/prof/concat_results.db.
