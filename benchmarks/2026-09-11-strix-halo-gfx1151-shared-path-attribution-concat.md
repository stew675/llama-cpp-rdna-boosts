# Strix Halo (RDNA3.5 / gfx1151) — remaining pp gap attributed to the shared path; concat port landed

Date: 2026-09-11. Commits: `2f8864cc8` (LLAMA_QSA_OFF gate) + `2bd516bab` (concat port) on
`3cb9168be`. Follows the prefill fix records (2026-09-11-*-ple-host-gather,
-*-managed-ple-batched-fetch).

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

## Remaining shared-path deltas at pp2048 (next, in order)

1. swiglu-input quantize: B fuses `quantize_mmq_q8_1_swiglu` (141 calls, ~0.035 s/pass); A runs
   separate swiglu + quantize (A quantize total ~0.17 s/pass vs B ~0.10 + fused).
2. Q8_0 gate/up mmq J128: A ~+13% on the same 1182 calls/pass (feeding differs).
3. A-only mm_ids_helper<10> launch overhead (423 calls/~0.10 s/pass).
4. Host per-ubatch submit (re-measure now that the concat + PLE splits are gone).
Keep QSA ON (default) throughout; re-check the sparse crossover + depth lead after each landing.
Decode followup (per-op kernel-mix profile) afterwards.

Raw: /tmp/gateA/probe-{Aoff,Aon,B}-2048.txt, concat-{A,B}.txt, concat-cli.txt, /tmp/prof/concat_results.db.
