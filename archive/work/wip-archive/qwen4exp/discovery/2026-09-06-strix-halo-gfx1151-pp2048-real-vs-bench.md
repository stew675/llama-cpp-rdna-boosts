# Strix Halo (RDNA3.5 / gfx1151) — pp2048 deficit: llama-bench inflation vs the real path

SUPERSEDED IN FRAMING by `2026-09-06-strix-halo-gfx1151-prefill-rootcause-investigation.md`
(the measurements stand; the conclusion is corrected: the llama-bench-only CPU input-embedding
cost is a REAL A-side ingest-path defect to fix, not a protocol excuse — B's embedding path is
fast in both bench and cli).

Date: 2026-09-05 session. Follow-up to the equal/surpass-B gap re-derivation
(`2026-09-05-strix-halo-gfx1151-ab-gap-default-on.md`), which showed the depth-0 shallow-row
llama-bench gaps (pp512-4096, worst pp2048 1.93x). This record splits that deficit into what is
real (serving path) vs what llama-bench inflates, and pins the remaining real components.

## What was measured (same session, A default at a18e24f97 / B c7af5c6c2)

- llama-bench pp2048 (protocol `--load-mode none`): A ~400 t/s vs B ~771 t/s (1.93x) — but see below.
- llama-cli pp2048 (same flags, both; sync per decode = the real serving path): A 494.8 t/s vs
  B 653.0 t/s -> **1.32x real gap**. A's llama-cli rate is the same with and without
  `--load-mode none` (494.3 vs 494.8) -> load-mode is not the trigger.

## Where llama-bench's extra A penalty comes from (instrumented, since reverted)

A's pp2048 decode is TWO sched splits: split 0 = CPU (4 nodes, `model.input_embed` = the PLE
embedding get_rows) and split 1 = GPU (7201 nodes). Per-decode wall (env-gated timing on
ggml_backend_sched_compute_splits + llama_context::decode, reverted after measurement):

- A pp2048: CPU split0 compute_async = **2700 ms** (!!) per decode; GPU split1 submit = 10-220 ms;
  GPU executes ~2.9 s -> llama-bench rep wall ~5.6 s (matches its ~350-400 t/s).
- The same instrumentation at pp8192/16384 shows per-ubatch compute ~2.7-3.3 s (multi-ubatch
  decodes pipeline, so the cost structure differs there and the deficit per ubatch is smaller).
- B's equivalent (not instrumented - B is untouched): total rep 2.66 s with ~2.4 s GPU busy ->
  B's host submit is ~0.25 s/ubatch, i.e. B does NOT pay a ~2.7 s host-side embedding.

So the single-ubatch llama-bench rows (pp512/1024/2048) on A carry a llama-bench-specific
host-side input-embedding cost (~2.7 s at 2048 tokens, scaling with the ubatch) that (a) does
not appear on llama-cli with identical flags and (b) does not exist on B. The llama-bench
shallow-row gap (1.58-1.93x) is therefore NOT the true serving gap: the real pp2048 gap is
1.32x. Root cause of the 2.7 s CPU get_rows is still OPEN (needs a CPU-side get_rows/PLE-lazy
profile; suspects: the per-layer PLE row path under llama-bench's random-token decode + the
hybrid-memory per-token bookkeeping; note the CPU split is only 4 nodes so the time is inside
the get_rows op itself).

## Remaining REAL pp2048 deficit (kernel-level, rocprof A vs B, same protocol)

A total GPU-busy 8.83 s vs B 7.24 s over the same ~3 decodes (+22%) - the wall delta is mostly
host-side (above), but the kernel deltas worth porting/optimizing (~0.5 s/pass of GPU time):

- concat: A `concat_non_cont` 0.41 s/111 calls vs B `concat_transposed_src1_dim0` 0.13 s/111 ->
  B has a specialized transposed-src1 concat kernel, A's generic one is ~3x slower.
- A-only `mm_ids_helper<10>` 423 calls / 0.31 s (expert-id helper on A's path; B has no
  equivalent launch overhead).
- quantize: A 0.50 s (1608 calls) vs B 0.29 s (1326) + B's fused `quantize_mmq_q8_1_swiglu`
  (141 calls, 0.11 s) - B fuses the expert-input swiglu quantize.
- MoE reduction tail: A `moe_weighted_reduction_f32` 0.30 s/144 vs B `weighted_expert_sum_f32`
  0.15 s/141 (~2x; B's is fused into the IQ4_NL-weighted kernel for the covered layers).
- Q8_0 gate/up mmq J128: A 1.56 s vs B 1.38 s on identical 1182 calls (+13% - same kernel,
  suggests A's build/quantize-feeding of it differs).

Near-parity already: IQ4_NL/IQ3_S routed-compact mmq (A == B), rocBLAS GEMMs (+6%), dense FA
(+18% - A's shortcut dense FA vs B's), hc_combine/mix (A +17% on hc_combine).

## Actionable conclusions

1. Re-measure shallow-row A-vs-B on the SERVING path (llama-cli) for the real gap story; the
   llama-bench single-ubatch rows on A are protocol-inflated until the CPU input-embedding cost
   is root-caused (next session: profile A's CPU get_rows/PLE path under llama-bench decode;
   check why llama-cli avoids it).
2. Real pp2048 gap = 1.32x (494.8 vs 653.0). Next port candidates in order: B's transposed
   concat kernel (3x on a 0.4 s/pass item), the swiglu-quantize fusion, and the Q8_0 gate/up
   mmq feeding difference (+13% on the single biggest kernel, 1.56 s/pass).
3. Depth-0 pp8192/16384 gaps (1.25x/1.04x) and all depth rows are unaffected by this artifact
   (multi-ubatch) and remain as recorded in the gap record.

Raw: /tmp/gateA/bnd-*.log, ub-*.log, cli-pp2048-{A,B}*.txt, /tmp/prof/pp{A,B}_results.db.
