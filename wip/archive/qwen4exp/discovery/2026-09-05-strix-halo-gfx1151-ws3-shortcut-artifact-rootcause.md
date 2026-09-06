# Strix Halo (RDNA3.5 / gfx1151) — WS3 #2 llama-bench artifact: ROOT CAUSE FOUND

Date: 2026-09-05 session. Investigation of the OPEN item from SESSION-BRIEF-2026-09-05-s3 /
wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-post-fusion-gap.md (WS3 #2 section): with the QSA
dense-shortcut ON (LLAMA_QSA_DENSE_SHORTCUT=1, A commit 151798ed2), llama-bench depth-0
pp4096/8192/16384 rows ran slower than the shortcut-OFF config (-17/-29/-36%) although rocprof
showed identical-or-smaller GPU kernel profiles and llama-cli/llama-server crossing prefills were
neutral (p5000 259.9 vs 258.8 t/s). Environment for this session's measurements: same machine/
build/model as the 2026-09-06 records (qwen4exp a1121cf2d — includes WS3 #3 routed mmq default
ON, which does not interact with the shortcut path), llama-bench `-ngl 99 -t 15 -r 1/2 -b 2048
-ub 2048 -fa on -ctk f16 -ctv f16 --load-mode none`.

## Reproduction (same-session, current build)

pp8192 r2 depth-0: shortcut OFF 515.90 ± 9.3 t/s vs ON 339.43 ± 10.6 (−34%). Reproduced.

## What the per-ubatch timing shows (env-gated instrumentation added to llama_context::decode +
llama_context::process_ubatch + ggml_backend_sched_alloc_splits, since removed)

llama-bench's test_prompt decodes the 8192-token prompt as 4 separate llama_decode calls of 2048
tokens each (n_batch chunking — identical structure to llama-cli), WITHOUT llama_synchronize
between them (sync only after the whole prompt). Per-decode walls (ms) for pp8192 r1:

| pass | config | decode1 pos0 | decode2 pos2048 | decode3 pos4096 | decode4 pos6144 |
|------|--------|-------------:|----------------:|----------------:|----------------:|
| 1 (warmup) | OFF | 2921 | 5794 | 6206 | 6224 |
| 1 (warmup) | ON  | 2906 | 5568 | 6139 | 6214 |
| 2 (timed)  | OFF | 2706 | 3246 | 3292 | 3320 |
| 2 (timed)  | ON  | 2700 | 5995 | 6123 | 6196 |

Pass 1 is equally slow in both configs (GPU-buffer first-touch + the per-ubatch layout growth
below). Pass 2 is ~2x faster in OFF (warm buffers, stable layout) but stays at pass-1 speed in
ON. decode1 is the dense-shortcut ubatch (n_kv 0..2048 <= width 2051); decode2-4 are the sparse
QSA ubatches (n_kv 4096/6144/8192 > width). Both configs rebuild their graph every ubatch
(~1 ms build — graph reuse never fires here because mask/cell tensors grow with n_kv), so the
delta is NOT graph-build cost.

## Root cause (mechanism, confirmed by direct instrumentation)

ggml_backend_sched_alloc_graph -> ggml_backend_sched_alloc_splits: when the new graph's layout
differs from the gallocr's single stored layout (node count or per-node sizes changed — checked
by ggml_gallocr_needs_realloc against ggml_gallocr's ONE node_allocs[] layout), the sched takes
the alloc-fallback branch which first does a FULL device synchronize (ggml_backend_synchronize on
every backend, waiting for the ENTIRE queued GPU backlog) and then re-reserves the gallocr. The
fallback sync is unconditional in that branch — it exists because re-reserving can move tensor
addresses that in-flight kernels of the previous (still queued) graph are reading.

llama_context::graph_compute uses ggml_backend_sched_graph_compute_ASYNC: llama_decode returns
without waiting for the GPU. llama-bench therefore pipelines its 4 decode calls back-to-back (deep
GPU queue). When an alloc-fallback fires mid-pipeline it must drain the whole queue first — with
~3 s of queued work per 2048-token ubatch, each fallback sync costs ~3 s. Confirmed by
instrumentation (GGML_SCHED_TIMING):

- OFF: 4 alloc-fallback syncs, all in pass 1 (3056/3497/3479 ms + one ~0); pass 2 = ZERO
  fallbacks (its graphs exactly repeat pass 1's max layout) -> pipelined -> fast.
- ON: 8 alloc-fallback syncs, in BOTH passes (2827/3397/3541 + one ~0 in pass 1; 3228/3438/3512
  + one ~0 in pass 2). decode1 (dense graph, 7273 nodes vs the sparse graphs' 7773) re-shrinks the
  stored layout every pass, so decode2-4 (sparse, sizes still growing with n_kv) each re-trigger
  the fallback -> serialized -> stays at pass-1 speed every rep.

Why real serving is immune: llama-cli / llama-server / common.cpp call llama_synchronize after
each llama_decode (common/common.cpp: llama_decode + llama_synchronize pairs), so their GPU queue
is empty at every alloc-fallback -> syncs cost ~0 -> no artifact (validated: p5000 llama-cli
neutral). The artifact is specific to llama-bench's sync-free multi-decode pipeline interacting
with ggml-gallocr's single-layout design when consecutive graphs alternate topologies
(dense-shortcut below the selection width, sparse above). Note this can bite any workload that
(a) pipelines async llama_decode calls and (b) alternates graph shapes (the same churn exists in
pass 1 of every config; OFF merely reaches a stable max layout by pass 2).

## Why the "structure-stable shortcut" fix is not attractive (same-session measurements)

Making the dense ubatch keep the sparse graph's topology (top-k chain + masked-dense kernel below
the width; = LLAMA_QSA_SPARSE_FA=0's structure at those depths) removes the layout flip (SF0 shows
ZERO pass-2 fallbacks) but forfeits most of the shortcut's point. Same-session pp2048@0 r2:
shortcut OFF (qsa kernel) 325.2; SF0-style dense-masked-with-chain 338.8 (+4.2%); true shortcut
(plain dense, chain omitted) 343.6 (+5.7%). tg128@0: SF0 == OFF (23.55 vs 23.54 — NO decode win),
true shortcut +4.2% (24.53). I.e. the structure-stable variant keeps ~3/4 of the pp2048 win but
NONE of the tg@0 win, at the cost of running the whole indexer/top-k chain below the width.

## Options for the maintainer

1. Keep LLAMA_QSA_DENSE_SHORTCUT opt-in default OFF (status quo). Wins (+3-5% pp512-2048@0,
   +4.2% tg@0, neutral-or-faster real-serving prefill) available to users who do not measure deep
   rows with llama-bench; the deep-row llama-bench rows are unrepresentative for this config.
2. ggml-level fix: give ggml_gallocr a small multi-layout cache (per graph-shape signature) so
   alternating dense/sparse (or any shape-changing) graphs don't re-reserve + full-sync each time.
   Real fix for all shape-alternating pipelined workloads; it is a core-ggml change (address
   stability across layout slots must be preserved) and needs its own careful session/validation
   on all archs. Recommended path if the shortcut's wins are wanted by default.
3. Default ON + accept the llama-bench deep-row artifact (document: bench rows with alternating
   graph shapes are protocol-broken; real serving measured neutral-or-better).

This investigation also explains why llama-bench pass-1 is always slower than later reps for this
model (layout warm-up) and documents the gallocr single-layout limitation for future work. No code
shipped this session; tree restored to a1121cf2d (instrumentation reverted).
