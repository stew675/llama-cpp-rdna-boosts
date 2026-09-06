# Strix Halo session brief — handoff after 2026-09-10 (compact me; run the NEXT SESSION prompt)

CONTINUE from this file + ~/make-strix-halo-faster.md + wip/strix-halo/notes-ws1-survey.md + the
dated records. ACTIVE GOAL (maintainer): equal/surpass the community repo (B, halo-box
`~/strix-llama.cpp` c7af5c6c2, untouched ~1% reference) for PREFILL and TG at EVERY data point.
The immediate task is the PREFILL root-cause + fix investigation — read
`benchmarks/2026-09-10-strix-halo-gfx1151-prefill-rootcause-investigation.md` FIRST.

## Commits / tree state (all clean, nothing pushed)

- ~/llama.cpp (qwen4exp) tip `a18e24f97` (on `1682d32a9`, on `a1121cf2d`...): the weighted-down
  decode fusion port (WS3 #4). Build current. Instrumentation from the pp2048 hunt was REVERTED
  (tree clean); re-add env-gated prints if re-measuring (GGML_SCHED_BOUNDARY in
  ggml_backend_sched_compute_splits, GGML_UB_TIMING around process_ubatch in llama-context.cpp).
- ~/llama-cpp-rdna-boosts (delivery): patches 1-8 staged as a COLLECTION (squash later);
  records/TODO/briefs updated. ~/strix-llama.cpp untouched. NO pushes (AGENTS: never from
  ~/llama.cpp; gfx1201 validation deferred until ALL Strix work is done; ggml fix NOT an
  upstream candidate now).

## What is DONE (2026-09-10 sessions)

1. WS3 #2 artifact FIXED at the ggml level (`fcfb0a522`): sched alloc-fallback syncs only when a
   buffer must actually grow (`ggml_gallocr_reserve_n_probe`). LLAMA_QSA_DENSE_SHORTCUT DEFAULT
   ON (`1682d32a9`; =0 restores the pre-flip path). Records: 2026-09-08-rootcause,
   2026-09-10-shortcut-fix.
2. WS3 #4 weighted-down fusion ported (`a18e24f97`, patch 8): decode MoE tail (mmid+mul+views+
   adds) -> one kernel; fires (rocprof), text-neutral, tg@d12288 +1.3%, tg@0 ~flat, pp unchanged.
   DECODE-only (single-token shape gate) - does NOT touch prefill.
3. A-vs-B gap re-derived on the new default (2026-09-10-ab-gap-default-on): depth-0 pp gaps
   1.58/1.76/1.93/1.39/1.25/1.04x (pp512..16384), tg@0 A ~25.0-25.3 vs B 25.96; depth-12k pp
   1.17-1.55x + tg A wins; depth-32k A wins everything. B@depth must use llama-bench -r 1 (B's
   cub argsort aborts under -r3 state-restore at depth). Same-box B runs 10-30% below the
   community PR table (use same-box B).

## THE PREFILL INVESTIGATION (the active task — see the record for full data)

Findings: (a) A's pp2048 llama-bench decode = split0 CPU `model.input_embed` (PLE get_rows) 4
nodes = 2700 ms + split1 GPU 7201 nodes; llama-cli identical flags = 494.8 t/s (no 2.7 s) ->
the llama-bench single-ubatch rows are inflated by a REAL A-side ingest defect (B is fast in
both bench and cli); (b) even on the real path A pp2048 = 494.8 (cold) vs B 653 = 1.32x;
(c) rocprof GPU-busy A 2.94 s vs B 2.41 s per decode -> real kernel deltas ~0.5 s/pass: concat
(A generic 0.41 s vs B transposed 0.13 s), swiglu-input quantize (B fuses it, A doesn't), Q8_0
gate/up mmq +13% on same calls, A-only mm_ids_helper<10> (0.31 s), host submit; near-parity on
the routed-compact expert mmq + rocBLAS + hc fusions.

NEXT SESSION IN ORDER:
1. ROOT-CAUSE + FIX the CPU input_embed 2.7 s: confirm llama-cli honors --load-mode; find why
   llama-bench's decode routes the PLE embedding to a CPU split; profile the CPU get_rows
   (suspects: host-mmap PLE row path / managed-reader locking / non-optimal CPU IQ4_NL
   get_rows / random-vs-text token ids); diff A vs B qwen4exp PLE placement + input-embedding
   build + host per-ubatch submit (~0.25 s B). Expect most of the pp512/1024/2048 delta to
   vanish (A warm GPU+host pp2048 ~620-700 t/s vs B 771 if the embed is fixed).
2. PORT B's transposed concat + swiglu-input quantize kernels (WS4-recipe coherence then
   same-session A/B).
3. Q8_0 gate/up mmq feed delta (+13% on the largest kernel, same call counts).
4. Re-establish the WARM real-path (llama-cli after a warmup decode) low-depth ladder A vs B.
5. Re-derive the full matrix on the new default; update records + TODO + this brief.

## Carried-forward open items (full ledger)

- Beta/qwen4exp README "Open items" (gfx1201 3xR9700 box): "The answer" 3-token K=2 multi-seq
  drift; mixed K/V cache crash; FA-off + tensor-split unsupported; decode levers (batch decode
  M>1, decode-expert mmvq sweep, GDN fold ~1.5-3%, body-op elementwise fusion); prefill thread
  (pp8192 ~2024); ML-Kernel/gpudh review items.
- Strix decode: tg@0 gap A ~25.0-25.3 vs B 25.96 (the real lever = per-op decode kernel-mix
  profile A vs B — A decode is thousands of small mmvq launches/token); MTP deferred
  (mtp-adaptive-methodology.md is the standing decode gate when re-enabled).
- Delivery/upstream monitors: ROCm unaligned split-load (re-check at each re-base); MXFP4 fused
  MoE MMQ (last block-13 item). Parallel: dual-7900XTX block-12 (community member). Parked:
  LFRU host->GPU migration.
- Envs: GGML_CUDA_DISABLE_HC_FUSION, LLAMA_QSA_DENSE_SHORTCUT (=0 pre-flip path), GGML_CUDA_
  DISABLE_MMQ_ROUTED, GGML_CUDA_DISABLE_WEIGHTED_DOWN (patch 8 opt-out), LLAMA_QSA_SPARSE_FA=0.

Hygiene: no parallel benches; warm page cache (dd 3 shards); long pp first (warm clock);
verify non-empty output in loops; depths 0/12k/32k; ~116 GB VRAM / ~419 GB disk; one server
(port 8033); dated records benchmarks/YYYY-MM-DD-strix-halo-*.md; scratch /tmp/gateA/ + /tmp/prof/
(rocprofv3 --kernel-trace -d /tmp/prof -o NAME -- <app>; summarize with /tmp/rocprof-sum.py).
