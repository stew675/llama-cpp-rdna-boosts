# Strix Halo (RDNA3.5 / gfx1151) — WS3 #2 artifact FIX at the ggml level + shortcut default ON

Date: 2026-09-10 session (task: the maintainer's Option 2 from the root-cause record
`benchmarks/2026-09-08-strix-halo-gfx1151-ws3-shortcut-artifact-rootcause.md` + SESSION-BRIEF-2026-09-10).
Environment: same machine/build/model as the 2026-09-08/09 records (qwen4exp, IQ4_XS UD
Qwen3.8-Flash-Next), llama-bench `-ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16
--load-mode none`, warm page cache, prompts descending in one process (pp16384 first), r3
same-session A/B. A commits: `fcfb0a522` (ggml fix) + `1682d32a9` (default flip) on `a1121cf2d`.

## The fix

Mechanism recap (root-cause record): llama-bench pipelines async llama_decode calls (no
llama_synchronize between 2048-token chunks); ggml-gallocr keeps ONE stored layout; any graph
whose node count or per-position sizes differ (ggml_gallocr_needs_realloc) sends
ggml_backend_sched_alloc_splits into its fallback, which did an UNCONDITIONAL full device sync
(drains the ~3 s queued backlog) + full re-reserve. With the shortcut ON the dense ubatch
(7273 nodes) alternates with sparse ubatches (7773, growing with n_kv) -> a fallback EVERY
ubatch of EVERY rep -> measured 8 syncs of 2.8-3.5 s over 2 passes at pp8192 r1 (ON) vs 4 in
pass 1 only (OFF) -> ON never reaches the warm layout state -> -17/-29/-36% at pp4096/8192/16384.

Design analysis that shaped the fix (differs from the brief's design-B premise): the brief's
"same-size tensors hash to stable addresses across reserves" is NOT true in this code — the
per-graph hash table is reset inside every reserve (alloc_graph_impl) and addresses come from
the dyn-tallocr free list by allocation sequence. The correct safety argument is ORDERING, not
address stability: gallocr buffers are GROW-ONLY (reserve only frees+reallocates when a chunk
grows past its current size), and all compute for a sched is ordered on the backend stream(s)
after the previous graph's compute. So a reserve that does NOT grow any buffer only re-points
the new graph's tensors — exactly what the layout-reuse path already does every decode with
zero sync — and is safe without a sync. The ONLY case requiring the sync is an actual buffer
growth (free + realloc moves addresses an in-flight graph may still be using).

Also found and removed: `backend_ids_changed` kept forcing the sync even when nothing grew.
The graphs carry ~16-20 CPU-assigned nodes (the KQ-mask views + the QSA indexer chain tensors;
16 CPU nodes dense vs 20 sparse), so the sched's POSITIONAL backend-id comparison fires on
every dense/sparse topology flip. But backend-id changes with no buffer growth are also just
re-pointing (nothing freed; per-device streams still in-order) -> removed from the sync
condition as well.

### Change (A commit `fcfb0a522`, 3 files, +57/-11)

- `ggml/include/ggml-alloc.h` + `ggml/src/ggml-alloc.c`:
  - `ggml_gallocr_reserve_n_probe()` — computes and STORES the graph layout without touching
    the existing buffers, returns whether any buffer would need to be grown (reallocated).
  - `reserve_n_impl`'s `no_alloc` path no longer frees the buffers (it did before, which made
    size-only calls destructive on a live gallocr); it records growth via a new out-param.
- `ggml/src/ggml-backend.cpp` (`ggml_backend_sched_alloc_splits`): the fallback now probes
  first; only when `buffers_grown` does it synchronize all backends + `reserve_n` (alloc mode)
  + `alloc_graph`. Otherwise the probed layout is used directly (`alloc_graph`), no sync.

No numerics change: no op semantics touched; the fallback and reuse paths execute the same
kernels. Core-ggml, arch-agnostic; validated on this box (see below). The multi-GPU /
pipeline-parallel configurations were NOT exercisable here — the ordering argument holds
per-device (each device's splits are stream-ordered and buffers are device-private), but a
gfx1201/dual-GPU validation should re-check before claiming those.

## Validation (all on the fixed build, this session)

### Sync-pattern instrumentation (env-gated, removed after measurement)

pp8192 r2 + pp16384 r2, shortcut ON, GGML_SCHED_TIMING counts (per-ubatch alloc-fallback):

- pre-fix ON: 8 syncs over 2 passes, ~3 s each (the artifact).
- post-fix ON: ZERO syncs in the whole run; every dense/sparse alternation took the NO-SYNC
  probe path (36 fallbacks, ~1 ms layout recompute each). Off-path (sparse-only, no fallbacks
  in pass 2) unchanged.

### Same-session r3 matrix (fixed build, shortcut ON vs OFF (=0), depth 0 + depth rows)

| test | shortcut ON | OFF (=0) | Δ |
|------|------------:|---------:|--:|
| pp16384 @0 | 574.07 ± 1.58 | 566.82 ± 2.03 | +1.3% |
| pp8192  @0 | 543.97 ± 0.56 | 534.96 ± 1.66 | +1.7% |
| pp4096  @0 | 487.91 ± 0.99 | 472.99 ± 0.77 | +3.2% |
| pp2048  @0 | 399.68 ± 0.16 | 382.73 ± 0.46 | +4.4% |
| pp1024  @0 | 418.61 ± 9.76 | 407.58 ± 8.60 | +2.7% |
| pp512   @0 | 407.71 ± 1.29 | 401.26 ± 0.85 | +1.6% |
| tg128   @0 | 25.25 ± 0.01 | 24.23 ± 0.01 | +4.2% |
| pp2048 @ d12288 | 329.29 ± 9.44 | 330.41 ± 9.18 | flat |
| tg128   @ d12288 | 22.09 ± 0.03 | 22.12 ± 0.04 | flat |
| pp2048 @ d32768 | 334.01 ± 0.75 | 333.11 ± 0.42 | flat |
| tg128   @ d32768 | 20.11 ± 0.03 | 20.13 ± 0.03 | flat |

vs the pre-fix artifact (root-cause record): pp8192 ON 339.4 -> 544.0 (+60%), pp16384 ON ~334 ->
574 (+72%), pp4096 ON 367.3 (09-06 r3) -> 487.9 (+33%). The artifact rows (previously -17/-29/
-36%) are now +1.3-3.2% ON over OFF. Depth rows (shortcut inert at depth: identical graphs both
configs) are flat through 32k; decode (mmvq) untouched.

### Coherence / determinism (seed-42 temp-0 llama-cli, this build)

- Default (shortcut OFF pre-flip) 7-tok probe text == stored known-good (`cli-restore.txt`,
  byte-identical generated text; only the printed pp t/s differs).
- Shortcut ON 7-tok probe == `LLAMA_QSA_SPARSE_FA=0` masked-dense reference (the shortcut's
  documented numerics identity below the width; identical text). The ON-vs-OFF difference at
  n_kv < 2051 is the pre-existing, documented dense-vs-sparse kernel signature (15.708 vs
  15.973 pp-last top0 at the 7-tok probe — an env-selectable regime A already ships; B runs
  dense by default too).
- Multi-ubatch p5000 crossing prompt with shortcut ON run twice: byte-identical generated text
  (deterministic under the new no-sync re-pointing path; no address-dependent corruption).
- ON-vs-OFF at the p5000 crossing diverges exactly as the pre-fix era did ("repeated many times"
  vs "seems repeated variations") — the long-standing regime signature, not the fix.
- ggml change is op-inert (no kernel or schedule-of-kernels change; only when/where the host
  re-points scratch while the GPU queue drains).

## Decision: LLAMA_QSA_DENSE_SHORTCUT default ON (A commit `1682d32a9`)

Maintainer rule: "default to that which is fastest, and coherent". With the artifact fixed the
shortcut is >= OFF at every depth-0 row (incl. the previously-broken llama-bench deep rows),
+4.2% tg@0, depth flat, deterministic, and its numerics = the dense reference (= B's default
regime below the width). The env is inverted to opt-OUT: unset or =1 = ON (dense below the
width), =0 = the pre-flip selection path (known-good numerics), matching B's env semantics for
cross-testing. Beta patches: `ggml-sched-fallback-sync.patch` (6th) + `ws3-shortcut-default-on.patch`
(7th), clean-apply verified at `a1121cf2d`, applied tree byte-identical to `1682d32a9`.

## Carried-forward notes

- The ggml fix is a candidate core-ggml/upstream change (general: any shape-alternating
  pipelined workload on the sched); upstreaming is the maintainer's call (AGENTS: never push
  from ~/llama.cpp; nothing here was pushed).
- gfx1201 / multi-GPU / pipeline-parallel validation of the ggml fix still pending a gfx1201
  box (see the ordering argument above); RDNA4 validation of WS3 #3 (patch 5) also still
  pending via the delivery flow.
- Raw evidence: /tmp/gateA/fix-{on,off,matrix,depth,d32768}*.log, fix-coh-*.txt (this session).
