# Decision register — where upstream and the delivery answer the same question

Each item is a **stop-and-ask** point: the rung that reaches it does not pick a winner, it measures and
presents.  `AGENTS.md` → "Report findings before shipping" and the WIP rules apply: nothing here changes
`patches/` until the maintainer decides, and a dropped delivery mechanism is recorded (a `WORKLOG.md` entry,
`TODO.md` item, or an `archive/work/` record) in the same change.

**Status vocabulary:** `OPEN` (not reached yet) · `MEASURING` (rung reached, numbers being taken) ·
`AWAITING` (numbers in hand, maintainer's call pending) · `DECIDED` (resolution + record).

**The delivery's standing principle** (from `AGENTS.md` and the maintainer): the set aims to be
*complementary* — a second preferred path, not a replacement — so most of these should resolve to "both
paths ship, ours stays the default where it is faster, upstream's stays reachable".  A drop only happens
where the two cannot coexist (one implementation of the same function, or upstream's is strictly better).

Items are reached in rung order; `R` names the rung that first touches the file(s).  The upstream commits
are `a691b2345..4f9dfa0e3` (164 commits); a delivery block is named by its number.

---

## D1 — MoE cache: upstream moved it out of `llama-graph` and gave its CPU part an async story

* **Upstream:** `951d465f5` (`#32157`) moved the cache code from `src/llama-graph.cpp` into
  `src/llama-moe-cache.{cpp,h}` (a new `apply()`/`ensure()` split), and `cf34d3113` (`#31888`) stopped the
  CPU part waiting for cached experts.  The cache itself (ClassLRU + `--moe-cache-cpu-share`) is unchanged
  from the r236 decision, where it was measured **behind the arena by 84 % prefill / 35 % decode** and kept
  shipped-but-dormant.
* **Delivery:** blocks 09/12 carry the host-resident arena/pool tier on a separate path
  (`moe-expert-cache.{cu,h}`, `ggml-cuda-slab.cu`, the scheduler copy hook, the Meta forwarding,
  `MOE_HOST_POOL_*`) and keep upstream's cache as a second, dormant path.
* **Relation:** the r236 "either/or" ruling still stands; what changed is *where* upstream's code lives, so
  the delivery's hooks into `llama-graph.cpp` move/shrink.  No new measurement is owed unless the move
  changes the cache's reach.
* **Questions:** (1) port the delivery's hook to the new `llama-moe-cache` seam; (2) does `#31888`'s async
  CPU part change the r236 measurement?  (It should not — our arena is the default and the cache is off.)
* **Status:** `OPEN` — R10.

## D2 — Chunked GDN: upstream **removed** the non-fused GDN path and every ubatch-dependent graph node

* **Upstream:** `ccd360f06` (`#32078`) deletes ~405 lines from `src/models/delta-net-base.cpp`, removes the
  non-fused GDN builders and the ubatch-dependent graph nodes across `clef`, `clover-moe`, `kimi-linear`,
  `qwen35`, `qwen35moe`, `qwen4exp`, `models.h`, `llama-context.*`, `llama-cparams.h`, `llama-graph.h`.
  `a8575a3ef` routes GATED_DELTA_NET to a SpacemiT RVV kernel (CPU), and `998200491` adds an SSM_SCAN
  d_state=64 fast path.
* **Delivery:** block 03 is the **chunked gated-delta-net prefill** (chunked conv/state prefill, the
  rollback tracker, the `LLAMA_RS_*` knobs).  It is the block the probe reports conflicting **only** in
  `src/models/delta-net-base.cpp`.
* **Relation:** this is the strongest subsume candidate in the range.  Two possibilities, and they need
  different answers: (a) upstream deleted the *non-fused* builder that our chunked path replaced — then our
  path is the surviving implementation and the merge is a de-registration of the deleted nodes; (b)
  upstream's fused builder is the one our chunked path also produced, and our separate graph path is now
  dead weight.
* **Questions:** read the two diffs side by side at R04; then measure prefill on a hybrid recurrent model
  (block 03's win) with the delivery path on and off on the merged tree.
* **Status:** **`DECIDED` at R04 (2026-11-13) — upstream did not subsume the delivery; both sides ship and
  nothing is dropped.**  `#32078` deleted *upstream's own* non-fused graph builders
  (`build_delta_net_chunking`/`build_delta_net_autoregressive`/the chooser) and its GDN fused-node tags and
  registry entries — all present in the old base, so none of it was delivery code.  The delivery's substance
  survives and is the surviving implementation of the simplified path: the `n_rs_batch` argument on
  `ggml_gated_delta_net()` (a chunked kernel may skip rollback snapshots for a batch that can never be rolled
  back into), the pre-batch snapshot copy in `build_recurrent_attn()`, and the `gated_delta_net_chunked*.cu`
  kernels — now reached because upstream always routes through that one fused op.  Upstream's tag deletion is
  taken (re-adding the enum values would not compile).  Measured: `whole-batch chunked GDN prefill n=2048
  K=1 n_seqs=1`, prefill **6908 t/s** on one R9700 (4B, 5 298-token prompt), coherence unchanged.
  `test-recurrent-state-depth` is red on this box on the shipped r277 too (73 failures) — pre-existing and
  config-sensitive, recorded in `RUNGS.md` R04, not a rung regression.

## D3 — KV cache types: upstream added FP8 (KV and FA)

* **Upstream:** `202c1a229` (`#28392`) adds FP8 E4M3/E5M2 KV cache (`--kv-type`/`-ctk`/`-ctv` values, the
  graph conversions, `llama-kv-cache.*`, the model loader) and `9e9619d03` (`#28413`) adds OCP FP8 MMA to
  CUDA (with the FP8 vec FA kernel and its spill fix in `bd2767d06`).
* **Delivery:** block 04 adds the BF16 KV cache and the native-BF16 FA path (the `bf16-bf16` FA instance,
  the `bf16_src1_cache`, the KV type predicates); block 15's window/pool assume the KV type set.
* **Relation:** additive, but a KV type / FA instance / predicate set that is inconsistent on either side
  is a crash on whichever backend reaches it (`GREEDY-PURITY.md` §20).  This is cross-backend consistency,
  not a scope question.
* **Questions:** does the merged tree expose every `{k,v}` type pair its own dispatch can reach (including
  fp8 and bf16), and does the delivery's BF16 path stay the default where measured faster?
* **Status:** **audit PASSED at R05 (2026-11-13).**  The merged `-ctk/-ctv` set is
  `f32 f16 bf16 f8_e4m3 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1`: the delivery's `bf16` and upstream's `f8_e4m3`
  coexist, and on the 4B (1 GPU, `-c 4096`, seed 42) the `bf16`, `f8_e4m3` and `q8_0` arms all run and give the
  same `sha=1c5d32ac537d`.  The BF16 FA instance is compiled in and reached.  **Closed at R06:** the FA instance matrix
  (block 05) merged on the union set, and upstream's FP8 vec FA instance is compiled and exercised
  (`f8_e4m3` cases in the 6996-case FLASH_ATTN_EXT oracle, 0 failures) — no reachable-but-uninstantiated
  type remains (`GREEDY-PURITY.md` §20).

## D4 — MMVF: upstream dispatches BF16 to MMVF for few rows at small batch

* **Upstream:** `459ab1cdd` (`#31959`) sends BF16 `MUL_MAT` with few rows and a small batch to MMVF, plus a
  test-oracle entry.
* **Delivery:** block 04's native-BF16 path and block 07's mmvq work both decide which kernel a BF16 src1
  reaches; the delivery also has `bf16_src1_cache`.
* **Relation:** possibly the same idea at a different width.  Either upstream's dispatch is a subset of ours
  (keep ours, let theirs be the fallback) or it is better at some shape and we gate ours at the widths it
  wins.
* **Questions:** measure the delivery's BF16 dispatch vs upstream's MMVF few-row arm at the shapes the
  delivery gates (decode rows 2-24, batch 1-8).
* **Status:** `OPEN` — R07/R08.

## D5 — Flash attention + few-row MMQ: the FA entry points and the MMQ dispatch moved

* **Upstream:** `9d7b2d739` (`#30285`) adds mma head sizes 40/72, `77e2db8ca` (`#27997`) an sm70
  DKQ/DV=256 case, `cd4ed902e` (`#32018`) drops unused `ids`/`type_src0`, `0fb8f1577` (`#31533`) adds
  **few-row MMQ** (`mmq-few.cu`, BS 2-24) and rewrites the MMQ/MMVQ dispatch, `5d4c8eeb2` (`#32067`) skips
  empty MoE tiles in the stream-k fixup, `0d69255c5` (`#26048`) adds an nvfp4 w_s epilogue.
* **Delivery:** block 05 is the FA band (native quantized K/V, derived kq mask, the `iq4_nl` instances);
  blocks 06/07/08 own the MMB/MMQ/MMVQ dispatch and the MoE tile compaction.
* **Relation:** the FA *band* is a delivery selector on top of upstream's kernel matrix — it must be
  re-expressed on the new instance set, not dropped.  The few-row MMQ may make part of block 06/07's
  few-row win redundant.
* **Questions:** re-derive the FA band over the new instance matrix (including FP8 and the 40/72 heads);
  measure MMB/MMQ few-row (BS 2-24) delivery vs upstream.
* **Status:** **`DECIDED` at R06 (2026-11-13) — complementary, nothing dropped.**  The FA half: the delivery's
  band is a selector over the kernel matrix, so upstream's FP8 vec FA, heads 40/72 and the sm70 case sit
  beside it.  The instance matrix was regenerated on the union (the delivery's 121 typed instances supersede
  the 16 head-agnostic files; upstream's 40/72 additions are declared in the typed set and were exercised),
  and upstream's `DKQp`/`DVp` padded-head mechanism was re-added because the delivery's inherited loops and
  Q-tile stride need it for head 72.  Oracle: `test-backend-ops -o FLASH_ATTN_EXT` **6996 OK / 0 FAIL**
  (18 derived, 385 iq4_nl, 7 f8_e4m3, 448 head-40, 1010 head-72).  The **few-row MMQ half of D5 is still
  open**: `#31533`'s `mmq-few.cu` meets the delivery's MMB/MMQ dispatch at R07, and that is where the
  delivery-vs-upstream few-row measurement belongs.

## D6 — CUDA graph and concurrent-stream state left `common.cuh`

* **Upstream:** `17701e48d` (`#32001`) moves ~300 lines of graph + concurrent-stream state out of
  `ggml/src/ggml-cuda/common.cuh` into a new `graph.cuh` (+119 in `ggml-cuda.cu`); `2e325dcd5` (`#31284`)
  re-instantiates CUDA graphs when a kernel's cluster dims change; `15759e0ca`/`cd4ed902e` touch the graph
  compat check.
* **Delivery:** blocks 08/10 add fusions and the async-input ring on top of that state, and block 10 owns
  the graph key.  Our grid-stall pad and fusion dispatch hooks live in `common.cuh` today.
* **Relation:** a pure structural move — our hooks must follow the state into `graph.cuh`.  The cluster-dim
  re-instantiation matters for us: our fusions change launch geometry, which is exactly the key it watches.
* **Questions:** does every delivery hook that touches graph/stream state still fire after the move, and
  does the cluster-dim key invalidate a captured delivery fusion correctly?
* **Status:** `OPEN` — R09/R11.

## D7 — MTP: upstream reads the target hidden states from a **device buffer**

* **Upstream:** `a8fad21e9` (`#31927`) adds a device-buffer path for the MTP target hidden states
  (`llama-context.*`, `llama-batch.*`, `llama-graph.*`, `llama-cparams.h`, `llama-ext.h`, `common/common.*`,
  `common/speculative.cpp`), and `7874988b4` (`#31294`) warns when `--spec-draft-p-min` meets probabilistic
  draft sampling.
* **Delivery:** block 02 is the **adaptive MTP draft depth** (the draft-depth controller, its knobs and the
  `llama-ext.h` surface); block 13 is the draft-vocabulary subset.
* **Relation:** the device buffer is a transport change under the same feature.  It may subsume the
  delivery's host round-trip (a win, then take it and keep our depth controller on top), or it may be a
  second path we select by shape.
* **Questions:** measure adaptive-MTP four-axis (the gate in `benchmarks/mtp-adaptive-methodology.md`) on the
  merged tree; compare the delivery's hidden-state transport with the device buffer at the field shape.
  Also check the `--spec-draft-p-min` warning does not fire for the delivery's defaults.
* **Status:** **measured at R03, `OPEN` as a decision (nothing is dropped).**  On the R03 tree
  (`50a0b30f8`), 35B-A3B Q8_0 bf16 KV with `draft-mtp-adaptive`: the adaptive controller is live on both
  layouts (acceptance 0.89130 on `-sm tensor`, 1.00000 on `-sm layer`), and upstream's device-buffer
  transport activates **only** on `-sm layer` (`target h rows on the device: yes`).  Under `-sm tensor` it
  declines with `the device has no events, extracting the h_nextn rows to host memory` — the output device is
  the Meta device, whose `iface.event_new` is null — and falls back to the host path (where the delivery's
  `llama_set_embeddings_nextn_from` skip still applies).  So the two are complementary, not competing, on the
  field layout; the delivery keeps its host path and its controller.  Upstream gap worth an `upstream/`
  candidate: give the Meta device backend events so `#31927` is reachable under `-sm tensor`.  Re-verify at
  R14 (the vocab-subset block changes what a draft sees) and measure on `-sm layer` in R18's MTP four-axis.

## D8 — The prune: `llama` and `ggml` removed dead surface

* **Upstream:** `8ca5a8008` (`#30565`) prunes `include/llama.h`, `src/llama-ext.h` (−64), `llama-context`,
  `common/arg.cpp`, `common/speculative.cpp`; `29dbbad2b` (`#29801`) prunes dead ggml ops and their docs.
* **Delivery:** blocks 02/10 add to `src/llama-ext.h`; blocks 11/12 add to `ggml.c`'s op table and RPC
  headers.  The probe reports `src/llama-ext.h` as a **content** conflict for block 02 and block 10 (not a
  delete only because the file survives).
* **Relation:** re-derivation, not a decision: re-add *only the delivery's* declarations to the file's new
  home; never restore a pruned symbol wholesale.  If the prune removed a symbol a delivery block calls, the
  call moves to the public API or the block drops the call.
* **Questions:** after each rung, `git grep` the delivery's symbols; confirm none points at a pruned
  declaration.
* **Status:** **hit at R03, `OPEN` mechanically — the prune also removes *includes*.**  Two concrete lessons:
  (a) the moved declarations (`llama_set_embeddings_nextn`, `llama_set_nextn_layer_offset`,
  `llama_model_dflash_selector_top_k`, `llama_model_target_layer_ids(_n)`) are *not* re-added to
  `src/llama-ext.h`, because upstream has public versions in `include/llama.h`; only delivery-only API stays.
  (b) `common/speculative.cpp` lost `#include "../src/llama-ext.h"` with the prune (the include's comment named
  the now-public calls), which is a **build break, not a merge conflict** — the block's include hunk attaches
  cleanly and the symbol then resolves nowhere.  Check the include surface after every prune rung; blocks 10
  and 11 touch `llama-ext.h` and will meet the same thing.  Re-check at R11/R12.

## D9 — mmap / lazy load / load-mode: upstream rewrote the Windows unbuffered load

* **Upstream:** `9d340ec34` (`#26014`) rewrites `src/llama-mmap.cpp` (+300, new `test-llama-mmap.cpp`) for a
  Windows-unbuffered model load; `58e38bdf6` (`#31957`) applies the iGPU lazy-mode fallback **only when
  mmap is off**; `9d340ec34`'s load-mode plumbing is adjacent to the delivery's `--load-mode`.
* **Delivery:** block 00's foundation hunks in `llama-mmap.cpp`; blocks 09/16 use the load/mmap path for the
  host pool and the GTT store (`--lazy-mode mlock`, `--load-mode`, `O_DIRECT` placement reads).
* **Relation:** the delivery's lazy-mode/mlock/O_DIRECT behaviour must be re-expressed on the new load path,
  not duplicated.  This is the *only* block-00 conflict in the probe.
* **Questions:** does `--lazy-mode mlock` still pin the PLE table without a copy on the new path?  Does the
  iGPU gating change which device the delivery's `--lazy-mode` fallback reaches?
* **Status:** `OPEN` — R01 (resolve), re-verify R10/R17.

## D10 — Scheduler upload/prefetch/graph-result reuse

* **Upstream:** `806b3c2be` (`#31419`) does not upload inputs that are not read; `3bce8bd50` (`#32100`)
  prefetches only inputs computed by earlier splits; `cb44fdb18` (`#32089`) reuses graph result instances
  across scheduler reserves; `4fffab623` (`#32043`) avoids touching unused graph metadata pages.
* **Delivery:** block 10's async input ring, block 12's backend-interface fills and the `ggml-backend`
  scheduler copy hooks the host-expert tier depends on.
* **Relation:** likely cooperative: upstream's "don't upload what is not read" is exactly the delivery
  ring's direction.  The risk is the delivery's forced uploads (host-expert staging) being skipped by the
  new reachability test.
* **Questions:** does every host-expert/staging upload still happen on the merged tree?  Do the async-ring
  arms still measure a win?
* **Status:** `OPEN` — R11/R13.

## D11 — `ggml.c` / RPC / test-surface refactors

* **Upstream:** `29dbbad2b` (op prune), `0eebb6edd` (TRI backward, op table), `57d04820d` (`#31084`,
  deterministic Top-K), `03d6c824e`/`1648d6411` (`#32077`/`#32083`, tests use only the public API),
  RPC header changes (block 11/12 both list `ggml/include/ggml-rpc.h`).
* **Delivery:** block 11's qwen4exp ops (op table + CUDA dispatch + test oracle) and block 12's backend
  fills/RPC surface.
* **Relation:** mechanical re-derivation: the delivery's ops/messages are re-registered on the new tables.
* **Questions:** every new op has one registration + one dispatch + one test oracle; every RPC message has
  a matching handler on both ends.
* **Status:** `OPEN` — R12/R13.

## D12 — BF16 RMS_NORM: upstream rewrote `norm.cu`

* **Upstream:** `31953bcf7` (`#32135`) adds BF16 support to RMS_NORM on CPU and CUDA and rewrites 180 lines
  of `ggml/src/ggml-cuda/norm.cu` (plus the dispatch in `ggml-cuda.cu`).
* **Delivery:** block 08/10 carry the **narrow-row norm fusion** (`norm-gated.cu`, `ggml_norm_rows_*`, the
  `gate-norm-rows-alias.sh` gate), which fuses row-shaped norms and has its own BF16 producer.
* **Relation:** possibly overlapping: if upstream now handles BF16 rows natively, our fusion's BF16 arm may
  be redundant; the fusion's *narrow-row* win is a different axis and should survive.
* **Questions:** re-run `gate-norm-rows-alias.sh` on the merged tree; measure narrow-row norm with the
  fusion on/off over the new kernel.
* **Status:** `OPEN` — R09.

## D13 — HIP: a fixed rocBLAS workspace

* **Upstream:** `432433637` (`#31838`) gives the HIP backend a fixed BLAS workspace instead of rocBLAS
  allocating one per call.
* **Delivery:** block 01's hybrid all-reduce and block 07's rocBLAS-adjacent work assume the previous
  allocation pattern; the delivery gates HIP features on `gfx1201` only.
* **Relation:** additive, but it changes allocation timing/lifetime; the delivery's VMM arena must not
  assume the old workspace.
* **Questions:** confirm the all-reduce dispatch and the arena placement still see the expected free VRAM.
* **Status:** `OPEN` — R02.

## D14 — GDN / SSM kernel additions

* **Upstream:** `a8575a3ef` (`#31897`) routes GATED_DELTA_NET to a SpacemiT RVV kernel (CPU), `998200491`
  (`#31818`) adds d_state=64 to SSM_SCAN (CUDA), and `ccd360f06` prunes the graph side (see `D2`).
* **Delivery:** block 03's chunked GDN prefill and its kernel/dispatch set.
* **Relation:** additive on CPU; on CUDA the d_state=64 fast path is in the same dispatch our chunked path
  may specialise.  Note **D2** decides the graph; this item only checks the kernels are consistent.
* **Status:** `OPEN` — R04.

## D15 — External PLE file loading

* **Upstream:** `5f490d7bf` (`#31185`) adds external PLE file loading and reshapes `common/fit.*` and
  `src/llama-model-loader.cpp`; `common/fit.cpp` is a probe conflict for blocks 09 and 15.
* **Delivery:** block 12's PLE `madvise(WILLNEED)` loop and block 00's PLE path; the `wip/ple-residency/`
  campaign (not part of the delivery, not applied here) measures a row store on top.
* **Relation:** the delivery's load-bearing `madvise` loop must be re-expressed on the new loader; the
  `fit.cpp` reshuffle is a merge, not a decision.
* **Status:** `OPEN` — R10/R16.

## D16 — MUL_MAT src1 precision tests and gating

* **Upstream:** `494d0f47b` (`#31896`) adds src1-precision tests and gates per-backend src1 type handling
  (`ggml-cuda.cu`, `mmvf.cu`, CPU, Metal, SYCL, …).
* **Delivery:** block 04's native-BF16 src1 cache/path and block 07/08's dispatch.
* **Relation:** the new gate may force a `supports_op` answer for the delivery's src1 shapes; the delivery
  must answer consistently with its dispatch.
* **Questions:** `test-backend-ops -o MUL_MAT` and `-o MUL_MAT_ID` must pass with the delivery's BF16 path
  enabled — and the gate must reach it, not a fallback.
* **Status:** `OPEN` — R08.

## D17 — VMM pool: reservation halved, retry on failure — **THE BASE BLOCKER**

* **Upstream:** `ac931b5bc` (`#30542`) halves the VMM pool reservation and retries `cuMemAddressReserve` on
  failure.
* **Delivery:** block 09/16's arena and GTT store both require VMM (`GGML_HIP_NO_VMM=OFF`), and block 16's
  single registered region assumes a large reservation.
* **Relation:** this is not a "who wins" question — the upstream commit **breaks the base on this box**.
  With the delivery's required build (VMM off), a pure `master` build never returns from its first GPU
  compute: the process spins in `ggml_cuda_pool_vmm::alloc` inside the HIP runtime.  Unpatched `master` with
  `GGML_HIP_NO_VMM=ON` works, and reverting `ac931b5bc` makes the VMM=OFF build work.  Full evidence:
  [`BASE-BLOCKER.md`](BASE-BLOCKER.md).
* **Questions / options:** (1) carry a full revert of `ac931b5bc` folded into block 00 (the ladder state —
  proven green); (2) HIP-gate the revert (CUDA keeps the shrink-on-OOM); (3) verify with
  `GGML_HIP_NO_VMM=ON` and leave R10/R17 blocked; (4) move the fork point to `ac931b5bc^`.  Recommended: (1)
  — it is one revert that disappears with the next base, and the commit only targets CUDA combined-memory
  devices this delivery does not validate.
* **Verification owed when decided:** `gate-uma-experts.sh --quick` on `halo` (the region size `-lme` asks
  for) and `gate-host-pool.sh` (the arena decline path) at R10/R17.
* **Status:** **`DECIDED` (2026-11-13, maintainer: option 1 — carry the full revert, folded into block 00).**
  R01's block-00 commit `43a418c10` is the decided shape (36 files, +1049/−357 against `4f9dfa0e3`): the
  pre-#30542 VMM pool on every backend, no HIP-only split.  The revert is dropped at the next re-base once
  upstream fixes the HIP pool (the provenance note lives in `BASE-BLOCKER.md` and this entry).  Verification
  still owed at R10/R17 (`gate-host-pool.sh`, `gate-uma-experts.sh --quick` on `halo`) — the arena/GTT store
  are the first delivery code that actually exercises a large reservation.

## D18 — SYCL gained a device-side MoE scheduler

* **Upstream:** `18c5a95e2` (`#30156`) adds a device-side MoE scheduler to SYCL (fused gemm/mmvq, op-table
  flags, test-oracle entries).
* **Delivery:** block 09's host-resident MoE is CUDA/HIP-only, but the op/type tables and `supports_op`
  predicates are shared.
* **Relation:** cross-backend consistency only — do not "fix" SYCL, do not let the SYCL flags change the
  CUDA predicate.
* **Status:** `OPEN` — R10.

## D19 — Recurrent cache tensor renaming removed in plamo2

* **Upstream:** `940a30ee3` (`#31984`) stops renaming the recurrent cache tensors in `src/models/plamo2.cpp`.
* **Delivery:** block 15's streamed KV and the recurrent memory instances the delivery's placement touches.
* **Relation:** verify: the delivery must not depend on the old names for a recurrent cache.
* **Status:** `OPEN` — R16.
