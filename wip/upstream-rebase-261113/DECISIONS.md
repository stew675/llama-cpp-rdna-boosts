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
* **Status:** **`OPEN` but ANSWERED at R10 (2026-11-13): there is no port to do — upstream moved its *own*
  cache, not ours.**  `#32157` relocated upstream's cache from `llama-graph.cpp` to
  `src/llama-moe-cache.{cpp,h}` (the `apply()`/`ensure()` split became `build_lookup()` / `get_experts()` /
  `copy()` / `build_experts()`), and `cf34d3113` stopped its CPU part waiting.  The delivery's tier is a
  **different implementation in a different place** — the CUDA-side arena (`moe-expert-cache.{cu,h}`), the
  movable-boundary slab, `llama_context::sched_copy_experts` (which already forwards to `moe_cache->copy()`
  for upstream's dormant path) and the `moe_cache_rearm`/`set_reserve` device hooks — and block 09 touches no
  file of upstream's cache at all (`git status | grep llama-graph` is empty).  So the r236 **either/or ruling
  stands**: arena default-on, upstream's cache shipped-but-dormant behind `--moe-cache-mib`.  The r143-era
  measurement in the block's own commit message (35B-A3B Q8_0, `-ncmoe 20`, `-c 8192`, 16.3 GiB host experts,
  1 GPU, equal 4080 MiB budgets) is why the ruling is "ours default, theirs dormant" rather than "theirs wins":
  arena prefill **547.6** / decode **53.5** t/s vs upstream cache **544.2** / **62.9** and streaming 470 / 49,
  text byte-identical — theirs is 17.6 % faster on decode at that shape.  No new measurement is owed at the
  rung; R18's arms update the numbers.  **The rung itself is parked for a different reason** (the `-sm tensor`
  arena fault — `RUNGS.md` R10 and `D10`), not for `D1`.

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

## D5-a — the merged FA instance matrix has no `fattn-mma-f16` case for heads 40/72 (found at R09)

* **Upstream:** `9d7b2d739` (`#30285`) added mma head sizes 40 and 72, dispatched from `fattn.cu`
  (`case 40:`, `case 72:`) and declared in `fattn-mma-f16.cuh`, with the instance matrix generated by
  `generate_cu_files.py` into **head-agnostic** `fattn-mma-f16-instance-ncols1_*-ncols2_*.cu` files.
* **Delivery (block 05):** replaced those 16 head-agnostic files with **126 typed**
  `...-dkqN-dvN.cu` instances (heads 64/80/96/112/128/256) plus RDNA4 band instances, and its generator loop
  is the typed one.
* **Relation:** R06 merged both sides *partially*: it kept the delivery's 126 typed files (correct — the band
  kernels are declared per head) and took upstream's `DKQp`/`DVp` padded-head mechanism, but it did **not**
  put heads 40/72 into the typed set, while upstream's `fattn.cu` dispatch for them survived.  Result:
  **32 undefined symbols** in `libggml-hip.so` (`ggml_cuda_flash_attn_ext_mma_f16_case<40,40,n1,n2>` × 16 and
  `<72,72,...>` × 16) — a reachable-but-uninstantiated kernel case (`GREEDY-PURITY.md` §20).
* **Evidence (2026-11-13, R09):** `nm -D --undefined-only libggml-hip.so | grep -c flash_attn_ext_mma_f16_case`
  = **0** for the r277 build and for the pristine `4f9dfa0e3` build, **32** for the merged tree;
  `git ls-tree` finds **0** `dkq40`/`dkq72` mma instance files at R08 and R09; block 08 touches no `fattn*`
  file, so this is R06's gap, not R09's.  Found by `gate-norm-rows-alias.sh` section A, which links a driver
  against the backend library.  The FLASH_ATTN_EXT oracle does not see it because heads 40/72 route through
  the tile/vec kernels on this build.
* **Second half:** the merged `generate_cu_files.py` still emits the **head-agnostic** mma files (25 of them,
  heads 40..256) while the tree ships the typed 126, so running the generator in-tree would `os.remove` all
  126 and silently revert block 05's instance matrix.
* **What upstream's switch actually cost us (measured 2026-11-13):** the typed set at the **old base**
  (`a691b2345`) and at **r277** is byte-identical — 126 files, heads
  {64, 80, 96, 112, 128, 192→128, 256, 320→256, 512, 576→512}; block 05 never touched the mma instances (it
  added tile/vec/band).  Upstream **converted typed -> head-agnostic inside the range** and folded its **two
  new heads (40, 72)** into that conversion, so the R06 "keep the delivery's typed set" resolution (correct)
  silently dropped exactly those two heads, and nothing else.
* **Exact repair (additive; block 05 owns both halves):**
  1. **32 new instance files**, one declaration each, following the tree's own convention
     (`template-instances/fattn-mma-f16-instance-ncols1_<N>-ncols2_<M>-dkq40-dv40.cu` /
     `-dkq72-dv72.cu`, body `DECL_FATTN_MMA_F16_CASE(40, 40, N, M);`): the 16 `(ncols1,ncols2)` pairs the
     generator's rule allows for a non-niche head — `(64,1) (32,2) (16,4) (8,8)`, `(32,1) (16,2) (8,4) (4,8)`,
     `(16,1) (8,2) (4,4) (2,8)`, `(8,1) (4,2) (2,4) (1,8)` — times {40, 72}.  CMake globs
     `template-instances/fattn-mma*.cu`, so **no CMake change** is needed.
  2. **Restore the mma loop in `generate_cu_files.py` as a typed loop** (one file per head per ncols pair, on
     `HEAD_SIZES_KQ`), because the merged script still emits upstream's 21 head-agnostic files: running it
     today would `os.remove` all 126 typed files and re-introduce the gap.  With heads 40/72 added the script
     then reproduces the tree exactly (158 files).
* **Options:** (1) the repair above — **recommended**; (2) instead delete upstream's head-40/72 dispatch
  (`fattn.cu` `case 40:`/`case 72:` + the `DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(40|72, …)` declarations) and keep
  the typed set as it is — a deliberate drop of an upstream capability, needing no build cost but leaving the
  delivery unable to serve a head-40/72 model on **any** backend, and adding an `upstream/`-shaped divergence
  nobody measured; (3) accept the hole and teach the driver-linking gates to link with
  `--allow-shlib-undefined` — keeps a live abort (see reachability) and hides it from the gates.  Only (1)
  preserves both the delivery's matrix and upstream's coverage.
* **Reachability (not NV-only):** `ggml_cuda_flash_attn_ext_mma_f16()`'s `case 40:`/`case 72:` are
  **unconditional**, and `ggml_cuda_get_best_fattn_kernel()` returns `BEST_FATTN_KERNEL_MMA_F16` for those heads
  whenever the shape is not steered to vec/tile (e.g. a prefill with `Q->ne[1] > 2` and f16 KV on gfx1201).
  So on AMD a head-40/72 model would abort on the first such call — a latent crash for a model upstream serves,
  not merely a link artefact.  None of the delivery's validated models are head 40/72 (4B/35B-A3B/Flash-Next
  are 128), which is why no gate caught it.
* **Cost:** 32 additional `fattn-mma-f16` template instantiations (~2-6 min on a cold configure; incremental
  ~0 on a warm ccache if the files are unchanged).  No runtime change on any head-128 model.
* **Verify:** `nm -D --undefined-only libggml-hip.so | grep -c flash_attn_ext_mma_f16_case` = **0**,
  build green, `test-backend-ops -o FLASH_ATTN_EXT` 6996/0 **and** now genuinely exercising heads 40/72 through
  mma where the predicate selects it, coherence `sha=1c5d32ac537d`, then `gate-norm-rows-alias.sh` links and
  runs, and `gate-per-block-build.sh` on the re-cut block 05.
* **Verify after either:** `nm -D --undefined-only` clean, `test-backend-ops -o FLASH_ATTN_EXT` still
  6996/0, coherence `sha=1c5d32ac537d`, then `gate-norm-rows-alias.sh` links and runs.
* **Status:** **`DECIDED` (2026-11-13, maintainer: option 1) and APPLIED as the rung-repair commit
  `9c94ce139`** on top of the green R09 — additive only, no existing instance file edited, nothing removed:
  the 32 mma head-40/72 instances, the 14 FP8 KV instances the same conversion hid, and a **typed**
  `generate_cu_files.py` loop.  Verified: undefined FA symbols **32 -> 0**, build green, FLASH_ATTN_EXT
  **6995/6995 with 448 head-40 and 1009 head-72 cases**, coherence unchanged, the generator idempotent on the
  tree, and `gate-norm-rows-alias.sh` links and runs (its remaining failure is R11's missing matcher).  It is
  **not** a rung: it is squashed into block 05 when `make-patches.sh` re-cuts the series at R18 (the ledger row
  is `R09r` in `RUNGS.md`).  It no longer blocks any gate.

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
* **Status:** **`CLOSED` at R11 (2026-11-13).**  The move is complete and the delivery's state follows it:
  `graph.cuh` owns `ggml_cuda_graph` / `ggml_cuda_stream_context` / `ggml_cuda_graph_instance`, while
  `ggml_backend_cuda_context` stayed in `common.cuh`.  Block 10's additions to the graph state were transplanted
  into `graph.cuh` — `capture_deferred` (a deferred capture while the slab's wide band was armed) and the
  **packed** replay guard (`src_properties` / `prop_nodes` / `prop_src_off` / `prop_srcs`), which replaces
  upstream's fixed-layout `node_properties`/`node_props` and removes its last user (the delivery's packed form
  spends 328 bytes/node against 1048, measured at 1.4 non-null sources per decode node).  Upstream's
  `ggml_cuda_graph_instance` + cluster snapshot is kept, so **`2e325dcd5`'s cluster-dim re-instantiation still
  fires** in `ggml_cuda_graph_update_executable()` (upstream's signature, with the delivery's HIP
  `hipGraphExecUpdate` leak workaround as its first body block).  The keyed cache (`ggml_cuda_graph_key`) and its
  eviction sweep are unchanged; the accessor is out-of-line in `ggml-cuda.cu`.  Verified by the R10/R11 coherence
  runs (graphs on by default); `gate-grid-stall-pad.sh` at R18 covers the pad site.

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
* **Status:** **`OPEN` mechanically, and it fired again at R11.**  Three lessons, each measured: (a) the moved
  declarations (`llama_set_embeddings_nextn`, `llama_set_nextn_layer_offset`, `llama_model_dflash_selector_top_k`,
  `llama_model_target_layer_ids(_n)`) are **not** re-added to `src/llama-ext.h` — upstream has public versions
  (R03); (b) `common/speculative.cpp` lost `#include "../src/llama-ext.h"` with the prune (R03) — a *build break,
  not a conflict*; (c) at R11 the prune's `29dbbad2b` had deleted **`ggml/src/ggml-cuda/diagmask.cuh`** itself,
  which block 10's extracted `ggml-cuda-try-fuse.cu` still included — dropped, nothing used from it.  Block 10's
  own `llama-ext.h` declarations all survive against `include/llama.h` (the build is the check).  Re-check at
  R12/R13 (`ggml.c` op table, RPC headers).

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
* **Status:** **`PARTIAL` — R01 resolved the only conflict; R10 took upstream's Windows rewrite wholesale.**
  R10's `src/llama-mmap.cpp` conflicts resolved to upstream's paired buffered/direct handle
  (`fp_win32`/`fp_win32_direct`, `init_buffered_io()`/`init_direct_io()`); upstream already carries
  `std::string fname;`, so the delivery's `llama_file::path()` and every `fname` use survive with no member
  redeclared, and the delivery's `llama_mmap::impl` work (`int fd`, the fragment-wise `unmap_fragment` with the
  post-unmap `MADV_DONTNEED`, r273) is untouched.  The substantive questions — is the PLE table still pinned,
  does the iGPU gating change the `--lazy-mode` fallback — are **R16/R17**'s.

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
* **Status:** **`CLOSED` for R10 (2026-11-13) — the union is validated, and the `-sm tensor` fault that was
  blamed on it is not it.**  `ggml-backend.cpp` merged as a union: upstream's
  `hv_tensor_requires_copy[hash_id(input)]` reachability test (set only when the *consuming node* carries
  `GGML_TENSOR_FLAG_COMPUTE`) **and** the delivery's `ggml_backend_sched_copy_input(..., st)` state argument, on
  both the non-host-weight and the host-weight pass.  The 1-device 35B-A3B census is the proof the delivery's
  staging survives it: **1991 fills through the copy callback** (`h=0.5205`, `takeover=540`).  The
  `gate-host-pool.sh` `tsplit` fault was attributed to this gate while R10 was parked and the attribution was
  **wrong**: the Meta backend at this rung has no `moe_cache_pool_backed` forwarding at all (the symbol exists
  nowhere in the tree) and **block 12 adds it** ("forward the pool-backed query to the simple devices.  A tensor
  split gives every simple device the SAME host master"), so a Meta-wrapped host expert weight is not reported
  as pool-backed and the fused kernel reads the pageable host master -> `Page not present` on both devices.  The
  two `tsplit*` arms are therefore **R13's** and must go green there.  **R11 closed the rest of it:** block 10's
  async input ring landed as the context-owned host-pinned **splice ring** (`splice_pin_slot` /
  `splice_pin_ring` / `splice_pin_idx` in `ggml_backend_cuda_context`) feeding
  `ggml_backend_cuda_set_tensor_2d_async()` for the strided-splice H2D, and it merged without touching the
  scheduler's upload/prefetch logic — upstream's `#31419` reachability test, `#32100` prefetch ordering and
  `#32089` graph-result reuse all coexist with it (the ring is *below* the scheduler).  The `-sm tensor`
  `tsplit*` arms remain the open R13 item; the ring's own A/B is `D4`/R18's `llama-bench` shape, not a
  correctness gate.

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
* **Status:** **`DECIDED` at R09 (2026-11-13) — the two destinations are complementary and both ship; the dual
  store is NOT collapsed.**  Upstream `#32135` made `rms_norm_f32` generic in the destination (`T * dst`,
  `rms_norm_store4()`, `ggml_cuda_cast<T>()`, `rms_norm_launch_typed()`) and added a typed *add* operand; the
  delivery's `uint16_t * dst16` side channel (`dst16c` + `norm_f2bf`, gated by `store_f32`) feeds the MMB/HC16
  producer slots and is an **independent second write to a different buffer**, so it cannot be expressed as
  upstream's typed destination.  Merged shape in both the 4-wide and the scalar path:
  `if (store_f32) { rms_norm_store4(dstc, i, r); }` **followed by** `if (dst16c != nullptr) { ... norm_f2bf ... }`
  (scalar: the delivery's `float v` chain, then `dstc[col] = ggml_cuda_cast<T>(v)` and the bf16 write).  Both
  includes kept (`convert.cuh` **and** `mmb.cuh`).  The `vec` gate keeps the delivery's opt-in latch
  (`ggml_cuda_rms_norm_vec()`, issue #229) on upstream's type-aware alignment test.  At R09 nothing fills
  `dst16` yet (`ggml_cuda_op_rms_norm_fused` passes null/true; the hook is block 15's), so the side channel is
  inert here and its own gate is **R16**'s.  Proven at R09 by: coherence **100 chars `sha=1c5d32ac537d`**
  (unchanged), `gate-hc-fusions.sh` **PASS** (7 arms byte-identical, 444 chars `sha=a512ee68c4b3`), and the
  Ling-3.0-tiny norm A/B **177 chars `sha=06e380d7475f`** identically with the fusion on and off.  Note for the
  r236 record: on this base the DSV4 HC pre/post ops are **byte-identical to their chain** (`a512ee68c4b3`),
  i.e. the r236 numeric-`SOFT_EXPECT` anchor (`6a102dbc62d6`) no longer describes them — a *stronger* result,
  recorded rather than "reported".  `gate-norm-rows-alias.sh` itself is **not** R09's (the r225 narrow-row
  fusion and its `norm-rows:` census arrive with block 10): it is owed at **R11**.  Its section A used to be
  blocked on `D5-a` (the driver link); that is **fixed** (`9c94ce139`) — the gate now builds and runs its driver
  and produces its recorded reference digest `bcc36422f9a120ac` on every arm with `fused=0 declined=0`, i.e. it
  fails only because the matcher it guards is not in the tree yet.

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
* **Status:** **`OPEN` (merge done at R10, verify at R16)** — R10 merged `common/fit.cpp` as upstream's
  `path_model` -> **`paths_model`** (`std::vector<std::string>`, the external-PLE form) **plus** the delivery's
  `host_expert_bytes` / `host_experts` out-params (the merged `common_get_device_memory_data_impl` already
  declares them with `= nullptr` defaults), and adapted the delivery's `dmds_min`/`dmds_ngl` preflight call
  sites and one `extra->path_model` that still used the single-path name.  The PLE `madvise(WILLNEED)` loop is
  block 12's (R13) and block 15's (R16); the loader-side question — does `--lazy-mode mlock`/WILLNEED still pin
  the PLE table on the new external-PLE load path — is R16's.

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
  **Partially verified at R10 (2026-11-13):** the slab and the arena allocate on the VMM pool with the revert in
  place — `SLAB_INIT dev=0 … slab 9.50 GiB`, per-device slabs under `-sm tensor` (`dev=0` 24.44 GiB / `dev=1`
  24.25 GiB), `MoE expert cache (auto): arena 8287.5 MiB`, and 1991 expert fills — so `#30542`'s halved
  reservation does not disturb the delivery's arena on this box.  Still owed at **R17**:
  `gate-uma-experts.sh --quick` on `halo` (the region size `-lmt`/`-lme` asks for).

## D18 — SYCL gained a device-side MoE scheduler

* **Upstream:** `18c5a95e2` (`#30156`) adds a device-side MoE scheduler to SYCL (fused gemm/mmvq, op-table
  flags, test-oracle entries).
* **Delivery:** block 09's host-resident MoE is CUDA/HIP-only, but the op/type tables and `supports_op`
  predicates are shared.
* **Relation:** cross-backend consistency only — do not "fix" SYCL, do not let the SYCL flags change the
  CUDA predicate.
* **Status:** **`CHECKED` at R10 (2026-11-13) — nothing to do.**  Block 09's conflict set contains no SYCL file
  and no change to a shared op-table flag, so the SYCL device-side MoE scheduler (`#30156`) and the delivery's
  CUDA/HIP host tier do not interact through the CUDA predicate.  The 1-device arena census (`h=0.5205`,
  `fills=1991`, `takeover=540`) is the evidence the CUDA side is unchanged.

## D19 — Recurrent cache tensor renaming removed in plamo2

* **Upstream:** `940a30ee3` (`#31984`) stops renaming the recurrent cache tensors in `src/models/plamo2.cpp`.
* **Delivery:** block 15's streamed KV and the recurrent memory instances the delivery's placement touches.
* **Relation:** verify: the delivery must not depend on the old names for a recurrent cache.
* **Status:** `OPEN` — R16.
