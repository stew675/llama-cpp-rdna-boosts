# The ladder — rungs, conflict sites, exit criteria and the ledger

One rung per delivery block, in block order (a later block's conflict is judged against the **rebased**
earlier blocks, so the order is not negotiable).  Block shas are the r235 chain (`6050b467c..69d36099b`).

How to work a rung: [`README.md`](README.md) §1.  Ledger columns: **status** `TODO` / `DOING` / `DONE` /
`DEFER`-`<decision>`; **sha** = the commit the rung produced on `rebase-r236`.

Legend for the conflict list: the paths the 2026-11-08 probe reported as conflicting for that block
(`CONFLICT-MAP.md`).  A path listed here is *expected* to conflict; a path that conflicts and is **not**
listed is a probe/cascade difference — note it in the rung log, do not assume the probe was wrong.

---

## R00 — setup · **DONE 2026-11-08**

* Campaign dir + `RUNGS.md` + `DECISIONS.md` + `CONFLICT-MAP.md` + `probe-conflicts.py`.
* `~/llama.cpp`: branch `rebase-r236` created at `a691b2345`; `master` untouched (pristine snapshot).
* `~/llama.cpp/build-rocm/bin-r235` — the r235 delivery binaries kept as the old-build reference
  (`build 11903, commit 69d36099b`).
* Probe: 16 blocks replayed with `git merge-tree` → 75 conflicted paths, blocks 04/06/14 clean.
* `wip/CAMPAIGNS.md` row + `WORKLOG.md` entry (campaign start; no delivery change).

---

## R01 — block 00: foundation (band uniformity, Meta wrapper, allocator) · **DONE 2026-11-08**

* block sha `a900d4703` · 56 files · +1366 / −909 · **expected conflicts: 12**
* **result: `4aaffd1c0` · 34 files · +986 / −340** (tree `10a57f7c4`), build green (1m22s, clean
  configure), coherence green, `test-backend-ops -o MUL_MAT_ID` **931/931** on gfx1201

| conflicted path | what to expect |
|---|---|
| `common/speculative.cpp` | upstream reworked the spec paths (#30141 MTP hidden state, #31350 draft checkpoint restore) |
| `ggml/src/ggml-backend-meta.cpp` | #31447 (compute view minimum 256) lands in the same allocator function our compute-margin hunk touches — **`D7`** |
| `ggml/src/ggml-cuda/mmf.cuh`, `mmid.cu`, `mmq.cu`, `mmq.cuh` | #31047 linearised MoE layout, #31188 gate/up+GLU, #31296 NVFP4 — the delivery's `mmid`/`mmq` band/census hunks re-derive |
| `src/CMakeLists.txt` | new model files upstream; our block-10/11 TU registration is not here yet (it is added by those blocks) — resolve additively |
| `src/llama-graph.cpp` | #31733 (`model` is now a class member of `llm_graph_context`), #31142 output structs |
| `src/llama-kv-cache.cpp` | #29510 `flash_attn_ext_rows`, #31073 rotation constants, #31683 kv shifting |
| `src/llama-model.cpp` | the model-loader refactors (#31141 `no_alloc`, #31142) |
| `src/llama-moe-cache.cpp` | the delivery touches this file only 13 lines (split-mode check); upstream rewrote the cache (#31233 ClassLRU) — **`D1`** adjacent, but this hunk is mechanical |
| `src/llama-vocab.cpp` | vocabulary special-token bookkeeping (block 13 depends on it) |

**Extra:** block 00's `GGML_COMPUTE_BUFFER_MARGIN_PCT` invariant is now partly duplicated by upstream #31447
— record whether the margin still fires (see `D7`); do not remove it at this rung.

**Exit:** build + coherence + `git grep '<<<<<<<'` clean + log.

### R01 resolution log (what was resolved, and why)

Upstream met this block in three places: `#31047` (MoE data-layout linearisation → `tile_spec`), the
graph/model/memory-instance refactors (→ 15 conflicts in `llama-model.cpp`), and `#31447` (Meta compute
views).  Everything else in the block was **rename-only `-Wshadow` ballast** and was not carried.

| site | resolution |
|---|---|
| `ggml-cuda/mmid.cu` | **re-implemented**: the delivery's per-occurrence compaction (generic + the pow2 fast path + the 512-expert/10-used kernel, `GGML_CUDA_DISABLE_MMID_512` kept) now emits upstream's `tile_spec` instead of `expert_bounds`.  Upstream's seam writer is reused verbatim (one seam-aligned entry per tile + one trailing entry per expert); the shared-memory `mm_ids_helper_store` staging buffer is gone with it.  Rationale: upstream has no duplicate handling at all (its `iex_used` is a single slot per token), so this is **not** a subsumption — it is our fix re-expressed on their grid, and it is now *simpler* (no in-kernel loop). |
| `ggml-cuda/mmq.cuh`, `mmq.cu`, `mmf.cuh`, `mmf.cu` | **upstream's** (`tile_spec` grid, `ntj` = `ceil(n_tokens*n_expert_used/J)+n_experts`, `fusion_device`) — the compact column count is exactly `n_tokens*n_expert_used`, so the grid covers the expanded list and the delivery's column-tile loop / `stream_k ? ne_get_rows : ne12` width hack are unnecessary.  Kept the delivery's **correctness gate**: `mul_mat_f_switch_ids` takes the compact kernel for **every** `MUL_MAT_ID` (upstream still uses the single-slot `mul_mat_f` path for `ncols_dst <= 16`). |
| `src/llama-model.cpp` | upstream's text + the three substantive delivery hunks re-applied: the `mlock_lazy` member, the `ml.lazy.mlock_ranges(...)` call in `load_tensors` (`--lazy-mode mlock`), and the `!is_resident` gate on the mmap-buffer path. |
| `ggml-backend-meta.cpp` | **both**: `compute_headroom = 128` (ours) *and* upstream's `compute_headroom_min = 256` floor in the `std::max(...)`.  Whether the multiplier is still needed is `TODO #286` / `D7`. |
| `src/CMakeLists.txt` | upstream (`Threads::Threads`, `#31785`); the `-Wshadow` enablement below it is ours and stays. |
| `common/speculative.cpp` | ours (the `gemma4-assistant` arch guard for `is_mem_shared`; upstream bug #23398 is still unfixed upstream) with the duplicate `llama_set_embeddings_nextn` calls dropped (the post-merge context already has them). |
| `src/llama-vocab.cpp` | ours (non-recursive `naive_trie::insert`; upstream's recursion on a GGUF-length token is the bug it fixes). |
| `src/llama-kv-cache.cpp`, `src/llama-moe-cache.cpp`, `src/llama-graph.cpp`, + 17 `src/models/*.cpp` | **upstream's, delivered hunks dropped**: the whole block-00 delta in these files is `-Wshadow` variable renames (`n_stream`→`n_stream_l`, `params`→`params_l`, `s`→`s_seq`, …), verified rename-only line-for-line before dropping. |

**Decision taken (reversible, needs the maintainer's blessing): stop carrying the `-Wshadow` rename sweeps.**
Measured cost: ~1500 diff lines in 21 files, all of which the later blocks and every future re-base touch again
(the repo's own `TODO #101` records the same cost for the `llama_model_loader * ml` rename).  Measured
consequence of dropping them: **94 `-Wshadow` warnings** in the llama target (the r235 CMake comment claimed
9 standing ones), 108 warnings total, none of them errors (`-Werror` is not on for this build).  `-Wshadow`
itself stays enabled for new code.  If the maintainer prefers a quiet log, the renames can be re-applied per
file *as those files are already being resolved* at a later rung — the ballast is mechanical.

**Traps this rung produced** (now in `README.md` §1):
* a rung build needs a **clean configure** (`~/bin/build-llama-rocm clean`) — a reused `CMakeCache.txt` keeps
  the *tip's* options (`GGML_CUDA_FA_QUANTS` carries block 05's `iq4_nl` instances, which the base rejects at
  configure time: this is the documented "r2 lesson" of `gate-per-block-build.sh`).
* `GGML_CUDA_FA_QUANTS` is set by hand on this box, so at **R06** (block 05 lands) the `iq4_nl-iq4_nl`
  instance must be re-set for the release configure.

- status: **DONE** · sha: `4aaffd1c0`

## R02 — block 01: hybrid HIP all-reduce · **DONE 2026-11-08**

* block sha `df940c14c` · 4 files · +2365 / −44 · **expected conflicts: 1**
* **result: `cadab8c45` · 4 files · +2532 / −46**, build green, coherence green, and the hybrid arm came up on
  the 3-GPU `-sm tensor` arm (`ggml_cuda_ar_pipeline_init: initialized HIP AllReduce pipeline: 3 GPUs,
  1024 KB chunked kernel staging + 32 MB copy-engine staging per GPU`)

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | #28967 (NCCL multi-node TP) and the `ggml_cuda_allreduce_*` neighbourhood; our hybrid dispatch is a sibling of `nccl`/`internal`. **`D10`** |

**Exit:** build + coherence + the all-reduce arm is still reachable
(`GGML_CUDA_ALLREDUCE=hybrid` on `-sm tensor --device 0,1,2`, `-lv 4` census line).

**Resolution:** all three sites are *both* sides, not either/or.
* the `comm_context` struct keeps upstream's `world_size` + `n_ranks()` (#28967 multi-process NCCL) **and** the
  delivery's copy-engine scratch block / `nccl_failed` / r165 auto-select state;
* the NCCL small-tensor crossover now speaks in **ranks** (`is_small(ne, n_ranks())` in both call sites) and
  keeps the delivery's measured **131072** threshold for <= 2 ranks (upstream's table says 32768 there);
* `ggml_backend_cuda_comm_init_rank()` (upstream's multi-process bootstrap) is kept as-is next to the
  delivery's hybrid dispatch comment.

- status: **DONE** · sha: `cadab8c45`

## R03 — block 02: adaptive MTP draft depth · **DONE 2026-11-08**

* block sha `470729be7` · 24 files · +1186 / −54 · **expected conflicts: 4**
* **result: `2f0385c71` · 23 files · +1381 / −57**, build green, coherence green, and the adaptive controller
  is live on the MTP arm (`-lv 4`: `adaptive draft depth enabled (draft-mtp-adaptive)`, draft acceptance
  0.849, mean len 3.50 on Qwen3.6-35B-A3B `-sm tensor` 3 GPUs)

| conflicted path | what to expect |
|---|---|
| `common/common.cpp` | arg plumbing; upstream added many flags (`--moe-cache-*`, realtime, router margin) |
| `common/speculative.cpp` | `D11`: #30141 (MTP hidden state save/restore), #31350, #31497 (no MoE cache on the draft) |
| `src/models/delta-net-base.cpp` | #30878 conv-state refactor — the adaptive controller's snapshots ride on this |
| `tests/CMakeLists.txt` | additive |

**Exit:** build + coherence + the adaptive controller still selects > 1 draft depth (`-lv 4` accept line) on
the MTP config; MTP four-axis is R17's job, not a per-rung cost — a single `-n 300` smoke is enough here.

**Resolution (and one upstream absorption worth knowing):**
* `common/common.cpp` — upstream factored the whole fit block into `common_fit_apply()`, **already including
  the draft/MTP extra model** (`shares_model = !has_draft`, `ctx_type = MTP`), so our inline copy is dropped
  and the call kept.  (Upstream detects the MTP type by `std::find` over `speculative.types`; the merged
  function keeps that.)
* `src/models/delta-net-base.cpp` — **upstream absorbed this delivery mechanism and went further**:
  `build_conv_state_store()` (upstream, `#30878`+`#31660`) writes the rollback snapshots with **one
  `ggml_set_rows` per conv layer** and already bounds them at `Kw = min(n_seq_tokens, K)` — the delivery's
  `t_min` bound — and its comment cites `rs_max` (upstream `#31368`).  Upstream's helper is taken; the
  delivery's fused-GDN path (blocks 03/10) must keep writing the same `Kw` snapshots, which is the same rule.
  This is a *good* absorption: same invariant, one op instead of `K` copies per layer.
* `common/speculative.cpp` — both sides again: upstream's `llama_set_embeddings_nextn` calls and shared-KV row
  reduction (`is_mem_shared`, `all_rows_output`) merged with the delivery's prompt-tail reduction
  (`ingest_prompt_end`, one row for a prompt batch) and its `accept_partial()`/`adaptive_feedback()` pair,
  which sits beside upstream's new `get_state`/`set_state` (the boundary stash) in the same impl class.
  The merged row selection is: shared-KV non-output -> 0 rows; prompt tail -> last row; shared-KV mixed
  outputs -> last row; otherwise every row.
* `tests/CMakeLists.txt` — upstream moved the generated-model fixtures out of the `NOT WIN32` branch
  (`#30880`), so our two `test-recurrent-state-depth` registrations were re-placed next to `test-llama-context`
  in the unconditional region; `llama_build(test-fusion.cpp)` is upstream's own test (the delivery's block 10
  does not add a file of that name — checked: both sides are the same 589-line file).

- status: **DONE** · sha: `2f0385c71`

## R04 — block 03: chunked gated-delta-net prefill · **DONE 2026-11-08 (`D4` DECIDED: tracker dropped)**

* block sha `670aa557d` · 25 files · +3182 / −15 · **expected conflicts: 1** (probe said 1 file; actually 2)
* **result: `5a214716b` · 25 files · +3324 / −25**, build green, coherence green, chunked GDN dispatch proven
  (`-lv 4`: `whole-batch chunked GDN prefill n=32 K=1 n_seqs=1 (> 16: no snapshot tail needed)`), and the
  block's own A/B is byte-identical (`GGML_CUDA_GDN_CHUNKED=0` vs default, 199 chars,
  sha `a2ce6989d94f`)

| conflicted path | what to expect |
|---|---|
| `src/llama-memory-recurrent.cpp` | **`D4` — the decision point of this rung.**  Upstream #31368 replaced "accept any rollback up to `n_rs_seq`" with a per-ubatch `rs_max` bound (and refuses past it); our r91 fold tracks the ubatch's rollback count and *warns*.  Same defect (issue #131), two solutions |

**Stop here for the maintainer** with: upstream's enforcement vs our tracker (which is stricter, which keeps
the `LLAMA_RS_ROLLBACK_DEBUG` diagnostic), the `archive/work/rs-rollback-boundary/` record, and the
`test-llama-context` rollback cases that now exist upstream (`rollback_bound`).  Recommended default:
upstream's `rs_max` enforcement + keep the diagnostic warn, drop our duplicate bookkeeping — pending the
maintainer's call.

**Exit:** build + coherence + chunked GDN still selected (`GGML_CUDA_GDN_CHUNKED` census) + the rollback
case above resolved and recorded.

**What the merge actually produced, and what the maintainer decided (2026-11-08: drop the tracker):**

| site | resolution |
|---|---|
| `src/models/delta-net-base.cpp` | **upstream's `build_conv_state_store()`**.  Block 03's inline store (per-snapshot `ggml_cpy` loop + the pre-batch copy into slot `n_seq_tokens`) is dropped: upstream's helper already writes `Kw = min(n_seq_tokens, K)` slots in one `set_rows`, and with `#31368`'s bound (`rs_max = min(n, n_rs_seq)` for a short ubatch, `0` for a longer one) the *pre-batch* slot is unreachable — a legal rollback is always `< n`.  Keeping our block would have written the same slots twice. |
| `src/llama-memory-recurrent.cpp` ctor | **both**: upstream's `no_alloc`/`ctxs_bufs(no_alloc)` (`#31141`) plus the delivery's `n_rs_batch` parameter (the call sites on both sides already pass both). |
| `src/llama-memory-recurrent.h/.cpp` | **the r91 ubatch rollback tracker is removed** (`last_ubatch_nseq_tokens[_by_seq]`, `last_ubatch_pos_last[_by_seq]`, `warned_rollback_boundary`, the RSROLL trace and the crossed-boundary warn): `seq_rm` keeps upstream's `rollback <= cell.rs_max` and nothing else.  `n_rs_batch` **stays** — it is not tracker bookkeeping but the GDN chunked threshold floor (`max(K, 16, n_rs_batch)`, `ggml_gated_delta_net()`'s op parameter) and the slab's verify reserve.  `LLAMA_RS_ROLLBACK_DEBUG` **stays** as the RSPREP prepare-order instrument (it never needed the tracker). |

**The tracker's removal exposed a real bug in the merged code — found, fixed, and it is also an upstream bug.**
Upstream's `find_slot()` gather/re-order loop swaps a cell's `pos`, `src` and `seq_id` (and re-points the tails)
but **not** `mem_cell::rs_max`, so a moved state kept the *destination* cell's bound while its snapshots live in
the *source* cell (the r195 plane move).  Symptom, before the fix, on the delivery's own
`test-recurrent-state-depth`: Phase C (moved cell) refused rollbacks 2..4 and 6..8 at `n_rs_seq` 4 and 8 in the
batched `[seq 0, seq 2]` arm only; after `std::swap(dst_cell.rs_max, src_cell.rs_max)` **all three phases pass**
(`PASS: total failures = 0`).  Upstream has the same omission and its `rollback_bound` cases never move a cell →
`TODO #289` (an `upstream/` PR: the one-line swap + Phase C as the test).

**Test contract updated (block 03's `tests/test-recurrent-state-depth.cpp`).**
* Phase A (the shipped speculative verify shape, `n_tokens = K = n_rs_seq+1`, rollback 1..`n_rs_seq`) —
  **PASS 1..15**, unchanged: `rs_max = n_seq_tokens - 1 = n_max`, so the real deep-draft rollback is legal.
* Phase B used to assert parity for a rollback into a batch of `n_rs_batch > K` tokens, which the delivery's
  tracker allowed and `#31368` refuses (`rs_max = 0` for `n_seq_tokens > n_rs_seq + 1`).  It now **asserts the
  refusal** instead (the old relaxation is gone with the tracker).  This costs nothing in a shipped config:
  `cparams.n_rs_seq = draft.n_max` and `n_rs_batch = draft.n_max + 1`, i.e. `n_rs_batch == n_rs_seq + 1`
  always, so a verify batch is exactly at the bound.
* Phase C — the moved-cell parity case, now passing (see above).

**Documentation owed at R17** (nothing is delivery until then): `ENVIRONMENT.md`'s `LLAMA_RS_ROLLBACK_DEBUG`
row loses its RSROLL half (RSPREP only) and `GGML_CUDA_GDN_CHUNKED`'s `n_rs_batch` note stays valid;
`patches/README.md`'s block-03 section loses the tracker/warn text; `INVARIANTS.md`'s guard entry is corrected
to "upstream's `rs_max`, with the delivery's diagnostic".

- status: **DONE** · sha: `5a214716b`

## R05 — block 04: BF16 KV cache and native-BF16 flash attention · **DONE 2026-11-08**

* block sha `4db2b86d9` · 8 files · +630 / −214 · **expected conflicts: 0** — clean, as the probe predicted
* **result: `a4c3c35c5` · 8 files · +602 / −191**, build green, and the bf16 arm runs:
  `llama_kv_cache: … K (bf16): 4096.00 MiB, V (bf16): 4096.00 MiB` with `flash_attn = enabled`

`D6`: upstream `162e2bb2e ggml : fix bf16` touches `ggml/src/ggml.c` + `tools/export-lora` — check it does
not invalidate a delivery bf16-KV assumption (it is a type-traits/lora fix, so the expectation is "no"), and
that `-ctk bf16 -ctv bf16` still runs the native FA arm.

**Exit:** build + coherence + a bf16-KV arm generates (`-ctk bf16 -ctv bf16`, native FA line at `-lv 4`).

**Result:** `D6` verified as "independent", as expected — upstream's `#31609` touches `ggml.c`'s type traits and
`export-lora`, not the bf16 cache type or the FA arm.  The block needed no hunk.  (`FA_QUANTS` on this box shows
`bf16-bf16` in the instance list, so the native bf16 FA arm is compiled in.)

- status: **DONE** · sha: `a4c3c35c5`

## R06 — block 05: flash-attention band, native quantized K/V, derived kq mask · **DONE 2026-11-08**

* block sha `41248b885` · 295 files · +5565 / −624 · **expected conflicts: 9 files** — found 9 files /
  **48 hunks** (the probe counted files, not hunks: `fattn-mma-f16.cuh` alone has 8, `llama-graph.cpp` 22)
* **result: `b5d32e0c7` · 295 files · +5565 / −623**, build green, and:
  * `derived kq mask flash attention enabled` (the probe now passes; see the ABI finding below),
  * derived vs `LLAMA_KQ_MASK_DERIVED=0` **byte-identical** at a 4096-ctx prefill (149 chars, sha
    `30c7fc8705a6`),
  * `-ctk q8_0 -ctv q8_0 -fa on` runs (`K (q8_0): 2176.00 MiB, V (q8_0): 2176.00 MiB`), coherence green.

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh`, `fattn.cu` | **`D5`** — #29510 `flash_attn_ext_rows` restructured the unified-KV read; the band kernels' dispatch table and the "which head/tile cap applies" logic must be re-derived, not patched blind |
| `src/llama-context.cpp` | #29510/#31073/#31683 call-site changes |
| `src/llama-graph.cpp`, `src/llama-graph.h` | `D5`/`D9`: the mask-fill and `flash_attn_ext` build shape |
| `src/llama-kv-cache.h` | the derived kq mask's cache side |

**Stop for the maintainer if** #29510's shape makes our band dispatch redundant at a head size/quant we
validate (that is a delivery-code drop).  Otherwise: re-derive the dispatch on the new signature, keeping the
delivery's band kernels.

**Exit:** build + coherence + `scripts/gate-prefill-logits.sh` on the FA-relevant arms is deferred to R17,
but a *same-build* A/B (`GGML_CUDA_FA_*` kill switch) must show the delivery arm still fires at `-lv 4`.

### R06 resolution log

`D5` is resolved the "complementary" way: **no delivery kernel was dropped.**  Upstream's entry-point
restructure is absorbed, and the delivery's band/tile/prefill kernels keep their dispatch.  What took the time
was a silent ABI collision and two upstream API removals that our code still used.

| site | resolution |
|---|---|
| `fattn-common.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh`, `fattn-mma-f16.cuh` | **both parameters** (upstream's `KV_rows` then the delivery's `kq_blocks`); loaders take upstream's kv_rows `indices` with the delivery's native-KV arguments (`swz`, `kv_native.*`); the VKQ C-tile sizing keeps the delivery's fp32-aware expression under upstream's renamed macro `AMD_WMMA_256B_AVAILABLE` (256B == RDNA3; upstream's branch is the same case, its expression is the fp32-at-`DV%32==0` bug our comment records). |
| `fattn.cu` | upstream's predicates + our `has_mask` helper extended to the derived form (see below): `use_gqa_opt`/`gqa_opt_applies` now count all three mask forms and test `K->ne[1]` for the stride (upstream's), not the kv_rows count. |
| `llama-graph.h`, `llama-graph.cpp` | upstream's **`llm_attn_mask` (raw/rows)** replaces the `self_kq_mask`/`_cnv` members; `build_attn_inp_kq_mask` stays the delivery's derived-aware **member** and `build_attn_inp_attn_mask` becomes a member too (it adds inputs, so it needs `res`/`ctx0`), with `allow_derived` threaded through.  The derived registration lives in the dense (non-`fused_kv_rows`) branch.  `llm_graph_input_kq_derived`'s `set_input`/`can_reuse` are restored. |
| `llama-context.cpp` | both: upstream's gemma4-assistant FA follow-through *and* the delivery's derived-mask env/probe block (`LLAMA_KQ_MASK_DERIVED`, `LLAMA_KQ_MASK_PACKED_RESERVE`, the multi-stream refusal); `graph_reserve` grows our `packed_kq_mask` argument on top of upstream's `no_alloc()` rename. |
| `llama-kv-cache.h` | upstream's declarations plus the delivery's derived-mask API. |

**Finding (correctness, silent): the derived kq mask ABI had to move — both mechanisms claimed `src[5]`.**
Upstream's `ggml_flash_attn_ext_rows` stores its `kv_rows` tensor in `dst->src[5]`; the delivery's
`ggml_flash_attn_ext_add_kq_derived` stored `cell_pos` there.  Merged as-is, the derived probe resolved to
`derived kq mask flash attention not supported` ("layer 3 is assigned to device ROCm0 but derived kq mask flash
attention is assigned to device CPU") and **every batch silently fell back to the packed mask** — no error, no
warning, just the 860 MiB the derived form exists to save.  Fix: the derived form is now `src[6]`/`src[7]`/`src[8]`
(`ggml.c`, `ggml-cpu/ops.cpp` ×2, the CUDA launcher, `fattn.cu`'s `has_mask` and support gate), with `src[5]`
reserved for kv_rows.  After the move the same probe prints `derived kq mask flash attention enabled`.  **Trap
for future ops: `FLASH_ATTN_EXT`'s `src[5]` is taken.**

**Finding (upstream absorbed a delivery mechanism): the rotation constants are no longer filled per ubatch.**
The block called `set_input_k_rot`/`set_input_v_rot` on the cache, an API that exists only at the base —
upstream `#31073` ("keep rotation constants on the GPU") removed it and consumes `get_k_rot()` at **graph-build**
time (`llama_mul_mat_hadamard`).  Our calls had no definition left to link to, so they and their two header
declarations are dropped.  No loss: the rotation is applied where upstream applies it.

**R17 note:** with block 05 in, the release configure's `GGML_CUDA_FA_QUANTS` may carry the `iq4_nl-iq4_nl`
instance again (the box sets it by hand; the clean configure used the default `q4_0,q8_0,f16,bf16`).

- status: **DONE** · sha: `b5d32e0c7`

## R07 — block 06: MMB quantized prefill GEMM · **DONE 2026-11-08**

* block sha `398fe6ab3` · 7 files · +2903 / −3 · **expected conflicts: 0** — clean, as the probe predicted
* **result: `75ba21116`** · 11 files · +2955 / −17 (MMB + the mm_ids_helper API change + `mmf32.cu` + `tests/test-logits-width-probe.cpp`), build green,
  `MMB_DENSE` fires (`MMB_CFG … dense_geom=1 … routed=0`), `GGML_CUDA_MMB=0` **byte-identical**
  (108 chars, sha `d94aef3f5033`) on Qwen3-Coder-30B-A3B.

`#31047` (linearised MoE layout) and `#31296` (NVFP4) are in the same files but did not conflict in the
probe — confirm the MMB dispatcher keys still compile and the RDNA arm is still selected.

**Exit:** build + coherence + MMB still dispatched (`-lv 4` line) on a Q8_0 prefill.

**Resolution:** the block applied clean but its **routed** path called the pre-re-base
`ggml_cuda_launch_mm_ids_helper`, which produced only `expert_bounds`.  Since block 00's re-implementation
emits upstream's `tile_spec`, the helper gained an **optional `expert_bounds` output**: one compaction serves
MMQ/MMF (`tile_spec`) and MMB (per-expert bounds), and both MMB call sites allocate the tile_spec buffer they
never read.  No MMB kernel logic changed.  (On gfx1201 `mmb_cfg().routed == 0`, so the routed path this touches
is RDNA3 coverage; the compile break was real either way.)

- status: **DONE** · sha: `75ba21116`

## R08 — block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad) · **DONE 2026-11-08**

* block sha `98bb7bc61` · 9 files · +879 / −137 · **expected conflicts: 1** — found 1
* **result: `f0a594fe1` · 9 files**, build green, coherence green (the rung's own mechanism lines fire)

| conflicted path | resolution |
|---|---|
| `ggml/src/ggml-cuda/mmvq.cu` | upstream's `get_mmvq_mmid_max_batch` arch table (`#31679`), the Q8_1 correction sum (`#31247`) and the per-shape `nwarps` work all land in the same neighbourhood as the delivery's VDR/Q6_K J bands.  Resolved as **both**: upstream's table and family split kept, the delivery's `mmvq_mmid_max_batch_band()` clamp layered on top of it (the two-arg `(cap, floor)` form, with `RDNA3_5` keeping its own floor and `RDNA4_M` following upstream). |

**Ledger-gap note (found 2026-11-08 at R11):** this rung was carried out and committed, but its
section and ledger row were left as `TODO`/empty — the ladder went R07 → R09 in the record while
the chain had the commit.  The row is filled below; the block's own verification is the standard
one (build + coherence + the touched quant types), and `test-backend-ops -o MUL_MAT_ID` (931/931)
and the per-quant `-o` sweeps were covered again at R09/R11.

- status: **DONE** · sha: `f0a594fe1`

## R09 — block 08: fused kernels · **DONE 2026-11-08 (`D2` DECIDED: complementary)**

* block sha `615e77323` · 35 files · +5500 / −440 · **expected conflicts: 5** — found 5 files / 8 hunks
* **result: `af0cae2f1`** · build green, `test-backend-ops -o MUL_MAT_ID` **931/931**, and same-seed
  greedy text byte-identical across default / `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` /
  `GGML_CUDA_DISABLE_FUSION=1` (121 chars, sha `cecacc0e438e`) on Qwen3-Coder-30B-A3B

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | op dispatch table; upstream added fused MoE entry points (#31188/#31288/#31296) |
| `ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh` | **`D2`** — upstream's fused gate/up+GLU lands in the same dispatch |
| `ggml/src/ggml-cuda/mmvq.cu` | same |
| `ggml/src/ggml-cuda/quantize.cu` | #31247 Q8_1 correction sum vs our Q8_1 cache keying |

**`D2` expectation:** arch-disjoint by construction (like #29948) — both fused paths ship on one dispatch,
the RDNA arm guarded by the delivery's arch/gate predicate, upstream's NVFP4/Turing path untouched.  Record
which predicate decides each arm and prove the RDNA fused arm still fires (`-lv 4`).

**Exit:** build + coherence + `scripts/gate-norm-rows-alias.sh` (the block's own gate) green.

**Correction to this rung's exit criteria:** the norm-rows gate **cannot** run here.  The fusion *dispatcher*
(`ggml-cuda-try-fuse.cu`) is **block 10's** TU, so on a blocks 00-08 tree every fusion matcher is compiled but
never invoked — measured: `gate-norm-rows-alias.sh` reports `fused=0 declined=0` on all four arms while
`nofuse` passes.  The gate therefore moves to **R11** (block 10) and must be green before R17.

**Resolution (`D2` — upstream's `has_fusion` vs the delivery's `has_gate`): both, as two template arms on one
kernel.**  Upstream's fused gate/up+GLU in MMQ (`#31188`/`#31288`/`#31296`, the interleaved/NVFP4 path) and the
delivery's RDNA fused gate are selected by separate predicates and share `mul_mat_q_process_tile`, so this is
the `#29948` pattern again — no delivery kernel dropped.

| site | resolution |
|---|---|
| `mmq.cuh` (3 hunks) | the delivery's routed-compact MoE dispatch (descriptors per real `(expert, J-tile)` pair) is kept, followed by upstream's fused/tiling branch **extended** with `has_gate`: `x_gate_arg`/`glu_op_arg`/`glu_limit_arg` chosen per arm and `x_scale_arg`/`gate_scale_arg` nullptr on the delivery's arm; the stream-k launch likewise; the MUL_MAT_ID tile body takes upstream's `tile_spec` call shape with `has_gate` as the extra template argument. |
| `mmq_args` | gains a per-expert **`expert_bounds`** next to upstream's `tile_spec`: the routed-compact dispatch keys on bounds, upstream's grid on tile_spec, and one compaction now emits both (see below). |
| `mmid.cu` | `ggml_cuda_launch_mm_ids_helper`'s `tile_spec` became **optional** (the bounds-only callers — the delivery's Q8_1 pair fusion, MMB — pass `nullptr`, and the seam writer returns early). |
| `quantize.cu` | the delivery's chunked/scatter quantizer body wins over the pre-restructure text it conflicted with (block 08 restructured that kernel; upstream's loop header is the same shape). |
| `mmvq.cu` (2 hunks) | the delivery's `mmvq_mmid_max_batch_band()` clamp plus upstream's `RDNA4_M` family. |
| `ggml-cuda.cu` | the delivery's SSM conv-input fusion block and the NVFP4 scale helpers (upstream had removed the block this hunk replaced). |

**Correction (2026-11-08, found at R11 while unblocking it): the "byte-identical across default /
`DISABLE_MOE_MMQ_FUSION` / `DISABLE_FUSION`" claim above was vacuous.**  The fusion *dispatcher*
(`ggml-cuda-try-fuse.cu`) is block 10's TU, so on a blocks-00-08 tree every matcher is compiled but
never invoked: the three arms ran the identical code path and of course produced identical text.
It is a true statement about this build only now that R11 exists, and it was re-measured there —
default / `GGML_PAIR_OFF` / `GGML_CUDA_DISABLE_MMQ_ROUTED` / `GGML_CUDA_DISABLE_FUSION` all read
`013d09f165a5` on Qwen3-Coder-30B-A3B.  Do not use a pre-R11 rung's fusion A/B as evidence again.

- status: **DONE** · sha: `af0cae2f1`

## R10 — block 09: host-resident MoE · **DONE 2026-11-08 (`D1` MEASURING — numbers in, decision pending)**

* block sha `11c5544be` · 68 files · +16121 / −540 · **expected conflicts: 7** — found 7 files / 12 hunks
* **result: `5654a5f95` · 68 files · +17412 / −541**, build green, dense coherence green

| conflicted path | what to expect |
|---|---|
| `common/common.cpp`, `common/fit.cpp` | `--moe-cache-*` flags and the new fit terms (#30966, #21231 router margin) alongside our pool/arena knobs |
| `ggml/src/ggml-backend-meta.cpp` | Meta forwarding of the delivery's iface members (remap/gather/update) |
| `ggml/src/ggml-backend.cpp` | the scheduler's copy hook: `D1` — upstream's cache copy callback and our pool fill share this entry |
| `ggml/src/ggml-cuda/common.cuh` | arch predicates (also #31373 RDNA4_M) |
| `src/llama-context.cpp` | `D1` — the context creates *either* the upstream `llama_moe_cache` (`--moe-cache-mib`) or our arena; upstream's CPU-share needs both to coexist |
| `src/llama-cparams.h` | additive |

**`D1` is this rung's decision.**  Our design already keeps upstream's cache as a *second, dormant* path
(`cparams.moe_cache_size` "upstream MoE cache (dormant unless --moe-cache-mib)") — so the shape to preserve is
coexistence.  Questions for the maintainer, with measurements:

1. Should the pool tier adopt upstream's `--moe-cache-cpu-share` idea (compute a share of misses on the CPU
   from the host copy instead of uploading)?  It is a *different* mechanism from our fill: our fills upload,
   upstream computes.  On the delivery's tiers the host copy is pinned, so upstream's own doc says the share
   "often does not help" — but it is exactly the `-sm layer`/near-mirror case our pool declines.
2. Does upstream's ClassLRU (#31233) admission beat our heat/RFU-based admission?  Only reachable for the
   dormant cache today; if it wins it would have to be ported *into* the arena, which is a real change.

**Exit:** build + coherence + `scripts/gate-host-pool.sh` (7 arms, `takeover=0` under the fill faults) +
the census lines fire.

### R10 resolution log

| site | resolution |
|---|---|
| `common/common.cpp`, `common/fit.cpp` | upstream factored the `--fit` block into **`common_fit_apply()`**; the delivery's draft reserve moved *into* it as `common_draft_reserve_derive()` (shared with the ctor's MoE preflight, which needs the same value).  `fit.cpp` keeps both sides: upstream's RAII `common_fit_log_scope` plus the delivery's host-expert accounting and its `hp_ngl_skip` + host-expert out-params — which also repaired the call sites written before upstream added the `n_skip` parameter. |
| `ggml/src/ggml-backend.cpp` | upstream's prefetch-split early-out **and** the delivery's graph-input sync condition; both event families freed at teardown (`prefetch_events` + the `GGML_SCHED_STAGE_SLOTS` pairs). |
| `ggml/src/ggml-cuda/common.cuh` | the delivery's **keyed** CUDA-graph cache (subgraph weight identity, r149) with upstream's idle-eviction window (`#30662`): eviction compares the key, not the first node pointer. |
| `ggml/src/ggml-backend-meta.cpp` | the delivery's `GGML_META_COMPUTE_HEADROOM` knob **plus** upstream's 256-view floor (`#31447`). |
| `src/llama-context.cpp`, `src/llama-cparams.h` | upstream's `kv_rows` auto-resolution and `moe_cache_cpu_share` plus the delivery's `drop_compute_buffers`, `fused_hc_mix`/`combine` and host-table accounting; `model.hparams.no_alloc` → `model.no_alloc()` (`#31141`). |

**Finding (build-level, blocks the whole host-expert tier on a default configure): the arena needs VMM.**
`ggml_cuda_slab_init_locked()` declines when the device does not report VMM, and **upstream's HIP default is
`GGML_HIP_NO_VMM=ON`** (`ggml/CMakeLists.txt:219`), which neither `~/bin/build-llama-rocm*` overrides.  On
`build-rocm` every host-expert run prints `the movable-boundary slab was armed … but could not be created (the
device does not report VMM …). Disabling the MoE expert cache and streaming the experts` and silently streams —
so a host-expert measurement on that build measures *stock `-ncmoe`*, not the delivery.  A second build dir
(`build-rocm-vmm`, `-DGGML_HIP_NO_VMM=OFF`) was made for the `D1` numbers below.  **Decide before R17: should
the delivery's build docs/scripts carry `GGML_HIP_NO_VMM=OFF` (and the ENVIRONMENT.md row at line 962 is
currently "an open investigation") — it is the difference between the tier existing and not.**

### `D1` first measurement (1 × R9700, 35B-A3B Q8_0, `-ncmoe 20`, `-c 8192`, 16.3 GiB host experts, VMM build)

| arm | cache | prefill t/s | decode t/s | text |
|---|---|---|---|---|
| A | delivery arena, `MOE_EXPERT_CACHE_MIB=4096` (realized 4080 MiB) | **547.6** | 53.5 | 533 chars `8d6e74e98d89` |
| B | upstream `--moe-cache-mib 4096` (our arena off) | 544.2 | **62.9** | identical |
| C | none (`MOE_EXPERT_CACHE_MIB=0`, stock `-ncmoe`) | 470.9 | 49.0 | identical |

Upstream's ClassLRU cache is **+17.6 % decode** over the delivery's arena at the same MiB on this shape, and
both beat streaming.  This is one shape, not the delivery's field config (2-3 GPU `-sm tensor`, bf16 KV,
262144 ctx), and the arms are either/or — coexistence (the shape the merge preserves) is untested.  `D1` stays
**MEASURING**; the decision is the maintainer's, and the register records what a proper session must add.

- status: **DONE (`D1` open)** · sha: `5654a5f95`

## R11 — block 10: slab, fusion dispatcher, async input ring · **DONE 2026-11-08 (unblocked: 3 re-base defects found, all delivery-code losses)**

* block sha `7342cd95a` · 84 files · +15813 / −5461 · **expected conflicts: 8** — found 8 files / 22 hunks
* **result: `ed1d36436`** (tree `d8fb72ad0`), build green, coherence green on the dense 4B, on
  Qwen3-Coder-30B-A3B and on the 3-GPU `-sm tensor` 35B-A3B, `gate-norm-rows-alias.sh` **PASS**,
  `test-backend-ops -o MUL_MAT_ID` **931/931**, and every fusion arm byte-identical to the unfused chain.
* history: the parked two-copy resolution (`r11-resolution.patch`, 83 files, +16987/−5484) built green but
  **GPU page-faulted at `-c 8192`** on a dense 4B.  The fault and the three defects behind it are below.

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | TU split: our slab/arena/try-fuse registration vs upstream's new files (`moe-weighted-reduction*`, `topk-moe*`, ...) — resolve additively |
| `ggml/src/ggml-cuda/mmid.cu`, `mmq.cuh`, `mmvq.cu` | the fusion dispatcher's hooks |
| `src/llama-context.cpp`, `src/llama-context.h` | #31733/#31141 refactors; the async input ring's call sites |
| `src/llama-ext.h` | delivery-only header, touched by upstream? (it is ours) — resolve to keep ours |
| `tests/test-backend-ops.cpp` | the `--fusion-gate` oracle's 33-case matrix vs upstream's op test refactors (#31344, #30880) |

`D9`: #30662 (graph eviction) and #31584 (view bounds by strides) share our graph-cache-key/alloc-dep
territory — the delivery's replay guard must be re-checked, not merged blind.

**Exit:** build + coherence + `gate-per-block-build` is R17; here run `scripts/gate-norm-rows-alias.sh`
and the fusion-gate oracle (`tests/test-backend-ops --fusion-gate`, if the flag survives the test refactor).

### R11 conflict map and the resolutions already worked out (in `/tmp/r11-resolution.patch`)

| site | resolution |
|---|---|
| `mmid.cu` (9 hunks) | **ours** (the branch): every hunk is block 10's pre-re-base `expert_bounds`-only helper against the re-based `tile_spec` + optional-bounds helper from R01/R09. |
| `ggml-cuda.cu` (3) | upstream's batched `set_tensor_n`/`get_tensor_n` **plus** the delivery's `ggml_cuda_peer_copy_staged` declaration; `ggml_cuda_should_fuse_mul_mat_q` gains upstream's NVFP4 scale params and stays non-static; **the dispatcher**: block 10 moves it to `ggml-cuda-try-fuse.cu`, upstream added arms to the inline copy — the interim answer is to keep both, delivery first and upstream's as a fallback (see the merge debt below). |
| `mmq.cuh` (2) | `#include "mmid.cuh"` **and** `"moe-expert-cache.h"`; upstream's `mmq_fusion_is_efficient(prec_requested, …)` plus the delivery's three `*_try_fuse` declarations. |
| `mmvq.cu` (2) | the delivery's two-arg `mmvq_mmid_max_batch_band(cap, floor)` clamp with upstream's `RDNA4_M` family folded in (RDNA3_5 keeps its own floor, RDNA4_M follows upstream). |
| `llama-ext.h`, `llama-context.h` | **both APIs**: upstream grew its own device-layer-input feature (`llama_set_embeddings_layer_inp_dev`, `llama_supports_embd_row`, `llama_batch_ext_add_embd_row`) next to the delivery's `llama_lf_*` F1 surface.  **This is a `D15` decision point for R13** — upstream's `embd_row` + `#31506` device-copy DFlash may subsume the F1 draft-feature path. |
| `llama-context.cpp` (3) | both: the `GGML_FORCE_N_RS_SEQ` diagnostic plus upstream's `!hparams.causal_attn` clamp condition; the F1 `lf_dev_rows` inverse map plus upstream's `embd_row` event recording; the early return for the device layer-input buffer. |
| `tests/test-backend-ops.cpp` (2) | both: the extra rms_norm widths, and `set_rows` BF16; the `{512,22}` top-k MoE case plus upstream's `{384,…}` ones. |
| clean-applied sites that needed the merged signatures | the FA chunked-ring launch was missing `kv_rows`; `topk-moe.cu`'s 384-expert arm was missing upstream's five expert-scale/ids arguments; `model.hparams.no_alloc` → `model.no_alloc()` (13 sites). |

**Merge debt created by the dispatcher split (now `TODO #290`):** block 10 moves the fusion dispatcher into
`ggml-cuda-try-fuse.cu`, but upstream added its own arms to the inline copy in the same range — the MoE
weighted-reduction fold, the norm+MUL→mmvq Q8_1 fold, the GDN/SSM state-scatter caches, the SSM
conv+l2+gate/beta pre-scan and the DSV4 HC post op (7 arms).  The interim merge keeps upstream's inline copy as
`ggml_cuda_try_fuse_legacy()` behind the delivery's dispatcher (delivery first, fallback second), so no arm is
lost and none is displaced; the debt is to fold the 7 arms into the single dispatcher in a reconciled priority
order and delete the duplicate.

### The blocking regression (bisected to this rung)

* **Repro:** `llama-cli -m Qwen3.5-4B-Q8_0.gguf -c 8192 -ngl 99 -f prompts/prose-rdna-boosts.txt -n 16
  --seed 42 --temp 0 --single-turn` → `Aborted` (exit 134), `Memory access fault by GPU node-1 … Page not
  present`, 11 page-fault lines in the kernel journal.
* **R10's tree at the same command: exit 0, zero faults.**  R10 also runs `-c 8192` fine on the *VMM* build
  (the D1 arms), so `GGML_HIP_NO_VMM=OFF` is not the trigger on its own.
* `gate-norm-rows-alias.sh` fails the same way on its model arms (both `fused` and `chain`), which is the first
  time that gate could run at all — the fusion dispatcher arrives with this block.
* **Eliminated by env bisect:** the slab (`GGML_CUDA_SLAB=0`), the expert arena (`MOE_EXPERT_CACHE_MIB=0`), the
  FA prefill staging arena (`GGML_CUDA_FA_STAGE_MAX_MB=1`) and the interim legacy-dispatcher fallback (rebuilt
  with it disabled — still faults).  The chunked FA ring is opt-in and off.
* **Remaining suspects, in the order to try:** the async graph-input ring (`GGML_SCHED_STAGE_SLOTS`,
  `GGML_SCHED_STAGE_*`), the `ggml-alloc`/`ggml-backend` staging changes in the same block, the peer-copy staged
  path, the CUDA-graph key/eviction changes in `common.cuh`, and the mmid/mmq/topk-moe launch adaptations above.
  The fastest discriminator is `GGML_SCHED_STAGE_SLOTS=0`-class switches first, then splitting block 10's own
  patch into its subsystems for a proper `git bisect`.

### R11 bisect: what the fault was, and the three defects it exposed

**The repro.**  `llama-cli -m Qwen3.5-4B-Q8_0.gguf -c 8192 -ngl 99 -f prompts/prose-rdna-boosts.txt -n 16
--seed 42 --temp 0 --single-turn` → `Aborted` (exit 134), `Memory access fault … Page not present`, faulting
kernel `mul_mat_q<Q8_0,128,…>` grid `[16384, 576, 1]`.  At `-lv 4` the death is at the FIRST prefill ubatch
(`chunked GDN prefill n=512`; `task.n_tokens = 5256`), i.e. prefill, not model load.

**Env bisect** (one run each): `GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_ASYNC_INPUT=0`, the async-batch form,
`GGML_CUDA_DISABLE_NORM_Q8_1=1`, `GGML_CUDA_DISABLE_NORM_ROWS=1`, `GGML_CUDA_DISABLE_FUSED_ADDMUL=1`,
`GGML_CUDA_FUSE_SWIGLU_MMQ=0` — all still fault.  `GGML_CUDA_DISABLE_FUSION=1` → **clean**, and
`GGML_PAIR_OFF=1` → **clean**: the fault is `ggml_cuda_mul_mat_q_pair`.  (The previous session's "legacy
dispatcher disabled → still faults" test was consistent with this: it removed upstream's arms, not the pair.)

**Defect 1 — a renamed field with changed meaning (`ntj`).**  A temporary launch dump showed the two pair
launches as `nrows_x=9216 ncols_dst=512 ntj=512 nti=72` while every correct single-node launch is `ntj=4`.
Upstream renamed `mmq_args.ncols_max` (a column COUNT the launcher divided by J itself) to `ntj` (the J-tile
count, consumed directly as `grid.x`); the re-base carried the old value over, so the dense gate+up pair
launched `grid.x = ncols` instead of `ncols/J` and enumerated tiles past `src1_q8_1`/`dst` → page fault on
unmapped VA.  The routed arm had two more pre-`#31047` conventions: `tile_spec` was passed as `nullptr`
(upstream's kernel reads `tile_spec[jt]` whenever `ids_dst` is set) and `nchannels_y = ne02` (upstream's ids
grid puts the expert in `tile_spec`, so `nchannels_y = 1` and the experts live in `nchannels_x`).  Fixed by
deriving both arms exactly like the single-node path, including a per-arm `tile_spec` (the compaction is
J-independent, so the shared quantize is still one launch).

**Defect 2 — the delivery's routed-compact MoE MMQ had been dead since R09, and was wrong behind that.**
Tracing the launches showed `compact=0` on ALL 1551 routed launches of the 30B-A3B: the compact gate keyed on
`args.nchannels_y`, which upstream's ids convention sets to 1.  Making it reachable exposed the second half:
`mul_mat_q_process_tile`'s parameter was renamed `tile_y_max_j` (inclusive max) → `tile_y_j_sup` (exclusive
count) and upstream's plain path now passes `col_high - col_low`, while the compact kernel still passed
`col_diff - jt*J - 1` — dropping the last column of every tile (measured: garbage text,
`betwee 140000000000`).  The two bugs had been hiding each other (unreachable + wrong = invisible).  Fixed:
gate on `nchannels_x` (+ require `expert_bounds`), pass `tile_y_j_sup = col_diff - jt*J`.  Result: compact ==
plain == pair-off, byte-identical `013d09f165a5`, and **+1.9 % prefill** (3489 vs 3423 t/s, 5256-token prompt,
2×2 reps).

**Defect 3 — the "upstream arms" fallback was not upstream's arms (TODO #290, now closed).**  The parked
resolution kept upstream's inline dispatcher as `ggml_cuda_try_fuse_legacy()`.  Auditing it against upstream
`a691b2345` showed it was ~490 lines BIGGER than upstream's `ggml_cuda_try_fuse`: upstream's block merged with
**stale duplicates of delivery arms** (`ssm_conv_in`, `op_ssm_conv_l2_gatebeta`, `op_rms_norm_q8_1`,
`add_softplus_mul`, `shexp_down_gate`, `l2_norm_pair`, `gate_beta_verify`).  One of them, the inline SSM
conv-input fold, fires on qwen35moe decode (6× per GDN layer) and produced text that **diverges from the
unfused chain** (`b97e5d155ea3` vs `90decf558d96`) in a state-carrying path.  Resolution: reconciled into ONE
dispatcher — every arm audited against upstream master; all but the DSV4 hyper-connection post are already
owned by the delivery's dispatcher (or its newer successor: `match_shared_expert_try_fuse` IS upstream's arm,
`moe_weighted_reduction_try_fuse` is its superset, `gdn_cache_cpy`/`ssm_scan_cache_cpy` wrap upstream's), so
the DSV4 HC post arm is folded in at upstream's position and `ggml_cuda_try_fuse_legacy()` is deleted
together with the superseded `ggml_cuda_match_moe_weighted_reduction` matcher + struct + constexpr (143 lines)
that only it used.  The interim `GGML_CUDA_FUSE_LEGACY` switch is gone with the duplicate.

**Measurement notes (both cost time here; both now recorded).**
* Fusion decisions are **not deterministic run-to-run** on this tree: identical runs of the same binary differ
  by a few fusions (4B: `CONCAT skip=1` 334/336, `CONT skip=2` 40/37) because several arms are
  address-selected (`ggml_cuda_check_fusion_memory_ranges`).  A per-node fusion trace is therefore NOT a valid
  A/B oracle; the text is.
* `Generation: t/s` on a short `-n 16` run is dominated by one-off setup.  It made the block-01 hybrid
  all-reduce look 29 % SLOWER than rccl (58.8 vs 82.7 t/s); at `-n 64` the same arms read **100.2/105.6
  (hybrid) vs 90.1/90.0 (rccl)**, i.e. hybrid +11-17 % as designed.  Never compare all-reduce arms on a short
  decode run.

- status: **DONE** · sha: `ed1d36436` · tree `d8fb72ad0`

## R12 — block 11: qwen4exp · **DONE 2026-11-08 (`D3` DECIDED: keep the delivery's implementation, measured)**

* block sha `0e1032213` · 46 files · +12091 / −433 · **expected conflicts: 7** — found 9 files / 34 hunks
  (the cascades were `llama-memory-recurrent.cpp` and `tests/CMakeLists.txt`)
* **result: `b1b0fa807`** (tree `7baf3ba40`), build green; dense 4B `1472ebdadd46` and
  Qwen3-Coder-30B-A3B `013d09f165a5` unchanged; `gate-hc-fusions.sh` **PASS** (5/5 hard arms
  byte-identical, 449 chars `d62340fbf566`); `gate-qwen4exp-quant-coherence.sh` **PASS**
  (IQ3_XXS / IQ4_NL / Q4_K_M gather on == off, no `////`)
* **the headline: upstream already HAS a qwen4exp** (see `D3` below), so this rung was not a
  port but a choice between two implementations of the same architecture

| conflicted path | what to expect |
|---|---|
| `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h` | #29510/#31683/#31073 plus our indexer-cache residency |
| `src/llama-memory-hybrid-idx.cpp` | delivery-only memory type (the indexer cache) vs the memory-instance refactors |
| `src/models/models.h` | #31142 output-struct refactor for every model |
| `src/models/qwen4exp.cpp` | **`D3`** — #31443 (conv state tail), #31243 (k-pool sparse selection), #31009 (QSA without kq_mask) all rewrite build functions we own |
| `tests/test-backend-ops.cpp`, `tests/test-llama-archs.cpp` | additive/refactor (#30880) |

**`D3` is this rung's decision:** #31243/#31009 build the k-pool sparse selection independently of the kq
mask — the same problem our derived mask + pooled sparse decode solved.  Measure on the rebased tree:
qwen4exp prefill/decode with our fusions on/off, and the sparse arm's selection vs upstream's, before
deciding to keep both, narrow ours, or drop one.

**Exit:** build + coherence + `scripts/gate-hc-fusions.sh <model.gguf>` (4 arms) +
`scripts/gate-qwen4exp-quant-coherence.sh`.

### R12 resolution log

**The rung is a fork decision, not a merge.**  `src/models/qwen4exp.cpp` already exists upstream
(1514 lines, upstream contributors: `#30965` "build the QSA selection without the kq_mask",
`#31243` "k-pool sparse selection independently of the kq_mask shape", `#31443`, `#30927`,
`#31086`), with its own HC mix/combine (plain ops), QSA selection (`build_qsa_sel`, an additive
mask) and indexer (plain ops).  The delivery's is 3197 lines: fused HC ops
(`GGML_OP_HC_MIX`/`HC_COMBINE`, HC16 planar layout), the mask-free QSA store+top-k selection,
`INDEXER_SCORE/TOPK/FILL`, a keys-only indexer cache, the MTP sparse draft and the managed lazy
reader.  Per the maintainer's ruling the decision was **measured first** (`C`: resolve far enough
to run both, then decide), not taken on the code's shape.

**`D3` measurement** (same GGUFs, same build path, gfx1201, ROCm 10.1, VMM build, `-c 8192`,
5256-token `prompts/prose-rdna-boosts.txt`, `-n 16 --seed 42 --temp 0`):

| config | implementation | prefill | decode | text |
|---|---|---|---|---|
| Flash-Next GSQ-IQ3_XXS, 3 GPU `-sm tensor` | upstream | 1399.7 t/s | 53.5 t/s | `7a8734a9f1a6` |
| Flash-Next GSQ-IQ3_XXS, 3 GPU `-sm tensor` | **delivery** | **1432.1 t/s** | **70.1 t/s** | `7a8734a9f1a6` (identical) |
| Qwen3.8-27B UD-Q4_K_M, 1 GPU | upstream | 1311.1 t/s | 32.4 t/s | `b905c9763a58` |
| Qwen3.8-27B UD-Q4_K_M, 1 GPU | **delivery** | 1303.2 t/s | 32.4 t/s | `b21e94a5b6e1` (diverges) |

On the config this repo validates the delivery's implementation is **bit-identical to upstream's
and +31 % decode / +2.3 % prefill**; on the 27B it is throughput-neutral and its greedy text
diverges at 16 tokens, which is a *chain* difference the R17 KLD gate has to quantify, not a
purity claim in either direction.  **Decision: keep the delivery's implementation** (`D3` →
DECIDED).  Rationale beyond the numbers: r235's qwen4exp validation and both delivery gates stay
meaningful, and the fused HC/indexer kernels are what the +31 % *is*.  Cost accepted: we own a
3200-line model file that upstream keeps rewriting -- the price of the 31 % on this config, to be
re-examined if the conflict cost per rung keeps growing (`GREEDY-PURITY.md` §84's doctrine).

| site | resolution |
|---|---|
| `llama-kv-cache.h/.cpp` | **both indexer flags**: upstream's `is_indexer` (indexer RoPE params) and the delivery's `v_enabled` (keys-only caches).  The delivery's `n_pad` ctor parameter *and* member are dropped -- `#31200` removed them and hardcodes the 256-cell floor in the size math -- so only the derived `n_pad_prefill` (`LLAMA_KV_N_PAD_PREFILL`, `_MIN`, `_FROM`) survives.  `get_n_kv()` now *requires* `n_tokens` (a default of 0 would silently turn the prefill pad off at that call site) and `llama-kv-cache-dsv4.cpp` passes it. |
| `llama-memory-hybrid-idx.cpp` | the keys-only indexer cache is built with upstream's post-`#31141` ctor (`no_alloc`, no `n_pad`) plus `is_indexer`/`v_enabled`; the pool body is the delivery's. |
| `llama-memory-recurrent.cpp` | R04 wins (upstream's `rs_max`, the tracker stays dropped); the block's two real additions are kept -- the getenv-once `rs_rollback_debug()` helper (issue #273) and the `has_recr` gate that stops a qwen4exp MTP context (recurrent filter matches nothing) spamming the non-consecutive-position warning. |
| `models.h`, `qwen4exp.cpp` | `#31733` made `model` a member of `llm_graph_context`, so both ctors lose their `const llama_model &` argument and the class its shadowing member; `#31073` turned the rotation constants into per-layer vectors (`self_k_rot[il]`); `#29510`/`#31683` replaced `get_kq_mask()` with `self_attn_mask.raw`; `build_attn_mha` grew `n_kv_max`. |
| `ggml-cuda/indexer-score.cu` | `ggml_rope_yarn_corr_dims` gained the truncate flag (2 sites). |
| `tests/CMakeLists.txt` | R03 had already re-placed every registration this block adds (`#30880`), so only `test-lazy-reader` is new. |
| `tests/test-llama-archs.cpp` | upstream's `get_model`/`get_model_and_ctx` split kept; the block's `lazy_buf_size` is handed over as `LLAMA_LAZY_BUF_MB` around the load, and the qwen4exp roundtrip passes `256 << 10`. |
| `tests/test-backend-ops.cpp` | **both** sides' Flash-Attention classes (upstream's `test_flash_attn_ext_kv_rows`, the delivery's `test_flash_attn_qsa`) and both perf-case blocks. |

**Two gate findings (both fixed/recorded here, not deferred):**
1. `gate-qwen4exp-quant-coherence.sh` picked the **DFlash2 drafter** as the model under test (its
   `ls | grep -v -e mmproj -e mtp | head -1` predates the DFlash asset landing in the quant
   directory), so every arm failed with "dflash requires ctx_other to be set".  Fixed: the
   selection now excludes `mtp-*`/`DFlash`/`dflash`.  Re-run: PASS on all three quants.
2. `gate-hc-fusions.sh`'s two REPORTED `dsv4_pre`/`dsv4_post` arms no longer agree with the chain
   on the BF16-hc class: chain `d62340fbf566`, dsv4 arms `6a102dbc62d6` -- which is the value the
   *chain* produced at r235.  So the HC chain itself moved with the re-base while the fused
   pre/post ops still reproduce the old numerics; the gate reports (does not fail) them, but its
   own header says byte-identity is expected on this class.  Recorded as `TODO` #292 for the R17
   KLD gate (and a decision on the soft-arm expectation).

- status: **DONE (`D3` DECIDED)** · sha: `b1b0fa807`

## R13 — block 12: campaign remainder (CPU wins, iface fills, DFlash, server) · **DONE 2026-11-09 (D15 DECIDED: F1 deleted, upstream's `embd_row` kept; host-experts guard dropped after measurement)**

* block sha `be4749a56` · 81 files · +5893 / −995 · **expected conflicts: 8** — found **8 files / 24 hunks**
* **parked, not blocked**: the cherry-pick was surveyed and rolled back (the branch is at the green R12 tip
  `b1b0fa807`).  The rung contains **three** places where upstream and the delivery built the same thing, so it
  needs decisions with measurements rather than a long merge: `D15` (F1 `llama_lf_*` vs upstream's
  `embd_row`/`#31506`), the state save/restore batching (delivery `state-pipe` vs upstream's new batched
  `llama_io_write_host`), and the server prompt-cache residency/parking vs upstream's server rework.
  `wip/upstream-rebase-261108/r13-survey.md` is the cold-start brief: per-file conflict table, the three
  decisions with the expected resolution and the gate that decides each, and the amended exit criteria
  (the survey also records that `ggml-backend.cpp`'s conflict is mechanical: free BOTH event families).

| conflicted path | what to expect |
|---|---|
| `common/speculative.cpp` | `D11` again (DFlash ingest realign) |
| `ggml/src/ggml-backend.cpp` | iface fills (`ALL_GATHER` etc.) vs upstream's copy API (#30881) |
| `src/llama-context.cpp` | `D8`/`D9` |
| `src/llama-kv-cache.cpp` | parking / residency hooks |
| `src/models/dflash.cpp`, `src/models/gemma4.cpp` | #31506 (device copies for hidden features) vs our r163 in-graph layer-input store |
| `tools/server/server-context.cpp`, `server-task.cpp` | upstream's server refactors (#30779, reference-counted checkpoints #30137) vs our prompt-cache residency + idle-slot parking |

This is the **refactor-heaviest** rung: upstream's `#31733` (`model` as a member of `llm_graph_context`)
touches 33 of this block's 81 files.  Expect mostly mechanical re-derivation.

`D8`: which of our CPU wins survive `#31854` (F32 src1), `#31179` (concat rewrite), `#31276`?

**Exit:** build + coherence + `scripts/gate-park-idle-slots.sh` + `scripts/gate-ckpt-reuse.sh` +
`scripts/gate-state-pipe.sh` (whichever of the three still has its code after the merge), **plus a DFlash2 draft
attach smoke**: at R12, `llama-cli -m <Flash-Next> -md <the DFlash2 drafter> --spec-draft-n-max 3` aborts with
`dflash requires ctx_other to be set` (llama-context.cpp), i.e. the drafter cannot be attached at all on the
pre-block-12 tree.  That is *not* a regression this campaign introduced -- the DFlash2 draft path (the ingest
tail, the KV/noise reserve, the in-graph layer-input store, the KV realign; block 12's r106/r130/r162/r163/r218)
is exactly what this rung carries -- but R13's exit must show the drafter attach and generate, because it is
the one thing the gates above do not cover.

**Result (`d84c5d74c` -> re-cut as `2b09a50b0` with the series identity, tree `87e2841f7`):** build green, no
conflict markers, warning-clean in the touched files.

* **`D15` — F1 is subsumed, deleted; upstream's `embd_row` surface kept.**  Removed tree-wide: the 5
  `llama_lf_*` API entries, `cparams.lf_layer_inp_dev` / `lf_dev_inject` / `lf_layer_inp_dev_store[]` /
  `lf_row_offset`, `llm_graph_result::set_layer_inp` / `lf_emit_layer_inp_store` / `lf_store_armed` /
  `lf_pending_il` / `t_lf_rows` / `get_lf_rows`, the `dflash.cpp` helper (`llm_graph_input_lf_rows`,
  `lf_dflash_target_features`), the `speculative.cpp` `lf_dev` / `batch_inject_tok` injection path, and the
  **31 arch-graph call sites** back to `res->t_layer_inp[il] = ...` (those files end up identical to
  upstream).  One thing carried, as predicted: **the Meta refusal** — upstream allocates the layer-input
  buffer on `model.dev_output()`, which under `-sm tensor` is the Meta device, and the draft's
  `ggml_get_rows` over it GPU-faults (`k_get_rows_float_vec`, 3x gfx1201, first injection).
  `layer_inp_dev_alloc()` now keeps the host-copies path there.
* **The F1 host-experts guard was re-measured and DROPPED.**  Condition: `n_seq > 1` + host-resident MoE
  experts (`-ncmoe`), where r163 had measured the armed slab breaking the draft's device reads.  Two-slot
  `llama-server` arms, `-sm layer`, 3 GPU, 5256-token prompt, greedy, guard bypassed (device buffer) vs guard
  active (host copies): Flash-Next GSQ + DFlash2 at `-ncmoe 24` **0.53793 / 0.51678 both ways** (78/145 and
  77/149, mean len 2.59/2.54, same wall time) and at `-ncmoe 10` **0.55245 / 0.47134 both ways**; 35B-A3B
  Q8_0 + its DFlash2 at `-ncmoe 10` **0.63918 / 0.62887 both ways**.  Identical to the accepted/generated
  count.  The guard's premise does not reproduce on the re-based slab, so only the Meta refusal remains, and
  `LLAMA_DFLASH_DEV=0` stays as the lever.  (Recorded in `../../INVARIANTS.md`.)
* **State save/restore: the delivery's staged/pipelined copy is NOT subsumed on ROCm.**  Upstream's
  `ggml_backend_tensor_get_n()` batches per buffer type, but `ggml_cuda_get_tensor_n()` returns false under
  `GGML_USE_HIP` (the pinned/batched path is CUDA/MUSA-only), so on a tensor split it degenerates to one
  pageable D2H copy per piece -- exactly the cost r96 removed.  The delivery's pinned `state_stage` +
  `LLAMA_STATE_PIPE_GROUPS` + threaded streaming copy is kept, re-derived on upstream's `winfos`/`base_ptr`
  shape, with `get_n`/`set_n` as the fallback; the delivery's per-tensor merge-gap code and
  `LLAMA_STATE_GET_MERGE_GAP` are gone.  **`gate-state-pipe.sh` PASS**: restore 43.7 ms vs 74.7 ms (sync),
  server save+load 19.0 vs 39.4 ms, `pipe1` -4.8 %, `copy0` -9.1 %, every arm 5/5 cycles byte-identical.
* **Server prompt cache: upstream's entry API + the delivery's policy.**  `find()` stays split out of
  `load(ctx_tgt, ctx_dft, id_slot)`.  Upstream's `shared_ptr` checkpoints forced one behavioural fix: a
  dropped checkpoint's buffers may only be recycled when uniquely owned, or the cached copy is emptied and a
  later restore "succeeds" from a state-less checkpoint and then fails `seq_rm` (the state-pipe gate's
  fresh-reference request aborted before the fix).  **`gate-park-idle-slots.sh` PASS** (6 parks / 2 unparks,
  park == lazyoff on A-D, parkoff still reproduces #210); **`gate-ckpt-reuse.sh`** hard checks PASS (3 kept,
  5 vs 8 checkpoints, A2-A4 33/32/32 vs 46/43/43 ms, texts and memory equal).
* **Two pre-existing R12 defects found and fixed (they blocked the measurements above):**
  (a) `hc_combine_norm`'s base-out pool copy was a plain local allocated before `bf16_src1_cache_put()`
  pushes its block -> LIFO violation -> `GGML_ASSERT` in `ggml_cuda_pool_vmm::free` on **every `-sm layer`
  run of a qwen4exp model** (reproduced identically on the R12 build; fixed by putting the copy on the pool
  stack); (b) the DFlash `graph<false>` and `graph_dsv4` injection blocks never registered their
  target-feature input (`res->add_input`), so the drafter was silently dead -- acceptance **0.000 -> 0.754**
  on 27B+DFlash2 1 GPU (mean len 3.26), 0.741 on 35B-A3B 3 GPU tensor, 0.405 on Flash-Next GSQ, device and
  host transports identical in every arm.
* **Coherence (same-session A/B against the R12 tree):** dense 4B `1de99b7e6ee6`, 30B-A3B `4b4d56191368`,
  35B-A3B 3 GPU tensor rccl `86b7f9b5aa80` -- all unchanged; Flash-Next GSQ `7a8734a9f1a6` (the only §4
  oracle that reproduces; the other three recorded hashes are pre-re-base and were re-recorded here).
* **DFlash2 drafter acceptance after the fix** (the R13 headline): 27B+DFlash2 1 GPU 0.754 (3.26),
  35B-A3B+DFlash2 3 GPU tensor 0.741 (3.15), Flash-Next GSQ+DFlash2 3 GPU tensor 0.405 (2.21) and
  `-sm layer` 0.384 (2.14), MTP control on the same Flash-Next arm 0.493 (2.42).
* `D8` (which CPU wins survive the CPU-backend rewrite): block 12's CPU hunks (`arch-fallback.h`,
  `arch/x86/quants.c`, `ggml-cpu.c`, `ggml-cpu.cpp`, `repack.cpp`) merged without conflict, and `#31854`'s
  F32-src1 `type_traits_cpu` entry did not collide with them, so nothing was dropped.  **No CPU measurement
  was taken at this rung** -- the `pp`/`tg` + `test-backend-ops` + `-lv 4` census arm stays owed (moved to
  `TODO.md` as a column for R17's release ladder).

- status: **DONE** · sha: `2b09a50b0` (tree `87e2841f7`) · session 2026-11-09

## R14 — block 13: MTP draft vocabulary subset · **DONE 2026-11-09**

* block sha `6ee8eb290` · 19 files · +569 / −6 · **expected conflicts: 3**

| conflicted path | what to expect |
|---|---|
| `src/llama-graph.h` | `D11`/`D9` — the graph context's new `model` member (#31733) |
| `src/llama-model.cpp`, `src/llama-model.h` | #31141/#31142 loader refactors; the vocab-subset state hangs off the model |

**Exit:** build + coherence + `--spec-draft-mtp-vocab N` runs and is inert when unset (default off).

**Result (`ed13a322e`, tree `c08c912a2`):** build green, no conflict markers, warning-clean.  4 files
conflicted (predicted 3; `common/common.cpp` extra — an arg-plumbing hunk):

* `common/common.cpp` — upstream replaced the inline fit block with `common_fit_apply()`, which already
  derives the draft's post-slab reserve (the delivery's block-09 fold, absorbed upstream).  Call site stays
  upstream's; block 13's one meaningful line (`cparams_dft.mtp_draft_vocab = params.speculative.draft.mtp_vocab`)
  moved into `common_fit_apply()`'s existing `if (spec_mtp)`.
* `src/llama-model.h` / `.cpp` — both sides add declarations at the same place; the `llama_rope_params` getters
  (base drift) and the new `llama_mtp_draft_vocab` struct / `mtp_draft_vocab_get` are kept side by side, and the
  shared `return res;` tail served **both** conflicted function bodies.
* `src/llama-graph.h` — `llm_graph_params::can_reuse` compares both sides' terms; the block's `arch ==
  other.arch` is dropped, because `arch` no longer exists on the params after `#31733` and `model ==
  other.model` already implies it.
* `src/llama-context.cpp` — `#31141` drift: `model.hparams.no_alloc` -> `model.no_alloc()` in the shared-NextN
  head probe.

**Functional check (both head paths):**
* shared-NextN head (Flash-Next GSQ + `mtp-...-shared-Q8_0`, 3 GPU `-sm tensor`), `--spec-draft-mtp-vocab 98304`:
  `mtp_draft_vocab = 98304: MTP draft head uses 98337 of 248320 rows (165.06 MiB of q5_K, Meta())` — built from
  the target's borrowed LM head through `cparams.ctx_other`, exercising this block's row-range splice over a
  vocabulary-split (Meta) tensor.  Acceptance identical to off (0.49333, 37/75, mean len 2.42); greedy text
  identical (`55a50dee05e0`) across unset / `-vocab 0` / `-vocab 98304`.
* self-headed (Gemma4-12B Q8_0 + its own MTP head, 1 GPU), cut 245760: 245766 of 262144 rows, acceptance and text
  identical to off (0.68852, 42/61, mean len 3.00; `f8ef9d420320`).  An aggressive cut (98304, which this
  tokenizer's distribution does not suit) drops acceptance to 0.31915 (30/94) and leaves the text **identical** —
  the promised property (the trim can only lose proposals, never change an accepted token).  Consistent with the
  block's own note that gemma4's useful cut is 245760 of 262144.
* coherence unchanged against R13 on four arms: 4B `1de99b7e6ee6`, 30B-A3B `4b4d56191368`, 35B-A3B 3 GPU tensor
  `86b7f9b5aa80`, Flash-Next GSQ `7a8734a9f1a6`.

- status: **DONE** · sha: `ed13a322e` (tree `c08c912a2`) · session 2026-11-09

## R15 — block 14: `-sm relaxed-tensor` (TEMPORARY) · **DONE 2026-11-09 (0 conflicts)**

* block sha `59a3e5206` · 11 files · +130 / −16 · **expected conflicts: 0**

Checked 2026-11-08: upstream's `#31147` only adds `LLM_ARCH_LAGUNA` to the **deny** list
(`llm_arch_supports_sm_tensor`), so the amax arches are still refused under `-sm tensor` and block 14 is not
subsumed.  Our block only changes the error text for `flint`/`mellum`/`mistral4` — if upstream grew a
relaxed mode in the meantime, re-check `src/llama-arch.cpp` and `llama_model_create()` before resolving.

**Exit:** build + coherence + `-sm relaxed-tensor` loads a flint/mellum/mistral4 MoE and matches `-sm layer`
text (the block's own claim).

**Result (`617ee7152`, tree `f6ce37555`):** **zero conflicts**, no drift edits — the survey's "0 expected" held
(the block's hunks sit in `common/arg.cpp`/`fit.cpp`, `ggml-backend-meta.cpp`, `include/llama.h`,
`src/llama-{context,impl,model,moe-cache}.cpp`, `src/llama.cpp`, `src/models/dflash.cpp`, none of which moved).

Exit verified on the real flint model (`/llm/models/RecursiveAI/Flint/Q8_0/flint-Q8_0.gguf`, 3 GPU):
* `-sm tensor` still refuses it with this block's exact message — upstream's `#31081` deny is intact, the block
  only adds the escape hatch: `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'flint'; this arch's MoE
  amax rescale needs '-sm relaxed-tensor'`.
* `-sm relaxed-tensor` loads and generates, and the greedy text is **identical to `-sm layer`**
  (`904c62d1d4f9`, 169 chars) — the block's own claim, on the arch it was validated for.  `mellum`/`mistral4`
  stay unvalidated here (no model — as the block itself says).
* The mode is a no-op for a non-amax arch: 35B-A3B Q8_0 3 GPU `-sm relaxed-tensor` == `-sm tensor`
  (`86b7f9b5aa80`).
* Coherence unchanged: 4B `1de99b7e6ee6`, 30B-A3B `4b4d56191368`, Flash-Next GSQ `7a8734a9f1a6` (the
  `ggml-backend-meta` / model-graph hunks touch the tensor-split path, so that arm matters).

- status: **DONE** · sha: `617ee7152` (tree `f6ce37555`) · session 2026-11-09

## R16 — block 15: the streamed KV (window + attendance pool) · **DONE 2026-11-08**

* block sha `69d36099b` · 33 files on this base · +6263 / −835 (the brief's +5294/−56 is stale: the block
  *moves* a large function block in `src/llama-model-loader.cpp`, 750/750, getenv latching + host-master buft)
  · **expected conflicts: 9**

| conflicted path | what to expect |
|---|---|
| `common/fit.cpp` | the window's `--fit` term vs upstream's fit changes (#31569 NextN, #21231 router margin) |
| `ggml/src/ggml-backend-meta.cpp` | per-child region fan-out vs #31447 |
| `src/llama-context.cpp` | the memory-instance construction refactors (#31141 `no_alloc`, #31200 `n_pad` removal) |
| `src/llama-ext.h` | delivery-only declarations |
| `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h` | **`D13`** — #31141/#31200/#30541/#31073/#31570 all land in the cache; our region provider binds at the buffer type |
| `src/llama-memory.h` | the memory interface after #31200 |
| `src/llama-model.cpp` | loader/split-state (#31141) |
| `src/models/qwen4exp.cpp` | `D3` residue |

`D12`: #30541 (on-device state copy sizes for quantized KV) and #31073 (rotation constants on GPU) change the
state/rotary read shape the streaming provider must keep addressing.

**Exit:** build + coherence + `scripts/gate-kv-residency.sh` (window and pool arms, plus the multi-device
`-sm layer` section) + `scripts/gate-kv-window-split.sh`.

**Result (`2dc4511b8`, tree `7d6e24c12`):** build green (0 errors; the only `-Wshadow` warnings in the touched
files are upstream's own `n_stream`/`v_trans`/`ml` shadows).  **8 files / 12 hunks** conflicted, not 9:
`src/models/qwen4exp.cpp` auto-merged (`D3` residue did not conflict; its 59/−7 lines all landed).  A
distinctive-added-line check over the 25 auto-merged files found no dropped delivery code.

| conflicted path | resolution |
|---|---|
| `common/common.cpp` | **decision (maintainer, option 1):** upstream's `common_fit_apply()` call kept; block 15's `mtp_shared` reserve branch folded into `common_draft_reserve_derive()` (the one function `--fit` *and* the MoE preflight call) |
| `common/fit.cpp` (3) | upstream's `common_fit_log_scope + the delivery's three kv out-params (`kv_bytes_out`/`kv_pool_vram_out`/`kv_pool_ram_out`) at the definition and at both call sites, each also taking upstream's new `hp_nsk` |
| `ggml/src/ggml-backend-meta.cpp` | both sides: upstream's `ggml_backend_meta_inner_extent`/`_view_slab_dim` (R14's splice) + the delivery's `ggml_backend_meta_kv_region_grow` |
| `src/llama-context.cpp` (2) | (a) both designated-init entries (`no_alloc = model.no_alloc()`, `kv_pool = cparams.ctx_other == nullptr`); (b) upstream's readback-sync hunk — the delivery side was stale F1 code (`layer_inp_host_fill`, deleted at R13) |
| `src/llama-ext.h` | upstream's `llama_model_classifier_free_guidance` + the whole delivery declaration block |
| `src/llama-kv-cache.cpp` (2) | (a) ctor tail `is_indexer, v_enabled, kv_pool_capable`; (b) `memory_breakdown()` **re-derived** over `ctxs_bufs.bufs` + `ctxs_bufs.no_alloc` + upstream's `attn_rot_bufs` sum (upstream moved the loop into `llama_memory_bufs`) |
| `src/llama-memory.h` | `llama_memory_params` keeps upstream's `no_alloc` and adds the delivery's `kv_pool`, ahead of upstream's new `struct llama_memory_bufs` |
| `src/llama-model.cpp` (2) | (a) block 15's two free functions above `create_memory`, and the parameter is upstream's `params` again (the delivery-side `params_l` rename is gone — the body was upstream's everywhere but two lines); (b) the MTP-window `llama_kv_cache` construction drops the `n_pad` argument (`#31200`) |

**`D12` — quantized-KV state and rotation: no re-derivation needed beyond the buffer type.**  The provider still
binds at the **buffer type**: `#31073`'s rotation constants are a per-cache, per-layer buffer (`attn_rot_bufs`),
not cell-indexed, so they are outside the region (they are summed into `memory_breakdown()` and nothing else);
the state copy keeps going through the region buffer's own copy.  Proven by the gates, whose spilling arms
really spill (16 mapping flips / 64 MiB host-backed) and whose `win2mtp` arm takes the region path for the
batched state save.

**`D13` — memory instances: mechanical, but three drifts and two real defects.**
* `hparams.no_alloc` → the ctor's `no_alloc` at all three block-15 sites (`#31141` moved it out of the hparams).
* the pool's cell count rounded to the cache's `n_pad`, which no longer exists (`#31200`): it now rounds to whole
  **pages** (the pool is addressed in pages of `compress_ratio` cells).
* `memory_breakdown()` as above; the pool's mirror→host-bucket and the window's resident/spill split accounting
  are unchanged.
* **`src/llama-memory-hybrid.cpp` — silent ctor argument shift.**  Upstream inserted `is_indexer` **before**
  `v_enabled`; the delivery's positional call passed the old tail, so `is_indexer = true` and
  `v_enabled = kv_pool` (= false) and the attention cache allocated **no V tensor** →
  `llama_kv_cache::cpy_v` null-deref at `+0x20` on the first graph build.  **Every hybrid (GDN) model
  segfaulted at load** — the dense 4B coherence arm never generated (SIGSEGV, exit 139, in the fit probe's
  `sched_reserve`).  Fixed by naming `/* is_indexer */ false`; the other call sites were audited (`hybrid-idx`
  already named it, `model.cpp` stops at `share`, `dsa`/`dsa-iswa`/`msa`/`dsv4` are upstream's and correct).
* **`ggml/src/ggml-cuda/indexer-score.cu` — latent R12 defect, NOT R16 drift** (`b1b0fa807`): the fused
  `GGML_OP_INDEXER_SCORE` node called `ggml_rope_get_truncate(dst)`, and upstream's `#30206` added
  `GGML_ASSERT(a->op == GGML_OP_ROPE || ROPE_BACK)`.  So **every sparse QSA decode run has aborted on this tree
  since R12** — the coherence arms and both qwen4exp gates run the dense arm, and only block 15's residency gate
  reaches the sparse one: its `pref` arm processed the whole 25 763-token prompt at 383.2 t/s and 63 tokens at
  51.3 t/s, then aborted at `ggml.c:4648`.  Both call sites now pass `/*truncate*/ true` — the graph's own rope
  node default (`ggml_rope_impl` initialises `no_trunc` 0; nothing calls `ggml_rope_set_truncate` on the
  indexer-key node) and the pre-`#30206` 5-argument helper's unconditional floor/ceil, i.e. pre-re-base numerics.
  Recorded as `GREEDY-PURITY.md` §85 and `TODO` #297 (that arm needs a gate).

**The r193 draft reserve is preserved** (r236 absorbed the fit plumbing, so the block's inline derivation was the
casualty): measured on 35B-A3B Q8_0, 1 GPU, `-ncmoe 48 -c 262144 --spec-type draft-mtp`, `-lv 4` — window `auto`
prints `slab draft reserve derived from '': draft CONTEXT 1102 MiB (kv 142 + compute 768 + slack 192)` → arena
**20 527.5 MiB / 62.9 %**; window `0` prints `keeping the flat 3584 MiB draft reserve` → arena 17 977.5 MiB /
55.1 %.  **+2 550 MiB of arena, +7.8 points of residency**, text unaffected.

**Coherence (this build family):** dense 4B `1de99b7e6ee6`, 30B-A3B `4b4d56191368`, 35B-A3B 3 GPU tensor rccl
`86b7f9b5aa80`, Flash-Next GSQ `7a8734a9f1a6` — all four identical to R13/R14/R15.

**Gates:**
* `gate-kv-residency.sh` **PASS**: window `wref == wres == wspill` `be904e86f248` (wres 0 flips / 0 MiB
  host-backed, wspill 16 flips / 64 MiB, every slide on the map grid); pool `pref == pres == pfloor`
  `f260d1824ab3`, `bad 0`, 8192 pages (408.0 MiB) on `pres` vs 4104 (204.4 MiB) on `pfloor`, 252 resolves both,
  claims 137.7 → 217.1 per resolve, hits 91.02 → 89.63 %; multi-device (`-sm layer` 0,1)
  `pmref == pmres == pmfloor` `f260d1824ab3`, one mirror/pool/counter block per device, **2 counter blocks
  summed**, 117.8 → 268.1 claims per resolve.
* `gate-kv-window-split.sh` **PASS**, 38/38 — one region per device on a 3-device 4B, a 27B at `-c 524288`
  sliding on **both** devices (32 flips each), `-c 1048576` windowed (the resident arm does not fit), 1-device
  unchanged, `-nkvo` still declines, all five fit arms, and `win2mtp == ref2mtp` with acceptance `0.46154`
  identical and the batched state save on the region path.
* `gate-spec-draft-window.sh` **PASS** — `LLAMA_SPEC_DRAFT_WINDOW=1024` → `1024 cells per sequence, cache 3072
  cells ((win + 512) x 2)`, all six requests across both slots byte-identical to win 0, acceptance identical.
* `check-gpu-faults.sh --since "2026-11-08 10:47:00"` clean (the journal's pre-session faults from the R15
  session were outside every gate's window, `TODO` #294).

**R16 follow-up (`328a155a9`, tree `a101f0d66`) — cross-platform validation found a real bug.**  Validating the
rung on the maintainer's RDNA3 (`fingon`, gfx1100) and RDNA3.5 (`halo`, gfx1151) boxes exposed the missing half of
the umem guard: on an integrated GPU "host" and "device" are ONE pool, and the tier that serves a host-expert
master is denied there — but the *placement* was not, so `-ncmoe N` put expert banks in a `ROCm_Host` buffer
nothing could stage and the consumer read a page the GPU cannot address (Qwen3.8-Flash-Next GSQ-IQ3_XXS, gfx1151,
`-ncmoe 48`/`24`: `mul_mat_q<Q2_0,16>` at a host VA, `HSA_STATUS_ERROR_MEMORY_FAULT`; with a drafter/KV live the
same arm died in the VMM pool's free-order assert instead).  **The pre-re-base r235 binary faults identically**, so
it is not a re-base regression.  `llama_model_loader::set_host_experts_umem()` now refuses every host-expert
placement on any `GGML_BACKEND_DEVICE_TYPE_IGPU` device (one WARN; `-ncmoe`/`-cmoe`/`-lme`/a CPU expert
`--override-tensor` are ignored and every weight stays on its layer's device).  Verified: halo Flash-Next
`-ncmoe 48`/`24` and 35B-A3B `-ncmoe 48` now exit 0 (no fault, no assert), the dense 4B arm unchanged
`1de99b7e6ee6`; on gfx1201 the guard is inert (4B `1de99b7e6ee6`, `35B-A3B -ncmoe 20` still host-pinned 16 320
MiB); on gfx1100 Flash-Next `-ncmoe 48` runs clean `7a8734a9f1a6`.

**Cross-platform results on the rung itself (all as expected, none a regression):**

| box | arm | result |
|---|---|---|
| halo (gfx1151, 1 dev, UMA) | build | green (VMM on, as block 15 requires) |
| halo | dense 4B coherence | `1de99b7e6ee6` — identical to gfx1201 |
| halo | `gate-kv-residency.sh` window section | **all PASS**, `wref == wres == wspill` `be904e86f248` (identical to gfx1201), `wspill` 16 flips / 64 MiB host-backed; provider selftest PASS |
| halo | `gate-kv-residency.sh` pool section | **blocked**: `pref` now gets past `-ncmoe` (the fix above) but still dies in the VMM pool free-order assert (`ggml-cuda.cu:1360`) once an MTP drafter is attached — r235 aborts identically (`:1375`), so pre-existing; `TODO` #298 |
| halo | selftest sensitivity arm | `wselftest0` **passes with `LLAMA_KV_REGION_QUARANTINE=0`**: the hazard the hold guards does not reproduce on gfx1151's driver.  The delivery is fine (the hazard case verifies); the gate's "must FAIL without the hold" is gfx1201-observed driver behaviour — gate defect, `TODO` #298 |
| fingon (gfx1100, 1 dev, 24 GiB) | build | green, but the box **rebooted 3× during full-parallel builds** (power draw); `JOBS=4` + a clean rebuild completed.  One power cut left a zero-length `refs/heads/rebase-r236` (repaired; `master` was intact) |
| fingon | dense 9B coherence (no 4B on the box) | `1de99b7e6ee6` |
| fingon | Flash-Next `-ncmoe 48` | exit 0, `7a8734a9f1a6` — **identical to the gfx1201 3-GPU tensor arm**, and host experts work on a discrete card |

**Two gate defects fixed in the same session** (they read FAIL on a healthy platform): `wselftest0` asserted a
*driver* property (the quarantine hold is needed) as if it were universal — the hazard does not reproduce on
gfx1151, where the whole hazard sequence verifies with the hold off — so the arm now asserts the MECHANISM on every
arch (the verdict names `quarantine hold on|off`, the hazard case must be present in both arms) and the corruption
expectation only for arches in `SELFTEST_HOLD_VACUOUS_ARCHS` (default `gfx1151`, added with that measurement;
`SELFTEST_STRICT=1` forces it anywhere); and the pool section now grades a platform's *decline* (`TODO` #300)
instead of residency numbers that cannot exist.  Rationale lives in the gate's own header.

- status: **DONE** · sha: `2dc4511b8` (tree `7d6e24c12`) + follow-up `328a155a9` (tree `a101f0d66`) · session 2026-11-08


## P1 — pre-R17: `TODO` #294, the gates' kernel-journal window · **DONE 2026-11-08**

The pre-R17 list runs as `P<n>` items, not rungs (a rung is a block).  `P1` is `TODO` #294: a gate that cannot
tell "a fault in my run" from "a fault earlier today" costs a re-run and invites the wrong conclusion.

**The handle's premise was slightly off, and the correction is the reason the change is bigger than 3 files:**

| gate | actual state before |
|---|---|
| `gate-grid-stall-pad.sh` | called the check with **no `--since`**, and only **inside `if B_ARM=1`** — an `A_ARM=1 B_ARM=0` run had *no* fault verdict |
| `gate-spec-draft-window.sh` | **never called it** (its header said "the caller's job") |
| `gate-spec-reuse-window.sh` | same — never called it |
| `gate-ckpt-reuse.sh`, `gate-state-pipe.sh`, `gate-spec-ckpt-device.sh` | `--since "30 min ago"` |
| `gate-spec-align.sh`, `gate-spec-width-band.sh` | `--since "120 min ago"` |
| `gate-hc-fusions.sh`, `gate-host-pool.sh`, `gate-kv-residency.sh`, `gate-kv-window-split.sh`, `gate-park-idle-slots.sh`, `gate-prefill-logits.sh`, `gate-qwen4exp-quant-coherence.sh` | **no fault check at all** |

**A fixed window is wrong in both directions**, which is why every site moved rather than just the 3: too wide
reaches back into an earlier arm's fault (the #294 false FAIL), and too narrow **misses the gate's own early
arms** on a run longer than the constant — a false PASS, which is the dangerous direction (`gate-kv-residency.sh`
runs 25-40 min).

**Resolution (maintainer's call: keep the 9, and add the missing checks).**  Every one of the 16 gates that
touches the GPU now:
* takes a **start timestamp before the first arm** (`GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S')"`,
  the form `gate-norm-rows-alias.sh` already used — one second of slack for the journal's timestamp granularity)
  and passes it as `--since`;
* accepts **`--no-faults`** (uniform interface across all 16; for a box with no readable kernel log);
* feeds **every log it wrote** to `--log` rather than a hand-picked subset;
* prints a **PASS line on success** — a silent check is indistinguishable from a skipped one, and the point of the
  handle is a trustworthy verdict.

`gate-per-block-build.sh` is the one gate with **no** check, by design and now documented in its header: it only
compiles, so it cannot record a GPU fault.

Two placements were wrong beyond the window: `gate-grid-stall-pad.sh`'s check was nested in its `B_ARM` block, and
`gate-norm-rows-alias.sh`'s was nested in its model-arm block and only ever scanned `fused.log`/`chain.log` — so
section A's GPU driver arms were never scanned.  Both now run unconditionally over `$KEEPOUT`.

**Verified (not just `bash -n`).**  `bash -n` on all 17 gate scripts; `--help`/`--no-faults`/unknown-arg→exit 2 on
the gates that gained flags; `check-gpu-faults.sh --since "YYYY-MM-DD HH:MM:SS"` accepted by journalctl; real runs
on the r236 build: `gate-grid-stall-pad.sh` **A-only PASS** (the new `==> GPU faults (since …)` block, 4 logs),
**B-only PASS**, **`--no-faults` correctly skips**; `gate-qwen4exp-quant-coherence.sh` (IQ3_XXS, `NGEN=8`) **PASS**
with `PASS  GPU faults: kernel journal (since 2026-11-08 14:52:54) and every arm log clean`;
`gate-park-idle-slots.sh` **PASS** with its own fault line.  A *stale* fault cannot be injected from userspace, so
that direction rests on the mechanism (the window starts at the gate) — which is the whole change.

- status: **DONE** · scripts only (no fork change) · session 2026-11-08

## P2 — pre-R17: `TODO` #297, the sparse QSA decode arm gets a gate · **DONE 2026-11-08**

R16 found that `indexer-score.cu` called `ggml_rope_get_truncate(dst)` on a `GGML_OP_INDEXER_SCORE` node
(upstream `#30206` asserts the op type), so **every** sparse QSA decode run aborted from R12 to R16 and nothing
noticed: the coherence arms and both qwen4exp gates run the *dense* arm.  The gap, not the abort, is the item.

**Why the dense arm is what everyone ran.**  The sparse arm is env-forced and conditional on depth:
`LLAMA_QSA_DENSE_DECODE_UNTIL=0` disables the dense-decode branch, but the sparse path still only engages once
`n_kv > width = indexer_top_k + r - 1` (~2051 tokens) — below that the `shortcut` branch stores keys and never
builds a QSA graph.  So any prompt under ~2.1 k tokens runs no QSA at all, silently (the model file's own comment
records the same trap for a `-c 2048` PPL exercise).

**The gate (section E of `gate-qwen4exp-quant-coherence.sh`, 1 GPU `-sm layer`, ~20 s/arm).**  Three arms on
`prompts/prose-rdna-boosts.txt` (5246 tokens, above the width), each `-n 32`, seed 42, temp 0:

| arm | env | census | text |
|---|---|---|---|
| `sparse` | `LLAMA_QSA_DENSE_DECODE_UNTIL=0` | **4 fused-score lines** | `200edcd422c7` (160 chars) |
| `chain` | + `GGML_CUDA_QSA_INDEXER_SCORE=0` | **0** | `200edcd422c7` — identical |
| `dense` | (shipped default) | **0** | `13406dc51399` (167 chars) — different |

Four assertions, in the order that makes each one non-vacuous: the sparse arm **launched the fused op** (a text
comparison cannot see an op that never runs), the census is **sensitive** (0 lines with the op off and on the
dense arm), **fused == chain** byte for byte (the purity contract: the fused op replicates the per-op rope +
`mul_mat` + relu + head-sum chain exactly), and **sparse != dense** (so "fused == chain" cannot be satisfied by
never taking the sparse arm at all).

**The census needed 23 lines of delivery code** (`GGML_CUDA_QSA_SCORE_LOG=1`, block 11's `indexer-score.cu`):
nothing logs the sparse path, `GGML_CUDA_IDX_RELU_SUM_LOG` cannot serve as the counter (its matcher needs
`nt*ns >= 64`, and decode is `nt = 1`), and `rocprofv3` would be a per-box dependency.  stderr, so it lands in
every arm's log at any verbosity and cannot perturb the generated text.  Measured **inert**: dense and sparse
hashes are identical with the census compiled out and in (`13406dc51399` / `200edcd422c7`), and the census
prints only when the env is set.  Fork commit `08d6c343b` (tree `ccbbcf7a4`).

**Negative-tested**: pointing `QSA_PROMPT` at a 5-token prompt (i.e. below the selection width, the exact
"unreachable arm" shape) makes the gate FAIL on two assertions — 0 census lines and sparse == dense — and exit 1.
The `test-backend-ops` case the handle called optional was **not** added: `GGML_OP_INDEXER_SCORE` has no CPU
implementation, so an op case would need a hand-written reference for the whole fused sequence; the per-op chain
arm *is* that reference, at the model level.  Recorded in `ENVIRONMENT.md`.

- status: **DONE** · fork `08d6c343b` (block 11) + `scripts/gate-qwen4exp-quant-coherence.sh` section E · session 2026-11-08

## P3 — pre-R17: `TODO` #298, the pool-order sweep (one live defect found and fixed) · **DONE 2026-11-08**

The rule (R13's `hc_combine_norm`, re-learnt at R16's `mul_mat_id_mmb_glu`): **a `pool_stack`-registered
allocation must be created before any plain device-pool allocation in the same scope.**  The pool is a strict
LIFO stack — `ggml_cuda_pool_vmm::free` asserts `ptr == pool_addr + pool_used` — and a registered block lives to
the end of the graph, so one allocated *after* a plain temporary is freed last, below a block that has to go
first.  The sweep is a *reader's* job ("it passed once" is not evidence: one build aborted 5/5 in a dense 4B
warmup and the next 0/4).

**Every `pool_stack_push` site and its callers, audited:**

| site | callers audited | verdict |
|---|---|---|
| `common.cuh` `bf16_src1_cache_put()` | `ggml_cuda_mul_mat_cublas_impl` (`ggml-cuda.cu`), `ggml_cuda_hc_combine_norm_set_bf16_src1` (`hyperconn.cu`) | **VIOLATION in the cublas caller** — `src0_alloc` filled before the put.  **Reproduced and fixed** (below).  The hyperconn caller allocates nothing in its scope: clean |
| `hyperconn.cu` `hc_base_copy_del` | `ggml_cuda_hc_combine_norm_try_fuse` | clean — nothing plain precedes the push (this is R13's fix, still correct) |
| `mmb.cu` `mmb_cache_insert()` | `mmb_bf16_activation` → `ggml_cuda_mul_mat_mmb` (2 sites), `mul_mat_id_mmb_glu`, `ggml_cuda_mmb_cache_reserve` | clean — every caller pushes before its first plain allocation |
| `mmb.cu` `mmb_slot_reserve()` | `mul_mat_id_mmb_glu` (slot 2), `ggml_cuda_mul_mat_mmb` (slots 1/3), `ggml_cuda_mmb_reserve_auto` (`dsv4-hc.cu` x2, `unary.cu` x6), `ggml_cuda_mmb_cache_reserve` | clean — R16's fix holds (the slot is reserved first in `mmb_glu`), and the `unary.cu`/`dsv4-hc.cu` wrappers allocate nothing before the call |

**The defect (live, not latent).**  `ggml_cuda_mul_mat_cublas_impl` declares `src0_alloc`/`src1_alloc`, fills
`src0_alloc` when `src0->type != compute_type`, and only then takes the BF16 src1 cache path
(`bf16_src1_cache_put` → registered, long-lived).  Arming it needs BF16 compute with a non-BF16 src0, which
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16` (documented) or an F16 GEMM at acc-precision BF16 provides.  Reproduced on a
4B **F16** model (`llama-quantize --allow-requantize` from Q8_0), `GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16`, 121-token
prompt:

```
free: device 0: pool free OUT OF ORDER: freeing 41943040 bytes at offset 0, but the LIFO contract expects
the newest allocation at offset 619520 (live extent 42562560)
```

40 MiB = the src0 F16→BF16 conversion; 619520 bytes = `2560 x 121 x 2` = the BF16 src1 copy sitting on top of it.
Fix (`9b7dfb47a`, tree `801148bb3`): reserve the cached block (lookup or put) before `src0_alloc` is filled, keep
both conversions in place so the stream order and the numerics are untouched.  After: rc=0, no assertion.

**Inertness measured on the class the cache serves** — Flash-Next GSQ-IQ3_XXS is the FUSED (BF16 `hc_attn_down`)
mixer class: dense `13406dc51399`, sparse `200edcd422c7`, chain `200edcd422c7` (all identical to the pre-fix
build), and `gate-hc-fusions.sh` **PASS** with all five hard arms at `d62340fbf566` (the two `dsv4` soft arms
still report the documented r235-numerics drift, `TODO` #292).

- status: **DONE** · fork `9b7dfb47a` (block 10) · session 2026-11-08

## P4 — pre-R17: `TODO` #301, the MoE prefill delta — localised, fixed, and attributed · **DONE 2026-11-08**

`TODO` #301 was the campaign's only *measured* regression: r236 −2 … −3 % on MoE prefill, decode flat.  The
earlier record ruled out MMB, MMQ and the fusion dispatcher by switching each off; two of the three arms were
real, and the profile below shows the answer was never in those switches.

**Localised by kernel census, not by bisect.**  `-lv 4` on both binaries: **identical graph** (3033 nodes, 2
splits, same buffers) and identical grids/blocks/launch counts for every kernel — so the delta is inside the
work of two kernels, and `rocprofv3 --kernel-trace` named them:

| kernel | r236 | r235 | delta |
|---|---|---|---|
| `mm_ids_helper<8>` | 282 launches, 19.0 ms | 188, 13.4 ms | **+94 launches** |
| `flash_attn_ext_f16<128,128,8,8,…>` | 19.6 ms (96) | 15.9 ms (96) | **+23 %** at identical geometry |

**The +94 was the re-base's own artefact.**  `ggml_cuda_mul_mat_q_pair` launched the ids compaction once per
arm in r236 (its own comment: both calls "write the same mapping and only tile_spec/expert_bounds differ"); on
r235 it launched it **once**.  R01's re-expression of the compaction onto upstream's `tile_spec` grid created
the second launch.  Fix (`30277a117`): one compaction + reuse the tile_spec on equal `J` (measured `J0 = J1 =
48`, always equal for a gate+up pair since `J` depends on type/`ne[1]`/ncols) or a new `mm_id_retile_kernel`
(one block per expert, from the bounds the helper already wrote).  Census after: **188**, r235's count.

**Result** (interleaved `llama-bench -p 512`, 2-4 rounds): 30B-A3B Q4_K_M 1 GPU **−0.7 … −1.1 %** (was
−2.0 … −3.2), 35B-A3B Q8_0 3 GPU tensor **−0.1 … −0.5 %** (was −2.0), dense 27B **−0.1 … −1.1 %** (was −1.4).
Decode untouched.  `MUL_MAT_ID` 931/931, texts byte-identical, `gate-hc-fusions` PASS.

**What remains is a different defect, and it is ours:** the FA kernel.  With the *packed* mask, r235's FA was
**14 % faster than unpatched upstream** and r236's is **20 % slower than r235's** (16.0 vs 13.4 ms), spilling
40 B more per thread (scratch 336 → 376) at the same 256 VGPRs; the only signature change is upstream's
`KV_rows_ptr`.  The derived-mask path costs +19-22 % over packed in *both* trees, so that part ported fine.
Carried as `TODO` #302 (a `fattn-mma-f16.cuh` port job), ~1 % of the MoE wall time.

**Also corrected:** `GGML_CUDA_DISABLE_MMQ` **does not exist** in this tree (only `GGML_CUDA_DISABLE_MMQ_ROUTED`),
so the earlier "not MMQ" arm re-measured the default.  Recorded in the new benchmark file, not by editing the
dated one.

- status: **DONE** · fork `30277a117` (block 00/08) · `benchmarks/2026-11-08-rebase-r235-vs-r236-localisation.md` · session 2026-11-08

## P5 — pre-R17: `TODO` #302, the FA tile loader's row indirection — recovered · **DONE 2026-11-08**

The `#301` residue.  r236's port of upstream's `kv_rows` (`#29510`) put a runtime `indices ? max(indices[i],0) : i`
select inside `flash_attn_ext_f16_load_tile`'s **per-chunk** loop, on every shape that never uses kv_rows.

**Measured, one arm, four builds** (`rocprofv3`, 30B-A3B Q4_K_M pp512, `flash_attn_ext_f16<128,128,8,8,...>`,
96 launches, identical grid and VGPR count):

| build | FA time | scratch |
|---|---|---|
| upstream `a691b2345`, unpatched | 15 539 us | 348 B |
| ours r235 | 15 867 us | 336 B |
| ours r236, as shipped | **19 576 us** | **376 B** |
| ours r236 + the fix | **15 727 us** | **336 B** |

**Three approaches that failed first** (recorded so nobody re-tries them): hoisting the index out of the chunk
loop and making it 32-bit changed nothing (19 284 us -- it is register pressure, not loop placement); a runtime
branch around a lambda-inlined pair of specialisations was **worse** (35 489 us, 1504 B scratch, because both
are inlined into one live range); and a kernel-level template bool would double the instantiations of **290**
template-instance TUs.

**The fix** (`2ff384571`): `load_tile_impl<..., bool use_indices>` (the body, compile-time), a `__noinline__`
`load_tile_rows` for the kv_rows variant (upstream's rare multi-sequence unified-KV path, disjoint from sparse by
the `GGML_ASSERT(!kv_rows || !use_sparse)` in `fattn-common.cuh`), and the existing `load_tile` entry now picks
one of the two **once per tile load**.  Call sites unchanged.

**Result:** scratch back to r235's 336 B and time to r235's level, i.e. the FA regression is gone.  Wall clock
(interleaved `-p 512 -r 5`, 3 rounds): 30B-A3B **+0.1/+1.8/-0.7 %**, 35B-A3B 3 GPU tensor
**-1.5/-0.3/+1.9 %** -- both parity (they were -2.0..-3.2 % before `#301`+`#302`).  Dense 27B is still **-1.5 %**:
its FA is now 8.8 vs r235's 8.5 ms, and the residue is `mul_mat_q` (**+1.4 %** at the *same* instantiation, grid
and launch count, VGPR 232 -> 216) -> `TODO` **#303**.

**Gates:** `gate-prefill-logits.sh` **PASS** against a base recorded from the **unpatched `a691b2345`** build
(mean KLD 0.000697, same-top-p 98.784 %) -- this also satisfies R17 step 5's "record from a clean unpatched
build" for the f16 arm, and `build-rocm-base` is kept for the `--kv q8_0` arm; `gate-spec-draft-window.sh` PASS
(the 2-slot unified-KV path, identical acceptance both arms); 30B text `e1325adad141`, Flash-Next
`13406dc51399` unchanged.

- status: **DONE** · fork `2ff384571` (block 05) · session 2026-11-08

## P6 — pre-R17: the upstream candidates and the remaining measurements (`#286`, `#287`, `#289`, `#290`) · **DONE 2026-11-08 (one item halved)**

Four handles off the list in one pass; no delivery code changed.  Record: `WORKLOG.md` 261108.007.

**`#286` (Meta compute headroom, `D7`) — closed, both constants stay.**  `max(128*n_tensors, 256)` views: the
floor only exceeds the multiplier at `n_tensors <= 2` (a Meta compute container holds far more), so it cannot
replace the multiplier; and the container **grows** on demand, so neither can cause a failure — the multiplier is
a first-context sizing heuristic.  Measured on Qwen3.5-4B Q8_0 (GDN hybrid), 2 GPU `-sm tensor`, headroom
**128 / 16 / 1** and `GGML_COMPUTE_BUFFER_MARGIN_PCT=0`: five arms, Meta compute buffer **75.94 MiB** (69.04 with
the byte margin off), one text `6541eadb9041`.  `D7`'s question answered: the hybrid recurrent graph allocates
fine with the margin at 0.

**`#289` (upstream PR owed: `find_slot()` must swap `mem_cell::rs_max`) — closed.**  `upstream/recurrent-find-slot-rs-max/`
(README + patch, applied tree `bf05f3c47`, sha256 `8247e093…`), one line beside the three existing swaps,
`git apply --check` **clean** on pristine `a691b2345`, repro + filing checklist in the README, row added to
`upstream/README.md`.

**`#287` (re-cut the mul-mat-id-dup candidate) — HALF DONE, parked as WIP.**  See
[`../upstream-mul-mat-id-dup-recut/README.md`](../upstream-mul-mat-id-dup-recut/README.md): the
per-occurrence compaction is ported onto upstream's `tile_spec` writer, both `mmf` gates are removed (the
delivery's own 6-line delta) and the `#if 0` dup cases are enabled — and it still fails (`f16` numerically at
every width, `q8_0` at `n=9` page-faults in `mul_mat_q<(ggml_type)8, 16, …>`).  The missing piece is the MMQ/MMF
**ids kernel** work, entangled with the delivery's pair fusion, so it must be written rather than lifted.

**`#290` (dispatcher merge debt) — pre-checked.**  `ggml_cuda_try_fuse_legacy()` / `GGML_CUDA_FUSE_LEGACY` are
absent, `ggml-cuda-try-fuse.cu` is the only dispatcher TU, so R11's reconciliation is what ships; the handle keeps
only its own R17-step-1 acceptance (confirm it **in the re-cut block-10 patch**).

**A new delivery handle fell out of the #287 work — `#304`.**  With upstream's dup cases enabled, the **delivery**
is **15/16**: everything passes except `MUL_MAT_ID(type_a=f16, n=9, dup=1)` (ERR 0.374).  Because upstream keeps
those cases `#if 0`, the duplicate path has no standing oracle in `patches/` at all.  Recorded with the recipe
(fix the narrow-`f16` compact path, then enable the cases in the delivery's test suite).

- status: **DONE** (`#286`, `#289`, `#290` pre-check) / **HALF** (`#287` → WIP) · session 2026-11-08

## P7 — pre-R17: `TODO` #303, the dense MMQ residue — it is the base commit, not the delivery · **DONE 2026-11-08**

`#303` was the last prefill gap: the dense 27B Q8_0 arm **−1.3 … −1.6 %** against the shipped r235, filed as
"the re-base's `mmq_args`/`tile_spec` re-expression changed the dense body (VGPR 232 → 216)".  It is closed with
**no delivery change**, because the kernel turned out to be upstream's, byte for byte.

**Method: the ISA, not the profiler, and 2 interleaved rounds when timing is unavoidable.**  The gfx1201 code
objects were unbundled out of `libggml-hip.so` and the dense instantiation compared as a normalised instruction
sequence (exact, noise-free): **r236's is identical to unpatched `a691b2345`'s** (1436 instructions, 0 diff lines).
The delivery has no code presence in that kernel, so VGPR 232 was never a mechanism — it was the *pre-`#31047`*
body r235 inherited, which R01 correctly replaced.

**Attribution, four builds × 2 interleaved rounds** (`rocprofv3`, 794 launches, identical geometry and `J = 128`
in all four — so no runtime `J`/`ntj`/`nthreads` difference exists):

| build | mean dense-kernel ms | delta | instrs | VGPR |
|---|---|---|---|---|
| `6050b467c` unpatched | 521.32 | — | 1432 | 216 |
| r235 = 6050 + delivery | 519.43 | **−0.36 %** | 1476 | 232 |
| `a691b2345` unpatched | 528.14 | **+1.31 %** | 1436 | 216 |
| r236 = a691 + delivery | 528.64 | **+0.09 %** | 1436 | 216 |

So ~¾ of the gap is upstream's own change between the two fork points, ~¼ is a 0.36 % allocation artefact of the
dropped body, the delivery is neutral on the new base and *protective* on the whole trace (+1.47 % base-only vs
+0.89 % with the blocks).  The same base also costs FA +8.2 % and ~1.85 ms of `fillBufferUnAligned` work — several
kernels move together, i.e. a base-level characteristic.

**Two fixes tried and both refuted at ISA level, free.**  (1) drop the dense per-tile `min()` (r235's column-range
shape, provably equivalent): compiled in, **1436 → 1435 instructions, VGPR unchanged**, arm wall clock unchanged
(−1.16/−1.22/−1.29 % vs r235).  (2) type-gate the NVFP4 scale plumbing behind `if constexpr` (exactly what
`a691b2345` restructured): compiled in, Q8_0 ISA **byte-identical**.  Both reverted.  What remains between the fast
and slow body is four static instructions (`v_cmpx_ge_i32` ↔ `v_cmpx_gt_i32`, one `s_min_i32`, four FMAs that stop
pairing into `v_dual_fmac_f32`) — nothing to remove.

**Dropped by decision:** the 232-VGPR allocation win (−0.36 % of one kernel) — recovering it needs a semantically
dead construct to steer ptxas, for ~0.3 % of an arm under a ±1 % measurement floor.

**Instrument facts that outlive the item:** two traces of the *same* binary differ by **1.1 %**, so a single-run
census cannot resolve this class of delta (interleave builds *within* a round); and the arm runs **794** dense
`mul_mat_q` launches — **986** is the *total* (`794` dense + `192` ids).  Both are in `INVARIANTS.md`.  Also
fixed on the way: `master` in `~/llama.cpp` was carrying a local #289 candidate (so "unpatched upstream" would have
silently been the wrong tree); it is pinned at `a691b2345` and `HANDOVER-303.md` §7 + §5 now say to build the
attribution base from `git checkout --detach a691b2345`, never from `master` (R17 step 5 records the KLD base from
it).

- status: **DONE, no delivery change** · record: `benchmarks/2026-11-08-dense-mmq-303.md` · `WORKLOG.md` 261108.008 · session 2026-11-08

## P8 — pre-R17: `TODO` #304 fixed (a wrong-answer bug with a standing oracle) and the pre-release verdict automated · **DONE 2026-11-08**

`#304` was the last correctness handle on the pre-R17 list and it was **shipping untested**: upstream keeps the
`[TAG_MUL_MAT_ID_DUPLICATE_ID]` cases `#if 0` "until CUDA backend support is added", the delivery has had that
support since block 00, and with the cases enabled it was **15/16** — `f16` at `ncols_dst = 9` returned wrong
numbers (ERR 0.374 vs 5e-4).

**Cause: two files disagreed about the compact column-tile width.**  `mmf.cu` builds the per-expert
seam/`tile_spec` grid and sizes `max_tiles` with a **fixed stride of 16**; `mul_mat_f_switch_cols_per_block`
derived the kernel's `cols_per_block` from `ncols_dst` **clamped to 16**.  For `ncols_dst < 16` the kernel
therefore processed only that many of every 16-wide tile's columns and **silently dropped the tail of each
expert's compact range**.  With duplicate ids an expert's range is its `(token, slot)` occurrence count, which can
be far wider than the token count — so the narrow width was the only broken one (`16/33/64` all clamp to 16) and
the quantized cases never noticed (they take `mmq`, not `mmf`).

**Fix** (fork `3b0978888`): name the contract — `MMF_IDS_COLS` is used by the seam builder and `max_tiles` in
`mmf.cu` and by the dispatch in `mmf.cuh`, and the compact path always dispatches that width; the non-compact ids
path and plain `MUL_MAT` still tile by `ncols_dst`.  **The duplicate cases are enabled** in
`tests/test-backend-ops.cpp` as the standing oracle (they are what found this; no f16 MoE model exists on the box,
so the suite is the only oracle for that path).  Verified: `dup=1` **16/16**, full `MUL_MAT_ID` **947/947**, dense
`MUL_MAT` **2517/2517**.

**`#303` got four more levers, all refuted at instruction level** (occupancy 2→1 for the Q8_0/J=128 row:
byte-identical kernel; the J tile-width knob: J=128 already optimal, J=64 −4.3 %; the delivery's own dense MMB
path: −0.15 %; a probe with the whole ids branch compiled out of the kernel: the dense loop still comes back in
the slow shape).  Six levers total, six null, on a kernel that is upstream's byte for byte — so the 1.2 % dense
arm stands as an inherited upstream cost and the fix belongs upstream.

**Automation: `scripts/gate-rebase-release.sh`** turns the pre-release question into one command and one verdict —
state (with `master` pinned at the fork point), build per block, prefill-logit KLD, the `MUL_MAT_ID`/`MUL_MAT`
oracles, coherence texts against recorded hashes, an **interleaved** performance A/B of every arm against the
shipped build, and the GPU-fault scan scoped to the run's own start.  It delegates every rule to the gate that
owns it, **fails rather than skips**, and exits `0`/`1`.  First run (1 round): **dense-27B-prefill −1.20 % FAIL,
moe-35B-prefill +0.75 % PASS, moe-35B-decode +0.02 % PASS** — `#303` is the only red arm, and the decode arm is
also the `#279` A/B, so that handle closes on the same run.

- status: **DONE** · fork `3b0978888` (block 00/08 surface: `mmf.cu`/`mmf.cuh` + the delivery's test suite) · `WORKLOG.md` 261108.009 · session 2026-11-08

## P9 — pre-R17: `TODO` #296, `D8`'s CPU-win measurement — two dead hunks dropped, and the measurement method itself was the finding · **DONE 2026-11-08**

`#296` asked which of block 12's CPU hunks (`arch-fallback.h`, `arch/x86/quants.c`, `ggml-cpu.c`, `ggml-cpu.cpp`,
`repack.cpp`) still pay after upstream's CPU-backend rewrite.  Answer: **three of the five are not perf hunks at
all, two were literally dead, and the one measured CPU win is in a file D8's list does not name.**

**Hunk by hunk (measured, not assumed).**
* `arch-fallback.h` — the delivery's only change was a **one-line `// quants.c` comment** (blob-level diff against
  `a691b2345`: one line added).  **Dropped**; the file is byte-identical to upstream's again.
* `ggml-cpu.cpp` — `static const char * env = getenv("GGML_CPU_MOE_OFFLOAD_THREADS")`, inside
  `ggml_backend_cpu_moe_offload_cap()`, which returns early from its own `static int cap` memoisation → the
  `static` can never execute twice.  **Dead; dropped.**
* `repack.cpp` — three `nullptr` entries for the new buffer-type fields.  No behaviour change, but compiled without
  them (same TU flags, `-Wextra`) the compiler emits `-Wmissing-field-initializers` on `repack.cpp:5254`, so the
  hunk is what keeps the tree warning-clean.  **Kept.**
* `ggml-cpu.c` — `GGML_OP_ALL_GATHER` → `dup` + its `ggml_get_n_tasks` entry: the cross-backend consistency fill
  (a CPU split carrying the delivery's all-gather node aborts without it).  **Kept.**
* `arch/x86/quants.c` — the AVX2 Q2_0 dot, pure by design (bit-identical to the generic body).  `test-backend-ops
  -b CPU -o MUL_MAT -p type_a=q2_0` **85/85**; throughput **0.66-0.69 TFLOPS** against upstream `#30447`'s body
  **1.23-1.34 TFLOPS** (n=1: 177.74 vs 95.10 µs) — the cost of purity, on a type no delivery model uses.
  **Kept**, recorded.
* The measured win is block 12's **AVX2 top-k block prefilter** in `common/sampling.cpp`: **+0.54 %** decode
  (103.93 vs 103.37 t/s, 3/3 rounds, 4B Q8_0, `--temp 0.8 --top-k 40`), text **byte-identical** (`7f950c7d0f76`)
  — a filter, not an approximation.  **Kept.**

**The bigger finding is the method.**  D8's "host-expert CPU path" `pp`/`tg` arm, taken the way the item said, said
the delivery is −15…−18 % prefill against unpatched `a691b2345`.  It is an artifact: `llama-bench` builds its
prompt from `std::rand() % n_vocab`, so every ubatch routes over every expert and the heat-ranked arena is inert
(pp1450 = 414.1 / 414.5 / 414.5 / 414.5 t/s across `MOE_EXPERT_CACHE_MIB` 0 / 4096 / 8192 / auto).  Driven with
real text, the same arm is **1064 vs 379 t/s (+181 %)** and the arena sweep is 401.6 → 583.6 → 676.9 → 1063.6.  The
residual floor under no locality (414 vs 508, flat across every knob: slab, devgather, host tier, cold path, host
buft, CPU offload threads) belongs to `TODO` #273 and is recorded there.

**Verification.**  Clean rebuild after the two drops (a widely included header); coherence 4B `1c5d32ac537d`
unchanged; the same host-expert arm reads `pp512` 437.2 / `tg64` 80.1 against 438.2/438.3 and 76-82 before, i.e.
inside its own spread.  **Gate extended**: new step **5b `textprefill`** in `scripts/gate-rebase-release.sh`
(real-text prompt-eval on `-ncmoe 48`, `--text-tolerance` default 3.0 %), first run r236 vs r235 **−0.45 %**
(−0.46 / −0.45 / −0.45, spread 0.01) PASS — so a future change to the arena or the host transport can no longer
pass a `llama-bench`-only gate.

- status: **DONE** · fork `8715156da` (tree `a4093f9da`) · `WORKLOG.md` 261108.011 ·
  `benchmarks/2026-11-08-host-expert-prefill-transport.md` · session 2026-11-08

## P10 — pre-R17: `TODO` #292, the dsv4 soft arms — no drift, a stale header line, and the anchor that closes the hole · **DONE 2026-11-08**

`#292` claimed the dsv4 `LLAMA_FUSED_DSV4_HC_PRE/_POST` arms "reproduce r235's numerics, not the re-based chain's",
i.e. that the HC chain moved with the re-base while the fused ops did not.  **It did not: the shipped r235 build
reproduces the re-based tree's split exactly.**

| arm | r236 `8715156da` | shipped r235 `69d36099b` |
|---|---|---|
| `chain_chain` (reference) | 449 `d62340fbf566` | 449 **`d62340fbf566`** |
| `mixed_comb` / `mixfused_comb` / `fused_fused` / `no_cuda_fuse` | `d62340fbf566` | `d62340fbf566` |
| `dsv4_pre_chain`, `dsv4_post_op` (soft) | 449 `6a102dbc62d6` | 449 **`6a102dbc62d6`** |

(Flash-Next GSQ-IQ3_XXS, BF16-hc class, 3 devices `-sm layer`, 96 tokens, prompt sha `18147e48a7f4`.)  The
"`6a102dbc62d6` is what the chain produced at r235" clause came from the gate's r156 header line -- and from
`MANIFESTS.md` gate 11's r148 row, where `6a102dbc62d6` is genuinely the chain *of that date*.  The chain has
since moved to `d62340fbf566`; the header was never updated.  So there was no re-base artefact to explain.

**The numerics, quantified (`gate-prefill-logits.sh --ab`, Flash-Next, wikitext 512 x 40, 3 devices `-sm layer`,
base PPL 2.8489):** fused pre op vs its chain **mean KLD 0.013217 / same-top-p 96.392 %**; fused post op vs its
chain **0.013198 / 96.186 %**.  That is the documented HC16 bf16 producer-slot re-baseline (the gate's own r156
IQ4_NL reference: 0.027936 / 96.471 % at base PPL 1.7888), and *smaller* in KLD.  Both texts stay fluent and
diverge from the first sentence; each arm is deterministic per class across builds, which is what makes an
anchor valid.

**Resolution (no kernel change -- `hyperconn.cu`/`dsv4-hc.cu` are correct):**
* the gate header no longer claims byte-identity on the BF16-hc class: both classes are numeric arms, with the
  measured KLD/same-top-p stated, and the r156 IQ4_NL figure kept for scale (noted alongside `MANIFESTS.md`
  gate 7's different IQ4_NL number, 0.021542 / 95.814 %, which is its own unnamed point -- neither is a target);
* **`SOFT_EXPECT` anchors the two numeric arms to their own recorded text** (defaulted to the fused-mixer
  class's `dsv4_pre_chain=dsv4_post_op=6a102dbc62d6`, 449 chars; unset for other classes so they do not inherit
  it).  A numeric arm no longer just prints "reported": if its sha moves, the arm **FAILs**.  That is the hole
  `#292` actually exposed -- a real change in either op used to pass silently.
* Negative-tested: with a deliberately wrong anchor the gate names the arm and exits **1**; with the recorded
  anchors it passes (exit 0 by construction, the final `if` block's `echo`).

- status: **DONE, no delivery-code change** · `WORKLOG.md` 261108.012 · session 2026-11-08

## P11 — pre-R17: `TODO` #293, keep or drop our qwen4exp — KEEP, measured at depth and concurrency · **DONE 2026-11-08**

`#293` asked whether to keep carrying the delivery's own `src/models/qwen4exp.cpp` (3264 lines) or take upstream's
(1514).  R12 kept it on a single-stream +31 % decode; the item's open half was the "field config" (Q8_0, 2/3 GPUs
`-sm tensor`) and an unexplained 27B text divergence.  Both are now measured.

**The 27B half was a category error.**  `Qwen3.8-27B-Q8_0` is arch **`qwen35`** (dense), so no `qwen4exp.cpp` swap
can touch it; its prefill-logit KLD against unpatched `a691b2345` is **0.000697 / 98.784 %** (rounding-level, PASS).
The R12 "greedy text diverged" observation was short-prompt noise on a near-tie, not an implementation difference.

**Upstream's file cannot be swapped in anyway**: `a691b2345:src/models/qwen4exp.cpp` does not compile against the
delivery tree (upstream's `llm_graph_input_kpool`, a `hc_mix` graph type and an 8-arg `ggml_hc_mix` were all
re-expressed by our k-pool/HC work).  So the comparison is ours-vs-unpatched, model device-resident on 3 devices
`-sm tensor` (no `-ncmoe`, so `TODO` #296's `llama-bench` artifact cannot enter).

| arm | ours | upstream | delta |
|---|---:|---:|---:|
| `pp512` d0 / d16 384 / d163 840 | 2721.8 / 2327.5 / 1822.2 | 2334.6 / 1807.6 / 666.6 | **+16.6 / +28.8 / +173.4 %** |
| `tg128` d0 / d16 384 / d163 840 | 86.4 / 82.8 / 57.7 | 43.6 / 41.6 / 34.3 | **+98.2 / +98.9 / +68.2 %** |
| real text ~16 K / ~160 K prefill | 1927 / 2134 | 1618 / 1133 | **+19.1 / +88.4 %** |
| batched total, 1 / 2 / 4 slots | 275.6 / 572.7 / 872.7 | 194.6 / 342.5 / 567.9 | **+41.6 / +67.2 / +53.7 %** |

No cell favours upstream, so the item's "mixed bag → merge session" branch never opens and the "wins at depth"
test passes at 160 K.  **DECIDED: keep ours.**

**Two instrument traps met on the way**, both recorded so they are not re-derived: (1) `llama-batched-bench`'s
npl=1 prefill row is a cold-first-pass artefact (ours 0.824 s first, 0.197 s warm; the same trap as
`archive/work/host-expert-eager-push/HANDOVER-R210.md` §3.1) — the steady state is +19.0 %, not −67 %;
(2) r235 cannot run `llama-perplexity` on this model at all (its own pool-order assert, `ggml-cuda.cu:1375`, the
class `#298` fixed), so a delivery-vs-delivery KLD is not available — delivery-vs-delivery was instead confirmed
byte-identical in text (87 chars `ac6445f74aa2` on both builds).

**Order of the KLD ladder** (ours vs unpatched, 512 × 40): dense 27B **0.000697** → MoE 35B-A3B **0.004980** →
qwen4exp **0.013426**, and the qwen4exp number does not move with any switch we ship (HC ops/fusions 0.013955,
derived-mask/QSA/GDN-bf16/rms/all-fusions 0.013081).  So the divergence tracks the delivery's own MoE/attention/HC
re-expression, not the model file — which is why keeping the file costs numerics nothing that a swap could recover.

- status: **DONE** · `benchmarks/2026-11-08-qwen4exp-keep-or-drop.md` · `WORKLOG.md` 261108.013 ·
  `DECISIONS.md` `D3` · session 2026-11-08

## P12 — pre-R17: `TODO` #295, the host-expert multi-slot DFlash detector — wired in as gate step 5c · **DONE 2026-11-08**

`#295` asked for a ladder arm that would catch the failure mode R13's dropped guard protected: with host-resident
experts (`-ncmoe`) and more than one sequence, draft acceptance collapsing to ~0 **while the generated text stays
correct**.  Nothing else in the ladder can see it — the MTP four-axis protocol, `gate-spec-align`,
`gate-hc-fusions` and the coherence arms are all single-sequence — so the resolution is a gate step, and that is
what landed: `scripts/gate-server-arms.sh`, delegated from `scripts/gate-rebase-release.sh` as **step 5c**
(`--only server`).

**The arms** (two-slot `llama-server`, DFlash drafter, `--spec-type draft-dflash --spec-draft-n-max 3`,
`-sm layer` 3 devices, greedy, the shipped prose prompt in both slots so the two sequences verify in the same
decode step, `-n_predict 128`, acceptance read from the per-slot `draft acceptance = …` line at `-lv 4`):

| arm | model + drafter | `-ncmoe` | acceptance | texts |
|---|---|---:|---|---|
| `fn24` | Flash-Next GSQ-RCO-IQ3_XXS + DFlash2-Q8_0 | 24 | 0.74359 / 0.72269 | `b73aa781bca5` / `b337648c3ef9` |
| `fn10` | the same | 10 | 0.74359 / 0.72269 | the same |
| `35b10` | 35B-A3B Q8_0 + its DFlash2-Q8_0 | 10 | 0.59124 / 0.44720 | `e26a28d3f3ae` / `2cf19becd4ee` |

These are **bit-reproducible**: two independent runs of each arm agreed to the digit, including the text, and
`fn24 == fn10` exactly (the `-ncmoe` level does not move the numerics on this shape).

**R13's numbers are not the criterion — and the handover's claim that `RUNGS` R13 carries the invocation is
wrong.**  R13 recorded 0.53793/0.51678, 0.55245/0.47134 and 0.63918/0.62887 for these arms but never wrote the
command down, and none of those values reproduce under any invocation tried (n_max 3 with `-n_predict` 128 on the
shipped prose prompt is what produces the table above; `-ctk/-ctv q8_0 -fa on`, n_max 5 and the GSQ-RCO-IQ3_S
quant were each tried and none lands there).  The item's own criterion — *acceptance not ~0, text unchanged* — is
what the gate enforces, with the baselines measured at r236:

* **HARD** the server comes up and **both slots draft** (`draft_n > 0`);
* **HARD** per-slot acceptance ≥ `ACC_FLOOR` (0.5) × baseline, so the collapse-to-~0 pathology fails;
* **HARD** per-slot acceptance within `ACC_TOL` (0.10) of baseline, so a genuine drift is visible too;
* **HARD** the two texts match the recorded sorted hash set;
* **HARD** `check-gpu-faults.sh` over the gate's own `--since` window.
`--record` prints replacement baseline lines for another model/config.

**Verified**: the gate PASSes on the r236 tree (3/3 arms, no GPU fault); the verdict logic was exercised offline on
the three paths (exact → ok, collapse to 0.0000/0.0003 → `collapsed`, a 0.14 drift → `drifted`, wrong slot count →
`slot count`, i.e. a FAIL each); and it is wired into `MANIFESTS.md` gate 22 + `scripts/README.md` alongside the
step that owns it.

- status: **DONE** · `scripts/gate-server-arms.sh` (new) + `scripts/gate-rebase-release.sh` step 5c ·
  `WORKLOG.md` 261108.014 · session 2026-11-08

## P13 — pre-R17: `TODO` #291 / `D1`, the arena vs upstream's ClassLRU at equal MiB — keep both, ours default · **DONE 2026-11-08**

The last measurement item before R17.  Real text (never `llama-bench`'s prompt, `TODO` #296), 1 x R9700,
35B-A3B Q8_0, two `-ncmoe` levels, two interleaved rounds, **text identical in every arm** (`504 a2cdef23588f`):

| arm (`-ncmoe 20`) | prefill t/s | decode t/s |
|---|---:|---:|
| **delivery arena (default)** | **1441.6** | **72.0** |
| delivery, arena disabled | 757.1 | 48.7 |
| delivery arena + upstream cache 4096 | 808.4 | 63.9 |
| upstream ClassLRU 4096 | 782.3 | 53.3 |
| upstream ClassLRU + `cpu-share` 0.5 / 0.95 | 778.3 / 778.4 | 55.7 / 58.3 |
| upstream, no cache | 747.5 | 42.5 |

`-ncmoe 40`: ours **1394.2 / 69.2**, arena off 371.1 / 31.8, coexist 399.0 / 34.5, upstream cache 455.8 / 32.6,
upstream off 431.3 / 30.8.  Arena census at `-ncmoe 20`: 11 411.2 MiB of 16 320.0 (69.9 %), settling at
10 850.2 MiB, `MoE arena = 0.8787 (27 289 hit / 31 056 reaches)`.

At equal MiB the arena is **+84 % prefill / +35 % decode** (+206 % / +112 % at `-ncmoe` 40); **coexistence is a
loss** (44 % prefill / 11 % decode) because the ClassLRU cache takes VRAM the arena sizes itself against; neither
ClassLRU nor `cpu-share` closes the gap.  **DECISION (maintainer): keep both -- ours default-on, upstream's
opt-in only, mostly for comparison** (it also stays the only device tier on a `GGML_HIP_NO_VMM=ON` platform,
where the arena declines).  "Keep both" is either/or, which is already the shipped default (`--moe-cache-mib` = 0).

**`D1`'s original inversion is stale, and half of it was a measurement-regime artifact.**  D1 recorded upstream
+17.6 % decode (ours 53.5 vs 62.9 at 4080 MiB).  On today's tree the same `llama-bench` arm reads ours **83.9**
against upstream **48.0**, and the arena's prefill is regime-dependent (838 t/s on random tokens vs 1442 on real
text) -- the `#296` effect.  No part of "upstream is faster" survives.

- status: **DONE, no delivery change** · `benchmarks/2026-11-08-moe-cache-arena-vs-classlru.md` ·
  `WORKLOG.md` 261108.015 · `DECISIONS.md` `D1` · session 2026-11-08

## R17 — the release ladder

See [`README.md`](README.md) §4.  Starts only when R01..R16 are DONE and every `DECISIONS.md` item that the
climb touched has a maintainer decision recorded.

- status: **TODO**

---

## Ledger

| rung | block | commit on `rebase-r236` | conflicts found | exit | session/date | notes |
|---|---|---|---|---|---|---|
| R00 | — | `a691b2345` (branch point) | — | n/a | 2026-11-08 | campaign setup; probe run |
| R01 | 00 | `4aaffd1c0` | 12 expected → 12 found | build + coherence + MUL_MAT_ID 931/931 | 2026-11-08 | ballast dropped (94 -Wshadow warnings); mmid re-expressed on upstream's `tile_spec` |
| R02 | 01 | `cadab8c45` | 1 expected → 3 found (one struct, one heuristic, one function) | build + coherence + AR pipeline line | 2026-11-08 | both sides kept; crossover keyed on `n_ranks()` |
| R03 | 02 | `2f0385c71` | 4 expected → 4 found | build + coherence + adaptive line | 2026-11-08 | upstream absorbed the conv-state snapshot store (one set_rows, same `Kw` bound) |
| R04 | 03 | `5a214716b` | 1 expected → 2 found (+1 in delta-net-base) | build + coherence + GDN census + kill-switch arm byte-identical + `test-recurrent-state-depth` **0 failures** (3 phases) | 2026-11-08 | upstream absorbed the conv-state store; `D4` decided: tracker dropped, `rs_max` is the bound; found+fixed an upstream `find_slot` bug (`rs_max` not swapped) → `TODO #289` |
| R05 | 04 | `92cf39297` | 0 expected → 0 found | build + bf16 KV/FA arm | 2026-11-08 | clean; upstream's bf16 fix is independent (`D6` closed) |
| R06 | 05 | `b5d32e0c7` | 9 files expected → 9 files / 48 hunks | build + coherence + derived==packed A/B + q8_0 KV | 2026-11-08 | **derived mask ABI moved to src[6..8]** (kv_rows took src[5]); upstream absorbed the per-ubatch rot fill; no band kernel dropped (`D5`) |
| R07 | 06 | `75ba21116` | 0 expected → 0 found | build + MMB_DENSE line + byte-identical MMB=0 arm | 2026-11-08 | helper API re-expressed with an optional `expert_bounds` output (MMB's own bounds representation kept) |
| R08 | 07 | `f0a594fe1` | 1 expected → 1 found | build + coherence + the touched quant fusions | 2026-11-08 | recorded late (section + row were left `TODO` while the commit existed); mmvq band clamp layered on upstream's `get_mmvq_mmid_max_batch` table |
| R09 | 08 | `af0cae2f1` | 5 expected → 5 files / 8 hunks | build + MUL_MAT_ID 931/931 + fused/chain byte-identical | 2026-11-08 | `D2` complementary (two template arms); `mmq_args` carries `expert_bounds` next to `tile_spec`; norm-rows gate moved to R11 |
| R10 | 09 | `5654a5f95` | 7 expected → 7 files / 12 hunks | build + dense coherence + D1 arms (VMM build) | 2026-11-08 | **arena needs `GGML_HIP_NO_VMM=OFF`** (upstream default is ON, so the tier is inert on the stock build); D1 first numbers: upstream cache +17.6% decode |
| R11 | 10 | `ed1d36436` | 8 expected → 8 files / 22 hunks | build + 3-model coherence + `gate-norm-rows-alias.sh` PASS + MUL_MAT_ID 931/931 + every fusion arm byte-identical to the unfused chain | 2026-11-08 | unblocked: pair `ncols_max`→`ntj` (the fault), routed-compact gate on `nchannels_y` + `tile_y_j_max`→`tile_y_j_sup` (two latent delivery-code losses, +1.9 % prefill), two-copy dispatcher reconciled to one (TODO #290 closed, stale delivery duplicates incl. a non-pure conv-input fold removed) |
| R12 | 11 | `b1b0fa807` | 7 expected → 9 files / 34 hunks | build + 4B/30B unchanged + `gate-hc-fusions.sh` PASS + `gate-qwen4exp-quant-coherence.sh` PASS | 2026-11-08 | **D3 DECIDED by measurement**: upstream has its own qwen4exp; the delivery's is bit-identical on Flash-Next 3-GPU tensor and +31 % decode (70.1 vs 53.5 t/s), neutral on 27B → keep ours; `n_pad` ctor/member dropped (#31200), `get_n_kv` requires n_tokens, `#31733`/`#31073`/`#29510` drift re-derived; quant-coherence gate's model selection fixed; dsv4 soft-arm drift recorded as `TODO` #292 |
| R13 | 12 | `2b09a50b0` | 8 expected → 8 files / 24 hunks | build + 4-model coherence (R12-identical) + `gate-state-pipe.sh` PASS + `gate-park-idle-slots.sh` PASS + `gate-ckpt-reuse.sh` PASS (hard checks) + DFlash2 attach/acceptance arms | 2026-11-09 | **`D15` DECIDED**: F1 deleted tree-wide, upstream's `embd_row` kept, Meta refusal carried, host-experts guard re-measured and dropped (identical acceptance on Flash-Next + 35B-A3B, 2 slots, `-ncmoe` 10/24). State pipe kept (gate: 43.7 vs 74.7 ms). Prompt cache = upstream API + delivery policy, with `ckpt_recycle` restricted to uniquely-owned checkpoints. Two pre-existing R12 defects fixed on the way: the `hc_combine_norm` pool-stack LIFO abort on `-sm layer` (`-sm layer` runs for the first time) and the DFlash injection input never registered (dead drafter: 0.000 → 0.754) |
| R14 | 13 | `ed13a322e` | 3 expected → 4 files / 5 hunks | build + 4-arm coherence (R13-identical) + `--spec-draft-mtp-vocab` on both head paths (shared-NextN via `ctx_other`, self-headed gemma4) + inert when unset | 2026-11-09 | clean rung: no decision needed. `arch == other.arch` dropped from `can_reuse` (`#31733` moved the model into the graph context; `model == other.model` subsumes it); `common_fit_apply()` already carried the delivery's draft-reserve fold; block 13's row-range splice over a vocab-split Meta tensor is exercised by the shared-NextN arm |
| R15 | 14 | `617ee7152` | 0 expected → 0 found | build + flint `-sm relaxed-tensor` == `-sm layer` text + `-sm tensor` still refused with the block's message + 4-arm coherence | 2026-11-09 | zero-conflict, zero-drift rung; the TEMPORARY block stays isolated for its `TODO` #104 removal. flint validated on the box; mellum/mistral4 still have no model |
| R16 | 15 | `2dc4511b8` | 9 expected → 8 files / 12 hunks (`qwen4exp.cpp` auto-merged) | build + 4-arm coherence (R13-identical) + `gate-kv-residency.sh` PASS + `gate-kv-window-split.sh` PASS 38/38 + `gate-spec-draft-window.sh` PASS | 2026-11-08 | **`D12`/`D13` re-derived**: provider still binds at the buffer type (rotation constants are per-cache, not cell-indexed), `hparams.no_alloc` → the ctor's `no_alloc`, pool cells round to whole PAGES (`n_pad` gone), `memory_breakdown()` over `ctxs_bufs.bufs` + `attn_rot_bufs`, `params_l` → `params`.  **Two real defects fixed**: the hybrid ctor's positional argument shift (upstream inserted `is_indexer`; `v_enabled` got `kv_pool` → no V tensor → `cpy_v` null-deref → every GDN model segfaulted at load) and a latent R12 defect in `indexer-score.cu` (`ggml_rope_get_truncate` on a non-ROPE node — the sparse QSA arm has aborted since R12, no gate reached it; §85, `TODO` #297).  r193's draft reserve preserved by folding the `mtp_shared` branch into `common_draft_reserve_derive()`: arena 20 527.5 vs 17 977.5 MiB (+7.8 points) |
| P9 | — | `8715156da` (fork, `ggml-cpu`: `arch-fallback.h` + `ggml-cpu.cpp`) | #296 / `D8`: which block-12 CPU hunks survive the CPU-backend rewrite | DECIDED: two hunks were dead (a comment line, a `static` inside a memoised function) -> dropped; `repack.cpp` keeps its warning fix, `ggml-cpu.c` keeps the `ALL_GATHER` fill, `quants.c` keeps the pure Q2_0 dot (cost measured: 0.66-0.69 vs upstream's 1.23-1.34 TFLOPS); the measured CPU win (AVX2 top-k prefilter) is +0.54 % with byte-identical text | 2026-11-08 | the host-expert `pp` arm turned out to be a `llama-bench` random-token artifact (arena inert: 414.1/414.5/414.5/414.5 across MIB 0/4096/8192/auto; real text 401.6 -> 1063.6) -- new gate step 5b `textprefill`, and `#273` carries the transport floor |
| P10 | — | (none — no delivery change; `scripts/gate-hc-fusions.sh` + docs) | #292: the dsv4 `DSV4_HC_PRE/_POST` soft arms were said to reproduce r235's numerics while the re-based chain moved | NOT DRIFT: the shipped r235 build reproduces r236's split exactly (hard `d62340fbf566`, soft `6a102dbc62d6` on both); the claim traced to a stale r156 header line. KLD vs chain measured 0.013217/96.392 % (pre) and 0.013198/96.186 % (post) at base PPL 2.8489 = the documented bf16 re-baseline. The soft arms are now sha-anchored (`SOFT_EXPECT`), so a real numeric change FAILs instead of printing `reported` | 2026-11-08 | no kernel change; `hyperconn.cu`/`dsv4-hc.cu` are correct |
| P11 | — | (none — no delivery change) | #293: keep or drop the delivery's own qwen4exp model file (3264 vs 1514 lines) | KEEP, measured: ours vs unpatched a691b2345 on Flash-Next GSQ 3 GPU -sm tensor — prefill +16.6/+28.8/+173.4 % at d0/16k/160k, decode +98.2/+98.9/+68.2 %, real-text prefill +19.1/+88.4 % at 16k/160k, batched total +41.6/+67.2/+53.7 % at 1/2/4 slots. The 27B 'divergence' was a category error (qwen35, not qwen4exp; its KLD vs upstream is 0.000697/98.784 %), and upstream's file does not compile against our tree (kpool/hc_mix API). qwen4exp-vs-upstream KLD 0.013426/96.431 % is the MoE/attention/HC re-expression, insensitive to every switch | 2026-11-08 | two instrument traps recorded: batched-bench npl=1 cold first pass, and r235's pool-order assert on perplexity |
| P12 | — | (delivery repo only: `scripts/gate-server-arms.sh`, `gate-rebase-release.sh` step 5c) | #295: the release ladder had no host-expert multi-slot DFlash arm, so R13's dropped guard has no detector | WIRED: three two-slot `-ncmoe` DFlash arms (Flash-Next GSQ + DFlash2 at -ncmoe 24 and 10, 35B-A3B Q8_0 + its DFlash2 at 10), baselines 0.74359/0.72269 and 0.59124/0.44720, bit-reproducible; HARD on both slots drafting, acceptance not collapsed (0.5x floor) and not drifted (0.10), text hashes, and a scoped fault check | 2026-11-08 | R13's own 0.53793/0.51678 etc. do not reproduce and its invocation was never recorded (the handover said RUNGS had it; it does not) -- recorded, not chased |
| P13 | — | (none — no delivery change) | #291 / D1: upstream's ClassLRU MoE cache vs the delivery's host-expert arena at equal MiB | DECIDED (maintainer): keep both, ours default-on, upstream's opt-in only. Real-text arms, text identical in every arm: -ncmoe 20 ours 1441.6/72.0 vs upstream ClassLRU 4096 782.3/53.3 (arena off 757.1/48.7, coexist 808.4/63.9, cpu-share 0.95 778.4/58.3); -ncmoe 40 ours 1394.2/69.2 vs 455.8/32.6. Coexistence costs 44 % prefill / 11 % decode, so either/or. D1's original +17.6 % inversion is stale (now ours 83.9 vs upstream 48.0 on the same llama-bench arm) and partly the #296 regime effect | 2026-11-08 | no code change; ClassLRU and --moe-cache-cpu-share not adopted |
| R17 | — | | | | | |
| P8 | — | `3b0978888` (fork, block 00/08: `mmf.cu`/`mmf.cuh` + `tests/test-backend-ops.cpp`) | #304: `MUL_MAT_ID` dup ids wrong for `f16` at `ncols_dst = 9` — the compact seam grid (16) and the kernel's `cols_per_block` (ncols_dst, clamped) disagreed, so an expert's compact range was truncated but only below 16 | FIXED + oracle enabled: dup=1 16/16 (was 15/16), MUL_MAT_ID 947/947, MUL_MAT 2517/2517 | 2026-11-08 | also closed #279 via the new perf A/B, and added `scripts/gate-rebase-release.sh` (one-command verdict, fails rather than skips) |
| P7 | — | (none — no delivery change) | #303: the dense `mul_mat_q` residue, filed as our R01/R08/R09 re-expression (VGPR 232 -> 216) | closed as INHERITED: r236's kernel ISA is identical to unpatched a691b2345 (1436 instrs, 0 diff); base commit = +1.31 %, r235's old body = -0.36 %, delivery = +0.09 %; two candidate fixes refuted at ISA level | 2026-11-08 | single-run census floor is +-1.1 % (two runs, same binary); 794 dense launches, 986 total; master was carrying a local #289 commit and is pinned at a691b2345 again |
| P6 | — | `upstream/recurrent-find-slot-rs-max/` (candidate) | #286/#287/#289/#290: two upstream candidates, one re-cut halved, one pre-check | #286 measured (5 arms), #289 `git apply --check` clean on a691b2345, #287 WIP, #290 pre-checked | 2026-11-08 | found `#304` (delivery is 15/16 on upstream's dup suite) |
| P5 | — | `2ff384571` (fork, block 05 hunk) | #302: the FA tile loader's per-chunk `kv_rows` select (r236's port) cost 25 % of the FA kernel | fixed (inlined fast path + `__noinline__` kv_rows variant); 30B/35B parity, KLD gate PASS | 2026-11-08 | scratch 376 -> 336 B, 19.6 -> 15.7 ms; residue is the dense `mul_mat_q` (#303) |
| P4 | — | `30277a117` (fork, block 00/08 hunk) | #301: MoE prefill −2..−3 % localised by kernel census → a duplicate ids compaction (our R01 re-expression) | fixed; 3 arms now −0.1..−1.1 %; 931/931 + texts identical; FA port split out as #302 | 2026-11-08 | one compaction + `mm_id_retile_kernel`; `GGML_CUDA_DISABLE_MMQ` correction recorded |
| P3 | — | `9b7dfb47a` (fork, block 10 hunk) | #298: one live pool-order violation in the cublas BF16 src1 cache path (plus 4 sites audited clean) | reproduced (F16 4B + `GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16`), fixed, inert (3 hashes + `gate-hc-fusions` PASS) | 2026-11-08 | the registered block is reserved before any plain pool allocation |
| P2 | — | `08d6c343b` (fork, block 11 hunk) | #297: the sparse arm had no gate (and no census to prove it ran) | three arms on the r236 build, incl. a below-the-width negative test that FAILs | 2026-11-08 | 23-line `GGML_CUDA_QSA_SCORE_LOG` census (inert, measured); gate section E |
| P1 | — | (delivery repo only) | #294 premise corrected: 8 gates had **no** journal check, 5 had a fixed window | `bash -n` ×17 + real PASS runs of `gate-grid-stall-pad` (A-only/B-only/`--no-faults`), `gate-qwen4exp-quant-coherence`, `gate-park-idle-slots` | 2026-11-08 | every GPU gate: `--since` = its own start timestamp, `--no-faults` opt-out, every log scanned, PASS line on success |
