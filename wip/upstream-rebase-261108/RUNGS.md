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

## R08 — block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad)

* block sha `98bb7bc61` · 9 files · +879 / −137 · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/mmvq.cu` | #31679 (`get_mmvq_mmid_max_batch` arch table), #31247 (Q8_1 correction sum), the per-shape `nwarps` work |

**Exit:** build + coherence + `test-backend-ops` on the touched quant types (the decode-numerics block).

- status: **TODO** · sha: —

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

- status: **DONE** · sha: `af0cae2f1`

## R10 — block 09: host-resident MoE

* block sha `11c5544be` · 68 files · +16121 / −540 · **expected conflicts: 7**

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

- status: **TODO** · sha: —

## R11 — block 10: slab, fusion dispatcher, async input ring

* block sha `7342cd95a` · 84 files · +15813 / −5461 · **expected conflicts: 8**

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

- status: **TODO** · sha: —

## R12 — block 11: qwen4exp

* block sha `0e1032213` · 46 files · +12091 / −433 · **expected conflicts: 7**

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

- status: **TODO** · sha: —

## R13 — block 12: campaign remainder (CPU wins, iface fills, DFlash, server)

* block sha `be4749a56` · 81 files · +5893 / −995 · **expected conflicts: 8**

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
`scripts/gate-state-pipe.sh` (whichever of the three still has its code after the merge).

- status: **TODO** · sha: —

## R14 — block 13: MTP draft vocabulary subset

* block sha `6ee8eb290` · 19 files · +569 / −6 · **expected conflicts: 3**

| conflicted path | what to expect |
|---|---|
| `src/llama-graph.h` | `D11`/`D9` — the graph context's new `model` member (#31733) |
| `src/llama-model.cpp`, `src/llama-model.h` | #31141/#31142 loader refactors; the vocab-subset state hangs off the model |

**Exit:** build + coherence + `--spec-draft-mtp-vocab N` runs and is inert when unset (default off).

- status: **TODO** · sha: —

## R15 — block 14: `-sm relaxed-tensor` (TEMPORARY)

* block sha `59a3e5206` · 11 files · +130 / −16 · **expected conflicts: 0**

Checked 2026-11-08: upstream's `#31147` only adds `LLM_ARCH_LAGUNA` to the **deny** list
(`llm_arch_supports_sm_tensor`), so the amax arches are still refused under `-sm tensor` and block 14 is not
subsumed.  Our block only changes the error text for `flint`/`mellum`/`mistral4` — if upstream grew a
relaxed mode in the meantime, re-check `src/llama-arch.cpp` and `llama_model_create()` before resolving.

**Exit:** build + coherence + `-sm relaxed-tensor` loads a flint/mellum/mistral4 MoE and matches `-sm layer`
text (the block's own claim).

- status: **TODO** · sha: —

## R16 — block 15: the streamed KV (window + attendance pool)

* block sha `69d36099b` · 33 files · +5294 / −56 · **expected conflicts: 9**

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

- status: **TODO** · sha: —

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
| R08 | 07 | | | | | |
| R09 | 08 | `af0cae2f1` | 5 expected → 5 files / 8 hunks | build + MUL_MAT_ID 931/931 + fused/chain byte-identical | 2026-11-08 | `D2` complementary (two template arms); `mmq_args` carries `expert_bounds` next to `tile_spec`; norm-rows gate moved to R11 |
| R10 | 09 | | | | | |
| R11 | 10 | | | | | |
| R12 | 11 | | | | | |
| R13 | 12 | | | | | |
| R14 | 13 | | | | | |
| R15 | 14 | | | | | |
| R16 | 15 | | | | | |
| R17 | — | | | | | |
