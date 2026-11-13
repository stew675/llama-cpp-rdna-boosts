# The ladder — rungs, conflict sites, exit criteria and the ledger

One rung per delivery block, in block order (a later block's conflict is judged against the **rebased**
earlier blocks, so the order is not negotiable).  Block shas are the r277 chain
(`a691b2345..9d7c0502c`), the same commits `release.json` hashes.

How to work a rung: [`README.md`](README.md) §1.  Ledger columns: **status** `TODO` / `DOING` / `DONE` /
`DEFER`-`<decision>`; **sha** = the commit the rung produced on `rebase-r278`.

Legend for the conflict list: the paths the 2026-11-13 probe reported as conflicting for that block
(`CONFLICT-MAP.md`).  A path listed here is *expected* to conflict; a path that conflicts and is **not**
listed is a probe/cascade difference — note it in the rung log, do not assume the probe was wrong.

| rung | block | status | sha | decision points |
|---|---|---|---|---|
| R00 | setup | **DONE 2026-11-13** | — | — |
| R01 | 00 foundation | **DONE (merge+build+coherence); base fix DECIDED (opt 1)** | `43a418c10` | `D9`, **`D17`/B1** |
| R02 | 01 hybrid all-reduce | **DONE** | `ab433a7dd` | `D13` |
| R03 | 02 adaptive MTP | **DONE** | `50a0b30f8` | `D7`, `D8` |
| R04 | 03 chunked GDN | **DONE (both-sides, nothing dropped)** | `43891459e` | **`D2` resolved as complementary**, `D14` |
| R05 | 04 BF16 KV | **DONE** | `4c6a5019a` | `D3` checked |
| R06 | 05 flash attention | **DONE** | `22fcde21e` | **`D5` resolved as complementary**, `D3` closed |
| R07 | 06 MMB prefill | **DONE** | `c1d8b7c04` | `D5`'s few-row half deferred to R18 |
| R08 | 07 k-quant boosts | **NEXT** | — | `D4`, `D16` |
| R09 | 08 fused kernels | **TODO** | — | `D6`, `D12` |
| R10 | 09 host-resident MoE | **TODO** | — | **`D1`**, `D9`, `D15`, `D17`, `D18` |
| R11 | 10 infrastructure | **TODO** | — | `D6`, `D8`, `D10` |
| R12 | 11 qwen4exp | **TODO** | — | `D11` |
| R13 | 12 campaign remainder | **TODO** | — | `D10`, `D11`, `D15` |
| R14 | 13 MTP vocab subset | **TODO** | — | `D7` verify |
| R15 | 14 relaxed-tensor (**temporary**) | **TODO** | — | — |
| R16 | 15 streamed KV | **TODO** | — | `D3`, `D19` |
| R17 | 16 UMA GTT store | **TODO** | — | `D17` verify |
| R18 | release ladder | **TODO** | — | all `AWAITING` items resolved |

---

## R00 — setup · **DONE 2026-11-13**

* Campaign dir + `README.md` + `RUNGS.md` + `DECISIONS.md` + `CONFLICT-MAP.md` + `probe-conflicts.py`
  (+ `conflict-map.json`).
* New base facts: old `a691b2345` -> new **`4f9dfa0e3`**, 164 commits, 751 files, +40 612 / −24 226.
* `~/llama.cpp`: branch **`rebase-r278`** created at `4f9dfa0e3`; `master` untouched (pristine snapshot).
* `~/llama.cpp/build-rocm/bin-r277` — the r277 delivery binaries kept as the old-build reference
  (`build 12222, commit 9d7c0502c`).
* Probe: 17 blocks replayed with `git merge-tree` → 55 conflicted paths + 16 modify/delete entries; blocks
  01/04/13/14/16 clean.
* `wip/CAMPAIGNS.md` row + `WORKLOG.md` entry (campaign start; no delivery change).
* **R00 addendum (same day, found at R01):** the new base does not run with the delivery's required
  `GGML_HIP_NO_VMM=OFF` build.  Upstream `ac931b5bc` (`#30542`) makes the HIP VMM pool hang on ROCm
  10.1/gfx1201; a pure `master` build hangs too, and unpatched `master` with `GGML_HIP_NO_VMM=ON` works.
  Full reproduction, evidence and the four options are in [`BASE-BLOCKER.md`](BASE-BLOCKER.md); the register
  entry is `D17`.  A candidate fix (revert `ac931b5bc`, folded into block 00) is on `rebase-r278` and the
  rung below is green with it — it is `AWAITING` the maintainer's decision, not shipped.

---

## R01 — block 00: foundation (band uniformity, Meta wrapper, allocator) · **DONE 2026-11-13 (merge +
build + coherence); the folded base fix is DECIDED (`D17` option 1)**

* block sha `f15430d1e` · 34 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `src/llama-mmap.cpp` | `9d340ec34` (`#26014`) rewrote the file (+300, new `test-llama-mmap.cpp`) for a Windows-unbuffered load; `58e38bdf6` (`#31957`) narrowed the iGPU lazy fallback.  Our lazy/mlock hunks re-derive on the new load path — **`D9`**. |

**Note:** this block was *much* worse in the previous climb (12 paths, mostly `-Wshadow` rename ballast that
the r236 resolution dropped).  The current 34-file block is the resolved r236 shape, so expect a small,
mechanical merge.  Also verify the `GGML_COMPUTE_BUFFER_MARGIN_PCT` invariant still fires (upstream's Meta
compute-view floor is unchanged in this range).

**R01 result: `43a418c10`** (block 00 `f15430d1e` + `git revert ac931b5bc`; 36 files, +1049 / −357 against
the new base).  The only conflict was `src/llama-mmap.cpp`, and it resolved as an **upstream-wins rename**:
block 00 renames the `read_aligned_chunk` parameter `size` -> `size_l` (the `-Wshadow` pass) while upstream's
Windows-unbuffered rewrite independently renamed it `size` -> `len`; the bodies are otherwise identical, so
the resolution takes upstream's `len` in both platform variants and keeps the delivery's real content
(`llama_mlock::size()/release()/granularity()` + the `raw_lock` rename), which auto-merged.  No marker, no
`TODO(rebase)`.

**Exit:** build green with the canonical `GGML_HIP_NO_VMM=OFF` (36-file delta; only the pre-existing
`ld.lld` unroll notes); coherence on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`, seed 42 -> **100 chars
`sha=1c5d32ac537d`** (`Prompt 256.0 / Gen 92.1` t/s); `test-backend-ops -o MUL_MAT_ID` **1040/1040** on
gfx1201 (the duplicate-id fix reaches the new tile grid); `git grep '<<<<<<<'` empty.

**Base fix (DECIDED 2026-11-13, `D17` option 1):** the revert of `ac931b5bc` is folded into block 00 — it is
what makes the VMM=OFF build run at all on this box (without it the unpatched base hangs —
`BASE-BLOCKER.md`).  Dropped at the next re-base once upstream fixes the HIP pool.

## R02 — block 01: hybrid HIP all-reduce (RDNA4-gated) · **DONE 2026-11-13**

* block sha `d3470daa6` · 4 files · **expected conflicts: 0** (CLEAN)

**R02 result: `ab433a7dd`** — 4 files, +2900 / −49 (it was 2 365 / −44 at r277: the new base has more to merge
around).  Applied **clean** (the probe was right), no resolution owed.  Build green; 1-GPU `-sm layer`
coherence unchanged (**100 chars `sha=1c5d32ac537d`**); the delivery's hybrid arm is reachable and initializes on
the 3-GPU arm exactly as it did at r277: `ggml_cuda_ar_pipeline_init: initialized HIP AllReduce pipeline: 3 GPUs,
1024 KB chunked kernel staging + 32 MB copy-engine staging per GPU` (`HIP_VISIBLE_DEVICES=0,1,2 -sm tensor`,
`-lv 4`), and the same 3-GPU arm's clean text is also `sha=1c5d32ac537d` (94.7 / 104.9 t/s).  **`D13`** checked:
`432433637` (fixed BLAS workspace) and the `#30542` revert in block 00 do not disturb the dispatch or the
all-reduce's scratch allocation.

## R03 — block 02: adaptive MTP draft depth · **DONE 2026-11-13**

* block sha `e16b66272` · 23 files · **expected conflicts: 3**

| conflicted path | what to expect |
|---|---|
| `common/speculative.cpp` | `a8fad21e9` (`#31927`) adds the MTP target-hidden-state device buffer + draft plumbing; `7874988b4` (`#31294`) adds a `--spec-draft-p-min` warning.  **`D7`**. |
| `src/llama-context.cpp` | the same `#31927` (+199 lines) — the device buffer's context surface meets the delivery's draft-depth control. |
| `src/llama-ext.h` | `8ca5a8008` (`#30565`) prunes the file (−64); the delivery adds to it.  Re-add *only* the delivery's declarations.  **`D8`**. |

**R03 result: `50a0b30f8`** — 23 files, +1391 / −65 (was +1186 / −54 at r277: upstream's `#31927` device-row
work merges in).  Build green; 1-GPU coherence unchanged (`sha=1c5d32ac537d`); `test-speculative-adaptive` all OK.

**Resolutions (all "both sides", no delivery code dropped):**
| site | resolution |
|---|---|
| `common/speculative.cpp` (members + ctor) | upstream's device-row members (`dev_rows`, `n_batch_tgt`, `verify_row0`) **and** the delivery's adaptive members (`adaptive`, `n_cap`, `n_last`, `adaptive_ctrl`, `align`, `band`, `n_hard`) coexist; the ctor keeps the delivery's extra `(adaptive, align)` parameters and its initializer, and upstream's body init (`n_batch_tgt`, `verify_row0`, `dev_rows = llama_set_embeddings_nextn_dev(...) && llama_set_ctx_state_src(...)`) is intact below it. |
| `src/llama-ext.h` | the prune moved `llama_set_embeddings_nextn` and `llama_set_nextn_layer_offset` to the public `include/llama.h` (**`D8`**), so their delivery declarations are **not** re-added; only the delivery-only `llama_set_embeddings_nextn_from` stays.  The same applies to `llama_model_dflash_selector_top_k` / `llama_model_target_layer_ids(_n)` (upstream now has public versions); `llama_model_set_embeddings_layer_inp_preset` is still delivery-only and stays. |
| `common/speculative.cpp` (includes) | **`D8` again, silent this time:** the prune deleted `#include "../src/llama-ext.h"` from this file (its comment named the two now-public calls), so our block's include hunk had nothing to attach to and the build failed on `llama_set_embeddings_nextn_from`.  Re-added with the comment narrowed to the `_from` call.  *A prune that removes an include is a build break, not a merge conflict — check the include surface after every prune rung.* |
| `src/llama-context.cpp` (nextn readback) | **the real `D7` merge**: upstream skips the host extraction when the device buffer is active (`!write_h_nextn`); the delivery skips it when every row is below the sequence's `llama_set_embeddings_nextn_from` threshold (`nextn_unneeded`, masked/vision rows always read).  Resolved as `if (nextn_rows && !write_h_nextn) { if (nextn_unneeded) { bookkeeping only } else { readback } }` — upstream's gate outer, the delivery's skip inner. |

**`D7` measurement (the device buffer vs our host path), 35B-A3B Q8_0, bf16 KV, `draft-mtp-adaptive`, `-n 200`:**
* `-sm tensor` (ROCm0,ROCm2, the field layout): `adaptive draft depth enabled`, acceptance **0.89130**
  (82/92), mean len 2.46 — but upstream's device path **declines**: `h_nextn_dev_alloc: the device has no
  events, extracting the h_nextn rows to host memory`.  The output device under `-sm tensor` is the Meta
  device and `ggml-backend-meta.cpp` sets `iface.event_new = nullptr`, so `ggml_backend_event_new()` returns
  null.  The fallback is clean (host rows, our `_from` skip still applies).
* `-sm layer` (ROCm0,ROCm2): `target h rows on the device: yes`, acceptance **1.00000** (19/19), mean len 2.0.
  So the device transport activates only on a layer split today.
* **Conclusion for the rung:** both paths coexist; ours is the only reachable one on the field's `-sm tensor`
  layout, upstream's is reachable on `-sm layer`.  No decode measurement is owed yet: the device path is
  inert where the delivery is validated.  **Open upstream gap recorded for `TODO`**: give the Meta device an
  `event_new` (or fall back to the output device's own backend) so `#31927` is reachable under `-sm tensor`;
  that is an upstream/ candidate, not a delivery change.  MTP four-axis stays R18's job.

## R04 — block 03: chunked gated-delta-net prefill · **DONE 2026-11-13 — `D2` needed no stop**

* block sha `dd81963db` · 25 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `src/models/delta-net-base.cpp` | **`ccd360f06` (`#32078`) removed the non-fused GDN and every ubatch-dependent graph node** (−405 lines here, plus `clef`/`clover-moe`/`kimi-linear`/`qwen35`/`qwen35moe`/`qwen4exp`/`models.h`/`llama-context.*`/`llama-cparams.h`/`llama-graph.h`).  **`D2`**. |

**R04 result: `43891459e`** — 25 files, +3363 / −31.  Build green; 1-GPU coherence unchanged
(`sha=1c5d32ac537d`); the delivery's chunked kernel is reached **after** upstream's removal:
`-lv 4` on the 4B with a 5 298-token prompt gives
`ggml_cuda_op_gated_delta_net_impl: whole-batch chunked GDN prefill n=2048 K=1 n_seqs=1 (> 16: no snapshot
tail needed)` and `n_rs_batch = 1` in the cparams banner, prefill **6908.1 t/s** on one GPU.

**`D2` outcome — upstream did NOT subsume the delivery, and nothing is dropped.**  The two sides are
complementary: upstream `#32078` deleted **its own** non-fused graph builders
(`build_delta_net_chunking`, `build_delta_net_autoregressive`, the `build_delta_net()` chooser) and the GDN
fused-node tags and registry entries — all of which existed in the **old base** (`a691b2345`) and were
therefore upstream code the delivery merely sat beside.  The delivery's substance is elsewhere: an extra
`n_rs_batch` argument on `ggml_gated_delta_net()` (so a chunked kernel can skip rollback snapshots for a
batch that can never be rolled back into), the pre-batch snapshot copy in `build_recurrent_attn()`, and the
`gated_delta_net_chunked*.cu` kernels.  Upstream now *always* routes through that same fused op, so the
delivery's kernels are the surviving implementation of the path upstream simplified to.

| conflicted hunk | resolution |
|---|---|
| `build_delta_net_fused` call | upstream's tag deletion is taken (the `LLM_FUSED_OP_GDN_AR/CH` registrations are **not** re-added — upstream removed the enum values and their registry entries, so re-adding them would not compile); the delivery's `/*n_rs_batch=*/1` argument stays. |
| `build_recurrent_attn` call | same: `ggml_gated_delta_net(..., K, cparams.n_rs_batch)` without the tags.  The delivery's pre-batch snapshot copy (the non-conflicting hunk above it) stays. |

**A note on the rollback test:** `test-recurrent-state-depth` reports 83 failures (Phase A numeric
mismatches, max diff ~0.23) on this box **and 73 on the shipped r277 build** with the same command — so it
is not a R04 regression and it is not caused by the merge; record it as a pre-existing observation.  The
r236 campaign recorded `PASS 0` on its own config, so the gate is config-sensitive; the command and both
counts are here so it can be re-derived (`TODO`-shaped, not a blocker).

**`D14` check:** `a8575a3ef` (SpacemiT GATED_DELTA_NET, CPU) and `998200491` (SSM_SCAN d_state=64) are in the
merged tree; neither is reachable on gfx1201.  Kernel-side consistency only.

## R05 — block 04: BF16 KV cache and native-BF16 flash attention · **TODO**

* block sha `7708d67ff` · 8 files · **expected conflicts: 0** (CLEAN)

**R05 result: `4c6a5019a`** — 8 files, +602 / −191, applied **clean** (the probe was right).

**`D3` audit (KV type set vs upstream's FP8):** the merged `-ctk/-ctv` set is
`f32 f16 bf16 f8_e4m3 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1` — upstream's `f8_e4m3` sits beside the delivery's
`bf16`, and neither the FA instance list nor a `supports_op` predicate aborts.  Measured on `Qwen3.5-4B Q8_0`
(1 GPU, `-sm layer`, `-c 4096`, same seed): `bf16`, `f8_e4m3` and `q8_0` all run and produce the same
**100 chars `sha=1c5d32ac537d`**.  The delivery's BF16 FA instance is compiled in
(`GGML_CUDA_FA_QUANTS` carries `bf16-bf16`), so the arm reaches the delivery path, not a fallback.  The BF16
KV decode A/B stays R18's job (it is a release number, not a rung cost).

## R06 — block 05: flash-attention band, native quantized K/V and derived kq mask · **DONE 2026-11-13**

* block sha `a45c508e9` · 295 files · **expected conflicts: 8 + 16 modify/delete**

| conflicted path | what to expect |
|---|---|
| `ggml/CMakeLists.txt`, `ggml/cmake/common.cmake` | the FA instance generator/build grouping moved (`cd4ed902e`, `9d7b2d739`, `77e2db8ca`). |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | new head sizes 40/72 and the sm70 256 case. |
| `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` | the matrix generator was rewritten; our instance list must be re-expressed in it, not appended by hand. |
| `src/llama-context.cpp`, `src/llama-cparams.h`, `src/llama-graph.h` | the kq-mask/FA-row plumbing meets `#31927`/`#32078`. |
| `tests/test-backend-ops.cpp` | `03d6c824e`/`1648d6411` moved the tests to the public API; our FA oracle entries move with it. |
| 16 × `template-instances/fattn-mma-f16-instance-*.cu` | **modify/delete**: upstream regenerated the ncols matrix; take upstream's set and re-add only the delivery-specific instances. |

**Expect:** **`D5`** — re-derive the delivery's FA band selector over the new instance matrix (including
the FP8 vec FA), keeping our band as the preferred path where it wins and upstream's kernel reachable.
`iq4_nl` instances are set by hand on this box (`GGML_CUDA_FA_QUANTS`) — re-set them when this rung lands.

**R06 result: `22fcde21e`** — 295 files, +5641 / −659 (232 instance files added by the block, 16 old ones
deleted).  Build green; coherence unchanged (`sha=1c5d32ac537d`, also with `-ctk/-ctv q8_0`); the derived
kq-mask probe resolves at load (`resolve_fused_ops: derived kq mask flash attention enabled`).
**`test-backend-ops -o FLASH_ATTN_EXT`: 6996 OK / 0 FAIL**, including 18 `derived=1` cases (the delivery's V3
mask), 385 `iq4_nl` (native quantized K/V), 7 `f8_e4m3` (upstream's FP8 vec FA), and **448 at head 40 +
1010 at head 72** (upstream's new MMA head sizes) — so the merged instance matrix covers both sides.

**`D5` outcome — complementary, nothing dropped.**  Upstream's FA changes (`#30285` heads 40/72, `#27997`
sm70 case, `#32018` unused params, `#28413`/`#28392` FP8) are additive to the delivery's band; the delivery's
band is a *selector* over the same kernel matrix, so both ship.

| site | resolution |
|---|---|
| `TYPES_KV` (generator), `FA_TYPES` (common.cmake), `GGML_CUDA_FA_QUANTS` default (ggml/CMakeLists.txt) | **union**: the delivery's list gains upstream's `f8_e4m3` (`...;iq4_nl-iq4_nl;f8_e4m3-f8_e4m3`), and `FA_TYPES` gains it at the end.  The box's cache now carries the union (the r277-era hand-set `iq4_nl-iq4_nl` trap is resolved by this rung, as the campaign predicted). |
| 16 × `template-instances/fattn-mma-f16-instance-ncols1_*-ncols2_*.cu` (**UD**) | **accept block 05's deletion.**  The delivery replaced the head-agnostic files with 121 typed `...-dkqN-dvN.cu` instances; the typed set already declares heads 40 and 72 (`ls | grep -c dkq40/dkq72` = 16), so upstream's additions are covered rather than lost — proven by the oracle above. |
| `fattn-mma-f16.cuh` (8 conflicts) | take the delivery's band-extended functions, **and re-add upstream's `DKQp`/`DVp = GGML_PAD(..,16)`**: upstream added the padded head width for head 72, and the delivery's inherited loops/Q-tile stride/VKQ_C sizing need it.  `stride_tile_Q = DKQp/2 + 4` in both functions; `VKQ_C` keeps the delivery's fp32/fp16 distinction (GREEDY-PURITY §46) but sizes on `DVp`, which is the bound the upstream rescale/combine loops iterate. |
| `src/llama-graph.h` | the delivery's `LLM_FUSED_OP_FLASH_ATTN_DERIVED`, `llm_graph_kq_derived`, `llm_graph_moe_cache_slots`; **not** `LLM_FUSED_OP_GDN_AR/CH` (upstream removed them at R04). |
| `src/llama-cparams.h`, `src/llama-context.cpp` | the delivery's derived/packed kq-mask config and reachability latch; **removed the now-dead `fused_gdn_ar`/`fused_gdn_ch`/`auto_fgdn`** (their consumers were the GDN fused-node tags upstream deleted — D2). |
| `tests/test-backend-ops.cpp` | union: upstream's `q_head_padding` + the delivery's `derived`/`derived_hole`/`mask_hole`.  `q_head_padding` goes **last** in the ctor (no upstream call site passes it, checked), so the delivery's positional test cases keep their meaning; `vars()` keeps upstream's `VARS_TO_STR18` + the three flags. |
| block-05 call sites vs upstream's context refactor | `graph_max_nodes(n_tokens)` -> **`graph_max_nodes()`** (upstream `#32078` removed the ubatch dependence; `gate-node-budget.sh` is owed at R11/R18), and `sched_new(max_nodes, false)` -> upstream's **`sched_new({.max_nodes, .parallel, .op_offload})`** struct form (`#32009`).  `graph_reserve(..., packed_kq_mask)` merged as-is (the header already carried both sides' parameters). |

## R07 — block 06: MMB quantized prefill GEMM · **DONE 2026-11-13**

* block sha `7f17e7534` · 11 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `0fb8f1577` (`#31533`) rewrote the MMQ/MMVQ dispatch and added `mmq-few.cu`; `0d69255c5` (`#26048`) added the nvfp4 w_s epilogue. |

**R07 result: `c1d8b7c04`** — 11 files, +3000 / −16.  Build green; coherence unchanged
(`sha=1c5d32ac537d`); the MMB path is **on by default and reached**: with `GGML_CUDA_MMB_LOG=1` a 5 298-token
prefill (`-b 2048 -ub 2048`) prints
`MMB_CFG device=0 cc=0x1001201 dense_geom=1 min_t=512 glu_thresh=32 ... glu=1 bf16w=1` and runs at
**6915.2 t/s** prefill.

| conflicted hunk | resolution |
|---|---|
| `ggml_cuda_mul_mat` MMVF branch | the delivery's two additive blocks (MMB dispatch, and the `ne11_mmvf` verify-batch pin) are kept, but the predicate call is re-expressed on **upstream's refactored signature** `ggml_cuda_should_use_mmvf(src0, cc, warp_size, ne11)` (`#29633`/`#31959` made it take the tensor and derive type/ne/nb inside).  Upstream's own call had the same 4-arg shape, so the merge is "our blocks + their predicate". |
| fused MoE gate+up dispatch | the delivery's MMB gate/up/swiglu block is kept; the shared predicate becomes upstream's `ggml_cuda_should_fuse_mul_mat_vec_f(up, gate)` (upstream added the `gate` argument; the merged declaration defaults it to `nullptr`, so both call shapes compile — the 2-arg form is used). |
| `// A transposed vector can still use MMVQ` | the delivery side had glued this comment onto the closing brace and renamed MMVF->MMVQ; restored upstream's wording (the following code is the MMVF transposed path). |

**`D5`'s few-row half:** upstream's `mmq-few.cu` (`#31533`) is in the tree and the delivery's MMB predicate is
prefill-only (`T >= 512`), so the two do not compete for the same shapes in the merged dispatch.  The
delivery-vs-upstream few-row *measurement* (BS 2-24) is a release A/B, so it is deferred to R18's
`llama-bench` arms rather than decided at this rung.  **Prefill-logit gate for this block is owed at R18**
(the rung's base record does not exist yet).

## R08 — block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad) · **TODO**

* block sha `42c96a0ea` · 9 files · **expected conflicts: 2**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `0fb8f1577` few-row MMQ, `494d0f47b` src1-precision gating, `5d4c8eeb2` MoE tiles. |
| `tests/test-backend-ops.cpp` | `494d0f47b`/`03d6c824e` reworked the MUL_MAT oracle. |

**Expect:** **`D4`** (MMVF BF16 few-row — measure against the delivery dispatch) and **`D16`** (the
src1-precision gate must reach our path, not a fallback).

## R09 — block 08: fused kernels · **TODO**

* block sha `7f1ed0723` · 38 files · **expected conflicts: 7**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/common.cuh` | `17701e48d` (`#32001`) moved graph + concurrent-stream state out to `graph.cuh`; our fusion hooks follow it.  **`D6`**. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the fusion dispatch meets the graph move + cluster-dim key (`2e325dcd5`). |
| `ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu`, `quantize.cu(h)` | `0fb8f1577` + `5d4c8eeb2` + `0d69255c5`. |
| `ggml/src/ggml-cuda/norm.cu` | `31953bcf7` (`#32135`) rewrote BF16 RMS_NORM (−180/+180).  **`D12`**. |

**Exit:** `gate-norm-rows-alias.sh` on the merged tree; the fusion-alias gate if the out-node set moved.

## R10 — block 09: host-resident MoE (iface, scheduler, Meta, slab, cache engine) · **TODO**

* block sha `495d00d45` · 69 files · **expected conflicts: 8**

| conflicted path | what to expect |
|---|---|
| `common/fit.cpp` | `5f490d7bf` (`#31185`) external PLE + the `fit.*` reshuffle.  **`D15`**. |
| `common/speculative.cpp` | `a8fad21e9`, `7874988b4`.  **`D7`**. |
| `ggml/src/ggml-backend.cpp` | `806b3c2be` (`#31419`) upload reachability, `3bce8bd50` (`#32100`) prefetch.  **`D10`**. |
| `ggml/src/ggml-cuda/common.cuh` | `17701e48d`, `432433637`.  **`D6`**. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the VMM pool (`ac931b5bc`) + graph move + MoE tile fix.  **`D17`**. |
| `src/llama-context.cpp`, `src/llama-context.h` | `951d465f5` (`#32157`) moved the MoE cache out of `llama-graph`; `a8fad21e9`.  **`D1`**. |
| `src/llama-mmap.cpp` | the new load path.  **`D9`**. |

**Expect:** port the delivery's cache hook to the new `llama-moe-cache` seam, keep the arena default-on and
upstream's cache dormant (the r236 either/or measurement stands).  **`D18`** is consistency only: do not let
the SYCL scheduler flags change the CUDA predicate.

## R11 — block 10: slab, fusion dispatcher and async input ring · **TODO**

* block sha `05f298fea` · 86 files · **expected conflicts: 10** (the largest cluster)

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/common.cuh` | `17701e48d` graph/concurrent state move.  **`D6`**. |
| `ggml/src/ggml-cuda/cpy.cu`, `mmvf.cu`, `mmvq.cu`, `norm.cu` | `0fb8f1577`, `494d0f47b`, `31953bcf7`, `432433637`. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | graph move + cluster dims + few-row dispatch. |
| `src/llama-context.cpp` | `cb44fdb18` (`#32089`) graph-result reuse, `4fffab623` (`#32043`) metadata pages, `a8fad21e9`.  **`D10`**. |
| `src/llama-ext.h` | the prune (−64) vs the delivery's additions.  **`D8`**. |
| `src/llama-graph.cpp` | `951d465f5` MoE-cache move, `cb44fdb18`, `4fffab623`. |
| `tests/test-backend-ops.cpp` | the public-API test refactor. |

**Exit:** `gate-node-budget.sh` (the async ring adds graph nodes), `gate-grid-stall-pad.sh` if the pad site
moved.

## R12 — block 11: qwen4exp (model, hyper-connections, lazy reader, QSA/indexer) · **TODO**

* block sha `ea4963486` · 51 files · **expected conflicts: 5**

| conflicted path | what to expect |
|---|---|
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c` | RPC surface + op table (`0eebb6edd` TRI backward, `57d04820d` Top-K) + the prune.  **`D11`**. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the op dispatch + the new graph/stream state. |
| `src/models/qwen4exp.cpp` | `ccd360f06` removed GDN/ubatch-dependent nodes from this model.  **`D2`**-adjacent. |
| `tests/test-backend-ops.cpp` | the public-API move. |

**Exit:** `gate-hc-fusions.sh` on the qwen4exp model, `gate-qwen4exp-quant-coherence.sh`.

## R13 — block 12: campaign remainder (CPU wins, backend iface fills, DFlash, FA build grouping, server prompt cache) · **TODO**

* block sha `c7628e974` · 62 files · **expected conflicts: 4**

| conflicted path | what to expect |
|---|---|
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c` | `D11` + `D8`. |
| `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` | the FA generator (FA build grouping). |
| `src/llama-context.cpp` | `a8fad21e9`, `cb44fdb18`.  **`D10`**. |

## R14 — block 13: MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`) · **TODO**

* block sha `f0c7f3359` · 19 files · **expected conflicts: 0** (CLEAN)

**Expect:** clean, but upstream's MTP device-buffer path (`D7`) changes what a draft sees — verify the
vocab-subset filter still applies to the drafted tokens.

## R15 — block 14: `-sm relaxed-tensor` (**TEMPORARY**) · **TODO**

* block sha `6b4d9865e` · 11 files · **expected conflicts: 0** (CLEAN)

**Expect:** clean.  Re-check upstream has not started allowing the arches the relaxation covers; if it has,
this block is dropped by decision, not by merge.

## R16 — block 15: the streamed KV — window (Goal A) + attendance pool (Goal B) + MTP draft window · **TODO**

* block sha `99d33117a` · 46 files · **expected conflicts: 5**

| conflicted path | what to expect |
|---|---|
| `common/fit.cpp` | the PLE/fit reshuffle.  **`D15`**. |
| `common/speculative.cpp` | `a8fad21e9` device-buffer MTP.  **`D7`**. |
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c` | `D11`. |
| `src/llama-kv-cache.cpp` | `202c1a229` (`#28392`) added the FP8 KV type to every instance/predicate table.  **`D3`**. |

**Exit:** `gate-kv-residency.sh`, `gate-kv-window-split.sh`, `gate-spec-draft-window.sh`,
`gate-spec-reuse-window.sh`.  `D19` (`940a30ee3`) is a verify-only item here.

## R17 — block 16: the UMA GTT expert store · **TODO**

* block sha `9d7c0502c` · 16 files · **expected conflicts: 0** (CLEAN)

**Expect:** clean.  Verify **`D17`**: the halved VMM reservation + retry must not change the region the
GTT store asks for (`-lme` on `halo`/`soar`).  Gate: `gate-uma-experts.sh --quick`.

## R18 — release ladder · **TODO**

See [`README.md`](README.md) §4.  Not a rung: `make-patches.sh`, `make-release.sh`, `validate-set.sh`,
`gate-per-block-build.sh`, `gate-prefill-logits.sh --record` from a clean `4f9dfa0e3` build,
`gate-rebase-release.sh`, the block gates, `check-gpu-faults.sh`, fork refresh, containers, records, and
this campaign moved to `archive/work/`.

**Every `DECISIONS.md` item marked `AWAITING` must be `DECIDED` before R18 starts.**
