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

## R01 — block 00: foundation (band uniformity, Meta wrapper, allocator)

* block sha `a900d4703` · 56 files · +1366 / −909 · **expected conflicts: 12**

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

- status: **TODO** · sha: —

## R02 — block 01: hybrid HIP all-reduce

* block sha `df940c14c` · 4 files · +2365 / −44 · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | #28967 (NCCL multi-node TP) and the `ggml_cuda_allreduce_*` neighbourhood; our hybrid dispatch is a sibling of `nccl`/`internal`. **`D10`** |

**Exit:** build + coherence + the all-reduce arm is still reachable
(`GGML_CUDA_ALLREDUCE=hybrid` on `-sm tensor --device 0,1,2`, `-lv 4` census line).

- status: **TODO** · sha: —

## R03 — block 02: adaptive MTP draft depth

* block sha `470729be7` · 24 files · +1186 / −54 · **expected conflicts: 4**

| conflicted path | what to expect |
|---|---|
| `common/common.cpp` | arg plumbing; upstream added many flags (`--moe-cache-*`, realtime, router margin) |
| `common/speculative.cpp` | `D11`: #30141 (MTP hidden state save/restore), #31350, #31497 (no MoE cache on the draft) |
| `src/models/delta-net-base.cpp` | #30878 conv-state refactor — the adaptive controller's snapshots ride on this |
| `tests/CMakeLists.txt` | additive |

**Exit:** build + coherence + the adaptive controller still selects > 1 draft depth (`-lv 4` accept line) on
the MTP config; MTP four-axis is R17's job, not a per-rung cost — a single `-n 300` smoke is enough here.

- status: **TODO** · sha: —

## R04 — block 03: chunked gated-delta-net prefill

* block sha `670aa557d` · 25 files · +3182 / −15 · **expected conflicts: 1**

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

- status: **TODO** · sha: —

## R05 — block 04: BF16 KV cache and native-BF16 flash attention

* block sha `4db2b86d9` · 8 files · +630 / −214 · **expected conflicts: 0**

`D6`: upstream `162e2bb2e ggml : fix bf16` touches `ggml/src/ggml.c` + `tools/export-lora` — check it does
not invalidate a delivery bf16-KV assumption (it is a type-traits/lora fix, so the expectation is "no"), and
that `-ctk bf16 -ctv bf16` still runs the native FA arm.

**Exit:** build + coherence + a bf16-KV arm generates (`-ctk bf16 -ctv bf16`, native FA line at `-lv 4`).

- status: **TODO** · sha: —

## R06 — block 05: flash-attention band, native quantized K/V, derived kq mask

* block sha `41248b885` · 295 files · +5565 / −624 · **expected conflicts: 9**

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

- status: **TODO** · sha: —

## R07 — block 06: MMB quantized prefill GEMM

* block sha `398fe6ab3` · 7 files · +2903 / −3 · **expected conflicts: 0**

`#31047` (linearised MoE layout) and `#31296` (NVFP4) are in the same files but did not conflict in the
probe — confirm the MMB dispatcher keys still compile and the RDNA arm is still selected.

**Exit:** build + coherence + MMB still dispatched (`-lv 4` line) on a Q8_0 prefill.

- status: **TODO** · sha: —

## R08 — block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad)

* block sha `98bb7bc61` · 9 files · +879 / −137 · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/mmvq.cu` | #31679 (`get_mmvq_mmid_max_batch` arch table), #31247 (Q8_1 correction sum), the per-shape `nwarps` work |

**Exit:** build + coherence + `test-backend-ops` on the touched quant types (the decode-numerics block).

- status: **TODO** · sha: —

## R09 — block 08: fused kernels

* block sha `615e77323` · 35 files · +5500 / −440 · **expected conflicts: 5**

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

- status: **TODO** · sha: —

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
| R01 | 00 | | | | | |
| R02 | 01 | | | | | |
| R03 | 02 | | | | | |
| R04 | 03 | | | | | |
| R05 | 04 | | | | | |
| R06 | 05 | | | | | |
| R07 | 06 | | | | | |
| R08 | 07 | | | | | |
| R09 | 08 | | | | | |
| R10 | 09 | | | | | |
| R11 | 10 | | | | | |
| R12 | 11 | | | | | |
| R13 | 12 | | | | | |
| R14 | 13 | | | | | |
| R15 | 14 | | | | | |
| R16 | 15 | | | | | |
| R17 | — | | | | | |
