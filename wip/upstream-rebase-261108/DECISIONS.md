# Decision register — where upstream and the delivery answer the same question

Each item is a **stop-and-ask** point: the rung that reaches it does not pick a winner, it measures and
presents.  `AGENTS.md` → "Report findings before shipping" and the WIP rules apply: nothing here changes
`patches/` until the maintainer decides, and a dropped delivery mechanism is recorded (a `WORKLOG.md` entry,
`TODO.md` item, or an `archive/work/` record) in the same change.

**Status vocabulary:** `OPEN` (not reached yet) · `MEASURING` (rung reached, numbers being taken) ·
`AWAITING` (numbers in hand, maintainer's call pending) · `DECIDED` (resolution + record).

**The delivery's standing principle** (from `AGENTS.md` and the maintainer, 2026-11-08): the set aims to be
*complementary* — a second preferred path, not a replacement — so most of these should resolve to "both
paths ship, ours stays the default where it is faster, upstream's stays reachable".  A drop only happens
where the two cannot coexist (one implementation of the same function, or upstream's is strictly better).

---

## D1 — The MoE cache: upstream's cache grew a policy and a CPU share; ours is a different tier

**Upstream** (same subsystem, `src/llama-moe-cache.cpp`): `#29887`/`#30112`/`#30637` gave it a VRAM cache of
hot host-resident experts with a 50/50 SLRU policy over multiple GPUs; this range adds `#31233` (**ClassLRU**,
an LRFU/MQ-like hybrid with a half-life) and `#30966` (**`--moe-cache-cpu-share F`** — compute a share of the
misses *on the CPU* from the host experts instead of uploading them, plus `scripts/moe-cache-tune.py`), and
`#31497` (do not apply the cache to the draft model).

**Delivery**: blocks 09/12 add the host-resident **arena/pool tier** on a separate path
(`ggml/src/ggml-cuda/moe-expert-cache.{cu,h}`, `ggml-cuda-slab.cu`, the scheduler copy hook, the Meta
per-device forwarding, `MOE_HOST_POOL_*`), and it explicitly keeps upstream's cache as a *second, dormant*
path — `src/llama-context.cpp` carries `// upstream MoE cache (dormant unless --moe-cache-mib)`.

**Relation:** complementary by design, but they share the scheduler copy callback and the context's
allocation of "the device cache", so the merge is the work.

**Questions:**
1. Keep coexistence as-is (upstream cache off by default, arena on)? — *expected: yes.*
2. Adopt upstream's **CPU-share** idea for the pool tier?  It is a genuinely new mechanism (our fill uploads;
   upstream computes), and it targets exactly the configs where our pool declines/uploads at low residency.
   Upstream's own doc says the share "often does not help" when the host experts are pinned — which is our
   mirror tier — so the honest test is the *unpinned* (`-lm mmap`) pool shape.
3. Adopt **ClassLRU** into the arena's admission?  Our admission is heat/RFU-based and validated; ClassLRU is
   reachable today only for the dormant cache, so this is a "port a policy" change, not a merge.

**Measurement (one session, 1 × R9700, Qwen3.6-35B-A3B Q8_0, `-cmoe`/`-ncmoe 40`-shaped placement):**
`prefill t/s`, `tg128`, the cache census lines (`-lv 4`), and text parity, across
(a) delivery arena+pool default, (b) upstream `--moe-cache-mib <same MiB as the arena>`, (c) upstream
`--moe-cache-mib + --moe-cache-cpu-share 0.5/0.95`, (d) the pool at `-lm mmap` with and without a
cpu-share-style arm.  Report the three questions together.

- status: **OPEN** · rung R10 · record: —

## D2 — Fused MoE gate/up+GLU in MMQ (and top-k MoE)

**Upstream:** `#31188` fuses `MUL_MAT_ID` gate/up + GLU in MMQ, `#31288` adds a top-k MoE fusion for 384
experts, `#31296` fuses NVFP4 gate/up + GLU, `#31047` linearises the MoE data layout for MMQ/MMF, `#31005`
sanitises post-bias NaNs in MoE selection.
**Delivery:** block 08's fused gate+up+GLU on the RDNA path (block 08 also owns the per-shape `mmvq` `nwarps`
and the conv/RMS/top-k fusions).

**Relation:** expected **arch-disjoint** — the same precedent as `#29948` (the 2026-10-25 re-base's NVIDIA
fused FFN kernel, which duplicated block 08's path and both shipped on one dispatch).  No delivery code drop
expected.

**What to prove at the rung:** every arm's predicate (which arch/quant/cap selects it), that the RDNA arm
still fires after the merge (`-lv 4` + a shape that reaches it), and that no NVIDIA-only path becomes
reachable on gfx1201.  `#31047`'s layout change is the real risk to our `mmid`/`mmq` hunks — re-derive, do
not patch around it.

- status: **OPEN** · rung R01/R09 · record: —

## D3 — qwen4exp, the indexer and the sparse selection

**Upstream:** `#31443` (write the conv-state tail without an intermediate `cont`), `#31243` (**build the
k-pool sparse selection independently of the `kq_mask` shape**), `#31009` (glm5-next: QSA without `kq_mask`
+ DSA), `#29050` (MFMA lightning indexer for CDNA2), `#31803` (indexer shared-memory bank conflicts),
`#31289` (hexagon indexer).
**Delivery:** block 11 (indexer cache + residency, fused scorer with quantized indexer keys, HC fusions, the
lazy reader, QSA), block 15's sparse decode arm, block 05's derived kq mask.

**Relation:** `#31243`/`#31009` attack the same structural problem our **derived kq mask** and the pooled
sparse selection solved ("the mask shape must not decide what the sparse selection can build").  This is the
highest-probability *subsumption* in the whole re-base.

**Questions:** does upstream's selection make our derived mask redundant (drop), or does it only remove the
blocker (keep both, ours now the cheaper path)?  Does our fused scorer still own the quantized-key case that
upstream leaves on the dequant path?

**Measurement:** on the rebased tree, qwen4exp on the delivery's config: prefill `t/s`, `tg128`, the KLD
gate, and the indexer/sparse census lines, across (a) delivery fusions on, (b) `GGML_CUDA_DISABLE_FUSION=1`
(the chain), (c) the derived-mask kill switch, (d) the sparse arm off.  Judge the fused arm against the
chain, never against the default (`MANIFESTS.md` gate 3, `GREEDY-PURITY.md` §81).

- status: **OPEN** · rung R12 (residue in R06/R16) · record: —

## D4 — The recurrent rollback bound: upstream enforces what we track and warn about

**Upstream:** `#31368` — the recurrent memory and the DSV4 cache record the deepest rollback the *last
ubatch* guarantees (`rs_max`: `n − 1` for a ubatch of `n ≤ n_rs_seq + 1` tokens, `0` for a longer one) and
`seq_rm` **refuses** anything past it.  New tests (`rollback_bound`) cover ubatches of 1, 2, `n_rs_seq + 1`
and `n_rs_seq + 2` tokens.
**Delivery:** block 03's **r91 fold** ("fold every ubatch into the per-sequence rollback tracker", issue
#131) tracks the per-sequence ubatch composition, and its guard **warns once** when a rollback crosses a
batch boundary (`LLAMA_RS_ROLLBACK_DEBUG=1` adds the per-ubatch instrumentation).  Record:
`archive/work/rs-rollback-boundary/`.

**Relation:** the same defect, two answers — ours diagnoses, upstream's refuses.  Upstream's is the stronger
contract; ours carries the diagnostic that produced the analysis.

**Question (needs a decision because it drops delivery code):** migrate to upstream's `rs_max` enforcement
and drop our duplicate bookkeeping?  Recommended shape: **take upstream's bound**, keep the
`LLAMA_RS_ROLLBACK_DEBUG` instrument if it is cheap on `update_rs_for_ubatches`, drop the warning path whose
job the refusal now does.  Verify with the rs-rollback-boundary scenario and `test-llama-context`'s rollback
cases, text identical.

- status: **OPEN** · rung R04 · record: —

## D5 — The FA entry points after `flash_attn_ext_rows`

**Upstream:** `#29510` `llama : flash_attn_ext_rows - more efficient unified kv for multi-seq`, `#31550`
(its VGPR spill list), `#31073` (rotation constants stay on the GPU), `#31683` (kv shifting, per-model
`build_rope`).
**Delivery:** block 05 (the RDNA band kernels and their dispatch, native quantized K/V staging, the derived
kq mask), block 10's FA staging ring.

**Relation:** `#29510` restructures how the unified KV is addressed, which is exactly what our band kernels'
dispatch and the mask fill assume.  This is a **re-derivation**, not a merge — the rung's cost centre.

**Question:** does the new shape make any of our band kernels unreachable (same cap, same head size) on the
configs we validate (gfx1201, gfx1151, gfx1100)?  Any unreachable kernel is a delivery-code drop.

**Measurement:** after the re-derivation: `gate-prefill-logits.sh` (with `--family-switch` if a family cap
moves, `--kv q8_0` for the staging arm) + prefill `t/s` per band on the 4B and 35B-A3B models.

- status: **DECIDED (R06, 2026-11-08) — no delivery kernel dropped.**  The entry-point restructure is absorbed
  (both parameters on every kernel), the band/tile/prefill kernels keep their dispatch, and the derived mask had
  to move its ABI to `src[6..8]` because upstream's `kv_rows` took `src[5]` (merged as-is, the derived probe
  silently failed and every batch fell back to the packed mask).  The remaining coverage question — does the
  band family still fire at head 256/512 — is the R17 `--family-switch` gate's, not a code decision.
  Rung log: `RUNGS.md` R06.

## D6 — `ggml : fix bf16`

**Upstream:** `#31609` (`ggml/src/ggml.c` + `tools/export-lora`) and `#31276` (AVX512 BF16 clang-cl).
**Delivery:** block 04 (the bf16 KV cache type + native-BF16 FA).
**Relation:** expected independent (a type-traits/`_flush` fix upstream, a new cache type and FA arm here) —
verify, do not assume: the block's bf16 arm must still generate and the FA arm must still be selected.

- status: **OPEN** · rung R05 · record: —

## D7 — Compute views in the Meta backend

**Upstream:** `#31447` — a buffer with few tensors can still get many views, so the compute headroom gets a
minimum of **256** views (`max(16 × n_tensors, 256)`).
**Delivery:** block 00/09 — the hybrid recurrent models create `~2 × (n_rs_seq + 1)` conv-state snapshot views
per recurrent layer, which overflowed the original 16× container; the delivery's answer is the
`GGML_COMPUTE_BUFFER_MARGIN_PCT` compute-margin path (`BASELINE.md` → provenance item 2).

**Relation:** the same failure class on a different axis (view count vs buffer bytes).  Upstream's floor may
make our margin unnecessary for the model that motivated it, and our margin may cover cases the floor does
not.

**Question:** does the hybrid recurrent allocation succeed with the margin at `0` on the new base?  If yes,
is the margin narrowed (not removed — it is a documented invariant, and `INVARIANTS.md` must be corrected in
the same change if it changes)?

**Measurement:** a hybrid recurrent model (qwen3.5/3.6/3.8 hybrid or deepseek41) with
`GGML_COMPUTE_BUFFER_MARGIN_PCT=0` vs default, watching for the Meta container overflow / a failed graph
alloc, plus coherence text.

- status: **OPEN** · rung R01 · record: —

## D8 — Which CPU wins survive the CPU-backend rewrite

**Upstream:** `#31854` (**F32 src1 in the CPU backend**: `vec_dot_type(F16)` is now F32, new
`ggml_vec_dot_f16_f32`, conv patches in F32, `supports_op` changes), `#31179` (concat as one row kernel),
`#27472`/`#30585` (SVE GEMM/GEMV kernels), `#31276` (AVX512 BF16 clang-cl).
**Delivery:** block 12's CPU files — `ggml/src/ggml-cpu/arch-fallback.h`, `arch/x86/quants.c`,
`ggml-cpu.c`, `ggml-cpu.cpp`, `repack.cpp`.

**Relation:** `#31854` changes the F16 `type_traits_cpu` entry our hunk sits next to, and it is a *numerics*
change on every CPU matmul.

**Question:** which of our CPU wins remain after the rewrite (the host-expert CPU paths and any `repack`
gains), and does anything become dead code?  Drop only what measurably no longer wins.

**Measurement:** CPU-side `pp`/`tg` on the host-expert config, `test-backend-ops` for the touched quant
types, and the delivery's `-lv 4` CPU-path census.

- status: **OPEN** · rung R13 · record: —

## D9 — Graph cache, views and the allocator

**Upstream:** `#30662` (prevent premature eviction of graphs about to be used), `#31584` (bound views by the
bytes their strides reach), `#31146` (relax scale to non-contiguous views), `#30881` (batched tensor copy
API), `#31447`, `#31141` (`no_alloc` out of `llama_hparams`), `#31200` (remove obsolete `n_pad`).
**Delivery:** block 00 (alloc view release, the compute margin), block 10 (the graph-cache key/replay guard,
the async input ring), block 15 (the memory instances).

**Relation:** no functional overlap, but these are the files where a *silent* behaviour change hides.  The
delivery's replay guard and alloc-deps are correctness-critical (`GREEDY-PURITY.md` §20-class).

**Measurement:** `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`, `gate-state-pipe.sh`, and coherence on a
hybrid model; for `#31584`, a `test-backend-ops` pass (view-bounds regressions show as aborts, not as text).

- status: **OPEN** · rung R11 (residue R13/R16) · record: —

## D10 — All-reduce

**Upstream:** `#28967` (NCCL for multi-node tensor parallelism), `#25051` (vulkan tensor-parallel allreduce),
`#30956` (sycl).
**Delivery:** block 01 (the hybrid HIP all-reduce for small tensors + the per-size dispatch vs RCCL).

**Relation:** different backend and different layer — expected to coexist; verify the dispatch order, that
`GGML_CUDA_ALLREDUCE=hybrid` is still reachable, and that the RCCL path still wins where it should.

**Measurement:** `gate-prefill-logits.sh --sm tensor --device 0,1,2` (mandatory for an AR arm, r165) with
`--ab "GGML_CUDA_ALLREDUCE=hybrid"`, plus the `-lv 4` `all-reduce: … RCCL ring` census.

- status: **OPEN** · rung R02 · record: —

## D11 — Speculative decoding: hidden state, conv snapshots, checkpoints

**Upstream:** `#30141` (save and restore MTP hidden state), `#31660` (store the MTP conv snapshots of every
recurrent model in one `set_rows`), `#31350` (restore the spec checkpoint for drafts longer than `n_rs_seq`),
`#30779` (visit parent slots before child slots), `#31497` (no MoE cache on the draft model), `#31569` (fit:
skipped NextN layers).
**Delivery:** block 02 (adaptive draft depth, `spec_draft_align_cache()`), block 13 (draft vocab subset),
block 12's DFlash ingest tail + draft-reserve split (issues #239/#240/#272).

**Relation:** `#30141`/`#31660` change the hidden-state/conv-snapshot contract the adaptive controller and the
DFlash realign both ride on — this is the one place where an upstream fix can silently invalidate a delivery
invariant rather than conflict textually.

**Measurement:** MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) after the rung that merges
`#30141`/`#31660`; `llama-batched-bench -npl 1,4,8`.

- status: **OPEN** · rung R03 (residue R13/R14) · record: —

## D12 — Quantized-KV state, rotation and `kv_rows`

**Upstream:** `#30541` (on-device state copy sizes for quantized KV caches), `#31073` (rotation constants on
the GPU), `#31570` (never return 0 `kv_rows` for sequences without cells), `#30471` (opencl KV views on
state restore), `#31368` (rollbacks).
**Delivery:** block 15 (the streaming provider keeps one contiguous VA range; `set_rows`, the rope shift,
`state_write/read`, the kq mask and the FA kernels all address it), block 12 (batched state restore).

**Relation:** every one of these touches the exact API surface the streaming provider must preserve
("**no kernel or op change**" is block 15's core claim).  If upstream's state copy stops being a plain
buffer-to-buffer copy, the provider's region binding is affected.

**Measurement:** `gate-kv-residency.sh` (window + pool + the multi-device `-sm layer` section),
`gate-state-pipe.sh`, `gate-kv-window-split.sh`, and the `-ctk/-ctv q8_0` arms.

- status: **OPEN** · rung R16 · record: —

## D13 — Memory instances after `no_alloc` and `n_pad`

**Upstream:** `#31141` (`no_alloc` out of `llama_hparams`), `#31200` (remove obsolete `n_pad` from memory
instances), `#31142` (model output structs), `#31733` (`model` as a class member of `llm_graph_context`).
**Delivery:** block 11's `llama_memory_hybrid_idx`, block 15's region provider, every model file our blocks
touch.  Pure refactor churn with no decision attached — resolve mechanically, but **read `#31733` first**:
33 of block 12's 81 files are affected by it and resolving against the new member shape from the start
avoids redoing hunks.

- status: **OPEN** · rungs R11..R16 · record: —
