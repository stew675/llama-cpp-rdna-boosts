# The delivery patches — what each block is and what it does

**16 blocks** (block 00 + blocks 01-15; block 09 amended in r210, repaired in r211) against `release.json.base` (**`6050b467c`**; the base is the
2026-10-25 re-base), one
`git am` commit each.  Block 14 is **temporary** (see its section): it is a local relaxation of an upstream
rule, kept separate on purpose so it can be dropped without touching any other block.  This file
is the per-block reference: what a block contains, what it changes, the knobs its code reads and the
purity notes that constrain it.  It is **not** a changelog — the release-by-release record is
[`../WORKLOG.md`](../WORKLOG.md), the apply order and verification contract is
[`../MANIFESTS.md`](../MANIFESTS.md), provenance and drift policy is [`../BASELINE.md`](../BASELINE.md),
and every environment variable is catalogued in [`../ENVIRONMENT.md`](../ENVIRONMENT.md).

> **Dated notes.**  This file used to be the running record of block amendments (16-block numbering,
> `r1`..`r93`).  That history is preserved verbatim in
> [`../archive/docs/patch-notes-history.md`](../archive/docs/patch-notes-history.md) and is what
> `WORKLOG.md` entries and `archive/work/<campaign>/` records cite.  Everything below describes the
> **current** blocks in **V2.0** numbering.

## Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$(jq -r .base /path/to/rdna-boosts/release.json)"
bash /path/to/rdna-boosts/scripts/apply-all.sh .          # creates branch rdna-boosts, 16 commits
bash /path/to/rdna-boosts/scripts/apply-all.sh --include-beta .   # + beta/*/*.patch (NOT the release tree)
```

All blocks with **`git am`**, in order.  `git apply` on the concatenated series silently drops hunks; the
manual equivalent is `git am patches/000[0-9]-*.patch patches/001[0-5]-*.patch`.  `apply-all.sh` asserts
the applied tree equals `release.json.tree` on the strict path.

## Block index

Sizes are `git diff` line counts against the block's parent (i.e. what that one commit adds), and matter
only as a rough shape.

| # | patch | subject | +/− | files |
|---|---|---|---|---|
| 00 | `0000-…-block-00-foundation-band-uniformity-Meta` | foundation: band uniformity, Meta wrapper, allocator | +1314 / −1000 | 54 |
| 01 | `0001-…-block-01-hybrid-HIP-all-reduce-RDNA4-gat` | hybrid HIP all-reduce | +2365 / −44 | 4 |
| 02 | `0002-…-block-02-adaptive-MTP-draft-depth` | adaptive MTP draft depth (+ r218: `spec_draft_align_cache()` realigns the drafter's KV with the batch it ingests, issue #272) | +1186 / −54 | 24 |
| 03 | `0003-…-block-03-chunked-gated-delta-net-prefill` | chunked GDN prefill + rollback tracker | +3182 / −15 | 22 |
| 04 | `0004-…-block-04-BF16-KV-cache-and-native-BF16-f` | BF16 KV cache + native-BF16 FA | +630 / −214 | 8 |
| 05 | `0005-…-block-05-flash-attention-band-native-qua` | FA band / native quantized K-V / derived kq mask (+ the three kq-mask fills that lacked their sibling's buffer guard, r131; + the band's fully-masked group skip, r133; + r160: the packed kq mask is only **reserved** where a batch needing it is reachable, issue #238; + r164: the derived input is reusable on hybrid models, PR #242; + r172: the derived mask covers batches of several sequences — 1.03 GiB device + 0.94 GiB host back on a 2-slot unified server, PR #247; + r208: RDNA2 (gfx1030) prefill — the `q_global` threshold 16 → 64 and the D=256 `ncols2` cap lift, and the clamped in-bounds tail — PR #270/#271) | +5565 / −624 | 295 |
| 06 | `0006-…-block-06-MMB-quantized-prefill-GEMM` | MMB quantized prefill GEMM (+ r132: the per-arch dispatch keys on the running device, not the primary) | +2903 / −3 | 7 |
| 07 | `0007-…-block-07-k-quant-boosts-mmvq-VDR-Q6_K-MM` | k-quant boosts (VDR, Q6_K MMQ, Q1_0, grid pad) | +879 / −137 | 9 |
| 08 | `0008-…-block-08-fused-kernels-old-08-fusions-ol` | fused kernels + fused MoE (+ r132: the Q8_1 cache's view-offset keying and source-range guard; + r145: the rms_norm reduction-order selector — strided by default, `GGML_CUDA_RMS_NORM_VEC=1` opts into upstream #29720's float4 grouping; + r161: the dense arm's `!ids` assert no longer gates the routed fused gate arm — issue #241) | +5500 / −440 | 35 |
| 09 | `0009-…-block-09-host-resident-MoE-iface-schedul` | host-resident MoE: iface, scheduler, cache, slab hooks (+ r120: the second draft scheduler's slot is per arch, DFlash2 fix; + r132: the split-state cache hit returns instead of copying; + r144: the traffic report's decode-band census + the `GGML_CUDA_GRAPH_TRACE` graph-call counters; + r149: the graph-cache key carries the subgraph's weight identity, so a split graph's per-layer subgraphs replay — TODO #107; + r203: PR #266's sync-free decode-band device remap and the in-place pool read, `GGML_MOE_DEVREMAP`/`MOE_HOST_POOL_COLD`; + r209: **Windows** — the WDDM free-VRAM figure behind `cudaMemGetInfo` (`GGML_CUDA_WDDM_FREE`) and the working-set release of an uploaded tensor's mapping — PR #269) | +16121 / −540 | 67 |
| 10 | `0010-…-block-10-slab-fusion-dispatcher-async-in` | slab + fusion dispatcher + async input ring (infrastructure) (+ r199: PR #262's `q4_0` KV cache write fold, `GGML_CUDA_KV_FOLD_Q4_0`; + r206: the slab headroom default is `1152 + 128 x n_gpu` MiB (1280 / 1408 / 1536), it was `640 + 128 x n_gpu`; + r132: its two gated-q8_1 launch arms take the same guard; graph_optimize selects its own device; + r139: the compute-buffer chunk follows a live slab, issue #214; + r158: the replay guard compares the live tensors in place with a source-packed store, the graph key is built once per call; + r159: the FA prefill staging arena gains a **KV-chunked ring** (`GGML_CUDA_FA_STAGE_CHUNK`) and an **eager allocation mode** (`GGML_CUDA_FA_STAGE_PREALLOC=2`), both opt-in; + r160: the reserve-reason note records that the M-RoPE term is a declared/reachable 2-D capability, not "any M-RoPE model" — see the block section; + r174: the M-RoPE section power is computed once, PR #249; + r183: PR #256's three GDN decode folds, its shared-expert gate fold and the 2-D copy guard; + r186: the conv fusion's single-device gate is gone, so the fusion and the r183 folds now run on `-sm tensor`/`-sm layer` too; + r187: PR #258's five quantized-KV decode folds; + r188: PR #259's shared-expert lane Q8_1 publish; + r189: the r183 2-D copy guard gets a vectorised fallback for the copies it refuses; + r198: PR #260's mixed K-quant x Q6_K row concat, its `.inc` ksplit body and the codegen fix for the r192 MTP hold; + r202: the fusion-gate oracle's 33-case matrix, tests only; + r203: PR #266's `moe_cache_remap` hook wrapper, relocated here with the other iface wrappers; + r205: the `q5_0` and `iq4_nl` KV cache write folds, `GGML_CUDA_KV_FOLD_Q5_0` / `GGML_CUDA_KV_FOLD_IQ4_NL`, and the shared `RMR_FWHT_LAUNCH` dispatch — `TODO` #120 closed) | +15578 / −5421 | 84 |
| 11 | `0011-…-block-11-qwen4exp-model-hyper-connection` | qwen4exp model/QSA/HC/lazy reader/indexer cache (+ r140/r141: the combine+norm fusion reads the pinned inject, reduces like the chain it replaces, and reaches merged-tensor GGUFs — issue #226; + r142: the separate-tensor mixer emits its inject GEMM beside the down GEMM, restoring the BF16 src1 reuse, +3.4 % prefill; + r148: the HC inject logit is the chain's 1/hc-**scaled** scatter logit in the fused mixer too, so the fused pair agrees with the model's own chain — issue #228; + r190: the fused indexer scorer and the derived indexer pool read **quantized** indexer keys (q8_0 + the q4/q5 family + iq4_nl), so `-ctk q8_0` takes the fast path, TODO #118; + r209: **Windows** — the host-resident PLE gather prefetches a ubatch's rows through `llama_prefetch_rows`, as gemma4 and deepseek41 already did — PR #269) | +12091 / −433 | 46 |
| 12 | `0012-…-block-12-campaign-remainder-CPU-wins-bac` | campaign remainder (+ r106: Meta `ALL_GATHER`, DFlash2 `-sm tensor`, slab narrow-2; + r111/r112: `--host-experts pool` drops the pinned master; FRAC 80 + O_DIRECT defaults; + r119: batched state restore; + r130: DFlash ingest tail, server prompt-cache residency; + r143: the expert cache drops a model's tables at `llama_model_free`; + r146: parking moves a mirroring indexer cache with the attention cache, issue #227; + r147: the host pool's fifth portability helper, issue #231; + r155: a restored context checkpoint is not saved again and dropped checkpoints' buffers are recycled, PR #236; + r162: a DFlash draft reserves its KV injection at `-ub` and its noise block at `n_seq × block`, not the lm_head over `n_ubatch`; + r163: the DFlash target writes its extracted layer inputs into F1's device store from inside the graph, so they are no longer graph outputs, and the host rows are reserved only when the host path runs — issue #240; + r173: a new prompt-cache state buffer is faulted in with huge pages, PR #248; + r203: PR #266's `g_dr_key_to_id` reindex in `moe_cache_forget_all` and the `heat_dec` tally in the lag policy; + r209: **Windows** — the host pool and the KV spill ceiling read `GlobalMemoryStatusEx` as their available-RAM figure — PR #269; + r218: the DFlash/DSpark drafter's KV is realigned with the batch it ingests so a partially reused prompt cannot invalidate it, issue #272) | +5893 / −995 | 80 |
| 13 | `0013-…-block-13-MTP-draft-vocabulary-subset-spe` | MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`, opt-in) | +569 / −6 | 19 |
| 14 | `0014-…-block-14-relaxed-tensor-split-mode-TEMPO` | **TEMPORARY**: `-sm relaxed-tensor` — tensor split with a per-device MoE amax scalar for `flint`/`mellum`/`mistral4` | +130 / −16 | 11 |
| 15 | `0015-…-block-15-windowed-KV-a-fixed-VRAM-reside` | **windowed KV** (opt-in, `LLAMA_KV_RESIDENT_CELLS=<cells>|auto`): the KV's VRAM cost is a **resident window** of the newest `W` cells, the rest in pinned system RAM, in **one VA reservation** — so `set_rows`, the rope shift, `state_write/read`, the kq mask and the FA kernels keep addressing one contiguous range (**no kernel or op change**).  The window slides in whole map units.  Payoffs: a context that OOMs now runs, and the VRAM the KV does not need *yet* goes to the expert-cache arena.  Unset = plain device buffer, byte-identical including the OOM.  Two driver hazards are held (a kind flip zeroes a neighbouring live host mapping without the `LLAMA_KV_REGION_QUARANTINE` hold; every tensor must start on the map granularity), and an idle-slot park is refused on a region-backed cache.  See below and `../archive/work/kv-residency/A2-MIXED-BACKING.md` §3.7.9/§3.7.11.  + r190: a cache that is **streaming** at the depth it reads (a window that the used extent has passed, or an attendance pool) takes the **sparse** QSA decode arm — measured +34 % (51k) / +47 % (180k) on a spilled KV  **The same block carries the ATTENDANCE POOL** (r190, Goal B of the same campaign): the KV's authoritative copy in a pinned host mirror plus a compact device pool of indexer-block pages, resolved by `GGML_OP_KV_POOL` -- a 204 MiB pool instead of 1.6-2.4 GiB of resident KV at 103k-180k depth, landing on the sparse decode ceiling, `--fit`-visible (the mirror is host RAM).  Both placements are **opt-in and default OFF** (fittability features; unset = byte-identical including the OOM) and mutually exclusive.  See the block section   + r204: the window also runs on a **tensor split** (`TODO` #124) -- one region per child device, the grow fanned out to every child, `--fit` charging the physical `W x cell_bytes` against the split's per-device budget; a mirrored dsv4/HRM cache declines it; + r209: **Windows** — the spill ceiling's available-RAM figure comes from `GlobalMemoryStatusEx` — PR #269 | +5294 / −56 | 32 |

Block 05's file count is `template-instances/*.cu` (the FA kernel instantiations); block 10 is the
infrastructure split out of `ggml-cuda.cu` (slab/arena/try-fuse TUs) and block 11 is the qwen4exp
architecture (model, QSA/indexer/HC kernels, lazy reader) that used to share block 10 with it — the r109
split, tree-identical.

---

## Block 00 — foundation: band uniformity, the Meta wrapper, the allocator

**What it is.**  The rules and the plumbing every later block assumes: *decode and verify must take the
same arithmetic*, the tensor-split wrapper must never abort on a legal graph, and the allocator must not
make output a function of the layout.  If a later block's purity claim is "one arm across `W = 1..8`", the
band rule it satisfies starts here.

**Contents.**

* **FA small-batch KV-split width invariance** (issue #25) — `launch_fattn`'s `parallel_blocks` heuristic
  maximised wave efficiency over `ntiles_dst`, a function of `Q->ne[1]`, so single-token decode and a
  verify batch grouped the online-softmax/PV partials differently.  It is evaluated as if `n_q == 1` for
  every `n_q <= 8` (`ntiles_dst_eff`); prefill (`n_q > 8`) is untouched.
* **CPU decode/verify band** — the llamafile `sgemm` path is gated to prefill (`n <= 8` takes `vec_dot`),
  because llamafile rejected `n < 2`, i.e. decode and verify accumulated differently on CPU.
* **Vulkan masked-V / freed-cell fixes** — `flash_attn.comp` / `flash_attn_cm1.comp` never read V for
  dead columns (baseline shaders, hence this block).
* **`-Wshadow` for the `llama` target** (GCC `-Wshadow=local`, clang plain) plus the sweep that makes it
  clean — the class of bug behind the 2026-09-11 dead-mask-chain finding (`GREEDY-PURITY.md` §23).  The
  sweep **keeps upstream's identifiers**: since r128 the loader parameter is `ml` again (the delivery had
  renamed it `ml_l` here, which re-skinned every re-base conflict in `src/llama-model.cpp` for no
  correctness gain — TODO #101), so clang reports that file's `ml` member vs `ml` parameter idiom and
  `build_amax_src1_f16()`'s `n_embd` local in `llama-graph.cpp`: **9 standing warnings, all benign and
  expected**, documented in `src/CMakeLists.txt`.  A `-Wshadow` hit anywhere else is the class the flag
  exists for.
* **`ggml-backend-meta`**: resolve a split-state through the view chain when the alias was never
  initialised (issue #170), return NULL from a failed per-device allocation instead of asserting (a
  clean OOM exit), and keep the split-state hot path at its pre-#170 shape with the #170 resolution in a
  `noinline, cold` helper.
* **The loader honours `--lazy-mode off` under a mapped load, and the Meta split uses the chunked async
  upload ring** (r122): an explicit `off` makes a `TENSOR_READ_LAZY` tensor (the PLE/engram tables) load
  into its own host buffer instead of being served from the `--load-mode mmap` mapping, so mmap keeps its
  fast mapped load for the bulk while the table is really resident (`auto` below 4 GiB and `none` are
  unchanged).  The stale `ggml_backend_dev_is_meta(dev) -> nullptr` bail is gone: the Meta async setter
  serves a mid-row chunk with `write_partial_row` (and falls back to the sync setter for a
  segmented/PARTIAL table), and the event proxy fans the staging-ring backpressure out to the simple
  devices.  Staging slots are 4 MiB (`LLAMA_LOAD_STAGE_MB`), the non-DIO default.  Cold Flash-Next 3-GPU
  `-sm tensor` load: `--load-mode none --lazy-mode off` **32.2 s -> 13.6 s**, and
  `--load-mode mmap --lazy-mode off` keeps the 27 GiB PLE resident at 23.8 s (vs `mmap` + `auto` 18.9 s
  with the PLE lazy); same-seed greedy text is byte-identical on Gemma4-E4B and Flash-Next.
* **`ggml-alloc`**: release view sources whose views are never consumed (a real leak), and fingerprint the
  graph layout by node order + output marking so a *different* graph cannot reuse a layout computed for
  another (issue #183).
* **`MUL_MAT_ID` repeated ids** — emit every `(token, iex)` occurrence as its own compact row and size the
  column grids for the expanded expert list (upstream bug; the delivery was the reproducer).
* **Shared-NextN MTP heads keep their own KV** — the draft driver inferred KV sharing from `ctx_other`
  alone, which is wrong for a `gemma4-assistant` head (upstream bug `04eb4c446`).
* **Multi-ubatch logits readback ordering** — the pinned async D2H copy is completed before the next
  ubatch runs, so a `llama-perplexity` `n_seq >= 4` run cannot read a torn readback.

**Files.**  `ggml/src/ggml-cuda/fattn-common.cuh`, `ggml-cuda/mmid.cu`/`mmf.cu`/`mmq.cu`,
`ggml-backend-meta.cpp`, `ggml-alloc.c`, `ggml-cpu/llamafile/sgemm.cpp`,
`ggml-vulkan/vulkan-shaders/flash_attn*.comp`, `src/llama-{graph,context,model,kv-cache,model-loader}.cpp`,
`src/llama-model-loader.h`, `include/llama.h`, `common/arg.cpp`, `tools/*/README.md`, `src/CMakeLists.txt`,
`src/models/*` (the shadow rename), `src/llama-grammar.cpp`, `common/speculative.cpp`.

**Knobs.**  `GGML_READBACK_SYNC` (diagnostic), `LLAMA_LOAD_STAGE_MB` (tuning, default `4`).  The rest are behaviour, not switches.

**Notes.**  Block 00 is the *only* block whose changes are not gated on RDNA: it fixes code paths every
backend shares.  A re-base conflict here is usually an upstream fix arriving for the same bug — keep
upstream's version and drop the local hunk.

---

## Block 01 — hybrid HIP all-reduce

**What it is.**  A device-side all-reduce for the small tensors `-sm tensor` reduces on every decode step,
plus a per-size dispatch: small tensors take the internal pipeline, and the large ones go to an arm chosen
per model -- RCCL's ring when the model offloads experts to host memory, the copy-engine (SDMA) path
otherwise, with **peer access never enabled** (r165; the choice itself is wired in block 10, which owns the
slab state it reads).  Four files, ~2.4k lines, all in `allreduce-hip.cu` + `ggml-cuda.cu`.

**Contents.**  Internal host-staged pipeline and butterfly kernels; a wall-clock soft/hard spin budget so a
lagging rank fails soft instead of hanging (issue #139); eager `ncclCommInitAll` (a lazy mid-run init
perturbed the split compute and broke `-sm tensor -ncmoe` MTP purity); a first-runtime-failure fallback
that warns once, disables NCCL and re-routes to the internal path; an opt-in P2P/large path and the
copy-engine (SDMA) large-tensor path; and the `AR_PROFILE` device-list initialisation fix.

**r165 -- the large-tensor arm, and `TODO #109` closed.**  The Linux/HIP default is now `auto`: the internal
pipeline for decode/verify-sized tensors and a **large-tensor arm chosen once, at the first large
all-reduce** -- RCCL's ring when the model offloads experts to host memory (`g_comm_host_experts`, set by
`ggml_cuda_slab_arm()` in block 10), the copy-engine arm otherwise -- and **peer access is never enabled**.
Why: `hipDeviceEnablePeerAccess` alone (the AR path held constant) costs a replayed dense decode
42.25 -> 38.77 t/s on 3x R9700; the GPU critical path is unchanged (a per-token sync makes the arms equal:
41.1 vs 41.4), and the cost is not reversible after RCCL has made its mappings (every
`cudaDeviceDisablePeerAccess` returns `no error` and the arm stays at 37.9), so it has to be prevented at
init.  The two arms are good at different shapes, which is what sets the routing rule: the copy engine wins
every dense shape (27B `pp2048` 2178 vs 2018, Gemma4 31B 1998 vs 1944, Flash-Next all-VRAM decode 67.2 vs
59.9) while RCCL wins a PCIe-saturated host-expert prefill at **every** offload depth (Flash-Next IQ4_NL
2 GPU `-ncmoe 20` 809 vs 604, `-ncmoe 32` 555 vs 414, `-b/-ub 4096` `pp4096` 2901 vs 2344) -- disabling
RCCL's P2P globally instead costs 8 % on that shape (731-747 t/s), so the routing is what keeps both.
The copy-engine arm no longer enables peer access either (`GGML_CUDA_AR_CE_PEER=1` opts in; staged and
direct peer copies measure the same, so the default is off).  `GGML_CUDA_ALLREDUCE=hybrid` keeps the
pre-r165 P2P chain, `ce` pins the copy-engine half.  Purity: the decode/verify band's small/large
crossover is unchanged, so those paths are bit-identical to r164; the large arm is a prefill-numeric change,
gated at **mean KLD 0.001448 / same-top-p 98.676 %** against the RCCL arm on 3 devices `-sm tensor`
(`scripts/gate-prefill-logits.sh --sm tensor --device 0,1,2`, `GREEDY-PURITY.md` §67).  Numbers and the
isolation: `benchmarks/2026-10-28-allreduce-peer-access.md`; `WORKLOG.md` 261028.007.

**Knobs.**  `GGML_CUDA_ALLREDUCE` (`auto` default on Linux/HIP since r165, `hybrid`, `nccl`, `internal`,
`ce`, `meta`), `GGML_CUDA_AR_CE_PEER`, `GGML_CUDA_AR_P2P*`,
`GGML_CUDA_AR_CE_SCRATCH_MIB`, `GGML_CUDA_AR_SPIN_TIMEOUT_MS` / `_HARD_MS`,
`GGML_CUDA_AR_SMALL_BLOCKS` / `_SMALL_BYTES`, `GGML_CUDA_AR_KER_EVENTS`, `GGML_CUDA_AR_PROFILE`.

**Purity.**  `GGML_CUDA_ALLREDUCE=nccl` is **not** bit-identical under `-sm tensor` (the internal pipeline
BF16-round-trips), so it is a smoke comparison, never a purity reference.  The hybrid dispatch's
small/large crossover must not move with batch width — the 2-device crossover was raised to the 3-device
value (131072 elements) for exactly that reason (`GREEDY-PURITY.md` §11, cause A).

**Arch gate.**  The internal AR is RDNA4-only; on other arches it falls back to RCCL.  Cross-backend
consistency: the HIP split lives in `allreduce-hip.cu`, upstream's CUDA file is untouched.

---

## Block 02 — adaptive MTP draft depth

**What it is.**  The `draft-mtp-adaptive` speculator: a controller that sets the MTP draft depth from the
observed acceptance, instead of a fixed `--spec-draft-n-max`.  The tuned bucketed controller is the
shipped one (`delta = n_accepted - depth`, `climb_budget(d) = 20 + 6*(d-1)`,
`drop_pressure(d) = max(60, 10*d)`, cold start `max(floor, cap-3)`, `--spec-draft-n-start N`).

**Contents.**  `common/speculative-adaptive.h` + `.cpp` and the CLI plumbing
(`--spec-type draft-mtp-adaptive`, `--spec-draft-n-start`); the draft-depth clamp raised from 7 to 15
(`LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escapes it; purity is promised only to 7 — see
`GREEDY-PURITY.md` §11); an opt-in MTP prompt-tail ingest and a last-row prompt copy (r52); the
target-side half of that tail (r118, PR #206): `llama_set_embeddings_nextn_from()` in `src/llama-ext.h`
plus a skip in `llama_context::decode` of the nextn readback and the end-of-ubatch readback wait for a
prompt ubatch that lies entirely below the installed tail (the `embd_batch_idxs` bookkeeping is still
appended, so `output_reorder()`'s nextn permutation stays aligned with the dense unmasked host rows);
a `--fit` SIGSEGV fix with `draft-mtp-adaptive` (issue #38); `tests/test-speculative-adaptive.cpp` and
`tests/test-recurrent-state-depth.cpp`; and the **embeddings-layer-input preset API**
(`llama_model_set_embeddings_layer_inp_preset`) that the draft setup calls before the target context
reserves its compute buffers (r94 moved it here, the block that calls it).

**Knobs.**  `--spec-draft-n-max`, `--spec-draft-n-start`, `--spec-draft-p-min`,
`LLAMA_SPEC_DRAFT_N_MAX_CLAMP`, `LLAMA_MTP_INGEST_TAIL`, `LLAMA_MTP_TAIL_READBACK` (r118: kill-switch
for the readback skip, on by default; `0` restores the pre-r118 behaviour with the tail still set),
`LLAMA_MTP_DRAFT_OP_OFFLOAD`, `LLAMA_SPEC_ALIGN` (r153: verify-batch alignment, on),
`LLAMA_SPEC_ALIGN_WIDTH_BAND` (r185: the verify width the aligned depth is bounded by on a band-0 config,
default 16, `0` = off).

**The drafter's cache is realigned with the batch it ingests (r218, issue #272).**  The `LLAMA_MTP_INGEST_TAIL`
skip is only exact while the draft's cache stays contiguous with the rows it ingests: the batch check wants the
first ingested position to be the cache's `seq_pos_max + 1`, and a **partially reused** prompt breaks that (the
slot keeps a prefix from an earlier task, so a turn that adds more than the tail's length leaves the draft's rows
ending below its first ingested row).  A strict (non-M-RoPE) target then answers the request with
`inconsistent sequence positions` -> `HTTP 500`, exactly the failure issue #272 reported on the DFlash/DSpark
side.  `process()` now calls the shared `spec_draft_align_cache()` (`common/speculative.cpp`, added here and
reused by block 12's DFlash/DSpark drafter): a stale prefix is dropped - every row of it is older than the
window of the prompt's end, so no later row can attend to it, and the batch that would have carried those
positions is gone - and the draft starts a fresh run at the first ingested row, exactly as it does for a
first-ever prompt.  A draft found AHEAD of the target is trimmed unconditionally (both batch-check branches
reject that), and a `gemma4`-style cache shared with the target is never touched.  No new knob: the realign is a
no-op unless the cache would otherwise be invalid.  **Not reached by any model held here** - every MTP-head model
on this box is M-RoPE (a forward jump is legal there), so the arm is documented, not measured; the DFlash/DSpark
half is gated end to end by `scripts/gate-spec-reuse-window.sh`.

**The verify batch is aligned across the sequences of one step (r153, issue #234).**
`llama_memory_hybrid::init_batch()` splits a decode with `llama_batch_allocr::split_equal()`, which takes one
token per sequence set per round and **stops when the shortest set runs out**, so two slots verifying different
draft lengths put the longer one's tail into a SECOND ubatch and the target runs a full extra pass for that
step.  A fixed `--spec-draft-n-max` never produced a difference (every slot drafts `n_max`), which is why this
survived; the adaptive controller's **per-sequence** depth makes it the normal case on a multi-slot server.
`draft-mtp-adaptive` gives every drafting sequence the deepest cap in the same `draft()` call, bounded by the
smallest **hard** cap in the call (`--spec-draft-n-max`, the MTP layer cap and the caller's per-slot
`n_draft_max`) and by the expert cache's band, so no sequence is ever asked for more than it would draft when it
runs alone.  The alignment lives **inside that drafter** and never touches another implementation's draft: r153
also shipped a generic cut of the finished drafts to the shortest, and measured on the n-gram-first chain it is a
loss (it truncates the other drafter's good draft: 65.6 -> 61.7 t/s, -6 %), so r154 removed it and a chain of
different drafters is inert again (64.6 vs 64.4 t/s, identical draft counts).  Measured (27B UD-Q4_K_XL, greedy `-n 400`, concurrent `recall` + `reasoning`): pair
aggregate **104.8-108.3 -> 140.1-140.6 t/s at 2 GPU (+30..34 %)**, **86.5-87.0 -> 133.1-133.7 at 3 GPU
(+54 %)**, 35B-A3B `-ncmoe 40` **116.2 -> 149.0 (+28 %)**; split decode steps 58 -> 0 at each; equal-length arm
-0.4..-2.1 %; single-sequence text byte-identical.  The low-acceptance slot of a mixed pair verifies at its
neighbour's width, which is a new numerics reference for that config rather than a defect: each config is
deterministic run to run -- a **single-slot** statement: a two-slot aligned pair is not reliably stable across
runs, because the alignment couples the slots (`GREEDY-PURITY.md` §63) -- and `LLAMA_SPEC_ALIGN=0` vs `=1` differ
just as `--spec-draft-n-max 4` vs `5` do.  **A slot that cannot draft at all** (its last token, or a full
context) still splits that one step, deliberately: that ubatch carries the neighbour's genuine draft rows, and
both alternatives measured neutral (`TODO.md` #111, analysed and declined).  Record:
`archive/work/spec-align-234/`.

**The aligned depth is bounded by a verify WIDTH where there is no arena (r185, `LLAMA_SPEC_ALIGN_WIDTH_BAND`).**
r153's bound was the expert-cache band (`band / n_drafting - 1` rows per sequence), and r176 made `band` answer
only for a config with host-resident experts -- so on a band-0 config (all-in-VRAM, `-sm layer`, a model with no
routed expert) a multi-slot draft was **unbounded**: four slots at `--spec-draft-n-max 7` present ~28-32 rows and
the batch leaves the decode kernels' small-batch families (MMVQ/MMVF -> MMQ, an FA family switch above 8 rows).
The bound is now `width / n_drafting - 1`, `width` being the arena band when one is active (unchanged) and
`LLAMA_SPEC_ALIGN_WIDTH_BAND` (default **16**, `0`/`1` = off) otherwise.  **16 and not 8:** at the two widths the
fixed-depth arms measure 16 rows AHEAD on 3 of 4 shapes (35B-A3B 3 GPU 432.7 vs 389.8 t/s, 2 GPU 387.2 vs
359.6; dense 27B 3 GPU 240.1 vs 211.2; qwen4exp Flash-Next GSQ-IQ3_XXS 2 GPU + a resident PLE table is the
exception, 204.6 vs 220.8), which is why PR #169's "above ~8 rows each extra draft token costs more than it
gains" is not the rule: the *unbounded* width was the defect.  Measured with 4 concurrent 300-token greedy
requests on one binary, band on vs off: **35B-A3B 3 GPU 330.8 -> 434.7 t/s (+31.4 %)**, qwen4exp 2 GPU + PLE
**175.5 -> 208.9 (+19.0 %)**, dense 27B 3 GPU **218.5 -> 249.6 (+14.2 %)**, a fixed `draft-mtp n_max 7` arm
310.9 -> 362.8 (+16.7 %); 3 slots +16.9 %; **1 and 2 slots inert** (16 / 2 - 1 = 7 = the default `n_max`, and the
2-slot text stays byte-identical), so every single-sequence gate is untouched.  A host-expert config keeps the
arena band (`width band 16` is what the log prints there too, so the path is unchanged).  Gate:
`scripts/gate-spec-width-band.sh`; record: `benchmarks/2026-10-31-multislot-verify-width.md`.

**The draft's KV cache type follows the main `-ctk` / `-ctv` (r125, issue #216).**  Upstream gives the
speculative draft context its own KV type pair (`--spec-draft-type-k` / `-ctvd`, defaulting to **f16**)
and applies it unconditionally, so `-ctk q4_0 -ctv q4_0` still built the draft context an f16 cache.
For an MTP draft that is invisible waste: it is the *same* model, its context only carries the nextn
layer, and at `-c 131072` on a dense 27B the single draft layer held **512 MiB** of f16 KV against
**144 MiB** at q4_0.  `common_base_params_to_speculative()` (`common/speculative.cpp`) now takes the
main model's type unless the draft's own type was set explicitly, so lowering the main KV type for VRAM
takes the draft with it; `-ctkd` / `-ctvd` (and `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K` / `_V`) still pin
the draft to whatever is asked for, f16 included (the parser records the explicit set in
`common_params_speculative_draft::cache_type_k_set` / `_v_set`).  The change cannot alter a generated
token: the draft only proposes and the target verifies, and the same-seed greedy text is byte-identical
between the two draft KV types (and to plain decode) on the dense 27B, on all four protocol-A axes.
Decode is unchanged within noise where the delivery is tuned: three alternating reps per arm gave
56.4-56.6 (f16) vs 56.4-56.5 t/s (q4_0) at `-c 8192 -n 400`, and 80.8-80.9 vs 80.6-80.7 t/s at
`-c 102400 -n 3000` (protocol A scale), with `ROCBLAS_USE_HIPBLASLT=0`; the MoE arm is 190.4 vs 190.8 t/s.
The quantized draft cache simply takes the native quantized-KV FA arm (block 05) instead of the f16 one.
A **fast small model under `-sm tensor`** does pay for it: 4B Q8_0, 3 GPU tensor, `-ctk q8_0 -ctv q8_0`,
MTP n3 = 237.7/236.1/240.0 t/s (f16 draft KV) vs 232.0/237.2/233.7 (q8_0 draft KV), **1.5-3 % (mean ~2 %)**,
with identical text.  It is not kernel work -- a rocprofv3 trace of both arms is identical (-0.04 %) --
but graph composition: a quantized KV row needs an F32 `src0` for `SET_ROWS`, so ~4 `MUL_MAT` node
evaluations replace 1 `SET_ROWS` per draft KV write, which is per-step host time.  1 GPU and `-sm layer`
are neutral, and `-ctkd f16 -ctvd f16` restores the f16 draft there.  A `-ctk`
left at f16 changes nothing.  Measured with `llama-cli -c 8192 -ub 1024` and with
the draft KV size read back from the loader (`llama_kv_cache: size = ... 1 layers ... K (q4_0)`) on both
the dense 27B and the 35B-A3B host-expert arm.

**Tail-readback purity (r118).**  Skipping the readback cannot change what the draft reads: an ingested row `p` reads `h(p - 1)`, and any ubatch holding a row at `pos >= ingest_from - 1` is never skipped, so that source row is always fresh.  A batch is skipped only when every row is below `ingest_from - 1`, which is why the threshold is `ingest_from - 1` and not `ingest_from`.  Vision rows are always read (M-RoPE jumps the text position after an image).  With the tail unset every threshold is `0`, so the skip cannot trigger and no existing path changes.  The gain has two halves: the MTP hook's early return, which stops `llama_get_embeddings_nextn()`'s `ctx->synchronize()` from draining both backends after every prompt ubatch (about +0.6 %), and the target's nextn readback skip (about +0.1 %); `LLAMA_MTP_TAIL_READBACK=0` disables both.  Verified: identical greedy text and identical draft counts (177/364, acc/pos 0.754/0.426/0.270) after a 31.5k prompt in every arm, prefill-logit gate mean KLD 0.000000 / same-top-p 100.000 %, and prefill 1488.6 -> 1509.8 t/s (+1.4 %, 95 % CI +0.9..+1.9) on 2x R9700.  Records: `archive/work/mtp-tail-readback/`,
`benchmarks/2026-10-23-pr206-mtp-tail-readback-promotion.md`.

**Gate.**  Judge any controller change on the four axes (reasoning / prose / code / recall) **and** the
phase-switching prompt (`prompts/code-reasoning-mixed.txt`), at `-n 3000` (`-n 2000` floor),
`--reasoning on` for R and `off` for P/C/K.  Short runs are a correctness smoke test only
(`benchmarks/mtp-adaptive-methodology.md` rule 0).  Records:
`benchmarks/2026-09-15-adaptive-mtp-tuning.md`, `archive/work/adaptive-mtp-ceiling-scaling/`.

---

## Block 03 — chunked gated-delta-net prefill + the recurrent rollback tracker

**What it is.**  The fused chunked GDN prefill kernel (bf16/WMMA), shipped as **two arch-segregated
files**: the RDNA4 kernel (`gated_delta_net_chunked_bf16.cu`) and a first-gen-WMMA port for gfx11
(`gated_delta_net_chunked_bf16_gfx11.cu`), selected at runtime by device `cc`, with no-op stubs so the
host launchers link on either pass.  Plus the per-sequence rollback tracker, which is what makes a
speculative rewind into a prefill safe.

**Contents.**

* Chunked prefill is **K-independent**: threshold `max(K > 16 ? K : 16, n_rs_batch)`.
* **Rollback snapshots**: only the sequential kernel writes the `K = n_rs_seq + 1` snapshot planes, and a
  whole-batch rollback also needs a pre-batch plane (copied into slot `n_tokens` when
  `0 < n_tokens < K`).
* `prepare()` folds in **every** ubatch of a prepare list (r91) — recording only `ubatches.back()` made a
  sequence that ran in a non-final ubatch keep an earlier prefill chunk as its "last batch", which is what
  the issue-#131 warning was reporting.
* `LLAMA_RS_ROLLBACK_DEBUG` (`RSPREP`/`RSROLL`) is the instrument that proved both.
* **A cell that `find_slot` moves keeps its rollback snapshot planes** (r195, contributor PR #261).
  `build_rs` relocates the "extra states" — the cells inside `[head, head + n_rs)` that no sequence of
  this ubatch uses — and used to copy **plane 0 only**, so from 3 cells on a moved cell's planes
  `1..n_rs_seq` stayed at the old index, where the cells that took their place overwrite them.  That
  sequence's next rollback then read another sequence's snapshot (or zeroed rows): silent wrong text on a
  multi-slot server, no log line.  `build_rs` now also copies planes `1..n_rs_seq` of the extra states
  through the new input `s_copy_extra`, gathered **before** this ubatch writes its snapshots (node order
  is the only ordering guarantee), with the source ids from `s_src()` rather than `s_copy()` because
  `s_copy()` consumes the sequence's pending rollback index.  Where `n_rs == n_seqs` (every
  single-sequence batch and a 2-cell cache) the graph does not change.  The defect is upstream's, so the
  same change is an upstream candidate (`upstream/recurrent-moved-cell-snapshots/`).

**Knobs.**  `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel (bit-identical, slow),
`GGML_CUDA_GDN_CHUNKED_BF16=0` forces the F32 chunked path, `LLAMA_RS_ROLLBACK_DEBUG` (diagnostic),
`GDN_DBG_*` (kernel-debug only).  The sequential-tail knob (`GGML_CUDA_GDN_TAIL`) is read by block 10's
copy of the launch site.

**Purity.**  The pure `none == draft-mtp` range is `n_max <= 7`; `n_rs_batch` is derived from the widest
enabled speculator (`common_speculative_n_max() + 1`, every draft *and* ngram type), so a verify batch can
never take the chunked path (`GREEDY-PURITY.md` §27).  The BF16 chunked path was the issue-#113 prefill-logit
regression: **run `scripts/gate-prefill-logits.sh` on any change here.**

**Gates.**  `test-backend-ops -o GATED_DELTA_NET`, `test-recurrent-state-depth` (**Phase C since r195**
is the moved-cell gate: 15/15 arms differ by 2.94-9.64 logits before the fold and by 0 after, on one GPU
and with `-sm tensor` on three), `test-recurrent-state-rollback` (compare the per-arm FAIL/PASS *set*,
never the count — it is per box, and a pre-r195 baseline has 15 extra Phase-C FAIL lines).

**This is the only chunked GDN kernel the delivery dispatches (r127, gfx1151 A/B).**  Upstream's
`gated_delta_net_mma` kernel (in the re-base window, `62df1815b`) is *eligible* on RDNA3.5 at
`n_tokens >= 2048` — qwen35's GDN layers have `H_k = ssm.group_count = 16`, which satisfies its
`H_k == 16` / `H in {16,32,48,64}` gate — but our chunked branch returns first, so it only runs where
our path stands down (`kda`, `S_v` outside `{16,32,64,128}`, or `GGML_CUDA_GDN_CHUNKED=0`).  Measured on
gfx1151 (halo, r127 chain, qwen35 4B Q8_0, pp2048, interleaved): **ours 3232 t/s vs upstream's MMA
3131 t/s (+3.2 %)**; prefill-logit A/B (qwen35 9B Q8_0, ctx512 x 40) **mean KLD 0.000074 / same-top-p
99.637 %**.  Keep exactly one chunked kernel per device class: do not add a second.

---

## Block 04 — BF16 KV cache + native-BF16 flash attention

**What it is.**  Makes bf16 a first-class K/V cache type with a **native** FA path instead of an F16
conversion, and fixes the gfx11 masked-V signed-zero leak that a bf16 cache exposes.

**Contents.**  `V_DOT2_F32_BF16_AVAILABLE` for the RDNA3/3.5/4 targets (`common.cuh`); native-BF16 tile
instantiations for every head size plus a BF16→F16 converting load for the F16-math fallback
(`fattn-tile.cuh`); the tile case-type chooser and `launch_fattn`'s `need_f16_K/V` decision
(`fattn-tile.cu`, `fattn.cu`); the gfx11 signed-zero masked-V guard in `fattn-mma-f16.cuh` (a fully masked
KV row fed its `x + (-0.0)` products to the WMMA and leaked stale/freed content); `rope.cu` bf16 support;
`-Wshadow`/warning cleanup; the `FLASH_ATTN_EXT` head-size matrix in `tests/test-backend-ops.cpp`.

**Knobs.**  None — bf16 is a cache type (`-ctk bf16 -ctv bf16`).  The staging/native *policy* switch
(`GGML_CUDA_FA_KV_NATIVE`) is block 05's.

**Purity.**  `-ctk`/`-ctv` must **match**: mixed K/V types are hard-rejected at context creation (every
mixed pair measured 1.7-3.6x slower and never smaller).

---

## Block 05 — the FA band, native quantized K-V, the derived kq mask

**What it is.**  The RDNA flash-attention work: a family of band kernels that serve the whole
decode/verify band with one reduction order, native (non-staging) reads of quantized K/V, and the derived
kq mask that removes the packed `n_kv × n_q` F16 mask from both compute and host memory.  287 files —
almost all of them `template-instances/*.cu`.

**Contents.**

* **The band kernels** (`fattn-rdna.cuh`): the RDNA4 head-256 GQA-6 band (the GQA group folded into
  `ncols2 = 8`, a round-robin KV split over a fixed `P`), extended to the 2-byte (f16/bf16) arm and capped
  by `GGML_HIP_FA_BAND_WMMA_GQA_MAX` (default 12) for GQA-12 models.  Since r92 the 2-byte arm uses the
  same 32-column block as the quantized arm with a `nsm/2` band split.
* **The FA chooser band rule (F1)**: a quantized K/V cache took VEC for `n_q <= 2` and TILE from `n_q = 3`;
  both VEC conditions sit inside the band, so the branch is deleted and the band is TILE
  (`GREEDY-PURITY.md` §14).
* **The RDNA4 MMA band row is prefill-free (r126, TODO #98 closed).**  `ggml_cuda_fattn_mma_get_config_rdna`
  (`fattn-mma-f16.cuh`) used to carry two RDNA4 GQA-8/head-256 band rows selected on `ncols` alone.  The
  64-column row was reached by **both** the W=5..8 verify band and every wide prefill (`ncols1 = 8,
  ncols2 = 8`), and since `nthreads`/`occupancy` feed `__launch_bounds__` the band geometry (16 warps,
  occupancy 1, `nbatch_K2/V2` 64, combine 32) was compiled into the prefill kernel instead of upstream's
  256-thread / occupancy-2 row: the FA kernel took 28.74 s vs upstream's 19.63 s for the same 980 dispatches
  at `pp512 @ d98304` (\-10 % end to end).  The 64-column band row is **deleted** (prefill falls through to
  `(256, 2, 64, 128, 128, 64, 1, true)`), which measured **+36 % at d98304, +19 % at d32768, +3 % at d0** and
  puts the delivery 20 % above upstream at depth.  The 32-column band row stays; its dispatch (`n_q <=
  32/ncols2`) can never be prefill, and the comment says so.  The W=5..8 band arm measured unchanged within
  noise (`-ncmoe 40`, 32k, q8_0 KV, n_max 7: 248.8/239.6 vs 241.2/249.7 t/s).  Gates: `FLASH_ATTN_EXT`
  6391/6391, prefill-logit PASS at the standard point, and the head-256 shape PASS in `--family-switch` mode
  against a same-kernel `-ub` control (candidate 0.005073/97.118 vs control 0.007251/96.912).
  `WORKLOG.md` 261024.007.  **RDNA4 only:** the reporter's gfx1031 card takes the RDNA2 tile kernel, so
  this row never applied to him; his depth crossover is still open as `TODO.md` #106.
* **The band's 64-column block has its own geometry and instantiation (r134, issue #224).**  The deletion
  above left query widths 5..8 on *prefill's* 8-warp row, and that row splits a tile's KV rows **2** ways
  per Q column where the 32-column block splits them **4** (`np = nwarps*cols_per_warp/ncols`).  A different
  split is a different reduction, so widths 5..8 were no longer bit-identical to widths 1..4: dense
  head-256 GQA-6 (Qwen3.8-27B UD-Q4_K_XL + DFlash2 `n_max 5`, 512 greedy tokens) differed from plain decode
  at token 325 (8K prompt) and 142 (110K), and was identical with `GGML_HIP_FA_BAND_WIDE=0`.  The row
  `fattn_mma_config(512, 1, 64, 64, 64, 32, 1, true)` is back but **band-only**, selected by a new
  compile-time `band_wide` flag threaded through the config getters, the case function and the kernel:
  prefill keeps `(256, 2, 64, 128, 128, 64, 1, true)` and its deep-prefill win, and widths 5..8 keep one
  pass over the KV cache (`nbatch_combine` 32 stays inside the 64 KiB LDS limit at 16 warps).  The band's
  64-column case + its q8_0 fetch-ahead variant are their own translation units
  (`template-instances/fattn-mma-f16-band-wide-*.cu`, emitted by `generate_cu_files.py`).  A `static_assert`
  on the two band rows (same `np`, same `nbatch_fa/K2/V2`, same `nstages_target`, same `Q_in_reg`) makes a
  repeat a build error rather than a gate failure.  Gates: `test-logits-width-probe` with **`P=4096`** (see
  block 06 — the historical `P=256` run never reached the block) on `KV=q8_0/f16/bf16` (PASS, maxdiff 0;
  the r126 geometry fails it at 0.199764), `plain == draft-dflash` at `-n 512` for `n_max` 3/5/7 x
  `KV=f16/bf16/q8_0` and at `-c 131072`, `FLASH_ATTN_EXT` 6414/6414, `pp512` 4764 t/s and
  `pp512 @ d98304` 1641 t/s (r126: 4624 / 1632, i.e. the prefill row is untouched), and the wide launch
  measured at parity with `GGML_HIP_FA_BAND_WIDE=0` end to end (75.2/74.6 vs 75.0/74.9 t/s at 8K;
  61.0/60.6 vs 60.9/60.8 at a 131072-cell FA range; 164.5/166.6 vs 166.3/164.9 on the W=8 MoE arm) and
  within +-2.5 % at the kernel level (`kv 16384`, q8_0, `n_q` 5: 163.3 vs 167.3 us; `n_q` 8: 174.0 vs
  170.0 us).  **The ratio axis is measured at 6, 8 and 12 since r166:** head-256 **GQA-8** is
  `Qwen3.6-35B-A3B` (16 q / 2 kv, `n_embd_head_k` 256) — the *exact* whole-group fold, the one ratio the
  `ncols2 = 8` tile does not have to mask — and it passes every arm of the r134 gate (maxdiff 0 on
  `KV=q8_0/f16`, `-sm layer`/`-sm tensor`, `RS=0`/`RS=from_w`, on `prompts/prose-rdna-boosts.txt` and on the
  model's own 25.8k-token corpus) plus a `rocprofv3` census that shows the 64-column **band** instantiation
  (not the prefill row) dispatching at widths 5..8.  A gemma4 model is not an arm for this axis: its SWA
  layers are head 256 at ratio 2 and its full-attention layers ratio 8 at head 512, so neither enters the
  predicate.  `GREEDY-PURITY.md` §51, `WORKLOG.md` 261025.004 / 261028.008,
  `benchmarks/2026-10-25-fa-band-width-purity.md`, `benchmarks/2026-10-28-fa-band-gqa8-arm.md`.
* **Native quantized K/V (V4/V5)**: the sub-F16 quant types (`q8_0`/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`iq4_nl`)
  read the cache natively instead of a whole-cache F16 conversion per decode step — a decode-*depth* win,
  and the margin that lets a high-context MTP config load (`GREEDY-PURITY.md` §34).
* **`q4_1`/`q5_0`/`q5_1`/`iq4_nl` as FA cache types** (previously reachable only via
  `GGML_CUDA_FA_ALL_QUANTS`, which also permits mixed K/V), each with its own band sweep and oracle
  (`GREEDY-PURITY.md` §20-22).
* **The derived kq mask (V3)**: per-token bounds + one per-cell position replace the packed mask; the FA
  kernel computes visibility on the fly.  Implemented in **both** the MMA and the tile kernels, and only
  for batches of more than 8 tokens (`n_tokens > 8`) — **not** "prefill-shaped batches": a single-sequence
  verify step reaches that predicate as soon as a draft is 8 tokens deep (a long n-gram draft, or MTP at
  `--spec-draft-n-max >= 8`), which is what made the r164 reuse bug below expensive.  The mask tensor is still
  created; a graph that never consumes it leaves it **unallocated**, so every fill site is guarded on
  `tensor && tensor->buffer` (r131 closed the last three — the hybrid-K one aborted `-np 1` on
  `bailingmoe3`; see Purity below).
* **The derived input is reusable through a hybrid memory context (r164, PR #242).** `can_reuse()` compares
  `params.mctx` — but a hybrid model's context is a `llama_memory_hybrid_context`, never the
  `llama_kv_cache_context` this input was built from, so the pre-r164 code rejected the reuse for *every*
  single-sequence batch of more than 8 tokens and each such step rebuilt its whole graph.  It now looks
  through `get_attn()` (exactly the cache `build_inp_mem_hybrid()` used) and then runs the existing checks
  unchanged.  Measured: reuse returns to parity with the packed path on every arm (n-gram `n_max 15` 1 GPU
  58 → 77; MTP depth 15: 0 → 97 at 3 GPU and 0 → 76 at 2 GPU), which is **+7.6 % / +5.7 %** decode for the
  MTP cases, with greedy text and draft acceptance byte-identical.  `LLAMA_KQ_MASK_DERIVED_REUSE=0` restores
  the rejection.  iSWA still rejects **necessarily**: `build_inp_mem_hybrid_iswa()` registers two derived
  inputs (one bound to the base cache, one to the SWA cache), so there is no single cache to look through —
  do not "fix" that without giving the input a record of which of the two it belongs to.  `WORKLOG.md`
  261028.006.
* **The packed mask is only reserved where it is reachable (r160, issue #238).**  The derived form is a
  *per-batch* decision, so `sched_reserve()` has to cover the packed mask for a batch it cannot serve
  (issue #42).  For an M-RoPE model that used to mean **every** M-RoPE context, because `n_pos_per_embd()
  > 1` was read as "2-D positions can arrive" — but Qwen3.x is IMROPE **even for text**, where every
  position is degenerate (`x == y == pos`) and `kq_mask_derivable()` accepts every batch.  The reserve
  therefore carried `n_kv × n_ubatch × 2` bytes (plus the device copy of it) that no text-only run ever
  allocated: **860 MiB of VRAM and 860 MiB of pinned host memory at `-c 220160 -ub 2048`** on a 27B, which
  is what issue #238 is about.  The M-RoPE term is now **`cparams.kq_mask_pos_2d`** (a new
  `llama_context_params.pos_2d`, which `common` sets from `--mmproj`) **or** a latch that `decode()` sets
  on the first batch that actually carries a non-degenerate 2-D position; the latch forces one
  **re-reserve** with the packed mask before that batch is split, so an undeclared 2-D caller (a library
  that submits M-RoPE image chunks itself) is repaired once instead of relying on a per-graph growth.
  Multi-sequence batches were unchanged by r160 (they kept the packed reserve; r172 extends them, see below),
  as are alibi and the `allow_derived == false` builders.  No kernel, no logit and no runtime path changes: the derived form still serves every
  derivable batch.  `LLAMA_KQ_MASK_PACKED_RESERVE=1` restores the pre-r160 worst-case reserve for every
  M-RoPE context (the A/B arm); `=0` forces the derived one and disables the latch.  Measured (see
  `WORKLOG.md` 261028.001): a text-only 27B at `-c 220160 -ub 2048` reserves **569.20 / 161.21 MiB**
  (device / host) instead of **1428.34 / 1020.36**; a `--mmproj` context reserves exactly what it did
  before; and an undeclared 2-D batch re-reserves to exactly that layout with byte-identical text.
* **The derived kq mask covers batches of several sequences (r172, PR #247).**  A `--parallel > 1
  --kv-unified` server (two slots prefilling or verifying together, one slot's prefill next to another's
  decode) used to keep the packed mask for every multi-sequence ubatch, and because the reserve measures
  with `n_seq_max` sequences it reserved it even when no batch ever held two — `n_kv × n_ubatch × 2` bytes
  on each device **and again pinned on the host** (**1.03 GiB + 0.94 GiB** at `-c 245760 -ub 2048
  --parallel 2`, dense 27B).  Sequence `k` of the ubatch now has its cells and its tokens' `[lo, hi]`
  shifted by `k * 2^23` (`KQ_DERIVED_SEQ_STRIDE`; the ranges are provably disjoint for `pos < 2^23`, and
  `static_assert((int64_t) KQ_DERIVED_SEQ_STRIDE*LLAMA_MAX_SEQ - 1 <= INT32_MAX)` makes an upstream
  `LLAMA_MAX_SEQ` bump a build error), so the one predicate `tok_lo <= cell_pos <= tok_hi` keeps every
  token inside its own sequence's range and **no kernel changes**.  Taken only when one KV stream holds the
  batch, every token is in exactly one sequence, no cell carries two of the batch's sequences (`llama_kv_cells`
  keeps an O(1) shared-cell counter; 0 unless `llama_memory_seq_cp` was used, in which case one O(used
  cells) scan decides) and every position is below the stride; a multi-sequence batch of at most
  **32 tokens** (the multi-slot verify steps, i.e. exactly the `gate-spec-align` boundary) and every other
  refused shape keep the packed mask.  `kq_mask_packed_reachable()` loses its `n_seq_max > 1` term and
  `decode()` latches `kq_mask_multiseq_seen` on the first pre-split batch the form cannot serve, then
  **re-reserves once** with the packed mask (the r160 2-D latch pattern, so no later batch grows the
  compute buffer — issue #42).  A memory `kq_mask_kvs()` does not cover (any other `llama_memory_i`) keeps
  the packed reserve untouched, and a graph that also holds an `allow_derived == false` builder
  (qwen4exp/QSA, MLA) creates and fills its packed mask anyway — measured identical VRAM and RSS on
  Flash-Next, so the win is a dense/plain-cache one.  Purity: the derived predicate *is* the packed one —
  the author's host oracle compares the two fills cell by cell (0 mismatches over llama-perplexity at 4
  sequences per ubatch and the 2-slot server shape, re-run on gfx1201 for dense 27B at 1 and 2 GPU
  tensor, Qwen3.5-4B hybrid, Gemma4-12B and gpt-oss-20b iSWA), and `gate-prefill-logits.sh --ub 2048 --ab
  "LLAMA_KQ_MASK_DERIVED_MULTISEQ=0"` reads **0.000000 / 100.000 %** at f16 and q8_0 KV.  Prefill time is
  unchanged (2 × 16K concurrent: 26.96 vs 27.23 s at 1 GPU, 17.07 vs 17.09 s at 2 GPU `-sm tensor`), single
  -sequence greedy text is byte-identical and `plain == draft-mtp` still holds.  `LLAMA_KQ_MASK_DERIVED_MULTISEQ=0`
  restores the single-sequence rule and the packed reserve.  A server that shares a prefix through
  `llama_memory_seq_cp` pays the one re-reserve and then keeps the packed mask — that is the documented
  trade, not a defect.  `GREEDY-PURITY.md` §70, `WORKLOG.md` 261029.003.
* **q8_0 fetch-ahead variant (r98, PR #195)**: with q8_0 K and V read in place (decode/verify, `n_q <= 8`,
  head 256) the band launches a second kernel, `flash_attn_ext_f16<..., band_fetch_ahead = true>`, that
  fetches the raw q8_0 bytes of the next K/V batches into registers ahead of use and converts each batch
  into shared memory only when it is due; its barriers order shared memory only, because `__syncthreads()`
  also waits for every outstanding global load and would drain the fetches.  Two generated translation
  units of its own (so the kernels without it compile to the same gfx1201 ISA), default on.  The
  instantiation rule the two generated files follow is in **Facts that span blocks**, below.
* **One tile K/V type selection + an RDNA2 checked tail (r99, PR #198)**: `ggml_cuda_fattn_tile_type(K, V, cc)`
  is the single source for both the tile instantiation and the F16 scratch sizing (they had drifted three
  times: the missing q4_0 arm, the mixed-K/V NaNs, the missing RDNA2 bf16 scratch).  On RDNA2 the last tile
  of the unchecked `ncols2 > 1` branch takes the already-compiled `oob_check = true` form, so an exact-size
  operand (test-backend-ops) cannot read into the neighbouring allocation; every other target compiles the
  branch unchanged.  The 511-row guard tail stays with its derivation documented.
* **RDNA2 (gfx1030) prefill: the `q_global` threshold and the D=256 `ncols2` cap (r208, PR #270/#271)**: the tile
  kernel was ~15-19 % below the stock kernel at depth on gfx1030 (issue #214's report), and the gap was the
  launch config plus the checked tail's register pressure, not the host side.  Three RDNA2-only changes:
  the global-half2 Q path (`GGML_CUDA_FATTN_TILE_Q_GLOBAL`) now needs **64** columns instead of 16, so at the
  prefill shape (32 columns) Q comes back from LDS; `rdna_cap_ncols2` is **lifted for `DKQ == 256`** (D=128
  keeps it), which lets D=256 take the GQA fold the cap was blocking; and the checked tail **clamps its
  out-of-range rows to the last in-bounds row** instead of branching to a local zero array.  The tail values
  are discarded by the `k_VKQ_sup` guards before the softmax, every read stays inside the KV cache, and the
  pointer-select the branch existed to avoid (the kv=113 flat-scratch fault of r76/r88) goes with it.  All
  seven launcher `q_h2` arguments move with the kernel condition (one that lags is the r88 null-`Q_h2`
  fault), and `i_sup - 1` is allowed to be `-1` on purpose: a V chunk wholly past the KV end reads the last
  valid row, and "fixing" that to `max(i_sup - 1, 0)` would read one row past the cache.

  **Why the cap was stale, and what makes it safe to lift:** it dated from r76, when HIP's RDNA2 occupancy
  query (which assumes a quarter of the WGP register file) returned 0 blocks for the wide D=256 variants and
  the launch asserted.  `launch_fattn` computes RDNA2 occupancy itself now, so the guard is `blocks >= 1`
  from `4*waves_per_simd/waves_per_block` (LDS-capped at `128K/lds`) instead.  Measured on the patched gfx1030
  HSACO: every D=256 shape the launcher can now pick reports **>= 2 blocks** (4x8: 235 VGPR, 29 696 B LDS,
  4 blocks), so the assert is unreachable, and the family's largest static LDS stays 63 488 B of gfx1030's
  65 536 B.  The checked tail's scratch goes **480/500 B -> 0** on D=256 4x8/16x2 (the submitter's 488/520
  on their toolchain).

  **Blast radius is RDNA2 only, and that is checked, not argued:** both changes are `#if defined(RDNA2)` or
  `GGML_CUDA_CC_IS_RDNA2`, so for gfx1201 every tile kernel's machine code and resource usage are
  bit-identical before and after (88/88 and 106/106 kernels compared per kernel on the build objects;
  two compiles of the *same* source differ in symbol-table order, so object hashes are not the oracle).
  RDNA3_0/RDNA3_5/NVIDIA/MUSA/SYCL are unchanged by the same preprocessor structure.  Timings are the
  submitter's (pp2048 @ d32768 1041.0 -> 1259.8 t/s, tg256 @ d131072 50.9 -> 62.4, V620 gfx1030); RDNA2 is
  best-effort per `CONTAINERS.md` and carries no gate in this repo.  `WORKLOG.md` 261103.013.
* **RDNA3_0 head-512 WMMA rows (r102, PR #199)**: the RDNA head-512 rows of
  `ggml_cuda_fattn_mma_get_config_rdna` were tuned on gfx1201 and keep Q in registers, so on gfx1100 every
  head-512 kernel sat at the 256-VGPR limit and spilled ~1.0-1.4k VGPRs per lane — which is why head 512 was
  capped back to the tile kernel on RDNA3_0 in r5 (issue #30).  RDNA3_0 now has its own rows for
  `DKQ == DV == 512` (Q in shared memory, 128 threads, 64-wide K/V/combine batches, 16/32/64 columns) and
  head 512 takes WMMA prefill again on top of the 256 cap.  Other heads above 256 keep the tile kernel,
  RDNA4/RDNA3_5 rows are untouched, and the decode/verify band stays on the tile kernel.  +34 % deep prefill
  at d32768 / +45 % at d65536 on gfx1100 (gemma-4-26B-A4B QAT, q8_0 K/V), decode unchanged.  This is a
  tile→WMMA **family switch**; judged with the gate's `--family-switch` mode (calibrated point + a
  same-kernel ubatch control, `GREEDY-PURITY.md` §46) it measures **0.001778 / 98.967 %**, inside the
  absolute limits.  The r102 ledger's `0.0293` was the *marginal* pair (default vs the delivery without these
  rows): against fully tile the same build is 0.0462, and a benign same-kernel `-ub` change is 0.105 at that
  point, so the wikitext point never could decide it — `WORKLOG.md` 261021.007,
  `archive/work/fa-rdna3-head512-purity/`.
  **r103:** the fp32 `VKQ_C` array is now sized by the C tile's column span (`T_C_VKQ::J` vs
  `2*T_C_VKQ::J`), so an fp32 accumulator at `DV % 32 == 0` no longer indexes out of bounds.  Every
  instantiated row is fp16 C, for which the expression — and the generated code — is unchanged.
* **Fully-masked KV group skip** (`GGML_CUDA_FA_MASK_SKIP`, master switch) — exact zeros, bit-identical.
  It classifies 256-cell KV groups from the derived per-cell state (single-sequence) or from the packed
  mask (per stream and query tile), then the kernel skips a group only when it is `-INF` for every query
  row of the tile.  **r133 (PR #223, block 05): the decode/verify band gets it too**
  (`GGML_CUDA_FA_BAND_SKIP`, on unless `0`; needs the master switch as well).  The band keeps its tuned
  round-robin KV split — block i still owns the iterations i, i+P, ... and only leaves out the fully
  masked ones, so its accumulation order over the rest is unchanged and each skipped iteration is the
  exact no-op an own-no-iteration block already had to represent (the neutral `-FLT_MAX/2` partial).  The
  gate is the band's own predicate, so it is RDNA4 + head 256 + the folded GQA group + `n_q <= 8`, i.e. the
  tile-kernel decode every other model uses is untouched.  This is the cost `LLAMA_SERVER_LAZY_IDLE_SLOTS`
  (r130) made common: with a ~100K conversation resident in a unified cache, an 8.6K conversation's decode
  goes 50.8 → 41.7 ms/step (**-18 %**, token ids identical; with a DFlash2 draft at `n_max 5`: 18.6 →
  14.8).  With nothing to skip it is a small win, not a tax (-0.6 % at 9.8K, neutral at 105K single-slot),
  because the padded tail group is skipped as well.
* **Hygiene**: the `(512, 512, 32)` RDNA MMA row retuned to a 4-warp shape (issue #188, Gemma-4 head-512
  prefill), the RDNA2 (gfx1030) gate fixes of PR #181, the masked-V HIP fixes, and the FA staging arena
  (prealloc + graceful degradation instead of abort).

**Knobs.**  `LLAMA_KQ_MASK_DERIVED`, `LLAMA_KQ_MASK_PACKED_RESERVE` (r160: `1` = reserve the packed kq
mask for every M-RoPE context, `0` = never/derived-only; unset = reserve it only where a 2-D batch is
reachable), `GGML_CUDA_FA_KV_NATIVE` (auto / force on / force staging),
`GGML_CUDA_FA_MASK_SKIP`, `GGML_CUDA_FA_BAND_SKIP` (r133; the master switch must be on too),
`GGML_HIP_FA_BAND_WMMA*` (`_SPLIT`, `_GQA_MAX`, `_WIDE`),
`GGML_HIP_FA_BAND_FETCH_AHEAD` (r98; `=0` runs the q8_0 band kernel without the fetch-ahead variant),
`GGML_CUDA_FA_WMMA_256` / `_MAX_HEAD`, `GGML_CUDA_FA_RDNA3_WMMA_512` (r102; `=0` sends gfx1100 head 512
back to the tile kernel), `GGML_CUDA_FA_ALL_QUANTS` (build), `GGML_CUDA_OP_TIMING` (diagnostic).

**Purity.**  One arm across `W = 1..8` is the contract; the guarantee is per KV type
(`GREEDY-PURITY.md` §§11-14, 36).  **Any change to an FA kernel or the derived mask must pass
`scripts/gate-prefill-logits.sh`** — decode-side coherence and MTP are blind to a uniform prefill shift.
A **family switch** (the r102 RDNA3_0 head-512 row set, §46) is judged with
`scripts/gate-prefill-logits.sh --family-switch`: it passes the same absolute limits at a point inside the
gate's calibrated domain (0.001778 / 98.967 % at base PPL 2.65) and is reported against a same-kernel
ubatch control measured on the spot.

**The three kq-mask fills that lacked the guard (r131, PR #222 / issue #221).**  Reported, root-caused and
validated by @RikshaDriver (RX 7600 `gfx1102`, ROCm 10.1, `bailingmoe3` / `Ling-3.0-tiny-IQ4_NL`, on
`v14-3f03b1772-r128`).  Only a **single-sequence** context enables the derived form
(`multi_stream = n_seqs > 1 && (!kv_unified || arch == DEEPSEEK4)` disables it at 2+ slots), so at `-np 1`
the packed mask in `llm_graph_input_mem_hybrid_k` has no consumer; the allocator gives an unconsumed input
no buffer, and the unguarded fill hit `llama_kv_cache::set_input_kq_mask`'s `GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))`
— a startup abort before the first token, at every ctx size.  **Ten of the thirteen** fill sites in
`src/llama-graph.cpp` already carried the guard (`if (mask && mask->buffer)`, the idiom
[`attn-k-null-mask-guard`](../upstream/attn-k-null-mask-guard/README.md) introduced); three did not:
`llm_graph_input_mem_hybrid_k` (the one a `bailingmoe3` graph reaches — its sibling
`llm_graph_input_mem_hybrid` guards the identical call) and `llm_graph_input_attn_k_dsa` (`mla` and `lid`
mask).  All three now carry it.  `can_reuse` needs nothing: `can_reuse_kq_mask()` reads `ne[]`, never
`buffer`.  The `attn_k_dsa` pair is **hardening** — that builder passes `allow_derived = false`
(`build_attn_inp_kq_mask(mctx_cur->get_mla(), …, false)`), so no delivery path leaves those two masks
unallocated today; `mem_hybrid_k` builds `attn_k` through `build_attn_inp_k_impl()`, which passes `true`,
which is why it was reachable.  Reproduced here on gfx1201 (r130 tree, `-np 1`, `-c 32768`):
`GGML_ASSERT(buffer)` at `ggml-backend.cpp:579` with the reported stack
(`llm_graph_input_mem_hybrid_k::set_input` -> `llama_kv_cache::set_input_kq_mask` -> `ggml_backend_buffer_is_host`),
rc 134; `-np 2` and `-np 4` unaffected in both builds.  Measured after the guard: `-np 1/2/4` greedy text
byte-identical (same md5), `test-llama-archs -a bailingmoe3` 2/2, the same-seed text identical to
`LLAMA_KQ_MASK_DERIVED=0`, `gate-prefill-logits.sh` on the Ling model **0.000000 / 100.000 %** derived vs
packed (40 chunks), and every previously-working arm of the r130 build byte-identical
(dense 4B 1 GPU + 3 GPU tensor, 27B UD-Q4_K_XL, 35B-A3B `-ncmoe 40`).  Without the guard
`LLAMA_KQ_MASK_DERIVED=0` is the work-around.  Rule: `GREEDY-PURITY.md` §54.  The remaining unguarded fills of the same class
(`set_input_k_idxs`, the pointer-only `s_copy`/`k_rot` checks) are **upstream's own** and unreachable, since
a `k_idxs`/`s_copy`/`k_rot` tensor is always consumed by the node that needed it — left as upstream.

**Upstream candidates.**  `fa-kv-split-width` (block 00), `fa-decode-verify-kernel-family`,
`fa-512-rdna-head-cap`, `attn-k-null-mask-guard` — see [`../upstream/`](../upstream/README.md).

---

## Block 06 — MMB and mmf32: the quantized and the F32 prefill GEMM

> **r217 amendment (issue #273 item 4, `getenv` audit):** `mmf32.cu`'s `GGML_CUDA_MMF32_LOG` latch read its
> variable twice (`getenv("X") ? atoi(getenv("X")) : 0`); it now reads it once into the stored pointer.  No
> behaviour change, no default moved.

**What it is.**  The prefill-GEMM block: `mmb.cu` (~2.4k lines) dequantizes quantized weights into bf16
WMMA fragments on the fly (dense, routed/compact MoE, GLU, F32-split, tall/narrow, tiny-M, IQ3_XXS, HC16,
plus a shadow cache; adapted from `pwilkin`'s `strix-halo` fork), and `mmf32.cu` (**r104**) serves the F32
shapes `mmb` cannot take.

**`mmf32` (r104).**  A non-quantized (F32) weight with more than 16 token columns cannot take `mmf` — there
is no F32 MMA form on RDNA (`amd_mfma_available` is CDNA-only, `amd_wmma_available` is f16/bf16) — so it fell
through `ggml_cuda_mul_mat_cublas()` to rocBLAS, i.e. to hipBLASLt.  That cost twice: hipBLASLt picks its
solution **per process start** from a heap-ordered container, so the prefill arithmetic (and the logits)
changed between runs (`GREEDY-PURITY.md` §44: 3 logit classes in 8 starts on 35B-A3B), and the solution the
heuristic lands on is the *slowest* in the pool for these shapes.  `mmf32` is a plain SIMT tiled GEMM for
exactly those shapes — 32 output rows per block, a 4×8 thread sub-tile, `BN=256` for `m <= 64` (the GDN
`ssm_alpha`/`ssm_beta` projections) and 64 above it (the MoE routers), the k range split into contiguous
chunks summed in increasing z, i.e. a **fixed reduction tree with no atomics**: the same bits on every run by
construction, and an exact F32 FMA accumulation (no approximation).  Measured on gfx1201: 12/12 fresh starts
one logits class (6/6 at 2 GPU `-sm tensor`), `+4.2 % pp512 / +4.1 % pp2048` with decode flat, kernel
0.032 ms vs 0.046 ms for the best hipBLASLt class on `32×512×2048`; `GREEDY-PURITY.md` §47 has the purity
judgement.

*Scope/limitation:* F32 only, `m <= 256`, `17 <= n <= 65536`, `256 <= k <= 8192`, `ldc == m`; everything else
keeps the rocBLAS call.  BF16/f16 prefill (Flash-Next) is untouched, so hipBLASLt keeps its large advantage
there (and its per-start lottery — the BF16 counterpart is not claimed).  A **full-F32 model's FFN GEMMs
(`m` ≫ 256) stay on rocBLAS**: the kernel would need a different tile to be competitive at wide `m`, and no
such model is validated.  Default-on for RDNA4 **and RDNA3_5**, opt-in on RDNA3 (`ENVIRONMENT.md` §10.1; RDNA3
measured −1.3…−2.2 % at `pp2048` on gfx1100, so the lever there would be a per-arch tile, not the gate).

**Contents.**  The `mmb` kernel family and its dispatch (`ggml-cuda.cu`); a per-quant type list
(`GGML_CUDA_MMB_TYPES`); per-arch RDNA3/RDNA4 tuning knobs; `mmf32.cu`/`mmf32.cuh` and its dispatch; and
`tests/test-logits-width-probe.cpp` — the **width probe** that every purity claim in the delivery is
measured with.  Since r166 the probe sizes its token buffer from the tokenizer
(`llama_tokenize(..., nullptr, 0, ...)` returns `-n`) instead of a fixed 8192, so a model's own multi-10k-token
in-domain corpus can be the probe prompt rather than failing with `tokenize failed (need N)`.

**Knobs.**  `GGML_CUDA_MMB` (master switch), `_TYPES`, `_DENSE`, `_ROUTED`/`_ROUTED_THRESH`, `_GLU`/`_GLU_THRESH`,
`_F32SPLIT`(+`_MIN_K`/`_MIN_M`), `_TALL`(+`_MIN_M`), `_TINY_M`/`_TINY_TT`, `_DOWN16`, `_HC16`, `_IQ3XXS`,
`_RDNA3`, `_CFG`, `_CACHE`/`_SHADOW*`, `_MIN_T`, `_TILE`, `_BF16W`, `_LOG`; `LLAMA_MMB_CVT_LOG`;
`GGML_CUDA_MMF32` (kill switch, `=0`), `GGML_CUDA_MMF32_LOG` (diagnostic).  Everything is default-on and
documented in `ENVIRONMENT.md` §§10/10.1.

**Per-device arch dispatch (r132).**  Every per-arch decision in `mmb.cu` (`mmb_enabled`, `mmb_cfg`,
`mmb_cfg_dump_once`, `mmb_dense_flag`, `mmb_wtype_mask`, `mmb_dense_tmask`) keys on
`ggml_cuda_info().devices[ggml_cuda_get_device()].cc` — the device the kernel runs on — not on `devices[0]`.MMB is designed off on RDNA2 (no WMMA), so on a mixed-arch box (gfx1100 primary + gfx1030 secondary) the old
primary-keyed policy let the RDNA2 card run RDNA3-tuned MMB kernels: byte-exact zero output-projection
logits (PPL == vocab) or garbled tail layers, depending on where the output projection sat.
`ggml_backend_cuda_graph_optimize()` selects its own context's device first, because the MMB marking half
consults the same predicates and under the Meta backend the pass is forwarded per child from meta
`graph_compute` (where the current device is still the previously computed one).  On a single-arch box this
is bit-for-bit the previous behaviour; `GREEDY-PURITY.md` §56.

**Purity.**  Both kernels change prefill arithmetic, not decode: run the **prefill-logit gate** on any tuning
change, and the width probe when a fusion/dispatch predicate moves.  For `mmf32` the gate is the
*`--family-switch`* mode at a confident point (`prompts/kld-corpus-35b-in-domain.txt`), with the same-kernel
control as the arbiter — see `GREEDY-PURITY.md` §47.

**The BF16-activation mark may not be preempted, and the stand-down predicates are the dispatch ones (r171,
`TODO` #115).**  The HC16 arm marks an activation **BF16-only** (its producer skips the F32 store) when every
consumer in the split is a bf16 reader, and a fusion that preempts one of those readers reads the buffer the mark
declared dead.  Two rules follow, and they live in this block because the predicates and the marking pass do:
(a) a fusion standing down for MMB asks `ggml_cuda_mmb_supported_mm()` / `_supported_mmid()` — the predicates
`ggml_cuda_mul_mat()`/`_mul_mat_id()` dispatch on — never a weight-type test (the type answer is a superset and
stands down for shapes MMB does not claim); (b) `classify_consumers()` no longer elides a F32 output when a
**routed** `MUL_MAT_ID` reads the activation — it marks the bf16 *copy* instead, so the MMB reader still finds it
and a preempting mmq fusion still reads live data.  Both predicates are null-safe, and a matcher probing an
arbitrary node pair must check every source slot first (the r171 first cut segfaulted on gfx1151).  Applied in
block 10's pair/dispatcher; `GREEDY-PURITY.md` §69.

---

## Block 07 — k-quant boosts: `mmvq` VDR, Q6_K MMQ, Q1_0, grid pad

**What it is.**  The decode-numerics block.  `VDR` (vectors-per-thread reduction) for `Q4_K`/`Q5_K`/`Q8_0`
(2→4) and `Q6_K` (1→2), the RDNA3_5 `mmvq` parameter table split out of the RDNA2 fallback, a Q6_K MMQ
prefill path, the Q1_0 `__byte_perm` → `v_perm_b32` port, and a grid-stall pad.

**Contents.**  `vecdotq.cuh` (VDR constants + the `ggml_cuda_byte_perm` helper), `mmvf.cu` (the Q6_K
decode path), `mmvq.cu` (the VDR-aware K-split loop and the RDNA3_5 table), `mmq-vec-dot.cuh` /
`mmq-load-tiles.cuh` / `mmq.cu` (Q6_K MMQ), `common.cuh` (grid pad), and the Q1_0/Q6_K cases in
`tests/test-backend-ops.cpp`.

**Knobs.**  `GGML_CUDA_GRID_STALL_PAD`, `GGML_CUDA_MMQ_J_MAX`, `GGML_CUDA_OP_TIMING` (diagnostic); build-time `-DGGML_HIP_Q6K_DENSE_VDR1` (r170, the dense Q6_K VDR=1 reproducibility escape — see `ENVIRONMENT.md` §14).

**Dense Q6_K VDR is upstream's now, on every arch (r170).**  The block's own per-arch gate
(`#if defined(RDNA4) || defined(RDNA3_0)` → VDR=2, upstream's VDR=1 elsewhere) went away with the
`6050b467c` re-base: upstream #30517 (`2744e48d1`, "faster MMVQ for low-bit types") made
`VDR_Q6_K_Q8_1_MMVQ` 2 unconditionally, so the macro arrives as *context* and RDNA3_5 (gfx1151) moved
1 → 2 as a side effect.  That is measured and kept: gfx1151 Qwen3.8 Flash-Next Q5_K (13 dense Q6_K
tensors incl. the lm_head) is within noise of VDR=1 at B=1/B=4 and at `tg128 @ d16384`, and **+1.2 % at
batched B=8**; width purity is unchanged (`plain == draft-mtp`, `n_max` 3 and 7 byte-identical).  Greedy
text *does* move against a pre-#30517 build — that is the documented association-order class, not a
defect.  `-DGGML_HIP_Q6K_DENSE_VDR1` restores the VDR=1 order for a consumer that owns an
output-identity gate across that base bump (see `ENVIRONMENT.md` §14 and `GREEDY-PURITY.md` §68).  The
MoE-expert VDR is per kernel (`GGML_CUDA_MMVQ_MOE_DENSE` collapses it to the dense VDR off
RDNA4/RDNA3_0) and is untouched.

**Purity.**  **This block changes decode numerics on every backend** — same real-number result, different
fp32 addition order.  Read `GREEDY-PURITY.md` §§1-9 before touching a VDR value: stock bits then require
re-cutting the set without this block (it is no longer the last patch, so it cannot be dropped with a
one-line edit).  A PPL run before/after is the gate: it must be **unchanged**.

---

## Block 08 — fused kernels (including the fused MoE gate+up+GLU)

**What it is.**  The CUDA fusion layer: matchers in `ggml_cuda_try_fuse` plus the kernels they dispatch to,
each one replacing a small chain of standalone launches, all constrained by the rule that a fused path
must be **bit-identical** to what it replaces at the widths it serves.

**Contents.**

* `mmvq.cu`: the dense `ncols == 1` K-split alignment, the fused dense gate+up+GLU, and (r93) the
  **RDNA3_0 per-shape `nwarps`** (`Q4_0/Q4_1/Q5_0/Q5_1` narrow at every row count, `Q8_0` narrow for
  `2048 <= M <= 8192`), forwarded to every fusion that reproduces a dense weight launch
  (`GREEDY-PURITY.md` §19 addendum).
* `mmq.cuh` / `mmq.cu`: the fused MoE gate+up+GLU MMQ prefill (**`has_gate`**, the RDNA arm — upstream
  #29948's interleaved **`has_fusion`** arm ships beside it, NVIDIA-only, and `static_assert(!(has_fusion &&
  has_gate))` keeps them from ever being on together; r138), the routed-compact dispatch, `mul_mat_q_pair`,
  the FP4/gate precision contract, and the Q6_K compact-J path.
* **A fusion that stands down for one of the delivery's own fused paths must ask that path's DISPATCH
  predicate, never a weight-type test (r171, `TODO` #115).**  `mul_mat_q_pair` is the fallback owner of a
  gate+up pair: it must stand down on `ggml_cuda_mmb_supported_mmid()` / `_supported_mm()` (the predicates
  `ggml_cuda_mul_mat()`/`_mul_mat_id()` dispatch on, a strict superset of the fused-GLU one) and on nothing
  weaker.  The type-only helpers this block carried (`ggml_cuda_mmb_dense_will_take` / `_routed_will_take`)
  are gone: they stood the pair down for shapes MMB never claims (`T < mmb_min_t()`, `n_used > 64`, a `K`
  that is not a whole super-block count) and, worse, disagreed with block 10's BF16-only activation marking,
  whose whole point is that a preempted reader's F32 buffer was never written.  With the exact claim the
  priority chain is well defined: **MMB's fused GLU -> the mmq/mmvq triple -> `mul_mat_q_pair` (the
  fallback) -> the two plain muls**.  Measured neutral (gfx1151 pair == triple within noise and
  byte-identical text; gfx1201 inert), and the matcher now null-checks both nodes' source slots because it
  runs on every graph node — the first cut segfaulted on gfx1151 loading Flash-Next at `-ub 256`.
  `GGML_PAIR_ROUTED_OFF` is the routed half's A/B arm.  **Chain hygiene (r172):** this block's
  `ggml_cuda_graph_optimize` also called `_routed_will_take` at one site, and r171 removed the helper from block
  06 — so blocks 08 and 09 no longer compiled as intermediate trees (the call site was only deleted later, by
  block 10's dispatcher move; found by walking the ladder instead of `--changed-from`).  That call site now asks
  `ggml_cuda_mmb_supported_mmid()` (its four arguments are the following `MUL_MAT_ID`'s), so every block builds
  again; the **tip tree is unchanged**, because block 10 deletes the whole region.  `GREEDY-PURITY.md` §69,
  [`../benchmarks/2026-10-29-mmb-pair-ownership.md`](../benchmarks/2026-10-29-mmb-pair-ownership.md).
* **The routed fused-gate arm takes `ids`; the dense arm's `!ids` assert must not gate it (r161, issue #241).**
  `ggml_cuda_mul_mat_q`'s `if (fusion)` preamble carried upstream #29948's dense precondition
  (`gate && !ids && !bias && !scale`) after the r138 re-base, but the same function serves this block's routed
  arm (`ids != nullptr`, `ggml_cuda_mul_mat_q_switch_type_gate`, no bias/scale) — so the MoE gate/up+GLU
  prefill fusion aborted on every config that reached it (`lfm2moe` server prefill on gfx1151; on gfx1201 the
  same triple is normally consumed earlier by `mul_mat_q_pair`, so `GGML_PAIR_OFF=1` is the repro).  The
  assert no longer names `ids`; bias/scale stay dense-only and a routed call can never reach upstream's dense
  interleaved kernel.  `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` is the A/B arm: fused == unfused, byte-identical
  greedy text and prefill-logit KLD -0.000000 / 100.000 % top-p.  `GREEDY-PURITY.md` §65.
* `moe-weighted-reduction.cu`: the MoE weighted-down reduction (+ column-blocked band epilogue).
* `norm.cu` / `quantize.cu` / `unary.cu`: `rms_norm`→Q8_1 folds, the verify-band q8_1 cache, scale-unary,
  norm-rows, the conv1d/PLE fusions and the shared-expert down gate.
* **The rms_norm reduction order is a numerics selector, not a knob (r145, issue #229).**
  `ggml_cuda_rms_norm_vec()` (`common.cuh`) is the single selector; `rms_norm_f32`'s dispatch and every
  kernel documented bit-identical to it consult it (`hc_combine_norm_f32`, block 11, takes it as a launch
  argument).  The **default is the pre-#29720 strided scalar reduction** — the reference every recorded
  MTP/acceptance number and the decode-band fusions were built on; upstream #29720's **float4** grouping
  (which the r127 re-base made unconditional) is `GGML_CUDA_RMS_NORM_VEC=1`.  The float4 path is worth
  **+1.0-2.7 % prefill and ~0 % decode**, but it moves MTP draft acceptance by a payload-dependent amount
  (issue #229: −7..−9 % tg on the reporter's box; +0.7..+1.3 points here on the default arm, and
  0.611 vs 0.792 acceptance, −18 % t/s, with the SSM decode fusions disabled) and it makes the fused norm
  kernels *approximate* (0.0005 mean KLD at `-ub 1`, where the contract says 0; exact on the default).
  Rule and evidence: `GREEDY-PURITY.md` §60,
  [`../benchmarks/2026-10-26-rms-norm-float4-ab.md`](../benchmarks/2026-10-26-rms-norm-float4-ab.md),
  `ENVIRONMENT.md` §7.5.
* **The Q8_1 quantize cache's identity and its source guard (r132).**  The cache is keyed by the **root**
  tensor's identity but the arena is filled from the (possibly offset) **view** pointer, so an offset view
  stored its rows under the root identity and a later op with the same root and the same canonical row
  layout could take them.  Every key walk that reaches the cache now stops at the **first pointer change**
  (this block: the four `mmvq` sites and the unary `mul_q8_1` site; block 10 adds its own GLU site), so
  offset-free views keep the shared root entry and an offset view gets its own entry.  `q8_1_src_range_ok()`
  (block 08's `common.cuh`) validates the source read range against the root buffer's allocation before
  every `quantize_row_q8_1_cuda` / fused gated-Q8_1 launch: a compute buffer reallocated under a
  still-matching cache entry used to fault (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`) or silently read
  reused memory, and now aborts loudly with the layout logged.  No knob; `GREEDY-PURITY.md` §55.  Not
  reachable on the maintainer's model set (0 offset-view keys in 14336-60416 key walks per config across
  seven configs on gfx1201), so it is a correctness fix for configurations the delivery does not run.
* `topk_moe`: the fused MoE router, made bit-identical to the softmax→argsort→get_rows→norm chain
  (`GREEDY-PURITY.md` §31) with a sorted-key argmax and a stable tie-break.
* The **RDNA3_5 single-token-only fusion skip** (the dense GLU and the weighted-down tail are not
  decode/verify bit-identical), `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` as the A/B.
* `ggml_backend_cuda_buffer_set_tensor_2d`'s **model-load staging** (r116, TODO #90): a `-sm tensor`
  load uploads one tensor at a time through a pinned gather buffer, and it used to
  `cudaMallocHost`+`cudaFreeHost` that buffer on **every tensor**.  A 27B Q8_0 at 3 devices does it 678
  times — measured in one `rocprofv3 --hip-trace` pass as 2.2 s of `hipHostMalloc` + 1.1 s of
  `hipHostFree` of a ~10.7 s load — so the staging buffer is now **one grow-only buffer per device**
  (bounded at 64 MiB retained; a larger tensor keeps the one-shot allocation) and the load drops from
  **12.7 s to 9.5 s** on that model (678 -> 5 allocations).  It is a byte-for-byte move of the same
  bytes on the same stream with a stream sync before reuse, so greedy text is unchanged; measured at
  2 and 3 devices on the dense 27B and Flash-Next GSQ `-ncmoe 48`.  `GGML_CUDA_STAGE_PIN=0` restores
  the per-call allocation.

**r201 (issue #265): the SSM gate/beta fusion computes the chain's softplus, verbatim.**  `ssm_op_softplus`
is now `(x > 20) ? x : log1pf(expf(x))` -- the same expression as `unary.cu`'s `op_softplus`, the node the
fusion replaces.  It was `logf(1.0f + expf(x))`, a different rounding, so the fused path was not the chain at
the widths an MTP verify batch runs (0.000562 KLD at `-ub 4`, and it moved the target's greedy text); the
r183 gate's `--ub 1` arm could not see it because at one token the SSM pre-scan fusion owns those nodes on
qwen3.5/3.6/3.8.  `GREEDY-PURITY.md` §81.

**r202 (issue #263, TODO #130 closed): the switch has two arms, so the fused value is the reference.**
`GGML_CUDA_FUSE_GATE_BETA_VERIFY` gates **only the verify band** — `mmvq.cu`'s predicate is
`ne[1] == 1 || (fuse_gate_beta_verify && ne[1] <= MMVQ_MAX_BATCH_SIZE)`, so the decode arm at one token is
unconditional.  A `=0` arm therefore compares the fused kernel (W=1) against the chain (W>=2): on gfx1100
that read `maxdiff 0.138` (`7def98c5` fused vs `5e1cbbb4` chain) before the softplus fix and reads 0 now, on
the same build, with `--ub 4` at `-0.000000 / 100.000 %`.  The pre-fix **default** build was the same fused
kernel at *every* width, i.e. width-uniform and wrong — which is why the defect looked like a kill-switch
artifact and why the default arm's own MTP acceptance moved with the fix (r202: pre-fix 977/710
`sha=eca29f334d6a`, post-fix 988/704 `sha=152af1d9feb7`, both arms identical).  Record:
`benchmarks/2026-11-01-kill-switch-arm-130.md`.

**Knobs.**  `GGML_CUDA_DISABLE_FUSION` (master), `_DISABLE_MOE_MMQ_FUSION`, `_DISABLE_MMQ_ROUTED`,
`_DISABLE_TOPK_MOE_FUSION`, `_DISABLE_CONV_FUSION`, `_DISABLE_NORM_ROWS`, `GGML_CUDA_TOPK_SORTED`,
`GGML_CUDA_FUSE_GDN_GATE`, `GGML_CUDA_FUSE_Q8_1_VERIFY`, `GGML_CUDA_MMVQ_RDNA3_SMALL_M`,
`GGML_CUDA_SCALE_UNARY`, `GGML_PAIR_OFF` / `GGML_PAIR_DENSE_OFF` / `GGML_PAIR_ROUTED_OFF` / `GGML_PAIR_2X`,
`GGML_CUDA_Q8_1_ARENA_FREE_OLD`,
`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS`, `GGML_CUDA_STAGE_PIN`.  Block 10 adds the rest of the `_DISABLE_*` / `_FUSE_*`
family (the qwen4exp/GDN fusions); the full catalogue is `ENVIRONMENT.md` §7.

**Purity.**  Every fusion carries the same requirement: one arm across `W = 1..8`, and — because some
matchers fire only at one width — the fused kernel must reproduce the standalone launch at that width
(`GREEDY-PURITY.md` §§39, 42, 25).  The `nwarps`/VDR choice is part of the reduction order: a
shape-scoped value must be forwarded to *every* kernel that reproduces that launch.

**Gates.**  `test-backend-ops -o MUL_MAT`/`MUL_MAT_ID`, the width probe on a MoE model, and the
stock-relative `llama-batched-bench -npl 1,4,8` arm.

---

## Block 09 — host-resident MoE: iface, scheduler, expert cache, staging rings

> **r215 amendment: the host pool's MIRROR fill and its pinned BACKING.**  Two costs in the pinned L2 pool
> that only a **mapped** load pays, both found by asking why `--load-mode mmap` reaches `listening` 1.6x later than
> `--load-mode none` on the same model (Flash-Next IQ3_S, `-ncmoe 48`, cold page cache, 1 device: 26.3 s against
> 16.7 s).  Neither arm does more *work* — both move the same 48 GiB of host-expert weights into RAM.
> **(1) The fill.**  A pool that covers a whole tensor is a **mirror**: its memory is one contiguous
> `n_experts * host_bytes` allocation and the GGUF expert layout is expert-major at the same stride, so the whole
> mirror is **one sequential read of the tensor** (slot `e` == file offset `file_offs + e*host_bytes`) instead of one
> `host_bytes` `memcpy` per heat-ranked slot out of the demand-faulted model mapping.  48 GiB went **10.8 s -> 7.4 s**
> (4.5 -> 6.5 GB/s, the rate the direct load gets) and it stops faulting a second 48 GiB into the page cache that
> cannot be kept beside the pinned mirror.  The order is content-free here (every expert is resident either way); a
> partial pool keeps the heat-ranked path.  `MOE_HOST_POOL_BULK=0` is the A/B arm.
> **(2) The backing.**  `cudaHostAlloc(Portable|Mapped)` pins page by page, so pinning the 48 GiB mirror cost **5.4 s**
> — and pinned a zero page for every 4 KiB page that the fill then overwrote.  The pool is now an anonymous
> `mmap` + `MADV_HUGEPAGE` region pinned with `cudaHostRegister(Portable|Mapped)` **after** the fill (~1 ms plus a pin
> folded into the fill's own page allocation), released with `cudaHostUnregister` + `munmap` in block 12's
> `pool_free_locked`.  `Mapped` is required: `Portable` alone leaves the region unreachable by an in-place cold read
> (a GPU page fault), which is why the two vendor headers gained the `cudaHostRegisterMapped` mapping.  The
> release pairing is not optional — `cudaHostUnregister` alone does **not** return the pages on this stack (the pin
> is tied to the VMA: `archive/work/host-expert-eager-push/` P4 §5), and a pool SLOT is index bookkeeping inside that
> one allocation, so recycling never pins or unpins (144 register / 144 unregister over a run with 35 559 slot
> fills; `AnonHugePages` 48.97 GiB -> 0 at teardown).  `MOE_HOST_POOL_ALLOC=portable` restores the r214 shape.
> **Gates** (r215, `gate-prefill-logits.sh --ab` on the host-expert arm): mean KLD **0.000000**, same-top-p
> **100.000 %** — the two arms are bit-identical at the logits; same-seed greedy text identical at 1 and 2 devices;
> `-npl 4/8` batched within noise (B=8 tg 121 vs 118 t/s, no sign of the r210 multi-slot collapse); MTP prose
> `-n 2000` acceptance **0.64545 (1318/2042) / mean len 2.94** in both arms with prefill 999.7 vs 984.0 t/s (the
> archived 1-device registered-source prefill regression, P4 §5d, is **not** reproduced); `llama-batched-bench`
> at `-c 16384` segfaults on this model in both arms (pre-existing, workload/model limit, not this change).
> Records: [`../WORKLOG.md`](../WORKLOG.md) 261104.008, `INVARIANTS.md` r215, `ENVIRONMENT.md`
> `MOE_HOST_POOL_ALLOC` / `MOE_HOST_POOL_BULK`.
>
> **r214 amendment: the draft reserve's `compute` term is a measured FLOOR, not a flat tier.**
> `common_draft_reserve_estimate()` charges `weights + kv + compute + slack` to the slab's `aux` reserve, and its
> `compute` term was a 512/768 MiB tier measured at `-c 16384`.  The draft context's compute buffer is not a constant:
> it is read back MEASURED (`sched_reserve`'s per-backend size) and grows with the draft's **context** in 256 MiB lumps
> — 0.75 GiB at `<= -c 65536` up to 1.75 GiB at `-c 262144` on the field config (1.25 on one device at 204800) — so
> the estimate was 2-3.5x short at any context that config is run at, which is what put the slab's narrow-2 region
> across its mapped extent (`TODO` #143).  It is now the measured minimum (768 MiB, +256 above a 4096 ubatch) and
> deliberately stays a floor: an over-estimate cannot be reclaimed by `slab_extend` on a full box, and a shortfall now
> costs only a bounded arena yield that block 12's pin maps.  Records: [`../WORKLOG.md`](../WORKLOG.md) 261104.007,
> `INVARIANTS.md` r214.
>
> **r210 amendment: the tier is one pinned per-expert pool, laid out as the arena's complement and then
> mirrored residents.**  `-lme auto|N` is the budget (`--host-experts pinned|pool` are deprecated shorthands):
> `auto` spends the whole host expert set on the pool when it fits `MOE_HOST_LIMIT_FRAC` (80 %) of `MemAvailable`
> — the pool is then a **mirror** of it and the tier behaves as `pinned` while prefilling faster — and keeps the
> historical `MOE_HOST_POOL_FRAC` layout otherwise (the not-fitting band: page cache / `O_DIRECT`, nothing hard
> pinned).  `-lme N` spends N GiB on the pool: complement first, then the arena-resident experts in heat order
> while capacity lasts.  A pool-resident expert is read in place by any device's kernel
> (`MOE_HOST_POOL_COLD_MULTIDEV`).  Measured at 2 devices, byte-identical text: prefill **1318** against a pinned
> master's 1105 (+19 %), decode 69.8 vs 71.9 (n=300) and 82.5 vs 82.9 (n=2000) for the same 18 662 MiB pinned.
> Records: [`../WORKLOG.md`](../WORKLOG.md) 261104.001, `INVARIANTS.md` r210,
> `wip/host-expert-eager-push/{P5-RESULTS,P6-POOL-REGION}.md`.
>
> **r211 repair + amendment: a covering pool is a complete cold source, and this block's tree builds on its
> own.**  Two things.  (1) The eager take-over's completeness guard bounded a pool-backed table by the
> **arena** (`t.slots < n_used * n_tok`), because `table_cold_ok()` is false for a pageable master — so at
> `llama-batched-bench -npl 8` (8x8 = 64 routed ids against 56 slots) **every routed op declined** and the run
> read the pageable mapping: tg **49.2 against 270.5** for a pinned master.  `pool_covers_table()` now exempts a
> pool that covers the tensor (every expert is pool-resident, so `id = n_res_slots + slot` is valid for all of
> them — the pinned master's own property) and the guard is
> `!table_cold_ok(t) && !pool_covers_table(t) && t.slots < n_used * n_tok`.  Measured at 2 devices: `-npl 8`
> **1829.5 pp / 270.3 tg** (pinned 1262.8 / 276.5; the pre-r210 tree 1266.4 / 273.0), `-npl 4` 1759.0 / 193.6,
> and the whole-coverage pool arm (`-lme 64`) 1820.1 / 271.4.  A **sub-complement** budget keeps the bound
> (`-lme 8` at `-npl 8`: 50.4 tg — `TODO` #141).  (2) The r210 fold had put this block's r210 **uses** ahead of
> the definitions blocks 10/12/15 own, so **blocks 09, 10 and 11 did not build** (a spliced
> `pool_rerank_from_tally_locked` body, `t.n_slices`, `heat_prior_t`, a block-12 iface entry, and the loader's
> `plan_host_expert_placement`/`llama_host_experts_mode_apply`/1-arg `host_expert_buft`).  Each early tree now
> carries the pre-fold body or a **hoisted definition**, and where the hoist would need a block-12-only backend
> API (`ggml_backend_dev_moe_host_master_buft`) the pinned host buft is the one answer until that API lands —
> honest for an intermediate tree, and the **tip tree is byte-identical** (`e2089b13`) before and after.
> `gate-per-block-build.sh --changed-from 27271aa91`: 5/5.  Records: [`../WORKLOG.md`](../WORKLOG.md) 261104.002,
> `INVARIANTS.md` r211, `archive/work/host-expert-eager-push/`.

**What it is.**  Everything that makes `-ncmoe`/`-cmoe` (and `--host-experts`) fast and safe without
fitting the whole model on the device: a pinned host master, a per-op H2D staging ring, a scheduler that
can hand an op to a host-resident table, and the MoE expert cache that keeps the hottest experts in
whatever VRAM is left.

**The decode band no longer reads the routing back to the host (r203, PR #266, `briansp2020`).**  The
eager hook pays a device synchronize per routed op to get the routing onto the host (`input_backend`'s sync
plus the ids readback's own sync), runs the residency policy and uploads the slot remap.  A new iface member
(`moe_cache_remap`, appended **last**; NULL in the meta, cpu and rpc backends, so `-sm tensor` keeps the
eager path) is offered the split input *first*: a kernel maps the device routing through a per-table
expert -> slot map into that table's `remap_dev` (`slots + e` = cold, read in place from the pinned master)
and writes the routing into mapped pinned memory; the policy consumes that routing at the **next** call and
re-uploads the map only after a residency change.  One launch serves a layer's tables (gate / up / down)
when they were served the same routing tensor on the same stream.  Every expert is read from the same bytes
as before -- an arena slot, or the same pinned master the cold read already used -- so the output does not
change: byte-identical greedy text (35B-A3B `-ncmoe 40` and gpt-oss-20b MXFP4 at 128 and 1000 tokens,
including a 2048 MiB arena under eviction pressure; pinned and `--host-experts pool`; a 2-slot server over
slot switching), MTP acceptance identical to five decimals (0.76089 = 1397/1836, `acc per pos` identical,
`-n 3000`; R axis 0.83340 = 2141/2569) and prefill KLD **0.000000 / 100.000 %**.  Worth **+8.4 % TG**
(60.6 -> 65.7 t/s), +2.4 % on the MTP arm, +14.1 % / +9.2 % at `-npl 4` / `-npl 8`, and inert under
`-sm tensor` (98.0 vs 98.1 t/s, identical text).  `GGML_MOE_DEVREMAP=0` restores the eager path.

Scope by construction (`devremap_ok_locked`): a pinned master (`table_cold_ok`), one unsplit table
(`n_slices == 0`), no identity table, `t.device == device`, the decode/verify band (`ne[2] <= band`) and
only after `g_sized` (the eager path primes, sizes and re-primes).  **The invariant to keep is that the map
is rebuilt from `slot_dirty` on every slot mutation** -- fill, eviction, stand-down, slab eviction, shrink,
device migration and re-arm all set it, and `devremap_free_locked` runs with `free_table_buffers_locked`.
A stale map reads the wrong arena slot, and the `slots + e` cold encoding additionally depends on the slot
COUNT, so a slot-count change without a rebuild would read past the master.  `devremap_free_locked` bumps
`ggml_cuda_graph_mem_freed` (the map and lag buffers are kernel arguments of a captured graph).  The lag
buffer is guarded by a system-scope atomic flag (one block per job) or a per-table event (wider batches),
and the `dr_lag_n` counter is zeroed whenever the buffers are freed, so a stale lag can only perturb the
*policy*, never a read address.  `ENVIRONMENT.md` §1.2 carries the r40 "removed, do not reintroduce" note
for the old device-*policy* family and why this is not it.

**`--host-experts pool` reads a pool-resident miss in place (r203, same fold).**  A pool-backed table has a
pageable master, so the in-place cold read stood down and every decode miss was an arena fill even when the
expert was already in the pinned pool (`colds=0` measured).  `MOE_HOST_POOL_COLD` makes such a miss cold from
its pool slot through the pool's device alias (cold id `n_res + pool slot`; `moe_cache_get_cold` returns the
pool base with the pool's channel stride and the table's row pitch), i.e. the same bytes the fill would have
copied.  Safe because the demand path never evicts a pool resident, a retiring slot (`slot_state == 2`) is
excluded, and the eager path it rides synchronizes the backend before the hook -- that sync is the guard, so
a change that removes it from the eager path must gate this path too.  Single visible device, single slice.
Measured: fills 21747 -> 15270, 8445 misses served in place, **+4.8 % TG** (57.6 -> 60.4 t/s), text
identical.  `MOE_HOST_POOL_COLD=0` restores the fill-every-miss arm.

Known scope limits (accepted, `TODO.md`): the pool-cold arm keys on `ggml_cuda_info().device_count == 1`,
so on a multi-GPU box running a one-GPU arm it is silently off (pin `HIP_VISIBLE_DEVICES` to one device).

**A host expert table's routing must be computed before the split that prunes its upload (r197, `TODO`
#129).**  `ggml_backend_sched_copy_input` prunes a host master's per-op upload to the experts the routing
selects, and that readback happens when the split's INPUTS are copied -- i.e. before the split computes.
This block looks for **any** `MUL_MAT_ID` consumer of the input inside the split (upstream only considers the
split's *first* node, and that generalisation is what lets one merged split hook the gate, up and down
tables), while the split pass only starts a new split for a host weight once the split already has an input
to reuse.  A config whose attention runs on the host (`-nkvo`) defeats that: the GPU split begins at the
attention output with no input yet, swallows the router chain and the routed op, and the readback then
returns the previous split's leftover memory -- `GGML_ASSERT(id >= 0 && id < n_expert)`, or, when a stale
word happens to land in `[0, n_expert)`, a **silent wrong-expert selection**.  Two parts, both here: when
the ids root (the `view_src` chain of `node->src[2]`) is a node of the split being copied, every
routing-dependent shortcut stands down (the pruned upload, the device gather and `moe_cache_update`) and the
general input path copies the whole table -- which is upstream's behaviour for that shape; and the split
pass starts the split AT such a routed op, after the router chain (`next_moe_layer` keeps one layer's routed
ops in the split that starts there), so the pruning and the expert cache keep working.  Measured on one
R9700, a 24-layer 128-expert MoE at `-ncmoe 24 -nkvo`: abort -> **24.6 t/s** (stand-down alone) -> **82.7
t/s** (both), against **176** with the KV on the GPU.  Text is byte-identical between the `-nkvo` and
default arms on bailingmoe3, 35B-A3B (also under `draft-mtp`), gpt-oss-20b, Flash-Next/qwen4exp and a dense
4B, and identical to the pre-r197 default arms.  The stand-down is also the invariant for the existing
multi-consumer case, whose in-range garbage ids would otherwise be silent.

**The verify batch of a multi-slot step is bounded by the band as a WHOLE (r153, issue #234).**
`LLAMA_SPEC_ALIGN` (block 02) makes every sequence that verifies in one step present the same row count, so a
batch is `n_drafting * (n_draft + 1)` rows.  The aligned depth is therefore bounded by
`band / n_drafting - 1` rows per sequence, read once from `llama_model_moe_cache_band()` after the target
context exists: two slots at depth 7 already fill the RDNA4 band (16), so a two-slot verify aligns to 7 each
rather than 15 each.  A sequence already deeper than that keeps its own cap -- the bound may only WIDEN a
shallow draft, never shorten a deep one, which is what keeps a single-sequence run bit-identical.  Measured
with the bound on 35B-A3B `-ncmoe 40`, 2 GPU: 116.2 -> 149.0 t/s (+28 %), split decode steps 58 -> 0.  The
value lives in block 09 because that is where the band API is (a block may not use a later block's symbol,
which the per-block build gate enforces).

**A band is answered only for a model with HOST-RESIDENT experts (r176, TODO #117).**
`llama_model_moe_cache_band()` used to consider `moe_host_expert_bytes` and then **fall back to
`model->devices`**; because the CUDA device hook is `moe_cache_enabled() ? moe_cache_max_tok_dev(dev) : 0`
and `g_enabled` is **true by default** (unset `MOE_EXPERT_CACHE_MIB` = AUTO), a **dense** model on a single
GPU — or any `-sm layer` run — answered **16** with no routed expert anywhere, against this function's own
"0 when the cache is not active" contract (a `-sm tensor` model was already 0, because the Meta device has no
band).  Two consumers then acted on a band that does not exist: `common`'s n-gram clamp cut a 64-token draft
to 15, and `align_draft_caps()` (block 12) bounded a multi-slot adaptive verify width to
`band / n_drafting - 1`.  The function now considers `model->moe_host_expert_bytes` alone, so it is empty
exactly when no routed expert is streamed from a host master (r97: a pageable CPU-only placement is
deliberately not accounted).  The device hook's `moe_cache_enabled()` gate is kept, so a slab-declined or
arena-released host-expert config still reports 0 (the r129 / TODO #96 case); `moe_cache_has_tables()` is
deliberately **not** used here, because tables register during the first compute, after
`llama_init_from_model`, which is exactly the r129 bug.  Measured (dense 27B UD-Q4_K_XL, 1 GPU,
`ngram-mod --spec-ngram-mod-n-max 64`, `prompts/recall.txt`, `-n 200`): **142.1 -> 186.8 t/s (+31 %)** with
the text byte-identical, and the pre-fix build's `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` control reads 187.9, so
the whole delta is the clamp.  `-ncmoe` / `-cmoe` configs still clamp (verified incl. `-sm tensor`), and
`MOE_EXPERT_CACHE_MIB=0` still reports no band.  `WORKLOG.md` 261029.008.

**Contents.**

* **A device allocation budget, and an over-budget `--fit` that says so (r135, issue #219).**  Two things,
  both at the *decision* layer rather than in a kernel:
  * `GGML_CUDA_MIN_FREE_MIB` (opt-in, default `0` = off) is a device-wide free-VRAM floor inside
    `ggml_cuda_device_malloc`, the single choke point every device allocation shares: an allocation that
    would leave less than the budget free returns `cudaErrorMemoryAllocation` exactly like a driver OOM, so
    the existing fail-soft chain (arena shrink/release, pool flush + retry, FA staging stand-down) runs
    unchanged and the run ends with a message instead of consuming the last bytes.  It exists because a
    **Windows/WDDM** allocation that over-commits is *not* refused: it is paged into shared system memory and
    the loss surfaces later as `unspecified launch failure` or a bugcheck inside `dxgkrnl` (issue #219: a
    16 GiB 9070 XT, the 11.29 GiB GSQ-RCO-IQ3_S 27B + mmproj + MTP at `-c 65536`).  `cudaMemGetInfo` still
    reports the real budget there, so a floor below the card's usable VRAM keeps the process off that path.
    **Inert while the movable-boundary slab is armed** (the slab parks nearly all free VRAM by design), so an
    `-ncmoe` MoE run is unchanged; it is a *budget*, never a reservation.
  * `common/fit.cpp`'s failure path now prints an ERROR block with the arithmetic (projected vs. free vs.
    target margin, the shortfall, the context's MiB and MiB/1024 tokens) and the ways out, instead of the one
    `W common_fit_params: failed to fit params ...: n_gpu_layers already set by user to 99, abort` line that
    read as harmless.  It also distinguishes "lower `-c` by about N tokens" from "even the smallest context
    cannot make this fit, lower `-ngl`".  Nothing is changed automatically and nothing is refused: the
    numbers are the fit's own, and `--fit` still only adjusts arguments the user did not pin.
* `moe_cache_release_arena()` only reports a release when it actually freed bytes (r135): on a dense model
  it logged `released the MoE expert cache arena (0.0 MiB) ... (experts come from the host)` and returned
  true, which sent a reporter looking for an arena that does not exist (same class as r125's gated flat
  draft reserve, issue #216).
* `moe-expert-cache.cu` (~5.1k lines) + `.h`: the cache engine — arena sizing, per-layer uniform slots,
  the slot remap, the routed-MMVQ decode/verify band (`moe_cache_band`), admission/heat policy, the
  persistent cross-run heat map (`MOE_EXPERT_HEATMAP`) and prefill cache staging.
* **A re-arm is layer-uniform or the layer is stood down, and a returned arena range is never kept (r168,
  TODO #70).**  `moe_cache_rearm()` re-allocates what a slab boundary move evicted, to the slot count the
  sizing pass settled on.  Its retry loop could exit with a **mixed-slot layer** (it set
  `target = min(achieved)` on the last attempt and the loop ended, leaving freshly allocated tables at their
  own counts); the sizing pass forbids that -- the gate+up fused kernel seeds the gate lane from the **up**
  table's remap -- and answers it by disabling the cache, which a re-arm must not do (the graphs already
  reference the tables, and a late `g_enabled = false` desynchronises the MTP draft from the target).
  `moe_cache_rearm_layer_locked()` now converges a layer on ONE count (retrying at the largest count any table
  reached, halving when one cannot hold a slot) and, if the retries run out, stands the **whole layer** down
  -- never a mixed layer; a smaller-but-uniform count is accepted with a warning, and a final sweep re-checks
  the touched layers (the sizing safety net, made local).  Two invariants it relies on: a stand-down must
  clear `t.arena`/`t.arena_reserved` once the range goes back (a slab range lands on the slab's free list,
  where the work pool or another table can take it, while `alloc_table_locked()` reads a non-null `t.arena`
  as *already allocated* -- `stand_down_table_locked()` used to keep it, so a re-arm of an
  allocation-OOM-dropped table claimed slots in a range it no longer owned), and any path that frees a
  table's device buffer bumps `ggml_cuda_graph_mem_freed` (`moe_cache_evict_slab_range()` hands ranges to the
  slab directly, so it bumps for itself).  `MOE_EXPERT_CACHE_VALIDATE=1` reports both states
  (`MIXED-LAYER` = a role stood down and the served tables still agree; `NONUNIFORM-LAYER` = two different
  non-zero counts).  Record: `archive/work/todo70-rearm-wideaux/`.
* **The cache has an offline policy instrument, and two host-tally gaps are closed** (r110, TODO #61 P1/P3
  + TODO #64.1/.2/.3).  `MOE_EXPERT_TRACE=<path>` dumps the raw routing of every `mul_mat_id` for
  `scripts/moe-trace-replay.py`, which scores the shipped F-LRU, an oracle static top-f, a per-layer
  capacity quota and pass-start pre-admission against a real trace — the instrument the coverage campaign
  lacked (it had only an aggregate heat vector).  DECODE records come from the scheduler hook (ids
  already on the host); PREFILL records come from the device entry and pay one readback per op, which is
  legal because a prefill ubatch is never captured.  `pool_rank_from_prefill_locked` gives the
  `--host-experts pool` warm order the *device* prefill tally (the eager `prefill_count` never runs under
  the staged prefill, so the pool was warmed in expert-id order and `pool_rerank_from_tally_locked` was
  dead); and a fully-resident (identity) table's decode routing — invisible to every host tally — is now
  tallied device-side, **sampled and weighted** (`MOE_EXPERT_HEATMAP_DECODE_SAMPLE`, default 64).  The
  heat mirror is pre-allocated at sizing because the decode band is a captured graph and a `cudaMalloc`
  there aborts the capture.  The report gains **metric parity**: the resident set's covered mass of this
  session's routed mass, the global-top-f static bound at the same slot count, and (already there)
  expert MiB/token.  `moe_cache_observe` (dead, superseded by `moe_cache_heat_observe`) is removed.  No
  numerics change: same-seed text is byte-identical.  `WORKLOG.md` 261022.005.
* **The verify-width band clamp (r123, TODO #59).**  The arena serves a routed `MUL_MAT_ID` from VRAM only
  up to its decode/verify band (`moe_cache_max_tok_dev()`; 16 tokens on RDNA4, 8 on RDNA3/NVIDIA); above it
  `moe_cache_update_host()` returns false and the routed experts are streamed from the pinned host master on
  every step.  A speculative verify batch is `n_draft + 1` rows, and while `--spec-draft-n-max` is bounded to
  15 (a 16-row verify) by `LLAMA_SPEC_DRAFT_N_MAX_CLAMP`, the **draftless** n-gram speculators had no width
  bound at all (`ngram_mod.n_max` 64 / `n_min` 48 by default, the ngram-map family's `size_m` 48) — so the
  README's own `draft-mtp-adaptive,ngram-mod` pick put every ngram step outside the band on a host-expert
  config and lost 2.3x decode.  This block adds the `moe_cache_band` device query (appended to
  `ggml_backend_device_i`; CUDA answers `moe_cache_enabled() ? moe_cache_max_tok_dev(device) : 0`),
  `llama_model_moe_cache_band()` (the smallest band over the host-expert devices), and
  `common_speculative_clamp_ngram_n_max()`: `common_init_from_params` clamps an **enabled** n-gram
  speculator to `band - 1`, lowering `n_min` with it.  Clamped vs
  unclamped, 2x R9700 Flash-Next IQ4_NL `-sm tensor -ncmoe 36`, `prompts/recall.txt`, `-n 2000`: 168.3 vs
  73.9 t/s (`draft-mtp-adaptive,ngram-mod`), 160.2 vs 74.1 (`draft-mtp,ngram-mod`); the cliff is one row
  wide (159.2 t/s at a 16-row verify, 55.0 at 17).  **Not** applied to a device-resident model or the CPU
  expert path (band 0): there the wide draft is the win (378.7 vs 144.7 t/s).  `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0`
  restores the uncapped draft; a `draft.n_max` above the band warns instead of clamping (the 15 bound is a
  correctness one).  **Since r129 the clamp runs after `llama_init_from_model`, on the arena's real state
  (TODO #96, block 09 + block 10).**  the slab is only *armed* at model load — it is created or declines at the first compute-buffer allocation,
  which is inside the context's widest reserve.  Clamping on the model-load answer clamped a slab-declined
  config that runs the stock streaming path with no arena at all (issue #214's reporter class) and cost it
  the wide draft (2x R9700 control: 67.6 vs 52.2 capped t/s).  The deferral is free (the speculator is built
  from `params.speculative` much later) but it means the clamp can no longer refresh `cparams.n_rs_batch`:
  that field stays the **unclamped** safety bound, because it is the widest batch a partial rollback can
  land in (and floors the chunked-GDN threshold), while the slab's narrow verify reserve takes the
  band-limited width instead (`llama_context::verify_batch_width()`), so a live arena reserves exactly what
  it did before (measured: narrow floor 1024/1280 MiB and `identity fast path ON` 142/142, with and without
  the speculator; the arena arm still clamps `64 -> 15`).  A decline that happens *after* the context exists
  (an OOM-yield arena release) still leaves the draft clamped — `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` is the
  escape.  The slab's decline messages now name every knob that can make it fit (`MOE_EXPERT_CACHE_MIB` /
  `GGML_CUDA_SLAB_MIN_ARENA_MIB` for the floor, `GGML_CUDA_SLAB_RING_MIB` for the H2D-ring hole,
  `GGML_CUDA_SLAB_RESERVE_MIB` for the reserve) and print the ring term it previously only mentioned.
  `GREEDY-PURITY.md` §50, `ENVIRONMENT.md` §3, `WORKLOG.md` 261024.002 / 261024.006 / 261024.010.
* **A band only exists where host experts do (r176, TODO #117).**  The clamp above, and the multi-slot
  alignment in `align_draft_caps()`, now see `0` for any model with no host-resident experts: see block 09's
  entry for the predicate, the measurements and the traps (`moe_cache_enabled()` is true by default with no
  table; `moe_cache_has_tables()` is the r129 bug and must not be used; the slab/arena state still gates it).
  `WORKLOG.md` 261029.008.
* **int64_t min/max in this file use local helpers, not `std::min`/`std::max` (r124, issue #217).**  With
  HIP on Windows (clang + the MSVC STL) clang's force-included `cuda_wrappers/algorithm` and the MSVC STL
  `<utility>` each define an inline `std::min<long long>`; `int64_t` is `__int64` there, the `enable_if`
  that makes clang's version win overload resolution is not part of the mangled name, and a translation
  unit that instantiates both fails to codegen with *"definition with same mangled name"* (r123 failed the
  reporter's Windows build in this file's host pass; r92 was clean, r112 introduced the first such call).
  `moe_cache_min_i64()` / `moe_cache_max_i64()` replace all three int64_t sites (the auto-arena floor and
  the heat-coverage bound here, and block 10's tables-registration slack guard).  Behaviour is unchanged —
  greedy text is byte-identical on the dense and host-expert arms, prefill-logit KLD 0.000000, MTP
  acceptance 0.73881 — so any NEW int64_t min/max in this file must use the helpers, and a rebuild of a
  HIP TU that adds one would reintroduce the Windows failure.  `WORKLOG.md` 261024.004.
* **Whole-table identity placement (r117, TODO #93).**  A table takes the decode band's identity fast
  path only at its FULL slot count (`t.slots == t.n_experts`); one slot short is not "almost" — it takes
  the remap path and pays the per-token routing readback like a table with no arena at all.  Each table's
  arena is rounded up to the slab arena unit, so a budget that is "just enough" in aggregate lands one unit
  per table short and **every** table drops off the path: measured on the field config (GSQ `-ncmoe 48
  -sm tensor`, 3 devices, plain decode) `MOE_EXPERT_CACHE_MIB=13800` → arena 40845.1 of 40925.0 MiB,
  0/432 identity, **46.7 t/s**, vs `15000` → 432/432, **65.0 t/s**.  `alloc_all_locked` now prefers
  whole-table identity: when at least 96 % of the layers can hold their full slot count, those layers get
  full residency and the coldest few are dropped to 0 slots (an existing, supported state: they run through
  the scheduler's host copy) instead of spreading the shortfall.  The drop is capped at ~4 % of the layers
  (a dropped layer is not free — 1 of 48 is +27.6 %, 6 of 48 is −30 %), placement never exceeds
  `MOE_EXPERT_CACHE_MIB`, and it is bit-identical: same text and bit-identical MTP acceptance
  (`MOE_EXPERT_CACHE_IDENTITY_PREF=0` restores the uniform placement).  `WORKLOG.md` 261023.002.
* `ggml-cuda.cu`: the **op-offload H2D staging ring** (three invariants: prefill is never graph-captured,
  the width gate is floored at 64 tokens, the arena lives outside the compute-graph reserve), the pinned
  `ROCm_Host` master, the per-device host buffer types, and the base of the async graph-input ring.
* **The Meta split uses the loader's chunked async ring** (r122, with block 00): the
  `ggml_backend_dev_is_meta(dev) -> nullptr` bail is gone; the Meta async setter serves a mid-row chunk
  with `write_partial_row` (and falls back to the sync setter for a segmented/PARTIAL table), and the
  event proxy fans the staging-ring backpressure out to the simple devices, so `--load-mode none` under
  `-sm tensor` streams through the 4 MiB pinned slots instead of the whole-tensor pageable `read_buf`
  (cold Flash-Next load 32.2 s -> 13.6 s).
* `ggml-backend.cpp`: the scheduler's staged cross-backend upload (including the cross-device
  `wait_before_overwrite` ordering that fixed issue #103), routed-expert rebalancing and env-read caching.
* **A second consumer of a staged expert table gets its own split** (r96, PR #186): a cache-managed
  `MUL_MAT_ID` whose weights are already an input of the current split now starts a new split, so the
  existing re-registration path stages them again for that consumer's own routing.  qwen4exp's gathered
  logits tail and its full-row MTP export tail are the motivating pair: merged, neither consumer could be
  served by the expert cache and the scheduler's guarded fast paths declined, so the last layer's whole
  table was copied for every prompt chunk on `-sm tensor` (-30 %/-47 % TTFT on follow-up turns,
  bit-identical text) and on `-sm layer` the second consumer read the arena slots staged for the first
  routing, which silently produced non-greedy-pure text with 0.00000 draft acceptance.  `GGML_SCHED_MOE_DUP_SPLIT=0`
  restores the merged split.  Invariant and evidence: `GREEDY-PURITY.md` §43.
* `ggml-backend-meta.cpp`: `MIRRORED` tensors serving arbitrary byte ranges and the per-device split of a
  host expert table.
* **The meta `get_tensor_async` reads an arbitrary row range** (r96, PR #189): it shares the synchronous
  read's split arithmetic instead of asserting `offset == 0` and one segment, so a caller can batch many
  per-device reads behind one `synchronize`.  Calls that worked before issue the same copies as before.
  The caller is block 11's batched state save.
* **The split-state pre-warm walk allocates no heap nodes** (r100, issue #184).  On a cache miss the
  per-buffer-context split-state cache walks the not-yet-cached ancestors on an explicit stack (the
  recursive form overflowed the thread stack on a 1200-node chain); its `wstack`/`wseen` containers
  allocated one heap node per visited tensor on **every graph re-allocation** — a 4-stream server
  re-allocates ~230 times per run, and the reporter's profile put 94 % of the `malloc` calls under
  `ggml_gallocr_alloc_graph` in that set, ~4.7k split states and ~150k tensors per re-allocation.
  Both containers now use a reused per-thread `monotonic_buffer_resource` (1 MiB, released per walk).
  Only the outermost call walks, so the resource is never re-entered; the walk visits the same tensors
  in the same order and computes the same states (proven with `GGML_META_SS_VERIFY=1`, which recomputes
  every outer hit), and an exhausted arena falls back to the upstream allocator, so nothing is capped.
  `GGML_META_SS_SCRATCH=0` restores the heap containers (A/B and bisection only).
* **Views created between graph evals get a caller-owned scratch container** (r150, issue #232, the
  reporter's patch).  `llama_kv_cache::park_move` moves a run in chunks no larger than the gap to its
  destination and creates 2 views per (layer, K/V) per chunk, so a nearly-full unified cache (two
  ~258k-token prompts in 524288 cells) parks in dozens of tiny chunks — and parking runs *between*
  graphs, so those views landed in the per-generation compute container and nothing reclaimed them.
  `ggml_backend_meta_set_scratch_views(true)` routes views that are not static tensors into
  `stc_scratch`, and `ggml_backend_meta_reset_scratch_views(buffer)` drops them once the chunk's copies
  are done (`park_move` does this per chunk, then disables it); the copies are synchronous — a meta
  buffer has no `.cpy_tensor`, so `ggml_backend_tensor_copy` stages through the host — so no scratch view
  outlives its reset.  Static tensors still resolve to `stc_static`, so nested-view lookup is unchanged.
  `gate-park-idle-slots.sh` PASS (park == lazy-off byte for byte on all four requests).  `TODO` #110:
  `llama_io_read_device`'s state-restore views still take the per-generation container.
* **A simply-tensor container grows; it is not a fixed budget** (r150, issue #232).  Every Meta buffer
  keeps a "compute" container of per-device *simple tensors* for the tensors a generation registers — a
  graph's own views plus the views an out-of-graph operation creates (`llama_io_read_device` registers one
  per KV cache chunk per layer on a state restore, `llama_kv_cache::park_move` one per move chunk).  The
  container used to be one context sized `128 * n_static_tensors` views, so one generation could exhaust
  it: the 48-tensor attention KV buffer budgeted **6144** views (a 2,260,992 B ggml context) and the
  server aborted in `ggml_new_object` with `not enough space in the context's memory pool (needed
  2261360, available 2260992)` — exactly one tensor short of the pool.  The reported generation was a
  `park_move` (above); a long-context state restore (~0.012 views/token: 432 for a 35k-token checkpoint)
  is the other contributor, and any new out-of-graph view source is covered by the growth.  A full context is now followed by another one (allocation is
  `ctx_for_new_tensor()`), and a rebuild resets and **releases** every context in the list, so the budget
  is a first-context size rather than a cliff.  `GGML_META_COMPUTE_HEADROOM` (default 128) overrides that
  size — a diagnostic/testing knob: a smaller value only makes the container grow sooner and never
  changes an output.  `ggml_new_object`'s overflow warning now also logs the context, its `mem_size`,
  its object count and the bytes used, so a report of this class is diagnosable from the log alone.
  Verified on 3x R9700 + Flash-Next GSQ-RCO IQ3_XXS (`-sm tensor`): the park trigger aborts r149 at the
  default budget and completes here (zero growth with the scratch scoping in place, 6144 -> 12288 without
  it); `GGML_META_COMPUTE_HEADROOM=1` aborts the r149 build at `sched_reserve` and runs to completion on
  this build (105 growth events), with byte-identical greedy text in every arm.
* **The split-state cache hit returns the entry's state instead of copying it** (r132, PR #213).  The hit
  path used to build the state in a named local (`split_state_entry::get()`) and copy it into the return
  slot, which is a ~2 KiB copy in a function that runs for every tensor of every graph allocation under
  `-sm tensor`.  It now returns `entry.get()`, a prvalue of the return type, so the state is built once in
  the caller's object.  Same values and the same assertions.  On this tree it is codegen-neutral — clang 24
  and clang 23 at `-O3` already build the entry directly in the return slot (identical function size, one
  instruction's worth of difference in the whole function) — so it is taken as an explicit-intent change,
  not for the speedup the PR measured on its own build.
* **The HIP graph cache key is the first node's `data` pointer, not its tensor object** (r115, TODO
  #51).  `ggml_cuda_graph_get_key()` was `{ nodes[0], n_tokens }`, and under the Meta backend every
  per-device subgraph is built from *simple tensors* that `ggml_backend_meta_graph_compute` re-creates
  on each rebuild — so the object address changed on every reallocation even when the graph was
  structurally identical and pointed at the same device buffer, and the graph cache **never warmed**
  (measured 3x R9700, 27B Q8_0, `-sm tensor`, 300-token decode: 3954 keys / 1206 captures / 484 ms of
  `hipGraphInstantiate`, against 67 keys / 9 ms with `-sm layer`; the host-side bursts were the
  user-visible 200-450 ms 3-GPU stutter).  Keyed on `data`: 144 keys / 45 captures / 15 ms, **+41 %**
  decode at 3 devices (+6 % at 2, unchanged at 1), byte-identical text and bit-identical MTP
  acceptance at every verify width.  Safety: `ggml_cuda_graph_update_required()` still re-checks every
  node's op/type/shape/`data`/`src` pointers before a replay (a meta rebuild assigns a fresh
  `cgraph_uid`, which forces the check), so a key collision costs a recapture, never a mismatched
  replay.  `GGML_CUDA_GRAPH_KEY_DATA=0` restores the object-address key.  This is *not* the dropped r25
  PR #106 patch 0002, which **added** data pointers to the object key and fragmented a warm cache.
  Record: `../archive/work/graph-key-churn/`.  The key gained a **weight identity + structural signature**
  in r149 (above): the first-node `data` pointer alone still collides across the per-layer subgraphs of a
  split graph, which is the defect that PR #106's key could not fix.  A key that does not name every
  dimension a subgraph varies in is not a fix — that is the lesson both halves of this story teach.
* **The copy callback is invoked again, so the upstream MoE cache works** (r99, TODO #56).  This block's
  rewrite of `ggml_backend_sched_copy_input` kept no `sched->callback_copy` call, and that call is the only
  driver of `llama_moe_cache::copy()`: with `--moe-cache-mib` set the slot map stayed `-1` while the graph had
  already swapped the routed op's `src0` for the cache bank, so `MUL_MAT_ID` indexed one expert below the bank
  (wrong values on gfx1201, an HSA fault on gfx1151).  Base's call is restored at base's position and in the
  branch with no `MUL_MAT_ID` consumer (where the slot map lands); the delivery's own expert branch is
  unchanged.  `sched_copy_experts` answers only for the upstream cache now.  Inert unless
  `--moe-cache-mib` is set; the upstream engine is correct but slower than `MOE_EXPERT_CACHE_MIB`.
* **The host-expert account keys a real device only** (r97): `moe_host_expert_account` skips a host
  buffer type that carries no device.  `ggml_backend_cpu_buffer_type()` is the plain **pageable** CPU
  buffer and has `.device = NULL` upstream (the FIXME in `ggml-backend.cpp`), so a CPU-only run, or
  `--override-tensor exps=CPU`, used to insert a null key and abort in `ggml_backend_dev_memory()` when
  the context auto-sized the staging ring — a CPU-only model could not create a context at all, and
  `test-llama-archs -a deepseek4` aborted on its first row.  A pageable master is not served by this
  block's machinery (the slab arena, the ring and the cache read the master in place from a device
  kernel, which needs a pinned buffer): it is copied by the upstream `llama_moe_cache` engine instead,
  so it must not be accounted.  Only a pinned host buft — which always carries its device — is
  accounted, and the per-device distribution is unchanged.
* **The `--fit` draft reserve charges only the KV a draft OWNS** (r101, campaign 0007).
  `common_draft_reserve_estimate()` charged the speculative draft a full `n_ctx` KV — 18022 of the
  19152 MiB it asked for per device — for a drafter that allocates **none**: `gemma4-assistant` declares
  `attention.shared_kv_layers = 4` and `llama_kv_cache`'s `layer_share_cb` reuses the target's tensors.
  It now reads `attention.shared_kv_layers` and charges `kv_layers - shared` (0 for
  `gemma4-assistant`); the indexer-cache term follows the same count and the log line reports the owned
  layers.  At the default 262144 context, gemma4 26B-A4B `-ncmoe 40 -sm layer` MTP n3: reserve
  19152 -> **1129 MiB**, arena 13752 -> **23141 MiB**, slots 4564 -> **7680**, `h` 0.897 -> **0.0000**,
  gen **41.0 -> 93.2 t/s** (plain 74.6).  Non-shared drafts are arithmetically unchanged (qwen4exp
  sidecar: `weights 2647 + kv 563 + compute 512 + slack 192`); an embedded MTP head never reaches the
  estimator.  (`common/draft-reserve.cpp`)
* **A loaded-but-unused draft model is not loaded, and gets no context** (r101, campaign 0011).
  `has_dft()` only means "`-md` was given", so `-md <gemma4-assistant> --spec-type none` used to load the
  model, create and warm up its context — which aborted, because the drafter's graph always needs the
  target's hidden state (`GGML_ASSERT(ubatch->embd_state)`).  `common_params_speculative::uses_dft()` /
  `has_aux()` (`common/common.h`) now gate the model load, the context creation, the MoE draft reserve
  and the preload on *an enabled type that actually drafts with a model*; `common_init_from_params`
  prints one WRN.  `-md X --spec-type none` is a working "plain baseline with the drafter loaded" and an
  unused drafter costs no VRAM.
* `common/fit.cpp`: `--fit` reserves the arena and the slab headroom per device.
* **The draft reserve is only computed, and only logged, for a model that has host-resident experts**
  (r125, issue #216).  The reserve is the `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` input to
  `llama_model_moe_cache_preflight()`, and that preflight walks `model->moe_host_expert_bytes` — which is
  empty for a dense model (no `-ncmoe` / `-cmoe`), so the call is a no-op there.  It was still derived
  and printed on every speculator run, and the line
  `common_init_result: keeping the flat 3584 MiB draft reserve (...)` then read as evidence that the
  slab or the expert-cache arena was involved in a **dense** model's VRAM failure (it was not: nothing
  consumed the value).  `common_init_result` now gates the whole block on
  `llama_model_moe_host_expert_dev_count(model) > 0`.  No behaviour change where the reserve is used:
  the derived value, the flat fallback and the `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` override are identical.
* **The draft reserve charges the draft CONTEXT at the main model's KV types** (r125, block 02's draft
  KV change).  `common_draft_reserve_estimate()` is called with the main `-ctk` / `-ctv`, which is now
  what the draft context allocates too, so `--fit`'s fallback reserve and the slab reserve match the
  actual allocation by construction.  An explicitly pinned draft type (`-ctkd` / `-ctvd`) is the one case
  the estimate does not see: it always assumes the main pair.
* Host-side: mmap reclaim (`LLAMA_LOAD_RELEASE_MMAP`), rear-fill host layers
  (`LLAMA_HOST_EXPERTS_FROM_END`), `--host-experts pool` (`MOE_HOST_POOL_MIB`), a second draft scheduler
  (`LLAMA_DRAFT_SCHED2`), compute-buffer drop/re-reserve, and the CPU MoE-offload thread cap.
* **The second draft scheduler's slot is per arch: `n_outputs` for MTP, the batch kind for DFlash**
  (r120, issue #208).  `sched_alt_select(n_outputs > 0)` fits MTP (a no-output ingest graph and a
  draft step with an output) but **not** a DFlash2 draft block, which also has no outputs -- its lattice
  is read from the nextn embeddings, not from logits (`output = !is_dflash2`).  The KV injection and the
  noise-block draft step then shared one scheduler **and** one `gf_res_prev` slot, so they overwrote each
  other's graph and the draft context rebuilt and re-allocated both on every verify step: measured **0 of
  362** draft ubatches reused, 0.27-0.30 ms of build + allocate per draft decode.  `sched_alt_for()`
  (`src/llama-context.cpp`) now returns, for `LLM_ARCH_DFLASH`, "the batch carries no target features"
  (`ubatch.embd == nullptr && !cparams.lf_dev_inject`) so the *draft step* takes the second scheduler and
  the injection (host feature rows or the F1 device injection) stays on the main one; every other arch
  keeps `n_outputs`, where the batch kind would merge MTP's ingest and step graphs instead of separating
  them.  `get_gf_res_prev()` takes the same decision, so the scheduler and the graph-reuse slot cannot
  disagree.  Measured (gfx1201): reuse 0 -> **97.8 %** (dense 27B UD-Q4_K_XL 2 GPU tensor) / **97.9 %**
  (35B-A3B Q4_K_M `-ncmoe 40`), decode **+1.4 %** / **+1.8 %** (27B 3 GPU) / **+4.0 %** (35B-A3B host
  experts) / +1.3 % (host injection) / +0.5 % (`-sm layer`) / +0.2 % (1 GPU) / +1.0-1.9 % (2-slot
  unified server, two concurrent requests), byte-identical text and acceptance everywhere, dense 4B
  coherence hash unchanged, adaptive-MTP four-axis gate identical, `LLAMA_DRAFT_SCHED2=0` reproduces the
  old behaviour.  Record: `archive/work/dflash2-draft-sched2/`.
* **The batched f32 CPY run** (`cpy-batch.cu` / `.cuh`, r95): a run of up to 16 consecutive same-layout
  f32 `GGML_OP_CPY` nodes (the per-position DeltaNet conv-state snapshots of a speculative verify batch
  are the motivating shape) as **one launch**, element copies in `ggml_cpy`'s flat order, so
  bit-identical.  Admitted on **RDNA4 and RDNA3_0 (gfx1100-1102; gfx1101-gfx1102 not explicitly
  validated)**, `GGML_CUDA_FUSE_CPY_BATCH=0` disables it, and the marker `GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1`
  prints every run.  `ggml_cuda_cpy_batch_try_fuse` (block 10) is the caller; the fusion is skipped while
  a split's concurrent events are live, but it still fires heavily on a multi-device graph (measured
  ~450 runs under `-sm layer` / ~1350 under `-sm tensor` per 60-token MTP run on 3x gfx1201).
  RDNA3_0 multi-device is an **open, accepted gap** (no RDNA3_0 multi-GPU box).
* **The appended device-iface fills.**  Every `ggml_backend_device_i` / `ggml_backend_i` entry this set adds
  (`stage_gather`, `slab_ring_set`, `slab_narrow_floor`, `slab_reserve_wide`, `slab_narrow2_floor`,
  `slab_compute_narrow2`, `slab_compute_plain`, the slab-ring hooks, `moe_cache_*`) is wired here as a real
  hook (CUDA/Meta) or NULL/no-op (CPU, RPC, every other backend).  The list is **positional**: the block
  that declares a field must also fill it in every backend, or the initializer silently shifts (r94
  block-boundary repair; previously deferred to block 11, which does not build on its own).
* **The Meta split state accepts what a per-row / whole-row layout actually means** (r101,
  campaign 0008/0009/0010).  Three related additions to `ggml-backend-meta.cpp`, each "the split does not
  straddle the reduction, so the op is local":
  * `handle_per_row` (RMS/NORM/GROUP/L2 norm, `SUM_ROWS`, `ARGSORT`/`TOP_K`) accepts an **axis-0** split
    whose every per-device extent is either empty or the **whole row**.  gemma4's full-attention layers
    have one KV head, so the 512-wide K/V row lands whole on one device (`ne_per_dev=[0,512]`); a
    straddling split still aborts.  `handle_bin_bcast` gains the elementwise case for a split tensor
    against a **MIRRORED same-shape** operand (the per-head norm weight).
  * `handle_mul_mat` accepts (MIRRORED `src0`, axis-0 `src1`) when the contraction is empty-or-whole per
    device and `out->ne[0] == src1->ne[0]` — the **rotated KV cache** path (`-ctk q8_0` =>
    `n_rot_k = 512`), where the Hadamard rotation `rot[512,512] (MIRRORED) x Kcur[512, rows]` has its
    contraction on axis 0.  The owning device computes the whole dot product; nothing is summed across
    devices.  A straddling contraction still aborts, with a message saying it needs a cross-device
    reduction.
  * `handle_flash_attn_ext` accepts a K/V slice that is **empty-or-whole on every device** (with the
    query empty-or-whole on the same devices).  Needed when one ubatch carries several sequences
    (`n_stream > 1`: `llama-server -np > 1`, `llama-batched-bench -npl > 1`,
    `llama-perplexity --kl-divergence`), where the FA K/V cache view spans the cache's stream dimension
    and the state arrives on the cache's embd axis instead of the kv-head axis.
  Net effect: gemma4 dense + `-sm tensor` works, byte-identically to `-sm layer` (`c740be0e131a` at 2
  and 3 GPU) in f16 **and** rotated q8_0, including multi-sequence batches.  `GREEDY-PURITY.md` §45.
* **The expert cache sizes on gemma4, and a fused table is cacheable under `-sm tensor`** (r101, campaign
  0002/0003/0004a/0004b — TODO #77 A/B/C):
  * `moe_host_expert_account()` counts only routed expert **banks** (`ne[2] >= 2`).  gemma4 has a
    per-layer `ffn_down_exps.scale` (512 B x 30 layers = 15360 B), which made the cache's completeness
    check unsatisfiable: no arena was ever allocated, so every per-op expert upload was a miss
    (`-sm layer` 7.9 t/s, `-sm tensor` 5.5).  The completeness slack stays `min(max(1 MiB,
    expected/4096), 16 MiB)` — deliberately much smaller than one table (35B-A3B's is 36.4 MiB), because
    a wider slack silently accepts a sizing pass with one table unregistered (-3.6 % tg128).
  * A cache table's device slot may be assembled from up to **8 ordered host ranges** (gemma4's fused
    `ffn_gate_up_exps` is `[gate_slice | up_slice]` under `-sm tensor`), and one shared
    `fill_slot_from_host()` serves both the eager fill and the identity fill so the two cannot disagree.
    `ggml_backend_moe_slice` + the extended `moe_cache_update` in `ggml-backend-impl.h`; the scheduler
    call site passes `nullptr, 0`.
  * A **row-aligned segmented `set_tensor_async`** splice (per-device `set_async`/`set_2d_async` plus the
    MMQ tail pad); mid-row ranges keep the synchronous setter (the loader's chunked upload owns a staging
    ring that may be reused on return).  `GGML_META_SEG_ASYNC=0` disables it.
  Result: gemma4 26B-A4B `-ncmoe 40` `-sm layer` **79.0** / `-sm tensor` **74.7** t/s (was 7.9 / 5.0).
  The trap this fix invites is recorded in `GREEDY-PURITY.md` §45.
* **The async graph-input ring is keyed per ORIGIN** (r101, campaign 0005): its ordering state
  (`pending`/`ev`/`q`/`flush_ev`/`done_ev`) was per **device**, but a stream is a `llama_context`, so
  MTP's draft and target contexts consumed each other's queued uploads and a context could replay a graph
  without waiting for its own copy.  A genuine cross-context bug (the MTP run-to-run nondeterminism that
  looked like its symptom was ROCm's hipBLASLt per-start solution pick, issue #67 — see
  `GREEDY-PURITY.md` §44).
* **The iSWA KV builder expands `q, v, k`** (r101, campaign 0001 — TODO #76): the r81 rope + FWHT fusion
  folds the two q8_0 K/V cache `SET_ROWS` launches into the fused rope kernel, but
  `build_attn(llm_graph_input_attn_kv_iswa*)` expanded `q, k, v`, so the V block sat between the K
  rotation and the K cache write and the matcher declined — **inert on every iSWA arch**.  Now mirrors
  the non-iSWA builder.  Measured on gemma4 12B q8_0: `k_set_rows_q8_0_group` 1152 -> 192 launches,
  `rms_norm_mul_rope_f32<...,kv=true>` 0 -> 480, total 13500 -> 12540; greedy text and
  `test-logits-width-probe` identical, `llama-bench` tg128 +0.6 %.  Known gap: gemma4's 512-wide
  full-attention layers stay outside the fused kernel (its `fwht_n` set is `{64,128,256}` and the V-row
  write assumes `ncols <= 256`).
* **A speculative verify batch never exceeds `n_batch`** (r101, campaign 0006): a verify batch carries
  `n_max + 1` tokens in one `common_batch::get()`, capped by `llama_n_batch()`, so `-b 1` aborted in a
  server worker after the model had loaded.  `common_init_from_params` raises `n_batch` to `n_max + 1`
  with a WRN (a logical-batch hint; `-ub` still bounds the per-launch memory).
* **The pre-loaded draft-model handoff** (r86; ownership completed in r94):
  `common_speculative_load_model`, the 4-argument `common_speculative_init_result` / `init_from_params`,
  and `common_init_result::take_draft_model` live in this block together, so the draft model is resident
  before the target context sizes the slab.  A caller and its definition must not be split across blocks.

**Knobs.**  `GGML_CUDA_GRAPH_KEY_WEIGHT` (r149, kill-switch: `0` restores the first-node-only key),
`MOE_EXPERT_CACHE_MIB` (auto default; `0` disables), `_MIN_MIB`, `_MIN_RES_PCT`,
`_RESERVE_MIB`, `_AUX_RESERVE_MIB`, `_UNIFIED` (APU opt-in), `_VALIDATE*`, `MOE_EXPERT_HEATMAP[+_DIR]`,
`MOE_HOST_POOL_MIB` / `MOE_HOST_POOL_FRAC`, `GGML_MOE_CACHE_MAX_TOK`, `GGML_MOE_GATHER_ONCE`,
`GGML_CUDA_SLAB`/`_RESERVE_MIB`/`_HEADROOM_MIB`/`_MIN_ARENA_MIB`/`_RING_MIB`/`_CHUNK_MIB`,
`GGML_COMPUTE_BUFFER_MARGIN_PCT`/`_CHUNK_MIB`, `GGML_CUDA_ASYNC_INPUT`, `GGML_SCHED_STAGE*`,
`GGML_STAGE_*`, `LLAMA_LOAD_RELEASE_MMAP`, `LLAMA_DROP_COMPUTE_BUFFERS`, `LLAMA_DROP_EXTRA_RESERVE_MIB`,
`GGML_CPU_MOE_OFFLOAD_THREADS`, `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD`, `LLAMA_TENSOR_HOST_BUFT`,
`GGML_HIP_GRAPH_FORCE_UPDATE`, `GGML_META_SS_VERIFY`, `GGML_META_SS_SCRATCH`, `GGML_SCHED_MOE_DUP_SPLIT`,
`LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP`.

**Invariants.**  Host experts are always pinned; a partially failed cache **falls back wholesale**; no
arena is freed while a kernel carrying its address is in flight; the slab is **MoE-only** (armed solely by
the MoE preflight — never enable it for a dense model); a `-ncmoe` host table is split per device, not
mirrored (`GGML_META_SPLIT_COPY=1`); a staged expert table gets **one consumer per split** (a second
`MUL_MAT_ID` consumer of the same weights starts its own split, because the cache arena can only hold one
routing's slots — `GREEDY-PURITY.md` §43); a verify batch must stay inside the arena band when the arena
serves the host experts (`GREEDY-PURITY.md` §50).

**Gates.**  Host-expert coherence (`-ncmoe`/`-cmoe`, and `cache-on == cache-off` text), `test-llama-archs`,
`MOE_EXPERT_CACHE_VALIDATE=1`, `scripts/check-gpu-faults.sh` on the run's journal + log.

**Diagnostics (r144).**  The exit report's `expert H2D traffic … MiB/token` divides a **decode-band op
census** (`g_band_ops`, counted at the one place every routed MoE op passes through, whatever path it
then takes) by a table count snapshotted **at sizing** (`g_band_n_tables`).  Both matter: the previous
`g_takeover / n_tables` read 0 on the identity fast path (and after the r143 teardown dropped the
tables), so every run printed `over ~0 tokens = 0.00 MiB/token`.  Do not "simplify" it back to a live
`g_tables` walk at report time.  A *short* run's MiB/token is mostly warmup (25.4 vs 3.3 MiB/token for
the same config at 200 vs 2000 tokens) — quote the 2000-token number.  `GGML_CUDA_GRAPH_TRACE=1` also
reports how every `graph_compute` call was served (`direct` / `replay` / `capture` /
`properties_changed`); a large `direct` with a matching `properties_changed` means the replay identity is
churning — it is not the same as "graphs are on".  Since r149 that churn is fixed for decode (see the
next item) and the census to expect is `direct` ~= 0 with `properties_changed` in the tens.

* **The graph-cache key carries the subgraph's weight identity, so a split graph's per-layer subgraphs
  replay (r149, TODO #107 closed).**  `ggml_cuda_graph_key` is (first node `data`, `n_tokens`) **plus
  `first_weight_ptr` and `shape_sig`**.  Under `-sm tensor` every layer's subgraph can begin on the *same*
  activation buffer (the subgraphs are structurally identical and their inputs do not overlap in
  lifetime), so one cache entry served all of them: each layer was compared against the **previous
  layer's** properties, the first differing member was a source `data` pointer that had "moved" by one
  layer's worth of weights, and the graph was run directly or recaptured on nearly every call.  Measured
  2x R9700 x8, Flash-Next IQ4_NL `-sm tensor -ncmoe 48`, 2000-token decode: `direct=549499 replay=31696
  capture=149 properties_changed=452439` (37.7 t/s, and `GGML_CUDA_DISABLE_GRAPHS=1` the same 37.3, i.e.
  the captures were worth nothing) -> `direct=3882 replay=574826 capture=2636 properties_changed=28`
  (**42.8 t/s, +13.5 %**).  `GGML_CUDA_GRAPH_KEY_WEIGHT=0` restores the old key.
  **The key is still only a key**: a replay requires `ggml_cuda_graph_update_required` to find every node
  unchanged, so a collision costs a recapture and never a mismatched replay.  Do **not** add a fingerprint
  that moves across a rebuild (an object address, an allocation pointer, a generation counter) to it —
  that is the r115 trap from the other direction, and a `.size()`-unbounded key set is bounded now only by
  the context's 10 s unused-graph sweep.
  **Deliberately not shipped:** the upstream PR's second half, an identity-free property compare (blank
  `src[]`/`view_src`/`buffer` before the memcmp).  It is worth **0.2-0.3 %** on top of the key
  (`42.6/112.8` vs `42.8/113.1` t/s; `properties_changed` 340/2054 vs 28/1588), and on a 3-device dense
  split it only moves the capture count (1161 -> 915) without changing the outcome, so it is not worth
  weakening the replay's only safety check.  The *cost* of that compare was instead removed without
  touching what it compares — see the r158 item in block 10.
* **Freeing any of a table's device buffers invalidates the captured graphs on that device (r149).**  The
  expert cache hands the MoE kernels **raw device pointers** as launch parameters — the arena slot table,
  the remap staging, the prefill routing staging, and the two heat mirrors — and none of them is a tensor
  source, so `ggml_cuda_graph_update_required` cannot see a change in them.  `ggml_cuda_graph_mem_freed()`
  is therefore bumped in `free_arena_backing`, `free_heat_dev_locked`, `free_table_buffers_locked`,
  `stand_down_table_locked` and `moe_cache_update_host`'s device move.  Before r149 this was masked (a
  decode graph on this shape never replayed); now that it does, a new free site must bump too, and the
  right place is the **helper** that returns the memory, not the caller.

* **Windows host memory: the free-VRAM figure counts every process, and a released mapping leaves the
  working set (r209, PR #269).**  Two Windows-only gaps, both in upstream code paths:
  **`cudaMemGetInfo`** maps to `ggml_hip_mem_get_info` (`vendors/hip.h`), which returns the smaller of
  HIP's figure and the adapter's own **WDDM** figure — `D3DKMTQueryStatistics` over the non-aperture
  segments, `CommitLimit - BytesResident`, i.e. every process on the card, adapter found by the LUID HIP
  reports, resolved from `gdi32.dll` at run time so nothing new is linked (`ggml-cuda/wddm-mem.cu`,
  Windows only: the file is an empty TU elsewhere).  On Windows `hipMemGetInfo` counts only this process,
  so the slab, the expert-cache sizing, `--fit` and `GGML_CUDA_MIN_FREE_MIB` all believed memory another
  process held was free; past that point WDDM pages to system RAM instead of failing (issue #216).
  `GGML_CUDA_WDDM_FREE=0` keeps HIP's figures.  It makes the *figure* honest — it does **not** make an
  overcommit fail, so `TODO` #216's accepted limitation stands.
  **`llama_mmap::unmap_fragment`** was `GGML_UNUSED(first); GGML_UNUSED(last);` on Windows, so the r49
  per-tensor release (`LLAMA_LOAD_RELEASE_MMAP`) and the end-of-load release freed nothing: every range
  copied to VRAM or to a pinned host buffer stayed resident next to its copy.  A view cannot be unmapped in
  part, so the range is now dropped from the working set (`VirtualUnlock` on unlocked pages, which reports
  `ERROR_NOT_LOCKED` and moves the clean pages to the standby list).  As on POSIX it is skipped under
  `--mlock`/`--check-tensors`/lazy tensors, and it unlocks any locked page inside a released range exactly
  as `munmap` would.  Reporter's numbers (Windows 10, 1 x R9700, Flash-Next GSQ-RCO-IQ3_XXS `--fit`): the
  working set after load **67.4 -> 21.2 GiB**, free RAM 5.3 -> 48.9 GiB, 8K prefill 275.5 -> 511.1 t/s,
  32K 326.5 -> 539.7 t/s, identical token output; the 21.2 GiB figure moves with the mapping release, the
  prefill figure needs the PLE prefetch as well (measured together, block 11).

---

## Block 10 — the slab, the fusion dispatcher, the async input ring (infrastructure)

> **r217 amendment (issue #273 items 1 and 4): the FA stream-K fixup grid and the GDN q/k l2-norm-pair grid go
> off the gfx1201 dispatch stall, and this block's `getenv` reads are latched.**  `launch_fattn`'s
> `flash_attn_stream_k_fixup_uniform` launch (one block per output tile) and
> `ggml_cuda_op_rms_norm_scale_pair_fused`'s `2*nrows*nchannels` launch now ask
> `ggml_cuda_grid_stall_pad()` and add one returning block, with a guard at the top of each kernel
> (`tile_idx * blocks_per_tile >= nblocks_stream_k`, `blockIdx.x >= 2*nrows`).  Arithmetic unchanged, i.e.
> bit-identical; `GGML_CUDA_GRID_STALL_PAD=0` still turns the whole pad off.  Worth it where the grid is short
> (the reporter's model launches these at 256 blocks: 15.48 -> 10.08 us and 15.8 -> 2.4 us per launch);
> neutral on a 16384-block grid, where the stall's fixed ~10 us is hidden by the kernel's own runtime
> (`INVARIANTS.md` r217 fact 1, `benchmarks/2026-11-05-issue-273.md`).  The same change latches every
> `getenv` in this block's files (`common.cuh`, `mmb.cu`, `mmvq.cu`, `norm.cu`, `fattn*.cu/.cuh`, `mmvf.cu`,
> `gdn-conv.cu`, `gated_delta_net.cu`, `ple-conv.cu`, `topk-moe.cu`, `unary.cu`, `norm-gated.cu`,
> `moe-weighted-reduction.cu`, `mmq.cu`, `concat.cu`, `cpy-batch.cu`): one read per variable, stored.  Two
> per-op reads this block owned are gone: `GGML_CUDA_MMB_TYPES` (per matmul, up to 22 913 calls per run) and
> `GGML_CUDA_FA_STAGE_CHUNK_DEBUG` (per FA launch, up to 431).  Greedy text identical on every gate arm;
> prefill-logit KLD 0.000000 / 100.000 %.

**What it is.**  The infrastructure half of the pre-r109 block 10: it owns the translation units split out
of `ggml-cuda.cu` (the movable-boundary **slab**, its arena, and the `ggml_cuda_try_fuse` dispatcher), the
async graph-input ring, and the GDN/MMB/rope decode fusions.  The **qwen4exp architecture it used to carry
is block 11** (r109 split; the tree is unchanged by the split).  Rule of thumb: a qwen4exp-only piece that
is implemented inside a general kernel TU (the GDN gate/beta fusion, the HC16 MMB BF16 streams, the
weighted-down reduction, the PLE conv fusion) stays here with the TU that owns it; a piece that needs this
block's own symbols (QSA/HC matchers, the indexer relu-sum, the HC_BLK16/RES16 marking pass) is block 11.

**Contents.**
* **The BF16 gate/beta epilogue computes the chain's softplus (r201, issue #265).**  `mmvf.cu`'s `gate_beta`
  epilogue (`GGML_CUDA_FUSE_GATE_BETA_BF16`) had the same `logf(1.0f + expf(x))` vs `op_softplus`'s
  `log1pf(expf(x))` rounding as the Q8_0 sibling in block 08; it now calls `log1pf` too.
  `tests/test-fusion-gate-beta-multiseq.cpp` (owned here) covers **both** weight types and the delivered
  models' real contraction length -- `{BF16, 1024}`, `{Q8_0, 1024}`, `{Q8_0, 5120}`, 11 `(T,S)` shapes, 33
  cases -- because BF16-at-1024 was the only arm the unit oracle had, and no GDN model in the delivery runs
  that.  **The extension landed in r202**, not r201: r201's entry recorded it as shipped while the tree still
  carried the 11-case BF16-only matrix, which is the exact blindness that let the softplus difference live
  (gfx1100 `33/33 shapes fused, 0 failed`; `test-logits-width-probe P=4096` would not have caught it either
  -- a width probe passes a value that is uniformly wrong).  `GREEDY-PURITY.md` §81.
* **The slab** (`ggml-cuda-slab.cu`, `ggml-cuda-arena.cu`): one VMM reserve per device split by a movable
  boundary — compute below, MoE arena above, no HIP call at runtime.  Plus the narrow-band learning, the
  wide-phase auxiliary region, pool-wide routing, and the slab transient absorption.
* **The headroom default is `1152 + 128 x n_gpu` MiB (r206; it was `640 + 128 x n_gpu`, i.e. 768 / 896 / 1024
  MiB, from r86).**  `GGML_CUDA_SLAB_HEADROOM_MIB` is the steady-state free VRAM the slab leaves outside itself,
  and one GPU now gets **1280** MiB.  The 640 base was sized on the field GSQ / `-ub 4096` shape (~300 MiB of
  outside transient per device); block 15's `-c 262144` host-expert runme needed 1536 there to stop
  `hipBLASLt out of memory` on the first prefill (r194), so the shipped default was one workspace short on a
  documented config.  **This is margin, not a demonstrated fix**: the r194 cliff is order-dependent and r206 did
  not reproduce it (5 246 / 25 763 / 51 475-token prefixes at `-c 262144`, 768 and 1280, all clean, text
  identical).  Cost, measured on the field shape (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`, `-c 32768` q8_0,
  5 246-token prompt, `-n 2000`, interleaved): arena 24 691.7 -> **24 576.4 MiB**, `h` 0.9603 -> **0.9602**,
  decode 47.95 -> 47.98 t/s, text byte-identical.  An explicit value is still taken verbatim.  `ENVIRONMENT.md`
  §1.3.
* **The wide-aux growth is decided ONCE per wide pass (r168, TODO #70).**  `ggml_cuda_slab_wide_aux_alloc()`
  (the FA prefill staging arena's home) is called on **every** FA op of a prefill, and a refusal used to be
  returned per allocation: a starved region became hundreds of decisions, each able to evict a band of arena
  tables -- or evict one and then refuse on a boundary move that raced the MTP draft context -- while a
  queued graph still held the previous placement (the r86 fault shape).  A **VA-limit** refusal (the pinned
  narrow-2 region, the top of the mapping) is structural and latched in `wide_aux_denied_want` -- cleared by a
  `ring_set()` pass transition, by a boundary **shrink** in `work_alloc`, and on slab (re-)init -- so a request
  at least as large as the refused one returns immediately while a smaller one may still fit (a grow only
  ever raises the extent); a **snapshot-race** refusal (the boundary moved while the lock was dropped for the
  eviction) is recoverable and is retried, bounded to `GGML_CUDA_SLAB_WIDE_AUX_RACE_MAX` (4) attempts per
  pass because each attempt costs another arena-band eviction.  One `SLAB_WIDE_AUX ... refused for this wide pass` WARN per
  pass (not gated on `GGML_CUDA_SLAB_STATS`), and the per-pass count is the `aux_refused=` field of the
  `SLAB_STATS` line; the per-growth `SLAB_WIDE_AUX` diagnostic is unchanged (it is the sizing tool).  The
  shape is unreachable since r87 forces the prefill -> decode drop, so it is validated by code review plus a
  healthy-run no-op.  Record: `archive/work/todo70-rearm-wideaux/`.
* **The compute-buffer chunk follows a LIVE slab** (r139, issue #214): `GGML_COMPUTE_BUFFER_CHUNK_MIB`
  used to key on `ggml_cuda_slab_enabled()` (armed), which stays true after the slab declines at the
  first compute allocation.  A 12 GiB card that streams its host experts therefore had a 510 MiB
  layout rounded to 768 MiB (2 x 256 + one spare), 258 MiB the run cannot use.  The chunk is now gated
  on `ggml_cuda_slab_active(device)` via `ggml_backend_cuda_buffer_type_device()`: the allocation that
  creates the slab and every allocation on a declined device keep the percentage margin (561 MiB), and
  a live slab uses the chunk from the first allocation after it exists.  Text byte-identical on the
  streamed and arena arms, decode/prefill at parity.  `WORKLOG.md` 261025.012.
* **The FA prefill staging arena's size hint is a CEILING, not a target** (r156, issue #237).
  `launch_fattn` passes the F16 size of the K/V view **roots** (per-operand capped by
  `GGML_CUDA_FA_STAGE_MAX_MB`) so a growth step can jump towards the eventual size instead of climbing the
  1.25x ladder -- on Windows/WDDM a freed allocation stays committed, so the ladder strands ~5x the final
  arena (issue #138).  But the first FA call of a large-context prefill requests almost nothing (2 MiB at
  `-ub 512` on a 262144-token cache), and allocating the hint there commits the whole arena up front: 1 GiB
  **per context**, which since r125 includes the MTP draft context (its KV follows `-ctk`), so a `q8_0` draft
  KV cost more than the f16 default it exists to save.  `fattn_stage_try_get()` therefore takes
  `min(hint, 4x the request that just arrived)` (never below the 16 MiB floor), which right-sizes a short
  prompt and still reaches the hint in <= 3 steps for a deep one.  Do not "simplify" it back to
  `max(16 MiB, hint)`: the WDDM stranding it would fix is bounded either way, and the overshoot is not.
  `GGML_CUDA_FA_STAGE_PREALLOC=0` is the ladder A/B arm; `GGML_CUDA_SLAB_STATS=1` prints the per-growth
  `request=/cached=/alloc=` line.  `WORKLOG.md` 261027.007.
* **The VRAM limit now reports itself, and the "silent over-commit" reading is retired (r167, TODO #103).**  The run's free-VRAM **low-water mark** is sampled at every `>= 16 MiB` device allocation on
  *every* config — until r167 it existed only while the movable-boundary slab was active and printed only
  under `GGML_CUDA_SLAB_STATS`, so a dense/plain run (the shape #103 was found on) recorded nothing — and
  `GGML_CUDA_LOW_FREE_MIB` (default **512**, `0` = off, skipped while the slab is armed) warns **once per
  device** when an allocation leaves less than that free, naming the knobs (`-c` / `-ub` / `-ngl`,
  `GGML_CUDA_MIN_FREE_MIB`).  A refused allocation now reports free/total plus the low-water mark instead of
  only `cudaMalloc failed: out of memory`.  Nothing is refused and no default path changes; measured neutral
  in an interleaved A/B (a 15.8k-token 27B prefill).
  *Why the diagnosis moved:* #103's r135 cell ("1 GiB less free VRAM turns 13 s into 384 s") is **not** a
  llama.cpp path.  On this box HIP **refuses** an allocation that does not fit and never demotes a resident
  working set (a 30 GiB resident set reads at 505-513 GB/s with 1.7 GiB free), the at-the-limit cliff is
  ~50 MiB wide (15 754 MiB free: full speed; 15 704 MiB free: clean `ROCm error: out of memory`), and the
  only degradation a squeeze produces is **page migration of the *other* client's buffers** — 0-48 % of
  prefill, tracking GTT volume: one 16.4 GB holder BO is migrated whole (GTT 16.7 GB, 577 t/s at the same
  free VRAM where the same hold split into 64 MiB BOs migrates nothing and measures 1074-1077 t/s).  The
  FA staging stand-down, graph churn and clocks were each ruled out by measurement.  Record:
  `archive/work/vram-limit-103/`.  (It also replaces the "30x because the driver evicts resident weight pages"
  sentence in block 09's `GGML_CUDA_MIN_FREE_MIB` comment, since this block owns the tail of the same function.)
* **The block-10 arms of the Q8_1 cache fix** (r132, with block 08): the fused GLU -> Q8_1 site's key walk
  stops at the first pointer change (the fused write is keyed by the matmul's `src1`), and both of this
  block's gated-q8_1 launch arms — the fused norm+gate kernel and the GLU write — validate their sources
  with `q8_1_src_range_ok()` before launching.  `GREEDY-PURITY.md` §55.
* **`ggml_backend_cuda_graph_optimize` selects its own context's device** (r132): the MMB predicates the
  marking pass consults (`mmb_enabled()`/`mmb_cfg()`, per-device since block 06's r132 amendment) read the
  *current* HIP device, and under the Meta backend the pass is forwarded per child from meta
  `graph_compute`, where the current device is still the previously computed one.  `ggml_cuda_set_device()`
  is a no-op when the device already matches, so a single-device run is unchanged.
* **The slab's peer-access grants are per device and best effort, and a cross-device copy must ask before it**
  (r121, found validating TODO #94).  `ggml_cuda_vmm_map_phys` used to grant READWRITE for the owner and every peer in ONE
  `cuMemSetAccess` call and, on failure, silently retried with the owner only.  With three ~29 GiB slabs that
  call fails on the second and third device — the peers have no free VRAM left for the driver to build a
  29 GiB peer mapping — so devices 1/2 ended up owner-only, and the scheduler's layer-boundary
  `hipMemcpyPeerAsync` into their compute buffers was issued into an ungranted range: host-side
  `ROCm error: invalid argument` without `GGML_CUDA_P2P`, and a **GPU fault** (`HSA_STATUS_ERROR_MEMORY_FAULT`
  in `__amd_rocclr_copyBuffer`, "Page not present") with it.  That is what made `-sm layer` + `-ncmoe`
  abort at warm-up on Flash-Next (reproduced on the r120 base with no drafter attached).
  Now: own grant first, then **each peer individually**, the granted set kept in `ggml_cuda_slab.peer_mask`
  (a refused pair warns once), and `ggml_cuda_slab_peer_copy_ok(dst, nbytes)` decides the copy path — a
  destination **outside** every slab keeps the raw peer copy (plain memory stages inside the driver), a
  destination **inside** a slab needs the grants of *every* live slab, because a copy between two fully
  granted devices still fails while a third slab's grants are incomplete (measured: three 28 GiB slabs
  {0,1,2}/{0,1}/{0} fail 0→1, three 20 GiB slabs {0,1,2} succeed).  When it says no, both copy sites
  (`ggml_backend_cuda_cpy_tensor_async`, `ggml_backend_cuda_buffer_cpy_tensor`) use
  `ggml_cuda_peer_copy_staged`: a grow-only pinned buffer per (src, dst) pair, D2H on the source stream, H2D
  on the destination stream, and a `done` event guarding the reuse.  No env knob.  Cost: none where the
  grants complete (peer copies unchanged; tensor-split DFlash2 97.19 → 97.2 t/s, text identical); a host
  bounce on the refused pairs, which is what makes `-sm layer` + `-ncmoe 48` work at all (Flash-Next GSQ:
  265 t/s prefill, 48.4 t/s decode, greedy text identical to the `GGML_CUDA_SLAB=0` arm).
* **The fusion dispatcher** (`ggml-cuda-try-fuse.cu`): the matcher arms moved out of `ggml-cuda.cu`.
* **Rope/FWHT fusions** (M-RoPE and NEOX, with the q8_0 K/V cache `SET_ROWS` folded in), the paired GDN
  q/k norms, and the `ADD + rms_norm_q8_1` one-token fusion (`GREEDY-PURITY.md` §42).
* **The quantized-KV decode fusions (r187, PR #258, `GREEDY-PURITY.md` §74).**  With `-ctk/-ctv q8_0`
  `build_attn` rotates Q/K after their ROPE and V before the cache write, and for V the attention output
  before the projection — which turned every fusion above off on that arm: 726 kernel launches per token
  against 510 with an f16 cache (Qwen3-30B-A3B).  Five default-on terms put them back, all bit-identical and
  each with its own kill switch:
  * `GGML_CUDA_QK_ROTATE_HADAMARD` — the `GGML_CUDA_QK_ROTATE` pass also takes Q's rotation
    (`RESHAPE`, Hadamard `MUL_MAT`, `RESHAPE`, so `q_end = i + 7`) and **defers V's rotation** (the Hadamards
    between the chains, the `RESHAPE`/`VIEW`s they read and everything downstream of them) behind the Q chain,
    leaving the projections in front.  1-token graphs only (prefill keeps the stock order), and the whole
    rotate is skipped if a node left in front would read a deferred one.
  * `GGML_CUDA_FUSE_V_FWHT` — the fused K kernel (norm+rope+FWHT+q8_0 K/V `SET_ROWS`) rotates the V row
    itself with `ggml_cuda_fwht_warp<64, 32>` and the same lane->element quantize map, so the V FWHT launch
    goes away.  Single-use rotated V, not a graph output, 32-lane wave, source disjoint from the cache rows.
  * `GGML_CUDA_FA_COMBINE_FWHT` — at a 1-token FA whose **combine kernel** runs (`parallel_blocks > 1`) and
    whose output feeds the 64-wide rotation + an mmvq projection, `flash_attn_combine_results<D, true>` stages
    each row in shared memory, applies the same FWHT, stores it and publishes the projection's Q8_1 (the
    `GGML_CUDA_FA_COMBINE_Q8` mechanism, so `GGML_CUDA_FA_Q8_1`-style cache accounting is unchanged).  If the
    combine does not run the rotation runs as before and **no** cache entry is created; the plain `<D>`
    instantiation is untouched.  This is the one term whose kernel lives in block 05's TU — it is an
    extension of the combine-q8 code block 10 already owns (`fattn-common.cuh`), not a change to block 05's
    arithmetic.
  * `GGML_CUDA_QK_PAIR_FWHT` — `GGML_CUDA_ROPE_QK_PAIR`'s quantized form: Q's norm+rope+rotation is deferred
    to the V-rotation + K site seven nodes later and both run as one `rms_norm_mul_rope_f32_qk_pair<..,
    fwht_n, true>` launch (each block runs the single launch's body).  Q's rotation output is checked against
    everything the K half reads and the K/V cache rows against everything the Q half reads; an orphaned Q
    falls back to the single launch (with the FWHT when that is what was deferred).
  * `GGML_CUDA_FUSE_ROPE_FWHT` — a ROPE with **no** norm in front (Llama-style) runs as one
    `rms_norm_mul_rope_f32<.., no_norm = true>` launch, taking the K `RESHAPE`/`VIEW`/q8_0 `SET_ROWS` (and the
    V row when allowed).  **Any token count, so prefill takes it too** — it is the only term of the five that
    does, so it carries the prefill-logit obligation (`gate-prefill-logits.sh --ab
    "GGML_CUDA_FUSE_ROPE_FWHT=0"`, plus the `--ub 1` arm).  Fires on gpt-oss on our models, not on
    qwen/gemma (their ROPE follows a norm).
  All five key on the cache `SET_ROWS` being **`GGML_TYPE_Q8_0`**, and since **r199**/**r205** also
  **`GGML_TYPE_Q4_0`** / **`GGML_TYPE_Q5_0`** / **`GGML_TYPE_IQ4_NL`** on a wave32 device (PR #262, r205); a
  **non-quantized** cache (`f16`/`bf16`/`f32`) carries no rotation at all, so there is nothing to fold (see the
  r205 note below).
  Measured on gfx1201 1 GPU, Qwen3-30B-A3B Q4_K_M q8_0 KV: tg128 **141.2 -> 160.1 t/s (+13.4 %)**,
  @4096 **127.3 -> 140.1 (+10.1 %)**, launches/token **726 -> 510**; F16 KV unchanged (162.7 vs 163.1);
  gpt-oss-20b q8_0 **+4.0 % / +6.1 %**.  Records: `WORKLOG.md` 261031.005,
  `benchmarks/2026-10-31-kv-rotation-decode.md`, `archive/work/kv-rotation-decode/`.
* **The `q4_0` KV cache write fold (r199, PR #262).**  `-ctk/-ctv q4_0` is the cache type for the 16 GiB cards, and
  on r187 it kept the whole per-layer cache write: K and V each ran their own q4_0 `k_set_rows_quant`, V's
  rotation ran as `fwht_cuda<64>`, and Q and K took two norm+rope launches instead of the pair (702 launches per
  token on Qwen3-30B-A3B against 510 for the q8_0 arm).  `ggml_cuda_quantize_q4_0_group` quantizes one block from
  the 32 lanes of a wave, **bit-identical** to `quantize_f32_q4_0_block` (the sequential scan's **first** element
  of largest |v| via an `fmaxf` butterfly plus a ballot for its lowest lane, a NaN never winning, `vmax = +0` for
  an all-zero group, the reference's two rounding statements, lane `j` packing element `j` with `j + 16`), and the
  rope + FWHT kernels take the cache type as a **template value** (`kv`: 0 = no write, 1 = q8_0, 2 = q4_0) so the
  q8_0 instantiations are untouched — a runtime select cost the q8_0 kernel 12.8 -> 15.1 us on a 256-wide
  rotation.  Wave32 only (the group quantizer is one 32-lane wave; the matcher gates it on the **running device's**
  `warp_size` and the launch sites assert it).  `GGML_CUDA_KV_FOLD_Q4_0=0` is the kill switch.  Census (gfx1201,
  graphs off): **8547 -> 6915**, i.e. the q8_0 arm's count — the q4_0 `k_set_rows_quant` (960) and two thirds of the
  `fwht_cuda` launches (528 -> 192) are gone, replaced by `rms_norm_mul_rope_f32_qk_pair<..,128,2>` x336 and
  `rms_norm_mul_rope_f32<..,128,2>` x144.  Measured with `llama-bench -fa 1 -p 0 -n 128 -d 0,4096 -r 5`, 3
  interleaved rounds, one idle box per arch: **gfx1201 +10.7 % tg128 / +9.5 % @4096** (145.9 -> 161.6 / 129.5 ->
  141.7), **gfx1100 +12.0 % / +11.2 %** (176.4 -> 197.5 / 161.6 -> 179.7), **gfx1151 +2.5 % / +2.6 %** (53.9 -> 55.3
  / 51.5 -> 52.9); `f16` and `q8_0` KV are unchanged and the kill-switch arm reads stock on all three.  Purity:
  `--kv q4_0 --ub 1` prefill-logit A/B **-0.000000 / 100.000 %**, the same at the default ubatch on gpt-oss (the
  rope-only form, the one prefill path), `test-logits-width-probe` identical at `P=3584`/`P>=4096` for both models,
  greedy text byte-identical on six arms (1 GPU, 3 GPU `-sm tensor`, gpt-oss head 64, 35B `-ncmoe 40`, dense 27B,
  q8_0 KV) and on gfx1100/gfx1151, `test-backend-ops -o ROPE,SET_ROWS,FLASH_ATTN_EXT` 7059/7059.  MTP is flat
  (the r191 hold's gfx1201 cost does not reproduce at r198: 1 GPU bf16 76.5 vs 76.4 t/s, q4_0 74.6 vs 74.5, 3 GPU
  tensor q4_0 130.9 vs 130.1), Protocol A passes on dense and MoE, and the band gate is flat.  Record:
  `benchmarks/2026-11-01-kv-fold-q4_0-promotion.md`, `WORKLOG.md` 261101.005.
* **The `q5_0` and `iq4_nl` KV cache write folds (r205, `TODO` #120 closed).**  The same five folds now serve a
  **`q5_0`** or **`iq4_nl`** cache, so the last two quantized KV types stop keeping their per-layer write.
  `ggml_cuda_quantize_q5_0_group` reproduces `quantize_f32_q5_0_block` from one 32-lane wave (q4_0's first-of-largest
  seed, the reference's `-16` divisor and `min(31, (int8_t)(x + 16.5f))`, lane `j` packing `j` with `j + 16`, and
  `qh` — the one 32-bit word of 5th bits — written whole by lane 0 from a **ballot over the 32 lanes**);
  `ggml_cuda_quantize_iq4_nl_group` reproduces `quantize_f32_iq4_nl_block` (`kvalues_iq4nl[0]`'s seed,
  `best_index_int8` per element, and the reference's **16-term scale search**, which is order-sensitive: lane `j`
  computes term `j` and the terms are broadcast **in `j` order** and added one at a time, because a butterfly
  rounds differently).  The cache quantizer stays the kernels' compile-time `kv` value (now 0..4), the launcher's
  3-width x 2-`has_ff` x 2-`no_norm` matrix is written once by `RMR_FWHT_LAUNCH(NORM, KV)`, both types are
  wave32-only, and `GGML_CUDA_KV_FOLD_Q5_0=0` / `GGML_CUDA_KV_FOLD_IQ4_NL=0` are the kill switches.  Census
  (gfx1201, graphs off, 12-token decode): **8547 -> 6915** on Qwen3-Coder-30B-A3B (the q8_0 arm's count;
  `rms_norm_mul_rope_f32_qk_pair<..,128,3|4>` x336 + `rms_norm_mul_rope_f32<..,128,3|4>` x144 replace the 960
  `k_set_rows_quant` and 960 plain rope launches) and **9831 -> 9063** on gpt-oss-20b (its rope-only
  `FUSE_ROPE_FWHT` form).  Measured with `llama-bench -fa 1 -p 0 -n 128 -d 0,4096 -r 5`, 3 interleaved rounds:
  **gfx1201 `q5_0` +10.2 % tg128 / +8.8 % @4096** (155.9 -> 171.8 / 139.3 -> 151.5) and **`iq4_nl` +24.8 % /
  +22.1 %** (130.4 -> 162.7 / 120.1 -> 146.6); **gfx1100 +12.2 %/+11.1 %** and **+23.4 %/+22.8 %**; **gfx1151 flat
  (-0.1 %) and +0.5 %** (a 35B-A3B whose decode is not launch-bound, and where the fold does fire: 16139 -> 15839
  launches).  `f16`, `q8_0` and `q4_0` KV are unchanged against a pristine r204 build.  Purity: `--kv q5_0` /
  `--kv iq4_nl` `--ub 1` prefill-logit A/B **0.000000 / -0.000000 / 100.000 %**, the gpt-oss default-ubatch
  rope-only arm `0.000000 / 99.990 %` and `/ 100.000 %`, `test-logits-width-probe` identical at `P=3584`/`P>=4096`,
  greedy text byte-identical on 25 arms across gfx1201/gfx1100/gfx1151 (r204 == patched == switch-off, `q8_0`/
  `q4_0`/`f16` controls included), `test-backend-ops -o ROPE,SET_ROWS,FLASH_ATTN_EXT` 7059/7059, MTP Protocol A
  flat (67.0 vs 67.1 t/s, drafts 2816 / accepted 2059 both), the band gate flat.  **`bf16` is deliberately not
  shipped**: `attn_rot_k`/`attn_rot_v` require `ggml_is_quantized(type_k/type_v)`, so a bf16 cache has
  `n_rot_k = 0` and no fold pattern (`k_set_rows<float,int64_t,nv_bfloat16>` x960 + a plain rope x960, zero folded
  launches); the arm was written, measured and removed, and its `--ub 1` A/B read the same `-0.000001` as the
  noise-floor arm built from the identical code.  Records:
  `benchmarks/2026-11-01-kv-fold-q5_0-iq4_nl-promotion.md`, `WORKLOG.md` 261101.017, `GREEDY-PURITY.md` §74,
  `INVARIANTS.md`.
* **bf16 RoPE destinations (r103, `TODO` #83, filed and closed in r103)**: `rope.cu` instantiated float/f16 destinations for the NORMAL
  and NEOX modes but not bf16, and since r101 the iSWA `k_cur`/`v_cur` reorder lets the ROPE → VIEW →
  SET_ROWS fusion fire for Gemma's K — so `-ctk bf16 -ctv bf16` prefill aborted at `rope.cu:672`.  The four
  missing instantiations are in; the fused store is one `ggml_cuda_cast<D>`, exactly like the `SET_ROWS` it
  replaces, so it is **bit-identical** to the `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` chain (verified: KLD
  −0.000000 / 100.000 % at bf16 K/V) and faster than it (pp512 5136.70 vs 4973.06 t/s).
* **The FA accuracy instrument (r103, `tests/test-backend-ops.cpp`)**: `LLAMA_TEST_OP_NMSE=1` prints the error
  magnitude every case already computes, and `LLAMA_TEST_SEED=<n>` pins the random input so two arms are
  comparable (unseeded, an NMSE moves by up to 13x).  Plus the gemma-4 head-512 prefill-shape cases
  (kv 32768, nb 512, GQA 8, f16 and q8_0) that the family-switch criterion reads.
* **The GDN / attention-gate decode fusions** (r102, PR #200): four bit-identical fusions with their own
  switches — the qwen4exp alpha/beta BF16 projections plus `softplus(alpha + dt) * a` in one
  `mul_mat_vec_f` launch (`GGML_CUDA_FUSE_GATE_BETA_BF16`), the q/k l2 norm pair folded into the direct conv
  kernel (`GGML_CUDA_FUSE_GDN_CONV_NORM`), the attention gate's row-strided view read in place by the
  gate → Q8_1 kernel so its `CONT` copies nothing (`GGML_CUDA_FUSE_CONT_GATE_Q8_1`), and qwen4exp's GDN
  output `RMS_NORM * w` computed inside the gate kernel (`GGML_CUDA_FUSE_GDN_NORM_GATE`).  **Multi-sequence
  fix (r108, issue #204):** `GGML_CUDA_FUSE_GATE_BETA_BF16` launched the gate chain's `MUL` as the `mmvf`
  dst, so with more than one sequence in a ubatch the launch geometry was the consumer's
  `[nr, n_seq_tokens, n_seqs]` view against a flat one-channel `y`: every sequence but the first got
  sequence 0's alpha/beta gates (text stayed coherent; only multi-sequence serving saw it).  It now launches
  with the projection's own shape, which is the same memory written flat, so single-sequence launches are
  byte-identical and multi-sequence matches the unfused chain exactly.  `tests/test-fusion-gate-beta-multiseq.cpp`
  (also r108) is the regression gate: it needs no model, compares fused vs forced-unfused for a matrix of
  `(n_seq_tokens, n_seqs)` shapes and fails if the fusion stops firing (`GREEDY-PURITY.md` §49).  Two correctness
  rules came with them: the conv-norm fold is taken only when both SCALE outputs are disjoint from the
  conv's `x`/state/weight/output (the allocator may place them over the state or `x`, which the direct
  kernel still reads — without the check 35B-A3B varied run to run), and the GDN-norm fold has its skip and
  its emit at two different nodes, so **while a norm is pending no other fusion may start before its gate
  node** and the pending norm's own (dead) `MUL` is skipped.  With that guard the two abort tripwires the
  submitted form carried are unreachable rather than merely unlikely.
* **The verify-band arch extension** (r95, PR #192 + PR #193):
  `ggml_cuda_should_fuse_mul_mat_vec_q(..., verify_band = true)` admits **RDNA3_0** as well as RDNA4 via
  one `verify_band_arch` term.  That predicate is the single gate the three 2..8-token Q8_1
  pre-quantize fusions share — residual `ADD` + `rms_norm` → Q8_1, plain `rms_norm` → Q8_1, and gated
  unary → Q8_1 (plus the `GLU` → Q8_1 mark) — so one change admits all of them; their kernels are
  row-generic and write the values `quantize_q8_1` writes, so `W = 1..8` stay bit-identical.
  `gdn-conv.cu`'s `verify_ok` takes the same `(RDNA4 || RDNA3_0)` form, so the direct GDN conv covers
  the 2..255-token verify batches too and keeps its pre-existing single-device restriction.
  `GGML_CUDA_FUSE_Q8_1_VERIFY=0` / `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` still disable them; RDNA3_5 is
  excluded by the macro (`0x1100 <= cc < 0x1150`).
* **The DSV4_HC_POST arm is chosen by the split mode** (r97).  The `hc_combine_norm` chain spelled out
  (scale → sigmoid → scale → repeat → mul → add → rms → mul) is faster than the op on **one** device
  (+0.9..+1.8 % prefill on qwen4exp IQ4_NL, gfx1151) and stays the default there, but the chain's
  broadcasts are split-state sensitive and it is **not correct under the Meta (tensor-split) device**:
  `test-llama-archs -a deepseek4` on 3× gfx1201 read NMSE 1.74e-01 vs the CPU reference through the
  chain and 6.5e-06 through the op.  `cparams.fused_dsv4_hc_post` therefore defaults to *on* whenever a
  Meta device is in play and to *off* otherwise; `LLAMA_FUSED_DSV4_HC_POST` still forces either arm.
* **The ring's teardown flush is unconditional** (r97): `ggml_cuda_async_input_flush_done` early-returned on
  `!g_async_input_on`, but since r90 the ring also carries the `set_tensor_async` snapshots *unconditionally*,
  so `GGML_CUDA_ASYNC_INPUT=0` left a pending done-event naming the stream the context destructor destroys
  and the next context on that device recorded on the dead handle (`invalid resource handle`).  A no-op when
  nothing is pending; both arms now give the same results.
* **Async graph-input ring**: batched uploads + lazy done event (PR #182), and the r90 fix that snapshots
  the source through the pinned ring — the scheduler uses `set_tensor_async` for a *CPU split's* output,
  which the next ubatch rewrites, so a direct copy from the caller's live pointer could read the next
  ubatch's bytes.  Default-on; `GGML_CUDA_ASYNC_INPUT=0` disables.  **The ordering state is per origin
  host thread** (r105, issue #201): r101-r104 keyed the per-origin map on `cudaStreamPerThread`, which is
  the sentinel `((hipStream_t)2)` — the same value in every thread — so there was exactly one origin per
  device and r101's per-origin ordering did not exist.  It is now keyed on `std::this_thread::get_id()`
  (uploads are still *issued* on `cudaStreamPerThread`; ordering is carried by the per-copy `ev` / batched
  `flush_ev`, never by a stream handle), only the owning thread may consume an origin's ordering state (a
  foreign wait may still *issue* its queue — the loader legitimately uploads from another thread — but must
  leave the owner's wait intact), the `has_flush` event is consumed even when the queue is empty, and the
  ring head wraps only after every origin's queue is issued and the device has drained (a foreign origin's
  *queued* uploads would otherwise be overwritten before they were ever issued).  `GGML_CUDA_ASYNC_INPUT_DEBUG`
  logs the origin count/thread id; the two-thread reproducer is in `archive/work/async-input-per-origin/`.
* **`GGML_CUDA_GDN_TAIL`**: the fixed-`D` sequential tail that makes a chunked-GDN prefill batch write its
  rollback snapshots (r91; `GREEDY-PURITY.md` §27).
* Graph-memory generations (`GGML_CUDA_GRAPH_MEM_GEN`) so a graph is recaptured when the pool / FA staging /
  H2D ring memory it captured is freed, **and when `ggml_cuda_slab_work_alloc` shrinks the work boundary**
  (the post-prefill drop hands the wide region back to the arena; a captured graph baked its addresses).
  Issue #187: without that invalidation a 4-stream MTP decode replayed a stale `set_rows` source and
  faulted with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`.
* **The replay guard compares the live tensors in place, over a source-packed store (r158).**  The guard
  (`ggml_cuda_graph_update_required`) runs on **every** `graph_compute`, not only after a rebuild: the
  scheduler mints a fresh `cgraph->uid` per split per compute (upstream #21764,
  `ggml_backend_sched_split_graph`), and the Meta rebuild mints fresh child uids, so the `uid` early-out
  cannot fire (verified with `GGML_CUDA_GRAPH_DEBUG=1`: zero "reused" lines on a decode).  Its old form
  therefore paid, per node per call, a 1048 B `node_properties prop = {}` zero-fill, a
  `sizeof(ggml_tensor)` (328 B) copy and a 1048 B `memcmp` — measured `__memcmp_evex_movbe` at **1.41 %**
  of a decode-only profile window (2x R9700, 35B-A3B `-ncmoe 40`).  Now `ggml_cuda_graph_props_match()`
  compares `cgraph->nodes[i]` **in place** (the same full node header, `src[]`/`view_src`/`buffer` object
  identity included) and then only the **non-null** sources' `data`/`ne`/`nb`, from a store packed to the
  sources a node has (`prop_nodes` / `prop_src_off` / `prop_srcs`) — measured: 1.4 sources per node, and
  70,752 -> 29,840 B per graph entry (**−58 %**, i.e. ~100 MB less host state at the 4526 keys a
  6000-token session reaches).  `memcmp` 1.41 % -> 0.58 % of the window.  The guard is *not* weakened: the
  node-header compare is byte-identical to the old one (which is also what makes the packed source run
  equivalent — a null/non-null layout change already fails it), and with the per-layer keys (r149) the
  old `src[]`/name churn it used to clean up no longer exists.  Together with it: the graph key is built
  **once** per call (it was built 2-3×, and `ggml_backend_cuda_graph_optimize` built and discarded one on
  every call while `GGML_CUDA_GRAPH_OPT` is off — `ggml_cuda_graph_get_key` 0.26 % -> 0.15 %), and
  `ggml_cuda_graph_check_compability` reads the arch **once** instead of a `cudaGetDevice()` runtime call
  per `MUL_MAT_ID` node.  **No throughput claim**: on this box the work was hidden behind the GPU (35B-A3B
  2 GPU +0.15 %, Flash-Next 2 GPU +0.75 % within noise over 6 pairs, 3-device dense and 1-GPU dense flat,
  even pinned to 2 CPU cores +0.7 %); it is worth what it saves on a host-limited box — CPU time, bus
  traffic and RAM.  The census is unchanged (`replay` identical in every pair) and greedy text is
  byte-identical in every arm; prefill-logit KLD 0.000000 / 100.000 %.
* **The M-RoPE section power is computed once** (r174, PR #249, briansp2020): in `rope_multi` and in the
  M-RoPE arm of the fused `rms_norm_mul_rope_f32`, every section arm multiplied its position index by the
  *same* `powf(theta_scale, iw/2)`, and adjacent lanes take different arms (`sector % 3`), so the divergent
  arms were serialized and `powf` executed once per arm taken.  It is hoisted once before the branch: same
  input, same value, i.e. **bit-identical** (the non-M-RoPE path and the "no arm matched, `theta_base`
  stays 0" case are untouched).  Kernel level on gfx1201: `rms_norm_mul_rope_f32` **3.20 -> 2.85 µs**
  (`__half` dst) and **3.06 -> 2.70 µs** (f32 dst) per launch, which is inside the wall-clock noise on
  RDNA4 (8 launches/token on a 4B) and **+0.25 to +0.65 % tg** on gfx1100 (7900 XTX: 35B-A3B Q4_K_M
  137.6 -> 138.6, 27B Q4_K_M +0.25 %, 9B Q8_0 +0.25 %).  Greedy text byte-identical on 4B Q8_0 / 27B
  UD-Q4_K_XL / 35B-A3B Q4_K_M, and the prefill-logit gate reads 0.000000 KLD / 100.000 % top-p at the
  default ubatch and at `-ub 1`; `test-backend-ops -o ROPE` 450/450.  **The submission's other half is
  deliberately NOT taken**: a GDN direct conv for 1-token decode (`gdn_conv_check`'s `T >= 2` gate dropped,
  with `ggml_cuda_ssm_conv_in_try_fuse` standing down so the two fusions cannot both claim the concat).  It
  only repays where the direct kernel also elides the q/k norm pair: there the launch count is equal and it
  measures **0 %** (its own author's gfx1100 35B-A3B Q4_K_M arm: 138.6 both arms, `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`
  control 138.7); where `gdn_conv_norm_match()` is refused (dense Qwen3.5 4B/9B/27B Q8_0) the `ssm_conv_in`
  epilogue was writing the whole CONCAT inside the mmvq for free, so the decode path gains a launch per GDN
  layer (`gdn_concat_tail`, 1.44 µs) and loses **-1.1 %** tg256 on gfx1201 — a fix would have to make the
  decode case conditional on the norm fold, and it would still need a measured win.  It also preempts the
  decode-only `ssm_prescan` 15-node fusion unconditionally, which no model here reaches but a Q8_0 GDN
  model with matching wiring could (`ssm_conv_l2_gatebeta` appears in no trace taken so far).  Record:
  `benchmarks/2026-10-29-gdn-direct-conv-decode.md`.

**Knobs.**  The slab/cache set listed under block 09, plus `GGML_CUDA_POOL_WIDE[_MIN_MIB]`,
`GGML_CUDA_SLAB_STATS` / `_GRAPH_TRACE`, `GGML_CUDA_ASYNC_INPUT_BATCH` / `_LAZY_DONE` / `_DEBUG`,
`GGML_CUDA_GDN_TAIL`, `GGML_CUDA_GRAPH_MEM_GEN`, `GGML_META_SPLIT_COPY`, `GGML_META_SEG_ASYNC`,
`GGML_CUDA_FUSE_GATE_BETA_BF16` / `GGML_CUDA_FUSE_GDN_CONV_NORM` / `GGML_CUDA_FUSE_CONT_GATE_Q8_1` /
`GGML_CUDA_FUSE_GDN_NORM_GATE` (r102, all default-on kill-switches; the first is multi-sequence-clean since
r108 and has a shipped regression test),
`GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS` (now live on iSWA archs, not only on the non-iSWA builder),
`LLAMA_KV_N_PAD_PREFILL`(+`_FROM`/`_MIN`).

**Split-state rule (r101).**  The Meta handlers may only accept a split the op can serve **locally**:
empty-or-whole per device for a per-row op, for a rotated-K `mul_mat`, and for the FA K/V of a
multi-sequence ubatch; a split that straddles the reduction is rejected, never approximated.  Every new
acceptance must come with the same-shape refusal for the straddling case (`GREEDY-PURITY.md` §45).

**Gate.**  Debugging the slab or the arena: `GGML_CUDA_SLAB_STATS=1` (arm/drop/destructor accounting,
`hole=0` and `outside_failed=0` are the acceptance signals).

---


### The FA prefill staging arena: two opt-in bounds (r159)

**What it is.**  `launch_fattn` dequantizes the K/V rows an op reads into a per-stream F16 arena
(`fattn_stage[]`) before launching, because reading the quantized cache once per *output* tile is slower
(worth +1.5 / +2.7 / +5.1 % pp at 8k/16k/32k on gfx1201, M0a).  The arena is transient (created and dropped
inside one op, per layer) but its **request** is O(`n_past + n_q`), which is what issues #138 (the growth
ladder strands ~5x its size on WDDM) and #237 (an eager ceiling commits up to 1 GiB at `-c 262144`) are
about.  r159 adds two **opt-in** bounds; the default arm is unchanged and bit-identical.

* **`GGML_CUDA_FA_STAGE_CHUNK=<rows>`** — the structural bound.  The op is launched **once per KV chunk**:
  convert that chunk into a one-slot ring, launch the same stream-K grid over it, fold, merge into the
  running accumulator with an online softmax, then normalize once at the end.  That needs a **per-row
  partial contract** in the prefill kernel: every block publishes `(unnormalized numerator, max, rowsum)`
  for its own rows — the no-seam (whole-tile-in-one-block) case included, which used to publish a
  normalized value and no meta at all — and the two stream-K fixup kernels hand the merged, still
  unnormalized value back instead of dividing (`float2 * kv_partial_meta`, nullptr = the shipping
  behaviour).  One `kv_start` int lets the chunk keep the two absolute-KV optimizations: the fully-masked
  group bitmap indexes `(kv_start + kb0*nbatch_fa)/FATTN_KQ_STRIDE` and the `KV_max` bound becomes
  `max(0, KV_max - kv_start)/nbatch_fa`; both are exact because chunks are `lcm(nbatch_fa,
  FATTN_KQ_STRIDE)`-aligned.  Arena: **64 MiB flat** at `=16384` (27B), vs 160 MiB at 33k / 1024 MiB at
  `-c 262144`.  Cost: +0.0 % at 32k, -4.1 % at 64k, ~1.5 % per chunk beyond the second (and the same shape
  in prefill-logit drift: KLD 0.001 at 2 chunks vs 0.0053 at 4).  **Facts a future edit must keep:**
  attention **sinks** belong to chunk 0 only (they add `exp(sink - max)` to the rowsum, so passing them to
  every chunk counts them `n_chunks` times); the no-seam publish must be gated on `k == 0`, not on a
  particular `stride_k` pass (a small `nbatch_combine`, e.g. 16, leaves the `warp_size` pass empty); the
  **packed** fully-masked-group bitmap cannot be sliced by a chunk start (its per-tile word count is derived
  from the launched KV length) so a chunked launch disables it — the **derived** (batch-wide) layout, which
  is what a single-sequence prefill uses, slices exactly; and a chunked launch runs `>= 2` chunks only (one
  chunk is the whole-prefix arena).  On a slab config the arena is served by the slab's wide-aux region and
  released at the prefill->decode drop, so this buys nothing there: leave it off.  Gate set: `gate-prefill-logits.sh
  --kv q8_0` (the chunked arm is a re-baseline, 2 chunks pass), `FLASH_ATTN_EXT` 6450/6450 **with chunking
  on**, `FLASH_ATTN_QSA` 26/26, the `W=1..8` probe (band untouched) and the pp matrix.  Records:
  `archive/work/fa-stage-chunking/` (`FINDINGS-2026-10-27-m3.md` = the numbers).
* **`GGML_CUDA_FA_STAGE_PREALLOC=2`** — the policy bound: one allocation of the root-based hint at the first
  request (r149 behaviour) instead of the r156 ceiling's ladder.  Allocation only, so numerics are
  bit-identical (the reporter's re-test: identical greedy text hash and PPL across every arm).  On WDDM it
  restores r149 (pp 592 -> 781 against r149's 791); the trade is the r149 **peak**, 1 GiB per context at
  `-c 262144`.  `0` = the pre-r53 1.25x ladder, `1` = the default ceiling.  A draft-context cap
  (`GGML_CUDA_FA_STAGE_DRAFT_MB`) was prototyped and **dropped**: it only lowered the draft arena's
  starting size and the context kept laddering (64 -> 417.5 MiB, late steps at 2:08), so it did not do what
  its name promised; `GGML_CUDA_FA_STAGE_MAX_MB` is the knob that actually bounds a draft.

### The RDNA3 decode fusions (r175, PR #250)

**What they are.**  Seven decode fusions contributed by `@briansp2020`, folded here because this block owns
the fusion dispatcher, the rope matcher and the MoE multi-token kernel (`mmvq-moe`) — no earlier block can
carry them.  Every one is **default-on with a `=0` kill switch**, bit-identical to the launch(es) it
replaces, and validated on the launch it removes:

| patch | what | knob | cost when off (30B-A3B, gfx1100/gfx1201) |
|---|---|---|---|
| FA combine publishes its output's Q8_1 | the attention output projection skips `quantize_q8_1` | `GGML_CUDA_FA_COMBINE_Q8` | 2.5 / 2.2 % |
| MoE gate/up+GLU publishes the down projection's Q8_1 | the down `MUL_MAT_ID` skips `quantize_q8_1` | `GGML_CUDA_MOE_GLU_Q8` | 2.1 / 1.0 % |
| the fused K norm+rope+SET_ROWS also writes the V row | the V `set_rows` launch goes away | `GGML_CUDA_ROPE_SET_ROWS_V` | 2.2 / — |
| MoE GEMV takes 4 rows per block (RDNA3_0) | half the y loads and warp reductions per wave | `GGML_CUDA_MOE_RPB_WIDE` | 3.6 / inert |
| MoE down + weighted reduction in one launch | the reduction node launches nothing | `GGML_CUDA_MWR_FOLD` | 1.0 / — |
| Q and K norm+rope in one launch | one launch per layer instead of two | `GGML_CUDA_ROPE_QK_PAIR` | 3.7 / 3.0 % |
| dense V/K/Q projections in one launch (RDNA3_0) | one ksplit launch instead of 2-3 | `GGML_CUDA_MMVQ_ROW_CONCAT` | 2.8 / inert |

**Facts a future edit must keep.**

* **The Q8_1 publishes write exactly what `quantize_q8_1` writes, including the padding.**  Both the FA
  combine and the GLU publish use its 32-lane `warp_reduce_max/sum` + `roundf` arithmetic verbatim; the GLU
  publish's row length is generally **not** a multiple of `MATRIX_ROW_PADDING` (768 on Qwen3-30B), so the
  block that completes a row's **last** group also zeroes that row's padding blocks — the arena is reused
  across graphs, so anything that walks a padded row later must not find stale blocks.  The FA publish takes
  only an unpadded row (`ne10 % MATRIX_ROW_PADDING == 0`), where the two layouts coincide.
* **The combine publish is indexed by the combine kernel's own unrolling,** `j_dst_unrolled*D + tid` where
  `j_dst_unrolled = (sequence*n_head + head)*n_tokens + token`, i.e. the flat element offset of the
  contiguous `[D*n_head, n_tokens*n_seqs]` row layout — correct for any token/sequence count because the
  q8 rows are in `(i13, i12, i11)` order.  Do not "simplify" it to a per-row index: that is the same value
  only at one token.
* **Neither publish checks that its consumer takes the mmvq path** (the `fwht_q8_1` matcher does, because it
  runs in the dispatcher; these run in the kernel).  That is harmless — an unused entry costs arena space for
  one graph — and at prefill the FA combine is not reached (`parallel_blocks == 1`), so measured peak VRAM is
  unchanged (27B, `pp8192`: 20.17 GiB both).  Re-check that if the combine ever runs at prefill widths.
* **A publish needs a self-resetting counter, and its `cudaMalloc` must not land in a captured graph.**  The
  counters (`GGML_CUDA_MOE_GLU_Q8_CNT`/`GGML_CUDA_MOE_RED_CNT`) follow block 10's existing `hc_q8_cnt`
  pattern: allocated on first use and zeroed once on the stream, then each kernel resets the counters it
  completes.  Both fire first on a 1-token decode step, i.e. after the prefill that has already sized the
  decode graphs' memory — do not move an allocation into a 1-token-only path that can run inside a capture.
* **The MoE down fold writes the reduction from the last expert lane of a row block,** in
  `moe_weighted_reduction`'s own order and expression (`sum = (e*scale)*w`, experts in order), and therefore
  also takes `8` rows per block to quarter the counter atomics.  It steps aside when the reduction output
  overlaps the expert-id tensor (later blocks of the same launch still read the ids).  Both fold paths are
  gated on `dst->ne[2] == dst->ne[3] == 1`, i.e. **one token**: a verify batch never takes them.
* **The row concat runs ahead of the dual-output fusion** (the submitter's order).  It preempts the
  dual-output on a pair of *equal-height* matmuls over one src1 — the dense FFN gate/up pair — which the
  dual-output is the older owner of.  Measured **neutral** on a dense 27B (row-concat off 43.05 vs on 43.13
t/s, gfx1100) and, once the submitter's `qwen3moe` model reorder is dropped, **identical either way**
  (30B: 192.8-193.6 vs 192.5-193.6 t/s, three alternating rounds), so it is kept as submitted and recorded
  here rather than reordered.  A `row_cat` template instance keeps the plain ksplit kernels byte-unchanged
  (an in-kernel segment check cost Qwen3.6 2 % with the feature off).
* **The submitter's `qwen3moe.cpp` graph reorder is NOT taken** (`LLAMA_QK_ADJACENT` does not exist here).
  It expands V and the raw K projection before the Q norm so the two norm+rope chains are adjacent, which is
  worth ~6.7 % of the 30B on top on gfx1100 (~3.3 % on gfx1201), i.e. the whole difference between +12.2 %
  and +19.7 % -- but it edits **upstream
  model code for all backends** to enable a CUDA fusion, on an arch family that is retired.  Without it the
  Q/K pair fires where the graph is already adjacent (Qwen3.6) and qwen3moe pairs rarely; the row concat
  still takes V/K/Q when the heights differ.
* **`GGML_CUDA_MOE_RPB_WIDE` and `GGML_CUDA_MMVQ_ROW_CONCAT` are RDNA3_0-gated** (`ggml_cuda_get_arch()` /
  the mmvq table id) because their launch geometry is row-count-only through `small_m` and the rows per
  block.  RDNA4's config depends on the row count twice more, so neither is armed there — do not widen the
  gate without that model.  Everything else is arch-free and fires on RDNA4 too (measured: 30B-A3B
  +6.9 %, 35B-A3B +2.9 %, dense 27B +0.4 % on gfx1201).

**Gates.**  Greedy text byte-identical to the previous release on 30B-A3B (both arches), 35B-A3B and dense
27B; `plain == draft-mtp` (width purity) on 27B dense and 35B-A3B at 1500 tokens, on 1 GPU and 2-GPU
`-sm tensor`, and 35B-A3B `-sm tensor -ncmoe 40`; prefill-logit KLD **-0.000000 / 100.000 %**; MTP Protocol A
unchanged (27B 3 GPU layer 51.6, tensor 100.5 -> 100.8); rule-5 `llama-batched-bench -npl 1,4,8` flat; the
per-patch switch-off deltas above; `test-backend-ops -o ROPE,MUL_MAT_ID` OK; no GPU faults.  Numbers:
`benchmarks/2026-10-29-rdna3-decode-fusions.md`.

#### r177 — the Q/K/V projection reorder and the dense row concat on RDNA4 (PR #252)

**What it is.**  Two of that PR's three graph-ordering patches, from `@briansp2020`, folded here for the
same reason (block 10 owns `ggml_backend_cuda_graph_optimize`'s fusion passes and `mmvq.cu`).

| patch | what | knob | worth (30B-A3B, gfx1100/gfx1201) |
|---|---|---|---|
| V/K projections move in front of the Q projection | Q can join the row concat and the Q/K norm+rope pair fires | `GGML_CUDA_QK_ROTATE` | 5.9 / 3.2 % |
| the dense row concat also runs on RDNA4 | one ksplit launch instead of 2-3 on RDNA4 too | `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4` | inert / 2.6 % |

**Why the reorder is needed.**  The model graph runs Q MUL_MAT, RESHAPE, RMS_NORM, MUL, ROPE and *then* the
V and K projections, so (a) the row concat's scan from the Q matmul stops at the Q norm (only V+K fuse) and
(b) the allocator hands the Q chain's freed buffers to V/K, so `ggml_cuda_qk_pair_k_at`'s overlap test
refuses the Q/K pair.  The pass matches the Q chain structurally, requires the nodes in between to be only
`MUL_MAT`/`RESHAPE`/`VIEW` and to read none of the Q nodes (resolving `view_src`), matches the K chain's
`RMS_NORM`/`MUL`/`ROPE` by identical rope parameters **and** position tensors, and then `std::rotate`s the
V/K block in front of the Q matmul — the same trick as the r175 shared-expert rotate, same nodes, one
stream, so it is bit-identical.  It is structural only (no data pointers), runs at every ubatch size, and is
idempotent (`j == i + 5` is the rotated shape, so a second pass declines).

**Facts a future edit must keep.**

* **The rotate sits in the alloc-deps half of the pass, i.e. it runs for the meta backend's `allocs_only`
  forward and NOT for `marks_only`.**  That is deliberate — the marks are per tensor and the fusion deps must
  be registered before allocation — but it means the two halves of a `-sm tensor` optimize see different
  node **orders**, so no matcher added to the marking half may depend on adjacency that this rotate changes.
  Verified byte-identical on 2-GPU `-sm tensor` (30B-A3B, text and prefill logits), which is the arm that
  exercises both forwards.
* **The pass is not arch-gated, and must not be**: it only re-enables fusions that are themselves gated
  (the row concat's own table-id gate, the Q/K pair's own matcher).  On both RDNA generations the reorder
  alone carries the whole win; the RDNA4 row concat is the increment on top.
* **The RDNA4 row concat needs every segment to have an even row count.**  RDNA4's single-token launch may
  take two rows per block and a 2-row block must not straddle a segment boundary; the check is `ne[0] % 2`
  on *every* segment including the main one.  RDNA3_0 keeps its own `small_m` + rows-per-block agreement
  rule.  **Do not widen the RDNA4 gate further**: the argument is that `ncols_dst == 1` makes
  `calc_nwarps_weight` row-independent, and that `two_rows` / `GGML_MMVQ_RDNA4_DECODE_GRID` / the
  `grid_stall_pad` block only change *which* block computes a row — a future row-count-dependent nwarps or
  VDR would break it silently, so re-check that predicate before touching it.
* **The submitter's third patch ships since r180** (`GGML_CUDA_MMVQ_CONCAT_REORDER`, PR #252 0003; see the
  r180 subsection below).  It needs a **quantized** `<= 256`-row weight and skips `Q8_0`, so it cannot fire
  in the UD quants (F32 `ssm_alpha`/`ssm_beta`) or in the Q8_0 quant; the measurement that took it is on the
  submitter's llama.cpp-quantized Qwen3.6-35B-A3B Q4_K_M (whose GDN small projections are Q4_K), and it is
  inert on every other GGUF we hold.

**Gates.**  Greedy text byte-identical to r176 on 30B-A3B (1 GPU and 2-GPU `-sm tensor`), 35B-A3B, dense
27B, Gemma4-31B and Flash-Next (`-ncmoe 48`, 2 GPU tensor); prefill-logit KLD **0.000000 / 100.000 %** at the
default ubatch and at `-ub 1`; MTP Protocol A on the dense 27B at 3 devices unchanged (layer 25.0 -> 51.7
t/s, acceptance 0.621; tensor 50.8 -> 101.2, acceptance 0.608; `plain == draft-mtp` text identical in both
split modes); rule-5 `llama-batched-bench -npl 1,4,8` flat (31.91/102.11/156.90 -> 31.94/102.20/157.18);
2-GPU `-sm tensor` 30B-A3B +4.7 % (d0) / +3.9 % (d4096); per-block build 2/2; `validate-set.sh` green;
no GPU faults.  Numbers: `benchmarks/2026-10-30-graph-order-r176.md`.

#### r180 — the GDN small-projection gather (PR #252's third patch)

**What it is.**  A second structural pass in `ggml_backend_cuda_graph_optimize`, placed right after the V/K
rotate and gated on RDNA3_0/RDNA4.  The Qwen3.5/3.6 GDN layers read the same normed input with
`ssm_beta` (32 rows), `ssm_alpha` (32 rows) and the `attn_gate` z projection (4096 rows), but sigmoid /
softplus / reshape nodes sit between them, so each ran as its own GEMV launch.  From the first
**quantized** `MUL_MAT` of `<= 256` rows (`a`), the pass scans up to 24 nodes ahead for `MUL_MAT`s with the
same `src1`, the same weight type, the same `K` and the same row stride (at most one of them larger than
256 rows), and `std::rotate`s up to 2 of them directly after `a`, so a group is at most 3 `MUL_MAT`s (the row
concat's cap) and `ggml_cuda_mul_mat_row_concat_try_fuse` runs them as one launch.

**Worth (gfx1201, 1 GPU, 3 interleaved rounds, `-fa 1 -p 0 -n 128 -d 0,4096 -r 10`).**  On the submitter's
file (lmstudio-community `Qwen3.6-35B-A3B-Q4_K_M`, sha256 `4ac6a06b…a410d`, 40 blocks with
`full_attention_interval = 4` → 30 GDN layers): tg128 **129.35 -> 132.93 (+2.8 %)**, d4096 **126.16 ->
129.65 (+2.8 %)**; `=0` returns 129.45 (≡ base).  With `GGML_CUDA_MMVQ_ROW_CONCAT=0` (gather on, concat off)
it is 130.7, i.e. the gain is the concat the gather enables.  Kernel census (`rocprofv3`, 44 decode steps):
44 808 -> 42 231 dispatches = **60 fewer launches per token** (30 layers x 2).  The submitter's +3.0 / +2.9 %
reproduce; their gfx1100 +3.5 % is their box.

**Facts a future edit must keep.**

* **Both skip guards are load-bearing, and neither is a tuning choice.**  A `MUL_MAT` whose output feeds a
  `CONCAT` within 8 nodes is never moved, because `ggml_cuda_ssm_conv_in_try_fuse` returns `j - i` and the
  nodes in between are never executed — so a node moved into that window would silently never run.  A
  `Q8_0` weight is never moved, because `ggml_cuda_ssm_prescan_try_fuse` matches the 16-node SSM chain **by
  position** (`alpha_w` at `i + 7`, `beta_w` at `i + 13`) and only fires when both small weights are `Q8_0`;
  moving one out of the window would silently drop the whole conv + l2-norm + gate/beta fusion.
* **The pass only reorders; it never fuses or elides.**  It moves a node earlier across nodes that a valid
  topological order guarantees cannot read it (its sources are the anchor's activation and its own leaf
  weight), so no reader is skipped and no buffer lifetime shortens.  Anything it enables is enabled by a
  fusion's own predicate at compute time.
* **It is arch-gated (unlike the V/K rotate) because it is a tuning pass, not a correctness enabler** — the
  fusion it feeds carries its own RDNA3_0/RDNA4 gate.  A new arch that wants it needs its own row-concat
  proof, not a wider arch list.
* **Purity is a decode-width property and was measured as one**: prefill-logit KLD **0.000000 / 100.000 %**
  at the default ubatch, at `-ub 1` and at `-ub 8` (the gather runs at every ubatch size; the row concat is
  `ne[1] == 1` only), greedy text byte-identical on 1 / 2 / 3 devices, in both MTP split modes (layer and
  tensor) and on the DFlash2 draft arm, whose `selector_hidden` is the only other `<= 256`-row quantized
  weight we hold (Q4_K, 256 rows) and which the gather does not match.

**Gates.**  The above plus MTP Protocol A unchanged (layer acceptance 0.62134 = 1951/3140, 51.6 t/s; tensor
0.60816 = 1937/3185, 101.5 t/s; `plain == draft-mtp` byte-identical in both split modes), rule-5
`llama-batched-bench -npl 1,4,8` flat (124.3 -> 127.2 / 322.5 -> 319.5 / 414.7 -> 417.3), the canonical
27B Q8_0 prefill-logit gate **0.000000 / 100.000 %**, per-block build 2/2, `validate-set.sh` green, no GPU
faults.  Numbers: `benchmarks/2026-10-30-graph-order-0003-r180.md`.

#### r198 — the dense row concat across a K-quant x Q6_K pair (PR #260)

**What it is.**  The r177 RDNA4 row concat required all three Q/K/V weights to share a type, but in the
K-quant "M" mixes `attn_v` is `Q6_K` in about half the layers while Q/K are `Q4_K`, `Q5_K` or `IQ4_XS`
(Qwen3-Coder-30B-A3B Q4_K_M: 12 of 48 attention layers; Qwen3.6-35B-A3B: 6 of 10), so V kept its own
launch.  `mul_mat_vec_q_ksplit_cat2<type, Q6_K>` runs the `Q6_K` body (with that segment's own row stride)
for V's row blocks and the launch type's body for the rest; both types must take the same nwarps (a
`static_assert`), every segment's row count must be even (a 2-row block must not straddle a boundary), and
the launch tensor must be the largest segment (its row count sizes the row guards).  The matcher takes
either graph order (the r180 rotate puts V first) and, in the mixed case, requires every segment that runs
the **launch** type's body to share the launch tensor's `nb[1]` as well as its type — the kernel passes one
row stride for those blocks, so a padded view would otherwise be read at the wrong offset.  Kill switch
`GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` (it restores exactly the same-type-only matcher).

**The codegen lesson (this is why the first cut cost 7 % of MTP).**  The first submission turned
`mul_mat_vec_q_ksplit` into an inlined `__device__` body called by the kernel.  That compiled the
**multi-column verify kernels** differently on gfx1201 (Q4_K ncols 5: 118 -> 97 VGPRs, 864 -> 1029
instructions), and cost **6-7 % of MTP decode** (dense 27B, `draft-mtp`, 1 GPU 76.2 -> 70.7 t/s, 3 GPU
`-sm tensor` 133.4 -> 126.4) with an identical kernel/graph census, compute buffers, text and acceptance.
The re-cut moves the body verbatim into `mmvq-ksplit-body.inc` and `#include`s it into the unchanged kernel
and into the cat2 launch, so the pre-existing kernel's source tokens (and its compiled code) are unchanged —
verified with a per-kernel VGPR census of the MTP profile (the only new kernels are the two cat2
instantiations).  This is `GREEDY-PURITY.md` §80 and closed `TODO` #128.

**Reach and worth.**  Fires only on a GGUF whose attention mixes a K-quant with `Q6_K` (the UD files carry
`Q8_0` attention and are inert).  Qwen3-Coder-30B-A3B Q4_K_M, f16 KV, 3 interleaved rounds: tg128
**161.54 -> 162.57 t/s (+0.6 %)**, @4096 **137.69 -> 139.14 (+1.05 %)**, 3 GPU `-sm tensor` tg128
**171.62 -> 173.53 (+1.1 %)**.  Census: 168 `cat2` launches replace 168 standalone `Q6_K` launches over the
8-step run.  The MTP arm is fully restored: 1 GPU **76.4/76.4/76.3/76.5/76.3 -> 76.3/76.2/76.3/76.4/76.3**
and 3 GPU tensor **133.0/134.1/133.9 -> 133.8/134.2/133.8**, drafts 417/360 and 670/575 identical.

**Purity.**  Every row still computes its own one-warp dot product, so the concat is bit-identical: 30B-A3B
greedy text `1065 chars sha=000d77b327e9` (f16), `1131 chars sha=9fb5b1832d34` (q8_0 KV) and
`1019 chars sha=bbdb61ea2adb` (3 GPU tensor) identical base vs patch; 27B `draft-mtp n3`
`1270 chars sha=5dbf95984c65` identical.  `gate-prefill-logits.sh --ub 1 --ab
"GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0"` on the 30B (the model that reaches the cat2 path) reads **0.000000 KLD
/ 100.000 %**, and the canonical 27B Q8_0 gate **0.000000 / 100.000 %**.

**Cross-arch.**  The cat2 body is compiled out off RDNA4 (`if constexpr (table_id == RDNA4)`, else
`NO_DEVICE_CODE`): gfx1100 and gfx1151 build clean, run **0** cat2 launches, read byte-identical text
(`30B-A3B 1263 chars sha=8392d4044a0e` gfx1100, `35B-A3B 1177 chars sha=21eee00556dd` gfx1151) and decode
flat (gfx1100 208.99/211.03 -> 211.30/211.30 t/s, gfx1151 63.31/63.13 -> 63.03/63.03).

**Gates.**  The four-axis adaptive gate at `-n 3000` (27B, 1 GPU) is identical on every axis, including
acceptance: R 58.5/58.4 t/s (3303/1897 both), P 72.0/72.0 (2160/1730), C 80.0/80.1 (2408/2196), K 85.5/85.8
(375/372).  Rule-5 `llama-batched-bench -npl 1,4,8` (27B, q8_0 KV, 3 GPU tensor) **48.96/159.88/267.24 ->
49.91/162.90/270.66**.  `test-backend-ops -o MUL_MAT` 2445/2445; per-block build 2/2; no GPU faults.
Numbers: `benchmarks/2026-11-01-row-concat-mixed-r198.md`.

#### r182 — the 1-token router fold (PR #253)

**What it is.**  A 1-token MoE decode runs the router (`ffn_gate_inp`) as `mul_mat_vec_f<float,float,1,256>`
and then the fused topk-moe as a second launch.  `topk_moe_cuda` gains a `router` mode: block `b` computes
logit `b` with the same thread map, `float2` loads, `#pragma unroll 4`, `ggml_cuda_mad` calls and warp +
cross-warp butterfly as `mul_mat_vec_f`, writes it to the (still written, still allocated) logits tensor, and
the block that finishes last — a `__threadfence()`, then one atomic on a self-resetting counter in the last
`moe_red_counters()` slot — runs the **unchanged** selection.  One launch replaces two; the dispatcher arm
lives in `ggml_cuda_topk_moe_match_at` + `ggml_cuda_router_topk_moe_ok` (r176's topk-moe matcher refactored
out so the router path can reuse it).

**Worth (1 GPU, `llama-bench -r 5`, two interleaved rounds, on vs off).**  gfx1100 Qwen3-30B-A3B Q4_K_M
(128 experts) tg128 **205.1 -> 212.0 (+3.1 %)**, d4096 **180.2 -> 183.1 (+1.7 %)**; Qwen3.6-35B-A3B
(256 experts) tg128 **141.1 -> 142.8 (+1.2 %)**, d4096 **+1.1 %**; gfx1201 the 128-expert model **+1.7 %**
(noisy) and the 256-expert model flat, because it **steps aside on RDNA4 above 128 experts** (`0002`), where
the one launch measured -0.3..-0.5 % on our R9700 and -0.8..-1.0 % on the submitter's.  Kernel census: the
per-layer router GEMV is gone; the `mul_mat_vec_f<...,1,256>` that remains is the **shared-expert** gate.

**Facts a future edit must keep.**

* **It is a 1-token shape; the width is the gate.**  `ggml_nrows(logits) == 1` in the dispatcher and
  `n_rows = 1` in the launch.  A verify batch (`n_tokens > 1`) must not fold: `mul_mat_vec_f` accumulates in
  a different order at `ncols_dst > 1` and the selection's row map is per-row blocks.  With a speculator the
  fold is therefore inert (a verify batch is wider than one token); it is a plain-decode win.
* **The election is a device-wide barrier inside one launch, and it needs the logits write.**  The fold
  elides the *launch*, never the tensor or its write, so another consumer of `logits` still works (any
  consumer is later in node order).  Do not replace the write with a private scratch buffer.
* **The counter is the last `moe_red_counters()` slot** (`GGML_CUDA_MOE_RED_CNT - 1`), which the MoE down fold
  cannot reach (`mmvq-moe.cu` indexes `red_cnt + blockIdx.x` under `dst->ne[0] <= 2*CNT - 2`).  Per (device,
  stream), zeroed at allocation, reset before return; a launch aborted mid-kernel leaves it non-zero (the same
  caveat the down fold carries).
* **The predicate mirrors the dispatcher's chain, in its own order** — F32 weight/activation/output,
  contiguous, `x->ne[1]` in 64/128/256, wave32, `!mmb_supported_mm`, `should_use_mmvf`, the same block-size
  choice, plus a refusal of a bias, a delayed softmax, a Hadamard-hinted router and a padded compute-buffer
  `src0`.  Keep it in sync with `ggml_cuda_mul_mat` (the `ggml_cuda_mul_mat_takes_mmq` discipline).
* **`0002` is a per-device tuning step-aside, not a correctness rule**; a new arch needs its own measurement.
  `GGML_CUDA_ROUTER_TOPK=0` is the A/B arm and the kill switch.

**Gates.**  Greedy text byte-identical on/off (30B-A3B on gfx1201 + gfx1100, 35B-A3B on gfx1100); the
**decode-band** prefill-logit probe `gate-prefill-logits.sh --ab "GGML_CUDA_ROUTER_TOPK=0" --ub 1` (every
token a 1-token graph, the fold firing) **0.000000 / 100.000 %** on 30B-A3B (gfx1201) and 35B-A3B (gfx1100);
canonical 27B Q8_0 gate **0.000000 / 100.000 %**; `test-backend-ops -o TOPK_MOE` **608/608**; MTP Protocol A
unchanged (51.53 t/s, acceptance 0.62134); rule-5 batched bench inert at npl>1; per-block build 2/2;
`validate-set.sh` green; no GPU faults.  Numbers: `benchmarks/2026-10-30-router-topk-fold.md`, record
`archive/work/router-topk-fold/PROMOTED.md`.

#### r183 — the GDN decode folds (PR #256) and the 2-D copy guard

**What it is.**  PR #256's four patches, all of them the follow-up the #249 review asked for, plus a load-abort
fix found while setting up its Flash-Next arm.

* **The direct conv also writes the recurrent snapshot tails** (`GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` off), so
  where the direct conv runs the `CONCAT` and the snapshot `CPY`s launch nothing.  Taken only when a runtime
  byte-span check shows nothing between the CONCAT and the conv writes over `state`/`x` or touches a snapshot
  destination and the destinations are disjoint; the batched-copy run stops at a folded snapshot `CPY` so
  `cpy_batch` cannot swallow one.
* **The conv-state gather fold** (`GGML_CUDA_FUSE_GDN_CONV_GATHER=0` off): `build_rs`'s `GET_ROWS` of the conv
  cache, feeding a CONCAT the direct conv takes, is deferred like the GDN state gather, and the direct kernel
  reads the cache row itself (row id = the gather's `src[1]`) with each channel's three state values held in
  registers before the snapshot writes, one of which is the write-back to that row.  One sequence, at most 16
  tokens, the CONCAT the gather's only reader, only folded snapshot `CPY`s writing the cache before the conv.
  A CONCAT over a deferred gather the direct conv does not take **aborts** rather than reading an unwritten
  `GET_ROWS` (the tripwire is asserted in `write_tail` and at the CONCAT).
* **The 1-token direct conv** (`GGML_CUDA_FUSE_GDN_CONV_DECODE=0` off): the #249 re-submission, now gated on
  `gdn_conv_norm_match() > 0` **and** on the decode-only `ssm_prescan` not taking the conv first, with the
  `ssm_conv_in` stand-down using `continue` instead of `return 0`.  The gate is data-pointer dependent, so it
  lives in the compute-time matchers (`match_at_concat` / `match_at_conv`, which every compute-time site calls);
  `graph_optimize` uses the structural `ggml_cuda_gdn_conv_pattern_at_concat`, a superset, so its alloc deps
  cover every case the compute-time matchers take.
* **The shared-expert gate fold** (`GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0` off): `MUL_MAT`(one gate row) ->
  `SIGMOID` -> `MUL` -> `ADD` runs in the `mul_mat_vec_f` launch the gate would take anyway; every block
  recomputes the one gate row and applies `sigmoid_mul_add_f32`'s own `op_sigmoid` and its
  `#pragma clang fp contract(off)` expression to its slice.  BF16/F32 gate weights only, decode/verify band,
  and the gate, the sigmoid and the product may have no other reader.
* **The 2-D copy guard**: `ggml_cuda_cpy_as_memcpy_2d` keeps upstream's `cudaMemcpy2DAsync` fast path only
  while the padded span (`height*pitch`) fits the tensor's own allocation.  A 2-D copy of a view of a
  *mirrored model weight* is refused with `invalid argument` when the weight sits in the slab region, which
  made `Flash-Next GSQ-IQ3_XXS -sm tensor -ncmoe 48` abort at load on r182 and earlier (device 1,
  `blk.1.ple_conv1d.weight`'s column view, qwen4exp's PLE conv; `GGML_CUDA_SLAB=0` and `-sm layer` both
  loaded).  The scalar kernels are the bit-identical fallback, and no other model we hold reaches the path
  (0 2-D copies in an instrumented decode+prefill run of 35B-A3B, 4B Q8_0, 27B and gpt-oss).

**Worth (1 GPU, `llama-bench -r 5`, two interleaved rounds, all four on vs all four off).**  gfx1100
Qwen3.6-35B-A3B Q4_K_M tg128 **142.3 -> 150.6 (+5.9 %)**, d4096 **139.0 -> 146.9 (+5.6 %)**; gfx1201 the
same model **105.0 -> 108.4 (+3.3 %)**, d4096 **+3.4 %**; dense 27B UD-Q4_K_XL **+1.7 %**; Qwen3.5-4B Q8_0
neutral.  Decomposed on gfx1201: `0003` alone **+0.3 %** (it is the enabler, not the win — exactly what #249
measured), the snapshot fold **+2.2 %**, the gather fold **+1.0 %**.  The shared-expert gate fold is inert on
every quantized model we hold (its gate weight must be BF16/F32) and measures **+0.6 %** on Flash-Next
GSQ-IQ3_XXS tg128.  The submitter's +7.2 % (7900 XTX) / +5.6 % (R9700) against r176 reproduce in direction;
against r182 they measure +5.9 % / +3.3 % here.

**Facts a future edit must keep.**

* **The folds were inert on a multi-device graph until r186** (the pre-existing `gdn_conv_enabled()` required
  one device unless `GGML_CUDA_CONV_FUSION_MULTI=1`).  **r186 removed that gate** — see the section below — so
  the recommended `-sm tensor -ncmoe` server shape now gets all of them.  The sentence in the r183 release note
  that said otherwise is historical.
* **`0003` must keep the `gdn_conv_norm_match() > 0` gate.**  Relaxing it re-opens the #249 loss: where the
  norm fold is refused the `ssm_conv_in` epilogue's free CONCAT write is replaced by a separate launch.
* **The compute-time matchers and `graph_optimize` must keep deciding the same way** — the decode gate reads
  data pointers, so a new compute-time caller goes through `ggml_cuda_gdn_conv_match_at_concat` /
  `_at_conv`; only `graph_optimize` may use the structural `_pattern_at_concat` (a superset).
* **The gather fold's tripwire is deliberate** (an abort, not a wrong read), and the direct kernel's
  "read the state row, then write it back" order (preload into registers first) is load-bearing.
* **`0004` elides stores, never tensor identities**: it needs BF16/F32 gate weights and needs the gate,
  sigmoid and product to have no other reader.

**Gates.**  Greedy text byte-identical on/off (35B-A3B 1 GPU, 4B Q8_0, 2-GPU `-sm tensor -ncmoe 40`, Flash-Next
`-sm layer`); `gate-prefill-logits.sh --ab "<all four off>" --ub 1` **0.000000 / 100.000 %** on 27B Q8_0;
canonical 27B Q8_0 gate **0.000000 / 100.000 %** against the r182 base; MTP on 35B-A3B `n_max 7` identical
(362/203); `--parallel 2 --kv-unified` server clean; Flash-Next `-sm tensor -ncmoe 48` now loads (and matches
the `GGML_CUDA_SLAB=0` arm's text); per-block build 3/3 `--fast`; `validate-set.sh` green; no GPU faults.
Numbers: `benchmarks/2026-10-30-gdn-decode-folds.md`; records `archive/work/gdn-decode-folds/PROMOTED.md`
and `archive/work/cpy-2d-memcpy-guard/README.md`.

#### r186 — the conv fusion (and all four r183 folds) run on every device count

**What it is.**  `gdn_conv_enabled()` / `ple_conv_enabled()` no longer test
`ggml_backend_cuda_get_device_count() <= 1`.  On 2026-09-23 (r13) the depthwise conv1d fusion was found to
produce a different greedy continuation under a 3x R9700 `-sm tensor` split while the raw ops and the upstream
`SSM_CONV+SILU` fusion agreed byte for byte, and it was gated to single-device graphs until root-caused
(`archive/work/closing-the-gap/2026-09-23-gfx1201-conv-fusion-tensor-split.md`).  **That divergence no longer
reproduces**, so the gate is gone: the fusion (and the r183 snapshot / gather / 1-token folds that ride it)
now run on `-sm tensor` and `-sm layer` as well, which is the recommended multi-GPU server shape.

**Evidence (gfx1201, r186 tree).**  `test-logits-width-probe` at `P=4096`, fused default vs
`GGML_CUDA_DISABLE_CONV_FUSION=1` — bit-identical `W8` and row-0 hashes on eight arms: 27B IQ3_S on 1 GPU,
2 GPU (`0,2`) tensor, 3 GPU tensor and 3 GPU layer; 35B-A3B Q3_K_M 3 GPU tensor and 3 GPU layer; 35B-A3B
Q4_K_M 3 GPU tensor with `-ncmoe 40`, with and without the expert-cache arena — with the direct conv
confirmed running on the shard (`C=2560` for 3 devices, `C=5120` for 2) and the snapshot fold active
(`snap=1`).  Greedy text identical on 4B Q8_0 (1 GPU, 3 GPU tensor), 27B Q4_K_XL (3 GPU tensor + layer) and
35B-A3B Q4_K_M (3 GPU tensor, and `-ncmoe 40`).  MTP Protocol A identical text **and** acceptance on both
split modes (3 GPU tensor: acc 0.63244 = 2445/3866; 3 GPU layer: 0.58861 = 1850/3143).  A DFlash2 draft arm
(35B-A3B + `draft-dflash n_max 5`) identical text and acceptance (0.52806 = 1449/2744).  The prefill-logit
gate on 3 GPU tensor reads mean KLD **0.000000** / same-top-p **99.990 %** for the fused arm against the
unfused one.  `test-backend-ops` SSM_CONV 60/60, CONCAT 183/183, CPY 251/251, 3/3 backends; per-block build
2/2 `--fast`; no GPU faults.

**Worth (3 GPU `-sm tensor`, `llama-bench -r 5`, three interleaved rounds, fused vs disabled).**
27B IQ3_S tg128 **58.52/58.54/58.72 -> 61.85/61.90/62.01 (+5.7 %)** and pp2048 **1938.9 -> 1965**ish
(+1.1 %); 35B-A3B Q4_K_M tg128 **118.7 -> 122.1 (+3.0 %)** with prefill flat.  Rule-5
`llama-batched-bench -npl 1,4,8` total t/s flat (+0.4 / -0.2 / -0.1 %).  Numbers:
`benchmarks/2026-10-31-conv-fusion-multi-device.md`.

**Facts a future edit must keep.**

* **The divergence was never root-caused** — the reach is *empirical*, not proven.  The likely fix is r183's
  rework of exactly the suspect area (the snapshot write moved into the conv kernel and the conv-state gather
  was deferred into it), but nothing establishes that.  A config outside the matrix above is unproven, which
  is why release gate 14 carries a multi-device arm.
* **`GGML_CUDA_CONV_FUSION_MULTI` is retired, not renamed.**  It only ever countermanded the gate; forcing the
  fusion on is now the default.  `GGML_CUDA_DISABLE_CONV_FUSION` (§7.3, presence-only) is the off switch and
  the A/B arm everywhere.
* **Do not re-add a device-count test to "restore purity".**  If a divergence is ever found again, gate it on
  the *config* that shows it with a witness, not on a device count that also disables the bit-identical
  `-sm layer` arms.

#### r188 — the shared-expert lane's Q8_1 publish (PR #259)

**What it is.**  The r175 GLU Q8_1 publish covers the **routed** lanes only: the 1-token fused MoE gate/up+GLU
kernel (`mul_mat_vec_q_moe`) writes each routed expert's GLU row into the quantize cache, so the routed down
`MUL_MAT_ID` skips its `quantize_q8_1`.  A model with a **shared expert** writes one more lane (the last
`blockIdx.y`, into `fusion.shared_dst`) and the shared down `MUL_MAT` then ran its own quantize.  The shared
lane now publishes too, into its own cache entry keyed on the **shared GLU node** (`fusion.shared_dst`, the
key that down looks up), with its counters in the lane **after** the routed ones
(`lane = gridDim.y - 1 = nchannels_dst`) and the same 32-lane `warp_reduce_max/sum` + `roundf` arithmetic and
zeroed padding blocks as `quantize_q8_1`.  `GGML_CUDA_MOE_SHARED_GLU_Q8=0` is the kill switch; it is nested
inside the routed arming block, so `GGML_CUDA_MOE_GLU_Q8=0` (and every condition that stands the routed
publish down: not a 32-lane wave, not one token, `n_ff % QK8_1 != 0`, non-contiguous `dst`, the counter
budget) stands both down.  That nesting is the invariant, not an accident — it is what guarantees the host
arms exactly the predicate the kernel implements, so a cache entry cannot exist that nothing filled.

**Reach is a quant property (the one fact that decides whether a claim about this is meaningful).**  Block
08's `shexp_down_gate` fusion (`shexp_down_gated_q8_0`) already consumes the shared GLU + shared down when
`ffn_down_shexp` is **Q8_0**, and every Unsloth **UD** quant keeps that tensor at Q8_0.  On those files the
shared down node does not exist, so this publish is inert — measured **839 launches/token before and after**
(`UD-Q4_K_M`), with `shexp_down_gated_q8_0` in the census.  It fires where the shared down is its own node
(conventional quants: the LMStudio `Q4_K_M` with Q6_K shared tensors, `UD-IQ3_S`), which is the file class
the submitter measured.  **Quote the GGUF with any number.**

**Worth.**  One launch per layer: Qwen3.6-35B-A3B launches/token **748 -> 708** (gfx1201) and **688 -> 648**
(gfx1100).  LMStudio `Q4_K_M`, 1 GPU, three interleaved rounds: gfx1201 tg128 **139.54 -> 141.10 (+1.1 %)**,
@4096 **136.47 -> 137.53 (+0.8 %)**; gfx1100 (7900 XTX) tg128 **177.90 -> 182.56 (+2.6 %)**, @4096
**178.16 -> 182.28 (+2.3 %)**.  Rule-5 `llama-batched-bench -npl 1,4,8`: +1.1 % at 1 slot, flat at 4/8 (a
multi-token batch never publishes).

**Gates.**  `gate-prefill-logits.sh --ab "GGML_CUDA_MOE_SHARED_GLU_Q8=0" --ub 1` **0.000000 / 100.000 %** (the
published blocks are `quantize_q8_1`'s, bit for bit); default-ub prefill logits unchanged; greedy text
byte-identical at 1 GPU, 2 GPU `-sm tensor` (both x8 pairs), 3 GPU `-sm tensor`, 3 GPU `-sm layer`, a
`-ncmoe 20` arm and on gfx1100; MTP identical text **and** acceptance on a dense 27B in both split modes
(inert) and on a firing MoE arm (35B-A3B `UD-Q4_K_M` with `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`);
`test-backend-ops -o MUL_MAT_ID` 931/931; no GPU faults.  Rule: `GREEDY-PURITY.md` §75.  Numbers:
`benchmarks/2026-10-31-shared-glu-q8.md`; record `archive/work/shared-glu-q8/README.md`.

#### r189 — the r183 2-D copy guard gets a vectorised fallback (`cpy_2d_vec16`)

**What it is.**  r183's guard (above, `ggml_cuda_cpy_as_memcpy_2d`) keeps `cudaMemcpy2DAsync` only while a copy's
**padded** span fits the tensor, and sends everything else to `cpy_scalar<cpy_1_scalar<float,float>>` — three
integer divisions and three modulos per element.  That fallback is reached in the delivery: qwen4exp with a
quantized indexer key cache (`-ctk q8_0`; the fused `INDEXER_SCORE`/`INDEXER_FILL` path is F32/BF16/F16 only)
takes the per-op chain and makes **one refused copy per layer per prefill chunk** (`w=512 sp=2048 dp=512`,
`h` = the QSA index-block count).  r189 adds `cpy_2d_vec16`: when the guard refuses a 2-D copy of two **F32**
tensors whose width and pitches are multiples of 16 B (both pointers 16-B aligned), copy the same block with one
16-byte unit per thread and the row index from the grid — bit-identical, no division.  Every other type, every
unaligned geometry and every copy the guard allows is unchanged, and **the guard itself is untouched**.

**The refusal is signalled, not inferred.**  `ggml_cuda_cpy_as_memcpy_2d` writes `width` before its
`spitch < width || dpitch < width` rejection as well, and that rejection is a *different* case (self-overlapping
rows, which need the flat element mapping), so the caller gets an explicit `guard_refused` out-parameter.  A
"non-zero width means the guard refused" test would send such a geometry to a row-per-row copy.  Rule:
`GREEDY-PURITY.md` §76.

**Worth — a regression fix, not a perf win, and it must be recorded as one.**  Refused-copy kernel time on a
105k-token prefill (gfx1201): **1010 -> 637 ms** (`cpy_scalar<float,float>` 1433.75 ms / 12428 calls ->
423.91 ms / 2732 + `cpy_2d_vec16` 636.77 ms / 9696); wall time **flat** — `pp16384` 339.3 vs 339.3, `tg64 @
d49152` 26.7 vs 26.7, deep-arm generation 30.9 vs 30.4 t/s — because on the delivery's arms the copies are
prefill-phase and ~0.3 % of the prefill's kernel time.  The campaign that found it measures 5.20 -> 1.83
ms/token on its sparse/pool **decode** shape; the delivery's refused copies stay prefill-phase even with
`LLAMA_KV_RESIDENT_CELLS` set (no per-step refused copy appears), so that decode delta is the campaign's arm,
not ours.

**Gates.**  Greedy text byte-identical at 4k (`905 6b9e214d3b59`) and **105k depth** (`94 fe7775042f82`), 3-GPU
`-sm tensor -ncmoe 48` qwen4exp q8_0 (`611 15561552e5d1`) and dense 27B 3-GPU (`678 692c3f7ca639`);
prefill-logit **0.000000 / 100.000 %** on qwen4exp q8_0 and **-0.000000 / 100.000 %** on the canonical dense
27B; `test-backend-ops -o CPY` 251/251; the r183 load arm (Flash-Next GSQ-IQ3_XXS, 3 GPUs, `-sm tensor
-ncmoe 48`) loads and generates; no GPU faults.  One trap: a fresh-process text run without
`ROCBLAS_USE_HIPBLASLT=0` read a different hash — the per-start hipBLASLt pick of issue #67 (§44), not a change.
Numbers: `benchmarks/2026-10-31-cpy-2d-vec16.md`; record `archive/work/cpy-2d-vec16/README.md` (the promoted
`archive/work/host-experts-window-beta` (beta 0016, withdrawn 2026-11-02), from the KV campaign's `f8e47e392`).

## Block 11 — qwen4exp: model, hyper-connections, lazy reader, QSA/indexer

> **r217 amendment (issue #273 item 4, `getenv` audit):** this block's switches (`ggml-cuda-try-fuse.cu`,
> `hc-mix.cu`, `indexer-score.cu`) and the recurrent rollback debug flag in `llama-memory-recurrent.cpp` are
> read once per process (the reporter's item 4: `LLAMA_RS_ROLLBACK_DEBUG` was a `getenv()` on every
> speculative rollback and on every recurrent `prepare()`).  No behaviour change: every value is a constant
> for the life of the process either way, and the arms are bit-identical.

**What it is.**  The qwen4exp (Qwen3.8-Flash-Next) architecture support, split out of block 10 in r109 so
the model graph, the sparse-attention op and the hyper-connections no longer share a commit with the slab,
the fusion dispatcher and the async input ring.  QSA moves with the model by the maintainer's decision
(it was added along with qwen4exp).  **The infrastructure it builds on is block 10** — this block adds no
slab/dispatcher/ring code of its own beyond the arms implemented in its own TUs.

**Contents.**

* **The QSA combine launch passes the FA combine's new `q8` argument as `nullptr` (r175), and its `had_dst`
  as `nullptr` (r187).**  Block 10's FA-combine Q8_1 publish added a `block_q8_1 * q8` parameter to
  `flash_attn_combine_results`; QSA's own combine (`fattn-qsa.cu`, created by this block) publishes nothing
  because its output is not fed to a quantized matmul.  r187's rotation fold (block 10,
  `GGML_CUDA_FA_COMBINE_FWHT`) added a trailing `float * had_dst` for the same reason — the apply has no
  quantized-KV output rotation either — so this block's call site carries `(float *) nullptr` for it.  A new
  combine call site owes both decisions explicitly.

* **The HC down/inject pairs are one tensor (r138, upstream #31028).**  `create_merged(
  LLM_TENSOR_HC_{ATTN,FFN}_DOWN_INJECT, …)` accepts the converter's merged weight and `build_hc_mix(…,
  w_down_inject, …)` does **one** GEMM for both products (the delivery's earlier form emitted two and
  relied on `bf16_src1_cache`).  The separate tensors are still accepted — the loader falls back when the
  merged tensor is absent — and the **MTP branch stays the delivery's own** attention (QSA top-k +
  derived visibility), not upstream's `build_layer_attn(...)`/fused-mask path.
* **qwen4exp** (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`): QSA sparse attention and the
  fused indexer top-k/score (`indexer-topk.cu`, `lightning-indexer`), hyper-connections (`hc-mix.cu`,
  `hyperconn.cu`), the managed lazy reader, PLE n-gram loading, the MTP draft head, and the **keys-only
  indexer cache** (`LLAMA_QSA_KEYS_ONLY`; no V tensor at all — `llama_kv_cache` gains `v_enabled`).  The
  indexer cache is a cell-for-cell mirror of the attention cache, so anything that **moves** attention cells
  must move it too — the idle-slot parking of block 12 does (r146, issue #227).
* **The DFlash/DSpark layer-input tap exports the HC-averaged state** (r121, TODO #94 / issue #211): the
  graph used to publish `t_layer_inp[il] = res_hc`, the **wide** hyper-connection residual (`hc * n_embd` =
  10240 on Flash-Next), while `llama_context::extract_layer_inp()` requires exactly `n_embd` floats per row
  — so attaching any DFlash2 head to a qwen4exp target aborted at
  `GGML_ASSERT(row_floats == model.hparams.n_embd)` (llama-context.cpp:3437), and the F1 device path hit its
  `t->ne[1] == n_tokens` assert instead.  `graph::build_hc_mean()` (same 8 lines as deepseek4/glm5-next)
  now collapses the streams, and the export is **gated on `cparams.embeddings_layer_inp[il]`** like
  deepseek4's, so a graph that taps nothing (plain decode, MTP, QSA) gains no node at all — plain decode is
  byte-identical, and Flash-Next GSQ prefill logits are unchanged (PPL 1.9404, mean KLD −0.000000,
  same-top-p 100.000 %).  The convention is settled by the head author's own sglang capture patch
  (`hidden_states.unflatten(-1, (hc, hidden)).mean(dim=-2)`, captured **after** the layer call, so
  `completed i` = llama.cpp's `t_layer_inp[i+1]` = the GGUF's `target_layers`).  Mean vs stream-0 measured
  on recall: acceptance length 4.32 vs 2.41.  Known limit: index `n_layer` (the "output of the last layer",
  which deepseek4 also supports) is not exported — after the loop `res_hc` may be row-gathered, so it needs
  the pre-gather residual; no DFlash2 head taps the final layer ([6,16,26,36,46]/48, [6,20,34,48,62]/64).
* **The QSA/indexer ops**: `fattn-qsa.{cu,cuh}` (VEC and the attached packed-block views),
  `fattn-qsa3.cu` (the packed-block WMMA prefill kernel, which takes its wide-phase workspace from the
  block-10 slab), `indexer-score.{cu,cuh}` (the fused score / fill plus the prefill relu-sum fusion),
  `indexer-topk.{cu,cuh}`, the 4-head WMMA scorer in `lightning-indexer.cu` (RDNA3_5 default, RDNA3_0
  opt-in, RDNA4 default), the dispatch (`GGML_OP_FLASH_ATTN_QSA` / `INDEXER_TOPK` / `INDEXER_SCORE` /
  `INDEXER_FILL`), the CPU references, the scheduler op allowlists and the Meta split states.
* **HC ops** `GGML_OP_HC_MIX` / `HC_COMBINE`: the ops and CPU references, the Meta split states, the
  `hc_combine_mix` / `hc_mix` / `hc_combine_norm` / `hc_gate_mix` fusion arms, the HC16 planar layout, and
  the graph-optimizer passes they need (the `LLAMA_HC_BLK16` / `LLAMA_HC_RES16` BF16-stream marks and
  their alloc deps — default OFF; the MMB side that consumes the marks is block 10).
  **The `hc_combine_norm` matcher reads the tensor the model pins (r140, issue #226).**  The repeat-anchored
  window starts at the `REPEAT`, so `scale1 -> sigmoid -> scale2` run standalone before it; the matcher
  therefore takes `inject = scale1` — the node `build_hc_combine` pins with `ggml_set_output(inject)` — with
  identity op params (`s1 = 1`, `b1 = 0`) and **requires** `GGML_TENSOR_FLAG_OUTPUT`.  It must not descend
  to `scale1->src[0]`: that tensor's last graph use is `scale1`, so the allocator hands its buffer to the
  chain's own SIGMOID, and r138's #31028 merge (which made `inject` a `SCALE`) did exactly that on a
  pre-#31028 separate-tensor GGUF — PPL 2690 instead of 2.45.  The separate **scale1-anchored A-chain**
  matcher keeps `scale1->src[0]` on purpose (its window starts *at* `scale1`, which is elided, so the read
  is live and `add`/`mulg` are already alias-checked).  Two r141 additions make the fusion faithful *and*
  reachable: the kernel owns the same columns in the RMS sum as `rms_norm_f32`'s vec4 path (its divergence
  from its own unfused chain was 0.0185 KLD / 95.6 % same-top-p and is now 0.000000 / 100.000 %, greedy text
  byte-identical to `GGML_CUDA_DISABLE_HC_COMB=1`), and `build_hc_mix`'s **merged-tensor** branch
  materialises the inject slice with `ggml_cont` so a post-#31028 GGUF is fusable too — the matcher accepts
  any F32, contiguous, `hc*n_tok`-element, **pinned** `inject` with `view_src == nullptr` (a pin on a view
  does not keep the root alive) and no longer requires a `SCALE`.  `GREEDY-PURITY.md` §59.
  **The separate-tensor mixer's inject GEMM is emitted next to the down GEMM (r142, issue #226
  follow-up).**  Both are BF16 GEMMs over the same F32 `xn` (the combine+norm output), and the CUDA BF16
  GEMM path holds **one** converted src1 (`bf16_src1_cache`, block 10) — so the reuse only happens if
  nothing converts in between.  #31028's shape emits the inject product *after* the gate product, whose
  src1 is the silu output and therefore evicts the entry, which converted the whole
  `[hc*n_embd, n_tok]` tensor twice per mixer: measured on Flash-Next GSQ-IQ3_XXS (3 GPU `-sm tensor`,
  all-VRAM, pp8192) as 4560 vs 2280 `convert_unary_cont_vec4<float, bf16>` launches (274 vs 85 ms in one
  pp2048 pass) and **pp8192 2205 -> 2282 t/s (+3.5 %), pp31500 2154 -> 2229 (+3.4 %)**.  The fix emits
  `lo` and the inject product back to back in the `else` (separate-tensor) branch and expands **both**
  there — `ggml_build_forward_expand` order, not node creation order, is what lands a node in the graph.
  The merged path is untouched (it is one GEMM).  Bit-identical: PPL 2.4392 / 2.4370 (both HC_POST arms)
  and all 24 chunks of the reporter's `--chunks 24` command unchanged, 6/6 same-seed greedy cells
  byte-identical, fused-vs-chain KLD still 0.000000 / 100.000 %.
  **The inject tensor carries the 1/hc-scaled scatter logit, in BOTH halves (r148, issue #228).**  The
  chain's contract is upstream's: `build_hc_mix` emits `inject = w_inject^T xn / hc` and
  `build_hc_combine` applies only `2*sigmoid(inject)`.  Through r147 the **fused** mixer published the RAW
  dot product and the **fused** combine divided it by `hc` again, i.e. the two fused ops encoded the *v14*
  convention (`inject` raw, the SCALE in the combine) and the `6050b467c` re-base inverted it under them.
  That pair is self-consistent, so the Q8_0/BF16-hc models (which take both fused halves) stayed right —
  but each half can be selected independently (`LLAMA_FUSED_HC_MIX` / `LLAMA_FUSED_HC_COMBINE`), and the
  other two arms double-scaled (chain mixer + fused combine: every hc-down quant that is not Q8_0/BF16, so
  Flash-Next Q5_K/IQ4_NL/Q4_K collapse with a greedy repetition loop) or did not scale at all (fused mixer
  + chain combine: an **empty** generation on a BF16-hc model).  r148 folds the `1/hc` into the fused
  mixer's inject tail (Q8_0-inject, F32-inject/planar and BF16 kernels) and drops it from the fused combine
  (CUDA kernel, fused combine+norm kernel, CPU reference), so all four arms reproduce the chain bit for
  bit — the BF16-hc default is **unchanged** (`246463129328`) and the chain arms are unchanged.  The oracle
  is `scripts/gate-hc-fusions.sh` (the four arms on a real GGUF); `test-backend-ops` cannot see this class
  because its CPU reference encoded the same convention.  `GREEDY-PURITY.md` §61.
* **The managed lazy reader** (`src/llama-lazy-reader.{cpp,h}`) and the loader's `managed_budget`
  (`LLAMA_LAZY_BUF_MB`, `TENSOR_SKIP_MANAGED`, `lazy_read::add_range`): a fixed-size host arena over the
  PLE n-gram table, clock-LRU, read on demand.  **Off by default** (it measured slower than both the mmap
  and the resident path on gfx1151); `LAZY_MODE_AUTO` without the env behaves as upstream auto.
* **Model/context plumbing**: the qwen4exp MTP draft context gets a hybrid-idx memory instead of a plain KV
  cache (`LLAMA_MTP_SPARSE=0` restores the dense draft), the recurrent memory stops warning about the
  speculative verify positions for a context whose recurrent filter matches nothing, the qwen4exp
  tensor-split arch gate in `llama-arch.cpp` (HIP-only; `ggml-hip/CMakeLists.txt` publishes `GGML_USE_HIP`
  to the main libraries so the dynamic-backend build reads it), and the prefetch exclusion of a
  managed-reader tensor.
* **The host-resident PLE gather prefetches its ubatch on Windows too (r209, PR #269).**  The Linux/macOS
  arm queues each distinct page with `MADV_WILLNEED` before reading the rows; the `_WIN32` arm of the same
  block now calls `llama_prefetch_rows`, which merges the rows and issues one `PrefetchVirtualMemory` — the
  helper gemma4 and deepseek41 already use (`llama_prefetch` resolves the entry point from `kernel32.dll`
  at run time, `_WIN32_WINNT >= 0x602`).  Ungated on both platforms (the Linux arm is too); the helper
  itself no-ops on a null/non-host buffer, so a device-placed table is untouched.  Reporter's arm: the
  Windows prefill figures in block 09's r209 note (measured together with the mapping release, no separate
  number — recorded in `archive/work/windows-host-memory/`).
* **Tests**: `test-backend-ops -o FLASH_ATTN_QSA` (the QSA CPU oracle), the `test_hc_mix` cases and the
  `test_hc_combine` / `test_hc_combine_mix` cases, `tests/test-lazy-reader.cpp`, and the lazy-reader
  handover in `test-llama-archs`.  **`HC_COMBINE` had no test case at all before r128** —
  `test-backend-ops -o HC_COMBINE` ran 0/0, so neither the op nor the default-on combine→mixer-norm fusion
  that consumes it had an oracle (TODO #100).  `test_hc_combine` compares the op against its CPU reference
  (the unfused `SCALE/SIGMOID/SCALE/REPEAT/MUL/ADD` chain) over both broadcast shapes, `hc` 2/4/8 and
  `nt` up to the CUDA band bound; `test_hc_combine_mix` is a whole graph, so it reaches
  `ggml_cuda_hc_combine_mix_try_fuse` (`GGML_CUDA_FUSE_HC_COMBINE_MIX=0` for the unfused control).

**Knobs.**  `LLAMA_QSA_*`, `GGML_QSA_*`, `LLAMA_HC_*` / `GGML_HC_*` (incl. `LLAMA_HC_BLK16` /
`LLAMA_HC_RES16` / `LLAMA_HC_GATEMIX`), `LLAMA_MTP_SPARSE`, `LLAMA_LAZY_BUF_MB` /
`LLAMA_LAZY_READER_STATS`, `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100` / `_GFX1201`.  `LLAMA_FUSED_HC_MIX` and
`LLAMA_FUSED_HC_COMBINE` are **two independent arms**, not one switch: every one of the four combinations
must reproduce the unfused chain (`scripts/gate-hc-fusions.sh`, once per hc-down weight class —
`GREEDY-PURITY.md` §61).

**Purity.**  qwen4exp's decode/verify band is `HC_FUSED_MAX_TOKENS = 8` and the QSA decode arm band is
`max(8, n_rs_batch)`; the arm must be chosen from the band and `n_kv`, never `n_tokens == 1`
(`GREEDY-PURITY.md` §§13, 16, 26, 32, 38).  The HC fused ops are the model chain's accelerators, not a
second definition of it: the inject *scatter logit* they publish is the chain's already-`1/hc`-scaled one,
so both halves reproduce `build_hc_mix`/`build_hc_combine` bit for bit in all four arm combinations (§61).
The QSA op has a CPU oracle — run
`test-backend-ops -o FLASH_ATTN_QSA` (18/18 minimum) and the sparse-vs-dense perplexity ratio, because the
`W = 1..8` matrix cannot see a width-uniform bug (§21).  Also run `scripts/gate-qwen4exp-quant-coherence.sh`
before changing the device gather or the head zero.  The HC ops have their own oracles now:
`test-backend-ops -o HC_COMBINE` (20 cases, the CPU reference for the fused residual combine) and
`-o HC_COMBINE_MIX` (6 cases, the default-on combine + BF16 mixer norm fusion) — §52; run both before
changing `hc_combine`, `hc-mix.cu`'s fusion arms, or the `HC_FUSED_MAX_TOKENS` / `HC_BF16_MAX_TOKENS` bands.

---

**Notes for the 3f03b1772 re-base (r127).**  Upstream's fused QSA-mask op (`GGML_OP_TOP_K_MASK`, #30640)
exists in the op tables next to this block's six ops (`FLASH_ATTN_QSA`, `INDEXER_TOPK/SCORE/FILL`,
`HC_MIX`, `HC_COMBINE`) and `GGML_OP_COUNT` is **108** after this block (109 after block 12's
`ALL_GATHER`).  qwen4exp does **not** use it: the QSA selection is still built in-kernel from the
derived-visibility keys (`additive` + `run_top_k`, no mask tensor), which is why the -800 MiB win
survives upstream's #30640/#30965 reworks.  The DFlash layer-input tap is upstream's
`hc_stream_mean()` (v148 `d9ee007a0`) — it is the same collapse this block used to carry as
`graph::build_hc_mean()` (TODO #94), so the delivery dropped its own copy; deepseek4 and glm5-next
keep their `build_hc_mean`.

#### r190 — the fused indexer scorer (and the derived indexer pool) accept QUANTIZED keys (TODO #118)

**What it is.**  The fused indexer path — `GGML_OP_INDEXER_SCORE`, which replaces `get_rows -> pool -> scale ->
rms_norm -> rope_multi -> score mul_mat -> relu -> head-sum -> bias add` — and the derived incremental indexer
cache (`GGML_OP_INDEXER_FILL`, `GGML_CUDA_QSA_INDEXER_CACHE`) both read raw cache rows natively and supported
**F32/BF16/F16 only**, so a quantized indexer key cache took the per-op chain and `pool_create` logged
*"derived indexer cache pool skipped (q8_0 keys)"*.  Since `-ctk` drives the indexer cache, that is
`-ctk q8_0` — the delivery's large-context KV type, and the type block 15's window and this block's pool takes by
construction (r189's copy fix above is what made the *chain* affordable there; this makes it unnecessary).
`indexer_score_ld_chan(row, c, nb0K, ktype)` replaces the element-address loader: a quantized row is addressed
in `nb0K`-byte blocks and dequantized with **ggml's own** `dequantize_q8_0/q4_0/q4_1/q5_0/q5_1/q4_nl` — the
same values `get_rows` gives the chain, which is what keeps the two paths byte-identical.  Two packings matter:
q8_0 holds one element per byte, while a 4/5-bit type puts a byte's low nibble at channel `b` and its high
nibble at `b + 16`.

Supported: **q8_0, the q4/q5 family and iq4_nl** — every quantized type whose 32-element block divides
`idx_dim` = 128.  **k-quants stay on the chain** (a 256-element superblock does not divide 128, so a row is not
a whole number of blocks) and that is an accepted limitation, not a TODO.  One predicate,
`ggml_indexer_key_type_supported()` (`ggml.h`/`ggml.c`), is what the two constructors, the CUDA kernels and the
memory layer all ask; `qwen4exp.cpp`'s `idx_key_float` became `idx_key_fused`, and
`llama-memory-hybrid-idx.cpp`'s `pool_create` builds the derived pool for the same set.  The A/B arm is the
per-op chain (`GGML_CUDA_QSA_INDEXER_SCORE=0`, pre-existing; `GGML_CUDA_QSA_INDEXER_CACHE=0` drops just the
derived pool).

**Worth** (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`, q8_0, 103k): plain decode decode-GPU-kernel time
**23.68 -> 17.73 ms/token** (**29.3 -> 36.3 t/s, +24 %**), the indexer bucket **5.31 -> 2.01 ms/token** — the
whole prize the bf16-key proxy arm had (17.83) with no KV-type change.  Under MTP-4 the chain is amortised over
~2.84 tokens/step, so the same change is **+0.8 %** there (sparse 54.26 -> 54.70, pool 54.52 -> 54.65, dense
58.63 -> 59.10).  The log now reads *"derived indexer cache (pool) = 12 layers x 128 dims x 1 streams,
ENABLED"* for q8_0.

**Gates.**  Greedy text **fused == chain on all eight key types at 5.3k** (q8_0 `296 91ef8940d66e` dense /
`332 dd1981213373` sparse — the pre-change hashes; f16 `333 c42a586a6d13`, bf16 `296 91ef8940d66e`, q4_0 and
q4_1 `307 097ff06194d6`, q5_0 `336 b82b7ed544c`, q5_1 `296 3205db133b1d`, iq4_nl `323 d7aba1c58e0c`), and q8_0
with `GGML_CUDA_QSA_INDEXER_CACHE=0` (fused, no derived pool) is the same hash, so the raw-pooling branch is
covered too; 51 475 depth q8_0 `854 43b30de151b9` (== pre-change) and q4_0 `841 65097e7c8630`;
`gate-prefill-logits.sh --ab "GGML_CUDA_QSA_INDEXER_SCORE=0" --ub 1` (the decode graph, chain as the base)
**0.000000 KLD / 100.000 % top-p**; `test-backend-ops -o FLASH_ATTN_QSA` pass.  Numbers and the census:
`../archive/work/kv-residency/BA-POOL-DESIGN.md` §9.3-§10.

#### r196 — plain decode takes the indexer top-k's BLOCK radix path (TODO #125)

The fused top-k (`GGML_OP_INDEXER_TOPK`, `indexer-topk.cu`) was already a radix select: four 8-bit passes over
the ordered F32 keys plus a deterministic ascending-cell gather, and a block key-shared path that does three of
the four passes at block granularity.  The gap was that **plain decode could not use the block path**: at
`n_tokens == 1` the graph passed the uploaded attention mask as the op's per-cell additive, and a per-cell
additive breaks the "every cell of a block shares one key" premise, so the op fell back to the cell-level radix
and re-read the `[n_kv]` F16 mask on every pass.  (The verify/MTP width already used the block path — derived
bias + derived visibility — so its 0.9 ms/step is unchanged by this.)

The per-cell half is now derived at ANY width: `build_qsa_top_k` hands the top-k `cell_vis`/`q_vis` instead of
the mask while the per-block bias stays the uploaded tensor (the fused decode score op applies it), so at
`n_tokens == 1` the block key is the score itself and `blk_idx`/`blk_tail` may be null.  The attention still
reads the mask (`qwen4exp_want_derived_vis` keeps its own `n_tokens > 1` gate), so this changes the top-k only.
`llm_graph_input_qsa::can_reuse` compares the same predicate through a stored `vis_supported` flag (the device
query needs the layer), and `indexer_topk_histogram_blocks` guards its incomplete-block skip on a non-null
`blk_idx`.  A new `test-backend-ops` case covers the null-`blk_idx` form against the CPU reference.

**Worth** (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`, q8_0, pool, plain decode, 103k, rocprofv3
short/long): the top-k bucket **2.062 -> 1.143 ms/token**, total decode kernel time **17.520 -> 16.363
ms/token**; wall A/B `-n 512`, interleaved, **40.48/40.48 -> 42.11/42.11 t/s (+4.0 %, -0.95 ms/token)**.
**Purity:** greedy text byte-identical to r194 at 5.3k (`332 dd1981213373`) and 103k
(`834 263da53898a3`), to `LLAMA_INDEXER_NOBLOCK=1` (block vs grouped-cell), and on the MTP `draft-mtp` depth-4
`-n 512` arm (`2259 f3553fb82f82`).  `INDEXER_TOPK` 4/4, `FLASH_ATTN_QSA` 26/26, prefill-logit 0.000000 /
100.000 % (decode-only), no GPU faults.  `GGML_QSA_DERIVED_VIS=0` is the kill switch for the derived per-cell
form (it also changes prefill policy, so it is not a no-op A/B; `LLAMA_INDEXER_NOBLOCK=1` is the isolation arm).

## Block 12 — campaign remainder

> **r217 amendment (issue #273 items 2, 3 and 4): `seq_rm` stops at the parked bound, the drafter's prompt is
> no longer copied per step, and this block's `getenv` reads are latched.**  (a) `llama_kv_cache::seq_rm()`
> scanned to `used_max_p1()`, which with `--kv-unified --parallel 2` is the whole cache because the parked idle
> slot sits at the top; a sequence that is not parked now stops at `used_max_p1_below(park_bound())` (a parked
> sequence holds only cells at or above the bound, and no other sequence has a cell there).  Measured on a
> 131 072-cell unified cache with a 64-cell parked sequence: **42.127 -> 1.358 us per removal** (31x), the
> parked cells and a full removal of the live sequence unchanged; `scripts/gate-park-idle-slots.sh` green.
> (b) `server-context.cpp` only fills `slot.spec_prompt` when `prompt.tokens.has_mtmd` -- without media the
> slot's own vector IS the prompt the drafters read inside `common_speculative_draft()` -- removing a
> **28.8 us/step** copy at 110K tokens.  (c) `ggml-cuda-slab.cu`, `ggml-cuda-moe-cache-hooks.cu` and this
> block's `getenv` lines are latched (the `GGML_CUDA_SLAB_STATS` 7 681-15 552 per-op reads are gone;
> `ggml_cuda_slab_stats_on()` is the shared accessor).  Same-seed text identical on all four parking-gate
> requests and on the reporter's server shape.  Records: `INVARIANTS.md` r217 facts 2 and 4,
> `benchmarks/2026-11-05-issue-273.md`.

> **r214 amendment: the slab's narrow-2 region is MAPPED for its whole extent.**
> `ggml_cuda_slab_narrow2_commit_locked()` pinned the MTP draft's compute region at the top of the slab's reserved VA
> and mapped it **only when the whole region lay above `mapped`**.  In the shortfall case -- the region **straddles**
> `mapped`, which is what the r106 arena-yield branch exists for -- it mapped *nothing*, so the top of the draft's
> compute buffer was reserved-but-absent VA: on a no-XNACK part the first device access past `mapped` is an
> unrecoverable `HSA_STATUS_ERROR_MEMORY_FAULT`.  Measured on a 2-device `-sm layer` Flash-Next at `-c 204800`
> (`-ncmoe 48`, q8_0 KV): 1.37 / 1.31 GiB of the draft's 1.50 GiB compute region unmapped, the fault in
> `rms_norm_f32` over the MTP head's `mtp_h_state`, while `-c 131072` (2 devices) and `-c 204800` (1 device) fit the
> spare and stayed clean.  The commit now maps whichever part of `[off, off+nf2)` has no physical behind it and
> advances `s.mapped`, so `ggml_cuda_slab_arena_top()`'s `min(mapped, ceiling)` clamp is load-bearing.  Every arm is
> byte-identical in text to r213 (`3ba9d426b43bc6b7` Flash-Next, `25678ef8d1fc4cfa` the 35B-A3B + DFlash2 narrow-2
> arm, `f9f1f58c296e`/`b065f37593b3`/`4ac3a523160f` the coherence arms); prefill-logit KLD 0.000000 / 100.000 %.
> Records: [`../WORKLOG.md`](../WORKLOG.md) 261104.007, `INVARIANTS.md` r214.
>
> **r212 amendment: the host pool's fill mode mirrors `--load-mode`.**  `MOE_HOST_POOL_DIO=auto` no longer guesses from
> `MemAvailable` ("the set does not fit beside the pool -> O_DIRECT"), which on a tight box always said O_DIRECT and silently
> overrode the operator's `--load-mode mmap`.  A mapped load is a request for the page cache, so the pool fills through it; an
> O_DIRECT fill is an explicit `MOE_HOST_POOL_DIO=1`.  A pool only exists on a mapped load (its master *is* the model mapping), so
> in practice auto = buffered.  Measured on a 113 GB Flash-Next, 2 devices, `-ncmoe 48 -lme 60`: the population drops from **95.20
> GiB of storage reads at 2 785 MiB/s (35.8 s)** to **95.20 GiB of mapping memcpys at 4 062 MiB/s (24.8 s)** with zero drive traffic.
> New iface entry `moe_cache_set_host_fill_mode`, APPENDED at the end of `ggml_backend_device_i` (it is initialized positionally);
> the loader reports its resolved `use_direct_io` once, from its constructor.  Value-identical: the buffered-vs-O_DIRECT
> prefill-logit A/B measures KLD -0.000000 / 100.000 %.  Records: `WORKLOG.md` 261104.003, `INVARIANTS.md` r212.

**What it is.**  The tail of the campaigns: CPU wins, the backend-iface fills that every non-CUDA backend
needs for the new hooks, DFlash, the FA instance build grouping, and the small allocator/model fixes that
had no other home.

**Contents.**

* **CPU wins**: an AVX2 block prefilter in the sampler (block max + NaN detection over 64 logits, with the
  scalar fallback preserved), and an AVX2 Q2_0 dot product that reproduces the generic two-rounding
  arithmetic bit-for-bit.
* **DFlash**: the F1 device-resident layer-input path made default-on with a warn-once fallback
  (`GGML_LF_DFLASH_DEV=0` disables), `GGML_LF_FAST_TOPK`, and — since **r106** — DFlash2 under
  `-sm tensor` (see the fold below; the r54 clean rejection is gone).
* **The DFlash draft's compute reserve measures both of its graphs at their real widths (r162, issue #239).**
  A DFlash/DFlash2 draft context alternates a KV-injection graph (target-feature rows, up to `n_ubatch`, on
  the main scheduler) and a noise-block decoder graph (at most `n_seq × block` tokens, on the alt scheduler).
  `sched_reserve()` measured both shapes from the generic token ubatch at `n_ubatch`, and the decoder applies
  the lm_head to every row, so `result_output` reserved `[n_vocab, n_ubatch]` f32 — **1940 MiB** of the draft
  context's **2444.41 MiB** device compute buffer at `-ub 2048` on a 248k vocabulary, for a width no graph
  ever uses (runtime decoder rows are `n_seq × (n_max + 1)`).  `graph_reserve()` takes `reserve_injection`,
  which builds the measure ubatch with a non-null embd marker (`llama_batch_allocr::ubatch_reserve(...,
  with_embd)`), and a DFlash draft now reserves the injection at `n_ubatch` plus the decoder at
  `n_seq_max × max(block, n_outputs_max_per_seq)` (the block covers lilicorr and DSpark).  The buffer keeps
  the larger of the two, so neither shape can grow it mid-run (issue #42), and the final layout stays at the
  injection graph, which is the one the main scheduler runs.  Measured (27B UD-Q5_K_M + DFlash2-Q4_K_M, 1×
  gfx1201, `-ub 2048`, q8_0 KV): draft device compute **2444.41 → 264.15 MiB (−2.18 GiB)**, free VRAM
  8622 → 10802 MiB; host compute 40.18 → 200.14 MiB (the injection's F32 feature input, which the host
  injection path needs and the F1 gather's output matches).  Greedy text byte-identical in every arm (1 GPU,
  2 GPU `-sm tensor`, F1 on/off, `LLAMA_DRAFT_SCHED2=0`, and a deterministic 1-slot server with
  `ngram-mod,draft-dflash`); draft counts and acceptance identical.  No knob: nothing is disabled, the
  reserve is simply smaller (`WORKLOG.md` 261028.004).
* **The backend-sampling compute buffers are reserved once per sequence, and again only for a WIDER
  chain (r169, contributor PR #245).**  `llama_context::set_sampler()` set `sched_need_reserve = true` on
  every set of an offloadable chain **and** on every removal; `sched_reserve()` re-creates the scheduler, so
  `llama-server --backend-sampling` — which sets a chain per request and clears it on release, and for a
  grammar request — freed and re-allocated every compute buffer, at the same size, before each first token.
  `sampling.reserved` now records what the buffers were reserved with and `set_sampler()` reserves only when
  the record does not cover the chain it is given; a chain set again, a narrower one, or a removal needs
  none (the graph is rebuilt anyway — `samplers_equal()` is part of the reuse check — the output buffers are
  sized per decode, and the allocator grows a buffer that needs more).  **The key is the chain's node count
  (`llama_sampler_backend_n_nodes()`), not the sequence id**, because `graph_max_nodes()` is recomputed only
  by a reserve and it sizes the graph's node capacity (`llm_graph_result::max_nodes` ->
  `ggml_new_graph_custom`, which `ggml_build_forward` asserts on), the scheduler's arrays and the
  compute-metadata arena — the allocator's growth covers buffer *sizes*, not node capacity.  The seq-id-only
  key the contributor shipped missed exactly one case, and it is the one
  `tests/test-backend-sampler.cpp`'s `test_set_sampler` performs: a 1-node `dist` chain replaced by
  `top_k + dist` on the same sequence (1 reserve with the seq-id key, 2 with the node count; PASS either
  way — the margin on a 27B is `max_nodes` 81920 against a 4076-node decode graph).  `LLAMA_BACKEND_SAMPLER_RESERVE_ONCE=0`
  restores the per-set reserve.  Numerics-neutral — no kernel, dispatch row or arithmetic touched.  Measured
  (27B UD-Q4_K_XL, 1x gfx1201, `--backend-sampling -c 65536 -ub 2048 --parallel 2 --kv-unified`, a
  2396-token prompt, seven requests, two interleaved starts): median prompt time **1963.9 -> 1895.2 ms
  (-3.5 %)** per sampled request, `sched_reserve` lines over three requests **4 -> 2**, request text
  byte-identical; dense 4B/27B coherence identical to r168 and prefill-logit KLD **0.000000 / 100.000 %**
  (`WORKLOG.md` 261028.011; upstream copy: `../upstream/backend-sampler-reserve-once/`).
* **The DFlash target writes its layer inputs into F1's store from inside the graph (r163, issue #240).**
  F1 kept the extracted layer inputs on the device, but `extract_layer_inputs()` copied them *after* the
  graph ran, so `llm_graph_result::set_outputs()` had to mark them as graph **outputs** — and an output
  cannot be reused by the allocator, so by the last layer every extracted input was still live: 5 ×
  `n_embd × n_ubatch` f32 = **188.8 MiB** of compute buffer at `-ub 2048` on a 27B (757.26 MiB with the
  extraction vs 568.42 with none, measured on the reporter's config).  `set_outputs()` now emits a
  `GGML_OP_SET_ROWS` into the persistent store instead, and `llm_graph_result::set_layer_inp()` (which every
  arch graph now calls instead of assigning `t_layer_inp[il]` directly) closes each layer's extraction
  window: at the next layer's recording it expands that layer's nodes and *then* emits the copy, so the copy
  lands right after the input's last consumer.  The row offsets are a graph **input** (`I32 [n_tokens]`,
  filled per ubatch by `llama_context::process_ubatch`), not a baked view offset: a graph with the same
  shape stays reusable across ubatches at different offsets, and a captured HIP graph replays with the right
  rows.  The store is created in `lf_layer_inp_dev_arm()` at the top of `decode()`, before the first graph
  that writes it, and the context drops and re-reserves its compute buffers once so the smaller layout is
  what the run settles on; the wait for the draft's read of the previous batch moves before that graph.
  The store lives on the device that owns the first extracted layer; a layer input on another device (a
  `-sm layer` split boundary — layers 48/62 on the second card in the 2-GPU arm) is copied across by the
  scheduler, exactly as the post-graph device copy used to be.  **Two side effects are part of the fix:**
  the pinned host rows the *host* path reads are now reserved lazily (`layer_inp_host_reserve()`) — F1 never
  read them, and they cost **200 MiB** of pinned host memory per context at `-b 2048` (the one exception is an
  embedding ubatch: it has no token ids, so the draft reads the host rows and `extract_layer_inputs()` fills
  them on demand) — and the per-ubatch
  `sync_readback()` of the layer inputs is skipped when there is no host readback to order (it was a device
  synchronize per ubatch of a multi-ubatch prefill).  `GGML_LF_DFLASH_DEV_GRAPH=0` restores the pre-r163
  shape (outputs + post-graph copy) for A/B and bisect.  Measured (27B UD-Q4_K_XL + DFlash2-Q4_K_M,
  `-ub 2048 -b 2048`, q8_0 KV): target compute buffer **757.26 → 625.27 MiB**, target output buffer
  **200.95 → 0.95 MiB**, host layer-input rows released after the arm; greedy text and draft acceptance
  **byte-identical** in every arm (1 GPU, 2 GPU `-sm layer`, 2 GPU `-sm tensor` host fallback, 35B-A3B
  `-ncmoe 20`, and a 2-slot `--kv-unified` server: identical text, acceptance 0.74163 in all three arms);
  decode flat within noise (106.2-106.7 t/s at `-n 1000`, four alternating arms) (`WORKLOG.md` 261028.005).
* **DFlash2 under `-sm tensor`** (r106, TODO #78): the selector builds `ggml_top_k` over the draft logits and
  `ggml_get_rows` to read the candidate values back, and those logits are the *target's* `output.weight`
  projection — placed vocabulary-split by the target's policy — so both ops reduce along the split axis and
  the Meta backend aborted on the straddling row.  `build_dflash2_selector()` now pushes the logits through a
  new **`GGML_OP_ALL_GATHER`** collective (this block; §"Facts that span blocks") when the draft is a
  ≥ 2-device tensor split, so every device holds the whole row and the selector runs locally.
  `extract_layer_inputs()` also takes its host path when the layer inputs live in a Meta buffer: the F1
  "device" copy is one pointer range, which a Meta buffer cannot express, and it aborted in
  `ggml_backend_meta_buffer_set_tensor` — **pre-existing for any** DFlash drafter that requests
  `dflash.target_layers` (the r54 guard only covered selector models).  Measured (27B-Q4_K_XL + its DFlash2,
  `-c 16384`, q8_0 KV, 5258-token prompt): **84.5 t/s vs 63.4 for the `-sm layer` config that used to be the
  only one supported** (+33 %), draft acceptance identical to `-sm layer` (0.67742, same per-position
  rates), 3-GPU 61.1 vs 60.8, `--parallel 2` green, `test-backend-ops` 3/3, prefill-logit KLD 0.000000 /
  same-top-p 100.000 %.  `Qwen3.6-35B-A3B-Q8_0` + its DFlash2: 191.0 t/s vs 161.9 (+18 %).  No new knob:
  `-sm layer` remains the escape hatch.  A true Meta *distributed* top-k (gathering `top_k × n_devices`
  candidates instead of the whole row) was the alternative; it needs the same cross-device primitive plus a
  merge, and the gather measured inside the draft step's noise, so the simpler shape shipped.
* **The slab's narrow-2 region yields from the arena** (r106): the draft context's compute buffer is pinned
  into the slab's narrow-2 region at the top of the reserved VA, and that placement only accepted the VA the
  creation reservation left unmapped — i.e. the *draft-context estimate* (`reserve − headroom`), whose
  `compute` term is a measured 512 MiB tier.  A **tensor-split** drafter runs the whole draft graph on every
  device, so its per-device compute buffer measured 1280 MiB, larger than the whole spare: the pin refused,
  the draft fell back to `cudaMalloc`, and with the slab holding the VRAM that failed
  (`graph_reserve: failed to allocate compute buffers`) for `-sm tensor` + a draft + `-ncmoe`.  The pin is
  now two-phase: decide under the lock, evict the arena tables in `[off, mapped)` via
  `moe_cache_evict_slab_range` outside it (the callback re-enters through `arena_free`), then commit —
  mapping the region when it lies above `mapped`, else capping the arena at `off` (`arena_limit` already
  returns `narrow2_off`) and dropping every free run at or above the new top so no later table can land on
  the draft's view.  The unmapped-spare placement is still preferred and unchanged when it fits (1 GPU
  `-ncmoe 40`: "… 0.75 GiB, unmapped VA spare"); plain `-ncmoe` never arms narrow-2.  Measured (35B-A3B-Q8_0
  + DFlash2, `-ncmoe 20 -c 8192`, no override env): 2 GPU **abort → 138.6 t/s**, 3 GPU **abort → 87.4 t/s**,
  acceptance 0.849 both.  `GGML_CUDA_SLAB_RESERVE_MIB` is no longer needed as a workaround.
* **Allocator/model**: `ggml_gallocr_drop_buffers` (free compute buffers and invalidate the layout so the
  next reserve sizes from the current graph), the per-device `max_host_expert_slice_bytes` split rule, the
  application of the embeddings-layer-input preset, and the `-sm tensor` H2D-ring auto-size + narrow-band
  pin.
* **DFlash prompt-ingest tail** (r130, PR #212): a DFlash/DFlash2 draft whose **every** layer attends over a
  sliding window has a window-sized KV cache, but `process()` used to feed it every prompt row — a 110K
  prompt encoded 110K rows, all but the last ~2048 dropped again without ever being read, and each prompt
  batch waited on the target before the draft could start, which also broke the queue of the target's
  prompt ubatches.  `common_speculative_impl_draft_dflash` now takes the prompt end from the block-02 hook
  (`set_prompt_end()`) and skips the prompt rows before the last `n_swa` positions; a batch left with no rows
  returns before the target wait, and the first generation row resets the threshold as the MTP impl does.
  Exact because DFlash sets `LLAMA_SWA_TYPE_STANDARD` unconditionally (`src/models/dflash.cpp`), so the mask
  is `p1 - p0 >= n_swa`: a draft query can attend exactly a prompt's last `n_swa` positions
  (`GREEDY-PURITY.md` §53).  A draft with any full-attention layer keeps feeding every row.  The
  window test needs the per-layer pattern, which `llama_model_meta_val_str()` cannot return (it skips array
  values), so a new staging query `llama_model_n_layer_swa()` sits in `src/llama-ext.h` next to
  `llama_model_dflash_selector_top_k()` and upstream's `llama_model_n_swa()`.  `LLAMA_DFLASH_INGEST_TAIL=0`
  feeds every row.
  **The tail's rows must also be dropped when a partially reused prompt puts the window past them (r218, issue
  #272):** skipping the ingest is only exact while the draft's cache stays contiguous with the rows it does
  ingest, and a slot that keeps a prefix from an earlier task breaks that as soon as a turn adds more than
  `n_swa` new tokens - a strict (non-M-RoPE) draft then has its injection batch REJECTED
  (`inconsistent sequence positions`, `decode() failed: failed to process speculative batch`, HTTP 500), and an
  M-RoPE draft silently ingests a batch that attends to nothing.  `process()` now realigns the cache with
  `spec_draft_align_cache()` (block 02) before injecting: the stale prefix is dropped, which is exact (every row
  of it is older than the window of the prompt's end, and the batch that would have carried it is gone) and
  leaves the draft starting a fresh window at the first ingested row, exactly as for a first-ever prompt.  Gate:
  `scripts/gate-spec-reuse-window.sh` - a derived non-M-RoPE copy of a DFlash2 draft (the reporter's shape),
  three turns adding a full 5246-token prose chunk each, tail on vs `LLAMA_DFLASH_INGEST_TAIL=0`, on one model
  pair; pre-fix the strict arm answers turn 2 with a HTTP 500 (`X = 5241, Y = 8445`), and post-fix all 12
  requests answer with the tail and no-tail text byte-identical.
  Measured (3x gfx1201, `-sm tensor -mg 0`, 27B UD-Q4_K_XL + DFlash2-Q4_K_M, `-fa on -ctk/ctv q8_0`):
  prefill **+8-12 %** at 6K/26K/88K tokens (88.6K: 1271 -> 1395 t/s; no-draft 1415 t/s, so the draft's share
  of the prefill goes 10.2 % -> 1.4 %), 35B-A3B Q4_K_M + its DFlash2 **+14.6 %** at 6.1K (our A3B draft
  reports all six layers windowed, so this arm engages the tail rather than staying off); greedy text and
  draft counts byte-identical in every arm (1 and 2 slots, `-ub 512` and `-ub 2048`).  Record:
  `archive/work/dflash-ingest-tail/`.
* **Server prompt-cache residency** (r130, PR #210): with a unified KV cache and `--cache-idle-slots`
  (default on) every new task saved each idle slot to the prompt cache and cleared it, so two conversations
  taking turns on two slots copied their whole state out and back on every switch (27B: 1.6-2.7 s per switch
  at 110K in the report).  An idle slot now keeps its context; the KV-full path saves the least recently
  used idle slot to the cache before clearing it; a restore that fails for want of cells held by another
  idle slot moves idle slots out one at a time with the entry being restored held out of that eviction and
  retries; and the slot-selection save holds the entry the task is about to restore out of its own
  eviction (`server_prompt_cache::find()` split out of `load()`).  Buffers of restored/evicted states are
  reused for the next save, counted against `--cache-ram` with the cached states.
  `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` / `LLAMA_SERVER_CACHE_REUSE=0` restore the eager behaviour.  The
  delivery's fix on top of the PR: `find()` skips an entry whose main state buffer was already consumed
  (a restore that failed on the draft buffer leaves one behind, and re-matching it compared against zero
  bytes and reported success).  `tools/server/tests/unit/test_kv_keep_only_active.py` is updated for the new
  default and covers both arms.  Measured on gfx1201: conversation switch at 11.5K tokens 0.5 -> 0.2-0.3 s
  (same text), the overflow case 8.3 s recompute -> 0.5 s restore with 4 tokens evaluated, 4-slot concurrent
  storm 13/13 requests green.  Record: `archive/work/server-prompt-cache-lazy/`.
* **Huge-page prompt-cache buffers** (r173, PR #248, overdoingism): a **new** prompt-cache state buffer of
  64 MiB or more (`server_prompt_cache::take()`) is reserved, its 2 MiB-aligned part is advised
  `MADV_HUGEPAGE`, and only then is it zero-filled, so a multi-GiB slot state is faulted in 2 MiB pages
  instead of 4 KiB ones (Linux; THP in `madvise` mode — `LLAMA_SERVER_CACHE_HUGEPAGE=0` restores 4 KiB
  pages; a recycled spare keeps its pages).  The zero fill, not the copy, is the dominant term of a
  whole-slot save on a 4 KiB-paged host: **3.47 GiB alloc+zero 470 -> 125 ms**, and the server's
  `prompt cache update took` for a 100K-token slot saving into a new 3.4 GiB buffer **828.1 -> 497.0 ms**
  (-40 %; the recycled-buffer case is unchanged at 292.6 -> 291.0, as is everything copied, so no text or
  logit can move).  **The submission's other half is deliberately NOT taken**: taking the batched state
  path for states above `LLAMA_STATE_ASYNC_MAX_MIB` by *cutting* a piece at a byte offset.  A state piece
  is a window handed to the backend that owns the tensor — under `-sm tensor` that is the meta backend,
  which only serves a window aligned to the tensor's own layout, so a cut trips
  `GGML_ASSERT(size % chunk_size_full == 0)` (2-device tensor split, 27B, `-c 266240`, 250K-token slot:
  the save aborts, where `LLAMA_STATE_ASYNC_BANDS=0` completes with identical logits).  Any future banding
  must therefore be a run of **whole** pieces, an oversized piece taking a band of its own — the rule
  `llama_state_copy_staged()` already follows for the device-state class.  It is also not a win here: a
  whole-slot state's pieces are 1-27 MiB, so the per-tensor path already runs at 8-14 GiB/s and the batched
  path's extra host copy costs more than the syncs it removes (324 MiB state, 1 GPU: save 32.0 -> 51.9 ms,
  restore 23.1 -> 53.9 ms; 3.47 GiB: save 304 -> 278, restore 242 -> 281).  Record:
  `benchmarks/2026-10-29-prompt-cache-hugepage.md`, `archive/work/prompt-cache-save-bands/`.
* **Idle-slot parking** (r136, PR #225, `archive/work/server-park-idle-slots/`): #210's resident idle slots
  change a request's `n_kv` and cell indices, so a sequence that runs alone no longer sees the empty-cache
  attention layout and near-ties can flip (the reporter's 4/16 repeat test; our same-slot schedule measured
  13/32 against lazy-off).  `llama_memory_seq_park(mem, seq_id, park)` (`llama-ext.h`, implemented in
  `llama-kv-cache.cpp` + `llama-memory.cpp`) moves a sequence's cells to the top of a unified cache, below
  the previously parked sequences, with a device copy of the K/V rows and the cell state
  (`llama_kv_cells::mv`, including `seq_pos` and the extension); `seq_unpark` moves one back to the lowest
  free run and `park_compact` keeps the stack together.  New cells are only taken below the lowest parked
  cell (`find_slot`) and `get_n_kv()` stops there, so a running sequence attends over the same range at the
  same indices as with the idle slots saved and cleared.  `llama_memory_can_park()` requires one stream,
  cells no other cache shares, V not transposed and no sliding window.  The server parks idle slots at task
  start and unparks the processing ones (`LLAMA_SERVER_PARK_IDLE_SLOTS`, on unless `0`, with the lazy idle
  slots and `--kv-unified`); a cache that cannot park keeps #210 and logs once.  A slot whose cells are not
  one run of its own (shared through `seq_cp`, e.g. a cached prefix restore) is saved and cleared, which
  is the lazy-off output.  **The delivery fix on top of the PR:** with `--no-cache-idle-slots` it is left
  resident instead (there is no saved copy to restore, and that flag means the lazy slots stay resident).
  Measured (gfx1201, 27B UD-Q4_K_XL + DFlash2, 4 slots, `--kv-unified`, q8_0 KV, 16 prompts two rounds,
  `cache_prompt: false`): parking **16/16** and **32/32** equal to `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`, while
  `LLAMA_SERVER_PARK_IDLE_SLOTS=0` is 13/16 and 13/32.  `-fa off`, a sliding-window model and a cache that
  cannot park are text-identical with the switch on/off; hybrid attention/recurrent is 16/16 equal to
  lazy-off; 2 GPU `-sm tensor` works.  `GREEDY-PURITY.md` §57.  The reporter did not cover MoE with the
  expert cache, vision, more than 4 slots or Windows, and we did not re-measure the switch-time table.
* **A memory whose second cache mirrors the attention cells parks both or neither (r146, issue #227).**
  `llama_memory_hybrid_idx` (qwen4exp and glm5-next) carries the QSA indexer cache beside the attention
  cache **cell for cell** — cell `j` is one token in both — and r136's `llama_memory_attn_caches()` matched
  the memory through its `llama_memory_hybrid` base and parked **only** the attention half.  qwen4exp then
  aborted on its own guard (`qwen4exp.cpp` `the indexer cache must track the attention cache cell for cell`)
  the moment a second slot launched on a server with an idle slot parked (2x R9700, issue #227: a 9.9k
  prompt on slot 0, a short prompt on another slot; `-sm tensor` **and** `-sm layer`).  The memory now
  answers `llama_memory_can_park()` / `llama_memory_seq_park()` itself: `can_park()` requires both caches,
  and `seq_park()` moves the attention cache then the indexer (undoing the first if the second refuses),
  refusing up front — with a warning, which the server turns into its save-and-clear fallback — when the two
  no longer mirror each other (`attn_idx_in_step()`, compared over the stream's whole `seq_pos` sets, since
  `park_compact()` also moves the sequences parked beside this one).  A successful move stales the pool
  layout for every sequence and drops the derived indexer pool's watermarks: the layout caches each
  sequence's `(position, cell)` pairs, and the derived rows are indexed by the block ordinal of the cells
  in the **decode window**, which a park changes (`pool_invalidate_all()`), so both are recomputed on the
  next ubatch.  `llama_memory_attn_caches()` is unchanged for every other memory type; `llama_kv_cache_dsa`
  / `_iswa` derive from `llama_memory_i` directly, so their indexer was never half-parked (they report
  "parking not supported" and keep #210).  Measured (Flash-Next GSQ-RCO IQ3_XXS + MTP, 4 slots, 10.6k-token
  prompt, `-sm tensor` 3 GPU all-VRAM, `-sm layer` 3 GPU, and 2 GPU with `-ncmoe 44` in both split modes):
  park **==** `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`
  byte for byte on all four requests (A parked, B on the freed low cells, C unparking A, D with two slots
  parked), `LLAMA_SERVER_PARK_IDLE_SLOTS=0` differs on two of them, both caches log every move, no GPU
  fault.  Gate: `scripts/gate-park-idle-slots.sh` (it fails on the r145 build with exactly the reported
  assert).  `GREEDY-PURITY.md` §57.
* **The batched state save and restore** (r96, PR #189; restore r119, PR #209): `llama_io_write_host`
  queues its deferred reads as `ggml_backend_tensor_get_async` into one pinned staging buffer kept by the
  context, synchronizes each backend once, then copies to the caller's buffer.  A recurrent model's
  checkpoint is about 75 small ranges, and under `-sm tensor` each was one synchronous read per device, so
  a checkpoint paid about 150 device syncs before every prompt chunk of a follow-up turn (Flash-Next + MTP:
  -55 ms/-59 ms TTFT, isolated by the same-binary `LLAMA_STATE_ASYNC=0` arm).  The restore side writes the
  same pieces the same way: `llama_io_read_host` packs them into the same `state_stage`, queues each with
  `ggml_backend_tensor_set_async` on the backend that owns it, and synchronizes each backend once.
  `LLAMA_STATE_ASYNC_READ=0` keeps the per-tensor synchronous restore.  States above
  `LLAMA_STATE_ASYNC_MAX_MIB` and any state with a tensor no context backend owns keep the synchronous
  path in both directions.  The stage is one buffer per context, so saves and restores on one context must
  not run concurrently.  Measured on r118 (2 GPU `-sm tensor`, 1571-token checkpoint, q8_0 KV): 35B-A3B
  restore 49 -> 40 ms, Flash-Next GSQ 89 -> 75 ms, the DFlash2 draft arm 68 -> 58 ms; one GPU is about
  2 ms slower because the extra staging copy has no multi-device sync to save, which is accepted for the
  multi-device win.  The file also gains `#include <algorithm>` (the PR #189 patch as submitted does not
  compile without it).
* **The batched copy is pipelined** (r151, PR #233): the pieces are cut into up to
  `LLAMA_STATE_PIPE_GROUPS` (default **8**; `1` = the unpipelined copy) size-balanced groups, so the host
  memcpy into/out of the pinned stage overlaps the device copies of another group.  A restore packs a group
  and queues its async sets before packing the next (the groups own disjoint stage ranges, so a queued copy
  is never overwritten); a save records one event per `(backend, group)` after that group's copies are queued
  and copies the group out as soon as its event completes.  A backend with no events (the CPU backend) falls
  back to the single `llama_state_sync_all` path, which is also what `LLAMA_STATE_PIPE_GROUPS=1` selects.
  Measured (gfx1201, prompt-cache restore of a 5246-token hybrid state, medians of 5 cycles): Flash-Next GSQ
  + MTP head 2 GPU `-sm tensor -ncmoe 40` **134.5 -> 129.7 ms** (-3.6 %), 35B-A3B Q4_K_M 1 GPU
  **69.4 -> 67.4 ms** (-2.9 %, below the pre-r119 synchronous path's 71.3 ms), 35B-A3B 2 GPU tensor within
  noise; the server's own save+load trace moves 81.7 -> 78.4 ms and 44.9 -> 39.9 ms.  The restored greedy text
  is byte-identical to `LLAMA_STATE_ASYNC=0` in every cycle of every arm (the save side isolated by
  `LLAMA_STATE_ASYNC_READ=0`), with no GPU faults.  Gate: `scripts/gate-state-pipe.sh` (it is the only gate
  that reaches this code: the file-based slot save/restore goes through `llama_io_write_file` /
  `read_file`, which call `ggml_backend_tensor_get`/`set` per tensor and never touch the stage).
* **The host side of that copy is streamed on two threads** (r152, PR #235): each piece of a group is copied
  with non-temporal (SSE2) stores over 64-byte-aligned runs, split over `LLAMA_STATE_COPY_THREADS` (default
  **2**; `1` = one thread but still streamed, `0` = the pre-r152 plain `memcpy` per piece, the A/B arm)
  threads.  A plain store reads the destination line into the cache first (read-for-ownership) and a
  streaming store writes straight to memory: that is the whole win on a box whose L3 cannot hold the
  checkpoint (the submitter's 3900X + DDR4: 150 MiB 19.4 -> 11.8 ms on one thread) and about 0 on this box
  (9950X3D, 192 MiB of L3, a 164 MiB checkpoint mostly resident), where the **threads** are the win (164 MiB
  in a 35B-A3B checkpoint's real piece sizes: 6.4 ms plain, 6.4 streamed on one thread, 3.0 on two, 3.3-3.7 on
  three or four - two is the knee on a dual-channel DDR5 controller).  A worker never crosses a group
  boundary: a group's stage range may only be read after its event, so a worker in the next group would read
  a range whose device copy is still in flight (a torn state, not a crash).  States below 16 MiB keep the
  plain copy, pieces below 64 KiB are copied with `memcpy`, and the streaming store is x86-only
  (`__SSE2__` / `_M_X64`), so other targets gain only the threads; every streamed run ends with
  `_mm_sfence()` before its bytes reach the caller (save) or the device (restore).  Measured (gfx1201,
  prompt-cache restore of a 5246-token hybrid state): the copy itself -12 / -21 / -29 % for a 62.8 / 73.2 /
  165.3 MiB save and -26 / -30 % for the restore's pack (35B-A3B 2 GPU tensor `-ncmoe 10`), -21 / -22 / -29 %
  for a 112.6 / 126.6 / 250.9 MiB save (Flash-Next GSQ + MTP 2 GPU `-ncmoe 40`); the server's save+load trace
  39.9 -> 33.4 ms (35B-A3B 1 GPU), 44.2 -> 40.5 ms (2 GPU tensor) and 79.3 -> 74.9 ms (Flash-Next + MTP).
  `LLAMA_STATE_TIMING=1` logs the enqueue/copy split of each batched read and write.  Gate:
  `scripts/gate-state-pipe.sh`, whose `copy0` arm is the plain-`memcpy` A/B, gated byte-for-byte like every
  other arm.
* **The device-state path (`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`) is staged, not bounced per piece** (r157, TODO #110).
  `llama_io_write_device`/`read_device` keep a sequence's memory in a *device* buffer (that is the whole point
  of the flag), and under a tensor split both ends are meta buffers, so every piece went through
  `ggml_backend_tensor_copy`'s fallback: a pageable `malloc` + a synchronous device read + a synchronous
  device write **per piece**.  The state path copies one piece per (layer, K/V) per cell range, so a
  1024-token 2-slot state is ~128 pieces (49 152 when two slots interleave their cells), and it measured
  **2.23 s to save / 2.28 s to restore** 213.6 MiB — **58x slower than the host path** it is supposed to
  beat (21.9/38.9 ms) and 700x slower than the same state on one GPU, where the copy is a real
  device-to-device `cpy_tensor` (3.0/3.2 ms).  `llama_state_copy_staged()` now copies a list of pieces in
  **bands** through the context's pinned `state_stage`: every piece of a band is queued with
  `ggml_backend_tensor_get_async` on the backend that owns it, each backend is synchronized once, then every
  piece is queued with `set_async` and synchronized once.  Two details decide the result: a piece is never
  cut (the meta backend serves a window only when it is aligned to the tensor's own layout, so a band is a
  run of whole pieces and a piece larger than the band gets a band of its own), and a piece is handed to the
  backend as **the tensor plus an offset**, never as a fresh `ggml_view_1d` of it (the view's inherited split
  state makes the reader splice it row by row: 4.5 ms for 2 MiB against 0.15 ms).  The arm is chosen per
  buffer type (the destination is a meta device), so a single device keeps its device-to-device copy
  untouched (3.02/3.22 ms before and after).  Measured: **14.7 ms / 17.2 ms** (150x), i.e. faster than the
  host path; the fragmented 49 152-piece shape 2.24 s -> 14.7 ms.  `LLAMA_STATE_STAGE_MIB` (default 64,
  `0` = the old path) sets the band; 8 MiB already reaches the plateau.  Oracle: `tests/test-llama-context`'s
  `cp_d`/`cp_d_s` (the restored KV blob is compared byte for byte) — nothing in the server or `llama-cli`
  uses this io class, which is why it went unnoticed.  `benchmarks/2026-10-27-state-device-path.md`.
* **The target's speculative checkpoint goes to device memory when it fits** (r184, PR #254,
  `archive/work/spec-ckpt-device/`).  On a recurrent or hybrid target the server saves a checkpoint of the
  target's recurrent state before a verify whose draft is longer than `n_rs_seq` and restores it when only part
  of the draft is accepted; with an n-gram drafter that is the long-draft steps of any copy-heavy request.  It
  went to host memory: 149.6 MiB of Qwen3.8-27B state, ~4 ms per save on this box and ~16 ms on a slower CPU.
  The server now passes `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` for it, so a save and a restore are device copies
  into one staging buffer per slot that the context keeps and reuses.  Decided once per slot at its first
  checkpoint and logged: device memory when the target is on exactly one GPU, its state is offloaded and that
  GPU has the state's size plus 512 MiB free (the staging buffer is allocated lazily, after an expert-cache
  arena or other caches took the GPU, and a failed allocation inside the io class aborts).  Otherwise host
  memory as before.  `LLAMA_SERVER_SPEC_CKPT_DEVICE=0` keeps host memory always.  Context checkpoints (the
  prompt cache) are not touched.  **The device arm is not a byte image**: `data_tgt` carries only the metadata
  (the io header, the cell ranges) while the payload lives in the context's per-seq staging buffer
  (`llama_context::mem_storage`), which the next save overwrites and the restore copies back into the live
  cells, so a save must be restored before the next save for the same sequence (the speculative path does
  exactly that: a draft round saves once and only ever restores that save, and the replay rounds save nothing)
  and one of these must never be written to anything persistent.  A host-flagged save (a context checkpoint)
  does not touch the stage, so those interleave freely.  Single GPU only, by design: under `-sm tensor` or a
  layer split the state is spread over the devices (and the r157 device io class would bounce through the host
  anyway), so the multi-GPU arms stay on the host path.  Measured 1 GPU (gfx1201, one binary, kill switch vs
  default, greedy, `ROCBLAS_USE_HIPBLASLT=0`): 27B UD-Q4_K_XL + DFlash2 + `ngram-mod` rewrite request **310.2
  -> 316.8 t/s (+2.1 %)**; 27B with `ngram-mod` alone **202.1 -> 205.4 (+1.6 %)**; 35B-A3B UD-Q4_K_M
  `-ncmoe 40` with the arena (850 MiB free) + DFlash2 **422.1 -> 440.4 (+4.3 %, 63 MiB state, +70 MiB VRAM)**;
  the zh-short-story and continuation requests **0 %** (they rarely take a checkpoint at all); 2 slots
  `--kv-unified` **+2.6 %**.  Text, drafted and accepted tokens identical in every arm; +150 MiB VRAM per used
  slot on the 27B (-150 MiB RSS).  **The submitter's +10.0 %/+15.8 % is host-CPU dependent, and so is this**:
  the win is exactly (host save cost) x (checkpoints taken), 47 saves in a 3039-token rewrite here at 4.5 ms
  each (the r96/r151/r152 state-copy folds already took the rest) against their 16 ms on a slower host.  Not in
  the `--fit` estimate (allocated lazily, at the first checkpoint).  Gate:
  `scripts/gate-spec-ckpt-device.sh` (text/draft identity across the arms, the kill switch, and the `-nkvo`
  fallback with its own reason - nothing else in the repo reaches this code);
  `benchmarks/2026-10-30-spec-ckpt-device.md`.
* **A restored context checkpoint is not saved again, and dropped checkpoints' buffers are recycled** (r155,
  PR #236, `archive/work/server-ckpt-reuse/`).  On a hybrid/recurrent (or SWA) model the memory cannot roll
  back, so the server saves a **context checkpoint** (`common_prompt_checkpoint`: the recurrent state plus the
  draft context's state and the drafter's stashed spec state) and a prompt that resumes from a cached prefix
  restores one.  It used to restore the checkpoint and then write the identical state straight back at the
  same position before its first batch (log: `superseding context checkpoint`), and again near the end of the
  prompt; each save allocated, zero-filled and faulted in a fresh host buffer of the recurrent state's size
  (150 MiB on 27B, 174 MiB with its DFlash2 draft).  A slot now remembers the `n_tokens` of the checkpoint its
  task restored; when a checkpoint lands at that same position nothing has been decoded since the restore, so
  that checkpoint already holds the current state - it is moved to the back of the list (exactly the ordering a
  re-save produces) and kept, with the spec state the restore re-applied, instead of being written again.  The
  buffers of erased, superseded and invalidated checkpoints are kept (4 in total, the largest kept) and an
  exact-size one is taken by the next checkpoint, so a save is only the copy.  `LLAMA_SERVER_CKPT_REUSE=0`
  restores the eager behaviour.  No arithmetic and no KV cell moves.  **The keep path only fires on a
  hybrid/recurrent target**: for a dense (or SWA) memory `seq_pos_min()` is the sequence's oldest cell, so a
  restore lands at `pos_max` = `n_tokens - 1` and the first checkpoint of the task is one token short of the
  restored one; a hybrid/recurrent memory reports `pos_min == pos_max` (its single recurrent cell carries the
  last position), which is what makes the re-save land at the same `n_tokens`.
  **The delivery's hardening on top of the PR:** `n_tokens_ckpt_restored` is also cleared in
  `server_slot::reset()` (so the invariant does not depend on every task entering the `SLOT_STATE_STARTED`
  branch), and the keep path refuses a checkpoint whose `data_tgt` is empty (`load_tgt` is a no-op for such a
  checkpoint, so it can never stand in for the live state).
  Measured (gfx1201, every arm's greedy text byte-identical between the switch's two settings): prompt time of a
  request that resumes from a checkpoint - 4B 1 GPU 45.1 -> 31.4 / 40.0 -> 28.7 / 46.4 -> 32.4 ms, the same at
  2 slots `--kv-unified` with idle-slot parking 43.4/42.2/44.6/42.2/43.9 -> 29.3/28.7/28.3/27.3/27.7 ms, 27B +
  DFlash2 2 GPU 197.1 -> 163.2 / 186.9 -> 161.9 / 144.1 -> 112.0 / 187.6 -> 126.7 ms and 3 GPU `-sm tensor`
  407.9 -> 373.6 / 291.2 -> 236.2 / 182.4 -> 142.0 / 251.7 -> 150.6 ms; the log reads `superseding` 3 times off
  against `keeping` 3 times on and creates 9 checkpoints off against 6 on; RSS +0.15 GiB (4B) / +0.45 GiB (27B)
  for the spares.  Text also unchanged across a slot save/restore endpoint round trip, which the PR lists as
  uncovered.  Gate: `scripts/gate-ckpt-reuse.sh` (it is the only gate that reaches the keep path: no other
  gate restores a context checkpoint, and the property it protects - a wrong `n_tokens_ckpt_restored` keeping a
  state the live context does not match - is a wrong answer on the *next* resume, not a crash).
* **Build time**: one FA template instantiation TU per `(ncols1, ncols2, head size)` instead of per
  `(ncols1, ncols2)` — same instantiations, grouped so the backend build is no longer one-file-bound.
* **The expected `--fit` probe failure logs as a warning** (r101, campaign 0012): `--fit` builds the extra
  (draft) model's context before the target context exists, so an arch that needs `ctx_other` fails there
  by design and fit falls back to the derived reserve.  `E llama_init_from_model: failed to initialize the
  context: … requires ctx_other to be set` is now a **W**; any other init failure stays an **E**.
* **The draft-context creation is gated on an enabled type that uses it** (r101, campaign 0011, the
  `common/speculative.cpp` half — see block 09 for the predicate in `common/`): with `--spec-type none`
  the extra model is neither loaded nor given a context, so the gemma4 drafter's warm-up decode no longer
  aborts on the missing state embedding.
* **`--host-experts pool` drops the full pinned master** (r111, TODO #55; this block, not 09/10 — see
  "A block owns its code" below).  The host expert tensors are placed in the read-only **model mapping**
  through a new per-device host buffer type `ROCm_HostMap` (`ggml_backend_dev_moe_host_master_buft` +
  `ggml_backend_host_buffer_from_ptr`), which still carries its device so the scheduler keeps the
  `MUL_MAT_ID` on the owning GPU, and **every** host read of a pooled table is staged through pinned
  memory: the bounded host pool (`pool_fill_slot_locked`: a `memcpy` out of the page cache, or a `pread`
  with `MOE_HOST_POOL_DIO=1`), a per-table pinned **bounce ring** when the pool declines (`expert_host_src`
  / `expert_src_done`, grown on demand, never reused until its event drains), and `moe_cache_upload_pageable`
  for the scheduler's used-expert copies (`set_tensor_async` / `stage_upload` / `stage_gather`).  A pageable
  `cudaMemcpyAsync` is never used for a pooled master: ROCm stages a pageable upload through a shared
  internal bounce and back-to-back copies tear (that is what made the first cut nondeterministic).  Staging
  serves the pre-sizing prefill too (`moe_cache_stage_table` no longer needs a built arena), it bypasses the
  pool (a whole-table staging reads every expert anyway, and the misses were evicting the decode band), and
  the cold-UVA serve / device gather stand down for a pageable master (issue #116).  Measured: `RssShmem`
  **18.6 → 6.8 GiB** on 35B-A3B `-ncmoe 40` (FRAC 25; the rest is reclaimable page cache), 40.8 → 11.2 GiB
  on the 100 GB Flash-Next; **a 12 GiB / 8 GiB cgroup OOM-kills the `pinned` arm and runs this one at
  ~93 t/s**; text identical to `pinned` on 1/2/3-GPU tensor, gemma4 (fused tables), Flash-Next and MTP;
  prefill-logit KLD -0.000000 / same-top-p 100.000 %; `MUL_MAT_ID` 947/947; decode parity everywhere, prefill
  -1..-6 % (one extra host memcpy per staged byte; a single-ubatch `pp512` on the 2-GPU tensor config -34 %).
  Three pre-existing pool bugs were fixed with it: the `pool_dio_read` shared static bounce raced by the
  background worker (now `thread_local`), `g_host_pools` destroyed under the detached worker at exit (now
  leaked, like the worker's mutex/cv/queue), and an AUTO budget that was both a 3x running sum and
  registration-order dependent (now the per-device max and a per-tensor share).  `MOE_HOST_POOL_VERIFY` and
  `LLAMA_HOST_MASTER_PINNED` are the new diagnostic / A/B knobs.  **r112** then tuned the defaults for the
  case the mode exists for (a set that does not fit in RAM): `MOE_HOST_POOL_FRAC` **25 → 80 %** and
  `MOE_HOST_POOL_DIO` **0 → 1** (O_DIRECT fills, no page-cache copy competing for RAM).  Measured on the
  field case (gfx1100, 30 GiB RAM, 35B-A3B Q8_0, `-ncmoe 40` = 31.9 GiB of host experts, 2000-token runs):
  **78.3 t/s decode / 0.3 GiB swap / 26.0 GiB pinned**, versus 48.6 t/s for the r111 defaults and 2-3 t/s
  (16 GiB of swap) for `pinned`.  80 % is the knee (90 % → 71.8 t/s *and* swapping).  The fill mode is no
  longer a fixed `DIO=1`: r114 decides it per box (see below).
* **The host pool is a LAYOUT, not a churning cache (r113, TODO #87).**  Two changes to POOL mode, both in
  `moe-expert-cache.cu`:
  1. **Content: the pool holds the arena's complement, ranked by the persisted heat prior**
     (`pool_rank_experts_locked` / `pool_rerank_from_heat_locked`).  It is laid out in bulk at sizing
     (after `apply_heat_seed_locked`/`apply_prefill_seed_locked`) and again at the first decode-band pass,
     replacing the old flat prompt-tally re-rank (which ranked by prefill routing — nearly uniform, because
     a prefill routes every expert).  Measured offline on a real 10.5K-prompt trace with the model's own
     `.moe-heatmap` (`scripts/moe-trace-replay.py --pool-frac`, new two-tier model), arena 369/pool 409
     slots per table: expert-id pool content = 3571 MiB of L3 reads over 451 decode tokens, heat-ranked
     complement = **237 MiB**; every eviction/hand-over policy measured within 0–246 MiB of that.
  2. **The demand path never evicts and never copies the arena victim into the pool.**  A resident expert
     is the source, a miss takes an EMPTY slot (`pool_peek_locked` / `pool_fill_empty_locked`), otherwise
     the per-table pinned bounce serves it.  The whole-table prefill staging (`moe_cache_stage_table`) now
     reads the pool too — it used to bypass it and re-read the entire host expert set from the file on
     every ubatch — and it fills empty slots, so a later ubatch finds them.  The background
     eviction-prefetch worker and its `MOE_HOST_POOL_BGFETCH` knob are gone with the eviction they served;
     `MOE_HOST_POOL_RAM_PCT` (default 80) caps the whole pinned pool at that % of `MemAvailable`.
     Measured on the gfx1201 field config (IQ4_NL Flash-Next, 2 GPU tensor, `-ncmoe 48`, MTP n7, 70K
     prompt, 128-token decode): pool hit **0.7999 → 0.9183**, evictions **4600 → 0**; 581→733 t/s prefill
     and 30.6→53.7 t/s decode at `DIO=0`, and a prefill that could not finish in 27 min (39 t/s) → 287 t/s
     at `DIO=1` (decode 23.1), against `pinned` 836 / 57.5.  Greedy text byte-identical old vs new.
     The measured-and-dropped candidates (a bounce-staged swap with the incoming's slot: 2x decode cost for
     0 MiB of L3; write-over-the-weakest-candidate: 105 MiB for 1 copy/eviction) are recorded in
     `archive/work/host-pool-layout/`.
* **POOL mode's fill mode is chosen per box, and its prefill staging is one launch (r114, TODO #87).**  The
  two remaining `#87` items, both in `moe-expert-cache.cu`:
  1. **`MOE_HOST_POOL_DIO` defaults to `auto`**: decided once, at the first pool build, from `MemAvailable`
     vs the host expert set — buffered (page cache as L3) when `planned pool + the whole set` fits in
     `MemAvailable`, O_DIRECT when it does not (`pool_decide_dio_locked`; `0`/`1` still force either arm).
     This closes the r113 "inverted optimum": the right arm was a hand-set guess that had to be
     re-measured on every box, and the shipped `DIO=1` cost a RAM-rich box 287/23.1 t/s (prefill/decode)
     against 733/53.7 at `DIO=0`.
  2. **The pool-backed prefill staging fills both tiers with ONE kernel launch.**  `moe_cache_stage_kernel`
     gained a pool source: its per-expert snapshot now encodes `>= 0` = arena slot, `<= -2` = pool slot
     `-v-2`, `-1` = in neither tier (the host already copied those, since each one is a separate read from
     the mapping/disk), so the ~200 `cudaMemcpyAsync` per table per ubatch that used to stage the
     tier-resident experts are one launch.  That per-copy issue cost was the rest of the gap to the
     pinned-master path, which has always staged with a single launch.
     Measured, 2x R9700 / 35B-A3B Q4_K_M `-ncmoe 40 --host-experts pool`, 70K prompt, `-ub 2048`: prefill
     **2103 t/s** vs **1564** with `MOE_HOST_POOL_STAGE_BATCH=0` (+34 %), decode 46.4 vs 46.2, against
     `pinned` 2099/49.3; `DIO=1` forced 2088/**22.7**.  Greedy text is byte-identical across every arm
     (`pinned`/pool/`DIO` 0 and 1/staging on and off) and `MOE_HOST_POOL_VERIFY=1` is clean.
  3. **Long-session adaptation: measured and deliberately not implemented.**  On a long, topic-shifting
     trace (a 1024-token code decode followed by a different-topic decode, through the two-tier model) the
     ceiling for *any* adaptation is the oracle layout: **7.45 MiB/token** of L3 reads vs **8.05** for a
     layout built from the previous topic and **7.55** for one built from the new topic — i.e. ~6 % of the
     L3 traffic per shift.  Every policy that adapts by churning costs several times that
     (`pool-retire` 24.6 MiB/token; re-admitting the arena victim into the pool trades 32.8 GiB of copies
     for 8.7 GiB of L3, the 2x-decode result r113 already dropped).  The one candidate left is a periodic
     bulk re-rank of the existing layout, bounded by that ~6 %.  Record:
     `archive/work/host-pool-swap-and-prefill/`.

* **The heat prior is split by band, and the arena seed ranks by the decode one (r114, TODO #88).**  `t.heat`
  is one counter fed by both bands (`moe_cache_heat_observe` for the prefill, `moe_cache_update_host` for
  the decode) and its consumers want opposite things: the **arena** serves the decode/verify band, while the
  **host pool** stages whole tables for a prefill.  A prefill ubatch routes ~`n_tok` experts, so on a
  long prompt the prefill outweighed the decode **41x** and the arena's seed was ranked by a nearly uniform
  distribution — the prior's top-112/table covered 80.3 % of the decode mass where its decode component
  covers **95.3 %** (the oracle's).  So: a second counter (`heat_dec`, fed by the decode host tally and by
  the *sampled identity* decode device tally through its own mirror `heat_dec_dev`), `heat_rank_vec()`
  for every HEAT-DRIVEN ARENA decision (the seed and the heat-ordered layer placement), and the combined
  vector kept for the pool's bulk layout and its `prefill_count` fallback.  The sidecar is **MHT2**
  (`all` then `dec` per table); **MHT1 still loads** with `dec = all`, i.e. the pre-r114 behaviour, so no
  map is invalidated.  Measured with the SAME map in both arms (35B-A3B Q4_K_M, 2 GPU tensor, `-ncmoe 40`,
  `MOE_EXPERT_CACHE_MIB=4096` = 43.8 % residency): `h` 0.8747 → **0.8862** (284-token prompt) and
  0.8698 → **0.8948** (5256-token prompt), evictions 18798 → 12402 on the long one; offline (decode band)
  0.9047 → **0.9400** and 0.8347 → **0.9397**.  Greedy text byte-identical (the prior only chooses which
  byte-identical copy is resident) and `gate-prefill-logits.sh` KLD 0.000000 / 100.000 %.

* **The expert cache's tables must not outlive the model that registered them (r143).**  `g_tables` is
  process-global state keyed by host-master pointers the MODEL owns, and nothing told it when a model went
  away: `llama_model_free` freed the host masters, and the deferred arena sizing (`alloc_all_locked`, which
  fills **every** registered table from its master) then read a freed pointer -- a host memcpy **SIGSEGV in
  `fill_slot_from_host`** from a table whose model no longer existed.  Reproduced on Flash-Next
  GSQ-IQ3_XXS, 2 GPU `-sm layer -ts 59,41 -ncmoe 24`: llama-bench runs one test per `-ts` value and frees the
  model between them, so the second test's first band pass died filling the first model's tables (gdb: a
  `memmove` faulting on a source address that is in **no** `/proc/self/maps` VMA).  `--host-experts pool`
  crashed the same way -- its master is the read-only model mapping, so a host-buffer-free hook could not
  have covered it.  `llama_model_free` now calls `llama_model_moe_cache_forget()` -> the new device iface
  entry `moe_cache_forget` -> `moe_cache_forget_all(device)`, which drops that device's tables (freeing each
  one's arena slot range, remap staging, heat mirror, bounce ring and **host pool**), re-points every id map
  so a reused `ggml_tensor*` registers a fresh table instead of inheriting a dead model's geometry, forgets
  the per-model preflight accounting (`g_preflight_device_bytes`/`_seen`, the unregisterable set, the
  latched floor) and un-sizes the cache.  Un-sizing is half the fix: a second model used to inherit
  `g_sized` and therefore stayed **uncached for its whole run**.  Single-model runs are unchanged (the
  teardown only adds the drop log lines) -- it is a lifecycle rule, not a policy.

* **The host pool's file I/O goes through five portability helpers, and a new call site must too (issue
  #231).**  The bounded host pool is a POSIX feature; every file call goes through `pool_open_file`,
  `pool_fd_align`, `pool_pread`, `pool_aligned_alloc` and (r147) `pool_close_file`, whose `_WIN32` versions
  are no-ops -- the pool never opens a file on Windows, marks itself failed, and a miss is served from the
  pinned master exactly as if it were disabled.  r143's teardown called `close()` directly in
  `pool_free_locked` and in `moe_cache_forget_all`'s table drop, which is a hard compile error with HIP on
  Windows (`close` undeclared; `<unistd.h>` is included only under `#if !defined(_WIN32)`), and r131 and
  earlier did not have them.  Behaviour on Linux is unchanged (`pool_close_file` is `close`).  No gate on
  this box can compile it -- there is no Windows runner -- so `scripts/check-windows-portability.sh` lints
  for it (tier-1 CI, via `validate-set.sh`): a new host-side file call in a file the delivery adds is
  either guarded on `_WIN32` or gets a sixth helper beside these five.

* **The host pool's available-RAM figure on Windows is `GlobalMemoryStatusEx` (r209, PR #269).**  The
  zero-copy lookups (`pool_available_ram_bytes`) and the windowed KV spill ceiling
  (`ggml_cuda_kv_available_ram_bytes`, block 15) read `MemAvailable` from `/proc/meminfo` and returned
  `-1` off Linux, so on Windows the pool always took the box for **memory-tight**: `MOE_HOST_POOL_DIO`
  resolved to `O_DIRECT` ("no MemAvailable") and neither `MOE_HOST_POOL_RAM_PCT` nor
  `LLAMA_KV_SPILL_RAM_PCT` applied a ceiling.  Windows now returns `ullAvailPhys` (free plus standby, the
  `MemAvailable` counterpart), so the existing policy decides from a real figure and `MOE_HOST_POOL_DIO`
  can resolve to `buffered`; forcing either fill mode still works.  Reporter's log lines (92 GiB box,
  `MOE_HOST_POOL_FRAC` 80 vs 50) land on the two arms the rule predicts.

**Knobs.**  `LLAMA_OUTPUT_PINNED`, `MTP_DRAFT_N_UBATCH`, `GGML_LF_DFLASH_DEV`, `GGML_LF_FAST_TOPK`,
`LLAMA_DFLASH_INGEST_TAIL` (r130), `LLAMA_STATE_GET_MERGE_GAP`, `LLAMA_STATE_ASYNC`, `LLAMA_STATE_ASYNC_MAX_MIB`,
`LLAMA_STATE_ASYNC_READ`, `LLAMA_STATE_PIPE_GROUPS` (r151), `LLAMA_STATE_COPY_THREADS` and
`LLAMA_STATE_TIMING` (r152), `LLAMA_STATE_STAGE_MIB` (r157),
`GGML_CPU_MOE_OFFLOAD_THREADS`, `GGML_CUDA_SLAB_RING_MIB`, and (r111+, POOL mode) `MOE_HOST_POOL_MIB` /
`MOE_HOST_POOL_FRAC` / `MOE_HOST_POOL_DIO` / `MOE_HOST_POOL_VERIFY` / `LLAMA_HOST_MASTER_PINNED`, plus
(r113) `MOE_HOST_POOL_RAM_PCT` (default **85** since r114), (r114) `MOE_HOST_POOL_STAGE_BATCH` and
(r130, server) `LLAMA_SERVER_LAZY_IDLE_SLOTS` / `LLAMA_SERVER_CACHE_REUSE`, (r155, server)
`LLAMA_SERVER_CKPT_REUSE`;
the heat prior's MHT2 sidecar is not a knob (an `MHT1` file is read as `dec = all`).
`MOE_HOST_POOL_BGFETCH` was removed in r113.

**Notes.**  A backend-iface entry is a compile-time contract: adding one to the struct means adding it to
**every** backend's initializer, or the build breaks for that backend.  Keep the difference documented when
a hook is CUDA-only (the non-CUDA entries stay NULL and the feature is simply inactive).

---

## Facts that span blocks

* **The band rule.**  Decode and every speculative verify width must take the same kernels, chosen from the
  `n_tokens` band and `n_kv` — never from the exact width.  Blocks 00/03/05/06/08/10 each contribute a piece
  of it; `GREEDY-PURITY.md` is the rulebook.
* **A block owns its code.**  A fix goes into the block whose surface it changes (even an early block); the
  per-block build gate (`scripts/gate-per-block-build.sh`) exists because appending fixes to the last block
  once broke blocks 06-14 silently.  **Four exceptions are deliberate and documented**: r106's
  `GGML_OP_ALL_GATHER` collective and the slab narrow-2 fix both landed in **block 12** rather than blocks
  09/10, because `ggml-cuda-slab.cu` does not exist before block 10 and re-cutting blocks 09/10's history for
  an additive, last-block-applied change would have rebuilt the whole chain for no delivery difference
  (blocks 00-10 are byte-identical to r105; only what is now block 12 changed — it was block 11 at r106);
  **r111's host-expert master drop** landed in block 12 rather than 09 for the same reason plus one of
  its own: its code depends on block 11's fused/`-sm tensor` slice geometry (`table_t::n_slices`), so a
  block-09 fold would have needed a second, pre-fused version of the pool path that block 11 then had to
  re-amend (blocks 00-11 are byte-identical to r110); and **r114's band-split heat prior** (a block-09
  surface) landed in block 12 because blocks 12's r111/r113 amendments already rewrote the same functions
  (`apply_heat_seed_locked`, the pool ranking) — a block-09 edit would have to be replayed through that
  rewrite, and the block-12 delta is where the reviewers' diff already is.
* **The Meta backend's only cross-device primitive was a sum-reduce (`PARTIAL`)** until r106 added
  `GGML_OP_ALL_GATHER` (block 12): the result is `MIRRORED`, it is its own subgraph boundary, it is dropped
  from every per-device child graph, and the collective copies each device's slice into every device's copy
  **one row (of every dim above the split axis) at a time** — the slice is contiguous in its own copy but
  strided in the full tensor, and the destination offsets are the canonical cumulative extents from
  `ggml_backend_meta_get_tensor_impl` (device order = tensor order; `rotation` only rotates which fraction a
  device gets).  Only a plain axis-0 split is implemented; anything else aborts by design.
* **Cross-backend consistency.**  The set is AMD-targeted but generic code: a new predicate, KV type or
  iface entry must stay consistent on every backend (an inconsistent set is a crash on whichever backend
  reaches it).  Behaviour/perf on non-AMD is out of scope — see AGENTS.md.
* **RDNA4 is the reference arch.**  The WMMA FA path, the chunked-GDN kernel and the VDR work were built
  and validated there first; RDNA3.0/3.5 get back-ports (their own kernel files, tuned head caps, their own
  `mmvq` parameter table) and RDNA2 is best-effort.
* **The arch gate is runtime** everywhere in the set, so one multi-arch binary picks the right path.
* **`ggml-cuda.cu` is shared by blocks 00/07/08/09/10/11** and is the usual re-base conflict site; the
  `template-instances/*.cu` files are block 05's and are generated (`generate_cu_files.py`).  A new FA
  kernel variant also goes into its own generated instance file, never implicitly into the dispatch TU:
  `nm -C` shows the kernel `W` in the instance TU and `U` in the `...-case-...` TU that selects it (the
  r98 q8_0 fetch-ahead variant is `fattn-mma-f16-band-fetch-ahead-ncols1_{4,8}-ncols2_8-dkq256-dv256.cu`).
  Give it its own TU whenever the two arms would otherwise change each other's compiled code.
* **A knob's `getenv` call site may land in a later block than the feature it controls** (the series is
  linear and `ggml-cuda.cu` accumulates): `ENVIRONMENT.md` is authoritative for what a variable does and
  its default, and `git grep` on the applied branch for its call sites.  A block's "knobs" list here is the
  set its own hunks read/write, not a claim of exclusive ownership.


### r207 (2026-11-03) -- the host-pool-speed campaign folded in

`archive/work/host-pool-speed/` was promoted here rather than as a new block (it is this block's pool code), and the campaign then retired there once its successor obsoleted the rest.
Everything below is `--host-experts pool` only; the default `pinned` mode is byte-for-byte unchanged.  The measured
record is `archive/work/host-pool-speed/README.md` §18 and `WORKLOG.md` 261103.009-012.

* **A POOL-backed table is served by the device gather in the prefill band by default.**  Its whole-table staging
  is the slow path at every width (the source is the pageable model mapping plus the pool's host alias, degrading
  as coverage rises): 122B-A10B, `-ncmoe 48`, 5258-token prompt, `-ub 2048`, three R9700s — pool FRAC 95 prefill
  **1507.7-1631.3 t/s** with the gather against `pinned`'s 634-638, and **16.9** with the gather off; decode at
  parity (67.2-67.7); prefill logits **bit-identical** (KLD -0.000000 / top-p 100.000 %).  A new optional iface
  query `moe_cache_pool_backed(backend, weight)` (CUDA: "this table's master is the read-only mapping"; forwarded
  by the Meta backend) plus a **band** gate in `sched_input_gatherable` (`n_tokens > moe_cache_band`) — the decode
  band keeps the arena's slot-remap read, because the gather there costs 1.2-4.1 t/s.  `GGML_SCHED_DEVGATHER` is
  three-valued: unset = pool-backed only (shipped), `0` = kill switch, `1`/`force` = the old A/B arm (not the
  default: it costs a pinned master ~47 % prefill at two devices).  The gather also resolves the arena first and
  the pool second, which subsumes `MOE_HOST_POOL_COLD` (`COLD=0` 19.57 vs 19.25-19.86 with it, 15.54 without).
* **`MOE_HOST_POOL_RAM_PCT` scales every table** instead of shaving the registration-order tail to the one-slot
  floor (`RAM_PCT=70` on the 122B: 10.94 -> **17.44 t/s** at identical pinned bytes, +73 % with `SIZING=1`;
  `RAM_PCT=85`: 19.52 -> 20.97), and `MOE_HOST_POOL_SIZING=1` fits its concentration plan *to* the ceiling.
* **`MOE_HOST_POOL_SIZING` / `_COV_PCT`** (opt-in): per-table slots from routing concentration — +7.5 % decode at
  FRAC 80, +16.4 % at FRAC 60 at identical pinned bytes, for -13/-15 % prefill.
* **Diagnostics**: `moe_cache_report` reports the run (it used to print an emptied cache in every mode — the
  artefact behind the "the pool never activated" false alarm), a forced `MOE_HOST_POOL_DIO` logs its mode,
  `bounce_acquire` owns the pinned bounce ring (2209-3108 MiB on the 122B, uncapped on purpose: a cap costs
  8-25 % of prefill at FRAC 80), pool fills are timed by source, and the gather reports its calls.
* **`MOE_HOST_POOL_PREFETCH` removed** (parsed, never read — `TODO` #268 closed).
* **Held back:** `MOE_HOST_POOL_FRAC=100` as the default (better on one GPU, 9 % worse on a split while the ceiling
  does not reserve the model's pageable footprint; over-committed a 91 GiB box once).  The default stays 80.

Gates: greedy text byte-identical on 1/2/3 GPUs (5835 `dc178c29fe89`, 6079 `01607c16b483`, 5917 `5421cf18bed5`;
257 `5531ae33f0f7` on the 64-token oracle); prefill logits KLD -0.000000 / top-p 100 %; MTP four-axis acceptance
identical to 5 decimals (0.81370 / 0.75408 / 0.89175 / 0.98148) with `none` == `draft-mtp`; `llama-batched-bench
-npl 4,8` pool 338.5/835.8 vs pinned 81.8/136.0; faults clean.

Open follow-ups (`../TODO.md`): #131 (the pool staging path's own root cause — now kill-switch-only), #132 (the
pair's remaining -21.7 % decode is one synchronous demand read per token), #133 (a fully-resident table is still
re-copied out of the arena per prefill ubatch — the *default* mode's own defect, up to ~3x).

## Block 13 — MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`, opt-in)

> **r217 amendment (issue #273 item 4, `getenv` audit):** `llama-graph.cpp`'s `LLAMA_GRAPH_RESULT_DEBUG` was
> read on every graph result (9 per run in the audit trace); it is latched now.  No behaviour change.

**What.**  The MTP draft head can score a **row subset of the LM head**: with `--spec-draft-mtp-vocab N` the
draft context copies the LM-head rows for token ids `< N` plus every control/user-defined id into its own
head (a **verbatim copy in the LM head's own quant type**), scores the draft matmul over that subset, and
scatters the result into a **full-vocabulary row of `-inf`**.  The speculator still receives a full-vocab
logit row; the target's verification is untouched, so the trim can only lose proposals, never change an
accepted token.  Default `0` = off: no subset is built and no graph node is added (measured byte-identical to
r106 on plain decode and on `draft-mtp` at n-max 1 / 9 / adaptive-9).

**Where.**  `common/arg.cpp` (+ `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB`) -> `llama_context_params::mtp_draft_vocab`
for the MTP draft context only (`common/common.cpp` for an `-md` sidecar, `common/speculative.cpp` for the
in-tree speculator) -> `llama_model::mtp_draft_vocab_get()` (`src/llama-model.cpp`) -> the graph hook in
`qwen35.cpp`, `qwen35moe.cpp`, `qwen4exp.cpp` and `gemma4-assistant.cpp`.  Ownership is per context
(`shared_ptr`, model-side weak-ref cache keyed by `(N, buffer type)`, compared on graph reuse).

**Guards.**  Arch must be one the hook knows; the block must score with the model LM head (a shared-NextN
sidecar borrows the target's, resolved through `cparams.ctx_other`); probe/`--fit` contexts (unallocated
weights) return `nullptr`; and the tokenizer must be BPE with ids in **near-merge order** (the check compares
each merge-produced id with the previous one and tolerates <= 1 % inverted pairs: qwen35 0, gemma4 30/236339).
A rejected combination logs `mtp_draft_vocab = N ignored: <why>` and drafts over the full vocabulary — the
activation line (`mtp_draft_vocab = N: MTP draft head uses R of V rows`) is `LLAMA_LOG_INFO`, so pass `-v`.

**The `ggml` half.**  Under `-sm tensor` the LM head is vocabulary-split, and a 2-D axis-1-split tensor used
to be readable only **whole** (`nb[axis + 1]` is the whole tensor for a 2-D tensor), so the subset build
aborted with `GGML_ASSERT(size % chunk_size_full == 0)`.  `ggml_backend_meta_get_tensor_impl` now splices an
arbitrary **row range** block by block (segment -> repeat -> device, with the per-device byte offsets advanced
exactly as the setter does).  Whole-tensor reads and every other split axis keep their old paths.

**Knobs.**  `--spec-draft-mtp-vocab N` / `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` (`ENVIRONMENT.md` §3).  **N is
payload-specific**: 98304 on Qwen3.8-27B Q8_0 (acceptance unchanged, all four gate axes improve), 64000 is
the reporter's gfx1151 value and is a **loss** on our 27B (prose -3.6 % at adaptive n-max 9), gemma4's only
lossless cut is 245760 of 262144 (a no-op).  Cost: a second copy of `N/n_vocab` of the LM head in VRAM
(510 MiB at 98304 on the 27B), per device under `-sm tensor`, not seen by `--fit`.  Leave `--spec-draft-n-start`
at its default 3 — it is the *initial* adaptive depth, not a ceiling, and setting it to n-max pins the
controller there (2-GPU prose 94.3 -> 87.6 t/s with a different acceptance).

**Device count — enable it only when `N * n_devices < n_vocab`.**  The subset head is allocated from the LM
head's buffer type, so under `-sm tensor` it is **mirrored on every device** while the untrimmed head is
**split**: the per-device draft matmul is `N` columns versus `n_vocab / n_devices`, plus a per-step
`fill` (a full-vocab `-inf` row) and `set_rows`.  On the 27B (vocab 248320), four axes at `-n 3000`:

| devices | per-device share | trim 0 | +98304 | Δ |
|---|---:|---:|---:|---:|
| 2 | 124160 | 109.4 | 110.9 | **+1.4 %** |
| 3 | 82773 | 113.1 | 110.6 | **-2.2 %** |

The width sweep is the control (98304 and 131072 have *identical* acceptance, so only the matmul width
differs): 2 GPU 94.3 / **95.8** / 93.1, 3 GPU **98.3** / 96.0 / 93.3 (prose axis).  At 3 GPU the width rule
(`N < 82773`) sits below the acceptance floor — 64000 measures 89.1 t/s (-9.4 %) — so **no N pays there**;
the 3-GPU advice is leave the trim off.  Depth-16384 decode is flat (2 GPU +0.1 %, 3 GPU -1.4 %).  The
gemma4 corollary (unmeasured): its only lossless cut is >= 245760, i.e. above the 2-device share, so even
the "no-op" cut would be a width loss.  Record:
`../archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md`; mechanism + numbers also in
`../archive/work/three-gpu-performance/FINDING-2026-10-22-dense-27b-and-mirrored-draft-subset.md`.

**Purity.**  `GREEDY-PURITY.md` §48: a draft-side restriction cannot change an accepted token, and inside the
`n_max <= 7` band the trim is *text*-identical to plain decode (verified on prose and code at `-n 3000`).
Above the band the trim shifts acceptance, the adaptive controller shifts depth, and verify-width numerics can
flip a near-tie — the same property the untrimmed build already shows between any two depths.  Gate it at
`n_max <= 7`, or on acceptance + MTP-vs-plain above it.

**Known interactions.**  adaptive controller (acceptance/depth, §48), `ngram-mod` (unaffected; shrinks the
trim's share), `-sm tensor` (subset head is Meta-mirrored — **the draft matmul is duplicated per device, so the
trim only pays when `N * n_devices < n_vocab`**; see "Device count" above), `--fit` (subset not estimated), the
purity band (above).  Detail and numbers: `../archive/work/spec-draft-mtp-vocab/README.md` §5.

## Block 14 — `-sm relaxed-tensor` for the MoE amax arches (**TEMPORARY**, expect removal)

> **Status: temporary, local, and NOT a supported feature of the delivery.**  This block exists so `flint`,
> `mellum` and `mistral4` can run `-sm tensor` today.  It relaxes a split-state rule upstream deliberately
tightened (PR #31081) and is expected to be **replaced or dropped** when upstream lands a real fix — the
> delivery is not committing to this mode.  Nothing else in the set depends on it: deleting patch `0014`
> restores exactly the r136 tree, and `-sm tensor` keeps upstream's strict behaviour either way.
> Read the limitations before using it; treat it as a stop-gap, not as a feature.

**Why upstream refuses.**  `llm_graph_context::build_amax_src1_f16()` (`src/llama-graph.cpp`) gives the MoE
down projection a per-`[expert, token]` rescale: it takes the max of the whole matmul row
(`ggml_pool_1d(abs(cur), POOL_MAX, k = s = n_ff_exp)`), divides `cur` by a power of two derived from it, and
multiplies the down-projection result back by the same value to restore f16/Q8 headroom.  Under
`--split-mode tensor` the MoE intermediate is split on axis 0 (`n_ff_exp`), so **no device holds a complete
row** and each one can only compute the max of its own slice.  #30861 handled `POOL_1D` by reporting the
result `MIRRORED`; that claim is false (the values differ), so #31081 removed the case and denied
`-sm tensor` for the three arches.

**What this block does instead of a blanket deny.**  A new split state,
`GGML_BACKEND_SPLIT_AXIS_PER_DEVICE`: same shape and element positions on every device, **its own values**
on each.  `handle_pool_1d` produces it for a whole-row `POOL_MAX` over an axis-0 split (anything else keeps
the old reject), `handle_bin_bcast` lets it be consumed elementwise and broadcast, and every path that would
copy, gather, reconcile or read it back **aborts with a message** instead of silently returning one device's
value.  `-sm relaxed-tensor` (`LLAMA_SPLIT_MODE_RELAXED_TENSOR`, `relaxed-tensor` in `--help`) is that
permission; every "is this the tensor path?" site goes through `llm_split_mode_is_tensor_like()`.

**Why it is exact, and the one piece that is easy to get wrong.**  The reduce point must stay **after** the
per-device undo, i.e. each device applies its own `e` to its own partial and only then is the cross-device
sum taken.  That is the delayed-reduce MUL-chain in `ggml_backend_meta_graph_compute`, and it only works
because it accepts `PER_DEVICE` next to `MIRRORED`; if it stops matching, the all-reduce lands on the
`mul_mat_id` output and **one device's `e` is applied to the summed result**— wrong, and invisible whenever
every `e` happens to be 1 (which is the common case: the shipped divisor is 1024, so `e > 1` needs a row max
above 1024).  Given the ordering, `e` is a power of two and `e_local <= e_global` (a local max cannot exceed
the global one), so the scaling commutes with the sum and each device keeps at least the reference's f16
headroom.

**Cost.**  Zero VRAM: the weights are placed exactly as `-sm tensor` places them (Flint Q8_0, 39.39 GiB:
95.1 % is the expert tables, which stay split — 20.66 GiB/device at 2 GPUs, 14.42 GiB at 3).  The only
per-device tensor is the rescale factor, `[1, n_expert_used, n_tokens]` per MoE layer, in the existing
compute buffer.

**Limitations — do not read this block as general `-sm tensor` support.**

| limitation | detail |
|---|---|
| not bit-identical everywhere | exact on RDNA/HIP (Q8_1's block scale is `max/127`, so a power-of-two factor passes through). A backend whose in-kernel `src1` rescale is **non-linear** (Vulkan's clamp-into-fp16-range, #30656) sees genuinely different values. Not validated outside ROCm |
| no `-ncmoe`/expert-cache arm | the MoE expert arena still refuses tensor-like modes, so `relaxed-tensor` inherits that (as `tensor` does) |
| `mellum`/`mistral4` unvalidated | enabled by the same change, no model on hand. `mistral4` was denied *before* the amax rescale, so it may still hit an unrelated unsupported path |
| no MTP/draft coverage | no draft was run against these arches under this mode |
| it is not a purity claim | the mode is *permitted*, not *promised*: `PER_DEVICE` is an assertion that the graph consumes the tensor only in the sound way, and any other consumer aborts |

**Validation (r137).**  Flint Q8_0, 2× R9700: `-sm relaxed-tensor` runs at 77.9 t/s generation vs 59.5 for
`-sm layer` (short decode-only run, not a depth-16384 measurement), greedy text **byte-identical** to
`-sm layer` and to itself at 3 GPUs.  With the amax divisor forced to 1 (so `e != 1`), an instrumented pool
kernel confirms each device pools its own 256-wide slice and the per-device maxima differ on ~half of the
~185 calls per run, and the text still matches `-sm layer` exactly.  A non-amax arch (Qwen3.5-4B) is
byte-identical under `tensor` and `relaxed-tensor`, so the mode is a no-op without the amax node;
`ggml_pool_1d` has exactly one producer in the graph, so the state is unreachable from any other arch.
Purity notes: `GREEDY-PURITY.md` §58.

## Block 15 — the streamed KV: the resident WINDOW (`LLAMA_KV_RESIDENT_CELLS`), the attendance POOL (`LLAMA_KV_REGION_POOL`) and the MTP draft WINDOW (`LLAMA_SPEC_DRAFT_WINDOW`, **default on**)

> **r217 amendment (issue #273 item 4, `getenv` audit):** the region path's diagnostics
> (`LLAMA_KV_REGION_TRACE`, `_DEBUG`, `_CANARY`) were read inside `vmm_map` / `vmm_unmap` / `vmm_remap`, i.e.
> once per mapping when a KV window is active; they are latched behind `vmm_trace_on()` / `vmm_debug_on()` /
> `vmm_canary_on()` now.  This block's other files (`ggml-cuda.cu`, `ggml-backend-meta.cpp`,
> `moe-expert-cache.cu`, `llama-context.cpp`, `llama-model-loader.cpp`, `llama-memory-hybrid-idx.cpp`,
> `models/qwen4exp.cpp`) are latched as well -- including `GGML_STAGE_TRACE`, which the loader read **per
> tensor** (144-240 calls per run).  No behaviour change; the windowed/pooled arms are unchanged.

This block is the delivery's "the KV does not have to be `O(-c)` of VRAM" work: **two** opt-in placements of the same cache, plus the third variant that is **default on** because it measures non-negative (the MTP draft window, r193).  The **window** (Goal A of `TODO` #118, r178) keeps the newest `W` cells in VRAM and spills the tail to pinned system RAM in the same VA reservation.  The **pool** (Goal B, r190) keeps the authoritative copy of the *whole* context in a pinned host mirror and only the pages a decode step attends to in VRAM, resolved through a page table.  They answer the same question from opposite ends (newest cells vs attended cells), they are mutually exclusive by construction (the pool declines while a window is active), and each is byte-identical to a build without it when its knob is unset.  Both report `is_streaming()`, so a streamed cache takes the sparse QSA decode arm (also this block: B2).

### The window (Goal A, r178)

Promoted from the `archive/work/kv-residency` campaign in **r178**.  Opt-in: with
`LLAMA_KV_RESIDENT_CELLS` unset (or `0`) the KV takes the **plain device buffer** and the run is
byte-identical to a build without this block, including its OOM.

* `LLAMA_KV_RESIDENT_CELLS=<cells>` — keep the newest `<cells>` cells of the KV in VRAM; everything older
  goes to pinned system RAM.  `=auto` lets `--fit` size it, and only for a context the user pinned with `-c`
  (the fit then reports the resident/spill split, the per-token read cost and the depth below which nothing
  spills).  `LLAMA_KV_SPILL_RAM_PCT` (default 85 % of `MemAvailable`) caps the pinned spill and **widens the
  window** rather than letting the run be OOM-killed.
  On Windows `MemAvailable` comes from `GlobalMemoryStatusEx` (r209, PR #269, the helper block 12 owns), so
  the ceiling applies there instead of being skipped.
* **`--fit` interaction:** the fit identifies the KV by the region buffer type, and its measurement context is
  a `no_alloc` probe that declines a window — so the probe takes the region buffer explicitly
  (`llama_kv_region_set_probe`, internal).  Without that (r178) the fit could not see the KV, reported an
  over-budget config and aborted instead of sizing the window (fixed in r179).
* **A tensor split (`-sm tensor`) carries the same window since r204 (`TODO` #124).**  The window is still one
  VMM region **per device**: the Meta (tensor-split) device gets a Meta buffer type whose children are the
  per-device region buffer types (`ggml_backend_meta_kv_region_buft`; all of them, or it declines), every child
  commits the same `W` cells of **its own slice** at its own cell stride, and the cell watermark's grow fans out
  to every child (`ggml_backend_meta_kv_region_grow`; a child it cannot reach is a refusal, never the silent 0
  the CUDA hook returns for "not a region buffer" -- that 0 is a batch that faults the GPU).  `--fit` sizes it
  for the split: the window is charged at its **physical** `W x cell_bytes` (the sum over devices), the per-device
  check is the ratio-scaled `min_i(target_i / ratio_i)` the branch already fits against, and the Meta entry's
  margin is the **sum** of the children's (it used to be child 0's, which omitted `(n_devices - 1) x -fitt`).
  A MIRRORED cache (dsv4 / HRM, where every device holds the whole cache) is **declined**: `n_devices x W` cells
  for one window, and the pinned-spill plan is per logical cache, so it would under-book by the device count.
  Before r204 the Meta device had no backend reg, so the region proc addresses were unreachable through it and
  the window was declined with one message (the r181 crash fix; see the block-15 note in `WORKLOG.md`
  261030.005).
* **Two paths a split exposed, fixed in the same block (r204).**  (a) The window's **init-time commit** was
  rounded up to a whole map unit past the tensor's own extent; on a split every child region is its own VA
  reservation, so the last tensor's inflated window ran past the end of a child reservation and its commit was
  refused (`the KV window (4.00 MiB) could not be committed in VRAM on device 1`).  It is now clamped to the
  region's end, in whole units.  (b) The **batched state save** issued one 2-D D2H per device
  (`ggml_backend_cuda_get_tensor_2d_async`), and a 2-D copy needs a device-resident source: a windowed region's
  spilled cells are pinned **host** memory mapped into its VA, so the checkpoint save failed
  (`ROCm error: invalid argument`, `cache_k_l64 (view)`) and a restore wrote into the region's VA without mapping
  the destination cells first (`grow_for` is what maps the right kind).  Both directions now go through the
  region buffer's own copy for a region-backed tensor (contiguous rows in one copy), ordered behind the compute
  stream like the pool mirror, and the projection (`set_tensor_2d`) is kept to a tight block only when **both**
  pitches equal the width -- a caller's destination pitch is the split's full row, wider than one device's slice.
  `scripts/gate-state-pipe.sh` with `LLAMA_KV_RESIDENT_CELLS` set is the arm that covers this.  (c) The slide
  itself now flips **several whole mappings** in one `ensure_span` call: one batch can advance the boundary by
  more than one unit, and a region whose unit is floored to the driver's granularity has smaller mappings than
  the unit it was asked for (r204 measured it at a 1 MiB request on a 2 MiB granularity).  A request that covers
  a mapping only in part is still refused.
* **`-nkvo` declines the window, and the streamed arm keys on the PLACEMENT (r197).**  A window is one VMM
  region per device plus a pinned host tail, and the per-layer buffer type is chosen inside `if (offload)`:
  with `-nkvo` the cache is pageable host RAM end to end, so there is nothing to window.  The plan block's
  guard did not test `offload`, so it still logged "WINDOWED KV ENABLED" (measured: a plain `CPU KV buffer`
  holding the whole context while announcing a 64-cell VRAM window), reserved pinned spill budget and left
  `kv_resident_cells` set -- which `is_streaming()` turned into qwen4exp's sparse decode arm for a KV with
  no region at all.  The window now declines once, with the reason (the pool already declined on
  `!offload`), and `is_streaming()` tests `kv_region`, the placement actually taken, which also covers a
  device cache whose region buft the backend refused.  Text is unchanged in every arm: the decline is a
  no-op, since `-nkvo` + window already read as `-nkvo` alone.
* **`auto` is a decision, not a guess (r181).**  The fit hands down three distinct states — a window, "the KV
  fits fully resident, keep it" (silent), and "nobody decided" (which still warns) — and it only takes the
  pinned-RAM clamp while the widened window still meets the VRAM target.  If the `MemAvailable` ceiling would
  need a window wider than VRAM allows, the fit says so and drops the window (then fits the context by
  shrinking `-c`), instead of planning a pinned spill larger than the machine.
* `LLAMA_KV_REGION=0` disables the provider; `LLAMA_KV_REGION_QUARANTINE=0` is the diagnosis switch that
  reproduces the driver hazard (never ship it), and `LLAMA_KV_REGION_DEBUG/TRACE/CANARY/DUMP` are the
  instruments.  `GGML_CUDA_VMM_REGION_SELFTEST=1` runs the provider's own selftest (it must PASS, and must
  FAIL with the hold disabled).
* **Cost:** the spill is read over the link on every step — `(used − W) × cell_bytes / 55.7 GB/s` on the
  validation box — so `W` belongs at or above the depth you actually run.  Measured at `-c 262144`, a `W` at
  the run's depth is free (+2.2…+2.9 GiB of arena, 122B-A10B +3.8 % decode / +4.7 % prefill); a `W` 35 k
  cells below it costs 11-26 %.
* **Purity:** a spilled KV is numerically transparent — prefill-logit KLD −0.000000 / 100.000 % on a spilling
  27B, −0.000000 / 99.994 % on a spilling Flash-Next (two KV caches), and every text arm byte-identical to
  its plain reference.  The state, context-checkpoint and MTP verify-width gates all pass on arms that
  provably spill (`FLIP > 0`).
* **Not in the window's own part:** Goal C (the MTP drafter ring, `LLAMA_SPEC_DRAFT_WINDOW`) and the
  superseded A1 two-base FA read (Goal B is the pool below).

### The attendance pool (Goal B, r190)

Promoted from the `archive/work/kv-residency` campaign in **r190** (the window above is Goal A of the same campaign;
this is Goal B).  Where the window keeps a positional tail of the KV in VRAM, the pool keeps the **whole**
context in pinned host RAM and only the pages a decode step actually attends to in VRAM.  Opt-in: with `LLAMA_KV_REGION_POOL` unset (or
`0`) the attention KV takes the plain device buffer and the run is byte-identical to a build without this
block, including its OOM.

**Measured cost when the KV already fits (r216).**  The page-fill path below is per *decode* step, so at
**prompt** time it is pure overhead: on one RX 7900 XTX at `-c 204800` with Qwen3.8-Flash-Next IQ3_S
(`-ncmoe 48`, q8_0 KV), `LLAMA_KV_REGION_POOL=16384` read **427.5 t/s at pp32K against 829.3** with the plain
device KV (-48 %; 409.2 against 808.2 at pp113K), and bought **+12 % decode** (59.4 against 52.8 t/s over 192
tokens) -- which is exactly the 2.3 GiB of expert arena it frees (12 084 against 9 742 MiB).  The KV of this
model is ~13 KiB per token (`2.6 GiB` at `-c 204800`), so on a 24 GiB card the pool's fittability case does not
arise.  **Do not enable it on a prompt-heavy server**; for the arena, prefer Goal A's window, which costs no
kernel change.  `WORKLOG.md` 261105.001, `INVARIANTS.md` r216.

**What it is.**

* The attention cache's **cell-indexed tensors become the pinned host MIRROR** and stay **authoritative**, so
  every existing path — the `set_rows` writers, `state_write`/`state_read`, the rope shift behind
  `defrag`/`seq_add`, `seq_park`, `get_k`/`get_v`, the kq mask — works on it **unchanged**.  That is the
  writer rule, and it is why the state, context-checkpoint and parking gates need no pool-specific code.
* A compact **DEVICE POOL** holds `LLAMA_KV_REGION_POOL` cells per layer as pages of **one indexer block**
  (`compress_ratio` cells), with a page table `tbl[block] -> slot`, its inverse, a per-slot epoch mark and one
  shared counter block.  At the measured shape that is **204 MiB** against the resident KV's 816 MiB
  (`-c 65536`) / 1 632 MiB (131 072) / 2 448 MiB (196 608).
* **`GGML_OP_KV_POOL`** resolves a decode step in three launches per layer: **MARK** (`use[slot] = epoch` for
  every page this launch reads and every page the batch just wrote, queueing the written ones for re-fill),
  **CLAIM** (one slot per *distinct* missing block, never one MARK protected), **MAP+FILL** (write each
  selected cell's pool position into the reader's id list and copy the claimed and dirty pages from the mirror;
  K and V share slots).
* The QSA reader takes a **second id list** (`ggml_flash_attn_qsa_set_kv_idx`), so the K/V gather follows the
  page table while the mask and visibility keys stay indexed by the **cell** number.

**Invariants (each one cost real time; the greedy text caught all three defects).**

1. **MARK then CLAIM** — a claim must never evict a page the same launch reads.  Without MARK the round-robin
   victim could be a page another row still needed (text diverged ~400 chars in; a pool larger than the whole
   cache hid it completely).
2. **Claim attempts are not distinct blocks** — a slot per *id* attempt wrapped the cursor within one 4-row
   batch, so two blocks shared a slot and a just-resolved block went unresident.  The table CAS dedupes first;
   `kv_pool_serves()` bounds the distinct count, and the pool is floored at one decode band's worth of blocks.
3. **No transient scratch in a captured graph** — the fill list lived in a `ggml_cuda_pool_alloc` freed at the
   end of the op, and a captured graph bakes the pointer, so a replay wrote into whatever op had been handed
   that block ("values verify clean, text differs run to run").  It lives in the counters tensor now.
4. **The mirror is authoritative** — an eviction costs nothing and nothing is invalidated, *except* by a path
   that rewrites cells in place without going through the writers (the K-shift, a state restore), which calls
   `kv_pool_invalidate()`.
5. **Eligibility is narrow on purpose** — one stream, non-MLA, K and V both present, `!v_trans`, a uniform
   `compress_ratio` and cell geometry, **no window active**, a device-offloaded cache, a mirror buffer type from
   **every** device the cache's layers are on (a `-sm layer` split qualifies since r213, a tensor split does not),
   and only the **trunk** context of a qwen4exp model (the MTP draft's cache is read densely: pooling it would
   only make its reads leave VRAM).  Anything else declines with one message and the KV is unchanged.

**Knobs.**  `LLAMA_KV_REGION_POOL=<cells>` (unset/`0` = off), `LLAMA_KV_POOL_VERIFY=1` (the oracle: every
selected page compared byte-for-byte with the mirror at read time, 0 mismatches expected),
`LLAMA_KV_POOL_STATS=<path>` (a **path**, appended at teardown: a line on stderr would sit inside the generated
text).  `ENVIRONMENT.md` §18.

**`--fit` (r190).**  The mirror is a *device* buffer type over a pinned host allocation, so `--fit` would
otherwise book its bytes as VRAM — at 103k-180k that reserves 1.6-2.4 GiB for memory that is really host RAM,
i.e. it hands back the very arena the pool exists to free.  `llama_kv_cache::memory_breakdown()` charges the
mirror to the **host** bucket (the same treatment block 15's spill gets), so the device tier is the compact
pool alone; `llama_get_kv_pool_bytes()` reports the split for the fit's report; and the fit **skips the
context-reduction interpolation** while the pool is active, because this KV's VRAM is fixed — reducing `-c`
frees nothing there and the interpolation would divide by a near-zero bytes-per-context delta and hand back a
nonsense (huge) context whose pinned mirror could OOM the host.

**What it is worth** (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`, q8_0, the PLE flags, arms interleaved,
`-n 512`, MTP depth 4):

| depth (ctx) | dense | sparse | **pool (204 MiB)** | text (`-n 64`) |
|---|---|---|---|---|
| 5 298 (16 384) | 72.09 | 71.58 | **71.27** | pool == sparse `332 dd1981213373` |
| 102 899 (131 072) | 59.10 | 54.70 | **54.65** | dense differs (a regime change) |
| 180 035 (196 608) | 48.94 | 43.16 | **43.41** | pool == sparse `861 263c7f2be4d9` |

The pool lands **on the sparse ceiling** at every depth (+0.5 / +0.6 / +0.6 %, i.e. noise), so its placement is
free and what remains against a resident KV (−1 % at 5.3k, −7 % at 103k, −11 % at 180k) is the **sparse decode
arm's own indexer cost** (block 11), not the KV's storage.  The freed VRAM goes to the expert arena: 103k plain
decode, `-n 2000`, 23 579.8 -> **25 018.6 MiB** (+1 438.8 MiB, +18 slots/table), expert hit **96.26 -> 97.02 %**,
cold reads **42.0 -> 31.6 GiB**.  The pool's own cost is **0.10-0.11 ms/token** (mark/claim/map+fill) with fills
0.03-0.05 ms/step at 92-99 % hits — which is why a better page policy (LRU / F-LRU / a heat prior) is **dead**
at this shape: the measured ceiling was 90.2 % vs 90.05 % observed.  At 51 475 depth a `-c 65536` run reports
**0 mismatches** and 91.5 % hits; at 5.3k, 96.0-99.3 %.

**Default OFF — do not "fix" that.**  It is a **fittability** feature like the window above, not a speed one: on a
config whose KV fits, the pool costs the sparse-vs-dense gap (−7 % at 103k, −11 % at 180k) and buys nothing.
It is worth it only where the resident KV does not fit — then the 204 MiB pool *is* the run (a Flash-Next arm
at `-c 262144 -ncmoe 16` aborts with `cudaMalloc failed: out of memory` / `failed to allocate buffer for kv
cache` with the knob unset and **runs** with it), and the arena it frees is worth +1.44 GiB.

**Multi-device (r213).**  The pool is **per layer and per device**: on a `-sm layer` split with N devices the cache
gets one pinned host mirror **buffer type per device**, one device pool and page table per layer, and **one counter
block per device** (`kvpool_ctr_<device>`); a layer's resolve node therefore has every source on its own device.
Two defects had to be fixed for that to hold, and both are "do not do that again": (1) the mirror buft's **name must
be device-unique** — `llama_kv_cache` keys its buft -> context map by name (`strcmp`), so a device-invariant name
collapsed the per-device mirrors into device 0's (the windowed KV's region buft is unique for the same reason);
(2) the counter block must not be shared — created once in the first pool layer's device it put device-0 memory in
every other device's resolve node.  Either defect alone makes the `KV_POOL` node unclaimable (the CUDA backend
rejects an op whose sources live on another device, the CPU backend has no `KV_POOL`) and the scheduler aborts with
`GGML_ASSERT(*cur_backend_id != -1)` at load.  `LLAMA_KV_POOL_STATS` **sums** the per-device counters and prints the
block count.  A **tensor split** still declines (its Meta device has no per-device mirror), one message, KV unchanged.

**Not in this block:** the pool declines a tensor split (one page table per device would need a per-device pool
and a split-aware resolve — block 15's window has the same limitation), the *mirror* grows on demand (r196) and
is bounded by the pinned-RAM ceiling `LLAMA_KV_SPILL_RAM_PCT`, which declines the pool loudly rather than being
OOM-killed, and k-quant indexer keys stay on the chain (block 11's loader needs a 32-element block that
divides `idx_dim`).

**Gates (r213).**  `gate-kv-residency.sh` PASS on tree `2125cfab7`, now including a **multi-device section**
(`pmref`/`pmres`/`pmfloor` on `MULTI_DEVICES=0,1`, `-sm layer`): text `f260d1824ab3` identical across the three 2-device
arms (`pmres == pmfloor == pmref`), the enable line reporting the 2-device span, **2 counter blocks summed**, `bad 0`,
and on the same 252 resolves `pmfloor` claiming 267.2 vs 118.0 per resolve with a lower hit rate (89.19 % vs 91.22 %).
Single-device arms unchanged (`pres` 145.3 / 90.83 %, `pfloor` 222.7 / 89.67 %); same-seed text byte-identical to r212 on
the 4B coherence arm (`1c5d32ac537d`) and on the 1-GPU pool arm (`f260d1824ab3`); `validate-set.sh` GREEN.

**Gates (r190, on the final base).**  Greedy text pool == the resident sparse arm at 5.3k
(`332 dd1981213373`), 51 475 depth (`854 43b30de151b9`) and 180k (`861 263c7f2be4d9`), with the dense arm
differing as documented; `LLAMA_KV_POOL_VERIFY=1` **0 mismatches** on every arm; `gate-prefill-logits.sh --ab
"LLAMA_KV_REGION_POOL=16416"` **0.000000 KLD / 100.000 % top-p** (base PPL 2.9225 — bit-identical to the
campaign's own recorded base, sha256 `acfb4145…`); `gate-park-idle-slots.sh` **PASS** (6 park + 2 unpark moves
traced, `park == lazyoff` on all four requests); `gate-ckpt-reuse.sh` **PASS** (keep 3, 5 checkpoints created vs
8, every request's text identical); `gate-state-pipe.sh` **PASS** (all five arms); `test-backend-ops -o
FLASH_ATTN_QSA` pass; `check-gpu-faults.sh` clean; the standard 27B Q8_0 prefill-logit point 0.000000 /
100.000 % against a base recorded on the shipped r189 build; `--fit` accepts the arms that only fit with the
pool (`-ncmoe 21`/`22` at `-c 262144`: refused without it, accepted with it) and reports both numbers
(`-fitp`: device context 3 784 -> 728 MiB, Host context 0 -> 3 264 MiB).  Numbers and the full campaign record:
[`../archive/work/kv-residency/BA-POOL-DESIGN.md`](../archive/work/kv-residency/BA-POOL-DESIGN.md) §7-§10.


#### r190 — a STREAMED cache takes the sparse decode arm

**What it is.**  A window below the depth the run actually reaches makes the tail of every KV read a PCIe read
instead of a VRAM read, so the *dense* QSA decode arm — which reads the whole used extent every step — becomes
the wrong arm there, while on a resident KV it is the right one (VRAM is ~10x the link).  The arch policy only
knew **depth** (`qsa_dense_decode_until`), never the cache's placement.  `llama_kv_cache::is_streaming(n_kv)`
(and the context accessor) answers "a window is active and the used extent has passed it" — or "this cache is
pooled" (this block's pool) — and `qwen4exp_kv_streams()` skips the dense decode arm when it holds.
`LLAMA_QSA_DENSE_DECODE_STREAM=0` restores the pre-r190 behaviour.

**Worth** (Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, q8_0, 1 GPU, `-n 400`, 2 reps, ±0.2 %): with a windowed KV a
streamed cache decodes **+34.3 %** at 102 899 depth (23.42 -> 31.43 t/s) and **+47.3 %** at 180 035
(16.51 -> 24.31); a resident KV is unchanged (`853 77a40bf40be1` with the switch on and off), and the arena on
the streamed arms was 25 338.3 / 22 701 MiB.

**Purity.**  This is a decode-ARM choice, i.e. a different attention regime: a *streamed* cache's greedy text
differs from the old dense-arm text by construction (`GREEDY-PURITY.md` §45/§63), while a **resident** cache is
byte-identical (`853 77a40bf40be1`).  Prefill is untouched (always QSA), so the prefill-logit gate is
unaffected.  Gate it with the text identity of the two resident arms plus a streamed arm's sparse-vs-default
identity.

#### r193 — Goal C: the MTP draft window, and the reserve that makes it pay

The MTP draft context's KV is a sliding window instead of `-c` cells.  **This is the one knob in this block
that is default ON**, because it measures non-negative everywhere and positive where the freed VRAM has
somewhere to go — the delivery's own rule (a beneficial, gated feature is on; the env var only disables).

* `LLAMA_SPEC_DRAFT_WINDOW`: unset or `auto` = the adaptive policy (**a quarter of the context, clamped to
  `[32768, 65536]` cells), `0` = off (byte-identical to a build without it), `N` = that many cells **per
  sequence** (floored at one ubatch).  The clamps are the safety rails: a window far below the context the
  drafter uses costs 7-9 % of decode, a window at or above it is free, and at `-c <= 32768` the window is
  dropped entirely.
* **The ring is the shipped reclaim**, not a new allocator: `llama_kv_cache::find_slot` already reclaims a
  cell more than `n_swa` behind, and `set_input_kq_mask` reads *this cache's* `n_swa`, so `hparams` (shared
  with the target) is untouched.  The derived kq mask stays valid as well — its predicate compares each
  cell's **position**, not its index, so a recycled cell reports its new one.  (The `wip` design note listed
  that as a hazard; it is not one.)
* **MTP only, and only an embedded MTP head** (qwen3.5/3.6/3.8).  A separate MTP draft GGUF (qwen4exp /
  Flash-Next) keeps its full cache and its flat reserve — its draft model's path is not known where the
  reserve is derived, so that case is deliberately left alone.
* **The reserve is what makes it worth shipping.**  An embedded MTP head has no GGUF of its own, so the slab's
  draft reserve was the flat 3584 MiB and the freed VRAM never reached the arena.  With the window active the
  reserve is now derived from the model's own metadata with the weights zeroed (they are the target's) and
  the KV sized by the window: **aux 3.50 -> 0.76 GiB, arena +2.8 GiB**.  An un-windowed run keeps the flat
  constant exactly, and a *declined* arena preflight now re-evaluates when the smaller real value arrives
  instead of latching on the load-time estimate.
* **`--fit`:** the draft part of the fit's margin is the derived reserve (`common_draft_reserve_estimate`), so
  a windowed MTP head is budgeted at its real size; the load-time arena preflight still books the flat
  constant (the model metadata is all it has), which is the conservative direction and is why the preflight
  re-evaluates.
* **The window is per SEQUENCE, and a UNIFIED cache is sized per sequence (r200, issue #264).**  With
  `--kv-unified` the one draft cache holds every sequence, so its cell count is
  `(win + one batch) x n_seq_max`, capped at the context; a `win + n_ubatch` cache was shared by all of them
  and a second sequence had no cell left to place a batch in once the first had filled the window.  A
  per-stream cache (`--no-kv-unified`) already sized each stream separately and is unchanged, and
  `n_seq_max == 1` (every `llama-cli`/`llama-bench` run) sizes exactly what it did before.
  `llama_spec_draft_kv_cells()` is the one geometry function — the cache, the slab's draft reserve and
  `common`'s "is a window active" test all call it, so a reserve can never under-charge an allocation.  The
  window is **declined**, not lowered, when `win` per sequence does not fit the context (a smaller window
  would save one batch of cells per sequence and mask the drafter's newest cells to do it).  Gate:
  `scripts/gate-spec-draft-window.sh`.

**Measured (1 GPU, one R9700, `-n 2000`/`-n 3000`, static depth 3, interleaved).**

| arm | arena | residency | decode | drafts |
|---|---|---|---|---|
| 35B-A3B Q8_0 `-ncmoe 48 -c 262144` q8_0 KV, 51 433-token prompt, window 0 | 20 910 MiB | 64.1 % | 111.2 t/s | 1995/3011 |
| the same, `auto` (65536) | **23 715** | **72.7 %** | **113.7** (+2.2 %; +2.8-4.9 % at `-n 2000`) | 1995/3011 (**identical**) |
| the same, `W=32768` (truncating) | 23 715 | 72.7 % | 110.3 at `-n 2000` (+1.7 %) | 1308/2071 |
| dense 27B UD-Q4_K_XL `-c 102400`, Protocol A `-n 3000` | — | — | layer 71.9 -> 72.1, tensor 72.6 -> 72.5 (neutral) | 2083/2747 all four |

Prefill +3.4 % on the arena arm.  **Purity:** target greedy text byte-identical across off/auto/4096, and
plain == draft-mtp == draft-mtp(W=512) at `-c 16384` with a 5256-token prompt and 3000 generated tokens
(10 158 chars, `sha=99b267e8dc3e`) — constant cell reuse moves no token.  The standard 27B Q8_0 prefill-logit
point reads 0.000000 / 100.000 % on the shipped tree (the change is decode-only by construction: every path
is gated on `ctx_type == MTP`).

**Gates on a windowed draft:** `gate-park-idle-slots.sh` (35B, `-ncmoe 48`, `-c 65536`: 3 park moves, park ==
lazyoff on A-D), `gate-ckpt-reuse.sh` (4B, `-c 131072`: draft cache 33 280 of 131 072, keep/eager arms
differ as required), `gate-state-pipe.sh` (35B, `-c 65536`: all five arms) — all PASS with `MTP draft KV
window = 32768 cells` in the logs.  Release gates: `validate-set.sh` 16/16 + tree equality, per-block build,
and the MTP Protocol A arm above.

**r194 — three corners the r193 release hit in practice.**  Found by running the released build against
Flash-Next (qwen4exp, `-ncmoe 48`, `-c 262144`, the pool) in a configuration battery:

* **The pool is now capped at the cache.**  `LLAMA_KV_REGION_POOL=1000000` used to allocate a 12.4 GiB device
  pool of 250 000 pages on a 262 144-cell context (65 536 blocks).  Pages past the cache's last block are never
  claimed, so the request is clamped to the cache's cell count: the same arm now takes 3264 MiB / 65 536 pages,
  which is the most it can use.  A small context is clamped the same way (`-c 8192` -> 2048 pages, the whole
  cache).  Requested sizes below one decode band are still floored to the band, and a non-page-multiple is
  still padded.
* **The draft KV window no longer sizes a SEPARATE draft's reserve.**  Flash-Next's MTP head is a `-md` GGUF
  whose draft context keeps its full `-c` cache, but the reserve estimate applied the window to it anyway:
  921 MiB derived where the honest number is 1566 (kv 217 instead of 862) at `-c 262144`.  That is the
  under-charge direction the reserve header warns about.  The window now enters the estimate only for a
  `shares_model` draft, i.e. the embedded-head cache `create_memory` actually windows (35B stays at 776 MiB).
* **The VRAM cliff needs a bigger outside cushion, and that is a config fact, not a code change.**  At
  `-c 262144` on a 32 GiB card with experts on the host, the then-default slab headroom (768 MiB on one GPU) is
  one hipBLASLt workspace short: the first *prefill* fails with a hard `hipBLASLt out of memory` while the
  server stays healthy.  `GGML_CUDA_SLAB_HEADROOM_MIB=1536` is 3/3 requests clean (arena 19.12 instead of
  19.88 GiB), and `ROCBLAS_USE_HIPBLASLT=0` is the other way out at up to -15 % decode.  **r206 raised the slab
  headroom default to 1280 MiB on one GPU** (block 10), so a fresh install no longer needs the env var here.
  Documented in `ENVIRONMENT.md` §1.3.

**The battery's other verdicts (all as designed, no code change):** pool + `LLAMA_KV_RESIDENT_CELLS` declines
("the windowed KV is already active"); `LLAMA_KV_SPILL_RAM_PCT=1` declines loudly ("no room for the mirror")
while `=0` (no cap) enables; `-np 2` without `--kv-unified` declines (single stream required) and **with**
`--kv-unified` it is eligible and text-identical to the pool-off arm on both a short and a 4 194-token prompt
(0 verify mismatches, two concurrent requests); `--fit on` with `-c 0` sizes and runs; q4_0 and bf16 KV both
work (108 / 385 MiB pools); `-c 8192` pools the whole cache; and a run without MTP is unaffected.  One
pre-existing failure is *not* ours: `-nkvo` with host experts (`-ncmoe 48`) aborts in
`ggml_backend_sched_copy_input` (`GGML_ASSERT(id >= 0 && id < n_expert)`) with the pool off, with no
speculation, and on the released r191 binary too.  Filed as `TODO` #129.

**Goal A's `auto` policy is P1 — the recommendation (maintainer, 2026-10-31, `TODO` #122; the arena keeps
winning the VRAM, and the arena half of Goal A needs a numeric `W`).**  P2 (unconditional arena-first) was
rejected — a fit cannot know the run's depth and a `W` 35k cells below it costs 11-26 % — and P3 (an explicit
per-token spill budget) is parked until measured.  Decision record:
[`../archive/work/kv-residency/A2-MIXED-BACKING.md`](../archive/work/kv-residency/A2-MIXED-BACKING.md) §3.7.14.

**r196 — the attendance pool's mirror grows on demand (`TODO` #123).**  The mirror used to be one
`cudaMallocHost` of the cache's whole cell-indexed size at creation (3264 MiB at `-c 262144` q8_0, 1632 at
131072), so a short run in a big `-c` pinned the whole context in non-swappable RAM.  The buffer type now
reserves one page-aligned `mmap` range and pins 1 MiB units with `cudaHostRegister` as the cell watermark
reaches them (host and device see the same UVA pointer, so every host path still works on `tensor->data`); a
unit is never unmapped, every write grows first (a state restore included), and a read above the pinned extent
is refused loudly.  `grow_regions()` drives it through the same padded `n_kv` the window uses.  Windows keeps
the one-shot pinned allocation.  `LLAMA_KV_POOL_MIRROR_UNIT_MIB` (default 1) is the grow unit.  **Parking on a
pooled cache declines to the server's save-and-clear fallback**: a park packs to the top of the cache, so
growing the mirror to its destination would pin the whole context again (the park gate reports a declined park
as a legitimate PASS and `park == lazyoff` still holds).  Measured: **72 MiB pinned of 3264** at `-c 262144`
and a 5 298-token prompt, **1296 of 1632** at 103k (used extent 1281).  Text is byte-identical to r194 on the
grown 103k arm (`834 263da53898a3`) and 5.3k (`332 dd1981213373`); `LLAMA_KV_POOL_VERIFY=1` 0 mismatches; a
fresh-process slot-file restore (525 MB state) restores 5313 tokens and prefills 4 with the same text;
park/ckpt/state gates PASS; `LLAMA_KV_SPILL_RAM_PCT=1` declines and pins 0; `--fit` still charges the full
mirror to the host bucket (it cannot know the run's depth).

**r200 — the draft window is granted per sequence (issue #264).**  See the per-sequence bullet above.
Reported on `--kv-unified -np 2` (dense 27B, `-c 262144`, `-ctk/-ctv q8_0`): after slot 0 held ~100 k tokens,
every request on slot 1 answered HTTP 500 `decode() failed: failed to process speculative batch` with
`W decode: failed to find a memory slot for batch of size 1024` in the log; `LLAMA_SPEC_DRAFT_WINDOW=0`
passed.  Cause: 66 048 cells = 65 536 (sequence 0's window) + 512, and that 512-cell tail is where sequence
0's own ring writes, so sequence 1's first batch had no contiguous run — `find_slot`'s SWA reclaim frees a
cell by its OWNER's position, so an idle sequence's cells are not reclaimable by another.  The cache is now
132 096 cells for that config (`(65536 + 512) x 2`), still 130 048 cells short of the full `-c` mirror.
The reserve estimate (§r193) now also charges `n_seq_max` streams for a **per-stream** (`--no-kv-unified`)
draft cache — it charged one stream, under-charging by `(n_seq_max - 1) x (win + n_ubatch)` cells
(35B-A3B `-ncmoe 40 -np 2 -c 8192`, window 1024: 3 072 cells reported on both arms now, 1 536 on the
per-stream arm before).
Reproduced and gated here with a forced 8192-cell window (before the fix: HTTP 500 on slot 1, draft cache
8 704 cells) and with a 1 024-cell window in the new gate: 6 requests over 2 slots (both rings wrapping,
prompts re-prefilled by swapping them between slots, then re-used from cache) are **byte-identical to
`LLAMA_SPEC_DRAFT_WINDOW=0` on every step** with identical draft acceptance (17/36, 23/23 per prompt).
`plain == draft-mtp == draft-mtp(W=1024) == draft-mtp(W=0)` at `-c 16384` (876 chars, `sha=8c8fd395402f`).
Concurrent 2-slot decode with the window active also passes (2 x 64 tokens), and `-np 6` at `-c 8192`
**declines** the window (per sequence it cannot hold a batch) and runs both slots with no window.
`validate-set.sh` 16/16 (tree `c6c2f03d`), the re-cut tip is `2ed3997f5`, prefill-logit **0.000000 / 100.000 %**
(build `2ed3997f5`); record `benchmarks/2026-11-01-spec-draft-window-264.md`.

## Dated history

[`../archive/docs/patch-notes-history.md`](../archive/docs/patch-notes-history.md) — every amendment note
from `r1` to `r93`, including the pre-V2.0 block-by-block notes, kept verbatim because `WORKLOG.md` and the
campaign records cite it.
