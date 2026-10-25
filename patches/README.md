# The delivery patches — what each block is and what it does

**14 blocks** (block 00 + blocks 01-13) against `release.json.base` (**`3f03b1772`** as of
`v14-3f03b1772-r128`; the base is the 2026-10-24 re-base and r128 is a block amendment at that base), one
`git am` commit each.  This file
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
bash /path/to/rdna-boosts/scripts/apply-all.sh .          # creates branch rdna-boosts, 14 commits
bash /path/to/rdna-boosts/scripts/apply-all.sh --include-beta .   # + beta/*/*.patch (NOT the release tree)
```

All blocks with **`git am`**, in order.  `git apply` on the concatenated series silently drops hunks; the
manual equivalent is `git am patches/000[0-9]-*.patch patches/001[0-3]-*.patch`.  `apply-all.sh` asserts
the applied tree equals `release.json.tree` on the strict path.

## Block index

Sizes are `git diff` line counts against the block's parent (i.e. what that one commit adds), and matter
only as a rough shape.

| # | patch | subject | +/− | files |
|---|---|---|---|---|
| 00 | `0000-…-block-00-foundation-band-uniformity-Meta` | foundation: band uniformity, Meta wrapper, allocator | +1314 / −1000 | 54 |
| 01 | `0001-…-block-01-hybrid-HIP-all-reduce-RDNA4-gat` | hybrid HIP all-reduce | +2365 / −44 | 4 |
| 02 | `0002-…-block-02-adaptive-MTP-draft-depth` | adaptive MTP draft depth | +1186 / −54 | 24 |
| 03 | `0003-…-block-03-chunked-gated-delta-net-prefill` | chunked GDN prefill + rollback tracker | +3182 / −15 | 22 |
| 04 | `0004-…-block-04-BF16-KV-cache-and-native-BF16-f` | BF16 KV cache + native-BF16 FA | +630 / −214 | 8 |
| 05 | `0005-…-block-05-flash-attention-band-native-qua` | FA band / native quantized K-V / derived kq mask (+ r129: the three kq-mask fills that lacked the sibling's buffer guard - `mem_hybrid_k`, `attn_k_dsa` mla/lid - `-np 1` abort on bailingmoe3) | +5050 / −609 | 289 |
| 06 | `0006-…-block-06-MMB-quantized-prefill-GEMM` | MMB quantized prefill GEMM | +2877 / −3 | 7 |
| 07 | `0007-…-block-07-k-quant-boosts-mmvq-VDR-Q6_K-MM` | k-quant boosts (VDR, Q6_K MMQ, Q1_0, grid pad) | +879 / −137 | 9 |
| 08 | `0008-…-block-08-fused-kernels-old-08-fusions-ol` | fused kernels + fused MoE | +5424 / −440 | 35 |
| 09 | `0009-…-block-09-host-resident-MoE-iface-schedul` | host-resident MoE: iface, scheduler, cache, slab hooks (+ r120: the second draft scheduler's slot is per arch, DFlash2 fix) | +14889 / −518 | 65 |
| 10 | `0010-…-block-10-slab-fusion-dispatcher-async-in` | slab + fusion dispatcher + async input ring (infrastructure) | +12058 / −4870 | 75 |
| 11 | `0011-…-block-11-qwen4exp-model-hyper-connection` | qwen4exp model/QSA/HC/lazy reader/indexer cache | +11776 / −433 | 46 |
| 12 | `0012-…-block-12-campaign-remainder-CPU-wins-bac` | campaign remainder (+ r106: Meta `ALL_GATHER`, DFlash2 `-sm tensor`, slab narrow-2; + r111/r112: `--host-experts pool` drops the pinned master; FRAC 80 + O_DIRECT defaults; + r119: batched state restore) | +2939 / −722 | 33 |
| 13 | `0013-…-block-13-MTP-draft-vocabulary-subset-spe` | MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`, opt-in) | +569 / −6 | 19 |

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
plus a per-size hybrid dispatch: small tensors take the internal pipeline, large ones go to RCCL/NCCL.
Four files, ~2.4k lines, all in `allreduce-hip.cu` + `ggml-cuda.cu`.

**Contents.**  Internal host-staged pipeline and butterfly kernels; a wall-clock soft/hard spin budget so a
lagging rank fails soft instead of hanging (issue #139); eager `ncclCommInitAll` (a lazy mid-run init
perturbed the split compute and broke `-sm tensor -ncmoe` MTP purity); a first-runtime-failure fallback
that warns once, disables NCCL and re-routes to the internal path; an opt-in P2P/large path and an opt-in
copy-engine (SDMA) scratch path; and the `AR_PROFILE` device-list initialisation fix.

**Knobs.**  `GGML_CUDA_ALLREDUCE` (`hybrid` default, `nccl`, `internal`, `meta`), `GGML_CUDA_AR_P2P*`,
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
`LLAMA_MTP_DRAFT_OP_OFFLOAD`.

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

**Knobs.**  `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel (bit-identical, slow),
`GGML_CUDA_GDN_CHUNKED_BF16=0` forces the F32 chunked path, `LLAMA_RS_ROLLBACK_DEBUG` (diagnostic),
`GDN_DBG_*` (kernel-debug only).  The sequential-tail knob (`GGML_CUDA_GDN_TAIL`) is read by block 10's
copy of the launch site.

**Purity.**  The pure `none == draft-mtp` range is `n_max <= 7`; `n_rs_batch` is derived from the widest
enabled speculator (`common_speculative_n_max() + 1`, every draft *and* ngram type), so a verify batch can
never take the chunked path (`GREEDY-PURITY.md` §27).  The BF16 chunked path was the issue-#113 prefill-logit
regression: **run `scripts/gate-prefill-logits.sh` on any change here.**

**Gates.**  `test-backend-ops -o GATED_DELTA_NET`, `test-recurrent-state-depth`,
`test-recurrent-state-rollback` (compare the per-arm FAIL/PASS *set*, never the count — it is per box).

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
  noise (`-ncmoe 40`, 32k, q8_0 KV, n_max 7: 248.8/239.6 vs 241.2/249.7 t/s), so no separate instantiation was
  cut; a future one must key on a compile-time band flag, not on `ncols` (`GREEDY-PURITY.md` §51, `TODO.md`
  #99).  Gates: `FLASH_ATTN_EXT` 6391/6391, prefill-logit PASS at the standard point, and the head-256 shape
  PASS in `--family-switch` mode against a same-kernel `-ub` control (candidate 0.005073/97.118 vs control
  0.007251/96.912).  `WORKLOG.md` 261024.007.
* **Native quantized K/V (V4/V5)**: the sub-F16 quant types (`q8_0`/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`iq4_nl`)
  read the cache natively instead of a whole-cache F16 conversion per decode step — a decode-*depth* win,
  and the margin that lets a high-context MTP config load (`GREEDY-PURITY.md` §34).
* **`q4_1`/`q5_0`/`q5_1`/`iq4_nl` as FA cache types** (previously reachable only via
  `GGML_CUDA_FA_ALL_QUANTS`, which also permits mixed K/V), each with its own band sweep and oracle
  (`GREEDY-PURITY.md` §20-22).
* **The derived kq mask (V3)**: per-token bounds + one per-cell position replace the packed mask; the FA
  kernel computes visibility on the fly.  Implemented in **both** the MMA and the tile kernels, and only
  for prefill-shaped batches (`n_tokens > 8`), so decode/verify are untouched.
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
* **Fully-masked KV group skip** (`GGML_CUDA_FA_MASK_SKIP`, prefill only) — exact zeros, bit-identical.
* **Hygiene**: the `(512, 512, 32)` RDNA MMA row retuned to a 4-warp shape (issue #188, Gemma-4 head-512
  prefill), the RDNA2 (gfx1030) gate fixes of PR #181, the masked-V HIP fixes, and the FA staging arena
  (prealloc + graceful degradation instead of abort).

**Knobs.**  `LLAMA_KQ_MASK_DERIVED`, `GGML_CUDA_FA_KV_NATIVE` (auto / force on / force staging),
`GGML_CUDA_FA_MASK_SKIP`, `GGML_HIP_FA_BAND_WMMA*` (`_SPLIT`, `_GQA_MAX`, `_WIDE`),
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

**The three kq-mask fills that lacked the guard** (r129 amendment; reported, root-caused and validated by
@RikshaDriver on RX 7600 `gfx1102`, ROCm 10.1, `bailingmoe3` / Ling-3.0-tiny-IQ4_NL).  The derived kq form
leaves the packed mask with no consumer at one slot and the allocator gives unconsumed tensors no buffer;
ten of the thirteen fill sites in `src/llama-graph.cpp` already guard for that, in the form
[`attn-k-null-mask-guard`](../upstream/attn-k-null-mask-guard/README.md) describes, and three did not:
`llm_graph_input_attn_k_dsa` (its `mla` and `lid` masks) and `llm_graph_input_mem_hybrid_k`.  The hybrid-K
class is the one a `bailingmoe3` graph reaches, and only single-sequence contexts enable the derived form
(`multi_stream = n_seqs > 1 && (!kv_unified || arch == DEEPSEEK4)`) - so `-np 1` aborted at
`GGML_ASSERT(buffer)` (ggml-backend.cpp:571, via `set_input_kq_mask`'s host-residency assert) before the
first token, at every ctx size tried, while `-np 2` and `-np 4` ran untouched.  The three sites now carry
the guard their siblings have; `can_reuse` needs nothing (`can_reuse_kq_mask()` reads `ne[]`, never
`buffer`).  Measured, card idle, `-st -n 128`, 2 reps: `-np 1` abort -> 88.7 / 88.6 t/s; `-np 2` 88.1 /
85.9 and `-np 4` 87.9 unchanged, the derived form being off at 2+ slots in both builds; greedy output is
token-identical to `LLAMA_KQ_MASK_DERIVED=0` and to CPU (md5 of the 144-char `-n 32 -s 42 --temp 0`
completion `5ea74e63b02f` on all three), so nothing reads the skipped fill.  Without the
guard, `LLAMA_KQ_MASK_DERIVED=0` is the work-around and is measured free at this shape (88.5 / 88.8).

**Upstream candidates.**  `fa-kv-split-width` (block 00), `fa-decode-verify-kernel-family`,
`fa-512-rdna-head-cap`, `attn-k-null-mask-guard` — see [`../upstream/`](../upstream/README.md).

---

## Block 06 — MMB and mmf32: the quantized and the F32 prefill GEMM

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
measured with.

**Knobs.**  `GGML_CUDA_MMB` (master switch), `_TYPES`, `_DENSE`, `_ROUTED`/`_ROUTED_THRESH`, `_GLU`/`_GLU_THRESH`,
`_F32SPLIT`(+`_MIN_K`/`_MIN_M`), `_TALL`(+`_MIN_M`), `_TINY_M`/`_TINY_TT`, `_DOWN16`, `_HC16`, `_IQ3XXS`,
`_RDNA3`, `_CFG`, `_CACHE`/`_SHADOW*`, `_MIN_T`, `_TILE`, `_BF16W`, `_LOG`; `LLAMA_MMB_CVT_LOG`;
`GGML_CUDA_MMF32` (kill switch, `=0`), `GGML_CUDA_MMF32_LOG` (diagnostic).  Everything is default-on and
documented in `ENVIRONMENT.md` §§10/10.1.

**Purity.**  Both kernels change prefill arithmetic, not decode: run the **prefill-logit gate** on any tuning
change, and the width probe when a fusion/dispatch predicate moves.  For `mmf32` the gate is the
*`--family-switch`* mode at a confident point (`prompts/kld-corpus-35b-in-domain.txt`), with the same-kernel
control as the arbiter — see `GREEDY-PURITY.md` §47.

---

## Block 07 — k-quant boosts: `mmvq` VDR, Q6_K MMQ, Q1_0, grid pad

**What it is.**  The decode-numerics block.  `VDR` (vectors-per-thread reduction) for `Q4_K`/`Q5_K`/`Q8_0`
(2→4) and `Q6_K` (1→2), the RDNA3_5 `mmvq` parameter table split out of the RDNA2 fallback, a Q6_K MMQ
prefill path, the Q1_0 `__byte_perm` → `v_perm_b32` port, and a grid-stall pad.

**Contents.**  `vecdotq.cuh` (VDR constants + the `ggml_cuda_byte_perm` helper), `mmvf.cu` (the Q6_K
decode path), `mmvq.cu` (the VDR-aware K-split loop and the RDNA3_5 table), `mmq-vec-dot.cuh` /
`mmq-load-tiles.cuh` / `mmq.cu` (Q6_K MMQ), `common.cuh` (grid pad), and the Q1_0/Q6_K cases in
`tests/test-backend-ops.cpp`.

**Knobs.**  `GGML_CUDA_GRID_STALL_PAD`, `GGML_CUDA_MMQ_J_MAX`, `GGML_CUDA_OP_TIMING` (diagnostic).

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
* `mmq.cuh` / `mmq.cu`: the fused MoE gate+up+GLU MMQ prefill, the routed-compact dispatch, `mul_mat_q_pair`,
  the FP4/gate precision contract, and the Q6_K compact-J path.
* `moe-weighted-reduction.cu`: the MoE weighted-down reduction (+ column-blocked band epilogue).
* `norm.cu` / `quantize.cu` / `unary.cu`: `rms_norm`→Q8_1 folds, the verify-band q8_1 cache, scale-unary,
  norm-rows, the conv1d/PLE fusions and the shared-expert down gate.
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

**Knobs.**  `GGML_CUDA_DISABLE_FUSION` (master), `_DISABLE_MOE_MMQ_FUSION`, `_DISABLE_MMQ_ROUTED`,
`_DISABLE_TOPK_MOE_FUSION`, `_DISABLE_CONV_FUSION`, `_DISABLE_NORM_ROWS`, `GGML_CUDA_TOPK_SORTED`,
`GGML_CUDA_FUSE_GDN_GATE`, `GGML_CUDA_FUSE_Q8_1_VERIFY`, `GGML_CUDA_MMVQ_RDNA3_SMALL_M`,
`GGML_CUDA_SCALE_UNARY`, `GGML_PAIR_2X`, `GGML_CUDA_Q8_1_ARENA_FREE_OLD`,
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

**What it is.**  Everything that makes `-ncmoe`/`-cmoe` (and `--host-experts`) fast and safe without
fitting the whole model on the device: a pinned host master, a per-op H2D staging ring, a scheduler that
can hand an op to a host-resident table, and the MoE expert cache that keeps the hottest experts in
whatever VRAM is left.

**Contents.**

* `moe-expert-cache.cu` (~5.1k lines) + `.h`: the cache engine — arena sizing, per-layer uniform slots,
  the slot remap, the routed-MMVQ decode/verify band (`moe_cache_band`), admission/heat policy, the
  persistent cross-run heat map (`MOE_EXPERT_HEATMAP`) and prefill cache staging.
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
  speculator to `band - 1` after the model load (the preflight's decision is known there), lowers `n_min`
  with it, and refreshes `cparams.n_rs_batch` so the slab's narrow verify floor and the GDN rollback bound
  see the capped depth (the uncapped default also cost ~5 GiB of arena: 32938 -> 27814 MiB).  Clamped vs
  unclamped, 2x R9700 Flash-Next IQ4_NL `-sm tensor -ncmoe 36`, `prompts/recall.txt`, `-n 2000`: 168.3 vs
  73.9 t/s (`draft-mtp-adaptive,ngram-mod`), 160.2 vs 74.1 (`draft-mtp,ngram-mod`); the cliff is one row
  wide (159.2 t/s at a 16-row verify, 55.0 at 17).  **Not** applied to a device-resident model or the CPU
  expert path (band 0): there the wide draft is the win (378.7 vs 144.7 t/s).  `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0`
  restores the uncapped draft; a `draft.n_max` above the band warns instead of clamping (the 15 bound is a
  correctness one).  **Known gating gap (TODO #96):** the clamp is applied in `common_init_from_params`
  before `llama_init_from_model`, i.e. before the slab is created or declines at the first compute-buffer
  allocation, so a config whose slab is armed and then declines (a small-VRAM card) is clamped even though
  it streams with no arena; `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` restores the wide draft there.  The slab
  decline message names `GGML_CUDA_SLAB_RESERVE_MIB`, but on such a card the binding terms are usually
  `GGML_CUDA_SLAB_MIN_ARENA_MIB` (the 2048 MiB floor) and `GGML_CUDA_SLAB_RING_MIB` (the H2D-ring hole).
  `GREEDY-PURITY.md` §50, `ENVIRONMENT.md` §3, `WORKLOG.md` 261024.002 / 261024.006.
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
  Record: `../archive/work/graph-key-churn/`.
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

**Knobs.**  `MOE_EXPERT_CACHE_MIB` (auto default; `0` disables), `_MIN_MIB`, `_MIN_RES_PCT`,
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

---

## Block 10 — the slab, the fusion dispatcher, the async input ring (infrastructure)

**What it is.**  The infrastructure half of the pre-r109 block 10: it owns the translation units split out
of `ggml-cuda.cu` (the movable-boundary **slab**, its arena, and the `ggml_cuda_try_fuse` dispatcher), the
async graph-input ring, and the GDN/MMB/rope decode fusions.  The **qwen4exp architecture it used to carry
is block 11** (r109 split; the tree is unchanged by the split).  Rule of thumb: a qwen4exp-only piece that
is implemented inside a general kernel TU (the GDN gate/beta fusion, the HC16 MMB BF16 streams, the
weighted-down reduction, the PLE conv fusion) stays here with the TU that owns it; a piece that needs this
block's own symbols (QSA/HC matchers, the indexer relu-sum, the HC_BLK16/RES16 marking pass) is block 11.

**Contents.**
* **The slab** (`ggml-cuda-slab.cu`, `ggml-cuda-arena.cu`): one VMM reserve per device split by a movable
  boundary — compute below, MoE arena above, no HIP call at runtime.  Plus the narrow-band learning, the
  wide-phase auxiliary region, pool-wide routing, and the slab transient absorption.
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

## Block 11 — qwen4exp: model, hyper-connections, lazy reader, QSA/indexer

**What it is.**  The qwen4exp (Qwen3.8-Flash-Next) architecture support, split out of block 10 in r109 so
the model graph, the sparse-attention op and the hyper-connections no longer share a commit with the slab,
the fusion dispatcher and the async input ring.  QSA moves with the model by the maintainer's decision
(it was added along with qwen4exp).  **The infrastructure it builds on is block 10** — this block adds no
slab/dispatcher/ring code of its own beyond the arms implemented in its own TUs.

**Contents.**

* **qwen4exp** (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`): QSA sparse attention and the
  fused indexer top-k/score (`indexer-topk.cu`, `lightning-indexer`), hyper-connections (`hc-mix.cu`,
  `hyperconn.cu`), the managed lazy reader, PLE n-gram loading, the MTP draft head, and the **keys-only
  indexer cache** (`LLAMA_QSA_KEYS_ONLY`; no V tensor at all — `llama_kv_cache` gains `v_enabled`).
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
`LLAMA_LAZY_READER_STATS`, `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100` / `_GFX1201`.

**Purity.**  qwen4exp's decode/verify band is `HC_FUSED_MAX_TOKENS = 8` and the QSA decode arm band is
`max(8, n_rs_batch)`; the arm must be chosen from the band and `n_kv`, never `n_tokens == 1`
(`GREEDY-PURITY.md` §§13, 16, 26, 32, 38).  The QSA op has a CPU oracle — run
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

## Block 12 — campaign remainder

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

**Knobs.**  `LLAMA_OUTPUT_PINNED`, `MTP_DRAFT_N_UBATCH`, `GGML_LF_DFLASH_DEV`, `GGML_LF_FAST_TOPK`,
`LLAMA_STATE_GET_MERGE_GAP`, `LLAMA_STATE_ASYNC`, `LLAMA_STATE_ASYNC_MAX_MIB`,
`GGML_CPU_MOE_OFFLOAD_THREADS`, `GGML_CUDA_SLAB_RING_MIB`, and (r111+, POOL mode) `MOE_HOST_POOL_MIB` /
`MOE_HOST_POOL_FRAC` / `MOE_HOST_POOL_DIO` / `MOE_HOST_POOL_VERIFY` / `LLAMA_HOST_MASTER_PINNED`, plus
(r113) `MOE_HOST_POOL_RAM_PCT` (default **85** since r114) and (r114) `MOE_HOST_POOL_STAGE_BATCH`;
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

## Block 13 — MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`, opt-in)

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

## Dated history

[`../archive/docs/patch-notes-history.md`](../archive/docs/patch-notes-history.md) — every amendment note
from `r1` to `r93`, including the pre-V2.0 block-by-block notes, kept verbatim because `WORKLOG.md` and the
campaign records cite it.
