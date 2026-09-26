# bf16-native-prefill — WIP (ACTIVE)

**Status: ACTIVE (opened 2026-09-18).  Not part of the delivery.**  Everything
here is experimental until promoted (see the AGENTS.md promotion rule).

> **New session?  Start with [`HANDOVER.md`](HANDOVER.md)** — the turnkey brief
> (mandate, environment, established facts, the task, and a copy-paste prompt).
> This README is the project record; the dated sections below are the log.

> **CURRENT STEP (2026-09-18): step 2 — cheapen the bf16→f16 conversion in the MMA
> tile loader, keeping f16 compute — CLOSED as a negative result.**  The conversion is
> already the RN minimum (`v_cvt_f16_f32` ×2); the only packed f32→f16 on gfx12 is
> round-toward-zero (`v_cvt_pkrtz_f16_f32`), and an RN-exact `v_pack_b32_f16` form is no
> faster.  None of the three candidate loaders reaches parity: the cost is per converted
> word (exposed because the AMD MMA loader is synchronous — no `cp_async`, `nstages = 0`),
> not per op.  The native *read* itself is ~4 % faster than the staged dense read.  See
> "Step 2 findings" below.  V5 stays opt-in.
>
> Step 1 (full native bf16 MMA = "no F16 at
> all") is **CLOSED as a negative result**: it is implemented and correct but
> *slower* (321.7 ms vs the staged 232.9 on the 35B FA kernel), because gfx1201 has
> no packed bf16 arithmetic and no bf16→f16 pack, so removing the conversion costs
> more than it saves.  See "Route 2 implemented", "Attempt 2" and "What it would
> actually take" below, and `patches/route2-native-bf16-mma.patch` (kept for
> re-testing on gfx950/CDNA4).  Step 2's floor and target are in the table in
> "What it would actually take".

## Goal (maintainer's "dream")

**Native bf16 K/V must reach prefill parity with the F16-staging path** — and,
if at all possible, beat it.  The delivery's V5 arm (`GGML_CUDA_FA_KV_NATIVE=1`)
already reads bf16 natively in the MMA kernel and removes the F16 staging
scratch, so a bf16 cache *costs what an f16 cache costs* in memory and output.
But it is **~1-2 % slower at prefill** than bf16-with-staging, which is why it
ships opt-in.  That penalty is the open problem: the repo's "native BF16
support" arc is not at parity yet.

The staging *type* (convert into F16 vs BF16) is a separate question and is not
the fix — see "Root cause".

## Current state (measured)

`ggml/src/ggml-cuda/fattn-mma-f16.cuh`: the native bf16 loader
(`flash_attn_ext_f16_load_tile_bf16`) converts each 16-byte staged chunk bf16→f16
in registers; the F16 path copy/stages from a launcher-made F16 copy.

27B Q8_0 (qwen35, head 256, 24 q / **4 kv** heads, gqa 6), 2×R9700 gfx1201,
`-ctk bf16 -ctv bf16 -fa 1 -sm layer`, `GGML_CUDA_FA_KV_NATIVE=1`,
`GGML_CUDA_FA_STAGE_MAX_MB=1` = force native prefill, unset = staged (512 MiB
cap).  `llama-bench`, r=2:

| test | staged | native | Δ |
|---|---|---|---|
| pp512 | 1442.6 | 1419.4 | −1.6 % |
| pp2048 | 2219.0 | 2196.7 | −1.0 % |
| pp8192 | 2490.1 | 2463.5 | −1.1 % |
| tg128 | 18.43 | 18.45 | +0.1 % |

The historical V5 A/B (block-15 campaign) shows the same curve growing with
prompt length: 4B pp2048 −0.22 %, pp8192 +0.27 %, pp20480 −1.06 %,
pp40960 −2.36 %; 27B pp20480 −0.76 %; decode ±0.1 %.
`archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` §"Cost".

## Root cause (recorded, confirmed by the campaign) — **SUPERSEDED, see "Step 2 findings"**

> Correction (2026-09-18, fifth pass): this section's "the conversion is free" is wrong — the raw
> read *is* fine/better, but the in-loader bf16→f16 conversion costs ~63 ms of the FA kernel.  See
> the second-pass findings below and the "Step 2 findings" section at the end.

* **The conversion is free** — native bf16 staging measures within **0.17 %** of
  an *f16* cache (both scratch-free).  So bf16→f16 conversion and the bf16
  source are not the cost.
* **The F16 staging pass is really a de-interleave.**  `launch_fattn()` runs
  `to_fp16` over the whole cache *and rewrites the strides to a dense,
  normalised layout* (`nb11 = ne[0]*2`, `nb12 = ne[1]*nb11`, …).
* **The raw WMMA cache is interleaved across the GQA heads.**  For the 27B a row
  is `256×2 = 512 B` but `nb[1]` (token stride) is `2048 B = 4 rows`, because the
  4 K/V heads sit between consecutive tokens.  For a fixed head, consecutive
  tokens are 4 rows apart.
* The tile loader re-reads that view on **every K/V staging pass**, so the
  strided access is paid per query tile and the penalty grows with the prompt
  length.
* `cp_async` is **not** the cause on RDNA4: `cp_async_available()` is
  NVIDIA-only (`common.cuh:373`), so both paths use the synchronous loader.

So: the native path skips a cheap conversion but reads memory in a worse
pattern in the hot tile-staging loop; the access-pattern cost slightly exceeds
the conversion it saves.  Staging *into bf16 instead of f16* would not help —
staging's value is the dense layout, not the conversion.

## Hypotheses to test (in order)

1. **The GQA interleaving is the cause, and it scales with the GQA ratio.**
   Test: measure the penalty on models with different `n_head_kv` / `nb[1]` (a
   `n_head_kv == 1` model has `nb[1] == ne[0]*2`, i.e. already dense → the
   native arm should be free).  If the penalty tracks `gqa_ratio`, confirmed.
2. **The native loader's read order is suboptimal for the interleaved view.**
   Test: profile (`rocprofv3`) achieved DRAM/L2 bandwidth and cache hit rate for
   native vs staged; the native kernel should show lower memory-pipe utilization
   or more L2 misses.
3. **Residual: something else in the loader** (register pressure, address math,
   the `el_off` path, or the conversion ALU).  Test: `GGML_CUDA_FA_KV_NATIVE`
   with an **f16** cache (native-vs-native baseline, should be 0), and a
   native-bf16-vs-f16-cache comparison (should be ≤0.2 %).

## Fix candidates

| # | idea | rough effort | notes |
|---|---|---|---|
| A | **Restrict the native arm to already-dense layouts** (`nb[1] == ne[0]*2`, single-KV-head models) | small | makes the arm free where it applies, but does not help GQA models (the common case) |
| B | **Read the native staging densely** — reorder the tile loader so the interleaved view is traversed coalesced (e.g. stage whole per-token rows, or process all K/V heads of a token together) | medium | the interesting one: aims at parity without touching the cache layout |
| C | **Make the KV cache non-interleaved** (head-major: `[n_head_kv][n_kv][dim]`) | large | removes the root cause for every backend/kernel, but touches the cache update, the split/quant paths and every FA kernel.  Likely upstream scope. |
| D | **Overlap the de-interleave with the tensor copy** (keep a dense staging copy but make it a pure bf16→bf16 de-interleave, no conversion) | ? | does not remove the staging pass or the scratch, so it cannot reach the native memory win; only interesting if the dense copy is genuinely cheaper than the native strided read |

B is the target: keep the native read (no scratch, no pass) and make the access
pattern match the dense copy.  If B is not achievable, C is the only route to
true parity, and A is a partial (single-KV-head) win.

## Findings (2026-09-18, second pass — roasted profile, root cause CORRECTED)

`rocprofv3` per-kernel profile, Qwen3.6-35B-A3B-Q8_0, `-sm layer`, 2×R9700, `-fa 1`,
pp8192, `GGML_CUDA_FA_KV_NATIVE=1`; staged = 512 MiB cap, native =
`GGML_CUDA_FA_STAGE_MAX_MB=1`.  Raw traces in `profiles/`.

**The recorded root cause (the GQA de-interleave) is WRONG, and the fix candidates
built on it (B — reorder the loader; C — head-major cache) are aimed at the wrong
3 %.**  The three arms, total time in the MMA FA kernel
(`flash_attn_ext_f16<256,256,8,8,false,false,false>`, 320 dispatches):

| arm | K/V read | loader | FA kernel | Δ vs staged |
|---|---|---|---|---|
| bf16 staged | dense F16 copy (de-interleaved by `to_fp16_nc`) | F16 copy | 213.2 ms | — |
| **f16 cache** | raw, **interleaved** | F16 copy (no conversion) | **219.3 ms** | **+2.9 %** |
| bf16 native | raw, interleaved | converting bf16 loader | 276.6 ms | +29.8 % |

* **Layout (dense vs interleaved) = ~3 %** (213.2 -> 219.3, same F16 loader).
* **The in-register bf16→f16 conversion = ~26 %** (219.3 -> 276.6).  It is re-paid on
  **every K/V tile re-read**, whereas the launcher's staging pass converts each element
  once (the whole-run conversion is only 19.5 ms over 640 dispatches, vs +63 ms added to
  the FA kernel).
* End-to-end at pp8192 the FA kernel is a small fraction of a MoE model's work, so the
  whole-run penalty stays ~1.2 % even though the kernel is ~30 % slower.

The real structural gap: **decode (TILE kernel, block 03) computes in native bf16; prefill
(the MMA kernel) is F16-only.**  V5 is a native *read* with an f16 *compute* — i.e. no
F16 removal at all.  See HANDOVER §"Direction" for the two routes.

`profiles/`: `staged/` (f16 arm), `native/` (bf16 native), and the bf16-staged arm
(re-runnable; its aggregate is the 213.2 ms row above).  `tools/agg-kernels.py` and
`tools/fa-stats.py` do the aggregation.

**Tooling note:** `rocprofv3` on this box aborts in its rocpd/SQLite writer
(`ROCPD_STATUS_ERROR_SQL_SCHEMA_INVALID_VERSION`) and then hangs in its signal handler.
Always pass `--output-format csv` to bypass that path (`tools/profile-fa.sh` does).

## Findings (2026-09-18, first measurement pass)

`llama-bench`, `-sm layer`, 2×R9700, `-fa 1`, `GGML_CUDA_FA_KV_NATIVE=1`;
staged = default (512 MiB cap), native = `GGML_CUDA_FA_STAGE_MAX_MB=1`.

**Qwen3.5-4B (n_head_kv=4, head 256, gqa 4), r=3:**

| test | bf16 staged | bf16 native | f16 native | native vs staged | bf16nat vs f16nat |
|---|---|---|---|---|---|
| pp2048 | 11707.6 | 11687.4 | 11638.5 | −0.17 % | +0.42 % |
| pp8192 | 12978.4 | 12916.9 | 12932.0 | −0.47 % | −0.12 % |
| pp20480 | 11859.3 | 11892.6 | 11971.8 | +0.28 % | −0.66 % |
| tg128 | 86.60 | 86.52 | 86.33 | −0.09 % | +0.22 % |

**Qwen3.6-35B-A3B MoE (n_head_kv=2, head 256, gqa 8), r=3:**

| test | bf16 staged | bf16 native | native vs staged |
|---|---|---|---|
| pp2048 | 6861.7 | 6813.5 | −0.70 % |
| pp8192 | 7644.3 | 7542.1 | −1.34 % |
| tg128 | 81.62 | 81.39 | −0.28 % |

### What this says (and does not)

* The penalty is **small (0.2–1.4 %)** and **noisy** at r=3; it needs an
  interleaved, higher-rep protocol to resolve.
* **The simple `n_head_kv` scaling hypothesis is NOT confirmed.**  The 35B-A3B
  has *half* the interleaving factor of the 4B (2 vs 4 rows between consecutive
  tokens of one head) yet shows a *larger* penalty.  The 27B (n_head_kv=4)
  measured ~1 % earlier.  So the interleave factor alone does not predict it.
* The likely confound: **the 4B is compute-bound** (~12-13k t/s pp) while the
  35B-A3B is memory-bound (~7.6k t/s), so a per-read access-pattern penalty is
  masked on the 4B and visible on the large model.  A clean test needs two
  models of similar size/boundness with different `n_head_kv`, which we do not
  have locally.
* `bf16 native ≈ f16 native` (within ~0.66 % on the 4B) — consistent with the
  campaign's "conversion is free (0.17 %)" claim.  The gap is therefore
  **staged-dense vs native-interleaved**, and an f16 cache is stuck on the
  native side too (f16 is the staging type, so it can never be staged).

## Next experiments (do these before designing B)

1. **Profile, don't guess.** `rocprofv3` (or `rocprof`) the FA kernel for staged
   vs native on the 35B-A3B at pp8192: compare kernel time, memory-pipe busy,
   L2 hit rate and DRAM bytes.  The 4B/35B delta inversion has to be explained
   before any loader rewrite.
2. **A controlled microbenchmark**: one model, but vary the KV layout directly
   (or emulate the interleave) so `n_head_kv`/stride is the only variable; run
   memory-bound (large model or large batch) to make the penalty visible.
3. **Higher-rep, interleaved A/B** (`-r 10`, alternate arms within one build) to
   get the error bars under ~0.2 %; the current r=3 spreads overlap.
4. Reproduce the campaign's exact `pp20480` numbers on the current r3 tree to
   check the historical curve still holds (the 4B pp20480 sign differs: +0.28 %
   now vs −1.06 % in the plan).
5. Only then pick a fix: A (dense-layout gate) is a quick partial; B (dense
   native read) needs the profile to say *which* access is costly; C (head-major
   cache) is the upstream-scope fallback.

## Validation protocol (when a fix exists)

Same five parts as the V5 plan (`BF16-NATIVE-KV-PLAN.md` §4):

1. Reserve matrix (ctx 204800): 4B (1 GPU), 27B (3-GPU Meta), gemma-4-E4B
   (1 GPU, ISWA), gemma-4-31B (3-GPU, ISWA), qwen4exp (3-GPU control) × ub
   2048/1024/512 × gate off/on, with f16 as the reference (bf16 == f16 every
   row).
2. Coherence: bf16 vs f16 vs gate-off, byte-identical generated text, short +
   40k prompt.
3. Adaptive-MTP gate (`benchmarks/mtp-adaptive-methodology.md`): 27B inline
   draft-mtp and qwen4exp draft probes; acceptance unchanged (27B 0.76744,
   qwen4exp 0.44262).
4. Op suites: `test-backend-ops -o FLASH_ATTN_EXT` (ROCm0 + CPU), both gate
   states; whole-suite backstop.
5. Throughput, interleaved same-binary A/B (llama-bench `-fa on`):
   `-ctk/-ctv bf16` gate off vs on, pp20480/ub 2048 + tg256, 4B and 27B, plus
   the f16 reference.

Plus the **negative control** (`GGML_CUDA_FA_KV_NATIVE=0` reproduces today) and
the **greedy-purity** invariant (`--spec-type none == draft-mtp` within the
`n_max <= 7` band) if any kernel arithmetic changes.

## Tools / commands

```bash
cd ~/llama.cpp   # delivery tree + this WIP on top (TBD: the WIP branch)
B=./build-rocm/bin/llama-bench
M27=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf

# staged (default) vs native prefill for bf16:
HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_FA_KV_NATIVE=1 \
  $B -m $M27 -ngl 99 -sm layer -ctk bf16 -ctv bf16 -fa 1 -p 512,2048,8192 -n 128 -r 3
HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_FA_KV_NATIVE=1 GGML_CUDA_FA_STAGE_MAX_MB=1 \
  $B -m $M27 -ngl 99 -sm layer -ctk bf16 -ctv bf16 -fa 1 -p 512,2048,8192 -n 128 -r 3
```

`GGML_CUDA_FA_STAGE_MAX_MB=1` forces the staging cap to fail → native prefill;
it only affects the prefill width (`native_width`), so decode is unchanged.

## References

- `archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` — the full V5
  plan (mechanism with call sites, design, ship rule, validation protocol).
- `archive/work/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md` — V4, the
  template V5 mirrors.
- `patches/README.md` — the V4/V5 block-15 notes and the `GGML_CUDA_FA_KV_NATIVE`
  policy (unset = auto: q8_0/q4_0 on, bf16 off).
- `TODO.md` — the parked bf16 follow-ups (options A/B/C above).

## Session log

- **2026-09-18 (open).**  Project created from the issue-#38 / tensor-fit
  investigation (the staging-arena blind spot led here).  Root cause and the
  measured penalty recorded.
- **2026-09-18 (first pass).**  Confirmed the cache layout in code
  (`llama-kv-cache.cpp:234`: `ggml_new_tensor_3d(..., n_embd_k_gqa, kv_size,
  n_stream)`, `n_embd_k_gqa = head_dim * n_head_kv` → `[token][head][dim]`,
  interleave factor = `n_head_kv`).  Measured 4B (n_head_kv=4) and 35B-A3B
  (n_head_kv=2) staged vs native vs f16; **the `n_head_kv` scaling hypothesis
  failed** (the 2-head model is worse than the 4-head one) — the effect is
  small, noisy, and confounded by model size/boundness.  Next: profile
  (`rocprofv3`) and a controlled microbenchmark before touching the loader.
  Local model spread available: E4B (kv 2, head 512, ISWA), Qwen3.5-4B/9B
  (kv 4), Qwen3.6/3.8-27B (kv 4), 35B-A3B (kv 2, gqa 8), Gemma4-31B (per-layer
  kv 16/4).

## Route 2 implemented — first result (2026-09-18, third pass)

The native bf16 MMA arm is **implemented and working** (env gate `GGML_CUDA_FA_BF16_MMA=1`,
temporary).  Design/plan: [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md).

**What was built** (all in `~/llama.cpp`, uncommitted local WIP):

* `mma.cuh`: bf16 low/high pack helpers, generic-tile `load_ldmatrix_trans` `I==32` branch,
  `get_bfloat162` (P conversion), the `nv_bfloat162` I_MAJOR_SCRAMBLED C tile + `unscramble`, and
  two bf16 `mma()` overloads — `tile<16,8,bf16>` bf16-accumulate (VKQ) and the SCRAMBLED
  `tile<16,16,bf16> x tile<32,8,bf16>` split (KQ uses the pre-existing f32-accumulate overload).
* `fattn-mma-f16.cuh`: `mma_tile_sizes_kv<DV,ncols,use_bf16>`, `bool use_bf16` on the kernel and
  `process_tile` (the `iter` body **derives** the element type from its tile types), `kv_t` smem
  tiles, bf16 Q-fill / P-conversion / V-tile zeroing / VKQ rescale / output combine, and a
  `kv_bf16` loader flag that turns the `FATTN_KV_NATIVE_BF16` arm into a plain 16-byte **byte copy**
  (bf16 and f16 are byte-identical, so the old F16 loader serves both — no conversion at all).
* `fattn-swizzle.cuh`: the `load_ldmatrix`/`_trans` helpers are now element-type-generic.
* `fattn-common.cuh`: `GGML_CUDA_FA_BF16_MMA` gate (implies the bf16 native read, so the launcher,
  the predicates and `get_alloc_size` agree) + a `force_native_kv` arg to `launch_fattn` so the
  bf16 kernel never stages.
* `fattn-mma-f16.cuh` case fn: runtime selection (see the trap below).

**Trap found (worth remembering):** `RDNA4` / `AMD_WMMA_AVAILABLE` are **device-pass-only** macros
(`__GFX12__` is not defined in the hipcc *host* pass).  A host-side `#if defined(RDNA4)` around the
kernel selection compiled the whole branch out, the bf16 kernel was never instantiated, and the arm
silently never fired.  The arch test must be the **runtime** `GGML_CUDA_CC_IS_RDNA4(cc)`; the
element type must be derived from the *tiles* (not the flag) so a non-RDNA4 device falls back to the
f16 tiles consistently.

**Correctness:** `test-backend-ops -o FLASH_ATTN_EXT` with the gate ON is **5952/5952** against the
CPU oracle, and a short greedy run is coherent.  (Note: the bf16 arm deliberately computes at bf16
precision, so it is *not* bit-identical to the f16 reference — that is the accepted route-2 trade.)

**Performance — the arm works but is SLOWER on gfx1201 (do not ship as is).**

`rocprofv3`, Qwen3.6-35B-A3B pp8192, 2×R9700, FA-kernel totals:

| arm | FA kernel | convert_unary | notes |
|---|---|---|---|
| f16 cache (raw, no conversion) | 219.3 ms | 0 | the ceiling to beat |
| bf16 staged (f16 kernel, dense F16 copy) | 213.2 ms | 19.7 ms | today's default |
| bf16 + V5 (`GGML_CUDA_FA_KV_NATIVE=1`) | 276.6 ms | 0.5 ms | today's opt-in |
| **bf16 + route 2 (`GGML_CUDA_FA_BF16_MMA=1`)** | **321.7 ms** | **0** | fires (`use_bf16=true`), no staging |

So removing the conversion is real (convert_unary 19.7 -> 0, F16 scratch gone) but the kernel got
*47 % slower* than the f16-cache arm — the bf16 WMMA path is a net loss as written.

**Why (diagnosed, not yet fixed).**  The WMMA instructions are *not* the cause: a warmed-up
microbench (`tools/wmma-bench`-style) shows `wmma_f32_16x16x16_bf16` = 0.99x the f16 equivalent and
`wmma_bf16_16x16x16_bf16` = 1.00x, and the two kernels have identical VGPR/SGPR (256/128) and
workgroup/grid shape.  The cost is the **element-wise glue**, where f16 has packed native ops and
bf16 does not:
* the VKQ accumulator rescale `VKQ_C[i].x[l] *= KQ_max_scale` — f16 = one `v_pk_mul_f16`;
  `__hip_bfloat162::operator*` converts both operands to f32, multiplies and packs back
  (~10 ops per element pair), and it runs over the whole accumulator on every k-tile iteration;
* the P conversion (`get_bfloat162`, scalar `__float2bfloat16` x2) vs one packed
  `__float22half2_rn`; the output combine (`kv_to_float2` bf16 = 2 scalar converts) vs one
  `__half22float2`.

**Next step (the promising fix):** stop doing arithmetic in bf16.  Switch `T_C_VKQ` to **f32**
(the MFMA path already does this, and it matches the TILE kernel's f32 accumulation): the VKQ mma
becomes the f32-accumulate `wmma_f32_16x16x16_bf16` (measured equal-rate to f16), the rescale
becomes a scalar f32 multiply, there is no `unscramble`, and the P/output glue becomes f32.  That
removes every per-element bf16 conversion from the inner loops.  The output-combine path stores the
f32 accumulator into `tile_Q` as 16-bit values, so that branch needs a bf16-aware variant (it
currently casts to `half*`).  Until that is done and measured, route 2 is **not** a win and V5
correctly stays opt-in.

### Attempt 2: f32 VKQ accumulator — also worse (2026-09-18, fourth pass)

Hypothesis: the cost was the bf16 *element-wise* glue (the VKQ rescale using
`__hip_bfloat162::operator*` instead of one `v_pk_mul_f16`).  Fix tried: set the bf16 arm's
`T_A_VKQ = tile<16,8,bf16>`, `T_C_VKQ = tile<16,16,float>` (the MFMA shape) so the VKQ mma is the
equal-rate f32-accumulate `wmma_f32_16x16x16_bf16`, the rescale is a scalar f32 multiply, there is
no `unscramble`, and the P/output glue is f32; the output-combine store got a bf16-aware 2-byte
variant.

**It is slower still** — and the metadata says why: an f32 VKQ accumulator spans twice the DV
width per tile, so `VKQ_C` doubles (DV/16 x 8 floats = 128 VGPRs) and the kernel, already pinned at
the 256-VGPR ceiling, spills much harder.

| arm | FA kernel | Scratch (spill) | VGPR |
|---|---|---|---|
| f16 raw cache (ceiling) | 219.3 ms | 520 B | 256 |
| bf16 staged | 213.2 ms | 520 B | 256 |
| bf16 + V5 (convert loader) | 276.6 ms | 520 B | 256 |
| bf16 full, bf16-accumulate VKQ | 321.7 ms | 372 B | 256 |
| bf16 full, f32-accumulate VKQ | 389.4 ms | 948 B | 256 |

**Decomposition (FA kernel, 35B-A3B pp8192):** the native read itself is fine (the raw f16 read with
the same interleaved stride and the same F16 loader is 219.3 ms), the in-register conversion costs
~57 ms (219.3 -> 276.6, the original V5 finding), and the **bf16 compute path costs a further
~45 ms versus the f16 compute path** (276.6 -> 321.7).  So the element type swap *costs more than
the conversion it removes* (~102 ms of bf16 compute penalty vs the 19.7 ms launcher conversion + the
~57 ms in-kernel conversion).

**Conclusion (route 2 does not reach parity on gfx1201):** the bf16 WMMA instructions are the same
rate as f16 (measured, warmed clocks: 0.99x and 1.00x) and occupancy is identical, yet a kernel
whose operands and smem tiles are bf16 is consistently and substantially slower than the f16 one.
The likely reasons are the absence of packed bf16 arithmetic (`v_pk_mul_f16` /
`__float22half2_rn` / `__half22float2` have no bf16 equivalents used here, so the glue goes through
f32) plus the compiler's handling of `__hip_bfloat162`, against a kernel that is already at the
256-VGPR ceiling in every variant.  Full bf16 accumulation also *doubles* the VKQ register
footprint, and the f32-accumulator variant measurably worsens the spills.

**Recommendation:** keep V5 opt-in as today; do **not** ship the full bf16 MMA arm.  If the ~1 %
bf16 prefill delta ever needs to be recovered, the route is *not* "remove the F16"; it is either a
GPU with packed bf16 element-wise ops, or an f16 *staging type* question (the de-interleave without
the type change), both out of scope here.  This is a well-evidenced negative result: the remaining
f16 is not what makes bf16 slow.

## What it would actually take (2026-09-18, ISA verification)

**The gfx1201 ISA is the limit — verified with the assembler, not assumed.**  Each candidate
instruction was compiled with `--cuda-device-only -c` for gfx1201 (a `-S` text emission proves
nothing; the assembler is what rejects):

| instruction | on gfx1201 | meaning |
|---|---|---|
| `v_pk_mul_bf16` | **not supported on this GPU** | packed bf16 multiply is CDNA4/gfx950-only |
| `v_pk_add_bf16` | **not supported on this GPU** | packed bf16 add likewise |
| `v_cvt_pk_bf16_f32` | **not supported on this GPU** | f32->bf16 packing is gfx950-only |
| `v_cvt_pk_f16_bf16` | **invalid instruction** | there is no direct bf16->f16 pack anywhere in the ISA |
| `v_cvt_f32_bf16` | not supported (under that name) | bf16->f32 is the integer shift below |
| `v_pk_mul_f16` | **OK** | f16 keeps the full packed arithmetic set |
| `v_lshlrev_b32` / `v_and_b32` | OK | bf16->f32 is `<<16` / `&0xffff0000` |
| `v_dot2_f32_bf16` | OK | the one packed-ish bf16 op RDNA has (f32 accumulate) |

And `__float22half2_rn(float2)` on gfx1201 lowers to **SALU**: `s_cvt_f16_f32` x2 +
`s_pack_ll_b32_b16` (i.e. f32->f16 is a scalar-side convert, quite unlike the old VOP3
`v_cvt_pk_f16_f32`, which does not exist under that name on gfx12).

So the "no F16 at all" route is **not** a software-tuning gap on gfx1201: with no packed bf16
arithmetic and no bf16->f16 pack, every element-wise bf16 operation must round-trip through f32;
the f32-accumulator workaround then doubles the VKQ register footprint on a kernel that is already
pinned at the 256-VGPR ceiling (scratch 520 -> 948 B) and loses more than it saves.  The matrix
engine is *not* the problem -- bf16 WMMA is equal-rate to f16 (0.99x / 1.00x, warmed clocks).

**Documented next steps (ordered by expected value):**

1. **Retest on hardware with packed bf16.**  CDNA4 / gfx950 has `v_pk_mul_bf16`,
   `v_pk_add_bf16` and `v_cvt_pk_bf16_f32`.  On such a part the bf16 glue becomes as cheap as f16's
   *and* the conversion disappears, which is exactly the combination that should make native bf16
   win.  Keep the arm (env-gated OFF) so it can be re-measured on future RDNA/CDNA silicon instead
   of being re-derived; the per-instruction check above is the go/no-go test.
2. **The realistic route to parity *today* is not "remove F16" but "make the bf16->f16 conversion
   in the V5 loader cheap"** -- i.e. attack the ~57 ms, not the ~102 ms.  bf16->f16 is *exact* for
   in-range values, and gfx12 can do it as `v_lshlrev_b32`/`v_and_b32` (bf16->f32) plus
   `s_cvt_f16_f32` + `s_pack_ll_b32_b16`, i.e. a short SALU sequence that runs on the scalar pipe
   rather than competing with the vector/memory pipe.  That keeps the f16 compute (so output stays
   bit-identical and every existing validation gate still applies) and keeps the no-scratch win.
   Ceiling: V5's FA kernel 276.6 ms -> ~230 ms, i.e. parity with the staged path's 232.9 ms
   (213.2 + 19.7) *while* removing the F16 scratch.  **This, not the bf16 MMA, is the recommended
   follow-up** -- it is the only measured path to the original goal (memory win at prefill parity)
   on gfx1201.

## Step 2 findings (2026-09-18, fifth pass) — CLOSED as a negative result

**Mandate (step 2):** make the V5 bf16-native arm reach prefill parity with the F16-staging path by
making the in-loader bf16->f16 conversion cheap, keeping f16 compute.  **Outcome: the conversion
cannot be cheapened in the loader on gfx1201.**  Three alternative conversion sequences were
implemented and measured in the real MMA kernel; none reaches parity.  V5 stays opt-in.

### The ISA reality (assembler-verified, `llvm-mc -triple=amdgcn -mcpu=gfx1201`, not `-S`)

| mnemonics | gfx1201 | note |
|---|---|---|
| `v_cvt_f16_f32` (incl. `.l`/`.h` dst) | OK | the RN convert; `__float22half2_rn` lowers to two of these |
| `v_cvt_pk_f16_f32` | **not supported** | the packed **RN** f32->f16 pack does not exist on gfx12 |
| `v_cvt_pkrtz_f16_f32` | OK | the only packed f32->f16, **round-toward-zero** |
| `v_pack_b32_f16` | OK | plain register pack (no conversion) |
| `v_perm_b32`, `v_alignbit_b32`, `v_lshlrev_b32`, `v_and_b32` | OK | the bf16->f32 unpack is `<<16` / `&0xffff0000` |

The compiler already emits the **RN minimum** for `__float22half2_rn(ggml_cuda_cast<float2>(x))`:
`v_lshlrev_b32` + `v_and_b32` + `v_cvt_f16_f32 v.l` + `v_cvt_f16_f32 v.h` -> **4 ops per 2 elements**.
The SALU suggestion in the earlier "next step 2" is impossible: the tile data is per-lane, so scalar
(SALU) converts cannot be used.  Every candidate is VALU.

### The three candidate sequences, measured in the real kernel (35B-A3B pp8192, 2xR9700, `-sm layer`)

FA-kernel total `flash_attn_ext_f16<256,256,8,8,...>`, 320 dispatches.  `native` =
`GGML_CUDA_FA_STAGE_MAX_MB=1` (bf16 read, no staging); `staged` = the same binary with staging
allowed (the reference).  Same-build pairs are marked together.

| loader arm | ops / 2 elems | native FA | staged FA | verdict |
|---|---:|---:|---:|---|
| **raw bit copy** (bitcast, no convert; floor) | 0 | **198.1 / 198.9 ms** | 206.3 ms | the native *read* is ~4 % **faster** than the dense staged read |
| RTZ packed (`v_cvt_pkrtz_f16_f32`) | 3 | 270.9 ms | 221.3 ms | still ~50 ms of conversion; wrong values anyway |
| PACK (`2x v_cvt_f16_f32` full-reg + `v_pack_b32_f16`) | 4 | 276.1 ms | 220.7 ms | **RN-exact** (0/1M mismatches) but no faster |
| RN baseline (compiler `.l`/`.h`) | 4 | 276.6 / **276.9 ms** | 219.3 / 214.1 ms | reference |

Per-run FA-kernel stats for every profile (old + step 2) are in
[`profiles/STEP2-FA-KERNEL-STATS.csv`](profiles/STEP2-FA-KERNEL-STATS.csv); the raw traces are the
`profiles/<label>/` directories.  The staged control drifts 206-221 ms across sessions (clocks/
thermals), so compare the native arm against a same-session staged run, not across sessions.

* **No sequence helps.**  Going from 4 ops to 3 saves ~5-6 ms of ~63; the RN-exact PACK variant is
  indistinguishable from the baseline.  The cost is **per converted word**, not per op: any
  conversion creates an exposed `load -> unpack -> convert -> store` chain (~2 ALU levels) that the
  loader cannot hide.  This is the key correction to the step-2 premise ("~15 ms of conversion would
  reach parity"): the conversion is ~63 ms and it is not op-selectable.
* **The read is not the problem** (the raw-copy arm is faster than the staged dense read), which
  also retires fixes A/B/C: the native read pattern is already good.
* **Root cause of the exposure (new, 2026-09-18):** `cp_async_available()` is **NVIDIA-only**, so
  `ggml_cuda_fattn_mma_get_nstages()` is **0 on RDNA4** — the MMA kernel has *no* multi-stage
  pipelining on AMD.  The K/V loader is synchronous, so the conversion ALU/latency sits in the
  critical path next to the MMA instead of overlapping it.  The kernel is at the 256-VGPR ceiling in
  every variant (scratch 504-536 B), so there is no register budget for deeper prefetch/double
  buffering.  Combined with the loader re-staging each K/V element per query tile (measured ~3.5x
  the cache size in conversions), the native path does ~3.5x the conversion work of the one-shot
  launcher pass (~20 ms), i.e. ~63 ms.
* The pure-launcher alternative (staging) is the only way to pay the conversion once, and that is
  exactly what V5 removes — so the trade is **memory (staging scratch) vs ~3.5x conversion ALU**, and
  on gfx1201 the ALU side cannot be won in the loader.

### Conclusion

**V5 bf16 native stays opt-in.**  The conversion is the RN minimum already; the packed alternatives
are either RTZ (not bit-exact) or no faster.  Reaching parity would require overlapping the conversion
with the MMA — i.e. pipelining the AMD loader (there is no `cp_async`, and the kernel has no VGPR
headroom for software double-buffering).  That is a kernel-structure change, not the loader-only task
this step was scoped to, and it would need its own A/B.  On gfx950/CDNA4 the picture may differ
(packed bf16 arithmetic exists there) — see the step-1 patch for that route.

### Correctness / validation (all re-run on the candidate builds)

* `test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT` **5952/5952** with the PACK loader (CPU oracle).
* The PACK conversion is bit-exact vs `__float22half2_rn` over 2^20 mixed bf16 words (0 mismatches)
  and exhaustive over the 2^16 bf16 values in the low half.
* 4B, 5246-token prompt (`prompts/prose-rdna-boosts.txt`), greedy 200 tokens, bf16 K/V:
  staged vs native text **byte-identical** (`ae0ca6cd1b53`), so the loader is value-transparent.
* End-to-end `llama-bench` (35B, pp512/2048/8192, r=5, interleaved): native is -0.3/-0.6/-0.8 % vs
  staged — the FA-kernel conversion cost (~60 ms) only partly offset by the saved launcher pass
  (~20 ms), diluted by the rest of the model.

### Tooling lesson (2026-09-18)

`tools/profile-fa.sh` forced the native read only when the label was **exactly** `native`; labels
like `pack_native` silently measured the STAGED arm and produced a false "parity" result that cost a
full build+profile round.  Fixed: a `*native*` glob now forces `GGML_CUDA_FA_STAGE_MAX_MB=1`.  Always
cross-check a `native` profile against a same-session `staged` control before believing a delta.

### Reproducing

```bash
# per-variant (edit GGML_CUDA_FA_BF16_CVT in fattn-mma-f16.cuh: 1=RN, 2=RTZ, 3=raw copy, 4=PACK),
# the candidate code lives only in this WIP's session history, not in the delivery:
cd ~/llama.cpp && cmake --build build-rocm --target ggml-hip llama-bench -- -j16   # ~4 min
M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
bash ~/llama-cpp-rdna-boosts/archive/work/bf16-native-prefill/tools/profile-fa.sh rn_native_true   "$M" 8192 0,1
bash ~/llama-cpp-rdna-boosts/archive/work/bf16-native-prefill/tools/profile-fa.sh staged_ctl      "$M" 8192 0,1
python3 ~/llama-cpp-rdna-boosts/archive/work/bf16-native-prefill/tools/fa-stats.py \
        ~/llama-cpp-rdna-boosts/archive/work/bf16-native-prefill/profiles/rn_native_true/prof_kernel_trace.csv 'flash_attn_ext_f16<'
```
