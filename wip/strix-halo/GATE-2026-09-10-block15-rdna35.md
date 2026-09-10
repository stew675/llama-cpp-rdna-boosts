# RDNA3_5 (gfx1151) validation of the 15-block delivery — raw record (2026-09-10)

> Raw session record.  The delivery-facing summary/folds live in
> `patches/README.md` (block-15 notes, "RDNA3_5 (gfx1151) validation"
> subsection), `MANIFESTS.md`, `WORKLOG.md`, `beta/block-15-campaign-wins/`.
> This file is the complete matrix; it is NOT part of the delivery set.

## Environment / state

| item | value |
|---|---|
| machine | Strix Halo APU, Radeon 8060S (gfx1151), 1 device, 123 GiB unified |
| ROCm | 7.14.0, `/opt/rocm-7.14-gfx1151` (toolchain + runtime) |
| Vulkan | RADV / Mesa 26.1.8, device `AMD Radeon 8060S Graphics (RADV STRIX_HALO)` |
| applied tree | `~/llama.cpp` `rdna-boosts` on `9113cc188`, 15 block commits, applied with `git am` (strict, `scripts/apply-all.sh`) |
| applied tip (before pass) | `83d81dc23` "rdna-boosts: block 15: campaign memory wins" (contains V5 `FATTN_KV_NATIVE_BF16`) |
| amended canonical tip | `377f8e790` (2026-09-10 RDNA3_5 amendment, `patches/0015` only; `0001`-`0014` byte-identical) |
| build (HIP) | Release, `AMDGPU_TARGETS=gfx1151`, `GGML_HIP=ON`, `GGML_HIP_RCCL=1`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_NO_VMM=ON`, `GGML_NATIVE=1`, `GGML_CUDA_FA_ALL_QUANTS=OFF` |
| `llama-cli -v` | `VMM: no`, `Wave Size: 32`, 1 device |
| build (Vulkan) | `build-vulkan` rebuilt; `flash_attn{,_cm1}.comp` touched to force shader regen (mtime-based generator gotcha) |
| probes | `fattn-probe-{rocm,vulkan}`, `fattn-diag-{rocm,vulkan}` rebuilt against the fresh libs, `GGML_PREC_F32` as llama.cpp uses |

STEP A markers all present in the applied tree: `FATTN_KV_NATIVE_BF16`
(`fattn-mma-f16.cuh`) 1, `ggml_cuda_fattn_dequantize_q8_0_chunk`
(`fattn-common.cuh`) 2, `kq_mask_derived` (`src/llama-cparams.h`) 2,
`signed-zero leak` (`fattn-mma-f16.cuh`) 1, `J_max_gate` (`mmq.cuh`) 3,
`RDNA4 (gfx1200/gfx1201)` (`allreduce-hip.cu`) 1, `no consumers`
(`ggml-alloc.c`) 1; `zero_freed` (`src/llama-kv-cache.cpp`) 0, `fattn-dbg`
0, `vk-fa-dbg` 0.

## 1. The two gfx1151 regressions found (and fixed) — V3 derived kq mask

Both are block-15 / V3 issues, host-side and architecture-independent in
cause; they only surfaced here because this is the first single-device
iGPU/`--parallel` run.

1. **V3 was silently disabled on the HIP iGPU.**  The derived-mask probe
   verifies `ggml_backend_dev_implements_kq_derived()`, whose switch routed
   only `GGML_BACKEND_DEVICE_TYPE_GPU` (and `META`) to the CUDA-family
   check; an APU reports `GGML_BACKEND_DEVICE_TYPE_IGPU`, so the probe
   logged `derived kq mask flash attention is assigned to ROCm0, but it is
   only implemented by the CUDA/HIP backend` and set V3 disabled.  The whole
   ~800 MiB compute + ~800 MiB host win was lost on this box.
2. **`n_seq_max > 1` aborted context creation.**  `llama-server --parallel 4
   --cache-type-k f16` died at startup:
   `/home/stew675/llama.cpp/ggml/src/ggml.c:5601: GGML_ASSERT(tok_lo->ne[0]
   == a->src[0]->ne[1]) failed` in `ggml_flash_attn_ext_add_kq_derived`,
   during the V3 probe graph.  Instrumentation showed the derived input was
   built with `n_tokens=64 n_seqs=1 n_seqs_unq=1 n_stream=1` while Q was
   reshaped `[256,32,16,2]`: `build_attn_mha` derives the attention stream
   count from the KV tensor (`k->ne[3]` == the cache's `n_stream` ==
   `n_seq_max`), not from `ubatch.n_seqs_unq`.  `kq_mask_derivable()` only
   checked `ubatch.n_seqs_unq == 1`.

**Fix** (amendment to `patches/0015`):

* `src/llama-context.cpp`: `ggml_backend_dev_is_cuda()` accepts
  `GGML_BACKEND_DEVICE_TYPE_IGPU` as well as `..._GPU` (the ROCm/CUDA reg
  name is still required); `ggml_backend_dev_implements_kq_derived()` wires
  `IGPU` to it.
* `src/llama-kv-cache.cpp`: `kq_mask_derivable()` rejects `n_stream != 1`.

Result on gfx1151: V3 enables (`derived kq mask flash attention enabled`),
and a multi-slot context keeps the packed mask (no abort).  Discrete GPUs
report `..._GPU` and normally run `n_stream == 1`, so RDNA4 behavior is
unchanged.

## 2. Determinism gate (16 identical greedy requests, cache_prompt, per-pos top-8 logprobs)

`run-gate.sh` + `gate16.py` unchanged, 4B Q8_0 (hsk 256, 16/4 heads).
`--parallel 1` (the `--parallel 4` that the harness ships aborted before the
fix; after the fix it passes).  `--cache-type-*` last-flag-wins.

| config (ROCm) | V3 on | V3 off |
|---|---|---|
| f16, arm off | PASS 16/16 | PASS 16/16 |
| bf16, arm off | PASS 16/16 | PASS 16/16 |
| bf16, `GGML_CUDA_FA_KV_NATIVE=1` (V5) | PASS 16/16 | PASS 16/16 |
| q8_0, arm on (V4) | PASS 16/16 | PASS 16/16 |
| q4_0, arm off | PASS 16/16 | PASS 16/16 |
| q4_0, arm on | PASS 16/16 | PASS 16/16 |
| bf16, `-fa off` (control) | PASS 16/16 | PASS 16/16 |
| q8_0, arm off (extra) | PASS 16/16 | — |
| `--parallel 4`, f16 / bf16+V5 (post-fix) | PASS 16/16 | — |

Vulkan (RADV), `--parallel 1`: f16 / bf16 / bf16+V5 / q8_0+V4 / q4_0 off /
q4_0 on / bf16 `-fa off` → **7/7 PASS**.

**Cross-arm byte-identity** (over 16 runs × 129 tokens = 2064 cells each):

| pair | result |
|---|---|
| V3 on vs off (f16, bf16, bf16+V5, q8_0+V4, q4_0 off, q4_0 on, bf16 fa-off) | **IDENTICAL** (0 diffs) |
| bf16 arm on (V5) vs bf16 arm off | **IDENTICAL** |
| q8_0 arm on (V4) vs q8_0 arm off | **IDENTICAL** |
| q4_0 arm on vs q4_0 arm off | **IDENTICAL** |
| bf16 (either arm) vs f16 | 16/16 DIFFERENT — expected: the bf16 cache stores lower precision; the arm does not change the math |

## 3. Isolated masked-column probes (`FPROBE_KGARB=1 FPROBE_VGARB=1`)

Matrix: nq ∈ {1,2,4,32,128}, kv 1280, n_valid ∈ {1025,1063,1100,1152},
hsk 256 nh 16 nh_kv 4; plus hsk 128/256 MHA and hsk 128 GQA; each config at
`GGML_CUDA_FA_KV_NATIVE=0` and `1`.  Diag = staggered causal, garbage in one
partially-masked column (live V).

| backend / type | masked-column configs | diag (live-V partial column) |
|---|---|---|
| ROCm bf16 | **34/34 OK** (both arms) | 2 LEAK-ROW0: hsk256 lmax 2.842e-14, hsk128 lmax 1.137e-13 — `det=0`, **arm-independent**, bf16-only (f16 clean); the documented out-of-scope live-cell artifact |
| ROCm f16 | **36/36 OK** (both arms) | 4/4 OK |
| Vulkan bf16 | **36/36 OK** | 4/4 OK |
| Vulkan f16 | **36/36 OK** | 4/4 OK |

Every previously-leaking gfx1151 geometry (HIP MMA f16/bf16, HIP bf16 TILE,
Vulkan cm1) is invariant to garbage in the fully-masked tail with the arm
OFF and ON — the block-14 fixes are effective and V5/V4 do not reintroduce a
masked-cell leak.

## 4. Arm engagement (a no-op run would prove nothing)

Compute buffer at ctx 8192/ub 2048, 4B bf16 KV: arm off 1322.02 MiB → arm
on 320.02 MiB (the F16 staging scratch disappears).  Reserve matrix below
confirms the same at ctx 204800.

V3 engages on gfx1151 after the fix: `derived kq mask flash attention
enabled`; ctx 8192/ub 2048 f16 compute 288.06 (V3 off) → 256.11 (V3 on),
host 72.07 → 40.12 (exactly `n_kv × n_ubatch × 2 B`).

Kernel per model on gfx1151 (WMMA capped at head ≤ 320 by block 04): head
256/128 prefills run the MMA kernel where V3/V4/V5 apply; head 512/576 falls
back to TILE and does not; Flash-Next uses the fused QSA path where V4/V5
are no-ops.

## 5. Reserves (compute buffer / host compute buffer, ctx 204800)

4B Q8_0, ub 2048 (f16/bf16/q8_0 × V3 × arm):

| KV | V3 | arm | compute | host |
|---|---|---|---|---|
| f16 | 1 | 0 | 256.86 | 40.87 |
| f16 | 1 | 1 | 256.86 | 40.87 |
| f16 | 0 | 0 | 1056.06 | 840.07 |
| bf16 | 1 | 0 | 968.86 | 40.87 |
| bf16 | 1 | 1 | **256.86** | 40.87 |
| bf16 | 0 | 0 | 1768.06 | 840.07 |
| q8_0 | 1 | 0 | 1001.13 | 41.13 |
| q8_0 | 1 | 1 | **257.13** | 41.13 |
| q8_0 | 0 | 0 | 1800.33 | 840.34 |

4B ub 512: f16 V3-on 64.80; bf16 arm off 842.80 → arm on **64.80**; q8_0 arm
off 851.07 → arm on **65.07**; f16 V3-off 264.02 / 210.02.

MoE Qwen3.6-35B-A3B True-Q3_K_M, ub 2048: f16 400.99; bf16 576.99 →
**400.99**; q8_0 609.26 → **401.26**; V3-off f16 1200.20 / 832.07.

27B Q8_0, ub 2048: f16 488.86; bf16 1072.86 → **488.86**; q8_0 1121.13 →
**489.13**; V3-off f16 1288.06 / 880.07, q8_0 1920.33 / 880.34.

**Every number reproduces the RDNA4 block-15 references exactly.**  Deltas:
V3 −799.20 compute / −799.21 host (scaling exactly `n_kv × n_ubatch × 2 B`);
V5 (bf16) −712.00 on the 4B, −584 on the 27B; V4 (q8_0) −744.00 on the 4B,
−632 on the 27B.  bf16 + arm-on reserves exactly what f16 reserves.

Flash-Next IQ4_XS, q8_0, ub 2048: all W on **3251.39 / 63.69**, indexer KV
**318.76** (regular KV 2550.00); all W off 6690.40 / 1262.70, indexer 956.26.
Deltas: W1+W2+W3 = −3439.01 compute, −1199.01 host, −637.50 indexer.  These
are the RDNA4 absolute values too.

## 6. Coherence / MTP

* V3 on/off same-seed greedy byte-identical: gate pairs above (7 configs ×
  2064 cells).  bf16 vs f16 differs by cache precision, not by arm.
* Adaptive MTP (`--spec-type draft-mtp`, temp 0, seed 42):
  * 27B Q8_0 q8_0 KV: V3 on 0.79762 (67/84) == V3 off 0.79762; acc per pos
    (0.929, 0.750, 0.714) both.
  * 27B Q8_0 bf16 KV: arm 0 0.79762 == arm 1 0.79762.
  * Flash-Next IQ4_XS + Q4_K_M draft, q8_0 KV: V3 on 0.52727 (58/110) == V3
    off 0.52727.  (Absolute acceptance is prompt-dependent; the gate is
    on == off, and the qwen4exp fused-QSA path is a V3/V4/V5 no-op.)

## 7. Ops

* `test-backend-ops -o FLASH_ATTN_EXT -b ROCm0`: **4596/4596** V3 on and
  off.  The GPU list is smaller than CPU's 7859 because this build has
  `GGML_CUDA_FA_ALL_QUANTS=OFF` (only FA-supported types; CPU is the
  host-side full list).  Derived cases: 5 OK + 1 `not supported`
  (`kv=4090`, not a multiple of the 256 derived stride — correctly
  declined), all green in both arm states.
* `test-backend-ops -o FLASH_ATTN_EXT -b CPU`: **7859/7859** V3 on and off
  (matches the RDNA4 host-side list; 6 derived cases).
* `test-alloc`: 4/4 pass.  `test-batch-alloc`: 198 assertions, 0 failures.
* W4 repro (`wip/arch-independent-memory/repro/ggml-alloc-unused-view.c`):
  arena **16.00 MiB** (fixed; 56.00 without).

## 8. Perf (interleaved same-binary llama-bench A/B, 4B, ub 2048, V3 on)

V5 (bf16, arm off → on), 2 passes:

| test | pass1 | pass2 | Δ |
|---|---|---|---|
| pp2048 | 2552.7 → 2540.7 | 2536.2 → 2534.3 | −0.4 % / −0.1 % |
| pp8192 | 2341.0 → 2319.4 | 2314.1 → 2316.4 | −0.9 % / +0.1 % |
| pp20480 | 1974.8 → 1959.2 | 1970.1 → 1956.1 | −0.8 % / −0.7 % |
| tg128 | 42.68 → 42.66 | 42.67 → 42.68 | ~0 |
| tg256 | 42.72 → 42.72 | 42.73 → 42.73 | ~0 |

V4 (q8_0, arm off → on), 2 passes:

| test | pass1 | pass2 | Δ |
|---|---|---|---|
| pp2048 | 2523.8 → 2513.4 | 2518.2 → 2513.7 | −0.4 % / −0.2 % |
| pp8192 | 2312.7 → 2279.5 | 2299.6 → 2288.1 | −1.4 % / −0.5 % |
| pp20480 | 1963.1 → **2013.8** | 1960.0 → **2013.6** | **+2.6 % / +2.7 %** |
| tg128 | 42.31 → 42.34 | 42.33 → 42.33 | ~0 |
| tg256 | 42.37 → 42.39 | 42.38 → 42.39 | ~0 |

**Arm-cost answer:** on gfx1151 the arm is *closer to free* than on RDNA4
and at long prefill is a win: V5 −0.4…−0.9 % prefill (RDNA4 −0.2…−2.4 %),
V4 **+2.6 % at pp20480** (RDNA4 −1.7 %).  Decode is within 0.1 % in every
case.  The native staging re-reads the interleaved cache view, and on this
iGPU's large MALL that cost is absorbed (indeed q8_0 wins at depth where the
RDNA4 F16 scratch conversion dominated).  It returns the same system RAM as
on RDNA4 (V5 712 MiB / V4 744 MiB on the 4B at this context), because the
scratch is host/unified memory here.

V3 on vs off (q8_0, arm off), 2 passes: pp2048 2518.6/2515.0 vs
2523.5/2526.7 (−0.2 % / −0.46 %); pp20480 1941.7/1940.6 vs 2005.0/2002.5
(**−3.2 % / −3.1 %**); tg128 42.31 vs 42.34/42.31 (~0).  V3's prefill cost
is larger on gfx1151 than the RDNA4 4B reference (−1.3 %), decode the same;
it buys −799 MiB compute + −799 MiB host.

Block-13 fused MoE re-check (Q3_K_M, q8_0, ub 2048, pp512 warmup first per
the 2026-09-05 protocol, `-t 16 -r 8`): fused on vs off = pp2048
1865.4/1853.4 vs 1858.5/1854.2 (+0.4 %/~0 %), pp16384 1614.2/1607.5 vs
1610.4/1606.7 (+0.2 %/~0 %), pp512 +1.2…+1.9 %.  The isolated
`GGML_CUDA_DISABLE_MOE_MMQ_FUSION` delta is now ~0 on this box (absolute
prefill is ~10–13 % *higher* than the 2026-09-05 record: 1855/1610 vs
1674/1423), consistent with the 2026-09-06 model-neutral Strix folds
(mul_mat_q_pair / swiglu-input quantize) capturing the same work when the
gate+up+GLU arm is off.  No regression: fusion fires, coherence holds,
decode untouched.

## 9. Amendment / regeneration

* `patches/0015` amended only: `GGML_BACKEND_DEVICE_TYPE_IGPU` (4 hits) and
  `n_stream != 1` (2 hits) present; canonical tip `377f8e790`.
* Regenerated via `git format-patch 9113cc188..377f8e790`; the `0001`-`0014`
  bodies are byte-identical to the pre-amendment delivery (only their
  `From <sha>` lines differ, as always between rebuilds — those files were
  not touched).
* Clean-apply sim: fresh worktree at `9113cc188` + `scripts/apply-all.sh`
  → "All 15 patches applied cleanly (strict git am)", zero whitespace
  warnings, applied tree `6f5d23b5` == the amended canonical tree.
* Gates re-run on the amended tree (the runs above are all on the fixed
  build): determinism matrix, probes, reserves, ops, MTP, perf.

## 10. Qualification

| gate / kernel | engaged on gfx1151? | notes |
|---|---|---|
| V3 derived kq mask (MMA) | **yes, after the amendment** (was silently off) | head ≤ 320 MMA; 5 derived op cases OK; −799 MiB; ~−3 % pp20480 |
| V4 native q8_0 (MMA + TILE) | yes when `GGML_CUDA_FA_KV_NATIVE=1` | −744 MiB; +2.6 % pp20480 |
| V5 native bf16 (MMA) | yes when `GGML_CUDA_FA_KV_NATIVE=1` | −712 MiB; −0.4…−0.9 % prefill |
| block-14 masked-V fixes | yes (HIP TILE bf16, HIP MMA f16/bf16; Vulkan cm1/scalar) | 34–36/36 probes each backend/type |
| W1/W2/W3 QSA | yes (Flash-Next) | exact RDNA4 reserve/indexer numbers |
| W4 ggml-alloc | yes | repro 16.00 MiB |
| block-13 fused MoE | fires; isolated delta ~0 on this build | see §8 |
| block-12 internal AR | n/a (single device; RDNA4-gated) | RCCL fallback, no AR |
