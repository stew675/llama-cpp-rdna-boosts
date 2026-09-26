# PLAN — per-16 → F16 WMMA, Q6_K proof of concept

Follow this top to bottom.  Every item is a checkbox; **do not skip a gate**.  Measurements go in a
dated `RESULTS-YYYY-MM-DD.md` next to this file (do not edit `MEASUREMENTS.md` — it is the frozen
baseline).

Conventions: work in `~/llama.cpp` (the fork checkout — its `rdna-boosts` branch is disposable, never
push it).  The delivery repo only carries this `wip/` tree.  Every in-tree experiment is
**env-gated OFF by default** until it passes Phase 3.

---

## Design

### Chosen architecture: a second MMQ inner path

The MMQ framework already has the exact tiling Q6_K needs (a 256-value super-block is one
`MMQ_ITER_K` tile), a per-type config table, a launcher, a write-back and a dispatch — all of which
we keep.  We add an **F16 inner path** for Q6_K:

1. **F16 weight loader** — `ggml_cuda_mmq_load_tiles_q6_K_f16`: read the Q6_K super-block from
   global, dequantize to `half2` **with the per-16 sub-scale and the per-256 `d` already folded**,
   and write the `half2` fragment layout into the sram `tile_x`.
2. **F16 activation** — produce an F16 copy of the activation once per GEMM (a cheap elementwise
   cast next to the existing `quantize_mmq_q8_1`) and load `half2` into `tile_y`.
3. **F16 vec_dot** — `ggml_cuda_mmq_vec_dot_q6_K_f16`: `tile<16,16,float> C`,
   `tile<16,8,half2> A/B`, `mma(C,A,B)` (gfx12 `wmma_f32_16x16x16_f16`), then `sum += C.x` — **no
   per-element scale epilogue**.
4. **Outer kernel** — the existing `mul_mat_q` kernel, either templated on the activation type or a
   sibling `mul_mat_q_f16`; the only difference is how `tile_y` is filled and which `vec_dot` runs.

Keep the config geometry (`I=128/256, J=128, nthreads=256`) initially; tune only after it works.

### Rejected alternatives (do not re-litigate without new data)

| alternative | why not |
|---|---|
| Modify `mmf`/`mul_mat_f` to take Q6_K A | its A access is a **K-column gather** (`x[(row)*stride + col]`), which is a scattered read for a 256-block format |
| Improve/extend `mmb` | `mmb_dq_row_q6k` is scalar and dequant-bound; forced Q6_K on RDNA4 measured **639-733 t/s** vs MMQ **981** |
| Repack Q6_K → per-32 affine (Q4_K layout) | mathematically **inexact**: a per-32 `(d,m)` cannot represent two different per-16 slopes |
| Merge the two 16-K sub-blocks into one 32-K int8 mma | the RDNA4 fragment layout interleaves K across warp halves → misassigns scales (attempted: slower *and* wrong) |
| Global Q6_K→F16 shadow in VRAM | 27 B params → ~54 GiB, does not fit |

---

## Phase 0 — Setup  ✅ (2026-09-26)

- [x] branch `archive/work/per16-f16-mma` off `main`
- [x] `archive/work/per16-f16-mma/` with `README.md`, `PLAN.md`, `MEASUREMENTS.md`, `HANDOVER.md`, `tools/`
- [x] baseline frozen in `MEASUREMENTS.md`
- [x] repro scripts in `tools/`

## Phase 1 — In-tree spike: loader + vec_dot + outer kernel, env-gated

Goal: an isolated measurement at `m=17408,n=512,k=5120` that is **correct and > 57 TFLOPS**
(stretch 70), with `test-backend-ops` correctness green.  No dispatch change yet — reach it with a
temporary env gate.

> **Engineer around the gfx12 fragment trap.**  The gfx12 F16 WMMA fragment is "two runs of four"
> (`k = 4*hi+{0..3} ∪ 4*hi+8+{0..3}`) and the C tile is J-major (`m = 8*hi+e`, `n = lane%16`) — a
> hand-rolled contiguous read is silently wrong (cost this session two failed spike attempts, see
> `RESULTS-2026-09-26.md` §2).  **Drive the kernel through `mma.cuh`'s `load_ldmatrix`/`mma`** (or
> copy the `mmb.cu` `mmb_ld_frag`/`MMB_ACC_M` shim verbatim); never invent the fragment layout.
>
> **Corrected ceiling:** F16 prefill is hipBLAS (no in-tree F16 tile kernel for `ncols > 16`), so the
> bar is the int8 MMQ's 57, not 90-107, and `mmb` (fused bf16) currently loses.  A negative Phase-1
> result is a real possibility — time-box it.

### 1a. F16 activation buffer
- [ ] in `ggml_cuda_mul_mat_q` (mmq.cu), allocate/keep an F16 activation tensor (per-context scratch,
      sized like the q8_1 one) and fill it with a cast of `src1` (the existing `src1->type` is
      F32/F16; add a cast kernel or reuse the existing convert path)
- [ ] keep the q8_1 path fully intact and default; the F16 buffer is only produced under the gate
- [ ] confirm the gate can be read on the host (env) and threaded to the launch

### 1b. F16 weight loader
- [ ] add `ggml_cuda_mmq_load_tiles_q6_K_f16` in `mmq-load-tiles.cuh`:
      - dequant math (Q6_K, `QK_K=256`): `value[j] = d * scales[j/16] * (q6[j] - 32)`,
        `q6 = (ql nibble) | (qh 2-bit << 4)`; use `mmb_dq_row_q6k` (`mmb.cu:704`) as the verified
        reference for the bit extraction
      - write the `half2` A-fragment layout consumed by `load_ldmatrix(tile<16,8,half2>)`
        (16 rows × 16 K per fragment; `get_i(0)=tid%16`, `get_j(0)=4*(tid/16)` — see `mma.cuh:844`)
      - hoist `d` and the 16 `scales` into registers/shared once per row
- [ ] unit-check the loader against `dequantize_row_q6_K` on a random block before wiring the GEMM

### 1c. F16 vec_dot
- [ ] add `ggml_cuda_mmq_vec_dot_q6_K_f16` in `mmq-vec-dot.cuh`:
      `tile<16,16,float> C; tile<16,8,half2> A/B; mma(C,A,B); sum[idx] += C.x[l];`
      (K = 16 per mma, accumulator f32 — no scale work)

### 1d. Outer kernel branch
- [ ] add the F16 arm to `ggml_cuda_mmq_get_util_funcs<type,J,fallback>()` for `Q6_K` **behind a
      compile-visible switch**, or add a sibling `mul_mat_q_f16` kernel that shares the body
- [ ] fill `tile_y` from the F16 activation instead of q8_1 when the path is taken
- [ ] **Gate A:** `test-backend-ops perf -o 'MUL_MAT.*' -p 'q6_K.*m=17408,n=512,k=5120'` ≥ **70 TFLOPS**
- [ ] record in `RESULTS-<date>.md` and `MEASUREMENTS.md`-style table

> The self-contained `kernel/q6k_f16_gemm.hip` microbench (Phase-1 fallback) is **deprioritised**:
> it is where the fragment bug bit, and its number is not directly comparable.  Prefer the in-tree
> version.  If the microbench is ever finished, it must use the `mmb_ld_frag`/`MMB_ACC_M` shim
> verbatim and verify a 16×16×16 product against a CPU reference before scaling up.

## Phase 2 — Correctness and dispatch

- [ ] `test-backend-ops test -b ROCm0 -o 'MUL_MAT.*' -p 'q6_K'` green (fp16 tolerance; compare against
      the CPU oracle).  Note: a pre-existing small `MUL_MAT_ID(type_a=q6_K,m=64,n=16,k=768)` case
      already FAILs in the baseline build — record it as pre-existing, do not attribute it here.
- [ ] `test-backend-ops test` over the full Q6_K matrix (all shapes, `o=1/0`, `src_overlap`)
- [ ] add the dispatch arm in `ggml_cuda_mul_mat` (ggml-cuda.cu:2381) / `ggml_cuda_should_use_mmq`:
      route to the F16 path only for `RDNA4 && Q6_K && prefill (n_tokens >= threshold)`
- [ ] env kill-switch `GGML_CUDA_MMQ_Q6K_F16=0` (default ON only after Phase 3; default OFF while
      in development).  Follow the **default-on policy**: when the gates pass, flip it to opt-out.
- [ ] confirm the decode/verify (`n_tokens <= MMVQ_MAX_BATCH_SIZE`) path is untouched (mmvq)
- [ ] confirm no regression for `n_tokens < threshold` (falls back to the int8 kernel)

## Phase 3 — Whole-model validation (the real gate)

- [ ] single GPU: `Q6_K pp512 / pp8192 / pp32768` **≥ 1200 t/s at pp8192**
- [ ] 2 GPU `-sm tensor`: `Q6_K pp8192` **≥ 1800 t/s**
- [ ] **quality**: same-seed greedy text coherent; perplexity (or the delivery's quality proxy)
      within noise of the int8 path.  **F16 accumulation is not bit-identical** — see the numerics
      warning in `README.md`; the maintainer decides if the trade is acceptable.
- [ ] **purity**: `W = 1..8` decode/verify bit-identical with the gate on and off (prefill-only gate)
- [ ] no regression on `Q8_0`/`Q4_K_XL`/F16 (the gate must be Q6_K-only)
- [ ] re-run the other per-16 types' correctness (`Q3_K`, `IQ2_XS`, `IQ2_S`) to prove the int8 path is
      unchanged while the gate is off

## Phase 4 — Per-16 rollout (only after Q6_K passes Phase 3)

- [ ] refactor the F16 loader into a small `dequant_to_half2<type>` so `Q3_K`/`IQ2_XS`/`IQ2_S` share it
- [ ] `Q3_K` (inner kernel `q8_0_16_q8_1_mma`) — same per-16 problem, same expected win
- [ ] `IQ2_XS`, `IQ2_S` — lower priority (rare in practice)
- [ ] keep the gates per type so each can be A/B'd and killed independently

## Phase 5 — Promote or archive

- [ ] if all gates pass: write up the win, ask the maintainer for a delivery-block amendment
      (env-gated, documented in `patches/README.md`, with the quality trade recorded)
- [ ] if any gate fails: move this tree to `archive/work/`, record the negative result, leave `main`
      untouched

---

## Open questions / risks

1. **Numerics** — the biggest one.  F16 accumulation changes prefill logits.  If the maintainer
   requires bit-identity with the int8 path, this campaign is dead on arrival; confirm early.
2. **Shared-memory pressure** — `half2` weights double the `tile_x` footprint vs int8; `sram_stride`
   and the config table may need a new `GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K_F16`.  Check LDS budget for
   `I=128` (and `I=256`) before committing to a config.
3. **Fragment layout** — the Q6_K→half2 loader must write exactly what `load_ldmatrix(tile<16,8,half2>)`
   reads; validate with a tiny standalone kernel before the GEMM (1b unit-check).
4. **The general MMQ ceiling** — the MMQ kernel runs at 31-34 % of the int8 WMMA ceiling with **no
   double-buffering** (`archive/work/q8-prefill-tuning/`).  A software-pipelined MMQ would lift *every* type.
   This campaign is complementary; if a pipeline lands first, re-baseline before comparing.
5. **Activation cast cost** — the F16 activation buffer is O(M·N) elementwise; measure it (it should
   be a few percent, like `quantize_mmq_q8_1`'s 3.7 % in the Q8_0 profile).

## Reference commands

```bash
# isolated GEMM perf
archive/work/per16-f16-mma/tools/repro-gemm-perf.sh gemm q6_K

# whole-model prefill
archive/work/per16-f16-mma/tools/repro-gemm-perf.sh model

# J-tile / config sweep
archive/work/per16-f16-mma/tools/sweep-j.sh
```
