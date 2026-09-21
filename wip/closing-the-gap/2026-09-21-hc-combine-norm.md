# Phase-1 item 1 — HC combine+norm fusion: root cause and first result (2026-09-21)

**Status:** WIP prototype. Fork branch `gap-closing` @ `121ad7935` (`~/llama.cpp`), built on gfx1151.
Default is **unchanged** — the revived matcher is inert unless `LLAMA_FUSED_DSV4_HC_POST=0`.

## The doc's premise was wrong in an interesting way

`closing-the-gap.md` §9.1/§10 item 1 said "the delivery's `hc_combine_norm` matcher never fires; debug
the matcher". That is true, but the diagnosis was not: the matcher is **not** bypassed by the
`DSV4_HC_POST` op (they are alternative fusions of the same chain — the plain path is only built when
`fused_dsv4_hc_post` is off). Even with `DSV4_HC_POST` disabled, the matcher still did not fire, and the
graph fell back to a fully unfused chain: a `k_bin_bcast op_repeat` materialization (192 calls) + a
separate `rms_norm_f32<1024>` (196 calls).

## Root cause (three bugs)

With `LLAMA_FUSED_DSV4_HC_POST=0` and the new `LLAMA_HC_CN_DEBUG=1` trace:

1. **`ggml_can_fuse_subgraph_ext` rejected the window.** `build_hc_combine` emitted the scatter-weight
   VIEW (`w = reshape_3d(scale2, 1, hc, nt)`) *after* the `block_out` REPEAT, because `mul` expands
   `src[0]` (the repeat) before `src[1]` (the w view). The fusion window
   `[repeat, mul, add, rms, mul(gamma)]` therefore contained an interleaved `RESHAPE` whose `view_src`
   (scale2) is outside the window, and the fuse check rejects external view sources:
   `canf: i=1 RESHAPE external view_src node_74`.
2. **The matcher's `ok_f` shape test was pre-rebase.** It required the gamma-MUL result to be the 2-D
   `[n_embd*hc, n_tok]`; the current qwen4exp graph produces the 3-D `[n_embd, hc, n_tok]` (the
   `reshape_2d` to `[hc_dim, nt]` happens *after* the gamma mul, for `build_lora_mm`).
3. **The kernel dispatch asserted the 2-D shape.** `ggml_cuda_op_hc_combine_norm` has
   `GGML_ASSERT(a.out_xn->ne[0] == n_embd * hc)`; handing it the 3-D `mul` aborted. The kernel's writes
   are `out_xn[row*n_embd + col]`, `row = t*hc + c`, which is exactly the same row-major layout for both
   shapes, so the assert only needed the element count.

## Fix (fork `gap-closing` @ `121ad7935`)

* `src/models/qwen4exp.cpp::build_hc_combine` — emit the `w` VIEW before the `block_out` REPEAT.
* `ggml/src/ggml-cuda/ggml-cuda.cu` — `ok_f` accepts either the 3-D or the 2-D gamma-mul shape.
* `ggml/src/ggml-cuda/hyperconn.cu` — relax the `out_xn` assert to the element count / both shapes.
* `src/llama-context.cpp` — `LLAMA_FUSED_DSV4_HC_PRE` / `_POST` env toggles (A/B + bisect).
* `LLAMA_HC_CN_DEBUG=1` — matcher/dispatch/fuse-check traces (diagnostic; remove before landing).

## Result — matcher arm vs the current `DSV4_HC_POST` arm

gfx1151, qwen4exp IQ4_NL, `-b/-ub 2048`, `-n 0`, interleaved, r=2:

| pp | `DSV4_HC_POST` (default) | revived matcher (`_POST=0`) | delta |
|---:|---:|---:|---:|
| 2048 | 822.1 | **834.5** | +1.5% |
| 8192 | 827.7 | **835.0** | +0.9% |
| 32768 | 797.1 | **811.5** | +1.8% |

The repeated `k_bin_bcast op_repeat` and the separate `rms_norm_f32<1024>` are absorbed by the fused
`hc_combine_norm_f32`. Same-seed greedy text was identical on a short smoke prompt; the full purity gate
has **not** been run.

## Why pwilkin is still faster here

The delivery's `hc_combine_norm_f32` is the 1024-thread/3-column variant; pwilkin's `_b256` uses 256
threads, two elements per thread and packed 32-bit accesses (`hc-cn.cu`, 554 ms/190 calls in the
2026-09-20 profile). Porting the `_b256` kernel is the obvious next step; it is a pure kernel swap
behind the same matcher.

## Next

1. Purity gate: `draft-mtp` / `plain` width purity + same-seed vs the current default, then a PPL ratio.
   The **correctness question is whether the matcher path is bit-identical to the current
   `DSV4_HC_POST` + `rms_norm` path** (the matcher kernel replays `rms_norm_f32<1024,true>`; the fused op
   may round differently).
2. Port the `_b256` variant.
3. The other half of item 1 — wire the beta's existing `hc_gate_mix_kernel` (`mmb.cu`,
   `ggml_cuda_hc_gate_mix`, currently no call site) or extend `DSV4_HC_PRE` to fold the gate GEMM — is
   untouched.
