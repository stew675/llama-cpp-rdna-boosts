# Bug fix — the MMB HC16 F32-elision is invalid under an eval callback (llama-imatrix)

**Status:** fixed, gated, exported as [`patches/0019`](patches/).  Fork `~/llama.cpp` branch
`gap-closing-r13`, commit `575c4c091`.  This closes **NEXT SESSION item 2**
([`closing-the-gap.md`](closing-the-gap.md), Phase-2 correctness).

## Symptom

`llama-imatrix` on `Nanbeige4.2-3B-BF16` (a plain BF16 model) aborts with the default MMB build:

```
E collect_imatrix: non-finite values detected in blk.21.attn_output.weight
```

`GGML_CUDA_MMB=0` and `GGML_CUDA_MMB_BF16W=0` both "fix" it (they take the BF16 dense weight out of
the MMB path, which silently disables the HC16 marking as a side effect).  The real switch is
**`GGML_CUDA_MMB_HC16=0`**.  `GGML_CUDA_MMB_BF16W=1` (the default) reproduces it.  None of this is
about the IQ2 work it was first observed next to — it reproduces with the type mask irrelevant.

Two faces of the same corruption:

| command | default MMB | `GGML_CUDA_MMB_HC16=0` / `MMB=0` |
|---|---|---|
| `llama-imatrix … -c 512 -b 512 --chunks 4` | **non-finite, `blk.21.attn_output.weight`** | clean, PPL 18.4558 |
| `llama-imatrix … -c 512 --chunks 2` (default `-b 2048`) | PPL **19 231 171** (garbage) | PPL 19.5146 |
| `llama-perplexity` (no callback), same model | PPL **19.5126** (fine) | PPL 19.5126 |
| `llama-cli` greedy, same model | coherent | coherent (identical text) |

So the forward is fine without a callback — the defect is specific to the **eval-callback** path
(`llama-imatrix`'s `ik_collect_imatrix`, `common/debug`'s `common_debug_cb_eval`).

## Root cause

The `MMB HC16` graph-optimizer pass (`ggml/src/ggml-cuda/ggml-cuda.cu`) marks a producer's activation
`bf16_only` when every graph consumer reads it through the MMB BF16 activation cache.  A `bf16_only`
mark lets the producer **skip its F32 store** and have the GEMM read the cached BF16 copy instead.
That is only valid while the whole graph runs as **one** backend compute, because
`ggml_cuda_mmb_begin_graph()` (called at the start of every `ggml_backend_cuda_graph_compute`) does

```c
void ggml_cuda_mmb_begin_graph() {
    for (auto & e : g_mmb_cache) delete e.buf;
    g_mmb_cache.clear();
    for (auto & e : g_mmb_slots) { e.root = nullptr; e.data = nullptr; e.n = 0; }
}
```

With an eval callback the scheduler deliberately does **not** run the graph as one compute — it runs
each callback node as its own sub-graph (`ggml_backend_sched_compute_splits`, the
`ggml_backend_graph_compute_async(split_backend, &gv)` loop).  `imatrix` asks for every `MUL_MAT`, so a
producer lands in a sub-graph *before* the GEMM's sub-graph.  When the GEMM's compute starts, its
`mmb_begin_graph()` has cleared the cache entry the producer wrote, so `mmb_bf16_activation()` falls
through to a fresh `mmb_cvt_f32_bf16` of the activation — which, because the tensor was marked
`bf16_only`, **was never written**.  The GEMM reads uninitialised device memory; `imatrix` then
accumulates non-finite sums and the model logits are garbage.

`all_bf16_consumers()` cannot see the callback: the observation is out-of-band (the callback grabs the
tensor with `ggml_backend_tensor_get`), not a graph consumer, so the mark is "correct" by its own
definition and the F32 elision is simply unsafe in that mode.

Scope note: the HC16 step-3 marking is arch-gated `GGML_CUDA_CC_IS_RDNA3_5` (a single iGPU), and the
`DOWN16` / `blk16` / `res16` marks are default OFF, so the scheduler-split-boundary variant of this
class of problem is not reachable in the shipping configuration.  All `bf16_only` sites are guarded
anyway.

## Fix

Plumb the fact that the scheduler will split at callback nodes to the backends, and stand the F32
elision down when it holds — the BF16 "wants a copy" hint is kept, so the GEMM still reads BF16, but
the F32 output is written and any consumer (in-graph or out-of-band) sees valid data.

* `ggml/src/ggml-backend-impl.h` — new `bool has_eval_callback` in
  `struct ggml_backend_graph_optimize_params`.
* `ggml/src/ggml-backend.cpp` — `ggml_backend_sched_split_graph()` sets it from
  `sched->callback_eval != nullptr` in the `opt_params` initializer (the only construction site).
* `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_backend_cuda_graph_optimize()` computes
  `const bool elide_f32 = params == nullptr || !params->has_eval_callback;` and gates every
  `ggml_cuda_mmb_mark_bf16_only()` site with it; step 3 downgrades to
  `ggml_cuda_mmb_mark_bf16_copy()` (keep F32 + emit the copy).

The change is **inert for every serving path**: without `cparams.cb_eval` the flag is false, `elide_f32`
is true and the F32-skip path is exactly as before.

## Gates (gfx1151, ROCm 7.14)

| gate | result |
|---|---|
| `llama-imatrix` NanBeige BF16, default MMB, `-c 512 -b 512 --chunks 4` | **clean, no non-finite, PPL 18.4558** (was non-finite) |
| same, default `-b 2048 --chunks 4` | **PPL 18.4558** (was 19 231 171) |
| imatrix `in_sum2` tensors, fixed default vs `GGML_CUDA_MMB_HC16=0` | **154/154 byte-identical** (sha256 match) |
| imatrix PPL vs `GGML_CUDA_MMB=0` | 18.4558 vs 18.4555 (rounding) |
| `llama-perplexity` (no callback), NanBeige BF16 | **19.5126**, unchanged from pre-fix |
| `test-logits-width-probe` NanBeige BF16, `P=1024 ub=512` | **width_purity=PASS (worst maxdiff 0)** |
| `test-logits-width-probe` qwen4exp IQ4_NL, `P=512 ub=512` | **width_purity=PASS (worst maxdiff 0)** |

## Files

`ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`.
Patch: [`patches/0019`](patches/).  This is a **correctness** fix and belongs in the `beta/mmb-general`
HC16 feature when that is cut into a delivery block; it is staged here as a gap-closing patch because
the campaign branch is r13 + beta + gap-closing.
