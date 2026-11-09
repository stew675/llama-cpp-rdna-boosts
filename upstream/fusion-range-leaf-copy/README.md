# UPSTREAM-PR: a leaf in a compute buffer is not an immutable weight (the fusion range check)

> **Upstream candidate** · status: prepared 2026-11-09; `git apply --check` clean on the delivery's fork point
> `a691b2345` · validated in the delivery tree on gfx1201 (1 GPU) · not filed.
> Delivery: **block 10** (folded at r239) · patch:
> [`UPSTREAM-PR-fusion-range-leaf-copy.patch`](UPSTREAM-PR-fusion-range-leaf-copy.patch)
> sha256 `0e124ca5dfe12f0c00964280ccb53bd19855be62387b7a17d4db4402d3c373f3`.

**Where it lives in the delivery:** `patches/0010-rdna-boosts-block-10-...patch` (the r239 amendment, in
`ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu`, plus the same rule at the rope-fwht matcher).  This directory's
patch is the **upstream form**: a single hunk in `ggml/src/ggml-cuda/ggml-cuda.cu`, where upstream still keeps
`ggml_cuda_check_fusion_memory_ranges()`, with no trace helper and no second call site.

## The bug

`ggml_cuda_check_fusion_memory_ranges()` is the byte-range gate that decides whether a fusion may run: for
every out node it walks every src of every node in the window and declines the fusion when an out node
overlaps a src the window does not produce.  It skips every `src->op == GGML_OP_NONE` because a leaf "is a
weight".  A scheduler copy of a host tensor is also a leaf, but it lives in the **compute** buffer, and
`ggml_backend_sched` computes the allocator's last use on the **unfused** graph: a fused window reads that copy
before the later node that owns its output runs, so the copy's block can already have been handed to the fused
output.  The fused kernel then writes the output while reading the weights it is overwriting.

Reached in the delivery with `--ncmoe`/`--cpu-moe` (host-resident expert tables copied per ubatch) plus
`GGML_OP_OFFLOAD_MIN_BATCH` small enough to offload a 9..16-token ubatch, which is what makes the routed MoE
eligible for the fused MMVQ `MUL_MAT_ID(gate) MUL_MAT_ID(up) GLU`: the logits go NaN.  On unadulterated
upstream the same window exists for the fused MMQ gate/up/GLU with ids (the reporter's note: op offload needs
32+ tokens by default there and the MMVQ MoE cap is 8, so it is harder to reach, and it has not been
reproduced upstream).

## The change

A leaf is skipped only when it cannot be a graph result:

```cpp
if (src->op == GGML_OP_NONE &&
    (src->buffer == nullptr ||
     ggml_backend_buffer_get_usage(src->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
    continue;
}
```

Conservative by construction: it can only **decline** a fusion, never admit one, and only when the out node
really overlaps the leaf's allocation.

## Validation (fork tree, gfx1201, 1x R9700, ROCm 10.1)

`scripts/gate-fusion-alias-copy.sh` (the delivery's gate for this shape; the upstream hunk is the same
condition):

* the repro arm (Qwen3.6-35B-A3B Q4_K_M, `--n-cpu-moe 99 -fa on -ub 14`, expert cache off,
  `GGML_OP_OFFLOAD_MIN_BATCH=9`) goes from 58 degenerate characters (`3333...`) to the **unfused chain's**
  text, byte for byte (`193 chars sha=3fab4da87774`, identical to `GGML_CUDA_DISABLE_FUSION=1` and to
  `GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1`);
* the failed-host-fill arm (`MOE_HOST_POOL_FILL_FAULT=200`) does the same;
* the production arm (expert cache on) is unchanged (`193 chars sha=3fab4da87774`), and the check declines
  238 windows there, so it is not inert;
* prefill and decode on that config are unchanged within noise (interleaved: `pp512` 1204.6 -> 1213.1 t/s,
  `tg128` 83.08 -> 83.03 t/s).

## What was NOT validated

* Only gfx1201 / ROCm 10.1, and only the shapes the delivery's gate covers.  No NVIDIA/CUDA, MUSA, SYCL or
  Vulkan run: the condition is backend-neutral but the *reachability* on those backends was not tested.
* Not reproduced on unadulterated upstream, so an upstream PR would need its own repro story (the reporter's
  reasoning is that the fused MMQ gate/up/GLU with ids has the same exposure).
* No measurement of what the extra declines cost on a non-AMD backend (they cannot cost anything here:
  0.7 % / -0.06 % on the delivery's own config).
