# The input-layer CPU burn — the token embedding runs on the GPU now

**Status:** fixed, gated, committed.  Fork `~/llama.cpp`, branch `gap-closing-r13`, commit
**`73a391aba`**; exported as [`patches/0024-input-layer-on-gpu-single-device.patch`](patches/0024-input-layer-on-gpu-single-device.patch).
**SUPERSEDED 2026-09-23 by [`patches/0025`](patches/0025-host-buffer-input-layer.patch) and
[`2026-09-23-host-buffer-input-layer.md`](2026-09-23-host-buffer-input-layer.md):** this record's
`0024` is a single-device heuristic that keeps `integrated = false` and puts the input weights in a
*device* buffer (~28 GiB VRAM on qwen4exp); `0025` root-causes the `k_set_rows`
`MEMORY_APERTURE_VIOLATION` and takes the intended zero-copy `ROCm_Host` path instead.  `0024` is
kept on branch `gap-closing-r13` only as the pre-`0025` A/B baseline.  The body below is the
session-11 record and is left intact.
This closes the "CPU cores light up during sparse MTP decode" observation and is a
**delivery-level** (not sparse-MTP-specific) change.

## Symptom

During an MTP decode (dense or sparse) all CPU cores sat at ~100% for the whole run:
`llama-cli` averaged **1068-1090 % CPU with 17 threads in `R`** (16 OpenMP workers + main),
while the reference fork (`~/pwilkin-llama-cpp`) doing the same work averaged **~160 % with 2
threads** (both ROCm runtime wait threads).  A stack sample of ours showed the CPU backend under
`common_speculative_impl_draft_mtp::draft() -> llama_decode -> ggml_backend_cpu_graph_compute ->
ggml_graph_compute -> __kmpc_fork_call`, i.e. the draft's token-embedding gather on the CPU backend
with LLVM OpenMP's default `KMP_BLOCKTIME=200 ms` spinning the pool after every step.

The trigger was `LLAMA_MTP_SPARSE=1`: with `LLAMA_MTP_SPARSE=0` ours already matched the reference
(152 %).  The hybrid-idx draft graph puts six extra CPU ops in the same split (the recurrent-state
zeroing `CPY`/`SCALE` pairs at `graph_mtp`), and that 7-node CPU graph keeps the pool hot; the dense
path's CPU split is only the one `GET_ROWS`.

## Root cause

The problem is not that the embedding is on the CPU by itself — it is that **the whole input layer is
on the CPU on our tree but on the GPU (ROCm0) in the reference**.  At 8K MTP:

```
OURS  node #0 (GET_ROWS): model.input_embed [CPU]   src: token_embd.weight (341M) [CPU]
OURS  node #8 (GET_ROWS): mtp_tok_embd-48   [CPU]   src: token_embd.weight (341M) [CPU]
REF   node #0 (GET_ROWS): model.input_embed [ROCm0] src: token_embd.weight (341M) [ROCm0]
REF   node #8 (GET_ROWS): mtp_tok_embd-48   [ROCm0] src: token_embd.weight (341M) [ROCm0]
```

The chain, instrumented step by step:

1. `llama-model.cpp` hardcodes `dev_input = { cpu_dev, &cpu_buft_list }` with the upstream comment
   "there is very little benefit to offloading the input layer".  Both trees start from that.
2. `select_weight_buft()` normally picks the **device host buffer** (`ROCm_Host`, GPU-accessible
   pinned host memory) for `token_embd` — it is first in `make_cpu_buft_list()`.  Both trees pick
   `ROCm_Host` (verified with a debug print in both).
3. The scheduler's `ggml_backend_sched_backend_from_buffer()` assigns the leaf to the first backend
   that supports its buffer type.  Instrumentation:
   - **OURS**: `i=0 ROCm0 supports_buft=0 supports_op=1`, `i=1 CPU supports_buft=1 supports_op=1`
     -> **CPU**.
   - **REF** : `i=0 ROCm0 supports_buft=1 supports_op=1` -> **ROCm0**.
4. `ggml_backend_cuda_device_supports_buft()` returns
   `(is_cuda(buft) && buft->device == dev) || (integrated && is_cuda_host(buft))`.  The
   `ROCm_Host` buffer only qualifies when **`integrated`** is true.
5. The reference restores `prop.integrated` on HIP:
   ```c
   #if defined(GGML_USE_HIP)
       info.devices[id].integrated = prop.integrated;
   #else
       info.devices[id].integrated = false; // ... corrupted output (e.g. #15034)
   #endif
   ```
   Our delivery follows upstream's #28604 revert and hardcodes `integrated = false`
   (`ggml-cuda.cu`, the block-06 "host-buffer revert"), so the host buffer is rejected and the
   scheduler falls through to the CPU backend.

So the CPU burn was a downstream symptom of forcing `integrated = false` on an APU where the
reference uses the host-buffer path.

## Why we did not take the reference's fix

Restoring `integrated = prop.integrated` on HIP **does** move both embeddings to ROCm0 and drops the
CPU to 155 %, but on our tree the **plain (no-MTP) path crashes** with a real GPU fault:

```
Queue error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION:
... kernel: void k_set_rows<float, long, __half>(...)
```

The reference's plain 8K `-n 200` run is fine (29.7 t/s), so our base is missing part of the
host-buffer support the reference has (the broad `integrated` path also lets the scheduler place
other host-buffer tensors — e.g. the QSA indexer cache — on the GPU).  Debugging that is a separate
investigation and a much larger blast radius; the crash is reproducible and the #15034 class.

## The fix (`patches/0024`)

Keep `integrated = false` (no broad host-buffer path, no crash) and move only the **input layer**
to the GPU:

```c
// assign the input layer.  With a single GPU the token-embedding weights (token_embd /
// per_layer_token_embd) are placed on the output layer's device, so their GET_ROWS runs on the
// GPU.  On an APU that avoids dispatching the CPU backend - and spinning its OpenMP thread
// pool - on every step (notably the MTP draft's mtp_tok_embd gather).  Multi-GPU setups keep
// the stock CPU placement for now.
if (n_devices() == 1) {
    pimpl->dev_input = get_layer_buft_list(n_layer_all);
} else {
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
}
```

This puts `token_embd` (and `per_layer_token_embd`, both `LLM_TENSOR_LAYER_INPUT`) in the device
buffer (ROCm0), so the `GET_ROWS` is a GPU op.  Multi-GPU keeps the stock CPU placement (the
follow-up will choose a host-buffer / device policy per split).

## Gates (gfx1151, qwen4exp IQ4_NL + MTP sidecar unless noted)

| gate | result |
|---|---|
| CPU during sparse MTP decode | **155 % / 2 threads** (was 1068-1090 % / 17), matching the reference |
| node placement | `model.input_embed` and `mtp_tok_embd-48` on **ROCm0** |
| 8K plain / dense `n1` / sparse `n1`,`n3` | all `3553e76d3a9e` |
| 40K plain / dense `n1` / sparse `n1`,`n3` | all `8285d12d40ca` |
| 128K plain / MTP `n_max 1` | all `770e770ae7d6` (2 MTP runs) |
| width probe `test-logits-width-probe` | **PASS (worst maxdiff 0)** |
| `llama-imatrix` NanBeige4.2-3B-BF16 | clean, PPL 25.0705, file **byte-identical to HC16=0** |
| dense NanBeige4.2-3B-Q8_0 coherence | coherent greedy continuation |
| plain 8K `-n 200` | no crash (the `integrated` variant crashed here) |

## Follow-ups

* The reference's broad `integrated = prop.integrated` path (host buffer for *all* GPU-supported
  weights, no VRAM cost) is still the better end state — the crash in `k_set_rows` on our base needs
  to be root-caused before re-landing it.
* Multi-GPU: `n_devices() > 1` keeps the CPU placement; a per-split host-buffer / device policy is
  the open item.
* Single discrete GPU: this change moves `token_embd` (and, for qwen4exp, the ~27 GiB
  `per_layer_token_embd`) into VRAM.  Fine on the 123 GiB APU; a discrete card with less VRAM than
  the model needs is a policy call.

## Files

`src/llama-model.cpp`.  The investigation also touched (then reverted) debug prints in
`src/llama-model-loader.cpp` and `ggml/src/ggml-backend.cpp` on both this tree and
`~/pwilkin-llama-cpp`.
