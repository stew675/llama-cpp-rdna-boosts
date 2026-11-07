# wip/host-input-sync — sched: copy a large host-resident split input synchronously

Written by Claude (AI): the diagnosis, the patch, the measurements and this text are Claude's.  One `git am` file with one commit, made against the `v16-6050b467c-r227` tree (base `6050b467c` + `rdna-boosts-all.patch`); built and measured on an r220-based fork, where the same hunk applies.  It touches only `ggml/src/ggml-backend.cpp` (+23 / -1).
`host-input-sync.patch` sha256 `f358643c95e9ba37734e6fff7eb66f90556ebe7b293056dbb9ab103e906a91f4`.

Kill switch: `GGML_SCHED_ASYNC_LARGE_HOST_INPUT=1` restores the previous asynchronous copy.

## Why

`ggml_backend_sched_copy_input`'s `host_src && dev_dst` branch (block 09) uploads a host-resident split input with `ggml_backend_tensor_set_async` on the split backend's stream.  The copy engine reads the source when the stream reaches the copy.  The CUDA backend's pinned-ring snapshot (block 10, `ggml_backend_cuda_set_tensor_async`, whose comment describes exactly this hazard) covers uploads up to `GGML_CUDA_ASYNC_INPUT_MAX` (1 MiB); a larger upload is read in place.

The one that bites is a CPU split's output.  With the token embedding table on the CPU (the default placement of the input layer; a `q4_K` `token_embd` cannot use `ROCm_Host` and lands in `CPU_Mapped`), every ubatch uploads `model.input_embed` / `node_1`: 41,943,040 bytes for a 2048-token ubatch.  The next ubatch's CPU split rewrites that buffer in place.  If the GPU stream is still busy when the host gets there, the queued copy reads the next ubatch's embeddings, or a torn mix.

The result is a corrupted prefill, not a rounding difference: the layer-1 input was 0.3% off (12% at the worst element) and 22% by layer 63, with different wrong values from run to run.  The bytes are real token embeddings, so generation stays fluent and the damage is hard to see in the text.

Plain llama.cpp `6050b467c` does not have it: its scheduler sends a CPU-to-GPU split input through `cpy_tensor_async`, which the CUDA backend declines for a CPU source, and falls back to the synchronous `ggml_backend_tensor_copy`.

## What the patch does

A host input above 1 MiB whose buffer is not a weight buffer skips the asynchronous branch and takes the existing synchronous fallback, as upstream ggml does for every host input.  Weights (partial offload, host expert masters) are never rewritten in place and keep the asynchronous copy at any size; inputs up to 1 MiB keep it too (the ring snapshots them).

## How it was found and checked

Radeon AI PRO R9700 (gfx1201), Linux, ROCm 10.0.0, Ryzen 9 3900X.  Qwen3.8-27B UD-Q4_K_XL (and a 27B test variant), `-ub 2048`, token embeddings on the CPU.  A small test program fed the same tokens through the model and compared the logits bit for bit, normal run vs `AMD_SERIALIZE_KERNEL=3`.

- It only showed with the stream kept busy: exporting the inputs of 32 or more layers per 2048-token ubatch to the host (`llama_set_embeddings_layer_inp`; 1.3-2.7 GB of device-to-host readbacks per ubatch).  16 layers or fewer, or the same 64 layers at `-ub 512`, did not trigger it.
- `AMD_SERIALIZE_KERNEL=3` and `AMD_SERIALIZE_COPY=2` (wait after each copy) removed it; `AMD_SERIALIZE_COPY=1` (wait before) did not.  HIP graphs off, `GGML_CUDA_ASYNC_INPUT=0`, `GGML_CUDA_ASYNC_INPUT_BATCH=0`, `GGML_CUDA_ASYNC_INPUT_LAZY_DONE=0`, `GPU_MAX_HW_QUEUES=1` and graph reuse off did not.  `ROCBLAS_USE_HIPBLASLT=0` changed nothing.
- Keeping the exports but skipping their readback copies removed it.  Logging the uploads above 1 MiB that are not snapshotted showed only `model.input_embed` / `node_1` (41,943,040 bytes per ubatch; 38,010,880 for the last 1856-token chunk), and making each of them wait for completion removed it.
- With the patch: normal run equal to the serialized run, twice; with `GGML_SCHED_ASYNC_LARGE_HOST_INPUT=1`: the corruption back.
- Not observed under a plain `llama-server` workload (32K prompts, one and two slots, 12 requests per arm, with and without the patch): identical first-token logprobs.  Anything that queues large device-to-host work between prefill ubatches could expose it.

## Cost

`-c 40960 -ub 2048`, 1 slot, no draft, token embeddings on the CPU; prefill 8K twice and 32K, 256 decode tokens each; 10 server starts in the order A B B A (with three other arms in between):

| | prefill 8K | prefill 32K | decode |
|---|---|---|---|
| this patch | -0.71% | -0.40% | -0.12% |

Tokens unchanged in every run; the unpatched arm itself drifted ~1.2% between its first and last start.  Measured alternatives, not included: an event recorded after the upload and waited on before a host split recomputes (-0.66% / -0.37%, the same within noise, more code); a pinned double-buffer snapshot of the large uploads in the CUDA backend (-1.66% / -1.28%).

## Not tested

- Multi-GPU (`-sm layer`, `-sm tensor`), the host-resident MoE paths with the arena (`-ncmoe`), CPU-only and Vulkan builds, Windows.
- The fallback's cost under pipeline parallelism (`n_copies > 1`).
- Not built on r227 (only `git apply` on the assembled r227 tree).

## Related, not changed here

Noticed while auditing the same ordering; both are safe today only because the host happens to have synchronized first:

1. `ggml_backend_cuda_buffer_set_tensor` copies an upload above 1 MiB on `cudaStreamPerThread` and synchronizes that stream only; the compute stream is non-blocking, so the copy is not ordered against a graph still running.  A compute-buffer input (e.g. a step slab) that grows past 1 MiB moves from the ordered ring path to this one.
2. The async-input ring's lazy done event is kept per host thread for the last graph only; a target and a draft context issuing graphs from the same thread overwrite each other's pending done event.
