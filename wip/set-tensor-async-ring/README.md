# wip/set-tensor-async-ring

Keeps r90's `set_tensor_async` snapshot (the copy reads bytes taken when the upload is issued), but takes the snapshot into a small pinned ring owned by the context and issues the copy on the backend's own stream, as before r90. With r90 each such upload also joined the graph-input upload ring's queue. That queue is flushed by a batch kernel, or, if anything drains it first, as a per-thread-stream copy with cross-stream event waits.

One format-patch on `v12-097cf33d9-r91` (tree `90507dfb`); applied tree `23c43146`. Patch sha256 `a048880d…`.

## What changes

`ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh`, +52 / −5, only in `ggml_backend_cuda_set_tensor_async` and the context:
- A small (<= `GGML_CUDA_ASYNC_INPUT_MAX`) upload into a COMPUTE buffer is copied into a pinned 2 x 4 MiB ring kept by the `ggml_backend_cuda_context`. Then one `cudaMemcpyAsync` H2D goes on `cuda_ctx->stream()`, which the scheduler has already ordered after its wait.
- When an upload does not fit the current half, the ring records an event for that half on the same stream, waits for the other half's event (recorded when the ring last left it), and switches. Events are only recorded between splits, so the wait never depends on work the host has not launched yet.
- The ring and both events are set up together on first use. If that fails, nothing is kept and the r90 path is used for the rest of the context's life.
- `set_tensor_2d_async` and the rest of r90 are unchanged.

## Correctness

**Your control** (`archive/work/async-input-graph-ordering`, FINDINGS-2026-10-19b):
- `llama-perplexity --kl-divergence-base` self-comparison, `-c 512 -ub 8 -b 8 -ngl 99 -fa on -t 8 -ctk q8_0 -ctv q8_0`, 1 GPU.
- `Qwen3.8-27B-Q4_K_M` (the closest we have to your 27B `UD-Q4_K_M` row), `GGML_CUDA_ASYNC_INPUT=1` in every arm.
- Each arm is its own build of that tree.

| build | 8 chunks: same top p / RMS Δp | 24 chunks: same top p / RMS Δp |
|---|---|---|
| r89 (before the fix) | **98.039 %** / 2.298 % | **97.386 %** / 2.656 % |
| stock r91 | 100.000 % / 0.001 % | 100.000 % / 0.001 % |
| r91 + this | 100.000 % / 0.001 % | 100.000 % / 0.001 % |

So the defect reproduces on this box, and the change keeps your fix.

**On the tensor-split config below** (G1), the same instrument gives 100.000 % at `-ub 8` (24 chunks) and `-ub 1` (8 chunks). r89 was also clean there, so on that config it only shows no regression.

**Greedy hashes** (400 tokens, prose / code), identical to stock r91:

| config | prose / code |
|---|---|
| G1: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` + shared Q8_0 MTP head, 2 GPU `-sm tensor`, `-ncmoe 10` | `a7a81f01b3bc` / `4bb43e3339c8` |
| G2: Qwen3.6-35B-A3B Q4_K_M, 1 GPU, `-ncmoe 10` | `047fd057911b` / `231119399578` |
| G3: Qwen3.6-35B-A3B Q4_K_M, 2 GPU `-sm tensor`, `-ncmoe 10` | `c0cf0739bdab` / `930bcf1f4405` |

G1 is the #186 command; G2 and G3 change `-m` and drop the MTP flags, and G2 sets `ROCR_VISIBLE_DEVICES=0`.

**Faults:** none in 18 servers.

## Speed

**Stock r91:** neutral on G1 (4 servers per arm). On G3 (2 per arm) and G2 (1 per arm) no difference above about 1 %, with too few servers for a noise estimate. Decode on the second 400-token greedy request:

| config | stock r91 | + this |
|---|---|---|
| G1 prose / code | 137.3 / 137.3 t/s | 136.8 / 137.4 t/s |
| G1 prefill 31.5k | 2094 t/s | 2108 t/s |
| G3 prose / code | 123.4 / 122.2 t/s | 123.0 / 122.5 t/s |
| G2 prose / code | 114.7 / 115.3 t/s | 114.0 / 114.1 t/s |

**Our hybrid Flash-Next build** (r91 + our patches, IQ3_S, `-ncmoe 47`, `-sm tensor`, MTP draft): decode +2.68 % (3 pairs, CI +2.47 to +2.88), prefill unchanged, greedy output identical. Turning r90's snapshot off there recovered the same ~2.3 points, which is what pointed at it.

**How often the path runs:**
- In a 64-token request on G1, stock r91 takes it 254 times:
  - `mtp_tok_embd`, 10 KB per draft step;
  - `node_1`, `model.input_embed` and `mtp_tok_embd`, 40 KB per verify step;
  - each once per device.
- Our build takes it about 306 times, so the call counts are similar.
- Our reading: the cost depends on whether something drains the queue between the upload and the graph. Our build's other host-side work does; stock r91 apparently does not. That part is a reading, not a measurement.

## Note on the version described in #182

The first version of this (the one our #182 comment measured) derived the ring half from the head offset. At an exact fill of a half, it could write past the ring or reuse a half early. These stock gates caught it as segfaults in `memcpy`. This patch tracks the half explicitly.

## Not tested

`-sm layer` and 4 concurrent streams on stock r91 with the G1 config, which hit the abort in #187.
