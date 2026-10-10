# decode-host-overhead: three host-side cuts in the speculative decode step (bit-identical)

Three format-patches on top of `v16-a55e952b8-r37`, applied with `git am`.  Each one stands alone.

| patch | file | what | switch |
|---|---|---|---|
| 0001 | `ggml/src/ggml-cuda/ggml-cuda.cu` | async graph-input uploads | opt-in, `GGML_CUDA_ASYNC_INPUT=1` |
| 0002 | `src/llama-kv-cache.cpp` | `seq_rm` scans only the used cell range | none (always on) |
| 0003 | `common/sampling.cpp` | AVX2 block prefilter in the S2 exact top-k fast path | runtime AVX2 check, scalar fallback |

## Why

In a decode trace of the server under `-sm tensor` (2 x R9700, MTP n-max 3), each GPU was idle for a while after
every draft and verify graph, and the idle stretches lined up with the host, not with the all-reduce.  Two
measurements of the host side:

- A timer around `set_tensor` / `get_tensor` (diagnostic build, not part of this): per decode step ~91
  host->device `set_tensor` calls, 3.7 MB total, 2.6 ms with the host blocked, plus 6 `get_tensor` at 0.56 ms - about
  12 % of a ~22 ms step.  Each is a `hipMemcpyAsync` + `hipStreamSynchronize` pair of 25-50 us.
- `perf` of the server main thread: `common_sampler::set_logits_topk` 14.4 %, `llama_kv_cache::seq_rm` 7.9 %.

## 0001: async graph-input uploads (opt-in)

`ggml_backend_cuda_buffer_set_tensor` waits for every copy.  With `GGML_CUDA_ASYNC_INPUT=1`, a write of <= 1 MiB
into a COMPUTE buffer is staged through a per-device 16 MiB pinned ring and copied asynchronously on
`cudaStreamPerThread` (the stream the synchronous path uses, so a later `get_tensor` stays ordered after it), and an
event is recorded after it.  Ordering:

- the next `graph_compute` on that device makes its stream wait for that event;
- a device->device `cpy_tensor_async` out of that device waits for it too;
- the copy itself waits (on the GPU) for an event recorded when the previous `graph_compute` on that device
  returned, so an input is never overwritten while the previous graph may still read it (llama reuses graphs and
  does not synchronise between them);
- the ring is reused only after draining the copy stream, once per lap.

Any failure to set up the ring (pinned alloc, events) falls back to the synchronous path.  The inputs are written
from one host thread (llama's `set_inputs`).

## 0002: `seq_rm` scans only the used range

Speculative decoding rolls back rejected drafts with `seq_rm` every step.  With `-c 262144` the loops walked all
262,144 cells, although an empty cell has `pos -1` and can never match.  Both loops now run over
`[used_min(), used_max_p1())`.

## 0003: AVX2 prefilter in the S2 top-k fast path

The exact top-k fast path scans the whole vocabulary (248k on qwen4exp) for every sampled position, and its per-64
block max / NaN loop was scalar (no float max reduction without fast-math).  A full block now goes through
`_mm256_max_ps` plus an unordered-compare NaN flag.  The block max is only a filter, and a NaN still sends the caller
to the full path, so the selection is the same.  `__attribute__((target("avx2")))` with a
`__builtin_cpu_supports("avx2")` check (`IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE)` on Windows, where
clang with the MSVC ABI does not link the compiler-rt symbol that check needs - fix and report by @DanoPTT); the scalar
loop remains the fallback.

## Results on r37 (2 x R9700 / gfx1201, PCIe 5.0 x8 each, ROCm 10.0)

Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head, all experts in VRAM, `--spec-draft-n-max 3
--spec-draft-p-min 0`, q8_0 KV, `-ub 2048 -b 2048`.  One warm-up request, then 3 greedy 400-token requests per arm;
each arm is a fresh server.

| arm | `-sm tensor`, 256K | `-sm layer -ts 59,41`, 128K |
|---|---|---|
| stock r37 | 125.1 / 126.1 / 126.1 t/s | 103.7 / 103.7 / 103.6 t/s |
| r37 + 0001-0003, async off (0002 + 0003 only) | 128.9 / 129.6 / 129.8 (+2.9 %) | - |
| r37 + 0001-0003, `GGML_CUDA_ASYNC_INPUT=1` | **137.1 / 137.9 / 137.8 (+9.4 %)** | **107.1 / 107.3 / 107.1 (+3.4 %)** |
| greedy sha, every request | `45e68f1bdfeb` | `f2471f845142` |

0 GPU faults in every arm.  Earlier, on our local build (r37 plus a few patches of ours) with the same three
changes and async on, the full server suite gave the same shas everywhere, prefill unchanged (1.8k / 37k / 155k),
2 concurrent requests 175 -> 194 t/s aggregate, the 900-building 36k recall OK, and the 259.6k-token VRAM peak
unchanged.

**One thing to flag on 0001:** on its first version (before the done-event ordering above was added), one greedy
request out of the whole suite gave a different sha (`b0b49a20ba97`) once; it did not reproduce in 49 later runs on
the first and the hardened build (24 fresh servers with warm-up + 2 greedy, 25 repeats on one server).  The
done-event ordering closes the one overwrite window we could find; that is why the switch is opt-in.

## Not measured

Host experts (`--n-cpu-moe`), RDNA3 / NVIDIA, 3-GPU layouts, CPUs without AVX2 (scalar path, same as before).
