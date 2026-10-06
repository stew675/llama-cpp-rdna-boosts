# wip/gpu-keepalive — optional keep-alive for an idle GPU (opt-in, bit-identical)

*The patch and this note were written by Claude (AI).*

One `git am` patch on `v16-a55e952b8-r15` (applied tree `0e9273f8…` → `05ae40fc…`). Two new files (`ggml/src/ggml-cuda/keepalive.cu/.cuh`, so a cmake reconfigure is needed) and six added lines in `ggml-cuda.cu`.

`GGML_CUDA_KEEPALIVE_MS=N` (milliseconds; unset or `0` = off, the default):

- A background thread per device launches one 1-thread kernel on its own non-blocking stream when, for N ms, no graph was submitted, no `synchronize` ran, and every submitted graph has finished on the GPU (an event recorded on each submitting stream).
- It stays quiet during prefill and decode, and touches only its own 4-byte buffer, so outputs are unchanged.
- When enabled, the log shows one line: `GGML_CUDA_KEEPALIVE_MS=N: device 0 gets a tiny kernel after N ms without work`.

Motivation: on Windows, with the desktop on an integrated GPU, an idle discrete R9700 was powered down and its memory contents moved out, so the next request first waited for the model to come back. It is the same idea as Metal's `GGML_METAL_RESIDENCY_KEEP_ALIVE_S`. It is not Windows-gated, since other platforms can power down an idle GPU too, but it is off unless set.

Recommended setting from our own use on Windows 10 (AMD Software: Adrenalin Edition PRO 26.Q3, driver 23.19.23.11-250701a-417877C): `GGML_CUDA_KEEPALIVE_MS=2000`.
