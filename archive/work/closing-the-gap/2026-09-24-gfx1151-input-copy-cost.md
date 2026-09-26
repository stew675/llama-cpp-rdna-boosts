# gfx1151 — §9 step 1: the per-ubatch host-input copy cost (the input-ring go/no-go)

**Status:** measured 2026-09-24, **step 1 complete**.  Build: `gap-closing-denseband`
(`gap-closing-final` + `0031`), tree `468c64963ae45e72367c73809efa7cc038217e8a`, single gfx1151 (`halo`).
Instrumentation: temporary commit `SPLIT_COPY_STATS` accounting around the `ggml_backend_sched_compute_splits`
split-input copies (`ggml/src/ggml-backend.cpp`), gated by `GGML_SPLIT_COPY_STATS=1`; reverted after
measuring (the delivery build is restored).

**Question (§9 step 1):** with `0025`'s stopgap in place, is the per-ubatch host-input copy actually
visible on the APU?  If not, the reference input ring is a nicety, not a win — stop.

## What is copied

`0025`'s guard (`ggml_backend_sched_buffer_supported`) forces a stream-ordered device copy for every
**host-resident graph input** (`GGML_TENSOR_FLAG_INPUT`, plus view-reached inputs such as the GDN
recurrent-state copy).  The HIP `set_tensor` path is a synchronous `hipMemcpyAsync` +
`cudaStreamSynchronize`, so each copy is serialized with compute.  `LLAMA_DEVICE_INPUT=1` (`0030`)
does **not** change these copies (it moves the input *layer*, not the graph inputs): the copy stats are
byte-for-byte identical with and without it.

## Measured (host-side copy + sync time, `GGML_SPLIT_COPY_STATS`)

| workload | bytes | count | max | copy ns | run time | **copy share** |
|---|---:|---:|---:|---:|---:|---:|
| pp8192 (`llama-bench -p 8192 -n 0 -r 1`, 2 prefills) | 172.7 MB | 648 | 20 MiB | 5.83 ms | 13.04 s | **0.045 %** |
| pp32768 (same, 2 prefills) | 721.3 MB | 2736 | 20 MiB | 18.10 ms | 53.58 s | **0.034 %** |
| MTP `n1000` draft-mtp n3, ub 2048 | 479.6 MB | 16 088 | 80 MiB | 142.8 ms | 17.79 s | **0.80 %** |
| plain `n1000` (spec none), ub 2048 | 357.8 MB | 12 313 | 20 MiB | 144.3 ms | 31.75 s | **0.46 %** |
| MTP `n1000` draft-mtp n3, ub 8192 | 479.5 MB | 16 099 | 215 MiB | 159.7 ms | 18.32 s | **0.87 %** |

Per generated token the copy+sync is **~143–144 µs** (MTP and plain alike); the percentage differs only
because MTP generates ~2× faster.  The copied bytes are essentially **independent of ubatch** (479 MB at
ub 2048 and ub 8192) — the cost is the fixed per-graph input/state set, not the token rows.

## Verdict — **measurable but marginal; recommend STOP (do not port the ring)**

* **Prefill (the campaign's priority): negligible — 0.03–0.05 %.**  No reason to touch it.
* **Decode: ~0.5–0.9 %** (≈140 µs/token).  Deterministic in the accounting, but **at or below the
  end-to-end benchmark noise floor** (±0.5–1 % on this box), and below the campaign's ~1 % win bar
  (`0014`'s +0.5 % is the low end of what has shipped).
* The ring is a **~500-line upstream port** (`83e8382ba` ring allocation + `1f2e34819`
  `ggml_backend_sched_prepare_inputs()` rotation + hardening) across `ggml-backend.cpp` /
  `llama-context.cpp` / `ggml-backend-impl.h`, with its own scheduling/race risk surface; the stopgap
  already delivers correctness and the zero-copy weights.

**Recommendation: keep the stopgap, do not port the input ring §9 step 2.**  Revisit only if a future
workload makes the per-ubatch host input materially larger (e.g. a much larger host-resident input
tensor set) or if the copy becomes a measurable end-to-end decode regression.

> **Maintainer decision (2026-09-24): confirmed — do not port the ring.**  The brief's literal rule
> ("if measurable → port") would say *go*, but the measured magnitude (<1 % decode, ~0 % prefill)
> does not justify the ~500-line scheduler port.  §9 is closed with the stopgap retained.

## Reproduce

```sh
# instrumentation: wrap the two ggml_backend_tensor_copy(input, input_cpy) sites in
# ggml_backend_sched_compute_splits with a nanosecond timer + vector byte counter, print at atexit;
# build with the patches, then:
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
export GGML_SPLIT_COPY_STATS=1
Q=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
build-rocm/bin/llama-bench -m "$Q" -ngl 99 -p 32768 -n 0 -b 2048 -ub 2048 -r 1 -fa auto 2>&1 | grep SPLIT_COPY_STATS
# ... and the MTP run from the p3 record; the SPLIT_COPY_STATS line prints at exit.
```
