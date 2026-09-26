# The host-buffer input layer — the `integrated` crash root-caused and fixed

**Status:** fixed, gated, committed.  Fork `~/llama.cpp`, branch **`gap-closing-hostbuf-integrated`**,
commit **`78320aaa6`**; exported as
[`patches/0025-host-buffer-input-layer.patch`](patches/0025-host-buffer-input-layer.patch).
**Supersedes `patches/0024`** (the single-device `dev_input` heuristic): `0025` reverts `0024` and
takes the reference's host-buffer path instead.  This closes **open item 1** and the **Next stage**
section of [`closing-the-gap.md`](closing-the-gap.md).

## Symptom

Restoring `info.devices[id].integrated = prop.integrated` on HIP (the reference's zero-copy APU path)
moved the input embeddings to ROCm0 and dropped the decode CPU from ~1000 % to ~155 %, but the plain
path aborted with:

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
... kernel: void k_set_rows<float, long, __half>(...)
```

`0024` had worked around it by keeping `integrated = false` and putting only the input layer on the
device buffer.  That is not the end state: it costs ~28 GiB of VRAM on qwen4exp
(`per_layer_token_embd`) and does not generalise past `n_devices() == 1`.

## What the crashing op actually is (corrects the handover's guess)

The handover guessed the QSA indexer key store.  The template arguments pin it down:
`k_set_rows<src_t=float, idx_t=long, dst_t=__half>` is the **KV cache store**,
`llama_kv_cache::cpy_k()` -> `ggml_set_rows(ctx, cache_k, k_cur, k_idxs)`: `cache_k` is f16, `k_cur`
is f32, `k_idxs` is I64 (`build_input_k_idxs` creates it as `GGML_TYPE_I64`).  The QSA mask store
(`qwen4exp.cpp`, `ggml_set_rows(kq_mask_all, zeros, top_k_3d)`) uses **I32** indices
(`ggml_indexer_top_k` returns `GGML_TYPE_I32`), so it would have been `k_set_rows<float, int32_t,
__half>` and is *not* the crashing kernel.  (It is the same *class* the reference fixed in
`14fff4f97`, but the -1 sentinel source is block selection, which this base does not have.)

## Root cause

The fork-off marker's own rationale was right: with `integrated = true`, an APU accepts the
**device host buffer** (`ROCm_Host`) as a compute-buffer type for ROCm0
(`ggml_backend_cuda_device_supports_buft`).  `llama_context` already hands the scheduler that host
buffer for the CPU slot (`backend_buft` substitution in `sched_reserve`).  The scheduler then sees
`ggml_backend_sched_buffer_supported(ROCm0, host_buft) == true`, **elides the split-input copy**, and
lets ROCm0 read the host tensor in place.

That host tensor is the one the host thread writes in `set_inputs` for the *next* ubatch.  Nothing
orders a CPU store against an in-flight GPU kernel, so the next prefill ubatch can overwrite
`k_idxs` while the previous `cpy_k` still reads it.  A torn I64 index is an out-of-bounds
`row_dst[...]` store inside `k_set_rows`; in device memory it can pass unnoticed, in
aperture-mapped host memory it faults.  That is the `MEMORY_APERTURE_VIOLATION`.

The same race is the #15034 "corrupted full-model output" the fork-off marker cited: with `-c 40000`
the naive fix (guard `t->flags & GGML_TENSOR_FLAG_INPUT`) still produced `80f92dbc955d` instead of
the baseline `8285d12d40ca`.  The missing half is that **an input can be reached only through a
view** — the recurrent-state copy is — and the view does not carry `GGML_TENSOR_FLAG_INPUT`.  The
reference's `1f2e34819` documents exactly this case for its ring.

## The fix (`patches/0025`, +43/-12 over `0024`)

Two independent halves:

1. **`ggml-cuda.cu`** — on HIP, report the real flag again:
   `info.devices[id].integrated = getenv("GGML_FORCE_NO_INTEGRATED") ? false : prop.integrated;`
   (non-HIP keeps upstream's `false`).  `GGML_FORCE_NO_INTEGRATED=1` is the A/B / bisect escape
   hatch.
2. **`ggml-backend.cpp`** — a host-resident **graph input** is never read in place by a compute
   backend while the input ring is off:
   ```c
   if (buft && ggml_backend_buft_is_host(buft) && sched->n_copies <= 1 &&
       ggml_backend_sched_graph_input(t) != NULL) {
       return false;   // force the split-input (stream-ordered) device copy
   }
   ```
   `ggml_backend_sched_graph_input()` walks `view_src` chains to find the `GGML_TENSOR_FLAG_INPUT`
   root.  **Weights are untouched** (they never carry the input flag), so `token_embd` /
   `per_layer_token_embd` stay zero-copy in `ROCm_Host` and their `GET_ROWS` runs on ROCm0.  The
   forced copy is exactly what `integrated = false` already did for these tensors, so this is not a
   regression against the delivery — it is what makes the zero-copy *weights* safe to enable.

`src/llama-model.cpp` goes back to the stock `dev_input = { cpu_dev, &cpu_dev_list }`: with the flag
on, `select_weight_buft()` picks `ROCm_Host` for the input tensors (verified with a temporary
`LLAMA_BUF_SEL_DEBUG` print) and no `n_devices()` heuristic is needed.

## Why not the reference's input ring

The reference's real fix is a graph-input **ring** (`83e8382ba` allocation + `1f2e34819`
`ggml_backend_sched_prepare_inputs()` rotation + three follow-up hardening commits, ~500 lines across
`ggml-backend.cpp` / `llama-context.cpp` / `ggml-backend-impl.h`, on an August base).  It keeps the
inputs zero-copy and avoids a per-ubatch copy.  Porting it here is a separate change with its own
A/B and risk surface; the guard above is the minimal correct stop: it matches upstream's
discrete-GPU behaviour for *inputs* and delivers the campaign's actual goal (the ~28 GiB
`per_layer_token_embd` stays in host RAM, no CPU backend dispatch).  The ring remains a viable
follow-up optimisation — `sched->n_copies <= 1` in the guard means it would compose with one.

## Gates (gfx1151, qwen4exp IQ4_NL + Q4_K_XL MTP sidecar, f16 KV, `--ctx-checkpoints 0`)

| gate | result |
|---|---|
| CPU during sparse MTP decode (`-c 16384 -n 1500`, `/usr/bin/time -v %CPU`) | **122 %** with the fix, **818 %** with `GGML_FORCE_NO_INTEGRATED=1`; the 1500-token text is byte-identical across the two (`da73bd41275e`) |
| node placement | `input_embed` / `mtp_tok_embd` on **ROCm0**; `sched_reserve: graph splits = 1` (was 2) |
| 8K plain / dense `n3` / sparse `n3` | all **`3553e76d3a9e`** |
| 40K plain / dense `n1`,`n3` / sparse `n1`,`n3` | all **`8285d12d40ca`** |
| 128K plain / MTP `n_max 1` (2 runs each) | all **`d140b40f0eee`**, deterministic |
| byte-identity vs the `0024` baseline | 8K `3553e76d3a9e`, 40K `8285d12d40ca`, 128K `d140b40f0eee` — all reproduced on a rebuilt `0024` tree under the same commands |
| width probe (`test-logits-width-probe`, local P extension) | **PASS, worst maxdiff 0** at **P=1024** and **P=32768** |
| oracles | `FLASH_ATTN_QSA` 2/2, `GATED_DELTA_NET` 2/2, `INDEXER_TOPK` 2/2 |
| VRAM | `load_tensors: ROCm_Host model buffer size = 27806.97 MiB` (the `per_layer_token_embd`); `ROCm0 model buffer = 67591.54 MiB`.  The 0024 baseline keeps that ~27 GiB on ROCm0 |
| crash stress | plain 8K/40K/128K, dense+sparse MTP 8K/40K/128K, width probe, oracles — no `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` |

The record's `770e770ae7d6` 128K hash is from a different command shape; under the commands used here
plain and MTP agree byte-for-byte at 128K on **both** the `0024` baseline and this build.

## Multi-GPU policy (stated, not measured here)

`integrated` follows `prop.integrated` **per device**, so this is a per-device policy with no
`n_devices()` branch: an APU device gets the host path, a discrete GPU keeps `integrated = false` and
the old copy path; the scheduler guard is generic.  It has only been validated on this single-GPU
gfx1151 box — the first thing a multi-GPU session should do is re-gate `-sm layer` / `-sm tensor`
with the same hashes.

## Debug aids (removed from the patch, re-add as needed)

* `LLAMA_BUF_SEL_DEBUG=1` — `select_weight_buft`, per candidate buft/device/`weight_buft_supported`.
* `LLAMA_SCHED_BUF_DEBUG=1` — `ggml_backend_sched_backend_from_buffer`, per backend
  `supports_buft`/`supports_op` for `input_embed`/`mtp_tok_embd`/`SET_ROWS`.
* `GGML_FORCE_NO_INTEGRATED=1` — the shipped A/B kill-switch.

## Files

`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-backend.cpp`, `src/llama-model.cpp`.
