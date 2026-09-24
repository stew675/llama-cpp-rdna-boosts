# OP-1.4 — draft sampler backend offload under `-sm tensor`: **closed, won't fix**

**Status:** closed 2026-09-24.  Two independent reasons: it is **structurally blocked** under
`-sm tensor` by the Meta backend, and it is **not a measurable win** even where it does work.  No code
change.

---

## 1. What the item was

`llama_context::set_sampler()` rejects the backend sampler outright when
`model.split_mode() == LLAMA_SPLIT_MODE_TENSOR`:

```c
if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
    LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
    ...
    return false;
}
```

The MTP draft builds a `top_k(10)` backend chain
(`common/speculative.cpp`, gated on `--spec-draft-backend-sampling`, default **on**); under
`-sm tensor` `llama_set_sampler()` refuses it, so the draft samples on the CPU: a full-vocab logits
D2H (`n_vocab * 4` bytes ≈ 1 MiB) per draft step plus a CPU `top_k`.

The guard is **upstream's deliberate design**, from PR #23287 ("Move to backend sampling for MTP draft
path"): *"Make backend sampling more robust and fallback to CPU on failure cases, such as with
`-sm tensor` or when a backend doesn't support TOP_K."*

## 2. Why it cannot work under `-sm tensor` (structural)

Under `-sm tensor` the model's device list is a single **Meta** device, so `set_sampler()` hands the
sampler the Meta buffer type and the sampler graph is appended to the main graph on the Meta backend.

`output.weight` is split on `GGML_BACKEND_SPLIT_AXIS_1` (vocab).  `handle_mul_mat` therefore returns
the logits `[n_vocab, n_tokens]` as `GGML_BACKEND_SPLIT_AXIS_0` — each device holds a **vocab shard**,
not the full row.  The sampler's `TOP_K`/`ARGSORT` dispatch to `handle_per_row()`, which begins with

```c
GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
```

so a global top-k over a vocab-sharded row is impossible in the current Meta backend.

**Reproduced:** lifting the guard behind `LLAMA_BACKEND_SAMPLING_TENSOR=1` (temporary, reverted) on
qwen4exp IQ4_NL + the shared-Q8_0 MTP sidecar, 3-GPU `-sm tensor`, `draft-mtp n3`:

```
ggml/src/ggml-backend-meta.cpp:544: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
```

(qwen4exp has a separate `output.weight`, `[2560, 248320]`, so the embeddings are *not* tied and the
logits really are AXIS_0-split.)

Making it work needs a **Meta-backend distributed top-k** — per-shard local top-k plus a cross-device
merge, or a MIRRORED all-gather of the logits before sampling.  That is a new backend capability with
its own kernels and its own correctness story (the servers's main sampler under `-sm tensor` hits the
same wall), not the "small" item the brief assumed.

## 3. Why it is not worth it anyway (measured)

Where backend draft sampling **does** work — `-sm layer`, where the logits live on the output device —
the A/B against `--no-spec-draft-backend-sampling` is **within noise**:

| arm | run 1 | run 2 |
|---|---:|---:|
| backend draft sampling ON (default) | 70.8 | **75.6** |
| `--no-spec-draft-backend-sampling` | 75.3 | 75.1 |

qwen4exp IQ4_NL + shared-Q8_0 sidecar, 3-GPU `-sm layer`, q8_0 KV, prose, `-c 16384 -n 1500`, seed 42.
No fallback warning fires in either arm, so the backend sampler really was engaged in the ON arm.  The
spread between identical arms (70.8 → 75.6) is larger than the arm difference: removing the per-step
full-vocab D2H and the CPU `top_k` does not move the needle.  Under `-sm tensor` the model is faster
(≈108 t/s), so the sampler's relative share is smaller still.

## 4. Verdict

* **Leave upstream's CPU fallback in place** under `-sm tensor`.
* The two prior OP-1 fixes already cover the real problem: `0029` keeps the CPU split from spinning
  (the draft sampler is part of the remaining ~1.2 cores), and `0030` removes the CPU split entirely
  for those who opt in.
* Revisit only if a *global* `-sm tensor` backend sampler is wanted for the server's own sampling
  chain (not just the draft) — that is the Meta-backend distributed-top-k feature above, and it should
  be scoped as such, with the `-sm layer` result above as the expected-magnitude prior.
