# OP-4 (unrun gates) + OP-5.2 (`0011` under `-sm tensor`) — gfx1201

**Date:** 2026-09-24 · **Box:** soar (3× Radeon AI PRO R9700, gfx1201) · **Build:** `~/llama.cpp`
`closing-gfx1201` tree `99b429a60d441f814c84737cfa57803bc15a2f6d`.

Harness note: qwen4exp IQ4_NL prefill A/Bs must be **interleaved** with a warm page cache — the 93 GiB
model plus its host PLE sits near the 163 GiB cache, and running arms in sequence produces a fake
monotonic ramp (default 2736 → 0001 2855 → 0008 3078 → 0011 3212 t/s at pp8192).  Interleaved, the
spread collapses.  Raw logs: `tools/runs/op4/`.

---

## OP-4(a) — `llama-imatrix`: **smoke check PASS; surfaced a pre-existing `-sm tensor` corruption**

`llama-imatrix` (`0019`/`0023`) on the 27B BF16 completes clean under `-sm tensor`: no abort, no
meta-split assert, no non-finite, an imatrix written.

```sh
build-rocm/bin/llama-imatrix -m /llm/models/Qwen3.8/27B/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf \
  -f prompts/prose-rdna-boosts.txt -ngl 99 -sm tensor -c 512 -b 512 -ub 512 --chunks 4 -o /tmp/im.dat
```

**But the output is wrong under `-sm tensor`:**

| | `-sm layer` | `-sm tensor` |
|---|---|---|
| reported PPL (27B BF16) | **8.1107** | **129104.9627** |
| reported PPL (4B Q8_0) | **16.3541** | **62943.1755** |
| imatrix `ffn_down.weight.in_sum2` | sane, corr 1.0 | **uncorrelated** (corr 0.04 on the 4B), max values ~10²–10⁴× too large |
| `llama-perplexity` (same model, same config) | 8.1107 | **8.1120** (fine) |

* **`llama-perplexity` under `-sm tensor` is sane (8.1120 vs 8.1107)**, so the split logits/forward are
  correct — this is **imatrix-specific**, not a tensor-split math bug.
* **Deterministic**: two `-sm tensor` runs are byte-identical (`2efea095…`), and the `-sm layer` file is
  stable too (`15a736e8…`).
* **Pre-existing**: the r13+beta baseline build (`~/llama-baseline` @ `e33e075cd`) produces the
  **byte-identical** `-sm tensor` imatrix (`2efea095…`) and the same 129104.9627 PPL.  Not a closing-set
  regression.
* **Not the beta `mmb` campaign**: `GGML_CUDA_MMB=0`, `GGML_CUDA_MMB_HC16=0`, `GGML_CUDA_MMB_MOE=0` all
  leave the `-sm tensor` imatrix byte-identical.
* **General**: reproduces on the 4B Q8_0 in seconds (`-sm tensor` 62943 vs `-sm layer` 16.35), 244/496
  `in_sum2` tensors off by >1%.

Likely site: the imatrix PPL path (`tools/imatrix/imatrix.cpp`, `num_batches == 1` branch) reads
`llama_get_logits_ith(ctx, seq*n_ctx)` and then walks `+ first*n_vocab` forward, and the activation
collection (`collect_imatrix` → `ggml_backend_tensor_get(src1, …)`) reads a split activation.  Neither
matches the plain `llama-perplexity` logits path.  **Verdict: pre-existing, imatrix-specific, out of
scope for the closing set — filed as a follow-up lead, not fixed here.**  A user who quantizes on a
3-GPU `-sm tensor` box with `llama-imatrix` gets a corrupt imatrix and should use `-sm layer` instead.

## OP-4(b) — `0001` `hc_combine_norm` isolated A/B: **validated (default fusion is the fast one)**

qwen4exp IQ4_NL `-p 8192,32768 -n 0 -b 2048 -ub 2048 -r 3`, interleaved, 2 reps (`S_PP t/s`):

| arm | pp8192 | pp32768 | Δ |
|---|---:|---:|---:|
| default (`hc_combine_norm` graph-optimizer fusion, the one `0001` revived) | 3367.7 | 3306.4 | — |
| `LLAMA_FUSED_DSV4_HC_POST=1` (the alternative `DSV4_HC_POST` op) | 3200.4 | 3142.9 | **-5.0 % / -4.95 %** |

So the default fusion is ~5 % faster than the op alternative on both depths.  `0001` stands.

## OP-4(c) — `0008` M=4 HC inject isolated A/B: **validated (default `TALL_MIN_M=16` holds)**

| arm | pp8192 | pp32768 | Δ |
|---|---:|---:|---:|
| default (`GGML_CUDA_MMB_TALL_MIN_M=16`) | 3367.7 | 3306.4 | — |
| `GGML_CUDA_MMB_TALL_MIN_M=0` (M=4 inject back in the tall tile) | 3347.7 | 3295.1 | **-0.59 % / -0.34 %** |

The default is (marginally) ahead, consistent with the `0008` record.  `0008` stands.

## OP-4(d) — M-RoPE image case (`0005`): **NOT RUN (tooling), deferred**

Needs the vision projector (`mmproj-Qwen3.8-Flash-Next-Q8_0.gguf`) + a text-then-image request on
3-GPU `-sm tensor`.  Left for a session with the mtmd tooling prepped; the `0005` M-RoPE guard is not
otherwise exercised here.  Mark as not-run rather than block.

---

## OP-5.2 — `0011` HC BF16 streams under `-sm tensor`: **runs now, +1.3 %, still default-OFF → park**

**The `0027` gap is closed.**  With `GGML_CUDA_HC_BLK16_DEBUG=1` and `LLAMA_HC_BLK16=1
LLAMA_HC_RES16=1`, the `graph_optimize` HC marking pass now fires under `-sm tensor` —
`HC_BLK16 comb=` is printed **582×** (564 with `comb=1`, because the Meta splitter hands
`graph_optimize` small subgraphs) where the pre-`0027` lossy-transfer record saw **0**.

**A/B** (qwen4exp IQ4_NL, `-p 8192,32768 -b 2048 -ub 2048 -r 3`, interleaved):

| arm | pp8192 | pp32768 | Δ vs default |
|---|---:|---:|---:|
| `-sm tensor` default | 3367.7 | 3306.4 | — |
| `-sm tensor` `LLAMA_HC_BLK16=1` | 3368.8 | 3301.3 | +0.03 % (inert) |
| `-sm tensor` `LLAMA_HC_RES16=1` | 3411.9 | 3349.2 | **+1.31 % / +1.30 %** |
| `-sm tensor` both | 3414.3 | 3349.3 | +1.38 % / +1.30 % |
| `-sm layer` default | 2586.6 | 2555.5 | — |
| `-sm layer` both | 2594.3 | 2563.3 | +0.30 % |

So under `-sm tensor` the win is **all from `LLAMA_HC_RES16` (the in-place BF16 residual stream)**;
`LLAMA_HC_BLK16` is inert.  A per-node trace shows why: every `HC_BLK16 blk …` candidate has
`prod=0` (the `ggml_cuda_mmb_supported_mm` / MoE-reduction producer check fails) under **both** splits,
so the block_out BF16 marks are not applied on qwen4exp in this build — the 2026-09-22
`hc-bf16-streams` record's large `blk16` win does **not** reproduce now.  The residual half is the one
that still lands.

**Decision: park.**  `0011` is lossy prefill and the maintainer asked for it to stay default-OFF, and
+1.3 % on one model with the `blk16` half dead does not change that.  Follow-up flag: the inert
`blk16` producer check on qwen4exp is a discrepancy against `2026-09-22-hc-bf16-streams.md` worth a
look if the lossy prefill is ever revisited, but it is not a closing regression.

---

## Summary

| item | result |
|---|---|
| OP-4(a) imatrix smoke | **PASS** (clean run) — but `-sm tensor` imatrix is corrupt, **pre-existing**, out of scope (new lead) |
| OP-4(b) `0001` | **validated** — default fusion +5 % over the op |
| OP-4(c) `0008` | **validated** — default `TALL_MIN_M=16` ahead |
| OP-4(d) M-RoPE image | **not run** (tooling), deferred |
| OP-5.2 `0011` `-sm tensor` | **`0027` gap closed, +1.3 % (res16 only), `blk16` inert → park** |
