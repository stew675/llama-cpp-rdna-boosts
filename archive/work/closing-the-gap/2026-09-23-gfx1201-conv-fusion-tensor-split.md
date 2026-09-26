# gfx1201 (2026-09-23) — `0004`'s depthwise conv1d fusion is not bit-identical under `-sm tensor`

**Status:** FIXED in [`patches/0004`](patches/0004-gap-closing-WIP-port-the-depthwise-conv1d-fusions-GD.patch)
(a single-device gate + `GGML_CUDA_CONV_FUSION_MULTI=1` A/B knob).  The underlying tensor-split
divergence is **not root-caused**; the fusion is simply kept off multi-device graphs until it is.

**Box:** 3× Radeon AI PRO R9700 (gfx1201), Ryzen 9 9950X3D2, ROCm `/opt/rocm-7.14.1-gfx102X`.
**Tree:** delivery r13 (`bb7b6d07…`) + `archive/work/mmb-general` (`79136a15…`) + the 25 closing patches
(integrated tree `1f09fd97…`; after this fix `1be654fa…`).

---

## The finding

`0004`'s GDN/PLE conv1d fusion matches the record on a **single device** (and on `-sm layer`), but
under `-sm tensor` it changes the greedy continuation:

| arm (27B UD-IQ3_S, prose, `-n 24`, seed 42, temp 0) | text hash |
|---|---|
| fusion default (`-sm tensor`) | `1c11d2cc3b3f` |
| `GGML_CUDA_DISABLE_CONV_FUSION=1` (upstream `SSM_CONV+SILU` fusion) | `6073add19dac` |
| `GGML_CUDA_DISABLE_FUSION=1` (raw ops) | `6073add19dac` |
| fusion default (`-sm layer`) | `6073add19dac` |
| fusion default (1 GPU) | `6073add19dac` |

The raw ops and the upstream `SSM_CONV + SILU` fusion agree byte for byte; **only the direct kernel
diverges**.  Same on the MoE: 35B-A3B UD-Q3_K_M tensor fused `697ccdeea033` vs unfused
`211639f7037d`.  The text is coherent in both arms (a greedy near-tie flip), so this is a
numerics/state divergence, not garbage.

Repro:

```sh
P=prompts/prose-rdna-boosts.txt
M=/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf
HIP_VISIBLE_DEVICES=0,1,2 ./build-rocm/bin/llama-cli -m "$M" -sm tensor -ngl 99 -fa auto \
  -ctk f16 -ctv f16 -c 8192 -n 24 --seed 42 --temp 0 --single-turn --no-display-prompt \
  --reasoning off -f "$P"                      # -> 1c11d2cc3b3f
HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_DISABLE_CONV_FUSION=1 ./build-rocm/bin/llama-cli ... # -> 6073add19dac
```

## What was checked (and ruled out)

* **Sharding is consistent.**  `GGML_CUDA_CONV_DEBUG=1` shows `-sm tensor` runs the matcher on the
  *shard* (`C=2560` / `3840`), `-sm layer`/1 GPU on the full channel count (`C=10240`).  The shard
  sizes sum to the full `10240` (3 devices); `x`, `w`, `state` are all sharded to the same `C`, and
  none is a dangling view (`w->view_src == nullptr`; `state` is a view into `[3*C, 1]`).
* **The kernel body matches the reference.**  `gdn_conv_direct_kernel` and `ssm_conv_f32<true>` use
  the same 4-tap reduction order (`sumf += x[j]*w[j]`, `j = 0..3`) and the same
  `ggml_cuda_op_silu_single`.
* **Not a cross-split consumer hole.**  A `full_graph`-aware scan for a concat consumer in another
  split (the `0023` HC16 class) does *not* fire under tensor split, so the scheduler keeps every
  `cc` consumer in the same split.
* The divergence reproduces with the fusion forced back on (`GGML_CUDA_CONV_FUSION_MULTI=1`, the new
  knob) and disappears with it off — i.e. it is the fusion, not an unrelated env.

The mechanism (why a sharded direct kernel differs from a sharded reference kernel when the inputs
look identical) is **still open**.  The leading candidate is a state/snapshot timing or view-offset
difference that only a sharded graph exposes.

## The fix

`gdn_conv_enabled()` / `ple_conv_enabled()` now also require `ggml_backend_cuda_get_device_count()
<= 1`:

```cpp
static bool gdn_conv_enabled() {
    if (getenv("GGML_CUDA_DISABLE_CONV_FUSION") != nullptr) return false;
    static const bool multi_ok = getenv("GGML_CUDA_CONV_FUSION_MULTI") != nullptr;
    return multi_ok || ggml_backend_cuda_get_device_count() <= 1;
}
```

* Single-device runs (the gfx1151/gfx1100 campaign boxes, and any `HIP_VISIBLE_DEVICES=0` gate) keep
  the fusion and its measured +3–7 % prefill.
* Multi-device graphs take the reference path (`CONCAT` + upstream `SSM_CONV+SILU`).
* `GGML_CUDA_CONV_FUSION_MULTI=1` forces the fusion on multi-device graphs for the A/B above (and for
  verifying a future root-cause fix); `GGML_CUDA_DISABLE_CONV_FUSION=1` remains the explicit off
  switch.

This is deliberately **coarser** than "tensor split only": it also disables the (bit-identical)
`-sm layer` multi-GPU fusion, because the backend has no clean tensor-vs-layer signal and purity
ranks above perf.  A finer gate is a follow-up.

## Verification

The fix was folded into the `0004` commit (`e91b89654`), the 25 patches were re-exported, and a
fresh apply of the set onto r13+beta reproduces the amended tree **`1be654fa71167e470ffcce70456bef7a23e6de25`**
(25/25 `git am`).

Post-fix text hashes (same commands as above):

| model / arm | before | after | reference (unfused) |
|---|---|---|---|
| 27B IQ3_S tensor | `1c11d2cc3b3f` | `6073add19dac` | `6073add19dac` |
| 35B Q3_K_M tensor | `697ccdeea033` | `211639f7037d` | `211639f7037d` |
| 27B IQ3_S 1 GPU | — | `6073add19dac` (fusion still fires, `GDN_CONV_DIRECT C=10240`) | — |
| 27B IQ3_S tensor `GGML_CUDA_CONV_FUSION_MULTI=1` | — | `1c11d2cc3b3f` (reproduces the divergence) | — |

## Follow-up

1. Root-cause the sharded direct-kernel divergence and, if it is a fixable bug (not just a
   non-associative reduction), remove the multi-device gate.
2. Witness: qwen4exp's PLE conv fusion is behind the same gate, so the campaign's qwen4exp prefill
   numbers measured on 3-GPU `-sm tensor` no longer include the PLE fusion (the GDN half is
   likewise off).  Re-measure the `0004` prefill delta on this box with
   `GGML_CUDA_CONV_FUSION_MULTI=1` to quantify what is being given up.
