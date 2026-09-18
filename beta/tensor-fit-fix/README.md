# tensor-fit-fix — BETA (`--fit` for `--split-mode tensor`)

> **STATUS (2026-09-18): BETA — not part of the delivery set.**
> `tensor-fit.patch` applies on top of the 16-block rdna-boosts delivery
> (fork point `ebbb18522`) and closes upstream's `--fit` omission for
> `-sm tensor`.  The change is **generic llama.cpp** — no delivery block
> touches `common/fit.cpp` or `docs/multi-gpu.md` (the delivery does touch
> `ggml-backend-meta.cpp`, but only in unrelated regions) — so the likely
> home is `upstream/` as an `UPSTREAM-PR-*` candidate.  The delivery
> placement is deliberately left open.

## The gap

`common_params_fit_impl()` opened with:

```cpp
if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
    throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
}
```

so `--fit` (default **on**) was a no-op under `-sm tensor` — the exception is
caught in `common_fit_params()` and the run continues — and users had to set
`-c` / `-ngl` / `-ts` by hand.  That is documented behaviour in
`docs/multi-gpu.md`, and it dates to the original fit-params PR (#22171).

## Why it was non-trivial

Under `-sm tensor` with >1 device, `llama_prepare_model_devices()` wraps all
GPUs in a single **Meta device**; every weight/KV/compute buffer is then a
Meta buffer whose sub-buffers live on the *simple* devices.  Two consequences:

1. The existing fit algorithm is **layer-granular** — `tensor_split[id] =
   n_layer`, `ngl_t { n_layer, n_part, overflow_type }`, whole-tensor CPU
   overflow.  It cannot express "shard every layer across the devices".
2. The `no_alloc` measurement the fit relies on
   (`llama_get_memory_breakdown()` →
   `ggml_backend_alloc_ctx_tensors_from_buft_size()`) is grid-level for a Meta
   buffer type, and the Meta row **mixes semantics**:
   - `model` is the whole offloaded tensor set counted once (≈ the per-device
     total once sharded);
   - `context` / `compute` are the **largest Meta sub-buffer**, i.e. a
     per-device value;
   - `ggml_backend_dev_memory()` on the Meta device returns the **sum** of the
     simple devices.

Measured on 2× R9700 (27B Q8_0, `-c 4096`, `llama-fit-params --fit-print`):

```
Meta() 25972 330 505      # model total, per-device context, per-device compute
Host  1288  0   20
```

vs. real per-device usage from `rocm-smi` (~13996 MiB each): the Meta row has
no per-device granularity to spend.

## The fix

Two small ggml accessors let `common/fit.cpp` expand the Meta device back into
its simple devices (both were file-static before):

- `ggml_backend_dev_is_meta()`
- `ggml_backend_meta_dev_n_devs()` / `ggml_backend_meta_dev_simple_dev()`

`common_params_fit_impl()` then takes a dedicated tensor-split path:

1. expands the Meta device into the simple devices;
2. `target_i = free_i - margin_i` (per-device free from the simple devices,
   per-device margin from `--fit-target`);
3. if the user pinned `-ts`, honors it as a constraint; otherwise sets
   `tensor_split[i] ∝ target_i` (MiB scale).  A device holding ratio `r_i`
   gets `r_i` of the estimate, so the per-device condition `r_i * D <=
   target_i` reduces to a single budget `D <= min_i(target_i / r_i)` — which
   for the auto split (`r_i = target_i / sum(target)`) is exactly
   `sum(target)`.  One formula covers both cases;
4. estimates total device use as `model + n_devices * (context + compute)`
   (context/compute are per-device maxima, so multiplying is the safe,
   slightly-high direction), plus the extra (draft/MTP) model measured the
   same way;
5. if it fits, returns; otherwise reduces `n_ctx` (**auto context only** — an
   explicit `-c` is never overridden) by linear interpolation between the
   minimum and requested context; if that is still not enough, binary-searches
   `n_gpu_layers` down.

### Interaction with a user-pinned `-ts`

The documented contract is `--fit` = "auto-fit **unset** args", and `-c` is
already treated as a constraint rather than a fit variable.  A user-pinned
`-ts` is treated the same way: the fit **keeps the requested balance** and only
chooses `-c`/`-ngl` so that balance fits.  It logs
`honoring the user tensor_split ratios` and reports the
`effective budget` (`min_i(target_i / r_i)`) next to the sum of targets, so it
is visible when a lopsided `-ts` is the binding constraint.  (The layer-split
path instead aborts on a user `-ts`, because its algorithm *writes* `-ts` as
layer counts; the tensor path does not need to own it.)

The extra model's Meta device is a distinct object from the main model's, so
`add_extra_memory()` cannot attribute it by device identity; the tensor path
measures it directly (and zeroes its `model` when `shares_model`, i.e. an
embedded MTP head where the MTP context runs on the main weights).

### Patch contents

| file | change |
|---|---|
| `common/fit.cpp` | the tensor-split path (+174 lines) |
| `ggml/include/ggml-backend.h` | expose the Meta-device accessors |
| `ggml/src/ggml-backend-meta.cpp` | de-static those accessors |
| `docs/multi-gpu.md` | `--fit` now supports `tensor` |

## Validation (2026-09-18)

Hardware: 2× and 3× Radeon AI PRO R9700 (gfx1201, 32 GiB each),
`HIP_VISIBLE_DEVICES=0,1[,2]`, ROCm 7.14.  Build: clean `llama-fit-params`,
`llama-cli`, `llama-server`.

Models (all at the paths the maintainer supplied):

| model | shape |
|---|---|
| `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` | dense, no MTP; also has an **embedded** MTP head |
| `/llm/models/Qwen3.8/27B/EfficientThink/Qwen3.8-27B-EfficientThink-SimPO-Q8_0.gguf` + `mtp-Qwen3.8-27B-Q4_0.gguf` | dense + **separate** MTP head (`draft-mtp-adaptive`) |
| `/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` | **MoE**, embedded MTP head |

Results (the trace lines are `common_params_fit_impl: tensor split: ...`):

| case | fit decision | loaded? |
|---|---|---|
| 27B, 2 GPU, `-c 4096`, default | fits; `-ts 31254,31254`, `-ngl -1` | ✅ |
| 27B, 3 GPU, `-c 4096`, default | fits; `-ts 31254,31254,31254` | ✅ |
| 27B, 2 GPU, asymmetric `--fit-target 1024,8000` | `-ts 31254,24278` | ✅ |
| 27B, 3 GPU, asymmetric `--fit-target 1024,2048,8000` | `-ts 31254,30230,24278` | ✅ |
| 27B, 2 GPU, `-c 4096 --fit-target 26000,26000` | target 12556 < use 27643 → **`-ngl 27`** (est 12212) | ✅ (2.8 t/s, CPU spill) |
| 27B, 2 GPU, `--fit-target 16000,16000` (auto ctx) | 262144 → **43264** ctx (est 59901 → 32556 target) | ✅ |
| 27B, 2 GPU, `-c 262144 --fit-target 16000,16000` | explicit ctx kept → **`-ngl 34`** | — |
| 27B, embedded MTP (`--spec-type draft-mtp`), 2 GPU | fits; estimate 28309 MiB (includes MTP ctx) | ✅ |
| 27B + separate MTP head (`--spec-type draft-mtp-adaptive`), 2 GPU | fits; estimate 29698 MiB | ✅ |
| 35B-A3B MoE, embedded MTP, 2 GPU | fits; estimate 36555 MiB | ✅ |
| 35B-A3B MoE, `--fit-target 26000,26000` | → **`-ngl 14`** (est 11829) | — |
| 3 GPU, `--fit-target 30000,30000,30000` | → **`-ngl 12`** (est 6820 ≤ 6834) | — |
| 27B, 2 GPU, pinned `-ts 1,3`, default | honored; budget 41672 = `min(31254/.25, 31254/.75)` | ✅ |
| 27B, 2 GPU, pinned `-ts 1,20`, default | honored; budget 32816 ≈ `31254/.952` | ✅ |
| 27B, 2 GPU, pinned `-ts 1,3 --fit-target 26000,26000` | budget 8370 < use 27718 → **`-ngl 17`** (est 8260) | ✅ (2.3 t/s, CPU spill) |

**Balance check (hardware ground truth).**  27B Q8_0, 2 GPU, asymmetric
`--fit-target 1024,8000` → fit writes `-ts 31254,24278`; real per-device
`rocm-smi` peak during generation was **15925 MiB / 12018 MiB**, vs. the
proportional prediction 15563 / 12080 MiB (within 3 %).  The proportional
split is honoured.

**Greedy sanity.**  All loaded runs generated coherent text; the fit path does
not touch the graph or the weights (it only selects `-c`/`-ngl`/`-ts`), so no
numeric change is expected.

## Known limitations / follow-ups

- **The estimate is deliberately conservative, not exact.**  `context`/`compute`
  come from the Meta buffer's largest sub-buffer and are counted once per
  device; `model` is counted once (replicated tensors such as norms are
  under-counted by `(nd-1) x` their size).  The default 1 GiB/device margin
  absorbs both (measured error ~0.4 GiB/device on 27B Q8_0).
- **True per-device accounting** would be the clean end state: make the Meta
  buffer type expose per-simple-buffer sizes in the `no_alloc` path so
  `memory_breakdown()` can report per-device model/context/compute exactly.
  That is a larger `ggml-backend-meta.cpp` change and is the natural follow-up
  if the approximation is ever shown to under-reserve.
- **`tensor_split` rounding/rotation.**  Upstream #28506 / PR #27209 notes
  that non-divisible tensor slices can drift from the requested ratio.  The
  fit relies on the loader honouring the ratio; where it does not, the
  per-device fill diverges from the target.
- **`-ts` scale.**  When the fit *chooses* the split it writes MiB-scale
  ratios (`llama-fit-params` renders it as `uint32_t`); a user-pinned `-ts`
  is left exactly as given.
- **Pinned `-ts` can make the fit more conservative.**  A lopsided split
  lowers the effective budget (e.g. `-ts 1,20` → `min(target/r)`), so the fit
  may reduce `-c`/`-ngl` more than the auto split would.  That is the user's
  explicit choice; the log line makes the binding constraint visible.
- **Embedded vs. separate MTP.**  Validated for both; the embedded case is
  covered by the `shares_model` branch (model not double-counted).
- **Not validated:** >3 devices, non-CUDA/HIP backends, `-sm row` (still
  unsupported upstream), and the `-sm tensor` FA/`mmproj` edge cases that are
  independent of `--fit`.

## Apply + test

```bash
# from a checkout with the 16-block delivery applied (or plain upstream ebbb18522)
git apply ~/llama-cpp-rdna-boosts/beta/tensor-fit-fix/tensor-fit.patch
# rebuild

# quick check (no generation):
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-fit-params -m <model.gguf> -sm tensor -c 4096 -lv 4
# expected: "tensor split: N devices ... " then the fitted "-c ... -ngl ... -ts ..."

# force each reduction path:
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-fit-params -m <model.gguf> -sm tensor -c 4096 --fit-target 26000,26000   # -ngl <N>
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-fit-params -m <model.gguf> -sm tensor --fit-target 16000,16000           # -c <smaller>
```

See `BETA-TESTING.md` for the full tester checklist.
