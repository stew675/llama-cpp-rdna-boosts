# `-of tp-ffn-even` — an even FFN / shared-expert tensor split (r338, block 09, opt-in)

Contributor PR #342 (`@briansp2020`, `wip/tp-ffn-even-r325/`).  This record is the delivery's own measurement of
it, on the delivery's own tree (`v17-919794500-r338`, tree `3d7017fbb`; the recorded `.tip` is `2f0d05010`), and it is what the promotion was
decided on.  The feature is **opt-in**: `-of tp-ffn-even` (or `LLAMA_TP_FFN_EVEN=1`), default OFF.

Box: gfx1201 (RDNA4), 3 x R9700 (the arms below use `HIP_VISIBLE_DEVICES=0,2` = the x8 pair, 2 devices, and
`0,1,2` for the 3-device controls), ROCm 10.2.0, `~/bin/build-llama-rocm`.

## 1. What it does

`ggml_backend_meta_split_state` (upstream tensor parallelism) splits an FFN tensor with granularity
`lcm(blck_size, 128)`, and the per-device boundary is `ne_s*(j+1)/n` floored to that granularity ("round the
granularity up to 128", for better kernels).  A 640-wide expert set at **two** devices therefore floors 320 to
**256** and splits **256 / 384** — the light GPU waits in every MoE layer's all-reduce, and the heavy one
alternates by layer:

| | device 0 | device 1 |
|---|---|---|
| `-sm tensor`, 2 devices, 640-wide, 32-block down (stock) | 256 | 384 |
| the same with `-of tp-ffn-even` | **320** | **320** |

The rule (FFN branch of `get_split_granularity` only, `src/llama-model.cpp`): with **no `-ts` (or an equal one)**
and a split-axis size that divides by the device count, halve the granularity from `lcm(blck_size,128)` until
each device's share is a multiple of it, but **never below `lcm(blck_size,64)`** — MMB declines a non-64 `K` on
the down (`mmb_supported_mm` / `mmb_supported_mmid`: `K % 64 != 0 -> false`).  Gate, up and down all take
`blck_size` from the down (`tensor_axis_0`), so the three stay on identical boundaries and only the grouping
changes.  `host-resident` slices follow, because the split state is the single source (`ffn_down_exps.bias` is
`AXIS_PARTIAL` off the same state).

One WARN per model load names it:
`tensor split: FFN granularity 128 -> 64 so blk.0.ffn_down_exps.weight splits 320 / device (opt-in: -of tp-ffn-even)`.

## 2. Where it fires (the blast radius, measured from the GGUFs themselves)

The rule is arithmetic, so the reach is enumerable.  It fires only where the per-device share is a multiple of
`lcm(blck_size,64)` but not of `lcm(blck_size,128)` — i.e. a share ≡ 64 (mod 128) for a 32-element block type.
Scanning every GGUF on this box:

| shape | devices | split |
|---|---|---|
| Flash-Next, **every** quant (GSQ IQ3_S/IQ3_XXS, IQ4_NL, Q4_K_M, UD-Q4_K_XL) **and its MTP draft models** (640 wide) | 2 | 256/384 -> 320/320 |
| `gpt-oss-20B-MXFP4_MOE` (2880 wide) | 3 | 896/1024/960 -> 960/960/960 |
| `gemma-4-26B-A4B` (2112 wide) | 3 | 640/768/704 -> 704 x3 |
| `gemma-4-26B-A4B` (2816 wide), `Inkling-2-Mini` (2304) | 4 | even |
| `Inkling-2-Mini` (1152 wide) | 2 | 512/640 -> 576/576 |

**Inert** (no WARN, text byte-identical): everything the delivery validates at 3 GPUs — 35B-A3B (512 wide),
27B/27B-BF16 (17408), 4B (9216), 9B, 122B, Qwen3-Coder (768, 256-block), DeepSeek-V4, Gemma4 12B/31B/E4B,
Ripple-65B, phi-4 — plus **Flash-Next at 3 devices** (640 % 3 ≠ 0) and **at 4 devices** (160 % 64 ≠ 0).  So the
delivery's own 3-GPU references do not move; the 2-device Flash-Next point does.

## 3. Speed (this box)

**2-device Flash-Next GSQ-RCO IQ3_S**, `-ncmoe 47 -lme 47 --load-mode dio -ub 4096` (the contributor's
`--n-cpu-moe 47` config), `MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536
GGML_CUDA_ALLREDUCE=ce`, `prompts/prose-rdna-boosts.txt` (5246 tok), `--seed 42 --temp 0`, **2 reps per arm**:

| arm | `draft-mtp -n-max 3`, `-n 256` | plain greedy `-n 128` | prefill (`draft-mtp`) |
|---|---|---|---|
| default | 70.0 / 70.8 t/s | 53.8 / 53.6 t/s | 812.8 / 810.5 t/s |
| `-of tp-ffn-even` | **74.4 / 76.3 t/s (+7.1 %)** | 53.1 / 53.5 (-0.7 %) | 794.0 / 791.3 (-2.3 %) |

Plain greedy prefill (same prompt, no drafter): **1498.7 / 1498.3 -> 1560.0 / 1556.6 t/s (+4.2 %)**, decode
53.8/53.6 -> 53.1/53.5 (-0.7 %).  The gate's own `--perf` arm (one interleaved pair) read 66.0 -> 73.5 t/s.

The contributor's claim reproduces in the shape it was made in: **decode +7 % with a drafter**, plain greedy
decode flat-to-slightly-down, prefill +4 % without a drafter / -2 % with one.  Their "+24 % on 8 short prompts"
did **not** reproduce here (+4.2 % on a 5.2 k prompt).

**2-device Flash-Next IQ4_NL** (HC16 + `GGML_CUDA_MMB_DOWN16=1` + the 32768-cell attendance pool): prefill
**1037.6 -> 1131.9 t/s (+9.1 %)**, decode 32.6 -> 34.3 (+5.2 %).

**3-device `gpt-oss-20B-MXFP4_MOE`** (all in VRAM, 3 interleaved rounds): prefill 357.1/362.4/353.4 ->
367.3/365.1/358.6 (**+1.5 %**), decode 179.6 x3 -> 182.5/181.2/183.4 (**+1.4 %**).  The 3-GPU win is ~1 %,
not 7 %: 896/1024/960 skews 7 %, while 256/384 skews 50 %.

**What it costs.**  A 320-row gate/up slice is not a multiple of 128, so those GEMMs take MMQ's bounds-checked
variant (`ggml_cuda_mmq_needs_fallback(nrows) = nrows % 128 != 0`) and lose the fused gate/up kernel in prefill
(the fusion is guarded on it at `ggml-cuda.cu`, before it is chosen).  That is the -2.3 % prefill with a
drafter; the balance still wins.

## 4. Numerics — why it is opt-in

The granularity **is** the reduction's grouping, so enabling it moves the logits of every shape it reaches.
`scripts/gate-prefill-logits.sh`, 2-device Flash-Next point (`--ctx 8192 --chunks 3 --ub 4096`, f16 KV,
`-ncmoe 47 -lme 47 --load-mode dio`):

| arm | mean KLD | same-top-p | base PPL |
|---|---|---|---|
| **`-of tp-ffn-even` vs the recorded default base** | **0.013963** (limit 0.005) | **96.467 %** (floor 98 %) | 2.6963 -> +0.59 % |
| **control: same kernel, `-ub 4096` vs `-ub 3840`** | **0.015248** | **96.516 %** | 2.7290 (+1.21 % vs the ub-4096 base) |

So the change's divergence is **smaller than a benign ubatch reordering on the same point**, and its accuracy
effect (+0.6 % PPL) is half the control's (+1.2 %).  It is a reordering, not an approximate path — but it is
**2.8x the gate's rounding-level limit**, so it cannot be default-on: a default run must not silently carry a
0.014 KLD shift on the delivery's own 2-device point.  Two more points, at 3 devices:

| point | candidate | control (one `-ub` step) |
|---|---|---|
| `gpt-oss-20B` 3 dev, **wikitext** (base PPL 5754 — unconfident, §46) | 0.033744 / 90.729 % | 0.032810 / 90.273 % |
| `gemma-4-26B-A4B` 3 dev, in-domain corpus (base PPL 2.888) | 0.059479 / 95.350 % | 0.049446 / 96.160 % |

The gemma point is the one arm where the change measures *above* its control (20 % KLD, 0.8 pp top-p), on a
**weak** control (the `-ub 2048 -> 1920` step only perturbs the last 6 % of each chunk's tokens).  Recorded as
"same class, not resolved" rather than as a clean pass — see `GREEDY-PURITY.md` §104.

**The default path is bit-identical**, which is the whole point of the opt-in:
`gate-prefill-logits.sh` on the promoted build reads **-0.000000 / 100.000 %** on the same point, and the
default greedy text is unchanged (`c420dfe04c42`, 653 chars, 2 devices, prose prompt, `-n 128`).

## 5. Text (a per-device-count reference — `GREEDY-PURITY.md` §3)

| arm | 2-dev Flash-Next, `-n 128` | 2-dev Flash-Next, `draft-mtp -n 256` | 3-dev gpt-oss, `-n 128` prose |
|---|---|---|---|
| default | 653 chars `c420dfe04c42` | 1213 chars `65a485058c41` | 565 chars `dda951d0cc18` |
| `-of tp-ffn-even` | 637 chars `cbdb63954bbb` | 1162 chars `af7114ee8a62` | 572 chars `78b0f879122c` |

Each arm is stable across reps.  3-device 35B-A3B Q8_0 is identical with and without the tag
(`a2cdef23588f`), and prints no granularity line.

## 6. Gates

* `scripts/gate-ffn-even-split.sh` (new, `MANIFESTS.md` gate 37): default inert (no WARN, recorded text), the
  tag reaches the mechanism (128 -> 64, 320/device), four spellings agree (`-of`, `LLAMA_TP_FFN_EVEN=1`,
  `LLAMA_ARG_OPTIONAL_FEATURES=tp-ffn-even`, `-of "tp-ffn-even,"`), an unknown tag is refused with the list,
  `-of list` prints it, and the 3-device 512-wide arm is untouched.  **PASS** (the `--perf` arm also passes,
  as a measurement).
* `scripts/gate-prefill-logits.sh`: **PASS -0.000000 / 100.000 %** on the default (the promotion gate).
* `scripts/validate-set.sh`: green, including both re-cut optional artifacts.
* `scripts/check-gen-docs.sh`: green (`-of` is in the three generated documents).
* 0 GPU faults in the kernel journal or any arm's log.
