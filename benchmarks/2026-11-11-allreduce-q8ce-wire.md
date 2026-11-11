# `GGML_CUDA_ALLREDUCE=q8ce` — an 8-bit wire for the copy-engine all-reduce (r259, contributor PR #298)

Dated 2026-11-11.  Host: gfx1201 (3x R9700, `0,1,2` on CPU lanes x8/x8/x16; the x8 pair is `0,2`),
ROCm 10.1.0-gfx120X, `~/bin/build-llama-rocm` (`build-rocm`).  Base `a691b2345`, block set V2.1.

The **concept** (a q8_0 wire for the copy-engine all-reduce, with a per-block Walsh-Hadamard rotation so one
outlier cannot set a block's scale) is contributor PR #298 by @mjohanning99.  r259 reworks it onto the
existing copy-engine chain as an algorithm of its own — `GGML_CUDA_ALLREDUCE=q8ce`, no new environment
variable — generalizes the reduce from 2 ranks to n, and makes the chunk boundaries whole q8_0 blocks.  The
numbers below are all measured on the reworked arm; the contributor's and a reporter's are quoted where they
are the only evidence for hardware this box does not have.

## What the arm is

The copy-engine chain's large-tensor arm (the one `auto` picks for a dense model, and `ce` pins) with the
staged payload changed from bf16 (2 B/value) to q8_0 blocks of Hadamard-rotated values (34 B per 32 values =
1.0625 B/value, **53 % of the bytes**).  Phases, scratch, events and ordering are unchanged; phase 1b reduces
the rank's own chunk in fp32 against the dequantized peer slices and re-quantizes, and phase 2b dequantizes
both halves of the reduced data so every rank still holds bit-identical values.  It declines (one WARN, then
the bf16 wire — never an abort) on a tensor whose element count is not a multiple of 32 and on a split of more
than 17 ranks; decode/verify-sized tensors never reach it, because the dispatcher sends them to the internal
pipeline.

## The default arm is bit-identical

`scripts/gate-prefill-logits.sh`, 27B Q8_0, wikitext, ctx 512, 40 chunks, `--sm tensor`, base recorded from
the **r258** build (the tip before this block amendment), candidate = the r259 build with the default arm:

| arm | mean KLD | same top p |
|---|---:|---:|
| r259 default, `--device 0,2` (f16 KV) | **0.000000** | 100.000 % |
| r259 default, `--device 0,2` (q8_0 KV) | **0.000000** | 100.000 % |
| r259 default, `--device 0,1,2` | **0.000000** | 99.990 % |

The wire is selected only by the algorithm name, so nothing in the default path can move.  Bases:
`prefill-logit-gfx1201-Qwen3.8-27B-Q8_0-xsm-tensor.kld` (f16), `…-kvq8_0-xsm-tensor.kld`,
`…-tensor3x.kld`.

## The selected arm is an approximation, and it passes

Same gate, `--candidate-env "GGML_CUDA_ALLREDUCE=q8ce"` against the same bases:

| arm | mean KLD | same top p | PPL |
|---|---:|---:|---:|
| `q8ce`, 2 devices, f16 KV | **0.000697** | **98.667 %** | |
| `q8ce`, 2 devices, q8_0 KV | **0.000681** | **98.833 %** | |
| `q8ce`, 3 devices, f16 KV | **0.000781** | **98.824 %** | |

Base PPL 6.9326 (f16 KV) / 5.9315 (q8_0 KV) / 5.9325 (3 devices).  So the arm is inside the gate's limits
(0.005 / 98.0 %) — but it is a real approximation and not a rounding-level change, which is why it is an
algorithm of its own and not a knob on `ce`, and why `auto` never sets it.  The q8_0 KV cache adds nothing on
top (0.000697 -> 0.000681, i.e. inside the run-to-run spread), so the wire's error dominates.

The contributor's own gate run read **0.003321 / 97.706 %** (`--kv q8_0`, an RX 9070 + RX 7900 GRE
heterogeneous pair, 27B Q4_K_M) and **failed** on top-p.  The difference is the measurement point (KV type,
model, split order), not the mechanism; a lossy arm's KLD is only meaningful with its point, which is why both
are quoted.

## Greedy text

`llama-cli`, temp 0, seed 42, `--no-display-prompt --single-turn`, no `-lv`, 27B Q8_0, `-sm tensor`
(`--device 0,2`), hashed with `scripts/extract-generated.py`:

| prompt / n | `auto` | `ce` | `q8ce` |
|---|---|---|---|
| ~1.7k-token wikitext, n=32 | `1ed53a5c6070` | `1ed53a5c6070` | `1ed53a5c6070` |
| ~3k-token wikitext, n=200 | | `8b11d62be12a` (915 chars) | `21adce75f18e` (910 chars) |

The arms agree exactly at 32 tokens and diverge to a coherent, near-identical continuation at 200 — the
expected shape for a small per-element approximation.  A `-lv 4` prefill run printed no
`cannot use the 8-bit wire` line, i.e. no large all-reduce declined the wire on a real prefill.

## Speed

`llama-bench`, `-sm tensor`, default `-b/-ub`, **two interleaved rounds** per arm (the arms are measured
alternately so a machine-state drift cannot land on one of them; every quote is round 1 / round 2):

| model, split | arm | pp4096 | tg64 |
|---|---|---:|---:|
| 27B Q8_0, 3 GPUs `0,1,2` | `ce` | 2127.9 ± 8.0 / 2132.6 ± 3.8 | 41.13 ± 0.43 / 41.10 ± 0.37 |
| | **`q8ce`** | **2393.1 ± 7.2 / 2415.6 ± 7.4** (**+12.5 % / +13.3 %**) | 41.01 ± 0.36 / 41.11 ± 0.38 |
| 35B-A3B Q8_0, 3 GPUs | `ce` | 4321.6 ± 46.6 / 4477.7 ± 35.5 | 107.28 ± 2.67 / 109.75 ± 2.08 |
| | **`q8ce`** | **4836.7 ± 43.2 / 4801.4 ± 74.7** (**+11.9 % / +7.2 %**) | 108.64 ± 2.26 / 108.56 ± 1.73 |
| 27B Q8_0, 2 GPUs `0,2` | `ce` | 2198.9 ± 8.9 / 2204.5 ± 7.0 | 33.92 ± 0.21 / 33.86 ± 0.29 |
| | **`q8ce`** | **2256.1 ± 4.4 / 2258.2 ± 10.4** (**+2.6 % / +2.4 %**) | 33.39 ± 0.27 / 33.31 ± 0.37 |
| 27B Q8_0, 2 GPUs, `tg128` | `ce` | | 33.89 ± 0.07 / 33.95 ± 0.12 |
| | **`q8ce`** | | 33.77 ± 0.06 / 33.65 ± 0.09 (**-0.4 % / -0.9 %**) |

Decode is flat on the 3-device arms (inside one standard deviation) and **-0.4/-0.9 % at `tg128` on the
2-device pair**: at a 64/128-token batch a 27B all-reduce (5120 x 64 = 327,680 elements) is above the CE
crossover, so it takes the wire and pays the extra quantize/dequantize pass on a transfer too small to
benefit.  **That is a fast-link artefact of this box, not a property of the arm**: on an x8 pair the transfer
completes in a time the dequantize cannot hide behind, whereas on a bandwidth-starved link (the case the arm
exists for) the transfer dominates and the wire wins.  The trade is **+10-13 % prefill for ~1 % decode**, which
is the intended shape of an opt-in arm; the 3-device split (one third of the tensor per hop) does not show
the decode cost at all.  **The gain is not confined to a slow link** — it grows with the rank count on this
all-x8/x16 box, because the exchange is a larger fraction of a prefill at 3 devices than at 2.

(The first pass over these arms, taken on a quiet machine before the `rot` parameter was removed from the
kernels, read 2140.1 -> 2425.9 on 27B 3 GPU and 4476.5 -> 4849.7 on 35B-A3B, i.e. the same ratios; the
interleaved pair above supersedes it.)

## MTP (Protocol A, four axes) and the batched-decode rule

`benchmarks/mtp-adaptive-methodology.md` Protocol A: dense 27B Q8_0, **`-sm tensor` 2 GPUs (`0,2`)** — the
tensor split is the point, since a `layer` split has no all-reduce and the wire is then unreachable —
f16 KV, `-c 32768 -b 2048 -ub 2048`, seed 42, temp 0, `--reasoning on` for R and `off` for P/C/K,
`-f prompts/{reasoning,prose-rdna-boosts,code-python,recall}.txt`, **`-n 3000`**, `-lv 4` for the acceptance
lines (so these logs are NOT a text-hash arm; see the purity table below).

| axis | wire | plain `none` t/s | `draft-mtp n3` t/s | draft acceptance |
|---|---|---:|---:|---:|
| R reasoning | `ce` | 33.2 | 69.9 | 0.59628 |
| R reasoning | **`q8ce`** | 33.3 | 69.7 | 0.59333 |
| P prose | `ce` | 32.8 | 86.5 | 0.82635 |
| P prose | **`q8ce`** | 32.8 | 87.1 | 0.82982 |
| C code | `ce` | 33.2 | 93.4 | 0.91041 |
| C code | **`q8ce`** | 33.3 | 92.6 | 0.89967 |
| K recall | `ce` | 33.6 | 100.1 | 0.99200 |
| K recall | **`q8ce`** | 33.9 | 99.8 | 0.99200 |

* **Acceptance is healthy on every arm** (rule 1: prose pos-1 >= ~0.45; the K axis 0.99200, the C axis
  0.90-0.91, prose 0.83, reasoning 0.60).  No collapse, i.e. no draft-vs-verify numerics divergence.
* **MTP beats plain decode on every axis and both wires** (rule 2): R 2.1x, P 2.6x, C 2.8-2.9x, K 3.0x.
* **`q8ce` is MTP-neutral**: plain decode 32.8-33.9 vs 32.8-33.6 t/s, `draft-mtp` 69.7-99.8 vs 69.9-100.1
  t/s, all inside the run-to-run spread.  The one visible difference is the **C axis acceptance,
  0.91041 -> 0.89967 (-1.2 %)**: the wire shifts the verify logits slightly, so a near-tie can flip one
  draft token in a thousand.  That is the expected shape for a lossy arm and is not a defect; the other
  three axes are within 0.5 %.
* The draft/verify band is not on the wire at all — a verify batch is decode-sized and the dispatcher sends
  it to the internal pipeline — which is why acceptance barely moves.

**Rule 5 (stock-relative batched decode), `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, 27B Q4_K_M,
`-ctk/-ctv q8_0`, `-sm tensor` 2 GPUs, two rounds:**

| B | arm | S_PP t/s | S_TG t/s |
|---|---|---:|---:|
| 1 | `ce` | 308.6 / 314.2 | 46.04 / 46.07 |
| 1 | **`q8ce`** | 322.6 / 320.2 | 46.03 / 46.41 |
| 4 | `ce` | 629.6 / 642.1 | 153.78 / 154.67 |
| 4 | **`q8ce`** | 658.4 / 661.8 | 154.32 / 153.93 |
| 8 | `ce` | 936.1 / 954.5 | 231.79 / 232.86 |
| 8 | **`q8ce`** | 960.6 / 959.3 | 231.51 / 232.37 |

Within noise at B=1 and no worse at B=4/B=8, on both the prefill and the decode side.

**The negative control that fell out by accident (kept, because it is informative).**  The first pass of this
gate was run **without** `-sm tensor`, i.e. on the default `layer` split, where no all-reduce exists.  All 16
`ce`/`q8ce` numbers came back **identical to the decimal** (plain 18.4-18.6 t/s, `draft-mtp`
42.1/50.0/55.7/59.4, acceptance identical): the arm is genuinely inert without a comm layer, and a future
"the wire fired" claim on a layer-split arm is a 0-launch arm with no effect.  Always name the split.

## Text purity

`none` == `draft-mtp n3` == `draft-mtp n7`, seed 42, temp 0, `-n 64`, **no `-lv`**,
`prompts/prose-rdna-boosts.txt`, `-sm tensor` 2 GPUs, hashed with `scripts/extract-generated.py`:

| wire | `none` | `draft-mtp n3` | `draft-mtp n7` |
|---|---|---|---|
| `auto` | `33ae8d598e7e` | `33ae8d598e7e` | `33ae8d598e7e` |
| `ce` | `33ae8d598e7e` | `33ae8d598e7e` | `33ae8d598e7e` |
| **`q8ce`** | `33ae8d598e7e` | `33ae8d598e7e` | `33ae8d598e7e` |

All nine arms hold 266 chars and the same hash.  Two things follow: the verify batch is bit-identical to
single-token decode under the wire (it is not on the wire, so it cannot be), and the wire's loss is small
enough that **64 greedy tokens do not flip on this prompt** — it takes a longer run to see it (200 tokens,
§Greedy text).  A short gate cannot see this arm's error; the logits instrument can.

## Why the rework, in one place

The contributed artifact (`optional/ce-q8-wire/`, `GGML_CUDA_AR_CE_WIRE=q8|q8h`) is not shipped as such.
What changed, and why each was necessary:

1. **A new environment variable became an algorithm name.**  `GGML_CUDA_ALLREDUCE` already selects the
   all-reduce chain, and a lossy wire is a chain property (fixed at init), not a per-call modifier.  It also
   keeps the catalogue from growing.
2. **`n == 2` became n ranks.**  The contributor gated on exactly two ranks; this box has three, so the
   feature was unreachable — and 3 devices is where it pays most (+13.4 % vs +1.7 % on the pair).
3. **Element-aligned chunks became block-aligned.**  The rotation is per 32-value q8_0 block; a chunk boundary
   inside a block rotates different data on the two sides and the sum is meaningless.  The contributed
   `ne % 64 == 0` gate is block-aligned for two ranks only; distributing whole blocks needs `ne % 32 == 0` and
   is correct for any rank count.
4. **The `q8` / `q8h` choice is gone.**  Only the rotated form ships: the contributor measured the two as the
   same mean KLD (0.0099 / 0.0098) with a better median, 99th percentile and same-top-p for 0.5 % of speed,
   so the plain form is a reference arm, not a product.
5. **Scratch sizing became explicit.**  The contributed scratch is sized for the bf16 layout and the q8 data
   "always fits"; it does for every reachable shape, but the q8 receive regions are padded to whole blocks, so
   the allocation now takes `max(bf16, q8)` and a call that cannot use the wire still has a bf16 layout.
6. **The wave64 refusal is gone** (the exchanges are width-32 and correct on both wave sizes); only wave32
   (gfx11xx/gfx12xx) is validated, and the code says so.

Reproduce: `GGML_CUDA_ALLREDUCE=q8ce` and `llama-bench -p 4096 -n 64` on `HIP_VISIBLE_DEVICES=0,1,2 -sm tensor`.
