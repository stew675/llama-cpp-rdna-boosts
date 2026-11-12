# 2026-11-12 — the copy-engine all-reduce's 2-rank arm exchanges once (r272, contributor PR #304)

**`WORKLOG.md` 261112.008** + [`patches/README.md`](../patches/README.md) (block 01) +
[`ENVIRONMENT.md`](../ENVIRONMENT.md) §6 + [`INVARIANTS.md`](../INVARIANTS.md) (r272).  The claim, the arms, the
result, and the one delivery correction the measurement forced.

## What was proposed

`briansp2020`, PR #304 (`wip/ce-ar-single-r268`, +43 lines in `ggml-cuda.cu`): for `n == 2`, the bf16
large-tensor arm sends **each rank's whole partial to its peer once** and forms every element as
`fp32(bf16(fp32(own) + fp32(peer)))` in a single kernel, instead of reduce-scatter (each rank ships half its
partial, the owner adds) + all-gather of the reduced halves + `to_fp32`.  Their claim: same bits, same link
bytes, one cross-device dependency instead of two, and no separate add / convert passes.

## Is the bit-identity claim right?

Yes, by construction and by measurement:

* the rounding is the **same expression on the same two bf16 inputs** — the two-phase form's owner computes
  `bf16(fp32(a) + fp32(b))` in `ggml_cuda_ce_add_bf16`, gathers that bf16 value and widens it with `to_fp32`;
  the new kernel's cast is the same `__float2bfloat16`, and fp32 addition is commutative, so both ranks and
  both forms produce the same bits;
* **empirically, byte for byte**: `llama-perplexity --kl-divergence-base` writes the per-position log-probs as
  `uint16` over the whole vocabulary (40 chunks x 256 positions, **5 065 891 540 B**).  Recorded once per arm
  on the 2-device `-sm tensor` arm, `cmp` is **clean** — a change in any logit of any position would move the
  file;
* `gate-prefill-logits.sh --device 0,2 --sm tensor --ab "GGML_CUDA_ALLREDUCE=ce GGML_CUDA_AR_CE_SINGLE=0"`:
  **mean KLD -0.000000 / same-top-p 100.000 %** (base PPL 5.9303);
* same-seed greedy text (`prompts/prose-rdna-boosts.txt`, `-n 96`, seed 42, temp 0) `10a82b5559a9` in all four
  arms — 2 devices two-phase, 2 devices single exchange, 3 devices, and the previous release's binary.

The `n == 2` guard is also what keeps the rest untouched: 3 ranks never enter the branch
(`gate-prefill-logits.sh --device 0,1,2 --sm tensor` against a base recorded from the r270 binary reads
**0.000000 / 99.990 %**, which is exactly what that binary reads against itself), and the `q8ce` 8-bit wire
returns before the branch, so it keeps the two-phase form.

## Speed (the delivery's own 2-device arm, canonical build r272)

`llama-bench`, 27B Q8_0, `-sm tensor`, `HIP_VISIBLE_DEVICES=0,2` (the x8 pair),
`GGML_CUDA_ALLREDUCE=ce`, default `-b/-ub`, `-r 3`, **two interleaved rounds per arm** (round 1 / round 2):

| arm | pp512 | pp2048 | pp4096 | tg64 |
|---|---:|---:|---:|---:|
| two-phase (`GGML_CUDA_AR_CE_SINGLE=0`) | 2298.62 / 2289.80 | 2257.86 / 2255.13 | 2230.86 / 2229.78 | 33.44 / 33.71 |
| **single exchange (default)** | **2317.43 / 2315.44** | **2285.42 / 2284.59** | **2254.80 / 2255.58** | 33.86 / 33.50 |
| delta | +0.8 / +1.1 % | +1.2 / +1.3 % | +1.1 / +1.2 % | flat |

That is smaller than the submitter's own numbers on their cards (Flash-Next IQ3_XXS all-VRAM `-ub 2048`:
+0.3-1.0 %; IQ3_S with host experts: +2.2-5.5 %), which is the expected shape: the arm removes one kernel pass
and one cross-device dependency, so it is a larger fraction of a prefill that is not already waiting on
something else.  Decode is flat because a single-token all-reduce (20 KB) is below the CE crossover; only
batched decode (`tg64`, verify widths, `-npl > 1`) takes the CE arm at all.

**The host-expert arm is NOT resolvable on this box, and it is the arm the submitter measured.**  Two
attempts, both rejected as evidence rather than quoted:

* `llama-bench -ncmoe 47` cannot set `-lme`, so it measures the *page-cache* expert path (no pool);
* a `-lme 47` `llama-cli` arm (`Flash-Next GSQ-RCO IQ3_S`, `-ub 6144 -b 6144 --load-mode dio`, 5 246-token
  prompt, warm runs only) has a **4-6 % spread within one arm**: `ce` measured 2083.6 / 2241.8 / 2234.4 /
  2238.3 t/s across four runs, while its own two-phase counterpart measured 2128.5 / 2026.0.  Note also that
  `--load-mode dio` makes the `-lme` pool **NOT APPLIED** (`a pool needs a mapped load`), so that config is a
  file-read arm either way.

## What the same measurement did find (and fix)

Probing the host-expert arm turned up a documentation/code disagreement in `GGML_CUDA_ALLREDUCE`:

| arm (Flash-Next IQ3_S, 2 devices, `-ncmoe 47 -lme 47`, 5 246-token prompt, warm, 2 rounds) | prompt eval t/s |
|---|---:|
| unset (the delivery default: RCCL ring for host experts) | 2164.0 / 2159.3 |
| `GGML_CUDA_ALLREDUCE=auto` **before r272** (the value `ENVIRONMENT.md` named) | 2041.7 / 2051.3 |
| `GGML_CUDA_ALLREDUCE=internal` | 2159.4 / 2157.5 |
| `GGML_CUDA_ALLREDUCE=ce` | 2083.6 / 2241.8 |

The `auto` arm was **5 % slower** than the unset default because `auto` was not a value the dispatcher accepted:
it hit the unknown-value branch and selected the butterfly.  r272 makes `auto` mean the unset default (the
dual-arm chain), keeps `none` / `meta` as the butterfly by name, and says in the WARN what an unrecognised value
does.  The arm-choice question (`ce` vs RCCL for a host-expert prefill) is **not** answerable on this arm: the
`ce` arm's own spread (2083.6-2241.8) is wider than any arm-to-arm difference.  The stable statement is the one
above: **the delivery's comm layer is worth ~5 % over the butterfly here**, and which of its two large arms runs
is inside this arm's noise.  (The submitter's single-run "CE beats the RCCL default" aside is consistent with
the same chain but is not evidence for an arm choice.)

## Gates run

* `gate-prefill-logits.sh --device 0,2 --sm tensor --ab "GGML_CUDA_ALLREDUCE=ce GGML_CUDA_AR_CE_SINGLE=0"` →
  **-0.000000 / 100.000 %**; `--device 0,1,2 --sm tensor` vs the previous build's base → **0.000000 / 99.990 %**.
  The base meta now records `device=` and `sm=` (r272), and the compat check enforces them — a 1-device run
  against this 2-device base is now **refused** (`base meta mismatch: device='0,2' != current '0'`) instead of
  silently compared.
* coherence: text `10a82b5559a9` on 2 and 3 devices, equal to the previous binary's.
* MTP, the axis that reaches this arm at decode widths (`draft-mtp n3` -> a 4-row verify batch, 2 devices,
  27B UD-Q4_K_XL `-sm tensor`, `-n 3000`, `protocol A` P axis, `GGML_CUDA_ALLREDUCE=ce`): draft acceptance
  **0.62074 (1951 accepted / 3143 generated, mean len 2.86) in BOTH arms**, generation 91.6 / 91.3 t/s --
  identical, because the values are.  (Prefill on that arm 1486.7 -> 1531.7 t/s, +3.0 %.)
* kernel journal + every run log clean (`scripts/check-gpu-faults.sh`, and the gate's own per-run window).
* `scripts/validate-set.sh`, `scripts/gate-per-block-build.sh --changed-from <previous tip>`: in `WORKLOG.md`
  261112.008.

## Reproduce

```bash
# the A/B (same build, one env var)
HIP_VISIBLE_DEVICES=0,2 GGML_CUDA_ALLREDUCE=ce \
  llama-bench -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -sm tensor -p 512,2048,4096 -n 64 -r 3
  #   with GGML_CUDA_AR_CE_SINGLE=0 for the two-phase arm

# the bit-identity check (5 GB, byte-compared)
HIP_VISIBLE_DEVICES=0,2 GGML_CUDA_ALLREDUCE=ce GGML_CUDA_AR_CE_SINGLE=0 \
  llama-perplexity -m <27B Q8_0> -f /llm/models/wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 \
    -ngl 99 -fa on -ctk f16 -ctv f16 -sm tensor --kl-divergence-base /tmp/two.kld
#   repeat with GGML_CUDA_AR_CE_SINGLE=1 into /tmp/single.kld, then: cmp /tmp/two.kld /tmp/single.kld

# the arm census
  ... -lv 4   # prints: all-reduce: copy-engine 2-rank arm -- ONE exchange, ...
```

Not tested here (and not claimed): NVIDIA/CUDA, RDNA3, gfx1151, Windows, and any rank count above 2 — the
branch is `n == 2` and everything else is byte-identical by construction.
