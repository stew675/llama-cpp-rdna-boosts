# BF16 dense GEMMs on MMB while hipBLASLt is off (r336, contributor PR #340)

**What this records.** The `r336` block-06 amendment that takes a qwen4exp model's **BF16 dense** GEMMs on
MMB *only while* `ROCBLAS_USE_HIPBLASLT=0` (§44's determinism control), plus the RDNA4 `hc16` dead-write
removal. Every number here was measured on this box, not inherited from the contributor's report; his own
`logs/ab-r325.log` is preserved verbatim in `archive/work/mmb-bf16-pinned-r325/logs/`.

**Box / shape.** 3x R9700 (gfx1201), 2 GPUs in the arm (`HIP_VISIBLE_DEVICES=0,2`, the x8 pair) with
`-sm tensor -mg 0`.  **Both ROCm lines were measured**: §1-§7 on **ROCm 10.1.0**
(`/opt/rocm-10.1.0-gfx120X`, the line the release was assessed on), and §8 re-runs the release-critical arms on
**ROCm 10.2.0** (`/opt/rocm-10.2.0-gfx120X`), which is the line the canonical build now uses. Model
`/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/` (qwen4exp, 83.6 GB, IQ3_S with **484 BF16 dense tensors**:
the hc mixer, the BF16 router and the shared-expert gate), host experts
`--n-cpu-moe 47 -lme 47 --load-mode dio`, `MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0
GGML_CUDA_SLAB_HEADROOM_MIB=1536 GGML_CUDA_ALLREDUCE=ce`, `-c 8192 -ub 6144`.
Base = `bin-base334` (the r334 tree, `269c15e0a`); patched = the r336 tree (`213764a779`, 18 blocks).

## 1. The mechanism, by census (`GGML_CUDA_MMB_LOG=1`)

533-token prompt (`prompts/code-python.txt`), `-n 1`:

| arm | BF16 dense MMB launches | prompt t/s |
|---|---|---|
| patched, hipBLASLt **on** (default) | **0** | 493.5 |
| patched, `ROCBLAS_USE_HIPBLASLT=0` | **80** (the log's cap; every launch is `type=bf16`) | 638.7 |
| patched, pin + `GGML_CUDA_MMB_BF16_DENSE=0` | **0** | 559.2 |

So the path is unreachable with hipBLASLt enabled and the kill switch is real. A default run pays nothing.

## 2. Prefill, pinned (the headline claim)

`prompts/prose-rdna-boosts.txt` (5,246 tokens = one `-ub 6144` ubatch), `-n 0`, 3 reps each:

| arm | prompt t/s (reps) | mean |
|---|---|---|
| base, `ROCBLAS_USE_HIPBLASLT=0` | 1687.8 / 1682.1 / 1705.6 | **1691.4** |
| patched, `ROCBLAS_USE_HIPBLASLT=0` | 2211.6 / 2198.6 / 2177.6 | **2196.0** (**+29.9 %**) |
| base, hipBLASLt on | 2029.0 / 2025.5 | 2027.3 |
| patched, hipBLASLt on | 2036.4 / 2042.0 | 2039.2 (+0.6 %, i.e. noise) |

The pinned arm goes from 13-23 % *below* the default to **above** it on this shape: the pin now costs
nothing and buys determinism. (The contributor measured 1880 -> 2428 at 31.5k tokens, +29 %: the ratio
matches, the absolute numbers differ because a 5.2k prompt is launch-bound.)

## 3. Prefill, hipBLASLt on — the default path (r336's only consumer-visible change)

`/tmp/prose3.txt` (15,790 tokens = 3x the prose prompt), `-c 20480`, `-n 0`, 3 reps each:

| arm | prompt t/s | note |
|---|---|---|
| base | 2550.7 / 2617.0 / 2610.7 (2592.8) | |
| patched | 2607.5 / 2601.9 / 2602.4 (2603.9) | +0.43 % |
| patched, `GGML_CUDA_MMB_HC16=1` (the r336 hc16 change reverted) | 2593.6 / 2598.0 / 2590.5 (2594.0) | = the base |

That isolates the **dead-write removal** (the hc gate's BF16 side copy, `2*T*10240` bytes per call — 84 MB at
`T 4096` — which RDNA4 wrote and never read because the marking passes are RDNA3_5-gated): +0.4 % prefill
on the one model class that has that gate, output bit-identical (§5). Small, real, and free.

## 4. Decode (unchanged, as designed)

`prompts/prose-rdna-boosts.txt`, `-n 128`, pinned, 2 reps:

| arm | generation t/s |
|---|---|
| base | 55.2 / 55.9 |
| patched | 55.0 / 56.1 |

Flat. The BF16 dense path has a 33-token floor, so the decode/verify band cannot reach it — and the width
probe proves that rather than asserting it (§5).

## 5. Purity

**Width probe** (`tests/test-logits-width-probe`, the block-06 arm), same model, pinned, `P=4096`, `ub 2048`:

| build | width_purity | per-width `row0/row1` hashes |
|---|---|---|
| base | PASS (worst maxdiff 0), flat **and** boundary arms | `945a4ac610c4de34/…` (flat), `cd916c5f64532fc5/…` (boundary) |
| patched | PASS (worst maxdiff 0), flat **and** boundary arms | **identical to the base, W=1..8** |

**Prefill-logit gate** (`scripts/gate-prefill-logits.sh`, Flash-Next point, `--device 0,2 --sm tensor
--extra "-ncmoe 47 -lme 47 --load-mode dio" --ctx 8192 --chunks 3 --ub 4096`):

| point | base recorded from | mean KLD | same-top-p | verdict |
|---|---|---|---|---|
| **default** (hipBLASLt on, no pin) | r334 build | **-0.000000** | **100.000 %** | PASS — the default path is bit-identical |
| **pinned** (`--pin ROCBLAS_USE_HIPBLASLT=0`) | r334 build | **0.013781** | **96.427 %** | FAIL against 0.005/98 % — a family switch, see §6 |

**Kill-switch identities on the pinned point** (`--ab`, patched build; each is bit-identical, i.e. the
contributor's "this changes no output" claims hold):

| arm | mean KLD | same-top-p |
|---|---|---|
| `GGML_CUDA_MMB_SMALL_TILE=0` (0006) | -0.000000 | 100.000 % |
| `GGML_CUDA_MMB_PRODUCER_BF16=0` (0005) | -0.000000 | 100.000 % |
| `GGML_CUDA_MMB_HC16=1` (0002, reverted) | -0.000000 | 100.000 % |
| `GGML_CUDA_MMB_TINY_BF16_XH=0` (0004) | 0.014143 | 96.402 % (rounds the activation — not claimed identical) |
| `GGML_CUDA_MMB_TINY_M_BF16=0` (0003) | 0.013174 | 96.728 % (a different kernel, not claimed identical) |

`test-backend-ops -o MUL_MAT -b ROCm0` with the pin: **2691/2692**, the only failure the known
`f16, n=2105344` case (src0 is f16 — this code is not involved; the contributor reports the same case on both
of his trees).

## 6. Why the pinned FAIL is not a regression

The pin is a *different kernel family* by definition (§44). Measured with the gate's `--ab` (base arm =
pinned, candidate = default), same model and point:

| build | default vs pinned |
|---|---|
| r334 (base) | **0.015033 / 96.24 %** |
| r336 (patched) | **0.014772 / 96.41 %** |

So the pinned arm sits no further from the default than the library already was, and marginally closer; and
the change itself (0.013781) is smaller than that gap. A pinned base recorded before r336 needs one
re-record — never a cross-release hash identity claim. Details: `GREEDY-PURITY.md` §44 addendum,
`INVARIANTS.md` r336.

## 7. Not run, and why

* **MTP Protocol A** (`benchmarks/mtp-adaptive-methodology.md`): no decode-band code path is reachable —
  proven by the width probe (§5) and the flat decode numbers (§4) — so the four-axis arm would measure the
  unchanged set. The contributor's decode columns agree (107.8 -> 112.8 pinned, 114.2 -> 114.0 enabled).
* **RDNA3_5 / RDNA3_0**: no such device in this fleet. The change is unreachable there by construction
  (every new default is a field of the RDNA4 row of `mmb_arch_defaults`; the tiny-M launcher refactor is a
  pure move and the launcher's `XBF16` plumbing is inert without `tiny_m_bf16`), which is the documented,
  accepted state for a non-validated arch.
* **ROCm 10.2**: `ROCBLAS_USE_HIPBLASLT` unset + hipBLASLt 1.5 (which carries `c50ea60130f`) means the pin is
  unnecessary there and the change is inert; the 10.2 question that *is* worth a run is whether §44's
  per-start lottery still reproduces at all (`TODO` #362).


## 8. The ROCm 10.2 line (the canonical build since 2026-11-21)

Both builds were made with `~/bin/build-llama-rocm` against `/opt/rocm-10.2.0-gfx120X` (r334 tree ->
`build-rocm/bin-r334-102`, r336 tree -> `build-rocm/bin-r336-102`; 5m44s cold, then 31 s), because the
canonical build script was switched from 10.1 to 10.2 that afternoon.  Every arm below is a **10.2 vs 10.2**
comparison, so it is a clean read of this release on that line.

| arm | result |
|---|---|
| coherence, 4B Q8_0, 3-GPU `-sm tensor` | r334 `7e6229b260f6` = r336 `7e6229b260f6` (2/2 starts) — and the *same* hash as the 10.1 line |
| **prefill-logit, default env** (base recorded on 10.2 from the r334 build) | **-0.000000 / 100.000 % PASS** |
| census, Flash-Next, `GGML_CUDA_MMB_LOG=1` | **0** BF16 dense launches (hipBLASLt on) / **80** (pin) / **0** (`BF16_DENSE=0`) / **0** (r334 build) |
| pinned prefill, 5,246-token prompt, 3 reps | r334 **1553.1 / 1661.8 / 1648.3** -> r336 **2146.3 / 2168.8 / 2164.3** t/s = **+33.3 %** |

So the release's headline effect is if anything larger on 10.2 (+33.3 % vs +29.9 %), the default path is
bit-identical there too, and the coherence hash is common to both lines.

### 8.1 The §44 question on 10.2 (`TODO` #362) — measured, **inconclusive**, and that is recorded

hipBLASLt 1.5 is supposed to carry `c50ea60130f` (the per-start solution-order fix), which would retire the
pin's *necessity* on this line.  Three arms were run with hipBLASLt **left enabled**; the instrument is the
width probe's `row0` logits hash (a fresh process is a fresh library state, which is exactly §44's variable):

| arm | ROCm 10.1 | ROCm 10.2 |
|---|---|---|
| 35B-A3B UD-Q4_K_M, `P=4096`, 10 fresh starts (the GDN `ssm_alpha`/`ssm_beta` call site) | 10/10 `9a5b10b9e13d7f5b` | 10/10 `9a5b10b9e13d7f5b` |
| Flash-Next GSQ IQ3_S, `P=4096` with `MOE_EXPERT_HEATMAP=load`, 6 fresh starts (BF16 dense GEMMs on hipBLASLt) | 6/6 `cbe0a5bcbdcbed2c` | 6/6 `cbe0a5bcbdcbed2c` |

**No spread on either line, on either arm** — i.e. **vacuous**, not evidence of a fix.  Two reasons, both worth
keeping: (a) §44's *own* rule is that stability is config-dependent, so a config that does not flip says
nothing about the mechanism; (b) in the delivery the two documented call sites are no longer served by
hipBLASLt at all — `mmf32` (block 06, r104/r105, `GREEDY-PURITY.md` §47) claims the `m <= 256` F32 GDN and
router shapes *by construction*, so a fresh-start arm on them cannot flip whatever the library does.  A
decisive 10.2 test needs a shape hipBLASLt still serves **and** that is known to flip on 10.1 — the recorded
gemma4-26B-A4B MTP arm (3 logit classes in 18 starts) is the candidate, and it is a bigger arm than this
session ran.  `TODO` #362 stays open with this arm map, so the next session does not re-derive it.
