# S14 — the B1-B9 gate matrix on the final gfx1201 tree

**Date:** 2026-09-21.  **Tree under test:** `mmb-port-qsa3` tip `8c686d9ef`, tree
`35fc853e6396cb0867e7e27c1e8e21093699db47` (10 patches, `git am` 10/10 onto r12 `c3ee45747`).
**Reference:** `~/llama-base` = the r12 delivery (tree `8a80535e556bef57666d2eaa4d3eb4cf93fb83f5`).
**Verdict: all gates green.  Nothing regressed; MTP works on gfx1201 for the first time.**

This is the S14 section of `gfx1201-porting.md` executed in full.  It was the campaign's biggest
coverage gap: B1-B7 had only been recorded at S1 as a delivery-vs-WIP pair, B8 was partial, and
**B9 (MTP) had never been run on gfx1201 at all.**

## 0. Hardware, software, config

3x AMD Radeon AI PRO R9700 (gfx1201, RDNA4), Ryzen 9 9950X3D2, 184 GiB.  ROCm
`/opt/rocm-7.14.1-gfx102X`.  All multi-GPU runs are `-sm tensor` with `GGML_CUDA_ALLREDUCE=hybrid`
(default).  No benches run in parallel; the page cache is warmed for the multi-shard models.

The resolved MMB config was captured before the matrix (§S14.3a) and matched the expected string
exactly:

```
MMB_CFG cc=0x1001201 dense_geom=1 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
        f32split=1(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=0 down16=0 gatemix=0
        blk16=0 res16=0 glu=1 bf16w=1 iq3xxs_glu=0 routed=0
```

So the matrix below tested the intended policy (`routed=0`, `f32split=1`, `dense_geom=1`), not a
stale tree or a stray env var.  **Every gate below ran with MMB on its default (opt-in) state,
`GGML_CUDA_MMB` unset = 0**, plus explicit `MMB=1` where the WIP's *change* is what is under test.
Note `GGML_CUDA_MMB` defaults to **0**, so "WIP, MMB off" is by construction the delivery plus the
arch-neutral groups.

## 1. B1/B2 — same-seed coherence (S14.3b)

Five reference hashes, each run on the delivery, on the WIP with MMB off and on the WIP with
`GGML_CUDA_MMB=1`.  **All three agreed on every row.**  `llama-cli --single-turn`, seed 42, temp 0,
text extracted with `scripts/extract-generated.py`.

The 27B Q8_0 needs `-lm none -lzm on` — without it `llama-cli` prints nothing and exits (the loader
takes the mmproj/vision path).  This was caught while preparing the brief and is now in it.

| model / command | delivery | WIP (MMB off) | WIP (MMB=1) |
|---|---|---|---|
| 27B Q8_0, `-n 20 -lm none -lzm on` | `110 chars da2e2d192e21` | `da2e2d192e21` | `da2e2d192e21` |
| Flash-Next, `-n 20 --reasoning off` | `35 chars 359ff4337837` | `359ff4337837` | `359ff4337837` |
| 27B UD-IQ3_S, `-n 24` | `119 chars 42cdf36d0633` | `42cdf36d0633` | `42cdf36d0633` |
| Flash-Next, `-n 24` | `135 chars d73f9238f6d6` | `d73f9238f6d6` | `d73f9238f6d6` |
| 35B UD-Q3_K_M, `-n 24` | `110 chars 461ca8cd0e88` | `461ca8cd0e88` | `461ca8cd0e88` |

Every reference hash from S1/S10-S13 reproduced exactly.  **B1/B2 green.**

## 2. B3/B4/B5 — throughput (S14.3c)

Interleaved back-to-back rounds (delivery, WIP-MMB-on, delivery, ...) in one warm session, as the
protocol requires.  **Delivery vs whole-WIP (MMB on)**; the mmb-only split is in §3.

### 2a. 27B UD-IQ3_S, 1 GPU (the S10 dense path)

| test | r | delivery | WIP MMB=1 | Δ |
|---|---|---:|---:|---:|
| pp8192 | 1 / 2 | 928.90 / 926.85 | 932.95 / 932.78 | **+0.44 / +0.64 %** |
| pp32768 | 1 / 2 | 851.55 / 851.63 | 856.73 / 856.72 | **+0.61 / +0.59 %** |

Reproduces S10's +0.52 / +0.46 %.  1-GPU numbers are very stable (delivery pp32768 varies 0.01 %
between rounds).

### 2b. 27B UD-IQ3_S decode (B4 — shallow **and** depth-16384)

| test | delivery (r1/r2/r3) | WIP MMB=1 (r1/r2/r3) |
|---|---|---|
| tg128 | 29.41 / 29.38 / 29.38 | 29.39 / 29.37 / 29.40 |
| tg128 @ d16384 | 27.94 / 27.93 / 27.93 | 27.93 / 27.93 / 27.94 |

**Flat, as designed** — MMB is prefill-only (`min_t = 512` keeps the `n_tokens <= 8` band off it).

### 2c. 35B-A3B UD-Q3_K_M, 1 GPU (MoE)

| test | r | delivery | WIP MMB=1 | Δ |
|---|---|---:|---:|---:|
| pp8192 | 1 / 2 | 5902.67 / 5891.18 | 5928.35 / 5928.12 | +0.43 / +0.63 % |
| pp32768 | 1 / 2 | 4829.86 / 4830.57 | 4855.97 / 4859.23 | +0.54 / +0.59 % |

S12 recorded this model as ≈neutral (−0.05/−0.09 %).  It measures mildly **positive** here, in all
four readings — a small improvement on the S12 landing, not a regression.

### 2d. 27B Q8_0, 3-GPU tensor (dense, type-excluded from MMB)

| test | r | delivery | WIP MMB=1 | Δ |
|---|---|---:|---:|---:|
| pp8192 | 1 / 2 | 2301.47 / 2302.09 | 2307.65 / 2309.40 | +0.27 / +0.32 % |
| tg128 | 1 / 2 | 36.71 / 36.65 | 36.69 / 36.72 | flat |

Q8_0 is excluded from the MMB weight policy, so this isolates the arch-neutral groups — slightly
positive, decode flat.

### 2e. Flash-Next IQ4_XS, q8_0 KV, 3-GPU tensor (the headline)

| depth | r | delivery | WIP MMB=1 | Δ |
|---|---|---:|---:|---:|
| pp32768 | 1 / 2 | 2371.96 / 2375.03 | 2897.25 / 2894.87 | **+22.1 / +21.9 %** |
| pp65536 | 1 / 2 | 2213.45 / 2214.36 | 2712.97 / 2711.70 | **+22.6 / +22.5 %** |
| pp98304 | 1 / 2 | 2073.46 / 2073.58 | 2556.81 / 2544.94 | **+23.3 / +22.7 %** |

The delivery figures reproduce the S1/S2 §3b record to within 0.3 % (recorded delivery: pp32768
2379, pp65536 2216, pp98304 2074), which is the check that the measurement is trustworthy — see §6
for why that mattered.

**The gap grows with depth** (G5 and qsa3 are both depth-scaling wins), which is the whole point of
the campaign: +22 % is the whole-WIP effect at depth, not a shallow artefact.

## 3. The decomposition — and the two errors it corrected

Interleaving the **same WIP binary** with `GGML_CUDA_MMB` off and on splits the +22 % cleanly:

| depth | delivery | WIP MMB off (G4+G5+G3a gate+qsa3) | WIP MMB on | mmb-only Δ |
|---|---:|---:|---:|---:|
| pp32768 | 2371.96 / 2375.03 | 2713.19 / 2709.72 | 2890.67 / 2887.36 | **+6.5 %** |
| pp65536 | 2213.45 / 2214.36 | 2552.21 / 2552.11 | 2709.83 / 2705.67 | **+6.2 %** |
| pp98304 | 2073.46 / 2073.58 | 2411.01 / 2409.25 | 2554.86 / 2555.88 | **+6.0 %** |

* **mmb alone = +6.0…+6.5 %** — this reproduces S13's recorded "+6.7/+6.5 % shallow and +6.2 % at
  64k/98k" for the S13 change, and identifies that recorded figure as the **mmb-only** delta.
* **The arch-neutral groups + qsa3 = +14.4 / +15.3 / +16.3 %** vs the delivery, growing with depth.

**Two corrections to the S14 brief (both were written from the S12/S13 "ON vs OFF" numbers, where
"OFF" meant *WIP with MMB off*, not *the delivery*):**

1. The brief's delivery Flash-Next reference row (`pp32768 ≈ 2728, pp65536 ≈ 2591, pp98304 ≈ 2465`)
   is **wrong** — those are the *qsa3-on, mmb-off* WIP numbers.  The real delivery is
   **2372 / 2213 / 2073**, which is exactly what the S1/S2 record says (2379 / 2216 / 2074) and what
   the 27B Q8_0 cross-check confirms.  The `2724` is verbatim S4's qsa3 pp32768 result.
2. The brief's "expected landed results … +6.7/+6.5 % … +6.2 %" is the **mmb-only** delta; the
   delivery-vs-WIP headline is **+22 %**.

## 4. B6 — the op oracles (S14.3d)

All four green, on a fresh build:

| oracle | result |
|---|---|
| `FLASH_ATTN_QSA` | **26 cases, 2/2 backends, all OK** — incl. the 3 `qsa3=1` arms (2x f16, 1x bf16) added in S4 |
| `GATED_DELTA_NET` | 2/2 backends passed |
| `INDEXER_TOPK` | 2/2 backends passed (the G5 oracle) |
| `FLASH_ATTN_EXT` | **0 FAIL** (see below) |

The `qsa3=1` arms passing is the important one: `FLASH_ATTN_QSA` is the *only* oracle the qsa3 WMMA
kernel has, and those three cases are the S4 addition that closed the "the kernel had no oracle on
any arch" gap.

### `FLASH_ATTN_EXT`, and why the brief's "5951/5951" is not reproducible

Four runs (three WIP, one delivery), all `exit=0`, `2/2 backends passed`, **zero FAIL**:

| run | OK | not supported (ROCm0) |
|---|---:|---:|
| delivery | 1949 | 2125 |
| WIP run 1 | 1947 | 2126 |
| WIP run 2 | 1951 | 2126 |
| WIP run 3 (same binary as run 1) | 1951 | 2126 |

The `±2-4` OK-count difference looked like a backend regression until the decisive control: **two
runs of the identical binary differ by 34 cases** (symmetric difference), while WIP-vs-delivery
differs by 40.  The case list is deterministic in the source, but `test-backend-ops` randomises part
of each case's parameters (`std::random_device`-seeded), so the OK/unsupported split moves between
runs and the counts are not comparable.  **The gate is "no FAIL", and it passes.**  The brief's
`5951/5951` came from a differently-configured build (more KV types supported rather than
"not supported") and must not be used as an expected count.

## 5. B7 — width purity (S14.3e)

`test-logits-width-probe` on 27B UD-IQ3_S, prose prompt, P=1024, ubatch 512, f16 KV:

| config | result | row-0 hash |
|---|---|---|
| MMB off | `width_purity=PASS (worst maxdiff 0)` | `04f8b6a575db6e32` |
| MMB on | `width_purity=PASS (worst maxdiff 0)` | `3e870a40c63d3f2e` |

Both `W = 1..8` groups agree exactly.  The row-0 hash differing between the two configs is expected
(MMB rounds weights to BF16 before the WMMA, so it changes prefill numerics); what matters for the
purity contract is that each config is internally width-invariant.  **B7 green.**

## 6. B8 — perplexity (S14.3f)

| model / form | delivery | WIP (MMB off) | WIP (MMB=1) |
|---|---:|---:|---:|
| 35B UD-Q3_K_M, 2 chunks | — | 14.6145 ± 0.98382 | 14.6711 ± 0.98788 (**+0.39 %**) |
| Flash-Next, prose 3x (~15.7k tok, q8_0 KV, 3-GPU) | 9.4293 ± 0.31680 | **9.4293 ± 0.31680** | 9.4314 ± 0.31641 (**+0.022 %**) |

Two things worth recording:

* **WIP (MMB off) is bit-identical to the delivery** on the long-context form (9.4293 either way).
  That is a strong result: the arch-neutral groups (G5 indexer, G4 non-temporal, the G3a gate) and
  qsa3 are **numerically neutral** at the PPL level, exactly as their bit-identical width probes
  implied.
* MMB on shifts PPL by +0.022 % (Flash-Next) and +0.39 % (MoE, CI ±0.98).  MMB is not expected to be
  bit-identical — it rounds weights to BF16 — so B8 is a *parity* gate, and it passes with room to
  spare.  A fragment-layout error would move PPL by orders of magnitude, not by 0.02 %.

**B8 green.**

## 7. B9 — MTP (S14.3g/h).  **The gate that had never run on gfx1201.**

Protocol A per `benchmarks/mtp-adaptive-methodology.md`: seed 42, temp 0, `-n 3000` (rule 0 — a short
run measures the controller's transient, not the mode), bf16 KV, `--single-turn`.

### 7a. Which models have a draft head — and the `-md` rule

This had to be checked before anything else, because the methodology and the brief disagree:

| model | built-in `nextn` head | how to run MTP |
|---|---|---|
| Qwen3.8-27B Q8_0 / UD-IQ3_S | **yes** (4 tensors, `qwen35.nextn_predict_layers`) | automatic — **do not pass `-md`** |
| Qwen3.6-35B-A3B UD-Q4_K_M / Q3_K_M | **yes** (`qwen35moe.nextn_predict_layers`) | automatic — **do not pass `-md`** |
| Qwen3.8-Flash-Next IQ4_XS | **NO** (`nextn tensors=0`, no field) | **must** pass `-md mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` |

So the brief's instruction to pass `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` is right **for qwen4exp only**;
for the dense and MoE models the methodology's "do not pass the old standalone `mtp-*.gguf`" applies.
The separate-head path is the one the r12 block-06 amendment covered, so it is worth having exercised.

### 7b. Dense — 27B UD-IQ3_S, 1 GPU, prose, `-n 3000`

| build | spec | gen t/s | acceptance | acc rate/pos | text |
|---|---|---:|---:|---|---|
| delivery | none | 28.2 | — | — | `a4fa30e462f6` |
| delivery | draft-mtp | **50.0** | 0.63624 | (0.797, 0.622, 0.490) | `a4fa30e462f6` |
| WIP | none | 28.3 | — | — | `a4fa30e462f6` |
| WIP | draft-mtp | **49.9** | 0.63624 | (0.797, 0.622, 0.490) | `a4fa30e462f6` |

MTP 50.0 vs plain 28.2 = **+77 %**.  Acceptance 0.63624 with **pos-1 = 0.797** (threshold ~0.45).
`plain == draft-mtp` **byte-identical** — that is B9b (the free purity check) passing.  And the WIP
matches the delivery on *every* value including the text hash, so the WIP's decode-adjacent changes
do not touch the dense MTP path at all.

### 7c. MoE — 35B-A3B UD-Q4_K_M, 1 GPU, prose, `-n 3000`

| build | spec | gen t/s | acceptance | acc rate/pos | text |
|---|---|---:|---:|---|---|
| delivery | none | 88.1 | — | — | `cf653c808ba3` |
| delivery | draft-mtp | **146.1** | 0.72372 | (0.856, 0.719, 0.596) | `cf653c808ba3` |
| WIP | none | 87.9 | — | — | `cf653c808ba3` |
| WIP | draft-mtp | **146.1** | 0.72372 | (0.856, 0.719, 0.596) | `cf653c808ba3` |

MTP 146.1 vs plain 88.1 = **+66 %**; acceptance 0.72372, pos-1 0.856; purity byte-identical; WIP ==
delivery exactly.  This is **ahead of the recorded 2026-09-02 MoE baseline** (plain 89.5, draft-mtp
125.8, acceptance 0.51) by +16 % on MTP throughput and +0.21 on acceptance — and the delivery shows
the same numbers, so it is the delivery's block-01/block-13 tuning carrying over, not a WIP effect.

### 7d. qwen4exp — Flash-Next IQ4_XS + `-md` head, 3-GPU tensor, `-n 3000`

| build | spec | gen t/s | acceptance | acc rate/pos | text (chars) |
|---|---|---:|---:|---|---|
| delivery | none | 49.1 | — | — | `3408262d314e` (10486) |
| delivery | draft-mtp | 76.8 | 0.64372 | (0.798, 0.632, 0.500) | `3408262d314e` |
| delivery | draft-mtp-adaptive | 76.6 | 0.64372 | (0.798, 0.632, 0.500) | `3408262d314e` |
| WIP | none | 49.3 | — | — | `cc2566a554f0` (9905) |
| WIP | draft-mtp | **80.7** | 0.70093 | (0.815, 0.700, 0.586) | `cc2566a554f0` |
| WIP | draft-mtp-adaptive | **82.4** | 0.70093 | (0.815, 0.700, 0.586) | `cc2566a554f0` |

* MTP ≥ plain: delivery 76.8 vs 49.1 (**+56 %**), WIP 80.7 vs 49.3 (**+64 %**).
* Acceptance above threshold on both; the WIP is *higher* (0.701 vs 0.644) and pos-1 0.815 vs 0.798.
  The WIP also generates +5 % faster.  (Acceptance is content-dependent, so part of this is the
  different continuation — see the axis results below for the controlled comparison.)
* `plain == draft-mtp` byte-identical **within each build** — purity holds.
* `draft-mtp-adaptive` ≈ `draft-mtp` on the prose axis (the controller sits at the floor for prose,
  as the tuning record predicts).
* **Cross-build the prose text differs** (`cc2566a554f0` vs `3408262d314e`).  That is expected and is
  the approved **qsa3 pre-baseline text change** from S4: the WIP changes prefill numerics (qsa3's
  packed WMMA path, MMB's BF16 weight rounding), so a long greedy run eventually diverges.  It is
  *not* a purity violation — purity is a within-build property (`plain == draft-mtp`), and both
  builds satisfy it.

### 7e. The four-axis gate — qwen4exp, `-n 3000`, `draft-mtp-adaptive`

Reasoning pinned per axis (rule 0): **R = on**, C/K = off.

| axis | prompt | delivery | WIP | text |
|---|---|---:|---:|---|
| C (code) | `code-python.txt` | 96.6 t/s, acc 0.90732 | 98.3 t/s, acc 0.90732 | `ca25c69383c9` **both** |
| K (recall) | `recall.txt` | 103.5 t/s, acc 1.00000 | 103.3 t/s, acc 1.00000 | `a87c4318b649` **both** |
| R (reasoning) | `reasoning.txt` | 66.7 t/s, acc 0.50377 | 67.8 t/s, acc 0.50377 | `8f46a4856d85` **both** |
| P (prose) | `prose-rdna-boosts.txt` | 76.6 t/s, acc 0.64372 | 82.4 t/s, acc 0.70093 | differs (§7d) |

* **C/K/R are byte-identical between the delivery and the WIP** (same hashes, same lengths, and
  therefore identical acceptance to 5 decimals).  Only P diverges — the open-ended prose
  continuation is the axis with the most degrees of freedom, so it is where a greedy near-tie flips
  first.  That is exactly the expected shape of the qsa3 pre-baseline change.
* The WIP is **+1.6…+1.8 %** on C and R (K is recall-bound and flat).
* Acceptance is healthy on every axis: 0.907 / 1.000 / 0.504 / 0.701.

### 7f. Rule 5 — the stock-relative verify-width gate

`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` on 27B UD-Q4_K_XL with a **q8_0 KV cache**, three
interleaved rounds (the instrument is the acceptance-free batched TG time):

| width | delivery S_TG t/s (r1/r2/r3) | WIP S_TG t/s (r1/r2/r3) |
|---|---|---|
| B=1 | 28.65 / 28.70 / 28.65 | 28.71 / 28.66 / 28.68 |
| B=4 | 77.15 / 76.50 / 76.27 | 76.71 / 76.18 / 76.37 |
| B=8 | 91.21 / 90.47 / 90.12 | 90.64 / 90.08 / 90.25 |

Within noise at every width (differences ≤0.6 % and in both directions across rounds).  **PASS** —
this is the check that caught the 2026-09-12 mmvq regression, so it is worth having re-run on the
tree that carries the whole port.

## 8. Findings, corrections and traps for the next session

1. **MTP works on gfx1201, on all three model families**, with healthy acceptance and MTP ≥ plain
   everywhere.  Nothing in the WIP perturbed it: the dense and MoE results are *byte-identical* to
   the delivery, including acceptance.
2. **A draft head cannot be loaded standalone.**  `-md <mtp-head>` together with `--spec-type none`
   aborts.  On 1 GPU it is a clean, deliberate error (`this model is an MTP draft head without a
   trunk; load it as a draft of its target model, not on its own`); on a `-sm tensor` split it trips
   `GGML_ASSERT(!suffix_fallback.empty())` at `llama-model.cpp:470` in the meta-split graph builder
   instead.  `-fit off` does **not** avoid it.  This is **pre-existing and identical on the delivery
   r12** — not a WIP regression — and the maintainer confirms it is upstream behaviour.  The
   operational rule for the harness: **the `plain` arm must not pass `-md`.**  (The standalone head
   *is* required when the model has no built-in one, i.e. qwen4exp.)
3. **Flash-Next has no built-in MTP head**; the dense 27B and both 35B-A3B models do.  Check with
   `GGUFReader` for `'nextn' in t.name` before wiring an MTP gate.
4. **The brief's Flash-Next delivery reference row was wrong** (it quoted mmb-off WIP numbers).  The
   correct delivery values are in §2e, and the S1/S2 record is the corroboration.  This is the
   second time in this campaign that "ON vs OFF" (intra-WIP) numbers were read as "WIP vs delivery".
5. **`FLASH_ATTN_EXT`'s case count is randomised run-to-run**; only "0 FAIL" is a gate.
6. **The 3-GPU qwen4exp bench drifts ±3 % at pp8192 but only ±0.1-0.4 % at 64k/98k** (S13).  The
   delivery-vs-WIP ratios here agree between rounds to 0.2 % at every depth, which is why the deep
   numbers are the ones to quote.
7. **gfx1151 is unaffected by everything in S10-S14** (S10/S11 verified the kernel set byte-unchanged
   at each step); this session changed no code, so that still holds.

## 9. What this session changed

**No code.**  S14 is a verification gate; the tree under test is unchanged at
`35fc853e6396cb0867e7e27c1e8e21093699db47` (10 patches).  Only the plan/brief and this record were
edited.  S15 (freeze, regenerate, gfx1100 hand-off) is the remaining work and needs no fix first.
