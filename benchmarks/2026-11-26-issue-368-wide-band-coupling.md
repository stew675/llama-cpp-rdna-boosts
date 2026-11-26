# Issue #368 follow-up — the wide band is armed by the prefill pass, not by the ring hole (r366)

**Claim.** r362 stopped the slab from carving its whole-table H2D ring hole on a pool-backed host-expert tier, and
that was correct: nothing stages a pool-backed routed table, which is gathered at every prefill width.  But the
hole was also — by construction — the *definition* of the slab's WIDE (prefill) band: `ring_armed`, which
`ggml_cuda_slab_wide_available()` and `ggml_cuda_slab_wide_aux_alloc()` gate on, was
`ring_region > 0 && …`.  Zeroing the hole therefore disabled the whole band, and the prefill view was then
classified *narrow* (the size/mark test also required a carved region), so it learned `narrow_floor` as its own
size.  Both are measured here on the reporter's own shape; the band is now armed by the context's wide mark and by
the boundary covering the live view, wherever that view was placed.  The hole stays declined on the pool tier.

**Why this is the reporter's axis.** GitHub issue #368 (`@DanoPTT`, r365 follow-up): on r365's default (no hole) the
90k prefill reads 821-825 t/s / 577 t/s and their ROCm0 compute buffer is **reallocated 1536 -> 2875 MiB**, while
the hole arm reads 2921-2953 t/s and never reallocates.  The 2875 MiB figure is this defect's signature — the
prefill view learning the narrow band — and the band's disappearance is the one code path that exists only in the
no-hole arm.  Their residency maps already ruled the cache out (identical line in both states), so the difference
had to be upstream of the residency policy.

## 0. The box and the shape

| item | value |
|---|---|
| GPU | 1 x R9700 (gfx1201, 32 624 MiB), ROCm 10.2.0, `HIP_VISIBLE_DEVICES=0` |
| link | PCIe 5.0 **x8** in firmware (the delivery's own probe reads 28.7 GB/s here, 55.3 on the x16 box `radix`) |
| CPU / RAM | 9950X3D, 91 GiB (the pool pins 40 012.5 MiB) |
| model | `/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/…-00001-of-00002.gguf` (75.8 GB) |
| flags | the reporter's: `--fit off -ngl all --n-cpu-moe 47 --host-experts pool --load-mode auto --lazy-mode on -c 131072 -fa on -ctk q8_0 -ctv q8_0 -b 8192 -ub 8192 -t 8 --cache-ram 4096` |
| env | `MOE_HOST_POOL_RAM_PCT=85 MOE_HOST_LIMIT_FRAC=85 GGML_CUDA_SLAB_HEADROOM_MIB=1536 MOE_GATHER_NSPLIT=4 ROCBLAS_USE_HIPBLASLT=1` |
| L1 prompt | `prompts/prose-rdna-boosts.txt` x17 -> **89 182 tokens** (the server's tokenizer count; the 2026-11-25 record's 89 199 was the cli's) |
| visit (server arms) | S1 (short, slot 1) -> L1 (89k, slot 0, n=300) -> S2 (short, slot 1) -> L2 (L1+short, slot 0, a cache hit), a fresh server per visit, `MOE_EXPERT_HEATMAP=0` (cold: no sidecar seed and no sidecar written) |
| cli arms | the same L1 file in one process, `-n 24`; they run with the model directory's persisted heat prior, which the visit arms do not — the two families are compared only within themselves |

Arms: **r365** = the shipped r365 binary (the r362 layout, no hole); **hole** = this build with
`GGML_CUDA_SLAB_RING_MIB=6144` (the r361 layout); **r366** = this build's default.

## 1. What the r362 layout actually did (diagnostics, hidden without `GGML_CUDA_SLAB_STATS=1 -lv 4`)

```
SLAB_WIDE_AUX dev=0 req=16.0 MiB refused: slab=1 ring_armed=0
FA_STAGE dev=0 stream=0 request=0.5 MiB cached=0.0 MiB alloc=16.0 MiB slab=0
SLAB_STATS wide-arm  ... work=1536.0 MiB (narrow=1792.0 ring=0.0 ...) pool=6.0 MiB
SLAB_STATS wide-drop ... work=3328.0 MiB (narrow=3328.0 ring=0.0 ...) pool=504.0 MiB ... aux_refused=924
```

* **924 `SLAB_WIDE_AUX` refusals in one 90k prefill** (0 with a ring): the band's consumers — the FA staging
  arena, the CUDA pool's large prefill allocations, the MMB activation cache, the chunked-GDN/QSA3 `wide_scratch` —
  all left the slab and drew on the ~1536 MiB outside cushion (`pool` 426 -> 504 MiB outside, plus the refusals).
* **`narrow_floor` 1792 -> 3328 MiB**, learned from the prefill view and never released: the arena loses the
  difference at every decode, and the band can never be re-armed.  That is the reporter's "1536 -> 2875 MiB".

With r366 the same run reads `aux_refused=0`, `wide[n_pop=924 …]` (the band absorbed and released those same 924
transients), `work=3648.0 MiB (narrow=1792.0 …)` and `pool=426.0 MiB`.

## 2. Measured, cli arms (3 arms, one process each)

| arm | L1 pp | L1 tg | `SLAB work` at the drop | band | `aux_refused` |
|---|---:|---:|---|---|---:|
| r365 (r362 layout) | 2073.3 / 2076.8 | 30.8 / 30.7 | 3328 MiB (**narrow=3328**) | **dead** | 924 |
| hole (`RING_MIB=6144`) | 1983.2 / 1981.5 / 1991.9 | 27.5 / 27.1 / 27.1 | 11584 MiB (1792 + 6144 + 3328) | live | 0 |
| **r366** | 2106.6 / 2107.4 / 2117.9 | 32.2 / 32.8 / 32.7 | **3648 MiB (narrow=1792** + wide 3328 + aux) | live | 0 |

Against r365: **+1.8 % prefill / +6 % decode**.  Against the hole: **+6.4 % / +19 %**.

## 3. Measured, server visits (the 2026-11-25 record's shape, 3 visits per arm, alternating)

| arm | L1 pp | L1 tg | S2 tg | L2 tg | slab-reclaim stand-downs | arena | residency |
|---|---:|---:|---:|---:|---|---:|---|
| hole (`RING_MIB=6144`) | 1888.9 (1888.1-1889.8) | 33.74 | 40.15 | 35.02 | `L31x1 … L47x3` = **51 tables** | 12416 MiB | L1..L48 261/512 |
| **r366** | **1959.5** (1958.9-1960.1) | **34.37** | **46.33** | 34.61 | `L47x2` (2 tables) | 20352 MiB | L1..L48 261/512 |

**+3.7 % L1 prefill, +1.9 % L1 decode, +15.4 % S2 decode, -1.2 % L2 decode** against the hole, with 51 whole
tables no longer stood down at every prefill.  The texts are identical at every visit point and in both layouts
(`S1/S2 f1bea30c5d55`, `L1 98c4f67d6ed9`, `L2 4b2af40814aa`), so the band's layout is not a text input on this
shape — consistent with the r362 record's eleven one-variable arms.

The L2 figure is a near-total cache hit in this harness (10 re-prefilled tokens), so it is a decode point at depth,
not the earlier record's 721-902 t/s re-prefill; the L2 *revisit* claim of that record is not re-derived here.

## 4. A second platform on a full-bandwidth link (`radix`, x16)

`radix` is a single R9700 on an uncontended PCIe 5.0 x16 link, 93 GiB: its own H2D probe reads **55.0 GB/s**
against this box's 28.7.  Its checkout carried an unrelated, uncommitted in-flight slab experiment (issue #367's
`GGML_CUDA_SLAB_ARENA_CAP_MIB`); with the maintainer's approval that was backed up
(`~/scratch/issue367-slab-experiment-backup.diff`) and the tree re-cut from `patches/` instead.  The re-cut there
reproduced the canonical ids exactly -- tip **`489810652`**, tree **`82afebd10`** -- which is a third independent
reproduction of `.tip` and the first on another machine.

| arm | L1 pp (3 visits) | L1 tg | S2 tg | L2 tg | stand-downs | arena (at the drop) |
|---|---:|---:|---:|---:|---|---:|
| hole (`RING_MIB=6144`) | 2217.5 (2216.5-2218.4) | 44.11 | 55.89 | 45.89 | `L31..L47x3` = 51 tables | 12 288 MiB |
| **r366** | **2262.9** (2262.4-2263.8) | **44.78** | **60.10** | 44.84 (43.23/45.64/45.66) | `L47x3` (3 tables) | 20 224 MiB |

**+2.0 % L1 prefill, +1.5 % L1 decode, +7.5 % S2 decode, -2.3 % L2 decode** (one 43.23 outlier; the other two
visits read 45.64/45.66 against 45.89, i.e. -0.5 %).  Texts identical (`L1 98c4f67d6ed9`, `L2 4b2af40814aa`),
residency `L1..L48 285/512` in both.  The cli arms on the same box agree: r366 **2359.3 / 2341.6** t/s prefill and
42.0 / 42.2 tg against the hole's 2264.0 / 2259.5 and 37.3 / 37.2 (**+4.1 % / +13 %**), `aux_refused=0` and
`work=3904 MiB (narrow=1792)` against the hole's 11 840.

So the fix's direction holds on a full-bandwidth link, at roughly half the x8 box's gain on prefill
(+2.0 % against +3.7 %) -- the narrower the link, the more a whole-table stand-down costs, which is the direction
the 2026-11-25 record established.  The reporter's flip was not reproduced there either (three fresh visits per arm,
one state, one text).

## 5. Gates (r366)

* `scripts/validate-set.sh` — green: base `919794500`, 19 default blocks applied strictly from a fresh tarball,
  applied tree `82afebd108d63b6220884d41080009561f0821e2`, both optional artifacts strict on the new tree, the
  rung reconciliation `0 unexplained` (my change adds no file, `#define`, env var or option).
* `scripts/gate-host-pool.sh` — **59/59 checks passed**, every arm at `83d058e8af9c` (`tsplit*` at
  `e8e834bc572a`), census printed once per arm, no GPU faults.
* `scripts/gate-kv-residency.sh` — PASS (window and attendance-pool arms, the `-sm layer` pool section, the
  per-device counters).
* `scripts/gate-prefill-logits.sh` — **mean KLD 0.000000 / same-top-p 100.000 %**.
* `scripts/gate-host-expert-text.sh` — 3/3 `7c5a512a8447`; the flipped-library arm still moves (`55a50dee05e0`).
* `scripts/check-shlib-symbols.sh` — 16/16; `scripts/check-gen-docs.sh` — ok; `scripts/check-gpu-faults.sh` —
  clean (every gate above runs it); the `all` build is green at the chain tip.
* Not run: the `gate-*-split`/probe family (no node, leaf, probe or split decision moves), `gate-uma-experts.sh`
  (the GTT store's arena is host memory and abstains from this slab), `gate-mwr-exact.sh`, `gate-lora-fusion.sh`
  (no fusion matcher or reader-count change).
