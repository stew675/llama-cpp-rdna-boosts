# 2026-11-15 — the prefill → decode drop region on 2 and 3 GPUs: what it is worth, and why taking it costs

> **Corrected the same evening (see §7).**  The first write-up of this record claimed the region "cannot be
> made useful because a prefill routes every expert, so there is no cold data at prefill".  **That is wrong**
> and it is not the campaign's policy.  The bytes a band holds *can* be dropped at a prefill for free; what
> the measurement actually bills is (a) the **hot** residency a whole-table stand-down destroys and (b) the
> **transport** that re-acquires it (a demand fill runs at ~4 GB/s on the critical path, the batched seed at
> 17-33 GB/s off it — §2b).  Both belong to the *take/reclaim mechanism in the tree* (the r292 settle grows
> every table at the drop, and `moe_cache_evict_slab_range` stands a whole table down), not to the idea.  The
> measurements below stand; the verdict they support is narrower than "refuted".

**Verdict, as measured.**  On a 2-device `-sm tensor` split the region the drop returns is **24.4 % of the
arena and buys 0.08-0.42 pp of routed mass** — that ceiling is set by the shape (85.6 % residency already
covers 99.58 % of routings, and only **0.75 % of decode reaches miss**), not by the mechanism.  Taking it
with what ships **costs -41 % of a 5.2 k prompt's prefill (-4.6 % of an 84 k one) and 3-6 % of decode**, so
leaving it alone — the r292 demand gate — is right *today*.  On 3 GPUs there is **no region at all**: the
whole 46.9 GiB expert set is resident at every ubatch and the wide ubatch is free.  The campaign's plan needs
one mechanism the tree does not have: **reclaiming the band without standing a table down** (§7).

Everything below is one session's measurements on this box with the r293 delivery tip (`b950836f9`).
Harness: [`band-ab.sh`](band-ab.sh).

---

## 0. The box and the shape (identical for every arm unless a row says otherwise)

* **2 × / 3 × R9700** (`gfx1201`, 32 624 MiB each, PCIe 5.0), 9950X3D, 93 GiB RAM, ROCm 10.1.0 (`gfx120X`),
  build `b12386-b950836f9` = delivery **r293**.
* Model: `Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`
  (`-ngl all -ncmoe 47` = nearly every expert host-side), q8_0 K/V, `--load-mode mmap --lazy-mode mlockr`,
  `-t 7`, `--split-mode tensor`, `--batch-size = --ubatch-size`, `-c 2·ub+512`, greedy `--seed 42`.
* Host tier: the pinned pool (`MOE_HOST_POOL_RAM_PCT=85 MOE_HOST_LIMIT_FRAC=95 GGML_MOE_CACHE_STAGE=0`);
  the heat prior is the read-only `MOE_EXPERT_HEATMAP=load` sidecar, so it cannot move between arms.
* `ROCBLAS_USE_HIPBLASLT=0` (a fresh-process comparison, `INVARIANTS.md` §44).
* Prompt: `prompts/prose-rdna-boosts.txt` (5 246 tokens, sha `fabdec65…`), or 16 copies of it for the long
  cell (83 988 tokens, `-c 86016`).  **≥ 1 000 generated tokens** everywhere (2 000 on the headline cells).
* "band taken" = `MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP=0`, i.e. the r292 settle with its demand gate forced
  open; "band left alone" = the shipped default (the gate says DISABLED on these shapes).
* Some rows are `-lv 4` census arms (`band-ab.sh LV=4`) and some are clean text arms; the llama-cli footer
  rates (`[ Prompt: … | Generation: … ]`) and `prompt eval time` are identical either way — only the
  *generated stream* is polluted by `-lv 4`, which is why a text hash is only taken on a clean arm.

## 1. 2 GPUs — the dose-response (5 246-token prompt, 1 000-2 000 generated tokens)

The region's size is a function of `-ub`, so the trade can be read off as a curve rather than one point.

| `-ub` | band offered | marginal prior gain | arena (default) | residency | **prefill** (left / taken) | **decode** (left / taken) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 096 | **none** | 0.00 pp | 46 900.0 MiB | **100 %** | 1539.8 / 1520.7 — *(nothing to take)* | **53.8 / 54.1** |
| 8 192 | 0.4 % | 0.00 pp | 46 716.8 MiB | 99.6 % | 1521.1 / 1529.5 | 49.3 / 49.1 |
| 12 288 | 14.6 % | **0.08 pp** | 43 647.9 MiB | 93.1 % | 1520.0 → **975.8** (-35.8 %) | 52.1 → **49.1** |
| 16 384 | 24.4 % | **0.42 pp** | 40 167.0 MiB | 85.6 % | 1470.3 / 1454.9 → **856.6 / 857.3** (-41 %) | 52.2 / 52.0 → **49.5 / 49.3** |

*Interleaved repeats at `-ub 16384`: prefill 1470.3 / 1458.8 / 1454.9 / 1447.8 (left) against 857.9 / 856.6 /
857.3 / 852.5 (taken); decode 52.2 / 52.0 / 51.5 / 51.6 against 49.5 / 49.3 / 48.6 / 48.6.*  Text is
**byte-identical** between the two arms (`sha=90675acf4b72`, 6 770 chars, 2 000 tokens), so the comparison is
at fixed output.

**Reading it.**  The prior is the campaign's own instrument (the persisted decode-band heat map): it says
85.6 % residency already covers **99.58 %** of the routed mass, so the whole 24.4 % band buys **0.42 pp** of
*hit-rate* value — and the decode census agrees (`hits=5 598 121 fills=42 101` over 2 000 tokens, i.e. only
**0.75 % of reaches miss**).  That ceiling is the *shape's*, not the mechanism's: on a shape whose arena holds
half the expert set the same band is worth several pp.  What the *tree* adds on top of it is a minus 3-6 %
decode (§3) and a minus 41 % prefill (§2) — both of which are the take/reclaim mechanism (eager whole-table
growth; whole-table stand-down), not a property of a lazily-retained band (§2b, §7).

## 2. 2 GPUs — the prefill cost is a **fill bill**, not a compute penalty (84 k tokens)

| `-ub` | prompt eval, left | prompt eval, taken | decode, left | decode, taken |
|---:|---:|---:|---:|---:|
| 4 096 | 48 105.8 ms (1 745.9 t/s) · 48 428 ms (1 734.3 t/s) | *(nothing to take)* | 46.7 / 48.3 t/s | — |
| 16 384 | **45 447.5 ms (1 848.0 t/s)** · 45 570 ms (1 842.9 t/s) | **47 752.4 ms (1 758.8 t/s)** · 47 745 ms (1 759.1) | 44.9 / 44.9 t/s | 41.7 / 42.0 t/s |

The same comparison on the short prompt (5 298 tokens) is **3 631.8 → 6 175.6 ms**, i.e. **+2 543.8 ms**; on
83 988 tokens it is **+2 304.9 ms**; at `-ub 12288` (a 14.6 % band) it is **+1 946 ms**.  So the cost is
**roughly constant per prefill and proportional to the band, ~0.25 s per GiB (3.5-4.5 GB/s)** — a *fill-rate*
cost paid once per prefill, not a per-token compute penalty.

**What the fill is** (corrected, §7): not "the band has to be re-read because a prefill touches everything".
It is the **hot residency of the stood-down tables** being re-acquired — the band is a contiguous VA range at
the arena's bottom, a table is ~0.7 % of the arena, and `moe_cache_evict_slab_range` stands a table down
**whole** when any part of its range is taken, so ~58 tables lose their *hot* slots too.  Those bytes then
come back through the **demand** path at ~4 GB/s instead of the batched seed's 17-33 GB/s (§2b).

## 2b. The lazy-fill half is already built (`MOE_EXPERT_SEED`, TODO #318) — and deferring a fill is NOT free

The campaign's policy needs two things: take the band **lazily** (fill it only with bytes that are already in
flight) and drop it at the next prefill for free.  The first half exists: `MOE_EXPERT_SEED=lazy|auto` starts a
subset tier's slots **empty** and fills them on first miss, keeping the persisted prior for ranking
(`seed_defer_table()`).  It is **not the default** (the default is `eager`, i.e. the batched seed is treated as
a prefetch the demand path would otherwise pay inline), and this is why:

| arm (`-ub 16384`, 2 GPUs, 2 000 tokens) | prefill | decode |
|---|---:|---:|
| seed **eager**, band left alone (**default**) | **1454.9 t/s** | **52.2 t/s** |
| seed **eager**, band taken (the r292 settle, gate forced open) | 857.3 | 49.5 |
| seed **lazy**, band taken | 545.7 | 46.1 |
| seed **lazy**, band left alone | **115.2** | 42.9 |

All four produce the same text (`sha=90675acf4b72`).  **The bytes are identical; the transport is not.**  A
demand fill is ~4 GB/s on the critical path; the batched seed moves the same bytes at 17-33 GB/s off it, so
*deferring* the fill to first touch costs **12x the prefill** on the flat arm (1 455 → 115 t/s).  "The weight
had to come to the GPU anyway" is therefore true only for a move that was **already scheduled**: *retaining*
it afterwards is the free part, and that is the half of the policy worth building.  It also means the ±4 GB/s
"bill" in §2 is a transport bill, not an I/O bill — a batched refill of the same bytes would be 4-8x cheaper.

That is why the same change reads as **-41 %** on a 5 k prompt (the bill is 70 % of a 3.6 s prefill) and
**-4.6 %** on an 84 k one (5 % of 45 s).  Neither is a win, and a server that sees many short prefills pays
the short-prompt number every time.

## 3. Where the decode loss comes from (and why the campaign's remedy (b) cannot rescue the idea)

Taking the band re-sizes **every** table at the first drop, and a re-size is `free_table_buffers_locked()` +
`alloc_table_locked()` + the heat-ranked refill of the whole layer, not a grow-in-place: the ON arm's log
shows `re-armed 282 stood-down expert-cache tables … 47 layer(s) settled UP … (arena now 46900.0 MiB)`, and
its decode window is **38 272.8 → 40 357.4 ms** (+2.08 s over 2 000 tokens), i.e. the re-seed.

That is exactly the campaign's step **(b) "preserve-on-grow / leave the new slots empty"**.  Fixing it moves
the decode column from 49.5 to ~52-53 t/s — the rate the *same* 100 % residency reaches when it is obtained
without a re-seed (§4, `-ub 4096`: 53.8-54.5) — and **leaves the prefill column exactly where it is**.  A
`+2-3 %` decode for a `-41 %` prefill is still a loss, and step **(a) "heat-targeted growth"** does not touch
the prefill bill at all.

## 4. The same residency is free: lower `-ub` instead of taking the region

On this shape the band exists **only because `-ub` is wide**: at `-ub 4096` the birth boundary is 1.75 GiB,
the arena is the whole mapped region, and the tables are **100 % resident with no band to take**.

| shape | prefill (5.2 k) | prefill (84 k) | decode (5.2 k) | decode (84 k) |
|---|---:|---:|---:|---:|
| `-ub 4096` (100 % residency, no region) | **1539.8 / 1520.7** | 1 745.9 / 1 734.3 | **53.8 / 54.1** | 46.7 / 48.3 |
| `-ub 16384`, region left (85.6 %) | 1470.3 / 1454.9 | **1 848.0 / 1 842.9** | 52.2 / 52.0 | 44.9 / 44.9 |
| `-ub 16384`, region taken (100 %) | 856.6 / 857.3 | 1 758.8 / 1 759.1 | 49.5 / 49.3 | 41.7 / 42.0 |

So on a saturated 2-device split:

* the **region taken** is dominated on every axis — it is the residency of `-ub 4096` bought at a 41 %
  prefill bill;
* the **wide ubatch** is a genuine, small trade at long prompts (+5.9 % prefill for -4 to -6 % decode) and a
  loss at short ones (the 5.2 k prompt prefill is *faster* at `-ub 4096`);
* **as a way to get residency, lowering `-ub` is the cheaper lever today** — same (or better) decode, no band,
  no bill — but note what it does *not* buy: it cannot give 100 % residency to a server that must keep the wide
  ubatch for its prefill, and it says nothing about a short-arena shape (§7.4, §8).  A cold band that the
  prefill could reclaim **without standing a table down** would give the same residency inside `-ub 16384`;
  that is the mechanism the campaign is actually missing (§7.3), and none of these arms has it.

## 5. 3 GPUs — there is no region (and the default slab reserve aborts at this shape)

3 × R9700, same shape, `GGML_CUDA_SLAB_RESERVE_MIB=2048` (see §6):

| `-ub` | arena | residency | evictions | prefill | decode |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 46 900.0 MiB | **100 %** (216 576 slots) | 0 | 1858.7 / 1857.4 | 56.5 / 57.6 |
| 16 384 | 46 900.0 MiB | **100 %** (216 576 slots) | 0 | **1967.7 / 1959.8** | 57.6 / 57.7 |

Census: `h=1.0000`, `fills=216 576`, `evictions=0` on both, and the slab reports `va_spare 4.50 GiB/device`
unmapped.  The whole 46.9 GiB expert set fits at either ubatch, **the prefill never stands a table down**, and
the wide ubatch is free (+5.6 % prefill, decode a tie).  There is nothing to make useful here — the campaign's
target does not exist on 3 GPUs.

## 6. Two defects found on the way (neither is this campaign's, both are real)

**(a) A 3-GPU `-sm tensor` wide-prefill run aborts with the default slab reserve.**  `3 × R9700`,
`-ub 16384 -c 33280` (and `-ub 8192 -c 16896`), everything else as in §0, **no explicit reserve**:
`ggml/src/ggml-cuda/ggml-cuda.cu:1301: HipVMM Failure: out of memory`, rc 134, immediately after the prompt
echo (the VMM pool cannot `cuMemCreate` the graph's scratch).  The default at 3 GPUs is 1 536 MiB
(`1152 + 128 × n_gpu`); the slab then maps `free − 1536` per device and leaves only ~1.4 GiB outside it.
It reproduces with the campaign's host-pool environment and with a clean one, and does **not** reproduce on
2 GPUs (1 380 MiB default, same shape).  `GGML_CUDA_SLAB_RESERVE_MIB=2048` (and 4096 / 6144) runs cleanly.
So the reserve formula is one step short on this shape **and the failure is a hard abort, not the documented
`hipBLASLt` message** — a user on three cards has no way to know which knob it is.  Not changed here: it is a
block 06/09 accounting decision.

**(b) The identity-preference placement is a cliff just below full residency.**  At `-ub 8192` the arena is
99.6 % full, so `MOE_EXPERT_CACHE_IDENTITY_PREF` (r117) decides the shortfall differently:

| placement at `-ub 8192` | arena | tables | decode |
|---|---:|---:|---:|
| default (drop the coldest layers wholesale, keep identity) | 46 716.8 MiB (99.6 %) | 276 (**6 = 2 layers at 0 slots**) | **48.4 / 49.3** |
| `MOE_EXPERT_CACHE_IDENTITY_PREF=0` (uniform shortfall) | 46 304.3 MiB (98.7 %) | 282 | **52.5 / 52.5** |

**+8.5 % decode for the uniform placement** at a 0.4 % shortfall: the two dropped layers run through the
host copy for the whole decode, which costs more than the identity fast path buys.  The r117 policy was
measured at a *large* shortfall (~40 % of decode); its break-even point is much closer to full residency than
"96 % of the layers can be full".

## 7. What the measurement says about the campaign's policy (this section replaces the original refutation)

**The original §7 was wrong.**  It argued that two-segment placement cannot pay because "a prefill routes
every expert, so there is no cold data at prefill, so the band's bytes must be streamed".  The first clause
is true and the conclusion does not follow: the comparison that matters is not "band full vs band empty", it
is "band filled by data that was already moving (retained) vs no band".  In the second case the same bytes are
streamed at the *next* prefill anyway, so a lazily-retained band costs the prefill **nothing** — the prefill's
residency is then identical to the arm that never had the band.  What the −41 % bills is the *other* two
things: the hot residency destroyed by a whole-table stand-down (§2) and the demand transport that
re-acquires it (§2b).

So the policy is sound and the blocker is exactly what the campaign's 2026-11-15 revision named, only sharper:
**reclaim granularity**.  A table's slots are one contiguous range (`arena + slot × expert_bytes`), the
arena's free space is one contiguous band at the bottom, and `moe_cache_evict_slab_range()` stands any
overlapping table down **whole**.  Consequences:

1. **In-place growth cannot put a table's cold slots in the band.**  Growth must be adjacent to the table's
   existing range (the kernels index `base + slot × eb`), and an allocation is bounded above by the next table
   up; only the table that currently sits at the arena's bottom has free VA below it.  One table, not 141.
2. **So the band's contents are whole tables, not per-table cold tails.**  A 24.4 % band on a 141-table arena
   covers ~34 tables whole; taking it therefore destroys those tables' hot residency, which is the entire
   measured bill.
3. **Two mechanisms would fix it, and only one avoids the kernel addressing change:**
   * **(i) two-segment / per-slot indirection** (the campaign's plan): cold slots live in the band, the
     prefill takes them and the hot segment survives.  Cost: every routed-op consumer's address arithmetic.
   * **(ii) a band-resident *retention store* that is not table slots**: one slab allocation in the band,
     holding copies of experts the demand path already moved, hashed on `(layer, expert)`, dropped
     **wholesale** at a boundary move (so no table is ever stood down, no hot residency is touched), and hit
     by a **D2D copy** into the slot instead of a host read.  No kernel change: the hit path is the existing
     fill, and a D2D copy of one expert beats a pinned-host read by ~10-40x in latency.
4. **The prize on *this* shape is small and that is a property of the shape, not the idea.**  85.6 % residency
   already covers 99.58 % of routings, and the decode census is `hits=5 598 121 fills=42 101` over 2 000
   tokens — **0.75 % of reaches miss** — so a perfect retention store can only address that 0.75 %, i.e.
   ~1-2 % of decode (the same 0.42 pp the prior quotes, plus the D2D-vs-host latency saving).  The campaign's
   real prize is a **short-arena shape** (residency ≤ 60 %), where 5-10 % of reaches miss; that cell is still
   unmeasured (§8).

## 8. What was not tested

* **A short-arena shape** (residency ≤ 60 %, e.g. a 60-70 GiB expert set on two cards).  That is the only
  place the band's *marginal prior mass* could grow large enough (several pp) to outbid its fill bill; the
  dose-response here tops out at 0.42 pp because 85.6 % residency already covers 99.58 % of routings.
* **A server** (`--parallel 2`, repeated prefills).  §2 predicts the bill is paid per prefill, so a server
  would be strictly worse than these one-shot numbers, not better.
* **`-sm layer` on 2 GPUs** (r292 measured its band at 0.14 pp; the same conclusion follows a fortiori).
* Anything requiring the `0300` relaxed-tensor split or the optional `0100`/`0101` artifacts.
* **A retention store in the band** (§7.3 (ii)) and **two-segment slot addressing** (§7.3 (i)): neither exists,
  so neither is measured.  Both need a maintainer decision before any code lands in a block, and (ii) is the
  cheaper of the two — one slab allocation, one hit path, one drop hook, no kernel arithmetic change.
* **A batched refill of a reclaimed band** (§2b's last paragraph): the 4 GB/s bill is a *transport* number, and
  the batched seed moves the same bytes at 17-33 GB/s, so a "shrink the band back in bulk after the drop"
  arm could cut the -41 % to single digits without touching the residency question at all.

## 9. Record hygiene

`scripts/check-gpu-faults.sh --since "3 hours ago"` over the five key logs (2-device left/taken, the 84 k
taken arm, the 3-GPU arms including the aborted one): **clean**.  No delivery change was made; nothing here is
in `patches/`.  Block/knob decisions belong to the maintainer.

## 10. The prototype: a drop-band **VRAM expert store** (built, measured, neutral on this shape)

Built on the maintainer's design ("hot table + cold-band table; a hot-table miss checks the cold table before
going to host"): `~/llama.cpp` branch `wip/drop-band-store` (`da1838da1`, **not pushed, not in `patches/`**),
knobs `MOE_EXPERT_L2` (`0` = A/B) and `MOE_EXPERT_L2_MIB`.

**What it is.**  ONE slab arena allocation per device taken from the band the drop returns, partitioned into
expert-sized entries per table.  It is **never addressed as table slots** (`arena + slot × eb` in every
routed-op kernel is untouched, so no kernel changes):

* an arena eviction **retains** the victim's bytes there — a D2D copy of bytes that are about to be
  overwritten, no host I/O;
* a miss whose expert is in the store is **served from it** (D2D into the slot) instead of from the pinned host
  pool — the LRU dropped the expert, VRAM never did;
* a prefill's boundary move **frees the region whole**, so no table is stood down, no remap is rebuilt, no slot
  count changes, and the prefill sees exactly the residency it would have seen with no store at all.

**It does exactly what it says** (2 × R9700, `-ub 16384`, 5 246-token prompt, heat prior pinned):

| | store off | store on |
|---|---:|---:|
| store size (planned at the drop) | — | 3 858 + 3 876 MiB (12 955 + 13 106 entries, ~28 MiB/table) |
| **prefill** | 1 454.4 t/s | 1 451.8 t/s (**unchanged** — dropped before the boundary moves into it) |
| decode, 2 000 tokens | 52.3 t/s | 52.0 t/s |
| decode, 20 000 tokens | **52.4 t/s** | **52.4 t/s** |
| greedy text | `sha=90675acf4b72` (2 k), `93fc5debf0a7` (20 k) | **identical** |
| misses served from VRAM | — | **7 260 of 42 101 (17.2 %)** at 2 k; **10 034 of 64 976 (15.4 %)** at 20 k |
| host fill traffic | 14 254 MiB | **11 964 MiB (-16 %)** at 2 k |

**Verdict: neutral, and the reason is the shape, not the store.**  The fill path on this configuration is
already fully overlapped: 0.29-0.62 % of reaches miss at 2 k-20 k tokens, and the whole run's host traffic is
0.36 GB/s against a 28 GB/s link, so converting 17 % of the misses into a D2D copy buys nothing measurable —
the prior's 0.42 pp ceiling was right, and this is its mechanism.  The **long-generation hypothesis was tested
directly**: at 20 000 generated tokens the miss rate is *lower* (0.29 % fills, 0.20 % in-place cold reads) and
the store is still worth 0.0 % of wall time.  It is also not a prefill cost (0.0 %), which was the design goal,
so the mechanism is proven free — it simply has nothing to buy here.

**Where it could still pay** (not measured, and the reason this is not a "no"): a **short-arena shape**
(residency ≤ 60 %, misses 5-10 % of reaches, traffic ~10x) and a **genuinely agentic 40 k-token workload**,
whose routing drifts across topics instead of continuing one prose passage.  The store's own sizing is the
cheap lever there: it took 7.7 GiB of the band on this shape and had 24 600 entries against a 42 k-miss
population (only 57-109 entries ever recycled), i.e. it is *not* capacity-limited — it is demand-limited.

## 11. What a fill actually costs, and where the store starts to pay (the crossover)

The §10 shape is a full **pinned mirror**, so its fills are ~12 µs and fully overlapped.  The store pays
exactly the fill cost it avoids, so the crossover is a *fill-source* question.  Measured on 2 × R9700,
`-ub 16384`, 5 246-token prompt, 1 000-2 000 generated tokens, `sha` identical in every arm except where
noted:

| shape | arena residency | host tier / fill source | fill cost | decode (store off -> on) |
|---|---:|---|---:|---:|
| Flash-Next GSQ-IQ3_S, `-lme auto` | 85.6 % | pinned **mirror** (h=1.000) | ~12 µs (332 KiB at the 28.7 GB/s H2D calibration) | 52.3 -> 52.0 (2 k) / **52.4 -> 52.4** (20 k) |
| same, `-lme 26` + `DIO=1` | 85.6 % | 26 GiB pool — covers the arena's complement, so still pinned | ~12 µs | 40.7 -> 40.7 |
| same, **`-lme 4`** + `FILL_SRC=file DIO=1` | 85.6 % | **storage** (2 083 MiB/s, 0.45 MiB/read) | **~300 µs** | 34.0 -> **35.4 (+4.4 %)** |
| Flash-Next **Q4_K_XL** (111 GiB), `-lme 26` + file | low | storage | ~300 µs | 21.7 -> 21.8 (+0.5 %) |
| **122B-A10B Q4_K_XL** (78 GiB), `-lme auto` | **70.6 %** | pinned mirror (h=1.000) | ~12 µs | 34.9 -> 34.9 |

* **The 13 µs was the pinned case, and it is real but hidden.**  21 fills/token × 12 µs would be 1.3 % of a
  19 ms token *if serialized*; measured 0.0 %, so the async input batch absorbs them.  The user's doubt was
  right for every other tier: a **storage** fill is ~300 µs (r270's own figure is 0.7-1.6 ms per 0.65 MiB
  fill on this NVMe), and there the store's arithmetic shows up exactly — 3 584 store hits × ~307 µs = 1.08 s
  of a 29.4 s decode = the measured **+4.4 %**.
* **The conversion rate is the other factor, and it is the limiter.**  On the disk shape the store served
  **10.8 %** of arena misses (3 584 of 33 208): of 11 480 retained victims only 31 % were re-touched inside
  1 000 tokens, and the store never recycled (6 entries), so it is *demand*-limited, not capacity-limited.
  **Its ceiling on that shape is +30 %** (29 624 fills × 300 µs = 8.9 s of the 29.4 s), i.e. the store is
  worth having where fills are storage-backed, and its hit rate is what to improve.
* **A heavier model does not create the opportunity — it removes the band.**  The 122B at 70.6 % residency
  has **no drop band at all**: its tables fill the mapped arena, so `share < expert_bytes` for every table and
  the store is planned-and-dropped 8 times with zero entries (the arm is inert and says so).  The band exists
  only where the mapped arena exceeds what the tables took — which is the *same* configuration where the
  hot set is already well covered (85.6 %, prior gain 0.42 pp).  **That anti-correlation is the real
  crossover answer**: "plenty of band" and "the hot set does not fit" do not co-occur on a 2-card box.  The
  shapes that break it are (a) a **wide `-ub`** on a model that does fit (the band is `boundary_birth -
  narrow_floor`), (b) **3 GPUs** (more mapped VA per expert), and (c) a **smaller arena** (a 24 GiB card, or
  `-lme`-driven pressure) — not a bigger model.
* **Two prototype bugs found by the arms above** (both fixed, `e550d842e`): the per-table share was derived
  from the *pre-descent* allocation, so on a fragmented band the last tables' entries were carved **past the
  end of the region** — silent VRAM corruption, caught because the 122B arm generated **1 032 characters
  instead of 3 499** (a wrong-text arm is a bug report, not a slow number; the text hash is what caught it);
  and a fragmented band was refused outright instead of descending to a run that fits.
* **Not covered by the prototype, and where the disk shapes' cost actually is**: the store serves only the
  arena *fill* path.  On the disk shape the census shows **134 749 per-expert storage reads (61.1 GiB,
  29 359 ms)** on the bounce path and 24 on the cold path, against 29 624 arena fills — i.e. 4.5x the traffic
  the store can see.  Extending the store to the bounce/cold path (a cold *id* that resolves to the store
  instead of the host master) is the change that would matter on a storage-backed box, and it is a kernel-id
  change, not a copy.

## 12. State for the next session (handover)

**BLOCKED on the `--load-mode dio` / `--load-mode mmap` rework** that is being done on the second development
machine (§12.4 **P3**: this campaign's whole crossover is the fill *source*, and the arms used
`MOE_HOST_POOL_DIO` / `MOE_HOST_POOL_FILL_SRC=file` as the proxy for a storage-backed demand tier — exactly the
mechanism being reworked).  Everything below is the state as it stands, so a fresh session can resume without
re-deriving anything.

### 12.1 Where the code is

* Prototype: `~/llama.cpp` branch **`wip/drop-band-store`** (2 commits, tip `e550d842e`), cut from the r293
  delivery tip `b950836f9`.  Build with `~/bin/build-llama-rocm` after `git checkout wip/drop-band-store`.
* Portable form of the same change: [`drop-band-store.patch`](drop-band-store.patch) — sha256
  `5131f29ad3f128e0e239bf2da5654b02aae038166e5232ac470c7c9b904bef32`, 345 insertions, **one file**
  (`ggml/src/ggml-cuda/moe-expert-cache.cu`): no kernel, no header, no CMake change.  Apply to a clean r293
  tree with `git apply` — **verified**: `git apply --check` passes against `b950836f9` in a fresh worktree.
* **Nothing is in `patches/`, `beta/` or `optional/`**, nothing was pushed to the fork, and `release.json` is
  untouched.  The block decision (if this ever ships) is the maintainer's.
* Harness: [`band-ab.sh`](band-ab.sh) — one llama-cli arm per invocation.  `MODEL=`, `LME=` (`-lme`), `EXTRA=`,
  `DEVS=`, `UB=`, `CTX=`, `NGEN=`, `LV=4` for a census arm, `PROMPT=`, plus any `MOE_*`/`GGML_*`/`LLAMA_*`
  variable the caller exports (`MOE_EXPERT_L2`, `MOE_HOST_POOL_DIO`, `MOE_HOST_POOL_FILL_SRC=file`,
  `MOE_EXPERT_SEED`, …).  It prints the footer rates, `prompt eval time`, the decision/census lines and (on a
  clean arm) the generated-text hash.
* Knobs added by the prototype: `MOE_EXPERT_L2` (default `1`; **`0` is the A/B arm and is bit-for-bit the
  pre-change path**) and `MOE_EXPERT_L2_MIB` (per-device cap; `0`/unset = the whole free band).
  Census: `L2 (VRAM) expert store:` at `-lv 4` — planned size/entries per device, and at exit
  `misses served / retained / recycled / regions dropped`.

### 12.2 The measurement matrix (2 × R9700 unless stated; `-ub 16384`; 5 246-token prose prompt)

| arm | arena / residency | demand tier | store decision | decode | text |
|---|---|---|---|---|---|
| IQ3_S, `MOE_EXPERT_L2=0` (baseline) | 40 121.5 MiB / 85.6 % | pinned mirror | — | 52.3 (2 k) / 52.4 (20 k) | `90675acf…` / `93fc5deb…` |
| IQ3_S, store on | same | pinned mirror | armed, 7.7 GiB | 52.0 / 52.4 | identical |
| IQ3_S, settle forced open (`MIN_GAIN_PP=0`) | 46 900 / 100 % | pinned mirror | — | 49.5 (**-5 %**) | identical |
| IQ3_S, `MOE_EXPERT_SEED=lazy` | 40 121 / 85.6 % | pinned mirror | — | 42.9 (**prefill 115 t/s**) | identical |
| IQ3_S, `-lme 4` + `FILL_SRC=file DIO=1` | 85.6 % | **storage** (~300 µs) | armed | 34.0 → **35.4 (+4.4 %)** | identical |
| IQ3_S, `-lme 26` + `DIO=1` | 85.6 % | pinned (pool covers the complement) | armed | 40.7 → 40.7 | identical |
| Flash-Next Q4_K_XL (111 GiB), `-lme 26` + file | low | storage | armed | 21.7 → 21.8 | identical |
| 122B Q4_K_XL (78 GiB), `-lme auto` | 46 331.7 MiB / **70.6 %** | pinned mirror | **no band** (planned+dropped 8×, 0 entries) | 34.9 → 34.9 | identical |
| 3 × R9700, IQ3_S, any `-ub` | 46 900 / **100 %** | pinned mirror | inert (no band) | 56.5-57.7 | — |

### 12.3 Facts established here (do not re-derive)

1. **The band's value on a saturated 2-device split is 0.08-0.42 pp** of routed mass (85.6 % residency already
   covers 99.58 % of routings; 0.75 % of decode reaches miss at 2 k, 0.49 % at 20 k).  Taking it the way the
   tree does costs **-41 % prefill** and -5 % decode.
2. **That prefill cost is a transport bill, not the band's bytes**: the demand fill runs ~4 GB/s on the critical
   path where the batched seed runs 17-33 GB/s, and it is paid because a whole-table stand-down destroys *hot*
   residency.  A lazily-retained band would owe nothing.
3. **Deferring a fill is expensive**: `MOE_EXPERT_SEED=lazy` costs **12x the prefill** (1 455 -> 115 t/s);
   *retaining* a fill that was already scheduled is the free half.  (It is not the default for that reason.)
4. **A pinned-mirror fill is ~12 µs and fully hidden** (21 fills/token would be 1.3 % of a token if serialized;
   measured 0.0 % over 20 000 tokens).  The store pays only the fill it avoids.
5. **A storage fill is ~300 µs** (0.45 MiB at 2 083 MiB/s) and the store's saving matches it exactly
   (3 584 hits × ~307 µs = 1.08 s of 29.4 s = +4.4 %).  **Ceiling on that shape: +30 %**, hit-rate-limited.
6. **The store's hit rate is demand-limited, not capacity-limited** (6 entries recycled in 1 000 tokens; 31 %
   of retained victims re-touched).  Bigger stores do not help; a longer reuse window might.
7. **A heavier model does not create the opportunity — it removes the band.**  122B at 70.6 % residency fills
   the mapped arena, so `share < expert_bytes` and the store is inert.  "Plenty of band" and "the hot set does
   not fit" do not co-occur on two cards; the shapes that break it are a wide `-ub` on a model that fits,
   3 GPUs, or a smaller arena (24 GiB card / `-lme` pressure).
8. **The store costs the prefill 0.0 % in every arm** (prefill t/s identical to 4 digits): it is dropped whole
   before the boundary moves into it, and no table is ever stood down for it.
9. **The text hash is the correctness instrument.**  The `+30 %` that turned out to be corruption (1 032 chars
   against 3 499) was visible only as a text-length/hash change; both `a wrong-text arm is a bug report` and
   the share/allocation-mismatch bug (a descent can shrink the block, so the per-table share must be derived
   **after** the allocation) are recorded in §11.
10. **`-ub` is the cheaper way to residency on a saturated split** (`-ub 4096` = 100 % residency, best 5 k-prefill
    and best decode, §4) — the store is what you use when the wide `-ub` must stay.

### 12.4 Open items, in the order that pays

* **P1 — extend the store to the bounce/cold path.**  The only version with a large number: on the disk arm the
  census reads **134 749 per-expert storage reads (61 141.6 MiB, 29 359 ms, 2 083 MiB/s)** on the bounce path
  against 29 624 arena fills — 4.5x the traffic the store can currently see.  Serving it needs a **cold id that
  resolves to the store** instead of the host master (`moe_cache_get_cold` + the remap's cold half), i.e. a
  kernel-id change rather than a copy.  Design note: the cold id space is shared with the pinned master and the
  pool, so the store needs either its own id range or its own base pointer per table.
* **P2 — raise the hit rate** (facts 6): measure the reuse-distance curve at 1 k / 5 k / 20 k / 40 k on the
  storage arm, and consider retaining experts that arrive *through the bounce path* (they never enter the arena,
  so today they are never stored).
* **P3 — re-derive the crossover against the new load-mode vocabulary** once the rework lands, and check whether
  a sub-complement `-lme` pool reaches storage **on its own** (the `-lme 26` arm showed it covering the
  complement, so today storage needs `-lme` below the complement or `FILL_SRC=file`, which is the knob being
  reworked).
* **P4 — the 3-GPU cell** has no band at all (100 % residency), so the only 3-GPU question left is the reserve
  accounting below; there is nothing for the store to do there.
* **Carried findings, not this campaign's, still awaiting a decision** (recorded, nothing opened): the 3-GPU
  default `GGML_CUDA_SLAB_RESERVE_MIB` **abort** at a wide prefill (`HipVMM Failure: out of memory`, rc 134;
  2 048+ MiB runs — §6a), and the `-ub 8192` (99.6 % residency) `MOE_EXPERT_CACHE_IDENTITY_PREF` **cliff**
  (**-8.5 %** decode against the uniform placement — §6b).  No `TODO` handles exist for either.
