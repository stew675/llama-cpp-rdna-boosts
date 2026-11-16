# 2026-11-15 — the prefill → decode drop region on 2 and 3 GPUs: measured, and it is not worth taking

**Verdict: REFUTED for the two configurations this campaign targets.**  On a 2-device `-sm tensor` split
the region the drop returns is **24.4 % of the arena and buys 0.08-0.42 pp of routed mass**; taking it costs
**-41 % of a 5.2 k prompt's prefill** (-4.6 % of an 84 k one) and **-3 to -6 % of decode**, and *not* taking
it — the r292 demand gate — is the right answer.  On 3 GPUs there is **no region at all**: the whole 46.9 GiB
expert set is resident at every ubatch, and the wide ubatch is free.  The campaign's premise ("keep the hot
slots above the widest prefill boundary and the cold band below it is free") does not survive the
measurement, because **a prefill routes every expert of every layer — at prefill there is no cold data**.
Two-segment slot addressing would change *which* slots survive a prefill, not the fill bill it pays.

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
85.6 % residency already covers **99.58 %** of the routed mass, so the whole 24.4 % band buys **0.42 pp** —
and a decode that misses 0.42 % less often is inside this box's noise.  The *measured* decode is **worse**
when the band is taken, by 3-6 %, because taking it grows every table and the growth is a `free` + host
re-seed of the whole table (see §3).  The prefill side of the same bytes is not noise: it is a third to a half
of the prefill.

## 2. 2 GPUs — the prefill cost is a **fill bill**, not a compute penalty (84 k tokens)

| `-ub` | prompt eval, left | prompt eval, taken | decode, left | decode, taken |
|---:|---:|---:|---:|---:|
| 4 096 | 48 105.8 ms (1 745.9 t/s) · 48 428 ms (1 734.3 t/s) | *(nothing to take)* | 46.7 / 48.3 t/s | — |
| 16 384 | **45 447.5 ms (1 848.0 t/s)** · 45 570 ms (1 842.9 t/s) | **47 752.4 ms (1 758.8 t/s)** · 47 745 ms (1 759.1) | 44.9 / 44.9 t/s | 41.7 / 42.0 t/s |

The same comparison on the short prompt (5 298 tokens) is **3 631.8 → 6 175.6 ms**, i.e. **+2 543.8 ms**; on
83 988 tokens it is **+2 304.9 ms**; at `-ub 12288` (a 14.6 % band) it is **+1 946 ms**.  So the cost is
**roughly constant per prefill and proportional to the band, ~0.25 s per GiB (3.5-4.5 GB/s)** — a *fill-rate*
bill paid once per prefill, not a per-token compute penalty.

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
* a two-segment (hot/cold) table would land at "`-ub 4096`'s decode, the current prefill bill" — strictly
  worse than just using `-ub 4096`, and it is a kernel-addressing change across every routed-op consumer.

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

## 7. Why the two-segment placement the campaign proposes cannot change the verdict

The campaign's 2026-11-15 revision replaces "freeze a tier" with "hot slots above the widest prefill
boundary, cold band below, and the real blocker is two-segment slot addressing".  The measurement says the
placement is not the blocker; **the fill bill is**:

* a prefill reads **every** expert of **every** layer it routes, so at prefill no slot is "cold".  The bill is
  per byte of non-resident expert data (§2: ~0.25 s/GiB, measured at two bands and two prompt lengths).
* two-segment tables keep the *hot* segments resident through a prefill, so the bill becomes "the cold band's
  bytes" instead of "the evicted tables' bytes" — the same band bytes, the same bill (§1 vs §4: `-ub 4096`
  reproduces the whole decode win with no band and no bill).
* their one unambiguous benefit — no whole-table re-seed at the drop — is worth `+2-3 %` decode (§3) and can
  be had from step (b) of the campaign's *contained* list without touching a kernel.

## 8. What was not tested

* **A short-arena shape** (residency ≤ 60 %, e.g. a 60-70 GiB expert set on two cards).  That is the only
  place the band's *marginal prior mass* could grow large enough (several pp) to outbid its fill bill; the
  dose-response here tops out at 0.42 pp because 85.6 % residency already covers 99.58 % of routings.
* **A server** (`--parallel 2`, repeated prefills).  §2 predicts the bill is paid per prefill, so a server
  would be strictly worse than these one-shot numbers, not better.
* **`-sm layer` on 2 GPUs** (r292 measured its band at 0.14 pp; the same conclusion follows a fortiori).
* Anything requiring the `0300` relaxed-tensor split or the optional `0100`/`0101` artifacts.

## 9. Record hygiene

`scripts/check-gpu-faults.sh --since "3 hours ago"` over the five key logs (2-device left/taken, the 84 k
taken arm, the 3-GPU arms including the aborted one): **clean**.  No delivery change was made; nothing here is
in `patches/`.  Block/knob decisions belong to the maintainer.
