# RESULTS 2026-11-07 (`radix`) — the crossover table, the parity answer, and where the prefill gap actually is

**Status: measured; this closes `TODO` #280 (PLAN.md P0.2/P0.3) and it found one shipping defect, fixed in
r231.**  This is the third pass of the campaign and the first one that is a *matrix* rather than a pair:
[`RESULTS-2026-11-06.md`](RESULTS-2026-11-06.md) (CLI decision cells + the field shape on `radix`),
[`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md) (the #276/#277/#278 closes on `soar`).  It ran on
**`radix`** — Ryzen 7 9800X3D, **one R9700** (gfx1201, PCIe 5.0 x16), 93 GiB RAM — inside the 56 GiB rig
([`wip/bench-56gb/`](../../../wip/bench-56gb/README.md)): a `MemoryMax=56G`/`MemorySwapMax=0` scope for the server plus a
32.4 GiB **mlocked ballast outside it**, which puts `MemAvailable` at **57.2 GiB** at load (a 59 GiB box).  No
number here is comparable with the `soar` pass (3 × R9700) or with the CLI cells in the 11-06 file.

**Environment of record.**  `~/llama.cpp` @ `2028e44cd` = `release.json` `v16-6050b467c-r230`, tree
`b336da22e`, `build-rocm` build `11903`.  Heat prior: the model-dir sidecar, **sha256 `a07822d3727d…`**, 601 376 B,
mode 444, frozen for the whole matrix (`MOE_EXPERT_HEATMAP=load`); the record's canonical `6195152358e1…` map is
**not on this box** (`README.md` §3), so the *absolute* numbers are this prior's and only within-run comparisons
travel.  Every arm: one model load, `-lv 4`, one arm at a time, `scripts/check-gpu-faults.sh` clean afterwards.

---

## 1. What the matrix is, and the two numbers per prefill

The field config of the rig (`runme-56gb-tier.sh`: Flash-Next GSQ-RCO-IQ3_S + its MTP draft, `-c 204800`,
`-ub 6144`, q8_0 KV, `-cram 1024`, `--parallel 1`, `-ncmoe 48`, `MOE_HOST_POOL_RAM_PCT=0`).  Per arm:

* three 600-token decode passes (`gen3` below is the third: the steady state);
* the **first-pass 50k prefill** (`pp50k#1`) — the client-visible `prompt eval time`, i.e. TTFT;
* a **second** 50k pass (`pp50k#2`) — the same prompt warm, which separates a tier's own miss path from the
  page-cache state;
* `MiB` = the server's `/proc/<pid>/io read_bytes` delta for that stage (a page-cache read is not a storage
  read, so this is the honest "storage MiB per request");
* **`tail s`** = `pp50k - proc`, where `proc` is the rate at the last `progress = 1.00` row.  The difference is
  the **band-transition tax**: the pool is re-laid-out (`pool_rerank_from_heat_locked`) and stood-down arena
  tables re-arm *after* the last ubatch, i.e. it is not prefill compute but the client pays it before the first
  token.  It turns out to be the single largest term in the field prefill gap — see §3.

The page cache is dropped (`echo 3 > drop_caches`) before every arm, so every arm starts cold; the `fills`
column is the pool's own count of `pool_rerank_from_heat_locked` slot fills over the whole run.

## 2. The crossover table (`radix`, 56 GiB rig, r230, one load per arm)

| arm | tier | slots | pinned GiB | gen3 t/s | gen MiB | pp50k#1 t/s | tail s | MiB | pp50k#2 t/s | MiB | fills | demoted | reuse % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **M1** | `-lme 47` **mirror** | 512/512 | 46.8 | **83.8** | 95 | **1 895** | 0.6 | 683 | 1 762 | 798 | 0 | 0 | — |
| M1w | mirror, `SINK=0` | 512/512 | 46.8 | 83.7 | 220 | 1 939 | 0.5 | 649 | 2 104 | 0 | 0 | 0 | — |
| A1 | `auto` **97.9 %** | 497/512 | 45.5 | 75.6 | 847 | 1 431 | 7.5 | 2 116 | 1 499 | 987 | 4 338 | 0 | — |
| A0 | `auto`, `SINK=0` | 495/512 | 45.3 | 75.5 | 965 | 1 810 | 1.3 | 2 215 | 2 026 | 403 | 4 884 | 0 | — |
| A2 | `auto`, `DIO=1` | 495/512 | 45.3 | 67.7 | 2 162 | 1 818 | 1.2 | 11 216 | 1 983 | 7 962 | 4 884 | 0 | — |
| A3 | `auto`, `SINK=0`, `DIO=1` | 493/512 | 45.1 | 67.9 | 2 378 | 1 785 | 1.3 | 12 534 | 1 978 | 8 884 | 5 442 | 0 | — |
| B1 | `-lme 44` **93.8 %** | 480/512 | 43.9 | 76.2 | 698 | **735** | 22.1 | 14 809 | 1 147 | 3 268 | 8 154 | 49 857 | 68.7 |
| B0 | `-lme 44`, `SINK=0` | 480/512 | 43.9 | 74.7 | 1 774 | 1 191 | 14.2 | 3 376 | 1 469 | 497 | 9 050 | 0 | — |
| B1d | `-lme 44`, `DIO=1` | 480/512 | 43.9 | 75.9 | 898 | **1 329** | 1.6 | 31 856 | 1 633 | 25 989 | 7 979 | 49 857 | 68.7 |
| B0d | `-lme 44`, `SINK=0`, `DIO=1` | 480/512 | 43.9 | 62.7 | 3 925 | 1 699 | 1.8 | 21 126 | 1 934 | 14 904 | 9 049 | 0 | — |
| C1 | `-lme 36` **76.8 %**, `DIO=1` | 393/512 | 36.0 | 76.1 | 2 236 | 1 037 | 3.6 | 66 057 | 1 133 | 59 652 | 23 630 | 50 793 | 67.6 |
| C0 | `-lme 36`, `SINK=0`, `DIO=1` | 393/512 | 36.0 | **45.4** | 19 011 | 1 000 | 4.1 | 75 624 | 1 161 | 54 902 | 24 095 | 0 | — |
| C1x | `-lme 36`, `SPAN=0` | 393/512 | 36.0 | 76.1 | 2 235 | 958 | 3.3 | 66 230 | 1 129 | 59 772 | 23 819 | 50 793 | 67.6 |

Reps (§5 has the full set): `A1`/`A1b` = 1 431/1 331 t/s · `A0`/`A0b` = 1 810/1 329 · `B1d` (one rep, the
DIO path is stable) · `C1`/`C1b` = 1 037/1 032 and 76.06/76.07 gen · `C0`/`C0b` = 1 000/995 and 45.43/45.56 gen.
**The `-lme 36` cell reproduces to the t/s and to the MiB**; the near-mirror cell's *prefill* does not (± 15 %).

### 2.1 Decode — flat from 98 % to 77 %, and the demote is what makes it flat

| tier | pinned GiB | gen3 t/s | vs mirror |
|---|---|---|---|
| mirror 100 % | 46.8 | 83.8 | — |
| `auto` 97.9 % | 45.5 | 75.6 | −10 % |
| `-lme 44` 93.8 % | 43.9 | 76.2 / 74.7 | −9 % |
| `-lme 36` 76.8 % (demote on) | 36.0 | **76.1** | **−9 %** |
| `-lme 36` 76.8 % (demote **off**) | 36.0 | **45.4** | **−46 %** |

The tier curve in *decode* is flat from 98 % to 77 % coverage — 10.8 GiB of pinned RAM saved for 9 % of decode
— but only with the eviction demote: the census says why.  At 76.8 % the demote made **50 793 copies (32.2 GiB)
with a 67.6 % reuse rate** and the decode turn read 2.2 GiB from storage; with `SINK=0` the same turn read
**19.0 GiB** (10.5 MiB/token) and decode fell to 45.4.  This is the campaign's strongest result: on the *field
shape* (MTP drafting, `-ub 6144`) the demote is worth **+67 % decode** at 77 % coverage, not the +8 % the CLI
cells of the 11-06 pass measured — a spec-decode step verifies four tokens at once, so the experts it re-uses
are exactly the ones a demote captures.

The mirror is still the fastest single configuration (+10 %, 0 MiB of storage traffic) for decode, and it is
**inert** for the demote (M1/M1w identical: 0 demotions, 144 mirror tables).

### 2.2 Prefill — monotone in coverage *once the fill mode is held fixed*, and the fill mode is the lever

| tier | buffered fills (`--load-mode mmap`) | `MOE_HOST_POOL_DIO=1` |
|---|---|---|
| mirror | 1 895 / 1 939 t/s, 0.6 s tail, 0.7 GiB | (n/a: a mirror fills nothing at prefill) |
| `auto` 97.9 % | 1 431 ± 100, **7.5 s tail**, 2.1 GiB | **1 818**, 1.2 s tail, 11.2 GiB |
| `-lme 44` 93.8 % | **735** (`SINK=1`) / 1 191 (`SINK=0`), **22.1 / 14.2 s tail**, 3.4-14.8 GiB | **1 329** (`SINK=1`) / 1 699 (`SINK=0`), 1.6 / 1.8 s tail, 21-32 GiB |
| `-lme 36` 76.8 % | (not run: a buffered 77 % tier re-reads 8 GiB per ubatch) | 1 037, 3.6 s tail, 66 GiB |

**`MOE_HOST_POOL_DIO=1` is worth +81 % at `-lme 44` (`SINK=1`) and +43 % at `SINK=0`, with decode unchanged
(76.2 → 75.9 with the demote on; −16 % with it off) for 2.1x the bytes.**  Buffered fills have the *better*
decode at `auto` (75.6 vs 67.7, −10 %) and read 5x fewer bytes there, so this is not a default flip — it is a
documented recommendation for a **sub-complement budget**, which is exactly what `briansp2020` measured
independently on his 56 GB container (`--load-mode mmap` 799 t/s versus `dio` 1 188, PR #267).  The mechanism is
the fill *shape*, not caching policy: a 0.65 MiB fill of one expert through a cold mapping is a page-fault
chain at depth 1, while one positional O_DIRECT read of the same bytes is a single request.

The span read fires and is worth **+8 %** at 77 % coverage (`C1` 1 037 vs `C1x` 958, 66 GiB coalesced) — smaller
than the r224 record's DIO-arm +14 % at the same budget but the same sign, and P0.3's question is answered: it
**does** engage on this box at this budget.

## 3. The transition tax — where the near-mirror prefill actually goes

Every arm above pays, after the last prefill ubatch, a `pool_rerank_from_heat_locked` pass: the arena's slot
count changed between the bands (179 → 79 slots per table when the prefill's work region takes the slab), so the
pool's "arena complement + mirrored residents" layout is re-derived, and it **frees and re-fills 1 400-6 400
pool slots, one 0.65 MiB expert copy at a time, in RANK order** — i.e. a random walk over a 84 GB file, at depth
one, inside the request.  Measured per-fill cost and total:

| arm | re-layout fills (first pass) | tail | per fill |
|---|---|---|---|
| M1 mirror | **0** (a covering pool returns early) | 0.6 s | — |
| A0 `auto` (warm cache) | 1 643 | 0.8 s | 0.5 ms |
| A1 `auto` (cold cache) | 1 468 | 6.8 s | **4.6 ms** |
| B1 `-lme 44`, buffered, idle drive | 1 973 | **21.3 s** | **10.8 ms** |
| C1 `-lme 36`, DIO | 6 428 | 2.9 s | 0.45 ms |

The attribution is exact: `pp-analysis.py`-style event accounting over the window between the last
`progress = 1.00` and the `prompt eval time` line puts **100 %** of the tail in `pool_rerank_from_heat_locked`
(the re-arm itself is < 0.1 s).  So the field prefill gap of a non-mirror tier is, in order: (1) the staging
reads of the "neither tier" experts (proportional to the gap × ubatches, the `MiB` column), (2) **this
re-layout**, which at `-lme 44` is 21 s of a 68 s request, and only then (3) the compute.

**Three consequences.**
1. It explains the 11-06 pass's unexplained 3x prefill swing: the swing is the drive's state applied to ~2 000
   latency-bound single-expert reads, not a compute change (the `proc` column moves by < 5 % where the tail
   moves by 6x).
2. It makes the `SINK=0` vs `SINK=1` prefill comparison at `auto` **not a SINK question**: `A1` (1 431) and
   `A0b` (1 329) both show the big tail, `A0` (1 810) does not — the pair reverses between reps.  r228's claim
   ("that cell is the `SINK=0` path to the counter") holds at the census level in both (`0 demoted`,
   `near-mirror 144`), and the prefill difference between them is cache state.  **Quote the pair, and never one
   rep** — the 11-06 file's own warning, now with a mechanism.
3. It is fixable without touching a decision (r231, §4).

## 4. r231 — the fills are issued in file order, not rank order

`pool_rerank_from_heat_locked`'s fill loop took `want` in rank order (heat order = random expert ids).  One
table's experts are contiguous in the file (`src_offs + e*host_bytes`), so **expert order is file order** and
the loop was issuing its ~2 000 single-expert reads as a random walk.  r231 sorts the to-fill set ascending
(the slot an expert lands in is arbitrary by construction — `pool_fill_empty_locked` takes the first empty one
— so nothing but the order changes), and `MOE_HOST_POOL_FILL_ORDER=rank` restores the old order for the A/B.

Measured on the cell with the largest tax (`-lme 44`, `SINK=1`, buffered, `radix`), old (r230) / new (r231)
interleaved; `proc` and `tail` as everywhere else, `h` the pool+arena census:

| arm | build | pp50k#1 | `proc` | **tail** | pp50k#2 | fills | gen3 |
|---|---|---|---|---|---|---|---|
| B1o  | r230 | 769 | 971 | **13.5 s** | 823 | 8 315 | 76.25 |
| B1n  | r231, file order | 874 | 1 054 | **9.7 s** | 718 | 7 352 | 76.07 |
| B1o2 | r230 | 780 | 1 100 | **18.6 s** | 1 164 | 8 314 | 76.02 |
| B1n2 | r231, file order | 779 | 913 | **9.4 s** | 992 | 7 358 | 76.13 |
| B1d  | r230, `DIO=1` | 1 329 | 1 386 | 1.6 s | 1 633 | 7 979 | 75.89 |
| B1dn | r231, `DIO=1` | 1 322 | 1 364 | 1.2 s | 1 702 | 7 380 | 76.09 |

So the order is worth **~30 % of the tail** (13.5-18.6 s → 9.4-9.7 s, consistently) and nothing on the DIO path
(1.6 → 1.2 s), but it is **not the main term**: the `DIO` row shows the fill *source* is (1.6 s against 13.5-18.6).

**What the tail really is: cgroup memory pressure, not read latency.**  A microbenchmark of the fill shape
(40 cold 0.65 MiB slices of this GGUF after `drop_caches`; `/tmp/fillbench.c`) measures **0.74-1.61 ms** for a
memcpy out of a cold mapping and **0.89-1.04 ms** for a buffered `pread` of the same bytes -- i.e. a cold
single-expert fill is ~1 ms on this NVMe, not the 4.6-10.8 ms the arms pay.  The arms run inside
`MemoryMax=56G` with a ~47 GiB pinned pool: every new mapping page must reclaim one, and `MemorySwapMax=0` gives
the kernel nothing to fall back on.  `MOE_HOST_POOL_DIO=1` reads around the page cache, so it both stops the
reclaim storm and stops paying for pages the box cannot keep -- which is why it is 10-15x cheaper here and why
a *buffered* pread (keeping the page cache) would not be the answer either.

**The first coherence attempt "failed" and that was a measurement artifact, not a defect.**  Four runs of the
same config produced four different `sha`s -- on *both* builds -- because `-lv 4` interleaves its statistics
lines into llama-cli's generated text, and `extract-generated.py`'s own docstring says to hash *without* it.
Re-run clean: five arms (r230 twice, r231 with`FILL_ORDER=rank`, r231 with the new order twice) all print
**`sha=cf7f8b23f404`**, and `scripts/gate-host-pool.sh` passes **37/37** on the r231 build with every arm at
`sha=a99ccbd97ecb`.  A pool *placement* change does not move a token, as `GREEDY-PURITY.md` requires.

**Shipped as r231** (`patches/` block 12, default on, `MOE_HOST_POOL_FILL_ORDER=rank` is the kill switch):
the tail it removes is 4-9 s of a field prefill, it cannot regress a decision (the fill *set* is unchanged), and
it also cuts the fill count ~12 % (8 315 -> 7 352).  The remaining tail is the fill *source*'s problem, and that
is a documentation matter (§2.2), not this loop's.

## 5. The parity answer

**No budget below a mirror is at parity, and the tier curve has two different shapes depending on the axis.**
On the field shape (`radix`, 56 GiB rig, r230, one load per arm, `-ub 6144`, MTP drafting):

* **Decode is at near-parity from 98 % down to 77 % coverage**: 75.6 t/s at `auto` (97.9 %), 76.2 at `-lme 44`
  (93.8 %), 76.1 at `-lme 36` (76.8 %), against a mirror's 83.8 — i.e. **−9…−10 % for 1.4 GiB (auto) to
  10.8 GiB (`-lme 36`) less pinned RAM**, and it is the demote that holds the line (45.4 t/s without it at 77 %).
* **Prefill is not**: the mirror's 1 895-1 939 t/s falls to 1 785-1 818 (`auto`, `DIO`), 1 329-1 699 (`-lme 44`,
  `DIO`) and 1 037 (`-lme 36`, `DIO`) — **−7 %, −13…−30 % and −46 %** — and on the shipped buffered fill mode the
  `auto` and `-lme 44` cells lose a further 0-23 s of band-transition tail (§3) rather than bandwidth.
* The storage/request column is where a tier's cost lands: a mirror reads 0.7 GiB per 50k prefill and 0.1 GiB
  per decode turn; `auto` reads 2.1 GiB / 0.85 GiB; `-lme 44` 14.8 GiB / 0.7 GiB; `-lme 36` **66 GiB** per
  prefill pass and 2.2 GiB per turn (19 GiB / turn with the demote off).

**The one sentence.**  A 76.8 % host-expert tier is a **decode-parity tier** on this box (−9 % for 10.8 GiB less
pinned RAM, with `MOE_HOST_POOL_SINK=1` and `-lme` well below the set), and it is **not a prefill-parity tier**
(−46 %); the smallest budget that keeps the *shipped field default* within 3 % of a mirror on decode is the whole
host set — `auto` at 97.9 % is −10 % — so the headline is the curve, not a parity point.  For a **sub-complement**
budget the prefill gap is dominated by the pool's fill mode (`DIO` +43…+81 %, §2.2) and by the band-transition
tail (§3), not by the decode design.

### 5.1 What this closes, per `PLAN.md` P0

| item | answer |
|---|---|
| **P0.2** parity + crossover (the campaign's product, `TODO` #280) | **done**: the table in §2, the curve in §2.1, the answer above |
| **P0.2** the fill-mode question (A1/A2, B1/B1d, B0/B0d) | **done**: `DIO` wins a sub-complement prefill decisively (+81 % at 93.8 % with the demote, +43 % without), loses ~10 % decode at `auto` and reads 2-5x the bytes.  **Documented, not defaulted** -- see `ENVIRONMENT.md` |
| **P0.3** where the span read runs (`SPAN=0/1` at `-lme 36`, both `DIO=1`) | **done**: it fires (66 GiB coalesced, 0 fallbacks) and is worth **+8 %** (1 037 vs 958).  The r224 `+81/+57/+14 %` remain DIO-arm cache-staging numbers; the buffered field path still does not coalesce |
| **P0.2/P0.3** `-lme 44` partial field row | **done** (B1/B0/B1d/B0d) |
| P1.0 (r228) | unchanged and confirmed: `auto` demotes 0 and reads the `SINK=0` path |
| P1.3 (`MOE_EXPERT_CACHE_TOUCH` 2 vs 3) | still open |
| P1.5 (`TODO` #279, narrow-2) | still open **at the time of this pass** — closed later the same day, see [`RESULTS-2026-11-07-narrow2.md`](RESULTS-2026-11-07-narrow2.md) (measured / do not implement) |

## 6. Exact reproduction

```bash
# the rig (once): 56 GiB emulation = a 32.4 GiB mlocked ballast OUTSIDE a MemoryMax=56G scope
cd ~/llama-cpp-rdna-boosts/wip/bench-56gb
gcc -O2 -o ballast ballast.c && python3 gen_prompt.py
awk 'BEGIN{n=0} {if(n>=250000) exit; print; n+=length($0)+1}' prompt_long.txt > prompt_50k.txt
setsid nohup ./ballast 32.4 >/tmp/ballast.log 2>&1 </dev/null &   # MemAvailable -> 57.2 GiB (a 59 GiB box)

# one arm (the tier knobs are POSITIONAL: sudo resets the environment inside the scope)
LME=auto SINK=1 DIO=auto ./parity-arm.sh A1          # runs/<tag>/parity.txt + server.log

# a whole matrix, one arm at a time, never in parallel
./parity-matrix.sh arms-pass1.txt matrix.log          # 10 arms, one load each, ~30 min
python3 parity-summary.py                             # the table, from runs/*/parity.txt + server.log

# the transition tax, attributed: window [last progress=1.00 .. prompt eval time] per pass
python3 pp-analysis.py runs/B1/server.log

setsid nohup ./ballast 32.4 ...                        # (keep it up for the whole matrix)
scripts/check-gpu-faults.sh --since "4 hours ago" --log runs/B1/server.log
```

**The two traps that cost this pass time, both worth inheriting.**
1. **`sudo` resets the environment**, and the rig starts the server inside `sudo systemd-run --scope`: an exported
   `LME=44 SINK=0 ./run.sh` reaches *nothing*.  The first attempt silently measured the default arm three times
   and looked plausible (the tier readback is what caught it).  The knobs are positional arguments now, and
   `parity-arm.sh` prints `applied tier: slots=N/512` read back from the server's own placement log — check it
   before trusting an arm.
2. **`-lv 4` interleaves statistics lines into llama-cli's generated text**, so a coherence hash taken with it
   reports a different `sha` on *identical* text.  Four runs "disagreed" about the model's output before the
   extractor's own docstring was re-read: without `-lv 4`, five runs (both builds, both fill orders) give
   `sha=cf7f8b23f404`.  A `-lv 4` arm is for the census, never for the hash.

## 7. Records

| what | where |
|---|---|
| this pass (the matrix, the crossover, the transition tax, r231) | this file |
| the CLI decision cells and the r228 fix | [`RESULTS-2026-11-06.md`](RESULTS-2026-11-06.md) |
| the #276/#277/#278 closes (`soar`) | [`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md) |
| the rig, its harness and the summary tool | [`../bench-56gb/`](../../../wip/bench-56gb/README.md) (`parity-arm.sh`, `parity-matrix.sh`, `parity-summary.py`, `pp-analysis.py`, `fillbench.c`-style microbench in §4) |
| the fix that shipped (r231) | `WORKLOG.md` 261107.002, `patches/README.md` block 12, `ENVIRONMENT.md` (`MOE_HOST_POOL_FILL_ORDER`) |
| the demote's policy rules (do not re-derive) | `INVARIANTS.md` (r228, r224) |
