# Results — session 3 (2026-11-26), branch `wip/prefill-transport-pingpong`

The session [HANDOVER-COLD-START.md](HANDOVER-COLD-START.md) asks for (A) an ABBA matrix, then (B) the
correctness block. This is that record. Every number below is **medians of interleaved, rotated runs** — the
box is period-2 bimodal (`INVARIANTS.md` r234) and the *staged* transport arms are bimodal by 5-9 % on top of
that, so a single run is not a result and no number here is a best-of.

Build: `~/llama.cpp` `wip/prefill-slot-pingpong` (r366 tip `489810652` + `0d9ca6964` + `5d03bc630` +
`aacb24d3b`, the last being this session's capture guard, `experiments/0005`). Harness:
[`abba.sh`](abba.sh) (new this session: N arms × C cycles with the order rotated per cycle, so phase drift is
not confounded with arm position). Fixed env for every arm: `MOE_EXPERT_HEATMAP=load` (residency is an input,
`GREEDY-PURITY.md` §89), the issue-#368 flags from `arm.sh`, `-ub/-b 8192`, `-c 131072`, `-fa on`,
`-ctk/-ctv q8_0`.

---


## 0. Naming — one table, used verbatim below

Every arm with `GGML_SCHED_DEVGATHER=0` is **"staged"** (the scheduler's host-weight ring). Three knobs vary
inside it:

| label | fill engine (`MOE_HOST_POOL_STAGE_BULK`) | consumes the ring slot… (`GGML_SCHED_STAGE_REDIRECT`) |
|---|---|---|
| `hybr-kern` | `0` — the cache's two-tier kernel (arena + host) | in place |
| `bulk1` | `1` — one copy-engine copy of the whole table | in place |
| `hyb-copy` | `2` — **hybrid**: coalesced host-run DMAs + one kernel for the arena side | **copy** into the graph input |
| **`hyb-redir` = "staged + in-place"** | `2` — **hybrid** (default) | **in place** (the ping-pong; default) |

"Staged + in-place" therefore means **hybrid fill + ping-pong redirect**, which is the prototype's default.

**Where a resident expert's bytes come from (the row everyone gets wrong).** In `hyb-redir`/`hyb-copy`, an expert
that is in the arena is served **from its arena slot** by the arena-side kernel launch (VRAM -> VRAM, no PCIe);
only the **non-resident** experts are fetched from the pool mirror over the link. Two deliberate exceptions, and
they are the reason "it depends" is the honest answer: a **merged gap** (`MOE_HOST_POOL_STAGE_MERGE*`) converts
the residents it absorbs into host-served experts, and `BULK=1` over-fetches *every* resident from the mirror.
Measured per arm once the accounting was fixed (the hybrid used to leave `g_stage_arena_bytes` at 0, so a staged
line read `arena=0.0 MiB ... (H2D 100.0%)` and hid this):

| 1 card, 89k, `-ncmoe 48` | arena-served | host-served | H2D share | host DMAs | over-fetched |
|---|---:|---:|---:|---:|---:|
| default (no merging) | 135 451 MiB | 303 036 MiB | 69.1 % | 105 468 | 0 |
| `MERGE_BYTES=2` (gap budget) | 58 445 MiB | 380 043 MiB | 86.7 % | 19 492 | 133 034 |
| `MERGE_FLOOR=8` (run floor) | 40 859 MiB | 397 629 MiB | 90.7 % | 14 586 | 159 291 |

**Three different "coverage" notions, and they are not interchangeable** (this cost the campaign a wrong claim):

| term | code | meaning |
|---|---|---|
| **covering pool / mirror** | `pool->slot_expert[e] == e` for all `e` | the **host** tier holds every expert in file order — what makes one whole-table DMA or run-coalescing possible at all |
| **arena static coverage** | `expert_slot.size() / n_experts` | how many experts have a **VRAM** slot (Flash-Next 1 card and the 2-card split: 256/512 = 50 %) |
| **reach hit rate `h`** | census `MoE expert cache: h=` | which of those slots the routing actually reaches (a property of the *placement*, i.e. an outcome of the fill policy) |

On the 2-card split the pool was **covering in both arms** and the static arena coverage was **identical**; the
arms differed in *which* experts were resident (`h` 0.647 vs 1.000). A transport selector therefore cannot be keyed
on "is the table covered" — the measured discriminator is **effective host traffic per ubatch** against that
ubatch's compute time, and `h` cannot be an input either because it is an outcome of which transport ran.
`GGML_SCHED_STAGE_AHEAD` (one-split lookahead) and `MOE_HOST_POOL_STAGE_MERGE` (over-fetch across short
resident runs) sit on top and are separate.

---

## 1. The ABBA matrix — Flash-Next IQ3_XXS, GPU0 (Gen5 **x8**, 28.6 GB/s), `-ncmoe 48`, 89 234-token prompt

5 cycles × 6 arms = 30 runs, order rotated each cycle. `GGML_SCHED_DEVGATHER=0` is "staged" (the scheduler's
host-weight ring); default is "gather" (the CUDA device gather).

| arm | env | median | min — max | spread |
|---|---|---:|---:|---:|
| `gather` | (default) | **2105.7** | 2102.9 — 2107.2 | 0.2 % |
| `hybr-kern` | staged, `MOE_HOST_POOL_STAGE_BULK=0` | 1884.1 | 1871.0 — 1890.4 | 1.0 % |
| `bulk1` | staged, `BULK=1` (one DMA of the whole table) | 2345.8 | 2338.2 — **2550.5** | 9.1 % |
| `hyb-copy` | staged, hybrid fill, `STAGE_REDIRECT=0` | 2356.5 | 2255.4 — 2428.7 | 7.4 % |
| `hybr-ahead0` | staged, hybrid, `STAGE_AHEAD=0` | 2372.7 | 2309.6 — 2454.6 | 6.1 % |
| `hyb-redir` | staged, hybrid, in-place slot (**default**) | **2440.2** | 2255.7 — 2451.7 | 8.0 % |

**What this establishes** (a delta is only claimed when the medians are separated by more than the largest
single-arm spread, per the handover):

1. **The fill ENGINE is the transport** — `bulk1`/hybrid vs the two-tier kernel fill is **+24.6 %**
   (1884 → 2346-2440) and the two arms' ranges do not overlap at all (max 1890 vs min 2256). This is the one
   clean step in the matrix and it reproduces session 2's engine finding with proper interleaving.
2. **Everything above the kernel fill is inside the noise.** `bulk1` (2346), hybrid+copy (2357), hybrid+in-place
   (2440) and AHEAD=0 (2373) sit within 4 % of each other while each has a 6-9 % run-to-run spread. The
   *median* ordering favours the in-place redirect (+3.5 % over the copy arm, +4.0 % over `bulk1`) and
   **`GGML_SCHED_STAGE_AHEAD` reads neutral-to-negative** (−2.8 %), which is what session 2 said about AHEAD
   but **not** what it said about the redirect's +6 %. At 5 samples per arm the redirect is a ~2-4 % effect,
   consistent with the +0.3 % the r250 retirement recorded, **not** a 6 % one.
3. **The variance is the staging path's, not the box's.** `gather` reads ±0.2 % and `hybr-kern` ±1.0 % across
   the same 30 interleaved runs, while every arm that pushes ~300 GB through the copy engine swings 5-9 %.
   The box phase is therefore *not* what moves these arms; the CE path is (§5).
4. **Against the gather default the win is +11.4 % (`bulk1`) to +15.9 % (hybrid+in-place)** at `-ncmoe 48` on
   x8 — i.e. a real, reproducible win, with the staged-vs-gather gap 3-4x the staged arm's own spread.

## 2. The same matrix on x16 (`radix`, Gen5 x16)

3 cycles × 3 arms, refreshed to the same build (`scripts/fork`-style re-cut from the branch; its `arm.sh`
replaced with this campaign's).

| arm | median | min — max | spread | vs gather |
|---|---:|---:|---:|---:|
| `gather` | 2343.4 | 2341.7 — 2357.1 | 0.7 % | — |
| `redir` (staged, hybrid, in-place) | 2458.1 | 2444.9 — 2497.3 | 2.1 % | **+4.9 %** |
| `bulk1` (one DMA of the whole table) | **2522.5** | 2496.8 — 2537.5 | 1.6 % | **+7.7 %** |

Two facts that do not hold on x8:

* **The win is much smaller at x16** (+5-8 % against +11-16 % at x8) — exactly as the transport share predicts.
* **The ranking flips: one whole-table DMA beats the hybrid at x16** (2522 vs 2458, ranges disjoint by 30 t/s).
  At x16 the over-fetch of resident experts costs little and a single large DMA avoids the hybrid's ~68 run
  DMAs per table; at x8 the run-coalesced hybrid is at least as good. **The transport choice is therefore
  link-dependent**, which is the same conclusion `wip/expert-transfer-engine` §7.3 reached for stream counts:
  the engine pays when the fill is dispatch-bound, and stops paying when it is link-bound.
* The staged arms are also far more *stable* at x16 (1.6-2.1 % vs 6-9 %) — consistent with §1.3's reading that
  the x8 variance lives in the CE path itself.

## 3. A second, harder shape: **Qwen3.5-122B-A10B Q4_K_XL** (no PLE), `-ncmoe 40`, 1 GPU (x8)

Single card, 120 host expert tables, `--host-expert pool`, **arena 9596 MiB = 17 % coverage**, `h=0.6315`,
**598 527 MiB staged per prefill (585 GB)** — twice the Flash-Next transport at a third of the compute. This is
the shape where the transport should matter most, and it has no PLE table at all (so no PLE page-in term).

| arm | median | min — max | spread | vs gather |
|---|---:|---:|---:|---:|
| `gather` | 963.4 | 962.7 — 964.0 | 0.1 % | — |
| `redir` (staged, hybrid, in-place) | **1345.9** | 1333.2 — 1356.9 | 1.8 % | **+39.7 %** |
| `bulk1` | 1345.9 | 1308.3 — 1368.8 | 4.5 % | +39.7 % |

Same-seed greedy text (128 tokens): gather `sha=c3f4da781f01`, staged `sha=c3f4da781f01` — **byte-identical**.

**+39.7 % prefill** on a shape the rewrite was not tuned for, from a transport change alone. This is the
strongest single result of the campaign and it is the one that should lead the pitch: the win scales with the
transport share, and on a 17 %-coverage 122B on one card the share is large.

## 3.5 A tensor split: the hybrid now fills a shard, and the result is residency-dependent

`-sm tensor` was the one geometry where the prototype *lost* (session 2's hybrid declined a sliced table and
the two-tier kernel ran). A one-shot geometry census (`-lv 4`) showed why, and both split shapes reduce to the
same thing — **one 2-D copy per coalesced run**, because the destination is always packed in expert order:

| shape | `n_copies` | `do2d` | `t.expert_bytes` | `t.host_bytes` | pool stride | copy |
|---|---|---|---|---|---|---|
| unsplit whole table | 1 | 0 | full expert | = expert | = expert | 1-D (degenerate 2-D) |
| split axis 1 (contiguous shard) | 512 | 0 | 189 440 (the shard) | 473 600 | 473 600 | height = experts, width = shard, pitch = 473 600 |
| split axis 0 (row slice) | 1 310 720 | 1 | 368 640 | 921 600 | 921 600 | height = experts × 2 560 rows, width = 144, pitch = 360 |

`experiments/0006` generalizes the gate from `n_copies == 1 && !do2d && pool->host_bytes == t.expert_bytes` to
`pool->host_bytes == t.host_bytes` plus a coverage check, and issues `cudaMemcpy2DAsync` per run
(`ce_rows`/`ce_bytes`/`ce_pitch`). Measured (text byte-identical in every arm below):

| arm (5.2k-token prompt, 64 tokens) | prefill | vs gather |
|---|---:|---:|
| **122B-A10B**, 2 devices, gather | 1554.3 | — |
| 122B-A10B, 2 devices, kernel fill (`BULK=0`) | 1196.0 | **−23.0 %** |
| 122B-A10B, 2 devices, **2-D hybrid** | **1790.4** | **+15.2 %** (**+49.7 %** over the kernel) |
| Flash-Next, 2 devices, gather | 2275.5 | — |
| Flash-Next, 2 devices, 2-D hybrid | 2018.1 | **−11.3 %** |
| Flash-Next, 3 devices, gather | 2054.0 | — |
| Flash-Next, 3 devices, 2-D hybrid | 2046.1 | −0.4 % (parity) |

Text identical on 2 and on 3 devices (`sha=7454a02b99bf` / `sha=e98c06d6427a`), and the census now shows
`BULK 26 calls 2781.6 MiB, 245 host DMAs` where it used to read `BULK 0 calls`.

**On the 89k prompt (the only apples-to-apples comparison with §1-§3)**, 3 cycles, spreads 0.1-0.3 %:

| arm | prefill | vs gather |
|---|---:|---:|
| **Flash-Next, 2 devices, gather** | **3140.8** | — |
| Flash-Next, 2 devices, hybrid + in-place | 2967.6 | **−5.5 %** |
| Flash-Next, 2 devices, `bulk1` | 2938.4 | −6.4 % |
| Qwen3.5-122B-A10B, 2 devices, gather | 1439.8 | — |
| Qwen3.5-122B-A10B, 2 devices, hybrid + in-place | **1615.0** | **+12.2 %** |

Two cards are *much* faster than one at 89k in both transports (3140 vs 2106 gather; 2968 vs 2440 staged). An
earlier draft of this section compared the 5.2k-prompt split arms against the 89k single-card arms and drew a
"−11 % is a regression" conclusion from it — **that was a prompt-length artifact, not a result**: prefill t/s is
not comparable across prompt lengths (one card reads ~1580 at 5.2k against 2106 at 89k with the same arm).

**Why the Flash-Next split still prefers the gather** — and it is *not* "the arena covers everything" (an earlier
claim here that was wrong): 40 925 MiB over 288 tables is 142 MiB/table, i.e. **256 of 512 experts per table, 50 %**,
the same coverage as one card. The `-lv 4` censuses of the two arms show the difference is the **cache state the
two transports leave behind**:

| 2 devices, 89k | staged | gather |
|---|---:|---:|
| `MoE expert cache h=` | 0.6471 | **1.0000** |
| staged calls / host bytes | 374 / 39 GB | — |
| arena remap (zero-copy take-over) | 592 tables served | 0 served (25/25 declined) |
| gather calls | — | 24 (all from arena slots) |

So on that shape the staged path moves only **39 GB** (592 tables were served by the arena remap and only ~34
tables/ubatch were staged at all) — there is almost no host traffic for a coalesced fill to improve — while the
gather path's routing-driven fills leave the arena holding what is actually reached (`h=1.000` vs `0.647`). A
third of the staged arm's reaches are therefore host reads that the gather arm does not pay. **No coalescing
policy can fix a worse placement**, which is why this box disagrees with the 122B split (+12.2 %, where the
arena cannot cover and the host streamer wins).

## 3.6 Coalescing tactics (the DMA chop) — link-dependent, and now explained

The hybrid spends **68 DMAs per table** on Flash-Next at `-ncmoe 48` (105 468 DMAs / 1 551 calls) because a
53 %-resident arena chops the host side into short runs. `MOE_HOST_POOL_STAGE_MERGE=<n>` already implements
the obvious tactic — over-fetch a resident run shorter than `n` experts so two host runs become one — so it was
swept (3 cycles, rotated; text identical throughout):

| link | MERGE=1 (default) | MERGE=2 | MERGE=4 | MERGE=8 |
|---|---:|---:|---:|---:|
| x8 | **2523.0** (4.7 %) | 2386.6 (1.6 %) | 2408.8 (2.3 %) | 2369.9 (4.4 %) |
| x16 | 2479.7 (1.9 %) | 2504.1 (1.0 %) | **2553.1** (0.2 %) | — |

and the trade it makes (x8 census): MERGE=4 takes **105 468 → 16 038 DMAs** (68 → 10 per table) for
**303 036 → 388 949 MiB** of transfer and 141 515 over-fetched experts.

**Why the boxes disagree, quantitatively.** `wip/expert-transfer-engine` §7.2's own probe prices a copy-engine
issue at ~18 µs (125 run-coalesced 0.65 MiB copies: 48.2 against 55.8 GB/s ⇒ 2.2 ms over 125 copies), and
over-fetching one 0.55 MiB expert costs `0.55 MiB / bw`:

| link | over-fetch cost | vs 18 µs/issue | measured |
|---|---:|---|---|
| x8 (28.7 GB/s) | ~20 µs | break-even, slightly worse | **merge loses ~4.5 %** |
| x16 (~51 GB/s) | ~11 µs | cheaper than the issue it saves | **merge wins +3.0 %** |

### Three coalescing policies were implemented and measured

The tactic is right; the *denomination* and the *target* are what matter, and the first two attempts were wrong:

1. **Expert count** (`MOE_HOST_POOL_STAGE_MERGE=<n>`) — merge a resident run shorter than `n` experts. A proxy:
   the same `4` is 2.2 MiB on Flash-Next and 3.5 MiB on the 122B, and it does nothing for a *large* resident
   region that leaves a small host run.
2. **Gap budget** (`MOE_HOST_POOL_STAGE_MERGE_BYTES=<MiB>`) — merge any resident gap up to XX MiB. Reads
   naturally but targets the wrong quantity, and measures flat on x16 (2495.6 / 2511.4 / 2516.1 for 0 / 1 / 2 MiB)
   and neutral on x8.
3. **Run floor** (`MOE_HOST_POOL_STAGE_MERGE_FLOOR=<MiB>`, `experiments/0007`) — **no host DMA smaller than XX**;
   a run below the floor absorbs its adjacent residents (shorter side first) until it reaches XX or hits a table
   boundary. This is the one the hardware model asks for, because the copy engine's penalty is a function of the
   *DMA's own size* (0.65 MiB ⇒ 41 GB/s, 2.87 MiB ⇒ 48, one 333 MiB ⇒ 55.8), not of the gap's.

| policy | x8 (median) | x16 (median) |
|---|---:|---:|
| none (default) | **2513.7** (2.8 %) | 2492.3 (3.4 %) |
| expert count 4 | 2408.8 (2.3 %) | **2553.1** (0.2 %) — +3.0 % |
| gap budget 1 / 2 MiB | ≈ neutral | 2511.4 / 2516.1 — ≈ neutral |
| run floor 2 / 8 / 16 / 32 MiB | 2436.9 / 2470.1 / — / — | 2498.9 / 2521.4 / **2527.7** (0.8 %) / 2484.4 |

**Conclusions.** (a) Below ~50 GB/s do not merge at all: every policy is neutral-to-negative at x8, because a
0.55 MiB over-fetch costs ~20 µs against an ~18 µs issue. (b) At x16 merging pays and the *run floor* is the
principled knob, with an optimum at **8-16 MiB** and a real fall-off by 32 MiB — exactly the size at which the
probe says the short-DMA penalty is gone, past which you are only buying bytes. (c) The expert count still
measured best at x16 (+3.0 %), but it is a coincidence of this model's expert size and it over-fetches ~20 % of
all reaches to get there; the floor gets most of the win with a quantity that transfers across models and links.
(d) **The floor is per DEVICE**: on a heterogeneous split each device's link sets its own XX. (e) Every policy
shrinks the variance (x8 4.7 % → 1.6 %; x16 3.4 % → 0.8-1.4 %), the clearest evidence yet that the staged arms'
spread lives in the DMA issue path.

**Shipped default (maintainer's decision, 2026-11-26): `MOE_HOST_POOL_STAGE_MERGE_FLOOR=auto` = 16 MiB when the
DEVICE's calibrated H2D link is >= `MOE_HOST_POOL_STAGE_MERGE_FLOOR_GBPS` (default 50), off below.** Implemented
with a per-device read-only peek at the existing calibration (`ggml_cuda_device_h2d_gbps`), so it can differ per
card on a heterogeneous split and never forces the 512 MiB probe. Verified on both boxes:

| box | calibrated link | resolved floor | host DMAs | over-fetched | arena-served |
|---|---:|---|---:|---:|---:|
| x8 | 28.7 GB/s | **off** | 105 468 | 0 | 135 451 MiB |
| x16 | 54.2 GB/s | **16 MiB** | 7 881 | 186 413 | 26 249 MiB |

A caution this session produced: the single census run of MERGE=4 read **2557.4 t/s** against MERGE=1's 2438.4,
while the rotated 3-cycle medians read 2408.8 against 2523.0 — the opposite order. One run per arm is worthless
here (that is the whole reason `abba.sh` exists).

## 4. Is it the prompt? (DanoPTT's hypothesis, issue #368 comment)

DanoPTT's good state reads 2961-2980 t/s with `MoE arena = 0.950/0.951` and reports his prompt is a
concatenated C# codebase, not prose. Tested three ways:

**(a) A code prompt, same build, same prior.** `c1-code-290k.txt` = the first 290 000 bytes of
`ggml/src/ggml-opencl/ggml-opencl.cpp` (`sha256 e92133b4ea9cb889…`). 3 cycles, interleaved with the prose arms
so both see the same drift:

| arm | prose | code |
|---|---:|---:|
| `gather` | 2106.9 | 2082.1 (−1.2 %) |
| staged + in-place | 2403.5 | 2415.0 (+0.5 %) |

**No prompt effect.** Code is within 1 % of prose on both transports, and the staged win holds (+14-16 % on
both).

**(b) Why — with the prior pinned, the *content* cannot move the transport.** The `-lv 4` censuses of the two
prompts are identical:

| | prose | code |
|---|---:|---:|
| `MoE expert cache h=` | 0.7507 (26686/35550) | **0.7511** (26702/35550) |
| staged host bytes | 303 036.0 MiB | **303 036.0 MiB** |
| host DMAs | 105 468 | **105 468** |

So a fixed placement prior yields the same hit rate and the same transported bytes on either workload. (Both
prompts are also ~the same length: 35 550 vs 35 550 reaches, i.e. `MUL_MAT_ID` lookups are structural, not
content-derived.)

**(c) Letting the prior adapt per workload** (`MOE_EXPERT_HEATMAP_DIR` pointed at a scratch dir so the curated
sidecar is untouched — its sha256 was verified unchanged before and after):

| arm | visit 1 (cold) | visit 2 (warm) |
|---|---|---|
| prose | h=0.6340, 428 478 MiB staged, 2360.3 t/s | h=0.8804, 295 558 MiB, 2390.5 t/s |
| code | h=0.6294, 428 478 MiB staged, 2537.9 t/s | h=0.8722, 295 558 MiB, 2447.4 t/s |

Both workloads converge to the same h (0.88) and the same staged volume, and **the prefill rate does not move
(+1.3 % prose, −3.6 % code) while the transported bytes fall 31 %.** Two conclusions:

* prompt content is **not** the carrier of DanoPTT's state on this box (both prompts behave identically);
* **the transport is fully hidden at this shape** — 132 GB less traffic buys nothing. So the residual
  `wall − ops` is not transport volume (§6), and DanoPTT's 0.95 is a *placement* property (his prior was built
  on his workload, or his arena is larger), not a property of code as a prompt.

## 5. Why the staged arms swing 5-9 % on x8 (and what is not the cause)

* Not the box phase: `gather` is ±0.2 % across the same interleaved runs (§1.3).
* Not PLE page-ins: `--lazy-mode off` (PLE fully resident) vs `on`, 3 cycles each, reads
  on 2434.8 (2242.1-2449.2, 8.5 %) against off 2460.9 (2346.8-2463.7, 4.8 %) — a ~1 % median difference inside
  the spread, and the spread does not collapse. This matches README §8's earlier "`--lazy-mode mlock` vs `on`
  is a wash (PLE page-cache-warm)".
* Not the transported bytes (§4c: −31 % bytes, no rate change).
* The CE rate itself is the remaining candidate: the per-process H2D calibration reads 28.6-28.7 GB/s on every
  run (identical), but that is a *synchronous pageable* copy, not the pinned CE fill the arm actually uses, so
  it does not measure the varying path. **Unresolved; needs a per-fill timing, not another end-to-end arm.**

## 6. `wall − ops`, and the op budget (handover §5.1)

One arm with `GGML_CUDA_OP_TIMING=1` (instrument inflates both sides — use shares, never an absolute rate):
staged+in-place, `-ncmoe 48`, x8. Prefill wall **38 923 ms**, cumulative op time **35 332 ms** (of which the
24-token decode is 1 425 ms), so **`wall − ops` ≈ 5.0 s = 13 %**.

Aggregated op shares (whole run, 35.33 s):

| kind | ms | share |
|---|---:|---:|
| `MUL_MAT_ID` | 10 756 | 30.4 % |
| `MUL_MAT` | 8 231 | 23.3 % |
| `FUSED REPEAT` | **2 622** | **7.4 %** |
| `FLASH_ATTN_QSA` | 2 582 | 7.3 % |
| `FUSED RMS_NORM` | 1 949 | 5.5 % |
| `DSV4_HC_PRE` | 1 581 | 4.5 % |
| `FUSED MUL` | **1 134** | **3.2 %** |
| `INDEXER_TOPK` | 1 071 | 3.0 % |
| `FUSED GATED_DELTA_NET` | 860 | 2.4 % |
| `GLU` | 755 | 2.1 % |
| `FUSED SSM_CONV` | 605 | 1.7 % |

**`FUSED REPEAT` + `FUSED MUL` = 3 756 ms = 10.6 % of ops** — the `ffn_moe_weighted` REPEAT+MUL pair, confirmed
on the new instrument at session 2's 3.9 s. That is `D1` in the handover and it is still the largest single
op-side prize.

`wall − ops` is **not** transport: §4c removes 31 % of the transport with no rate change, and §1 removes the
whole gather's exposure (which *was* ~0.9 s/ubatch on the compute stream) by moving to the CE. What remains is
either (i) the arena-side fill kernel of the hybrid (one launch per table, VRAM→VRAM) plus the CE's DRAM
contention, or (ii) gaps the per-op events do not bracket. Separating them needs a split-level timeline — the
handover's answer stands, but the candidate list is now one item shorter.

## 7. Correctness block (handover B) — results

| check | result |
|---|---|
| **B1 same-seed greedy text**, `NGEN=128`, no `-lv`, prior pinned, Flash-Next `-ncmoe 48` | `gather`, `bulk1`, staged+in-place all **`sha=43d0b1f4629e`** (611 chars) |
| B1 on the 122B-A10B | `gather` == staged+in-place == **`sha=c3f4da781f01`** (556 chars) |
| **B2 prefill-logit KLD** (`gate-prefill-logits.sh`, Flash-Next host-expert `-ncmoe 48`, `--pin MOE_EXPERT_HEATMAP=load`, `--ab "GGML_SCHED_DEVGATHER=1"`, ctx 512 × 12 chunks) | **mean KLD −0.000000, same-top-p 100.000 % → PASS**; no GPU faults |
| **B3 decode, ≥512 tokens** (5.2k-token prompt so the prefill does not pollute residency, `NGEN=512`) | gather 49.1 / 50.0 t/s, staged+in-place 48.4 / 48.3 → **−2.4 %**; text identical `sha=ae6016a1b986` (2275 chars) |
| **B4 capture safety** | **code added** (`experiments/0005`): the redirect re-checks the split width against the capture limit instead of trusting the staging floor, and DECLINES to the copy path below it. Proven: `-ub 8192` logs `batch_tokens=8192 (> 16)`; `GGML_SCHED_STAGE_MIN_TOKENS=0` (which lowers the staging floor and would have redirected a decode split) logs `DECLINED: batch_tokens=0` and the text is **identical** (`sha=3098a0b3693c` both arms); a **12-row** prefill ubatch on the host-expert shape takes **0** redirects while the 9..16-token graph capture line fires — the capture band and the redirect cannot overlap. Neutrality at the campaign shape: 2454.8 t/s after the guard (vs 2440.2 median before). |
| **B5 cross-backend / other geometries** | 2-device `-sm tensor`: text identical (`sha=91ef8940d66e`) but **−12 % prefill** (2253 gather vs 1981 staged) — see §8, the hybrid's geometry gate declines on a split (`BULK 0 calls`) and the two-tier kernel runs. `--host-experts pinned`: 1269.5 → 1401.8 (**+10.4 %**), text identical. Dense 4B Q8_0: 7486 / 7456 / 7388 t/s across gather/staged/copy, text identical (`sha=be904e86f248`) — inert. `scripts/check-shlib-symbols.sh`: **OK**, all 16 shared libraries resolve every symbol. |
| **B7 variables** | not shipped yet (see §9) — `MOE_HOST_POOL_STAGE_BULK/MERGE/STANDDOWN`, `GGML_SCHED_STAGE_AHEAD/REDIRECT` are prototype-only and need an `ENVIRONMENT.md`/`DIAGNOSTIC.md` entry **and a default-on decision** when a block is cut |

## 8. Handover §5's open questions, answered

1. **What is the 5.8 s of `wall − ops`?** Still unseparated (arena-side fill kernel vs CE/DRAM contention vs
   unbracketed gaps), **but the transport-volume hypothesis is now excluded**: −31 % staged bytes with no rate
   change (§4c), and removing the gather's compute-stream exposure is already counted. The instrument to
   finish it is a split-level timeline, not another arm.
2. **Did the stand-down gate survive the bulk fill?** **Yes, and it matters more, not less.** With
   `MOE_HOST_POOL_STAGE_STANDDOWN=0` (the pre-patch refusal) the staged arm reads **1943** (2/2, ±1 % — stable,
   so not a phase) against the 2440 five-cycle median: **+25 %**. Session 2 measured the same gate at +3 % with
   the kernel fill, because then the refused split merely took a slower kernel path; now it loses the whole
   fast transport. The gate is load-bearing, not dead code.
3. **What still takes the 2061-range bounce?** **The decode-width copy path, not the prefill transport.** The
   counter is `2061 ranges / 3207.2 MiB` and is **byte-identical** for the prose and the code prompt, and the
   line is **not printed at all** with `GGML_SCHED_DEVGATHER=1`. So it is what the scheduler does with a
   host-expert table at a batch below the staging gate once the device gather is off — ~3.2 GB and the −2.4 %
   decode in B3. It is not on the hybrid's path and it is not prefill.

## 9. Retirement assessment (handover C) — for the maintainer's decision

**Updated after §3.5/§3.6: the two-tier kernel fill is still the fallback for a *partial* pool only — a
tensor split no longer needs it (`experiments/0006`), and the merge policy must become link-derived.**

Nothing was removed; `patches/`, `release.json` and every gate are untouched.

| candidate | verdict from this session |
|---|---|
| `moe_cache_gather_*` (device gather) | **Do NOT retire wholesale.** It is also the decode-width path: with `GGML_SCHED_DEVGATHER=0` the decode falls to the per-expert bounce (−2.4 % at 512 tokens, and that is where the 2061 ranges come from). The correct shape is the one `GGML_SCHED_DEVGATHER` mode 1 nearly has already — **gather at/below the cache band, stage above it** — i.e. flip only the *prefill* half of mode 1. |
| `moe_cache_upload_pageable`'s per-expert bounce | **Cannot be retired while the gather can be off**: it *is* the decode path in that configuration (§8.3). Retire it only together with the "gather owns the band" rule. |
| the two-tier `moe_cache_stage_kernel` + `stage_span_fill` | **Keep, but only for a non-covering (partial/heat-ranked) pool.** A tensor split no longer falls back to it: `experiments/0006` fills a shard with one 2-D copy per run (+15.2 % over the gather on a partially-resident 122B split, +49.7 % over the kernel fill), text byte-identical on 2 and 3 devices. |
| `GGML_SCHED_STAGE_AHEAD` | **Retire as a knob if the prototype ships**: neutral-to-negative at both links (2373 vs 2440 on x8). Keep the double-buffered *code* if it is needed to keep the drain off the compute stream — but do not ship the variable. |
| `MOE_HOST_POOL_STAGE_STANDDOWN` | **Not dead — load-bearing** (§8.2, +25 %). Keep. |
| `MOE_HOST_POOL_STAGE_MERGE*` (all three forms) | **keep the RUN FLOOR form only** (`..._MERGE_FLOOR`), **default `auto` = 16 MiB at >= 50 GB/s per device, off below** (maintainer's decision, implemented and verified on both boxes). The expert-count and gap-budget forms are measurements, not candidates. |

Worth recording for the promotion decision: **the win is real and large where the transport share is large**
(+39.7 % on a 17 %-coverage 122B on one card, +11-16 % on Flash-Next at `-ncmoe 48` on x8, +5-8 % on x16,
+10.4 % on a pinned master) and **negative on a tensor split** (−12 %) where the fill falls back to the kernel.
A default-on rule therefore has to be geometry-aware: stage when the covering-mirror, single-copy, whole-table
geometry holds; otherwise keep the gather.

## 10. Files this session

* `abba.sh` — new interleaved/rotated A/B harness (`EXTRAENV`, per-arm `PROMPT=`, medians + spreads).
* `experiments/0005-redirect-capture-guard.patch` — the capture-safety check (§7 B4).
* `experiments/0006-tensor-split-hybrid-fill.patch` — the hybrid fill on a tensor split (§3.5), plus a
  `-lv 4` geometry census (`bulk-decline diag:`) that should be removed before promotion.
* `experiments/0007-merge-floor-and-arena-accounting.patch` — `MOE_HOST_POOL_STAGE_MERGE_BYTES` (gap budget),
  `MOE_HOST_POOL_STAGE_MERGE_FLOOR` (run floor, the policy §3.6 endorses) and the arena-side byte accounting the
  hybrid was missing (which is what exposed that residents come from the arena unless over-fetched).
* `c1-code-290k.txt` — the code prompt (`sha256 e92133b4ea9cb889…`, 290 000 B of `ggml-opencl.cpp`; generated,
  never committed as a shipped `prompts/` entry).
* `results-session3/` — the committed per-run tables (`abba-s3-*.tsv`, the rotated-run summaries) and the
  suite logs (`abba-s3-*.log`) behind every number here, plus the x16 run's table from `radix`. The individual
  run logs are NOT committed (20 MiB, regenerable with `arm.sh`/`abba.sh`).


## 11. D1 probe: the REPEAT+MUL pair — where it actually is, and two attempts

The handover's D1 said "`ffn_moe_weighted` REPEAT+MUL, 3.9 s = 12 % of ops; fuse them". Measured with
`GGML_CUDA_OP_TIMING` on the staged arm: **`FUSED REPEAT` 2 622 ms (7.4 %) + `FUSED MUL` 1 134 ms (3.2 %) = 3.76 s,
10.6 % of the op budget.** Three findings, in the order they arrived:

1. **The pair is not what D1 said it was.** `weight_before_ffn` (the `repeat_4d(cur) -> mul` site in
   `build_moe_ffn`, `src/llama-graph.cpp:2696`) is **`LLM_ARCH_LLAMA4`-only**. On Flash-Next the REPEAT that costs
   2.6 s is the **hyper-connection expansion** (`[2560 x 4 x T]`, the 4 HC streams) — a different pair with the
   same shape, so the same treatment applies, but the D1 write-up does not describe this model's graph.
2. **Fusing the pair is a −4.4 % LOSS.** `experiments/0009` implements it (anchor at the REPEAT, broadcast-multiply
   kernel, skip the MUL; `GGML_CUDA_FUSE_MOE_WEIGHT`, now default OFF after the measurement): text
   **byte-identical** (`sha=43d0b1f4629e`), rate **2305.9 vs 2412.1** (3 cycles). Cause: **`GGML_OP_MUL` is
   in-place-capable** (`ggml_op_can_inplace`, `ggml-alloc.c`), so the unfused pair already writes the broadcast
   ONCE and multiplies it in place — there is no second 839 MB buffer to remove, and the fusion instead adds a
   redundant source re-read plus a second live buffer. The "2.5 GB of traffic per layer per ubatch" in session 2's
   note is therefore **not** what the graph does.
3. **The cost is the REPEAT kernel's engineering, not its redundancy.** `REPEAT` dispatches to the *generic
   binary-broadcast kernel* (`bin_bcast_cuda<op_repeat, 0>`), which computes a broadcast source index per element
   and moves scalars: ~3.1 ms for 335 MB ≈ **108 GB/s**. A vectorised broadcast-copy kernel (`GGML_CUDA_REPEAT_VEC`)
   is implemented and **correct** (text identical, +4.3 % measured) but **unproven**: it only sees the
   *generation*-width repeats (`[2560 x 4 x 2]`), because the 89k-prefill REPEAT is served by a **fused window** in
   `ggml-cuda-try-fuse.cu` (no census change: 2683 vs 2671 ms). So it is default OFF until it has a per-width win.

**Where the 2.6 s actually has to be attacked**: inside the fusion that serves the HC chain, or by making that
chain not need the broadcast at all. That is a block-10 change with the full fusion gate set
(`gate-lora-fusion.sh`, `gate-fused-probe-splits.sh`, `gate-prefill-logits.sh`), and it is the next op-side item —
not the fusion of two ops that the allocator already overlaps.


## 12. The two-buffer ping-pong (the maintainer's design) — implemented and measured

**`experiments/0010`.** Everything before this section was measured on the **slot ring** (`GGML_SCHED_STAGE_SLOTS`
= 8, round-robin, one staged tensor per slot, a one-split lookahead bolted on). The ring was never the design
that was asked for; it was reused because it was the shortest path to a measurement, and this section replaces it.

**The design, as implemented.** `GGML_SCHED_PINGPONG` (default ON; `=0` selects the ring for the A/B):

* **N buffers**, each sized ONCE to the **largest split's host-weight set** (`sched_pp_split_set()` sums the
  stageable host weights of a split, 256-byte aligned), allocated from the backend's stage slots;
* **N is fixed at scheduler construction** from the device's **free VRAM at startup**
  (≤14 GiB → 1, ≤24 GiB → 2, else 3; `GGML_SCHED_PINGPONG_BUFS` overrides). Measured on the 32 GB R9700:
  **26.0 GiB free at construction → 3 buffers** of 912.5-1137.5 MiB;
* host split k is filled into `buf[k % N]`; the fill for split **k+N** is issued into that same buffer as soon as
  k's compute is enqueued, waiting on that buffer's **free** event on the COPY stream;
* split k consumes `buf[k % N]` **in place** at its entry offset (no drain, no copy);
* the prologue fills buffers 0..N-1 with the first N host sets, so split 0 computes while the rest stream in.

What it deletes: the slot ring (`stage_n_slots`, `stage_slot_next`, per-slot growth and slot-exhaustion declines),
the `[2]` plan arrays, the `stage_d2d` copy-out, and `GGML_SCHED_STAGE_AHEAD` (there is no "drain" step to
anticipate).

**Measured** (89k prefill, `-ncmoe 48`, `GGML_SCHED_DEVGATHER=0`, 3 rotated cycles each, text byte-identical to
the ring and to the gather: `sha=43d0b1f4629e` at 128 tokens):

| staging structure | median t/s | spread |
|---|---:|---:|
| ping-pong, **1** buffer | 1943.5 | 0.2 % |
| ping-pong, **2** buffers | 2401.4 | 0.7 % |
| **ping-pong, 3 buffers** | **2446.1** | 1.0 % |
| **ring** (8 slots + lookahead) | ~2458 | 3.6-5.8 % |

So: **one buffer is the serial baseline** (−21 %: the fill and the compute want the same buffer, so the wait on its
free event exposes the whole transfer), **two buffers are correct but ~2.3 % behind the ring** (the fill for k+2
cannot start until k's kernels have finished reading `buf[k%2]` — a *device* dependency, not CPU blocking: the
wait is a `cudaStreamWaitEvent` on the copy stream), and **three buffers reach parity with the ring while cutting
the run-to-run spread by ~4x** (1.0 % against 3.6-5.8 %). The variance result is the design's real win: the ring's
in-flight depth is an emergent property of the ubatch's split layout, which is why the ring arms swung 5-9 % while
`gather` read ±0.2 %.

**A fixed count is mandatory.** An earlier revision re-chose the depth per graph execution by probing allocations;
it flapped (1 buffer on the priming pass while the slab's transient band is held, 3 once it settles, then 1 again
on a later probe) and every 1-buffer graph costs ~21 %. It also logged only its first choice, which read as
"this run used 1 buffer" when it did not. The count is now decided once, at construction, from free VRAM at that
moment, and never changes.

**Still open from this design** (the maintainer's two questions):

* **the gather could read the ping-pong buffer instead of copying into it** — with the buffers, every byte a routed
  op needs is VRAM-resident and addressable, so a gather variant that assembles the op input from *pp buffer +
  arena slot* (rather than pp buffer holding a dense copy with the holes filled in) would remove the hole-fill
  copy. That is the next experiment, and it is only possible with the ping-pong: the ring's slot indirection is
  what made it awkward.
* **staging, in the ring sense, is already gone**; what remains is the *fill* (`moe_cache_stage_table`'s hybrid:
  coalesced mirror runs + the arena-side kernel), which stays because it is the only thing that knows which
  experts are arena-resident.


### 12.1 The memory model, and the decode transition

The buffers **sit where the ring's slots sat** (the backend's stage buffers), the count is fixed at startup from
free VRAM, and they are **released at the prefill -> decode transition** so the slab gets the VRAM back
(`experiments/0011`). Three things had to be fixed to make that correct:

1. **The transition trigger is WIDTH, not the presence of host weights.** Every graph -- decode steps included --
   still carries the routed expert table as a host-weight input, so "no host-weight splits" never happens. The
   staging width gate is the same predicate that decides whether a split is staged at all, so "no split in this
   graph is wide enough to stage" *is* decode-only. Verified: `3 buffer(s) released at the prefill -> decode
   transition (4 tokens <= the 64-token staging gate)`.
2. **The ping-pong must honour that gate too.** My first version did not, so it staged the whole table at DECODE
   widths as well: generation **36.5 t/s against the ring's 48.4**. With the gate applied: **48.4 vs 48.2** --
   parity, and the release then fires on the same predicate.
3. **A `stage_release` hook** (appended to the backend iface, so the other backends' positional initializers stay
   valid) returns a slot to wherever it came from: the slab's ring hole via `ggml_cuda_slab_arena_free`, or the
   allocator. The scheduler synchronises the backend first, so no fill can be in flight.

**What is still missing for the full model**: releasing the buffers returns VRAM to *free* memory, but the ARENA
does not re-grow into it -- measured arena is 21 840.7 MiB before and after the release. Growing the arena on the
prefill -> decode hand-back is the slab's settle/re-arm path (`TODO` #42 territory: the cache sizes itself once and
a stood-down table is re-armed at the transition), so that is where the last part of "the slab stays maximal for
decode" belongs. The decode numbers are already at parity with the ring, so the remaining work is about the arena
*getting bigger* during decode, not about decode being slowed.

Also recorded: the pp buffers on this box do **not** move the arena or the eviction count while the wide phase is
active (measured identical: 21 840.7 MiB, ~10 590 evictions, decode 47.9/48.1/48.0 for ring/2/3), because they are
allocated from free VRAM rather than donated by the slab. Once the buffers are carved from the slab's wide phase
(the follow-up above), the intended trade appears: prefill evicts (it is bandwidth-bound, not residency-bound) and
decode gets the arena back.


### 12.2 Can the prefill read the arena with NO copy? (where that idea actually lives)

The maintainer's last question for this campaign: with the ping-pong buffers everything a routed op needs is
VRAM-resident, so the **arena -> buffer hole fill** should be avoidable. Two ways to get there, and only one of
them is reachable on this box:

1. **Gather from two sources** (pp buffer for streamed experts + arena slots for residents) into the op's input.
   That is still a copy, so the prize is only the one measured in §12: **132 GiB of VRAM->VRAM = 94 ms = 0.24 %**
   of a 38.9 s run. Not the 2.3 %.
2. **Read the arena in place** and fill only the *missing* experts into arena slots (evicting its own landing
   slots, never a decode resident) — the real no-copy arm. **It is the GTT expert store's prefill path**
   (`pf_in_place_locked`), gated on `g_gtt_arena`, which is set only for the UMA/APU tier:

   > `g_gtt_arena = true; … "the GTT EXPERT STORE is this run's MoE arena: … the pageable model mapping as the
   > fill source, and no in-place cold read"`

   On a discrete `--host-experts pool` run `g_gtt_arena` is false, so `pf_in_place_locked` declines and **this
   experiment cannot be run on the R9700 box**. The only no-copy arm available here is the **identity take-over**
   (`device remap` in the census), which needs the whole table resident — not achievable at this shape on one
   32 GB card (144 tables x 284 MiB ≈ 40 GB of experts).

**So the "no-copy prefill" question is a GTT/UMA-box experiment** (Strix Halo / gfx1151, or an APU with `-lme`
sized below the expert set), and it should be the first item of the follow-up session. Also recorded there, from
the maintainer: the arena will not always re-grow into the VRAM the pp buffers hand back, because growing it
exposes the hot set to eviction on the next prefill — a consequence of **table-granular** eviction (evicting by
table of experts rather than by expert). That redesign is explicitly out of scope for this campaign and is a
separate session.


## 13. The ring is REMOVED (maintainer's decision), and what that cost

`experiments/0012`. The slot ring, its lookahead (`GGML_SCHED_STAGE_AHEAD`), its slot count
(`GGML_SCHED_STAGE_SLOTS`), its per-slot plan arrays and the redirect-vs-copy drain are **deleted**. The ping-pong
is the only staged transport. Rationale (the maintainer's): the ring's throughput advantage sat inside the staged
arms' own 5-9 % variance, while its slot bookkeeping is what produced the corruption class this campaign chased
for two months.

| build | 89k prefill median (3 rotated cycles) | spread | text |
|---|---:|---:|---|
| ring + lookahead | 2446.1 | 1.0 % | `43d0b1f4629e` |
| **ping-pong only (ring deleted)** | **2509.1** | 2.5 % | `43d0b1f4629e` |

So the removal did not cost anything measurable — the median is *higher* (inside the two builds' spreads) and the
text is byte-identical. What went with it: ~300 lines of scheduler bookkeeping, the per-slot event table, and the
`GGML_SCHED_STAGE_SLOTS` / `_AHEAD` knobs.

## 14. Decode, arena and the buffer count at a 2000-token generation

Protocol: 5 246-token prompt, `NGEN=2000` (long enough for the expert cache to stabilise), `--lazy-mode off` (PLE
resident, so no page-in term), heat prior pinned, `-ncmoe 48`, two runs each alternating:

| arm | decode | arena | slots | fills / evictions | buffers |
|---|---:|---:|---:|---|---|
| pp2 | 50.0 / 50.2 | 21 840.7 MiB | 39 684 | 97 648 / 24 151 | 231.2 MiB x2 |
| pp3 | 49.7 / 50.1 | 21 840.7 MiB | 39 684 | 97 648 / 24 151 | 450.0 MiB x3 |

**Decode is identical and the arena is NOT larger with pp2** -- same size, same slot count, bit-identical
fill/eviction counters. The reason is ordering, not physics:

1. the arena is sized at init, **before the pp buffers are allocated**, so the third buffer costs free VRAM and
   never arena;
2. the release at the prefill -> decode transition hands the VRAM back but the arena is **not re-grown into it**
   (the table-granular eviction problem, out of scope for this campaign).

To make the buffer count a real memory decision, either **commit the buffers at construction** (allocate/announce
them to the slab before the cache sizes itself, so the arena sizes with 3 x set already spoken for), or re-grow the
arena on the hand-back once eviction is expert-granular. Both are follow-ups, not this campaign.
