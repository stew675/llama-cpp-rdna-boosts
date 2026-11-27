# RESULTS — session 7: the final review pass, and the §4 tests re-run on the fixed tree

This is the session that `HANDOVER-session7.md` asked for: one last code-review pass (read as a reviewer, fix what
it finds in the same commit, no features), then the quick tests, then **STOP** for the block decision.  The review
itself is [`REVIEW-squashed.md`](REVIEW-squashed.md) §10; this file is the **measurement** half.

Nothing in the delivery changed: no block, no `release.json`, no fork push, nothing promoted.

| | |
|---|---|
| `~/llama.cpp` | branch `wip/prefill-slot-pingpong-squashed`, tip **`79d48465e`** — ONE commit on `af7a57ba8` (r368 block tip), working tree clean |
| the change as a patch | [`SQUASHED-campaign-r368.patch`](SQUASHED-campaign-r368.patch) (**5119 lines, 18 files**; was 5055 before this pass) |
| build | `~/llama.cpp/build-rocm` from that tip, `exit=0`, no warning from any file the change touches (the only warning in the build is the pre-existing `test-backend-ops.cpp` `-Wreorder-ctor` one) |
| box | idle at each run (`pgrep -x llama-cli|llama-server` empty, 59 MB VRAM/card, 46 GB RAM free); `PORT=8199`; `LD_LIBRARY_PATH` unset for every arm |
| base for an A/B | the reviewed pre-pass tree is `dc335794b`, still in the object store (`git checkout dc335794b` + rebuild) |

## 1. The campaign arm (1 GPU, GSQ-IQ3_XXS, 89k) — text and rate

```sh
EXTRAENV="NCMOE=47 NGEN=128" ./abba.sh s7-89k 2 default
```

| cycle | prefill t/s | gen t/s | sha256 of the generated text |
|---|---|---|---|
| c1 | 2492.4 | 37.8 | **`43d0b1f4629e`** |
| c2 | 2478.8 | 38.2 | **`43d0b1f4629e`** |
| median | **2485.6** | 38.0 | — |

The handover's expectation (`43d0b1f4629e` on every cycle, 2485-2505 t/s) is met, and the text is byte-identical to
the pre-pass tree — i.e. **the review fixes changed no result on the campaign's shape**, which is the first thing to
prove: one of them (the restored in-place guard, finding 2) can veto staging per split, and it does not here.

### 1b. The same arm as a CENSUS (`-lv 4`, `NGEN=32` so the hash differs by design)

```sh
VERBOSE="-lv 4" EXTRAENV="NCMOE=47 NGEN=32" ./abba.sh s7-89k-cen 1 default   # 2489.7 t/s
```

```
transport buffers: 3 -> 3.  Settled arena 20553.3 MiB (51.4 % of the 40012.5 MiB host set), 450.0 MiB per buffer
pipeline staging: 138 host-weight split(s), 3 buffer(s) of 450.0 MiB (the largest set, sized once; the transport's
                  depth (expert-cache residency rule, or the user's override))
pipeline staging: 3 buffer(s) released at the prefill -> decode transition (4 tokens <= the 64-token staging gate)
prefill fills: ok=1380 ... arena_copies=155491.1 MiB host_copies=234008.9 MiB (H2D 60.1 % of filled);
               BULK 1380 calls 234008.9 MiB, 124720 host DMAs (0 over-fetched experts); PRS rebuilds=0
```

* the depth is re-derived (`3 -> 3`) from a **51.4 %** coverage arena and the request stays `1350 = 3 x 450 MiB`;
* **138 splits staged** — the in-place guard did not take a single one away (a fully resident arena would have; this
  one is 51 %);
* the bulk fill ran (124 720 copy-engine DMAs, **0** over-fetched experts), the census is non-vacuous, and there is
  **no** `transport region exhausted`, `h2d_stage_buffer`, `no transport region is carved` or `pipeline buffer grows`
  line (count 0).

## 2. The field sequence (small -> long -> small), 3 requests, one server process

```sh
PORT=8199 ./field-seq.sh s7-field
```

| req | prompt tok | prefill t/s | gen t/s | arena after the drop |
|---|---|---|---|---|
| 1 (645) | 645 | 226.4 | 25.62 | 15 797.9 MiB |
| 2 (71 816) | 71 816 | **2136.2** | 19.70 | 11 501.3 -> 11 411.7 -> 11 322.1 -> 11 211.0 -> 11 099.9 MiB, then **15 754.3** |
| 3 (239) | 239 | 152.3 | 25.75 | 12 998.3 -> 12 900.8 -> **15 600.7** MiB |

The checklist items all reproduce:

```
transport buffers: 3 -> 2.   Settled arena 14307.8 MiB (35.0 % of the 40925.0 MiB host set), 450.0 MiB per buffer
                             ... region request goes 1350.0 -> 900.0 MiB
pipeline staging: 144 host-weight split(s), 2 buffer(s) of 450.0 MiB
pipeline staging: 2 buffer(s) released at the prefill -> decode transition (4 tokens <= the 64-token gate)
narrow floor) raised 1536.0 -> 1792.0 MiB, inside the transport region's headroom cushion
narrow floor) raised 1792.0 -> 2048.0 MiB, inside the transport region's headroom cushion
band_raise[absorbed_by_cushion=2 overrun_deferred=0]        (final; 0 overruns at every point in the run)
```
and **zero** occurrences of `residency mirrors drifted`, `pipeline buffer grows`, `h2d_stage_buffer`,
`no transport region is carved` or `transport region exhausted`.

**Where this differs from session 6, and why it is not the change.**  Request 2 reads 2136.2 t/s against 2157.4 in
`runs/s6-unit-field` (-1 %), and the arena-after-drop samples are 65-100 MiB lower at each of the same points
(`15 797.9 / 11 501.3 ... 15 754.3 / 12 998.3 / 15 600.7` against `15 862.9 / 11 607.5 ... 15 862.9 / 13 165.7 /
15 686.5`).  Both are the box's 2-5 % phase and the field shape's unpinned heat prior (§6 of the handover:
`field-seq.sh` does **not** pin `MOE_EXPERT_HEATMAP`, and the reserve partition moves with it).  The **pattern** —
the decision, the two-buffer depth, the 144 x 2 x 450 MiB transport, the band walk and the re-armed arena — is
identical, which is what this arm is for.

**The transport at the transitions** (the evidence for review finding 3, that the carve must stay):

```
wide-arm  narrow=1536 transport=0.0   transport_used=0.0   transport_base=2048 headroom=512
wide-drop narrow=1536 transport=0.0   transport_used=0.0
wide-arm  narrow=1792 transport=0.0   transport_used=0.0   transport_base=2048 headroom=256
wide-drop narrow=1792 transport=960.0 transport_used=960.0
wide-arm  narrow=2048 transport=960.0 transport_used=0.0   transport_base=2048 headroom=0
wide-drop narrow=2048 transport=960.0 transport_used=960.0
```

The **buffers** are given back at every drop (`transport_used` 960 -> 0 and the release line above); the **carve**
(960 MiB) persists, which is what `llama_context`'s narrow -> wide drop is gated on.  The old code zeroed it in a
narrow placement that ran after the release — a latent serial-upload path, removed in this pass.

## 3. The four buffer-count branches (all four reached)

`M122=.../Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf`, `M35=.../Qwen3.6-35B-A3B-Q8_0.gguf`, 1 GPU,
`VERBOSE="-lv 4"`, `NCMOE=40` (122B) / `48` (35B).

| arm | settled arena | decision | buffers staged | prefill t/s | handover's expectation |
|---|---|---|---|---|---|
| 35B-A3B Q8_0, auto | — | `3 -> 3` | 3 x 272.0 MiB | **3210.4** | `3 -> 3`, ~3200 |
| 122B, auto | 55974 MiB host set, cap-derived | `3 -> 2` | 2 x 630.0 MiB | **1196.8** | `3 -> 2`, ~1190 |
| 122B, `MOE_EXPERT_CACHE_MIB=4096` | 3935.7 MiB (7.0 %) | `3 -> 2` | 2 x 630.0 MiB | **1201.6** | `3 -> 1`, ~1030 |
| 122B, `MOE_EXPERT_CACHE_MIB=3072` | 3061.1 MiB (5.5 %) | `3 -> 1` | 1 x 630.0 MiB | **1028.4** | `3 -> 0` (GATHER), ~930 |
| 122B, `MOE_EXPERT_CACHE_MIB=2560` | 2405.1 MiB (4.3 %) | **`3 -> 0` (GATHER)** | none (request 1890 -> **0** MiB) | **930.8** | `3 -> 0`, ~930 |

**The handover's labels were about a smaller arena, not about a different rule.**  My arenas settle ~500-900 MiB
higher than session 6's on the same cap, so each cap lands one branch *up*; the throughput tracks the **buffer
count**, not the label (1 buffer -> 1028.4 here, the handover's own "1 buffer" number; 2 -> 1196.8/1201.6 against its
"2 buffers" 1190; 0 -> 930.8 against its "0" 933).  The rule's arithmetic is on the decision line and each answer is
the documented one:

```
cap 4096: left after 1/2/3 = 3305.7/2675.7/2045.7 MiB; 2 >= the 2048 MiB minimum, 3 leaves 3.7 % (< 40 %)  -> 2
cap 3072: left after 1/2/3 = 2431.1/1801.1/1171.1 MiB; 2 < the minimum                                      -> 1
cap 2560: left after 1/2/3 = 1775.1/1145.1/ 515.1 MiB; 1 < the minimum                                      -> 0
```

The 0 branch is the important one: `transport declines for this run`, the region request becomes **0**, and the arm
runs at the gather's rate with **zero** refusal lines — 0 plumbed through the request, the selector and the fill
path, and a one-buffer pipeline is not silently forced on it.  (`GGML_CUDA_SLAB_MIN_ARENA_MIB` is still **not** a
probe for the rule — it also moves the slab's creation floor; capping the arena is the lever.)

## 4. Tensor split, 2 devices (`-sm tensor`) — the default is the GATHER

```sh
DEV=0,2 VERBOSE="-lv 4" EXTRA="-sm tensor -dev ROCm0,ROCm1" EXTRAENV="NCMOE=48 NGEN=16" ./abba.sh s7-ts2 1 default
```

**3429.3 t/s** prefill / 48.4 gen (handover: ~3430), and the only transport line in the log is

```
pipeline staging off: Meta(ROCm0,ROCm1) stages per device (`stage_input`), and the pipeline needs backend slots
                       -- a split uses the whole-split upload
```

so on a 2+ device split: the pipeline declines by design, **no** per-device ring is attempted (no
`transport: staging through ...`, no refusal line), and the split takes the whole-split upload — the measured-better
arm.  This is the state `HANDOVER-session7.md` §3.1 describes (the Meta staged transport is unreachable at every
fill because the per-device region is never carved), and its decision is still the maintainer's.

## 5. What this pass did NOT do

* No gate was run (`gate-prefill-logits` is the first of the ladder, and the ladder is **not** started: it needs the
  block decision, which is the maintainer's).
* Nothing was promoted, cut, renumbered or folded into a block; `patches/`, `release.json` and the fork are untouched.
* The §3.1 decision (delete the Meta staged transport, or finish it) is still open — this session fixed nothing about
  it beyond the two layers the previous session had already fixed, and re-confirmed with the arm above that the
  default is unaffected either way.

## 6. The tensor-split transport, WIRED IN and measured (the maintainer's instruction)

The maintainer's call after §11: the ring's removal was deliberate and is not being revisited; the *pipeline* is the
mechanism, it had simply never been wired to the Meta device, so it had never been compared with the gather.  Their
instruction was to wire the support in, measure, then choose the policy — and to keep the support whatever the policy
says, because policies move.  This section is that measurement.

### 6.1 What was wired

1. **The selector no longer encodes "the gather always wins on that tier".**  `sched_stage_exposure_decides` used to
   return "keep the gather" for any backend without `stage_buffer`/`stage_upload`/`stage_wait` — i.e. structurally for
   the Meta backend, before anything was measured.  It now asks only whether the backend can stage in **either** form
   (backend slots, *or* its own `stage_input`) and lets the exposure test decide the transport.  The two forms are
   documented as different transports (the slot form consumes in place; the `stage_input` form fills each device's slot
   and copies it in), and the comment says explicitly why the choice must stay a number and not a structural refusal.
2. **The depth and the region are the pipeline's, on both forms.**  The Meta ring's slot count is read from the same
   `slab_stage_slots` authority (`ggml_backend_cuda_transport_bufs`, i.e. the residency decision or the user's
   override), and the region request is the same `unit x depth` (the Meta device now forwards the region hooks, so
   `llama_context`'s arming and its "is anything carved?" test see the children).  The observed decision lines on the
   arms below: `transport buffers: 3 -> 2` at 40 %, `3 -> 3` at 60 %, with `720 / 270 / 450 MiB` per buffer as the
   shape's split set dictates.
3. **The region is really carved on a split now** — this is what the review's §10.3 fix bought.  Before it, the
   carve was zeroed at a narrow placement and the ring was refused at every fill (the §9 finding); with it, the ring
   stages, and the log says so: `transport: staging through Meta(ROCm0,ROCm1)'s per-device ring`.

### 6.2 The measurement (2 cards `-sm tensor`, GSQ-IQ3_XXS, 89k prompt, `-ub 8192`)

**Flags: the maintainer's reproducible configuration — `HOSTEXP=pinned LME=mirror LAZY=off`** (`-lme mirror
--lazy-mode off`, default `--host-experts`), `NGEN=16`, `abba.sh` rotation, `MOE_EXPERT_CACHE_MIB` setting the arena
(hence the coverage).  With that configuration the per-arm **spread is 0.1-0.5 %**; the same arms on the pool tier with
`--lazy-mode on` spread up to 25 %, which is what the earlier "it is erratic" reading was.

| coverage (settled arena) | staged (median) | gather (median) | winner |
|---|---|---|---|
| **40 %** (16 346 MiB) | **2869.8** (2855.8 / 2870.7 / 2869.8, n=3) | 2655.3 (2645.8 / 2657.5 / 2655.3, n=3) | **staged +8.1 %** |
| **60 %** (24 539 MiB) | **3009.6** (3008.1 / 3011.1, n=2) | 2910.9 (2909.8 / 2912.0, n=2) | **staged +3.4 %** |
| **80 %** (32 732 MiB) | 3100.0 (3098.1 / 3101.9, n=2) | **3164.7** (3162.6 / 3166.8, n=2) | gather +2.1 % |
| **~100 %** (uncapped) | 3428.2 (single arm) | **3429.8** (3427.5 / 3432.2, n=2) | tie |

`staged` = `GGML_SCHED_DEVGATHER=0` (force the staged transport, gather branch disabled); `gather` =
`GGML_SCHED_DEVGATHER=2` (force the gather above the band).  The **policy arm** is the default (no knob), and it now
follows the exposure rule on this tier: **STAGE at 40 % and 60 %** (`transport: exposed 24618.9 MiB / 8192 tok at 28.7
GB/s = 109.9 us/tok vs 38.0 -> STAGE`, and `16386.0 MiB ... 73.1 us/tok -> STAGE`), **gather at 80 % and 100 %**
(read `3429.8`/`3430.9` t/s, i.e. the gather's own number).

**Read it as a curve**: the crossover on this 2-card shape sits between 60 % and 80 % coverage (≈70 %), and the
exposure rule's own prediction for the set — the 38 us tie at ~78 % coverage, from §11.5 — lands within one arm of it.
So the tensor-split transport obeys the same physics as the single-device one, and **the staging transport is worth
8 % at the low-coverage end where a tensor split spends most of its time** on a model that does not fit the cards.

### 6.3 The honest scope of "the pipeline" on this tier

This is the per-device slot ring driven at the pipeline's depth, **not** the pipeline's full mechanism, and the
difference is structural, not a shortcut:

* **No fill-ahead.**  The `stage_input` form fills one split's set at its copy-input point.  The Meta drain
  (`graph_compute`) consumes **every** pending entry and clears the list, so a scheduler-level prefetch (fill k+N at
  split k) would be drained at split k's compute into the split-(k+N) input — correct data, but the device-to-device
  copy would be issued N layers early on the compute stream instead of overlapping, i.e. it would serialize.  A real
  fill-ahead needs a per-split drain plus a `stage_prefilled` flag on the split; that is the next step, and it is a
  change to the Meta backend's staging lifecycle, not a policy.
* **No in-place consume.**  The drain is a device-to-device copy because the redirect is **not correct** for an
  offloaded expert table (its consumer does not read `simple_tensor->data`; measured: one layer destroyed, MTP acc
  1.00 against 3.88 — `ggml_backend_meta_stage_input`'s comment keeps the receipt, `GGML_STAGE_META_REDIRECT=1` is
  A/B only).  So the tensor split gets the pipeline's *depth and authority*, not its in-place consume; the in-place
  win (the single-device +17 %) needs that redirect problem solved first.
* **Consequence for expectations**: on one card the pipeline is +18 % over the gather and +2.6 % over the old
  ring+lookahead; on a tensor split, today, the staged transport is +8.1 % at 40 % coverage — a real win, and not
  yet the full pipeline.

### 6.4 The support stays

Whatever policy is chosen later (default stage-vs-gather, a threshold, or a per-tier rule), the Meta transport must
stay **reachable and selectable**: the selector treats both staging forms as eligible, `GGML_SCHED_DEVGATHER=0|2`
remain the two controls, and no configuration may encode "the gather always wins on that tier" again.  Removing it is
a maintainer decision, recorded as such — which is why it is written here rather than left as a comment in a
threshold.

## 7. Coherence, checked the way the maintainer asked (clean text, tail first)

The arms in §6 were `-lv 4` runs (the transport lines need it), and `scripts/extract-generated.py` does **not** strip
log lines from its span — so those hashes contained interleaved timestamped lines and differed run to run for *every*
arm.  That is a methodology trap, not a result.  Re-run with **no `-lv`** (a text arm takes none; the census is a
separate run) and with the **tail** checked, not the head:

| coverage | staged t/s | gather t/s | winner | generated text |
|---|---|---|---|---|
| 40 % | **2861.0** | 2648.2 | staged **+8.0 %** | **321 chars `sha=184dc307181b`** (both) |
| 80 % | 3091.1 | **3171.1** | gather +2.6 % | **321 chars `sha=184dc307181b`** (both) |
| ~100 % | 3453.4 | 3431.2 | tie | **321 chars `sha=184dc307181b`** (both) |

**The staged Meta transport and the gather are byte-identical on this shape at every coverage point** (the 60 % point
was not re-run clean; its `-lv` rate was 3009.6 vs 2910.9).  The tails are coherent prose on every arm
(`... User likely expects maybe summarize? There is no question. We need produce final`).  So the landed measurement
has no purity question attached: the transport changes the *rate*, not the *answer*.

`tailcheck.sh` (this directory) is the tool: last N characters of the extracted text, a warning when the log carries
`-lv` lines inside the stream, and a flag for a 20+ repeated character.
