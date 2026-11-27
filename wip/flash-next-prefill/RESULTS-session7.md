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
