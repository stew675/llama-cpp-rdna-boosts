# Two external analyses, assessed (R18, 2026-11-21)

Two messages arrived from the `@overdoingism` camp while the r328 ladder was finishing. This file records
what each one actually claims, what I verified, what was done about it and what is **waiting on the
maintainer**. Nothing here is a delivery change by itself: the ladder tip is untouched
(`8406cbd86`, tree `4e222b2ecba3b05e5e35484a328ae3f35842772b`).

| message | subject | state |
|---|---|---|
| 1 | an overlap verdict on the offered `llama-Frankenstein` patch set (MIT + Apache-2.0 parts, tracks r327) | **assessed below**; nothing is taken into r328 |
| 2 | a licensing obligation: block 03's chunked-GDN kernels are libr4d ports, and libr4d now ships in `radiance` (Apache-2.0 + NOTICE) | **fixed in the delivery repo**; one in-patch line still needs approval |

---

## 1. The licensing obligation — verified, and fixed at the repo level

### 1.1 The claim is true, and the tree states it itself

Both kernel files that block 03 adds say where they come from — this is not a third-party inference:

| file (added by block 03) | header says |
|---|---|
| `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16.cu` | *"Ported from libr4d's `r4d_gdn_kkt_solve_k128_c64` and `r4d_gdn_chunk_scan_k128_v128_c64`"*, and the fragment helpers are *"libr4d's `r4d_gdn_wmma.h`"* |
| `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16_gfx11.cu` | the same two kernels, re-validated on the first-gen WMMA fragment layouts |

`git grep` over the whole ladder tree finds **no other** libr4d-derived file.  Block 01's all-reduce only
*cites* radiance's fence discipline for fine-grained peer VRAM (`r4d_ar_oneshot_2rank_exact.hip` in
`archive/work/allreduce-small-msg-latency/README.md`) — an idea, not copied code, so no licence file is owed
for it (it stays credited in `patches/README.md`).

Before this change the delivery carried **only** its own MIT `LICENSE` and no `NOTICE` at all, so a
redistribution of the applied tree (the patches, the fork branch, the published container images) carried no
attribution for the Apache-2.0 parts.  Apache-2.0 §4(d) requires the NOTICE to travel with the derivative.

### 1.2 What was done (delivery repo only — no patch content changed)

| path | what |
|---|---|
| `third_party/radiance/LICENSE` | radiance's Apache-2.0 text, **verbatim**, pinned at radiance commit `04247d11253321f0bce34218c2951302ca3207b8` (`main`, fetched 2026-11-21; sha256 `cfc7749b…`) |
| `third_party/radiance/NOTICE` | radiance's NOTICE, **verbatim** (sha256 `82e8e590…`) — radiance now vendors `libr4d`, which is why radiance's notice is the applicable one |
| `third_party/radiance/README.md` | the pin, the two sha256s, why the files are here, and the two affected `.cu` files |
| `NOTICE` (repo root) | the scope statement: repo is MIT **except** those two files, which stay Apache-2.0; names what a redistributor must carry |
| `README.md` → *License* | MIT *with one documented exception*, pointing at `NOTICE` + `third_party/radiance/` |
| `patches/README.md` → Block 03 | a **Provenance / licence** paragraph in the block's own section (it is per-block reference, so the block that owns the files states it) |
| `AGENTS.md` → layout table | the `third_party/radiance/` row: never edit the two vendored files, replace them whole on a new pin |

Two things I chose deliberately, against the contributor's own shape:

* **Sanctioned shape, not their file paths.** They vendor `third_party/radiance/{LICENSE,NOTICE}` and add a
  root `NOTICE` — the same shape, which is right: the obligation is attribution, not a vendored subtree.
  No radiance source is (or was) copied into the tree; the `.cu` files are the delivery's own adaptation
  (llama.cpp layouts, fp32 I/O, the `h % H_k` GQA mapping, its own runtime-cc dispatch).
* **The pin is recorded, not inferred.** Their NOTICE pins radiance 1.16.0 / `769b4c64a`; I pinned the
  current `main` (`04247d1125`) instead of copying their pin, because the delivery's port predates both and
  a *newer* NOTICE over-attributes rather than under-attributes.  radiance's later snapshots only add
  contributors; if the maintainer prefers to pin to `769b4c64a` (byte-identical to theirs) that is a
  one-command swap and the sha256s in `third_party/radiance/README.md` move with it.

### 1.3 The in-file attribution — **applied 2026-11-21** (was: needed the maintainer)

The applied tree's own headers should point at the licence, so a consumer who only has the patch set knows.
Proposed (one added line per file, inside block 03):

```
 // Licence: Apache-2.0, derived from radiance/libr4d (see the rdna-boosts NOTICE and
 // third_party/radiance/); the rest of this file's surrounding code is MIT.
```

**Approved by the maintainer and applied** as a `fixup` into **block 03** (`f3e8fdc5c`, ladder tip
`73a74454b`, tree `0c46248f7371a0eee84a980a1e16ad25a6fbd38f`): a four-line `// Licence: Apache-2.0 …` block
after each file's `#if defined(GGML_USE_HIP)` guard, naming radiance/libr4d, pointing at
`third_party/radiance/`, and stating that the surrounding llama.cpp code stays MIT.  The diff against the
previous tip is exactly those ten comment lines; the rebuild is clean, coherence is `1c5d32ac537d` (unchanged)
and the kernel journal is clean.  The alternative — shipping `NOTICE` + `third_party/radiance/` *inside* the
patch so the llama.cpp checkout carries them — stays rejected: upstream's tree would grow delivery files, and
the fork branch re-cut is meant to be llama.cpp plus the blocks only.

---

## 2. The overlap verdict — mostly already handled, one genuinely new idea

Their point is that upstream has caught up on a lot of ground.  **The ladder already met that ground**, and
that is worth stating precisely, because it is the part of the message that could otherwise trigger
unnecessary rework:

| their pointer | what the ladder found, with the record |
|---|---|
| **DFlash** | upstream now ships `src/models/dflash.cpp` + `dflash2.cpp` at `919794500`; the ladder **merged onto upstream's class split** (`dflash_options` / `init(opt)`) and kept the delivery's feature set on measurement — **D17**, `RUNGS.md` §R12.4, `benchmarks/2026-11-20-r328-r12-dflash-merge.md`. Text and per-position acceptance identical in every pair; the delivery's ingest tail is +29.8 % pp at 26K on the 2-GPU 35B.  Nothing left to decide |
| **NORM + MUL fusion** | upstream `#32454`'s `{NORM,MUL}` and `{NORM,MUL,ADD}` arms were **the only two arms the delivery's dispatcher was missing**, and R10.1 added them in r327's own position: the delivery's narrow-row matcher wins where it matches, upstream's general epilogue is the fall-through (`RUNGS.md` §R10.1 items 1 and 4; gate: `scripts/gate-norm-rows-alias.sh`, **owed at R18, not yet run on the finished tree**) |
| **the server's slot scheduling** | upstream moved `tools/server/` 32 times in the re-base window (14 of them `server-context.cpp`); block 16 (prompt admission) and block 12 (idle-slot parking) were resolved against that code — `gate-prefill-fairness` PASS on three shapes + `--self-test`, `gate-prefill-duty` PASS dense and `draft-mtp`; `gate-park-idle-slots.sh` is **owed at R18** |

So the answer to "what to keep, drop or rewrite" for the *re-base* is: **nothing changes**.  The three
overlaps they name were resolved rung by rung, with a gate and a record each.  Two gates from that list are
still owed at R18 and are on the checklist (`gate-norm-rows-alias.sh`, `gate-park-idle-slots.sh`).

### 2.1 Their offered groups, mapped to what the delivery already ships

"Upstream covers it" is not the only reason to decline: several of their groups are things the delivery
solved differently, or already has.  Verdicts are mine, and every "take it" is a *future campaign*, not r328.

| group | their claim | delivery's position (verified in this repo) | verdict |
|---|---|---|---|
| **J1** q8_0r cache + batch-invariant decode/verify attention, KV split points that do not move with the length | exact | block 05 ships the band + native quantized K-V; the **split-point-identity weakness is known and documented** — `INVARIANTS.md` r315 (the band's `P = nsm` fixup can flip once per run; a verify batch straddling the 256-cell boundary takes the other family for one step, measured 0.327 maxdiff) and `GREEDY-PURITY.md` §36 (a decode step and its verify must take one kernel). **No batch-invariant kernel exists** | **highest-value idea offered.** It targets a defect class we have measured and documented but only *bounded*. A campaign, not a patch |
| **J1** fp8 prefill attention | lossy, KLD 1.05-1.52× floor | nothing in `patches/`; prior art is `archive/work/prefill-gap-attribution/FP8-PRIOR-ART.md` + the `cllm` branch (not delivery) | would need the prefill-logit gate and a default-on decision; big prefill claim (+12/+29/+65 %) |
| **J1** 9-16-row verification through the decode kernel | lossy (per-step −12 % @32K, −29 % @110K) | we just took **r326** (`GGML_CUDA_GRAPH_MAX_TOKENS` 16, gate replay band) — different lever, overlapping intent | conflict of interest between the two; measure together or not at all |
| **J2** fp8 prefill matmuls (hipBLASLt, 64 outlier channels), MXFP4→fp8 | lossy, KLD 4.0-4.4× floor, prefill +83…+115 % | block 06 (MMB) is our quantized prefill GEMM; no fp8 path | the largest single prefill number offered. Lossy default-on is allowed by policy *with* a gate — a real campaign |
| **J2** fp8 scales fixed only for the normalized residual streams (`6d`) | within floor on 22 content kinds, prefill +1.2-2.1 % | — | cheap, small, well-measured; good `optional/` candidate |
| **J2** own WMMA f16 GEMM decoding K-quants (Q4_K/Q5_K/Q6_K/IQ4_XS) | exact-ish (same f16 values as the f16 conversion, f32 accum) | block 06/07 own the K-quant prefill path; block 08 the fused MoE | overlapping territory — a re-port onto **our** MMB/dl path, not a drop-in |
| **J2** opt-in prompt tail in f16 | exact | no equivalent | small, self-contained; a plausible `optional/` artifact |
| **J3** 1-8 token K-quant matmuls in one launch, 4 more grid-stall pads, fused decode ops | exact | block 07 (grid pad, VDR), block 10 (the decode folds), r326 (16-row replay); `gate-grid-stall-pad.sh` covers our sites | mostly covered; the "one launch for 1-8 rows" shape is the maybe-new part |
| **J3** GDN draft verification with an anchor + a replay instead of one state per draft token | exact | block 03's rollback tracker keeps `K = n_rs_seq + 1` snapshot planes (`patches/README.md` Block 03) — the *opposite* design: memory per plane instead of a replay | **second-highest value**: our planes are slot pressure on a multi-slot server. Needs a re-derivation against our snapshot/$n_rs$ contract, not a port |
| **J3** draft model's output layer scored from an int2 copy, candidates recomputed | lossy, mmvk 0.59-0.82× floor, −6…−8 % per step | no equivalent (our spec draft head is exact) | decode-time win, lossy; `optional/`-shaped |
| **J4** end of turn sampled inside the reasoning becomes the reasoning's end tag | `sampling` guard | none | **out of scope**: common/ chat behaviour, not RDNA, and it changes the sampling rule. Declined unless the maintainer wants it as `optional/` |
| **J5** adaptive draft width from acceptance | speed | **block 02 is exactly this** (adaptive MTP draft depth, `--spec-draft-n-max` + the four-axis protocol) | already ours |
| **J5** drafts of one length for slots verifying together | speed | **`LLAMA_SPEC_ALIGN`** (r326's `gate-graph-replay-width` / `gate-spec-align`) | already ours |
| **J5** n-gram drafts from the prompt | speed | upstream's `ngram-mod`/`ngram-map` + block 02's `n_rs_batch` handling (`patches/README.md` Block 03, r282) | already upstream + already reconciled |
| **J5** lenience for sampled drafts | `sampling`, off by default | none | policy question (it changes the acceptance rule). Not RDNA, not r328 |
| **J6** the speculative loop on the GPU (sampling/acceptance/next input on device, host-checked), two sequences in one step | speed/guard | no equivalent; the host cost is tracked in `archive/work/decode-host-overhead/` | **a real decode lever**; large patch, large re-port |
| **J7** prompt cache + context checkpoints in pinned memory, copied in the background | speed | **already ours**: r96/PR #189's pinned, pipelined `state_stage` (block 09/12; `scripts/gate-state-pipe.sh` is the gate that reaches it), block 12's batched restore (r119) | likely already covered — verify before porting anything; "cut on the batch grid" may be the only new part |

### 2.2 Why none of it can land in r328 (the mechanical blocker)

Every patch in that repo is cut **against r327's tree** (`4f9dfa0e3` + the 18 blocks) and names it in
`README.md` → *Reproduction conditions*.  r328 is `919794500` + a re-cut block set, and the ladder moved the
exact seams those patches sit on:

* the speculator class split (`RUNGS.md` §R12.2 — `dflash_options`/`init(opt)`; J1/J5/J6 all touch it);
* the FA band's launchers, its 128/256-dim work gates and the fixup period (r314/r315, R05's D16);
* `GGML_OP_COUNT` 107 → **109** (§R14.1) and the backend/device iface tables (§R14.3);
* the fusion dispatcher TU (`ggml-cuda-try-fuse.cu`, §R10.1) — J1/J3 add launches next to it;
* the `nb32` widening (R05's repair) and the `#32387` guard drop (R04).

So "take anything from it" means **re-porting onto the new base**, per group, with the delivery's own gates
— which is a fresh campaign per group, not an item on R18's checklist.

### 2.3 The caveats that travel with their numbers

Worth recording now, so a future campaign does not re-derive them:

1. **Their measurements are one R9700 under Linux (ROCm 10), plus a second R9700 under Windows.** Nothing on
   multi-GPU.  The delivery's validated configurations are 2- and 3-device `-sm tensor` and `-sm layer`; a
   decode/fusion change is judged there (`AGENTS.md` → *Pick the gate for what you changed*), and a
   single-GPU win is not evidence for them (`INVARIANTS.md` has the multi-device regressions that prove it).
2. **Their "exact" is bit-identical on their box, at their batch widths.**  Ours is a same-seed greedy-text
   hash, per device count (`GREEDY-PURITY.md`).  A port must reproduce text at 1/2/3 devices, not inherit the
   claim.
3. **Two licences.**  MIT for the patches, Apache-2.0 for the radiance-derived parts (their `NOTICE` says so,
   and their J2 patch edits *our* libr4d-derived `.cu` file).  Taking any of it means carrying their
   attribution too, in the same shape §1.2 just established here.
4. **`-lv`, seeds and the purity rules** are ours: whatever is ported is verified with a text arm that has no
   `-lv`, then a separate `-lv 4` census — never a single `-lv 3` run.

---

## 3. Recommendation (for the maintainer)

1. **Ship nothing from the patch set in r328.**  Its input is r327's tree, and the re-base's remaining
   obligation is the release, not a new feature import.  Record the offers so they are not lost: this file.
2. **The licence fix is done at the repo level** (§1.2).  Approve or refuse the one-line header amendment
   (§1.3); if approved it rides along with R18's patch regeneration.
3. **Two ideas are worth a campaign, in this order** — each is a `wip/` campaign that earns a block only on
   a validated win *here* (3-device box), never on their numbers:
   * **J1's batch-invariant attention** — it closes a *documented* purity weakness (r315's one-step family
     flip; §36), which is worth more than the speed claim;
   * **J3's GDN draft-verification anchor + replay** — it would remove `n_rs_seq` snapshot planes from the
     multi-slot path, i.e. state memory, on the block that owns the rollback tracker (`02`/`03`).
4. **Skip J4, J5 and J7** unless a measurement says otherwise: J5 is block 02 + `LLAMA_SPEC_ALIGN` +
   upstream's n-gram drafts already; J7 is the pinned pipelined `state_stage` already; J4 is common/ chat
   behaviour and out of RDNA scope.
5. **J2's fp8 work is the biggest number offered** (+83…+115 % prefill) and the biggest risk (lossy,
   KLD 4× the floor, a new hipBLASLt path, and it conflicts with a default-on purity policy without a
   prefill-logit gate).  Park it as a `wip/` idea with the prior art pointer
   (`archive/work/prefill-gap-attribution/FP8-PRIOR-ART.md`) rather than as a candidate for r328+1.

**Nothing in §3 items 1/3/4/5 is decided here** — the block structure and any new campaign are the
maintainer's call (`AGENTS.md` → *Report findings before shipping*, *WIP and promotion rules*).
