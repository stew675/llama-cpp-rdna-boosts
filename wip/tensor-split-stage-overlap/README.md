# Tensor-split staging — the drain overlap and the in-place consume (FOLLOW-UP campaign)

**Status: OPEN — follow-up work, NOT part of the delivery, nothing in `patches/`.**  Opened 2026-11-27, on the
maintainer's instruction, after an attempted drain overlap was **reverted for corrupting output**.  It exists so the
investigation has a home that is not a landed change, and so the numbers and the root causes are not re-derived.

It continues `archive/work/flash-next-prefill/` (the campaign that produced the staging pipeline and, on the maintainer's
decision, wired the Meta/tensor-split transport in).  Read first:
[`../flash-next-prefill/REVIEW-squashed.md`](../flash-next-prefill/REVIEW-squashed.md) **§12** (the wiring and the
measured curve) and [`../flash-next-prefill/RESULTS-session7.md`](../flash-next-prefill/RESULTS-session7.md) **§6**
(the measurement), then this file.  The landed state is `~/llama.cpp`
`wip/prefill-slot-pingpong-squashed`, tip **`a666f8637`** — the transport is wired in, measured, and coherent.

## 1. What is being chased, and what it is worth

The Meta (tensor-split) staged transport is a per-device slot ring: each device stages the split's host-weight set
into its own slot on its copy stream, and then a **device-to-device copy** moves the slot into the split input's real
buffer before the consumer runs.  Two things about that drain are on the table:

| idea | evidence | status |
|---|---|---|
| **Overlap the drain** (issue it on the copy stream behind its own H2D, free the slot there, consumer waits an event) | a *redirect* probe (drain removed entirely, `GGML_STAGE_META_REDIRECT=1`, rate-only) read **3066.4** t/s against **2868.6** for the drain — so **+6.9 %** is in the drain's latency alone.  An attempted overlap of the drain then read **3379-3393** t/s (+18.6 % over the landed 2869.8) ... and produced **garbage text** | **REVERTED, corrupt** |
| **In-place consume** (point the simple tensors at the slot, no drain at all) | the same switch, and the code's own receipt: `////` text and MTP acc **1.00** against 3.88 for the drain | **known-incorrect, not attempted again** |

The rate ceiling the broken overlap reached (≈3390 t/s, against the fully-resident 3430) is **not a result** — it is
what the overlap might be worth, and it has to be re-earned with correct output.

## 2. The two root causes, as established (2026-11-27)

**(a) Why the in-place redirect is wrong for a host-offloaded expert table.**  The consumer of such a table does not
read it through `simple_tensor->data`: it resolves the table through the expert cache (`moe_cache_get_table` → the
arena / the per-pass remap), so a slot the tensor merely *points at* is ignored, and the stale real buffer is read
instead.  This is not a scheduling detail to patch around — the cache serves the op from the arena, and it has no mode
that says "the staged split input IS the table, in identity layout, for this pass".  Getting in-place therefore means
a **cache-side** feature (an identity view of the slot with the take-over promise's lifetime rules), not a Meta-side
pointer swap.  The switch must stay A/B-only until that exists.

**(b) Why the attempted drain overlap corrupted output.**  The drain was moved to the fill side, on the device's
auxiliary copy stream, with a new `copy_ev` the consumer waits on.  But `ggml_backend_i.event_record` records on the
device's **main (compute) stream** — so `copy_ev` recorded the compute stream's position, not the copy stream's, and
the consumer was ordered against nothing.  The kernels read the split input before the drain landed → the staged
text became `3333…` while the gather arm (and every pre-drain arm) was coherent.  The mechanism needed already exists
inside `stage_upload`, which records its DONE event on the copy stream; it needs to be exposed (an appended
`stage_record`-style hook, or a `stage_d2d` that takes the event), not re-invented.

## 3. The plan (in order — each step gated before the next)

1. **Expose a copy-stream event record** for the stage slots (appended `ggml_backend_i` member, per the append rule),
   and a text test that proves the ordering: the staged arm's output must be coherent and must agree with the gather
   arm up to a near-tie.  Nothing else proceeds until that passes.
2. **Re-do the drain overlap** with the correct event, and keep the slot's free-event discipline explicit (the slot is
   dead after its drain; the free event goes on the copy stream right behind the drain, *only* for the drain path —
   the redirect path's slot is read by the consumer and must stay free-after-launch).
3. **Gate it**, in two SEPARATE runs, because one run cannot do both:
   * **text run, NO `-lv`** (`llama-cli`'s default is errors-only).  `extract-generated.py` does not strip log lines,
     so a `-lv 4` run hashes interleaved timestamped lines — that alone makes hashes differ between arms and between
     cycles, and it faked a divergence on 2026-11-27 (`WORKLOG.md` 261127.016; `flash-next-prefill/tailcheck.sh`
     warns when a log carries `-lv` lines inside the stream);
   * **census run, `-lv 4`**, separately, for the transport lines.

   Then judge **coherence at the TAIL, not the head** (`./tailcheck.sh LOG 200`): a divergence often shows up half-way
   through generation.  A configuration difference *may* legitimately move near-ties (`GREEDY-PURITY.md`), but on this
   shape it does not: on clean text the landed baseline is **byte-identical** between the two transports at 40 / 80 /
   100 % coverage (321 chars `sha=184dc307181b` on all of them).  So after a drain change **any** divergence is a real
   defect, not a configuration artifact -- gate on 4 coverage points x 3 rotated cycles, both transports, tail checked
   on every single arm.
4. **Only then** the in-place consume, as a cache feature with its own text + KLD gate; the redirect switch is deleted
   when it lands (or stays labelled A/B until it does).
5. **Re-measure the crossover** and hand the result to the policy question: on the landed tree the exposure rule's
   38 us threshold stages at 80 % coverage where the measurement says the gather wins by 2.1 %, so the threshold needs
   a per-tier calibration (or a per-device exposure figure) when this graduates.

## 4. Numbers of record from the landed tree (do not re-derive)

2 cards `-sm tensor`, GSQ-IQ3_XXS, 89k prompt, `-ub 8192`, `HOSTEXP=pinned LME=mirror LAZY=off`, `NGEN=16`:

| coverage | staged | gather | winner | text |
|---|---|---|---|---|
| 40 % | **2869.8** (n=3) | 2655.3 (n=3) | staged +8.1 % | coherent both (NGEN=64 check) |
| 60 % | **3009.6** (n=2) | 2910.9 (n=2) | staged +3.4 % | coherent both |
| 80 % | 3100.0 (n=2) | **3164.7** (n=2) | gather +2.1 % | coherent both |
| ~100 % | 3428.2 | **3429.8** (n=2) | tie | coherent both |

The broken-overlap attempt (reverted): 3379.0 / 3393.5 t/s with `3333…` text — kept here only as the size of the prize
and as the reason the step is gated.

## 5. How to run it

```sh
# the campaign harness now takes the reproducible configuration as first-class flags
cd ../flash-next-prefill
DEV=0,2 VERBOSE="-lv 4" EXTRA="-sm tensor -dev ROCm0,ROCm1" \
  EXTRAENV="NCMOE=48 NGEN=64 HOSTEXP=pinned LME=mirror LAZY=off MOE_EXPERT_CACHE_MIB=8192" \
  ./abba.sh <suite> 3 staged:GGML_SCHED_DEVGATHER=0 gather:GGML_SCHED_DEVGATHER=2
# text: python3 ../../scripts/extract-generated.py runs/<suite>/<arm>-c1/run.log --text
```

`MOE_EXPERT_CACHE_MIB` sets the arena and therefore the coverage (8192 MiB/device ≈ 40 %, 12288 ≈ 60 %,
16384 ≈ 80 %, unset ≈ 100 % on this shape).  Without `-lme mirror --lazy-mode off` the same arms spread up to 25 %,
so a number taken any other way is not comparable.
