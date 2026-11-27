# HANDOVER — session 5 (cold start): make the campaign promotion-ready

> **DONE — see [`RESULTS-session5.md`](RESULTS-session5.md).**  §2.1 (the `transport_base` cushion, `N` = 2
> compute chunks, `GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS`) and §2.2 (the field verification: the 2048 MiB
> raise now applies, `overrun_deferred=0`) are implemented and measured, and the chain was **re-based onto
> r368** (the delivery had advanced r367/r368 since this was written).  The promotion gate ladder of §2.3 is
> still owed and is listed in `RESULTS-session5.md` §4; the block decision of §2.4 was asked and not taken.
> The state below is the state this session *found* — read it as the brief, not as the current state.

**Read this first, then `RESULTS-session4.md` (the measurements of record) and `HANDOVER-session4.md` §6 (the
audit that started session 4).**  Sessions 1-3 are history: `HANDOVER-COLD-START.md`, `RESULTS-session2.md`,
`RESULTS-session3.md`.

| what | where |
|---|---|
| prototype | `~/llama.cpp`, branch `wip/prefill-slot-pingpong`, tip **`7af09a636`** |
| prototype patches | `wip/flash-next-prefill/experiments/0014` … `0021` on top of `86ff5f9ce` (each one experiment) |
| delivery record | this repo, branch `wip/prefill-transport-pingpong`, `wip/flash-next-prefill/RESULTS-session4.md` |
| delivery | **untouched**: no block, no `release.json`, no fork push |

## 1. State of play — what is done and verified

Session 4 closed the audit, repaired `-sm tensor`, and turned the ping-pong into the default transport where it
can run.  All of this is **verified by measurement**, with the recipe in §5:

| item | result |
|---|---|
| reporting audit | every arena number named by KIND (`arena_region=` extent vs `arena_resident=`/`resident=` usage; `transport=`/`transport_used=` the carve; `arena_copies=`/`host_copies=` what moved); `RESULTS-session4.md` §1 |
| transport region sizing | 6618 → 3412 MiB from the ping-pong's real unit (`llama_model::max_host_expert_layer_bytes`); §3 |
| 1 GPU prefill | gather 2108.7 → **2491.1 t/s (+17.6 %)**, text `43d0b1f4629e`; §7 |
| 122B-A10B, 1 GPU | 960.6 → 1035.5 (+7.8 %), text `c3f4da781f01`; §7 |
| `-sm tensor` 2 cards | **runs** (it could not before): gather 3419 / split's ring 3029, text `e6fb96a8cda2` |
| `-sm tensor` 3 cards | 3205 / 3144, text `a5d2fad1af0f` |
| `-sm layer` 2 cards | 2539 / 2473, text `43d0b1f4629e` |
| the selector | `sched_stage_exposure_decides()`: stage iff the gather's host read ≥ 38 µs/token (tie measured at 1.07 MB/token on an x8 link); never stages where the ping-pong cannot run (a Meta split); never stages the FIRST wide ubatch (§5.1: a single-ubatch prompt loses ~7 %) |

Field defects found and fixed by the user-experience runs (§7.1-§7.5 of RESULTS): the Meta ring's depth vs the
region's, the selector staging on a Meta split, the transport region's bump cursor (only usable once per
process), the carve never being released, and the deferred decode/verify compute-band raise.

## 2. What is NOT done — in the order I would do it

1. **`transport_base` headroom (the user's proposal, agreed, not implemented).**  `narrow_floor` (the
   always-resident decode/verify COMPUTE band) is corrected upward by the context after the fit probe, but the
   transport region sits directly above it, so a raise can only be applied when nothing holds a region block —
   and on a server a prefill precedes every drop, so the raise is deferred indefinitely (measured: floor stuck at
   1536, layers cut to 2-4 slots, arena 16 112 → 11 330 MiB).

   **Intent (the maintainer's words, and the reason for the change):** this is a CUSHION, not a guarantee — "a
   happy case before we have to start being more serious, whereas today it's more serious all the time".  A very
   long prompt can still want to grow past the cushion and reach the bottom of the region again; that case keeps
   the deferred path (`experiments/0020`) as the CORRECTNESS fallback.  So implement both, and instrument the
   handover between them.

   Concretely: `size_t transport_base = narrow_floor + N × GGML_COMPUTE_BUFFER_CHUNK_MIB` at the first carve,
   **FIXED for the run** (default **N = 2**); the region lives at `transport_base`, and a raise that keeps
   `narrow_floor <= transport_base` is applied immediately with no VA move.  Touches `arena_alloc_transient`
   (`base + narrow_floor + ring_used`), `wide_place` (`narrow_floor + ring_region`) and `set_narrow_floor`.
   **The quantum is the compute chunk, not a magic 256/512** — a raise is always a whole number of chunks, and
   the field shape needed exactly two of them (1536 → 1792 → 2048), which is where N = 2 comes from: the smallest
   cushion that covers the observed case.

   **Instrumentation owed with it** (so the next session can tell whether the cushion is right instead of
   guessing): a distinct message when a raise EXCEEDS the headroom (`raise to X MiB exceeds the transport_base
   headroom by Y MiB -> deferred`), a per-run counter of that event, and both in the exit census.  If the field
   sees it, N is too small for that shape — that is the number to tune, not the presence of the fallback.

   Note the cost model: the cushion is paid ONCE, at the first carve, and the arena is sized after the carve with
   the raised floor known — so if the band does grow into it, the MiB were not wasted; if it never grows, they are
   N chunks of residency (~1.2 pp on a 40 GB host set at N=2).
2. **The 2048 MiB raise is still not applied** on the field `runme` (the two-slot MTP verify width).  It is the
   same one-way door; `transport_base` closes it.  Verify on the field runme with the three-request sequence.
3. **Promotion gates (not run this session, all owed).**  Per `AGENTS.md` → "Pick the gate for what you changed",
   for a transport/slab/cache default change: `scripts/validate-set.sh` (after the blocks are cut),
   `scripts/gate-prefill-logits.sh` (the prefill-logit KLD — **this is the one that can still say no**: the
   ping-pong changes the prefill transport, and only text identity has been verified so far),
   `scripts/gate-host-pool.sh`, `scripts/gate-kv-residency.sh`, `scripts/gate-fused-probe-splits.sh`,
   `scripts/check-shlib-symbols.sh` **plus a `GGML_HIP_NO_VMM=ON` build** (the pool fix is VMM-only),
   `scripts/check-gen-docs.sh` if an option ships, and the MTP four-axis
   (`benchmarks/mtp-adaptive-methodology.md`) because the selector changes which transport a decode-adjacent
   prefill uses.
4. **The promotion decision itself is the maintainer's**, and is a block decision: candidates are the selector +
   the re-keyed zero-carve rule into **block 09**, the ping-pong replacing the ring (already prototype-only), and
   the VMM pool's fail-soft growth into **block 10**.  `ENVIRONMENT.md`/`DIAGNOSTIC.md` entries are owed for
   `GGML_SCHED_STAGE_SELECT`, `GGML_SCHED_STAGE_EXPOSE_US`, `GGML_SCHED_STAGE_SELECT_DEBUG`,
   `GGML_CUDA_SLAB_TRANSPORT_MIB` (+ the deprecated `_RING_MIB`), `GGML_CUDA_SLAB_TRANSPORT_FALLBACK`, and the
   `MOE_HOST_POOL_STAGE_BULK_GBPS` threshold.
5. **`-ub` guidance for consumers**: on the field GSQ shape, `-ub 8192` reads ~2682 t/s against ~1726 at
   `-ub 2048` (same prompt, same seed).  Not a defect, but worth a line in the block note.
6. **The 122B staged arm reads ~23 % behind the deleted ring** (1035 today vs 1346 in session 3 at the same
   gather baseline) — unexplained, and it is the campaign's headline shape.

## 3. Traps this session proved (add to your instincts)

* **One prefill per process is not a test.**  Every arm in sessions 1-4 ran a single prefill; three defects above
  (the bump cursor, the return of the carve, the deferred raise) only appear on the *second* one.  Any arm on this
  transport must issue **two prompts with different content**.
* **A sampled, unseeded decode arm on a `draft-mtp` config is not evidence** — decode t/s tracks draft acceptance,
  and it manufactured a fake 20 % difference that had to be retracted.  Use **same seed, greedy, ≥ 2000 tokens,
  interleaved cycles**, and print the arena-after-drop beside the rate.
* `pkill -f "llama-server.*8033"` **kills your own shell** (the pattern matches the command line).  Use
  `pkill -x llama-server`.
* A leftover server holds 67 GB RSS and the GPUs; a second instance then fails to load and looks like a code
  regression.  Check `pgrep -x llama-server` and `free -g` first.
* The two field runmes (`GSQ-IQ3_XXS`, `STEW675-IQ4_NL`) **export `/opt/rocm-10.1.0-gfx120X/lib` in
  `LD_LIBRARY_PATH`** in front of a 10.2 build — the documented footgun.  Unset it for any measurement.

## 4. Where the evidence is

* `RESULTS-session4.md` — the audit table, the sweep, the selector, the final matrix, and §7.1-§7.5 (the four
  field defects with their logs and fixes).
* `experiments/0014` … `0021` — the whole session as eight patches on `86ff5f9ce`
  (`scratch-tree.sh arm <patch>` applies one).
* Field logs on the reporting box: `~/no-recover.md`, `/tmp/{ux,arena,dec,nr}*.log` (not committed).

## 5. How to measure (the recipes that work)

```sh
# 1 GPU campaign arm (1 device, pool tier, 89k prompt, text hash + rate)
cd wip/flash-next-prefill && ./arm.sh TAG MOE_EXPERT_HEATMAP=load NCMOE=47 NGEN=128
python3 ../../../scripts/extract-generated.py runs/TAG/run.log | tail -1     # expect 43d0b1f4629e

# interleaved, rotated A/B (never trust a single run: the staged arms swing 2-9 %)
./abba.sh SUITE 3 default 'gather:GGML_SCHED_STAGE_SELECT=0'

# tensor split on the two fast cards (0+2), and on all three
DEV=0,2 ./arm.sh TAG MOE_EXPERT_HEATMAP=load NCMOE=47 NGEN=128 EXTRA="-sm tensor -dev ROCm0,ROCm1"

# the field server sequence that finds the second-prefill class of defect
pkill -x llama-server; cd /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS
unset LD_LIBRARY_PATH; export HIP_VISIBLE_DEVICES=0 GGML_CUDA_SLAB_STATS=1
SERVER=/home/stew675/llama.cpp/build-rocm/bin/llama-server ./runme -lv 4 > /tmp/s.log 2>&1 &
# then THREE completions with different prompts (618 tok / 68 491 tok / 209 tok) and read, after each:
#   grep -oE "arena now [0-9.]+ MiB|compute band \(narrow floor\)[^.]*" /tmp/s.log
# expected on tip 7af09a636: 16 111.8 -> 15 977.7 -> 15 977.7 MiB
```

`-lv` discipline: no `-lv` for a text arm, `-lv 4` for a census arm, never `-lv 3`.
`GGML_CUDA_SLAB_STATS=1` for the layout; `GGML_SCHED_STAGE_SELECT_DEBUG=1` for the transport decision;
`GGML_MOE_STAGE_DEBUG=1` for a fill that declines.
