# HANDOVER — session 6: tombstone purge, fresh review, then STOP for the maintainer's user test

# HANDOVER — session 6: tombstone purge, fresh review, then STOP for the maintainer's user test

> **DONE (session 6) -- the user test passed and the follow-up rework is in.**  The purge and the review are in
> [`REVIEW-squashed.md`](REVIEW-squashed.md) §7-§8, the tests of §4 below and the rework in
> [`RESULTS-session6.md`](RESULTS-session6.md) (§8: `slab_stage_*`, the 0/1/2/3 buffer table, the over-carved unit
> fixed, the graph-token cap).  Tip **`dfdb8296d`**, patch **4947 lines / 18 files**, all four tests pass, the review
> is behaviour-neutral against the pre-review build, and **nothing is promoted**.  What is left is §5 below, which
> starts with the block decision and needs the maintainer's explicit all-clear.

**Read this, then [`REVIEW-squashed.md`](REVIEW-squashed.md) (the review that stands) and
[`RESULTS-session5.md`](RESULTS-session5.md) (the measurements of record).  Sessions 1-4 are history; every dated
detail is in `WORKLOG.md` (today's entries) and `archive/worklog/`.**

## 0. The ask, in order (do not reorder, do not run ahead)

1. **Squash any outstanding commits** and **do a fresh code review** with one rule on top of the usual ones:
   **no tombstones anywhere for dead code** — not in the source, not in `ENVIRONMENT.md`, not in `DIAGNOSTIC.md`.
   No comment explaining what something used to be, no empty function, no unused knob kept "as the record", no
   deprecated shim beyond a genuinely live alias.  If it is gone, it is gone; the record for "why it is gone" is
   `WORKLOG.md` and `REVIEW-squashed.md`, not a comment.
2. **Then run the quick tests** in §4 and record the numbers.
3. **Then STOP and hand over to the maintainer** for user-level testing (their own `runme` server, real prompts).
4. **Only after an EXPLICIT all-clear from the maintainer** — not "the gates are green", not "it looks fine", not
   silence — proceed to the promotion steps in §5.  Promotion is a **block decision** and is always asked first
   (`AGENTS.md`).

## 1. State at handover

| | |
|---|---|
| `~/llama.cpp` | branch `wip/prefill-slot-pingpong-squashed`, tip **`538391438`** — **ONE commit** on `af7a57ba8` (r368 block tip, base `919794500`), working tree clean |
| build | `~/llama.cpp/build-rocm` built from that tip (`llama-server`, `llama-cli`, `llama-perplexity`) |
| the change as a patch | [`SQUASHED-campaign-r368.patch`](SQUASHED-campaign-r368.patch) (4113 lines) |
| the 21-commit history | branch `wip/prefill-slot-pingpong-r368`, **history only** — never the review target |
| delivery repo | `main` = **`6ddc45ce`** plus this handover's own commit, pushed to `github.com:stew675/llama-cpp-rdna-boosts`; **nothing in `patches/` was touched** |
| delivery | untouched: no block, no `release.json` change, no fork push |
| box | idle, no `llama-server`, GPUs free |

The one-paragraph design, so a fresh reader does not have to reconstruct it: the slot-ring staging transport is
replaced by a **one-to-three-buffer PIPELINE** whose buffers are a carve inside the slab (the *transport region*),
filled one set ahead and consumed **in place**.  A split with a stageable host-weight set is filled into a buffer
and read there; the buffers are released at every prefill → decode transition and the region is re-carved when the
wide compute view is placed.  The **selector** decides stage-vs-gather per prefill width from the host bytes the
gather would pull per token (tie measured at 1.07 MB/token ≈ 38 µs at 28.7 GB/s), keeps the gather for the first
wide ubatch, and gathers wherever the ping-pong cannot run (a Meta split).  The region is **one buffer × the
pipeline's depth**, its depth comes from the expert-cache **residency** once the arena settles, and it sits on a
**cushion** of N compute chunks above the decode/verify band so that band can grow without moving it.

## 2. The tombstone purge: what to look for (this is the actual work of step 1)

Grep first, then read the hits — these are the places a tombstone is likely to survive:

```sh
cd ~/llama.cpp
grep -rn "retired\|RETIRED\|no longer\|was removed\|is gone\|used to\|deprecated alias\|_DEPRECATED\|obsolete" \
     ggml/src src ggml/include | grep -v "^Binary"
grep -rn "GGML_CUDA_SLAB_TRANSPORT_MIB\|GGML_CUDA_SLAB_RING_MIB\|ping-pong\|staging ring\|ring hole" \
     ggml/src src ggml/include
grep -rn "GGML_SCHED_PINGPONG\|GGML_SCHED_STAGE_SLOTS\|GGML_SCHED_STAGE_AHEAD" ggml/src src ggml/include
```

Known items to decide (the maintainer's rule points one way — remove — but each needs its own judgement because
some are *live aliases* rather than dead code):

| item | status | note |
|---|---|---|
| `GGML_CUDA_SLAB_TRANSPORT_MIB` (+ `_RING_MIB`) | **retired in code, no tombstone** (session 5, `WORKLOG` 261127.007) | `ENVIRONMENT.md` / `DIAGNOSTIC.md` **still document it** (they describe the shipped r368).  Dropping the entries is part of promotion, not of this repo's wip tree — but the review must **list every mention** so the edit is mechanical, and the new state ("the region is derived: one buffer × the pipeline's depth; depth = `GGML_SCHED_PIPELINE_BUFS`; cap = VRAM/4") must be written into the entry that replaces it. |
| `GGML_SCHED_PINGPONG_BUFS`, `GGML_SCHED_PINGPONG`, `GGML_SCHED_STAGE_SLOTS` | **live deprecated aliases** (read in one place, the CUDA authority) | Keep or drop is the maintainer's call.  They are not tombstones — they change behaviour — but session 1-4 spellings are all wip-only except `GGML_SCHED_STAGE_SLOTS` (which the delivery never shipped either).  If dropped, the parsing shrinks to one name and the docs get simpler; if kept, say so once and stop explaining it in three files. |
| `GGML_CUDA_SLAB_TRANSPORT_FALLBACK` (+ `_RING_FALLBACK`) | **live kill switch** (restores the cudaMalloc slot) | Keep as a `DIAGNOSTIC.md` entry; it is the A/B for the refusal path. |
| the D1 probes: `GGML_CUDA_FUSE_MOE_WEIGHT`, `GGML_CUDA_REPEAT_VEC` (+ `_DEBUG`) | **default-off, unreachable by default** — `binbcast.cu`'s vectorised REPEAT and `moe-weighted-reduction.cu`'s broadcast fusion | These are dead code by the rule above.  Their value was "the record of a measured negative", which now lives in `RESULTS-session3.md` — so the likely outcome is **delete the kernels and the knobs** and leave the measurement in the record.  Ask before removing the REPEAT vector kernel if it is referenced anywhere else. |
| `h2d_stage_*` / `slab_ring_*` internal names | live code, ring-era names | Session 4 deliberately kept them ("no consumer sees them; a mechanical rename is churn").  The maintainer renamed PINGPONG → PIPELINE in the *user-facing* vocabulary (session 5, J4); whether to finish the job internally is a judgement call for this session — if it is done, it is **one** mechanical pass with a build after it. |
| `slice_floor`, `pp_free_gib`, `sched_stage_min_tokens_for*` style leftovers | already removed in the session-5 review | The **compiler is the tool**: build with warnings on and treat `-Wunused-but-set-variable` / `-Wunused-function` in a touched file as a finding (one bit session 5: `slice_floor`). |
| duplicated/cross-referencing comments ("see `experiments/0020`", "the handover's §2.1") | campaign-internal references | Fine in `wip/` docs, wrong in shipped source: the source should explain the *current* rule, not the campaign's history. |

Also purge the same way in the delivery docs that ship: `ENVIRONMENT.md`, `DIAGNOSTIC.md`, and
`patches/README.md`'s block note for whatever block the transport lands in — **but only when promotion is
approved** (§5); before that, list the edits.

**Remember the amend trap** (it cost a push in session 5, `WORKLOG` 261127.006): a tool that edits files leaves them
**unstaged**, and `git commit --amend` takes the **index**.  `git add -A` first, and verify with
`git show HEAD:<file> | grep <marker>` or `git status` — the message is not the diff.  Keep the deliverable as **one
commit**: amend, re-export `SQUASHED-campaign-r368.patch`, and check the line count moved.

## 3. Review checklist for step 1 (beyond the tombstones)

The session-5 review's §1-§4 stand; the fresh one should at least re-check, on the *current* tree:

* every user-visible log line says what the code does (`pipeline staging`, `transport region request`, the refusal
  message's two causes, `band_raise[absorbed_by_cushion=… overrun_deferred=…]`);
* lock discipline: any function that takes `g_mutex`/`g_slab_mutex` and is called from a context that already
  holds it is a deadlock — **this campaign hit it three times**, so grep the call sites of `moe_cache_host_pool_requested`,
  `report_locked`, `moe_cache_residency_audit*` and the `_locked` helpers before calling any of them;
* one authority per number: the transport **depth** (`ggml_backend_cuda_transport_bufs`), the region **unit**
  (`slab_set_ring_unit`), the **cap** (VRAM/4), and the scheduler/Meta reading the depth through `slab_ring_slots`
  rather than parsing the environment;
* the `_locked`-vs-locking convention on every new helper, and `wide_armed`-guarded VA moves in the slab;
* format strings: build with `-Wformat` and read **every** warning in a touched file (session 5 found two).

## 4. Quick tests (step 2 — after the purge, before handing over)

All of them are in `wip/flash-next-prefill/`; `cd` there first.  **Give the server a port the maintainer's own
server is not using** (`PORT=8199`) — a field `runme` on 8033 will make the harness talk to the wrong instance.

```sh
# 1. campaign arm: 89k prefill, 2 rotated cycles, greedy text hashed from the same run
EXTRAENV="NCMOE=47 NGEN=128" ./abba.sh post-purge 2 default
python3 ../../../scripts/extract-generated.py runs/abba-post-purge/default-c1/run.log | tail -1
#    expect: sha=43d0b1f4629e, prefill ~2450-2500 t/s (the box is bimodal by 2-5 %)

# 2. the field sequence, three requests with different content (small -> long -> small)
PORT=8199 ./field-seq.sh post-purge-field
#    expect in runs/post-purge-field/server.log:
#      transport buffers: 3 -> 2 ... The transport region request goes 3412.5 -> 2275.0 MiB
#      transport region base 2048.0 MiB = decode/verify band 1536.0 MiB + a 512.0 MiB headroom cushion
#      decode/verify compute band (narrow floor) raised 1792.0 -> 2048.0 MiB, inside the ... cushion
#      pipeline staging: 144 host-weight split(s), 2 buffer(s) of 231.2 MiB (...)
#      arena after each drop 15 960.2 / 15 960.2 / 15 827.7 MiB, prefill ~2220 t/s
#      at exit: band_raise[absorbed_by_cushion=2 overrun_deferred=0], MoE expert cache: ... resident=15827.7 MiB
#      NO "residency mirrors drifted", NO "pipeline buffer grows", NO "h2d_stage_buffer" refusal

# 3. the short-prompt arm (the defect the maintainer's own test found, session 5)
PORT=8199 SHORT_TOK=560 LONG_TOK=560 GEN=8 ./field-seq.sh post-purge-short
#    expect: ~255 t/s for the 581-token request, ~338 t/s for the 239-token one, and neither refusal nor grow.
#    (A single-ubatch prompt cannot hide a fill behind a big ubatch, so it GATHERS by design -- that is the arm,
#     not a defect.)

# 4. the pipeline knob still pins (non-vacuity of the depth authority)
GGML_SCHED_PIPELINE_BUFS=1 PORT=8199 SHORT_TOK=560 LONG_TOK=560 GEN=8 ./field-seq.sh post-purge-pin1
#    expect: "pinned at 1 by GGML_SCHED_PIPELINE_BUFS", pipeline staging: 1 buffer(s), the 89k arm ~2000 t/s
```

Then record §4's numbers in `WORKLOG.md` and hand over with: the tip, the patch's line count, the four test
verdicts, and a one-line "ready for your user test; nothing promoted".

## 5. Promotion (ONLY after the maintainer's explicit all-clear)

Order, and each is a separate checkpoint:

1. **The block decision is asked first** (`AGENTS.md`).  The candidates, unchanged since session 4: the selector +
   the re-keyed zero-carve rule into **block 09**; the pipeline replacing the ring (the region/slab work) into
   **block 10**; and the VMM pool's fail-soft growth into **block 10**.  Nothing is cut, renumbered or folded
   without that answer.
2. **Re-base check**: `release.json` may have moved (it is `r368`, fork tip `af7a57ba8` today).  If it has, replay
   the squashed commit onto the new block tip first (`git rebase --onto <new fork tip> af7a57ba8`), rebuild, and
   re-run §4 — a re-base is a decode *and* prefill change (`AGENTS.md` rule 8).
3. **Gates** (the handover's session-5 list, still owed): `scripts/gate-prefill-logits.sh` **first** (the one that
   can still say no on a changed prefill transport — record the base from a clean build of the release tree),
   then `scripts/gate-host-pool.sh`, `scripts/gate-kv-residency.sh`, `scripts/gate-fused-probe-splits.sh`,
   `scripts/gate-moe-promise.sh`, `scripts/check-shlib-symbols.sh` **plus a `GGML_HIP_NO_VMM=ON` build**, and the
   MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`).
4. **Docs**: `ENVIRONMENT.md` / `DIAGNOSTIC.md` entries for every variable that ships (and the `TRANSPORT_MIB`
   removal + the "region is derived" restatement), `patches/README.md`'s block note, `INVARIANTS.md` for the new
   facts, `GREEDY-PURITY.md` if any purity claim moved, `check-gen-docs.sh` if an *option* ships (none do today:
   every knob here is an environment variable).
5. **Cut**: blocks, `scripts/make-patches.sh` (from a canonical chain rebuilt with `apply-all.sh`),
   `scripts/make-release.sh`, `scripts/validate-set.sh`, `scripts/check-rung-file-sets.py` per rung, then the
   release commit/tag, `scripts/fork-refresh.sh --build --gates --push` (personal fork only), the container images
   by the worker + `--verify-only`, and the `WORKLOG.md` cut entry.

**Still-open measurements to resolve or consciously accept** (they are not gates, they are risks):
the 122B-A10B staged arm reads ~23 % behind the ring that was deleted (1035 vs 1346 t/s at the same gather
baseline); the 38 µs threshold is calibrated at one link and one ubatch width (the x16 box owes a confirm); and the
`-ub 8192`-vs-`-ub 2048` guidance (~2682 vs ~1726 t/s on the field shape) is a block-note line, not a defect.

## 6. Traps this campaign proved (read before you run anything)

* **`git commit --amend` takes the index.**  `git add -A` first, and verify the content, not the message.
* **A hung `llama-server` ignores SIGTERM** — `kill -9`, and check `rocm-smi` before the next run: its leftover
  holds ~25 GB of VRAM and the next load fails with an OOM that reads like a fresh defect.
* **Two deadlock classes**, both "a lock taken twice on the same thread": `moe_cache_host_pool_requested()` from
  `alloc_all_locked`, and a census helper that took `g_mutex` from `report_locked()`.  Both were found by the
  *server hanging*, one of them after answering every request.
* **One prefill per process is not a test.**  A server's second and third prefill is where the session-4 defects
  lived; `field-seq.sh` exists for that.
* **A sampled, unseeded decode on a `draft-mtp` config is not evidence** (it faked a 20 % difference).  Same seed,
  greedy, ≥ 2000 tokens, interleaved cycles, and print the arena beside the rate.
* **The two field `runme`s export `/opt/rocm-10.1.0-gfx120X/lib`** in `LD_LIBRARY_PATH` in front of a 10.2 build:
  `unset LD_LIBRARY_PATH` for every measurement.
* `pkill -x llama-server`, never `pkill -f "llama-server.*<port>"` (it kills your own shell); `pgrep -x` and
  `free -g` before any arm; **use a port the maintainer's own server is not on**.
* **The box is bimodal by 2-5 %**: one run per arm is not a result, and a 2 % "win" is not a win.  Text hashes and
  the arena after a drop are the evidence; rates need rotated cycles.

## 7. Where the evidence is

* [`REVIEW-squashed.md`](REVIEW-squashed.md) — the review that stands: findings, the checked-and-clean list, the
  four judgement calls, and the two field defects the end-user test found.
* [`RESULTS-session5.md`](RESULTS-session5.md) — the measurements of record (cushion, buffer count, re-base).
* `WORKLOG.md` 261127.001-007 — the session, including the amend trap and the retirement decision.
* `experiments/0014`-`0023` — each session's delta as a patch (the squashed one supersedes them for review).
* `runs/` — every arm quoted above (not committed; regenerated by the scripts).
* Field logs on the reporting box: `/tmp/{ux,arena,dec,nr}*.log`, `~/no-recover.md` (session 4's).
