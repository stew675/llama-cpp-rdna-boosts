# HANDOVER for A19 — the KV-residency campaign after r196

Written 2026-11-01 at the end of session **A18** (the #123/#125 promotion).  This file supersedes
[`HANDOVER-A18.md`](HANDOVER-A18.md) as the cold start; A18 remains the record of the A17-era plan and the Goal
A/B/C history.  [`COLD-START-PROMPT.md`](COLD-START-PROMPT.md) is the block to paste into a fresh session.

---

## 1. Where things stand

* **The current release is `v16-6050b467c-r197`** (16 blocks, canonical tip `ba7c2b7b8…`, tree
  `eaf673210…`; `release.json` is the source of truth).  r197 (A19) closed `TODO` #129 — block 09's
  routing-readiness fix — and fixed the block-15 sibling (`-nkvo` now declines the window, and the streamed arm
  keys on the placement); [`C3-NKVO-HOST-EXPERTS.md`](C3-NKVO-HOST-EXPERTS.md), `WORKLOG.md` 261101.002.  It
  contains:
  * **r195** (from another agent, PR #261): block 03's moved-recurrent-cell snapshots
    (`test-recurrent-state-depth` Phase C).  **Do not redo it**; it was already published while A18 was in
    flight.
  * **r196** (A18): block 11's indexer top-k takes its **block radix path at plain decode** (`TODO` #125), and
    block 15's attendance-pool **mirror grows on demand** (`TODO` #123).  Both are in the delivery now; their
    campaign records are [`C1-MIRROR-GROW.md`](C1-MIRROR-GROW.md) and [`C2-RADIX-TOPK.md`](C2-RADIX-TOPK.md),
    and `WORKLOG.md` 261101.001 is the release record.
* Goals A (window), B (pool + streamed decode arm) and C (MTP draft window) are delivered in block 15 (r178 /
  r190 / r193).  `TODO` #121 is closed, #122 is decided (P1 — fittability only).  `TODO` #123 and #125 are
  **closed to r196**.
* The r196 tree also carries a real behavioural decision: **parking on a pooled cache declines** to the
  server's save-and-clear fallback (the alternative, growing the mirror to the park bound, pins the whole
  context again).  The park gate accepts the declined fallback and `park == lazyoff` holds.

## 2. What is left, in the order I would take it

| item | what | size | blocked by |
|---|---|---|---|
| **#128** the gfx1201 decode-module MTP cost | a decode-module change costs ~7 % of MTP decode on this box while behaviourally inert; it **blocks #120's q4_0 re-land** and any decode-module change on gfx1201 | investigation | needs a per-node/kernel census + a driver check (7.14 vs 10.1) |
| **#120** the r187 KV rotations are `q8_0`-only | `q4_0` is done/assessed and returned to the submitter (PR #262, blocked by #128 on gfx1201); remaining `q5_0`, `iq4_nl`, `bf16` | M | #128 for `q4_0` |
| **#127** `cpy_2d_vec16` is F32-only | the vectorised 2-D fallback covers F32 / 16-byte-aligned; extend it if a BF16/F16 copy shows up | S | nothing |
| **#124** Goal A across `-sm tensor` | designed only (`A6-TENSOR-SPLIT-WINDOW.md`) | L | **hardware** (one card here) |
| **#116** native FP8 E4M3 | parked behind a GEMM regression (campaign `wip/fp8-support/`) | L | the fp8 GEMM/repack (see `TODO.md` §116) |

Two **candidates** from A17 that are still not TODO rows (decide whether to open them):

* **The qwen4exp separate MTP draft is not windowed.**  Goal C's window applies to an **embedded** MTP head
  only; Flash-Next's `-md` head keeps the full `-c`.  Extending it is a different access pattern (KV injection
  on the main scheduler + a decoder graph on the alt one) and has to be measured.
* **A pool-size policy for a 200k+ working set.**  A17 measured `=16416` hitting 97.9 % at 5 246 tokens and
  86.0 % at 102 800, and `=32768` bought 0.2 points and no speed.  The fixed cell count is fine at the shapes
  we run; nobody has asked what it should be at 200k+.

### 2.1 `TODO` #129 — `-nkvo` + host experts — **CLOSED in r197**

Full record: [`C3-NKVO-HOST-EXPERTS.md`](C3-NKVO-HOST-EXPERTS.md).  It was never the expert cache and never
block 15: block 09's generalised consumer search reads the routing ids when a split's inputs are copied, and
with `-nkvo` the GPU split began at the host attention output with no input yet, so it swallowed the router
chain and the routed op and read ids that split had not computed (out of range aborted, in range silently
picked the wrong experts — a **delivery reachability**, not upstream's, which copies the whole table).  Fixed
with a whole-table stand-down plus a split boundary at such a routed op; the block-15 sibling (the window now
declines under `-nkvo`, `is_streaming()` keys on `kv_region`) shipped in the same release.

### 2.2 `TODO` #128 / #120 — the gfx1201 decode-module cost, then the remaining KV write types

`#128` is the investigation that blocks decode-module changes on this box: on 3x R9700, dense 27B
UD-Q4_K_XL `--spec-type draft-mtp` reads 76.2 -> 70.7 t/s (1 GPU) / 133.4 -> 126.4 (3 GPU `-sm tensor`) with
either PR #260/#262 in the tree, with an identical kernel census, graph census, compute buffers, text and MTP
acceptance, with the fold's kill switch off and with graphs disabled; `gfx1100`/`gfx1151` are flat.  The shipped
r191/r194/r196 tree is the fast arm; the cost appears only when such a change is added.  Evidence:
`benchmarks/2026-10-31-kv-fold-q4_0-assessment.md`, `TODO.md` #128.

`#120` is then the remaining KV-rotation types: `q4_0` is assessed and returned to the submitter (correct and
+11.8 % on gfx1100, but 5-7 % of MTP decode on gfx1201 with the fold inert — needs a gfx1201 repro/bisect, then
re-land with `ggml_cuda_kv_fold_type_ok` on `RDNA3_0`/`RDNA3_5`).  `q5_0` has the same shape as `q4_0`;
`iq4_nl` searches its scale (separate work); `bf16` has no fused write kernel.  `#127` is the
`cpy_2d_vec16` F32-only limitation — a note, not work, unless a BF16/F16 block copy shows up.

### 2.3 `TODO` #124 — Goal A across a `-sm tensor` split (hardware-blocked)

Design only: [`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md) has the five changes (a Meta buft
variant, the grow fan-out, fit recognition, `region_capable`, fit sizing inside the split branch) and the four
`common/fit.cpp` findings a blind implementation gets wrong (`margins[0]` is simple device 0's target while
`free_dev` is the summed free; `kv_bytes_full[0]` is the whole logical KV; and the split `estimate()`'s
`ntd×` KV term).  Both block-15 placements decline a tensor split with one message; **do not remove those
refusals without the arms**.  Nothing here can be done on this box.

## 3. Environment

* Box: one R9700 (gfx1201, 32 624 MiB) plus an iGPU.  **Pin `HIP_VISIBLE_DEVICES=0` for every arm** (an
  unpinned `-sm tensor` enumerates the iGPU and aborts at model load with `invalid kernel file`).
* Build: `~/bin/build-llama-rocm` (ROCm 10.1.0 → `build-rocm/`), `~/bin/build-llama-rocm-714` for a 7.14 check
  (`build-rocm-714/`).  **Never pass `-DCMAKE_HIP_FLAGS`.**  `cmake --build build-rocm --target llama-cli
  llama-server -j 16` is the fast loop with ccache warm; keep ccache ≥ 30 GiB.
* Repos and branches:
  * delivery: `~/llama-cpp-rdna-boosts` (`main`), `release.json` = **r196**, tip `bb6599e80`.
  * fork: `~/llama.cpp`, branch **`rdna-boosts`** = the r196 tree (refreshed by `fork-refresh.sh`), plus the
    campaign branches `wip-kv-residency-123` and `wip-kv-residency-125` (both off r194, their records are the
    C1/C2 files), the older `wip-kv-residency-c` / `-b` / `-a` campaign branches, and the working chain
    `fold-r196` (the r195 chain + the two amendments; its tip has the same tree as `rdna-boosts` but a
    different SHA).  **Develop on a branch; never commit campaign code to `rdna-boosts`**, and never touch
    `patches/` or `release.json` outside a release.
* Models: `/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS` (qwen4exp, 76 GB, has `runme` and the MTP sidecar
  `mtp-…-shared-Q8_0.gguf`), `Qwen3.8/27B/{Q8_0,Q4_K_XL}`, `Qwen3.6/35B-A3B/{Q8_0,Q4_K_M}`,
  `Qwen3.5/122B-A10B/Q4_K_XL`, `Qwen3.5/4B/Q8_0`.
* Prompts: `prompts/prose-rdna-boosts.txt` (5 298 tokens at the 27B), `prompts/kld-corpus-35b-in-domain.txt`
  (`cat` it 4x for a 102 899-token prompt, 7x for 180 035).  Both are hash-pinned in `prompts/README.md`.
* **Every Flash-Next arm carries `-ot per_layer_token_embd.*=CPU -lm none -lzm off`** (the 27 GiB PLE table
  otherwise moves identical runs by ±20 %).
* A working Flash-Next server shape (`/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/runme`): `-c 262144`,
  `-ncmoe 48`, q8_0 KV, `-np 1`, `LLAMA_KV_REGION_POOL=16416`, `GGML_CUDA_SLAB_HEADROOM_MIB=1536`, adaptive MTP
  n-max 7.  Measured on r194: 102 800-token prompt → prefill 575 t/s, decode 53.2 t/s, acceptance 0.603, pool
  hit 86 %.  On r196 the pool's mirror is **grown on demand** (`LLAMA_KV_POOL_MIRROR_UNIT_MIB`, default 1);
  `LLAMA_KV_REGION_DEBUG=1` prints one `KV pool mirror pinned X of Y MiB (+Z)` line per grow step, and
  `LLAMA_KV_POOL_STATS=<path>` still takes a **path**.

## 4. Rules that cost sessions (keep them)

1. **`-n 2000`** for anything about the expert arena (`-n 128` is drowned by warm-up), and **`-ncmoe` = the
   model's full layer count** (35B-A3B and Flash-Next 48, 122B-A10B 48).
2. **Interleave arms** in one batch; this box's clocks drift.  Never two GPU jobs at once.
3. **A speculative arm's t/s is (tokens/step) x (step rate)**: check `draft_n`/`draft_n_accepted` before reading
   a speed delta, and use `-n 512` (or more), never a short window.
4. **Static MTP depth ≤ 4** for the campaign's reference arms; the delivery's tuned controller is
   `--spec-type draft-mtp-adaptive` (the runme uses n-max 7).
5. **Quote the device count** with every MTP number.  The `AGENTS.md` reference arm is **3 devices** (dense 27B
   layer 51.7 / tensor 90.7 t/s); this box measures 1 device.
6. **`ROCBLAS_USE_HIPBLASLT=0`** for every fresh-process numeric comparison (the library picks its solution per
   process start), and never judge run-to-run determinism on a greedy text hash.
7. `llama-cli` prints its accounting **only with `-v`**, and `-v` interleaves log text into the generation: hash
   text from a run **without** `-v` (`scripts/extract-generated.py`), take timings from a separate `-v` run.
8. Always `--single-turn --no-display-prompt`, always `timeout`, and check the journal with
   `scripts/check-gpu-faults.sh` rather than grepping a log for `page fault`.
9. **`GGML_QSA_DERIVED_VIS=0` is not a no-op A/B.**  It disables the derived per-cell form at every width and
   also changes the prefill scorer's strip trimming / `LLAMA_QSA_SCORE_WMMA` auto choice; the r194 binary reads
   different greedy text with it too.  It is the kill switch, not a purity control; use
   **`LLAMA_INDEXER_NOBLOCK=1`** for the top-k block-vs-grouped-cell isolation arm.
10. A pooled cache's **park declines** (save-and-clear fallback).  That is deliberate and the park gate accepts
    it; do not "fix" it by growing the mirror to the park bound unless the maintainer chooses parking speed over
    the host-RAM win.

## 5. Gates — the ones this campaign owes

* **Pool**: text identity against the pool-off arm, `LLAMA_KV_POOL_VERIFY=1` (0 mismatches),
  `LLAMA_KV_POOL_STATS=<path>` (a path, never stderr), `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`,
  `gate-state-pipe.sh` on a **pooled** cache, `test-backend-ops -o FLASH_ATTN_QSA`, and `check-gpu-faults.sh`.
  A change to the mirror's growth also wants a **grown** arm (a 100k+ prompt) and a fresh-process slot-file
  restore (the `set_tensor` path).
* **Window / draft window**: text identity on a spilling arm, the standard prefill-logit point
  (`scripts/gate-prefill-logits.sh`, 0.000000 / 100.000 %), and for the draft window the MTP Protocol A arm
  (`-n 3000`, reasoning pinned, both split modes, device count quoted).
* **Indexer top-k / qwen4exp**: `test-backend-ops -o INDEXER_TOPK` and `-o FLASH_ATTN_QSA`, same-seed text on a
  pooled qwen4exp cache, the MTP arm, and a `rocprofv3` census (the `kv-kernel-census.py` short/long method).
* **Delivery**: `scripts/validate-set.sh` (16/16 strict apply + tree equality), `gate-per-block-build.sh
  --changed-from <previous tip>` (and the full `all` build at the tip), and `scripts/gate-prefill-logits.sh` on
  the 27B Q8_0 point before any tag.  A base is per ROCm line; a cross-toolchain compare is refused (#126).

## 6. Traps found the hard way (do not re-learn)

* **The delivery may move under you.**  r195 (PR #261) was published by another agent while A18 was in flight;
  always `git fetch` the delivery `main` (https, the ssh key does not work) and check `release.json` / the tag
  list before cutting a release.  A local `main` "ahead of origin" can still be behind upstream work.
* **`gate-ckpt-reuse.sh` only passes `--spec-type` when `DRAFT` is set.**  For an embedded-head MTP arm pass
  `EXTRA_ARGS="--spec-type draft-mtp --spec-draft-n-max 3"` (state and park gates take the same trick).
* **`gate-park-idle-slots.sh` defaults `DRAFT` to the Flash-Next sidecar.**  For any other model pass `DRAFT=`
  explicitly or the MTP context fails on `token_embd.weight`'s shape.
* **A windowed draft needs `-c > 32768`**, else the adaptive policy correctly declines and the gate proves
  nothing about windows.
* **At the VRAM cliff** (`Flash-Next`, `-c 262144`, `-ncmoe 48`) the 768 MiB default slab headroom is one
  hipBLASLt workspace short: the first prefill fails with `hipBLASLt out of memory` while the server stays
  healthy, and it is order-dependent, so it looks flaky.  `GGML_CUDA_SLAB_HEADROOM_MIB=1536` fixed it 3/3.
* **A harness that kills a subshell leaves the server running**: kill the server pid, and wait for the port to
  be free before the next arm.
* **`git push` over ssh fails on this box** (no working key): use the https URLs
  (`https://github.com/stew675/llama-cpp-rdna-boosts.git`,
  `https://github.com/stew675/llama.cpp.git`).  `fork-refresh.sh --push` checks for the ssh origin, so push the
  fork branch with an explicit https `git push --force-with-lease`.
* **Two different failure modes look like a hang in a gate**: a leaked server holding the port, and a run that
  aborted at model load because `-md` was forgotten on an MTP arm.  Read the log tail before rerunning.

## 7. Records

| file | what it holds |
|---|---|
| [`C1-MIRROR-GROW.md`](C1-MIRROR-GROW.md) | **r196**: the pool mirror grown on demand (#123) — implementation, numbers, gates, the park decision |
| [`C2-RADIX-TOPK.md`](C2-RADIX-TOPK.md) | **r196**: plain decode takes the top-k block radix path (#125) — the census, the purity arms, the `DERIVED_VIS` note |
| [`C3-NKVO-HOST-EXPERTS.md`](C3-NKVO-HOST-EXPERTS.md) | **r197**: `-nkvo` + host experts (#129, closed) — the split-pass root cause, the two-hunk block-09 fix, the block-15 window guard, the arms |
| [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) | the pool: design, invariants, §7-§10 measured |
| [`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md) | Goal B's plan of record, the placement matrix |
| [`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) | the window: design, purity hunt, spill cost, the `auto` decision |
| [`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md) | the tensor-split design (`#124`) |
| [`O3-DRAFTER-RING.md`](O3-DRAFTER-RING.md) | the MTP draft window (§2.3 is the delivered session) |
| [`HANDOVER-A18.md`](HANDOVER-A18.md) | the A17-era plan and the Goal A/B/C record (historical) |
| `patches/README.md` block 11 / 15 | the shipped descriptions, including the r196 subsections |
| `WORKLOG.md` 261101.001 | the r196 release record |
