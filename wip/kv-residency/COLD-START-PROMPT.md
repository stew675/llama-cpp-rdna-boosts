# Cold-session prompt — `wip/kv-residency` (TODO #118, remaining work)

Paste the block below into a fresh session.  It assumes only what is in the repo (nothing from a previous
session's context).  **Rewritten 2026-10-31, end of session A14.**

> **Goal A and Goal B are DELIVERED.**  Goal A = block 15 (`LLAMA_KV_RESIDENT_CELLS`, r178); Goal B = the same block 15
> (`LLAMA_KV_REGION_POOL`, r190) plus the two repairs it exposed (r189's `cpy_2d_vec16` in block 10, r190's
> quantized indexer keys in block 11) and B2 (a streamed cache takes the sparse decode arm, block 15).  `TODO`
> #118 is closed by `WORKLOG.md` 261031.008.  **Do not re-measure any of it** — `BA-POOL-DESIGN.md` §7-§10 and
> `A2-MIXED-BACKING.md` §3.7 hold the numbers, the gates and the commands.

 ---

 What is left is decisions and follow-ups, not a campaign:

 ## 1. Goal C — the MTP drafter ring (`O3-DRAFTER-RING.md`, `TODO` #121) — the maintainer's call

 Measured, **not** promoted: free at a usable window (46k context, `W=32768`: acceptance −2.3 % rel, generation
 flat), 7-9 % lost far below the working context, and the VRAM prize (one draft layer, ~1 GiB at `-c 262144`)
 does not reach the expert arena.  Three questions before any promotion: **default-on or opt-in, one window
 size or per-model, MTP-only or DFlash2 too.**  If it is promoted it becomes its own block (17) with a
 default-off rationale like blocks 15/16, and it owes the MTP draft-determinism gates
 (`benchmarks/mtp-adaptive-methodology.md`).

 ## 2. The Goal A `auto` policy (`A2-MIXED-BACKING.md` §3.7.11.4 A-C, `TODO` #122) — the maintainer's call

 The fit's ordering is settled (weights -> arena reserve -> `W` from the residue); what is not is which of the
 three measured `auto` policies ships as the recommendation.  No new measurement needed.

 ## 3. Follow-ups opened by r190 (`TODO` #123-#126)

 * **#123 grow-on-demand pool mirror**: the mirror is allocated in full at creation (2.4 GiB at `-c 196608`,
   ~4.9 GiB at 262144), bounded by `LLAMA_KV_SPILL_RAM_PCT`.  The model is block 15's host-half growth
   (`grow_cells`/`grow_for`).
 * **#125 the radix top-k** (0.9 ms/verify step under MTP, 2.2 ms/token plain at 103k on a sparse/pooled
   qwen4exp cache): the largest sparse-arm item left after A13 (which took the indexer score chain from 5.31 to
   2.01 ms/token).  Optional, needs its own campaign.
 * **#124 Goal A across `-sm tensor`**: designed only (`A6-TENSOR-SPLIT-WINDOW.md`), hardware-blocked on this
   one-card box.  Both blocks decline a tensor split with one message (the r181 crash fix) — do not remove
   those refusals without a multi-device arm.

 ## Rules that still apply

 * Every Flash-Next arm carries `-ot per_layer_token_embd.*=CPU -lm none -lzm off` and the separate MTP head.
 * `-n 512` and matched tokens/step before reading any t/s delta; static MTP depth <= 4.
 * `ROCBLAS_USE_HIPBLASLT=0` for every fresh-process numeric comparison.
 * One GPU job at a time; text from runs **without** `-v` (`scripts/extract-generated.py`), timings from a
   separate `-v` run; always `--single-turn --no-display-prompt`; always `timeout`.
 * A pool arm's gates are `LLAMA_KV_POOL_VERIFY=1` (0 mismatches), `LLAMA_KV_POOL_STATS=<path>` (a path, never
   stderr), the pool-vs-sparse text identity, and `gate-park-idle-slots.sh` / `gate-ckpt-reuse.sh` /
   `gate-state-pipe.sh` on a **pooled** cache (`patches/README.md` block 15).
 * Never push to upstream llama.cpp; the only push targets are `stew675/llama-cpp-rdna-boosts` and
   `git@github.com:stew675/llama.cpp.git`.
