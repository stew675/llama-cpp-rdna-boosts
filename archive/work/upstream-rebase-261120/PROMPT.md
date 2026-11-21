# Cold-session prompt — resume the r328 re-base ladder at R06

Paste the block below into a fresh session.  It is self-contained; it names the files to read, the exact state,
the invariants, and the first action.  Kept in the repo so a later session does not have to re-derive it.

---

We are mid-way through re-basing the `rdna-boosts` delivery patch set onto a new upstream llama.cpp snapshot.
**Continue the ladder climb from R06.**  Work in `~/llama.cpp` (the fork checkout) and record everything in
`~/llama-cpp-rdna-boosts/archive/work/upstream-rebase-261120/`.

**Read first, in this order** (they are short and they are the contract):
1. `~/llama-cpp-rdna-boosts/archive/work/upstream-rebase-261120/HANDOVER.md` — state, what each rung cost, what is owed
   at R10/R17/R18, the traps that bit, the measured baselines, and the standing bias with its limit.
2. `…/README.md` §1 (the method and the traps) and §3 (session protocol).
3. `…/RUNGS.md` — the ledger plus the full resolution log of R00-R05.  Read R05.1-R05.3 before touching
   `fattn-*`: the delivery's FA band iteration model is the one in the tree and upstream's `#32407` rotation is
   deliberately absent.
4. `…/DECISIONS.md` — `D11` (the MMQ/MMVQ A/B you are about to start) and `D16` (how to decide a contested
   hunk).

**State, exactly.**  Branch `rebase-r328` in `~/llama.cpp`, clean, HEAD `c3860af1e` (tree `b52803164`).  Rungs
R00-R05 are done and green, plus two rung repairs (R00's top-k gather, R05's `nb32` widening).  The target is the
next release after `v17-4f9dfa0e3-r327` (r328) from fork point `919794500`.  **`patches/`, `release.json`,
`rdna-boosts-all.patch` and the fork's `rdna-boosts` branch are untouched — do not regenerate anything until
R18.**

**First action — R06:**

```
cd ~/llama.cpp && git cherry-pick f0564df7d     # block 06, MMB quantized prefill GEMM
# conflicts: ggml/src/ggml-cuda/ggml-cuda.cu, ggml/src/ggml-cuda/mmid.cu
```

Then: resolve, build, verify, log the rung in `RUNGS.md`, commit.  R06 starts **`D11`** — upstream retuned the AMD
MMQ configs (`mmq-config-gcn.cuh`) and the MMVQ `ne11` gate in this range.  **Measure both, pick the better of
each knob** (the maintainer's instruction; the plan table is in `DECISIONS.md` D11).  A mixed outcome — upstream's
table, our VDR — is the expected shape.  Never average two tables without re-measuring.

**Hard rules (violating any of these is how session 1 lost time):**
- One rung = one `git cherry-pick` + resolve + build + coherence + a `RUNGS.md` entry.  Never regenerate
  `patches/` mid-climb.
- **Build with a sentinel**, never by polling processes:
  `(cmake --build build-rocm -j 24 > /tmp/rNN.log 2>&1; echo "BUILD_RC=$?" >> /tmp/rNN.log)`.
- **Coherence oracle every rung**: `HIP_VISIBLE_DEVICES=0 llama-cli -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
  -ngl 99 -sm layer -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn`,
  hashed with `scripts/extract-generated.py` — it must read **`100 chars sha=1c5d32ac537d`**.  Then prove the
  rung is non-vacuous with the block's own switch (the arms and their baselines are in `HANDOVER.md` §5).
- **Never run two GPU workloads at once.**  `-fa on` is required for any KV-cache-type arm.  `-sm tensor` needs
  `HIP_VISIBLE_DEVICES` pinned (`--device` is not a valid flag at this base).
- **Docs are generated** (`llama-gen-docs`) — union, then regenerate, never hand-merge the tables.
- **Report before you drop anything.**  If a hunk is a genuine choice between upstream's and the delivery's
  version of the *same mechanism*, stop and ask the maintainer (that is `D15`/`D16` territory).  If it is the
  arithmetic or contract of upstream's *own new feature*, take upstream's (the `nb32` lesson).
- **`TODO` #353** is parked: Inkling-2 needs upstream PR #32695 (unmerged).  Do not try to load either Inkling-2
  file; nothing on this box can, at this base.

**When you stop** (end of session, or at a decision): update `RUNGS.md` for every rung touched, add any decision
outcome to `DECISIONS.md`, append to `HANDOVER.md` §0/§2 if the obligations change, and commit in
`~/llama-cpp-rdna-boosts` with a message that says which rungs and which decisions.  Do not leave a
half-resolved cherry-pick behind — `git cherry-pick --abort` and write down how far you got.
