# wip/batch-composition-determinism — PR #383 (fair-share rotation) and issue #384 (the 9..16-row verify band)

**NOT PART OF THE DELIVERY.**  Two assessments done in one session on `v18-919794500-r378`, on one gfx1201
(R9700, 32 GB), ROCm 10.2, Linux: a **contributor PR that is superseded** (PR #383, the fair-share rotation
restarting at slot 0) and the **issue it does not fix** (issue #384, two slots verifying in one 9..16-row
ubatch).  Both come down to one question — *when may the same tokens produce the same values* — so they are
kept together.  Nothing here is in `patches/`; no block, no gate and no knob changes.

## 1. PR #383 (`wip/prefill-fair-restart` by @overdoingism) — REJECTED: the mechanism was removed at r373

**The claim.**  `share_prompt_budget()` (block 16) advances `slot_next` on every step, decode-only steps
included, and never resets it, so two prompts that arrive together are split by a rotation left over from the
earlier requests; the same two-prompt request then gives different greedy replies from one request to the next
in one server.  The PR restarts the rotation at slot 0 on a step with no prompt to place
(`LLAMA_SERVER_PREFILL_FAIR_RESTART`, default on).

**Why it is rejected.**  The claim is correct for the tree it was measured on — **r368** — and that tree is ten
revisions stale.  **r373 (contributor PR #376) removed the mechanism**: the share's pass divided the *running*
`budget` (what the pass's earlier slots had already taken) instead of `budget_before`, so with two prompting
slots and a 2048-token window the one-token floor left 2046, the first slot of the rotated order took 1023 and
the second 511, and the pass closed at **1365 / 682** — alternating with `slot_next`.  That alternation is the
period-2 pattern in the PR's own table (`4224` `d988` `4224`): a request that starts at the other parity gets a
different per-step chunk *width* for both prompts, and the width moves MMVQ/MMQ tile boundaries into a near-tie.
r373 divides `budget_before`, so a pass gives every remaining slot `budget / n_active`: **1024 / 1024 on every
step, wherever the rotation is**.  Record: `INVARIANTS.md` r373, `DIAGNOSTIC.md` §22, `WORKLOG.md` 261127.026.

With `LLAMA_SERVER_PREFILL_SKEW = 1` (the default) the walk is **order-invariant by construction**: every
share in a pass is `floor(budget_before / n_active)` for every slot, the set that leaves a pass is decided by
`remaining` (not by position), and the residual budget is therefore order-independent too.  The rotation's only
reachable input left is the skew's tie-break when `LLAMA_SERVER_PREFILL_SKEW > 1` **and** two slots have equal
remaining tokens — a non-default tuning value.

**Measured** (`fair-share-repeat.py`, this directory; 27B UD-Q4_K_XL, `--parallel 2 --kv-unified -c 65536
-ub 2048 -ctk/-ctv q8_0 -fa on`, `LLAMA_SERVER_PREFILL_DUTY=1`, `ROCBLAS_USE_HIPBLASLT=0`, one `/completion`
carrying an array of two prompts = two tasks admitted in the same step, greedy, `cache_prompt: false`, sent 3x
to ONE server, 640 tokens each, prompts 10879 + 5246 tokens):

| build | prompt 1, requests 1 / 2 / 3 | prompt 2, requests 1 / 2 / 3 |
|---|---|---|
| r378 | `41a0c021` `41a0c021` `41a0c021` | `7d1428e1` `7d1428e1` `7d1428e1` |
| r378 **+ the PR's patch** | `41a0c021` `41a0c021` `41a0c021` | `7d1428e1` `7d1428e1` `7d1428e1` |

The patched build reproduces the unmodified build's hashes **bit for bit**: on the current tree the change is
inert, and the unmodified tree is already reproducible across requests (the same was measured on 4B Q8_0 at
`-ub 2048`).  A `-lv 5` per-step census added to `share_prompt_budget()` for the measurement reads
`budget = 1024 1024` on every prefill step of all three requests, with `slot_next` = 0, 1, 0 at their starts.
The patch applies to r378 (offset 26 lines), so applicability is not the problem.

The one remaining rotation input was forced to prove it is not the defect either: `LLAMA_SERVER_PREFILL_SKEW=4`
with two **identical** prompts and `n_predict 65` (so the parity flips between requests) gives request 1 the split
`1637 / 410`, request 2 `410 / 1637`, request 3 `1637 / 410` — and **identical greedy text** in all three,
because the two splits are mirror images of the same per-step amounts and a token's value depends on the width it
computes in, not on which slot's rows precede it.

**What a revived submission would need:** re-cut against the current `release.json` tree (the PR's artifact
pins r368), re-run the `-lv 5` census to show the split actually moves, and demonstrate an end-to-end effect.  The
class's gate is `scripts/gate-prefill-fairness.sh`'s `share split` check, which reads the per-step census; a text
hash alone is not the evidence this repo asks for.

## 2. Issue #384 (9..16-row verify ubatches) — mostly the documented trade; two claims corrected

**What the issue reports.**  With `--parallel 2 --kv-unified` and speculative decoding, two slots verifying in
one step share a ubatch of up to 16 rows (2 x (n_max + 1)), and from 9 rows on several ops leave the decode /
verify band, so a slot's tokens depend on whether another request overlapped it.  It lists the quantized
MUL_MAT (MMVQ to 8 columns, MMQ or few-row MMQ above), the band-only fusions (`add_rms_q8_1`, the GDN beta
sigmoid fold, the MMQ pair/triple/swiglu-down fusions) and FlashAttention's q8_0 9..16-row kernel, and it
proposes a "fix by halves" that measured +10.6 % wall time on the two-prompt request.

**This is the delivery's documented doctrine, and the boundary is deliberate.**
`GREEDY-PURITY.md` §11 records the exact measurement — *"`W = 1..8` is one logits class and `W = 9..16`
another (27B UD-Q4_K_XL, 2026-09-13, with FA disabled too, so it is not only the FA chooser)"* — and names
Cause B: the FA tile->WMMA switch at `Q->ne[1] > 8` and, with FA off, the `mmvq`->`MMQ` switch at `ncols == 8`;
keeping the guarantee there would mean giving up WMMA for 9..N-token batches.  §63 already states that a
**two-slot co-batched pair is not a determinism statement** ("a two-slot aligned pair is not stable run to run
either, so per-config determinism is a single-slot statement"), `INVARIANTS.md` r377 records the arrival-timing
dependence of a two-request pair, and `LLAMA_SPEC_ALIGN_WIDTH_BAND` (default 16) *deliberately* aligns two
drafting sequences' verify batches into one ubatch.  So the reported behaviour is expected, is not a regression
this delivery introduced (the kernel switch is upstream's), and the single-sequence arm is the guarantee that
is actually sold.

### 2a. Corrected: "every quantized type differs at 9..16, and q4_1 / q5_1 already differ at <= 8" — not reproduced

`mulmat-width-probe.cpp` (this directory) compares every column of an N-column `MUL_MAT` with the same column
computed alone (`ggml_mul_mat`, 12 quantized types, N = 1..16, bitwise on the fp32 output).  Weights fixed per
type, `src1` shared across N:

| M x K | q4_0 | q4_1 | q5_0 | q5_1 | q8_0 | q2_K | q3_K | q4_K | q5_K | q6_K | iq4_nl | iq4_xs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 x 4096 | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` |
| 512 x 2560 | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` |
| 5120 x 2560 | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` |
| 13824 x 5120 | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` | 1..8 `=` |

Every one of those shapes reads **N = 1..8 bit-identical to N = 1, first difference at N = 9**, and **q4_1 /
q5_1 are in-band pure on r378 / ROCm 10.2** where the report says they are not (the same holds at
`M x K` = 512 x 1024, 7168 x 2048 and 4096 x 5120).  The report's own 9-row boundary is confirmed exactly.

### 2b. Corrected: the boundary is per-shape, and the delivery's dense band pushes it to 16 columns

`ggml_cuda_mul_mat()` keeps the dense MMVQ band for an RDNA4/RDNA3_5 quantized weight whose output-row count is
**not** a multiple of 128 (`ggml_cuda_should_use_mmvq()` declines it, the band takes it, `ne11 <=
MMVQ_MOE_MAX_BATCH_SIZE = 16`, `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND` is the kill switch).  So on such a shape
**N = 1..16 is one numerics class**: measured `M x K` = 100 x 4096 and 32 x 4096, all 16 widths bit-identical,
and `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1` restores `first difference at N = 9` on the same shape.

That does not save the issue's whole-model arm — a layer whose row count *is* a multiple of 128 still leaves the
band at 9, and that is what its layer-1 observation is — but the accurate statement is **"the pure verify width
is 8 rows on the shapes that leave MMVQ at 9, and up to 16 on the dense-band shapes"**, not "9..16 always
differs".  `GREEDY-PURITY.md` §11's caveat list now carries this (it already says the 8-row figure is not a
shape-independent theorem).

### 2c. What would be actionable

Not a defect report, so nothing is opened as a bug.  Either of these would be:

1. a **single-sequence** arm whose reply depends on something other than its own context (the two-slot case is
   expected and documented), or
2. a **measured win** that pays for pinning the verify width: the issue's own "fix by halves" costs +10.6 % on
   the two-prompt request (+0.5 % single-slot) and the wide band exists because it is MMQ/WMMA.  A single-pass
   9..16-column kernel that keeps each column's summation order is the interesting form of that idea; it owes a
   `test-logits-width-probe` arm at `P = 3584` and `P >= 4096` plus a prefill-logit point, and the claim it would
   buy is two-slot determinism — a property §63 explicitly does not sell today.

## 3. Reproducing both

```bash
# 1. the fair-share repeat harness (PR #383): needs a built llama-server
python3 wip/batch-composition-determinism/fair-share-repeat.py \
  --server ~/llama.cpp/build-rocm/bin/llama-server --repeat 3 --tag r378
#    --same (both prompts identical), --skew N, --duty K, --n-predict N, --extra "-ncmax 5"

# 2. the op-level width probe (issue #384): build it against the same build
g++ -O2 -std=c++17 wip/batch-composition-determinism/mulmat-width-probe.cpp \
  -I ~/llama.cpp/ggml/include -I ~/llama.cpp/ggml/src \
  -L ~/llama.cpp/build-rocm/bin -lggml -lggml-base \
  -Wl,-rpath,$HOME/llama.cpp/build-rocm/bin -o /tmp/mulmat-width-probe
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 \
  GGML_BACKEND_PATH=$HOME/llama.cpp/build-rocm/bin /tmp/mulmat-width-probe 512 4096
#    add GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1 for §2b's control arm
```

Both harnesses print a verdict line; neither is a gate.  The probe's `M x K` pair is `output rows x reduction
width` (`ggml_mul_mat` on a `[K, M]` quantized weight), and its `=` / `X` grid is per column, so a single
near-tie can never be read as a band verdict.
