# GREEDY-PURITY findings — the dated narratives behind every `§N`

This is the **evidence and narrative half** of [`../../GREEDY-PURITY.md`](../../GREEDY-PURITY.md).  The
rulebook keeps, per `§N`, the claim, the rule and the numbers; this file keeps the investigation --
*how* a finding was made, which instruments failed, and which false trails were ruled out.  Section
numbers and titles are the same in both files, so every `§N` citation resolves either way.

It was seeded 2026-09-12 (11) with the closed cases of that period and extended to hold the full
narrative for §§1-42 (last moved 2026-10-20, when GREEDY-PURITY.md was condensed to its summary form).

**Numbering note.**  Two sections were originally mis-numbered into already-used numbers and are kept
here under their historical numbers to preserve citations: `§39` (the `ssm_gate_beta` decode-only
fusion rule) and `§39` (the MoE expert-cache band) are distinguished in the rulebook as **§39** and
**§39b**; likewise **§40** (a thread count is not a numerical parameter) and **§40b** (a reserve graph
must match its stream count).  A citation written before 2026-10-20 that means the cache band is §39b.

---

## 1. The one-sentence version

Block 10 does not make decode any more or less *correct* than stock — it
computes the **same real-number result** through a **different order of fp32
additions**, which produces a different (equally valid) rounding path; stock
is not the mathematically special order, it is just the *reference standard*
that was chosen first and that everything else reproduces.

## 2. What block 10 changes

Block 10 is the single patch in the set that alters decode numerics on the
K-split decode paths (superseded for block-13-installed sets — see §9). It
does so through two mechanisms, both in the mmvq decode path:

| mechanism | change | effect |
|-----------|--------|--------|
| **VDR (vectors-per-thread reduction)** | `VDR_Q4_K_Q8_1_MMVQ 2→4`, `VDR_Q5_K_Q8_1_MMVQ 2→4`, `VDR_Q8_0_Q8_1_MMVQ 2→4`, `VDR_Q6_K_Q8_1_MMVQ 1→2` | each thread accumulates 4 (or 2) chunk partials serially instead of 2 (or 1) |
| **RDNA3_5 parameter table** (ex-block 06) | `MMVQ_PARAMETERS_RDNA3_5` split from the RDNA2 fallback, `nwarps=2` for Q8_0 on gfx1151 | fewer warps → different cross-warp combine tree |

Everything else in block 10 (the mmq prefill scale-load hoist, the MoE mmid
whitelist, the perf-harness cases) is arithmetic-preserving or test-only and
does not change any production numerics.

## 3. The reduction structure (identical machinery, different leaf assignment)

The decode kernel `mul_mat_vec_q` (in `mmvq.cu`) computes each output row as
a dot product of quantized x/y blocks. The dot products themselves are
**exact**: `ggml_cuda_dp4a` is an integer SIMD dot product, and the scale
multiplications `d8 * (dot1 * sc)` are single multiplies. The only place
rounding can differ is the **fp32 accumulation**:

```
// per-thread serial chain:
for (int kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    tmp[j][i] += vec_dot_q_cuda(...);        // ONE serial add per iteration
}
// cross-thread combine (identical in stock and block 10):
tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];   // across warps (shared memory)
tmp[j][i]  = warp_reduce_sum<warp_size>(tmp[j][i]); // across lanes (shuffle tree)
```

`warp_reduce_sum` is a fixed, hard-coded binary tree:

```cpp
for (int offset = width/2; offset > 0; offset >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, offset, width);   // 16, 8, 4, 2, 1
```

**Crucially, the cross-thread tree is byte-identical in both paths.** What
VDR changes is *which* partial sums feed which leaves of that tree, and how
many serial adds happen per thread before the tree runs.

## 4. Why VDR changes the rounding path

FP32 addition is **not associative**: `(a+b)+c` and `a+(b+c)` can round
differently. For a fixed row, the VDR value determines:

- `blocks_per_iter = vdr * nwarps*warp_size / qi` — how many chunks each
  thread owns per iteration (doubles from VDR=2 to VDR=4),
- the per-thread serial chain length (2 chunks then a `tmp +=` at VDR=2;
  4 chunks then a `tmp +=` at VDR=4),
- `kqs = vdr * (tid % (qi/vdr))` — which quant groups each thread touches.

So the *set* of partial products is identical, but the *association order*
of their summation differs: stock effectively rounds `((c0+c1)+(c2+c3))`
through the tree, block 10 rounds `(((c0+c1)+c2)+c3)` per thread before the
tree — and the thread-to-chunk assignment is different too. Same math,
different rounding events.

### A note on "fewer rounding accumulations"

If anything, the VDR=4 path does **fewer** serial rounding steps per thread,
not more: for the same K, `blocks_per_iter` doubles, so the outer `tmp +=`
chain runs half as often, and the internal adds amortize over more chunks
(~17% fewer serial fp32 adds per thread at VDR=4, measured from the loop
structure). This is not a claim of higher accuracy — it is simply evidence
that "more rounding happens in the boosted path" is not the correct
intuition. Both paths round a comparable number of times in comparable
magnitude ranges; neither is provably closer to the exact result.

## 5. What "correctness" means here — and what it doesn't

- **Same real-number result.** In exact arithmetic, every VDR configuration
  computes the identical dot product. There is no approximation being
  introduced — the products are exact, the scales are exact, the sums are of
  the same terms.
- **Stock is not more correct.** The upstream ordering was chosen by the
  original author for performance and simplicity on the hardware of the day,
  not because it is the mathematically optimal summation order. There is no
  correctness bound, no error analysis, no "canonical" association in the
  reference implementation — it is simply the ordering that shipped first.
- **Block 10 is not less correct.** It is a different, equally deterministic
  path through the same fp32 approximation space.
- **"Bit-identical to stock" is a reproducibility requirement, not a
  correctness requirement.** It matters when you need byte-exact comparison
  against stock output: golden test vectors, cross-build test harnesses,
  speculative-verify batches that must match decode (the `ncols_dst <=
  MMVQ_MAX_BATCH_SIZE` rule), or audit trails. If your goal is to reproduce
  stock's bits, you must reproduce stock's exact association order — that is
  the entire reason block 10 is structured as one excludable patch.

## 6. Speculative decoding: same kernels, same drift, orthogonal composition

Speculative decoding (draft + verify) does not add a separate source of
variance — it runs the *same* mmvq decode kernels on a wider batch. The
target model verifies the draft tokens in **one batch of `n_draft+1` rows**
(default `n_draft = 8`, so up to 9 rows), which hits the identical
`mul_mat_vec_q` kernels as single-token greedy, just with
`ncols_dst = n_draft+1`. Acceptance then compares each sampled token
against the draft:

```cpp
for (; i < draft.size(); i++) {
    const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], ...);
    if (draft[i] != id) break;    // acceptance decision
}
```

Two invariants make block 10's interaction with speculation safe and
bounded:

1. **VDR is per-type only**, not per-column-count: `get_vdr_mmvq(type)`
   returns the same value for n=1 (single-token decode) and n=2..8
   (verify batch). Block 10 changes the *values* uniformly, so both paths
   use the same VDR=4 kernels.
2. **Block 08's `ncols_dst <= MMVQ_MAX_BATCH_SIZE` rule** forces the
   verify batch onto the same nwarps as decode, so its per-row
   accumulation is bit-identical to one-at-a-time decode *within a
   build*.  The nwarps **value** is per `(type, K)` since the
   2026-09-12 (18) block-13 amendment (dense Q8_0 `K < 4096` -> 8, else
   1, with the pinned fusion ops kept at the band-uniform 1); K is fixed
   for a given weight, so the band stays uniform per weight and the
   invariant is untouched.

Together these guarantee the distribution-preservation property that
speculative decoding relies on: the acceptance criterion reproduces the
target model's distribution exactly — with or without block 10. Block 10
changes the *rounding path* of that distribution, uniformly across both
paths; it does not break the consistency between them.

The interaction therefore splits along the same line as plain decoding:

| | temp = 0 (greedy) | temp > 0 (stochastic) |
|---|---|---|
| **acceptance decisions** | draft accepted iff `argmax(target logits) == draft` — a deterministic comparison; a ≤0.184 logit drift can flip near-tie acceptances | acceptance is a random draw from the (slightly shifted) distribution; the drift perturbs *probabilities*, not decisions |
| **block 10's effect** | can change which tokens are accepted in near-ties → a different greedy stream (the 1-of-3-prompts divergence, same as plain greedy) | shifts acceptance probabilities by ≤~20% relative on the worst token, typically far less; the RNG draw varies run-to-run by more |
| **correctness impact** | none — both are valid deterministic paths | none — the sampling *distribution* is what matters, and PPL is bit-unchanged |

**The one nuance:** at temp>0 the drift does not vanish — it slightly
changes the *acceptance rate* (how many draft tokens get accepted per
block), a marginal throughput effect. It does not change what the model
produces in any meaningful sense: the output distribution is preserved
either way, which is the entire point of the spec-decode acceptance
criterion.

**Bottom line:** block 10 + speculation = block 10 without speculation,
from the output-distribution standpoint. Speculation is an accelerator;
block 10 is a rounding-path choice; they compose orthogonally. The only
place the composition shows up is greedy acceptance (temp=0), where the
≤0.184 drift can flip near-tie acceptances — the same bounded,
characterized divergence described throughout this document.

## 7. Measured magnitude of the variance

From the validation runs on gfx1201 (ROCm 7.14):

| metric | value | meaning |
|--------|-------|---------|
| max |logit| diff (block 10 in vs out) | **0.184** | the largest single-logit change anywhere in the test prompts |
| same, for flash-attn on vs off (yardstick) | 0.203 | a variance source already accepted upstream — block 10's drift is comparable to it |
| per-token probability change at the max-diff logit | ~20% at t=1.0 | *modest*, not astronomically small — see note below |
| logit magnitude in the top region | ~10–30 nats | 0.184 is ~2 orders of magnitude below the logit scale |
| typical top-2 logit gap | ~0.5–5 nats | 0.184 is ~9% of a 2-nat gap — enough to flip greedy *only* in near-ties |
| PPL (wikitext-2, Q8_0 model) | 6.3162 / 6.3563, **identical** to pre-block-10 | the aggregate output distribution is unaffected |
| greedy streams | deterministic *within* a build; 1 of 3 test prompts diverged at 1 character | flips only when two logits sit within the drift of each other |

### Honest framing of the numbers

An earlier draft of this note described the drift as "four orders of
magnitude below the temperature scale." That phrasing is **not accurate** and
is not used here. A 0.184 logit difference is ~20% relative probability on
the single most-affected token at t=1.0 — small relative to the logit scale
(~2 orders below typical magnitudes) and small enough that PPL is
bit-unchanged, but it is *not* vanishingly small. What keeps it benign in
practice:

1. It is a **max** over all tokens; typical logits drift far less.
2. PPL — the aggregate measure of output quality — is bit-identical.
3. Greedy output changes only when the top-2 logit gap happens to be smaller
   than the drift (a near-tie), which is why 1 of 3 prompts diverged at a
   single character and the other 2 did not diverge at all.

If you require bit-exact greedy reproduction of stock, exclude block 10. If
you require a *correct, deterministic, high-quality* model, block 10 is
unambiguously fine: it is one of many valid rounding paths, and the evidence
shows it neither helps nor hurts what the model produces — only whether the
bits match stock's arbitrary-but-pinned order.

## 8. Practical guidance

- **Within a single build:** greedy output is fully deterministic, with or
  without block 10. The variance only appears when comparing *different
  builds* (block 10 in vs out, flash-attn on vs off).
- **Reproducing stock bits:** exclude `0010-rdna-boosts-block-10-k-quant-boosts-Q4_K-Q5_K-Q6_K-Q.patch` (it is
  applied last; one-line change in `scripts/apply-all.sh`). This restores
  every VDR constant and the RDNA3_5 table to stock values, so the reduction
  order matches upstream exactly, on RDNA3, RDNA3_5 and RDNA4 alike.
- **Mixing the two:** do not apply block 10 partially or hand-tune VDR
  values in production builds that must match stock — any deviation from the
  stock association order is a new rounding path and breaks bit-reproduction
  just as surely as block 10 does.
- **Comparisons:** when A/B testing block 10 against stock, compare
  *distributions* (PPL, benchmark scores, sampling quality), not individual
  greedy tokens — individual tokens are expected to differ in near-ties, in
  exactly the same way they differ between flash-attn on and off.

## 9. Block 13 changes decode numerics on its rewritten mmvq rows (addendum, 2026-09-02)

Sections 1-8 were written for the 12-block set. Block 13 (`0013`, amended
2026-09-02 with the two MTP regression fixes) adds a second source of
decode-numerics change, so the "block 10 is the single patch" statements in
§2 and the §8 stock-reproduction guidance are **superseded for any set that
includes block 13**:

1. **What block 13 rewrote.** Block 13 replaced the §3 K-split loop in the
   small-batch decode kernel `mul_mat_vec_q` with an item-split over
   (token, row, kblock) plus an RDNA `rows_per_block` override. On the rows
   that run it, the item-split feeds *different partial sums* into the §3
   cross-thread tree — the same class of fp32 reordering as block 10's VDR,
   and the same same-seed consequence: greedy streams are deterministic
   within a build but can flip vs stock (observed: 13-block vs 12-block
   outputs diverged on near-ties).
2. **What the 2026-09-02 ksplit fix restored.** ncols 2..8 (the speculative
   verify batch) and ncols==1 rows with K >= 4096 now dispatch to the
   pre-block-13 K-split kernel (`mul_mat_vec_q_ksplit`), restoring the §3
   reduction structure on those rows — dense-model decode is again
   byte-identical to the 12-block build (verified: dense qwen35 same-seed
   outputs match). The remaining block-13 kernel surface is ncols==1 rows
   with K < 4096 (MoE down/router and other short-K projections on RDNA2+
   tables); those rows still carry block-13 variance vs stock.
3. **MoE multi-token decode.** The second 2026-09-02 fix gates the
   block-08 rms_norm->mmvq Q8_1 quantize-cache fold off multi-token
   MUL_MAT_ID (it corrupted the cached y on that path — MTP acceptance
   0/1527). Batched MoE decode now follows the unfused path, restoring
   verify==decode consistency (acceptance 0.51). Separately, MoE
   single-token decode carries fusion-ordering drift (fused vs unfused
   differ on near-ties; deterministic within a build; accepted trade-off).
4. **Practical consequence.** Reproducing stock bits with block 13
   installed is not a one-line drop like block 10: the K<4096 ncols==1
   mmvq rows run block 13's item-split kernel (no env switch; the ksplit
   dispatch condition in `mul_mat_vec_q_switch_ncols_dst` is the gate).
   Dense long-K decode (the common case) is back to block-10-only variance;
   MoE/short-K decode carries block-13 variance. Full-set greedy output
   remains fully deterministic within a single build.

## 10. Block 00 makes the verify width irrelevant on the small-batch path (2026-09-10)

Block 00 (the structural block) removes a second, previously separate
source of greedy variance: the flash-attention KV split. `launch_fattn`'s
`parallel_blocks` heuristic keyed off `ntiles_dst`, a function of
`Q->ne[1]`, so single-token decode (`n_q = 1`) and a speculative verify
batch (`n_q = 3`, `5`, …) grouped the fp32 online-softmax / PV partial
sums differently and produced different logits.  `--spec-draft-n-max 2`
and `4` therefore streamed apart at near-ties even though both ran the
same kernel instantiation (issue #25).  Block 00 evaluates the heuristic
as if `n_q == 1` for every `n_q <= 8`; decode and every verify width now
reduce identically, and plain decode is byte-identical (only `n_q >= 2`
moves).  `n_q = 1` vs a verify batch can still differ because the FA
*kernel* is selected from `Q->ne[1]` (decode may take the VEC kernel);
that is a decode-vs-verify difference, not a draft-length one.

## 11. The real purity range, and its two causes (2026-09-11 correction + fix)

**The guarantee is `--spec-draft-n-max <= 7` in every configuration** (the boundary is on the *verify
batch width*, not on `n_max`: the target verifies the drafts plus the last committed token, so `K = n_max + 1`
and `n_max = 8` is already a 9-token batch).  Earlier notes claimed `n_max <= 15`; they had only ever
been validated to `n_max = 4`.  Two independent causes bound it:

> **2026-09-13 (issue #30) update.**  This section's *guarantee* is unchanged, but the CLI's policy is:
> the draft depth is **no longer clamped at 7** — `--spec-draft-n-max 8..15` is kept with a visible
> notice that `plain` vs `draft-mtp` may no longer be bit-identical, and only `> 15` is clamped (the
> recurrent rollback snapshot bound, a *correctness* bound, not this purity one).  Cause B is now known
> to be broader than the FA chooser: the real models also flip at `W = 9` with flash attention
> **disabled**, because the matmul family changes at `ncols == MMVQ_MAX_BATCH_SIZE`/`MMVF_MAX_BATCH_SIZE`
> = 8 (27B UD-Q4_K_XL, f16, P=2500: W=1..8 `8ef5ce3ab2d942dd` vs W=9..16 `7d1e01e82be4ce6b`; `FA=0`
> `b5d4df87caa0348e` vs `0c0cb329b59aedc9`).  A third, fork-specific cause (the qwen4exp QSA dense arm
> flipping to sparse above `W = 8`) is fixed in block 14 — see §32; the deterministic recurrent
> snapshot sweep (`tests/test-recurrent-state-depth`) shows there is **no corruption** in 1..15.

| config | guaranteed for | binding cause |
|---|---|---|
| 1 GPU (no split) | `n_max <= 7` (W <= 8) | B |
| 2-GPU `-sm layer` / no split | `n_max <= 7` | B |
| 2-GPU `-sm tensor` | `n_max <= 5` **before** the §11 fix, `n_max <= 7` after | A, then B |
| 3-GPU `-sm tensor` | `n_max <= 7` | B |
| 2-GPU `-sm tensor` + `GGML_CUDA_ALLREDUCE=meta` | `n_max <= 7` | B |

Probe hashes (27B Q8_0, `RS=0`, P=256; identical within a row = bit-identical token-0 logits):

| config | `W = 1..8` | `W = 9` (= `n_max 8`) |
|---|---|---|
| 1 GPU | `4089b4d4` | `72af52db` |
| 2-GPU `-sm layer` | `4089b4d4` | `72af52db` |
| 2-GPU `-sm tensor` | `a4817ee6` | `b059daa6` |
| 3-GPU `-sm tensor` | `91434ea9` | `bc3faabd` |

* **Cause A (fork-specific, FIXED 2026-09-11): block 12's size-based all-reduce dispatch.**
  `ggml_backend_cuda_comm_is_small()` routed a reduction to the internal host-staged pipeline below a
  per-device-count element count (32768 for 2 devices) and to NCCL above it, and the two are **not
  bit-identical** (different summation order; the internal pipeline always BF16-round-trips).  Under
  `-sm tensor` the reduced tensors scale with the batch width, so a 7-token verify (35840 elements)
  crossed the old 2-device limit while 1..6-token decode stayed below it.  The 2-device crossover is now
  131072 (the 3-device value); `GGML_CUDA_ALLREDUCE=internal` reproduces the fixed hashes for `W=1..8`,
  only `W = 7,8` moved, and MTP throughput *rose* (+12 % at `n_max 6` and at `n_max 12`) because the
  internal pipeline is faster at these sizes.
* **Cause B (deliberate; by design): the flash-attention tile-vs-WMMA switch at `Q->ne[1] > 8`.**
  Beyond `W = 8` the launcher prefers the much faster WMMA kernel and its reduction order differs.  The
  fork's own comment requires speculative verify batches (`n_q <= 8`) to stay on the tile kernel, so
  `n_max <= 7` **is** the designed guarantee; removing Cause B would mean giving up WMMA for 9..N-token
  batches.  `MMVQ_MAX_BATCH_SIZE = 8` sits at the same boundary, so Cause B is at `W <= 8` whichever of
  the two fires first.

**Rules.**  Use the **raw-logit probe, not the text gate, to establish a boundary** — on 3 GPUs the
300-token greedy text at `n_max = 8` matched plain while the logits had already diverged (text equality
is evidence *for* purity, never *against* divergence, and near-ties are rare).  Above the range, gate on
acceptance + MTP-vs-plain throughput instead (`benchmarks/mtp-adaptive-methodology.md`); adaptive MTP's
`n_max = 12` is deliberately outside the guaranteed range.  Upstream is **worse**, not better: on
`9cf3bf256` (CPU, 4B Q8_0, P=256) `W = 1` gives `9024dd2e…` while `W = 2..12` all give `3cd0eb0e…`, i.e.
it diverges at the first width step (its `calc_nwarps`/`MMVQ_MAX_BATCH_SIZE` dispatch is the same shape).

Records: `patches/README.md` block-12 notes, the 2026-09-11 WORKLOG entry,
`archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.  Narrative (the `n_max` sweep, the
localisation code block, why the GDN was not the boundary): `../archive/docs/GREEDY-PURITY-FINDINGS.md` §11.

## 12. The guarantee depends on the KV cache type (2026-09-11, measured during the Block-15 revalidation)

> **The `q8_0`/`q4_0` impurity below is FIXED** (2026-09-11, block-08 amendment) — it was the FA
> *kernel-family* chooser, not the KV staging; see **§14**.  The §12 measurements stand as the
> pre-fix record, and its F16-staging explanation of the slow types is refined in §14.

Everything above was measured with an **f16** (or bf16) K/V cache.  Extending the probe matrix to every
same-type KV pair (`tools` + evidence: `archive/work/kv-quant-purity-followups/README.md`) shows the guarantee is
**not universal in the cache type**:

| K/V cache (same type) | `W = 1..8` (i.e. `n_max <= 7`) | notes |
|---|---|---|
| f16, bf16 | **pure** | the reference configuration |
| q4_1, q5_0, q5_1, iq4_nl | **pure** | no native FA kernel — they stage through the F16 scratch, ~3.4x slower |
| **q8_0** | **impure**: `W=1 == W=2`, `W=3..8 == W=3..8`, but the two groups differ | the boundary is `W=2→3`, *not* block 00's `n_q <= 8` |
| **q4_0** | **impure**, same shape as q8_0 | ditto |

Text-level impact (27B, 3-GPU `-ts 1/1/1`, ctx 8192, 300 greedy tokens): with a q8_0 cache, `--spec-type
none` gives `8ed58aa9` and `draft-mtp --spec-draft-n-max 3` (== `7`) gives `da56855b` — a real
divergence, not a near-tie.  The f16 control is `ce7b9a75` for all three (pure).

Two facts worth keeping straight:

* This is **pre-existing** and **not** a delivery-block effect: the same hashes come out of a build
  without any of the blocks' F1/F3-relevant code (verified: identical on the 15-block delivery build
  and on the block-15 tree).  `GGML_CUDA_FA_KV_NATIVE` on/off is identical too, so it is not V4's
  native staging; 1 GPU alone reproduces it, so it is not the all-reduce.
* The impure set is exactly the two KV types that have a **fast native** both-quantized FA path
  (>7700 t/s pp512).  So this is a *trade*, not an oversight: the fast implementations are
  width-dependent, the generic (F16-staging) ones are not.  Any future native path (see the sub-q8_0
  parity item) must be built width-invariant by construction.

**Guidance (as of the §12 measurements; `q8_0`/`q4_0` are fixed — see §14).**  For plain-vs-speculative greedy purity, use **f16 or bf16** for K and V.  If a q8_0/q4_0
cache is required, treat plain-vs-spec text equality as *not* guaranteed and gate on adaptive-MTP
acceptance/throughput instead.  Mixed K/V *types* are a rejected configuration (see
`archive/work/block-15-campaign-wins/README.md` §7) and are irrelevant to this table.  This table is orthogonal
to Causes A and B in §11: those are about the *fork's* widths and the FA kernel switch, this is about
which dequant kernels the cache type selects.

## 13. qwen4exp and the hyper-connection band (2026-09-11, Block 14 amendment)

qwen4exp's fused decode chain (`GGML_OP_HC_MIX` / `GGML_OP_HC_COMBINE`, replacing
`SCALE/SILU/MUL_MAT/SIGMOID/MUL/ADD`) was gated to a **single-token** batch, so a 1-token decode and an
n-token verify took different arithmetic (per-node dump: 98 `HC_COMBINE` dispatches at `W=1`, **0** at
`W>=2`).  **Fixed** by block 14's 2026-09-11 amendment: the whole **decode/verify band `1 <= nt <= 8`**
(§the `HC_FUSED_MAX_TOKENS` constant) runs the fused ops with the token index on `blockIdx.y`, so every
token in the band runs the per-token kernel sequence a single-token decode runs.

Measured then (3 GPUs, f16 KV, P=256, `RS=0`): `W = 1..4` was **all one hash** on both splits
(`3adeb313042a871b` layer, `dcf1ae667f730879` tensor) and the `W >= 5` grouping was **cause 2**, fixed
later by §15 — so the band is now `n_max <= 7`.  Two caveats: a `<= 8`-token **prefill** chunk also takes
the fused path (indistinguishable from a verify batch — that is the point), and with a `q8_0`/`q4_0` **KV
cache** the cache's own impurity (then §12, fixed in §14) dominated.

Narrative + the per-width hash table: `../archive/docs/GREEDY-PURITY-FINDINGS.md` §13.

## 14. F1 fixed (2026-09-11, block-08 amendment): the decode/verify band no longer spans two FA kernel families

§12's `q8_0`/`q4_0` impurity was the FA **kernel-family** chooser, not the KV staging:
`ggml_cuda_get_best_fattn_kernel()` returned **VEC** for `n_q <= 2` with a quantized K/V and **TILE**
from `n_q = 3`, and the two families order the online-softmax/PV reduction differently, so `W = 1,2`
disagreed with every verify width.  Both VEC conditions sit inside the `n_q <= 8` band, so the branch is
**deleted** and the band is TILE throughout — the same shape as the block-08 WMMA guard and block 00's
`ntiles_dst_eff` (which is why every earlier F1 suspect measured clean).

| K/V cache (same type) | before | after §14 |
|---|---|---|
| f16, bf16 | pure | pure, hashes **bit-identical** (never took VEC) |
| q4_1, q5_0, q5_1, iq4_nl | pure | unchanged (they take the generic fallback — see the F3 reframing) |
| **q8_0**, **q4_0** | `W=1,2` != `W=3..8` | **`W=1..8` bit-identical in all four split configs** |

Measured (3x gfx1201): 4B q8_0 1 GPU `W=1..8` all `31a0c1bace68`, 2-GPU tensor `abebfb93`, 3-GPU tensor
`7fe106f5`; 27B Q8_0 3-GPU `W = 1,2,3,4,5,8` all `d4156dbeb225`; text level (27B, q8_0 KV) plain ==
`n_max 3` == `n_max 7` = `3537bc2b36be`, f16 control `f32aac948600`.  Every new value equals that
config's **previous verify** value, so only `W = 1,2` moved: MTP is bit-unchanged (27B acceptance
0.90789) and the cost is decode-only (tg128 -0.9 % 4B / -0.5 % 27B, pp512 -0.2 %).  **The pure range is
still `n_max <= 7`** — now for *every* supported KV type.

**F3 reframing.**  §12's "no native FA kernel" explanation is wrong in detail: `q4_1`/`q5_0`/`q5_1`/
`iq4_nl` are rejected by `ggml_cuda_fattn_kv_type_supported()` unless `GGML_CUDA_FA_ALL_QUANTS` is
enabled, so the chooser returns `NONE` *before* any VEC/TILE choice and attention takes the generic
fallback — width-invariant by construction (hence pure) and ~3.4x slower.  F3's first experiment is a
build-flag A/B of `GGML_CUDA_FA_ALL_QUANTS=ON` plus §14's band rule.

**Harness lesson.**  `llama-cli`'s `\/|` spinner is `\b`-based and timing-dependent and the ASCII
banner embeds the build SHA: apply backspaces, strip the banner and the `[ Prompt: … | Generation: … ]`
footer before hashing output — and always run a control that must agree before believing a divergence.

Narrative: `../archive/docs/GREEDY-PURITY-FINDINGS.md` §14.

## 15. F2 cause 2 fixed (2026-09-11, block-13 amendment): the MoE decode/verify band is band-uniform

§13's `W >= 5` grouping was **upstream's per-type mmvq cap** (`get_mmvq_mmid_max_batch_*`), not the
gate+up+GLU fusion coverage the first pass blamed.  The cap does two things: (1) it sizes
`mul_mat_vec_q_moe`'s `__launch_bounds__` (`cap × warp_size`) while the block is
`(warp_size, ncols_dst)`, so it is a *capability* limit (an `IQ3_S` launch with `ncols_dst = 5` exceeds
the bound and aborts); (2) it chooses mmvq vs MMQ (`ne2 <= cap → mmvq`, else `should_use_mmq → MMQ`) and
gates the `mul_mat_q_pair` fusion.  **mmvq and MMQ reduce in different orders**, so every cap boundary
inside the band is a numeric boundary.  It is a *graph-identical* bug — `[GD]` dumps show the same node
counts (…2647/1668/1565) at `W=4` and `W=5` with the MoE matmuls at the same indices — and the quant is
what exposes it: the UD-IQ4_XS file mixes expert types per layer (47 layers `IQ3_S` gate/up → cap 4;
layer 2 `IQ4_XS` → cap 5; down `IQ4_NL`/`Q8_0` → cap 7), which predicts the observed
`{5} {6,7} {8}` grouping exactly (fused layers 48/48/48/48/1/0/0/0 for `W = 1..8`).

**Fix** (block 13 amendment, completing block 13's own `has_ids` "decode == verify invariant"): floor the
cap at `MMVQ_MAX_BATCH_SIZE` for every AMD lookup and size the MoE kernel's launch bound at the band.
Every call site is `MUL_MAT_ID`-only, so dense models cannot be affected (verified).

| split | W=1..8 (f16 KV, P=256, `RS=0`) |
|---|---|
| `-sm layer`  | **all `3adeb313042a871b`** (= the pre-fix `W = 1` decode) |
| `-sm tensor` | **all `dcf1ae667f730879`** (= the pre-fix `W = 1` decode) |

Every width moved onto that split's pre-fix `W = 1` value (the F1 "move the cheap side" pattern); also
pure with `RS=from_w`.  At `n_max 3` (`W = 4`, a no-op width) pre/post runs are **byte-identical**
(acceptance 0.76744, 80.0 vs 80.1 t/s), and at `n_max 7` it is **+16-18 % t/s** (acceptance 0.59375 vs
0.55556; `n_max 3` == `n_max 7` text `8a50ea24e8d5`, previously disagreeing).  It is also a large
throughput win at exactly the verify widths (`llama-batched-bench`, fixed/baseline):

| model | b1 | b2 | b4 | b5 | b6 | b7 | b8 |
|---|---|---|---|---|---|---|---|
| qwen4exp 3-GPU tensor | 50.5/50.4 | 85.4/85.6 | 134.1/132.9 | **149.5/118.4** | **162.5/130.6** | **171.4/147.0** | **178.0/155.4** |
| 35B-A3B MoE 1 GPU | 98.3/98.1 | 156.4/156.2 | 254.2/254.1 | – | – | – | **341.3/289.9** |
| 4B dense 1 GPU | 100.6/100.5 | 167.2/167.3 | 294.0/294.9 | – | – | – | 414.5/413.3 |

So upstream's per-type mmvq caps were costing 14-26 % at exactly the speculative-verify widths on RDNA4
with the fork's mmvq + fused-GLU kernels.  (Cause 3 — the qwen4exp *text* divergence this section
recorded as open — is fixed in **§16**.)

Narrative: `../archive/docs/GREEDY-PURITY-FINDINGS.md` §15.

## 16. Cause 3 fixed (2026-09-11, block-14 amendment): the QSA decode arm is band-uniform

After §13-§15 the dense models, the MoE and qwen4exp's hyper-connection band were pure, but qwen4exp
still had a **text** divergence (`plain` vs `draft-mtp`), and it was neither a kernel nor the sparse-FA
attention: `LLAMA_QSA_OFF=1` made both runs byte-identical (so the defect is in the QSA **indexer**
machinery), `LLAMA_QSA_SPARSE_FA=0` did not fix it (the kernel is exonerated), and the single-step width
probe is pure on both splits — its blind spot is exactly this bug (it keeps `n_kv` below the QSA
selection width, so every width takes the same arm).

**Mechanism** (`build_layer_attn`): the indexer has three arms — (1) dense + store keys while
`n_kv <= width`, (2) the dense arch-policy arm, (3) the sparse top-k selection — and arm 2 was gated on
`n_tokens == 1`.  `width = indexer_top_k + r - 1` = 2051 on qwen4exp, so from the first decode graph
(`n_kv = 2304 > 2051`) arm 1 no longer applies and the two runs split: `--spec-type none`
(`n_tokens=1`) took **arm 2 (dense)**, `draft-mtp n_max 3` (`n_tokens=4`) took **arm 3 (sparse)** — the
same state, two attention regimes, purely because of the batch width.

**Fix**: `QSA_DECODE_BAND = 8` (the `n_max <= 7` purity band, the same constant class as
`HC_FUSED_MAX_TOKENS`); arm 2 now takes `n_tokens <= QSA_DECODE_BAND`.  Prefill is untouched (it is far
above the band and keeps arm 3 — the arch policy is "prefill is always QSA", §26).  Measured:
`plain == n_max 3 == n_max 7` = `804de0576868` (704 chars, f16 KV) and `plain == n_max 3` =
`75d8530c5bb1` (660 chars, q8_0 KV).  The plain stream moved with the fix (658 -> 704 chars) because the
shared 4-token non-decode shape at `n_kv = 2304` also moved onto the dense arm — the same "the band must
take one path" trade as §14, and the value chosen is the policy-consistent one.

Narrative (the arm trace and the localisation steps): `../archive/docs/GREEDY-PURITY-FINDINGS.md` §16.

## 17. The MoE shared-expert epilogue is band-uniform too (2026-09-11, block-13 amendment)

After §15, Qwen3.6-35B-A3B was still strictly `W=1 ac8825358d9adfda` vs `W>=2 bd138ad2326fbbf2`: the
fused shared-expert down epilogue (`dst = down(swiglu) * sigmoid(gate(x)) + moe_out + ffn_residual`) was
gated `down_mm->src[1]->ne[1] == 1 && gate_mm->src[1]->ne[1] == 1` (decode only) with an in-code note
that the fused gate reduction does not reproduce the standalone mmvq/MUL_MAT order.

**Fix**: the kernels are now token-generic and the **whole band takes the fused path** (keeping the
+3.1 % decode win instead of discarding it):
* `shexp_gate_sigmoid` — one warp per token, the token index only selecting the input column;
* `shexp_down_gated_q8_0` — one block per `(output row, token)`, addressing the epilogue operands at the
  token's offset;
* **`nwarps` is pinned to the single-token value** (`calc_nwarps(type, 1, table_id)`): `calc_nwarps`
  returns 4 for `ncols_dst 1..4` but 2 for `5..8`, and `nwarps` sets `blocks_per_iter`, i.e. the
  reduction order — the same trap as §15's cap.  Pinning it is what makes every width bit-identical;
* the fusion arm accepts `1 <= ne[1] <= MMVQ_MAX_BATCH_SIZE`; `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`
  still selects the unfused reference.

**Measured**: default probe `W = 1,2,3,4,8` all `ac8825358d9adfda` (the pre-fix fused value); with the
kill-switch all `bd138ad2326fbbf2` (uniform unfused = the pre-fix `W>=2` value); qwen4exp unaffected
(`plain == n3 == 804de0576868`).  The asterisk is gone: MoE decode and verify take one arithmetic, so
`n_max <= 7` covers MoE.  The epilogue's later column-blocked restructure (2026-09-12) repaid the
band's `pl=8` cost — see **§24**.

Narrative: `../archive/docs/GREEDY-PURITY-FINDINGS.md` §17.

## 18. The QSA *sparse* regime on gfx1151: the MTP-export logits ULP and the q8_0/q5_0 residual are both root-caused and fixed (2026-09-12; extended 2026-09-12 (6); closed 2026-09-12 (12); gfx1151-validated 2026-09-12 (14))

§18 previously recorded **two** width-dependences in the QSA sparse regime, measured 2026-09-11 during
the cause-3 hunt (on the 3-GPU gfx1201 box, with the sparse arm forced by
`LLAMA_QSA_DENSE_DECODE_UNTIL=0`) — i.e. **before** the 2026-09-12 block-13 RDNA3_5 mmvq-fusion
amendment (§25).  Re-measured on gfx1151 with the current delivery, **neither reproduces**:

1. **The fused indexer score is byte-identical to the per-op chain.**  A 512-token forced-sparse A/B
   (qwen4exp UD-IQ4_XS, f16 KV, `P=5000`) gives the *same* text for the fused score (default),
   the per-op chain (`GGML_CUDA_QSA_INDEXER_SCORE=0`) and the no-derived-cache form
   (`GGML_CUDA_QSA_INDEXER_CACHE=0`) — `0d29890e0f04` for both widths in all three (bf16 the same,
   `945f89766e3c`).  The positive control (`GGML_CUDA_QSA_INDEXER_CACHE=2`, the unfilled-pool probe)
   *does* move the W=1 text, proving the fused path is the one running.  The "not byte-identical"
   claim does not hold on gfx1151.
2. **The residual split was the block-13 mmvq fusion.**  Forced sparse, f16, `P=5000`: the current
   delivery is `plain == n3 = cb2912b186b9`, and the pre-fix behaviour reproduces exactly with the
   opt-in `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` (`plain 471ea250f8e2` vs `n3
   cb2912b186b9`) — the "706-char prefix then divergence" was the dense gate+up+GLU / weighted-down
   single-token fusion non-byte-identity (§25), not the indexer machinery.

**Default gfx1151 configs are pure** (plain == `draft-mtp n_max 3` byte-identical): shallow dense
decode on every tested KV type (q8_0 included, `e8f8bba3942b`), and deep sparse decode at ~74K (f16
`83e0ed0f0f80`, q8_0 `7205399d367d` — the maintainer's `-ctk q8_0` serving config).  The 64K crossover
therefore stays: there is no purity driver to make gfx1151 dense-decode at every depth.

**One residual, q8_0-only and prompt-dependent (open).**  With the arm *forced* sparse at shallow
context, qwen4exp + `-ctk q8_0` diverges on one prompt (`/tmp/p5000.txt`: `plain a57bc13bbf2a` vs
`n3 3124adfd2b94`), reproducibly; f16/bf16/q4_0/q4_1/iq4_nl and q8_0 on other prompts are pure, and
the default deep q8_0 config is pure.  `LLAMA_QSA_SPARSE_FA=0` does **not** reconcile it (so the
standard masked-FA path is affected too, not just the fused `fattn-qsa` kernel), `LLAMA_QSA_OFF=1`
does, and `GGML_CUDA_DISABLE_FUSION=1` / `GGML_CUDA_GDN_CHUNKED=0` each perturb it to purity.  It is
tracked as TODO item 4.

**Extended the same day (2026-09-12 (6)) — it is *not* a width dependence.**  A multi-step,
teacher-forced replay (new instrument `archive/work/strix-halo/qsa-item4/mstep.cpp`) of the plain greedy sequence
in the exact residual config is **bit-pure at every verify width** — 200 positions, `W = 1..8`, with a
spec-like batch+rollback schedule (`RB`), with deliberately unrelated tokens in the rolled-back rows
(`JUNK`), and with `n_rs_seq` 0 vs 2/3 — the recurrent snapshot rollback restore is exact and the
rolled-back content does not leak.  The divergence has a sharp **binary toggle at `--spec-draft-n-max`
2** (`n_max 1` pure at 31.1 t/s, i.e. MTP genuinely active; `n_max 2/3/5/7` all identical, first diff at
char 458).  Ruled out: `n_rs_seq`, `n_outputs_max` (`1 + n_max`), CUDA-graph capture
(`GGML_CUDA_GRAPH_OPT=0`), and the chunked-GDN prefill boundary — the boundary is a real hazard (moving
it by one token changes the greedy text) and it is why `GGML_CUDA_GDN_CHUNKED=0` moves the *plain* stream
at char 49, but an instrumented `gated_delta_net.cu` shows the actual chunked-GDN call sequence is
**identical** between the two runs (144 calls, same sizes).  So those two "reconciles" are perturbations,
not localisers.  The one measurable *structural* plain-vs-MTP difference on this model is that the MTP
driver turns on the target's `embeddings_nextn` (`common/speculative.cpp:1431`), which makes qwen4exp's
last-layer output gather defer (`gather_now` in `src/models/qwen4exp.cpp`) and shifts the **prefill's
last-position logits by a ULP** (`ad3acaa7…` vs `b624a79f…`) — a real logits-level violation of the
`plain == draft-mtp` guarantee, but on its own it does not flip the replayed greedy tokens, so it is the
leading partial cause, not the whole story.  The residual is therefore a **driver-level (plain-vs-MTP)
divergence**; the next step is a faithful mini-MTP driver (target + draft + real proposals + driver
rollback, per-step target-logit dump), because everything cheaper is exhausted.  Repro + instruments:
`archive/work/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md`.

**Closed 2026-09-12 (12).**  The faithful driver was built on the **real** server loop (a temporary
target-logits dump in `tools/server/server-context.cpp`, the in-process engine `llama-cli` actually
runs — not the standalone speculative example, whose raw-prompt input EOGs immediately).  It pins
the first divergence exactly: at target position **4432** the accepted token is identical (381) but
the target logits' argmax flips **264 -> 9859**, so this is a QSA-indexer **selection/state**
divergence, not a forward width dependence.  Excluded on the current tip by direct measurement:
forward width (the `mstep` W=1..8 replay is bit-pure), the GDN rollback bound and the checkpoint
restore (forcing `n_rs_seq = 16` so every rollback takes the plain `seq_rm` path still diverges;
`test-recurrent-state-rollback` PASS), `n_outputs_max`, CUDA-graph capture, the chunked-prefill
boundary, the fused indexer score / derived cache (both bypassed with a quantized key cache), and
the sparse FA kernel (`LLAMA_QSA_SPARSE_FA=0` also diverges — the shared dense masked path is hit).
The only reconcilers are `LLAMA_QSA_OFF=1` (the user affordance) and `GGML_CUDA_GDN_CHUNKED=0`
(a trajectory perturbation).  Sub-item **(a)** is fixed in the block-14 seventh amendment — the
`embeddings_nextn` export no longer defers the logits gather (see §28) — and the prefill ULP no
longer exists.  Sub-item **(b)** — forced-arm (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`), shallow — was then
**root-caused and fixed** by the gfx1201 investigation (2026-09-12 (13), block-14 eighth amendment,
§29): the flattened indexer score (`ne11 = 4 * n_tps`) crossed `MMVF_MAX_BATCH_SIZE` at `n_tps = 3`,
putting the verify batch on MMF while decode stayed on MMVF.  **gfx1151 cross-check validated
2026-09-12 (14):** the forced-sparse text residual is gone (`a57bc13bbf2a` both, was n3
`3124adfd2b94`, first diff char 458), all eight native KV types are pure at n_max 1/2/3/5/7, and
`mstep` `W = 1,2,3,4,5,8` is 0 mismatches with decode's `Thash` unchanged.  Evidence/records:
`archive/work/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md`, `archive/work/strix-halo/qsa-item4/` (harness).

## 19. Purity first: the measured trade (2026-09-11, policy)

**A purity fix may cost a few percent of raw non-MTP throughput.  That is not a gate failure.**
The gate is bit-identical decode/verify plus MTP acceptance/throughput, and the 2026-09-11 work
measured why the order of those priorities is not a matter of taste:

* the MoE shared-expert band fix (§17) **costs** ~2.4 % at the widest verify batch (`pl=8` 332.8 vs
  341.0 with the unfused chain; `pl=4` −0.9 %, `pl=1` flat) — and it **raised MoE `draft-mtp`
  acceptance from 0.51 to 0.81707** (167.3 t/s vs plain 96.9, **+73 %**), because the verify batch now
  computes what the draft's single-token decode steps compute;
* the QSA decode-arm fix (§16) **costs** ~1.5-2 % at `pl=5/6` — and it makes `plain == draft-mtp`
  byte-identical with `n_max 3` pos-1 acceptance 0.615 at 63.9 t/s vs plain 50.1 (**+28 %**).

The asymmetry is structural: the draft model's proposals *are* decode steps, so any arithmetic
difference between decode and verify shows up as draft-vs-verify disagreement, and lost acceptance
costs a multiple of whatever the "faster" path saved in the kernel it changed.  A few percent of
attention/GEMM throughput cannot pay for a lower acceptance rate.

**Gate for a purity fix** (replace any previous "must not regress" wording):

1. **correctness**: `plain` vs `draft-mtp` byte-identical across the whole band (`n_max <= 7`), on
   every supported KV type, or a documented reason why the affected regime is unreachable by default;
2. **MTP**: acceptance at **pos 1** >= ~0.45 and `draft-mtp` throughput >= `plain` at the default
   depth (`n_max 3`); a fixed high depth (`n_max 7+`) may lose — that is over-drafting, not a
   regression (see `benchmarks/mtp-adaptive-methodology.md`);
3. **perf**: no *hard* requirement at the wide verify widths.  If a fix costs > ~1 % anywhere, record
   the delta and file the optimisation follow-up (e.g. TODO's column-blocked shared-expert kernel),
   but land the purity fix first — and if a choice exists, prefer the side whose arithmetic is the
   *decode* side, since that is what the draft reproduces.

Corollary for tuning: an apparent "win" measured before a purity fix (e.g. a dense/sparse crossover
depth, or a fusion's "+x % decode") may have been measured with a *width-dependent* path, which is
exactly what happened to the gfx1151 QSA crossover below.

### A shape-scoped launch parameter must be forwarded to every kernel that reproduces that launch (2026-10-20, r93)

`nwarps` / `rows_per_block` / VDR are chosen per weight shape, which is width-uniform **only** if every
kernel that computes the same tensor uses the same value.  The fused single-token kernels
(`ggml_cuda_op_ssm_gate_beta`, `ggml_cuda_op_ssm_conv_l2_gatebeta`, `ggml_cuda_op_shexp_down_gate`, the
rms_norm→Q8_1 fold, the gate/bias folds) each *replace* a dense mmvq weight launch, so they must call
the same selector (`calc_nwarps_weight(type, ncols, table, long_k, <row scope>)`).  A scope that is
applied to the plain launch but not to the fused one runs `W = 1` at one warp count and `W = 2..8` at
another — a decode-vs-verify impurity with exactly the symptom this rulebook exists to catch
(measured: 35B-A3B 0.150, gemma-26B **3.35** on the `W = 1..8` probe).  The ksplit K loop
(`kbx = tid/(qi/vdr) + m*blocks_per_iter`) is a fixed per-thread residue class, so a *correctly
forwarded* per-shape value needs no further derivation.  The scope was the `!has_fusion` exclusion in
the `archive/work/nwarps/` candidate, not a missing reduction-mapping derivation; the host must still compute
the value once and pass it to both the launch dims and the kernel template (§20).  Record:
`archive/work/nwarps/`.

## 20. A newly enabled KV-cache type is a new kernel family (2026-09-11, F3 step 1)

The band guarantee is a property of *every* op on the decode/verify path, and a KV-cache type is one
of the inputs that selects an op.  `q4_1`/`q5_0`/`q5_1` were enabled as FlashAttention cache types on
2026-09-11 (block 08; `GGML_CUDA_FA_ALL_QUANTS` was the only way to reach them before, and that flag
also permits *mixed* K/V pairs).  Each newly reachable (K,V) diagonal is therefore treated exactly
like a new kernel family: it gets its own `W = 1..8` sweep on every split (plus `RS=0`/`RS=from_w`),
its own `plain == draft-mtp` text gate and its own MTP acceptance reading before it is offered.

Two consequences worth recording:

* **On gfx1201 the enabled types are band-uniform by construction** — with a quantized cache the whole
  band takes `BEST_FATTN_KERNEL_TILE` with the launcher's f16 staging (`need_f16_K/V = 1`), and the
  guard that made that true is block 08's F1 fix (§14).  A quantized cache never reaches the vec
  family on this arch, so the new diagonal instances exist for the *other* backends (where the chooser
  still picks VEC for small `n_q`) and for consistency with the predicate.
* **A type the model's *other* attention ops cannot read is a correctness trap, not just a
  performance one.**  qwen4exp + a quantized cache + `-sm tensor` aborted in the meta splitter
  (`GGML_BACKEND_SPLIT_AXIS_UNKNOWN` on the `attn_gated` MUL): the graph kept building the fused sparse
  QSA op for a type its kernel cannot read (f16/bf16/q8_0 only), so that op was never split across the
  devices while the attention gate still was.  Block 14 now takes the dense masked path whenever the
  cache type is not QSA-native.  The lesson generalises: when a cache type is enabled for one
  attention op, check every op that consumes the cache in that graph (and every split mode).

## 21. A shared staging tile makes a block head-homogeneous (2026-09-11, block-14 amendment; and the two instruments that found it)

The QSA kernel (`ggml/src/ggml-cuda/fattn-qsa.cu`) gives one block up to `QSA_MAX_HEADS = 16` query
heads (one warp each) and stages **one** K/V tile into shared memory for all of them: the cooperative
gather is issued by every thread of the block, and each thread adds *its own head's* K/V offset
(`nb12*(head/gqa_ratio)`) before it writes the shared rows.  That is only valid while every q-head in
the block maps to the **same K/V head** - which the old `head_base += QSA_MAX_HEADS` chunking silently
assumed.  qwen4exp is 24 q-heads / 2 kv-heads = **gqa 12 < 16**, so a 16-warp block covered heads
0..15 and heads 12..15 wrote the *second* K/V head's rows into the same smem slots: 16 of 24 heads
attended over a tile that mixed both heads' V rows.  The chunking is now
`min(QSA_MAX_HEADS, gqa_ratio)` heads per block (a no-op at `gqa_ratio >= 16`).

**This is the general rule, not a QSA detail:** whenever several heads/sequences/rows share one
staged buffer, the block's work partition must be homogeneous in every index that the *staging* reads
but the *compute* does not re-apply.  QSA applies the head only to Q (and to K/V, before staging), so
the head is exactly such an index.

### Why nothing caught it (and the two instruments that do)

1. **Width purity cannot see it.**  The bug is perfectly *width-uniform*: every `W = 1..8` takes the
   same arm, the same tile composition and the same wrong rows, so the primary instrument of every
   earlier QSA finding (the `W=1..8` logits-purity matrix) reports a clean, self-consistent band.  An
   *internal* consistency gate is necessary but never sufficient: it can only prove that widths agree
   with **each other**, not that they agree with the definition of the op.
2. **The probe was blind to the op.**  The QSA op is only built above the indexer selection width
   (`indexer_top_k + r - 1` = 2051 for qwen4exp) or when the dense-shortcut/decode gates are off; the
   probe's context is `n_ctx = 2048` (max `P = 2040`), so the *default* probe configuration never
   executed `GGML_OP_FLASH_ATTN_QSA` at all.  The matrix becomes meaningful only with
   `LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0` (force the selection path at every
   width) - which is now how the QSA kernel's purity gate is run.
3. **MTP acceptance points the wrong way here.**  The draft context and the main context run the same
   wrong attention, so the corrupted pair is *self-consistent* and accepts **more** than the correct
   pair (0.65 vs 0.49): acceptance is a quality signal only when both sides are known-good, never when
   the same defect sits in both.
4. **The instruments that do catch it** (both added in the same amendment):
   * a **CPU reference for the op** - `ggml_compute_forward_flash_attn_qsa` already implemented
     f16/bf16/q8_0 and now the four nibble types too, and a new `test_flash_attn_qsa` case in
     `tests/test-backend-ops.cpp` compares the GPU kernel against it over the KV types, both cache-
     type widths (gqa 1 and gqa 8), the three head sizes, `n_tps = 1` and `4` and the sliced+combined
     walk (**18 cases**; the old kernel scores NMSE ~1.0, the fixed one < 5e-4);
   * the **dense masked path as an oracle** - `LLAMA_QSA_SPARSE_FA=0` computes the *same* attention
     (the same top-k cells, unmasked, through the well-tested FA kernels), so the sparse/dense
     perplexity over one token stream is a directly comparable quality metric.  On 3x R9700
     (`-sm layer`, 8 x 4096 tokens) the old kernel is **7.3269 +/- 0.151** against the dense
     **6.5306 +/- 0.132** (+12 %), and the fixed kernel **6.5267 +/- 0.132** - i.e. the fix recovers
     exactly the oracle.  The same table for `q4_1` (6.5787 vs 6.5805) and `q5_0` (6.5444 vs 6.5375)
     validates the newly enabled quantized paths.

The corollary for this project's gate list: for any *fused* op, a purity sweep is not a correctness
gate.  Pair it with an independent oracle (a CPU/reference implementation, or a well-tested
alternative path computing the same math) and keep the oracle in the test suite, not just in a
one-off measurement.

## 22. `iq4_nl` (F3 step 2, 2026-09-11 (10)): the type is not the only thing an enablement touches

`iq4_nl` is now a first-class FA cache type (block 08's fifth amendment + block 14's fifth), so §20
applies in full: the diagonal gets its `W = 1..8` sweep on every split and model, the
`plain == n_max 3 == n_max 7` text gate, the MTP acceptance reading, and - the part §21 added - the
CPU/backend-op oracle and the perplexity-vs-dense comparison.  Results: pure everywhere, text
`acd18ad2d55c` (tensor) / `a38a6e2d8efa` (layer) with the f16/`q4_1` controls unmoved, MTP pos-1
0.757, `FLASH_ATTN_QSA` 22/22 (`iq4_nl` at D=128/gqa=8 **and** at the model geometry D=256/gqa=12),
perplexity sparse 6.5244 vs dense 6.4930.

Three things this enablement taught that generalise:

* **A type rejected by the FA probe hides *unreachable code paths*, not just slow ones.**  The
  non-contiguous staging converter (`ggml_get_to_fp16_nc_cuda`) had never listed `iq4_nl`, and
  `launch_fattn` calls its result unconditionally when K/V is a *view* - i.e. an enablement can turn a
  dormant `nullptr` into a null-pointer call on the first token.  It was the **backend-op suite** that
  found it (`-o FLASH_ATTN_EXT` SIGSEGV in `launch_fattn`), because that suite already contained 336
  `iq4_nl` cases that had only ever been *skipped*.  When a predicate starts accepting a type, assume
  the type's coverage in *every* consumer of that predicate is stale and run the whole suite.
* **"The type is 3.4x slower" is a statement about the pre-state, not the type.**  The `iq4_nl`
  pre-state (2269.8 pp512 / 48.5 tg32 on the 4B) was the *no-FA-at-all* path; with FA it is
  7931.8/95.0, i.e. within 2 % of f16 on dense models.  Record the pre-state *with the regime*
  attached, or the old number becomes a false expectation (this amendment's brief predicted exactly
  this correction, and the qwen4exp numbers needed the same care).
* **A type-specific cost can live outside the type's kernels - prove it with a profile before
  optimizing.**  On qwen4exp `iq4_nl` costs ~8-12 % of prefill versus f16/`q4_0`/`q4_1` at pp8192+
  (`q4_0` has the identical 18-byte layout and is flat), which looks like a dequant problem.  It is
  not: `rocprofv3` puts the QSA kernel's `iq4_nl` instantiation within 1.3 % of `q4_0`'s (same
  VGPR/LDS/occupancy), the dequant kernels at an identical 1.2 ms, the traced kernel *sum* lower, and
  the *executed graph* identical (1010 nodes, 0 diff) - so the delta is host/launch-side.  The
  instrument that settles it is `rocprofv3 --kernel-trace` + a `[GD]`-style full-graph dump (both
  cheap); the trap is to "optimize" the dequant on a hunch and call the result a fix.  Follow-up filed
  in `TODO.md`.
* **A change in one op can be visible only through another type.**  The Block 15 beta re-cut turned
  out to break its own oracle arm (`LLAMA_QSA_SPARSE_FA=0`, PPL ~1.05 for *every* type) - a defect no
  same-seed/MTP gate can see, and one that only surfaced because this session's gate list runs the
  perplexity oracle as a matter of course.  Keep the oracle in the list, even when its answer is
  expected to be "unchanged".

## 23. A silently dead chain: the shadowed variable (2026-09-11 (11), Block 15 dense-arm blocker)

Block 15's V2/V3 refactor added an *outer* `ggml_tensor * kq_mask_top_k = nullptr;` in `build_attn_qsa`
while the mask chain inside the new `if (kq_mask != nullptr) { … }` wrapper kept its own
`ggml_tensor * kq_mask_top_k = ggml_set_rows(…)` — a **new local** that shadowed the outer one.  The
chain was built, but the attention read the outer `nullptr`, so the dense masked arm attended with **no
mask at all**: a full causal leak (random text PPL `1.0205` broken vs `19.0589` healthy; qwen4exp PPL
`1.0558` vs the delivery's `6.49-6.55`).

Four lessons that generalise:

1. **A graph tensor with no consumer is silently dropped** — the chain was unreachable from the output,
   so `ggml_build_forward_expand` never emitted it, the allocator left the packed mask unallocated, and
   the input's own `if (self_kq_mask->buffer)` fill guard then skipped filling it.  "The input is created
   in the graph" proves nothing about it being *filled*, and an unfilled mask is indistinguishable from a
   correct one until you measure quality (a defensive assert is worth considering for the beta).
2. **When an executed-graph dump is *missing* nodes, suspect the graph builder, not the allocator** —
   the delivery consumed `attn_inp_kq_mask` 36 times and emitted the `FILL`/`SET_ROWS` chain; the beta
   consumed it **zero** times and emitted **no** `FILL`.  Dead code is absent from the graph by
   construction; that is the signature.
3. **`-Wshadow` would have caught this class outright** — the fork does not enable it; enabling it (at
   least for `src/`) would make the failure mode a compile error.  The fix itself is one token.  See the
   2026-09-12 `-Wshadow` audit (`archive/work/shadow-warnings/`).
4. **The leak instrument to reach for first is random text.**  A model that can see the target predicts
   even noise: `llama-perplexity -f /tmp/rand-text.txt --chunks 1 -c 2560 -b 2560 -ub 2560` gives the
   broken build `1.0205` and a healthy one `19.0589`.  Natural or repetitive text is a *bad* leak
   detector (a repetitive prompt scores ≈1 in a healthy build too).  Keep both the dense-masked oracle
   and the random-text probe: they answer different questions ("is the fused path as good as the dense
   one?" vs "is the attention causal?").

Narrative: `../archive/docs/GREEDY-PURITY-FINDINGS.md` §23.

## 24. The cost of a band amendment can be structural, not arithmetic (2026-09-12)

The 2026-09-11 fused shared-expert epilogue band amendment (§17) bought MoE decode/verify purity with a
*measured* cost: `llama-batched-bench` `pl=8` 461.0 t/s fused vs 472.7 unfused (−2.4 %), because the band
was served by `grid = (nrows, ncols)` — one block per `(output row, token)` — so the down-weight row was
re-read per token and the block's two barriers, cross-warp reduction and epilogue were duplicated per
token.  That shape is the *absent-minded* half of the amendment, not a requirement of it: the token only
selects input/output columns, so the band can live **inside** the block.  Three lessons:

1. **Band-uniformity constrains the arithmetic, not the launch shape.**  What the invariant needs is that
   each token's per-thread accumulation order and cross-warp reduction order are the ones the
   single-token kernel used — which survives any restructuring that keeps that order per token (here: a
   `ncols_dst`-templated kernel with the token loop inside the k-block loop, per-token accumulators, the
   weight block read once per `(row, k-block)`, `nwarps` still pinned because it sets `blocks_per_iter`).
   When a band fix costs a few percent, first ask whether the cost is the *invariant* or the *shape* it
   was implemented with.  Here it was the shape, and repaying it turned `pl=8` from −2.4 % into +3.1 %
   with **zero** numerical change.
2. **The strongest control for a "this must change nothing" claim is two binaries, not one hash.**  A
   documented gate hash reproduced by the new build only shows the new build is *as documented*; keeping
   both `libggml-hip.so` builds of the same tip and swapping them in (`tools/sobench.sh`'s idiom) shows
   the two builds agree *with each other* at every gate — probe `W = 1..8` fused and under the kill-switch,
   the §5 model matrix, the §19 `plain == n_max 3 == n_max 7` text gate, MTP acceptance, the backend
   suites.  That is what makes "no-op at the gate config" a fact rather than a hope.
3. **A no-op at the gate config can still be a large win in wall clock** — the two are independent
   measurements and both are required.  Contrariwise, an *arithmetic* change that happens to keep a hash
   stable at one width is not purity; the band is the unit of purity, so the A/B must span `W = 1..8`.

Related: a band/dispatch exercise must also *check the premise of its own control*.  Item 6's probe assumed
a Q4_K MoE model did not take the RDNA4 routed-compact MMQ dispatch and could serve as the "plain"
reference; `rocprofv3 --kernel-trace` showed it takes it (480 `mul_mat_q_routed_compact<(ggml_type)12,
32>` launches per `pp512`/`ub512`), and that `GGML_CUDA_DISABLE_MMQ_ROUTED=1` disables only the compact
*enumeration* — the per-expert J selection stays in both arms.  The usable control was the dispatch's
*reach* (0 compact launches at decode, i.e. prefill-only), not the model choice.  A control you have not
measured is not a control.

## 25. The RDNA3_5 single-token-only mmvq fusions are not decode/verify bit-identical (2026-09-12, block-13 amendment)

The block-13 band work (§15, §17, the dense `ncols==1` ksplit alignment, the `MUL_MAT_ID` dispatch fix)
made the **standalone** mmvq path band-uniform for `W = 1..8`.  On RDNA3_5 (gfx1151) two
**single-token-only** fusions still route `W=1` through fused kernels that do not reproduce the
standalone arithmetic, so a 1-token decode and an n-token speculative verify of the same layer compute
different bits — this is the issue-25 record's "block-13 `n_q=1` short-K mmvq variance", localised:

* the **dense gate+up+GLU mmvq fusion** — `mul_mat_vec_q<..., ncols=1, has_fusion=true>`; `mmvq.cu`
  restricts fusion to `ncols_dst == 1` (`GGML_ASSERT(!has_fusion && "fusion only supported for
  ncols_dst=1")`), so it fires at `W=1` only.  The matchers and `ggml_cuda_should_fuse_mul_mat_vec_q`
  are **upstream at the fork point**; block 13's mmvq item-split rewrite changed the `n_q=1` path they
  route through.
* the **MoE weighted-down tail** `ggml_cuda_mul_mat_id_weighted_rdna3_5` — RDNA3_5-only, single-token by
  its shape fingerprint (`ggml_nelements(y) == 640 * n_used`).  Kernel/`_ok` added by block 13, matcher
  by block 14.

Measured (qwen4exp UD-IQ4_XS, `P=100`, f16, gfx1151, probe `logits-dump-kv.cpp`):

| config | `W=1` | `W=8` |
|---|---|---|
| default | `8abc6206d1e80709` | `453eaa618738273d` |
| `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1` | `8d036a7b8c8a5ce5` | `453eaa618738273d` |
| dense-GLU fusion disabled | `969940599c5426e9` | `453eaa618738273d` |
| **both disabled** | **`453eaa618738273d`** | **`453eaa618738273d`** |
| `GGML_CUDA_DISABLE_FUSION=1` | `5a7e4c21e86e34c2` | `5a7e4c21e86e34c2` |

Each fusion moves `W=1` independently; only both together make it equal to the `W=8` standalone
reference.  `GGML_CUDA_DISABLE_MMVQ_MAT` (all dense `MUL_MAT` mmvq fusions) + weighted-down off also
reconciles, so the whole gap is inside the mmvq fusion family.  A fusion trace confirms the `W=1`-only
arms as exactly `n=21 MUL_MAT_ID(ffn_moe_down)` and `n=3 MUL_MAT(ffn_gate)`.  The other `W=1`-only arms
(dual-output K/V, SSM conv-input, GDN `ssm_gate_beta`, L2-norm pair) are bit-identical to their unfused
reference and stay enabled, as does the `MUL_MAT_ID` gate+up+GLU (it fires at every width and is
band-uniform).

**Fix** (block 13, folded 2026-09-12): skip both on RDNA3_5 unless
`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` (A/B).  The dense GLU is guarded at the six
`{op,op,GLU}`/`{op,bias,op,bias,GLU}` matchers in `ggml_cuda_try_fuse` (with `ids != nullptr` keeping
the `MUL_MAT_ID`/MoE fusions on), and the weighted-down in `ggml_cuda_mul_mat_id_weighted_rdna3_5_ok`.

**Validation** (post-fix, `W = 1,2,4,8` one hash per config): qwen4exp f16 `453eaa61`, q8_0 `113696b9`,
MoE 35B-A3B `18999a78`; the 27B dense (`e165ef98`) was already pure and is unchanged; the gfx1201 path
is untouched (`GGML_CUDA_CC_IS_RDNA3_5`-only).  **Cost** ≈ −0.9 % `tg128` on qwen4exp (25.53 vs 25.77
t/s), prefill flat — the §19 trade.  The optimisation follow-up that would restore it is to make the
fused `ncols_dst==1` kernel reproduce the standalone reduction (pin `nwarps`/`rps`/item-split) instead
of skipping the fusion.

**Scope note.**  This is *within* the 8-wide band, unlike §11's `n_max <= 7` bound: it broke draft-vs-
verify purity for qwen4exp and the MoE even at `n_max <= 7` (the dense GLU also affects pure-attention
models, which is the Gemma4-12B `W1-W3 = 1.016` line in the issue-25 record).  The band edge above 8 is
unchanged: FA's tile→WMMA switch (`Q->ne[1] > 8`) and the `MMVQ_MAX_BATCH_SIZE` mmvq→MMQ crossover.

## 26. A regime policy is only pure if the *band* takes one arm — including the prefill half (2026-09-12 (9))

§§11/16-18 are about the decode/verify band; the same structural rule governs the *prefill* side of the
QSA arch policy, which block 14's 2026-09-12 (sixth) amendment made depth-configurable
(`qsa_dense_prefill_until`; `patches/README.md`, record
`archive/work/strix-halo/qsa-item9/RECORD-2026-09-12-qsa-prefill-crossover.md`).

The invariant, in the form a review can check: **an arm may only be selected from state that is
identical for every graph that computes the same sequence position — the `n_tokens` *band*
(`<= QSA_DECODE_BAND` vs above it) and `n_kv`.**  The prefill arm is therefore defined as
`n_tokens > QSA_DECODE_BAND && n_kv < threshold`, i.e. disjoint from the decode arm by construction,
so a W=1 decode and a W=(`--spec-draft-n-max` + 1) verify step still take the same arm; a prefill
chunk of `<= 8` tokens is indistinguishable from a verify batch, so it takes the decode arm (that is
the design).  An arm gated on `n_tokens == 1` — the shape of the original cause-1/cause-3 defects —
is what breaks it, and the same trap applies to any new depth-keyed policy: key it on the band, not
on the exact width.

**The prefill arm ships disabled, and that is its purity guarantee.**  `qsa_dense_prefill_until`
defaults to `0` = QSA prefill always, which is the documented 2026-09-07 arch policy on both arches
("prefill is always QSA"; Soar wins from ~8K monotonically to +181 % @160K) — so the delivered default
selects one arm for prefill at every depth and no reference hash moves.  It is an opt-in A/B knob
(`LLAMA_QSA_DENSE_PREFILL_UNTIL`), and switching it on is a deliberate regime change whose default
would have to be justified with an **at-depth** measurement: the 2026-09-07 record explicitly rejects
the whole-prompt banner shape, which is exactly what a first attempt at a default used (the record
carries those numbers, flagged non-comparable, as a description of the knob rather than evidence).

Verified on the amended tip (gfx1151): the default is byte-identical to the pre-amendment build —
f16 `plain == n_max 3` = `0fc4910d5824` (632 chars) and q8_0 `plain == n_max 7` = `e8f8bba3942b`
(626 chars, the recorded pre-amendment shallow q8_0 value) — and the `mstep` width probe is 0
mismatches at W=4 (the W=8 38-mismatch item-4 residual class is unaffected and its position list is
identical with and without the knob, which is how it was separated from this change).

Two corollaries worth keeping for whenever such an arm *is* enabled:

* **A regime knob is a quality knob, and the default is the arbiter.**  Below a crossover the sparse
  arm is both an approximation and (in the whole-prompt measurement) slower, so a dense arm there is
  not obviously wrong — but "the arm is faster in my measurement shape" is not the same as "the
  project's default should change", which is why the shipped value is the policy.
* **A self-limiting arm cannot lose at depth.**  The arm only ever covers chunks whose `n_kv` is still
  below the threshold, so a long prefill keeps the sparse chunks that measured faster (sparse wins
  pp32768 by 17.4 % when it is *all* sparse) while the shallow ones go dense.  That property is what
  would make a future at-depth-justified default safe rather than a crossover gamble.

## 27. A rollback bound is a purity bound: the batch you roll back into must be written by the kernel whose snapshots you read (2026-09-12 (10))

The recurrent (GDN) rollback keeps `K = n_rs_seq + 1` snapshot planes, and only the **sequential** GDN
kernel writes them (slot `s` = the state `s` tokens back).  The **whole-batch chunked** prefill kernel
writes slot 0 only, so it is safe for a batch only if that batch is never rolled back into.  Block 02
originally expressed that as a constant: "a batch longer than `max(K, 16)` cannot be a speculative
verify batch", which held only while every speculator's maximum draft was bounded by `n_rs_seq`
(sized from `speculative.draft.n_max`).  It was **false** for ngram-style long drafts: with ngram-mod
able to draft 64 tokens, a 65-token verify batch took the chunked path and a small tail rollback then
restored a plane that batch never wrote - a silent, finite-but-wrong recurrent state.

The invariant to check in any future change here: **every batch that can be rolled back into must run
the kernel that writes the snapshots it will read.**  The bound therefore has to come from the
*speculator's* maximum draft, not from the snapshot depth - `n_rs_batch =
common_speculative_n_max() + 1`, consumed as `GDN_CHUNKED_MIN_TOKENS = max(K > 16 ? K : 16,
n_rs_batch)`.  Consequences worth keeping:

* **Default configs are unaffected**, so this is not a purity trade for the delivery: no speculator
  gives `n_rs_batch = 1` and MTP `n_max 7` gives 8, both below the 16 floor, so the chunked path keeps
  its whole-batch, K-independent shape and the recorded hashes do not move (verified: 27B
  `plain == draft-mtp n_max 7` = `e164f09af338`, qwen4exp `plain` = `0fc4910d5824`, unchanged; 27B
  pp2048/8192 within noise).  Only a long-draft speculator's verify batches move to the sequential
  kernel - and for those the alternative is corruption, so correctness decides (§19's trade).
* **A whole-batch rollback needs a pre-batch plane.**  The sequential kernel writes slots
  `0..min(n_tokens,K)-1`, so a rollback of exactly the whole batch needs slot `n_tokens`, which no
  kernel wrote: the graph now copies the pre-batch ssm and conv state there when `0 < n_tokens < K`
  (`delta-net-base.cpp`).  That is what `test-recurrent-state-rollback`'s `multi_seq_split_replay`
  exercises - it failed with `max diff 6.5366, first at seq 0 pos 16` before and matches with
  `max diff 0` after (both cache fills) on this box.
* **The guard stays a real invariant check.**  `llama_memory_recurrent::seq_rm` compares the last
  ubatch's per-seq token count against `n_rs_batch` (not `n_rs_seq + 1`), so it still warns - once -
  if a future change lets a rollback cross a batch boundary.

## 28. An export-only tail must not redefine the logits path (2026-09-12 (12), block-14 amendment (seventh))

**The invariant:** an extra graph output (here the MTP draft's `t_h_nextn`) may add work, but it must
not move the arithmetic of the *logits* path.  The trap is a shared tail whose batch width is chosen
for the export: an unmasked `embeddings_nextn` export needs a hidden row for **every** token, so
`src/models/qwen4exp.cpp` deferred the last layer's output-row gather (`gather_now`) until after
`t_h_nextn` was taken — and the last layer's hyper-connection combine + ffn tail then ran on the full
prefill ubatch instead of the gathered output rows.  The wide ffn's reduction order depends on the
batch width, so the prefill's last-position logits shifted by a ULP (`ad3acaa75d19ddf2` vs
`b624a79f19b1b1f0`) — a real logits-level violation of `plain == draft-mtp`, invisible to the text
gates (the ULP did not flip the sampled token) and only observable in a per-step logits dump.

**The fix (block 14, seventh amendment):** the last layer now **always** gathers its output rows
before the tail — exactly what `--spec-type none` does — and builds a **second, full-row tail** whose
only consumer is `t_h_nextn`, and only when the export is on and the chunk actually drops rows
(`n_outputs < n_tokens`, i.e. a prefill; a decode/verify batch drops none, so nothing is duplicated
there).  The separate tail is not on the logits path, so it must be expanded explicitly
(`ggml_build_forward_expand(gf, h_nextn)`); `ggml_set_output()` alone does not add a tensor to the
graph.

**Why the gates missed it until now:** the `plain == draft-mtp` text gates compare *sampled tokens*,
and a one-ULP logits shift only matters when the argmax is within a ULP of a tie.  The instrument that
sees it is a teacher-forced per-step logits replay with the export toggled (`mstep` `NEXTN=1` in
`archive/work/strix-halo/qsa-item4/`), which went from 1 mismatch at `pos = 4293` to 0.  Generalise: when a
graph gains an extra output that is read by a *different* consumer, check the batch width of every
node the logits traverse — and prefer "compute the logits path exactly as the plain arm does, then
compute the export separately" over "compute once and gather for the logits".  The cost is one extra
tail at output-row width (cheap; a prefill has ~1 output row), and the non-export paths are
byte-identical by construction.

## 29. A flattened batch is not a token count (2026-09-12 (13), block-14 amendment (eighth))

The decode/verify band guarantee is a property of **every op** on the path, and the engine's
"keep a verify batch on the decode (MMVF) family" guard
(`ggml_cuda_mul_mat`: `ne11_mmvf = ne11 <= MMVF_MAX_BATCH_SIZE ? 1 : ne11`) keys on the matmul's
`ne11`.  That is correct while `ne11` is the token count.  The QSA indexer score is the counterexample:
its matmul carries the **indexer heads** in the N dimension, so its `ne11` is
`n_idx_h * n_tps` (**`4 * n_tps`** for qwen4exp).  From `n_tps = 3` the guard stopped rescuing it:
decode (`n_tps = 1`, `ne11 = 4`) stayed on MMVF, the verify batch (`ne11 = 12`) fell through to MMF,
and the two families accumulate the truncated dot product differently.  The indexer score then differed
by a ULP and flipped a top-k near-tie.

**Signature to recognise it by:** the forward is *exactly* bit-identical to decode for many steps and
then diverges once, at a fixed position (here target position 4395, 102 tokens in) - a selection flip,
not drift.  The greedy **text** may stay equal for a given prompt, so this is a logits-level
`plain != draft-mtp` violation that text gates cannot see; `mstep` (`archive/work/strix-halo/qsa-item4/`) is the
instrument, and W=2 pure / W>=3 impure is the clean boundary.

**Instruments that localised it** (worth reusing): a `rocprofv3 --kernel-trace` diff of the two widths -
the only exclusive kernels were ncols-templated MMVF/ksplit variants, and the score moved from
`mul_mat_vec_f<float,float,8,64>` (N=8) to no MMVF instantiation at N=12; and a temporary "force the
fallback family for every F32 matmul" diagnostic, which made the band pure again and confirmed
"one family across the band" as the fix (the reverse - forcing MMVF at every width - was already the
intent of the guard).

**The fix (block 14, eighth amendment):** `MMVF_MAX_BATCH_SIZE_FLAT` (`MMVF_MAX_BATCH_SIZE * 4 = 32`)
and the guard widened to it, with `mul_mat_vec_f` instantiated for `ncols_dst` 9..32.  The band stays
on the *decode* family, which is the side the draft's single-token decode reproduces (the §19 rule).
`ne11 <= 8` and `ne11 > 32` are unchanged, so ordinary decode/verify and prefill are untouched.

**The general rule:** the band guard's "batch size" must be the **maximum flattened N** the op can
produce in the band, not the token count.  Any matmul whose N is a product of a per-token factor and a
per-head (or per-expert, per-group) factor is a candidate; audit it the same way a newly enabled KV
type is audited (§20) - one family, W = 1..8, on every split.

## 30. A support predicate decides the *layout*, and the layout decides the fusion (2026-09-13, block-08 amendment (sixth))

The 2026-09-13 `iq4_nl` `GET_ROWS` fix (TODO item 3) is a support-predicate bug, not an arithmetic one:
the CUDA `GET_ROWS` case required `ne[0] % QK_K == 0` for the 32-value sub-block types, so a 128-wide
`iq4_nl` indexer-key gather was rejected and the scheduler ran that single node on the CPU.  That is a
**performance** defect (13 host round trips per qwen4exp prefill graph, GPU busy/span 0.62 vs 0.96), but
it also changes the *allocation*: removing the host split re-addresses the whole compute buffer, and
`ggml_cuda_check_fusion_memory_ranges()` selects the MoE-router `topk_moe` fusion by buffer-address
overlap.  The fused router is **not** bit-identical to the generic chain (a temporary
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION` A/B moved the qwen4exp `iq4_nl` greedy text `14a1a3f257f4` ->
`086df944f6af`), so the model output moved with the layout (`c0d44c479ee1` -> `14a1a3f257f4`) even though
the moved op itself is bit-exact against the CPU at every width.

**Two rules to take from it:**

* **A support predicate is a correctness surface for *layout*, not only for reachability.**  When you
  widen a predicate the op's execution backend can change (CPU -> GPU), which relocates every tensor in
  the graph.  Re-run the full gate set, not just the op's own backend-op test - the observable can move
  in a *different* op.
* **An address-overlap-driven fusion whose fused arithmetic is not bit-identical to the generic chain
  makes the output allocation-dependent.**  That is pre-existing upstream behaviour (filed as TODO item
  19); the delivery's own invariants (width `W = 1..8`, `plain == draft-mtp`, the control hashes) are
  unaffected because the coverage is width-uniform within a config, but the absolute text of a config
  is not a property of the numbers alone.

**Instrument that localised it:** `GGML_SCHED_DEBUG=1` prints the per-graph split count (`iq4_nl` 142 vs
`q4_0` 22) and `=2` prints the per-node backend (`node #611 (GET_ROWS) ... CPU#cache_idx_k_l3`) - cheaper
and more direct than a profiler for "which op left the GPU".  Both need **`-lv 5`** (the dump is
`GGML_LOG_DEBUG`; `-lv 4` prints nothing, which has cost a debugging session).

## 31. An address-selected fusion must be bit-identical to the path it replaces (2026-09-13, block-08 amendment (seventh))

Item 30's follow-up (TODO item 19): the fused MoE router (`ggml_cuda_op_topk_moe`) was **not**
bit-identical to the generic `soft_max -> argsort -> get_rows -> norm` chain, and the fusion is selected
by `ggml_cuda_check_fusion_memory_ranges()`'s **buffer-address overlap** test - so the model output was a
function of the allocation plan.  The fused kernel is now made to reproduce the generic arithmetic
exactly:

* **Softmax reduction order.**  The generic `soft_max_f32` launches one thread per column and reduces
  with `block_reduce`: a per-warp butterfly over each consecutive 32-column group, then a cross-warp
  butterfly over the per-warp results.  The fused kernel held column `l + i*32` in lane `l`, so its
  single flat butterfly was a different association (36 % of random 512-value rows differ by up to
  2.4e-7 relative).  The fix does the generic per-"virtual warp" `warp_reduce_sum(vals[i])` phase first
  and the cross-warp phase second.  Max is exact under any order, so it needed no change.
* **Normalization.**  The generic chain is `sum_rows -> clamp -> div`; the fused kernel accumulated the
  winners in their own lanes and multiplied by `1/sum`.  It now sums in the `reduce_rows_f32` order
  (`warp_reduce_sum(lane j < n_expert_used ? output_weights[0] : 0)`, lane `j` = selection `j`'s weight)
  and **divides**.
* **Argsort tie-break.**  The generic CUDA argsort is a non-stable bitonic network, so for exact ties its
  top-k set/order is a function of the network, not the index - while the fused iterative argmax breaks
  ties by the smaller index, and the CUDA CUB `argsort` path (`SortPairsDescending`) is stable too.  The
  bitonic comparator now breaks ties by index, so all three agree.  Ties are real here: 4 occurred in
  one 3.3k-prefill + 64-token qwen4exp run.

**Rules to take from it:**

* **A fusion whose selection depends on memory addresses must be numerically transparent** - the fused
  kernel has to reproduce the chain it elides, or the same numbers give different text depending on the
  allocator.  "Within a config, the coverage is width-uniform" is not enough; an unrelated layout change
  can flip the coverage between runs.
* **A non-stable sort is a latent fusion hazard.**  The bitonic argsort and the CUB argsort are two
  implementations of the same op that already disagreed on ties; making the bitonic path stable (and
  consistent with CUB) is a correctness improvement, not a tuning choice.  The CPU `std::sort`
  comparator leaves ties unspecified, so no cross-backend tie contract is broken, and the backend-op
  `ARGSORT` case is tie-free by construction.
* **Verify with the kill-switch, not the hashes.**  The gate is `fused == GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1`
  (and == force-fuse) for every native KV type on both split modes; `plain == draft-mtp` alone would
  have passed with either path (both are internally width-uniform).  The force-fuse probe
  (`GGML_CUDA_TOPK_MOE_IGNORE_ALIAS`, temporary) additionally covers the call sites the address guard
  would otherwise skip.

## 32. A decode/verify band policy must scale with the configured draft depth (2026-09-13, issue #30, block-14 amendment)

**Claim.**  `QSA_DECODE_BAND = 8` matched the qwen4exp dense decode arm to the `n_max <= 7` verify, but
it was a fixed constant while the verify width is `n_max + 1`.  Once the draft-depth clamp moved from 7
to 15, a depth-8..15 verify became a 9..16-token batch: above the indexer selection width
(`indexer_top_k + r - 1` = 2051) the W=1 decode took the **dense** arm while the verify fell through to
the **sparse** top-k selection — the "triggers after ~2051 tokens" qwen4exp depth-15 divergence.  This is
the same *class* as §26 (a regime policy is only pure if the whole band takes one arm), re-opened because
the band was pinned to a compile-time constant instead of the configured draft depth.

**Evidence (real qwen4exp IQ4_XS, 3x R9700 `-sm tensor`, f16, `P=2500 > 2051`, token-0 logits).**

| build | `W = 1..8` | `W = 9..16` |
|---|---|---|
| pre-fix | `643a8166d8dad677` | `1354757f9daf03db` (== the `LLAMA_QSA_DENSE_DECODE_UNTIL=0` forced-sparse hash) |
| post-fix | `643a8166d8dad677` | `05be2f7f30dbc426` (the dense arm; the base kernel-family impurity of §11 remains) |

The dummy `qwen4exp-moe` isolates it cleanly (no matmul-family flip at these sizes): pre-fix W=1..8
`5009c55bca5e01ca` vs W=9..16 `596ec8bf7461da1a`; post-fix all W=1..16 `5009c55bca5e01ca`, and pure for
all eight native KV types.  With the QSA arm routed to sparse on **both** sides
(`LLAMA_QSA_DENSE_DECODE_UNTIL=0`) the width matrix is pure, so the fix does not weaken the sparse arm.

**Fix.**  `qsa_decode_band = max(QSA_DECODE_BAND, cparams.n_rs_batch)`; the decode arm uses
`n_tokens <= qsa_decode_band` and the prefill arm `n_tokens > qsa_decode_band`, keeping them disjoint.
`cparams.n_rs_batch` (`common_speculative_n_max() + 1`, the recurrent rollback bound) is exactly the
widest batch that can be a verify, so the arm now tracks the configured depth.  Default configs
(`n_max 3` -> `n_rs_batch 4`; `n_max 7` -> `8`) leave the band at 8 and are byte-identical to the
pre-amendment build, so every recorded reference hash holds.

**Rules to take from it:**

* **A band constant must be derived from the quantity it is matching.**  `QSA_DECODE_BAND = 8` was
  correct only while the clamp made 8 the maximum verify width; it silently rotted when the policy
  changed.  The verify width is `n_rs_batch`, so use it.
* **A bug that only fires past a regime threshold needs the threshold crossed in the test.**  The
  1..8-width probe was green because below 2051 the `shortcut` arm covers every width; the divergence
  appears only for a `W > 8` verify with `n_kv >` the selection width.  The regression is the real-model
  width matrix at `P > 2051` (`archive/work/wip-archive/recurrent-rewind-depth/`), not the short-context probe.
* **A depth-policy change has to re-run the width matrix for every model family the band touches.**
  The clamp relaxation was validated on the recurrent snapshot machinery (deterministic sweep) and on
  qwen4exp's QSA arm; the residual `W=9` difference on all models is the documented §11 kernel-family
  trade, not a defect.

## 33. A purity (or throughput) gate must run the content the axis intends, long enough to reach steady state (2026-09-13)

**Claim.**  The four-axis adaptive-MTP gate was first run with the model's default reasoning mode and at
`-n 256`.  Both invalidated it, in opposite directions:

* **Reasoning.**  Qwen3.8 emits a thinking trace for instruction-like prompts, so the "code" axis at the
  template default never reached any Python (at `-n 256`) and the "prose" axis answered as a request.
  The code acceptance read 0.56 instead of 0.89-0.92; the gate was measuring the thinking trace.
* **Length.**  `-n 256` measures the drafter/adaptive-controller warm-up, not the mode.  The code axis at
  adaptive ceiling 12 read **-5%** vs fixed `n3` at `-n 256` (mean accepted length 4.32) and **+28%** at
  `-n 3000` (mean length 7.02).  The ranking inverted.

**Why it matters for purity specifically.**  Purity above `n_max 7` is a probabilistic near-tie
property, so it needs enough tokens to hit a near-tie *and* the content the axis is meant to produce.
At `-n 256` all four axes matched plain; at `-n 3000` reasoning and recall stayed byte-identical while
prose and code diverged -- the documented above-7 kernel-family trade, made visible only by the longer
run.  A short purity check can therefore report "pure" while the real workload is not.

**Rules to take from it:**

* **Run the gate at a length that matches the workload** (`-n 3000` here; `-n 2000` floor).  A short run
  is a correctness smoke test (output non-empty, acceptance > 0), never a performance or purity verdict.
* **Pin the content mode** (`--reasoning off` for P/C/K, `on` for R) for every gate that compares
  builds or depths; a workload that is not actually produced cannot be measured.
* **A spot test that passes is not evidence that a long run passes** -- the same class of blind spot as
  §11's "text equality is evidence for purity, never against divergence".

Results: `benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`; protocol: gate rule 0 in
`benchmarks/mtp-adaptive-methodology.md`; prompts: `prompts/README.md`.

## 34. A K/V staging policy is a decode-depth policy (and can be the margin for a high-context load) (2026-09-14, issue #30, block-15 amendment)

**Claim.**  The block-15 `V4`/`V5` native-staging arms were documented and shipped as a *memory* feature
(they remove the F16 staging scratch), with decode "unaffected".  That framing was incomplete: for a
**quantized** K/V the F16 staging pass is a **whole-cache conversion that runs on every decode step and
scales with `n_kv`**, so it is a *decode-depth* cost, and the choice of staging source is a decode-depth
policy.  It can also be the difference between loading and not loading a high-context MTP config.

**Evidence (1 GPU, gfx1201, 27B UD-Q4_K_XL, `-fa auto`).**  With V4 opt-in off, delivery q8_0 `tg64`
d65536 was 18.92 t/s (66.1 % of its d0 rate) vs stock's 22.43 (80.1 %).  Turning native q8_0 staging on
(and adding the missing q4_0 arm) gives **23.29** (80.7 %) for ~1.2 % prefill; q4_0 19.72 -> **22.82**
(68.9 % -> 79.2 %, stock 21.03 / 75.9 %).  The values are **bit-identical** to the staging conversion
(same-seed text), so this is a pure speed/memory change, and `W = 1..8` stays one logits hash on every
supported type.  The same ~744 MiB/GPU scratch was the **260 MiB margin** the adaptive-MTP draft context
was short at `-c 196608` q8_0 / ceiling 12, so the default now loads at the default 4 slots
(`GGML_CUDA_FA_KV_NATIVE=0` reproduces the OOM).

**Rules.**

* **A native staging arm must reproduce the launcher's conversion bit-for-bit** (q8_0's
`dequantize_block_q8_0_f16`, q4_0's `dequantize_block_q4_0`) — otherwise the recorded reference hashes
move.  Gate it with the same-seed text gate first, the width matrix second, and the MTP acceptance third.
* **The activation policy is per type class.**  Native staging is the default for the sub-F16 quants
(q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl), where the F16 pass is the cost; bf16 already has a native tile/vec
path, so its MMA-scratch arm (V5) stays opt-in.  `GGML_CUDA_FA_KV_NATIVE` is a three-state policy
(unset = auto, 1 = force on, 0 = force the staging path), so the old behaviour is one env away.
* **A memory win can be a load-time win.**  When a config fails to load by a few hundred MiB, the fix
may be elsewhere in the same graph (here the staging scratch), not in the buffer named by the OOM.
Measure the *total* budget, not the failing allocation.
* The remaining headroom lever and the opt-in f32 -> bf16 snapshot trade (a purity trade to be *measured*,
not assumed) are filed in `archive/work/issue-30-mtp-decode-regression/RECURRENT-SNAPSHOT-BUDGET.md`.

## 35. A prefill kernel config must be arch- and split-keyed (2026-09-14, issue #30, block-04 amendment)

**Claim.**  The head-256 WMMA config row and the `ncols2` rule were single values applied to *every*
RDNA arch and *every* split mode.  Both were wrong as globals:

* The `ncols=64` row was a **gfx1151 (Strix Halo) "halo row"** (`nbatch_fa 32`, `nbatch_V2 64`,
  `Q_in_reg=false`), but the launcher uses `ncols=64` for *every* `n_q > 8` attention on gfx1201 too —
  ~1.5x per attention cell, i.e. the delivery's prefill slope was ~51 % steeper than stock and only
  showed at depth (delivery won at 4K, crossed at ~64K, lost at 150K).
* The `ncols2` rule is a **compute-vs-bandwidth** trade: a whole card is compute-bound and prefers
  `ncols2` that divides the GQA ratio (no wasted lanes); a tensor split is per-GPU bandwidth-bound and
  prefers the wider `ncols2` (fewer K/V re-reads).  One value cannot serve both.

**Rules to take from it.**

* **A config tuned on one arch is not a config for the family.**  Key it on the arch (`RDNA3_5`/`RDNA4`
  macros on the device, `GGML_CUDA_CC_IS_RDNA3_5(cc)` on the host — and they must agree).
* **A regime (whole-card vs tensor-parallel) is a first-class axis**, not a constant; the frontend knows
  the split mode, so pass it explicitly.  Here: `ggml_set_fa_tensor_parallel` (set in `llama_context`
  from `split_mode() == LLAMA_SPLIT_MODE_TENSOR && n_cuda_dev > 1`).  Note `llama_model::n_devices()` is
  **1** under tensor split (the meta device wraps the GPUs) — count the CUDA sub-devices instead.
* **An `op_params` socket is not a reliable channel**: the graph is copied before execution and the copy
  loses it (the builder's write was visible, the chooser read the default).  Pass through the
  model/context layer.
* **The instrument has a blind spot**: `-sm tensor` hides single-card regressions (its parallelism masks
  per-cell cost).  Screen a prefill change with the `t = a + b*n` slope fit at pp8-48K and **always
  measure one card as well**.

Results and the full matrix: `archive/work/issue-30-mtp-decode-regression/MEASUREMENTS.md` §D; `WORKLOG.md`
2026-09-14 (later); `patches/README.md` (2026-09-14 block-04 section).

## 36. §14's `W=1..8` purity is a *measured* claim, not a guarantee — relax it for the coarse quants (2026-09-14, issue #30)

**Finding.**  §14 deleted the VEC/TILE family split for a quantized K/V, and that part holds — on AMD the
chooser's fallback returns TILE unconditionally, so the whole `n_q <= 8` band is TILE, and (verified
2026-09-15 by dumping the launcher state at every width, see below) the **KV split is width-invariant
too**.  What remains is a **logits-level band edge** between `n_q = 1` and `n_q >= 2` that only becomes
visible as a near-tie flip, i.e. when the values land near a rounding boundary.  Measured on the 4B
(gfx1201; `P` = prefill length, `W` = decode batch width, one logits hash per `W=1..8`, prose prompt;
2026-09-15, after the item-2 arms):

| K/V | P=192 | P=200 | P=208 | P=224 | P=256 |
|---|---|---|---|---|---|
| f16 | pure | pure | pure | pure | pure |
| bf16 | pure | **edge** | pure | pure | pure |
| q8_0 | pure | pure | pure | pure | pure |
| q4_0 | pure | pure | pure | **edge** | **edge** |
| q4_1 | pure | **edge** | pure | pure | pure |
| q5_0 | pure | pure | pure | pure | pure |
| q5_1 | pure | pure | pure | pure | pure |
| iq4_nl | pure | pure | pure | pure | pure |

A **"edge"** is a genuine logits-level band edge, not a token-level one.  Measured at (P=200, `W=1` vs
`W=4`, 4B, gfx1201): bf16 -> `argmax 9146` in both, top-1 **23.6626 vs 23.6770** (delta 0.014), top-2
margin **2.24 vs 2.32**; q4_1 -> `argmax 9146` in both, delta 0.064, margin 2.73 vs 2.64.  So the greedy
token is unchanged and the top-2 margin dwarfs the delta — it is a *hash* difference, not a decoding
difference.  It is **pre-existing and independent of the native arms**: reproduced byte-identically with
`GGML_CUDA_FA_KV_NATIVE=0` (the pure F16 staging path) and unchanged by the prefill-staging split.  The
four small quants are measured **pure at all five P** here — arming them (item 2) moved their `W=1` path
from the F16 tile to the native tile, so their pre-2026-09-15 entries (a q4_1 edge at P=200) are
superseded, not contradicted.

**What is *not* the cause (verified 2026-09-15).**  A temporary launcher dump (`GGML_CUDA_FA_DEBUG2`,
snippet in `archive/work/issue-30-mtp-decode-regression/tools/fattn-launch-dump.patch`) shows the decode calls at
every width in the band have **identical** reduction structure — 4B, bf16, `P=200`, cache padded to 256:

| | `n_q=1` | `n_q=2` | `n_q=4` |
|---|---|---|---|
| `ntiles_x` / `ntiles_dst` | 1 / 4 | 2 / 8 | 4 / 16 |
| `parallel_blocks` (the KV split) | **8** | **8** | **8** |
| `blocks_num` | (1,8,4) | (2,8,4) | (4,8,4) |
| `ncols1`/`ncols2`/`nwarps`/`nbatch_fa` | 1/4/4/32 | same | same |
| `ntiles_KV`/`ntiles_z_gqa`/`stream_k` | 8/1/0 | same | same |

So the KV-split tree, the number of online-softmax rescale steps over KV *tiles* and the fixup condition
are all width-invariant — block 00's `ntiles_dst_eff` fix covers the whole band, and the tile path never
enters the stream-K branch (`stream_k=0`).  `ncols1 = 1` also means the tile has **no phantom query
columns**: a 1-row tile computes exactly one query per x-tile and pads nothing, so an earlier note here
attributing the edge to "the `n_q = 1` launch running its whole `cols_per_block`" was **wrong** and is
withdrawn.  The only quantity that changes with `W` is `ntiles_x` (the number of independent query
tiles), which is per-tile independent.

**Leading hypothesis (not proven).**  That leaves the FA path's *values*: the mask-derived KV support
range (`i_sup`) is a per-tile bound, so a 1-row tile can process one fewer KV tile than a `>= 2`-row tile
at the same cache length (the padded tail only the later rows can attend to), changing the last bits
through the online-softmax rescales.  This fits everything observed — inside the FA kernel, mechanism
type-independent, visibility a rounding-boundary lottery that depends on where the cache padding falls
(hence P=200 yes, P=208 no).  Note that "f16 is pure at P=200" is **not** evidence that the FA path is
the wrong place: the mechanism is probably type-independent, and a pure hash only means the perturbation
did not move a logit bit in that kernel's arithmetic.  Isolating it would need an op-level A/B (hash the
attention output rather than the logits) — deliberately not done, see the decision below.

**Decision (2026-09-15, maintainer).**  **Won't fix.**  The reward is unmeasurable — the greedy token never
moved, MTP acceptance is bit-identical across the arms (`0.96712` at `n_max 8`, `0.93186` at `n_max 15`),
and the perturbation is 0.014-0.064 logits against a top-2 margin of 2.2-2.7, one to two orders of
magnitude *below* the error the coarse KV quantization itself imposes.  The plausible fix (make every
width process the same KV range) adds work to the single-token decode — the latency-critical width — for
nothing, and it is the same class of retrofit §19 records as costing 0.5-9 % on whichever axis it touches
and recurring with every new single-token-tuned kernel.  The **trigger to revisit is an *argmax* change**
(a quality event); a hash change with a 2.2+ margin is not one.

**Doctrine (2026-09-14, maintainer decision; sharpened 2026-09-15 by the table above).**  Tier by
*coherence*, not by bits, and be explicit about *which* level the guarantee is at:

* **The contract the delivery actually guarantees is a text/acceptance-level one**: for **f16, bf16 and
  q8_0** the greedy text is bit-identical across the whole decode/verify band (`plain == draft-mtp` for
  `n_max <= 7`, MTP acceptance unchanged), and that is *tested*, not merely measured.
* **Logits-level band purity is a measurement, and it is stated as one.**  For f16/bf16/q8_0 it held at
  every (prompt, P) tested **except bf16 at P=200 on the 4B**, which is recorded above — with the argmax
  unchanged.  A new single-token-tuned kernel can add another such edge anywhere, so the delivery
  re-measures this table on the prompt/P grid rather than promising the grid cannot move.
* **For the small quants (q4_0, q4_1, q5_0, q5_1, iq4_nl) we make a good-faith effort**: their values are
  bit-identical to the reference conversion (the dequant matches `convert.cu`, which is what the op-test
  oracle checks) and the kernel family is uniform across the band, but we do **not** promise logits-level
  purity; a near-tie may flip, and the flip stays argmax-stable as above.
* **Why**: at those quantizations the K/V cache is already the dominant long-context coherence loss, so
  the discrepancy a near-tie flip makes reproducible is the same order as the error the quantization
  itself introduces.  Guaranteeing bit-identity there buys a promise about a regime the model should not
  be run in, at a real and *recurring* engineering cost — every new single-token-tuned fusion recreates a
  band, and the retrofits measured here each cost 0.5–9 % on the axis they touched (or had to disable the
  fusion outright).  A user who needs long-context coherence — and with it the strongest purity — uses
  f16/bf16/q8_0 K/V; the small quants are a memory valve, documented as best-effort.
* Consequence, the same one §19 already states for `n_max > 7`: once the cache is coarse, `plain` and
  `draft-mtp` may disagree on a near-tie.

Evidence: `archive/work/issue-30-mtp-decode-regression/MEASUREMENTS.md` §G; the 2026-09-15 per-quant grid and the
item-2 arms are §I (`results/2026-09-15-purity-native-arms-{a,b,c,d}.txt` -
gfx1201, `results/2026-09-15-item2-purity-halo.txt` - gfx1151, where **all eight types are pure at
P=256**).

## 37. A fully-masked KV group is an exact no-op; skipping it must stay prefill-only (2026-09-26, issue #48)

**Claim.**  A KV group whose kq-mask entries are all `-INF` for every query row contributes
`exp(-inf - max) = 0` to the online softmax, leaves `KQ_max` unchanged and adds exact zeros to
`KQ_rowsum`/`VKQ`; skipping it is therefore **bit-identical**, not merely equivalent.  The issue-#48 skip
(`GGML_CUDA_FA_MASK_SKIP`, default on) is gated on `Q->ne[1] > 8` for the packed mask and on the derived
`cell_pos` (already prefill-only), so the decode/verify band's round-robin KV split and the §10/§11
width-purity invariants are untouched — verified with the 27B `draft-mtp` acceptance and text unchanged.

**Why it is needed.**  A `--kv-unified` cache puts the other slots' cells inside the attention range;
they are fully masked for the current sequence but every prefill ubatch used to process them.
`flash_attn_mask_to_KV_max` trims only the **tail** (and is not even launched at `-ub 512`, because of
its `Q->ne[1] >= 1024 || Q->ne[3] > 1` gate), so the interior blocks were never removed.

**The gate.**  The classification must be per (Q stream, query tile, 256-cell group) on the packed
path: continuous batching puts more than one sequence in a ubatch, so rows of one query tile can belong
to different sequences and a batch-wide test would never fire there.  The derived path stays batch-wide
because `kq_mask_derivable()` guarantees a single sequence and stream.

**Evidence.**  `test-backend-ops -o FLASH_ATTN_EXT` **6354/6354** on both the MMA and tile paths, with
deterministic contiguous interior-hole cases on the derived and packed paths (plus multi-stream, ALiBi,
the `n_q = 9..16` wide-verify band and an odd query count) — before the tests were added the random
masks only planted 128x64 `-INF` blocks, smaller than one 256-cell group, so the packed skip was never
exercised.  `-kvu` B=4 prefill +22 % (MMA) / +72 % (tile); concurrent-server total prefill -7.6 % (MMA)
/ -21 % (tile); single-stream prefill within noise; 27B `draft-mtp` acceptance 0.59331 / adaptive
0.58129 and generated text `832aed3d869d` byte-identical with the skip on or off.  Record:
`WORKLOG.md` (2026-09-26 r10), `patches/README.md`.

## 38. The purity guarantee is scoped to a fixed batch; concurrent serving is not deterministic (2026-09-26)

**Claim.**  The delivery's bit-identity invariants (the `W = 1..8` decode/verify band, `plain ==`
`draft-mtp`, `native == staging`) hold for a **fixed** batch configuration: the same tokens, in the same
order, on the same device, produce the same logits.  They do **not** promise that a *concurrent server*
is run-to-run deterministic, and it is not: the batch composition (which other requests are in flight,
and in which ubatch they land) changes the kernel and reduction order, so a greedy near-tie can flip.

**Evidence (2026-09-26).**  Same prompt, same seed, `-np 2 --kv-unified`, two concurrent requests,
three fresh server runs: the same request yielded different content hashes across runs, and the two
slots' outputs tracked the batch grouping.  It reproduces on **one GPU** with no all-reduce, so it is
not the hybrid AR and not `-sm tensor`; and two **sequential** requests on the same server were
byte-identical across three runs (`e9d0d22658` every time).  It is therefore inherent to continuous
batching (an upstream property), not something the delivery introduced.

**Consequence.**  Never use a concurrent-server text hash as a purity gate or a bisection oracle —
pin the request schedule (sequential, or a fixed arrival order) and compare that, or use
`test-backend-ops` for the op-level guarantees.  The archived investigation is
`archive/work/unified-cache-decode/`.

**Refined 2026-10-11 (issue #134 / TODO #52 part C); made exact and CLOSED 2026-10-19.**  The cause is
narrower than "continuous batching": it is **unified KV** specifically.  `--no-kv-unified` with the same concurrent schedule is
fully reproducible, because each sequence then owns a contiguous region starting at cell 0, so the
flash-attention reduction grid is anchored per sequence.  Under `--kv-unified` a sequence's logits
depend on the physical **set** of KV cells it occupies in the shared stream.  The **exact rule** is: the
result is a function of the sequence's KV cells' tile indices **modulo the per-launch KV-split period**
`pb × nbatch_fa`, where `pb` is `parallel_blocks` for the decode/verify tile kernel (chosen from
`ntiles_KV = ceil(n_kv/nbatch_fa)`, i.e. from the *shared* view length) and the fixed band `P`
(`nsm`, or `(3*nsm)/4` for 2-byte K/V) otherwise.  The kernel takes KV iterations round-robin by
`pb` and `flash_attn_combine_results` sums the partials linearly in block order, so a shift by exactly
one period is **bit-identical** and any smaller shift can change the rounding.  Measured on r91
(gfx1151, 4B Q8_0): offsets 32/64/96/128 (all tile-aligned) give four different hashes, while every
offset is bit-identical to `offset + 960` (`pb` 30 × `nbatch_fa` 32); 35B-A3B (`pb` 60) is identical
under `+1920`.  The decisive variable in the server case is the **decode head**: every new token takes
the shared head, so its residue class follows the co-resident session's allocation history.  It
reproduces on unmodified upstream, so it is upstream-inherent, not a delivery defect.  **Workaround:
`--no-kv-unified` when strict reproducibility is required.**  The proposed cheap fix (chunk sessions by
the band width / align prompt starts) is **refuted** — the quantum is the period, not the tile, and
prompt-start alignment cannot pin the decode head.  A sufficient fix (period-granular per-sequence cell
*ownership*: whole period blocks per session, its head inside its own block, ~30-38 MiB of reserved
cells per live session on the tested models) is an upstream allocator redesign with no throughput win
and is deliberately not implemented.  **CLOSED 2026-10-19**; record:
[`archive/work/kv-cell-offset-nondeterminism/RESOLUTION.md`](../../archive/work/kv-cell-offset-nondeterminism/RESOLUTION.md)
(evidence, tools and the validation plan for a future fix).

**Correction 2026-10-16 (issue #151 follow-up).**  This covers the `--kv-unified` case only.  The
reporter additionally sees a start-to-start greedy flip with HIP graphs on, under both `--kv-unified`
and `--no-kv-unified`.  We could **not** reproduce it on Linux/ROCm 7.14: fresh-process decode-logit
probes are bit-identical across starts, graphs on and off (r69 revalidation,
`archive/work/kv-cell-offset-nondeterminism/FINDINGS-2026-10-16-revalidation.md`).  Under `--no-kv-unified` the
concurrent repeats are reproducible but solo differs from concurrent (a cross-sequence batching
effect, not the cell set).  The reported flip is most likely the known Windows-only per-start flip
(the r25 rope/set_rows fix, and the external hipBLASLt solution selection).  Tracked as TODO #67.

**No performance angle.**  Making concurrent serving deterministic would require either pinning the
batch shape/grouping or carrying the per-sequence-region (period-granular ownership) redesign; neither is
a throughput win, and the field symptom is only a rounding-level near-tie flip, so there is nothing to
gain by fixing it now.
The one performance-relevant knob is batching efficiency, which is orthogonal.  This is the reason the
campaign was closed without a fix (2026-10-19).

## 39. A decode-only fusion must match the standalone launch it replaces (2026-09-28, r18)

**Claim.**  A CUDA fusion that fires **only** in one width of the decode/verify band (here `ne[1] == 1`,
i.e. single-token decode) must reproduce the arithmetic of the standalone kernel it replaces at that
width, or the band is not width-uniform: `plain` and `draft-mtp` diverge as soon as a near-tie is
reached.  Invariant 1 ("one arm, chosen from the `n_tokens` band, never from the exact width") is violated
by construction by a `ne[1] == 1` gate unless the fused kernel is bit-identical to the unfused chain.

**The bug** (found 2026-09-28 while validating the `archive/work/moe-expert-cache/` expert cache).  Block 08's
`ggml_cuda_op_ssm_gate_beta` fuses the qwen35moe alpha/beta Q8_0 projections + their softplus/sigmoid
gating chain, but only at `alpha_w->src[1]->ne[1] == 1`.  Block 13 later added `calc_nwarps_weight()`
(2026-09-12 (18)) - the standalone dense mmvq *weight* launch picks the wide block (8 warps) for Q8_0 with
`K < 4096` - while the decode-only fusion kept plain `calc_nwarps()` (1 warp).  So the W=1 fused kernel and
the W>=2 unfused chain reduced K with different warp counts.  (The comment above `calc_nwarps_weight`
claiming the fusion ops must keep `calc_nwarps` predates this: it is right for the *band-internal* fusions,
e.g. `shexp_down_gate`, whose `nwarps` pins a single-token reduction over the whole band, and wrong for a
decode-only one.)

**Symptom.**  Qwen3.6-35B-A3B Q8_0 (`n_embd` 2048 < 4096), delivered `-ncmoe 99` path, fusions on: r17
`none = 6744006631df`, `n3 = 431bbf3a1605` - not byte-identical, first difference ~200 tokens (a
paraphrase: "not the first engineer and not the last" vs "not first and not last among the engineers").
Fusions off is pure (`6b5dfe0de946`), and the cache campaign's force-copy A/B proved it was not the expert
cache reading stale bytes.  Only `GGML_CUDA_DISABLE_SSM_GATE_BETA=1` restored purity - every other fusion
moves the arithmetic *uniformly* at both widths, so no other kill-switch can equalise `none` and `n3`.

**Fix (r18, block 13).**  `ggml_cuda_op_ssm_gate_beta` now selects
`calc_nwarps_weight(GGML_TYPE_Q8_0, 1, get_device_table_id(cc), long_k)` with `long_k = src1->ne[0] >= 4096`
- exactly the standalone launch's selector.  The fused decode is then bit-identical to the unfused verify
chain and the fusion stays ON.  It is a **no-op for `K >= 4096`**, so the long-K models keep the pinned
single-token order and are unchanged (verified on the 4B: `GGML_CUDA_DISABLE_SSM_GATE_BETA` on/off hashes
equal).

**Result.**  `none == n1 == n3 == n7 == 431bbf3a1605` (300 tokens) and `none == n3` over 1000 tokens on the
delivered path; MTP acceptance unchanged (0.77654, pos-1 0.883); `MUL_MAT_ID` / `GATED_DELTA_NET` /
`SSM_CONV` backend ops 2/2; the 4B coherence gate byte-unchanged.  `v16-84e76d8a2-r18`, tip `135ce8b73`,
tree `df3f6ec94`.

**Rule.**  Any new decode/verify fusion whose matcher has a width condition (`ne[1] == 1`, `ne[2] == 1`,
`n_tokens == 1`) must either be bit-identical to its unfused chain at that width (preferred - this one) or
serve the whole band (`shexp_down_gate`).  A `ne[1] == 1` fusion that does neither is a latent
width-impurity on every model where it fires.

## 40. A thread count is a scheduling parameter, not a numerical one (2026-09-28, r19)

Block 06's r19 amendment lets the CPU run a graph with a *different* thread count depending on what the
graph contains: the offloaded-MoE decode is now capped at `max(1, hardware_concurrency()/2)` instead of
being serialised on one thread (the r17 heuristic's behaviour).  That is only permissible because the CPU
backend's op decomposition never lets the thread count change a result - every output element is reduced
by exactly one thread, so narrowing or widening the pool changes *when* work happens, not *how* it adds up.

Evidence: the delivered `-ncmoe 99` path (Qwen3.6-35B-A3B Q8_0, 300 tokens, seed 42, fusions ON) is
byte-identical at `-t 1`, `-t 8`, `-t 12` and `-t 16`, with `GGML_CPU_MOE_OFFLOAD_THREADS=0` and `=12`, and
with `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1` - all `431bbf3a1605` - and `--cpu-mask` does not move it
either.  The block-13 `calc_nwarps_weight` family (§39) is the contrast: it changes the *K-split inside an
op*, so it did need the full width re-validation.

**Rule.**  A change that only selects a thread count (for a graph, a backend, or an op) is a perf change
and does not need a purity re-run; a change that moves a reduction boundary, a `vec_dot`, or a K-split
*inside* an op does.  State which one a patch is before claiming it is purity-neutral.

## 41. A runtime, address-selected fusion is a cross-process purity risk; a compile-time one is not (2026-09-29, r24/r25, issue #58 item D -> issue #67)

Issue #58 item D: on the reporter's Windows / ROCm 10 box, Q6_K greedy output swaps one near-tie across
fresh `llama-server` starts (~1 in 8-12) while staying deterministic within a start.  This is the §30/§31
mechanism again - `ggml_cuda_check_fusion_memory_ranges()` decides a fusion from **buffer-address
overlap**, so a fusion that is not bit-transparent makes the output a function of the allocation plan.
The `topk_moe` router was the MoE instance and was made bit-identical; this is a dense-model instance.

**Reproduced at the logits level** (gfx1201 / ROCm 7.14, `Qwen3.8-27B-Q6_K`, `prompts/recall.txt`,
`test-logits-width-probe`, P=256, W=1 row-0 hash):

| build / switch | W=1 row-0 hash |
|---|---|
| default | `f6d62323d9339541` |
| `GGML_CUDA_DISABLE_FUSION=1` | `3ab223a4f08afd6e` |
| all address-gated fusions off (temporary probe) | `3ab223a4f08afd6e` |
| `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` | `3ab223a4f08afd6e` |
| `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1` | `f6d62323d9339541` (no change) |
| the other per-fusion switches (conv, norm-rows, pair, q8_1, gdn-gate, rms-scale, ...) | `f6d62323d9339541` (no change) |
| `GGML_CUDA_DISABLE_GRAPHS=1` / `GGML_CUDA_FA_KV_NATIVE=0` | `f6d62323d9339541` (no change) |
| 20 fresh default process starts | `f6d62323d9339541` (20/20) |

So the only lever is fusion, and specifically the address-overlap-gated subset.  The upstream
`ROPE -> VIEW -> SET_ROWS` fusion (`ggml_cuda_should_fuse_rope_set_rows`, upstream #16884) is the trigger
here: disabling just it reproduces the all-fusions-off hash, while disabling the sibling
`RMS_NORM + MUL + ROPE` fusion does not.  On Linux / ROCm 7.14 the layout is stable enough that 20 starts
agree; the reporter's per-start flip is expected wherever the allocator moves, because the guard then
selects the fused or unfused rounding at startup and swaps the near-tie.

The exact arithmetic of the fused kernel has now been separated from the layout side-effect, and it is
the **fused kernel**, not the allocator: a canonicalised per-graph dump of the whole allocation plan
(node addresses, aliasing, buffer sizes) is byte-identical between the two configs, so the `add_alloc_deps`
pass needs no rope entry.  A device-side dump of the exact failing element shows both template
instantiations compute the same `x0`/`x1`/`cos`/`sin` and the same float result `beb67000`
(-0.3563232421875, exactly the f16 midpoint), yet `<float,__half>` stored -0.3562 while `<float,float>` +
`k_set_rows` stored -0.3564: clang contracts the two instantiations' multiply-adds differently, and one
element of the 256x1024 prefill write crosses the rounding boundary.  **Fixed in r25** with
`#pragma clang fp contract(off)` at the top of `ggml/src/ggml-cuda/rope.cu`, so every rope instantiation
uses the same rounding; the W=1 hash is now `60e77916673db071` for both configs and the 4B same-seed gate
is unchanged.  The two kill switches from r24 (`GGML_CUDA_DISABLE_ROPE_SET_ROWS=1`,
`GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1`) stay for bisection.  See `WORKLOG.md` 2026-09-29 (r25).

**Closure (2026-10-03): §41 is a two-part finding, and the residual flip is hipBLASLt.**  The r25 fix
above removes the rope-fusion component, but the reporter still saw a per-start flip on Windows / ROCm 10
(only 1 in 80 with all fusions off, so fusions alone did not explain it).  The remainder was localized to
**hipBLASLt solution selection** in ROCm: the F32 `ssm_alpha`/`ssm_beta` `mul_mat` at 512 prefill columns
goes through `hipblasSgemm` -> `rocblaslt_matmul` and picks a per-process solution
(`HIPBLASLT_LOG_MASK=160`: reference pair `140231`/`140232` vs deviating pair `140216`/`140217`), so the
whole run diverges at startup.  It reproduces on stock upstream with none of these patches; this is
[ROCm/rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126), with
`ROCBLAS_USE_HIPBLASLT=0` as the workaround.  Issue #67 is closed as external (see `WORKLOG.md`
2026-10-03 and `README.md` "Cross-start determinism on ROCm (issue #67)").  The rule below stands on its
own for the rope-fusion defect; the cross-start *symptom* had two independent causes.

**Rule.**  A per-type *compile-time* constant may move the rounding path uniformly across widths and
starts (§4/§19); a *runtime* selection that reads allocator addresses may not.  If a fusion's
`ggml_cuda_check_fusion_memory_ranges()` outcome can change with the allocator, its fused kernel must be
bit-identical to the chain it replaces - otherwise the same weights, seed and prompt can emit different
text across process restarts.

## 39. The MoE expert-cache band follows the routed MMVQ band (2026-10-08, r31, PR #115)

The MoE expert cache took over a routed `MUL_MAT_ID` only for batches of `<= 8` tokens.  The cache serves
such a batch only through the routed-expert **MMVQ** kernel (it reads the arena through the slot remap;
MMQ/MMF do not), and since the extended routed MMVQ band landed (block 10/13) that kernel serves **16**
columns on RDNA4.  A literal 8 therefore stranded 12-16-token MTP verify batches (3+ concurrent streams
at `n_max 3`, or a single stream at `n_max >= 8`) on the host path - ids readback, full device sync and a
copy of every used expert - with the arena unused.

**Fix (block 06, r31).**  The band is now `moe_cache_max_tok_dev(device)`, the narrowest
`get_mmvq_mmid_max_batch` over the device's quantized types (never below the historical 8): 16 on RDNA4,
and 8 on RDNA3 (where the 16-wide routed band is incorrect on gfx1100) and NVIDIA.  It has a single owner
through a new `moe_cache_band` backend iface hook (the Meta backend reports the narrowest over its
devices; no hook = 8), so the scheduler's band decisions and the Meta `moe_cache_update` can never
disagree with the CUDA cache.  `GGML_MOE_CACHE_MAX_TOK=8` restores the old band.  `alloc_all_locked()`'s
`n_tok <= 8` is untouched (it keys the compute-buffer drop, not the cache band).

**Purity.**  Nothing changes inside the `n_max <= 7` guarantee: `none` and `draft-mtp n-max 7` are
byte-identical to the pre-r31 build (`sha 92daa37ab115`), and B=4/B=8 batched decode is unchanged.  At the
newly served widths (9-16 tokens) the guarantee never covered the operation: the arithmetic is the same
routed MMVQ kernel the all-VRAM build already used, so the band-16 cache path is bit-identical to all-VRAM
at `n_max 8` (`sha b1a0ddf528c7`) and the kill-switch restores the pre-r31 text exactly.  Full numbers:
`archive/work/moe-cache-band16/README.md`, `WORKLOG.md` 2026-10-08 (r31).

## 40. A reserve/measure graph must match its memory context's stream count (2026-10-08, r32, issue #48)

`build_attn_mha` splits the query by the cache's stream dim (`n_stream = k->ne[3]`) before flash
attention, and `ggml_flash_attn_ext` asserts `q->ne[3] == k->ne[3]`.  A graph built against a memory
context whose stream count differs from the ubatch it was built for therefore cannot be made valid: the
query view truncates (`q->ne[2]/n_stream == 0`) and the attention output has zero tokens.

**Fix (block 06).**  The TODO #42 post-prefill re-reserve in `llama_context::process_ubatch` passed a
hard-coded `n_seqs = 1` while holding the current ubatch's memory context, so a multi-sequence decode
reserved a 1-sequence graph against an `n`-stream cache and `llama-batched-bench -npl 4` aborted in
`ggml_reshape_3d` (from `build_qkvz`).  The reserve now uses `max(1, ubatch.n_seqs)`, matching the
context.  Single-sequence (MTP) reserves are unaffected, and the multi-stream throughput is unchanged
(16-token MoE batch 705.7 t/s vs r31's 702.5).  See `WORKLOG.md` 2026-10-08 (r32).

## 42. A width gate is not a purity rule: extending a fused chain *down* to one token (2026-10-17, r78)

Two of the r78 block-10 fusions extend an existing fused chain or move a launch, and both keep §39's rule
("a decode-only fusion must match the standalone launch it replaces"):

* **ADD + `rms_norm_q8_1` at one token (PR #174).**  The matcher excluded `ne[1] <= 1` because the
  *attention* residual ADD is folded into the output projection's mmvq epilogue at one token.  The *FFN*
  residual ADD is not (its producer is `moe_weighted_reduction` / a non-matmul), so it ran as a standalone
  `k_bin_bcast` and then the norm.  The fused kernel's values are the unfused pair's (same per-warp
  reduction, same Q8_1 rounding), and the claim is measured, not assumed: same-seed text is
  **byte-identical** to r77 with and without the fusion (dense 27B `sha 7ef45a05dcfe`, MoE 35B-A3B
  `sha e726a23ca0cc`, `-n 3000`, q8_0 KV, `none == draft-mtp n3`), the `W=1..8` probe is unchanged
  (row-0 hash identical), and the prefill-logit gate is unchanged.  The lesson is the converse of §19:
  a band gate (`ne[1] > 1`) can be a *superset* of the shapes a fusion is safe for, and widening it
  needs its own measurement rather than reusing the band's.
* **The KV-rotation FWHT in the rope kernel, and the projection Q8_1 in the attention-output FWHT
  (PR #172).**  Both are bit-identical by construction (the transform is the same helper compiled under
  the same contraction mode; the Q8_1 write is `quantize_q8_1`'s own reduction and rounding under the
  mmvq launcher's own cache key) and both are measured: the `W=1..8` probe at P=512/3584 returns the r77
  hashes with maxdiff 0, the Gemma4 prefill-logit gate against an r77-recorded base is
  **0.000000 / 100.000 %** (the fused rope runs at prefill too), and the greedy text is identical with
  and without `GGML_CUDA_GRAPH_OPT=1`.

Reachability is part of the claim.  #172 fires only for NORMAL/NEOX rope with the norm+mul+rope chain
(Gemma4 and, for the Q8_1 half, gpt-oss-20b on this box); on `rope_multi` models it does nothing, so a
"purity + speed" claim taken from one model does not transfer to another.  PR #173 (folding the q8_0 K/V
cache `SET_ROWS` into the same kernel) was rejected for r78 for exactly this reason -- its matcher
cannot fire on any model here, so no correctness result about it could be produced at all.

---

## Part 2 -- narratives extracted from the rulebook (2026-11)

Extracted from `GREEDY-PURITY.md` in 2026-11, when each entry there was reduced to its claim, rule, numbers
and record.  Each block below is the **pre-compression entry, verbatim** (so it also repeats what the rulebook
kept); the `§` numbers match and nothing here is normative -- the rulebook is.


### §11 -- The real purity range: `n_max <= 7`

The boundary is on the **verify batch width** (`K = n_max + 1`), not on `n_max`.  Two causes:

* **Cause A (fork-specific, FIXED):** the size-based all-reduce dispatch switched to NCCL above a
  per-device-count element count, and the two paths are not bit-identical (the internal pipeline
  BF16-round-trips).  Under `-sm tensor` the reduced tensor scales with the batch width, so a 7-token
  verify crossed the old 2-device limit.  The crossover is now 131072, and MTP throughput *rose*
  (+12 % at `n_max 6` and 12) because the internal pipeline is faster at these sizes.
* **Cause B (deliberate):** the FA tile→WMMA switch at `Q->ne[1] > 8`, and (with FA off) the
  `mmvq`→`MMQ` switch at `ncols == 8`.  Beyond the band the fast kernels have a different reduction
  order; keeping the guarantee would mean giving up WMMA for 9..N-token batches.  There is **no
  corruption** in 1..15 (deterministic recurrent-state sweep).

**Rules.**  Use the **raw-logit probe, not the text gate**, to establish a boundary (text equality is
evidence *for* purity, never *against* divergence — near-ties are rare).  Above the range, gate on
acceptance + MTP-vs-plain throughput; adaptive MTP's `n_max = 12` is deliberately outside.  Upstream is
worse: on `9cf3bf256` (CPU) it diverges at the first width step.

**RDNA3_0 joins the band (r95).**  The boundary above is about the *kernels* a width selects, not about
an architecture, so RDNA3_0 (gfx1100-1102) takes the same verify-band fusions RDNA4 has had since r78:
the 2..8-token Q8_1 pre-quantize fusions (through the one `ggml_cuda_should_fuse_mul_mat_vec_q(...,
verify_band)` predicate all three share), the direct GDN conv for 2..255-token batches, and the batched
f32 `GGML_OP_CPY` run.  It is a **purity** claim, not only a performance one, and it is measured at the
widths:

| gfx1100, `KV=q8_0`, default vs all three fusions off | row 0, `W=1..8` | verdict |
|---|---|---|
| 27B UD-Q4_K_M, P=256 | `c37241bd00c1d561` (both arms) | PASS, worst maxdiff 0 |
| 27B UD-Q4_K_M, P=3584 | `86541410b7c618b0` (both arms) | PASS, worst maxdiff 0 |
| 30B-A3B Q4_K_M, P=256 | `899608bf91600e1b` (both arms) | PASS, worst maxdiff 0 |
| 30B-A3B Q4_K_M, P=3584 | `63ee46bb518130a7` (both arms) | PASS, worst maxdiff 0 |
| 35B-A3B Q4_K_M, P=256 | `423a5f353f0a0e45` (both arms) | PASS, worst maxdiff 0 |
| 35B-A3B Q4_K_M, P=3584 | `6cdc0c5a6601264d` (both arms) | PASS, worst maxdiff 0 |

Same-binary kill-switch identity: default, `GGML_CUDA_FUSE_Q8_1_VERIFY=0`,
`GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`, `GGML_CUDA_FUSE_CPY_BATCH=0` and all three off give the same per-`W`
hashes in every cell above; greedy text at 400 tokens is identical for `none` and `draft-mtp n3` (27B
prose `0f57c09e9ce2`, 35B-A3B prose `4111be939ff4`, 27B code `cfcc349aaa5a`); prefill-logit is mean KLD
`-0.000000` / same-top-p `100.000 %`; and acceptance does not move (R axis `0.52065`, acc per pos
(0.722, 0.500, 0.340)).  RDNA3_5 stays excluded (`0x1150` is outside `GGML_CUDA_CC_IS_RDNA3_0`) and was
verified inert on gfx1151 (same `W=1..8` hashes as r94 and as all-off, no `cpy_batch` run, no
`GDN_CONV_DIRECT` for the verify band).  **gfx1101-gfx1102 are inside the predicate but are not
explicitly validated**, and RDNA3_0 multi-device is an accepted open gap (`TODO.md` #79,
`archive/work/rdna3-verify-fusions/RESOLUTION.md`).

Record: `archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3; narrative
`archive/docs/GREEDY-PURITY-FINDINGS.md` §11.

### §19 -- Purity first: the measured trade

**A purity fix may cost a few percent of raw non-MTP throughput.  That is not a gate failure.**  The
asymmetry is structural: the draft's proposals *are* decode steps, so any arithmetic difference between
decode and verify shows up as draft-vs-verify disagreement, and lost acceptance costs a multiple of
whatever the "faster" path saved.

* the shared-expert band fix (§17) cost ~2.4 % at `pl=8` and **raised MoE acceptance 0.51 → 0.817**
  (167.3 t/s vs plain 96.9, **+73 %**);
* the QSA decode-arm fix (§16) cost ~1.5-2 % at `pl=5/6` and gave **+28 %** (63.9 vs 50.1 t/s).

**Gate for a purity fix:**

1. **correctness** — `plain == draft-mtp` across the whole band (`n_max <= 7`) on every supported KV type,
   or a documented reason why the regime is unreachable by default;
2. **MTP** — pos-1 acceptance >= ~0.45 and `draft-mtp` >= `plain` at the default depth (`n_max 3`); a fixed
   high depth may lose (that is over-drafting, not a regression);
3. **perf** — no hard requirement at the wide verify widths: record the delta, file the optimisation
   follow-up, land the purity fix first, and prefer the side whose arithmetic is the **decode** side.

**Corollary:** a "win" measured before a purity fix may have been measured with a width-dependent path.

**Addendum (2026-10-20, r95) — admitting an architecture to a band is a purity claim, so it is gated at
the widths.**  Extending the verify-band fusions to RDNA3_0 (PR #192 + PR #193) is not a perf-tuning
change: it decides *which arithmetic the RDNA3_0 verify batch runs*, so the deliverable evidence is the
`W = 1..8` probe, not a throughput number.  The design that made it safe is that every fusion was
already **bit-identical to the path it replaces** — the Q8_1 pre-quantize kernels write the values
`quantize_q8_1` writes, the GDN conv keeps its 4-tap order and `silu` with the snapshot tail still
materialised, and the batched CPY copies in `ggml_cpy`'s flat order — so no kernel had to change, only
the arch predicate.  Gate any future arch admission the same way: one predicate, the probe on every
model class the band reaches, `none == draft-mtp`, and the same-binary kill-switch identity.  Numbers
and hashes: §11; record `archive/work/rdna3-verify-fusions/RESOLUTION.md`.

**Addendum (2026-10-20, r93) — a shape-scoped launch parameter must be forwarded to every kernel that
reproduces that launch.**  The fused single-token kernels (`ssm_gate_beta`, `ssm_conv_l2_gatebeta`,
`shexp_down_gate`, the `rms_norm`→Q8_1 fold, the gate/bias folds) each *replace* a dense mmvq weight
launch, so they must call the same selector (`calc_nwarps_weight(type, ncols, table, long_k, <row scope>)`).
A scope applied to the plain launch but not the fused one runs `W=1` at one warp count and `W=2..8` at
another (measured: 35B-A3B 0.150, gemma-26B **3.35** on the width probe).  The scope was the `!has_fusion`
exclusion in the `archive/work/nwarps/` candidate — not a missing reduction-mapping derivation.  The host
must compute the value once and pass it to both the launch dims and the kernel template (§20).

### §37 -- A fully-masked KV group is an exact no-op

A KV group whose kq-mask entries are all `-INF` contributes exact zeros and leaves `KQ_max`/`KQ_rowsum`/
`VKQ` unchanged, so skipping it is **bit-identical**.  `GGML_CUDA_FA_MASK_SKIP` (default on) classifies the
groups per (Q stream, query tile, 256-cell group) on the packed path and batch-wide on the derived path.

**The decode/verify band is covered since r133** (`GGML_CUDA_FA_BAND_SKIP`, default on, PR #223; the
master switch must be on too).  It keeps its round-robin KV split: block i still owns the iterations
i, i+P, i+2P, ... and only leaves out the fully masked ones, so the accumulation order over the remaining
iterations is unchanged, and a block that owns no unmasked iteration writes the same neutral partial
(`max -FLT_MAX/2`, rowsum 0, VKQ 0) the fixup already had to treat as an exact no-op.  The gate is the
band's own predicate, so this is RDNA4 + head 256 + GQA 5..12 + `n_q <= 8`; the tile-kernel decode other
models use keeps the `Q->ne[1] > 8` gate.  **Evidence:** `FLASH_ATTN_EXT` **6414/6414** including 12 new
decode-width packed-hole cases (`nb` 1/2/3/8, f16/bf16/q8_0, all four band kernel variants: fetch-ahead
on/off x wide on/off) — and a mutation that makes the kernel skip one *visible* group fails all 14 of the
hole cases, so the coverage is real; the `W = 1..8` logits probe is **PASS, worst maxdiff 0**, with
bit-identical row-0 hashes for `q8_0`/`f16`/`bf16` KV with the skip on and off; `plain == draft-mtp` text
identical at `-n 3000` on 1 GPU and 2 GPU `-sm tensor`; the four-axis MTP gate's acceptance identical to
the last digit.  **Why it is needed:** a `--kv-unified` cache puts the other slots' cells inside the
attention range, and `flash_attn_mask_to_KV_max` only trims the tail, so the band read them all on every
step; `LLAMA_SERVER_LAZY_IDLE_SLOTS` (r130) then made "a long conversation sitting idle in front of an
active one" the common case (a resident ~105K context: **50.8 -> 41.7 ms/step, -18 %**, token ids
identical).  With nothing to skip the per-layer prepass is **not** a tax (the padded tail group is skipped
too: -0.6 % at 9.8K, neutral at 105K single-slot), which supersedes the 2026-09-26 break-even reading in
`archive/work/unified-cache-decode/` (that campaign measured a 2-slot *batched-bench* cell with a small
foreign fraction, where the prepass and the saving did cancel).

The `--kv-unified` non-determinism of §38 still applies: a concurrent server's batch composition changes
the kernel/reduction order, so use a sequential schedule or `test-backend-ops` as the purity oracle.

### §44 -- A per-start hipBLASLt solution pick is not a code defect — pin it before comparing starts

**Claim.** On this box (gfx1201, Linux, **ROCm 10.1.0**) the *whole* residual run-to-run divergence of the
`gemma4 26B-A4B` + MTP config is ROCm's **hipBLASLt solution selection being random per process start**
for an untuned FP32 GEMM size — issue #67 / [ROCm/rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126).
`ROCBLAS_USE_HIPBLASLT=0` removes it.  It is **not** a switch to ship: it is the tester's determinism
control, and disabling hipBLASLt costs decode (see **Cost** below) — the delivery keeps it **enabled**.

**Which GEMM.** Exactly one hipBLASLt dispatch in the run: `m=128 n=375 k=2816`, `transA T`, `transB N`,
`f32_r`, **30 calls** (= one per target layer) — `m = 128` is Gemma4 26B-A4B's `n_expert` and `k = 2816` its
`n_embd`, so this is the F32 **MoE router** (`ffn_gate_inp`) at the one prefill width that lands in the
untuned-FP32 window (other widths take mmvf/MMQ).  `HIPBLASLT_LOG_MASK=160` shows `solution_index 140231`
or `140232` per process, and on *this* shape the two are **not** bit-identical: `6f6923ef36` (140231) vs
`51274cb0f2` (140232), first-draft-call top-1 **0.458 vs 0.352**.

**Numbers** (gemma4 26B-A4B Q8_0 + `mtp-gemma-4-26B-A4B`, `-ngl 99 -ncmoe 40 -sm layer --fit false
-c 16384 --spec-type draft-mtp --spec-draft-n-max 3 -n 300 -t 8 --seed 42 --temp 0 --ignore-eos`, 2 GPU,
draft-candidate-logit fingerprint, one fresh process per sample):

| arm | result |
|---|---|
| default environment, N=18 | **3 classes**: `6f6923ef36` ×12, `51274cb0f2` ×4, `10e1f21f4c` ×2 |
| `ROCBLAS_USE_HIPBLASLT=0`, N=10, `-n 300` | `6f6923ef36` **×10** |
| `ROCBLAS_USE_HIPBLASLT=0`, N=3, `-n 3000` | `19eed3da33` **×3** (904 draft calls) |
| cost on *this* config | prompt **340.1** vs 327.3 t/s, gen 93.02 (sd **0.06**) vs 91.86 (sd **1.65**) |
| cost in general | **not free**: the maintainer's wider measurements show up to **-15 % decode** with
hipBLASLt disabled (partly offset by a prefill gain — the gemma4-MTP number above).  Hence: keep hipBLASLt
**enabled** by default, set `ROCBLAS_USE_HIPBLASLT=0` only for a run that must be reproducible |

**Second call site — the GDN F32 `ssm_alpha`/`ssm_beta` GEMM (2026-10-21, closes `TODO #75`).**  The
same lottery, on a different op: Qwen3.6-35B-A3B Q4_K_M, `llama-perplexity -c 512 --chunks 1 -fa on`, fresh
process per sample on **this box with ROCm 10.1.0** — **3 logit classes in 8 starts** (`3629d90733a1` x5,
`a855c2000912` x2, `6d32945460bd` x1) vs **1 class in 8/8** with `ROCBLAS_USE_HIPBLASLT=0` (`2d9ed9ce03cb`).
One hipBLASLt shape in the whole run, `A[F32 2048x32] x B[F32 2048xN] T,N` (1080 dispatches; the
`llama-cli` prefill+decode arm serves two shapes, that one and the `ffn_gate_inp` router's
`A[F32 2048x256]`, 159 dispatches total), and the class is
the TensileLite spine it picked: `MT128x64x8` / `MT64x32x32` / `MT128x128x8` -> three different logit hashes.
Mean KLD between two classes **0.003479**, same-top-p **98.431 %** — *inside* the prefill-logit gate, so the
gate cannot flag this by construction.  Also: the ordering is **stable in some configs** (`test-logits-width-
probe` was 10/10 single-valued on the same build), so a config that does not flip is no evidence about the
mechanism; and greedy text was identical 12/12 across class-varying starts — judge at the logits level.
**Cost here is nil** (35B-A3B, 1 GPU, two interleaved `llama-bench` visits): `ROCBLAS_USE_HIPBLASLT=0` measures
**+1-3 % prefill** (pp512 4749.7/4840.9 -> 4899.8/4887.2, pp2048 4804.5/4798.1 -> 4841.9/4830.3) and flat
decode (tg128 108.2/108.0 vs 107.9/108.0), and peak VRAM 21822 -> 21752 MiB.  It is the model- and
runtime-dependence of that number that makes `ROCBLAS_USE_HIPBLASLT=0` a tester's control and not a shipping
default: on gemma4 above it is a real prefill+decode *win*, elsewhere it is the -15 % decode cost.  Removing
the dispatch instead of pinning it is block 06's `mmf32` (`GREEDY-PURITY.md` §47; campaign record in
`archive/work/hipblaslt-bypass/`, `TODO #82` closed at r104).

**Its outside-slab footprint is a constant ~70 MiB (2026-10-21, measured).**  Qwen3.8-Flash-Next GSQ, 2 GPU
`-sm tensor -ncmoe 48`, `GGML_CUDA_SLAB_STATS=1`: the slab's `free_min` is 53-88 MiB higher with
`ROCBLAS_USE_HIPBLASLT=0`, **independent of ubatch (512..8192), prompt length (602..68 262 tokens), context
(16K..131K) and model**.  It does not scale, so it is TensileLite code objects plus a small fixed arena, not a
per-call workspace (the log shows 0 / 12 / 24 / 64 KiB requested per call).  Do **not** attribute the slab's
~2 GiB headroom floor to it — that floor is the MTP draft buffer, the FA-QSA workspace, the pool high-water
mark and the model's own PLE/attention allocations.  What hipBLASLt is *worth* is model-dependent: free on
35B-A3B, **+51 % prefill** on Flash-Next (`-ub 8192`, 68 262-token prompt: 2405 vs 1595 t/s), which is why a
BF16 bypass is a performance proposition while the F32 one is a correctness win.

ROCm 10.1.0 does **not** have the fix (its `rocm-libraries` pin branched 2026-09-09); it is in
`c50ea60130f`, develop 2026-10-02, TheRock nightlies from 2026-10-06.  So this *does* reproduce on **Linux**
with 10.x — §41's "Linux / 7.14 does not reproduce it" holds for 7.14 only.

**Retraction this replaces — do not re-derive the graph story.** With the library path off, *graphs on and
graphs off land on the same single class*, so graph capture/replay was never implicated.  Every capture arm
of the r100 hunt was this lottery resolving to a particular class: "target graphs off is stable at
`8f795fbaf0`" (5/5, correlated with the graphs-off arm), "draft-only graphs off", "target 1-token graphs
off", "target verify graphs off" (a lottery), "force a fresh capture on every replay" (deterministic — so
not stale baked pointers either).  Also cleared by direct arms: the draft's input embedding is bit-identical
across classes (`h_nextn`/`pending_h` hashes of the three prompt ubatches agree exactly), and the
async-input ring is not the residual (post-`0005`: `ASYNC_INPUT_BATCH=0` is itself a stable class,
`ASYNC_INPUT=0` is still a lottery).

**Rule.**  A cross-start (fresh-process) purity, coherence or text comparison must pin the external
solution selection (`ROCBLAS_USE_HIPBLASLT=0`) or must not be used to accept/reject a code change.  A
per-start difference is not evidence about a kernel, a graph or a cache until it survives that pin.  Decode
at `n=1` is mmvf and unaffected, so the exposure is **prefill-width**: it can move a *reference* recorded in
another process, which is what happened to two "discrepancies" in the r100 handover (non-reproducible gemma4
text hashes; gen 74.9/72.8 vs the recorded 79.0/74.7).  **Do not pin it in the delivery's default
environment**: the upstream fix lands at the source, and disabling hipBLASLt costs up to -15 % decode.  The
env var is the tester's determinism switch, not a shipped default.

Where: `archive/work/fix-gemma-4/evidence/MEASUREMENTS-2026-10-21d.md`; discussion #196.

### §45 -- A split the op cannot serve locally is a refusal, not an approximation

**Claim.**  The Meta backend's per-op split-state handlers decide whether an op is **local** given how a
tensor is divided across devices.  The only splits they may accept are ones where no device **straddles the
reduction**: each device's slice along the reduction axis must be either empty or the whole thing.  A
straddling split needs a cross-device reduction that these handlers do not implement, and the honest
behaviour is a hard refusal naming that, never a "close enough" arm — a wrong split state computes part of
the result on a device whose inputs are missing, which is silent corruption, not a rounding difference.

**The three shapes that are local (r101).**

* **Per-row ops** (`NORM`, `RMS_NORM`, `GROUP_NORM`, `L2_NORM`, `SUM_ROWS`, `MEAN`, `ARGMAX`, `ARGSORT`,
  `TOP_K`): dim 0 is the reduction axis, so an axis-0 split is normally cross-device.  It is local when
  every per-device extent is empty or the **whole row** — gemma4's full-attention layers have one KV head,
  so their 512-wide K/V row lands whole on one device (`ne_per_dev = [0, 512]`, measured).  A straddle
  (`[256, 256]`) still aborts.  The elementwise companion — a split tensor times a **MIRRORED same-shape**
  operand (the per-head norm weight) — follows the split operand, because nothing is reduced and the
  mirrored operand is present in full everywhere.
* **The rotated KV cache** (`-ctk q8_0` => `n_rot_k = 512`, `llama_mul_mat_hadamard`): `rot` is
  `512x512` MIRRORED and `Kcur` is reshaped to `[n_rot_k, rows]`, so the **contraction is axis 0** of
  `src1`.  Local iff every per-device contraction extent is empty or whole **and** `out->ne[0] ==
  src1->ne[0]` (a square contraction, so each device's output slice maps 1:1 onto its contraction slice);
  the owning device computes the complete dot product because every `rot` column is available to it.
* **Flash-attention K/V** with more than one sequence in the ubatch (`n_stream > 1`: `llama-server -np > 1`,
  `llama-batched-bench -npl > 1`, `llama-perplexity --kl-divergence`): the FA cache view spans the cache's
  stream dimension, so gemma4's whole-on-one-device KV arrives split on the cache's **embd** axis instead
  of the kv-head axis.  Both mean the same thing, and the extra condition is that the **query** is
  empty-or-whole on the same devices — a device holding queries but no K/V would compute rows nothing
  serves, which is exactly the broken combination to keep refusing.

**Why this is a purity rule, not a convenience.**  Each accepted shape's alternative is an abort, so the
temptation is to accept a straddling split "since the data is all there".  It is not: with a straddle, no
device has the whole reduction, so every device produces a *partial* sum that must be reduced — a feature
this layer does not have.  Accepting it would produce coherent-looking text whose logits are wrong, the
failure class §7 and the prefill-logit gate exist to catch.

**Measured.**  gemma4 12B Q8_0 with `-sm tensor` is byte-identical to `-sm layer` (`c740be0e131a`, 143
chars) in f16 **and** rotated `-ctk/-ctv q8_0`, at 2 and 3 GPU; the rotated single-device prefill-logit
gate reproduces its recorded base to six digits (`-0.000000 / 99.990 %`), and under `-sm tensor` the
rotated arm is indistinguishable from the f16 one (`0.2041 / 86.0 %` vs `0.2134 / 86.7 %` against the
`-sm layer` reference — the ~0.21 is the tensor split's own all-reduce rounding, identical with no rotation
in the picture).  The multi-sequence fix is verified the same way: `llama-batched-bench -npp 512 -npl 1,4`
runs (3353 t/s at B=4 vs 2896 `-sm layer`), and two concurrent 400-token requests on `llama-server -np 2`
both return 200 where the process used to abort.

Where: `archive/work/fix-gemma-4/RESOLUTION.md`; measurements
`archive/work/fix-gemma-4/evidence/MEASUREMENTS-2026-10-21f.md`.

---

### §46 -- A tile→WMMA family switch is judged at a calibrated point, against a same-kernel control

**Claim.** The prefill-logit gate's absolute limits (mean KLD `<= 0.005`, same-top-p `>= 98 %`) decide a
change *inside* a kernel family: a quant type, a precision change, an approximate path.  A **kernel-family
switch** (the r5/r99 FA head caps, the r102 RDNA3_0 head-512 rows) does not share a reduction order with
the kernel it replaces, so its divergence is the tuning, not a defect.  Two guards make it decidable:

1. **The point must be one the model is confident about** (base PPL inside the gate's calibrated domain,
   ~6).  At a point it finds hard, *every* reordering is far outside the limits, so the number says
   nothing about the candidate.  This was the whole of the r102 "failure" (`TODO.md` #81, closed
   2026-10-21); the model there is a reasoning model on raw wikitext, 45x outside its confident range.
2. **A same-kernel control measured on the spot** — the same kernel at a different ubatch — so the
   candidate is compared with the floor of what a reordering costs there, not only with an absolute
   number.

**The measurement (gfx1100, gemma-4-26B-A4B-QAT UD-Q4_K_XL, f16 K/V, one base file per point, one base
per arm.)**

| point | base PPL | same-kernel control (`-ub 512` vs `-ub 256`) | head-256 rows vs fully tile | head-512 rows, marginal (vs the delivery without them) |
|---|---:|---|---|---|
| wikitext-2 `-c 512 --chunks 40` (the default gate point) | 607 | **0.104989 / 85.000 %** | 0.044055 / 90.627 % | 0.029285 / 92.588 % |
| in-domain text `-c 512 --chunks 30` | 13.7 | 0.036654 / 93.660 % | 0.022562 / 96.000 % | 0.011303 / 97.451 % |
| in-domain text `-c 2048 --chunks 10` (`prompts/fa-family-switch-gemma4-26B-A4B.txt`) | **2.65** | 0.013962 / 97.221 % | 0.005994 / 98.324 % | **0.001778 / 98.967 %** |

Per switch at that point (`--candidate-env`): head-512 marginal **0.001778 / 98.967 % → PASS**; head-256
alone vs fully tile **0.005994 / 98.324 % → FAIL** (20 % over the KLD limit, top-p passing, and 2.3x better
than the control); the whole set vs fully tile 0.006150 / 98.227 % → FAIL.  So the remaining
delivery-relevant residual is the **r5 head-256 rows**, not the head-512 ones.

So the head-512 rows **pass the absolute limits at a calibrated point** (0.00178 / 98.967 %, 2.8x inside
the KLD limit and 8x better than the same-kernel control), and at every point they are more faithful than
the head-256 rows shipped default-on since r5 — which are the bigger *total* contributor only because
there are 25 of those layers against 5.

**No row parameter moves the number** (all measured against the same bases, gfx1100, deployable binary):
an fp32 PV accumulator (`T_C_VKQ`; the one structural difference from the tile kernel's fp32 accumulator)
gave 0.030489 marginal at the wikitext point and cost **-32 %** throughput; `nbatch_fa` 64 gave 0.030865
at -19 %; `nbatch_fa` 128 (+`Q_in_reg=true`, the tile kernel's strip length) gave 0.029081 at -14 %;
`Q_in_reg=true` alone is **bit-identical** (KLD to every printed digit) at **-29 %** — the win really is
the spills and it has no accuracy effect.  The divergence is the reduction order; attribution table and
per-variant records: `archive/work/fa-rdna3-head512-purity/`.

**The op-level accuracy reference is the second, deterministic leg.**  `FA_DBG_NMSE=1 test-backend-ops -o
FLASH_ATTN_EXT` prints the NMSE each case already computes against the CPU backend, so the two families can
be compared on the model's own prefill shape (`hsk=hsv=512, nr23=[8,1], kv=32768, nb=512`): f16 5.99e-5
(WMMA) vs 5.71e-5 (tile), q8_0 2.30e-5 vs 1.64e-5, threshold 5e-4 — within a factor 1.3, neither an
accuracy regression; at nb=24/64 WMMA is 1.1-2.8x *closer* to the CPU reference.  (`FA_DBG_SEED=1` pins the
random input so the two arms are comparable; without it the test data is redrawn per run and NMSE moves by
up to 13x.)

**Rule.** For a family switch run `scripts/gate-prefill-logits.sh --family-switch` (protocol:
`benchmarks/prefill-logit-methodology.md`): it refuses a point whose base PPL is above `--family-max-ppl`
(default 10), records a control base with the fallback arm at `--control-ub` (default 256), always prints
the control, and passes on the absolute limits or on "no worse than the control".  Record the op-level
NMSE pair too.  This is not a license for within-family changes: those still meet 0.005 / 98 % at the
default point.

**Historical correction.**  The r102 numbers that motivated this campaign were the *marginal* pair measured
against the delivery without the 512 rows (`0.0293 / 92.59 %`), while the fully-tile absolute pair is
`0.0462 / 90.42 %` — and the delivery's own head-256 rows already cost `0.0441 / 90.63 %` against fully
tile.  The `WORKLOG` 261021.004 line labels the marginal pair "head-512 WMMA vs tile".

### §47 -- An exact F32 GEMM in a different summation order is a family switch — and owning the path is the fix

**Claim.**  On RDNA the F32 prefill GEMMs with more than 16 token columns (the GDN `ssm_alpha`/`ssm_beta`
projections and the MoE routers) used to fall through `ggml_cuda_mul_mat_cublas()` to rocBLAS, i.e. to
hipBLASLt.  That is a *per-start* purity defect, not a numeric one: hipBLASLt picks its solution for a shape
from a heap-ordered container, so a fresh process lands on a different TensileLite spine and the prefill
logits move (§44; TODO #75: **3 classes in 8 starts** on 35B-A3B `llama-perplexity`, 3 in 18 on gemma4).  It
is also slow: on `m=32 n=512 k=2048` the heuristic's class costs 0.13-0.19 ms against 0.046 ms for a
well-chosen solution.

**Fix (r104, block 06).**  `mmf32`: our own SIMT tiled GEMM for exactly those shapes (`mmf32.cu`).  There is
no F32 MMA form on RDNA (§2 arm B / `amd_mfma_available` is CDNA-only), so this is a plain tiled kernel — 32
output rows per block, a 4×8 thread sub-tile, and the k range split into contiguous chunks summed in
increasing z.  That is a **fixed reduction tree with no atomics**: one logits class in **12/12** fresh starts
at 1 GPU and **6/6** at 2 GPU `-sm tensor`, and `--chunks 8` dumps byte-identical.  No approximation anywhere
(it is an F32 FMA accumulation in a pinned order), and 0.032 ms on `32×512×2048`.

**It is still a family switch, so it is gated as one.**  Its divergence from the rocBLAS path *is* the
reordering, and it cannot be driven to zero (§46).  Measured with `--family-switch` (2026-10-21, gfx1201):

| point | base PPL | candidate | same-kernel control | `ROCBLAS_USE_HIPBLASLT=0` |
|---|---:|---|---|---|
| wikitext, `-c 512 --chunks 40` | 5.8194 | **0.005787 / 96.480 %** | 0.006017 / 96.588 % | 0.005969 / 96.667 % |
| in-domain, `-c 2048 --chunks 7` (`prompts/kld-corpus-35b-in-domain.txt`) | **1.4345** | **0.014646 / 98.045 %** | 0.018100 / 97.808 % | — |

At the confident point it is no worse than the control on **both** metrics (PASS with
`--allow-control-pass`); at the wikitext point it is closer to the default than both the control and the
`=0` arm.  **The absolute limits are missed at both points by every reordering, including the control** —
that is §46's calibration, not a defect of this kernel.

**The op-level accuracy leg** (deterministic, r103's instrument): `LLAMA_TEST_OP_NMSE=1 LLAMA_TEST_SEED=1
test-backend-ops -o MUL_MAT` puts the F32 kernels at **6.1e-14** (ours) vs **6.9e-13** (rocBLAS) against the
CPU reference on the two cases the kernel intercepts, threshold 5e-4 — i.e. both at F32 rounding, ours
closer, and a case the kernel does not take is bit-identical.

**Rule.**  A GEMM family we own must be (a) exact — no precision change hiding behind a "faster kernel";
(b) deterministic by construction (fixed reduction order, no atomics, no per-start choice); (c) judged by
§46's family-switch protocol at a confident point **and** by the op-level NMSE pair; and (d) **re-baselined**
once: a shipping default that moves the prefill arithmetic means the prefill-logit base is recorded on the
new arm (`benchmarks/prefill-logit-methodology.md` §5b).  Any later tuning of the kernel's tiling is a
within-family change and must meet the absolute limits at the default point.

**Scope, and what is deliberately not claimed.**  F32 only, `m <= 256`, `17 <= n <= 65536`,
`256 <= k <= 8192` — decode cannot reach it (`n >= 17`, so `W = 1..8` band uniformity is untouched by
construction) and BF16/f16 prefill is untouched, so Flash-Next keeps both hipBLASLt's +51 % and its own
per-start lottery (a BF16 counterpart is a separate job, not delivered).  A full-F32 model's FFN GEMMs
(`m` ≫ 256) still reach hipBLASLt: an accepted limitation, recorded in `patches/README.md` (block 06).
Default-on for **RDNA4 and RDNA3_5** (r105 extended it); **RDNA3 stays opt-in** — gfx1100 measured
−1.3…−2.2 % at `pp2048` (one visit a tie), so there the lever would be a per-arch tile (`BN`/`ks`), not the
gate.  RDNA3_5 (gfx1151) passed §19's arch-admission legs: `test-backend-ops -o MUL_MAT` 2444/2444 with
NMSE **6.11e-14** vs rocBLAS 6.83e-13 (a case the kernel does not take is bit-identical), **12/12** fresh
starts one logits class, **+1.0 % pp512 / +1.6 % pp2048** with decode flat, and this same family-switch
protocol at the shipped point: base PPL 1.4345, candidate **0.013016 / 98.673 %** vs same-kernel control
**0.019125 / 97.975 %** (no worse on both, PASS with `--allow-control-pass`).

### §49 -- A fused chain's launch geometry comes from the flat projection, not from a consumer's reshaped view

**Claim.**  The GDN gate/beta fusions replace an unfused chain whose tensors are reshaped for the graph
(`alpha = MUL_MAT(wa, y)` is `[nr, n_tokens]`, then `RESHAPE3D` to `[nr, n_seq_tokens, n_seqs]`, then
`ADD(dt)`, `SOFTPLUS`, `MUL(a)` and a `RESHAPE4D` back).  When that chain is collapsed into one `mmvf` /
`mmvq` launch, the launch geometry must describe the **projection**: `dst->ne[1]` columns and `dst->ne[2]`
channels with `y`'s own shape.  The r102 form (`GGML_CUDA_FUSE_GATE_BETA_BF16`, `mmvf.cu`) passed the gate
chain's `MUL` as dst instead, so with more than one sequence in the ubatch the launch saw
`[nr, n_seq_tokens, n_seqs]` against a flat `y` with one channel: every channel past the first re-ran the
beta rows and wrote the values computed from `y`'s first `n_seq_tokens` columns.  **Every sequence but the
first silently took sequence 0's alpha/beta gates.**  The text stays coherent (the gates are all plausible
decays), so same-seed greedy text, `W = 1..8` width probes and MTP acceptance are all blind to it — only
multi-sequence serving sees it (issue #204: 4-stream draft acceptance 0.85 → 0.82, 4-stream throughput
−3.5 %).  The fix is to launch with the projection's own shape (`ggml_tensor dst_flat = *alpha_w;
dst_flat.data = gate_mul->data;`): both tensors are contiguous with the same element count, so it is the
same memory written flat, and a single-sequence launch is byte-for-byte unchanged.

**Rule.**  Gate a fusion on the **multi-sequence** batch, not only on the single-stream width band:
`tests/test-fusion-gate-beta-multiseq.cpp` (shipped in r108) builds the chain for a matrix of
`(n_seq_tokens, n_seqs)` shapes with no model, compares the fused launch against the same graph with the
fusion made ineligible (one extra consumer of the `ADD` output makes `ggml_can_fuse_subgraph` reject the
window), requires bit-identical gate and beta values, and proves the fusion fired by leaving the skipped
`SOFTPLUS` output at a NaN sentinel — so a change that silently stops matching fails the test instead of
passing it vacuously.  Any new fusion that consumes a reshaped chain belongs in that matrix.

Two facts a reviewer needs.  (1) **The exposure band is per model**, because a fusion only runs where the
plain matmul would be an `MMVF`/`MMVQ` of the right family: `ggml_cuda_should_use_mmvf` returns
`ne11 <= 3` for BF16 on a device with bf16 MMA when `should_use_mmf` is applicable (a 48-row projection such
as Qwen3.8-27B / Flash-Next keeps the whole 8-token band; a 32-row one, e.g. Qwen3.5-4B, is limited to 3), so
"4 streams × 2 tokens" is exposed on the 27B-class projections on gfx1201 too — the incident's shape.  (2)
`GGML_CUDA_FUSE_GATE_BETA_VERIFY` (the Q8_0 sibling, `mmvq.cu`) was **never** affected: it sizes itself from
`ggml_nelements(dst) == nrows*ncols` with contiguous dsts and asserts it, which is exactly the shape rule
above.

**Evidence (gfx1201, 2026-10-22).**  Lockstep probe (one token per sequence in one batch, distinct prompts,
8× R9700/…): `n_seqs = 1` fused == unfused and pre-fix == post-fix (bit-identical logits); `n_seqs = 2, 3`
(pre-fix) differ from the unfused arm at every step, sequence 0 unaffected, sequences ≥ 1 wrong and equal to
each other; post-fix all arms bit-identical, and the post-fix fused arm equals the pre-fix **unfused**
reference.  Same on Qwen3.8-27B-BF16 (3 GPU, `-sm layer` and `-sm tensor`) and on Flash-Next GSQ IQ3_XXS
(qwen4exp, 3 GPU) — the reporter's model class.  `test-fusion-gate-beta-multiseq`: **exit 2 with 7 failing
rows** on the r107 build, **PASS (11/11 fused, 0 failed)** on r108.  Single-stream is unchanged: 4B/27B/
35B-A3B greedy text identical pre/post, and the prefill-logit A/B against the pre-fix build measures mean
KLD **0.000000** at `-ub 8` and `-ub 512` (27B Q8_0) — mmvf is the decode/verify band only.  Full family
sweep (7 switches × the models that fire each arm): every other arm was already multi-sequence-clean.

### §57 -- Parking the idle slots restores the save-and-clear attention range for a request that runs alone

**Claim.**  A unified KV cache makes a sequence's attention result depend on the cells of every resident
sequence (`§38`): the FA tiling/split reads `n_kv` and the cell indices.  `LLAMA_SERVER_LAZY_IDLE_SLOTS`
(r130) keeps idle slots resident, which made that placement dependence reach plain sequential serving: the
same 16 requests sent one at a time in two rounds agreed on 4/16 in the reporter's test (13/32 in ours).
Parking the idle slots at the top of the cache and moving the running slot to the low cells (`seq_park` /
`seq_unpark`) gives the running slot the same range and indices as when the idle slots are saved and
cleared, so it reproduces the `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` output.

**Rule.**  `LLAMA_SERVER_PARK_IDLE_SLOTS` (on unless `0`) is part of the lazy-idle-slot path and applies
only to a `--kv-unified` cache.  A cache can park only when it has one stream, cells no other cache shares,
V not transposed and no sliding window (`llama_memory_can_park()`); anything else keeps #210 unchanged and
logs once.  New cells (`find_slot`) and `get_n_kv()` stop below the lowest parked cell, and every path that
moves or removes cells keeps the parked ranges honest (`park_refresh` on `seq_rm`/`seq_keep`; `seq_cp`
gives parking up).  An idle slot whose cells are not one run of its own (shared through `seq_cp`, e.g. a
cached prefix restore) cannot be parked: it is saved and cleared by default, or left resident with
`--no-cache-idle-slots`, so the output stays the lazy-off output either way.  Do not claim determinism for
concurrent decoding: parking does not change batch composition (§38).  **A memory whose second cache
mirrors the attention cells parks both or neither (r146, issue #227):** `llama_memory_hybrid_idx` (qwen4exp,
glm5-next) answers `can_park()` / `seq_park()` itself, moves the attention cache and then the indexer
(undoing the first if the second refuses), and refuses — leaving the slot to the save-and-clear fallback —
when the two no longer mirror each other (`attn_idx_in_step()`).  A successful move stales the pool layout
and drops the derived indexer pool's watermarks, because the derived rows are indexed by the block ordinal
of the cells *in the decode window*, which a park changes.  Never park a cache whose cells another cache
indexes by cell number without moving that cache: r136 matched `llama_memory_hybrid_idx` through its
`llama_memory_hybrid` base and left the indexer behind, which aborted the server on the model's own guard.

**Evidence (r136, gfx1201, ROCm 10.1.0).**  27B UD-Q4_K_XL + DFlash2-Q4_K_M, 4 slots, `-c 131072
--kv-unified -fa on -ctk/ctv q8_0`, 16 varied prompts, two rounds, `cache_prompt: false`: parking
**16/16** rounds and **32/32** equal to lazy-off; parking off (#210) **13/16** and **13/32**.  With
`cache_prompt: true` and a shared prefix (cells shared by `seq_cp`), both parking and lazy-off are 12/16
across rounds and still 32/32 equal; the fallback is by design.  Dense Qwen3.5-4B (hybrid) parking ==
lazy-off 16/16; 2 GPU `-sm tensor` on the 4B has no parking failure; `-fa off` (V transposed) and
Gemma-4 E4B (sliding window) report parking unsupported and are text-identical on/off.  The prefill-logit
gate on the release build is KLD 0.000000 / 99.980 %, and the 4B coherence is byte-identical to r135 on
1 GPU and 3 GPU `-sm tensor`.  The reporter did not cover MoE with the expert cache, vision, >4 slots or
Windows, and the PR's switch-time table was not independently re-measured.

**Evidence (r146, issue #227, Flash-Next GSQ-RCO IQ3_XXS + MTP, gfx1201).**  The r145 build aborts with
`qwen4exp.cpp: GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv())` on the reporter's
sequence — a ~9.9k prompt completes on slot 0, a short prompt starts on another slot and the parked slot's
attention cells move while the indexer's stay behind — in **both** `-sm tensor` and `-sm layer`.  With both
caches moved, the same 4-request sequence (A 10.6k on slot 0; B on slot 1; C back on slot 0, which unparks
A; D on slot 2 with two slots parked) is byte-identical to `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` on every
request, in `-sm tensor` (3 GPU, all-VRAM) and `-sm layer` (3 GPU), and on 2 GPU with `-ncmoe 44` in both
(`-sm layer -ts 59,41` is the reporter's shape); `LLAMA_SERVER_PARK_IDLE_SLOTS=0`
differs on 2 of the 4, i.e. the workload does reproduce #210's placement dependence.  Both caches log each
move under `LLAMA_KV_PARK_DEBUG=1`, and no GPU fault is reported.  `scripts/gate-park-idle-slots.sh` is the
repeating gate (it fails on r145 with the reported assert).

Record: `patches/README.md` (block 12), `ENVIRONMENT.md` §15, `archive/work/server-park-idle-slots/`,
`WORKLOG.md` 261025.006, PR #225.


### §59 -- A fused window reads a tensor the graph keeps alive for it, never an unpinned ancestor

A backend-side fusion is invisible to the graph allocator: the allocator frees a tensor after its *last
graph use*, which for a fused window can be a node the fusion then skips or a node that runs *before* the
window.  So "the fused kernel reads tensor X later than the allocator thinks X dies" is only sound when X
is guaranteed to stay live, and the only such guarantee in a ggml graph is `GGML_TENSOR_FLAG_OUTPUT`
(`ggml-alloc` never recycles a pinned tensor) or a tensor the window itself writes (which the alias checks
cover).

The qwen4exp combine+norm matcher violated that in r138 (issue #226).  The repeat-anchored window starts at
the `REPEAT`, so the chain's `scale1 -> sigmoid -> scale2` nodes all run standalone **before** it.  The
matcher read `inject = scale1->src[0]` — the raw GEMM output — whose last graph use is `scale1`, so the
allocator handed its buffer to the chain's own SIGMOID, and the fused kernel applied the whole scale chain
to a sigmoid output.  r138's #31028 merge had made `inject` a `SCALE` node (the `1/hc` moved into
`build_hc_mix`), which is what put a node between the pinned tensor and the read; a `ggml_set_output` on
the *handed* tensor was still there, one node too late.

**Rule.**  Read the tensor the emitter pins, with **identity** op params if the pinned tensor is the output
of a chain the matcher would otherwise fold (here: `inject = scale1`, `s1 = 1`, `b1 = 0` — bit-identical,
because the SCALE kernel already stored `s1*x + b1` and `hc_mul_rn(1,x) = x`, `hc_add_rn(x,0) = x`), and
**require the pin** (`flags & GGML_TENSOR_FLAG_OUTPUT`): an unpinned emitter must lose the fusion, not read
a recycled buffer.  An emitter that hands a `SCALE` to such a matcher is the shape that breaks — a new one
must either pin the tensor the matcher reads or not fuse at all.

The companion case is the reason the rule is stated per-matcher rather than globally: the **scale1-anchored
A-chain** matcher anchors *at* `scale1`, so that node is elided and `scale1->src[0]` really is live at the
window (and `add`/`mulg`, the only writers, are alias-checked).  Descending is correct there and wrong in
the repeat-anchored one; both are documented in place so a future "simplification" cannot unify them.

**Second half of the rule: reduce like the kernel you replace (r141).**  Reading the right tensor is not
enough — the fused combine also *computes the RMS norm*, and a norm's result depends on the summation
grouping.  `hc_combine_norm_f32` owned three strided scalar columns per lane, while `rms_norm_f32`'s **vec4**
path (taken for `n_embd % 4 == 0` with aligned operands) gives a lane the four columns `4*i .. 4*i+3` and
sums them as one `a*a + b*b + c*c + d*d`.  (**r145, issue #229:** which of the two groupings the chain takes
is now the *numerics selector* of §60 — the **strided** one by default, the vec4 one only under
`GGML_CUDA_RMS_NORM_VEC=1` — and `hc_combine_norm_f32` follows it.  The rule is "mirror the chain", not
"mirror the vec4 path".)  Different partials -> a different `block_reduce` input -> a
different `scale`.  Measured on Flash-Next GSQ against the unfused chain: **0.018523** mean KLD / 95.588 %
same-top-p (and 0.021338 / 95.882 % on the pre-r138 r135 build, so it had always been there), which is
outside this rulebook's 0.005 / 98 % limits even though PPL agreed to 4 dp.  With the kernel owning the
chain's columns: **−0.000000 / 100.000 %** on `-sm tensor` and **0.000000 / 100.000 %** on
`-sm layer -ts 59,41`, and the *greedy text* of the fused arm is byte-identical to
`GGML_CUDA_DISABLE_HC_COMB=1` (182 chars sha `a99ccbd97ecb`).  Ruled out on the way (all measured, all
unchanged): `GGML_CUDA_MMB=0`, `GGML_CUDA_BF16_SRC1_CACHE=0` (so not the BF16 src1 cache write) and
`LLAMA_HC_PIN_BLOCK_OUT=1` (so not the `block_out` pool copy); the harness control (chain vs chain with the
*other* hc fusion disabled) is exactly 0.000000 / 100.000 %.

**Corollary — the pin is per tensor object (r141).**  A pin on a *view* does not keep the view's root alive,
so the repeat-anchored matcher now requires `inject->view_src == nullptr` as well as the pin.  That is what
lets a **merged-tensor** (post-#31028) GGUF take the fusion: its `inject` is a strided view of the merged
GEMM, so `build_hc_mix` materialises it with `ggml_cont` (a plain row copy) and `build_hc_combine` pins the
result.  The matcher no longer requires `inject` to be a `SCALE` at all — it reads `sigm->src[0]` with
identity op params, which is what a separate-tensor GGUF hands it too (the r140 shape).

**Evidence (r140 + r141, gfx1201, ROCm 10.1.0).**  Qwen3.8-Flash-Next GSQ-RCO-IQ3_XXS (the reporter's
pre-#31028 GGUF), `HC_POST=0`: r139 PPL **2690.39** (chunks 1711.7 / 2932.5 / 3267.4 / 2690.4) -> r140
**2.4538** -> r141 **2.4546** (= the chain's exact value); the default `DSV4_HC_POST` arm is 2.4409
throughout; the 24-chunk run gives **2.4159** (r140) / **2.4238** (r141, both arms) against the reporter's
validated pin (2.4209) and r137 (2.4202); the r140 fix was **bit-identical to that pin's patch on 24/24
chunk PPLs**.  `-sm layer -ts 59,41`: 2.4850 / 3.7629 / 2.9181 / 2.4235 (r140) and 2.4395 fused == chain
(r141); their broken arm was 1781.5 / 3135.2.  6/7 coherence cells byte-identical across the r139, r140 and
r141 builds, the 7th being the chain arm each release changes.
Record: `WORKLOG.md` 261025.014 and 261025.015, `patches/README.md` (block 11).


### §60 -- A reduction-order change is a new numerics reference; every fused norm kernel must reduce like the chain, and the delivery's reference is the pre-#29720 strided one

**Claim.**  An RMS-norm sum's *partial-sum grouping* decides its last bits.  Every layer then moves by
~1 ulp, and greedy MTP draft acceptance is a **discrete** function of that: a payload-dependent amount of
acceptance can move with no error anywhere.  So a grouping change is not a neutral refactor and not a free
prefill optimization — it is a new numerics reference, and the reference is whatever the fused kernels
that stand in for `rms_norm_f32` reproduce.

**What happened (r145, issue #229).**  The r127 re-base adopted upstream
[#29720](https://github.com/ggml-org/llama.cpp/pull/29720): `rms_norm_f32` gained a float4 path (a lane
owns the four contiguous columns `4*i .. 4*i+3` of `i = tid + k*block_size`) and took it unconditionally
for `ncols % 4 == 0` with 16-byte-aligned operands.  The reporter (1x R9700, Windows, ROCm 10.0, graphs
off, Qwen3.8-27B UD-Q5_K_XL, `draft-mtp n3`) bisected a **−7..−9 % tg** regression to exactly that commit
and measured that `GGML_CUDA_RMS_NORM_VEC=0` restores r100/r126 bit-for-bit.  Reproduced here in kind:
Qwen3.8-27B UD-Q4_K_XL, 1 GPU, greedy, `draft-mtp n3`, 5.2 k prompt — with the SSM decode fusions
disabled (the reporter's carry) **acceptance 0.611 (58.0 t/s) with float4 vs 0.792 (68.7 t/s) with the
strided reduction**, deterministic over repeats; on the default (fusions on) arm the same change is
*positive* here (+0.7..+1.3 points of acceptance, 94k and 5k contexts), and at 94k context 58.9 vs
58.5 t/s.  Both arms are pure in every case (`plain == draft-mtp` byte-identical, so this is **not** a
draft/verify inconsistency), PPL agrees to 4 dp, and the prefill-logit gate passes at 0.0005 mean KLD —
i.e. the sign is a coin flip on the payload, which is exactly why it cannot be a shipping default.

**What the re-base also did, silently.**  The delivery's decode-band norm kernels are documented
*bit-identical* to `rms_norm_f32`: the rms_norm→Q8_1 fold (`rms_norm_q8_1_f32`), `gdn-conv.cu`'s folded
q/k norms, the GDN norm gate, `rms_norm_scale_pair_f32`, `rms_norm_mul_rope_f32`, and
`hc_combine_norm_f32` (which r141 had moved *onto* the vec4 grouping).  With float4 as the default, all of
them became **approximate**: the decode-shape probe
`scripts/gate-prefill-logits.sh --ab "<fusion env>" --ub 1` measured **0.000511** (fold) / **0.000529**
(GDN norm pair) / **0.000441** (`rms_norm_mul_rope`) mean KLD where the contract says 0 — and exactly
**−0.000000 / 100.000 %** for every one of them under the strided reduction.  r141
(`WORKLOG` 261025.014/015) fixed one of these by aligning it to vec4; the rest were never noticed because
no gate compares a fusion against its chain at decode widths.

**Rule.**
1. **One reduction, and it is the strided one by default.**  `ggml_cuda_rms_norm_vec()` (`common.cuh`)
   is the single selector; `rms_norm_f32`'s dispatch and **every** kernel documented bit-identical to it
   consult it (`hc_combine_norm_f32` takes it as a launch argument).  Never hard-code the vec4 grouping.
2. **A new/changed fused norm kernel owes the `--ub 1` KLD probe** against its own disable switch
   (`GGML_CUDA_DISABLE_NORM_Q8_1`, `GGML_CUDA_DISABLE_NORM_PAIR`, `GGML_CUDA_DISABLE_CONV_FUSION`,
   `GGML_CUDA_FUSE_GDN_NORM_GATE=0`, `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE`, `GGML_CUDA_DISABLE_HC_COMB`)
   and must read **0.000000 / 100.000 %**; the same-seed greedy text is the weak instrument here.
3. **`GGML_CUDA_RMS_NORM_VEC=1` is an A/B arm**, not a shipping mode: it keeps every fusion *on* but makes
   them approximate (0.0005 KLD), and it re-rolls the acceptance lottery.  Anything measured with it on is
   not comparable to a recorded MTP/acceptance number.
4. **Do not "fix" an MTP acceptance report by touching the draft.**  Measure the target's numerics first:
   if `plain == draft-mtp` holds, the draft/verify band is consistent and the difference is the reduction
   order (§59, second half), not the draft.

**Evidence.**  The A/B tables (1 GPU and 3 GPU tensor prefill, 35B-A3B, the 94k-context MTP arm, the
SSM-fusions-off arm, and the per-fusion KLD probes) are in
[`benchmarks/2026-10-26-rms-norm-float4-ab.md`](benchmarks/2026-10-26-rms-norm-float4-ab.md); the release
record is `WORKLOG.md` 261026.002 (r145); the switch is `ENVIRONMENT.md` §7.5.
### §61 -- A fused op must reproduce the model's own chain — an intermediate's scale convention belongs to the emitter, and a self-consistent op pair can still be wrong

**Claim.**  The oracle of a fused op is the **unfused chain the model emits** — not the op's partner, not
its own CPU reference, and not another build of itself.  An intermediate tensor's *scale convention* (where
a `1/hc` or a `2.0` lives) is a property of the **emitter**, and when a model has two interchangeable
emitters for the same intermediate, each fused consumer must match the *chain*, not the sibling op.  A pair
of fused ops written hand in hand stays mutually consistent and can still both be wrong.

**What happened (r148, issue #228).**  qwen4exp's hyper-connection stream is steered by a per-stream
**scatter logit**: `build_hc_mix` emits it and `build_hc_combine` turns it into `w = 2*sigmoid(.)`.  Each
half exists twice — a ggml chain (upstream's model definition) and a fused op (`GGML_OP_HC_MIX` /
`GGML_OP_HC_COMBINE`) — and `LLAMA_FUSED_HC_MIX` / `LLAMA_FUSED_HC_COMBINE` select them **independently**,
so there are four arms.  The v14 model (base `097cf33d9`) emitted `inject = w_inject^T xn` **raw** and put
the `1/hc` in the combine (`sigmoid(scale(inject, 1/hc))`); the fused pair was written against exactly that,
publishing a raw tail from the mixer and dividing by `hc` in the combine.  The `6050b467c` re-base
**inverted the convention** (the mixer now emits `inject/hc`, the combine applies no scale — same final
numerics, different intermediate) and nothing re-derived the ops: the fused pair kept the v14 convention
and became chain-inconsistent.  Three consequences, all silent:

* chain mixer + fused combine (every hc-down quant that is not Q8_0/BF16 — Q5_K, IQ4_NL, IQ4_XS, Q4_K, …)
  divided by `hc` **twice**: `w` collapses towards 1 instead of sweeping `(0, 2)`, and a Q5_K_M requant
  degenerates into a greedy repetition loop (issue #228: kl ~15 vs the v14 engine).
* fused mixer + chain combine (any Q8_0/BF16-hc model run with the documented `LLAMA_FUSED_HC_COMBINE=0`
  kill-switch) applied **no** scale at all: the BF16-hc model produced an **empty** generation.
* fused + fused (the arm every dense/GSQ validation ran) was right, which is why ten releases passed.

**Why no gate saw it.**  `test-backend-ops -o HC_MIX,HC_COMBINE` compares each op against a CPU reference
that had encoded the same convention, so the set agreed with itself — the same lesson as §52, one level up.
Greedy coherence survives a wrong combine while the two halves still match each other (`w` is wrong but
shared by every stream, so the text stays plausible), and the arms were never *crossed*: every recorded
qwen4exp number came from one arm, so each A/B compared the broken arm with itself.  The knobs that *look*
like they cross the arms do not: `GGML_CUDA_DISABLE_HC_FUSION` / `_HC_MIX` / `_HC_COMB` disable matchers
over the **unfused chain**, and are therefore inert on a default build (measured: identical text on a
fused-mixer model).  Their "A/B is byte-identical" entries in the r140..r147 records are true but vacuous,
which is exactly what the reporter's own `GGML_CUDA_DISABLE_HC_COMB=1` arm showed ("changes nothing").

**Rule.**
1. **A fused op's reference is the model's chain.**  When you change an op or a chain, cross the arms:
   `scripts/gate-hc-fusions.sh MODEL.gguf` runs all four mix/combine combinations (plus the CUDA-side
   chain fusions and the DSV4 HC pre/post pair) and fails unless every hard arm is **byte-identical to the
   unfused chain**.  Run it once per hc-down weight class — Q8_0/BF16 `hc_attn_down` takes the fused mixer,
   any other quant takes the chain (`--types MODEL.gguf` prints the class).  Two arms of a two-way switch
   is not coverage.
2. **An intermediate's scale belongs to the emitter.**  If two emitters (chain and fused) can produce the
   same intermediate, the convention is stated once and both implement it; a `/hc` (or any scalar) inside a
   *consumer* is a convention claim that a re-base can invalidate.  `GGML_OP_HC_MIX`'s inject tail now
   carries the scaled logit exactly as `build_hc_mix` does, and `GGML_OP_HC_COMBINE` divides by nothing.
3. **A kill-switch must be numerically inert, and an A/B knob must actually change the arm.**  `LLAMA_FUSED_*=0`
   exists for A/B and bisection; an arm that changes the *result* invalidates every A/B taken with it and
   can hide a defect — it hid this one (the reporter's rescue arm *was* the reference).  The converse bit
   too: the three CUDA-side matcher knobs (`GGML_CUDA_DISABLE_HC_FUSION`, `_HC_MIX`, `_HC_COMB`) look for
   patterns in the **unfused chain**, so on a default build (fused ops on) they are **inert** and a
   "byte-identical" A/B with them proves nothing (measured: all three produce the default arm's text on a
   fused-mixer model).  Verify that the knob changes the graph or the kernel under the arm you compare it
   on before recording it as evidence.  When an arm is approximate by design (the DSV4 `HC_POST` op on one
   device, §59), the gate reports it, never passes it silently.
4. **Re-base checklist item:** a re-base that touches a model definition can invert an intermediate's
   convention.  Diff the model's `build_*` emitters against the fused ops' contract and run the
   fusion-equivalence gate; "block 11 owns the HC ops" is not enough.

**Evidence.**  `scripts/gate-hc-fusions.sh` on three GGUFs of both classes — Flash-Next Q5_K_M (chain
mixer), IQ4_NL (chain mixer), GSQ-RCO IQ3_XXS (BF16, fused mixer): all 6 hard arms byte-identical to the
chain (Q5_K 445 chars `c8e579144ff1`; BF16-hc 449 chars `6a102dbc62d6`), while the pre-fix build fails
every non-Q8_0/BF16 arm and produces an empty generation on the BF16-hc model's combine-off arm.  The arm
the delivery had right does not move: the BF16-hc default's greedy text is **bit-identical before vs after**
(`246463129328`).  The fusions keep paying: `llama-bench -p 512 -n 64` on GSQ-IQ3_XXS **34.78 t/s** default
vs **31.73** with both arms off (+8.7 %), prefill flat; IQ4_NL 34.49/34.35 and Q5_K 33.80/33.57 (parity,
the chain-mixer class).  Release record: `WORKLOG.md` 261026.006 (r148); op contract:
`patches/README.md` (block 11); switch catalogue: `ENVIRONMENT.md` §7.4.

**Closed by re-baseline at r157 (measured at r156).**  On the IQ4_NL class the `dsv4_pre_chain` arm does
**not** match the chain, and never has in this line: 444 vs 453 chars, identical on an r148/r154 build
(the r149 record logged it first).  At the logits the default fused `GGML_OP_DSV4_HC_PRE` reads **mean KLD
0.021542 / same-top-p 95.814 %** against its own F32 chain (`gate-prefill-logits.sh --ab
"LLAMA_FUSED_DSV4_HC_PRE=0"`, the standard 40-chunk point; a 10-chunk exploratory run read 0.027936 /
96.471 %), i.e. the **same order the combine+norm fusion has always shown** (0.0185-0.0213 / 95.6 %,
`WORKLOG.md` 261025.014).  That is a **numeric re-baseline, not a near-tie and not a defect**:
`dsv4_hc_pre_f32`'s `wbf16`/`xbf16` arms read the HC16 bf16 producer slots by design, so
`LLAMA_FUSED_DSV4_HC_PRE=0` swaps the producer precision as well as the fusion (`GGML_CUDA_MMB_HC16=0`
changes neither arm).  r157 therefore: **`dsv4_pre_chain` is a reported arm like `dsv4_post_op`**; the
Flash-Next IQ4_NL prefill-logit base is re-recorded from the shipped default (`base_ppl` **2.7527**, and the
pre-r156 build reproduces it at 0.000000 / 100.000 %, i.e. the class is stable and the base is a reference
rather than a snapshot of one build); and the hard oracle is unchanged -- the four mix/combine arms plus the
CUDA-side chain fusions still match byte for byte on **both** hc classes (IQ4_NL 444 chars `7ea23f309705`,
BF16-hc 449 chars `6a102dbc62d6`).  `WORKLOG.md` 261027.007/261027.008.
### §66 -- A copy that only exists to keep a tensor alive is a graph node, not an output (r163, issue #240)

**Claim.**  F1 (the DFlash device-resident layer inputs) is pure only because the store holds **exactly** the
bytes the post-graph extraction copied there, at the **same rows**.  Moving that copy inside the graph is a
relocation, not a numerics change: `GGML_OP_SET_ROWS` writes a value byte for byte, so the draft's
`ggml_get_rows` source is identical to the one the post-graph device copy produced.

**Rule.**  Three things must hold together, and any one of them missing is a silent wrong answer on the
*next* step rather than a crash:

1. **The row index is a graph input, never a baked offset.**  The store is indexed by the token's row in the
   decode batch, which changes per ubatch; a `ggml_cpy` into a view at a build-time offset would bake it into
   a captured HIP graph and a replay would write the wrong rows (the r149 key lesson, §62).  The indices ride
   in an `I32 [n_tokens]` input tensor filled per ubatch — the same shape the draft's own gather reads with.
2. **The copy's position in the node order is the whole point.**  `ggml-alloc` frees a tensor after its last
   consumer (§"the copy into a view of preallocated memory" idiom), so the copy must be expanded *after* the
   layer that reads the input and *before* the next layer.  An arch that records its layer inputs through
   `llm_graph_result::set_layer_inp()` gets that placement automatically (the next layer's recording expands
   the layer's nodes, then emits the copy); an arch that does not is still correct (the copy lands at the end
   of the graph) but keeps the memory cost it was supposed to remove.
3. **A graph built without an armed store must not be reused once the stores exist.**  The store changes the
   graph's topology (an output becomes a `SET_ROWS`), so `llm_graph_params::allow_reuse()` compares
   `cparams.lf_layer_inp_dev_store` (armed vs not) exactly as it already compares `lf_dev_inject`.
4. **An embedding ubatch still reads the host rows.**  A token row addresses its store row; an image/audio row
   does not (it has no token id), so the draft takes its host path there and the target must fill the host
   rows for that ubatch as well.  The rows are reserved on demand, so only a multimodal run pays for them.

**Evidence.**  27B UD-Q4_K_XL + DFlash2-Q4_K_M, `-ub 2048 -b 2048`, q8_0 KV, `prompts/prose-rdna-boosts.txt`
(5246 tokens, so the target's decode is 3 ubatches and the store row offsets 0/2048/4096 are all exercised).
Greedy text is **byte-identical** between the in-graph store, `GGML_LF_DFLASH_DEV_GRAPH=0` (post-graph copy)
and `GGML_LF_DFLASH_DEV=0` (host path) — `950 chars sha=8016577f2ed8` on 1 GPU and on a 2-GPU `-sm layer`
arm (text alone is *not* the oracle here: the draft only proposes, so garbage rows would still verify to the
same text).  The oracle is the **draft**: acceptance and the per-position acceptance rates are identical to
the fifth decimal (`0.44337 (137/309)`, `acc rate/pos = (0.806, 0.516, 0.387, 0.306, 0.194)`) in every arm,
and on a 2-slot `--kv-unified` server the concurrent pair is identical too (`0.74163 (155/209)`, both
responses 688/709 chars, all three arms).  A mixed row mapping would show up as a collapsed acceptance, which
is how the F1 multi-sequence mapping was validated in the first place.

**Cost.**  Five extra `SET_ROWS` launches per target graph (decode: 5 rows of `n_embd`): decode is flat
within noise (106.2 / 106.7 / 106.2 / 106.7 t/s over four alternating `-n 1000` arms), and the win is
**132 MiB of compute buffer** plus **200 MiB of pinned host rows** that F1 never read.


### §68 -- A re-base that inherits an upstream association-order change is a re-baseline, not a purity break (r170, issue #228 follow-up)

**What was reported.**  A downstream consumer on gfx1151 (Strix Halo) runs an *output-identity* gate across
base bumps: the logit fingerprint of each release against its predecessor had stayed within 0.007 for eight
cutovers, and r150's first-token top-5 KL against their r120 reference was **0.497** with greedy text moving
within a few dozen tokens on 7/7 prompts.  They bisected it kernel-directory → file → commit and landed on
upstream `2744e48d1` (llama.cpp #30517, "faster MMVQ for low-bit types"), which sets
`VDR_Q6_K_Q8_1_MMVQ` 1 → 2 and rewrites the q6_K `mmvq` kernel.  Their payload's 13 Q6_K tensors (the
lm_head and 12 attention outputs) are exactly what that changes.  Nothing in the delivery's own kernel work
was implicated: with the r150 `vecdotq.cuh` reverted and every other r150 file kept, their arm reads r150
bit-for-bit, and the mmq tile-loader rework alone is numerically pure.

**Why it is not a purity break.**  Every purity claim in this file is a **within-build** invariance: a
verify batch reproduces single-token decode (§6), a fused op reproduces the chain it replaces (§9, §59,
§61), one arm spans the whole band (§19, §24, §25, §26), a replayed graph computes what its capture
computed (§62).  None of them is "text matches the previous release", and §5 says why: bit-identity is a
**reproducibility** requirement, not a correctness one.  A re-base that adopts an upstream reduction-order
change re-baselines every quantized dot that change touches.  The consumer's gate is a legitimate
reproducibility requirement *of theirs*; it is not a contract this delivery ever made.

**The measurement (gfx1151, 2026-10-28, ROCm 10.1.0, `GGML_HIP_NO_VMM=ON`).**  Payload: Qwen3.8 Flash-Next
Q5_K (339 Q5_K / 199 Q8_0 / 94 IQ3_S / 44 IQ4_NL / 2 IQ4_XS / **13 Q6_K** — the same census class the
report filed).  Two builds from the same tree, differing only in the macro:

| arm | tg128 | batched B=1 | B=4 | B=8 | tg128 @ d16384 |
|---|---|---|---|---|---|
| VDR=2 (shipped default) | 33.86 / 33.86 | 33.50 / 33.49 | 81.76 / 81.72 | **104.14 / 103.98** | 31.86 |
| VDR=1 (pre-#30517 order) | 33.83 / 33.81 | 33.48 / 33.47 | 81.75 / 81.55 | 102.94 / 102.60 | 31.86 |

So the wide chunk is neutral at B=1/B=4 and at depth, **+1.2 % at batched B=8**: it stays on (default-on
policy, and it is upstream's default anyway).  Greedy text moves between the two builds, exactly as the
association-order rule predicts (128 tokens, one prompt: `d47f08557f75` vs `bf8ddd5271d1`).  Width purity is
untouched at VDR=2: `plain`, `draft-mtp --spec-draft-n-max 3` and `--spec-draft-n-max 7` all produce the
identical 1302-char stream (`115141867744`) on the Q5_K target with the shared Q8_0 MTP head — the property
that actually matters, and the one the reporter's gate cannot see.

**Doctrine.**

1. **Classify a cross-release move by its carrier.**  A reduction-order carrier (a VDR value, a `nwarps`
   choice, a tile/atom selection, a norm grouping) is a *re-baseline*: the delivered invariants still hold
   within the build.  A scale/convention/identity carrier (§59, §61) is a *defect*, and from the outside the
   two look the same — a different, still-coherent greedy stream.  Bisect to the file, then read the file:
   here `vecdotq.cuh`'s define, not a delivery kernel.
2. **The oracle for a re-baseline is width purity + a distribution check, never a cross-release text hash.**
   `plain == draft-mtp` at `n_max <= 7` on the affected payload class, plus PPL unchanged, decides it.
3. **A consumer's output-identity gate is answered with a build-time escape, not a runtime arm.**  VDR is a
   `constexpr` that the host's K-split arithmetic also reads (`get_vdr_mmvq` → `blocks_per_iter`, `kqs`,
   `calc_nwarps`), so a runtime knob means instantiating the Q6_K `mmvq` kernel family twice and adding a
   third numerics arm to every future gate — for a delta that is +1.2 % at one width.  The flag
   (`-DGGML_HIP_Q6K_DENSE_VDR1`) gates the macro and the device selection together, so the host partition
   and the kernel cannot disagree; a knob that changed only one of them is a wrong answer, not a different
   one.
4. **Re-base checklist item:** diff `vecdotq.cuh`'s `VDR_*` defines and the `get_vec_dot_q_cuda()` dispatch
   against the previous base.  An upstream default flip re-scopes the delivery's own per-arch gate silently,
   turning a block-07 tuning decision into an inherited one — which is how this report became possible.


---
### §48 -- A draft-side vocabulary restriction cannot change an accepted token — the schedule can change the text

**Claim.**  `--spec-draft-mtp-vocab N` (block 13; block 12 before the r109 split) makes the MTP draft score only the LM-head rows for token ids
`< N` (+ control/user ids) and scatters them into a full-vocab row of `-inf`.  A speculative token is accepted
only if the *target* agrees, and the target still scores the full vocabulary, so the trim can only **lose
proposals** — it can never change an accepted token.  The `-inf` also guarantees the draft sampler cannot pick
a row outside the subset.

**Rule (the part that is easy to get wrong).**  Do **not** gate this feature on "trim == no-trim greedy bytes"
across arbitrary configurations.  The trim changes acceptance, acceptance changes the adaptive controller's
draft depth, and depth changes the verify batch width — and verify-width numerics can flip a near-tie.  That is
pre-existing behaviour of the delivery, measured with **no campaign code applied**: the r106 build gives three
different greedy texts on `prompts/code-reasoning-mixed.txt` (512 tok, `-sm tensor`) for `draft-mtp` n-max 3
(`92010829c9f6`), n-max 9 (`ec1a197011c4`) and `draft-mtp-adaptive` n-max 9 (`5612aa128fc3`); under `-sm layer`
n3 (`348b4bb78c8a`) and n9 (`921afe655051`) differ too.

**Why that is not a defect.**  §11 fixes the purity promise at `n_max <= 7` (an 8-token verify batch: the FA
tile->WMMA switch is at `Q->ne[1] > 8` and `mmvq`->MMQ at `ncols == 8`, and beyond it the reduction order
changes).  Inside the band, all widths take the same kernels, so the schedule does not matter.

**Evidence (gfx1201, 2x R9700, 2026-10-22, `-n 3000`).**

| run | arms | greedy text |
|---|---|---|
| prose, `draft-mtp` n-max 7 | `--spec-type none`, trim 0, trim 64000, trim 98304 | **all `fa4867aa074e`** |
| code, `draft-mtp` n-max 7 | `--spec-type none`, trim 98304 | **both `f7256a7b1380`** |
| prose, `draft-mtp-adaptive,ngram-mod` n-max 9 (outside the band) | trim 0 vs trim 64000 | agree for 1952 chars, then a near-tie flips on a code/reasoning phase switch |

So: gate the trim at `n_max <= 7` (text identity against plain decode is then a valid, strong check), or above
it on acceptance + MTP-vs-plain — exactly what `benchmarks/mtp-adaptive-methodology.md` rule 4 already says for
any speculator.  Recommended configuration for the block: `--spec-draft-n-max 7 --spec-draft-mtp-vocab 98304`
(`--spec-draft-n-start` at its default 3 — the *initial* depth, not a ceiling).

**Device count (r109 validation, 2026-10-22) — purity is unaffected, the *throughput* is not.**  The subset
head is *mirrored* on every device under `-sm tensor` while the untrimmed head is *split*, so the trim pays
only when `N * n_devices < n_vocab`.  27B (vocab 248320), four axes at `-n 3000`: +1.4 % aggregate at 2 GPU
(share 124160), **-2.2 % at 3 GPU** (share 82773), and no N pays at 3 GPU (64000 loses acceptance,
0.68878 -> 0.67260).  Purity holds in all of them — 3-GPU shallow `none` / trim 0 / trim 98304 all
`c5421068b4e0`, and all `12048e399854` at depth 16384 — so this is a recommendation-scope note, not a
weakening of the claim.  Record: `../archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md`.


### §50 -- A draft length is a verify width *and* a memory-tier parameter

**Claim.**  On a config with host-resident MoE experts served by the expert-cache arena, a verify batch
wider than the arena's decode/verify band leaves the arena **entirely**: `moe_cache_update_host()` returns
false for `n_tok > MOE_EXPERT_CACHE_MAX_TOK` (`moe_cache_max_tok_dev()`; 16 tokens on RDNA4, 8 on
RDNA3/NVIDIA), so the routed `MUL_MAT_ID` is served by the per-op upload path and its experts are streamed
from the pinned host master on **every** step.  The transition is a cliff, not a slope, and it is one row
wide.

**Rule.**  Every speculator's draft length is a verify width (`n_tok = n_draft + 1`).  With the arena
active, keep every draft **inside** the band, and never generalise a wide-draft throughput number from a
run whose experts are not served through the arena:

* `--spec-draft-n-max` is clamped to 15 by `LLAMA_SPEC_DRAFT_N_MAX_CLAMP` — a 16-row verify, i.e. exactly
  the RDNA4 band (the bound itself is correctness: §27's `n_rs_batch`).
* The **draftless** n-gram speculators had no width bound at all (`ngram-mod` `n_max` 64 / `n_min` 48 by
  default), so `common` now clamps an enabled n-gram speculator to `band - 1` once the model (and the
  cache preflight's decision) is known, lowering `n_min` with it and refreshing `cparams.n_rs_batch`
  (r123, block 09; `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` is the A/B).  A deep draft also cost ~5 GiB of
  arena, because the slab's narrow verify floor is measured at `n_rs_batch` (32938 -> 27814 MiB).
* A model with **no active arena** — fully device-resident, or the CPU expert path
  (`MOE_EXPERT_CACHE_MIB=0`) — reports no band and is never clamped.  **A band is answered only where
  host-resident experts exist (r176, TODO #117):** `llama_model_moe_cache_band()` considers
  `model->moe_host_expert_bytes` alone, because `moe_cache_enabled()` is true by default (AUTO) even with no
  registered table, and the pre-r176 `model->devices` fallback made a **dense** model on a single GPU or any
  `-sm layer` run answer 16 — which clamped `ngram-mod --spec-ngram-mod-n-max 64` to 15 and bounded every
  multi-slot MTP/DFlash alignment to `band / n_drafting - 1` rows on a model with no routed expert at all.
  Measured on the dense 27B, 1 GPU, `prompts/recall.txt` (`-n 200`): **142.1 -> 186.8 t/s (+31 %)** with
  byte-identical text, the pre-fix `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` control reading 187.9.  The same holds for a host-expert
  config whose movable-boundary slab is armed and then **declines** (a card too small for the work buffer +
  `GGML_CUDA_SLAB_MIN_ARENA_MIB` + the ring hole): it runs the same streaming path, where the wide draft is
  the win (2x R9700: 67.6 wide vs 52.2 capped t/s).  **Since r129 that case is handled**: the clamp is
  applied *after* the context exists, so it sees the arena's real state rather than the model-load `armed`
  flag (issue #214's reporter class — slab armed, then declined, then streamed — kept its wide draft; the
  TODO #96 defect).  The deferral is why `cparams.n_rs_batch` is no longer refreshed by the clamp: it stays
  the **unclamped** safety bound (the widest batch a partial rollback can land in, floored into the
  chunked-GDN threshold), while the slab's narrow (decode/verify) reserve takes the band-limited width
  (`llama_context::verify_batch_width()`) so a live arena is still sized for the clamped draft (measured
  unchanged: narrow floor 1024/1280 MiB and `identity fast path ON` 142/142 with and without the
  speculator).  A decline that happens *after* the context exists (an OOM-yield arena release) still
  leaves the draft clamped: `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` is the escape.

**Evidence (gfx1201, 2x R9700, Flash-Next IQ4_NL + MTP head, `-sm tensor -ncmoe 36`, `-c 16384`,
`prompts/recall.txt`, `-n 2000 --temp 0 --seed 42`, `draft-mtp,ngram-mod` with the ngram cap swept).**

| `--spec-ngram-mod-n-max` | widest verify | gen t/s | arena hit | counted cache `reaches` |
|---:|---:|---:|---:|---:|
| 12 | 13 | 154.8 | 0.977 | 1086540 |
| **15** | **16 (in band)** | **159.2** | 0.978 | 1093862 |
| **16** | **17** | **55.0** | 0.898 | 129660 |
| 32 | 33 | 48.1 | 0.893 | 103740 |
| default (64/48) | 65 | 74.1 | 0.893 | 103740 |

The README's own `draft-mtp-adaptive,ngram-mod --spec-draft-n-max 9` recommendation measured **73.9 ->
168.3 t/s** with the clamp (+128 %), and the plain-MTP baseline is untouched (101.4 -> 101.0 t/s, no clamp).

**Why the same combination is fine elsewhere (do not "fix" it globally).**  Device-resident (3 GPU,
`-ncmoe 0`): MTP n3 144.7 -> ngram default **378.7** vs 244.6 clamped — the wide draft is the product.
CPU-compute experts (`MOE_EXPERT_CACHE_MIB=0`): 38.5 -> 67.6 (wide) vs 52.2 clamps — the weights are
local, so extra rows cost compute and the wide batch amortises the per-token overhead.  That is upstream's
shape, and the issue-#214 reporter's update confirms it: all three of his r120 runs logged the slab decline
and no `moe_cache_report`, i.e. his box was in the no-arena streaming class, and his decode already matched
upstream (the remaining regression there was deep-context **prefill** depth scaling, TODO #97 — not this band
and not a decode symptom; that closed at r129, RDNA4 in r126 and RDNA2 on the reporter's own sweep).  On an arena config the same wide flags halve decode.  Purity is unaffected in every arm
(the clamp changes the verify *schedule*, §48): `plain` / `draft-mtp n3` / ngram hand-capped /
ngram auto-clamped all hash `9d914f47aebb` with `ROCBLAS_USE_HIPBLASLT=0` (§44).

Record: `archive/work/ngram-mod-host-experts/EVIDENCE-2026-10-24.md`.


### §52 -- An op with no test case has no oracle — and a default-on fusion needs the pair covered, not just the op

**Claim.**  `GGML_OP_HC_COMBINE` (the qwen4exp hyper-connection residual combine,
`out[e,c,t] = residual[e,c,t] + block_out[e,t] * 2*sigmoid(inject[c,t]/hc)`) shipped for four releases with
**no `test-backend-ops` case at all**: `-o HC_COMBINE` ran **0/0**, so the op was never compared to a
reference.  Its sibling `HC_MIX` had 30 cases throughout, which is what made the gap invisible — the file
"has HC coverage" if you only look at the summary line.  A wrong combine (an inverted or mis-centred
`2*sigmoid` weight, a `stream`/`token` index swap, an in-place residual read) is a *uniform* corruption of
the residual stream: it changes no kernel's shape, so same-seed text, MTP acceptance and the width probe all
still agree with themselves, and on a MoE-bound decode the FA/mixer kernel is not the wall clock (§51).
`HC_MIX`'s 30 cases cannot see it either: its `x` is a random tensor in the test, so a combine that produced
garbage would just feed a different — equally random — input.

**Rule.**  Every op a block adds owes a `test-backend-ops` case whose reference is the *unfused* chain (for a
fused op, the CPU backend's implementation of the same op is that oracle).  Cover the shapes the model
actually runs (both broadcast forms, the `hc` values in use, the band widths), and when a **fusion** of two
ops is default-on, add a whole-graph case for the pair: `run_whole_graph()` reaches the fusion dispatcher,
a single-op case cannot.  Name the pair case after both ops (`HC_COMBINE_MIX`) so it is selectable, and run
both arms (`=0` kills the fusion) so the fused and unfused paths are each compared to the same CPU
reference.

**Evidence (r128, gfx1201).**  `test_hc_combine`: 20 cases (`n_embd` 512/2560, `hc` 2/4/8, `nt`
1/2/3/5/8/16, `block_out`+`inject` broadcast and per-token) — **20/20**.  `test_hc_combine_mix`: 6 whole-graph
cases at the Flash-Next shapes (n_embd 2560, hc 4, low rank 320, BF16 mixer), `nt` 1/2/3/5/8 plus the
`w_inject`-less head call — **6/6** with the fusion on and 6/6 with `GGML_CUDA_FUSE_HC_COMBINE_MIX=0`; a
temporary instrument confirmed `ggml_cuda_hc_combine_mix_try_fuse` fires on the default arm and not on the
control.  `test-llama-archs -a qwen4exp` stays green (GPU NMSE 4.42e-14, mixed batch 2.07e-14), and nothing
in the delivery's text moved: dense 27B `plain == draft-mtp` sha `0cd2d424a564` (identical to r127),
35B-A3B `plain == draft-mtp` identical, Flash-Next GSQ `plain == draft-mtp` identical, prefill-logit KLD
0.000000 / 99.980 %.

Record: `TODO.md` #100 (closed), `patches/README.md` block 11, `MANIFESTS.md` (per-block row 11),
`WORKLOG.md` 261024.009.


### §53 -- A draft with a window on every layer never reads the prompt rows before the window, so skipping their ingest is exact

**Claim.**  A DFlash draft whose every layer attends over a sliding window of `n_swa` positions cannot
reach a prompt row older than the last `n_swa` positions of the prompt: the attention mask is
`is_masked_swa(n_swa, swa_type, p0, p1)` and DFlash sets `LLAMA_SWA_TYPE_STANDARD` unconditionally
(`src/models/dflash.cpp`, both the per-layer pattern path and the DSpark DSV4 path), so a query at `p1`
masks every key at `p1 - p0 >= n_swa`.  The first generation row sits at `p1 = prompt_end`, so its window
starts exactly at `prompt_end - n_swa`.  The draft's KV cache is sized for the window (SWA cell mapping is
position-modular: a row at `pos` owns cell `pos % n_swa` plus a per-sequence offset, and the `n_ubatch`
slack), so ingesting only the tail writes the same cells the full ingest leaves behind.  A skipped row can
therefore change no draft logit, and the target verifies every drafted token anyway.

**Rule.**  Skip only the rows a *mask* proves unreachable, and prove it from the arch's own window type, not
from `n_swa` alone: a `LLAMA_SWA_TYPE_CHUNKED` window reaches back to the chunk start (up to `2*n_swa - 1`)
and a `SYMMETRIC` one half a window forward, so either would need a wider tail.  Require **every** layer of
the draft to be windowed (a mixed pattern keeps full prompt ingest), and reset the per-sequence threshold
when the first generation row is seen so a later position shift (a new prompt on the same slot, a
checkpoint rewind) can never skip rows.  The skip must happen before the target synchronization, or the
speed-up it exists for (not waiting on the target for a batch the draft will not consume) is not obtained.

**Evidence (r130, gfx1201).**  3x R9700 `-sm tensor -mg 0`, Qwen3.8-27B UD-Q4_K_XL + DFlash2-Q4_K_M,
`-fa on -ctk/ctv q8_0`.  Tail on vs off (`LLAMA_DFLASH_INGEST_TAIL=0`) at 6.1K / 26.5K / 88.6K tokens:
prefill 1683/1787/1395 vs 1556/1591/1271 t/s (+8/+12/+10 %; no-draft at 88.6K is 1415 t/s, so the draft's
share of the prefill falls from 10.2 % to 1.4 %), greedy text sha identical in every arm, draft acceptance
identical.  35B-A3B Q4_K_M + its DFlash2-Q8_0 (+14.6 % prefill at 6.1K) and a 2-slot unified run and
`-ub 512` arms: text and draft counts identical.  Coherence against the r129 build: 6/6 configs identical
(dense 4B, dense 27B, 27B MTP, 27B DFlash2, 35B-A3B DFlash2, 35B-A3B `-ncmoe`), and the prefill-logit gate
KLD 0.000000 / same-top-p 99.980 %.

Record: `patches/README.md` block 12, `MANIFESTS.md` (per-block row 12), `archive/work/dflash-ingest-tail/`,
`WORKLOG.md` 261024.011.


### §54 -- A fill of an input the graph may not consume is guarded on `buffer`, never on the pointer

**Claim.**  ggml-alloc allocates no buffer for an input tensor the executed graph never consumes, and
`llama_kv_cache::set_input_kq_mask` (and its siblings `set_input_k_idxs` / `set_input_v_idxs` /
`set_input_k_rot` / `set_input_v_rot`) open with
`GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))`.  So a *fill* of an unconsumed input is a hard
abort before the first token, not a no-op: `if (tensor) fill(tensor)` is the wrong test, and
`if (tensor && tensor->buffer) fill(tensor)` is the right one.  Block 05's derived kq form is exactly a
producer of unconsumed masks — it is registered with `allow_derived = true`, and at one KV stream its
per-cell/per-query inputs *replace* the packed `n_kv x n_q` mask, which is still created but has no
consumer.  `bailingmoe3` (18 KDA + 6 MLA layers) at `-np 1` aborted in
`llm_graph_input_mem_hybrid_k::set_input` because of it.

**Rule.**  Every fill of a graph-input tensor goes through the `tensor && tensor->buffer` guard, in the
idiom `llm_graph_input_attn_kv` established (and `llm_graph_input_kq_derived` applies to all three of its
tensors).  A new `allow_derived = true` builder or a new hybrid/DSA input class inherits the requirement;
`can_reuse` does **not** need one (`can_reuse_kq_mask()` reads `ne[]` only).  Do not "fix" the abort by
dropping the mask from the graph or by making the setter tolerant: the guard states the invariant, and the
packed mask still has to be filled for every batch that does consume it.

**Evidence (r131, gfx1201).**  r130 tree, `bailingmoe3` / Ling-3.0-tiny IQ4_NL, `-ngl 999 -c 32768 -np 1`:
`GGML_ASSERT(buffer)` rc 134 with the stack
`llm_graph_input_mem_hybrid_k::set_input -> llama_kv_cache::set_input_kq_mask -> ggml_backend_buffer_is_host`;
guarded, the same command runs and its greedy text is byte-identical to `-np 2`, to `-np 4` and to
`LLAMA_KQ_MASK_DERIVED=0` (md5 `d5a0e389`, recall prompt, 128 tokens, seed 42).  The derived path is
bit-identical to the packed path at the logits (`gate-prefill-logits.sh --ab "LLAMA_KQ_MASK_DERIVED=0"`
on that model: **0.000000 / 100.000 %**, 40 chunks x 512), `test-llama-archs -a bailingmoe3` is 2/2, and
every previously-working arm of the r130 build is byte-identical (dense 4B 1 GPU / 3 GPU tensor, 27B
UD-Q4_K_XL, 35B-A3B `-ncmoe 40`).  The `attn_k_dsa` pair (its `mla`/`lid` masks) is hardening: that builder
passes `allow_derived = false`, so nothing in the delivery leaves those unallocated today.

Record: `patches/README.md` (block 05), `MANIFESTS.md` (per-block row 05), `AGENTS.md` (the invariant),
`upstream/attn-k-null-mask-guard/README.md`, `WORKLOG.md` 261024.012, PR #222 / issue #221.

---


### §55 -- A quantize cache keyed on a root tensor must match the pointer it fills from, and validate its read range

**Claim.**  The verify-band Q8_1 cache (`ggml_backend_cuda_context::q8_1_cache_get`) is keyed by the
**root** tensor's identity, but its arena is filled from the caller's (possibly offset) **view** pointer, and
the entry records the key tensor's `data`.  Walking a view chain to the root without checking the pointer
therefore lets an offset view store its quantized rows under the root's identity: a later op with the same
root and the same canonical row layout (`q8_1_rows_canon`) hits them and computes with another tensor's
activation.  It also leaves the entry's stored pointer valid after the view's compute buffer has been
reallocated (graph replay / pool reuse), so the quantize kernel can read a stale pointer — a GPU fault
(`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on the reporter's RDNA2 decode) or silent stale reads.

**Rule.**  A key walk that crosses a view stops at the **first pointer change**:
`while (key->view_src && src->data == key->data) key = key->view_src;`.  Offset-free views keep sharing the
root's entry (the common case, unchanged); an offset view gets its own entry.  Before every
`quantize_row_q8_1_cuda` / fused gated-Q8_1 launch, `q8_1_src_range_ok()` checks the source's read extent
against the root buffer's allocation and **aborts with the layout logged** rather than launching: a source
that has moved inside its buffer is a stale-pointer defect, not a tuning matter.  A new site that fills the
cache (or a new fused writer that fills it for a matmul) carries both halves.

**Evidence (r132, gfx1201).**  Grafted from PR #218 (@slavap), whose V620 reproduced the fault on decode.
Not reachable on the maintainer's model set: a probe at the four `mmvq` sites counted **0 offset-view keys**
in 14336-60416 key walks per config across seven configs (dense 4B 1/3 GPU tensor, 27B UD-Q4_K_XL 3 GPU
tensor, 27B IQ3_S 3 GPU tensor, 35B-A3B `-ncmoe 40` and `-ncmoe 10`, Flash-Next GSQ 2 GPU tensor with MTP,
Flash-Next `-sm layer`, Ling IQ4_NL) — so every same-seed arm is byte-identical (12/12) and MTP acceptance is
identical on the four axes (1859/3416, 1420/1710, 2171/2483, 372/375); prefill-logit KLD 0.000000 on both
recorded points.  `test-backend-ops -o MUL_MAT` 2445/2445 and `-o MUL_MAT_ID` 931/931 with
`LLAMA_TEST_SEED` fixed (a random-data marginal `q4_1`/`q5_0` tolerance case fails identically on the base
build at seed 41 — ERR 0.000511759 on all four builds — which is why the seeded form is the comparison).

Record: `patches/README.md` (blocks 08/10), `MANIFESTS.md` (per-block rows 08/10), `archive/work/pr213-pr218-r132/`,
`WORKLOG.md` 261025.001, PR #218.

---


### §56 -- A per-arch dispatch decision keys on the running device, and the pass that consults it selects its own

**Claim.**  MMB's policy predicates (`mmb_enabled`, `mmb_cfg`, `mmb_cfg_dump_once`, `mmb_dense_flag`,
`mmb_wtype_mask`, `mmb_dense_tmask`) read `ggml_cuda_info().devices[0].cc` — the **primary's** cc.  On a
mixed-arch box that freezes one architecture's policy for every device, so a gfx1030 (RDNA2, no WMMA)
secondary behind a gfx1100 primary ran RDNA3-tuned MMB kernels: byte-exact zero output-projection logits
(PPL == vocab size) or garbled tail-layer GEMMs, depending on which device held the output projection.

**Rule.**  A per-arch decision keys on the device the kernel runs on —
`ggml_cuda_info().devices[ggml_cuda_get_device()].cc`, the idiom `mmq`/`mmvq`/`fattn` already use — and a
per-arch table is cached per architecture **class**, not once per process.  A pass that consults such a
predicate without launching (the `graph_optimize` marking half) selects its own context's device first:
under the Meta backend the pass is forwarded per child from meta `graph_compute`, so the current device is
still the previously computed one, and a marking decision made on the wrong arch can elide an F32 output
that the launch path then never fills.

**Evidence (r132, gfx1201).**  Grafted from PR #218 (@slavap), who measured the V620 arm at PPL 248320 ->
2.9844 and the XTX-primary/output-on-XTX garble (917.54) -> 2.9842 on a 2x XTX + V620 box.  Bit-for-bit
identical on a single-arch box by construction (the current device's cc *is* `devices[0].cc`), validated
here: 12/12 same-seed configs byte-identical, prefill-logit KLD 0.000000 on both recorded points, no GPU
faults.  The gfx1030 arm itself is **not** validated on this fleet (no RDNA2 hardware; the reporter's
numbers are the only mixed-arch measurements).

Record: `patches/README.md` (block 06), `MANIFESTS.md` (per-block row 06), `archive/work/pr213-pr218-r132/`,
`WORKLOG.md` 261025.001, PR #218.

---


### §58 -- A per-device scalar is sound only if its inverse lands on that device's partial before the reduce (r137, block 14) — TEMPORARY

**Status: doctrine for a temporary block.**  `-sm relaxed-tensor` exists only so `flint`/`mellum`/`mistral4`
can tensor-split; it is expected to be dropped when upstream fixes the MoE amax under a tensor split
(`TODO` #104).  The rule below is what makes it sound, and it applies to any future "replicated tensor whose
values differ per device" `PER_DEVICE` state.

**The graph shape.**  `build_amax_src1_f16` takes the max of a whole matmul row and divides `cur` by a
power of two `e` derived from it, undoing it on the down-projection output.  Under a tensor split no device
holds a complete row (`n_ff_exp` is split on axis 0), so each device's `e` is the max of its own slice.  The
claim is that `sum_d W_d . (x_d / e_d) * e_d == sum_d W_d . x_d` **exactly**: `e_d` is a power of two, so the
scaling commutes with the sum and with the Q8_1 block quantization (whose block scale is `max/127`), and
`e_d <= e_global` (a local max cannot exceed the global one) means every device keeps at least the
reference's f16 headroom.

**The rule.**  That equality only holds if the multiply by `e_d` is part of the device's own subgraph, i.e.
if the reduce point stays **after** the per-device undo.  In the Meta backend that is the delayed-reduce
MUL-chain in `ggml_backend_meta_graph_compute`, and it must accept `PER_DEVICE` next to `MIRRORED`.  If it
stops matching, the all-reduce lands on the `mul_mat_id` output and the multiply then uses **one** device's
`e` on the **summed** partial.  That failure mode is silent in the common case: with the shipped divisor
(1024) `e = 1` unless a row max exceeds 1024, so a broken ordering and a correct one produce identical
output on ordinary prompts.  **Verify this mechanism with `e != 1` forced** (the r137 check temporarily
lowered the divisor to 1), not with the shipped one.

**What is permitted.**  The state may be produced only by `handle_pool_1d` for a whole-row `POOL_MAX` over
an axis-0 split (a windowed pool would change the output *shape* per device).  It may be consumed only
elementwise/broadcast on the device that produced it (`div` into `cur`, `mul` out of the down projection).
Every copy, gather, reconcile or read-back path aborts.  Anything else is `MIRRORED` semantics misapplied,
which is exactly what upstream #31081 removed.

**Scope.**  Exact on RDNA/HIP.  A backend whose in-kernel `src1` rescale is **non-linear** (Vulkan's
clamp-into-fp16-range, #30656) is not equivalent, which is why this is an opt-in mode and not part of
`-sm tensor`.  `ggml_pool_1d` has exactly one producer in the llama graph, so the state is unreachable from
any arch that does not build the amax rescale.

**Evidence (r137, gfx1201, ROCm 10.1.0).**  Flint Q8_0 2x R9700: greedy text byte-identical to `-sm layer`
and to itself at 3 GPUs; with the amax divisor forced to 1, an instrumented pool kernel shows each device
pooling its own 256-wide slice with the per-device maxima differing on ~half of the ~185 calls per run, and
the text still matches `-sm layer` exactly.  Qwen3.5-4B (no amax node) is byte-identical under `tensor` and
`relaxed-tensor`.  `scripts/check-gpu-faults.sh` clean.

Record: `patches/README.md` (block 14), `WORKLOG.md` 261025.007, `TODO` #104.


### §65 -- A fusion's precondition belongs to the arm it dispatches to: the dense-only `!ids` assert must not gate the routed fused-gate arm (r161, issue #241)

`ggml_cuda_mul_mat_q` serves **two** fused gate/up/GLU arms whose preconditions are disjoint:

* upstream #29948's dense interleaved arm (`has_fusion`): `ids == nullptr`, bias/scale allowed, NVIDIA-only by
  construction (its write-back relies on the MMA accumulator layout); and
* block 08's routed arm (`has_gate`, `ggml_cuda_mul_mat_q_switch_type_gate`): `ids != nullptr`, **no**
  bias/scale, both expert weight tables read with the same ids mapping, RDNA.

The r138 re-base took upstream's whole `if (fusion) { ... GGML_ASSERT(gate && !ids && !bias && !scale); }`
preamble into the function that also serves the routed arm, so the MoE gate/up+GLU prefill fusion -- which
passes `ids` and only `gate` -- aborted on **every** configuration that reached it (issue #241, r159:
`lfm2moe` server prefill).  The rule: an assert in a shared entry point must state the **union** of the arms'
preconditions, and each arm asserts its own where it dispatches (`ids == nullptr` -> dense,
`ids != nullptr` -> the routed kernel).  Bias/scale stay dense-only because the routed arm sets none.

**Guarding a matcher with `ids == nullptr` is not the fix**, and the reporter's branch-1 guard is provably
inert: `ggml_cuda_should_fuse_mul_mat_q` -> `ggml_cuda_mul_mat_q_fusion_matches` requires
`up->op == GGML_OP_MUL_MAT`, so the dense predicate can never match a `MUL_MAT_ID`.  Guarding branch 1 only
*looks* like a fix because the env kill switch (`GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`) is what actually
stopped the abort; the branch that fires is the routed one.

**Why it survived five releases** (r138-r160): the triple matcher only runs when the *pair* fusion stands
down, and the stand-down is arch-dependent.  On RDNA4 `mmb_cfg().routed = 0` (MMB's routed path is a
measured loss there), so `mul_mat_q_pair` takes the gate/up pair before the triple is ever considered; on
gfx1151 `routed = 1`, the pair stands down *for MMB*, and MMB's fused GLU usually takes the triple -- only a
model whose GLU shape MMB declines (the reporter's `lfm2moe`) falls through to the MMQ triple.  So every
maintainer arm (gfx1201 dense/MoE, gfx1151 Qwen3.x) missed it, and the issue is only reachable on an arch
whose MMB routed path is on with a model MMB's GLU does not cover.  Repro on gfx1201 is
`GGML_PAIR_OFF=1` (which stops the pair from hiding it); on gfx1151 it is the plain server path.

**Evidence** (`LFM2.5-8B-A1B-i1-Q6_K`, the reporting model, and `Qwen3.6-35B-A3B-UD-Q4_K_M`, single GPU):

* gfx1151: `llama-server` + a `/v1/chat/completions` request aborts at `mmq.cu:468` on the r148 tree; the same
  request on the r161 build answers, and its 256-token output is **byte-identical** to
  `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` (`965 chars sha=d55bc8a8def9`).  Prefill there is **flat** against the MMB
  individuals the arm replaces (5421/5425 vs 5397/5398 t/s, three `cache_prompt:false` requests each), so
  restoring it is not a perf loss and MMB's stand-down predicate did not have to change in this release.  The
  two predicates still disagree (`mmb_routed_will_take` vs `ggml_cuda_mmb_supported_glu`) and that mismatch is
  what made this abort reachable at all; it is `TODO.md` #115.
* fused vs `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` (both with `GGML_PAIR_OFF=1` to reach the arm on gfx1201):
  **byte-identical** 200-token greedy text (`1264 chars sha=727257c78293`) and per-token **prefill-logit KLD
  -0.000000 / same-top-p 100.000 %** (`llama-perplexity --kl-divergence`, 40 chunks, `-c 512`, gfx1201).

A routed fused gate arm is therefore a *width* and *shape* claim like any other fused kernel: it owes the
unfused triple its exact logits, and it owes the graph an `ids != nullptr` precondition that the dense arm's
assert must not shadow.


### §69 -- A fusion that stands down for MMB must ask the dispatch predicate, not a weight type (r171, `TODO` #115 closed)

**The claim.**  Two matchers decide who serves a routed (MoE) or dense gate+up pair, and one of them decided on a
different question than the other: `mul_mat_q_pair_try_fuse` stood down whenever
`ggml_cuda_mmb_routed_will_take(w)` / `_dense_will_take(w)` said MMB supported the **weight type**, while the later
`{MUL_MAT_ID, MUL_MAT_ID, GLU}` triplet matcher (and the `mmb` dispatch itself) decides on the **shape**
(`ggml_cuda_mmb_supported_mmid()` / `_supported_mm()`: `T >= mmb_min_t()`, `n_used <= 64`, `K` a whole super-block
count, a contiguous `dst` of the expected layout, ...).  The type-only predicate is a *superset*, so the pair stood
down for shapes MMB never claims — and the mmq/mmvq triple took the gate/up instead, i.e. the pair never ran as the
fallback it exists to be.

**Why it is a correctness rule and not only a preference.**  The same shape predicates are what the graph optimizer
uses to mark an activation **BF16-only**: when every consumer of an activation in the split is a bf16 reader, the
producer skips its F32 store entirely (`ggml_cuda_mmb_mark_bf16_only`, gfx1151's HC16 arm).  A fusion that preempts
one of those readers must therefore not read the F32 tensor — a pair or triple that fuses a pair of MMB-claimed
muls reads exactly the buffer the mark declared dead.  Two rules follow, and they are the invariant:

1. **A fusion stands down on the exact claim** — both muls `ggml_cuda_mmb_supported_mmid()` (routed) /
   `_supported_mm()` (dense).  That is a strict superset of the fused-GLU predicate (MMB's GLU additionally requires
   `mmb_glu()`, a plain unswapped SWIGLU and a matching `dst` layout), so the priority chain is well defined and can
   never preempt an MMB claim: **MMB's fused GLU -> the mmq/mmvq triple -> the shared-quantize pair -> the two plain
   muls**.  The pair is the fallback for the shapes none of them takes.
2. **The marking pass must not elide an activation whose reader a fusion may consume.**  A routed `MUL_MAT_ID`
   consumer no longer authorises `mark_bf16_only` (it marks the bf16 *copy* instead, so the MMB reader still finds
   it).  With (1) in place this is belt-and-braces — the fusion and the mark cannot both be wrong — but the two live
   in different passes (marking in `graph_optimize`, matchers in `graph_compute`), so each must hold on its own.

**Purity is preserved because the arms are the same arithmetic.**  The mmq pair and the mmq triple accumulate the
same dot products with the same `J` tile and apply the GLU elementwise, so the arms produce **bit-identical** output
on every model measured, not merely similar: gfx1151 (LFM2 Q6_K 765 chars, 35B-A3B Q4_K_M 723 chars) reads the same
text for pair == triple == unfused (`GGML_CUDA_DISABLE_MOE_MMQ_FUSION`) == `GGML_CUDA_MMB=0`, and gfx1201 is inert
(`mmb_cfg().routed = 0` there, so the pair already won).  The prefill-logit gate reads 0.000000 / 100.000 % on 27B
Q8_0 and 35B-A3B Q4_K_M.  Time is neutral: gfx1151 pair vs triple 3379/3399 vs 3348/3407 t/s (LFM2) and 2304/2312 vs
2296/2313 (35B-A3B) at pp512, interleaved `r=3` with the dense MMB path untouched.

**Two traps this cost.**  (a) A matcher that probes an *arbitrary* node pair must null-check every source slot
before calling these predicates: the pair matcher runs on every graph node, and its first version dereferenced a
missing `src[1]` on an op followed by a `MUL_MAT` — an instant SIGSEGV, found only because a gfx1151 model's warm-up
graph happened to have that shape (`ggml_cuda_mmb_supported_mm`/`_mmid` are null-safe now too).  (b) A model whose
GRAPH the item was filed against is not necessarily one that reaches the arm: with the gfx1151 weight-type mask,
LFM2 Q6_K is accepted by MMB's **fused GLU** (`M=1792 K=2048 T=512 n_used=4`, `MMB_GLU fused` in the log), so the
"MMB-declined MoE" premise of `TODO` #115 did not hold and the original fused-vs-`GGML_CUDA_DISABLE_MOE_MMQ_FUSION`
A/B compared two MMB-GLU arms.  Reproduce the *decision* with `GGML_CUDA_MMB_LOG=1` before attributing a number to a
matcher.

---


### §70 -- A second sequence in a batch is a *stride* in the mask predicate, not a new mask (r172, PR #247)

**The claim.**  The derived kq mask's encoding is one predicate, `tok_lo <= cell_pos <= tok_hi`, evaluated per
(cell, query token) by both FA kernel families.  It separates "own" cells from "foreign" ones with the `INT32_MIN`
sentinel, which only works while a ubatch holds **one** sequence — so a `--parallel > 1 --kv-unified` context (two
slots prefilling or verifying together, one slot's prefill next to another's decode) kept the packed mask *and*
reserved it, because `sched_reserve()` measures with `n_seq_max` sequences.  Giving sequence `k` of the ubatch its own
position range — cells and tokens both shifted by `k * KQ_DERIVED_SEQ_STRIDE` (`2^23`) — makes the same predicate
exact for a batch of several sequences with **no kernel change**: for `0 <= pos < 2^23` and `k' != k`, a `k'`-cell
lands at `pos + k'*2^23`, which is `>= (k+1)*2^23 > tok_hi + k*2^23 <= (k+1)*2^23 - 1` for `k' > k` and
`<= k*2^23 - 1 < tok_lo + k*2^23` for `k' < k`, so the ranges cannot overlap.

**The rule.**  Four conditions, and every one of them is a refusal, not a fallback surprise: one KV stream
(`n_stream == 1`), every token in exactly one sequence (`n_seq_id == 1`), **no cell carrying two of the batch's
sequences** (a prefix copied with `llama_memory_seq_cp` — the shared system prompt — would have to be visible to two
ranges at once, which one shifted window cannot express), and every position below the stride.  A multi-sequence
batch of **at most 32 tokens** keeps the packed mask: that is exactly the multi-slot verify width
(`2 x 16` rows at `--spec-draft-n-max 15`), the reserve's slack covers its few MiB, and it keeps the per-ubatch
M-RoPE degeneracy scan and the O(used cells) sharing scan off the per-step path.

**Correctness never depends on the reserve latch.**  `kq_mask_derivable()` is evaluated per ubatch when the graph is
built and re-runs the whole predicate, so a batch the form cannot serve takes the packed mask by itself.  The latch
(`kq_mask_multiseq_seen`) exists only for the *reserve*: a pre-split batch that cannot be served sets it, `decode()`
re-reserves once with the packed mask, and no later batch has to grow the compute buffer (issue #42, the same pattern
as the r160 2-D latch).  A multi-sequence batch is the one shape whose refusal depends on the **cache**, not only on
the batch, which is why the reserve cannot predict it and the latch must exist.

**How to judge a change here.**  The oracle is the packed fill itself: the author's diagnostic
(`LLAMA_KQ_MASK_DERIVED_VERIFY=1`) rebuilds each derived ubatch's mask with `set_input_kq_mask` and compares
visibility cell by cell.  Re-run on gfx1201 with **0 mismatches** on every derived multi-sequence ubatch of
llama-perplexity at `-c 512 -b 2048 -ub 2048 --kv-unified` (4 sequences per batch) for dense 27B (1 GPU and 2 GPU
`-sm tensor`), Qwen3.5-4B (hybrid attention+GDN), Gemma4-12B and gpt-oss-20b (both iSWA, the latter with a live SWA
window), plus the 2-slot server shape.  Logit level:
`gate-prefill-logits.sh --ctx 512 --ub 2048 --ab "LLAMA_KQ_MASK_DERIVED_MULTISEQ=0"` reads **0.000000 / 100.000 %**
at f16 and q8_0 KV.  `gate-park-idle-slots.sh` passes (3 GPU tensor, Flash-Next + MTP, 3 slots).  Single-sequence
inputs are byte-identical by construction (offset 0, unbounded non-causal window): dense 27B, 35B-A3B `-ncmoe 30`
and hybrid 4B greedy text is unchanged, and `plain == draft-mtp` still holds (2 GPU tensor, 400 tokens).

**The shared-cell counter is the one thing that can silently lie.**  `llama_kv_cells::get_n_multi_seq()` is a
`uint32_t` maintained by every mutator (`rm`, `seq_rm`, `seq_keep`, `seq_add`, `pos_add`'s empty-out branch, both
`cp` forms, both `set` forms, `reset`; `mv` is net-neutral because its destination is asserted empty).  An
**under**-count is a wrong mask — the fast path would claim "no shared cells" and derive — so the counter is not a
cache: the O(used cells) scan is the semantic answer and the counter only licenses skipping it.  Validate it with the
randomized test (`wip`-side, 400 000 random operations of every kind on 512 cells x 6 sequences, counter == recount
after each, shared cells present in 96 % of the states), and re-run it when a new mutator appears.  The audit warning
the same way: a *new* way to put two sequences in one cell (a future `seq_cp` variant) must bump the counter or the
whole predicate is unsound.

**What it costs, and what it does not buy.**  The batch-wide skip bitmap
(`flash_attn_kq_derived_blocks`, §37) is a *window* over the whole batch, so a multi-sequence prefill loses the
packed path's per-tile skip: the shifted union covers every batch cell and no group is skipped.  That is measurable
in principle and neutral here — two concurrent 16K prefills measure 26.96 s (packed) vs 27.23 s (derived) at 1 GPU and
17.07 vs 17.09 s at 2 GPU `-sm tensor`, and a 112K-resident decode beside a 16K prefill 13.97 vs 13.96 s, because the
removed host fill (`n_kv x n_ubatch x 2` bytes per ubatch) was the larger term.  The win is memory: **27.51 -> 26.41 GB
device and 1.97 -> 0.98 GiB server RSS** at load on `-c 245760 -ub 2048 --parallel 2 -ctk/-ctv q8_0`, dense 27B.  It
is **zero** on a graph that also holds an `allow_derived == false` builder (qwen4exp/QSA, MLA): that builder creates
and fills the packed mask anyway, so Flash-Next measures identical VRAM and RSS — the win is a dense/plain-cache one,
not a universal one.  And a server that actually shares prefixes through `llama_memory_seq_cp` pays one re-reserve and
then keeps the packed mask for the rest of the run: the memory is given back, the 1 GiB is not.

**Record:** `WORKLOG.md` 261029.003, `archive/work/kq-derived-multiseq/`, `patches/README.md` (block 05),
`benchmarks/2026-10-29-derived-kq-multiseq.md`, `ENVIRONMENT.md` (`LLAMA_KQ_MASK_DERIVED_MULTISEQ`).

### §43 -- One split holds one consumer of a staged expert table

The MoE expert cache serves a routed `MUL_MAT_ID` from the **arena**, and the arena holds the slots staged
for *one* routing.  Two `MUL_MAT_ID` consumers of the same expert weights in one split therefore cannot
both be served: the scheduler's guarded fast paths (`moe_cache_take_over`, the used-expert pruning) are
already declined when they count two consumers, but the op-level arena read is not, so the second consumer
reads the first routing's slots.  qwen4exp is the model that exposes it, because its last layer has two
consumers of the same table: the gathered logits tail and the full-row MTP export tail.

Measured on r95 (before the fix), Flash-Next GSQ-IQ3_XXS + the shared Q8_0 MTP head, 2 GPU, `-sm layer`,
`-ncmoe 10`, `draft-mtp` n-max 3, `-n 48`, seed 42, temp 0:

| arm | text | acceptance |
|---|---|---|
| plain decode (`--spec-type none`) | 172 chars `de61b9b1bed7` (the truth) | n/a |
| r95 + `draft-mtp`, 6 natural runs | 5x `de61b9b1bed7`, 1x `a2b6c01a0a19` (223 chars) | `0.00000` on the bad runs |
| r95 + the split fix (PR #186), 6 runs | 6x `de61b9b1bed7` | healthy (`0.63636`) |

The flip is driven by the **persisted MoE heat map**: pinning one captured heat-map state reproduces the
bad output 2/2, and `MOE_EXPERT_CACHE_MIB=0` restores purity, while `GGML_SCHED_DEVGATHER=0`,
`GGML_SCHED_STAGE=0` and `GGML_SCHED_STAGE_MIN_TOKENS=999999` do not.  `-sm tensor` is immune (the meta
backend has no cache gather) and 35B-A3B is immune (one consumer only).  Under `-sm tensor` the bug shows up
as throughput instead: with both consumers in one split the last layer's 1062 MiB table is copied for every
prompt chunk (Flash-Next `-ncmoe 10`, follow-up turns 645/641 ms -> 449/341 ms, bit-identical text).

**Rule:** a split may hold only **one** consumer of a cache-managed expert table.  The scheduler enforces it
by starting a new split at a second `MUL_MAT_ID` consumer of weights the current split already stages
(`ggml_backend_sched_split_graph`, r96, PR #186); the re-registration path then stages them again for that
consumer's own routing, which is ready by then.  `GGML_SCHED_MOE_DUP_SPLIT=0` restores the merged split for
A/B only.  Any future "recompute this layer's FFN on all rows" path must either get its own split the same
way or keep the table out of the cache.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §43.
### §63 -- A verify batch is only as cheap as its widest sequence set: aligning one step's sequences into one ubatch, and the oracle that can judge it (r153, issue #234)

`llama_memory_hybrid::init_batch()` splits a decode batch with
`llama_batch_allocr::split_equal(n_ubatch, !unified, n_rs_seq + 1)`, and `split_equal()` takes **one token per
sequence set per round and stops when the shortest set runs out**.  A decode step in which two slots verify
different draft lengths therefore becomes **two ubatches**, i.e. two complete passes of the target model over
the same step.  Padding the shorter sequence to the same length is not an option on this model class: a padded
row advances the recurrent state in `mem_recr->prepare()`.

Nothing in the delivery caused it, and a fixed `--spec-draft-n-max` never exposed it (every slot drafts
`n_max`, so the lengths are equal for free).  The adaptive MTP controller's **per-sequence** depth does: two
concurrently verifying slots routinely sit a step apart.  Measured on the r152 tree (27B UD-Q4_K_XL, greedy,
`n_predict 400`, `--parallel 2 --kv-unified`, concurrent high- and low-acceptance prompts): the pair aggregates
**108.1 t/s at 2 GPUs and 86.9 at 3 GPUs**, against **145.6 / 132.2** for the same two prompts whose drafts
happen to be the same length -- +35 % / +52 %, and the penalty grows with the number of devices the duplicated
pass spans.  That is why the gate runs both device counts.

The rule: **the sequences that verify in one step must present the same row count.**  The fix (`LLAMA_SPEC_ALIGN`,
on) makes that true **inside the drafter whose per-sequence caps created the mismatch**
(`draft-mtp-adaptive`): every drafting sequence gets the deepest cap in the same `draft()` call, bounded by the
smallest hard cap in the call (a response running out of tokens) and by the expert cache's band.

It is deliberately NOT a cut of the finished drafts to the shortest, and that was measured rather than assumed:
r153 shipped that cut and it is a **loss** on the n-gram-first chain, where the lengths differ because of
CONTENT rather than a cap.  On the reporter's own case (`--spec-type ngram-mod,draft-dflash` + a DFlash2 draft,
27B, 2 GPU, recall + prose, `-n 200`) the `min` fires on every step whose n-gram match is shorter than the DFlash
block, truncates the OTHER slot's good draft, and the pair aggregates 65.6 -> 61.7 t/s (-6 %) while the prose
slot's acceptance falls 74 % -> 50 %: splits 12 -> 1, and the lost proposals cost far more than the removed pass.
r154 removed it, so a chain of different drafters behaves exactly as before (measured inert: 64.6 vs 64.4 t/s,
identical draft counts) and `gate-spec-align.sh` gates that arm.  A corollary worth stating: a length mismatch is
only worth equalizing when a CAP caused it; when the content did, the two drafts are not comparable and the
second ubatch is the cheaper arm.  A slot that drafted **nothing** is split by cause: a drafter that gave up (a
`--spec-draft-p-min` cutoff, or an implementation that returns no tokens) is CONTENT, while a slot that could not
draft at all -- `get_n_draft_max() <= 0`, i.e. its last token of the response or a full context waiting for a
shift -- is a CAP.  That cap case is deliberately **not** equalized (2026-10-28, TODO #111: the step keeps
splitting, and the gate counts and bounds it -- see the addendum below).

**The oracle: the promise is PER CONFIG, and that is what makes a two-slot text diff the wrong instrument.**
Any fix for this must change some slot's verify width (align up, or cut the other slot down), and a verify
width is a numeric axis like any other (§36, §45, §60): the target's kernels see a different number of rows.
`LLAMA_SPEC_ALIGN=0` and `=1` are therefore two **configs**, exactly as `--spec-draft-n-max 4` and `5` are, and
the rulebook promises greedy text *per config*, not equality between configs.

Both halves of that promise were measured, and neither is a "near-tie that might flip from run to run":

* **Each config is deterministic.**  Every arm of the gate -- knob on and off, 2 GPU, 3 GPU and the host-expert
  arm -- reproduced its text **character for character across every run** (4/4 runs at 2 GPU, 2/2 at 3 GPU and
  on the host-expert arm), with `ROCBLAS_USE_HIPBLASLT=0` (§44).  The on/off delta itself is stable: the same
  divergence character, char 220 of 1202/1352 at 2 GPUs and char 627 of 1234/1315 at 3 GPUs, in every run.
  **(2026-10-28) this is scoped to the single-slot arms:** a two-slot *aligned* pair is not reliably stable
  across runs -- see the addendum below, which records two basins on one unchanged binary.
* **The delta is the width, not the change.**  A **fixed `draft-mtp` pair at depth 5 against depth 4** -- both
  slots equal length in both arms, no alignment in play -- moves the same slot at the same kind of point
  (char 259: "Ada not on Monday" vs "but Ada not on Monday").

And the token *rule* is untouched: every emitted token is still the target's own greedy sample at that
position; what moves is how many positions are verified per pass.  The purity statements this change makes are
therefore the ones with no width axis in them:

* **`n_seq == 1` is byte-identical.**  One drafting sequence has nothing to align, so the code is inert by
  construction and the text is unchanged (verified: `llama-cli`, knob on vs off; and every single-slot server
  request in the gate, at 2 and 3 GPUs and on the host-expert arm).
* **The equal-length arm does not regress** (two identical prompts: -0.4..-2.1 %).
* **The cross-process text comparison is stable** with `ROCBLAS_USE_HIPBLASLT=0` -- two runs of the same arm
  agreed character for character in every slot, which is what makes the pair's delta attributable at all (§44).

Corollary for any future change to a *verify width* (a draft depth, a band bound, an alignment): text identity
is owed on the arms where the width does not change (single-sequence, single-slot) and against a same-width
control; a cross-width text difference is a new per-config reference, not a defect.  It is still worth knowing
that a multi-slot server upgraded across such a change can produce different -- equally valid -- text, which is
why the delivery's release note says so rather than a gate failing.

**Addendum (2026-10-28, TODO #111) -- a slot that cannot draft at all is declined, and the determinism claim is
corrected.**

The last exception (the single-row split step of the paragraph above) was investigated in full and
**deliberately left as it is**; nothing in the delivery changed for it.  Two things are worth recording, because
one of them contradicts what this section said before:

* **The trade had to be checked, not assumed, because it differs from the depth-mismatch case.**  There the
  second ubatch is pure overhead -- aligning up keeps every row and removes a pass, which is the +23..+30 %.
  Here the second ubatch carries the neighbour's **genuine draft rows**, i.e. real work: split = `2 passes /
  ~6 tokens`, while the only fix that removes the split (suppress that step's drafts for everybody) = `1 pass +
  the next step's 1 pass / ~7 tokens`.  Two passes either way, so it measured **-0.3 %** (below the floor) and
  was declined -- change for change's sake.  There is no correctness at stake either: a verify width is a width,
  and both arms emit the target's own greedy sample at every position.  The residual is now **counted and
  bounded** by `scripts/gate-spec-align.sh` (`TAIL_MAX`, default 2 = one per response boundary) instead of being
  excluded from its split count, so a change that manufactures single-row ubatches is visible.
  Record: `benchmarks/2026-10-28-spec-align-111.md`.
* **The two-slot pair is NOT reliably deterministic, and this section was wrong to say it was.**  With four runs
  of one **unchanged** binary at 3 GPUs the pair landed in two basins -- 1237 and 1232 characters for the
  low-acceptance slot, with a 9 % aggregate t/s difference between them (159 vs 174 t/s) -- so the basin is a
  property of the *run*, not of a build.  The alignment is why: it makes one slot's verify width depend on its
  neighbour's adaptive state and on which rows share its batch, so an innocent difference in batch composition
  propagates into a near-tie.  The gate never gated the pair's text (it reports the divergence point as INFO, and
  that point moves between runs -- char 942 and char 34 in two runs of the same binary), which is the right
  call; what follows is that a two-slot text diff is not an instrument at all, and that the rulebook's
  determinism claim is scoped to the **single-slot** arms.  Judge this code by the split count, the single-slot
  text identity and the pair's aggregate t/s (which spreads ~10 % for the same reason), never by cross-run pair
  text.

---

### §51 -- A kernel-config row keyed on one tile dimension also configures every query width that reaches it

**Claim.**  The MMA flash-attention config table is selected at compile time by
`(DKQ, DV, ncols1*ncols2, ncols2)`, and its `nthreads`/`occupancy` feed `__launch_bounds__`, so a row is part
of the *kernel*, not a runtime choice.  `ncols` alone does not identify the regime: the query-width dispatch
gives `ncols1 = 8, ncols2 = 8` (ncols 64) to **both** the W=5..8 decode/verify band and every wide prefill
(`Q->ne[1] > 4`).  A band row placed there therefore also re-tunes prefill: measured on RDNA4 GQA-8/head-256,
prefill ran the band geometry (16 warps, occupancy 1, `nbatch_K2/V2` 64, combine 32) instead of upstream's
256 threads / occupancy 2 and lost **26 % of the FA kernel / 10 % end to end** at `pp512 @ d98304`
(`TODO.md` #97/#98, `WORKLOG.md` 261024.006).  Deleting the row (r126) is +36 % at d98k, +19 % at d32k and
+3 % at d0 -- i.e. the band row was a *prefill* regression, and the delivery went from -10 % below upstream
at depth to +20 % above it.  The W=5..8 band arm it also configured measured unchanged within noise on a
MoE-bound decode, and r129 re-measured the premise for a separate band instantiation and declined it
(`TODO.md` #99 retired: the shipping 256-thread row beat the existing two-tile 16-warp arm 110.1/110.2 vs
108.9/107.6 t/s on the W=8 arm, text byte-identical — no decode-bound head-256/GQA-8 workload exists here).

**The band's other end of the same trade: its per-column KV split is part of the width contract (r134, issue
#224).**  A band block splits each Q column's KV rows `np = nwarps*cols_per_warp/ncols` ways.  The 32-column
block (widths 1..4) is 8 warps -> `np` 4; leaving widths 5..8 on prefill's 8-warp 64-column row gives `np` 2,
a different reduction, and widths 5..8 stop being bit-identical to widths 1..4.  Measured on dense head-256
GQA-6 (Qwen3.8-27B UD-Q4_K_XL + DFlash2 `n_max 5`, 512 greedy tokens, q8_0 KV): greedy text with the draft
differed from plain decode at token **325** (8K prompt) and **142** (110K), identical with
`GGML_HIP_FA_BAND_WIDE=0` and on r123 — while every gate in r126..r133 passed.  So the 64-column block runs
the 16-warp row (`fattn_mma_config(512, 1, 64, 64, 64, 32, 1, true)`, `nbatch_combine` 32 to stay inside the
64 KiB LDS limit) through its own case instantiation (`band_wide`, `template-instances/fattn-mma-f16-band-wide-*.cu`),
prefill keeps the 256-thread table row (the paragraph above is unchanged, and `pp512 @ d98304` is still 1641
t/s), and a `static_assert` on the two band rows (`np`, `nbatch_fa/K2/V2`, `nstages_target`, `Q_in_reg`) turns
a repeat into a **build error**.  End to end the single pass measures at parity with the two-tile arm (8K
75.2/74.6 vs 75.0/74.9 t/s; a 131072-cell FA range 61.0/60.6 vs 60.9/60.8; the W=8 MoE arm 164.5/166.6 vs
166.3/164.9, text byte-identical), so purity costs nothing measurable here.

**A band-width probe must reach the block it is probing.**  `tests/test-logits-width-probe` ran at `P=256`
for years and never touched widths 5..8: the wide launch's gate is `K->ne[1] >= 4096` and `K->ne[1]` is the
context's **used** length, so `n_ctx` (a fixed 4096) was irrelevant and the probe's prefix was 256 cells.  That
is why `width_purity=PASS` was reported on every r126..r133 arm.  The probe's `n_ctx` is now `P + W + 8` and
a purity run must pass **`P >= 4096`**: at that point the r126 geometry fails (worst maxdiff 0.199764 on q8_0
K/V, the row-0 hash splitting between W=4 and W=5) and the shipped geometry passes with maxdiff 0.  A probe
prompt also has to be *tokenizable*: the buffer is sized from `llama_tokenize(..., nullptr, 0, ...)` since
r166, so a model's own 25k-token in-domain corpus can be the prompt instead of failing on a fixed 8192.

**The ratio axis is measured at 6, 8 and 12 (r166, `TODO` #102 closed).**  `ncols2 = 8` folds a whole GQA
group in one pass, a partial one with a masked tail, and ratios 9..12 in two passes -- so those are three
distinct code shapes and each needs its own arm.  Ratio **6** = dense 27B (24 q / 4 kv), ratio **12** =
qwen4exp Flash-Next (24 / 2), and ratio **8** = Qwen3.6-35B-A3B (16 q / 2 kv at `n_embd_head_k` 256) -- the
exact-fold case, which was carried as an accepted limitation because "no head-256 GQA-8 model exists in the
fleet" while one had been in it all along.  All three are width-pure at `P=4096` (maxdiff 0, `KV=q8_0` and
`f16`, `-sm layer` and `-sm tensor`, `RS=0` and `RS=from_w`), and at ratio 8 a `rocprofv3` census shows the
64-column **band** instantiation (not the prefill row) dispatching at widths 5..8.  A gemma4 model is not an
arm for this axis: its SWA layers are head 256 at ratio 2 and its full-attention layers ratio 8 at head
512, so neither enters the band predicate.

**Rule.**  A config row that exists for the decode/verify band must be reachable **only** from the band: key
it on a compile-time band flag (or on a distinct instantiation), never on `ncols`, `n_kv` or `n_q` at
runtime alone.  The band's own widths must also share one geometry -- the per-column KV split (`np`) and the
KV batch shape -- and a width probe that is meant to see a block must be run above that block's dispatch
threshold.  Write in the comment which query widths reach the row.  When two regimes share an
instantiation, the shared row must be the one safe for the wider regime and the band gets its own
instantiation.  **A row that reaches prefill owes `scripts/gate-prefill-logits.sh`** (changing the row
changes the KV-row split and combine order -- the family-switch class of §34/§46), plus the band arm's own
decode/verify A/B so the band's measured gain is not silently traded away.

Record: `TODO.md` #97/#98/#99 (all closed) and #102 (closed r166), issue #224, `patches/README.md` block 05,
`WORKLOG.md` 261024.006 / 261024.007 / 261024.010 / 261025.004 / 261028.008,
`benchmarks/2026-10-25-fa-band-width-purity.md`, `benchmarks/2026-10-28-fa-band-gqa8-arm.md`.

