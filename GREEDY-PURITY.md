# Greedy purity, VDR, and the nature of the block 10 variance

Status: **reference document**. Read this if you are shipping block 10
(`0010-rdna-boosts-block-10-k-quant-boosts-Q4_K-Q5_K-Q6_K-Q.patch`) and care about bit-exact reproduction of greedy
decode vs a stock llama.cpp build. It explains *why* block 10 changes decode
numerics, what that does and does not mean for correctness, and how to reason
about the variance in practice. Sections 1-8 were written for the 12-block
set; block 13 adds a second decode-numerics source on its rewritten mmvq
rows — see **§9 (2026-09-02 addendum)** before relying on the
"block 10 is the single patch" claims.


## How this document is organised

**Read the summary entry for a section, not the whole file.**  Each `§N` here carries the *claim, the
measured numbers and the rule*; the long "how we found it" narratives for the closed cases live in
[`archive/docs/GREEDY-PURITY-FINDINGS.md`](archive/docs/GREEDY-PURITY-FINDINGS.md) under the same `§`
numbers, and the delivery-side record of each amendment is in `patches/README.md` + `WORKLOG.md`.
Section numbers are **stable** — dozens of docs cite them, so they are never renumbered; new findings
are appended.

**The core invariants (everything below is evidence for these):**

1. **Inside the decode/verify band, one arm — chosen from the `n_tokens` band and `n_kv`, never from the
   exact width, `n_kv` alone, or `n_tokens == 1`.**  A W=1 decode and a W=(`--spec-draft-n-max` + 1)
   verify of the same state must take the same kernels (§§11, 13-17, 25, 26).
2. **A batch you can roll back into must be written by the kernel whose snapshots you read** — the
   rollback bound is a purity bound, not a tuning knob (§27).
3. **Same real-number result, different rounding order** is what the block 10/13 numerics changes are
   (§§1-9): stock is the reference *standard*, not the mathematically special order.
4. **A newly enabled KV type / op argument / fusion is a new kernel family** and needs the full band
   sweep, the text gate, the oracle and the perplexity comparison (§§20-22).
5. **Purity outranks raw non-MTP throughput**: land bit-identical decode/verify even at a few percent of
   the widest verify batch, record the delta, then repay it structurally (§§19, 24).
6. **Use the instrument that can see the defect**: the raw-logit width probe for boundaries, the
   dense-masked path + a CPU oracle for a fused tensor op, random text for a causal leak, and an
   *executed-graph* dump when a chain looks dead (§§11, 21, 23).

**Section index** (status: *doctrine* = read before shipping / *current* = live state / *fix* = closed
finding, narrative moved to the findings file):

| § | claim | status |
|---|---|---|
| 1-5 | block 10 changes only the fp32 addition *order* on the K-split decode paths; VDR/`blocks_per_iter` is the mechanism | doctrine |
| 6-8 | speculative verify runs the same kernels on a wider batch, so the variance composes orthogonally; measured magnitude and stock-reproduction guidance | doctrine (§9 supersedes §2/§8 for block-13+ sets) |
| 9 | block 13's rewritten mmvq rows are a **second** decode-numerics source | doctrine |
| 10 | block 00 makes the verify width irrelevant on the small-batch FA path (`ntiles_dst_eff`) | doctrine |
| 11 | the guarantee is `--spec-draft-n-max <= 7`; cause A (block-12 AR crossover, fixed) + cause B (FA tile->WMMA at `Q->ne[1] > 8`, by design) | current |
| 12 | the guarantee depends on the KV-cache type; the two fast native types were impure (fixed by §14) | fix + the fast/slow trade |
| 13 | qwen4exp's hyper-connection band is `1 <= nt <= 8` (`HC_FUSED_MAX_TOKENS`) | fix |
| 14 | F1: the FA chooser spanned VEC (`n_q <= 2`) and TILE (`n_q >= 3`) for quantized K/V — one family now | fix |
| 15 | F2 cause 2: upstream's per-type mmvq caps are numeric boundaries inside the band (fixed; also a 14-26 % win) | fix |
| 16 | cause 3: the QSA dense decode arm was `n_tokens == 1` — now `<= QSA_DECODE_BAND` (8) | fix |
| 17 | the MoE shared-expert epilogue is token-generic with `nwarps` pinned | fix |
| 18 | the QSA sparse regime on gfx1151: the two 2026-09-11 width-dependences do not reproduce; one forced-arm q8_0 residual remains | current |
| 19 | purity first: a correctness fix may cost a few percent — land it, record it, repay it | doctrine |
| 20 | a newly enabled KV type is a new kernel family (the enablement checklist) | doctrine |
| 21 | a shared staging tile makes a block head-homogeneous (the QSA K/V-head bug + the two instruments that found it) | doctrine + fix |
| 22 | `iq4_nl`: the type is not the only thing an enablement touches | fix |
| 23 | a shadowed variable left a chain dead and the mask unfilled (four lessons, incl. random-text leak probing) | fix |
| 24 | the cost of a band amendment can be structural, not arithmetic (the column-blocked epilogue) | doctrine + fix |
| 25 | the RDNA3_5 single-token-only mmvq fusions are not decode/verify bit-identical (block-13 gate) | current |
| 26 | the prefill half of a regime policy must be band-keyed too (and defaults follow the documented policy) | doctrine |
| 27 | a rollback bound is a purity bound (`n_rs_batch`, the GDN chunked kernel's snapshots) | doctrine + fix |


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
   build*.

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
`wip/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.  Narrative (the `n_max` sweep, the
localisation code block, why the GDN was not the boundary): `../archive/docs/GREEDY-PURITY-FINDINGS.md` §11.

## 12. The guarantee depends on the KV cache type (2026-09-11, measured during the Block-15 revalidation)

> **The `q8_0`/`q4_0` impurity below is FIXED** (2026-09-11, block-08 amendment) — it was the FA
> *kernel-family* chooser, not the KV staging; see **§14**.  The §12 measurements stand as the
> pre-fix record, and its F16-staging explanation of the slow types is refined in §14.

Everything above was measured with an **f16** (or bf16) K/V cache.  Extending the probe matrix to every
same-type KV pair (`tools` + evidence: `wip/kv-quant-purity-followups/README.md`) shows the guarantee is
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
`beta/block-15-campaign-wins/README.md` §7) and are irrelevant to this table.  This table is orthogonal
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

## 18. The QSA *sparse* regime is width-pure on gfx1151; one q8_0 residual remains (2026-09-12; extended 2026-09-12 (6))

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
teacher-forced replay (new instrument `wip/strix-halo/qsa-item4/mstep.cpp`) of the plain greedy sequence
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
`wip/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md`.

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
   2026-09-12 `-Wshadow` audit (`wip/shadow-warnings/`).
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
`wip/strix-halo/qsa-item9/RECORD-2026-09-12-qsa-prefill-crossover.md`).

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
