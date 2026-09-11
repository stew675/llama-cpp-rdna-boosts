# Greedy purity, VDR, and the nature of the block 10 variance

Status: **reference document**. Read this if you are shipping block 10
(`0010-rdna-boosts-block-10-k-quant-boosts-Q4_K-Q5_K-Q6_K-Q.patch`) and care about bit-exact reproduction of greedy
decode vs a stock llama.cpp build. It explains *why* block 10 changes decode
numerics, what that does and does not mean for correctness, and how to reason
about the variance in practice. Sections 1-8 were written for the 12-block
set; block 13 adds a second decode-numerics source on its rewritten mmvq
rows — see **§9 (2026-09-02 addendum)** before relying on the
"block 10 is the single patch" claims.

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

Earlier notes (and the 2026-09-11 block-02 entries) claimed
`--spec-type none == draft-mtp` for `n_max <= 15`.  **That was wrong.**  The
claim had only ever been validated up to `n_max = 4`.  There are in fact **two
independent causes**, and they bound different configurations:

| config | `none == draft-mtp` guaranteed for | binding cause |
|---|---|---|
| 1 GPU (no split) | `n_max <= 7` (W <= 8) | B |
| 2-GPU `-sm layer` / no split | `n_max <= 7` | B |
| 2-GPU `-sm tensor` | `n_max <= 5` **before** this fix, `n_max <= 7` after | A, then B |
| 3-GPU `-sm tensor` | `n_max <= 7` | B |
| 2-GPU `-sm tensor` + `GGML_CUDA_ALLREDUCE=meta` | `n_max <= 7` | B |

The boundary is on the **verify batch width**, not on `--spec-draft-n-max`
directly: the target verifies the drafts *plus* the last committed token, so
`K = n_max + 1` and `n_max = 8` is already a 9-token batch.  Measured with the
raw-logit probe (27B Q8_0, `RS=0`, P=256; identical within a row = bit-identical
token-0 logits):

| config | `W = 1..8` | `W = 9` (= `n_max 8`) |
|---|---|---|
| 1 GPU | `4089b4d4` | `72af52db` |
| 2-GPU `-sm layer` | `4089b4d4` | `72af52db` |
| 2-GPU `-sm tensor` | `a4817ee6` | `b059daa6` |
| 3-GPU `-sm tensor` | `91434ea9` | `bc3faabd` |

So the guarantee is **`--spec-draft-n-max <= 7`** in every configuration, and
the first violating depth is `n_max = 8`.  (Before the fix, 2-GPU tensor broke
at `W = 7`.  1 GPU and `-sm layer` share a hash because layer splitting does
not change any kernel.)

**Use the probe, not the text gate, to establish a boundary.**  On 3 GPUs the
300-token greedy text at `n_max = 8` matched the plain run (`5037ef2e` in both)
while the logits had already diverged (`bc3faabd` vs `91434ea9`) -- no greedy
near-tie happened to flip inside that window.  Text equality is evidence *for*
purity, never evidence *against* divergence; this is the same near-tie rarity
noted in `../wip/sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`.

**Cause A (fork-specific; FIXED 2026-09-11): block 12's size-based all-reduce
dispatch.**  `ggml_backend_cuda_comm_is_small()` routes a reduction to the
internal host-staged pipeline below a per-device-count element count (32768 for
2 devices, 131072 for 3, 262144 for 4+) and to NCCL above it.  The two paths
are **not bit-identical** (different summation order; the internal pipeline
always does the FP32->BF16 round-trip).  Under `-sm tensor` the reduced tensors
scale with the batch width (`ne = ne0 * n_tokens`, `ne0 = 5120` here), so a
**7-token** verify batch is 35840 elements and crossed the old 2-device limit
while 1..6-token decode stayed below it: the same logical reduction, a
different algorithm, purely because the batch got one token wider.  Raising the
2-device crossover to 131072 (the 3-device value) fixes it -- the largest
verify batch, `n_max 16` -> 17 tokens, is 87040 elements, well under it, and
still far below the internal pipeline's own 1 MB (262144 element) cap.
Measured: `GGML_CUDA_ALLREDUCE=internal` (one algorithm for every size) gives
`W = 1/6/7/8` **all** `a4817ee6`, and so does the fix, with `W <= 6` keeping
their original hash (plain decode bit-unchanged) and only `W = 7,8` moving onto
the internal pipeline.  MTP throughput at `n_max 6` went 63.6 -> 71.3 t/s
(+12%) and at `n_max 12` 51.7 -> 58.0 t/s (+12%), pp/tg unchanged -- the
internal pipeline is the faster path at these sizes, so the fix is a win on
both axes.  1 GPU has no cross-device reduction at all, 3 GPUs stay under their
boundary until `W = 26`, and `-sm layer` never splits the reduction dimension:
all three were unaffected.

**Cause B (deliberate; open): the flash-attention tile-vs-WMMA switch at
`Q->ne[1] > 8`.**  Beyond `W = 8` the FA launcher prefers the WMMA kernel (much
faster at prefill-scale batches) and its reduction order differs from the tile
kernel's.  This is not accidental -- the fork's own comment in
`ggml-cuda/fattn.cu` says speculative verify batches (`n_q <= 8`) must stay on
the tile kernel because decode never uses WMMA.  So `n_max <= 7` **is the
designed guarantee**, and any configuration whose reduction tensors stay under
Cause A's crossover is pure all the way through it.  Removing Cause B would
mean giving up WMMA for 9..N-token batches.

The table below is the original `n_max` sweep (pre-fix).  Both builds in it
carried Cause A (block 12 was identical in them) and Cause B, so it shows their
*joint* effect; the GDN prefill boundary was not a factor in either.

| `--spec-draft-n-max` | none / 1 / 4 / 5 | 6 / 7 | 8 / 9 / 10 | 12 | 16 |
|---|---|---|---|---|---|
| delivered build (KTAIL=16) | equal | `5037ef2e` | `e721b8b5` | `e721b8b5` | `5037ef2e` |
| whole-batch chunked prefill | equal | `b6d86d62` | `ed922c76` | `5037ef2e` | `4f3ee41c` |

The two builds diverge in exactly the same place, so this is **not the GDN
prefill boundary's doing**: block 02's change fixed the *prefill* (probe
`RS=6 W=6 == RS=0 W=1` — the prefill state is now K-independent), and this cap
is Cause A + Cause B, both of which were present in both builds.

**Localisation (as measured).**  Both causes are pure *decode-batch-width*
effects; neither is `K`, and neither is the GDN.  At `RS=0` (no snapshots at
all, `K = 1`, so the GDN cannot be involved) the probe separates on width alone:

```
2-GPU tensor, pre-fix:  W = 1..6 -> a4817ee6   W = 7,8 -> e286b75c   W = 9 -> 24f302f6
```

i.e. adding columns to the batch changes column 0's own result.  `W = 7` is
Cause A.  `W = 9` is Cause B, and it coincides with `MMVQ_MAX_BATCH_SIZE = 8` in
`mmvq.cuh` (beyond 8 columns `ggml_cuda_mul_mat` leaves the vector kernels for
MMQ); both the FA WMMA gate and the MMVQ/MMQ crossover sit at that boundary, so
Cause B is fixed at `W <= 8` regardless of which of the two fires first.
`GGML_CUDA_GDN_CHUNKED=0` does not help (it is a prefill switch); block 13's
dense `ncols==1` ksplit alignment does not either — it aligns `ncols = 1` with
the `2..8` *verify* dispatch, and these boundaries sit above that.

**Is upstream affected?**  Cause B is upstream-inherited; Cause A is
fork-specific (upstream has no internal AR pipeline).  Upstream master's own
`calc_nwarps`/`calc_rows_per_block`
(`ggml/src/ggml-cuda/mmvq.cu`) switch on `ncols_dst`, and
`ggml_cuda_mul_mat` selects MMVQ only for `ncols_dst <= MMVQ_MAX_BATCH_SIZE`,
so any batched evaluation uses different kernels than a one-token decode.
Measured on **upstream master `9cf3bf256`** (clean checkout, unmodified
`mmvq.cu`), CPU backend, 4B Q8_0, `P = 256`: `W = 1` gives `9024dd2e...`
while `W = 2..12` all give `3cd0eb0e...` — upstream diverges at the *first*
width step and is therefore **worse** than the fork, not better.  (That build
was CPU-only, so upstream's ROCm boundary was not measured; the fork's
`n_max <= 7` is a fork *result*, not an upstream guarantee.)

**Practical consequence.**  With Cause A fixed the guarantee is `n_max <= 7` for
2-GPU `-sm tensor` too (it already applied to 1 GPU, 3-GPU tensor and
`-sm layer`).  Do not use `none == draft-mtp` byte-equality as a gate above
that; use acceptance + MTP-vs-plain throughput (see
`benchmarks/mtp-adaptive-methodology.md`).  Adaptive MTP's recommended
`n_max = 12` remains outside the *guaranteed* range by Cause B, which is
deliberate.  Records: `patches/README.md` block-12 notes, the 2026-09-11
WORKLOG entry, `wip/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.

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

qwen4exp (Qwen3.8-Flash-Next) has its own fused decode chain: the hyper-connection mixer
(`GGML_OP_HC_MIX`) and residual combine (`GGML_OP_HC_COMBINE`) replaced the unfused
`SCALE/SILU/MUL_MAT/SIGMOID/MUL/ADD` chain, but only for a **single-token** batch, so a 1-token decode
and an n-token verify batch took different arithmetic (measured in the per-node dump: 98
`HC_COMBINE` dispatches at `W=1`, **0** at `W>=2`).  Block 14, amended 2026-09-11, routes the whole
**decode/verify band `1 <= nt <= 8`** through the fused ops with the token index on `blockIdx.y`, so
every token in the band runs exactly the per-token kernel sequence a single-token decode runs.
Measured (3 GPUs, f16 KV, P=256, RS=0):

| split | W=1..4 | W=5 | W=6,7 | W=8 |
|---|---|---|---|---|
| `-sm layer`  | **all `3adeb313042a871b`** (= the W=1 decode) | `c999233926f0` | `a8c532e12f9c` | `c56ebb61963a` |
| `-sm tensor` | **all `dcf1ae667f730879`** (= the W=1 decode) | `2bfb89f59ec2` | `e8b1253ea93e` | `a7c5dfd26a56` |

So qwen4exp was width-pure for **`--spec-draft-n-max <= 3`** *at the time of this measurement*
(2026-09-11, block-14 amendment); **§15 fixes the `W >= 5` half, so the band is now `n_max <= 7`**.
The `W >= 5` grouping is **cause 2** (§11): the same
`ncols_dst`/`ne11` kernel-selection band **that was assumed to be the same site as the `q8_0`/`q4_0` KV
impurity in §12**.  That hypothesis was **refuted on 2026-09-11**: fixing F1 (§14) left cause 2's
`{5} {6,7} {8}` grouping completely unchanged, so cause 2 is a *separate* site (a matmul/MoE dispatch
band), not the FA kernel-family chooser.  Two caveats: a `<= 8`-token **prefill** chunk also takes the
fused path (indistinguishable from a verify batch — the point is that such a batch gets the decode
arithmetic); and with a `q8_0`/`q4_0` **KV cache** the cache's own impurity (§12) dominates, so the band
does not restore text equality there (the `W=1` decode is still unchanged).

## 14. F1 fixed (2026-09-11, block-08 amendment): the decode/verify band no longer spans two FA kernel families

§12's `q8_0`/`q4_0` impurity is fixed.  Root cause, found with a new kernel-chooser trace (committed for
reuse as `wip/kv-quant-purity-followups/tools/fa-kernel-chooser-trace.patch`, `GGML_CUDA_FA_TRACE=1`):
`ggml_cuda_get_best_fattn_kernel()` (`ggml/src/ggml-cuda/fattn.cu`) returned **VEC** for `n_q <= 2` with
a quantized K/V and **TILE** from `n_q = 3`.  The two families order the online-softmax/PV reduction
differently, so token-0 logits at `W = 1,2` disagreed with every verify width.  Both VEC conditions are
always inside the `n_q <= 8` band, so the branch is deleted and the band is TILE throughout — the same
shape as the block-08 WMMA guard (`Q->ne[1] > 8`) and block 00's `ntiles_dst_eff`.  The launcher plan
itself was already width-independent (`ntiles_dst_eff`; `parallel_blocks == ntiles_KV` at every width),
which is why every earlier F1 suspect measured clean.

| K/V cache (same type) | before §14 | after §14 |
|---|---|---|
| f16, bf16 | pure | pure, hashes **bit-identical** (they never took VEC) |
| q4_1, q5_0, q5_1, iq4_nl | pure | unchanged (see the F3 reframing below) |
| **q8_0** | `W=1,2` != `W=3..8` | **`W=1..8` bit-identical in all four split configs** |
| **q4_0** | same shape as q8_0 | **same** |

Measured (3x gfx1201): 4B `q8_0/q8_0` 1 GPU `W=1..8` all `31a0c1bace68`, 2-GPU `-sm tensor` `abebfb93`,
3-GPU `-sm tensor` `7fe106f5` (`q4_0`: `619c151e48c7` / `240bc37d` / `483a850e`); 27B Q8_0 3-GPU
`W = 1,2,3,4,5,8` all `d4156dbeb225`.  Text level (27B, 300 greedy tokens, q8_0 KV): plain ==
`n_max 3` == `n_max 7` = `3537bc2b36be` (was `73b2565bce47` vs `3537bc2b36be`); f16 control
`f32aac948600` for both.  Every new value equals that config's **previous verify** value: only
`W = 1,2` moved, so MTP is bit-unchanged (27B acceptance 0.90789 identical) and the cost is
decode-only — tg128 -0.9% (4B) / -0.5% (27B), pp512 ~-0.2%, reserves byte-identical.

**The pure range is still `n_max <= 7`** (the verify batch is `n_max + 1` and the designed FA
tile-vs-WMMA switch sits at `Q->ne[1] > 8`) — now for *every* supported KV type, not just the float
ones.

**F3 reframed.**  §12's "no native FA kernel" explanation is wrong in detail: `q4_1`/`q5_0`/`q5_1`/
`iq4_nl` are rejected by `ggml_cuda_fattn_kv_type_supported()` unless the build enables
`GGML_CUDA_FA_ALL_QUANTS` (OFF here), so `ggml_cuda_get_best_fattn_kernel()` returns `NONE` *before*
any VEC/TILE choice (0 `[FATPATH]` lines for `q4_1`, 1+ for `q8_0`) and attention takes the generic
fallback — width-invariant by construction (hence pure) and ~3.4x slower.  F3's first experiment is a
build-flag A/B of `GGML_CUDA_FA_ALL_QUANTS=ON` (plus §14's band rule, which keeps any newly-enabled
native path width-invariant).

**Harness lesson (three false positives in one session).**  `llama-cli`'s `/\|` spinner is `\b`-based
and timing-dependent, and the ASCII banner embeds the build SHA: apply backspaces, strip the banner and
the `[ Prompt: ... | Generation: ... ]` footer before hashing output — and always run a control that
must agree (the f16 pair) before believing any divergence.

## 15. F2 cause 2 fixed (2026-09-11, block-13 amendment): the MoE decode/verify band is band-uniform

§13's `W >= 5` grouping is fixed, and the *mechanism* is **not** the gate+up+GLU fusion coverage the
first pass blamed — it is upstream's **per-type mmvq cap** (`get_mmvq_mmid_max_batch_*`), which does two
things:

1. it sizes `mul_mat_vec_q_moe`'s `__launch_bounds__` (`cap × warp_size`) while the block is
   `(warp_size, ncols_dst)` — so it is a *capability* limit (launching `IQ3_S`, cap 4, with
   `ncols_dst = 5` is 160 threads > the bound and aborts with `unspecified launch failure`);
2. it chooses mmvq vs MMQ (`ggml_cuda_mul_mat_id`: `ne2 <= cap → mmvq`, else `should_use_mmq → MMQ`)
   and gates the `mul_mat_q_pair` fusion (`use_mmvq`, `ggml-cuda.cu:3730`) — which is what ran at
   `W = 5..7`.  **mmvq and MMQ reduce in different orders**, so every cap boundary inside the band is a
   numeric boundary.

This is a *graph-identical* bug: `[GD]` full-graph dumps give the same node counts (2647/2404/2271/1863/
1668/1565) at `W=4` and `W=5`, with `MUL_MAT_ID(ffn_moe_gate)` / `MUL_MAT_ID(ffn_moe_up)` /
`GLU(ffn_moe_swiglu)` at the same indices in both.  The quant is what makes it visible: the UD-IQ4_XS
file mixes expert types per layer (47 layers `IQ3_S` gate/up → cap 4; layer 2 `IQ4_XS` → cap 5; down
`IQ4_NL`/`Q8_0` → cap 7), which predicts the observed grouping **exactly** — fused layers
48/48/48/48/1/0/0/0 for `W = 1..8`, i.e. the 4→5 and 5→6 boundaries, and the down's cap 7 for 7→8.

**Fix** (block 13 amendment; it completes block 13's own `has_ids` "decode == verify invariant"):
floor the cap at `MMVQ_MAX_BATCH_SIZE` for every AMD lookup and size the MoE kernel's launch bound at
the band.  All cap call sites are `MUL_MAT_ID`-only ⇒ dense models cannot be affected (verified).

| split | W=1..8 (f16 KV, P=256, RS=0) |
|---|---|
| `-sm layer`  | **all `3adeb313042a871b`** (= the pre-fix W=1 decode) |
| `-sm tensor` | **all `dcf1ae667f730879`** (= the pre-fix W=1 decode) |

Every width moved onto that split's **pre-fix `W = 1`** value: plain decode is bit-unchanged, and only
`W = 5..8` moved (the F1 "move the cheap side" pattern).  Also pure with `RS=from_w`.  At the MTP gate
config (`n_max 3` = `W=4`, a no-op width) pre/post-fix runs are **byte-identical** (acceptance 0.76744,
80.0 vs 80.1 t/s) — the fix provably does not touch what already worked — and at `n_max 7` it is
**+16-18 % t/s** with acceptance 0.59375 vs 0.55556; `n_max 3` == `n_max 7` text (`8a50ea24e8d5`) where
they previously disagreed (`8a50ea24e8d5` vs `e6918a7af1f9`).

**The fix is also a large throughput win** (`llama-batched-bench`, interleaved, swappable
`libggml-hip.so`; fixed/baseline):

| model | b1 | b2 | b4 | b5 | b6 | b7 | b8 |
|---|---|---|---|---|---|---|---|
| qwen4exp 3-GPU `-sm tensor` | 50.5/50.4 | 85.4/85.6 | 134.1/132.9 | **149.5/118.4** | **162.5/130.6** | **171.4/147.0** | **178.0/155.4** |
| 35B-A3B MoE 1 GPU | 98.3/98.1 | 156.4/156.2 | 254.2/254.1 | – | – | – | **341.3/289.9** |
| 4B dense 1 GPU | 100.6/100.5 | 167.2/167.3 | 294.0/294.9 | – | – | – | 414.5/413.3 |

So upstream's per-type mmvq caps were costing 14-26 % at exactly the speculative-verify widths on RDNA4
with the fork's mmvq + fused-GLU kernels.

**Cause 3 (open).**  `plain` still differs from `draft-mtp` text (`plain` `3ee9daee5c07` vs
`n_max 3 == n_max 7` `8a50ea24e8d5`) — and this fix cannot be responsible: at `n_max 3` (`W = 4`) it is
a verified no-op (bit-identical logits, byte-identical text, byte-identical acceptance).  A control
confirms the harness is deterministic and that plain decode is unchanged (pre-fix and post-fix plain
text are both `3ee9daee5c07`).  Because the single-step width probe is bit-identical across `W = 1..8`
on both splits *and* with the state-sequence dimension (`RS=0` and `RS=from_w`), the divergence must be
a **multi-step / roll-back** effect — **localised 2026-09-11 (further measurement): it is in the QSA *machinery*, and the site class is the same as cause 1's.**  `LLAMA_QSA_OFF=1` makes `plain` == `draft-mtp --spec-draft-n-max 3` **byte-identical** (`d4499ac8db72` both, 711 chars) — and the knob provably fires (the plain text moves `3ee9daee5c07` -> `d4499ac8db72`) — while `LLAMA_QSA_SPARSE_FA=0` (dense attention, indexer still on) leaves two different texts (`25f300a81b9e` vs `0d466b2dcf09`), so the defect is **not** the sparse-FA kernel but the **indexer/score machinery** (`indexer-topk.cu` + the `qwen4exp.cpp` gates).  Both QSA-side `n_tokens == 1` gates are the prime suspects — `src/models/qwen4exp.cpp:1094` (`idx_score_fused`, the fused indexer score) and `:1419` (`qsa_dense_decode_until`, the early-decode dense shortcut) — i.e. exactly the cause-1 pattern, and the single-step width probe cannot see them because it never reaches the sparse/indexer decode regime.  The divergence appears only after ~100 chars (~20 tokens) of a 3.3k-prompt greedy run (the first steps agree), so it is not a prefill-state difference; `GGML_CUDA_GDN_CHUNKED=0` moves both sides without making them agree (the known Issue #25 chunked-prefill item is a separate contributor, not this).  **Kill-switch for users meanwhile: `LLAMA_QSA_OFF=1`.**

## 16. Cause 3 fixed (2026-09-11, block-14 amendment): the QSA decode arm is band-uniform

After §13-§15 the dense models, the MoE and qwen4exp's hyper-connection band were pure, but qwen4exp
still had a **text** divergence: `--spec-type none` and `draft-mtp --spec-draft-n-max 3` / `7` shared
only ~100 of ~700 generated characters (f16 KV, `/tmp/prompt3k.txt`, 128 greedy tokens, 3-GPU
`-sm tensor`).  It was **not** a kernel and **not** the sparse-FA attention:

* `LLAMA_QSA_OFF=1` makes both runs **byte-identical** (711 chars), and the knob provably fires (the
  plain text moves `3ee9daee5c07` → `d4499ac8db72`), so the defect lives in the QSA **indexer**
  machinery (store / score / top-k selection);
* `LLAMA_QSA_SPARSE_FA=0` does **not** fix it (both texts move, both stay different) — the
  `fattn-qsa` kernel is exonerated;
* the single-step width probe is **pure** on both splits and with `RS=from_w`: its blind spot is
  exactly this bug (it prefills `P <= 2048` tokens and decodes one step, so `n_kv` stays below the
  QSA selection width and every width takes the same arm).

**Mechanism** (`build_layer_attn`, `src/models/qwen4exp.cpp`).  The indexer picks one of three arms:

```cpp
if (shortcut && n_kv <= width)                                        // 1: dense, store keys
else if (qsa_dense_decode_until > 0 && n_tokens == 1 && n_kv < ...)   // 2: dense policy arm
else  top_k = build_qsa_top_k(...);                                   // 3: sparse selection
```

`width = indexer_top_k + r - 1`; on qwen4exp `indexer_top_k = 2048`, `r = 4` (only every 4th layer has
an indexer), so `width = 2051`.  From the arm trace at the first decode graph: `n_kv = 2304 > 2051`, so
arm 1 no longer applies — and arm 2 is gated on `n_tokens == 1`:

| run | shape | arm |
|---|---|---|
| `--spec-type none` | `n_tokens=1 n_kv=2304` | **2 (dense)** |
| `draft-mtp n_max 3` | `n_tokens=4 n_kv=2304` | **3 (sparse top-k selection)** |

Same state, two attention regimes, purely because of the batch width.  (Both runs are identical for the
first 11 graph builds; the split starts at the first decode graph.)

**Fix**: `QSA_DECODE_BAND = 8` (the `n_max <= 7` purity band, the same constant class as
`HC_FUSED_MAX_TOKENS`), and arm 2 takes `n_tokens <= QSA_DECODE_BAND` instead of `n_tokens == 1`.
Prefill is untouched (`n_tokens` is far above the band, so it keeps arm 3 — the arch policy "prefill is
untouched, QSA always"), and on gfx1201 `qsa_dense_decode_until = 1 << 62`, so decode is now dense at
every width — which is the policy the arm's own comment describes.  Post-fix arm trace: `n_tokens=4
n_kv=2304` gives **arm 2** in *both* runs.

**Measured**: `plain == n_max 3 == n_max 7` = `804de0576868` (704 chars, f16 KV) and `plain == n_max 3`
= `75d8530c5bb1` (660 chars, q8_0 KV).  The plain stream moved with the fix (658 → 704 chars): the fix
also moves the shared 4-token non-decode shape at `n_kv = 2304` onto the dense arm, which is the same
"the band must take one path" trade as §14 — the *value* chosen is the policy-consistent dense one.

## 17. The MoE shared-expert epilogue is band-uniform too (2026-09-11, block-13 amendment)

§15 fixed the `W >= 5` split; what remained for Qwen3.6-35B-A3B was strictly `W=1 ac8825358d9adfda` vs
`W>=2 bd138ad2326fbbf2` — the fused shared-expert down epilogue (`dst = down(swiglu) *
sigmoid(gate(x)) + moe_out + ffn_residual`, a 6-node fusion in `ggml-cuda.cu`), gated

```cpp
down_mm->src[1]->ne[1] == 1 && gate_mm->src[1]->ne[1] == 1; // decode only
```

with an in-code note that the fused gate reduction (`shexp_gate_sigmoid`) does not reproduce the
standalone mmvq/MUL_MAT order — so `W=1` ran the fused epilogue and `W>=2` the unfused chain.

**Fix**: the two kernels are now token-generic, and the band takes the *fused* path (which keeps the
+3.1 % decode win instead of throwing it away):

* `shexp_gate_sigmoid` — one block (one warp) per token: the token index only selects the input column
  (`grid: (ncols)`), so each token's dot is computed by the same 32-thread warp reduction as before;
* `shexp_down_gated_q8_0` — one block per `(output row, token)` (`grid: (nrows, ncols)`), addressing
  `y_swiglu`, `moe_out`, `ffn_residual` and `dst` at the token's offset;
* **`nwarps` is pinned to the single-token value** (`calc_nwarps(type, 1, table_id)`): `calc_nwarps`
  returns 4 for `ncols_dst 1..4` but 2 for `5..8`, and `nwarps` sets `blocks_per_iter`, i.e. the
  reduction order — the same trap as §15's cap.  Pinning it is what makes every width bit-identical;
* the fusion arm accepts `1 <= ne[1] <= MMVQ_MAX_BATCH_SIZE` (with both matmuls the same width and the
  three epilogue operands contiguous); `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` still selects the unfused
  reference.

**Measured**: default probe `W = 1, 2, 3, 4, 8` all `ac8825358d9adfda` (the pre-fix `W=1`/fused value);
with the kill-switch all of them `bd138ad2326fbbf2` (a uniform unfused reference, = the pre-fix `W>=2`
value); qwen4exp was unaffected (`plain == n3 == 804de0576868`).  The asterisk is gone: the MoE decode
and verify batches now take one arithmetic, so the `n_max <= 7` guarantee covers MoE too.

## 18. Two width-dependences remain in the QSA *sparse* regime (open, 2026-09-11)

§16 fixes the **default** regime, which is what gfx1201 (and gfx1151 below its 64K crossover) runs.
Two further items were found while localising it and are **open**:

1. **The fused indexer score is not byte-identical, and it is itself `n_tokens == 1`-gated.**
   `GGML_CUDA_QSA_INDEXER_SCORE` (default **ON**, `src/models/qwen4exp.cpp` — the `idx_score_fused`
   gate at the `build_qsa_top_k` entry) replaces the per-op score chain with one kernel and claims to
   "replicate the per-op F32 arithmetic byte-identically".  Measured **false**: with the sparse regime
   forced on both widths (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`), `GGML_CUDA_QSA_INDEXER_SCORE=0` changes
   *both* streams (`361c9daa7ed8`/`9959840e2747` → `14924014c66f`/`e5c7a3f91699`).  Because the gate is
   `n_tokens == 1`, a W=1 decode and an n-token verify consume the indexer state through different
   score paths.  It is **unreachable on gfx1201's default config** (the plain run never builds scores
   at all — the dense arm skips `build_qsa_top_k`), but it is the default path on **gfx1151 above the
   64K crossover**, where decode is sparse.  (This also explains why the earlier `SCORE=0` test in the
   default regime was a *legitimate* null: neither run executes the score path there.)
2. **A residual split survives even with one arm.**  With `LLAMA_QSA_DENSE_DECODE_UNTIL=0` (both
   widths on the sparse arm) `plain` and `n3` agree for 706 chars instead of 100 — i.e. the arm split
   was the first cause — but they still diverge, so at least one more width-dependence lives in the
   sparse-regime state path (indexer store / derived-cache pooling).  `GGML_CUDA_QSA_INDEXER_CACHE=0`
   alone does not reconcile them (in the default regime that knob is a no-op: the dense arm never fills
   or reads the derived cache).

Consequence: the exhaustive greedy-purity guarantee is complete for **RDNA4/gfx1201's default regime**
and for the dense models and the MoE; the **gfx1151 sparse regime above 64K** still needs the two items
above.  Repro knobs: `LLAMA_QSA_DENSE_DECODE_UNTIL=0`, `GGML_CUDA_QSA_INDEXER_SCORE=0`,
`GGML_CUDA_QSA_INDEXER_CACHE=0`; the arm trace is kept at
`wip/kv-quant-purity-followups/tools/qsa-arm-trace.patch`.
