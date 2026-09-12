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

**The bug, in one line:** block 15's V2/V3 refactor added an *outer* `ggml_tensor * kq_mask_top_k =
nullptr;` in `build_attn_qsa` while the top-k mask chain *inside* the new `if (kq_mask != nullptr) { ... }`
wrapper kept its own `ggml_tensor * kq_mask_top_k = ggml_set_rows(...)` — a **new local** that shadowed the
outer one.  The chain was therefore built whenever the mask existed, and the attention
(`build_attn_mha(q, k, v, nullptr, kq_mask_top_k, ...)`) read the outer one: `nullptr`.  The dense masked
arm attended with no mask at all — a full causal leak, seen as PPL `1.0558` on qwen4exp for every KV type
where the delivery gives `6.49-6.55`, and as ≈`1.02` on *random* text where a working build gives `18.4`.

**Four generalisable lessons:**

1. **A graph tensor with no consumer is silently dropped.**  The chain's nodes were unreachable from the
graph output, so `ggml_build_forward_expand` never emitted them; with no consumer the allocator left the
packed mask unallocated, and block 15's own `if (self_kq_mask && self_kq_mask->buffer)` guard in
`llm_graph_input_attn_kv::set_input` then skipped `set_input_kq_mask` entirely.  So "the input is created
in the graph" proves nothing about it being *filled* — and an unfilled mask is indistinguishable from a
correct one until you measure quality.  A defensive check (assert the mask was filled when a consumer
exists) is worth considering for the beta.
2. **When an executed-graph dump is *missing* nodes, suspect the graph builder, not the allocator.**  The
   `[ND]` node dump showed the delivery's dense prefill consuming `attn_inp_kq_mask` 36 times (12 indexer
   layers × 3 devices) and emitting the `FILL`/`SET_ROWS`/zeros chain, while the beta consumed it **zero**
   times and emitted **no** `FILL` at all.  Dead code is absent from the graph *by construction* — that is
   the signature, and it points straight at the builder.
3. **`-Wshadow` would have caught this class outright.**  The fork does not enable it.  Adding it (at
   least for `src/` on the CI path) would make this whole failure mode a compile error; the fix itself is
   one token.  Filed as a follow-up.
4. **The leak instrument to reach for first is random text.**  A model that can see the target predicts
   *anything* — including noise — near-perfectly: `llama-perplexity -f /tmp/rand-text.txt --chunks 1
   -c 2560 -b 2560 -ub 2560` gave the broken build `1.0205` and the delivery `19.0589`.  Natural, or even
   repetitive, text is a *bad* leak detector (a repetitive prompt scores ≈1 in a perfectly healthy build,
   which sent this session down a false trail until random text settled it).  Keep both the oracle and the
   random-text probe in the gate list; they answer different questions ("is the fused path as good as the
   dense one?" vs "is the attention causal?").

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
