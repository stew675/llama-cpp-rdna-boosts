# UPSTREAM-PR: ggml-cuda: keep the decode/verify band on one flash-attention kernel family

**Status:** candidate, written and validated against the delivery (canonical tip `1bcf4e82d`); not yet
filed.  Per the scope policy in `AGENTS.md` ("RDNA first, other backends uninjured"), the fork lands
only the AMD-reachable half of this fix and leaves the NVIDIA arms to this PR — that is a deliberate
split, not an omission.  The fork-side form of this fix is the 2026-09-11 block-08 amendment
(`patches/0008-rdna-boosts-block-08-fused-core-prefill-kernels-and-.patch`); this copy is the
self-contained upstream form (it also carries the NVIDIA/Ada arm the fork does not need — see Scope).

**File:** `ggml/src/ggml-cuda/fattn.cu` — `ggml_cuda_get_best_fattn_kernel()`, two hunks, net -7 lines.

## The bug

The chooser's VEC ("generic vector kernel") heuristics are tuned per batch size, but every condition
they use is satisfied only for `n_q <= 2`, i.e. always *inside* the decode/verify band that
speculative decoding depends on:

```c
} else {                                    // quantized K or V
    if (cc >= GGML_CUDA_CC_ADA_LOVELACE) { if (Q->ne[1] <= 2) return BEST_FATTN_KERNEL_VEC; }
    else                                 { if (Q->ne[1] == 1) return BEST_FATTN_KERNEL_VEC; }
}
...
} else {
    if (Q->ne[1] <= 2) { return BEST_FATTN_KERNEL_VEC; }
}
```

So with a `q8_0` (or `q4_0`) KV cache, `n_q = 1,2` runs the VEC kernel and `n_q >= 3` runs a
tile/MMA kernel.  The two families order the online-softmax + PV reduction differently, so a one-token
decode is **not bit-identical** to an `n_draft + 1`-token verify batch: the same finite-precision
prefix produces slightly different logits, and greedy sampling then flips near-ties.  The visible
symptom is that `--spec-type none` and `--spec-type draft-mtp` produce different tokens — with a
quantized KV cache only (an f16/bf16 cache never takes VEC here, which is why this is easy to miss).

## The fix

Both conditions only ever fire inside the band, so drop them: the whole band uses the tile/MMA family.
Nothing about prefill changes (large batches already fell through to TILE).  Cost is decode-only, and
small: on gfx1201 with `q8_0` KV, tg128 -0.5 % (27B) / -0.9 % (4B), pp512 ~-0.2 %; and because the new
`W = 1,2` values are exactly the values the *verify* widths already produced, speculative decoding
throughput is unchanged (acceptance bit-identical).  A band-consistent family costs a little
single-token throughput; a split band costs greedy equivalence, which spec decoding is built on.

## Scope

* The patch also removes the non-quantized `n_q == 1 && !gqa_opt_applies` VEC return, which has the
  same defect (VEC at `n_q = 1`, TILE from `n_q = 2`) in configurations we could not instantiate here
  (no mask / ALiBi / MHA).  It is a strict consistency win; if a maintainer prefers to keep that
  heuristic, guarding only the quantized arm is enough to fix the reported symptom.
* **Left alone deliberately:** the Ada `n_q == 1` VEC shortcut in the tensor-core branch
  (`cc >= ADA && Q->ne[1] == 1 && ...`) is a *float* decode-throughput heuristic that also splits the
  band.  Removing it costs measured decode perf on Ada and is a maintainer trade-off call, not an
  obvious bug fix.
* The same class of defect exists in the surrounding width-dependent choices (`8/ncols2`, `16/ncols2`
  instantiation switches, `ntiles_dst`-derived split counts); the tile-launcher one was fixed in the
  fork by `ntiles_dst_eff` (see `UPSTREAM-PR-fa-kv-split-width.md`, already written up here).
* **Why the NVIDIA arm here matters more since 2026-09-11:** the delivery now enables
  `q4_1`/`q5_0`/`q5_1` as flash-attention KV types (`GGML_CUDA_FA_ALL_QUANTS` is no longer required;
  see the block-08 notes and `GREEDY-PURITY.md` §20) and adds their three diagonal vec instances.  On
  an Ada+ NVIDIA part those types therefore take `VEC` at `n_q <= 2` and the MMA family above it —
  i.e. they now inherit exactly the split this PR removes, whereas before the enablement they were
  rejected by the support predicate and never reached flash attention at all.  (The fork's own fix
  scope is AMD, where the Turing/Volta branches are unreachable dead code: both predicates require
  `GGML_CUDA_CC_IS_NVIDIA(cc)`, and on RDNA4 the band reaches the fallback, which now returns TILE
  unconditionally — `amd_mfma_available` is CDNA-only and the WMMA branch is gated `Q->ne[1] > 8`.)

## Repro

Any CUDA/HIP build, a model whose KV cache is `q8_0`/`q4_0`, and a seed-fixed greedy run compared
between no speculation and `--spec-type draft-mtp --spec-draft-n-max 3..7`.  A one-token-logit hash
probe over batch widths `W = 1..8` is the sensitive instrument (text comparison is not: it is common
for two different token streams to coincide, and a `\b`-based progress spinner plus a build-id-bearing
banner will fabricate differences — strip both).

## Validation (3x R9700, gfx1201, ROCm 7.14, unpinned)

* `q8_0/q4_0` `W = 1..8` bit-identical on all four split configs (1 GPU / 2-GPU layer / 2-GPU tensor /
  3-GPU tensor): 4B `31a0c1bace68` / `abebfb93` / `7fe106f5` (`q4_0`: `619c151e48c7` / `240bc37d` /
  `483a850e`), 27B `d4156dbeb225`.
* Text level (27B, 300 greedy tokens, `q8_0` KV): plain == `n_max 3` == `n_max 7`; f16 control
  unchanged.
* f16/bf16 hashes bit-identical before/after; MTP acceptance bit-identical (0.90789, n = 96);
  `test-backend-ops -o FLASH_ATTN_EXT` 4591/4591 (4/4 backends) and `-o GATED_DELTA_NET` 46/46;
  reserves byte-identical.

## How to file

Re-create the branch from current upstream master, re-apply
`UPSTREAM-PR-fa-decode-verify-kernel-family.patch` (it applies cleanly to upstream `9113cc188`),
re-run the repro, and reference the sibling `UPSTREAM-PR-fa-kv-split-width.md` candidate.
