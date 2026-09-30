# `wip/issues-59-60` — gfx1100 candidate fix for issues #59 / #60

**WIP branch, not a release, not tagged.**  **Read `FINDINGS-2026-09-29.md`** for the reporter's
original data and `SESSION-2026-09-30-r27-wi2.md` for the r27 re-integration and the #60 fix.
`ANALYSIS.md` is the original root-cause analysis.  Issue **#59 is closed** (confirmed fixed by the
reporter); **#60 is the open one**.

## Where this branch is now (2026-09-30, r27 + #60 fix)

* Rebased/re-integrated onto `main` = **r27** (`7d68a31`; block 15 now also carries PRs #68/#73/#74/#75
  and the issue-#71 fix).  The whole branch is **one squashed commit on top of r27**; the pre-r27
  head is kept as tag `wip-59-60-pre-r27`.
* Fork tip **`8aa6d0443`**, tree **`a8c7380efdabd167a2bc93318d6c6e364460791e`**.
* `patches/` (16 patches) and `release.json` are regenerated; release string
  **`v16-84e76d8a2-wip-issue59-60`** (no tag, not on `main`).
* `scripts/validate-set.sh` strict **16/16** on a fresh `84e76d8a2` tarball.

## The fix

1. **qsa3 off on RDNA3_0 by default** (`ggml/src/ggml-cuda/fattn-qsa3.cu`), `GGML_CUDA_QSA3=1` opt-in,
   `=0` force-off.  `fattn-qsa.cu` makes a packed op's support equal the qsa3 predicate, and the
   qwen4exp graph probes the packed op through the backend before building the two natural-F16 K/V
   packs, so disabling qsa3 frees that memory too.  RDNA3_5/RDNA4 defaults unchanged.  (Closed #59.)
2. **Geometry-aware fused-vs-chain prefill score** (open #60).  `ggml_cuda_lightning_indexer_supported`
   no longer rejects the 4-head shape on RDNA3_0.  In `build_qsa_top_k`'s `use_wmma`, auto uses the
   fused op on RDNA3_0 once `score_bytes` exceeds **`LLAMA_QSA_SCORE_WMMA_MB` MiB (default 64)** and
   the faster chain below it; `0` means always fused.  For the reporter's `r=4` geometry the default
   is `n_kv` about 64K, which keeps the chain at their d30K/d64K points and bounds its reserve before
   the 196K prefill ceiling.  `LLAMA_QSA_SCORE_WMMA=0/1` forces the chain/fused op.  The unfused
   chain's `mul_mat+relu` and chunked `mul_mat+relu` now use `ggml_relu_inplace`, removing the
   chain's 2x-score reserve peak (bit-identical).
3. **FA prefill staging-arena accounting** in `llama_get_memory_breakdown` / `--fit` (bounded by
   `2 x GGML_CUDA_FA_STAGE_MAX_MB` and the F16 attention-KV size).

## Reproducing

```bash
git clone https://github.com/stew675/llama-cpp-rdna-boosts
cd llama-cpp-rdna-boosts && git checkout wip/issues-59-60

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 84e76d8a2
bash ../llama-cpp-rdna-boosts/scripts/apply-all.sh .
# build for gfx1100 (GPU_TARGETS=gfx1100)
```

## Open work

* The reporter's 196K gate on the r27 head (no qwen4exp model locally).  Expected: the default now
  switches to the fused op at `n_kv` about 64K, so the 196K prefill fits; the chain still covers
  their d30K and d64K points.
* Promotion to `main` only after that gate passes.
