# qwen4exp branch recreation on the 9cffdcc80 re-base (2026-09-02)

## Result

`~/llama.cpp` `qwen4exp` @ `236002e4a` = **master `9cffdcc80` (new tip) + the
13 re-based rdna-boosts blocks + the 15 qwen4exp commits + the managed-ngrams
7-commit set + 2 rebase-fix commits**. Working tree clean, NOT pushed.

Canonical source used: `~/qwen4exp.cpp` (the surviving full-history repo;
`qwen4exp` @ `9ed77c905` = old master `0eadefebd` + the 13 original blocks +
15 qwen4exp commits). Fetched as a local remote (`qwen4exp-src`) and
cherry-picked 1:1 (history preserved).

## Branch contents (top to bottom)

```
236002e4a qwen4exp: re-allow LLAMA_SPLIT_MODE_TENSOR        <- rebase fix
a0e8a430b qwen4exp rebase fixes: CPU INDEXER_TOPK ref, hc_mix
          type gate, reader F32 path                       <- rebase fixes
6dd87c2d1 qwen4exp: apply managed-ngrams lazy-reader work   <- 0001-0007
8b6faa7f1 .. 739079f56 (15 qwen4exp commits, cherry-picked 1:1)
8f2838d1c .. 04122bfb5 (13 re-based blocks)
9cffdcc80 (new master tip)
```

## Conflicts resolved

1. `qwen4exp.cpp` (0ff151bda = ITEM B): the new master added an `n_kv_max`
   param to `build_attn_mha`; the QSA sparse branch kept + the fallback call
   got the new `0` arg.
2. Managed-ngrams 0003 (`qwen4exp.cpp` PLE load block + includes): the new
   master already carries the user-path PLE row derivation (upstream mtmd
   work) and its own test fixture, so the managed branch was merged onto the
   upstream's block instead of duplicating.
3. Managed-ngrams 0004 (`test-llama-archs`): the upstream fixture PLE keys
   supersede the patch's duplicate; kept the upstream fixture + applied the
   `n_lazy_buf_size` roundtrip plumbing.

## Rebase validation fixes (a0e8a430b + 236002e4a)

The new master's `test-llama-archs` qwen4exp fixture (upstream 36b101543)
surfaced three gaps the old tree never exercised (the fixture is upstream-only):
1. **CPU INDEXER_TOPK reference** (ops.cpp/ggml-cpu.c): the fused indexer
   top-k op was GPU-only; any CPU graph build aborted. Added a reference with
   the GPU radix semantics (threshold = k-th largest value; above-threshold
   cells then boundary ties, each in cell order).
2. **hc_mix type gate**: the fused op asserts Q8_0 hc weights; synthetic
   (F32/F16) models now fall back to the generic chain.
3. **llama_lazy_reader F32 path**: the reader aborted on F32 tables (no
   to_float trait); wired the planned memcpy copy path.
4. **llama-arch.cpp**: upstream excluded qwen4exp from `SPLIT_MODE_TENSOR`
   with a TODO; re-allowed (the whole decode campaign ran --split-mode
   tensor; the test's Meta row now passes NMSE 1.01e-13).

## Validation (all green)

- Build clean (llama + tests + server).
- test-lazy-reader: all tests passed.
- test-llama-archs qwen4exp: MoE OK on 3x R9700 (NMSE 9.7e-14) + CPU
  (0.00) + Meta tensor-split (1.01e-13) + ROUNDTRIP OK (the managed-reader
  roundtrip). Full arch matrix: 607 OK / 0 fail.
- Real model (UD-Q4_K_XL, 3x R9700, --split-mode tensor):
  - K=1 single-seq decode = `198 42750 367 367...` = **byte-matches the
    pre-rebase reference** (the decode campaign's numerics survived intact).
  - K=2 same prompt: seq0 == seq1 == the K=1 stream (the multi-seq layout
    fix = intact).
  - K=2 unequal longer prompts (6/5 tokens): both seqs == their K=1 runs
    token-for-token.
  - Managed-ngrams determinism: `--lazy-buffer-size` run == mmap run
    **byte-identical** (verified even at a tiny budget; the reader stats
    confirm the managed path is live: 512 misses, rows read via pread).

## Known follow-up (not a blocker)

- K=2 multi-seq with the short 3-token prompt "The answer" drifts from its
  K=1 single-seq stream at decode step 3 (369 9542 11 694... vs 369 9542 13
  198...), while seq0 == seq1 (self-consistent, not the layout bug) and the
  longer prompts match exactly. The single-seq gate is untouched; the drift
  is a numeric divergence of the multi-seq batch path for short odd-length
  prompts, most likely from upstream's 36b101543 (block-position keying /
  per-seq cell changes) interacting with the multi-seq decode. Investigate
  if the multi-seq path must be bit-equal to single-seq for every prompt
  length.

## Stale-harness lesson

The old /tmp harness binaries (bseq_val etc.) were built against the
pre-`n_lazy_buf_size` llama_model_params; the struct grew, so the stale
binaries read params at the wrong offsets and the model load appeared to
crash ("n_layer_all=0 / vocab_only garbage" in load_hparams). Rebuild any
harness that links libllama whenever llama_model_params changes.
