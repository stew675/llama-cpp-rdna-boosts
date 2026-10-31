# PR #258 — quantized-KV decode rotations (`wip/kv-rotation-decode-r184`) — PROMOTED in r187

Submitter: `bsp2020`.  Five patches (`0001`-`0005`, +1238 lines) cut against the r184 tree, delivered as
`wip/kv-rotation-decode-r184/` in their fork.  **Accepted and folded into block 10** (block 11 takes the
one-line `fattn-qsa.cu` call-site update, because that file is created there) — released as
`v16-6050b467c-r187`, block 10 amended, no block renumbered.

## What it does

With `-ctk/-ctv q8_0`, `build_attn` runs a Hadamard rotation on Q and K (after their ROPE), on V (before the
cache write) and on the attention output (before the projection).  Those extra nodes broke every decode fusion
the delivery has for an f16 cache, so the arm ran **726 kernel launches per token against 510** (Qwen3-30B-A3B).
The series restores them, one default-on term each:

| patch | term | what |
|---|---|---|
| 0001 | `GGML_CUDA_QK_ROTATE_HADAMARD` | the `GGML_CUDA_QK_ROTATE` pass takes Q's rotation and **defers V's** (the Hadamards, the views they read, everything downstream) behind the Q chain, projections in front.  1-token graphs only; skipped if a node left in front would read a deferred one |
| 0002 | `GGML_CUDA_FUSE_V_FWHT` | the fused K kernel rotates the V row itself (`ggml_cuda_fwht_warp<64, 32>` + the q8_0 group quantize map), killing the standalone `fwht_cuda` |
| 0003 | `GGML_CUDA_FA_COMBINE_FWHT` | at a 1-token FA whose combine kernel runs, `flash_attn_combine_results<D, true>` stages the row in shared memory, applies the same FWHT, stores it and publishes the projection's Q8_1; on any other path the rotation runs as before and no cache entry is created |
| 0004 | `GGML_CUDA_QK_PAIR_FWHT` | Q's norm+rope+rotation deferred to the V-rotation + K site and run as one `rms_norm_mul_rope_f32_qk_pair<.., fwht_n, true>` launch |
| 0005 | `GGML_CUDA_FUSE_ROPE_FWHT` | a ROPE with no norm in front (Llama-style) runs as `rms_norm_mul_rope_f32<.., no_norm = true>` + the K/V cache writes.  **Any token count, so prefill takes it** |

All five key on the cache `SET_ROWS` being `GGML_TYPE_Q8_0` — the quantizer they reproduce.  Other KV types keep
their launches: `TODO.md` #120.

## Assessment (gfx1201, 3x R9700, ROCm 10.1.0; full numbers in `benchmarks/2026-10-31-kv-rotation-decode.md`)

* **Claims reproduce.**  launches/token **726 -> 510** (exactly the submitter's number); Qwen3-30B-A3B-class
  q8_0 KV tg128 **141.2 -> 160.1 (+13.4 %)**, @4096 **127.3 -> 140.1 (+10.1 %)** against the PR's +10.8 / +10.7
  (gfx1100/gfx1201); gpt-oss-20b (the rope-only arm) **+4.0 % / +6.1 %**.
* **F16 KV unchanged** — 162.67 -> 163.10 tg128, 138.72 -> 138.82 @4096.  The submitter's flagged -0.8 % F16
  "code layout" effect does not reproduce here.
* **Purity**: greedy text byte-identical on six arches at 1 GPU, on 3-GPU `-sm tensor` and `-sm layer` (two
  models) and on a `-ncmoe 16` host-expert arm; MTP four axes at `-n 3000` identical text **and** acceptance to
  the last digit; prefill logits `0.000000 / 100.000 %` against the recorded bases for both f16 and q8_0 KV; the
  per-fusion `--ab` A/B is at the harness floor (the off-vs-off control reads the same 99.990 %); width probe
  (KV=q8_0, `P=3584`/`P>=4096`, 3 runs) bit-identical to the all-switches-off reference, single device and
  multi-device; `test-backend-ops` ROPE/SET_ROWS/FLASH_ATTN_EXT 7059/7059; no GPU faults.
* **Gaps in the submission, covered here**: it never ran multi-GPU or `-sm tensor`; it never ran the prefill-logit
  gate (it ran wikitext KLD + a `-b 1 -ub 1` decode-shaped check, which we reproduced); and it did not exercise
  the rope-only form on a model whose prefill reaches it other than Llama-3.2-1B (gpt-oss does here).
* **Two code nits, neither a defect.**  (1) `0002` computes the V source offset by subtracting `data` pointers;
  `0003` replaces that with `v_set_rows->src[0]->view_offs` plus an assert that the V view's `view_src` is the
  rotation — the fold lands that final form, so the weaker intermediate never ships.  (2) The FA request is a
  file-scope `thread_local` (`ggml_cuda_fa_had`) where the context already carries the analogous
  `qk_pend_graph/_i/_j`; a context member would be more consistent.  Cosmetic; left as submitted.
* **Judgement recorded (maintainer, 2026-10-31)**: include `0005` (the only prefill-affecting patch) — it is
  bit-identical in the prefill-logit A/B and the width probe, and worth +4-6 % decode on gpt-oss-20b.

## Fold mechanics

The five applied to the r186 tree with offsets only.  In the block chain, `git diff` of the five was applied
into **block 10** (`0d917affa`), the author/date preserved, blocks 11-15 replayed on top, and the
`fattn-qsa.cu` one-liner amended into **block 11** (the block that adds that file).  The resulting tree is
byte-identical to the validated arm (`7cd527bde8…`), and the re-cut via `scripts/apply-all.sh` produced the
canonical tip `9ee1862a6` (tree `7cd527bd`).

Rules a future edit must keep: `GREEDY-PURITY.md` §74.  Variable catalogue: `ENVIRONMENT.md` §7.4.  Block
reference: `patches/README.md` (block 10 "The quantized-KV decode fusions (r187, PR #258)", block 11
"`had_dst`").  Release record: `WORKLOG.md` 261031.005.
