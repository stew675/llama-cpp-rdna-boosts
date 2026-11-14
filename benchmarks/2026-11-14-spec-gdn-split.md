# The speculator set chose the chunked-GDN prefill split (issue #309, r282)

**Verdict.** `plain`, `draft-mtp` and `draft-mtp-adaptive,ngram-mod` did not reproduce the same greedy
text, and the cause was not the draft, the sampler, the model file or the KV cache: it was
`GDN_CHUNKED_MIN_TOKENS` being floored with `n_rs_batch`, the widest draft any enabled speculator can
produce.  The DRAFTLESS n-gram family is unbounded (`--spec-ngram-mod-n-max` defaults to 64), so merely
adding `ngram-mod` to `--spec-type` moved the chunked-GDN prefill tail from 16 tokens to 65 and changed
the state the prefill handes to the decode, with no draft ever being accepted.  The threshold is now the
snapshot set only, `max(K > 16 ? K : 16)`, and the wide draft is kept.

## Report

GitHub issue #309 (slavap): a Bartowski `Qwen3.8-27B-Q8_0` on 3 x R9700, `-sm tensor -ts 1,2,2`,
`--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 40 --spec-draft-n-max 4`, deep
agentic context.  The same command on the Unsloth Q8_0 of the same model was reported as fine.  In the
server log the throughput jumps from ~56 t/s to ~390 t/s with `draft acceptance = 0.99329 ... mean len =
55.28` and a 32 000-token answer, i.e. the model settles into a self-reinforcing echo.  The reporter
suspected the `speculative checkpoints of the target in host memory: the target is not on exactly one
GPU` notice; that notice is benign (it only says where the spec checkpoint lives) but it is printed at
the first draft deeper than `n_rs_seq`, which is the same moment the n-gram speculator starts drafting
deep, so it is a marker of the regime rather than its cause.

Two file differences were checked and neither is the cause:

| | Unsloth Q8_0 | Bartowski Q8_0 |
|---|---|---|
| `blk.N.ssm_alpha/beta.weight` (49 layers) | Q8_0 | **F32** (the whole 69 MB file-size delta) |
| `tokenizer.ggml.add_bos_token` | absent (=> true) | `false` |
| `tokenizer.ggml.padding_token_id` | 248055 `<|vision_pad|>` | 248044 `<|endoftext|>` |
| chat template | `tojson` on tool arguments | `tojson \| safe` |
| tokens / merges / token_type / arch / 866 tensors | identical (hashed) | identical (hashed) |

`padding_token_id` is not read outside the vocab (grep), so it cannot matter; and the defect reproduces on
BOTH files, so it is not the quant at all.

## Mechanism

`ggml/src/ggml-cuda/gated_delta_net.cu` (block 10 owns the fixed-D tail, block 03 the kernel) selected
between the chunked and the sequential GDN kernel with

```c
const int64_t GDN_CHUNKED_MIN_TOKENS = std::max<int64_t>(K > 16 ? (int64_t) K : 16, n_rs_batch);
```

and `common` sets `cparams.n_rs_batch = common_speculative_n_max() + 1`, which includes
ngram-simple/map/mod/cache in `common_speculative_n_max()`.  A single-sequence batch above the threshold
chunks its first `n_tokens - D` tokens and finishes the last `D` on the sequential kernel
(`GGML_CUDA_GDN_TAIL`, r91).  So `D` is both the tail length and the chunked/sequential boundary, and
`D` moved from 16 to 65 when `ngram-mod` joined the speculator set.  The chunked kernel is not bit-exact
with the sequential one (fp32: rounding order; bf16/WMMA: materially lossy), so the post-prefill state
differed and the greedy trajectory diverged.

`GGML_CUDA_GDN_CHUNKED=0` (sequential everywhere) collapses the difference, which is the identification:

```
LLAMA_SPEC_NGRAM_DEPTH_CLAMP=0 (wide draft, pre-fix threshold)
  ngram-mod n_max   4 -> 5ecb4b287875   == plain
  ngram-mod n_max   8 -> 5ecb4b287875   == plain
  ngram-mod n_max  15 -> 5ecb4b287875   == plain     (n_rs_batch 16, threshold 16)
  ngram-mod n_max  16 -> 777f1bb14276   differs     (n_rs_batch 17, threshold 17)
  ngram-mod n_max  64 -> be86ee8fc4b8   differs     (n_rs_batch 65, threshold 65)
  with GGML_CUDA_GDN_CHUNKED=0 all of them -> 339e89ba336b (one text)
```

The flip is exactly at `n_rs_batch = 17`: the threshold is a constant 16 for `K <= 16`, and starts
following `n_rs_batch` above it.

## Why the threshold does not need `n_rs_batch`

Upstream bounds a partial rollback per cell (`llama_memory_recurrent::prepare()`):

```c
cell.rs_max = n_seq_tokens <= n_rs_seq + 1 ? n_seq_tokens - 1 : 0;
```

and `seq_rm()` only performs a partial rollback when `rollback <= cell.rs_max`.  So:

* a batch with more than `K = n_rs_seq + 1` per-seq tokens **cannot be partially rolled back at all** -
  the caller restores a full state checkpoint instead, which is what the tools do whenever a draft is
  deeper than `n_rs_seq` (the server/cli `spec_ckpt_*` path, the very notice the reporter saw);
* a batch that CAN be rolled back has `n_seq_tokens <= K <= D`, so it is at or below the threshold and
  takes the sequential kernel, which writes every snapshot slot it may read.

Both halves hold with `D = max(16, K)`, i.e. the snapshot set, and neither depends on the draft width.

Provenance: the `n_rs_batch` floor was added 2026-09-12 (`WORKLOG.md` 260912.010) when the chunked path
had **no** sequential tail, so a whole-batch chunked run wrote no snapshots and any rollback restored an
unwritten plane (`test-recurrent-state-rollback`, max diff 6.5366).  The fixed-D tail landed at r91 and
made the floor redundant for the snapshots, but it stayed, and the r129 verification that looked at it
compared text on one model where no near-tie flipped.

## What shipped

`GDN_CHUNKED_MIN_TOKENS = K > 16 ? K : 16` (block 10), with the reasoning above in the kernel's comment
block, the tail length and `n_rs_batch` both in the `-lv 4` census, and no new environment variable.  The
draftless n-gram width is untouched (no clamp): it is a throughput knob again.

## Gates (gfx1201, ROCm 10.1.0, `~/bin/build-llama-rocm`; 3 x R9700, `-sm tensor -ts 1,2,2` unless noted)

| arm | pre-fix | post-fix |
|---|---|---|
| `scripts/gate-spec-gdn-split.sh` (Unsloth Q8_0, 4 speculator sets) | **FAIL** tail `16/16/16/65`, text `5c381ee010d7 != fffe3d3d88fe` | **PASS** tail `16` in all four, `n_rs_batch` `0/5/16/65`, text `fffe3d3d88fe` in all four |
| the same with `MODEL_ALT` = Bartowski Q8_0 | (not run pre-fix; same code) | **PASS** tail `16` throughout, text `fa09a6c3519d` in all four |
| copy-prompt n-gram mean accepted length | 53.33 (default) / 16.00 (bounded) | unchanged: the wide draft still drafts deep |
| verbatim-copy decode, `draft-mtp-adaptive,ngram-mod` | 480.3 t/s (impure) | **528.2 t/s and pure**; the 15-token clamp read 252.4 |
| prose prompt, `plain` vs the n-gram set | — | 36.5 vs 96.3 t/s (the win), texts identical |
| `test-llama-context -m Qwen3.5-4B -c 1024` | — | 17/18, 1 skip, **0 fail** (`rollback`, `rollback_bound` both OK) |
| `test-backend-ops -o GATED_DELTA_NET` | — | **55/55** |
| `test-recurrent-state-depth -c 1024` | 73 failures | **byte-identical** (its `n_rs_seq >= 8` Phase-A failures are the documented 8-row verify trade, `K <= 16` either way) |
| `scripts/gate-prefill-logits.sh` | 0.000697 / 98.784 % | 0.000697 / 98.784 % (unchanged) |
| `scripts/gate-ckpt-reuse.sh` | — | **PASS** (all four requests' text equal, 5 < 8 checkpoints) |
| coherence, plain path | 4B `ae0ca6cd1b53`, 27B 3-GPU prose `b2179f193df2` | **identical** (the fix is inert without a deep draft) |
| `llama-batched-bench -npl 1,4,8` (4B) | — | 101.7 / 340.4 / 511.6 t/s at `npp 512` (neutral) |
| `gate-per-block-build.sh --fast --changed-from 69296dde5` | — | **2/2** (block 10 279 s, tip 8 s) |
| `validate-set.sh` | — | green, tree `7bb5113de5` |

## Reproduce

```bash
# the defect (pre-fix build), and the fix's own control arm
scripts/gate-spec-gdn-split.sh                        # PASS/FAIL as above
MODEL_ALT=/llm/models/Qwen3.8/27B/Bartowski-Q8_0/Qwen3.8-27B-Q8_0.gguf scripts/gate-spec-gdn-split.sh

# the mechanism in one pair
GGML_CUDA_GDN_CHUNKED=0 scripts/gate-spec-gdn-split.sh          # every arm identical even pre-fix
```

The prompt pair is `prompts/prose-rdna-boosts.txt` (ordinary prose: rejections, the rollback path) and
`prompts/recall.txt` (verbatim copy: deep drafts, the win), both sha256-pinned in `prompts/README.md`.
