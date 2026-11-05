# Issue #272 — a partially reused prompt invalidated the drafter's KV (r218)

**Reported by @hermeticvm** ([issue #272](https://github.com/stew675/llama-cpp-rdna-boosts/issues/272)):
Qwen3.6-35B-A3B + a `draft-dspark` drafter, RX 6700 XT (gfx1031 via `HSA_OVERRIDE_GFX_VERSION=10.3.0`), two
slots, unified KV, `-ctxcp 4`.  The third request of a three-turn SWE-bench replay answered
`HTTP 500 decode() failed: failed to process speculative batch`, with

```
E init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 1995
 - the tokens for sequence 0 in the input batch have a starting position of Y = 2242
 it is required that the sequence positions remain consecutive: Y = X + 1
```

`--spec-type ngram-mod` alone was fine, `cache_prompt: false` was fine, `-ctxcp 0` was fine (it also disabled
reuse), and their upstream control build was fine.  They could not tell whether the mechanism was theirs alone.

**It is ours, it is not RDNA2-specific, and it is the block-12 DFlash/DFlash2 prompt-ingest tail (r130).**

## Root cause

`common_speculative_impl_draft_dflash::process()` ingests the *target's* rows into the draft's own KV cache, and
since r130 it ingests only the last `n_swa` rows of the prompt (`LLAMA_DFLASH_INGEST_TAIL`; the MTP drafter has
the same skip as `LLAMA_MTP_INGEST_TAIL`, block 02 r118).  That is exact — a row below the window of the prompt's
end can never be attended (`GREEDY_PURITY.md` §53) — but it assumes the draft's cache stays **contiguous** with
the rows it does ingest, because `llama_batch_allocr::init()` requires the first ingested position to be the
cache's `seq_pos_max + 1`.

A prompt is routinely only *partially* reused: the slot keeps a prefix from an earlier task.  When a turn adds
more than `n_swa` new tokens, the new prompt's window starts **past** the rows the draft kept, so the draft's
cache ends at the reused prefix (P-1) while its first ingested row is at `prompt_end - n_swa > P`.  Two batch
rules then decide the outcome:

* **strict** (`n_pos_per_embd == 1`, i.e. a DFlash/DSpark draft with no `dflash.rope.dimension_sections` — the
  `NORM`/`NEOX` backbone class): `Y = X + 1` is required, the batch is rejected, and the request fails;
* **M-RoPE** (`n_pos_per_embd == 4`): a forward jump is legal, so the batch runs and its first rows attend to
  nothing (a draft-quality hole, not a failure).

The reporter's numbers fit exactly: `Y - X - 1 = 246` rows skipped, `Y = prompt_end - n_swa` with `n_swa = 2048`,
and `X + 1 = 1996` = the "1996 cached tokens" their upstream control reported.

## Reproduction (gfx1201, this box)

No DSpark drafter exists for a model here, so the reporter's draft **class** was reproduced by derivation: copy
the 35B-A3B DFlash2-Q8_0 draft and zero its `dflash.rope.dimension_sections` (`[64,0,0,0] -> [0,0,0,0]`), which
switches `llama_model_rope_type()` from `MROPE` to `NEOX` and therefore `n_pos_per_embd` from 4 to 1.  The
derived copy is *measurably the same model*: identical text and identical draft acceptance on every turn
(0.42424 = 14/33, 0.20833 = 10/48, 0.35135 = 13/37), i.e. a one-section M-RoPE rotation is numerically NeoX.  Only
the batch-check rule changes, which is exactly the variable under test.

Server: `35B-A3B UD-Q4_K_M` + that draft, `--spec-type draft-dflash -ngl 99 -ngld 99 -fa on -c 49152 -np 2
--kv-unified -ctxcp 4 --jinja -b 1024 -ub 512 --spec-draft-n-max 4`.  Replay: three turns of one `/completion`
prompt, each adding one full `prompts/prose-rdna-boosts.txt` (5246 tokens, far more than the 2048 window),
`cache_prompt: true`, greedy, `ignore_eos`.

| arm | pre-fix (r217) | post-fix (r218) |
|---|---|---|
| strict draft, tail on | **HTTP 500 on turn 2**: `inconsistent sequence positions ... X = 5241 ... Y = 8445` | 200 / 200 / 200 |
| strict draft, `LLAMA_DFLASH_INGEST_TAIL=0` | 200 / 200 / 200 | 200 / 200 / 200 |
| shipped (M-RoPE) draft, tail on | 200 / 200 / 200 | 200 / 200 / 200 |
| shipped (M-RoPE) draft, tail off | 200 / 200 / 200 | 200 / 200 / 200 |

Text is byte-identical between the tail and no-tail arms on all three turns in both draft classes, on both builds
(cached 0 / 5242 / 10516, new 5246 / 5251 / 5224 per turn) — the drafter proposes and the target verifies, so a
draft-cache policy can move acceptance (it does not here) but never a token.  The same four arms pass with
`GGML_LF_DFLASH_DEV=0`, i.e. through the host injection path (`llama_process`) rather than F1's device path
(`llama_decode`).

`LLAMA_DFLASH_INGEST_TAIL=0` passing on the pre-fix build is the isolation: **the tail creates the gap.**

## Fix (r218, blocks 02 and 12)

`spec_draft_align_cache()` in `common/speculative.cpp`, called by both drafters' `process()` right before the
injection batch is built:

* **draft behind the batch** (the tail case above, or a checkpoint restore that skipped the task forward): a
  strict (non-M-RoPE) draft drops its stale prefix — every row of it is older than the window of the prompt's end,
  so no later row can attend it, and the batch that would have carried those positions is long gone — and starts
  a fresh window at the first ingested row, exactly as it does for a first-ever prompt.  An M-RoPE draft keeps the
  jump it already tolerated (its rows are attended while they are still inside the window), so **nothing changes
  on that path**;
* **draft ahead of the batch** (a restore can put it past the position the task resumes from): the overlapping tail
  is trimmed in every case, because **both** batch rules reject that; this call re-ingests those positions.

No new environment variable and no default changes: the realign is a no-op unless the cache would otherwise be
invalid.  A `gemma4`-style cache shared with the target is never touched (`is_mem_shared`).

78 lines: the helper plus the two call sites.  `scripts/gate-spec-reuse-window.sh` is the gate (pre-fix it fails
with the issue's own message; post-fix it passes) and is registered in `MANIFESTS.md`, `scripts/README.md` and the
`AGENTS.md` gate table.

## Gates (r218; `build-base` = r217, `build-rocm` = r218, one box)

| gate | result |
|---|---|
| `gate-spec-reuse-window.sh` | **pre-fix FAIL** (strict arm, turn 2, `X = 5241 / Y = 8445`), **post-fix PASS**: 12/12 requests answered, tail == no-tail byte for byte, tail armed at 2048 in both tail arms; also PASS with `GGML_LF_DFLASH_DEV=0` |
| coherence, same-seed greedy | text **identical in 6/6**: dense 4B `1c5d32ac537d`, 27B + `draft-mtp n3` `d45e2221805a`, 27B + DFlash2 `d45e2221805a`, 35B-A3B + DFlash2 `0cf4a62aa63f`, 35B-A3B `-ncmoe 40` `0cf4a62aa63f`, 35B-A3B 3 GPU `-sm layer` `0cf4a62aa63f` |
| MTP four-axis (dense 27B UD-Q4_K_XL, 1 GPU, `-c 32768`, bf16 KV, `-n 2000`, `draft-mtp-adaptive`, seed 42, reasoning per axis) | text **identical 4/4**: R 9243 `0ffad0c34953`, P 5830 `4f0d1ea854e9`, C 8966 `95603a6a89a4`, K 2174 `a87c4318b649`; t/s R 54.5/54.5, P 71.1/71.4, C 75.1/75.1, K 83.0/82.6; acceptance identical at `-n 400 -lv 4`: **0.74932 (275/367)**, acc per pos **(0.837, 0.756, 0.642)** |
| `gate-spec-draft-window.sh` (the MTP drafter is touched) | PASS: window 1024 cells/seq, cache 3072, **6/6** requests identical to `LLAMA_SPEC_DRAFT_WINDOW=0` |
| `gate-ckpt-reuse.sh` (the checkpoint's draft buffer is the other gap cause) | PASS without a draft (keep 3, created 5 < 8) and **with the 35B-A3B DFlash2 draft** (keep 3, created 5 < 8, text == in 4/4) |
| prefill logits (`gate-prefill-logits.sh`) | **KLD 0.000000, same-top-p 100.000 %** |
| per-block build (`--fast --changed-from 541a94a32`) | **3/3** (block 02 78 s, block 12 6 s, the tip); the other 13 block deltas are byte-identical to r217 |
| GPU faults (`check-gpu-faults.sh --since "3 hours ago"`, 20 run logs) | clean |
| `validate-set.sh` | green — checksums, fresh-tarball strict **16/16**, applied tree `d362b5d67` == `release.json.tree` |

## Not validated

* **The MTP half of the realign.**  Every MTP-head model on this box (Qwen3.8-27B, Flash-Next, qwen4exp) is
  M-RoPE, so the forward-jump path is legal and the MTP call is a no-op by construction; it is recorded as
  documented-not-measured.  The gates above (four-axis, draft window, ckpt-reuse) are its inertness evidence.
* **A real DSpark drafter** (no Qwen3.6 DSpark GGUF here; the derived non-M-RoPE copy stands in, as justified
  above), and therefore **RDNA2/gfx1030** itself — the defect is in `common/` and cannot be device-specific, but
  it was reproduced on gfx1201 only.
* **A multimodal (M-RoPE image) prompt with the tail** — unchanged by this fix (the realign is gated on the token
  path, because an embedding batch may legitimately pin all of its rows to one position).
