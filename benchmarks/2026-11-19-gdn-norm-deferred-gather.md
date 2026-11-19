# The GDN conv + q/k norm fold on a deferred conv-state gather (r320, PR #316)

**Question.** The fold is refused whenever a SCALE output overlaps the conv state. With a deferred
conv-state gather the kernel reads the *cache row* (`state_ids`), never `m.state->data`, so that refusal is
not a correctness rule. How much does it cost, and where is it reachable?

**Answer.** It is reachable in every speculator verify arm that does **not** take the rollback `SET_ROWS`
store, at a verify width of 6 or more, and it is worth **0.02 %** of decode there. It is unreachable in
`draft-dflash`, `draft-dflash,ngram-mod` and `draft-mtp-adaptive`, which take the store.

## Environment

* 1x Radeon AI PRO R9700 (gfx1201), ROCm 10.1.0, `HIP_VISIBLE_DEVICES=0` (single device; no tensor-split arm)
* Builds: the r319 canonical re-cut (`f6ffa8042`) and the r320 tree (`3334a4eb31`, gated before the release
  commit added `Co-authored-by` to the block-10 message; the message change does not move the tree), same source path
* Models: `Qwen3.8-27B-UD-Q4_K_M` + `DFlash2-Q4_K_M` (the PR's own scenario), `Qwen3.8-27B-UD-Q4_K_XL`
  (four-axis), `Qwen3.6-35B-A3B-UD-Q4_K_M` (MoE)
* Prompts: `prompts/recall.txt`, `prompts/prose-rdna-boosts.txt`, `prompts/reasoning.txt`,
  `prompts/code-python.txt`

## Reach: the matcher census (r318 tree, one line per distinct (T, deferred, base verdict))

`GGML_CONV_NORM_DEBUG=1` in `gdn_conv_norm_match()`; `base=` is what the *delivery* decides (the change not
applied).

| arm (27B Q4_K_M) | verify widths that reach the matcher | base verdict |
|---|---|---|
| `--spec-type none` (plain decode) | `T=1..4` | folded |
| `draft-dflash --spec-draft-n-max 7` | **none** (`GDN_CONV_STORE T=2/4/8/5/6/3/9`) | n/a |
| `draft-dflash,ngram-mod` (field config) | **none** (store) | n/a |
| `draft-mtp-adaptive --spec-draft-n-max 9` (35B-A3B) | **none** (store `T=2/4/7/6/5`) | n/a |
| `ngram-mod --spec-ngram-mod-n-max 7` | `T=2,3,4,8` | `T=8`: **overlap_state** |
| `ngram-mod --spec-ngram-mod-n-max 15` | `T=2,4,11,16` | `T=11,16`: **overlap_state** |
| `ngram-cache --spec-draft-n-max 7` | `T=2,4,5,6,9` | `T=6,9`: **overlap_state** |
| `ngram-map-k --spec-draft-n-max 7` | `T=2,4,16` | `T=16`: **overlap_state** |
| `draft-simple --spec-draft-n-max 7` | `T=1..8` | `T=6,7,8`: **overlap_state** |
| prefill (`T=42/44/512`) | enters, `deferred=0` | refuses on `x`/state: legitimate |

`test-logits-width-probe MODEL prose-rdna-boosts.txt 4096 512` with `KV=q8_0` reads
`T=6,7,8 deferred=1 base=overlap_state` - that is the arm the r289 review found, and it is the only one that
does not need a speculator.

## The change (block 10, `ggml/src/ggml-cuda/gdn-conv.cu`)

`gdn_conv_norm_match()` takes the backend context and, exactly when the conv state's root is in
`ctx.gdn_deferred_gathers`, tests the **cache tensor** (`root->src[0]`) instead of `m.state` in the overlap
loop. `x`, `w` and the conv output are unchanged, and so is the `T=1` decide (`gdn_conv_decode_ok()` still
calls the matcher without a context).

## Correctness

* Greedy text, r319 vs r320, `-n 512`, seed 42, temp 0, 1 GPU: **identical** on all five arms -
  `ngram-mod` n15 `ba147eef7e7c`, `ngram-cache` `ba147eef7e7c`, `draft-simple` `ba147eef7e7c`,
  `draft-dflash` `ba147eef7e7c` (27B) and `draft-mtp-adaptive` `a1783d90d107` (35B-A3B).
* `test-logits-width-probe` `KV=q8_0`, `P=3584` and `P=4096`, 27B **and** 35B-A3B: `width_purity=PASS`,
  worst maxdiff 0, and every `W=1..8` hash plus both row0/row1 hashes **byte-identical r319 vs r320**.
  Also identical to `GGML_CUDA_DISABLE_CONV_FUSION=1` and to each fold kill switch
  (`GGML_CUDA_FUSE_GDN_CONV_GATHER=0`, `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1`,
  `GGML_CUDA_FUSE_GDN_CONV_DECODE=0`, `GGML_CUDA_FUSE_GDN_CONV_NORM=0`) on the 27B at `P=4096`.
* `test-backend-ops -o SSM_CONV,GATED_DELTA_NET,RMS_NORM`: **236/236**.
* Four-axis MTP Protocol A (1 GPU, 27B Q4_K_XL, bf16, `-n 3000`): plain vs `draft-mtp n3`, r319 == r320 on
  every cell; acceptance **0.60747 / 0.87198 / 0.90924 / 0.99200** and t/s 64.0 / 79.4 / 84.5 / 90.9 both
  builds (plain 31.7 / 31.2 / 31.7 / 31.9 both).
* `gate-prefill-logits.sh` (27B Q8_0, canonical point): **KLD 0.000000, same-top-p 100.000 %**.
* `llama-batched-bench -npl 1,4,8 -npp 512 -ntg 128` (27B Q4_K_M, 1 GPU): TG **154.48 / 424.06 / 576.00**
  (r319) against **154.44 / 423.97 / 575.80** (r320), i.e. noise.

## The win, and why it is not the reason to ship this

`--spec-type ngram-mod --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-match 45`, 27B Q4_K_M, `-n 2048`, greedy,
5 interleaved pairs, wall clock (the `[ Prompt | Generation ]` footer only resolves 0.1 t/s, which is too
coarse here):

| arm | mean wall | vs r320 |
|---|---|---|
| r319 (refuses `T=11,16`) | 38 692.4 ms | +0.018 % |
| **r320 (folds them)** | **38 685.6 ms** | - |
| r319, `GGML_CUDA_FUSE_GDN_CONV_NORM=0` | 39 262.4 ms | +1.49 % |

`ngram-map-k`, `-n 1024`: base 67.1, patched 67.1, fold-off 66.0 t/s. The refusal costs a launch plus one
read/write of the conv output per GDN layer on the verify steps whose draft is 6 or more tokens deep, and the
ngram draft is that deep only occasionally, so the end-to-end effect is under the resolution of the
measurement.

**A note on the fold-off arm**: `GGML_CUDA_FUSE_GDN_CONV_NORM=0` also changes the `T=1` decide (the direct
conv is only taken there when the norm pair folds too), so 1.49 % is an upper bound on the norm fold's worth
in this arm, not a like-for-like baseline.

## What this is NOT

It is not the PR's claim. `draft-dflash` never enters the matcher on r244 and later: `gdn-conv-store.cu`'s
one kernel writes `q_out`/`k_out` itself, so the "the q/k norms run as a separate kernel in every GDN layer of
every verify step" premise is false for the arm the PR measured, and the -0.7 ms/cycle number comes from a
2026-11-03 tree that predates that kernel (r244, 2026-11-09). See `INVARIANTS.md` r320 and `WORKLOG.md`
261119.004.
