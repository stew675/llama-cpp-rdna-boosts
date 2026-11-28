# wip/pr362 - contributor PR #362: the RDNA3 padded GQA-6 tile fold (assessment)

**Status: OPEN - assessment complete 2026-11-25, the submission was REJECTED as-is and a rework request was
posted on the PR. Nothing here is in `patches/`, `release.json`, the default tree, the fork branch or any
release gate, and nothing here may be applied to a checkout: `pr-submitted.patch` and `assessment-delta.patch`
are a review record, not candidates.**

**What the PR is.**  `optional/gqa6-fold-rdna3/` (author @kapojko), one source change of +122/-27 in
`ggml/src/ggml-cuda/fattn-tile.cuh`: for a head-256 / GQA-6 / native-q8_0-K-V shape at query width 1..8, the
tile launcher stops selecting `ncols2 = 2` (three logical KV traversals per query row) and pins
`ncols1 = 1, ncols2 = 8` instead (one traversal, columns 6/7 zero-Q and never stored).  Submitted default
**ON** with `GGML_HIP_FA_TILE_GQA6_FOLD=0` as a kill switch, gated to `gfx1101`/`gfx1102`, cut against the
r353 tree.  PR: <https://github.com/stew675/llama-cpp-rdna-boosts/pull/362>.

| file | what |
|---|---|
| `pr-submitted.patch` | the submitted artifact patch verbatim, sha256 `0768b60b3b33e71237c309d4c64b9a9ea7fc382696919715c071a91d991761d7` |
| `assessment-delta.patch` | the review delta (sha256 `bd118045f101a6021bd696c1d8b5158594d2778eff24a534dcbda82eca2f28d7`), a diff from commit `75d821e1f` (the release tip `003c88e1f` + `pr-submitted.patch`) |

## Why it was rejected as submitted

Two blockers, both about the code *around* the fold rather than the fold itself.  Both are measured.

### B1. The fold code changes the codegen of the SHARED tile instantiations

The fold is a runtime branch (`gqa_ratio` is not compile time), so it is compiled into every
`flash_attn_tile` instantiation.  That moves the register allocation, which moves the occupancy the launcher
derives the decode/verify KV split (`parallel_blocks`, i.e. the summation order) from.  The fold does not even
have to be reachable for this to bite:

| arm, decode width, 4096-cell KV, `--kv q8_0 --ub 1` | gfx1201 (RDNA4 box) | gfx1100 (this box) |
|---|---|---|
| fold **off** vs an unpatched r359 build, mean KLD | **0.000391** | **0.000000** |
| fold off, max KLD | 0.0161 | 0.000060 (= the logits file's own noise floor, self-compare 4.7-5.9e-5) |
| fold off, same-top-p | **98.97 %** | **100.000 %** |
| fold off, greedy text | flips vs unpatched | identical (4/4 runs, 2 of them the untouched binary) |
| fold off, tg64 d2048 / d8192 / d32768 | 32.01 -> 31.90 (flat) | 52.81 / 51.59 / 47.08 against 52.68-52.83 / 51.49-51.64 / 47.04-47.09 (**untouched**) |
| fold off, FA op | +1.4 to +5.7 % cost at every shape (fold unreachable) | not separable from noise |

So on gfx1100 the off arm really is inert, and on **gfx1201 it is not**: `llvm-readobj` on the RDNA4 box reads
VGPR 140 -> 94 on `flash_attn_tile<256,256,8,2,...>` (others move too), the occupancy-derived split changes, and
decode logits move at the shallow depths where the RDNA4 band does **not** cover (the band owns
`K->ne[1] * passes_saved >= 20000`, i.e. from ~10 000 cells).  `GGML_HIP_FA_TILE_GQA6_FOLD=0` therefore does not
restore the original path, and a default-off artifact would still charge every GQA-6/head-256/q8_0 consumer on
gfx1201 an attention-time cost and an invisible numerics shift for a feature that cannot engage there.

That is an arch-dependent codegen accident, so it has to be removed **by construction**, not measured away:
see R1 below.

### B2. The prefill-logit evidence is structurally blind to the change

`scripts/gate-prefill-logits.sh` defaults to **f16 KV** (the fold needs native q8_0, so it cannot run at all) and
scores ubatch-wide rows (`Q->ne[1] > 8`, so the fold is inert).  The submitted "mean KLD 0.000000, same-top-p
100.000 %" is therefore guaranteed without the patch and is not evidence.

The point that *can* see it is the decode width: `--kv q8_0 --ub 1`.  Measured here (gfx1100, 27B IQ3_S,
`--ctx 512 --chunks 2`), fold on vs fold off: **mean KLD -0.000001, same-top-p 100.000 %, max KLD 0.000054**.
So the change is *rounding-level*, not approximate, and the greedy text still moves at a near-tie - which is why
it can only ever be an opt-in.

## What the delivery measured on gfx1100 (the value case)

Single RX 7900 XTX, Qwen3.8-27B IQ3_S, `-ctk q8_0 -ctv q8_0`, same binary, env flip, 2 runs per arm:

| arm | fold off | fold on | delta |
|---|---|---|---|
| tg64 @ d2048 | 52.68 / 52.83 | 52.55 / 52.64 | **-0.3 %** |
| tg64 @ d8192 | 51.49 / 51.64 | 51.71 / 51.80 | **+0.4 %** |
| tg64 @ d32768 | 47.04 / 47.09 | 48.58 / 48.65 | **+3.3 %** |
| 32K-token prompt, plain, 128 greedy tokens | 47.1 | 48.6 | +3.2 % |
| 32K-token prompt, `draft-mtp n3` | 61.0 / 61.0 | 76.7 / 76.7 | +25.7 %, **but** acceptance 0.531 -> 0.685 (mean accepted len 2.57 -> 3.02), so the kernel's own share is about **+7 %** |

The MTP row is a methodology trap, not a kernel win: the acceptance re-roll (a near-tie class) supplies most of
it.  Any MTP arm quoting this feature must print both arms' acceptance, and the plain arms are the honest ones.

Numerics mechanism, measured with a temporary print in `launch_fattn` (not kept): fold off = `ncols2 2,
ntiles_z_gqa 3, max_blocks_per_sm 18, P 18`; fold on = `ncols2 8, ntiles_z_gqa 1, max_blocks_per_sm 6, P 6`.
Block b owns KV iterations b, b+P, ... and the combine sums the P partials in block order, so P **is** the
summation order (`GREEDY-PURITY.md` 99/104).  The fold's numerics change is not padded-column bookkeeping.

Correctness on gfx1100 with the fold ON: the delivered `FLASH_ATTN_EXT` suite passes **7471/7471, 0 FAIL** (its
head-256/GQA-6 block does reach the fold: 384 cases, 48 of them q8_0; the census fires on them), and the added
coverage below passes **515/515**.  Windows path and portability checks pass on the patch.

## RDNA4: not a target, and no fold needed there

From the RDNA4 assessment (1x Radeon AI PRO R9700, gfx1201):

* the fold is a genuine kernel win **if** the arch gate is widened: attention op -18.6 % at 4096 cells, -28 % at
  8192, flat from 16384 up (`0.0 / +0.3 / +0.1 %` at 16384 for widths 1/3/8);
* but the band owns every shape past ~10000 cells and the tile window below it is not attention-bound, so
  end-to-end plain decode is **flat** (+-0.5 %) and the MTP-3 arm as shipped reads **-3.7 %**;
* band vs tile+fold at 16384 cells: width 1 band 92.1 us against tile+fold 79.5 us; width 3 band 94.7 us against
  tile+fold 198.6 us.  The band gate cannot be lowered to "use the fold" without breaking the
  one-kernel-per-decode/verify-family rule, and it should not be.

So the arch gate stays `RDNA3_0`.  RDNA4 is a non-regression obligation only, which is exactly what B1 breaks.

## Rework required before this can be looked at again

R1. **Make the fold a template parameter** (`flash_attn_tile<..., type_KV, /*fold=*/false>`) so every pre-existing
     instantiation compiles to pre-patch code, and add ONE `fold = true` instantiation for
     `(DKQ 256, DV 256, ncols1 1, ncols2 8, GGML_TYPE_Q8_0)` (one generated TU in `template-instances/`, generated
     by `generate_cu_files.py`; precedent: the delivery's own `fattn-mma-f16-band-*` instances).  `warp_valid` and
     `ncols2_eff` then plumb through as `if constexpr`, which also removes two implicit couplings: the current
     `fold_gqa = kv_is_native && ncols2 > gqa_ratio` only works because the launcher picks `ncols2` as a divisor of
     `gqa_ratio`, and `ncols1_eff = ncols/ncols2_eff` truncates (8/6).  Acceptance evidence for R1: the off path is
     KLD 0.000000 / 100 % at `--kv q8_0 --ub 1`, identical text, and the op-level tile timing unchanged, **on
     gfx1201 as well as gfx1100**.
R2. **Default OFF with an opt-in env** (`GGML_HIP_FA_TILE_GQA6_FOLD=1`), catalogued in `ENVIRONMENT.md` as an
     opt-in, not a kill switch.  It moves greedy text by construction.
R3. **Arch gate** `GGML_CUDA_CC_IS_RDNA3_0(cc)` (gfx1100/1101/1102/1103).  gfx1100 is the delivery's validated
     RDNA3 part and the band is RDNA4-only, so that is where the fold is reachable.  gfx1151 (RDNA3_5) out until
     measured; RDNA4 out on the measurements above.
R4. **Packaging**: re-cut against the current `release.json` default tree (not r353), refresh every identity and
     the sha256, and give it a number: `optional/*/01[0-9][0-9]-*.patch` is the only pattern
     `scripts/apply-all.sh --include-optional` and `scripts/validate-set.sh` check 8 discover, so an unnumbered
     patch is invisible to both.  A one-shot `INFO` census (proof the arm ran at `-lv 4`) is required: a
     numerics-changing opt-in whose arm cannot be observed is not reviewable.
R5. **Evidence**: drop the prefill-KLD claim (see B2) and quote the decode-width point instead; add the three
     missing `test-backend-ops` cases - the inert boundary `nb 9..16`, a deep cache (`kv 32768`), and the
     **derived-mask** form (what a real qwen35 dense 27B run uses at width <= 8); state the acceptance of both MTP
     arms next to any MTP throughput.
R6. **Claim withdrawal**: `GGML_HIP_FA_TILE_GQA6_FOLD=0` does not restore the original path on gfx1201 (B1), and
     "no prefill path is intentionally changed" should read "the fold is width-gated, so a `-ub <= 8` prefill takes
     it".

`assessment-delta.patch` already implements R1's *intent* only partially: it has R2, R3, the census, the geometry
hardening and R5's tests, but **not** R1 (it keeps the runtime branch, which is the blocker).  It is a state record
and a starting point, not a candidate.

## How to reproduce this session's state

```bash
# the tree: release tip + the PR patch + the review delta
bash scripts/scratch-tree.sh --gpu-target gfx1100 --fast blocks        # r359 tree, cold build ~7 min
bash scripts/scratch-tree.sh --gpu-target gfx1100 --fast arm wip/pr362/pr-submitted.patch
cd ~/llama-scratch && git apply ../llama-cpp-rdna-boosts/wip/pr362/assessment-delta.patch
cmake --build build-rocm --target llama-cli llama-server llama-bench test-backend-ops -j 16

# the off arm is inert check (base recorded from the UNPATCHED canonical build)
LLAMA_PPL=~/llama.cpp/build-rocm/bin/llama-perplexity scripts/gate-prefill-logits.sh \
  --base-dir ~/.cache/rdna-boosts/gates-pr362 --model /llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf \
  --kv q8_0 --ub 1 --ctx 4096 --chunks 1 --device 0 --no-faults --record
# then compare the patched build with the same line (no --record, no env) and with
# --candidate-env "GGML_HIP_FA_TILE_GQA6_FOLD=1"

# correctness (the delivered suite reaches the geometry; the added cases are in the delta)
GGML_HIP_FA_TILE_GQA6_FOLD=1 ./build-rocm/bin/test-backend-ops -o FLASH_ATTN_EXT -b ROCm0
./build-rocm/bin/test-backend-ops -o FLASH_ATTN_EXT -b ROCm0 -p 'hsk=256.*nr23=\[6,1\]'

# throughput
./build-rocm/bin/llama-bench -m /llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf -ngl 99 \
  -ctk q8_0 -ctv q8_0 -p 0 -n 64 -d 2048,8192,32768 -r 3
# 32K prompt: /tmp/prose-32k.txt is prompts/prose-rdna-boosts.txt concatenated 6x (sha256 ea64e5e9925ac6317cacb6121e219695d0f90c67987ac8d4f2e49f324c18cdae)
```

Model on this box: `/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf` (qwen35, n_head 24,
n_head_kv 4 = ratio 6, head 256/256, so it is exactly the target geometry).  The canonical `~/llama.cpp`
checkout at `003c88e1f` is the untouched reference build (`build-rocm/bin-*`, `test-backend-ops`); it was never
modified.

## What to check when the author resubmits

1. R1 is really structural: diff the VGPR/occupancy of the pre-existing `fattn-tile` objects against `003c88e1f`
    (or, cheaper, run the off arm's decode-width KLD point on **gfx1201**, which is where the regression showed).
2. Default is OFF, the env is an opt-in in `ENVIRONMENT.md` §8, and the census line exists and prints at `-lv 4`.
3. The arch gate is `RDNA3_0`, no RDNA3_5, no RDNA4 (or RDNA4 only with its own measurement and a decode-width
    KLD point).
4. It is cut against the *current* `release.json` tree, numbered `01xx` (or folded into a block by the maintainer),
    with refreshed identities and `scripts/validate-set.sh` green.
5. The three test cases are present and pass with the fold on.
6. The PR text no longer claims prefill-KLD evidence or that `=0` restores the original path.
7. Re-measure here (gfx1100) with the plain arms first: a fold that does not reproduce at least the +3.3 % tg64 at
    32K and the 8K neutrality is not worth the opt-in.
