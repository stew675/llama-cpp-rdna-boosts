# KV-quant purity and parity follow-ups (found 2026-09-11)

**Status: OPEN — nothing here is a delivery blocker, and nothing here is caused by any of the 16
blocks.**  All three items reproduce **bit-identically on a build without Block 15**, which is how
they were classified as pre-existing.  They were found while re-validating the Block 15 beta patch
against the 15-patch delivery (`beta/block-15-campaign-wins/HANDOVER.md` §10, `README.md` §7).

| item | one-line |
|---|---|
| **F1** | `q8_0` and `q4_0` K/V caches break the dense `n_max <= 7` greedy-purity guarantee (`W=1,2` agree and `W=3..8` agree, but the two groups differ). |
| **F2** | qwen4exp's fused sparse QSA attention is not width-invariant (exempt from the guarantee in practice, but the exemption is undocumented). |
| **F3** | The sub-`q8_0` KV quants (`q4_1`, `q5_0`, `q5_1`, `iq4_nl`) are pure and much smaller but run ~3.4x slower than `f16` because they have no native FA path. |
| **policy** | Differing K and V cache *types* are **rejected** (maintainer decision 2026-09-11). |

## Why the machinery to reproduce this is cheap

Everything below uses one probe (`tools/logits-dump-kv.cpp`, a copy of
`../sm-tensor-plain-vs-spec/logits-dump-singlewidth.cpp` extended with `CTK`/`CTV`) and one driver
(`tools/rv.sh`).  The probe prefills a fixed 256-token prompt, then runs **one** decode batch of width
`W` (env `W`) and prints an FNV hash of token 0's logits.  Same prefix + same position ⇒ the hash must
be identical for every `W` in the decode/verify band (`W = 1..8`; `n_max = W-1`).  This is the sensitive
instrument: **text equality is evidence for purity, never against divergence.**

```sh
# build the probe against the tree under test (BIN = its bin/ dir)
clang++ -O2 -std=c++17 -I <tree>/include -I <tree>/ggml/include \
  tools/logits-dump-kv.cpp -o /tmp/lw-kv -L$BIN -lllama -lggml -lggml-base -Wl,-rpath,$BIN
# one width
env W=3 NGL=99 SPLIT=tensor HIP_VISIBLE_DEVICES=0,1,2 CB=0 RS=0 \
    CTK=q8_0 CTV=q8_0 /tmp/lw-kv MODEL text.txt 256 512      # prints [L] W=3 logits0_hash=…
# a whole matrix (driver prints one hash per width)
BIN=$BIN PROBE=/tmp/lw-kv tools/rv.sh width 27b tensor 0,1,2 "1 2 3 4 5 6 7 8 9" 0 auto
BIN=$BIN PROBE=/tmp/lw-kv tools/rv.sh width 4b tensor 0 "1 2 3" 0 auto CTK=q4_0 CTV=q4_0
```

Notes: `CB=0` is mandatory (the callback changes MoE numerics and aborts under the meta backend);
`RS=0` means "no draft-head reservation", matching the recorded matrices; the prompt is
`../sm-tensor-plain-vs-spec/p0long.txt`; `-fa off` is **not** available as a control for a quantized
V cache — upstream refuses it (`quantized V cache requires flash_attn to be enabled`, from upstream
#25871, present at the fork point, not fork-specific).

## F1 — `q8_0` / `q4_0` K/V caches break the dense width purity

Measured with the probe at `W=1..9`, 4B, 1 GPU (identical shape on 1 GPU, 2-GPU `-sm layer`, 2-GPU
`-sm tensor`, 3-GPU `-sm tensor`, so it is **not** an all-reduce/tensor-split effect):

| `CTK`/`CTV` | W=1 | W=2 | W=3 | W=4 | W=5 | W=8 | W=9 |
|---|---|---|---|---|---|---|---|
| f16/f16 | `671d6096` | = | = | = | = | = | `ba078b26` (expected: cause B) |
| bf16/bf16 | `b5d7e7b4` | = | = | = | = | = | — |
| q4_1 / q5_0 / q5_1 / iq4_nl | pure (all widths equal) | | | | | | — |
| **q8_0/q8_0** | `0edf55a1` | `0edf55a1` | `31a0c1ba` | `31a0c1ba` | `31a0c1ba` | `31a0c1ba` | — |
| **q4_0/q4_0** | `8125e094` | `8125e094` | `619c151e` | `619c151e` | `619c151e` | `619c151e` | — |

**The boundary is `W=2 → W=3`, not block 00's `n_q <= 8`.**  Block 00's `ntiles_dst_eff` fix
(`fattn-common.cuh`, `launch_fattn`) makes the KV split query-width-independent for `Q->ne[1] <= 8` —
it evidently does not cover whatever differs here.  Text-level impact (27B, 3-GPU `-ts 1/1/1`,
ctx 8192, 300 greedy tokens, q8_0 KV): plain `8ed58aa9` (1330 chars) vs `n_max 3 == n_max 7`
`da56855b` (1406 chars) — a real divergence, not a near-tie; the f16 control is `ce7b9a75` for all
three (pure).

Ruled out already:

* **not** `V4`/native staging — `GGML_CUDA_FA_KV_NATIVE` on/off gives identical hashes;
* **not** the all-reduce or tensor split — 1 GPU shows it;
* **not** K-only or V-only — but note this isolation is **inconclusive**: the mixed pairs (`q8_0/f16`,
  `f16/q8_0`) *are* width-pure, but they run on a **different, 2–3.4x slower path**
  (pp512 4266 / 2231 vs 7713 for q8_0/q8_0), so they are not a valid control.

Correlation worth exploiting: the impure set is exactly the two KV types with a **fast native**
both-quantized FA path (>7700 t/s pp512).  Everything else (including f16/bf16, which are also fast)
stages through the F16 scratch.  So the likely culprit is the split/staging plan *inside the native
`q8_0×q8_0` / `q4_0×q4_0` path* — start by dumping the chosen `ntiles_dst` / `ntiles_KV` /
`parallel_blocks` and the staging branch for `W=1..4` in `launch_fattn`
(`ggml/src/ggml-cuda/fattn-common.cuh`), then look at the tile/MMA kernel selection for
`type_K == type_V == q8_0|q4_0`.

Acceptance criteria for a fix: `W=1..8` bit-identical for both q8_0/q8_0 and q4_0/q4_0 on 1 GPU,
2-GPU layer, 2-GPU tensor and 3-GPU tensor; no prefill/decode regression; f16/bf16 and every other
quant type unchanged; `test-backend-ops -o FLASH_ATTN_EXT` still 7859/7859 on ROCm0 and CPU.

## F2 — qwen4exp is not width-pure

```
qwen4exp (IQ4_XS), 3-GPU `-sm tensor`, f16 KV, P=256, RS=0:
  W=1 -> dcf1ae667f730879     W=3 -> 1c801d63666ba416     (identical on both builds)
```

The fused sparse-QSA attention (block 14) is a different attention implementation from the dense
tile/MMA FA path that block 00 fixed, so the fix simply never applied to it.  Its repo gate is the
adaptive-MTP **acceptance** rate (`benchmarks/mtp-adaptive-methodology.md`; rule 3 exempts MoE-style
models from byte-identity), which passes (block 15 0.47826, delivery 0.50000 — the delta is the
documented layout sensitivity, since the raw logits are bit-identical across builds).

Two acceptable outcomes, in order of preference:

1. **Fix it** the block-00 way: make the QSA sparse path's split/plan query-width-independent, then
   re-run the plain-vs-spec greedy probe on qwen4exp (`none` vs `n_max 3` vs `n_max 7` must be
   text-identical, as f16 dense already is).
2. **Document the exemption** explicitly in `../../GREEDY-PURITY.md` (the matrix there currently
   covers the dense models only, and nothing states that the sparse path is exempt).

## F3 — sub-`q8_0` KV quant parity (the biggest win available)

4B, ctx 204800, ub 2048, 1 GPU, same-type pairs:

| type | KV self | pp512 | tg32 | pure | native FA path? |
|---|---|---|---|---|---|
| f16 | 6400 MiB | 7765 | 99 | yes | yes |
| bf16 | 6400 MiB | 7742 | 99 | yes | yes |
| q8_0 | 3400 MiB | 7713 | 97 | **no (F1)** | yes |
| q4_0 | 1800 MiB | 7696 | 95 | **no (F1)** | yes |
| q4_1 | 2000 MiB | 2287 | 64 | yes | no — F16 staging |
| q5_0 | 2200 MiB | 2270 | 56 | yes | no |
| q5_1 | 2400 MiB | 2197 | 60 | yes | no |
| **iq4_nl** | **1800 MiB** | **2293** | **61** | **yes** | no |
| mxfp4 | — | — | — | — | rejected for KV ("failed to create context") |

The four slow types stage the whole cache through the F16 scratch, which is a ~3.4x prefill penalty —
and Block 15 **already ships the mechanism to fix exactly that**: V4/V5 replaced the per-operand
staging source with one shared type code
(`FATTN_KV_NATIVE_{NONE,Q8_0,BF16}` in `ggml/src/ggml-cuda/fattn-common.cuh`), consumed by the
launcher, the alloc-size query and the kernels so the three cannot disagree.  Extending that enum with
`Q4_1`/`Q5_0`/`Q5_1`/`IQ4_NL` variants (and the corresponding in-register conversion while staging the
tiles) is the natural next step; the q8_0 arm is the worked example.

**Compelling target: `iq4_nl`.**  It is the *same size* as `q4_0` (1800 MiB), it is **pure**, and it is
only slow because of the staging path.  A native `iq4_nl` would therefore dominate `q4_0` outright
(same memory, same speed, no purity gap) — plausibly making `q4_0`/`q4_0` unnecessary.

**Hard constraint:** any new native path must be **width-invariant by construction** (verify with the
probe matrix above at `W=1..8` before believing it) — do not repeat the F1 mistake.  Note the conflict
this exposes: the two paths that were made fast (q8_0, q4_0) are exactly the two that are impure, so
"make it native" and "keep it pure" have to be designed together, not sequentially.

## Policy decided 2026-09-11 (mixed K/V types)

Differing K and V cache *types* are **rejected as an accepted limitation** of the repo / Block 15.
Evidence: every mixed pair measured is **1.7–3.6x slower** than the same-type equivalent and never
smaller (4B, pp512/tg32: `f16/q8_0` 2152/57.9, `q8_0/f16` 4475/66.6, `f16/q4_0` 2362/59.7,
`q4_0/f16` 4080/63.4, `q8_0/q4_0` 2704/57.7, `q4_0/q8_0` 2816/56.2, `bf16/q8_0` 2168/58.9,
`q8_0/bf16` 2643/60.8 — versus 7713–7838 / 95–99 for same-type).  Upstream already enforces
same-K/V for DeepSeek V4 (PR #25871, commit `69e62fc77`, present at the fork point), so a repo-level
rule follows upstream's direction.  Whether this becomes a hard error, a warning, or documentation is
a maintainer call — the current fork behaviour is to accept the pair and silently take the slow path.

## Tooling inventory (committed here so this is reproducible)

* `tools/logits-dump-kv.cpp` — the width probe with `CTK`/`CTV` (`f16`, `bf16`, `q8_0`, `q4_0`,
  `q4_1`, `q5_0`, `q5_1`, `iq4_nl`, `mxfp4`; an unknown name exits 2 rather than silently becoming
  f16).  Build command above.
* `tools/rv.sh` — the revalidation driver used for all of the 2026-09-11 measurements:
  `res`/`kv` (reserves via `-v`), `coh` (same-seed text), `mtp` (27B / qwen4exp / MoE acceptance),
  `bench` (interleaved llama-bench), `width` (the probe matrix).  `BIN`/`PROBE` select the build
  under test; extra `KEY=VAL` args are exported for that run.
* Auxiliary inputs (not committed; recreate if missing): the 40k prompt (`/tmp/t-q40.txt` was used),
  `p0long.txt` (in this repo) and `/tmp/tiny.txt`.
