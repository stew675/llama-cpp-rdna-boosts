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

## F2 — qwen4exp is not width-pure: ROOT-CAUSED into TWO stacked causes; **cause 1 FIXED 2026-09-11**

> **Cause 1 is fixed** in the block-14 amendment (canonical tip `1d8f53594`, delivery):
> `ggml/src/ggml-cuda/hc-mix.cu` + the `src/models/qwen4exp.cpp` gates now serve the whole band
> **`1 <= nt <= 8`** (token index on `blockIdx.y`, every per-token pointer offset by the tensor's own
> stride — `inject` with its view stride; at `nt == 1` all added terms are zero, so decode is
> unchanged and was verified byte-identical for f16/bf16/q8_0/q4_0).  Result: `-sm layer` W=1..4 all
> `3adeb313042a871b`, `-sm tensor` W=1..4 all `dcf1ae667f730879` — both equal to their W=1 decode;
> plain == `--spec-type draft-mtp --spec-draft-n-max 3` greedy text (byte-identical); f16 MTP
> acceptance **0.50000 -> 0.76744**, MTP generation **63.3 -> 79.9 t/s**; decode/perf/reserves
> unchanged; 27B/MoE untouched.  **Cause 2 (`W >= 5`) is still open** and is the same band as F1 —
> fix it there, not here.  With a `q8_0`/`q4_0` KV cache the cache's own impurity (F1) dominates
> (acceptance 0.50 -> 0.43), so those configurations must be re-measured after F1; they are not a
> valid gate for this fix.  The table below is the **pre-fix** state.

```
qwen4exp (IQ4_XS), 3-GPU f16 KV, P=256, RS=0 — logits0 hash per decode width (PRE-FIX):
  delivery build:  W=1 dcf1ae667f73 | W=2,3,4 1c801d63666b | W=5 fa34f99951fb | W=6,7 96dbf8375adf | W=8 2ca9b9f5e801
  -sm layer:       W=1 3adeb313042a | W=2,3,4 044715b66e72 | W=5 bdaa8fc57381 | W=6,7 1ffc73e03571 | W=8 15786ddeffad
  after the fix:   W=1..4 = W=1's value on both splits; W=5 c999233926f0 | W=6,7 a8c532e12f9c | W=8 c56ebb61963a (layer)
```

The earlier guess in this file ("the sparse-QSA attention path") was **wrong**.  The divergence is
**not** the attention at all; it is two independent width-selected code paths in the *non-attention*
part of the graph.  Both were found with the per-node dump instrument (see "Tooling" below); the
first divergent node is layer 0's MoE router, and its *input* (`hc_mixed`) is already divergent — i.e.
the HC (hyperconnection) block, not the indexer.

### Excluded, each with a measurement (all leave the W-grouping unchanged)

| candidate | how it was excluded |
|---|---|
| QSA / indexer / selection / arch decode policy | `LLAMA_QSA_OFF=1` (the **whole** QSA regime off) → identical hashes; `LLAMA_QSA_SPARSE_FA=0` (dense masked FA) → identical; `LLAMA_QSA_DENSE_SHORTCUT=0`, `LLAMA_QSA_DENSE_DECODE_UNTIL=0` → identical |
| CUDA graphs, CUDA-side fusions | `GGML_CUDA_DISABLE_GRAPHS=1` → identical; `GGML_CUDA_DISABLE_FUSION=1` (and `_HC_MIX`/`_HC_COMB`/`_HC_FUSION`, `_SCALE_UNARY`) → values change, **grouping unchanged** |
| the float mmvf family | the first divergent node is the router and its weight is **F32** (`blk.0.ffn_gate_inp.weight`, dims [2560,512]), so `ggml_cuda_should_use_mmvf` picks `ne11 <= 3` on fp32-MMA AMD parts (a W-switch at 4).  Widening that band to 8 → *grouping unchanged*; disabling F32 mmvf entirely (all F32 → cuBLAS) → *grouping unchanged* |
| batch **content** | `REPEAT=1` in the probe (batch = W copies of the same token) reproduces the same width's hash byte-for-byte → the divergence is purely width-*selected*, not a content/`l_last` effect |
| all-reduce / tensor split | reproduced with `-sm layer` (whole layers per device, no partial sums) **and** on 1 GPU with `-ngl 4` |

### Cause 1 — the HC fusions are gated on `nt == 1` (fix identified, needs kernel work)

`src/models/qwen4exp.cpp:386` (`build_hc_mix`) and `:458` (`build_hc_combine`) gate the fused
`GGML_OP_HC_MIX` / `GGML_OP_HC_COMBINE` on `nt == 1`:

```cpp
if (nt == 1 && cparams.fused_hc_mix     && fused_ok) { ... ggml_hc_mix(...)     }   // "decode"
if (nt == 1 && cparams.fused_hc_combine)            { ... ggml_hc_combine(...) }   // "decode"
```

Measured in the dump: **98 `op=HC_COMBINE` dispatches at W=1 and 0 at W≥2**; `hc_mixed` is a plain
`VIEW` at W=1 but a fused `SCALE` window at W≥2.  The fused and unfused paths are **not
bit-identical**, so "decode" (nt=1) and "verify" (nt=2..8) disagree.

* **Disabling them is not an acceptable fix**: the fusion is worth **+13.1 % decode** on qwen4exp
  (tg128 48.49 with vs 42.15 without; prefill at parity), measured with
  `LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0` (note: those are the graph-side knobs — the
  `GGML_CUDA_DISABLE_HC_*` envs are different, CUDA-side arms, which is why they did nothing here).
  With them off, W=1 becomes pure against W=2..4, which *proves* cause 1 and its location.
* **Widening the graph gate alone crashes**: `ggml/src/ggml-cuda/hc-mix.cu:273` and `:445` assert
  `GGML_ASSERT(n_tokens == 1);   // decode-only fused op`.  The kernels *below* those host functions
  (`dsv4-hc.cu`: `dsv4_hc_comb_f32` / `dsv4_hc_pre_f32` / `dsv4_hc_post_f32`) are **already
token-generic** (`int64_t n_tokens`, `if (it >= n_tokens) return;`).
* **The fix**: make the two `hc-mix.cu` host paths serve `nt <= 8` (and drop the graph gate to the same
  bound), so the whole decode/verify band uses the fused path — keeping the +13 % **and** restoring
  purity.  Then re-run: `W=1..4` pure (already demonstrated with the env mitigation) and, once cause 2
  is also fixed, `W=1..8`.

### Cause 2 — a kernel-dispatch band at W ≥ 5 (OPEN, and it is F1's cause)

With the HC fusions off, the residual grouping is `{1,2,3,4} {5} {6,7} {8}` and it **survives every
CUDA fusion being disabled** → it is a *kernel dispatch* boundary, not a fusion.  That is the same
`ncols_dst` / `ne11` band class as **F1** (the matmul kernel selection crossing at ne11 = 4/5 and 8),
i.e. the two findings share this cause.  ⇒ **Fix cause 2 together with F1/F3**, then re-check qwen4exp:
the two causes stack, so neither alone restores the `n_max <= 7` band for qwen4exp.

### Tooling added for this investigation (all committed under `tools/`)

* `logits-dump-kv.cpp` — now also takes `CTK`/`CTV` (all KV types) and `REPEAT` (batch = W copies of
  one token), and exits 2 on an unknown KV type instead of silently using f16.
* `node-dump-instrumentation.patch` — re-applies the per-node `[ND]` dump to `ggml-cuda.cu`
  (`GGML_CUDA_NODE_DUMP=1|2` + `/tmp/nodedump_on`, which the probe creates around the decode batch;
  prints `idx/tag/op/ne/nb/h0..h3/dev/name`).  **This is what located the divergence**; apply it, rebuild
  `ggml-hip`, and diff two widths.  Traps (all hit at least once): sync before reading; read the whole
  view address span and gather with the real `nb[]`; key diffs on *(name, node index)*; names are not
  unique and auto-`node_N` names *shift* between widths (only `cb()`-named tensors are comparable);
  `fdst` lines carry the fused window's dest index, not the head's.
* `rv.sh` — the driver (`res`/`kv`/`coh`/`mtp`/`bench`/`width`).

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
