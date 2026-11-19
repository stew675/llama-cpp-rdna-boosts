# Benchmark results: llama.cpp master vs rdna-boosts

The current, full benchmark suite. **v2 (2026-08-27)** is the complete
gamut; the **v1** record (curl + `/completion`) is preserved for history.

> **MTP gate (2026-09-02):** adaptive/draft MTP is NOT covered by these
> decode suites (llama-bench/benchy decode 1 token/step and never touch
> the speculative verify batch or draft context). MTP regressions slip
> through them (worked example: the 2026-09-02 block-13/block-08 MoE MTP
> collapse, 0/1527 draft acceptance). The MTP protocol, baselines and
> acceptance gate live in [mtp-adaptive-methodology.md](mtp-adaptive-methodology.md).
>
> **Prefill-logit gate (2026-10-08):** same-seed coherence and MTP are also
> blind to a uniform prefill-logit shift. The default-on BF16 chunked GDN
> (issue #113) passed both while shifting prefill logits by mean KLD 0.03 to
> 0.62. The protocol and thresholds live in
> [prefill-logit-methodology.md](prefill-logit-methodology.md); run it for any
> prefill-kernel or default-on approximate-path change.
>
> **Freshness note:** these are the v1/v2 records of the 11-block era
> (builds: master `fe235f434` vs `build-rdna-boosts` @ `a265041b1`). The
> current delivery adds block 12 (hybrid all-reduce, RDNA4-gated) — its
> numbers live in `patches/README.md` (tg64 38.12 / tg512 41.08 on the
> clean-apply build; depth-16384 3-GPU hybrid 38.71 t/s unpinned) and the
> benchy depth-16384 protocol in `wip/HANDOFF.md`.

## 2026-11-19 — a fusion's overlap test must test the buffers the kernel reads: the GDN conv + q/k norm fold on a deferred conv-state gather (r320, PR #316)

**`WORKLOG.md` 261119.004** + [`2026-11-19-gdn-norm-deferred-gather.md`](2026-11-19-gdn-norm-deferred-gather.md) —
the contributor's premise (every DFlash2 verify call refused, -0.7 ms/cycle) belongs to a tree from before r244:
since then the one-kernel `gdn_conv_store` owns that band and already writes `q_out`/`k_out`, so the matcher is never
entered (census: `GDN_CONV_STORE T=2/4/8/5/6/3/9`, no matcher entry; same for `draft-dflash,ngram-mod` and
`draft-mtp-adaptive`).  The mechanism is real in the arms that do **not** take the rollback store: with a deferred
conv-state gather the direct conv reads the cache row and never `m.state->data`, the allocator puts the q `SCALE`
output over that dead node (the spans are `128*nk*T` against `3*C`, colliding from `T=6`), and the matcher refused.
Measured refusals: `ngram-mod` `T=8/11/16`, `ngram-cache` `T=6,9`, `ngram-map-k` `T=16`, `draft-simple` `T=6,7,8`,
width probe `W=6,7,8`.  The matcher now tests the **cache tensor** in exactly that case (stricter than the
contributor's skip).  Worth **0.02 %** of decode on the one delivery arm that reaches it (38 692.4 -> 38 685.6 ms
over 2048 tokens, 5 interleaved pairs, against 39 262.4 ms fold-off), i.e. a correctness fix; the width probe's
`W=1..8` hashes are unchanged, so the non-vacuity instrument is the matcher census.

## 2026-11-19 — the FA tile kernel's KV split must be a run constant, not a function of the padded KV extent (r319, issue #330)

**`WORKLOG.md` 261119.003** + [`2026-11-19-fa-split-boundary-purity.md`](2026-11-19-fa-split-boundary-purity.md) —
on a 3-device `-sm tensor` split `--spec-type none` and `--spec-type draft-mtp` emitted different greedy text
(q8_0 prose `120c74935967` against `c1f9d0b3f5b6`, bf16 reasoning `7dca501eeb0e` against `759c1a4515c6`).  At the FA
op `K->ne[1]` is `GGML_PAD(used, 256)` computed after the batch is applied, so a W-row verify sees `pad(p+W)` against
the decode's `pad(p+1)`; the tile launcher's KV split `parallel_blocks` **is** the reduction order and was chosen
from `ntiles_KV`, so at a 256-cell multiple the verify took a different split (176 against 175) and the logits moved
by up to **0.232868** -- a greedy near-tie flipped.  The split is now `min(P_const, ntiles_KV)` for `Q->ne[1] <= 8`
(`P_const` = the search's one-full-wave value), exact by construction; `GGML_CUDA_FA_SPLIT_CONST=0` is the pre-fix
control and reproduces the old pair byte for byte.  Only the plain 3-device tensor arm moves, and decode is flat at
every depth to 131k (tg128 46.6 / 45.6 / 42.3 / 39.3 / 36.8 / 34.6 at d0/8k/32k/65k/98k/131k).  The width probe gained
a **256-cell-boundary** prefix arm (the flat arm on a 256-multiple prefix cannot see this class: it read PASS through
r318); pre-fix the boundary arm reads **0.232868** (q8_0) / **0.196008** (bf16), post-fix **0**.  Gate:
`scripts/gate-spec-purity.sh`.

## 2026-10-29 — the derived kq mask for batches of several sequences (r172, PR #247)

**`WORKLOG.md` 261029.003** + [`2026-10-29-derived-kq-multiseq.md`](2026-10-29-derived-kq-multiseq.md) —
`--parallel > 1 --kv-unified` gives each of the ubatch's sequences its own position range (`k * 2^23`), so the one
predicate `tok_lo <= cell_pos <= tok_hi` stays exact for a multi-sequence batch with **no kernel change**, and a
context no longer reserves the packed mask it could not consume: **-1.031 GiB device and -0.937 GiB pinned host** at
`-c 245760 -ub 2048 --parallel 2 -ctk/-ctv q8_0` (dense 27B), with two concurrent 16K prefills unchanged (26.96 vs
27.23 s at 1 GPU, 17.07 vs 17.09 s at 2 GPU `-sm tensor`) and a 112K-resident decode beside a 16K prefill unchanged
(13.97 vs 13.96 s).  The predicate is the packed one, verified cell by cell (0 mismatches on dense 27B at 1 and 2 GPU
tensor, Qwen3.5-4B hybrid, Gemma4-12B and gpt-oss-20b iSWA, 4 sequences per perplexity batch) and at the logits
(**0.000000 / 100.000 %** at f16 and q8_0 KV), with `plain == draft-mtp` intact.  Two negatives are recorded there: it
buys nothing on qwen4exp/Flash-Next (its QSA builder keeps the packed mask anyway) and a server that really shares
prefixes through `llama_memory_seq_cp` latches once and keeps the packed mask.

## 2026-10-29 — MMB owns what MMB claims: the pair fusion's stand-down predicate (r171, `TODO` #115 closed)

**`WORKLOG.md` 261029.001** + [`2026-10-29-mmb-pair-ownership.md`](2026-10-29-mmb-pair-ownership.md) —
`mul_mat_q_pair` stood down for MMB on a weight-*type* test while the mmb dispatch, the fused-GLU matcher and the
mmq triple decide on the *shape*, so on every shape MMB declines (`T < mmb_min_t()`, `n_used > 64`, ...) the pair
had already stood down and the triple served the gate+up pair.  Both pair arms now ask
`ggml_cuda_mmb_supported_mm()`/`_supported_mmid()` (the dispatch predicates, a superset of the fused-GLU one), which
also closes a latent hole in the optimizer's BF16-only activation marking, and the marking pass no longer elides an
activation with a routed `MUL_MAT_ID` reader.  Measured **inert-to-neutral**: gfx1201 byte-identical text on 27B
Q8_0 / 35B-A3B Q4_K_M (1 and 3 GPU) / LFM2 Q6_K with prefill-logit KLD 0.000000 / 100.000 %; gfx1151 pair == triple
== unfused == `GGML_CUDA_MMB=0` byte-identical, pp512 pair vs triple 3379/3399 vs 3348/3407 (LFM2) and
2304/2312 vs 2296/2313 (35B-A3B).  The exercise also found a first-cut **SIGSEGV** (a null `src[1]` probed on an
arbitrary node pair) that only gfx1151's Flash-Next warm-up graph hit.

## 2026-10-26 — decode graphs replay: the graph-cache key must name the layer, not only the activation (r149, TODO #107 closed)

**`WORKLOG.md` 261026.007** + [`2026-10-26-graph-replay-keys.md`](2026-10-26-graph-replay-keys.md) —
`ggml_cuda_graph_key` was `(first node data, n_tokens)`, and under a split mode every layer's subgraph can
begin on the same activation buffer, so one entry served them all and each layer was compared against the
previous layer's properties.  The key now also carries the subgraph's **weight identity** and a structural
signature.  2x R9700 x8, Flash-Next IQ4_NL `-sm tensor -ncmoe 48`, 2000-token decode: **37.7 -> 42.8 t/s
(+13.5 %)** and `properties_changed` 452439 -> 28; +12.0 % on the same shape at 3 devices, +1.4-2.0 % with a
draft, flat on 1 GPU / 2-3-device dense / `-sm layer` / `llama-bench tg128`; **-6..-9 % on a 3-device
tensor-split dense *plain* decode**, which `GGML_CUDA_ALLREDUCE=internal` removes (`TODO #109`).  Text is
byte-identical in 10/10 config pairs and draft acceptance is bit-identical; prefill KLD 0.000000.

## 2026-10-26 — the qwen4exp HC fused ops must reproduce the model's chain, and the arms are the oracle (r148, issue #228)

**`WORKLOG.md` 261026.006** + [`2026-10-26-hc-fusion-arms.md`](2026-10-26-hc-fusion-arms.md) —
`GGML_OP_HC_MIX` published the **raw** inject scatter logit while `GGML_OP_HC_COMBINE` divided it by `hc`
again.  That pair is self-consistent, so the Q8_0/BF16-hc models (which take both fused halves) stayed
right — but `LLAMA_FUSED_HC_MIX` / `LLAMA_FUSED_HC_COMBINE` select the halves **independently**, and the
`6050b467c` re-base had inverted the convention under the ops (upstream's chain now emits `inject/hc` and
the combine applies only `2*sigmoid`; v14 emitted it raw with the scale in the combine).  Measured on
gfx1151 (halo): every hc-down quant that is not Q8_0/BF16 (Flash-Next Q5_K_M, IQ4_NL) degenerated into a
greedy repetition loop at default config, and a BF16-hc model (GSQ-RCO IQ3_XXS) produced an **empty
generation** with the documented `LLAMA_FUSED_HC_COMBINE=0` kill-switch.  After the fix all four arms (plus
the CUDA-side chain fusions and the DSV4 HC pre arm) are **byte-identical to the unfused chain** on all
three GGUFs, the BF16-hc default is **bit-identical before vs after**, and the fusions keep their value
(decode tg64 34.78 vs 31.73 t/s with both arms off, +9.6 %, prefill flat).  New gate:
`scripts/gate-hc-fusions.sh`.  Rule: `GREEDY-PURITY.md` §61.

## 2026-10-26 — the rms_norm float4 reduction is opt-in; the strided reduction is the default again (r145, issue #229)

**`WORKLOG.md` 261026.002** + [`2026-10-26-rms-norm-float4-ab.md`](2026-10-26-rms-norm-float4-ab.md) —
upstream #29720 (adopted by the r127 re-base) replaced `rms_norm_f32`'s strided scalar reduction with a
float4 one.  Measured on gfx1201: it is worth **+1.0-2.7 % prefill and ~0 % decode** (27B 1 GPU pp2048
1403 vs 1366, pp512 1397 vs 1371, pp8192 1341 vs 1316, tg128 30.89 vs 30.87; 35B-A3B pp2048 4976 vs 4922;
27B 3 GPU tensor pp2048 1990 vs 1972) — but a reduction-order change is a *numerics* change, and MTP draft
acceptance is a discrete function of it: **+0.7..+1.3 points** on this box's default arm (27B, 5.2 k and
94 k contexts) versus **0.611 vs 0.792 acceptance, −18 % t/s** on the same model with the SSM decode
fusions disabled (the shape the issue reporter runs), and **−7..−9 % tg** on his box.  The arms are pure in
every case (`plain == draft-mtp` byte-identical), so it is a payload-dependent lottery, not a defect.  The
default is therefore the **strided** reduction (which is also what the decode-band norm fusions are
bit-identical to); `GGML_CUDA_RMS_NORM_VEC=1` opts into float4.  Rule: `GREEDY-PURITY.md` §60.

## 2026-10-26 — where a 2-GPU out-of-VRAM MoE decode token actually goes (r144, block 09 diagnostics)

**`WORKLOG.md` 261026.001** + [`archive/work/moe-cpu-overlap/FINDINGS-2026-10-26.md`](../archive/work/moe-cpu-overlap/FINDINGS-2026-10-26.md)
— the O0 measurement that retired `TODO #36` (genuine CPU/GPU overlap for the MoE misses).  2x R9700 on
the **PCIe 5.0 x8** root ports (`HIP_VISIBLE_DEVICES=0,2`), Qwen3.8 Flash-Next IQ4_NL (100.4 GB, 512
experts x 10 used -> 64.8 GB of expert weights, 85.4 % slot residency at the auto arena), **2000-token**
generations with a frozen heat prior:

| arm (`-ncmoe 48`, `-fa 1`, q8_0 KV) | t/s | ms/token | access hit | expert H2D |
|---|---:|---:|---:|---:|
| `-sm tensor`, auto arena (85 % residency) | 35.53 | 28.15 | 99.74 % | 3.3 MiB/tok |
| `-sm tensor`, `MOE_EXPERT_CACHE_MIB=8192` (25 %) | 34.04 | 29.37 | 98.71 % | -- |
| `-sm tensor`, `MOE_EXPERT_CACHE_MIB=2048` (6 %, the floor) | 34.44 | 29.04 | 95.66 % | 55 MiB/tok |
| `-sm layer`, auto arena | 25.73 | 38.86 | 99.48 % | 6.6 MiB/tok |
| `-sm tensor`, `GGML_CUDA_DISABLE_GRAPHS=1` | 35.14 | 28.46 | -- | -- |

Two conclusions worth keeping: **the expert misses are not the cost** (14x smaller arena = 0.9 ms of a
28 ms token; 3.3 MiB/token of H2D at ~28 GB/s is 0.12 ms), and **the token is dispatch-bound** — 297
`graph_compute` calls / 12 258 node dispatches per token = 13.5 ms of host time, with the decode graph
captured but never replayed (93 % `properties_changed`; hence `DISABLE_GRAPHS` costs nothing).  Also:
`-sm tensor` is **+38 %** over the `-sm layer` CLI default on this shape.  Follow-up: `TODO #107` /
`archive/work/graph-replay-placement/` (closed by r149's graph-cache key).

## 2026-10-25 — the expert cache at a model teardown (block 12, r143)

**`WORKLOG.md` 261025.017** — a table whose host master was freed by `llama_model_free` stayed registered,
and the deferred arena sizing read it: a host memcpy SIGSEGV in `fill_slot_from_host` (reproduced on
Flash-Next GSQ, 2 GPU `-sm layer -ts 59,41 -ncmoe 24`, where a `-ts` *list* makes llama-bench run one test
per value and reload the model between them; `--host-experts pool` failed identically).  `llama_model_free`
now drops the model's tables and un-sizes the cache, so the next model sizes on its own first band pass
instead of running uncached.  Single-model runs unchanged (same cache lines, same text, pp4096 570.20 ->
569.91 t/s).  The same record carries the reporter's **IQ3_S** arm for r142: pp8192 **2185 -> 2255 t/s
(+3.2 %)** at 31.5k **2134 -> 2197 (+2.9 %)**, `HC_POST=0` **+2.2 %**, PPL and greedy text unchanged.

## 2026-10-25 — qwen4exp HC mixer: the inject GEMM must sit beside the down GEMM (block 11, r142)

**[2026-10-25-qwen4exp-hc-inject-adjacency.md](2026-10-25-qwen4exp-hc-inject-adjacency.md)** —
issue #226 follow-up.  The backend's per-graph BF16 src1 cache holds one converted tensor, so the mixer's
two BF16 GEMMs over the same F32 `xn` only share it while they are **adjacent in the graph**; #31028's
emission order (inject last, behind the gate GEMM whose src1 evicts the entry) converted
`[hc*n_embd, n_tok]` twice per mixer.  Flash-Next GSQ-IQ3_XXS, 3 GPU `-sm tensor`, all-VRAM: pp8192
**2208 -> 2284 t/s (+3.4 %)**, pp2048 +3.2 %, pp31500 **2158 -> 2228 (+3.2 %)**, and the trace shows the
`convert_unary_cont_vec4<float,bf16>` launches on `xn` **4560 -> 2280** (274 -> 85 ms in one pp2048 pass).
Bit-identical throughout (PPL to the last digit, 24/24 chunk PPLs, 7/7 greedy cells, fused-vs-chain KLD
-0.000000 / 100.000 %); `GGML_CUDA_BF16_SRC1_CACHE=0` is the mechanism control.

## 2026-10-23 — PR #206 promotion: MTP prompt-tail readback skip (block 02, r118)

**[2026-10-23-pr206-mtp-tail-readback-promotion.md](2026-10-23-pr206-mtp-tail-readback-promotion.md)** —
with `LLAMA_MTP_INGEST_TAIL=4096`, a prompt ubatch entirely below the ingested tail no longer forces the
MTP hook's `llama_get_embeddings_nextn()` (whose `ctx->synchronize()` drained both backends after every
prompt batch) or the target's nextn readback.  31,476-token prompt, 2x R9700, Flash-Next IQ3_XXS + shared
Q8_0 MTP head: prefill **1488.3 -> 1506.8 t/s (+1.24 %)**; the `LLAMA_MTP_TAIL_READBACK=0` arm lands at
1490.0 (+0.1 %, i.e. r117).  Identical greedy text and identical draft counts (177/364) on every load,
decode flat, prefill logits bit-identical, no change at all with the tail unset.  Also records the 2x2
that splits the win (most of it is the drain the hook used to force, not the 21 MB copy) and the side
finding that `decode()`'s per-ubatch `sync_readback` only runs when another ubatch follows.

## 2026-09-27 — block 06 tiny-CPU-graph heuristic vs CPU-offloaded FFN (issue #52, r13)

**[2026-09-27-block-06-tiny-cpu-graph-issue52.md](2026-09-27-block-06-tiny-cpu-graph-issue52.md)** —
the r8 single-thread heuristic sized a graph by its node outputs, so a CPU-offloaded FFN chunk ran on
one thread (r12 3.22 t/s); the r13 fix also counts node inputs (4.76-5.02 t/s, matching the
kill-switch's 4.82-4.95), with `GET_ROWS` `src0` exempt.  Same-seed greedy text byte-identical on/off.

## 2026-09-20 — Qwen3.8-Flash-Next IQ4_XS prefill: delivery base vs the `mmb` WIP

**[2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md](2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md)** —
prefill (`-n 0`) on the 94 GiB qwen4exp model, gfx1151/ROCm 7.14, bf16 KV, `-b/-ub 2048`, `-r 3`, across
the **true delivery base** (`8a2567e1e`, unmodified — built in a separate worktree) and two WIP arms:
MMB/HC16 off and all gates on.  Headline: the WIP is **+32 % at pp2048 and +43–48 % from pp4096 to
pp32768** end to end, of which ~+14–16 % at 8K–32K is the WIP's always-on work (qsa3, always-QSA,
non-temporal, indexer) and ~+27–29 % is MMB/HC16.  Also records the base's pp4096 QSA dense-shortcut
cliff that the always-QSA flip removes.  Read it before quoting any prefill delta: the older "MMB off"
arm was the WIP tree with only the MMB gates off, **not** the delivery base.

## Adaptive MTP — four-workload records

**[2026-09-15-adaptive-mtp-tuning.md](2026-09-15-adaptive-mtp-tuning.md)** is the **current**
adaptive-MTP controller record: it replaced the mean-reverting table with the **credit bucket** and
tuned it (the block-01 amendment).  Read this one for the delivery's controller numbers.

**[2026-09-13-adaptive-mtp-4-axis-n12.md](2026-09-13-adaptive-mtp-4-axis-n12.md)** is the
authoritative four-workload measurement of the **pre-tuning table controller**:
`draft-mtp-adaptive` at its recommended **ceiling 12** (the
2026-09-13 clamp relaxation), across the R/P/C/K workloads, at a realistic **`-n 3000`** with
reasoning pinned per axis (`--reasoning off` for P/C/K).  Against fixed `n3` it is reasoning
-1%, prose +13%, code +28%, recall +61%; a 256-token run measured the warm-up and inverted the
code ranking.  It supersedes all earlier adaptive records -- both the `n_max <= 7` clamp and the
`-n 256` / default-reasoning protocol were artifacts.  **Beware (2026-09-17):** its prose acceptance
(0.50654) is the *table's*; the delivery credit bucket's is 0.60232, so cite this record as the table
arm, **not** as "the delivery".  **Beware (2026-09-18):** its ceiling 12 is for **UD-Q4_K_XL on one
card** -- on a Q8_0 27B with a 2-card `-sm tensor` split, 12 *loses* to 7 (-5.8 % on the code prompt),
so treat the ceiling as shape-dependent (single card ~9, multi-GPU 6-7) and see the caveat in
[mtp-adaptive-methodology.md](mtp-adaptive-methodology.md).  The gate protocol (length, reasoning,
baselines) lives in [mtp-adaptive-methodology.md](mtp-adaptive-methodology.md).

## Block 12 — hybrid all-reduce env matrix (2026-08-30)

**[block12-hybrid-ar-matrix.md](block12-hybrid-ar-matrix.md)** — the
`GGML_CUDA_ALLREDUCE` (hybrid / nccl / internal / none) x
`NCCL_P2P_DISABLE` (0/1) matrix on 2x R9700 (gfx1201), ROCm 7.14, at
depth-16384: hybrid wins prefill (1620 t/s), internal/none unchanged by
P2P (as designed), RCCL legs lose ~6% with P2P disabled.

## 2026-09-13 — Qwen3.8-27B Unc Q8 ngram-mod 16/2/96

**[2026-09-13-qwen38-unc-ngram-mod.md](2026-09-13-qwen38-unc-ngram-mod.md)** — HIP 12-set, 2× R9700. Stock `--spec-type ngram-mod` with **n-min 2** drafts on dense Unc (farm 24/48/64 did not). Prefill ≈ nospec; MTP n-max 3 still wins unique-prose median. Not qwen4exp managed-ngrams.

## v2 — the current results (llama-benchy live-server suite)

**[v2-results.md](v2-results.md)** is the canonical results document:
24 throughput rows (16 ROCm + 8 Vulkan) × f16/bf16 KV, plus 18 PPL
corners.

Full matrix: build (master `fe235f434` vs rdna-boosts `a265041b1`) × KV
cache (f16 vs bf16) × 4 model/card sets (1-card Q6_K, 1-card Q4_K_XL,
2-card Q8_0, 3-card Q8_0), on both ROCm and Vulkan.

- **[v2-results.md](v2-results.md)** — the headline numbers, tables, and analysis.
- **All charts on one page**: [graphs/ALL-CHARTS.md](graphs/ALL-CHARTS.md)
  (16 charts, split by KV type: 3 series per chart, red=master,
  amber=boosts, blue=vulkan).
- **[benchy-methodology.md](benchy-methodology.md)** — protocol (llama-benchy
  v0.4.0, `--pp 2520 --tg 240 --depth 0 4096 8192 16384 32768 65536 131072
  --no-cache --runs 2`), harness validation, and why it matches v1.
- Harness: [`scripts/`](scripts/) (`benchy-run.sh`, `run-benchy-suite.sh`,
  `run-benchy-vulkan-suite.sh`, `run-all.sh`, `run-vulkan-all.sh`,
  `benchy-json-to-md.py`, `make-v2-graphs.py`).
- Raw data: [`results/benchy/`](results/benchy/) (per-row JSON + md) and
  [`results/ppl/`](results/ppl/).

### v2 headline

Master's ROCm BF16 KV silently degrades decode with depth (−2% shallow to
**−39 to −51% at 128K**); rdna-boosts fixes that and goes further — its
BF16 config is **faster than master-f16 at every depth and more accurate
(PPL) on every model**, and even its f16 config beats master-f16 at the
same KV type. Vulkan is a genuine native-BF16 reference with superb
single-card prefill, but its decode collapses on multi-card and its BF16
costs prefill at depth. No performance or accuracy reason to choose
either baseline approach.

## v1 — the historical record

**[v1-results.md](v1-results.md)** (2026-08-26) was the first benchmark
pass, using a curl + `/completion` protocol against a llama.cpp server.
It covered 4 configs (A/B/C/V) × 1/2/3 GPUs. The v2 suite supersedes it
(wider KV matrix, both backends, PPL corners), and v2's A/B/C/V corners
reproduce v1's baselines exactly — so the two are directly comparable.

## Environment

Machine: 3× Radeon R9700 (gfx1201, 34 GiB, ~640 MB/s each), ROCm 7.14,
9950X3D host. ROCm builds: `build-rocm` = master `fe235f434`,
`build-rdna-boosts` = `a265041b1` (all 10 blocks), `build-vulkan` = master
`fe235f434` (RADV/Mesa 26.1.7). Models: Q8_0 (29 GB), Q6_K (22.9 GB),
Q4_K_XL (17.6 GB); wikitext-2 for PPL.
