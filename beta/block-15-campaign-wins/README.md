# beta Block 0015 — campaign wins (beta record)

> **Start here: [`HANDOVER.md`](HANDOVER.md)** — the plan, the maintainer decisions of
> 2026-09-10, the merge/gate/validate/cut steps, the state inventory and the open questions.
> This file is the reference: inventory, gate audit, validation protocol, reference numbers.

**Status: BETA — cut and staged (2026-09-10).**  The campaign is complete and
Block 15 is now the 15th delivery patch: `patches/0015-rdna-boosts-block-15-campaign-memory-wins.patch`
(this directory's `block-15-campaign-wins.patch` is the same file).  The
beta window (~4–5 days) **opens now**; promotion into the delivery set is
already done in the sense that the patch is in `patches/` and applies with
the rest (`scripts/apply-all.sh`, strict 15/15 `git am`) — the window is
for tester feedback before the block is declared stable and this README is
turned into the final promotion record.  All seven wins are on by default
except **V4 and V5, which are opt-in through the same switch**
(`GGML_CUDA_FA_KV_NATIVE=1`; see the gate table below) — the policy
exception the maintainer approved on 2026-09-10 (a sub-2 % prefill loss
with a large memory win and no cheap fix ships as an enable switch rather
than a kill-switch), and the explicit instruction for the bf16 item
("treat it similarly to V4 ... gated by the same environment variable").

**Amendment (2026-09-10): V5 native bf16 K/V.**  Folded into the block the
same day it was designed: a bf16 K/V cache no longer needs the F16 staging
scratch, so with the switch on it costs exactly what an f16 cache costs
(4B 968.86 → **256.86** MiB/GPU at ub 2048, 27B 1072.86 → **488.86**,
gemma-4-E4B 1062.89 → **404.89**, gemma-4-31B 2068.89 → **716.89**;
ub 1024/512 win 756/778 on the 4B and 746 on the 27B; qwen4exp unchanged)
for 0.2–2.4 % prefill depending on prompt length, decode untouched.  The
amendment touched `patches/0015` only (`0001`–`0014` stayed byte-identical)
and was re-validated end to end from the delivered patches — see the V5
amendment section in `../../patches/README.md` and §3.4/§9 of
`../../wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`.

**Beta start: 2026-09-10.**  Canonical tip `f5ab5350b` (the block-15 commit,
amended in place with V5; the original cut was `09a137566`) on a fork
rebuilt at `9113cc188` (the reference `~/llama.cpp` checkout had drifted two upstream
master commits past the fork point at cut time; always regenerate from a
canonical fork rebuilt at the fork point — see `../../BASELINE.md`).
The result: **3.44 GiB/GPU + 1.2 GiB host** reclaimed on qwen4exp at ctx
204800 / ub 2048 / q8_0 KV (6690.40 → 3251.39 MiB/GPU compute, 1262.70 →
63.69 MiB host, indexer KV 956.26 → 318.76 MiB/GPU), **800 MiB/GPU + 800
MiB host** on every dense model, and a further **744/632 MiB/GPU** with
V4 enabled — with byte-identical same-seed output, unchanged MTP
acceptance and a ~1.3 % prefill / ~0.3 % decode default cost.

**Validation as a combination** (the important part: the per-win records do
not carry over) was completed on 2026-09-10 both on the merged work tree and
again on the tree built from the **delivered patches** (fresh worktree at
`9113cc188`, `apply-all.sh`, strict 15/15 `git am`, fresh build):

| check | result |
|---|---|
| reserve matrix, 5 models × ub 2048/1024/512 × V4 off/on | every number reproduces the per-win records; the wins compose additively (qwen4exp: pristine 6690.40 → W1 4450.40 → W1+W2 **3251.39**; W2 alone 5491.39; both W gates off = 6690.40/1262.70 exactly) |
| same-seed coherence, gates flipped | **byte-identical** on 4B, 27B, gemma-4-E4B (ISWA), gemma-4-31B (ISWA) and qwen4exp — 4 configs each (V3×V4), 7 for qwen4exp (W1/W2/W3/V3/V4) — at a short and a 40k-token prompt |
| adaptive-MTP gate | 27B 0.76744 (66/86, mean 3.28) **identical in all four gate combinations**; qwen4exp 0.44262 (54/122) identical in all six and equal to the block-14 baseline; MTP +26 % over plain decode |
| op suites | `FLASH_ATTN_EXT` on ROCm0 (both V4 gates) + CPU (incl. the six derived cases), VIEW/CONT/CPY/DUP/CONCAT, `test-alloc`, `test-batch-alloc`: all pass; W4 repro 56.00 → 16.00 MiB and the revert restores `ggml-alloc.c` byte-identically |
| prefill cost (interleaved same-binary A/B, pp20480/ub 2048) | V3 −1.28 % (4B) / +0.28 % (27B); V4 a further −1.85 % (4B) / −1.72 % (27B); decode within noise |
| clean-apply simulation | fresh worktree at `9113cc188` + `scripts/apply-all.sh` (15/15 strict) + fresh gfx1201 build: reserves, coherence, MTP and op suites all reproduced |

**Found during the combination pass and documented, not fixed** (out of
scope; reproduces on block 14): `gemma-4-E4B-it` on 3 GPUs with `-sm
tensor` aborts in the meta splitter because its 2 KV heads are fewer than
the 3 devices.  It works on 1 GPU, on 2 GPUs and on 3 GPUs with `-sm
layer`; every other model is unaffected.  (Also caught and fixed before
the cut: the W3 gate was initially wired with inverted polarity — that is
why the combination pass exists.)

## 1. Inventory (all validated, all now in `patches/0015`)

| # | win | source | measured effect | validated on |
|---|---|---|---|---|
| W1 | **L2 score-chain memory**: relu before the 4-D reshape + `n_blocks`-chunked assembly with `ggml_concat` | `wip/qwen4exp/qsa-memory/patches/0001-L2a-L2m-qsa-score-memory.patch` (1 file, `src/models/qwen4exp.cpp`) | compute reserve 6690.40 → 4450.40 MiB/GPU at ub2048 (ub1024 3346.50 → 2274.35, ub512 1724.56 → 1188.56); bit-identical output | qwen4exp coherence, ub sweep |
| W2 | **L1 derived QSA bias + visibility, mask prune, input-fill guards** (incl. the `llm_graph_input_attn_k` null-mask guard) | `wip/qwen4exp/qsa-memory/patches/0002-derived-qsa-block-bias.patch` (10 files, +557/−80, base = W1) | compute 4450.40 → **3251.39** MiB/GPU and host 1262.70 → **63.69** MiB at ub2048 (ub1024 1675.33/33.64); coherence byte-identical (derived vs mask, and vs the dense FA fallback); MTP 0.61616 | qwen4exp: 3k + 40k prompts, sparse/dense A/B in one binary, MTP probe, ub sweep |
| W3 | **keys-only QSA indexer cache** (the indexer V buffer is dead: keys-only scoring) | `wip/qwen4exp/keys-only-indexer/0001-keys-only-qsa-indexer-cache.patch` (3 files, 62 lines) | indexer KV 956.25 → 318.75 MiB; box total 88.58 → 86.70 GiB (V was triplicated); perf parity; validation matrix (bf16/f32 ctx, parallel 2, unified, prompt-cache/checkpoint round-trips, MTP 0.741, f16 byte-identical) | qwen4exp |
| W4 | **ggml-alloc: release unused view sources** (3b) | `upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch` (+35 lines, `ggml/src/ggml-alloc.c`; upstream-applicable, clean on `9cf3bf256`) | repro 56.00 → 16.00 MiB; removes the leak that forced W1's concat assembly; no change on current models (latent trap) | master: repro + `test-alloc` + `test-batch-alloc`; fork: coherence byte-identical on 4B/27B/qwen4exp, reserves and MTP unchanged, `test-backend-ops` VIEW/CONT/CPY/DUP/CONCAT OK |
| V3 | **derived kq mask** (the packed `n_kv × n_tps` F16 mask and its host mirror are no longer materialised; the MMA FA kernel derives visibility from compact per-cell state) | `wip/arch-independent-memory/patches/0003` (engine: op + CPU reference) + `0004` (CUDA MMA kernel) + `0005` (graph plumbing + backend probe + enable) | compute **−799.20 MiB/GPU** and host **−799.21 MiB** at ctx 204800/ub 2048 (4B 1800.33 → 1001.13 / 840.34 → 41.13; 27B 1920.33 → 1121.13 / 880.34 → 81.13; gemma-4-E4B ISWA −809/−809; gemma-4-31B −811/−811; scaling exactly `n_kv × n_tps × 2 B`); coherence byte-identical incl. both SWA gemmas and 40k prompts; MTP identical (0.76744); **ON BY DEFAULT** (`LLAMA_KQ_MASK_DERIVED=0` forces the packed mask); cost prefill −1.1 %, decode −0.7 % | 4B/27B/gemma-4-E4B/gemma-4-31B/qwen4exp + `test-backend-ops` FLASH_ATTN_EXT (6 derived cases on CPU, 3 on ROCm0, full suite) + the phase-1 host oracle |
| V4 | **native q8_0 K/V in the FA kernels** (dequantize while staging the shared tiles; the whole-cache F16 staging scratch and its per-ubatch conversion pass are gone for q8_0) | `wip/arch-independent-memory/patches/0006-v4-native-q8-kv.patch` (6 files, +357/−47, base = W1+W2+V3) | compute **−744 MiB/GPU on the 4B** (1001.13 → 257.13) and **−632 MiB on the 27B** (Meta 1121.13 → 489.13) at ctx 204800/ub 2048, more at smaller ub (4B ub 1024 −772, ub 512 −786; gemma-4-31B −1224); qwen4exp control unchanged; coherence byte-identical (4B/27B/both gemmas/qwen4exp, incl. 40k prompts and the TILE ub-8 path); MTP acceptance identical (0.76744); **OPT-IN** (`GGML_CUDA_FA_KV_NATIVE=1`, default off) - cost prefill −1.7 %, decode ±0.1 % | same matrix + `test-backend-ops` FLASH_ATTN_EXT with both gates |

### 1b. Block 15 is CUT (2026-09-10) — the wins are in `patches/0015`

| id | what | expected effect (dense models, ctx 204800, ub 2048, q8_0 KV) | spec |
|---|---|---|---|
| **V3** | derived kq mask for the plain attention path: stop materialising the `n_kv × n_tps` F16 mask and its host mirror; derive visibility in the FA prefill/MMA kernel from compact per-cell state (packed mask kept for decode, small batches and every unsupported case).  **DONE 2026-09-10 (phases 1+2a+2b+2c), ON BY DEFAULT** — the predicate is proven bit-exact on the host (incl. SWA and non-causal), the op + a CPU reference + the CUDA MMA derived path are validated standalone (6/6 derived `test-backend-ops` cases on CPU, 3/3 on ROCm0, 5104/5104 FA suite), and the graph plumbing/probe shipped as `patches/0005-*`.  The packed mask is still created in every graph — it simply ends up with no consumer on the derived path, so the allocator leaves it unallocated and the existing fill guards skip the fill (no consumer can ever be mis-served) | **−799 MiB/GPU VRAM − 799 MiB host** measured (4B 1800.33→1001.13 compute + 840.34→41.13 host; 27B 1920.33→1121.13 + 880.34→81.13; gemma-4-E4B ISWA −809/−809; gemma-4-31B −811/−811; scaling exactly `n_kv × n_tps × 2 B`).  Cost: prefill −1.1 % (pp20480/ub 2048), decode −0.7 %; MTP acceptance identical (0.76744); qwen4exp unchanged | **`wip/arch-independent-memory/V3-DERIVED-KQ-MASK-PLAN.md`** (verified predicate + spec + §4.3-§4.5 record); `DERIVED-MASK-DESIGN.md` §2–§5 + §7; `HANDOVER.md` §3.1 |
| **V4** | native q8_0 K/V in the FA path (MMA **and** TILE loaders): dequantize into the shared K/V tiles instead of staging a F16 copy of the whole cache.  **DONE 2026-09-10, OPT-IN** (`GGML_CUDA_FA_KV_NATIVE=1`, default off - a sub-2 % prefill loss with a large memory win and no cheap way to close it, per the maintainer's rule).  The staged values are bit-identical to the F16 conversion (`dequantize_block_q8_0_f16` computes a single F16 rounding of the exact `int8 × half` product); decode/verify unaffected | **−744 MiB/GPU** (4B 1001.13→257.13) / **−632 MiB** (27B Meta 1121.13→489.13) / −1224 (gemma-4-31B) at ctx 204800 ub 2048, more at ub 1024/512; cost prefill −1.7 %, decode ±0.1 % | **`wip/arch-independent-memory/V4-NATIVE-Q8-KV-PLAN.md`** (mechanism + design + implementation table + reserve matrix + validation + the on-by-default follow-up); `HANDOVER.md` §3.2 |
| **V5** | **native bf16 K/V in the MMA FA kernel** (convert each 16-byte staged chunk in registers — bf16 and f16 tiles have the same byte layout — instead of copying from the whole-cache F16 staging scratch; the per-operand staging source is one shared type code `FATTN_KV_NATIVE_{NONE,Q8_0,BF16}`, subsuming V4's flags) | `wip/arch-independent-memory/patches/0007-v5-native-bf16-kv.patch` (4 files, +171 net, base = block 15) | compute 4B 968.86 → **256.86** (ub 1024 884.82 → 128.82, ub 512 842.80 → 64.80), 27B Meta 1072.86 → **488.86** (ub 512 868.80 → 122.80), gemma-4-E4B 1062.89 → **404.89**, gemma-4-31B 2068.89 → **716.89** MiB/GPU; qwen4exp unchanged; TILE/verify (ub 8) unchanged at 8.09; coherence byte-identical (on vs off vs f16, all models, short + 3k/40k prompts); MTP identical (27B 0.82716, qwen4exp 0.44262); **OPT-IN** through V4's `GGML_CUDA_FA_KV_NATIVE` — cost prefill −0.2 % (pp2048) / +0.3 % (8192) / −1.06 % (20480) / −2.36 % (40960) on the 4B and −0.76 % on the 27B, decode ±0.1 % | `BF16-NATIVE-KV-PLAN.md` §9; `test-backend-ops` FLASH_ATTN_EXT 7859/7859 (2704 bf16 + 365 q8_0 cases green in both arm states) |
| **V2** | *fallback for V3 only*: 1-bit packed mask (bit-exact by construction, no per-cell state) if V3 phase 3.1 proves too invasive | −750 MiB/GPU − 750 MiB host | same, §4 (V2) |

Not in Block 15 at all: **3a** (through-view reuse — measured **zero** reserve win on the 27B/4B) and
everything in `archive/work/`.

## 2. Gate audit

Policy: every win is **on by default**; a tester must be able to switch each one off individually.
**Exception, decided 2026-09-10: V4 and V5 ship opt-in (default off, one shared switch)** - V4 trades
~1.7 % prefill for a large memory win, V5 0.2-2.4 % (growing with prompt length) for a bf16 cache costing
exactly what an f16 one does; in both cases the gap could not be closed cheaply.  The gate is documented
in the table below.

| win | gate today | needed |
|---|---|---|
| W1 | none: the reshape order is unconditional and the chunking is a size threshold (`score_bytes > 128 MB`) | add `GGML_QSA_SCORE_MEM=0` → both off (default 1).  Keep the threshold as the internal policy, not as the A/B knob. |
| W2 | `GGML_QSA_DERIVED_BIAS` (0 = tensor path, 1 = derived, default), `GGML_QSA_DERIVED_VIS` (0/1, default 1), `LLAMA_QSA_SPARSE_FA` (dense masked-FA fallback) | keep as-is, but **strip the `GGML_QSA_DERIVED_BIAS=2|3` diagnostic modes** before promotion. Document the interaction: with `LLAMA_QSA_SPARSE_FA=0` the packed mask must be kept (the gate already encodes this via the shared `qwen4exp_qsa_sparse()` predicate). |
| W3 | none | add `LLAMA_QSA_KEYS_ONLY=0` → keep the (dead) V buffer (default 1). |
| W4 | none — **decided 2026-09-10: it is a bug fix, not a policy** | **no gate.**  A/B with the ready-made revert: `git apply ab/w4-revert.patch` (verified round trip: W4 → +35 lines → revert → pristine) or `git apply -R ../../upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch`; rebuild and re-run `../../wip/arch-independent-memory/repro/ggml-alloc-unused-view.c` (56.00 MiB again = the bug is back). |
| V3 | **DONE (2026-09-10)** | `LLAMA_KQ_MASK_DERIVED` (default 1; 0 = always the packed mask) + the backend/cache capability check (the feature turns itself off where it cannot be correct). |
| V4 + V5 | **DONE (2026-09-10), OPT-IN (one shared switch)** | `GGML_CUDA_FA_KV_NATIVE` (**default 0**; `1` = dequantize q8_0 K/V *and* convert bf16 K/V while staging the FA tiles, skipping the F16 staging scratch for that operand — a model has one KV type, so the two arms never compete).  Unlike the other wins this one is an *enable* switch, not a kill-switch: the maintainer's rule of 2026-09-10 is that a sub-2 % loss with a large memory win and no cheap way to close the gap ships opt-in (`V4-NATIVE-Q8-KV-PLAN.md` §5).  The A/B is the same variable: unset/0 = the F16 staging path, 1 = native. |

Gate names follow the existing convention (`GGML_QSA_*` for the graph-level knobs, `LLAMA_QSA_*` for
the cache-level ones).

## 3. Integration work (DONE 2026-09-10)

1. **Merge the wins into one tree.**  They were validated individually; W2 and W3 overlap in
   `src/models/qwen4exp.cpp` and `src/llama-memory-hybrid-idx.cpp`, and W1 is W2's base — so the
   natural order is W1 → W3 → W2 (or W1 → W2 → W3 with a 3-way merge), then a re-diff of W2 against
   the new base.
2. **Add/repair the gates** from §2 and strip the diagnostic modes.
3. **Re-validate the combination** (protocol in §4).  A merged tree can move the peak in a way no
   single win shows, and the host-side mask build interacts with all of them.
4. **Cut the block** the way the other blocks are cut: commit the combined state on the fork's
   `rdna-boosts` branch as a 15th block commit (never pushed — see `AGENTS.md`), then
   `scripts/make-patches.sh` with the updated tip, and place the exported patch + this README's
   validation record here.  (Nothing under `wip/` is folded into `patches/` without the maintainer's
   go-ahead — Block 15 exists precisely so the campaign's wins *do* get that go-ahead as one package.)

## 4. Combined validation protocol

* **Coherence, byte-identical** (same seed), each gate on and off, in one binary where possible:
  4B / 27B / qwen4exp, at a short and a long prompt; sparse vs dense FA for qwen4exp.
* **Reserve matrix**: compute + host at ub 512 / 1024 / 2048 and at 1x / 2x context, per model; expect
  the W1/W2/W3 numbers above, or better.
* **Adaptive MTP gate** (`tools/mtp-ab.sh`): acceptance ≥ ~0.45 and MTP ≥ plain at depth 3 — the repo
  rule for anything that can move buffer layout.
* **Performance parity**: `tools/ub-sweep.sh` (llama-bench pp20480 / tg256, r=3) at ub 2048 and 1024 vs
  the 14-block build; the L1 measurement showed +1 % prefill, nothing else.
* **`test-backend-ops`** (at least VIEW/CONT/CPY/DUP/CONCAT/FLASH_ATTN_EXT on CPU + ROCm0),
  `test-alloc`, `test-batch-alloc`.
* **No reserve growth** across repeated graph builds (the W4 acceptance criterion).

## 5. Open questions for the maintainer — ALL RESOLVED 2026-09-10

1. **W4 in two places?** — **RESOLVED: both.**  W4 ships in Block 15
   (`patches/0015`) *and* stays staged in `upstream/UPSTREAM-PR-ggml-alloc-unused-view.{md,patch}`;
   the upstream-drop check on 2026-09-10 (against `9cf3bf256`, GitHub
   unreachable from this host) confirmed it is still absent upstream, so
   nothing is dropped at this regeneration.  **A1 is also staged now**:
   `upstream/UPSTREAM-PR-kv-cache-keys-only.{md,patch}` (verified on master:
   the upstream indexer cache really does allocate the dead V — 72.00 MiB
   at ctx 8192, K 24 + **V 48**, with the patch 24.00 MiB and byte-identical
   output).
2. **Split the generic part of W2 out?** — **RESOLVED: yes, and it is staged.**
   `upstream/UPSTREAM-PR-attn-k-null-mask-guard.{md,patch}` (verified on
   master: applies clean, compiles, byte-identical same-seed text).  Stated
   honestly in its notes as **hardening, not a live fix**: every upstream
   construction site builds a mask (and `can_reuse_kq_mask` itself
   dereferences it), so the guarded branch is unreachable upstream today —
   it is what the sibling `attn_kv` class already does, and it is the
   prerequisite for any future null-mask graph (the fork's V3 derived mask is
   the only one that exists).  It stays in Block 15 until upstream takes it.
3. **Block 15 scope.** — **RESOLVED: W1–W4 + V3 + V4 are all IN Block 15**
   (D5 revised earlier: the block waits for V3/V4 rather than following
   them).  V2 (the 1-bit packed mask fallback) was never needed and is not
   in the block; the derived-mask facility shipped as V3.
4. **Beta tester material.** — **RESOLVED: yes**, `BETA-TESTING.md` in this
   directory (gate table, the three measurements, the report template).
5. **Naming.** — **RESOLVED:** the slug is `campaign-memory-wins` →
   `patches/0015-rdna-boosts-block-15-campaign-memory-wins.patch`
   (this directory keeps the `block-15-campaign-wins` name it was created
   with).

## 6. Reference numbers to protect (do not regress)

qwen4exp, ctx 204800, `-ctk/-ctv q8_0`, 3× R9700, per GPU:

| state | ub2048 compute | ub2048 host | ub1024 compute | ub1024 host |
|---|---|---|---|---|
| pristine (14 blocks) | 6690.40 | 1262.70 | 3346.50 | ~ |
| + W1 | 4450.40 | 1262.70 | 2274.35 | ~ |
| W2 alone (W1 off) | 5491.39 | 63.69 | (not measured) | |
| + W1+W2 | **3251.39** | **63.69** | 1675.33 | 33.64 |
| + W3 | (W3 does not change the compute buffer; it shrinks the indexer KV cache 956.26 → **318.76** MiB/GPU) | | | |
| + V3 + V4 (block 15, defaults / V4 on) | 3251.39 / 3251.39 (qwen4exp is not affected by V3/V4) | | | |

bf16 KV (V5), ctx 204800, per GPU (arm off → on):

| model | ub2048 | ub1024 | ub512 |
|---|---|---|---|
| Qwen3.5-4B-Q8_0 (1 GPU) | 968.86 → **256.86** | 884.82 → **128.82** | 842.80 → **64.80** |
| Qwen3.8-27B-Q8_0 (3-GPU Meta) | 1072.86 → **488.86** | — | 868.80 → **122.80** |
| gemma-4-E4B-it (1 GPU, ISWA) | 1062.89 → **404.89** | — | — |
| gemma-4-31B-it (3-GPU, ISWA) | 2068.89 → **716.89** | — | — |
| qwen4exp (3-GPU) | 3298.81 → 3298.81 (no-op) | — | — |

With V5 on, a bf16 cache reserves exactly what an f16 cache does (f16: 4B 256.86, 27B 488.86,
gemma-4-E4B 404.89, gemma-4-31B 716.89 — re-measured in the same session), so the switch removes
bf16's memory penalty without imposing its own.

Dense controls (no qwen4exp involved; block 15 measured with V3 on): Qwen3.5-4B-Q8_0
**1001.13/41.13** (V4 on: 257.13) and Qwen3.8-27B-Q8_0 **1121.13/81.13** (V4 on: 489.13) at ub2048
(the pre-block-15 values 1800.33/840.34 and 1920.33/880.34 are the V3-off states).  Block 15 must
leave the V3-off numbers unchanged and the V3-on numbers as above — both were re-measured from the
delivered patches on 2026-09-10.
