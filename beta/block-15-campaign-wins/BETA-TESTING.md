# Block 0015 beta — A/B test checklist

For beta testers with a machine that can build the Block-15 tree.  Purpose: confirm the campaign's
memory wins on *your* models and hardware, and — if something looks wrong — isolate it to a single win
without rebuilding five times.  Every win except W4 is switchable by environment variable.

> **Status: OPEN — the beta window started 2026-09-10; re-cut and re-validated 2026-09-11.**
> Block 15 is **staged in this directory**
> (`block-15-campaign-wins.patch`), **NOT in the delivery** (the delivery is the 15-patch set:
> block 00 + blocks 01-14), and every row below was re-checked **as a combination** on the tree built
> from the beta patch on top of the delivered **15-block** set (fresh worktree at `9113cc188`, strict
> **15/15** `git am` + the beta patch, fresh build): reserves, byte-identical coherence on all five
> models, the MTP gate, and the op suites all reproduce.  The 2026-09-11 re-cut is tip **`fe4f55278`**
> (base `389c5341f`, tree `928852cdc`) and every 2026-09-10 number reproduced **to the last decimal**
> — see `README.md` and `HANDOVER.md` §10.
> V3 is **on by default** (`LLAMA_KQ_MASK_DERIVED`), V4 and V5 are **opt-in through one switch**
(`GGML_CUDA_FA_KV_NATIVE=1`, V4 for q8_0 K/V, V5 for bf16 K/V — see the amendment note in `README.md`),
> default off — a ~1.7 % prefill cost for a large memory win).  What testers should do is reproduce the
> two measurements in §2 on their own hardware/models and report through the template in §3.

> **Amendment (2026-09-10, RDNA3_5 / gfx1151):** V3 now engages on a ROCm/HIP **iGPU** (the derived
> probe previously rejected `GGML_BACKEND_DEVICE_TYPE_IGPU`, silently disabling the win), and a context
> with `--parallel` / `n_seq_max > 1` no longer aborts in `ggml_flash_attn_ext_add_kq_derived` (the
> derived form now requires a single KV stream, so a multi-slot context keeps the packed mask).  On a
> Strix Halo APU a single-slot context gets the full V3 win and the V4/V5 arms are *cheaper* than on
> RDNA4 (V4 **+2.6 %** at pp20480).  iGPU testers: V3's win needs head ≤ 320 (the MMA kernel); head
> 512+ and the qwen4exp fused-QSA path do not use it.

## 0. What Block 15 promises

With everything **on** (including the opt-in V4, i.e. `GGML_CUDA_FA_KV_NATIVE=1`): the generated text
(same seed, `--temp 0`) is **byte-identical** to the previous build, the buffers are smaller, and
throughput does not drop.  With the **defaults** (V4 off) the buffers are the "V3" column below.  Sizes at ctx 204800 / ub 2048 /
`-ctk q8_0 -ctv q8_0`, per GPU:

| model | before Block 15 | with Block 15 |
|---|---|---|
| Qwen3.8-Flash-Next IQ4_XS (qwen4exp) | 6690.40 MiB compute + 1262.70 host | **3251.39 + 63.69** (W1+W2) |
| Qwen3.8-27B-Q8_0 (dense, Meta) | 1920.33 + 880.34 | **1121.13 + 81.13** (V3) / **489.13 + 81.13** (V3+V4) |
| Qwen3.5-4B-Q8_0 (dense) | 1800.33 + 840.34 | **1001.13 + 41.13** (V3) / **257.13 + 41.13** (V3+V4); with a **bf16** cache **256.86** (V3+V5) vs 968.86 without |
| gemma-4-E4B-it-Q8_0 (ISWA) | 1887.35 + 935.37 | **1078.17 + 126.19** (V3) / **452.17 + 126.19** (V3+V4) |
| gemma-4-31B-it-qat-Q4_K_XL (ISWA) | 2753.35 + 897.36 | **1942.18 + 86.18** (V3) / **718.18 + 86.18** (V3+V4) |

## 1. The gates

Set to `0` to disable a win (or to the value noted) and re-measure.  `LLAMA_QSA_SPARSE_FA=0` restores the
dense masked flash-attention path (the packed mask is then kept automatically).

| gate | default | disables | expected effect of disabling (ctx 204800 / ub 2048) |
|---|---|---|---|
| `GGML_QSA_SCORE_MEM` | `1` | W1 — the QSA score-chain memory cuts | qwen4exp compute **+2240 MiB** |
| `GGML_QSA_DERIVED_BIAS` | `1` | W2 — the derived per-block bias (`0` = uploaded 400 MiB tensor) | **+400 MiB** |
| `GGML_QSA_DERIVED_VIS` | `1` | W2 — the derived visibility (the packed `n_kv × n_tps` mask comes back) | **+800 MiB** compute **and +800 MiB host** |
| `LLAMA_QSA_SPARSE_FA` | `1` | the fused sparse QSA flash-attn (dense masked fallback) | slower prefill; the mask is required and kept |
| `LLAMA_QSA_KEYS_ONLY` | `1` | W3 — the keys-only QSA indexer cache (V buffer allocated again) | **+638 MiB** indexer KV |
| `GGML_CUDA_FA_KV_NATIVE` | `0` (**opt-in**) | **V4 + V5** — stage the K/V tiles natively instead of through the F16 staging scratch: q8_0 is dequantized (V4), bf16 is converted (V5), so the scratch (~800 MiB/GPU at ctx 204800 with q8_0, 712/584/658/1352 MiB with bf16) and its per-ubatch conversion pass are gone for that operand.  Measured q8_0: −744 MiB/GPU (4B), −632 (27B), −1224 (gemma-4-31B); bf16: 4B 968.86 → **256.86**, 27B 1072.86 → **488.86**, gemma-4-E4B 1062.89 → **404.89**, gemma-4-31B 2068.89 → **716.89** (i.e. a bf16 cache then costs exactly an f16 one); coherence byte-identical, MTP acceptance unchanged.  Cost: **q8_0 prefill −1.7 %, bf16 prefill −0.2 % (pp2048) to −2.4 % (pp40960), decode ±0.1 %** — that is why it is off by default.  f16 K/V and every other type keep the old path | with `1`: **−744 MiB** compute (4B q8_0), or the bf16 table above with a bf16 cache; nothing else changes.
  A/B recipe for V5: `-ctk bf16 -ctv bf16` with the switch unset (= 968.86 MiB on the 4B ub 2048) vs `=1` (= 256.86, the same as `-ctk f16 -ctv f16`); same-seed text must be identical in all three.  **Do not mix K/V types** (`-ctk bf16 -ctv q8_0`): any mixed pair drops the attention off the GPU path (pre-existing, documented in `../../patches/README.md`) |
| `LLAMA_KQ_MASK_DERIVED` | `1` | V3 — the derived kq mask (dense + SWA prefill; the mask is not materialized).  Now ON: measured −799 MiB compute **and −799 MiB host** (4B/27B), −809/−811 on the gemmas; prefill −1.1 %, decode −0.7 %, MTP acceptance unchanged.  Auto-disables itself where it cannot apply (decode, small batches, non-MMA kernel, non-CUDA backend, alibi, M-RoPE 2-D, multi-sequence) — so a run that shows no change may simply not have qualified | **−800 MiB** compute **and −800 MiB host** when set to `0` |
| **W4** | always on | — | it is a bug fix, not a policy.  To A/B it: `git apply ab/w4-revert.patch`, rebuild |

## 2. The three measurements per model

```bash
# environment used for every run
# NOTE: do NOT set GGML_CUDA_FA_WMMA_256=0 here - that caps the WMMA path at head 128 and for a
# 256-wide head (4B, 27B, ...) it also disables V3 (the derived op needs the MMA kernel), which then
# looks like "the gate does nothing".  The campaign tools under wip/qwen4exp/qsa-memory/tools/ were
# written with =0 for other reasons; drop it when testing V3.
export HIP_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib

# (1) coherence: same seed, temp 0, and compare ONLY the generated text
./build/bin/llama-cli -m MODEL -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 24 \
    --seed 42 --temp 0 --no-display-prompt --single-turn

# (2) buffers (compute + host reserve) — llama-bench does NOT print these
./build/bin/llama-cli -m MODEL -ngl 99 -sm tensor -mg 0 -c 204800 -b 2048 -ub 2048 -fa auto \
    -ctk q8_0 -ctv q8_0 -n 1 -v --single-turn --no-display-prompt 2>&1 |
    grep -E "compute buffer size|memory breakdown"

# (3) speed — ONE bench at a time, never in parallel/background
./build/bin/llama-bench -m MODEL -p 20480 -n 256 -r 3 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 \
    -fa on -ngl 99 -sm tensor

# (3b) if you use draft-MTP, also report the draft acceptance rate
./build/bin/llama-cli -m MODEL -md DRAFT -p "$(head -c 4000 /tmp/prompt3k.txt)" -n 96 \
    --seed 42 --temp 0 --no-display-prompt --single-turn 2>&1 | grep -i "draft acceptance"
```

## 3. Report template

Copy this into your message/issue and fill it in:

```
model/quant:            e.g. Qwen3.8-27B-Q8_0
hardware + GPUs:        e.g. 3x R9700 (gfx1201), 32 GB each, unpinned
flags:                  -ngl 99 -sm tensor -mg 0 -c 204800 -b 2048 -ub 2048 -fa auto -ctk/-ctv q8_0
context / ubatch:       204800 / 2048
block-15 commit/patch:  <sha or patch name>
gate under test:        e.g. LLAMA_KQ_MASK_DERIVED=0
coherence:              IDENTICAL / DIFFERS (attach the two texts)
compute / host:         X MiB / Y MiB      (previous build: X' / Y')
pp20480 / tg256:        A / B t/s          (previous build: A' / B')
MTP acceptance:         0.xxx              (previous build: 0.xxx)   [only with draft-MTP]
what broke / what looks off:
```

## 4. Known and accepted differences — do not report these

* the `[ Prompt: … | Generation: … ]` footer always differs (compare the text, not the process log);
* sparse vs dense flash-attention produce legitimately different floats (different kernels) — compare
  sparse-vs-sparse, dense-vs-dense;
* the adaptive-MTP acceptance probe may read e.g. `0.61616` where another build reads `0.64583`: that is
  a deterministic **buffer-layout** sensitivity of the engine, not an arithmetic difference (proven by
  running the same binary with a zero placeholder tensor).  The gate is **≥ ~0.45**;
* a *higher* reserve for a model that does not use a given win is not a regression if that win's gate
  was off;
* **with a `q8_0` or `q4_0` K/V cache, plain decode and speculative decode legitimately disagree** —
  see §4b.  Do **not** report it against Block 15: it reproduces byte-identically without Block 15.

### 4b. KV cache type: what the measurements in §2 mean (2026-09-11)

The two §2 measurements below use `-ctk q8_0 -ctv q8_0`.  The revalidation of 2026-09-11 showed that
the **KV cache type changes what "same-seed coherence" can mean**, and this is **pre-existing**
(identical results with and without Block 15 — it is not a campaign win):

| K/V type | width purity (`n_max <= 7`, i.e. plain == spec) | KV size, 4B @ ctx 204800 | pp512 / tg32 |
|---|---|---|---|
| f16, bf16 | **pure** | 6400 MiB | 7765 / 99 |
| q4_1, q5_0, q5_1, iq4_nl | **pure** (no native kernel: ~3.4x slower) | 2000 / 2200 / 2400 / 1800 MiB | 2197–2293 / 56–64 |
| **q8_0** | **NOT pure** — `W=1,2` agree, `W=3..8` agree, they differ from each other | 3400 MiB | 7713 / 97 |
| **q4_0** | **NOT pure** — same shape as q8_0 | 1800 MiB | 7696 / 95 |

So: **use `f16` or `bf16` when you are testing plain-vs-speculative *coherence*** (that is the pair whose
purity is guaranteed), and treat a q8_0/q4_0 plain-vs-spec text difference as expected rather than a
regression.  The adaptive-MTP **acceptance** gate (≥ ~0.45) is unaffected — it passes on every type.

**Do not mix K and V types** (`-ctk bf16 -ctv q8_0`, etc.).  This is now a **rejected configuration**
(maintainer decision 2026-09-11): every mixed pair measured is 1.7–3.6x slower than the same-type
equivalent (pp512 2152–4476 vs 7713–7838) and never smaller, so it has no upside.

## 5. What is not in Block 15

* **3a** (allocator through-view reuse) — measured zero reserve win on the dense models;
* **V4** if it costs prefill throughput: it then ships **opt-in** (default off) for people who need the
  last 832 MiB, and its gate row above must be marked `default 0`;
* anything under `archive/work/`.

## 6. If you find a real regression

1. Re-run with the suspected gate off.  If the regression disappears, it belongs to that win — say so
   and the session that owns it will fix or gate it.
2. If it persists with every gate off, it is probably not a campaign win: report it as a
   block-15-vs-15-block difference with the §3 template.
3. Always attach the *generated text* for coherence issues (a diff of the two outputs is enough) and the
   exact command lines — they are usually enough to reproduce it without your model.

## 2026-09-11 (later) — third re-cut

The base moved a third time the same day: block 08's decode/verify FA kernel-family amendment (F1,
`../../GREEDY-PURITY.md` §14) changed `ggml/src/ggml-cuda/fattn.cu`.  Block 15 touches that file too,
but the hunks are far apart, so the re-cut is **metadata/offset-only** (0 changed body lines):

* base `1bcf4e82d` (tree `4104e7d34`) -> **beta tip `0c8099ca2`**, tree `7335b923d`
* apply the patch in *this directory*; it needs the 15-block delivery at `1bcf4e82d`

Nothing in the tester checklist changes: the revalidation numbers above were taken on the previous
re-cut and the delta is metadata only.


---

**Re-cut 2026-09-11 (fourth), after block 13's F2 cause-2 amendment.**  The canonical tip moved from
`1bcf4e82d` to `bfaa83d8a` (tree `4e5f2952f`) because block 13's MoE decode/verify **band-uniformity**
fix changed `ggml/src/ggml-cuda/mmvq.cu`.  Block 15 does not touch that file, so this re-cut is
**metadata-only**: the patch is 3722 lines before and after and only its `From <sha>` line differs; a
re-apply on the new base reproduces the recorded tree exactly.

* base `bfaa83d8a` (tree `4e5f2952f`) -> **beta tip `3f4e0747d`**, tree `d50b4e121`
* apply the patch in *this directory*; it needs the 15-block delivery at `bfaa83d8a`
* the 2026-09-10/11 validation numbers above are unaffected (the base amendment does not touch any
  block-15 operand or kernel: it changes only `MUL_MAT_ID` mmvq-vs-MMQ selection and the
  `mul_mat_vec_q_moe` launch bound, which is block-13 territory)

---

**Re-cut 2026-09-11 (seventh), after the fourth block-14 amendment.**  The canonical tip moved from
`6f07fe67a` (tree `0c9dece6b`) to **`a0cd6ce02`** (tree `0966e66731`) because block 14 gained the QSA
quantized-KV enablement (`q4_0`/`q4_1`/`q5_0`/`q5_1` are now read natively by the fused sparse kernel)
plus the K/V-head chunking fix.  Merge: `git am -3` stops once, in `src/models/qwen4exp.cpp` (block 15's
refactored `qwen4exp_qsa_sparse()` needs the extended type conjunct); `fattn-qsa.cu`, `ggml-cpu/ops.cpp`
and `tests/test-backend-ops.cpp` auto-merge - but the test file needed a **semantic** fix that only the
build found: block 15 adds `cell_vis`/`q_vis` to `ggml_flash_attn_qsa`, so the new `test_flash_attn_qsa`
must pass `nullptr, nullptr` (the tree did not compile until it did).  Beta commit **`5a0734c9d`**, tree
`6b1155b68b1741d7e7c6e8f80b88ed90ce406bd6`, patch **3 787 lines**, subject `[PATCH 15/15]`.
**No beta number changes**: the merge is a no-op at every gate config, verified against the delivery
build - qwen4exp f16 plain `804de0576868` (704 chars) and q4_1 plain `886292b17a93` (694 chars) on both,
27B f16 `--spec-draft-n-max 3` acceptance `0.82716` (67/81, mean len 3.48) on both.  The patch
round-trips (`git am -3` on a fresh `a0cd6ce02` reproduces the tree).

**Re-cut 2026-09-11 (sixth), after the F3 step-1 amendments.**  The canonical tip moved from
`5ad11fd35` (tree `3e7accbd7`) to **`6f07fe67a`** (tree `0c9dece6b`) because block 08 gained the
quantized KV-type enablement (`q4_1`/`q5_0`/`q5_1` as FlashAttention cache types) and block 14 the
QSA-vs-KV-type arm gate + the tensor-split gate narrowing.  Block 15 patches the same
`build_attn_qsa` region, so `git am -3` stops with a real conflict this time and the merge is
functional, not just offset arithmetic: the KV types travel through the graph via two new
`llama_cparams` fields and `qwen4exp_qsa_sparse()` gains the same "QSA can read this type" conjunct the
delivery carries inline.  **Every beta config in this document uses f16 or q8_0 KV, for which the new
conjunct is unconditionally true, so the recorded numbers stand** — but unlike the previous re-cuts the
patch body really does differ (+43 lines) — but the re-cut was checked: the tree builds and, at the
gate configs, the beta output equals the delivery's byte for byte (qwen4exp f16 plain `804de0576868`,
27B f16 `n_max 3` acceptance `0.82716` including the mean length; 27B `q4_1` `0.80723`).

* base `6f07fe67a` (tree `0c9dece6b`) -> **beta commit `8c377b958d89add4d6b6441973e482f02898f359`**,
  tree `34527a2926246893104015d6ca5d12844b14f037`
* a plain `git am` fails (as in the fifth re-cut, and for the same reason) — **use `git am -3`**

---

**Re-cut 2026-09-11 (fifth), after the two same-day band amendments.**  The canonical tip moved from
`bfaa83d8a` (tree `4e5f2952f`) to `5ad11fd35` (tree `3e7accbd7`) because block 13 gained the fused
shared-expert epilogue band (`mmvq.cu`, `ggml-cuda.cu`) and block 14 the QSA decode arm
(`src/models/qwen4exp.cpp`).  **This is the first re-cut that is not purely metadata:** block 15 also
patches `src/models/qwen4exp.cpp`, so the block-14 amendment's 9 added lines shift block 15's hunk
headers and a plain `git am` fails — **apply with `git am -3`** (the 3-way merge resolves it; all other
files still apply cleanly).  Nothing else changes: the patch is 3722 lines before and after and the
only body differences are the `From <sha>` line and the `qwen4exp.cpp` hunk headers/offsets.

* base `5ad11fd35` (tree `3e7accbd7`) -> **beta commit `f3ece1e123905a98059025a7e7a3c7e8e28f54dc`**,
  tree `5316920f130e585e23b9a38eef6e2c3c5940259e`
* the 2026-09-10/11 validation numbers above are unaffected: neither amendment touches a block-15
  operand or kernel (block 13's is the shared-expert epilogue and the mmvq cap; block 14's is the QSA
  indexer arm choice in the *graph builder*), and the beta re-apply reproduces the recorded tree.
  **Re-run the beta A/B gates only if a block-15 file changed** — here nothing did.
