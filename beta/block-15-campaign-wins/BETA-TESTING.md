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
| `LLAMA_QSA_SPARSE_FA` | `1` | the fused sparse QSA flash-attn (dense masked fallback) | slower prefill; the mask is required and kept.  **FIXED 2026-09-11 (11)**: the ninth re-cut fixed the shadowing bug that broke this arm (it gave PPL `1.0558` on qwen4exp for *every* KV type); it is again the valid quality oracle — `tools/qsa-ppl-oracle.sh` must be part of every beta gate sweep (see §4c) |
| `LLAMA_QSA_KEYS_ONLY` | `1` | W3 — the keys-only QSA indexer cache (V buffer allocated again) | **+638 MiB** indexer KV |
| `GGML_CUDA_FA_KV_NATIVE` | `0` (**opt-in**) | **V4 + V5** — stage the K/V tiles natively instead of through the F16 staging scratch: q8_0 is dequantized (V4), bf16 is converted (V5), so the scratch (~800 MiB/GPU at ctx 204800 with q8_0, 712/584/658/1352 MiB with bf16) and its per-ubatch conversion pass are gone for that operand.  Measured q8_0: −744 MiB/GPU (4B), −632 (27B), −1224 (gemma-4-31B); bf16: 4B 968.86 → **256.86**, 27B 1072.86 → **488.86**, gemma-4-E4B 1062.89 → **404.89**, gemma-4-31B 2068.89 → **716.89** (i.e. a bf16 cache then costs exactly an f16 one); coherence byte-identical, MTP acceptance unchanged.  Cost: **q8_0 prefill −1.7 %, bf16 prefill −0.2 % (pp2048) to −2.4 % (pp40960), decode ±0.1 %** — that is why it is off by default.  f16 K/V and every other type keep the old path | with `1`: **−744 MiB** compute (4B q8_0), or the bf16 table above with a bf16 cache; nothing else changes.
  A/B recipe for V5: `-ctk bf16 -ctv bf16` with the switch unset (= 968.86 MiB on the 4B ub 2048) vs `=1` (= 256.86, the same as `-ctk f16 -ctv f16`); same-seed text must be identical in all three.  **Do not mix K/V types** (`-ctk bf16 -ctv q8_0`): any mixed pair drops the attention off the GPU path (pre-existing, documented in `../../patches/README.md`) |
| `LLAMA_KQ_MASK_DERIVED` | `1` | V3 — the derived kq mask (dense + SWA prefill; the mask is not materialized).  Now ON: measured −799 MiB compute **and −799 MiB host** (4B/27B), −809/−811 on the gemmas; prefill −1.1 %, decode −0.7 %, MTP acceptance unchanged.  Auto-disables itself where it cannot apply (decode, small batches, non-MMA kernel, non-CUDA backend, alibi, M-RoPE 2-D, multi-sequence) — so a run that shows no change may simply not have qualified | **−800 MiB** compute **and −800 MiB host** when set to `0` |
| **W4** | always on | — | it is a bug fix, not a policy.  To A/B it: `git apply ab/w4-revert.patch`, rebuild |

## 2. The three measurements per model

> **Added 2026-09-11 (10): a fourth measurement — the perplexity oracle.**  Before promoting, run
> `wip/kv-quant-purity-followups/tools/qsa-ppl-oracle.sh <split> <kv>` against the beta build: the
> *sparse* column must match the delivery's value (qwen4exp tensor: f16 `6.5394`, `iq4_nl` `6.5244`)
> and the *dense* column must be a sane PPL (`6.49–6.55` on qwen4exp, not `~1.05`).  This gate is what
> found the `LLAMA_QSA_SPARSE_FA=0` breakage recorded in §4c; the same-seed/MTP gates cannot see it
> (the production sparse arm is byte-identical to the delivery).

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

### 4c. `LLAMA_QSA_SPARSE_FA=0` (the dense masked oracle) — **FIXED 2026-09-11 (11) in the ninth re-cut
(one line: a shadowed chain variable); kept as the record of the finding.**  Found 2026-09-11 (10) by the eighth re-cut's gate sweep: the dense masked arm gave PPL
`1.0558 ± 0.003` on qwen4exp for **every** KV type (f16 `1.0558`, q4_0 `1.0552`, q4_1 `1.0500`, q8_0
`1.0552`, `iq4_nl` `1.0554`) where the delivery gives `6.49–6.55` (`iq4_nl` `6.4930`, f16 `6.5377`) — a
near-1 PPL means the model effectively sees the answer.  It is **block-15-inherent** (the seventh re-cut
on the older tip `5a0734c9d` reproduces `1.0558` exactly) and **no Block 15 gate fixes it**
(`LLAMA_KQ_MASK_DERIVED=0`, `GGML_QSA_DERIVED_BIAS=0 GGML_QSA_DERIVED_VIS=0`, `GGML_QSA_SCORE_MEM=0`,
`LLAMA_QSA_KEYS_ONLY=0`, and all of them together → all `1.0558`), while `LLAMA_QSA_OFF=1` (`6.5376`) and
the production sparse arm (`6.5244` for `iq4_nl`, `6.5394` for f16) are byte-identical to the delivery.
Repro: `BIN=<beta>/build-beta/bin tools/qsa-ppl-oracle.sh tensor f16`.  **Session brief (2026-09-11,
with the narrowed search space): `../../wip/block15-dense-arm/HANDOVER-2026-09-11-block15-dense-arm.md`.**
Since this was written: the dense arm also differs in a plain text run (delivery `2daa19579316` vs beta
`d910d0b499ec`; the sparse arm stays byte-identical), it *still* differs with **`-fa off`** (so neither
the FA kernels nor V3's derived-mask arm is at fault), and **W4 is exonerated**
(`ab/w4-revert.patch` + rebuild + re-measure → unchanged) — the epicenter is the top-k mask chain in
`build_attn_qsa`, which only this arm builds.

**Resolution (2026-09-11 (11), ninth re-cut).**  It was a variable-shadowing bug in block 15's own dense
path: the V2/V3 refactor wrapped the mask chain in `if (kq_mask != nullptr) { ... }` and declared an
*outer* `kq_mask_top_k`, so the chain's own `ggml_tensor * kq_mask_top_k = ggml_set_rows(...)` inside the
block became a **new local** and the attention read the outer `nullptr`.  The chain's nodes were then
unreachable from the graph output (ggml emitted none of them), the packed mask lost its only consumer
(the allocator dropped it, and the `if (self_kq_mask->buffer)` fill guard skipped `set_input_kq_mask`),
and the dense arm attended with no mask at all.  Removing the inner `ggml_tensor *` fixes it.  **Gates
(identical configs, against the delivery build):** `tools/qsa-ppl-oracle.sh tensor f16` → sparse `6.5394`
/ dense `6.5377` (was `1.0558`); dense-arm texts byte-identical to the delivery — tensor f16
`2daa19579316` (720 chars), tensor `iq4_nl` `3c46e47ab345` (680), layer f16 `e656b50f2cc8` (685), layer
f16 `-fa off` `b96459bf02ca` (703); random-text PPL (`/tmp/rand-text.txt`, gated on the leak: a model that
sees the target scores ≈1 on noise) `19.0589` @ c2560/ub2560 and `7.9682` @ c4096/ub512, both equal to
the delivery's (broken: `1.0205` / `1.0323`); the production arm is untouched.  Full evidence chain:
`../../wip/block15-dense-arm/HANDOVER-2026-09-11-block15-dense-arm.md`.

### 4d. `iq4_nl` (added to the delivery 2026-09-11 (9)/(10)) is W2-sensitive, benign.  With the default
sparse arm the `iq4_nl` greedy text differs from the delivery (`fcb2d47f94cf` vs `acd18ad2d55c`) and the
forced-QSA tensor probe too (`34975a35691aa387` vs `a2e272ce51bf663f`), while f16/q4_1/bf16/q8_0 are
byte-identical.  `GGML_QSA_DERIVED_BIAS=0 GGML_QSA_DERIVED_VIS=0` restores the delivery's values exactly:
W2 re-derives the per-block bias and its last ULP can flip an indexer top-k boundary — and the indexer
cache is quantized with the same `-ctk`, so a different KV quantization moves that boundary.  Benign:
the sparse PPL is `6.5244`, identical to the delivery, and `W=1..8` purity holds.  **2026-09-11 (11):**
the adaptive-MTP acceptance is sensitive for `iq4_nl` too — beta `acc 0.46203`, pos-1
`(0.717, 0.434, 0.226)` vs the delivery's `0.53061` / `(0.755, 0.510, 0.327)` on the same command, and
`GGML_QSA_DERIVED_BIAS=0 GGML_QSA_DERIVED_VIS=0` restores the delivery's numbers exactly (verified), so
it is the same W2 boundary flip and not a quality regression; f16 is bit-identical
(`acc 0.56028` / pos-1 `(0.681, 0.553, 0.447)` on both builds).

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

## 2026-09-12 — eleventh re-cut: block 13's column-blocked shared-expert epilogue

The base moved for a *performance* amendment (no new rule, no changed default): block 13's
`shexp_down_gated_q8_0` was restructured from one block per `(output row, token)` to one block per row
with the whole decode/verify band block-internal, which is **bit-identical** (old-vs-new `.so` A/B: MoE
probe `W = 1..8` `ac8825358d9adfda`, kill-switch `bd138ad2326fbbf2`, §19 text `68c0a24ed8d4`, MTP
`0.87179` — all unchanged) and buys back the 2026-09-11 band amendment's cost (`pl=8` 461.0 -> 475.4 t/s,
`pl=4` +2.4 %, `pl=1` flat).  Base `124abba9e` -> **beta tip `a90f75896`**, tree
`ed6ee74df8b690c5a1584adb3f85c45eda70a09b`, patch **3 811 lines**; `git am -3` merged cleanly and the
patch differs from the tenth re-cut only in the `From <sha>` line (the changed file, `mmvq.cu`, is absent
from this patch).

**Nothing in the tester checklist changes** — the gate table above was re-run on this re-cut's build and
reproduces it exactly (qwen4exp f16 sparse text `804de0576868`, QSA oracle sparse `6.5394` / dense
`6.5377`).

## 2026-09-11 (12) — tenth re-cut: two new delivery rules apply to the beta

The base moved (block 01: the `--spec-draft-n-max ≤ 7` clamp with the `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0`
escape hatch; block 14: mixed K/V types hard-rejected).  Base `484231cb9` -> **beta tip `a796a1d49`**,
tree `b48565e69f77f0c20a20cd75d87c2559d11e6de2`, patch **3 811 lines**; `git am -3` merged cleanly (no
conflict this time), and the only delta vs the ninth re-cut is those three delivery files, so the
gate results above carry over unchanged.

**Two new rules testers must respect:**

* **`-ctk` and `-ctv` must match** for every command (any mismatch now fails context creation with
  `models require the same K and V cache types`).  Every script in this directory already passes the
  same type twice; older notes that used mixed pairs to measure the slow path stay valid as historical
  measurements only.
* **`--spec-draft-n-max` above 7 is clamped to 7** with a visible notice (the binary prints an `E`-level
  line naming the env var; the clamp happens in `common_init_from_params`, so it is visible at the
  default verbosity).  `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` keeps the configured depth and prints a `W`
  notice — use it only for deliberate divergence experiments, and note that `n_max > 15` also
  re-introduces the K-dependent chunked-GDN boundary.  Any recorded acceptance number taken at
  `n_max > 7` (e.g. the adaptive-MTP baseline) must be re-measured at 7 or with the env set.

## 2026-09-11 (11) — ninth re-cut: the §4c dense-arm blocker is FIXED (one line)

The base did **not** move (still the delivery tip `6d3155faa`, tree `0c3f0c2c2f4e7439d9489d45573a4021a8eee106`);
only block 15 changed, by one line of `src/models/qwen4exp.cpp` (plus a paragraph in the commit message
documenting it):

* the chain's `ggml_tensor * kq_mask_top_k = ggml_set_rows(...)` became an assignment to the outer
  variable (it had been shadowing it since the V2/V3 refactor) — see §4c for the mechanism and the
  evidence chain
* base `6d3155faa` -> **beta tip `3712e2dc1`**, tree `e39f8c2b6f0593113b93c4e57c512bc7373a2250`, patch
  **3 811 lines** (the 8th re-cut + 1 diff line + the message paragraph; the file list is unchanged at 23)
* `git am -3` on a fresh `6d3155faa` reproduces that tree exactly; the tree builds (`build-beta`, clean)
* gates (all against the delivery build, identical configs): the oracle sparse `6.5394` / dense `6.5377`;
  dense texts tensor f16 `2daa19579316`, tensor `iq4_nl` `3c46e47ab345`, layer f16 `e656b50f2cc8`, layer
  f16 `-fa off` `b96459bf02ca` (all == the delivery); random-text PPL `19.0589`/`7.9682` (== the delivery;
  the broken build gave `1.0205`/`1.0323`); production untouched: sparse f16 `804de0576868`, q4_1
  `886292b17a93`, `plain == n_max 3 == n_max 7`, MTP f16 `0.56028`/`(0.681, 0.553, 0.447)`==delivery,
  `iq4_nl` MTP the known W2 delta (§4d), `LLAMA_QSA_OFF=1` `6.5376`; KV reserves unchanged
  (`1600.00 + 600.00` MiB vs the delivery's `1600.00 + 1800.00`, both arms — the mask win holds);
  backend suites OK (`FLASH_ATTN_QSA`, `GATED_DELTA_NET`, `FLASH_ATTN_EXT`)

**Tester note:** the recompute-your-own-numbers list is unchanged; only the exact `iq4_nl` MTP/text values
differ from the delivery (§4d) and f16 is bit-identical.


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

---

**Re-cut 2026-09-11 (eighth), after block 08's fifth amendment + the iq4_nl half of F3 step 2.**  The
canonical tip moved from `a0cd6ce02` (tree `0966e66731`) to `6d3155faa` (tree
`0c3f0c2c2f4e7439d9489d45573a4021a8eee106`): block 08 now enables `iq4_nl` as a flash-attention K/V
type (predicate + vec dispatch + the 15 missing `fattn-vec-instance-iq4_nl-*.cu` files + the three
CMake default lists + `dequantize_q4_nl` + the three non-contiguous conversion switches) and block 14
gained the matching QSA/CPU-reference/`qsa_kv_native`/tensor-split-gate/backend-test entries.  The
merge is the same single conflict as the seventh re-cut (`qwen4exp_qsa_sparse()` must accept
`GGML_TYPE_IQ4_NL`) — **use `git am -3`** — and the test file needs no fix this time (block 15's body
already carries the `nullptr, nullptr` call).

* base `6d3155faa` (tree `0c3f0c2c2f4e7439d9489d45573a4021a8eee106`) -> **beta commit `d0f71b2e8`**,
  tree `39540b7f4fd8e8569dee64bfa3ee84bf1b20e75d`, patch 3 787 lines
* the tree builds (`build-beta`, clean) and the exported patch round-trips on a fresh base
* gates (all against the delivery build): qwen4exp f16 plain `804de0576868`, q4_1 plain
  `886292b17a93`, f16 `n_max 3` `0.47009` (pos-1 0.615), `iq4_nl` `n_max 3` `0.52727` (pos-1 0.757),
  27B f16 `0.82716`; width purity `W=1..8` equal to the delivery for tensor/layer × f16/`iq4_nl` ×
  default/QSA-forced (the two exceptions: `iq4_nl`'s forced-QSA tensor hash and greedy text, both
  W2-sensitivity — §4d); `FLASH_ATTN_QSA` 22/22, `GATED_DELTA_NET` 46/46, `FLASH_ATTN_EXT` 5940/5940
* ~~**the new `LLAMA_QSA_SPARSE_FA=0` defect (§4c) is a promotion blocker**~~ **FIXED 2026-09-11 (11)**
  (ninth re-cut): the shadowing bug in `build_attn_qsa` is fixed and the dense arm is again byte-identical
  to the delivery (`qsa-ppl-oracle.sh` sparse `6.5394` / dense `6.5377`).  It stays a mandatory gate.
