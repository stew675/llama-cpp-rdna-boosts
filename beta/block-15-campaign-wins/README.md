# beta Block 0015 — campaign wins (beta record)

> **Start here: [`HANDOVER.md`](HANDOVER.md)** — the plan, the maintainer decisions of
> 2026-09-10, the merge/gate/validate/cut steps, the state inventory and the open questions.
> This file is the reference: inventory, gate audit, validation protocol, reference numbers.

**Status: BETA — staged, NOT promoted (2026-09-10); REVALIDATED 2026-09-11
against the 15-patch delivery.**  The campaign is
complete and the block-15 patch lives **only in this directory**
(`block-15-campaign-wins.patch`, re-cut 2026-09-11 (10) on base `6d3155faa`); it is **not part of the
delivery** (`patches/` is the 15-patch set: block 00 + blocks 01-14) and is
applied manually on top of the 15-block tree.  The beta window (~4–5 days) is open for tester feedback;
promotion into the delivery set requires the maintainer's go-ahead (at
which point `scripts/apply-all.sh` becomes a 15-block flow).  All seven wins are on by default
except **V4 and V5, which are opt-in through the same switch**
(`GGML_CUDA_FA_KV_NATIVE=1`; see the gate table below) — the policy
exception the maintainer approved on 2026-09-10 (a sub-2 % prefill loss
with a large memory win and no cheap fix ships as an enable switch rather
than a kill-switch), and the explicit instruction for the bf16 item
("treat it similarly to V4 ... gated by the same environment variable").

**Revalidation 2026-09-11 — DONE.**  The delivery moved on since the cut (it is
now a **15-patch set**: block 00 + blocks 01-14 at canonical tip **`389c5341f`**,
tree `928852cdc`, with block 13 amended twice on 2026-09-11), so the beta patch was
re-cut and re-validated end to end.

* **Re-cut an eighth time 2026-09-11 (10)** after the fifth block-08 amendment + the iq4_nl half of the
  F3 step-2 work moved the canonical tip (block 08 gained `iq4_nl` in the FA predicate/vec dispatch/three
  CMake default lists + the 15 missing `fattn-vec-instance-iq4_nl-*.cu` files + `dequantize_q4_nl` + the
  three non-contiguous conversion switches; block 14 gained the QSA/CPU/oracle/`qsa_kv_native`/tensor-split
  entries): base **`6d3155faa`** (tree `0c3f0c2c2f4e7439d9489d45573a4021a8eee106`) -> **beta commit
  `d0f71b2e8`**, tree **`39540b7f4fd8e8569dee64bfa3ee84bf1b20e75d`**, patch **3 787 lines** (`git am -3`;
  one real conflict again, the same place: `qwen4exp_qsa_sparse()` must now accept `GGML_TYPE_IQ4_NL` -
  the *only* body delta vs the seventh re-cut apart from `index`/offset lines; `fattn-qsa.cu`,
  `ggml-cpu/ops.cpp` and `tests/test-backend-ops.cpp` auto-merged, and the test file needed no fix this
  time because the seventh re-cut's `nullptr, nullptr` fix is already in block 15's own body).  The
  exported patch round-trips (`git am -3` on a fresh `6d3155faa` reproduces tree `39540b7f4` exactly) and
  the tree builds (`build-beta`, 0 errors and 0 warnings from the change).
  * **Gate verification against the delivery build (2026-09-11 (10))** - all green except the finding
    below: qwen4exp f16 plain `804de0576868`, q4_1 plain `886292b17a93`, f16 `--spec-draft-n-max 3`
    acceptance `0.47009` (pos-1 0.615), `iq4_nl` `n_max 3` `0.52727` (pos-1 0.757), 27B f16 `0.82716`
    (67/81); width purity `W=1..8` one hash per (split, type) - tensor f16 `dcf1ae667f730879`, tensor
    `iq4_nl` `676fe273a633e4da`, layer `iq4_nl` `3a94c47dbc1470e4`, QSA-forced tensor f16
    `f400a002bd0af7df`, QSA-forced layer `iq4_nl` `146ec7b517ce1576` - **all equal to the delivery**;
    backend-op suites `FLASH_ATTN_QSA` **22/22** (incl. the two new model-geometry `iq4_nl` cases, so
    block 15's `cell_vis` plumbing is right for the new type), `GATED_DELTA_NET` 46/46,
    `FLASH_ATTN_EXT` **5940/5940**; `LLAMA_QSA_OFF=1` PPL `6.5376` == the delivery's `6.5376`; the
    production sparse PPL for `iq4_nl` `6.5244` == the delivery's `6.5244`.
  * **BLOCKER FOUND (block 15, pre-existing, not a re-cut artefact): the dense masked arm is broken.**
    `LLAMA_QSA_SPARSE_FA=0` gives **PPL `1.0558 +/- 0.003` for every KV type** (f16 `1.0558`, q4_0
    `1.0552`, q4_1 `1.0500`, q8_0 `1.0552`, `iq4_nl` `1.0554`) where the delivery gives `6.49-6.55` (f16
    `6.5377`) - a near-1 PPL is the signature of a lost causal constraint, and that arm is this repo's
    documented *quality oracle*, so block 15 cannot be promoted in this state.  Reproduced on the
    **seventh** re-cut too (old tip `5a0734c9d`: same `1.0558`), so it is block-15-inherent, and **no
    block-15 gate restores it**: `LLAMA_KQ_MASK_DERIVED=0`, `GGML_QSA_DERIVED_BIAS=0
    GGML_QSA_DERIVED_VIS=0`, `GGML_QSA_SCORE_MEM=0`, `LLAMA_QSA_KEYS_ONLY=0`, and all five together all
    give `1.0558`; `LLAMA_QSA_OFF=1` is unaffected (`6.5376` = the delivery).  Repro:
    `BIN=beta/build-beta/bin tools/qsa-ppl-oracle.sh tensor f16` (sparse column fine, dense column 1.05).
    The production sparse arm and the no-indexer dense regime are provably untouched, which is why the
    beta's own same-seed/MTP gates never caught it - **`qsa-ppl-oracle.sh` must be added to the beta
    gate list** (see `BETA-TESTING.md`).
  * **Known caveat for the new type: W2's derived bias is not bit-exact for `iq4_nl`.**  With the default
    (sparse) arm the `iq4_nl` greedy text differs from the delivery (`fcb2d47f94cf`/640 chars vs the
    delivery's `acd18ad2d55c`/671 chars) while f16/q4_1 are byte-identical; the forced-QSA tensor-split
    probe differs the same way (`34975a35691aa387` vs `a2e272ce51bf663f`).  `GGML_QSA_DERIVED_BIAS=0
    GGML_QSA_DERIVED_VIS=0` restores the delivery's exact values, i.e. it is W2's re-derivation of the
    per-block bias whose last ULP flips an indexer top-k boundary on a given prompt - and since the
    indexer key cache is quantized with the same `-ctk`, a *different* KV quantization moves that
    boundary.  It is benign: the sparse-arm PPL is `6.5244`, identical to the delivery's `6.5244`, and
    width purity holds.  The promotion record's "byte-identical on every model" claim needs this caveat.
* **Re-cut a seventh time 2026-09-11 (9)** after the fourth block-14 amendment (the QSA quantized-KV
  enablement + the K/V-head chunking fix): base **`a0cd6ce02`** (tree `0966e66731`) -> **beta commit
  `5a0734c9d`**, tree **`6b1155b68b1741d7e7c6e8f80b88ed90ce406bd6`**, patch **3 787 lines**.  One real
  conflict again, in `src/models/qwen4exp.cpp` (block 15's `qwen4exp_qsa_sparse()` helper needs the
  extended "QSA can read this type" conjunct - the amendment added the four nibble types there);
  `fattn-qsa.cu`, `ggml-cpu/ops.cpp` and `tests/test-backend-ops.cpp` auto-merged.  **Auto-merge was not
  enough in the test file:** block 15 adds two inputs to `ggml_flash_attn_qsa` (the derived-visibility
  `cell_vis`/`q_vis`), so the new `test_flash_attn_qsa` case had to pass `nullptr, nullptr` - a *semantic*
  merge fix that only the build caught (the tree compiled only after it).  Verified as a no-op at the gate
  configs against the delivery build: qwen4exp f16 plain `804de0576868` (704 chars) and q4_1 plain
  `886292b17a93` (694 chars) on both, 27B f16 `--spec-draft-n-max 3` acceptance `0.82716` (67/81, mean len
  3.48) on both.  Patch round-trip re-verified (`git am -3` on a fresh `a0cd6ce02` reproduces tree
  `6b1155b68`).
* **Re-cut a sixth time 2026-09-11** after the F3 step-1 amendments moved the canonical tip (block 08
  gained the quantized KV-type enablement, block 14 the QSA-vs-KV-type arm gate + the tensor-split gate
  narrowing): base `6f07fe67a` (tree `0c9dece6b`) -> **beta commit `8c377b958`**, tree
  **`34527a2926246893104015d6ca5d12844b14f037`**.  This one is **not metadata-only**: the block-14
  amendment reworks the very region block 15 refactored (the arm gate now lives in
  `qwen4exp_qsa_sparse()`), so the merge threads the KV type in — `llama_cparams` gains
  `type_k`/`type_v` (filled where the memory module is created) and `qwen4exp_qsa_sparse()` now also
  requires a QSA-native cache type (f16/bf16/q8_0), taking the dense masked path for every other type
  exactly like the delivery's inline predicate.  **It is a no-op for every validated beta config**
  (f16/bf16/q8_0 ⇒ the new conjunct is always true), so no beta number needs re-measuring; the delta
  is +43 patch lines (`llama-context.cpp` +4, `llama-cparams.h` +3, `qwen4exp.cpp` +22 net).
  Verified on the re-cut itself: the full tree builds (`build-beta`) and, against the delivery build at
  the **same** config, the beta's output is byte-identical — qwen4exp f16 plain `804de0576868` (704
  chars) on both, and 27B f16 `--spec-draft-n-max 3` acceptance **0.82716** (67/81, mean len 3.48) on
  both; the new type behaves there too (27B `q4_1` acceptance 0.80723 = the delivery's value).
* **Re-cut a fifth time 2026-09-11** after the two same-day band amendments moved the canonical tip
  again (block 13's fused shared-expert epilogue band, block 14's QSA decode arm): base `5ad11fd35`
  (tree `3e7accbd7`) -> **beta commit `f3ece1e12`**, tree `5316920f13`.  Block 15 touches
  `src/models/qwen4exp.cpp`, which is exactly where the block-14 amendment landed, so a plain `git am`
  now fails — **use `git am -3` / `git apply -3`** (it auto-merges: my 9 added lines shift block 15's
  hunk headers by +9).  After that the patch is again **3722 lines** and metadata-only apart from those
  offsets: the only body differences are the `From <sha>` line and the `qwen4exp.cpp` hunk headers.
* **Re-cut a fourth time 2026-09-11** after block 13's F2 cause-2 decode/verify band-uniformity
  amendment moved the canonical tip: base `bfaa83d8a` (tree `4e5f2952f`) -> **beta tip `3f4e0747d`**
  (tree `d50b4e121`).  Block 15 does not touch `ggml/src/ggml-cuda/mmvq.cu` at all, so the re-cut is
  metadata-only: **0 changed body lines** (3722 lines before and after; only the `From <sha>` line
  differs), and a re-apply on the new base reproduces the recorded tree `d50b4e121...` exactly.  The
  new beta commit differs from the previous one (`0c8099ca2`) in exactly one file — `mmvq.cu`, i.e.
  the base amendment.  No renumbering (`[PATCH 15/15]`).
* **Re-cut a third time 2026-09-11** after block 08's decode/verify FA kernel-family amendment (F1,
  `GREEDY-PURITY.md` §14) moved the canonical tip: base `1bcf4e82d` -> **beta tip `0c8099ca2`** (tree
  `7335b923d`).  Block 15 *does* touch `fattn.cu`, but its hunks sit at lines 166-326 while the fix is
  at ~690, so the re-cut is metadata/offset-only again: **0 changed body lines**.  No renumbering
  (`[PATCH 15/15]`).
* **Re-cut again 2026-09-11** after block 14's hyper-connection band amendment moved the canonical
  tip: base `1d8f53594` -> **beta tip `54859fdda`** (tree `543ccc015`).  Block 15 touches
  `src/models/qwen4exp.cpp` but only in the QSA/indexer regions (its hunks start after the amended HC
  gates), so the re-cut diff is metadata/offset-only: **0 changed body lines**, only the `From`/`index`
  lines and hunk offsets (the qwen4exp hunks shift by +24 lines).  No renumbering (`[PATCH 15/15]`).
* **New beta tip `fe4f55278`** (tree `ffe197e2f`, parent `389c5341f`); the re-cut patch
  replaced `block-15-campaign-wins.patch` in this directory.  Previous tip `377f8e790`
  was cut on `b425aa8f7` = block 14 of the old **14-block** chain.
* **Dependency delta: exactly one file.**  `ggml/src/ggml-cuda/fattn-common.cuh`
  `7442bc22a` → `22eec7d57` (block 00's `ntiles_dst_eff` fix inside `launch_fattn` — the
  query-width-independent KV split).  The other 22 touched files are byte-identical to the
  cut base, so the re-cut changes only the `From <sha>` line, that one `index` line and one
  `@@` hunk header (`+8` lines offset).  Textual apply on `389c5341f`: clean, no rejects.
* **Numbering: no renumbering was needed** (this corrects the first draft of
  `HANDOVER.md` §10.1).  The repo convention is `git format-patch --start-number 0`, so the
  denominator is the *last block index*, not the count: the delivered 15-patch set reads
  `[PATCH 00/14]`…`[PATCH 14/14]`, and the beta patch's long-standing **`[PATCH 15/15]` was
  already correct**.  On promotion the whole set's denominator simply moves `/14` → `/15`
  (regenerate with `make-patches.sh`; 15/15 of the regenerated delivery patches are
  byte-identical apart from that denominator).
* **Clean-apply**: fresh `9113cc188` + `scripts/apply-all.sh` → strict **15/15**, **0
  whitespace warnings**, tree `928852cdc`; then the re-cut block-15 patch → 16 commits, tree
  **`ffe197e2f`** (exactly the re-cut tree).

| check | 2026-09-11 revalidation result |
|---|---|
| reserve matrix (5 models × ub × V4/V5, qwen4exp W1/W2/W3, bf16/V5 table) | **every 2026-09-10 number reproduced to the last decimal** — e.g. 27B 1920.3284/880.3360 → 1121.1252/81.1329; 4B 1800.3284/840.3360 → 1001.1252/41.1329 → 257.1252/41.1329; gemma-4-E4B 1887.3517→1078.1740→452.1740; gemma-4-31B 2753.3517→1942.1759→718.1759; qwen4exp 6690.3987/1262.6954 → W1 4450.3987 → W2-only 5491.3909/63.6876 → W1+W2+W3 **3251.3909/63.6876** with indexer KV 956.26 → **318.76**; bf16 4B 968.8596→256.8596 (ub1024 884.8206→128.8206, ub512 842.8010→64.8010), 27B 1072.8596→488.8596 / 868.8010→122.8010, E4B 1062.8927→404.8927, 31B 2068.8947→716.8947; **bf16+V5 costs exactly f16 (256.8596)** |
| dense width-purity probe (block 00's guarantee, block 15 active) | 27B hashes **identical to the delivered reference**: 1 GPU `4089b4d4` (W=9 `72af52db`), 2-GPU tensor `a4817ee6` (`b059daa6`), 3-GPU tensor `91434ea9` (`bc3faabd`); 4B pure in all four split configs (1 GPU, 2-GPU layer, 2-GPU tensor, 3-GPU tensor), W=9 divergent.  ⇒ **`n_max <= 7` still holds and block 15 changes no FA numerics** |
| staging invariance (the flagged risk: block 00's split heuristic and V4/V5's operand staging share `launch_fattn`) | V4 on == V4 off and V5 on == V5 off **bit-identically** (4B 3-GPU tensor q8_0 `e826b757…`/`7fe106f5…`/`6f0f0cd0…`; bf16 `07a57be6…`/`c80d981b…`) — stronger than the recorded text coherence |
| same-seed coherence, gates flipped | **byte-identical** (only the timing footer differs) on 4B (default/V3off/V4on), gemma-4-E4B SWA, gemma-4-31B SWA, 27B (short **and** 40k prompt, default/V3off/V4on), qwen4exp (default vs all QSA gates off) |
| MoE asterisk | **identical on both builds**: default W1 `ac8825358d9adfda` / W3 `bd138ad2326fbbf2`; `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` → both `bd138ad2326fbbf2` |
| adaptive-MTP gate | 27B 0.90789 (69/76) **identical on both builds**; MoE 0.58378 (324/555) **identical on both builds**; qwen4exp 0.47826 (block 15) vs 0.50000 (delivery) — one token, and qwen4exp's raw logits are **bit-identical across builds** (`dcf1ae66…`/`1c801d63…`), so it is the documented buffer-layout sensitivity, not arithmetic; both pass the >= 0.45 gate |
| op suites | `FLASH_ATTN_EXT` **7859/7859 ROCm0** (6 derived cases) and **7859/7859 CPU** (4/4 backends), `GATED_DELTA_NET` OK, `test-alloc` 0 failures, `test-batch-alloc` OK; W4 repro 16.00 MiB, revert → **56.00 MiB** (bug back), restore → 16.00 MiB with `ggml-alloc.c` byte-identical |
| prefill/decode cost (interleaved same-binary, pp20480/ub2048) | 4B V3 −1.6 % / V4 −1.75 %, decode flat (101.0–101.4); 27B V3 −0.3 %, decode flat (36.99); headline vs delivery: 27B 2-GPU tensor 1993.9 vs 2028.7 pp512 (**−1.7 %**) / 32.019 vs 31.999 tg128, MoE 1 GPU 4801 vs 4816 pp512 (−0.3 %) / 99.94 vs 101.21 tg128 (−1.25 %) — all inside the documented envelope |
| RDNA3_5 (gfx1151) | **not re-run** — no such hardware on this host (3× gfx1201 + a gfx1036 iGPU).  The 2026-09-10 gfx1151 record stands; the re-cut changes no gfx1151-relevant code (the `fattn-common.cuh` hunk only) |
| `--parallel 4` (n_seq_max > 1) | runs with V3 on and off, no abort (the 2026-09-10 `kq_mask_derivable()` single-stream guard holds) |

**Findings recorded during the revalidation (all PRE-EXISTING — identical hashes on the
delivery build — and all outside Block 15's scope; see §7 and
`../../wip/kv-quant-purity-followups/README.md`):**

1. **The dense `n_max <= 7` purity guarantee does not hold for a fast-quantized KV cache.**
   Same-type pairs: f16, bf16, q4_1, q5_0, q5_1 and iq4_nl are **pure** (`W=1..8` bit-identical);
   **q8_0/q8_0 and q4_0/q4_0 are impure** (`W=1 == W=2`, then `W=3..8` — the boundary is `W=2→3`).
   The impure set is exactly the two KV types with a **fast native** both-quantized FA path
   (>7700 t/s pp512); every other quant is ~3.4x slower because it stages through the F16 scratch.
   Text level (27B 3-GPU, ctx 8192, 300 greedy tokens, q8_0 KV): plain `8ed58aa9` (1330 chars) vs
   `n_max 3 == n_max 7` `da56855b` (1406 chars) — a real greedy divergence.  f16 control: all three
   `ce7b9a75` (pure).  `BETA-TESTING.md` §2 prescribes q8_0 KV, so beta MTP/coherence numbers taken
   that way carry this pre-existing impurity — use f16/bf16 when purity matters.
2. **qwen4exp is not width-pure — root-caused 2026-09-11 into TWO stacked causes, and the earlier
   "fused sparse QSA" attribution was wrong** (the whole QSA regime can be switched off with no effect).
   (a) the hyperconnection fusions are gated `nt == 1` (`qwen4exp.cpp:386/458`), so a 1-token decode
   uses `GGML_OP_HC_MIX`/`HC_COMBINE` while an n-token verify batch uses the unfused chain — measured
   98 `HC_COMBINE` dispatches at W=1 vs 0 at W>=2; the fusion is worth +13.1 % decode, so the fix is to
   make `ggml/src/ggml-cuda/hc-mix.cu:273/445` (`GGML_ASSERT(n_tokens == 1)`) serve the decode/verify
   band, not to disable it; (b) a kernel-dispatch band at W>=5 that survives all fusions disabled —
   the same cause as F1.  Both must be fixed for the band to return: `wip/kv-quant-purity-followups/`.
3. **Mixed K/V types are never worth it**: every mixed pair measured is 1.7–3.6x slower than the
   same-type equivalent (pp512 2152–4476 vs 7713–7838) while using **more** memory than the
   same-type quantized pair (e.g. f16/q8_0 is slower than q8_0/q8_0 and larger).  The maintainer's
   2026-09-11 decision is therefore to **reject differing K/V cache types as an accepted limitation**
   of this repo/block (aligned with upstream #25871, which already enforces same-K/V for DeepSeek V4).
4. **The sub-q8_0 quants are the parity gap**: q4_1/q5_0/q5_1/iq4_nl give 1800–2400 MiB (vs 3400 for
   q8_0, 6400 for f16) and are pure, but cost 2197–2293 pp512 / 56–64 tg32.  Note **iq4_nl is the same
   size as q4_0 (1800 MiB), is pure, and is 3.4x slower** — making it native would obsolete q4_0.

**Amendment (2026-09-10): V5 native bf16 K/V.**  Folded into the block the
same day it was designed: a bf16 K/V cache no longer needs the F16 staging
scratch, so with the switch on it costs exactly what an f16 cache costs
(4B 968.86 → **256.86** MiB/GPU at ub 2048, 27B 1072.86 → **488.86**,
gemma-4-E4B 1062.89 → **404.89**, gemma-4-31B 2068.89 → **716.89**;
ub 1024/512 win 756/778 on the 4B and 746 on the 27B; qwen4exp unchanged)
for 0.2–2.4 % prefill depending on prompt length, decode untouched.  The
amendment touched only this directory's `block-15-campaign-wins.patch`
and was re-validated end to end — see §3.4/§9 of
`../../wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`.

**Beta start: 2026-09-10 (tip `377f8e790`, cut on the 14-block chain `b425aa8f7`); re-cut
2026-09-11 to tip `fe4f55278` on the 15-patch delivery (`389c5341f`, tree `928852cdc`).**  The
2026-09-10 tip was `377f8e790` (block 15, amended with V5 and the RDNA3_5/gfx1151 V3 fix; the
original cut was `09a137566`, then `f5ab5350b`).  It applies on top of the **15-block**
delivery built at the fork point `9113cc188` (always regenerate the
delivery from a canonical fork rebuilt at the fork point — see
`../../BASELINE.md`).
The result: **3.44 GiB/GPU + 1.2 GiB host** reclaimed on qwen4exp at ctx
204800 / ub 2048 / q8_0 KV (6690.40 → 3251.39 MiB/GPU compute, 1262.70 →
63.69 MiB host, indexer KV 956.26 → 318.76 MiB/GPU), **800 MiB/GPU + 800
MiB host** on every dense model, and a further **744/632 MiB/GPU** with
V4 enabled — with byte-identical same-seed output, unchanged MTP
acceptance and a ~1.3 % prefill / ~0.3 % decode default cost.

**Validation as a combination** (the important part: the per-win records do
not carry over) was completed on 2026-09-10 both on the merged work tree and
again on the tree built from the **beta patch on top of the delivered 14-block set** (fresh worktree at
`9113cc188`, `apply-all.sh` (14/14) + the beta patch, fresh build):

| check | result |
|---|---|
| reserve matrix, 5 models × ub 2048/1024/512 × V4 off/on | every number reproduces the per-win records; the wins compose additively (qwen4exp: pristine 6690.40 → W1 4450.40 → W1+W2 **3251.39**; W2 alone 5491.39; both W gates off = 6690.40/1262.70 exactly) |
| same-seed coherence, gates flipped | **byte-identical** on 4B, 27B, gemma-4-E4B (ISWA), gemma-4-31B (ISWA) and qwen4exp — 4 configs each (V3×V4), 7 for qwen4exp (W1/W2/W3/V3/V4) — at a short and a 40k-token prompt |
| adaptive-MTP gate | 27B 0.76744 (66/86, mean 3.28) **identical in all four gate combinations**; qwen4exp 0.44262 (54/122) identical in all six and equal to the block-14 baseline; MTP +26 % over plain decode |
| op suites | `FLASH_ATTN_EXT` on ROCm0 (both V4 gates) + CPU (incl. the six derived cases), VIEW/CONT/CPY/DUP/CONCAT, `test-alloc`, `test-batch-alloc`: all pass; W4 repro 56.00 → 16.00 MiB and the revert restores `ggml-alloc.c` byte-identically |
| prefill cost (interleaved same-binary A/B, pp20480/ub 2048) | V3 −1.28 % (4B) / +0.28 % (27B); V4 a further −1.85 % (4B) / −1.72 % (27B); decode within noise |
| clean-apply simulation | fresh worktree at `9113cc188` + `scripts/apply-all.sh` (14/14 strict) + the beta patch + fresh gfx1201 build: reserves, coherence, MTP and op suites all reproduced |
| **RDNA3_5 (gfx1151) pass** (2026-09-10, single Strix Halo, ROCm 7.14; amendment to `0015`, tip `377f8e790`) | two V3 regressions found and fixed — the derived probe rejected `GGML_BACKEND_DEVICE_TYPE_IGPU` (V3 silently off) and `n_seq_max > 1` aborted in `ggml_flash_attn_ext_add_kq_derived` (`kq_mask_derivable()` now rejects `n_stream != 1`; `IGPU` accepted).  After the fix the reserves reproduce the RDNA4 numbers exactly (4B V3 −799.20 compute/−799.21 host, V5 bf16 968.86→256.86, V4 q8_0 1001.13→257.13; 27B 488.86 / 1072.86→488.86 / 1121.13→489.13; Flash-Next W on 3251.39/63.69, indexer 318.76); 14 ROCm + 7 Vulkan gates PASS 16/16, V3/arm byte-identical over 2064-cell pairs, probes clean in both arms on both backends, FLASH_ATTN_EXT 4596/4596 ROCm0 + 7859/7859 CPU, MTP identical (27B 0.79762, Flash-Next draft 0.52727); `--parallel 4` serving restored.  Arm cost lower than RDNA4: V5 −0.4…−0.9 %, V4 **+2.6 %** at pp20480, decode ±0.1 %.  Raw record `../../wip/strix-halo/GATE-2026-09-10-block15-rdna35.md`. |

**Found during the combination pass and documented, not fixed** (out of
scope; reproduces on block 14): `gemma-4-E4B-it` on 3 GPUs with `-sm
tensor` aborts in the meta splitter because its 2 KV heads are fewer than
the 3 devices.  It works on 1 GPU, on 2 GPUs and on 3 GPUs with `-sm
layer`; every other model is unaffected.  (Also caught and fixed before
the cut: the W3 gate was initially wired with inverted polarity — that is
why the combination pass exists.)

## 1. Inventory (all validated, all now in `block-15-campaign-wins.patch`)

| # | win | source | measured effect | validated on |
|---|---|---|---|---|
| W1 | **L2 score-chain memory**: relu before the 4-D reshape + `n_blocks`-chunked assembly with `ggml_concat` | `wip/qwen4exp/qsa-memory/patches/0001-L2a-L2m-qsa-score-memory.patch` (1 file, `src/models/qwen4exp.cpp`) | compute reserve 6690.40 → 4450.40 MiB/GPU at ub2048 (ub1024 3346.50 → 2274.35, ub512 1724.56 → 1188.56); bit-identical output | qwen4exp coherence, ub sweep |
| W2 | **L1 derived QSA bias + visibility, mask prune, input-fill guards** (incl. the `llm_graph_input_attn_k` null-mask guard) | `wip/qwen4exp/qsa-memory/patches/0002-derived-qsa-block-bias.patch` (10 files, +557/−80, base = W1) | compute 4450.40 → **3251.39** MiB/GPU and host 1262.70 → **63.69** MiB at ub2048 (ub1024 1675.33/33.64); coherence byte-identical (derived vs mask, and vs the dense FA fallback); MTP 0.61616 | qwen4exp: 3k + 40k prompts, sparse/dense A/B in one binary, MTP probe, ub sweep |
| W3 | **keys-only QSA indexer cache** (the indexer V buffer is dead: keys-only scoring) | `wip/qwen4exp/keys-only-indexer/0001-keys-only-qsa-indexer-cache.patch` (3 files, 62 lines) | indexer KV 956.25 → 318.75 MiB; box total 88.58 → 86.70 GiB (V was triplicated); perf parity; validation matrix (bf16/f32 ctx, parallel 2, unified, prompt-cache/checkpoint round-trips, MTP 0.741, f16 byte-identical) | qwen4exp |
| W4 | **ggml-alloc: release unused view sources** (3b) | `upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch` (+35 lines, `ggml/src/ggml-alloc.c`; upstream-applicable, clean on `9cf3bf256`) | repro 56.00 → 16.00 MiB; removes the leak that forced W1's concat assembly; no change on current models (latent trap) | master: repro + `test-alloc` + `test-batch-alloc`; fork: coherence byte-identical on 4B/27B/qwen4exp, reserves and MTP unchanged, `test-backend-ops` VIEW/CONT/CPY/DUP/CONCAT OK |
| V3 | **derived kq mask** (the packed `n_kv × n_tps` F16 mask and its host mirror are no longer materialised; the MMA FA kernel derives visibility from compact per-cell state) | `wip/arch-independent-memory/patches/0003` (engine: op + CPU reference) + `0004` (CUDA MMA kernel) + `0005` (graph plumbing + backend probe + enable) | compute **−799.20 MiB/GPU** and host **−799.21 MiB** at ctx 204800/ub 2048 (4B 1800.33 → 1001.13 / 840.34 → 41.13; 27B 1920.33 → 1121.13 / 880.34 → 81.13; gemma-4-E4B ISWA −809/−809; gemma-4-31B −811/−811; scaling exactly `n_kv × n_tps × 2 B`); coherence byte-identical incl. both SWA gemmas and 40k prompts; MTP identical (0.76744); **ON BY DEFAULT** (`LLAMA_KQ_MASK_DERIVED=0` forces the packed mask); cost prefill −1.1 %, decode −0.7 % | 4B/27B/gemma-4-E4B/gemma-4-31B/qwen4exp + `test-backend-ops` FLASH_ATTN_EXT (6 derived cases on CPU, 3 on ROCm0, full suite) + the phase-1 host oracle |
| V4 | **native q8_0 K/V in the FA kernels** (dequantize while staging the shared tiles; the whole-cache F16 staging scratch and its per-ubatch conversion pass are gone for q8_0) | `wip/arch-independent-memory/patches/0006-v4-native-q8-kv.patch` (6 files, +357/−47, base = W1+W2+V3) | compute **−744 MiB/GPU on the 4B** (1001.13 → 257.13) and **−632 MiB on the 27B** (Meta 1121.13 → 489.13) at ctx 204800/ub 2048, more at smaller ub (4B ub 1024 −772, ub 512 −786; gemma-4-31B −1224); qwen4exp control unchanged; coherence byte-identical (4B/27B/both gemmas/qwen4exp, incl. 40k prompts and the TILE ub-8 path); MTP acceptance identical (0.76744); **OPT-IN** (`GGML_CUDA_FA_KV_NATIVE=1`, default off) - cost prefill −1.7 %, decode ±0.1 % | same matrix + `test-backend-ops` FLASH_ATTN_EXT with both gates |

### 1b. Block 15 is CUT (2026-09-10) — the wins are in `block-15-campaign-wins.patch`

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
  the 15-block build; the L1 measurement showed +1 % prefill, nothing else.
* **`test-backend-ops`** (at least VIEW/CONT/CPY/DUP/CONCAT/FLASH_ATTN_EXT on CPU + ROCm0),
  `test-alloc`, `test-batch-alloc`.
* **No reserve growth** across repeated graph builds (the W4 acceptance criterion).

## 5. Open questions for the maintainer — ALL RESOLVED 2026-09-10

1. **W4 in two places?** — **RESOLVED: both.**  W4 ships in Block 15
   (`block-15-campaign-wins.patch`) *and* stays staged in `upstream/UPSTREAM-PR-ggml-alloc-unused-view.{md,patch}`;
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
   `block-15-campaign-wins.patch`
   (this directory keeps the `block-15-campaign-wins` name it was created
   with).

## 6. Reference numbers to protect (do not regress)

> **Revalidated 2026-09-11** against the 15-patch delivery (base `389c5341f`, tree
> `928852cdc`) with block 15 re-cut to `fe4f55278` (tree `ffe197e2f`): **every number in the
> tables below reproduced to the last decimal**, and the pristine rows reproduce the 15-block
> delivery build exactly.  The row label "pristine (14 blocks)" is the 2026-09-10 record's name
> for the then-current delivery — at revalidation it is the 15-block delivery.

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
the beta patch on top of the 14-block tree on 2026-09-10.

## 7. Follow-up work from the revalidation (2026-09-11) — NOT Block 15's scope

All three were found while re-validating, all reproduce **identically on the delivery build** (so
they are pre-existing, not block-15 regressions).  The full brief, evidence, repro commands and
hypotheses live in **`../../wip/kv-quant-purity-followups/README.md`**; the summary:

| # | item | measured | why it matters |
|---|---|---|---|
| F1 | **quantized-KV width purity** — `q8_0/q8_0` and `q4_0/q4_0` break the dense `n_max <= 7` guarantee (`W=1 == W=2` then `W=3..8`); text level: plain `8ed58aa9` vs spec `da56855b` on the 27B | 4B 1 GPU `0edf55a1`/`31a0c1ba` (q8_0), `8125e094`/`619c151e` (q4_0); f16/bf16/q4_1/q5_0/q5_1/iq4_nl pure | anyone speculating with a q8_0 or q4_0 cache gets a different greedy result than plain decode; `BETA-TESTING.md` §2 prescribes q8_0 KV |
| F2 | **qwen4exp width purity** — TWO stacked causes, root-caused 2026-09-11 (the "fused sparse QSA" attribution was wrong) | (1) the HC fusions are gated `nt == 1` (`qwen4exp.cpp:386/458`; 98 `HC_COMBINE` at W=1 vs 0 at W>=2), and the fused path is worth **+13.1 % decode**; (2) a kernel-dispatch band at W>=5 that survives all fusions off — the same cause as F1 | (1) make `hc-mix.cu:273/445` (`GGML_ASSERT(n_tokens == 1)`) serve the decode/verify band (the `dsv4-hc.cu` kernels are already token-generic) so the +13 % is kept; (2) fix it with F1/F3, then re-check (the causes stack) — evidence in `../../wip/kv-quant-purity-followups/` |
| F3 | **sub-q8_0 KV quant parity** — q4_1/q5_0/q5_1/iq4_nl are pure and much smaller (1800–2400 MiB vs 3400 q8_0 / 6400 f16 at ctx 204800) but run at 2197–2293 pp512 / 56–64 tg32 vs 7713–7838 / 95–99 | they have **no native FA path** (F16 staging scratch); the native ones are f16, bf16, q8_0, q4_0 | extending block 15's own `FATTN_KV_NATIVE_{NONE,Q8_0,BF16}` type-code design to them would buy ~3.3x prefill; **iq4_nl is the same size as q4_0 (1800 MiB) and is pure**, so a native iq4_nl obsoletes q4_0 outright.  **Any new native path must be built width-invariant** — do not repeat the q8_0/q4_0 mistake |

**Maintainer decision 2026-09-11 (recorded here):** differing K **and** V cache *types* are to be
**rejected as an accepted limitation** of this repo / Block 15 — the measurements show mixed pairs are
always 1.7–3.6x slower than the same-type equivalent and never smaller, so the configuration has no
upside (upstream already enforces same-K/V for DeepSeek V4, #25871).
