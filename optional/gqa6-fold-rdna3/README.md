# RDNA3 padded GQA-6 tile flash-attention fold (proposed artifact)

**Status: proposed, r359-cut, full runtime validation GREEN; NOT registered in `release.json`.** The artifact is
unnumbered, so `scripts/validate-set.sh` / `apply-all.sh --include-optional` do not discover it. Upstream `main` is
now **r368**; this r359 artifact does **not** strict-apply on r368 (block 05 changed extensively), so a rebase +
re-validation is owed before numbering/registration. See "Base drift and packaging" below.

## What it does

For the supported GQA-6 geometry the ordinary tile launcher selects `ncols2 = 2`, traversing each K/V head three
times per query row. The fold pins `ncols1 = 1, ncols2 = 8`, placing all six real query heads in one padded
eight-column block (one traversal). Columns 6/7 receive zero Q and are never stored; padding-only warps still join
cooperative K/V loads and barriers but skip KQ, softmax, and VKQ arithmetic.

## Implementation (isolation rework)

The fold lives in a **dedicated self-contained kernel TU** (`fattn-tile-gqa6-fold.cuh` + one generated instance
`fattn-tile-gqa6-fold-dkq256-dv256-q8_0.cu`). Ordinary tile instantiations carry **no** fold template parameter and
compile to baseline device code **byte-identically**. Shared changes are host-only: `fattn-common.cuh` `force_split`
plumbing compiled out via `if constexpr`, and `fattn-tile.cuh` host dispatch + census. This removes the
ordinary-kernel codegen perturbation that exposed intermittent gfx1102 BF16/D=96 corruption, and the device-code
template-parameter conflict with optional 0101.

Optional and **default OFF**: `GGML_HIP_FA_TILE_GQA6_FOLD=1` opts in (read once per process); unset/`0` select the
ordinary launcher. Gated to `GGML_CUDA_CC_IS_RDNA3_0`. A one-shot INFO census at `-lv 4` proves the folded arm
launched. Six files: the fold kernel + generated TU + generator input, host-only `fattn-common.cuh`/`fattn-tile.cuh`,
and curated `test-backend-ops` coverage (widths 9..16, KV 32768, derived-mask).

## Identities

| item | identity |
|---|---|
| Base release | `v18-919794500-r359` |
| Canonical upstream base | `919794500475862f6768c353255469dc661cb734` |
| Delivery default tip | `003c88e1f2eb6297fc28fe99559a3c6a7c7545cc` |
| Delivery default tree | `3910c43846612385fbf120d6b8ae0f1b4ecf867a` |
| Clean delivery commit | `efc78d4384e14f6711f065eb7f904f8a27ef5e15` |
| Applied tree | `a9b4a3e0c944efe8022c4cea330bd47822ddde89` |
| Patch SHA-256 | `60baf2dff642ecc70a0ac5ae1c8a51423f4f8db782ea7cc386a6c3e83334286a` |
| Source branch | https://github.com/kapojko/llama.cpp/tree/gqa6-fold-rdna3-r359-delivery |
| Validation record | https://github.com/kapojko/llama-cpp-rdna-boosts-decode/blob/main/validation/r359.md |

## Applicability and fallback

Folding engages only when all existing GQA-optimization requirements hold and: architecture is RDNA3_0; K/Q and V
head dimensions are 256/256; GQA ratio exactly 6; symmetric native Q8_0 K/V; query width 1..8; the env opt-in is set
before process start. Everything else retains the ordinary r359 selection. A small prefill with `-ub <= 8` can also
reach the fold (width-gated, not decode-only).

## Validation (r359, gfx1101 RX 7800 XT + gfx1102 RX 7600 XT) — GREEN

- **Binary detachment:** objdump 21/21 ordinary softcap=false BF16/D=96 compute kernels byte-identical to baseline on
  both GPUs; the previously-failing `<96,96,16,4,false,BF16>` identical (prologue `s_delay_alu` jitter gone).
- **Correctness:** short gate 84/84 ×6 (default/off/on × both GPUs); full `FLASH_ATTN_EXT` 7585/7585 ×4 (off/on × both
  GPUs). The intermittent gfx1102 BF16/D=96 corruption no longer reproduces; failure-dumps empty.
- **Neutrality:** decode-width Q8_0/`--ub 1` exact gate vs **unpatched r359**: default/OFF mean KLD `-0.000000`,
  same-top-p `100.000%`; fold-ON identical; GPU journals clean.
- **Production:** DaisyRose-27B-Q3_K_XL dual-GPU 103:29 split, Q8_0 target / Q4_0 draft K/V, greedy, 36 records
  (depths 2k/32k/96k × think/code/agentic × plain+MTP-3, 600 tok). Fold-on/off greedy text **byte-identical (SHA-256)
  18/18**; MTP acceptance + accepted length unchanged off/on. Fold decode-TG gain (on/off):

  | Depth | Plain off | Plain on | Gain | MTP-3 off | MTP-3 on | Gain | Acceptance off/on |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 2K | 25.992 | 26.227 | 1.0091x | 46.894 | 47.347 | 1.0097x | 0.7758 / 0.7758 |
  | 32K | 22.544 | 23.463 | 1.0407x | 35.809 | 38.077 | 1.0633x | 0.7731 / 0.7731 |
  | 96K | 17.454 | 19.187 | 1.0993x | 21.925 | 24.948 | 1.1379x | 0.7067 / 0.7067 |

  Production cells run once — not repeatability bounds. The production top-5 KLD is a common-support approximation;
  the calibrated exact gate is the decode-width point above.

## Numerical behavior

The enabled fold changes K/V traversal, occupancy, and reduction grouping, so greedy near-ties and MTP acceptance can
move; fold-on output is not *promised* byte-identical to fold-off or an unpatched build. In this r359 validation it
was byte-identical in all 18 cells with unchanged acceptance, and default/OFF is bit-neutral vs unpatched r359.

## Base drift and packaging (open, maintainer decision)

- **r368 drift:** upstream `main` is `v18-919794500-r368` (tree `bc7898a2…`). Block 05 changed extensively
  (kv_rows ABI, derived-mask `src[6..8]`, band work-gate, P=16, r328 re-base); this r359 patch will not strict-apply on
  r368. A rebase + re-validation is required before a numbered, registered, `validate-set.sh`-green artifact.
- **Optional-0101 overlap:** standalone 0102 applies strictly on the r359 default tree (tree `a9b4a3e0…`); 0100+0101
  reproduce `optional_tree` `707eb474…`; but 0102 after 0101 **fails** at `fattn-tile.cuh:1775` (host dispatch) and
  `generate_cu_files.py:17`. The isolation removed the device-code template conflict, not this host/generator
  packaging overlap. Needs a composition decision (or orthogonal re-cut). Not registered; no cumulative tree claimed.
- **Arch coverage:** gfx1101/gfx1102 only. Ordinary-TU source is baseline-identical for all archs by construction;
  gfx1100/gfx1201 runtime acceptance (review R1) still owed.

## Reproduction (standalone, r359)

```bash
git checkout 919794500475862f6768c353255469dc661cb734
bash scripts/apply-all.sh . /path/to/llama-cpp-rdna-boosts   # v18-919794500-r359
test "$(git rev-parse HEAD^{tree})" = 3910c43846612385fbf120d6b8ae0f1b4ecf867a
git am optional/gqa6-fold-rdna3/gqa6-fold-rdna3.patch
test "$(git rev-parse HEAD^{tree})" = a9b4a3e0c944efe8022c4cea330bd47822ddde89
```

## AI usage disclosure

YES — isolation rework designed and implemented with **Qwen4-Max** (model `qwen4-max`) under @kapojko's control; full
end-to-end validation performed on the r359 base. The earlier pre-isolation r359 candidate was assisted by Kimi and is
superseded; the original r353 submission (Kimi-K3.1-Pro) was rejected with rework request R1-R6, addressed here.
