# 2026-11-09 -- GitHub issue #285: the plain norm-rows match carried the gated match's gate index

Reproduction, fix and gates for release `v16-a691b2345-r238` (fork point `a691b2345`, 16 blocks, tip
`513866c5783ed6dba6fcfa7e43a66caf51b92b24`, tree `3d7a69fa5`).  Reported by **@overdoingism** in
[issue #285](https://github.com/stew675/llama-cpp-rdna-boosts/issues/285), with a one-line diff and a
follow-up that measured the effect on stock `r235` as nil.

Box: 1x R9700 (gfx1201, 32 GiB), Ryzen 9 9950X3D, 91 GiB, ROCm 10.1.0-gfx120X, `~/bin/build-llama-rocm`
(ccache), `HIP_VISIBLE_DEVICES=0` pinned on every arm.  The **before** arm is the shipped r237 build
(`f978dcb82` tree), the **after** arm is the r238 tree; the two are alternated run for run, each a fresh
process, and the pre/post binaries are two `bin/` copies of the same build dir (RUNPATH is `$ORIGIN`).

## The defect

`ggml_cuda_norm_rows_try_fuse()` (`norm-gated.cu`, block 10) runs the GATED matcher first and reuses the
same `ggml_cuda_norm_gated_match` for the plain matcher when it declines.  The gated matcher records the
gate `MUL_MAT`'s node index in `m.pre` while it scans, and can still return 0 after doing so: the `UNARY`
after the scan must be `SIGMOID`, and Qwen3.x's delta-net output norm is **SILU**-gated
(`src/models/qwen35.cpp`, `build_norm_gated()`).  The plain matcher set `x`/`w`/`z`/`dst`/`eps` and left
`pre` alone, so the shared tail took `nm.pre >= 0` as "the gated form": with the gate output aliasing `x`
`norm_rows_alias_conflict()` declined the fusion as `gate-out`, and with a non-overlapping allocation the
gate `MUL_MAT` was dispatched early *and* computed again later in graph order.

The reporter's diff (clear `m.pre` in the plain matcher) applies unchanged at r237 and is what r238 ships,
extended so the gated matcher keeps `pre` in a **local** and writes the struct only when the match
succeeds (so no future caller can inherit it), plus default-initialised struct fields.

## Reproduction (`GGML_CUDA_NORM_ROWS_DEBUG=1`, `-lv 4`, `prompts/prose-rdna-boosts.txt`, 5256 tokens)

| model | flags | before (r237) | after (r238) |
|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL | `-c 8192 -ub 2048 -b 2048 -fa on -ctk/-ctv q8_0` | **144 `declined=gate-out`**, 96 `fused=1 gated=0` | **0 declined**, 240 `fused=1 gated=0` |
| Qwen3.5-4B Q8_0 | `-c 8192 -ub 2048 -b 2048 -fa on` | 72 `declined=gate-out`, 48 plain | 0 declined, 120 plain |
| Ling-3.0-tiny IQ4_NL (SIGMOID gate) | `-c 8192` | 180 `fused=1 gated=1`, 0 declined | identical |

Ling is unaffected because its gated match **succeeds**, which is why the existing section B of
`scripts/gate-norm-rows-alias.sh` passed both before and after the fix.  Section C (new) runs
Qwen3.5-4B and fails on the pre-fix build with 264 `gate-out` declines (80 plain fusions of 344
decisions at `-v`).

## What it is worth (dense 27B UD-Q4_K_XL, prompt-eval t/s, interleaved fresh processes)

| prompt | before | after | |
|---|---:|---:|---|
| 5256-token prose, `-c 8192` | 1276.0 (6 runs: 1286.0 / 1276.5 / 1274.3 / 1270.2 / 1272.6 / 1276.6) | **1283.3** (6: 1291.9 / 1287.4 / 1277.9 / 1277.7 / 1281.4 / 1283.3) | **+0.57 %** |
| 31.5 K tokens (prose x6), `-c 40960` | 1167.2 (4: 1162.3 / 1168.3 / 1169.2 / 1169.0) | **1174.6** (4: 1173.0 / 1175.2 / 1175.3 / 1174.9) | **+0.63 %** |

The win is **smaller here than the reporter's own fork measurement** (they read 1.3-1.9 % on `r235` at
4 K/8 K/32 K) and larger than their stock-`r235` follow-up (+0.0/+0.2/-0.3 %).  The reason a declined
norm-rows node is not two launches: the generic `RMS_NORM + MUL` fusion
(`ggml_cuda_try_fuse` -> `ggml_cuda_op_rms_norm_fused`) absorbs it, so what is lost is the 8-rows-per-block
kernel, not a whole extra launch.  Both of those numbers are config- and revision-dependent, so neither is
"the" size of the bug; the defect is a plain match dispatching a node it does not own.

Correctness is unchanged in either direction: same-seed 32-token greedy text is identical in all four arms
(pre/fix x fusion-on/`GGML_CUDA_DISABLE_NORM_ROWS=1`), `149 chars sha=6a4f4535a2d7`, and the driver gate
proves the fused launches bit-identical to the unfused chain at the gated shape
(`bcc36422f9a120ac` in both arms).

## Gates

- `scripts/gate-norm-rows-alias.sh`: **PASS** (section A driver, section B Ling 180/180 gated 0 declined,
  section C Qwen3.5-4B 344/344 plain 0 declined, text `d2ffb97ccb76` == the kill-switch arm).  Negative
  test on the pre-fix build: section C **FAIL** with 264 `declined=gate-out`.
- `scripts/gate-rebase-release.sh` against the shipped r237 build, and `scripts/validate-set.sh` (strict
  `git am`, tree `3d7a69fa5` == `release.json.tree`): recorded in `WORKLOG.md` 261109.002.
