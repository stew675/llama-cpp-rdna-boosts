# WORKLOG — dated delivery records

## 2026-09-15 — `v16-790cf51aa-r4`: the reporter's q4_0 NaN, the prefill band split, and the last four native KV arms

**Release.**  `v16-790cf51aa-r4`, tip `b19c70b341f9ed439bcda2a636fe6e5fa4fa634b`, tree
`7fab975d9518b29aa7d890c1163f13a6c393c5df`; 16 patches + `rdna-boosts-all.patch` + `release.json`
regenerated, `scripts/validate-set.sh` **PASSED** (strict 16/16 `git am`, applied tree == the recorded
tree).  Amends **block 15** only.

**Why (issue #30's second round).**  @briansp2020 re-ran on r3 and reproduced every claim — and found a
regression the gates had missed: **4 NaN failures in `test-backend-ops -o FLASH_ATTN_EXT`** with a
`q4_0` K/V.  Chasing them surfaced a second bug and two open items, all now closed.

**1. The NaN: the tile kernel's K/V type contract.**  The tile kernel is instantiated with ONE `type_KV`
for both operands (that is why it needs a single-type predicate) and builds its native operand
descriptors from that compile-time type; it ignores the launcher's runtime native-type arguments.
`launch_fattn` chose its native read **per tensor**, so a mixed pair (K=q4_0/V=f16, K=q4_0/V=q8_0,
K=f16/V=q4_0, K=q8_0/V=q4_0) fell back to the `F16` tile with the native operand's staging **skipped**,
and the kernel read raw q4_0 bytes as F16.  `launch_fattn` now takes the kernel's native type explicitly
(`kv_native_kernel`; tile = its `type_KV`, vec = `NONE`, MMA = per-operand).  Unreachable from a normal
run (llama.cpp hard-rejects mixed caches) — only the op test could see it.

**2. The second bug: the q4_0 arm's memory win was never delivered.**
`ggml_cuda_flash_attn_ext_get_alloc_size`'s TILE case learned the q8_0 arm but not the q4_0 one, so a
q4_0 cache reserved the F16 scratch the launcher no longer used: `-c 196608` q4_0 **849.04 -> 123.04
MiB**.  The case now mirrors `ggml_cuda_flash_attn_ext_tile_case_type`.

**3. The prefill band split + the arena + the arch gate (TODO item 21, CLOSED).**  Native staging removed
the whole-cache F16 pass (a decode win at depth) but paid a per-tile dequant at prefill.  A prefill
(`n_q > 8`) now stages and decode/verify (`n_q <= 8`) reads natively.  The scratch moved out of the
compute-graph reserve into a per-context, per-stream arena, because the reserve graph's `K->ne[1]` is
`n_ctx` — that is the ~726 MiB the adaptive-MTP `-c 196608` load was short of.  Safe because a
multi-token graph is never CUDA-graph captured.  gfx1201 q8_0 `pp150000` **691.4/1076.9/1199.0**
(1/2/3 GPU, from 661.0/996.0/1111.4), decode and the 123 MiB reserve unchanged.  Arch-gated: gfx1151
measures the native prefill faster at *every* depth, so `prefill_stages = !GGML_CUDA_CC_IS_RDNA3_5(cc)`.

**4. Native arms for `q4_1`/`q5_0`/`q5_1`/`iq4_nl` (TODO item 2, CLOSED).**  The four chunk dequantizers +
the plumbing, closing the last gap in the V4 set.  The 2026-09-14 WIP failed the op test 4703/5951; two
real bugs hid behind that one symptom: the **tile loader's native branch was a hand-written
`type_KV == Q8_0 || Q4_0` test** (so the new instantiations took the F16 branch and read a staging buffer
`need_f16_K == false` had left unwritten — the `hsk=72` NaNs), and the **q5_0/q5_1 chunk helpers took the
low nibble in both halves** (the `lo ? ... : b0 >> 4` test was missing).  A third "fix" — using `qh` bit
`e-4` for the upper half — was wrong and reverted: the reference's `xh_1 = (qh >> (j + 12)) & 0x10` masks
bit 4 of the *shifted* value, i.e. `qh` bit `j + 16`, so the 5th bit is the element index in both halves
(settled by a host test against `dequantize_row_q5_0`).  Gains: tg64 @ d32768 staged -> default gfx1201
+9.9/+10.8/+12.5/+8.8 %, gfx1151 +22.9/+26.0/+26.7/+22.0 % for a 0.6-1.1 % prefill cost.

**Gates.**  `test-backend-ops -o FLASH_ATTN_EXT` **5951/5951 on gfx1201 and gfx1151**; same-seed greedy
text `native == staging` **identical for all eight KV types on both**; `W=1..8` one logits hash per type
(all eight PURE on gfx1151; gfx1201 pure except the pre-existing q4_0 band edges).

**Doctrine update (`GREEDY-PURITY.md` §36).**  The per-quant purity grid was re-measured across all eight
types and five prefill lengths on gfx1201.  It shows the guarantee must be stated **at the level it
holds**: the *text/acceptance* contract (`plain == draft-mtp`, MTP acceptance, one greedy text) is what
the delivery guarantees for f16/bf16/q8_0, while *logits-level* `W=1..8` purity is a **measurement** —
bf16 has one recorded edge (P=200 on the 4B).  Every observed edge is logits-level with the **argmax
unchanged** and the top-2 margin at 2.2+ (bf16 delta 0.014, q4_1 delta 0.064), and all are pre-existing
(reproduced byte-identically with `GGML_CUDA_FA_KV_NATIVE=0`).

**Docs.**  `../AGENTS.md` (the KV-purity critical-facts bullet), `GREEDY-PURITY.md` §36, `TODO.md`
(items 2 and 21 -> Closed), `patches/README.md` (the 2026-09-15 block-15 amendment section),
`wip/issue-30-mtp-decode-regression/MEASUREMENTS.md` §F-H + §I.

**Also 2026-09-15 — the `W=1` vs `W>=2` logits edge: investigated, documented, WON'T FIX.**  §36 had
described a residual `n_q = 1` vs `n_q >= 2` difference and attributed it to the `n_q = 1` launch running
the tile's whole `cols_per_block`.  A launcher dump (`tools/fattn-launch-dump.patch`, `GGML_CUDA_FA_DEBUG2`)
**withdrew that explanation**: the KV split is already width-invariant (`parallel_blocks=8` at `n_q=1/2/4`;
block 00's `ntiles_dst_eff` fix covers the band, `stream_k=0` on the tile path) and `ncols1=1` means there
are no phantom query columns at all.  What remains is a rounding edge inside the FA path — `argmax`
identical in every observed case, delta 0.014-0.064 logits against a top-2 margin of 2.2-2.7, MTP
acceptance bit-identical across arms, and one to two orders of magnitude below the error the coarse KV
quantization itself imposes.  Leading (unproven) candidate: the per-tile mask-derived `i_sup` bound.
Decision: don't chase it — the fix would make every width process the same KV range, taxing the
single-token decode for no measurable reward, and it is the same recurring 0.5-9 % retrofit class as §19.
**Revisit only on an `argmax` change**; re-run the 8-type x 5-length grid (~20 min) whenever a
single-token-tuned kernel changes.  Detail: `GREEDY-PURITY.md` §36 (withdrawn claim marked in place),
`MEASUREMENTS.md` §J, `TODO.md` Closed.

## 2026-09-14 (later) — block-04 amendment: RDNA prefill tuning, now arch- and split-aware

**Why.**  Issue #30's reconciliation left one open finding: the delivery's prefill fell off faster with
depth than stock (`llama-bench -p 150000`, 1 GPU, 27B UD-Q4_K_XL, f16: delivery 609.5 vs stock 686.9;
KV-type-independent, so not q8_0).  Fitting `t = a + b*n` over pp4K..64K showed the delivery's `a`
smaller (the low-depth wins) but its depth slope `b` ~51 % larger (5.73e-9 vs 3.79e-9) — a
per-(query x KV)-cell attention cost.

**Root cause — two FA-config issues.**
* The head-256 `ncols=64` entry in `ggml_cuda_fattn_mma_get_config_rdna` was a **Strix Halo (gfx1151)
  "halo row"** (`nthreads 256, occupancy 1, nbatch_fa 32, nbatch_V2 64, Q_in_reg=false`).  With gqa 6 the
  launcher picks `ncols1 = 64/ncols2 = 8` for every `n_q > 8`, so **all** prefill/wide-verify attention
  used it; that half-tile/Q-out-of-registers shape is ~1.5x per attention cell on RDNA4/RDNA3_0.
* The delivery omitted upstream #28102's AMD `switch_ncols2` block ("minimize wasted compute"), so for
  gqa 6 it picked `ncols2 = 8` (2 of 8 GQA lanes wasted) where stock picks `ncols2 = 2`.

**Change (block 04).**
* The RDNA config is `cc`-aware: `is_rdna3_5` keeps the halo row (it was tuned there), RDNA4/RDNA3_0 take
  upstream's `(256, 2, 64, 128, 128, 64, 1, true)`.  `RDNA3_5`/`RDNA4` are per-gfx in `vendors/hip.h`, so
  the host dispatch uses `GGML_CUDA_CC_IS_RDNA3_5(cc)` and the device constexpr uses the macro.
* `ncols2` is **split-aware**: a new frontend hint `ggml_set_fa_tensor_parallel` (ggml.h/ggml.c), set once
  in the `llama_context` constructor from `split_mode() == LLAMA_SPLIT_MODE_TENSOR && n_cuda_dev > 1`
  (`n_devices()` is 1 under tensor split because the meta device wraps the GPUs, so it counts CUDA
  sub-devices via `ggml_backend_dev_is_cuda`).  The chooser uses generic `ncols2=8` for tensor parallel,
  stock's AMD `ncols2=2` for a whole card.

**Measured (27B UD-Q4_K_XL, f16, `pp150000`; stock 686.9 single / 1111.8 tensor).**

| mode | before | after |
|---|---|---|
| 1 GPU | 609.5 | **703.4 (+2.4 %)** |
| 2-GPU tensor | — | **1087.5 (+6.9 %)** |
| 3-GPU tensor | — | **1218.6 (+9.6 %)** |

`pp64K` single 876.1 -> 946.8.  q8_0 KV prefill is at parity with stock (−1.2 / −0.2 / +2.4 % across
1/2/3 cards); its 4-8 % gap to f16 is the V4 native-staging prefill cost, filed as TODO item 21.  Purity
held: the 4B q4_0 `W = 1..8` band is one hash.

**Verification.**  Block 04 amended in place (blocks 05-15 replayed; two `ggml.h` conflicts resolved by
keeping both declaration sets).  Patches regenerated (16), `rdna-boosts-all.patch` re-cut, `release.json`
-> **`v16-790cf51aa-r3`** (tip `a2c8d06a7`, tree `eb5b7583`).  `scripts/validate-set.sh` PASSES (strict
16/16 `git am` on a fresh `790cf51aa` tarball).  Testing lesson: **`-sm tensor` masked the single-card
regression** — screen with the slope fit at pp8-48K and always measure 1 GPU too.

## 2026-09-14 — block-15 amendment: V4 native staging is the default for sub-F16 KV quants + the q4_0 native arm (issue #30)

**Why.**  Issue #30's reconciliation (`wip/issue-30-mtp-decode-regression/`) isolated a real
delivery-specific regression: with a **quantized** K/V cache the delivery's decode falls off faster with
context depth than stock.  Measured on 1 GPU (27B UD-Q4_K_XL, `tg64`, `-fa auto`): the delivery q8_0
retained 66.1 % of its d0 rate at d65536 vs stock's 80.1 % (18.92 vs 22.43 t/s), q4_0 68.9 % vs 75.9 %
(19.72 vs 21.03).  The delivery's BF16 path was fine and ahead of stock's f16 at every depth (the block-03
predicate); only the quantized types diverged.

**Root cause.**  Block 08's F1 fix (`GREEDY-PURITY.md` §14) deleted the VEC fallback upstream uses for a
quantized K/V at small `n_q`, so the whole delivery band takes the **tile** kernel — which for a
quantized cache is preceded by a **whole-cache F16 staging pass** (`need_f16_K/V = 1`).  That pass is
proportional to `n_kv` and runs on every decode step, so its cost grows with depth.  The block-15 `V4`
native-staging arm removes it but was **opt-in** (`GGML_CUDA_FA_KV_NATIVE=1`), and q4_0 had no arm at
all.

**Change (block 15, one amendment).**

* `GGML_CUDA_FA_KV_NATIVE` becomes a **three-state policy**: unset = **auto** (native q8_0/q4_0 **on**,
native bf16 off), `=1` forces all on, `=0` forces the pre-amendment F16-staging path (the escape
hatch).  The F16-staging pass is the cost for the sub-F16 quants; bf16 already has a native tile/vec
path, so its MMA-scratch arm (V5) stays opt-in.
* A **native q4_0 arm** (`ggml_cuda_fattn_dequantize_q4_0_chunk`, arithmetic-identical to `convert.cu`'s
`dequantize_block_q4_0`) beside the q8_0/bf16 ones, wired through the tile and MMA loaders
(`FATTN_KV_NATIVE_Q4_0`, the predicates, `flash_attn_tile_load_tile_native` /
`flash_attn_ext_f16_load_tile_native`).

**Measured (1 GPU, gfx1201, 27B UD-Q4_K_XL).**  q8_0 d65536 18.92 -> **23.29** (+23 %, stock 22.43) and
q4_0 19.72 -> **22.82** (+16 %, stock 21.03), for ~1.2-1.3 % prefill.  Numerics are unchanged: same-seed
greedy text native == staging (q8_0 `ab94eb7db4d4`, q4_0 `edafcdc7f8df`), `W=1..8` is one logits hash for
every supported type, and MTP `n_max 3` acceptance is unchanged (0.75182).

**Side effect — adaptive MTP at high context loads again.**  The `--spec-draft-n-max 12 -c 196608
-ctk/ctv q8_0` load failure (reported in issue #30) was the **same root cause**: the ~744 MiB/GPU F16
staging scratch was exactly the 260 MiB the MTP draft context was short.  With the new default the exact
config loads at the default `n_slots = 4` and generates (34.76 t/s, acceptance 0.3404);
`GGML_CUDA_FA_KV_NATIVE=0` reproduces the failure.  The deeper recurrent-state snapshot budget and its
levers (including an opt-in f32 -> bf16 snapshot trade to be measured) are filed in
`wip/issue-30-mtp-decode-regression/RECURRENT-SNAPSHOT-BUDGET.md`.

**Action E (#28867 head-256 WMMA threshold) — investigated, no delivery change.**  The reporter's ~20 %
regression is upstream-master-specific: the delivery's `Q->ne[1] > 8` guard already keeps the whole
purity band (`W <= 8`, his repro range) on TILE, and for `n_q = 9..N` the tuned block-04 head-256 WMMA
configs are at parity with TILE (recall `n_max 8` 115.10 vs 115.72 t/s, `n_max 15` 147.19 vs 147.80 t/s,
acceptance bit-identical).  Adopting the MFMA threshold 64 is a ~0.4 % neutral selection change, not a
purity change; left out.

**Canonical chain / verification.**  Block 15 amended in place (`b36517087` -> `9ee71c356`), net tree
`58317e0d64dd01a3622ba90b159ae12d1619c835`; `patches/` regenerated (16 patches) and
`rdna-boosts-all.patch` re-cut.  `scripts/validate-set.sh` PASSES against a fresh `790cf51aa` tarball:
checksums OK, base tree == `97726d3760…`, strict **16/16** `git am`, applied tree == `58317e0d…`.
Release `release.json` bumped to **`v16-790cf51aa-r2`**.

**Follow-ups filed.**  TODO item 2 (native arms for `q4_1`/`q5_0`/`q5_1`/`iq4_nl`; they track stock but
sit ~12-16 % behind f16 at d32k) and item 20 (the issue-#30 umbrella: the recurrent-snapshot budget/levers
and the f32 -> bf16 opt-in measurement).

## 2026-09-13 (latest) — release infrastructure: tag-driven CI, `release.json` as single source of truth, first tagged release `v16-790cf51aa`

**Why.**  The GHCR container workflow failed on every push to `main`.  The run failed *before*
Docker, in all three matrix jobs, at `git am`:

```
error: patch failed: common/speculative.cpp:1621
error: common/speculative.cpp: patch does not apply
...
error: sha1 information is lacking or useless (common/arg.cpp).
error: could not build fake ancestor
```

Two independent bugs.  **(1) A stale fork point:** the workflow pinned `FORK_POINT: 9113cc188`, but the
delivery had been re-based onto `790cf51aa` the same day; the `git am -3` fallback could never rescue
it, because a fresh `git init` over a codeload tarball has none of the preimage blobs named in the
patches' `index` lines.  **(2) The tarball recipe dropped tracked files:** `git add -A` honours
`.gitignore`, so three upstream-tracked files (`build-xcframework.sh` via `/build*`,
`benches/dgx-spark/run-aime-120b-t8-x8-high.log` via `*.log`, and an Xcode `xcshareddata` plist) were
dropped and the reconstructed base tree was `b19ff2b596…` instead of the canonical
`97726d37607304e0215f19aee6af7fd33d1e65d4`.  `git add -A -f` restores the exact tree.

**Redesign (delivery infrastructure, no `patches/` content change).**

- **`release.json` is now the single source of truth** — fork point, canonical base tree, canonical
  tip/tree, block count, and the sha256 of every artifact.  `apply-all.sh`, `validate-set.sh` and both
  workflows read it, so the fork point can no longer drift in one place while another stays stale.
- **`scripts/make-release.sh`** regenerates it (patch hashes are derived; the metadata is inherited
  unless `--base`/`--base-tree`/`--tip`/`--tree` are passed on a re-base).
- **`scripts/validate-set.sh`** is the cheap gate (~1 min, no compiler/Docker): artifact checksums +
  strict `git am` on a fresh tarball of `release.json.base` + base-tree and applied-tree equality.
- **`.github/workflows/validate.yml`** (new) runs that gate on every push/PR.
- **`.github/workflows/docker-ghcr.yml`** is now **tag-driven**: `push.tags: ["v*"]`, manual dispatch and
  the weekly schedule.  The `push: branches: [main]` trigger — the "spawn nine image builds per docs
  commit" behaviour — is **removed**.  The base is read from `release.json`, each build asserts the
  reconstructed tree equals `release.json.tree`, and a `release` job (only on a tag) creates the GitHub
  Release with `rdna-boosts-all.patch`, `patches.tar.gz`, `release.json` and `SHA256SUMS`.
- **`scripts/apply-all.sh`** asserts the applied tree == `release.json.tree` on the strict path (so a
  stale fork point fails immediately, even outside CI).  `actions/checkout` bumped v4 -> v5.
- Docs: `CONTAINERS.md` (release process), `README.md` (#Releases), `AGENTS.md` layout table.

**First release.**  `release.json` now records `release: v16-790cf51aa`, `base: 790cf51aa`,
`base_tree: 97726d37607304e0215f19aee6af7fd33d1e65d4`, `tip: c45244c728dfcbcad86ae95aa97ae76f94ee9f7f`,
`tree: a5683e1b008e3ad197ac2a9e3f99e5b0652df7d4`, `n_blocks: 16`.  Annotated tag **`v16-790cf51aa`**
created on the commit carrying this manifest; the tag push runs the container matrix and cuts the
GitHub Release.

**Clean-apply / verification.**  `scripts/validate-set.sh` PASSES locally against a fresh
`790cf51aa` codeload tarball: checksums OK, base tree == `97726d3760…`, strict 16/16 `git am`, applied
tree == `a5683e1b008e…`.  This is the exact CI step reproduced outside CI.

## 2026-09-13 — issue #30 clamp policy: `--spec-draft-n-max` is raised from 7 to 15 (block 01) + the QSA decode-arm band fix (block 14)

**Mission (issue #30 follow-up).**  The `--spec-draft-n-max` clamp had to be re-decided: the maintainer
wants depth 15, and the park reason was a claim that depth > 7 allows **rewind-induced recurrent (chunked
GDN) corruption** on qwen4exp.  The rule the session was given: no rewind corruption at any allowed
depth; purity above 7 may be traded with a prominent warning; preferred end state 15 everywhere;
fallback 7 for QSA models only.  Result: **there is no rewind corruption, the clamp is now 15 with a
purity notice above 7, and the reported qwen4exp depth-15 divergence past the 2051 selection width was
a QSA decode-arm band flip, now fixed.**

**Canonical chain amended in place** (block 01 `10a7c331d` -> `38fc37c5e`, block 14 `378c9a9d6` ->
`55c733d5c`, block 15 replayed; new tip **`c45244c728dfcbcad86ae95aa97ae76f94ee9f7f`**, net tree
**`a5683e1b008e3ad197ac2a9e3f99e5b0652df7d4`**).  `patches/` regenerated; a fresh `790cf51aa` worktree +
`apply-all.sh` applies **strict 16/16 `git am`**, zero whitespace warnings, produced tree == canonical.
`make-patches.sh` default tip updated; `rdna-boosts-all.patch` regenerated (`sha256
39eab5fa917ea28ad2e43a5925fb5cb03481a27951435b245281c20f3fb19056`).

**1. No rewind corruption at depth 15 — `test-recurrent-state-depth` (new, block 01).**  A deterministic
sweep over the recurrent snapshot machinery: for every `n_rs_seq` 1..15, decode the full verify-shaped
batch, partial-rollback `r` tokens through the snapshot path, replay them, and compare the replayed
logits against a *reference context that never decoded past the rollback point* (bitwise, `eps=1e-5`).
Phase A is the verify shape (`n_tokens = K = n_rs_seq+1`, every `rollback` 1..`n_rs_seq`); Phase B is a
deep draft (`n_tokens = n_rs_batch > K`, which is what `n_rs_batch` exists for).  **All green** on
`qwen35-dense` / `qwen4exp-moe` / `deepseek4-moe` / `kimi-k3-moe` (the generated dummy models), i.e. the
band the delivery allows covers the snapshot set exactly.  The gate is registered as
`test-recurrent-state-depth` + `test-recurrent-state-depth-qwen4exp` in `tests/CMakeLists.txt`.

**2. The qwen4exp depth-15 divergence was the QSA decode arm, not the recurrent state (block 14).**  A
real-model decode/verify width matrix (3x R9700 `-sm tensor`, `P=2500 > width = indexer_top_k + r - 1 =
2051`, f16 KV, token-0 logits) shows the pre-fix behaviour: **W = 1..8 one hash, W = 9..16 another** and
the upper group == the forced-sparse hash — `QSA_DECODE_BAND = 8` gated the dense decode arm
`n_tokens <= 8`, so a depth-8..15 verify batch fell through to the approximate sparse top-k selection
while the W=1 decode stayed dense.  That is the same class as the block-14 cause-2/cause-3 amendments,
re-opened for the built-in draft widths (it only manifests once `n_kv` passes the 2051 selection width —
the "triggers after ~2051 tokens" report).  The arm band is now
`max(QSA_DECODE_BAND, cparams.n_rs_batch)` (`cparams.n_rs_batch` = the longest enabled draft + 1, the
verify-width bound), so the whole verify band takes the same arm as the W=1 decode; the prefill arm is
made disjoint on the same effective band.  Default configs are unaffected (`n_max 3` -> `n_rs_batch 4` ->
band 8; `n_max 7` -> 8), so every recorded reference hash still holds.

Measured (real qwen4exp IQ4_XS, 3-GPU tensor, f16, `P=2500`, `RS=15`): pre-fix W=1..8 `643a8166d8dad677`
/ W=9..16 `1354757f9daf03db` (sparse); post-fix W=1..8 `643a8166d8dad677` / W=9..16 `05be2f7f30dbc426`
(dense).  The dummy `qwen4exp-moe` is now pure W=1..16 for **all eight native KV types** (f16/bf16/q4_0/
q4_1/q5_0/q5_1 `5009c55bca5e01ca`, q8_0 `3d51c0592b7cf913`, iq4_nl `bc19354924bcfb84`); pre-fix
it split `5009c55bca5e01ca` (W<=8) vs `596ec8bf7461da1a` (W>8).

**3. Purity above 7 is lost to the kernel families, not to a defect (the accepted trade).**  On the
real models W=1..8 and W=9..16 never agree even with QSA and FA off (27B UD-Q4_K_XL, f16, `P=2500`:
`8ef5ce3ab2d942dd` vs `7d1e01e82be4ce6b`; with `FA=0` `b5d4df87caa0348e` vs `0c0cb329b59aedc9`, and
qwen4exp with `LLAMA_QSA_OFF=1` likewise), because a verify wider than 8 rows switches kernel family in
more than one place: the FA tile/MMA chooser (`Q->ne[1] > 8`) **and** the matmul family
(`ncols <= MMVQ_MAX_BATCH_SIZE`/`MMVF_MAX_BATCH_SIZE` = 8 uses the decode kernels, above it MMQ).  That
is a near-tie trade, not corruption — end to end the depth-15 output is coherent and only differs from
`plain`/`n_max 7` where a greedy near-tie flipped (27B code-replay 3000 tokens: `plain` == `n_max 7` =
`57776c25503d`; `n_max 15` = `269a445fe4e8`, both rc=0 and coherent; qwen4exp code-replay `n_max 7`
57.9 t/s vs `n_max 15` 40.9 t/s — depth 15 over-drafts on that prompt, it is not corrupt).

**4. The clamp is now 15 with a purity notice above 7 (block 01).**  `common/common.cpp` clamps `> 15`
to 15 (visible `E`-level notice, `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escape hatch) and prints a visible
notice for any depth `> 7` stating that `--spec-type none` and `draft-mtp` may no longer be bit-identical
(the output stays valid and coherent); `common/arg.cpp` help now says `max: 15`.  The old comment blamed
FA purity alone; the new one states the hierarchy — the 15 bound is the **recurrent rollback snapshot
bound** (`n_max + 1 = K <= 16`, the constant the K-independent chunked-GDN threshold was built around), and
purity above 7 is the accepted trade.  The default `--spec-draft-n-max` is still 3, so the clamp
relaxation changes nothing unless the user asks for it.

**5. Revalidation.**  `tests/test-recurrent-state-rollback` unchanged and PASS (qwen35-dense,
qwen4exp-moe); `test-recurrent-state-depth` PASS on four recurrent/hybrid dummy archs; the qwen4exp
`FLASH_ATTN_QSA` suite is a kernel-op suite (graph-arm change only) and is unaffected; the depth-15
same-seed outputs are coherent on both real models.  Docs updated: `patches/README.md` (the block-01
and block-14 amendment notes + the current-state header), `AGENTS.md` (the block-01 bullet + Critical
facts), `GREEDY-PURITY.md` §11/§19, `benchmarks/mtp-adaptive-methodology.md`, `TODO.md`,
`MANIFESTS.md`/`BASELINE.md`/`README.md` headers.  The parked issue-#30 response is corrected (its
clamp description) and its revision bumped.

**6. Adaptive MTP is presented at its recommended ceiling 12, on realistic-length runs (same day).**
With the clamp gone, the adaptive-MTP four-axis table (`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`)
re-measures `--spec-type draft-mtp-adaptive --spec-draft-n-max 12` (block 001's original recommendation)
instead of the clamp-limited 7.  **The first two cuts of that table were wrong and were re-measured the
same day, and the two mistakes are the point:**

* **Reasoning.**  They ran every axis with the model's default reasoning mode; Qwen3.8 emits a thinking
trace for the prose and code prompts, so those two columns measured *thinking*, not content (the code
prompt at `-n 256` never reached any Python).  Corrected with `--reasoning off` for P/C/K and
`--reasoning on` for R (the flag is part of the chat template).
* **Length.**  They used `-n 256`, which measures the drafter/controller warm-up, not the mode.  At
  `-n 256` the code axis at ceiling 12 read **-5 %** vs fixed `n3`; at `-n 3000` it is **+28 %**.  The
  controller's mean accepted length goes 4.32 -> 7.02 as it warms up.  A short spot test inverts the
  ranking.

**Final protocol and result** (27B UD-Q4_K_XL, 1 GPU, f16, `-n 3000`, reasoning pinned): plain ~28.5-28.9
t/s everywhere; fixed `n3` R 46.4 / 0.57781 / 2.73, P 55.9 / 0.79379 / 3.38, C 63.7 / 0.91663 / 3.75,
K 68.1 / 0.99200 / 3.98 (stock `n3` within ~2 % on every axis); **adaptive `n12` R 46.0 / 0.57632 / 2.74,
P 63.4 / 0.50654 / 5.10, C 81.8 / 0.57863 / 7.02, K 109.6 / 0.96320 / 8.95**.  So against fixed `n3` the
mode is reasoning -1 %, prose **+13 %**, code **+28.5 %**, recall **+61 %**; against the old ceiling 7
it is prose +26 %, code +35 %, recall +44 %.  Acceptance is *lower* than fixed `n3` (it drafts deeper and
rejects more) but throughput is higher -- acceptance alone is not the metric.  At the reporter's exact
`n8 + p-min 0.55` configuration the delivery is ahead on every axis at both `n7` and `n8` (+0.8 % to
+8.2 %).  Text purity at `-n 3000` (no `-lv 4`): fixed `n3` == plain on every axis; adaptive `n12` ==
plain on reasoning (`98d4e36a79fb`) and recall (`a87c4318b649`) and diverges on prose (`27f3f7d3f80c`
vs `7ec08bc22946`) and code (`48241ec079f6` vs `a2eceaad5743`) -- the documented above-7 kernel-family
trade, which only appears once the run is long enough to hit a near-tie (at `-n 256` all four matched).
Stock is not pure under the protocol.  Depth 12 is inside the hard 15 bound.

The **length and reasoning requirements are now recorded in `benchmarks/mtp-adaptive-methodology.md`
(gate rule 0) and `prompts/README.md`**.  A short run is valid only as a correctness smoke test, never
as a performance verdict.

**Lesson.**  "Depth > 7 is unsupported" had been resting on one stated reason (FA purity) while the real
qwen4exp effect was a different band (the QSA arm).  The no-corruption result came from a deterministic
reference-context sweep, not from acceptance numbers: acceptance is not a correctness signal (a
self-consistent corrupted pair can accept *more*), and over-drafting at depth 15 looks like a drop too.

## 2026-09-13 (latest) — block-14 amendment (ninth): the re-base's `ncols_opt` broke the pair fusion (dense prefill −14-48 %)

**Canonical chain amended in place** (block 14 `20bf37962` -> `378c9a9d6`, block 15 replayed; new tip
**`f27dc6d8006188d00ff96dadab6eb0edf79e2b7c`**, net tree
**`bbbe005e95381301fdc71e5d636f448bab147a65`**).  `patches/` regenerated; a fresh `790cf51aa` worktree +
`apply-all.sh` applies **strict 16/16 `git am`**, zero whitespace warnings, tree == canonical.
`make-patches.sh` default tip updated; the net patch regenerated.

**The bug.**  The 2026-09-13 re-base merged upstream `d4abd573f`, which added `ncols_opt` to `mmq_args`
(the tile heuristic optimises `ntiles_x = ceil(ncols_opt/J)` and stops at the first `J` that covers the
row).  The standalone MMQ path passes it, but block-14's `ggml_cuda_mul_mat_q_pair` builds its
`mmq_args` by hand in **both** arms and still stopped one initialiser short, so the field defaulted to
`0` and every `J` gave `ntiles_x == 0` - the loop kept the first candidate, `J = 8`, the narrowest and
slowest tile.  Dense FFN gate+up pairs want `J = 64..128`, so the fusion was up to **2.2x slower than
not fusing**.  Invisible on the qwen4exp `MUL_MAT_ID` pair it was written for (each expert sees few
tokens, correct `J` ~8), and the pair A/B had only ever been run on qwen4exp.

**The fix.**  `mmq.cu`: both pair arms set `ncols_opt` like the standalone (dense: `dst->ne[1]`;
`MUL_MAT_ID`: the RDNA per-expert average `(ne12*n_expert_used + ne02 - 1)/ne02`).  `mmq.cuh`: the
heuristic falls back to `ncols_max` when `ncols_opt <= 0`, so a caller that predates the field can
never silently pick the worst tile again.

**Measured** (`llama-bench`, f16 KV, pp4096, `-r 2`, 3x R9700; pre-rebase = the archived tip
`907799de3` @ `9113cc188` rebuilt in a worktree, buggy = the pre-fix rebased tip `6303f0489`):

| model / config | pre-rebase | rebased (buggy) | **rebased + fix** | stock `790cf51aa` |
|---|---|---|---|---|
| 27B Q8_0, 1 GPU | 1348.3 | 623.0 | **1363.2** | 1203.2 |
| 27B Q8_0, `-sm tensor` | 2147.7 | 1717.6 | **2175.5** | 2001.7 |
| 27B UD-Q4_K_XL, 1 GPU | 1262.1 | 904.6 | **1264.0** | 1094.1 |
| 27B UD-Q4_K_XL, `-sm tensor` | 2016.4 | 1692.7 | **2039.9** | 1839.3 |
| 4B Q8_0, 1 GPU | 7127.9 | 5386.1 | **7303.6** | 5807.5 |

Numerics unchanged: the fused pair is byte-identical to the unfused path (27B same-seed
`d03d0bc727a8` with and without `GGML_PAIR_DENSE_OFF=1`); only the tile width changes.  qwen4exp is
unaffected (controlled pre-rebase A/B, `-b 2048 -ub 2048`, tensor, pp8192: f16 sparse 2405.3 ->
**2435.9**, f16 dense 2586.8 -> **2657.0**, `iq4_nl` sparse 2043.0 -> **2456.7** from item 3).

**Lesson.**  The pair fusion's A/B was only ever run on qwen4exp; the dense-arm prefill A/B was never
added to the re-base checklist, so a silent aggregate-init default slipped through.  The `mmq.cuh`
fallback is the guard against the class.  See `patches/README.md` (2026-09-13 block-14 (ninth)) and
`TODO.md` (Closed).

**Test-infrastructure follow-up (same day):** added `prompts/` — versioned, hash-stable test prompts
(`prompts/README.md` records size, token count and sha256 per prompt; a shipped prompt is never edited
in place).  First entry: `prompts/prose-rdna-boosts.txt` (16074 B, 5298 tokens, `sha256 fabdec65…`),
the prompt used for the issue-#30 reproduction and the Protocol-A MTP gate.  The issue-#30 reply now
points at it instead of pasting the prompt inline, so reported numbers are tied to a committed hash.
Also added `scripts/extract-generated.py`, the backspace-aware generated-text extractor the purity gate
hashes with (a naive `sed`/`grep` slice does not reproduce the values).  Also corrected the MTP test
procedure: the drafter is the **MTP head built into the target GGUF** (`blk.<n>.nextn.*`,
`nextn_predict_layers`), used automatically when no `-md` is passed.  The old standalone
`mtp-Qwen3.8-27B-Q4_0.gguf` is a different drafter and changes the numbers (27B UD-Q4_K_XL, n3:
0.57554 acceptance / 46.9 t/s with `-md` vs 0.61654 / 47.1 t/s with the built-in head).  Docs now use
no `-md`; the separate file is not needed.

## 2026-09-13 (even later) — block-08 amendment (seventh): the fused MoE router is bit-identical — TODO item 19 closed

**Canonical chain amended in place** (block 08 `8c072080a` -> `ffa7c1c1b`, the rest replayed; new tip
**`6303f04894fa6251f7e8c9e9eff8742a24267113`**, net tree
**`311f3acebe82a65b1b6f38d3e77997c31910c7dd`**).  `patches/` regenerated from the rebuilt
`~/llama.cpp` chain; a fresh `790cf51aa` worktree + `apply-all.sh` applies **strict 16/16 `git am`**,
zero whitespace warnings, and its tree equals the amended tip tree.  `scripts/make-patches.sh` default
tip updated; the single net patch regenerated.

**The bug (TODO item 19).**  The sixth amendment's absolute `iq4_nl` text move exposed it: the fused
MoE router (`ggml_cuda_op_topk_moe`) was **not** bit-identical to the generic
`soft_max -> reshape -> argsort -> view -> get_rows -> [norm] -> [scale]` chain, and whether the fusion
fires is decided by `ggml_cuda_check_fusion_memory_ranges()`'s **buffer-address overlap** test.  So the
model output depended on the allocation plan: moving the QSA indexer `get_rows` off the CPU flipped the
fusion coverage and changed the greedy text.  Three independent gaps: (1) the fused softmax used a flat
32-lane butterfly while the generic `soft_max_f32`/`block_reduce` uses a per-warp butterfly over each
consecutive 32-column group followed by a cross-warp butterfly over the per-warp results (36 % of
random 512-value rows disagree, up to 2.4e-7 relative); (2) the fused norm accumulated the selected
weights in the per-winner lanes and multiplied by `1/sum` while the generic chain is `sum_rows -> clamp
-> div` (`weights[i] / sum`); (3) the generic CUDA argsort is a **non-stable** bitonic network, so its
top-k set/order for exact ties (4 in one 3.3k-prefill + 64-token run) disagrees with the fused
iterative argmax's smaller-index tie-break — and the CUDA CUB argsort path (`SortPairsDescending`) **is**
stable, so the two CUDA argsort implementations already disagreed with each other.

**The fix.**  `ggml/src/ggml-cuda/topk-moe.cu`: the softmax reproduces the generic two-phase
`block_reduce` order (with the `experts_per_thread == 1` single-warp path preserved), the norm sums the
selected weights in the generic `reduce_rows_f32` order (`warp_reduce_sum(lane j < n_expert_used ?
output_weights[0] : 0)`, lane `j` holding selection `j`'s weight) and **divides** by the clamped sum.
`ggml/src/ggml-cuda/argsort.cu`: the bitonic network breaks ties by index (smaller index first for
`DESC`), matching CUB and the fused router.  `ggml/src/ggml-cuda/ggml-cuda.cu`: the
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` A/B kill-switch (kept).

**Validation** (3x R9700 gfx1201, ROCm 7.14, qwen4exp `IQ4_XS`, `/tmp/prompt3k.txt`, `--seed 42
--temp 0`, `-c 32768 -b 2048 -ub 2048`):

* fused == `GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` for **all eight native KV types**
  (f16/bf16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl), on `-sm tensor` and on `-sm layer` (the split where the
  tie divergence reproduced: `6e2290d44875` vs `8bd14f326f2b` pre-fix, one hash post-fix); forcing
  the fusion (guard ignored) gives the same hash as both.
* `plain == n_max 3 == n_max 7` within every native KV type (`iq4_nl` `086df944f6af`, `f16`
  `92d01d72f895`, `q8_0` `c4000a0285f3`, `q4_0` `28857dc2b3d1`, `bf16` `ba4d858ae2f6`, `q4_1`
  `3e04ba1e7908`, `q5_0` `348c743eb1b2`, `q5_1` `a5b6a81c33fa`) and on `-sm layer` for iq4_nl/f16.
* the pre-fix *unfused* reference is now the fused hash too (`iq4_nl` tensor `086df944f6af`; pre-fix
  fused `14a1a3f257f4` != unfused `086df944f6af`).
* MTP `n_max 3` iq4_nl acceptance 0.59091 (pos-1 0.783, mean len 2.70).
* `test-backend-ops test` **18065/18065** (`ARGSORT`/`TOP_K`/`GET_ROWS` pass; `test_argsort` data is
  tie-free by construction); 4B `Qwen3.5-4B-Q8_0` `-sm tensor` coherence `1c5d32ac537d` unchanged
  (dense, no router).
* qwen4exp pp2048/pp8192/tg128 (`llama-bench`, iq4_nl) 1739/1748/48.1 -> 1715/1741/48.0 t/s, within the
  run-to-run noise.

**Scope note.**  The argsort change makes the CUDA bitonic path deterministic and consistent with the
CUDA CUB path; the CPU `std::sort` comparator leaves ties unspecified, so there is no cross-backend tie
contract to preserve.  See `patches/README.md` (2026-09-13 block-08 (seventh) section), `TODO.md`
(item 19 closed) and `GREEDY-PURITY.md` §31, and `upstream/UPSTREAM-PR-moe-router-tie-break.{md,patch}`.

## 2026-09-13 (later) — block-08 amendment (sixth): the `iq4_nl` `GET_ROWS` CPU fallback — TODO item 3 closed

**Canonical chain amended in place** (block 08 `de5246ada`, the rest replayed; new tip
**`ab2fabb440ac909e02e0482cabd673c339106b57`**, net tree
**`e279b222e8e98a7574814929d4b6d97edae32a48`**).  `patches/` regenerated from the rebuilt
`~/llama.cpp` chain; a fresh `790cf51aa` worktree + `apply-all.sh` applies **strict 16/16 `git am`**
and its tree equals the amended tip tree.  `scripts/make-patches.sh` default tip updated; the single
net patch regenerated.

**The bug (TODO item 3).**  qwen4exp prefill with `--cache-type-k iq4_nl` was ~8-12 % slower than
f16/`q4_0`/`q4_1` at pp8192 and the gap grew with context (pp32768 1992.1 vs 2434.5) although `iq4_nl`
and `q4_0` share the 18-byte block layout.  The QSA indexer key cache tracks `type_k`, so the indexer
gather (`ggml_get_rows` over the 128-wide indexer key view) got an `iq4_nl` source; the CUDA
`GET_ROWS` support predicate required `ne[0] % QK_K == 0` for `IQ4_NL`/`MXFP4` (those types were only
wired to the QK_K super-block kernel), and 128 % 256 != 0, so the HIP backend rejected the op and the
scheduler ran it on the **CPU**.  One `GET_ROWS` per indexer-bearing layer became a D2H/H2D round trip
with a `hipStreamSynchronize`; a qwen4exp prefill graph went from 2 to **26** splits and the GPU sat at
0.62 busy/span vs `q4_0`'s 0.958.  The dense-shortcut arm hid it below the indexer selection width
(2051), which is why pp2048 was flat.

**The fix.**  `getrows.cu` dispatches `iq4_nl` on `ne00 % QK_K` (whole super-blocks keep
`get_rows_cuda_kq<32, ..., dequantize_iq4_nl>`, any other width takes
`get_rows_cuda_q<QK4_NL, QR4_NL, dequantize_q4_nl>`); the `GET_ROWS` predicate accepts
`ne00 % QK4_NL == 0` for `IQ4_NL` (`MXFP4` keeps the `QK_K` requirement — it has no sub-block
dequantize); `test-backend-ops.cpp` gains `iq4_nl` `GET_ROWS` cases at 32/128/160/224 columns.

**Validation** (3x R9700 gfx1201, ROCm 7.14, 3-GPU `-sm tensor`):

* `test-backend-ops test -o GET_ROWS`: **219/219** (was 215; the four new sub-`QK_K` cases run on the
  GPU and match the CPU).  With `max_nmse_err()` temporarily forced to 0 for `iq4_nl`, the new path is
  **bit-exact** against the CPU at 32/128/160/224/256/512/1024 columns — the moved op itself is pure.
* Graph splits for `iq4_nl` pp4096: **142 -> 22** (`q4_0` is 22).
* qwen4exp `iq4_nl` prefill, interleaved same-session: pp8192 **1815-1951 -> 2385-2422 t/s** (= f16
  2348-2416 / `q4_0` 2316-2413); pp32768 **1754-1781 -> 2423-2430** (+36 %).
* `plain == --spec-type draft-mtp n_max 3 == n_max 7` for `iq4_nl` (tensor):
  **`c0d44c479ee1` -> `14a1a3f257f4`**; the f16 (`30d27ad1fc6d`) and `q4_0` (`912c03f2effc`) controls
  are unmoved, and the 4B `Qwen3.5-4B-Q8_0` `-sm tensor` coherence is `1c5d32ac537d` (unchanged).
* MTP `n_max 3` iq4_nl acceptance 0.670 -> 0.677 (pos-1 0.812 -> 0.906), both far above the gate.

**The absolute `iq4_nl` text hash moves — and it had to.**  The `get_rows` values are bit-identical,
but removing the host split changes the buffer addresses, and `ggml_cuda_check_fusion_memory_ranges()`'s
address-overlap test then flips the **MoE-router `topk_moe`** fusion coverage: pre-fix `iq4_nl` ran the
fused router for ~540 sites and the generic chain for ~612 per trace (an address-layout accident),
where `q4_0` runs 24/1128.  Post-fix `iq4_nl` is layout-identical to `q4_0`; a temporary
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION` A/B moves the text (`14a1a3f257f4` -> `086df944f6af`), i.e. the
fused router is **not** bit-identical to the generic chain and its selection is address-dependent.
The purity invariants that matter (width `W = 1..8`, `plain == n_max 3 == n_max 7`, the controls, the
4B coherence) all hold; only `iq4_nl`'s absolute text moves.  The router-fusion address sensitivity is
filed as a new `TODO.md` item (upstream `ggml-cuda.cu`).

**Instruments.**  `GGML_SCHED_DEBUG=1` (split count) and `=2` (per-node backend assignment) are what
localised it; `rocprofv3 --kernel-trace` was used to cross-check the host-side signature.  The
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION` A/B was a temporary instrument, reverted before landing.

## 2026-09-13 — re-based onto master `790cf51aa` (the 16-block set, 70 upstream commits)

The delivery moved from the 2026-09-08 fork point `9113cc188` to current master
**`790cf51aa`** ("chat : improve parsing of complex types in qwen3-coder (#28742)",
**70 commits** ahead).  The canonical 16-block chain was rebuilt at the new base
(tip **`43ec14228c60b0b8cb90205365c8e0aabec8bc7b`**, net tree
**`5cc664170a29cd78975f8679936d4d0adf28c605`**); `patches/` applies **strict 16/16
`git am`**, zero whitespace warnings, and the applied tree equals the canonical one.

### Upstream clashes resolved

* **`16378d93f` — CUDA/HIP: Flash Attention tuning (gfx1201)** rewrote the exact AMD-WMMA FA gate
  line block 04 owns (`Q->ne[0] <= 128` -> `<= 256`, plus a `> (ne0 <= 128 ? 8 : 16)` batch
  threshold), the `(256,256,32/64)` MMA config cases, upstream's AMD `switch_ncols2` preference
  (`gqa % 8/4/2` -> ncols1 8/4/2) and `should_use_stream_k` (stream-K on AMD only for `DKQ == 64`).
  **Head-to-head result (gfx1201, 27B head 256, single R9700, `llama-bench -r 3`, two `.so`
  variants measured interleaved):**

  | built-in `q8_0` | pp2048 | pp16384 | tg128 |
  |---|---|---|---|
  | PURE (ours) | 902.41 | 843.71 | 28.73 |
  | upstream FA | 902.02 | **847.95** | 28.75 |

  `f16`: PURE pp16384 843.7–845.0, upstream 851.0–851.5 (~+0.8 %); pp2048/tg flat; 4B Q8_0 within
  noise on both.  So upstream's tuning is worth only **~+0.5–0.9 % at `pp16384`**, flat elsewhere —
  but it **breaks the 4B decode/verify width purity** (`q4_0` `W=1 2b4c0165dc73567d` vs
  `W>=2 98e60bfd6e242b47`), while our block-04 configs return the whole band to **one hash,
  byte-identical to the (18) delivery** (`q4_0 bb6ae482f50502b3`).  Resolution: keep our
  **block-04 `(256,256,32/64)` configs** and drop upstream's AMD `switch_ncols2` block; **keep**
  upstream's `should_use_stream_k` (`DKQ == 64`) and its gate threshold (both are purity-neutral
  here and preserve upstream's stream-K preference).  Per the purity-first rule the sub-1 %
  long-prefill gain is not taken.

  > **Open finding:** the impurity is triggered by the *prefill* path yet appears as a `W=1` vs
  > `W>=2` **decode** difference for `q4_0` only, while the TILE decode is width-invariant by
  > construction.  The shipped build reproduces the validated (18) hashes exactly, so it is as pure
  > as the recorded delivery, but the underlying prefill-sensitive width sensitivity deserves a
  > proper upstream-quality repro rather than being considered fully explained.

  Full validation below.
* **`5a4d0feca` — CUDA: replace `GGML_FA_ALL_QUANTS` with `GGML_FA_QUANTS`** rewrote the FA-quant
  selection.  Block 08's 2026-09-11 enablement (`q4_1`/`q5_0`/`q5_1` and the 15 `iq4_nl` vec
  instances) is re-homed onto the new mechanism: `iq4_nl` is added to `FA_TYPES` in
  `ggml/cmake/common.cmake`, the default `GGML_CUDA_FA_QUANTS` becomes the **eight diagonals**
  (`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `iq4_nl`, `bf16`, `f16`), `ggml_cuda_get_fattn_vec_case()`
  gains the 15 `iq4_nl` pairs, and `ggml_cuda_fattn_kv_type_supported()` lists `iq4_nl`.
  Upstream's runtime fallback (uncompiled pair -> f16-f16 with a one-time warning) is kept, so an
  uncompiled type degrades instead of aborting.
* **`d4abd573f` — CUDA: size routed MoE MMQ N-tiles from typical expert width on RDNA3 (#28552)**
  added `int64_t ncols_opt` to `mmq_args` and switched `mul_mat_q_switch_J`'s `ntiles_x` to it.
  Merged additively with block 13's `x_gate`/`glu_op`/`glu_limit` fields and `J_max_gate` caps;
  `mmq_args args_gate = args` carries `ncols_opt` into the fused gate kernel.
* **`311d4211b` — memory: avoid allocating V cache for indexer (#28330)** sets
  `n_embd_head_k/v_mla_impl` to make the indexer cache look like MLA.  It composes additively with
  block 15 W3's `LLAMA_QSA_KEYS_ONLY` (`v_enabled=false`): both skip the dead V buffer, and W3 keeps
  its A/B kill-switch.
* **`b0dcb8192` (`common/speculative.cpp`)** renamed `common_speculative_draft_params::n_past` to
  `pos0`; block 01's added `n_cap` clamp now uses `dp.pos0`.
* **Upstream CMake refactors** (`LLAMA_CORE_SOURCES`, `llama_build`/`llama_build_and_test`) folded
  block 14's `llama-lazy-reader.cpp` / `test-lazy-reader.cpp` into the new structures; block 14's
  second `ggml_gated_delta_net` test call gained the `n_rs_batch` argument.

### Validation (gfx1201, ROCm 7.14, single R9700 unless noted)

* `test-backend-ops test`: **18061/18061 passed** (was 17999 at the old base; upstream's new
  head-256 `FLASH_ATTN_EXT` cases included).
* 4B `Qwen3.5-4B-Q8_0` width probe, W = 1..8, **all 8 KV types PURE**, hashes **byte-identical to
  the (18) delivery**: `f16 e3c53c3432c7815b`, `bf16 7254fecf4a9728df`, `q8_0 46a961911ca1fc12`,
  `q4_0 bb6ae482f50502b3`, `q4_1 32df01d9f1c4aef1`, `q5_0 b15ab98c50aa8f51`,
  `q5_1 bed6c581183172ce`, `iq4_nl b73b73f83ef30a12`.
* 27B `Qwen3.8-27B-UD-Q4_K_XL` width probe PURE, byte-identical: `q8_0 45313682f9d41816`,
  `f16 bf3348c0a49e461c`, `bf16 e3ad7b8a5ab74ed1`.
* 27B 8-KV-type text gate (`--spec-type none` == `draft-mtp n_max 3` == `n_max 7`): **PURE for all
  eight**, byte-identical to the (18) delivery (`f16/bf16/q8_0/q5_0 bf9a4fb7ddb5`,
  `q4_0 2c6003ae4688`, `q4_1 a46ef09ed137`, `q5_1 587344c9e92e`, `iq4_nl e031e49a4b16`).
* Rule-5 verify-width gate (27B `q8_0`, `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, TG total
  seconds, lower better): NEW **1.172 / 1.653 / 2.793** == OLD (18) **1.175 / 1.656 / 2.799**;
  stock `9113cc188` 1.155 / 1.683 / **2.881**.
* 27B server MTP (`runarm.sh` + `bench_mtp.py`, ctx 8192, `--spec-draft-p-min 0.55`, median of 5):
  NEW **40.29 / 36.37 t/s** (n3/n7) == OLD **40.21 / 36.34**; stock 37.80 / 33.90.  Acceptances
  match the (18) delivery exactly (n3 0.6111, n7 0.4746).
* MoE `Qwen3.6-35B-A3B-UD-Q4_K_M` (f16 KV, ctx 8192, median of 3): NEW plain **93.05**, n3
  **133.38** (acc 0.5430), n7 **92.30** (acc 0.2636) == OLD **92.64 / 133.20 / 92.41**.
* `qwen4exp` (Qwen3.8-Flash-Next-UD-Q4_K_XL, 4 shards) on **3 GPUs `-sm tensor`** with the
  dedicated MTP drafter (`/models/.../mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`), ctx 8192, f16 KV:
  `plain == draft-mtp n_max 3 == n_max 7` **PURE and byte-identical to the (18) delivery**
  (`87a30d8cef7a`).  The QSA-derived kq-mask probe warning ("was not used in the probe graph") is
  expected; the 8-native-KV-type qwen4exp matrix was validated on other hardware.

### Housekeeping

`scripts/make-patches.sh` default baseline/tip and `scripts/apply-all.sh`'s baseline comment now
name `790cf51aa` / `43ec14228…`; block 13's hand-carried `--- 2026-09-12 amendment ---` note was
re-inserted into `patches/0013` after regeneration (git drops `--- `-prefixed body lines).  No
delivery patch content changed apart from the rebase resolutions above.

## 2026-09-13 — CI fix: GHCR container workflow's stale commit-count assertion

The `.github/workflows/docker-ghcr.yml` apply step hard-asserted
`test "$(git rev-list --count HEAD)" -eq 16` ("base commit + 15 block commits") and
`git log --oneline -15`.  It was authored 2026-09-10, when the set was 15 blocks
(block 00 + blocks 01-14 = 16 commits); **block 15 was promoted 2026-09-12**, making
the set 16 blocks and the applied history **17 commits**.  The patches still applied
strict 16/16 `git am` and `apply-all.sh` reported success, so the sanity check — not
the delivery — aborted every matrix job (all three ROCm entries) with exit code 1
before any image was built (runs 34733460496, 34733771241).

Fix: derive the expected count from the patch set instead of hardcoding it, so it
cannot drift on the next block:

```sh
n_blocks="$(find "$GITHUB_WORKSPACE/patches" -maxdepth 1 -name '[0-9][0-9][0-9][0-9]-*.patch' | wc -l)"
test "$(git rev-list --count HEAD)" -eq "$((n_blocks + 1))"
git log --oneline -"$n_blocks"
```

The header comment now also says `patches/0000..0015 (block 00 + blocks 01-15)`.
Reproduced the exact CI step locally with a fresh `9113cc188` tarball + `apply-all.sh`:
16 patches -> 17 commits, dynamic check passes (the old `-eq 16` fails as observed).
The three base image tags (`rocm/dev-ubuntu-24.04:{7.2.4-complete,7.14.1-full,10.0.0-full}`)
were confirmed to exist, so the pipeline can proceed past the fixed step.  No delivery
patch content changed.

## 2026-09-12 (18) — block-13 amendment: dense mmvq weight per-(type, K) nwarps (targeted MoE recovery)

Follow-up to (16)/(17).  The (16) band-uniform RDNA4 `nwarps = 1` fixed the dense verify widths but
cost the MoE model's **dense** Q8_0 layers ~9 % at single-token decode (diagnosed in (17); the MoE
expert kernel does not use `calc_nwarps`, so the loss is the dense MUL_MATs).  The dense mmvq
**weight** kernel `mul_mat_vec_q_ksplit` now picks `nwarps` **per `(type, K)`** via a new
`calc_nwarps_weight()`: RDNA4 + `Q8_0` + `K < 4096` -> the pre-2026-09-11 wide block (`nwarps = 8`),
every other shape -> 1.  `K = ncols_x >= 4096` is computed host-side and threaded as a compile-time
`long_k` template bool (launch bounds and the shared-memory-sized reduction stay compile-time).
Crucially the **pinned fusion ops keep plain `calc_nwarps`** (their `calc_nwarps(GGML_TYPE_Q8_0, 1,
...)` call is a single-token reduction-order anchor) — leaking the rule into them made the 27B `f16`
KV width probe impure at W=1; scoping it fixed that.

**Measured** (1 GPU; MoE 35B-A3B f16 `-ntg 64`, dense 27B UD-Q4_K_XL q8_0):

| metric | (17) `nwarps=1` | (18) per-(type,K) |
|---|---|---|
| dense MTP `n_max 7` | 35.75 | 35.65 (~0, 27B bit-identical) |
| dense B=8 | 2.865 | 2.875 (~0) |
| MoE B=1 plain | 0.7835 | **0.752 (+4.0 %)** |
| MoE MTP `n_max 3` | 161.0 | **164.2 (+2.0 %)** |
| MoE MTP `n_max 7` | 159.5 (acc 0.63115) | **177.1 (+10.0 %, acc 0.73148)** |
| MoE B=8 batched TG | 1.4445 | 1.485 (−2.8 %) |

The 27B is bit-identical (Q8_0 K >= 5120 -> `long_k` -> 1), so the dense issue-#30 fix is untouched;
the MoE batched B=8 cost is the deliberate trade and still beats the pre-(16) 1.494.

**The opposite assignment was measured and rejected**: giving the dense kernel the MoE expert
kernel's wide VDR=4 on the same short-K Q8_0 shapes buys 1.6 % on the batched B=8 but cancels the
MTP gain (`n_max 7` 177.1 -> 161.2, acceptance back to 0.63115).  The two knobs have independent
per-kernel optima — dense wants wide `nwarps` + narrow `VDR`, the MoE expert kernel wants wide `VDR`.

**Validation** (clean-apply build): 4B all-8-KV-type width probe **PURE** with re-pinned hashes
(`f16` `e3c53c3432c7815b`, `bf16` `7254fecf4a9728df`, `q8_0` `46a961911ca1fc12`, `q4_0`
`bb6ae482f50502b3`, `q4_1` `32df01d9f1c4aef1`, `q5_0` `b15ab98c50aa8f51`, `q5_1` `bed6c581183172ce`,
`iq4_nl` `b73b73f83ef30a12`); 27B `q8_0`/`f16`/`bf16` probe **bit-identical** to (17); 27B 8-KV-type
text gate **bit-identical** to (17); MoE plain `W=1..8` PURE; `test-backend-ops` ROCm0
**17999/17999**.

**Clean-apply**: canonical rebuild at `9113cc188` + the regenerated 16-patch set, strict **16/16**
`git am`, zero whitespace warnings, applied tree **`c2e284c2acc032238ef85cb35d427c1598ed0949`**
(rebuilt canonical tip `907799de3e6a7dcbd206d03b2daef4c248144ca9`).  Block 0013 is the only content
change vs the (17) regeneration; block 13's hand-carried RDNA3_5 note is preserved.

## 2026-09-12 (17) — block-10 amendment: the mmvq VDR is per kernel (dense upstream, MoE expert block-10)

Follow-up to (16).  The (16) revert of block 10's `VDR=4` was **global**; a code read shows the
**MoE expert kernel** `mul_mat_vec_q_moe` (the `MUL_MAT_ID` path) is *not* reached by the block-08
`nwarps` change (it launches `(warp_size, ncols_dst)` — one warp per token — and never calls
`calc_nwarps`) but *is* reached by `get_vdr_mmvq`/`get_vec_dot_q_cuda`, so the global revert took the
wide chunk from the one kernel it was tuned for.  The VDR is now selected **per kernel**: the dense
`mul_mat_vec_q` (item-split), `_ksplit` and their fused variants keep the upstream VDR
(Q4_K/Q5_K/Q6_K 2/2/1, Q8_0 2, `moe = false` default); `mul_mat_vec_q_moe` takes block 10's values
through `get_vec_dot_q_cuda(type, true)` / `get_vdr_mmvq(type, true)` — Q4_K/Q5_K `..._vdr4`, Q6_K
`..._vdr2`, Q8_0 `vec_dot_q8_0_q8_1_moe` (`VDR_Q8_0_Q8_1_MMVQ_MOE`, 4 on RDNA4/RDNA3_0, else 2).
`vecdotq.cuh` returns to the block with the `_vdr4`/`_vdr2` functions **only** — the dense macros
stay upstream, so `hc-mix.cu` and every dense reference hash are unchanged.  Both kernels stay
band-uniform internally (the VDR is a compile-time per-type constant).

**Measured** (1 GPU gfx1201; MoE 35B-A3B Q4_K_M f16 `-ntg 64`, dense 27B UD-Q4_K_XL q8_0 `-ntg 32`;
`llama-batched-bench` TG total seconds, MoE MTP `draft-mtp n_max 3`):

| build | dense B=1 | dense B=8 | MoE B=1 | MoE B=8 | MoE MTP n3 |
|---|---|---|---|---|---|
| pre-(16) | 1.149 | 3.915 | 0.713 | 1.494 | 166.6 t/s |
| (16) amended | 1.175 | 2.798 | 0.782 | 1.506 | 160.2 t/s |
| **(17) per-kernel VDR** | 1.174 | **2.795** | 0.783 | **1.452** | 161.2 t/s |

So (17) keeps the dense fix and recovers the **VDR-caused part** of the MoE loss (B=8 1.506 -> 1.452,
better than pre-(16)).

**Correction to the (16) attribution**: the *larger* MoE single-token/MTP loss is the band-uniform
`nwarps = 1` on the dense layers, **not** the VDR.  A diagnostic build restoring the pre-(16)
per-type `nwarps = 8` (RDNA4) while keeping the per-kernel VDR recovers MoE B=1 to **0.716 s** and
MoE MTP to **167.4 t/s** — but costs dense MTP (35.9 -> 34.3 t/s at `n_max 7`) and MoE B=8
(1.452 -> 1.499).  The 27B's Q8_0 decode path is hit too, so no per-type split satisfies both models
(the same Q8_0 type serves the MoE attention and the 27B decode).  `nwarps = 1` is kept — it is what
the (16) dense verify fix requires — and the residual MoE single-token/MTP delta is a **documented
trade**, not a fixed regression.

**Validation** (clean-apply `deliver-verify` build): the 4B all-8-native-KV-type width probe and the
27B `q8_0`/`f16`/`bf16` probe reproduce every (16) hash; the 27B 8-KV-type text gate
(`plain == mtp3 == mtp7`) reproduces every (16) hash; dense MTP `n_max 7` 35.9 t/s / `n_max 3`
46.7 t/s; MoE MTP `n_max 3` 161.2 t/s / acceptance 0.87179; `test-backend-ops` ROCm0
**17999/17999** passed, 0 FAIL.

**Clean-apply**: canonical rebuild at `9113cc188` + the regenerated 16-patch set, strict **16/16**
`git am`, zero whitespace warnings, applied tree **`2833f1369bdea4cb45f68f85dbb2898fd98aab66`**
(rebuilt canonical tip `a05225f7361ea5a1116d7185ebec8867cfe4afe2`).  Block 0010 is the only content
change vs the (16) regeneration; block 13's hand-carried 2026-09-12 RDNA3_5 note is preserved.

## 2026-09-12 (16) — block-08 + block-10 amendment: the MTP decode regression (issue #30)

Issue **#30** (briansp2020, single R9700 gfx1201, dense **Qwen3.8-27B UD-Q4_K_XL**, `q8_0` KV)
reported the delivery ~14 % **slower** on MTP decode than stock `9113cc188` at the same fork point,
with much faster prefill.  Reproduced on the maintainer rig (27B UD-Q4_K_XL, `q8_0` KV, 1 GPU): stock
`--spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-p-min 0.55` **37.51 t/s** vs the delivery
**30.34 t/s**; plain decode was fine (delivery slightly faster).  Root cause, via
`llama-batched-bench` (no speculation, so no acceptance confound) — the **multi-token verify path**
was up to **+35 %** slower at B=8, and the penalty grew with width from B=3 on.  Two band-uniform
mmvq knobs (made uniform by the 2026-09-11 MTP purity work, but left at their **single-token-tuned
values**):

* **block 10** — the `VDR=4` mmvq boost for Q4_K/Q5_K/Q6_K (the 32-element-per-call variants lose on
the verify widths).  **Reverted in full** (`vecdotq.cuh` back to the upstream VDR set — Q4_K/Q5_K/Q6_K
2/2/1, Q8_0 2); `vecdotq.cuh` drops out of the block.
* **block 08** — the RDNA4 `calc_nwarps` per-type whitelist (`nwarps=8` for the simple-vec_dot types,
only ever tuned at `ncols_dst == 1`).  The RDNA4 band is now **band-uniform `nwarps=1`**; RDNA3_0 and
RDNA3_5 tables unchanged.

**Result** (27B UD-Q4_K_XL, `q8_0` KV, 1 GPU; `llama-batched-bench` TG total for 32 steps):

| build | plain | B=1 | B=4 | B=8 | MTP n_max 7 | acc | MTP n_max 3 | acc | MTP-adaptive n_max 7 | acc |
|---|---|---|---|---|---|---|---|---|---|---|
| stock `9113cc188` | 28.25 | 1.157 | 1.726 | 2.929 | 37.51 | 0.484 | — | — | — (no adaptive) | — |
| delivery (pre-amendment) | 29.34 | 1.147 | 2.121 | 3.958 | 30.34 | 0.466 | — | — | 30.57 | 0.4201 |
| **amended** | 28.62 | 1.175 | 1.657 | **2.798** | **36.32** | 0.475 | **40.15** | 0.611 | **38.47** | 0.4226 |

So the amended **verify path is faster than stock's** and the residual MTP difference is the
single-token `nwarps=8` (B=1 1.175 vs 1.157) that the band-uniform purity constraint forbids, plus the
`n_max 7` clamp.  The `nwarps` sweep (band-uniform 1/2/4/8 → B=8 2.790/2.894/3.162/3.307, MTP
36.59/36.00/33.91/32.78) picks **1**; single-token is flat (1.160–1.175) and per-type mixing never
helped.

**Validation** (amended clean-apply build, `deliver-verify`):
* **width probe** (`logits-dump-kv`, W=1..8): 4B **all 8 native KV types** PURE; 27B `q8_0`/`f16`/`bf16`
  PURE.
* **text gate** (`--spec-type none` == `draft-mtp n_max 3` == `n_max 7`), 27B, **all 8 native KV
  types** byte-identical.
* **MoE MTP gate** (35B-A3B Q4_K_M, 1 GPU, f16 KV): plain 84.3 → `draft-mtp n_max 3` **160.2 t/s**,
  acceptance **0.87179** (unchanged).
* **`test-backend-ops`**: ROCm0 **17999/17999** passed, 0 FAIL.
* Same-seed coherence: coherent; the 4B reference re-baselines `f069f69475e7` → `3eeb3d9d333e`
  (deliberate reduction-order change).

**Clean-apply**: canonical rebuild at `9113cc188` + the regenerated set, strict **16/16** `git am`, zero
whitespace warnings, applied tree **`56a1c5f23c54c038f78d7242dc05b181d872b69b`** (canonical tip for
this rebuild `1837856e3f8120449090c0f44594427573a541ed`).  Only blocks 0008 and 0010 change content;
block 13's hand-carried 2026-09-12 RDNA3_5 amendment paragraph is re-added to the patch body (it is
dropped by `git am` scissors handling).

**Gate gap closed**: `../benchmarks/mtp-adaptive-methodology.md` gains a stock-relative
verify-width `llama-batched-bench` check — the existing gate only tested acceptance at the default
depth 3 and `llama-bench tg128` (the one width that never regressed).

## 2026-09-12 (15) — block 15 promoted to the delivery (TODO item 1 closed)

The attention-memory campaign (block 15) was **promoted from `archive/work/block-15-campaign-wins/` to the
delivery**.  The beta window closed with the maintainer's go-ahead; the beta patch is now
`patches/0015-rdna-boosts-block-15-campaign-memory-wins.patch`, so the delivery is a **16-patch set**
(block 00 + blocks 01-15) and `scripts/apply-all.sh` / `scripts/make-patches.sh` are 16-block flows
(the old "beta patch applied manually on top of the 15-block tree" flow is gone).

* **Clean-apply**: a canonical fork rebuilt at `9113cc188` from the current `patches/`
  (`scripts/apply-all.sh`, strictly) produced tree `3b0874b6aa367fea846a437b45f1689bd173b38c`; the
  promoted block-15 patch applied on top with strict `git am` (no `-3`), producing the re-validated
  beta tree **`c3142fe0b311757f458647f172f623859f5bc983`** and canonical 16-block tip
  **`0f4f83f9ef01ffd1662f58d714d62b9155325a62`**.  A fresh worktree at `9113cc188` + the updated
  `apply-all.sh` then applied strict **16/16** `git am`, zero whitespace warnings, applied tree ==
  `c3142fe0b3`.
* **Patch identity**: `patches/0015` is byte-identical to the beta patch except its `From <sha>` line;
  blocks `0000`-`0014` were regenerated from the canonical rebuild and are byte-identical to the
  previous delivery apart from the `From` lines and the `[PATCH NN/14]` -> `[PATCH NN/15]` series
  denominator (block 13's hand-carried 2026-09-12 RDNA3_5 amendment paragraph is preserved verifiably
  — it is dropped by `git am`'s scissors handling, so it is re-added to the patch body as before).
* **`rdna-boosts-all.patch`** regenerated as `git diff 9113cc188..0f4f83f9e` (127 files).
* **The seven wins and their gates are unchanged** (W4 has no gate; V4/V5 share the opt-in
  `GGML_CUDA_FA_KV_NATIVE`, default 0).  The revalidation that the promotion rests on reproduced every
  reserve number to the last decimal, the width-probe reference hashes (1 GPU `4089b4d4`, 2-GPU tensor
  `a4817ee6`, 3-GPU tensor `91434ea9`; `W=9` divergent as accepted), byte-identical same-seed coherence
  across gates on 4B / gemma-4-E4B (ISWA) / gemma-4-31B (ISWA) / 27B (short + 40k) / qwen4exp, the op
  suites (`FLASH_ATTN_EXT` 7859/7859 ROCm0 + CPU, `GATED_DELTA_NET` 46/46, `FLASH_ATTN_QSA` 22/22),
  the unchanged MTP gate (27B `0.76744`, qwen4exp `0.44262`), and the W4 round trip 56.00 -> 16.00 MiB.
  The accepted W2-`iq4_nl` ULP caveat is recorded in `archive/work/block-15-campaign-wins/BETA-TESTING.md` §4d.
* **Docs**: `patches/README.md` (the 0015 row + the promotion section), `README.md`, `MANIFESTS.md`,
  `BASELINE.md`, `AGENTS.md`, `TODO.md` (item 1 moved to Closed) and the beta README (marked
  **PROMOTED**) all moved to the 16-patch state.  The block-15 gate table, the per-win mechanism notes
  and the gfx1151 pass stay in `archive/work/block-15-campaign-wins/README.md`;
  `archive/work/strix-halo/GATE-2026-09-10-block15-rdna35.md` is the gfx1151 record.

## 2026-09-12 (14) — gfx1151 cross-check of the block-14 (eighth) fix: TODO item 4 fully closed

TODO item 17 (the gfx1151 cross-check) is resolved and item 4 is fully closed.  Validated on gfx1151
(Strix Halo, ROCm 7.14 at `/opt/rocm-7.14-gfx1151`) against branch `block14-band-uniformity`: fresh
worktree at `9113cc188` + `scripts/apply-all.sh` -> strict **15/15** `git am`, 0 whitespace warnings,
applied tree **`3b0874b6aa367fea846a437b45f1689bd173b38c`** (== canonical).

* **The forced-sparse text residual is gone.**  `LLAMA_QSA_DENSE_DECODE_UNTIL=0` + q8_0 + `p5000.txt`
  (seed 42, temp 0, n 128, `-sm layer`): pre-fix (the amendment-7 build) `plain a57bc13bbf2a` vs n3
  `3124adfd2b94` (first diff **char 458**); post-fix `plain == n3 == a57bc13bbf2a` (632 chars).  All
  eight native KV types are pure in the forced-sparse regime (f16 `cb2912b186b9`, bf16 `945f89766e3c`,
  q8_0 `a57bc13bbf2a`, q4_0 `9afd1d55a5ae`, q4_1 `aff1978cf720`, q5_0 `296f8ebcd246`, q5_1
  `a88803f4ebf9`, iq4_nl `8e4437794660`); pre-fix only q8_0 and q5_0 were impure.  **The full
  n_max 1/2/3/5/7 sweep is pure for all eight native KV types in *both* the forced-sparse and the
  default (dense) regimes.**  Default (dense) gates unchanged: q8_0 `e8f8bba3942b` (626), f16
  `0fc4910d5824` (632).
* **The `mstep` matrix is 0 mismatches at every width**: `W = 1,2,3,4,5,8` (forced-sparse q8_0) all PURE
  with a stable `Thash = ea713a1c1f515bc1`, **unchanged vs the pre-fix build**.  (gfx1151's mstep was
  already pure at default params pre-fix, unlike gfx1201's W=2/W>=3 boundary, so the text gate is the
  discriminator on this arch.)
* **Op suites**: `FLASH_ATTN_QSA` **22/22**, `GATED_DELTA_NET` **46/46**, `FLASH_ATTN_EXT` **5935/5935**.
* **MTP acceptance (Protocol A, n_max 3)**: forced-sparse q8_0 `0.51333` (77/150), pos-1
  `(0.740, 0.420, 0.380)`; default q8_0 `0.57554` (80/139) and default f16 `0.51678` (77/149) =
  bit-identical to the pre-fix values.
* **Beta**: the 17th block-15 re-cut applies cleanly on the new tree (`git am -3` -> tree
  `c3142fe0b311757f458647f172f623859f5bc983`, the recorded beta tree).
* **Outcome**: TODO item 4(b) dropped from *Documented* and item 17 closed; the delivery branch merges
  into `main` with no code change beyond the (eighth) amendment already in `patches/`.  Records: this
  entry, `TODO.md`, `patches/README.md` (the (eighth) section), `GREEDY-PURITY.md` §29, the harness
  `archive/work/strix-halo/qsa-item4/`.

## 2026-09-12 (13) — block-14 amendment (eighth): the QSA indexer-score decode/verify band-uniformity fix

Block-14 amendment (eighth), found by the gfx1201 investigation of TODO item 4's q8_0 forced-sparse
residual.  Canonical tip `c6f1e8e78` -> **`d306d4b4b`** (tree `e1e42e23c` ->
**`3b0874b6aa367fea846a437b45f1689bd173b38c`**); block 14 amended in place (the tip block, so no
replay), `make-patches.sh` default tip updated, `rdna-boosts-all.patch` regenerated; strict **15/15**
`git am` on a fresh worktree at `9113cc188` (0 whitespace warnings, applied tree == canonical).

* **The defect**: the QSA indexer score's matmul carries the indexer heads in its N dimension, so its
  `ne11` is `n_idx_h * n_tps` (**4 * n_tps** for qwen4exp).  Block 08's "keep the verify batch on the
  decode kernel" guard (`ne11_mmvf = ne11 <= MMVF_MAX_BATCH_SIZE ? 1 : ne11`) assumed `ne11` *is* the
  token count, so from `n_tps = 3` the guard stopped rescuing the verify batch: decode (`n_tps = 1`)
  stayed on the MMVF family while the verify fell through to MMF, and the two families accumulate the
  truncated dot product differently.  The indexer score then differed by a ULP and flipped a top-k
  near-tie - the forward was **bit-identical to decode for 101 steps and then diverged** at target
  position 4395.  On the `p5000` prompt the greedy **text** happened to stay equal, so it was a
  logits-level `plain != draft-mtp` violation, not a visible text change.
* **How it was found** (gfx1201, 3x R9700): a `mstep` width matrix showed W=2 pure, W>=3 impure with the
  first divergence at a fixed position (4395, not the first batch), i.e. a selection flip rather than
  drift; `LLAMA_QSA_SPARSE_FA=0` / `LLAMA_QSA_OFF=1` were pure and block-15's gates irrelevant.  A
  `rocprofv3 --kernel-trace` diff of W=2 vs W=3 showed the only exclusive kernels were ncols-templated
  MMVF/ksplit variants, with the score moving from `mul_mat_vec_f<float,float,8,64>` (N=8) to no MMVF
  instantiation at W=3.  Forcing the fallback family for *every* F32 matmul (a temporary diagnostic)
  made the band pure again - confirming "one family across the band" as the fix.
* **The fix**: `MMVF_MAX_BATCH_SIZE_FLAT` (`= MMVF_MAX_BATCH_SIZE * 4 = 32`) in `mmvf.cuh`; the block-08
  guard widened to it in `ggml-cuda.cu`; `mul_mat_vec_f_cuda_switch_ncols_dst` instantiates
  `ncols_dst` 9..32 in `mmvf.cu` (+168 lines).  The guard stays at the decode family (MMVF) so the
  verified arithmetic is the one the draft's single-token decode reproduces.  The guard is block 08's;
  block 14 extends it because block 14 is the block that introduces the flattened batch.
* **Validation (canonical delivery tree, gfx1201, 3x R9700, layer split)**: `mstep` W = **1,2,3,4,5,8**
  q8_0 all **0 mismatches** (pre-fix W>=3 impure), and the W=1 reference `Thash` is unchanged
  (`2bd73063dd0a9524`) so **decode numerics are untouched**; f16 W=4 pure; forced-sparse q8_0 and
  default text gates byte-identical (`a4cdc10dfb6c` 678 chars / `2e078b6966c0` 682 chars);
  `FLASH_ATTN_QSA`, `GATED_DELTA_NET` and `FLASH_ATTN_EXT` all OK (4/4 backends); 27B dense
  `plain == n_max 3` (`da2e2d192e21`); MTP acceptance healthy (forced-sparse q8_0 `0.46497`, pos-1
  `(0.698, 0.415, 0.264)`; default `0.44444`, pos-1 `(0.673, 0.418, 0.218)`).
* **No delivery behaviour change outside the flattened band**: for `ne11 <= 8` (decode/verify of every
  ordinary op) and `ne11 > 32` (prefill) the guard decision is unchanged, so dense models are
  unaffected by construction (verified: 27B `plain == draft-mtp`).
* **Open cross-check** (confirmed 2026-09-12 (14): the gfx1151 forced-sparse text residual is gone and
  all eight native KV types are pure — see the (14) entry): whether this also removes the gfx1151
  `plain != draft-mtp` text residual that
  `TODO.md` *Documented* records (same signature, different arch - gfx1151's `mstep` was reported pure)
  is to be confirmed by the gfx1151 box against this branch.  The fix is arch-independent in the engine
  (per-arch MMVF tables aside), so the branch is the test vehicle.
* Records: this entry, `GREEDY-PURITY.md` §29, `patches/README.md` (the block-14 amendments list),
  `archive/work/strix-halo/qsa-item4/` (the `mstep` harness).

## 2026-09-12 (12) — TODO item 4 closed: the block-14 MTP-export logits-purity fix + the q8_0 forced-sparse residual recorded as a limitation

TODO item 4 is closed.  It had two sub-items; **(a)** is fixed and landed as a block-14
amendment (seventh), **(b)** survives a genuine driver-level investigation and is recorded as a
measured, deliberately-NOT-fixed limitation.  Canonical tip `47a9d4d86` ->
**`c6f1e8e78cfb2a70958998cdd81fad363e869f93`** (tree `c24871386c479865d41476726cf1f01c43b23ea6` ->
**`e1e42e23c2913cd529b0064eb1cb74525a746098`**); block 14 amended in place (the tip block, so no
replay), `make-patches.sh` default tip updated; strict **15/15** `git am` on a fresh worktree at
`9113cc188` (0 whitespace warnings, applied tree == canonical); `rdna-boosts-all.patch`
regenerated; beta block-15 **re-cut 16th** (`bdd09891d588225e139a67e510094d972acd1858`, tree
`3a47913c0bdca7f1154a8f0310a20435a36c0faa`, patch 206 454 bytes, round-tripped strict `git am`).

* **(a) `embeddings_nextn` broke logits-level `plain == draft-mtp` on qwen4exp.**  The unmasked MTP
export needs a hidden row for every prefill token, so the last layer's output-row gather was
deferred; the last layer's ffn tail then ran on the full ubatch and the prefill's last-position
logits shifted by a ULP (`ad3acaa75d19ddf2` vs `b624a79f19b1b1f0`).  The last layer now always
gathers the output rows before its tail (exactly the plain path) and builds a **second, full-row
tail** solely for `t_h_nextn` when the chunk drops rows (`n_outputs < n_tokens`); a decode/verify
batch drops none, so nothing is duplicated there.  **Verified (gfx1151):** the `mstep` `NEXTN=1`
prefill mismatch is gone (`0` mismatches; was `1` at `pos = 4293`); the `W=4 RB=3 RS=3 JUNK=1`
width probe is `0` mismatches and the `W=8` 38-mismatch position list is byte-identical pre/post
(all 38 positions); default and forced-sparse text gates byte-identical (`e8f8bba3942b` /
`0fc4910d5824`); MTP acceptance bit-identical (f16 `0.51678` = 77/149 both builds); the graph is a
no-op on every non-NEXTN path (the gather already ran with `gather_now == true`).  Instrument:
`archive/work/strix-halo/qsa-item4/`.
* **(b) the forced-sparse shallow q8_0 residual is recorded, not fixed.**  Repro:
`LLAMA_QSA_DENSE_DECODE_UNTIL=0` + `-ctk/-ctv q8_0` + `p5000.txt` + `draft-mtp n3` on qwen4exp ->
`plain a57bc13bbf2a` vs `n3 3124adfd2b94` (632/657 chars).  A logits-level first divergence was
localised with a temporary target-logits dump in the real `server-context.cpp` driver: at target
position **4432** the accepted token is identical (381) but the target logits argmax flips
**264 -> 9859** - a QSA-indexer *selection/state* divergence, not a forward width dependence (the
`mstep` replay is bit-pure).  Excluded on the current tip: forward width (`mstep` W=1..8 pure), the
GDN rollback bound and checkpoint restore (`n_rs_seq = 16` forced - still diverges;
`test-recurrent-state-rollback` PASS), `n_outputs_max`, CUDA-graph capture, the chunked-prefill
boundary, the fused indexer score and the derived cache (both bypassed with quantized keys), and the
sparse FA kernel (`LLAMA_QSA_SPARSE_FA=0` does not fix it - the shared dense masked path is
affected).  `LLAMA_QSA_OFF=1` fixes it and `GGML_CUDA_GDN_CHUNKED=0` only perturbs the trajectory to
purity; the delivery default (dense decode below 64K) is pure, so this is a forced-arm,
prompt-dependent, q8_0-only low-severity limitation.  Record:
`archive/work/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md` + `GREEDY-PURITY.md` §18/§28.
* **Gates (gfx1151, current tip):** `FLASH_ATTN_QSA` 22/22, `GATED_DELTA_NET` 46/46, `FLASH_ATTN_EXT`
5935/5935; dense-masked oracle (Sherlock corpus, 4x4096, f16) sparse `1.0539` vs dense `1.0544`;
band purity default q8_0/f16 pure and forced-sparse f16 pure; `draft-mtp n_max 3/5/7` acceptance /
text unchanged; beta re-cut revalidated (`GATED_DELTA_NET` 46/46, `FLASH_ATTN_QSA` 22/22,
`test-recurrent-state-rollback` PASS, the four gate combos + `draft-mtp n_max 3` all
`0fc4910d5824`).
* **No new Active item**; item 4 is removed from Active with a Closed one-liner, and the residual is
one entry in *Documented, deliberately NOT fixed*.

## 2026-09-12 (10) — block-02 amendment: the chunked-GDN snapshot bound (`n_rs_batch`) + the pre-batch slot

Integrated from the gfx1201 investigation in `~/ngram-mod/` (record: `archive/work/gdn-rs-rollback/README.md`;
originals `~/ngram-mod/{README.md,fix-ngram-mod.md,gdn-rs-rollback-bound.patch}`).  Canonical tip
`890a9c5b1` -> **`47a9d4d86`** (tree `0edf654cdea653b9969f866977a541ee4429f846` ->
**`c24871386c479865d41476726cf1f01c43b23ea6`**); block 02 amended in place and blocks 03-14 replayed with
**no conflicts** (the net delta is byte-exactly the patch: 20 files, +96/-23), and patch bodies
`0003`-`0014` changed **only in their `From`/`index` lines plus hunk offsets** (verified: all 52 changed
lines in `0014` are hunk headers).  `make-patches.sh` default tip updated; strict 15/15 `git am` on a
fresh worktree at `9113cc188` (0 whitespace warnings, applied tree == canonical); beta block-15
**re-cut 15th** (`eb15f3ee1`, tree `ffa3a11c30ba6d42dea2520f402126370df3bbb6`, patch 3 819 lines,
round-tripped, cherry-pick clean).

* **The defect**: the whole-batch chunked GDN path wrote no rollback snapshots for batches above its
  threshold, on the assumption that such a batch is "not a verify batch".  `n_rs_seq` comes from
  `speculative.draft.n_max` (7) but `--spec-ngram-mod-n-max` can draft 64, so a 65-token verify batch
  took the chunked path and a small tail rollback restored an unwritten plane - a silent
  recurrent-state rewind.  The block-02 `seq_rm` guard (2026-09-11) is the detector; the reported
  warning is real.
* **The fix**: `n_rs_batch` (longest draft any enabled speculator can produce + 1, from
  `common_speculative_n_max()`) is threaded `llama_context_params` -> `llama_cparams` ->
  `ggml_gated_delta_net()` op param 1 -> the CUDA dispatch, where the threshold becomes
  `max(K > 16 ? K : 16, n_rs_batch)`; plus the pre-batch ssm/conv state is written into slot
  `n_tokens` when `0 < n_tokens < K`, so a whole-batch rollback has the state it needs.  No snapshot
  memory change (sizing `n_rs_seq = 64` would have cost ~+8 GiB).
* **Validation (gfx1151)**: in-tree `test-recurrent-state-rollback` **FAIL -> PASS** (unpatched:
  `multi-seq split replay logits mismatch (max diff 6.5366, first at seq 0 pos 16)`; patched:
  `matched (max diff 0)` for both cache fills + the seq-1-only case); `GATED_DELTA_NET` **46/46**;
  neutrality: 27B `plain == draft-mtp n_max 7` = `e164f09af338` and qwen4exp `plain` = `0fc4910d5824`
  identical before/after, 27B pp2048/8192 within noise; beta re-cut revalidated (`GATED_DELTA_NET`
  46/46, `FLASH_ATTN_QSA` 22/22, rollback test PASS, all four gate combos + `draft-mtp n_max 3`
  byte-identical `0fc4910d5824`).
* Trade recorded: batches in `(max(K,16), n_rs_batch]` now run the sequential kernel (correctness
  requires it - the chunked kernel cannot write those snapshots).  Delivery configs are unaffected
  because their `n_rs_batch <= 16`.

## 2026-09-12 (9) — TODO item 9 resolved and closed: the configurable QSA prefill arm + the device-query arm gate

Block-14 amendment (sixth).  Canonical tip `13af95ac1` -> **`890a9c5b1`** (tree
`f4791066f4a582316b1ca95f51c96cd10b905ef7` -> **`0edf654cdea653b9969f866977a541ee4429f846`**);
`make-patches.sh` default tip updated; strict 15/15 `git am` re-verified on a fresh worktree at
`9113cc188` (0 whitespace warnings, applied tree == canonical); beta block-15 **re-cut 14th** on the
new base (`86c7df1f5`, tree `66f0762a2ec19cbc34b1842d1b5984bb82ecec45`, patch 3 819 lines,
round-tripped).  Full record: `archive/work/strix-halo/qsa-item9/RECORD-2026-09-12-qsa-prefill-crossover.md`.

* **9(a) the prefill arm is now configurable, and its default is the documented policy: `0` = QSA
  prefill always.**  Prefill previously had no depth axis at all (only the decode crossover
  `qsa_dense_decode_until`), so `qsa_dense_prefill_until` (env `LLAMA_QSA_DENSE_PREFILL_UNTIL`,
  `K/M/G`, `0` disables the arm) is a genuine addition; a prefill ubatch whose `n_kv` is still below
  the threshold attends dense while storing the indexer keys, so the sparse path takes over above it.
  The default is `0` on every arch and split because that is the ARCH POLICY -- `beta/qwen4exp/README.md`
  ("decode uses the dense attend below a per-arch depth and QSA above; **prefill is always QSA**") and
  the 2026-09-07 crossover record ("**Soar: QSA for prefill ALWAYS** (wins from ~8K, monotonically to
  +181 % @160K); dense for decode ALWAYS"; Halo from ~16K).  **The delivery's default behaviour is
  therefore byte-identical to the pre-amendment build** (f16 `0fc4910d5824`, q8_0 `e8f8bba3942b` = the
  recorded pre-amendment shallow values; `plain == draft-mtp n_max 3 == n_max 7`), so no reference hash
  moves, and the arm ships as an opt-in A/B.
  *Correction recorded on purpose:* this session's first pass set a default (gfx1151 8192, tensor split
  16384) from a **whole-prompt** `llama-bench` A/B plus a parenthetical in `patches/README.md`, and the
  maintainer corrected it -- the gfx1201 decision is QSA prefill always, dense never better.  The
  2026-09-07 record already carried the reason that A/B cannot decide a default: its tables are
  `pp2048` measured *at depth*, and it explicitly rejects the shape ("the old \"dense wins prefill at
  30K\" record is obsolete ... also a non-comparable whole-prompt llama-cli banner").  The A/B numbers
  are kept in the record as a description of what the knob does, flagged non-comparable, and the
  default is the policy.
* **9(b) the arm gate asks the device instead of mirroring the kernel's type list.**  `qsa_kv_native`
  was a hand-maintained copy of `ggml_cuda_flash_attn_qsa_supported()` (kept in lockstep by comment)
  and its staleness is what made the 2026-09-11 third amendment an abort in the meta splitter instead
  of a fallback.  `qsa_op_supported()` now builds a minimal probe tensor and asks
  `ggml_backend_dev_supports_op(model.dev_layer(il), probe)`; under `-sm tensor` that device is the
  Meta device, whose `supports_op()` is `all_of(sub-devs)`, so the query is the meta-split safety
  condition.  The `LLM_FUSED_OP_FLASH_ATTN_QSA` probe the item suggested is structurally impossible
  (a QSA node exists only above the 2051 selection width, so a reserve-time probe graph has none).
  Probe table: 0 mismatches vs the old list on gfx1151, plus an unsupported head size (D=80) now
  rejected where the list accepted it; same-seed text byte-identical to the pre-amendment build;
  cost 0.112 us/call.
* **Gates:** strict 15/15 apply (tree == canonical) and `FLASH_ATTN_QSA` 22/22 + `FLASH_ATTN_EXT`
  pass; the default is byte-identical to the pre-amendment build (f16 `0fc4910d5824` 632 chars for
  `plain == n_max 3`, q8_0 `e8f8bba3942b` 626 chars for `plain == n_max 7`); beta re-cut builds clean,
  its `FLASH_ATTN_QSA` suite is 22/22, and all four gate combos (default / `GGML_QSA_SCORE_MEM=0` /
  `GGML_QSA_DERIVED_*=0` / `LLAMA_QSA_KEYS_ONLY=0`) plus `draft-mtp n_max 3` are byte-identical
  (`0fc4910d5824`, 632 chars) = the delivery's value.  The 14th re-cut also **folded the missing
  `nullptr, nullptr` argument into the beta commit**: the 13th re-cut's exported patch had it only in
  the worktree, not in the commit, so a clean `git am` of that patch would not have compiled.
* **TODO**: item 9 removed from Active (Closed one-liner added); Active is now items 3 and 4 only.
  One observation recorded, not filed as an item: on the substitute PPL text the halo sparse path
  reads 24.71 against the same selection computed densely at 22.28 - the documented oracle text is
  absent on this box, so this is not comparable with the recorded 6.5267/6.5306 parity and is left as
  an observation (the patch does not touch that path).

## 2026-09-12 (8) — TODO triage: Active cut from 13 items to 3, item 11 closed with a measurement

No delivery change (one experiment implemented, measured and **reverted**).

- **Item 11 (MXFP4/NVFP4 fused gate+up+GLU MMQ) attempted and closed — the type-list edit is a no-op.**
  Implemented the planned change (`GGML_TYPE_MXFP4` in `MMQ_GATE_TYPES` + the generated gate instance, the
  `ggml_cuda_mul_mat_q_switch_type_gate` case, `moe_mmq_type`), built it, and instrumented the gate case
  with a one-shot counter: **0 firings** over a full `gpt-oss-20b-MXFP4` prefill with the arm enabled.
  The model's MoE graph is the expert-bias `{MUL_MAT_ID, ADD_ID, MUL_MAT_ID, ADD_ID, GLU}` pattern, whose
  only fused arm is the **mmvq/decode** one — there is no MMQ (prefill) fused arm for it and the MMQ
  fused epilogue has no `x_bias`/`gate_bias`/scale support.  Perf ~0 (pp2048 1741.3 vs 1742.0 t/s,
  pp16384 1506.7 vs 1501.6, fused vs `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`), same-seed text byte-identical.
  Experiment reverted; `wip`-free.  Side finding: `generate_cu_files.py`'s `SOURCE_MMQ_GATE` re-emits the
  file header on append, so re-running the generator mutates the 5 committed gate instance files.
- **TODO restructure (the point of the session):** Active is now only what this repo will work on next —
  items **3** (`iq4_nl` prefill), **4** (QSA sparse residual + the `embeddings_nextn` logits caveat) and
  **9** (QSA knobs).  Items 1/6/8/12 → *Waiting on others* (maintainer go-ahead, other hardware, upstream
  filing); 5(c)/5(d)/5(g)/13 → *accepted limitations* (item 5(d): the mmq `sum[]` overflow is latent — no
  upstream config violates `I >= nwarps*16`, so there is no reproducer to file); 5(a)/5(b)/15/16 →
  *Parked*; item 14 → *Closed* (canonical chain re-verified at `13af95ac1`).  No item content was deleted —
  every moved item keeps its body under the new heading, and the details stay in the dated records.

## 2026-09-12 (7) — item 16 re-scoped (the "pin" plan is a dead end) and item 15's `-Wshadow` audit

No delivery change.

- **TODO item 16** (restore the ~0.9 % `tg128` the block-13 RDNA3_5 fusion skip costs): the suggested
  "pin `nwarps`/`rps`/item-split" fix does **not** apply.  Verified against the delivery: the fused and
  unfused dense `ncols_dst==1` arms already share the same `mul_mat_vec_q_ksplit<…,has_fusion,…>`
  template, the same `calc_nwarps(type,1,table_id)` (RDNA3_5: 2 for `Q8_0`, else 1), `rows_per_block` 1
  and identical launch dims; the fused epilogue uses the same `ggml_cuda_op_silu_single` as the standalone
  GLU (`op_silu`), and `up * silu(gate)` is commutative.  Two live candidates: **(a) codegen**
  (`has_fusion` adds registers + a second `vec_dot` in the inner loop and may contract the `tmp` FMAs
  differently) and **(b) the Q8_1 cache** (`common.cuh:1611` — keyed on the src1 tensor/layout only, not
  the weight type, while `quantize_row_q8_1_cuda` takes `src0->type`; fusing changes which call fills it).
  Next step: dump `tmp`/`tmp_gate` from the ksplit kernel under an env at `W=1`.  Record
  `archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md` §9.
- **TODO item 15** (`-Wshadow` for `src/`, which would have caught the Block-15 dead-mask bug): audited by
  replaying the tree's own host compile commands for the 186 `src/` TUs with `-Wshadow` — **128 warnings
  in 27 files**, 46 of them the risky `shadows a local variable` class (82 are benign `shadows a field`,
  mostly constructor params).  `src/models/qwen4exp.cpp` is clean.  Revised proposal:
  `-Wshadow -Wno-shadow-field-in-constructor` for `src/` + fix the ~46 local sites in their own cleanup
  block.  Record `archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md` (full 46-site list).

## 2026-09-12 (6) — QSA forced-sparse q8_0 residual (TODO item 4): it is not a width dependence; `embeddings_nextn` breaks logits-level `plain == draft-mtp`

No delivery change.  Deep dive on the one open item-4 residual (forced sparse + `-ctk q8_0` + the
`p5000` prompt: `plain a57bc13bbf2a` vs `n3 3124adfd2b94`).

- **It is not a decode/verify width dependence.**  A new multi-step teacher-forced replay
  (`archive/work/strix-halo/qsa-item4/mstep.cpp`) of the plain greedy sequence in the exact residual config is
  bit-pure at every width: 200 positions, `W = 1..8`, with a spec-like batch+rollback schedule, with
  unrelated tokens in the rolled-back rows, and with `n_rs_seq` 0 vs 2/3 — 0 mismatches.  The recurrent
  snapshot rollback restore is exact and rolled-back content does not leak.
- **Sharp signature:** pure at `--spec-draft-n-max 1` (MTP genuinely active, 31.1 t/s vs plain 23.7);
  `n_max 2/3/5/7` all land on the *same* divergent text (first diff char 458).
- **Ruled out:** `n_rs_seq`, `n_outputs_max` (`1+n_max`), CUDA-graph capture (`GGML_CUDA_GRAPH_OPT=0`),
  and the chunked-GDN prefill boundary — the boundary is a real hazard (moving it by one token changes
  the text) and it is why `GGML_CUDA_GDN_CHUNKED=0` moves the *plain* stream at char 49, but an
  instrumented `gated_delta_net.cu` shows the actual chunked-GDN call sequence is **identical** between
  the runs (144 calls, same sizes).  So `GDN_CHUNKED=0` / `DISABLE_FUSION=1` "reconcile" by perturbing
  the trajectory, not by localising the cause (correcting the earlier record's reading).
- **New concrete defect:** the MTP driver enables the target's `embeddings_nextn`
  (`common/speculative.cpp:1431`), which makes qwen4exp's last-layer output gather defer
  (`gather_now` in `src/models/qwen4exp.cpp`) so the last layer runs on the full ubatch — the **prefill's
  last-position logits shift by a ULP** (`ad3acaa7…` vs `b624a79f…`).  That is a real logits-level
  violation of the `plain == draft-mtp` guarantee (item 4(a)), though it does not by itself flip the
  replayed tokens.
- **Disposition:** item 4 stays open, re-scoped to a driver-level divergence; the next step is a faithful
  mini-MTP driver (target + draft, per-step target-logit dump), since everything cheaper is exhausted.
  Records: `archive/work/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md`; analysis `GREEDY-PURITY.md` §18;
  `TODO.md` item 4.

## 2026-09-12 (4) — QSA sparse-regime width purity on gfx1151: items 4/7 re-measured (item 4 re-scoped, item 7 closed)

No delivery change.  Re-measured the two QSA-*sparse*-regime width dependences that TODO item 4 recorded
on 2026-09-11 (on the 3-GPU gfx1201 box, sparse arm forced) — both were measured **before** the
2026-09-12 block-13 RDNA3_5 mmvq-fusion amendment, and **neither reproduces on gfx1151 with the current
delivery**:

- the fused indexer score **is** byte-identical to the per-op chain: a 512-token forced-sparse A/B
  (qwen4exp UD-IQ4_XS, f16/bf16, `P=5000`) gives the same text for `GGML_CUDA_QSA_INDEXER_SCORE` and
  `_CACHE` at their defaults and at 0 (`0d29890e0f04` f16), and the `CACHE=2` unfilled-pool probe does
  move the W=1 text (so the fused path is the one running);
- the recorded "residual split" was the block-13 single-token mmvq fusion (§25): the current delivery is
  `plain == n3 = cb2912b186b9`, and the pre-fix impurity reproduces exactly with
  `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` (`471ea250f8e2` vs `cb2912b186b9`).

**Default gfx1151 configs are pure**: shallow dense decode on every tested KV type (q8_0 included) and
deep sparse decode at ~74K (f16 `83e0ed0f0f80`, q8_0 `7205399d367d` — the maintainer's `-ctk q8_0`
config).  Item 7 (the "dense decode at every depth" workaround) is therefore **closed** — the 64K
crossover stays.

One residual remains and is **open/unlocalised**: a prompt-dependent q8_0 width dependence in the
*forced*-sparse shallow regime (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`, `/tmp/p5000.txt`: `plain a57bc13bbf2a`
vs `n3 3124adfd2b94`).  `LLAMA_QSA_SPARSE_FA=0` does not reconcile it (the standard masked-FA path is
affected too), `LLAMA_QSA_OFF=1` does, and `GGML_CUDA_DISABLE_FUSION=1` / `GGML_CUDA_GDN_CHUNKED=0` each
perturb to purity.  It is a ULP-level effect (the default deep q8_0 config is pure).  Next step: a
node-dump/op-trace rebuild to diff the W=1 and W=4 graphs.  Item 4 is re-scoped to this.  Record:
`archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`; analysis `GREEDY-PURITY.md` §18; docs updated
(`AGENTS.md`, `TODO.md`).

## 2026-09-12 (5) — item 5(f): the block-13 fused MoE gate+up+GLU arm still wins on Strix Halo

Re-measured on the current delivery tip (35B-A3B Q4_K_M, 1 GPU, interleaved
`GGML_CUDA_DISABLE_MOE_MMQ_FUSION` off/on ×3, pp2048 and pp16384): the fusion is still worth
**+0.6 %** prefill at both sizes (pp2048 1711.9/1710.2 vs 1710.1/1701.4 t/s; pp16384 1485.3/1485.8 vs
1476.4/1478.6 — the first p2048 off-run 1733.3 is a warm-up outlier) and the fusion fires, so TODO item
5(f) is **closed: keep the arm**.  Docs-only; no delivery change.

## 2026-09-12 (3) — TODO.md audit: the Active list is active-only, closed items moved out, state header refreshed

Docs-only tracker cleanup (no delivery change).  `TODO.md`'s *Active* list had accumulated finished-work
footnotes, so the file no longer told a reader what was actually open: item 2 was an empty heading for the
fixed issue-25 GDN divergence (heading deleted; already in Closed), item 1's Block-15 dense-arm blocker
narrative was a Closed record repeated in Active (trimmed to the live 12th-re-cut + beta-window state),
item 6 carried the completed gfx1201 port and Phase-2.5 narrative (moved to Closed, leaving only the open
gfx1100/gfx1151 legs), item 5(e) (gfx1100/gfx1201) duplicated item 6 and was dropped, and the state header
still named the superseded canonical tip `124abba9e` / tree `d7c8e898…` (now `13af95ac1` /
`f4791066f4…`, matching `make-patches.sh`).  Added the Closed one-liner for the 2026-09-12 (2) block-13
RDNA3_5 mmvq-fusion purity amendment and a new Active item 16 for its perf follow-up (make the fused
`ncols_dst==1` kernels reproduce the standalone reduction rather than skip the fusion).  Also prepared the
post-compaction brief `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.

## 2026-09-12 (2) — block 13: the RDNA3_5 single-token-only mmvq fusions are not decode/verify bit-identical (folded)

**Canonical tip `13af95ac1`** (tree `f4791066f4a582316b1ca95f51c96cd10b905ef7`), 15 blocks,
clean-apply strict 15/15 `git am` with 0 whitespace warnings and the applied tree equal to the
canonical one.  One block amendment (block 13), one net-patch regeneration.  Full record:
`GREEDY-PURITY.md` §25 and the (now folded) `archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md`.

**Block 13 — the two RDNA3_5 single-token-only mmvq fusions are skipped on gfx1151.**  The
2026-09-11 block-13 band work made the *standalone* mmvq path `W = 1..8`-uniform, but on gfx1151 two
**single-token-only** fusions still ran at `W=1` only and their fused kernels do not reproduce the
standalone arithmetic, so a 1-token decode and an n-token verify of the same layer were not
bit-identical (the issue-25 "block-13 `n_q=1` short-K mmvq variance"): the dense gate+up+GLU mmvq
fusion (`mul_mat_vec_q<..., ncols=1, has_fusion=true>`; `mmvq.cu` restricts fusion to `ncols_dst == 1`)
and the MoE weighted-down tail `ggml_cuda_mul_mat_id_weighted_rdna3_5` (RDNA3_5-only, single-token by
its shape fingerprint).  Measured (qwen4exp UD-IQ4_XS, `P=100`, f16): `W=1` `8abc6206` vs `W=8`
`453eaa61`; each fusion moves `W=1` independently and only both together equal the `W=8` standalone.
The fix guards the six `{op,op,GLU}`/`{op,bias,op,bias,GLU}` matchers in `ggml_cuda_try_fuse` (keeping
the band-uniform `MUL_MAT_ID`/MoE fusions) and `ggml_cuda_mul_mat_id_weighted_rdna3_5_ok`, both gated
on RDNA3_5 unless `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` (A/B).  Post-fix `W = 1,2,4,8` is
one hash per config: qwen4exp f16 `453eaa61`, q8_0 `113696b9`, MoE 35B-A3B `18999a78`; the 27B dense
(`e165ef98`) was already pure and is unchanged.  Cost ≈ −0.9 % `tg128` on qwen4exp (25.53 vs 25.77
t/s), prefill flat — the §19 trade; the follow-up is to make the fused `ncols_dst==1` kernel reproduce
the standalone reduction instead of skipping the fusion.  The gfx1201 path is untouched
(`GGML_CUDA_CC_IS_RDNA3_5`-only).

**Placement.**  The dense GLU matchers are upstream at the fork point and the weighted-down `_ok` is a
block-13 addition, so the whole fix lands in block 13 — not block 00 (which is generated from the fork
point and touches only `fattn-common.cuh` + Vulkan shaders, and the weighted-down matcher does not
exist there), and not block 14 (which owns the weighted-down *matcher*; guarding in `_ok` keeps block
13 self-contained and avoids a mid-chain rebase of block 14's overlapping `ggml-cuda.cu` hunks).

**Regeneration.**  Rebuilt the canonical chain by `scripts/apply-all.sh` at `9113cc188` from the
pre-amendment `main` patches, amended block 13 (`f5d0cdd25`), replayed block 14 (`13af95ac1`) and ran
`scripts/make-patches.sh`; blocks 00-12 and 14 patch bodies are byte-identical apart from the
`From`/`index`/hunk-header lines, only block 13's body changed.  `rdna-boosts-all.patch` regenerated
(`git diff 9113cc188 13af95ac1`) and verified equal to the regenerated `patches/` applied at the base.

**Beta block-15 re-cut (12th).**  Re-cut on this base: base `13af95ac1`, beta tip `888a59ee0`, tree
`476d2d1e95947de7cc8cd806c40efc0f01927cd3`; the exported patch is byte-identical to the 11th re-cut
apart from the `From <sha>` line (block 13's amendment touches only `ggml-cuda.cu`/`mmvq.cu`, which the
block-15 patch does not touch), and strict `git am` applies.  Beta-tree revalidation: build clean;
width probe `W = 1,4,8` one hash on qwen4exp f16 (`453eaa61`) / q8_0 (`113696b9`); a same-seed greedy
run is byte-identical delivery-vs-beta; `FLASH_ATTN_QSA` + `GATED_DELTA_NET` pass.  See
`archive/work/block-15-campaign-wins/BETA-TESTING.md` (12th-re-cut section).

## 2026-09-12 — block 13: the fused shared-expert epilogue is column-blocked (the item-5 cost repaid), and the routed-compact MoE MMQ claim re-verified

**Canonical tip `124abba9e`** (tree `d7c8e8984b8bd65838d8ae58c0f5de449d9c5d4d`), 15 blocks, clean-apply
strict 15/15 with 0 whitespace warnings and the applied tree equal to the canonical one; the sim build
(`/tmp/simx`) reproduces the MoE probe gate `ac8825358d9adfda` at `W = 1,4,8`.  One block amendment
(block 13), one staged-beta re-cut (11th), one validation record (the gfx1201 routed-compact probe).
Both items come from `TODO.md` items 10 and 6, worked to the brief
`archive/work/items-6-10-wrapup/HANDOVER-2026-09-12-items-6-and-10.md`.

**Block 13 — `shexp_down_gated_q8_0` is now column-blocked (band-internal).**  The 2026-09-11 band
amendment made the fused shared-expert epilogue serve the whole decode/verify band but launched it as
`grid = (nrows, ncols)` — one block per `(output row, token)` — so the down-weight row was re-read once
per token and the whole block (two barriers, the cross-warp reduction and the epilogue) was duplicated
per token.  On the reachable geometry this is severe: for Qwen3.6-35B-A3B (`k_down` 512 → 16 k-blocks,
`vdr` 4, `nwarps` 8) `blocks_per_iter` = 128 > 16, so **only warp 0 of 8 does any work** (88 % of the
block idles) and the weight row is read 8x over at `pl = 8`.

The kernel is now templated on `ncols_dst` as well, with the **token loop inside the k-block loop**, a
per-token accumulator per thread and the weight block read once per `(row, k-block)` for the whole band;
`grid` is `(nrows)` with the band block-internal.  Two invariants are preserved exactly, which is what
makes the change numerically invisible:

* `nwarps` stays pinned to the single-token value (`calc_nwarps(GGML_Q8_0, 1, table_id)`), because it
  sets `blocks_per_iter` and hence the down-projection reduction order;
* each token keeps the single-token path's per-thread accumulation order *and* the same cross-warp
  reduction order (serial `sh_down[l]` adds in `l` order, then `warp_reduce_sum`), so `decode == verify`
  holds by construction, not by measurement.  The `__fmul_rn` epilogue (no FMA contraction) and the
  `dst[t*nrows + row]` layout are unchanged.

Perf (`llama-batched-bench`, 35B-A3B Q4_K_M, 3-GPU tensor, `-npp 2048 -ntg 128 -npl 1,2,4,8`, f16 KV,
two interleaved reps, `pl` is the batch width = `n_max + 1`):

| `pl` | before (fused) | after (fused) | unfused reference |
|---|---|---|---|
| 1 | 95.84 / 95.64 | 95.57 / 95.55 | 93.55 / 93.11 |
| 2 | 167.69 / 167.48 | 168.65 / 168.45 | 165.88 / 165.02 |
| 4 | 299.07 / 299.49 | **306.52 / 305.79** | 299.86 / 299.96 |
| 8 | 461.00 / 461.09 | **475.41 / 473.07** | 472.65 / 470.72 |

i.e. `pl 8 +3.1 %`, `pl 4 +2.4 %`, `pl 2 +0.6 %`, `pl 1` flat — and the fused default is now **ahead of**
the unfused `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` reference at every width, where before it lost 2.4 % at
`pl 8`.  With `--spec-draft-n-max` capped at 7 the payout band is exactly `pl <= 8`, the widest
*supported* verify batch.

Numerical invisibility was proven with a **direct old-vs-new A/B** (both `libggml-hip.so` builds of the
same tip kept side by side and swapped in, the `tools/sobench.sh` idiom) rather than by trusting
documented values:

* MoE probe (`/tmp/lw-f2`, 1 GPU, `SPLIT=layer`, `RS=0`, `CB=0`, `P=256`, `p0long.txt`): fused
  `W = 1..8` **all `ac8825358d9adfda`** before and after; unfused all `bd138ad2326fbbf2` before and
  after.  Both are the documented gate values, so the fix is a **no-op at the gate config** — the
  strongest available control.
* The §5 acceptance matrix is unchanged: qwen4exp tensor all-W `dcf1ae667f730879`, layer
  `3adeb313042a871b`; 27B layer `4089b4d40b91090c`, tensor `91434ea90f2cbfa0`.
* §19 text gate on 35B-A3B (Protocol A prompt, 96 tokens, 1 GPU, f16 KV):
  `--spec-type none == draft-mtp n_max 3 == n_max 7` = `68c0a24ed8d4` (447 chars) before and after.
* MTP acceptance (Protocol A, `n_max 3`, `n=96`): `0.87179` before and after (identical
  `68 accepted / 78 generated`, mean len 3.62).
* `test-backend-ops`: `FLASH_ATTN_EXT`, `FLASH_ATTN_QSA`, `GATED_DELTA_NET` all pass on ROCm 0/1/2.
* The change also leaves qwen4exp's documented sparse text `804de0576868` untouched (re-checked while
  validating the re-cut).

**Beta re-cut (11th).**  Base `124abba9e` → beta tip **`a90f75896`**, tree
**`ed6ee74df8b690c5a1584adb3f85c45eda70a09b`**, patch still **3 811 lines**; `git am -3` applies with no
conflict and the exported patch differs from the 10th re-cut **only in the `From <sha>` line** (block 13's
amendment does not touch any file the beta patch hunk-touches).  Round-tripped (fresh worktree at the base
+ `git am -3` → identical tree), builds clean (`/tmp/blk15z/build-rec11`), and the smoke gates reproduce
the 10th re-cut's values exactly: qwen4exp f16 sparse text `804de0576868`, QSA oracle sparse `6.5394` /
dense `6.5377`.

**Item 6 — the gfx1201 routed-compact MoE MMQ ("Phase 2.5") re-verified; two corrections.**  The port's
in-code claim is "*Numerics are bit-identical to the plain `mul_mat_q` path (same `mul_mat_q_process_tile`,
same per-tile accumulation order; only the tile enumeration differs)*".  Re-checked on the current tip
with `GGML_CUDA_DISABLE_MMQ_ROUTED` on/off:

* **Byte-identity holds, on two different expert types/J bands.**  qwen4exp (IQ4_XS, J=64): same-seed
  greedy text `804de0576868` both ways; 35B-A3B Q4_K_M (Q4_K, J=32): `68c0a24ed8d4` both ways.  Probe
  hashes `W = 1..8` identical on both models under both settings (MoE `ac8825358d9adfda`, qwen4exp tensor
  `dcf1ae667f730879`), and the MoE MTP acceptance is `0.87179` either way.
* **Correction 1 — the brief's premise was wrong.**  It assumed the 35B-A3B Q4_K_M does *not* take the
  routed path and could serve as the "plain" control.  It does: `mmq_rdna3_5_id_use_compact` accepts
  Q4_K/Q5_K/Q6_K, and a `rocprofv3 --kernel-trace` count shows **480
  `mul_mat_q_routed_compact<(ggml_type)12, 32, false>`** launches per `pp512`/`ub512` run (type 12 =
  Q4_K, J = 32), i.e. the Q4_K experts take it too.  Both available MoE models therefore exercise the
  compact dispatch — which *strengthens* the validation to two type/J bands but removes the proposed
  control.  The control is instead the **prefill-only reach**: a `tg` run shows **0** compact launches
  and **0** descriptor-builder launches, because decode and the verify band go through mmvq
  (`ncols_dst <= MMQ_MAX_BATCH_SIZE`), which is also why the compact dispatch cannot affect width
  purity.
* **Correction 2 — the env opt-out does not isolate the whole port.**  `GGML_CUDA_DISABLE_MMQ_ROUTED=1`
  disables only the compact *enumeration*; the per-expert J selection (`mmq_rdna3_5_id_get_J` in
  `mul_mat_q_switch_J`) stays active in both arms (the code says so explicitly).  So ON==OFF proves the
  compact enumeration is arithmetic-neutral, not the J selection.  The J change is arithmetic-neutral by
  construction (J is the output-row tile width; an output element's accumulation is over K only), and it
  is additionally covered by the delivered hash table, the MoE probe/text/MTP gates above and
  `test-backend-ops -o MUL_MAT_ID` (which also passes).
* Perf claim re-measured (interleaved, 2 reps, ub2048, 3-GPU tensor, f16 KV, `-r 2`): qwen4exp pp512
  **+11.1 % / +9.3 %**, pp2048 **+5.6 % / +4.6 %**, pp8192 **+4.5 % / +3.1 %**, pp16384
  **+4.0 % / +3.6 %**, tg128 flat (51.57 vs 51.58); 35B-A3B pp512 **+5.1 % / +5.3 %**, pp2048
  **+7.7 % / +7.8 %**, pp8192 **+7.4 % / +7.2 %**, pp16384 **+6.9 % / +7.0 %**, tg128 flat (99.61 vs
  99.45).  The 2026-09-06 record's "+4-8 % prefill, tg flat" is reproduced on both models.

**One measurement caveat recorded for the follow-ups.**  The cross-day comparison against the port's
2026-09-06 *absolute* numbers is not usable: qwen4exp `pp2048` (f16 KV, same config) ran
2042.6 -> 1933.7 -> 1906.1 -> 1822.0 -> 1730.8 t/s over one session (a monotone -15 % drift while the box
sits at 141 GiB buff/cache with swap full), while the 35B-A3B `pp512` control reproduced to 0.2 % in the
same window (5372.2 vs 5360.1/5353.9).  Only same-session interleaved brackets are meaningful for this
model's prefill; the same warning is now on `TODO.md` item 3 (the qwen4exp `iq4_nl` prefill-delta item,
which is an 8-12 % claim measured on this same axis).  A quick QSA-arm check in the same window
(`LLAMA_QSA_OFF=1` +2.2 % pp2048 / +7.4 % pp8192, `LLAMA_QSA_SPARSE_FA=0` +2.3 % / +2.7 %) shows the QSA
machinery is *not* the explanation for the drift.

**Pushing/tips:** `scripts/make-patches.sh` default tip -> `124abba9e`; `rdna-boosts-all.patch`
regenerated by hand (22 347 lines, 115 files, `git apply --check` clean at `9113cc188` and a full apply
reproduces the canonical tree).

## 2026-09-11 (12) — mixed K/V types hard-rejected, `--spec-draft-n-max` capped at 7, and issue #25's GDN divergence re-verified

**Canonical tip `484231cb9`** (tree `fc3c73da4ac68e92348043b992fb963b006e14df`), 15 blocks, clean-apply
strict 15/15 with 0 whitespace warnings and the applied tree equal to the canonical one (sim build
verified).  Two block amendments, both from maintainer decisions of 2026-09-11:

**Block 14 — mixed K/V cache types are now HARD-REJECTED for every model.**  Upstream enforces
`type_k == type_v` for MLA/DeepSeek4 only; the condition is dropped, so `params.type_k !=
params.type_v` now fails context creation for every architecture with

```
E llama_init_from_model: models require the same K and V cache types, got K=q8_0 and V=f16; set
  --cache-type-v to match --cache-type-k (both default to f16)
```

Rationale (measured, `TODO.md` accepted limitations / `GREEDY-PURITY.md`): every mixed pair is
1.7–3.6× slower than the same-type equivalent and never smaller, and the attention path — including the
split/flash-attention one, whose type gate lives a few lines above — assumes `type_k == type_v`.  Both
types default to f16, so only an explicit `--cache-type-k`/`-v` can trigger it.  Verified: `-ctk q8_0`
(V=f16) and `-ctk q8_0 -ctv q4_0` both fail with the message above; `-ctk q8_0 -ctv q8_0` runs normally.

**Block 01 — `--spec-draft-n-max` is capped at 7** (a clamp with a visible notice, **not** an error,
per the maintainer's instruction).  A verify batch decodes `n_max + 1` query rows and the HIP
flash-attention chooser switches the band from the tile kernel to the MMA/WMMA kernel above 8 rows
(`fattn.cu`, the `Q->ne[1] > 8` switch); the two kernels are not bit-identical, so a deeper draft makes
decode and verify disagree and greedy output can change between `--spec-type none` and `draft-mtp`
(upstream master has the same class of boundary).  The guarantee published in `GREEDY-PURITY.md` §11 is
therefore enforced rather than documented:

* the clamp lives in **`common_init_from_params`**, not in the argument parser, because a warning
  emitted while parsing is *below the default log threshold* and never reaches the user (verified: the
  control `--log-mmap` combination warning is equally invisible; `--log-verbosity 4` shows both) — it
  runs before the model/context and before the speculative engine are created, so all of them see the
  capped depth;
* the notice is emitted at `LOG_ERR` level deliberately (llama-cli's default verbosity hides `W` but
  shows `E`; `common_fit_params` uses the same pattern for its non-fatal abort notice) and **names the
  escape hatch**: `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` keeps the configured value (with a `W` notice);
* the help string now reads "(default: 3, max: 7)".

Verified end to end on the 27B (2-GPU): `--spec-draft-n-max 12` → the notice **at the default
verbosity** and the GDN log line showing `K=8` (= n_max 7 + 1); `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` →
`K=13` (= 12 + 1, i.e. the env really reaches the kernels) with the "keeping it" notice; `n_max 7` and
`n_max 4` are silent and give `K=8`/`K=5`.  Note (documented in the code): unclamping to `n_max > 15`
re-introduces the K-dependent chunked-GDN boundary as well.

**Issue #25's GDN plain-vs-spec divergence: already fixed, re-verified, and the records corrected.**
A concurrent gfx1151 session reported that `--spec-type none` and `draft-mtp` disagreed through the GDN
chunked prefill.  That was fixed on 2026-09-11 by block 02's **K-independent whole-batch chunked
prefill** (`GGML_CUDA_GDN_ALIGN_BOUNDARY` and both K-dependent branches deleted); the `TODO.md` item and
the `archive/work/issue-25-mtp-batch-width/` status lines still described the superseded opt-in gate, and are now
corrected.  **Fresh gate on the current tree** (27B Q8_0, 2-GPU `-sm tensor -ts 1/1`, `p0long.txt`, 512
greedy tokens, `-c 8192 -ctk f16 -ctv f16 -fa auto`): `--spec-type none == draft-mtp n_max 1 == 4 ==
5`, all `299566b902bb` (2727 chars) — byte-identical.  Control: `GGML_CUDA_GDN_CHUNKED=0` changes the
plain text (`60777872b890`), which is the expected chunked-vs-sequential kernel difference (and that
switch remains the fully-snapshot-safe fallback), not a plain-vs-spec divergence.

**Beta:** tenth re-cut — base `484231cb9` → beta tip **`a796a1d49`**, tree
**`b48565e69f77f0c20a20cd75d87c2559d11e6de2`**, patch **3 811 lines**; `git am -3` merged the new
`llama-context.cpp` region **without a conflict**, and the diff vs the ninth re-cut is exactly the three
new delivery files (`common/arg.cpp`, `common/common.cpp`, `src/llama-context.cpp`) — no block-15 content
changed.  Beta testers must pass matching `-ctk`/`-ctv` from now on (the hard reject applies to the beta
too); see `archive/work/block-15-campaign-wins/BETA-TESTING.md`.

## 2026-09-11 (11) — Block 15's dense-arm blocker fixed (a shadowed variable); no delivery change

**Delivery unchanged** (`main` still the 15-patch set at canonical tip `6d3155faa`, tree
`0c3f0c2c2f4e7439d9489d45573a4021a8eee106`): the defect lived in block 15's own `build_attn_qsa` dense
path, which is **not** in the delivery (the delivery has no `if (kq_mask != nullptr)` wrapper and no outer
declaration), so nothing in `patches/` changes.  Only the staged beta patch is amended — the ninth re-cut.

**Root cause (one line, found by instrumentation after every hypothesis in the handover was excluded):**
the V2/V3 refactor wrapped the top-k mask chain in `if (kq_mask != nullptr) { ... }` and declared an
*outer* `ggml_tensor * kq_mask_top_k = nullptr;`, leaving the chain's own
`ggml_tensor * kq_mask_top_k = ggml_set_rows(...)` inside the block as a **new local**.  The chain was
therefore built whenever the mask existed, but its result never reached the attention — `build_attn_mha`
received the outer `nullptr`.  Consequences: the chain's nodes were unreachable from the graph output (so
`ggml_build_forward_expand` never emitted them), the packed mask lost its only consumer (the allocator
left it unallocated, and block 15's own `if (self_kq_mask && self_kq_mask->buffer)` guard in
`llm_graph_input_attn_kv::set_input` then skipped `set_input_kq_mask`), and the dense arm attended with **no
mask at all** — a full causal leak.  Fix: drop the inner `ggml_tensor *` so the block assigns the outer
variable.

**How it was isolated** (full detail: `archive/work/block15-dense-arm/HANDOVER-2026-09-11-block15-dense-arm.md`):
the dense arm also differed in a plain text run; it *still* differed with `-fa off` (⇒ not the FA kernels,
not V3's derived-mask arm); `archive/work/block-15-campaign-wins/ab/w4-revert.patch` + rebuild changed nothing (⇒
not W4); `LLAMA_KQ_MASK_DERIVED=0` removed the resolver's derived-mask warnings (a working positive
control) but not the leak (⇒ not V3); then the node dump
(`archive/work/kv-quant-purity-followups/tools/node-dump-instrumentation.patch`, `GGML_CUDA_NODE_DUMP=1/2` +
`/tmp/nodedump_on`, `--verbose` needed for the ggml-level INFO lines) showed the delivery's dense prefill
consuming `attn_inp_kq_mask` 36 times (12 indexer layers × 3 devices) while the beta consumed it **zero**
times and emitted **no** `FILL`/`SET_ROWS` chain nodes at all; a temporary `[QDM]` log then printed
`kq_mask=1` (the guard passes) with `outer_top_k=0` (what the attention reads is still null) — the
shadowing, in one line.  A cheap by-product instrument is now the first thing to try for any "is the model
seeing the future?" question: **random text** (`/tmp/rand-text.txt`, 40 000 random words) — a model that
can see the target scores ≈1 on noise, where the broken beta gave `1.0205` and the delivery `19.0589`.

**Gates after the fix (identical configs, against the delivery build):**
`tools/qsa-ppl-oracle.sh tensor f16` → sparse `6.5394` / dense `6.5377` (= the delivery; the blocker's
`1.0558` is gone); dense-arm greedy texts byte-identical to the delivery — tensor f16 `2daa19579316` (720
chars), tensor `iq4_nl` `3c46e47ab345` (680), layer f16 `e656b50f2cc8` (685), layer f16 `-fa off`
`b96459bf02ca` (703); random-text PPL `19.0589` @ c2560/ub2560 and `7.9682` @ c4096/ub512 (= the delivery);
production arm untouched — sparse f16 `804de0576868`, q4_1 `886292b17a93`, `plain == n_max 3 == n_max 7`,
MTP f16 `acc 0.56028` / pos-1 `(0.681, 0.553, 0.447)` bit-identical to the delivery on the same command,
`LLAMA_QSA_OFF=1` `6.5376`; the KV reserves are unchanged by the fix and still show the campaign's
mask-elision win (`1600.00 + 600.00` MiB at c204800/ub512 f16 vs the delivery's `1600.00 + 1800.00`, in
both arms); backend suites OK.  The pre-existing `iq4_nl` W2 sensitivity is unchanged (its greedy text
`fcb2d47f94cf` and MTP `0.46203`/`(0.717, 0.434, 0.226)` stay off the delivery's values, and
`GGML_QSA_DERIVED_* =0` restores them exactly — verified) because the sparse arm never enters the fixed
block.

**Beta:** ninth re-cut — base `6d3155faa` → beta tip **`3712e2dc1`**, tree
**`e39f8c2b6f0593113b93c4e57c512bc7373a2250`**, patch **3 811 lines** (the 8th re-cut + 1 diff line + the
commit-message paragraph); `git am -3` on a fresh base reproduces the tree exactly.  Records:
`archive/work/block-15-campaign-wins/{README,BETA-TESTING,HANDOVER}.md`; the revalidation pointer in
`TODO.md`.

**Lessons recorded in `GREEDY-PURITY.md` §23:** (1) a graph tensor with no consumer is *silently* dropped —
the allocator leaves it unallocated and the input fill is skipped, so "the input is in the graph" proves
nothing; (2) in a refactor that adds an outer declaration, an inner `Type * name = ...` **shadows** it and
the result is dead code that still compiles — `-Wshadow` (not currently enabled) would have caught this
class outright; (3) when a chain's nodes are missing from an executed-graph dump, suspect the *graph
builder* (reachability), not the allocator.

## 2026-09-11 (10) — `iq4_nl` becomes a first-class FA KV type (F3 step 2), and the beta re-cut finds a Block 15 blocker

**Canonical tip `6d3155faa`** (block 08 amended a fifth time, block 14 a fifth time), net tree
`0c3f0c2c2f4e7439d9489d45573a4021a8eee106`, 15 blocks, clean-apply strict 15/15 with 0 whitespace
warnings and the applied tree equal to the canonical one; the sim build's generated text is
byte-identical to the canonical build's (only its `build : <sha>` banner line differs, because the sim
chain has its own commit SHAs) and its `iq4_nl` text gate reproduces the canonical value.  Delivery
`main` carries the regenerated set (`rdna-boosts-all.patch` 22 233 lines, 115 files, +17 203/-935) and
the **8th** block-15 beta re-cut (`d0f71b2e8`, tree `39540b7f4fd8e8569dee64bfa3ee84bf1b20e75d`, patch
3 787 lines).

**The task: F3 step 2 = `iq4_nl`** (brief
`archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-f3-step2-iq4_nl.md`) — the last sub-`q8_0` KV type,
and the smallest cache of the set (288 MiB at c=32768 on the 4B, tied with `q4_0`, -72 % vs f16).
Before this, `iq4_nl` produced **no flash-attention call at all**: the predicate's `default:` clause
rejected it, the FA probe then disabled FA for the whole context.  After: **4B pp512 2269.8 -> 7931.8
t/s, tg32 48.5 -> 95.0** (`q4_0` 7913.1/96.8, f16 7981.7/99.7); dense models unchanged (27B 3-GPU
tensor pp8192/16384 within 0.7 % of f16, 4B pp8192 -2 %); `-sm tensor` now accepts the type.

**The mechanics were bookkeeping, not a new kernel** — the tile/MMA families stage K/V through
`ggml_get_to_fp16_cuda`, which already covers `iq4_nl` upstream.  What was missing: the predicate case,
the **15 `fattn-vec-instance-iq4_nl-*.cu` pairs** (upstream's generated cross product never had them
because `TYPES_KV` did not list the type - they ship with that list now, and `FA_ALL_QUANTS` gains its
15 pairs so that build mode stays complete), the K-side `vec_dot_fattn_vec_KQ_iq4_nl` (perm-based
`get_int_from_table_16` lookup, no bias) and V-side `dequantize_V_iq4_nl` (the q4_0/q5_0 nibble layout,
the `kvalues_iq4nl` table, no `-8`/`-16`) in `fattn-common.cuh`, the three CMake default lists, and - the
**one real latent bug** - the non-contiguous FA staging converter: `ggml_get_to_fp16_nc_cuda()` returned
`nullptr` for `iq4_nl` and `launch_fattn` called it, so any K/V *view* would have been a null-pointer
call.  Unreachable before (no FA path for the type), instant on the first `iq4_nl` backend-op case: the
very first `-o FLASH_ATTN_EXT` run **SIGSEGV'd in `launch_fattn<64,2,1>`**.  Fixed with
`dequantize_q4_nl` + all three NC switches.

**Gates** (final binary): `FLASH_ATTN_EXT` **5935/5935** (was 5599 - the 336 `iq4_nl` cases now run,
incl. mask/sink/alibi/softcap/permute/view variants), `FLASH_ATTN_QSA` **22/22** (two new cases at the
model's own geometry D=256 / gqa=12), `GATED_DELTA_NET` 46/46; `W=1..8` pure on 4B (1 GPU, both `RS`),
27B (both splits), MoE, gemma-4-E4B and qwen4exp (both splits, default **and** QSA-forced); qwen4exp text
`plain == n_max 3 == n_max 7` = `acd18ad2d55c` (tensor) / `a38a6e2d8efa` (layer) with the f16/q4_1
controls unmoved; MTP `n_max 3` 0.52727 (pos-1 0.757) and 27B f16 0.82716; perplexity oracle qwen4exp
tensor `iq4_nl` sparse 6.5244 / dense 6.4930 (controls within +-0.006, `iq4_nl` +0.031).  The vec-family
helpers are NVIDIA-only code on AMD, so they were validated by **forcing** the chooser to VEC with a
temporary env-gated instrument: 5935/5935 again with 880 forced hits (the instrument was reverted before
landing).

**Two open items, both filed** (`TODO.md`): (a) qwen4exp prefill is ~8-12 % slower for `iq4_nl` than for
f16/`q4_0`/`q4_1` at pp8192+, growing with context, even though `q4_0` has the identical byte layout —
`rocprofv3` shows it is **not** this amendment's code (QSA `iq4_nl` 1318.5 ms vs `q4_0` 1335.8 ms, same
VGPR/LDS/occupancy; dequant kernels identical at 1.2 ms; the executed graph identical at 1010 nodes, 0
diff; the traced kernel sum *lower* for `iq4_nl`), so the follow-up targets the host/launch side (the
per-type indexer op counts and the dense/sparse topology-flip sync); (b) **Block 15's
`LLAMA_QSA_SPARSE_FA=0` dense masked arm is broken for every KV type** (PPL ~1.05 vs the delivery's
6.49-6.55) — found by the 8th re-cut, pre-existing (the 7th re-cut reproduces it), not fixable by any
Block 15 gate, and a **promotion blocker** because that arm is this repo's quality oracle; the beta
records (`BETA-TESTING.md` §4c/§4d) now carry the evidence and add the oracle to the beta gate list.  The
re-cut also confirmed the beta's production path is byte-identical to the delivery (f16/q4_1 texts,
`iq4_nl` text, MTP acceptances, width purity, `FLASH_ATTN_QSA` 22/22, `FLASH_ATTN_EXT` 5940/5940,
`LLAMA_QSA_OFF=1` PPL) apart from W2's ULP-level derived-bias sensitivity on `iq4_nl` (benign: identical
sparse-arm PPL).

## 2026-09-11 (9) — the QSA kernel gets an oracle, four more KV types, and a head-group fix (quality)

**Canonical tip `a0cd6ce02`** (block 14 amended a fourth time; block 13 `1a88c92f5`), net tree
`0966e66731a4c3da85ffd96525688865a89242cd`, 15 blocks, clean-apply strict 15/15 with 0 whitespace
warnings, applied tree == canonical, sim build clean and its coherence hash equal to the canonical
build's (`1c5d32ac537d`).  Delivery `main` carries the regenerated set (`rdna-boosts-all.patch`
21 750 lines) and the 7th block-15 beta re-cut (`8a0e2eb3f`, tree `764808b4c`, patch 3 774 lines).

**The task was "let the fused sparse QSA op read the quantized caches" — it turned into a correctness
finding.**  Two changes, one amendment:

* **Quantized KV for QSA.**  `q4_0`/`q4_1`/`q5_0`/`q5_1` rows are now dequantized to F16 while a tile is
  staged (`get_dequantize_V<type_KV, half, 4>`, the idiom the vec FA kernel and the lightning indexer
  already use), with the four types threaded through the dispatch, `ggml_cuda_flash_attn_qsa_supported()`
  and `qsa_kv_native`.  Effect on 3x R9700 `-sm tensor`: `q4_1` prefill 2076.4 -> **2384.2 t/s at
  32 768** (dense masked reference 2078.1, f16 sparse 2380.9) — the quantized cache now tracks f16
  exactly, at pp8192 2404.1 (f16 2376.5) — i.e. the ~13.8 % long-context prefill the type used to lose
  is recovered, which was the measured prize that started this.
* **The head-group fix (the important half).**  A QSA block stages ONE K/V tile into shared memory and
  every warp reads it, so all of the block's q-heads must map to the same K/V head.  The chunking was
  `head_base += QSA_MAX_HEADS` (16) — and qwen4exp is 24 q-heads / 2 kv-heads = **gqa 12**, so a 16-warp
  block mixed heads 0..11 (kv 0) with 12..15 (kv 1) into the same smem rows (each staging thread adds
  its own head's K/V offset before the cooperative gather).  16 of 24 heads attended over the wrong V
  rows.  Now `min(QSA_MAX_HEADS, gqa_ratio)` heads per block (a no-op at gqa >= 16; for qwen4exp two
  blocks of 12).  Quality, measured as perplexity over 8 x 4096 tokens, 3-GPU `-sm layer`:
  **7.3269 +/- 0.151 -> 6.5267 +/- 0.132**, versus the dense masked oracle **6.5306 +/- 0.132** (the
  dense path computes the same top-k attention through the well-tested FA kernels).  The same table
  validates the new types (`q4_1` 6.5787 vs 6.5805 dense, `q5_0` 6.5444 vs 6.5375).

**It also fixed a hole in the test suite.**  `test-backend-ops` had **no** `FLASH_ATTN_QSA` coverage, and
the CPU reference (`ggml_compute_forward_flash_attn_qsa`) knew only f16/bf16/q8_0 — so the kernel that
serves qwen4exp's default attention path had *no oracle anywhere*.  This entry adds the four types to
the CPU reference and 18 `test_flash_attn_qsa` cases (all seven KV types; gqa 1 and 8; the three head
sizes; `n_tps` 1 and 4; sliced+combined top-k walks).  **0/18 -> 18/18**: the old kernel scores NMSE
~1.0 (i.e. it computes something else entirely), the fixed one < 5e-4.

**Why every earlier gate missed it** (`GREEDY-PURITY.md` §21): the corruption is *width-uniform*, so the
`W=1..8` purity matrix — the instrument behind every previous QSA finding — is structurally blind to
it; the probe never even executed the op (the QSA op only exists above the indexer selection width
`indexer_top_k + r - 1` = 2051, and the probe's `n_ctx` is 2048, so its max `P` = 2040 — forcing the
selection path with `LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0` is now part of the QSA
gate); and MTP acceptance pointed the *wrong way* (draft and main run the same wrong attention, so the
corrupted pair is self-consistent and accepts **more**: 0.65 vs 0.49).  The instruments that catch it
are the CPU oracle and the dense path as a reference — both now permanent.

**Validation** (all 3x R9700 gfx1201, canonical `a0cd6ce02`): `FLASH_ATTN_QSA` 18/18, `FLASH_ATTN_EXT`
5599/5599, `GATED_DELTA_NET` 4/4; probe purity with the QSA op forced at every width, all seven types,
both splits (f16 tensor `f400a002bd0af7df` is **identical** for the pre-fix and fixed builds — the fix
is provably a no-op in the tensor split, where the kernel sees one K/V head per device: `Q.ne2=12,
K.ne2=1`; layer f16 `18bc218586c80f91` -> `9aef99f6a614de4c`; the four new types
layer `83b460071c92c4be`/`9e6035525c2e3f07`/`c5e332fed9aa1c18`/`a0bad36e46adaf57`, tensor
`85cd44e288fe6124`/`595721104be83ac1`/`524d2df8d1be0987`/`3b4a5b989b988134`, all `W=1..8` pure);
text purity (`/tmp/prompt3k.txt` = 2122 tokens, just over the selection width, so the sparse arm really
runs) tensor f16 `804de0576868` (**unchanged** = the recorded reference), layer f16 `95817e5d366a`,
tensor `q4_1` `886292b17a93`, layer `q4_1` `b15e1c98dbf8`, tensor `q4_0` `26065aab382c`, each
`plain == n_max 3 == n_max 7`; MTP `n_max 3` pos-1 acceptance 0.651 (layer `q4_1`, aggregate 0.402) /
0.771 (tensor `q4_1`) / 0.49 (layer f16) / 0.47009 (tensor f16, unchanged); prefill `-sm tensor`
p8192/16384/32768 f16 2376.5/2435.1/2380.9, `q4_1` 2404.1/2452.5/2384.2, dense 2493.0/2411.8/2079.9;
decode `-sm tensor` d0/8192/32768 tg128 f16 51.4/51.8/49.9 (dense-decode default) vs 51.1/47.6/44.7
(forced sparse) — **the arch decode policy was re-measured on the fixed kernel and stands** (dense wins
at every depth); non-QSA regression: 4B/27B probe hashes reproduce exactly and every non-QSA file is
untouched.

**Landing**: block 14 amended in place (`git commit --amend`, the delta byte-identical to the validated
working diff — it is the tip, so no rebase), `scripts/make-patches.sh` default tip -> `a0cd6ce02`,
`rdna-boosts-all.patch` refreshed by hand, clean-apply sim re-verified, block 15 re-cut a seventh time
(one real conflict in `src/models/qwen4exp.cpp`: block 15's refactored `qwen4exp_qsa_sparse()` needs the
extended type conjunct; `fattn-qsa.cu`/`ops.cpp`/`test-backend-ops.cpp` auto-merged), beta patch
re-exported (3 774 lines, subject `[PATCH 15/15]`, round-trip verified).

**Next session's task (F3 step 2, `iq4_nl`) has its brief**:
`archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-f3-step2-iq4_nl.md` — the same two-block shape
(block 08 for the dense FA enablement, block 14 for QSA + the CPU oracle + the test), with the measured
pre-state (`-ctk iq4_nl` on the 4B is 2269.8 pp512 / 48.5 tg32 today because FA is disabled for the
whole context) and the prize (`iq4_nl` is the smallest KV cache of the set: 288 MiB vs f16's 1024 at
c=32768 on the 4B).

**Also recorded**: `AGENTS.md` gained the "RDNA first, other backends uninjured" scope policy (the F1
VEC arms stay as they are — AMD can't reach them, NVIDIA has its own maintainers) and the QSA-oracle
critical fact; `patches/README.md` gained the fourth-amendment section; `GREEDY-PURITY.md` §21 records
the shared-staging-tile rule and the instrument analysis; `TODO.md` marks the task done and lists
`iq4_nl` (F3 step 2), the tensor-tuned prefill crossover knob and the QSA fused-op probe as follow-ups.

## 2026-09-11 (8) — F3 step 1: `q4_1`/`q5_0`/`q5_1` become first-class KV cache types

**Canonical tip `6f07fe67a`** (block 08 `1a488fcf0`, block 14 `6f07fe67a`), net tree
`0c9dece6b0798e41360b8a8366187f38f37e1566`, 15 blocks, clean-apply strict 15/15 with 0 whitespace
warnings and the applied tree equal to the canonical one.  Two blocks amended: 08 (the FlashAttention
KV-type enablement) and 14 (the QSA-vs-KV-type arm + the tensor-split gate).

**Step 0 of the job was an instrument, not code.**  `--cache-type-k/v q4_1|q5_0|q5_1` were width-pure
and cheap (27B, ctx 204800: 1375/1512/1650 MiB vs 2337 `q8_0` / 4400 f16) but 3.4x slower prefill and
1.7x decode.  The `[FATPATH]`/`[FATTRACE]` trace settled the mechanism: f16/`q4_0`/`q8_0` take
`BEST_FATTN_KERNEL_TILE` at every width **with `need_f16_K/V = 1`** — i.e. the launcher stages f16
copies and the tile/mma families consume every type `ggml_get_to_fp16_cuda` covers — while `q4_1`
produced **no FA call at all**, because `ggml_cuda_fattn_kv_type_supported()` returned false and
`llama_context::resolve_fused_ops()`' FlashAttention probe then disabled FA for the whole context (the
non-FA attention path).  So the fix is not a new kernel: it is to let the FA path accept the types and
keep the vec family's instance list consistent.

**Block 08 (second 2026-09-11 amendment): the three types are enabled.**  `Q4_1`/`Q5_0`/`Q5_1` lose
their `#ifndef GGML_CUDA_FA_ALL_QUANTS` guard, the default vec dispatch gains the three diagonal cases,
and `ggml-{cuda,hip,musa}/CMakeLists.txt` gain the three diagonal instances (3 TUs).  `FA_ALL_QUANTS`
stays the knob for the 42 *mixed* `K != V` pairs; with it off the chooser still enforces `K == V`, so
the reachable pair set is exactly the diagonals and the predicate cannot disagree with the instances.
Measured (4B, 1 GPU, pp512/tg32): `q4_1` **2119.6/55.94 -> 7366.3/94.16** (+248 %/+68 %), on par with
`q4_0` (7376.0/93.9) and `q8_0` (7337.7/93.8); qwen4exp 3-GPU `-sm tensor` `q4_1` within 1 % of f16 at
every width (pp512 476.1, tg pl=1 40.60 / pl=4 126.44 / pl=8 176.16).

**Block 14 (third 2026-09-11 amendment): qwen4exp's QSA arm respects the KV type, and the tensor-split
gate is narrowed.**  Narrowing the gate alone was not enough — qwen4exp + a *quantized* KV cache +
`-sm tensor` **aborted** (`ggml-backend-meta.cpp:538`, `ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN`)
— and it aborted for **`q4_0` too**, which the delivery's own gate allowed: this is a pre-existing bug,
not a consequence of the enablement.  Instrumented, the op with the unknown split state is
`MUL name=attn_gated-<il>`, whose sources are the (mirrored) attention output and the hidden-split
attention gate.  The graph built `GGML_OP_FLASH_ATTN_QSA` for a cache type the fused QSA kernel cannot
read (`ggml_cuda_flash_attn_qsa_supported()`: f16/bf16/q8_0 only), so the op was never split and the
split states stopped agreeing.  `LLAMA_QSA_OFF=1` and `LLAMA_QSA_SPARSE_FA=0` both made it work; the fix
takes the dense masked path whenever the cache type is not QSA-native (`qsa_sparse` now also requires
f16/bf16/q8_0).  The tensor-split gate is narrowed to the types that really have a native FA read path
(`llama_kv_type_has_native_fa`, mirroring the backend predicate), which turns the pre-existing `q4_0`
abort into the clean error; the message lists the allowed set.  On dense models the newly enabled types
split fine (`27B` + `q4_1`/`q5_0`/`q5_1` + 3-GPU `-sm tensor` validated), and on qwen4exp only the
*fused sparse* prefill arm is given up for quantized caches — the decode band was already dense there
by arch policy, so its logits are unchanged (the `q4_1` probe hash is identical with and without
`LLAMA_QSA_SPARSE_FA=0`).

**Validation (gfx1201, per type).**  Width purity (probe, P=256, `W=1..8`, `CB=0`, `RS=0` and
`RS=from_w`) on 4B/1-GPU, 27B/`-sm layer`, 27B/`-sm tensor`, MoE-35B-A3B/1-GPU, gemma-4-E4B (SWA)/1-GPU
and qwen4exp/`-sm tensor`: **one hash per (model, split, RS)** for f16/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`q8_0`,
with every pre-existing value reproducing its recorded reference (`671d6096987470cb` 4B f16,
`31a0c1bace68e211` 4B q8_0, `619c151e48c76613` 4B q4_0, `4089b4d40b91090c` 27B layer f16,
`91434ea90f2cbfa0` 27B tensor f16, `d4156dbeb2252022` 27B tensor q8_0, `ac8825358d9adfda` MoE f16,
`dcf1ae667f730879` qwen4exp tensor f16).  Greedy purity: 27B and qwen4exp, `plain` ==
`--spec-draft-n-max 3` == `7` byte-identical for `q4_1`/`q5_0`/`q5_1` (qwen4exp `q4_1`
`42dfe66f25ed`, qwen4exp `q8_0` control `75d8530c5bb1` = the recorded item-1 value, 27B `q5_0`
`baca8ae6b30e`, 27B `q5_1` `675a1aa57b90`).  MTP gate: qwen4exp `q4_1` pos-1 acceptance **0.628**
(`q8_0` 0.700) with 61.2 t/s vs plain 45.9; 27B `q4_1` **0.893** (`q8_0` 0.962) with 85.7 vs 36.9 t/s.
`test-backend-ops -o FLASH_ATTN_EXT` **5599/5599** (up from 4591/4591 — the new pairs are now covered)
and `-o GATED_DELTA_NET` 4/4.  Coherence: 4B 3-GPU `-sm tensor` same-seed `1c5d32ac537d` on both the
canonical and the clean-apply sim build.  The block-15 beta patch was re-cut a sixth time on the new
tip (**base `6f07fe67a` -> beta commit `8c377b958`**, tree `34527a292`) — this re-cut is *not*
metadata-only: the merge threads the KV type into block 15's refactored `qwen4exp_qsa_sparse()` via new
`llama_cparams::type_k/type_v` fields, and it is a no-op for every validated beta config (f16/bf16/q8_0).

**Next (F3 step 2):** `iq4_nl` (same memory class as `q4_0`, but no V-side dequant at all in the FA
kernels — needs `dequantize_V_iq4_nl` + an instance + the vec/cross-product instance decision + the
same sweeps); see `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-f3-kv-diagonals.md`.

## 2026-09-11 (7) — the QSA decode arm and the MoE shared-expert epilogue are band-uniform

**Canonical tip `5ad11fd35`** (block 13 `ee6b7d53d`, block 14 `5ad11fd35`), net tree
`3e7accbd7f46c3d196e168a4d29a0350f813f5ff`, 15 blocks, clean-apply strict 15/15 with 0 whitespace
warnings.  Two width-dependences of the same shape as the F1/F2/HC fixes — a band gate written as
`n_tokens == 1` — closed in one session, each in its owning block.

**Block 13 (fourth amendment) — the MoE shared-expert epilogue serves the band.**
`ggml_cuda_op_shexp_down_gate` (the fused `down(swiglu) * sigmoid(gate(x)) + moe_out + ffn_residual`)
was gated `down_mm->src[1]->ne[1] == 1 && gate_mm->src[1]->ne[1] == 1` *because* its fused gate
reduction does not reproduce the standalone mmvq order — so `W=1` ran the fused epilogue and `W>=2`
the unfused chain: the last width-impurity in the MoE class (`W=1 ac8825358d9adfda` vs
`W>=2 bd138ad2326fbbf2`, 35B-A3B Q4_K_M).  The kernels are now token-generic (`shexp_gate_sigmoid`:
one warp per token; `shexp_down_gated_q8_0`: one block per `(row, token)`) with **`nwarps` pinned to
the single-token value** (`calc_nwarps` returns 4 for `ncols_dst 1..4` but 2 for `5..8`, and `nwarps`
sets `blocks_per_iter` = the reduction order), and the fusion arm accepts
`1 <= ne[1] <= MMVQ_MAX_BATCH_SIZE` (same width on both matmuls, contiguous epilogue operands).
Probe: `W = 1,2,3,4,8` all `ac8825358d9adfda`; kill-switch (`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`)
all `bd138ad2326fbbf2` (uniform unfused reference).  **MoE MTP improved**: 35B-A3B, 1 GPU, f16,
`n_max 3`, `n=96`: acceptance **0.81707** (was 0.51) with 167.3 t/s vs plain 96.9 (**+73 %**) — the
verify now uses the same epilogue arithmetic as the draft's single-token decode steps.  Cost: the
fused kernel re-reads the down weight row per token, so at the widest verify batches it loses a
little to the unfused chain (pl=8 332.1 vs 341.5, pl=4 252.0 vs 254.2); the decode win is kept
(pl=1 97.9 vs 98.3) and the fix (a column-blocked fused kernel that reads the weight row once per
`(row)` block) is a follow-up in `TODO.md`.  `patches/README.md` block-13 notes; `GREEDY-PURITY.md`
§17.

**Block 14 (second 2026-09-11 amendment) — the QSA decode arm serves the band.**  qwen4exp was still
not `plain == draft-mtp` in *text* (only ~100 of ~700 characters in common) even after the
hyper-connection band fix.  Localised to the QSA **indexer** arm choice: `LLAMA_QSA_OFF=1` is
byte-identical (`d4499ac8db72`) while `LLAMA_QSA_SPARSE_FA=0` is not, so the sparse-FA kernel is
exonerated; the single-step width probe is pure (it cannot reach the bug: `P <= 2048` keeps `n_kv`
below the selection width).  An arm trace (`build_layer_attn`): the middle arm — the arch policy's
dense decode arm — was gated `n_tokens == 1`, and with `width = indexer_top_k + r - 1 = 2051`
(`n_kv = 2304` at the first decode graph) `--spec-type none` took **arm 2 (dense)** while
`draft-mtp` (`n_tokens=4`) fell through to **arm 3 (sparse top-k selection)**; identical for the first
11 graph builds, split at the first decode graph.  Fix: `QSA_DECODE_BAND = 8` (the `n_max <= 7` purity
band), arm 2 takes `n_tokens <= QSA_DECODE_BAND`; prefill keeps the sparse selection (the policy
"prefill is untouched: QSA always").  Measured: `plain == n_max 3 == n_max 7` = `804de0576868`
(f16 KV, 704 chars) and `plain == n_max 3` = `75d8530c5bb1` (q8_0 KV, 660 chars); MTP `n_max 3` pos-1
acceptance 0.615 with 63.9 t/s vs plain 50.1 (**+28 %**), `n_max 7` pos-1 0.618.  The plain stream
moves with the fix (658 -> 704 chars) — the shared 4-token non-decode shape at `n_kv = 2304` also
moves to the dense arm.  Cost: the verify is now dense, slightly more attention work than the top-k
selection when the cache has just crossed the budget (pl=5 146.7 vs 149.5, pl=6 160.1 vs 162.5,
pl=1/2/4/7/8 flat or better).  Two width-dependences remain in the **sparse** regime and are recorded
in `GREEDY-PURITY.md` §18 + `TODO.md`: the fused indexer score's "byte-identical" claim is measurably
false and is itself `n_tokens == 1`-gated (unreachable on gfx1201 by default, but the default path on
gfx1151 above its 64K crossover), and a residual split survives even with one arm (706-character
common prefix instead of 100, then divergence).

Full gate set: probes on both platforms, `plain`/`n_max 3`/`n_max 7` text purity on f16 + q8_0 KV,
MTP acceptance/throughput gates (MoE + qwen4exp), `llama-batched-bench` pl=1..8 on both models,
`test-backend-ops -o GATED_DELTA_NET` and `-o FLASH_ATTN_EXT` (4/4 backends), 4B coherence smoke.
The block-15 beta patch was re-cut on the new tip on the new tip: base `5ad11fd35` -> beta commit `f3ece1e123905a98059025a7e7a3c7e8e28f54dc`, tree `5316920f130e585e23b9a38eef6e2c3c5940259e` (the `qwen4exp.cpp` hunk headers shift by +9 lines; `git am -3`/`git apply -3` resolves it cleanly, a plain `git am` does not).

Reverse-chronological log of every delivery-affecting change to the
**rdna-boosts 15-patch set** (block amendments, community-fix
integrations, re-baselines, regeneration + clean-apply re-verifications).
Newest entry first.  The README's
[Current state](README.md) section is a lean summary and points here
for the full record; per-block technical notes live in
`patches/README.md`, the verification contract in `MANIFESTS.md`.

---

## 2026-09-11 (6) — cause 3 localised: qwen4exp's plain-vs-spec gap is the QSA indexer path (not the FA kernel)

Follow-up measurement on the cause-3 item of entry (5).  All three runs are qwen4exp, 3-GPU
`-sm tensor`, f16 KV, `/tmp/prompt3k.txt` (~3.3k prompt), 128 greedy tokens, `--temp 0 --seed 42`,
`n_max 3` where applicable; the emitted text is extracted by backspace-stripping and hashing the
generation between the `> ` prompt echo and the `[ Prompt: ... ]` footer.

| run | plain (`--spec-type none`) | `draft-mtp --spec-draft-n-max 3` |
|---|---|---|
| default | `3ee9daee5c07` (658 chars) | `8a50ea24e8d5` (729) |
| `GGML_CUDA_GDN_CHUNKED=0` | `dad4f4442580` (721) | `9d29b773906f` (665) |
| **`LLAMA_QSA_OFF=1`** | **`d4499ac8db72` (711)** | **`d4499ac8db72` (711)** — identical |
| `LLAMA_QSA_SPARSE_FA=0` | `25f300a81b9e` (723) | `0d466b2dcf09` (721) |

* **The kill-switch that works is `LLAMA_QSA_OFF=1`**: plain == `draft-mtp` byte-identical, and the
  knob provably fires (the plain text moves `3ee9daee5c07` -> `d4499ac8db72`).
* **`LLAMA_QSA_SPARSE_FA=0` does *not* fix it** (two different texts, both moved — so the knob fired):
  the sparse-FA kernel (`fattn-qsa.cu`) is therefore **exonerated**, and the defect is in the rest of
  the QSA machinery — the **indexer/score** path (`indexer-topk.cu` plus the `qwen4exp.cpp` gates).
  `LLAMA_QSA_OFF=1`'s own comment says it "forces the dense no-indexer regime everywhere", which is
  exactly the part `LLAMA_QSA_SPARSE_FA=0` keeps.
* **The site class is cause 1's**: `src/models/qwen4exp.cpp:1094` gates the fused indexer score on
  `idx_score_fused && idx_key_float && n_tokens == 1 && ...` and `:1419` gates the early-decode dense
  shortcut on `qsa_dense_decode_until > 0 && n_tokens == 1 && n_kv < qsa_dense_decode_until` — so a
  1-token decode and an n-token verify batch take different QSA paths.  The single-step width probe is
  pure because it never reaches the sparse/indexer decode regime (its one decode step sits in the
  dense window).
* **It is not a prefill-state difference**: the divergence appears only after ~100 chars (~20 generated
  tokens) of the 3.3k-prompt run, i.e. the first steps agree (and `n_max 3` == `n_max 7` text is
  **identical** — `8a50ea24e8d5` — which is the cause-2 fix's win, since pre-fix they disagreed:
  `8a50ea24e8d5` vs `e6918a7af1f9`).
* The known **Issue #25 GDN chunked-prefill** item is a *separate* contributor, not this one: its
  kill-switch moves both texts (`3ee9daee5c07` -> `dad4f4442580`, `8a50ea24e8d5` -> `9d29b773906f`)
  without making them agree.  So cause 3 is **not** the GDN item and closing the GDN item will not
  close qwen4exp's plain-vs-spec gap.

**Consequence for the backlog:** cause 3 is a *small, well-scoped* fix in the established F2-cause-1
pattern (make the QSA decode band take one path for `n_tokens = 1..8`), with two identified sites and a
proven kill-switch — **not** a deep kernel issue.  Until it lands, `LLAMA_QSA_OFF=1` restores
`plain == draft-mtp` for qwen4exp byte-identically.

## 2026-09-11 (5) — F2 cause 2 FIXED: the MoE decode/verify band is band-uniform (block-13 amendment)

**qwen4exp is now width-pure `W = 1..8`**, so the designed `--spec-draft-n-max <= 7` verify batch is
bit-identical to the 1-token decode — the remaining *logit-level* condition for `plain == draft-mtp`.
Canonical tip **`bfaa83d8a`**, net tree **`4e5f2952f016f1ac160c53261f7b01d346322534`**; only
`ggml/src/ggml-cuda/mmvq.cu` changed (26 insertions / 11 deletions).

**Task 1 answered by measurement, and it moved the diagnosis.**  `[GD]` full-graph dumps show the graphs
are **identical** at every stage (2647/2404/2271/1863/1668/1565 nodes at both `W=4` and `W=5`), so the
previous entry's question ("fusion-applied vs graph-built-with-fewer-ops") is settled: the graph always
contains `MUL_MAT_ID(ffn_moe_gate)`, `MUL_MAT_ID(ffn_moe_up)`, `GLU(ffn_moe_swiglu)` at the same node
indices (`k=76/77/78`), and only the *fusion coverage* differs.  But the cause was **not** the
`mul_mat_id_glu_ops` fusion the previous entry blamed:

* `mul_mat_vec_q_moe`'s `__launch_bounds__` was `get_mmvq_mmid_max_batch_for_device<type>()*warp_size`
  — the upstream **per-type mmvq cap compiled into the kernel**, while the block is
  `(warp_size, ncols_dst)`.  Launching `IQ3_S` (cap 4) with `ncols_dst = 5` is 160 threads > the bound
  and dies with `ROCm error: unspecified launch failure`, so the cap is a *capability* limit, not just
  a heuristic.
* the same cap routes the upper band to MMQ: `ggml_cuda_mul_mat_id` takes `ne2 <= cap → mmvq` else
  `should_use_mmq → MMQ`, and `use_mmvq` (`ggml-cuda.cu:3730`) gates the `mul_mat_q_pair` fusion
  (which is what actually fired at `W = 5..7`).  mmvq (one warp per token, `mul_mat_vec_q_moe`) and
  MMQ reduce in different orders, so the band splits.
* the **UD-IQ4_XS quant mixes expert types per layer** — 47 layers `IQ3_S` gate/up (cap 4), layer 2
  `IQ4_XS` (cap 5), down `IQ4_NL`/`Q8_0` (cap 7) — which *predicts the census exactly*: fused layers
  48/48/48/48/1/0/0/0 for `W = 1..8` (measured `ffn_moe_up` MUL_MAT_ID counts 0/0/0/0/47/48/48).  That
  is the 4→5 and 5→6 boundary; the down's cap 7 is the 7→8 boundary.

**Fix.**  Block 13 already carries the invariant — `mul_mat_vec_q_switch_ncols_dst`'s `has_ids` branch
("this must cover `ncols_dst == 1` as well … the decode == verify invariant", added by block 13
2026-09-01) routes every `MUL_MAT_ID` to the column-generic MoE kernel.  The fix completes it for the
whole band:

1. `mmvq_mmid_max_batch_band(cap)` floors the per-type cap at `MMVQ_MAX_BATCH_SIZE` (the decode/verify
   band), applied to every AMD arch lookup, host *and* device;
2. `mul_mat_vec_q_moe`'s launch bound becomes `MMVQ_MAX_BATCH_SIZE*warp_size`, so the kernel can
   actually be launched across the band.

No other path changes: the caps' call sites are all `MUL_MAT_ID`-only, so dense models are untouched.

**Validation.**  `W = 1..8` all `3adeb313042a871b` (`-sm layer`) and `dcf1ae667f730879`
(`-sm tensor`) — i.e. every width equals that split's **pre-fix `W = 1` value**, so plain decode is
bit-unchanged and only `W = 5..8` moved onto it (the F1 "move the cheap side" pattern).  Also pure with
the state-sequence dimension exercised (`RS=0` and `RS=from_w`).  Controls: the **pre-fix vs post-fix
`plain` text is byte-identical** (`3ee9daee5c07`), and at the MTP gate config (`n_max 3` = `W=4`, a
no-op width) the runs are byte-identical: acceptance `0.76744` (66/86), generation 80.1 vs 80.0 t/s.
Text level: pre-fix `n_max 3` ≠ `n_max 7`; **with the fix they agree** (`8a50ea24e8d5`).

**Perf — the fix is a large win at the verify widths** (`llama-batched-bench`, fixed vs baseline
interleaved, swappable `libggml-hip.so`):

| model | batch 1 | 2 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| qwen4exp 3-GPU `-sm tensor` tg128 | 50.5 / 50.4 | 85.4 / 85.6 | 134.1 / 132.9 | **149.5 / 118.4 (+26 %)** | **162.5 / 130.6 (+24 %)** | **171.4 / 147.0 (+17 %)** | **178.0 / 155.4 (+14.5 %)** |
| 35B-A3B MoE 1 GPU tg128 | 98.3 / 98.1 | 156.4 / 156.2 | 254.2 / 254.1 | – | – | – | **341.3 / 289.9 (+17.8 %)** |
| 4B dense 1 GPU tg128 | 100.6 / 100.5 | 167.2 / 167.3 | 294.0 / 294.9 | – | – | – | 414.5 / 413.3 |

Widths inside the caps are unchanged (and bit-identical), dense is untouched, and MTP `n_max 7` goes
**41.8–42.5 vs 36.1 t/s (+16–18 %)** with acceptance `0.59375` vs `0.55556`.  The upstream per-type
mmvq caps were actively *costing* throughput on RDNA4 with the fork's mmvq + fused-GLU kernels.

**Cross-checks.**  `GATED_DELTA_NET` 4/4 backends OK; `FLASH_ATTN_EXT` 4/4 OK; MoE asterisk intact
(`ac8825358d9adfda` / `bd138ad2326fbbf2`); reserves unchanged (no allocation changes); clean-apply
simulation strict 15/15 with **0 whitespace warnings** and tree == canonical.

**Open — cause 3 (new, pre-existing, independent of cause 2).**  `plain` still differs from
`draft-mtp` text for qwen4exp even after the fix (`plain` `3ee9daee5c07` vs `n_max 3 == n_max 7`
`8a50ea24e8d5`), and the fix **cannot** be responsible: `n_max 3` uses `W = 4`, where the fix is a
verified no-op (bit-identical logits, byte-identical text, byte-identical acceptance).  Since the
single-step probe shows bit-identical logits for `W = 1..8` across both splits and the state-sequence
dimension, the divergence must be a **multi-step** effect — i.e. the speculative roll-back itself.
Prime suspect: the **masked (freed/stale) KV cells** written by rejected drafts, which block 14 keeps
at exactly `+0.0` in the HIP `fattn-tile`/`fattn-mma-f16` and Vulkan paths but **not** in qwen4exp's
**QSA sparse-attention path** (`fattn-qsa.cu`).  Next instrument: a multi-step probe (prefill P, then
feed a *fixed* token sequence, comparing the logits at each position between `W = 1` steps and `W = k`
chunks) — the single-step probe and `RS` dimension cannot see a cell that is only stale after a
roll-back.  **Corrected in the 2026-09-11 (6) entry: the cause-3 site is the QSA *indexer* machinery,
not the sparse-FA kernel, and the proven kill-switch is `LLAMA_QSA_OFF=1`.**

## 2026-09-11 (4) — F2 cause 2 localised: it is the MoE gate+up+GLU fusion flipping at `n_q = 5`, not a kernel-dispatch band

**Instrument.**  The per-node `[ND]` dump (`GGML_CUDA_NODE_DUMP=1`, re-appliable from
`archive/work/kv-quant-purity-followups/tools/node-dump-instrumentation.patch`) on qwen4exp, `-sm layer`,
P=256, RS=0, at W=4/5/6/7.

**Result — the executed-op census is the signature, and it is unambiguous:** only one op's count changes
across the whole band, and it changes at exactly the boundary:

| width | `ffn_moe_down` | `ffn_moe_up` | total nodes |
|---|---|---|---|
| `W=1..4` | 48 | **0** | 1920 |
| `W=5`   | 48 | **47** | 1967 |
| `W=6,7` | 48 | **48** | 1968 |

So the MoE **gate+up+GLU fusion** (`mul_mat_id_glu_ops = {MUL_MAT_ID, MUL_MAT_ID, GLU}`,
`ggml-cuda.cu:3324`, admitted via `ggml_cuda_should_fuse_mul_mat`) is applied for `n_q <= 4` and
abandoned from `n_q = 5`, and the fused GLU epilogue and the separate `MUL_MAT_ID` + `GLU` chain do not
sum identically — which is the impurity.  The `W=6`/`W=7` pair is a **perfect calibration** (`+0` nodes,
`0` differing ops) — that is *why* they hash identically, and it validates the census (the previous
session's node-dump diff was unusable because it had no such calibration, and because shape equality is
not sufficient: cache/state tensors legitimately differ with W).

**Refuted by measurement (the pre-HC-fix exclusion list was unreliable — the `W=1` vs `W>=2` break
dominated those hashes):** the block-13 `get_mmvq_mmid_max_batch` cap and its MMQ pair arm (forcing MMVQ
across the band via a temporary `GGML_CUDA_MOE_MMVQ_BAND=1` is **byte-identical**, and `should_use_mmq`
is false for `n_q <= 8`, so that arm never fires in the band); the MoE expert kernel (`mul_mat_vec_q_moe`
is **provably width-invariant** — `rpb` derives from `blocks_per_row_x`, a K property, and
`block_dims = (warp_size, ncols_dst)` is one warp per token); `LLAMA_QSA_OFF`,
`GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`, `GGML_CUDA_DISABLE_WEIGHTED_DOWN`,
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE` (all leave `W=5` = `c999233926f0`; positive control
`LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0` -> `bdaa8fc57381`, the recorded HC-off value, proving the
env plumbing); `ggml_cuda_should_use_mmvf(F32)` on gfx1201 = `ne11 <= 3` (a 3/4 boundary that does not
appear).

**Consequence for the brief:** cause 2 is a **fusion-coverage** band, not an `ncols_dst`/`ne11`
kernel-dispatch band — so it is *not* the same workstream as F3, and the fix is the F1/HC shape: keep the
gate+up+GLU fusion for the whole decode/verify band (`n_q <= 8`) rather than only `n_q <= 4`, measuring
the verify-throughput cost the way F1's was.  Next step: re-run the `GGML_CUDA_DISABLE_FUSION=1` width
matrix **post-HC-fix** (the earlier "survives all fusions disabled" observation predates it) to confirm
the unfused path is itself width-invariant.  Debug tooling to reuse: the node census above (nothing is
committed as code — it is the existing `[ND]` dump plus 30 lines of parsing), and the
`W=6` vs `W=7` calibration trick.

## 2026-09-11 (3) — Block 08 amended: the decode/verify band no longer spans two FlashAttention kernel families (F1 fixed)

**What changed.**  `ggml_cuda_get_best_fattn_kernel()` (`ggml/src/ggml-cuda/fattn.cu`) no longer returns
`BEST_FATTN_KERNEL_VEC` for small batches.  The fallback was upstream code (`11f0af550`, "for small
batch sizes the vector kernel may be preferable"): VEC for `n_q == 1` when `!gqa_opt_applies`, and for
`n_q <= 2` whenever K or V is quantized.  Both conditions are *always* inside the `n_q <= 8`
decode/verify band (prefill fell through to TILE anyway), so the branch only ever split the band; it is
deleted and the whole band uses TILE — the same shape of fix as the block-08 WMMA guard added
2026-08-29 (`Q->ne[1] > 8`) and block 00's `ntiles_dst_eff` in `launch_fattn`.

**Why.**  Measured with a new `GGML_CUDA_FA_TRACE` instrumentation (committed for reuse as
`archive/work/kv-quant-purity-followups/tools/fa-kernel-chooser-trace.patch`): with `q8_0` or `q4_0` K/V the
chooser returned **VEC (100) at `n_q = 1,2` and TILE (200) at `n_q >= 3`**; the two families order the
online-softmax/PV reduction differently, so token-0 logits at `W = 1,2` disagreed with every verify
width.  The launcher's own plan was *already* width-independent (`ntiles_dst_eff`, `parallel_blocks`
== `ntiles_KV` at every width), which is why the earlier F1 suspects (KV-type staging, the KV-cache
write path, `stream_k` rounding) all measured clean.

**Measured (3x gfx1201, ROCm 7.14, unpinned).**
- 4B Q8_0 `q8_0/q8_0` **1 GPU `W=1..8` all `31a0c1bace68`**, 2-GPU `-sm tensor` `abebfb93`, 3-GPU
  `-sm tensor` `7fe106f5`; `q4_0/q4_0` `619c151e48c7` / `240bc37d` / `483a850e` — all four split
  configs pure, and every value is that config's *previous verify* value (only `W=1,2` moved).
- 27B Q8_0 `q8_0/q8_0` 3-GPU tensor `W = 1,2,3,4,5,8` all `d4156dbeb225`.
- f16/bf16 configs byte-identical (they never took VEC): 4B f16 `671d60969874`, bf16 `b5d7e7b4`.
- text level, 27B 3-GPU tensor, ctx 8192, 300 greedy tokens, `q8_0` KV: plain == `n_max 3` ==
  `n_max 7` = `3537bc2b36be` (before: plain `73b2565bce47`/2810 chars vs verify `3537bc2b36be`/2801);
  f16 control `f32aac948600` for both.  **Harness note:** `llama-cli`'s `/\|` spinner is ``-based and
  timing-dependent and the banner embeds the build SHA — apply backspaces and strip both before
  hashing; three "divergences" this session were spinner noise.
- MTP: 27B `n=96` q8_0 KV acceptance **0.90789 (69/76), identical** to the pre-fix build.  MoE
  asterisk unchanged (`ac8825358d9adfda` / `bd138ad2326fbbf2`, and both `bd138ad2326fbbf2` with
  `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`).
- perf (llama-bench, q8_0 KV, interleaved, same binary): 4B pp512 7609.8 -> 7597.9 (-0.15% = noise),
  tg128 98.03 -> 97.11 (**-0.9%**); 27B 3-GPU tensor pp512 2257.3 -> 2253.2 (-0.2%), tg128 38.46 ->
  38.28 (**-0.5%**).  Reserves byte-identical (27B ub2048 q8_0: dev 1920.3284 / host 880.3360).
- op suites **with the fix active**: `test-backend-ops -o FLASH_ATTN_EXT` **4591/4591, 4/4 backends**;
  `-o GATED_DELTA_NET` 46/46, 2/2.  Quantized-KV coherence (the only configs that move) —
  gemma-4-E4B / 27B / qwen4exp with q8_0 KV: deterministic across runs and coherent.
- clean-apply: fresh `9113cc188` + `scripts/apply-all.sh` -> strict **15/15 `git am`**, **0 whitespace
  warnings**, applied tree == **`4104e7d34dd8cf9cb5488d46dcbba1b17eaa32d3`**.
- block-15 beta re-cut against the new base: **`0c8099ca2`**, tree **`7335b923d`** — metadata/offset
  only, 0 changed body lines (block 15's `fattn.cu` hunks sit at lines 166-326, the fix at ~690).

**New canonical tip `1bcf4e82d`**, tree `4104e7d34` (block 08 = `38cffdece`; blocks 09-14 got new SHAs,
bodies metadata-only — 2 lines each).  `rdna-boosts-all.patch` refreshed (95 files).

**This is NOT F2 cause 2.**  qwen4exp's `W >= 5` residual (`{1..4} {5} {6,7} {8}`) is **completely
unchanged** by this fix (W=1,4 `3adeb313042a`, W=5 `c999233926f0`, W=8 `c56ebb61963a`), which
**refutes the "F1 and F2 cause 2 share a cause" hypothesis** — cause 2 is not the kernel-family
chooser (it is a matmul/MoE dispatch band, still open).

**F3 refinement.**  The "slow pure" KV types are not missing a native kernel: they are rejected by
`ggml_cuda_fattn_kv_type_supported()` unless the build sets **`GGML_CUDA_FA_ALL_QUANTS`** (this build:
OFF), so `ggml_cuda_get_best_fattn_kernel()` returns `NONE` *before* the VEC/TILE choice (0 `[FATPATH]`
lines for `q4_1` vs 1+ for `q8_0`) and the attention takes the generic fallback — width-invariant by
construction (hence pure) and ~3.4x slower.  F3's first experiment is therefore a build-flag A/B.

## 2026-09-11 (2) — Block 14 amended: the fused hyper-connection ops serve the decode/verify band (qwen4exp width purity, cause 1 of 2)

**What changed.**  `ggml/src/ggml-cuda/hc-mix.cu` (`ggml_cuda_op_hc_mix`, `ggml_cuda_op_hc_combine`) and
the two graph gates in `src/models/qwen4exp.cpp` no longer require `nt == 1`: the fused
hyper-connection (HC) chain now serves the whole **decode/verify band `1 <= nt <= 8`**
(`HC_FUSED_MAX_TOKENS`, asserted in both ops).  The four mix kernels and the combine kernel take the
token index from `blockIdx.y` and offset every per-token pointer with the tensor's own stride
(`inject` is read with its view stride); at `nt == 1` every added term is zero, so the decode result is
unchanged (verified byte-identical for f16/bf16/q8_0/q4_0).  A `<= 8`-token **prefill** chunk also takes
the fused path — it cannot be told apart from a verify batch, and both must use the decode arithmetic;
wider chunks keep the unfused chain.  The ops are otherwise the same arithmetic, so no kernel numerics
were touched (the env fallback `LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0` reproduces the pre-fix
adaptive-MTP numbers exactly).

**Why.**  qwen4exp failed the decode==verify invariant ("F2"): a 1-token decode used the fused HC ops
while an n-token verify batch used the unfused chain, so the two computed the same position differently
and plain decode and `draft-mtp` disagreed.  Root-caused 2026-09-11 into **two stacked causes** (the
second is a `W >= 5` kernel-dispatch band shared with F1); this lands **cause 1**, as a block-14
amendment (block 14 introduced `hc-mix.cu` and the qwen4exp HC paths, so it owns them — the same
owner-based rule used for the block-02/12/13 amendments, not block 00).

**Measured** (3x gfx1201, ROCm `/opt/rocm-7.14-gfx1201`, unpinned):
- width probe, qwen4exp IQ4_XS f16 KV P=256 RS=0: `-sm layer` W=1..4 all **`3adeb313042a871b`** (was
  W=1 `3adeb313042a` + W=2..4 `044715b66e72f077`), `-sm tensor` W=1..4 all **`dcf1ae667f730879`**;
  **W=1 byte-identical to the pre-fix build on both splits and for every KV type** (f16
  `3adeb313042a`, bf16 `42e1bcfa57c1`, q8_0 `cb018394fd37`, q4_0 `688835658f30`).
- W=5 `c999233926f0` / W=6,7 `a8c532e12f9c` / W=8 `c56ebb61963a` (`-sm layer`) still grouped = **cause 2**,
  the `ncols_dst`/`ne11` selection band at `W >= 5`, shared with the `q8_0`/`q4_0` KV impurity (F1).
- greedy text: plain == `--spec-type draft-mtp --spec-draft-n-max 3`, byte-identical (3275 chars);
  `n_max 7` still differs (cause 2).  qwen4exp is therefore width-pure for **`n_max <= 3`**.
- adaptive-MTP (f16 KV, n=96): acceptance **0.50000 -> 0.76744**, MTP generation **63.3 -> 79.9 t/s**.
  With a `q8_0` KV cache: 0.50000 -> 0.43089 — that configuration is already width-impure via F1 (its
  W=1 decode is also unchanged), so it must be re-measured once F1 is fixed; recorded, not gated.
- perf: `-sm tensor` f16 pp512 1288-1300 (**parity**), tg128 48.30/48.72 (**decode unchanged** vs the
  pre-fix build, and the fusion's +14% over the unfused fallback 42.28/42.32 is kept).
- no regressions: 27B 1 GPU W=1/W=8 `4089b4d4`, W=9 `72af52db`; MoE W1 `ac8825358d9adfda` / W3
  `bd138ad2326fbbf2`; reserves byte-identical (qwen4exp ub2048 q8_0 dev 6690.3987 / host 1262.6954 /
  kvbuf 956.26; f16 6642.1331 / 1262.4297 / 1800.00); `test-backend-ops -o FLASH_ATTN_EXT` and
  `-o GATED_DELTA_NET` both 4/4 OK; `llama-batched-bench` B=1..8 clean.
- clean-apply: fresh `9113cc188` + `scripts/apply-all.sh` -> strict **15/15 `git am`**, **0 whitespace
  warnings**, applied tree == canonical **`e36263da57b8985cb98018af59fe639be0290dc4`**.
- the block-15 beta patch was re-cut against the new base (**beta tip `54859fdda`**, tree
  **`543ccc015`**, parent `1d8f53594`): metadata/offset-only, **0 changed body lines**.

**New canonical tip `1d8f53594`**, tree `e36263da5`.  Blocks 00-13 are byte-identical to the previous
regeneration; only `patches/0014-…` changed (the `From`/`index`/hunk-offset metadata plus the band fix).

**Follow-ups.**  Cause 2 (`W >= 5`) — fix together with F1/F3 (`archive/work/kv-quant-purity-followups/`);
the HC ops have no `test-backend-ops` coverage (a CUDA-vs-CPU band test would close that gap).

- **Block 15 beta revalidation (2026-09-11): re-cut against the current 15-patch delivery + full re-validation.**
  The Block-15 beta patch was cut on `b425aa8f7` (block 14 of the old **14-block** chain, block 13
  `e61676292`) — before block 00 existed and before the 2026-09-11 block-02/12/13 amendments — so it
  was re-cut against the current delivery (base `389c5341f`, tree `928852cdc`) and re-validated end to
  end.  Block 15 is still **staged in `archive/work/block-15-campaign-wins/`, NOT a delivery patch**; this is a
  beta-record update, not a delivery change (no `patches/` file and no block SHA moved).

  - **Re-cut**: new beta tip **`fe4f55278`** (tree `ffe197e2f`, parent `389c5341f`); the patch in the
    beta directory was replaced.  Measured dependency delta: **exactly one file** —
    `ggml/src/ggml-cuda/fattn-common.cuh` `7442bc22a` → `22eec7d57`, i.e. block 00's
    `ntiles_dst_eff` fix inside `launch_fattn`; the other 22 touched files are byte-identical to the
    cut base, so the re-cut changes only the `From` line, that one `index` line and one `@@` hunk
    header (+8 offset).
  - **Numbering correction**: the first draft of the revalidation plan claimed the beta patch had to be
    renumbered `[PATCH 15/15]` → `[PATCH 16/16]`.  That was **wrong**: `make-patches.sh` uses
    `git format-patch --start-number 0`, so the denominator is the *last block index* — the delivered
    15-patch set is `[PATCH 00/14]`…`[PATCH 14/14]` and block 15 is correctly `[PATCH 15/15]`.
    Verified by regenerating the 16-commit range with the same convention: all 15 delivery patch
    *bodies* byte-identical, block 15 emitted as `[PATCH 15/15]`.
  - **Clean-apply**: fresh `9113cc188` + `scripts/apply-all.sh` → strict **15/15**, **0 whitespace
    warnings**, tree `928852cdc`; + the re-cut block-15 patch → 16 commits, tree `ffe197e2f`.
  - **Result: every 2026-09-10 Block-15 claim reproduced.**  Reserves to the last decimal (27B
    `1920.3284/880.3360` → `1121.1252/81.1329`; 4B `1800.3284/840.3360` → `1001.1252/41.1329` →
    `257.1252/41.1329`; gemma-4-E4B/E4B-31B and the qwen4exp W1/W2/W3 chain incl. indexer KV
    `956.26` → `318.76` and the bf16/V5 table with bf16+V5 costing exactly f16); the 27B width-purity
    probe hashes **identical to the delivered reference** (`4089b4d4` / `a4817ee6` / `91434ea9`,
    `W=9` `72af52db`/`b059daa6`/`bc3faabd`) so `n_max <= 7` holds and block 15 changes no FA numerics;
    V4/V5 on == off **bit-identically** (the flagged `launch_fattn` risk is cleared); same-seed output
    byte-identical across gates on 4B/27B (short + 40k)/both SWA gemmas/qwen4exp; MoE asterisk intact
    (`ac8825358d9adfda`/`bd138ad2326fbbf2`, `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` → both
    `bd138ad2326fbbf2`); MTP 27B `0.90789` and MoE `0.58378` identical on both builds; op suites
    `FLASH_ATTN_EXT` **7859/7859 ROCm0 + 7859/7859 CPU** (6 derived), `GATED_DELTA_NET` OK,
    `test-alloc`/`test-batch-alloc` clean, W4 round trip 16.00 → 56.00 → 16.00 MiB; cost inside the
    documented envelope (4B prefill V3 −1.6 % / V4 −1.75 %, 27B V3 −0.3 %, decode flat; vs the
    delivery build 27B pp512 −1.7 % / tg128 flat, MoE pp512 −0.3 % / tg128 −1.25 %).  gfx1151 was
    **not** re-run (no such hardware on this host).
  - **Three PRE-EXISTING findings (not Block-15 regressions — identical hashes on the delivery build),
    now tracked in `archive/work/kv-quant-purity-followups/` + `TODO.md` + `GREEDY-PURITY.md` §12:** (F1) a
    **q8_0 or q4_0 K/V cache breaks the dense `n_max <= 7` purity guarantee** (`W=1 == W=2` then
    `W=3..8`; text-level plain `8ed58aa9` vs spec `da56855b` on the 27B) — the impure set is exactly
    the two types with a fast native both-quantized FA path; (F2) qwen4exp's fused sparse QSA path is
    not width-invariant; (F3) the sub-`q8_0` quants (q4_1/q5_0/q5_1/iq4_nl) are pure and 1800–2400 MiB
    but ~3.4x slower because they have no native FA path.  **Policy decided (maintainer
    2026-09-11): differing K/V cache types are rejected as an accepted limitation** (mixed pairs are
    1.7–3.6x slower than the same-type equivalent and never smaller; upstream #25871 already enforces
    same-K/V for DeepSeek V4).
  - Records updated: `archive/work/block-15-campaign-wins/{README,HANDOVER,BETA-TESTING}.md`,
    `GREEDY-PURITY.md` §12, `TODO.md`, `AGENTS.md`, `archive/work/kv-quant-purity-followups/`.

- **Block 13 amendment (2026-09-11, second): MoE `MUL_MAT_ID` decode/verify dispatch fix + the shared-expert fusion kill-switch.**
  Root-causes and closes the qwen35moe batch-width residual
  (`archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 2).  The residual was
  **not** in the MoE expert GEMM kernels.  A per-node, stride-aware dump of the
  decode graph that also covers fused-window destinations localised the first
  divergence to the **fused shared-expert window**
  (`ggml_cuda_op_shexp_down_gate`, gated `// decode only` on `ne[1] == 1`).  Two
  independent causes, both in that region:

  1. **`MUL_MAT_ID` never used the dedicated MoE kernel at `ncols_dst == 1`.**
     `mul_mat_vec_q_switch_ncols_dst` returned early only for `has_ids &&
     ncols_dst > 1`, so a single-token `MUL_MAT_ID` fell through to the **dense
     ksplit kernel with an ids gather** while a multi-token verify batch ran
     `mul_mat_vec_q_moe` -- two kernels, two accumulation orders, so a 1-token
     decode and an n-token verify batch of the same MoE matmul were not
     bit-identical.  The dense half of this was fixed earlier the same day (dense
     `MUL_MAT` rows always ksplit); the MMID half was still open
     ("`MUL_MAT_ID`/MoE keeps the item-split").  **Fixed**: route all `MUL_MAT_ID`
     through the MoE kernel -- it is column-generic (one warp per token column;
     `n_groups` and `warp_reduce_sum` depend only on `warp_size`), so decode and
     verify now share one path.  **+6.2% MoE decode** (tg128 95.62 -> 101.52),
     +1.4% pp512 (4790.6 -> 4858.6); dense 27B flat (tg128 31.95 -> 32.00,
     pp512 2021.9 -> 2033.5).
  2. **The fused shared-expert epilogue is not bit-exact with the unfused chain**:
     its gate dot uses `shexp_gate_sigmoid`'s own reduction order (not the
     standalone mmvq order), and its epilogue multiply was contracted into an FMA.
     The FMA is now removed (`__fmul_rn`, one rounding, matching the separate MUL
     kernel) -- necessary but not sufficient while the gate reduction differs.
     Making the whole window bit-exact needs the gate dot to reproduce
     `mul_mat_vec_q`'s order; scoped as future work.  The fusion is worth **+3.1%
     MoE decode** (101.5 vs 98.5 t/s), so it stays ON by default behind a
     first-class kill-switch: **`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`**.

  Verified: with the kill-switch set (plus fix 1) qwen35moe decode is
  **bit-identical** to the verify batch (`bd138ad2` both -- the W=3 value, so the
  decode path moves and the reference is preserved); by default both hashes are
  unchanged from the previous tip (no regression).  MoE MTP gate unchanged
  (acceptance 0.58378 = the canonical baseline exactly; per-pos 0.785/0.575/0.382;
  draft 130.9 vs plain 91.6 t/s).  Dense gates unchanged (`none == n_max 3 ==
  n_max 7` = `13acc229`; `n_max 8` still divergent = cause B).
  `test-backend-ops -o GATED_DELTA_NET` 2/2 OK.
  **Accepted residual**: by default the MoE decode/verify pair is still not
  byte-identical -- MoE is exempt from that gate by
  `benchmarks/mtp-adaptive-methodology.md` rule 3, and the gate it *is* held to
  passes.  Canonical chain re-cut: block 13 `c43beca1b` -> `855515420`, block 14
  `daf32f804` -> `389c5341f`, tip **`389c5341f`**, net tree **`928852cdc`**;
  clean-apply strict 15/15 `git am`, zero whitespace warnings, applied tree ==
  canonical.  All temporary instrumentation reverted.

- **Correction: `GGML_CUDA_ALLREDUCE=nccl` was never a bit-identical reference under `-sm tensor` (2026-09-11).**
  Several docs used "hybrid vs RCCL coherence IDENTICAL" as a validation gate
  (`AGENTS.md`'s verify recipe, `patches/README.md` block-13 note, `RUN.md`).
  It does not hold: the internal AR pipeline always BF16-round-trips
  (`GGML_CUDA_AR_BF16_THRESHOLD` defaults to 1) while the NCCL path reduces
  *small* tensors in FP32 ("Reduces as FP32 for small tensors and BF16 for
  large", `allreduce.cu`), so the two backends differ by design.  Measured
  2-GPU `-sm tensor`, 27B Q8_0, 300-token greedy `--spec-type none`:
  text `6e8ccd25` (hybrid) vs `6129e077` (nccl), and the token-0 logits differ
  in the decode/verify band (W=6: `a4817ee6` vs `73ff91bf`).  The gate only
  holds where no cross-device reduction happens at all -- 1 GPU and
  `-sm layer` both gave `8fd24746` under either backend, because the AR is
  never reached.  Past records that quote the gate (e.g. `BASELINE.md`'s dated
  validation lines) are left as written per the dated-record policy; they were
  probably true at the text level for the split mode used, or a near-tie
  collision.  Docs corrected: `AGENTS.md` (verify recipe -- now says
  "smoke comparison only", with the measurements), `patches/README.md`
  (block-13 note), `archive/work/sm-tensor-plain-vs-spec/RUN.md` (the gate list).
  Not a correctness bug: the default (hybrid) path is self-consistent, which is
  what the n_max sweep gates.  It is a reminder that **text equality is
  evidence for purity, never evidence against divergence** -- the same trap as
  the 3-GPU `n_max = 8` false negative recorded in the entry above.

- **Block 12 amendment: verification matrix completed, and the boundary is `W = 8` / `n_max = 7`, not `n_max = 8` (2026-09-11).**
  Completes the entry above with the full per-configuration probe matrix and a
  correction to how the boundary is established.  The verified guarantee is
  **`--spec-draft-n-max <= 7`** (an 8-token verify batch) on the dense 27B with
  MTP, for 1 GPU, 2-GPU `-sm layer`, 2-GPU `-sm tensor` and 3-GPU
  `-sm tensor` alike; the first violating depth is `n_max = 8` (a 9-token
  batch).  The target verifies the drafts *plus* the last committed token, so
  `K = n_max + 1` -- `n_max = 8` is a 9-token batch, one past the designed
  `Q->ne[1] > 8` limit (cause B).
  Raw-logit probe (27B Q8_0, `RS = 0`, P = 256), `W = 1..8` -> `W = 9`:
  `4089b4d4` -> `72af52db` (1 GPU); `4089b4d4` -> `72af52db` (2-GPU `-sm layer`);
  `a4817ee6` -> `b059daa6` (2-GPU `-sm tensor`); `91434ea9` -> `bc3faabd`
  (3-GPU `-sm tensor`).  Uniform: bit-identical through `W = 8` everywhere,
  divergent at `W = 9` everywhere.  1 GPU and `-sm layer` share a hash because
  layer splitting changes no kernel; tensor splitting is the only configuration
  with different numeric paths (and the only one cause A could affect).
  **Method correction:** the 3-GPU 300-token *text* gate at `n_max = 8` matched
  the plain run (`5037ef2e` both) even though the logits had already diverged --
  no greedy near-tie flipped inside that window.  Text equality is evidence for
  purity, never evidence against divergence; boundaries must be established with
  the probe.  (This is the near-tie rarity noted in
  `archive/work/sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`.)
  Cause B is left as-is by maintainer decision: correctness through `W = 8`
  is already far beyond what upstream delivers (upstream's CPU path diverges at
  the first width step, `W = 2`), and removing it would give up WMMA for
  9..N-token batches.  The gain from the block-12 fix (+12% MTP) is retained.

- **Block 12 amended: the hybrid all-reduce's size-based dispatch changed the reduction algorithm with the batch width (2026-09-11).**
  `ggml_backend_cuda_comm_is_small()` sent reductions below a per-device-count
  element count to the internal host-staged pipeline and everything above it to
  NCCL.  The two paths are **not bit-identical** (different summation order;
  the internal path always does the FP32->BF16 round-trip).  Under `-sm tensor`
  the reduced tensors scale with the batch width (`ne = ne0 * n_tokens`; ne0 =
  5120 on the 27B), so with the old 2-device value 32768 a **7-token**
  speculative verify batch (35840 elements) was reduced by NCCL while 1..6-token
  decode stayed on the internal pipeline: the same logical reduction, a
  different algorithm, purely because the batch got one token wider.  This -- not
  the GDN, and not MMVQ -- is what the earlier entries in this log called a
  pre-existing `n_max >= 6` divergence for 2-GPU `-sm tensor`.  (A second,
  *separate and deliberate* boundary remains at `W >= 9`: the FA launcher's
  tile-vs-WMMA switch at `Q->ne[1] > 8`, which caps the guaranteed range at
  `n_max <= 7` by design.  `GREEDY-PURITY.md` section 11 now states both causes
  and the per-configuration ranges.)
  **Fix:** raise the 2-device crossover 32768 -> 131072 (the 3-device value).
  The largest verify batch (`--spec-draft-n-max 16` -> 17 tokens = 87040
  elements) stays well under it, and still far below the internal pipeline's own
  1 MB (262144 element) cap, so nothing is pushed off the fast path.  Only
  7..25-token tensors change path; one-token decode and prefill (>25 tokens) are
  untouched.
  **Evidence** (27B Q8_0, 2-GPU tensor, `-ts 1/1`, probe/`llama-cli`):
  `GGML_CUDA_ALLREDUCE=internal` (one algorithm for every size) makes probe
  `W = 1/6/7/8` all `a4817ee6`; the fix does the same, with `W <= 6` keeping
  their previous hash, i.e. plain decode is bit-unchanged and only `W = 7,8`
  move onto the internal pipeline.  Text `none == n4 == n6 == n7` (`6e8ccd25`;
  previously pure only to `n_max 4`/5).  1-GPU `W=1 == W=8`; 3-GPU text
  `none == n6`; GDN K-independence `RS=6 W=6 == RS=0 W=1`; determinism `W=7`
  twice identical.
  **Perf is a win on both axes:** MTP `n_max 6` acceptance 0.509 -> 0.533 and
  63.6 -> 71.3 t/s (+12.1%); `n_max 12` 51.7 -> 58.0 t/s (+12.2%); llama-bench
  pp512 2009.08 -> 2004.12, pp4096 1905.00 -> 1907.41, tg128 31.92 -> 31.97
  (all unchanged within noise).
  **Localisation method** (temporary instrumentation, since reverted): a
  backend-side per-node digest dump in `ggml_backend_cuda_graph_compute`, gated
  on a phase file, reading each tensor through its own `nb[]` strides after a
  device sync.  `cb_eval` is unusable for this (it changes MoE numerics and
  aborts in the meta backend under tensor split), and `ggml_backend_tensor_get`
  flattens from the view's base pointer ignoring `nb[]`, which alone produced a
  field of false positives.  The dump showed the layer-0 GDN chain bit-exact and
  the first divergence exactly at the first cross-device reduction
  (`attn_residual-0`), whose buffer the meta backend rewrites in place between
  the producing MUL_MAT and its consumer.
  Canonical: block 12 `eec68b2ad` -> `cac14423e` (blocks 13/14 re-cut: `070096e16`
  -> `c43beca1b`, `30d119ea9` -> `daf32f804`), tip `daf32f804`, net tree
  `10f94d635`; clean-apply strict 15/15 `git am`, zero whitespace, applied tree
  == canonical; blocks 00-11 bodies byte-identical.

- **Block 02 amended (final form): the whole-batch K-independent chunked GDN prefill — free, no tail, no gate (2026-09-11).**
  Follows the KTAIL=16 entry below, which it supersedes.  Option B from
  `archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md`: instead of *sharing a
  sequential tail* between the plain (`K == 1`) and MTP (`K > 1`) prefills, both
  paths now make the **same call** — a batch with more than `max(K, 16)` tokens
  is chunked **whole**, exactly what `K == 1` already did, and anything smaller
  stays on the sequential kernel.  No tail, so the previous -0.3..-0.8 % tail
  cost goes to **zero**: 27B Q8_0 1 GPU pp512/2048/4096 = 1385.3/1356.4/1328.2
  vs 1384.7/1355.0/1327.8 for the old K-dependent boundary (parity), tg
  unchanged.  A batch larger than `max(K, 16)` cannot be a verify batch (those
  decode `<= K` tokens) and is never rolled back into, so its K snapshots are
  skipped; every verify batch keeps them.
  **Guard added** (the invariant is empirical): `llama_memory_recurrent::seq_rm`
  tracks the last batch's per-seq token count and logs a once-only warning if a
  rollback ever crosses that boundary.  Measured: 449 rollbacks over llama-cli
  `draft-mtp` n_max 1/4/8/16 + 20 in llama-server `--cache-reuse`, all preceded
  by a `<= K`-token batch, 0 warnings.
  **Removed**: `GGML_CUDA_GDN_ALIGN_BOUNDARY`, the `align_boundary` variable and
  both K-dependent branches (~118 lines) — they were unreachable with the gate
  ON, and the opt-out is superseded by `GGML_CUDA_GDN_CHUNKED=0`, which is
  *both* correct (all snapshots written) and bit-identical plain-vs-MTP.
  **Gate re-verification** (all corrected against the new build): 27B 2-GPU
  tensor `none == n1 == n4 == n5` (`6e8ccd25`), 3-GPU tensor `none == n4`,
  1-GPU 4B probe `671d6096`, 27B prefill probe `W = 1/3/5/6` all `a4817ee6`
  with `RS=6 W=6 == RS=0 W=1` (prefill now K-independent), `test-backend-ops -o
  GATED_DELTA_NET` OK, 3x determinism check identical.
  **Correction (important):** the `n_max <= 15` purity claim in the entries
  below — and in every doc — was **wrong**; it was never validated past
  `n_max = 4`.  The real `none == draft-mtp` range is **`n_max <= 5`**, and the
  cause is a **pre-existing** multi-token-verify-batch MUL_MAT dispatch
  difference (identical divergence pattern on the delivered KTAIL=16 build;
  pure at `RS=0` up to `W = 6`, breaks at `W = 7`, again at `W >= 9` where MMVQ
  hands over to MMQ).  Upstream master is affected too and *worse*: on upstream
  `9cf3bf256` (CPU, 4B) `W = 1` already differs from `W >= 2`.  Docs corrected
  (`GREEDY-PURITY.md` §11, `benchmarks/mtp-adaptive-methodology.md` rule 4,
  `patches/README.md`, `AGENTS.md`); root-causing it is
  `archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.
  Canonical re-cut: block 02 `63f8ab023` -> `6e81ed5ed`, tip **`30d119ea9`**,
  net tree **`29714ad1f`**; clean-apply strict 15/15 `git am`, zero whitespace,
  applied tree == canonical.

- **Block 02 amended: GDN alignment tail shortened to `KTAIL=16` (2026-09-11). — SUPERSEDED by the entry above.**  The aligned
  boundary's cost is entirely its sequential tail (which writes the K rollback snapshots), and the
  tail was 64 — ~16x longer than the default `--spec-draft-n-max 3` needs.  `KTAIL=16` covers
  `K <= 16` / `n_max <= 15`, including adaptive MTP's recommended `n_max = 12`; for deeper drafts the
  new `K > 16 ? K : 16` floor keeps the snapshots exact (those reproduce the pre-alignment `K > 1`
  boundary, i.e. correct-but-not-bit-identical, instead of reading stale snapshot slots).  Measured
  (27B Q8_0, 1 GPU, interleaved `-r 5`): `KTAIL=64` ≈ -1.5 %, **`KTAIL=16` ≈ -0.3..-0.8 %**,
  `KTAIL=8` ≈ 0; decode unchanged.  Bit-identity re-verified with `KTAIL=16`: 27B 2-GPU tensor
  probe W=1/3/5 and text `none == n1 == n2 == n4` (`e386b50d`), 3-GPU tensor `none == n4`, 4B 1-GPU.
  Canonical re-cut: block 02 `d60bb52ef` -> `63f8ab023`, tip **`30d119ea9`**, net tree
  **`29714ad1f`**; clean-apply strict 15/15 `git am`, zero whitespace, applied tree == canonical.
  Follow-ups (free GDN prefill alignment + the MoE batch-width residual) written up in
  `archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md`.

- **Block 02 amended: `GGML_CUDA_GDN_ALIGN_BOUNDARY` flipped to default ON (opt-out), 2026-09-11.**
  The K-independent chunked-GDN boundary is now enabled by default (`GGML_CUDA_GDN_ALIGN_BOUNDARY=0`
  opts out and restores the K-dependent boundary).  This is the second of the two independent fixes
  required for `--spec-type none == draft-mtp`: with the block-13 dense-MMVQ alignment in place, the
  default is `none == n1 == n2 == n4` on 27B 2-GPU tensor (`5037ef2e`), 3-GPU tensor (`f60b79d0`),
  2-GPU layer and 1-GPU (`7d566fee`), and on the 4B 1-GPU probe.  Cost of the default
  (27B Q8_0, 1 GPU, llama-bench, `-r 5`, two alternating runs): pp512 1393.7/1384.5 ->
  1363.6/1363.8 (**-1.8 / -1.5 %**), pp2048 1362.2/1358.3 -> 1337.3/1338.1 (-1.8 / -1.5 %),
  pp4096 1329.5/1328.5 -> 1309.2/1309.6 (-1.5 %); decode unchanged (tg128 20.43 -> 20.40).  The
  maintainer accepted the prefill cost to close the divergence.  Canonical chain re-cut: block 02
  `38641280b` -> `d60bb52ef`, tip **`33ccf7e28`**, net tree **`31e153fe3`** (later re-cut again for KTAIL=16:
tip `30d119ea9`, tree `29714ad1f`); clean-apply strict 15/15
  `git am`, zero whitespace warnings, applied tree == canonical.

- **Block 13 amended: dense decode/verify MMVQ kernel alignment (`mmvq.cu`, 2026-09-11).**
  Closes the remaining batch-width half of the `-sm tensor` plain-vs-spec divergence.  Root cause:
  the block-13 `ncols_dst == 1` dispatch kept **dense** rows with `K < 4096` on the item-split/rpb
  kernel while the ncols 2..8 dispatch (and `K >= 4096` ncols==1) unconditionally use the ksplit
  kernel; the two accumulate K in different orders, so a single-token dense `MUL_MAT` is not
  row-identical to the same row in a 2..8-token verify batch (~1e-6 at the first divergent
  projection, amplified by the recurrent GDN).  Visible as `--spec-type none` != `draft-mtp` on
  small dense models (`n_embd < 4096`, e.g. Qwen3.5-4B, the cheap 1-GPU repro) and under
  `-sm tensor` on any model whose per-GPU K shard drops below 4096 (Qwen3.8-27B 5120 -> 2560).
  Fix: `!has_ids || ncols_x >= 4096` — dense rows always ksplit; `MUL_MAT_ID`/MoE keeps the
  item-split (its multi-token kernel is `mul_mat_vec_q_moe`).  Verified per-process (token-0 logit
  hash, callback-free): 4B 1-GPU and 27B 2-GPU-tensor W=1/3/5 bit-identical (was 0.133 on the
  27B).  Perf neutral (4B/27B/MoE-A3B within noise, MoE tg128 95.66 -> 96.02); MTP gates
  0.487/36.5 (dense) and 0.675/153.1 (MoE); `GATED_DELTA_NET` 46/46; hybrid-vs-NCCL coherence
  identical.  **The default-config `-sm tensor` text equality additionally requires the block-02
  `GGML_CUDA_GDN_ALIGN_BOUNDARY=1` gate** (K-dependent chunked-GDN prefill boundary); that gate
  stays opt-in because it costs ~2-2.6% prefill.  Canonical chain re-cut: block 13
  `fc7f52f96` -> `029b07b30`, tip **`27bd754b6`**, net tree **`c0775c33c`**; clean-apply strict
  15/15 `git am`, zero whitespace warnings, applied tree == canonical.  Record:
  `archive/work/sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`; block-13 notes in `patches/README.md`.

- **Block 02 amended: opt-in K-independent chunked-GDN boundary (`GGML_CUDA_GDN_ALIGN_BOUNDARY=1`, 2026-09-11).**
  Fixes the fork-only plain-vs-spec divergence found during the gfx1151 issue-#25 validation (the issue
  #25 *follow-up*): the chunked GDN prefill had a **K-dependent** chunk/sequential boundary (plain
  `K == 1` chunked the whole prompt; MTP `K == n_max + 1` chunked `n_tokens - K` + a K-token tail), so
  the post-prefill SSM state depended on `n_rs_seq` and `--spec-type none` disagreed with `draft-mtp`
  (greedy near-ties flipped).  The amendment adds a gated third branch that chunks `n_tokens - 64` and
  runs the sequential kernel over the last 64 for both `K == 1` and `K > 1`, giving one boundary and one
  state; the tail also emits the K snapshots (rollback <= 63 exact), and `n_seqs > 1` keeps the old
  whole-ubatch path.  **Default OFF** — the fork's existing boundary is deliberate and ~1.1-1.2 % faster
  prefill; the gate only guards the two existing branch conditions, so the default output is
  **byte-identical** (`d9bf6850`), while with the gate on `none == n2 == n4` (`1a9ef0a1`, which also
  equals the `GGML_CUDA_GDN_CHUNKED=0` reference on the short prompts).  gfx1201 probe (`RS=from_w`,
  P=256): `W1-W3/W3-W5 = 0.136693/0.182106` default (unchanged) -> `0.000000/0.000000` gated;
  `test-backend-ops -o GATED_DELTA_NET` 46/46 in default, gated and gated+fp32.  Record:
  `archive/work/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FIX.md`.  Canonical fork rebuilt at `9113cc188`,
  block 02 (`5cbfbafd9` -> `38641280b`) amended by rebase, new tip **`7b79930b2`**, net tree
  `fcf3e4bb7`; clean-apply **strict 15/15 `git am`**, zero whitespace warnings, applied tree ==
  canonical.  A separate `-sm tensor` (2/3-GPU) plain-vs-spec divergence — independent of GDN and of
  this gate — is documented there as an open follow-up (the server's 3-GPU tensor-split config is
  affected).

- **Block 00 (structural and architecture fixes) added; the set is now 15 patches and the masked-V
  freed-cell fixes are re-homed (2026-09-10).**  A new first block, `patches/0000`, holds baseline-level
  fixes every later block builds on:
  1. **FA small-batch KV-split width invariance (issue #25).**  `launch_fattn`'s non-stream-K
     `parallel_blocks` heuristic keys off `ntiles_dst`, which is a function of `Q->ne[1]`, so
     single-token decode (`n_q = 1`) and speculative verify batches (`n_q = 3`, `5`, …) chose different
     KV splits, fed different partial sums into the online-softmax/PV combine and produced different
     logits; greedy near-ties then flipped, so MTP `--spec-draft-n-max 2` and `4` streamed apart.  The
     heuristic now evaluates `ntiles_dst` as if `n_q == 1` for every `n_q <= 8` (prefill unchanged).
  2. **Vulkan masked-V / freed-cell fixes** (`flash_attn_cm1.comp`, `flash_attn.comp`): dead columns
     never read V.  These are baseline shaders, so they belong in the structural block.
  The **HIP** masked-V fixes do **not** belong in block 00: the `fattn-tile.cuh` half uses the native
  bf16 PV staging (`V_k0`/`KQ_k`/`nv_bfloat162`) that **block 03** introduces, and the
  `fattn-mma-f16.cuh` half fixes the same class of leak on that path — so, per the maintainer, both HIP
  halves were **moved into block 03** (the earliest block that exercises the leaking code).  Block 14 no
  longer carries any masked-V/freed-cell hunk.  The net tree is unchanged from the previous regeneration
  (`26690e4d9`).  The block-15 attention-memory campaign is unaffected and remains staged in
  `archive/work/block-15-campaign-wins/`.
  Layout: `0000` = block 00, `0001`–`0014` = the old blocks 01–14 (renumbered by
  `git format-patch --start-number 0`, so the file prefix still equals the block number; the subjects
  read `[PATCH 00/14]`…`[PATCH 14/14]`).  Canonical fork rebuilt at `9113cc188`, tip **`505637d6e`**;
  `scripts/apply-all.sh` and `scripts/make-patches.sh` updated (15 blocks, `0000` included);
  `rdna-boosts-all.patch` regenerated.
  Validation (3× gfx1201, ROCm 7.14): clean-apply sim → strict **15/15 `git am`, zero whitespace
  warnings**, applied tree `26690e4d9` == canonical; issue #25 → `--spec-draft-n-max 2 == 4` on 2-GPU
  p0/p2/p3 and 3-GPU p0, `draft-mtp-adaptive` == both; plain decode (`--spec-type none`) byte-identical
  to the pre-block-00 canonical on 2-GPU and 1-GPU; MTP acceptance gate holds (dense 0.479, MoE 0.669,
  MTP >> plain both).  A `structural-fixes` branch (block 00 + blocks 01–14, based directly on
  `9113cc188` = the fork's master) was pushed to the personal fork for the gfx1151 investigation; the
  upstream-PR candidate `upstream/UPSTREAM-PR-fa-kv-split-width.{patch,md}` was filed under `upstream/`.

- **Block 15 un-promoted from the delivery — it belongs only in `archive/work/block-15-campaign-wins/`
  (2026-09-10).**  Block 15 was promoted into `patches/0015` by mistake; the maintainer never
  approved cutting it as a delivery patch.  The delivery is a **14-patch set** again
  (`patches/0001`-`0014`, canonical tip `ff2b35f49`), `scripts/apply-all.sh` and
  `scripts/make-patches.sh` are back to 14 blocks, `rdna-boosts-all.patch` is the 14-block net,
  and the docs/headers no longer present Block 15 as delivered.  The block-15 work (including
  the 2026-09-10 V5 and RDNA3_5/gfx1151 amendments) continues to live only in
  `archive/work/block-15-campaign-wins/block-15-campaign-wins.patch` and is applied manually on top of
  the 14-block tree, pending the maintainer's promotion go-ahead.  The `0001`-`0014` bodies are
  unchanged from the promoted set; only the `From <sha>` line and the `[PATCH NN/15]` →
  `[PATCH NN/14]` series count differ.  Clean-apply sim: fresh worktree at `9113cc188` +
  `apply-all.sh` → strict 14/14 `git am`, zero whitespace warnings, applied tree `6ce36849` ==
  the canonical 14-block tree.  (The dated entries below that say "cut" / "15-patch" record the
  promotion as it happened; this entry reverses it.)

- **RDNA3_5 (gfx1151) validation of the 14-block delivery + the beta block-15 patch; V3 iGPU enablement + multi-stream
  guard folded into the beta block-15 patch (2026-09-10, single Strix Halo, ROCm 7.14).**  The first
  single-device iGPU run of the delivery (Radeon 8060S, `VMM: no`, 1 device).  Block-14
  masked-V fixes, V3 derived mask, V4 native q8_0 and V5 native bf16 were exercised with
  a BF16 KV cache in both arm states, per the sign-leak campaign matrix.  Two V3
  regressions found and fixed as a dated amendment to the **beta** block-15 patch (the delivery
  stays 14 patches; beta patch tip `377f8e790`):
  1. the derived-mask probe rejected `GGML_BACKEND_DEVICE_TYPE_IGPU`, so V3 was silently
     disabled on the HIP iGPU and its ~800 MiB compute + ~800 MiB host win was lost;
     `ggml_backend_dev_is_cuda()` / `ggml_backend_dev_implements_kq_derived()` now accept
     `IGPU` (ROCm/CUDA reg name still required);
  2. `n_seq_max > 1` aborted context creation in `ggml_flash_attn_ext_add_kq_derived`
     (`GGML_ASSERT(tok_lo->ne[0] == a->src[0]->ne[1])`): `build_attn_mha` derives the
     stream count from `k->ne[3]` (the cache's `n_stream` == `n_seq_max`), while
     `kq_mask_derivable()` only checked `ubatch.n_seqs_unq`; it now rejects
     `n_stream != 1`, so a multi-slot context keeps the packed mask (no abort) and a
     single-stream context keeps the win.  `llama-server --parallel 4`, which aborted on
     the pre-amendment tree, now serves and passes the 16-run gate.
  Validation on the amended tree: reserves reproduce the RDNA4 block-15 numbers exactly
  (4B ctx 204800/ub 2048 V3 −799.20 compute / −799.21 host, V5 bf16 968.86 → 256.86,
  V4 q8_0 1001.13 → 257.13; 27B f16/bf16/q8_0 488.86 / 1072.86→488.86 /
  1121.13→489.13; Flash-Next W on 3251.39/63.69 indexer 318.76, W off 6690.40/1262.70
  indexer 956.26).  Determinism: 14 ROCm + 7 Vulkan gate runs PASS 16/16, V3 on vs off
  byte-identical over 7 × 2064 cells, bf16 arm on/off (V5), q8_0 arm on/off (V4) and
  q4_0 arm on/off byte-identical; bf16 vs f16 differs only by cache precision.  Isolated
  probes clean in both arms on both backends (ROCm bf16 34/34, f16 36/36, Vulkan
  bf16/f16 36/36; only the documented deterministic live-cell bf16 diag ≤1.1e-13).
  `test-backend-ops` FLASH_ATTN_EXT 4596/4596 ROCm0 (FA_ALL_QUANTS=OFF; 5 derived cases
  OK) + 7859/7859 CPU, `test-alloc`/`test-batch-alloc` pass, W4 repro 16.00 MiB.  MTP
  acceptance identical V3 on/off and arm on/off (27B 0.79762, Flash-Next draft 0.52727).
  Arm cost on gfx1151 is *lower* than RDNA4 — V5 −0.4…−0.9 % prefill, V4 **+2.6 %** at
  pp20480, decode ±0.1 % (the large MALL absorbs the interleaved-view re-reads); V3
  ~−3.2 % pp20480.  Clean-apply sim: fresh worktree at `9113cc188` + `apply-all.sh` →
  strict 15/15 `git am`, zero whitespace warnings, applied tree `6f5d23b5` == amended
  canonical.  Block-13 fused MoE re-check on Q3_K_M: the isolated
  `GGML_CUDA_DISABLE_MOE_MMQ_FUSION` delta is ~0 on this build (fusion fires, coherence
  holds, decode untouched) — absolute prefill is ~10–13 % above the 2026-09-05 record,
  consistent with the 2026-09-06 model-neutral Strix folds capturing the same work.
  Raw matrix: `archive/work/strix-halo/GATE-2026-09-10-block15-rdna35.md`.

- **V5 native bf16 K/V folded into Block 15 (opt-in, same switch as V4) — D12 closed
  (2026-09-10).**  The bf16 lever is implemented, validated and packaged as a **dated
  amendment to block 15** (`patches/0015`, canonical tip `f5ab5350b` on `9113cc188`;
  the amendment touched `0015` only — `0001`-`0014` stayed byte-identical).  A bf16
  KV cache no longer needs the F16 staging scratch: bf16 and f16 tiles have the same
  byte layout, so the MMA loader converts each 16-byte staged chunk in registers
  (`__float22half2_rn(ggml_cuda_cast<float2>(bf16x2))`, bit-identical to the
  launcher's `ggml_get_to_fp16_cuda(GGML_TYPE_BF16)`) instead of copying from the
  scratch, and the scratch sizing + whole-cache conversion pass are skipped for that
  operand.  The per-operand staging source is now one shared type code
  (`fattn_kv_native_t{FATTN_KV_NATIVE_NONE,Q8_0,BF16}`, subsuming V4's flags), so the
  launcher, `get_alloc_size` and the kernels cannot disagree.  Scope per D10: the F16
  fragments/`cp_async` design is untouched (no bf16 WMMA fragments).
  **Measured (ctx 204800, bf16 KV, arm on vs off):** 4B ub 2048 968.86 -> **256.86**
  MiB/GPU (== the f16 cache; ub 1024 884.82 -> 128.82, ub 512 842.80 -> 64.80), 27B
  1072.86 -> **488.86** (ub 512 868.80 -> 122.80), gemma-4-E4B 1062.89 -> **404.89**,
  gemma-4-31B 2068.89 -> **716.89**; qwen4exp unchanged (f16 == bf16 == on/off there,
  its FA path never staged bf16) and its q8_0 control reproduced 3251.39/63.69
  exactly, confirming the refactor left V4 alone; ub 8 (TILE/verify) 8.09 either way.
  **Cost** (interleaved same-binary A/B, off -> on, bf16): 4B -0.22 % (pp2048),
  +0.27 % (8192), -1.06 % (20480), -2.36 % (40960); 27B -0.76 % (20480); decode
  within 0.1 %.  The conversion itself is free (native bf16 staging is within 0.17 %
  of an *f16* cache) — the loss is the removed scratch, which is a dense, normalised
  copy of the cache view (the GQA heads are interleaved: `nb[1]` is 2048 B for a
  512 B row on the 4B), while the native path re-reads the interleaved view on every
  staging pass.  **Decision: opt-in via `GGML_CUDA_FA_KV_NATIVE` (default 0), i.e.
  the maintainer's explicit instruction for this item ("treat it similarly to V4 ...
  gated by the same environment variable"), consistent with D9.**
  **Gates:** same-seed text byte-identical (arm on vs off vs f16) on 4B, 27B,
  gemma-4-E4B (ISWA) and gemma-4-31B (ISWA), short + 3k/40k prompts, with V3's
  derived mask active (wins additive: -799.2 derived mask, -712.0 bf16 scratch on
  the 4B); MTP 27B 0.82716 and qwen4exp 0.44262 identical on/off (q8_0 references
  0.76744/0.44262 unchanged); `test-backend-ops` FLASH_ATTN_EXT 7859/7859 ROCm0+CPU,
  with the 2704 bf16 K/V cases (all head sizes incl. the 576/512 MLA
  `v_is_view_of_k` layout) and 365 q8_0 cases green in both arm states, identical case
  lists.  Re-validated end to end **from the delivered patches**: fresh worktree at
  `9113cc188` -> `apply-all.sh` strict 15/15 `git am`, tree identical to `f5ab5350b`,
  build, reserves/coherence/MTP/op-suite all reproduced.  One pre-existing
  unrelated full-build warning recorded in `TODO.md`
  (`llama-kv-cache.h:274` `-Wunused-private-field` for W3's `v_enabled`).
  Records: the V5 amendment section in `patches/README.md`, the outcome section in
  `archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`, the block-15 beta record.

- **bf16-native MMA K/V planned as the next essential follow-up (D12); two pre-existing findings
  recorded (2026-09-10).**  With Block 15 cut, the maintainer picked the bf16 lever as the one
  follow-up.  The executable plan is **`archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`**:
  measured before-state in the *delivered* tree, the mechanism with exact call sites, the design
  (keep the F16 fragments, `cp_async` the raw bf16 bytes into the same shared offsets — a 16-byte
  chunk is 8 elements either way — then convert the tile in place), the validation protocol and a
  three-way ship rule (expectation: **on by default**, unlike V4, because the `cp_async` pipeline is
  kept).  Also `HANDOVER.md` D11/D12 and the §8 prompt.
  **Before-state (ctx 204800, V3 on, f16 = reference):** 4B (1 GPU) ub 2048 256.86 -> **968.86**
  (+712.00), ub 1024 +756.00, ub 512 +778.00; 27B (3-GPU Meta) ub 2048 488.86 -> **1072.86** (+584.00),
  ub 512 +746.00; ub 8 (TILE/verify) **identical** at 8.09 MiB; `GGML_CUDA_FA_KV_NATIVE=1` (V4) changes
  no bf16 row (it is q8_0-only).
  **Finding 1 (pre-existing, documented not fixed): mixed K/V types fall off the GPU attention path.**
  Any mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`, either direction) reserves `graph splits = 18` (vs 2),
  moves ~1.5 GiB into the host compute buffer and loses the FA scratch; 4B pp2048/tg128: `q8_0/q8_0`
  7924.47/98.94, `bf16/q8_0` 640.25/61.57, `q8_0/bf16` 1048.66/68.54, `f16/q8_0` 852.57/54.39.  So
  "bf16 keys + q8_0 values" is not usable today; same-type K/V is the practical choice.  Fixing it
  needs the FA kernels to accept a mixed `(type_K, type_V)` pair — larger than V3/V4, out of scope.
  **Finding 2 (pre-existing): gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter
  (`n_head_kv = 2` < 3 devices); maintainer's call (D11): **document only, do not fix** — small model,
  unlikely configuration; it runs on 1/2 GPUs and with `-sm layer`.
- **Block 15 cut (2026-09-10) — the attention-memory campaign wins;
the set is now 15 patches (block-15 tip `09a137566` on the canonical fork
rebuilt at `9113cc188`), beta-staged in `archive/work/block-15-campaign-wins/`.**
  The campaign (`archive/work/arch-independent-memory/`, `archive/work/qwen4exp/qsa-memory/`)
  was merged into one block by replaying the validated work-branch tree
  onto block 14, then re-validated **as a combination** (the per-win
  records did not carry over on their own).  Six wins, each with an
  environment A/B gate; **V4 is opt-in** (an *enable* switch) per the
  maintainer's rule of 2026-09-10 (a sub-2 % loss with a large memory win
  and no cheap fix ships opt-in):

  | win | mechanism | gate (default) | measured (ctx 204800, q8_0 KV, ub 2048) |
  |---|---|---|---|
  | W1 | QSA score chain: relu before the 4-D reshape + `n_blocks`-chunked `ggml_concat` assembly | `GGML_QSA_SCORE_MEM` (1) | qwen4exp 6690.40 -> 4450.40 MiB/GPU (ub1024 3346.50 -> 2274.35) |
  | W2 | derived QSA per-block bias + derived visibility; bias/mask no longer materialised; input-fill null guards (incl. the `llm_graph_input_attn_k` one) | `GGML_QSA_DERIVED_BIAS` (1), `GGML_QSA_DERIVED_VIS` (1), `LLAMA_QSA_SPARSE_FA` (sparse) | qwen4exp 4450.40 -> **3251.39** MiB/GPU, host 1262.70 -> **63.69** MiB |
  | W3 | keys-only QSA indexer cache (`v_enabled` in `llama_kv_cache`; no V tensor, no V-side op) | `LLAMA_QSA_KEYS_ONLY` (1) | indexer KV 956.26 -> **318.76** MiB/GPU |
  | W4 | ggml-alloc releases view sources whose views are never consumed (the uncounted-view leak) | none — a bug fix; `archive/work/block-15-campaign-wins/ab/w4-revert.patch` | repro 56.00 -> 16.00 MiB; no reserve change on any model |
  | V3 | derived kq mask: `GGML_OP_FLASH_ATTN_EXT` src[5..7] carry compact per-cell state and the MMA FA kernel derives visibility in-kernel; the packed mask tensor is still built in every graph and simply loses its consumer (so no model allowlist and no mis-served consumer) | `LLAMA_KQ_MASK_DERIVED` (1; `0` = packed) | 4B 1800.33 -> **1001.13**, 27B 1920.33 -> **1121.13** MiB/GPU; host -799.21; gemma-4-E4B/-31B (ISWA) -809.18/-811.17; scales as `n_kv x n_tps x 2 B` |
  | V4 | native q8_0 K/V in the FA kernels: dequantise during the shared-tile staging (16-byte chunk = 8 elements = a quarter q8_0 block) instead of staging a whole-cache F16 copy | `GGML_CUDA_FA_KV_NATIVE` (**default 0 = opt-in**) | 4B -> **257.13**, 27B -> **489.13**, gemma-4-31B -1224 MiB/GPU; qwen4exp unchanged |

  **The wins compose additively** — qwen4exp ub 2048: pristine 6690.40 ->
  W1 only 4450.40 -> W2 only 5491.39 -> W1+W2 3251.39 (W1 -2240, W2
  -1199, W3 -637.5/GPU, V3 -799, V4 -744/-632); both W gates off
  reproduces the pristine 6690.40/1262.70 exactly.  **Cost**: V3 -1.28 %
  prefill (4B pp20480/ub 2048, interleaved same-binary A/B) / +0.28 %
  (27B), decode -0.32 %/-0.15 %; V4 a further -1.85 % (4B) / -1.72 %
  (27B) prefill — the loss is the lost `cp_async` pipeline (a quantized
  source cannot be copied asynchronously; a 2-byte-access pass changed
  nothing), decode within noise, hence opt-in.

  **Combination validation (all on the merged tree, and then re-run from
  the delivered patches — see below):** reserve matrix on 4B (1 GPU), 27B
  (3-GPU Meta), gemma-4-E4B (1 GPU), gemma-4-31B (3-GPU) and qwen4exp
  (3-GPU) at ub 2048/1024/512 x V4 off/on — every number matches the
  per-win records; same-seed generated text **byte-identical** on all
  five models across every gate combination (V3 x V4 on the dense
  models; W1/W2/W3/V3/V4 — 7 configurations — on qwen4exp) at a short and
  a 40k-token prompt; adaptive-MTP gate **unchanged** (27B inline draft
  0.76744 (66/86, mean 3.28) in all four gate combinations; qwen4exp
  draft 0.44262 (54/122) in all six, **equal to the block-14 baseline**,
  and MTP stays +26 % over plain decode at ctx 32768); `test-backend-ops`
  FLASH_ATTN_EXT on ROCm0 (both V4 gates) and CPU, the six derived FA
  cases, VIEW/CONT/CPY/DUP/CONCAT, `test-alloc`, `test-batch-alloc`; the
  W4 revert restores `ggml-alloc.c` byte-identically to block 14.

  **Two things worth recording.**  (1) Re-validation caught a real wiring
  bug before the cut: the W3 gate was passed to `v_enabled` with the
  wrong polarity, so the indexer cache stayed keys-only-disabled (956.26
  MiB) while `LLAMA_QSA_KEYS_ONLY=0` enabled it — fixed and re-verified
  (`956.26 -> 318.76` on the default, `956.26` with the gate off).  This
  is exactly what the combination pass is for.  (2) A **pre-existing**
  bug was found (it reproduces on block 14=HEAD, so it is not a block-15
  regression): `gemma-4-E4B-it` on **3 GPUs with `-sm tensor`** aborts in
  the meta splitter (`ggml-backend-meta.cpp:1177`) on a FLASH_ATTN_EXT
  node whose K source has zero extent on one buffer, because `n_head_kv =
  2` is fewer than the device count (2 heads / 3 devices leaves one
  device with nothing).  It runs on 1 GPU, on 2 GPUs and on 3 GPUs with
  `-sm layer`; the 27B (4 KV heads) and gemma-4-31B (4/16) are
  unaffected.  Diagnosed by instrumenting the failing assert to print the
  op/tensor/split geometry (temporary change, reverted).  Left unfixed —
  out of scope for this block — and documented in `patches/README.md`.

  **Regeneration and delivery mechanics.**  The reference fork checkout
  (`~/llama.cpp`, branch `rdna-boosts`) had been rebased onto a master
  that is **two commits newer than the recorded fork point** (`f3f1a8f27`
  iGPU lazy-load default + `304665fe7` SYCL IQ-type-for-MoE, both
  2026-09-08/09, i.e. after `9113cc188`), so `format-patch
  9113cc188..tip` there would have exported those two upstream commits as
  patches 0001/0002 — a latent trap for any future regeneration.  The
  patches were therefore regenerated from a **canonical fork rebuilt at
  `9113cc188`** via `scripts/apply-all.sh` (strict 15/15 `git am`, zero
  whitespace warnings), and the resulting tree was verified identical to
  the validated tree except for the 3 files of those two upstream commits
  (`ggml-sycl` x2, `src/llama-model.cpp` — outside the validated paths).
  The delivered `0001`-`0014` files were kept byte-for-byte (the
  regenerated ones differ only in the `From <sha>` line and the
  `[PATCH NN/15]` series count, verified content-identical hunk by hunk);
  `0015-rdna-boosts-block-15-campaign-memory-wins.patch` is new.
  `make-patches.sh`'s default tip is now `09a137566`, the canonical
  block-15 commit (the local branch `block15-canonical` in the fork
  checkout keeps that chain alive).  `rdna-boosts-all.patch` = `git diff
  9113cc188..09a137566` (98 files).

  **Clean-apply simulation (the delivered artifact, end to end):** fresh
  worktree at `9113cc188` -> `scripts/apply-all.sh` (15/15 strict
  `git am`) -> fresh `gfx1201` Release build -> reserves (4B 1001.13 /
  257.13, 27B 1121.13 / 489.13, qwen4exp 3251.39 with the indexer KV at
  318.76), byte-identical coherence on 4B/27B/gemma-4-E4B/qwen4exp with
  every gate flipped, MTP 0.76744 / 0.44262, and the op suites — all
  green.

  **Upstream-drop check (2026-09-10):** GitHub was unreachable from this
  host (SSH key denied), so the check ran against the recorded upstream
  base `9cf3bf256`: the `ggml-alloc` unused-view release (W4), the
  keys-only indexer cache (W3) and the `llm_graph_input_attn_k`
  null-mask guard are all **still absent upstream** (the first two apply
  cleanly, the guard's call site is still unguarded while its own
  `can_reuse_impl` accepts a null mask), so Block 15 keeps every hunk.
  Re-check after the next `git fetch` before filing the `upstream/`
  candidates.

  **Upstream candidates A1/A2 prepared on a pristine master worktree
  (2026-09-10):** the `upstream/` backlog is now empty (four candidates,
  each with its own `.md` evidence):
  - **A1 `UPSTREAM-PR-kv-cache-keys-only`** (win W3): verified on
    unadulterated master `9cf3bf256` (CPU build, the real 3-shard qwen4exp
    IQ4_XS GGUF) -- the upstream indexer KV buffer is **72.00 MiB at ctx
    8192 (K 24.00 + V 48.00)** and drops to **24.00 MiB (K only)** with the
    patch; same-seed text byte-identical; `test-alloc` all PASSED,
    `test-batch-alloc` 0 failures.  The shape is worth noting: the store
    overrides the *key* head to the indexer size (128) but inherits the
    model's *value* head (256), so the dead V is twice the K it never
    accompanies.  Method note: the first upstream A/B was measured with
    `git apply -3` (which stages), so `git checkout -- .` did not revert it
    and both runs measured the patched tree; the `git reset --hard` re-run
    is the real unpatched number above.
  - **A2 `UPSTREAM-PR-attn-k-null-mask-guard`** (part of win W2): verified
    on master -- applies clean, compiles, byte-identical same-seed text;
    recorded in its notes as **hardening, not a live fix** (every upstream
    construction site builds a mask, and `can_reuse_kq_mask` itself
    dereferences it, so the guarded branch is unreachable upstream today).
    It is what the sibling `attn_kv` class already does and the prerequisite
    for a future null-mask feature.
  Both patches were apply-checked on pristine master (individually and
  together: 4 files, +18/-7); the master worktrees were reset afterwards.

- **Block-14 amendment (2026-09-10) — freed-cell KV handling moved from the
  host-side zeroing to kernel-side masked-V elimination; the gfx1151-only
  `zero_freed` host zeroing (2026-09-09 amendment) is REMOVED (block-14 tip
  `ff2b35f49` on `9113cc188`, regenerated 2026-09-10; blocks 01-13 patch
  files byte-identical).**  `src/llama-kv-cache.{cpp,h}` are back to the
  upstream state — no `zero_freed`/`rows_hw`/`sharers` wiring, no env
  `LLAMA_KV_ZERO_FREED`, no per-free GPU memsets; evicting a resident KV
  sequence is pure host cell bookkeeping again on every device.  In its
  place block 14 now carries the three **kernel-side** fixes that make the
  content of fully-masked (freed/stale) flash-attention cells unreadable,
  so the host workaround is unnecessary:
  - HIP `fattn-tile.cuh` (packed-bf16 PV path): zero the per-warp V
    register copies of rows whose P is +0.0 across the warp's columns
    before the bf16 dot.
  - HIP `fattn-mma-f16.cuh`: after each V-tile slice is staged in shared
    memory, zero the rows the mask tile marks blocked (-inf) for every
    query column of the block; one extra uniform barrier, masked path
    (`ncols2 > 1 || mask_h`) only; compile-time excluded for the
    `V_is_K_view` and NVIDIA-swizzled (`swz_V`) paths.
  - Vulkan `flash_attn_cm1.comp` + `flash_attn.comp` scalar path: never
    read V of fully masked columns (dead columns keep V = +0.0).
  All three are unconditional in their kernel paths (no arch/env gating) —
  generic correctness fixes for masked/freed FA cells (batch serving, KV
  eviction) active by default on every device.  Root cause (Strix Halo,
  gfx1151): WMMA f16 `x + (-0.0)` is inexact, so a masked column leaked
  the sign of whatever V its cell last held; the fix guarantees masked
  cells contribute exactly +0.0 at the multiply.  Validation on the
  gfx1151 box (ROCm 7.14-gfx1151 + Vulkan RADV), host zeroing disabled:
  16/16 identical-request determinism gates PASS on every KV cache type
  each backend's FA supports — ROCm f16/bf16/q8_0/q4_0 (plus ON==OFF
  bit-identical over 2064 cells/run), Vulkan also q4_1/q5_0/q5_1/iq4_nl;
  `test-backend-ops` FLASH_ATTN_EXT vs CPU 4591/4591 (ROCm) and
  7822/7822 (Vulkan); CPU same-seed greedy 51/64 tokens identical,
  divergence only at a near-tie (CPU non-FA vs GPU FA numerics);
  depth-16384 llama-bench decode tg128 within 0.05% of pre-fix, pp within
  single-run drift.  Full record:
  `archive/work/strix-halo/kvzero/RECORD-2026-09-09.md` +
  `archive/work/kv-sign-leak/HANDOVER-2026-09-09-mma-f16.md`.  Delivery:
  regenerated `patches/0014` only (blocks 01-13 patch bodies
  byte-identical) + `rdna-boosts-all.patch`; clean-apply sim at
  `9113cc188` strict 14/14 `git am`, zero whitespace warnings, applied
  tree == fork tip `ff2b35f49`; final-tree rebuild (delta vs the
  validated kernel-fix tree = the llama-kv-cache revert only) passes the
  16/16 gate and no longer logs the freed-cell zeroing.

- **Block-14 amendment (2026-09-09) — freed-cell KV-zeroing gated to gfx1151
  (fork block-01 commit `7c4d9c4e0`, block-14 tip `27485f1ca`, 14 commits on
  `9113cc188`; previous tip `0f2b7a4e1` superseded).**  Block 14's
  `seq_rm`/`seq_keep`/`clear` row zeroing (freed KV cells kept at +0.0 as a
  masked-column guard for the gfx1151/Strix-Halo WMMA f16 `x+(-0.0)`
  inexactness, ported from the strix lineage commit aad5adb08f) is now
  **enabled only when a KV-cache buffer device description carries `gfx1151`**
  (env `LLAMA_KV_ZERO_FREED=0/1` overrides the auto detection).  Everywhere
  else the pre-block-14 behavior is restored: evicting a resident KV sequence
  is pure host cell bookkeeping again.  Reason: without the gate, freeing an
  N-token sequence issued ~48×N per-cell 512-byte memsets (ggml's
  meta/multi-buffer memset decomposes one per-layer zeroing call into one
  synced `cudaMemsetAsync` per cell across the GPU head-split sub-buffers,
  each ~30-60 µs), so replacing a ~13k-token KV stalled ~18-24 s before the
  new prefill began on multi-GPU RDNA4 (3x R9700 gfx1201; reproduced on a
  plain dense 4B model too — model-agnostic).  Verified: on gfx1201 the
  A/B stall is gone (identical workload 24.5 s -> ~6 s) and the zeroing-off
  determinism gate passes (16 + 8 identical greedy requests, per-position
  top-8 logprobs float64-compared — the same gate that found the leak on
  gfx11); on the gfx1151 Halo box the gate enables
  ("freed-cell KV row zeroing enabled (gfx1151)") and the 16-run control is
  unchanged.  Regenerated `patches/0014` only (blocks 01-13 patch bodies
  byte-identical); clean-apply sim at `9113cc188` strict 14/14 `git am`,
  zero whitespace warnings, applied tree == fork tip `27485f1ca`.
  Follow-up (open): develop a performant gfx1151 flash-attn kernel-side fix
  so the host-side zeroing can be removed entirely.

- **Block-01 refresh (2026-09-09) — adaptive MTP draft depth updated to the
  llama.cpp PR #27210 review head (fork block-01 commit `7c4d9c4e0`,
  block-14 tip `0f2b7a4e1`, 14 commits on `9113cc188`).**  Block 01 was cut
  from PR #27210 (author: stew675) at its `0994374fd` state; the PR then
  advanced through a maintainer review round (`8408cdabf` comment fixes +
  `d236d41a2`, the review-response changeset).  The block is now refreshed
  to the PR head `d236d41a2`, still delivered as **one squashed patch
  block** (`git diff 9113cc188..d236d41a2` = 15 files, 519+/35-, applied
  as the single block-01 commit; blocks 02-14 re-based on top untouched).
  Review-round content now in block 01: `common_params_speculative::
  has_mtp()` helper (arg.cpp/common.cpp/server-context.cpp/init result
  refactored through it); a new `accept_partial()` virtual +
  `common_speculative_accept_partial()` so a partial acceptance the
  context could not apply (checkpoint-restore path in tools/server and
  examples/speculative-simple) is reported once and the following replay
  round cannot feed stale draft counts to the adaptive controller
  (non-adaptive accept path unchanged); the adaptive depth reset moves
  ahead of the empty-prompt early return in `begin()`; `
  --spec-draft-n-min-adaptive` rejects values < 1 and is documented
  (docs/speculative.md, tools CLI/server READMEs); the invalid-range
  check is `GGML_ABORT` -> `std::runtime_error`; draft-mtp +
  draft-mtp-adaptive together are rejected (shared ctx_dft); the delta-
  net conv-state snapshot-bound rationale comment; stale "defaults to 2"
  test comment fixed (default is 3) + value-0 rejection case.
  Regeneration mechanics: canonical fork rebuilt at `9113cc188` from the
  previous set (am-tip `050ec89ce`), block 01 replaced in place by the
  squashed PR-head changeset, blocks 02-14 `git rebase --onto` (clean,
  no conflicts — blocks 02-13 touch no block-01 file, block 14's
  common/arg/common.h hunks are disjoint).  Tree verification: old-tip..
  new-tip delta is exactly the review changeset (13 files, 129+/70-, ==
  `0994374fd..d236d41a2`), every other file byte-identical; regenerated
  0002-0013 patch bodies byte-identical to the previous delivery, 0014
  refreshed only in index lines/hunk offsets for the 3 common files;
  regenerated 0001 diff body byte-identical to the PR head changeset.
  Verification (local 3x R9700, gfx1201, ROCm 7.14): clean-apply sim at
  `9113cc188` strict 14/14 `git am`, zero whitespace warnings, applied
  tree == fork tip; rebuilt `test-arg-parser` + `test-speculative-
  adaptive` pass; plain-decode same-seed coherence (seed 42/temp 0,
  Qwen3.5-4B-Q8_0) token-IDENTICAL to the known-good `050ec89ce` build.
  The refresh touches no GPU kernels and no non-speculative host decode
  path — all changes live in the MTP-typed/adaptive code, the option
  parser and comments/docs.

- **Re-base (2026-09-08) — delivery moved to upstream master `9113cc188`
  (block-14 tip `78e67a3d8`).**  Upstream moved 14 commits past the
  `050dde50c` fork point (server checkpoint eviction, Kimi-K3 recurrent
  rollback, chat-parser split, ggml_prec spec, metal/vulkan/opencl fixes,
  spec single-device meta-wrapper handling #28390, and — decisive for this
  re-base — `d4389a4dd`/PR #28604 which **reverted #24233**, the very
  change block 06 diverged from).  An `apply-all.sh` run against the fresh
  master tip failed at block 06 in a way even `git am -3` cannot fix: the
  upstream revert deleted block 06's pre-image, so the block's change is a
  no-op on the new base (nothing left for the patch to do).  Resolution:
  block 06 was reduced to a host-buffer **rationale marker** commit (6
  comment lines above the now-unconditional `integrated = false` in
  `ggml-cuda.cu`), keeping the 14-block structure and all downstream block
  numbers intact; block 14's quantized-KV tensor-split gate merged
  **additively** with #28390's single-device `SPLIT_MODE_TENSOR` warn in
  `src/llama-context.cpp` (both kept, in sequence; #28390's code comment
  shows the same single-device-no-meta-wrapper intent as block 07, so no
  semantic collision).  Content verification against the previous delivery
  (re-applied at `050dde50c`): blocks 01-05 and 07-13 are byte-identical;
  block 06 differs as designed; block 14 differs only in the
  llama-context.cpp resolution region.  Regenerated at `9113cc188`
  (`f84549d23..78e67a3d8`) and clean-apply re-verified (strict 14/14
  `git am`, zero whitespace warnings, applied tree == fork tip
  `78e67a3d8`).  Coherence verified on the Strix box (gfx1151, ROCm 7.14):
  llama-cli same-seed output IDENTICAL to the canonical `72f0ee944` build
  (tensor + layer split x f16/q8_0/bf16 KV, and a long-prompt run at depth
  16384), clean runtime diagnostics, and the dense adaptive-MTP gate green
  on the new build (draft acceptance 0.833 at acc/pos 0.944/0.833/0.722;
  draft-mtp 20.3 t/s vs plain 7.9 t/s on the same prose prompt; MTP
  same-seed byte-identical old-vs-new).  The previous `050dde50c`-based
  regeneration (`d65a96084..ce641322e`) is superseded; the pre-re-base fork
  chain is preserved at `backup-rdna-boosts-bfcc4be99` and the known-good
  `72f0ee944` binary under `/tmp/rdna-ref-bin/` (session-local).

- **Block-14 amendment (3rd on 2026-09-08) — quantized-KV tensor-split
  gate:** the `q4_1`-family KV cache types (`q4_1`, `q5_0`, `q5_1`,
  `iq4_nl`) aborted during the first graph reserve under multi-GPU
  `SPLIT_MODE_TENSOR` on gfx1201 (3x R9700) —
  `ggml-backend-meta.cpp:538 GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)`
  — on both dense qwen35 (Qwen3.6-27B) and qwen4exp (Flash-Next), with
  `f32/f16/bf16/q8_0/q4_0` KV and layer split passing.  Root cause is
  **upstream**: reproduced on pristine vanilla llama.cpp at the fork
  point `050dde50c` (identical assert, non-qwen4exp Qwen3.5-4B; also at
  1 GPU, since upstream wraps even a single device in the Meta backend)
  and still unfixed on current upstream master.  Tensor split forces
  flash attention, whose CUDA/HIP kernels read the quantized K/V cache
  natively only for `q4_0`/`q8_0` (plus the float types); for the
  q4_1-family types the attention subgraph is externalized into
  op-NONE graph leaves (MIRRORED split state) which collide with the
  AXIS-0 elementwise gate branch of the qwen35/qwen4exp gated attention
  at the `attn_gated` `MUL` — the meta splitter cannot reconcile
  MIRRORED x AXIS-0.  Fix: a context-creation gate in
  `llama_init_from_model` (`llama-context.cpp`, block-14-owned in the
  set) that rejects K/V types outside FA's native set with a clear
  error when the Meta device is actually in use (tensor split over
  >= 2 GPUs; the fork's single-GPU "tensor" mode skips the Meta wrapper
  and is untouched — upstream, whose 1-GPU mode also wraps Meta, gets
  the clean error too).  Validated 2026-09-08 on gfx1201 (3x R9700,
  ROCm 7.14): KV-type matrix on dense 27B Q8_0 + Flash-Next IQ4_XS
  (3-GPU tensor) — `f32/f16/bf16/q8_0/q4_0` generate;
  `q4_1/q5_0/q5_1/iq4_nl` and `k=q4_1 v=bf16` / `k=bf16 v=q4_1` fail
  cleanly (zero asserts, actionable message); layer split + q4_1
  Flash-Next 25.9 t/s (unchanged); qwen4exp derived-cache pool-gate
  byte identity holds (tokens identical with the pool skipped vs
  `GGML_CUDA_QSA_INDEXER_CACHE=1`); dense-27B same-seed coherence A/B
  (gate stripped vs applied on the same tree) byte-identical;
  test-llama-archs qwen4exp all OK (NMSE 1.01e-13).  Canonical fork
  rebuilt at `050dde50c` (am-commits `d65a96084..ce641322e`, block-14
  tip `ce641322e`); set regenerated with `scripts/make-patches.sh`;
  clean-apply sim re-verified 2026-09-08 (14/14 strict `git am`, zero
  whitespace warnings, applied tree == fork tip `ce641322e`, full build
  clean, coherence byte-identical to the validation tree).
 the 2026-09-07 local
  delivery (`9850143`: block-14 **derived-cache pool gate**, regen at
  fork tip `bfcc4be99`) had never been pushed; the 2026-09-08 lineage on
  `origin/main` (issue #18 MUL_MAT_ID pair-fusion layout gate + issue
  #19 moe_weighted_reduction float4 remainder, both folded into blocks
  13/14; the block-14 compiler-warning cleanup; the qwen4exp
  tensor-split HIP gate — regen tip `2f1dc384b`) had been authored from
  a clone without it.  The two block-13/14 regens touched disjoint
  source hunks, so blocks 13/14 now carry all of it: QSA quantized-KV
  decode gate, derived-cache pool gate, the issue-18/19 fixes, the
  warning cleanup and the tensor-split backend gate.  Canonical fork
  rebuilt at `050dde50c` (am-commits `7df708e66..72f0ee944`, block-14
  tip `72f0ee944`); set regenerated with `scripts/make-patches.sh`;
  clean-apply sim re-verified 2026-09-08 (14/14 strict `git am`, zero
  whitespace warnings, applied tree == fork tip `72f0ee944`).
- **Block-14 amendment (2nd) — qwen4exp tensor-split backend gate
  (2026-09-08):** follow-up to the Vulkan validation sweep: block 14
  had removed upstream's `case LLM_ARCH_QWEN4EXP: // TODO: fix
  test-llama-archs` from `llm_arch_supports_sm_tensor`, enabling
  qwen4exp tensor split for every backend.  That is validated on
  ROCm/HIP only (3x R9700, NMSE 9.87e-14 vs CPU); on backends that
  cannot run the fused QSA/HC/WS4 ops on-device (Vulkan, Metal, SYCL;
  NVIDIA CUDA untested) the CPU-fallback subgraphs leave the meta
  splitter unable to reconcile mirrored-vs-split operand states and it
  aborts at graph reserve (`ggml-backend-meta.cpp` `handle_generic`,
  e.g. the qwen4exp gated-attention `MUL` on Vulkan — `test-llama-archs`
  died at the qwen4exp Meta row).  The enablement is now `#ifdef
  GGML_USE_HIP`, restoring upstream's clean "not implemented" error /
  arch-test SKIP on all other builds.  Verified: Vulkan — full
  test-llama-archs sweep completes RC=0 (457 rows, statuses identical
  to upstream 050dde50c, qwen4exp Meta SKIP like upstream), qwen4exp
  single-device still OK (9.01e-08, roundtrip OK), llama-cli
  qwen4exp `-sm tensor` fails with the upstream message; HIP —
  qwen4exp Meta still OK 9.87e-14 (validated path unchanged).  Canonical
  fork rebuilt at `050dde50c`; block-14 tip `13719e3ca` →
  `2f1dc384b`; set regenerated; clean-apply sim re-verified (14/14
  `git am`, zero whitespace warnings, applied tree == fork tip).
- **Block-14 amendment — compiler-warning cleanup (2026-09-08):** the
  block-14 sources warned under the `build-llama-vulkan` (system clang
  16.2.1, `-Wall -Wextra`) and `build-llama-rocm-714` (ROCm clang)
  host builds.  Five warnings, all from block-14 code, fixed and
  folded into the block-14 commit:
  - `ggml.c` — unused `n_blocks` local in the `ggml_indexer_fill`
    builder (removed).
  - `ggml-cpu.c` — `-Wswitch`: the exhaustive CPU compute-forward
    switch had no case labels for the new `GGML_OP_INDEXER_SCORE` /
    `GGML_OP_INDEXER_FILL` ops (GPU-only fused ops; the CPU plan
    phase already aborts on them as "op not implemented" before
    compute, so the case is an unreachable `GGML_ABORT`, mirroring
    `GGML_OP_COUNT`).
  - `ggml-cpu/ops.cpp` — two `-Wunreachable-code-break` warnings: the
    `break` after the noreturn `GGML_ABORT("fatal error")` in the
    `HC_MIX`/`HC_COMBINE` CPU type dispatchers' default cases
    (dropped, matching upstream convention).
  - `qwen4exp.cpp` — `idx_cache` was narrowed to `bool`, making the
    documented `GGML_CUDA_QSA_INDEXER_CACHE=2` debug probe
    (`idx_cache != 2`) tautologically true (`-Wtautological-constant-
    out-of-range-compare`); restored to an `int` with the 0/1/2
    tri-state so probe-2 (pool read without the fill) is reachable
    again.  `-Wsign-compare` in the gfx-id sniff loop (`size_t`
    counter vs `ggml_backend_dev_count()`).
  No generated-code or runtime-behavior change in default configs.
  Verified: the four TUs compile warning-free with the exact
  build-vulkan flags; full Vulkan + ROCm 7.14 (gfx1201) builds clean
  on the re-applied sim tree.  Canonical fork rebuilt at `050dde50c`;
  block-14 tip moved `3529b3497` → `13719e3ca`; set regenerated
  (14/14 `git am`, zero whitespace warnings, applied tree
  byte-identical to the fork tip); `rdna-boosts-all.patch` refreshed.
- **Block-14 amendment — MUL_MAT_ID pair-fusion layout gate (2026-09-08,
  issue #18):** community report + detailed root-cause analysis by
  `briansp2020` (production single-R9700 deployment of the 14-block
  set, ROCm 10): the block-13/14 MUL_MAT_ID gate+up pair fusion
  aborted the process with `GGML_ASSERT(ne11 == 1 && n_expert_used > 1)`
  in `ggml_cuda_mul_mat_q_pair` whenever two MUL_MAT_ID nodes shared
  src1/ids in a layout the fused kernel does not express (src1->ne[1] > 1
  or top-1 routing) — `test-backend-ops -b ROCm0` died in the
  MUL_MAT_VEC_FUSION group.  The dispatcher gate now requires the
  callee's layout preconditions; such pairs fall back to the per-node
  path, and the qwen4exp sparse-MoE pair (standard layout) still fuses.
- **Block-13 amendment — moe_weighted_reduction float4 remainder fix
  (2026-09-08, issue #19):** community report by `briansp2020`: the
  2026-09-06 mwr-float4 fold dropped the last `n_embd % 4` columns of
  every output row for `n_embd % 4 != 0` (silent wrong output;
  `MOE_WEIGHTED_REDUCTION(n_embd=63, ...)` failed).  The float4 quad
  kernel is now gated to `n_embd % 4 == 0` (where it is also
  alignment-safe) and the upstream scalar bounds-checked kernel covers
  the rest; the aligned path is byte-unchanged.
  Both fixes validated here (3x R9700 gfx1201, ROCm 7.14):
  `test-backend-ops -b ROCm0` **16590/16590** with the fusion active,
  MUL_MAT_VEC_FUSION 1265/1265, MOE_WEIGHTED_REDUCTION 6/6, same-seed
  llama-cli streams byte-identical (Flash-Next IQ4_XS 3-GPU and dense
  27B single-GPU; default vs `GGML_PAIR_OFF=1`/`GGML_PAIR_DENSE_OFF=1`),
  prefill A/B confirms the pair fusion still fires (pp2048/pp8192
  default > pair-off beyond noise), Flash-Next full model runs clean on
  CPU (`-ngl 0`).  Fork tip moved `3529b3497`; set regenerated;
  clean-apply sim re-verified (14/14 `git am`, zero whitespace
  warnings).  Full record:
  [`patches/README.md`](patches/README.md).
- **Block-14 amendment — QSA quantized-KV decode gate (2026-09-07):** a
  quantized KV cache type (e.g. `--cache-type-k q8_0`) aborted qwen4exp
  context init (`GGML_ASSERT` in `ggml_indexer_fill`: the fused decode
  indexer ops read raw cache rows in F32/BF16/F16 only, but the indexer
  sub-cache shares the main `--cache-type-k`).  `build_qsa_top_k` now
  falls back to the per-op chain for quantized indexer keys.  Validated
  on Strix Halo across the full KV-type matrix f32/f16/bf16/q8_0/
  q4_0/q4_1/iq4_nl/q5_0/q5_1 (start + generate, zero errors; BF16 fused
  path unregressed).  Fork tip moved `60aa4173d`; set regenerated.
- **Block-08 amendment — PR #15 integrated (2026-09-07):** community
  report + fix by DanoPTT (single R9700, production since 2026-09-07):
  block 08's mul_mat+bias fusion through a view node handed the
  mmvq/mmvf kernels a destination whose shape the guards never checked
  (a reshape moves tokens between dimensions on multi-sequence
  batches) → `GGML_ASSERT(ids || dst->ne[1] == 1)` abort.  Fix folded
  into the block-08 commit (delivery convention): require the
  through-view destination to satisfy the kernels' shape constraint
  before fusing.  Fork tip moved `3bebffd6b`; set regenerated;
  verified here (3x R9700): clean-apply sim tree-identical, build
  clean, test-backend-ops 6759/6759, dense same-seed byte-identical
  pre vs post fix, 3-GPU hybrid == RCCL, parallel 2-slot decode clean.
- **Re-baseline to upstream master `050dde50c` + block 14 (2026-09-07):**
  fork point moved from `465e49b9c` to the current master tip (22
  upstream commits; the ggml-cuda-touching ones — `b74f590ea` f16 FA
  divergent-barrier fix #27870, `73ab7599b` branchless Q4_K/Q5_K mmvq
  unpack #26705, `473599738` gfx90c HIP support #26454 — merged in
  disjoint hunks).  The `~/llama.cpp` `rdna-boosts` fork was rebuilt
  from `patches/` via `scripts/apply-all.sh` (13/13 `git am` clean at
  `050dde50c` after one manual block-04 conflict in
  `tests/test-backend-ops.cpp` — upstream LEAKY_RELU perf cases kept
  alongside block 04's) and **block 14 (qwen4exp support) was promoted
  from `beta/qwen4exp`** (fork delta `c261553a1..dd4301fb4`, re-based;
  one manual `common.cuh` conflict — upstream gfx90c APU macros kept
  alongside the block's `GGML_CUDA_CC_IS_GFX1151`).  Set regenerated
  with `scripts/make-patches.sh` (base `050dde50c`, canonical
  am-commits `90a816a68..3bebffd6b`, 14 blocks) and
  `rdna-boosts-all.patch` refreshed (87 files).  Clean-apply sim at
  `050dde50c` re-verified 2026-09-07 (applied tree byte-identical to
  the fork tip).  Full record:
  [`patches/README.md`](patches/README.md).
- **Re-baseline to upstream master `465e49b9c` (2026-09-06):** fork point
  moved from `9cffdcc80` to the current master tip (18 upstream commits
  past the fold-verified base `8b4b3558f`, 57 past the old fork point;
  the ggml-cuda-touching ones — `73a43d1f6` mmid/mmf race fixes #28475,
  `5fdfa6282` GDN l2-norm fix #28068 — merged in disjoint hunks, zero
  conflicts).  The `~/llama.cpp` `rdna-boosts` fork was rebuilt from
  `patches/` via `scripts/apply-all.sh` (13/13 `git am` clean, zero
  whitespace warnings; per-file content check on all 112
  upstream-touched files passed) and the set regenerated with
  `scripts/make-patches.sh` (base `465e49b9c`, canonical am-commits
  `45bf4d291..c261553a1`).  Two prerequisites: the 0044cfe fold had
  stripped the format-patch mail headers from 0002/0004/0008/0013 —
  restored from the pre-fold originals (delivery commit 0610b75) — and
  the block-13 message's fold-amendment trailer was re-dated to the
  fold's true date (tip amended `b4b760eb8` -> `c261553a1`).
  `rdna-boosts-all.patch` refreshed (45 files; was stale at 41,
  pre-fold).  Clean-apply sim at `465e49b9c` re-verified 2026-09-06
  (applied tree byte-identical to the fork tip).  The `qwen4exp` fork
  branch was rebuilt on the new base + the consolidated beta support
  patch (see `beta/qwen4exp/README.md`).
- **Campaign date re-stamp (2026-09-06):** the gfx1151/qwen4exp campaign
  docs had run a week ahead of the real calendar; every
  `wip/`/`beta/`/archive date (filenames + text) was collapsed onto the
  real git dates (2026-09-05/06) and the moved records' stale
  `benchmarks/2026-09-*` references were repointed at
  `archive/work/wip-archive/qwen4exp/discovery/`.
- **Block-13 RDNA3.0 gate relaxation (2026-09-05, folded into block 13):**
  the fused MoE gate+up+GLU MMQ prefill arm + its `J_max_gate` tile
  caps are now also on RDNA3_0 (gfx1100), validated on a single RX
  7900 XTX (ROCm 7.14) with Qwen3.6-35B-A3B True-Q3_K_M (ub 2048,
  1-GPU pinned): fusion fires, same-seed coherence IDENTICAL fused-on
  vs off, prefill gains pp2048 +9.4% (5405 vs 4939), pp16384 +7.8%
  (4487 vs 4162), decode unchanged (tg128 130.3 vs 130.4).  The
  RDNA4-tuned J caps transfer (uncapping regressed pp2048 5405 -> 4819
  / pp16384 4487 -> 4070, below the 3-op fallback; a Q3_K@96 probe
  also lost to the cap 64).  Set regenerated from a canonical fork
  rebuilt at `9cffdcc80` (13 am-commits, block-13 tip `8c2ace510`);
  clean-apply sim verified (zero whitespace warnings, applied tree
  byte-identical to the fork tip).  Full record:
  [`archive/work/wip-archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`](archive/work/wip-archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md).
- **Block-13 RDNA3.5 gate relaxation (2026-09-05, folded into block 13):**
  the fused MoE gate+up+GLU MMQ prefill arm + its `J_max_gate` tile
  caps were RDNA4-only; validated on Strix Halo (Ryzen AI MAX+ 395 /
  Radeon 8060S, gfx1151, ROCm 7.14) with Qwen3.6-35B-A3B True-Q3_K_M
  (ub 2048): same-seed coherence IDENTICAL fused-on vs off, prefill
  gains match RDNA4 (pp2048 +5.3% 1590 -> 1674, pp16384 +4.6% 1360 ->
  1423), decode unchanged (tg128 71.5). The RDNA4-tuned J caps
  transfer (uncapping regressed pp2048 1674 -> 1111 / pp16384 1423 ->
  1334).  Full record:
  [`archive/work/wip-archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`](archive/work/wip-archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md).
- **Block-13 MTP regression fixes (2026-09-02, folded into block 13):**
  (1) dense adaptive-MTP collapse (18.3 -> 27.5 t/s) — the mmvq
  item-split/rpb kernel is register-bound at multi-token decode batches
  (ncols 2..8 = the speculative verify step); fixed with a re-added
  pre-block-13 K-split kernel (`mul_mat_vec_q_ksplit`) for those batches
  and long-K single-token rows (plain decode 29.0 -> 30.1, output
  bit-identical to the 12-block build).  (2) MoE adaptive-MTP collapse
  (draft acceptance 0/1527, 53 t/s vs plain 90) — the block-08
  rms_norm->mmvq Q8_1 quantize-cache fold corrupts multi-token MUL_MAT_ID,
  so verify logits diverge from single-token decode; the fold is now gated
  to single-token MMID + plain MUL_MAT consumers (acceptance 0 -> 0.51,
  MTP 126 t/s vs upstream ~113).  MoE MTP had no baseline data, which is
  why it slipped.  Details + verification: `patches/README.md` block-13
  notes.  The adaptive-MTP baseline gate and expectations now live in
  [`benchmarks/mtp-adaptive-methodology.md`](benchmarks/mtp-adaptive-methodology.md)
  — run Protocol A there before shipping decode/fusion changes.
- **Fork tip:** the fork block-12 commit was amended 2026-09-04 with the
  runtime NCCL-failure fallback (issue #13); block 13 was amended
  2026-09-02 with the two MTP regression fixes, 2026-09-05 with the
  RDNA3.5 (Strix Halo) then RDNA3.0 (gfx1100) fused-MoE-MMQ gate
  relaxations and 2026-09-06 with the model-neutral Strix MoE mmq
  folds.  The set was regenerated 2026-09-06 from a canonical fork
  rebuilt at `465e49b9c` (13 am-commits, block-13 tip
  `c261553a1`); the clean-apply sim at `465e49b9c` applies with zero
  conflicts/whitespace warnings and its tree is byte-identical to the
  fork tip.
- **Fork point (baseline):** llama.cpp master at `465e49b9c` (re-based
  2026-09-06 from `9cffdcc80`, itself re-based 2026-09-02 from
  `0eadefebd`; 57 commits of drift from the old fork point — see
  `patches/README.md` for the dated re-base record, incl. the 2026-09-02
  manual merges vs upstream's #27970 (sparse-fa) and #25952 (fused MoE
  expert reduction)).
- **Set:** 14 patches in `patches/` (`0001`-`0014`).
- **Verified:** clean apply + full build + llama-cli same-seed coherence
  IDENTICAL (hybrid vs RCCL, 3-GPU) on the rebuilt fork; the clean-apply
  sim at `465e49b9c` applies with zero conflicts/whitespace warnings and
  its tree is byte-identical to the fork tip (`c261553a1`; 2026-09-06
  regeneration — earlier regenerations were re-verified on the RX 7900
  XTX box with sim build coherence identical + perf reproduced). tg64
  38.12 / tg512 41.08 and the block-13 numbers are unchanged — the
  re-base is content-identical plus upstream's additions.
- **Whitespace-clean apply:** the regenerated set applies with **zero git
  whitespace warnings** (`git am` 01-13; re-verified 2026-09-02 on a
  fresh checkout at `9cffdcc80`, re-verified 2026-09-04 after the
  block-12 amendment, re-verified 2026-09-05 after the block-13 RDNA3.5
  gate relaxation and again after the RDNA3.0/gfx1100 fold,
  re-verified 2026-09-06 on the `465e49b9c` re-base).
- **Deployment:** 3-GPU hybrid (`HIP_VISIBLE_DEVICES=0,1,2`, unpinned) gives
  depth-16384 decode 38.71 t/s (+21.8% vs 2-GPU). See
  [`patches/README.md`](patches/README.md) for block-12 env knobs and the
  server config.
- **RDNA4-only gate:** block 12 refuses to init off gfx1200/gfx1201 and
  falls back to RCCL (community RDNA3 verification pending).
- **Runtime NCCL-failure fallback (2026-09-04, issue #13):** block 12 no
  longer aborts when NCCL/RCCL fails at runtime — on the first failure it
  clears the sticky HIP errors on each AR device, warns once, stops using
  NCCL for the rest of the run and re-routes AllReduce to the internal
  pipeline (or the meta backend's butterfly).  This covers RCCL >= 2.30.4
  refusing kernel dispatch on a PCIe root port without AtomicOp completer
  support (e.g. PCH/Z390; `ncclCommInitAll` succeeds — see
  ROCm/ROCm#6520).  Folded into the block-12 commit; re-verified
  2026-09-04 (clean-apply sim, build, same-seed coherence IDENTICAL pre
  vs post fix on 27B Q8_0, depth-16384 tg unregressed: 2-GPU 32.48 ->
  32.40, 3-GPU 39.33 -> 39.31).
- **Block-12 AR_PROFILE fix (2026-09-01, PR #8):** AR-profile `devices[]`
  init order fixed — `GGML_CUDA_AR_PROFILE=1` no longer faults GPU 1
  under MTP (pre-fix reproduced on 3x R9700; post-fix clean, profiler
  dumps on every device).  Regenerated into the set; coherence unchanged.
- **Block-02 MTP chunked-GDN prefix (2026-09-01, PR #9):** block 02 now
  runs its chunked WMMA GDN on long single-sequence MTP prefills (prefix
  `n_tokens-K` + sequential K-tail) — +7.5% prefill at ~5.5k prompt,
  +7.7% at ~38k on 3x R9700, 64-token same-seed output token-identical
  to sequential.  Opt out: `GGML_CUDA_GDN_CHUNKED=0`.
