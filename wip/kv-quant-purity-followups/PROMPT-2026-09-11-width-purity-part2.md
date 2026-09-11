# NEXT-SESSION PROMPT — Width purity, part 2: F1 (q8_0/q4_0 KV) + F2 cause 2 (qwen4exp W >= 5) + F3 (native sub-q8_0 KV), designed together

> Hand this file's contents (or this path) to the next session.  Written 2026-09-11, immediately after
> the block-14 hyper-connection amendment landed (delivery `c6489c1`, canonical tip `1d8f53594`).
> Everything below was measured on 3x gfx1201 (R9700) with ROCm `/opt/rocm-7.14-gfx1201`.

> **STATUS 2026-09-11: F1 is DONE** — fixed as a block-08 amendment (canonical tip `1bcf4e82d`); the
> cause was the FA *kernel-family* chooser, not the staging (see `../../GREEDY-PURITY.md` §14).  **The
> remaining work in this brief is F2 cause 2 and F3**, and the F1 sections below are the pre-fix
> record — read them for the instruments and the excluded-suspect list, not for the cause.  The F1 fix
> installed a band rule (`n_q <= 8` must use one FA family) that any new native path must respect;
> F3's first experiment is a `GGML_CUDA_FA_ALL_QUANTS=ON` build A/B.

## Mission

The decode==verify invariant holds for dense models with **f16/bf16/q4_1/q5_0/q5_1/iq4_nl** KV, and now
for **qwen4exp up to `--spec-draft-n-max 3`** (block-14 amendment, done 2026-09-11).  Three gaps remain,
and they look like one family — width-selected **kernel dispatch** bands:

* **F1** — a `q8_0` or `q4_0` K/V cache breaks the dense `n_max <= 7` guarantee.
* **F2 cause 2** — qwen4exp is still impure at `W >= 5` (`{1,2,3,4} {5} {6,7} {8}`); cause 1 is fixed.
* **F3** — the sub-`q8_0` KV types are pure but ~3.4x slower (no native FA path).  The native path is
  where the *impurity* lives (F1), so F1 and F3 must be designed together: **any new native path must
  be width-invariant by construction.**

Deliverable: a **delivery** fix for F1 (correctness, block-owned) and for F2 cause 2 if it is the same
site; F3 stays a **block-15 beta** feature (the mechanism already shipped there).  Do not assume they
share a cause — the measured boundaries differ (F1 `W=2->3` for 4B q8_0, cause 2 `4/5`, `6,7`, `8`) —
**localize both first**, then decide.

## Read first

1. `GREEDY-PURITY.md` §11 (causes A/B), **§12 (the KV-type scope — F1's table)**, **§13 (the qwen4exp
   hyper-connection band — F2's matrix)**.
2. `wip/kv-quant-purity-followups/README.md` — the F1/F2/F3 briefs, the evidence tables, the excluded
   candidate lists, and the F2 blow-by-blow (how the HC divergence was found).
3. `AGENTS.md` — MANDATORY (block-amendment flow, owner-based block choice, pushing policy, the
   coherence gate, the KV-type purity bullet, `GGML_CUDA_ALLREDUCE=nccl` is not a bit-identical
   reference).
4. `benchmarks/mtp-adaptive-methodology.md` (Protocol A, rule 3) before any decode/fusion change.
5. `beta/block-15-campaign-wins/README.md` + `HANDOVER.md` §10 (block 15 owns the KV-native staging).

## Environment and state

* Canonical fork `/tmp/canon-llama` @ **`1d8f53594`** (tree **`e36263da5`**), branch `rdna-boosts`,
  **clean**, 15 blocks; build dir `build-base` (built 2026-09-11 = the current canonical build).
  Blocks: 00 `1c7ab0e89`, 01 `aa4108b9d`, 02 `6e81ed5ed`, 03 `4dc962aa9`, 04 `03d004517`,
  05 `70f330aed`, 06 `d2fc2cb34`, 07 `110b5391d`, 08 `5ea46d1b2`, 09 `29880b1e4`, 10 `33a1e5f27`,
  11 `f4e75a30a`, 12 `cac14423e`, 13 `855515420`, 14 `1d8f53594`.  If `/tmp` is gone: clone
  `ggml-org/llama.cpp`, checkout `9113cc188`, `scripts/apply-all.sh .` (15/15 strict, tree
  `e36263da5`), build with `BUILD_DIR=build-base EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`.
  Fast loop: `cmake --build build-base --target llama-cli llama-bench ggml-hip -j 16`.
* Delivery repo `~/llama-cpp-rdna-boosts`, `main` = **`c6489c1`** (pushed).  Block 15 beta: tip
  **`54859fdda`** (tree `543ccc015`, base `1d8f53594`), patch in
  `beta/block-15-campaign-wins/block-15-campaign-wins.patch`, `[PATCH 15/15]`.
* 3x gfx1201 (R9700); ROCm `/opt/rocm-7.14-gfx1201`; `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib`.
* Models: 4B `/home/stew675/Qwen3.5-4B-Q8_0.gguf`; 27B `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf`;
  MoE `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`;
  qwen4exp `/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
  (+ draft `/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, **and the model has
  512 experts** — relevant, see the MoE band below); gemma-4-E4B
  `/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf` (SWA); gemma-4-31B
  `/llm/models/Gemma4/31B-QAT/Q4_K_XL/gemma-4-31B-it-qat-Q4_K_XL.gguf`.

## Instruments (both worked; use them before theorising)

* **Width probe** (the sensitive instrument — the raw-logit hash, *not* text):

  ```sh
  clang++ -O2 -std=c++17 -I /tmp/canon-llama/include -I /tmp/canon-llama/ggml/include \
    wip/kv-quant-purity-followups/tools/logits-dump-kv.cpp -o /tmp/lw-f1 \
    -L/tmp/canon-llama/build-base/bin -lllama -lggml -lggml-base -Wl,-rpath,/tmp/canon-llama/build-base/bin
  ```

  env `W` (decode batch width), `CTK`/`CTV` (`f16|bf16|q8_0|q4_0|q4_1|q5_0|q5_1|iq4_nl|mxfp4`),
  `SPLIT=layer|tensor`, `NGL`, `RS=0`, **`CB=0` (mandatory)**, `REPEAT=1` (batch = W copies of one
  token); usage `<model> <text.txt> [P=256] [ubatch=512]`; text
  `wip/sm-tensor-plain-vs-spec/p0long.txt`; prints `[L] W=%d logits0_hash=%016llx nv=%d`.
  It creates/removes `/tmp/nodedump_on` around the decode batch.  **Driver**: `rv.sh`
  (`res`/`kv`/`coh`/`mtp`/`bench`/`width`; `BIN=`/`PROBE=` select the build) — see its header.
  `BIN=/tmp/canon-llama/build-base/bin` + `PROBE=/tmp/lw-f1` run against the current tree.
* **Node-dump instrument** (found the F2 divergence): `git apply
  wip/kv-quant-purity-followups/tools/node-dump-instrumentation.patch`, rebuild `ggml-hip`, run with
  `GGML_CUDA_NODE_DUMP=1|2` (+ the probe's `/tmp/nodedump_on`), then diff two widths keyed on
  **`(name, node index)`**.  **Revert it before landing.**  Traps: sync before reading; read the whole
  view address span and gather with the real `nb[]`; auto `node_N` names **shift across widths** (only
  `cb()`-named tensors are comparable); `fdst` lines carry the fused window's dest index.
* **Mandatory controls** (each of these has produced a false positive at least once): a) W=1 must be
  byte-identical to the pre-fix build for every KV type and both splits; b) a same-build null control
  must show zero divergence; c) strip the `[ Prompt: ... | Generation: ... ]` footer **and the ASCII
  banner (it contains the build id)** before hashing any stdout; d) text equality is evidence *for*
  purity, never against; e) never run benches in parallel; f) always pass `--single-turn` to
  `llama-cli`; g) `test-backend-ops` with `HIP_VISIBLE_DEVICES` as needed.
* A `ggml_reshape_3d` contiguity assert (`ggml_is_contiguous(a)`) is the classic symptom of a **strided
  view** reaching a batched consumer; and any band gate must be `nt >= 1` — **reservation-only graphs
  are built with `nt == 0`** (this crashed the F2 first attempt).

## F1 — the measured facts (do not re-derive)

* 4B, **all four split configs** (1 GPU / 2-GPU `-sm layer` / 2-GPU `-sm tensor` / 3-GPU `-sm tensor`),
  `q8_0/q8_0`: `W=1` `0edf55a1` == `W=2`, then `W=3..8` all `31a0c1ba` → **boundary `W=2->3`**, NOT
  block 00's `n_q <= 8`.  `q4_0/q4_0`: `8125e094` -> `619c151e`.
* 27B 3-GPU, ctx 8192, 300 greedy tokens, q8_0 KV: plain `8ed58aa9` (1330 chars) vs
  `n_max 3 == n_max 7` `da56855b` (1406 chars) — real text divergence.  **f16 control**: all three
  `ce7b9a75`.  Both builds identical.
* **Not** the all-reduce (1 GPU reproduces it); **not** block 15's V4 staging
  (`GGML_CUDA_FA_KV_NATIVE` on/off identical).
* Same-type matrix (4B, 1 GPU, W=1,2,3,5,8): **PURE** = f16 `671d6096`, bf16 `b5d7e7b4`, q4_1
  `691484e2`, q5_0 `c1c994dc`, q5_1 `f706e18a`, iq4_nl `05a1955d`; **IMPURE** = q8_0, q4_0;
  `mxfp4` unsupported for KV (context creation fails).  The impure set is **exactly the two types with
  a fast native both-quantized FA path** (pp512 > 7700 vs ~2200 for the F16-staging ones).

### Prime suspects for F1 (start here, in this order)

1. **The native quantized-K/V tile path's own split/plan choice** — a sibling of, or a branch inside,
   `launch_fattn`.  Block 00 fixed exactly this class for the f16 path in
   `ggml/src/ggml-cuda/fattn-common.cuh:1176`:
   `const int ntiles_dst_eff = Q->ne[1] <= 8 ? (ntiles_z_gqa * K->ne[2] * Q->ne[3]) : ntiles_dst;`
   (evaluated as if `n_q == 1` so every small batch agrees — issue #25).  **Check whether that guard
   actually covers the KQ8_0/KQ4_0/VQ8_0/VQ4_0 instantiations** (`FATTN_VEC_CASES_ALL_D(...)` in
   `fattn.cu` ~399-450: `Q8_0/KQ4_0`, `Q4_0/KQ8_0`, `Q4_0/KQ4_0`, `Q8_0/KQ8_0` …) and whether the
   `ncols1`/`ncols2` selection they get (`fattn.cu:147`, `:328` both branch on `Q->ne[1] <= 8`) feeds a
   *different* `parallel_blocks`/`ntiles_x` (`ntiles_x = ceil(Q->ne[1]/ncols1)`,
   `parallel_blocks = min(parallel_blocks, ntiles_KV)`, `fattn-common.cuh:1090/1167`) into the
   online-softmax/PV partial-sum order.  A `W=2->3` boundary is consistent with the tile `ncols1`
   changing at `n_q = 3` (i.e. the *number of Q tiles* changes while the KV split does not agree).
2. The KV-cache **write/quantize** path: if the K/V rows are quantized in groups whose layout depends on
   the batch width, the cached values themselves differ (`quantize_row_q8_0`-style vs a multi-column
   variant).  Test by re-quantizing or by comparing the cache contents across widths.
3. Only then the generic matmul band (see cause 2) — but note the FA path is where the KV type matters.

## F2 cause 2 — the measured facts

* With cause 1 fixed, qwen4exp (f16 KV, P=256, RS=0) is `W=1..4` = the W=1 decode on both splits, and
  still grouped: `-sm layer` `W=5 c999233926f0`, `W=6,7 a8c532e12f9c`, `W=8 c56ebb61963a`;
  `-sm tensor` `W=5 2bfb89f59ec2`, `W=6,7 e8b1253ea93e`, `W=8 a7c5dfd26a56`.
* It **survives `GGML_CUDA_DISABLE_FUSION=1`**, so it is a kernel-dispatch band, not a fusion.
* Prime suspects (the boundary pattern `{5} {6,7} {8}` is suggestive): the **MMVQ/MoE batch bands** —
  `get_mmvq_mmid_max_batch_rdna4(type)` (= **7** for Q4_K, so `>7` switches kernel — matches the
  `{6,7}` vs `{8}` split), `MMVQ_MAX_BATCH_SIZE`, any `ncols_dst > 4` condition, and
  `ggml_cuda_should_use_mmvf` (returns `ne11 <= 3` on fp32-MMA AMD parts).  **Important correction:** the
  `mmvf` widening/disable experiments that appeared to exonerate it in the earlier session were done
  **before** the HC fix (when the W=1 vs W>=2 break dominated) — **re-run them**.  qwen4exp is a *MoE*
  model (512 experts), so the `MUL_MAT_ID` band logic is a strong candidate and it is the same class as
  the block-13 `MUL_MAT_ID`/mmvq fixes.  Start with the node dump at `W=4` vs `W=5` and name the *first*
  divergent node.

## F3 — the design constraint (beta, not delivery)

* Block 15 ships the mechanism: one shared per-operand staging type code
  `FATTN_KV_NATIVE_{NONE,Q8_0,BF16}` in `fattn-common.cuh`, consumed by the launcher, the alloc-size
  query *and* the kernels so the three cannot disagree, behind `GGML_CUDA_FA_KV_NATIVE` (**opt-in,
  default 0**).  V4 (q8_0) costs ~1.7% prefill (the lost `cp_async` pipeline) for -744/-632 MiB/GPU;
  V5 (bf16) 0.2-2.4% for a bf16 cache to cost what an f16 one does.
* Extend it with `Q4_1`/`Q5_0`/`Q5_1`/`IQ4_NL` **width-invariant by construction**, and target
  **`iq4_nl` first**: it is the *same size* as `q4_0` (1800 MiB at ctx 204800/ub2048), is **pure**, and
  is 3.4x slower *only* because it lacks a native path — so a native `iq4_nl` would obsolete `q4_0`
  outright and remove an impure type from the matrix.
* KV size/speed reference (4B, ctx 204800, ub2048, same-type): f16 6400 MiB `7765/99`, bf16 6400
  `7742/99`, q8_0 3400 `7713/97`, q4_0 1800 `7696/95`, q4_1 2000 `2287/64`, q5_0 2200 `2270/56`,
  q5_1 2400 `2197/60`, iq4_nl 1800 `2293/61` (pp512/tg32).  Mixed K/V *types* remain a **rejected**
  configuration (all pairs 1.7-3.6x slower; maintainer decision 2026-09-11).

## Acceptance criteria

* **F1**: `W=1..8` bit-identical for `q8_0/q8_0` **and** `q4_0/q4_0` on **all four split configs**;
  W=1 unchanged vs the current build; all other KV types unchanged (re-run the same-type matrix);
  no prefill/decode regression (interleaved, same binary, never parallel); `test-backend-ops -o
  FLASH_ATTN_EXT` still **7859/7859** on ROCm0 *and* CPU; 27B text-level plain == `n_max 3` == `n_max 7`
  with q8_0 KV (today: `8ed58aa9` vs `da56855b`).
* **F2 cause 2**: qwen4exp `W=1..8` bit-identical, plain == `draft-mtp --spec-draft-n-max 7` greedy
  text, f16 MTP acceptance >= the current 0.76744, perf/reserves unchanged.
* **F3** (if carried): per-type width-purity (same discipline as F1), reserve deltas as documented,
  `GGML_CUDA_FA_KV_NATIVE=0` behaviour byte-identical to today, both `-sm layer` and `-sm tensor`.
* Cross-checks for any change: gemma-4-E4B/31B coherence (**SWA**), MoE asterisk (`ac8825358d9adfda` /
  `bd138ad2326fbbf2`, and `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` -> both `bd138ad2326fbbf2`),
  `test-backend-ops -o GATED_DELTA_NET`, `llama-batched-bench` B=1..8, `--parallel 4` smoke.

## Landing

* **Owner-based block choice** (the rule used for the block-02/12/13/14 amendments): the FA
  dispatch/staging fix belongs to the block that owns those lines — block 00 (the `launch_fattn`
  width-invariance fix, `fattn-common.cuh`) or block 03 (native-BF16 FA + the HIP masked-V kernels),
  whichever owns the code the fix touches; a MoE/matmul-band fix belongs to block 13.  **Do not** put
  it in block 00 just because a previous plan said so — check `git log --diff-filter=A`/`git blame`
  first, and prefer amending a **tip-adjacent or tip** block unless a rebase is genuinely needed.
* Amend flow: `git stash` (if dirty) -> `GIT_SEQUENCE_EDITOR="sed -i 's/^pick <sha>/edit <sha>/'"
  git rebase -i <prev-block-sha>` -> change -> `git add -A && git commit --amend --no-edit` ->
  `git rebase --continue`.  (A `git rebase -i` that refuses to start means unstaged changes exist;
  never `--amend` on the tip by accident without checking `git log -1`.)
* Then: `./scripts/make-patches.sh /tmp/canon-llama 9113cc188 <new tip>`; update the default tip in
  `scripts/make-patches.sh`; refresh `rdna-boosts-all.patch`; **clean-apply sim** (fresh clone at
  `9113cc188` + `scripts/apply-all.sh .` -> strict 15/15, **0 whitespace warnings**, applied tree ==
  the new canonical tree; delete the `rdna-boosts` branch in the clone first).
* **Block 15's beta is coupled**: block 15 *also* touches `fattn-common.cuh` (its `FATTN_KV_NATIVE`
  staging) and `src/models/qwen4exp.cpp`, so **if your fix lands in either file, re-cut block 15**
  (`HANDOVER.md` §10.5: apply, squash, `format-patch --start-number 0` over the full range, copy the
  `0015-…` file, verify the diff is metadata/offset-only, check whether the changed body lines
  actually overlap).  Block 15 stays a beta — do not fold it into `patches/`.
* Docs: new dated `WORKLOG.md` entry at the top; block notes in `patches/README.md`; update
  `GREEDY-PURITY.md` §12/§13 (the KV-type table and the qwen4exp band), `AGENTS.md` (canonical tip/tree +
  the purity critical-fact bullet), `TODO.md`, `wip/kv-quant-purity-followups/README.md`, and the beta
  records.  Never edit dated records in place — add a new dated one.
* Push only to **this repo's `origin`**.  Nothing is ever pushed from `~/llama.cpp`, and nothing from
  `wip/` is folded into `patches/`.

## Do not re-derive (already excluded)

For F2: the whole QSA regime (`LLAMA_QSA_OFF=1`), `LLAMA_QSA_SPARSE_FA=0`, the dense shortcut, the arch
decode policy, graphs, every CUDA-side fusion, batch content (`REPEAT=1`), the all-reduce/tensor split,
and (pre-HC-fix) the float `mmvf` band — the last one must be **re-tested** at `W=4` vs `W=5` now.
Note the graph-side HC knobs are `LLAMA_FUSED_HC_MIX`/`LLAMA_FUSED_HC_COMBINE` (not
`GGML_CUDA_DISABLE_HC_*`), and `HC_FUSED_MAX_TOKENS = 8` in `src/models/qwen4exp.cpp` (asserted in
`hc-mix.cu`) is the band that cause 1 now serves.
