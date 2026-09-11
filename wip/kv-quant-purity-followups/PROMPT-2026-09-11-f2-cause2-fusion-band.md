# NEXT-SESSION PROMPT — F2 cause 2: make the MoE gate+up+GLU fusion cover the decode/verify band (qwen4exp width purity)

> Hand this file (or its path) to the next session.  Written 2026-09-11, immediately after cause 2 was
> localised.  Read the whole thing before touching the GPU — the first two tasks are *measurements* that
> decide which fix is even correct, and one of them corrects an over-confident claim in the commit log.

## Mission

qwen4exp (Qwen3.8-Flash-Next, a dense+MoE hybrid) is **not width-pure above `W = 4`**: a 1-token decode
and an `n_draft + 1`-token speculative verify batch produce different logits, so plain greedy decode
and `--spec-type draft-mtp` disagree.  Cause 1 (the hyper-connection `nt == 1` gates) is **fixed** (the
block-14 amendment).  Cause 2 is localised but **not fixed**: the MoE **gate+up+GLU** path is used at
`n_q <= 4` and abandoned from `n_q = 5`.

Deliverable: a **delivery** fix that restores `W = 1..8` bit-identity for qwen4exp (and therefore
`plain == draft-mtp` up to the designed `--spec-draft-n-max 7`), with the throughput cost measured
rather than assumed.

## Read first

1. `wip/kv-quant-purity-followups/README.md` — **the F2 cause-2 section** (the census table, the
   refutation list, the calibration trick) and the F2 cause-1 record for the fix pattern.
2. `WORKLOG.md` — the entries of **2026-09-11 (3)** (the F1 fix: how a band rule was validated and
   landed) and **(4)** (this localisation).
3. `GREEDY-PURITY.md` §13 (the qwen4exp band matrix) and §14 (F1: the band-rule precedent and the
   harness lessons).
4. `AGENTS.md` — MANDATORY (block-amendment flow, owner-based block choice, pushing policy, the
   coherence gate, the KV-type purity bullet).
5. `benchmarks/mtp-adaptive-methodology.md` — Protocol A, rule 3 (a decode/fusion change must pass the
   adaptive-MTP gate, not just llama-bench).

## Environment and state

* Canonical fork `/tmp/canon-llama` @ **`1bcf4e82d`** (tree **`4104e7d34`**), branch `rdna-boosts`,
  **clean**, 15 blocks; `build-base` built from that tip.  Blocks: 00 `1c7ab0e89`, 01 `aa4108b9d`,
  02 `6e81ed5ed`, 03 `4dc962aa9`, 04 `03d004517`, 05 `70f330aed`, 06 `d2fc2cb34`, 07 `110b5391d`,
  08 `38cffdece`, 09 `3484c378f`, 10 `d60105926`, 11 `e07549b55`, 12 `f6198fbc9`, 13 `1e5580ee9`,
  14 `1bcf4e82d`.  If `/tmp` is gone: clone `ggml-org/llama.cpp`, checkout `9113cc188`,
  `scripts/apply-all.sh .` (15/15 strict, tree `4104e7d34`), build with
  `BUILD_DIR=build-base EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`.
  Fast loop: `cmake --build build-base --target llama-cli llama-bench ggml-hip -j 16`.
* Delivery repo `~/llama-cpp-rdna-boosts`, `main` = **`e7a0a1c`** (pushed).  Block-15 beta: tip
  **`0c8099ca2`** (tree `7335b923d`, base `1bcf4e82d`), patch in
  `beta/block-15-campaign-wins/block-15-campaign-wins.patch`.
* 3x gfx1201 (R9700); ROCm `/opt/rocm-7.14-gfx1201`; `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib`.
* qwen4exp: `/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
  (**`/models/…`, not `/llm/models/…`**; 3 shards ~103 GB; shard 1 is metadata-only) + draft
  `/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`.  MoE, 512 experts, 48 layers.
* **Reference hashes (f16 KV, P=256, RS=0, `CB=0`, `REPLY=1`, current canonical build)**:
  * `-sm layer` (3 GPUs): `W=1..4` `3adeb313042a871b` | `W=5` `c999233926f0` | `W=6,7` `a8c532e12f9c` |
    `W=8` `c56ebb61963a`
  * `-sm tensor` (3 GPUs): `W=1..4` `dcf1ae667f730879` | `W=5` `2bfb89f59ec2` | `W=6,7` `e8b1253ea93e` |
    `W=8` `a7c5dfd26a56`
  * **HC off** (`LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0`, `-sm layer`) = the pre-cause-1
    counterfactual: `W=1..4` `044715b66e72` | `W=5` `bdaa8fc57381` | `W=6,7` `1ffc73e03571` |
    `W=8` `15786ddeffad`.  **Use `W=5` -> `bdaa8fc57381` as the positive control for any env knob**
    (it proves the knob actually reached the process).
* Probe: `wip/kv-quant-purity-followups/tools/logits-dump-kv.cpp` ->
  `clang++ -O2 -std=c++17 -I /tmp/canon-llama/include -I /tmp/canon-llama/ggml/include \
   tools/logits-dump-kv.cpp -o /tmp/lw-f2 -L/tmp/canon-llama/build-base/bin -lllama -lggml -lggml-base \
   -Wl,-rpath,/tmp/canon-llama/build-base/bin`
  env `W` (decode batch width), `CTK`/`CTV`, `SPLIT=layer|tensor`, `NGL=99`, `TS`, `RS=0`, **`CB=0`**,
  `REPEAT=1`; usage `<model> <text.txt> [P=256] [ubatch=512]`; text
  `wip/sm-tensor-plain-vs-spec/p0long.txt`; prints `[L] W=%d logits0_hash=%016llx nv=%d`.
  It creates/removes `/tmp/nodedump_on` around the decode batch (`CB=0` is mandatory).
* Node-dump instrumentation: `tools/node-dump-instrumentation.patch` (apply to `ggml-cuda.cu`, rebuild
  `ggml-hip`, run with `GGML_CUDA_NODE_DUMP=1`) — **revert before landing**.

## The localisation (measured, do not re-derive)

Executed-op census (`[ND]` dump), qwen4exp `-sm layer`, P=256, RS=0:

| width | `ffn_moe_down` | `ffn_moe_up` | total nodes |
|---|---|---|---|
| `W=1..4` | 48 | **0** | 1920 |
| `W=5`   | 48 | **47** | 1967 |
| `W=6,7` | 48 | **48** | 1968 |

Every other op's count is identical at every width.  `W=6` vs `W=7` is a **perfect calibration**
(`+0` nodes, `0` differing ops) — that is *why* they hash identically, and it is what makes the census
a trustworthy signature.  The candidate fusion is
`mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU }` (`ggml-cuda.cu:3324`,
matched at `:3341`, admitted via `ggml_cuda_should_fuse_mul_mat` at `:1852`).  The **47** at `W=5` is a
second, per-type threshold (one layer's expert type takes a different cap — the same
`get_mmvq_mmid_max_batch*` table that gates the other MoE arms).

## TASK 1 (do this first, ~15 min): fusion-applied vs graph-built-with-fewer-ops

The commit log says the `ffn_moe_up` absence is a *fusion skip*.  **That was reasoned, not proven** —
the `[ND]` dump only prints executed nodes, so a node that is *in* the graph but fused away is
indistinguishable from a node the graph never built.  The two need different fixes:

* **fusion applied** -> fix the fusion's *eligibility* (or the band rule) in the CUDA arm;
* **graph built fewer ops** -> fix a *graph-builder* branch (in `qwen4exp.cpp` or upstream
  `llama-graph.cpp`'s MoE builder) — and the `<= 4` will be an explicit condition there.

Cheapest discriminator: print `cgraph->n_nodes` (and, for the MoE layer, whether the `ffn_moe_up` node
exists in the graph before fusing) for each width — e.g. add to the instrumentation, or log
`cgraph->n_nodes` in `ggml_cuda_graph_evaluate_and_capture`.  Whichever it is, **also find the actual
`<= 4`**: instrument the decision (`ggml_cuda_try_fuse` arm / `ggml_cuda_should_fuse_mul_mat` /
`ggml_can_fuse_subgraph` shapes, and the graph-side MoE builder) so it prints `n_tokens` + the shapes
it rejects on, and run it at `W=4` and `W=5`.  `ggml_cuda_should_fuse_mul_mat` itself is purely
structural (no width test), so if it is the fusion arm the `4/5` threshold lives in a *shape/layout*
it is handed — that shape's origin is the thing to find.

## TASK 2 (decides the fix): is the unfused path width-invariant?

Run the width matrix with **all CUDA-side fusions disabled**, on the current (post-cause-1) build:

```sh
for w in 1 2 3 4 5 6 7 8; do
  HIP_VISIBLE_DEVICES=0,1,2 W=$w NGL=99 SPLIT=layer RS=0 CB=0 GGML_CUDA_DISABLE_FUSION=1 \
    timeout 1500 /tmp/lw-f2 <qwen4exp> wip/sm-tensor-plain-vs-vspec/p0long.txt 256 512
done
```

* If all eight widths agree -> the unfused path is width-invariant and the fix is the **F1/HC band
  rule**: keep the gate+up+GLU form consistent across `n_q <= 8` (either always fused or always
  unfused for the band — whichever the `<= 4` gate makes reachable).
* If they still group -> there is a **second residual** under the fusion; localise it with the census
  (calibration = the pair of widths that already agree) before touching anything.
* Record the eight hashes either way; this matrix is the acceptance test's baseline.

**The earlier session's "cause 2 survives all fusions disabled" observation predates the cause-1 fix**,
when the `W=1` vs `W>=2` break dominated those hashes — do not cite it as evidence.

## Acceptance criteria

* qwen4exp `W=1..8` **bit-identical** on both `-sm layer` and `-sm tensor` (the acceptance test, not a
  text comparison).
* plain decode == `--spec-type draft-mtp --spec-draft-n-max 7` greedy text; the `none` run must be
  unchanged vs the pre-fix build **unless** the band rule necessarily moves `W=1` (F1 precedent: if it
  does, move the *cheap* side — the widths that are fewest and least performance-critical — keep the
  verify widths' value stable, and document it; "W=1 unchanged" is not always satisfiable).
* adaptive-MTP gate (f16 KV, n=96, `-c 32768 -ctk q8_0 -ctv q8_0 -f /tmp/prompt3k.txt`): acceptance not
  below the current `0.76744`, and report the throughput (`t/s`) delta — **a fix that restores purity by
  losing the fusion's win should FAIL the gate, not pass it.**
* no prefill/decode regression (interleaved, same binary, never parallel); reserves measured.
* cross-checks: `test-backend-ops -o FLASH_ATTN_EXT` (all pass, 4/4 backends) and `-o GATED_DELTA_NET`;
  the MoE asterisk (`ac8825358d9adfda` / `bd138ad2326fbbf2`, and both `bd138ad2326fbbf2` with
  `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`); dense 27B `q8_0`/f16 purity (`4089b4d4` on 1 GPU,
  `91434ea9` 3-GPU tensor, `d4156dbeb225` with q8_0 KV); gemma-4-E4B / 27B / qwen4exp same-seed
  coherence (SWA models included); `llama-batched-bench` B=1..8; `--parallel 4` smoke.

## Landing

* **Owner-based block choice** (check `git log --diff-filter=A` / `git blame` first, do not guess): the
  CUDA fusion arms live in block 13 (`ggml-cuda.cu` try_fuse, the MoE fusion); a `qwen4exp.cpp`
  graph-builder gate lives in **block 14** (block 14 also owns `hc-mix.cu`, and cause 1 was fixed as a
  block-14 amendment).  Prefer a tip-adjacent/tip block unless a rebase is genuinely needed; if you
  amend block 13, blocks 14/15 replay on top (bodies stay metadata-only — verify that).
* Amend flow: `git stash` (if dirty) -> `GIT_SEQUENCE_EDITOR="sed -i 's/^pick <sha>/edit <sha>/'"
  git rebase -i <prev-block-sha>` -> change -> `git add -A && git commit --amend --no-edit` ->
  `git rebase --continue`.
* Then: `./scripts/make-patches.sh /tmp/canon-llama 9113cc188 <new tip>`; update the default tip in
  `scripts/make-patches.sh`; refresh `rdna-boosts-all.patch` with
  `git -C /tmp/canon-llama diff 9113cc188..<new tip>` (the script does **not** write it); **clean-apply
  sim** (fresh clone at `9113cc188` + `scripts/apply-all.sh .` -> strict 15/15, **0 whitespace
  warnings**, applied tree == the new canonical tree; `git branch -D rdna-boosts` in the clone first).
* **Block 15 is coupled**: it touches `fattn.cu`, `fattn-common.cuh` and `qwen4exp.cpp` — if your fix
  lands in any of those, re-cut the beta (`HANDOVER.md` §10.5: apply, `format-patch --start-number 0`
  over the full range, copy the `0015-…` file, verify metadata/offset-only, check for real body
  overlap) and update the beta records with a *new dated* note.
* Docs: a new dated `WORKLOG.md` entry at the top; the block notes in `patches/README.md`;
  `GREEDY-PURITY.md` §13 (the qwen4exp band matrix) + a pointer in the stale place; `AGENTS.md`
  (canonical tip/tree + the purity critical fact); `TODO.md`; the wip README; the beta records.  Never
  edit dated records in place.
* Push only to this repo's `origin`.  Nothing is ever pushed from `~/llama.cpp`; nothing from `wip/`
  is folded into `patches/`.

## Do not re-derive (refuted by measurement on 2026-09-11)

* the block-13 `get_mmvq_mmid_max_batch` cap / the `use_mmq`-gated MMQ pair arm — forcing MMVQ across
  the band is **byte-identical**, and `should_use_mmq` is false for `n_q <= 8` so that arm never fires
  in the band;
* the MoE expert kernel — `mul_mat_vec_q_moe` is **provably width-invariant** (`rpb` derives from
  `blocks_per_row_x`, a K property; `block_dims = (warp_size, ncols_dst)` = one warp per token);
* `LLAMA_QSA_OFF`, `GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`,
  `GGML_CUDA_DISABLE_WEIGHTED_DOWN`, `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE` — all no-ops for this probe
  (positive control: the HC knob moves `W=5` to `bdaa8fc57381`; **always run a positive control** —
  without it "unchanged" is indistinguishable from "the env never applied");
* `ggml_cuda_should_use_mmvf(F32)` on gfx1201 = `ne11 <= 3`, a 3/4 boundary that does not appear;
* the graph-side cause-1 knobs are `LLAMA_FUSED_HC_MIX` / `LLAMA_FUSED_HC_COMBINE`, **not**
  `GGML_CUDA_DISABLE_HC_*`.

## Instrument traps (each has already produced a false positive)

* `llama-cli`'s `/\\|` spinner is `\\b`-based and timing-dependent, and the ASCII banner embeds the
  build SHA: apply backspaces, strip the banner **and** the `[ Prompt: ... | Generation: ... ]` footer
  before hashing output.  **Run a control that must agree (the f16 pair, or `W=6` vs `W=7`) before
  believing any divergence.**
* Node-dump diffs: auto `node_N` names shift across widths (only `cb()`-named tensors are comparable);
  **shape equality is not sufficient** (cache/state tensors legitimately differ with `W` — use the
  calibration pair instead); sync before reading; gather with the real `nb[]` strides;
  `ggml_backend_tensor_get` ignores `nb[]`.
* `--single-turn` on every `llama-cli` run; `-ts 1/1` is a per-device *weight* list (not device ids);
  never run benches in parallel; `GGML_CUDA_ALLREDUCE=nccl` is **not** a bit-identical reference under
  `-sm tensor`; `-fa off` cannot be used as a control with a quantized V cache.
* Probe runs are heavy (~2-3 min load each) — decide the widths you need *before* launching a matrix,
  and reuse the census/calibration logs instead of re-dumping.
