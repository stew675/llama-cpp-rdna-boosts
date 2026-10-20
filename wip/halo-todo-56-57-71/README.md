# halo — architecture-independent fixes found while scoping TODO #56/#57/#71

**Status: OPEN (2026-10-20) — three fixes rebased onto r96 and validated; #56 reproduced; #57 does not
reproduce here.**  Not part of the delivery; the patch is for the main development box to integrate
(fold each commit into its owning block).  Nothing here is pushed or applied by `apply-all.sh`.

Box: `halo` — gfx1151 (Radeon 8060S), **1 device**, 123 GiB unified, ROCm 10.1.0, **r96** tree.
Release: **`v12-097cf33d9-r96`** (base `097cf33d9`, canonical tip `2fc3b2f53`, tree
`cad83cc3…`); the fork's `origin/rdna-boosts` is `a0bddc9d1` with the same tree.
Branches: delivery `halo/todo-56-57-71` (rebased onto `origin/main` = `461edf24`); fork
`~/llama.cpp` `halo/todo-56-57-71-r96` (3 commits on `origin/rdna-boosts`).  The main box's `main` is
untouched by this work.

Cold start: `AGENTS.md` (report-before-shipping, WIP rules), then this file.

## The patch (for the main box)

Three commits on top of the r96 tree (fork `halo/todo-56-57-71-r96`, commits `eebdc2f30`, `6537f51d4`,
`b99817258`).  `git format-patch` series and a combined diff are in this directory:

| file | sha256 |
|---|---|
| `0001-halo-MoE-host-expert-accounting-must-not-key-a-devic.patch` | `8a677c98e621b2634d83195d00b702196b6b33205bb879abb09b21495ff9bbda` |
| `0002-halo-gate-the-QSA-derived-visibility-form-on-the-act.patch` | `3d0494a02c6b94e52ab363f248dd4c0833640da6e44ea3e5824247559a593748` |
| `0003-halo-the-rollback-narrow-reserve-must-use-the-real-s.patch` | `18718dd2bd7c3ab8d2210d7ab0287a76ed2c3866075d25c5db4aac4f1d0c7f87` |
| `halo-arch-fixes.patch` (all three, `git diff`) | `d452cf2898b01abd67af5f7134851a80baf2bcc09f9155d2612cea8ddaf8fd5c` |

Apply to a checkout of the r96 tree (or fold per block):

```bash
git am /path/to/000{1,2,3}-halo-*.patch
# or, as one diff:
git apply /path/to/halo-arch-fixes.patch
```

Owning blocks: **0001 → block 09** (`llama-model-loader.cpp`), **0002 → block 10**
(`ggml-cpu/ggml-cpu.cpp` + `src/models/qwen4exp.cpp`), **0003 → block 10**
(`src/llama-context.cpp`; the rollback/narrow reserve is r91 block-10 code).

## H1 — CPU-only MoE aborted at context creation (FIXED, commit 0001)

`moe_host_expert_account()` (block 09) did `acc[buft_dev] += bytes` unconditionally.  The plain CPU
buffer type has `.device = NULL` **upstream** (`ggml-backend.cpp`, the `FIXME` on the CPU buft), so a
CPU-only model keyed a null device and `llama_context` aborted in `ggml_backend_dev_memory(kv.first, …)`
(`src/llama-context.cpp:765`, the ring auto-size).

* This is **ours** (block 09 / `wip/fit-slab-accounting`, r38), not upstream and not a misconfiguration:
  upstream never keys that map, and a CPU-only build/run is supported.
* Fix: fall back to the layer's real device (the test harness's host-expert override is exactly
  CPU-buft-on-a-GPU-layer), and skip when there is no device at all.
* Verified: CPU-only `gpt-oss-20b -ngl 0` generates; `test-llama-archs -a qwen4exp` CPU-only runs
  (`Mixed batch 3.32e-14`); 35B-A3B `-ncmoe 20` is unaffected.

Repro (pre-fix):

```bash
HIP_VISIBLE_DEVICES= ./build-rocm/bin/llama-cli \
  -m /llm/models/gpt-oss/20B/gpt-oss-20b-MXFP4.gguf -ngl 0 -n 4 \
  -p "The capital of France is" --single-turn --no-display-prompt
# /ggml/src/ggml-backend.cpp:921: GGML_ASSERT(device) failed
```

## H2 — qwen4exp CPU/derived-visibility abort (FIXED, commit 0002)

`test-llama-archs -a qwen4exp` aborted at the first GPU row in the **CPU** QSA reference:

```
ggml/src/ggml-cpu/ops.cpp:9653: flash_attn_qsa: the derived cell visibility requires the CUDA backend
```

The sparse path allocated `cell_vis`/`q_vis` (`src[5]`/`src[6]`) whenever `derived_bias` was on, even on
a CPU layer; the CPU reference implements the mask form only.  Fixed by:

* CPU `supports_op` refuses the derived form (`op->src[5] == nullptr`) — it is the truth, there is no CPU
  kernel for it;
* `qsa_derived_vis_supported()` probes the form the graph will actually build, and `qwen4exp` allocates /
  takes the derived path only when the acting device (or every child of a Meta split) accepts it;
  otherwise the dense masked path.

Verified gfx1151: `test-llama-archs -a qwen4exp` all rows pass (GPU 1.40e-13, CPU 0.00e+00, Meta
1.40e-13); `test-backend-ops -o FLASH_ATTN_QSA` passes (the CPU oracle still serves the mask form).

## H3 — deepseek2 graph build aborted on the reserve stream count (FIXED, commit 0003)

```
ggml/src/ggml.c:5540: GGML_ASSERT(q->ne[3] % mask->ne[3] == 0) failed
  llama_model_deepseek2::graph::graph -> get_logits
```

The r91 rollback/narrow reserve used `ubatch.n_seqs` as the sequence count.  For a 64-token
single-sequence ubatch that is 64 (`n_tokens/n_seq_tokens`) while the real count (`n_seqs_unq`) is 1, so
the synthetic reserve ubatch built a 64-stream 4-D kq mask against a single-stream query.  Use
`ubatch.n_seqs_unq` for the reserve.

The one exception is the **dsv4 compressed cache** (`LLM_ARCH_DEEPSEEK4`, and DFLASH with
`dsv4_hc_mult > 0`): its raw attention needs the per-stream index fanout shape, and a one-stream reserve
makes the next real decode abort in `cpy_k`.  It keeps `ubatch.n_seqs` (exactly the r95 behaviour); a
clean follow-up is a side-effect-free reserve for that cache.

This is confirmed a delivery regression: the upstream base `097cf33d9` passes `test-llama-archs -a
deepseek2` (CPU and all GPU rows), and H1-only + H2 still pass it — only the reserve change differs.

## #56 is now reachable, and reproduces on gfx1151

With H3 fixed, `test-llama-archs -a deepseek2 -s 42` runs its rows and reaches the `MoE cache` config,
which faults exactly as `TODO.md` #56 describes:

```
HSA_STATUS_ERROR_MEMORY_FAULT
kernel: void mul_mat_vec_f<float, float, 1, 64, false, true>(...)
faulting addr: 0x7f… (a HOST/pinned range)
```

GPU rows 1-4 (GPU, CPU, Meta skip, Host experts) pass; `Host experts` NMSE 1.36e-09.  So **#56 is
reproducible on a single gfx1151** once H3 is in; it is not gfx1201-specific.  This patch deliberately
does **not** touch #56 (that is the next TODO item).

## #57 does not reproduce here

`test-llama-archs -a qwen4exp -s 42` CPU-only gives `Mixed batch 3.32e-14 (OK)`, matching the upstream
base value in `TODO.md` #57 (`0.00e+00` with `GGML_QSA_DERIVED_VIS=0`).  So #57's r50 failure
(`3.52e-01`) is either fixed since r50 or gfx1201/CPU-ISA-specific.  Do not close #57 on this alone.

## Test matrix

`test-llama-archs -s 42` full sweep on gfx1151 **r96**: **373/373 passed** with the deepseek2-only
host-expert / MoE-cache rows skipped (`ARCHS_SKIP_HOST_EXPERTS=1` was a local instrument, not in the
patch) — the only remaining known failure in the sweep is #56 itself.  Re-validated after the r96
rebase: qwen4exp 3/3, deepseek4 3/3, deepseek2 CPU 1/1, CPU-only `gpt-oss-20b -ngl 0` generates, and
deepseek2's `MoE cache` row still reproduces the #56 fault.

## Main-box integration

A ready-to-paste prompt for the main build machine is [`MAIN-BOX-PROMPT.md`](MAIN-BOX-PROMPT.md).
In short: fold 0001 into block 09, 0002 and 0003 into block 10, regenerate with
`scripts/make-patches.sh` + `scripts/make-release.sh`, and report any gfx1201 difference before shipping.
No behaviour change on gfx1201 is expected for 0001/0002 (CPU-only / probe), and 0003 only changes the
synthetic reserve shape (dsv4 keeps its existing shape).

## Next

1. Main box: integrate the three commits (see the prompt).
2. Then back to the TODO items on this box: #56 (now reproducible), #71 (independent), #57 re-checked on
   gfx1201.
