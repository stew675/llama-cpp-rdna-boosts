# halo — TODO #56/#57/#71 on the gfx1151 box

**Status: OPEN (2026-10-20), investigation + blocker fixes.**  Not part of the delivery; nothing here is
applied to the fork/`patches/` without the maintainer's go-ahead (see the WIP rules in `AGENTS.md`).

Box: `halo` — gfx1151 (Radeon 8060S), **1 device**, 123 GiB unified, ROCm 10.1.0, r95 tree.
Branches: delivery `halo/todo-56-57-71`; fork `~/llama.cpp` `halo/todo-56-57-71` (one commit on top of
the r95 tree `5129694…`).  The main development box's `main` is untouched.

Cold start: `AGENTS.md` (report-before-shipping, WIP rules), `TODO.md` #56/#57/#71, then this file.

## H1 — CPU-only MoE aborted at context creation (FIXED on the work branch)

`moe_host_expert_account()` (block 09, `src/llama-model-loader.cpp`) did `acc[buft_dev] += bytes`
unconditionally.  The plain CPU buffer type has `.device = NULL` **upstream** (`ggml-backend.cpp`, the
`FIXME` at the CPU buft), so a CPU-only model keyed a null device and `llama_context` aborted in
`ggml_backend_dev_memory(kv.first, …)` (`src/llama-context.cpp:765`, the ring auto-size).

* This is **ours** (block 09 / `wip/fit-slab-accounting`, r38), not upstream and not a misconfiguration:
  upstream never keys that map, and a CPU-only build/run is a supported configuration.
* Fix on the branch: fall back to the layer's real device (the test harness's host-expert override is
  exactly CPU-buft-on-a-GPU-layer), and skip when there is no device at all.
* Verified: CPU-only `gpt-oss-20b -ngl 0` generates; `test-llama-archs -a qwen4exp` (`HIP_VISIBLE_DEVICES=`)
  now runs its CPU row (`Mixed batch 3.32e-14`, OK); a 35B-A3B `-ncmoe 20` run is unaffected
  (host-expert accounting still keyed to the device buft).

Repro (pre-fix):

```bash
HIP_VISIBLE_DEVICES= ./build-rocm/bin/llama-cli \
  -m /llm/models/gpt-oss/20B/gpt-oss-20b-MXFP4.gguf -ngl 0 -n 4 \
  -p "The capital of France is" --single-turn --no-display-prompt
# /ggml/src/ggml-backend.cpp:921: GGML_ASSERT(device) failed
```

## H2 — qwen4exp GPU row aborts in the CPU QSA reference (OPEN)

`test-llama-archs -a qwen4exp -s 42` (GPU present) aborts at the first GPU row:

```
ggml/src/ggml-cpu/ops.cpp:9653: flash_attn_qsa: the derived cell visibility requires the CUDA backend
```

The block-10 CPU reference implements the mask form but not the derived-visibility form (`src[5]`), and
the scheduler places that op on the CPU on this path.  `GGML_QSA_DERIVED_VIS=0` makes the whole qwen4exp
matrix pass on gfx1151 (GPU 1.40e-13, CPU 0.00e+00, Meta 1.40e-13), so the trigger is exactly the derived
form, not the arch.  Candidate fixes: have the CPU backend's `supports_op` reject the derived form
(`src[5] != nullptr`) so the op stays on the GPU, or implement the derived form in the CPU reference.

## H3 — deepseek2 arch test aborts at graph build (OPEN)

`test-llama-archs -a deepseek2 -s 42` aborts before any row, on CPU-only **and** GPU:

```
ggml/src/ggml.c:5540: GGML_ASSERT(q->ne[3] % mask->ne[3] == 0) failed
  llama_model_deepseek2::graph::graph -> get_logits
```

This is in the shared `build_attn_inp_kq_mask` / FA stream reshape (`src/llama-graph.cpp:36-48`, block 05),
not in a kernel.  It prevents #56's host-expert / MoE-cache rows from ever running (they are deepseek2-only
in the harness), so **#56 is blocked behind H3** on this tree.

## Result on #57

The **CPU row now passes on gfx1151/r95**: `test-llama-archs -a qwen4exp -s 42` CPU-only gives
`Mixed batch 3.32e-14 (OK)`, matching the upstream-base value in `TODO.md` #57, and with
`GGML_QSA_DERIVED_VIS=0` the CPU row is `0.00e+00 (OK)`.  So #57's r50 failure **does not reproduce** on
this box/tree — it is either fixed since r50 or gfx1201/CPU-ISA-specific.  Do not close #57 on this
evidence alone; re-run it on the gfx1201 box, or bisect it here against r50.

## Next steps

1. Bisect H2 and H3 per block (`scripts/gate-per-block-build.sh` for the builds, then the single-arch
   `test-llama-archs` run) — both are quick aborts and need no model weights.
2. Re-test #56 once H3 is fixed (deepseek2 + `MoE cache` config, 1536 KiB cache).
3. #71 (shared tile K/V type selection) is independent of all of the above; gfx1151 prefill-logit bases
   already exist in `~/.cache/rdna-boosts/gates/`.
