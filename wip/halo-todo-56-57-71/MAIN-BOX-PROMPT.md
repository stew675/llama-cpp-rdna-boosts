# Prompt for the main build machine — integrate the halo arch-independent fixes

Paste the block below into the agent on the main development box.  It assumes the delivery repo
(`llama-cpp-rdna-boosts`) is the working repo and `main` is at **r96** (`461edf24`,
`v12-097cf33d9-r96`, base `097cf33d9`, canonical tip `2fc3b2f53`, tree `cad83cc3…`).

---

You are on the main development box.  The gfx1151 (`halo`) box produced a **3-commit,
architecture-independent fix set** while scoping TODO #56/#57/#71.  It is rebased onto r96 and validated
on gfx1151; integrate it into the delivery.

**Source of the changes**

Delivery repo branch `halo/todo-56-57-71` (rebased onto `main` = `461edf24`) — the three commits and
their patch files are under `wip/halo-todo-56-57-71/`:

| file | sha256 |
|---|---|
| `0001-halo-MoE-host-expert-accounting-must-not-key-a-devic.patch` | `8a677c98e621b2634d83195d00b702196b6b33205bb879abb09b21495ff9bbda` |
| `0002-halo-gate-the-QSA-derived-visibility-form-on-the-act.patch` | `3d0494a02c6b94e52ab363f248dd4c0833640da6e44ea3e5824247559a593748` |
| `0003-halo-the-rollback-narrow-reserve-must-use-the-real-s.patch` | `18718dd2bd7c3ab8d2210d7ab0287a76ed2c3866075d25c5db4aac4f1d0c7f87` |
| `halo-arch-fixes.patch` (all three, `git diff`) | `d452cf2898b01abd67af5f7134851a80baf2bcc09f9155d2612cea8ddaf8fd5c` |

The fork branch is `halo/todo-56-57-71-r96` on top of `origin/rdna-boosts` (`a0bddc9d1`, tree
`cad83cc3…`); commits `eebdc2f30`, `6537f51d4`, `b99817258`.  If the branches are not fetchable on this
box, use the patch files above — they apply to the r96 tree (`git am` verified from a clean r96
worktree; the result tree is `3b13ee89…`).

**What each commit does and where it belongs**

1. **`0001` — CPU-only MoE abort** (`src/llama-model-loader.cpp`, **block 09**).  `moe_host_expert_account()`
   keyed `acc[buft_dev]` unconditionally; `ggml_backend_cpu_buffer_type()` has `.device = NULL` upstream,
   so a CPU-only model inserted a null device and `llama_context` aborted in
   `ggml_backend_dev_memory()`.  Fall back to the layer's device and skip when there is none.  This is
   our bug (the map is from `wip/fit-slab-accounting`), not upstream and not a misconfiguration.
2. **`0002` — QSA derived-visibility gating** (`ggml/src/ggml-cpu/ggml-cpu.cpp`,
   `src/models/qwen4exp.cpp`, **block 10**).  The CPU reference implements the mask form of
   `FLASH_ATTN_QSA` only; the sparse path allocated `cell_vis`/`q_vis` (`src[5]`/`src[6]`) even on a CPU
   layer and aborted.  CPU `supports_op` now refuses the derived form, and `qwen4exp` probes the form it
   will actually build (`qsa_derived_vis_supported`), falling back to the dense masked path when the
   device cannot run it.
3. **`0003` — rollback/narrow reserve stream count** (`src/llama-context.cpp`, **block 10**).  The
   synthetic reserve used `ubatch.n_seqs` (a shape value: 64 for a 64-token single-sequence ubatch) as
   the sequence count, so it built a 64-stream 4-D kq mask against a single-stream query and deepseek2
   aborted in `ggml_flash_attn_ext`.  Use `ubatch.n_seqs_unq`.  **The dsv4 compressed cache is kept on
   `ubatch.n_seqs`** (its raw attention needs the per-stream fanout shape; a one-stream reserve makes the
   next real decode abort in `cpy_k`) — that preserves its current behaviour exactly.  A
   side-effect-free reserve for that cache is a documented follow-up, not part of this set.

**Integration method**

- Follow `AGENTS.md`: canonical fork rebuilt at `release.json.base` (`097cf33d9`) with
  `scripts/apply-all.sh`, then fold each commit into its owning block (fixup + autosquash or an
  interactive rebase).  Do **not** hand-edit the committed patches and do **not** leave this as a
  `wip/` overlay in the delivery.
- Regenerate with `scripts/make-patches.sh` (and `scripts/make-release.sh` if this becomes a release).
  `git am` the series from a clean r96 worktree first if you want a pre-fold check.
- Keep the `wip/halo-todo-56-57-71/` record on the delivery branch until the integration is confirmed,
  then move it to `archive/work/halo-todo-56-57-71/` / `WORKLOG.md` per the WIP rules (no stub).

**Validation required before shipping**

- `scripts/validate-set.sh` green; every block still builds (`scripts/gate-per-block-build.sh`) if a
  block commit is amended.
- `test-llama-archs -a qwen4exp -s 42` and `-a deepseek4 -s 42`: all rows pass.  On gfx1201 confirm the
  qwen4exp derived-visibility path is still taken (the probe returns true; the GPU row uses the sparse
  derived path, not the dense fallback).
- `test-backend-ops -o FLASH_ATTN_QSA` passes (the CPU oracle still serves the mask form).
- CPU-only: `HIP_VISIBLE_DEVICES= llama-cli -m <MoE gguf> -ngl 0 -n 4 …` generates.
- `test-llama-archs -a deepseek2 -s 42` reaches the `MoE cache` row and **still faults** — that is
  **TODO #56**, which this set deliberately does not fix (it now reproduces on one gfx1151 too).  Do not
  report it as a regression from this patch; the graph-build abort it used to hit is what `0003` fixes.
- If `0002`/`0003` touch a prefill kernel path in your fold, run `scripts/gate-prefill-logits.sh` as
  usual.
- Run the gates on gfx1201; gfx1151 evidence is in `wip/halo-todo-56-57-71/README.md` (full sweep
  373/373 with the #56 rows skipped).

**Report before shipping**

Per `AGENTS.md`, surface any gfx1201 difference, conflict resolution or review finding to the
maintainer and wait for a decision before the delivery-affecting commit / tag.  Only the delivery repo
and the personal fork may be pushed; never upstream `ggml-org/llama.cpp`.
