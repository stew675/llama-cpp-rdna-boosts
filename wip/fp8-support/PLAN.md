# PLAN — revive native FP8 E4M3 on the current delivery base

Follow top to bottom.  **Every gate must pass before the next phase.**  New measurements go in a
dated `RESULTS-YYYY-MM-DD.md` next to this file (never edit `MEASUREMENTS.md`).

Work happens in a **fresh worktree/clone**, not in the shared `~/llama.cpp` (which carries
`rdna-boosts` + the `beta-integration` worktree).  `~/cllm` is the FP8 source of truth; keep it
untouched and re-base into a new tree.

---

## Phase 0 — Prepared  ✅ (2026-09-26)

- [x] `wip/fp8-support/` with the 23-commit series (`patches/`), the net diff, the reference docs
- [x] frozen baseline + AITER reference (`MEASUREMENTS.md`)
- [x] provenance pinned: base `6ea215d17`, tip `cllm` `7c17faffc` (`~1` unpushed commit from `origin/cllm`)

## Phase 1 — Build `~/cllm` and reproduce the 4B result  **(gate 1)**

- [ ] build `~/cllm` for gfx1201 (`-DVLLM_CPP_HIP`-equivalent here: the repo's own Hip build flags;
      see `reference/HANDOVER.md` for the exact box rules)
- [ ] `llama-bench -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 0 -dev <free>` on a
      **free** GPU; Q8_0 control on the same box
- [ ] **gate 1: fp8 pp512 ≥ 7184** (the L7/L9 figure) and clearly ahead of the Q8_0 control
- [ ] read `reference/LEVERS.md` §0 for the exact per-pp512 kernel-time breakdown to compare against

## Phase 2 — Re-base onto the current base  **(gate 2)**

Apply the 23-commit series **curated**, not blindly.  `git am` will conflict because the GDN / ssm-conv
work largely **already landed** as delivery block 02 and the current `mmul`/`mmq`/FA code moved on.

- [ ] create a fresh clone/worktree of the fork at the current base (`84e76d8a2` + the delivery
      patches, i.e. the `a3dc4bbb…` tree)
- [ ] `git am wip/fp8-support/patches/*.patch`, resolving each conflict; **drop** commits whose
      content is already in the delivery (expected: `0009`-`0014`, the chunked-GDN series)
- [ ] expected conflict files: `ggml/src/ggml-cuda/gated_delta_net.cu`, `ssm-conv.cu`, `ggml-cuda.cu`,
      `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/ops.cpp`, `convert_hf_to_gguf.py`/`conversion/*`
- [ ] build clean; delivery gates green (`test-backend-ops`, same-seed greedy text for the int8 models)
- [ ] **gate 2: fp8 4B pp512 still ≥ 7184** on the re-based tree (the MMB/GEMM work must not regress it)

## Phase 3 — The 27B: convert, measure  **(gate 3 — the number the maintainer wants)**

- [ ] convert `/llm/models/Qwen3.8/27B/FP8/` → `F8_E4M3` GGUF
      (`convert_hf_to_gguf.py … --outtype fp8_e4m3`)
- [ ] verify the HF fp8 weight-scale → `block_f8_e4m3` mapping (per-tensor/per-channel → 128-block);
      this is the correctness risk of the conversion
- [ ] `llama-bench -p 8192 -n 0` on gfx1201; compare against
      `../prefill-gap-attribution/MEASUREMENTS.md` §1 (Q8_0 1371 / Q6_K 975 / Q4_K_XL 1254)
- [ ] **gate 3: 27B fp8 pp8192 beats the int8 Q8_0 baseline (1371 t/s)**
- [ ] kernel-trace the 27B fp8 prefill (`../prefill-gap-attribution/tools/attribution.sh`): confirm
      `mul_mat_fp8_wmma` is the same ~50 % share and note its TFLOP/s on the 27B shapes

## Phase 4 — Lift AITER's gfx1201 fp8 GEMM tuning  **(gate 4)**

- [ ] transcribe the relevant `aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE*.json`
      tiles / `GROUP_SIZE_M` / `kpack` / `M_LEQ_x` selection into `mul_mat_fp8_wmma`
- [ ] cross-check our Qwen3.8-27B prefill fusions against ActiveFPX PromptForge's (fused gate/up,
      SwiGLU-to-down packing, merged QKV/Z) before writing any new fusion work — `ROCmFPX-ASSESSMENT.md` §4
- [ ] A/B each change with rocprof kernel times (not `llama-bench`); **do not** adopt `bpreshuffle`
- [ ] **gate 4: `mul_mat_fp8_wmma` ≥ 110 TFLOP/s effective** on the large 27B shapes (AITER 121-137)

## Phase 5 — Quality, then promote or archive

- [ ] PPL against the vLLM fp8 oracle (the cllm record: fp8 6.2250 vs Q8_0 6.2464) and the same-seed
      greedy gate; fp8 is **not** bit-identical to any int8 path, so the purity contract is a
      PPL/oracle comparison, not a text hash
- [ ] decide the delivery shape: new `GGML_TYPE_F8_E4M3` + kernels + loader + convert is a large
      block; a maintainer call on whether it is delivery material
- [ ] if all gates pass: write it up, ask for promotion; else: record the negative result and archive

---

## Risks

1. **Re-base cost.**  7 weeks of drift and ~44 files; the GDN/ssm-conv overlap is real.  Budget the
   re-base as the largest single task.
2. **The conversion is the correctness cliff.**  HF fp8 (per-tensor/per-channel weight scales,
   dynamic per-token activation) → the cllm 128-block `block_f8_e4m3` grid.  A wrong scale mapping
   produces plausible-but-wrong text; gate on PPL + an oracle, not on "it runs".
3. **Precision honesty.**  FP8 is a *different numerics* path.  It does not need bit-identity with
   int8, but it does need to match vLLM-fp8's tokens within the recorded tolerance.
4. **The 4B result may not scale.**  The +13-17 % was on a 4B GDN model; the 27B has different
   shapes, a bigger lm_head, and more attention layers.  Phase 3 is the real test.
5. **The delivery's MMB work may already overlap.**  If block 08/13's `mmb` fused-bf16 path is close
   to the fp8 WMMA efficiency, the marginal FP8 win is smaller than hoped; measure, do not assume.
6. **Don't bench on an occupied GPU** — llama-bench silently CPU-offloads (see `MEASUREMENTS.md`).
7. **"Dequantize to bf16 + hipBLAS" is a trap.**  Upstream already measured that on **RDNA4 MMQ beats
   dequantization + hipBLAS** (`ggml-cuda/mmq.cu:608`, PR #18537).  The fp8 win must be a **native
   fp8 WMMA GEMM** (`mul_mat_fp8_wmma`), not a dequant pipeline.  See `ROCmFPX-ASSESSMENT.md` §2.

## Reference commands

```bash
# the FP8 source of truth
git -C ~/cllm log --oneline 6ea215d17..cllm        # the 23 commits
ls  wip/fp8-support/patches/                        # the same series, ready for git am
# the attribution harness (sibling campaign)
wip/prefill-gap-attribution/tools/attribution.sh /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf 8192
```
