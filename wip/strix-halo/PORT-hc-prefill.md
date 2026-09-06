# Port B's prefill hyperconn fusions into A — implementation brief (2026-09-05)

Status: reference files copied into A; CODE SPLICE NOT DONE. See the WS3
answer + numerics analysis in notes-ws1-survey.md. Plan file:
~/make-strix-halo-faster.md (WS4 section is now the implementation owner).

## What is already done in A (~/llama.cpp, branch qwen4exp)

- Files copied (from B c7af5c6c2, VERBATIM): 
  ggml/src/ggml-cuda/hyperconn.cu and hyperconn.cuh
  (originals also saved in wip/strix-halo/port-ref/).
- The HIP build globs ggml/src/ggml-cuda/*.cu, so a full rebuild via
  ~/bin/build-llama-rocm-714 picks hyperconn.cu up with NO CMake edit.

## What remains (in order)

1. SPLICE into ~/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:
   a. Add `#include "ggml-cuda/hyperconn.cuh"` next to the existing
      `#include "ggml-cuda/dsv4-hc.cuh"`.
   b. Copy B's `ggml_cuda_match_hc_mix(...)` (B ggml-cuda.cu ~line 3529,
      saved at wip/strix-halo/port-ref/B-match-hcmix.txt, starts at its
      def) in just ABOVE `ggml_cuda_try_fuse`.
   c. Insert B's two try_fuse pattern blocks into A's `ggml_cuda_try_fuse`
      (A ~line 3537), mirroring B's placement (elementwise section, after
      the rms_norm/ssm fused patterns, BEFORE the mul_mat/moe fusion
      region; A anchors in that zone are near line ~3720-3790). Blocks:
      - hc stream-mix reduction (B lines ~4736-4760; saved in
        port-ref/B-tryfuse-hcmix.txt). No anchor op guard: call
        ggml_cuda_match_hc_mix(cgraph, i, args, ops) for every node; skip
        when args.dst->ne[1] == 1 (decode: "Decode preserves bit-identical
        routing. Prefill uses the fused reduction."); then the
        ggml_can_fuse_subgraph_ext + ggml_cuda_op_hc_mix_reduce launch.
      - hc combine + next grouped rms_norm (B lines ~4760-4880; saved in
        port-ref/B-tryfuse-hccomb.txt). Anchor: node->op == GGML_OP_SCALE
        && F32 && next is sigmoid && next-next is SCALE. Includes the
        view-skip scan, shape/alias checks, single_block fallback for
        n_tok==1 && hc<=4, ggml_cuda_hc_combine_norm_supported() gate, and
        ggml_can_fuse_subgraph_ext with out_nodes {j+2, j+4}.
      - NOTE B also has a plain hc_combine block (scale->sigmoid->scale->
        repeat->mul->add, no norm) at B ~4880+ (saved in
        port-ref/B-tryfuse-tail.txt). A's model already emits the add whose
        consumer is the next hc_mix rms_norm, so the combine+norm variant
        is the one that fires in pp; port the plain combine block only if
        the +norm matcher does not cover all combines (verify by counting
        fired kernels vs expected 2 per hc combine site).
   d. Add an env opt-out for A/B testing, following A's idiom
      (GGML_CUDA_DISABLE_FUSION / GGML_CUDA_DISABLE_MOE_MMQ_FUSION):
      static env GGML_CUDA_DISABLE_HC_FUSION gating BOTH new blocks.
2. BUILD: cd ~/llama.cpp && ~/bin/build-llama-rocm-714 (~10 min, full
   reconfigure picks up hyperconn.cu). Fix any API drift vs A's
   common.cuh/backend (same upstream base, expect few or none).
3. VERIFY KERNELS FIRE: repeat the rocprof pp2048 run (see
   notes-ws1-survey.md for the command) and check
   hc_mix_reduce_f32 (~190 calls) + hc_combine_norm_f32 (~188 calls)
   appear and the elementwise/norm/cpy category collapses from 4102
   dispatches toward B's ~1600. Note: only if the graph op sequence
   matches (A's build_hc_mix/combine emit the same ggml calls as B's;
   any mismatch shows up here as zero fired kernels -> then adapt the
   matcher to A's exact chain from the graph dump).
4. BIT-EXACT GATE (the point of the numerics analysis): fused vs unfused
   (GGML_CUDA_DISABLE_HC_FUSION=1) must produce IDENTICAL logits on a
   LONG prompt, not just identical 20-token text. Use llama-eval-callback
   or a small harness comparing final logits; B's kernels replay the
   unfused rounding (RN non-FMA asm) so this should pass bit-exact. If a
   sub-chain is off by ulps, do NOT relax - fix the kernel expression.
5. COHERENCE: llama-cli same-seed (A AGENTS command) with fusion default-on
   vs off -> identical text.
6. PP MEASUREMENT: repeat the clean depth-0 ladder (bench-A/B-pp0-clean.md
   protocol, pp512..16384, r3, warm-clock order) on the new build; expect
   pp2048@0 ~348 -> +10-18% and more at pp512/1024 (launch-count win).
   Watch tg@0 and at-depth tg (decode must not regress - fusion is
   prefill-only by the ne[1]==1 guard, so expect no change).
7. RECORD: dated bench file wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-*.md + notes
   update; DELIVERY ROUTING per AGENTS (qwen4exp tree change -> beta
   patch or block amendment; NEVER push from ~/llama.cpp).

## Semantics recap (kernel math, from B hyperconn.cu)

- hc_mix_reduce: mixed[e,t] = scale*sum_c xn[c*n_embd+e,t]*sigmoid(gate[...])
  + bias; streams summed 0..hc-1; hc_mul_rn/hc_add_rn asm (no FMA);
  hc_sigmoid = 1/(1+expf(-x)) (== op_sigmoid).
- hc_combine_norm: per (stream c, token t) block of 1024 threads:
  w = s2*sigmoid(s1*inject[c]+b1)+b2; res = residual + block_out*w
  (RN mul/add); then rms_norm(res row) with block_reduce<SUM,1024> tree ==
  rms_norm_f32<1024>, scale=rsqrtf(mean+eps), xn = scale*res*gamma[c].
- Args from op params: s1/b1/s2/b2 = the two SCALE nodes' (scale,bias)
  params; eps from rms op_params. For A's model: first SCALE is 1/hc with
  bias 0, second is 2.0 (A's combine: sigmoid(scale(inject,1/hc)) then
  scale(·,2)) -> s1=1/hc,b1=0,s2=2,b2=0. mix: scale=1/hc (A's final
  ggml_scale(1/hc)), bias=0. These come from op params automatically.
- Aliasing/staging logic + single_block variant must be kept (graph
  allocator reuses the block_out buffer for xn etc.).

## Risks / watch-items

- A's build_hc_mix generic chain vs B's differ slightly BEFORE the sigmoid
  anchor (A: reshape BEFORE mul(w_norm); B: weight reshape + mul adjacent,
  reshape after) - irrelevant to the mix matcher (it only needs xn, gate,
  and the sigmoid->mul->reshape->view->cont->adds->scale tail). A's tail is
  identical to B's. The combine+norm matcher needs add->rms_norm adjacency;
  if A's graph interposes nodes, adapt with a small skip scan like B's
  view-skip (verify empirically via kernel counts, step 3).
- Env gate default: mirror B (fusion ON by default once verified) but keep
  GGML_CUDA_DISABLE_HC_FUSION for A/B.
