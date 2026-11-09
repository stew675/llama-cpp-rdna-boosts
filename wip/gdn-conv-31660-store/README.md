# wip/gdn-conv-31660-store — the GDN conv fusion on upstream #31660's rollback store, and the DFlash layer-input copies

Written by Claude (AI).  Three `git am` files, one commit each, made on `v16-a691b2345-r241` (canonical tip `7ef43022b`); each one
also applies on r241 by itself (`git apply --check`).  As announced in #291.  Every commit has its own kill switch; text, draft
counts and top-5 logprobs are identical to r241 with each of them on and off.

| file | touches | sha256 |
|---|---|---|
| `0001-gdn-conv-take-upstream-31660-s-rollback-store-the-4-.patch` | `ggml/src/ggml-cuda/gdn-conv.cu` (+14) | `6706687edc311cba02932ba05e0ad106bf6f6a1ab14a2258cb6c0f7e67cf094f` |
| `0002-llama-graph-the-DFlash-layer-input-copies-right-afte.patch` | `src/llama-graph.cpp` (+25) | `1bda5bc0f8337dd475a1384545c56051872ef46797861e7e64f7bfd688ad6380` |
| `0003-gdn-conv-store-a-decode-verify-GDN-conv-step-as-one-.patch` | new `ggml/src/ggml-cuda/gdn-conv-store.cu` / `.cuh`, `ggml-cuda-try-fuse.cu` (+6), `ggml-cuda.cu` (+11) | `163d8929ecc793be20e26095c34ab4dab822dcc181a54ee81c5a587eabbaca86` |

## 0001 — `gdn_conv_check` takes the #31660 store again

**Symptom.**  Since r236 the GDN direct conv declines on every layer, **single-sequence ubatches included** (INVARIANTS r241
records the multi-sequence case).  The prefill and the verify step run the plain CONCAT, SSM_CONV and q / k norm pair.

**Cause.**  `build_conv_state_store()` (#31660) writes the rollback snapshots with one 4-D view of Kw overlapping 3-column
windows of the concat (`ne = [3, Kw, C, n_seqs]`, dim-1 stride one element) and one `SET_ROWS` that reads it.  `gdn_conv_check()`
rejects both: the view is not a 3 × C one, and an unknown reader of the concat ends the match.

**Change.**  The window view's columns join the materialized tail (`gdn_concat_tail` writes them).  The `SET_ROWS` (`src[0]` the
window view, `src[1]` the slot order, `src[2]` the cache slots) is accepted as a reader of that tail.  It is not folded into the
direct kernel: `snap` stays off, so the state gather is not deferred either.  A multi-sequence ubatch still declines
(`st->ne[2] == 1`), as recorded.  Kill switch: the existing `GGML_CUDA_DISABLE_CONV_FUSION=1`.

## 0002 — the DFlash layer-input copies right after each layer

**Symptom.**  With a DFlash draft the target's compute buffer is ~757 MiB at `-ub 2048` on dense 27B; with F1 it was ~582 MiB
(r235's startup log: 757.75, then 581.75 MiB once F1 armed; r239 / r241 stay at 757 MiB).

**Cause.**  `build_layer_inp_dev()` (#31506) appends the `SET_ROWS` that copy the extracted layer inputs into the device buffer
after the whole graph.  Each extracted layer input (5 × `[5120, 2048]` f32 here) then stays alive through every later layer.

**Change.**  Each layer's copy (its VIEW, RESHAPE and SET_ROWS) moves to just after the next layer's input node, where F1 put its
store.  Only the node order changes; every node computes the same.  The move is made only when the next layer's input is a node
and the source comes before it.  Kill switch: `LLAMA_LAYER_INP_EARLY=0`.

## 0003 — a decode / verify GDN conv step as one kernel (`gdn-conv-store.cu`)

**What it replaces.**  With 0001 a verify step of a GDN layer still runs the state gather, the concat tail, the `SET_ROWS`, the
direct conv and the norm pair.  The direct conv's own norm fold stands down here, because the allocator puts the norm outputs
over x / the state.  r235 had one kernel per layer.

**Change.**  For 1..16 tokens, one sequence, RDNA4: one kernel per layer.
- It reads the state, x and the taps.  The state is the cache row itself when the `GET_ROWS` is deferred: nothing else reads it,
  and nothing but the folded `SET_ROWS` writes the cache before the conv.
- It writes the conv + silu output, the normalized q / k (the two SCALE outputs), and the Kw snapshot windows straight into the
  rollback slots.
- The CONCAT, the window view and the `SET_ROWS` launch nothing.
  - With a per-layer slot-order chain (`LLAMA_SHARE_SLOT_REV=0`) the CONCAT takes the chain too.
  - With r241's shared chain, the chain runs as before for the later layers, and the `SET_ROWS` is skipped where it stands.
- `graph_optimize` keeps x and the state alive until the k norm's SCALE.
- A thread per channel; it reads its three state values before it writes any slot (the deferred read and slot 0 may be one
  row).  22 VGPRs, no spills.

**Arithmetic.**
- The conv is the direct conv's (the same 4-tap order and silu).
- The q / k sums are `rms_norm_scale_pair_f32<256>`'s: the warp butterflies, a head's four warp sums as ((s0 + s2) + (s1 + s3)),
  no contraction (there the square is summed in the column loop, so it does not fuse with the first butterfly add), and a
  run-time divisor.
- A first version that copied the direct conv's norm fold (contracted, `/ 128` as a constant) gave different logprobs.  The two
  effects: 0 + y·y fusing with the first butterfly add, and `sum * 2^-7` fusing with `+ eps`.
  So the direct conv's own norm fold may not be bit-identical to the pair.  We did not check that further.

Kill switch: `GGML_CUDA_FUSE_GDN_CONV_STORE=0`.  0003 does not need 0001, but was tested on top of it.

## Results

Linux, Ryzen 9 3900X, one R9700 (gfx1201), ROCm 10.0.0.  `ROCBLAS_USE_HIPBLASLT=0` (GREEDY-PURITY §44) in every arm.  r241 =
stock `v16-a691b2345-r241`; PR = r241 + the three commits.  Every arm is a freshly started server with a few seconds of warm-up.

**Same output.**  Each arm against stock r241: text and draft counts of every request, and the top-5 logprobs of 64 tokens after
2K / 8K / 30K prompts (drafted, so verify steps included).

| config | arms against r241 | requests | differing |
|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL, DFlash2 draft 5, `-np 2 --kv-unified -c 131072 -ub 2048`, q8_0 K/V | all on; 0001 only; 0003 off; 0002 off | 25: tool-calling ERP ×3, an 8K copy, 7 prompt lengths, two prompts in one request, prompt-cache continuation and round trips, 110K, two requests at once, logprobs ×3 | none |
| same, `--spec-type draft-mtp --spec-draft-n-max 3` | all on; 0001 only | 7: ERP ×3, the copy, logprobs ×3 | none |
| Qwen3.6-35B-A3B UD-Q4_K_M + its DFlash draft | all on | 6: ERP ×3, logprobs ×3 | none |
| dense 27B, no draft (the old one-CPY store) | all on | logprobs ×3 | none |

**Speed** (dense 27B XL, `-np 2 --kv-unified -c 131072`, raw-text prompts without the cache, 256 greedy tokens; one session,
r241 / 0001 only / all on / all on / 0001 only / r241, medians; step = predicted_ms / (predicted_n − accepted)):

| draft | prompt | 0001 only: prefill | 0001 only: step | all on: prefill | all on: step |
|---|---|---|---|---|---|
| DFlash2, 5 | 8K | +0.73 % | −0.02 % | +0.64 % | −0.77 % |
| DFlash2, 5 | 32K | +1.01 % | −0.27 % | +0.99 % | −1.44 % |
| MTP, 3 | 8K | +1.33 % | −0.16 % | +1.31 % | −1.47 % |
| MTP, 3 | 32K | +0.84 % | +0.29 % | +1.12 % | −0.96 % |

- r241 itself: DFlash2 prefill 1383 / 1239 t/s, step 46.32 / 49.55 ms; MTP prefill 1298 / 1170 t/s, step 46.06 / 51.66 ms.
- An earlier session (r241 / all on / all on / r241) gave DFlash2 +0.80 / +1.13 % prefill, −1.09 / −1.36 % step, and MTP +1.29 / +1.09 %
  prefill, −1.42 / −0.63 % step.
- So 0001 is the prefill part, and 0003 is the step part.

**VRAM** after the run: DFlash2 25047 → 24871 MiB (−176, 0002); MTP unchanged; 35B-A3B + DFlash 24350 → 24223 MiB.

## Not validated

- Several GPUs, `-sm tensor`, `-ncmoe` / host-resident experts.
- Windows: not built or run.  It does not touch the host-side MTP cost in #291.
- `GGML_CUDA_DISABLE_GRAPHS=1`.
- RDNA3, other GDN architectures (qwen3next, qwen4exp).
- A multi-sequence ubatch: 0001 and 0003 still decline there, as recorded; 3+ slots.
- Speed on 35B-A3B.
- The direct conv's own norm fold (see the note above).
