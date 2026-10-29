Thank you for r170's fix of the routed-MMB / MoE MMQ assert (#241). We swapped a 7900 XTX into our box this week and hit exactly that with r149 on 9-511 token prompts; on r170 and r171 the same prompt sizes run cleanly on both Qwen MoE models we tried.

Two small patches, offered in case they're useful. Both gave identical PPL / KLD and byte-identical greedy replies in our tests (the concurrent case is covered under Gates).

**0001: compute the M-RoPE section power once** (`rope.cu`: `rope_multi` and the fused `rms_norm_mul_rope_f32`). Every section arm computed the same `powf(theta_scale, i0/2)`; only the position index differs. Adjacent lanes take different arms (`sector % 3`), so the divergent arms were serialized within a wave and `powf` executed once per arm taken. It is now computed once before the branch: same input, same value, and when no arm matches `theta_base` still stays 0. Traced on Qwen3.6-35B-A3B decode, the K / Q norm+rope kernels go from 7.96 / 7.71 to 6.16 / 5.97 us.

**0002: GDN direct conv for 1-token decode** (`gdn-conv.cu`, `mmvq.cu`). The verify gate in `gdn_conv_check` needed T >= 2, so plain decode (no MTP or other speculative batch) never took the direct conv on RDNA4 / RDNA3_0. Allowing T = 1 alone is not enough. The `ssm_conv_in` epilogue at the in-proj MUL_MAT still consumes the CONCAT, the direct conv then matches at SSM_CONV and reads the plain matmul output that the epilogue never wrote. Our first try produced garbage text.
- `ggml_cuda_ssm_conv_in_try_fuse` now returns 0 when `ggml_cuda_gdn_conv_match_at_concat` takes the concat. That is the same check the direct path runs at the CONCAT and at the SSM_CONV, so exactly one of the two paths handles each graph.
- Multi-device graphs, `GGML_CUDA_DISABLE_CONV_FUSION`, `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`, other archs, C % 256 != 0 and multi-sequence ubatches all fail that check and keep the `ssm_conv_in` epilogue as before.
- `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` now also turns off the decode case; the comment says so, since the name predates it.

Two notes from an independent review of the logic, which raised no correctness issues beyond these:
- The decode-only `ssm_prescan` fusion is tried after the direct conv, so on these archs it can no longer claim a 1-token conv. It did not fire for Qwen3.6 on stock r171 here (no prescan kernel in the trace), but it may on other models; we have not checked.
- This predates the patch, flagging for completeness: `gdn_conv_check` does not check that the conv output does not alias `x` or the state. 0002 extends that path to T = 1; we saw no such reuse in practice.

**Setup:** RX 7900 XTX (gfx1100), 1 GPU, Qwen3.6-35B-A3B Q4_K_M (unsloth). Stock r171 (tree `2f5870713e94`, matches `release.json`) vs the same + 0001 + 0002, both built with `-DAMDGPU_TARGETS=gfx1100`, your default env.

**Gates:**
- wikitext-2 perplexity / KLD (24 chunks, `-c 512`, against logits from a plain upstream HIP build): identical to the last digit, PPL 5.920938, KLD 0.009509.
- llama-server greedy, 4 chats x 3 turns: byte-identical replies (743 tokens).
- 2 concurrent requests decoding together: stock r171 itself gave three different outputs over three runs (the batch composition depends on timing); the patched build gave the same three outputs as stock.
- 0 GPU faults.

**Throughput**, `llama-bench -fa 1 -r 5`, t/s, two alternating rounds. The middle arm is the patched build with `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`, i.e. 0001 alone:

| arm | tg128 | tg128 @ d4096 | pp512 |
|---|---|---|---|
| stock r171 | 153.64 / 153.30 | 151.58 / 150.80 | 5143 / 5078 |
| + 0001 | 154.47 / 154.14 | 152.14 / 151.82 | 5138 / 5097 |
| + 0001 + 0002 | 156.43 / 156.13 | 154.00 / 153.84 | 5120 / 5103 |

So about +0.4 to 0.6 % decode from 0001 and +1.2 to 1.3 % from 0002 (two rounds each); prefill is within noise. Per token, the direct conv replaces `ssm_conv_f32` + `rms_norm_scale_pair_f32` with `gdn_concat_tail` + the direct kernel: the same launch count and about 110 us less GPU time per token in the trace (the wall-clock tg128 gain is about 80 us per token).

On our own stack the same 0002 gives more, +7 % decode, because there the direct path also folds the conv-state GET_ROWS and its write-back CPY, which removes three launches per DeltaNet layer. Those are two folds we keep locally: the snapshot fold from #177 and a conv-state gather fold that builds on it. Until now they only fired on MTP verify steps, so their gain looked small; with 1-token decode on the direct path they show real promise. We'll follow up separately with more information on both and would like to revisit #177 with you then, including whatever determinism checks you'd want to see.

Not tested on RDNA4: our R9700s are out of the box this week. The arch gate includes RDNA4, so the 1-token path is new there too. We're happy to run the 2-GPU checks when they're back, or anything else you'd like. No urgency on our side.

Files: `0001-*.patch`, `0002-*.patch` (`git format-patch` against the r171 tree, apply with `git am`) and this README.

(Claude-assisted, as before.)
