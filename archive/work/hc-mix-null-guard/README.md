Thanks for the fused hc-mixer path in `build_hc_mix`. This only guards its entry checks: a small null-pointer case we came across, flagged in case it's useful. Nothing is urgent on our side.

**What happens.** The ggml-org `Qwen3.8-Flash-Next` Q8_0 GGUF (`ggml-org/Qwen3.8-Flash-Next-GGUF`, 2 shards, sha256 `f4ad75b0...` / `f83d260b...`) uses the merged layout from the upstream #31028 converter: `hc_*_down` and `hc_*_inject` are one tensor, so `w_down` is null in `llama_model_qwen4exp::graph::build_hc_mix`. The `fused_bf16` and `fused_ok` checks read `w_down->ne[1]` / `w_down->type` before anything tests it, and the process segfaults at the first graph build (`sched_reserve`). We hit it on r292 with `llama-perplexity`; the code is unchanged in r299.

**The change** (`src/models/qwen4exp.cpp`, +3 / -2): `w_down != nullptr &&` in front of both checks, so a merged-layout model takes the unfused chain, which already handles a null `w_down`. Separate-tensor GGUFs are unaffected: the condition is true for them and the rest of the expression is unchanged.

**Tested.** r292 + this guard ran the Q8_0 to the end: `llama-perplexity`, 2 x 8192 tokens of code text, `-sm tensor --n-cpu-moe 48 -lme 32`, final PPL 1.3477. With the same guard on our own tree, this file served as the KLD reference for a quant comparison. It is the only merged-layout file we have run.

Files: `0001-*.patch` (`git format-patch` against r298; applies to r299 with `git am`) and this README. Happy to test other merged-layout files or builds here if that would help.

(Claude-assisted, as before.)
