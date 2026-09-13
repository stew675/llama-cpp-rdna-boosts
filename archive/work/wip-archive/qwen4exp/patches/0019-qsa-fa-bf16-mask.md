# Patch 0019: fix QSA sparse FA BF16 branch uninitialized mask smem

llama.cpp commit 1f5d431f7, on top of a61a7d4b9.
Apply: `git apply patches/0019-qsa-fa-bf16-mask.patch`

## The bug

With a REAL bf16 KV cache (`--cache-type-k bf16 --cache-type-v bf16`,
which this tree supports since bf16 stopped being silently downgraded to f16)
AND the QSA sparse flash-attention path (`LLAMA_QSA_SPARSE_FA=1`), decode
produced garbage: the model emitted "/" x n at any temperature, on any prompt,
raw or templated. f16 KV + sparse = coherent; bf16 KV + dense = coherent; only
the bf16 + sparse combination broke. The decode gates never caught it: the
logitdump/bseq gates and every llama-bench run use the default f16 cache.

Root cause: in `flash_attn_qsa` (fattn-qsa.cu), the per-cell mask `M_smem`
was staged only inside the F16/Q8_0 cooperative gather. The BF16 branch staged
only K into shared memory and never wrote `M_smem`, but its score pass reads
`M_smem[cell]` for every K/V type - so every bf16 sparse attention score was
corrupted by uninitialized shared memory. The bf16 branch was dead code until
true bf16 KV landed (it previously never ran).

## The fix

Stage `M_smem[flat] = maskh[...]` in the BF16 gather too (mirrors the F16
branch exactly).

## Verification

- Server, bf16 KV + sparse env: "What is the capital of France?" ->
  "The capital of France is Paris. Paris has been the..." (was "///...").
- Sustained 121-token generation coherent (Rayleigh scattering answer).
- User's full config at ctx 102400 + sparse env: reasons coherently
  ("<think> The user is asking a straightforward factual question...").
- f16-path gates unchanged by construction (F16/Q8_0 branch untouched):
  llama-bench pp512 1537.9 / tg128 45.69; bseq_val K=2 streams identical.
