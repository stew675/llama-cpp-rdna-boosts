## What Unsloth actually did

The ngram table is quantized separately from the weights, always at a much higher precision, never at the model's level:

```
┌─────────────────────┬─────────────────────────────────────────────────┬────────────────────────────┬────────────┬────────────┐
│ Build               │ Model weights (dominant tensors)                │ ngram per_layer_token_embd │ ngram size │ total      │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ BF16                │ BF16                                            │ BF16                       │ 102.4 GB   │ 355 GB     │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ Q8_0                │ Q8_0 everywhere (indexer BF16, norms F32)       │ Q8_0                       │ 54.4 GB    │ 188 GB     │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ UD-Q6_K_XL          │ experts Q6_K, rest Q8_0                         │ Q8_0                       │ 54.4 GB    │ ~163 GB    │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ UD-Q5_K_XL          │ experts Q5_K, rest Q8_0                         │ Q8_0                       │ 54.4 GB    │ 158 GB     │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ UD-Q4_K_XL          │ experts Q4_K + down Q5_1, attn/embd/output Q8_0 │ IQ4_NL                     │ 28.8 GB    │ 111 GB     │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ UD-Q3_K_XL          │ (not parsed; sizes imply)                       │ IQ4_NL                     │ 28.8 GB    │ 90 GB      │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ UD-Q2_K_XL          │ experts IQ2_XS, down IQ4_NL, attn Q5_K          │ IQ4_NL                     │ 28.8 GB    │ 79 GB      │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ UD-IQ4_XS / IQ3_XXS │ (sizes imply)                                   │ IQ4_NL                     │ 28.8 GB    │ 94 / 82 GB │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ UD-IQ1_M            │ experts IQ1_M, attn Q5_K, embd/output Q4_K      │ IQ4_NL                     │ 28.8 GB    │ 74.5 GB    │
├─────────────────────┼─────────────────────────────────────────────────┼────────────────────────────┼────────────┼────────────┤
│ UD-IQ1_S            │ experts IQ1_S, attn Q5_K, embd/output Q4_K      │ IQ4_NL                     │ 28.8 GB    │ 72.5 GB    │
└─────────────────────┴─────────────────────────────────────────────────┴────────────────────────────┴────────────┴────────────┘
```

Key observations:

- The ngram is a step-function, not proportional: Q8_0 (1.0625 bpw) in the top 4 builds, then IQ4_NL (0.5625 bpw) in everything from Q4_K_XL down.
  Even the 1-bit IQ1_M build keeps it at IQ4_NL — exactly what their docs say ("1-bit is 75GB and uses 4-bit for the Ngram/PLE").
  It never follows the model's bit level.
- "UD" = per-tensor-group mixed precision, not one level.
  E.g. Q4_K_XL is output Q8_0 / token_embd Q8_0 / attention Q8_0 / experts Q4_K (down Q5_1) / ngram IQ4_NL. Q5_K_XL/Q6_K_XL are even simpler: the Q8_0 build with only the MoE expert tensors re-quantized.
- It's all one GGUF — the ngram tensor lives in the same split files as everything else, just at a different type.
  In the Q8_0-style builds it happens to occupy its own split (file 00003, 54.4 GB) because a single 54.4 GB tensor doesn't fit the ~49 GB split budget; in the IQ4_NL builds it's packed into file 00002 alongside the output head.
