# PROVENANCE — where the model facts come from

Everything in this work that concerns the Qwen3.8-Flash-Next model was
reverse-engineered from public artifacts in the 2026-08-30 session (and
recorded before the compaction). No private or NDA material involved.

## Source artifacts

| Artifact | Where |
|---|---|
| Unsloth GGUF repo | Hugging Face `unsloth/Qwen3.8-Flash-Next-GGUF` (all quant builds: Q8_0, Q5_K_XL, Q6_K_XL, Q4_K_XL, ..., IQ1_M) |
| Split-file headers | downloaded `*-0000x-of-00006.gguf` heads, truncated to the tensor-info section |
| UD-Q4_K_XL model files | `/models/Qwen3.8/Flash-Next/Q4_K_XL/` (4 splits, ~111 GB) on this machine |
| llama.cpp | local checkout at `a7cc83bba` + the `ngram-disk` branch (this work) |

## Verified facts

### Quantization scheme (Unsloth)

The PLE n-gram table (`per_layer_token_embd.weight`) is always quantized
**separately from the model weights**, and never at the model's level:

- Q8_0 / Q5_K_XL / Q6_K_XL builds: ngram = **Q8_0, 54.4 GB** — one identical
  LFS blob across the three builds, sha256
  `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20`,
  always alone in `*-00003-of-00006.gguf`.
- Q4_K_XL and below: ngram = **IQ4_NL, 28.8 GB** (even the 1-bit IQ1_M build
  keeps ngram IQ4_NL).

So a "frankenstein" Q4_K_XL-weights + Q8_0-ngram model is a byte-graft of the
Q8_0 table into the Q4_K_XL file (see `plan/qwen38-flash-q4-q8.md` and
`tests/graft-ngram-q8.py`), not a re-quantization.

### GGUF / ggml facts (verified against the compiled repo)

- Type IDs: Q8_0=8, Q8_1=9, Q4_K=12, Q5_K=13, Q6_K=14, IQ1_S=19, IQ4_NL=20,
  IQ2_XS=17, IQ1_M=29, BF16=30.
- GGUF tensor data is tight-packed: `nbytes = nelems * type_size / blck_size`,
  no per-row or per-tensor 32-byte padding. `ggml_row_size()` pads and must
  NOT be used for file offsets.
- All 11 Unsloth split-file headers are offset-consistent.

### UD-Q4_K_XL per-tensor map (82.5 GB weights + 28.8 GB IQ4_NL ngram)

- `ffn_down_exps` Q5_1, except layers 2, 4, 30, 46, 47 -> Q8_0
- `ffn_gate_exps` / `ffn_up_exps` Q4_K, except layer 2 -> Q5_K
- `indexer.k_proj` / `indexer.q_proj` BF16
- norms, `gate_inp`, `ssm_alpha/beta`, `hc_*_inject`, `ple_conv1d` F32
- everything else Q8_0
- 48 layers; full attention only at layers 3, 7, 11, ..., 47; n_embd 2560,
  n_head 24 (head_dim 256), n_head_kv 2, top_k 2048, expert_count 512
  (10 used), shared ffn 640

### Real PLE parameters

- `ple.ngram_size = 3`, `ple.heads_per_ngram = 8` -> `ple_n_heads = 16`
  rows/token
- `ple_head_dim = 160` (Q8_0 row = 170 B; IQ4_NL row = 90 B)
- table 320,001,536 x 160; 16 heads x ~20.0M-row vocabs
  (sum 320,001,446; padded to 320,001,536)
- `layer_multipliers = [23703573157769, 20109073645365, 8052911324071]`
- PLE on layer 1 (recurrent); eos 248044; image 248056
- hash: `mixed_n = t[p]*m[0] ^ t[p-1]*m[1] ^ ...`; row = `mixed % vocab + off`;
  EOS resets the window; predecessors come from the attention KV cells'
  `ext.tok`

### Lazy-loading mechanism in llama.cpp (at `a7cc83bba`)

- `per_layer_token_embd` is created with `TENSOR_READ_LAZY`
  (`src/models/qwen4exp.cpp`); the loader gives it a CPU buffer wrapping the
  mmap; `ggml_get_rows` dequantizes rows straight out of the mmap
  (`ggml_compute_forward_get_rows_q`, `ggml/src/ggml-cpu/ops.cpp`).
- Lazy ranges get `POSIX_MADV_RANDOM`; the non-lazy complement of each file
  gets `POSIX_MADV_WILLNEED` at load (`src/llama-mmap.cpp`,
  `ranges_complement`).
- The model retains `llama_mmaps` but NOT `llama_files`
  (`src/llama-model.cpp`), which is why the managed reader `dup()`s the fd.
- `LLAMA_LAZY_MODE_AUTO` lazy-loads only tensors > 4 GiB.

## How the runtime facts were measured

- Logits: `tests/logitdump.cpp` (deterministic, sampled positions,
  byte-compare via `cmp`/md5).
- Residency: `/proc/<pid>/smaps` samplers (`tests/smaps_*.py`) — file-backed
  bytes inside the ngram window, and anonymous vs file-backed totals.
- VRAM: `rocm-smi --showmeminfo vram` sampled during runs.
- RSS peak: `/usr/bin/time -v`.
