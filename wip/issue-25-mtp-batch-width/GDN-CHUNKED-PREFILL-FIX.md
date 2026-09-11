# GDN chunked-prefill plain-vs-spec divergence — gated fix (block 02 amendment)

Date: 2026-09-11.  Box: 3x R9700 (gfx1201, RDNA4), ROCm 7.14
(`/opt/rocm-7.14-gfx1201`).  Tree: `~/llama.cpp` `rdna-boosts`
(`8afa5fb78` = delivery 00-14 + block 15), branch `gdn-fix`.
Model: `Qwen3.8-27B-Q8_0.gguf` unless stated; f16 KV, `-fa auto`.

Follow-up to `GDN-CHUNKED-PREFILL-FOLLOWUP.md` (the gfx1151 finding).  Root
cause confirmed on gfx1201, fix implemented, validated, and landed as a
**block-02 amendment** (`patches/0002-…`).  Patch:
`gdn-chunked-align-boundary.patch` — the ORIGINAL opt-in form of the
dispatch-only change to `ggml/src/ggml-cuda/gated_delta_net.cu`, +90/-2;
superseded by the delivery's block 02, which now carries the same gate with
the default flipped ON (opt out with `=0`).

**`GGML_CUDA_GDN_ALIGN_BOUNDARY`, later flipped to default ON (opt-out with
`=0`), 2026-09-11.**  Originally landed opt-in because the fork's existing
boundary is deliberate and slightly faster; it was then made the default after
the `-sm tensor` follow-up showed it is the second of the two independent
causes of the plain-vs-spec drift (the first being the block-13 dense-MMVQ
alignment).  `=0` restores the K-dependent boundary and its ~1.5-1.8 % prefill
edge.  See `../sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`.

## Root cause (confirmed)

`gated_delta_net.cu` had two chunked branches with a **K-dependent**
chunk/sequential boundary:

- plain prefill (`K == 1`): chunked over the **whole** prompt;
- MTP prefill (`K == n_max + 1`): chunked over `n_tokens - K`, sequential tail
  of `K` (or fully sequential when `n_tokens <= K + 64`).

The chunked kernel is not bit-exact with the sequential kernel (the bf16/WMMA
path is deliberately near-lossless; the fp32 path also differs — measured
below), so plain and MTP preprocess the same prompt into **different SSM
states**.  Greedy near-ties then flip: `--spec-type none` != `draft-mtp`.

gfx1201 probe (`logits-width`, `RS=from_w`, `P=256`, row 0, chunked on, gate
off): `max|W1-W3| = 0.136693`, `max|W3-W5| = 0.182106`; both `0.000000` with
`GGML_CUDA_GDN_CHUNKED=0`.  With `GGML_CUDA_GDN_CHUNKED_BF16=0` (fp32 chunked,
*not* the bf16 path): `0.131098` / `0.135354` — so this is not a bf16-only
effect.

## Fix — fixed-boundary chunked prefill (follow-up direction #1)

A new gated branch leaves the existing two branches untouched.  When
`GGML_CUDA_GDN_ALIGN_BOUNDARY=1`, the chunk/sequential boundary becomes a
function of `n_tokens` only:

```
n_prefix = n_tokens > GDN_CHUNKED_KTAIL ? n_tokens - GDN_CHUNKED_KTAIL : 0
chunked kernel over [0, n_prefix)  ->  prefix_state
sequential kernel over the last (n_tokens - n_prefix) tokens, with keep_rs = (K > 1)
```

`GDN_CHUNKED_KTAIL` is now **16** (`K > 16 ? K : 16`).  Plain (`K == 1`) and
MTP (`K == n_max + 1`) share exactly one boundary and one tail length for
`K <= 16`, so they compute the **same state regardless of whether the chunked
kernel is exact**.  The sequential tail also produces the K snapshots (slot j =
j tokens back), so rollback <= KTAIL-1 is exact.  The tail must be a *fixed*
constant (a K-derived boundary would give the two paths different prefixes and
defeat the branch); **16** covers `K <= 16` / `n_max <= 15`, including adaptive
MTP's recommended `n_max = 12`.  The `K > 16 ? K : 16` floor keeps deeper
drafts exact by reproducing the pre-alignment `K > 1` boundary (correct
snapshots, correct-but-not-bit-identical).  The tail is the entire cost
(27B pp512/2048/4096: KTAIL 64 = -1.5 %, 16 = -0.3..-0.8 %, 8 = ~0).  The
bf16/fp32 kernel selection, the launch-rejection fallback and the fused
GDN->cpy (`cache != nullptr`) behaviour are unchanged from the default
branches.  `n_seqs > 1` keeps the whole-ubatch chunked path (the chunked
`n_tokens` override is the sequence stride, so only `n_seqs == 1` may
truncate); `n_seqs > 1 && K > 1` stays sequential.  `GGML_CUDA_GDN_CHUNKED=0`
still forces the sequential kernel in both modes.

## Validation (gfx1201)

Logits probe (`RS=from_w`, `P=256`):

| mode | W1-W3 / W3-W5 |
|---|---|
| default (gate off) | `0.136693` / `0.182106` — byte-identical to pre-change |
| `GGML_CUDA_GDN_ALIGN_BOUNDARY=1` | `0.000000` / `0.000000` |
| gate on + `GGML_CUDA_GDN_CHUNKED_BF16=0` | `0.000000` / `0.000000` |

Text (`temp 0 --top-k 1 --seed 42`, sha1 of generated text; 27B p0, 1 GPU):

| mode | none / n2 / n4 |
|---|---|
| default (gate off) | none = `d9bf6850` (pre-fix output) |
| gate on | `1a9ef0a1` = `1a9ef0a1` = `1a9ef0a1` |
| `GGML_CUDA_GDN_CHUNKED=0` (reference) | `1a9ef0a1` = `1a9ef0a1` |

So with the gate on the chunked path is **byte-identical to the
fully-sequential reference** on the short prompts — the *default* plain
chunked-over-all (`d9bf6850`) is the outlier there.  (On a long prompt the
inherent chunked-vs-sequential difference remains — that is the kernel's
documented non-exactness, not this bug.)

Additional gate-on checks: 27B p2 (1 GPU) `b40c4858` = `b40c4858` = `b40c4858`;
27B fp32 chunked, 1 GPU `7d566fee` ×3; 27B long (~2.8k) n2 = n4 = `b452bc29`
(2 GPU tensor); 27B p0 3-GPU **layer** split none = n2; MoE 35B-A3B p0
chunked-on == chunked-off = `1447cbe3`.

`test-backend-ops -o GATED_DELTA_NET`: **46/46** in all of default, gate-on,
and gate-on + `GGML_CUDA_GDN_CHUNKED_BF16=0`.

MTP gates (default mode, gfx1201):
- dense 27B, `draft-mtp --spec-draft-n-max 3`, 1 GPU: acceptance **0.487**
  (0.740/0.433/0.279 per pos), **36.5 t/s** (recorded baseline 0.479 / 36.1).
- MoE 35B-A3B Q4_K_M, `draft-mtp --spec-draft-n-max 3`: acceptance **0.633**,
  **144.9 t/s** vs plain **93.5 t/s**.

Performance (27B Q8_0, 1 GPU, `-r 3`), gate on vs off:

| | pp512 | pp2048 | pp4096 | tg128 |
|---|---|---|---|---|
| gate off (default) | 1394.20 | 1363.53 | 1324.91 | 20.45 |
| gate on | 1379.03 | 1348.96 | 1308.60 | 20.43 |
| `GGML_CUDA_GDN_CHUNKED=0` | 1227.84 | 1205.27 | 1184.87 | 20.43 |

The gate costs **~1.1-1.8 %** prefill (the 64-token sequential tail); the
chunked win over sequential is still **~12 %**, decode unchanged.  This was
why it was initially opt-in; it is now the default (opt out with `=0`).

## Separate finding — `-sm tensor` plain-vs-spec divergence (NOT this bug)

Under `-sm tensor` on 2/3 GPUs, `--spec-type none` still differs from
`draft-mtp` (any n-max, including 1):

```
27B p0, 2 GPU tensor: none=85d321d9  n2=n4=5037ef2e  (diverge@1140)
27B p0, 3 GPU tensor: none=bb95820c  n2=n4=f60b79d0  (diverge@1253)
```

This is **independent of the GDN chunked path and of the gate**:
`GGML_CUDA_GDN_CHUNKED=0` gives byte-identical hashes to chunked-on (same
divergence), and the gate makes no difference under tensor split.  It also
survives `LLAMA_KQ_MASK_DERIVED=0` (block-15 V3) and `GGML_CUDA_ALLREDUCE=nccl`,
and does **not** occur with `-sm layer` or on 1 GPU.  So it is a second,
tensor-split-specific plain-vs-spec divergence (candidate: the tensor-split FA
/ recurrent-state path with `n_rs_seq > 0`), out of scope here.  Worth a
follow-up investigation, especially since the server runs 3-GPU tensor split.

## Provenance

Fork-only (block 02).  Upstream `9113cc188` ships only the sequential
`gated_delta_net.cu`/`.cuh`; all chunked kernels + gates + the MTP prefix
dispatch arrive in block 02.  So this is a **block-02 amendment**, never
block 00.
