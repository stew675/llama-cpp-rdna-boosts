# WIP follow-up — GDN chunked-prefill makes plain decode and speculative verify disagree

**Status:** OPEN, immediately actionable.  Found 2026-09-11 during the gfx1151
issue-#25 validation; **not** part of issue #25 (that is FA `parallel_blocks`,
fixed by Block 00) and **not** an upstream defect — see "Provenance".
**Owner box:** the GFX1201 machine (faster tests) — the maintainer is moving
this work there after the updated delivery `main` is pushed.
**Source of record:** the GFX1201 agent's issue-#25 investigation
(`README.md` in this directory) and the gfx1151 validation
(`../strix-halo/issue25/GATE-2026-09-11-block00-rdna35.md`).

## Symptom

On one build and one prompt, `--spec-type none` and MTP produce different
greedy text, and the difference is *entirely* the GDN chunked prefill — it
disappears with `GGML_CUDA_GDN_CHUNKED=0`.

gfx1151, 15-patch build (Block 00 + 01-14), Qwen3.8-27B Q8_0, f16 KV,
`-fa auto`, greedy seed 42, prompt p0:

| arm | sha1 | with `GGML_CUDA_GDN_CHUNKED=0` |
|---|---|---|
| `--spec-type none` | `9216c6d1` | `bba7741d` |
| `--spec-type draft-mtp --spec-draft-n-max 2` | `bba7741d` | `bba7741d` |
| `--spec-type draft-mtp --spec-draft-n-max 4` | `bba7741d` | `bba7741d` |

With the chunked path off, **`none == n-max 2 == n-max 4`** (byte-identical).

Note this is a *plain-vs-spec* divergence, not the n-max 2 vs n-max 4 issue.
n-max 2 vs 4 is fixed by Block 00 either way.

## Mechanism (fork-only, block 02)

The GDN op carries `K = cparams.n_rs_seq + 1` snapshot slots
(`src/models/delta-net-base.cpp`); `common/common.cpp` sets
`n_rs_seq = speculative.draft.n_max` for MTP.  `gated_delta_net.cu` then has
two chunked branches ahead of the sequential kernel:

1. `(cache == nullptr || K == 1) && K == 1 && n_tokens > 1` — the **plain
   multi-token prefill** path.  Plain decode has `n_rs_seq = 0` → `K = 1`, so a
   prompt prefill runs the **chunked** kernel.
2. `K > 1 && n_seqs == 1 && n_tokens > K + 64` — the **MTP chunked prefix**.
   It computes `n_prefix = n_tokens - K` with the chunked kernel and the last
   `K` tokens with the sequential kernel.

Spec prefill has `K = n_max + 1 > 1`, so branch 1 is skipped; for a short prompt
(`n_tokens <= K + 64`) branch 2 is skipped as well → **sequential**.  So the
same prompt is preprocessed by different GDN kernels in plain vs spec mode, and
the resulting SSM state differs.

Second effect of the same code: when branch 2 *does* fire (prompt
`> K + 64`), `n_prefix` is a function of `K = n_max + 1`, so the boundary
between chunked and sequential shifts with the draft depth and the post-prefill
state becomes `n_max`-dependent.

### Evidence (logits probe, `../strix-halo/issue25/logits-width.cpp`)

MTP-faithful `n_rs_seq = W-1`, row 0, max |logit| diff:

| prefill P | env | W1-W3 | W3-W5 |
|---|---|---|---|
| 256 | default (chunked on) | 0.298597 | **0.210405** |
| 256 | `GGML_CUDA_GDN_CHUNKED=0` | 0.000000 | **0.000000** |
| 32 | default | 0.140793 | 0.000000 |
| 1024 | `GGML_CUDA_GDN_CHUNKED=0` | 0.000000 | 0.000000 |

(`P=1024` with chunked off is the FA-isolation case used for Block 00; with
chunked off everything is 0.)

### Observed impact so far: latent

A 2.8k-token prompt, real MTP, 200 generated tokens: n-max 2 and n-max 4 both
`2b0b6d6d`, with chunked on **and** off.  So the divergence exists in the state
but did not flip greedy within that run.  Treat it as a correctness/consistency
defect (plain decode should be reproducible by the spec path) rather than a
reported symptom.

## Provenance — this is our own work, not upstream

- upstream `9113cc188` ships only `ggml/src/ggml-cuda/gated_delta_net.cu` /
  `.cuh`, and `gated_delta_net.cu:180` reads
  `//TODO: Add chunked kernel for even faster pre-fill`.
- The fork adds `gated_delta_net_chunked.cu/.cuh`,
  `gated_delta_net_chunked_bf16.cu`, `gated_delta_net_chunked_bf16_gfx11.cu`,
  the `GGML_CUDA_GDN_CHUNKED` / `GGML_CUDA_GDN_CHUNKED_BF16` gates, and the
  MTP chunked-prefix dispatch.  All of it arrives in **block 02**
  (`patches/0002-fused-chunked-gated-delta-net-p.patch`); the MTP prefix
  dispatch was added 2026-09-01 (fork PR #9).
- Upstream *does* own the GDN op and the `keep_rs = K > 1` snapshot path, and
  the sequential kernel is K-independent — which is why `GDN_CHUNKED=0` makes
  plain == spec.

**Therefore this belongs to block 02 (or a fork follow-up), never Block 00.**
Block 00 is reserved for upstream-originated structural flaws (FA
`parallel_blocks`, the Vulkan masked-V shaders).

## Fix directions (for the GFX1201 session)

Ranked by preference:

1. **Make the chunked/sequential boundary independent of `K`.**  Pick
   `n_prefix = n_tokens - K_tail` with `K_tail` a fixed value (>= the largest
   supported `n_max`, or a multiple of the chunk size), and let the sequential
   tail produce the `K` snapshots needed.  Then `n_max` cannot move the
   boundary, and plain (`K=1`) vs spec (`K=n_max+1`) agree.
2. **Derive the snapshots without a K-dependent boundary** — e.g. run the
   chunked kernel over the whole prompt and materialise the snapshot slots from
   it, so there is no chunked->sequential hand-off to shift.
3. **Bit-exact the chunked kernel** against sequential.  This is the only route
   that also fixes the deliberate "near-lossless" bf16 chunked default
   (PPL +0.056 % / KL 0.0036); note the fp32 chunked path is already documented
   as bit-exact, the bf16 one is not.
4. **Gate only.**  If a K-independent design is not cheap, gate the chunked
   path so it is used only where the snapshot/plain equivalence does not matter,
   and document the residual.  Weakest option — it removes the prefill win.

Do **not** simply route the plain multi-token path through sequential: that
throws away the chunked-prefill win the block exists for.

## Validation gate for any fix

- `--spec-type none` and `--spec-type draft-mtp` (n-max 2 and 4) must be
  **byte-identical** on the same prompt with `GGML_CUDA_GDN_CHUNKED` at its
  default (`on`) — currently they are not.
- `logits-width` probe, `P=256`, `RS=from_w`, chunked **on**: `W1-W3` and
  `W3-W5` both `0.000000` (currently `W3-W5 = 0.210405`).
- Long-prompt (`> K + 64` tokens) n-max 2 vs 4 byte-identity, chunked on.
- Re-run the delivery coherence gate + `benchmarks/mtp-adaptive-methodology.md`
  after the change; confirm the chunked prefill win is preserved (Block 02's
  reason to exist).

## Repro (GFX1201)

```sh
# plain vs spec, same build/prompt, chunked on (default)
HIP_VISIBLE_DEVICES=0,1 llama-cli -m Qwen3.8-27B-Q8_0.gguf -ngl 99 -sm tensor -ts 1/1 \
  -c 8192 -ctk f16 -ctv f16 -fa auto -p '<p0 prompt>' -n 512 --seed 42 --temp 0 \
  --top-k 1 --no-display-prompt --single-turn \
  --spec-type {none | draft-mtp --spec-draft-n-max 2 | draft-mtp --spec-draft-n-max 4}
# -> none differs; add GGML_CUDA_GDN_CHUNKED=0 -> all three identical
```

Probe (build logits-width against the tree's own libs):

```sh
RS=from_w GGML_CUDA_GDN_CHUNKED=1 <lw> Qwen3.8-27B-Q8_0.gguf <text> 256 512   # W3-W5 != 0
RS=from_w GGML_CUDA_GDN_CHUNKED=0 <lw> Qwen3.8-27B-Q8_0.gguf <text> 256 512   # W3-W5 == 0
```
