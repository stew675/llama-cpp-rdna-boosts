# Shared-NextN MTP heads (`nextn_shared_target_tensors`) — the `X < Y` draft crash fixed 2026-09-22

**Status:** fixed and **delivered in the delivery set as block 00, release `v16-ebbb18522-r13`**
(2026-09-22; canonical tip `8491bf2bff8eb3a56e5120c3c9c17533a94ea6bf`, tree
`bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12`).  The WIP export `wip/closing-the-gap/patches/0015` is
**superseded** — a campaign rebuilt on r13 must not apply it (it is already in block 00).  The bug is
**upstream** (`04eb4c446 "llama : add Gemma4 MTP (#23398)"`, present at the fork point); upstream is not
ours to change, so the fundamental fix lives in block 00 (the structural base every later block builds
on).  It is also the `upstream/UPSTREAM-PR-mtp-shared-nextn` candidate.

## Symptom

A shared-NextN MTP sidecar — the one the `<model>/IQ4_NL/` directory ships
(`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `qwen4exp.nextn_shared_target_tensors = true`) — loads,
but **every** speculative round dies on the second draft step:

```
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 584
 - the tokens for sequence 0 in the input batch have a starting position of Y = 584
 for M-RoPE, it is required that the position satisfies: X < Y
E decode: failed to initialize batch
E spec  draft: llama_decode[1] returned -1
```

A draft never exceeds one token, the adaptive controller is inert, and throughput is *below* plain
decode.  The non-shared sidecar (`mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, which carries its own
`token_embd`/`output`) has no errors.

## Root cause

`nextn_shared_target_tensors = true` means the head ships **no** `token_embd.weight` / `output.weight`
and reuses the **target's**.  `llama_context` therefore sets `cparams.ctx_other = ctx_tgt` to satisfy
the tensor-sharing path (`src/llama-context.cpp`):

```cpp
if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH || model.arch == LLM_ARCH_QWEN4EXP) {
    if (model.tok_embd == nullptr || model.output == nullptr) {
        ...
        cparams.ctx_other = params.ctx_other;
    }
}
```

The MTP draft driver in `common/speculative.cpp` then inferred **KV sharing** from the same
`ctx_other` pointer:

```cpp
is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;   // WRONG
```

and took the gemma4-assistant arm, which
* uses the same position `dp.pos0` for **every** draft token, and
* skips the catch-up decode in `process()`.

So step 0 stores position `pos0`, step 1 re-adds `pos0` — `X = Y` — and M-RoPE rejects it.  But
`ctx_other` is used for two different reasons: **gemma4 shares the target KV**, while
**eagle3/dflash/qwen4exp shared-NextN heads only borrow the target's `token_embd`/`output`** and keep
their own KV.  Only the former may reuse the target position and skip the catch-up.

## Fix

Gate `is_mem_shared` on the arch as well (this is the reference's line at
`~/pwilkin-llama-cpp:common/speculative.cpp:1425`):

```cpp
char arch[64] = {0};
llama_model_meta_val_str(llama_get_model(ctx_dft), "general.architecture", arch, sizeof(arch));
is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt && std::strcmp(arch, "gemma4-assistant") == 0;
chain_heads   = n_mtp_layers > 1 && !is_mem_shared;
```

One new public-API call (`llama_model_meta_val_str`, already declared) and one condition.  No kernel,
graph or loader change; the shared tensors themselves were already reachable through `ctx_other`.

## Verification (gfx1151, qwen4exp IQ4_NL + the shared Q8_0 head, `-c 8192 -b/-ub 2048`, `-n 256`)

| arm | `X < Y` errors | acc per pos | generation |
|---|---:|---|---:|
| before (`is_mem_shared` bug) | 78 | — (draft always 1 token) | 28.3 t/s |
| after, `draft-mtp n_max 8` | **0** | 0.679, 0.462, 0.333, 0.256, 0.205, 0.141, 0.115, 0.077 | — |
| after, `draft-mtp-adaptive n_max 8` (the `runme`) | **0** | 0.725, 0.382, 0.284, 0.078, 0.020, 0.000, 0.000, 0.000 | 35.4 t/s |

The adaptive controller now reports real depth transitions (`5 -> 4`, `4 -> 3`), which it could not
do while every draft was one token wide.  The non-shared head is unchanged (it never set `ctx_other`).

**Not fixed here:** the greedy `plain != draft-mtp` text residual on qwen4exp.  It reproduces with
this fix disabled for A/B as well (`LLAMA_QSA_SCORE_STRIP=0`), so it is pre-existing and independent
of the shared head; the delivery's `GREEDY-PURITY.md` cause-3 narrative and the `n_rs_batch` chunked-GDN
threshold are the places to look.

## Provenance / where the fix lives

* Fork: `common/speculative.cpp`, `common_speculative_impl_draft_mtp` ctor.
* **Upstream bug, not block 01**: the inference comes from `04eb4c446 "llama : add Gemma4 MTP
  (#23398)"` and is in `master` at the fork point.  Block 01 (`patches/0001`) keeps the line only as
  hunk context.
* **Delivered home: block 00** (`patches/0000`, r13).  A fundamental correctness fix that must precede
  every later block; upstream is not ours to change.  Blocks 01-15 rebased unchanged (their patches
  differ only in the `From <sha>` and hunk context).
* **`upstream/UPSTREAM-PR-mtp-shared-nextn.*`** remains the candidate for when upstream fixes it.
* `nextn_shared_target_tensors` needs no explicit metadata handling: the tensor sharing is already
  driven by `tok_embd == nullptr || output == nullptr` in `llama-context.cpp`, and the arch guard is
  what distinguishes memory sharing from tensor sharing.
