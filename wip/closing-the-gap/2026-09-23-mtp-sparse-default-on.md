# Sparse MTP draft prefill — **default ON**

**Status:** promoted to default-on, gated, committed.  Fork `~/llama.cpp`, branch
`gap-closing-hostbuf-integrated`, commit **`df67fd133`**; exported as
[`patches/0026-mtp-sparse-default-on.patch`](patches/0026-mtp-sparse-default-on.patch).
Closes **open item 2** of [`closing-the-gap.md`](closing-the-gap.md).

## Why now

`LLAMA_MTP_SPARSE` was default-OFF only because the campaign's HC16 F32-elision made the sparse
draft's depth text diverge (and the 128K MTP runs nondeterministic).  That was **fixed** by
[`patches/0023`](patches/0023-mmb-hc16-mtp-per-context.patch) (per-backend-context activation state +
whole-graph consumer scan), so the feature is pure with HC16 on and is promotion-eligible.  The
in-code comment still cited the old HC16 reason — now rewritten.

## What the default flips

* The qwen4exp MTP draft uses the trunk's **hybrid-idx memory** (an indexer cache instead of a plain
  KV cache) and attends its **prefill** sparsely (QSA) above
  `LLAMA_MTP_SPARSE_MIN_KV` (default **32768**).
* The **decode/verify** sparse arm stays opt-in (`LLAMA_MTP_SPARSE_DECODE=1`): every measured depth
  lost (the SIMT selected-cell decode + per-step indexer cost more than the dense attention it
  replaces at `n_tokens <= 8`).
* `LLAMA_MTP_SPARSE=0` is now the **opt-out** (plain KV cache + dense draft attention) — the
  default-on policy's disable direction.

## Performance A/B (gfx1151, qwen4exp IQ4_NL + `Q4_K_XL` sidecar, f16 KV, MTP n3)

Arm A = `LLAMA_MTP_SPARSE=0` (plain KV, the old default); arm B = default (hybrid + sparse prefill).

| shape | arm A | arm B | Δ |
|---|---:|---:|---:|
| pp150K prefill-only (`-n 1`, `-b/-ub 2048`) | 937.1 t/s | **1007.9 t/s** | **+7.6 %** |
| pp16K prefill-only (`-n 1`) | 1119.2 t/s | 1115.6 t/s | −0.3 % |
| 8K decode (`-c 8192`, `n 1000`, `-b/-ub 4096`) | 56.5 t/s | 56.4 t/s | parity |
| 40K full (`-c 40000`, `n 300`, `-b/-ub 4096`) | prompt 1106.6 / gen 35.7 t/s | prompt 1116.8 / gen 35.9 t/s | +0.9 % / +0.6 %, text byte-identical |

The shallow cost is the indexer-key store on every draft step (the draft is one token wide, so it is
sub-0.5 %); the deep prefill win is the sparse attention (+7.6 % at 150K, matching the campaign's
earlier +6.9 %).  The 40K full-run text is byte-identical between the arms.

## Gates (same box/model, f16 KV)

| gate | result |
|---|---|
| 8K plain / default / `LLAMA_MTP_SPARSE=0` | all **`3553e76d3a9e`** |
| 40K plain / default `n3` / `LLAMA_MTP_SPARSE=0` `n3` | all **`8285d12d40ca`** |
| 128K plain / default MTP `n_max 1` | all **`d140b40f0eee`** |
| MTP acceptance (`-c 12288 -n 3000`, `--log-verbosity 4`) | **0.85035 (1199/1410)** default and `=0` — the recorded baseline |
| memory engagement | default: `create_memory: MTP context uses a hybrid-idx memory (sparse draft attention)`; `=0`: absent |
| width probe / oracles | unchanged by this switch (no MTP participation); `0025` run: width probe PASS at P=1024/32768, QSA/GDN/INDEXER_TOPK green |

## Files

`src/models/qwen4exp.cpp` (the switch + its comment), `src/llama-model.cpp` (the memory-selection
comment + switch).  No kernel changes — the default flip reuses the `patches/0020` implementation.
