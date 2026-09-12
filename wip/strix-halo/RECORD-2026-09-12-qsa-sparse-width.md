# 2026-09-12 — QSA *sparse*-regime width purity on gfx1151 (item 4 / item 7 disposition)

Box: Strix Halo APU (gfx1151), 1 device, ROCm 7.14 (`/opt/rocm-7.14-gfx1151`).
Build: the delivered 15-patch set, worktree `/home/stew675/ll25/verify` (tip `9f46e926f`, tree
`f4791066f4a582316b1ca95f51c96cd10b905ef7`, `build= b10881-9f46e926f`).
Model: Qwen3.8-Flash-Next UD-IQ4_XS + the Q4_K_M MTP draft.  Gate: `qsa-text-gate` style —
`plain (--spec-type none)` vs `draft-mtp --spec-draft-n-max 3`, greedy, `--seed 42 --temp 0`,
text hashed with `tools/textgen.py`.

## Why this run

`TODO.md` item 4 recorded **two** width-dependences in the QSA *sparse* regime (the regime gfx1151
uses for decode **above its 64K crossover**), and item 7 proposed making gfx1151 dense-decode at
every depth on purity grounds.  Both were measured on 2026-09-11 (the 3-GPU gfx1201 box, sparse arm
forced with `LLAMA_QSA_DENSE_DECODE_UNTIL=0`) during the cause-3 hunt, i.e. **before** the
2026-09-12 block-13 RDNA3_5 mmvq-fusion amendment (`GREEDY-PURITY.md` §25).  This record re-measures
them on gfx1151 with the current delivery.

## 1. The two recorded items do not reproduce on gfx1151

**(a) The fused indexer score is byte-identical to the per-op chain.**  Forced sparse, qwen4exp, 512
greedy tokens, `P=5000`, f16 KV — four configurations, plain and n3 identical in each:

| config | plain | n3 |
|---|---|---|
| default (`GGML_CUDA_QSA_INDEXER_SCORE=1`, `CACHE=1`) | `0d29890e0f04` | `0d29890e0f04` |
| `GGML_CUDA_QSA_INDEXER_SCORE=0` (per-op chain) | `0d29890e0f04` | `0d29890e0f04` |
| `GGML_CUDA_QSA_INDEXER_CACHE=0` (no derived cache) | `0d29890e0f04` | `0d29890e0f04` |
| `GGML_CUDA_QSA_INDEXER_CACHE=2` (unfilled-pool **probe**) | `64244713deb3` | `cb2912b186b9` |

The `CACHE=2` row is the positive control: the fused path **is** the one running (passing the
unfilled pool moves the W=1 text), and it differs from the per-op chain *only* when deliberately
corrupted.  bf16 is the same (`945f89766e3c` for default and `SCORE=0`; `CACHE=2` moves W=1).  So
the "not byte-identical" claim from the 2026-09-11 measurement does not hold on gfx1151.

**(b) The residual split was the block-13 mmvq fusion.**  Forced sparse, f16, `P=5000`:

| build knob | plain | n3 |
|---|---|---|
| current delivery (fusions skipped on RDNA3_5) | `cb2912b186b9` | `cb2912b186b9` |
| `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` (pre-fix) | `471ea250f8e2` | `cb2912b186b9` |

The opt-in reproduces the pre-fix impurity exactly, and the shipped fix removes it — i.e. the
"common prefix 706 chars then divergence" was the dense gate+up+GLU / weighted-down single-token
fusion non-byte-identity (§25), not the indexer machinery.  (`GGML_CUDA_DISABLE_FUSION=1` also
reconciles, consistent with a fusion.)

## 2. Default gfx1151 configs are pure

plain == `draft-mtp n_max 3`, byte-identical:

| config | depth | KV | text |
|---|---|---|---|
| shallow dense decode (default: `n_kv < 64K`) | 100 | f16 / bf16 / q4_0 / q4_1 / iq4_nl | all pure (see §3) |
| shallow dense decode, q8_0 | 100 | q8_0 | `e8f8bba3942b` |
| **deep sparse decode (default: `n_kv > 64K`)** | ~74K | f16 | `83e0ed0f0f80` |
| **deep sparse decode (default), q8_0** | ~74K | q8_0 | `7205399d367d` |

The q8_0 deep row is the maintainer's actual serving config (`--cache-type-k q8_0 -ctk 163840`
class).  **The 64K crossover stays — there is no purity reason to make gfx1151 dense-at-every-depth
(item 7).**

## 3. One residual, q8_0-only and prompt-dependent (open)

With the sparse arm **forced** at shallow context (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`), qwen4exp +
`-ctk q8_0` on `/tmp/p5000.txt` diverges, reproducibly: `plain a57bc13bbf2a` vs `n3 3124adfd2b94`.
It is **not** universal:

| KV | forced-sparse, `p5000` | forced-sparse, `longprompt` | forced-sparse, `iq4_nl` etc. |
|---|---|---|---|
| f16 / bf16 | pure | pure | — |
| q4_0 / q4_1 / iq4_nl | pure | — | pure |
| **q8_0** | **diverge** | pure (1023 chars; 512-token too) | — |

and the default deep q8_0 config is pure (§2), so this is a ULP-level width dependence that only
flips a token on some states/prompts.

Localisation so far (each a full plain-vs-n3 pair on the `p5000`/q8_0 case):

| knob | reconciles? |
|---|---|
| `LLAMA_QSA_OFF=1` (no indexer at all) | yes |
| `LLAMA_QSA_SPARSE_FA=0` (standard masked FA) | **no** → not the fused `fattn-qsa` kernel |
| `GGML_CUDA_DISABLE_FUSION=1` | yes (perturbation or a fusion) |
| `GGML_CUDA_GDN_CHUNKED=0` | yes (perturbation or the chunked GDN path) |
| `GGML_CUDA_DISABLE_HC_FUSION/_MIX/_COMB=1` | no |
| `GGML_CUDA_SCALE_UNARY=0` | no |
| `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` + `..._WEIGHTED_DOWN=1` | no |
| `GGML_CUDA_DISABLE_MMQ_ROUTED=1` | no |

`GGML_CUDA_DISABLE_MMVQ_MAT` (used in an early run) **does not exist in the source** — that test was
void.  The two "reconciles" rows are both arithmetic perturbations (they move the whole stream, not
just the diverging width), so they localise nothing by themselves; the honest state is **root cause
unlocalised**.  Next step is a node-dump / op-trace rebuild (the `GGML_CUDA_OP_TIMING` path is not
compiled into the shipped build; `tools/node-dump-instrumentation.patch` is the instrument) to diff
the W=1 and W=4 graphs and their intermediate tensors in the sparse regime.

## 4. Disposition

* **TODO item 4** — re-scoped: the two originally-recorded items (fused indexer score, residual
  split) are resolved/not-reproducing on gfx1151; the q8_0 sparse residual above is the one open
  piece, low severity (default configs pure, prompt-dependent under a forced arm).
* **TODO item 7** — closed: the sparse regime is pure in the default configs, so the "dense decode
  at every depth" workaround has no purity driver; the 64K crossover stands.  A pure per-*perf*
  MTP-side crossover re-measure is parked, not needed.
* `GREEDY-PURITY.md` §18, `AGENTS.md` (the §18 bullet) and `TODO.md` are updated to match.

## Repro

```sh
cd /home/stew675/ll25/verify
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
M=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
D=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
for spec in "--spec-type none" "--spec-type draft-mtp -md $D --spec-draft-n-max 3"; do
  LLAMA_QSA_DENSE_DECODE_UNTIL=0 ./build-rocm/bin/llama-cli -m "$M" $spec -f /tmp/p5000.txt \
    -n 128 --seed 42 --temp 0 --single-turn --no-display-prompt -c 8192 -b 2048 -ub 2048 \
    -ctk q8_0 -ctv q8_0 -fa auto -ngl all -sm layer -mg 0 2>&1 \
    | python3 /home/stew675/llama-cpp-rdna-boosts/wip/kv-quant-purity-followups/tools/textgen.py /dev/stdin
done
```
