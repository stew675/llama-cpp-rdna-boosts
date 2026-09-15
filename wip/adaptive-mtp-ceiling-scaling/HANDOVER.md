# HANDOVER — adaptive-MTP ceiling scaling

Turnkey brief for the next session.  Read [`README.md`](README.md) first (the finding, the data, the
confirmed/not-confirmed split).  **This is not issue #30.**

## TL;DR

Confirmed on `v16-d1d3c3396-r1`: **Qwen3.8-27B Q8_0, 2-card `-sm tensor`, adaptive
`--spec-draft-n-max 12` loses ~5-6 % to `n_max 7`** (code and prose, f16 and BF16 KV).  Depth 10 is
between the two.  1-card Q8_0 still wins from 12, and Q4/Q6 2-card still win here — so the loss is
specific to **Q8_0 × tensor split**, and it is a *tuning/performance* issue, not a purity bug.

## Setup

```sh
# Delivery build (already built once):
cd ~/llama.cpp && git worktree add -f ~/llama-cpp-rebase rdna-boosts-v17
cd ~/llama-cpp-rebase && BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714     # ~6 min, gfx1201
# Stock reference at the same fork point, if needed:
cd ~/llama.cpp && git worktree add -f ~/stock-d1d3 $(git rev-parse d1d3c3396)
cd ~/stock-d1d3 && BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
```

Models: `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf`, `.../Q6_K/...`,
`.../Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf`.  Prompts under `prompts/`.

## Reproduce

```sh
wip/adaptive-mtp-ceiling-scaling/repro.sh new        # the whole sweep (~20 min)
# the key cell by hand:
HIP_VISIBLE_DEVICES=1,2 build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf \
  -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off \
  -p "$(cat prompts/code-python.txt)" --spec-type draft-mtp-adaptive --spec-draft-n-max 7 \
  -sm tensor -ts 1/1 -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
# repeat with --spec-draft-n-max 12; compare the footer Generation t/s and the -lv 4 acceptance.
```

**llama-cli MUST always be invoked with `--single-turn`** (else it blocks in the chat loop).  Use a
separate run per cell; do not run benches in parallel with anything else.

## Where to look (delivery blocks)

| area | files / notes |
|---|---|
| adaptive controller, `--spec-draft-n-max` | block 01; `wip/issue-30-*` and `patches/README.md` block-01 notes; the ceiling-12 record |
| wide-verify matmul family (MMVQ/MMVF → MMQ at `ncols == 8`) | block 08/10/13; `patches/README.md`; `GREEDY-PURITY.md` §11/§19 |
| FA chooser (tile/MMA at `n_q > 8`) | block 00/03/04/08; `GREEDY-PURITY.md` §11 |
| tensor-split dispatch / AR | block 12; but note the reporter **ruled the AR out** (P2P/internal A/B) |
| prefill `ncols2` split hint | `ggml_set_fa_tensor_parallel` — **ruled out** (prefill only; forcing 0/1 left decode flat) |

## Gates

Before proposing a change, run:

1. The key cell above, `n7` vs `n10` vs `n12`, on **Q8_0 2-card** — the change must remove the loss.
2. `benchmarks/mtp-adaptive-methodology.md` rule 5 (verify-width): `llama-batched-bench -npp 16 -ntg 32
   -npl 1,4,8` with a quantized KV on a dense K-quant — no regression at B=4/B=8.
3. Purity: `--spec-type none == draft-mtp --spec-draft-n-max 3` byte-identical (27B, all supported KV
   types).  A ceiling cap must not change single-token decode or the `<= 7` band.
4. The four-axis gate (`-n 3000`) to confirm the Q4/Q6 single-card wins are untouched.
5. Cross-arch spot check on `halo` (gfx1151) — the ceiling default is arch-independent, but the
   `n_gpu` cap interacts with split mode.

## Acceptance criteria for a fix

* Q8_0 2-card, code/prose, `n12` no longer below `n7` (ideally within noise; a cap at 7 is acceptable
  since `n7` is the observed optimum there).
* Q4/Q6 single-card ceiling-12 wins unchanged (the block-01 feature is not lost).
* No purity change anywhere in the `n_max <= 7` band.
* Document the rule (and, regardless of a code fix, add the README/methodology caveat the reporter
  asked for).

## Do not re-derive

* The AR is not the cause (reporter A/B).  The `ggml_set_fa_tensor_parallel` hint is prefill-only.
* BF16 vs f16 KV is not the cause (same shape; bf16 ~2 % faster absolute).
* The absolute t/s in this dossier are our local cli footers; the reporter's and the historical server
  record are 10-15 % lower on the same cell.  Compare ratios, not absolutes.
* A separate, untriaged item from the same report: ROCm **7.2.4** breaks purity (clean on 7.14) — a
  toolchain note, not this investigation.
