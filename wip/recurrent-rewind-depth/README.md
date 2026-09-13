# Issue #30 — is depth 15 safe? (2026-09-13)

Investigation record for the `--spec-draft-n-max` clamp policy.  Outcome: **no rewind corruption at
depth 15; the clamp moved to 15 with a purity notice above 7; the qwen4exp depth-15 divergence was a
QSA decode-arm band flip, now fixed in block 14.**  Delivery record: `WORKLOG.md` 2026-09-13 (latest),
`patches/README.md` (the issue-#30 section), `GREEDY-PURITY.md` §11/§32.

## Tools

* `wallp.cpp` — width-purity matrix, real models: load once, then for each `W` in a list build a fresh
  context, prefill `P`, decode a `W`-token batch, hash the token-0 logits.  Build with `build.sh`
  (reuses the `test-recurrent-state-rollback` compile/link flags of an existing build dir).
* `wprobe.cpp` — the single-width variant (one process per width, uses `W=`, `RS=` env).
* `build.sh <src.cpp> <out>` — compile a probe against `~/llama.cpp/build-rocm`.
* `run-arm.sh` / `run-qwen4.sh` — the end-to-end greedy text arms (dense 1-GPU / qwen4exp 3-GPU tensor).

The deterministic snapshot sweep itself is now in-tree as `tests/test-recurrent-state-depth.cpp`
(`test-recurrent-state-depth` + `test-recurrent-state-depth-qwen4exp`).

## The decisive width matrix

`wallp model text P ubatch W1,W2,...` with `SPLIT=tensor HIP_VISIBLE_DEVICES=0,1,2 RS=15`, `P=2500`
(crosses the QSA selection width 2051), f16 KV.

| model / build | W=1..8 | W=9..16 |
|---|---|---|
| qwen4exp IQ4_XS, pre-fix | `643a8166d8dad677` | `1354757f9daf03db` (== `LLAMA_QSA_DENSE_DECODE_UNTIL=0`, i.e. sparse) |
| qwen4exp IQ4_XS, post-fix | `643a8166d8dad677` | `05be2f7f30dbc426` (dense arm) |
| qwen4exp IQ4_XS, `LLAMA_QSA_OFF=1` | `1966a4f400e647a0` | `19febd036df87048` |
| 27B UD-Q4_K_XL, fa auto | `8ef5ce3ab2d942dd` | `7d1e01e82be4ce6b` |
| 27B UD-Q4_K_XL, `FA=0` | `b5d4df87caa0348e` | `0c0cb329b59aedc9` |

The last two show the residual `W=9` difference is **not** the QSA arm and **not** FA alone: the matmul
family changes at `ncols == MMVQ_MAX_BATCH_SIZE`/`MMVF_MAX_BATCH_SIZE` = 8 (MMVQ/MMVF decode kernels
below, MMQ above).  That is the documented accepted purity trade above depth 7.

On the dummy `qwen4exp-moe` the QSA flip is isolated with no matmul-family confound: pre-fix W=1..8
`5009c55bca5e01ca` vs W=9..16 `596ec8bf7461da1a`; post-fix all W=1..16 `5009c55bca5e01ca`, pure for all
eight native KV types (`wprobe` with `RS=15`).

## The snapshot sweep

`test-recurrent-state-depth` (Phase A: `n_tokens = K = n_rs_seq+1`, every rollback 1..`n_rs_seq`;
Phase B: deep draft `n_tokens = n_rs_batch > K`) is **green for `n_rs_seq` 1..15** on `qwen35-dense`,
`qwen4exp-moe`, `deepseek4-moe`, `kimi-k3-moe` — no corruption.  It compares against a reference
context that never decoded past the rollback point (bitwise).

## End-to-end text arms

27B UD-Q4_K_XL, 1 GPU, f16, code-replay prompt, 3000 tokens: `plain == n_max 7` = `57776c25503d`;
`n_max 15` = `269a445fe4e8` (coherent; a near-tie flip, not corruption).

qwen4exp IQ4_XS, 3-GPU tensor, code-replay, 3000 tokens: `n_max 7` 57.9 t/s, `n_max 15` 40.9 t/s
(depth 15 over-drafts on that prompt — not a correctness signal).

## Commands

```sh
cd ~/llama-cpp-rdna-boosts/wip/recurrent-rewind-depth/tools
REPO=~/llama.cpp ./build.sh wallp.cpp "$PWD/wallp"
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:~/llama.cpp/build-rocm/bin
HIP_VISIBLE_DEVICES=0,1,2 SPLIT=tensor RS=15 \
  ./wallp /models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt 2500 2048 1,2,7,8,9,12,16
```
