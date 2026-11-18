# The Q8_0 hyper-connection mixer path (r313, block 11) — decode measurement

**Question.** The contributor of PR #326 reported **+7 to +10 %** decode on Flash-Next UD-IQ3_XXS with a host-pool
server configuration, from taking the fused Q8_0 `HC_MIX` op from ~8 launches per hc site to 3-4.  Does it reproduce
on this box, on a different UD file, and does the switch set account for all of it?

**Answer: yes — +7.1 % (interleaved median of 3) on `UD-Q4_K_XL`, and the all-switches-off arm of the same binary
reads exactly the r312 rate.**

## Point

| item | value |
|---|---|
| model | `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` (111 GB, 4 shards) |
| hc class | `blk.0.hc_attn_down` = **Q8_0**, `hc_attn_up` = Q8_0, `hc_attn_inject` = **F32** (the unsloth UD layout the patch set targets; separate tensors, not the post-#31028 merged layout) |
| GPUs | 2 x Radeon AI PRO R9700 (gfx1201), `HIP_VISIBLE_DEVICES=0,2` |
| ROCm | 10.1.0 (`/opt/rocm-10.1.0-gfx120X`) |
| build | `~/bin/build-llama-rocm` (r312 = `8a9e34b57`; r313 = the release tree) |
| command | `llama-cli -m <model> -c 8192 -ngl 99 -fa on -sm tensor --n-cpu-moe 42 -lme auto --load-mode mmap --lazy-mode auto --threads 7 -b 4096 -ub 4096 -p "Write a short paragraph about the Eiffel Tower." -n 128 --seed 42 --temp 0 --no-display-prompt --single-turn` |
| pins | `MOE_EXPERT_HEATMAP=load`; `ROCBLAS_USE_HIPBLASLT` at the default (shipping) for the rate, `=0` for the text arms |
| rate | `llama-cli`'s own footer `[ Prompt: … | Generation: … ]`, one process per arm, arms interleaved, never in parallel |

No MTP head in this table (the PR's own numbers use one); the decode band is the same `nt == 1` graph either way.
The MTP verify-band text and acceptance arms are in `archive/work/hc-mix-q8/README.md`.

## Rates (t/s, 3 interleaved repetitions)

| arm | rep 1 | rep 2 | rep 3 | median | vs r312 |
|---|---|---|---|---|---|
| r312 (`8a9e34b57`) | 45.0 | 45.1 | 46.4 | **45.1** | — |
| r313, all switches default | 49.6 | 48.3 | 48.3 | **48.3** | **+7.1 %** |
| r313, all seven switches `=0` | 45.0 | 45.1 | 46.5 | **45.1** | +0.0 % |
| prompt rate (all arms) | 36.3-36.9 | | | | unchanged |

The all-off row is the control that makes the win attributable: the same binary, with `GGML_CUDA_HC_DONE_PERSIST=0
GGML_CUDA_HC_Q8_EPI=0 GGML_CUDA_HC_UP_PRELOAD=0 GGML_CUDA_HC_UP_FUSE=0 GGML_CUDA_HC_INJECT_DOWN=0
LLAMA_HC_MIX_PLANAR=0 GGML_CUDA_FUSE_HC_COMBINE_MIX=0`, cannot be told from r312 — so the switch set covers every
change and nothing of the gain leaks in from elsewhere.

Second point, same build pair, 5246-token prompt (`prompts/prose-rdna-boosts.txt`, `-f`, MTP head, 96 tokens,
`ROCBLAS_USE_HIPBLASLT=0` pinned): **48.9 -> 51.3 t/s (+4.9 %)**, text `500 chars b30c608e0c9c` on both.

Release build (the tree that ships), 2 reps: **44.7 / 44.8 -> 48.2 / 48.3 t/s (+7.7 %)**, prompt rate 36.7-36.9
in both arms.

## The adaptive-MTP four axes, on the affected model (`-n 512`, `draft-mtp n3`)

The delivery's canonical Protocol A arm is a **dense** 27B and never builds `GGML_OP_HC_MIX`, so it cannot see this
change at all; the meaningful four-axis point is the affected class.  Same command as above plus
`-md mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3`, `-f` each prompt file,
`-n 512`, `--temp 0 --seed 42`, `-lv 4` (the acceptance lines; text is not hashed in this arm):

| axis | prompt | acceptance r312 | acceptance r313 | generation t/s r312 -> r313 |
|---|---|---|---|---|
| **K** (recall) | `prompts/recall.txt` | 0.86385 (368/426) | **identical** | 87.1 -> **92.9** (+6.7 %) |
| **R** (reasoning) | `prompts/reasoning.txt` | 0.67126 (341/508) | **identical** | 81.0 -> **87.4** (+7.9 %) |
| **C** (code) | `prompts/code-python.txt` | 0.50993 (308/604) | **identical** | 69.7 -> **75.9** (+8.9 %) |
| **P** (prose) | `prompts/prose-rdna-boosts.txt` | 0.52872 (313/592) | **identical** | 70.1 -> **75.7** (+8.0 %) |

Every acceptance count is identical **to the drafted and generated totals**, which is the sharpest statement
available that the mixer's logits did not move: the drafter's verify logits decide accept/reject per round.  The
prompt rates (290.1/298.8, 241.8/245.7, 306.6/315.2, 1112.9/1139.4 t/s) are the *unchanged* prefill path and sit
inside their own run-to-run spread.

## Text purity (the same point)

`461 chars sha=8b39a8568222` for r312 **and** for r313 in each of: default, each switch alone, all switches off.
With MTP on (verify band `nt = 4`): `615 chars sha=7fe3c15a90cd` on both builds, draft acceptance `0.68548
(85 accepted / 124 generated), mean len 3.02` on both.  `-sm layer`: `485 chars 876834231fd4` on both.  `-ub 4`
and `-ub 1`: `294 chars 21e32c653820` on both.  Full fusion-arm table:
`scripts/gate-hc-fusions.sh` (see `archive/work/hc-mix-q8/README.md`).

## Reproducing

```bash
# r312 tree at 8a9e34b57, r313 tree from release.json
HIP_VISIBLE_DEVICES=0,2 MOE_EXPERT_HEATMAP=load \
  llama-cli -m /llm/models/Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -c 8192 -ngl 99 -fa on -sm tensor --n-cpu-moe 42 -lme auto --load-mode mmap --lazy-mode auto \
  --threads 7 -b 4096 -ub 4096 -p "Write a short paragraph about the Eiffel Tower." \
  -n 128 --seed 42 --temp 0 --no-display-prompt --single-turn
```

Run each arm three times, interleaved, one process at a time.  A fresh-process *text* comparison additionally
needs `ROCBLAS_USE_HIPBLASLT=0` on that run only (`INVARIANTS.md` §44) and the residency prior pinned
(`MOE_EXPERT_HEATMAP=load`, `GREEDY-PURITY.md` §89); the rate arms above use the shipping default instead, which is
why their numbers are a little below the text arms'.
