# wip/mtp-draft-op-offload

An opt-in switch that builds the MTP draft context with `op_offload = false`. With the drafter's experts on the host
(`-otd exps=CPU`), it frees the 2.55 GB of VRAM that the draft context otherwise reserves to stage them. One `git am`
patch, one file (`common/speculative.cpp`, +9):

- `0001-spec-LLAMA_MTP_DRAFT_OP_OFFLOAD-0-keeps-the-MTP-draf.patch` (r6 tree `2b57533c` -> `e148ca1e`).

Usage: `LLAMA_MTP_DRAFT_OP_OFFLOAD=0`. Unset (or any other value) keeps the current behaviour. The target context is
never touched.

## Why

With `--spec-type draft-mtp -otd exps=CPU`, the MTP layer's experts stay in host memory. For the qwen4exp Q8_0 head,
these are `blk.48.ffn_{gate,up,down}_exps`, 850 MiB each.

For prefill-sized draft batches (the drafter's catch-up decode of each prompt chunk), op offload runs those
`MUL_MAT_ID`s on the device. It reserves room for full-size copies of all three tables in the draft context's compute
buffer. From a verbose load at 256K:

| draft context | MiB |
|---|---|
| weights on device (head / embeddings shared with the target) | 97 |
| KV (f16; 306 with `-ctkd q8_0 -ctvd q8_0`) | 576 |
| compute buffer | **2,550** = 3 x 850 staging copies |

The draft context already caps `n_outputs_max` at 4, so the buffer isn't logits. With the switch, those ops run on the
host in place and the 2,550 MiB reservation goes away. Measured right after load with otherwise identical flags, VRAM
is 16.2 GB with offload on vs 15.1 GB with the switch. On a 32 GB card at 256K, that is the difference between
MTP plus a 4 GB `MOE_EXPERT_CACHE_MIB` and having to shrink the cache to make room for the drafter.

## Cost

We expected the host-side catch-up to cost prefill speed and draft-step speed. We couldn't measure a cost in either.

R9700 (gfx1201, PCIe 5.0 x16), Ryzen 9 9900X, Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, shared Q8_0 MTP head, r6 + the
patch. Setup: 256K, `-ncmoe 48 -ub 2048 -b 2048`, q8_0 KV, `GGML_SCHED_STAGE_SLOTS=16 GGML_SCHED_STAGE_MAX_MB=8192`,
MTP n3 `--spec-draft-p-min 0.5 -ctkd q8_0 -ctvd q8_0`, `LLAMA_MTP_SPARSE=0`, default slots. All values are t/s,
one run each; run-to-run noise is a few percent.

| | greedy | prose x2 | code x2 | 2 concurrent (sum) | prefill 37k | prefill 155k |
|---|---|---|---|---|---|---|
| no MTP, MIB 4096 | 42.1 | 42.6 / 43.8 | 42.2 / 39.0 | 58.1 | 1750 | 1665 |
| MTP, switch on, MIB 4096 | 54.6 | 53.4 / 45.9 | 57.5 / 60.0 | 53.9 | 1924 | 1665 |
| MTP, switch on, MIB 2048 | 49.1 | 44.3 / 45.5 | 52.4 / 55.0 | 48.5 | 1883 | 1465 |
| MTP, switch off (current), MIB 2048 | 47.9 | 47.2 / 45.3 | 44.3 / 50.5 | 46.6 | 1881 | 1676 |

- At the same cache size, the switch is no slower at decode and in some runs faster.
- With the switch, MTP prefill on a 155k prompt equals no-MTP. The 1,465 in the third row is an outlier we didn't
  re-run.
- Greedy output is `c39d78416cd1` in every row.
- A 259.6k-token prompt gave the correct needle answer, but prefill was 1,412 t/s vs 1,729 without MTP. We haven't
  separated how much of that comes from the drafter's catch-up and how much from VRAM pressure at that depth.

Other drafter levers we tried had no measurable effect: `-td 24 -tbd 24`, and the drafter's experts requantized to
Q4_K (acceptance unchanged).

## Alternatives

- A CLI flag mirroring the existing `--no-op-offload` (e.g. `--no-op-offload-draft`) would be the more upstream-style
  spelling. We kept an env var to match the other `LLAMA_MTP_*` switches.
- It could also be automatic: off whenever the draft model has tensor overrides to a host buffer type.
