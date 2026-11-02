# rdna-boosts two-lane prefill (block 16): LLAMA_PREFILL_LANES

**1 patch** that applies on top of the 16-block rdna-boosts delivery set
(`release.json`, `v16-6050b467c-r205`, applied tree `d6e71901`). Strict
`git am` on the recorded base reproduces the recorded tree `3249229a`
byte-for-byte (validated on a fresh worktree at base `6050b467c`: 16/16
base blocks strict, tree assert, add-on strict, tree assert).

## What it adds

`LLAMA_PREFILL_LANES=2` (default 1) builds two consecutive prefill ubatches
on a second backend scheduler and submits their splits alternately
(`ggml_backend_sched_graph_compute_async_pair`), so one device's routed
experts overlap the other device's attention on a wide prefill. Opt-in;
degrades to the existing single-lane path when < 2 non-CPU backends,
pipeline parallel, `no_alloc`, or a draft context (`ctx_other`) is present.
A ubatch is paired only when it is wide, output-free, and no nextn (MTP) or
F1 layer-input readback is active — MTP and DFlash stay single-lane.

The decisive part is the copy-event design. On HIP a queued
`cudaStreamWaitEvent` resolves to the LATEST `cudaEventRecord` at execution
time. The stock eager-copy path records ONE shared per-device
`copy_event`; with two lanes, the other lane's copies re-record it and
rebind the already-queued consumer waits — every pair serializes
(measured: ~0% gain in every layout tested). The eager path now takes
`cpy_tensor_async_nowait`: the copy is issued on the source stream and the
caller's per-copy ring event is recorded there; the consumer waits on its
own event. No shared event is ever re-waited. `cpy_tensor_async` keeps its
old behavior for its two remaining callers.

Surface: `llama_context` second scheduler + lane-aware
prepare/graph_reserve/compute; `ggml_backend` lane-aware split build and
pair submission; stream-ordered host input copies; `llama_n_prefill_lanes()`
accessor; `GGML_SCHED_PAIR_DEBUG` pair/hazard diagnostics with per-lane
per-backend GPU event spans (`ggml_backend_event_elapsed_time`, an optional
device-iface hook, `hipEventElapsedTime` on the unified CUDA/HIP backend).

## Reference layout (MoE on eGPU + APU)

`-sm layer -ts 1,0 --fit off -fa on -ub 2048 -b 4096 -np 1`, routed-expert
layers steered to the second card via
`-ot 'blk.<moe-idxs>.ffn_(gate|up|down)_exps=ROCm1'`. Lanes never move
work: `-ts` placement decides which card owns which tensors, the interleave
only changes submission order.

## Validation (slavap-pd-7900: RX 7900 XTX gfx1100 + Radeon Pro V620 gfx1030,
gfx1030;gfx1100 dual build, llama-cli prompt-eval, p8192, ABBA 3 rounds)

| shape | 1 lane | 2 lanes | gain |
|---|---|---|---|
| dense (Qwen3.8-27B Q6_K_XL, `-ts 1,1`) | 511.9 t/s | 580.2 t/s | **+13.3%** |
| MoE prod-mirror (Qwen3.6-35B-A3B Q8_0, 38 expert layers → v620) | 2113.9 t/s | 2835.7 t/s | **+34.1%** |

- the pair engages on every 2-lane arm (`pair decided`, `interleaved=1`);
  intra-arm spread < 0.4%, no ABBA-order drift
- greedy trajectories: the 2-lane runs stay in the 1-lane jitter family.
  Box-level FP non-determinism: the 1-lane arm is itself non-deterministic
  across runs on this box, so byte-identity is not a valid gate here
- MTP (`-md`): the draft context (`ctx_other != nullptr`) stays single-lane,
  the target engages; draft acceptance unaffected, no aborts
- decode: the 2-lane arm runs the stream-ordered input copies on the main
  scheduler (halo parity) — tg128 70.96 → 81.87, no regression
