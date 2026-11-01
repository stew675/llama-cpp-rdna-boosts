# Host-expert decode without the per-op routing sync (PR #266) — r203

**Submitted** by `briansp2020` as PR #266 (`wip/pinned-decode-r196`, two `git format-patch` commits against
r196 + a README).  **Promoted** in r203 into **block 09** (engine, scheduler hook, iface member, CUDA hook
wrapper, and the pool-cold read), with one hunk (the `g_dr_key_to_id` reindex inside `moe_cache_forget_all`)
folded into **block 12**, which owns that function, and the CUDA wrapper placed in block 10's
`ggml-cuda-moe-cache-hooks.cu` alongside the other iface wrappers.

Two changes, each with its own kill switch:

* **`GGML_MOE_DEVREMAP`** (default on) — the decode band's routing is mapped to slot ids on the device
  instead of being read back to the host.  The policy stays on the host and runs one call late.
* **`MOE_HOST_POOL_COLD`** (default on) — a `--host-experts pool` miss whose expert is resident in the
  pinned pool is read in place through the pool's device alias instead of being filled into the arena first.

## Why it was accepted

It removes a device synchronize per routed op (the eager path pays `ggml_backend_synchronize(input_backend)`
plus the ids readback's own sync) while keeping every expert read on the same bytes, and it is deliberately
much narrower than the device-*policy* family removed in r40 (`ENVIRONMENT.md` §1.2 records the difference:
host policy, one small map kernel, no device mirrors, no KSLOT, no promotion, no prefill path, and it
declines under `-sm tensor`).

## Validation (this box: 1 x R9700, gfx1201, ROCm 10.1.0; 35B-A3B UD-Q4_K_M `-ncmoe 40`, `MOE_EXPERT_CACHE_MIB=8000`)

| check | eager arm | remap / pool-cold arm |
|---|---|---|
| greedy text, `prompts/prose-rdna-boosts.txt`, `-n 1000` | `3657 chars sha=52eb99f71f16` | identical |
| greedy text, `prompts/recall.txt`, `-n 128` | `942 sha=1b3aa35865e1` | identical |
| 2048 MiB arena (eviction pressure), `-n 192` | `1217 sha=03da3104f3eb` | identical |
| gpt-oss-20b MXFP4 `-ncmoe 20`, `-n 192` | `1317 sha=50b884c1febf` | identical |
| `--host-experts pool` (4 combinations of the two switches), `-n 128` and `-n 1000` | `1b3aa35865e1` / `52eb99f71f16` | identical |
| 2-slot server (`--parallel 2 --kv-unified`), sequential requests incl. slot switching | `450 0b3b17b32e25` / `393 69fc927b3441` / `450 0b3b17b32e25` | identical |
| MTP `draft-mtp`, `-n 3000`, prose | acc **0.76089** = 1397/1836, len 3.28, 110.5 t/s | acc identical, **113.2 t/s (+2.4 %)** |
| MTP `draft-mtp`, `-n 3000`, `prompts/reasoning.txt` | acc **0.83340** = 2141/2569, 117.9 t/s | acc identical, **121.8 t/s (+3.3 %)** |
| prefill logits vs the r201 base (`Qwen3.8-27B-Q8_0`, wikitext) | -- | **0.000000 / 100.000 %** |
| `test-backend-ops -o MUL_MAT_ID` | 931/931 | 931/931 |
| GPU faults (`check-gpu-faults.sh`, journal + logs) | clean | clean |

**Throughput** (`llama-batched-bench`, 1 x R9700; two alternating rounds each unless noted):

| arm | eager | patched | delta |
|---|---|---|---|
| pinned, `-npp 64 -ntg 512 -npl 1` | 60.54 / 60.69 (60.6) | 65.02 / 66.43 (65.7) | **+8.4 %** |
| pinned, `-npp 512 -ntg 128 -npl 4` | 177.04 / 180.60 | 202.28 / 206.14 | **+14.1 %** |
| pinned, `-npp 512 -ntg 128 -npl 8` | 261.42 / 263.20 | 289.61 / 287.31 | **+9.2 %** |
| pinned, `pp512` | 651.31 / 610.40 | 653.81 / 652.31 | unchanged (band-only path) |
| `--host-experts pool`, `-npp 64 -ntg 512` (`MOE_HOST_POOL_COLD=0` vs default) | 57.37 / 57.83 (57.6) | 60.21 / 60.48 (60.4) | **+4.8 %** |
| `-sm tensor` 2 GPU (declines by construction) | 98.02 / 98.42 | 98.06 | inert |

Pool mechanism counters (35B-A3B, `-n 128`, pool arm): `MOE_HOST_POOL_COLD=0` → `fills=21747 colds=0`;
default → `fills=15270 colds=8445`, i.e. 8445 misses served in place from the pool.

## Notes / limits

* The pool-cold arm keys on `ggml_cuda_info().device_count == 1` (process-wide), so on a multi-GPU box
  running a one-GPU arm it is silently off — pin `HIP_VISIBLE_DEVICES` to the one device.
* The remap declines on a pool-backed table (`table_cold_ok` needs a pinned master) and under `-sm tensor`
  (the Meta iface member is NULL), so the two changes never apply to one table at the same time.
* A concurrent (not sequential) two-slot text comparison is **not** an oracle: the unchanged eager build
  disagrees with itself run to run (`869 chars 80d26000974c` vs `768 5452fe998dc6`) — see
  `GREEDY-PURITY.md` §38.  The sequential arm above is the deterministic one.
* The submitter's `alias_find_checked: ignoring a stale MoE-cache alias (…); 1 refused so far` line also
  appears in our runs and on the unpatched tree: pre-existing, not from these patches (the guard refuses
  the lookup and the op falls back to the non-cache path).

## Files

* `submission/README.md`, `submission/0001-*.patch`, `submission/0002-*.patch` — the submission as filed.
* Promoted content: block 09 (engine `moe_cache_remap_dev` + `devremap_*`, the scheduler hook in
  `ggml_backend_sched_copy_input`, the `moe_cache_remap` iface member + NULL fills, the `cudaEventQuery`
  HIP mapping, `MOE_HOST_POOL_COLD`), block 10 (the `ggml_backend_cuda_moe_cache_remap` wrapper in
  `ggml-cuda-moe-cache-hooks.cu`/`.cuh` and the iface registration), block 12 (the `g_dr_key_to_id`
  reindex + `devremap_batch_clear_locked()` in `moe_cache_forget_all`, and the `heat_dec` tally in the lag
  policy).
