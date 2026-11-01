Thank you for r191-r196, and for the careful MTP numbers on #260 / #262 (we've replied there). This one is separate. It covers decode with host experts: `-ncmoe` with the default `--host-experts pinned`, and `--host-experts pool`. Two patches on r196, each with its own kill switch:

**0001: sync-free decode device remap for `--host-experts pinned`** (`GGML_MOE_DEVREMAP=0`: off).
- The eager path reads the routing back to the host before each routed op: one device synchronize per op. It then runs the residency policy and uploads the remap.
- With 0001, `ggml_backend_sched_copy_input` first offers the input to the backend through a new `moe_cache_remap` iface member. It is the last member and NULL in the meta, cpu and rpc backends.
- A kernel maps the device routing to slot ids through a per-table expert -> slot map, where `slots + e` means cold: read in place from the pinned master.
- The same launch writes the routing into mapped pinned memory. The residency policy reads it at the next call (one call late) and re-uploads the map only after a residency change. One launch serves a layer's tables (gate / up / down).
- Every expert is read from the same bytes as before (its arena slot or the pinned master), so the output does not change.
- Scope: pinned masters, a single slice, no identity table, not under `-sm tensor`, and only up to the decode / verify band. The eager path still primes and sizes the cache.

**0002: `--host-experts pool` reads a pool-resident miss in place** (`MOE_HOST_POOL_COLD=0`: off).
- A pool-backed table has a pageable master, so the in-place cold read was off. Every decode miss was an arena fill, even when the expert was already in the pinned pool.
- With 0002, such a miss is read cold from its pool slot through the pool's device alias (cold id `n_res + pool slot`; `moe_cache_get_cold` returns the pool base for the table). These are the same bytes the fill would have copied.
- The slot stays valid for the read: the demand path never evicts a pool resident, and the scheduler synchronizes the backend before the next hook. Single device, single slice.

**Gates**
- llama-server greedy, 4 chats x 3 turns: byte-identical replies with each switch on vs off.
  - 7900 XTX, Qwen3-30B-A3B: both modes, on r191, r195 and r196.
  - R9700, Flash-Next: the patch matched stock r196 on every arm.
- 0 GPU faults.

**RX 7900 XTX (gfx1100)**, ROCm 10.1, r196 + the patches. Qwen3-30B-A3B Q4_K_M (unsloth) requantized with `ffn_gate/up_exps` IQ3_S and `ffn_down_exps` IQ4_NL (14.7 GB), `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 --n-cpu-moe 48`, `MOE_EXPERT_CACHE_MIB=3000 GGML_CUDA_MOE_RPB_WIDE=0` (the same in every arm). `llama-batched-bench -npp 64 -ntg 512 -npl 1`, TG t/s, two alternating rounds:

| | switch 0 | patch | |
|---|---|---|---|
| `--host-experts pinned` (0001) | 71.4 / 72.1 | **80.1 / 81.1** | +12.3 % |
| `--host-experts pool` (0002) | 36.3 / 36.3 | **39.7 / 38.5** | +7.6 % |

**Radeon AI PRO R9700 (gfx1201)**, ROCm 10.1, r196 + the patches against stock r196. Flash-Next GSQ-RCO IQ3_S on 1 GPU, `-ncmoe 47`, MTP with a Q4_K draft head, q8_0 KV, 256K context, your default slab placement. llama-server greedy t/s, two passes per run, arms alternated:

| | stock r196 | patch | patch, switch 0 |
|---|---|---|---|
| `--host-experts pinned` | 73.1, 71.7 / 73.4, 72.6 (72.7) | **74.8, 73.3 / 76.7, 73.9 (74.7)** | 72.8, 72.4 (72.6) |
| `--host-experts pool` | 29.3, 29.1 (29.2) | **38.9, 38.6 (38.75)** | |

Pinned is +2.7 %, and with the switch at 0 it matches stock, so the remap is the whole gain. Pool is +33 %. Greedy replies matched stock on every arm, with the same 16 slab evictions on stock and patch.

Notes:
- The pinned gain is smaller on the R9700 run. We haven't broken that down: the model, MTP and the card all differ from the 7900 XTX run, so we can't say yet which of them matters.
- The remap kernel's completion flag uses `__hip_atomic_store` on HIP and `atomicExch` after `__threadfence_system()` on CUDA. The CUDA side is untested: we have no NVIDIA card to build or run it on.
- Both stock and patched builds log one `alias_find_checked: ignoring a stale MoE-cache alias (table device 0 layer 5, op device 0 layer 3)` at load on r196 (layer 1 / layer 0 on r191), so it isn't from these patches. Flagging it in case it's useful; we haven't looked into what triggers it.

Files: `0001-*.patch` and `0002-*.patch` (`git format-patch` against r196; they apply with `git am`, and also cleanly to r195 and r191) and this README. Happy to run any other shapes on either card. No rush on any of this.

(Claude-assisted, as before.)
