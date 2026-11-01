# The GPU-resident non-weight budget — inventory, scaling laws, and how to measure it

Everything below was measured or read on **r174** (`v15-6050b467c-r174`, build `5bdec04ab`, gfx1201,
`/opt/rocm-10.1.0-gfx120X`).  One R9700 (32 624 MiB) was attached for these runs, so all numbers are 1-GPU;
the multi-GPU arms only change *where* the buffers are, not their size.

## 0. Taxonomy — say which "metadata" is meant

**Scope (2026-10-29): this campaign is the KV cache.**  T2 and the KV cache's own bookkeeping are in scope;
T3, the non-KV parts of T4 and T5 are recorded here only so the boundary is explicit and a later exploration
starts from measurements instead of re-deriving them (`README.md` §10).

The ask ("reduce the metadata held on the GPU") can mean four different things, and they have very different
prices.  This campaign keeps them apart; §2 is the byte mass, §3 is bookkeeping.

| tier | what | scaling | device? | in scope |
|---|---|---|---|---|
| T1 | weights | model | always | — |
| T2 | **KV data** — K/V per layer per stream, the qwen4exp **indexer keys**, the DSA light-indexer keys | `O(n_ctx)` | always | **yes** |
| T3 | **state data** — GDN recurrence `S` + conv `R`, MLA/latent caches, F1 layer-input stores | `O(1)` per layer, `O(ub)` for the stores | always | no (deferred) |
| T4 | **bookkeeping (metadata proper)** — kq mask, `cell_pos`/`tok_lo`/`tok_hi`, per-stream cell copies, the expert cache's page/remap/heat tables, graph input scalars | `O(n_ctx)`, `O(n_ctx×ub)`, `O(#experts)` | mostly | **only the KV cache's own** (mask inputs, cell metadata, a residency map) |
| T5 | **scratch** — the scheduler's compute buffers (per graph, reserved at the widest shape), the FA prefill staging arena, the H2D ring, the q8_1 arena | `O(ub)`, `O(ub²)` | yes | no (deferred) |

Only T4 is "metadata" in the strict sense, and on the shipped default paths it is **small** (§3) — the
r160/r172 work already removed the one T4 term that was not (`kq_mask` packed).  The byte mass in the ask is
T2, and T2 is what a streaming design has to move.  T5 is scratch, and shrinking it is a different campaign
(the slab's narrow/wide regions).

## 1. Measured: four representative shapes

Log lines verbatim (`-v`), commands in §5.

### A. dense 27B UD-Q4_K_XL + MTP draft, `-c 65536`, 1 GPU

```
llama_kv_cache: size = 4096.00 MiB ( 65536 cells,  16 layers,  1/1 seqs), K (f16): 2048.00 MiB, V (f16): 2048.00 MiB
llama_kv_cache: size =  256.00 MiB ( 65536 cells,   1 layers,  1/1 seqs)   <- the MTP draft context
llama_memory_recurrent: size = 598.50 MiB (1 cells, 64 layers), R (f32): 22.50 MiB, S (f32): 576.00 MiB
sched_reserve: ROCm0 compute buffer size = 132.27 MiB   |   ROCm_Host compute buffer size = 40.28 MiB
llama_context:  ROCm_Host  output buffer size = 0.95 MiB
```

Weights 16.4 GiB, T2+T3 = **4.8 GiB** (29 % of the weights), T5 = 0.17 GiB, T4 ≈ 1 MiB (derived mask).

### B. 35B-A3B Q4_K_M, `-ncmoe 24`, `-c 65536`, 1 GPU (MoE + slab)

```
llama_kv_cache: size = 1280.00 MiB ( 65536 cells,  10 layers), K (f16): 640.00 MiB, V (f16): 640.00 MiB
llama_memory_recurrent: size = 62.81 MiB (1 cells, 40 layers), R (f32): 2.81 MiB, S (f32): 60.00 MiB
moe_cache_preflight: auto cache preflight: device 0 projected arena 31534.0 MiB (free 32558.0 - reserve 1024.0 - aux 0.0)
SLAB_INIT dev=0 free=21.21 GiB reserve=1.00 GiB headroom=0.75 GiB size(VA)=20.44 GiB mapped=20.19 GiB
                boundary=0.56 GiB va_spare=0.25 GiB chunk=64.0 MiB ring_req=2.05 GiB
ggml_cuda_slab_init_locked: slab 20.19 GiB (chunk 64.0 MiB): work 0.56 GiB + arena 19.62 GiB
ggml_backend_cuda_buffer_type_alloc_buffer_usage: compute buffer 561.00 MiB from the slab work region
alloc_all_locked: MoE expert cache (auto): arena 11238.0 MiB of 11238.0 MiB host experts (100.0% residency)
```

### C. the same, `-c 262144` — the KV is taken out of the arena, 1:1

```
SLAB_INIT dev=0 free=17.46 GiB (was 21.21)  ->  size(VA)=16.69 GiB mapped=16.44 GiB
ggml_cuda_slab_init_locked: slab 16.44 GiB: work 0.56 GiB + arena 15.88 GiB   (was 19.62)
```

3840 MiB more context → 3.75 GiB less arena, because **the KV is allocated before the slab exists** and the
slab maps `free VRAM - reserve` (`ggml-cuda-slab.cu`, `SLAB_INIT`).  The arena was not the binding constraint
here only because 11 238 MiB of host experts fit in either; on a shape where it does bind, every 1 024 tokens
of context is ~20 MiB (35B-A3B) of lost expert residency, and the r117 identity cliff (46.7 → 65.0 t/s) is
what that turns into.

### D. The link constant, and the cost of a non-resident KV byte

> **RE-BASELINED 2026-10-30: the box was re-cabled and the link DOUBLED.**  The values immediately below are the
> historical 28.8 GB/s calibration that every number in this campaign was measured at; the current box measures
> **55.7 GB/s H2D / 56.8 D2H / 55.7 GB/s kernel read of pinned host memory**, i.e. `bytes / 55.7 GB/s` is the
> present cost model and every earlier t/s figure in this campaign must be re-measured before it is compared.
> See `A2-MIXED-BACKING.md` §3.7 for the new numbers and the re-run probe output.

`archive/work/kv-residency/bw-probe.cc` (`hipcc -O3 --offload-arch=gfx1201`), 512 MiB per arm, 5 reps, best of:

```
device: AMD Radeon AI PRO R9700 (gfx1201), 512 MiB per arm, 5 reps
  H2D pinned                18.65 ms     28.8 GB/s
  H2D pageable (wall)       18.71 ms     28.7 GB/s
  D2H pinned                19.07 ms     28.2 GB/s
  D2D                        1.84 ms    291.4 GB/s
  kernel read pinned        18.51 ms     29.0 GB/s
  kernel read VRAM           0.95 ms    565.0 GB/s
```

The card itself negotiates **x16 @ 32 GT/s**, but it sits behind a PCIe switch whose **host uplink is x8**
(`01:00.0 -> 00:01.1`: `max_link_width 8`), so ~28.8 GB/s is the ceiling in *both* directions, and the
delivery's own calibration agrees (`ggml_backend_cuda_stage_h2d_gbps: 28.7 GB/s`).  Read the **root port's**
width, not the card's — on a box where the card is x16 straight into the CPU this constant doubles to ~55-57
GB/s and every cost below halves.  A kernel reading pinned host memory through its device mapping gets the
full link rate (**29.0 GB/s**), so a Strata-style resolve copy costs exactly `bytes / 28.8 GB/s`.

**The all-host-KV arm, for scale.**  Qwen3.5-4B Q8_0, `llama-bench -n 128 -d 4096`, 3 rounds, same binary:

| arm | pp512 @ d4096 | tg128 @ d4096 |
|---|---:|---:|
| KV in VRAM | 7640.14 ± 476.95 | **100.67 ± 0.44** |
| `-nkvo 1` (pageable host KV, FA on GPU) | 5060.95 ± 137.79 | **38.71 ± 0.16** |

The 4B's KV is 32 KiB/token, so a full-attention token at depth 4096 reads 128 MiB of cached K/V.
Baseline is 9.9 ms/token; the host-KV arm is 25.8 ms/token → **~16 ms for those 128 MiB, ≈ 8 GB/s
*effective***, against a 4.4 ms link floor.  The gap is the access pattern (pageable, one round of latency per
access, no staging), i.e. the all-host arm is *worse* than the link constant, not better — a staged /
pinned read is the 28.8 GB/s figure above.  Either way the law stands: **every byte attention reads this step
crosses the link if it is not resident.**

## 2. Per-token arithmetic (the numbers that decide everything)

KV bytes per token = `n_kv_layers × n_head_kv × (n_embd_head_k + n_embd_head_v) × type_size`.

| model | KV layers | per token (f16) | at 262144 | indexer keys per token | at 262144 |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B (qwen35) | 16 of 65 | 64 KiB | 16 GiB | — | — |
| Qwen3.6-35B-A3B (qwen35moe) | 10 of 40 (trunk) | 20 KiB | 5 GiB | — | — |
| Qwen3.8-Flash-Next (qwen4exp) | 12 of 48 | 24 KiB | 6 GiB | 12 KiB (`4 × 128`, f16) | 3 GiB |
| MTP draft (27B, 1 layer) | 1 | 4 KiB | 1 GiB | — | — |

`gguf-dump` equivalent for Flash-Next: `block_count 48`, `head_count_kv 2`, `key_length/value_length 256`,
`full_attention_interval 4`, `indexer.head_count 4`, `indexer.key_length 128`, **`indexer.top_k 2048`**,
`compress_ratios [... 4 ...]`, `expert_count 512`, `expert_used_count 10`.

`-ctk/-ctv q8_0` halves every row (block 04); the ratio does not change with type, only the multiplier.

## 3. T4 — the bookkeeping that is device-resident

Rows marked **in scope** are the KV cache's own; the rest are the deferred later areas.

| item | size | device-resident today? | where |
|---|---|---|---|
| **packed kq mask** | `n_kv × n_ubatch × 2 B` — 960 MiB at `-c 245760 -ub 2048`, ×2 (device + pinned host) | **only on the packed path**; r160 made it lazy and r172 extended the derived form to multi-sequence batches | `len` in `llama-graph.cpp:48`; reserve gate `llama-context.cpp:1294` |
| derived mask inputs | `cell_pos (n_kv × n_stream) I32` = 983 KiB at 245760; `tok_lo`/`tok_hi` (ub × n_stream) | yes, tiny | `llama-graph.cpp:60-62` |
| qwen4exp indexer *box* vectors ("derived cache") | `F32 [idx_dim × ceil(kv_size/ratio) × n_stream]` per indexer layer | yes | `llama-memory-hybrid-idx.h:178-202` |
| F1 layer-input device store | `n_store × n_embd × n_ubatch × 4 B` (5 rows on a 27B at `-ub 2048` = 188.8 MiB before r163's position fix) | yes, when a DFlash drafter arms it | `llama-context.cpp:3725`, `llama-graph.cpp:1573` |
| expert cache tables | page/remap/staging/heat + per-table arena bookkeeping | yes, `O(#experts)`, MiB-scale | `moe-expert-cache.cu` |
| recurrent cell metadata (`v_cells`) | `n_ctx × ~16 B` per layer | **host only** | `llama-kv-cells.h` |
| attention cell metadata (`v_cells`) | `n_ctx × (pos, delta, seq bitmap)` ≈ `n_ctx × 24 B` on a 1-stream cache | **host only** | `llama-kv-cells.h` |

So the strict-metadata device footprint on a default run is **~1 MiB + MiB-scale arena tables**; the 960 MiB
term only exists on the packed path (a multimodal/M-RoPE context, or `LLAMA_KQ_MASK_PACKED_RESERVE=1`).
**This is why the campaign's priority is T2, not T4** — but T4 is where a *streaming* design would add cost
(a residency map is ≈ `n_cells/page × 5 × 4 B` = 1.2 MiB at 245760 with a 4-cell page), and that must be
budgeted up front rather than discovered.

## 4. T3 — the state that cannot be streamed, only recomputed

`llama_memory_recurrent` on the 27B is **598.5 MiB** (`S` 576 MiB: 64 layers × 24 576 vals × 4 B), independent
of context.  It is a *recurrence*, so it cannot be windowed: evicting it loses the prefix.  Anything that
reduces it is a layout/precision question (bf16 `S` = 288 MiB), not a residency one — out of scope here, and
worth its own note only if the bytes matter after T2.

## 5. Reproducing

```bash
cd ~/llama.cpp     # r174: rdna-boosts @ 5bdec04ab
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib

# the link constant (28.8 GB/s here: the card is x16, its switch uplink is x8)
/opt/rocm-10.1.0-gfx120X/bin/hipcc -O3 --offload-arch=gfx1201 bw-probe.cc -o /tmp/bw-probe
HIP_VISIBLE_DEVICES=0 /tmp/bw-probe 512 5

# A/B/C: the buffer and slab accounting
GGML_CUDA_SLAB_STATS=1 ./build-rocm/bin/llama-cli -v \
  -m /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 -ncmoe 24 -c 65536 -t 8 -p hi -n 4 --seed 42 --temp 0 \
  --no-display-prompt --single-turn 2>&1 | grep -E "SLAB_INIT|slab .* work|KV buffer|size =|arena .* of .* host experts"

# D: the all-host-KV bound
for k in 0 1; do ./build-rocm/bin/llama-bench -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -n 128 -d 4096 -r 3 -t 8 -nkvo $k; done
```

`SLAB_STATS` is the layout instrument (it prints the boundary, the work/arena split and the free-VRAM
low-water mark); `llama_kv_cache::memory_breakdown()` (`llama-kv-cache.cpp:1021`) is the per-buft KV
accounting the API exposes; `common/fit.cpp:287-360` is where `--fit` already models "weights + KV are one
per-device pool", i.e. where a residency term would go.
