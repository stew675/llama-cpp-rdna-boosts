# QFN memory usage — where the VRAM goes on a 24 GiB card

> **QFN** = *Qwen Flash Next* = `Qwen3.8-Flash-Next` (`general.architecture = qwen4exp`).
> This page is the measured memory accounting for the reference one-card `runme`
> (`GSQ-RCO-IQ3_S`, experts on the host, MTP draft on, resident KV). Every figure below is
> tagged with the log line, the metadata key or the arithmetic it came from, and the
> [reproduction recipe](#reproduction-recipe) at the end regenerates all of it.
>
> **The short version:** on that config the *model* occupies 4.2 GiB of VRAM, not 24. The
> other ~12 GiB that looks like "overhead" in the startup log is the **MoE expert cache**
> — resident expert slots, deliberately grown into every byte the weights/KV/draft did not
> need. The PLE table (26.8 GiB) and its lookup index are **not** in VRAM at all.

Measured **2026-11-14** on `gfx1100` (RX 7900 XTX, 24 GiB), build `b12386 (b1f559df7)`,
ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx1100`), 16-core 7950X / 63 423 MiB RAM.

---

## 1. What the model actually is (77.9 GiB of file)

The three-tier split is the whole story; everything else on this page follows from it.

| tier | size | where it lives | why |
|---|---:|---|---|
| 48 layers × 3 expert tensors (512 experts each) | **46.84 GiB** | host, pinned per-expert pool (`-ncmoe 48`, `-lme auto`) | `-ncmoe` offloads them; 512 experts × 640 ff is far more than a 24 GiB card can hold |
| `per_layer_token_embd` (the PLE table) | **26.82 GiB** | host, memory-mapped, read lazily per ubatch | `TENSOR_READ_LAZY`; it is all of shard 2 of the GGUF |
| everything dense (attn, SSM, shared-expert FFN, hyper-connections, `output`) | **4.20 GiB** | **VRAM** | `--n-gpu-layers all` |

Derived by parsing the GGUF tensor table (dims × type block size; no inference needed):

```text
/home/stew675/llama.cpp/build-rocm/bin/llama-server ...  →  "ROCm0 model buffer size = 3983.00 MiB"
4 305.1 MiB (all dense tensors)  −  322.07 MiB (token_embd, see below)  =  3 983.0 MiB   ✔ exact
```

The model card is `qwen4exp`: `block_count 48`, `expert_count 512`, `expert_used_count 10`,
`embedding_length 2560`, `attention.head_count 24`, `head_count_kv 2`, `key_length 256`,
`value_length 256`, `ssm.*` present, `attention.compress_ratios = [0,0,0,4, 0,0,0,4, …]`
(i.e. 36 linear-attention/SSM layers + **12 full-attention layers**, at 3, 7, 11 … 47).

Two details that surprise people looking at the log:

- **`token_embd.weight` (322.07 MiB) stays on the host** — it shows up as
  `CPU_Mapped model buffer size = 322.07 MiB`, not in `ROCm0`. It is a graph input, so the
  embedding lookup costs one H2D of the gathered rows, not 322 MiB of VRAM.
- **The MTP draft is a second model** (`mtp-…-Q4_K_M.gguf`, 1 808 MiB of weights) with its
  own context (1 223 MiB) — it lands *after* the main slab is created, which is why the
  slab's reserve carries a derived "draft reserve".

---

## 2. The server command used

The command is the `runme` in the model directory, unmodified except for `-lv 4` (added only
so the `LLAMA_LOG_INFO` accounting lines are visible — text/throughput arms must not pass it).
The environment, also from `runme`:

```bash
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S
HIP_VISIBLE_DEVICES=0 ./runme -lv 4
```

`runme` (verbatim, header comment elided) expands to:

```bash
SERVER=${SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib        # gfx1201 tree on the R9700 boxes
export HIP_VISIBLE_DEVICES=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1280

exec "$SERVER" \
  --model Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --spec-draft-model mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --alias Qwen3.8-Flash-Next-GSQ-IQ3_S \
  --host 0.0.0.0 --port 8033 \
  --fit off \
  --n-gpu-layers all \
  --n-cpu-moe 48 \
  --ctx-size 80000 \
  --parallel 1 \
  --flash-attn on \
  --limit-host-experts auto \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --override-tensor 'per_layer_token_embd.*=CPU' \
  --load-mode auto \
  --lazy-mode auto \
  --threads 7 \
  --cache-ram 4096 \
  --batch-size 2048 --ubatch-size 2048 \
  --temp 0.0 --top-k 20 --min-p 0.01 \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  -lv 4
```

Everything relevant starts at `-lv 4`; `--parallel 1` matters too, because a multi-slot
server multiplies the KV and the compute buffers (not the weights or the arena multiplier).

---

## 3. The VRAM accounting (24 560 MiB card)

| what | MiB | evidence |
|---|---:|---|
| **slab** (the movable-boundary VMM reserve) | **14 336** | `SLAB_INIT dev=0 … mapped=14.00 GiB` |
|   ├─ its *work* region (compute buffers) | 1 600 | `boundary=1.56 GiB`; `compute buffer 1539.07 MiB from the slab work region (boundary 1600.00 MiB)` |
|   └─ its *arena* (MoE expert cache) | 12 736 | `work 1.56 GiB + arena 12.44 GiB`; cache slots `arena 12365.3 MiB` |
| model buffer, dense tensors | 3 983 | `load_tensors: ROCm0 model buffer size = 3983.00 MiB` |
| MTP draft weights | 1 808 | `slab draft reserve derived from 'mtp-…': … the draft's 1808 MiB of weights` |
| MTP draft context (kv 263 + compute 768 + slack 192) | 1 223 | same line — the slab's **reserve estimate**; the measured context is 83 + 10 + 512 + 26 ≈ 610 MiB (`llama_kv_cache` 83.14 / indexer 10.39 / `compute buffer` 512.00 / `ROCm_Host` 26.32) |
| main KV, q8_0, 80 128 cells, 12 full-attn layers | 998 | `llama_kv_cache: ROCm0 KV buffer size = 997.69 MiB` |
| indexer KV (QSA lightning indexer, **K only**) | 125 | `creating indexer KV cache, size = 80128 cells` → `124.71 MiB … K (q8_0): 124.71, V: 0.00` |
| SSM recurrent state (48 layers: R 16.88 + S 432.00 + P 1.41) | 450 | `llama_memory_recurrent: ROCm0 RS buffer size = 450.28 MiB` |
| rope rotation buffers (256-d + 128-d indexer) | 4 | `init_rotation: rotation buffer size = 3.19 / 0.75 MiB` |
| `ROCm_Host` compute buffer (**the CPU backend's** arena, pinned host RAM — not in the slab) | 101 | `sched_reserve: ROCm_Host compute buffer size = 101.42 MiB` |
| driver / allocator / alignment remainder | ~391 | `mem_info_vram_used` − the rows above |
| **measured used** | **23 419** | `/sys/class/drm/card1/device/mem_info_vram_used` |
| free (the intentionally-kept cushion) | 1 141 | `GGML_CUDA_SLAB_HEADROOM_MIB=1280` is what the slab leaves outside; the rest is real free VRAM |

So the honest one-line summary is:

```text
24 GiB = 4.2 GiB weights + 3.0 GiB MTP draft (weights+context) + 1.6 GiB KV/indexer/SSM
       + 1.6 GiB compute (work region) + 12.4 GiB expert-cache arena + ~1.2 GiB cushion
```

> **The `ROCm0` / `ROCm_Host` buffers in this log are not one thing — four different objects share those two
> names.**  Only two of them are compute arenas, and the two compute arenas are **independent allocations in
> different memories** (not two halves of one buffer, not aliased, neither inside the other):
>
> | buffer | what it is | where |
> |---|---|---|
> | `ROCm0 compute buffer` | the **GPU backend's** arena (buffer type `ggml_backend_cuda_buffer_type`) | **inside the slab's work region** — the slab wraps the device compute alloc, so this is the one the 1 600 MiB boundary covers |
> | `ROCm_Host compute buffer` | the **CPU backend's** arena, whose default buffer type `llama-context.cpp:846-853` *swaps* for the first GPU's **host** buffer type ("for faster transfer of the intermediate state") → `cudaMallocHost`, pinned and device-readable; pageable fallback with a loud warning; 4 GiB max | outside the slab, **never** in it |
> | `ROCm_HostMap model buffer` (~47 GiB over ~50 lines) | the **mmap'd host-expert tables** (`-ncmoe`) | a *model* buffer, not compute |
> | `ROCm_Host output buffer` (0.95 MiB) | `llama_context`'s logits/output buffer | neither |
>
> The two compute arenas are sized separately by the same `sched_reserve` pass (1 539.07 vs 101.42 MiB at
> `-ub 2048`; 512.00 vs 26.32 for the draft at `-ub 512`), and what makes them one mechanism is only the
> pipeline: a node lives in exactly one of them and a tensor crossing the boundary is a copy — which is
> exactly why the CPU arena is **pinned** rather than pageable.  It tracks the *graph*, not the context:
> `ROCm_Host` reads 100.19 MiB at `-c 4096` against 101.42 at `-c 80000`, and only **0.3457 MiB** for a
> `llama-cli` graph of the same model with no draft and no backend sampler.

---

## 4. The "12 GB arena" is the expert cache, not overhead

The slab is a single VMM reservation that is split into a *work* region (compute buffers) and
an *arena* (the MoE expert cache's slots). Once the weights, KV and draft are resident, the
slab **grows into everything the model did not need**, down to `GGML_CUDA_SLAB_HEADROOM_MIB`:

```text
moe_cache_preflight: auto cache preflight: device 0 projected arena 15786.0 MiB
                     (free 20394.0 - reserve 1024.0 - aux 3584.0) >= floor 8633.2 MiB
SLAB_INIT dev=0 free=16.49 GiB reserve=2.44 GiB headroom=1.25 GiB aux=1.19 GiB
               size(VA)=15.19 GiB mapped=14.00 GiB boundary=1.56 GiB va_spare=1.19 GiB ring_req=3.52 GiB
ggml_cuda_slab_init_locked: device 0: slab 14.00 GiB: work 1.56 GiB + arena 12.44 GiB
alloc_all_locked: MoE expert cache (auto): arena 12365.3 MiB of 47962.5 MiB host experts (25.8% residency)
```

The arena is **spent**, not reserved: after the first turn the server reports its slot
inventory and hit rate, and the size settles slightly lower (the sizing pass re-arms uniformly):

```text
slot print_timing: MoE arena = 0.6997 (24610 hit / 35170 reaches this turn), arena 11907.3 MiB
```

That is a **70 % hit rate against 25.8 % residency** — the heat-ordered slots are doing their
job. Shrinking the arena would not free a resource; it would move those 24 610 hits back onto
the host pool. The levers, if you want to move VRAM between the two sides:

| lever | effect |
|---|---|
| `MOE_EXPERT_CACHE_MIB=N` | explicit arena size instead of "everything left" (`0` disables the cache) |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | the permanent outside cushion (the only true "overhead"; 1 280 here) |
| `--ctx-size` | KV at **12.75 KiB/token** on this model (2.6 GiB at 204 800) — every GiB of ctx is a GiB not given to experts |
| `--ubatch-size` | sizes the work region (1.6 GiB at `-ub 2048`) |
| `--spec-draft-model` off | returns 3.0 GiB (draft weights + its context) to the arena |
| `--cache-type-k/v` | q8_0 here; q4_0 halves the KV again |

**KV per token, derived and confirmed:** 12 full-attn layers × 2 KV heads × (256 K + 256 V)
× 34/32 B (q8_0) = 13 056 B = **12.75 KiB/token**, i.e. 997.7 MiB at 80 128 cells ✔ matches
`llama_kv_cache` to the byte. (The `runme` header's "12.6 KiB/token" is the same number
rounded down.) The indexer cache is separate and K-only: 12 × 128 dims, q8_0, 125 MiB.

---

## 5. The PLE: neither the table nor its index is in VRAM

This comes up every time the startup log is read, so it is worth being precise.

**The table (26.82 GiB) is host-side by architecture.** `qwen4exp.cpp` creates
`per_layer_token_embd` with `TENSOR_READ_LAZY`, the loader serves it from the file mapping and
records the range as lazy so the load-time `WILLNEED` prefetch does not fault the whole table
in (`load_tensors: enabling prefetch for 'per_layer_token_embd.weight'`). The rows are gathered
**on the host** into the graph input (`llm_graph_input_ple::set_input()`; `LLAMA_PLE_DIRECT`,
`LLAMA_PLE_THREADS`), and only the gathered rows go up. The `--override-tensor
'per_layer_token_embd.*=CPU'` in `runme` agrees with that architecture-level decision — it
costs no VRAM either way, and on a 24 GiB card a VRAM-resident PLE is not even arithmetically
possible (26.82 > 24). The device-table code path exists (`LLAMA_QSA_PLE_HOSTGATHER=0` →
graph `ggml_get_rows`) but this config never takes it.

**The lookup index is arithmetic, not a buffer.** Per token, `ple_ngram_size 3` and
`ple_heads_per_ngram 8` give `ple_n_heads = 16`; each head hashes the token n-gram and adds its
own base: `row = (t[p]*m[0] ^ … ^ t[p-n+1]*m[n-1]) % vocab[h] + offset[h]` (host code — ggml
has no int64 and no xor). So the index is a `std::vector<int32_t>` of
`n_heads × n_tokens`:

| | bytes / token | at `-ub 2048` |
|---|---:|---:|
| PLE row index (host `int32`) | 16 × 4 = **64 B** | 128 KiB |
| gathered rows sent to VRAM (`ple_embd`, F32 16×160) | 16 × 160 × 4 = **10 240 B** | 20 MiB |

The gathered rows land inside the compute buffer (they are `ple_embd` in the graph); the index
never leaves the host — and even the device-table path passes `rows` as an **input** leaf,
which the scheduler pins to the CPU backend. The head geometry
(`ple.head_offsets`, `ple.head_vocab_sizes`, 16 × u32 each) is `hparams` metadata, not a
tensor. **128 KiB against a 12 GiB arena is 0.001 % — there is nothing to reclaim here.**

---

## 6. Where QFN's memory actually hurts on this box: RAM, not VRAM

Same run, same log — the first prompt was `190.19 ms/token` (5.26 t/s prompt eval), which is
not a VRAM problem:

```text
pool_report_layout_once_locked: host pool layout (RESOLVED, first table): 512/512 slots =
    100.0 % coverage -> a MIRROR … Budget: a DERIVED budget of 47962.5 MiB
pool_ram_cap_locked: host pool RAM ceiling: 85% of MemAvailable = 51745.1 MiB
```

On a 61 GiB box, `-lme auto` resolves to a **mirror** (all 46.84 GiB pinned), and the 26.8 GiB
lazy PLE then has no page cache to live in, so the host gather faults from disk. This is
`INVARIANTS.md` r237 (*"`mlock` a multi-GiB lazy table and let the expert arena auto-size, and
the two fight for the model mapping"*) in its general form. Knobs, in the order worth trying:

1. `--lazy-mode mlockr` — populate the table's pages, then release the pin (r230's finding:
   "pin it" is the wrong default for a table that is only read).
2. a smaller `-lme N` (e.g. 34) so the mirror does not claim the whole box.
3. `--load-mode mmap --lazy-mode off` — the "fast mapped bulk + resident PLE" combination, at
   the cost of 27 GiB of anonymous, swappable RAM.

Throughput only: same-seed text is unaffected by any of these.

---

## 7. Reproduction recipe

```bash
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S

# 1) start it (add -lv 4 only for the accounting; a text/hash arm must not)
HIP_VISIBLE_DEVICES=0 ./runme -lv 4 > /tmp/qfn-vram.log 2>&1 &

# 2) one turn, so the arena is sized and the first fill happens
curl -s http://127.0.0.1:8033/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three colors."}],"max_tokens":16,"temperature":0}'

# 3) the accounting
grep -nE "slab|arena|KV buffer|RS buffer|compute buffer|model buffer|draft reserve|indexer" /tmp/qfn-vram.log

# 4) what the driver thinks
for c in /sys/class/drm/card*/device/mem_info_vram_used; do echo "$c $(($(cat $c)/1024/1024)) MiB"; done
pid=$(pgrep -f "llama-server --model .*Flash-Next" | head -1)
grep -E "^VmRSS|^RssAnon|^RssFile|^VmLck" /proc/$pid/status     # host side: PLE residency, pool
```

Model composition (§1) is re-derivable from the GGUF header alone — walk the tensor table and
multiply each tensor's element count by its type's bytes-per-block; `general.architecture`,
`block_count`, `expert_count`, `compress_ratios` and `ple.*` come from the same header.

---

## 8. Caveats — what this page does not claim

- **One run, one config, first turn.** The arena size is a function of *free VRAM at the moment
  it sizes*, so it moves with driver/context state; ±0.5 GiB between starts is normal. Treat
  12.4 GiB as "the arena takes everything left", not as a constant.
- **This is an accounting, not a benchmark.** The single prompt in the recipe exists to make the
  cache size and hit rate print; its t/s is the cold first-turn number and says nothing about
  steady-state throughput (see `benchmarks/` for those).
- **The host-expert numbers depend on `-lme`.** A 61 GiB box and a 93 GiB box resolve `auto`
  differently (mirror vs. bounded pool); the VRAM side is unchanged by that, the RAM side is.
- **`gfx1201` (32 GiB) shifts every row.** Same model, same flags: the arena absorbs the extra
  8 GiB (~47 % residency at `-ncmoe 48`), which is why the 32 GiB and 24 GiB records must not
  be mixed.

---

## See also

- [`INVARIANTS.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/INVARIANTS.md) —
  the slab/arena sizing rules, r230 (PLE residency), r237 (mirror vs. PLE page cache),
  r243 (host PLE gather)
- [`patches/README.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/README.md)
  — block 15 (streamed KV) and block 16 (`-lme` host-expert tiers, which is what `-ncmoe 48`
  plus `-lme auto` resolves into)
- [`ENVIRONMENT.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/ENVIRONMENT.md)
  — `GGML_CUDA_SLAB_HEADROOM_MIB`, `MOE_EXPERT_CACHE_MIB`, `LLAMA_PLE_*`, `MOE_HOST_POOL_RAM_PCT`
- [`wip/slab-draft-aux-reserve/`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/wip/slab-draft-aux-reserve/README.md)
  — the open campaign this page's MTP/arena numbers came out of, and the **2.4 GiB draft-reserve
  accounting error** it found (the slab's `aux` reserve is written by the *last* preflight, so a
  host-resident draft re-arms it with a weights-inclusive constant)
