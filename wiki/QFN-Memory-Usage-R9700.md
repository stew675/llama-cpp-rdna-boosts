# QFN memory usage on a 32 GiB R9700 — where the VRAM goes

> **QFN** = *Qwen Flash Next* = `Qwen3.8-Flash-Next` (`general.architecture = qwen4exp`).
> This is the **`gfx1201` / 32 GiB companion** to [QFN Memory Usage](QFN-Memory-Usage) (the
> 24 GiB `gfx1100` record). Same model file, same three-tier split, different command — and a
> very different arena, because the extra 8 GiB of card went to *context* and *compute*, not to
> experts.
>
> **The short version:** on this box the MoE expert cache is **12 646 MiB right after load** and
> **17 405 MiB after the first turn**. It grows because the wide prefill compute buffer drops
> back to its narrow size and the slab hands the difference to the arena. So "the arena says
> ~16 GiB" is the *settled* number, not a number that was ever free: 17.0 GiB of a 32 GiB card is
> expert slots. The other ~14.9 GiB is 5.7 GiB of weights (main + draft), 3.5 GiB of KV/SSM state,
> 0.6 GiB of MTP-head scratch, 1.75 GiB of compute, 2.4 GiB of arena span the slot banks have not
> claimed, ~0.4 GiB free, and ~0.5 GiB of hipBLASLt/staging that no broker accounts for. **Nothing
> is leaking, and there is no 16 GiB of "overhead".**

Measured **2026-11-15** on `gfx1201` (AMD Radeon AI PRO R9700, 32 624 MiB), delivery
**`v16-4f9dfa0e3-r292`** (tip `2b7a65309`), ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), one card
(`HIP_VISIBLE_DEVICES=0`), 9800X3D / 93 552 MiB RAM.

---

## 1. The three tiers are unchanged — only the placement moves

The 77.86 GiB `GSQ-RCO-IQ3_S` file splits exactly as the 24 GiB page records it; nothing about
the model differs between the two boxes:

| tier | size | where it lives on this box |
|---|---:|---|
| 48 layers × 3 expert tensors (512 experts) | **46.84 GiB** | host, pinned per-expert pool (`-ncmoe 48`, `-lme auto`) |
| `per_layer_token_embd` (PLE) | **26.82 GiB** | host, `--lazy-mode mlockr` → page cache, pin released after populate |
| everything dense (attn, SSM, shared-expert FFN, hyper-connections, `output`) | **3.89 GiB** | VRAM (`--n-gpu-layers all`) |

```text
load_tensors:   CPU_Mapped model buffer size =   322.07 MiB      ← token_embd (host, graph input)
load_tensors:        ROCm0 model buffer size =  3983.00 MiB      ← the dense weights in VRAM
load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB      ← the PLE table (lazy)
load_tensors: ROCm_HostMap model buffer size = … (144 lines)     ← 46.84 GiB host-expert tables
add: tensor per_layer_token_embd.weight (size = 27465 MiB) lazy read enabled, to be populated in the page cache (--lazy-mode mlockr)
```

`ROCm_HostMap` is a **host** buffer type that carries its device (so `MUL_MAT_ID` can name a
device for a host-resident weight) — it is RAM, and it is why that 46.84 GiB never shows up in
the VRAM ledger even though it prints 144 lines of it.

---

## 2. The command measured

`/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/runme`, unmodified except `-lv 4` (added only so
the `LLAMA_LOG_INFO` accounting lines print; a text/hash arm must not pass it). The differences
from the 24 GiB page's command are the whole story of §5:

| | 24 GiB XTX page | this page (R9700) |
|---|---|---|
| `--ctx-size` | 80 000 | **204 800** |
| `--batch-size / --ubatch-size` | 2048 | **16 384** |
| `--lazy-mode` | `auto` | **`mlockr`** |
| `--limit-host-experts` | `auto` (explicit) | unset → `auto` |
| attention pool | off | **off** (`--kv-attendance-pool` not passed; see §6) |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | 1 280 | 1 280 (default; the `runme` export is commented out) |

```bash
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S
HIP_VISIBLE_DEVICES=0 ./runme -lv 4
```

expands to `llama-server` with `--fit off --n-gpu-layers all --n-cpu-moe 48 --ctx-size 204800
--parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --load-mode auto
--lazy-mode mlockr --threads 7 --cache-ram 4096 --batch-size 16384 --ubatch-size 16384
--spec-draft-model mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-type draft-mtp
--spec-draft-n-max 3`, with `MOE_HOST_POOL_RAM_PCT=85` and `MOE_HOST_LIMIT_FRAC=95`.

---

## 3. The card-side accounting (32 624 MiB)

Two states matter, and the arena is 4.75 GiB bigger in the second one. **At load**, before any
request, the wide (16 384-token) prefill compute buffer is armed:

```text
SLAB_INIT dev=0 free=22.45 GiB reserve=3.09 GiB headroom=1.25 GiB aux=1.85 GiB
               size(VA)=21.19 GiB mapped=19.31 GiB boundary=6.62 GiB va_spare=1.88 GiB
ggml_cuda_slab_init_locked: device 0: slab 19.31 GiB: work 6.62 GiB + arena 12.69 GiB
llama_kv_cache:      ROCm0 KV buffer size =  2550.00 MiB     (204 800 cells, 12 full-attn layers)
llama_kv_cache:      ROCm0 KV buffer size =   318.76 MiB     (the indexer cache, K only)
llama_memory_recurrent:      ROCm0 RS buffer size =   450.28 MiB   (48 layers: R 16.88 + S 432.00 + P 1.41)
sched_reserve:      ROCm0 compute buffer size =  6736.28 MiB (the wide band, -ub 16384)
alloc_all_locked: MoE expert cache (auto): arena 12646.4 MiB of 47962.5 MiB host experts (26.4% residency)
h_nextn_dev_alloc:      ROCm0 h_nextn buffer size =   640.04 MiB
driver view: 31 915 MiB used / 32 624 MiB (mem_info_vram_used)
```

**After one turn**, the wide band is gone and the arena has taken its place:

```text
compute buffer 1792.00 MiB from the slab work region (boundary 1792.00 MiB) on device 0
settle_decide_locked: arena settle: the first drop offers 44.33% of the arena (sizing 19440 -> target 27936 slots);
    the prior covers 67.97% -> 79.88% of the routed mass, gain 11.92 pp vs the 1.00 pp threshold -- settle ENABLED
moe_cache_rearm: re-armed 144 stood-down expert-cache tables after a compute-buffer drop
    (45 layer(s) settled UP to the arena the drop returned) (arena now 17404.7 MiB)
slot print_timing: MoE arena = 0.5925 (16769 hit / 28300 reaches this turn), arena 17404.7 MiB
driver view: 32 225 MiB used / 32 624 MiB
```

The full ledger, in one table. Every row names the line it came from; the last row is measured by
difference and named rather than guessed.

| what | MiB | evidence |
|---|---:|---|
| **slab** — one VMM reservation, `size(VA)=21.19 GiB` | **21696** | `SLAB_STATS … mapped=21696.0 MiB` |
|   ├─ arena **span** (the experts' VA) | 19904 | same line `arena=19904.0 MiB` |
|   ├─ arena **slot banks** actually allocated | **17404** | `report_locked: … slots=26907, arena=17404.7 MiB` |
|   └─ *work* region = the compute buffer | 1792 | `compute buffer 1792.00 MiB from the slab work region (boundary 1792.00)` |
| model weights, dense tensors | 3983 | `load_tensors: ROCm0 model buffer size = 3983.00 MiB` |
| MTP draft weights (a second model) | 1808 | `load_tensors: ROCm0 model buffer size = 1808.36 MiB` |
| main KV, q8_0, 204 800 cells × 12 layers | 2550 | `llama_kv_cache: ROCm0 KV buffer size = 2550.00 MiB` |
| indexer KV (QSA lightning indexer, K only) | 319 | `… KV buffer size = 318.76 MiB`, `V (q8_0): 0.00` |
| SSM recurrent state (48 layers) | 450 | `llama_memory_recurrent: ROCm0 RS buffer size = 450.28 MiB` |
| rope rotation buffers (256-d + 128-d indexer) | 4 | `init_rotation: … 3.19 / 0.75 MiB` |
| draft KV + its indexer (1 layer) | 239 | `… 212.50 MiB` + `… 26.56 MiB` |
| `h_nextn` — the MTP head's device scratch | 640 | `h_nextn_dev_alloc: … h_nextn buffer size = 640.04 MiB`; also the slab's `outside_max=640.0` |
| HIP context, hipBLASLt Tensile code objects + workspace, FA-prefill staging (16 MiB), workspace-pool high-water | ~536 | by difference; the class `ENVIRONMENT.md` §1 calls un-routeable |
| free (live, post-turn) | 399 | `mem_info_vram_used` = 32 225 of 32 624 MiB; run low-water `free_min=826.0 MiB` |
| **card** | **32624** | `ggml_backend_dev_memory` / `mem_info_vram_total` |

> Do **not** read the `free=` field of `SLAB_STATS` as the live free VRAM: it is printed at
> context exit, i.e. *inside* the teardown that is already un-sizing the arena's slot banks
> (`free=1054.0` and then `free=3266.0` on two successive teardown lines in this run). The live
> figure is the driver's, and the ledger above closes against it to the MiB.

> **The two `arena` numbers are not the same quantity.** 19 904 MiB is the slab's arena *span*
> (VA handed to the cache); 17 404.7 MiB is what the slot banks occupy. The 2 499 MiB between
> them is the slab's 64 MiB chunk granularity across 144 tables × 3 expert tensors (432 banks,
> ~5.8 MiB of average rounding). It is not waste — it is what the *next* re-arm allocates from.

So the honest one-liner for this box is:

```text
32 GiB = 17.0 GiB expert-cache slots (+2.4 GiB of arena span slack) + 1.75 GiB compute (post-drop)
       + 5.7 GiB weights (main + draft) + 3.5 GiB KV/SSM/rotation (main + draft)
       + 0.6 GiB h_nextn + ~0.5 GiB HIP/BLAS + 0.4 GiB free
```

**Answering "is everything else really taking 16 GiB?"** No. Everything that is not a live expert
slot is ~14.9 GiB, and every row of it is an object you can point at in the log or in the slab's
VA map: 5.7 GiB of weights (main model + MTP draft), 3.5 GiB of KV/SSM/rotation, 0.6 GiB of
MTP-head scratch, 1.75 GiB of compute band, 2.4 GiB of arena span slack, 0.4 GiB free, 0.5 GiB of
un-routeable HIP/BLAS allocations. Add the 17 405 MiB of live slots and you are at the card's
32 624 MiB to the MiB. There is no hidden multi-GiB block anywhere in it.

---

## 4. The slab is one object that is re-split every turn

The single most confusing thing about this log is that the arena and the compute buffer are
**the same allocation**. The slab maps `free − reserve` at `SLAB_INIT` and divides it:

```text
at load:   mapped 19.31 GiB = work 6.62 GiB (compute) + arena 12.69 GiB
post-drop: mapped 21.19 GiB = work 1.79 GiB (compute) + arena 19.44 GiB   (+ 1.25 GiB narrow-2)
```

The sequence in the log, in order:

1. `SLAB_INIT … free=22.45 GiB reserve=3.09 GiB` — 3.09 GiB is deliberately left **outside** the
   slab: `headroom 1.25 GiB` (`GGML_CUDA_SLAB_HEADROOM_MIB`) + `aux 1.85 GiB` (the derived draft
   reserve: `draft CONTEXT 1890 MiB (kv 674 + compute 1024 + slack 192)`).
2. `narrow-2 (draft compute) pinned at [19.94, 21.19] GiB (1.25 GiB …); the arena now tops out at
   13.31 GiB` — the draft's compute buffer gets the top of the VA, and since r214 that region is
   **mapped for its whole extent**, i.e. 1.25 GiB of committed VRAM, not a reservation.
3. `compute buffer 6736.28 MiB from the slab work region` — the wide band, sized by `-ub 16384`.
4. First prefill → `compute buffer 1792.00 MiB (boundary 1792.00)` — the wide band is released.
5. `arena settle … settle ENABLED` + `re-armed 144 stood-down expert-cache tables after a
   compute-buffer drop (45 layer(s) settled UP) (arena now 17404.7 MiB)` — the arena takes the
   4.9 GiB the wide band gave back, in uniform per-table steps.
6. `SLAB_STATS context-exit`: `mapped=21696 work=1792 (narrow=2048 ring=8192.0 wide_aux=0.0
   narrow2=1280.0) arena=19904 … outside_failed=0.0 wide[n_pop=0 hole=0 inval=0 rebase=0
   aux_refused=0]` — the acceptance signals (`outside_failed=0`, `hole=0`, `inval=0`) are all clean.

Two VA figures in that line are **not** VRAM: `ring=8192.0` is the H2D staging ring's VA span
(`ring_req=7.96 GiB`, `ring_armed=0` in the same run — it grows on use, not at arm time), and
`narrow=2048` is the draft's narrow band bound.

**Practical consequence:** `-ub 16384` costs 4.9 GiB of arena *while a prefill is in flight* and
gives it back at the drop. It does not cost steady-state arena — it costs the settle (one
re-arm per turn, visible as the `re-armed … after a compute-buffer drop` line).

---

## 5. Why 32 GiB buys only ~5 GiB more arena than 24 GiB

| | 24 GiB XTX (`-c 80000`, `-ub 2048`) | 32 GiB R9700 (`-c 204800`, `-ub 16384`) | delta |
|---|---:|---:|---:|
| card | 24 560 | 32 624 | **+8 064** |
| main KV + indexer | 1 122 | 2 869 | +1 747 |
| SSM recurrent | 450 | 450 | 0 |
| main weights | 3 983 | 3 983 | 0 |
| draft weights + context | 1 808 + 610 | 1 808 + 239 + 1 280 (narrow-2) | +909 |
| h_nextn | — (not a row on that page) | 640 | +640 |
| compute, settled | 1 600 (wide band) | 1 792 | +192 |
| arena slots | 12 365 | 17 405 | **+5 040** |
| free | 1 141 | 399 | −742 |
| unassigned (the two records' own remainders and rounding) | | | +278 |
| **card** | **24 560** | **32 624** | **+8 064** |

So of the extra 8 GiB: **+5.0 GiB went to experts, +1.75 to the 2.5× longer context, +0.9 to the
draft's committed narrow-2 region, +0.6 to the MTP head, +0.2 to the wider compute band, −0.7 to
free**, and ~0.3 GiB is the two records' remainders. That is the whole budget — which is the point:
at 204 800 tokens of context the card is *not* under-used, it is over-committed to context.

Restated as per-token costs on this model (derived and confirmed, same arithmetic as the 24 GiB page):

- main KV: 12 full-attn layers × 2 KV heads × (256 K + 256 V) × 34/32 B (q8_0) = **12.75 KiB/token**
  → 2 550 MiB at 204 800 cells ✔ matches `llama_kv_cache` to the byte.
- indexer: 12 × 128 dims, K only, q8_0 = **1.59 KiB/token** → 318.75 MiB ✔.
- the residency miss path is measured, not estimated: the first turn's expert traffic is
  `fills=7532.7 MiB … over ~5 tokens = 1506.55 MiB/token` (§7). That is a **host** read per token
  at 26 % residency, which is why the arena is worth its VRAM even when the pool is a full mirror.

---

## 6. The attendance pool arm (measured, and why the `runme` header is stale)

`--kv-attendance-pool on` (block 14, **default off**) moves the authoritative attention KV to a
pinned host mirror and keeps only the indexer's working set on the card:

```text
ATTENDANCE POOL ENABLED for the attention KV: a 204.4 MiB device pool of 4104 pages x 4 cells per layer
    (sized by the system to one decode band of the indexer working set),
    with the authoritative copy in a pinned host mirror (up to 2550.00 MiB at full context)
llama_kv_cache:      ROCm0 KV buffer size =   207.14 MiB
llama_kv_cache: ROCm_KV_Pool_Mirror0 KV buffer size =  2550.00 MiB
arena 14988.3 MiB (startup, 31.2% residency)  ->  19872.0 MiB (after the first turn, 19.4 GiB)
driver view: 31 937 MiB at load / 32 241 MiB after the turn
```

| | resident KV | pool |
|---|---:|---:|
| attention KV in VRAM | 2 550 MiB | **207 MiB** |
| attention KV in pinned host RAM | 0 | 2 550 MiB |
| arena after the first turn | 17 404.7 MiB | **19 872.0 MiB** |
| first-turn arena hit rate | 59.25 % | 61.44 % |
| total VRAM used | 32 225 MiB | 32 241 MiB |

So the pool buys **+2.47 GiB of arena for 2.34 GiB of pinned host RAM** and leaves the card at the
same occupancy — it is a *fittability* feature, exactly as `patches/README.md` block 14 says, and it
is off by default because it costs decode on a config whose KV already fits.

> **`runme` drift, worth knowing before you copy it.** The file's header says it runs *with* the
> pool, and line 31 exports `LLAMA_KV_REGION_POOL=102400` — but that line is **commented out, and
> the variable itself was removed in r246** (promoted to the first-class option
> `--kv-attendance-pool`, whose env form is `LLAMA_ARG_KV_REGION_POOL` taking `on|off` — a numeric
> value is accepted and **ignored** with a WARN; the size is derived and is not a knob since r247).
> Uncommenting line 31 therefore does nothing at all. As shipped, the `runme` runs the **plain
> resident KV**; the "measured shape (r193)" in its header describes the pool arm, which you now
> get only by adding `--kv-attendance-pool on`.

---

## 7. Where the RAM goes on a 93 GiB box (and why it is not the 61 GiB story)

```text
plan_host_expert_placement: host experts: auto -> a pinned per-EXPERT pool of 47962.5 MiB of 47962.5 MiB
pool_report_layout_once_locked: host pool layout (RESOLVED, first table): 512/512 slots = 100.0 % coverage
    -> a MIRROR (every expert of the table is resident); the arena exchange is INERT for it
pool_ram_cap_locked: host pool RAM ceiling: 85% of MemAvailable = 69628.8 MiB (MOE_HOST_POOL_RAM_PCT)
mlock_ranges: released the pin on 27465 MiB of populated tables (--lazy-mode mlockr); they stay in the page cache
report_locked: host pool (L2): 144 tensors, 73728 slots, 47962.5 MiB pinned; h=1.0000 (20250/20250), evictions=0
memory breakdown: |   - Host | 75750 = 75750 + 0 + 0 |
```

The `Host` row is exact and worth reading as a formula:

```text
75 750 MiB = 47 962.5 (the expert mirror) + 27 465.95 (the PLE, in page cache) + 322.07 (token_embd)
VmRSS at idle = 75.9 GiB, VmLck = 0 (mlockr populated the PLE and then released its pin)
```

On this box `-lme auto` resolves to a **full mirror**, so every expert fill is a host hit
(`h=1.0000`, `evictions=0`, `takeover=… cold_reaches=0`) and the drive is read once at start-up —
the opposite of the 61 GiB box in the [24 GiB page](QFN-Memory-Usage) §6, where the same `auto`
leaves the PLE to fault from disk. There is nothing to tune for VRAM here; the RAM side is the
thing to watch, and `MOE_HOST_POOL_RAM_PCT` (85 %) is the ceiling that stops the mirror from
taking the machine.

First-turn expert traffic, for reference: `fills=7532.7 MiB cold-reads=0.0 MiB total=7532.7 MiB
over ~5 tokens = 1506.55 MiB/token` — that is the residency miss path, and it is why the arena
(`h=0.5925`, 16 769 hits of 28 300 reaches) pays for itself.

---

## 8. Levers — how to move VRAM between the two sides

| lever | effect on the R9700 (measured or derived) |
|---|---|
| `MOE_EXPERT_CACHE_MIB=N` | explicit arena instead of "everything the slab has left" (`0` disables the cache) |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | the true outside cushion; raising it costs arena one-for-one (1 280 → `free≈1.05 GiB`) |
| `--ctx-size` | 12.75 KiB/token + 1.59 KiB/token indexer ⇒ 2 550 + 319 MiB at 204 800; `-c 131 072` returns ~1.15 GiB |
| `--ubatch-size` | sizes the wide band (6 736 MiB at `-ub 16384`); it is *transient*, the arena takes it back at the drop |
| `--spec-draft-model` off | returns 1 808 (weights) + 1 280 (narrow-2) + 239 (KV) + 640 (h_nextn) ≈ **3.9 GiB** to the arena |
| `--kv-attendance-pool on` | −2 343 MiB of device KV, +2 550 MiB of pinned host RAM ⇒ **+2.47 GiB arena** (§6) |
| `--cache-type-k/v` | q8_0 here; q4_0 halves the KV again |
| `-ncmoe` | moves whole expert layers between the host pool and the arena; `48` is all of them |

`--fit` (off here, `--fit off`) is the other way to attack this: it plans against the same
breakdown and will trade context for arena — but on a host-expert-bound model it cannot create
experts out of nothing, which is why this `runme` pins the shape instead.

---

## 9. Reproduction recipe

```bash
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S

# 1) start it; -lv 4 is for the accounting only (a text/hash arm must not pass it)
HIP_VISIBLE_DEVICES=0 GGML_CUDA_SLAB_STATS=1 ./runme -lv 4 > /tmp/qfn-32g.log 2>&1 &
until grep -q "listening on http" /tmp/qfn-32g.log; do sleep 2; done

# 2) once at load (arena 12 646 MiB) and once after a turn (arena 17 405 MiB)
echo "at load: $(($(cat /sys/class/drm/card0/device/mem_info_vram_used)/1024/1024)) MiB"
curl -s http://127.0.0.1:8033/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three colors."}],"max_tokens":16,"temperature":0}' >/dev/null
echo "post-turn: $(($(cat /sys/class/drm/card0/device/mem_info_vram_used)/1024/1024)) MiB"

# 3) the accounting
grep -nE "SLAB_INIT|SLAB_STATS|arena|KV buffer|RS buffer|compute buffer|model buffer|h_nextn|memory breakdown" /tmp/qfn-32g.log

# 4) the pool arm and the host side
HIP_VISIBLE_DEVICES=0 ./runme -lv 4 --kv-attendance-pool on > /tmp/qfn-32g-pool.log 2>&1 &
pid=$(pgrep -f "llama-server --model .*Flash-Next" | head -1)
grep -E "^VmRSS|^RssAnon|^RssFile|^VmLck" /proc/$pid/status
```

`GGML_CUDA_SLAB_STATS=1` is a diagnostic only (it changes no allocation policy) and it is what
makes the `mapped/work/arena/narrow2/outside_max` split of §3 readable; without it you get only
the two `SLAB_INIT`/`re-arm` WARNs.

---

## 10. Caveats — what this page does not claim

- **One run, one config, one turn.** The arena is re-armed against the region the wide compute
  band returns at the prefill→decode drop, so ±0.5 GiB between starts is normal. 17.0 GiB means
  "the arena takes everything the slab has left after the narrow compute band and the committed
  narrow-2", not a constant.
- **`-ub 16384` is why the load-time and steady-state arena differ by 5 GiB.** A `-ub 2048` config
  on the same card would show a much smaller gap; a `-c 262144` one would sit on the VRAM cliff
  (`ENVIRONMENT.md` §1: the arena goes to 19.12 GiB with 1 536 MiB of headroom, or the first
  prefill dies with `hipBLASLt out of memory`).
- **The 24 GiB page's §8 estimate ("`gfx1201` … ~47 % residency at `-ncmoe 48`") does not apply to
  this command.** It was an extrapolation from the 24 GiB flags; measured here, the same model at
  `-c 204800 -ub 16384` reaches 26.4 % at sizing and 36.5 % (26 907 of 73 728 slots) once settled.
  The extra 8 GiB went to context and the committed draft region, not to experts.
- **This is an accounting, not a benchmark.** The single prompt exists to make the settle and the
  hit rate print; its t/s says nothing about steady-state throughput (see `benchmarks/`).
- **`gfx1201` and `gfx1100` records must not be mixed.** Same model, different card, different
  ROCm line (`/opt/rocm-10.1.0-gfx120X` vs `-gfx1100`) and different flags — the ledgers are not
  interchangeable rows.

---

## See also

- [QFN Memory Usage](QFN-Memory-Usage) — the 24 GiB `gfx1100` record this page is a companion to
- [`ENVIRONMENT.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/ENVIRONMENT.md) —
  `GGML_CUDA_SLAB_HEADROOM_MIB`, `MOE_EXPERT_CACHE_MIB`, §1.3 (the slab, its headroom and the draft's
  narrow-2 region), §17.1 (the attendance pool)
- [`DIAGNOSTIC.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/DIAGNOSTIC.md) — the
  kill-switches behind those knobs (same section numbers)
- [`patches/README.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/README.md)
  — block 14 (streamed KV / attendance pool) and block 15 (`-lme` host-expert tiers)
- [`INVARIANTS.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/INVARIANTS.md) —
  the slab/arena sizing rules and the r214 narrow-2 commitment
- [`wip/slab-draft-aux-reserve/`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/wip/slab-draft-aux-reserve/README.md)
  — the open campaign behind the `aux=1.85 GiB` reserve on this page
