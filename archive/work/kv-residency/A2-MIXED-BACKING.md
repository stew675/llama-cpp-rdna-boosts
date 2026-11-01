# A2 — the mixed-backing windowed KV: what it does and what it measures

**Not delivery work.** `TODO` **#118**.  Code: fork branch `wip-kv-residency` at `0e5b0d59d` (on top of A1's
`e49e1cef4`, which is kept on branch `wip-kv-residency-a1-two-base`).  Nothing here is in `patches/`,
`release.json` or any block.  Design of record: [`MIXED-BACKING.md`](MIXED-BACKING.md) (the probe and the
shape); this file is the build and the numbers.

## 1. What is implemented

| piece | where |
|---|---|
| the mapping **table** (`(off, bytes, kind)`), `map_span(off, bytes, host)`, `mapped_host()` | `ggml-cuda-vmm-region.{h,cu}`.  The driver's **minimum** granularity (4 KiB) is queried and a span is rounded outward to it; the bulk still maps in `unit` steps.  The selftest gained a HOST-backed span arm (device writes it, reads it back) and still passes: `VMM region ... PASS (... a HOST-backed span ...)` |
| `ggml_backend_cuda_kv_region_buffer_type_split(device, tail_bytes)`; `grow_cells` maps host `[0, tail)` + device `[tail, cells)` as the **cell watermark** advances; `grow_for` picks the kind by position (state restore writes arbitrary indices) | `ggml-cuda.cu` |
| the split point, the loud warning, the uniform-geometry check | `llama-kv-cache.cpp` (`LLAMA_KV_REGION=1` enables the provider, `LLAMA_KV_RESIDENT_CELLS=W` sets the window) |

Knobs (campaign only, all default-off): `LLAMA_KV_REGION=1`, `LLAMA_KV_RESIDENT_CELLS=W`,
`LLAMA_KV_REGION_DEBUG=1`, `GGML_CUDA_VMM_REGION_SELFTEST=1`.

Two lines say what happened:

```
W llama_kv_cache: WINDOWED KV ENABLED: 35091 of 262144 cells spilling to pinned system RAM
  (36.41 MiB per tensor), 227053 cells resident in VRAM. ... a token at depth costs about
  (cells beyond the window) x 34.0 KiB / 28.8 GB/s ...
I ggml_backend_cuda_kv_region_grow_cells: device 0: windowed KV -- 8704.00 MiB of KV mapped:
  8160.00 MiB in system RAM (pinned, non-swappable) + 544.00 MiB in VRAM, of a 8704.00 MiB reservation
```

**Declines rather than guesses:** if the K/V cell geometry is not uniform across the layers a cache holds
(and K vs V), the split is not offered and the cache is fully resident — today's fit-or-fail, with a warning.

## 2. Gates (1 GPU, gfx1201)

| gate | result |
|---|---|
| text identity, 4B Q8_0 `-c 8192` | plain `ef279d67a708` = region-only = window 2048 = window 4096 |
| text identity, 27B Q4_K_XL `-c 8192` q8_0 KV | plain `5ba44647188f` = window 2048 = window 4096 (and the same hash A1 produced) |
| text identity, 35B-A3B Q4_K_M `-ncmoe 24 -c 16384` q8_0 KV | plain `061add8deaba` = window 4096 |
| prefill logits (`gate-prefill-logits.sh --ab "LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024"`, ctx 2048) | **0.000000 mean KLD / 100.000 % same-top-p** |
| GPU faults (`check-gpu-faults.sh --since 21:00`, every log) | clean |
| the Goal A OOM config | **now runs** (below) |

## 3. The numbers

### 3.1 The OOM config runs (this is Goal A's headline)

27B **Q8_0** (27.1 GiB of weights — 1.29 GiB of which llama.cpp maps from the host on this card anyway),
`-c 262144`, `-ctk q8_0 -ctv q8_0`, 51 475-token prompt:

| arm | KV | result |
|---|---|---|
| plain, `-c 262144` | 8704 MiB in VRAM | **`cudaMalloc failed: out of memory`** (as recorded before the campaign) |
| window, `-c 262144` | **8160 MiB system RAM + 544 MiB VRAM** | **runs**: prefill **933–1086 t/s**, decode 10.8 t/s at depth |

### 3.2 What the spill costs at 51 475 tokens of depth

Same prompt, same model, `-c 262144` (so the same graph) with the window anchored at the used depth — the
prediction is `(used − resident) × 34 KiB/cell / 28.8 GB/s` on top of the resident token time:

| resident window | spilled cells | spill read per token | predicted | **measured** |
|---|---|---|---|---|
| all 51 475 (plain `-c 65536`, fully VRAM) | 0 | 0 | 53 ms | **18.7 t/s** |
| 32 768 | 18 707 | 636 MiB | 53 + 22 = 75 ms | **13.6 t/s** |
| 16 384 | 35 091 | 1.19 GiB | 53 + 41 = 94 ms | **10.8 t/s** |

The model holds to within ~3 %: the tail read achieves ~28 GB/s effective in the FA band's access pattern,
not just in a pure stream.  **The overflow cost is linear in the resident fraction, exactly as `O2-PLAN.md`
§3 said, and the resident fraction is the knob.**

### 3.3 The slab half of Goal A: the arena gets the VRAM

35B-A3B Q4_K_M, `-ncmoe 24`, `-c 262144`, q8_0 KV (2720 MiB), `GGML_CUDA_SLAB_STATS=1`:

| arm | `SLAB_INIT` free | slab mapped (= work + arena) | decode |
|---|---|---|---|
| plain KV (fully resident) | 19.80 GiB | 18.75 GiB | 91.6 t/s |
| window, `W=32768` of 262144 | **22.46 GiB** | **21.44 GiB** | 91.1 t/s |

**+2.69 GiB of arena** — the KV's VRAM the window frees (2720 MiB of KV − 64 MiB resident), and decode is
flat at a shallow depth.  This is the second half of the ask ("free VRAM for the decode arena, higher expert
hit rates"), and unlike O1 it is not a lazy growth fighting an arena that already owns the VRAM: the KV never
commits it.

## 3.4 The SLIDING window (built 2026-10-29, `9190fde16`)

The static window above anchored the resident set at the end of `-c`, so a run that has not filled the context
spills its whole used extent (§4).  The window now slides: **the resident set is the newest `W` cells of the
USED extent** and the host/VRAM boundary moves forward with the watermark.

* `ggml_cuda_vmm_region_ensure_span(off, bytes, host)` maps a piece whose kind is right, and **flips** one
  whose kind is wrong: stage the bytes out through a pinned scratch → `hipMemUnmap` → map the other kind →
  stage them back, with a device synchronise around the unmap.  The VA never changes, so captured graphs and
  every existing data path stay valid; the table stays one entry per mapping (the driver refuses a sub-range
  unmap, so a flip is a whole-mapping operation).
* The boundary is quantised **down to whole map units**, so a batch does nothing but map its delta (one cell
  normally) and a remap happens once per `unit/nb1` cells — ~1900 cells on the 27B, ~3900 on the 35B.  Cells
  between quantised boundaries stay resident a little longer, which is free.
* Monotone: a cell handed to system RAM is not taken back.

| arm (1 GPU) | before the slide | **with the slide** |
|---|---|---|
| 27B Q8_0 `-c 262144`, 51 475 tok, `W=49152` | 8.9 t/s | **18.6 t/s** (= the 18.7 t/s fully-resident reference) |
| 27B Q8_0 `-c 262144`, 51 475 tok, `W=16384` | 10.8 t/s | 10.3 t/s (same 1.19 GiB/token of spill; within this box's run-to-run spread) |
| 35B-A3B **Q8_0** `-ncmoe 24 -c 262144`, 5256 tok, `W=32768` | arena 15070 MiB, **58.2 t/s** (the whole used extent spilled) | arena 12393 → **15070 MiB**, residency 63.3 → **77.0 %**, prefill 602 → 615, decode **64.5 → 64.9 t/s with zero spill** |

Text is byte-identical to the fully resident arm on every one of those, **and byte-identical to the pre-slide
hash** (`ef279d67a708` 4B, `5ba44647188f` 27B q8_0) — which is the proof that a remap preserves every staged
chunk, since a lost chunk would have changed the generated text.

**Default `W = -c`** (no spill, today's cache); the opt-in (`LLAMA_KV_RESIDENT_CELLS`) is what spills.

## 3.5 The arena's value at -n 2000 (and the bigger the model, the more it is worth)

**Rule (maintainer, 2026-10-29): always `-n 2000`.**  A shorter run's decode is drowned out by the expert
cache's warm-up (the same 35B-A3B Q8_0 arm reads 54.6 t/s at `-n 128` and 66.3 at `-n 2000`), so every number
here is a 2000-token decode after a 5256-token prompt.

| model / config (1 GPU, `-c 262144`, q8_0 KV unless noted) | arena | host-expert residency | prefill | decode |
|---|---|---|---|---|
| 35B-A3B Q8_0 `-ncmoe 40` plain | 25 245 MiB | 77.3 % | 373 t/s | 66.3 t/s |
| 35B-A3B Q8_0 `-ncmoe 40` window `W=32768` | 27 922 (**+2.62**) | 85.5 % | 382 t/s | 67.4 (**+1.7 %**) |
| 35B-A3B Q8_0 `-ncmoe 40` **bf16 KV** plain | 22 822 MiB | 69.9 % | 375 t/s | 65.4 t/s |
| 35B-A3B Q8_0 `-ncmoe 40` **bf16 KV** window | 27 922 (**+5.10**) | 85.5 % | 373 t/s | 66.2 (**+1.2 %**) |
| 122B-A10B Q4_K_XL `-ncmoe 40` plain | 8 746 MiB | **15.6 %** | 220 t/s | 21.8 t/s |
| 122B-A10B Q4_K_XL `-ncmoe 40` window `W=16384` | 12 026 (**+3.18**) | **21.5 %** | 233 t/s | 24.0 (**+10.1 %**) |
| **122B-A10B Q4_K_XL `-ncmoe 48` (ALL 48 layers) plain** | 19 661 MiB | 29.3 % | 202.8 t/s | 22.9 t/s |
| **122B-A10B Q4_K_XL `-ncmoe 48` window `W=16384`** | 22 807 (**+3.15**) | **34.0 %** | **220.3 t/s** | **24.7 t/s (+7.9 %)** |

| **Flash-Next GSQ-IQ3_XXS** (qwen4exp) `-ncmoe 48` plain | 22 061 MiB | 53.9 % | 394.0 t/s | 40.8 t/s |
| **Flash-Next GSQ-IQ3_XXS** `-ncmoe 48` window `W=16384` | 25 818 (**+3.63**) | **63.1 %** | **445.6 (+13.1 %)** | 41.6 (**+2.0 %**) |

`-ncmoe` **must** be the model's full layer count when the point is the arena: at `-ncmoe 40` eight layers of
experts (218.6 MiB per expert across the host layers) still sit on the device, so the arena is only 8.7 GiB of
the 65.5 GiB host set (15.6 %); at `-ncmoe 48` it is 19.7 GiB (29.3 %).  That is the config that matters for
this model, and it is also its fastest: **24.7 t/s** with the window vs 22.9 plain.

**The window's benefit splits into prefill and decode, and they have different laws.**

* **Prefill is capacity-bound.**  A prefill routes *every* expert of every layer, so the arena size is
  directly the staged-bytes saved: 35B-A3B +1.8 %, 122B-A10B +5.9 % (`-ncmoe 40`) / **+8.6 %** (`-ncmoe 48`),
  **Flash-Next +13.1 %** (512 experts × 48 layers, the hungriest set here).  This is the most consistent win
  in the whole table.
* **Decode is hotness-bound.**  10 of 512 experts are routed per token, and the hot set is lopsided, so at
  ~54-85 % residency the extra GiB mostly holds *cold* experts the run never re-routes: +1.2-2.0 % on the 35B
  and Flash-Next.  It only converts to a large decode gain where the arena is genuinely starved — the 122B at
  **15.6 %** residency gains **+10.1 %** (and +7.9 % at `-ncmoe 48`).

Flash-Next also widens the picture on the caches themselves: it has **two** `llama_kv_cache` instances (the
attention KV, 3264 MiB, and the QSA **indexer** cache, 408 MiB) and the window applies to both — the log shows
one `WINDOWED KV ENABLED` per cache (0.5 KiB/cell and 0.1 KiB/cell).  The indexer cache is the O(n_ctx) term
the campaign's Goal B/O4 notes call the second-biggest, so it now participates in the window instead of being a
fixed VRAM cost.  122B numbers:
48 layers, 256 experts, 8 used, `n_embd_k_gqa` 512, 55 974 MiB of host experts, 218.6 MiB per expert across
its 40 host layers.

**The fittability half at `-n 2000`** (27B Q8_0, 51 475-token prompt, 53 475 used by the end):

| arm | decode |
|---|---|
| `-c 65536`, fully resident (the reference) | 18.4 t/s |
| `-c 262144`, `W=55296` (nothing spills) | 18.6 t/s |
| `-c 262144`, `W=49152` (131 MiB/token of spill) | **18.6 t/s** |

So the 262 144-cell context runs at the small one's speed, and the spill hides behind the 27B Q8_0's 1.29 GiB
of CPU-mapped weights (which is what actually bounds this arm: ~1.29 GiB/token over the link).

### 3.6 The deep arm (the caches actually spill) — and the window's break-even

Flash-Next, `-c 65536` (so `-c` is ~1.4× the 46k-token prompt), `-ncmoe 48`, q8_0 KV, `-n 2000`; at depth
~15k cells of BOTH caches are host-backed (attention KV 0.5 KiB/cell + QSA indexer 0.1 KiB/cell):

| arm | `SLAB_INIT` free | arena | residency | prefill | decode |
|---|---|---|---|---|---|
| plain (both caches fully resident) | 27.06 GiB | 24 539 MiB | 60.0 % | 465.7 t/s | **45.2 t/s** |
| window `W=32768` (both spill at depth) | 27.46 GiB | 24 939 MiB | 60.9 % | 477.2 t/s | **38.6 t/s (−14.6 %)** |

**The window is a bet that `-c` is much larger than the depth you run at**, and this arm is the other side of
it: at `-c 65536` the whole KV is 918 MiB, so the window frees only **0.39 GiB** of arena and costs **14.6 %**
of decode (213 MiB/token of spill at depth).  Compare §3.5 at `-c 262144`, where the same model frees 3.63 GiB
and spills nothing at a 5256-token depth (+13.1 % prefill, +2.0 % decode).  Break-even is therefore a **`--fit`
decision**: size `W` from the depth (and leave `W = -c`, i.e. no spill, when `-c` is close to it) — exactly the
`--fit` half already on the owed list.

### A bug this measurement found (and fixed): the resident window must be committed BEFORE SLAB_INIT

The deep arm first **aborted** at depth: `growth refused (region 816 MiB, wanted 500..503 MiB)`, with
`SLAB_STATS ... free=0.0 MiB`.  The resident window's VRAM was being mapped lazily, so `SLAB_INIT` mapped
`free VRAM − reserve` first and the arena owned all of it by the time the watermark advanced — the ordering
rule `O2-PLAN.md` §6.2 already states, which A2 was violating for the **device** half (the host half is
supposed to be lazy; the resident half is not).  `init_tensor` now commits `min(W, n_cells)·nb1` of VRAM per
tensor at buffer creation and keeps the spill lazy (`ba5c7205e`).

### A bug this measurement found (and fixed): the boundary must be quantised in BYTES

The slide first quantised its boundary in *cells* (`unit/nb1`, integer division), which is not map-unit
aligned: every unit straddled it, `ensure_span` fell back to 4 KiB pieces, and a **host** chunk costs ~130 µs
to create (`mixed-backing-probe.cc` T8) — one per tensor per token was ~4 ms/token.  The same arm read
**14.7 t/s before the fix and 18.6 after** (`f50c33be5`).  The boundary is now `(n_cells − W)·nb1` rounded
down to whole units, so both segments map in whole units (one chunk per ~3855 cells on a 27B).  `grow_for`
reads the same byte boundary, so a state restore cannot map into the wrong kind.

## 4. FINDING (fixed by §3.4) — a static window anchored at the end of `-c` is the wrong product shape

The window is cells `[kv_size − W, kv_size)`: the **last** W cells of the *context*, not of the *used
extent*.  So at 51 475 tokens in a `-c 262144` cache, the entire used extent (cells `[0, 51475)`) is still in
the tail, and the "resident" window covers cells nobody has written yet:

| `-c 262144` arm | W | what is actually resident at 51k depth | decode |
|---|---|---|---|
| `W=49152` | 49152 | cells `[212992, 262144)` — **none used** | 8.9 t/s |
| `W=227053` | 227053 | cells `[35091, 51475)` — the newest 16k used cells | 10.8 t/s |

That is not a bug, it is the static split being anchored on the wrong end for a run that has not filled the
context.  Three ways out, and the choice is the maintainer's:

1. **Slide the window with the watermark** (the natural design): keep the newest W *used* cells resident and
   move one chunk from VRAM to system RAM (or back) as the watermark advances.  The probe's T7 showed exactly
   this is possible — `hipMemUnmap` + re-map the same VA from the other kind, a few µs — so it is a
   bookkeeping change between batches, not a new kernel.  Cost: a 2 MiB copy per advance.
2. **Size W from the actual depth** at the first batch (`W = min(kv_size, depth + slack)`), which is a
   creation-time decision that can be revisited once (`--fit`'s job in A3).
3. **Default W = `-c`** (no spill) and let the user pick, which is today's semantics plus an explicit
   "I accept the spill" flag.

Until one of those lands, `W` must be chosen with the intended depth in mind — which is why every number in
§3.2 uses a window anchored at 51 475 cells.

## 5. Still owed before this can be called done

* ~~the sliding window~~ — **done, §3.4**.

* `scripts/gate-ckpt-reuse.sh` and `scripts/gate-state-pipe.sh` — **the two gates A1 never needed**: state
  save/restore now crosses the host/VRAM boundary.  The probe showed the copy engine handles it (T5) and the
  code maps the right kind by position, but the gates are what prove the whole path.
* the `MemAvailable` gate before pinning (4.35 GiB at `-c 131072`, 8.5 GiB at 262144, non-swappable).
* `--fit` reporting the resident bytes instead of the whole KV, so an over-budget config is *reported*
  rather than warned about and then run.
* sizing `W` from the depth (`--fit`), now that the slide makes W a *cache* rather than a cap.
* MTP verify widths on the windowed arm (A1 gated it; A2 changes the memory, not the kernel, so the risk is
  lower — but it is cheap).
* a 2–3 device arm: only one card is attached, so nothing here says anything about `-sm tensor`.

---

# §3.7 A3 (2026-10-30) — the re-baseline, `--fit` sizing, the RAM gate, and what the re-baseline broke open

Session scope (the maintainer's list): **(1)** `--fit` sizing + reporting of the window, **(2)** the two state
gates on a windowed arm, **(3)** a `MemAvailable` gate before pinning the spill, **(4)** MTP verify widths on a
windowed arm.  All four are done; the re-baseline also found **four** real bugs, and one **open** numeric
question that blocks the campaign's purity claim (§3.7.6).

## 3.7.1 The hardware swap happened: re-baseline first

`bw-probe.cc` (512 MiB, 5 reps) and `mixed-backing-probe.cc` re-run on the re-cabled box; the PCIe root port
now reads `current_link_width 16 @ 32 GT/s` (was x8):

```
bw-probe 1024 MiB:  H2D pinned 55.8   H2D pageable 54.0   D2H pinned 56.8
                    D2D 286.9        kernel read pinned 55.7   kernel read VRAM 635.1  GB/s
mixed-backing-probe (--mib 512): T3 device read 54.7 | T4 host-backed 54.8, both halves 105.5 |
                    T8 768 host chunks 55.0 ms, 53.4 GB/s, 0 failures | 0 failing checks
```

**The link doubled: 28.8 → 55.7 GB/s.**  Every cost model in this campaign becomes `bytes / 55.7 GB/s`, i.e. the
spill cost halves and the `--fit` break-even of §3.6 moves.  Do not compare a post-swap t/s number with a
pre-swap one (not even the plain arm).  The mixed-backing capability is unchanged (T1..T8 all pass).

The identity gates were re-run and re-baselined: 27B Q4_K_XL `-c 8192` q8_0 KV still hashes
`5ba44647188f`; the prefill-logit base was re-recorded (`PPL 5.0803`, sha256 `8fad054d95af…`) and the window
arm compared at **0.000000 / 100.000 %**; the plain `-c 262144` 27B Q8_0 arm still OOMs (`cudaMalloc failed:
out of memory`, 8704.00 MiB) — the feature-off baseline is intact.

## 3.7.2 `--fit` now sizes and reports the window (item 1)

`LLAMA_KV_RESIDENT_CELLS=auto` leaves the number to `--fit`, which is the only thing that knows both budgets
(the VRAM left after the arena/slab reserve, and the pinned-RAM ceiling).  `common_params_fit` now:

* measures the KV's own bytes (`0e5b0d59d`'s region buffer type is identifiable, so the KV's share of the
  context tier is known exactly, `kv_bytes / kv_size` = bytes per cell);
* sizes `W` from `free − margin − everything-but-the-KV`, clamps it against the pinned-RAM ceiling, and hands
  it down through `llama_kv_region_set_resident_cells()` (a process global the KV cache reads at creation);
* **returns early** when the windowed config fits, i.e. it keeps the user's `-c` instead of shrinking it;
* **reports** the split at WARN (visible by default):

```
W common_params_fit_impl: the KV cache is WINDOWED: 65658 of 262144 cells resident (2180.1 MiB of 8704.0 MiB),
  6523.9 MiB spilled to pinned system RAM
W common_params_fit_impl:   -> it costs 6523.9 MiB less VRAM (31386.0 MiB projected vs a 1024 MiB target,
  1024.0 MiB free); the spill reads 34.0 KiB per cell over the link (53 GB/s, calibrated) on every step, so a
  token at the full context reads 6523.9 MiB = about 129.3 ms at that rate, and nothing spills below depth 65658
```

The link constant and the per-cell bytes come from the backend (`ggml_backend_cuda_kv_region_link_gbps`, the
one-off H2D calibration — it reads 53 GB/s here, and it is why no campaign number is hard-coded any more).

Policies, deliberately: `auto` **only spills a context the user pinned** (with `-c` unset the fit keeps
choosing a fully resident context and prints how to ask for more); a **numeric** `W` is honoured and reported
(so `-c 262144 -W 49152` is now *reported as fitting* rather than refused); `auto` without `--fit` declines
the window with a warning.  `llama_kv_cache::memory_breakdown()` reports only the resident bytes against the
device buffer type and the spill against the CPU one, so the memory table and `--fit` both see where the KV
really lives.

Measured end-to-end: the Goal A config (`27B Q8_0 -c 262144 -ctk q8_0`, 35.6 GiB of weights+KV) now runs via
`--fit` alone (`W=123385`, 4096 MiB resident), and `llama-fit-params` reports the arithmetic in 0.4 s.

## 3.7.3 The `MemAvailable` gate on the spill (item 3)

`LLAMA_KV_SPILL_RAM_PCT` (default **85**, `0` disables) caps the pinned spill at that % of MemAvailable, the
`MOE_HOST_POOL_RAM_PCT` pattern, with **two** enforcements:

* **creation-time clamp** (`llama_kv_cache`): the worst case — every tail cell of the context — is compared
  with what is left of the ceiling, and `W` is raised until the spill fits.  A second cache (Flash-Next has
  two) sees what the first reserved (`..._spill_ram_reserve`).  If the clamp would have to make the whole KV
  resident, the window declines and says so (today's fit-or-fail).
* **run-time backstop** (`grow_cells`/`grow_for`): every host mapping asks the ceiling first, and a refusal is
  a loud error naming the knob instead of an OOM kill.

Measured (`4B Q8_0 -c 262144`, `LLAMA_KV_SPILL_RAM_PCT=1` → 795 MiB): `W` is raised 1024 → 236699 cells so the
worst-case spill is exactly the ceiling; the fit reports "(widened by the pinned-RAM ceiling)".

## 3.7.4 The state gates on a windowed arm (item 2) — and the abort they found

* `scripts/gate-ckpt-reuse.sh` with `LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=2048` (4B, hybrid, ctx 16384):
  **PASS** — keep 5 vs eager 8 checkpoints, 3 restored-and-kept, every request byte-identical, no GPU fault.
* `scripts/gate-state-pipe.sh` (`NGPU=1 SM=layer NCMOE=0`, window on): first run **aborted the server** on the
  batched save path —
  `GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(device) && "unsupported buffer type")` at
  `ggml-cuda.cu:5750`, from `llama_io_write_host::flush_async` → `ggml_backend_cuda_get_tensor_async`.  The
  batched state copy only accepted the *plain* CUDA buffer type, so a windowed KV (a region buffer type, which
  is a CUDA buffer by `ggml_backend_buft_is_cuda`, the identity the scheduler itself gates on) aborts the
  moment a prompt-cache save touches it.  All four sites relaxed to `ggml_backend_buft_is_cuda`; the gate then
  reads **PASS** (5 arms, byte-identical restored text, pipelined −5.5 %, threaded copy −7.1 %).
* **This is why the gates were on the list**: no other gate reaches that path, and the failure is a crash at
  the first prompt-cache save, not a wrong number.

## 3.7.5 MTP verify widths on a windowed arm (item 4)

4B Q8_0, `-c 4096`, q8_0 KV, spilling window (`W=2048`), 96 tokens, prose prompt:

| arm | text |
|---|---|
| plain decode | 96 chars `6541eadb9041` |
| plain `draft-mtp` (`n_max 7`) | 96 chars `6541eadb9041` |
| windowed decode | 96 chars `6541eadb9041` |
| windowed `draft-mtp` | 96 chars `6541eadb9041` |

Byte-identical, i.e. the verify widths do not interact with the window (as expected: A2 changes memory, not
kernels).  **Caveat added 2026-10-30 (A4):** this arm did not actually spill (see §3.7.6's closure note), so it
proves the verify widths are unaffected by the *window*, not that they survive a *spill*.  Re-run it on a
spilling 27B arm before quoting it for the latter.  The 27B is not a usable arm for this until §3.7.6 is resolved (its spilled text already differs).

## 3.7.6 CLOSED (A4) — a spilled KV changes the numbers on the 27B

> **CLOSED 2026-10-30 (A4) — see §3.7.8 and §3.7.9.**  The data path this section blamed was correct: the
> flip's stage-out equals an **independent copy-engine read** of the same mapping byte for byte, and its
> restore equals the stage byte for byte, both exhaustively over 32/32 flips.  The fault is one level down, in
> the **VMM primitive**: `cuMemUnmap` of a device mapping, followed by `cuMemCreate`(HOST)/`cuMemMap` at that
> VA, zeroes a **different, live** host-location mapping in the same reservation.  Fixed with a hold in
> `vmm_map`.  **Three readings in this section are wrong and must not be re-derived:**
>
> 1. **"A 4B arm with the same spill shape is byte-identical" — it never spilled.**  A 4B at `-c 4096` with
>    `W` 1024/2048 needs `n_cells - W >= unit/nb1 = 3855` cells before `lead_bytes` leaves zero, so `FLIP=0`
>    (`LLAMA_KV_REGION_DEBUG=1` prints it).  Given `-c 16384` it corrupts exactly like the 27B
>    (26 chars `158c1671e511` -> 21 `9bef5c01369c`), and with the fix it is identical again.
> 2. **"It is not the quantized KV type" — right conclusion, wrong evidence.**  The 27B arm diverges with
>    **f16** KV too (303 -> 289 chars), so the quant path was never involved.
> 3. **`nb1` 1088 vs 2048 is a red herring.**  The victims are the mappings whose VA offset is 2 MiB-aligned
>    (34/68/102... MiB) in *both* models; the geometry does not matter, the map sequence does.
>
> The step-2 hypothesis ("a lost kernel write") is also excluded: the spilled cells are written *before* the
> flip and never rewritten, the flip verifies byte-exact, and a 0xAB canary placed after the flip is
> overwritten with zeros — so the loss is a real write that arrives after the flip, and it is not the KV
> writer (which only ever writes cells above the watermark).

Found while re-baselining, and **not** resolved:

* 27B Q4_K_XL, `-c 8192 -ctk q8_0 -ctv q8_0`, 5256-token prose, 64 tokens: plain `305 chars
  sha=e2cea6fa6670` vs windowed (`W=1024`, a **one-unit = 4 MiB** spill per tensor) `325 chars
  sha=fb65430049e9`.  `W=1024` and `W=1280` give the same boundary and the same text (so it is
  deterministic), and it reproduces with `--fit off`.
* Prefill-logit gate at `--ctx 8192 --chunks 4 --kv q8_0`, `W=1024`: **KLD 0.004973 / 98.761 %** (limit 0.005).
  `W=8192` (= `-c`, window enabled, nothing spills) reads **0.000000 / 100.000 %**, and the region alone
  (no window) reads **0.000000**, so it is the *spill*, not the buffer type.  The same point's self-comparison
  is `-0.000000 / 100 %`, so the point is stable and this is a real difference.
* A 4B arm with the *same shape* of spill (4 MiB per tensor, f16 **and** q8_0 KV) is byte-identical.  So it is
  not the quantized KV type and not "any spill".
* The flip itself is byte-exact: a debug verify (`LLAMA_KV_REGION_DEBUG=1`) reads every flipped mapping back
  and compares it with the stage — 32/32 flips ok on the 27B arm that produces the divergent text.

So the data path we control is correct and something else about having a spilled KV changes the result.

### The bisect is already narrowed (2026-10-30, last measurement of the session)

`LLAMA_KV_REGION_SPILL_SKIP=<tensor indices>` (added for this, DIAGNOSTIC ONLY) keeps the listed tensors of a
windowed buffer fully resident.  Same 27B arm, 64 tokens, `W=1024`:

| arm | text | note |
|---|---|---|
| plain (no window) | 305 chars `e2cea6fa6670` | the reference |
| window, **every** tensor skipped (no spill at all) | 305 chars `e2cea6fa6670` | **the control: the windowed apparatus is inert when nothing spills** |
| window, only tensor 0 spills (4 MiB, one tensor) | 310 chars `ccdb98b7b254` | **one tensor is enough** |
| window, tensors 0..15 spill (8 layers) | 317 chars `df3128cb54d5` | |
| window, all 32 tensors spill | 325 chars `fb65430049e9` | |

So it is **not** scale-dependent (a single spilled 4 MiB tensor moves the text) and not a stress/race effect
(two `W` values with the same boundary give the same text).  The spilled cells are the *oldest* cells of the
tensor, they were written before the flip, and the flip's byte round-trip verifies.

Two hypotheses worth trying first, in order:

(a) **The stage-OUT may be capturing the wrong bytes, and the verify is blind to it.**  The debug verify
    compares the *stage* with the *restore* (`stage == readback`), which holds even if the stage itself was
    filled with zeros/wrong data.  Take a snapshot of the mapping *before* the flip and compare it with the
    stage after the detach (a `debug`-only `GGML_CUDA_D2H` of the same range immediately before
    `vmm_remap`).  If the stage already differs, the bug is on the read side of the flip; if it matches, the
    spilled cells really do hold the right bytes and the fault is in the *consumer*.
(b) **A lost kernel write into host-backed memory.**  `cudaMemcpy` into a host-location mapping silently does
    nothing (§3.7.7) — it is entirely possible that a *store* to it from certain kernels (the q8_0
    `set_rows`/`cpy` path, or any kernel using non-temporal / L2-bypass stores) behaves the same, which would
    leave a re-written spilled cell holding the *previous* chunk's bytes: exactly the "deterministic, small
    KLD, only when spilled" signature.  The discriminating test is `--chunks 2` vs `--chunks 1` on the KLD
    gate (chunk 2+ REWRITES cells that are already spilled, whereas a single-chunk run only ever writes them
    before the flip), plus `LLAMA_STATE_TIMING`-style instrumentation is not needed — the gate's per-chunk
    logits already isolate it.  Note that the failing 27B *text* arm is single-chunk, so (b) can only be part
    of the story; check (a) first.

Also worth one run: the same bisect on the **4B** (which is byte-identical for a full spill).  If the 4B is
identical even when *one* tensor spills, the difference between the two models is the thing to chase
(`nb1` 1088 B vs 2048 B, 16 vs 8 KV layers, 32 vs 16 tensors) — and `nb1 = 1088` is **not** a multiple of 128,
which is worth checking against every vectorized access (16 B is fine: 1088 = 68 x 16).

## 3.7.7 The three bugs the re-baseline found in the slide (fixed)

1. **A coalesced mapping straddling the boundary silently undid the spill.**  `ensure_span` flipped the *whole*
   mapping whose range the boundary fell inside; `vmm_insert` coalesced same-kind neighbours, so the boundary
   landed inside an entry and the next VRAM step flipped it straight back.  Measured: a 4B arm with
   `W=2048 -c 16384` mapped **0 bytes** to host while the warning claimed the tail was spilled.  Fix: every
   mapping in a windowed region is now on the **map-unit grid** (`grow_cells` maps `used_up`, `init_tensor`
   commits `up(W·nb1, unit)`, `vmm_insert` no longer coalesces), and a flip whose range is not exactly one
   mapping is **refused loudly** instead of being silently widened.
2. **`cudaMemcpy` into/out of a host-location mapping does not land.**  A direct H2D restore into a freshly
   mapped host chunk completed "successfully" and left zeros (measured: `stage=0x53 read=0x00` at byte 0 of a
   4 MiB restore); a D2D whose *source* is one behaves the same.  Every staging leg now goes through a
   **device kernel copy** (`vmm_device_copy`), which is the access the probes measured working (53 GB/s).  The
   debug verify above is the regression test for it.
3. **A multi-piece rewrite of one mapping is not safe** on this driver: detaching a mapping and re-attaching
   it as head/mid/tail (the obvious way to honour a boundary inside a mapping) loses the middle piece's bytes.
   Hence the unit-grid invariant plus the loud refusal, rather than a cleverer flip.

The VMM selftest (`GGML_CUDA_VMM_REGION_SELFTEST=1`) now covers all of it: an aligned window flip moves
exactly one mapping and preserves every byte, and a straddling request is refused and moves nothing.

## 3.7.8 The fix (A4, 2026-10-30) — the hazard is in the primitive, and one hold closes it

**The finding.**  A `cuMemUnmap` of a device mapping followed by `cuMemCreate`(HOST)/`cuMemMap` at that VA
**zeroes a different, live host-location mapping elsewhere in the same reservation**.  Whole 4 MiB mappings,
deterministically, from the 6th flip of a 27B-shaped reservation onward.  Proof that it is a write and not a
stale read: a 0xAB canary placed in the mapping after its flip reads back as all zeros, and a
kernel-vs-copy-engine cross-check of the same VA agrees (both see zeros).  The victims are the mapping flipped
one step earlier, specifically those whose offset is 2 MiB-aligned.

**Isolation.**  `hostmap-hazard-probe.cc` (this directory) replays the real shape — 32 tensors × 8.5 MiB, a
4 MiB window mapped from VRAM at setup, flipped to pinned host one tensor at a time, re-verifying every
previously flipped segment.  Its arms:

| arm | result |
|---|---|
| default | **FAIL** (a segment reads all zeros) |
| `--grab N` — hold the physical the unmap just released, N >= the unit | **PASS** |
| `--precreate` (create the host handle *before* the unmap) | FAIL — ordering is not the trigger |
| `--no-flip` (remap as VRAM instead of host) | FAIL — the kind is not the trigger |
| `--no-stage` (drop the pinned staging round trip) | FAIL |
| `--vram-mib N` (N GiB already allocated) | FAIL — pressure is not the trigger |
| `--cpu-store` | **SIGSEGV** — the VMM VA is not CPU addressable, which is why the staging legs are kernels |

**The fix.**  `vmm_map` takes a hold — a `cudaMalloc` of the mapping's size — *before* `cuMemCreate` and
releases it on return, so the block the unmap released is out of the allocator's free list for the whole
create.  It is taken on **every** map site, not only the flip, because the hazard belongs to the create that
follows a release and `ensure_span`/`grow_for`/a state restore can follow one too.
`LLAMA_KV_REGION_QUARANTINE=0` is the kill-switch (A/B and bisection only — with it off the spill is **not**
transparent).  A refused hold **fails the mapping loudly** rather than proceeding corrupt: the region's
contract is to refuse and let the caller fall back to today's fit-or-fail.

**Cost.**  One transient allocation of one map unit (4 MiB) per mapping created, freed on the same call —
sequential, never nested, so it cannot accumulate; the allocator reuses the block, so it is a plateau of one
unit, not growth.  Frequency is one per tensor per flip event, i.e. about one per 3855 tokens at 27B geometry.
It must be the runtime allocator (which owns the free list); a VMM handle of our own does not consume the
released block.

**Gates (1 GPU, gfx1201, hardened default).**

| gate | result |
|---|---|
| 27B q8_0 `-c 8192`, `W=1024`, 64 tok | 305 `e2cea6fa6670` = plain; `QUARANTINE=0` -> 328 `d8a6e67b8a3d` (wrong) |
| 27B **f16** KV, same shape | identical (it diverged before) |
| 4B `-c 16384`, `W=1024` (**a real spill**) | 26 `158c1671e511` = plain (was 21 `9bef5c01369c`) |
| §6 identity trio, `W=2048`/`4096` spilling | 526 `5ba44647188f` x3 |
| prefill-logit `--ctx 8192 --chunks 4 --kv q8_0`, `W=1024` | **-0.000000 / 100.000 %** (was 0.004973 / 98.761 %) |
| the same arm with `QUARANTINE=0` | 0.004973 / 98.761 % — the recorded block, reproduced |
| the §3.7.6 bisect ladder (one tensor / all 32) | both 305 `e2cea6fa6670` |
| OOM config, feature off | still OOMs (8704.00 MiB) |
| VMM selftest / `check-gpu-faults.sh` | PASS / clean (14 logs) |

**A gate that is NOT sensitive, and must not be used as the regression test:** the §6 identity trio at
`W=2048`/`4096` with `-n 128` **passed while the spill was corrupted**.  The sensitive arms are the `W=1024`
`-n 64` text arm and the prefill-logit gate; prefer the latter, because a text hash is also blind to the
*magnitude* of a shift.

**And never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm**: its fingerprint re-reads and hashes every host
mapping at every growth, which is 10-40x on a spilling arm (a 27B read 397 and 936 ms/token where the product
reads 38.5 and 44.8).

## 3.7.9 What the spill costs at the new link (the descent is linear, not a cliff)

27B Q4_K_XL, depth 25 763, `-c 32768`, q8_0 KV, 1 GPU, 128 tokens, `ROCBLAS_USE_HIPBLASLT=0` (so the absolute
t/s is lower than a normal run; the deltas are the measurement).  Two reps on the ends, agreement <= 0.1 ms.

| arm | spill read (MiB/token) | decode | delta |
|---|---|---|---|
| plain, no window | 0 | 36.07 ms (27.72 t/s) | — |
| window `W=32768`, nothing spilled | 0 | 34.90 ms (28.65 t/s) | -1.2 ms (noise) |
| window `W=24576`, nothing spilled | 0 | 34.88 ms (28.67 t/s) | -1.2 ms (noise) |
| `W=16384` | 256 | 38.51 / 38.55 ms (25.96 t/s) | +3.63 ms |
| `W=8192` | 512 | 42.68 ms (23.43 t/s) | +7.78 ms |
| `W=4096` | 640 | 44.76 / 44.78 ms (22.34 t/s) | +9.88 ms |

**A straight line through the origin**: about **16.3 us per MiB of spill/token**, i.e. **~61 GB/s effective**
(the link is 55.7 GB/s; part of the tail read overlaps compute), intercept ~0, and the two zero-spill arms are
identical to each other — no fixed cost for enabling the window and no threshold anywhere.  The per-token cost
is proportional to `(used - W) * cell_bytes`, so the *rate* decays hyperbolically with depth
(`ms/token ~ base + spill/61 GB/s`), never in steps.  Prefill degrades the same way: 1186.7 t/s plain -> 1170.2
(window, no spill) -> 1116.0 (512 MiB spill) = -5.9 %.

Rule of thumb at this link: **128 MiB/token ~ +2 ms (-6 %), 256 ~ -9 %, 512 ~ -18 %, 640 ~ -22 %.**

## 3.7.10 What is left (updated 2026-10-30, A4)

**§3.7.6 is CLOSED** (§3.7.8).  The purity claim now holds for a *spilling* windowed arm on 1 GPU.

Owed before promotion, in order:

1. **The correctness corners the fix reopened.**  `gate-ckpt-reuse.sh`, `gate-state-pipe.sh` and the MTP verify
   widths must be re-run on arms that **provably spill** (`FLIP > 0`, `mapped_host > 0`) — the A3 passes were 4B
   arms that never reached a flip (§3.7.6's closure note), so they prove nothing about a spill.  The state path
   is also where a mid-run `map-new` happens, i.e. the other half of the hazard.
2. **A Flash-Next (qwen4exp) spilling arm** — the only model with **two** KV caches, the attention KV and the
   **QSA indexer**, plus §3.7.4's park coupling (`llama_memory_hybrid_idx` moves the indexer with the attention
   cells).  Add `test-backend-ops -o FLASH_ATTN_QSA` on that arm.
3. ~~The hazard as a self-test~~ — **DONE (A4 (ii))**: `GGML_CUDA_VMM_REGION_SELFTEST=1` replays 8 tensors ×
   8.5 MiB with a per-tensor 4 MiB window flipped one at a time through the real path, re-verifying every earlier
   window after each flip.  It is sensitive: with `LLAMA_KV_REGION_QUARANTINE=0` it **FAILS** on *"hazard: every
   already-flipped window still holds its bytes"* from the 5th flip.  Note it runs only when the selftest env var
   is set, so a release gate still has to ask for it (`gate-per-block-build.sh`/`validate-set.sh` do not).
4. **The §7.2 re-measurement** at the new link: the arena tables (`-n 2000`, `-ncmoe` = the model's full layer
   count, interleaved), the deep arm, and the moved `--fit` break-even.  §3.7.9 covers the cost side.
5. The promotion conversation (a block, `ENVIRONMENT.md`, the default-on call) — the maintainer's, and it needs
   1-2 first.  Goal B is still not started; multi-device is still not measurable here (one card).

---

# §3.7.11 A5 (2026-10-30, ninth session) — the corners on spilling arms, the Flash-Next arm, and the two bugs it found

Session scope: **§7.2 items 1 → 2 → 4** (the state/MTP gates on arms that provably spill; a Flash-Next qwen4exp
spilling arm; the re-measurement at the 55.7 GB/s link).  All three are done.  Item 2 found **two** real bugs:
one is fixed (`0013`, A5 (i)), the other is measured, isolated and **left for the maintainer** (§3.7.11.4).  Box:
**one R9700 (gfx1201, 32 624 MiB), `HIP_VISIBLE_DEVICES=0` on every arm**; every number here is 1 GPU.

## 3.7.11.1 Item 1 — the three gates, on arms that PROVABLY spill

Every arm below was first proved to spill (`LLAMA_KV_REGION_DEBUG=1` → `FLIP > 0`, and/or `KVREGION_DUMP=/dev/null`
→ `KVREGION_FREE … host=` > 0); the *timed* runs never carry `LLAMA_KV_REGION_DEBUG` (§5 rule 6).

| gate | arm (all q8_0 KV, W=1024) | spill proof | result |
|---|---|---|---|
| MTP verify widths | 27B Q4_K_XL, `-c 8192`, prose (5246 tok), `-n 64`, `n_max 7` | `FLIP=32`, 32/32 VERIFY ok (same shape) | **plain-decode / plain-mtp / win-decode / win-mtp = one hash**, 305 chars `e2cea6fa6670` |
| `gate-ckpt-reuse.sh` | 4B Q8_0 (hybrid), `CTX=16384`, prompt = prose×2 (**10 502 tok**) | `KVREGION_FREE … mapped=192.00 host=128.00` on both arms | **PASS** (keep 5 vs eager 8 checkpoints, 3 kept, every request byte-identical, 0 GPU faults) |
| `gate-state-pipe.sh` | 35B-A3B UD-Q4_K_M, `NGPU=1 SM=layer NCMOE=0`, `CTX=16384` | `mapped=160.00 host=80.00` in **all five** arms | **PASS** (5/5 cycles byte-identical to `sync`; pipe −7.0 %, copy0 −7.4 %); the mid-run `map-new` of a restore (`grow_for`) is clean |

Notes that matter for the next session:

* **The default `PROMPT` never spills on the 4B.**  A spill needs `used > W + unit/nb1`; the 4B has `nb1 = 1088`,
  i.e. `unit/nb1 = 3855` cells, so `W=1024` starts spilling at **4879** used cells and `W=2048` at **5903** — and
  the prose prompt is ~5250.  A3's `W=2048` pass was therefore a no-flip arm.  The doubled prompt is what makes the
  arm decisive; `-c` alone does not (A3 already used `-c 16384`).
* The MTP arm has **two** windowed caches (the target's KV and the MTP draft's own), one warning line each — the
  verify widths are therefore covered with the draft's KV windowed too.

## 3.7.11.2 Item 2 — the Flash-Next spilling arm, and bug 1 (FIXED, `0013`)

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`, `-ncmoe 48`, `-c 65536`, q8_0 KV, `W=1024`, 51 475-token prompt.  Two KV
caches take the window (attention 816 MiB, QSA indexer 102 MiB), two warning lines.

**It aborted at the first slide** — the first spilling arm this campaign has run on a model whose KV tensors are
not unit-sized:

```
llama-kv-cache.cpp:1903: grow_regions: the KV region cannot be grown to 32000 cells ...
ggml_cuda_vmm_region_ensure_span: device 0: refusing to flip [8.50, 12.50) MiB: the mapping
  [8.50, 12.50) MiB covers it only in part
```

**The cause is tensor PLACEMENT, not the flip.**  A slide step is a whole-mapping kind flip (§3, invariant 7), so
a flip request must coincide *exactly* with a mapping; the window quantises its boundary in whole map units, so a
flip is `[tensor_off + k·unit, tensor_off + (k+1)·unit)` and that only matches a mapping while every tensor
**starts** on the driver's map granularity.  The plain CUDA alloc size pads a quantized tensor's rows
(`MATRIX_ROW_PADDING`), which is **not** gran-aligned: the QSA indexer's q8_0 rows pad by **+512 B per tensor**, so
tensor *k* started `512·k` B off the grid (`TRACE` showed `off=8.500 MiB mod=512`, `17.001 mod=1024`, …),
`init_tensor`'s window commit floored to a 4 KiB-aligned start and left a 4 KiB tail mapping, and the second
tensor's first flip straddled it → refused → `grow_regions` aborts (by design: a batch above the mapped extent
would fault the GPU).  **The 27B and the 4B have unit-sized rows, which is why every earlier gate was green.**

### The fix (`0013`)

The **windowed** buffer type rounds each tensor's place up to `ggml_cuda_vmm_region_granularity()`.  At most
`gran − 1` unused VA bytes per tensor, **no mapped byte changes** (the resident window is committed per tensor at
its own rounded-up size either way), and it makes invariant 7 true *by construction*.  The plain (O1) buft is
untouched, and the reporting cost is 48 KiB on the Flash-Next indexer cache (102.00 → 102.05 MiB).

### After the fix

| measurement | value |
|---|---|
| slide steps | 156 flips, **156/156 `VERIFY … ok (exhaustive)`**, 0 refusals, 0 aborts (`LLAMA_KV_REGION_DEBUG=1` run of this exact config; the timed run below reads the same spill totals) |
| spill (at 51 475 depth) | attention **576 of 672 MiB** host, indexer **48 of 96 MiB** host |
| prefill / decode (`-v`, no DEBUG) | **683.9 t/s** / **26.48 t/s** (37.8 ms/token, of which ~11 ms is the 624 MiB/token of spill) |
| text, `ROCBLAS_USE_HIPBLASLT=0`, no `-v` | plain == windowed, **701 chars `6c0561298cf1`** |
| prefill-logit KLD (`--ctx 32768 --chunks 2 --kv q8_0`, base PPL 4.0810) | **−0.000000 / 99.994 %**, reproduced on the final tip after the fix commit — and the gate arm itself spills (`mapped=408.00 host=384.00` at `--chunks 1`) |
| `test-backend-ops -o FLASH_ATTN_QSA` | **26/26** |
| GPU faults | `check-gpu-faults.sh` clean |

Recovery record: the fix was found with two throw-away diagnostics (an exact-bytes print in the refusal, and the
raw offsets in the `LLAMA_KV_REGION_TRACE` line).  Both were reverted before the gates above; the trace line is
still worth adding permanently if this class recurs.

## 3.7.11.3 Item 2 — bug 2 (FIXED, `0014`): a park on a region-backed KV faults the GPU

`gate-park-idle-slots.sh` (the r136/r146 parking path, the model's own gating gate) **kills the server** on a
region-backed KV.  Same config throughout: Flash-Next, `-ncmoe 48`, 1 GPU, MTP draft, prose×2 / 51 475-token
request A.

| arm | result |
|---|---|
| window `W=1024`, `-c 65536`, deep prompt | request A completes, **request B faults**: `HSA_STATUS_ERROR_MEMORY_FAULT` in `__amd_rocclr_copyBuffer`, *"Page not present"*, GPU hang analysis, coredump |
| window `W=1024`, `-c 16384`, **shallow** prompt (5 404 used of 16 384 — **nothing spills**) | **identical fault** ⇒ the spill is not the trigger |
| `LLAMA_KV_REGION=1` with **no window** (the O1 grow-on-demand region) | **identical fault** ⇒ the window is not the trigger either |
| **window OFF** (plain CUDA KV), same config | **PASS** — and the gate reports *"every park moved the attention cache and the indexer cache together"* |

So the fault belongs to the **region provider**, not to the window: the region maps only the cell **watermark**
(`grow_cells`, and the window's own `init_tensor` commit), while a park relocates an idle slot's cells **near the
top of the cache**, i.e. above the mapped extent — unmapped VA, and the park's move is a copy-engine transfer,
which *faults* instead of returning an error.  (The same copy-engine-vs-host-mapping limitation as §3.7.7 #2,
with the opposite symptom.)

**The two candidates** (option 3, refusing the window, is out):

1. `llama_kv_cache::seq_park` grows the region to the destination cell range before moving cells.  Cheap and right
   on an **unwindowed** region (VRAM→VRAM, native D2D, no PCIe), but on a **windowed** cache the parked (cold)
   cells then take the resident window while the ACTIVE cells fall below the spill boundary — every decode step
   reads the running context over the link: a *per-token* cost (10 k active cells at Flash-Next geometry =
   109 MiB/token ≈ 2 ms/token ≈ 10 % of a 20 ms token) against save-and-clear's *one state copy* per transition.
   It also needs the copy leg to be kernel-based for any spilled range (invariant 8) or it silently does not land.
2. Decline the park on a region-backed cache and let the server use the fallback it **already has**
   (`park_idle_slots_for_launch`: *"cannot park the idle slot: saving it to the prompt cache and clearing it"*) —
   i.e. exactly the arm this gate compares against.  No new code in the copy/staging path.

**Chosen: (2), implemented in `0014`** (`seq_park` returns false, warn once; `LLAMA_KV_REGION`-only, so the
default path is untouched).  `can_park()` is deliberately **not** the place: a false there makes the server log
*"not supported … the idle slots stay where they are"*, which is the **#210** behaviour (idle slots resident), not
the fallback.

Measured (Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, `-c 65536`, `W=1024`, MTP draft, 51 475-token request A then three
short ones, 96 tokens each, 1 GPU, windowed arm — `KVREGION_FREE … host=1152` so the state path ran over a spill):

| arm | A (51 k tok) | B | C | D | tokens B/C/D |
|---|---|---|---|---|---|
| `park` = decline + save-and-clear | 85 521 ms | 1 992 | 2 129 | 2 074 | 155 / 159 / 162 |
| `parkoff` (no parking, #210) | 85 432 ms | 2 369 | 1 846 | 2 169 | 113 / 117 / 162 |
| `lazyoff` (the reference) | 85 458 ms | 1 934 | 2 150 | 2 096 | 155 / 159 / 162 |

⇒ **byte-identical to the reference on A–D** (the #210 placement dependence is still removed — `parkoff` differs),
the fallback's cost is **inside the arm-to-arm noise** (one state copy per idle transition), and there is no GPU
fault and no abort.  The three *"cannot park the idle slot: saving it to the prompt cache and clearing it"* lines
are the fallback firing once per idle transition.  **Control, no region: the park gate PASSES unchanged** ("every
park moved the attention cache and the indexer cache together"), so the default path is untouched.

**Owed (a gate gap, not a code gap):** `scripts/gate-park-idle-slots.sh` still HARD-fails this arm on *"parking
moved nothing"* — the gate does not recognise the server's save-and-clear fallback, which it should, because that
path exists for **any** slot whose cells are not one run (a prefix shared through `seq_cp`, for instance).  The
proposed change: when the park arm logged the fallback line, require `park == lazyoff` (it is) and report
*"parking declined → save-and-clear fallback"* instead of failing.

## 3.7.11.4 Item 4 — the re-measurement at the 55.7 GB/s link

`-n 2000`, `-ncmoe` = the model's full layer count, `-c 262144`, q8_0 KV, 51 433-token prompt, **arms interleaved
back-to-back with two reps each** (the first arm of a batch is a cold-page-cache artefact: 35B plain reads 652 t/s
cold and 1195 t/s warm, which is why the rule exists).

### A. A window at/above the run depth — no spill, and the arena still gains

The KV's VRAM is committed **lazily** (`init_tensor` commits W cells, `grow_cells` the rest), so a `W` above the
run's depth costs the arena nothing *and* spills nothing.  This is the Goal-A-2 shape:

| model (`-ncmoe 48`, `-c 262144`) | arm | arena | residency | prefill t/s | decode t/s |
|---|---|---|---|---|---|
| 35B-A3B **Q8_0** | plain | 25 245.0 MiB | 77.3 % | 1197.1 / 1192.2 | 67.81 / 67.76 |
|  | `W=53248` (nothing spills) | **27 412.5** | **84.0 %** | 1230.7 / 1242.4 | 66.85 / 67.28 (≈ −0.9 %) |
| 122B-A10B **Q4_K_XL** | plain | 19 661.1 MiB | 29.3 % | 365.1 / 366.0 | 28.74 / 28.75 |
|  | `W=49152` (nothing spills) | **22 282.6** | **33.2 %** | 382.7 / 382.6 (+4.7 %) | **29.84 / 29.84 (+3.8 %)** |

### B. A window far below the depth — the spill dominates

| model | arm | arena | residency | prefill t/s | decode t/s |
|---|---|---|---|---|---|
| 35B-A3B Q8_0 | plain | 25 245.0 | 77.3 % | 1195.7 | 67.53 |
|  | `W=16384` (**320 MiB/token** of spill) | 27 667.5 | 84.8 % | 1172.4 | **49.69 (−26 %)** |
| 122B-A10B Q4_K_XL | plain | 19 661.1 | 29.3 % | 366.6 | 28.81 |
|  | `W=16384` (**384 MiB/token**) | 22 544.8 | 33.6 % | 372.4 | **25.57 (−11 %)** |

Read together with §3.7.9: the arena's whole prize here is **+2.2 to +2.9 GiB** (residency +3.9 to +6.7 points),
worth **+3.8 %** decode at the 122B's 29 % residency and ≈0 at the 35B's 77 % — while a `W` 35 k cells below the
depth costs 11-26 %.  **The window's arena half only pays when `W` is at or above the depth the user actually
runs**, i.e. it is a *capacity* trade (the VRAM the KV does not need *yet*), not a hot/cold one.  This supersedes
the pre-swap §8 rows for this box; do not mix the two.

### C. `--fit` still refuses to spill for the arena's sake — a policy gap, not a bug

`LLAMA_KV_RESIDENT_CELLS=auto` sizes a window only when the KV does **not** fit at the pinned `-c`:

```
35B-A3B Q8_0 -ncmoe 48 -c 262144:  "auto: no window is needed -- the 262144-cell KV (2720.0 MiB) fits fully
                                   resident (21899.2 MiB of VRAM left over the 6899 MiB target)"
27B Q8_0        -c 262144:          "WINDOWED: 112190 of 262144 cells resident (3725.1 MiB of 8704.0 MiB),
                                   4978.9 MiB spilled ... about 95.2 ms at that rate, and nothing spills below
                                   depth 112190"
```

So the arena half of Goal A needs a **numeric** `W` (or a new fit policy: "spill to grow the arena, up to N MiB of
spill per token"), and the number to hand it is table A above.  The moved break-even at this link is the cost side
of §3.7.9 (16.3 µs/MiB pre-swap → **≈8-11 µs/MiB** here): 128 MiB/token ≈ +1.1 ms (−4 % of a 28 ms token),
256 ≈ −7 %, 512 ≈ −15 %, 640 ≈ −18 %.  `--fit` no longer hard-codes 28.8 GB/s (it calibrates from the backend and
reports 54-55 GB/s here), so its own arithmetic already reflects this.

## 3.7.11.5 What is left

1. ~~The corners on spilling arms~~ — **DONE** (§3.7.11.1).
2. ~~The Flash-Next spilling arm~~ — **DONE** (§3.7.11.2), including the QSA op test; **both bugs fixed**
   (bug 1 §3.7.11.2 `0013`, bug 2 §3.7.11.3 `0014`; the parking gate's own fallback blindness is the one owed
   follow-up).
3. ~~The re-measurement~~ — **DONE** (§3.7.11.4 A-C).
4. The promotion conversation (a block, `ENVIRONMENT.md`, the default-on call for the window) — the maintainer's.
   It now needs a decision on §3.7.11.3 (park) and, if the arena half is wanted, on the `auto`-policy gap
   (§3.7.11.4 C).
5. Goal B is still not started; multi-device is still not measurable here (one card).

## 3.7.12 r181: the multi-device `--fit` crash, and three `auto` wiring defects (found while gating r180's PR work)

The campaign's box is single-GPU, so nothing here had ever run `-sm tensor` with the block-15 KV plumbing.  It
**aborted at model load on every multi-device tensor split, with `--fit` at its default**: the KV-region glue
gets its backend entry points from `ggml_backend_dev_backend_reg(dev)` and a **Meta** (tensor-split) device has
no backend reg (`device->reg == NULL`), so `ggml_backend_reg_get_proc_address(NULL, …)` asserts
(`ggml-backend.cpp:1052`).  Three paths, all reachable from a plain `llama-cli`:

1. `common/fit.cpp`'s KV probes (`fit_buft_is_kv_region` / `fit_kv_spill_ram_budget` / `fit_kv_link_gbps`) —
   any run, any model, because `--fit` walks the buffer types of every model device.
2. `llama_kv_cache`'s region-buffer lookup (`llama_kv_region_buft_split(meta_dev, W)`) — a numeric `W`.
3. the spill budget / link lookups in the same constructor — a numeric `W` below `-c`.

Fixed in r181 by reading a reg-less device as "the backend does not answer": the window is **declined** on a
tensor split (one message from the fit, one from the cache), and `-sm layer` / one device keep their region
buffers.  The fit additionally resolves the Meta wrapper to a **simple child** so the identification is still
the backend's answer if a multi-device window ever exists.

**Still open (the campaign's, not the delivery's):** a window *across* a tensor split would need per-device
region bufts wrapped in a split buft plus a fit that sizes them together; nothing here implements or validates
that, and this box still has one card.

Three `auto` wiring defects were fixed in the same change, all reproducible on ONE device:

* `auto`'s decided "no window needed" (`0`) was reported by the cache as *"--fit did not run"*, because 0 was
  indistinguishable from "nobody decided".  Now a decided-0 flag exists (silent) and only a genuinely
  undecided `auto` warns.
* every run **without** the knob logged "the windowed KV is unavailable for this cache (no spilled cells)".
  It now fires only when a window was requested and declined for a real reason.
* the `LLAMA_KV_SPILL_RAM_PCT` clamp was applied without checking the VRAM budget, so the fit could report a
  window that the ceiling had already made unfittable (or, worse, accept one).  **That is a live hazard:** an
  intermediate form of the fix accepted a **235 GiB** pinned-spill plan for a 4B at `-c 8388608`; creating it
  OOM-killed the box (83 GB anon-rss, kernel OOM).  The clamp is now taken only while the widened window still
  meets the VRAM target; otherwise the fit reports that the ceiling cannot be served at this `-c` and drops the
  window (then fits by shrinking `-c`).

**The `auto` policy itself is unchanged, and still the maintainer's call** (§3.7.11.4 C): `auto` sizes the
window to the VRAM left after the weights, the margin **and the arena/slab reserve**, so on a `-ncmoe` config
the arena wins the VRAM and the KV window comes out small by design; and it only spills a `-c` the user pinned
(with `-c` unset the fit prefers a fully resident context).  A window that is "too small to be worth it" is
therefore usually the arena reservation showing up in `margins[0]`, not a sizing bug: the arithmetic is
`free − margin − everything-but-the-KV`, and it reproduces to the MiB on both boxes (27B Q8_0
`-c 262144 -ctk/-ctv q8_0`: 123265 of 262144 cells, 4092.8 MiB resident, 4611.2 MiB spilled).

---

## 3.7.14 DECISION (2026-10-31, maintainer) — P1 is the `auto` policy; P2 is rejected, P3 is not promoted

Of the three candidates §3.7.11.4 A-C left open, **P1 (fittability only) ships**: `auto` sizes a window only
when the KV does not fit at the pinned `-c`, and the expert arena keeps winning the VRAM through `margins[0]`.
**P2 (unconditional arena-first)** is rejected: a fit cannot know the depth the run will use, and table B
measures a `W` 35k cells below the depth at -11..-26 %, against the +3.8 % the arena half pays where residency
is starved (122B-A10B).  **P3 (an explicit per-token spill budget, so the user buys arena with a printed
cost)** is the right *shape* for that opt-in but is not promoted until it measures better than P1 as a
default -- it needs its own `fit.cpp` change and a measurement, so it stays a parked idea rather than a
recommendation.

What remains owed is documentation only: block 15's note should say P1 is the recommendation and that the
arena half of Goal A needs a numeric `W` (a numeric `W` at or above the run's depth is what tables A/B reward).

# §3.7.13 A6 (2026-10-30, tenth session) — the parking gate's fallback blindness, and the split-window design

Session scope: the campaign's remainder in HANDOVER §11.  Two items were actionable on this box; the rest are
decisions.  Box: **one R9700 (gfx1201, 32 624 MiB), `HIP_VISIBLE_DEVICES=0`**; every run 1 GPU.

## 1. The owed TEST gap is CLOSED (awaiting the maintainer's commit approval)

`scripts/gate-park-idle-slots.sh` now recognises the server's save-and-clear fallback.  Two things were wrong,
not one:

* the move counter was `grep -c "seq_park:"` — which also counts the **refusal warning**, logged under the
  `seq_park` logger prefix, so a region-backed arm read `seq_park=1 seq_unpark=0` and the gate failed on
  *"parking moved nothing"* while `park == lazyoff` on all four requests.  It now counts actual **moves**
  (`seq_park: seq N [lo, hi) -> [dst, bound)`).
* when no move happened **and** the park arm logged *"cannot park the idle slot: saving it to the prompt cache
  and clearing it"*, that is the legitimate decline (a region-backed/windowed KV, or a slot whose cells are not
  one run): the gate reports the fallback and still requires `park == lazyoff` — the same arm parking falls
  back to.  A mixed arm (some parks, some fallbacks) passes with a note.

Measured (Flash-Next GSQ-IQ3_XXS, MTP draft, `NGPU=1 SM=layer NCMOE=48 CTX=40960`, prose prompt, 96 tokens):

| arm | before the fix | after the fix |
|---|---|---|
| `LLAMA_KV_RESIDENT_CELLS=1024` (2 windows, 3 fallbacks) | **FAIL** `parking moved nothing (seq_park=1 seq_unpark=0)`; `park == lazyoff` on A-D | **PASS** `parking declined -> save-and-clear fallback (3 idle transitions)`, `park == lazyoff` A-D, `parkoff != lazyoff` on B/C/D (the workload still reproduces #210) |
| no window (control) | (had passed) | **PASS** `6 park move(s), 2 unpark move(s) traced`, unchanged oracle |

`check-gpu-faults.sh` clean on every arm's log and the kernel journal; no aborts.  Nothing in the delivery's
patch set changed (the fix is `scripts/`-only, so `release.json`'s artifact hashes are untouched).

## 2. A `-sm tensor` window: [`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md) (design only)

Not implemented (one card, nothing to validate against).  The note's cross-check against `common/fit.cpp`
found four things a blind implementation would get wrong:

1. `margins[0]` in the window block is **simple device 0's** `--fit-target`, not the sum, while `free_dev` is
   the Meta device's **summed** free — so on a split the budget silently omits `(ntd-1) × target`.
2. the probe's `kv_bytes_full[0]` is the **whole logical** KV (the probe is `no_alloc`), so `cell_bytes` and
   `W × cell_bytes` are already **aggregate** figures — correct for the aggregate budget, but they cannot tell
   whether one device can hold *its* share, so per-device feasibility must be checked separately.
3. the pinned-spill **plan** (`llama-kv-cache.cpp`, logical `cell_all`) under-counts by `n_devices×` for a
   **MIRRORED** cache (`cache_idx_*`, the dsv4/glm5-next KV) — invariant 14 says bound the plan, so it must be
   built from the per-device geometry.
4. the split branch's `estimate()` is `model + ntd*(ctx+comp)` and its KV component is the whole logical KV, so
   the KV is counted `ntd×` there — pre-existing, independent of the window, and it is why the window decision
   (which changes `ttarget`) has to move into that branch rather than follow it.

Two code facts make the split window smaller than it looks: the **cell axis is never split** (axis-0 KV-head
split ⇒ `W` is shared), and the Meta buft's `alloc_buffer_n` / `init_tensor` / `get`/`set` already fan out per
child, so the window commit and the lazy host growth need no new per-device logic.  The gaps are the Meta
buft variant, a grow **fan-out** (the CUDA grow proc currently returns success for any non-CUDA buffer — a
silent no-op that would fault the GPU), a Meta-aware `is_kv_buft` for the fit, `region_capable`, and C5 above.

## 3. Not started / decisions (unchanged)

The `auto` policy and Goal C's three questions are the maintainer's (HANDOVER §11); Goal B is not started.

## 4. Goal B's baseline (B0) — naive streaming is a 14 % loss

Goal B's plan is [`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md).  Its first job was to measure what a *streamed
attended set* costs today, with no mechanism at all: block 15 at `W=1` puts both KV caches' tails in pinned host
RAM, and the QSA reader (already sparse and exact, `fattn-qsa.cuh`) then reads its attended set from host.  So
`W=1` **is** the naive streaming arm.

Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, `-c 65536`, q8_0 KV, 51 475-token prompt, `-n 2000`,
`ROCBLAS_USE_HIPBLASLT=0`, 1 GPU, interleaved, 1 rep each:

| arm | prefill t/s | decode t/s | ms/token | arena MiB | residency | hit |
|---|---|---|---|---|---|---|
| plain | 584.13 | **39.19** | 25.52 | 24 539.0 | 60.0 % | 0.9686 |
| `W=1` (spilled; `KVREGION_FREE … host=` > 0) | **604.49 (+3.5 %)** | **33.67 (−14.1 %)** | 29.70 | **25 338.3 (+799)** | **61.9 %** | 0.9727 |

* The attended set is `top_k + r - 1 = 2051` cells/query/layer (GGUF: `indexer.top_k 2048`,
  `compress_ratios 4`) = 3.98 % of a 51 475-token context → ~26 MiB/token of attention read; the indexer scores
  every block, ~82 MiB/token at this depth.  Together ~108 MiB/token ≈ 2 ms at the peak link rate, but the
  measured cost is **4.18 ms** ⇒ either scattered host reads run at ~26 GB/s or part of the traffic is
  unattributed.  **That attribution is B1's job before any policy is designed**, and it is why the plan does
  not start with a page table.
* A 799 MiB arena gain for a 14 % decode loss is the wrong trade *as it stands*: Goal B has to earn those ~12
  points back.  Prefill moves the other way (+3.5 %, capacity-bound), so the loss is a decode-side read.
* Purity is not in question for a mechanism of this class (residency changes *where* the bytes live): the oracle
  is the prefill-logit gate at **100.000 %**, not the drafter ring's acceptance trade.

### B1 (same session) — the crossover is placement, not mechanism

The B0 arm was measuring a **dense** read: on gfx1201 single-token decode is dense at every depth by arch policy
(`qwen4exp.cpp:2506`, `qsa_dense_decode_until = 1<<62`).  Forcing the sparse arm (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`)
and moving the KV between VRAM and host gives four arms per depth (Flash-Next IQ3, `-ncmoe 48`, q8_0, `-n 400`):

| depth | dense+resident | sparse+resident | dense+streamed | sparse+streamed |
|---|---|---|---|---|
| 51 475 | **46.5** | 38.0 | 33.4 | 37.2 |
| 102 899 | **42.4** | 32.6 | 21.0 | 27.8 |
| 180 035 | **37.0** | 25.3 | 14.7 | **21.9** |

⇒ **resident → dense** (VRAM is ~10x the link; sparse adds `O(depth)` indexer compute), **streamed → sparse**
(+32…49 %, growing with depth), and the whole of Goal B is a *fittability/arena* feature like block 15, not a
speed one.  The one missing piece is that the decode-arm policy does not know the KV is streaming — no page
table is needed for the win.  Plan §3.2 has the detail and the revised B2.

### B2 (same session) — implemented: a streaming KV takes the sparse decode arm (+34 / +47 %)

Branch `wip-kv-residency-b` `f33a9d8fe`.  `llama_kv_cache::is_streaming(n_kv)` + `qwen4exp_kv_streams()`; kill
switch `LLAMA_QSA_DENSE_DECODE_STREAM=0`.  Measured with the **PLE flags** (`-ot per_layer_token_embd.*=CPU -lm
none -lzm off` — 27 465 MiB of PLE is otherwise lazy-mapped and made identical runs move ±20 %), 1 GPU,
`-ncmoe 48`, q8_0, `-n 400`, 2 reps, ±0.2 %:

| depth | resident + default | streamed + dense (pre-B2) | streamed + **B2** | arena |
|---|---|---|---|---|
| 102 899 | 42.16 | 23.42 / 23.54 | **31.43 / 31.46 (+34.3 %)** | 25 338.3 MiB / 61.9 % |
| 180 035 | 37.10 | 16.51 / 16.50 | **24.31 / 24.29 (+47.3 %)** | 25 338.3 / 61.9 % |

Text: resident default == `STREAM=0` (853 `77a40bf40be1`, inert on a resident cache); streamed default ==
forced-sparse (1159 `e6080aaf2318`); streamed dense 1158 `0f1ad14496c9` (a documented regime change).  Prefill
untouched.  Plan §3.3.

### B2's context: MTP x depth settles the "Strata is fast with little resident" question

Flash-Next IQ3, `-ncmoe 48`, 1 GPU, PLE flags: resident plain 42.2 / 37.1 t/s at 103k / 180k; **resident + MTP
53.6** at 180k and **75.4** at 5.3k depth; streamed `W=1` + B2 + MTP **36.4** at 180k and **72.8** at 5.3k.  So the
80+ t/s figure is a working-context MTP number (reproduced), and streaming's cost is **−3.5 % at 5.3k, −25 % at
103k, −32 % at 180k**.  At depth, a positional window holding **17 %** of the KV gains **+3 %** over ~0 %
(`W=32768` 25.03 vs `W=1` 24.32 at 180k, sparse arm) — recency is not attendance, which is why Strata's
attendance-driven pool + page table (`STRATA-KV-STREAMING.md` §1) beats a recency window at the deep end.  The
deciding unmeasured number is the attended set's churn / pool hit rate: plan §3.4, B1(c).

### B1(c): the attended set's stability (the number a KV pool's design hangs on)

`qsa-census.py` over a 48-step decode trace (Flash-Next IQ3, `-c 131072`, 102 912 cells, sparse arm,
`GGML_CUDA_DISABLE_GRAPHS=1`): **2051 attended cells/layer/step**, consecutive-step Jaccard median **0.586**;
an LRU pool hits **84.5 % at 4096 cells/layer** and **90.2 % at 8192** (90.3 % plateau = churn-bound), i.e.
**2.5 MiB/token** of host read instead of 25.5.  At equal capacity an **adaptive** policy beats a
**pre-calculated static ordering** (84-90 % vs 63-68 % at 4096-8192) — the same verdict the expert-cache
campaign reached for the arena, so a KV pool should reuse the delivery's policy machinery, not Strata's static
ordering.  A 8192-cell/layer pool costs 107 MiB of VRAM.  Strata's cell is 1056 B/cell/layer (int8, 4-cell
per-layer blocks) vs our 1088 (q8_0) — **no 4x size/accuracy difference**; their sparsity is the model's own
`indexer.top_k = 2048`.  Plan §3.5.
