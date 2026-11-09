# prefill-arena-arm — the `-ncmoe` + `llama-bench` regression is the arena never arming (OPEN, fix measured; nothing in `patches/`)

**Status: OPEN — root cause measured, a 148-line fix candidate is gated locally (`0001-prefill-arena-arm.patch`),
nothing promoted.**  `TODO` **#305**.  Found 2026-11-08 while closing the r236 re-base records: the re-base left one
number unexplained — on the `llama-bench` arm the delivery's host-expert prefill was **−15…−18 %** against unpatched
upstream (`TODO` #296), and upstream's own `--moe-cache-mib` did not change that (its advantage is CPU compute, not
the cache).  This campaign says why, and fixes it.

## 0. The one-line version

A **prefill-only** workload (llama-bench `-p`, llama-perplexity, the embedding tools, `llama-cli -n 0` on a build
without the server warmup) **never arms the expert arena at all**: there is no `≤8`-token pass to trigger the
deferred sizing, and the H2D staging ring stages the expert tables *before* the host path can register them with the
cache.  With no arena, every ubatch streams the **whole host expert set** over the link, which is exactly what the
`-ncmoe` + `llama-bench` number was measuring.  Arm the arena and the same arm reads **2.7× faster than the
delivery and 2.3× faster than upstream**.

## 1. The measurement that names it (1 × R9700, gfx1201, ROCm 10.1, 35B-A3B Q8_0, `-ncmoe 48`, `-fa 1 -t 8`)

`llama-bench -p 512 -n 0 -r 2 -ub 512`, the reported arm:

| build | pp512 |
|---|---:|
| shipped r236 (`build-rocm`, pre-fix, measured directly) | **438.5** |
| shipped r235 binary (pre-rebase proxy) | 438.3 |
| unpatched `a691b2345` (`build-rocm-base`) | 517 / 529 / 532 (3 rounds) |
| **same arm after a narrow pass has armed the arena: `-p 8,512`** | **1116 / 1162** |

`-lv 4` / `-v` on the failing arm says it outright:

```
report_locked: prefill cache staging: ok=0 geom_decline=0 unusable=0 nomatch=234 unsized=0 (the priming pass); arena=0.0 MiB host=0.0 MiB
report_locked: MoE expert cache: h=0.0000 (0/0 reaches), fills=0 evictions=0, tables=0, slots=0, arena=0.0 MiB total
```

`tables=0` — the cache has **no tables**: not a residency, not a hit-rate, not a transport problem.  `nomatch=234`
is the ring asking the cache to fill a table it has never heard of.  The arithmetic agrees: 32 640 MiB of host
experts at the box's measured 28.7 GB/s H2D (`ggml_backend_cuda_stage_h2d_gbps`) is **1.11 s per ubatch**, and the
arm reads 1.17 s per ubatch.  The `-ub` sweep is a pure "bytes per ubatch × ubatch count" law for the same reason.

## 2. Why the arena is never armed (three measured facts)

1. **Deferred sizing waits for a narrow pass.**  `moe_cache_update` arms the arena at priming only when
   `n_tok <= 8` ("OPEN 2": the wide prefill work layout is live, so sizing mid-prefill is smaller).  Every
   `llama-server`/`llama-cli` run has a 2-token warmup pass, so it arms there and the field path is healthy.  A
   prefill-only tool has no such pass, so the condition never becomes true.
2. **The staging ring hides the table set from the cache.**  `sched_stage_issue` stages a host-weight split into the
   ring *before* the input loop, so the loop's `moe_cache_update` call (the only place a table registers, primes and
   sizes) never runs.  Hence `tables=0`, and the sizing cannot happen even if the width gate were relaxed.
3. **The staging hook cannot register on its own** — its 1-D whole-table form has the bytes but not `n_experts`.

## 3. The fix (`0001-prefill-arena-arm.patch`, 7 files, +148/−1, cut against `release.json` tree `a4093f9da`)

sha256 `75b0fcfbb99cdac19762e97446543f54faf4b1e7ca7d7c24aba0048dd479e196` (the file next to this README).

1. **`moe_cache_pending(backend)` — a new `ggml_backend_i` query** (appended, so other backends stay NULL): "the cache
   is enabled here, host expert tables are expected on this device, and the arena has NOT armed".  While it is true
   `sched_stage_issue` leaves a routed host-expert table on the host path, so the cache gets one pass in which to
   observe the table set.  A dense model, `-ncmoe 0` and `MOE_EXPERT_CACHE_MIB=0` all answer "no opinion", so the ring
   is untouched there.
2. **Sizing at any width on the first completing pass.**  `moe_cache_update`'s priming-time `alloc_all_locked()` no
   longer requires `n_tok <= 8`.  A warmup-first workload is unchanged (its warmup arms it first); a prefill-first one
   arms at the end of its first pass.  One `WARN` records that the wide layout was live.
3. **`moe_cache_stage_ready(backend, weight)` — a second query** ("sized, and THIS table has slots").  A routed
   host-expert table the cache cannot fill is left to the plain whole-table upload instead of being put through a ring
   slot plus a slot→input D2D for nothing.  It answers TRUE when it has no opinion (cache off / no host experts), which
   is what keeps the `MOE_EXPERT_CACHE_MIB=0` path byte-identical to r236.

Kill switches, both existing: `MOE_EXPERT_CACHE_MIB=0` (cache off ⇒ both queries are no-ops ⇒ r236 behaviour),
`GGML_SCHED_STAGE=0` (no ring).  No new environment variable.

## 4. What it measures (all on the box, interleaved where it matters; `-r` as shown)

**The reported arm** — `llama-bench -p 512 -n 0 -ub 512 -ncmoe 48`, 1 GPU:

| build | pp512 | vs delivery | vs upstream |
|---|---:|---:|---:|
| shipped r236 | 438.5 | — | −17 % |
| unpatched upstream | 529 (517-533) | +21 % | — |
| **candidate** | **1183-1204** (6 runs) | **+173 %** | **+126 %** |

**`-p 4096` ub sweep** (fix vs shipped r235): `-ub` 512 **1191 vs 439 (+171 %)**, 1024 **1857 vs 877 (+112 %)**,
2048 **2655 vs 1742 (+52 %)**, 4096 **2648 vs 1742 (+52 %)** (`-b` clamps at 2048, so the last two are one shape).
Every cell improves; the pre-fix "3446 at `-ub 4096`" that `TODO` #273 quotes is **not** this invocation — the shipped
binary reads 1741 there (verified today), so there is no large-ubatch cell to protect.

**`-ncmoe` sweep** (pp512, fix vs unpatched upstream): 12 → 1187 / 1185 (parity: the whole host set fits the arena),
24 → 1143 / 756, 48 → 1192 / 536.

**Two devices**: `-sm layer` **1350 vs 454 (+197 %)**; `-sm tensor` **799 vs 799** (deliberately unchanged, §5).

**Field model, `--lazy-mode mlock`** — Qwen3.8-Flash-Next GSQ-RCO-IQ3_S, `-ncmoe 48`, 1 GPU, `-ub 1024`,
`-p 512 -n 64`: prefill **388 vs 296 (+31 %)**, decode 44.6 vs 45.3 (−1.6 %, inside the arm's ±7-8 spread).

**Neutrality (the general case is a server or a CLI):**

| arm | candidate | shipped |
|---|---:|---:|
| `tg128` decode (`-ncmoe 48`, 1 GPU) | 82.94 ± 3.8 | 82.98 ± 3.6 |
| `MOE_EXPERT_CACHE_MIB=0` contract (pp512) | 437.5 ± 0.2 | 438.5 ± 0.3 |
| real-text prefill, `--host-experts auto` (5256-token prose, `-ub 512`) | 1428 t/s | 1423 t/s |
| real-text prefill, `--host-experts pinned` | 999 t/s | 999 t/s |
| 35B-A3B greedy text (139 chars, both tiers, `--seed 42 --temp 0`) | `e2dde701fc04` | `e2dde701fc04` |
| 4B Q8_0 coherence anchor | `1c5d32ac537d` | `1c5d32ac537d` |
| dense 27B Q8_0 pp512 (3 interleaved rounds) | 1543 t/s | 1524 t/s (−1.2 %) |

The dense row is **not** this patch: −1.2 % is r236's documented inherited dense `mul_mat_q` cost (`TODO` #303,
−1.28 % measured in the release gate), and both new queries are structurally inert on a dense model (the preflight
marks a device only when its host expert bytes are > 0, so `pending` is false and `stage_ready` has no opinion).

## 5. Two traps this campaign already paid for (do not re-derive)

* **Do not forward the two new queries through the `-sm tensor` Meta backend.**  The Meta `moe_cache_update` declines
  every prefill width (`n_tok > band`), so under a tensor split the table set can never be observed during a
  prefill-only run: a forwarded `pending` would never clear and the ring would stay disabled for the whole run —
  measured **356 vs 799 t/s**.  The patch therefore leaves the Meta backend alone and a tensor-split pp-only arm keeps
  its r236 behaviour.  Registration at graph build (the Meta split state knows the per-device geometry) is the real
  answer: `TODO` #305.
* **Do not gate on residency instead of width.**  A "stage only at ≥ 75 % residency" variant was tried and reverted:
  at 73 % residency staging still wins **2×** at `-ub 512` (1195 vs 585), and the arm that motivated the threshold
  (`-ub 4096`) was a stale number from another invocation.

## 6. What is left, in order (each is `TODO` #305's sub-items)

1. **`-sm tensor` registration at graph build** (the Meta split state has the geometry) — then the same fix applies to
   the tensor-split shapes, which is where Flash-Next's field config lives.
2. **A post-drop re-size.**  The arena now arms mid-prefill on a prefill-first workload, so it sizes against the wide
   work layout (70.7 % residency at `-ub 4096` vs 84 % at `-ub 512`).  For a prefill-only workload that is strictly
   better than no arena; a server without a warmup would latch the smaller arena.  Re-running the sizing after the
   compute-buffer drop (the `moe_cache_rearm` machinery already re-arms stood-down tables) closes it properly.
3. **The cache-aware prefill (the real prize).**  The candidate still assembles the WHOLE table into a ring slot on
   every ubatch, so ~73-84 % of the bytes are a device-to-device re-copy **out of the arena** (`TODO` #133's original
   finding).  At 73 % residency that assembly costs ~0.1 s of a 0.43 s ubatch; serving the resident experts *in place*
   (the decode band's remap read, extended above the band, with a cache-aware MMQ fusion) is the ~3× that remains and
   the "one transport" direction of `wip/expert-transfer-engine/`.
4. **Promotion gates** if this is promoted: `scripts/gate-prefill-logits.sh` (prefill path), same-seed greedy text on
   `-sm layer` / `-sm tensor` / 1-3 devices, `scripts/gate-host-pool.sh` (the pool tier takes the same scheduler path),
   `scripts/gate-kv-residency.sh`, `scripts/gate-per-block-build.sh`, `llama-batched-bench -npl 1,4,8`, the MTP
   four-axis protocol, `scripts/check-gpu-faults.sh`, and `scripts/validate-set.sh` after the re-cut.  A
   `MOE_EXPERT_CACHE_MIB=0` arm must stay exactly at the shipped number (it is the "cache off" contract).

## 7. How to reproduce

```bash
cd ~/llama.cpp                      # rdna-boosts at release.json.tip 0e6daae4e
git apply ~/llama-cpp-rdna-boosts/wip/prefill-arena-arm/0001-prefill-arena-arm.patch
~/bin/build-llama-rocm
M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench -m $M -ngl 99 -ncmoe 48 -fa 1 -t 8 -ub 512 -p 512 -n 0 -r 5
# ~1200 t/s.  Revert the patch and the same command reads ~438; `-v` shows `tables=0` / `nomatch=234`.
```

Leave ~10 s between arms: the host-expert tier's locked memory is only released a few seconds after the process
exits, and back-to-back arms contaminate each other.
