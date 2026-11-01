# HANDOVER — `archive/work/kv-residency` (TODO **#118**), cold-session start

**Rewritten 2026-10-30 at the end of the tenth session (A6 — the promotion).**  Read this top to bottom
before touching anything; it is self-contained.  Then read [`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) (the measured state of the
build — **§3.7.11 is this session**) and [`MIXED-BACKING.md`](MIXED-BACKING.md) (the design + the driver probe).

> **State in one line: GOAL A IS SHIPPED (block 15; current release **r181**).**  r181 fixed the multi-device
> `--fit` crash (below) and three `auto` wiring defects; **the window is single-device only** — a `-sm tensor`
> run declines it with one message (there is no per-device region buffer type behind a Meta wrapper), which is
> a *documented limitation*, not a bug.  r178 shipped it with a
> regression that made the block unusable with the default `--fit on` — the fit identifies the KV by the region
> buffer *type*, and the fit's `no_alloc` measurement context declined a window, got the plain buffer, so the
> fit could not see the KV and aborted with "does NOT fit in device memory" before the window existed.  **r179
> fixes it** (`llama_kv_region_set_probe`: the probe takes the region buffer explicitly; the runtime path is
> untouched).  Lesson for any future change here: the fit's probe is a *context* too, and it declines a
> window — check `llama-fit-params` on a config that needs the window, not only `llama-cli` on one that
> fits.  The windowed KV is in the delivery as an
> **opt-in** block (`LLAMA_KV_RESIDENT_CELLS=<cells>|auto`; unset = the plain device buffer, byte-identical
> including the OOM), with `ENVIRONMENT.md` §17, `patches/README.md` "Block 15", and the release record in
> `WORKLOG.md` 261030.002.  Both bugs the Flash-Next arm found are fixed in it (`0013` the tensor placement on
> the map grid, `0014` the park decline).  The shipped form differs from the campaign tree in three ways, by
> decision: **A1 is stripped** (the superseded two-base FA read — one VA range needs no second base), the
> **O1 fallback is not shipped** (the region buffer is taken only when a window is active, because O1
> `GGML_ABORT`s on a slab-armed MoE config where the plain arm completes), and the **provider is on by
> default** (`LLAMA_KV_REGION=0` is the kill switch).  What is left in the campaign is **Goal B (not
> started)**, **Goal C** (the drafter ring: measured, not promoted, three questions in `O3-DRAFTER-RING.md`
> §7), the split-window design (A6, not implementable on this one-card box), and the parking gate's
> save-and-clear fallback (fixed in A6, awaiting the maintainer's commit approval; §3.7.13).
>
> **The box.**  One R9700 (gfx1201), 32 624 MiB; the PCIe rework (2026-10-30) makes the link **55.7 GB/s**
> H2D / 56.8 D2H.  Every pre-swap t/s number in this campaign is at 28.8 GB/s and **must not be compared**
> with a new one; the cost model is `bytes / 55.7 GB/s`.
>
> **FIXED in r181 (was the unowned item found during the r180 gating): `--fit` + `-sm tensor` aborted.**  The KV
> probes this block added look up backend proc addresses through `ggml_backend_dev_backend_reg(dev)`, and a
> **Meta** (tensor-split) device has no reg (`device->reg == NULL`), so every multi-device `-sm tensor` run hit
> `GGML_ASSERT(reg)` at model load (three paths: the fit probe, the runtime buft lookup with a numeric `W`, and
> the spill budget/link lookups with a `W` below `-c`).  r181 makes a reg-less device read as "the backend does
> not answer": the window is **declined** on a tensor split with one message from the fit and one from the
> cache (the window is one VMM region per device; a split has no per-device region buft), while `-sm layer`
> and one device are untouched.  The same change fixed three `auto` wiring defects (a decided "no window" was
> reported as "the fit did not run"; every no-knob run logged a KV warning; the pinned-RAM clamp could be
> taken past the VRAM budget).  **That last one is a live hazard worth remembering: an intermediate form
> accepted a 235 GiB pinned-spill plan for a 4B at `-c 8388608` and OOM-killed the box.**  See
> `WORKLOG.md` 261030.005 and [`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) §3.7.12.  **Not done, and still
> the campaign's**: making the window work across a tensor split (per-device region bufts wrapped in a split
> buft), and the `auto` policy questions (it only spills a pinned `-c`, and the arena reserve wins the VRAM).

---

## 1. State in one screen

| what | where |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch `main`, **pushed** (`origin` = `git@github.com:stew675/llama-cpp-rdna-boosts.git`) |
| delivery release | **r181** (`v16-6050b467c-r181`, `release.json`; r178 = the block with a fit/probe regression, r179 = its hotfix, r180 = PR #252 in block 10, r181 = the multi-device `--fit` crash + three `auto` wiring defects) — 16 blocks; **block 15 is this campaign's Goal A**, cut against r177's tip.  The campaign's own patch series is still cut against **r174** (`5bdec04ab`) and is the *measurement* tree, not the shipped one |
| the campaign's TODO handle | **#118** (retired from `TODO.md` once delivered; `WORKLOG.md` 261031.008 closes it and 261031.011 records the retirement.  The older text says `#117`, which was a dangling reference) |
| campaign code | **fork branch `wip-kv-residency`** in `~/llama.cpp`, **pushed to the personal fork** `git@github.com:stew675/llama.cpp.git`, tip **`7c7769e50`** (A5 (ii): the tensor-placement fix `0013` and the park decline `0014`, on top of the A4 tip).  The **shipped** form is the delivery's `rdna-boosts` at **`83d2c391a`** (`release.json.tip`), which carries r181's fixes and differs from this branch |
| the A1 fallback | fork branch `wip-kv-residency-a1-two-base` (`e49e1cef4`), also pushed (superseded; unchanged this session) |
| `rdna-boosts` | **never touched by this campaign.**  Locally `5bdec04ab`; the fork's branch follows the delivery releases (r177) |
| patch snapshot | **two series.**  **Goal A**: `patches/0001`...`0014` + `README.md` + `diffstat.txt` (3222 insertions, 18 files) — apply **in order** to `5bdec04ab` (r174); the shipped form is delivery `patches/0015` (block 15), so this series is the Goal A measurement record only.  **Goal B**: `patches/goal-b/0001`...`0007` + `SHA256SUMS` + `diffstat.txt` — apply **in order** to `8f9dad0be` (r186, `RDNA_BRANCH=rebase-r186`); verified 2026-10-31 to reproduce tree `14e76aa2af4135b13cd3657895f04460d3d80774` == branch `wip-kv-residency-b` (`b98cb3e43`).  Its 0006/0007 (A12, A13) are **delivery fixes** for blocks 10/11, not campaign code. |
| box | **one R9700 (gfx1201), 32 624 MiB.**  Pin `HIP_VISIBLE_DEVICES=0` on every arm.  Every number here is 1-GPU; say so, never claim multi-GPU |

## 2. What Goal A asked for (the maintainer's words, unchanged)

> a **Windowed KV** … for **fully resident dense/MoE models**, because doing so would allow users to run large
> contexts with a Q8_0 quantization of Qwen3.8-27B, or a Q6_K quantization of 35B-A3B, on a 32 GB card, and
> **not run out of VRAM** … and for **host-resident MoE's (slab mode)**, because it would free up more VRAM
> for the decode arena, allowing for higher expert hit rates, leading to higher decode speeds.

**Opt-in semantics (agreed):** default **off** = byte-identical to today including the OOM; the user opts in
to "the KV's tail may spill into system RAM"; the warning is loud and always printed; `--fit` reports the split
instead of refusing.

## 3. The mechanism as built (do not re-derive)

**One KV tensor, one VA reservation, mixed backing.**  The cells below the window are backed by **pinned system
RAM**, the window by **VRAM**, in the same VA range.  That is why there is **no kernel change and no op change**:
the `set_rows` writers, the in-place rope shift behind defrag/`seq_add`, `state_write`/`state_read`, `clear`, the
kq mask and the FA kernels all keep addressing one contiguous range.  (A1 — a two-base read inside the FA
kernels — was built, gated byte-identical, and **superseded**; kept on its branch as the fallback.)

| piece | where |
|---|---|
| the mapping **table** `(off, bytes, kind)`; `map_span(off, bytes, host)`; `ensure_span(...)` which **flips** a mapping's kind through a pinned scratch; `mapped_host()`; the minimum-granularity query | `ggml/src/ggml-cuda/ggml-cuda-vmm-region.{h,cu}` (extends O1's slice 1/2) |
| `ggml_backend_cuda_kv_region_buffer_type_split(dev, cells)`; `grow_cells` (host + device as the watermark advances); `grow_for` (arbitrary-index writes, state restore); the INFO line that reports RAM vs VRAM bytes | `ggml/src/ggml-cuda/ggml-cuda.cu` |
| the window decision (`LLAMA_KV_RESIDENT_CELLS=W`), the loud warning, the uniform-geometry decline | `src/llama-kv-cache.cpp` |
| the A1 policy half (still used) | `llama_kv_cache::get_kv_split_n_tail` |

**The invariants that cost real time to find — keep them:**

1. **The resident window's VRAM must be committed *before* `SLAB_INIT`** (`init_tensor`).  The slab maps
   `free VRAM − reserve`; a lazily-mapped window has no physical left by the time the watermark advances and the
   batch **aborts** (`growth refused … free=0.0 MiB`, measured on Flash-Next).  The **host** half is meant to be
   lazy (do not pin 7 GiB for a 50k-token run); the resident half is not.  (`ba5c7205e`)
2. **The slide boundary is quantised in BYTES to a whole map unit**, never in cells.  `unit/nb1` is not
   unit-aligned, so a cell-quantised boundary makes every unit straddle it and creates a 4 KiB **host** chunk per
   tensor per token (~130 µs each ≈ 4 ms/token: **14.7 → 18.6 t/s**).  (`f50c33be5`)
3. **The slide is monotone** (a cell handed to system RAM is not taken back) and flips the trailing unit to host
   *before* mapping a new one above, so the window's steady-state VRAM is exactly W cells.
4. **One W for every tensor** ⇒ the K/V cell geometry must be uniform across the layers a cache holds; anything
   else **declines** the window (warn once, no spill, today's fit-or-fail) rather than guess.
5. **Every `llama_kv_cache`** is windowed independently — Flash-Next has two (the attention KV and the QSA
   **indexer** cache) and both take the window, one warning line each.
6. A `remap`/flip destroys a chunk's contents: `ensure_span` stages it through a pinned scratch with a device
   synchronise around the unmap.  A capture/graph holds the **VA**, which never changes, so graphs stay valid.
7. **Every mapping in a windowed region must sit on the map-unit grid** (r-A3).  A flip is a whole-mapping
   operation (the driver refuses a sub-range unmap, and re-attaching one mapping as head/mid/tail loses the
   middle piece's bytes — measured), so a boundary *inside* a mapping can only be refused.  Hence `grow_cells`
   maps `used_up` (unit-rounded), `init_tensor` commits `up(W·nb1, unit)`, `vmm_insert` does **not** coalesce,
   and `ensure_span` refuses a request that is not exactly one mapping.  Before this, a coalesced entry that
   straddled the boundary was flipped back whole and the window spilled **nothing** while its warning said it
   had.
8. **A staging copy to/from a host-location mapping must be a kernel copy** (`vmm_device_copy`), never
   `cudaMemcpy`: a direct H2D into a freshly mapped host chunk completes "successfully" and leaves zeros, as
   does a D2D whose source is one (measured: `stage=0x53`, `read=0x00`).  `LLAMA_KV_REGION_DEBUG=1` verifies
   every flip against its stage.
9. The spill is **pinned, non-swappable RAM**: `LLAMA_KV_SPILL_RAM_PCT` (85 % of MemAvailable) clamps the
   window at creation (raising `W`) and refuses a run-time mapping loudly rather than being OOM-killed.

10. **A kind flip needs the hazard hold, or the spill is silently wrong (A4, `A2-MIXED-BACKING.md` §3.7.8).**
    `cuMemUnmap`(device) + `cuMemCreate`(HOST)/`cuMemMap` at that VA zeroes a *different, live* host-location
    mapping in the same reservation.  `vmm_map` takes a hold (a `cudaMalloc` of the mapping's size, before the
    create, freed after) on **every** map site.  Never remove it, never take it only on the flip path, and never
    let a failed hold degrade silently -- a refused hold must fail the mapping.

11. **Every tensor in a WINDOWED region must START on the driver's map granularity (A5, `0013`).**  A flip is a
    whole-mapping operation, so it must coincide exactly with a mapping; the window quantises its boundary in
    whole map units, so a flip is `[tensor_off + k*unit, + (k+1)*unit)` and that only lands on a mapping while
    the tensor's *offset* is gran-aligned.  The plain CUDA alloc size pads a quantized tensor's rows
    (`MATRIX_ROW_PADDING`) **not** on that grid — the Flash-Next QSA indexer pads +512 B per tensor — so the
    windowed buffer type now rounds each tensor's place up to `ggml_cuda_vmm_region_granularity()`.  Without it
    the second tensor's first flip is refused and `grow_regions` **aborts the process** (measured 2026-10-30).
    The 27B/4B are unit-sized by luck; never re-derive a gate on those two alone.

12. **A cell move that the region has not mapped is a GPU fault, not an error — a park is one (A5, FIXED).**  The
    region maps only the cell **watermark**; `llama_kv_cache::seq_park` relocates an idle slot's cells at
    `park_bound()` = the **top of the cache**, above that watermark, and the park's move is a **copy-engine**
    transfer: the GPU faults (`__amd_rocclr_copyBuffer`, *Page not present*).  Measured with **and without** the
    window (§3.7.11.3), so it is the region provider.  `0014` makes a region-backed cache **decline** the park
    (`seq_park` false, warn once) so the server save-and-clears the slot — do **not** move that refusal into
    `can_park()` (a false there means "idle slots stay resident", i.e. #210) and do **not** "fix" it by growing
    the region on a windowed cache (the parked cold cells would take the resident window and the active cells
    would fall into the spilled tail).  Anything that writes above the watermark must grow the region first (a
    host-driven write does it through `grow_for`; a device/copy-engine move does not).

13. **A Meta (tensor-split) device has NO backend reg — never hand it to `ggml_backend_reg_get_proc_address`
    (r181).**  `ggml_backend_dev_backend_reg(meta_dev)` is `NULL` and the proc-address helper ASSERTS on a null
    reg (`ggml-backend.cpp:1052`), so every multi-device `-sm tensor` run aborted at model load — through three
    paths in this block (the fit's KV probes on ANY run, the runtime buft lookup with a numeric `W`, the spill
    budget/link lookups with a `W` below `-c`).  Both selectors now resolve the reg through a helper that
    tolerates a reg-less device (`llama_kv_region_dev_reg` in `llama-kv-cache.cpp`; `fit_dev_reg` in
    `common/fit.cpp`, which walks a Meta device's **simple child** so the answer stays the backend's), and a
    reg-less device reads as "the backend does not answer" → the window is **declined** with one message.
    **So the window is single-device only**: a split would need per-device region bufts wrapped in a split buft
    plus a fit that sizes them together — not implemented, and not measurable on this one-card box (the
    multi-device arms were validated by the maintainer on a 2/3-device host).  A new backend entry point in this
    block owes the same check.

14. **Bound the PLANNED spill, never only the resident window (r181).**  `LLAMA_KV_SPILL_RAM_PCT`'s clamp widens
    `W` so the worst-case spill fits RAM; the fit used to apply it unconditionally and then report the result as
    "does not make this fit either".  It now takes the wider window **only while it still meets the VRAM
    target**, and otherwise reports that the ceiling cannot be served at this `-c` and drops the window.  **Not
    cosmetic:** an intermediate form accepted a **235 GiB** pinned-spill plan for a 4B at `-c 8388608` and
    creating it OOM-killed the test box (83 GB anon-rss, kernel OOM, session lost).  Any change to this
    arithmetic must bound the spill *plan*; a big-`-c` + tiny-ceiling arm is the test.

Knobs (all campaign-only): `LLAMA_KV_REGION` (the provider; **default on**, `=0` disables), `LLAMA_KV_RESIDENT_CELLS=W` (the window)
or `=auto` (**`--fit` sizes it** -- the product shape; unset = no window, i.e. today's cache),
`LLAMA_KV_SPILL_RAM_PCT` (85, the pinned-RAM ceiling), **`LLAMA_KV_REGION_QUARANTINE` (default ON; `=0` is the
A/B arm)** -- the hold that keeps the driver from zeroing a live host mapping when a mapping is created, i.e.
**without it a spill is not numerically transparent**; `LLAMA_KV_REGION_TRACE=1` (every address-level
map/unmap/memset); `LLAMA_KV_REGION_DEBUG=1` (per-growth lines, a byte-exhaustive restore verify, a stage-out
cross-check, and a fingerprint of every host mapping -- **10-40x on a spilling arm, never quote a timing from
it**); `LLAMA_KV_REGION_CANARY=1`, `LLAMA_KV_REGION_DUMP=<path>` (hazard instruments);
`GGML_CUDA_VMM_REGION_SELFTEST=1`, `GGML_CUDA_SLAB_STATS=1`.
**DIAGNOSTIC ONLY:** `LLAMA_KV_REGION_SPILL_SKIP=<tensor indices>` keeps the listed tensors of a windowed buffer
fully resident (it bisected A2 §3.7.6; not shipped behaviour).

## 4. Environment and build

* Build: `~/bin/build-llama-rocm` (ROCm 10.1.0 → `build-rocm/`, gfx1201, ccache).  **Never** pass
  `-DCMAKE_HIP_FLAGS`.  A warm `llama-cli` relink ≈ 5 s; a `.cu` delta 15-30 s; a ubiquitous header
  (`fattn-common.cuh`, `ggml-cuda.cu`) 7-9 min.  Keep ccache ≥ 30 GiB.
* `LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib`; **`HIP_VISIBLE_DEVICES=0`** always.
* `llama-cli` prints its accounting **only with `-v`** — and `-v` interleaves log text into the generation, so
  **hash text from a run WITHOUT `-v`** with `~/llama-cpp-rdna-boosts/scripts/extract-generated.py`.  Read timings
  and the cache/arena lines from a **separate `-v` run** of the same command.
* `--single-turn --no-display-prompt` always; wrap everything in `timeout`; **never two GPU jobs at once**.
* Models: `/llm/models/Qwen3.8/27B/Q8_0` (the Goal A target, 27.1 GiB weights), `…/27B/Q4_K_XL` (iteration),
  `Qwen3.6/35B-A3B/Q8_0` (the Q8_0 MoE arm), `Qwen3.6/35B-A3B/Q4_K_M`, `Qwen3.5/122B-A10B/Q4_K_XL` (48 layers,
  **the arm that shows the arena's value**), `Qwen3.8/Flash-Next/GSQ-IQ3_XXS` (qwen4exp, 512 experts, two KV
  caches; **supported** — qwen4exp is in r174's block 11), `Qwen3.5/4B/Q8_0` (fast smoke).
* Prompts: `prompts/prose-rdna-boosts.txt` (5256 tokens at the 27B), `prompts/kld-corpus-35b-in-domain.txt`
  (23k; `cat` it twice into `/tmp/prompt-46k.txt` for a 51 475-token prompt).

## 5. Measurement rules (each of these cost a session)

1. **`-n 2000` for anything about the expert arena.**  A short run's decode is drowned out by the cache warm-up:
   the same 35B-A3B Q8_0 arm reads 54.6 t/s at `-n 128` and 66.3 at `-n 2000`.  (Maintainer's rule.)
2. **`-ncmoe` must be the model's full layer count** when the point is the arena (35B-A3B 48, 122B-A10B 48,
   Flash-Next 48).  `-ncmoe 40` left eight layers of experts on the device and the arena was half the size.
3. **Read a delta only from arms interleaved back-to-back in one batch** — this box's clocks drift.
4. **Never judge run-to-run determinism on a greedy text hash** (hipBLASLt per-process solving, `AGENTS.md`).
5. Two `llama-cli` runs of the same command differ in text only if something is wrong; the identity gates below
   are the oracle.
8. **Static MTP: never `--spec-draft-n-max` > 4** (maintainer, 2026-10-30: almost always worse above 4; the
   adaptive controller is the tuned path, and a static depth of 4 is also what the Strata comparison uses).
7. **Every Flash-Next arm must carry `-ot per_layer_token_embd.*=CPU -lm none -lzm off`.**  The PLE table is
   **27 465 MiB** and is otherwise lazy-mapped + prefetched, so its reads depend on page-cache residency
   beside 41 GiB of host experts: identical runs moved **±20 %** without the flags and **±0.2 %** with them
   (found 2026-10-30 in B2; all B1/B2 numbers were re-measured under them).  The park gate's `EXTRA_ARGS`
   already had them — that is where they came from.
6. **Never quote a timing from a `LLAMA_KV_REGION_DEBUG` arm.**  Its fingerprint re-reads and hashes every host
   mapping at every growth: a 27B read 397 and 936 ms/token where the product reads 38.5 and 44.8.  (This cost a
   session-half of confusion on 2026-10-30 -- the "slowdown" was the instrument.)
9. **A speculative arm's t/s is (tokens/step) x (step rate): check the draft/accepted counters BEFORE reading a
t/s delta, and use `-n 512`, not `-n 128`.**  Two arms that generate different text can accept 55 % vs 37 %
   inside the first 128 tokens, and a single arm's acceptance decays as the text leaves the prompt's domain
   (180k dense: 55.1 % over 128 tokens, 44.1 % over 512).  Measured back-to-back at 180k, MTP-4: dense **53.53**
   in a `-n 128` window and **48.72** over `-n 512`; sparse 38.41 and 42.92.  The whole "-20 % / -28 % sparse
   penalty" of A9/A10 was this (BA-POOL-DESIGN.md §8.3); the arm's own cost is -7.5 % (103k) / -11.9 % (180k).
   A `-n 512` window on a deep arm costs 10 s of decode against a 200-370 s prefill -- there is no reason to
   measure short.  (At a *shallow* depth a short window additionally carries a 0.3-0.5 s one-time lazy-load
   cost: 5.3k reads 62.3/59.5/58.9 at `-n 128` and 70.9/70.7/70.3 at `-n 512`.)

## 6. Gates — the exact commands and what they must read

```bash
cd ~/llama.cpp
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
P=~/llama-cpp-rdna-boosts

# 0. the primitive -- also covers the HOST-backed span, the aligned window flips, and (A4 (ii)) the
#    HOST-MAPPING HAZARD: 8 tensors x 8.5 MiB flipped one at a time, every earlier window re-verified.
#    Run it a second time with LLAMA_KV_REGION_QUARANTINE=0: it MUST then print FAIL (that is what makes it a
#    test rather than a formality).
GGML_CUDA_VMM_REGION_SELFTEST=1 ./build-rocm/bin/llama-cli -v -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -c 512 -p hi -n 2 --no-display-prompt --single-turn 2>&1 | grep "selftest.*PASS"

# 1. text identity: plain vs window (must be byte-identical).  Hash from runs WITHOUT -v.
for W in "" 2048 4096; do
  LLAMA_KV_REGION=${W:+1} LLAMA_KV_RESIDENT_CELLS=$W ./build-rocm/bin/llama-cli \
    -m /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 \
    -f $P/prompts/prose-rdna-boosts.txt -n 128 --seed 42 --temp 0 --no-display-prompt --single-turn > /tmp/id$W.log 2>&1
  python3 $P/scripts/extract-generated.py /tmp/id$W.log          # all three must print the same 526 chars/sha
done
#   known-good: 27B q8_0 = 526 chars sha=5ba44647188f ; 4B f16 = 591 chars sha=ef279d67a708 ;
#               35B-A3B -ncmoe 24 -c 16384 = 339 chars sha=061add8deaba

# 2. prefill logits (a residency change moves bytes, not values): must read 0.000000 / 100.000 %
$P/scripts/gate-prefill-logits.sh --ctx 2048 --chunks 8 --base /tmp/kld.kld \
  --ab "LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024"

# 3. GPU faults (never grep a log for "page fault" by hand)
$P/scripts/check-gpu-faults.sh --since "<session start>" --log /tmp/id.log … 

# 4. the OOM baseline must still OOM with the feature OFF
./build-rocm/bin/llama-cli -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ngl 99 -c 262144 \
  -ctk q8_0 -ctv q8_0 -p hi -n 4 … 2>&1 | grep "cudaMalloc failed: out of memory"
```

**Added in A3 (all pass, see `A2-MIXED-BACKING.md` §3.7):**

```bash
# the RAM ceiling: a tiny ceiling must widen W instead of being OOM-killed
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_SPILL_RAM_PCT=1 ... llama-cli -c 262144 ...
# --fit sizes and reports the window (the product shape)
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=auto ./build-rocm/bin/llama-fit-params -m <27B Q8_0> -c 262144 -ctk q8_0 -ctv q8_0
# the state paths, on a windowed arm (these found an ABORT: the batched save rejected the region buft)
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=2048 ./scripts/gate-ckpt-reuse.sh
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=2048 NGPU=1 SM=layer NCMOE=0 ./scripts/gate-state-pipe.sh
# MTP verify widths on a windowed arm (4B, spilling): plain/window x decode/draft-mtp must be one hash
```

**Added in A4 (the hazard, all pass):**

```bash
# the sensitive purity arm -- text alone is not enough, and the §6 trio is NOT sensitive (see below)
ROCBLAS_USE_HIPBLASLT=0 $P/scripts/gate-prefill-logits.sh --record \
  --ab "LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024" --ctx 8192 --chunks 4 --kv q8_0 --base /tmp/k.kld
#   must read  -0.000000 / 100.000 %    (with LLAMA_KV_REGION_QUARANTINE=0: 0.004973 / 98.761 %, the old block)
# the text arm that shows it (27B q4_K_XL -c 8192 q8_0, prose prompt, -n 64)
#   plain 305 sha=e2cea6fa6670 ; windowed MUST be the same ; QUARANTINE=0 gives 328 (wrong)
# a 4B arm that REALLY spills (needs -c 16384; at -c 4096 it never reaches a flip)
#   4B -c 16384 -W 1024 : plain 26 sha=158c1671e511, windowed must match
# the isolated reproducer (from the delivery repo; default FAILS, --grab 4 PASSES)
hipcc -O3 --offload-arch=gfx1201 -isystem /opt/rocm-10.1.0-gfx120X/include \
  $P/archive/work/kv-residency/hostmap-hazard-probe.cc -o /tmp/hh && HIP_VISIBLE_DEVICES=0 /tmp/hh [--grab 4]
```

**Added in A5 (item 1 + item 2, all pass on arms that PROVABLY spill — `FLIP > 0`, `KVREGION_FREE … host=` > 0):**

```bash
# the three corners, on spilling arms (proof first, timings from a separate run)
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_REGION_DEBUG=1 ./build-rocm/bin/llama-cli -v \
  -m /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 \
  -f $P/prompts/prose-rdna-boosts.txt -n 8 ... | grep -c 'FLIP device'      # 32 -> it spills
#   MTP verify widths: 27B -c 8192 -n 64, plain/window x decode/draft-mtp = 305 chars e2cea6fa6670
#   gate-ckpt-reuse.sh: 4B, CTX=16384, PROMPT=/tmp/prose-x2.txt (prose twice = 10 502 tok; the shipped prose
#       prompt never reaches a flip at W=2048) -> PASS, KWREGION_FREE mapped=192 host=128
#   gate-state-pipe.sh: NGPU=1 SM=layer NCMOE=0 CTX=16384 -> PASS, mapped=160 host=80 in all five arms

# a Flash-Next (qwen4exp, TWO KV caches) spilling arm -- the fix of 0013 is what makes it run at all
LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_REGION_DUMP=/dev/null ./build-rocm/bin/llama-cli -v \
  -m /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -ngl 99 -ncmoe 48 -c 65536 -ctk q8_0 -ctv q8_0 -f /tmp/prompt-46k.txt -n 32 ...
#   2 x WINDOWED, 156 flips, 156/156 VERIFY ok, KVREGION_FREE attention 576/672 + indexer 48/96 host,
#   plain == windowed text 701 chars 6c0561298cf1, prefill 683.9 t/s, decode 26.5 t/s
# prefill-logit on that arm (LLAMA_ARG_N_CPU_MOE=48 carries -ncmoe through the gate):
LLAMA_ARG_N_CPU_MOE=48 ROCBLAS_USE_HIPBLASLT=0 $P/scripts/gate-prefill-logits.sh --model <flash-next gguf> \
  --ctx 32768 --chunks 2 --kv q8_0 --base /tmp/fnx.kld --ab "LLAMA_KV_REGION=1 LLAMA_KV_RESIDENT_CELLS=1024"
#   -0.000000 / 99.994 %
$P/scripts/gate-park-idle-slots.sh    # NGPU=1 SM=layer NCMOE=48: FAILS on a region/window arm (BUG 2, open)
```

**Added in r181 (the multi-device crash and the `auto` wiring): the gates this box could not have caught, and
how to check them from here.**

```bash
# 1. The reg-less Meta device.  This box has ONE card, so a real split cannot be run: check the CODE PATH
#    instead -- no `ggml_backend_dev_backend_reg(dev)` may reach `ggml_backend_reg_get_proc_address` without a
#    null/reg-less check (r181 funnels them through `llama_kv_region_dev_reg` / `fit_dev_reg`).  A contributor
#    or a 2-device host is the only place to run the arms: --fit on (default) x {-sm tensor, -sm layer} x
#    {no knob, numeric W, auto} must produce TOKENS and no assert, with exactly one message per requested
#    window.  On this box `HIP_VISIBLE_DEVICES=0` and single-device runs still cover the fit's probe walk.
# 2. `auto` wiring, all reproducible on ONE device:
./build-rocm/bin/llama-cli -m <27B> -ngl 99 -c 8192 -p hi -n 4 -v ...   # no knob -> NO "windowed KV" line at all
LLAMA_KV_RESIDENT_CELLS=auto ./build-rocm/bin/llama-cli -v -c 8192 ...   # fits -> "no window is needed", and NO
                                                                        # "sized by --fit, which did not run"
#    and a config where the KV does not fit with a numeric W -> one WINDOWED line, one report
# 3. The planned-spill bound (invariant 14): a big -c with a tiny ceiling must REPORT and drop the window, never
#    plan a giant pinned spill.  The maintainer's case -- 4B, -c 8388608, LLAMA_KV_SPILL_RAM_PCT=1 -- used to
#    plan 235 GiB and OOM-kill the box; with `llama-fit-params` (which does not allocate the spill) it is a
#    cheap check, and it is the arm to run before touching that arithmetic.
```

**A gate that is NOT sensitive, and must not be the regression test:** the §6 identity trio at `W=2048`/`4096`
with `-n 128` **passed while the spill was corrupted**.  Also: **any arm must be checked for `FLIP > 0**
(`LLAMA_KV_REGION_DEBUG=1`) before its result means anything about a spill** -- the campaign's "the 4B is
immune" reading came from arms that never flipped.

**Owed: ~~§7 items 1-3~~ **1, 2 and 4 are done in A5** (§3.7.11); item 3 was done in A4.  What remains is the
**decision** on §3.7.11.3 (a park on a region-backed KV faults the GPU) and the promotion conversation.
Purity for a spilling arm is established on 1 GPU for the 27B, the 4B and Flash-Next.

## 7. What remains

### 7.1 CLOSED (A4) — the spilled KV is numerically transparent

The bug, the isolation, the fix and the gate table are in `A2-MIXED-BACKING.md` **§3.7.8**; the cost is in
**§3.7.9**.  In one paragraph: the flip was always byte-faithful (stage-out == an independent copy-engine read,
restore == stage, exhaustively, 32/32), but `cuMemUnmap` of a device mapping followed by
`cuMemCreate`(HOST)/`cuMemMap` at that VA **zeroes a different, live host-location mapping** in the same
reservation -- so the *spilled cells* of an unrelated tensor were wiped after their flip.  `vmm_map` now takes a
hold (a `cudaMalloc` of the mapping's size, taken before the create and freed after) on **every** map site;
`LLAMA_KV_REGION_QUARANTINE=0` is the A/B arm.  A refused hold fails the mapping loudly instead of proceeding
corrupt.

Do not re-derive any of: the quant type (f16 diverges too), `nb1` alignment (the victims are 2 MiB-aligned VA
offsets, in both models), the 4B's "immunity" (its arms never reached a flip), "a lost kernel write" (the
spilled cells are never rewritten; a 0xAB canary placed after the flip is overwritten with zeros), or the flip's
read/restore paths.

### 7.2 Next, in order

1. ~~**The correctness corners the fix reopened.**~~ -- **DONE (A5, §3.7.11.1)**: `gate-ckpt-reuse.sh`
   (4B, 10 502-token prompt), `gate-state-pipe.sh` (35B-A3B, 5 arms) and the MTP verify widths (27B
   `-c 8192 -n 64`) all **PASS on arms proved to spill** (`FLIP=32` / `KVREGION_FREE … host=128` and
   `host=80`).  Note the shipped prose prompt never reaches a flip on the 4B at `W=2048`; use
   `PROMPT=/tmp/prose-x2.txt` (prose twice).
2. ~~**A Flash-Next (qwen4exp) spilling arm.**~~ -- **DONE (A5, §3.7.11.2)** *after fixing a bug it found*:
   a windowed region's tensors were not placed on the map grid, so the second tensor's first slide was
   refused and the process **aborted** (`0013`, now 156 flips / 156 VERIFY ok, both caches spilling).
   Text, prefill-logit KLD (`-0.000000 / 99.994 %`) and `test-backend-ops -o FLASH_ATTN_QSA` (26/26) all
   green on that arm.  **The park coupling is fixed too** (`0014`): a region-backed KV declines the park
   and the server save-and-clears the slot (byte-identical to the gate's oracle, no fault, cost in the
   noise).  Owed: `gate-park-idle-slots.sh` should recognise that fallback instead of failing on "parking
   moved nothing" (§3.7.11.3).
3. ~~The hazard as a self-test~~ -- **DONE (A4 (ii))**, `GGML_CUDA_VMM_REGION_SELFTEST=1` now includes it and is
   **sensitive** (`LLAMA_KV_REGION_QUARANTINE=0` makes it FAIL).  It only runs under that env var, so keep asking
   for it in every release gate.
4. ~~**The §7.3 re-measurement** at the new link~~ -- **DONE (A5, §3.7.11.4)**: the arena gains **+2.2 to
   +2.9 GiB** (+3.9 to +6.7 points of residency) whichever way the window is sized, but it only pays when `W`
   is **at or above the depth actually run** (no spill: 122B **+3.8 %** decode / +4.7 % prefill, 35B −0.9 % /
   +3 %); a `W` 35 k cells below the depth costs 11-26 %.  `--fit`'s `auto` still refuses to spill *for the
   arena's sake*, so the arena half needs a numeric `W` or a policy change.
5. **The promotion conversation** (a block, `ENVIRONMENT.md`, and the default-on call for the window itself) --
   the maintainer's.  It now needs the §3.7.11.3 decision too.
6. **The split window** -- designed in A6 ([`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md));
   implementing it is the maintainer's call, and its arms need a 2/3-device host.  The §3.7.13 findings
   (`margins[0]`, the mirrored-cache spill plan, the split `estimate()`'s `ntd×` KV) are owed by any change here.

### 7.3 The re-measurement (owed since the 2026-10-30 re-cable) -- **DONE in A5** (`A2-MIXED-BACKING.md` §3.7.11.4)

The arena tables (`-n 2000`; `-ncmoe` = the model's full layer count; arms interleaved), the deep arm, and the
moved `--fit` break-even.  The cost side is already measured (`§3.7.9`): **~61 GB/s effective, ~16.3 us per
MiB of spill per token, a straight line through the origin**, i.e. 128 MiB/token ~ -6 %, 256 ~ -9 %, 512 ~ -18 %,
640 ~ -22 %, and the window apparatus itself is free (zero-spill arms == plain).

## 8. The numbers (1 GPU; all at **28.8 GB/s** H2D — STALE, see §9)

> **Do not quote anything in this section as a current number.**  The link doubled (55.7 GB/s) and the
> maintainer's decision is that these tables are re-measured **after** §7.1 is closed.  The only post-swap
> numbers are the ones in `A2-MIXED-BACKING.md` §3.7 (fit sizing/reporting, the RAM ceiling, the state gates,
> the MTP arm) and they are correctness/behaviour, not throughput.

**Fittability** (27B **Q8_0**, 51 475-token prompt → 53 475 used, `-n 2000`):

| arm | decode |
|---|---|
| `-c 65536` fully resident (the reference) | 18.4 t/s |
| `-c 262144`, `W=55296` (nothing spills) | **18.6 t/s** |
| `-c 262144`, `W=49152` (131 MiB/token of spill) | **18.6 t/s** |

The OOM config runs (8704 MiB KV = 8160 MiB system RAM + 544 MiB VRAM), prefill 933-1113 t/s, and the spill
hides behind that arm's 1.29 GiB/token of CPU-mapped weights.

**The arena's value** (`-c 262144`, `-n 2000`, all experts on the host):

| model | arena | residency | prefill | decode |
|---|---|---|---|---|
| 35B-A3B Q8_0 `-ncmoe 40` | 25 245 → 27 922 MiB | 77.3 → 85.5 % | +1.8 % | +1.7 % |
| 122B-A10B Q4_K_XL `-ncmoe 48` | 19 661 → 22 807 | 29.3 → 34.0 % | +8.6 % | **+7.9 %** |
| Flash-Next IQ3_XXS `-ncmoe 48` | 22 061 → 25 818 | 53.9 → 63.1 % | **+13.1 %** | +2.0 % |

**Prefill is capacity-bound** (a prefill routes *every* expert); **decode is hotness-bound** (10 of 512 routed,
and the hot set already fits where residency is high) — it only converts to a big decode gain where the arena
is starved (the 122B at 15.6 % residency gained **+10.1 %**).

## 8.1 Absolute references measured on the RE-CABLED box (A4, 2026-10-30, 1 GPU)

These are the numbers to compare a new arm against.  They are the *current* box (55.7 GB/s link); the §8 tables
below are the pre-re-cable ones and must not be mixed with them.

**27B Q8_0 fully resident (`qwen3.8-27B-Q8_0`, 27.05 GiB weights), q8_0 KV, decode:**

| arm | depth | t/s |
|---|---|---|
| `llama-bench tg128` | 0 | 18.27 ± 0.50 |
| `tg128 @ d8192` | 8 192 | 19.41 ± 0.36 |
| `tg128 @ d16384` | 16 384 | 19.46 ± 0.01 |
| `tg128 @ d32768` | 32 768 | 19.12 ± 0.01 |
| `llama-cli -c 65536` | 5 298 | 19.76 |
| `llama-cli -c 65536` | 51 475 | 18.69 |
| `llama-bench tg128`, **f16 KV** | 0 | 20.16 |
| `tg128 @ d16384`, **f16 KV** | 16 384 | 19.43 |

So the floor is ~18.3 and the ceiling ~20, peaking in the 8-16k band; **q8_0 KV costs ~10 % of shallow decode
against f16** (that is the price of the VRAM saving).  **A remembered ~22 t/s for this arm does NOT reproduce**
(checked: clocks and temperatures are fine, 54/58/64 C).  If that figure was this exact arm on an older
release, there is a 10-15 % regression between then and r176 and it is worth a `tg128 -d 0` bisect; the other
candidates are f16 KV (+10 %), a different quant (27B Q4_K_XL measures 27.7 t/s at 25.7k), or MTP.

**The spill's cost** (`A2-MIXED-BACKING.md` §3.7.9, 27B Q4_K_XL at 25 763 depth, `-c 32768`): decisive fact --
it is a **straight line through the origin**, ~**16.3 us per MiB of spill per token** (~61 GB/s effective; the
link is 55.7), and the window apparatus itself is free (zero-spill arms measure the same as plain).  128
MiB/token ~ -6 %, 256 ~ -9 %, 512 ~ -18 %, 640 ~ -22 %.  Never a cliff.

**Worst case (`W=1`) and the product shape (`--fit`), 27B Q8_0, `-c 262144`, 51 475-token prompt:**

| arm | KV VRAM | KV pinned host | prefill | decode |
|---|---|---|---|---|
| `LLAMA_KV_RESIDENT_CELLS=1` | **128 MiB** | **8 576 MiB** | 944.8 t/s | **81.06 ms/token = 12.34 t/s** |
| `LLAMA_KV_RESIDENT_CELLS=auto` + `--fit` (sized `W=123385`) | 4 096.8 MiB | ~0 at this depth | 1113.4 t/s | **53.57 ms/token = 18.67 t/s** |

* **W=1 and W~3855 are the same arm**: the slide boundary is quantised to a whole 4 MiB map unit, so the real
  window floor is 32 x 4 MiB = 128 MiB of VRAM, not one cell.  (A cell-quantised boundary is what cost 14.7 ->
  18.6 t/s in A2 (ii).)
* `--fit` reports and keeps the user's context: *"WINDOWED: 123385 of 262144 cells resident (4096.8 MiB of
  8704.0 MiB), 4607.2 MiB spilled ... 31386.0 MiB projected vs a 1024 MiB target"*, and it declines to spill at
  all below depth 123 385.  At the full 262144 context it prints the spill read as ~90.7 ms/token; W=1 at full
  depth works out to ~142 ms.
* **The worst case is still PURE**: `-c 65536` at the same 51k depth, plain (fully resident) vs `W=1` (spilling
  ~1.6 GiB/token) -> **579 chars `7a4d02bc9c2b`, byte-identical**.
* Extrapolated to the full 262144 context: W=1 ~5 t/s, fitted window ~7 t/s -- slow, but the alternative for
  that config is `cudaMalloc failed: out of memory`.

A full 262144-token *prefill* was not run (it is quadratic in the spill: ~2.3 TB of link reads, ~42 s on top of
the compute).  Do not quote one.

**The arena half of Goal A, measured at this link (A5, `A2-MIXED-BACKING.md` §3.7.11.4).**  `-n 2000`,
`-ncmoe` = full layer count, `-c 262144`, q8_0 KV, 51 433-token prompt, two interleaved reps each:

| model | arm | arena | residency | prefill | decode |
|---|---|---|---|---|---|
| 35B-A3B Q8_0 | plain | 25 245.0 MiB | 77.3 % | 1194.7 | 67.79 |
|  | `W=53248` (no spill) | 27 412.5 | 84.0 % | 1236.6 | 67.07 |
| 35B-A3B Q8_0 | `W=16384` (320 MiB/tok) | 27 667.5 | 84.8 % | 1172.4 | **49.69** |
| 122B-A10B Q4_K_XL | plain | 19 661.1 | 29.3 % | 365.6 | 28.75 |
|  | `W=49152` (no spill) | **22 282.6** | **33.2 %** | 382.7 | **29.84** |
| 122B-A10B Q4_K_XL | `W=16384` (384 MiB/tok) | 22 544.8 | 33.6 % | 372.4 | **25.57** |

**The arena's prize is +2.2 to +2.9 GiB whichever way the window is sized; it only pays if `W` is at or above
the depth actually run** (then the KV's *lazy commitment* — not a spill — is what frees the VRAM).  A `W` 35 k
cells below the depth costs 11-26 % of decode.  `--fit`'s `auto` still sizes a window only when the KV does not
fit at all ("*no window is needed — the 262144-cell KV (2720.0 MiB) fits fully resident*" on the 35B), so the
arena half needs a numeric `W` or a policy change.

**Flash-Next spilling arm (A5, §3.7.11.2)**, `-ncmoe 48 -c 65536` q8_0, `W=1024`, 51 475-token prompt: two
windows (attention 816 MiB, QSA indexer 102 MiB), 156 flips / 156 VERIFY ok, **576 + 48 MiB spilled**, prefill
**683.9 t/s**, decode **26.48 t/s** (37.8 ms/token, ~11 ms of it the 624 MiB/token spill), text `6c0561298cf1`
on both arms.

## 9. The box HAS changed (2026-10-30) — re-baselined

The PCIe rework is done: the root port is x16 @ 32 GT/s, so **H2D 55.7 / D2H 56.8 / kernel read of pinned
host 55.7 GB/s** (`bw-probe.cc`, 512 MiB and 1024 MiB, 5 reps, best of), and `mixed-backing-probe.cc` still
passes every check (T8: 768 host chunks, 53.4 GB/s).  Consequences:

* the cost model is `bytes / 55.7 GB/s` — a spill costs **half** what the campaign's tables assumed, so the
  `--fit` break-even of §3.6 moved;
* the prefill-logit base was re-recorded on the new box (`PPL 5.0803`), and the window arm still compares at
  **0.000000 / 100.000 %** at `--ctx 2048`;
* `llama-fit-params` and the KV warning line now read the **calibrated** link from the backend
  (`ggml_backend_cuda_stage_h2d_gbps`, 53 GB/s here) instead of a hard-coded 28.8;
* `LLAMA_KV_SPILL_RAM_PCT` (85 %) caps the pinned spill; the campaign's "up to 8.5 GiB at `-c 262144`" is now
  enforced rather than assumed.

## 10. Records

| record | content |
|---|---|
| [`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) | **the state of the build**: what is implemented, every gate, §3.5 the arena numbers, §3.6 the deep arm + the break-even, §3.7 the re-baseline at the new link, and **§3.7.11 = the A5 session** (the spilling-arm gates, the Flash-Next arm, the two bugs it found and their fixes, the arena re-measurement) |
| [`MIXED-BACKING.md`](MIXED-BACKING.md) | the design + the driver probe (one VA range can be backed partly by pinned host memory; host chunks; the real shape at scale) |
| [`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md) | **A6**: a windowed KV across a `-sm tensor` split — design only (five changes, the `fit.cpp` split-accounting cross-check, the invariants, the arms a multi-device host owes) |
| [`qsa-census.py`](qsa-census.py) | the attended-set census replayer (LRU / F-LRU / static / region-unit / page-table pool) |
| [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) | **B-A**: the attendance pool for the attention KV — design, the measured hit rates it stands on, what it is worth, the four gated build steps, and the two refuted alternatives |
| [`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md) | **Goal B**: stream a sparse attended set — the plan of record (B0's measured baseline, the three candidate mechanisms, the staged plan B1-B4 with each stage's gates, the eligibility rule and the risks) |
| [`A1-TWO-BASE-KV.md`](A1-TWO-BASE-KV.md) | A1 (two-base read in the FA kernels): built, gated, **superseded**, kept as the fallback |
| [`O2-PLAN.md`](O2-PLAN.md) | the plan of record as written (its §6.3 mechanism is now historical — mixed backing won) |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | O1 (the growable KV region) — **parked**; A2 reuses its buffer-type skeleton |
| [`INVENTORY.md`](INVENTORY.md), [`VMM-PROBE.md`](VMM-PROBE.md) | the taxonomy and link constants; the VMM driver facts |
| [`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md), [`O3-DRAFTER-RING.md`](O3-DRAFTER-RING.md) | reference designs; Goal C (done, last priority) |
| `patches/` | `0001`-`0014` + `README.md` + `diffstat.txt` — the campaign's code, apply to `5bdec04ab` in order |
| [`COLD-START-PROMPT.md`](COLD-START-PROMPT.md) | **the prompt to paste into a fresh session** (refreshed for the next one at the end of every session) |
| `mixed-backing-probe.cc`, `vmm-probe.cc`, `bw-probe.cc` | the three standalone probes |
| [`hostmap-hazard-probe.cc`](hostmap-hazard-probe.cc) | **the isolated reproducer of the A4 hazard**: replaying the window flip shows a live host mapping being zeroed; `--grab` passes, `--precreate`/`--no-flip`/`--no-stage`/`--vram-mib` still fail |

## 11. First 30 minutes of the next session

**Goal A is shipped (block 15; r178-r181, set now at r184) — do not re-measure it and do not re-promote it.**  The
shipped description is `patches/README.md` "Block 15", `ENVIRONMENT.md` §17, `WORKLOG.md` 261030.002 / .003 /
.005, and `A2-MIXED-BACKING.md` §3.7.11 + §3.7.12 (the latter is the maintainer's r181 analysis).  What is left
in the campaign:

| item | state |
|---|---|
| **Goal A, single device** | **SHIPPED**: opt-in (`LLAMA_KV_RESIDENT_CELLS=<cells>|auto`), unset = the plain device buffer and the same OOM; purity proven on spilling arms; r181 fixed the multi-device crash and three `auto` wiring defects |
| **Goal A across a `-sm tensor` split** | **DESIGNED, not implemented** — [`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md): the five changes (Meta buft variant, grow fan-out, fit recognition, `region_capable`, fit sizing inside the split branch), the `common/fit.cpp` cross-check (four findings, `margins[0]` first) and the arms a 2/3-device host must run.  Still hardware-blocked here (one card) |
| **`auto` policy** (the arena half) | **the maintainer's call, unchanged by r181.**  `auto` sizes the window to `free − margin − everything-but-the-KV`, where the margin includes the **arena/slab reserve**, so on a `-ncmoe` config the arena wins the VRAM and the window comes out small *by design*; and it only spills a pinned `-c`.  The measurement that decides it is `A2-MIXED-BACKING.md` §3.7.11.4 A–C.  `W` stays a cell count; a percentage-of-slab form is deferred — do not re-propose it |
| **owed: a TEST gap in the delivery** | **FIXED (A6; committed as `e85c0164`)**  : `scripts/gate-park-idle-slots.sh` now counts actual park MOVES (it used to count the region refusal warning, logged under the same `seq_park` prefix) and, when the park arm logs *"cannot park the idle slot: saving it to the prompt cache and clearing it"*, reports *"parking declined -> save-and-clear fallback"* and still requires `park == lazyoff`, instead of failing on *"parking moved nothing"*.  Verified on a windowed Flash-Next arm (3 fallbacks -> PASS) and on the no-window control (6 park / 2 unpark moves -> PASS).  `A2-MIXED-BACKING.md` §3.7.13 |
| **Goal B** — a sparse attended set (QSA/DSA) streamed instead of a full KV | **DELIVERY IS THE NEXT SESSION'S GOAL — the close-out plan is §12 (start there, not with the measurements below).**  `B1, B2, B1(c) DONE (A9); B-A STEPS 1+2 BUILT AND GATED (A10); A12/A13 fixed and gated (A12/A13 sessions)** on branch `wip-kv-residency-b` (tip `b5d895c59`): `LLAMA_KV_REGION_POOL=<cells>` = pinned host mirror (authoritative) + compact device pool of indexer-block pages + page table + an in-graph resolve (`GGML_OP_KV_POOL`) with a second id list for the reader's K/V gather. Gates: text identical with the pool on vs off (854 `43b30de151b9` at 51 475 depth), the pool's verify kernel 0/0 mismatches, prefill-logit 0.000000/100.000 %, park/ckpt/state gates PASS, `FLASH_ATTN_QSA` 27/27, no GPU faults, **89.7 % hit rate at 4104 pages**.  And the headline: at 103k with MTP-4 the pool is **48.3 t/s = the sparse ceiling (sparse+resident 47.9)**, +54 % over B2's spilled arm, at **204 MiB instead of 816 MiB** — the residual -20 % against resident-dense is the SPARSE ARM's own cost, not placement.  **B3 is measured too**: at `-n 2000`, `-c 131072`, `-ncmoe 48`, the arena goes 23 579.8 -> 25 018.6 MiB (**+1.44 GiB**, +18 slots/table, expert hit 96.26 -> 97.02 %, cold reads 42.0 -> 31.6 GiB) for a 204 MiB pool instead of an 816 MiB KV; plain decode pays the sparse arm's -23 % (43.1 -> 33.0 t/s), and `identity fast path ON` is 0 in both arms (313/512 slots/table, so the r117 cliff is not in play).  `BA-POOL-DESIGN.md` §7.4-§7.5.  Previous:  Plan: [`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md) (§3.1-§3.6); pool spec: [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md); census tool: [`qsa-census.py`](qsa-census.py).  **B1**: a *resident* KV wants DENSE decode, a *streamed* KV wants SPARSE (+34/47 % at 103k/180k).  **B2 (implemented, branch `wip-kv-residency-b`)**: `is_streaming(n_kv)` + `qwen4exp_kv_streams()` → +34.3 % at 103k, +47.3 % at 180k; `LLAMA_QSA_DENSE_DECODE_STREAM=0` is the A/B arm; text inert on a resident cache.  **B1(c)**: 2051 attended cells/layer/step, Jaccard 0.586; LRU pool 84.5 % @4096 cells/layer, 90.2 % @8192 (2.5 MiB/token vs 25.5 zero-hit); **adaptive beats a static ordering** (84-90 % vs 63-68 %); and block 15's region cannot express it (its 4 MiB unit = 3855 contiguous cells → **6 % hit, 90 ms/step** of flips).  **B-A = a compact pool + page table (page = 4 cells = one indexer block) + a host-authoritative mirror + an in-graph resolve, lag-1 admission, adaptive policy: 85-90 % hit, 0.05 ms/step of fills, 107 MiB for 12 layers.**  **Next = B-A step 1** (mirror + pool + page table as a KV cache mode; the design's build order).  Then B3 (arena half at `-n 2000`), the MTP draft's sparse-decode re-measure under a streamed cache, and the promotion decision  **A11 (2026-10-31) finished all three of those and reframed the campaign** (`BA-POOL-DESIGN.md` §8): the acceptance table is complete at **5.3k / 103k / 180k** with `-n 512` interleaved arms — dense **70.9 / 58.3 / 48.7** t/s, sparse 70.7 / 53.9 / 42.9, **pool 70.3 / 54.2 / 43.1** (i.e. the pool is ON the sparse ceiling everywhere, and the sparse penalty is **-0.4 / -7.5 / -11.9 %**, not -20 %: A10's `-n 128` window was measuring the two arms' *acceptance*, which differs 55 % vs 37 % in the first 128 tokens because they generate different text — §5 rule 9).  A rocprofv3 census attributes the sparse arm per verify step at 103k: indexer score chain **+6.7 ms**, radix top-k +0.9, **FA -5.8 (3.14 -> 1.09)**, pool resolve **+0.3**, copies +2.8, text/expert drift +3.4; plain decode shows the chain is *preparation* of the indexer keys (dequantizing key read 2.17 ms/token, whole-cache copy 1.83, bias 1.18, rope 1.08, rms 0.40) rather than the score matvec (0.28).  **So the pool's own machinery is 0.10-0.11 ms/token and its fills 0.03-0.05 ms/step: a page policy (step 3) is dead**, and the lever is the indexer score chain.  Step 3's other half also closed: `LLAMA_MTP_SPARSE_DECODE=1` under a pooled target is **-7.8 %** with identical text and acceptance, so its default-OFF stands |
| **Goal C** — the drafter ring (`LLAMA_SPEC_DRAFT_WINDOW`) | **measured, NOT promoted** (`O3-DRAFTER-RING.md`): free at a usable window (46k context, `W=32768`: acceptance −2.3 % rel, generation flat), 7-9 % lost far below the working context; the VRAM prize is one draft layer (~1 GiB at `-c 262144`) and it does not reach the arena (the draft KV lives in the slab's `aux_reserve`).  Three open questions in §7 |

**A13 (2026-10-31) added — the indexer item is DONE:** the fused `INDEXER_SCORE` + derived `INDEXER_FILL` path
now accepts **quantized** indexer keys (q8_0 and the q4/q5 family + iq4_nl; the k-quants cannot be served because a
256-element superblock does not divide `idx_dim` = 128).  Branch `b98cb3e43`; the A/B is the pre-change chain,
`GGML_CUDA_QSA_INDEXER_SCORE=0`.  **Byte-identical on all eight key types** (5.3k greedy: q8_0 == the pre-change
hashes 296 `91ef8940d66e` / sparse 332 `dd1981213373`; f16/bf16/q4_0/q4_1/q5_0/q5_1/iq4_nl each fused == chain),
51 475-depth pool arm 854 `43b30de151b9`, prefill-logit `--ub 1` A/B **0.000000 / 100.000 %**, op test OK, no
faults.  Worth at 103k: **plain decode 23.68 -> 17.73 ms/token of kernels and 29.3 -> 36.3 t/s (+24 %)**, i.e. the
dequantizing kernel delivers exactly what the bf16/fused proxy did; MTP-4 54.26/54.52 -> **54.70/54.65** (dense
59.10), small because the indexer is amortised over 2.84 tokens/step.  Record: `BA-POOL-DESIGN.md` **§10**.

**A12 (2026-10-31) added:** the branch is **rebased onto r186** (`rebase-r186` = `8f9dad0be`, tree ==
`release.json`; the pre-rebase tip is kept as branch `wip-kv-residency-b-r182` = `b5d895c59`), gates green (text
byte-identical to the pre-rebase build, prefill-logit 0.000000/100.000 %, `FLASH_ATTN_QSA` OK, no faults).
The rebase exposed a **delivery regression** in the sparse arm and it is **fixed**: r183's 2-D copy guard
refuses the qwen4exp indexer chain's per-step whole-cache F32 copy (47 MiB at 103k, once per layer), and its
fallback (`cpy_scalar`, 3 integer divisions per element) cost 5.20 ms/token — the arm read 50.42 t/s against
the pre-r183 53.92.  `cpy_2d_vec16` (one 16-byte unit per thread, bit-identical, the guard untouched) gives
**54.26 / 54.52 t/s** (sparse / pool) and 23.68 ms/token plain decode (r182-era: 53.92 / 54.20 / 23.78).
Commit `f8e47e392`; it belongs in **block 10**, and it is a delivery fix candidate, not part of B-A.  The same
session found the chain's **root cause**: with a quantized indexer key cache (`-ctk q8_0`, the delivery's
large-context type) the fused/derived scorer is skipped wholesale (`idx_key_float` in `qwen4exp.cpp`;
*"derived indexer cache pool skipped (q8_0 keys)"* in the log), and the bf16 arm shows what the fused path is
worth — plain decode **27.0 -> 17.8 ms/token** of kernels (**+23.5 %** wall), MTP-4 15.3 -> 14.2 (flat, the 2x
KV bytes cost the arena).  Full record: `BA-POOL-DESIGN.md` **§9**; the item's restatement is §9.4.

**A11 (2026-10-31) added, so do not redo it:** the acceptance table at all three depths, the sparse arm's
kernel census and the MTP-draft re-measure (`BA-POOL-DESIGN.md` **§8**), plus §5's new **rule 9** (an MTP arm's
t/s is tokens/step x step rate — match the acceptance or the delta is meaningless).  The harness (exact env per
arm, prompts, `-c`, mirror sizes) is §8.1; the census method (two `-n` lengths of the same arm, subtract, filter
to kernels whose call *count* scales) is §8.4.  Headline: the pool is on the sparse ceiling at 5.3k/103k/180k
(70.3 / 54.2 / 43.1 t/s), the sparse penalty at depth is 7-12 % rather than the 20 % A10 recorded (window
artefact), the pool's own cost is ~0.1 ms/token so the **page policy is dead**, and the next engineering item is
the **indexer score chain** (~6.7 ms of a verify step; its per-step key re-derivation is the bulk).  Nothing was
changed in code — this session is measurement + records only, so the campaign branch tip is unchanged.

**A10 (2026-10-31) added, so do not redo it:** B-A steps 1 and 2 (`BA-POOL-DESIGN.md` §7): the mirror + pool +
page table + resolve are built and gated, and the measured answer to the design's acceptance question is that
**the pool reaches the sparse arm's ceiling and the sparse-vs-dense gap is what remains** — so a page POLICY
(step 3) can only recover ~0.05 ms/step, and the next lever is the sparse decode arm's indexer cost.  Three
invariants came out of it (§7.3): mark-then-claim, distinct-block claims, and no per-launch scratch in a captured
graph.  Machine note: `git push` over the `git@` remote fails on this box (no ssh-agent, no working key) —
`git push https://github.com/stew675/llama.cpp.git <branch>` works and is the same personal fork.

**A9 (2026-10-30) added, so do not redo it:** B1's placement matrix (resident→dense, streamed→sparse at 5.3k /
103k / 180k), the **MTP x depth** framing (resident + MTP 75.4 / 58.2 / 53.6 t/s), the **PLE rule** (§5 rule 7) and
the **static-MTP depth <= 4** rule (§5 rule 8), B2 on `wip-kv-residency-b`, the census instrument +
`qsa-census.py`, and [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md).  Twice corrected by measurement: a recency window
cannot stand in for attendance, and neither can the region's unit granularity.

1. `cd ~/llama-cpp-rdna-boosts && git pull` (and `cd ~/llama.cpp && git fetch origin`); read this file, then
   [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) (the spec you are building), then [`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md)
   §3.4-§3.6 (why the pool is the only form left), then `A2-MIXED-BACKING.md` §4.
2. Branch state: the **shipped** tree is `rdna-boosts` at `release.json.tip` (r184 at the time of writing); the
   **Goal B campaign** branch is **`wip-kv-residency-b`**, based on that tip, tip **`c045bf866`** (B2 + the census
   diagnostic; pushed).  `wip-kv-residency` (`7c7769e50`) is Goal A's measurement tree — do not develop on it.
   Never commit campaign code to `rdna-boosts`, and never touch `patches/` or `release.json`.
3. **This box is one card and may be busy.**  Check `rocm-smi --showmeminfo vram` before any gate — a live
   server (the maintainer runs Flash-Next here) holds ~34 GB and makes every fit/gate read nonsense
   (`free = 2 MiB`).  Never run a second GPU job; if the card is in use, say so and stop.
4. Anything you change owes the gates in §6, plus this feature's own: text byte-identical on a **resident** arm
   (pool off vs on), the pool arm's text equal to the resident **sparse** hash, `test-backend-ops -o
   FLASH_ATTN_QSA`, and `check-gpu-faults.sh` over every log.  Report findings to the maintainer before committing
   any delivery-affecting change, and **push the campaign branch after every step** (these arms can OOM this box).

## 12. Goal B close-out plan — the next session's job (deliver it, don't re-measure it)

> **STATUS 2026-10-31 (A14): DONE — Goal B shipped in r190 (`v16-6050b467c-r191` after the fold).**  All twelve rows of §12.1 below
> were carried out: the 180k acceptance row was measured on the final build (§8.2), the campaign branch was
> re-based onto the released r189 tree, and the three deliverables were folded into the blocks that own them --
> **block 15 = the attendance pool** (the pool + its async-state mirror handling + this session's `--fit`
> accounting, folded together with the window that was already block 15) and **B2** (a streamed cache takes the sparse decode arm) and **block 11 = A13**
> (quantized indexer keys); A12 shipped separately as r189.  The `--fit` blocker (§12.2) was **fixed**, so the
> rollback criterion did not apply: the mirror is charged to the host bucket in `memory_breakdown()`,
> `llama_get_kv_pool_bytes()` reports the split, and the fit refuses to interpolate `-c` while the pool is
> active.  The grow-on-demand mirror (§12.3) shipped as a **documented limitation**, now `TODO` **#123**.
> `TODO` #118 is closed by `WORKLOG.md` 261031.008; what remains of the campaign is `TODO` #121 (Goal C),
> #122 (the Goal A `auto` policy), #124 (Goal A across `-sm tensor`), #125 (the radix top-k) and #123.
> The gate list of §12.5 is recorded in `patches/README.md` block 15 and in the WORKLOG entries; the numbers
> reproduce this document's bit for bit (the pool's prefill-logit base is the same PPL 2.9225 / sha256
> `acfb4145…` this campaign recorded).

**The maintainer's decision (2026-10-31): Goal B is to be finalised, wrapped up and DELIVERED into the delivery
repo.**  Goal C and the two open Goal A items are carried forward, not dropped.  Report any finding to the
maintainer before each delivery-affecting commit (AGENTS.md), but the *promotion itself is decided*: do not stop to
re-ask whether to promote.

Everything below assumes the rules in §5 (measurement) and §6 (gates) and the delivery flow in `AGENTS.md`
("Pushing policy", "Common tasks → Regenerate the patches / Build the fork").  **Re-base first**: the campaign
branch is on r186 and the delivery has moved on (r188 at the time of writing) — always re-base onto
`release.json.tip`, never a hard-coded SHA.

### 12.1 Order of work

| # | task | done when |
|---|---|---|
| 0 | **Produce the last stale number**: the 180k acceptance row (`acceptance-arms.sh 180k`, one interleaved batch, `-n 512`, MTP 4) | §8.2's 180k row is measured on the final build |
| 1 | **Re-base** the campaign branch onto the current `release.json.tip` (`git rebase --onto <new-tip> rebase-r186 wip-kv-residency-b`; keep the old tip as a `*-r<NNN>` branch, push both) | text + pool-verify + prefill-logit + the three state gates are green on the new base |
| 2 | **Ship the two repairs first** (they are not campaign code): fold **A12** (block 10) and **A13** (block 11) into their owning blocks and cut the 16-patch set — *or* ship them as their own micro-release ahead of the pool (the maintainer's call; the micro-release is the lower-risk option and de-risks the pool) | `scripts/validate-set.sh` green, coherence + prefill-logit + the op tests green on the folded set |
| 3 | **`--fit` accounting for the pool — the hard blocker** (see 12.2) | `llama-fit-params` reports the pool's 204 MiB and *not* the mirror's 1.6-2.4 GiB as VRAM, and a config that only fits with the pool is accepted |
| 4 | **Decide grow-on-demand mirror** (12.3): implement, or ship it as a documented limitation | decision recorded in the block note + `TODO.md` |
| 5 | **Package the pool as a block** (12.4): `patches/0016-*`, `release.json` via `scripts/make-release.sh`, `patches/README.md` block entry, `ENVIRONMENT.md` §, `MANIFESTS.md` apply order + verification contract, `AGENTS.md` naming (16 → 17 blocks), `WORKLOG.md` entry, `TODO.md` #118 closed to that entry | `validate-set.sh` + the tier-1 CI workflow green; the block's own gate list (12.5) recorded |
| 6 | **Release it**: tag, `scripts/fork-refresh.sh --build --gates --push`, then the container watcher picks the tag up (never build the images on the primary box) | tag pushed, fork `rdna-boosts` tip == `release.json.tip`, images published by the worker |
| 7 | **Refresh the campaign record**: re-export `archive/work/kv-residency/patches/goal-b/` from the final branch, update `SHA256SUMS`/`diffstate`, and shrink `COLD-START-PROMPT.md` to what is left (Goal C + the Goal A `auto` decision) | the patch series reproduces the branch tree; the prompt names only open work |

### 12.2 The `--fit` blocker (task 3, in detail)

`common/fit.cpp` recognises **only** the windowed-KV buffertype — `fit_buft_is_kv_region()` (line ~138) asks the
backend for `ggml_backend_cuda_kv_region_is_kv_buft` and treats a match's `mb.context` bytes as "the KV cache's
whole footprint at this `-c`, of which only the resident window is committed VRAM".  The **pool's mirror buft has
no such predicate**, so with `LLAMA_KV_REGION_POOL` set its bytes are booked as ordinary **VRAM `mb.context`**:
`--fit` reserves ~1.6 GiB (103k) / 2.4 GiB (180k) of VRAM for memory that is really pinned host RAM — it hands
back exactly the arena the pool exists to free, and can refuse a fit that would work.

What to do (mirror block 15's shape end to end):

1. Backend: add `ggml_backend_cuda_kv_pool_is_kv_buft` beside the mirror-buft proc (`ggml-cuda.cu`, near
   `ggml_backend_cuda_kv_pool_mirror_buft`) — true for the pool's mirror buft only.
2. `common/fit.cpp`: extend `fit_buft_is_kv_region()` (or add a sibling) to accept either proc, and make the KV
   bytes it carries out `min(mirror bytes, cells x cell bytes)` — i.e. what the *device pool* actually commits.
   The pool size is `LLAMA_KV_REGION_POOL` cells x cell bytes; when the env is `auto` (if that is added) it is a
   decision like block 15's.
3. `llama-fit-params` must print both numbers (pool VRAM, mirror host RAM) — the host half is already bounded by
   `llama_kv_region_spill_reserve()` at cache creation, so the RAM ceiling story is done; it is the VRAM side that
   is blind.
4. Arms: a config that only fits with the pool (Flash-Next `-c 262144` on this 32 GB card) must be *accepted* with
   `LLAMA_KV_REGION_POOL` set and *refused* without it; and `--fit` must not reduce the arena reserve by the mirror.
5. If this cannot be made to read correctly, **do not ship the pool** — ship task 2 alone and leave the pool in
   `wip/` (the rollback criterion).

### 12.3 Grow-on-demand mirror (task 4)

Today the whole `-c` is pinned at creation (204 MiB / 1.6 GiB / 2.4 GiB at 16k / 131k / 196k; ~4.9 GiB at 262k).
The design's own note calls a grow-on-demand mirror "the obvious next step".  Shipping it as a *documented
limitation* is acceptable (block 15's spill is lazy for the same reason and it shipped with a `LLAMA_KV_SPILL_RAM_PCT`
ceiling); implementing it means mirror mappings grow with the cache watermark exactly as the region's host half
does (`llama_kv_cache`'s `grow_cells`/`grow_for` is the model).  Decide, record, and if deferred add it to
`TODO.md` as an open follow-up rather than a comment.

### 12.4 Packaging the pool as a block (task 5, the pieces)

* **Number**: r190 gave the pool its own block 16; **r191 folds it into block 15** on the maintainer's call
  (the block is the delivery's "the KV does not have to be `O(-c)` of VRAM" work), tree-identical.  Note the
  `beta/0016-copy-2d-vec16/` directory is a *beta apply-order* prefix, not this block number — rename it if that
  reads badly after the fact.
* **Block note** in `patches/README.md`: the mirror-is-authoritative writer rule, the three invariants
  (mark-then-claim, distinct-block claims, no transient scratch in a captured graph), the declines (single stream,
  non-MLA, no window active, uniform geometry, `-sm tensor`, k-quant indexer keys), and the numbers.
* **`ENVIRONMENT.md` §**: `LLAMA_KV_REGION_POOL` (cells; unset/0 = off = byte-identical), `LLAMA_KV_POOL_VERIFY`
  (the oracle; 0 mismatches expected), `LLAMA_KV_POOL_STATS` (**a path**, appended — never stderr on a text arm).
* **`AGENTS.md`**: "16 blocks" → 17 in the naming paragraph, the V2.1 map gains **16 attendance pool**, and the
  default-on policy note ("a fittability feature like block 15 → default OFF, env enables").
* **`MANIFESTS.md`**: apply order 17 commits, the block's verification contract, and its gate rows.
* **Default**: **off** (`LLAMA_KV_REGION_POOL` unset).  It is a fittability feature: on a config whose KV fits it
  costs the sparse-vs-dense gap (7-12 % at depth), so it is opt-in like block 15 — state that plainly in the block
  note so nobody "fixes" the default.
* **`TODO.md` #118** closes to a single `WORKLOG.md` entry (Goal B delivered; Goal C + the `auto` policy remain, as
  their own items).  Do **not** archive `archive/work/kv-residency/` while Goal C is still open — the campaign directory
  stays, shrunk to the open work, and the archived copy happens when the last item closes.

### 12.5 The gate list the block owes on the final base (§6 has the commands)

Text identity on a **resident** arm (pool off vs on) and pool == the resident **sparse** hash, plus the deep
reference at 51 475 depth (`854 43b30de151b9`); `LLAMA_KV_POOL_VERIFY=1` 0 mismatches; prefill-logit **on a
spilling arm** and on the pool arm (`--ab "LLAMA_KV_REGION_POOL=…"`, 0.000000 / 100.000 %); `gate-park-idle-slots.sh`,
`gate-ckpt-reuse.sh`, `gate-state-pipe.sh` (the mirror is meant to make these trivial — that is the writer rule);
`test-backend-ops -o FLASH_ATTN_QSA`; `check-gpu-faults.sh` over every log; the OOM baseline with the knob unset;
`check-windows-paths.sh` + `check-windows-portability.sh` (clean on the two files the block adds —
`kv-pool.cu`/`kv-pool.cuh` already are); and the 180k row from task 0 as the release evidence.

### 12.6 Carried forward (not this session's delivery, but do not lose them)

* **Goal C, the drafter ring** (`O3-DRAFTER-RING.md`): measured, **not** promoted — free at a usable window (46k
  context, `W=32768`: acceptance −2.3 % rel, generation flat), 7-9 % lost far below the working context; the VRAM
  prize (one draft layer, ~1 GiB at `-c 262144`) does not reach the arena.  Three open questions for the maintainer
  before any promotion: default-on or opt-in, one window size or per-model, MTP-only or DFlash2 too.
* **Goal A `auto` policy** (`A2-MIXED-BACKING.md` §3.7.11.4 A-C): the arena-vs-window trade — the maintainer's call,
  measured, no new work needed.
* **Goal A across `-sm tensor`** (`A6-TENSOR-SPLIT-WINDOW.md`): designed only, hardware-blocked on this one-card box.
* **The radix top-k** (2.03 ms/token plain decode, 0.9 ms/step under MTP): the largest sparse-arm item left after
  A13; optional, needs its own campaign.
* **k-quant indexer keys**: accepted limitation (a 256-element superblock does not divide `idx_dim` = 128).
