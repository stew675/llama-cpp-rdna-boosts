# IMPLEMENTATION PLAN — the unified host-expert tier (execute without waiting for the maintainer)

> **EXECUTED 2026-11-04 — see [`P5-RESULTS.md`](P5-RESULTS.md).**  Steps **a, b, c, d, e** are implemented
> on `~/llama.cpp` branch `p5-host-expert-tier` (`72bc81b88`, on top of `c1478f911`, parent `a644115a0`), patch
> [`patches/0010-p5-host-expert-tier.patch`](patches/0010-p5-host-expert-tier.patch), and the §3/§4
> acceptance bars are met: the fit case reproduces `pinned` (1105.4 / 71.9 at 2 devices, 645.2 / 53.4 at 1)
> with byte-identical text and no GPU faults, and the not-fitting band is unchanged (713.7 / 51.8).  Two
> amendments to the plan as written:
>
> * **Step a's per-expert region is not needed for a whole-tensor budget.**  The master is expert-indexed by
>   stride, so a wholly pinned tensor already IS the region; `host_pinned` becomes a per-table truth and no
>   presence bitmap / hole encoding is involved.  The bitmap belongs to a per-EXPERT region, i.e. step b2.
> * **Step b2 is DONE, as the pool** ([`P6-POOL-REGION.md`](P6-POOL-REGION.md)): at equal RAM a partial budget is
>   better spent per-EXPERT (the pool) than per-tensor — 8 064 MiB pinned as whole tensors is 505.1 prefill /
>   50.1 decode against 713.7 / 51.8 for 9 331 MiB of pool.  The expert-indexed region the plan asks for already
>   exists (it is the pool: slots + `slot_state` + `base_dev`, heat-ranked); it only needed the right to be a
>   cold-read source on a multi-device split, which is now enabled (`MOE_HOST_POOL_COLD_MULTIDEV`, +11.5 %
>   decode at `-lme 8`, byte-identical).  The cold-read encoding needed **no** change.  `-lme N` therefore
>   spends the budget per expert below 90 % of the host expert set and keeps the whole-tensor pinned-master tier
>   at or above it (the fit case, where a pinned *master* wins decode outright).  Step c's PARTIAL state
>   collapsed with it (a table is ALL_RESIDENT or has holes, which is already per table).
>
> Two traps found while executing, both silent (P5 §4): the `-ncmoe` tail bound must be
> `hparams.n_layer()` — not `n_layer_all`, which shifts the tail by one and plans a pinned MTP head — and
> `moe_cache_set_host_pool()` must re-resolve `g_pool_enabled`, because the loader decides after
> `parse_env()` (without it the not-fitting band silently lost the pool: 37.4 against 51.8 t/s).
> **The promotion itself (§5/§6, the delivery docs and the block cut) is NOT done** — `AGENTS.md` requires
> the findings to be reported and a decision taken first.

**Audience: a fresh session with no memory of the campaign.**  Everything you need is here: the decision, the
measured targets, what to delete, what to keep, the code shape, the exact tests, and the traps.  The maintainer
has reviewed and agreed to every decision marked **[DECIDED]**; do not re-litigate those, and do not stop to ask
unless a **[STOP]** marker says so.  Where a choice is left open it is marked **[YOUR CALL]** with a default.

Read [`P4-ROOT-CAUSE.md`](P4-ROOT-CAUSE.md) for the evidence behind each number quoted here, and `AGENTS.md` at
the repo root for the delivery's policy (gates, docs, pushing).  Do **not** re-derive the measurements.

---

## 0. State of the world

| what | where |
|---|---|
| the fork under test | `~/llama.cpp`, branch **`p2-eager-push-r209`**, HEAD **`a644115a0`**, built with `~/bin/build-llama-rocm` (`build-rocm`, ROCm 10.1.0) |
| the release tip the branch is rebased onto | `27271aa91` (`release.json.tip`, base `6050b467c`) |
| this campaign's patches | `~/llama-cpp-rdna-boosts/wip/host-expert-eager-push/patches/0001..0009` (see §2 for what survives) |
| the campaign's records | `P4-ROOT-CAUSE.md` (findings), `HANDOVER.md` (older), `P3-MULTIGPU-RESULTS.md`, `runs/` (every log) |
| build | `cd ~/llama.cpp && CMAKE_BUILD_PARALLEL_LEVEL=16 cmake --build build-rocm --target llama-cli -j 16` (~40 s incremental) |
| hardware | 3× R9700 (gfx1201, 32 GiB each): **devices 0 and 2 are PCIe x8, device 1 is x4** |

**[STOP]** if `~/llama.cpp` HEAD is not `a644115a0` or the tree is dirty: the nine campaign patches replay in
order onto a clean `27271aa91` (`git am`), so rebuild the branch first rather than building on unknown state.

### 0.1 First 15 minutes: confirm the box is in the same state

Do this before touching code; if any line does not match, stop and report rather than working from a different
baseline.

```bash
cd ~/llama.cpp && git rev-parse --short HEAD          # expect a644115a0 on p2-eager-push-r209
CMAKE_BUILD_PARALLEL_LEVEL=16 cmake --build build-rocm --target llama-cli -j 16   # ~40 s, must be clean
cd ~/llama-cpp-rdna-boosts/wip/host-expert-eager-push
P=~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt

# the two reference arms (serial!), 2 devices, both PCIe x8:
GPU=0,2 SM=tensor EOS=1 N=300 ./p3-matrix.sh ref-pinned              # expect ~72.0 t/s decode
GPU=0,2 SM=tensor N=1 PROMPT_FILE=$P ./p3-matrix.sh ref-pinned-pre   # expect ~1105 t/s prefill
GPU=0,2 SM=tensor EOS=1 N=300 ./p3-matrix.sh ref-pool                # expect ~51 t/s decode
```
(As of `a644115a0`, `p3-matrix.sh` still passes `--host-experts pool` itself, so it prints a deprecation
warning — expected, and step (d) fixes it.  It also still needs `LME=`/`LAZY=` for the new knobs.)

`pinned` ≈ 72 / 1105 and `pool` ≈ 51 decode ⇒ the box matches the record and you can start at step (a).

---

## 1. The decision being implemented **[DECIDED]**

Today the delivery has **two modes** that differ in *where the host-expert master lives*:

- `--host-experts pinned`: the master is a per-tensor `ROCm_Host` (`cudaMallocHost`) buffer → `t.host_pinned =
  true` → single-launch prefill staging, in-place cold read available.
- `--host-experts pool`: the master is the **pageable model mapping**, plus a bounded pinned **slot pool** (a
  *permutation* of whole experts) → `t.host_pinned = false` → pool-backed staging (two-tier snapshot rebuilt
  per op, per-expert copies), no cold read.

Measured, with the **same full 18.6 GiB pool (30 720 slots)** in both arms, 2 devices (0,2):

| arm | prefill (prose, 5 246 tok) | decode (`-n 300`) |
|---|---|---|
| pinned master, pool present (`LLAMA_HOST_MASTER_PINNED=1`) | **1 105.3** | **71.9** |
| pageable master + full pool (`-lme 64`) | 532.9 | 22.6 |
| full pool + pinned master (`-lme 64` + `LLAMA_HOST_MASTER_PINNED=1`) | 1 009.4 | 71.9 |

So a fully pinned **slot set** does not behave like a pinned **master**; the mode-keyed switch is the 3×, not the
transport.  And a pool *beside* a pinned master costs ~9 % prefill for nothing (1 009.4 vs 1 105.4).

**Target: one pinned host region, addressed by expert index, with a presence bitmap.**

1. A process-wide pinned region (one `hipHostMalloc` per host tensor, or one arena — **[YOUR CALL]**, default:
   per host tensor, allocated whole tensor at a time, because that is how the loader already works).
2. **Filled greedily at load in descending hotness**, after VRAM has been filled with the hottest weights: put
   each host expert into the region while `-lme` allows, otherwise leave it in the pageable model mapping.
3. **`-lme auto`** = `MOE_HOST_POOL_FRAC` (80 %) of `MemAvailable` measured at the first host-expert tensor;
   **`-lme N`** = N GiB (clamped to the host expert set, refused above 90 % of available memory — already
   implemented).
4. **Releasable** with `hipHostFree` (proven: 18 329 / 18 376 MiB returned for an 18.33 GiB region; the kernel
   immediately reused it as page cache).
5. **The pageable mapping is the per-expert fallback through the holes**, so `t.host_pinned` becomes a per-table
   truth about *the weights* and the two staging branches collapse into one hole-aware path.
6. **Census** reports the tiers: `pinned X GiB of Y GiB (cap C GiB), pageable the rest`.

**Acceptance target (the reason for all of this):** a host expert set that fits must hit the `pinned` numbers —
**1 105.4 prefill / 72.0 decode at 2 devices; 645.9 / 53.5 at 1 device** — while a set that does not fit keeps
the pinned numbers *for the resident part* and pays the pageable rate only for the overflow
(~532 / 22.6 at 2 devices).

---

## 2. What to keep, what to drop

**Keep and fold into the delivery** (these are fixes, not experiments):

| patch | why it stays | action |
|---|---|---|
| `0005` once-only diagnostics | the P1-PUSH messages landed inside the generated text and corrupted two hashes (P3 §1b) | keep; it becomes part of the diagnostics block |
| `0008` loud fallbacks + degenerate-split warning | `PINNED HOST ALLOCATION FAILED → pageable` and `DEGENERATE TENSOR SPLIT`; both verified firing and silent where correct. This is the pair of diagnostics that would have saved this campaign | keep verbatim |
| `0009` `-lme/--limit-host-experts` | the new single knob; verified (90 % refusal, 8 GiB pool, clamp to 18 662 MiB, deprecation warnings) | keep, **but re-target** (§3 step b): `N` must cap *placement*, not the pool budget |

**Drop from the deliverable** (keep the patch files in `wip/` as the historical record — do **not** delete them
from the repo):

| patch | why it dies |
|---|---|
| `0001` `MOE_HOST_PUSH_SRC=mapping` (pageable push) | superseded: the pageable source is the *slow* path (141 µs/copy against 0.3–0.5 for a pinned/registered source) |
| `0002` mlock working set | superseded: `hipHostMalloc` *is* the residency tier now; `mlock` over a mapping keeps pages but does not make them a DMA source |
| `0003` heat-seeded mlock set | superseded: heat ordering moves into the resident-region fill (§3 step b) |
| `0004` rotation + device-set epoch gate | the mlock-specific half dies; **the epoch-gate idea stays available** if the region ever needs to release while copies are in flight — copy it forward only if step (d) needs it |
| `0006` attribution probe | diagnostic only; keep in `wip/` |
| `0007` registered source (`hipHostRegister`) | **dead**: registration is 26 %/14 % behind `hipHostMalloc` on prefill and cannot be released while the mapping lives (§5a).  Do not use it anywhere |

**Do not touch**: `ggml-cuda-slab.cu`, the VMM/KV machinery, the split code in `src/llama-model.cpp`.  The
mirrored-split behaviour is **documented and warned**, not fixed (§5g/§5h).

---

## 3. The work, in order

Each step: implement → build → test → commit on the campaign branch → export the patch.  **Do not batch steps**;
each has its own acceptance bar, and a step that fails its bar must be reported (see §7) rather than papered over.

### Step a — the expert-indexed region exists and a full region behaves like `pinned`

Files: `ggml/src/ggml-cuda/moe-expert-cache.cu` (primary), optionally a new
`ggml/src/ggml-cuda/moe-host-region.{h,cu}` if the code grows past ~300 lines.

Shape:
- A per-**host tensor** structure: `{ char * base; size_t expert_bytes; size_t host_bytes; int n_experts;
  std::vector<uint8_t> present; size_t bytes_pinned; }` — expert `e` lives at `base + e * host_bytes` (the same
  stride as the master, so a *full* region is byte-for-byte a drop-in master and
  `cudaHostGetDevicePointer(base)` succeeds).
- Allocate `n_experts * host_bytes` with `cudaMallocHost` and `memcpy` the whole tensor from the model mapping
  (that is what `pinned` mode already does per tensor; measured 1.9 s for 18.6 GiB at ~9.8 GB/s).
- Register it in the existing bookkeeping so `expert_host_src()` can serve from it, and make the table's
  `host_pinned`/`host_dev` reflect it (the existing `bind_host_dev_locked` already does this if `t.host` points
  at the region).

**Acceptance**: with the region covering the whole host expert set, a 2-device run reproduces the pinned numbers
within noise — prefill ≥ 1 090, decode ≥ 71.0 — and the census says `pinned 18 662 MiB of 18 662 MiB`.  If it
does not, the region is not being used as the master: check `P1-PUSH: fill source: master=pinned/registered ...`
(the `0005`/`0008` stderr diagnostics) before going further.

### Step b — greedy load-time fill in hotness order, capped by `-lme`

Files: `src/llama-model-loader.cpp` (`host_expert_buft()`, the per-tensor placement decision),
`common/common.cpp` (cap plumbing), `common/arg.cpp` (already parses `-lme`).

Shape:
- The cap: `auto` → `MOE_HOST_POOL_FRAC` (default 80) % of `MemAvailable` read once at the first host-expert
  tensor; `N` → `N` GiB (already clamped/refused in the option parser).
- Running counter `pinned_bytes`; for each host-expert tensor in **descending hotness** (the heat map from
  `-ncmoe` layer ordering; reuse the existing heat-prior machinery if present, otherwise the existing
  `moe_cache_heat` table): if `pinned_bytes + nbytes <= cap` → place it in the region (step a) and bump the
  counter; else leave it in the model mapping.
- **[DECIDED]** whole-tensor granularity in this step.  Partial (per-expert) residency is step (b2) and is only
  attempted if step (c) shows it is needed.
- Log once, on stderr: `limit-host-experts: pinned X GiB of Y GiB of host experts (cap C GiB); the remaining
  Z GiB is pageable`.
- **Re-target `-lme N`**: it must set *this cap*, not `MOE_HOST_POOL_MIB` (patch `0009` currently does the
  latter — change it here and re-verify).

**Acceptance**: three runs, 2 devices, `-n 300` decode and the prose prefill:
| `-lme` | expected |
|---|---|
| `auto` on 35B-A3B (18.6 GiB set, box has ~82 GiB available) | pins everything → 1 105 / 72, census `X == Y` |
| `8` | pins ~8 GiB in hotness order → decode **must exceed 22.6** (the full-pool figure) and land between the pageable and pinned rates; report the actual |
| `64` | clamped to 18 662 MiB → identical to `auto` |

### Step c — hole-aware staging: one branch, `host_pinned` per table

Files: `moe-expert-cache.cu` (`moe_cache_stage_table`, `expert_host_src`, the table's
`host_pinned`/`cold_safe` derivation).

Shape:
- A per-table state: `ALL_RESIDENT | PARTIAL | NONE`.  `ALL_RESIDENT` takes the existing single-launch pinned
  branch unchanged.  `NONE` keeps today's pool/pageable behaviour.  `PARTIAL` is the new case: resident experts
  come from the region (no host memcpy), holes from the mapping/bounce — reuse the two-tier snapshot kernel,
  with the region as an additional tier.
- **[STOP]** if making `PARTIAL` correct requires changing the cold-read encoding
  (`id - n_res_slots → pool slot`): that encoding is shared with the KV/arena paths.  In that case implement
  `PARTIAL` as "region tier + existing pool tier for holes", leave the cold encoding alone, and record the
  limitation.

**Acceptance**: a `-lme 8` run must show the resident experts served from the region (add a one-line counter
`region fills / pool fills / bounce fills`) and must not regress `ALL_RESIDENT` (`-lme auto`) numbers.

### Step d — retire the mode distinction

Files: `common/arg.cpp`, `common/common.h`, `src/llama-model-loader.cpp`, `src/llama-model.cpp`,
`ggml/src/ggml-cuda/moe-expert-cache.cu`.

- `-lme` is the only knob.  `--host-experts pinned|pool` stay as deprecated shorthands (already warning):
  `pinned` = cap = the whole set, `pool` = `auto`.
- Remove `MOE_HOST_POOL_MIB`/`_FRAC`/`_RAM_PCT` from the *user-facing* path if `-lme` supersedes them — **but
  keep the env vars working** (they are documented in `ENVIRONMENT.md`) and give the CLI precedence.
- Delete the now-dead code paths (`LLAMA_HOST_EXPERTS_MODE_POOL` as a mode, the slot-permutation pool *only if*
  step (c) removed its last user).

**Acceptance**: `grep -rn "LLAMA_HOST_EXPERTS_MODE_POOL" src ggml common` shows only the deprecation mapping;
`--host-experts pool` and `-lme auto` produce the same numbers; `scripts/validate-set.sh` unused-env and
build warnings clean.

### Step e — the census line, then the gates

- One line, always at INFO: `host experts: pinned X MiB of Y MiB (cap C MiB), pageable Z MiB; region
  (fills/all/part) ...`.  Keep the `0008` warnings.
- Then §5.

---

## 4. The tests, with expected numbers

All runs: `HIP_VISIBLE_DEVICES` pinned, `ROCBLAS_USE_HIPBLASLT=0`, serial (never two arms at once),
`--seed 42 --temp 0`, and the campaign harness
`wip/host-expert-eager-push/p3-matrix.sh` (`GPU=`, `SM=`, `N=`, `EOS=`, `NCMOE=`, `CACHE_MIB=`, `LME=`, `LAZY=`,
`PROMPT_FILE=`, `MODEL=`, `LV=`).

Model paths (resolve against `/llm/models/`):

| model | path | notes |
|---|---|---|
| the primary | `Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 48 layers, 256 experts, `-ncmoe 40` puts the expert set (18.6 GiB) on the host |
| large MoE | `Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | **always `--lazy-mode off`** (`LAZY=off`) or the PLE weights stay pageable and confound everything; ~21 GiB of PLE lives on the host, which is why it fits 2 GPUs |
| dense | `Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf`, `Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` | split-scaling references only |

Prompt: `prompts/prose-rdna-boosts.txt`, 5 246 tokens, `sha256 fabdec65f5859e55…` (see `prompts/README.md`).

**The gate matrix per step**: 1 GPU (`GPU=0`) and 2 GPUs (`GPU=0,2`) — never `0,1` (x4) for a pair, never the
3-device set for a scaling claim.  Decode `EOS=1 N=300`; prefill `N=1 PROMPT_FILE=<prose>`.

Reference numbers (the bar for the fit case, all measured this session):

| | prefill 1 dev | decode 1 dev | prefill 2 dev | decode 2 dev |
|---|---|---|---|---|
| `pinned` (masters pinned) | 645.9 | 53.5 | **1 105.4** | **72.0** |
| pool, pageable master (today's `pool`) | 616.6 | 49.0 | 707.1 | 51.3 |
| pageable master + full pool | 826.7 | 39.8 | 532.9 | 22.6 |

**Correctness gates** (every step): same-seed greedy text must be **byte-identical** between the arms at the
same device count (`scripts/extract-generated.py`, which needs stdout/stderr separated — the harness does this),
and `scripts/check-gpu-faults.sh --since "10 min ago" --log <log>` clean.

---

## 5. Before promoting into the delivery (all mandatory, per `AGENTS.md`)

1. `scripts/validate-set.sh` green after regenerating the block set (`scripts/make-patches.sh`, then
   `scripts/make-release.sh`).
2. Coherence gate at 1 and 2 devices, byte-identical text.
3. `scripts/gate-per-block-build.sh --changed-from <prev tip>` (a clean build per block commit).
4. `scripts/check-gpu-faults.sh` clean on every arm you report.
5. **Docs**: `ENVIRONMENT.md` (any env var added or repointed, with default + class), the CLI reference for
   `-lme`, `patches/README.md` (the block that now owns the host-expert tier), a `TODO` entry for anything
   deferred (partial residency if it is deferred; the mirrored-split fallback fix), and a **newest-first
   `WORKLOG.md` entry** dated the real day with `YYMMDD.VVV` and the `rNN` of the new revision.
6. `INVARIANTS.md`: append the facts this session established, so nobody re-derives them:
   - `hipHostFree` returns the pages to the kernel (18 329 / 18 376 MiB back for an 18.33 GiB region, no
     accumulation, and the kernel immediately reused them as page cache), while **`hipHostUnregister` does
     not** release the pin while the mapping lives, and `hipFreeHost` rejects a registered range;
   - a pageable-source H2D copy costs ~141 us per 590 KiB fill against 0.3-0.5 us from a pinned source -- the
     staging is per CALL, so a tensor split (half the bytes, twice the calls) pays it twice per byte;
   - the prefill gap between a pinned and a pageable *master* (1 105.4 / 71.9 against 532.9 / 22.6 at two
     devices) is the staging branch that `t.host_pinned` selects, not the transport and not the byte count;
   - a host expert tier costs exactly its budget in RAM (18 476-18 651 MiB for an 18 662 MiB tier), not double;
   - the meta tensor split is equal-division only, rounded to the quant block, so a split is expressible iff
     `width / quant_block >= n_devices`; otherwise a device gets a zero extent and is handed the WHOLE tensor
     (`DEGENERATE TENSOR SPLIT`), e.g. 35B-A3B's 512-wide expert rows with 256-element K-quant blocks at 3
     devices (~20 t/s for every transport, worse than one GPU).
7. **Report the findings to the maintainer before the delivery commit** (`AGENTS.md` mandates this): a short
   summary of what changed, the numbers, and anything you could not make work.  The maintainer's sleep is the
   only reason this plan exists — do not treat that as permission to skip the report.

---

## 6. Traps that cost this campaign real time

1. **`HOSTMODE=pinned` must be set on the *script invocation line*** (`HOSTMODE=pinned ./p3-matrix.sh ...`), not
   after the tag: an env assignment in `"$@"` reaches `env` but not the script's own `--host-experts "$HOSTMODE"`,
   which silently made **every** early "pinned" run a pool run.  That single bug hid the pinned mode for the
   whole campaign.
2. **`GGML_LOG_WARN`/`INFO` is FILTERED at llama-cli's default log level.**  A user-facing warning must use
   `fprintf(stderr, ...)` (that is why `0008` does).  Use `LV=4` to see the census.
3. **A harness that prints a summary hides the run's own failure**: a `symbol lookup error` in one arm looked
   like "the warning never fires" for several runs.  Always check the log body, not just the summary.
4. **Device 1 is x4.**  Use `0,2` for pairs.  A 3-device arm is never a scaling point.
5. **3-GPU `-sm tensor` on K-quant MoE mirrors an expert set** (`DEGENERATE TENSOR SPLIT`): 35B-A3B expert rows
   are 512 elements against a 256-element block = **two** aligned chunks, so a third device gets a zero extent
   and is handed the whole tensor.  Model-specific; Flash-Next IQ3_XXS splits fine (five chunks).  Do not chase
   it — it is documented (`P4` §5g/§5h) and warned.
6. **`--lazy-mode off` for Flash-Next**, or the 21 GiB PLE weights stay lazy and every number moves.
7. **Never `git apply` the block series**; `scripts/apply-all.sh` only.  Never push to upstream.
8. **`MOE_HOST_POOL_MIB` currently wins over FRAC** (`0009` sets it from `-lme N`); do not leave two sources of
   truth after step (d).
9. **A pageable host buffer is not device-readable** — if the pinned allocation fails you silently get the pool
   path.  That is what `0008`'s warning exists to catch; if you see it, stop and fix memory, do not measure.

---

## 7. If a step cannot meet its bar

Do **not** relax the bar and do not leave a half-change in the tree.  Instead:

1. Revert the step (`git checkout -- <files>` or `git reset --hard <last good commit>`), keeping the patch file
   you exported for the record.
2. Write the finding into `P4-ROOT-CAUSE.md` (what was tried, the numbers, what the evidence says).
3. Continue with the *next independent* step if one exists (steps a–b are independent of c; d/e depend on a–c).
4. Leave the tree in a state where the last good commit builds and its tests pass, and say plainly in your final
   report which steps are done, which are not, and what the blocker was.

**Definition of done for this plan**: steps a–e implemented, §4's matrix run at 1 and 2 devices for `-lme auto`,
`-lme 8` and `-lme 64`, the fit case within 2 % of the pinned reference (1 105 / 72), byte-identical text, no GPU
faults, `validate-set.sh` green, docs and `WORKLOG`/`INVARIANTS` updated, patches `0001`–`0004`, `0006`, `0007`
absent from the deliverable (still present in `wip/` as history), and the findings reported before the delivery
commit.
