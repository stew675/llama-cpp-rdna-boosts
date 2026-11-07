# RESULTS 2026-11-06 (`soar`) — the two closes: `TODO` #278 and #276 (and #277 with it)

**Status: measured; both items close, nothing ships.**  This is the second pass of the day and it ran on a
**different box** from [`RESULTS-2026-11-06.md`](RESULTS-2026-11-06.md): that one is `radix` (one R9700, 93 GiB,
the non-canonical prior `a07822d3…`), this one is `soar` (Ryzen 9 9950X3D, **3 × R9700**, 91 GiB, three PCIe 5.0
x16 links, the **canonical** prior).  So **no number here is comparable to that file's** — only within-run
deltas are.  The two questions, and the reason this box could answer them where `radix` could not:

| `TODO` | question | why `soar` |
|---|---|---|
| **#278** | does the slab drop expert-cache arena tables at a **decode→prefill** transition, and what does the re-fill cost? | needs the arena to be *slab-bound*; also runnable at the field ubatch |
| **#276** | is PR #267's copy/drain hazard (a small pinned copy leaving later queued work unsubmitted, stalling the **paired** peer past its all-reduce spin timeout) reachable? | `radix` has one GPU, so no paired all-reduce exists there; this box pairs them |

**Environment of record.**  Tree `~/llama.cpp` @ `5c3ef0354`, tree `622d017f7a1fac093b24b7bb8dc65a8e15de4cb7`
= `release.json` (`v16-6050b467c-r229`), build `5c3ef0354` (2026-11-06).  Heat prior: the **canonical** map
`6195152358e132a50189272102d19ba54b3aec9d012751988b601e8e72651cdd`, 601 376 B, mode 444, present in both
`~/.cache/llama-moe-heatmap/` and the model dir — every arm **seeded** from it and left it untouched
(`heat map … is read-only; seeding from it, not updating it`).  One arm at a time, `HIP_VISIBLE_DEVICES` pinned,
`-lv 4` throughout; `scripts/check-gpu-faults.sh` is clean on the kernel journal and on the five logs quoted here.

---

## 1. `TODO` #278 — the arena IS dropped at the transition, and the re-fill is not a measurable share

### 1.1 What fires, and where

`moe_cache_evict_slab_range` is the design's only eviction (`moe-expert-cache.cu`): when the work pool needs
more chunks the slab reassigns its **lowest** arena chunks to it and every table in that range is dropped, then
`moe_cache_rearm` re-arms what fits after the compute-buffer drop.  It therefore fires only when the work region
grows **past its reservation** — which needs the arena to be slab-bound and the real prefill wider than the
priming pass.

| arm | shape (Flash-Next GSQ-RCO-IQ3_S, 1 GPU, `-ncmoe 48 -lme 44 -ctk/-ctv q8_0 -lm mmap`) | evictions | evicted | re-arms | pp rows, t/s |
|---|---|---|---|---|---|
| E1 | `-c 16384 -ub 2048 -npp 512,1024,2048,4096` | 5 | 620 MiB | 3 × 3 tables | 2.68\* / 1007.8 / 1572.6 / **1578.9** |
| E2 | field ubatch `-c 204800 -ub 6144 -npp 2048,8192` | 3 | 1062 MiB | 1 × **12 tables** | 95.4\* / 1704.2 |
| E3 | field ubatch, `-npp 4096,4096,4096` | 2 | 184 MiB | 2 × 3 tables | 154.0\* / 1406.9 / **1629.0** |
| N1 | **3 GPU `-sm tensor`**, `-c 16384 -ub 2048 -npp 1024,2048,4096` | **0** | — | — | 16.3\* / 2084.1 / 2207.5 |
| N2 | 35B-A3B **Q4_K_M**, 1 GPU (`-ncmoe 40` *and* `-ncmoe 48`; arena auto *and* `MOE_EXPERT_CACHE_MIB=26000`) | **0** | — | — | — |
| **E4** | **35B-A3B Q8_0**, 1 GPU, `-ncmoe 40 -lme 12 -c 16384 -ub 2048 -npp 4096,4096,4096` | **2** | **380 MiB** | 2 × 3 tables | 565.6\* / 3579.3 / **3649.2** |

`*` the first row of every arm is the pool build/prepopulate (a one-off 45 GiB read for the Flash-Next arms),
never a cache number.

**E4 is the cheap reproduction** (Q8_0 loads in a fraction of Flash-Next's 84 GB and runs at 3.6 k t/s): 2 events,
380 MiB, `re-armed 3 stood-down` twice, arena 22 695.0 MiB inside a 25.81 GiB region (slab 28.00 GiB, work
2.19 GiB), and the post-transition row is again **+2.0 %** over its predecessor.  The eviction is driven by the
*work region outgrowing its reservation*, not by the arena filling the slab — which is why Q4_K_M, whose priming
pass already saw the full `-ub` prefill and so reserved enough, never evicts (N2).

### 1.2 The cost, bounded two ways

* **By the bytes lost.**  A dropped table loses its slots, and the next prefill re-fills at most that many bytes
  from the pool/mapping.  At 19-28.7 GB/s (28.7 GB/s is this build's own H2D calibration line; 19 GB/s is the
  per-device pool-read figure `TODO` #131 measured): **184 MiB ≈ 6-10 ms** (E3) and **1062 MiB ≈ 37-56 ms** (E2).
  Against the prompts those rows ran: **≤ 0.4 %** of the 4 096-token row and **≤ 1.2 %** of the 8 192-token row,
  **≤ 0.2 %** of the campaign's 50k row.
* **By the paired rows.**  E3 runs three identical 4 096-token rows; row 3 is preceded by a transition that
  evicted 184 MiB and re-armed 3 tables twice, and it measured **+15.8 %** over row 2 (1 629.0 vs 1 406.9 t/s).
  E1's row 4 (1 578.9) is +0.4 % over row 3 (1 572.6) with an extra eviction in between.  The warm-up gradient
  dominates; there is no transition penalty to find.

### 1.3 Why the cheaper Q4_K_M arm cannot reproduce it (a trap worth recording)

35B-A3B **Q4_K_M** does not evict on a 32 GiB card in any configuration tried: its whole host expert set is only
**18 662 MiB** (120 tables x 256 experts, `-ncmoe 40` already hosts every MoE layer — `-ncmoe 48` is the same set),
`alloc_table_locked` **clamps slots at `n_experts`** so no `MOE_EXPERT_CACHE_MIB` can force the arena past it, and
its priming pass already sees the full `-ub 2048` prefill so the work region is reserved at the size the real
prefills need (`work 8.62 GiB + arena 19.19 GiB` at `-ub 8192`).  **Zero evictions in every Q4_K_M arm.**  The
cheap reproduction is **Q8_0** of the same model (E4: 380 MiB, 2 events) — the set is ~33 GiB, the arena lands at
22 695.0 MiB and the work region does outgrow its reservation.  The cost is the same nothing: 380 MiB is 13-20 ms
at 19-28.7 GB/s, <=1.4 % of a 4 096-token prompt, and measured as **+2.0 %** (faster) across the transition.

### 1.4 Verdict

**`TODO` #278 closes: measured, condition not met.**  P1.4's own do-it-if is "the field census shows arena tables
actually dropped at the transition **and** the re-fill is a measurable share of the prompt".  The first half is
confirmed (12 tables / 1062 MiB at the field ubatch); the second fails (≤ 0.2-1.2 %, invisible in paired rows),
and on the 3-GPU `-sm tensor` shape it does not fire at all.  **Do not implement PR 0003b.**

**A campaign claim is corrected with it.**  r228's frame (`PR-267-REMAINDER.md` §3.0 item 2) ranked 0003b as
"now the highest-value prefill item in this list" and "the field prefill's main residual".  It is not: the field
residual at `auto` (97.7 %) is the **near-mirror miss path** (r228's own 1 837 vs 1 932-1 938 t/s), which
`SINK=0` already takes, and the transition contributes ≤ 1 % of it.

**Re-entry condition.**  A shape where the arena is *exactly* slab-full **and** the prompt is short (≤ 4k), so
that ~1 GiB of re-fill is > 5 % of the prompt — e.g. the 56 GiB rig (`wip/bench-56gb/`) at `-c 204800 -ub 6144`
with `-lme 44`, measured directly rather than inherited.

---

## 2. `TODO` #276 — the hazard is unreachable: the demote never runs beside a paired all-reduce

PR #267's 0004/0006/0007 rationale is that a **small pinned copy** can leave later queued work unsubmitted on
ROCm, and with the all-reduce pairing two devices that stalls the peer past its spin timeout.  That needs both a
copy **and** a paired all-reduce.  In the delivery they are mutually exclusive:

| 2 GPU, 35B-A3B Q4_K_M, `-ncmoe 40 -lme 12 MOE_EXPERT_CACHE_MIB=4096 -c 16384 -ub 2048` | all-reduce in the log | demoted | declines |
|---|---|---|---|
| `-sm tensor` | **`all-reduce: host-resident experts -- large tensors use the RCCL ring`** | **0** | **14 480 `geometry (structural)`** |
| `-sm layer` | none | **7 862** (4 771.4 MiB) | 0 geometry |

The decline is structural, not a policy: `moe_cache_evict_slab_range`'s sibling in the fill path declines when
`t.n_slices != 0` — a tensor-split table has no single contiguous pool image (`g_sink_decl_geom`), because the
D2H would be a strided 2-D copy, which r113 already measured as a **2× decode loss**.  So on the only layout with
an intra-token all-reduce the demote issues **no copy at all**, and on the layout where it does fire (`-sm layer`,
7 862 copies) there is no all-reduce to poison.  The cost side is separately answered and net-positive: the
demote is worth **+19-21 %** generation at `-lme 44` (r228's frame), so 0004's copy-kernel is a transport
optimisation with no reachable defect to fix.

**Verdict: `TODO` #276 closes as unreachable by construction.**  Re-entry: the demote would have to be extended to
an `n_slices != 0` table — a strided 2-D D2H into the pool, against r113's measurement.

**`TODO` #277 (precopy) retires with it**, by its own condition ("only if #276 leaves a residual on a
multi-device arm").  The condition fails, and its shape was already decided to need a dedicated K-spare ring
rather than more headroom (`PR-267-REMAINDER.md` §2).

---

## 3. Exact reproduction

Nothing here needs a rebuild, a patch, or the 56 GiB rig.  `soar` = 3 × R9700 (gfx1201); keep
`HIP_VISIBLE_DEVICES` pinned on every arm and `-lv 4` on (all quoted counters are `INFO`).  One arm at a time,
never in parallel.  Logs were not kept; these commands regenerate them.

**#278, the transition that fires (E2 — 3 evictions, 1062 MiB, 12 tables re-armed):**

```bash
cd ~/llama.cpp && HIP_VISIBLE_DEVICES=0 MOE_HOST_POOL_FRAC=80 MOE_HOST_POOL_RAM_PCT=0 \
  timeout 2400 ./build-rocm/bin/llama-batched-bench \
    -m /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    -ngl 999 -ncmoe 48 -lme 44 -fa 1 -ctk q8_0 -ctv q8_0 -lm mmap \
    -c 204800 -b 6144 -ub 6144 -npp 2048,8192 -ntg 32 -npl 1 -t 8 -lv 4 2>&1 | tee /tmp/278.log

grep -c  "evicted .* arena tables"                  /tmp/278.log   # expect 3
grep -o  "evicted [0-9.]* MiB"               /tmp/278.log | awk '{s+=$2} END {print s" MiB"}'   # 1062
grep -o  "re-armed [0-9]* stood-down"        /tmp/278.log | sort | uniq -c                     # 1x12 tables
grep -o  "arena now [0-9.]* MiB"             /tmp/278.log | uniq -c                            # the oscillation
grep -E  "^\|\s+[0-9]+\s+\|"                 /tmp/278.log                                     # pp rows
```

Replace `-npp 2048,8192` with `-npp 4096,4096,4096` (E3) to isolate the cost: three identical rows, the third
preceded by a transition, and it measures **faster** than the second.

Controls: the same command with **`-sm tensor` at 3 devices** (`HIP_VISIBLE_DEVICES=0,1,2`) is **0 evictions**;
**35B-A3B Q4_K_M at 1 GPU** is 0 evictions for every `MOE_EXPERT_CACHE_MIB` / `-ncmoe` / `-ub` tried (§1.3).

**The cheap reproduction (E4, 35B-A3B Q8_0 — no 84 GB model, 3.6 k t/s):**

```bash
cd ~/llama.cpp && HIP_VISIBLE_DEVICES=0 MOE_HOST_POOL_FRAC=80 MOE_HOST_POOL_RAM_PCT=0 \
  timeout 1800 ./build-rocm/bin/llama-batched-bench \
    -m /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf \
    -ngl 99 -ncmoe 40 -lme 12 -fa 1 -c 16384 -b 2048 -ub 2048 \
    -npp 4096,4096,4096 -ntg 32 -npl 1 -t 8 -lv 4 2>&1 | tee /tmp/278-q8.log

grep -c "evicted .* arena tables" /tmp/278-q8.log          # expect 2
grep -o "evicted [0-9.]* MiB" /tmp/278-q8.log | awk '{s+=$2} END {print s" MiB"}'   # 380
grep -oE "re-armed [0-9]+ stood-down" /tmp/278-q8.log | sort | uniq -c              # 2x 3 tables
grep -E "^\|\s+[0-9]+\s+\|" /tmp/278-q8.log                                      # row 3 >= row 2
```

Reference: slab 28.00 GiB (`work 2.19 GiB + arena 25.81 GiB`), arena ends at 22 695.0 MiB, rows
565.6 / 3579.3 / 3649.2 t/s.  The measured control is the same shape on **Q4_K_M** (N2), where the count is **0**.

**#276, the two layouts:**

```bash
cd ~/llama.cpp && M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

# tensor split -- paired all-reduce, demote declines 100 %
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=4096 MOE_HOST_POOL_FRAC=80 MOE_HOST_POOL_RAM_PCT=0 \
  timeout 1800 ./build-rocm/bin/llama-batched-bench -m $M -ngl 99 -ncmoe 40 -lme 12 -sm tensor \
    -fa 1 -c 16384 -b 2048 -ub 2048 -npp 1024,2048 -ntg 32 -npl 1 -t 8 -lv 4 2>&1 | tee /tmp/276-ts.log

# layer split -- no all-reduce, demote fires
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=4096 MOE_HOST_POOL_FRAC=80 MOE_HOST_POOL_RAM_PCT=0 \
  timeout 1800 ./build-rocm/bin/llama-batched-bench -m $M -ngl 99 -ncmoe 40 -lme 12 -sm layer \
    -fa 1 -c 16384 -b 2048 -ub 2048 -npp 1024,2048 -ntg 32 -npl 1 -t 8 -lv 4 2>&1 | tee /tmp/276-ly.log

grep -E "all-reduce:|sink \(MOE_HOST_POOL_SINK\)" /tmp/276-ts.log   # RCCL banner; 0 demoted, 14480 geometry
grep -E "all-reduce:|sink \(MOE_HOST_POOL_SINK\)" /tmp/276-ly.log   # no banner;   7862 demoted, 0 geometry
```

Then, as for every arm here:

```bash
cd ~/llama-cpp-rdna-boosts && ./scripts/check-gpu-faults.sh --since "90 min ago" --log /tmp/278.log --log /tmp/276-ts.log --log /tmp/276-ly.log
```

`-lme 12` (61 % coverage) is what makes the demote **want** to fire, so the tensor arm's `14480 geometry
(structural)` declines are offers that were actually made — not an idle switch.  Confirm the prior is the
canonical one before quoting any number:
`sha256sum ~/.cache/llama-moe-heatmap/*.moe-heatmap` → `6195152358e1…`.
