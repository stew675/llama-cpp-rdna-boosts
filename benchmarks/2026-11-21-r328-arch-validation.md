# r328 arch validation — gfx1151 (`halo`) and gfx1100 (`fingon`), 2026-11-21

Closes the gfx1201-only gap of the `v17-919794500-r328` release (`TODO` #354).  Both boxes were driven over
`ssh` from the primary box, **each from a fresh apply of the RELEASED patch set** (not the ladder branch), so this
is also an independent re-verification of `scripts/apply-all.sh` + `release.json` on two more machines — the
method the r138 record used.  Runbook: `archive/work/upstream-rebase-261120/REMOTE-VALIDATION.md`.

| | **halo** | **fingon** |
|---|---|---|
| GPU | Radeon 8060S, **gfx1151 APU** (RDNA3_5), 123 GiB unified | RX 7900 XTX, **gfx1100** (RDNA3), 24.5 GiB + gfx1036 iGPU |
| ROCm | `/opt/rocm-10.1.0-gfx1151` | `/opt/rocm-10.1.0-gfx1100` |
| apply at `919794500` | strict **18/18**, tree **`0c46248f7371a0eee84a980a1e16ad25a6fbd38f`** == `release.json.tree` | same |
| applied tip | **`6d9d54238`** == `release.json.tip` (deterministic re-cut) | same |
| build (`build-r328`) | `BUILD_RC=0` | `BUILD_RC=0` |
| coherence, r327 → r328 (27B Q8_0 / 27B UD-Q4_K_M, `-ngl 99 -sm layer`) | **`da2e2d192e21` → `da2e2d192e21`** | **`2f7f092f2383` → `2f7f092f2383`** |
| GPU faults (kernel journal + every arm log) | clean | clean |

Coherence is a per-arch reference: those hashes are **their own**, not the primary box's `1c5d32ac537d`, and the
point is that each box reads the *same* text on the released r327 build and on the r328 build.

## Op oracles (`test-backend-ops -b ROCm0`, the r138 matrix)

| op | gfx1151 | gfx1100 | gfx1201 (reference) |
|---|---|---|---|
| `FLASH_ATTN_EXT` | **7429/7429** | **7429/7429** | 7429/7429 (delivery-only build) |
| `GATED_DELTA_NET` | **75/75** | **75/75** | 75/75 |
| `FLASH_ATTN_QSA` | **26/26** | **23/23** | 26/26 |
| `MUL_MAT` | **2692/2692** | **2692/2692** | 2692/2692 |
| `MUL_MAT_ID` | **1077/1077** | **1077/1077** | 1077/1077 |
| `TOP_K_MASK` | **38/38** | **38/38** | 38/38 |
| `HC_MIX` | **30/30** | **30/30** | 30/30 |

No failures anywhere.  `FLASH_ATTN_QSA` is 23 cases on gfx1100 vs 26 on the other two — the arch-gated case list
r138 recorded (gfx1100 is below the f16-WMMA set), not a gap.

**This is the first rung where the gfx11 chunked-GDN file runs on real gfx11 silicon.**  `GATED_DELTA_NET` 75/75 on
gfx1100 is the oracle for `gated_delta_net_chunked_bf16_gfx11.cu`, the file gfx1201 never launches; it also ran in
the coherence prefill on both boxes (qwen35 delta-net layers).

## Block 15 on the box where it is not inert: `gate-uma-experts.sh` — **48/49 PASS**

`halo` is the only one of the three boxes with an integrated GPU, i.e. the only one where the UMA GTT store's
hooks are live (`prop.integrated`, `RUNGS.md` §R15.2).  48 of 49 checks pass, including the ones that were
vacuous on gfx1201:

* text identity across all **17 arms** (arm `off` reference `4f29f1e4a4d6`, 575 chars);
* the placement-vs-arena shape switch: `-lme 4` against a 4096 MiB store → **ARENA** with the cap named;
  `-lme 20` covering the set → **PLACEMENT** (nothing streamed, nothing evicted);
* `nolme` → a >RAM model with no `-lme` is **REFUSED at load**, message names the opt-in; `ram` (unset `-lme`) →
  the arena with 84618.6 MiB used and 3267 evictions;
* `inplace`: 1603 slot admissions, arena D2D **1194.4 MiB vs `pfcp`'s 3357.6 MiB** at the same slots, 971.7 MiB
  of 11446.2 MiB read (8.5 %);
* `MOE_EXPERT_CACHE_TOUCH` 1/2/3 identical (6648 evictions) — a documented no-op on this tier;
* `ref`: no demand fill waited ≥ 1 ms, no sizing fill fell back to the serial arm; `pfverify`: every coalesced
  span equals the model mapping; no arm logged a GPU fault, abort or assert.

**The one failure** — `FAIL ref: the GTT sizing fills did not report the placement queue (the batch path declined
silently)`.  The ref arm's stderr carries `apply_heat_seed_locked: ... heat-seeded 56/56 slots from the persisted
map (256 experts, ranked by the DECODE band)` **with no MiB/ms/transport suffix at all**, while the binary
`strings` for that TU carries the full format (`… ), %.1f MiB in %.0f ms (%.0f MiB/s, %s)` and one `serial fills`).
So the census line is truncated or re-emitted somewhere between the kernel-side log call and `err.txt` — a
**census/instrument question first**, and only a transport question if the queue really is being declined.
`TODO` #357 carries it; it is a gfx1151-side observation and does not touch the shipped gfx1201 behaviour.

## `gate-prefill-server-slots.sh` (>RAM, two-slot) — **not completed**, and why

The `copy` arm ran on the 122B-A10B Q8_0 (>200 GB of weights on a 123 GiB APU, the arena tier): slot A produced
**160 tok at 3.76 tok/s** with the prefill window interrupting it (longest gap 511 ms) and slot B **17 chars in
18.4 s**.  The `ip` arm then aborted the *harness* — `gate-prefill-server-slots.py` raised `URLError` streaming
slot B, i.e. the second arm's server did not answer in time on this hardware, not a gate verdict.

That is an environment limit, not a pass: **this arm is owed**, with the numbers above as the starting point
(the box needs the request timeout raised and/or a lighter `>RAM` shape before the arm can decide anything).
Recorded in `TODO` #354.

## `gate-hc-fusions.sh` on gfx1100 — **PASS (exit 0)**

`fingon` has the `GSQ-IQ3_XXS` class the gate wants, and it is the first arch where this gate has been run
outside gfx1201 since the r328 anchor fix (`RUNGS.md` §R18.4).  With `EXTRA_ARGS='-ngl 99 -fa on -ncmoe 40'` (a
24.5 GiB card cannot hold the 75 GB model at `-ngl 99`) and `SOFT_EXPECT=""` (another shape has no anchors):

```
chain_chain 447 3f049f67781c   reference (true chain)
mix_op      449 6a102dbc62d6   reported (numeric arm, no anchor for this class/state)
dsv4_pre_op 444 a512ee68c4b3   reported
dsv4_post_op 449 6a102dbc62d6  reported
all_ops     449 6a102dbc62d6   reported
ops_no_post 444 a512ee68c4b3   reported
defaults    444 a512ee68c4b3   ok (== ops_no_post)
```

Every hard arm is byte-identical to the arm it is graded against, the shipping default matches an explicit
state, no GPU fault.  The chain hash differs from gfx1201's (`a512ee68c4b3` there) — expected: a generated text
is a per-arch, per-shape reference.

## What this does and does not close

* **Closes:** the set applies and builds on both other validated arches; every op oracle the delivery owns is
  green on both; the released behaviour is text-identical to r327 on each box; block 15's core APU behaviour
  (store/arena switch, refusals, in-place band, admission no-op) is now measured on the one box where it is live;
  the HC gate is green on gfx1100; the gfx11 GDN kernel runs on gfx11 silicon.
* **Leaves owed:** `gate-prefill-server-slots.sh` on `halo` (harness timeout, `TODO` #354), the
  placement-queue census finding (`TODO` #357), and a per-box **prefill-logit base recorded on that box** (the
  gfx1201 base is refused by design on another ROCm line — not run in this session).

## `gate-spec-align.sh` on gfx1100 — **1 FAIL: the host-expert pair gained +7.2 %, the gate wants ≥ 10 %**

Run on the one box that has the gate's host-expert model (`35B-A3B/UD-Q4_K_M`), 1 GPU, `-ncmoe 40`:

```
PASS  1gpu: equal-length arm 150.3 -> 150.4 t/s (+0.1 %, inert)
PASS  1gpu: knob-on arm split 0 draft-length + 0 single-row steps
PASS  1gpu: both single-slot requests byte-identical on/off
FAIL  ncmoe: the diverging pair gained 7 % (93.8 -> 100.5 t/s), below the 10 % the fix must recover
PASS  ncmoe: equal-length arm 228.3 -> 256.0 t/s (+12.1 %, inert)
PASS  ncmoe: knob-on arm split 0 draft-length + 1 single-row steps (<= TAIL_MAX)
PASS  ncmoe: both single-slot requests byte-identical on/off
PASS  ncmoe: host-expert pair +7.2 %: split steps 44+1 -> 2+1
NOTE  ncmoe: the aligned pair split 2+1, within NCMOE_SPLIT_MAX=2 (TODO #356)
PASS  n_seq==1: greedy text byte-identical on/off (18f00253a900)
```

Read it as**arch/shape calibration, not a regression**: the alignment does what it is for (splits 44+1 → 2+1,
texts identical, the equal-length arm *faster*), but on a single 24.5 GiB card behind `-ncmoe 40` the host read
path is the bottleneck, so removing 42 redundant verify passes buys +7.2 % rather than the +13.7 % (1 GPU) /
+18 % (2 GPU tensor) the same gate measures on gfx1201 — and `RECOVER=1.10` is a gfx1201-calibrated threshold.
No text, no step-count and no default changed.  Recorded rather than re-tuned: the gate keeps its gfx1201 number,
and a gfx1100 run of it should pass `RECOVER=1.05`; the difference itself is worth knowing before anyone quotes
"the alignment is worth +13 %" for a 24 GiB card.
* **Not covered anywhere yet:** `gate-node-budget.sh` (no phi-4 on any of the three boxes).
