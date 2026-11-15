# The slab's draft (aux) reserve is overwritten by the draft model's own preflight — a 2.4 GiB VRAM accounting error

**Status: OPEN (2026-11-14) — found and measured, NOT fixed. Nothing in `patches/`; no `TODO` handle yet.**
The fix's owner block is a maintainer decision (§7): the code is spread over blocks **09**, **10** and **14**.

**The claim, in one line.** `ggml_cuda_slab_set_aux_reserve()` is called from *every* preflight that returns
"cache on", so the slab's reserve is whatever the **last** call carried — and when the draft model has
host-resident experts it runs its *own* preflight last, passing the flat, **weights-inclusive**
`LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT` (3 584 MiB) instead of the target's derived draft-**context**
reserve (1 223 MiB). The slab then holds back 3.50 GiB it does not need, and the MoE arena loses it.

Measured on the reference one-card QFN `runme` (`--spec-draft-cpu-moe`): **arena 12 365.3 → 11 803.3 MiB**
— *below* the arm that never moved anything, despite 2 326 MiB of draft weights leaving the device. That is
**2 888 MiB** of freed VRAM absorbed by the reserve instead of the arena. `MOE_EXPERT_CACHE_AUX_RESERVE_MIB`
at the derived value recovers **+2 436 MiB** (arena 14 238.9 MiB, 29.7 % residency, hit 0.773 → 0.820).

---

## 0. Measurement context

| | |
|---|---|
| box | one RX 7900 XTX (gfx1100, **24 560 MiB**), Ryzen 7950X, **63 423 MiB** RAM, ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx1100`) |
| tree | `~/llama.cpp` build **b12386 (`b1f559df7`)**, the r289 `rdna-boosts` tree |
| model | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (qwen4exp, 512 experts × 10 used, 48 blocks of which 12 full-attention) + `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` |
| config | the model directory's `runme`: `-ncmoe 48`, `-lme auto` (a 46.84 GiB pinned mirror), `-c 80000`, q8_0 KV, `-fa on`, `-ub 2048`, `-t 7`, `GGML_CUDA_SLAB_HEADROOM_MIB=1280` |
| draft footprint | 1 808.4 MiB of weights, of which **1 750.0 MiB is one MoE block's 512 experts** (`blk.48.ffn_{down,gate,up}_exps`); the rest (58.4 MiB) is attention/hc/nextn |

## 1. The arms and what they measured

One measured turn each: **6 991-token prefill + 2 000 greedy tokens** (`ignore_eos`), after one warm-up turn,
heat prior pinned (`MOE_EXPERT_HEATMAP=load`), **interleaved ABAB in one session** (INVARIANTS r221 fact 7).
Raw rows: [`results.jsonl`](results.jsonl) (21 runs). Harness: [`ab-server.sh`](ab-server.sh).

| arm | what | arena MiB | resid. | hit | pp t/s | **tg t/s** | draft `dur(g)` ms |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** as shipped ×6 | MTP, draft experts **device**-resident | 12 365.3 | 25.8 % | 0.781 | 779.6 | **61.44** | 1 894 |
| B¹ `--spec-draft-n-cpu-moe 1` ×2 | **no-op** (see §8) | 12 365.3 | 25.8 % | 0.781 | 788.9 | 61.02 | 1 947 |
| **C** no draft ×4 | arena ceiling reference | 15 831.4 | 33.0 % | 0.826 | 849.5 | 44.80 | — |
| **D** no draft, `MOE_EXPERT_CACHE_MIB=12365` ×2 | arena-only control (C−D = the arena's decode slope) | 12 271.7 | ~25.8 % | 0.792 | 859.1 | 42.14 | — |
| **B3** `--spec-draft-cpu-moe` ×2 | draft experts **host**-resident → **the bug** | **11 803.3** | 24.6 % | 0.773 | 774.7 | 57.20 | 3 547 |
| B4 same, `-td 16 -tbd 16` + `GGML_CPU_MOE_OFFLOAD_THREADS=0` ×2 | the thread hypothesis | 11 803.3 | 24.6 % | 0.773 | 769.6 | 50.52 | 6 244 |
| **E** B3 + `MOE_EXPERT_CACHE_AUX_RESERVE_MIB=1223` ×2 | the workaround | **14 238.9** | 29.7 % | 0.820 | 804.2 | 58.19 | 3 848 |

Purity: **all 21 runs produced the identical greedy text** (`e961a6587b9e`), MTP vs plain and device vs host
draft alike. Acceptance moves by one draft with placement (A 0.94929 = 1 479/1 558; B3/E 0.94683 =
1 478/1 561, mean length 3.84 in both) — the residency-class coupling of `GREEDY-PURITY.md` §89, not a text
change.

Two side results this campaign is *not* about, but measured while the rig was up:

- **MTP is worth +37 % decode** (61.44 vs 44.80) and costs 3 466 MiB of arena. The arena's own decode slope
  at this config is **+6.3 % for +7.2 pp residency** (C against D, ≈ 0.87 %/pp) — so giving MTP up to buy
  arena is a ~6 % win against a 37 % loss. **Keep MTP.** (Note that 0.87 %/pp is *not* r144's flat 6→25 %
  curve on a 2-GPU split; 1 GPU, 2 000-token generations and a pinned mirror is a different regime.)
- Drafting from host costs **+1.7-2.0 s per 2 000 tokens** (1 894 → 3 547 ms) even with the arena restored
  (E), which is why arm E is still ≈ −5 % on arm A. The extra threads make it *worse* (6 244 ms): the cost
  is ~34 MiB of host reads per draft token, not thread starvation, and 16 uncapped threads on this box is
  the recorded loser (INVARIANTS r19: 8 capped → 38.9 vs 16 uncapped → 19.6 t/s).

## 2. The mechanism

Two call sites feed the reserve, and the wrapper re-arms the slab on *every* one that returns "on":

```c
/* ggml/src/ggml-cuda/moe-expert-cache.cu:7271 — the LATCH is a max, on purpose (see the comment above it:
   a --spec-draft-kv-window decline must not survive the smaller real number) */
if (g_preflight_done && aux_reserve_bytes <= g_preflight_aux && g_enabled) {
    return g_enabled;                              // early return, but "on"
}
g_preflight_done = true;
g_preflight_aux  = std::max(g_preflight_aux, aux_reserve_bytes);
...
/* ggml/src/ggml-cuda/ggml-cuda.cu:11219 — and the wrapper still writes the slab, with THIS call's value */
const bool on = moe_cache_preflight(ctx->device, host_expert_bytes, aux_reserve_bytes);
if (on && ...) {
    ggml_cuda_slab_set_aux_reserve(aux_reserve_bytes);   // LAST CALL WINS
    ...
}
```

The reserve itself is `headroom + aux` (`ggml-cuda-slab.cu:413-440`, memoised on first use), so the last
caller decides how much VRAM the slab withholds from the arena for the whole run.

Who calls it, and with what:

| call site | value | note |
|---|---|---|
| `src/llama.cpp:478-488` (per model load) | `MOE_EXPERT_CACHE_AUX_RESERVE_MIB`, else the **flat 3 584 MiB** when `params.load_mtp` | the flat constant is documented as *"exactly the FIELD draft's footprint (2 647 weights + ~430 KV + 512 compute)"* — **weights-inclusive** |
| `common/common.cpp:1466-1514` (after the target model, before the slab) | the **derived draft context** — 1 223 MiB here (kv 263 + compute 768 + slack 192) | "the draft's 1 808 MiB of weights are loaded BEFORE this slab and are measured from the free VRAM instead of reserved" — the correct accounting |

Ordering, and why the two arms differ:

- **Arm A** (draft experts on the device): target load (3 584) → `common` derived (1 223) → the **draft's own
  load preflights nothing**, because `llama_model_moe_cache_preflight()` iterates the model's *host-expert
  map* and the draft's is empty. Final aux **1 223 MiB** → `reserve 2.44 GiB`, arena 12 365.3.
- **Arm B3/B4** (`--spec-draft-cpu-moe`): identical, **then the draft model loads and its host-expert map is
  non-empty**, so it preflights with the flat constant (3 584) and the slab takes it. Final aux
  **3 584 MiB** → `reserve 4.75 GiB`, arena 11 803.3.

The log states it plainly — same preflight line, different slab:

```text
A:  SLAB_INIT dev=0 free=16.49 GiB reserve=2.44 GiB headroom=1.25 GiB aux=1.19 GiB ... mapped=14.00 GiB
B3: SLAB_INIT dev=0 free=18.20 GiB reserve=4.75 GiB headroom=1.25 GiB aux=3.50 GiB ... mapped=13.44 GiB
```

`free` is **1.71 GiB larger** in B3 (the draft's weights are not resident) and `mapped` is **0.56 GiB
smaller**. 1.71 + 0.56 = 2.27 GiB of the freed VRAM went to the reserve instead of the arena.

## 3. Why it matters

- The reserve double-counts the draft's **weights**. They are loaded before the slab (so they are already
  inside `free_b` and are measured there) *and* they are inside the flat constant the draft's own preflight
  re-applies.
- It silently taxes **every** configuration whose draft has host-resident experts — `--spec-draft-cpu-moe`,
  `-otd exps=CPU` on a draft, and the r10 recipe ("keep the draft context's host-resident ops on the host",
  `LLAMA_MTP_DRAFT_OP_OFFLOAD=0`) whose whole point is freeing that VRAM.
- It makes the *workaround* look like the fix: without `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` the measured cost
  of moving the draft's experts host-side is "−0.56 GiB of arena **and** −7 % decode", which reads as "the
  idea is bad" rather than "the accounting is wrong".
- It is 2.4 GiB on a 24 GiB card — 10 % of the device, ~3 pp of expert residency.

## 4. The proof

`MOE_EXPERT_CACHE_AUX_RESERVE_MIB=1223` (the value `common` derived and logged) on arm B3 → arm **E**:

```text
B3: arena 11 803.3 MiB (24.6 %)  hit 0.773  VRAM 21 319 MiB (2.3 GiB of the card idle)
E : arena 14 238.9 MiB (29.7 %)  hit 0.820  VRAM 23 687 MiB (card full again)
```

Nothing else changed: same flags, same prompt, same heat prior, same text. `MOE_EXPERT_CACHE_AUX_RESERVE_MIB`
sets *both* call sites, which is exactly what a correct implementation would do.

## 5. What a fix must preserve

1. **The latch is deliberate.** `moe_cache_preflight` must keep re-running the floor check when the aux grows
   *and* while the cache is declined, or a `--spec-draft-kv-window` draft (much smaller reserve than the flat
   constant) keeps a pessimistic decline, and a bigger real reserve keeps a cache that should have declined.
2. **The completeness gate's max.** `g_preflight_aux` is also what the deferred sizing and the transport
   queries key on; lowering the *slab's* reserve must not lower that.
3. **The slab still needs a reserve** — `headroom + aux` is what stops a post-slab allocation from
   over-committing (-spec `WINDOW`/narrow-2 pin, the draft's KV and compute). The derived context value is
   the right *size*; the bug is only in *who writes it last*.
4. **No behaviour change on the pinned/coverless paths.** A run with no draft, or with a device-resident
   draft, must be bit-identical (arm A must not move).

## 6. Fix options (to be chosen by the maintainer)

| option | change | risk |
|---|---|---|
| **a. writer identity** | only the *derived* (`common/common.cpp`) call may write the slab aux; the wrapper takes a flag/`aux_is_derived` | smallest surface; needs the wrapper's signature to grow one argument |
| **b. min-of-derived** | `ggml_cuda_slab_set_aux_reserve` keeps the **minimum** of the values seen after the derived one | blunt: a genuinely larger later reserve (a second draft) would be ignored |
| **c. per-model derivation** | the draft model's preflight passes its *own* context-only estimate (kv + indexer + compute + slack ≈ 610 MiB here, measured: 83.14 + 10.39 + 512.00 + 26.32) instead of the flat constant | more correct in general (a draft's own post-slab allocations are what it should reserve); touches `src/llama.cpp`'s load path and needs the draft's geometry plumbed there |
| **d. document only** | ship nothing; keep `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` as the operator knob and say so in `ENVIRONMENT.md` | leaves a 2.4 GiB trap in the default path |

Whichever is chosen, the *flat constant* itself deserves a comment: it is weights-inclusive by construction,
so it is only correct for a model that has not loaded yet — which is never the case at the point it is used
for a draft.

## 7. Owner block — ask first

`ggml-cuda-slab.cu` arrives in **block 10** (amended by **12**); `moe_cache_preflight` /
`ggml_cuda_slab_set_aux_reserve` in **09** (amended by 10/15); `common_draft_reserve_derive` and
`LLAMA_MOE_CACHE_AUX_RESERVE_MIB` in **09** and **14**. So the fix is an amendment to 09/10/14 — **a block
decision, which `AGENTS.md` says is never a gate outcome**. Ask before cutting it.

Gates for the change (nothing here is a prefill-kernel change):

1. `scripts/gate-host-pool.sh` — the pool/fill/takeover contract is the slab's neighbour and must not move.
2. `scripts/gate-uma-experts.sh` — block 15's tier keys on the same preflight.
3. Same-seed greedy text on: no draft, device draft, host draft — all three identical to the pre-fix build
   (`e961a6587b9e` is the recorded reference text for this config).
4. The **SLAB_INIT census** in all three arms: `aux` must be 1.19 GiB in every one of them, and arm B's arena
   must rise to ≈ E's 14 238.9 MiB (that is the whole point).
5. `scripts/validate-set.sh` + `scripts/gate-per-block-build.sh --changed-from <prev tip>`.

## 8. Reproduce

```bash
cd wip/slab-draft-aux-reserve
./ab-server.sh 2 A B E          # ~12 min: 2 interleaved rounds of the bug and its workaround
./ab-server.sh 1 C D            # the arena-only slope control
```

The harness prints one JSON row per run (`tee`'d to `$OUT/results.jsonl`, default `/tmp/slab-draft-aux`) and
records the fields that decide this question: `pp_tps`, `tg_tps`, `arena_MiB`, `residency_pct`, `hit_rate`,
`acceptance`, `mean_len`, `draft_gen_ms`, `slab_init_free_reserve_headroom_aux_GiB`, `vram_*`, `text_sha12`.

It pins the four things that make the arms comparable — identical prompt, `MOE_EXPERT_HEATMAP=load`,
interleaved ABAB, no parallel runs — and it rebuilds the prompt every run: **23 023 B, 6 991 tokens,
`sha256 d1eaaa5eefced21e3b96c84edc5690140a5d850573a15be84658c27cf73c1ce3`** (the delivery's own
`prompts/prose-rdna-boosts.txt`, `recall.txt`, `code-python.txt`, `reasoning.txt`,
`code-reasoning-mixed.txt` joined by blank lines; never edit those files).

**Two traps this campaign hit, both worth keeping.** (1) `--spec-draft-n-cpu-moe N` indexes *"the last N
**trunk** layers"*, and a one-block MTP draft's only MoE block **is** the MTP block — so `--spec-draft-n-cpu-moe 1`
is silently a **no-op** (measured: arms B1/B2 are arm A to the digit). Use `--spec-draft-cpu-moe` / `-otd`.
(2) `--spec-draft-cpu-moe` alone *loses* decode on this box (drafting 1 894 → 3 547 ms, +34 MiB of host
reads per draft token); that is a separate finding from the accounting bug and it does not go away when the
arena is restored (arm E).

## 9. Open questions

- Does the same last-call-wins path bite a **DFlash** draft, or a `--spec-draft-kv-window` draft, where the
  derived value is *smaller* than the flat constant by design? (The latch's comment says that case was
  considered for the *floor check*, not for the slab write.)
- Is the flat 3 584 MiB ever right for a draft's own preflight? Its own post-slab allocations are its KV +
  indexer KV + compute + slack — 610-620 MiB here, 1.2 GiB for the field draft at a longer context.
- A `--spec-draft-cpu-moe` arm on a **dense** model (no target host experts) never reaches any of this: the
  preflight is a no-op for a model with no host-expert map (`common/common.cpp:1469`). Confirm that stays
  true when the fix lands.

## 10. Related records

- `ENVIRONMENT.md` — `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` (the override this campaign used), `MOE_EXPERT_CACHE_MIB`
- `INVARIANTS.md` — r125 (the draft reserve is the input to this preflight), r19 (thread sizing / the MoE
  offload cap), r144 (arena residency vs decode on a 2-GPU split), r221 fact 6/7 (arena levers, interleaving)
- `patches/README.md` — block 09 (host-resident MoE + the preflight), block 10 (the slab), block 14 (the
  streamed KV and the draft reserve)
- `wiki/QFN-Memory-Usage.md` — the one-card accounting this work started from (the arena/reserve numbers here
  are the same run's)
- `archive/work/host-expert-transport/`, `archive/work/fa-stage-peak/` — neighbouring slab/defence work
- `WORKLOG.md` 261114 (the session this campaign came out of)
