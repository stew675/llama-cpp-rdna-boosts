# beta Block 0015 — campaign wins (staging)

> **Start here: [`HANDOVER.md`](HANDOVER.md)** — the plan, the maintainer decisions of
> 2026-09-10, the merge/gate/validate/cut steps, the state inventory and the open questions.
> This file is the reference: inventory, gate audit, validation protocol, reference numbers.

**Status: STAGING — not built yet (2026-09-10).**  This directory is the collection point for the
memory campaign's validated wins on their way to becoming a single **Block 0015** patch: all wins on
by default, each with an environment-variable kill-switch so beta testers can A/B (and bisect) any
issue during the beta window (~4–5 days).  **V3 is DONE (2026-09-10, on by default, −799 MiB/GPU and
−799 MiB host measured); the critical path is now V4 only** (§1b, and §3.2 of `HANDOVER.md` for the V4
map).  Block 15 waits for V4, so the beta window does not open until it lands.  Only after that window
does Block 15 get promoted into the
delivery set (`patches/0015-rdna-boosts-block-15-<slug>.patch`, `scripts/apply-all.sh` 14 → 15,
`scripts/make-patches.sh` tip, `MANIFESTS.md`/`README.md` headers, a `WORKLOG.md` entry, and this
README turned into the promotion record).

Promotion is the campaign's end-gate; nothing gets promoted before it is (a) merged with the other
wins, (b) fully gated, and (c) re-validated **as a combination** — every item below was validated on
its own, not together.

## 1. Inventory (all validated, all currently in `wip/`)

| # | win | source | measured effect | validated on |
|---|---|---|---|---|
| W1 | **L2 score-chain memory**: relu before the 4-D reshape + `n_blocks`-chunked assembly with `ggml_concat` | `wip/qwen4exp/qsa-memory/patches/0001-L2a-L2m-qsa-score-memory.patch` (1 file, `src/models/qwen4exp.cpp`) | compute reserve 6690.40 → 4450.40 MiB/GPU at ub2048 (ub1024 3346.50 → 2274.35, ub512 1724.56 → 1188.56); bit-identical output | qwen4exp coherence, ub sweep |
| W2 | **L1 derived QSA bias + visibility, mask prune, input-fill guards** (incl. the `llm_graph_input_attn_k` null-mask guard) | `wip/qwen4exp/qsa-memory/patches/0002-derived-qsa-block-bias.patch` (10 files, +557/−80, base = W1) | compute 4450.40 → **3251.39** MiB/GPU and host 1262.70 → **63.69** MiB at ub2048 (ub1024 1675.33/33.64); coherence byte-identical (derived vs mask, and vs the dense FA fallback); MTP 0.61616 | qwen4exp: 3k + 40k prompts, sparse/dense A/B in one binary, MTP probe, ub sweep |
| W3 | **keys-only QSA indexer cache** (the indexer V buffer is dead: keys-only scoring) | `wip/qwen4exp/keys-only-indexer/0001-keys-only-qsa-indexer-cache.patch` (3 files, 62 lines) | indexer KV 956.25 → 318.75 MiB; box total 88.58 → 86.70 GiB (V was triplicated); perf parity; validation matrix (bf16/f32 ctx, parallel 2, unified, prompt-cache/checkpoint round-trips, MTP 0.741, f16 byte-identical) | qwen4exp |
| W4 | **ggml-alloc: release unused view sources** (3b) | `upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch` (+35 lines, `ggml/src/ggml-alloc.c`; upstream-applicable, clean on `9cf3bf256`) | repro 56.00 → 16.00 MiB; removes the leak that forced W1's concat assembly; no change on current models (latent trap) | master: repro + `test-alloc` + `test-batch-alloc`; fork: coherence byte-identical on 4B/27B/qwen4exp, reserves and MTP unchanged, `test-backend-ops` VIEW/CONT/CPY/DUP/CONCAT OK |

### 1b. Planned in Block 15 — the critical path (Block 15 waits for these)

| id | what | expected effect (dense models, ctx 204800, ub 2048, q8_0 KV) | spec |
|---|---|---|---|
| **V3** | derived kq mask for the plain attention path: stop materialising the `n_kv × n_tps` F16 mask and its host mirror; derive visibility in the FA prefill/MMA kernel from compact per-cell state (packed mask kept for decode, small batches and every unsupported case).  **DONE 2026-09-10 (phases 1+2a+2b+2c), ON BY DEFAULT** — the predicate is proven bit-exact on the host (incl. SWA and non-causal), the op + a CPU reference + the CUDA MMA derived path are validated standalone (6/6 derived `test-backend-ops` cases on CPU, 3/3 on ROCm0, 5104/5104 FA suite), and the graph plumbing/probe shipped as `patches/0005-*`.  The packed mask is still created in every graph — it simply ends up with no consumer on the derived path, so the allocator leaves it unallocated and the existing fill guards skip the fill (no consumer can ever be mis-served) | **−799 MiB/GPU VRAM − 799 MiB host** measured (4B 1800.33→1001.13 compute + 840.34→41.13 host; 27B 1920.33→1121.13 + 880.34→81.13; gemma-4-E4B ISWA −809/−809; gemma-4-31B −811/−811; scaling exactly `n_kv × n_tps × 2 B`).  Cost: prefill −1.1 % (pp20480/ub 2048), decode −0.7 %; MTP acceptance identical (0.76744); qwen4exp unchanged | **`wip/arch-independent-memory/V3-DERIVED-KQ-MASK-PLAN.md`** (verified predicate + spec + §4.3-§4.5 record); `DERIVED-MASK-DESIGN.md` §2–§5 + §7; `HANDOVER.md` §3.1 |
| **V4** | native quantized K/V in the MMA FA path: dequantize into the shared K/V tiles instead of staging an F16 copy of the whole cache in a global scratch | **−832 MiB/GPU**, exactly ctx-linear | same, §1.3 + §4 (V4); plan in `HANDOVER.md` §3.2 |
| **V2** | *fallback for V3 only*: 1-bit packed mask (bit-exact by construction, no per-cell state) if V3 phase 3.1 proves too invasive | −750 MiB/GPU − 750 MiB host | same, §4 (V2) |

Not in Block 15 at all: **3a** (through-view reuse — measured **zero** reserve win on the 27B/4B) and
everything in `archive/work/`.

## 2. Gate audit

Policy: every win is **on by default**; a tester must be able to switch each one off individually.

| win | gate today | needed |
|---|---|---|
| W1 | none: the reshape order is unconditional and the chunking is a size threshold (`score_bytes > 128 MB`) | add `GGML_QSA_SCORE_MEM=0` → both off (default 1).  Keep the threshold as the internal policy, not as the A/B knob. |
| W2 | `GGML_QSA_DERIVED_BIAS` (0 = tensor path, 1 = derived, default), `GGML_QSA_DERIVED_VIS` (0/1, default 1), `LLAMA_QSA_SPARSE_FA` (dense masked-FA fallback) | keep as-is, but **strip the `GGML_QSA_DERIVED_BIAS=2|3` diagnostic modes** before promotion. Document the interaction: with `LLAMA_QSA_SPARSE_FA=0` the packed mask must be kept (the gate already encodes this via the shared `qwen4exp_qsa_sparse()` predicate). |
| W3 | none | add `LLAMA_QSA_KEYS_ONLY=0` → keep the (dead) V buffer (default 1). |
| W4 | none — **decided 2026-09-10: it is a bug fix, not a policy** | **no gate.**  A/B with the ready-made revert: `git apply ab/w4-revert.patch` (verified round trip: W4 → +35 lines → revert → pristine) or `git apply -R ../../upstream/UPSTREAM-PR-ggml-alloc-unused-view.patch`; rebuild and re-run `../../wip/arch-independent-memory/repro/ggml-alloc-unused-view.c` (56.00 MiB again = the bug is back). |
| V3 | **DONE (2026-09-10)** | `LLAMA_KQ_MASK_DERIVED` (default 1; 0 = always the packed mask) + the backend/cache capability check (the feature turns itself off where it cannot be correct). |
| V4 (planned) | — | one kill-switch (e.g. `GGML_CUDA_FA_STAGE_QUANT_KV=0` = the global F16 scratch path). |

Gate names follow the existing convention (`GGML_QSA_*` for the graph-level knobs, `LLAMA_QSA_*` for
the cache-level ones).

## 3. Integration work before the patch can be cut

1. **Merge the wins into one tree.**  They were validated individually; W2 and W3 overlap in
   `src/models/qwen4exp.cpp` and `src/llama-memory-hybrid-idx.cpp`, and W1 is W2's base — so the
   natural order is W1 → W3 → W2 (or W1 → W2 → W3 with a 3-way merge), then a re-diff of W2 against
   the new base.
2. **Add/repair the gates** from §2 and strip the diagnostic modes.
3. **Re-validate the combination** (protocol in §4).  A merged tree can move the peak in a way no
   single win shows, and the host-side mask build interacts with all of them.
4. **Cut the block** the way the other blocks are cut: commit the combined state on the fork's
   `rdna-boosts` branch as a 15th block commit (never pushed — see `AGENTS.md`), then
   `scripts/make-patches.sh` with the updated tip, and place the exported patch + this README's
   validation record here.  (Nothing under `wip/` is folded into `patches/` without the maintainer's
   go-ahead — Block 15 exists precisely so the campaign's wins *do* get that go-ahead as one package.)

## 4. Combined validation protocol

* **Coherence, byte-identical** (same seed), each gate on and off, in one binary where possible:
  4B / 27B / qwen4exp, at a short and a long prompt; sparse vs dense FA for qwen4exp.
* **Reserve matrix**: compute + host at ub 512 / 1024 / 2048 and at 1x / 2x context, per model; expect
  the W1/W2/W3 numbers above, or better.
* **Adaptive MTP gate** (`tools/mtp-ab.sh`): acceptance ≥ ~0.45 and MTP ≥ plain at depth 3 — the repo
  rule for anything that can move buffer layout.
* **Performance parity**: `tools/ub-sweep.sh` (llama-bench pp20480 / tg256, r=3) at ub 2048 and 1024 vs
  the 14-block build; the L1 measurement showed +1 % prefill, nothing else.
* **`test-backend-ops`** (at least VIEW/CONT/CPY/DUP/CONCAT/FLASH_ATTN_EXT on CPU + ROCm0),
  `test-alloc`, `test-batch-alloc`.
* **No reserve growth** across repeated graph builds (the W4 acceptance criterion).

## 5. Open questions for the maintainer

1. **W4 in two places?**  Recommended: keep it in Block 15 *and* in `upstream/` — the delivery must
   carry the fix until upstream takes it; if upstream merges it, Block 15 drops it at the next
   regeneration (the rule already stated in `upstream/README.md`).  Alternative: keep it out of Block
   15 and ship it only as an upstream PR.
2. **Split the generic part of W2 out?**  W2 carries one upstream-generic fix — the
   `llm_graph_input_attn_k::set_input` null-mask guard (a latent crash in upstream code: its own
   `can_reuse_impl()` already accepts a null mask).  Worth a separate tiny PR in `upstream/`?
3. **Block 15 scope.**  Recommended: freeze Block 15 on W1–W4 (all validated now) and let V2/V3/V4
   (the derived-mask facility, the 1-bit mask, native quantized K/V in the MMA FA path) land as Block
   16 or later additions.  Confirm — otherwise the beta window starts much later.
4. **Beta tester material.**  Want a one-page A/B checklist here (env-var matrix, what to report:
   coherence divergence, reserve numbers, perf deltas) to hand to testers?
5. **Naming.**  `beta/block-15-campaign-wins/` + `block-15-campaign-wins.patch`, promoting to
   `patches/0015-rdna-boosts-block-15-campaign-wins.patch` — confirm the slug, or pick another
   (e.g. `memory-wins`).

## 6. Reference numbers to protect (do not regress)

qwen4exp, ctx 204800, `-ctk/-ctv q8_0`, 3× R9700, per GPU:

| state | ub2048 compute | ub2048 host | ub1024 compute | ub1024 host |
|---|---|---|---|---|
| pristine (14 blocks) | 6690.40 | 1262.70 | 3346.50 | ~ |
| + W1 | 4450.40 | 1262.70 | 2274.35 | ~ |
| + W2 | **3251.39** | **63.69** | 1675.33 | 33.64 |
| + W3 | (W3 does not change the compute buffer; it shrinks the indexer KV cache, i.e. the "context" column) | | | |

Dense controls (no qwen4exp involved): Qwen3.5-4B-Q8_0 1800.33/840.34 and Qwen3.8-27B-Q8_0
1920.33/880.34 at ub2048 — Block 15 must leave these unchanged (W1–W3 do not touch them; W4 leaves
them unchanged as well).
