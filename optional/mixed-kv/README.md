# `mixed-kv` — optional block `0101`: a q8_0 K cache with a smaller V cache, read natively

**Optional artifact, NOT part of the delivery.**  It was promoted into the default set as **block 17** in
r287 (`v18-4f9dfa0e3-r287`) and moved OUT again by the **r288 block-number re-org**, which is what gives it
this `0101` optional number.  Nothing here is in `patches/`, `release.json.tree`, `release.json.tip`,
`rdna-boosts-all.patch` or the fork's `rdna-boosts` branch, and a default `scripts/apply-all.sh` checkout
never contains it.  Only `scripts/apply-all.sh --include-optional` (asks `[y/N]`) or
`--include-optional-all` applies it, on top of the verified default tree.

* `0101-mixed-kv.patch` — the artifact.  sha256
  `33902efebd4051d6eaa94b49a4584133e3df249441f306e934402617dbc609c9` (the r291 cut; the r288/r290 cuts are recorded below)
* cut against the **r288 default delivery tree** `79d18f41b9ab9b235d35fd3a5034858eddd94844` (official blocks
  `0000`-`0015` plus the temporary `0300`; that is the r287 release tree with block 17 removed)
* produces tree `89b2555296146290f556f1cf017ed6ee124dfce6` when applied alone on that tree — byte-identical
  to the r287 release tree, which is the proof that moving the feature out of the default set did not change
  it (r288 invariant 2)
* cumulative with `0100` (`0100` first — numeric order) the pair produces
  `d85868cdd23447b4d7fa0f9b9f5f7bb5fa7363cf`; the two artifacts do not conflict
* **r290 re-cut**: applies strictly at the r290 default tree
  `4f0770d7538b6e633ede9271bebdc5f83340a3c9` and produces
  `097c51af763fc338ad0d33cf7f0a853be2aa4b17`; cumulative with `0100` is
  `00f2150fad1ad44c2d40c1456c708ea4c7103390`.  The produced tree is no longer the r287 release tree's
  bytes — expected, since the r290 default tree moved (block 07's opt-in WMMA verify family) while this
  artifact's own diff is unchanged.
* **r291 re-cut**: applies strictly at the r291 default tree
  `8423b2b5eac33f97ad530229b8c35240aa64fa08` and produces
  `02f8e9d0b9c0077e795c71a2e32b2ae890b41ab1`; cumulative with `0100` (numeric order) is
  `310386642121ffd22eb74268a6446b31433aada0`.  The default tree moved by block 12's r291 amendment
  (sequence parallelism), not by anything this artifact owns.
* **its own gate is `scripts/gate-mixed-kv.sh`**, run on the **optional-applied build**; it is **not** a
  default release gate any more, and the delivery's own gates do not cover a mixed pair.

## What it is

The delivery's context gate refuses every `type_k != type_v` pair by default, because a mixed pair misses the
native decode band and is staged through the whole-cache F16 scratch (1.7-3.6x slower, never smaller, when it
was first refused on 2026-09-11).  The artifact gives the tile kernel a second template parameter for V
(`type_V`, defaulting to `type_KV`) and instantiates it for **head 256 only**: K `q8_0` with V `q4_0`, `q5_0`
or `q5_1`.  The tile dispatch's V type comes from `ggml_cuda_fattn_tile_type_V()`, the same function the
F16-scratch allocator asks, so the instantiation, the launcher and the allocator cannot disagree — the
contract the three earlier drifts broke (the missing q4_0 arm, the mixed-K/V NaNs, the missing RDNA2 bf16
scratch).  `ggml_cuda_fattn_kv_mixed_native()` names the pairs, the RDNA4 band predicate accepts them, and
`llama_init_from_model` accepts them on a model whose attention layers are all 256-wide and that does not use
QSA (qwen4exp is refused; `GGML_OP_FLASH_ATTN_QSA` asserts `K->type == V->type`).  Every other head size and
pair keeps the same-type rule, and the error text names the one mixed form.

**Knob.**  `GGML_CUDA_FA_KV_MIXED=0` (kill-switch, default on): the mixed pair takes the staged F16 path (the
pre-promotion behaviour); the context gate still accepts the pair.  A same-type cache does not reach the new
code at all.  The three mixed instances are always compiled, so a build cannot accept a pair it did not
compile (the context gate cannot see the backend's build flags; making them conditional would need a backend
query).  `ENVIRONMENT.md` §8 is the catalogue entry.

## Which layouts it reaches — and which it cannot

It is a **cache-type** feature, so it is orthogonal to the device count: it works on one GPU, `-sm layer` and
`-sm tensor`, on gfx1201 / gfx1100 / gfx1151 (all three were gated).

| layout | result |
|---|---|
| Qwen3.5 / 3.6 / 3.8 hybrids (dense and MoE) whose attention layers are all 256-wide | **engages** — the pair is accepted and read natively |
| any model with a different head width (e.g. 128-wide attention) | **refused** by the context gate, same error text as before |
| a qwen4exp model (QSA sparse attention) | **refused** — its QSA op asserts `K->type == V->type` |
| any other mixed pair (e.g. K q6_0 with V q5_1) | **refused** |

## What it measures here (r287 gate ladder, 3 x R9700 gfx1201 unless noted)

* `test-backend-ops -o FLASH_ATTN_EXT` **7021/7021** on gfx1201, gfx1100 and gfx1151, in the default (native
  read), `GGML_CUDA_FA_KV_MIXED=0` and `GGML_CUDA_FA_KV_NATIVE=0` settings.
* W=1..8 width purity **maxdiff 0** for q8_0/q8_0 and all three pairs, on 1 GPU, `-sm layer` and `-sm tensor`.
* plain == draft-mtp at n_max 1/3/7 on all three arches.
* At `-c 131072`, 27B Q4_K_M, 3-GPU `-sm tensor`: KV **4352 MiB** (q8_0/q8_0) → **3712 / 3584 / 3328**
  (q5_1 / q5_0 / q4_0), the same compute buffers and `graph splits = 2`; decode at 32768 cells is 0.3-3.2 %
  below q8_0/q8_0, with mixed between the same-type neighbours (47.0 vs 51.6 / 46.0 t/s at 3 GPUs,
  q8_0/q5_1).  The staged kill-switch arm is 12 % behind the native read at 32768 cells and 25 % at 100000.
* Quality (27B, 3-GPU tensor, WikiText-2, 8 x 4096): mean KLD against an f16 base 0.0071 (q8_0/q8_0), 0.0079
  (q8_0/q5_1), 0.0082 (q8_0/q5_0), 0.0074 (q8_0/q4_0) — the mixed pair is within ~11 % of the same-type
  q8_0/q8_0 cost.

The trade is why it is an opt-in artifact: it buys a smaller KV cache (up to ~1 GiB at 131072 cells on 27B) at
a few percent of decode, which is worth it when the cache is what does not fit and is a small loss when it
does.  The default set stays at 16 official blocks plus the temporary `0300`.

## Not covered

Head sizes other than 256 (still refused), other mixed pairs (still refused), and quality at depth (the KLD
run is 4096-token chunks).  The artifact is not covered by the delivery's release gates; `scripts/gate-mixed-kv.sh`
on an optional-applied build is the gate.

## History

* PR #312 (@mjohanning99) — the submission, the review and the original measurements are in
  `../../archive/work/mixed-kv/` (revision 2 added the QSA refusal and the one-V-type-function change).
* r287 — promoted into the delivery as **block 17**, after the gate ladder.
* r288 — the block-number re-org moved it out of the default set as **optional block `0101`**.  The moved
  block's bytes are unchanged: applying this patch on the default tree reproduces the r287 release tree
  exactly (`89b25552…`).
