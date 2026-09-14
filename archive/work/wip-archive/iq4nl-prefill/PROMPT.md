# Session prompt — qwen4exp weight-IQ4_NL prefill fast path (new item, propose #18)

Paste the block below into the fresh session.

---

You are working in the delivery repo `~/llama-cpp-rdna-boosts` (packages the
`stew675/llama.cpp` `rdna-boosts` fork as a 16-patch set). **Read `AGENTS.md` first** — its
apply/regeneration/landing/push rules and its purity rules are binding.

## Task

Close the **prefill** gap on Qwen3.8-Flash-Next (`qwen4exp`) between our fork and pwilkin's
`strix-halo` branch, by porting **only the generic parts** of his stack that fit our framework
(weight-GEMM fast path first; fusions second). This is a **new TODO item** (propose #18) — it is
**not** TODO item 3, which is the `iq4_nl` **KV-cache** axis and stays as recorded.

**Success criterion: > 1100 t/s prefill at pp16384 on pwilkin's uniform-IQ4_NL model.**
Today: rdna-boosts **787**, pwilkin's stack **1409** (1.79x). We are already ahead on decode
(31.26 vs 30.36).

## Read first (in this order)

1. `wip/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md` — full context, the A/B, the
   mechanism, the exact env, the scope, the plan, gates, landing.
2. `wip/iq4nl-prefill/launcher-env.txt` — pwilkin's stack env, sourceable.
3. `TODO.md` item 3 (the *separate* KV-cache axis) and `GREEDY-PURITY.md` (the purity rulebook).

## Constraints

- Port only generic aspects: the **`mmb`-style bf16-WMMA dequant weight GEMM** (`IQ4_NL`/`Q6_K`)
  first. No parallel env framework; add our own gates in the delivery's idiom (env-gated, default
  off until validated, folded into the owning block as an amendment).
- **Prefill-only, above the decode/verify band** (`T >= 512`): `W = 1..8` must stay bit-identical
  with the new path on and off, and the MTP acceptance gate must hold.
- Validate: same-seed coherence on our models; `test-backend-ops -o FLASH_ATTN_QSA` (22/22),
  `-o GATED_DELTA_NET` (46/46), `-o FLASH_ATTN_EXT` (5935/5935); cross-arch compile/consistency
  (gfx1100/gfx1201/gfx1151). RDNA-first scope policy applies.
- Land per `AGENTS.md`: amend the owning block, regenerate `patches/` from a canonical fork rebuilt
  at `9113cc188`, strict 16/16 `git am`, docs updates, push only to the delivery repo's own origin.
  Never push from `~/llama.cpp`.

## Start

**Phase 0** — reproduce the A/B in §6 of the handover **in-session** (builds and models are already
on disk; see §6 for paths), report the numbers, then do the two cheap build-flag checks
(`-DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON` on our build; our script vs his CMake flags for his
build). Do **not** start kernel work until the baseline is reproduced and the build configuration is
ruled in or out. Rule of thumb from the repo: same-session interleaved brackets only — the box drifts.
