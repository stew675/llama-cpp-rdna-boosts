# Beta testing — arena steady-state settle (v2)

> **PROMOTED into block 09 at r292.** This checklist is retained as the tester record; the candidate no
> longer lives in `beta/`.

A one-file candidate patch. Nothing in `beta/` is applied by `scripts/apply-all.sh` without
`--include-beta`, so it cannot leak into a release tree.

## Apply

```bash
# a clean checkout at release.json.base, with the 17 default blocks applied
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 4f9dfa0e3
bash <delivery>/scripts/apply-all.sh .
bash <delivery>/scripts/apply-all.sh --include-beta .    # 17 blocks + this patch
~/bin/build-llama-rocm                                   # build-rocm

# the applied tree after the beta commit is 9cd799ca3e86435eacd1d1b8e9492f1130bc0354
```

Kill-switch, no rebuild needed: `MOE_EXPERT_CACHE_STEADY_REARM=0` restores the pre-change behaviour, so an
A/B is one environment variable and the same binary. `MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP` (default 1.0)
is the marginal prior-mass gain that makes taking the band worth its whole-table re-seed.

## The checklist

1. **Text purity** — same seed, `--temp 0`, switch on vs off, with `MOE_EXPERT_HEATMAP=load` so the heat
   prior cannot move between arms:
   ```bash
   MOE_EXPERT_HEATMAP=load MOE_EXPERT_CACHE_STEADY_REARM=1 ...   # vs =0
   ```
   Expected: **byte-identical**. Already measured on every arm in `README.md` §3.
2. **The gate decided something sensible** — `-lv 4`, then:
   ```bash
   grep -E "arena settle: the first drop|arena settle: the persisted|settled UP" <log>
   ```
   Expect one `the first drop offers ... gain X pp ... settle ENABLED/DISABLED` line. A saturated split
   (2-device `-sm tensor`, 3-device, or a small `-ub`) should say **DISABLED** and be byte-identical to
   base. A short/1-device arena should say **ENABLED** and grow.
3. **Throughput at depth** — **≥ 1 000 generated tokens, 2 000 preferred.** A short decode understates
   every residency effect; 128 tokens understated this one by ~4x, and a few-slot growth still pays the
   whole-table re-seed.
   ```bash
   llama-server ... --parallel 1 -ub 16384    # then a 4 k+ prompt, n_predict 2000, temp 0
   ```
   Report `predicted_per_second` for both switch arms, interleaved ABAB, at least 2 cycles.
4. **Both host tiers, and the SMALLEST card you have.** The value scales with `1/arena_size`: measured
   **+8.2 %** on a 32 GiB R9700 (arena 23 GiB), **+19.5 %** on a 24 GiB RX 7900 XTX (arena 16 GiB) and
   **+36.8 %** on a partial pool. A 16 GiB card should be worth more again.
5. **The census** (`-lv 4`, at exit): `expert access:`, `expert H2D traffic:` and `arena settle:` before
   and after. The miss bytes are the hardware-independent figure — quote them, not just t/s.
6. **`-ub 32768`** if the card can hold it: this is where the defect is worst (49 % of the arena lost).
   Measured +31.9 % decode and -70 % miss traffic at `-ub 32768` on the 32 GiB card.
7. **MTP on** (`--spec-type draft-mtp --spec-draft-n-max 3`): the arena serves the verify band, so the
   four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`) is the gate, not a single t/s number.
   Text and acceptance must be identical; decode should be positive where the band is taken.
8. **A two-slot server** (`--parallel 2`), the transition the maintainer cares about:
   * one slot decodes while the other prefills;
   * the prefill completes and that slot joins the decode;
   * the short slot finishes and **starts a new prefill** while the other still decodes;
   * a fresh prefill after both are idle.
   Expected: every response coherent, **no** `could not be re-armed slot-uniformly`, **no**
   `is not slot-uniform`, **no** layer stood down, no GPU fault, and the settle fires at most once (later
   drops re-arm at the achieved count). Concurrent text need not be byte-deterministic (KV placement can
   move it) — that is explicitly not a failure.

## Report back

* your card(s), RAM, link width/speed, `-ub`, ctx, and which host tier (§4 above);
* the switch-on/off `predicted_per_second`, 2 cycles each, interleaved;
* the two censuses (`expert access:`, `expert H2D traffic:`) and the `arena settle:` decision line;
* the text hashes from §1;
* the two-slot transitions from §8 if you run them;
* anything that looks like a non-uniform layer — `re-armed uniformly at N slots`, `stood the whole layer
  down`, `could not be re-armed slot-uniformly`. Any of those on a run where the settle fired is a bug
  report, not a slow number.
