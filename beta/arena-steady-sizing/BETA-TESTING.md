# Beta testing — arena steady-state settle

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

# verify the tree you got (see README §"applied tree" — it is recorded after the first cut)
```

Kill-switch, no rebuild needed: `MOE_EXPERT_CACHE_STEADY_REARM=0` restores the pre-change behaviour, so an
A/B is one environment variable and the same binary.

## The checklist

1. **Text purity** — same seed, `--temp 0`, switch on vs off, with `MOE_EXPERT_HEATMAP=load` so the heat
   prior cannot move between arms:
   ```bash
   MOE_EXPERT_HEATMAP=load MOE_EXPERT_CACHE_STEADY_REARM=1 ...   # vs =0
   ```
   Expected: **byte-identical**. Already measured here (`sha256 af0ec4edd999a414`).
2. **The settle actually ran** — `-lv 4`, then:
   ```bash
   grep -E "settled UP|arena settle:" <log>
   ```
   Expect `N layer(s) settled UP to the arena the drop returned` and an `arena now` figure above the
   sizing figure. If the drop never fires (a decode-only workload) expect `settled-up 0` — that is correct.
3. **Throughput at depth** — **≥ 1 000 generated tokens, 2 000 preferred.** A short decode understates
   every residency effect; 128 tokens understated this one by 2x.
   ```bash
   llama-server ... --parallel 1 -ub 16384    # then a 4 k+ prompt, n_predict 2000, temp 0
   ```
   Report `predicted_per_second` for both switch arms, interleaved ABAB, at least 2 cycles.
4. **Both host tiers.** The value depends on what a miss costs, so please report which tier you are on:
   * a **mirror** (`-lme auto` covering the set) — expect a few percent;
   * a **partial pool** (`-lme N` below the set) or a **disk** tier (`MOE_HOST_POOL_DIO=1`) — expect much
     more. Measured here: +4.0 % mirror, +29 % partial pool.
5. **The census** (`-lv 4`, at exit): `expert access:` and `expert H2D traffic:` before/after, and
   `arena settle:`. The miss bytes are the hardware-independent figure — quote them, not just t/s.
6. **`-ub 32768`** if the card can hold it: this is where the defect is worst (49 % of the arena lost).
7. **MTP on** (`--spec-draft-model`): the arena serves the verify band, so the four-axis protocol
   (`benchmarks/mtp-adaptive-methodology.md`) is the gate, not a single t/s number.

## Report back

* your card(s), RAM, link width/speed, `-ub`, ctx, and which host tier (§4 above);
* the switch-on/off `predicted_per_second`, 2 cycles each, interleaved;
* the two censuses (`expert access:`, `expert H2D traffic:`) and the `arena settle:` line;
* the text hashes from §1;
* anything that looks like a non-uniform layer — `re-armed uniformly at N slots`, `stood the whole layer
  down`, `could not be re-armed slot-uniformly`. Any of those on a run where the settle fired is a bug
  report, not a slow number.
