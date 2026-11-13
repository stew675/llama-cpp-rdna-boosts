# BASE BLOCKER B1 — upstream `ac931b5bc` (#30542) hangs the HIP VMM pool on ROCm 10.1 / gfx1201

**Found 2026-11-13 at rung R01.  Status: candidate fix applied to the ladder branch, `AWAITING` the
maintainer's decision.**  Registered as `DECISIONS.md` `D17`.

## What happens

Any GPU compute on the **unpatched** new base `4f9dfa0e3`, built the way the delivery builds (ROCm 10.1.0
`/opt/rocm-10.1.0-gfx120X`, `gfx1201`, `GGML_HIP_NO_VMM=OFF` — the delivery's host-expert arena needs VMM),
**never returns**.  The process spins at ~100 % of one core; the CLI's "Loading model..." spinner keeps
running and the first decode (or the warmup empty run) never completes.  `timeout` cannot stop it gracefully
(SIGTERM is ignored — it is a spin, not a wait), so it has to be `SIGKILL`ed.

```
HIP_VISIBLE_DEVICES=0 ./build-rocm-base2/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm layer -p "The capital of France is" -n 20 \
  --seed 42 --temp 0 --no-display-prompt --single-turn      # hangs
```

It is **not** a block-00 problem: the same command hangs on a pure `master` build (`build 12369, commit
4f9dfa0e3`, no delivery patches).  `-ngl 0` (CPU) works, so it is the GPU path.

## What the evidence says

| observation | value |
|---|---|
| stack of the spinning thread | `ggml_cuda_pool_vmm::alloc` -> `libamdhip64.so.7` -> `hipGetErrorString` (the `CU_CHECK` error path) |
| first call parameters | `size=5760 avail=0 reserve_size=2097152 pool_size=0 max_size=34359738368 granularity=2097152` |
| the first `cuMemAddressReserve` | returns `0` (success) in an instrumented build |
| HIP API log (`AMD_LOG_LEVEL=3`) | after the hang starts, a tight loop of `hipMemAddressReserve(ptr, 0, 0, NULL, 0)` each returning `hipErrorInvalidValue` — i.e. the retry loop is entered and never terminates on this runtime |
| unpatched base, `GGML_HIP_NO_VMM=ON` | **works** (`Prompt 265 t/s / Gen 89 t/s`) — so the hang is VMM-pool-specific |
| old base `a691b2345`, VMM=OFF | works (the r277 release ran this way) |
| delivery r277 binary, VMM=OFF | works |

So: the HIP VMM pool path in this base is broken on this box, and the delivery cannot simply build with
`GGML_HIP_NO_VMM=ON` because the host-expert arena (blocks 09/16) needs VMM.

## Which commit

`ac931b5bc` — *cuda : halve VMM pool reservation size and retry on cuMemAddressReserve failure (#30542)*
(2026-11-10, Michael Paulauski).  It adds a `max_size` member, reserves the VMM address range **before**
`cuMemCreate`, and retries with a halved reservation while the error is `CUDA_ERROR_OUT_OF_MEMORY`
(defined for HIP as `hipErrorOutOfMemory`).

* **Reverting it makes the unpatched base work** (verified: rebuilt `build-rocm-base2` at
  `4f9dfa0e3` + `git revert ac931b5bc`, the 4B coherence arm runs at `263.7 / 90.2` t/s).
* Re-ordering only the reserve back after `cuMemCreate` (keeping the retry) does **not** fix it, so the
  trigger is not just the call order.
* The exact failing call is not pinned yet.  The error-report path itself spins inside
  `libamdhip64` (`hipGetErrorString`), and the instrumented prints after the successful first reserve are
  lost, which is consistent with the thread being stuck in a HIP call immediately after it.  Treat
  "the HIP VMM pool sequence added by #30542" as the proven trigger and the per-call mechanism as an
  open detail (an upstream bug report would want it, the delivery does not).

## Candidate fix (what is on `rebase-r278`)

**A full revert of `ac931b5bc`, folded into block 00** (`rdna-boosts: block 00: foundation ...`), i.e. the
delivery ships the pre-#30542 VMM pool behaviour on every backend.

* R01 `43a418c10` = `f15430d1e` (block 00) + `git revert ac931b5bc`, build green with the canonical
  `GGML_HIP_NO_VMM=OFF`, coherence `100 chars sha=1c5d32ac537d` (`Prompt 256.0 / Gen 92.1` t/s),
  `test-backend-ops -o MUL_MAT_ID` **1040/1040**.
* Reverting wholesale (not HIP-gated) is deliberate: the commit exists for "CUDA GPUs with combined memory"
  (Jetson / GB-class), a configuration this delivery does not validate, and the pre-#30542 behaviour is what
  r277 shipped on every backend.  A HIP-only gate would need the `max_size`/`CUresult` surface split across
  both paths for no validated gain.
* When upstream fixes it (or if the maintainer prefers a HIP-only gate), the revert is dropped at the next
  re-base and the base advances.

## Options presented to the maintainer

1. **Carry the revert in block 00** (current ladder state; recommended — it is the only configuration that
   restores the delivery's required VMM path, and it is one revert commit that disappears with the next
   base).
2. **HIP-gate the revert** (CUDA keeps #30542).  More code, no validated benefit on this repo's scope.
3. **Verify rungs with `GGML_HIP_NO_VMM=ON`** and leave the arena/GTT rungs (R10, R17) blocked until
   upstream fixes it.  Non-canonical build for the whole climb; not recommended past R09.
4. **Move the fork point to `ac931b5bc^`** (i.e. one commit earlier) — loses 60+ unrelated upstream commits.

## Consequence for R18

The prefill-logit base record wants an **unpatched** `4f9dfa0e3` build.  Unpatched + VMM OFF does not run,
so the base record is taken on unpatched **VMM=ON** (the only working unpatched configuration) and the
candidate on the canonical VMM=OFF build.  The pool changes allocation only (no arithmetic), so the KLD
comparison stands, but the asymmetry must be stated in the gate record.  If the maintainer picks option 1,
an alternative is to record the base from `4f9dfa0e3 + revert` and note that it differs from upstream by one
allocation-only commit.
