# Fused-stage + host-side pacing — archived experiment (sessions 8-9, 2026-08-29)

**Status: CLOSED as net-negative on ROCm 7.14 / RDNA4 (gfx1201).  Archived for
the historical record and for re-evaluation after future ROCm/library updates.
The code is env-gated WIP; it is NOT part of the delivery patch set.**

## What this was

Two attempts to eliminate the ~12.7 us one-sided all-reduce wait (dev0 /
bus-06 dispatch-gap asymmetry, see `wip/HANDOFF.md` → "Item B"):

1. **Fused-stage** (`GGML_CUDA_AR_FUSED`, default 0): capture the AR phase-1
   staging (shard -> wire + arrival token) as the LAST node of each device's
   subgraph graph, so peers stop spinning at dev0's *subgraph end* instead of
   after the separate AR kernel's dispatch (the graph->kernel premium).
2. **Host-side lockstep pacing** (`GGML_CUDA_AR_PACE`, default 0): before
   enqueueing AR call N, busy-poll every peer's previous-call end-of-AR event.

## What was built (four fused-stage designs)

- **v1**: compute-kernel stage in the graph (write wire + arrival in one
  captured kernel).  Mechanism proven: decode spins collapsed 21/22 us ->
  2.4/2.4 us.  But `__threadfence_system()` inside a graph replay costs
  ~50 us on RDNA4 (drains the graph's L2), making it net-negative.
- **v2**: same kernel, control (slot/token) read from per-device device
  memory at replay so the baked graph's params rotate per call.  Coherent.
- **v3**: split the copy from the arrival -> SILENT CORRUPTION (thread-scoped
  fences cannot order another kernel's GTT writes; the writer must fence its
  own wire writes before any arrival).  Caught by the llama-cli coherence gate.
- **v4** (SDMA): convert kernel -> D2H memcpy NODES (wire + arrival).  Found
  DEAD ON THIS PLATFORM: D2H `hipMemcpyAsync` nodes are NOT captured into HIP
  graphs (verified: captured graph had 16 kernel nodes, 0 memcpy nodes); the
  memcpys execute once at hook time, copying uninitialized scratch.  Plus a
  wire-type bug (stage always wrote BF16; small tensors use F32 wire) and an
  L2-staleness failure mode (SDMA bypasses the GPU L2).

## Final coherent state (v2, what the archive patches represent)

- Stage kernel: convert F32 shard -> BF16, write host_wire (GTT), system
  fence (the ~50 us in-graph cost), signal ALL 8 blocks' arrival slots.
- Reduce: 8-block launch, uniform `stage_done` (validates control
  slot/token/types + own arrival), skips phase-1, volatile peer reads not
  needed once the wire is kernel-written (the writer fence orders it).
- Coherence: bit-exact wire (STAGE w0=3f96 == REDUCE w0=3f96 per token),
  llama-cli same-seed output IDENTICAL to the unfused path.

## Why it is net-negative (the mechanism, 27B tg64)

```
FUSED=0: decode AR 36.2 us/call, tg64 34.45
FUSED=1: decode AR 55.5 us/call, tg64 30.87
PACE=1 : decode AR 36.0 us/call, tg64 34.53 (wash)
```

1. The one-sided wait PERSISTS: the stage fires at each device's
   subgraph-END, and the subgraph-ends are asymmetric (dev0 +12-20 us late —
   the Item B CP-level dispatch-gap asymmetry).  dev0's spin collapsed
   (3.5 us) but dev1/dev2 still spin 20-22 us — the wait moved earlier, it
   did not disappear.
2. The in-graph GTT write + system fence drains the graph's L2 and costs
   more than the non-captured phase-1 it replaces.
3. The validation's volatile GTT reads add ~4 us/call.
4. The pacing gate is redundant with the AR's own barrier semantics (the
   previous call is long done; the stagger is the subgraph-END difference).

**Conclusion: the dispatch-gap asymmetry is a platform-level CP/driver
property, not reachable from the AR kernel, the graph tail, or host-side
pacing.  The rocprof "fix" is a per-dispatch GPU-side scheduling perturbation
that cannot be replicated deliberately.**

## Files

- `patches/0001-*` — the fused-stage v2 commit (allreduce-hip.cu + hook +
  stubs; the coherent fused path with GGML_CUDA_AR_FUSED).
- `patches/0002-*` — the pacing commit (GGML_CUDA_AR_PACE).
- `allreduce-hip.cu.fused-snapshot` — the full fused+pacing state of
  allreduce-hip.cu (70 KB) for reference/re-application.
- Session notes: `../../wip/HANDOFF.md` (sections "Fused-stage experiment" and
  "Fused-stage + pacing: DEFINITIVE CONCLUSION").

## To re-enable (if a ROCm update changes the calculus)

1. Check whether D2H memcpy nodes are captured into HIP graphs now
   (`hipGraphGetNodes` enumeration — v4's blocker).
2. Re-measure the in-graph `__threadfence_system()` cost outside the L2-drain
   case (v2's blocker).
3. Re-apply `patches/0001` + `0002` on top of the delivery patch set, or
   restore `allreduce-hip.cu.fused-snapshot` and diff against the delivery
   allreduce-hip.cu.
4. Gate: llama-cli same-seed coherence BEFORE any tg measurement; monitor
   rocm-smi (the KFD wedge that ended session 8 was likely the abort storm,
   not the code, but be careful with deadlocked spins — the spin has no
   timeout).
