# The Meta split-state warm walk, and `--load-mode mlock` against an `-lme` budget (r296, `TODO` #330 / #333)

Two measurements from the r296 release.  Both are about a **dynamically decided tier** meeting a **static
policy**: the split-state cache's version mechanism (block 09) against a graph that is re-allocated with a
different shape, and the loader's `mlock` sweep against the `-lme` host-expert budget (block 14).  Neither
changes a kernel, so neither has a KLD arm; both have a text or a `/proc` oracle, and each gate asserts the
mechanism rather than the number (`scripts/gate-seqpar.sh`, `scripts/gate-host-read-transport.sh`).

Box: 3 x R9700 (gfx1201), ROCm 10.1.0, 91 GiB RAM, 32 threads.  `HIP_VISIBLE_DEVICES` pinned on every arm.

## 1. `TODO` #330 -- can the warm walk still be stranded?

**The mechanism (r291's diagnosis, now confirmed).**  `ggml_backend_meta_get_split_state` pre-warms the
per-buffer-context split-state cache with a non-recursive walk, because `calculate_split_state` recurses into
every `src` and a qwen4exp graph's chain is ~500-1200 nodes deep.  The walk fills the cache version that is
current when it starts; a nested lookup of a tensor whose cached bytes changed **switches the version**, and
every entry the walk already filled becomes invisible.  When the switch happens on the root's own lookup
(after the walk) the whole walk is stranded and the recursion descends the chain.  Versions are a performance
mechanism only: an entry is valid iff the key (tensor, `assume_sync`) is present and the tensor's raw bytes
are identical, so the fix is to make one walk fill one version.

**The fix (r296).**  (a) The tensor's own version resolution now runs **before** the walk, so the walk starts on
the version the lookup will end on.  (b) While the walk runs, a nested lookup that finds a stale entry for its
own key **erases it and computes into the current version** instead of switching (`ss_pin`), so the walk cannot
be stranded; the erased entry was invalid for those bytes anyway.

**Evidence.**

* **Recursion depth, instrumented** (temporary `[SSDEPTH]` probe in `calculate_split_state`, one line per new
  maximum): **1** on every arm — qwen4exp `-sm tensor` 2 devices, `LLAMA_SEQPAR_MIN=2`, `-ub 48` at `pp4096`,
  and the multi-shape sweep `-p 24,48,64,80,112,160,256,512` in one process (the shape alternation that retires
  versions).  Depth 1 means every source of the tensor being computed was a cache hit: the walk did its job.
* **`GGML_META_SS_VERIFY=1`** (recompute every outermost cache hit and abort on a stale entry) is clean over the
  same sweeps and over `-p 24,48,112,256` — i.e. no cached state disagrees with a recomputation from the cached
  sources.
* **The band below 128** (r291 recorded crashes at T = 48/64/80/112 and clean at 32/96/128+): every shape is
  clean here, at `LLAMA_SEQPAR_MIN=2`, including the chunk sequences whose **tail** lands below the floor
  (`-p 23 -ub 17` = 17 + 6 tokens; `-p 64 -ub 48` = 48 + 16).
* **The 2-token failure r291 recorded as a token-scatter assertion is the same op family, not a second
  defect**: the abort is `ggml-backend-meta.cpp:1953`, the first `GGML_OP_HC_MIX` assert, and it fires for any
  token-split ubatch at or below the fused hyper-connection band (measured: **8 and 16 abort, 17 and up run
  clean**).  `GGML_OP_HC_MIX`/`HC_COMBINE` assert mirrored inputs and have no token-split form, so the band is
  a graph property.  The seqpar floor is therefore **17** (`HC_BF16_MAX_TOKENS + 1`), not 128: the floor is the
  graph's own limitation, and `LLAMA_SEQPAR_MIN` is clamped to it with one WARN.

**Text is unchanged** for every arm that does not depend on the new floor (same seed, `ROCBLAS_USE_HIPBLASLT=0`,
r295 build against this one): qwen4exp Flash-Next 2 devices `-sm tensor` seqpar ON **166 chars
`97a8c2a4929b`**, seqpar OFF **`97a8c2a4929b`**; 27B Q4_K_XL 3 devices `-sm tensor` **96 chars
`db556809f0e6`**; 4B Q8_0 3 devices `-sm tensor` and `-sm layer` **96 chars `6541eadb9041`**.

## 2. `TODO` #333 -- what does `--load-mode mlock` pin?

`grow_to()` locks everything from the mapping base up to its target, and `load_all_data` calls it for every
tensor that points into the mapping.  With an `-lme` budget those mapping-backed tensors are exactly the
**holes**, so an mlock mode pinned the swept range across the whole hole set — the tier budget was silently
overridden.

**The A/B** (Flash-Next GSQ-RCO IQ3_XXS, 2 devices `-sm tensor -ngl all -ncmoe 48 -lme 30 --load-mode
mmap+mlock`, `-c 4096`, RLIMIT_MEMLOCK raised to `unlimited` with `sudo`; `VmLck` sampled from
`/proc/<pid>/status` every 200 ms):

| build | VmLck peak | the hole set |
|---|---:|---|
| before r296 | **45 590 460 kB = 43.5 GiB** | 40 925 MiB pageable by design |
| after r296 | **266 752 kB = 260 MiB** | 40 925 MiB named pageable in the census |

The 260 MiB is the non-hole mapped tensors (the mlock census prints it, with the pageable hole set beside it).
The tier's own pin is untouched: the pool reports `pinned 30 693.8 MiB` in both arms, because a host-resident
expert must be served from a wired page on an RDNA GPU (no XNACK) whatever the load mode says.

**What the flag is actually for** (4B Q8_0, `--load-mode mmap+mlock`, this box):

| arm | mapping pinned | reading |
|---|---:|---|
| `-ngl 0` (CPU inference) | **4 274 MiB** | the whole model: every token re-reads it from host RAM |
| `-ngl 99` (fully offloaded) | **654 MiB** | what still lives in the mapping (the output layer, any `-ot` tensor) |

So `mlock` is a **residency** request for the tensors the **CPU** reads, plus the load itself — not a read
transport and not a device-side decision.  A model whose weights are all in device buffers has little left for
it to keep, which is why the two flags can compose instead of one outranking the other.

## Arms to reproduce

```bash
# the seqpar band and the floor (no crash, engagement):
for T in 8 16 17 23 24 32 64 112 128; do
  LLAMA_SEQPAR_MIN=2 HIP_VISIBLE_DEVICES=0,2 build-rocm/bin/llama-bench -m <Flash-Next> -p $T -n 0 \
      -sm tensor -ngl 99 -ub $T -r 1
done
#   expected: 8 and 16 exit 0 and do NOT split (the fused HC band); 17 and up exit 0 and split.

# the walk is not stranded (debug switch, aborts on a stale entry):
GGML_META_SS_VERIFY=1 LLAMA_SEQPAR_MIN=2 HIP_VISIBLE_DEVICES=0,2 build-rocm/bin/llama-bench \
    -m <Flash-Next> -p 24,48,64,80,112,160,256,512 -n 0 -sm tensor -ngl 99 -r 1

# the mlock/hole composition:
sudo bash -c 'ulimit -l unlimited; exec build-rocm/bin/llama-cli -m <Flash-Next> -ngl all -ncmoe 48 \
    -lme 30 --load-mode mmap+mlock -c 4096 -n 4 -p "The capital of France is" ...'
#   expected: the "holes are left PAGEABLE" WARN, "mlock: <small> MiB ... (<large> MiB left pageable)".
#   `scripts/gate-host-read-transport.sh` asserts it, and asserts the quantity when sudo is available.
```

## 3. Found on the way: the KV residency gate's pool text reference was the side that drifted

`scripts/gate-kv-residency.sh`'s multi-device section asserted `pmres == pmref` ("a per-device pool is
byte-identical to the 2-device sparse arm") and was red before r296 touched anything.  Measured identically on
the **r295** and the **r296** build (Flash-Next GSQ-IQ3_XXS + MTP sidecar, `-sm layer` on `0,1`, `-ncmoe 48`,
`-ot per_layer_token_embd.*=CPU`, `-lm none -lzm off`, `-c 32768`, `-ctk/-ctv q8_0`, 25 763-token prefix,
`-n 64`, `MOE_EXPERT_HEATMAP=load`, `ROCBLAS_USE_HIPBLASLT=0`):

| arm | text | chars |
|---|---|---:|
| `pmref` -- no pool, 2 devices, `LLAMA_QSA_DENSE_DECODE_UNTIL=0` (forced sparse decode) | `1e944569dbd6` | 831 |
| `pmres` -- `--kv-attendance-pool on`, 2 devices | `f260d1824ab3` | 842 |
| `pmlegacy` -- `--kv-region-pool 32768` (accepted and ignored), 2 devices, `LLAMA_KV_POOL_VERIFY=1` | `f260d1824ab3` | 842 |
| `pres` -- the single-device pool arm of the same run | `f260d1824ab3` | 842 |

Every earlier record of this identity agrees with itself and disagrees with today: r213 lists `f260d1824ab3`
for all three of its 2-device arms (`pmres == pmfloor == pmref`) and r284 (`GREEDY-PURITY.md` §89, after the
heat-map pin) lists `b08d3dbcb1e9` for all three, while today the pooled arms read `f260d1824ab3` against the
resident arm's `1e944569dbd6` -- a re-baseline moved one side and the assertion was never re-cut.  It also
conflates the placement with the device count (text is a per-device-count reference under `-sm layer`) and gives
only the reference arm the forced-sparse decode switch.  The divergence is 539 characters into a coherent continuation
(`...answer user's last request: "Explain, step by step...` against `...answer user's request. User: "Explain,
step by step...`), and `LLAMA_KV_POOL_VERIFY=1` reports `bad 0`: every resolved page is the authoritative
mirror's bytes, so this is not a pool data defect.

The gate now asserts what the section exists for -- `pmlegacy == pmres` and `pmres == pres` (the pool's text is
device-count invariant; a split that mis-claims a resolve corrupts it) -- and reports the resident arm's number
and char delta instead of asserting it.  Re-graded on the same logs and re-run on the release build.
