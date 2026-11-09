# Issue #291: the per-layer rollback index chain in the MTP verify graph (r241)

Reporter: @DanoPTT, 2026-11-09. Dense Qwen3.8-27B + `--spec-type draft-mtp` at ~100k context,
decode -12.5 % and prefill -5.5 % from `v16-6050b467c-r220` (P) to `v16-a691b2345-r236` (R) on
Windows 11 / TheRock ROCm 10.1, GPU kernel times flat or lower, `GGML_CUDA_DISABLE_GRAPHS=1`.

The reporter's own analysis named the node: their target graphs grew **+96 nodes** (verify
1268 -> 1364, decode 1228 -> 1324), from upstream `040ce199a` (#31660), the per-conv-layer
`s_slot_rev` (`arange` + `scale_bias` + `cast`). This record is what that cost, what it was worth,
and what r241 changes.

## Reproduction (Linux, gfx1201, one R9700)

Config is the reporter's: `-fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 -np 2 --kv-unified
--spec-type draft-mtp --spec-draft-n-max 3`, `GGML_CUDA_DISABLE_GRAPHS=1`, server `/completion`,
`cache_prompt false`, temperature 0, `n_predict 128`, `top_k 1`.

Two prompts: `prompts/prose-rdna-boosts.txt` x2 in a chat template (**10523 tokens**) and x19
(**99845 tokens**). Builds: r220 (`build-rocm` in `~/llama-scratch`), r236 and r240
(`build-rocm-r236`, `build-rocm`).

| arm | prefill 10.5k (t/s) | MTP decode 10.5k (t/s) | prefill 100k | MTP decode 100k | plain decode 10.5k |
|---|---:|---:|---:|---:|---:|
| r220 | 1137.1 / 1151.6 | **58.97 / 59.72** | 787.6 / 786.7 | **58.99 / 59.31** | 26.77 |
| r236 | 1078.0 | 57.69 | 750.7 / 749.7 | 57.44 / 57.73 | 26.83 |
| r240 (shipped) | 1132.7 | 57.71 | 775.6 | 57.51 / 57.67 | 26.82 |
| r240 + r241 fix | 1145.2 / 1134.3 | **59.01 / 58.81** | - | - | 27.11 |

Plain (non-MTP) prefill: r220 1205.1, r236 1178.7, r240 1174.1, r241 1178.0.
Draft acceptance identical in every arm (124 drafted / 84 accepted at 10.5k, 112/89 at 100k), and
the generated text is byte-identical (`535c/3c3913603ed9`, 535 chars) across every build here.

**What reproduces and what does not.** The direction reproduces: at 100k, plain prefill -1.5 %
(r220 -> r236) and MTP decode -2.7 %, with plain decode flat. The MTP-specific prefill penalty is
r236-only and already gone in r240 (r236's MTP prefill 1078.0 -> r240 1132.7). The magnitude does
not: the reporter sees -5.5 % / -12.5 %, we see -1.5 % / -2.7 %. The remaining attributed cost here
is per **launch** and per **kernel**, not per host call, so a platform whose launches cost more
pays more for the same graph.

## What the +96 nodes are, and what they cost

`build_conv_state_store()` (`src/llama-graph.cpp`) emits, for the rollback path
(`cparams.n_rs_seq > 0`, i.e. any speculative run):

```
s_slot_rev = arange(0, K)          K = n_rs_seq + 1 <= 8
s_slot_rev = scale_bias(s_slot_rev, -1, K-1)
s_slot_rev = cast(s_slot_rev, I32)
slot_order = view_1d(s_slot_rev, Kw)
set_rows(slots_view_of_state, windows_view_of_concat, slot_order)
```

per **conv layer**, per **graph**. On the dense 27B (48 conv layers, 16 full-attention) that is
**144 nodes** of a 1268-node MTP verify graph, all rebuilding the same 4-element constant every
speculative round. The reporter's +96 is the same mechanism on a graph that already had `set_rows`.

Per verify round, `GGML_CUDA_OP_TIMING` on the verify graph:

| | r220 | r240 |
|---|---:|---:|
| graph nodes | 1172 | 1268 |
| dispatch entries | 1060 | 963 |
| graph GPU time | 44.29 ms | 45.39 ms |
| conv-state store | `FUSED CPY cache_r_l*` x192, 0.69 ms | `SET_ROWS` x48 + `CPY s_slot_rev` x48 + `CONCAT` x48, 1.28 ms |

Two mechanisms are visible, and only one is ours to own:

1. **The index chain (fixed in r241).** 141 of the 192 store nodes are the constant index. It is now
   built once per (graph, device) and each layer takes a view. Verify graph **1268 -> 1127 nodes**.
   A/B on one build (`LLAMA_SHARE_SLOT_REV=0` restores the per-layer chain): **58.37 / 58.31 ->
   59.01 / 58.88 t/s**, i.e. **+1.03 %** MTP decode, bit-identical text.
2. **The CONCAT (not fixed, recorded).** `CONCAT conv_input-*` is dispatched 48 times per verify
   round in r240 and ~4 times in r220, 0.010 ms each. In r220 it was absorbed by
   `ggml_cuda_ssm_conv_in_try_fuse` and/or the GDN direct conv; the GDN match now declines at
   `gdn_conv_check`'s state test whenever the ubatch has more than one sequence
   (`-np 2 --kv-unified` gives `st->ne[2] == 2`), and with the store's window view inserted between
   the CONCAT and the conv the `ssm_conv_in` epilogue no longer takes it either. Worth ~0.5 ms of a
   45 ms verify round (~1 %). It needs the matcher to accept the new window view, which is a
   prefill-kernel change and was **not** shipped on the strength of one arm.

## The r241 change

`build_conv_state_store()` caches the cast index per graph keyed on the **buffer type of the
recurrent state it writes** (the device), and each layer takes `{Kw-1, ..., 0}` as a `ggml_view_1d`
of it. A view is not a graph node, so the per-layer cost collapses to one chain per graph per
device. The device key matters: sharing across devices would turn every consumer on another device
into a cross-device copy of the index. `LLAMA_SHARE_SLOT_REV=0` restores the per-layer chain for
A/B and bisection.

## Gates on the shipped tree

* `scripts/gate-slot-rev-share.sh` PASS: one index chain per graph instead of 48, text identical,
  and the decode arm is not slower.
* Coherence `text-qwen35-4B 1c5d32ac537d` (the r239/r240 anchor).
* `scripts/gate-prefill-logits.sh` PASS, **mean KLD 0.000697, same-top-p 98.784 %**, identical to
  the recorded r239/r240 numbers, i.e. the change is numerically invisible as intended.
* `scripts/gate-per-block-build.sh --fast --tip HEAD --changed-from a7cb0ab18`: 2/2 (block 03, tip).
* `scripts/validate-set.sh`: checksums OK, strict `git am` of 16 patches on a fresh `a691b2345`
  tarball, applied tree `d1fee048a` == `release.json.tree`, Windows portability scan clean.
* GPU faults: none in the run windows.
