# gfx1201 port — S11: per-arch tuning defaults (2026-09-21)

Session S11 of `gfx1201-porting.md` §13 (§7 point 3).  **Host-side only; no device code changed.**
**Headline: every `mmb_*` tunable is now `env override || arch default` selected from the device cc, and
the S10 dense geometry is a per-arch constant in that same table — so the three RDNA arches (gfx1151 /
gfx1201 / gfx1100) can hold different values without the user setting anything, and a profiler run can
record what was actually used.**

## 1. Why

The plan §7.3 requires per-arch constants selected by `ggml_cuda_info().devices[0].cc`.  Only
`mmb_wtype_mask()`, `mmb_dense_flag()`/`mmb_dense_tmask()` and `mmb_enabled()` did this; every other
tunable was a hard-coded gfx1151 value behind a `getenv`.  Two consequences:

* no arch could differ without a wall of env vars, and
* the §12.5 problem: the gates are lazy host `getenv`s, so `rocprofv3` **cannot see them** — a trace
  cannot tell you which policy produced it.

## 2. The table

```cpp
enum mmb_geom_id { MMB_GEOM_GFX11_SPLIT = 0, MMB_GEOM_R4_256x128 = 1 };

struct mmb_arch_cfg {
    int min_t = 512, glu_thresh = 32, routed_thresh = 32, tall_mode = 2;
    int tiny_m = 1, tiny_tt = 1;
    int f32split_mode = 1, f32split_min_k = 0, f32split_min_m = 128;
    int cache_max = 4, shadow_mode = 0, shadow_cap_mb = 6144, iq3xxs_glu = 0;
    int hc16 = 0, down16 = 0, gatemix = 0, blk16 = 0, res16 = 0;
    int glu = 1, bf16w = 1;
    int dense_geom = MMB_GEOM_GFX11_SPLIT;
};

static mmb_arch_cfg mmb_arch_defaults(const int cc) {
    mmb_arch_cfg c;                                   // the gfx1151 / pre-S11 values
    if (GGML_CUDA_CC_IS_RDNA4(cc)) c.dense_geom = MMB_GEOM_R4_256x128;   // S10
    // TODO(S12): routed/GLU thresholds, tall/tiny-M, f32split_*, cache still need a per-arch measurement
    return c;
}
static const mmb_arch_cfg & mmb_cfg() { static const mmb_arch_cfg c = mmb_arch_defaults(ggml_cuda_info().devices[0].cc); return c; }
```

Every accessor is now `getenv("GGML_CUDA_MMB_X") ? atoi(...) : mmb_cfg().x`, so the env var remains the
A/B override and the arch default is what a bare `GGML_CUDA_MMB=1` resolves to.

**The RDNA4 row carries no invented tuning.**  The only field with an RDNA4-specific value is
`dense_geom`, and that is the measured S10 geometry.  Everything else is the gfx1151 value, which is
exactly the pre-S11 behaviour — the table's job in this session is to make the per-arch *mechanism*
exist and to mark the fields that still need a per-arch measurement (`TODO(S12)`).  Inventing numbers
for gfx1201 that were never measured would be worse than the env-only state.

The dense geometry also stops being an inline `GGML_CUDA_CC_IS_RDNA4(...)` test in
`ggml_cuda_mul_mat_mmb` and becomes `c.dense_geom`, so the arch policy for the tile is visible in the
same place as every other knob.

## 3. The diagnostic

`GGML_CUDA_MMB_CFG=1` (or `GGML_CUDA_MMB_LOG=1`) prints the resolved config exactly once:

```
MMB_CFG cc=0x1001201 dense_geom=1 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
        f32split=1(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=0 down16=0 gatemix=0
        blk16=0 res16=0 glu=1 bf16w=1 iq3xxs_glu=0
```

That is the answer to the §12.5 concern: a trace can be accompanied by the config that produced it
without any env bookkeeping.  (Verified: `cc` prints `0x1001201` — `GGML_CUDA_CC_IS_RDNA4` handles the
flag bits.)

## 4. Verification

| gate | result |
|---|---|
| build | `ggml-hip` + `llama-cli` + `llama-bench`, 0 errors (this change cannot touch device code) |
| same-seed greedy, 27B UD-IQ3_S | `42cdf36d0633` for `MMB=1`, `MMB=1 GGML_CUDA_MMB_GLU_THRESH=16`, and `MMB=0` — **unchanged** (a threshold is a perf knob, so the hash must not move) |
| **gfx1151 device asm** | compile `mmb.cu` for gfx1151 before/after and split on `.type`/`.size`: **90 kernels, 0 differing instructions**, 0 new, 0 gone — the kernel set is *byte-unchanged* (S11 is host-only, unlike S10 which added 11 kernels) |
| 27B UD-IQ3_S interleaved r=5 | OFF 928.50 / 852.99 -> ON 933.04 / 856.91 = **+0.45 % pp8192 / +0.46 % pp32768** (S10: +0.52 / +0.46) — the S10 win is preserved |
| config dump | `MMB_CFG` line above |

The gfx1151 exit gate ("numbers unchanged") is argued from two facts: every non-RDNA4 field of the
table is the previous hard-coded value, and the compiled gfx1151 kernel set is identical.  There is no
gfx1151 box in this session, so no gfx1151 *runtime* number is claimed.

## 5. Patch layout

Landed as **patch 8** (tree `e5dc99b4d59dc5244de879825d1e8aa025b76263`, `git am` **8/8** verified on a
fresh r12 worktree) because `mmb.cu` is shared by patches 1/3/4/6/7 and there is no single theme to fold
it into.  Patches 6, 7 and 8 are a chain on `mmb.cu`; patch 8 is the last word on the RDNA4 policy
table, patch 7 on the geometry, patch 6 on the scope split.

## 6. Handing on

* **S12** should fill the `TODO(S12)` fields with measured RDNA4 values (routed/GLU thresholds,
  `tall_mode`, `tiny_m*`, `f32split_*`, `cache_max`) — now a one-line change per field instead of an
  env-var story — and, more urgently, re-decide the **routed MoE default** (the S7 +6.7 % does not
  reproduce; `gfx1201-s10-dense-geometry.md` §6).
* S13 (HC16 producers) gets its `hc16`/`down16`/`gatemix`/`blk16`/`res16` fields for free.
* gfx1100 (S15/handover) inherits the table: its row is currently the gfx1151 row, and its job is to
  give itself a `GGML_CUDA_CC_IS_RDNA3_0` arm once the tile/threshold constants are re-tuned there.
