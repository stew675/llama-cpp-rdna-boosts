# BASELINE - provenance and drift policy

## Baseline

All patches in `patches/` are generated against **llama.cpp upstream master at
`7584430716ee229751771ed0d6bbcb780d105eeb`** ("tests : disable DOTS3NOTE arch test
for WebGPU (#27654)"), the parent commit of the fork branch `chunked-gdn` in
`stew675/llama.cpp`.

- Upstream range this branch was generated and verified against:
  `7584430716ee229751771ed0d6bbcb780d105eeb` (single point; the fork has not been
  rebased past this tip).
- Verified: applying all 10 patches in manifest order to a fresh checkout of the
  baseline SHA reproduces the fork branch tree **byte-identical for every
  production file**. The only intentional divergence is the relocation of block
  10's decode-shape test cases inside `tests/test-backend-ops.cpp` (see
  `MANIFESTS.md`, block 10 note).
- The fork's full 48-commit branch (`chunked-gdn`, 9 commits pushed) remains the
  source of truth: `git@github.com:stew675/llama.cpp.git`.

## Per-block provenance (source commits on `chunked-gdn`)

| patch | source commit SHAs (history order) |
|-------|-------------------------------------|
| `01-adaptive-mtp.patch` | `87ad1db26` `b56926039` `d0d7ff27e` `8d70e21f5` `0cf87e989` |
| `02-chunked-gdn.patch` | `876ef1f0b` `5d2090e96` `b220647b1` `a4982afa2` `659f94987` `2a1e5c5a8` `1da07e19b` `77d51ee28` `abfa24265` `be46c7621` `3441b7d40` `246136122` `05cab3c41` |
| `03-bf16-kv-cache.patch` | `5485e79e4` `b98265cfd` `07767a88a` `ef3673358` `b6bfa422e` `5e6072558` `bd5bf0ea3` `d33ce1adf` |
| `04-wmma-flash-attn.patch` | `beaf69fb6` |
| `05-bit-identical-decode-cpu.patch` | `89ac4ba1f` |
| `10-fused-core.patch` | `14e5dd427` `d0e6119a7` `333e8f950` `c11752b18` `10e016df4` `85387ba3a` `8e1300159` `ac08b6d85` `a84112dcf` `9b4554626` `555e79ab2` `00f53040f` `ec09a818e` `bb64338f9` `4c0440841` `3d65d7979` |
| `06-gfx1151-mmvq-table.patch` | `5b320ed94` |
| `07-host-buffer-revert.patch` | `edb8d44c0` |
| `08-meta-device-wrapper-skip.patch` | `32670eec8` |
| `09-q6k-mmvq-vdr2.patch` | `cd35abd19` |
| `rdna-boosts-all.patch` (repo root) | all 48 commits of `758443071..chunked-gdn` |

## Drift policy

The patches are static. If a patch fails to apply against a newer upstream
master:

1. Try `git apply -3` (3-way merge against the baseline blobs).
2. If 3-way fails, rebase the failing hunks manually against the current master
   and continue.
3. Do NOT hand-edit the committed patches as the permanent fix: when the fork is
   rebased onto a newer upstream tip, regenerate the whole set with
   `scripts/make-patches.sh` and cut a NEW `baseline/<new-sha>` branch here,
   leaving the old branches as the known-good sets for older upstream versions
   (rule of thumb: cut a new baseline branch whenever more than one block needs
   manual re-base hunks).

`MANIFESTS.md` records the upstream range (baseline SHA .. last verified upstream
SHA) per branch so consumers can match their upstream version.
