# Prebuilt ROCm containers (GHCR)

The ROCm container images are built and published by the off-host worker
(`radiance`) running [`scripts/container-watch.sh`](scripts/container-watch.sh),
which invokes [`scripts/build-containers.sh`](scripts/build-containers.sh)
(`ghcr.io/stew675/llama-cpp-rdna-boosts`).  A tag-push CI build took ~5 h for
two ROCm lines, and because a failed/timed-out run wrote no registry cache,
every retry started cold; the same build on a worker is
minutes-to-tens-of-minutes and fails with a usable log.  The old matrix lives on in
[`.github/workflows/docker-ghcr.yml`](.github/workflows/docker-ghcr.yml) as a
**manual `workflow_dispatch` fallback only**.

The script does **not** rebuild or ship the patches themselves; it re-creates
the patched tree the same way the consumer workflow does:

1. download upstream `ggml-org/llama.cpp` at the fork point (read from
   `release.json`; currently `a55e952b8`)
   as a tarball (no full history) — unless the canonical tip
   (`release.json.tip`) is already in the local fork checkout, in which case it
   materialises that tip directly and verifies its git tree,
2. `git init` + one base commit (`git add -A -f`, so upstream-tracked files
   that match `.gitignore` are kept and the base tree is canonical), then
   `scripts/apply-all.sh` applies `patches/0000..0015` with strict `git am`,
3. write the build-context `.dockerignore` (from `.devops/dockerignore`) and
   build with [`.devops/rdna-rocm.Dockerfile`](.devops/rdna-rocm.Dockerfile),
   an adaptation of upstream llama.cpp's `.devops/rocm.Dockerfile`
   (adds `-DGGML_HIP_RCCL=ON`, an RDNA-only `AMDGPU_TARGETS`, and a
   `ROCM_IMAGE_SUFFIX` switch for the `-complete`/`-full` base images),
4. tag `full`, `light` and `server` per ROCm line with the moving tag, the
   fork-point tag and the **release** tag (`<tag>-<release>`, r222), pass the
   release provenance as build args (`RELEASE_TAG`/`RELEASE_TIP`/`RELEASE_BLOCKS`
   become the OCI labels) and, with `--push`, push them to GHCR.

## Images

Registry path: `ghcr.io/<owner>/<repo>` (here
`ghcr.io/stew675/llama-cpp-rdna-boosts`).

| ROCm | base image | tags |
|------|-----------|------|
| 7.2  | `rocm/dev-ubuntu-24.04:7.2.4-complete`  | `rocm-7.2`, `server-rocm-7.2`, `light-rocm-7.2`, `full-rocm-7.2` *(manual dispatch only)* |
| 7.14.1 | `rocm/dev-ubuntu-24.04:7.14.1-full`   | `rocm-7.14.1`, `server-rocm-7.14.1`, `light-rocm-7.14.1`, `full-rocm-7.14.1` |
| 10.1   | `rocm/dev-ubuntu-24.04:10.1.0-full`   | `rocm-10.1`, `server-rocm-10.1`, `light-rocm-10.1`, `full-rocm-10.1`, `latest` |

**ROCm 7.2 is not built by default.**  Its `-complete` base image compiles several times more slowly
than the `-full` lines, and from release r10 onward its full image no longer fitted the 6 h build
`timeout-minutes`; a timed-out build also never wrote the registry cache, so every retry started cold
and failed the same way -- which blocked the (then build-chained) `release` job and left every tag
release uncut.  The 7.2 line is still buildable with `--rocm 7.2` (or a manual `workflow_dispatch`
asking for `7.2`), but expect it to be slow and possibly to time out.

Each tag also has an immutable `<tag>-<fork-point>` variant pinned to the fork
point (currently `<tag>-6050b467c`; earlier releases used `<tag>-a55e952b8`,
`<tag>-84e76d8a2`, `<tag>-ebbb18522`, `<tag>-d1d3c3396` and `<tag>-790cf51aa`)
and a `<tag>-<release>` variant pinned to the RELEASE (e.g.
`ghcr.io/stew675/llama-cpp-rdna-boosts:server-rocm-10.1-v16-6050b467c-r222`).
`rocm-<version>` is an alias of `server-rocm-<version>` (the serving image, with
its own `-<release>` variant); `latest` points at the newest ROCm (10.1) server
image.

**Provenance: pin the release, and read the labels (r222, issue #272 follow-up).**
The fork-point variants cannot say *which release* an image carries, because one
fork point serves many releases, and the moving tags (`rocm-10.1`, `latest`) are
overwritten every release.  A reporter on issue #272 could not tell whether the
"newer fork image" he had tested was r218 or r219.  So every image also carries
`-<release>` tags, and its labels name the release instead of only the fork point:

```
$ podman inspect --format '{{json .Labels}}' ghcr.io/stew675/llama-cpp-rdna-boosts:rocm-10.1 | python3 -m json.tool
{
  "org.opencontainers.image.version": "v16-6050b467c-r222",     # the release (release.json .release)
  "org.opencontainers.image.revision": "80bfb80be...",          # its canonical tip (.tip)
  "io.github.stew675.rdna-boosts.fork-point": "6050b467c",      # the fork point (.base)
  "io.github.stew675.rdna-boosts.blocks": "16",
  "org.opencontainers.image.created": "...",
  ...
}
```

Images built before r222 carry the fork point in `version`/`revision` and no
release label; treat their release as unknown.  When reporting a container
issue, quote the `-<release>` tag or the `org.opencontainers.image.version`
label, never a digest alone.

The binaries are built for `gfx1030;gfx1100;gfx1150;gfx1151;gfx1200;gfx1201`
(RDNA2 best-effort / RDNA3 / RDNA3.5 / RDNA4) with runtime dispatch, so one
image serves every targeted GPU family. `server` exposes the HTTP API on
`8080`, `light` is CLI-only, `full` adds the Python conversion tooling.

### Strix Point (gfx1150)

`gfx1150` (Strix Point: Ryzen AI 9 HX 370, Radeon 890M) is part of the target
set.  It is the same RDNA3.5 family as `gfx1151` but different silicon, and it
was added after a validation on a Strix Point host (issue #88): a
`gfx1150;gfx1151` build is clean, the same-seed greedy text is byte-identical
between the native `gfx1150` code and the `gfx1151` code, and `GET_ROWS`
220/220, `MUL_MAT_ID` 931/931, `FLASH_ATTN_EXT` 6358/6358, `FLASH_ATTN_QSA`
26/26, `INDEXER_TOPK` 3/3, `HC_MIX` 30/30, `GATED_DELTA_NET` 46/46 and
`RMS_NORM` 51/51 are green.

Images published **before this change** (for example `server-rocm-10.0` built at
`2dbece4`) carry no `gfx1150` code object and fail on the first kernel launch
with:

```
ggml_cuda_compute_forward: GET_ROWS failed
ROCm error: device kernel image is invalid
  current device: 0, in function ggml_cuda_compute_forward at .../ggml/src/ggml-cuda/ggml-cuda.cu:3172
```

For those, either build from source with `gfx1150` in the target list, or make
the runtime present the device as `gfx1151` with
`HSA_OVERRIDE_GFX_VERSION=11.5.1`:

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri --group-add video \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -v ~/models:/models -p 8080:8080 \
  ghcr.io/stew675/llama-cpp-rdna-boosts:rocm-7.14.1 \
  -m /models/<model>.gguf -ngl 999
```

(compose equivalent: `environment: [HSA_OVERRIDE_GFX_VERSION=11.5.1]`.)  The
override is a workaround for the older images, not a substitute for the native
target.  On the validation host `llama-bench` (9B Q8_0) measured 577.27 vs
580.30 t/s `pp512` and 10.61 vs 10.65 t/s `tg128` for native `gfx1150` vs the
`gfx1151` override, within noise.  Reported and reproduced in
[#88](https://github.com/stew675/llama-cpp-rdna-boosts/issues/88).

> **Toolchain caveat (7.2).**  ROCm 7.14.1 is the toolchain the delivery's claims
> and validation records are measured on.  ROCm 7.2 has been **reported** (not yet
> reproduced here) to break greedy purity where 7.14 is clean, see the "Separate,
> untriaged" note in `TODO.md` item 22; it was also dropped from the automatic release
> matrix because its image build outgrew the runner timeout (see the note above).
> Prefer `rocm-7.14.1` (or `rocm-10.1`) for speculative decoding and for anything
> compared against the recorded hashes; if a 7.2 run disagrees with a 7.14 run,
> suspect the toolchain before the delivery.

The ROCm `>= 7.14` `-full` base images do not register `/opt/rocm/lib` with the
dynamic loader (no `/etc/ld.so.conf.d` entry, no `LD_LIBRARY_PATH`), so the
image sets `LD_LIBRARY_PATH=/opt/rocm/lib` in its `base` stage — without it the
HIP backend cannot dlopen and llama.cpp reports "no usable GPU found".

### RDNA2 (gfx1030) — best-effort, externally validated

`gfx1030` (Navi 21: RX 6800/6800 XT/6900 XT, Radeon Pro V620/W6800) is
**compiled into the image** so RDNA2 owners get the generic, runtime-dispatched
paths instead of a first-kernel `ROCm error: invalid kernel file` (issue #158).
It is **not** a supported target: the maintainer has no usable RDNA2
device, so there are no performance claims and no purity gates for it.  Treat an
RDNA2 result as informational, never as a delivery gate.

It is, however, **no longer unvalidated**: TODO #66 was closed 2026-10-20 on a Radeon Pro V620
(`gfx1030`) run — `test-backend-ops` `FLASH_ATTN_EXT` 6389/6389 and `MUL_MAT` 2444/2444, plus coherent
`-fa 1` generation in a 3-card mixed `gfx1100`+`gfx1030` `-sm tensor` split to 32K depth
(`archive/work/rdna2/RESOLUTION.md`, `WORKLOG.md` 261020.012).  That validates the *platform*, not a
performance number, and it does not add an RDNA2 gate to this repo.

What applies / what does not:

* The RDNA3/4-only features fall back at runtime by design: `mmb_enabled()`
  and the QSA3 support predicate return false for RDNA2, the dp4a / tile-FA
  fallbacks are upstream, and the generic wins (scheduling, host-resident MoE,
  k-quant decode, prefill MMQ) do apply.
* **r76 folds `@slavap`'s tuned RDNA2 paths** (PR #165, from
  `sixvolts/llama-halo-hybrid`): the tile kernel gets a per-DKQ gfx1030 config
  table, gfx1030 launch caps and a WGP-aware occupancy compute (HIP's query
  assumes a quarter of the RDNA2 register file and reported 0 for the tuned
  configs), the partial-batch load branch fixes the kv=113 no-flat-scratch fault
  (`-fa on` is now usable), and the wide kernels read Q from a pre-converted
  global half2 buffer (+~30 % on the 512-column kernel).  The k-quant path gets
  the Navi21 `MMVQ_PARAMETERS_NAVI21` rows/warps table and the MMQ float scale.
  **All of it is gated to RDNA2** (`#if defined(RDNA2)` /
  `GGML_CUDA_CC_IS_RDNA2`), so gfx1100/gfx1151/gfx1201 keep their previous
  codegen and numerics.
* Two WMMA helper shims had to gain an RDNA2 compile branch for the set to
  build at all (`mmb.cu`, `fattn-qsa3.cu`).  They are now `#if defined(RDNA4)
  ... #elif defined(RDNA3) ... #else <no-op> #endif`, so RDNA2 compiles a dead
  arm that is never launched; RDNA3/RDNA4 codegen is byte-identical.
* **BLAS routing is covered.**  rocBLAS in the `7.14.1-full` and `10.1.0-full`
  base images carries `gfx1030` Tensile kernels (verified), so the `cublas`
  fallback works.  hipBLASLt does **not** ship `gfx1030`, but llama.cpp uses
  rocBLAS unless `ROCBLAS_USE_HIPBLASLT` is set — which this delivery does not
  set.
* **`-fa on` should now be safe on RDNA2** (the kv=113 partial-batch fault is
  fixed), but it remains unvalidated by the maintainer; slavap's V620 run of
  `test-backend-ops -o FLASH_ATTN_EXT` is a full pass on the add-on tree.  If a
  long-context prompt still aborts, run `-fa off` and report it on issue #158.
* **The RDNA2 long-context prefill deficit is closed behind r208.**  Issue #214 reported the delivery
  14-19 % *slower* than stock at 120k on gfx1030 / gfx1031 (the reporter's RX 6700 XT), and @slavap's
  V620 reproduced it.  It was not the host side: the tile kernel's launch config and the checked tail's
  register pressure.  r208 folds three RDNA2-only changes into block 05 (the `q_global` threshold 16 -> 64,
  the D=256 `ncols2` cap lift, and a clamped in-bounds tail) which lift pp2048 @ d32768 from 1041.0 to
  1259.8 t/s (stock 1231.7) and tg256 @ d131072 from 50.9 to 62.4.  The mechanism, why the cap was stale
  and what keeps this RDNA2-only are in `patches/README.md` block 05 (r208) and `WORKLOG.md` 261103.013.
  **Still the submitter's numbers**: they were measured on a Radeon Pro V620 (`gfx1030`, Navi 21),
  not by the maintainer, and RDNA2 stays a best-effort target with no gate in this repo.

Validation path: build from source with `gfx1030` in the target list, or pull
this image, and report back on issue #158.  The tester checklist is
`archive/work/rdna2/BETA-TESTING.md`.

## Running

```bash
# server, all RDNA families, ROCm 7.14.1
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v ~/models:/models -p 8080:8080 \
  ghcr.io/stew675/llama-cpp-rdna-boosts:rocm-7.14.1 \
  -m /models/Qwen3.5-4B-Q8_0.gguf -ngl 99 -sm tensor -mg 0
```

If the package is private, authenticate first:
`echo "$GHCR_TOKEN" | docker login ghcr.io -u <user> --password-stdin`.

## Triggering and releases

Releases are tagged by hand and the images are built and pushed by the off-host
worker (see "Offload the image build to a worker" below); only the cheap asset
packaging stays in CI:

- **tag push** (`v<blocks>-<base>-r<N>`, the release identity) →
  [`.github/workflows/release.yml`](.github/workflows/release.yml) attaches
  `rdna-boosts-all.patch`, `patches.tar.gz`, `release.json` and `SHA256SUMS`
  to the GitHub Release (about a minute; no compiler, no Docker).  The tag is
  checked against `release.json.release` and the `v<blocks>-<base>-r<N>` shape.
- **off-host image build** → the `radiance` watcher (`scripts/container-watch.sh`)
  sees the new tag and runs `scripts/build-containers.sh --rocm "7.14.1 10.1"
  --push --prune` there.  For a manual build, run the same script on whichever box
  you prefer; either way it reads the tree from `release.json`, not from the tag.
  **The watcher runs the script as it exists in the TAG it is building**, so a
  fix to the script or the Dockerfile only reaches the images of a release whose
  tag already contains it.  To re-label an already-published release, run the
  build by hand from `main` on the worker (same command), holding the watcher's
  `flock` (`$HOME/.cache/rdna-containers/.lock`) so cron skips, then write the
  tag into `$HOME/.cache/rdna-containers/last-built` so the watcher does not
  rebuild the unlabelled images (r222 did exactly this).
- `docker-ghcr.yml` no longer runs automatically (no tag or schedule trigger);
  it is a manual `workflow_dispatch` fallback.

Ordinary commits to `main` (docs / `WORKLOG.md` / `benchmarks/`) run
[`.github/workflows/validate.yml`](.github/workflows/validate.yml) instead,
which applies the patch set and checks it against `release.json` in about a
minute.  Building the six images (2 ROCm lines x 3 targets) for a docs commit
was wasted runner time, and it let a docs push fail at `git am` when the
workflow's fork point had gone stale — the breakage this split exists to
prevent.

### Cutting a release

`release.json` is the single source of truth: the fork point, the canonical
upstream tree of that fork point, the canonical tip/tree of the applied set,
the block count, and the sha256 of every shipped artifact.  `apply-all.sh`,
`validate-set.sh` and the workflows all read it, so the fork point cannot
drift out of sync in one place while another stays stale.

To cut a release after a re-base or a block amendment:

```bash
# refresh the artifact hashes and stamp the release name (the git tag is the
# release identity; it MUST equal release.json.release)
./scripts/make-release.sh --release v<blocks>-<base-sha>-r<N>
# or, on a re-base, set all four metadata values together:
./scripts/make-release.sh \
  --release v<blocks>-<base-sha>-r1 \
  --base <new-base-sha> \
  --base-tree "$(git -C ~/llama.cpp rev-parse <new-base-sha>^{tree})" \
  --tip  <canonical-tip> \
  --tree "$(git -C ~/llama.cpp rev-parse <canonical-tip>^{tree})"

./scripts/validate-set.sh          # strict apply + tree/hash/checksum gate

# The images are built and pushed by the off-host worker once the tag is pushed
# (see "Offload the image build to a worker").  To smoke-test or to build on this
# box instead, run the script manually (do the full build on the worker during a
# release so it does not tie up the primary box):
./scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune

# freeze it: annotated tag on the commit that carries this release.json
git tag -a v<blocks>-<base-sha>-r<N> -m "rdna-boosts v<blocks>-<base-sha>-r<N>"
git push origin main v<blocks>-<base-sha>-r<N>
```

**Release naming.**  One tag per release, `v<blocks>-<base-sha>-r<N>` (e.g. `v15-6050b467c-r138`), where `N` is
that base's revision: `r1` is the release that lands the re-base and `r2`,
`r3`, ... each later release on the same base.  `release.json.release` must be
exactly the tag — `validate.yml` checks the manifest, and
[`release.yml`](.github/workflows/release.yml) refuses to package a tag that
disagrees with it or does not match `v<blocks>-<base-sha>-r<N>`.  Only a **tag push**
cuts a release; an image build never bumps the revision and never
creates a Release.  (The historical `v16-790cf51aa` tag predates the explicit
`-rN`; it is the `r1` of its base.  Two revisions were never tagged —
`v16-790cf51aa-r5` was built by a dispatch and `v16-d1d3c3396-r1` is the
re-base — so they exist only in `release.json` history.)

The `v*` tag push runs [`release.yml`](.github/workflows/release.yml), which
attaches `rdna-boosts-all.patch`, `patches.tar.gz`, `release.json` and
`SHA256SUMS` to the GitHub Release.  Consumers can pin the tag and verify the
checksums instead of tracking a moving `main`.  The tag's name is checked
against `release.json.release` before anything is packaged, so a release can
never ship under a name the manifest does not record.  The images themselves
are pushed separately by the off-host worker running `scripts/build-containers.sh`.

The `tree` field in `release.json` is the strongest check available: CI
rebuilds the patched source from a tarball and asserts the resulting git tree
is byte-for-byte the recorded canonical tree.  If it fails, the patch set
and/or the recorded base are not the delivery.

To build the images for a branch before tagging, run
`scripts/build-containers.sh` against `release.json` — it builds from the
canonical tree, so it does not need the branch to be merged or tagged.  The
manual fallback remains available via `workflow_dispatch` on
`.github/workflows/docker-ghcr.yml` (merge first, since GitHub only exposes it
once the workflow is on the default branch).

## Manual build

The normal release path is the off-host worker above.  This is the manual or fallback
path (and the fast smoke test) — build and push the release images without CI:

```bash
gh auth refresh -h github.com -s write:packages            # once; podman needs write:packages
gh auth token | podman login ghcr.io -u stew675 --password-stdin

./scripts/build-containers.sh --prepare-only               # reconstruct + verify the release tree
./scripts/build-containers.sh --rocm "7.14.1 10.1" --push # build + push the release images

# fast smoke: one ROCm line, one GPU arch, one target, no push
./scripts/build-containers.sh --rocm 10.1 --arch gfx1201 --targets server
```

`--src DIR` overrides the build-context tree (default `$HOME/llama-src`);
`--llama DIR` is the local fork checkout the canonical tip is borrowed from
(default `$HOME/llama.cpp`; when the tip is absent the script falls back to the
tarball + `apply-all.sh`).  `--arch LIST` trims `AMDGPU_TARGETS` for a fast
smoke build — the published image uses the Dockerfile's full RDNA set
(`gfx1100;gfx1150;gfx1151;gfx1200;gfx1201`).

### ccache

`.devops/rdna-rocm.Dockerfile` installs the upstream static ccache **4.12.3**
(the version the maintainer's host validates, not Ubuntu's older package) and
compiles with `-DCMAKE_{HIP,C,CXX}_COMPILER_LAUNCHER=ccache`, mirroring
`~/bin/build-llama-rocm`.  The cache lives in a `/root/.ccache` build
cache-mount, so it is never an image layer and does not change the published
images.  Measured single-arch (`gfx1201`, `light`): a cold compile is ~10 min;
a `--no-cache` rebuild replays from ccache in **~1m28s** (835/836 direct hits).
A release whose sources mostly match the previous one therefore recompiles only
the delta.  `podman system prune` clears the cache.

### Cleanup

After a successful push, pass `--prune` to remove the locally built
`ghcr.io/...` images and dangling build layers while **keeping the ROCm base
images and the ccache store** — re-pulling the ~20 GB base images on every
release is the waste a bare `podman system prune -a` causes:

```bash
./scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune
```

### Registry cleanup (GHCR)

GHCR keeps every manifest ever pushed: a multi-arch image is an index version
plus one version per architecture, and every time a moving tag advances the
manifest it left behind stays as an untagged version.  That is thousands of
unreachable versions per year.  On every `--push`, `build-containers.sh` now
prunes them with [`scripts/cleanup-ghcr.py`](scripts/cleanup-ghcr.py):

* **keep** any version created within `--keep-days` (default **7**, about one
  release cycle, so the previous base survives as a rollback), and
* **keep** any version carrying a `*-<base>` tag for the current release's base
  (`--keep-base`, from `release.json`), so the images just pushed can never be
  deleted,
* **keep** any *untagged* version created within an hour of a kept tagged one
  (a multi-arch index's platform children are pushed as separate untagged
  versions; today's images are single-manifest, so this is a guard for the
  future),
* **delete** everything else.

```bash
# normal release: cleanup is on with --push
./scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune

# opt out, or widen/narrow the window
./scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune --no-cleanup
./scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune --keep-days 14
```

The prune needs a token with the **`delete:packages`** scope (once):

```bash
gh auth refresh -h github.com -s delete:packages
```

A GitHub bulk-delete throttle (the secondary rate limit on destructive calls) is
reported as a warning, not a failure — the images are published either way, and
re-running finishes the prune.  Without the scope the cleanup is skipped with a
warning.  The tool is dry-run by default (`--apply` deletes); run it by hand any
time to see the plan, or to prune a different package (e.g. a defunct one):

```bash
./scripts/cleanup-ghcr.py --package llama-cpp-rdna-boosts --keep-days 7
./scripts/cleanup-ghcr.py --package llama-cpp-rdna-boosts --keep-days 7 --apply
```

---

## Offload the image build to a worker

Building the images ties up the primary box for tens of minutes per release.  A
second machine can poll for new release tags and build/push in the background via
`scripts/container-watch.sh`.

**Setup (once, on the worker).**  The worker needs `podman`, `gh`, `jq`, `python3`,
`curl`, `git` and `flock`, plus network access to `codeload.github.com` (the base
tarball), Docker Hub (the ROCm base images) and `ghcr.io`.  A clone of this repo
(HTTPS is enough: the worker only fetches) and, optionally, a `~/llama.cpp`
checkout.  Then:

```bash
# clone (or update) the delivery repo
git clone https://github.com/stew675/llama-cpp-rdna-boosts.git

# install the watcher OUTSIDE the checkout: it checks out release tags in the
# repo, and bash reads a running script incrementally, so it must not replace
# itself mid-run
install -m 755 ~/llama-cpp-rdna-boosts/scripts/container-watch.sh ~/bin/rdna-container-watch.sh

# authorization (interactive, once).  A classic PAT needs write:packages (push)
# and delete:packages (the prune); with a PAT, skip the refresh and use
#   echo ghp_... | gh auth login -h github.com --with-token
gh auth login -h github.com
gh auth refresh -h github.com -s write:packages -s delete:packages

# keep rootless podman usable from cron after logout/reboot:
sudo loginctl enable-linger "$(id -un)"

# poll every 10 minutes
( crontab -l 2>/dev/null; echo '*/10 * * * * $HOME/bin/rdna-container-watch.sh >> $HOME/.cache/rdna-containers/watch.log 2>&1' ) | crontab -
```

The watcher refreshes `~/.config/containers/auth.json` from `gh auth token` on every
run (`REGISTRY_AUTH_FILE`), so the credential is persistent and does not depend on the
login session's temporary `$XDG_RUNTIME_DIR/containers/auth.json`; no separate
`podman login` is needed once `gh` is authenticated.  `loginctl enable-linger` is
required so rootless podman still has a runtime dir when the user is logged out.

The watcher fetches tags, takes the newest release tag (`v<blocks>-<base>-r<N>`) **by tag creation date** — it does not
pin the block count, so a `v12-*` -> `v13-*` rename cannot make it go quiet (it did: r107/r108, fixed 2026-10-22)
(not a version sort: the base SHA is in the tag name, so a re-base to a lower-sorting
base would otherwise be missed), and if it differs from
`~/.cache/rdna-containers/last-built`, checks it out and runs that tag's
`scripts/build-containers.sh --rocm "7.14.1 10.1" --push --prune`.  `--prune`
removes the images just built plus dangling layers after a successful push, so the
worker's disk stays bounded; the ROCm base images (~20 GB each) and the ccache
store (bounded at `CCACHE_MAXSIZE=20G`) are deliberately kept.  A `flock` serialises
builds; the log is `~/.cache/rdna-containers/build-<tag>.log`.  `FORCE_TAG=<tag>` and
`BUILD_EXTRA=--no-cleanup` (or the other envs in the script header) override behaviour.

**Failure handling (r138, 2026-10-25).**  Three things the unattended path must get right:

* **A tag is never retried forever.**  The watcher records the attempt count in
  `~/.cache/rdna-containers/failed-<tag>` and stops retrying after `MAX_ATTEMPTS`
  (default **3**), logging `ERROR: <tag> has failed N times; not retrying`.  Without
  this, cron every 10 minutes re-runs a 10-40 minute build indefinitely while a release
  stays broken, pinning the worker.  A successful build deletes the counter; to revive a
  given-up tag, remove the `failed-<tag>` file (or use `FORCE_TAG`).  A *new* tag is
  always attempted.
* **A failed tag checkout aborts instead of building.**  `git checkout --force --detach
  <tag>` is checked, and `HEAD` is verified to be that tag's commit.  Unchecked, a
  checkout failure would build whatever tree was left behind and publish it under the
  **new** tag — a wrong image, silently.
* **Build logs are bounded.**  `KEEP_LOGS` (default **10**, one log is ~8 MB) prunes
  `build-*.log` after a successful build, so `~/.cache/rdna-containers` does not grow
  without bound (it had reached 67 logs / 768 MB).
To build a release immediately instead of waiting for cron:

```bash
FORCE_TAG=v15-6050b467c-r138 ~/bin/rdna-container-watch.sh
```

The primary box can keep using `scripts/build-containers.sh` directly for a manual or
smoke build; the watcher is only the unattended path.

**A re-pointed release tag must not wedge the watcher (r166, 2026-10-28).**  The release flow tags,
then lands a trailing record commit and moves the tag onto it, so a tag can legitimately be re-pointed
after the watcher has already fetched it.  The fetch is therefore `git fetch --tags --force`, and its
failure is logged.  Before that, a plain `--tags` fetch refused the moved tag (`! [rejected] … (would
clobber existing tag)`), exited 1, and under the script's `set -e` killed the watcher on that line with
**no log output at all** — r166 sat unbuilt through 40 minutes of silent one-second cron runs while the
log's last line was the previous release.  If a worker ever goes quiet, run
`git -C <repo> fetch --tags --force origin` by hand and read its exit status before suspecting podman.
