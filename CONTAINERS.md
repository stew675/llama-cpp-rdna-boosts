# Prebuilt ROCm containers (GHCR)

The workflow [`.github/workflows/docker-ghcr.yml`](.github/workflows/docker-ghcr.yml)
builds llama.cpp with this repo's patch set applied and publishes ROCm
container images to the GitHub Container Registry.

It does **not** rebuild or ship the patches themselves; it re-creates the
patched tree the same way the consumer workflow does:

1. download upstream `ggml-org/llama.cpp` at the fork point (read from
   `release.json`; currently `d1d3c3396`)
   as a tarball (no full history),
2. `git init` + one base commit (`git add -A -f`, so upstream-tracked files
   that match `.gitignore` are kept and the base tree is canonical), then
   `scripts/apply-all.sh` applies `patches/0000..0015` with strict `git am`,
3. build with [`.devops/rdna-rocm.Dockerfile`](.devops/rdna-rocm.Dockerfile),
   an adaptation of upstream llama.cpp's `.devops/rocm.Dockerfile`
   (adds `-DGGML_HIP_RCCL=ON`, an RDNA-only `AMDGPU_TARGETS`, and a
   `ROCM_IMAGE_SUFFIX` switch for the `-complete`/`-full` base images),
4. push `full`, `light` and `server` images to GHCR.

## Images

Registry path: `ghcr.io/<owner>/<repo>` (here
`ghcr.io/stew675/llama-cpp-rdna-boosts`).

| ROCm | base image | tags |
|------|-----------|------|
| 7.2  | `rocm/dev-ubuntu-24.04:7.2.4-complete`  | `rocm-7.2`, `server-rocm-7.2`, `light-rocm-7.2`, `full-rocm-7.2` |
| 7.14 | `rocm/dev-ubuntu-24.04:7.14.1-full`     | `rocm-7.14`, `server-rocm-7.14`, `light-rocm-7.14`, `full-rocm-7.14` |
| 10.0 | `rocm/dev-ubuntu-24.04:10.0.0-full`     | `rocm-10.0`, `server-rocm-10.0`, `light-rocm-10.0`, `full-rocm-10.0`, `latest` |

Each tag also has an immutable `<tag>-<fork-point>` variant pinned to the fork
point (currently `<tag>-d1d3c3396`; earlier releases used
`<tag>-790cf51aa`). `rocm-<version>` is an alias of `server-rocm-<version>` (the serving
image); `latest` points at the newest ROCm (10.0) server image.

The binaries are built for `gfx1100;gfx1151;gfx1200;gfx1201` (RDNA3 /
RDNA3.5 / RDNA4) with runtime dispatch, so one image serves every supported
GPU family. `server` exposes the HTTP API on `8080`, `light` is CLI-only,
`full` adds the Python conversion tooling.

The ROCm `>= 7.14` `-full` base images do not register `/opt/rocm/lib` with the
dynamic loader (no `/etc/ld.so.conf.d` entry, no `LD_LIBRARY_PATH`), so the
image sets `LD_LIBRARY_PATH=/opt/rocm/lib` in its `base` stage — without it the
HIP backend cannot dlopen and llama.cpp reports "no usable GPU found".

## Running

```bash
# server, all RDNA families, ROCm 7.14
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v ~/models:/models -p 8080:8080 \
  ghcr.io/stew675/llama-cpp-rdna-boosts:rocm-7.14 \
  -m /models/Qwen3.5-4B-Q8_0.gguf -ngl 99 -sm tensor -mg 0
```

If the package is private, authenticate first:
`echo "$GHCR_TOKEN" | docker login ghcr.io -u <user> --password-stdin`.

## Triggering and releases

The release pipeline is **tag-driven** (see `.github/workflows/docker-ghcr.yml`):

- push of a `v*` tag — the normal release path (build images, push them, and
  cut a GitHub Release carrying the packaged patch set),
- `workflow_dispatch` — pick the ROCm release lines (`7.2 7.14 10.0` by
  default) and whether to push; unchecking push runs a build-only validation.
  This path is **image-only**: it does not create a Release and must not be
  treated as a release (only a tag push bumps the revision),
- weekly `schedule` — rebuild the `rocm-*`/`latest` images (no release is cut).

Ordinary commits to `main` (docs / `WORKLOG.md` / `benchmarks/`) do **not**
trigger the container build; they run
[`.github/workflows/validate.yml`](.github/workflows/validate.yml) instead,
which applies the patch set and checks it against `release.json` in about a
minute.  Building the nine images (3 ROCm lines x 3 targets) for a docs commit
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
./scripts/make-release.sh --release v16-<base-sha>-r<N>
# or, on a re-base, set all four metadata values together:
./scripts/make-release.sh \
  --release v16-<base-sha>-r1 \
  --base <new-base-sha> \
  --base-tree "$(git -C ~/llama.cpp rev-parse <new-base-sha>^{tree})" \
  --tip  <canonical-block-15-tip> \
  --tree "$(git -C ~/llama.cpp rev-parse <canonical-block-15-tip>^{tree})"

./scripts/validate-set.sh          # strict apply + tree/hash/checksum gate

# freeze it: annotated tag on the commit that carries this release.json
git tag -a v16-<base-sha>-r<N> -m "rdna-boosts v16-<base-sha>-r<N>"
git push origin main v16-<base-sha>-r<N>
```

**Release naming.**  One tag per release, `v16-<base-sha>-r<N>`, where `N` is
that base's revision: `r1` is the release that lands the re-base and `r2`,
`r3`, ... each later release on the same base.  `release.json.release` must be
exactly the tag — `validate.yml` checks the manifest, and `docker-ghcr.yml`
now refuses to build a tag whose name disagrees with it.  Only a **tag push**
cuts a release; a `workflow_dispatch` or the weekly `schedule` rebuilds and
pushes the `rocm-*`/`latest` images but never bumps the revision and never
creates a Release.  (The historical `v16-790cf51aa` tag predates the explicit
`-rN`; it is the `r1` of its base.  Two revisions were never tagged —
`v16-790cf51aa-r5` was built by a dispatch and `v16-d1d3c3396-r1` is the
re-base — so they exist only in `release.json` history.)

The `v*` tag push runs the container matrix and then creates the GitHub
Release with `rdna-boosts-all.patch`, `patches.tar.gz`, `release.json` and
`SHA256SUMS` attached.  Consumers can pin the tag and verify the checksums
instead of tracking a moving `main`.  The tag's name is checked against
`release.json.release` before any image is built, so a release can never ship
under a name the manifest does not record.

The `tree` field in `release.json` is the strongest check available: CI
rebuilds the patched source from a tarball and asserts the resulting git tree
is byte-for-byte the recorded canonical tree.  If it fails, the patch set
and/or the recorded base are not the delivery.

To run the container build for a branch before tagging, use
`workflow_dispatch` (merge first, since GitHub only exposes it once the
workflow is on the default branch).

## Local build

`docker`/`podman` can build the same image without the workflow:

```bash
# 1. patched source tree (the fork point comes from release.json)
base="$(jq -r .base release.json)"
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$base"
bash <this-repo>/scripts/apply-all.sh .
# 2. build
docker build -f <this-repo>/.devops/rdna-rocm.Dockerfile \
  --build-arg ROCM_VERSION=7.14.1 \
  --build-arg AMDGPU_VERSION=7.14.1 \
  --build-arg ROCM_IMAGE_SUFFIX=full \
  --target server -t llama-rdna:rocm-7.14 .
```
