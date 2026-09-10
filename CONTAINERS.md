# Prebuilt ROCm containers (GHCR)

The workflow [`.github/workflows/docker-ghcr.yml`](.github/workflows/docker-ghcr.yml)
builds llama.cpp with this repo's patch set applied and publishes ROCm
container images to the GitHub Container Registry.

It does **not** rebuild or ship the patches themselves; it re-creates the
patched tree the same way the consumer workflow does:

1. download upstream `ggml-org/llama.cpp` at the fork point (`9113cc188`)
   as a tarball (no full history),
2. `git init` + one base commit, then `scripts/apply-all.sh` applies
   `patches/0001..0015` with strict `git am`,
3. build with [`.devops/rdna-rocm.Dockerfile`](.devops/rdna-rocm.Dockerfile),
   an adaptation of upstream llama.cpp's `.devops/rocm.Dockerfile`
   (adds `-DGGML_HIP_RCCL=ON`, an RDNA-only `AMDGPU_TARGETS`, and a
   `ROCM_IMAGE_SUFFIX` switch for the `-complete`/`-full` base images),
4. push `full`, `light` and `server` images to GHCR.

## Images

Registry path: `ghcr.io/<owner>/<repo>` (here
`ghcr.io/mrdrmccoy/llama-cpp-rdna-boosts`).

| ROCm | base image | tags |
|------|-----------|------|
| 7.2  | `rocm/dev-ubuntu-24.04:7.2.4-complete`  | `rocm-7.2`, `server-rocm-7.2`, `light-rocm-7.2`, `full-rocm-7.2` |
| 7.14 | `rocm/dev-ubuntu-24.04:7.14.1-full`     | `rocm-7.14`, `server-rocm-7.14`, `light-rocm-7.14`, `full-rocm-7.14` |
| 10.0 | `rocm/dev-ubuntu-24.04:10.0.0-full`     | `rocm-10.0`, `server-rocm-10.0`, `light-rocm-10.0`, `full-rocm-10.0`, `latest` |

Each tag also has an immutable `<tag>-9113cc188` variant pinned to the fork
point. `rocm-<version>` is an alias of `server-rocm-<version>` (the serving
image); `latest` points at the newest ROCm (10.0) server image.

The binaries are built for `gfx1100;gfx1151;gfx1200;gfx1201` (RDNA3 /
RDNA3.5 / RDNA4) with runtime dispatch, so one image serves every supported
GPU family. `server` exposes the HTTP API on `8080`, `light` is CLI-only,
`full` adds the Python conversion tooling.

## Running

```bash
# server, all RDNA families, ROCm 7.14
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v ~/models:/models -p 8080:8080 \
  ghcr.io/mrdrmccoy/llama-cpp-rdna-boosts:rocm-7.14 \
  -m /models/Qwen3.5-4B-Q8_0.gguf -ngl 99 -sm tensor -mg 0
```

If the package is private, authenticate first:
`echo "$GHCR_TOKEN" | docker login ghcr.io -u <user> --password-stdin`.

## Triggering

- `workflow_dispatch` — pick the ROCm release lines (`7.2 7.14 10.0` by
  default) and whether to push; unchecking push runs a build-only
  validation.
- push to `main` (and the `ci/docker-ghcr` development branch).
- weekly `schedule` (the images are expensive, so no per-push rebuild).

The fork point is the `FORK_POINT` env var in the workflow; bump it (and the
patch set) together when the delivery is re-based.

## Local build

`docker`/`podman` can build the same image without the workflow:

```bash
# 1. patched source tree
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 9113cc188
bash <this-repo>/scripts/apply-all.sh .
# 2. build
docker build -f <this-repo>/.devops/rdna-rocm.Dockerfile \
  --build-arg ROCM_VERSION=7.14.1 \
  --build-arg AMDGPU_VERSION=7.14.1 \
  --build-arg ROCM_IMAGE_SUFFIX=full \
  --target server -t llama-rdna:rocm-7.14 .
```
