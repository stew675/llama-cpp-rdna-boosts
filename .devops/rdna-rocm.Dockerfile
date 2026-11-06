# syntax=docker/dockerfile:1
#
# rdna-boosts ROCm container image.
#
# Derived from upstream llama.cpp's .devops/rocm.Dockerfile
# (https://github.com/ggml-org/llama.cpp). The build context is a llama.cpp
# source tree with the rdna-boosts patch set applied (see the delivery repo's
# scripts/apply-all.sh); this file only carries the delivery-specific deltas:
#
#   * ROCM_IMAGE_SUFFIX selects the ROCm dev image flavour: the same file
#     builds against -complete (<= 7.2.x) and -full (>= 7.14) base images;
#   * -DGGML_HIP_RCCL=1, required by block 12's hybrid all-reduce on
#     non-RDNA4 GPUs (the complete/full ROCm images ship librccl);
#   * an AMDGPU_TARGETS default covering gfx1030 plus the RDNA3/3.5/4 set
#     gfx1150 is Strix Point, validated 2026-10-05, issue #88); runtime
#     dispatch means one binary serves every targeted GPU family.  gfx1030
#     (RDNA2: RX 6800/6900, V620) is **best-effort / not validated on
#     hardware** -- it is compiled in so RDNA2 owners get the generic boosts
#     instead of `invalid kernel file`, issue #158; see README
#     "Supported architectures" and CONTAINERS "RDNA2 (gfx1030)";
#   * OCI labels pointing at the delivery repo so GHCR links the package to
#     the repository.

ARG UBUNTU_VERSION=24.04

# Must generally match the container host's environment.
ARG ROCM_VERSION=7.2.4
ARG AMDGPU_VERSION=7.2.4

# Base image flavour: "complete" for ROCm <= 7.2.x, "full" for >= 7.14.
ARG ROCM_IMAGE_SUFFIX=complete

ARG BASE_ROCM_DEV_CONTAINER=docker.io/rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-${ROCM_IMAGE_SUFFIX}

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
# PROVENANCE (issue #272 follow-up): APP_VERSION is the fork point and is ALSO the web UI's build number,
# so it cannot name the release.  These carry the release identity instead: `RELEASE_TAG` is
# `release.json .release` (e.g. v16-6050b467c-r222) and `RELEASE_TIP` its canonical tip commit, which is
# what a reporter needs to say WHICH release an image carries.  A reporter on issue #272 could not tell
# whether the image he tested was r218 or r219, because version/revision both named only the fork point.
ARG RELEASE_TAG=N/A
ARG RELEASE_TIP=N/A
ARG RELEASE_BLOCKS=N/A
ARG IMAGE_URL=https://github.com/stew675/llama-cpp-rdna-boosts
ARG IMAGE_SOURCE=https://github.com/stew675/llama-cpp-rdna-boosts

ARG NODE_VERSION=24

FROM docker.io/node:$NODE_VERSION AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

### Build image
FROM ${BASE_ROCM_DEV_CONTAINER} AS build

# Unless otherwise specified, we make a fat build. This is tied to the
# rocBLAS/hipBLASLt supported archs; the rdna-boosts set supports RDNA3
# (gfx1100), RDNA3.5 (gfx1150/1151) and RDNA4 (gfx1200/1201), and carries
# RDNA2 (gfx1030) best-effort (issue #158; see README/CONTAINERS). Trim this
# list to your own GPU for a much faster build.
ARG ROCM_DOCKER_ARCH='gfx1030;gfx1100;gfx1150;gfx1151;gfx1200;gfx1201'

# Set ROCm architectures (also consumed by the ROCm device library toolchain).
ENV AMDGPU_TARGETS=${ROCM_DOCKER_ARCH}

# ccache is the sanctioned build-speed answer (WORKLOG 260918.007): 4.12.3
# caches the ROCm clang HIP *device* objects, so a rebuild of unchanged sources
# replays the compiler's own objects (bit-identical).  A static upstream binary
# is installed so the container runs the same version the maintainer's host
# validates, rather than Ubuntu's older ccache.  The cache lives in a build
# cache-mount (see the cmake RUN below), never in an image layer, so it neither
# enlarges nor changes the published images; `podman system prune` clears it and
# CCACHE_MAXSIZE bounds it.
#
# `ccache -z` before the build: the store outlives a release, so a bare trailing
# `ccache -s` reports the *lifetime* hit rate (a ~50 % average across many
# releases) and reads like a per-build miss.  Zeroing makes the logged rate the
# rate of THIS build, which is the number worth reading (a no-change rebuild is
# ~99.9 % hits; a release that touches a common HIP header legitimately misses
# every HIP TU).
ARG CCACHE_VERSION=4.12.3
ENV CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=20G

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    curl \
    libgomp1 \
    xz-utils \
    && curl -fsSL -o /tmp/ccache.tar.xz \
        "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz" \
    && tar -xJf /tmp/ccache.tar.xz -C /tmp \
    && install -m 0755 "/tmp/ccache-${CCACHE_VERSION}-linux-x86_64/ccache" /usr/local/bin/ccache \
    && rm -rf /tmp/ccache.tar.xz "/tmp/ccache-${CCACHE_VERSION}-linux-x86_64" \
    && ccache --version | head -1

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

# Build.  GGML_HIP_RCCL turns block 12's hybrid all-reduce on for non-RDNA4
# pairs and needs librccl from the base image (find_package(rccl REQUIRED)).
# The CMAKE_*_COMPILER_LAUNCHER flags mirror ~/bin/build-llama-rocm-714; the
# /root/.ccache cache-mount keeps ccache's store across builds, so a failed or
# partial build (or a later release whose sources mostly match) does not pay the
# full recompile again.
RUN --mount=type=cache,target=/root/.ccache,sharing=locked \
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON \
        -DGGML_HIP_RCCL=ON \
        -DAMDGPU_TARGETS="$ROCM_DOCKER_ARCH" \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && ccache -z \
    && cmake --build build --config Release -j$(nproc) \
    && ccache -s

RUN mkdir -p /app/lib \
    && find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r conversion /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ${BASE_ROCM_DEV_CONTAINER} AS base

# ROCm >= 7.14 dev images install the runtime under /opt/rocm/core-<ver>/lib
# (exposed as /opt/rocm/lib) and do not register it with the dynamic loader:
# there is no /etc/ld.so.conf.d entry and the images set no LD_LIBRARY_PATH.
# Without the line below libggml-hip.so cannot resolve libhipblas / librocblas /
# libamdhip64 / librccl, the HIP backend fails to dlopen and llama.cpp falls
# back to "no usable GPU found ... compiled without GPU support".  Harmless on
# the <= 7.2 -complete images, which register the path via ldconfig.
ENV LD_LIBRARY_PATH=/opt/rocm/lib

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG RELEASE_TAG=N/A
ARG RELEASE_TIP=N/A
ARG RELEASE_BLOCKS=N/A
ARG IMAGE_URL=https://github.com/stew675/llama-cpp-rdna-boosts
ARG IMAGE_SOURCE=https://github.com/stew675/llama-cpp-rdna-boosts
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$RELEASE_TAG \
      org.opencontainers.image.revision=$RELEASE_TIP \
      io.github.stew675.rdna-boosts.fork-point=$APP_VERSION \
      io.github.stew675.rdna-boosts.blocks=$RELEASE_BLOCKS \
      org.opencontainers.image.title="llama.cpp (rdna-boosts)" \
      org.opencontainers.image.description="llama.cpp patched with the rdna-boosts RDNA feature/perf set (ROCm)" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

RUN apt-get update \
    && apt-get install -y libgomp1 curl ffmpeg \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

### Full
FROM base AS full

COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    git \
    python3-pip \
    python3 \
    python3-wheel \
    && pip install --break-system-packages --upgrade setuptools \
    && pip install --break-system-packages -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama /app/full/llama-cli /app/full/llama-completion /app/

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama /app/full/llama-server /app/

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
