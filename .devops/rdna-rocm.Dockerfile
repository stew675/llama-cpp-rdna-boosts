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
#   * an RDNA-only AMDGPU_TARGETS default (gfx1100/1151/1200/1201); runtime
#     dispatch means one binary serves every supported GPU family;
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
ARG IMAGE_URL=https://github.com/MrDrMcCoy/llama-cpp-rdna-boosts
ARG IMAGE_SOURCE=https://github.com/MrDrMcCoy/llama-cpp-rdna-boosts

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
# (gfx1100), RDNA3.5 (gfx1150/1151) and RDNA4 (gfx1200/1201). Trim this list
# to your own GPU for a much faster build.
ARG ROCM_DOCKER_ARCH='gfx1100;gfx1151;gfx1200;gfx1201'

# Set ROCm architectures (also consumed by the ROCm device library toolchain).
ENV AMDGPU_TARGETS=${ROCM_DOCKER_ARCH}

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    curl \
    libgomp1

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

# GGML_HIP_RCCL turns block 12's hybrid all-reduce on for non-RDNA4 pairs and
# needs librccl from the base image (find_package(rccl REQUIRED)).
RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON \
        -DGGML_HIP_RCCL=ON \
        -DAMDGPU_TARGETS="$ROCM_DOCKER_ARCH" \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j$(nproc)

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

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/MrDrMcCoy/llama-cpp-rdna-boosts
ARG IMAGE_SOURCE=https://github.com/MrDrMcCoy/llama-cpp-rdna-boosts
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
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
