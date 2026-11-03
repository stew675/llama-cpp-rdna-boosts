// wip/host-expert-pinned-window: does holding the host expert set as 144 REGISTERED ranges slow down
// kernels that never touch it?
//
// The clue this probe chases: with the prefill staging fixed, `pinned` and `--host-experts window` issue the
// SAME kernels with identical launch counts, yet the window arm's kernels all run 7-21 % slower -- including
// `build_mmq_routed_descriptors` (+31 %) and `mm_ids_helper` (+18 %), which only touch ids/descriptors in
// VRAM and never read host memory.  A per-read cost cannot explain that; a process-global one can, and the
// obvious candidate is GPU page-table/TLB pressure from registering ~40 GiB of 4 KiB pages in 144 ranges
// (the BAR/VRAM side has 2 MiB pages).
//
// Method: one latency-bound VRAM kernel (dependent loads at pseudo-random offsets in a 256 MiB device
// buffer) timed (a) alone, (b) while a pinned master is streamed host->device on another stream, (c) while a
// registered file mapping is streamed, (d) the same mapping with MADV_COLLAPSE'd 2 MiB pages.  If (c) is
// slower than (b) the registration itself taxes unrelated kernels; if (d) recovers it, the fix is huge pages.
//
//   build: hipcc -O2 -std=c++17 --offload-arch=gfx1201 -o build/probe-pollution probe-pollution.cpp
//   run:   HIP_VISIBLE_DEVICES=0 ./build/probe-pollution [GGUF] [STREAM_MIB]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HIP_CHECK(call)                                                              \
    do {                                                                             \
        hipError_t e_ = (call);                                                      \
        if (e_ != hipSuccess) {                                                      \
            fprintf(stderr, "HIP error %s: %s\n", #call, hipGetErrorString(e_));      \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

// A latency-bound VRAM kernel: a dependent chain of loads at pseudo-random offsets, sized so the working set
// (256 MiB) is far larger than any L2 but far smaller than the device TLB could cover with 4 KiB pages.
static __global__ void latency_kernel(const unsigned * __restrict__ buf, unsigned n_words, unsigned * out,
                                      int iters, unsigned seed) {
    unsigned idx = seed + blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (int i = 0; i < iters; i++) {
        idx  = (idx * 1664525u + 1013904223u) % n_words;
        acc += buf[idx];          // dependent: the next index depends on nothing, but each load is a
        idx ^= acc & 0xffffu;     // cache miss, which is what a TLB miss costs
    }
    if (acc == 0xdeadbeefu) {
        *out = acc;                   // never true; keeps the loads alive
    }
}

static __global__ void stream_kernel(char * __restrict__ dst, const char * __restrict__ src,
                                     int64_t chunk, int64_t n_chunks) {
    const int64_t c = blockIdx.x;
    if (c >= n_chunks) {
        return;
    }
    size_t i = (size_t) threadIdx.x * 8;
    const size_t n = (size_t) chunk;
    char * d = dst + c * chunk;
    const char * s = src + c * chunk;
    for (; i + 8 <= n; i += (size_t) blockDim.x * 8) {
        *reinterpret_cast<uint64_t *>(d + i) = *reinterpret_cast<const uint64_t *>(s + i);
    }
}

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
                                 : "/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/"
                                   "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    const size_t stream_mib = argc > 2 ? (size_t) atoi(argv[2]) : 512;   // host bytes streamed concurrently
    const size_t n_stream   = stream_mib << 20;
    const size_t chunk      = 1u << 20;
    const int64_t n_chunks  = (int64_t) (n_stream / chunk);

    hipSetDevice(0);
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    char * map = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    if ((size_t) st.st_size < 4 * n_stream) {
        fprintf(stderr, "file too small\n");
        return 1;
    }
    char * reg_src = map + ((size_t) (st.st_size / 4) & ~(page - 1));
    char * hg_src  = map + ((size_t) (st.st_size / 2) & ~(page - 1));

    // The in-situ registration shape: 144 ranges of 242.5 MiB (473600 B x 512 experts).
    const size_t tbl     = 473600ull * 512ull;
    const size_t n_tbl   = 40ull << 30;                 // ~40 GiB, the srv cell's host expert set
    const size_t room    = ((size_t) st.st_size - (size_t) (reg_src - map)) / tbl;
    const size_t tables  = n_tbl / tbl < room ? n_tbl / tbl : room;
    for (size_t i = 0; i < tables; i++) {
        HIP_CHECK(hipHostRegister(reg_src + i * tbl, tbl, hipHostRegisterDefault));
    }
    fprintf(stderr, "registered %zu ranges x %.1f MiB = %.1f GiB (in-situ shape)\n", tables,
            (double) tbl / (1 << 20), (double) (tables * tbl) / (1u << 30));

    void * pin = nullptr;
    HIP_CHECK(hipHostMalloc(&pin, n_stream, hipHostMallocDefault));
    memcpy(pin, reg_src, n_stream);
    // The huge-page arm: a SEPARATE mapping, collapsed to 2 MiB pages before registering, so the only thing
    // it changes against the 4 KiB arm is the page size of the registered range.
    char * map2 = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map2 == MAP_FAILED) { perror("mmap2"); return 1; }
    char * hg2_src = map2 + ((size_t) (3 * (size_t) st.st_size / 4) & ~(page - 1));
    (void) madvise(hg2_src, n_stream, MADV_HUGEPAGE);
    if (madvise(hg2_src, n_stream, MADV_COLLAPSE) != 0) {
        fprintf(stderr, "  MADV_COLLAPSE refused: %s (huge-page arm is then just another 4 KiB range)\n",
                strerror(errno));
    }
    HIP_CHECK(hipHostRegister(hg2_src, n_stream, hipHostRegisterDefault));

    const size_t vram = 256u << 20;
    unsigned * dbuf = nullptr;
    HIP_CHECK(hipMalloc((void **) &dbuf, vram));
    HIP_CHECK(hipMemset(dbuf, 0x5a, vram));
    char * sdev = nullptr;
    HIP_CHECK(hipMalloc((void **) &sdev, n_stream));
    unsigned * out = nullptr;
    HIP_CHECK(hipMalloc((void **) &out, 4));

    hipStream_t sc, strm;
    HIP_CHECK(hipStreamCreate(&sc));
    HIP_CHECK(hipStreamCreate(&strm));

    const int iters = 3000;
    const unsigned n_words = (unsigned) (vram / 4);

    struct arm { const char * name; const char * src; };
    const arm arms[] = {
        { "compute alone",                      nullptr },
        { "compute + pinned stream",            (const char *) pin },
        { "compute + registered stream",        (const char *) reg_src },
        { "compute + registered(hugepage) str", (const char *) hg2_src },
    };
    printf("\n%-36s %8s %8s\n", "arm", "wall s", "it/s");
    printf("--------------------------------------------------------\n");
    double alone = 0.0;
    for (const arm & a : arms) {
        HIP_CHECK(hipDeviceSynchronize());
        const double t0 = now_s();
        // the compute stream: a long dependent-load chain
        for (int r = 0; r < 40; r++) {
            latency_kernel<<<512, 256, 0, sc>>>(dbuf, n_words, out, iters, (unsigned) r * 7919u);
        }
        if (a.src != nullptr) {
            // a continuous host->device stream on the other stream, at ~line rate
            for (int r = 0; r < 40; r++) {
                stream_kernel<<<(int) n_chunks, 256, 0, strm>>>((char *) sdev, a.src, (int64_t) chunk, n_chunks);
            }
        }
        HIP_CHECK(hipStreamSynchronize(sc));
        HIP_CHECK(hipStreamSynchronize(strm));
        const double dt = now_s() - t0;
        if (a.name[8] == 'a') {   // "compute alone"
            alone = dt;
        }
        printf("%-36s %8.2f %8.1f%s\n", a.name, dt, 40.0 * 512 * 256 * iters / dt / 1e6,
               alone > 0 && dt > alone + 0.001 ? "   (+compute slowdown vs alone)" : "");
    }
    printf("\nThe interesting comparison is row 2 vs row 3 vs row 4: the compute kernel never reads host\n"
           "memory, so any difference is the cost of the REGISTRATION on unrelated work.\n");
    return 0;
}
