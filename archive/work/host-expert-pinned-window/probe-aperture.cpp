// wip/host-expert-pinned-window: WHY is a staged copy out of the registered file mapping ~1.43x slower
// than the same copy out of a `hipHostMalloc` buffer?
//
// The prefill cell (`srv-test.sh`, Flash-Next GSQ-RCO-IQ3_XXS, `-ncmoe 48`) stages the IDENTICAL host
// byte volume in `pinned` and in `--host-experts window` (3762 calls, host 407417.9 MiB both arms), and
// `pinned` finishes it at ~18.6 GiB/s against window's ~13.5 GiB/s.  So the gap is neither the launch
// count nor the geometry (`moe_cache_stage_kernel` is the same kernel) -- it is how fast the device can
// READ the master.
//
// This probe isolates that, with no model and no cache:
//   * sources : a `hipHostMalloc` buffer, the read-only model mapping registered in place, and the same
//               mapping with `MADV_HUGEPAGE`+`MADV_COLLAPSE` tried first (the mapping is 4 K pages here:
//               `/proc/meminfo` `FilePmdMapped: 0 kB`, THP = madvise),
//   * transports: the copy engine (`hipMemcpyAsync`) and the SM kernel the staging path uses
//               (`moe_cache_policy_copy`'s 8-byte strided loop, one block per expert-sized chunk).
//
//   build: ./build.sh && ./build/probe-aperture [GGUF] [GIB] [CHUNK_MIB]
//
// Reading it: if `registered` CE/SM lands far below `pinned` at the SAME size, the master's aperture is
// the prefill gap and no dispatch change in `moe_cache_stage_table` can close it.  If `+hugepage` closes
// it, the fix is an `madvise` on the host range (cheap, and it works for `-sm tensor` because it is
// applied per host tensor, before the per-device children register).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

// The staging kernel's copy loop, at the same width and block shape.
static __device__ __forceinline__ void policy_copy(char * dst, const char * src, size_t n, int tid, int nt) {
    size_t i = (size_t) tid * 8;
    for (; i + 8 <= n; i += (size_t) nt * 8) {
        *reinterpret_cast<uint64_t *>(dst + i) = *reinterpret_cast<const uint64_t *>(src + i);
    }
    const size_t base = (n / 8) * 8;
    for (size_t j = base + (size_t) tid; j < n; j += (size_t) nt) {
        dst[j] = src[j];
    }
}

// One block per chunk (the staging kernel is one block per expert, 256 threads).
static __global__ void read_kernel(char * __restrict__ dst, const char * __restrict__ src,
                                   int64_t chunk, int64_t n_chunks) {
    const int64_t c = blockIdx.x;
    if (c >= n_chunks) {
        return;
    }
    policy_copy(dst + c * chunk, src + c * chunk, (size_t) chunk, threadIdx.x, blockDim.x);
}

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// FilePmdMapped for the mapping, straight out of smaps: the only proof a huge page was really used.
static void report_pmd(const void * addr, size_t len) {
    FILE * f = fopen("/proc/self/smaps", "r");
    if (f == nullptr) {
        return;
    }
    char line[512];
    bool in = false;
    long long pmd = -1;
    while (fgets(line, sizeof(line), f) != nullptr) {
        unsigned long lo = 0, hi = 0;
        if (sscanf(line, "%lx-%lx", &lo, &hi) == 2) {
            in = (const void *) lo == addr;
            continue;
        }
        if (in && strncmp(line, "FilePmdMapped:", 14) == 0) {
            pmd = atoll(line + 14);
            break;
        }
    }
    fclose(f);
    fprintf(stderr, "  mapping %.1f GiB: FilePmdMapped = %lld kB%s\n", (double) len / (1u << 30), pmd,
            pmd > 0 ? " (2 MiB pages in use)" : " (4 KiB pages only)");
}

static const char * clip;   // set to 1 to make the compiler keep the reads

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
                                 : "/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/"
                                   "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    const size_t gib  = argc > 2 ? (size_t) atoi(argv[2]) : 2;
    const size_t cmib = argc > 3 ? (size_t) atoi(argv[3]) : 1;   // chunk ~= one host expert (1.0-1.1 MiB)
    const size_t n    = gib << 30;
    const size_t chunk = cmib << 20;
    const int64_t n_chunks = (int64_t) (n / chunk);
    const int iters = 5;

    hipSetDevice(0);
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device: %s  (%d SMs)%s\n", prop.name, prop.multiProcessorCount,
            prop.unifiedAddressing ? "" : "  NO UVA");

    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    if ((size_t) st.st_size < n) {
        fprintf(stderr, "file is smaller than %zu GiB\n", gib);
        return 1;
    }
    char * map = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    char * src = map + ((size_t) (st.st_size / 4) & ~(page - 1));   // a real expert range, page aligned

    // The pinned master: what `--host-experts pinned` gives the staging kernel.
    void * pin = nullptr;
    HIP_CHECK(hipHostMalloc(&pin, n, hipHostMallocDefault));
    memcpy(pin, src, n);

    // The window master: the mapping itself, registered in place (as `bind_host_dev_locked` does).
    HIP_CHECK(hipHostRegister(src, n, hipHostRegisterDefault));
    void * reg_alias = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&reg_alias, src, 0));

    // And the same range with huge pages attempted first.
    char * map2 = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map2 == MAP_FAILED) { perror("mmap2"); return 1; }
    char * src2 = map2 + ((size_t) (st.st_size / 2) & ~(page - 1));
    const int madv_rc  = madvise(src2, n, MADV_HUGEPAGE);
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
    const int coll_rc  = madvise(src2, n, MADV_COLLAPSE);   // synchronous, needs the pages resident
    fprintf(stderr, "  MADV_HUGEPAGE=%d MADV_COLLAPSE=%d (%s)\n", madv_rc, coll_rc, strerror(errno));
    HIP_CHECK(hipHostRegister(src2, n, hipHostRegisterDefault));
    void * reg2_alias = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&reg2_alias, src2, 0));

    char * dev = nullptr;
    HIP_CHECK(hipMalloc((void **) &dev, n));

    struct arm {
        const char * name;
        void *       host;     // CPU-side source (copy engine)
        const char * alias;    // device-side source (SM kernel)
        int          transport;  // 0 = CE, 1 = SM
    };
    const arm arms[] = {
        { "pinned  hipHostMalloc      ", pin,   (const char *) pin,      0 },
        { "pinned  hipHostMalloc      ", pin,   (const char *) pin,      1 },
        { "registered mapping         ", src,   (const char *) reg_alias, 0 },
        { "registered mapping         ", src,   (const char *) reg_alias, 1 },
        { "registered +hugepage (MAP2)", src2,  (const char *) reg2_alias, 0 },
        { "registered +hugepage (MAP2)", src2,  (const char *) reg2_alias, 1 },
    };

    printf("\n%-30s %-4s %8s %10s\n", "source", "xport", "best", "all (GiB/s)");    printf("-------------------------------------------------------------------------\n");
    for (const arm & a : arms) {
        double best = 0.0;
        char   all[256] = {0};
        size_t used = 0;
        for (int it = 0; it < iters; it++) {
            HIP_CHECK(hipDeviceSynchronize());
            const double t0 = now_s();
            if (a.transport == 0) {
                HIP_CHECK(hipMemcpyAsync(dev, a.host, n, hipMemcpyDefault, 0));
            } else {
                read_kernel<<<(int) n_chunks, 256, 0, 0>>>((char *) dev, a.alias, (int64_t) chunk, n_chunks);
            }
            HIP_CHECK(hipDeviceSynchronize());
            const double dt = now_s() - t0;
            const double gbs = (double) n / dt / (double) (1u << 30);
            if (gbs > best) {
                best = gbs;
            }
            used += snprintf(all + used, sizeof(all) - used, "%.2f ", gbs);
            if (it == 0) {   // the first pass may pay the faults; report it separately, not in `best`
                best = 0.0;
                snprintf(all + used - 6, 6, "%s", "");
                used = snprintf(all, sizeof(all), "%.2f ", gbs);
            }
        }
        printf("%-30s %-4s %8.2f  first+then: %s\n", a.name, a.transport == 0 ? "CE" : "SM", best, all);
    }
    report_pmd(src,  n);
    report_pmd(src2, n);

    // ---- the in-situ shape -------------------------------------------------------------------------
    // The real staging loop is NOT one big range: `bind_host_dev_locked` registers ONE RANGE PER HOST
    // TENSOR, 144 of them on the srv cell at 473600 B x 512 experts = 242.5 MiB each, and the staging
    // kernel runs one grid of 512 blocks x 256 threads per range, flat-copying 473600 B per block.  If
    // throughput here is below the single-big-range arm, the per-registration aperture is the residual.
    const size_t hb      = 473600;                     // host_bytes per expert (from the CACHE_GEOM line)
    const int    nex     = 512;                        // experts per table on this model
    const size_t tbl     = hb * (size_t) nex;          // 242.5 MiB
    char * map3 = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map3 == MAP_FAILED) { perror("mmap3"); return 1; }
    char * src3 = map3 + ((size_t) (3 * (size_t) st.st_size / 4) & ~(page - 1));   // untouched by the arms above
    size_t n_tbl        = ((size_t) 12 << 30) / tbl;  // 12 GiB of host set: past any single-range TLB reach
    const size_t room   = ((size_t) st.st_size - (size_t) (src3 - map3)) / tbl;   // never read past EOF
    if (n_tbl > room) { n_tbl = room; }
    fprintf(stderr, "\n  in-situ shape: %zu tables x %zu GiB total, %zu B per expert\n",
            n_tbl, (n_tbl * tbl) >> 30, hb);

    void * pin_many = nullptr;
    HIP_CHECK(hipHostMalloc(&pin_many, n_tbl * tbl, hipHostMallocDefault));
    memcpy(pin_many, src3, n_tbl * tbl);
    for (size_t i = 0; i < n_tbl; i++) {
        HIP_CHECK(hipHostRegister(src3 + i * tbl, tbl, hipHostRegisterDefault));
    }
    void * alias0 = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&alias0, src3, 0));   // per-range aliases are contiguous here
    char * dev1 = nullptr;
    HIP_CHECK(hipMalloc((void **) &dev1, tbl));

    for (int variant = 0; variant < 2; variant++) {
        const char * name = variant == 0 ? "pinned  144 x 242 MiB tbls" : "registered 144 x 242 MiB";
        const char * base = variant == 0 ? (const char *) pin_many : (const char *) alias0;
        double best = 0.0;
        for (int it = 0; it < 3; it++) {
            HIP_CHECK(hipDeviceSynchronize());
            const double t0 = now_s();
            for (size_t i = 0; i < n_tbl; i++) {   // exactly what moe_cache_stage_table issues
                read_kernel<<<nex, 256, 0, 0>>>((char *) dev1, base + i * tbl, (int64_t) hb, nex);
            }
            HIP_CHECK(hipDeviceSynchronize());
            const double dt  = now_s() - t0;
            const double gbs = (double) (n_tbl * tbl) / dt / (double) (1u << 30);
            if (gbs > best) { best = gbs; }
        }
        printf("%-30s %-4s %8.2f  (%zu tables, one 512-block grid each)\n", name, "SM", best, n_tbl);
    }

    printf("\n(rate is bytes moved / wall time for one %zu GiB pass at %zu MiB per chunk)\n", gib, cmib);
    return 0;
}
