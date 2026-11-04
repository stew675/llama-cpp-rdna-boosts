// probe-memcpy2d -- the ONE thing the other probes did not model: a 2-D (rowed, pitched) H2D copy out of
// the pageable model mapping, which is the actual per-fill geometry of `ffn_down_exps` under `-sm tensor`.
//
// WHY (wip/host-expert-eager-push, 2026-11-04): `GGML_CUDA_CACHE_GEOM_DEBUG=1` shows a 2-device tensor
// split produces TWO geometries per layer:
//
//   ffn_gate_exps / ffn_up_exps : split_axis=1, host_pitch=0   -> contiguous 1-D copy (src_off 0 / 294912)
//   ffn_down_exps               : split_axis=0, host_pitch=352 -> 2-D copy, 2048 rows of 176 B
//
// and `copy_slot_from_src_impl` issues that as ONE `cudaMemcpy2DAsync` with a PAGEABLE source when
// MOE_HOST_PUSH_SRC=mapping.  The other probes all issued 1-D pageable copies, which is why none of them
// reproduced the 107-191 us/fill the real path pays at 2-3 devices: a pageable 2-D copy is staged PER ROW
// by the runtime, so its cost tracks the ROW COUNT (2048), not the bytes.
//
// This probe measures, at the exact real geometry, for 1 and 2 devices:
//
//   MODE=1d|2d  SRC=anon|pinned|file|reg  DST=malloc|vmm
//
// `SRC=reg` is the candidate fix: `hipHostRegister` over the model mapping.  Registration pins the pages
// IN PLACE (no duplicate copy -- the RAM win the campaign is after) and makes the pointer a DMA source, so
// the copy is a normal pinned transfer.  `SRC=file` (mmap, NOT registered) is the current pageable case.
//
//   build:
//     clang++ -x hip --offload-arch=gfx1201 -O2 -I/opt/rocm-10.1.0-gfx120X/include \
//       -L/opt/rocm-10.1.0-gfx120X/lib -lamdhip64 -pthread -o /tmp/probe-memcpy2d probe-memcpy2d.cpp
//
//   run (the headline comparison, 2 devices, the ffn_down_exps geometry):
//     HIP_VISIBLE_DEVICES=0,2 MODE=2d SRC=file   /tmp/probe-memcpy2d 2 30 64
//     HIP_VISIBLE_DEVICES=0,2 MODE=2d SRC=pinned /tmp/probe-memcpy2d 2 30 64
//     HIP_VISIBLE_DEVICES=0,2 MODE=2d SRC=reg    /tmp/probe-memcpy2d 2 30 64
//     HIP_VISIBLE_DEVICES=0,2 MODE=1d SRC=file   /tmp/probe-memcpy2d 2 30 64
//
//   geometry knobs (defaults = the real 2-device ffn_down_exps table):
//     ROWS=2048 ROW=176 PITCH=352   (MODE=1d uses ROWS*ROW bytes, one contiguous copy)
//
//   ADV=1 (default) advances the source window by PITCH*ROWS bytes per fill, so each copy reads a
//   DIFFERENT 704 KiB of the mapping -- the real access pattern (an expert miss is a new expert).  ADV=0
//   reads the same window every time, which is what a probe that re-reads one buffer measures.
//   STARTOFF=<bytes> picks where in the file the window starts (default 1 GiB, 4096-aligned).
//   BUSYMB=<MiB> enqueues a D2D copy of that many MiB on the fill stream BEFORE the round's fills, so the
//   fills are issued behind live GPU work -- which is what the real path does and what none of the earlier
//   probes modelled.  If a pageable copy's host time jumps when the stream is busy, the cost is a
//   SYNCHRONISATION with the queue (the driver's staging cannot be reused while its DMA is outstanding),
//   not the bytes, the geometry or the destination.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}
static long envi(const char * k, long dflt) { const char * v = getenv(k); return v ? atol(v) : dflt; }
static std::string envs(const char * k, const char * dflt) { const char * v = getenv(k); return v ? v : dflt; }

// A VMM (hipMemAddressReserve + hipMemCreate + hipMemMap) destination is what the expert-cache arena
// actually is: the slab (`GGML_CUDA_SLAB`, default on for host-expert models) hands tables a span of a
// VMM region, not a plain hipMalloc.  DST=vmm exists to rule the destination in or out (HANDOVER exp 1).
static void * alloc_vmm_region(int dev, size_t bytes) {
    HIPCK(hipSetDevice(dev));
    void * va = nullptr;
    HIPCK(hipMemAddressReserve(&va, bytes, 0, nullptr, 0));
    hipMemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;
    size_t gran = 0;
    HIPCK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    const size_t phys = ((bytes + gran - 1) / gran) * gran;
    hipMemGenericAllocationHandle_t handle;
    HIPCK(hipMemCreate(&handle, phys, &prop, 0));
    HIPCK(hipMemMap(va, phys, 0, handle, 0));
    hipMemAccessDesc acc;
    memset(&acc, 0, sizeof(acc));
    acc.location.type = hipMemLocationTypeDevice;
    acc.location.id = dev;
    acc.flags = hipMemAccessFlagsProtReadWrite;
    HIPCK(hipMemSetAccess(va, phys, &acc, 1));
    return va;
}

int main(int argc, char ** argv) {
    const int    ndev    = argc > 1 ? atoi(argv[1]) : 2;
    const int    rounds  = argc > 2 ? atoi(argv[2]) : 30;
    const int    per_rd  = argc > 3 ? atoi(argv[3]) : 64;
    const size_t rows    = (size_t) envi("ROWS", 2048);
    const size_t row     = (size_t) envi("ROW", 176);
    const size_t pitch   = (size_t) envi("PITCH", 352);
    const std::string mode = envs("MODE", "2d");
    const std::string src  = envs("SRC", "file");
    const std::string dstk = envs("DST", "malloc");
    const bool   adv     = envi("ADV", 1) != 0;
    const size_t startoff = (size_t) envi("STARTOFF", 1L << 30);
    const std::string file = envs("FILE",
            "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf");
    const bool   two_d   = mode == "2d";
    const size_t busymb  = (size_t) envi("BUSYMB", 0);

    const size_t span  = pitch * rows;              // one expert's host bytes (704 KiB)
    const size_t bytes = two_d ? row * rows : row * rows;   // bytes moved per fill (360 KiB)
    const size_t nwin  = bytes ? (bytes + span) / span : 1; // source windows needed
    const size_t srcsz = startoff + (adv ? (size_t) per_rd : 1) * span + 4096;

    printf("probe-memcpy2d: %d dev, MODE=%s SRC=%s DST=%s, %zu x %zu B (pitch %zu) = %zu B/fill, "
           "%d fills/round, %d rounds%s\n", ndev, mode.c_str(), src.c_str(), dstk.c_str(), rows, row, pitch,
           bytes, per_rd, rounds, adv ? ", source window ADVANCES (fresh pages per fill)" : ", same window");

    // ---- source
    char * sbase = nullptr;
    const size_t sind = startoff;   // file offset of the first window
    int fd = -1;
    if (src == "anon" || src == "pinned") {
        if (src == "pinned") {
            HIPCK(hipHostMalloc((void **) &sbase, (adv ? (size_t) per_rd : 1) * span + 4096,
                                hipHostMallocDefault));
        } else {
            if (posix_memalign((void **) &sbase, 4096, (adv ? (size_t) per_rd : 1) * span + 4096) != 0) {
                return 1;
            }
        }
        memset(sbase, 1, (adv ? (size_t) per_rd : 1) * span + 4096);
    } else {
        fd = open(file.c_str(), O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }
        const size_t maplen = srcsz - startoff;
        char * m = (char *) mmap(nullptr, maplen, PROT_READ, MAP_SHARED, fd, (off_t) startoff);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        sbase = m;
        if (src == "reg") {
            const double r0 = now_ms();
            const hipError_t e = hipHostRegister(sbase, maplen, hipHostRegisterReadOnly | hipHostRegisterPortable);
            const double r1 = now_ms();
            if (e != hipSuccess) {
                printf("  hipHostRegister FAILED: %s (%s)\n", hipGetErrorString(e), e == hipErrorInvalidValue ?
                       "invalid value" : e == hipErrorOutOfMemory ? "OOM" : "other");
                return 2;
            }
            printf("  hipHostRegister(%zu MiB of %s @ +%zu) OK in %.1f ms (%.1f GB/s of registration)\n",
                   maplen >> 20, file.c_str(), startoff, r1 - r0, (double) maplen / 1e9 / ((r1 - r0) / 1000.0));
            // touch every page once: registration alone does not fault them in, and a copy that also pays
            // first-touch faults would misattribute them to the copy.
            volatile long long acc = 0;
            for (size_t o = 0; o < maplen; o += 4096) { acc += sbase[o]; }
            (void) acc;
        }
        printf("  source: %s @ file offset %zu, %zu MiB, %s\n", file.c_str(), startoff, maplen >> 20,
               src == "reg" ? "REGISTERED (pinned in place)" : "pageable (NOT registered)");
    }
    (void) nwin;

    // ---- destination + stream, one per device
    std::vector<char *>       d((size_t) ndev, nullptr);
    std::vector<hipStream_t>  st((size_t) ndev, nullptr);
    for (int k = 0; k < ndev; k++) {
        HIPCK(hipSetDevice(k));
        HIPCK(hipStreamCreateWithFlags(&st[(size_t) k], hipStreamNonBlocking));
        if (dstk == "vmm") {
            d[(size_t) k] = (char *) alloc_vmm_region(k, (bytes + 4096) * 4);
        } else {
            HIPCK(hipMalloc((void **) &d[(size_t) k], (bytes + 4096) * 4));
        }
        HIPCK(hipMemset(d[(size_t) k], 0, bytes));
    }

    // BUSY: one plain D2D copy of BUSYMB MiB per device, enqueued on the FILL stream before the fills.
    std::vector<char *> busy((size_t) ndev, nullptr), busy2((size_t) ndev, nullptr);
    if (busymb > 0) {
        for (int k = 0; k < ndev; k++) {
            HIPCK(hipSetDevice(k));
            HIPCK(hipMalloc((void **) &busy[(size_t) k], busymb << 20));
            HIPCK(hipMalloc((void **) &busy2[(size_t) k], busymb << 20));
            HIPCK(hipMemset(busy[(size_t) k], 0, busymb << 20));
            HIPCK(hipStreamSynchronize(st[(size_t) k]));
        }
        printf("  BUSY: a %zu MiB D2D copy is enqueued on each device's stream ahead of every round\n", busymb);
    }
    const auto busy_issue = [&](int k) {
        if (busymb == 0) { return; }
        HIPCK(hipSetDevice(k));
        HIPCK(hipMemcpyAsync(busy2[(size_t) k], busy[(size_t) k], busymb << 20, hipMemcpyDeviceToDevice,
                             st[(size_t) k]));
    };

    double issue_us = 0.0;   // HOST time inside the copy calls -- the metric the in-situ probe reports
    const auto issue = [&](int k, int fill) {
        const auto _t0 = std::chrono::steady_clock::now();
        HIPCK(hipSetDevice(k));
        const size_t so = adv ? (size_t) fill * span : 0;
        const char * s = sbase + so;
        if (two_d) {
            HIPCK(hipMemcpy2DAsync(d[(size_t) k], row, s, pitch, row, rows, hipMemcpyHostToDevice,
                                   st[(size_t) k]));
        } else {
            HIPCK(hipMemcpyAsync(d[(size_t) k], s, bytes, hipMemcpyHostToDevice, st[(size_t) k]));
        }
        issue_us += std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
                std::chrono::steady_clock::now() - _t0).count();
    };

    // warm-up (2 rounds, untimed)
    for (int r = 0; r < 2; r++) {
        for (int k = 0; k < ndev; k++) {
            for (int f = 0; f < per_rd; f++) { issue(k, f); }
        }
        for (int k = 0; k < ndev; k++) { HIPCK(hipSetDevice(k)); HIPCK(hipStreamSynchronize(st[(size_t) k])); }
    }

    // VERIFY=1: did the 2-D call actually transfer the SPARSE pattern (176 of every 352 bytes), and not
    // 2048x176 contiguous bytes?  A "fast" 2-D pageable copy is only believable if the bytes are right.
    if (envi("VERIFY", 0) != 0) {
        const size_t so = 0;
        const char * s = sbase + so;
        HIPCK(hipSetDevice(0));
        HIPCK(hipMemset(d[0], 0, bytes));
        HIPCK(hipStreamSynchronize(st[0]));
        if (two_d) {
            HIPCK(hipMemcpy2DAsync(d[0], row, s, pitch, row, rows, hipMemcpyHostToDevice, st[0]));
        } else {
            HIPCK(hipMemcpyAsync(d[0], s, bytes, hipMemcpyHostToDevice, st[0]));
        }
        HIPCK(hipStreamSynchronize(st[0]));
        std::vector<char> got(bytes), want(bytes);
        HIPCK(hipMemcpy(got.data(), d[0], bytes, hipMemcpyDeviceToHost));
        for (size_t r = 0; r < rows; r++) {
            memcpy(want.data() + r * row, s + r * pitch, row);
        }
        size_t bad = 0;
        for (size_t i = 0; i < bytes; i++) { if (got[i] != want[i]) { bad++; } }
        printf("  VERIFY: %zu/%zu bytes differ from the expected gather, %zu/%zu bytes are zero (a dropped copy "
               "shows as all-zero)\n", bad, bytes,
               (size_t) std::count(got.begin(), got.end(), (char) 0), bytes);
    }

    const double t0 = now_ms();
    for (int r = 0; r < rounds; r++) {
        for (int k = 0; k < ndev; k++) {
            busy_issue(k);
            for (int f = 0; f < per_rd; f++) { issue(k, f); }
        }
        for (int k = 0; k < ndev; k++) { HIPCK(hipSetDevice(k)); HIPCK(hipStreamSynchronize(st[(size_t) k])); }
    }
    const double ms = now_ms() - t0;
    const double nfill = (double) per_rd * (double) rounds * (double) ndev;
    const double total = (double) bytes * nfill;
    printf("  RESULT %8.2f ms  %7.2f GB/s  %7.1f us/fill   (%.1f ms per round)"
           "   HOST %.1f us in the copy call itself\n",
           ms, total / 1e9 / (ms / 1000.0), 1000.0 * ms / nfill, ms / (double) rounds, issue_us / nfill);
    return 0;
}
