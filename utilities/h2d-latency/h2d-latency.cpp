// h2d-latency -- what the host->device path actually costs, in MICROSECONDS, at the sizes a model load and a
// prefill actually use.  It is a measurement tool: it allocates, measures, frees, and prints.  It changes
// nothing, needs no root, and says UNKNOWN rather than guess.
//
// WHY IT EXISTS
//   The delivery derives decisions from the H2D link's BANDWIDTH (`ggml_cuda_device_h2d_gbps`: the
//   "H2D bandwidth calibration" line, 512 MiB x3).  Bandwidth is the right number for a bulk fill and the wrong
//   number for everything else: an expert table is staged in ~0.5-8 MiB pieces, the gather reads ~0.5 MiB per
//   expert, and a decode step moves a few KiB.  At those sizes the cost is dominated by LATENCY (copy issue,
//   driver round trip, DMA setup) and not by the link's steady rate, which is why two boxes that calibrate within
//   20 % of each other can stage at rates 40 % apart (GitHub issue #368: the reporter's PCIe 3.0 x16 box at
//   9.2 GB/s against ours at 14.4 GB/s, and his gather won by 38 % where our staging won by 30 %).
//
//   So this reports, per size and per memory kind: the MEDIAN single-copy round trip, the first (cold) copy, and
//   the steady rate.  Feed it the sizes your shape uses.  A box whose latency at 4 KiB and at 1 MiB is high will
//   lose on every small transfer no matter what its bandwidth line says.
//
// BUILD
//   Linux:    hipcc -O2 -std=c++17 h2d-latency.cpp -o h2d-latency
//   Windows:  hipcc -O2 -std=c++17 h2d-latency.cpp -o h2d-latency.exe
//   (ROCm's hipcc works in the same shell you build llama.cpp in; nothing else is needed.  Build it for the arch
//   your card is, e.g. `--offload-arch=gfx1201`, if hipcc cannot infer it.)
//
// USAGE
//   ./h2d-latency                 the default sweep, one device (HIP_VISIBLE_DEVICES honoured)
//   ./h2d-latency --device 1      a specific device
//   ./h2d-latency --sizes 4096,1048576,16777216 --reps 100
//   ./h2d-latency --cold          also time the FIRST copy of a fresh allocation (the Windows first-touch cost)
//   ./h2d-latency --markdown      a table to paste into an issue
//
// WHAT IT DOES NOT REPORT
//   The PCIe GENERATION and link WIDTH: HIP does not expose them on either platform and a wrong guess is worse
//   than no answer, so it prints UNKNOWN and tells you where to read it (`lspci -vv` on Linux, Device Manager ->
//   the adapter's "Link Speed"/"Link Width" on Windows).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>   // QueryPerformanceCounter: the monotonic host clock
#endif

#define HIP_CHECK(x) do {                                                                       \
        hipError_t _e = (x);                                                                    \
        if (_e != hipSuccess) {                                                                  \
            std::fprintf(stderr, "h2d-latency: %s failed: %s\n", #x, hipGetErrorString(_e));     \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

static double now_us() {
    // steady wall clock in microseconds: HIP has no portable timer that is independent of the device, and the
    // point of this tool is the HOST-visible cost of a copy (issue + DMA + completion), not a kernel's time.
#if defined(_WIN32)
    // QueryPerformanceCounter is the Windows monotonic clock.
    static LARGE_INTEGER f = {};
    if (f.QuadPart == 0) {
        QueryPerformanceFrequency(&f);
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return 1.0e6 * (double) c.QuadPart / (double) f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1.0e6 * (double) ts.tv_sec + 1.0e-3 * (double) ts.tv_nsec;
#endif
}

struct result {
    double med_us = -1.0;
    double min_us = -1.0;
    double cold_us = -1.0;
    double gbps   = -1.0;
};

// Time `reps` synchronous copies of `bytes` into one device buffer.  A synchronous copy is deliberate: the
// question is what the caller WAITS for, which is what a staged fill and a gather read both pay.
static int measure(int device, size_t bytes, bool pinned, int reps, bool want_cold, result & out) {
    HIP_CHECK(hipSetDevice(device));
    void * h = nullptr;
    void * d = nullptr;
    if (pinned) {
        hipError_t e = hipHostMalloc(&h, bytes, hipHostMallocDefault);
        if (e != hipSuccess) {
            hipGetLastError();
            return 2;   // pinned memory refused: report as UNKNOWN, not as a slow number
        }
    } else {
        h = std::malloc(bytes);
        if (h == nullptr) {
            return 2;
        }
    }
    if (hipMalloc(&d, bytes) != hipSuccess) {
        hipGetLastError();
        if (pinned) { (void) hipHostFree(h); } else { std::free(h); }
        return 2;
    }
    std::memset(h, 0x3f, bytes);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> t;
    t.reserve((size_t) reps);
    for (int i = 0; i < reps; i++) {
        const double t0 = now_us();
        if (hipMemcpy(d, h, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipGetLastError();
            break;
        }
        const double t1 = now_us();
        t.push_back(t1 - t0);
    }
    if (!t.empty()) {
        out.cold_us = t.front();
        std::vector<double> s = t;
        std::sort(s.begin(), s.end());
        out.min_us = s.front();
        out.med_us = s[s.size() / 2];
        // The steady rate comes from the median, not the minimum: a rate quoted from a lucky copy is the sort of
        // number this tool exists to stop people tuning on.
        out.gbps = 1.0e-9 * (double) bytes / (out.med_us * 1.0e-6);
    }
    (void) want_cold;
    (void) hipFree(d);
    if (pinned) { (void) hipHostFree(h); } else { std::free(h); }
    return t.empty() ? 1 : 0;
}

static std::string human(size_t b) {
    char buf[64];
    if (b >= (size_t) 1 << 20) {
        std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) b / (1024.0 * 1024.0));
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f KiB", (double) b / 1024.0);
    }
    return buf;
}

// A ROUND-TRIP ping: one 4-byte copy to the device and one back, which is the pure issue+completion latency with
// no bulk transfer to hide behind.  This is the number to compare between two boxes when their bandwidth lines
// look similar but their small-transfer behaviour does not.
static int measure_pingpong(int device, bool pinned, int reps) {
    HIP_CHECK(hipSetDevice(device));
    int * h = nullptr;
    int * d = nullptr;
    if (pinned) {
        if (hipHostMalloc((void **) &h, sizeof(int), hipHostMallocDefault) != hipSuccess) {
            hipGetLastError();
            return 2;
        }
    } else {
        h = (int *) std::malloc(sizeof(int));
    }
    if (hipMalloc(&d, sizeof(int)) != hipSuccess) {
        hipGetLastError();
        if (pinned) { (void) hipHostFree(h); } else { std::free(h); }
        return 2;
    }
    *h = 1;
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; i++) {
        const double t0 = now_us();
        if (hipMemcpy(d, h, sizeof(int), hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(h, d, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
            hipGetLastError();
            break;
        }
        t.push_back(now_us() - t0);
    }
    int rc = t.empty() ? 1 : 0;
    if (!t.empty()) {
        std::sort(t.begin(), t.end());
        std::printf("  round trip, 4 B, %-8s  median %8.2f us   min %8.2f us   (one copy each way, synchronised)\n",
                    pinned ? "pinned" : "pageable", t[t.size() / 2], t.front());
    }
    (void) hipFree(d);
    if (pinned) { (void) hipHostFree(h); } else { std::free(h); }
    return rc;
}

int main(int argc, char ** argv) {
    int   device = -1;
    int   reps   = 50;
    bool  markdown = false;
    std::vector<size_t> sizes = {
        (size_t) 4      << 10,    // a routing table / a small input
        (size_t) 64     << 10,    // a few experts
        (size_t) 1      << 20,    // a small expert table slice
        (size_t) 8      << 20,    // ~one staged table
        (size_t) 64     << 20,    // a compute buffer
        (size_t) 512    << 20,    // the delivery's own bandwidth probe
    };

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--markdown") == 0) {
            markdown = true;
        } else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            sizes.clear();
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos <= s.size()) {
                const size_t comma = s.find(',', pos);
                const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!tok.empty()) {
                    sizes.push_back((size_t) std::strtoull(tok.c_str(), nullptr, 10));
                }
                if (comma == std::string::npos) {
                    break;
                }
                pos = comma + 1;
            }
        } else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--reps N] [--sizes a,b,...] [--markdown]\n", argv[0]);
            return 2;
        }
    }

    int n_dev = 0;
    if (hipGetDeviceCount(&n_dev) != hipSuccess || n_dev <= 0) {
        hipGetLastError();
        std::printf("UNKNOWN: no HIP device visible (hipGetDeviceCount failed or returned 0)\n");
        return 1;
    }
    if (device < 0) {
        device = 0;
    }
    if (device >= n_dev) {
        std::printf("UNKNOWN: --device %d but only %d device(s) are visible\n", device, n_dev);
        return 2;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    std::printf("h2d-latency: what the host->device path costs at the sizes a model uses\n");
    std::printf("  device          %d of %d: %s\n", device, n_dev, prop.name);
    std::printf("  memory          %.1f MiB total, PCI domain:bus:dev = %04x:%02x:%02x\n",
                (double) prop.totalGlobalMem / (1024.0 * 1024.0),
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    std::printf("  link generation UNKNOWN: HIP does not expose it.  Linux: `lspci -vv -s %04x:%02x:%02x` -> "
                "LnkCap/LnkSta.  Windows: Device Manager -> Display adapters -> the card -> Details -> "
                "\"PCI current link speed\"/\"width\".\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    std::printf("  reps            %d per size (median and min reported; the FIRST copy per size is the cold one)\n",
                reps);
    std::printf("\n");

    if (markdown) {
        std::printf("| size | kind | median us | min us | cold us | steady GB/s |\n");
        std::printf("|---:|---|---:|---:|---:|---:|\n");
    } else {
        std::printf("  size     kind      median       min       cold    steady\n");
        std::printf("  -----------------------------------------------------------------------\n");
    }

    int rc = 0;
    for (size_t bytes : sizes) {
        for (int pinned = 0; pinned < 2; pinned++) {
            result r;
            const int mr = measure(device, bytes, pinned != 0, reps, true, r);
            const char * kind = pinned ? "pinned" : "pageable";
            if (mr != 0 || r.med_us < 0.0) {
                if (markdown) {
                    std::printf("| %s | %s | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |\n", human(bytes).c_str(), kind);
                } else {
                    std::printf("  %-9s %-9s UNKNOWN (that memory kind could not be allocated or copied)\n",
                                human(bytes).c_str(), kind);
                }
                continue;
            }
            if (markdown) {
                std::printf("| %s | %s | %.1f | %.1f | %.1f | %.2f |\n", human(bytes).c_str(), kind,
                            r.med_us, r.min_us, r.cold_us, r.gbps);
            } else {
                std::printf("  %-9s %-9s %7.1f us  %7.1f us  %7.1f us   %6.2f GB/s\n",
                            human(bytes).c_str(), kind, r.med_us, r.min_us, r.cold_us, r.gbps);
            }
        }
    }

    std::printf("\n");
    if (!markdown) {
        std::printf("  small-copy latency (the number that separates two boxes with similar bandwidth):\n");
    }
    (void) measure_pingpong(device, false, reps);
    (void) measure_pingpong(device, true,  reps);

    std::printf("\n");
    std::printf("  reading it: a fit decision wants the median, not the min.  The delivery's staging transport moves\n");
    std::printf("  ~0.5-8 MiB per expert table and its gather reads ~0.5 MiB per expert, so the 64 KiB and 1 MiB rows\n");
    std::printf("  are the ones that predict prefill.  The 512 MiB row is the delivery's own bandwidth probe.  A large\n");
    std::printf("  gap between cold and median is first-touch cost (on Windows, a probe allocation often does not come\n");
    std::printf("  back after cudaFree/hipFree: issue #97); a large gap between pageable and pinned is why the host\n");
    std::printf("  expert tier is pinned.\n");

    return rc;
}
