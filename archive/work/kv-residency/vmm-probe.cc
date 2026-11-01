// wip/kv-residency/vmm-probe.cc -- the ROCm VMM facts the KV residency design rests on.
//
// Why: a lazily-mapped KV region (campaign option O1) and a page-resident KV pool (O2) both need
// things the delivery never does today.  The slab maps ONE handle over its whole range and never gives
// physical back, and its header records two measurements that constrain any other design:
//
//   * ROCm rejects `hipMemUnmap` of a SUB-range (hipErrorInvalidValue), so a range's unit of give-back
//     is one whole mapping;
//   * a large buffer mapped as ~5900 small mappings then broke an *unrelated* `hipMemcpy2DAsync`
//     in-tree (cpy.cu:479) even though a standalone map of the same size succeeded.
//
// So the probe measures, on the real driver: reserve-VA-only + map-later (KV-grow), N independent
// mappings inside one reservation, per-chunk unmap/release/remap (physical reuse), the sub-range
// refusal, the mapping COUNT at which an unrelated 2D copy breaks, map/unmap latency, and whether a
// freshly mapped chunk arrives zeroed.
//
// Build (the -isystem is mandatory: without it HIP headers resolve from Fedora's ROCm 7.1 in
// /usr/include and the probe silently builds against a different driver API -- see AGENTS.md):
//   hipcc -O3 --offload-arch=gfx1201 -isystem /opt/rocm-10.1.0-gfx120X/include \
//         vmm-probe.cc -o /tmp/vmm-probe
// Run:
//   HIP_VISIBLE_DEVICES=0 /tmp/vmm-probe [--va-mib N]
//
// The tree's `vendors/hip.h` maps the CUDA driver names onto these HIP ones for HIP builds; the probe
// calls the HIP names directly so it needs neither header (it is a standalone tool, not a project file).

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static int g_fail = 0;

static const char * r(const hipError_t e) { return hipGetErrorString(e); }

#define H(x) do { const hipError_t e_ = (x); std::printf("    %-46s %s\n", #x, e_ == hipSuccess ? "ok" : r(e_)); \
    if (e_ != hipSuccess) return false; } while (0)

// hipDeviceptr_t is a void pointer here, so all VA arithmetic goes through this
static inline hipDeviceptr_t off(hipDeviceptr_t base, size_t n) {
    return (hipDeviceptr_t) (uintptr_t) ((char *) (uintptr_t) base + n);
}

static double now_ms() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

__global__ void fill_kernel(uint32_t * p, size_t n, uint32_t v) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) p[i] = v;
}

// Map `bytes` of fresh physical at `addr` and grant READWRITE to the owning device.  Returns true on
// success; on success the handle has been released (the mapping keeps it alive), as CUDA/HIP require.
static bool map_phys(hipDeviceptr_t addr, size_t bytes, int phys) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = phys;
    hipMemGenericAllocationHandle_t h = 0;
    if (hipMemCreate(&h, bytes, &prop, 0) != hipSuccess) { (void) hipGetLastError(); return false; }
    if (hipMemMap(addr, bytes, 0, h, 0) != hipSuccess) { (void) hipGetLastError(); (void) hipMemRelease(h); return false; }
    hipMemAccessDesc a = {};
    a.location.type = hipMemLocationTypeDevice;
    a.location.id = phys;
    a.flags = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess(addr, bytes, &a, 1) != hipSuccess) {
        (void) hipGetLastError(); (void) hipMemUnmap(addr, bytes); (void) hipMemRelease(h); return false;
    }
    (void) hipMemRelease(h);
    return true;
}

// Does an unrelated 2D copy still work?  This is the in-tree failure mode the slab header records.
// `vdst` is an optional VMM-mapped destination (as a compute buffer in the slab is); when it is null a
// plain hipMalloc destination is used.  Prints every arm so a broken test cannot look like a broken driver.
static bool sanity_copy(const char * tag, void * vdst = nullptr, size_t vdst_size = 0) {
    const size_t dpitch = 64 * 16, spitch = 64 * 16;
    const size_t w = 64 * 4, h = 64;
    void * d = vdst;
    void * src = nullptr;
    if (d == nullptr) {
        if (hipMalloc(&d, dpitch * h) != hipSuccess) { std::printf("    [%s] hipMalloc failed\n", tag); return false; }
    } else if (vdst_size < dpitch * h) {
        std::printf("    [%s] vdst too small\n", tag);
        return false;
    }
    // the source extent is spitch*height, not spitch: allocating only one row makes the driver reject
    // the copy with hipErrorInvalidValue (measured -- that is what made the first version of this probe
    // report a hazard with a single mapping and no mapping at all)
    hipError_t e = hipMalloc(&src, spitch * h);
    if (e != hipSuccess) { std::printf("    [%s] src hipMalloc -> %s\n", tag, r(e)); return false; }
    const hipError_t e2 = hipMemcpy2D(d, dpitch, src, spitch, w, h, hipMemcpyDeviceToDevice);
    const hipError_t e3 = hipMemcpy2DAsync(d, dpitch, src, spitch, w, h, hipMemcpyDeviceToDevice, 0);
    (void) hipStreamSynchronize(0);
    if (e2 != hipSuccess || e3 != hipSuccess) {
        std::printf("    [%s] 2D copy into %s -> sync:%s async:%s  <-- hazard\n", tag,
                    vdst ? "VMM" : "malloc", r(e2), r(e3));
    }
    (void) hipFree(src);
    if (vdst == nullptr) { (void) hipFree(d); }
    return e2 == hipSuccess && e3 == hipSuccess;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // progress must be visible if a driver call hangs
    size_t va_mib = 2048;
    int    max_mappings = 5900;
    bool   run_t9 = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--va-mib") && i + 1 < argc)       { va_mib = (size_t) atoll(argv[++i]); }
        else if (!std::strcmp(argv[i], "--max-mappings") && i + 1 < argc) { max_mappings = atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--no-t9"))                   { run_t9 = false; }
    }

    hipDeviceProp_t prop;
    int dev = 0;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess || hipGetDevice(&dev) != hipSuccess) {
        std::printf("no device\n");
        return 2;
    }
    const int phys = dev;
    std::printf("device: %s (%s), total %zu MiB\n", prop.name, prop.gcnArchName, (size_t) (prop.totalGlobalMem >> 20));

    // ---- T1: granularity ---------------------------------------------------------------
    std::printf("\nT1 allocation granularity\n");
    hipMemAllocationProp ap = {};
    ap.type = hipMemAllocationTypePinned;
    ap.location.type = hipMemLocationTypeDevice;
    ap.location.id = phys;
    size_t gran_min = 0;
    size_t gran_rec = 0;
    const bool g1 = hipMemGetAllocationGranularity(&gran_min, &ap, hipMemAllocationGranularityMinimum) == hipSuccess;
    const bool g2 = hipMemGetAllocationGranularity(&gran_rec, &ap, hipMemAllocationGranularityRecommended) == hipSuccess;
    std::printf("    minimum / recommended          %zu KiB / %zu MiB\n", gran_min >> 10, gran_rec >> 20);
    if (!g1 || !g2) { std::printf("    hipMemGetAllocationGranularity FAILED\n"); g_fail++; }

    // ---- T0: baseline, before any VMM mapping exists -----------------------------------
    std::printf("\nT0 baseline: 2D copy with no VMM mapping at all\n");
    {
        const bool ok_base = sanity_copy("baseline");
        std::printf("    baseline 2D copy                  %s\n", ok_base ? "ok" : "BROKEN (the test itself)");
        if (!ok_base) { std::printf("    -> the hazard tests below mean nothing; fix the harness first\n"); g_fail++; }
    }

    // ---- T2: reserve VA only, map a chunk later ----------------------------------------
    std::printf("\nT2 reserve VA only, then map 128 MiB into it\n");
    const size_t chunk = (size_t) 128 << 20;
    const size_t va = va_mib << 20;
    hipDeviceptr_t base = nullptr;
    const hipError_t rv = hipMemAddressReserve(&base, va, 0, 0, 0);
    std::printf("    hipMemAddressReserve(%zu MiB)      %s\n", va_mib, rv == hipSuccess ? "ok" : r(rv));
    if (rv != hipSuccess) { std::printf("    cannot continue\n"); return 1; }
    if (!map_phys(base, chunk, phys)) { std::printf("    map into a reserved range FAILED\n"); g_fail++; return 1; }
    std::printf("    map 128 MiB into the reserve      ok  (VA-only reserve, then map -- KV-grow)\n");

    // ---- T3: N independent mappings inside one reservation ----------------------------
    const int n_chunks_max = (int) (va / chunk);
    const int n_chunks = n_chunks_max < 8 ? n_chunks_max : 8;
    std::printf("\nT3 one reservation, N independent 128 MiB mappings, per-chunk content (N=%d)\n", n_chunks);
    int mapped = 1;
    for (int i = 1; i < n_chunks; ++i) {
        if (map_phys(off(base, (size_t) i * chunk), chunk, phys)) { mapped++; }
    }
    std::printf("    mapped %d/%d chunks                %s\n", mapped, n_chunks, mapped == n_chunks ? "ok" : "SHORT");
    if (mapped != n_chunks) {
        std::printf("    -> only %d chunks mapped; stopping before anything touches the unmapped tail\n", mapped);
        g_fail++;
        (void) hipMemAddressFree(base, va);
        return 1;
    }
    {
        for (int i = 0; i < n_chunks; ++i) {
            fill_kernel<<<256, 256>>>((uint32_t *) off(base, (size_t) i * chunk), chunk / 4, 0x1000u + (uint32_t) i);
        }
        const hipError_t se = hipDeviceSynchronize();
        std::printf("    fill 8 distinct patterns          %s\n", se == hipSuccess ? "ok" : r(se));
        if (se != hipSuccess) { g_fail++; }
        int bad = 0;
        for (int i = 0; i < n_chunks; ++i) {
            for (size_t o = 0; o < chunk; o += chunk / 4) {
                uint32_t v = 0;
                if (hipMemcpy(&v, off(base, (size_t) i * chunk + o), 4, hipMemcpyDeviceToHost) != hipSuccess || v != 0x1000u + (uint32_t) i) {
                    bad++;
                }
            }
        }
        std::printf("    verify per-chunk patterns         %s\n", bad == 0 ? "ok" : "MISMATCH");
        if (bad != 0) { g_fail++; }
    }
    if (!sanity_copy("8 mappings")) { g_fail++; }

    // ---- T4: per-chunk unmap, release, remap the same VA -------------------------------
    std::printf("\nT4 unmap chunk 3 (its exact mapping), remap the same VA\n");
    {
        const hipDeviceptr_t a = off(base, (size_t) 3 * chunk);
        const hipError_t u = hipMemUnmap(a, chunk);
        std::printf("    hipMemUnmap(chunk 3)              %s\n", u == hipSuccess ? "ok" : r(u));
        if (u != hipSuccess) { g_fail++; }
        uint32_t v = 0;
        const hipError_t nb = hipMemcpy(&v, off(base, (size_t) 2 * chunk), 4, hipMemcpyDeviceToHost);
        std::printf("    neighbour chunk 2 still readable  %s (0x%x)\n", nb == hipSuccess ? "ok" : r(nb), v);
        if (nb != hipSuccess || v != 0x1002u) { g_fail++; }
        if (!map_phys(a, chunk, phys)) { std::printf("    remap chunk 3 FAILED\n"); g_fail++; }
        else {
            (void) hipMemcpy(&v, a, 4, hipMemcpyDeviceToHost);
            std::printf("    remapped chunk 3 content          = 0x%x (0x1003 => zeroed, else STALE)\n", v);
        }
    }

    // ---- T5: is a sub-range unmap refused? --------------------------------------------
    std::printf("\nT5 unmap a SUB-range of a mapping (the slab header's documented refusal)\n");
    {
        const hipError_t u = hipMemUnmap(off(base, (size_t) 1 * chunk), chunk / 2);
        std::printf("    hipMemUnmap(chunk 1, half)        %s\n", u == hipSuccess ? "ok <-- ALLOWED (header says it must fail)" : r(u));
        if (u == hipSuccess) { g_fail++; }
    }

    // ---- T6: mapping count vs an unrelated 2D copy ------------------------------------
    std::printf("\nT6 mapping COUNT vs an unrelated hipMemcpy2D (the in-tree hazard)\n");
    const int counts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5900};
    for (int n : counts) {
        if (n > max_mappings) { continue; }
        const size_t cs = (size_t) 4 << 20;
        hipDeviceptr_t b2 = nullptr;
        if (hipMemAddressReserve(&b2, cs * n, 0, 0, 0) != hipSuccess) { std::printf("    n=%-5d reserve failed\n", n); continue; }
        int ok = 0;
        for (int i = 0; i < n; ++i) {
            if (map_phys(off(b2, (size_t) i * cs), cs, phys)) { ok++; }
        }
        char tag[32];
        std::snprintf(tag, sizeof tag, "n=%d", n);
        // only ever touch the range that actually MAPPED: reading reserved-but-unmapped VA faults the
        // GPU with "Page not present" (measured the hard way), which is also the rule a KV pool needs
        const size_t mapped_bytes = cs * (size_t) ok;
        const bool fine  = ok == n && sanity_copy(tag);
        const bool finev = ok == n && sanity_copy("into VMM", (void *) b2, mapped_bytes);
        if (ok == n) { fill_kernel<<<256, 256>>>((uint32_t *) b2, mapped_bytes / 4, 0x77u); }
        const bool finek = ok == n && hipDeviceSynchronize() == hipSuccess;
        std::printf("    n=%-5d mapped %5d/%d | 2D copy malloc:%s VMM:%s | kernel:%s\n", n, ok, n,
                    fine ? "ok" : "BROKEN", finev ? "ok" : "BROKEN", finek ? "ok" : "BROKEN");
        for (int i = 0; i < n; ++i) { (void) hipMemUnmap(off(b2, (size_t) i * cs), cs); }
        (void) hipMemAddressFree(b2, cs * n);
        if (!fine || !finev || !finek) { g_fail++; }
    }

    // ---- T7: map / unmap latency ------------------------------------------------------
    std::printf("\nT7 create+map / unmap latency (the grow-between-steps budget)\n");
    {
        hipDeviceptr_t b3 = nullptr;
        if (hipMemAddressReserve(&b3, (size_t) 1024 << 20, 0, 0, 0) == hipSuccess) {
            const size_t sizes[3] = {(size_t) 4 << 20, (size_t) 64 << 20, (size_t) 256 << 20};
            for (int si = 0; si < 3; ++si) {
                const size_t cs = sizes[si];
                const int reps = cs > ((size_t) 64 << 20) ? 4 : 16;
                double t_map = 1e9, t_unmap = 1e9;
                for (int i = 0; i < reps; ++i) {
                    const double t0 = now_ms();
                    hipMemAllocationProp p3 = {};
                    p3.type = hipMemAllocationTypePinned;
                    p3.location.type = hipMemLocationTypeDevice;
                    p3.location.id = phys;
                    hipMemGenericAllocationHandle_t h = 0;
                    const hipError_t ce = hipMemCreate(&h, cs, &p3, 0);
                    if (ce != hipSuccess) { std::printf("    chunk %zu MiB: hipMemCreate -> %s\n", cs >> 20, r(ce)); break; }
                    const hipError_t me = hipMemMap(b3, cs, 0, h, 0);
                    if (me != hipSuccess) { std::printf("    chunk %zu MiB: hipMemMap -> %s\n", cs >> 20, r(me)); (void) hipMemRelease(h); break; }
                    const double t1 = now_ms();
                    (void) hipMemRelease(h);
                    const double t2 = now_ms();
                    (void) hipMemUnmap(b3, cs);
                    const double t3 = now_ms();
                    if (t1 - t0 < t_map) { t_map = t1 - t0; }
                    if (t3 - t2 < t_unmap) { t_unmap = t3 - t2; }
                }
                std::printf("    chunk %3zu MiB: create+map %8.3f ms   unmap %8.3f ms\n", cs >> 20, t_map, t_unmap);
            }
            (void) hipMemAddressFree(b3, (size_t) 1024 << 20);
        }
    }

    // ---- T8: does a freshly mapped chunk arrive zeroed? --------------------------------
    std::printf("\nT8 freshly mapped physical: zeroed or stale?\n");
    {
        bool any = false;
        for (int i = 0; i < 6; ++i) {
            hipDeviceptr_t b4 = nullptr;
            if (hipMemAddressReserve(&b4, chunk, 0, 0, 0) != hipSuccess) { break; }
            if (!map_phys(b4, chunk, phys)) { (void) hipMemAddressFree(b4, chunk); break; }
            for (size_t o = 0; o < chunk; o += chunk / 64) {
                uint32_t v = 0;
                if (hipMemcpy(&v, off(b4, o), 4, hipMemcpyDeviceToHost) == hipSuccess && v != 0) { any = true; }
            }
            (void) hipMemUnmap(b4, chunk);
            (void) hipMemAddressFree(b4, chunk);
        }
        std::printf("    6 fresh 128 MiB mappings, 64 samples each: %s\n",
                    any ? "STALE BYTES PRESENT -> a KV region must be cleared on map" : "all zero");
    }

    // ---- T9: what growing actually costs -- first touch of fresh physical vs a rewrite ----------
    std::printf("\nT9 first touch of a freshly mapped chunk vs a rewrite (the real growth cost)\n");
    if (run_t9) {
        const size_t cs = (size_t) 256 << 20;
        hipDeviceptr_t b5 = nullptr;
        if (hipMemAddressReserve(&b5, cs, 0, 0, 0) == hipSuccess) {
            for (int arm = 0; arm < 2; ++arm) {
                if (!map_phys(b5, cs, phys)) { std::printf("    map failed\n"); break; }
                const double t0 = now_ms();
                fill_kernel<<<512, 256>>>((uint32_t *) b5, cs / 4, 0xa5u);
                const hipError_t se = hipDeviceSynchronize();
                const double t1 = now_ms();
                std::printf("    %s 256 MiB: %7.3f ms  %6.1f GB/s  %s\n",
                            arm == 0 ? "cold (fresh physical)" : "warm (rewrite)       ",
                            t1 - t0, (double) cs / ((t1 - t0) * 1e6), se == hipSuccess ? "" : r(se));
                if (arm == 0) { (void) hipMemUnmap(b5, cs); }
            }
            (void) hipMemUnmap(b5, cs);
            (void) hipMemAddressFree(b5, cs);
        }
    }

    for (int i = 0; i < n_chunks; ++i) { (void) hipMemUnmap(off(base, (size_t) i * chunk), chunk); }
    (void) hipMemAddressFree(base, va);

    std::printf("\n=== %s (%d failing check(s)) ===\n", g_fail ? "FINDINGS TO ACT ON" : "all as expected", g_fail);
    return g_fail ? 1 : 0;
}
