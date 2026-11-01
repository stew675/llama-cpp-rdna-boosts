// wip/kv-residency/mixed-backing-probe.cc -- can ONE VMM reservation be backed partly by VRAM and
// partly by pinned system RAM?  (O2 / TODO #118, Goal A, A2 design choice (i).)
//
// WHY THIS EXISTS
//   Goal A wants the KV cache's resident window in VRAM and the older cells ("the tail") in system RAM.
//   Two designs were on the table:
//
//     (i)  ONE tensor, mixed backing: the KV keeps its full VA range; the resident cells are mapped as
//          device chunks, the tail as HOST chunks, in that one reservation.  Nothing else in the tree
//          changes -- the cell-walking paths (set_rows writers, the in-place rope shift that backs
//          defrag/seq_add, state save/restore, clear) keep addressing one VA range, and the FA kernels
//          keep one base pointer.
//     (ii) the tail as a SEPARATE tensor in pinned host memory + a two-base read inside the FA kernels
//          (built and gated as A1 on the campaign branch).  Guaranteed to work, but defrag/seq_add
//          becomes a cross-buffer move and the KV writers split into two nodes per layer.
//
//   `vmm-probe.cc` proved the VMM primitives for DEVICE-location chunks only.  A host-location
//   hipMemCreate handle inside a device VA reservation is a different API path (`hipMemLocationTypeHost`
//   / `hipMemLocationTypeHostNuma`), and if the driver refuses it, (i) is impossible -- there is no
//   workaround, a hipHostAlloc pointer cannot be placed inside someone else's VA range.
//
//   So this probe answers, on the real driver: can a host-location handle be created, mapped into a
//   reservation, and granted device access; can the DEVICE read it, WRITE it (the KV writer path), and
//   at what bandwidth (expect ~29 GB/s -- a PCIe crossing either way); does the COPY ENGINE accept it
//   (state save/restore); and does a single VA range really work with the two kinds interleaved (the
//   actual shape, since a KV buffer lays its tensors out consecutively, K0 V0 K1 V1 ...).
//
// Build (the -isystem is mandatory -- without it HIP resolves from Fedora's ROCm 7.1 in /usr/include):
//   hipcc -O3 --offload-arch=gfx1201 -isystem /opt/rocm-10.1.0-gfx120X/include \
//         mixed-backing-probe.cc -o /tmp/mixed-probe
// Run:
//   HIP_VISIBLE_DEVICES=0 /tmp/mixed-probe [--mib N] [--no-write-test]
//
// Reading reserved-but-unmapped VA faults the GPU (VMM-PROBE.md trap 1), so nothing here ever touches a
// range it did not map first, and every device write is one element + a sync before a full pass.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static int g_fail = 0;

static const char * r(const hipError_t e) { return hipGetErrorString(e); }

static inline hipDeviceptr_t off(hipDeviceptr_t base, size_t n) {
    return (hipDeviceptr_t) (uintptr_t) ((char *) (uintptr_t) base + n);
}

static double now_ms() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// grid-stride read: sums everything (so a broken mapping cannot be optimized away) at the FA read pattern
__global__ void read_kernel(const uint32_t * p, size_t n, uint64_t * sink) {
    uint64_t acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        acc += p[i];
    }
    if (acc == 0x1234567890abcdefull) { *sink = acc; }
}

__global__ void write_kernel(uint32_t * p, size_t n, uint32_t v) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) p[i] = v;
}

// one element, so a refused write cannot walk the range
__global__ void poke_kernel(uint32_t * p, uint32_t v) { p[0] = v; }

__global__ void readback_kernel(const uint32_t * p, uint32_t * out) { *out = p[0]; }

static bool map_chunk(hipDeviceptr_t addr, size_t bytes, const hipMemLocationType loc_type, int loc_id,
                      const char ** why) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = loc_type;
    prop.location.id = loc_id;
    hipMemGenericAllocationHandle_t h = 0;
    hipError_t e = hipMemCreate(&h, bytes, &prop, 0);
    if (e != hipSuccess) { *why = "hipMemCreate"; std::printf("      hipMemCreate            %s\n", r(e)); (void) hipGetLastError(); return false; }
    std::printf("      hipMemCreate            ok\n");
    e = hipMemMap(addr, bytes, 0, h, 0);
    if (e != hipSuccess) { *why = "hipMemMap"; std::printf("      hipMemMap               %s\n", r(e)); (void) hipGetLastError(); (void) hipMemRelease(h); return false; }
    std::printf("      hipMemMap               ok\n");
    hipMemAccessDesc a = {};
    a.location.type = hipMemLocationTypeDevice;
    a.location.id = loc_id;
    a.flags = hipMemAccessFlagsProtReadWrite;
    e = hipMemSetAccess(addr, bytes, &a, 1);
    if (e != hipSuccess) { *why = "hipMemSetAccess"; std::printf("      hipMemSetAccess         %s\n", r(e)); (void) hipGetLastError(); (void) hipMemUnmap(addr, bytes); (void) hipMemRelease(h); return false; }
    std::printf("      hipMemSetAccess         ok (device rw)\n");
    (void) hipMemRelease(h);
    return true;
}

static bool map_device_chunk(hipDeviceptr_t addr, size_t bytes, int dev) {
    const char * why = "";
    return map_chunk(addr, bytes, hipMemLocationTypeDevice, dev, &why);
}

// host-location chunk: try plain Host first, then HostNuma(node)
static bool map_host_chunk(hipDeviceptr_t addr, size_t bytes, int dev, int numa) {
    const char * why = "";
    std::printf("    [location Host, id=0]\n");
    if (map_chunk(addr, bytes, hipMemLocationTypeHost, 0, &why)) { return true; }
    std::printf("    -> Host refused (%s); trying HostNuma id=%d\n", why, numa);
    std::printf("    [location HostNuma, id=%d]\n", numa);
    if (map_chunk(addr, bytes, hipMemLocationTypeHostNuma, numa, &why)) { return true; }
    std::printf("    -> HostNuma refused (%s)\n", why);
    return false;
}

// how fast can the device read a chunk? (the only number that matters for Goal A's cost model)
static double read_gbs(hipDeviceptr_t p, size_t bytes, uint64_t * sink_d) {
    const int blocks = 512, threads = 256;
    read_kernel<<<blocks, threads>>>((const uint32_t *) p, bytes / 4, sink_d);   // touch once
    if (hipDeviceSynchronize() != hipSuccess) { return -1.0; }
    const double t0 = now_ms();
    for (int i = 0; i < 3; ++i) { read_kernel<<<blocks, threads>>>((const uint32_t *) p, bytes / 4, sink_d); }
    if (hipDeviceSynchronize() != hipSuccess) { return -1.0; }
    const double t1 = now_ms();
    return (double) (3 * bytes) / ((t1 - t0) * 1e6);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    size_t mib = 128;
    size_t t8_mib = 2048;
    bool   do_write = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--mib") && i + 1 < argc) { mib = (size_t) atoll(argv[++i]); }
        else if (!std::strcmp(argv[i], "--t8-mib") && i + 1 < argc) { t8_mib = (size_t) atoll(argv[++i]); }
        else if (!std::strcmp(argv[i], "--no-write-test")) { do_write = false; }
    }

    hipDeviceProp_t dp;
    int dev = 0;
    if (hipGetDeviceProperties(&dp, 0) != hipSuccess || hipGetDevice(&dev) != hipSuccess) {
        std::printf("no device\n"); return 2;
    }
    std::printf("device: %s (%s), %zu MiB VRAM\n", dp.name, dp.gcnArchName, (size_t) (dp.totalGlobalMem >> 20));

    int pageable = -1, unified = -1, direct_host = -1;
    (void) hipDeviceGetAttribute(&pageable,    hipDeviceAttributePageableMemoryAccess, dev);
    (void) hipDeviceGetAttribute(&unified,     hipDeviceAttributeUnifiedAddressing, dev);
    (void) hipDeviceGetAttribute(&direct_host, hipDeviceAttributeDirectManagedMemAccessFromHost, dev);
    std::printf("attrs: pageableAccess=%d unifiedAddressing=%d directManaged=%d\n", pageable, unified, direct_host);

    int numa = 0;
    {
        // the PCIe root's NUMA node, for the HostNuma attempt
        char path[128];
        std::snprintf(path, sizeof path, "/sys/bus/pci/devices/0000:03:00.0/numa_node");
        if (FILE * f = std::fopen(path, "r")) { if (std::fscanf(f, "%d", &numa) != 1) { numa = 0; } std::fclose(f); }
        if (numa < 0) { numa = 0; }
    }
    std::printf("numa node (gpu 0): %d\n", numa);

    const size_t bytes = mib << 20;

    // ---- T1: granularity for a HOST-location property ---------------------------------
    std::printf("\nT1 allocation granularity for host-location chunks\n");
    for (int which = 0; which < 2; ++which) {
        hipMemAllocationProp ap = {};
        ap.type = hipMemAllocationTypePinned;
        ap.location.type = which == 0 ? hipMemLocationTypeHost : hipMemLocationTypeHostNuma;
        ap.location.id = which == 0 ? 0 : numa;
        size_t gmin = 0, grec = 0;
        const hipError_t e1 = hipMemGetAllocationGranularity(&gmin, &ap, hipMemAllocationGranularityMinimum);
        const hipError_t e2 = hipMemGetAllocationGranularity(&grec, &ap, hipMemAllocationGranularityRecommended);
        std::printf("    %-10s minimum/recommended %s : %zu KiB / %zu MiB\n",
                    which == 0 ? "Host" : "HostNuma", (e1 == hipSuccess && e2 == hipSuccess) ? "ok" : r(e1 != hipSuccess ? e1 : e2),
                    gmin >> 10, grec >> 20);
    }

    // ---- T2: reserve a VA range big enough for two chunks, map a HOST chunk into it ----
    std::printf("\nT2 reserve 2x%zu MiB, map the FIRST half from HOST memory\n", mib);
    hipDeviceptr_t base = nullptr;
    if (hipMemAddressReserve(&base, 2 * bytes, 0, 0, 0) != hipSuccess) { std::printf("    reserve failed\n"); return 1; }
    std::printf("    hipMemAddressReserve    ok\n");
    if (!map_host_chunk(base, bytes, dev, numa)) {
        std::printf("    == (i) IS NOT AVAILABLE ON THIS DRIVER: a host-location chunk cannot live in a device VA reservation\n");
        (void) hipMemAddressFree(base, 2 * bytes);
        std::printf("\n=== FINDINGS TO ACT ON (1): use the separate-tensor design (ii) ===\n");
        return 1;
    }
    std::printf("    HOST chunk mapped into the reservation  ok  <-- the capability (i) needs\n");

    // ---- T3: can the DEVICE read it, and how fast? ------------------------------------
    std::printf("\nT3 device read of the host-backed chunk (expect ~29 GB/s, the link)\n");
    uint64_t * sink = nullptr;
    if (hipMalloc((void **) &sink, sizeof(uint64_t)) != hipSuccess) { std::printf("    sink alloc failed\n"); return 1; }
    {
        write_kernel<<<512, 256>>>((uint32_t *) base, bytes / 4, 0xabcd0001u);   // host pages: a device write
        const hipError_t se = hipDeviceSynchronize();
        std::printf("    device fill (write path)   %s\n", se == hipSuccess ? "ok" : r(se));
        if (se != hipSuccess) { g_fail++; }
        const double gbs = read_gbs(base, bytes, sink);
        std::printf("    device read                %s  %6.1f GB/s\n", gbs >= 0 ? "ok" : "FAILED", gbs);
        if (gbs < 0) { g_fail++; }
        // is the device's write visible to the host through a D2H copy? (the state-save path)
        uint32_t v = 0;
        const hipError_t e = hipMemcpy(&v, off(base, 4096), 4, hipMemcpyDeviceToHost);
        std::printf("    D2H read (copy engine)     %s  0x%08x\n", e == hipSuccess ? "ok" : r(e), v);
        if (e != hipSuccess || v != 0xabcd0001u) { g_fail++; }
        // ... and can the host write it (state-restore path)?
        const uint32_t hv = 0xbeef0002u;
        const hipError_t e2 = hipMemcpy(off(base, 4096), &hv, 4, hipMemcpyHostToDevice);
        (void) hipDeviceSynchronize();
        uint32_t * out = nullptr;
        (void) hipMalloc((void **) &out, 4);
        readback_kernel<<<1, 1>>>((const uint32_t *) off(base, 4096), out);
        uint32_t bv = 0;
        (void) hipDeviceSynchronize();
        (void) hipMemcpy(&bv, out, 4, hipMemcpyDeviceToHost);
        std::printf("    H2D write + device readback %s  0x%08x\n", (e2 == hipSuccess && bv == hv) ? "ok" : "MISMATCH", bv);
        if (e2 != hipSuccess || bv != hv) { g_fail++; }
        (void) hipFree(out);
    }

    // ---- T4: the actual shape -- device chunk and host chunk in ONE range --------------
    std::printf("\nT4 mixed range: [0,%zu) HOST + [%zu,%zu) VRAM, one VA range, one kernel over both\n", mib, mib, 2 * mib);
    const hipDeviceptr_t second = off(base, bytes);
    if (!map_device_chunk(second, bytes, dev)) {
        std::printf("    device chunk into the same reservation FAILED (that alone would sink the shape)\n");
        g_fail++;
    } else {
        write_kernel<<<512, 256>>>((uint32_t *) base, (2 * bytes) / 4, 0x5a5a0003u);
        const hipError_t se = hipDeviceSynchronize();
        std::printf("    fill across BOTH kinds      %s\n", se == hipSuccess ? "ok" : r(se));
        if (se != hipSuccess) { g_fail++; }
        uint32_t a = 0, b = 0;
        (void) hipMemcpy(&a, off(base, bytes / 2), 4, hipMemcpyDeviceToHost);
        (void) hipMemcpy(&b, off(base, bytes + bytes / 2), 4, hipMemcpyDeviceToHost);
        const bool same = a == 0x5a5a0003u && b == 0x5a5a0003u;
        std::printf("    read back both halves       %s (host 0x%08x, vram 0x%08x)\n", same ? "ok" : "MISMATCH", a, b);
        if (!same) { g_fail++; }
        const double gh = read_gbs(base, bytes, sink);
        const double gd = read_gbs(second, bytes, sink);
        std::printf("    read BW  host-backed %6.1f GB/s | vram-backed %6.1f GB/s\n", gh, gd);
        if (gh <= 0 || gd <= 0) { g_fail++; }
        // the FA kernels read one contiguous span; a range that straddles the two kinds must still work
        const double gm = read_gbs(base, 2 * bytes, sink);
        std::printf("    read BW  both halves in one pass  %6.1f GB/s\n", gm);
        if (gm <= 0) { g_fail++; }
    }

    // ---- T5: does the copy engine accept a host-backed chunk? (state save/restore) ----
    std::printf("\nT5 copy engine (hipMemcpy2D / D2D) into and out of the host-backed chunk\n");
    {
        void * d = nullptr;
        const size_t n = 1 << 20;
        if (hipMalloc(&d, n) == hipSuccess) {
            hipError_t e1 = hipMemcpy2D(off(base, 0), 4096, d, 4096, 4096, 8, hipMemcpyDeviceToDevice);
            hipError_t e2 = hipMemcpy2D(d, 4096, off(base, 0), 4096, 4096, 8, hipMemcpyDeviceToDevice);
            (void) hipDeviceSynchronize();
            std::printf("    2D copy  vram->host-backed  %s\n", e1 == hipSuccess ? "ok" : r(e1));
            std::printf("    2D copy  host-backed->vram  %s\n", e2 == hipSuccess ? "ok" : r(e2));
            if (e1 != hipSuccess || e2 != hipSuccess) {
                std::printf("    -> a kernel copy is the fallback for the whole-cache walks (state io, defrag)\n");
                g_fail++;
            }
            (void) hipFree(d);
        }
    }

    // ---- T6: is a host chunk's content zeroed on map? (the KV clear question) ----------
    std::printf("\nT6 fresh host chunk: zeroed or stale?\n");
    {
        hipDeviceptr_t b2 = nullptr;
        if (hipMemAddressReserve(&b2, bytes, 0, 0, 0) == hipSuccess) {
            const char * why = "";
            if (map_chunk(b2, bytes, hipMemLocationTypeHost, 0, &why) ||
                map_chunk(b2, bytes, hipMemLocationTypeHostNuma, numa, &why)) {
                std::vector<uint32_t> sample(64);
                (void) hipMemcpy(sample.data(), b2, sample.size() * 4, hipMemcpyDeviceToHost);
                size_t nz = 0;
                for (uint32_t v : sample) { if (v != 0) { nz++; } }
                std::printf("    64 samples: %zu non-zero -> %s\n", nz,
                            nz ? "STALE -> a KV region must clear on map" : "all zero");
                (void) hipMemUnmap(b2, bytes);
            }
            (void) hipMemAddressFree(b2, bytes);
        }
    }

    // ---- T7: give the memory back, and can the tail be re-pointed somewhere else? -----
    std::printf("\nT7 unmap the host chunk and remap its VA from VRAM (a window that moves)\n");
    {
        const hipError_t u = hipMemUnmap(base, bytes);
        std::printf("    hipMemUnmap(host chunk)     %s\n", u == hipSuccess ? "ok" : r(u));
        if (u != hipSuccess) { g_fail++; }
        if (u == hipSuccess) {
            if (!map_device_chunk(base, bytes, dev)) { std::printf("    remap from VRAM FAILED\n"); g_fail++; }
            else {
                write_kernel<<<64, 256>>>((uint32_t *) base, bytes / 4, 0xcafe0004u);
                const hipError_t se = hipDeviceSynchronize();
                uint32_t v = 0;
                (void) hipMemcpy(&v, base, 4, hipMemcpyDeviceToHost);
                std::printf("    VRAM remap usable           %s 0x%08x\n", (se == hipSuccess && v == 0xcafe0004u) ? "ok" : "FAILED", v);
                if (se != hipSuccess || v != 0xcafe0004u) { g_fail++; }
            }
        }
    }

    (void) hipMemUnmap(base, bytes);
    (void) hipMemUnmap(second, bytes);
    (void) hipMemAddressFree(base, 2 * bytes);

    // ---- T8: the REAL shape -- thousands of 2 MiB chunks, and a per-tensor boundary that is NOT a
    //          multiple of the unit (a KV buffer lays tensors out consecutively, and each tensor's tail
    //          is s*cell_stride bytes: 34 KiB/cell on the 27B at q8_0, so the boundary lands mid-chunk).
    //          Design: bulk 2 MiB chunks + <=2 small chunks per boundary at the 4 KiB minimum.
    std::printf("\nT8 real shape: %zu MiB as 2 MiB chunks + a simulated tensor boundary at 34 KiB granularity\n",
                (size_t) t8_mib);
    {
        const size_t host_total = (size_t) t8_mib << 20;
        const size_t unit       = (size_t) 2 << 20;
        const size_t cell       = 34 * 1024;             // the 27B q8_0 KV cell (K or V), rounded up
        const size_t n_cells    = host_total / cell;
        const size_t tail_cells = n_cells * 3 / 4;       // a 3/4 tail, i.e. a 1/4 resident window
        const size_t tail_bytes = tail_cells * cell;     // the per-tensor host extent
        const size_t n_bulk     = tail_bytes / unit;     // whole 2 MiB host chunks
        const size_t rem        = tail_bytes % unit;     // the boundary remainder, 4 KiB-granular

        hipDeviceptr_t reg = nullptr;
        if (hipMemAddressReserve(&reg, host_total, 0, 0, 0) != hipSuccess) { std::printf("    reserve failed\n"); g_fail++; }
        else {
            const double t0 = now_ms();
            int mapped = 0, failed = 0;
            for (size_t i = 0; i < n_bulk; ++i) {
                hipMemAllocationProp pr = {};
                pr.type = hipMemAllocationTypePinned;
                pr.location.type = hipMemLocationTypeHost;
                pr.location.id = 0;
                hipMemGenericAllocationHandle_t h = 0;
                if (hipMemCreate(&h, unit, &pr, 0) != hipSuccess) { failed++; (void) hipGetLastError(); continue; }
                if (hipMemMap(off(reg, i * unit), unit, 0, h, 0) != hipSuccess) { failed++; (void) hipGetLastError(); (void) hipMemRelease(h); continue; }
                (void) hipMemRelease(h);
                mapped++;
            }
            // the boundary remainder: one small host chunk, rounded to the 4 KiB minimum
            size_t rem_al = (rem + 4095) & ~(size_t) 4095;
            if (rem_al > 0) {
                hipMemAllocationProp pr = {};
                pr.type = hipMemAllocationTypePinned;
                pr.location.type = hipMemLocationTypeHost;
                pr.location.id = 0;
                hipMemGenericAllocationHandle_t h = 0;
                if (hipMemCreate(&h, rem_al, &pr, 0) == hipSuccess &&
                    hipMemMap(off(reg, n_bulk * unit), rem_al, 0, h, 0) == hipSuccess) {
                    (void) hipMemRelease(h);
                    mapped++;
                } else { failed++; (void) hipGetLastError(); }
            }
            hipMemAccessDesc a = {};
            a.location.type = hipMemLocationTypeDevice;
            a.location.id = dev;
            a.flags = hipMemAccessFlagsProtReadWrite;
            const hipError_t ae = hipMemSetAccess(reg, tail_bytes, &a, 1);
            const double t1 = now_ms();

            std::printf("    %zu host chunks (%zu x 2 MiB + 1 x %zu KiB), %d failed, setAccess %s\n",
                        (size_t) mapped, n_bulk, rem_al >> 10, failed, ae == hipSuccess ? "ok" : r(ae));
            std::printf("    create+map total %.1f ms  (%.1f us/chunk, the context-creation cost)\n",
                        t1 - t0, 1e3 * (t1 - t0) / (double) (mapped ? mapped : 1));
            if (failed != 0 || ae != hipSuccess) { g_fail++; }

            write_kernel<<<512, 256>>>((uint32_t *) reg, tail_bytes / 4, 0x11223344u);
            const hipError_t se = hipDeviceSynchronize();
            const double gbs = se == hipSuccess ? read_gbs(reg, tail_bytes, sink) : -1.0;
            std::printf("    device write + read over all %zu MiB: %s  %6.1f GB/s\n",
                        tail_bytes >> 20, se == hipSuccess ? "ok" : r(se), gbs);
            if (se != hipSuccess || gbs <= 0) { g_fail++; }
            uint32_t v = 0;
            (void) hipMemcpy(&v, off(reg, tail_bytes - 4096), 4, hipMemcpyDeviceToHost);
            std::printf("    last mapped 4 KiB readable  %s (0x%08x)\n", v == 0x11223344u ? "ok" : "MISMATCH", v);
            if (v != 0x11223344u) { g_fail++; }

            for (size_t i = 0; i < n_bulk; ++i) { (void) hipMemUnmap(off(reg, i * unit), unit); }
            if (rem_al > 0) { (void) hipMemUnmap(off(reg, n_bulk * unit), rem_al); }
            (void) hipMemAddressFree(reg, host_total);
        }
    }

    (void) hipFree(sink);

    std::printf("\n=== %s (%d failing check(s)) ===\n", g_fail ? "FINDINGS TO ACT ON" : "mixed backing WORKS on this driver", g_fail);
    return g_fail ? 1 : 0;
}
