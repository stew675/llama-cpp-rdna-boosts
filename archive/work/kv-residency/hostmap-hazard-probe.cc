// wip/kv-residency/hostmap-hazard-probe.cc -- the VMM host-mapping hazard, isolated (TODO #118, A2 §3.7.9).
//
// WHY THIS EXISTS
//   The windowed KV keeps one VA reservation whose cells are backed partly by VRAM and partly by pinned system
//   RAM, and the resident window SLIDES: a mapping is flipped device -> host as the cell watermark advances.
//   Measured 2026-10-30: after such a flip a DIFFERENT, live host-location mapping in the same reservation
//   reads back as all zeros -- deterministically, from the 6th flip on.  In the model that shifted the
//   prefill logits (KLD 0.004973 / 98.761 % on a 27B q8_0 arm) and changed the greedy text of every arm that
//   actually spilled, and it is why the campaign's "the 4B is immune" reading was wrong (those 4B arms never
//   reached a flip at all).
//
// WHAT IT SHOWS
//   Replays the real shape -- 32 tensors of 8.5 MiB (a 27B q8_0 KV at -c 8192), each with a 4 MiB window
//   mapped from VRAM at setup and flipped to pinned host one tensor at a time -- and re-verifies EVERY
//   previously flipped segment after each flip.  The default arm fails; the arms below separate the cause:
//
//     (default)          FAIL   unmap(device) + map(host) at the same VA, then the spill loses a segment
//     --grab N           PASS   hold the physical the unmap just released (N MiB, >= the unit) across the map
//     --precreate        FAIL   creating the host handle BEFORE the unmap does not help
//     --no-flip          FAIL   mapping the destination as VRAM instead of host does not help
//     --no-stage         FAIL   dropping the pinned staging round trip does not help
//     --vram-mib N       FAIL   (with N GiB of VRAM already allocated) pressure is not the trigger
//     --cpu-store        SIGSEGV the VMM VA is not CPU addressable -- which is why the staging legs are kernels
//
//   So the fault is the RELEASE by unmap feeding the following create, and `vmm_map` now takes a hold for the
//   duration of every create (`LLAMA_KV_REGION_QUARANTINE`, default on; `=0` is the A/B arm that reproduces
//   this probe's failure inside the model).
//
// Build / run (the -isystem is mandatory -- without it HIP resolves from Fedora's ROCm 7.1 in /usr/include):
//   hipcc -O3 --offload-arch=gfx1201 -isystem /opt/rocm-10.1.0-gfx120X/include \
//         wip/kv-residency/hostmap-hazard-probe.cc -o /tmp/hostmap-probe
//   HIP_VISIBLE_DEVICES=0 /tmp/hostmap-probe            # FAIL, the hazard
//   HIP_VISIBLE_DEVICES=0 /tmp/hostmap-probe --grab 4   # PASS, the workaround
//
// It is a probe, not a gate: the in-tree regression should end up in GGML_CUDA_VMM_REGION_SELFTEST.

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fails = 0;
static inline void * at(void * base, size_t off) { return (char *) base + off; }

__global__ void cp16(uint4 * dst, const uint4 * src, size_t n4) {
    for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x*blockDim.x) dst[i] = src[i];
}
__global__ void pat16(uint4 * p, size_t n4, unsigned v) {
    for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x*blockDim.x) {
        uint4 x; x.x = v + i; x.y = v + i; x.z = v + i; x.w = v + i; p[i] = x;
    }
}
static bool devcopy(void * dst, const void * src, size_t bytes) {
    cp16<<<512,256>>>((uint4*)dst, (const uint4*)src, bytes/16);
    return hipDeviceSynchronize() == hipSuccess;
}
static bool mapm(void * base, size_t off, size_t bytes, bool host) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = host ? hipMemLocationTypeHost : hipMemLocationTypeDevice;
    prop.location.id = host ? 0 : 0;
    hipMemGenericAllocationHandle_t h = 0;
    if (hipMemCreate(&h, bytes, &prop, 0) != hipSuccess) { printf("  create %s\n", hipGetErrorString(hipGetLastError())); return false; }
    if (hipMemMap((hipDeviceptr_t)at(base, off), bytes, 0, h, 0) != hipSuccess) { printf("  map %s\n", hipGetErrorString(hipGetLastError())); hipMemRelease(h); return false; }
    hipMemAccessDesc a = {};
    a.location.type = hipMemLocationTypeDevice; a.location.id = 0; a.flags = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess((hipDeviceptr_t)at(base, off), bytes, &a, 1) != hipSuccess) printf("  setaccess %s\n", hipGetErrorString(hipGetLastError()));
    hipMemRelease(h);
    return true;
}
static hipMemGenericAllocationHandle_t makeh(size_t bytes, bool host) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = host ? hipMemLocationTypeHost : hipMemLocationTypeDevice;
    prop.location.id = host ? 0 : 0;
    hipMemGenericAllocationHandle_t h = 0;
    if (hipMemCreate(&h, bytes, &prop, 0) != hipSuccess) { printf("  create %s\n", hipGetErrorString(hipGetLastError())); return 0; }
    return h;
}
static bool useh(void * base, size_t off, size_t bytes, bool host, hipMemGenericAllocationHandle_t h) {
    if (hipMemMap((hipDeviceptr_t)at(base, off), bytes, 0, h, 0) != hipSuccess) { printf("  map %s\n", hipGetErrorString(hipGetLastError())); hipMemRelease(h); return false; }
    hipMemAccessDesc a = {};
    a.location.type = hipMemLocationTypeDevice; a.location.id = 0; a.flags = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess((hipDeviceptr_t)at(base, off), bytes, &a, 1) != hipSuccess) printf("  setaccess %s\n", hipGetErrorString(hipGetLastError()));
    hipMemRelease(h);
    return true;
}
static void unmapm(void * base, size_t off, size_t bytes) { hipMemUnmap((hipDeviceptr_t)at(base, off), bytes); }

static bool g_no_stage = false, g_no_flip = false, g_precreate = false, g_cpu_store = false, g_cpu_touch = false;
static size_t g_grab_mib = 0;
int main(int argc, char ** argv) {
    size_t vram_mib = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--vram-mib") && i+1 < argc) vram_mib = atoi(argv[++i]);
        if (!strcmp(argv[i], "--no-stage")) g_no_stage = true;
        if (!strcmp(argv[i], "--no-flip"))  g_no_flip  = true;
        if (!strcmp(argv[i], "--precreate")) g_precreate = true;
        if (!strcmp(argv[i], "--cpu-store")) g_cpu_store = true;
        if (!strcmp(argv[i], "--cpu-touch")) g_cpu_touch = true;
        if (!strcmp(argv[i], "--grab") && i+1 < argc) g_grab_mib = atoi(argv[++i]);
    }
    std::vector<void*> hog;
    for (size_t done = 0; vram_mib && done < vram_mib; done += 1024) {
        void * p = nullptr; if (hipMalloc(&p, (size_t)1024 << 20) != hipSuccess) { (void) hipGetLastError(); break; } hog.push_back(p);
    }
    const size_t TENS = 8912896, UNIT = 4194304; const int NT = 32; const size_t SIZE = TENS*NT;
    hipDeviceptr_t base = 0;
    if (hipMemAddressReserve(&base, SIZE, 0, 0, 0) != hipSuccess) { printf("reserve failed\n"); return 1; }
    // app-identical staging: a pinned host stage and a device stage
    void * stage = nullptr, * devstage = nullptr;
    hipHostMalloc(&stage, UNIT); hipMalloc(&devstage, UNIT);
    for (int t = 0; t < NT; t++) {
        if (!mapm((void*)base, t*TENS, UNIT, false)) return 1;   // window, device
        if (!mapm((void*)base, t*TENS + UNIT, UNIT, false)) return 1;
        pat16<<<256,256>>>((uint4*)at((void*)base, t*TENS), UNIT/16, 0x100000u*t); hipDeviceSynchronize();
    }
    for (int t = 0; t < NT; t++) {
        size_t off = t*TENS;
        // ---- vmm_detach ----
        hipMemGenericAllocationHandle_t pre = 0;
        if (g_precreate) { pre = makeh(UNIT, !g_no_flip); if (!pre) return 1; }
        hipDeviceSynchronize();
        if (!devcopy(devstage, at((void*)base, off), UNIT)) { printf("stageout copy failed\n"); return 1; }
        if (!g_no_stage) hipMemcpy(stage, devstage, UNIT, hipMemcpyDeviceToHost);
        unmapm((void*)base, off, UNIT);
        void * grabbed = nullptr;
        if (g_grab_mib) {   // hold the physical the unmap just released, so the next create cannot reuse it
            if (hipMalloc(&grabbed, g_grab_mib << 20) != hipSuccess) { (void) hipGetLastError(); grabbed = nullptr; }
        }
        // ---- vmm_attach ----
        if (g_precreate) { if (!useh((void*)base, off, UNIT, !g_no_flip, pre)) return 1; }
        else if (!mapm((void*)base, off, UNIT, !g_no_flip)) return 1;
        if (g_cpu_touch) { volatile char * q = (volatile char *)at((void*)base, off); for (size_t i = 0; i < UNIT; i += 64) q[i] = q[i]; }
        if (g_cpu_store) {
            memcpy(at((void*)base, off), stage, UNIT);   // CPU store straight into the host-location mapping
        } else {
            if (!g_no_stage) hipMemcpy(devstage, stage, UNIT, hipMemcpyHostToDevice);
            if (!devcopy(at((void*)base, off), devstage, UNIT)) { printf("restore copy failed\n"); return 1; }
        }
        if (grabbed) { (void) hipFree(grabbed); }
        hipDeviceSynchronize();
        // verify this one + all earlier flipped (=host) windows
        for (int v = 0; v <= t; v++) {
            std::vector<uint32_t> got(UNIT/4);
            devcopy(devstage, at((void*)base, (size_t)v*TENS), UNIT);
            hipMemcpy(got.data(), devstage, UNIT, hipMemcpyDeviceToHost);
            size_t bad = 0;
            for (size_t i = 0; i < got.size(); i++) if (got[i] != 0x100000u*v + (uint32_t)(i/4)*0 + (uint32_t)(i/4)) bad++;
            // expected: pat16 wrote x=v+i (i = uint4 index) into 4 words -> word w has value v + w/4
            bad = 0;
            for (size_t w = 0; w < got.size(); w++) if (got[w] != 0x100000u*v + (uint32_t)(w/4)) { bad++; }
            if (bad) { printf("flip t=%2d off %6.2f MiB: segment v=%2d off %6.2f MiB CORRUPT %zu words, first=%08x exp=%08x\n",
                        t, off/1048576.0, v, v*TENS/1048576.0, bad, got[0], 0x100000u*v); fails++; break; }
        }
    }
    printf("%s (%d failing checks)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
