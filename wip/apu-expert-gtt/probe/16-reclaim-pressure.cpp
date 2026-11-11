// 16-reclaim-pressure.cpp -- CAN the kernel take a REGISTERED region's pages away under real memory pressure?
// (The one probe `../README.md` §5 says the budget's "it is a reservation" claim rests on.)
//
// WHY: probe 09 shows a registered range keeps 1 048 576 of 1 048 576 pages after `fadvise(DONTNEED)` AND
// `madvise(DONTNEED)`, where an unregistered control releases all of them.  But that is a USER-INITIATED
// discard.  The kernel's own reclaim under memory pressure is a different mechanism: `hipHostRegister` pins the
// pages with a page reference (userptr registration), so the shrinker should skip them -- and if it does, the
// GTT store is a reservation.  If it does NOT, a >RAM run can have the GTT store's pages swapped out from under a live
// GPU mapping, which is the catastrophic case (no fault-and-retry on this part: an unregistered or evicted page
// is not in the GPU's page tables, so it is a fatal `HSA_STATUS_ERROR_MEMORY_FAULT`, not a slow path).
//
// WHAT IS MEASURED, per arm, as anonymous-memory pressure grows:
//   * `mincore()` residency of the target region and of an equal-sized UNREGISTERED control in the same process
//   * that is the discriminator: the control must lose pages (swap them out) for the test to have teeth
//   * `/proc/meminfo`: MemFree, MemAvailable, Cached, SwapCached, SwapFree, Unevictable, Mlocked -- because
//     "pinned by the driver" and "on the unevictable LRU" are NOT the same thing, and a monitor that trusts
//     `Unevictable` may be blind to the GTT store
//   * `smaps` of the region: Rss, AnonHugePages, Locked
//   * and for the registered arms, a GPU sum + a host checksum of the same bytes AFTER the pressure: the pattern
//     is written per offset, so a silently swapped-out-then-reinstated page, a zero-fill or a corruption shows up
//
// SAFETY: this probe allocates a large anonymous hog to create the pressure.  It stops when `MemAvailable` falls
// below the floor (`--floor-mib`, default 8192) or the hog reaches `--max-mib` (default 90% of MemTotal), so it
// does not invite the OOM killer.  Nothing here needs root.
//
// Build: hipcc -O2 -o 16-reclaim-pressure 16-reclaim-pressure.cpp
// Run:   ./16-reclaim-pressure ctl     [MiB=4096]
//        ./16-reclaim-pressure reg     [MiB=4096]
//        ./16-reclaim-pressure reg_w   [MiB=4096]
//   (each arm in its own process, like 02-fault: the pressure state must start clean)
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cerrno>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec/1e9; }

// The same byte-sum kernel as 12: order-independent, so it must equal the host's checksum exactly.
__global__ void sum(const unsigned char * p, unsigned long long * out, size_t n) {
    size_t stride = (size_t) gridDim.x * blockDim.x, i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long acc = 0;
    for (; i < n; i += stride) { const volatile unsigned char * q = p; acc += q[i]; }
    atomicAdd(out, acc);
}
static unsigned long long gpu_sum(const void * p, size_t n) {
    unsigned long long * d; CK(hipMalloc(&d, 8)); CK(hipMemset(d, 0, 8));
    sum<<<512, 256>>>((const unsigned char *) p, d, n); CK(hipDeviceSynchronize());
    unsigned long long v; CK(hipMemcpy(&v, d, 8, hipMemcpyDeviceToHost)); (void) hipFree(d); return v;
}
static unsigned long long host_sum(const unsigned char * p, size_t n) {
    unsigned long long v = 0; for (size_t i = 0; i < n; i++) v += p[i]; return v;
}

// The pattern is offset-dependent, so a page that came back as zeros (or as another page) is detectable.
static void fill_pattern(unsigned char * p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char) ((i * 31u + (i >> 13)) & 0xff);
}

static size_t meminfo_kib(const char * key) {
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) return 0;
    char line[256]; const size_t k = strlen(key); size_t out = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (strncmp(line, key, k) == 0 && line[k] == ':') { out = (size_t) strtoull(line + k + 1, nullptr, 10); break; }
    }
    fclose(f);
    return out;
}

static void meminfo_line(const char * tag) {
    printf("  %-22s MemFree %8.0f  MemAvailable %8.0f  Cached %8.0f  SwapCached %6.0f  SwapFree %8.0f  "
           "Unevictable %7.0f  Mlocked %7.0f  (MiB)\n", tag,
           meminfo_kib("MemFree")/1024.0, meminfo_kib("MemAvailable")/1024.0, meminfo_kib("Cached")/1024.0,
           meminfo_kib("SwapCached")/1024.0, meminfo_kib("SwapFree")/1024.0, meminfo_kib("Unevictable")/1024.0,
           meminfo_kib("Mlocked")/1024.0);
}

// smaps of the ONE VMA starting at `base`: Rss / AnonHugePages / Locked.
static void smaps_of(const void * base, const char * tag) {
    FILE * f = fopen("/proc/self/smaps", "r");
    if (f == nullptr || base == nullptr) { if (f) fclose(f); return; }
    // Match the VMA that CONTAINS `base`, not one that starts at it: the kernel merges adjacent anonymous
    // VMAs, so a range we mapped alone can come back as the tail of a bigger one (measured: a 2048 MiB
    // control read as Rss 0.0 with the start-address match while mincore said 2048.0 resident).
    const unsigned long long want = (unsigned long long) (uintptr_t) base;
    char line[512]; bool in = false;
    size_t rss = 0, thp = 0, locked = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (line[0] >= '0' && line[0] <= '9' && strchr(line, '-') != nullptr) {
            unsigned long long lo = 0, hi = 0;
            if (sscanf(line, "%llx-%llx", &lo, &hi) == 2) {
                in = want >= lo && want < hi;
            } else {
                in = false;
            }
            continue;
        }
        if (!in) continue;
        if      (strncmp(line, "Rss:", 4) == 0)            rss    = (size_t) strtoull(line + 4, nullptr, 10);
        else if (strncmp(line, "AnonHugePages:", 14) == 0) thp    = (size_t) strtoull(line + 14, nullptr, 10);
        else if (strncmp(line, "Locked:", 7) == 0)         locked = (size_t) strtoull(line + 7, nullptr, 10);
    }
    fclose(f);
    printf("  %-22s smaps: Rss %8.1f  AnonHugePages %8.1f  Locked %8.1f  (MiB)\n", tag,
           rss/1024.0, thp/1024.0, locked/1024.0);
}

// The memory cgroup this process is in, if any: `memory.current` / `memory.swap.current` / `memory.max`.
// Inside a cgroup the pressure is CONTAINED (the kernel reclaims within the cgroup), so this is both the
// safe way to run the probe and the proof that a swap-out really happened.
static void cgroup_line(const char * tag) {
    FILE * f = fopen("/proc/self/cgroup", "r");
    if (f == nullptr) return;
    char path[512] = {0}, line[600];
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (line[0] == '0' && line[1] == ':') {   // cgroup v2: "0::/path"
            char * p = strchr(line + 2, '/');
            if (p != nullptr) { snprintf(path, sizeof(path), "%s", p); path[strcspn(path, "\n")] = 0; }
            break;
        }
    }
    fclose(f);
    if (path[0] == 0) return;
    char base[600];
    snprintf(base, sizeof(base), "/sys/fs/cgroup%s", strcmp(path, "/") == 0 ? "" : path);
    auto read1 = [](const char * p, const char * key) -> double {
        FILE * g = fopen(p, "r"); if (g == nullptr) return -1;
        char l[128]; double v = -1;
        while (fgets(l, sizeof(l), g) != nullptr) {
            if (strncmp(l, key, strlen(key)) == 0) { v = strtod(l + strlen(key), nullptr); break; }
        }
        fclose(g); return v;
    };
    char f1[700], f2[700], f3[700];
    snprintf(f1, sizeof(f1), "%s/memory.current", base);
    snprintf(f2, sizeof(f2), "%s/memory.swap.current", base);
    snprintf(f3, sizeof(f3), "%s/memory.max", base);
    const double cur = read1(f1, ""), swp = read1(f2, ""), mx = read1(f3, "");
    if (cur < 0 && mx < 0) return;
    printf("  %-22s cgroup %-40s memory.current %7.0f  memory.swap.current %6.0f  memory.max %7.0f  (MiB)\n",
           tag, path, cur / 1048576.0, swp / 1048576.0, mx < 0 ? -1 : mx / 1048576.0);
}

// Resident (in-RAM) pages of a range, in MiB.  A swapped-out page is NOT resident -> mincore() returns 0 for it.
static double mincore_resident_mib(void * base, size_t n) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t npg  = n / page;
    unsigned char * vec = (unsigned char *) malloc(npg);
    if (vec == nullptr) return -1;
    if (mincore(base, n, vec) != 0) {
        printf("  mincore: %s\n", strerror(errno));
        free(vec);
        return -1;
    }
    size_t in = 0;
    for (size_t i = 0; i < npg; i++) if (vec[i] & 1) in++;
    free(vec);
    return (double) in * page / 1048576.0;
}

int main(int argc, char ** argv) {
    const char * arm = argc > 1 ? argv[1] : "reg";
    const size_t SZ  = (size_t) (argc > 2 ? atoll(argv[2]) : 4096) << 20;
    size_t floor_mib = 8192, max_mib = 0;
    {
        // the floor grows with the region so a big --region still leaves the machine alone
        FILE * f = fopen("/proc/meminfo", "r");
        size_t total_mib = 0;
        if (f != nullptr) {
            char line[256];
            while (fgets(line, sizeof(line), f) != nullptr) {
                if (strncmp(line, "MemTotal:", 9) == 0) { total_mib = (size_t) strtoull(line + 9, nullptr, 10) / 1024; break; }
            }
            fclose(f);
        }
        max_mib = total_mib ? total_mib * 90 / 100 : 110000;
        const char * e = getenv("PROBE16_FLOOR_MIB");
        if (e != nullptr) floor_mib = (size_t) atoll(e);
        if (const char * h = getenv("PROBE16_HOG_MIB")) max_mib = (size_t) atoll(h);   // e.g. inside a memory cgroup
    }
    const bool ctl  = strcmp(arm, "ctl") == 0;
    const bool hall = strcmp(arm, "hostalloc") == 0;   // hipHostMalloc: a PINNED allocation, not a registration
    const bool mlok = strcmp(arm, "mlock") == 0;        // mmap + mlock: a reservation that KEEPS the huge pages
    const bool ro   = strcmp(arm, "reg_w") != 0;        // reg = ReadOnly, reg_w = writable

    printf("16-reclaim-pressure: arm=%s region=%zu MiB floor=%zu MiB hog max=%zu MiB\n", arm, SZ >> 20, floor_mib, max_mib);

    // `hostalloc` is the control for "is it the REGISTRATION or is it the PIN?": hipHostMalloc pins at
    // allocation (the delivery's `ROCm_Host` master does this), where hipHostRegister on an mmap does not.
    unsigned char * reg = nullptr;
    if (hall) {
        if (hipHostMalloc((void **) &reg, SZ, hipHostMallocMapped | hipHostMallocPortable) != hipSuccess) {
            printf("hipHostMalloc(%.0f MiB) failed\n", (double) SZ / 1048576.0);
            return 2;
        }
    } else {
        reg = (unsigned char *) mmap(nullptr, SZ, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (reg == MAP_FAILED) { perror("mmap reg"); return 2; }
    }
    if (!hall) {
        madvise(reg, SZ, MADV_HUGEPAGE);
    }
    fill_pattern(reg, SZ);
    if (mlok) {
        // The third answer to "make it not-reclaimable": keep the anonymous mapping (and its huge pages) and
        // make the pages unevictable explicitly.  Needs RLIMIT_MEMLOCK (`ulimit -l`); it is UNLIMITED on this
        // box, which is why this arm is worth measuring -- a stock 8 MiB limit would refuse a real GTT store.
        if (mlock(reg, SZ) != 0) {
            printf("  mlock(%.0f MiB) FAILED: %s  (check `ulimit -l`)\n", (double) SZ / 1048576.0, strerror(errno));
            return 2;
        }
        printf("  mlock(%.0f MiB) OK (RLIMIT_MEMLOCK permits it)\n", (double) SZ / 1048576.0);
    }

    // An equal-sized UNREGISTERED control, faulted in and filled the same way.  This is what tells us the test
    // has teeth: under the same pressure it must lose pages, or the pressure was never real.
    unsigned char * ctl_mem = (unsigned char *) mmap(nullptr, SZ, PROT_READ | PROT_WRITE,
                                                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ctl_mem == MAP_FAILED) { perror("mmap ctl"); return 2; }
    fill_pattern(ctl_mem, SZ);

    unsigned long long cpu_before = host_sum(reg, SZ);
    unsigned long long gpu_before = 0;
    double gpu_ms_before = 0;
    void * dev = reg;
    if (!ctl) {
        if (!hall) {
            CK(hipHostRegister(reg, SZ, hipHostRegisterMapped | (ro ? hipHostRegisterReadOnly : 0u)));
        }
        CK(hipHostGetDevicePointer(&dev, reg, 0));
        const double t0 = now();
        gpu_before = gpu_sum(dev, SZ);
        gpu_ms_before = (now() - t0) * 1e3;
        printf("  %s; CPU sum %llu / GPU sum %llu %s (GPU read %.1f ms)\n",
               hall ? "hipHostMalloc (PINNED)" : (ro ? "registered ReadOnly" : "registered writable"),
               cpu_before, gpu_before,
               cpu_before == gpu_before ? "(match)" : "(MISMATCH)", gpu_ms_before);
    } else {
        printf("  CONTROL arm: nothing registered (this is the *unpinned* reference)\n");
    }

    // Let any transient dirty-page writeback happen before the baseline, or the baseline is measured mid-drain.
    sync();
    meminfo_line("before pressure");
    cgroup_line("before pressure");
    smaps_of(reg, "target (registered)");   // for the ctl arm this is just an anon range
    smaps_of(ctl_mem, "control (anon)");
    printf("  resident: target %.1f MiB, control %.1f MiB\n",
           mincore_resident_mib(reg, SZ), mincore_resident_mib(ctl_mem, SZ));

    // ---- the pressure ---------------------------------------------------------------------------------------
    // Grow an anonymous hog in 1 GiB steps, touching every page, and re-measure.  Anonymous (not page cache)
    // because page-cache pressure never forces an anon page out -- only real anon pressure with swap does.
    const size_t step = 1ull << 30;
    size_t hog_mib = 0;
    unsigned char * hog = nullptr;
    size_t hog_cap = 0;
    const size_t check_every_mib = 8192;
    double t_press = now();
    while (hog_mib < max_mib) {
        if (hog_mib == hog_cap) {
            const size_t old_cap = hog_cap;
            hog_cap += 16384;   // reserve VA in 16 GiB steps; only the touched prefix is real
            void * p;
            if (hog == nullptr) {
                p = mmap(nullptr, hog_cap << 20, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            } else {
                p = mremap(hog, old_cap << 20, hog_cap << 20, MREMAP_MAYMOVE);
            }
            if (p == MAP_FAILED) { perror("hog mmap/mremap"); break; }
            hog = (unsigned char *) p;
        }
        unsigned char * page = hog + ((size_t) hog_mib << 20);
        for (size_t i = 0; i < step; i += 4096) page[i] = (unsigned char) (i >> 12);
        hog_mib += step >> 20;

        if (meminfo_kib("MemAvailable") / 1024 < floor_mib) {
            printf("  hog %zu MiB -> MemAvailable under the %zu MiB floor, stopping\n", hog_mib, floor_mib);
            break;
        }
        if (hog_mib % check_every_mib == 0) {
            const double t = mincore_resident_mib(reg, SZ), c = mincore_resident_mib(ctl_mem, SZ);
            printf("  hog %6zu MiB  MemAvail %8.0f  target resident %7.1f  control resident %7.1f  SwapFree %8.0f\n",
                   hog_mib, meminfo_kib("MemAvailable") / 1024.0, t, c, meminfo_kib("SwapFree") / 1024.0);
            fflush(stdout);
        }
    }
    const double press_s = now() - t_press;
    printf("  pressure: %zu MiB of anonymous hog in %.1f s\n", hog_mib, press_s);

    sync();
    meminfo_line("after pressure");
    cgroup_line("after pressure");
    smaps_of(reg, "target (registered)");
    smaps_of(ctl_mem, "control (anon)");
    const double res_t = mincore_resident_mib(reg, SZ);
    const double res_c = mincore_resident_mib(ctl_mem, SZ);
    printf("  resident: target %.1f of %.1f MiB, control %.1f of %.1f MiB\n",
           res_t, (double) SZ / 1048576.0, res_c, (double) SZ / 1048576.0);

    // ---- the verdict ---------------------------------------------------------------------------------------
    // ORDER MATTERS AND IT IS THE WHOLE POINT: the GPU reads the region FIRST, before anything on the host
    // touches it.  A host read of a swapped-out page faults it back in, so a host-first order hides exactly the
    // case being measured -- and a kernel that took a page the driver cannot reinstate turns the GPU read into a
    // fatal fault (no XNACK, no retry), which kills this process by design (probe 02 shows the message).
    unsigned long long gpu_after = 0;
    double gpu_ms_after = 0;
    bool gpu_ok = true;
    if (!ctl) {
        const double t0 = now();
        gpu_after = gpu_sum(dev, SZ);
        gpu_ms_after = (now() - t0) * 1e3;
        printf("  GPU read AFTER the pressure (the first touch of the region): %.1f ms, sum %llu\n",
               gpu_ms_after, gpu_after);
        const double res_after_gpu = mincore_resident_mib(reg, SZ);
        const double t1 = now();
        const unsigned long long gpu_again = gpu_sum(dev, SZ);
        const double gpu_ms_again = (now() - t1) * 1e3;
        printf("  target resident after the GPU read: %.1f MiB; a SECOND GPU read takes %.1f ms (sum %llu %s)\n",
               res_after_gpu, gpu_ms_again, gpu_again, gpu_again == gpu_after ? "(same)" : "(CHANGED)");
    }
    // Only now the host reads it: content check (a page reinstated as zeros is the probe-06 trap and no residency
    // counter would show it), and for the GPU-read arms the sum must equal what the GPU just read.
    const unsigned long long cpu_after = host_sum(reg, SZ);
    const bool content_ok = cpu_after == cpu_before;
    if (!ctl) {
        gpu_ok = gpu_after == cpu_after;
    }
    printf("  content: CPU sum %llu -> %llu %s\n", cpu_before, cpu_after, content_ok ? "(intact)" : "(CHANGED)");
    if (!ctl) {
        printf("  gpu read: %llu %s (%.1f ms before the pressure, %.1f ms after)\n", gpu_after,
               gpu_ok ? "(matches)" : "(MISMATCH)", gpu_ms_before, gpu_ms_after);
    }

    // Free the hog so the box goes back to normal before we say anything else.
    if (hog != nullptr) munmap(hog, hog_cap << 20);
    const double control_lost = (double) SZ / 1048576.0 - res_c;
    printf("\n  VERDICT (%s, %zu MiB):\n", arm, SZ >> 20);
    printf("    control (unregistered) lost %.1f MiB of pages -> the pressure was %s\n",
           control_lost, control_lost > 1.0 ? "REAL" : "NOT REAL ENOUGH (raise --hog or lower the floor)");
    printf("    target lost %.1f MiB of pages -> %s\n", (double) SZ / 1048576.0 - res_t,
           res_t >= (double) SZ / 1048576.0 - 1.0 ? "HELD (the kernel did not take a registered page)"
                                                  : "RECLAIMED (the budget is NOT a reservation)");
    printf("    content %s, gpu %s\n", content_ok ? "intact" : "CORRUPTED",
           ctl ? "n/a" : (gpu_ok ? "matches" : "MISMATCH"));

    if (!ctl) {
        if (hall) {
            CK(hipHostFree(reg));
        } else {
            CK(hipHostUnregister(reg));
            if (mlok) {
                munlock(reg, SZ);
            }
            munmap(reg, SZ);
        }
    } else {
        munmap(reg, SZ);
    }
    munmap(ctl_mem, SZ);
    const bool held   = res_t >= (double) SZ / 1048576.0 - 1.0;
    const bool teeth  = control_lost > 1.0;
    return (content_ok && gpu_ok && (ctl ? true : held) && teeth) ? 0 : 1;
}
