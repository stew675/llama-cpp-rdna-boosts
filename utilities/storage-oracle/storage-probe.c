// storage-probe.c -- the measurement engine behind utilities/storage-oracle.
//
// Design rules (see README.md -- "Why it is built this way"):
//   * COLD BY CONSTRUCTION, NO ROOT, NO CLEARING THE PAGE CACHE.  O_DIRECT reads bypass the cache
//     entirely; the page-cache mechanisms (buffered pread, mmap faults) evict just the 4 KiB they are
//     about to touch with POSIX_FADV_DONTNEED, so every sample is a genuine device read and the
//     operator's machine is left as it was found.
//   * ONE LINE OF key=value PER MEASUREMENT on stdout.  The driver (storage-oracle.sh) parses that, so
//     the engine never has to know what a verdict is.
//   * NO THREADS, NO ASYNC.  Everything is QD1 unless the driver deliberately runs two probes against
//     the same file at once (that is the "cold read under a stream" arm).
//
// Usage:
//   storage-probe map    FILE CHUNK_MIB STEPS [BASE_MIB] [SPAN_MIB]   band map, O_DIRECT
//   storage-probe bulk   FILE dio|buf IO_MIB COUNT [BASE_MIB]         sequential throughput
//   storage-probe rand   FILE dio|buf|mmap IO_BYTES N REGION_MIB [BASE_MIB] [GAP_MS]
//   storage-probe pop    FILE REGION_MIB [BASE_MIB]                   mmap(MAP_POPULATE) fault-in
//   storage-probe cached FILE                                         sampled page-cache residency
//
// Build: cc -O2 -o storage-probe storage-probe.c

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define ALIGN 4096
#define GIB  (1024.0 * 1024.0 * 1024.0)
#define MIB  (1024.0 * 1024.0)

static void die(const char * what) {
    fprintf(stderr, "storage-probe: %s: %s\n", what, strerror(errno));
    exit(1);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
}

static uint64_t rnd64(void) {              // xorshift64*
    static uint64_t s = 0x9e3779b97f4a7c15ull;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 0x2545f4914f6cdd1dull;
}

static int cmp_u64(const void * a, const void * b) {
    const uint64_t x = *(const uint64_t *) a, y = *(const uint64_t *) b;
    return x < y ? -1 : (x > y);
}

static uint64_t at(const uint64_t * t, size_t n, double q) {   // q in [0,1]
    size_t i = (size_t) (q * (double) (n - 1) + 0.5);
    return t[i > n - 1 ? n - 1 : i];
}

static off_t size_of(int fd) {
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat");
    if (st.st_size > 0) return st.st_size;
    return lseek(fd, 0, SEEK_END);          // block device
}

static uint64_t arg_u(const char * s) { return strtoull(s, NULL, 0); }

static int open_dio(const char * p) {
    int fd = open(p, O_RDONLY | O_DIRECT);
    if (fd < 0) die("open(O_DIRECT)");
    return fd;
}
static int open_buf(const char * p) {
    int fd = open(p, O_RDONLY);
    if (fd < 0) die("open");
    return fd;
}

// ---------------------------------------------------------------- making a page cold
// POSIX_FADV_DONTNEED is only advisory AND it works at FOLIO granularity: on a filesystem that caches
// large folios (XFS does), dropping a 4 KiB range out of a 2 MiB folio silently does nothing and the
// "cold" read comes back from the cache.  Measured here before this check existed: p50 0.2 us on a
// file that had 39 GiB of hot page cache.  So: ask politely, then VERIFY with mincore on a bare
// mapping (no faults), escalating the window until the page is really gone.  A sample that cannot be
// made cold is counted, never timed -- a garbage number is worse than no number.
#define EVICT_WINDOWS 3
static const size_t evict_win[EVICT_WINDOWS] = { 2u << 20, 32u << 20, 256u << 20 };

// returns 1 if the page at off is now NOT resident, 0 if it could not be evicted
static int make_cold(int fd, char * map, off_t base, off_t region, off_t off, long pg) {
    unsigned char v = 1;
    for (int i = 0; i < EVICT_WINDOWS; i++) {
        const off_t w = (off_t) evict_win[i];
        const off_t s = off & ~(w - 1);
        const off_t e = s + w;
        if (map != NULL) {
            // Our own PTEs pin the folio against fadvise: zap them first, otherwise the page we
            // faulted last iteration can never be dropped and every sample would look warm.
            const off_t ms = s > base ? s : base;
            const off_t me = e < base + region ? e : base + region;
            if (me > ms) madvise(map + (ms - base), (size_t) (me - ms), MADV_DONTNEED);
        }
        posix_fadvise(fd, s, w, POSIX_FADV_DONTNEED);
        if (mincore(map + (off - base), (size_t) pg, &v) != 0) return 0;
        if (!(v & 1)) return 1;
    }
    return 0;
}

// read exactly io bytes at off, retrying short reads
static void read_exact(int fd, void * buf, size_t io, off_t off) {
    size_t done = 0;
    while (done < io) {
        ssize_t r = pread(fd, (char *) buf + done, io - done, off + (off_t) done);
        if (r < 0) {
            if (errno == EINTR) continue;
            die("pread");
        }
        if (r == 0) { fprintf(stderr, "storage-probe: short read at %lld\n", (long long) off); exit(1); }
        done += (size_t) r;
    }
}

// ---------------------------------------------------------------- map: the band map
// Reads CHUNK_MIB with O_DIRECT at STEPS evenly spaced offsets, so the driver can see whether the
// file sits in a uniformly fast region of the device or straddles a slow band (measured today: the
// same shard is 1.7 GB/s in its first 4 GiB and 7.6 GB/s from 8 GiB on -- a whole-file average hides
// that, and with it the only actionable finding).
static int cmd_map(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "map FILE CHUNK_MIB STEPS [BASE_MIB] [SPAN_MIB]\n"); return 2; }
    const char * path  = argv[2];
    size_t chunk = arg_u(argv[3]) * (size_t) MIB;
    size_t steps = arg_u(argv[4]);
    off_t  base  = (off_t) (argc > 5 ? arg_u(argv[5]) : 0) * (off_t) MIB;
    if (steps == 0 || chunk == 0) return 2;

    int fd = open_dio(path);
    const off_t fsize = size_of(fd);
    off_t span = argc > 6 ? (off_t) arg_u(argv[6]) * (off_t) MIB : fsize - base;
    if (span <= 0) { fprintf(stderr, "map: empty span\n"); return 1; }
    off_t stride = (span / (off_t) steps) & ~(off_t) (ALIGN - 1);
    if (stride < (off_t) chunk) stride = (off_t) chunk;

    void * buf = NULL;
    if (posix_memalign(&buf, ALIGN, chunk)) die("posix_memalign");

    for (size_t i = 0; i < steps; i++) {
        off_t off = base + (off_t) i * stride;
        if (off + (off_t) chunk > fsize) off = (fsize - (off_t) chunk) & ~(off_t) (ALIGN - 1);
        if (off < 0) break;
        double t0 = now_s();
        read_exact(fd, buf, chunk, off);
        double dt = now_s() - t0;
        // A fully-zero first 4 KiB means the extent is very likely unwritten/trimmed, which a
        // controller answers from its zero-fill path at bus speed -- label it so the driver never
        // mistakes free space for a fast band.
        unsigned zero = 1;
        for (size_t k = 0; k < ALIGN; k++) if (((unsigned char *) buf)[k]) { zero = 0; break; }
        printf("PROBE map step=%zu off_gib=%.1f gbs=%.2f secs=%.3f zeros=%u\n",
               i, (double) off / GIB, (double) chunk / GIB / dt, dt, zero);
        fflush(stdout);
    }
    free(buf); close(fd);
    return 0;
}

// ---------------------------------------------------------------- bulk: sequential throughput
static int cmd_bulk(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "bulk FILE dio|buf IO_MIB COUNT [BASE_MIB]\n"); return 2; }
    const char * path = argv[2];
    const int    dio  = strcmp(argv[3], "dio") == 0;
    size_t io    = arg_u(argv[4]) * (size_t) MIB;
    size_t n     = arg_u(argv[5]);
    off_t  base  = (off_t) (argc > 6 ? arg_u(argv[6]) : 0) * (off_t) MIB;

    int fd = dio ? open_dio(path) : open_buf(path);
    const off_t fsize = size_of(fd);
    off_t off = base & ~(off_t) (ALIGN - 1);
    if (n * io > (size_t) (fsize - off)) n = (size_t) (fsize - off) / io;

    void * buf = NULL;
    if (posix_memalign(&buf, ALIGN, io)) die("posix_memalign");

    double t0 = now_s();
    for (size_t i = 0; i < n; i++) read_exact(fd, buf, io, off + (off_t) i * (off_t) io);
    double dt = now_s() - t0;
    double bytes = (double) n * (double) io;

    printf("PROBE bulk mode=%s io_mib=%zu count=%zu base_mib=%lld bytes=%.0f secs=%.3f gbs=%.2f\n",
           dio ? "dio" : "buf", io / (size_t) MIB, n, (long long) (base / (off_t) MIB), bytes, dt,
           bytes / GIB / dt);
    free(buf); close(fd);
    return 0;
}

// ---------------------------------------------------------------- rand: cold per-op latency
static int cmd_rand(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "rand FILE dio|buf|mmap IO_BYTES N REGION_MIB [BASE_MIB] [GAP_MS]\n"); return 2; }
    const char * path = argv[2];
    const char * mode = argv[3];
    size_t io     = arg_u(argv[4]);
    size_t n      = arg_u(argv[5]);
    size_t region = arg_u(argv[6]) * (size_t) MIB;
    off_t  base   = (off_t) (argc > 7 ? arg_u(argv[7]) : 0) * (off_t) MIB;
    size_t gap_ms = argc > 8 ? (size_t) arg_u(argv[8]) : 0;

    io = (io + ALIGN - 1) & ~(size_t) (ALIGN - 1);
    base &= ~(off_t) (ALIGN - 1);

    const int is_dio  = strcmp(mode, "dio")  == 0;
    const int is_mmap = strcmp(mode, "mmap") == 0;
    int fd = is_dio ? open_dio(path) : open_buf(path);
    const off_t fsize = size_of(fd);
    if (region == 0) region = (size_t) (fsize - base);
    if (base + (off_t) region > fsize) region = (size_t) (fsize - base);
    region &= ~(size_t) (ALIGN - 1);

    size_t  nblocks = region / io;
    if (nblocks < 2) { fprintf(stderr, "rand: region too small\n"); return 1; }

    void * buf = NULL;
    char * mp  = NULL;
    const long pg = sysconf(_SC_PAGESIZE);
    if (posix_memalign(&buf, ALIGN, io)) die("posix_memalign");
    if (!is_dio) {   // the mapping is the eviction check, and the read itself in mmap mode
        mp = mmap(NULL, region, PROT_READ, MAP_SHARED | MAP_NORESERVE, fd, base);
        if (mp == MAP_FAILED) die("mmap");
    }

    uint64_t * t = calloc(n, sizeof *t);
    if (!t) die("calloc");
    size_t n1 = 0, n5 = 0, n50 = 0, done = 0, resid = 0;

    const double wall0 = now_s();
    for (size_t i = 0; i < n; i++) {
        off_t off = base + (off_t) (rnd64() % nblocks) * (off_t) io;
        if (gap_ms) usleep(gap_ms * 1000);
        if (!is_dio && !make_cold(fd, mp, base, (off_t) region, off, pg)) { resid++; continue; }
        const double a = now_s();
        if (is_mmap) {
            volatile unsigned char c = mp[off - base];   // one fault == one device read
            (void) c;
        } else {
            read_exact(fd, buf, io, off);
        }
        const double d = now_s() - a;
        t[done++] = (uint64_t) (d * 1e6);
        if (d > 0.001)              n1++;
        if (d > 0.005)              n5++;
        if (d > 0.050)             n50++;
    }
    const double wall = now_s() - wall0;
    if (done == 0) { fprintf(stderr, "storage-probe: no sample could be made cold\n"); return 1; }

    qsort(t, done, sizeof *t, cmp_u64);
    uint64_t sum = 0;
    for (size_t i = 0; i < done; i++) sum += t[i];

    printf("PROBE rand mode=%s io=%zu n=%zu region_mib=%.0f base_mib=%lld gap_ms=%zu"
           " min_us=%.1f p50_us=%.1f p90_us=%.1f p99_us=%.1f max_us=%.1f mean_us=%.1f"
           " mib_s=%.1f gt1ms=%zu gt5ms=%zu gt50ms=%zu skipped_warm=%zu\n",
           mode, io, done, (double) region / MIB, (long long) (base / (off_t) MIB), gap_ms,
           (double) t[0], (double) at(t, done, 0.50), (double) at(t, done, 0.90),
           (double) at(t, done, 0.99), (double) t[done - 1], (double) sum / (double) done,
           ((double) done * (double) io / MIB) / wall, n1, n5, n50, resid);

    if (mp) munmap(mp, region);
    free(buf);
    free(t); close(fd);
    return 0;
}

// ---------------------------------------------------------------- evict: drop a range from the cache
// A range-level DONTNEED does work where a page-level one does not (the range covers whole folios),
// which is what "was this file cold when I read it" needs.  Verified, and reports what survived.
static int cmd_evict(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "evict FILE REGION_MIB [BASE_MIB]\n"); return 2; }
    const char * path = argv[2];
    size_t region = arg_u(argv[3]) * (size_t) MIB;
    off_t  base   = (off_t) (argc > 4 ? arg_u(argv[4]) : 0) * (off_t) MIB;

    int fd = open_buf(path);
    const off_t fsize = size_of(fd);
    if (base + (off_t) region > fsize) region = (size_t) (fsize - base);
    if (region == 0) { fprintf(stderr, "evict: empty region\n"); return 1; }
    const long pg = sysconf(_SC_PAGESIZE);

    size_t resid_pct = 100;
    char * m = mmap(NULL, region, PROT_READ, MAP_SHARED | MAP_NORESERVE, fd, base);
    if (m == MAP_FAILED) die("mmap");
    for (int attempt = 0; attempt < 3; attempt++) {
        posix_fadvise(fd, attempt == 0 ? base : 0, attempt == 0 ? (off_t) region : 0,
                      POSIX_FADV_DONTNEED);
        const size_t nsamp = 256;
        size_t hit = 0;
        for (size_t i = 0; i < nsamp; i++) {
            off_t o = ((off_t) (rnd64() % (region / (size_t) pg))) * pg;
            unsigned char v = 0;
            if (mincore(m + o, (size_t) pg, &v) == 0 && (v & 1)) hit++;
        }
        resid_pct = 100 * hit / nsamp;
        if (resid_pct <= 5) break;
    }
    printf("PROBE evict region_mib=%.0f base_mib=%lld resident_after_pct=%zu\n",
           (double) region / MIB, (long long) (base / (off_t) MIB), resid_pct);
    munmap(m, region); close(fd);
    return 0;
}

// ---------------------------------------------------------------- pop: mmap fault-in of a region
// MAP_POPULATE over a range that is already cached measures the page-table walk (83 GB/s on this box),
// not the drive, so evict first and report how much of the range was still resident.
static int cmd_pop(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "pop FILE REGION_MIB [BASE_MIB]\n"); return 2; }
    const char * path = argv[2];
    size_t region = arg_u(argv[3]) * (size_t) MIB;
    off_t  base   = (off_t) (argc > 4 ? arg_u(argv[4]) : 0) * (off_t) MIB;

    int fd = open_buf(path);
    const off_t fsize = size_of(fd);
    if (base + (off_t) region > fsize) region = (size_t) (fsize - base);
    if (region == 0) { fprintf(stderr, "pop: empty region\n"); return 1; }

    // evict the range (whole-file if that fails) and verify on a sample of pages
    size_t resid_pct = 0;
    const long pg = sysconf(_SC_PAGESIZE);
    for (int attempt = 0; attempt < 3; attempt++) {
        posix_fadvise(fd, attempt == 0 ? base : 0, attempt == 0 ? (off_t) region : 0,
                      POSIX_FADV_DONTNEED);
        char * probe_map = mmap(NULL, region, PROT_READ, MAP_SHARED | MAP_NORESERVE, fd, base);
        if (probe_map == MAP_FAILED) break;
        const size_t nsamp = 128;
        size_t hit = 0;
        for (size_t i = 0; i < nsamp; i++) {
            off_t o = ((off_t) (rnd64() % (region / (size_t) pg))) * pg;
            unsigned char v = 0;
            if (mincore(probe_map + o, (size_t) pg, &v) == 0 && (v & 1)) hit++;
        }
        munmap(probe_map, region);
        resid_pct = 100 * hit / nsamp;
        if (resid_pct <= 5) break;
    }

    const double t0 = now_s();
    char * m = mmap(NULL, region, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, base);
    const double dt = now_s() - t0;
    if (m == MAP_FAILED) die("mmap(MAP_POPULATE)");

    printf("PROBE pop region_mib=%.0f base_mib=%lld secs=%.3f gbs=%.2f resident_before_pct=%zu\n",
           (double) region / MIB, (long long) (base / (off_t) MIB), dt, (double) region / GIB / dt,
           resid_pct);
    munmap(m, region); close(fd);
    return 0;
}

// ---------------------------------------------------------------- cached: page-cache residency
static int cmd_cached(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "cached FILE\n"); return 2; }
    const char * path = argv[2];
    int fd = open_buf(path);
    off_t fsize = size_of(fd);
    if (fsize <= 0) { printf("PROBE cached pct=0 pages=0 resident=0\n"); return 0; }

    const long pg = sysconf(_SC_PAGESIZE);
    size_t len = (size_t) fsize;
    char * m = mmap(NULL, len, PROT_READ, MAP_SHARED | MAP_NORESERVE, fd, 0);
    if (m == MAP_FAILED) {
        // Fall back to a sampled range rather than failing the whole report.
        len = (size_t) (len > (off_t) (1 << 30) ? (size_t) (1 << 30) : len);
        m = mmap(NULL, len, PROT_READ, MAP_SHARED | MAP_NORESERVE, fd, 0);
        if (m == MAP_FAILED) die("mmap");
    }
    size_t pages = (len + (size_t) pg - 1) / (size_t) pg;
    unsigned char * vec = calloc(pages, 1);
    if (!vec) die("calloc");
    if (mincore(m, len, vec) != 0) die("mincore");
    size_t res = 0;
    for (size_t i = 0; i < pages; i++) if (vec[i] & 1) res++;

    printf("PROBE cached pct=%.1f pages=%zu resident=%zu sampled_mib=%.0f\n",
           100.0 * (double) res / (double) pages, pages, res, (double) len / MIB);
    free(vec); munmap(m, len); close(fd);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s map|bulk|rand|pop|cached ...  (see the header of this file)\n", argv[0]);
        return 2;
    }
    const char * cmd = argv[1];
    if (!strcmp(cmd, "map"))    return cmd_map(argc, argv);
    if (!strcmp(cmd, "bulk"))   return cmd_bulk(argc, argv);
    if (!strcmp(cmd, "rand"))   return cmd_rand(argc, argv);
    if (!strcmp(cmd, "evict"))  return cmd_evict(argc, argv);
    if (!strcmp(cmd, "pop"))    return cmd_pop(argc, argv);
    if (!strcmp(cmd, "cached")) return cmd_cached(argc, argv);
    fprintf(stderr, "storage-probe: unknown command '%s'\n", cmd);
    return 2;
}
