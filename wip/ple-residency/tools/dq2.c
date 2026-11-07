// dq2 -- first-touch vs repeat cost of a PLE row read through the mapping, with the page cache
// in a known state.  Answers: when a row's page is already in the page cache, does the *first*
// access in a process still cost the disk?
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/mman.h>
#include "ggml.h"
#include "ggml-quants.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bench(uint8_t *map, int64_t *pages, int n, int sweep, const char *tag) {
    float dst[160]; double t0=now();
    for (int s=0;s<sweep;s++) for (int i=0;i<n;i++) dequantize_row_iq4_nl((const block_iq4_nl *)(map + pages[i]), dst, 160);
    double dt=(now()-t0)/1e3/(double)(n*sweep);
    printf("  %-28s %7.1f us/row\n", tag, dt);
    return dt;
}
int main(int argc, char **argv) {
    const char *path = argv[1];
    int n = argc > 2 ? atoi(argv[2]) : 7500;
    int mlock_gib = argc > 3 ? atoi(argv[3]) : 0;
    int fd = open(path, O_RDONLY);
    const size_t total = 28800138240LL;
    uint8_t *map = mmap(NULL, total, PROT_READ, MAP_PRIVATE, fd, 0);
    madvise(map, total, MADV_RANDOM);
    int32_t *rows = malloc(8u<<20);
    FILE *rf = fopen("/tmp/ple/rows_ids.bin","rb"); fseek(rf,0,SEEK_END); size_t np=ftell(rf)/4; fseek(rf,0,SEEK_SET);
    if (fread(rows,4,np,rf)!=np) return 1; fclose(rf);
    // spread the sample over the whole table so the pages are distinct
    int64_t *pages = malloc(sizeof(int64_t)*n); int pagesz = getpagesize();
    for (int i=0;i<n;i++) { int64_t off = 192 + (int64_t) rows[(size_t)i*np/n]*90; pages[i] = off & ~(int64_t)(pagesz-1); }
    if (mlock_gib) {
        size_t sz = (size_t)mlock_gib<<30; char *p = malloc(sz);
        for (size_t i=0;i<sz;i+=4096) p[i] = 1;              // resident, unevictable-ish
        if (mlock(p, sz)) perror("mlock (ignored)");
        printf("pinned %d GiB of anonymous memory first\n", mlock_gib);
    }
    // make the sample's pages resident in the page cache via pread (fincore-verifiable)
    char *buf = malloc(1<<20);
    double t0 = now();
    for (int i=0;i<n;i++) { if (pread(fd, buf, 1024, pages[i]) != 1024) return 1; }
    printf("warmed %d distinct pages into the page cache: %.1f us/page (%.1f MiB)\n",
           n, (now()-t0)/1e3/n, n*1024.0/1048576.0);
    printf("[%d distinct pages, page cache warm]\n", n);
    bench(map, pages, n, 1, "first touch (mapping)");
    bench(map, pages, n, 3, "repeat (shared mapping)");
    // and the same rows through pread (what the managed reader does)
    t0 = now(); for (int i=0;i<n;i++) if (pread(fd, buf, 1024, pages[i]) != 1024) return 1;
    printf("  %-28s %7.1f us/row\n", "pread (page cache hit)", (now()-t0)/1e3/n);
    // now evict and repeat
    posix_fadvise(fd, 192, total, POSIX_FADV_DONTNEED);
    printf("[same pages, evicted]\n");
    bench(map, pages, n, 1, "first touch (cold)");
    bench(map, pages, n, 3, "repeat after faulting");
    return 0;
}
