// bulkread -- time a bulk load of a known set of PLE pages or rows from the shard-2 file.
// usage: bulkread pages <file> <off.bin>            (int64 offsets of pages, sorted)
//        bulkread rows  <file> <rowids.bin> <512|1024>   (int32 row ids, sorted, O_DIRECT)
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>

static int   g_fd;
static int64_t *g_off;
static int32_t *g_row;
static size_t g_n;
static int g_gran, g_nthreads;
static double g_secs;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

static void *worker_pages(void *arg) {
    long id = (long) arg;
    const size_t bufsz = 1<<20;
    char *buf = malloc(bufsz);
    for (size_t i = id; i < g_n; i += g_nthreads) {
        off_t o = (off_t) g_off[i];
        size_t want = 45*90;
        ssize_t r = pread(g_fd, buf, want, o);
        if (r != (ssize_t) want) { fprintf(stderr, "short read %zd at %lld\n", r, (long long) o); exit(1); }
    }
    free(buf);
    return NULL;
}

static void *worker_rows(void *arg) {
    long id = (long) arg;
    const size_t bufsz = 1<<20;
    void *buf;
    if (posix_memalign(&buf, 4096, bufsz)) { exit(1); }
    for (size_t i = id; i < g_n; i += g_nthreads) {
        off_t raw = 192 + (off_t) g_row[i] * 90;
        off_t o = raw & ~(off_t)(g_gran-1);
        size_t want = g_gran;
        if ((size_t)(raw - o) + 90 > want) want = 2*g_gran;
        ssize_t r = pread(g_fd, buf, want, o);
        if (r != (ssize_t) want) { fprintf(stderr, "short read %zd at %lld\n", r, (long long) o); exit(1); }
    }
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s pages|rows <file> <list.bin> [512|1024]\n", argv[0]); return 1; }
    const char *mode = argv[1], *path = argv[2], *list = argv[3];
    int odirect = 0;
    if (!strcmp(mode, "rows")) { g_gran = argc > 4 ? atoi(argv[4]) : 512; odirect = 1; }
    g_nthreads = 16;
    FILE *f = fopen(list, "rb"); fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    g_n = bytes / (mode[0]=='p' ? 8 : 4);
    void *data = malloc(bytes); if (fread(data, 1, bytes, f) != (size_t) bytes) return 1; fclose(f);
    g_off = data; g_row = data;
    g_fd = open(path, O_RDONLY | (odirect ? O_DIRECT : 0));
    if (g_fd < 0) { perror("open"); return 1; }

    double t0 = now();
    pthread_t th[64];
    for (int i = 0; i < g_nthreads; i++)
        pthread_create(&th[i], NULL, mode[0]=='p' ? worker_pages : worker_rows, (void*)(long)i);
    for (int i = 0; i < g_nthreads; i++) pthread_join(th[i], NULL);
    g_secs = now() - t0;
    double mb = mode[0]=='p' ? (double) g_n*45*90/1048576.0 : (double) g_n*(double)g_gran/1048576.0;
    printf("%s: %zu units, %.1f MiB in %.2f s = %.0f MiB/s (%d threads)\n", mode, g_n, mb, g_secs, mb/g_secs, g_nthreads);
    return 0;
}
