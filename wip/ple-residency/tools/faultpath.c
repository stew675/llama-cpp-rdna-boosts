// faultpath -- how does a 90 B row read from a cold 4 KiB page behave under the loader's
// mmap flags?  16 rows/step is what qwen4exp's PLE gather does per decode token.
//   A: memcpy straight from the MADV_RANDOM mapping (what the host-gather path does today)
//   B: MADV_WILLNEED then memcpy                      (same, the mapping hint the code issues)
//   C: POSIX_FADV_WILLNEED on the fd then memcpy      (readahead that ignores the VMA flags)
//   D: parallel pread into scratch (what the managed reader does, 4 threads)
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int g_fd; static const char *g_path; static uint8_t *g_map; static int64_t g_off[512];
static void evict_all(void) { posix_fadvise(g_fd, 192, 28800138240LL, POSIX_FADV_DONTNEED); }
static void *pread_worker(void *a){ int i=(int)(long)a; static __thread char *buf; if(!buf) buf=malloc(4096);
    if (pread(g_fd, buf, 1024, g_off[i]) != 1024) { fprintf(stderr,"short\n"); } return NULL; }

int main(int argc, char **argv) {
    g_path = argv[1];
    const int G = 32, R = 16;              // groups of 16 rows (one decode step each)
    g_fd = open(g_path, O_RDONLY);
    const long long total = 28800138240LL;
    g_map = mmap(NULL, total + 4096, PROT_READ, MAP_PRIVATE, g_fd, 0);
    if (g_map == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(g_map, total, MADV_RANDOM);    // what llama_mmap sets on the lazy ranges

    int64_t *pages = malloc(sizeof(int64_t)*7u*1024*1024);
    FILE *pf = fopen("/tmp/ple/pages_off.bin","rb"); fseek(pf,0,SEEK_END); long nb=ftell(pf); fseek(pf,0,SEEK_SET);
    size_t np = nb/8; if (fread(pages,8,np,pf)!=np) return 1; fclose(pf);
    // reverse the page list so we take pages from the far, cold end and never reuse one
    for (size_t i=0;i<np/2;i++){ int64_t t=pages[i]; pages[i]=pages[np-1-i]; pages[np-1-i]=t; }
    size_t cur = 0;
    const char *names[4] = {"A memcpy only      ","B MADV_WILLNEED    ","C FADV_WILLNEED    ","D parallel pread   "};
    for (int mode = 0; mode < 4; mode++) {
        evict_all();
        double t0 = now();
        for (int g = 0; g < G; g++) {
            for (int i = 0; i < R; i++) g_off[i] = pages[cur++];
            if (mode == 1) for (int i=0;i<R;i++) madvise(g_map + g_off[i], 1024, MADV_WILLNEED);
            if (mode == 2) for (int i=0;i<R;i++) posix_fadvise(g_fd, g_off[i]-192, 1024, POSIX_FADV_WILLNEED);
            if (mode == 3) {
                pthread_t th[4];
                for (int t=0;t<4;t++) pthread_create(&th[t],NULL,pread_worker,(void*)(long)t);
                for (int t=0;t<4;t++) pthread_join(th[t],NULL);
                for (int t=4;t<R;t++){ pthread_create(&th[0],NULL,pread_worker,(void*)(long)t); pthread_join(th[0],NULL); }
            }
            volatile unsigned long sum = 0;
            for (int i = 0; i < R; i++) sum += g_map[g_off[i]-192];   // the 90 B row's first byte
            (void) sum;
        }
        double dt = now() - t0;
        printf("%s %d steps of %d rows: %.2f ms/step (%.1f us/row)\n", names[mode], G, R, 1e3*dt/G, 1e6*dt/(G*R));
    }
    return 0;
}
