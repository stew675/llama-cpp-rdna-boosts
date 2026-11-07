// preadbench -- cost and DEVICE BYTES of one random 90 B row read, under the three policies
// that matter: buffered (default readahead), buffered + FADV_RANDOM, O_DIRECT sectors.
// usage: preadbench <file> <n> <threads> <mode 0=buffered 1=fadv_random 2=odirect>
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static int g_fd, g_nthr, g_mode, g_n; static int32_t *g_rows; static size_t g_next;
static pthread_mutex_t g_m = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long io_read_bytes(void){ FILE*f=fopen("/proc/self/io","r"); char l[256]; unsigned long long v=0;
    while(f&&fgets(l,sizeof l,f)){ unsigned long long x; if(sscanf(l,"read_bytes: %llu",&x)==1){v=x;break;} } if(f)fclose(f); return v; }
static size_t next_idx(void){ pthread_mutex_lock(&g_m); size_t i=g_next++; pthread_mutex_unlock(&g_m); return i; }
static void *worker(void *a){ (void)a; char *buf; posix_memalign((void**)&buf,4096,8192);
    for(;;){ size_t i=next_idx(); if(i>=(size_t)g_n) break;
        off_t raw = 192 + (off_t)g_rows[i]*90; off_t off; size_t len;
        if (g_mode==2) { off = raw & ~(off_t)511; len = ((raw-off)+90 > 512) ? 1024 : 512; }
        else { off = raw; len = 90; }
        if (pread(g_fd, buf, len, off) != (ssize_t)len) { fprintf(stderr,"short read\n"); _exit(1); } }
    free(buf); return NULL; }
int main(int argc, char **argv) {
    const char *path = argv[1]; g_n = atoi(argv[2]); g_nthr = atoi(argv[3]); g_mode = atoi(argv[4]);
    g_fd = open(path, O_RDONLY | (g_mode==2 ? O_DIRECT : 0));
    if (g_fd < 0) { perror("open"); return 1; }
    if (g_mode==1) posix_fadvise(g_fd, 0, 0, POSIX_FADV_RANDOM);
    g_rows = malloc(sizeof(int32_t)*g_n);
    FILE *rf=fopen("/tmp/ple/rows_ids.bin","rb"); fseek(rf,0,SEEK_END); size_t np=ftell(rf)/4; fseek(rf,0,SEEK_SET);
    int32_t *all=malloc(np*4); if(fread(all,4,np,rf)!=np) return 1; fclose(rf);
    srand(1234); for (int i=0;i<g_n;i++) g_rows[i]=all[(size_t)rand()%np];
    posix_fadvise(g_fd, 192, 28800138240LL, POSIX_FADV_DONTNEED);
    unsigned long long b0 = io_read_bytes(); g_next = 0; double t0 = now();
    pthread_t th[64]; for (int i=0;i<g_nthr;i++) pthread_create(&th[i],NULL,worker,NULL);
    for (int i=0;i<g_nthr;i++) pthread_join(th[i],NULL);
    double dt = now()-t0; unsigned long long b1 = io_read_bytes();
    printf("%-14s %6d reads %2d thr: %6.1f us/read  %6.2f MiB from device total, %6.1f KiB/read\n",
        g_mode==0?"buffered":g_mode==1?"fadv_random":"odirect_512",
        g_n, g_nthr, dt/1e3/g_n, (b1-b0)/1048576.0, (b1-b0)/1024.0/g_n);
    return 0;
}
