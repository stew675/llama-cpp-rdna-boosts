// lonepread -- is a cold 90 B read's cost a function of how long the device has been idle?
// usage: lonepread <file> <n> <gap_ms>
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static void gap_ms(int ms){ if(ms<=0) return; struct timespec t={ms/1000,(long)(ms%1000)*1000000L}; nanosleep(&t,NULL); }
int main(int argc, char **argv) {
    const char *path=argv[1]; int n=atoi(argv[2]); int gap=atoi(argv[3]);
    int fd=open(path,O_RDONLY);
    int32_t *rows=malloc(8u<<20);
    FILE *rf=fopen("/tmp/ple/rows_ids.bin","rb"); fseek(rf,0,SEEK_END); size_t np=ftell(rf)/4; fseek(rf,0,SEEK_SET);
    if(fread(rows,4,np,rf)!=np) return 1; fclose(rf);
    char buf[128]; double tot=0;
    for (int i=0;i<n;i++) {
        int32_t r = rows[(size_t)(i*7919)%np];
        off_t off = 192 + (off_t)r*90;
        off_t pg = off & ~(off_t)4095;
        posix_fadvise(fd, pg, 4096, POSIX_FADV_DONTNEED);          // make it cold
        gap_ms(gap);                                               // let the device idle
        double t0=now(); if (pread(fd,buf,90,off)!=90) return 1; double dt=now()-t0;
        tot += dt;
        printf("  read %2d after %4d ms idle: %7.1f us\n", i, gap, dt/1e3);
    }
    printf("mean %.1f us over %d reads, gap %d ms\n", tot/1e3/n, n, gap);
    return 0;
}
