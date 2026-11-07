// burst -- cold-read cost vs access pattern: isolated, burst-of-16 every 30 ms, and
// repeat-reads of a cached page.  Distinguishes a device idle/latency effect from a
// per-read overhead.
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static void msleep(int ms){struct timespec t={ms/1000,(long)(ms%1000)*1000000L};nanosleep(&t,NULL);}
int main(int argc,char**argv){
    const char*path=argv[1]; int mode=atoi(argv[2]);
    int fd=open(path,O_RDONLY);
    int32_t*rows=malloc(8u<<20);
    FILE*rf=fopen("/tmp/ple/rows_ids.bin","rb");fseek(rf,0,SEEK_END);size_t np=ftell(rf)/4;fseek(rf,0,SEEK_SET);
    if(fread(rows,4,np,rf)!=np)return 1;fclose(rf);
    char buf[128];
    if(mode==0){ // isolated: 8 cold reads, 30 ms apart
        for(int i=0;i<8;i++){int32_t r=rows[(size_t)(i*7919)%np];off_t off=192+(off_t)r*90;
            posix_fadvise(fd,off&~(off_t)4095,4096,POSIX_FADV_DONTNEED);msleep(30);
            double t0=now();pread(fd,buf,90,off);printf("  isolated  %6.1f us\n",(now()-t0)/1e3);}}
    if(mode==1){ // burst of 16 every 30 ms, each burst fully cold
        for(int b=0;b<8;b++){double t0=now();size_t base=(size_t)b*16;
            for(int i=0;i<16;i++){int32_t r=rows[(base*7919+i*104729)%np];off_t off=192+(off_t)r*90;
                posix_fadvise(fd,off&~(off_t)4095,4096,POSIX_FADV_DONTNEED);}
            msleep(30);
            double t1=now();
            for(int i=0;i<16;i++){int32_t r=rows[(base*7919+i*104729)%np];off_t off=192+(off_t)r*90;pread(fd,buf,90,off);}
            printf("  burst16   %6.2f ms total, %6.1f us/read\n",(now()-t1)/1e6,(now()-t1)/1e3/16);(void)t0;}}
    if(mode==2){ // warm repeat with the same 30 ms gap
        int32_t r=rows[12345];off_t off=192+(off_t)r*90;pread(fd,buf,90,off);
        for(int i=0;i<8;i++){msleep(30);double t0=now();pread(fd,buf,90,off);
            printf("  warm+gap  %6.1f us\n",(now()-t0)/1e3);}}
    return 0;
}
