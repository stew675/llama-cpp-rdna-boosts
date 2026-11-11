#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <unistd.h>
#include <cstdio>
#include <ctime>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
int main(int argc,char**argv){
  size_t SZ=1ull<<30;
  int fd=open(argv[1],O_RDONLY); unsigned char* src=(unsigned char*)mmap(0,SZ,PROT_READ,MAP_PRIVATE,fd,0);
  for(size_t i=0;i<SZ;i+=4096)(void)src[i];   /* page-cache hot */
  unsigned char* d; CK(hipMalloc(&d,SZ));
  double t=now(); CK(hipMemcpy(d,src,SZ,hipMemcpyHostToDevice)); double h2d=now()-t;
  printf("  hipMemcpy H2D (page cache -> GTT arena): %.3f s = %.1f GB/s\n", h2d, SZ/h2d/1e9);
  t=now(); std::memcpy((void*)d,(const void*)src,SZ); double cpu=now()-t;
  printf("  host memcpy     (page cache -> GTT arena): %.3f s = %.1f GB/s\n", cpu, SZ/cpu/1e9);
  t=now(); CK(hipMemcpy(d,src,SZ,hipMemcpyHostToDevice)); printf("  hipMemcpy again (warm): %.3f s = %.1f GB/s\n", now()-t, SZ/(now()-t)/1e9);
  return 0;
}
