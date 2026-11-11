// 14-expert-cost.cpp -- the per-fill cost at the REAL granularity the expert store evicts and fills at, plus
// the per-miss round-trip floor a decode step pays.  (P1.0 assumption #4, and the input to the ADMISSION
// threshold: how expensive is a cold expert read, i.e. how many touches must an expert earn before it deserves
// a resident slot?)
//
// WHY: probes 04/07/08 measure whole-file and whole-buffer rates.  An expert store fills ~0.5-0.9 MiB slabs at
// scattered file offsets, several per layer per step, and the step cannot start until the fill lands.  So what
// matters is: cold per-slab LATENCY (the admission cost), warm per-slab rate, the layer-span rate (the prefill
// shape), and the wall clock of `pread -> launch -> sync`.
//
// ALIGNMENT (measured here, 2026-11-11): a GGUF tensor offset is aligned to `general.alignment` (32 bytes),
// NOT to 4096, so an O_DIRECT pread at a real expert offset is refused unless the read is rounded out to the
// block size -- the expert's bytes then land at `slot + (off - align_down(off, 4096))`.  This probe does that
// and reports both arms.
//
// Geometry comes from gguf-expert-geom.py (an expert's rows are one contiguous run in the file):
//   ./gguf-expert-geom.py MODEL.gguf --layer N --shell   ->  OFFSET SLAB_BYTES N_EXPERT
//
// Build: hipcc -O2 -o 14-expert-cost 14-expert-cost.cpp
// Run:   ./14-expert-cost <file> <offset> <slab_bytes> [count=16]
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <ctime>
#include <algorithm>
#include <vector>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static const size_t ALIGN=4096;
static off_t  adn(off_t o){ return (o/(off_t)ALIGN)*(off_t)ALIGN; }
static size_t aup(size_t n){ return ((n+ALIGN-1)/ALIGN)*ALIGN; }
__global__ void touch(const unsigned char* p, unsigned long long* out){
  if(threadIdx.x==0) *out += p[0] + p[1024];
}
static void stats(const char* tag, std::vector<double>& ms){
  if(ms.empty()){ printf("  %-30s (no successful read)\n", tag); return; }
  std::sort(ms.begin(), ms.end());
  const double sum=[&]{ double s=0; for(double v:ms) s+=v; return s; }();
  printf("  %-30s n=%2zu  min %7.3f  med %7.3f  mean %7.3f  max %7.3f ms\n",
         tag, ms.size(), ms.front(), ms[ms.size()/2], sum/ms.size(), ms.back());
}

int main(int argc,char**argv){
  if(argc<4){ printf("usage: %s <file> <offset> <slab_bytes> [count=16]\n",argv[0]); return 2; }
  const char* path=argv[1];
  const off_t off=(off_t)atoll(argv[2]);
  const size_t slab=(size_t)atoll(argv[3]);
  const int count=argc>4?atoi(argv[4]):16;
  const off_t  base=adn(off), delta=(off-base);
  const size_t need=aup(slab+(size_t)delta);
  const size_t span=aup(slab*(size_t)count);
  printf("14-expert-cost: %s\n  at %lld (align_down %lld, +%lld), slab %.3f MiB x %d (span %.1f MiB)\n",
         path,(long long)off,(long long)base,(long long)delta,slab/1048576.0,count,span/1048576.0);

  int fd=open(path,O_RDONLY|O_DIRECT); if(fd<0){ perror("open O_DIRECT"); return 2; }
  int fdb=open(path,O_RDONLY);         if(fdb<0){ perror("open"); return 2; }
  const size_t BUF=std::max(need,span)+ALIGN;
  unsigned char* buf=(unsigned char*)mmap(nullptr,BUF,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  madvise(buf,BUF,MADV_HUGEPAGE); memset(buf,0xAA,BUF);

  // ---- (a) COLD single-slab latency, one distinct slab per rep (the admission cost)
  std::vector<double> od_cold, bf_cold, od_warm, bf_warm;
  const int R=std::min(count,8);
  for(int i=0;i<R;i++){
    const off_t o=off+(off_t)i*(off_t)slab;
    posix_fadvise(fdb,adn(o),(off_t)aup(slab+ALIGN),POSIX_FADV_DONTNEED);
    double t=now(); ssize_t g=pread(fd,buf,need,adn(o)); double dt=now()-t;
    if(g<0) printf("  O_DIRECT pread: %s (offset %lld, %zu bytes)\n", strerror(errno),(long long)adn(o),need);
    else od_cold.push_back(dt*1e3);
  }
  for(int i=0;i<R;i++){
    const off_t o=off+(off_t)i*(off_t)slab;
    posix_fadvise(fdb,o,(off_t)slab,POSIX_FADV_DONTNEED);
    double t=now(); ssize_t g=pread(fdb,buf+delta,slab,o); double dt=now()-t;
    if(g<0) printf("  buffered pread: %s\n", strerror(errno)); else bf_cold.push_back(dt*1e3);
  }
  // ---- warm (already in the page cache)
  for(int i=0;i<R;i++){
    const off_t o=off+(off_t)i*(off_t)slab;
    double t=now(); (void)pread(fd,buf,need,adn(o)); od_warm.push_back((now()-t)*1e3);
    t=now(); (void)pread(fdb,buf+delta,slab,o);   bf_warm.push_back((now()-t)*1e3);
  }
  stats("COLD  one slab  O_DIRECT", od_cold);
  stats("COLD  one slab  buffered", bf_cold);
  stats("warm  one slab  O_DIRECT", od_warm);
  stats("warm  one slab  buffered", bf_warm);

  // ---- (c) the layer span: the prefill shape
  posix_fadvise(fdb,base,(off_t)span,POSIX_FADV_DONTNEED);
  double t=now(); ssize_t g=pread(fd,buf,span,base); double dt=now()-t;
  printf("  cold  %2d-slab span  O_DIRECT : %8.3f ms = %5.2f GB/s%s\n", count, dt*1e3,
         g>0?g/dt/1e9:0, g<0?" (refused)":"");
  t=now(); g=pread(fd,buf,span,base); dt=now()-t;
  printf("  warm  %2d-slab span  O_DIRECT : %8.3f ms = %5.2f GB/s%s\n", count, dt*1e3,
         g>0?g/dt/1e9:0, g<0?" (refused)":"");

  // ---- (d) the per-miss round-trip floor: pread -> launch -> sync (a decode step waits for this)
  CK(hipHostRegister(buf,BUF,hipHostRegisterMapped|hipHostRegisterReadOnly));
  void* d=buf; CK(hipHostGetDevicePointer(&d,buf,0));
  unsigned long long* acc; CK(hipMalloc(&acc,8)); CK(hipMemset(acc,0,8));
  std::vector<double> rt; const int RR=8;
  for(int i=0;i<RR;i++){
    const off_t o=off+(off_t)(i%R)*(off_t)slab;
    t=now(); (void)pread(fd,buf,need,adn(o));
    touch<<<1,32>>>((const unsigned char*)((char*)d+delta),acc); CK(hipDeviceSynchronize());
    rt.push_back((now()-t)*1e3);
  }
  stats("fill->launch->sync floor", rt);
  double med = rt.empty()?0:([&]{ std::sort(rt.begin(),rt.end()); return rt[rt.size()/2]; }());
  printf("  projection (warm source, one slab per miss, 10 experts x 48 layers): %.1f ms of fill per step\n",
         med*10*48);
  hipFree(acc); CK(hipHostUnregister(buf)); munmap(buf,BUF); close(fd); close(fdb);
  return 0;
}
