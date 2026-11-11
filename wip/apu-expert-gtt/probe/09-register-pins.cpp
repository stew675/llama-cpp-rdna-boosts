#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static void show(unsigned char* p,size_t SZ,const char* tag){
  long ps=sysconf(_SC_PAGESIZE); size_t np=SZ/ps,c=0; std::vector<unsigned char> v(np);
  if(mincore(p,SZ,v.data())==0) for(size_t i=0;i<np;i++) if(v[i]&1) c++;
  printf("    %-42s %8zu / %zu pages resident (%zu MiB)\n",tag,c,np,c*ps>>20);
}
int main(int argc,char**argv){
  std::string mode=argv[1]; const char* path=argv[2]; size_t SZ=strtoull(argv[3],0,0);
  int fd=open(path,O_RDONLY); unsigned char* p=(unsigned char*)mmap(0,SZ,PROT_READ,MAP_SHARED,fd,0);
  for(size_t i=0;i<SZ;i+=4096) (void)p[i];
  show(p,SZ,"after touch-all (page cache hot)");
  auto drop=[&]{ posix_fadvise(fd,0,SZ,POSIX_FADV_DONTNEED); madvise(p,SZ,MADV_DONTNEED); usleep(100000); };
  if(mode=="ctl"){ drop(); show(p,SZ,"CONTROL: after DONTNEED, never registered"); return 0; }
  if(mode=="reg"){
    CK(hipHostRegister(p,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly));
    printf("    registered\n");
    drop(); show(p,SZ,"registered: after DONTNEED");
    CK(hipHostUnregister(p)); drop(); show(p,SZ,"unregistered (VMA kept): after DONTNEED");
    return 0;
  }
  if(mode=="cost"){
    for(int i=0;i<4;i++){
      double t=now(); CK(hipHostRegister(p,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly));
      double r=now()-t; t=now(); CK(hipHostUnregister(p)); double u=now()-t;
      printf("    hot register %6.1f ms (%.1f GB/s)   unregister %6.1f ms\n", r*1e3, SZ/r/1e9, u*1e3);
    }
    return 0;
  }
  return 0;
}
