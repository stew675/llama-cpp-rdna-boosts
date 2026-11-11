#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ printf("HIPFAIL %s: %s\n", #x, hipGetErrorString(e)); exit(2);} }while(0)

__global__ void sum_bytes(const unsigned char* p, unsigned long long* out, size_t n){
  size_t stride = (size_t)gridDim.x*blockDim.x;
  unsigned long long acc=0;
  for(size_t k=(size_t)blockIdx.x*blockDim.x+threadIdx.x; k<n; k+=stride)
    acc += ((volatile const unsigned char*)p)[k];
  atomicAdd(out, acc);
}

static size_t resident(void* p, size_t n){
  long ps = sysconf(_SC_PAGESIZE);
  size_t np = (n+ps-1)/ps, c=0;
  std::vector<unsigned char> v(np);
  if(mincore(p, n, v.data())!=0){ perror("mincore"); return (size_t)-1; }
  for(size_t i=0;i<np;i++) if(v[i]&1) c++;
  return c;
}
static void lck(const char* tag){
  FILE* f=fopen("/proc/self/status","r"); char line[256]; 
  while(fgets(line,sizeof line,f)) if(!strncmp(line,"VmLck",5)) printf("[%s] %s", tag, line);
  fclose(f);
}

int main(int argc, char** argv){
  std::string mode = argc>1 ? argv[1] : "hot";
  const char* path = "/tmp/xnack-probe/blob.bin";
  size_t SZ = 64u<<20;
  int fd = open(path, O_RDONLY);
  if(fd<0){ perror("open"); return 1; }

  bool is_file = mode.rfind("anon",0)!=0;
  unsigned char* p;
  if(is_file){ p = (unsigned char*)mmap(nullptr, SZ, PROT_READ, MAP_SHARED, fd, 0); }
  else { p = (unsigned char*)mmap(nullptr, SZ, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
         for(size_t i=0;i<SZ;i+=4096) p[i] = (unsigned char)(i/4096); }
  if(p==MAP_FAILED){ perror("mmap"); return 1; }

  bool reg = mode.find("reg")!=std::string::npos && mode.find("unreg")==std::string::npos;
  void* dev = p;
  if(reg){
    CK(hipHostRegister(p, SZ, hipHostRegisterMapped|hipHostRegisterReadOnly));
    CK(hipHostGetDevicePointer(&dev, p, 0));
    printf("[%s] hipHostRegister OK, dev=%p host=%p, resident pages after register: %zu\n",
           mode.c_str(), dev, p, resident(p, SZ));
  }
  if(mode.rfind("cold",0)==0 || mode.rfind("anon",0)==0){
    if(is_file) posix_fadvise(fd, 0, SZ, POSIX_FADV_DONTNEED);
    madvise(p, SZ, MADV_DONTNEED);
    size_t r = resident(p, SZ);
    printf("[%s] after DONTNEED: resident pages = %zu of %zu\n", mode.c_str(), r, SZ/4096);
  }
  lck(mode.c_str());

  unsigned long long* dsum; CK(hipMalloc(&dsum, sizeof(*dsum))); CK(hipMemset(dsum,0,sizeof(*dsum)));
  size_t blocks = 256, threads = 256;
  printf("[%s] launching kernel over %zu MiB at dev ptr %p ...\n", mode.c_str(), SZ>>20, dev);
  fflush(stdout);
  sum_bytes<<<blocks, threads>>>((const unsigned char*)dev, dsum, SZ);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  unsigned long long h=0; CK(hipMemcpy(&h, dsum, sizeof(h), hipMemcpyDeviceToHost));
  printf("[%s] GPU SUM = %llu (survived)\n", mode.c_str(), h);
  if(mode=="hot"){
    unsigned long long c=0; for(size_t i=0;i<SZ;i++) c += p[i];
    printf("[%s] CPU SUM = %llu -> %s\n", mode.c_str(), c, (c==h)?"MATCH":"MISMATCH");
  }
  return 0;
}
