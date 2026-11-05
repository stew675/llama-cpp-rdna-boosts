#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    size_t gib = argc > 1 ? (size_t) atoi(argv[1]) : 31;
    size_t n = gib << 30;
    char * p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    memset(p, 1, n);
    if (mlock(p, n) != 0) { perror("mlock"); return 1; }
    printf("ballast %zu GiB locked\n", gib);
    fflush(stdout);
    pause();
    return 0;
}
