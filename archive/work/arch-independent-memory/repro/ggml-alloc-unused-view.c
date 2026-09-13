// Minimal repro for the ggml-alloc view-accounting leak documented in ../README.md section 3b.
//
// Mechanism: the counting pass increments view_src->n_views for every view *node* in the graph, but
// the free pass only decrements it when the view node itself is released - and a view node with no
// consumers is never released. A ggml_cpy expanded into the graph purely for its side effect (the
// "copy into a view of preallocated memory" idiom) is exactly such a node, so its view source's
// count stays inflated forever, which blocks both the release and the in-place reuse of the view
// source. Each layer here builds a 4 MiB view source, fills it through dangling copies, then consumes
// it; the leaked counts keep all 12 of them alive.
//
// Build (from a llama.cpp checkout, after scripts/apply-all.sh or on the L1 tree):
//   gcc -O1 -o /tmp/ggml-alloc-unused-view repro/ggml-alloc-unused-view.c \
//       -I ggml/include -L build-rocm/bin -lggml -lggml-base -lggml-cpu \
//       -Wl,-rpath,$PWD/build-rocm/bin
// Run:   /tmp/ggml-alloc-unused-view 1     # the idiom  -> 56.00 MiB without the fix, 16.00 MiB with
//        /tmp/ggml-alloc-unused-view 0     # control    -> 16.00 MiB either way
// With GGML_ALLOCATOR_DEBUG + an instrumented ggml-alloc.c the free lines show it directly:
// 1 free (unfixed) vs 13 (fixed), and "view_src ...: 4 views" never returning to 0.
// "big" (4 MiB), fills it by copying chunks into VIEWS of it (cpy results dangling = the idiom), then
// consumes it immediately.  With the leaked view count the bigs cannot be freed/reused, so the arena
// grows by the sum of the layers.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    const int dangling = argc > 1 ? atoi(argv[1]) : 1;
    const int NLAYER = 12, NCH = 4, W = 1024, H = 1024;   // big = 4 MiB
    struct ggml_init_params ip = { 64*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, H);
    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, H);
    ggml_set_input(x); ggml_set_input(src);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    struct ggml_tensor * out = x;
    for (int l = 0; l < NLAYER; l++) {
        struct ggml_tensor * big = ggml_add(ctx, x, src);
        for (int c = 0; c < NCH; c++) {
            struct ggml_tensor * sv = ggml_view_2d(ctx, src, W/NCH, H, src->nb[1], (size_t) c*(W/NCH)*sizeof(float));
            struct ggml_tensor * v  = ggml_view_2d(ctx, big, W/NCH, H, big->nb[1], (size_t) c*(W/NCH)*sizeof(float));
            struct ggml_tensor * cp = ggml_cpy(ctx, sv, v);
            if (dangling) { ggml_build_forward_expand(gf, cp); }
            else          { out = ggml_add(ctx, out, cp); }
        }
        out = ggml_add(ctx, out, big);
        ggml_build_forward_expand(gf, out);      // expand per layer -> big dies right here
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_reserve(galloc, gf)) { printf("reserve FAILED\n"); return 1; }
    ggml_gallocr_alloc_graph(galloc, gf);
    printf("RESULT dangling=%d n_nodes=%d arena=%.2f MiB  (12 x 4 MiB layers; ideal ~16 MiB)\n",
           dangling, ggml_graph_n_nodes(gf), ggml_gallocr_get_buffer_size(galloc, 0)/1048576.0);
    ggml_gallocr_free(galloc); ggml_free(ctx);
    return 0;
}
