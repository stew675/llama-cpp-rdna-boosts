# reference material

Extracted for offline study; all of it is **WIP reference, not delivery**.

| file | provenance | what |
|---|---|---|
| `pwilkin-strix-halo-964c6f2f0-tiled-gdn.patch` | `pwilkin/llama.cpp` `strix-halo`, commit `964c6f2f0b66008243883298f3f535bc99f5998c`, parent `0540c6948`, 2026-09-12, Piotr Wilkin | the tiled GDN commit under study (+211/−1 in `gated_delta_net.cu`) |
| `pwilkin-halobox-8ab5a8373-dpp-tiled-kda.patch` | `pwilkin/llama.cpp` `strix-halo-for-halobox`, commit `8ab5a837394fa51c55dfb217afe5551840f1242e`, parent `90ad6cd26`, 2026-09-03, **Gaetan Puleo** | the mature DPP + multi-column tiled commit (+230/−6); lineage also carries a KDA tiled kernel, the sequential `num_warps=32` retune, and four tunable tiled shapes |
| `port-spike-gated_delta_net.patch` | this session, against `~/llama.cpp` `rdna-boosts` | our minimal gfx1201 prototype: DPP helpers + `gated_delta_net_tiled_cuda` + `GGML_CUDA_GDN_TILED` dispatch (16×4 / 8×8) |
- The journey page's relevant sentence: *"Earlier work measured a delta-net rewrite at roughly 3 %
  end-to-end and dropped it over a 0.21 % perplexity cost — a reasonable call, except that it
  compared a chunked rewrite against this tiled kernel, both already fast."*  Step 11's own text
  records the tiled reduction as *"the same pairing order, so the sum is bit-identical"*, and
  attributes the 0.21 % PPL cost to the chunked rewrite.

## Notes

- `964c6f2f0` and `8ab5a8373` are two independent evolutions of the same idea (both come out of the
  Strix Halo work).  `964c6f2f0` is the one on the branch the TODO names; `8ab5a8373` is further
  along (more shapes, KDA, DPP also on the sequential kernel) but lives on a different branch and
  is attributed to a different author.
- The journey page's relevant sentence: *"Earlier work measured a delta-net rewrite at roughly 3 %
  end-to-end and dropped it over a 0.21 % perplexity cost — a reasonable call, except that it
  compared a chunked rewrite against this tiled kernel, both already fast."*  Step 11's own text
  says the tiled reduction *"has the same pairing order, so the sum is bit-identical."*
- Do not apply either pwilkin patch to a llama.cpp checkout as a shortcut; apply only
  `port-spike-gated_delta_net.patch` to the scratch fork if reproducing this session's numbers.
