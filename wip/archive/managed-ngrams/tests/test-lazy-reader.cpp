#include "../src/llama-lazy-reader.h"

#include "ggml.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__HAIKU__)
#define TEST_HAS_POSIX 1
#include <fcntl.h>
#include <unistd.h>
#else
#define TEST_HAS_POSIX 0
#endif

// ---------------------------------------------------------------------------
// tiny Q8_0 implementation (block = fp16 scale + 32 x int8), matching
// ggml's quantize_row_q8_0_ref / dequantize_row_q8_0 byte for byte
// ---------------------------------------------------------------------------

struct block_q8_0_t {
    ggml_fp16_t d;
    int8_t      qs[32];
};

static void test_quantize_row_q8_0(const float * x, block_q8_0_t * y, int64_t k) {
    GGML_ASSERT(k % 32 == 0);
    for (int64_t i = 0; i < k / 32; i++) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            amax = std::max(amax, std::fabs(x[i*32 + j]));
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < 32; j++) {
            y[i].qs[j] = (int8_t) std::round(x[i*32 + j] * id);
        }
    }
}

static void test_dequantize_row_q8_0(const block_q8_0_t * x, float * y, int64_t k) {
    GGML_ASSERT(k % 32 == 0);
    for (int64_t i = 0; i < k / 32; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < 32; j++) {
            y[i*32 + j] = x[i].qs[j] * d;
        }
    }
}

static void test_assert(bool cond, const char * msg) {
    if (!cond) {
        throw std::runtime_error(std::string("assertion failed: ") + msg);
    }
}

// row i element d: deterministic function of both, keeps Q8_0 lossy but stable
static float test_row_value(uint64_t row, int64_t d) {
    return 1.0f + 0.5f * std::sin(double(row) * 0.37 + double(d) * 0.11) +
                 0.2f * std::cos(double(row) * 0.13 - double(d) * 0.07);
}

#if TEST_HAS_POSIX

static int test_make_file(uint64_t n_rows, int64_t row_nelems, std::vector<uint8_t> & data) {
    const size_t row_bytes = (size_t) row_nelems * ggml_type_size(GGML_TYPE_Q8_0) / ggml_blck_size(GGML_TYPE_Q8_0);
    data.resize((size_t) n_rows * row_bytes);
    std::vector<float> row(row_nelems);
    std::vector<block_q8_0_t> blocks(row_nelems / 32);
    for (uint64_t i = 0; i < n_rows; i++) {
        for (int64_t d = 0; d < row_nelems; d++) {
            row[d] = test_row_value(i, d);
        }
        test_quantize_row_q8_0(row.data(), blocks.data(), row_nelems);
        std::memcpy(data.data() + (size_t) i * row_bytes, blocks.data(), row_bytes);
    }

    char path[] = "/tmp/test-lazy-reader-XXXXXX";
    const int fd = mkstemp(path);
    test_assert(fd >= 0, "mkstemp failed");
    unlink(path); // removed on close
    size_t off = 0;
    while (off < data.size()) {
        const ssize_t w = write(fd, data.data() + off, data.size() - off);
        test_assert(w > 0, "write failed");
        off += (size_t) w;
    }
    return fd;
}

static void test_check_gather(llama_lazy_reader & reader, const std::vector<uint8_t> & data,
        const std::vector<int32_t> & rows, int64_t row_nelems) {
    const size_t row_bytes = (size_t) row_nelems * ggml_type_size(GGML_TYPE_Q8_0) / ggml_blck_size(GGML_TYPE_Q8_0);
    std::vector<float> got(rows.size() * (size_t) row_nelems);
    std::vector<float> ref(rows.size() * (size_t) row_nelems);
    reader.gather(rows.data(), rows.size(), got.data());
    for (size_t j = 0; j < rows.size(); j++) {
        test_dequantize_row_q8_0((const block_q8_0_t *) (data.data() + (size_t) rows[j] * row_bytes),
                ref.data() + j * row_nelems, row_nelems);
    }
    test_assert(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0, "gather != reference decode");
}

int main() {
    try {
        const uint64_t n_rows = 10000;
        const int64_t  row_nelems = 160; // Q8_0: 170 B/row, 24 rows/page
        const size_t   row_bytes = (size_t) row_nelems * ggml_type_size(GGML_TYPE_Q8_0) / ggml_blck_size(GGML_TYPE_Q8_0);
        const size_t   rows_per_page = 4096 / row_bytes;
        const size_t   n_pages = (size_t) ((n_rows + rows_per_page - 1) / rows_per_page);
        test_assert(row_bytes == 170, "row size");
        test_assert(rows_per_page == 24, "rows per page");
        test_assert(n_pages == 417, "page count");

        std::vector<uint8_t> data;
        int fd = test_make_file(n_rows, row_nelems, data);

        // case 1: correctness -- page interior, page boundaries, final partial
        // page, repeated rows in one call. Budget covers the whole table.
        {
            llama_lazy_reader reader(llama_lazy_reader::config{
                dup(fd), 0, row_bytes, n_rows, GGML_TYPE_Q8_0, row_nelems,
                /*budget*/ (n_pages + 1) * 4096});
            test_assert(reader.n_pages_total() == n_pages, "n_pages_total");

            const std::vector<int32_t> rows = {
                0, 1, 23, 24, 25, 47, 48, /* page interiors */ 5,
                9999,                      // last row (partial page)
                9984,                      // first row of the last page
                23, 24, 5,                 // repeats
            };
            test_check_gather(reader, data, rows, row_nelems);

            // one pass over the whole table, then a second that must hit
            std::vector<int32_t> all(n_rows);
            for (uint64_t i = 0; i < n_rows; i++) {
                all[i] = (int32_t) i;
            }
            test_check_gather(reader, data, all, row_nelems);
            test_check_gather(reader, data, all, row_nelems);

            test_assert(reader.misses() == n_pages, "one miss per page on cold table");
            // gather 1 faulted pages {0,1,2,416} (4 hits on the first sweep), the
            // second sweep hit all 417
            test_assert(reader.hits() == n_pages + 4, "all hits on warm table");
            test_assert(reader.bytes_read() == (size_t) n_rows * row_bytes,
                    "bytes read == table size (partial page clamped)");
        }

        // case 2: eviction -- budget below the working set; sweep a sliding
        // window repeatedly and verify every row still decodes correctly
        {
            llama_lazy_reader reader(llama_lazy_reader::config{
                dup(fd), 0, row_bytes, n_rows, GGML_TYPE_Q8_0, row_nelems,
                /*budget*/ 8 * 4096});
            std::vector<int32_t> win(rows_per_page * 4); // 4 pages per sweep
            for (uint64_t page0 = 0; page0 + 4 <= n_pages; page0 += 3) {
                for (size_t k = 0; k < win.size(); k++) {
                    win[k] = (int32_t) ((page0 + k / rows_per_page) * rows_per_page + k % rows_per_page);
                }
                test_check_gather(reader, data, win, row_nelems);
            }
            // verify a spread of rows again after the sweeps (re-faults)
            std::vector<int32_t> spread;
            for (uint64_t i = 0; i < n_rows; i += 97) {
                spread.push_back((int32_t) i);
            }
            spread.push_back((int32_t) (n_rows - 1));
            test_check_gather(reader, data, spread, row_nelems);
            test_assert(reader.misses() > 0 && reader.hits() > 0, "eviction exercised both paths");
        }

        // case 3: budget below one page clamps to a single slot
        {
            llama_lazy_reader reader(llama_lazy_reader::config{
                dup(fd), 0, row_bytes, n_rows, GGML_TYPE_Q8_0, row_nelems,
                /*budget*/ 1000});
            test_assert(reader.budget() == 4096, "budget clamps to one page");
            test_check_gather(reader, data, {0, 23, 24, 9999}, row_nelems);
            test_check_gather(reader, data, {0, 23, 24, 9999}, row_nelems); // thrash
            test_check_gather(reader, data, {5, 9984, 1}, row_nelems);
            test_assert(reader.misses() > 0, "single slot must miss");
        }

        // case 4: row_bytes > page_size bumps the page size
        {
            llama_lazy_reader reader(llama_lazy_reader::config{
                dup(fd), 0, row_bytes, n_rows, GGML_TYPE_Q8_0, row_nelems,
                /*budget*/ (n_pages + 1) * 4096,
                /*page_size*/ 64});
            test_assert(reader.n_pages_total() == n_pages, "page size bump keeps row mapping");
            test_check_gather(reader, data, {0, 23, 24, 9999}, row_nelems);
        }

        // case 5: threaded -- results identical to serial under contention
        {
            llama_lazy_reader reader(llama_lazy_reader::config{
                dup(fd), 0, row_bytes, n_rows, GGML_TYPE_Q8_0, row_nelems,
                /*budget*/ 16 * 4096});
            const int n_threads = 4;
            const int per_thread = 500;
            std::vector<std::vector<int32_t>> row_sets(n_threads);
            for (int t = 0; t < n_threads; t++) {
                row_sets[t].reserve(per_thread);
                for (int k = 0; k < per_thread; k++) {
                    row_sets[t].push_back((int32_t) ((t * 997 + k * 101) % n_rows));
                }
            }
            std::vector<std::vector<float>> got(n_threads);
            std::vector<std::thread> threads;
            for (int t = 0; t < n_threads; t++) {
                got[t].resize(per_thread * (size_t) row_nelems);
                threads.emplace_back([&reader, &row_sets, &got, t]() {
                    reader.gather(row_sets[t].data(), row_sets[t].size(), got[t].data());
                });
            }
            for (auto & th : threads) {
                th.join();
            }
            for (int t = 0; t < n_threads; t++) {
                // serial reference on the same (warm) reader; must match
                std::vector<float> ref(per_thread * (size_t) row_nelems);
                reader.gather(row_sets[t].data(), row_sets[t].size(), ref.data());
                test_assert(std::memcmp(got[t].data(), ref.data(), ref.size() * sizeof(float)) == 0,
                        "threaded gather != serial gather");
            }
        }

        // case 5b: F32 rows (no to_float trait; the reader must copy them as-is)
        {
            const uint64_t n_f32_rows = 1000;
            const int64_t  n_f32_elems = 64;
            const size_t   f32_row_bytes = (size_t) n_f32_elems * sizeof(float); // 256
            std::vector<uint8_t> f32_data((size_t) n_f32_rows * f32_row_bytes);
            for (uint64_t i = 0; i < n_f32_rows; i++) {
                for (int64_t d = 0; d < n_f32_elems; d++) {
                    ((float *) (f32_data.data() + (size_t) i * f32_row_bytes))[d] = test_row_value(i, d);
                }
            }
            char path[] = "/tmp/test-lazy-reader-f32-XXXXXX";
            int f32_fd = mkstemp(path);
            unlink(path);
            test_assert(f32_fd >= 0, "mkstemp f32 failed");
            size_t off = 0;
            while (off < f32_data.size()) {
                const ssize_t w = write(f32_fd, f32_data.data() + off, f32_data.size() - off);
                test_assert(w > 0, "write f32 failed");
                off += (size_t) w;
            }
            llama_lazy_reader reader(llama_lazy_reader::config{
                dup(f32_fd), 0, f32_row_bytes, n_f32_rows, GGML_TYPE_F32, n_f32_elems,
                /*budget*/ 4 * 4096}); // 16 rows/page, 256 pages total, 4 slots
            std::vector<int32_t> rows = {0, 15, 16, 31, 32, 999, 500, 15, 999};
            std::vector<float> got(rows.size() * (size_t) n_f32_elems);
            reader.gather(rows.data(), rows.size(), got.data());
            for (size_t j = 0; j < rows.size(); j++) {
                const float * ref = (const float *) (f32_data.data() + (size_t) rows[j] * f32_row_bytes);
                test_assert(std::memcmp(got.data() + j * n_f32_elems, ref, f32_row_bytes) == 0,
                        "f32 gather != reference");
            }
            close(f32_fd);
        }

        // case 6: fd is owned -- closed on destruction
        {
            int dup_fd = dup(fd);
            test_assert(dup_fd >= 0, "dup failed");
            {
                llama_lazy_reader reader(llama_lazy_reader::config{
                    dup_fd, 0, row_bytes, n_rows, GGML_TYPE_Q8_0, row_nelems, 4096});
            }
            test_assert(fcntl(dup_fd, F_GETFD) == -1, "fd closed by destructor");
        }

        close(fd);
        printf("test-lazy-reader: all tests passed\n");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "test-lazy-reader: %s\n", e.what());
        return 1;
    }
}

#else // !TEST_HAS_POSIX

int main() {
    printf("test-lazy-reader: skipped (POSIX only)\n");
    return 0;
}

#endif
