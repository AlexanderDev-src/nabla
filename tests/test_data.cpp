// M6: loading MNIST. Grouped by step, in the order of the M6 plan.
#include "harness.hpp"
#include "nabla/nabla.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;
using Dims = std::vector<std::size_t>;

// A stream holding exactly these bytes, so a test can build a tiny IDX
// file in memory.
std::istringstream stream_of(const Bytes &b) {
    return std::istringstream(std::string(b.begin(), b.end()));
}

// true if f throws std::runtime_error; any other exception fails the test
template <class F> bool throws(F f) {
    try {
        f();
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// a labels file: magic 00 00 08 01, one dimension of 3, then 3 labels
const Bytes three_labels = {0, 0, 0x08, 1, 0, 0, 0, 3, 5, 0, 4};

// true if v holds exactly these values, each within 1e-5. Written so that
// NaN fails.
bool values_are(const std::vector<float> &v,
                std::initializer_list<float> want) {
    if (v.size() != want.size()) {
        return false;
    }
    std::size_t k = 0;
    for (const float w : want) {
        if (!(std::fabs(v[k++] - w) < 1e-5f)) {
            return false;
        }
    }
    return true;
}

using Indices = std::vector<std::size_t>;
using Batches = std::vector<Indices>;

Indices sizes_of(const Batches &batches) {
    Indices sizes;
    for (const auto &b : batches) {
        sizes.push_back(b.size());
    }
    return sizes;
}

// every index from every batch, sorted
Indices sorted_indices(const Batches &batches) {
    Indices all;
    for (const auto &b : batches) {
        all.insert(all.end(), b.begin(), b.end());
    }
    std::sort(all.begin(), all.end());
    return all;
}

// 0, 1, ..., n - 1
Indices zero_to(std::size_t n) {
    Indices v(n);
    std::iota(v.begin(), v.end(), std::size_t{0});
    return v;
}

// true if t is rows x cols and holds these values in row-major order
bool tensor_is(const nabla::Tensor &t, std::size_t rows, std::size_t cols,
               std::initializer_list<float> want) {
    if (t.rows() != rows || t.cols() != cols || want.size() != rows * cols) {
        return false;
    }
    auto w = want.begin();
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (!(std::fabs(t.at(r, c) - *w++) < 1e-5f)) {
                return false;
            }
        }
    }
    return true;
}

// 4 images of 3 features, so rows and columns differ. Image k holds
// 10k, 10k + 1 and 10k + 2.
const std::vector<float> four_images = {0,  1,  2,  10, 11, 12,
                                        20, 21, 22, 30, 31, 32};
const Bytes four_labels = {7, 8, 9, 6};

// Hands its input back as the logits and records how it was called, so a
// test can see the batches evaluate() makes and the mode it runs them in.
struct Spy : nabla::Module {
    Indices batch_rows;
    bool saw_grad = false;
    bool saw_training = false;

    nabla::Tensor forward(const nabla::Tensor &x) override {
        batch_rows.push_back(x.rows());
        saw_grad = saw_grad || nabla::grad_enabled;
        saw_training = saw_training || is_training();
        return x;
    }
};

// 5 examples whose 3 features are the logits, so they predict 1, 0, 2, 2
// and 1. The labels make 4 of them right.
const std::vector<float> five_logits = {0, 1, 0, 3, 1, 2, 0, 0,
                                        1, 1, 2, 3, 0, 5, 4};
const Bytes five_labels = {1, 0, 2, 0, 1};

} // namespace

// ---- M6.2 · read_u32_be ----

// the MNIST training count, as it sits in the header
TEST(u32_be_reads_mnist_count) {
    auto in = stream_of({0x00, 0x00, 0xEA, 0x60});
    CHECK(nabla::read_u32_be(in) == 60000u);
}

// four different bytes, so reading them little-endian gives 0x78563412
TEST(u32_be_first_byte_is_highest) {
    auto in = stream_of({0x12, 0x34, 0x56, 0x78});
    CHECK(nabla::read_u32_be(in) == 0x12345678u);
}

// bytes of 0x80 and up are negative as a plain char on x86
TEST(u32_be_high_bytes_stay_unsigned) {
    auto in = stream_of({0xFF, 0x80, 0xC0, 0xFE});
    CHECK(nabla::read_u32_be(in) == 0xFF80C0FEu);
}

TEST(u32_be_short_stream_throws) {
    auto in = stream_of({0x00, 0x00, 0xEA});
    CHECK(throws([&] { nabla::read_u32_be(in); }));
}

// ---- M6.2 · read_idx ----

TEST(idx_reads_labels) {
    auto in = stream_of(three_labels);
    const auto idx = nabla::read_idx(in);
    const Dims dims = {3};
    const Bytes data = {5, 0, 4};
    CHECK(idx.dims == dims);
    CHECK(idx.data == data);
}

// 2 images of 3 x 4, so every dimension differs. The pixels are
// 0, 11, 22, ..., 253: 11 is a whitespace byte, and 253 is above 0x7F.
TEST(idx_reads_images) {
    Bytes file = {0, 0, 0x08, 3, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4};
    Bytes pixels;
    for (int i = 0; i < 24; ++i) {
        pixels.push_back(static_cast<std::uint8_t>(i * 11));
    }
    file.insert(file.end(), pixels.begin(), pixels.end());

    auto in = stream_of(file);
    const auto idx = nabla::read_idx(in);
    const Dims dims = {2, 3, 4};
    CHECK(idx.dims == dims);
    CHECK(idx.data == pixels);
}

// a file that was never gunzipped starts 1F 8B, not 00 00
TEST(idx_gzip_file_throws) {
    auto in = stream_of({0x1F, 0x8B, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0x03});
    CHECK(throws([&] { nabla::read_idx(in); }));
}

// type 0x0D means float, which read_idx does not handle
TEST(idx_other_type_throws) {
    auto in = stream_of({0, 0, 0x0D, 1, 0, 0, 0, 2, 1, 2, 3, 4, 5, 6, 7, 8});
    CHECK(throws([&] { nabla::read_idx(in); }));
}

// the header says 2 dimensions, but the file ends after the first
TEST(idx_short_header_throws) {
    auto in = stream_of({0, 0, 0x08, 2, 0, 0, 0, 4});
    CHECK(throws([&] { nabla::read_idx(in); }));
}

// the header promises 4 labels, but only 3 follow
TEST(idx_short_data_throws) {
    auto in = stream_of({0, 0, 0x08, 1, 0, 0, 0, 4, 5, 0, 4});
    CHECK(throws([&] { nabla::read_idx(in); }));
}

// dims 2^20, 2^16 and 2^28 multiply to 2^64, which wraps a 64-bit size_t to
// 0, so without a check the header would pass with no data at all
TEST(idx_size_overflow_throws) {
    auto in =
        stream_of({0, 0, 0x08, 3, 0, 0x10, 0, 0, 0, 1, 0, 0, 0x10, 0, 0, 0});
    CHECK(throws([&] { nabla::read_idx(in); }));
}

// ---- M6.2 · load_idx ----

TEST(load_idx_reads_a_file) {
    const auto path =
        std::filesystem::temp_directory_path() / "nabla_test_labels.idx";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(three_labels.data()),
                  static_cast<std::streamsize>(three_labels.size()));
    }
    const auto idx = nabla::load_idx(path.string());
    std::filesystem::remove(path);

    const Dims dims = {3};
    const Bytes data = {5, 0, 4};
    CHECK(idx.dims == dims);
    CHECK(idx.data == data);
}

TEST(load_idx_missing_file_throws) {
    CHECK(throws([] { nabla::load_idx("no/such/file-idx1-ubyte"); }));
}

// ---- M6.3 · normalize_mnist ----

// 0 and 255 are the two ends of the range; 51 is 0.2 after dividing by 255,
// so 51 / 255 done in integers (0) or 51 / 256 both show up here
TEST(normalize_mnist_pixel_values) {
    const auto out = nabla::normalize_mnist({0, 51, 255});
    CHECK(values_are(out, {-0.4242129f, 0.2249270f, 2.8214865f}));
}

// a loop that runs while k <= pixels.size() - 1 never ends here, because
// 0 - 1 wraps to the largest size_t
TEST(normalize_mnist_empty_gives_empty) {
    CHECK(nabla::normalize_mnist({}).empty());
}

// ---- M6.4 · shuffled_batches ----

// 10 indices in batches of 4: sizes 4, 4, 2, and every index exactly once
TEST(batches_cover_every_index_once) {
    std::mt19937 rng(0);
    const auto batches = nabla::shuffled_batches(10, 4, rng);
    const Indices sizes = {4, 4, 2};
    CHECK(sizes_of(batches) == sizes);
    CHECK(sorted_indices(batches) == zero_to(10));
}

// 8 in batches of 4 is exactly 2 batches, with no empty third one
TEST(batches_divide_evenly) {
    std::mt19937 rng(0);
    const auto batches = nabla::shuffled_batches(8, 4, rng);
    const Indices sizes = {4, 4};
    CHECK(sizes_of(batches) == sizes);
    CHECK(sorted_indices(batches) == zero_to(8));
}

// The first of 10 batches out of 100 should hold 10 indices from all over,
// not 0 to 9. That fails both without a shuffle and with a shuffle inside
// each batch only, which would keep the same groups every epoch.
TEST(batches_are_shuffled) {
    std::mt19937 rng(0);
    const auto batches = nabla::shuffled_batches(100, 10, rng);
    CHECK(sizes_of(batches) == Indices(10, 10));
    if (!batches.empty()) {
        Indices first = batches[0];
        std::sort(first.begin(), first.end());
        CHECK(first != zero_to(10));
    }
}

// The same rng used for two epochs gives two different orders. A copy of
// the rng would start from the same state and repeat the first order.
TEST(batches_differ_between_epochs) {
    std::mt19937 rng(0);
    const auto epoch1 = nabla::shuffled_batches(100, 10, rng);
    const auto epoch2 = nabla::shuffled_batches(100, 10, rng);
    CHECK(!epoch1.empty());
    CHECK(epoch1 != epoch2);
}

// the same seed gives the same batches, so a training run can be repeated
TEST(batches_repeat_with_same_seed) {
    std::mt19937 rng_a(7);
    std::mt19937 rng_b(7);
    const auto a = nabla::shuffled_batches(100, 10, rng_a);
    const auto b = nabla::shuffled_batches(100, 10, rng_b);
    CHECK(!a.empty());
    CHECK(a == b);
}

TEST(batches_size_zero_throws) {
    std::mt19937 rng(0);
    bool threw = false;
    try {
        nabla::shuffled_batches(10, 0, rng);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---- M6.5 · make_batch ----

// row i of x is image indices[i], in the order the indices come
TEST(make_batch_picks_rows_in_index_order) {
    const auto batch = nabla::make_batch(four_images, four_labels, 3, {2, 0});
    CHECK(tensor_is(batch.x, 2, 3, {20, 21, 22, 0, 1, 2}));
    const Indices y = {9, 7};
    CHECK(batch.y == y);
}

// the last image, asked for twice
TEST(make_batch_last_image_twice) {
    const auto batch = nabla::make_batch(four_images, four_labels, 3, {3, 3});
    CHECK(tensor_is(batch.x, 2, 3, {30, 31, 32, 30, 31, 32}));
    const Indices y = {6, 6};
    CHECK(batch.y == y);
}

// there are 4 images, so index 4 is one past the end
TEST(make_batch_index_past_end_throws) {
    bool threw = false;
    try {
        nabla::make_batch(four_images, four_labels, 3, {1, 4});
    } catch (const std::out_of_range &) {
        threw = true;
    }
    CHECK(threw);
}

// ---- M6.7 · argmax_rows ----

// 3 x 4, so rows and columns differ. The second row is all negative, and
// the third has a tie, which goes to the first of the two.
TEST(argmax_rows_picks_largest_column) {
    const auto logits = nabla::Tensor::from(
        3, 4, {0.1f, 0.7f, 0.2f, 0.0f, -1, -3, -0.5f, -2, 2, 5, 5, 1});
    const Indices want = {1, 2, 1};
    CHECK(nabla::argmax_rows(logits) == want);
}

// ---- M6.7 · evaluate ----

TEST(evaluate_gives_share_correct) {
    Spy model;
    CHECK_NEAR(nabla::evaluate(model, five_logits, five_labels, 3, 2), 0.8f);
}

// 5 examples in batches of 2, in order: forward sees 2, 2 and 1 rows
TEST(evaluate_runs_in_batches) {
    Spy model;
    nabla::evaluate(model, five_logits, five_labels, 3, 2);
    const Indices rows = {2, 2, 1};
    CHECK(model.batch_rows == rows);
}

// forward runs in eval mode, so Dropout is off, and records no graph
TEST(evaluate_uses_eval_and_no_grad) {
    Spy model;
    nabla::evaluate(model, five_logits, five_labels, 3, 2);
    CHECK(!model.batch_rows.empty());
    CHECK(!model.saw_training);
    CHECK(!model.saw_grad);
}

// a model that was training trains again afterwards, one in eval stays in
// eval, and gradients are switched back on
TEST(evaluate_restores_mode) {
    Spy training;
    nabla::evaluate(training, five_logits, five_labels, 3, 2);
    CHECK(training.is_training());
    CHECK(nabla::grad_enabled);
    nabla::grad_enabled = true; // so a failure here cannot break later tests

    Spy evaluating;
    evaluating.eval();
    nabla::evaluate(evaluating, five_logits, five_labels, 3, 2);
    CHECK(!evaluating.is_training());
}
