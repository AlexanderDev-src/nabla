// M7.1: saving and loading parameters. Grouped by step, in the order of the
// M7 plan.
#include "harness.hpp"
#include "nabla/nabla.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;
using nabla::Tensor;

std::istringstream stream_of(const Bytes &b) {
    return std::istringstream(std::string(b.begin(), b.end()));
}

Bytes bytes_of(const std::ostringstream &out) {
    const std::string s = out.str();
    return Bytes(s.begin(), s.end());
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

// true if a and b have the same shape and exactly the same values
bool same_tensor(const Tensor &a, const Tensor &b) {
    return a.rows() == b.rows() && a.cols() == b.cols() &&
           a.impl()->data == b.impl()->data;
}

const float pi = std::numbers::pi_v<float>;

// A is 1x2 {1, -2} and B is 2x1 {0.5, pi}: the same number of values, so
// only the shape tells them apart. Each float is its 4 IEEE 754 bytes, low
// byte first: 1.0f is 0x3F800000, pi is 0x40490FDB.
const Bytes two_tensors = {
    'N',  'B',  'L',  'A',  // magic
    1,    0,    0,    0,    // version 1
    2,    0,    0,    0,    // 2 tensors
    1,    0,    0,    0,    // A: 1 row
    2,    0,    0,    0,    //    2 columns
    0x00, 0x00, 0x80, 0x3F, //    1.0f
    0x00, 0x00, 0x00, 0xC0, //    -2.0f
    2,    0,    0,    0,    // B: 2 rows
    1,    0,    0,    0,    //    1 column
    0x00, 0x00, 0x00, 0x3F, //    0.5f
    0xDB, 0x0F, 0x49, 0x40, //    pi
};

// two_tensors with byte k set to v
Bytes with_byte(std::size_t k, std::uint8_t v) {
    Bytes b = two_tensors;
    b[k] = v;
    return b;
}

// two_tensors cut to its first n bytes
Bytes first_bytes(std::size_t n) {
    return Bytes(two_tensors.begin(), two_tensors.begin() + n);
}

// 3 -> hidden -> ReLU -> 2, with weights drawn from seed
nabla::Sequential mlp(std::size_t hidden, unsigned seed) {
    std::mt19937 rng(seed);
    return nabla::Sequential{std::make_shared<nabla::Linear>(3, hidden, rng),
                             std::make_shared<nabla::ReLU>(),
                             std::make_shared<nabla::Linear>(hidden, 2, rng)};
}

// a copy of every parameter, to check later that nothing changed
std::vector<Tensor> snapshot(nabla::Module &m) {
    std::vector<Tensor> copies;
    for (const auto &p : m.parameters()) {
        copies.push_back(p.clone());
    }
    return copies;
}

bool unchanged(nabla::Module &m, const std::vector<Tensor> &before) {
    const auto now = m.parameters();
    if (now.size() != before.size()) {
        return false;
    }
    for (std::size_t i = 0; i < now.size(); ++i) {
        if (!same_tensor(now[i], before[i])) {
            return false;
        }
    }
    return true;
}

std::string temp_path(const char *name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

} // namespace

// ---- M7.1a · write_u32_le, read_u32_le ----

TEST(u32_le_writes_low_byte_first) {
    std::ostringstream out;
    nabla::write_u32_le(out, 0x12345678u);
    CHECK(bytes_of(out) == (Bytes{0x78, 0x56, 0x34, 0x12}));
}

// the MNIST training count again, this time low byte first
TEST(u32_le_reads_low_byte_first) {
    auto in = stream_of({0x60, 0xEA, 0x00, 0x00});
    CHECK(nabla::read_u32_le(in) == 60000u);
}

// bytes of 0x80 and up are negative as a plain char; read without
// unsigned char, 0xFF would spread its sign over the whole number
TEST(u32_le_reads_high_bytes) {
    auto in = stream_of({0xFF, 0x00, 0x80, 0x00});
    CHECK(nabla::read_u32_le(in) == 0x008000FFu);
}

TEST(u32_le_short_stream_throws) {
    auto in = stream_of({1, 2, 3});
    CHECK(throws([&] { nabla::read_u32_le(in); }));
}

// ---- M7.1b · write_checkpoint ----

TEST(write_checkpoint_exact_bytes) {
    const auto a = Tensor::from(1, 2, {1.0f, -2.0f});
    const auto b = Tensor::from(2, 1, {0.5f, pi});
    std::ostringstream out;
    nabla::write_checkpoint(out, {a, b});
    CHECK(bytes_of(out) == two_tensors);
}

TEST(write_checkpoint_no_tensors) {
    std::ostringstream out;
    nabla::write_checkpoint(out, {});
    const Bytes empty = {'N', 'B', 'L', 'A', 1, 0, 0, 0, 0, 0, 0, 0};
    CHECK(bytes_of(out) == empty);
}

// ---- M7.1c · read_checkpoint ----

TEST(read_checkpoint_exact_bytes) {
    const auto a = Tensor::zeros(1, 2);
    const auto b = Tensor::zeros(2, 1);
    auto in = stream_of(two_tensors);
    nabla::read_checkpoint(in, {a, b});
    CHECK(same_tensor(a, Tensor::from(1, 2, {1.0f, -2.0f})));
    CHECK(same_tensor(b, Tensor::from(2, 1, {0.5f, pi})));
}

// every bit comes back, not just the first few digits
TEST(checkpoint_stream_round_trip) {
    std::mt19937 rng(0);
    const auto a = Tensor::randn(2, 3, rng);
    const auto b = Tensor::randn(1, 3, rng);
    std::stringstream file;
    nabla::write_checkpoint(file, {a, b});

    const auto a2 = Tensor::zeros(2, 3);
    const auto b2 = Tensor::zeros(1, 3);
    nabla::read_checkpoint(file, {a2, b2});
    CHECK(same_tensor(a2, a));
    CHECK(same_tensor(b2, b));
}

TEST(read_checkpoint_bad_magic_throws) {
    auto in = stream_of(with_byte(0, 'X'));
    CHECK(throws([&] {
        nabla::read_checkpoint(in, {Tensor::zeros(1, 2), Tensor::zeros(2, 1)});
    }));
}

TEST(read_checkpoint_bad_version_throws) {
    auto in = stream_of(with_byte(4, 2));
    CHECK(throws([&] {
        nabla::read_checkpoint(in, {Tensor::zeros(1, 2), Tensor::zeros(2, 1)});
    }));
}

TEST(read_checkpoint_fewer_tensors_throws) {
    auto in = stream_of(two_tensors);
    CHECK(throws([&] { nabla::read_checkpoint(in, {Tensor::zeros(1, 2)}); }));
}

// the header says 1 tensor and the model has 2; B is still in the file,
// so only the count can catch it
TEST(read_checkpoint_more_tensors_throws) {
    auto in = stream_of(with_byte(8, 1));
    CHECK(throws([&] {
        nabla::read_checkpoint(in, {Tensor::zeros(1, 2), Tensor::zeros(2, 1)});
    }));
}

// B is 2x1 in the file and 1x2 here: same size, different shape
TEST(read_checkpoint_shape_mismatch_throws) {
    auto in = stream_of(two_tensors);
    CHECK(throws([&] {
        nabla::read_checkpoint(in, {Tensor::zeros(1, 2), Tensor::zeros(1, 2)});
    }));
}

TEST(read_checkpoint_short_file_throws) {
    for (const std::size_t n : {0uz, 6uz, 20uz, 40uz, 43uz}) {
        auto in = stream_of(first_bytes(n));
        CHECK(throws([&] {
            nabla::read_checkpoint(in,
                                   {Tensor::zeros(1, 2), Tensor::zeros(2, 1)});
        }));
    }
}

// A fits, then B fails, so nothing may have been written into A yet
TEST(read_checkpoint_failure_changes_nothing) {
    const auto a = Tensor::full(1, 2, 7.0f);
    const auto b = Tensor::full(1, 2, 7.0f);
    auto bad_shape = stream_of(two_tensors);
    CHECK(throws([&] { nabla::read_checkpoint(bad_shape, {a, b}); }));
    CHECK(same_tensor(a, Tensor::full(1, 2, 7.0f)));

    const auto c = Tensor::full(1, 2, 7.0f);
    const auto d = Tensor::full(2, 1, 7.0f);
    auto cut = stream_of(first_bytes(40));
    CHECK(throws([&] { nabla::read_checkpoint(cut, {c, d}); }));
    CHECK(same_tensor(c, Tensor::full(1, 2, 7.0f)));
    CHECK(same_tensor(d, Tensor::full(2, 1, 7.0f)));
}

// ---- M7.1d · save_checkpoint, load_checkpoint ----

TEST(checkpoint_file_round_trip) {
    const auto path = temp_path("nabla_test_round_trip.nbla");
    auto saved = mlp(4, 1);
    auto loaded = mlp(4, 2);
    nabla::save_checkpoint(path, saved);
    nabla::load_checkpoint(path, loaded);
    std::filesystem::remove(path);

    CHECK(unchanged(loaded, snapshot(saved)));
    const auto x = Tensor::from(2, 3, {1, -2, 3, 0.5f, 0, -1});
    CHECK(same_tensor(loaded.forward(x), saved.forward(x)));
}

// 4 hidden units in the file, 5 in the model
TEST(checkpoint_mismatched_architecture_throws) {
    const auto path = temp_path("nabla_test_mismatch.nbla");
    auto saved = mlp(4, 1);
    nabla::save_checkpoint(path, saved);

    auto wider = mlp(5, 2);
    const auto before = snapshot(wider);
    CHECK(throws([&] { nabla::load_checkpoint(path, wider); }));
    CHECK(unchanged(wider, before));

    std::mt19937 rng(3);
    nabla::Linear one_layer(3, 2, rng);
    CHECK(throws([&] { nabla::load_checkpoint(path, one_layer); }));
    std::filesystem::remove(path);
}

TEST(load_checkpoint_missing_file_throws) {
    auto model = mlp(4, 1);
    CHECK(throws([&] { nabla::load_checkpoint("no/such/model.nbla", model); }));
}

TEST(save_checkpoint_bad_path_throws) {
    auto model = mlp(4, 1);
    const auto path = temp_path("nabla_no_such_dir") + "/model.nbla";
    CHECK(throws([&] { nabla::save_checkpoint(path, model); }));
}
