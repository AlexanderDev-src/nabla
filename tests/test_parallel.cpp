// M7.2: matmul on several threads. Each test runs the same work on one
// thread and on four, and wants the same bits back: two threads adding into
// the same grad element lose updates, and the totals come out different.
// Built without OpenMP, both runs are the same single thread.
#include "harness.hpp"
#include "nabla/nabla.hpp"
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using nabla::Tensor;

// Runs f on n OpenMP threads, then puts the old thread count back
template <class F> void on_threads(int n, F f) {
#ifdef _OPENMP
    const int before = omp_get_max_threads();
    omp_set_num_threads(n);
    f();
    omp_set_num_threads(before);
#else
    (void)n;
    f();
#endif
}

// A @ B worked out in double, one element at a time
std::vector<double> product(const Tensor &a, const Tensor &b) {
    std::vector<double> c(a.rows() * b.cols(), 0.0);
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            for (std::size_t p = 0; p < a.cols(); ++p) {
                c[i * b.cols() + j] += double(a.at(i, p)) * b.at(p, j);
            }
        }
    }
    return c;
}

// the transpose of t, as a new Tensor
Tensor transposed(const Tensor &t) {
    auto out = Tensor::zeros(t.cols(), t.rows());
    for (std::size_t r = 0; r < t.rows(); ++r) {
        for (std::size_t c = 0; c < t.cols(); ++c) {
            out.at(c, r) = t.at(r, c);
        }
    }
    return out;
}

// true if every value is within 1e-4 of want, scaled up for big values.
// Written so that NaN fails.
bool close_to(const std::vector<float> &got, const std::vector<double> &want) {
    if (got.size() != want.size()) {
        return false;
    }
    for (std::size_t k = 0; k < got.size(); ++k) {
        if (!(std::fabs(got[k] - want[k]) < 1e-4 * (1 + std::fabs(want[k])))) {
            return false;
        }
    }
    return true;
}

struct Grads {
    std::vector<float> a, b;
};

// The grads of sum(mask * (A @ B)), which makes the upstream grad of the
// matmul equal to mask: dA = mask @ B^T and dB = A^T @ mask.
Grads matmul_grads(const Tensor &a, const Tensor &b, const Tensor &mask) {
    const auto a2 = a.clone().set_requires_grad(true);
    const auto b2 = b.clone().set_requires_grad(true);
    nabla::sum(nabla::mul(nabla::matmul(a2, b2), mask)).backward();
    return {a2.impl()->grad, b2.impl()->grad};
}

// 256 rows of A, so on four threads 64 rows each add into every element of
// B's grad at the same time. Odd sizes leave the threads uneven chunks.
struct Case {
    Tensor a, b, mask;
};

Case big_case() {
    std::mt19937 rng(7);
    auto a = Tensor::randn(256, 67, rng);
    auto b = Tensor::randn(67, 45, rng);
    auto mask = Tensor::randn(256, 45, rng);
    return {a, b, mask};
}

} // namespace

// ---- M7.2a · forward ----

TEST(matmul_threads_forward_same_bits) {
    const auto c = big_case();
    std::vector<float> one, four;
    on_threads(1, [&] { one = nabla::matmul(c.a, c.b).impl()->data; });
    on_threads(4, [&] { four = nabla::matmul(c.a, c.b).impl()->data; });
    CHECK(four == one);
    CHECK(close_to(four, product(c.a, c.b)));
}

// ---- M7.2b · backward ----

TEST(matmul_threads_backward_same_bits) {
    const auto c = big_case();
    Grads one, four;
    on_threads(1, [&] { one = matmul_grads(c.a, c.b, c.mask); });
    on_threads(4, [&] { four = matmul_grads(c.a, c.b, c.mask); });
    CHECK(four.a == one.a);
    CHECK(four.b == one.b);
    CHECK(close_to(four.a, product(c.mask, transposed(c.b))));
    CHECK(close_to(four.b, product(transposed(c.a), c.mask)));
}

// a race does not lose updates every time, so give it several chances
TEST(matmul_threads_backward_repeatable) {
    const auto c = big_case();
    Grads first;
    on_threads(4, [&] { first = matmul_grads(c.a, c.b, c.mask); });
    for (int run = 0; run < 5; ++run) {
        Grads again;
        on_threads(4, [&] { again = matmul_grads(c.a, c.b, c.mask); });
        CHECK(again.a == first.a);
        CHECK(again.b == first.b);
    }
}

// only B needs grad, the way the input batch x never does
TEST(matmul_threads_backward_one_side) {
    const auto c = big_case();
    const auto b = c.b.clone().set_requires_grad(true);
    on_threads(4, [&] {
        nabla::sum(nabla::mul(nabla::matmul(c.a, b), c.mask)).backward();
    });
    CHECK(close_to(b.impl()->grad, product(transposed(c.a), c.mask)));
    CHECK(close_to(c.a.impl()->grad,
                   std::vector<double>(c.a.rows() * c.a.cols(), 0.0)));
}

TEST(matmul_threads_backward_shared_operand) {
    std::mt19937 rng(8);
    const auto original = Tensor::randn(17, 17, rng);
    const auto mask = Tensor::randn(17, 17, rng);
    const auto left = product(mask, transposed(original));
    const auto right = product(transposed(original), mask);
    std::vector<double> expected(left.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] = left[i] + right[i];
    }
    std::vector<float> one, four;
    for (int threads : {1, 4}) {
        const auto a = original.clone().set_requires_grad(true);
        on_threads(threads, [&] {
            nabla::sum(nabla::mul(nabla::matmul(a, a), mask)).backward();
        });
        CHECK(close_to(a.impl()->grad, expected));
        (threads == 1 ? one : four) = a.impl()->grad;
    }
    CHECK(one == four);
}
