// M3: every op checked against finite differences through gradcheck, plus
// tests that gradcheck itself catches a wrong backward rule.
#include "gradcheck.hpp"
#include "harness.hpp"
#include "nabla/nabla.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

using nabla::Tensor;
using nabla_test::gradcheck;
using Inputs = std::vector<Tensor>;

// Like CHECK(r.ok), but on failure also prints how far off the worst
// element was, which says whether the rule is wrong or the tolerance tight.
#define CHECK_GRAD(result)                                                     \
    do {                                                                       \
        const auto &r_ = (result);                                             \
        CHECK(r_.ok);                                                          \
        if (!r_.ok) {                                                          \
            std::printf("  max_error %.3e over %zu elements\n", r_.max_error,  \
                        r_.checked);                                           \
        }                                                                      \
    } while (0)

namespace {

// N(0, 1) values from a fixed seed, so every run checks the same numbers
Tensor rand_tensor(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    return Tensor::randn(rows, cols, rng);
}

// Like rand_tensor, but every value is at least 0.2 away from 0. relu has a
// kink at 0, and a step of h = 1e-2 must never cross it.
Tensor away_from_zero(std::size_t rows, std::size_t cols, unsigned seed) {
    auto t = rand_tensor(rows, cols, seed);
    for (float &v : t.impl()->data) {
        if (std::fabs(v) < 0.2f) {
            v += (v < 0.0f) ? -0.2f : 0.2f;
        }
    }
    return t;
}

// Doubles its input, but the backward rule forgets the factor of 2
Tensor bad_double(const Tensor &x) {
    auto out = nabla::make_node(x.rows(), x.cols(), {x});
    for (std::size_t k = 0; k < x.impl()->data.size(); ++k) {
        out.impl()->data[k] = 2.0f * x.impl()->data[k];
    }
    if (out.requires_grad()) {
        auto xi = x.impl();
        out.impl()->backward_fn = [xi](const std::vector<float> &g) {
            for (std::size_t k = 0; k < g.size(); ++k) {
                xi->grad[k] += g[k];
            }
        };
    }
    return out;
}

// Square identity whose backward sends each grad to the transposed
// position. With a seed of ones every position gets the same grad, so the
// mistake is invisible; only a mask that weights positions differently
// exposes it.
Tensor bad_transposed_identity(const Tensor &x) {
    auto out = nabla::make_node(x.rows(), x.cols(), {x});
    out.impl()->data = x.impl()->data;
    if (out.requires_grad()) {
        auto xi = x.impl();
        out.impl()->backward_fn = [xi](const std::vector<float> &g) {
            const std::size_t n = xi->rows;
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    xi->grad[i * n + j] += g[j * n + i];
                }
            }
        };
    }
    return out;
}

} // namespace

// --- gradcheck itself ------------------------------------------------------

// gradcheck compares every element of every input: 6 + 12 here
TEST(gradcheck_counts_every_element) {
    auto r = gradcheck([](const Inputs &in) { return matmul(in[0], in[1]); },
                       {rand_tensor(2, 3, 1), rand_tensor(3, 4, 2)});
    CHECK(r.checked == 18);
    CHECK_GRAD(r);
}

// a backward rule with the wrong scale must be reported
TEST(gradcheck_catches_wrong_scale) {
    auto r = gradcheck([](const Inputs &in) { return bad_double(in[0]); },
                       {rand_tensor(3, 3, 3)});
    CHECK(r.checked == 9);
    CHECK(!r.ok);
    CHECK(r.max_error > 0.1);
}

// a backward rule that swaps positions must be reported, which only works
// because of the random mask
TEST(gradcheck_catches_swapped_indices) {
    auto r = gradcheck(
        [](const Inputs &in) { return bad_transposed_identity(in[0]); },
        {rand_tensor(3, 3, 4)});
    CHECK(r.checked == 9);
    CHECK(!r.ok);
    CHECK(r.max_error > 0.1);
}

// gradcheck puts every input value back exactly as it found it
TEST(gradcheck_restores_inputs) {
    auto a = rand_tensor(2, 3, 5);
    const std::vector<float> before = a.impl()->data;
    gradcheck([](const Inputs &in) { return tanh(in[0]); }, {a});
    for (std::size_t k = 0; k < before.size(); ++k) {
        CHECK(a.impl()->data[k] == before[k]);
    }
}

// an input that already holds a grad from an earlier backward() must still
// check clean, so gradcheck has to clear grads before its own backward()
TEST(gradcheck_ignores_stale_grads) {
    auto a = rand_tensor(2, 3, 6).set_requires_grad(true);
    sum(a).backward();
    auto r = gradcheck([](const Inputs &in) { return tanh(in[0]); }, {a});
    CHECK_GRAD(r);
}

// --- every op against finite differences ----------------------------------

TEST(gc_matmul) {
    CHECK_GRAD(gradcheck([](const Inputs &in) { return matmul(in[0], in[1]); },
                         {rand_tensor(3, 4, 10), rand_tensor(4, 2, 11)}));
}

TEST(gc_add_same_shape) {
    CHECK_GRAD(gradcheck([](const Inputs &in) { return add(in[0], in[1]); },
                         {rand_tensor(3, 4, 12), rand_tensor(3, 4, 13)}));
}

TEST(gc_add_broadcast_bias) {
    CHECK_GRAD(gradcheck([](const Inputs &in) { return add(in[0], in[1]); },
                         {rand_tensor(3, 4, 14), rand_tensor(1, 4, 15)}));
}

TEST(gc_relu) {
    CHECK_GRAD(gradcheck([](const Inputs &in) { return relu(in[0]); },
                         {away_from_zero(3, 4, 16)}));
}

TEST(gc_mul) {
    CHECK_GRAD(gradcheck([](const Inputs &in) { return mul(in[0], in[1]); },
                         {rand_tensor(3, 4, 17), rand_tensor(3, 4, 18)}));
}

TEST(gc_sum) {
    CHECK_GRAD(gradcheck([](const Inputs &in) { return sum(in[0]); },
                         {rand_tensor(3, 4, 19)}));
}

TEST(gc_tanh) {
    CHECK_GRAD(gradcheck([](const Inputs &in) { return tanh(in[0]); },
                         {rand_tensor(3, 4, 20)}));
}

// diamond: a reaches the output along two paths, through relu and tanh
TEST(gc_diamond) {
    CHECK_GRAD(gradcheck(
        [](const Inputs &in) {
            return sum(mul(relu(in[0]), tanh(in[0])));
        },
        {away_from_zero(3, 4, 21)}));
}

// reuse: a * a, whose grad is 2a
TEST(gc_reuse) {
    CHECK_GRAD(gradcheck(
        [](const Inputs &in) { return sum(mul(in[0], in[0])); },
        {rand_tensor(3, 4, 22)}));
}
