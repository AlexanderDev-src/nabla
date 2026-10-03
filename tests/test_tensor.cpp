#include "nabla/nabla.hpp"
#include "harness.hpp"
#include "nabla/ops.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <type_traits>

using nabla::Tensor;

// Two-layer network, built only from ops that already exist:
//   h = relu(x @ w1 + b1)
//   y = h @ w2 + b2
static nabla::Tensor two_layer(const nabla::Tensor &x, const nabla::Tensor &w1,
                               const nabla::Tensor &b1, const nabla::Tensor &w2,
                               const nabla::Tensor &b2) {
    auto h = nabla::relu(nabla::add(nabla::matmul(x, w1), b1));
    auto y = nabla::add(nabla::matmul(h, w2), b2);
    return y;
}

// zeros: right shape, all elements start at 0
TEST(zeros_shape_and_values) {
    auto t = Tensor::zeros(3, 2);
    CHECK(t.rows() == 3);
    CHECK(t.cols() == 2);
    CHECK_NEAR(t.at(0, 0), 0.0f);
    CHECK_NEAR(t.at(2, 1), 0.0f);
}

// from: values land in row-major order
TEST(from_row_major) {
    auto t = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    CHECK_NEAR(t.at(0, 0), 1.0f);
    CHECK_NEAR(t.at(0, 1), 2.0f);
    CHECK_NEAR(t.at(1, 0), 3.0f);
    CHECK_NEAR(t.at(2, 1), 6.0f);
}

// from: a size that does not match rows * cols is rejected
TEST(from_rejects_wrong_size) {
    bool threw = false;
    try {
        Tensor::from(3, 2, {1, 2});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// full: every element carries the fill value
TEST(full_fills_every_element) {
    auto t = Tensor::full(2, 3, 7.5f);
    CHECK(t.rows() == 2);
    CHECK(t.cols() == 3);
    CHECK_NEAR(t.at(0, 0), 7.5f);
    CHECK_NEAR(t.at(1, 2), 7.5f);
}

// at: the non-const overload writes through to the buffer
TEST(at_writes_through) {
    auto t = Tensor::zeros(2, 2);
    t.at(1, 0) = 9.0f;
    CHECK_NEAR(t.at(1, 0), 9.0f);
    CHECK_NEAR(t.at(0, 1), 0.0f);
}

// randn: the same seed produces the same draws
TEST(randn_same_seed_same_draws) {
    std::mt19937 rng_a(42), rng_b(42);
    auto a = Tensor::randn(2, 2, rng_a);
    auto b = Tensor::randn(2, 2, rng_b);

    CHECK_NEAR(a.at(0, 0), b.at(0, 0));
    CHECK_NEAR(a.at(1, 1), b.at(1, 1));
}

// handle semantics: a copy names the same buffer, so writes are shared
TEST(copy_shares_buffer) {
    auto a = Tensor::zeros(2, 2);
    auto b = a;
    b.at(0, 0) = 3.0f;
    CHECK_NEAR(a.at(0, 0), 3.0f);
}

// matmul: matches the hand-computed product
TEST(matmul_forward) {
    auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    auto b = Tensor::from(2, 2, {7, 8, 9, 10});
    auto c = matmul(a, b);
    CHECK(c.rows() == 3);
    CHECK(c.cols() == 2);
    CHECK_NEAR(c.at(0, 0), 25.0f);
    CHECK_NEAR(c.at(0, 1), 28.0f);
    CHECK_NEAR(c.at(1, 0), 57.0f);
    CHECK_NEAR(c.at(1, 1), 64.0f);
    CHECK_NEAR(c.at(2, 0), 89.0f);
    CHECK_NEAR(c.at(2, 1), 100.0f);
}

// matmul: inner dimensions that differ are rejected
TEST(matmul_rejects_mismatched_shapes) {
    auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    bool threw = false;
    try {
        matmul(a, a);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// add: same shape, elementwise
TEST(add_same_shape) {
    auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    auto b = Tensor::from(3, 2, {10, 20, 30, 40, 50, 60});
    auto c = add(a, b);
    CHECK_NEAR(c.at(0, 0), 11.0f);
    CHECK_NEAR(c.at(1, 1), 44.0f);
    CHECK_NEAR(c.at(2, 1), 66.0f);
}

// add: [3x2] + [1x2] adds the one bias row to every row
TEST(add_row_broadcast) {
    auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    auto bias = Tensor::from(1, 2, {10, 20});
    auto c = add(a, bias);
    CHECK(c.rows() == 3);
    CHECK_NEAR(c.at(0, 0), 11.0f);
    CHECK_NEAR(c.at(0, 1), 22.0f);
    CHECK_NEAR(c.at(2, 0), 15.0f);
    CHECK_NEAR(c.at(2, 1), 26.0f);
}

// add: shapes that neither match nor broadcast are rejected
TEST(add_rejects_mismatched_shapes) {
    auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    auto b = Tensor::from(2, 2, {1, 2, 3, 4});
    bool threw = false;
    try {
        add(a, b);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// make_node: the result needs grad when any parent does
TEST(make_node_requires_grad) {
    auto w = Tensor::zeros(2, 2).set_requires_grad(true);
    auto x = Tensor::zeros(2, 2);
    auto y = matmul(x, w);
    CHECK(y.requires_grad());
    CHECK(y.impl()->parents.size() == 2);

    auto z = add(x, x);
    CHECK(!z.requires_grad());
    CHECK(z.impl()->parents.empty());
}

// backward: the root is seeded with 1.0 in every element
TEST(backward_seeds_ones) {
    auto w = Tensor::zeros(2, 3).set_requires_grad(true);
    auto x = Tensor::zeros(4, 2);
    auto y = matmul(x, w);
    y.backward();
    CHECK_NEAR(y.grad_at(0, 0), 1.0f);
    CHECK_NEAR(y.grad_at(3, 2), 1.0f);
}

// backward: a tensor that does not require grad refuses
TEST(backward_refuses_without_grad) {
    auto x = Tensor::zeros(2, 2);
    bool threw = false;
    try {
        x.backward();
    } catch (const std::logic_error &) {
        threw = true;
    }
    CHECK(threw);
}

// add backward: [3x2] + [1x2], the bias row sums the grad of all 3 rows
TEST(add_backward_broadcast) {
    auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6}).set_requires_grad(true);
    auto bias = Tensor::from(1, 2, {10, 20}).set_requires_grad(true);
    auto c = add(a, bias);
    c.backward();
    CHECK_NEAR(a.grad_at(0, 0), 1.0f);
    CHECK_NEAR(a.grad_at(2, 1), 1.0f);
    CHECK_NEAR(bias.grad_at(0, 0), 3.0f);
    CHECK_NEAR(bias.grad_at(0, 1), 3.0f);
}

// add backward: add(x, x) reaches x twice, so both paths accumulate
TEST(add_backward_reused_input) {
    auto x = Tensor::from(2, 2, {1, 2, 3, 4}).set_requires_grad(true);
    auto y = add(x, x);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 2.0f);
    CHECK_NEAR(x.grad_at(1, 1), 2.0f);
}

// matmul backward: [3x2] @ [2x3], non-square so rows and cols differ.
// With a seed of ones, A's grad is B's row sums in every row, and
// B's grad is A's column sums in every column.
TEST(matmul_backward_non_square) {
    auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6}).set_requires_grad(true);
    auto b =
        Tensor::from(2, 3, {7, 8, 9, 10, 11, 12}).set_requires_grad(true);
    auto c = matmul(a, b);
    c.backward();
    CHECK_NEAR(a.grad_at(0, 0), 24.0f);
    CHECK_NEAR(a.grad_at(0, 1), 33.0f);
    CHECK_NEAR(a.grad_at(2, 0), 24.0f);
    CHECK_NEAR(a.grad_at(2, 1), 33.0f);
    CHECK_NEAR(b.grad_at(0, 0), 9.0f);
    CHECK_NEAR(b.grad_at(0, 2), 9.0f);
    CHECK_NEAR(b.grad_at(1, 0), 12.0f);
    CHECK_NEAR(b.grad_at(1, 2), 12.0f);
}

// matmul backward: an input that does not need grad is left untouched
TEST(matmul_backward_skips_no_grad_input) {
    auto x = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    auto w =
        Tensor::from(2, 3, {7, 8, 9, 10, 11, 12}).set_requires_grad(true);
    auto y = matmul(x, w);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 0.0f);
    CHECK_NEAR(x.grad_at(2, 1), 0.0f);
    CHECK_NEAR(w.grad_at(0, 2), 9.0f);
    CHECK_NEAR(w.grad_at(1, 0), 12.0f);
}

// matmul backward: two matmuls in a row, so the upstream grad reaching
// the first one is not all ones. A rule that forgets to multiply by g
// gives [24, 33] here instead of [50, 68].
TEST(matmul_backward_chained) {
    auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6}).set_requires_grad(true);
    auto b = Tensor::from(2, 3, {7, 8, 9, 10, 11, 12});
    auto e = Tensor::from(3, 1, {1, 2, 3});
    auto d = matmul(matmul(a, b), e);
    d.backward();
    CHECK_NEAR(a.grad_at(0, 0), 50.0f);
    CHECK_NEAR(a.grad_at(0, 1), 68.0f);
    CHECK_NEAR(a.grad_at(2, 0), 50.0f);
    CHECK_NEAR(a.grad_at(2, 1), 68.0f);
}

// relu forward: negatives and 0 become 0, positives pass through
TEST(relu_forward) {
    auto x = Tensor::from(2, 3, {-2, 0, 0.5f, 1, -3, 4});
    auto y = relu(x);
    CHECK(y.rows() == 2);
    CHECK(y.cols() == 3);
    CHECK_NEAR(y.at(0, 0), 0.0f);
    CHECK_NEAR(y.at(0, 1), 0.0f);
    CHECK_NEAR(y.at(0, 2), 0.5f);
    CHECK_NEAR(y.at(1, 0), 1.0f);
    CHECK_NEAR(y.at(1, 1), 0.0f);
    CHECK_NEAR(y.at(1, 2), 4.0f);
}

// relu backward: the grad is 1 where x > 0 and 0 elsewhere, including
// at x == 0 exactly (the same choice PyTorch makes)
TEST(relu_backward_mask) {
    auto x =
        Tensor::from(2, 3, {-2, 0, 0.5f, 1, -3, 4}).set_requires_grad(true);
    auto y = relu(x);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 0.0f);
    CHECK_NEAR(x.grad_at(0, 1), 0.0f);
    CHECK_NEAR(x.grad_at(0, 2), 1.0f);
    CHECK_NEAR(x.grad_at(1, 0), 1.0f);
    CHECK_NEAR(x.grad_at(1, 1), 0.0f);
    CHECK_NEAR(x.grad_at(1, 2), 1.0f);
}

// relu backward: followed by a matmul, so the upstream grad is [1, 2, 3]
// in every row rather than all ones. A rule that ignores g gives 1
// where this expects 3.
TEST(relu_backward_upstream) {
    auto x =
        Tensor::from(2, 3, {-2, 0, 0.5f, 1, -3, 4}).set_requires_grad(true);
    auto w = Tensor::from(3, 1, {1, 2, 3});
    auto y = matmul(relu(x), w);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 0.0f);
    CHECK_NEAR(x.grad_at(0, 2), 3.0f);
    CHECK_NEAR(x.grad_at(1, 0), 1.0f);
    CHECK_NEAR(x.grad_at(1, 1), 0.0f);
    CHECK_NEAR(x.grad_at(1, 2), 3.0f);
}

// relu backward: add(relu(x), x) reaches x twice. The add rule puts 1
// into x first, so a relu rule that writes with = instead of += wipes
// that 1 out.
TEST(relu_backward_reused_input) {
    auto x =
        Tensor::from(2, 3, {-2, 0, 0.5f, 1, -3, 4}).set_requires_grad(true);
    auto y = add(relu(x), x);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 1.0f);
    CHECK_NEAR(x.grad_at(0, 1), 1.0f);
    CHECK_NEAR(x.grad_at(0, 2), 2.0f);
    CHECK_NEAR(x.grad_at(1, 0), 2.0f);
    CHECK_NEAR(x.grad_at(1, 1), 1.0f);
    CHECK_NEAR(x.grad_at(1, 2), 2.0f);
}

// NoGradGuard: inside the guard no op records the graph, even when an
// input needs grad, so backward() refuses. Forward values are unchanged.
TEST(no_grad_stops_graph) {
    auto x = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    auto w = Tensor::from(2, 2, {7, 8, 9, 10}).set_requires_grad(true);
    nabla::NoGradGuard guard;
    auto y = relu(add(matmul(x, w), x));
    CHECK(!y.requires_grad());
    CHECK(y.impl()->parents.empty());
    CHECK_NEAR(y.at(0, 0), 26.0f);
    CHECK_NEAR(y.at(2, 1), 106.0f);

    bool threw = false;
    try {
        y.backward();
    } catch (const std::logic_error &) {
        threw = true;
    }
    CHECK(threw);
}

// NoGradGuard: once the guard leaves scope, graph building is back on
TEST(no_grad_restores_on_scope_exit) {
    auto w = Tensor::from(2, 2, {7, 8, 9, 10}).set_requires_grad(true);
    {
        nabla::NoGradGuard guard;
    }
    CHECK(matmul(w, w).requires_grad());
}

// NoGradGuard: guards nest. When the inner guard ends, the outer one is
// still active, so graph building must stay off.
TEST(no_grad_nests) {
    auto w = Tensor::from(2, 2, {7, 8, 9, 10}).set_requires_grad(true);
    nabla::NoGradGuard outer;
    {
        nabla::NoGradGuard inner;
    }
    CHECK(!matmul(w, w).requires_grad());
}

// NoGradGuard: an exception thrown inside the guard still switches
// graph building back on, because leaving scope runs the destructor
TEST(no_grad_restores_on_exception) {
    auto w = Tensor::from(2, 2, {7, 8, 9, 10}).set_requires_grad(true);
    try {
        nabla::NoGradGuard guard;
        throw std::runtime_error("inference failed");
    } catch (const std::runtime_error &) {
    }
    CHECK(matmul(w, w).requires_grad());
}

// NoGradGuard: a copy would restore the flag a second time, so copying
// is not allowed
TEST(no_grad_not_copyable) {
    CHECK(!std::is_copy_constructible_v<nabla::NoGradGuard>);
    CHECK(!std::is_copy_assignable_v<nabla::NoGradGuard>);
}

// two-layer network: [2x3] -> 4 hidden units -> 2 outputs. The hidden
// pre-activation is [[0.5, -0.5, -2.5, 7], [2.5, 1, 1, -2]], so relu
// closes some units in each row, and those gates show up as zeros in
// w1's grad. Expected grads were checked by finite difference.
TEST(two_layer_network) {
    auto x = Tensor::from(2, 3, {1, 2, -1, 0, 1, 2});
    auto w1 =
        Tensor::from(3, 4, {1, -1, 0.5f, 2, 0, 1, -1, 1, 1, 0.5f, 1, -2})
            .set_requires_grad(true);
    auto b1 = Tensor::from(1, 4, {0.5f, -1, 0, 1}).set_requires_grad(true);
    auto w2 = Tensor::from(4, 2, {1, 2, 2, -0.5f, -1, 3, 0.5f, -2})
                  .set_requires_grad(true);
    auto b2 = Tensor::from(1, 2, {1, -1}).set_requires_grad(true);

    auto y = two_layer(x, w1, b1, w2, b2);
    const bool shape_ok = y.rows() == 2 && y.cols() == 2;
    CHECK(shape_ok);
    CHECK(y.requires_grad());

    // only run backward once forward is in shape, so a wrong forward
    // shows up as FAIL lines instead of an uncaught exception
    if (shape_ok && y.requires_grad()) {
        CHECK_NEAR(y.at(0, 0), 5.0f);
        CHECK_NEAR(y.at(0, 1), -14.0f);
        CHECK_NEAR(y.at(1, 0), 4.5f);
        CHECK_NEAR(y.at(1, 1), 6.5f);

        y.backward();

        CHECK_NEAR(b2.grad_at(0, 0), 2.0f);
        CHECK_NEAR(b2.grad_at(0, 1), 2.0f);

        CHECK_NEAR(w2.grad_at(0, 0), 3.0f);
        CHECK_NEAR(w2.grad_at(1, 1), 1.0f);
        CHECK_NEAR(w2.grad_at(2, 0), 1.0f);
        CHECK_NEAR(w2.grad_at(3, 1), 7.0f);

        CHECK_NEAR(b1.grad_at(0, 0), 6.0f);
        CHECK_NEAR(b1.grad_at(0, 1), 1.5f);
        CHECK_NEAR(b1.grad_at(0, 2), 2.0f);
        CHECK_NEAR(b1.grad_at(0, 3), -1.5f);

        CHECK_NEAR(w1.grad_at(0, 0), 3.0f);
        CHECK_NEAR(w1.grad_at(0, 1), 0.0f);
        CHECK_NEAR(w1.grad_at(0, 2), 0.0f);
        CHECK_NEAR(w1.grad_at(1, 0), 9.0f);
        CHECK_NEAR(w1.grad_at(1, 3), -3.0f);
        CHECK_NEAR(w1.grad_at(2, 1), 3.0f);
        CHECK_NEAR(w1.grad_at(2, 2), 4.0f);
        CHECK_NEAR(w1.grad_at(2, 3), 1.5f);

        CHECK_NEAR(x.grad_at(0, 0), 0.0f);
    }
}

// backward: a 10000-deep chain does not overflow the stack, and b,
// which is added 10001 times, collects a grad of 1 from each use
TEST(backward_deep_chain) {
    auto b = Tensor::zeros(1, 1).set_requires_grad(true);
    Tensor y = b;
    for (int k = 0; k < 10000; ++k) {
        y = add(y, b);
    }
    y.backward();
    CHECK_NEAR(y.grad_at(0, 0), 1.0f);
    CHECK_NEAR(b.grad_at(0, 0), 10001.0f);
}

// mul forward: element by element, no summing
TEST(mul_forward) {
    auto a = Tensor::from(2, 3, {1, -2, 3, 0.5f, 4, -1});
    auto b = Tensor::from(2, 3, {2, 3, -1, 4, 0.5f, -2});
    auto c = mul(a, b);
    CHECK(c.rows() == 2);
    CHECK(c.cols() == 3);
    CHECK_NEAR(c.at(0, 0), 2.0f);
    CHECK_NEAR(c.at(0, 1), -6.0f);
    CHECK_NEAR(c.at(0, 2), -3.0f);
    CHECK_NEAR(c.at(1, 0), 2.0f);
    CHECK_NEAR(c.at(1, 1), 2.0f);
    CHECK_NEAR(c.at(1, 2), 2.0f);
}

// mul: [2x3] and [3x2] hold the same number of values but are different
// shapes, so a check that only compares sizes lets this through
TEST(mul_rejects_mismatched_shapes) {
    auto a = Tensor::from(2, 3, {1, 2, 3, 4, 5, 6});
    auto b = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    bool threw = false;
    try {
        mul(a, b);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// mul backward: with a seed of ones, a's grad is b and b's grad is a.
// A rule that uses its own input instead of the other one fails here.
TEST(mul_backward_swaps_inputs) {
    auto a = Tensor::from(2, 3, {1, -2, 3, 0.5f, 4, -1}).set_requires_grad(true);
    auto b = Tensor::from(2, 3, {2, 3, -1, 4, 0.5f, -2}).set_requires_grad(true);
    auto c = mul(a, b);
    c.backward();
    CHECK_NEAR(a.grad_at(0, 0), 2.0f);
    CHECK_NEAR(a.grad_at(0, 2), -1.0f);
    CHECK_NEAR(a.grad_at(1, 1), 0.5f);
    CHECK_NEAR(b.grad_at(0, 0), 1.0f);
    CHECK_NEAR(b.grad_at(0, 2), 3.0f);
    CHECK_NEAR(b.grad_at(1, 1), 4.0f);
}

// mul backward: mul(x, x) reaches x through both inputs, so the grad is
// 2x. Writing with = instead of += gives x.
TEST(mul_backward_reused_input) {
    auto x = Tensor::from(2, 3, {1, -2, 3, 0.5f, 4, -1}).set_requires_grad(true);
    auto y = mul(x, x);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 2.0f);
    CHECK_NEAR(x.grad_at(0, 1), -4.0f);
    CHECK_NEAR(x.grad_at(0, 2), 6.0f);
    CHECK_NEAR(x.grad_at(1, 0), 1.0f);
    CHECK_NEAR(x.grad_at(1, 1), 8.0f);
    CHECK_NEAR(x.grad_at(1, 2), -2.0f);
}

// mul backward: followed by a matmul, so the upstream grad is [1, 2, 3] in
// every row. A rule that ignores g gives a's grad = b here. b does not
// need grad, so its grad stays 0.
TEST(mul_backward_upstream) {
    auto a = Tensor::from(2, 3, {1, -2, 3, 0.5f, 4, -1}).set_requires_grad(true);
    auto b = Tensor::from(2, 3, {2, 3, -1, 4, 0.5f, -2});
    auto w = Tensor::from(3, 1, {1, 2, 3});
    auto y = matmul(mul(a, b), w);
    y.backward();
    CHECK_NEAR(a.grad_at(0, 0), 2.0f);
    CHECK_NEAR(a.grad_at(0, 1), 6.0f);
    CHECK_NEAR(a.grad_at(0, 2), -3.0f);
    CHECK_NEAR(a.grad_at(1, 0), 4.0f);
    CHECK_NEAR(a.grad_at(1, 1), 1.0f);
    CHECK_NEAR(a.grad_at(1, 2), -6.0f);
    CHECK_NEAR(b.grad_at(0, 0), 0.0f);
    CHECK_NEAR(b.grad_at(1, 2), 0.0f);
}

// sum forward: every element added into one [1x1] value
TEST(sum_forward) {
    auto x = Tensor::from(2, 3, {1, -2, 3, 0.5f, 4, -1});
    auto s = sum(x);
    CHECK(s.rows() == 1);
    CHECK(s.cols() == 1);
    CHECK_NEAR(s.at(0, 0), 5.5f);
}

// sum backward: with a seed of 1, every element of x gets grad 1
TEST(sum_backward_ones) {
    auto x = Tensor::from(2, 3, {1, -2, 3, 0.5f, 4, -1}).set_requires_grad(true);
    auto s = sum(x);
    s.backward();
    CHECK_NEAR(x.grad_at(0, 0), 1.0f);
    CHECK_NEAR(x.grad_at(0, 2), 1.0f);
    CHECK_NEAR(x.grad_at(1, 1), 1.0f);
    CHECK_NEAR(x.grad_at(1, 2), 1.0f);
}

// sum backward: sum(x) times 3, so the upstream grad reaching sum is 3.
// A rule that adds 1 instead of g gives 1 here.
TEST(sum_backward_upstream) {
    auto x = Tensor::from(2, 3, {1, -2, 3, 0.5f, 4, -1}).set_requires_grad(true);
    auto c = Tensor::from(1, 1, {3});
    auto y = mul(sum(x), c);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 3.0f);
    CHECK_NEAR(x.grad_at(1, 2), 3.0f);
}

// tanh forward: values checked against Python's math.tanh
TEST(tanh_forward) {
    auto x = Tensor::from(2, 2, {-1, 0, 0.5f, 2});
    auto y = tanh(x);
    CHECK(y.rows() == 2);
    CHECK(y.cols() == 2);
    CHECK_NEAR(y.at(0, 0), -0.7615942f);
    CHECK_NEAR(y.at(0, 1), 0.0f);
    CHECK_NEAR(y.at(1, 0), 0.4621172f);
    CHECK_NEAR(y.at(1, 1), 0.9640276f);
}

// tanh backward: with a seed of 1 the grad is 1 - tanh(x)^2
TEST(tanh_backward) {
    auto x = Tensor::from(2, 2, {-1, 0, 0.5f, 2}).set_requires_grad(true);
    auto y = tanh(x);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 0.4199743f);
    CHECK_NEAR(x.grad_at(0, 1), 1.0f);
    CHECK_NEAR(x.grad_at(1, 0), 0.7864477f);
    CHECK_NEAR(x.grad_at(1, 1), 0.0706508f);
}

// tanh backward: tanh(x) times [1, 2, 3, 4], so the upstream grad is not
// all ones. A rule that ignores g gives the tanh_backward values here.
TEST(tanh_backward_upstream) {
    auto x = Tensor::from(2, 2, {-1, 0, 0.5f, 2}).set_requires_grad(true);
    auto c = Tensor::from(2, 2, {1, 2, 3, 4});
    auto y = mul(tanh(x), c);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 0.4199743f);
    CHECK_NEAR(x.grad_at(0, 1), 2.0f);
    CHECK_NEAR(x.grad_at(1, 0), 2.3593432f);
    CHECK_NEAR(x.grad_at(1, 1), 0.2826033f);
}

// tanh backward: add(tanh(x), x) reaches x twice. The add rule puts 1 into
// x first, so a tanh rule that writes with = instead of += wipes it out.
TEST(tanh_backward_reused_input) {
    auto x = Tensor::from(2, 2, {-1, 0, 0.5f, 2}).set_requires_grad(true);
    auto y = add(tanh(x), x);
    y.backward();
    CHECK_NEAR(x.grad_at(0, 0), 1.4199743f);
    CHECK_NEAR(x.grad_at(0, 1), 2.0f);
    CHECK_NEAR(x.grad_at(1, 0), 1.7864477f);
    CHECK_NEAR(x.grad_at(1, 1), 1.0706508f);
}
