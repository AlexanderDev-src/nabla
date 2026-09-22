#include "nabla/nabla.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);             \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b) CHECK(std::fabs((a) - (b)) < 1e-5f)

int main() {
    using nabla::Tensor;

    // zeros: right shape, all elements start at 0
    {
        auto t = Tensor::zeros(3, 2);
        CHECK(t.rows() == 3);
        CHECK(t.cols() == 2);
        CHECK_NEAR(t.at(0, 0), 0.0f);
        CHECK_NEAR(t.at(2, 1), 0.0f);
    }

    // from: values land in row-major order
    {
        auto t = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
        CHECK_NEAR(t.at(0, 0), 1.0f);
        CHECK_NEAR(t.at(0, 1), 2.0f);
        CHECK_NEAR(t.at(1, 0), 3.0f);
        CHECK_NEAR(t.at(2, 1), 6.0f);
    }

    // from: a size that does not match rows * cols is rejected
    {
        bool threw = false;
        try {
            Tensor::from(3, 2, {1, 2});
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    // full: every element carries the fill value
    {
        auto t = Tensor::full(2, 3, 7.5f);
        CHECK(t.rows() == 2);
        CHECK(t.cols() == 3);
        CHECK_NEAR(t.at(0, 0), 7.5f);
        CHECK_NEAR(t.at(1, 2), 7.5f);
    }

    // at: the non-const overload writes through to the buffer
    {
        auto t = Tensor::zeros(2, 2);
        t.at(1, 0) = 9.0f;
        CHECK_NEAR(t.at(1, 0), 9.0f);
        CHECK_NEAR(t.at(0, 1), 0.0f);
    }

    // randn: the same seed produces the same draws
    {
        std::mt19937 rng_a(42), rng_b(42);
        auto a = Tensor::randn(2, 2, rng_a);
        auto b = Tensor::randn(2, 2, rng_b);

        CHECK_NEAR(a.at(0, 0), b.at(0, 0));
        CHECK_NEAR(a.at(1, 1), b.at(1, 1));
    }

    // handle semantics: a copy names the same buffer, so writes are shared
    {
        auto a = Tensor::zeros(2, 2);
        auto b = a;
        b.at(0, 0) = 3.0f;
        CHECK_NEAR(a.at(0, 0), 3.0f);
    }
    // matmul: matches the hand-computed product
    {
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
    {
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
    {
        auto a = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
        auto b = Tensor::from(3, 2, {10, 20, 30, 40, 50, 60});
        auto c = add(a, b);
        CHECK_NEAR(c.at(0, 0), 11.0f);
        CHECK_NEAR(c.at(1, 1), 44.0f);
        CHECK_NEAR(c.at(2, 1), 66.0f);
    }

    // add: [3x2] + [1x2] adds the one bias row to every row
    {
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
    {
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
    {
        auto w = Tensor::zeros(2, 2).set_requires_grad(true);
        auto x = Tensor::zeros(2, 2);
        auto y = matmul(x, w);
        CHECK(y.requires_grad());
        CHECK(y.impl()->parents.size() == 2);

        auto z = add(x, x);
        CHECK(!z.requires_grad());
        CHECK(z.impl()->parents.empty());
    }

    if (failures) {
        printf("\n%d FAILED\n", failures);
    } else {
        printf("\nall passed\n");
    }
    return failures;
}
