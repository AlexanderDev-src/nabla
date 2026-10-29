// M5: optimisers. Grouped by step, in the order of the M5 plan.
#include "harness.hpp"
#include "nabla/nabla.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <random>
#include <vector>

using nabla::Tensor;

namespace {

// Optimizer is abstract. These tests only need its zero_grad().
class NoStep : public nabla::Optimizer {
  public:
    using nabla::Optimizer::Optimizer;
    void step() override {}
};

// loss = sum(w * c), so dloss/dw = c exactly
void backward_weighted_sum(const Tensor &w, const Tensor &c) {
    nabla::sum(nabla::mul(w, c)).backward();
}

Tensor make_param(std::size_t rows, std::size_t cols,
                  std::initializer_list<float> data) {
    Tensor t = Tensor::from(rows, cols, data);
    t.set_requires_grad(true);
    return t;
}

// grad of t equals scale * c, element by element
bool grad_is(const Tensor &t, const Tensor &c, float scale) {
    for (std::size_t k = 0; k < t.impl()->grad.size(); ++k) {
        // !(x < tol) rather than x >= tol, so a NaN counts as a mismatch
        if (!(std::fabs(t.impl()->grad[k] - scale * c.impl()->data[k]) <
              1e-5f)) {
            return false;
        }
    }
    return true;
}

bool all_grad_zero(const Tensor &t) {
    for (float g : t.impl()->grad) {
        if (g != 0.0f) {
            return false;
        }
    }
    return true;
}

} // namespace

// ---- M5.1 · Optimizer base and zero_grad ---------------------------------

// Why zero_grad exists: without it a second backward() adds onto the first.
TEST(optim_two_backwards_without_zero_grad_double) {
    auto w = make_param(2, 3, {1, 2, 3, 4, 5, 6});
    auto c = Tensor::from(2, 3, {0.5f, -1, 2, 3, -0.25f, 1.5f});
    backward_weighted_sum(w, c);
    backward_weighted_sum(w, c);
    CHECK(grad_is(w, c, 2.0f));
}

TEST(optim_zero_grad_then_backward_gives_single) {
    auto w = make_param(2, 3, {1, 2, 3, 4, 5, 6});
    auto c = Tensor::from(2, 3, {0.5f, -1, 2, 3, -0.25f, 1.5f});
    NoStep opt({w});
    backward_weighted_sum(w, c);
    backward_weighted_sum(w, c);
    opt.zero_grad();
    backward_weighted_sum(w, c);
    CHECK(grad_is(w, c, 1.0f));
}

// two params of different shapes, so "only the first one" or "only the
// first row" both show up
TEST(optim_zero_grad_clears_every_param) {
    auto w = make_param(2, 3, {1, 2, 3, 4, 5, 6});
    auto b = make_param(1, 2, {7, 8});
    backward_weighted_sum(w,
                          Tensor::from(2, 3, {0.5f, -1, 2, 3, -0.25f, 1.5f}));
    backward_weighted_sum(b, Tensor::from(1, 2, {4, -2}));
    NoStep opt({w, b});
    opt.zero_grad();
    CHECK(all_grad_zero(w));
    CHECK(all_grad_zero(b));
}

TEST(optim_zero_grad_keeps_data) {
    auto w = make_param(2, 3, {1, 2, 3, 4, 5, 6});
    backward_weighted_sum(w,
                          Tensor::from(2, 3, {0.5f, -1, 2, 3, -0.25f, 1.5f}));
    NoStep opt({w});
    opt.zero_grad();
    const std::vector<float> want = {1, 2, 3, 4, 5, 6};
    CHECK(w.impl()->data == want);
}

// the optimiser must update the model's own tensors, not copies of them
TEST(optim_holds_the_same_tensors) {
    auto w = make_param(2, 3, {1, 2, 3, 4, 5, 6});
    auto b = make_param(1, 2, {7, 8});
    NoStep opt({w, b});
    CHECK(opt.params().size() == 2);
    if (opt.params().size() == 2) {
        CHECK(opt.params()[0].impl() == w.impl());
        CHECK(opt.params()[1].impl() == b.impl());
    }
}

TEST(optim_zero_grad_on_linear_parameters) {
    std::mt19937 rng(11);
    nabla::Linear layer(3, 2, rng);
    auto x = Tensor::from(4, 3, {1, -2, 3, 0.5f, 1, -1, 2, 2, 0, -3, 1, 4});
    nabla::sum(layer.forward(x)).backward();
    CHECK(!all_grad_zero(layer.weight));
    CHECK(!all_grad_zero(layer.bias));
    NoStep opt(layer.parameters());
    opt.zero_grad();
    CHECK(all_grad_zero(layer.weight));
    CHECK(all_grad_zero(layer.bias));
}

// ---- M5.2 · SGD with momentum and weight decay ---------------------------
// Grads are set by hand so every expected value can be worked out on paper.

namespace {

void set_grad(const Tensor &t, std::initializer_list<float> g) {
    t.impl()->grad = std::vector<float>(g);
}

bool data_is(const Tensor &t, std::initializer_list<float> want) {
    std::size_t k = 0;
    for (float w : want) {
        // !(x < tol) rather than x >= tol, so a NaN counts as a mismatch
        if (!(std::fabs(t.impl()->data[k] - w) < 1e-5f)) {
            std::printf("  data[%zu] = %.6f, want %.6f\n", k, t.impl()->data[k],
                        w);
            return false;
        }
        ++k;
    }
    return true;
}

} // namespace

// data - lr * grad
TEST(sgd_plain_step) {
    auto w = make_param(2, 3, {1, 2, 3, 4, 5, 6});
    set_grad(w, {0.5f, -1, 2, 3, -0.25f, 1.5f});
    nabla::SGD opt({w}, 0.1f);
    opt.step();
    CHECK(data_is(w, {0.95f, 2.1f, 2.8f, 3.7f, 5.025f, 5.85f}));
}

TEST(sgd_step_keeps_grad) {
    auto w = make_param(2, 3, {1, 2, 3, 4, 5, 6});
    set_grad(w, {0.5f, -1, 2, 3, -0.25f, 1.5f});
    nabla::SGD opt({w}, 0.1f, 0.9f, 0.01f);
    opt.step();
    const std::vector<float> want = {0.5f, -1, 2, 3, -0.25f, 1.5f};
    CHECK(w.impl()->grad == want);
}

// step 1: v = g1 = [0.5, 1],          data = [0.95, -2.1]
// step 2: v = 0.9 v + g2 = [-0.55, 2.9], data = [1.005, -2.39]
TEST(sgd_momentum_two_steps) {
    auto w = make_param(1, 2, {1, -2});
    nabla::SGD opt({w}, 0.1f, 0.9f);
    set_grad(w, {0.5f, 1});
    opt.step();
    CHECK(data_is(w, {0.95f, -2.1f}));
    set_grad(w, {-1, 2});
    opt.step();
    CHECK(data_is(w, {1.005f, -2.39f}));
}

// g = [0.5, 1] + 0.01 * [2, -4] = [0.52, 0.96]
TEST(sgd_weight_decay_step) {
    auto w = make_param(1, 2, {2, -4});
    set_grad(w, {0.5f, 1});
    nabla::SGD opt({w}, 0.1f, 0.0f, 0.01f);
    opt.step();
    CHECK(data_is(w, {1.948f, -4.096f}));
}

// weight decay goes into g before the momentum line, so it is inside v
// step 1: g = [0.52, 0.96],       v = g,             data = [1.948, -4.096]
// step 2: g = [0.51948, 0.95904], v = [0.98748, 1.82304],
//         data = [1.849252, -4.278304]
TEST(sgd_momentum_with_weight_decay) {
    auto w = make_param(1, 2, {2, -4});
    set_grad(w, {0.5f, 1});
    nabla::SGD opt({w}, 0.1f, 0.9f, 0.01f);
    opt.step();
    CHECK(data_is(w, {1.948f, -4.096f}));
    opt.step();
    CHECK(data_is(w, {1.849252f, -4.278304f}));
}

// each parameter keeps its own velocity, and shapes differ so a mix-up shows
// a: v    = [1, 0, 1]   then [1.9, 0, 1.9]
//    data = [0.9, 1, 0.9] then [0.71, 1, 0.71]
// b: v    = [0, 2]   then [0, 3.8]
//    data = [1, 0.8] then [1, 0.42]
TEST(sgd_momentum_per_param) {
    auto a = make_param(1, 3, {1, 1, 1});
    auto b = make_param(2, 1, {1, 1});
    set_grad(a, {1, 0, 1});
    set_grad(b, {0, 2});
    nabla::SGD opt({a, b}, 0.1f, 0.9f);
    opt.step();
    opt.step();
    CHECK(data_is(a, {0.71f, 1, 0.71f}));
    CHECK(data_is(b, {1, 0.42f}));
}

// ---- M5.3 · Linear regression with SGD ------------------------------------

// y = 2 x0 - 3 x1 + 0.5 with no noise, so SGD should find those exact numbers
TEST(sgd_linear_regression_recovers_weights) {
    std::mt19937 rng(0);
    const std::size_t n = 64;
    Tensor x = Tensor::randn(n, 2, rng);
    Tensor y = Tensor::zeros(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        y.impl()->data[i] = 2 * x.at(i, 0) - 3 * x.at(i, 1) + 0.5f;
    }
    nabla::Linear model(2, 1, rng);
    nabla::SGD opt(model.parameters(), 0.1f);

    for (int step = 0; step < 200; ++step) {
        opt.zero_grad();
        auto prev = model.forward(x);
        auto loss = nabla::mse_loss(prev, y);
        loss.backward();
        opt.step();
    }

    CHECK(std::fabs(model.weight.at(0, 0) - 2.0f) < 1e-3f);
    CHECK(std::fabs(model.weight.at(1, 0) + 3.0f) < 1e-3f);
    CHECK(std::fabs(model.bias.at(0, 0) - 0.5f) < 1e-3f);
}

// ---- M5.4 · Adam --------------------------------------------------------

// On the first step m_hat = g and v_hat = g * g, so every element moves by
// lr * g / |g| = lr * sign(g), whether g is 0.001 or 1000. g = 0 stays put.
TEST(adam_first_step_is_lr_times_sign) {
    auto w = make_param(2, 3, {1, 1, 1, 1, 1, 1});
    set_grad(w, {0.001f, -5, 1000, -0.02f, 0, 3});
    nabla::Adam opt({w}, 0.1f);
    opt.step();
    CHECK(data_is(w, {0.9f, 1.1f, 0.9f, 1.1f, 1, 0.9f}));
}

TEST(adam_step_keeps_grad) {
    auto w = make_param(1, 2, {1, -2});
    set_grad(w, {0.5f, -1});
    nabla::Adam opt({w}, 0.1f);
    opt.step();
    const std::vector<float> want = {0.5f, -1};
    CHECK(w.impl()->grad == want);
}

// step 1: m = [0.05, -0.1],    v = [0.00025, 0.001],      data = [0.9, -1.9]
// step 2: m = [-0.055, 0.11],  v = [0.00124975, 0.004999]
//         m_hat = m / 0.19,    v_hat = v / 0.001999
//         data = [0.936610, -1.936610]
TEST(adam_two_steps_by_hand) {
    auto w = make_param(1, 2, {1, -2});
    nabla::Adam opt({w}, 0.1f);
    set_grad(w, {0.5f, -1});
    opt.step();
    CHECK(data_is(w, {0.9f, -1.9f}));
    set_grad(w, {-1, 2});
    opt.step();
    CHECK(data_is(w, {0.93661f, -1.93661f}));
}

// With a constant g, m_hat = g and v_hat = g * g at every t, so each step
// moves exactly lr * sign(g). That only holds if both params see the same t
// and each keeps its own m and v; shapes differ so a mix-up shows.
TEST(adam_two_params_own_state_same_t) {
    auto a = make_param(1, 3, {1, 1, 1});
    auto b = make_param(2, 1, {1, 1});
    set_grad(a, {1, 0, -2});
    set_grad(b, {0.5f, -3});
    nabla::Adam opt({a, b}, 0.1f);
    opt.step();
    opt.step();
    opt.step();
    CHECK(data_is(a, {0.7f, 1, 1.3f}));
    CHECK(data_is(b, {0.7f, 1.3f}));
}
