// M4: sigmoid, modules (Linear, activations, Sequential) and the two fused
// losses. Grouped by step, in the order of the M4 plan.
#include "gradcheck.hpp"
#include "harness.hpp"
#include "nabla/nabla.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using nabla::Tensor;
using nabla_test::gradcheck;
using Inputs = std::vector<Tensor>;

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

Tensor rand_tensor(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    return Tensor::randn(rows, cols, rng);
}

bool same_values(const Tensor &a, const Tensor &b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }
    for (std::size_t k = 0; k < a.impl()->data.size(); ++k) {
        if (std::fabs(a.impl()->data[k] - b.impl()->data[k]) >= 1e-5f) {
            return false;
        }
    }
    return true;
}

// The smallest Module there is, so the base class can be tested alone
struct Identity : nabla::Module {
    Tensor forward(const Tensor &x) override { return x; }
};

} // namespace

// --- M4.1 sigmoid ----------------------------------------------------------

TEST(sigmoid_forward) {
    auto x = Tensor::from(2, 3, {0, 2, -2, 1, -1, 4});
    auto y = sigmoid(x);
    CHECK(y.rows() == 2);
    CHECK(y.cols() == 3);
    CHECK_NEAR(y.at(0, 0), 0.5f);
    CHECK_NEAR(y.at(0, 1), 0.8807971f);
    CHECK_NEAR(y.at(0, 2), 0.1192029f);
    CHECK_NEAR(y.at(1, 0), 0.7310586f);
    CHECK_NEAR(y.at(1, 1), 0.2689414f);
    CHECK_NEAR(y.at(1, 2), 0.9820138f);
}

// sigmoid at +-100: exp(100) overflows float to inf, and the result must
// still be 0 and 1, not nan. exp(x) / (1 + exp(x)) gives inf / inf = nan.
TEST(sigmoid_large_inputs) {
    auto x = Tensor::from(1, 2, {-100, 100}).set_requires_grad(true);
    auto y = sigmoid(x);
    CHECK(std::isfinite(y.at(0, 0)));
    CHECK(std::isfinite(y.at(0, 1)));
    CHECK_NEAR(y.at(0, 0), 0.0f);
    CHECK_NEAR(y.at(0, 1), 1.0f);
    sum(y).backward();
    CHECK(std::isfinite(x.grad_at(0, 0)));
    CHECK(std::isfinite(x.grad_at(0, 1)));
}

TEST(gc_sigmoid) {
    CHECK_GRAD(gradcheck([](const Inputs &in) { return sigmoid(in[0]); },
                         {rand_tensor(3, 4, 30)}));
}

// add(sigmoid(a), a): the add rule writes into a first, so a sigmoid rule
// that uses = instead of += wipes that out
TEST(gc_sigmoid_reused_input) {
    CHECK_GRAD(gradcheck(
        [](const Inputs &in) { return add(sigmoid(in[0]), in[0]); },
        {rand_tensor(3, 4, 31)}));
}

// --- M4.2 Module -----------------------------------------------------------

TEST(module_starts_in_training) {
    Identity m;
    CHECK(m.is_training());
}

TEST(module_train_eval_switch) {
    Identity m;
    m.eval();
    CHECK(!m.is_training());
    m.train();
    CHECK(m.is_training());
}

TEST(module_default_has_no_parameters) {
    Identity m;
    CHECK(m.parameters().empty());
}

// --- M4.3 Linear -----------------------------------------------------------

// Linear(3, 5): weight [3x5], bias [1x5], and [4x3] in gives [4x5] out
TEST(linear_shapes) {
    std::mt19937 rng(1);
    nabla::Linear layer(3, 5, rng);
    CHECK(layer.weight.rows() == 3);
    CHECK(layer.weight.cols() == 5);
    CHECK(layer.bias.rows() == 1);
    CHECK(layer.bias.cols() == 5);
    auto y = layer.forward(rand_tensor(4, 3, 2));
    CHECK(y.rows() == 4);
    CHECK(y.cols() == 5);
}

TEST(linear_bias_starts_at_zero) {
    std::mt19937 rng(3);
    nabla::Linear layer(3, 5, rng);
    for (float v : layer.bias.impl()->data) {
        CHECK(v == 0.0f);
    }
}

// weights and bias are leaves that need grad, so the output does too
TEST(linear_requires_grad) {
    std::mt19937 rng(4);
    nabla::Linear layer(3, 2, rng);
    CHECK(layer.weight.requires_grad());
    CHECK(layer.bias.requires_grad());
    CHECK(layer.forward(rand_tensor(2, 3, 5)).requires_grad());
}

// He init: std = sqrt(2 / fan_in) = sqrt(2 / 200) = 0.1 for 60000 weights.
// fan_out would give 0.082 and 2 / fan_in (the variance) would give 0.01,
// so a 5% window tells all three apart.
TEST(linear_he_init_std) {
    std::mt19937 rng(6);
    nabla::Linear layer(200, 300, rng);
    const auto &w = layer.weight.impl()->data;
    double mean = 0.0;
    for (float v : w) {
        mean += v;
    }
    mean /= static_cast<double>(w.size());
    double var = 0.0;
    for (float v : w) {
        var += (v - mean) * (v - mean);
    }
    const double stddev = std::sqrt(var / static_cast<double>(w.size()));
    CHECK(std::fabs(mean) < 0.005);
    CHECK(std::fabs(stddev - 0.1) < 0.005);
}

// parameters() must hand back the layer's own tensors, not copies, or an
// update to them never reaches the layer
TEST(linear_parameters_are_the_weights) {
    std::mt19937 rng(7);
    nabla::Linear layer(3, 2, rng);
    auto params = layer.parameters();
    CHECK(params.size() == 2);
    if (params.size() == 2) {
        CHECK(params[0].impl() == layer.weight.impl());
        CHECK(params[1].impl() == layer.bias.impl());
    }
}

// x @ W + b with hand-picked numbers: x [2x3], W [3x2], b [1x2]
TEST(linear_forward_values) {
    std::mt19937 rng(8);
    nabla::Linear layer(3, 2, rng);
    layer.weight.impl()->data = {1, 0, 0, 1, 1, -1};
    layer.bias.impl()->data = {0.5f, -0.5f};
    auto y = layer.forward(Tensor::from(2, 3, {1, 2, 3, -1, 0, 2}));
    CHECK(y.rows() == 2);
    CHECK(y.cols() == 2);
    CHECK_NEAR(y.at(0, 0), 4.5f);
    CHECK_NEAR(y.at(0, 1), -1.5f);
    CHECK_NEAR(y.at(1, 0), 1.5f);
    CHECK_NEAR(y.at(1, 1), -2.5f);
}

// --- M4.4 ReLU, Tanh, Sigmoid modules -------------------------------------

TEST(activation_modules_match_ops) {
    auto x = Tensor::from(2, 3, {-1.5f, 0.5f, 2, -0.2f, 0, 1});
    CHECK(same_values(nabla::ReLU().forward(x), relu(x)));
    CHECK(same_values(nabla::Tanh().forward(x), nabla::tanh(x)));
    CHECK(same_values(nabla::Sigmoid().forward(x), sigmoid(x)));
}

TEST(activation_modules_have_no_parameters) {
    CHECK(nabla::ReLU().parameters().empty());
    CHECK(nabla::Tanh().parameters().empty());
    CHECK(nabla::Sigmoid().parameters().empty());
}

// --- M4.5 Sequential -------------------------------------------------------

TEST(sequential_chains_forward) {
    std::mt19937 rng(9);
    auto l1 = std::make_shared<nabla::Linear>(3, 4, rng);
    auto l2 = std::make_shared<nabla::Linear>(4, 2, rng);
    nabla::Sequential model{l1, std::make_shared<nabla::ReLU>(), l2};
    auto x = rand_tensor(5, 3, 10);
    auto y = model.forward(x);
    CHECK(y.rows() == 5);
    CHECK(y.cols() == 2);
    CHECK(same_values(y, l2->forward(relu(l1->forward(x)))));
}

// layer order, and inside each layer weight before bias
TEST(sequential_collects_parameters_in_order) {
    std::mt19937 rng(11);
    auto l1 = std::make_shared<nabla::Linear>(3, 4, rng);
    auto l2 = std::make_shared<nabla::Linear>(4, 2, rng);
    nabla::Sequential model{l1, std::make_shared<nabla::ReLU>(), l2};
    auto params = model.parameters();
    CHECK(params.size() == 4);
    if (params.size() == 4) {
        CHECK(params[0].impl() == l1->weight.impl());
        CHECK(params[1].impl() == l1->bias.impl());
        CHECK(params[2].impl() == l2->weight.impl());
        CHECK(params[3].impl() == l2->bias.impl());
    }
}

// eval() on the model must reach every layer inside it, and train() too
TEST(sequential_eval_reaches_children) {
    std::mt19937 rng(12);
    auto l1 = std::make_shared<nabla::Linear>(3, 4, rng);
    auto act = std::make_shared<nabla::ReLU>();
    nabla::Sequential model{l1, act};
    model.eval();
    CHECK(!model.is_training());
    CHECK(!l1->is_training());
    CHECK(!act->is_training());
    model.train();
    CHECK(model.is_training());
    CHECK(l1->is_training());
    CHECK(act->is_training());
}

// --- M4.6 mse_loss ---------------------------------------------------------

// diffs 1, 0, -2, 3, squares 1, 0, 4, 9, mean 14 / 4
TEST(mse_value) {
    auto p = Tensor::from(2, 2, {1, 2, 3, 4});
    auto t = Tensor::from(2, 2, {0, 2, 5, 1});
    auto loss = mse_loss(p, t);
    CHECK(loss.rows() == 1);
    CHECK(loss.cols() == 1);
    CHECK_NEAR(loss.at(0, 0), 3.5f);
}

// [2x3] on purpose: rows and cols differ, so a loop that stops at the wrong
// one misses a column. Squares 1, 0, 4, 9, 0, 16, mean 30 / 6.
TEST(mse_value_non_square) {
    auto p = Tensor::from(2, 3, {1, 2, 3, 4, 5, 6});
    auto t = Tensor::from(2, 3, {0, 2, 5, 1, 5, 2});
    CHECK_NEAR(mse_loss(p, t).at(0, 0), 5.0f);
}

// grad is 2 (p - t) / N with N = 4
TEST(mse_grad) {
    auto p = Tensor::from(2, 2, {1, 2, 3, 4}).set_requires_grad(true);
    auto t = Tensor::from(2, 2, {0, 2, 5, 1});
    mse_loss(p, t).backward();
    CHECK_NEAR(p.grad_at(0, 0), 0.5f);
    CHECK_NEAR(p.grad_at(0, 1), 0.0f);
    CHECK_NEAR(p.grad_at(1, 0), -1.0f);
    CHECK_NEAR(p.grad_at(1, 1), 1.5f);
}

TEST(mse_rejects_mismatched_shapes) {
    auto p = Tensor::from(2, 3, {1, 2, 3, 4, 5, 6});
    auto t = Tensor::from(3, 2, {1, 2, 3, 4, 5, 6});
    bool threw = false;
    try {
        mse_loss(p, t);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

TEST(gc_mse) {
    const auto target = rand_tensor(3, 4, 41);
    CHECK_GRAD(gradcheck(
        [target](const Inputs &in) { return mse_loss(in[0], target); },
        {rand_tensor(3, 4, 40)}));
}

// --- M4.7 softmax_cross_entropy -------------------------------------------

// row 0: log(e^1 + e^2 + e^3) - 3 = 0.4076060
// row 1: log(3 e^1) - 1 = log 3 = 1.0986123
// mean of the two rows
TEST(ce_value) {
    auto logits = Tensor::from(2, 3, {1, 2, 3, 1, 1, 1});
    auto loss = softmax_cross_entropy(logits, {2, 0});
    CHECK(loss.rows() == 1);
    CHECK(loss.cols() == 1);
    CHECK_NEAR(loss.at(0, 0), 0.7531091f);
}

TEST(ce_grad) {
    auto logits =
        Tensor::from(2, 3, {1, 2, 3, 1, 1, 1}).set_requires_grad(true);
    softmax_cross_entropy(logits, {2, 0}).backward();
    CHECK_NEAR(logits.grad_at(0, 0), 0.0450153f);
    CHECK_NEAR(logits.grad_at(0, 1), 0.1223642f);
    CHECK_NEAR(logits.grad_at(0, 2), -0.1673795f);
    CHECK_NEAR(logits.grad_at(1, 0), -0.3333333f);
    CHECK_NEAR(logits.grad_at(1, 1), 0.1666667f);
    CHECK_NEAR(logits.grad_at(1, 2), 0.1666667f);
}

// logit 1000: e^1000 is inf in float, so without the row max subtracted
// the loss is inf or nan. Row 0 picks the big logit (loss 0), row 1 picks
// one 1000 below it (loss 1000), so the mean is 500.
TEST(ce_large_logits) {
    auto logits = Tensor::from(2, 3, {1000, 0, -1000, 1000, 0, -1000})
                      .set_requires_grad(true);
    auto loss = softmax_cross_entropy(logits, {0, 1});
    CHECK(std::isfinite(loss.at(0, 0)));
    CHECK_NEAR(loss.at(0, 0), 500.0f);
    loss.backward();
    for (float g : logits.impl()->grad) {
        CHECK(std::isfinite(g));
    }
    CHECK_NEAR(logits.grad_at(0, 0), 0.0f);
    CHECK_NEAR(logits.grad_at(1, 0), 0.5f);
    CHECK_NEAR(logits.grad_at(1, 1), -0.5f);
    CHECK_NEAR(logits.grad_at(1, 2), 0.0f);
}

TEST(ce_rejects_wrong_label_count) {
    auto logits = Tensor::from(2, 3, {1, 2, 3, 1, 1, 1});
    bool threw = false;
    try {
        softmax_cross_entropy(logits, {0, 1, 2});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// 3 classes means labels 0, 1, 2; label 3 would read past the row
TEST(ce_rejects_label_out_of_range) {
    auto logits = Tensor::from(2, 3, {1, 2, 3, 1, 1, 1});
    bool threw = false;
    try {
        softmax_cross_entropy(logits, {0, 3});
    } catch (const std::out_of_range &) {
        threw = true;
    }
    CHECK(threw);
}

TEST(gc_softmax_cross_entropy) {
    const std::vector<std::size_t> labels = {0, 2, 1, 2};
    CHECK_GRAD(gradcheck(
        [labels](const Inputs &in) {
            return softmax_cross_entropy(in[0], labels);
        },
        {rand_tensor(4, 3, 50)}));
}

// --- M4.8 full 2-layer MLP -------------------------------------------------

// Linear -> ReLU -> Linear -> loss, checked against every parameter.
// The weights are fixed here instead of left to Linear's init, so every
// relu input stays at least 0.38 from 0; a step of h moves it 0.02 at most.
TEST(gc_mlp) {
    std::mt19937 rng(60);
    auto l1 = std::make_shared<nabla::Linear>(3, 4, rng);
    auto l2 = std::make_shared<nabla::Linear>(4, 2, rng);
    l1->weight.impl()->data = rand_tensor(3, 4, 61).impl()->data;
    l1->bias.impl()->data = rand_tensor(1, 4, 62).impl()->data;
    l2->weight.impl()->data = rand_tensor(4, 2, 63).impl()->data;
    l2->bias.impl()->data = rand_tensor(1, 2, 64).impl()->data;
    nabla::Sequential model{l1, std::make_shared<nabla::ReLU>(), l2};

    const auto x = rand_tensor(5, 3, 70);
    const std::vector<std::size_t> labels = {0, 1, 1, 0, 1};
    auto r = gradcheck(
        [&](const Inputs &) {
            return softmax_cross_entropy(model.forward(x), labels);
        },
        model.parameters());
    CHECK(r.checked == 26);
    CHECK_GRAD(r);
}
