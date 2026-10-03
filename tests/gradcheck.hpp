#pragma once
// Gradient checker: compares what backward() computes against a central
// finite difference, for every element of every input.
#include "nabla/autograd.hpp"
#include "nabla/nabla.hpp"
#include "nabla/ops.hpp"
#include "nabla/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

namespace nabla_test {

struct GradcheckResult {
    double max_error = 0.0;  // worst |fd - analytic| / max(1, |analytic|)
    std::size_t checked = 0; // how many input elements were compared
    bool ok = false;         // checked > 0 and max_error <= tol
};

using GradFn = std::function<nabla::Tensor(const std::vector<nabla::Tensor> &)>;

inline GradcheckResult gradcheck(const GradFn &f,
                                 std::vector<nabla::Tensor> inputs,
                                 double h = 1e-2, double tol = 1e-4,
                                 unsigned seed = 0) {
    GradcheckResult r;
    for (auto &t : inputs) {
        t.set_requires_grad(true);
    }

    nabla::Tensor out = f(inputs);
    std::mt19937 rng(seed);
    nabla::Tensor mask = nabla::Tensor::randn(out.rows(), out.cols(), rng);

    for (auto &t : inputs) {
        std::fill(t.impl()->grad.begin(), t.impl()->grad.end(), 0.0f);
    }

    nabla::Tensor loss = nabla::sum(nabla::mul(f(inputs), mask));
    loss.backward();
    std::vector<std::vector<float>> analytic;
    for (auto &t : inputs) {
        analytic.push_back(t.impl()->grad);
    }

    nabla::NoGradGuard guard;
    auto masked_loss = [&]() {
        nabla::Tensor y = f(inputs);
        double total = 0.0;

        for (std::size_t m = 0; m < y.impl()->data.size(); ++m) {
            const double y_val = y.impl()->data[m];
            const double mask_val = mask.impl()->data[m];
            total += y_val * mask_val;
        }
        return total;
    };

    for (std::size_t n = 0; n < inputs.size(); ++n) {
        std::vector<float> &data = inputs[n].impl()->data;
        for (std::size_t k = 0; k < data.size(); ++k) {
            float &slot = data[k];
            const float v = slot;
            slot = v + static_cast<float>(h);
            const double loss_plus = masked_loss();
            slot = v - static_cast<float>(h);
            const double loss_minus = masked_loss();
            slot = v;

            const double fd = (loss_plus - loss_minus) / (2.0 * h);
            const double a = analytic[n][k];
            const double error =
                std::fabs(fd - a) / std::max(1.0, std::fabs(a));
            r.max_error = std::max(r.max_error, error);
            ++r.checked;
            r.ok = r.checked > 0 && r.max_error <= tol;
        }
    }

    return r;
}

} // namespace nabla_test
