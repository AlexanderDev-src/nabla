#pragma once
#include "nabla/ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nabla {

inline Tensor mse_loss(const Tensor &pred, const Tensor &target) {
    if (pred.rows() != target.rows() || pred.cols() != target.cols()) {
        throw std::invalid_argument("mse_loss: shape mismatch " +
                                    shape_str(pred) + " vs " +
                                    shape_str(target));
    }

    auto out = make_node(1, 1, {pred});

    float total = 0.0f;
    for (std::size_t i = 0; i < pred.rows(); ++i) {
        for (std::size_t j = 0; j < pred.cols(); j++) {
            float d = pred.at(i, j) - target.at(i, j);
            total += pow(d, 2);
        }
    }
    out.at(0, 0) = total / (pred.rows() * pred.cols());

    if (out.requires_grad()) {
        auto pi = pred.impl();
        auto ti = target.impl();
        out.impl()->backward_fn = [pi, ti](const std::vector<float> &g) {
            for (std::size_t k = 0; k < pi->data.size(); ++k) {
                float d = pi->data[k] - ti->data[k];
                pi->grad[k] += g[0] * 2.0f * d / pi->data.size();
            }
        };
    }
    return out;
}

inline Tensor softmax_cross_entropy(const Tensor &logits,
                                    const std::vector<std::size_t> &labels) {
    const std::size_t B = logits.rows(), C = logits.cols();
    if (B != labels.size()) {
        throw std::invalid_argument(
            "softmax_cross_entropy: got " + std::to_string(labels.size()) +
            " labels for " + std::to_string(B) + " rows");
    }
    for (std::size_t label : labels) {
        if (label >= C) {
            throw std::out_of_range("softmax_cross_entropy: label " +
                                    std::to_string(label) + " with " +
                                    std::to_string(C) + " classes");
        }
    }

    auto out = make_node(1, 1, {logits});

    std::vector<float> probs(B * C); // softmax of every row, for backward
    float total = 0.0f;
    for (std::size_t i = 0; i < B; ++i) {
        float m = logits.at(i, 0);
        for (std::size_t j = 1; j < C; ++j) {
            m = std::max(m, logits.at(i, j));
        }
        float s = 0.0f;
        for (std::size_t j = 0; j < C; ++j) {
            probs[i * C + j] = exp(logits.at(i, j) - m);
            s += probs[i * C + j];
        }
        for (std::size_t j = 0; j < C; ++j) {
            probs[i * C + j] /= s;
        }
        total += std::log(s) + m - logits.at(i, labels[i]);
    }
    out.at(0, 0) = total / static_cast<float>(B);

    if (out.requires_grad()) {
        auto li = logits.impl();
        out.impl()->backward_fn = [li, probs, labels, B,
                                   C](const std::vector<float> &g) {
            for (std::size_t i = 0; i < B; ++i) {
                for (std::size_t j = 0; j < C; ++j) {
                    float y = (j == labels[i]) ? 1.0f : 0.0f;
                    li->grad[i * C + j] +=
                        g[0] * (probs[i * C + j] - y) / static_cast<float>(B);
                }
            }
        };
    }
    return out;
}

} // namespace nabla
