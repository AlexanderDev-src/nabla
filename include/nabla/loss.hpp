#pragma once
// Fused losses: forward and backward written as one op instead of built
// from smaller ops, so the backward rule is short and stays exact.
#include "nabla/ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nabla {

// Mean of (pred - target)^2 over every element
inline Tensor mse_loss(const Tensor &pred, const Tensor &target) {
    // TODO: (M4.6a) throw std::invalid_argument when the shapes differ

    // target is data, not a parent, so it never gets a grad
    auto out = make_node(1, 1, {pred});

    // TODO: (M4.6b) forward: out at (0, 0) = sum of (p - t)^2 / N, where
    //           N is the number of elements

    if (out.requires_grad()) {
        auto pi = pred.impl();
        auto ti = target.impl();
        out.impl()->backward_fn = [pi, ti](const std::vector<float> &g) {
            for (std::size_t k = 0; k < pi->data.size(); ++k) {
                // TODO: (M4.6c) pi->grad at k += g[0] * 2 (p - t) / N
            }
        };
    }
    return out;
}

// Mean over the batch of -log(softmax(row i) at labels[i]).
// logits is [B, C]; labels holds one class index per row.
inline Tensor softmax_cross_entropy(const Tensor &logits,
                                    const std::vector<std::size_t> &labels) {
    const std::size_t B = logits.rows(), C = logits.cols();
    // TODO: (M4.7a) throw std::invalid_argument when labels.size() != B,
    //           and std::out_of_range when any label >= C

    auto out = make_node(1, 1, {logits});

    std::vector<float> probs(B * C); // softmax of every row, for backward
    float total = 0.0f;
    for (std::size_t i = 0; i < B; ++i) {
        // TODO: (M4.7b) m = largest logit in row i
        // TODO: (M4.7c) probs at (i, j) = exp(z - m); add them up in s
        // TODO: (M4.7d) divide row i of probs by s
        // TODO: (M4.7e) total += loss of row i, which is -log(probs at
        //           label). Rewrite it with log(s) and m so it stays
        //           finite at logit 1000.
    }
    out.at(0, 0) = total / static_cast<float>(B);

    if (out.requires_grad()) {
        auto li = logits.impl();
        out.impl()->backward_fn = [li, probs, labels, B,
                                   C](const std::vector<float> &g) {
            // TODO: (M4.7f) li->grad at (i, j) += g[0] * dL/dz at (i, j).
            //           Derive dL/dz on paper first; ce_grad has the
            //           numbers to check against.
        };
    }
    return out;
}

} // namespace nabla
