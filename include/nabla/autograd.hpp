#pragma once
#include "nabla/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <unordered_set>
#include <utility>

namespace nabla {

inline Tensor make_node(std::size_t rows, std::size_t cols,
                        std::initializer_list<Tensor> parents) {
    auto impl = std::make_shared<TensorImpl>(rows, cols);

    bool any_requires_grad = false;

    for (const auto &p : parents) {
        if (p.requires_grad()) {
            any_requires_grad = true;
            break;
        }
    }

    if (any_requires_grad) {
        impl->requires_grad = true;
        for (const auto &p : parents) {
            impl->parents.push_back(p.impl());
        }
    }

    return Tensor::from_impl(impl);
}
inline void Tensor::backward() const {
    if (!impl_->requires_grad) {
        throw std::logic_error("backward: tensor does not require grad");
    }

    std::vector<TensorImpl *> order;
    std::unordered_set<const TensorImpl *> visited;
    std::vector<std::pair<TensorImpl *, std::size_t>> stack;

    stack.emplace_back(impl_.get(), 0);
    visited.insert(impl_.get());
    while (!stack.empty()) {
        auto [n, i] = stack.back();
        if (i < n->parents.size()) {
            stack.back().second = i + 1;
            TensorImpl *next = n->parents[i].get();
            if (visited.insert(next).second) {
                stack.emplace_back(next, 0);
            }
            continue;
        }
        order.push_back(n);
        stack.pop_back();
    }

    for (TensorImpl *node : order) {
        if (!node->parents.empty()) {
            std::fill(node->grad.begin(), node->grad.end(), 0.0f);
        }
    }

    std::fill(impl_->grad.begin(), impl_->grad.end(), 1.0f);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        TensorImpl *node = *it;
        if (node->backward_fn) {
            node->backward_fn(node->grad);
        }
    }
}

} // namespace nabla
