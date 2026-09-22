#pragma once
#include "nabla/tensor.hpp"

#include <cstddef>
#include <initializer_list>
#include <memory>

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
        return Tensor::from_impl(impl);
    }

    return Tensor::from_impl(impl);
}

} // namespace nabla
