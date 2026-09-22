#pragma once
#include "nabla/autograd.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace nabla {

inline std::string shape_str(const Tensor &t) {
    return "[" + std::to_string(t.rows()) + "x" + std::to_string(t.cols()) +
           "]";
}

inline Tensor matmul(const Tensor &a, const Tensor &b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matmul: shape mismatch " + shape_str(a) +
                                    " @ " + shape_str(b));
    }

    auto out = make_node(a.rows(), b.cols(), {a, b});

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            float sum = 0;
            for (std::size_t p = 0; p < a.cols(); ++p) {
                sum += a.at(i, p) * b.at(p, j);
            }
            out.at(i, j) = sum;
        }
    }
    return out;
}

inline Tensor add(const Tensor &a, const Tensor &b) {
    const bool same_shape = a.rows() == b.rows() && a.cols() == b.cols();
    const bool row_broadcast = b.rows() == 1 && a.cols() == b.cols();
    if (!same_shape && !row_broadcast) {
        throw std::invalid_argument("add: shape mismatch " + shape_str(a) +
                                    " + " + shape_str(b));
    }

    auto out = make_node(a.rows(), a.cols(), {a, b});
    for (std::size_t i = 0; i < a.rows(); ++i) {
        const std::size_t bi = (b.rows() == 1) ? 0 : i;
        for (std::size_t j = 0; j < a.cols(); ++j) {
            out.at(i, j) = a.at(i, j) + b.at(bi, j);
        }
    }
    return out;
}

} // namespace nabla
