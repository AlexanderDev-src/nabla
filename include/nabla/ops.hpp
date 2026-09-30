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

    if (out.requires_grad()) {
        auto ai = a.impl();
        auto bimpl = b.impl();
        out.impl()->backward_fn = [ai, bimpl](const std::vector<float> &g) {
            // A = m * k , B = k * n, g = m * n -> matrix
            const std::size_t m = ai->rows, k = ai->cols, n = bimpl->cols;
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    const float g_ij = g[i * n + j];
                    for (std::size_t p = 0; p < k; ++p) {
                        if (ai->requires_grad) {
                            ai->grad[i * k + p] +=
                                g_ij * bimpl->data[p * n + j];
                        }
                        if (bimpl->requires_grad) {
                            bimpl->grad[p * n + j] +=
                                ai->data[i * k + p] * g_ij;
                        }
                    }
                }
            }
        };
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
    if (out.requires_grad()) {
        auto ai = a.impl();
        auto bimpl = b.impl();
        out.impl()->backward_fn = [ai, bimpl](const std::vector<float> &g) {
            const std::size_t rows = ai->rows, cols = ai->cols;
            for (std::size_t i = 0; i < rows; ++i) {
                std::size_t bi = (bimpl->rows == 1) ? 0 : i;
                for (std::size_t j = 0; j < cols; ++j) {
                    if (ai->requires_grad) {
                        ai->grad[i * cols + j] += g[i * cols + j];
                    }
                    if (bimpl->requires_grad) {
                        bimpl->grad[bi * bimpl->cols + j] += g[i * cols + j];
                    }
                }
            }
        };
    }
    return out;
}

inline Tensor relu(const Tensor &x) {
    auto out = make_node(x.rows(), x.cols(), {x});

    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j) {
            float val = x.at(i, j);
            out.at(i, j) = (val > 0.0f) ? val : 0.0f;
        }
    }

    if (out.requires_grad()) {
        auto xi = x.impl();
        out.impl()->backward_fn = [xi](const std::vector<float> &g) {
            // relu works element by element, so one flat index k is the
            // same position in x's data, x's grad and g
            if (!xi->requires_grad)
                return;
            for (std::size_t k = 0; k < xi->data.size(); ++k) {
                if (xi->data[k] > 0.0f) {
                    xi->grad[k] += g[k];
                }
            }
        };
    }
    return out;
}

// Element-wise product. Both inputs must have exactly the same shape; there
// is no broadcasting here.
inline Tensor mul(const Tensor &a, const Tensor &b) {
    // TODO: (1) throw std::invalid_argument when the shapes differ, with a
    //           message built from shape_str like matmul and add do

    auto out = make_node(a.rows(), a.cols(), {a, b});

    // TODO: (2) forward: out at (i, j) is a at (i, j) times b at (i, j)

    if (out.requires_grad()) {
        auto ai = a.impl();
        auto bimpl = b.impl();
        out.impl()->backward_fn = [ai, bimpl](const std::vector<float> &g) {
            for (std::size_t k = 0; k < g.size(); ++k) {
                // TODO: (3) for each input that needs grad, accumulate
                //           g at k times the OTHER input's data at k
            }
        };
    }
    return out;
}

} // namespace nabla
