#pragma once
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nabla {

struct TensorImpl {
    std::size_t rows, cols;
    std::vector<float> data;
    std::vector<float> grad;

    TensorImpl(std::size_t rows, std::size_t cols)
        : rows(rows), cols(cols), data(rows * cols, 0.0f),
          grad(rows * cols, 0.0f) {} // Constructor
};
// handle-semantics : copy a tensor copies the shared_ptr not data
// That's why two tensor can name the same
//
// Chosen for identity, not for speed. Once the autograd graph arrives in
// M2, a single tensor may appear at several places in the graph, and every
// one of those places has to accumulate into the same `grad` buffer.
// Deep-copying on assignment would scatter the gradient across copies that
// nothing ever reads.
//
// The cost is aliasing. Copying is shallow, so after `Tensor b = a;` a
// write through `b` shows up in `a` as well, which surprises anyone
// expecting the deep copy that std::vector would give.
//
// To keep that contained, no operation mutates its inputs: matmul, add and
// the rest always return a fresh Tensor, so sharing only shows up when a
// caller writes through at() on purpose. A caller who wants independent
// storage asks for it explicitly with clone().

class Tensor {
  public:
    static Tensor zeros(std::size_t rows, std::size_t cols) {
        return Tensor(std::make_shared<TensorImpl>(rows, cols));
    }

    std::size_t rows() const { return impl_->rows; }
    std::size_t cols() const { return impl_->cols; }

    float &at(std::size_t r, std::size_t c) {
        return impl_->data[r * impl_->cols + c];
    }
    float at(std::size_t r, std::size_t c) const {
        return impl_->data[r * impl_->cols + c];
    }

    static Tensor full(std::size_t rows, std::size_t cols, float val) {
        Tensor t = zeros(rows, cols);
        t.impl_->data.assign(rows * cols, val);
        return t;
    }

    static Tensor from(std::size_t rows, std::size_t cols,
                       std::initializer_list<float> data) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument(
                "Tensor::from: got " + std::to_string(data.size()) +
                " values, expected " + std::to_string(rows * cols) + " (" +
                std::to_string(rows) + "x" + std::to_string(cols) + ")");
        }

        auto impl = std::make_shared<TensorImpl>(rows, cols);
        impl->data.assign(data.begin(), data.end());
        return Tensor(impl);
    }

    static Tensor randn(std::size_t rows, std::size_t cols, std::mt19937 &rng,
                        float mean = 0.0f, float stddev = 1.0f) {
        auto impl = std::make_shared<TensorImpl>(rows, cols);
        std::normal_distribution<float> dist(mean, stddev);

        for (float &val : impl->data) {
            val = dist(rng);
        }
        return Tensor(impl);
    }
    Tensor clone() const {
        return Tensor(std::make_shared<TensorImpl>(*impl_));
    }

  private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
    std::shared_ptr<TensorImpl> impl_;
};
inline std::string shape_str(const Tensor &t) {
    return "[" + std::to_string(t.rows()) + "x" + std::to_string(t.cols()) +
           "]";
}

inline Tensor matmul(const Tensor &a, const Tensor &b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matmul: shape mismatch " + shape_str(a) +
                                    " @ " + shape_str(b));
    }

    auto out = Tensor::zeros(a.rows(), b.cols());

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

    auto out = Tensor::zeros(a.rows(), a.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        const std::size_t bi = (b.rows() == 1) ? 0 : i;
        for (std::size_t j = 0; j < a.cols(); ++j) {
            out.at(i, j) = a.at(i, j) + b.at(bi, j);
        }
    }
    return out;
}
} // namespace nabla
