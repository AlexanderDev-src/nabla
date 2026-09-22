#pragma once
#include <cstddef>
#include <functional>
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
    bool requires_grad = false;
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void(const std::vector<float> &upstream)> backward_fn;

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
        auto impl = std::make_shared<TensorImpl>(impl_->rows, impl_->cols);
        impl->data = impl_->data;
        return Tensor(impl);
    }
    bool requires_grad() const { return impl_->requires_grad; }
    Tensor &set_requires_grad(bool on) {
        impl_->requires_grad = on;
        return *this;
    }
    float grad_at(std::size_t r, std::size_t c) const {
        return impl_->grad[r * impl_->cols + c];
    }

  private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
    std::shared_ptr<TensorImpl> impl_;
};
} // namespace nabla
