#pragma once
#include <bits/stdc++.h>

namespace nable {

struct TensorImpl {
    std::size_t rows, cols;
    std::vector<float> data;
    std::vector<float> grid;

    TensorImpl(std::size_t rows, std::size_t cols)
        : rows(rows), cols(cols), data(rows * cols, 0.0f),
          grid(rows * cols, 0.0f) {} // Constructor
};

class Tensor {
  public:
    static Tensor zeros(std::size_t rows, std::size_t cols) {
        return Tensor(std::make_shared<TensorImpl>(rows, cols));
    }
    std::size_t rows() const { return impl_->rows; }
    std::size_t cols() const { return impl_->cols; }

  private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
    std::shared_ptr<TensorImpl> impl_;
};
} // namespace nable
