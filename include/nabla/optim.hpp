#pragma once
#include "nabla/tensor.hpp"
#include <utility>
#include <vector>

namespace nabla {

// Base for every optimiser. It holds a flat list of parameters and knows
// nothing about layers, so it never includes nn.hpp.
class Optimizer {
  public:
    explicit Optimizer(std::vector<Tensor> params)
        : params_(std::move(params)) {}

    virtual ~Optimizer() = default;

    virtual void step() = 0;

    void zero_grad() {
        for (auto &p : params_) {
            auto &grad = p.impl()->grad;
            for (std::size_t k = 0; k < grad.size(); ++k) {
                grad[k] = 0;
            }
        }
    }

    const std::vector<Tensor> &params() const { return params_; }

  protected:
    std::vector<Tensor> params_;
};

// Plain SGD when momentum and weight_decay are 0. Per element, every step:
//   g = grad + weight_decay * data
//   v = momentum * v + g
//   data -= lr * v
// step() never touches grad; clearing it is zero_grad()'s job.
class SGD : public Optimizer {
  public:
    SGD(std::vector<Tensor> params, float lr, float momentum = 0.0f,
        float weight_decay = 0.0f)
        : Optimizer(std::move(params)), lr_(lr), momentum_(momentum),
          weight_decay_(weight_decay) {
        for (auto &p : params_) {
            velocity_.push_back(
                std::vector<float>(p.impl()->data.size(), 0.0f));
        }
    }

    void step() override {
        for (std::size_t i = 0; i < params_.size(); ++i) {
            auto &data = params_[i].impl()->data;
            auto &grad = params_[i].impl()->grad;
            auto &v = velocity_[i];
            for (std::size_t k = 0; k < data.size(); ++k) {
                float g = grad[k] + weight_decay_ * data[k];
                v[k] = momentum_ * v[k] + g;
                data[k] -= lr_ * v[k];
            }
        }
    }

  private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<std::vector<float>> velocity_;
};

} // namespace nabla
