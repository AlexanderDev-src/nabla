#pragma once
#include "nabla/tensor.hpp"
#include <cmath>
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

// Adam keeps two running averages per element, m of g and v of g * g, and
// divides one by the root of the other, so each element gets its own step
// size. Every step, with t counting steps from 1:
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   data -= lr * m_hat / (sqrt(v_hat) + eps)
// step() never touches grad; clearing it is zero_grad()'s job.
class Adam : public Optimizer {
  public:
    Adam(std::vector<Tensor> params, float lr = 1e-3f, float beta1 = 0.9f,
         float beta2 = 0.999f, float eps = 1e-8f)
        : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
          eps_(eps) {
        for (auto const &p : params_) {
            m_.push_back(std::vector<float>(p.impl()->data.size(), 0.0f));
            v_.push_back(std::vector<float>(p.impl()->data.size(), 0.0f));
        }
    }

    void step() override {
        t_++;
        auto const corrections1 = 1 - std::pow(beta1_, t_);
        auto const corrections2 = 1 - std::pow(beta2_, t_);
        for (std::size_t i = 0; i < params_.size(); ++i) {
            auto &data = params_[i].impl()->data;
            const auto &grad = params_[i].impl()->grad;
            auto &m = m_[i];
            auto &v = v_[i];
            for (std::size_t k = 0; k < data.size(); ++k) {
                const float g = grad[k];

                m[k] = beta1_ * m[k] + (1.0f - beta1_) * g;
                v[k] = beta2_ * v[k] + (1.0f - beta2_) * g * g;

                const float m_hat = m[k] / corrections1;
                const float v_hat = v[k] / corrections2;

                data[k] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }

  private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    int t_ = 0;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
};

} // namespace nabla
