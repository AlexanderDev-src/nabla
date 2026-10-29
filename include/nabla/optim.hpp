#pragma once
#include "nabla/tensor.hpp"
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>
#include <vector>

namespace nabla {

// Base for every optimiser. It holds a flat list of parameters and the
// learning rate, and knows nothing about layers, so it never includes nn.hpp.
// lr lives here rather than in each optimiser so a schedule can change it
// through set_lr() without knowing which optimiser it is.
class Optimizer {
  public:
    explicit Optimizer(std::vector<Tensor> params, float lr = 0.0f)
        : params_(std::move(params)), lr_(lr) {}

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

    float lr() const { return lr_; }

    // the next step() uses the new lr
    void set_lr(float lr) { lr_ = lr; }

  protected:
    std::vector<Tensor> params_;
    float lr_;
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
        : Optimizer(std::move(params), lr), momentum_(momentum),
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
        : Optimizer(std::move(params), lr), beta1_(beta1), beta2_(beta2),
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
    float beta1_;
    float beta2_;
    float eps_;
    int t_ = 0;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
};

// Treats every grad of every parameter as one long vector and scales them
// all by the same factor, so that vector's L2 norm is at most max_norm.
// Grads already within max_norm stay as they are. Returns the norm from
// before clipping, so a training loop can log how large the grads were.
inline float clip_grad_norm(const std::vector<Tensor> &params, float max_norm) {
    float sum_sq = 0.0f;

    for (const auto &p : params) {
        for (const float g : p.impl()->grad) {
            sum_sq += g * g;
        }
    }

    const float total = std::sqrt(sum_sq);
    if (total > max_norm) {
        for (const auto &p : params) {
            for (float &g : p.impl()->grad) {
                g *= (max_norm / total);
            }
        }
    }
    return total;
}

// Learning rate for step t (counting from 0) of a run of `total` steps:
// a straight line from 0 up to max_lr over the first `warmup` steps, then
// half a cosine wave from max_lr down to min_lr, reached at t = total.
// Past total it stays at min_lr. Use it before each step:
//   opt.set_lr(cosine_lr(t, warmup, total, max_lr));
inline float cosine_lr(std::size_t t, std::size_t warmup, std::size_t total,
                       float max_lr, float min_lr = 0.0f) {
    if (t < warmup) {
        return max_lr * static_cast<float>(t) / static_cast<float>(warmup);
    }

    if (t >= total) {
        return min_lr;
    }

    const double p =
        static_cast<double>(t - warmup) / static_cast<double>(total - warmup);
    constexpr double pi = 3.14159265358979323846;

    return static_cast<float>(min_lr + 0.5 * (max_lr - min_lr) *
                                           (1.0 + std::cos(pi * p)));
}

} // namespace nabla
