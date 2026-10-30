#pragma once
#include "nabla/ops.hpp"
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace nabla {

class Module {
  public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor &x) = 0;

    virtual std::vector<Tensor> parameters() { return {}; }

    virtual void set_training(bool on) { training_ = on; }

    void train() { set_training(true); }

    void eval() { set_training(false); }

    bool is_training() const { return training_; }

  private:
    bool training_ = true;
};

class Linear : public Module {
  public:
    Linear(std::size_t in, std::size_t out, std::mt19937 &rng)
        : weight(Tensor::zeros(in, out)), bias(Tensor::zeros(1, out)) {
        weight = Tensor::randn(in, out, rng, 0.0f, std::sqrt(2.0f / in));
        weight.set_requires_grad(true);
        bias.set_requires_grad(true);
    }

    Tensor forward(const Tensor &x) override {
        return add(matmul(x, weight), bias);
    }

    std::vector<Tensor> parameters() override { return {weight, bias}; }

    Tensor weight;
    Tensor bias;
};

class ReLU : public Module {
  public:
    Tensor forward(const Tensor &x) override { return relu(x); }
};

class Tanh : public Module {
  public:
    Tensor forward(const Tensor &x) override { return tanh(x); }
};

class Sigmoid : public Module {
  public:
    Tensor forward(const Tensor &x) override { return sigmoid(x); }
};

// Inverted dropout. In training each value is kept with probability 1 - p
// and divided by 1 - p, or else set to 0, so every output keeps the same
// expected value; each call draws a new mask. In eval it returns x as it is,
// and because of the division in training, eval needs no rescaling. The
// layer keeps a reference to rng, so rng must outlive it.
class Dropout : public Module {
  public:
    Dropout(float p, std::mt19937 &rng) : p_(p), rng_(rng) {
        if (!(p >= 0.0f && p < 1.0f)) {
            throw std::invalid_argument("Dropout: p must be in [0, 1)");
        }
    }

    Tensor forward(const Tensor &x) override {
        if (!is_training()) {
            return x;
        }
        std::bernoulli_distribution keep(1.0f - p_);
        const float scale = 1.0f / (1.0f - p_);
        Tensor mask = Tensor::zeros(x.rows(), x.cols());
        for (std::size_t r = 0; r < x.rows(); ++r) {
            for (std::size_t c = 0; c < x.cols(); ++c) {
                mask.at(r, c) = keep(rng_) ? scale : 0.0f;
            }
        }
        return mul(x, mask);
    }

  private:
    float p_;
    std::mt19937 &rng_;
};

class Sequential : public Module {
  public:
    Sequential(std::initializer_list<std::shared_ptr<Module>> layers)
        : layers_(layers) {}

    Tensor forward(const Tensor &x) override {
        Tensor h = x;
        for (const auto &layer : layers_) {
            h = layer->forward(h);
        }
        return h;
    }

    std::vector<Tensor> parameters() override {
        std::vector<Tensor> all;
        for (const auto &layer : layers_) {
            for (const auto &p : layer->parameters()) {
                all.push_back(p);
            }
        }
        return all;
    }

    void set_training(bool on) override {
        Module::set_training(on);
        for (const auto &layer : layers_) {
            layer->set_training(on);
        }
    }

  private:
    std::vector<std::shared_ptr<Module>> layers_;
};

} // namespace nabla
