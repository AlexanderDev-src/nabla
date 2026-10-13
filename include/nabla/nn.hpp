#pragma once
#include "nabla/ops.hpp"
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <random>
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
