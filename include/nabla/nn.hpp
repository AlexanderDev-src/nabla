#pragma once
// Layers. A Module turns a Tensor into a Tensor and owns its own weights.
#include "nabla/ops.hpp"
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <random>
#include <vector>

namespace nabla {

// virtual: a call through a Module& runs the subclass's version.
// = 0: no body here, so every subclass must write its own.
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

// y = x @ weight + bias, with x [batch, in] and y [batch, out]
class Linear : public Module {
  public:
    Linear(std::size_t in, std::size_t out, std::mt19937 &rng)
        : weight(Tensor::zeros(in, out)), bias(Tensor::zeros(1, out)) {
        weight = Tensor::randn(in, out, rng, 0.0f, std::sqrt(2.0f / in));
        weight.set_requires_grad(true);
        bias.set_requires_grad(true);
    }

    // override: the compiler checks that Module really has this function.
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

// Runs its layers in order: the output of one is the input of the next.
class Sequential : public Module {
  public:
    // Lets you write Sequential model{layer1, layer2, ...}
    Sequential(std::initializer_list<std::shared_ptr<Module>> layers)
        : layers_(layers) {}

    Tensor forward(const Tensor &x) override {
        Tensor h = x;
        // TODO: (M4.5a) for each layer in layers_: h = layer->forward(h)
        return h;
    }

    std::vector<Tensor> parameters() override {
        std::vector<Tensor> all;
        // TODO: (M4.5b) for each layer, push_back every tensor of
        //           layer->parameters() into all
        return all;
    }

    void set_training(bool on) override {
        // TODO: (M4.5c) Module::set_training(on) for the model itself,
        //           then layer->set_training(on) for each layer
    }

  private:
    std::vector<std::shared_ptr<Module>> layers_;
};

} // namespace nabla
