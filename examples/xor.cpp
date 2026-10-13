#include "nabla/nabla.hpp"
#include <cstdio>
#include <memory>
#include <random>

using namespace nabla;

int main() {
    std::mt19937 rng(0);
    Sequential model{std::make_shared<Linear>(2, 8, rng), std::make_shared<Tanh>(),
                     std::make_shared<Linear>(8, 1, rng), std::make_shared<Sigmoid>()};
    const Tensor x = Tensor::from(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    const Tensor y = Tensor::from(4, 1, {0, 1, 1, 0});
    const float lr = 0.5f;

    for (int step = 0; step <= 2000; ++step) {
        // TODO: (M4.9a) loss = mse_loss of model.forward(x) against y
        // TODO: (M4.9b) loss.backward()
        // TODO: (M4.9c) for each p in model.parameters(), for each k:
        //           data[k] -= lr * grad[k], then grad[k] = 0, because
        //           leaf grads add up across backward() calls
        // TODO: (M4.9d) every 250 steps, print step and loss
    }
    // TODO: (M4.9e) print model.forward(x) for each of the 4 rows
}
