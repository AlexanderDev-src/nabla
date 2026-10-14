#include "nabla/nabla.hpp"
#include <cstdio>
#include <memory>
#include <random>

using namespace nabla;

int main() {
    std::mt19937 rng(0);
    Sequential model{
        std::make_shared<Linear>(2, 8, rng), std::make_shared<Tanh>(),
        std::make_shared<Linear>(8, 1, rng), std::make_shared<Sigmoid>()};
    const Tensor x = Tensor::from(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    const Tensor y = Tensor::from(4, 1, {0, 1, 1, 0});
    const float lr = 0.5f;

    for (int step = 0; step <= 2000; ++step) {
        Tensor loss = mse_loss(model.forward(x), y);
        loss.backward();
        for (auto &p : model.parameters()) {
            auto &data = p.impl()->data;
            auto &grad = p.impl()->grad;
            for (std::size_t k = 0; k < data.size(); ++k) {
                data[k] -= lr * grad[k];
                grad[k] = 0;
            }
        }
        if (step % 250 == 0) {
            std::printf("step %4d  loss %.6f\n", step, loss.at(0, 0));
        }
    }
    Tensor out = model.forward(x);
    for (std::size_t i = 0; i < 4; ++i) {
        std::printf("%g xor %g = %.4f\n", x.at(i, 0), x.at(i, 1), out.at(i, 0));
    }
}
