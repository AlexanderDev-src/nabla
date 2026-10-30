#include "nabla/nabla.hpp"
#include <chrono>
#include <cmath> // std::isfinite
#include <cstdio>
#include <exception> // std::exception
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace nabla;

int main(int argc, char **argv) {
    std::size_t epochs = 15;
    std::size_t batch_size = 64;
    float lr = 1e-3f;
    std::string save_path;

    try {
        for (int i = 1; i < argc; i += 2) {
            const std::string_view flag = argv[i];

            if (flag != "--epochs" && flag != "--batch" && flag != "--lr" &&
                flag != "--save") {
                std::fprintf(stderr, "Unknown flag: %s\n", argv[i]);
                return 1;
            }

            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", argv[i]);
                return 1;
            }

            const std::string value = argv[i + 1];
            std::size_t parsed = 0;

            if (flag == "--save") {
                if (value.empty()) {
                    throw std::invalid_argument(
                        "--save path must not be empty");
                }
                save_path = value;
                continue;
            }

            if (flag == "--lr") {
                lr = std::stof(value, &parsed);

                if (parsed != value.size() || !std::isfinite(lr) ||
                    lr <= 0.0f) {
                    throw std::invalid_argument(
                        "--lr must be a positive finite number");
                }
            } else {
                if (value.empty() || value.find_first_not_of("0123456789") !=
                                         std::string::npos) {
                    throw std::invalid_argument(
                        "Epochs and batch size must be positive integers");
                }

                const auto number = std::stoul(value, &parsed);

                if (parsed != value.size() || number == 0) {
                    throw std::invalid_argument(
                        "Epochs and batch size must be positive integers");
                }

                if (flag == "--epochs") {
                    epochs = number;
                } else {
                    batch_size = number;
                }
            }
        }

        const auto train_images = load_idx("data/train-images-idx3-ubyte");
        const auto train_labels = load_idx("data/train-labels-idx1-ubyte");
        const auto test_images = load_idx("data/t10k-images-idx3-ubyte");
        const auto test_labels = load_idx("data/t10k-labels-idx1-ubyte");

        const auto train_x = normalize_mnist(train_images.data);
        const auto test_x = normalize_mnist(test_images.data);

        std::mt19937 rng(0);

        Sequential model{std::make_shared<Linear>(784, 256, rng),
                         std::make_shared<ReLU>(),
                         std::make_shared<Dropout>(0.2f, rng),
                         std::make_shared<Linear>(256, 128, rng),
                         std::make_shared<ReLU>(),
                         std::make_shared<Dropout>(0.2f, rng),
                         std::make_shared<Linear>(128, 10, rng)};

        const auto params = model.parameters();
        Adam opt(params, lr);

        const std::size_t n = train_labels.data.size();

        if (n == 0) {
            throw std::runtime_error("Training dataset is empty");
        }

        const std::size_t steps = n / batch_size + (n % batch_size != 0);
        const std::size_t total = epochs * steps;
        std::size_t t = 0;

        model.train();

        for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
            const auto start = std::chrono::steady_clock::now();
            float loss_sum = 0.0f;

            for (const auto &indices : shuffled_batches(n, batch_size, rng)) {
                const auto batch =
                    make_batch(train_x, train_labels.data, 784, indices);

                opt.set_lr(cosine_lr(t, steps / 2, total, lr));
                ++t;

                opt.zero_grad();

                const auto logits = model.forward(batch.x);
                Tensor loss = softmax_cross_entropy(logits, batch.y);

                loss.backward();
                clip_grad_norm(params, 1.0f);
                opt.step();

                loss_sum += loss.at(0, 0);
            }

            const double secs = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();

            const float accuracy =
                evaluate(model, test_x, test_labels.data, 784, 1000);

            std::printf("epoch %zu  loss %.4f  accuracy %.2f%%  time %.2fs\n",
                        epoch, loss_sum / steps, accuracy * 100.0f, secs);

            if (!save_path.empty()) {
                save_checkpoint(save_path, model);
            }
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }

    return 0;
}
