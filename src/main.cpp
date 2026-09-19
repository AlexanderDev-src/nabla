#include "nabla/scalar.hpp"
#include <cstdio>
#include <vector>

using nabla::Scalar;

int main() {
    const std::vector<std::vector<float>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<float> T = {0, 0, 0, 1};

    Scalar w1(0.0f), w2(0.0f), b(0.0f);
    const float lr = 0.05f;

    for (int epoch = 0; epoch < 500; ++epoch) {
        Scalar loss(0.0f);

        for (std::size_t i = 0; i < X.size(); ++i) {
            Scalar x1(X[i][0]), x2(X[i][1]);
            Scalar p = w1 * x1 + w2 * x2 + b;
            Scalar err = p - T[i];
            loss = loss + err * err;
        }
        loss.backward();
        w1.step(lr);
        w2.step(lr);
        b.step(lr);

        if (epoch % 10 == 0) {
            printf("epoch %3d  loss=%.4f  w1=%.4f w2=%.4f b=%.4f\n", epoch,
                   loss.data(), w1.data(), w2.data(), b.data());
        }
    }
    for (std::size_t i = 0; i < X.size(); ++i) {
        const float p = w1.data() * X[i][0] + w2.data() * X[i][1] + b.data();
        printf("(%.0f,%.0f) p=%+.3f -> %d   AND=%.0f\n", X[i][0], X[i][1], p,
               p > 0.5f, T[i]);
    }

    return 0;
}
