#include "nabla/value.hpp"
#include <cstdio>

using nabla::Value;

int main() {
    const float X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const float T[4] = {0, 0, 0, 1};

    Value w1(0.0f), w2(0.0f), b(0.0f);
    Value loss(0.0f);
    for (int i = 0; i < 4; ++i) {
        Value x1(X[i][0]), x2(X[i][1]);
        Value p = w1 * x1 + w2 * x2 + b;
        Value err = p - T[i];
        loss = loss + err * err;
    }
    loss.backward();
    printf("loss=%.4f  w1.grad=%.4f w2.grad=%.4f b.grad=%.4f\n", loss.data(),
           w1.grad(), w2.grad(), b.grad());
}
