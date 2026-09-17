<h1 align="center">nabla</h1>

<p align="center">
  A tiny reverse-mode autodiff engine and neural network library in C++23.<br>
  Header-only, zero dependencies.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++23">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT"></a>
</p>

---

`nabla` builds a computation graph as your code runs, then differentiates it
automatically. You write the forward pass; the backward pass comes for free.

```cpp
#include "nabla/nabla.hpp"

int main() {
    std::mt19937 rng(42);

    auto model = nabla::nn::Sequential{
        nabla::nn::make<nabla::nn::Linear>(2, 8, rng),
        nabla::nn::make<nabla::nn::Tanh>(),
        nabla::nn::make<nabla::nn::Linear>(8, 1, rng),
    };

    auto x = nabla::Tensor::from(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    auto y = nabla::Tensor::from(4, 1, {0, 1, 1, 0});   // XOR

    nabla::optim::Adam opt(model.parameters(), {.lr = 0.05f});

    for (int step = 0; step < 500; ++step) {
        opt.zero_grad();
        auto loss = nabla::mse_loss(model(x), y);
        loss.backward();
        opt.step();
    }
}
```

That's the whole training loop. No layer ever implements `backward()` — the graph
knows how to reverse itself.

> XOR is the smallest honest test of an autodiff engine: a single linear layer
> provably cannot solve it, so if this converges, the hidden layer and its
> gradients are genuinely working.

---

## Table of contents

- [What this is](#what-this-is)
- [Build](#build)
- [How reverse-mode autodiff works](#how-reverse-mode-autodiff-works)
- [The fused softmax + cross-entropy gradient](#the-fused-softmax--cross-entropy-gradient)
- [API tour](#api-tour)
- [Results](#results)
- [Benchmark](#benchmark)
- [Design notes](#design-notes)
- [Roadmap](#roadmap)

---

## What this is

A learning-grade automatic differentiation engine, written to understand how
PyTorch's `autograd` actually works — and then made fast enough and complete
enough to train a real model on a real dataset.

**What it has**

- Reverse-mode automatic differentiation over a dynamically built graph
- `matmul`, broadcasting `add`, `relu`, `tanh`, `sigmoid`, `dropout`, reductions
- Fused, numerically stable `softmax_cross_entropy`
- `Linear`, `Sequential`, `Dropout`, He initialisation
- SGD with momentum and Nesterov, Adam and AdamW
- Gradient clipping, cosine LR schedule with warmup
- Checkpoint save/load
- MNIST loader that parses the original IDX files
- Every operation verified against finite differences in the test suite

**What it deliberately doesn't have**

- **2-D tensors only.** An MLP never needs more, and skipping N-d strides keeps
  every backward pass short enough to read in one sitting.
- **`float` only.** No dtype dispatch, no templates over scalar types.
- **CPU only.** OpenMP on the hot loops, no CUDA.
- **No BLAS.** The matmul is a hand-written loop. It is slower than OpenBLAS and
  that is the intended trade — the point is that you can read it.

These are scope decisions, not missing features. A library that does one thing
and can be understood end to end is more useful for learning than one that does
everything opaquely.

---

## Build

Header-only. Copy `include/nabla/` into your project and add it to your include
path, or use CMake:

```bash
git clone https://github.com/AlexanderDev-src/nabla.git
cd nabla
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Requires a C++17 compiler. Tested on GCC 11+, Clang 14+, MSVC 19.3+.

With CMake in your own project:

```cmake
add_subdirectory(nabla)
target_link_libraries(your_target PRIVATE nabla)
```

Run the examples:

```bash
./build/xor
./scripts/get_mnist.sh
./build/mnist_mlp --data data --epochs 15
```

---

## How reverse-mode autodiff works

This is the part worth understanding, so here it is in full.

### The graph

Every operation returns a new tensor that remembers two things: which tensors it
came from, and how to push gradient back into them.

```
 forward ──────────────────────────────────────────────────────►

   x ──┐
       ├──[matmul]──► h₁ ──┐
   W ──┘                   ├──[add]──► h₂ ──[relu]──► h₃ ──► ... ──► loss
                     b ────┘

 ◄────────────────────────────────────────────────────── backward
```

Each node holds:

```cpp
std::vector<TensorPtr>  parents;      // where this value came from
std::function<void()>   backward_fn;  // reads my .grad, adds into parents' .grad
```

Nothing is precomputed. The graph is whatever your control flow happened to
execute — `if`, loops, recursion all work, because the graph is a record of what
ran, not a description of what might run.

### The backward pass

Calling `.backward()` on a scalar does three things:

1. **Topologically sort** the graph reachable from that scalar.
2. **Seed** its gradient with 1.0 — because ∂L/∂L = 1.
3. **Walk the sorted list in reverse**, calling each node's `backward_fn`.

The sort is what makes this correct. If a tensor feeds two different operations,
its gradient is the _sum_ of both paths, and it must not propagate anything until
both contributions have landed. Reverse topological order guarantees exactly
that: a node is visited only after every node that consumes it.

This is why every `backward_fn` accumulates with `+=` rather than assigning. Miss
that in a single op and a diamond-shaped graph will silently lose half its
gradient — the model still trains, just worse, with nothing to point at.

### Why reverse mode

For a function ℝⁿ → ℝ — which is what every loss function is — reverse mode
computes **all n partial derivatives in one backward pass**. Forward mode would
need n separate passes, one per input. With n in the hundreds of thousands, that
difference is the entire reason deep learning is tractable.

---

## The fused softmax + cross-entropy gradient

Softmax and cross-entropy are implemented as a single operation. Two reasons, and
both matter.

**Numerical stability.** Softmax needs `exp`, and `exp(1000.0f)` is `inf`. The
standard fix is to subtract the row maximum, which leaves the result unchanged
because the constant cancels in the ratio:

$$p_j = \frac{e^{z_j - m}}{\sum_k e^{z_k - m}} = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

Cross-entropy then takes the log of that, and the `log` undoes the `exp` — so the
fused version never materialises a large exponential at all:

$$L = -\log p_y = -\Big( (z_y - m) - \log \textstyle\sum_k e^{z_k - m} \Big)$$

**A gradient that collapses.** Differentiating the fused expression with respect
to the logits gives:

$$\frac{\partial L}{\partial z_j} = p_j - \mathbb{1}[j = y]$$

The softmax probabilities, minus one at the true class. Averaged over a batch of
size _m_, that is the entire backward pass:

```cpp
const float g = out_grad / m;
for (int j = 0; j < C; ++j) logit_grad[row + j] += g * probs[row + j];
logit_grad[row + labels[i]] -= g;
```

Three lines, no `exp`, no division, nothing that can overflow. Computing the two
operations separately gives the same answer mathematically while being both
slower and less stable.

---

## API tour

```cpp
// Tensors — 2-D, row-major
auto a = nabla::Tensor::zeros(3, 4, /*requires_grad=*/true);
auto b = nabla::Tensor::randn(4, 2, /*stddev=*/0.1f, rng, true);
auto c = nabla::Tensor::from(2, 2, {1, 2, 3, 4});

// Operations build the graph
auto y = nabla::relu(nabla::matmul(a, b));
auto loss = nabla::mean(y);
loss.backward();

// Gradients live next to the data
a.grad()[0];

// Inference: no graph, no allocation for it
{
    nabla::NoGradGuard no_grad;
    auto logits = model(x);
}

// Checkpoints
auto params = model.parameters();
nabla::save("model.nbla", params);
nabla::load("model.nbla", params);
```

---

## Results

<!-- TODO: fill in after training. Do not publish numbers you have not measured. -->

Model: `784 → 256 → ReLU → Dropout(0.2) → 128 → ReLU → Dropout(0.2) → 10`

|                |                                          |
| -------------- | ---------------------------------------- |
| Parameters     | 235,146                                  |
| Test accuracy  | _TBD_                                    |
| Time per epoch | _TBD_                                    |
| Epochs         | 15                                       |
| Optimiser      | AdamW, lr 1e-3, cosine decay with warmup |
| Machine        | _TBD_                                    |

Reproduce with:

```bash
./build/mnist_mlp --data data --epochs 15 --seed 1337
```

---

## Benchmark

<!-- TODO: fill in. An honest loss here reads better than a missing section. -->

Same architecture, same hyperparameters, PyTorch on CPU vs `nabla`:

|                     | PyTorch (CPU) | nabla |
| ------------------- | ------------- | ----- |
| Time per epoch      | _TBD_         | _TBD_ |
| Final test accuracy | _TBD_         | _TBD_ |

`nabla` is expected to be slower. The matmul here is a plain i-k-j loop with
OpenMP over rows — no cache blocking, no SIMD kernels, no BLAS. PyTorch dispatches
to MKL or OpenBLAS, which are decades of hand-tuned assembly. Closing that gap is
a different project than understanding autodiff, so it is out of scope by choice.

---

## Design notes

A few decisions that are easy to get wrong.

**`Tensor` is a handle, not a value.** It wraps a `shared_ptr<TensorImpl>`, so
copying a `Tensor` gives you another reference to the same node. If it deep-copied
instead, two operations could not share an input, and the graph could not have a
diamond in it.

**Backward closures capture raw pointers, never `shared_ptr`.** A node whose
closure holds a `shared_ptr` to itself can never be destroyed — a reference cycle
that shows up as memory climbing steadily through training and nothing else. The
graph is kept alive by the output tensor for as long as the backward pass needs
it.

**The topological sort is iterative.** A recursive DFS is shorter, but graph
depth is unbounded — an unrolled loop is a deep graph — and blowing the C++ stack
is an ugly way to find that out.

**Gradients accumulate.** `zero_grad()` is the caller's responsibility, exactly as
in PyTorch, because a tensor can legitimately receive gradient from several
places before a step.

**The optimiser knows nothing about layers.** It takes a flat
`std::vector<Tensor>` and never includes `nn.hpp`. That one-way dependency is what
keeps the two halves of the library independent.

---

## Roadmap

- [ ] `Conv2D` and pooling
- [ ] N-dimensional tensors with general broadcasting
- [ ] Cache-blocked and SIMD matmul kernels
- [ ] Batch normalisation and layer normalisation
- [ ] Browser demo compiled to WebAssembly

---

## Acknowledgements

The scalar-autograd approach was the starting point — the same idea Andrej
Karpathy's [micrograd](https://github.com/karpathy/micrograd) demonstrates in
Python, rebuilt here on matrices in C++. The API borrows its shape from PyTorch
because that shape is genuinely good.

## License

MIT — see [LICENSE](LICENSE).
