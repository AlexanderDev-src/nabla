<h1 align="center">nabla</h1>

<p align="center">
  A tiny reverse-mode autodiff engine and neural network library in C++23.<br>
  Header-only, zero dependencies.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-23-blue.svg" alt="C++23">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT"></a>
</p>

---

`nabla` builds a computation graph as your code runs, then differentiates it
automatically. You write the forward pass; the backward pass comes for free.

This is [`examples/xor.cpp`](examples/xor.cpp), as it stands today:

```cpp
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
    SGD optimizer(model.parameters(), 0.5f);

    for (int step = 0; step <= 2000; ++step) {
        optimizer.zero_grad();
        Tensor loss = mse_loss(model.forward(x), y);
        loss.backward();
        optimizer.step();
        if (step % 250 == 0) {
            std::printf("step %4d  loss %.6f\n", step, loss.at(0, 0));
        }
    }
    Tensor out = model.forward(x);
    for (std::size_t i = 0; i < 4; ++i) {
        std::printf("%g xor %g = %.4f\n", x.at(i, 0), x.at(i, 1), out.at(i, 0));
    }
}
```

```
step    0  loss 0.357442
step 2000  loss 0.000462
0 xor 0 = 0.0128
0 xor 1 = 0.9768
1 xor 0 = 0.9769
1 xor 1 = 0.0247
```

No layer ever implements `backward()` — the graph knows how to reverse itself.
Leaf gradients accumulate across `backward()` calls, so every step starts with
`zero_grad()`; `step()` then moves each parameter against its gradient.

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
- [Checkpoints](#checkpoints)
- [Results](#results)
- [Benchmark](#benchmark)
- [Design notes](#design-notes)
- [Roadmap](#roadmap)

---

## What this is

A learning-grade automatic differentiation engine, written to understand how
PyTorch's `autograd` actually works — and then made fast enough and complete
enough to train a real model on a real dataset. It is built in numbered
milestones; M0 to M7 are done.

**What it has today**

- Reverse-mode automatic differentiation over a dynamically built graph
- Ops: `matmul`, `add` with row broadcasting, element-wise `mul`, `relu`, `tanh`,
  `sigmoid`, and `sum`
- `mse_loss`, and a fused, numerically stable `softmax_cross_entropy`
- `Module`, `Linear` with He initialisation, `ReLU`, `Tanh`, `Sigmoid`, inverted
  `Dropout`, and `Sequential`, with `train()` / `eval()`
- `NoGradGuard` for inference without building a graph
- `SGD` with momentum and weight decay, `Adam`, `clip_grad_norm`, and a cosine
  LR schedule with linear warmup
- An MNIST pipeline: a parser for the original IDX files, pixel normalisation,
  shuffled mini-batches, and a batched `evaluate()`
- Binary checkpoints that a ten-line Python function can read, see
  [Checkpoints](#checkpoints)
- OpenMP in `matmul`: forward and backward split their rows over threads, and
  give the same bits on any number of threads
- A finite-difference gradient checker that tests every op, both losses, and a
  full two-layer MLP
- `Scalar`, the single-number engine from M0 that the tensor version grew out of
  ([`scalar.hpp`](include/nabla/scalar.hpp), demo in [`src/main.cpp`](src/main.cpp))

**Planned** (see [Roadmap](#roadmap))

- The optional end of M7: `t[i, j]` indexing behind
  `__cpp_multidimensional_subscript`

**What it deliberately doesn't have**

- **2-D tensors only.** An MLP never needs more, and skipping N-d strides keeps
  every backward pass short enough to read in one sitting.
- **`float` only.** No dtype dispatch, no templates over scalar types.
- **CPU only.** No CUDA.
- **No BLAS.** The matmul is a hand-written loop. It is slower than OpenBLAS and
  that is the intended trade — the point is that you can read it.

These are scope decisions, not missing features. A library that does one thing
and can be understood end to end is more useful for learning than one that does
everything opaquely.

---

## Build

Header-only. Copy `include/nabla/` into your project and add `include/` to your
include path, then `#include "nabla/nabla.hpp"`. To build the tests and examples
in this repository:

```bash
git clone https://github.com/AlexanderDev-src/nabla.git
cd nabla
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/nabla_tests
```

Requires a C++23 compiler and CMake 3.20 or newer (the first version that
accepts `CMAKE_CXX_STANDARD 23`). Tested on GCC 16.2 and
Clang 23.1 on Linux. The CMake file passes GCC/Clang warning flags, so MSVC is
not supported as is.

If the compiler has OpenMP, CMake links it and `matmul` runs on several
threads; `OMP_NUM_THREADS` sets how many. Without OpenMP everything runs on one
thread. To build without it anyway, add
`-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON`.

The build makes five targets:

| Target             | What it is                                       |
| ------------------ | ------------------------------------------------ |
| `nabla_tests`      | The test suite (`tests/*.cpp`)                   |
| `xor`              | The XOR example above                            |
| `mnist_mlp`        | The MNIST trainer, see [Results](#results)       |
| `nabla`            | The M0 `Scalar` demo in `src/main.cpp`           |
| `harness_selftest` | Checks the test harness itself                   |

Run the example:

```bash
./build/xor
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

Each node (`TensorImpl`) holds:

```cpp
std::vector<float> data;
std::vector<float> grad;
bool requires_grad = false;
std::vector<std::shared_ptr<TensorImpl>> parents;                  // where this value came from
std::function<void(const std::vector<float> &upstream)> backward_fn; // adds into parents' grad
```

Nothing is precomputed. The graph is whatever your control flow happened to
execute — `if`, loops, recursion all work, because the graph is a record of what
ran, not a description of what might run.

### The backward pass

Calling `.backward()` on a scalar does three things:

1. **Topologically sort** the graph reachable from that scalar.
2. **Seed** its gradient with 1.0 — because ∂L/∂L = 1.
3. **Walk the sorted list in reverse**, calling each node's `backward_fn` with
   that node's gradient.

The sort is what makes this correct. If a tensor feeds two different operations,
its gradient is the _sum_ of both paths, and it must not propagate anything until
both contributions have landed. Reverse topological order guarantees exactly
that: a node is visited only after every node that consumes it.

This is why every `backward_fn` accumulates with `+=` rather than assigning. Miss
that in a single op and a diamond-shaped graph will silently lose half its
gradient — the model still trains, just worse, with nothing to point at.

Before the walk, `backward()` zeroes the gradient of every intermediate node, so
calling it twice on the same graph does not double them. Leaf tensors, such as
weights, are left alone: their gradients keep accumulating until the caller
zeroes them.

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
`B` rows, that is the entire backward pass
([`loss.hpp`](include/nabla/loss.hpp)):

```cpp
for (std::size_t i = 0; i < B; ++i) {
    for (std::size_t j = 0; j < C; ++j) {
        float y = (j == labels[i]) ? 1.0f : 0.0f;
        li->grad[i * C + j] +=
            g[0] * (probs[i * C + j] - y) / static_cast<float>(B);
    }
}
```

No `exp`, nothing that can overflow: the probabilities were already computed in
the forward pass and kept for this. Computing the two operations separately
gives the same answer mathematically while being both slower and less stable.
The test suite checks that a logit of 1000 gives no `inf` or `nan`.

---

## API tour

```cpp
#include "nabla/nabla.hpp"
using namespace nabla;

std::mt19937 rng(42);

// Tensors — 2-D, row-major. Copying a Tensor copies the handle, not the data.
Tensor a = Tensor::zeros(3, 4);
Tensor b = Tensor::randn(4, 2, rng, /*mean=*/0.0f, /*stddev=*/0.1f);
Tensor c = Tensor::from(2, 2, {1, 2, 3, 4});
a.set_requires_grad(true);
b.set_requires_grad(true);

// Operations build the graph
Tensor loss = sum(relu(matmul(a, b)));
loss.backward();

// Gradients live next to the data
float g = a.grad_at(0, 0);

// Layers
Sequential model{std::make_shared<Linear>(784, 128, rng),
                 std::make_shared<ReLU>(),
                 std::make_shared<Linear>(128, 10, rng)};
std::vector<Tensor> params = model.parameters();   // weight, bias, weight, bias
model.eval();                                      // passed down to every layer

// Losses
Tensor l1 = mse_loss(pred, target);
Tensor l2 = softmax_cross_entropy(logits, labels); // labels: std::vector<std::size_t>

// Optimisers take the flat parameter list. One training step:
Adam opt(params, /*lr=*/1e-3f);
opt.zero_grad();                              // grads accumulate otherwise
l2.backward();
float norm = clip_grad_norm(params, 1.0f);    // returns the norm before clipping
opt.set_lr(cosine_lr(t, /*warmup=*/100, /*total=*/1000, /*max_lr=*/1e-3f));
opt.step();

// Inference: no graph is recorded inside the guard
{
    NoGradGuard no_grad;
    Tensor logits = model.forward(x);
}

// Checkpoints hold parameter values only, so load into a model built the same way
save_checkpoint("model.nbla", model);
load_checkpoint("model.nbla", model);   // throws if the count or a shape differs
```

---

## Checkpoints

`save_checkpoint` writes `model.parameters()` in order, every number
little-endian:

```
"NBLA"                  4 bytes
version                 u32, 1
count                   u32, the number of tensors
then for each tensor:
  rows, cols            u32 each
  rows * cols values    f32 each, row-major
```

There are no layer names, grads or optimiser state, so a checkpoint fits only a
model built the same way. `load_checkpoint` checks the count and every shape, and
reads the whole file before it touches the model: if it throws, the model is
exactly as it was.

The format is simple enough that Python reads it in ten lines
([`scripts/read_checkpoint.py`](scripts/read_checkpoint.py)):

```python
def read_checkpoint(path):
    with open(path, "rb") as f:
        magic, version, count = struct.unpack("<4sII", f.read(12))
        if magic != b"NBLA" or version != 1:
            raise ValueError(f"{path} is not a version 1 nabla checkpoint")
        tensors = []
        for _ in range(count):
            rows, cols = struct.unpack("<II", f.read(8))
            data = np.frombuffer(f.read(4 * rows * cols), dtype="<f4")
            tensors.append(data.reshape(rows, cols))
        return tensors
```

---

## Results

[`examples/mnist_mlp.cpp`](examples/mnist_mlp.cpp) trains this MLP on MNIST:
`784 → 256 → ReLU → Dropout(0.2) → 128 → ReLU → Dropout(0.2) → 10`

|                |                                                           |
| -------------- | --------------------------------------------------------- |
| Parameters     | 235,146                                                   |
| Test accuracy  | 98.40% after 15 epochs, and 98% or more from epoch 5      |
| Time per epoch | about 5.8 s on 6 threads, about 17 s on one               |
| Epochs         | 15                                                        |
| Batch size     | 64                                                        |
| Optimiser      | Adam, lr 1e-3, cosine decay after half an epoch of warmup |
| Gradient clip  | global norm 1.0                                           |
| Machine        | AMD Ryzen 7 7445HS (6 cores), GCC 16.2, `-O3`, OpenMP     |

```
epoch 1  loss 0.5573  accuracy 95.47%  time 5.69s
epoch 5  loss 0.0743  accuracy 98.05%  time 5.88s
epoch 10  loss 0.0266  accuracy 98.32%  time 6.02s
epoch 15  loss 0.0130  accuracy 98.40%  time 5.94s
```

Before OpenMP (M7.2) an epoch took about 47 s. The loss and accuracy are the
same to every printed digit as in that run; only the time changed. Why one
thread also got faster, and why 6 threads beat 12, is in
[Design notes](#design-notes).

To run it, from the repository root, since it reads `data/`:

```bash
./scripts/get_mnist.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target mnist_mlp
OMP_NUM_THREADS=6 ./build/mnist_mlp --epochs 15 --batch 64 --lr 1e-3
```

Set `OMP_NUM_THREADS` to the number of physical cores, not hardware threads.

`--save model.nbla` writes a checkpoint after every epoch.
[`scripts/eval_checkpoint.py`](scripts/eval_checkpoint.py) runs that checkpoint
on the test set in numpy, without nabla. After one epoch both printed 96.43%,
and after 15 both printed 98.40%:

```bash
./build/mnist_mlp --epochs 1 --save model.nbla
python3 scripts/eval_checkpoint.py model.nbla
```

---

## Benchmark

The same MLP, initialisation, optimiser, schedule, clipping and batch size in
PyTorch 2.15 on the CPU ([`scripts/mnist_torch.py`](scripts/mnist_torch.py)),
15 epochs each, on the machine from [Results](#results). Times are the median
epoch of one run. [`scripts/benchmark.py`](scripts/benchmark.py) runs all four
and prints the numbers.

| Threads | nabla   | PyTorch | nabla is slower by | nabla acc | PyTorch acc |
| ------- | ------- | ------- | ------------------ | --------- | ----------- |
| 1       | 17.77 s | 1.65 s  | 10.8×              | 98.40%    | 98.47%      |
| 6       | 6.01 s  | 1.71 s  | 3.5×               | 98.40%    | 98.50%      |

Both reach the same accuracy, so nabla's gradients, Adam and schedule do their
job; the small difference comes from different random numbers.

nabla is about 11 times slower on one thread and 3.5 times slower on six. An
epoch is about 60 GFLOP of matmul work, so nabla runs at about 3.6 GFLOP/s on
one thread and 10 on six, against about 37 for PyTorch on one. Nearly all of
that arithmetic is matmul, and that is where the two differ:

- **SIMD.** PyTorch calls MKL, which multiplies and adds 8 or 16 floats per
  instruction (AVX2, AVX-512). nabla is built without `-march=native`, and its
  forward loop adds into a single `sum` in order, which the compiler may not
  reorder, so it mostly does one at a time.
- **Cache.** MKL works on blocks that fit in cache and reuses each value it
  loads many times. nabla's forward reads B down a column, a whole row apart per
  step.
- **Batch 64 is small.** Much of PyTorch's time is the cost of each operation
  rather than the arithmetic, which is why six threads do not help it.

Single epochs varied a lot on this laptop, which changes its clock speed: nabla
on one thread took 15.0 to 20.4 s, PyTorch on six 1.1 to 2.0 s. Take the ratios
as rough. Closing the gap would mean writing a BLAS, which is a different
project than understanding autodiff, so it is out of scope by choice.

To run it, after building `mnist_mlp` in Release as in [Results](#results):

```bash
uv venv .venv
uv pip install --python .venv torch numpy \
    --index-url https://download.pytorch.org/whl/cpu
python3 scripts/benchmark.py
```

The CPU index matters: the default one installs the CUDA build, several GB.

---

## Design notes

A few decisions that are easy to get wrong.

**`Tensor` is a handle, not a value.** It wraps a `shared_ptr<TensorImpl>`, so
copying a `Tensor` gives you another reference to the same node. If it deep-copied
instead, two operations could not share an input, and the graph could not have a
diamond in it. When you do want separate storage, ask for it with `clone()`.

**Backward closures capture their parents, never their own node.** Each
`backward_fn` holds `shared_ptr`s to the tensors it read from, which keeps them
alive for the backward pass. A closure that held a `shared_ptr` to its own node
would form a reference cycle and the node could never be destroyed — memory
climbing steadily through training and nothing else to show for it. Ownership
only ever points from a node back to its inputs, so the whole graph is freed
when the output tensor goes out of scope.

**The topological sort is iterative.** A recursive DFS is shorter, but graph
depth is unbounded — an unrolled loop is a deep graph — and blowing the C++ stack
is an ugly way to find that out. `backward()` uses an explicit stack instead.

**Gradients accumulate.** Zeroing leaf gradients is the caller's responsibility,
exactly as in PyTorch, because a tensor can legitimately receive gradient from
several places before a step. In a training loop that is
`Optimizer::zero_grad()`.

**The optimiser knows nothing about layers.** It takes a flat
`std::vector<Tensor>` and never includes `nn.hpp`. That one-way dependency is
what keeps the two halves of the library independent.

**Every grad element has one writer.** `matmul`'s backward is two loops: A's
grad is split over threads by its rows `i`, B's grad by its rows `p`. B's
`grad[p, j]` gets a term from every row of A, so one loop split by `i` would
have threads adding into the same element and losing updates. With one writer
per element, each sum is added in the same order on any number of threads, so 6
threads give the same bits as one. Splitting the loops also took one thread from
about 47 s to about 17 s per epoch: the inner loop for B's grad now walks memory
in order instead of jumping a whole row per step, which is what the cache wants.

**More threads is not faster.** The Ryzen 7 7445HS has 6 cores with 2 hardware
threads each. On all 12, an epoch took roughly twice as long as on 6: two
threads on one core share its arithmetic units, and OpenMP's idle threads spin
while the main thread runs the parts that are not parallel (`add`, `relu`,
Adam), taking the core it shares with them. `OMP_WAIT_POLICY=passive` makes idle
threads sleep instead, which brought 12 threads back to about 6.6 s.

---

## Roadmap

- [x] M0 · `Scalar` autograd on single numbers
- [x] M1 · 2-D `Tensor` with `matmul` and broadcasting `add`, forward only
- [x] M2 · The graph: `parents`, `backward_fn`, `relu`, and `backward()`
- [x] M3 · Finite-difference gradient checker over every op
- [x] M4 · `Module`, `Linear`, `Sequential`, activations, `mse_loss`,
      `softmax_cross_entropy`, and the XOR example
- [x] M5 · Optimisers: SGD with momentum, Adam, gradient clipping, cosine LR schedule
- [x] M6 · MNIST: IDX loader, batching, `Dropout`, and an MLP at about 98% test accuracy
- [x] M7 · Checkpoints, OpenMP matmul, and a benchmark against PyTorch

---

## Acknowledgements

The scalar-autograd approach was the starting point — the same idea Andrej
Karpathy's [micrograd](https://github.com/karpathy/micrograd) demonstrates in
Python, rebuilt here on matrices in C++. The API borrows its shape from PyTorch
because that shape is genuinely good.

## License

MIT — see [LICENSE](LICENSE).
