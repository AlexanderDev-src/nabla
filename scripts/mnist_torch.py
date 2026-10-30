#!/usr/bin/env python3
"""The PyTorch twin of examples/mnist_mlp.cpp, for the benchmark: the same
model, initialisation, optimiser, schedule, clipping and batches, and the same
output line, so scripts/benchmark.py can time both the same way.

Usage: .venv/bin/python scripts/mnist_torch.py [--epochs 15] [--threads 6]
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_idx(name, header_bytes):
    return np.fromfile(Path("data") / name, np.uint8, offset=header_bytes)


def normalize_mnist(pixels):
    x = pixels.reshape(-1, 784).astype(np.float32)
    x = (x / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)
    return torch.from_numpy(x)


def labels(pixels):
    return torch.from_numpy(pixels.astype(np.int64))


# nabla::cosine_lr with min_lr = 0
def cosine_lr(t, warmup, total, max_lr):
    if t < warmup:
        return max_lr * t / warmup
    if t >= total:
        return 0.0
    p = (t - warmup) / (total - warmup)
    return 0.5 * max_lr * (1 + math.cos(math.pi * p))


# Stops the script unless every Linear starts like nabla's: weights normal
# with std sqrt(2 / fan_in), biases zero. PyTorch's own default is uniform
# with a smaller std, so leaving it would compare two different models.
def check_init(model):
    for layer in model:
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach()
            want = math.sqrt(2 / layer.in_features)
            assert abs(w.std().item() / want - 1) < 0.1, "weight std is wrong"
            assert (w.abs() > 2 * want).any(), "weights are not normal"
            assert (layer.bias == 0).all(), "biases are not zero"


# nabla::evaluate: the share of y the model gets right, in batches, in eval
# mode without building a graph. Puts the model back in the mode it was in.
def evaluate(model, x, y, batch_size=1000):
    was_training = model.training
    correct = 0
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(y), batch_size):
                predictions = model(x[start:start + batch_size]).argmax(1)
                correct += int((predictions == y[start:start + batch_size]).sum())
    finally:
        model.train(was_training)
    return correct / len(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    train_x = normalize_mnist(load_idx("train-images-idx3-ubyte", 16))
    train_y = labels(load_idx("train-labels-idx1-ubyte", 8))
    test_x = normalize_mnist(load_idx("t10k-images-idx3-ubyte", 16))
    test_y = labels(load_idx("t10k-labels-idx1-ubyte", 8))

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    )

    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    check_init(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(train_y)
    steps = math.ceil(n / args.batch)
    total = args.epochs * steps
    t = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        loss_sum = 0.0

        for indices in torch.randperm(n).split(args.batch):
            x, y = train_x[indices], train_y[indices]
            for group in opt.param_groups:
                group["lr"] = cosine_lr(t, steps // 2, total, args.lr)
            t += 1

            opt.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()

        secs = time.perf_counter() - start
        accuracy = evaluate(model, test_x, test_y)
        print(f"epoch {epoch}  loss {loss_sum / steps:.4f}  "
              f"accuracy {accuracy * 100:.2f}%  time {secs:.2f}s", flush=True)


if __name__ == "__main__":
    main()
