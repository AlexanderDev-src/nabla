#!/usr/bin/env python3
"""Run a checkpoint from `mnist_mlp --save` on the MNIST test set, in numpy
and without nabla, to show the file holds the trained model.

The checkpoint must be Linear layers with ReLU between them, as in
examples/mnist_mlp.cpp. Dropout has no parameters and does nothing at test
time, so it needs no code here.

Usage: python3 scripts/eval_checkpoint.py model.nbla [data_dir]
"""
import sys
from pathlib import Path

import numpy as np

from read_checkpoint import read_checkpoint


def main():
    tensors = read_checkpoint(sys.argv[1])
    data = Path(sys.argv[2] if len(sys.argv) > 2 else "data")

    # IDX headers are 16 bytes for images and 8 for labels; the md5 check in
    # get_mnist.sh already vouches for the rest
    images = np.fromfile(data / "t10k-images-idx3-ubyte", np.uint8, offset=16)
    labels = np.fromfile(data / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    x = images.reshape(-1, 784).astype(np.float32)
    x = (x / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)

    # weight, bias, weight, bias, ...: ReLU after every layer but the last
    layers = list(zip(tensors[0::2], tensors[1::2]))
    for i, (weight, bias) in enumerate(layers):
        x = x @ weight + bias
        if i < len(layers) - 1:
            x = np.maximum(x, 0)

    accuracy = np.mean(x.argmax(axis=1) == labels)
    print(f"test accuracy {accuracy * 100:.2f}%  ({len(labels)} images)")


if __name__ == "__main__":
    main()
