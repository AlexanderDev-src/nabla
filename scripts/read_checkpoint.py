#!/usr/bin/env python3
"""Read a nabla checkpoint into a list of numpy arrays, in parameters() order.

Usage: python3 scripts/read_checkpoint.py model.nbla
"""
import struct
import sys

import numpy as np


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


if __name__ == "__main__":
    for i, t in enumerate(read_checkpoint(sys.argv[1])):
        print(f"{i}: {t.shape[0]}x{t.shape[1]}  mean {t.mean():+.6f}")
