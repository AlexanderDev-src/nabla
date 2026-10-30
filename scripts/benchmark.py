#!/usr/bin/env python3
"""Train the MNIST MLP with nabla and with PyTorch on each thread count, and
print a markdown table of the median time per epoch and the final test
accuracy of every run.

Run from the repository root, after building mnist_mlp in Release and
installing torch into .venv (see the Benchmark section of the README).

Usage: python3 scripts/benchmark.py [--epochs 15] [--threads 1 6]
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

LINE = re.compile(r"accuracy ([\d.]+)%  time ([\d.]+)s")


def run(name, cmd, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    times, accuracy = [], None
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True) as p:
        for line in p.stdout or ():
            progress = f"{name}, {threads} threads: {line.rstrip()}"
            print(progress, file=sys.stderr)
            if m := LINE.search(line):
                accuracy, t = float(m[1]), float(m[2])
                times.append(t)
    if p.returncode != 0 or not times:
        sys.exit(f"{name} failed: {' '.join(cmd)}")
    return statistics.median(times), accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--nabla", default="build/mnist_mlp")
    parser.add_argument("--python", default=".venv/bin/python")
    args = parser.parse_args()

    epochs = ["--epochs", str(args.epochs)]
    rows = []
    for threads in args.threads:
        nabla = run("nabla", [args.nabla, *epochs], threads)
        torch = run("PyTorch",
                    [args.python, "scripts/mnist_torch.py", *epochs,
                     "--threads", str(threads)], threads)
        rows.append((threads, nabla, torch))

    print("| Threads | nabla s/epoch | PyTorch s/epoch "
          "| nabla acc | PyTorch acc |")
    print("| ------- | ------------- | --------------- "
          "| --------- | ----------- |")
    for threads, (nt, na), (tt, ta) in rows:
        print(f"| {threads} | {nt:.2f} | {tt:.2f} | {na:.2f}% | {ta:.2f}% |")


if __name__ == "__main__":
    main()
