#!/usr/bin/env bash
# Downloads the four MNIST IDX files into data/, checks them, and unpacks
# them. Files already in data/ are skipped, so it is safe to run again.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
base=https://ossci-datasets.s3.amazonaws.com/mnist

# md5 of each .gz, the same values torchvision checks against
while read -r md5 name; do
    [[ -f data/$name ]] && continue
    echo "downloading $name"
    curl -fsSL "$base/$name.gz" -o "data/$name.gz"
    echo "$md5  data/$name.gz" | md5sum -c --quiet
    gunzip "data/$name.gz"
done <<'LIST'
f68b3c2dcbeaaa9fbdd348bbdeb94873 train-images-idx3-ubyte
d53e105ee54ea40749a09fcbcd1e9432 train-labels-idx1-ubyte
9fb629c4189551a2d022fa330f9573f3 t10k-images-idx3-ubyte
ec29112dd5afa0611ce80d1b7f02629c t10k-labels-idx1-ubyte
LIST
