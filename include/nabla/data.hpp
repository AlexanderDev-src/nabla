#pragma once
#include "nabla/autograd.hpp"
#include "nabla/nn.hpp"
#include "nabla/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nabla {

// One IDX file: the size of each dimension, then every value in row-major
// order. MNIST images come out as dims {N, 28, 28}, labels as dims {N}.
struct Idx {
    std::vector<std::size_t> dims;
    std::vector<std::uint8_t> data;
};

// Reads 4 bytes as one big-endian number: the first byte is the most
// significant, so 00 00 EA 60 is 60000. Throws std::runtime_error if the
// stream ends first.
inline std::uint32_t read_u32_be(std::istream &in) {
    unsigned char bytes[4];

    in.read(reinterpret_cast<char *>(bytes), 4);
    if (in.gcount() != 4) {
        throw std::runtime_error("Expected 4 bytes");
    }

    return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) |
           (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
}

// IDX layout: 2 zero bytes, a type byte (0x08 means unsigned byte), a byte n
// with the number of dimensions, n sizes as big-endian u32, then the data,
// one byte per value. Throws std::runtime_error on a bad header, on sizes
// whose product does not fit in size_t, or on a short file.
inline Idx read_idx(std::istream &in) {
    unsigned char header[4];
    in.read(reinterpret_cast<char *>(header), 4);

    if (in.gcount() != 4 || header[0] != 0 || header[1] != 0 ||
        header[2] != 0x08) {
        throw std::runtime_error("Invalid IDX header");
    }

    Idx result;
    std::size_t count = 1;

    for (unsigned int i = 0; i < header[3]; ++i) {
        const auto dim = read_u32_be(in);
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("IDX dimensions are too large");
        }
        result.dims.push_back(dim);
        count *= dim;
    }
    if (count > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("IDX data is too large");
    }

    result.data.resize(count);

    if (count != 0) {
        const auto bytes = static_cast<std::streamsize>(count);
        in.read(reinterpret_cast<char *>(result.data.data()), bytes);

        if (in.gcount() != bytes) {
            throw std::runtime_error("IDX data is shorter than expected");
        }
    }

    return result;
}

// Opens path in binary mode and reads it with read_idx. Throws
// std::runtime_error if the file cannot be opened.
inline Idx load_idx(const std::string &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error("Cannot open IDX file: " + path);
    }

    return read_idx(in);
}

// Turns pixel bytes 0..255 into floats with mean about 0 and standard
// deviation about 1 over the training set:
//   (x / 255 - 0.1307) / 0.3081
// 0.1307 and 0.3081 are the mean and standard deviation of every training
// pixel after dividing by 255.
inline std::vector<float>
normalize_mnist(const std::vector<std::uint8_t> &pixels) {
    std::vector<float> result;
    result.reserve(pixels.size());

    for (auto x : pixels) {
        result.push_back((x / 255.0f - 0.1307f) / 0.3081f);
    }

    return result;
}

// The indices 0..n-1 in a random order, cut into batches of batch_size; the
// last batch is shorter when batch_size does not divide n. Only the indices
// move, never the images, so an epoch shuffles n numbers instead of
// n * 784 floats. Pass the same rng every epoch so each epoch gets a new
// order.
inline std::vector<std::vector<std::size_t>>
shuffled_batches(std::size_t n, std::size_t batch_size, std::mt19937 &rng) {
    if (batch_size == 0) {
        throw std::invalid_argument("shuffled_batches: batch_size is 0");
    }

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0uz);

    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<std::vector<std::size_t>> batches;

    for (std::size_t start = 0; start < n;) {
        const auto count = std::min(batch_size, n - start);

        batches.emplace_back(indices.begin() + start,
                             indices.begin() + start + count);

        start += count;
    }

    return batches;
}

// One training step's worth of examples: x holds one image per row, and
// y[i] is the label of row i.
struct Batch {
    Tensor x;
    std::vector<std::size_t> y;
};

// Copies the examples listed in indices out of the full set: row i of x is
// image indices[i], and y[i] is its label. images holds every image one
// after another, `features` floats each (784 for MNIST). Throws
// std::out_of_range if an index is past the last image.
inline Batch make_batch(const std::vector<float> &images,
                        const std::vector<std::uint8_t> &labels,
                        std::size_t features,
                        const std::vector<std::size_t> &indices) {
    Tensor x = Tensor::zeros(indices.size(), features);
    std::vector<std::size_t> y;
    y.reserve(indices.size());

    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto image_index = indices[i];

        if (image_index >= labels.size() ||
            (features != 0 && image_index >= images.size() / features)) {
            throw std::out_of_range("make_batch: image index out of range");
        }

        for (std::size_t j = 0; j < features; ++j) {
            x.at(i, j) = images[image_index * features + j];
        }

        y.push_back(labels[image_index]);
    }

    return {x, y};
}

// The column of the largest value in each row: for logits, the class the
// model picks. A tie goes to the first of the largest.
inline std::vector<std::size_t> argmax_rows(const Tensor &logits) {
    std::vector<std::size_t> result;
    result.reserve(logits.rows());

    for (std::size_t r = 0; r < logits.rows(); ++r) {
        std::size_t best_col = 0;

        for (std::size_t c = 1; c < logits.cols(); ++c) {
            if (logits.at(r, c) > logits.at(r, best_col)) {
                best_col = c;
            }
        }

        result.push_back(best_col);
    }

    return result;
}

// The share of examples, from 0 to 1, whose largest logit is at their
// label. Runs the model in eval mode under NoGradGuard, batch_size examples
// at a time and in order, then puts the model back in the mode it was in.
inline float evaluate(Module &model, const std::vector<float> &images,
                      const std::vector<std::uint8_t> &labels,
                      std::size_t features, std::size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("evaluate: batch_size is 0");
    }

    if (labels.empty()) {
        return 0.0f;
    }

    const bool was_training = model.is_training();
    model.eval();
    NoGradGuard guard;

    std::size_t correct = 0;

    try {
        for (std::size_t start = 0; start < labels.size();) {
            const auto count = std::min(batch_size, labels.size() - start);

            std::vector<std::size_t> indices(count);
            std::iota(indices.begin(), indices.end(), start);

            const auto batch = make_batch(images, labels, features, indices);
            const auto predictions = argmax_rows(model.forward(batch.x));

            for (std::size_t i = 0; i < count; ++i) {
                if (predictions[i] == batch.y[i]) {
                    ++correct;
                }
            }

            start += count;
        }
    } catch (...) {
        if (was_training) {
            model.train();
        }
        throw;
    }

    if (was_training) {
        model.train();
    }

    return static_cast<float>(correct) / labels.size();
}

} // namespace nabla
