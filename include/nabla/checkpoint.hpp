#pragma once
#include "nabla/nn.hpp"
#include "nabla/tensor.hpp"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nabla {

// A checkpoint file. Every number is little-endian:
//
//   "NBLA"                  4 bytes
//   version                 u32, always 1 for now
//   count                   u32, the number of tensors
//   then for each tensor, in the order of parameters():
//     rows, cols            u32 each
//     rows * cols values    f32 each, row-major
//
// Only the values are saved: no grad, no requires_grad, no layer names. So
// a checkpoint fits only a model built the same way, and read_checkpoint
// checks the count and every shape to catch one that does not.
//
// Little-endian because x86 and ARM already keep numbers in memory that way,
// and Python reads it with struct "<I" and numpy "<f4"
// (scripts/read_checkpoint.py).
static_assert(std::numeric_limits<float>::is_iec559,
              "checkpoints store float as IEEE 754 single precision");

// Writes v as 4 bytes, least significant first: 60000 is 60 EA 00 00.
inline void write_u32_le(std::ostream &out, std::uint32_t v) {
    const char bytes[4] = {static_cast<char>(v & 0xFF),
                           static_cast<char>((v >> 8) & 0xFF),
                           static_cast<char>((v >> 16) & 0xFF),
                           static_cast<char>((v >> 24) & 0xFF)};
    out.write(bytes, 4);
}

// Reads 4 bytes written by write_u32_le. Throws std::runtime_error if the
// stream ends first.
inline std::uint32_t read_u32_le(std::istream &in) {
    unsigned char bytes[4];
    in.read(reinterpret_cast<char *>(bytes), 4);
    if (in.gcount() != 4) {
        throw std::runtime_error("Expected 4 checkpoint bytes");
    }
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
           (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
}

// Writes params to out in the layout above. A float goes out through
// write_u32_le as its 4 bytes: std::bit_cast<std::uint32_t>(x) gives those
// bytes as a number. static_cast would convert the value (1.5f to 1).
inline void write_checkpoint(std::ostream &out,
                             const std::vector<Tensor> &params) {
    const auto max = std::numeric_limits<std::uint32_t>::max();
    if (params.size() > max) {
        throw std::runtime_error("Too many checkpoint tensors");
    }
    for (const auto &p : params) {
        if (p.rows() > max || p.cols() > max) {
            throw std::runtime_error("Checkpoint dimensions are too large");
        }
    }
    out.write("NBLA", 4);
    write_u32_le(out, 1);
    write_u32_le(out, static_cast<std::uint32_t>(params.size()));
    for (const auto &p : params) {
        write_u32_le(out, static_cast<std::uint32_t>(p.rows()));
        write_u32_le(out, static_cast<std::uint32_t>(p.cols()));
        for (float x : p.impl()->data) {
            write_u32_le(out, std::bit_cast<std::uint32_t>(x));
        }
    }
    if (!out) {
        throw std::runtime_error("Could not write checkpoint");
    }
}

// Reads a checkpoint from in into params, which must already have the shapes
// in the file. Throws std::runtime_error on a wrong magic or version, a count
// or shape that does not match params, or a file that ends early.
//
// If it throws, params are exactly as they were: read the whole file first,
// and only copy into params once every tensor has been read and checked.
inline void read_checkpoint(std::istream &in,
                            const std::vector<Tensor> &params) {
    char magic[4];
    in.read(magic, 4);
    if (in.gcount() != 4 || std::string_view(magic, 4) != "NBLA") {
        throw std::runtime_error("Invalid checkpoint magic");
    }
    if (read_u32_le(in) != 1) {
        throw std::runtime_error("Unsupported checkpoint version");
    }
    if (read_u32_le(in) != params.size()) {
        throw std::runtime_error("Checkpoint tensor count mismatch");
    }
    std::vector<std::vector<float>> buffers;
    buffers.reserve(params.size());
    for (const auto &p : params) {
        const auto rows = read_u32_le(in);
        const auto cols = read_u32_le(in);
        if (rows != p.rows() || cols != p.cols()) {
            throw std::runtime_error("Checkpoint tensor shape mismatch");
        }
        auto &values = buffers.emplace_back(p.impl()->data.size());
        for (float &x : values) {
            x = std::bit_cast<float>(read_u32_le(in));
        }
    }
    for (std::size_t i = 0; i < params.size(); ++i) {
        std::copy(buffers[i].begin(), buffers[i].end(),
                  params[i].impl()->data.begin());
    }
}

// Opens path in binary mode and writes model.parameters() to it. Throws
// std::runtime_error if the file cannot be opened or written.
inline void save_checkpoint(const std::string &path, Module &model) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open checkpoint for writing: " +
                                 path);
    }
    write_checkpoint(out, model.parameters());
    out.close();
    if (!out) {
        throw std::runtime_error("Could not finish writing checkpoint: " +
                                 path);
    }
}

// Opens path in binary mode and reads it into model.parameters(). Throws
// std::runtime_error if the file cannot be opened, and whatever
// read_checkpoint throws.
inline void load_checkpoint(const std::string &path, Module &model) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open checkpoint for reading: " +
                                 path);
    }
    read_checkpoint(in, model.parameters());
}

} // namespace nabla
