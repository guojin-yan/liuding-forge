// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#pragma once

// Example-only, bounded reader for prepared MNIST IDX files. Downloading,
// SHA-256 verification and gzip decompression belong to prepare_mnist.py.
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace mnist_data {

/** @brief Reads one complete IDX file within the MNIST image payload bound. */
inline std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  constexpr std::uintmax_t kMaxBytes = 60000ULL * 28 * 28 + 16;
  const auto size = std::filesystem::file_size(path);
  if (size > kMaxBytes)
    throw std::runtime_error("MNIST IDX file exceeds limit");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open MNIST IDX file");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("MNIST IDX file size changed while reading");
  return bytes;
}

/** @brief Reads a bounded unsigned big-endian IDX header field. */
inline std::uint32_t read_be32(const std::vector<std::uint8_t>& bytes,
                               std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4)
    throw std::runtime_error("truncated MNIST IDX header");
  std::uint32_t result = 0;
  for (std::size_t index = 0; index < 4; ++index)
    result = (result << 8U) | bytes[offset + index];
  return result;
}

/** @brief CPU tensors retained independently of the input file buffers. */
struct IdxDataset {
  torch::Tensor images;  ///< Float32 [N,1,28,28] in [0,1].
  torch::Tensor labels;  ///< Int64 [N] labels in [0,9].
};

/** @brief Validates matching IDX headers and exact payloads before allocation.
 */
inline IdxDataset load_mnist_idx(const std::filesystem::path& images_path,
                                 const std::filesystem::path& labels_path) {
  auto images = read_file(images_path);
  auto labels = read_file(labels_path);
  if (read_be32(images, 0) != 2051 || read_be32(labels, 0) != 2049)
    throw std::runtime_error("invalid MNIST IDX magic");
  const auto count = read_be32(images, 4);
  if (count == 0 || count > 60000 || read_be32(labels, 4) != count ||
      read_be32(images, 8) != 28 || read_be32(images, 12) != 28)
    throw std::runtime_error("invalid MNIST IDX count or dimensions");
  if (images.size() != 16U + static_cast<std::size_t>(count) * 784U ||
      labels.size() != 8U + static_cast<std::size_t>(count))
    throw std::runtime_error("invalid MNIST IDX payload size");
  for (std::size_t index = 8; index < labels.size(); ++index)
    if (labels[index] > 9)
      throw std::runtime_error("MNIST labels must be in [0,9]");
  const auto samples = static_cast<std::int64_t>(count);
  return {
      torch::from_blob(images.data() + 16, {samples, 1, 28, 28}, torch::kUInt8)
          .to(torch::kFloat32)
          .div_(255.0),
      torch::from_blob(labels.data() + 8, {samples}, torch::kUInt8)
          .to(torch::kInt64)};
}

}  // namespace mnist_data
