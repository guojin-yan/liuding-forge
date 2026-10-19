// MNIST data utilities for the examples: a minimal gzip (RFC 1952) decoder
// and an IDX file reader. Self-contained so the example has no dependencies
// beyond LibTorch.
//
// The DEFLATE decoder supports stored, fixed-Huffman and dynamic-Huffman
// blocks, which covers every MNIST mirror's gzip output.

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace mnist_data {

// ---- DEFLATE (RFC 1951)
// ------------------------------------------------------

namespace deflate {

struct BitReader {
  const std::vector<uint8_t>& data;
  size_t byte_pos = 0;
  uint32_t bit_buffer = 0;
  int bit_count = 0;

  explicit BitReader(const std::vector<uint8_t>& bytes) : data(bytes) {}

  int read_bit() {
    if (bit_count == 0) {
      if (byte_pos >= data.size()) {
        throw std::runtime_error("DEFLATE stream ended unexpectedly");
      }
      bit_buffer = data[byte_pos++];
      bit_count = 8;
    }
    const int bit = bit_buffer & 1;
    bit_buffer >>= 1;
    --bit_count;
    return bit;
  }

  int read_bits(int count) {
    int value = 0;
    for (int i = 0; i < count; ++i) {
      value |= read_bit() << i;
    }
    return value;
  }

  void align_to_byte() { bit_count = 0; }
};

// Canonical Huffman decoder built from code lengths (RFC 1951 3.2.2).
struct HuffmanTable {
  std::vector<int> counts;   // number of codes per length
  std::vector<int> symbols;  // symbols ordered by (length, symbol)

  explicit HuffmanTable(const std::vector<int>& lengths) {
    counts.assign(lengths.size() + 1, 0);
    for (const int length : lengths) {
      if (length > 0) {
        ++counts[length];
      }
    }
    std::vector<int> offsets(lengths.size() + 1, 0);
    for (size_t len = 1; len < counts.size(); ++len) {
      offsets[len] = offsets[len - 1] + counts[len - 1];
    }
    symbols.assign(lengths.size(), 0);
    for (size_t symbol = 0; symbol < lengths.size(); ++symbol) {
      if (lengths[symbol] > 0) {
        symbols[offsets[lengths[symbol]]++] = static_cast<int>(symbol);
      }
    }
  }

  template <typename Iter>
  HuffmanTable(Iter first, Iter last)
      : HuffmanTable(std::vector<int>(first, last)) {}

  int decode(BitReader& reader) const {
    int code = 0;
    int first = 0;
    int index = 0;
    for (int length = 1; length < static_cast<int>(counts.size()); ++length) {
      code |= reader.read_bit();
      const int count = counts[length];
      if (code - first < count) {
        return symbols[index + (code - first)];
      }
      index += count;
      first = (first + count) << 1;
      code <<= 1;
    }
    throw std::runtime_error("invalid Huffman code in DEFLATE stream");
  }
};

constexpr int kLengthBase[] = {3,  4,  5,  6,   7,   8,   9,   10,  11, 13,
                               15, 17, 19, 23,  27,  31,  35,  43,  51, 59,
                               67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr int kLengthExtra[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr int kDistBase[] = {1,    2,    3,    4,     5,     7,    9,    13,
                             17,   25,   33,   49,    65,    97,   129,  193,
                             257,  385,  513,  769,   1025,  1537, 2049, 3073,
                             4097, 6145, 8193, 12289, 16385, 24577};
constexpr int kDistExtra[] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                              4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                              9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

inline void inflate_block(BitReader& reader, std::vector<uint8_t>& output,
                          const HuffmanTable& literal_table,
                          const HuffmanTable& distance_table) {
  while (true) {
    const int symbol = literal_table.decode(reader);
    if (symbol < 256) {
      output.push_back(static_cast<uint8_t>(symbol));
    } else if (symbol == 256) {
      return;  // end of block
    } else {
      const int length_index = symbol - 257;
      if (length_index < 0 || length_index > 28) {
        throw std::runtime_error("invalid DEFLATE length symbol");
      }
      const int length = kLengthBase[length_index] +
                         reader.read_bits(kLengthExtra[length_index]);
      const int distance_symbol = distance_table.decode(reader);
      if (distance_symbol < 0 || distance_symbol > 29) {
        throw std::runtime_error("invalid DEFLATE distance symbol");
      }
      const int distance = kDistBase[distance_symbol] +
                           reader.read_bits(kDistExtra[distance_symbol]);
      if (distance > static_cast<int>(output.size())) {
        throw std::runtime_error("DEFLATE distance exceeds output");
      }
      const size_t start = output.size() - static_cast<size_t>(distance);
      for (int i = 0; i < length; ++i) {
        output.push_back(output[start + static_cast<size_t>(i)]);
      }
    }
  }
}

inline void inflate_fixed_block(BitReader& reader,
                                std::vector<uint8_t>& output) {
  std::vector<int> lengths(288, 8);
  for (int i = 144; i < 256; ++i) {
    lengths[i] = 9;
  }
  for (int i = 256; i < 280; ++i) {
    lengths[i] = 7;
  }
  inflate_block(reader, output, HuffmanTable(lengths),
                HuffmanTable(std::vector<int>(30, 5)));
}

inline void inflate_dynamic_block(BitReader& reader,
                                  std::vector<uint8_t>& output) {
  const int hlit = reader.read_bits(5) + 257;
  const int hdist = reader.read_bits(5) + 1;
  const int hclen = reader.read_bits(4) + 4;

  constexpr int kOrder[] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                            11, 4,  12, 3, 13, 2, 14, 1, 15};
  std::vector<int> code_lengths(19, 0);
  for (int i = 0; i < hclen; ++i) {
    code_lengths[kOrder[i]] = reader.read_bits(3);
  }
  const HuffmanTable code_table(code_lengths);

  std::vector<int> lengths(
      static_cast<size_t>(hlit) + static_cast<size_t>(hdist), 0);
  size_t index = 0;
  while (index < lengths.size()) {
    const int symbol = code_table.decode(reader);
    if (symbol < 16) {
      lengths[index++] = static_cast<uint8_t>(symbol);
    } else if (symbol == 16) {
      if (index == 0) {
        throw std::runtime_error("DEFLATE repeat with no previous length");
      }
      const uint8_t previous = lengths[index - 1];
      const int repeat = 3 + reader.read_bits(2);
      for (int i = 0; i < repeat && index < lengths.size(); ++i) {
        lengths[index++] = previous;
      }
    } else if (symbol == 17) {
      const int repeat = 3 + reader.read_bits(3);
      index = std::min(index + static_cast<size_t>(repeat), lengths.size());
    } else if (symbol == 18) {
      const int repeat = 11 + reader.read_bits(7);
      index = std::min(index + static_cast<size_t>(repeat), lengths.size());
    } else {
      throw std::runtime_error("invalid DEFLATE code-length symbol");
    }
  }

  inflate_block(reader, output,
                HuffmanTable(lengths.begin(), lengths.begin() + hlit),
                HuffmanTable(lengths.begin() + hlit, lengths.end()));
}

// Decodes a raw DEFLATE stream.
inline std::vector<uint8_t> inflate(const std::vector<uint8_t>& compressed) {
  std::vector<uint8_t> output;
  BitReader reader(compressed);
  bool final_block = false;
  while (!final_block) {
    final_block = reader.read_bit() != 0;
    const int block_type = reader.read_bits(2);
    if (block_type == 0) {
      // Stored block: byte-align, copy LEN bytes.
      reader.align_to_byte();
      if (reader.byte_pos + 4 > compressed.size()) {
        throw std::runtime_error("truncated stored DEFLATE block");
      }
      const int len =
          compressed[reader.byte_pos] | (compressed[reader.byte_pos + 1] << 8);
      reader.byte_pos += 4;  // LEN + NLEN
      for (int i = 0; i < len; ++i) {
        output.push_back(compressed[reader.byte_pos++]);
      }
    } else if (block_type == 1) {
      inflate_fixed_block(reader, output);
    } else if (block_type == 2) {
      inflate_dynamic_block(reader, output);
    } else {
      throw std::runtime_error("reserved DEFLATE block type");
    }
  }
  return output;
}

}  // namespace deflate

// ---- gzip (RFC 1952)
// ----------------------------------------------------------

inline std::vector<uint8_t> gunzip(const std::vector<uint8_t>& gz) {
  if (gz.size() < 18 || gz[0] != 0x1F || gz[1] != 0x8B || gz[2] != 8) {
    throw std::runtime_error("not a gzip stream");
  }
  const uint8_t flags = gz[3];
  size_t pos = 10;            // fixed header
  if ((flags & 0x04) != 0) {  // FEXTRA
    const int extra = gz[pos] | (gz[pos + 1] << 8);
    pos += 2 + static_cast<size_t>(extra);
  }
  if ((flags & 0x08) != 0) {  // FNAME
    while (pos < gz.size() && gz[pos] != 0) {
      ++pos;
    }
    ++pos;
  }
  if ((flags & 0x10) != 0) {  // FCOMMENT
    while (pos < gz.size() && gz[pos] != 0) {
      ++pos;
    }
    ++pos;
  }
  if ((flags & 0x02) != 0) {  // FHCRC
    pos += 2;
  }
  if (pos + 8 > gz.size()) {
    throw std::runtime_error("truncated gzip stream");
  }
  std::vector<uint8_t> compressed(gz.begin() + static_cast<long>(pos),
                                  gz.end() - 8);
  return deflate::inflate(compressed);
}

// ---- IDX files
// -----------------------------------------------------------------

inline std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open file: " + path);
  }
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

inline int read_be32(const std::vector<uint8_t>& bytes, size_t offset) {
  return (bytes[offset] << 24) | (bytes[offset + 1] << 16) |
         (bytes[offset + 2] << 8) | bytes[offset + 3];
}

struct IdxDataset {
  torch::Tensor images;  // {N, 1, 28, 28} float32 in [0, 1]
  torch::Tensor labels;  // {N} int64
};

inline IdxDataset load_mnist_idx(const std::string& images_path,
                                 const std::string& labels_path) {
  const auto image_bytes = read_file(images_path);
  const auto label_bytes = read_file(labels_path);

  const auto images = gunzip(image_bytes);
  const auto labels = gunzip(label_bytes);

  if (read_be32(images, 0) != 2051) {
    throw std::runtime_error("bad MNIST image file magic");
  }
  if (read_be32(labels, 0) != 2049) {
    throw std::runtime_error("bad MNIST label file magic");
  }
  const int count = read_be32(images, 4);
  const int rows = read_be32(images, 8);
  const int cols = read_be32(images, 12);
  if (rows != 28 || cols != 28) {
    throw std::runtime_error("expected 28x28 MNIST images");
  }
  if (static_cast<int>(labels.size()) < count + 8) {
    throw std::runtime_error("MNIST label file too small");
  }

  auto images_tensor = torch::empty(
      {count, 1, rows, cols}, torch::TensorOptions().dtype(torch::kFloat32));
  auto labels_tensor =
      torch::empty({count}, torch::TensorOptions().dtype(torch::kInt64));
  auto image_accessor = images_tensor.accessor<float, 4>();

  for (int n = 0; n < count; ++n) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const size_t offset = 16 + static_cast<size_t>(n) * 784 +
                              static_cast<size_t>(r) * 28 +
                              static_cast<size_t>(c);
        image_accessor[n][0][r][c] =
            static_cast<float>(images[offset]) / 255.0f;
      }
    }
    labels_tensor[n] =
        static_cast<int64_t>(labels[8 + n]);  // data after magic + count
  }
  return {images_tensor, labels_tensor};
}

}  // namespace mnist_data
