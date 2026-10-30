// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TOOLS_DATASETS_CODEC_SUPPORT_HPP
#define JYPPX_LIUDING_TOOLS_DATASETS_CODEC_SUPPORT_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace jyppx::liuding::datasets::detail {
inline constexpr std::size_t kCodecChunkBytes = 64U * 1024U;
inline constexpr std::size_t kCodecAllocationBudgetBytes = 1024U * 1024U;
inline constexpr std::size_t kMaxDatasetPngBytes = 8192U;

// Tool-private codec for the two frozen offline datasets. The caller owns
// stable non-link paths, preflights source identities, and enforces cumulative
// decoded/record/output limits. max_decoded_read_bytes limits each read/exact
// span, not total expansion. Both codecs bound zlib live allocation separately.
class GzipReader final {
 public:
  GzipReader(const std::filesystem::path& path, std::uint64_t compressed_bytes,
             std::string_view compressed_sha256,
             std::size_t max_decoded_read_bytes = kCodecChunkBytes);
  ~GzipReader();
  GzipReader(const GzipReader&) = delete;
  GzipReader& operator=(const GzipReader&) = delete;

  // Returns at most output.size() bytes; an empty span is a no-op. A nonempty
  // span returns zero only after the gzip member reached its validated end.
  std::size_t read(std::span<unsigned char> output);
  void exact(std::span<unsigned char> output);
  // Required before publication, even when only a subset was selected.
  // Rejects additional decoded bytes, trailing compressed bytes, concatenated
  // members, truncated data, CRC/ISIZE errors and changed compressed SHA/size.
  // Idempotent after successful verification; destruction alone does not
  // verify.
  void finish();

 private:
  struct State;
  std::unique_ptr<State> state_;
};

// Private fixed-data helper, not a general image encoder/public format promise.
// Accepts tightly packed RGB8 with exact width*height*3 bytes and dimensions
// 1..32; emits filter-0, zlib level1, one IDAT and no ancillary metadata.
// It retains the previous reference PNG byte contract, pending native gates.
std::vector<unsigned char> encode_rgb_png(
    std::span<const unsigned char> interleaved_rgb, std::uint32_t width,
    std::uint32_t height);
}  // namespace jyppx::liuding::datasets::detail
#endif  // JYPPX_LIUDING_TOOLS_DATASETS_CODEC_SUPPORT_HPP
