// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "codec_support.hpp"

#include "preparation_support.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <string>
#include <zlib.h>

#if ZLIB_VERNUM != 0x1320
#error "Offline dataset preparation requires reviewed zlib 1.3.2 headers"
#endif

namespace jyppx::liuding::datasets::detail {
namespace {
void require_codec_version() {
  if (std::string_view(zlibVersion()) != "1.3.2")
    throw CompatibilityError(
        "Offline dataset preparation requires exact zlib 1.3.2 runtime");
}

// Allocation callbacks do not throw across the C boundary. Header sizes are
// included in the charged live bytes and retain max_align_t alignment.
struct CodecMemory {
  struct alignas(std::max_align_t) Header {
    std::size_t bytes;
  };
  std::size_t live = 0U;
  static voidpf allocate(voidpf opaque, uInt count, uInt size) noexcept {
    auto& self = *static_cast<CodecMemory*>(opaque);
    if (size != 0U && count > kCodecAllocationBudgetBytes / size)
      return nullptr;
    const auto payload = static_cast<std::size_t>(count) * size;
    if (payload > kCodecAllocationBudgetBytes - sizeof(Header)) return nullptr;
    const auto length = sizeof(Header) + payload;
    if (length > kCodecAllocationBudgetBytes - self.live) return nullptr;
    auto* block = static_cast<Header*>(std::calloc(1U, length));
    if (!block) return nullptr;
    block->bytes = length;
    self.live += length;
    return block + 1;
  }
  static void release(voidpf opaque, voidpf address) noexcept {
    if (!address) return;
    auto& self = *static_cast<CodecMemory*>(opaque);
    auto* block = static_cast<Header*>(address) - 1;
    self.live -= block->bytes;
    std::free(block);
  }
  void configure(z_stream& stream) noexcept {
    stream.zalloc = allocate;
    stream.zfree = release;
    stream.opaque = this;
  }
};
[[noreturn]] void codec_error(int code, std::string_view operation) {
  if (code == Z_MEM_ERROR)
    throw ResourceExhausted("Offline dataset zlib allocation budget exceeded");
  throw CompatibilityError(std::string(operation));
}
void append_be32(std::vector<unsigned char>& output, std::uint32_t value) {
  for (const auto shift : {24U, 16U, 8U, 0U})
    output.push_back(static_cast<unsigned char>(value >> shift));
}
void png_chunk(std::vector<unsigned char>& output, std::string_view kind,
               std::span<const unsigned char> payload) {
  if (output.size() + 12U + payload.size() > kMaxDatasetPngBytes)
    throw ResourceExhausted("Offline dataset PNG exceeds its byte limit");
  append_be32(output, static_cast<std::uint32_t>(payload.size()));
  const auto start = output.size();
  output.insert(output.end(), kind.begin(), kind.end());
  output.insert(output.end(), payload.begin(), payload.end());
  const auto crc =
      crc32(0U, output.data() + start, static_cast<uInt>(4U + payload.size()));
  append_be32(output, static_cast<std::uint32_t>(crc));
}
}  // namespace

struct GzipReader::State {
  State(const std::filesystem::path& path, std::uint64_t compressed_bytes,
        std::string_view compressed_sha256, std::size_t max_read_bytes)
      : expected_bytes(compressed_bytes), max_read(max_read_bytes) {
    const auto lower_hex = [](unsigned char character) {
      return (character >= '0' && character <= '9') ||
             (character >= 'a' && character <= 'f');
    };
    if (expected_bytes == 0U || compressed_sha256.size() != 64U ||
        !std::all_of(compressed_sha256.begin(), compressed_sha256.end(),
                     lower_hex) ||
        max_read == 0U || max_read > kCodecChunkBytes)
      throw InvalidArgument(
          "Invalid private gzip source identity or per-read bound");
    expected_sha256 = compressed_sha256;
    require_codec_version();
    reject_links(path);
    input.open(path, std::ios::binary);
    if (!input)
      throw IoError("Could not open offline dataset compressed source");
    memory.configure(stream);
    const auto code = inflateInit2(&stream, 15 + 16);
    if (code != Z_OK) codec_error(code, "Could not initialize gzip decoder");
    initialized = true;
  }
  ~State() {
    if (initialized) inflateEnd(&stream);
  }

  std::size_t read(std::span<unsigned char> output) {
    if (output.size() > max_read)
      throw ResourceExhausted(
          "Offline dataset decoded read exceeds its byte limit");
    if (output.empty() || ended) return 0U;
    stream.next_out = output.data();
    stream.avail_out = static_cast<uInt>(output.size());
    while (stream.avail_out != 0U && !ended) {
      if (stream.avail_in == 0U && loaded < expected_bytes) {
        const auto size = static_cast<std::size_t>(
            std::min<std::uint64_t>(buffer.size(), expected_bytes - loaded));
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(size));
        if (static_cast<std::size_t>(input.gcount()) != size)
          throw CompatibilityError(
              "Truncated offline dataset compressed source");
        hash.update(std::as_bytes(std::span(buffer.data(), size)));
        loaded += size;
        stream.next_in = buffer.data();
        stream.avail_in = static_cast<uInt>(size);
      }
      const auto before_in = stream.avail_in;
      const auto before_out = stream.avail_out;
      const auto code = inflate(&stream, Z_NO_FLUSH);
      if (code == Z_STREAM_END)
        ended = true;
      else if (code != Z_OK)
        codec_error(code, "Invalid or truncated offline dataset gzip stream");
      if (!ended && before_in == stream.avail_in &&
          before_out == stream.avail_out)
        throw CompatibilityError(
            "Offline dataset gzip stream made no progress");
    }
    return output.size() - stream.avail_out;
  }
  void finish() {
    if (verified) return;
    std::array<unsigned char, 1> tail{};
    if (read(tail) != 0U)
      throw CompatibilityError("Extra decoded offline dataset payload");
    if (!ended || stream.avail_in != 0U || loaded != expected_bytes)
      throw CompatibilityError(
          "Trailing gzip bytes or concatenated members are not permitted");
    char extra = 0;
    if (input.get(extra))
      throw CompatibilityError("Offline dataset source grew during conversion");
    if (!input.eof())
      throw IoError("Could not finish reading compressed source");
    if (sha256_digest_hex(hash.finish()) != expected_sha256)
      throw CompatibilityError(
          "Offline dataset source SHA-256 changed during conversion");
    verified = true;
  }

  std::uint64_t expected_bytes;
  std::string expected_sha256;
  std::size_t max_read;
  std::ifstream input;
  CodecMemory memory;
  z_stream stream{};
  bool initialized = false;
  bool ended = false;
  bool verified = false;
  std::uint64_t loaded = 0U;
  std::array<unsigned char, kCodecChunkBytes> buffer{};
  Sha256 hash;
};

GzipReader::GzipReader(const std::filesystem::path& path,
                       std::uint64_t compressed_bytes,
                       std::string_view compressed_sha256,
                       std::size_t max_decoded_read_bytes)
    : state_(std::make_unique<State>(path, compressed_bytes, compressed_sha256,
                                     max_decoded_read_bytes)) {}
GzipReader::~GzipReader() = default;
std::size_t GzipReader::read(std::span<unsigned char> output) {
  return state_->read(output);
}
void GzipReader::exact(std::span<unsigned char> output) {
  if (read(output) != output.size())
    throw CompatibilityError("Truncated offline dataset decoded record");
}
void GzipReader::finish() { state_->finish(); }

std::vector<unsigned char> encode_rgb_png(
    std::span<const unsigned char> interleaved_rgb, std::uint32_t width,
    std::uint32_t height) {
  if (width == 0U || width > 32U || height == 0U || height > 32U ||
      interleaved_rgb.size() != static_cast<std::size_t>(width) * height * 3U)
    throw InvalidArgument(
        "Private dataset PNG requires exact RGB8 data and dimensions in "
        "[1,32]");
  require_codec_version();
  const auto row_bytes = static_cast<std::size_t>(3U * width);
  const auto stride = row_bytes + 1U;
  std::array<unsigned char, 32U * (1U + 3U * 32U)> rows{};
  for (std::size_t y = 0U; y < height; ++y)
    std::copy_n(interleaved_rgb.data() + y * row_bytes, row_bytes,
                rows.data() + y * stride + 1U);
  CodecMemory memory;
  z_stream stream{};
  memory.configure(stream);
  // Exact original zlib.compress(level=1) settings and one complete input.
  auto code = deflateInit2(&stream, 1, Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY);
  if (code != Z_OK) codec_error(code, "Could not initialize PNG compressor");
  std::array<unsigned char, kMaxDatasetPngBytes - 57U> compressed{};
  stream.next_in = rows.data();
  stream.avail_in = static_cast<uInt>(stride * height);
  stream.next_out = compressed.data();
  stream.avail_out = static_cast<uInt>(compressed.size());
  code = deflate(&stream, Z_FINISH);
  const auto size = static_cast<std::size_t>(stream.total_out);
  const auto remaining = stream.avail_in;
  const auto end_code = deflateEnd(&stream);
  if (code != Z_STREAM_END || remaining != 0U || end_code != Z_OK)
    codec_error(code, "Could not encode bounded offline dataset PNG");
  std::vector<unsigned char> output{137U, 80U, 78U, 71U, 13U, 10U, 26U, 10U};
  output.reserve(kMaxDatasetPngBytes);
  const std::array<unsigned char, 13> ihdr{
      0U, 0U, 0U, static_cast<unsigned char>(width),
      0U, 0U, 0U, static_cast<unsigned char>(height),
      8U, 2U, 0U, 0U,
      0U};
  png_chunk(output, "IHDR", ihdr);
  png_chunk(output, "IDAT", std::span(compressed.data(), size));
  png_chunk(output, "IEND", {});
  return output;
}
}  // namespace jyppx::liuding::datasets::detail
