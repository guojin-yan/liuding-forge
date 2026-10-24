// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/mask_decoder.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <stb_image.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace jyppx::liuding::data::image {
namespace {

constexpr std::string_view kComponent = "data.image.mask";
using Bytes = std::span<const std::byte>;
using ColorMap = std::unordered_map<std::uint32_t, std::int64_t>;

[[noreturn]] void Invalid(std::string_view operation, std::string_view reason) {
  throw InvalidArgument("Invalid mask decoder options",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::string(operation),
                                     .actual = std::string(reason)});
}

[[noreturn]] void Malformed(std::string_view operation,
                            std::string_view reason) {
  throw SerializationError("Invalid or unsupported label PNG",
                           ErrorDetails{.component = std::string(kComponent),
                                        .operation = std::string(operation),
                                        .actual = std::string(reason)});
}

[[noreturn]] void Resource(std::string_view operation) {
  throw ResourceExhausted("Mask decoder resource limit exceeded",
                          ErrorDetails{.component = std::string(kComponent),
                                       .operation = std::string(operation)});
}

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Mask read cancelled",
                    ErrorDetails{.component = std::string(kComponent),
                                 .operation = "mask.cancel"});
}

unsigned Byte(std::byte value) { return std::to_integer<unsigned>(value); }

std::uint32_t Big32(Bytes bytes) {
  return (Byte(bytes[0]) << 24U) | (Byte(bytes[1]) << 16U) |
         (Byte(bytes[2]) << 8U) | Byte(bytes[3]);
}

std::uint32_t RgbKey(unsigned red, unsigned green, unsigned blue) {
  return red | (green << 8U) | (blue << 16U);
}

constexpr auto CrcTable() {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t i = 0; i < table.size(); ++i) {
    auto value = i;
    for (unsigned bit = 0; bit < 8; ++bit)
      value = (value >> 1U) ^ ((value & 1U) ? 0xedb88320U : 0U);
    table[i] = value;
  }
  return table;
}

std::uint32_t Crc32(Bytes bytes, std::stop_token stop) {
  static constexpr auto kTable = CrcTable();
  std::uint32_t value = 0xffffffffU;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i % 65'536U == 0) CheckStop(stop);
    value = kTable[(value ^ Byte(bytes[i])) & 255U] ^ (value >> 8U);
  }
  return value ^ 0xffffffffU;
}

std::uint32_t Adler32(Bytes bytes, std::stop_token stop) {
  std::uint32_t low = 1;
  std::uint32_t high = 0;
  for (std::size_t i = 0; i < bytes.size();) {
    CheckStop(stop);
    const auto end = i + std::min<std::size_t>(5552, bytes.size() - i);
    for (; i < end; ++i) {
      low += Byte(bytes[i]);
      high += low;
    }
    low %= 65521U;
    high %= 65521U;
  }
  return (high << 16U) | low;
}

ColorMap ValidateOptions(const MaskDecodeOptions& options) {
  if (options.limits.max_input_bytes == 0 ||
      options.limits.max_dimension == 0 || options.limits.max_pixels == 0 ||
      options.limits.max_output_bytes == 0 || options.max_working_bytes == 0 ||
      options.max_chunks == 0 || options.max_color_entries == 0)
    Invalid("mask.limits", "all limits must be positive");
  switch (options.encoding) {
    case MaskEncoding::class_index:
    case MaskEncoding::palette_index:
    case MaskEncoding::rgb_mapping:
    case MaskEncoding::panoptic_rgb:
      break;
    default:
      Invalid("mask.encoding", "a known explicit mask convention is required");
  }
  if (options.colors.size() > options.max_color_entries)
    Resource("mask.colors");
  if (options.encoding != MaskEncoding::rgb_mapping &&
      (!options.colors.empty() || options.unknown_color_is_ignore))
    Invalid("mask.colors", "color policies require rgb_mapping");
  if (options.encoding == MaskEncoding::rgb_mapping && options.colors.empty())
    Invalid("mask.colors", "an explicit nonempty RGB table is required");
  if (options.unknown_color_is_ignore && !options.ignore_id)
    Invalid("mask.ignore", "unknown-color fallback requires ignore_id");

  ColorMap colors;
  colors.reserve(options.colors.size());
  for (const auto& entry : options.colors) {
    if (entry.id < 0 && entry.id != options.ignore_id &&
        entry.id != options.void_id)
      Invalid("mask.colors",
              "negative labels must be declared ignore/void IDs");
    if (!colors
             .emplace(RgbKey(entry.rgb[0], entry.rgb[1], entry.rgb[2]),
                      entry.id)
             .second)
      Invalid("mask.colors", "duplicate RGB mapping key");
  }
  return colors;
}

struct Pass {
  std::uint32_t x = 0, y = 0, dx = 1, dy = 1;
  std::uint32_t width = 0, height = 0;
  std::size_t row_bytes = 0;
};

struct Png {
  std::uint32_t width = 0, height = 0;
  unsigned depth = 0, color = 0;
  bool interlaced = false;
  std::vector<MaskPaletteEntry> palette;
  std::vector<std::byte> compressed;
  std::vector<Pass> passes;
  std::size_t filtered_size = 0;
};

void ReadHeader(Png& png, Bytes data, const MaskDecodeOptions& options) {
  if (data.size() != 13) Malformed("mask.header", "invalid IHDR length");
  png.width = Big32(data.first(4));
  png.height = Big32(data.subspan(4, 4));
  png.depth = Byte(data[8]);
  png.color = Byte(data[9]);
  if (png.width == 0 || png.height == 0 || png.width > 0x7fffffffU ||
      png.height > 0x7fffffffU)
    Malformed("mask.dimensions",
              "PNG dimensions must be positive 31-bit values");
  const auto pixels = static_cast<std::uint64_t>(png.width) * png.height;
  if (png.width > options.limits.max_dimension ||
      png.height > options.limits.max_dimension ||
      pixels > options.limits.max_pixels)
    Resource("mask.dimensions");
  if (pixels > options.limits.max_output_bytes / sizeof(std::int64_t))
    Resource("mask.output");
  if (Byte(data[10]) != 0 || Byte(data[11]) != 0 || Byte(data[12]) > 1)
    Malformed("mask.header",
              "unsupported compression, filter or interlace method");
  png.interlaced = Byte(data[12]) == 1;
  const bool small_depth = png.depth == 1 || png.depth == 2 || png.depth == 4;
  const bool gray =
      png.color == 0 && (small_depth || png.depth == 8 || png.depth == 16);
  const bool palette = png.color == 3 && (small_depth || png.depth == 8);
  const bool rgb = png.color == 2 && png.depth == 8;
  if ((options.encoding == MaskEncoding::class_index && !gray) ||
      (options.encoding == MaskEncoding::palette_index && !palette) ||
      ((options.encoding == MaskEncoding::rgb_mapping ||
        options.encoding == MaskEncoding::panoptic_rgb) &&
       !rgb))
    Malformed("mask.encoding",
              "PNG color type/bit depth does not match mask convention");

  constexpr std::array<Pass, 7> kAdam7{{{0, 0, 8, 8},
                                        {4, 0, 8, 8},
                                        {0, 4, 4, 8},
                                        {2, 0, 4, 4},
                                        {0, 2, 2, 4},
                                        {1, 0, 2, 2},
                                        {0, 1, 1, 2}}};
  const auto add_pass = [&](Pass pass) {
    if (pass.x >= png.width || pass.y >= png.height) return;
    pass.width = (png.width - pass.x - 1) / pass.dx + 1;
    pass.height = (png.height - pass.y - 1) / pass.dy + 1;
    const auto channels = png.color == 2 ? 3U : 1U;
    const auto row =
        (static_cast<std::uint64_t>(pass.width) * channels * png.depth + 7) / 8;
    const auto size = (row + 1) * pass.height;
    if (size > options.max_working_bytes - png.filtered_size ||
        size > static_cast<std::size_t>((std::numeric_limits<int>::max)()) -
                   png.filtered_size)
      Resource("mask.raster");
    pass.row_bytes = static_cast<std::size_t>(row);
    png.filtered_size += static_cast<std::size_t>(size);
    png.passes.push_back(pass);
  };
  if (png.interlaced) {
    for (const auto& pass : kAdam7) add_pass(pass);
  } else {
    add_pass({});
  }
}

Png ParsePng(Bytes bytes, const MaskDecodeOptions& options,
             std::stop_token stop) {
  if (detect_image_format(bytes) != ImageFormat::png)
    Malformed("mask.format",
              "label reader requires PNG; lossy images are not masks");
  Png png;
  std::vector<Bytes> idat_chunks;
  std::size_t compressed_size = 0;
  bool header = false, palette = false, transparency = false;
  bool idat = false, idat_ended = false;
  std::size_t count = 0;
  for (std::size_t offset = 8; offset < bytes.size();) {
    CheckStop(stop);
    if (++count > options.max_chunks) Resource("mask.chunks");
    if (bytes.size() - offset < 12)
      Malformed("mask.chunk", "truncated chunk header");
    const auto length = Big32(bytes.subspan(offset, 4));
    if (length > 0x7fffffffU || length > bytes.size() - offset - 12)
      Malformed("mask.chunk", "truncated or invalid chunk length");
    const auto type_data = bytes.subspan(offset + 4, 4);
    const std::string_view type(reinterpret_cast<const char*>(type_data.data()),
                                4);
    for (const auto ch : type) {
      if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')))
        Malformed("mask.chunk", "invalid chunk type");
    }
    if (type[2] >= 'a')
      Malformed("mask.chunk", "PNG reserved chunk bit is set");
    const auto data = bytes.subspan(offset + 8, length);
    if (Crc32(bytes.subspan(offset + 4, static_cast<std::size_t>(length) + 4),
              stop) != Big32(bytes.subspan(offset + 8 + length, 4)))
      Malformed("mask.crc", "PNG chunk CRC mismatch");
    offset += static_cast<std::size_t>(length) + 12;
    if (!header && type != "IHDR")
      Malformed("mask.header", "IHDR must be first; CgBI is unsupported");
    if (idat && type != "IDAT") idat_ended = true;
    if (type == "IHDR") {
      if (header) Malformed("mask.header", "duplicate IHDR");
      ReadHeader(png, data, options);
      header = true;
    } else if (type == "PLTE") {
      if (palette || transparency || idat || png.color == 0 || length == 0 ||
          length % 3 != 0 || length > 768 ||
          (png.color == 3 && length / 3 > (1U << png.depth)))
        Malformed("mask.palette", "invalid or misplaced PLTE");
      palette = true;
      if (png.color == 3) {
        for (std::size_t i = 0; i < length; i += 3)
          png.palette.push_back({{static_cast<std::uint8_t>(Byte(data[i])),
                                  static_cast<std::uint8_t>(Byte(data[i + 1])),
                                  static_cast<std::uint8_t>(Byte(data[i + 2]))},
                                 255});
      }
    } else if (type == "tRNS") {
      if (transparency || idat || !palette || png.color != 3 || length == 0 ||
          length > png.palette.size())
        Malformed("mask.transparency",
                  "only pre-IDAT palette tRNS is supported");
      transparency = true;
      for (std::size_t i = 0; i < length; ++i)
        png.palette[i].alpha = static_cast<std::uint8_t>(Byte(data[i]));
    } else if (type == "IDAT") {
      if (idat_ended || (png.color == 3 && !palette))
        Malformed("mask.idat", "noncontiguous IDAT or missing palette");
      idat = true;
      const auto available = options.max_working_bytes - png.filtered_size;
      if (length > available - compressed_size ||
          length > static_cast<std::size_t>((std::numeric_limits<int>::max)()) -
                       compressed_size)
        Resource("mask.working");
      compressed_size += length;
      idat_chunks.push_back(data);
    } else if (type == "IEND") {
      if (length != 0 || !idat || compressed_size == 0 ||
          offset != bytes.size())
        Malformed("mask.end", "invalid IEND or trailing data");
      png.compressed.resize(compressed_size);
      std::size_t copied = 0;
      for (const auto chunk : idat_chunks) {
        CheckStop(stop);
        if (!chunk.empty())
          std::memcpy(png.compressed.data() + copied, chunk.data(),
                      chunk.size());
        copied += chunk.size();
      }
      return png;
    } else if (type == "acTL" || type == "fcTL" || type == "fdAT") {
      Malformed("mask.animation",
                "animated PNG requires an explicit frame contract");
    } else if (type[0] >= 'A' && type[0] <= 'Z') {
      Malformed("mask.chunk", "unknown critical PNG chunk");
    }
  }
  Malformed("mask.end", "missing IEND");
}

unsigned Paeth(unsigned left, unsigned above, unsigned corner) {
  const auto p = static_cast<int>(left) + static_cast<int>(above) -
                 static_cast<int>(corner);
  const auto distance = [](int value) { return value < 0 ? -value : value; };
  const auto dl = distance(p - static_cast<int>(left));
  const auto da = distance(p - static_cast<int>(above));
  const auto dc = distance(p - static_cast<int>(corner));
  if (dl <= da && dl <= dc) return left;
  return da <= dc ? above : corner;
}

void Unfilter(std::span<std::byte> row, Bytes previous, unsigned filter,
              unsigned stride) {
  if (filter > 4) Malformed("mask.filter", "unknown PNG row filter");
  for (std::size_t x = 0; x < row.size(); ++x) {
    const auto left = x >= stride ? Byte(row[x - stride]) : 0U;
    const auto above = previous.empty() ? 0U : Byte(previous[x]);
    const auto corner =
        previous.empty() || x < stride ? 0U : Byte(previous[x - stride]);
    unsigned predictor = 0;
    switch (filter) {
      case 1:
        predictor = left;
        break;
      case 2:
        predictor = above;
        break;
      case 3:
        predictor = (left + above) / 2;
        break;
      case 4:
        predictor = Paeth(left, above, corner);
        break;
      default:
        break;
    }
    row[x] = std::byte((Byte(row[x]) + predictor) & 255U);
  }
}

std::int64_t LabelAt(Bytes row, std::size_t x, const Png& png,
                     const MaskDecodeOptions& options, const ColorMap& colors) {
  if (png.color == 2) {
    const auto key =
        RgbKey(Byte(row[x * 3]), Byte(row[x * 3 + 1]), Byte(row[x * 3 + 2]));
    if (options.encoding == MaskEncoding::panoptic_rgb) return key;
    const auto found = colors.find(key);
    if (found != colors.end()) return found->second;
    if (options.unknown_color_is_ignore) return *options.ignore_id;
    Malformed("mask.color_unknown", "RGB pixel has no explicit label mapping");
  }
  unsigned id = 0;
  if (png.depth == 16) {
    id = (Byte(row[x * 2]) << 8U) | Byte(row[x * 2 + 1]);
  } else if (png.depth == 8) {
    id = Byte(row[x]);
  } else {
    const auto bit = x * png.depth;
    id = (Byte(row[bit / 8]) >> (8U - png.depth - bit % 8)) &
         ((1U << png.depth) - 1);
  }
  if (png.color == 3 && id >= png.palette.size())
    Malformed("mask.palette", "pixel index is outside PLTE");
  return id;
}

DecodedMask Decode(Bytes bytes, const MaskDecodeOptions& options,
                   std::stop_token stop) {
  const auto colors = ValidateOptions(options);
  if (bytes.size() > options.limits.max_input_bytes) Resource("mask.input");
  auto png = ParsePng(bytes, options, stop);
  if (png.compressed.size() < 6)
    Malformed("mask.zlib", "truncated zlib stream");
  CheckStop(stop);
  std::vector<std::byte> filtered(png.filtered_size);
  // Use the pinned, privately linked stb DEFLATE backend; label interpretation
  // and PNG filtering are independent from its palette-to-RGB image decoder.
  const auto count = stbi_zlib_decode_buffer(
      reinterpret_cast<char*>(filtered.data()),
      static_cast<int>(filtered.size()),
      reinterpret_cast<const char*>(png.compressed.data()),
      static_cast<int>(png.compressed.size()));
  if (count < 0 || static_cast<std::size_t>(count) != filtered.size())
    Malformed("mask.zlib", "DEFLATE failure or incorrect decoded raster size");
  if (Adler32(filtered, stop) != Big32(Bytes(png.compressed).last(4)))
    Malformed("mask.adler", "zlib Adler-32 mismatch");

  DecodedMask result;
  result.ids = torch::empty(
      {png.height, png.width},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto* output = result.ids.data_ptr<std::int64_t>();
  std::size_t offset = 0;
  const auto stride = std::max(1U, (png.color == 2 ? 3U : 1U) * png.depth / 8);
  for (const auto& pass : png.passes) {
    Bytes previous;
    for (std::uint32_t y = 0; y < pass.height; ++y) {
      CheckStop(stop);
      const auto filter = Byte(filtered[offset++]);
      auto row = std::span(filtered).subspan(offset, pass.row_bytes);
      Unfilter(row, previous, filter, stride);
      const auto out_y = pass.y + y * pass.dy;
      for (std::uint32_t x = 0; x < pass.width; ++x)
        output[static_cast<std::size_t>(out_y) * png.width + pass.x +
               x * pass.dx] = LabelAt(row, x, png, options, colors);
      previous = row;
      offset += pass.row_bytes;
    }
  }
  CheckStop(stop);
  result.encoding = options.encoding;
  result.width = png.width;
  result.height = png.height;
  result.source_bit_depth = static_cast<std::uint8_t>(png.depth);
  result.source_interlaced = png.interlaced;
  result.palette = std::move(png.palette);
  result.ignore_id = options.ignore_id;
  result.void_id = options.void_id;
  return result;
}

}  // namespace

DecodedMask decode_mask(std::span<const std::byte> bytes,
                        const MaskDecodeOptions& options,
                        std::stop_token stop) {
  CheckStop(stop);
  try {
    return Decode(bytes, options, stop);
  } catch (const std::bad_alloc&) {
    Resource("mask.allocation");
  } catch (const c10::Error& error) {
    detail::rethrow_backend_error(
        error, detail::BackendErrorContext{
                   .component = std::string(kComponent),
                   .operation = "mask.tensor",
                   .failure_kind = detail::BackendFailureKind::device});
  }
}

DecodedMask read_mask_file(const std::filesystem::path& path,
                           const MaskDecodeOptions& options,
                           std::stop_token stop) {
  CheckStop(stop);
  try {
    static_cast<void>(ValidateOptions(options));
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error)
      throw IoError("Could not stat mask file",
                    ErrorDetails{.component = std::string(kComponent),
                                 .operation = "mask.file.size",
                                 .actual = error.message()});
    if (size > options.limits.max_input_bytes ||
        size > std::numeric_limits<std::size_t>::max())
      Resource("mask.input");
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw IoError("Could not open mask file",
                    ErrorDetails{.component = std::string(kComponent),
                                 .operation = "mask.file.open"});
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    constexpr std::size_t kChunk = 1U << 20U;
    for (std::size_t offset = 0; offset < bytes.size();) {
      CheckStop(stop);
      const auto count = std::min(kChunk, bytes.size() - offset);
      input.read(reinterpret_cast<char*>(bytes.data() + offset),
                 static_cast<std::streamsize>(count));
      if (input.gcount() != static_cast<std::streamsize>(count))
        throw IoError("Could not read complete mask file",
                      ErrorDetails{.component = std::string(kComponent),
                                   .operation = "mask.file.read"});
      offset += count;
    }
    return decode_mask(bytes, options, stop);
  } catch (const std::bad_alloc&) {
    Resource("mask.allocation");
  }
}

}  // namespace jyppx::liuding::data::image
