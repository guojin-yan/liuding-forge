// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_decoder.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <stb_image.h>
#include <string>
#include <vector>

namespace jyppx::liuding::data::image {
namespace {

constexpr std::string_view kComponent = "data.image.decoder";

[[noreturn]] void Invalid(std::string_view operation, std::string_view expected,
                          std::string_view actual = {}) {
  throw InvalidArgument("Invalid image decoder options",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::string(operation),
                                     .expected = std::string(expected),
                                     .actual = std::string(actual)});
}

[[noreturn]] void Malformed(std::string_view operation,
                            std::string_view reason = {}) {
  throw SerializationError(
      "Image decode failed",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::string(operation),
                   .actual = std::string(reason),
                   .hint = "Verify the encoded file and select its explicit "
                           "format when needed"});
}

[[noreturn]] void Resource(std::string_view operation,
                           std::string_view expected,
                           std::string_view actual = {}) {
  throw ResourceExhausted("Image decode resource limit exceeded",
                          ErrorDetails{.component = std::string(kComponent),
                                       .operation = std::string(operation),
                                       .expected = std::string(expected),
                                       .actual = std::string(actual)});
}

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Image decode cancelled",
                    ErrorDetails{.component = std::string(kComponent),
                                 .operation = "decode.cancel"});
}

std::size_t CheckedProduct(std::uint64_t width, std::uint64_t height,
                           std::uint64_t channels, std::size_t element_size,
                           const ImageDecodeLimits& limits) {
  if (width == 0 || height == 0 || channels == 0)
    Malformed("decode.dimensions", "zero dimension or channel count");
  if (width > limits.max_dimension || height > limits.max_dimension)
    Resource("decode.dimension", std::to_string(limits.max_dimension));
  if (width > std::numeric_limits<std::uint64_t>::max() / height)
    Resource("decode.pixels", "integer range");
  const auto pixels = width * height;
  if (pixels > limits.max_pixels)
    Resource("decode.pixels", std::to_string(limits.max_pixels),
             std::to_string(pixels));
  if (pixels > std::numeric_limits<std::uint64_t>::max() / channels)
    Resource("decode.output", "integer range");
  const auto elements = pixels * channels;
  if (elements > std::numeric_limits<std::uint64_t>::max() / element_size)
    Resource("decode.output", "integer range");
  const auto bytes = elements * element_size;
  if (bytes > limits.max_output_bytes)
    Resource("decode.output", std::to_string(limits.max_output_bytes),
             std::to_string(bytes));
  if (bytes > std::numeric_limits<std::size_t>::max())
    Resource("decode.output", "size_t range");
  return static_cast<std::size_t>(bytes);
}

int DesiredChannels(ImageChannels channels) {
  switch (channels) {
    case ImageChannels::source:
      return 0;
    case ImageChannels::gray:
      return 1;
    case ImageChannels::gray_alpha:
      return 2;
    case ImageChannels::rgb:
      return 3;
    case ImageChannels::rgba:
      return 4;
  }
  Invalid("decode.channels", "a known channel policy");
}

std::uint8_t PublishedChannels(int source_channels, ImageChannels requested) {
  if (requested == ImageChannels::source) {
    if (source_channels < 1 || source_channels > 4)
      Malformed("decode.channels", "backend returned an invalid channel count");
    return static_cast<std::uint8_t>(source_channels);
  }
  return static_cast<std::uint8_t>(DesiredChannels(requested));
}

bool IsKnown(ImageFormat format) {
  return format != ImageFormat::auto_detect && format != ImageFormat::unknown;
}

void CheckFormat(ImageFormat requested, ImageFormat detected) {
  if (requested == ImageFormat::auto_detect) return;
  if (!IsKnown(requested)) Invalid("decode.format", "a known image format");
  if (IsKnown(detected) && requested != detected)
    Malformed("decode.format", "explicit format does not match image magic");
}

std::string BackendReason() {
  const auto* reason = stbi_failure_reason();
  return reason == nullptr ? std::string("unknown stb_image failure")
                           : std::string(reason);
}

template <typename T>
using StbiPtr = std::unique_ptr<T, decltype(&stbi_image_free)>;

template <typename T>
StbiPtr<T> OwnStbi(T* pointer) {
  return StbiPtr<T>(pointer, &stbi_image_free);
}

Tensor MakeTensor(const void* data, std::int64_t width, std::int64_t height,
                  std::int64_t channels, bool source_is_16bit,
                  const ImageDecodeOptions& options) {
  const auto dtype = source_is_16bit ? torch::kUInt16 : torch::kUInt8;
  const auto source =
      torch::from_blob(const_cast<void*>(data), {height, width, channels},
                       torch::TensorOptions().dtype(dtype).device(torch::kCPU));
  Tensor tensor;
  if (options.dtype == ImageDtype::float32_normalized) {
    tensor = source.to(torch::kFloat32);
    tensor.div_(source_is_16bit ? 65535.0 : 255.0);
  } else {
    tensor = source.clone();
  }
  if (options.layout == ImageLayout::chw)
    tensor = tensor.permute({2, 0, 1}).contiguous();
  if (options.pin_memory) {
    if (!torch::cuda::is_available())
      throw DeviceError(
          "Pinned image memory requires a CUDA-capable LibTorch runtime",
          ErrorDetails{.component = std::string(kComponent),
                       .operation = "decode.pin_memory",
                       .hint = "Use pin_memory=false on CPU-only runtimes"});
    tensor = tensor.pin_memory();
  }
  return tensor;
}

std::vector<std::byte> ReadBytes(const std::filesystem::path& path,
                                 const ImageDecodeOptions& options,
                                 ImageResourceLease& input_lease,
                                 std::stop_token stop) {
  CheckStop(stop);
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error)
    throw IoError("Could not stat image file",
                  ErrorDetails{.component = std::string(kComponent),
                               .operation = "file.size",
                               .actual = error.message()});
  if (size > options.limits.max_input_bytes)
    Resource("file.bytes", std::to_string(options.limits.max_input_bytes),
             std::to_string(size));
  if (size > std::numeric_limits<std::size_t>::max())
    Resource("file.bytes", "size_t range");
  if (options.resource_budget)
    input_lease = options.resource_budget->reserve(
        ImageResourceKind::encoded_bytes, size);
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw IoError("Could not open image file",
                  ErrorDetails{.component = std::string(kComponent),
                               .operation = "file.open",
                               .actual = path.string()});
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::size_t offset = 0;
  constexpr std::size_t kChunk = 1U << 20U;
  while (offset < bytes.size()) {
    CheckStop(stop);
    const auto count = std::min(kChunk, bytes.size() - offset);
    input.read(reinterpret_cast<char*>(bytes.data() + offset),
               static_cast<std::streamsize>(count));
    if (input.gcount() != static_cast<std::streamsize>(count))
      throw IoError("Could not read image file",
                    ErrorDetails{.component = std::string(kComponent),
                                 .operation = "file.read",
                                 .actual = path.string()});
    offset += count;
  }
  return bytes;
}

}  // namespace

ImageFormat detect_image_format(std::span<const std::byte> bytes) noexcept {
  const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
  const auto size = bytes.size();
  const auto starts = [&](std::initializer_list<unsigned char> prefix) {
    if (size < prefix.size()) return false;
    std::size_t index = 0;
    for (const auto value : prefix)
      if (data[index++] != value) return false;
    return true;
  };
  if (starts({0xffU, 0xd8U, 0xffU})) return ImageFormat::jpeg;
  if (starts({0x89U, 0x50U, 0x4eU, 0x47U, 0x0dU, 0x0aU, 0x1aU, 0x0aU}))
    return ImageFormat::png;
  if (starts({'B', 'M'})) return ImageFormat::bmp;
  if (size >= 6U && std::memcmp(data, "GIF87a", 6U) == 0)
    return ImageFormat::gif;
  if (size >= 6U && std::memcmp(data, "GIF89a", 6U) == 0)
    return ImageFormat::gif;
  if (size >= 2U && data[0] == 'P' &&
      (data[1] == '5' || data[1] == '6' || data[1] == '7'))
    return ImageFormat::pnm;
  if (size >= 10U && (std::memcmp(data, "#?RADIANCE", 10U) == 0 ||
                      std::memcmp(data, "#?RGBE", 6U) == 0))
    return ImageFormat::hdr;
  return ImageFormat::unknown;
}

std::string_view image_format_name(ImageFormat format) noexcept {
  switch (format) {
    case ImageFormat::auto_detect:
      return "auto";
    case ImageFormat::jpeg:
      return "jpeg";
    case ImageFormat::png:
      return "png";
    case ImageFormat::bmp:
      return "bmp";
    case ImageFormat::gif:
      return "gif";
    case ImageFormat::tga:
      return "tga";
    case ImageFormat::pnm:
      return "pnm";
    case ImageFormat::hdr:
      return "hdr";
    case ImageFormat::pic:
      return "pic";
    case ImageFormat::unknown:
      return "unknown";
  }
  return "unknown";
}

DecodedImage decode_image(std::span<const std::byte> bytes,
                          const ImageDecodeOptions& options,
                          std::stop_token stop) {
  CheckStop(stop);
  if (options.layout != ImageLayout::chw && options.layout != ImageLayout::hwc)
    Invalid("decode.layout", "CHW or HWC");
  if (options.dtype != ImageDtype::preserve &&
      options.dtype != ImageDtype::float32_normalized)
    Invalid("decode.dtype", "preserve or float32_normalized");
  if (bytes.empty()) Malformed("decode.input", "empty input");
  if (bytes.size() > options.limits.max_input_bytes)
    Resource("decode.input", std::to_string(options.limits.max_input_bytes),
             std::to_string(bytes.size()));
  if (options.limits.max_dimension == 0 || options.limits.max_pixels == 0 ||
      options.limits.max_output_bytes == 0)
    Invalid("decode.limits", "positive resource limits");

  const auto detected = detect_image_format(bytes);
  CheckFormat(options.format, detected);
  if (detected == ImageFormat::png) {
    constexpr std::array<unsigned char, 8> kIhdr{0,   0,   0,   13,
                                                 'I', 'H', 'D', 'R'};
    if (bytes.size() < 33U ||
        std::memcmp(bytes.data() + 8U, kIhdr.data(), kIhdr.size()) != 0)
      Malformed("decode.png.header",
                "expected a standard PNG IHDR; Apple CgBI PNG is unsupported");
  }
  if (detected == ImageFormat::hdr || options.format == ImageFormat::hdr ||
      options.format == ImageFormat::pic)
    Malformed("decode.format", "HDR/PIC require a separate scalar contract");
  const auto* encoded = reinterpret_cast<const stbi_uc*>(bytes.data());
  if (bytes.size() >
      static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    Resource("decode.input", "INT_MAX", std::to_string(bytes.size()));
  int width = 0;
  int height = 0;
  int source_channels = 0;
  if (!stbi_info_from_memory(encoded, static_cast<int>(bytes.size()), &width,
                             &height, &source_channels))
    Malformed("decode.info", BackendReason());
  if (width <= 0 || height <= 0 || source_channels <= 0)
    Malformed("decode.info", "invalid image metadata");
  const bool source_is_16bit =
      stbi_is_16_bit_from_memory(encoded, static_cast<int>(bytes.size())) != 0;
  const auto output_channels =
      PublishedChannels(source_channels, options.channels);
  const auto element_size = options.dtype == ImageDtype::float32_normalized
                                ? sizeof(float)
                            : source_is_16bit ? sizeof(std::uint16_t)
                                              : sizeof(std::uint8_t);
  const auto output_bytes = CheckedProduct(
      static_cast<std::uint64_t>(width), static_cast<std::uint64_t>(height),
      output_channels, element_size, options.limits);
  const auto source_bytes = CheckedProduct(
      static_cast<std::uint64_t>(width), static_cast<std::uint64_t>(height),
      output_channels,
      source_is_16bit ? sizeof(std::uint16_t) : sizeof(std::uint8_t),
      options.limits);

  CheckStop(stop);
  const auto desired = DesiredChannels(options.channels);
  DecodedImage result;
  ImageResourceLease scratch_lease;
  try {
    if (options.resource_budget) {
      const auto pixels = static_cast<std::uint64_t>(width) * height;
      result.resource_leases = std::make_shared<ImageResourceLeaseSet>();
      auto& leases = *result.resource_leases;
      leases.push_back(
          options.resource_budget->reserve(ImageResourceKind::pixels, pixels));
      leases.push_back(options.resource_budget->reserve(
          ImageResourceKind::decoded_bytes, output_bytes));
      if (options.pin_memory)
        leases.push_back(options.resource_budget->reserve(
            ImageResourceKind::pinned_bytes, output_bytes));
      const auto extra_bytes = std::max(source_bytes, output_bytes);
      if (source_bytes >
          std::numeric_limits<std::uint64_t>::max() - extra_bytes)
        Resource("decode.scratch", "uint64 range");
      // stb pixels and one overlapping LibTorch conversion copy are known here.
      scratch_lease = options.resource_budget->reserve(
          ImageResourceKind::codec_scratch_bytes, source_bytes + extra_bytes);
    }
  } catch (const std::bad_alloc&) {
    Resource("decode.allocation", "available memory");
  }
  result.format = detected;
  result.layout = options.layout;
  result.width = static_cast<std::uint32_t>(width);
  result.height = static_cast<std::uint32_t>(height);
  result.channels = output_channels;
  result.source_is_16bit = source_is_16bit;
  result.pixels.layout = options.layout;
  constexpr std::array kColors{PixelColorModel::gray,
                               PixelColorModel::gray_alpha,
                               PixelColorModel::rgb, PixelColorModel::rgba};
  result.pixels.color = kColors[output_channels - 1U];
  result.pixels.alpha = output_channels == 2 || output_channels == 4
                            ? PixelAlphaMode::straight
                            : PixelAlphaMode::none;
  result.pixels.range = options.dtype == ImageDtype::float32_normalized
                            ? PixelValueRange::unit_interval
                            : PixelValueRange::unsigned_full_scale;

  try {
    if (source_is_16bit) {
      auto decoded = OwnStbi(
          stbi_load_16_from_memory(encoded, static_cast<int>(bytes.size()),
                                   &width, &height, &source_channels, desired));
      if (!decoded) Malformed("decode.pixels", BackendReason());
      result.tensor = MakeTensor(decoded.get(), width, height, output_channels,
                                 true, options);
    } else {
      auto decoded = OwnStbi(
          stbi_load_from_memory(encoded, static_cast<int>(bytes.size()), &width,
                                &height, &source_channels, desired));
      if (!decoded) Malformed("decode.pixels", BackendReason());
      result.tensor = MakeTensor(decoded.get(), width, height, output_channels,
                                 false, options);
    }
    result.pixels.dtype = result.tensor.scalar_type();
    validate_pixels(result.tensor, result.pixels);
  } catch (const c10::Error& error) {
    detail::rethrow_backend_error(
        error, detail::BackendErrorContext{
                   .component = std::string(kComponent),
                   .operation = "decode.tensor",
                   .failure_kind = detail::BackendFailureKind::device,
                   .hint = "Check LibTorch dtype and pinned-memory support"});
  } catch (const std::bad_alloc&) {
    Resource("decode.allocation", "available memory");
  }
  CheckStop(stop);
  return result;
}

DecodedImage read_image_file(const std::filesystem::path& path,
                             const ImageDecodeOptions& options,
                             std::stop_token stop) {
  ImageResourceLease input_lease;
  const auto bytes = ReadBytes(path, options, input_lease, stop);
  return decode_image(bytes, options, stop);
}

}  // namespace jyppx::liuding::data::image
