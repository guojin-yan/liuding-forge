// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_encoder.hpp>

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace jyppx::liuding::data::image {
namespace {

constexpr std::string_view kComponent = "data.image.encoder";

[[noreturn]] void Invalid(std::string_view operation, std::string_view expected,
                          std::string_view actual = {}) {
  throw InvalidArgument("Invalid image encoder options",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::string(operation),
                                     .expected = std::string(expected),
                                     .actual = std::string(actual)});
}

[[noreturn]] void Resource(std::string_view operation,
                           std::string_view expected,
                           std::string_view actual = {}) {
  throw ResourceExhausted("Image encode resource limit exceeded",
                          ErrorDetails{.component = std::string(kComponent),
                                       .operation = std::string(operation),
                                       .expected = std::string(expected),
                                       .actual = std::string(actual)});
}

[[noreturn]] void BackendFailure(std::string_view operation,
                                 std::string_view actual = {}) {
  throw SerializationError(
      "Image encoding failed",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::string(operation),
                   .actual = std::string(actual),
                   .hint = "Check the image dimensions and writer limits"});
}

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Image encode cancelled",
                    ErrorDetails{.component = std::string(kComponent),
                                 .operation = "encode.cancel"});
}

std::uint8_t Channels(const PixelSpec& spec) {
  switch (spec.color) {
    case PixelColorModel::gray:
      return 1;
    case PixelColorModel::gray_alpha:
      return 2;
    case PixelColorModel::rgb:
    case PixelColorModel::bgr:
      return 3;
    case PixelColorModel::rgba:
    case PixelColorModel::bgra:
      return 4;
  }
  Invalid("encode.color", "gray, gray_alpha, RGB or RGBA");
}

std::size_t CheckedProduct(std::uint64_t width, std::uint64_t height,
                           std::uint64_t channels,
                           const ImageEncodeLimits& limits) {
  if (width == 0 || height == 0 || channels == 0)
    Invalid("encode.dimensions", "positive dimensions and channels");
  if (width > limits.max_dimension || height > limits.max_dimension)
    Resource("encode.dimension", std::to_string(limits.max_dimension));
  if (width > std::numeric_limits<std::uint64_t>::max() / height)
    Resource("encode.pixels", "integer range");
  const auto pixels = width * height;
  if (pixels > limits.max_pixels)
    Resource("encode.pixels", std::to_string(limits.max_pixels),
             std::to_string(pixels));
  if (pixels > std::numeric_limits<std::uint64_t>::max() / channels)
    Resource("encode.input", "integer range");
  const auto bytes = pixels * channels;
  if (bytes > limits.max_input_bytes)
    Resource("encode.input", std::to_string(limits.max_input_bytes),
             std::to_string(bytes));
  if (bytes > std::numeric_limits<std::size_t>::max())
    Resource("encode.input", "size_t range");
  return static_cast<std::size_t>(bytes);
}

void CheckLimits(const ImageEncodeLimits& limits) {
  if (limits.max_input_bytes == 0 || limits.max_dimension == 0 ||
      limits.max_pixels == 0 || limits.max_output_bytes == 0 ||
      limits.max_working_bytes == 0)
    Invalid("encode.limits", "positive resource limits");
}

void CheckFormat(const ImageEncodeOptions& options, std::uint8_t channels) {
  if (options.format != ImageEncodeFormat::png &&
      options.format != ImageEncodeFormat::jpeg)
    Invalid("encode.format", "PNG or JPEG");
  if (options.format == ImageEncodeFormat::jpeg &&
      (channels == 2 || channels == 4))
    Invalid("encode.jpeg.alpha", "gray or RGB without alpha");
  if (options.format == ImageEncodeFormat::jpeg &&
      (options.jpeg_quality < 1 || options.jpeg_quality > 100))
    Invalid("encode.jpeg.quality", "an integer from 1 through 100");
}

struct CallbackState {
  std::vector<std::byte> bytes;
  std::size_t limit = 0;
  bool exceeded = false;
};

void Append(void* context, void* data, int size) {
  auto& state = *static_cast<CallbackState*>(context);
  if (size <= 0) return;
  const auto count = static_cast<std::size_t>(size);
  if (count > state.limit - std::min(state.limit, state.bytes.size())) {
    state.exceeded = true;
    return;
  }
  const auto* begin = static_cast<const std::byte*>(data);
  state.bytes.insert(state.bytes.end(), begin, begin + count);
}

Tensor HwcBytes(const Tensor& pixels, const PixelSpec& spec,
                std::uint8_t channels, std::size_t input_bytes,
                std::size_t& working_bytes) {
  if (!pixels.device().is_cpu()) Invalid("encode.device", "a CPU tensor");
  if (pixels.scalar_type() != torch::kUInt8 || spec.dtype != torch::kUInt8 ||
      spec.range != PixelValueRange::unsigned_full_scale)
    Invalid("encode.dtype", "a uint8 unsigned full-scale image");
  if (spec.layout != ImageLayout::chw && spec.layout != ImageLayout::hwc)
    Invalid("encode.layout", "CHW or HWC");
  const auto expected_dim = spec.layout == ImageLayout::chw ? 3 : 3;
  if (pixels.dim() != expected_dim)
    Invalid("encode.tensor", "a three-dimensional image tensor");
  if (spec.layout == ImageLayout::chw &&
      pixels.size(0) != static_cast<std::int64_t>(channels))
    Invalid("encode.channels", "tensor and PixelSpec channel counts agree");
  if (spec.layout == ImageLayout::hwc &&
      pixels.size(2) != static_cast<std::int64_t>(channels))
    Invalid("encode.channels", "tensor and PixelSpec channel counts agree");
  if (input_bytes > std::numeric_limits<std::size_t>::max() / 2U)
    Resource("encode.working", "size_t range");
  working_bytes = pixels.is_contiguous() && spec.layout == ImageLayout::hwc
                      ? 0U
                      : input_bytes;
  if (spec.layout == ImageLayout::chw)
    return pixels.permute({1, 2, 0}).contiguous();
  return pixels.contiguous();
}

}  // namespace

EncodedImage encode_image(const Tensor& pixels, const PixelSpec& pixel_spec,
                          const ImageEncodeOptions& options,
                          std::stop_token stop) {
  CheckStop(stop);
  CheckLimits(options.limits);
  validate_pixels(pixels, pixel_spec);
  if (pixel_spec.alpha != PixelAlphaMode::none &&
      pixel_spec.alpha != PixelAlphaMode::straight)
    Invalid("encode.alpha", "none or straight alpha");
  const auto channels = Channels(pixel_spec);
  CheckFormat(options, channels);
  if (pixels.dim() != 3) Invalid("encode.tensor", "a 3-D image tensor");
  const auto width =
      pixel_spec.layout == ImageLayout::chw ? pixels.size(2) : pixels.size(1);
  const auto height =
      pixel_spec.layout == ImageLayout::chw ? pixels.size(1) : pixels.size(0);
  const auto input_bytes = CheckedProduct(static_cast<std::uint64_t>(width),
                                          static_cast<std::uint64_t>(height),
                                          channels, options.limits);
  if (pixels.numel() != static_cast<std::int64_t>(input_bytes))
    Invalid("encode.tensor", "tensor element count matches PixelSpec");
  std::size_t staging_bytes = 0;
  Tensor hwc;
  try {
    hwc = HwcBytes(pixels, pixel_spec, channels, input_bytes, staging_bytes);
    const auto writer_working =
        input_bytes > (std::numeric_limits<std::size_t>::max() / 2U)
            ? std::numeric_limits<std::size_t>::max()
            : input_bytes * 2U;
    const auto required_working = std::max(staging_bytes, writer_working);
    if (required_working > options.limits.max_working_bytes)
      Resource("encode.working",
               std::to_string(options.limits.max_working_bytes),
               std::to_string(required_working));
    ImageResourceLease decoded_lease;
    ImageResourceLease pixel_lease;
    ImageResourceLease scratch_lease;
    if (options.resource_budget) {
      decoded_lease = options.resource_budget->reserve(
          ImageResourceKind::decoded_bytes, input_bytes);
      pixel_lease = options.resource_budget->reserve(
          ImageResourceKind::pixels,
          static_cast<std::uint64_t>(width) *
              static_cast<std::uint64_t>(height));
      scratch_lease = options.resource_budget->reserve(
          ImageResourceKind::codec_scratch_bytes, required_working);
    }
    if (width > std::numeric_limits<int>::max() ||
        height > std::numeric_limits<int>::max() ||
        static_cast<std::uint64_t>(width) * channels >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      Resource("encode.backend_dimensions", "INT_MAX");

    CallbackState state;
    state.limit = options.limits.max_output_bytes;
    state.bytes.reserve(std::min(options.limits.max_output_bytes,
                                 std::max<std::size_t>(1024U, input_bytes)));
    const auto* data = hwc.data_ptr<std::uint8_t>();
    const auto stride = static_cast<int>(width * channels);
    int ok = 0;
    if (options.format == ImageEncodeFormat::png) {
      ok = stbi_write_png_to_func(&Append, &state, static_cast<int>(width),
                                  static_cast<int>(height), channels, data,
                                  stride);
    } else {
      ok = stbi_write_jpg_to_func(&Append, &state, static_cast<int>(width),
                                  static_cast<int>(height), channels, data,
                                  options.jpeg_quality);
    }
    if (state.exceeded || state.bytes.size() > options.limits.max_output_bytes)
      Resource("encode.output",
               std::to_string(options.limits.max_output_bytes));
    if (!ok) BackendFailure("encode.backend");
    CheckStop(stop);
    ImageResourceLease output_lease;
    if (options.resource_budget) {
      output_lease = options.resource_budget->reserve(
          ImageResourceKind::encoded_bytes,
          static_cast<std::uint64_t>(state.bytes.size()));
    }
    return EncodedImage{std::move(state.bytes),
                        options.format == ImageEncodeFormat::png
                            ? ImageFormat::png
                            : ImageFormat::jpeg,
                        static_cast<std::uint32_t>(width),
                        static_cast<std::uint32_t>(height), channels,
                        std::move(output_lease)};
  } catch (const c10::Error& error) {
    detail::rethrow_backend_error(
        error, detail::BackendErrorContext{
                   .component = std::string(kComponent),
                   .operation = "encode.tensor",
                   .failure_kind = detail::BackendFailureKind::device,
                   .hint = "Check the CPU uint8 tensor and LibTorch runtime"});
  } catch (const std::bad_alloc&) {
    Resource("encode.allocation", "available memory");
  }
}

void write_image_file(const std::filesystem::path& path, const Tensor& pixels,
                      const PixelSpec& pixel_spec,
                      const ImageEncodeOptions& options, std::stop_token stop) {
  const auto encoded = encode_image(pixels, pixel_spec, options, stop);
  CheckStop(stop);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    throw IoError("Could not open encoded image destination",
                  ErrorDetails{.component = std::string(kComponent),
                               .operation = "file.open",
                               .actual = path.string()});
  output.write(reinterpret_cast<const char*>(encoded.bytes.data()),
               static_cast<std::streamsize>(encoded.bytes.size()));
  if (!output)
    throw IoError("Could not write encoded image destination",
                  ErrorDetails{.component = std::string(kComponent),
                               .operation = "file.write",
                               .actual = path.string()});
  output.flush();
  if (!output)
    throw IoError("Could not flush encoded image destination",
                  ErrorDetails{.component = std::string(kComponent),
                               .operation = "file.flush",
                               .actual = path.string()});
}

}  // namespace jyppx::liuding::data::image
