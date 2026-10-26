// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_IMAGE_DECODER_HPP
#define JYPPX_LIUDING_DATA_IMAGE_IMAGE_DECODER_HPP
/**
 * @file image_decoder.hpp
 * @brief Bounded built-in image decoding from bytes or a local file.
 *
 * The decoder is deliberately independent from OpenCV. It uses the vendored
 * stb_image adapter for JPEG, PNG, BMP, GIF, TGA, PNM and related still-image
 * formats, then publishes an owning CPU LibTorch tensor. Dataset manifests and
 * annotation readers can use the same API without requiring callers to write a
 * format-specific loader.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/data_image/pixel_spec.hpp>
#include <jyppx/liuding/data_image/resource_budget.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stop_token>
#include <string_view>

namespace jyppx::liuding::data::image {

/** @brief Image container family detected from encoded bytes. */
enum class ImageFormat : std::uint8_t {
  auto_detect,
  jpeg,
  png,
  bmp,
  gif,
  tga,
  pnm,
  hdr,
  pic,
  unknown,
};

/** @brief Requested channel conversion policy. */
enum class ImageChannels : std::uint8_t {
  source,  ///< Preserve the encoded channel count (one through four).
  gray,    ///< Decode or convert to one luminance channel.
  gray_alpha,
  rgb,
  rgba,
};

/** @brief Scalar representation of the published tensor. */
enum class ImageDtype : std::uint8_t {
  preserve,            ///< uint8 for 8-bit sources, uint16 for 16-bit PNG.
  float32_normalized,  ///< float32 values scaled to [0, 1].
};

/** @brief Independent limits applied before allocation or third-party decode.
 */
struct ImageDecodeLimits {
  std::size_t max_input_bytes = 256U << 20U;  ///< Encoded byte limit.
  std::uint32_t max_dimension = 65'536;       ///< Width and height limit.
  std::uint64_t max_pixels = 268'435'456;     ///< Width times height limit.
  std::size_t max_output_bytes = 1U << 30U;   ///< Tensor storage limit.
};

/** @brief Explicit policy for one decode operation. */
struct ImageDecodeOptions {
  ImageDecodeLimits limits = {};  ///< Encoded and decoded resource bounds.
  ImageFormat format = ImageFormat::auto_detect;  ///< Optional input assertion.
  ImageChannels channels = ImageChannels::source;  ///< Output channel policy.
  ImageLayout layout = ImageLayout::chw;    ///< Output tensor dimension order.
  ImageDtype dtype = ImageDtype::preserve;  ///< Bit-depth conversion policy.
  bool pin_memory =
      false;  ///< Publish CPU storage through the pinned allocator.
  std::shared_ptr<ImageResourceBudget> resource_budget;
  ///< Optional aggregate budget for known allocations, not process RSS or
  ///< private codec/backend buffers.
};

/** @brief Owning decoded image and stable source metadata. */
struct DecodedImage {
  std::shared_ptr<ImageResourceLeaseSet> resource_leases;
  ///< Holds decoded storage, pixel and optional pinned reservations.
  Tensor tensor;                              ///< Owning CPU tensor.
  ImageFormat format = ImageFormat::unknown;  ///< Detected source family.
  ImageLayout layout = ImageLayout::chw;      ///< Tensor order.
  std::uint32_t width = 0;                    ///< Encoded width in pixels.
  std::uint32_t height = 0;                   ///< Encoded height in pixels.
  std::uint8_t channels = 0;                  ///< Published channel count.
  bool source_is_16bit = false;  ///< True for a 16-bit source image.
  PixelSpec pixels;  ///< Published scalar/channel/range semantics; no ICC/EXIF
                     ///< conversion.
};

/**
 * @brief Identifies a format from a bounded magic prefix.
 * @param bytes Encoded bytes; only a small prefix is inspected.
 * @return A known format, or `unknown` when the prefix is inconclusive.
 */
ImageFormat detect_image_format(std::span<const std::byte> bytes) noexcept;

/**
 * @brief Returns a stable lowercase format name.
 * @param format Format value to render.
 * @return A static name, including `unknown` for an invalid value.
 */
std::string_view image_format_name(ImageFormat format) noexcept;

/**
 * @brief Decodes one encoded still image into an owning CPU tensor.
 * @param bytes Encoded image bytes; the caller retains ownership.
 * @param options Explicit output, format and resource policy.
 * @param stop Cooperative cancellation checked at decode boundaries.
 * @return A contiguous HWC or CHW tensor with metadata.
 * @note Caller-owned encoded bytes are not charged. Keep the returned object
 * or its resource_leases alive while retaining its tensor separately.
 * @throws InvalidArgument Invalid limits or an impossible channel/layout
 * policy.
 * @throws SerializationError Unsupported, malformed or inconsistent image data.
 * @throws ResourceExhausted Input, pixel or output limits are exceeded.
 * @throws DeviceError Pinned allocation or a LibTorch operation failed.
 * @throws Cancelled Cancellation was requested before publication.
 */
DecodedImage decode_image(std::span<const std::byte> bytes,
                          const ImageDecodeOptions& options = {},
                          std::stop_token stop = {});

/**
 * @brief Reads and decodes one local image file.
 * @param path File to read; no dataset-relative path interpretation is applied.
 * @param options Decode and resource policy.
 * @param stop Cooperative cancellation checked while reading and decoding.
 * @return A decoded owning image.
 * @note With a resource budget, file bytes are charged before allocation and
 * released after decode; the result retains decoded and pixel leases.
 * @throws IoError File access, size or read failure.
 * @throws InvalidArgument, SerializationError, ResourceExhausted, DeviceError,
 *         or Cancelled as documented by decode_image().
 */
DecodedImage read_image_file(const std::filesystem::path& path,
                             const ImageDecodeOptions& options = {},
                             std::stop_token stop = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_DECODER_HPP
