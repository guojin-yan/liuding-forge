// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_IMAGE_ENCODER_HPP
#define JYPPX_LIUDING_DATA_IMAGE_IMAGE_ENCODER_HPP
/**
 * @file image_encoder.hpp
 * @brief Bounded built-in PNG/JPEG encoding from a CPU image tensor.
 */

#include <jyppx/liuding/data_image/image_decoder.hpp>
#include <jyppx/liuding/data_image/resource_budget.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stop_token>
#include <utility>
#include <vector>

namespace jyppx::liuding::data::image {

/** @brief Encoded still-image formats supported by the built-in writer. */
enum class ImageEncodeFormat : std::uint8_t {
  png,   ///< Lossless uint8 PNG with one through four channels.
  jpeg,  ///< Quality-controlled uint8 grayscale or RGB JPEG without alpha.
};

/** @brief Independent limits applied before allocation and during encoding. */
struct ImageEncodeLimits {
  std::size_t max_input_bytes = 1U << 30U;  ///< Logical input pixel-byte bound.
  std::uint32_t max_dimension = 65'536;     ///< Width and height bound.
  std::uint64_t max_pixels = 268'435'456;   ///< Width times height bound.
  std::size_t max_output_bytes = 512U << 20U;  ///< Encoded byte bound.
  std::size_t max_working_bytes = 1U << 30U;
  ///< Conservative tensor staging plus live writer heap payloads; excludes
  ///< encoded-result capacity, stack and allocator bookkeeping.
};

/** @brief Explicit output format, quality and resource policy. */
struct ImageEncodeOptions {
  ImageEncodeLimits limits = {};  ///< Positive independent per-call limits.
  ImageEncodeFormat format = ImageEncodeFormat::png;  ///< Explicit byte format.
  int jpeg_quality = 90;  ///< Inclusive range 1..100; ignored for PNG.
  std::shared_ptr<ImageResourceBudget> resource_budget;
  ///< Shared input/staging/writer heap and encoded vector-capacity accounting.
};

/** @brief Move-only encoded image and dimensions used by the writer. */
struct EncodedImage {
  /** @brief Creates an empty encoded result. */
  EncodedImage() = default;
  /** @brief Adopts bytes and their optional capacity reservation.
   * @param payload Complete encoded bytes; ownership moves into the result.
   * @param encoded_format PNG or JPEG container identity.
   * @param image_width Encoded width in pixels.
   * @param image_height Encoded height in pixels.
   * @param image_channels Encoded channel count.
   * @param lease Reservation for payload capacity, if accounting is enabled.
   */
  EncodedImage(std::vector<std::byte> payload, ImageFormat encoded_format,
               std::uint32_t image_width, std::uint32_t image_height,
               std::uint8_t image_channels, ImageResourceLease lease = {})
      : bytes(std::move(payload)),
        format(encoded_format),
        width(image_width),
        height(image_height),
        channels(image_channels),
        output_lease(std::move(lease)) {}
  /** @brief Frees encoded storage before releasing its reservation. */
  ~EncodedImage() { std::vector<std::byte>().swap(bytes); }
  EncodedImage(const EncodedImage&) = delete;
  EncodedImage& operator=(const EncodedImage&) = delete;
  /** @brief Moves payload and capacity reservation together.
   * @param other Result left with no owned reservation.
   */
  EncodedImage(EncodedImage&& other) noexcept = default;
  /** @brief Replaces storage before replacing its capacity reservation.
   * @param other Result left with no owned reservation.
   * @return This result.
   */
  EncodedImage& operator=(EncodedImage&& other) noexcept = default;

  std::vector<std::byte> bytes;  ///< Owned complete encoded file payload.
  ImageFormat format = ImageFormat::unknown;  ///< PNG or JPEG after encoding.
  std::uint32_t width = 0;    ///< Encoded horizontal pixel count.
  std::uint32_t height = 0;   ///< Encoded vertical pixel count.
  std::uint8_t channels = 0;  ///< Encoded channel count, including PNG alpha.
  /** @brief Holds the encoded vector's capacity charge until consumption. */
  ImageResourceLease output_lease;
};

/**
 * @brief Encodes one CPU uint8 image tensor as bounded PNG or JPEG bytes.
 * @param pixels CPU CHW or HWC uint8 tensor.
 * @param pixel_spec Explicit layout, color and unsigned full-scale semantics.
 * @param options Output format and resource policy.
 * @param stop Cancellation checked at entry, output callbacks and completion.
 * @return Owning encoded bytes; PNG is lossless, JPEG is quality-controlled.
 * @throws InvalidArgument Invalid tensor/spec/format/quality/limits or an
 * unsupported alpha JPEG request.
 * @throws ResourceExhausted Input, pixel, working or output limits exceeded.
 * @throws SerializationError The image backend could not encode the image.
 * @throws DeviceError A LibTorch tensor operation failed.
 * @throws Cancelled Cancellation was requested before publication.
 *
 * The encoder rejects CUDA tensors instead of transferring them implicitly.
 * PNG supports one through four uint8 channels; JPEG supports gray or RGB.
 * BGR/BGRA inputs are reordered to RGB/RGBA in owned staging storage; the
 * caller's pixels and PNG alpha values are preserved. Staging and all writer
 * heap payloads are admitted before allocation, including overlapping growth.
 * Encoded vector capacity is charged separately; growth retains the old
 * reservation until the old allocation is freed. Limits exclude codec stack,
 * allocator bookkeeping and private LibTorch buffers, and are not RSS limits.
 * It does not apply color management, EXIF, normalization or annotation
 * compositing. Use visualization::render_annotations before encoding an
 * annotated RGB image.
 */
EncodedImage encode_image(const Tensor& pixels, const PixelSpec& pixel_spec,
                          const ImageEncodeOptions& options = {},
                          std::stop_token stop = {});

/**
 * @brief Encodes and writes one local image file.
 * @param path Destination path; parent directories are not created.
 * @param pixels CPU uint8 image tensor and explicit pixel semantics.
 * @param pixel_spec Explicit layout, color and unsigned full-scale semantics.
 * @param options Encoding and resource policy.
 * @param stop Cooperative cancellation checked during the operation.
 * @throws IoError Destination open/write/flush failure.
 * @throws InvalidArgument, ResourceExhausted, SerializationError, DeviceError
 *         or Cancelled as documented by encode_image().
 * @note Existing files are truncated after encoding succeeds; a failed write
 * can leave partial output. Use ImageOutputSink for staged publication.
 */
void write_image_file(const std::filesystem::path& path, const Tensor& pixels,
                      const PixelSpec& pixel_spec,
                      const ImageEncodeOptions& options = {},
                      std::stop_token stop = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_ENCODER_HPP
