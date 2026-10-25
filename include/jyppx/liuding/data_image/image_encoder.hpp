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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stop_token>
#include <vector>

namespace jyppx::liuding::data::image {

/** @brief Encoded still-image formats supported by the built-in writer. */
enum class ImageEncodeFormat : std::uint8_t {
  png,
  jpeg,
};

/** @brief Independent limits applied before allocation and during encoding. */
struct ImageEncodeLimits {
  std::size_t max_input_bytes = 1U << 30U;     ///< Tensor storage bound.
  std::uint32_t max_dimension = 65'536;        ///< Width and height bound.
  std::uint64_t max_pixels = 268'435'456;      ///< Width times height bound.
  std::size_t max_output_bytes = 512U << 20U;  ///< Encoded byte bound.
  std::size_t max_working_bytes = 1U << 30U;   ///< Tensor staging/writer bound.
};

/** @brief Explicit output format, quality and resource policy. */
struct ImageEncodeOptions {
  ImageEncodeLimits limits = {};
  ImageEncodeFormat format = ImageEncodeFormat::png;
  int jpeg_quality = 90;  ///< Inclusive range 1..100; ignored for PNG.
};

/** @brief Owning encoded image and dimensions used by the writer. */
struct EncodedImage {
  std::vector<std::byte> bytes;
  ImageFormat format = ImageFormat::unknown;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint8_t channels = 0;
};

/**
 * @brief Encodes one CPU uint8 image tensor as bounded PNG or JPEG bytes.
 * @param pixels CPU CHW or HWC uint8 tensor.
 * @param pixel_spec Explicit layout, color and unsigned full-scale semantics.
 * @param options Output format and resource policy.
 * @param stop Cooperative cancellation checked before and after the backend.
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
 * @param options Encoding and resource policy.
 * @param stop Cooperative cancellation checked during the operation.
 * @throws IoError Destination open/write/flush failure.
 * @throws InvalidArgument, ResourceExhausted, SerializationError, DeviceError
 *         or Cancelled as documented by encode_image().
 */
void write_image_file(const std::filesystem::path& path, const Tensor& pixels,
                      const PixelSpec& pixel_spec,
                      const ImageEncodeOptions& options = {},
                      std::stop_token stop = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_ENCODER_HPP
