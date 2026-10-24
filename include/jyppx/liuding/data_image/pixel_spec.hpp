// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_PIXEL_SPEC_HPP
#define JYPPX_LIUDING_DATA_IMAGE_PIXEL_SPEC_HPP
/**
 * @file pixel_spec.hpp
 * @brief Explicit scalar, channel, alpha and range semantics for one image.
 *
 * RGB describes channel order, without asserting a transfer function or ICC
 * conversion. Layout describes dimensions, without requiring contiguous
 * storage. A value range describes the encoding, not observed image extrema.
 */

#include <jyppx/liuding/core/types.hpp>

#include <cstdint>

namespace jyppx::liuding::data::image {

/** @brief Dimension order for an individual image, excluding batch axes. */
enum class ImageLayout : std::uint8_t {
  chw,  ///< [channels, height, width].
  hwc,  ///< [height, width, channels].
};

/** @brief Explicit order and meaning of the published channels. */
enum class PixelColorModel : std::uint8_t {
  gray,
  gray_alpha,
  rgb,
  rgba,
  bgr,
  bgra,
};

/** @brief Alpha association; absence and premultiplication are distinct. */
enum class PixelAlphaMode : std::uint8_t { none, straight, premultiplied };

/** @brief Encoding range, independent of the values present in an image. */
enum class PixelValueRange : std::uint8_t {
  unsigned_full_scale,  ///< uint8 [0,255] or uint16 [0,65535].
  unit_interval,        ///< Floating-point intensity encoded in [0,1].
  unbounded,            ///< Floating-point values, e.g. standardized features.
};

/** @brief Pixel semantics carried alongside an owning or borrowed tensor. */
struct PixelSpec {
  Dtype dtype = torch::kUInt8;  ///< Exact tensor scalar representation.
  ImageLayout layout = ImageLayout::chw;         ///< Explicit dimension order.
  PixelColorModel color = PixelColorModel::rgb;  ///< Channel order and count.
  PixelAlphaMode alpha = PixelAlphaMode::none;   ///< Alpha association.
  PixelValueRange range = PixelValueRange::unsigned_full_scale;
  ///< Encoding range; validation does not scan tensor values or synchronize.

  /** @brief Returns the declared number of channels.
   * @return One to four channels.
   * @throws InvalidArgument The color model is unknown.
   */
  std::uint8_t channels() const;
};

/**
 * @brief Checks tensor shape, dtype and internally consistent pixel semantics.
 * @param pixels Defined CPU or CUDA single-image tensor; striding is allowed.
 * @param spec Explicit semantics; no inference from a three-channel shape.
 * @throws InvalidArgument Invalid enum, shape, scalar, alpha or range contract.
 * @note Does not inspect pixel values, convert channels, apply ICC/EXIF, or
 * copy.
 */
void validate_pixels(const Tensor& pixels, const PixelSpec& spec);

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_PIXEL_SPEC_HPP
